"""Verification invalidates active work only after its paid effects settle."""

from __future__ import annotations

import asyncio
from collections import Counter
import threading

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import BudgetValidationError, Sentinel, SentinelAlert
from smythe.checkpoint import FileCheckpointStore, build_state
from smythe.executor import Executor
from smythe.executor_base import NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import Artifact, CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.swarm import Swarm
from smythe.tracer import Tracer


def graph_with_consumer():
    graph = ExecutionGraph([Topology.FORK_JOIN], [
        Node("draft", id="draft"),
        Node("judge", id="judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
        Node("consumer", id="consumer", depends_on=["draft"]),
    ])
    for node in graph.nodes:
        node.metadata["estimated_cost_usd"] = 0.5
    return graph


def make_executor(provider, *, executor_class=AsyncExecutor, budget=None, **kwargs):
    return executor_class(provider=provider, registry=Registry(), tracer=Tracer(),
                          budget=budget, max_concurrency=3, **kwargs)


class CountedProvider(Provider):
    def __init__(self):
        self.calls = Counter()
        self.prompts = []

    def called(self, prompt):
        name = prompt.splitlines()[0]
        self.calls[name] += 1
        self.prompts.append((name, prompt))
        return name, self.calls[name]


class RecordingSupervisor(Supervisor):
    def __init__(self):
        self.reviews = []

    async def review(self, graph, completed_node, **kwargs):
        self.reviews.append((completed_node.id, completed_node.status, completed_node.result))
        return None


@pytest.mark.asyncio
async def test_rejection_cancels_active_consumer_before_new_generation():
    started, cancelled = asyncio.Event(), asyncio.Event()
    graph = graph_with_consumer()

    class ActiveConsumer(CountedProvider):
        async def complete(self, system, prompt, model):
            name, attempt = self.called(prompt)
            if name == "draft":
                if attempt == 2:
                    assert cancelled.is_set(), "new draft started before stale consumer settled"
                text = f"draft-v{attempt}"
            elif name == "judge":
                if attempt == 1:
                    await started.wait()
                text = "FAIL" if attempt == 1 else "PASS"
            else:
                if attempt == 1:
                    assert "draft-v1" in prompt
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                assert "draft-v2" in prompt and "draft-v1" not in prompt
                text = "consumer-v2"
            return CompletionResult(text, cost_usd=0.125)

    provider, budget = ActiveConsumer(), Sentinel(10)
    await asyncio.wait_for(make_executor(provider, budget=budget, artifact_dir=None).run(graph), 3)
    assert provider.calls == {"draft": 2, "judge": 2, "consumer": 2}
    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes)
    assert graph.nodes[2].result == "consumer-v2"
    assert budget.breakdown() == {"draft": 0.25, "judge": 0.25, "consumer": 0.125}
    assert budget._reservations == {}


@pytest.mark.asyncio
async def test_billed_response_returned_during_cancellation_is_settled_then_invalidated():
    started, finalized = asyncio.Event(), asyncio.Event()
    graph = graph_with_consumer()

    class LateResponse(CountedProvider):
        async def complete(self, system, prompt, model):
            name, attempt = self.called(prompt)
            if name == "draft":
                if attempt == 2:
                    assert finalized.is_set()
                    assert graph.nodes[2].result is None
                text = f"draft-v{attempt}"
            elif name == "judge":
                if attempt == 1:
                    await started.wait()
                text = "FAIL" if attempt == 1 else "PASS"
            else:
                if attempt == 1:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # The remote call already completed when cancellation
                        # reached this adapter. Its valid bill must survive.
                        return CompletionResult("consumer-v1", cost_usd=0.125)
                assert "draft-v2" in prompt
                text = "consumer-v2"
            return CompletionResult(text, cost_usd=0.125)

    loop = asyncio.get_running_loop()

    class ObservedFinalization(AsyncExecutor):
        def finalize_node_result(self, node, result):
            super().finalize_node_result(node, result)
            if result.text == "consumer-v1":
                loop.call_soon_threadsafe(finalized.set)

    provider, budget = LateResponse(), Sentinel(10)
    executor = make_executor(provider, executor_class=ObservedFinalization,
                             budget=budget, artifact_dir=None)
    await asyncio.wait_for(executor.run(graph), 3)
    assert provider.calls == {"draft": 2, "judge": 2, "consumer": 2}
    assert graph.nodes[2].result == "consumer-v2"
    assert budget.breakdown() == dict.fromkeys(("draft", "judge", "consumer"), 0.25)
    assert budget.total_cost_usd == 0.75
    assert budget._reservations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("discard_artifacts", [False, True])
async def test_text_only_regeneration_removes_stale_artifact_metadata(tmp_path, discard_artifacts):
    consumer_completed = asyncio.Event()
    graph = graph_with_consumer()

    class ArtifactsThenText(CountedProvider):
        async def complete(self, system, prompt, model):
            name, attempt = self.called(prompt)
            if name == "judge":
                if attempt == 1:
                    await consumer_completed.wait()
                return CompletionResult("FAIL" if attempt == 1 else "PASS", cost_usd=0.125)
            if name == "consumer" and attempt == 2:
                assert "draft-v2" in prompt
                assert "draft-v1" not in prompt
                assert "Artifact files" not in prompt
            artifacts = [Artifact(data=b"first-generation-artifact", mime_type="image/png")] if attempt == 1 else []
            return CompletionResult(f"{name}-v{attempt}", artifacts=artifacts, cost_usd=0.125)

    def updated(node):
        if node.id == "consumer" and node.result == "consumer-v1":
            consumer_completed.set()

    provider = ArtifactsThenText()
    executor = make_executor(provider, artifact_dir=None if discard_artifacts else tmp_path,
                             on_node_update=updated)
    await asyncio.wait_for(executor.run(graph), 3)
    assert provider.calls == {"draft": 2, "judge": 2, "consumer": 2}
    for node in (graph.nodes[0], graph.nodes[2]):
        assert node.status is NodeStatus.COMPLETED
        assert node.result == f"{node.id}-v2"
        assert "artifacts" not in node.metadata
        assert "artifacts_discarded" not in node.metadata


@pytest.mark.parametrize("serial", [False, True])
def test_skipped_failed_judge_does_not_create_a_verification_gate(serial):
    class FailedJudge(CountedProvider):
        async def complete(self, system, prompt, model):
            name, _ = self.called(prompt)
            if name == "judge":
                raise RuntimeError("FAIL: provider unavailable, no verdict produced")
            return CompletionResult("draft", cost_usd=0.125)

    graph = graph_with_consumer()
    graph.nodes[1].failure_policy = FailurePolicy.SKIP
    provider = FailedJudge()
    options = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None)
    if serial:
        Executor(**options).run(graph)
    else:
        asyncio.run(asyncio.wait_for(AsyncExecutor(**options).run(graph), 3))
    assert provider.calls == {"draft": 1, "judge": 1, "consumer": 1}
    assert graph.nodes[1].status is NodeStatus.SKIPPED
    assert graph.nodes[1].metadata.get("regenerations_used", 0) == 0
    assert "verification_receipt" not in graph.nodes[1].metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", ["same", "overlapping", "disjoint"])
async def test_completed_batch_judges_do_not_apply_stale_verdicts(targets):
    barrier = asyncio.Barrier(2)
    nodes = [Node("draft", id="draft")]
    if targets != "same":
        nodes.append(Node("other", id="other", depends_on=["draft"] if targets == "overlapping" else []))
    dependencies = [node.id for node in nodes]
    nodes.extend([
        Node("judge-a", id="judge-a", depends_on=["draft"] if targets == "disjoint" else dependencies,
             verifies="draft", max_regenerations=1),
        Node("judge-b", id="judge-b", depends_on=["other"] if targets == "disjoint" else dependencies,
             verifies="draft" if targets == "same" else "other", max_regenerations=1),
    ])
    graph = ExecutionGraph([Topology.FORK_JOIN], nodes)

    class BatchProvider(CountedProvider):
        async def complete(self, system, prompt, model):
            name, attempt = self.called(prompt)
            text = ("FAIL" if attempt == 1 else "PASS") if name.startswith("judge") else f"{name}-v{attempt}"
            return CompletionResult(text, cost_usd=0.125)

    provider = BatchProvider()

    class SameBatchExecutor(AsyncExecutor):
        async def _execute_node(self, node, graph):
            first = provider.calls[node.id] == 0
            await super()._execute_node(node, graph)
            if first and node.id.startswith("judge"):
                await barrier.wait()

    budget = Sentinel(10)
    supervisor = RecordingSupervisor()
    executor = make_executor(provider, executor_class=SameBatchExecutor, budget=budget, artifact_dir=None,
                             supervisor=supervisor, max_revisions=1)
    await asyncio.wait_for(executor.run(graph), 3)
    assert provider.calls == dict.fromkeys((node.id for node in nodes), 2)
    assert all(node.status is NodeStatus.COMPLETED for node in nodes)
    assert nodes[0].result == "draft-v2"
    if targets != "same":
        assert nodes[1].result == "other-v2"
    assert nodes[-2].metadata["regenerations_used"] == 1
    assert nodes[-1].metadata.get("regenerations_used", 0) == (1 if targets == "disjoint" else 0)
    assert budget.total_cost_usd == len(nodes) * 0.25
    assert all(status is NodeStatus.COMPLETED and result is not None
               for _, status, result in supervisor.reviews)
    assert {name for name, _, result in supervisor.reviews if result == "PASS"} == {"judge-a", "judge-b"}


@pytest.mark.parametrize("serial", [False, True])
def test_rejecting_judge_reaches_supervisor_only_after_fresh_completion(serial):
    class RejectedDraft(CountedProvider):
        async def complete(self, system, prompt, model):
            name, attempt = self.called(prompt)
            text = ("FAIL" if attempt == 1 else "PASS") if name == "judge" else f"draft-v{attempt}"
            return CompletionResult(text)

    graph = graph_with_consumer()
    graph.nodes.pop()  # Only the generated draft and its verifier.
    provider, supervisor = RejectedDraft(), RecordingSupervisor()
    options = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                   supervisor=supervisor, max_revisions=1)
    if serial:
        Executor(**options).run(graph)
    else:
        asyncio.run(asyncio.wait_for(AsyncExecutor(**options).run(graph), 3))
    assert supervisor.reviews == [
        ("draft", NodeStatus.COMPLETED, "draft-v1"),
        ("draft", NodeStatus.COMPLETED, "draft-v2"),
        ("judge", NodeStatus.COMPLETED, "PASS"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal", ["accounting", "finalization", "budget"])
async def test_fatal_error_during_regeneration_settlement_stops_new_generation(fatal, tmp_path):
    started = asyncio.Event()
    error = BudgetValidationError("invalid late provider accounting")
    finalization_error = OSError("late artifact write failed")
    graph = graph_with_consumer()

    class FatalLateResponse(CountedProvider):
        async def complete(self, system, prompt, model):
            name, _ = self.called(prompt)
            if name == "consumer":
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if fatal == "accounting":
                        raise error
                    return CompletionResult("late-response", cost_usd=2 if fatal == "budget" else 0.125)
            if name == "judge":
                await started.wait()
            return CompletionResult("FAIL" if name == "judge" else "draft-v1", cost_usd=0.125)

    class FailedFinalization(AsyncExecutor):
        def finalize_node_result(self, node, result):
            if node.id == "consumer" and fatal == "finalization":
                raise finalization_error
            super().finalize_node_result(node, result)

    provider, budget = FatalLateResponse(), Sentinel(1.5)
    executor = make_executor(provider, executor_class=FailedFinalization, budget=budget, artifact_dir=None)
    expected = {"accounting": BudgetValidationError, "finalization": NodeFinalizationError,
                "budget": SentinelAlert}[fatal]
    with pytest.raises(expected) as caught:
        await asyncio.wait_for(executor.run(graph), 3)
    assert provider.calls == {"draft": 1, "judge": 1, "consumer": 1}
    assert graph.nodes[2].status is NodeStatus.FAILED
    assert graph.nodes[1].metadata.get("regeneration_intent")
    assert budget.breakdown()["draft"] == budget.breakdown()["judge"] == 0.125
    if fatal == "accounting":
        assert caught.value is error
        assert graph.nodes[2].metadata["accounting_invalid"] is True
        assert budget._reservations == {"consumer": 0.5}
        assert budget.total_cost_usd == 0.75
    else:
        if fatal == "finalization":
            assert caught.value.cause is finalization_error
        assert budget.breakdown()["consumer"] == (2 if fatal == "budget" else 0.125)
        assert budget._reservations == {}

    # Recovery sees the same saved intent and charges, including an unresolved
    # accounting marker whose live reservation is absent from legacy cost maps.
    store = FileCheckpointStore(tmp_path)
    store.save("failed", build_state(
        execution_id="failed", status="failed", model="test", graph=graph,
        registry=Registry(), task=None, max_budget_usd=budget.max_budget_usd,
        node_costs=budget.breakdown(),
    ))

    class RecoveredProvider(CountedProvider):
        async def complete(self, system, prompt, model):
            name, _ = self.called(prompt)
            return CompletionResult("PASS" if name == "judge" else f"fresh-{name}", cost_usd=0.125)

    recovered = RecoveredProvider()
    swarm = Swarm(provider=recovered, model="test", checkpoint_store=store,
                  max_concurrency=1, artifact_dir=None)
    if fatal == "accounting":
        with pytest.raises(BudgetValidationError, match="Cannot resume unresolved"):
            await swarm.aresume("failed")
        assert recovered.calls == {}
    elif fatal == "budget":
        with pytest.raises(SentinelAlert):
            await swarm.aresume("failed")
        assert recovered.calls == {}
        assert store.load("failed")["budget"]["node_costs"] == budget.breakdown()
    else:
        # An explicit resume after repairing local persistence can regenerate;
        # it retains the prior billed response instead of refunding that cost.
        result = await swarm.aresume("failed")
        assert recovered.calls == {"draft": 1, "judge": 1, "consumer": 1}
        assert result.graph.nodes[2].result == "fresh-consumer"
        assert "artifacts" not in result.graph.nodes[2].metadata
        assert result.total_cost_usd == 0.75


@pytest.mark.asyncio
async def test_blocked_artifact_finalizer_survives_repeated_consumer_cancellation(tmp_path):
    finalizing, first_cancel, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    graph = graph_with_consumer()
    consumer_task = None

    class ObservedTask(asyncio.Task):
        def cancel(self, msg=None):
            accepted = super().cancel(msg)
            if self.get_name() == "smythe-node-consumer":
                first_cancel.set()
            return accepted

    previous_factory = loop.get_task_factory()
    loop.set_task_factory(lambda loop, coro, **kwargs: ObservedTask(coro, loop=loop, **kwargs))

    class ArtifactProvider(CountedProvider):
        async def complete(self, system, prompt, model):
            nonlocal consumer_task
            name, attempt = self.called(prompt)
            if name == "draft" and attempt == 2:
                assert finished.is_set(), "new generation escaped an unfinished artifact thread"
            if name == "judge":
                if attempt == 1:
                    await finalizing.wait()
                text = "FAIL" if attempt == 1 else "PASS"
            else:
                text = f"{name}-v{attempt}"
            if name == "consumer" and attempt == 1:
                consumer_task = asyncio.current_task()
                return CompletionResult(text, artifacts=[Artifact(data=b"old", mime_type="image/png")],
                                        cost_usd=0.125)
            return CompletionResult(text, cost_usd=0.125)

    class BlockedFinalization(AsyncExecutor):
        def finalize_node_result(self, node, result):
            if result.text == "consumer-v1":
                loop.call_soon_threadsafe(finalizing.set)
                if not release.wait(5):
                    raise TimeoutError("test did not release artifact writer")
            super().finalize_node_result(node, result)
            if result.text == "consumer-v1":
                loop.call_soon_threadsafe(finished.set)

    provider, budget = ArtifactProvider(), Sentinel(10)
    executor = make_executor(provider, executor_class=BlockedFinalization,
                             budget=budget, artifact_dir=tmp_path)
    run_task = asyncio.create_task(executor.run(graph))
    try:
        await asyncio.wait_for(first_cancel.wait(), 3)
        assert finalizing.is_set() and not finished.is_set()
        assert consumer_task is not None and not consumer_task.done()
        consumer_task.cancel("second cancellation while awaiting billed artifact")
        release.set()
        await asyncio.wait_for(asyncio.shield(run_task), 3)
        assert finished.is_set()
        assert graph.nodes[2].status is NodeStatus.COMPLETED
        assert graph.nodes[2].result == "consumer-v2"
        assert "artifacts" not in graph.nodes[2].metadata
        assert budget.breakdown() == dict.fromkeys(("draft", "judge", "consumer"), 0.25)
        assert provider.calls == {"draft": 2, "judge": 2, "consumer": 2}
    finally:
        release.set()
        loop.set_task_factory(previous_factory)
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
