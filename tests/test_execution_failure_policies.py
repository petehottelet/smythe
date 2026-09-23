"""Serial and concurrency-one execution share terminal failure semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.checkpoint import FileCheckpointStore
from smythe.executor import Executor
from smythe.executor_base import SKIPPED_DEPENDENCY_RESULT, TERMINAL_DELIVERABLE_NOTE
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, OutputTruncatedError, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.tools import ToolCall, ToolResult, ToolSpec
from smythe.tracer import Tracer


@pytest.fixture(params=["serial", "parallel-c1"])
def mode(request):
    return request.param


def make_graph(policy, *, timeout_s=None):
    nodes = [
        Node("prior", id="prior"),
        Node("problem", id="problem", failure_policy=policy, max_retries=2,
             timeout_s=timeout_s),
        Node("queued", id="queued"),
        Node("skip-child", id="skip-child", depends_on=["problem"],
             failure_policy=FailurePolicy.SKIP),
        Node("grandchild", id="grandchild", depends_on=["skip-child"]),
    ]
    for node in nodes:
        node.metadata["estimated_cost_usd"] = 0.25
    return ExecutionGraph([Topology.FORK_JOIN], nodes)


class ScriptedProvider(Provider):
    def __init__(self, failures=()):
        self.failures = list(failures)
        self.calls = []

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        self.calls.append(label)
        if label == "problem" and self.failures:
            raise self.failures.pop(0)
        return CompletionResult(f"done: {label}", cost_usd=0.125)


def run(mode, provider, graph, budget, **kwargs):
    options = dict(provider=provider, registry=Registry(), tracer=Tracer(),
                   budget=budget, artifact_dir=None, **kwargs)
    if mode == "serial":
        return Executor(**options).run(graph)
    return asyncio.run(AsyncExecutor(**options, max_concurrency=1).run(graph))


def assert_halted(graph):
    prior, problem, *queued = graph.nodes
    assert prior.status is NodeStatus.COMPLETED
    assert prior.result == "done: prior"
    assert problem.status is NodeStatus.FAILED
    assert all(node.status is NodeStatus.PENDING and node.result is None for node in queued)


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3)])
def test_terminal_failure_stops_queued_work_and_preserves_original_exception(mode, policy, attempts):
    errors = [RuntimeError(f"attempt {i}") for i in range(attempts)]
    provider = ScriptedProvider(errors)
    graph, budget = make_graph(policy), Sentinel(10)

    with pytest.raises(RuntimeError) as caught:
        run(mode, provider, graph, budget)

    assert caught.value is errors[-1]
    assert provider.calls == ["prior"] + ["problem"] * attempts
    assert_halted(graph)
    assert graph.nodes[1].result == str(errors[-1])
    assert budget.breakdown() == {"prior": 0.125}
    assert budget.total_cost_usd == 0.125
    assert budget._reservations == {}


@pytest.mark.parametrize("policy,failures", [(FailurePolicy.SKIP, 1), (FailurePolicy.RETRY, 2)])
def test_consumed_failure_policy_allows_later_work(mode, policy, failures):
    provider = ScriptedProvider([RuntimeError("transient")] * failures)
    graph, budget = make_graph(policy), Sentinel(10)

    assert run(mode, provider, graph, budget) is graph

    attempts = 1 if policy is FailurePolicy.SKIP else failures + 1
    assert provider.calls == ["prior"] + ["problem"] * attempts + [
        "queued", "skip-child", "grandchild",
    ]
    expected_status = NodeStatus.SKIPPED if policy is FailurePolicy.SKIP else NodeStatus.COMPLETED
    assert graph.nodes[1].status is expected_status
    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes if node.id != "problem")
    expected_ids = {node.id for node in graph.nodes if node.status is NodeStatus.COMPLETED}
    assert budget.breakdown() == dict.fromkeys(expected_ids, 0.125)
    assert budget.total_cost_usd == len(expected_ids) * 0.125
    assert budget._reservations == {}


class HangingProvider(ScriptedProvider):
    def __init__(self):
        super().__init__()
        self.cancelled = 0

    async def complete(self, system, prompt, model):
        if prompt.splitlines()[0] != "problem":
            return await super().complete(system, prompt, model)
        self.calls.append("problem")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3),
                                           (FailurePolicy.SKIP, 1)])
def test_timeout_obeys_failure_policy_and_settles_before_return(mode, policy, attempts):
    provider = HangingProvider()
    graph, budget = make_graph(policy, timeout_s=0.01), Sentinel(10)

    if policy is FailurePolicy.SKIP:
        run(mode, provider, graph, budget)
        assert graph.nodes[1].status is NodeStatus.SKIPPED
        assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes[2:])
        expected_tail = ["queued", "skip-child", "grandchild"]
    else:
        with pytest.raises(TimeoutError, match="'problem' timed out after 0.01s"):
            run(mode, provider, graph, budget)
        assert_halted(graph)
        expected_tail = []

    assert provider.cancelled == attempts
    assert provider.calls == ["prior"] + ["problem"] * attempts + expected_tail
    assert budget.breakdown() == dict.fromkeys(["prior", *expected_tail], 0.125)
    assert budget.total_cost_usd == (1 + len(expected_tail)) * 0.125
    assert budget._reservations == {}


class EmptyTools:
    @asynccontextmanager
    async def open(self, agent):
        yield SimpleNamespace(tools=[])


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3),
                                           (FailurePolicy.SKIP, 1)])
def test_failure_retains_cost_of_completed_provider_turns(mode, policy, attempts):
    error = RuntimeError("later conversation turn failed")

    class PaidThenFails(ScriptedProvider):
        async def chat(self, system, messages, model, tools=None):
            if messages[0].content.splitlines()[0] != "problem":
                return await super().chat(system, messages, model, tools)
            self.calls.append("problem")
            if len(messages) == 1:
                return CompletionResult("continue", stop_reason="pause_turn", cost_usd=0.0625)
            raise error

    provider = PaidThenFails()
    graph, budget = make_graph(policy), Sentinel(10)
    if policy is FailurePolicy.SKIP:
        run(mode, provider, graph, budget, tool_runtime=EmptyTools())
        assert graph.nodes[1].status is NodeStatus.SKIPPED
        tail = ["queued", "skip-child", "grandchild"]
    else:
        with pytest.raises(RuntimeError) as caught:
            run(mode, provider, graph, budget, tool_runtime=EmptyTools())
        assert caught.value is error
        assert_halted(graph)
        tail = []

    assert provider.calls == ["prior"] + ["problem"] * (2 * attempts) + tail
    costs = {"prior": 0.125, "problem": attempts * 0.0625, **dict.fromkeys(tail, 0.125)}
    assert budget.breakdown() == costs
    assert budget.total_cost_usd == sum(costs.values())
    assert graph.nodes[1].metadata["cost_usd"] == costs["problem"]
    assert budget._reservations == {}


class TruncatingProvider(ScriptedProvider):
    """The "problem" step hits max_tokens on its first `truncations` calls."""

    def __init__(self, truncations):
        super().__init__()
        self.truncations = truncations

    async def complete(self, system, prompt, model):
        if prompt.splitlines()[0] == "problem" and self.truncations:
            self.truncations -= 1
            self.calls.append("problem")
            return CompletionResult("half an answ", cost_usd=0.0625, stop_reason="max_tokens")
        return await super().complete(system, prompt, model)


@pytest.mark.parametrize("policy,truncations,outcome", [
    (FailurePolicy.HALT, 1, NodeStatus.FAILED),
    (FailurePolicy.RETRY, 1, NodeStatus.COMPLETED),
    (FailurePolicy.RETRY, 3, NodeStatus.FAILED),
    (FailurePolicy.SKIP, 1, NodeStatus.SKIPPED),
])
def test_truncated_output_is_a_billed_failure_under_the_node_policy(
    mode, policy, truncations, outcome,
):
    provider = TruncatingProvider(truncations)
    graph, budget = make_graph(policy), Sentinel(10)

    if outcome is NodeStatus.FAILED:
        with pytest.raises(OutputTruncatedError) as caught:
            run(mode, provider, graph, budget)
        assert caught.value.stop_reason == "max_tokens"
        assert_halted(graph)
    else:
        run(mode, provider, graph, budget)

    problem = graph.nodes[1]
    assert problem.status is outcome
    attempts = 3 if policy is FailurePolicy.RETRY else 1
    truncated = min(truncations, attempts)
    assert provider.calls.count("problem") == min(truncations + 1, attempts)
    if outcome is NodeStatus.COMPLETED:
        assert problem.result == "done: problem"
    else:
        # The partial text is never presented as the node's output.
        assert "half an answ" not in problem.result
        assert "truncated" in problem.result
    # Every truncated call was billed, and each charge survives.
    expected = truncated * 0.0625 + (0.125 if outcome is NodeStatus.COMPLETED else 0)
    assert budget.breakdown()["problem"] == expected
    assert problem.metadata["cost_usd"] == expected
    assert budget._reservations == {}


@pytest.mark.parametrize("hard_ceiling", [False, True])
def test_retry_after_truncation_passes_budget_admission_again(mode, hard_ceiling):
    """The truncated call consumed the node's reservation; the retry needs one."""

    class AlwaysTruncates(ScriptedProvider):
        async def complete(self, system, prompt, model):
            self.calls.append(prompt.splitlines()[0])
            return CompletionResult("half an answ", cost_usd=0.05, cost_usd_is_estimate=hard_ceiling,
                                    stop_reason="max_tokens")

        def budget_estimate_usd(self, model):
            return 0.05 if hard_ceiling else None

        def requires_explicit_budget_estimate(self, model):
            return hard_ceiling

    provider = AlwaysTruncates()
    graph = ExecutionGraph([Topology.SERIAL], [
        Node("problem", id="problem", failure_policy=FailurePolicy.RETRY, max_retries=2),
    ])
    budget = Sentinel(0.08 if hard_ceiling else 0.05)

    with pytest.raises(SentinelAlert):
        run(mode, provider, graph, budget)

    # One billed call, and no second dispatch past the budget.
    assert provider.calls == ["problem"]
    assert budget.breakdown() == {"problem": 0.05}
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert budget._reservations == {}


def test_retry_after_truncation_counts_concurrent_reservations():
    """A bare budget check admitted the retry while a sibling still held its
    reservation, so the run overspent its cap before reconciliation noticed."""

    class Scripted(Provider):
        def __init__(self):
            self.calls = []
            self.retry_sent = asyncio.Event()

        async def complete(self, system, prompt, model):
            label = prompt.splitlines()[0]
            self.calls.append(label)
            if label == "A":
                if self.calls.count("A") == 1:
                    return CompletionResult("half", cost_usd=0.3, stop_reason="max_tokens")
                self.retry_sent.set()
                return CompletionResult("A done", cost_usd=0.3)
            try:  # B stays in flight, holding its reservation.
                await asyncio.wait_for(self.retry_sent.wait(), 0.5)
            except TimeoutError:
                pass
            return CompletionResult("B done", cost_usd=0.3)

    provider = Scripted()
    graph = ExecutionGraph([Topology.FORK_JOIN], [
        Node("A", id="A", failure_policy=FailurePolicy.RETRY, max_retries=2),
        Node("B", id="B"),
    ])
    budget = Sentinel(max_budget_usd=0.8, cost_per_token=0.001)
    executor = AsyncExecutor(provider, Registry(), Tracer(), budget=budget,
                             estimated_tokens_per_node=300, max_concurrency=2, artifact_dir=None)

    with pytest.raises(SentinelAlert):
        asyncio.run(executor.run(graph))

    assert provider.calls.count("A") == 1
    assert budget.total_cost_usd <= budget.max_budget_usd


def test_retry_after_truncation_reserves_the_nodes_explicit_estimate():
    class TruncatesOnce(Provider):
        calls = 0

        async def complete(self, system, prompt, model):
            TruncatesOnce.calls += 1
            stop_reason = "max_tokens" if TruncatesOnce.calls == 1 else "end_turn"
            return CompletionResult("text", cost_usd=0.6, stop_reason=stop_reason)

    node = Node("A", id="A", failure_policy=FailurePolicy.RETRY, max_retries=2,
                metadata={"estimated_cost_usd": 0.5})
    budget = Sentinel(max_budget_usd=1.0)

    with pytest.raises(SentinelAlert):
        Executor(TruncatesOnce(), Registry(), Tracer(), budget=budget, artifact_dir=None).run(
            ExecutionGraph([Topology.SERIAL], [node]))

    # Reserving the 0.5 estimate after a 0.6 charge exceeds the 1.0 cap.
    assert TruncatesOnce.calls == 1
    assert budget.total_cost_usd == pytest.approx(0.6)


def test_truncated_tool_turn_runs_no_tools_and_keeps_its_charge(mode):
    ran = []

    class Tools:
        @asynccontextmanager
        async def open(self, agent):
            async def call(tool_call):
                ran.append(tool_call.arguments)
                return ToolResult(tool_call_id=tool_call.id, content="ran")
            yield SimpleNamespace(tools=[ToolSpec("x.write", "Write", {"type": "object"})],
                                  call=call)

    class TruncatedToolCall(ScriptedProvider):
        async def chat(self, system, messages, model, tools=None):
            self.calls.append(messages[0].content.splitlines()[0])
            return CompletionResult(
                "", tool_calls=[ToolCall("t1", "x.write", {"path": "/tm"})],
                stop_reason="max_tokens", cost_usd=0.0625,
            )

    provider = TruncatedToolCall()
    graph = ExecutionGraph([Topology.SERIAL], [Node("problem", id="problem")])
    budget = Sentinel(10)
    with pytest.raises(OutputTruncatedError):
        run(mode, provider, graph, budget, tool_runtime=Tools())

    assert ran == []
    assert provider.calls == ["problem"]
    assert budget.breakdown() == {"problem": 0.0625}


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3)])
def test_halted_checkpoint_resumes_failed_and_pending_work_only(mode, policy, attempts, tmp_path):
    errors = [RuntimeError(f"attempt {i}") for i in range(attempts)]
    provider = ScriptedProvider(errors)
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, model="test-model", parallel=mode == "parallel-c1",
                  max_concurrency=1, max_budget_usd=10, checkpoint_store=store,
                  checkpoint_every_n_nodes=100, artifact_dir=None)
    graph = make_graph(policy)
    with pytest.raises(RuntimeError) as caught:
        swarm.execute(graph)
    assert caught.value is errors[-1]
    assert_halted(graph)
    assert provider.calls == ["prior"] + ["problem"] * attempts
    [execution_id] = store.list_ids()
    failed = store.load(execution_id)
    assert failed["status"] == "failed"
    assert [node["status"] for node in failed["graph"]["nodes"]] == [
        "completed", "failed", "pending", "pending", "pending",
    ]
    assert failed["budget"]["node_costs"] == {"prior": 0.125}

    # A fresh process-equivalent Swarm must not repurchase completed work or
    # treat the never-dispatched SKIP child as already resolved.
    resumed_provider = ScriptedProvider()
    resumed = Swarm(provider=resumed_provider, model="test-model", max_concurrency=1,
                    checkpoint_store=store, artifact_dir=None)
    result = resumed.resume(execution_id)
    assert resumed_provider.calls == ["problem", "queued", "skip-child", "grandchild"]
    assert all(node.status is NodeStatus.COMPLETED for node in result.graph.nodes)
    assert result.graph.nodes[0].result == "done: prior"
    assert result.total_cost_usd == 0.625
    completed = store.load(execution_id)
    assert completed["status"] == "completed"
    assert completed["budget"]["node_costs"] == dict.fromkeys(
        [node.id for node in graph.nodes], 0.125,
    )
    assert resumed.resume(execution_id).output == result.output
    assert resumed_provider.calls == ["problem", "queued", "skip-child", "grandchild"]


SKIP_ERROR = "HTTP 500 from upstream: stack trace /srv/secret/path.py line 42"


class PromptRecordingProvider(Provider):
    """Fails the "Fetch" step and records every other prompt by label."""

    def __init__(self, *, crash_summary=False):
        self.prompts = {}
        self.crash_summary = crash_summary

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        if label == "Fetch":
            raise RuntimeError(SKIP_ERROR)
        self.prompts[label] = prompt
        if label == "Summarize" and self.crash_summary:
            raise RuntimeError("process died")
        return CompletionResult(f"done: {label}", cost_usd=0.125)


def make_skip_graph():
    return ExecutionGraph([Topology.SERIAL], [
        Node("Fetch", id="fetch", failure_policy=FailurePolicy.SKIP),
        Node("Other", id="other"),
        Node("Summarize", id="summ", depends_on=["fetch", "other"]),
    ])


def assert_marker_not_error(prompt):
    assert f"[fetch]: {SKIPPED_DEPENDENCY_RESULT}" in prompt
    assert "[other]: done: Other" in prompt
    assert "secret" not in prompt and "HTTP 500" not in prompt
    # The terminal note still asks for carried-forward context; only the
    # marker, never the error, is there to carry.
    assert TERMINAL_DELIVERABLE_NOTE in prompt


def test_skipped_dependency_passes_marker_not_error_text(mode):
    provider = PromptRecordingProvider()
    graph, tracer = make_skip_graph(), Tracer()
    options = dict(provider=provider, registry=Registry(), tracer=tracer, artifact_dir=None)
    if mode == "serial":
        Executor(**options).run(graph)
    else:
        asyncio.run(AsyncExecutor(**options, max_concurrency=1).run(graph))

    fetch = graph.nodes[0]
    assert fetch.status is NodeStatus.SKIPPED
    # The error stays on the node and in the trace for diagnosis.
    assert fetch.result == SKIP_ERROR
    assert any(span["node_id"] == "fetch" and span["error"] == SKIP_ERROR
               for span in tracer.summary())
    assert_marker_not_error(provider.prompts["Summarize"])


@pytest.mark.parametrize("parallel", [False, True])
def test_resumed_dependent_sees_skip_marker_and_checkpoint_keeps_error(parallel, tmp_path):
    store = FileCheckpointStore(tmp_path)
    first = PromptRecordingProvider(crash_summary=True)
    swarm = Swarm(provider=first, model="test-model", parallel=parallel,
                  checkpoint_store=store, artifact_dir=None)
    with pytest.raises(RuntimeError, match="process died"):
        swarm.execute(make_skip_graph())
    assert_marker_not_error(first.prompts["Summarize"])
    [execution_id] = store.list_ids()
    saved = {node["id"]: node for node in store.load(execution_id)["graph"]["nodes"]}
    assert saved["fetch"]["status"] == "skipped"
    assert saved["fetch"]["result"] == SKIP_ERROR

    resumed_provider = PromptRecordingProvider()
    resumed = Swarm(provider=resumed_provider, model="test-model", parallel=parallel,
                    checkpoint_store=store, artifact_dir=None).resume(execution_id)

    assert list(resumed_provider.prompts) == ["Summarize"]
    assert_marker_not_error(resumed_provider.prompts["Summarize"])
    assert "secret" not in resumed.output
    final = {node["id"]: node for node in store.load(execution_id)["graph"]["nodes"]}
    assert final["fetch"]["status"] == "skipped"
    assert final["fetch"]["result"] == SKIP_ERROR
