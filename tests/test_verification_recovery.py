"""Durable verification decisions survive each control-transition crash point."""

from __future__ import annotations

import asyncio
import copy

import pytest

from smythe import VerificationRecoveryError
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    FileCheckpointStore,
    build_state,
    graph_from_dict,
    graph_to_dict,
)
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.verifier import CallableVerifier
from smythe.tracer import Tracer


class ScriptedProvider(Provider):
    def __init__(self, judge="PASS"):
        self.calls = []
        self.judge = judge

    async def complete(self, system, prompt, model):
        node_id = prompt.splitlines()[0]
        self.calls.append(node_id)
        text = self.judge if node_id == "judge" else f"fresh {node_id}"
        return CompletionResult(text=text, cost_usd=0.1)


def graph_with_gate():
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"],
             verifies="draft", max_regenerations=1),
        Node(id="consumer", label="consumer", depends_on=["draft", "judge"]),
    ])


def completed_gate(*, state="pending", passed=False):
    graph = graph_with_gate()
    draft, judge, _ = graph.nodes
    draft.status = judge.status = NodeStatus.COMPLETED
    draft.result = "old draft"
    judge.result = "PASS" if passed else "FAIL old draft"
    judge.metadata["verification_receipt"] = {
        "version": 1, "state": state, "target_id": "draft",
        "judge_generation": 0, "target_generation": 0,
    }
    draft.metadata["cost_usd"] = judge.metadata["cost_usd"] = 0.1
    return graph


def save_graph(store, graph, *, version=CHECKPOINT_VERSION, completed=False):
    state = build_state(
        execution_id="run", status="completed" if completed else "failed",
        model="test", graph=graph, registry=Registry(), task=None,
        max_budget_usd=10.0, node_costs={"draft": 0.1, "judge": 0.1},
        output="stale cached output" if completed else None,
    )
    state["version"] = version
    store.save("run", state)
    return state


@pytest.mark.parametrize("completed", [False, True])
def test_resume_consumes_pending_verdict_before_dispatch_or_cached_output(tmp_path, completed):
    graph = completed_gate()
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=completed)
    provider = ScriptedProvider()
    result = Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == ["draft", "judge", "consumer"]
    assert result.output == "fresh consumer"
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert result.total_cost_usd == pytest.approx(0.5)
    assert all(n.metadata["execution_generation"] == 1 for n in result.graph.nodes)


@pytest.mark.parametrize("passed", [False, True])
def test_consumed_verdict_is_not_rejudged_on_resume(tmp_path, passed):
    graph = completed_gate(state="consumed", passed=passed)
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph)
    provider = ScriptedProvider()

    def must_not_rejudge(*args):
        pytest.fail("a consumed verdict was evaluated again")

    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   verifier=CallableVerifier(must_not_rejudge)).resume("run")
    assert provider.calls == ["consumer"]
    assert result.graph.nodes[0].result == "old draft"


@pytest.mark.parametrize("parallel", [False, True])
def test_control_saves_bypass_node_batching_and_snapshot_before_callbacks(tmp_path, parallel):
    class RecordingStore(FileCheckpointStore):
        def __init__(self):
            super().__init__(tmp_path)
            self.states = []

        def save(self, execution_id, state):
            self.states.append(state)  # Deliberately retain, without our own deepcopy.
            super().save(execution_id, state)

    store = RecordingStore()
    graph = graph_with_gate()
    provider = ScriptedProvider(judge="FAIL")
    result = Swarm(provider=provider, model="test", parallel=parallel,
                   checkpoint_store=store, checkpoint_every_n_nodes=100).execute(graph)
    judges = [state["graph"]["nodes"][1] for state in store.states]
    pending = next(j for j in judges if j["metadata"].get("verification_receipt", {}).get("state") == "pending")
    intent = next(j for j in judges if "regeneration_intent" in j["metadata"])
    reset = next(j for j in judges if j["metadata"].get("execution_generation") == 1
                 and j["status"] == "pending")
    assert pending["status"] == "completed"
    assert "regenerations_used" not in pending["metadata"]
    assert intent["metadata"]["regenerations_used"] == 1
    assert reset["result"] is None
    assert "regeneration_intent" not in reset["metadata"]
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert all(s["version"] == 3 for s in store.states)


@pytest.mark.parametrize("sync", [False, True])
def test_pending_control_write_failure_preserves_paid_verdict_without_retry(sync):
    graph = graph_with_gate()
    graph.nodes[1].failure_policy = FailurePolicy.RETRY
    graph.nodes[1].max_retries = 3
    provider = ScriptedProvider(judge="FAIL")
    budget = Sentinel(10.0)

    def broken_control_write():
        raise OSError("disk unavailable")

    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), budget=budget,
                  artifact_dir=None, on_control_update=broken_control_write)
    with pytest.raises(VerificationRecoveryError, match="persist") as caught:
        if sync:
            Executor(**kwargs).run(graph)
        else:
            asyncio.run(AsyncExecutor(**kwargs).run(graph))
    assert isinstance(caught.value.__cause__, OSError)
    assert provider.calls == ["draft", "judge"]
    assert graph.nodes[1].status is NodeStatus.COMPLETED
    assert graph.nodes[1].result == "FAIL"
    assert graph.nodes[1].metadata["verification_receipt"]["state"] == "pending"
    assert budget.total_cost_usd == pytest.approx(0.2)


@pytest.mark.parametrize("sync", [False, True])
def test_regular_update_failure_after_pending_save_cannot_retry_the_paid_judge(sync):
    graph = graph_with_gate()
    graph.nodes[1].failure_policy = FailurePolicy.RETRY
    graph.nodes[1].max_retries = 3
    provider = ScriptedProvider(judge="FAIL")
    saved = []

    def save_control():
        saved.append(graph_to_dict(graph))

    def broken_node_update(node):
        if node.id == "judge":
            raise OSError("observer failed after receipt save")

    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                  on_control_update=save_control, on_node_update=broken_node_update)
    with pytest.raises(VerificationRecoveryError) as caught:
        if sync:
            Executor(**kwargs).run(graph)
        else:
            asyncio.run(AsyncExecutor(**kwargs).run(graph))
    assert isinstance(caught.value.__cause__, OSError)
    assert provider.calls == ["draft", "judge"]
    assert graph.nodes[1].status is NodeStatus.COMPLETED
    assert saved[0]["nodes"][1]["metadata"]["verification_receipt"]["state"] == "pending"


@pytest.mark.parametrize("partial_reset", [False, True])
def test_saved_intent_replays_idempotently_without_reevaluation_or_refilled_allowance(tmp_path, partial_reset):
    graph = completed_gate()
    graph.nodes[2].status = NodeStatus.SKIPPED
    graph.nodes[2].result = "obsolete skipped result"
    graph.nodes[0].metadata["artifacts"] = [{"path": "obsolete.png"}]
    graph.nodes[2].metadata["artifacts_discarded"] = 2
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    assert intent is not None
    if partial_reset:
        # Model a reset interrupted before its control commit. Absolute
        # generations/counters make replay independent of the interruption point.
        graph.nodes[0].status = NodeStatus.PENDING
        graph.nodes[0].result = None
        graph.nodes[0].metadata["execution_generation"] = 1
        # A durable intent is authoritative; its old receipt is no longer needed.
        graph.nodes[1].metadata.pop("verification_receipt")
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph)
    provider = ScriptedProvider()

    def must_not_rejudge(*args):
        pytest.fail("a saved rejection intent was evaluated again")

    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   verifier=CallableVerifier(must_not_rejudge)).resume("run")
    assert provider.calls == ["draft", "judge", "consumer"]
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert all(n.metadata["execution_generation"] == 1 for n in result.graph.nodes)
    assert all("artifacts" not in n.metadata and "artifacts_discarded" not in n.metadata
               for n in result.graph.nodes)
    assert result.total_cost_usd == pytest.approx(0.5)
    before = list(provider.calls)
    Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == before


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("legacy_kind", ["ambiguous", "completed", "exhausted", "advisory"])
def test_legacy_verdict_recovery_is_conservative(tmp_path, version, legacy_kind):
    graph = completed_gate()
    judge = graph.nodes[1]
    judge.metadata.pop("verification_receipt")
    if legacy_kind == "exhausted":
        judge.metadata["regenerations_used"] = 1
    if legacy_kind == "advisory":
        judge.max_regenerations = 0
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, version=version, completed=legacy_kind == "completed")
    provider = ScriptedProvider()
    swarm = Swarm(provider=provider, model="test", checkpoint_store=store)
    if legacy_kind == "ambiguous":
        with pytest.raises(VerificationRecoveryError, match="Legacy checkpoint"):
            swarm.resume("run")
        assert provider.calls == []
    else:
        result = swarm.resume("run")
        assert provider.calls == ([] if legacy_kind == "completed" else ["consumer"])
        assert result.graph.nodes[0].result == "old draft"


@pytest.mark.parametrize("field,value", [
    ("judge_generation", True), ("judge_generation", -1),
    ("judge_generation", 2), ("target_generation", "0"),
    ("target_generation", 1), ("target_id", "consumer"),
    ("version", True), ("state", "unrecognized"),
])
def test_invalid_receipt_blocks_even_completed_cached_output(tmp_path, field, value):
    graph = completed_gate(state="consumed")
    graph.nodes[1].metadata["verification_receipt"][field] = value
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=True)
    provider = ScriptedProvider()
    with pytest.raises(VerificationRecoveryError):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []


@pytest.mark.parametrize("mutation", ["count", "inventory", "generation", "null"])
def test_malformed_intent_rejection_is_atomic(tmp_path, mutation):
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    if mutation == "count":
        intent["regenerations_used"] = 2
    elif mutation == "inventory":
        intent["affected_generations"].pop("consumer")
    elif mutation == "generation":
        intent["affected_generations"]["draft"] = True
    else:
        graph.nodes[1].metadata["regeneration_intent"] = None
    before = copy.deepcopy(graph_to_dict(graph))
    with pytest.raises(VerificationRecoveryError):
        executor.run(graph)
    assert graph_to_dict(graph) == before


@pytest.mark.parametrize("target_id", [[], {}, None])
@pytest.mark.parametrize("completed", [False, True])
def test_malformed_intent_target_blocks_resume_with_typed_error(tmp_path, target_id, completed):
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    intent["target_id"] = target_id
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=completed)
    provider = ScriptedProvider()
    with pytest.raises(VerificationRecoveryError):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []


def test_intent_cannot_overwrite_work_from_a_newer_generation():
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    executor.prepare_regeneration(graph.nodes[1], graph)
    graph.nodes[0].metadata["execution_generation"] = 1
    graph.nodes[0].result = "already newer work"
    before = graph_to_dict(graph)
    with pytest.raises(VerificationRecoveryError, match="newer result"):
        executor.run(graph)
    assert graph_to_dict(graph) == before


def test_graph_snapshots_detach_nested_control_state():
    graph = completed_gate()
    snapshot = graph_to_dict(graph)
    restored = graph_from_dict(snapshot)
    graph.nodes[1].metadata["verification_receipt"]["state"] = "consumed"
    assert snapshot["nodes"][1]["metadata"]["verification_receipt"]["state"] == "pending"
    assert restored.nodes[1].metadata["verification_receipt"]["state"] == "pending"


@pytest.mark.parametrize("version", [True, False, 3.0, "3", None])
def test_checkpoint_version_requires_an_actual_supported_integer(tmp_path, version):
    store = FileCheckpointStore(tmp_path)
    save_graph(store, completed_gate(state="consumed"), version=version, completed=True)
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="version"):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []
