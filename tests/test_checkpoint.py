"""Tests for checkpointing — store, serialization, and crash/resume."""

import pytest

from smythe.checkpoint import (
    FileCheckpointStore,
    graph_from_dict,
    graph_to_dict,
    node_from_dict,
    node_to_dict,
    reset_incomplete_nodes,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.swarm import Swarm


class ScriptedProvider(Provider):
    """Succeeds except for labels in fail_labels, which fail exactly once."""

    def __init__(self, fail_labels: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._fail = set(fail_labels or ())

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        label = prompt.split("\n")[0]
        self.calls.append(label)
        if label in self._fail:
            self._fail.discard(label)
            raise RuntimeError(f"boom: {label}")
        return CompletionResult(text=f"ok: {label}", prompt_tokens=10, completion_tokens=10)


def _serial_graph() -> ExecutionGraph:
    a = Node(id="a", label="A")
    b = Node(id="b", label="B", depends_on=["a"])
    c = Node(id="c", label="C", depends_on=["b"])
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])


# ---------------------------------------------------------------------------
# FileCheckpointStore
# ---------------------------------------------------------------------------


def test_store_save_load_roundtrip(tmp_path):
    store = FileCheckpointStore(tmp_path)
    state = {"version": 1, "execution_id": "abc123", "nested": {"x": [1, 2]}}
    store.save("abc123", state)
    assert store.load("abc123") == state


def test_store_load_unknown_id_returns_none(tmp_path):
    store = FileCheckpointStore(tmp_path)
    assert store.load("doesnotexist") is None


def test_store_delete_and_list(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save("one", {"v": 1})
    store.save("two", {"v": 2})
    assert store.list_ids() == ["one", "two"]
    store.delete("one")
    store.delete("never-existed")  # no-op
    assert store.list_ids() == ["two"]


def test_store_rejects_path_traversal_ids(tmp_path):
    store = FileCheckpointStore(tmp_path)
    with pytest.raises(ValueError, match="execution_id"):
        store.save("../escape", {})
    with pytest.raises(ValueError, match="execution_id"):
        store.load("a/b")


def test_store_save_is_atomic_no_tmp_left_behind(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save("abc", {"v": 1})
    store.save("abc", {"v": 2})
    assert [p.name for p in tmp_path.iterdir()] == ["abc.json"]
    assert store.load("abc") == {"v": 2}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_graph_roundtrip_preserves_every_field():
    node = Node(
        id="n1",
        label="Do a thing",
        agent_id="agent-42",
        depends_on=["n0"],
        result="the answer",
        status=NodeStatus.COMPLETED,
        metadata={"model": "claude-opus-4-8", "cost_usd": 0.01},
        failure_policy=FailurePolicy.RETRY,
        max_retries=3,
        required_capabilities=["research"],
        timeout_s=45.0,
    )
    other = Node(id="n0", label="First")
    graph = ExecutionGraph(
        topology=[Topology.FORK_JOIN, Topology.SERIAL],
        nodes=[other, node],
        estimated_cost_usd=0.25,
    )

    restored = graph_from_dict(graph_to_dict(graph))

    assert restored.topology == [Topology.FORK_JOIN, Topology.SERIAL]
    assert restored.estimated_cost_usd == 0.25
    r = restored.nodes[1]
    assert r.id == "n1"
    assert r.label == "Do a thing"
    assert r.agent_id == "agent-42"
    assert r.depends_on == ["n0"]
    assert r.result == "the answer"
    assert r.status == NodeStatus.COMPLETED
    assert r.metadata == {"model": "claude-opus-4-8", "cost_usd": 0.01}
    assert r.failure_policy == FailurePolicy.RETRY
    assert r.max_retries == 3
    assert r.required_capabilities == ["research"]
    assert r.timeout_s == 45.0


def test_non_json_result_falls_back_to_str():
    class Weird:
        def __str__(self) -> str:
            return "weird-object"

    node = Node(id="n", label="L", result=Weird())
    assert node_to_dict(node)["result"] == "weird-object"


def test_reset_incomplete_nodes():
    done = Node(id="done", label="d", status=NodeStatus.COMPLETED, result="kept")
    skipped = Node(id="skip", label="s", status=NodeStatus.SKIPPED)
    running = Node(id="run", label="r", status=NodeStatus.RUNNING, result="partial")
    failed = Node(id="fail", label="f", status=NodeStatus.FAILED, result="boom")
    graph = ExecutionGraph(
        topology=[Topology.SERIAL], nodes=[done, skipped, running, failed],
    )

    reset = reset_incomplete_nodes(graph)

    assert set(reset) == {"run", "fail"}
    assert done.status == NodeStatus.COMPLETED and done.result == "kept"
    assert skipped.status == NodeStatus.SKIPPED
    assert running.status == NodeStatus.PENDING and running.result is None
    assert failed.status == NodeStatus.PENDING and failed.result is None


# ---------------------------------------------------------------------------
# Crash and resume through Swarm
# ---------------------------------------------------------------------------


def test_crash_then_resume_completes_without_rerunning_finished_nodes(tmp_path):
    store = FileCheckpointStore(tmp_path)
    provider = ScriptedProvider(fail_labels={"B"})
    swarm = Swarm(
        provider=provider, model="test-model", checkpoint_store=store, parallel=True,
    )

    with pytest.raises(RuntimeError, match="boom: B"):
        swarm.execute(_serial_graph())

    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    assert state["status"] == "failed"
    statuses = {n["id"]: n["status"] for n in state["graph"]["nodes"]}
    assert statuses == {"a": "completed", "b": "failed", "c": "pending"}

    # Simulate a fresh process: new Swarm, same store and provider.
    resumed = Swarm(provider=provider, model="test-model", checkpoint_store=store)
    result = resumed.resume(execution_id)

    assert result.execution_id == execution_id
    assert "ok: A" in result.output
    assert "ok: B" in result.output
    assert "ok: C" in result.output
    assert provider.calls.count("A") == 1, "completed node was re-executed on resume"
    assert store.load(execution_id)["status"] == "completed"


def test_resume_restores_budget_accounting(tmp_path):
    store = FileCheckpointStore(tmp_path)
    provider = ScriptedProvider(fail_labels={"B"})
    swarm = Swarm(
        provider=provider, model="test-model", checkpoint_store=store,
        parallel=True, max_budget_usd=1.0,
    )

    with pytest.raises(RuntimeError):
        swarm.execute(_serial_graph())

    [execution_id] = store.list_ids()
    cost_after_crash = sum(
        store.load(execution_id)["budget"]["node_costs"].values()
    )
    assert cost_after_crash > 0  # node "a" already paid for

    resumed = Swarm(provider=provider, model="test-model", checkpoint_store=store)
    result = resumed.resume(execution_id)

    final_costs = store.load(execution_id)["budget"]["node_costs"]
    assert set(final_costs) == {"a", "b", "c"}
    assert result.total_cost_usd == pytest.approx(sum(final_costs.values()))
    assert result.total_cost_usd > cost_after_crash


def test_resume_of_completed_execution_returns_stored_result(tmp_path):
    store = FileCheckpointStore(tmp_path)
    provider = ScriptedProvider()
    swarm = Swarm(
        provider=provider, model="test-model", checkpoint_store=store, parallel=True,
    )

    first = swarm.execute(_serial_graph())
    calls_before = len(provider.calls)

    result = swarm.resume(first.execution_id)

    assert result.output == first.output
    assert len(provider.calls) == calls_before, "resume of a finished run made calls"


def test_resume_without_store_raises():
    swarm = Swarm(provider=ScriptedProvider(), model="test-model")
    with pytest.raises(ValueError, match="checkpoint_store"):
        swarm.resume("whatever")


def test_resume_unknown_id_raises(tmp_path):
    swarm = Swarm(
        provider=ScriptedProvider(), model="test-model",
        checkpoint_store=FileCheckpointStore(tmp_path),
    )
    with pytest.raises(KeyError, match="unknownid"):
        swarm.resume("unknownid")


def test_resume_rejects_unknown_version(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save("old", {"version": 99, "graph": {"topology": ["serial"], "nodes": []}})
    swarm = Swarm(
        provider=ScriptedProvider(), model="test-model", checkpoint_store=store,
    )
    with pytest.raises(ValueError, match="version"):
        swarm.resume("old")


def test_sync_execution_also_checkpoints(tmp_path):
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(
        provider=ScriptedProvider(), model="test-model",
        checkpoint_store=store, parallel=False,
    )

    result = swarm.execute(_serial_graph())

    state = store.load(result.execution_id)
    assert state["status"] == "completed"
    assert all(n["status"] == "completed" for n in state["graph"]["nodes"])
    assert state["output"] == result.output


def test_max_tool_iterations_roundtrips_and_old_checkpoints_default():
    node = Node(id="n", label="L", max_tool_iterations=4)
    restored = node_from_dict(node_to_dict(node))
    assert restored.max_tool_iterations == 4

    legacy = node_to_dict(Node(id="m", label="L2"))
    legacy.pop("max_tool_iterations")  # pre-M2 checkpoint shape
    assert node_from_dict(legacy).max_tool_iterations is None
