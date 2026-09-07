"""Durable provenance survives graph handoffs and cannot become an unmanaged run."""

from copy import deepcopy

import pytest

from smythe import FileCheckpointStore, OfflineProvider, Swarm, Task
from smythe.checkpoint import build_state, graph_from_dict, graph_to_dict
from smythe.graph import ExecutionGraph, Node, Topology, snapshot_run_ref
from smythe.registry import Registry


REFERENCE = {
    "version": 1, "store_id": "store_fixture", "run_id": "run_fixture",
    "recipe_sha256": "a" * 64,
}


def graph():
    return ExecutionGraph(
        [Topology.SERIAL], [Node(id="one", label="One")],
        task=Task("One", context={"source": "Data"}), run_ref=deepcopy(REFERENCE),
    )


def test_graph_exports_detach_durable_identity_and_complete_task():
    source = graph()
    checkpoint_graph = graph_to_dict(source)
    restored = graph_from_dict(checkpoint_graph)
    assert restored.run_ref == REFERENCE
    assert restored.task.context == {"source": "Data"}
    checkpoint_graph["run_ref"]["run_id"] = "different"
    assert restored.run_ref == REFERENCE and source.run_ref == REFERENCE
    public_export = source.to_json()
    assert public_export["run_ref"] == REFERENCE
    public_export["run_ref"].clear()
    assert source.run_ref == REFERENCE


@pytest.mark.parametrize("bad", [
    {}, [], "run_fixture", {**REFERENCE, "version": True},
    {**REFERENCE, "version": 1.0}, {**REFERENCE, "version": 2},
    {**REFERENCE, "store_id": "../outside"}, {**REFERENCE, "run_id": ""},
    {**REFERENCE, "recipe_sha256": "a" * 63},
    {**REFERENCE, "recipe_sha256": "A" * 64},
    {**REFERENCE, "extra": "hidden binding"},
])
def test_invalid_reference_is_rejected_before_becoming_a_handoff(bad):
    with pytest.raises(ValueError):
        snapshot_run_ref(bad)
    data = graph_to_dict(graph())
    data["run_ref"] = bad
    with pytest.raises(ValueError):
        graph_from_dict(data)


class NoCalls(OfflineProvider):
    async def complete(self, *args, **kwargs):
        raise AssertionError("A durable graph cannot dispatch through unmanaged execution")


@pytest.mark.parametrize("parallel", [False, True])
def test_unmanaged_execute_rejects_durable_graph_before_provider_calls(parallel):
    with pytest.raises(ValueError, match="matching durable run_store"):
        Swarm(provider=NoCalls(), parallel=parallel).execute(graph())


@pytest.mark.parametrize("completed", [False, True])
def test_unmanaged_resume_rejects_durable_reference_even_for_cached_result(tmp_path, completed):
    store = FileCheckpointStore(tmp_path)
    state = build_state(
        execution_id="old_format", status="completed" if completed else "failed",
        model="offline", graph=graph(), registry=Registry(), task=graph().task,
        max_budget_usd=1, node_costs={}, output="stale" if completed else None,
    )
    store.save("old_format", state)
    with pytest.raises(ValueError, match="matching durable run_store"):
        Swarm(provider=NoCalls(), checkpoint_store=store).resume("old_format")


def test_legacy_graph_without_reference_remains_readable():
    data = graph_to_dict(graph())
    del data["run_ref"]
    assert graph_from_dict(data).run_ref is None
