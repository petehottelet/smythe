"""Frozen graph limits across public durable workflow entry points."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from smythe import (
    OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task,
    WorkflowBindingError, WorkflowGraphPolicy,
)
from smythe.checkpoint import graph_to_dict
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.registry import Registry
from smythe.workflow_store import WorkflowConflictError


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "policy.db") as value:
        yield value


def make_graph(count=2):
    return ExecutionGraph([Topology.SERIAL], [
        Node(id=f"n{i}", label=f"Step {i}", depends_on=[f"n{i - 1}"] if i else [])
        for i in range(count)
    ])


def policy_swarm(store, policy=None, **kwargs):
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, graph_policy=policy, **kwargs)


@pytest.mark.parametrize("value", [None, 0, -1, True, False, 1.0, "8"])
def test_max_nodes_is_a_required_strict_positive_integer(value):
    with pytest.raises(WorkflowBindingError, match="max_nodes"):
        WorkflowGraphPolicy(value)


@pytest.mark.parametrize("field", ["max_retries", "max_regenerations"])
@pytest.mark.parametrize("value", [-1, True, False, 0.0, "0"])
def test_optional_limits_are_strict_nonnegative_integers(field, value):
    with pytest.raises(WorkflowBindingError, match=field):
        WorkflowGraphPolicy(8, **{field: value})


@pytest.mark.parametrize("value", ["", "  ", True, 1])
def test_node_model_is_nonempty_exact_text(value):
    with pytest.raises(WorkflowBindingError, match="node_model"):
        WorkflowGraphPolicy(8, node_model=value)


def test_policy_rejects_subclasses_and_mapping_impostors(store):
    class Text(str):
        pass

    class Derived(WorkflowGraphPolicy):
        pass

    with pytest.raises(WorkflowBindingError, match="node_model"):
        WorkflowGraphPolicy(8, node_model=Text("offline"))
    with pytest.raises(WorkflowBindingError, match="built-in"):
        Derived(8)
    with pytest.raises(WorkflowBindingError, match="built-in"):
        policy_swarm(store, {"max_nodes": 8})


def test_policy_is_frozen_and_snapshots_are_detached(store):
    policy = WorkflowGraphPolicy(8, node_model="offline", max_retries=1, max_regenerations=0)
    with pytest.raises(FrozenInstanceError):
        policy.max_nodes = 9
    description = policy.to_dict()
    assert WorkflowGraphPolicy.from_dict(description) == policy
    description["max_nodes"] = 99
    assert policy.max_nodes == 8
    swarm = policy_swarm(store, policy)
    runtime = swarm._workflow_runtime()
    assert runtime.graph_policy == swarm._graph_policy == policy
    assert runtime.graph_policy is not swarm._graph_policy and swarm._graph_policy is not policy
    # Even an explicit dataclass escape on the source cannot change a saved binding.
    object.__setattr__(policy, "max_nodes", 99)
    assert runtime.recipe["graph_policy"]["max_nodes"] == swarm._graph_policy.max_nodes == 8


@pytest.mark.parametrize("change", ["missing", "extra", "version", "bool-version", "list"])
def test_policy_description_has_an_exact_versioned_schema(change):
    value = WorkflowGraphPolicy(8).to_dict()
    if change == "missing":
        del value["max_retries"]
    elif change == "extra":
        value["extra"] = 1
    elif change == "version":
        value["version"] = 2
    elif change == "bool-version":
        value["version"] = True
    else:
        value = list(value.items())
    with pytest.raises(WorkflowBindingError):
        WorkflowGraphPolicy.from_dict(value)


def test_validation_uses_effective_model_and_never_stamps_or_clamps():
    graph = make_graph()
    graph.nodes[1].metadata["model"] = "offline"
    before = graph_to_dict(graph)
    policy = WorkflowGraphPolicy(2, node_model="offline", max_retries=1, max_regenerations=0)
    policy.validate(graph, default_model="offline")
    assert graph_to_dict(graph) == before
    graph.nodes[1].metadata["model"] = "other"
    before = graph_to_dict(graph)
    with pytest.raises(WorkflowBindingError, match="model"):
        policy.validate(graph, default_model="offline")
    assert graph_to_dict(graph) == before
    with pytest.raises(WorkflowBindingError, match="max_retries"):
        WorkflowGraphPolicy(2, max_retries=0).validate(graph, default_model="offline")
    assert graph.nodes[0].max_retries == 1  # HALT does not silently bypass a field cap.
    WorkflowGraphPolicy(2).validate(graph, default_model="offline")


def test_node_count_includes_completed_skipped_and_verification_nodes():
    graph = make_graph(3)
    graph.nodes[0].status = NodeStatus.COMPLETED
    graph.nodes[1].status = NodeStatus.SKIPPED
    graph.nodes[2].verifies = "n0"
    with pytest.raises(WorkflowBindingError, match="max_nodes=2"):
        WorkflowGraphPolicy(2).validate(graph, default_model="offline")


def test_policy_requires_durable_store_before_provider_work(tmp_path, monkeypatch):
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider called"))
    with pytest.raises(WorkflowBindingError, match="run_store"):
        Swarm(model="offline", provider=OfflineProvider(), graph_policy=WorkflowGraphPolicy(8))
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n  - id: first\n    label: First\n", encoding="utf-8")
    with pytest.raises(WorkflowBindingError, match="run_store"):
        Swarm.from_yaml(str(path), graph_policy=WorkflowGraphPolicy(8))


def test_yaml_checks_limits_before_adoption_and_runs_valid_policy(store, tmp_path, monkeypatch):
    path = tmp_path / "graph.yaml"
    path.write_text("nodes:\n  - id: first\n    label: First\n  - id: last\n    label: Last\n",
                    encoding="utf-8")
    original = Swarm._stamp_model
    with monkeypatch.context() as local:
        local.setattr(Swarm, "_stamp_model", lambda *args: pytest.fail("Invalid graph stamped"))
        with pytest.raises(WorkflowBindingError, match="max_nodes"):
            Swarm.from_yaml(str(path), model="offline", provider=OfflineProvider(), run_store=store,
                            graph_policy=WorkflowGraphPolicy(1))
    assert Swarm._stamp_model is original
    swarm = Swarm.from_yaml(str(path), model="offline", provider=OfflineProvider(), run_store=store,
                            graph_policy=WorkflowGraphPolicy(2, node_model="offline"))
    result = swarm.execute()
    assert len(result.graph.nodes) == result.workflow_accounting["call_count"] == 2
    assert store.load_run(result.execution_id)["config"]["graph_policy"]["max_nodes"] == 2


def test_eight_node_boundary_executes_with_exact_model_and_zero_retry_caps(store):
    graph = make_graph(8)
    for node in graph.nodes:
        node.max_retries = 0
    policy = WorkflowGraphPolicy(8, node_model="offline", max_retries=0, max_regenerations=0)
    result = policy_swarm(store, policy).execute(graph)
    assert len(result.graph.nodes) == result.workflow_accounting["call_count"] == 8
    assert all(node.status is NodeStatus.COMPLETED for node in result.graph.nodes)
    assert all(node.metadata["model"] == "offline" for node in result.graph.nodes)
    assert all(node.agent_id is None and node.metadata == {} for node in graph.nodes)
    assert store.audit(result.execution_id)["ok"]


def test_omitted_optional_limits_allow_supported_model_and_control_overrides(store):
    graph = make_graph()
    graph.nodes[-1].metadata["model"] = "another-offline-model"
    graph.nodes[-1].max_retries = 4
    graph.nodes[-1].max_regenerations = 3
    result = policy_swarm(store, WorkflowGraphPolicy(2)).execute(graph)
    assert result.graph.nodes[-1].metadata["model"] == "another-offline-model"
    assert result.graph.nodes[-1].max_retries == 4 and result.graph.nodes[-1].max_regenerations == 3


@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations"])
def test_caller_graph_rejection_precedes_assignment_checkpoint_and_dispatch(store, monkeypatch, violation):
    graph = make_graph(9 if violation == "nodes" else 2)
    if violation == "model":
        graph.nodes[-1].metadata["model"] = "other"
    elif violation == "retries":
        graph.nodes[-1].max_retries = 2
    elif violation == "regenerations":
        graph.nodes[-1].max_regenerations = 1
    before = graph_to_dict(graph)
    monkeypatch.setattr(Registry, "assign", lambda *args: pytest.fail("Invalid graph assigned"))
    monkeypatch.setattr(store, "save_checkpoint", lambda *args, **kwargs: pytest.fail("Invalid checkpoint"))
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider called"))
    policy = WorkflowGraphPolicy(8, node_model="offline", max_retries=1, max_regenerations=0)
    with pytest.raises(WorkflowBindingError, match="Graph policy"):
        policy_swarm(store, policy).execute(graph)
    assert graph_to_dict(graph) == before


@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations"])
def test_edited_handoff_rejection_preserves_saved_plan_and_journal(store, violation):
    policy = WorkflowGraphPolicy(1, node_model="offline", max_retries=1, max_regenerations=0)
    swarm = policy_swarm(store, policy)
    planned = swarm.plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    saved = deepcopy(store.get_checkpoint(run_id))
    calls = store.inspect_run(run_id)["calls"]
    if violation == "nodes":
        planned.nodes.append(Node(id="extra", label="Extra"))
    elif violation == "model":
        planned.nodes[0].metadata["model"] = "other"
    elif violation == "retries":
        planned.nodes[0].max_retries = 2
    else:
        planned.nodes[0].max_regenerations = 1
    before = graph_to_dict(planned)
    with pytest.raises(WorkflowBindingError, match="Graph policy"):
        swarm.execute(planned)
    assert store.get_checkpoint(run_id) == saved
    assert store.inspect_run(run_id)["calls"] == calls
    assert graph_to_dict(planned) == before


@pytest.mark.parametrize("changed", [
    None, WorkflowGraphPolicy(9, node_model="offline", max_retries=1, max_regenerations=0),
    WorkflowGraphPolicy(8, max_retries=1, max_regenerations=0),
    WorkflowGraphPolicy(8, node_model="offline", max_retries=2, max_regenerations=0),
    WorkflowGraphPolicy(8, node_model="offline", max_retries=1, max_regenerations=1),
])
def test_resume_cannot_remove_or_weaken_saved_policy(store, changed, monkeypatch):
    policy = WorkflowGraphPolicy(8, node_model="offline", max_retries=1, max_regenerations=0)
    planned = policy_swarm(store, policy).plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    saved = deepcopy(store.get_checkpoint(run_id))
    calls = store.inspect_run(run_id)["calls"]
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider called"))
    with pytest.raises(WorkflowConflictError, match="saved recipe"):
        policy_swarm(store, changed).resume(run_id)
    assert store.get_checkpoint(run_id) == saved
    assert store.inspect_run(run_id)["calls"] == calls


def test_legacy_recipe_omits_policy_and_keeps_recovery_compatible(store):
    swarm = policy_swarm(store)
    before = swarm._workflow_runtime().recipe
    assert "graph_policy" not in before
    result = swarm.execute(make_graph())
    assert store.load_run(result.execution_id)["config"] == before
    bounded = policy_swarm(store, WorkflowGraphPolicy(8))._workflow_runtime().recipe
    assert {key: value for key, value in bounded.items() if key != "graph_policy"} == before
    assert swarm.resume(result.execution_id).output == result.output


def test_policy_does_not_change_legacy_unmanaged_execution():
    result = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect()).execute(
        make_graph(9)
    )
    assert len(result.graph.nodes) == 9 and result.workflow_accounting is None
