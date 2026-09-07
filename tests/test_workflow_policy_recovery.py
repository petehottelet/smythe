"""Graph policy enforcement survives saved operations and graph recovery."""

from copy import deepcopy
import json

import pytest

from smythe import (
    LocalOnly, OfflineProvider, SQLiteWorkflowStore, Swarm, Task, WorkflowBindingError,
    WorkflowGraphPolicy,
)
from smythe.checkpoint import graph_to_dict, node_to_dict
from smythe.graph import Node, NodeStatus, Revision
from smythe.provider_responses import OpenAIResponsesProvider
from test_workflow_native_runtime import native_transport as native_transport
from test_workflow_policy import make_graph, policy_swarm


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "recovery.db") as value:
        yield value


class ProcessLost(BaseException):
    pass


def capture_runs(store, monkeypatch):
    runs = []
    original = store.create_run

    def create(*args, **kwargs):
        run = original(*args, **kwargs)
        runs.append(run["run_id"])
        return run

    monkeypatch.setattr(store, "create_run", create)
    return runs


def lose_checkpoints(store, monkeypatch, condition):
    original = store.save_checkpoint
    lost = []

    def save(lease, revision, state, **kwargs):
        if lost or condition(state):
            lost.append(True)
            raise ProcessLost("Checkpoint not committed")
        return original(lease, revision, state, **kwargs)

    monkeypatch.setattr(store, "save_checkpoint", save)
    return lambda: monkeypatch.setattr(store, "save_checkpoint", original)


def violate_graph(value, violation):
    if violation == "nodes":
        value["nodes"].append(node_to_dict(Node(id="extra", label="Extra")))
    elif violation == "model":
        value["nodes"][0]["metadata"]["model"] = "other"
    elif violation == "retries":
        value["nodes"][0]["max_retries"] = 2
    else:
        value["nodes"][0]["max_regenerations"] = 1


@pytest.mark.parametrize("status", ["planned", "running", "completed"])
@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations"])
def test_restored_graph_is_checked_before_any_dispatch_or_new_checkpoint(
    store, monkeypatch, status, violation,
):
    policy = WorkflowGraphPolicy(1, node_model="offline", max_retries=1, max_regenerations=0)
    swarm = policy_swarm(store, policy)
    planned = swarm.plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    if status == "running":
        restore = lose_checkpoints(store, monkeypatch, lambda state: state["status"] == "completed")
        with pytest.raises(ProcessLost):
            swarm.execute(planned)
        restore()
        assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "running"
    elif status == "completed":
        swarm.execute(planned)
    original = store.get_checkpoint
    saved = deepcopy(original(run_id))
    calls = store.inspect_run(run_id)["calls"]

    def load(identity):
        value = deepcopy(original(identity))
        violate_graph(value["checkpoint"]["graph"], violation)
        return value

    monkeypatch.setattr(store, "get_checkpoint", load)
    monkeypatch.setattr(store, "save_checkpoint", lambda *args, **kwargs: pytest.fail("Invalid checkpoint"))
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider called"))
    with pytest.raises(WorkflowBindingError, match="Graph policy"):
        swarm.resume(run_id)
    assert original(run_id) == saved
    assert store.inspect_run(run_id)["calls"] == calls


@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations"])
def test_completed_planning_operation_cannot_bypass_policy_after_checkpoint_loss(
    store, monkeypatch, violation,
):
    policy = WorkflowGraphPolicy(1, node_model="offline", max_retries=1, max_regenerations=0)
    swarm = policy_swarm(store, policy)
    runs = capture_runs(store, monkeypatch)
    restore = lose_checkpoints(store, monkeypatch, lambda state: state["status"] == "planned")
    with pytest.raises(ProcessLost):
        swarm.plan(Task("Report"))
    run_id = runs[0]
    assert store.get_checkpoint(run_id) is None
    operation = store.load_operation(run_id, "planning")
    assert operation["state"] == "completed"
    restore()
    original = store.load_operation

    def load(identity, key):
        value = deepcopy(original(identity, key))
        if value is not None and key == "planning":
            violate_graph(value["result"]["graph"], violation)
        return value

    monkeypatch.setattr(store, "load_operation", load)
    monkeypatch.setattr(store, "save_checkpoint", lambda *args, **kwargs: pytest.fail("Invalid checkpoint"))
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider called"))
    with pytest.raises(WorkflowBindingError, match="Graph policy"):
        swarm.resume(run_id)
    assert store.get_checkpoint(run_id) is None
    assert original(run_id, "planning") == operation
    assert store.inspect_run(run_id)["call_count"] == 0


def test_valid_saved_planning_operation_recovers_exact_graph_under_same_policy(store, monkeypatch):
    swarm = policy_swarm(store, WorkflowGraphPolicy(1, node_model="offline"))
    runs = capture_runs(store, monkeypatch)
    restore = lose_checkpoints(store, monkeypatch, lambda state: state["status"] == "planned")
    with pytest.raises(ProcessLost):
        swarm.plan(Task("Report"))
    operation = store.load_operation(runs[0], "planning")
    restore()
    result = swarm.resume(runs[0])
    assert [node.id for node in result.graph.nodes] == [
        node["id"] for node in operation["result"]["graph"]["nodes"]
    ]
    assert store.load_operation(runs[0], "planning")["state"] == "applied"
    assert store.inspect_run(runs[0])["call_count"] == 1 and store.audit(runs[0])["ok"]


@pytest.mark.parametrize("violation", ["model", "retries", "regenerations"])
def test_generated_graph_controls_are_rejected_before_execution(store, monkeypatch, violation):
    plan = {"nodes": [{"id": "draft", "label": "Draft", "metadata": {}}]}
    violate_graph(plan, violation)
    swarm = Swarm(model="offline", provider=OfflineProvider(plan=plan), run_store=store,
                  graph_policy=WorkflowGraphPolicy(
                      1, node_model="offline", max_retries=1, max_regenerations=0,
                  ))
    runs = capture_runs(store, monkeypatch)
    with pytest.raises(WorkflowBindingError, match="Graph policy"):
        swarm.plan(Task("Report"))
    accounting = store.inspect_run(runs[0])
    assert accounting["call_count"] == 1 and accounting["calls"][0]["key"]["phase"] == "planning"
    assert accounting["calls"][0]["billing_state"] == "known"
    assert store.get_checkpoint(runs[0]) is None


def test_rejected_native_planning_stays_metered_and_replays_without_rebuying(
    store, monkeypatch, native_transport,
):
    model = "gpt-6-astra"
    native_transport.planning_outputs = [json.dumps({"nodes": [
        {"id": f"n{i}", "label": f"Step {i}", "max_retries": 0} for i in range(9)
    ]})]
    swarm = Swarm(
        model=model, provider=OpenAIResponsesProvider(api_key="dummy-no-network", max_output_tokens=100),
        run_store=store, max_budget_usd=1,
        graph_policy=WorkflowGraphPolicy(8, node_model=model, max_retries=0, max_regenerations=0),
    )
    runs = capture_runs(store, monkeypatch)
    with pytest.raises(WorkflowBindingError, match="max_nodes=8"):
        swarm.execute(Task("Report"))
    run_id = runs[0]
    before = store.inspect_run(run_id)
    assert len(native_transport.requests) == len(native_transport.counts) == before["call_count"] == 1
    assert before["calls"][0]["key"]["phase"] == "planning"
    assert before["calls"][0]["billing_state"] == "known"
    assert before["confirmed_nanousd"] == 1_500_000
    assert before["reserved_nanousd"] == before["unknown_nanousd"] == 0
    assert store.get_checkpoint(run_id) is None
    assert store.load_run(run_id)["config"]["components"]["architect"]["type"] == "llm_architect"
    with pytest.raises(WorkflowBindingError, match="max_nodes=8"):
        swarm.resume(run_id)
    after = store.inspect_run(run_id)
    assert len(native_transport.requests) == 1
    assert after["calls"] == before["calls"]
    assert store.get_checkpoint(run_id) is None and store.audit(run_id)["ok"]


@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations", "cycle"])
def test_rejected_revision_is_atomic_durable_and_does_not_use_revision_allowance(store, violation):
    reviews = []
    additions = []

    class Review:
        async def review(self, graph, node, **kwargs):
            reviews.append(node.id)
            if node.id != "n0":
                return None
            added = Node(id="extra", label="Extra", depends_on=["n0"])
            additions.append(added)
            if violation == "nodes":
                return Revision(add_nodes=(added,), rewire={"n1": ()})
            if violation == "model":
                added.metadata["model"] = "other"
            elif violation == "retries":
                added.max_retries = 2
            elif violation == "regenerations":
                added.max_regenerations = 1
            else:
                added.depends_on = ["extra"]
            return Revision(add_nodes=(added,), drop_node_ids=("n1",))

    policy = WorkflowGraphPolicy(2, node_model="offline", max_retries=1, max_regenerations=0)
    swarm = policy_swarm(store, policy,
                         supervisor=LocalOnly(Review, "policy-review", "1", role="supervisor"),
                         max_revisions=1)
    result = swarm.execute(make_graph())
    assert [node.id for node in result.graph.nodes] == ["n0", "n1"]
    assert result.graph.nodes[1].depends_on == ["n0"]
    disposition = result.graph.nodes[0].metadata["workflow_supervision"]
    assert disposition["state"] == "applied" and disposition["revision_applied"] is False
    assert store.get_checkpoint(result.execution_id)["checkpoint"]["revisions_used"] == 0
    operation = store.load_operation(result.execution_id, "supervision/n0/0")
    assert operation["state"] == "applied" and operation["result"]["revision"] is not None
    assert store.inspect_run(result.execution_id)["call_count"] == 2
    assert reviews == ["n0", "n1"] and additions[0].agent_id is None
    assert any(span["status"] == "revision_rejected" for span in result.trace)
    assert store.audit(result.execution_id)["ok"]


@pytest.mark.parametrize("override_model", [False, True])
def test_valid_drop_and_add_at_cap_inherits_actual_execution_model(store, override_model):
    added = Node(id="extra", label="Extra", depends_on=["n0"])

    class Review:
        async def review(self, graph, node, **kwargs):
            return Revision(add_nodes=(added,), drop_node_ids=("n1",))

    model = "explicit-worker-model" if override_model else "offline"
    graph = make_graph()
    if override_model:
        for node in graph.nodes:
            node.metadata["model"] = model
    swarm = policy_swarm(store, WorkflowGraphPolicy(2, node_model=model),
                         supervisor=LocalOnly(Review, "replace-review", "1", role="supervisor"),
                         max_revisions=1)
    result = swarm.execute(graph)
    assert [node.id for node in result.graph.nodes] == ["n0", "extra"]
    assert all(node.metadata["model"] == model for node in result.graph.nodes)
    assert all(node.status is NodeStatus.COMPLETED for node in result.graph.nodes)
    assert added.metadata == {} and added.agent_id is None
    assert store.get_checkpoint(result.execution_id)["checkpoint"]["revisions_used"] == 1
    assert result.graph.nodes[0].metadata["workflow_supervision"]["revision_applied"] is True
    assert store.inspect_run(result.execution_id)["call_count"] == 2


def test_rejected_revision_replays_saved_proposal_after_disposition_checkpoint_loss(store, monkeypatch):
    reviews = []

    class Review:
        async def review(self, graph, node, **kwargs):
            reviews.append(node.id)
            if node.id == "n0":
                return Revision(add_nodes=(Node(id="extra", label="Extra"),), rewire={"n1": ()})
            return None

    swarm = policy_swarm(store, WorkflowGraphPolicy(2),
                         supervisor=LocalOnly(Review, "recover-review", "1", role="supervisor"),
                         max_revisions=1)
    runs = capture_runs(store, monkeypatch)
    restore = lose_checkpoints(store, monkeypatch, lambda state: any(
        node["metadata"].get("workflow_supervision", {}).get("state") == "applied"
        for node in state["graph"]["nodes"]
    ))
    with pytest.raises(ProcessLost):
        swarm.execute(make_graph())
    run_id = runs[0]
    operation = store.load_operation(run_id, "supervision/n0/0")
    assert operation["state"] == "completed" and reviews == ["n0"]
    restore()
    result = swarm.resume(run_id)
    assert reviews == ["n0", "n1"]
    assert [node.id for node in result.graph.nodes] == ["n0", "n1"]
    assert result.graph.nodes[1].depends_on == ["n0"]
    saved = store.load_operation(run_id, "supervision/n0/0")
    assert saved["result"] == operation["result"] and saved["state"] == "applied"
    assert result.graph.nodes[0].metadata["workflow_supervision"]["revision_applied"] is False
    assert store.get_checkpoint(run_id)["checkpoint"]["revisions_used"] == 0
    assert store.inspect_run(run_id)["call_count"] == 2 and store.audit(run_id)["ok"]


def test_checkpoint_guard_refuses_internal_policy_violation_before_serializing(store, monkeypatch):
    runtime = policy_swarm(store, WorkflowGraphPolicy(1))._workflow_runtime()
    runtime.graph = make_graph(2)
    before = graph_to_dict(runtime.graph)
    monkeypatch.setattr(store, "save_checkpoint", lambda *args, **kwargs: pytest.fail("Invalid checkpoint"))
    with pytest.raises(WorkflowBindingError, match="max_nodes"):
        runtime._save()
    assert graph_to_dict(runtime.graph) == before
