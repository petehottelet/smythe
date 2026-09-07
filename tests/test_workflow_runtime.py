"""Offline integration of durable graph handoffs and control recovery."""

import asyncio
from copy import deepcopy

import pytest

from smythe import LocalOnly, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task
from smythe.graph import ExecutionGraph, Node, NodeStatus, Revision, Topology
from smythe.registry import Registry
from smythe.workflow import WorkflowRuntime, _nanousd
from smythe.workflow_store import WorkflowConflictError


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


def swarm(store, **kwargs):
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **kwargs)


def graph():
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="first", label="First"), Node(id="last", label="Last", depends_on=["first"]),
    ])


def test_plan_execute_resume_preserves_task_run_and_local_cost_scope(store):
    runtime = swarm(store)
    planned = runtime.plan(Task("Report", constraints=["Brief"], context={"source": "supplied"},
                                done_when=["Answers the question"]))
    run_id = planned.run_ref["run_id"]
    result = runtime.execute(planned)
    assert result.execution_id == run_id
    assert result.graph.task.context == {"source": "supplied"}
    assert result.total_cost_usd == 0 and result.cost_is_complete
    assert result.cost_scope == "complete_text_workflow"
    assert result.workflow_accounting["call_count"] == len(planned.nodes)
    before = store.inspect_run(run_id)
    cached = runtime.resume(run_id)
    after = store.inspect_run(run_id)
    assert cached.output == result.output and cached.execution_id == run_id
    assert before["calls"] == after["calls"]
    assert before["revision"] == after["revision"]


def test_caller_graph_has_no_invented_task_or_shared_agent_mutations(store):
    registry = Registry()
    source = graph()
    result = swarm(store, registry=registry).execute(source)
    assert result.graph.task is None
    assert store.load_run(result.execution_id)["task"] is None
    assert registry.list_agents() == []
    assert all(node.agent_id is None and node.status is NodeStatus.PENDING for node in source.nodes)


def test_pending_plan_can_be_reviewed_and_edited_before_execution(store):
    runtime = swarm(store)
    planned = runtime.plan(Task("Report"))
    planned.nodes[0].label = "Reviewed task wording"
    result = runtime.execute(planned)
    assert result.graph.nodes[0].label == "Reviewed task wording"
    saved = store.get_checkpoint(result.execution_id)["checkpoint"]
    assert saved["graph"]["nodes"][0]["label"] == "Reviewed task wording"


@pytest.mark.parametrize("change", ["model", "budget", "concurrency", "task", "store", "recipe"])
def test_changed_handoff_binding_rejected_before_calls(store, tmp_path, change):
    runtime = swarm(store)
    planned = runtime.plan(Task("Report"))
    before = store.inspect_run(planned.run_ref["run_id"])["call_count"]
    if change == "model":
        runtime.model = "changed"
    elif change == "budget":
        runtime.max_budget_usd = 2
    elif change == "concurrency":
        runtime.max_concurrency = 4
    elif change == "task":
        planned.task.goal = "changed"
    elif change == "recipe":
        planned.run_ref["recipe_sha256"] = "b" * 64
    else:
        with SQLiteWorkflowStore(tmp_path / "other.db") as other:
            with pytest.raises(WorkflowConflictError):
                swarm(other).execute(planned)
        return
    with pytest.raises(WorkflowConflictError):
        runtime.execute(planned)
    assert store.inspect_run(planned.run_ref["run_id"])["call_count"] == before


@pytest.mark.parametrize("field,value", [
    ("max_concurrency", None), ("max_concurrency", True), ("max_concurrency", 0),
    ("max_revisions", True), ("max_revisions", -1),
])
def test_invalid_policy_rejected_before_planning(store, field, value, monkeypatch):
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider call"))
    runtime = swarm(store, **{field: value})
    with pytest.raises(ValueError):
        runtime.execute(Task("Report"))


def test_config_snapshot_and_parallel_runs_do_not_mutate_sources(store):
    provider = OfflineProvider(echo_prefix="[isolated]")
    runtime = Swarm(model="offline", provider=provider, run_store=store,
                    architect=SimpleArchitect(), parallel=True)

    async def run():
        return await asyncio.gather(runtime.execute_async(Task("One")), runtime.execute_async(Task("Two")))

    first, second = asyncio.run(run())
    assert first.execution_id != second.execution_id
    assert first.graph.task.goal == "One" and second.graph.task.goal == "Two"
    assert runtime._registry.list_agents() == []
    assert set(call["call_id"] for call in store.inspect_run(first.execution_id)["calls"]).isdisjoint(
        call["call_id"] for call in store.inspect_run(second.execution_id)["calls"]
    )


class Crash(BaseException):
    pass


def crash_checkpoints(monkeypatch, store, condition):
    original = store.save_checkpoint
    triggered = []

    def save(lease, revision, state, **kwargs):
        if triggered or condition(state, kwargs):
            triggered.append(True)
            raise Crash("Simulated process loss")
        return original(lease, revision, state, **kwargs)

    monkeypatch.setattr(store, "save_checkpoint", save)
    return lambda: monkeypatch.setattr(store, "save_checkpoint", original)


def test_raw_worker_result_replays_after_lost_graph_checkpoint(store, monkeypatch):
    runtime = swarm(store)
    planned = runtime.plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    restore = crash_checkpoints(monkeypatch, store, lambda state, _: any(
        node["status"] == "completed" for node in state["graph"]["nodes"]
    ))
    with pytest.raises(Crash):
        runtime.execute(planned)
    accepted = store.inspect_run(run_id)["calls"]
    assert len(accepted) == 1 and accepted[0]["result_state"] == "accepted"
    restore()
    original = OfflineProvider.complete
    calls = []

    async def complete(self, *args, **kwargs):
        calls.append(args)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    result = runtime.resume(run_id)
    assert len(calls) == len(planned.nodes) - 1
    assert result.cost_is_complete and store.audit(run_id)["ok"]


def test_completed_plan_operation_restores_exact_ids_after_checkpoint_loss(store, monkeypatch):
    runtime = swarm(store)
    ids = []
    original = store.create_run

    def create(*args, **kwargs):
        run = original(*args, **kwargs)
        ids.append(run["run_id"])
        return run

    monkeypatch.setattr(store, "create_run", create)
    restore = crash_checkpoints(monkeypatch, store, lambda state, _: state["status"] == "planned")
    with pytest.raises(Crash):
        runtime.plan(Task("Report"))
    operation = store.load_operation(ids[0], "planning")
    planned_ids = [node["id"] for node in operation["result"]["graph"]["nodes"]]
    restore()
    result = runtime.resume(ids[0])
    assert [node.id for node in result.graph.nodes] == planned_ids
    assert store.load_operation(ids[0], "planning")["state"] == "applied"


@pytest.mark.parametrize("revision", [False, True])
def test_supervisor_replays_saved_decision_after_control_checkpoint_loss(store, monkeypatch, revision):
    reviews = []

    class Review:
        async def review(self, graph, node, **kwargs):
            reviews.append((node.id, deepcopy(kwargs)))
            if revision and node.id == "first":
                return Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=["last"]),))
            return None

    supervisor = LocalOnly(Review, "test-review", "1", role="supervisor")
    runtime = swarm(store, supervisor=supervisor, max_revisions=1)
    recorded = []
    create = store.create_run

    def capture(*args, **kwargs):
        result = create(*args, **kwargs)
        recorded.append(result["run_id"])
        return result

    monkeypatch.setattr(store, "create_run", capture)
    restore = crash_checkpoints(monkeypatch, store, lambda state, _: any(
        node["metadata"].get("workflow_supervision", {}).get("state") == "applied"
        for node in state["graph"]["nodes"]
    ))
    with pytest.raises(Crash):
        runtime.execute(graph())
    assert [item[0] for item in reviews] == ["first"]
    restore()
    result = runtime.resume(recorded[0])
    assert [item[0] for item in reviews].count("first") == 1
    assert ("extra" in {node.id for node in result.graph.nodes}) is revision
    assert store.get_checkpoint(recorded[0])["checkpoint"]["revisions_used"] == int(revision)


def test_offline_llm_planning_is_journaled_zero_api_cost(store):
    provider = OfflineProvider(plan={"nodes": [{"id": "write", "label": "Write the report"}]})
    runtime = Swarm(model="offline", provider=provider, run_store=store)
    result = runtime.execute(Task("Write a report"))
    records = store.inspect_run(result.execution_id)
    assert "planning" in {call["key"]["phase"] for call in records["calls"]}
    assert result.total_cost_usd == 0 and records["confirmed_nanousd"] == 0


def test_nanousd_budget_floors_without_decimal_context_rounding():
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 2
        assert _nanousd(12345.678901234) == 12345678901234
        assert _nanousd(.0000000009) == 0


def test_preflight_is_local_for_unsupported_components(store, monkeypatch):
    monkeypatch.setattr(OfflineProvider, "complete", lambda *args: pytest.fail("Provider call"))
    with pytest.raises(ValueError, match="LocalOnly"):
        Swarm(model="offline", provider=OfflineProvider(), run_store=store,
              architect=object()).execute(Task("No paid calls"))


def test_legacy_result_scope_does_not_claim_complete_accounting():
    result = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect()).execute(Task("Test"))
    assert result.cost_scope == "execution_and_synthesis" and result.workflow_accounting is None


def test_runtime_preflight_snapshot_is_not_mutated_by_later_provider_edit(store):
    source = swarm(store)
    runtime = source._workflow_runtime()
    assert isinstance(runtime, WorkflowRuntime)
    source.max_revisions = 2
    assert runtime.recipe["max_revisions"] == 0


@pytest.mark.parametrize("metadata", [
    {"execution_generation": 1}, {"regenerations_used": 2},
    {"workflow_supervision": {"state": "applied", "generation": 0}},
    {"native_receipts": []}, {"response_error": {}}, {"cost_usd": 0},
])
def test_new_graph_cannot_import_prior_execution_history(store, metadata):
    source = graph()
    source.nodes[0].metadata.update(metadata)
    with pytest.raises(ValueError, match="inherit execution"):
        swarm(store).execute(source)


@pytest.mark.parametrize("field,value", [
    ("max_retries", True), ("max_regenerations", -1), ("timeout_s", float("inf")),
    ("timeout_s", 0),
])
def test_invalid_node_policy_is_rejected_before_dispatch(store, field, value):
    source = graph()
    setattr(source.nodes[0], field, value)
    with pytest.raises(ValueError):
        swarm(store).execute(source)
