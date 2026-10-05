"""Offline integration of durable graph handoffs and control recovery."""

import asyncio
from contextlib import closing
from copy import deepcopy
import json
from dataclasses import replace
import sqlite3
from types import MappingProxyType

import pytest

import fakes
from fakes import ManualClock, NativeTransport, release_price_version
from smythe import LocalOnly, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task
from smythe import workflow
from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE
from smythe.graph import ExecutionGraph, Node, NodeStatus, Revision, Topology
from smythe.pricing import PRICE_VERSION, price_table
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.workflow import WorkflowRuntime, _nanousd
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import CallKey, WorkflowConflictError, WorkflowCorruptionError


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
    ("retry_backoff_s", True), ("retry_backoff_s", -1), ("retry_backoff_s", float("nan")),
    ("retry_backoff_s", float("inf")),
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
    {"rejected_generations": [{"version": 1, "generation": 0}]},
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


def native_swarm(store):
    provider = OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)
    return Swarm(model="gpt-6-astra", provider=provider, max_budget_usd=1, run_store=store,
                 synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE))


def test_run_saved_before_a_price_release_resumes_at_recorded_and_current_prices(tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    path = tmp_path / "release.db"
    with SQLiteWorkflowStore(path) as store:
        planned = native_swarm(store).plan(Task("Write and check"))
        run_id = planned.run_ref["run_id"]
        restore = crash_checkpoints(monkeypatch, store, lambda state, _: any(
            node["status"] == "completed" for node in state["graph"]["nodes"]
        ))
        with pytest.raises(Crash):
            native_swarm(store).execute(planned)
        restore()
        saved = {call["call_id"] for call in store.inspect_run(run_id)["calls"]}
        recipe = store.load_run(run_id)["config_sha256"]
    sent = len(transport.requests)
    assert sent == len(saved) == 2
    # The release doubles the Astra output price and adds a model, so the
    # saved recipe's price version and model list both differ.
    rates = dict(price_table().rates)
    rates["gpt-6-astra"] = replace(rates["gpt-6-astra"], output=100_000)
    rates["gpt-7-test"] = rates["gpt-5.6-sol"]
    release_price_version(monkeypatch, version="openai-test-v2", rates=rates)
    with SQLiteWorkflowStore(path) as store:
        result = native_swarm(store).resume(run_id)
        versions = {call["call_id"]: store.lookup_call(run_id, CallKey(**call["key"]))["price_version"]
                    for call in store.inspect_run(run_id)["calls"]}
        assert store.load_run(run_id)["config_sha256"] == recipe
        assert store.audit(run_id)["ok"]
    assert result.output == "Final accepted output."
    # The accepted draft replays; only the check and synthesis are new calls.
    assert len(transport.requests) == sent + 2
    assert {versions[call_id] for call_id in saved} == {PRICE_VERSION}
    assert {version for call_id, version in versions.items() if call_id not in saved} == {"openai-test-v2"}
    # 1,500,000 nanoUSD per saved call at its recorded rates, 2,000,000 per new call.
    assert result.workflow_accounting["confirmed_nanousd"] == 2 * 1_500_000 + 2 * 2_000_000
    assert result.workflow_accounting["call_count"] == 4


def test_saved_recipe_naming_an_unregistered_price_version_is_refused(tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    path = tmp_path / "newer.db"
    # A newer release saved this run under a version this one does not register.
    with pytest.MonkeyPatch.context() as newer:
        release_price_version(newer, version="openai-test-v2")
        with SQLiteWorkflowStore(path) as store:
            run_id = native_swarm(store).plan(Task("Write and check")).run_ref["run_id"]
    sent = len(transport.requests)
    with SQLiteWorkflowStore(path) as store:
        with pytest.raises(WorkflowConflictError):
            native_swarm(store).resume(run_id)
        with pytest.raises(WorkflowCorruptionError):
            store.inspect_run(run_id)
    assert len(transport.requests) == sent


def test_default_recipe_keeps_its_legacy_fields(store):
    recipe = swarm(store)._workflow_runtime().recipe
    assert set(recipe) == {"type", "version", "model", "provider", "components", "max_revisions",
                           "retry_backoff_s", "max_concurrency", "verifier", "checkpoint_every_n_nodes"}
    assert (recipe["type"], recipe["version"], recipe["verifier"]) == ("text_workflow", 1, "token-v1")


def test_behavior_version_fields_are_read_from_the_saved_recipe(store, monkeypatch):
    legacy = swarm(store).plan(Task("Report")).run_ref["run_id"]
    recipe = store.load_run(legacy)["config_sha256"]
    # A later release adds a behavior version whose default is 2.
    monkeypatch.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType({"example_version": (1, 2)}))
    runtime = swarm(store)._workflow_runtime()
    assert runtime.recipe["example_version"] == 2
    assert asyncio.run(runtime.resume(legacy, max_concurrency=1)).execution_id == legacy
    assert runtime.behavior == {"example_version": 1}
    assert store.load_run(legacy)["config_sha256"] == recipe
    # A new run records its default and keeps it when a later default changes.
    started = asyncio.run(swarm(store)._workflow_runtime().plan(Task("Report"))).run_ref["run_id"]
    assert store.load_run(started)["config"]["example_version"] == 2
    monkeypatch.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType({"example_version": (1, 3)}))
    later = swarm(store)._workflow_runtime()
    asyncio.run(later.resume(started, max_concurrency=1))
    assert later.behavior == {"example_version": 2}
    # A value configured explicitly must equal the saved one.
    explicit = swarm(store)._workflow_runtime(behavior_settings={"example_version": 3})
    with pytest.raises(WorkflowConflictError, match="example_version"):
        asyncio.run(explicit.resume(started, max_concurrency=1))
    assert store.load_run(started)["config"]["example_version"] == 2


def test_new_run_records_an_explicit_behavior_version_and_keeps_it(store, monkeypatch):
    monkeypatch.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType({"example_version": (1, 2)}))
    runtime = swarm(store)._workflow_runtime(behavior_settings={"example_version": 3})
    assert runtime.recipe["example_version"] == 3
    planned = asyncio.run(runtime.plan(Task("Report"))).run_ref["run_id"]
    assert store.load_run(planned)["config"]["example_version"] == 3
    assert runtime.behavior == {"example_version": 3}
    recipe = store.load_run(planned)["config_sha256"]
    # Executing a Task creates its run the same way. A configured value equal
    # to the one a recipe without the field means is not recorded.
    unrecorded = swarm(store)._workflow_runtime(behavior_settings={"example_version": 1})
    executed = asyncio.run(unrecorded.execute(Task("Report"), max_concurrency=1)).execution_id
    assert "example_version" not in store.load_run(executed)["config"]
    assert unrecorded.behavior == {"example_version": 1}
    # A later release changes the default; each run keeps the value it recorded.
    monkeypatch.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType({"example_version": (1, 4)}))
    later = swarm(store)._workflow_runtime()
    result = asyncio.run(later.resume(planned, max_concurrency=1))
    assert result.execution_id == planned and result.output
    assert later.behavior == {"example_version": 3}
    assert store.load_run(planned)["config_sha256"] == recipe
    again = swarm(store)._workflow_runtime(behavior_settings={"example_version": 3})
    asyncio.run(again.resume(planned, max_concurrency=1))
    assert again.behavior == {"example_version": 3}
    other = swarm(store)._workflow_runtime()
    asyncio.run(other.resume(executed, max_concurrency=1))
    assert other.behavior == {"example_version": 1}


def test_behavior_settings_are_checked_and_fixed_when_the_runtime_is_built(store, monkeypatch):
    with pytest.raises(WorkflowBindingError, match="example_version"):
        swarm(store)._workflow_runtime(behavior_settings={"example_version": 3})
    monkeypatch.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType({"example_version": (1, 2)}))
    with pytest.raises(WorkflowBindingError):
        swarm(store)._workflow_runtime(behavior_settings={"example_version": float("nan")})
    with pytest.raises(WorkflowBindingError, match="mapping"):
        swarm(store)._workflow_runtime(behavior_settings=[("example_version", 3)])
    runtime = swarm(store)._workflow_runtime(behavior_settings={"example_version": 3})
    # The recipe is built once, so its settings cannot change afterwards.
    with pytest.raises(AttributeError):
        runtime.behavior_settings = {"example_version": 4}
    with pytest.raises(TypeError):
        runtime.behavior_settings["example_version"] = 4
    assert runtime.recipe["example_version"] == 3
    assert store.list_runs() == []


def gated_graph():
    """Research feeds a draft that an advisory judge checks; no node depends on the judge."""
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="research", label="Research"),
        Node(id="draft", label="Draft", depends_on=["research"]),
        Node(id="judge", label="Check", depends_on=["draft"], verifies="draft"),
    ])


def node_prompts(store, run_id):
    """Each node's prompt exactly as its journaled request carries it."""
    prompts = {}
    for call in store.inspect_run(run_id)["calls"]:
        if call["key"]["scope_id"].startswith("node/"):
            request = store.lookup_call(run_id, CallKey(**call["key"]))["request_json"]
            prompts[call["key"]["scope_id"].removeprefix("node/")] = json.loads(request)["prompt"]
    return prompts


# The requests every earlier release sent for gated_graph(): the judge is told
# it is the final step, and the draft it judges is not.
LEGACY_PROMPTS = {
    "research": "Research",
    "draft": "Draft\n\n\nContext from prior steps:\n\n[research]: offline: Research",
    "judge": ("Check\n\n\nContext from prior steps:\n\n[draft]: offline: Draft\n\n"
              + TERMINAL_DELIVERABLE_NOTE),
}
# The same requests under prompt_assembly 2, which ordinary runs use.
DELIVERABLE_PROMPTS = {
    "research": "Research",
    "draft": LEGACY_PROMPTS["draft"] + "\n\n" + TERMINAL_DELIVERABLE_NOTE,
    "judge": "Check\n\n\nContext from prior steps:\n\n[draft]: offline: Draft",
}


def test_durable_recipe_without_a_prompt_assembly_version_keeps_its_request_bytes(store):
    result = swarm(store).execute(gated_graph())
    assert "prompt_assembly" not in store.load_run(result.execution_id)["config"]
    assert node_prompts(store, result.execution_id) == LEGACY_PROMPTS


def test_durable_run_opts_into_prompt_assembly_2_through_its_recipe(store):
    runtime = swarm(store)._workflow_runtime(behavior_settings={"prompt_assembly": 2})
    assert runtime.recipe["prompt_assembly"] == 2
    run_id = asyncio.run(runtime.execute(gated_graph(), max_concurrency=1)).execution_id
    assert store.load_run(run_id)["config"]["prompt_assembly"] == 2
    assert node_prompts(store, run_id) == DELIVERABLE_PROMPTS
    # A runtime that configures nothing resumes it under its saved version.
    later = swarm(store)._workflow_runtime()
    asyncio.run(later.resume(run_id, max_concurrency=1))
    assert later.behavior == {"prompt_assembly": 2, "context_assembly": 0}
    # A configured version must equal the saved one, here a legacy run's 1.
    legacy = swarm(store).plan(Task("Report")).run_ref["run_id"]
    with pytest.raises(WorkflowConflictError, match="prompt_assembly"):
        asyncio.run(swarm(store)._workflow_runtime(behavior_settings={"prompt_assembly": 2})
                    .resume(legacy, max_concurrency=1))
    assert store.inspect_run(legacy)["call_count"] == 0


@pytest.mark.parametrize("settings,expected", [
    (None, LEGACY_PROMPTS), ({"prompt_assembly": 2}, DELIVERABLE_PROMPTS),
], ids=["legacy", "opted-in"])
def test_interrupted_durable_run_resends_the_request_its_recipe_assembled(
    tmp_path, monkeypatch, settings, expected,
):
    # The process is killed after the judge's call was prepared and reserved,
    # before it was sent. Resuming must prepare exactly the same request, or
    # the journal refuses it as a different call.
    image = tmp_path / "killed.db"
    judge = CallKey("verification", "node/judge")
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        def due(lease, call_id, _):
            record = store.lookup_call(lease.run_id, judge)
            return record is not None and record["call_id"] == call_id

        point = fakes.crash_when(monkeypatch, store, "claim_dispatch", due, image=image,
                                 connection=store._db)
        runtime = swarm(store)._workflow_runtime(behavior_settings=settings)
        with pytest.raises(fakes.Crash):
            asyncio.run(runtime.execute(gated_graph(), max_concurrency=1))
        (run,) = store.list_runs()
    assert point.fired
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        saved = recovered.lookup_call(run["run_id"], judge)
        assert saved["state"] == "reserved"
        result = swarm(recovered).resume(run["run_id"])
        assert recovered.lookup_call(run["run_id"], judge)["call_id"] == saved["call_id"]
        assert node_prompts(recovered, run["run_id"]) == expected
        assert recovered.audit(run["run_id"])["ok"]
    assert result.output == "offline: Draft"


def test_unsupported_prompt_assembly_version_is_refused_before_a_run_exists(store):
    for value in (0, 3, True, 2.0, "2", None):
        with pytest.raises(WorkflowBindingError, match="prompt_assembly"):
            swarm(store)._workflow_runtime(behavior_settings={"prompt_assembly": value})
    assert store.list_runs() == []


def test_saved_recipe_with_an_unsupported_prompt_assembly_version_is_refused(store):
    with pytest.MonkeyPatch.context() as newer:
        # A newer release records a version this one does not know.
        newer.setattr(workflow, "_BEHAVIOR_VERSIONS", MappingProxyType({"prompt_assembly": (1, 2, 3)}))
        runtime = swarm(store)._workflow_runtime(behavior_settings={"prompt_assembly": 3})
        run_id = asyncio.run(runtime.plan(Task("Report"))).run_ref["run_id"]
    calls = store.inspect_run(run_id)["call_count"]
    with pytest.raises(WorkflowConflictError, match="prompt_assembly"):
        swarm(store).resume(run_id)
    assert store.inspect_run(run_id)["call_count"] == calls


def test_checkpoint_save_cost_does_not_grow_with_journal_history(tmp_path, monkeypatch):
    saves = []
    original = WorkflowRuntime._save

    def counted(self, *args, **kwargs):
        statements = []
        self.store._db.set_trace_callback(statements.append)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.store._db.set_trace_callback(None)
            saves.append(len(statements))

    monkeypatch.setattr(WorkflowRuntime, "_save", counted)

    def resumed_saves(prior):
        """SQL statements per save while resuming a run with ``prior`` finished nodes."""
        ids = [f"prior-{index}" for index in range(prior)]
        nodes = [Node(id=node, label=node, depends_on=ids[index - 1:index]) for index, node in enumerate(ids)]
        nodes.append(Node(id="last", label="Last", depends_on=ids[-1:]))
        with SQLiteWorkflowStore(tmp_path / f"history-{prior}.db") as store:
            runtime = swarm(store)
            prepare = store.prepare_call

            def stop_before_last(lease, key, **kwargs):
                if key.scope_id == "node/last":
                    raise Crash("Simulated process loss")
                return prepare(lease, key, **kwargs)

            monkeypatch.setattr(store, "prepare_call", stop_before_last)
            with pytest.raises(Crash):
                runtime.execute(ExecutionGraph([Topology.SERIAL], nodes))
            monkeypatch.setattr(store, "prepare_call", prepare)
            (run,) = store.list_runs()
            history = store.inspect_run(run["run_id"])
            assert history["call_count"] == prior and history["revision"] >= prior
            saves.clear()
            assert runtime.resume(run["run_id"]).cost_is_complete
            return list(saves)

    short, long = resumed_saves(1), resumed_saves(8)
    # Opening checkpoint, the last node's checkpoint and the completed one.
    assert len(short) == len(long) == 3
    assert short == long
class _TickingClock(ManualClock):
    """A journal clock that moves forward one millisecond on every reading."""

    def __call__(self):
        self.now_ns += 1_000_000
        return self.now_ns


def test_durable_timing_comes_from_journal_wall_clock_within_one_call(tmp_path):
    clock = _TickingClock()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        result = swarm(store).execute(Task("Report"))

    timing = result.timing
    assert timing["clock"] == "wall"
    assert (timing["resumed"], timing["latency_complete"]) == (False, True)
    [segment] = timing["segments"]
    assert segment["interrupted"] is False
    assert segment["started_at"] < segment["ended_at"]
    assert timing["elapsed_ms"] == segment["elapsed_ms"] == round(
        (segment["ended_at"] - segment["started_at"]) * 1000, 1)
    assert timing["elapsed_ms"] < timing["run_wall_ms"] < timing["elapsed_ms"] + 5


def test_durable_timing_keeps_time_between_calls_out_of_each_call(tmp_path):
    clock = ManualClock()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        planned = swarm(store).plan(Task("Report"))
        clock.advance(3600)
        result = swarm(store).execute(planned)
        clock.advance(60)
        cached = swarm(store).resume(result.execution_id)

    assert result.timing == {
        "clock": "wall", "elapsed_ms": 0.0, "run_wall_ms": 3_600_000.0,
        "latency_complete": False, "resumed": True,
        "segments": [
            {"started_at": 1000.0, "ended_at": 1000.0, "elapsed_ms": 0.0, "interrupted": False},
            {"started_at": 4600.0, "ended_at": 4600.0, "elapsed_ms": 0.0, "interrupted": False},
        ],
    }
    assert cached.timing["run_wall_ms"] == 3_660_000.0
    assert [segment["started_at"] for segment in cached.timing["segments"]] == [1000.0, 4600.0, 4660.0]


def test_durable_timing_marks_a_session_that_never_released_its_lease(tmp_path):
    clock = ManualClock()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        planned = swarm(store).plan(Task("Report"))
        run_id = planned.run_ref["run_id"]
        clock.advance(10)
        # A process takes the lease and is killed before it releases it.
        store.acquire_lease(run_id, "lost-owner", ttl_s=30)
        clock.advance(40)
        result = swarm(store).execute(planned)

    planning, lost, current = result.timing["segments"]
    assert [planning["interrupted"], lost["interrupted"], current["interrupted"]] == [False, True, False]
    # A lost session ends at its last journal event, here its lease acquisition.
    assert (lost["started_at"], lost["ended_at"]) == (1010.0, 1010.0)
    assert current["started_at"] == 1050.0
    assert result.timing["run_wall_ms"] == 50_000.0


# ---------------------------------------------------------------------------
# Routing decision records
# ---------------------------------------------------------------------------


class TemplatesArchitect:
    """A local architect that records the template it planned from under its own key."""

    def plan(self, task):
        graph = ExecutionGraph([Topology.SERIAL], [Node(id="write", label="Write")])
        graph.planning = {"templates": [{"name": "report", "version": 1}]}
        return graph, Registry()


def routed(store, autonomous=None):
    """A durable Swarm whose offline classifier's echoed reply names no tier."""
    router = WhiteRabbit(classifier_provider=OfflineProvider(), classifier_model="offline",
                         autonomous=autonomous or SimpleArchitect())
    return Swarm(model="offline", provider=OfflineProvider(), router=router, run_store=store,
                 max_budget_usd=1)


def classifier_calls(monkeypatch):
    original, calls = OfflineProvider.complete, []

    async def complete(self, system, prompt, model):
        if system.startswith("You are a task classifier."):
            calls.append(prompt)
        return await original(self, system, prompt, model)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return calls


def routing_call_ids(store, run_id):
    return [call["call_id"] for call in store.inspect_run(run_id)["calls"]
            if call["key"]["phase"] == "routing"]


def test_the_planning_operation_carries_the_routing_decision(tmp_path, monkeypatch):
    from smythe.router import RoutingDecision

    NativeTransport().install(monkeypatch)
    provider = OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)
    router = WhiteRabbit(classifier_provider=provider, classifier_model="gpt-6-astra",
                         autonomous=SimpleArchitect())
    with SQLiteWorkflowStore(tmp_path / "routed.db") as store:
        result = Swarm(model="gpt-6-astra", provider=provider, max_budget_usd=1, run_store=store,
                       router=router).execute(Task("Write and check"))
        run_id = result.execution_id
        operation = store.load_operation(run_id, "planning")
        checkpoint = store.get_checkpoint(run_id)["checkpoint"]
        [call_id] = routing_call_ids(store, run_id)

    routing = operation["result"]["routing"]
    assert routing == result.graph.planning["routing"] == checkpoint["graph"]["planning"]["routing"]
    assert operation["result"]["graph"]["planning"] == {"routing": routing}
    assert (routing["tier"], routing["method"], routing["matched_token"]) == (
        "autonomous", "classifier", "autonomous")
    # The journal holds the classifier's charge; the record names its call.
    assert routing["usage"] is None and routing["call_ids"] == [call_id]
    assert RoutingDecision.from_dict(routing).call_ids == (call_id,)
    assert all("routing" not in node["metadata"] for node in checkpoint["graph"]["nodes"])


def test_replay_after_a_crash_restores_the_routing_decision_without_a_classifier_call(
    store, monkeypatch,
):
    runtime = routed(store, LocalOnly(TemplatesArchitect, "templates-architect", "1"))
    restore = crash_checkpoints(monkeypatch, store, lambda state, _: state["status"] == "planned")
    with pytest.raises(Crash):
        runtime.plan(Task("Report"))
    restore()
    (run,) = store.list_runs()
    run_id = run["run_id"]
    result = store.load_operation(run_id, "planning")["result"]
    recorded = result["graph"]["planning"]
    assert set(recorded) == {"templates", "routing"}
    assert result["routing"] == recorded["routing"]
    assert recorded["routing"]["method"] == "fallback_unmatched"
    assert recorded["routing"]["call_ids"] == routing_call_ids(store, run_id)
    calls = classifier_calls(monkeypatch)

    resumed = runtime.resume(run_id)

    assert calls == []
    assert resumed.graph.planning == recorded
    assert store.get_checkpoint(run_id)["checkpoint"]["graph"]["planning"] == recorded
    assert len(routing_call_ids(store, run_id)) == 1
    assert runtime.resume(run_id).graph.planning == recorded


def test_a_planning_operation_saved_without_routing_restores_no_record(tmp_path, monkeypatch):
    from smythe.workflow_store import _canonical, _sha

    path = tmp_path / "legacy.db"
    with SQLiteWorkflowStore(path) as store:
        runtime = routed(store)
        restore = crash_checkpoints(monkeypatch, store, lambda state, _: state["status"] == "planned")
        with pytest.raises(Crash):
            runtime.plan(Task("Report"))
        restore()
        (run,) = store.list_runs()
    run_id = run["run_id"]
    # Rewrite the operation as a release that recorded no routing saved it.
    with closing(sqlite3.connect(path)) as db, db:
        text = db.execute("SELECT result_json FROM workflow_operations WHERE run_id=? AND "
                          "operation_key='planning'", (run_id,)).fetchone()[0]
        result = json.loads(text)
        del result["routing"], result["graph"]["planning"]
        text = _canonical(result)
        db.execute("UPDATE workflow_operations SET result_json=?, result_sha=? WHERE run_id=? "
                   "AND operation_key='planning'", (text, _sha(text), run_id))
    calls = classifier_calls(monkeypatch)
    with SQLiteWorkflowStore(path) as store:
        resumed = routed(store).resume(run_id)
        checkpoint = store.get_checkpoint(run_id)["checkpoint"]

    assert calls == []
    assert resumed.graph.planning is None
    assert "planning" not in checkpoint["graph"]


def test_routing_records_leave_the_recipe_and_unrouted_runs_unchanged(store):
    runtime = routed(store)._workflow_runtime()
    offline = {"adapter_version": "offline-static-v1", "config": {"echo_prefix": "offline: ", "plan": None},
               "decoder_version": "offline-static-v1", "endpoint_scope": "offline", "kind": "offline",
               "price_version": "offline-zero-v1", "supported_models": None}
    assert runtime.recipe["components"]["router"] == {
        "type": "white_rabbit", "version": 1,
        "classifier": {"provider": offline, "model": "offline"},
        "deterministic": {}, "constrained": {"type": "none"},
        "autonomous": {"type": "simple_architect", "version": 1},
    }
    routed_run = routed(store).execute(Task("Report")).execution_id
    assert store.load_run(routed_run)["config"] == runtime.recipe

    plain = swarm(store).execute(Task("Report"))
    operation = store.load_operation(plain.execution_id, "planning")
    assert set(operation["result"]) == {"graph", "registry"}
    assert "planning" not in operation["result"]["graph"]
    assert plain.graph.planning is None
    assert "planning" not in store.get_checkpoint(plain.execution_id)["checkpoint"]["graph"]
