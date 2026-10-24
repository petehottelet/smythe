"""Template versions put in effect through the policy registry, and pinned in running work.

A target ``planner.template.<name>`` holds the template version in effect.
An ordinary ``ConstrainedArchitect`` takes the active versions when it is
built and keeps that menu. A durable run's recipe pins the versions it
started with, so a rollback changes only new runs: an interrupted run
resumes on its pinned versions, rebuilt from the catalog, and refuses the
current ones. A plan composed from records names their versions in
``graph.planning["templates"]``, never in node metadata.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
from copy import deepcopy
import json
import sqlite3

import pytest

import smythe
from fakes import Crash, crash_when
from smythe._canonical import sha256_hex
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.constrained_prompts import CONSTRAINED_SYSTEM_PROMPT
from smythe.graph import Node
from smythe.optimize import (
    Candidate,
    ExperimentContract,
    ExperimentLedger,
    MetricObjective,
    MutableFieldRule,
    OptimizationRunner,
    PolicyRegistry,
    TrialContext,
    TrialOutcome,
)
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.swarm import Swarm
from smythe.task import Task
from smythe.templates import (
    TemplateCatalog, TemplateRecord, TemplateRecordError, pinned_templates, template_target,
)
from smythe.workflow_binding import ComponentBinding, LocalOnly, bind_component, local_component
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowConflictError

TARGET = "planner.template.report"
CAMPAIGN = "report-template"
SECTIONS = ("outline", "draft", "conclusions")


def _section(node_id: str, depends_on: tuple[str, ...]) -> dict:
    return {"id": node_id, "label": f"Write the {node_id} of a report on {{goal}}",
            "depends_on": list(depends_on), "capabilities": [], "profile": None,
            "verifies": None, "max_regenerations": 0, "failure_policy": "halt",
            "max_retries": 0, "timeout_s": None, "attach_dep_artifacts": False, "metadata": {}}


def report(sections: tuple[str, ...], *, name: str = "report", parent: str | None = None,
           description: str = "A sectioned report") -> TemplateRecord:
    """A serial template that writes each section after the one before it."""
    blueprint = [_section(section, sections[index - 1:index]) for index, section in enumerate(sections)]
    return TemplateRecord(name=name, description=description, blueprint=blueprint,
                          parent_version=parent)


V1 = report(SECTIONS)
# Harmful: it drops the conclusions every report needs. Nothing it declares
# requires that section, so validation against v1 accepts it.
V2 = report(SECTIONS[:2], parent=V1.template_version)
SUMMARY = report(("notes",), name="summary", description="A short summary")
V1_IDS = ["report-0-outline", "report-0-draft", "report-0-conclusions"]
V2_IDS = V1_IDS[:2]


def ids(graph) -> list[str]:
    return [node.id for node in graph.nodes]


def used(version: str, *, index: int = 0, name: str = "report") -> dict:
    return {"name": name, "index": index, "version": version}


class Selects(Provider):
    """A template-selection model that picks the named templates in order."""

    def __init__(self, *names: str) -> None:
        self.names = names

    async def complete(self, system, prompt, model):
        return CompletionResult(json.dumps([{"template": name} for name in self.names]))


@pytest.fixture
def offline(monkeypatch):
    """Offline durable calls: planning selects ``report`` and each section is written."""
    systems = []

    async def complete(self, system, prompt, model):
        systems.append(system)
        if system == CONSTRAINED_SYSTEM_PROMPT:
            return CompletionResult(json.dumps([{"template": "report"}]), cost_usd=0)
        return CompletionResult("Section written", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return systems


def durable(store, templates) -> Swarm:
    provider = OfflineProvider()
    return Swarm(model="offline", provider=provider, run_store=store,
                 architect=ConstrainedArchitect(provider, templates, model="offline"))


def pinned_versions(config) -> list[str]:
    return [entry["builder"]["version"]
            for entry in config["components"]["architect"]["templates"]]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """A catalog of both report versions and a summary; tests only read it."""
    catalog = TemplateCatalog(tmp_path_factory.mktemp("catalog"))
    for record in (V1, V2, SUMMARY):  # v2 is compared with v1, its stored parent.
        assert catalog.put(record) == record.template_version
    return catalog


# ---------------------------------------------------------------------------
# A harmful version is promoted, activated and rolled back
# ---------------------------------------------------------------------------


def _campaign(path, catalog) -> None:
    """A campaign whose fooled evaluator rewards fewer sections and never looks for conclusions."""
    contract = ExperimentContract(
        name="report_template",
        objectives=(MetricObjective("quality", "maximize", primary=True),),
        mutable_fields=(TARGET,),
        mutable_field_rules={TARGET: MutableFieldRule(
            "string", choices=(V1.template_version, V2.template_version))},
        required_gates=("valid",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=2,
        max_parallel_candidates=1,
        max_trials=14,
        max_wall_seconds=600,
        max_budget_microusd=0,
        per_trial_reservation_microusd=0,
        confidence=0.8,
        min_improvement=0.1,
        base_seed=23,
    )

    async def shorter_scores_higher(context: TrialContext) -> TrialOutcome:
        record = catalog.get(context.candidate.policy[TARGET])
        nodes, _ = record.to_template().builder(Task("Write the quarterly report"))
        return TrialOutcome(metrics={"quality": 10.0 - len(nodes)}, gates={"valid": True},
                            actual_cost_microusd=0)

    incumbent = Candidate(contract, {TARGET: V1.template_version}, "three sections")
    challenger = Candidate(contract, {TARGET: V2.template_version}, "two sections read faster",
                           parent=incumbent)
    with ExperimentLedger(path, durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, shorter_scores_higher,
                                    evaluator_hash="sha256:" + "e" * 64,
                                    bootstrap_resamples=10, campaign_id=CAMPAIGN)
        result = asyncio.run(runner.run(incumbent, (challenger,)))
    assert result.promoted and result.selected_candidate_id == challenger.candidate_id


def _ledger(path):
    """The ledger's bytes and SQL dump, and its decisions and trials, all read without writing."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as raw:
        dump = list(raw.iterdump())
    with ExperimentLedger(path, read_only=True) as ledger:
        decisions = ledger.list_decisions(CAMPAIGN)
        trials = [trial.to_dict() for trial in ledger.list_trials(CAMPAIGN)]
    assert len(decisions) == 1 and decisions[0]["promoted"] is True and trials
    return path.read_bytes(), dump, decisions, trials


def test_a_rolled_back_version_leaves_running_work_and_earlier_architects_on_theirs(
    tmp_path, catalog, offline, monkeypatch,
):
    ledger_path, registry_path = tmp_path / "optimize.sqlite3", tmp_path / "registry.sqlite3"
    _campaign(ledger_path, catalog)
    with PolicyRegistry(registry_path) as registry, SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        review = registry.record_review(ledger_path, CAMPAIGN, reviewer="report-owner",
                                        verdict="approve", note="higher quality score")
        registry.activate(TARGET, review.review_id, expected_seq=0, actor="release-operator",
                          reason="approved for new report runs")
        assert registry.resolve(TARGET).policy == {TARGET: V2.template_version}

        # Built while v2 is in effect, an ordinary architect offers v2.
        earlier = ConstrainedArchitect(Selects("report"), catalog.active_templates(registry, ["report"]))
        graph, _ = earlier.plan(Task("Summarize the quarter"))
        assert ids(graph) == V2_IDS
        assert graph.planning == {"templates": [used(V2.template_version)]}

        # A durable run started with v2 is interrupted after its planning call,
        # before the plan is saved.
        point = crash_when(monkeypatch, store, "complete_operation",
                           lambda lease, key, *args, **kwargs: key == "planning")
        with pytest.raises(Crash):
            durable(store, catalog.active_local_templates(registry, ["report"])).execute(
                Task("Summarize the quarter"))
        point.restore()
        (run,) = store.list_runs()
        run_id = run["run_id"]
        saved = store.load_run(run_id)
        assert pinned_versions(saved["config"]) == [V2.template_version]
        assert store.get_checkpoint(run_id) is None

        ledger_before = _ledger(ledger_path)
        rolled_back = registry.rollback(TARGET, to_seq=1, expected_seq=2, actor="release-operator",
                                        reason="v2 drops the conclusions")
        # Rollback writes only the registry: never the ledger or running work.
        assert _ledger(ledger_path) == ledger_before
        assert store.load_run(run_id) == saved
        assert registry.resolve(TARGET).policy == {TARGET: V1.template_version}

        # New ordinary and durable runs resolve v1.
        current = ConstrainedArchitect(Selects("report"), catalog.active_templates(registry, ["report"]))
        graph, _ = current.plan(Task("Summarize the quarter"))
        assert ids(graph) == V1_IDS
        assert graph.planning == {"templates": [used(V1.template_version)]}
        fresh = durable(store, catalog.active_local_templates(registry, ["report"])).execute(
            Task("Review the annual budget"))
        assert ids(fresh.graph) == V1_IDS
        assert fresh.graph.planning == {"templates": [used(V1.template_version)]}
        assert pinned_versions(store.load_run(fresh.execution_id)["config"]) == [V1.template_version]
        # The earlier architect keeps the menu it was built with.
        graph, _ = earlier.plan(Task("Summarize the quarter"))
        assert ids(graph) == V2_IDS

        # Resuming with the version now in effect is refused, and changes nothing.
        with pytest.raises(WorkflowConflictError, match="saved recipe"):
            durable(store, catalog.active_local_templates(registry, ["report"])).resume(run_id)
        assert store.load_run(run_id) == saved

        # The pinned versions, rebuilt from the catalog, finish the run on v2.
        planning_calls = offline.count(CONSTRAINED_SYSTEM_PROMPT)
        resumed = durable(store, pinned_templates(saved["config"], catalog)).resume(run_id)
        assert ids(resumed.graph) == V2_IDS
        assert resumed.output == "Section written"
        assert resumed.graph.planning == {"templates": [used(V2.template_version)]}
        assert offline.count(CONSTRAINED_SYSTEM_PROMPT) == planning_calls  # replayed, not asked again
        finished = store.load_run(run_id)
        assert (finished["status"], finished["config_sha256"]) == ("completed", saved["config_sha256"])
        assert store.get_checkpoint(run_id)["checkpoint"]["graph"]["planning"] == {
            "templates": [used(V2.template_version)]}
        assert not any("template" in key for node in resumed.graph.nodes for key in node.metadata)

        history = registry.history(TARGET)
    assert _ledger(ledger_path) == ledger_before
    with PolicyRegistry(registry_path, read_only=True) as reader:
        assert reader.history(TARGET) == history
    assert [(event.seq, event.action) for event in history] == [
        (1, "baseline"), (2, "activate"), (3, "rollback")]
    assert [event.policy for event in history] == [
        {TARGET: V1.template_version}, {TARGET: V2.template_version}, {TARGET: V1.template_version}]
    assert (history[1].review_id, history[2].restores_seq, history[2]) == (
        review.review_id, 1, rolled_back)


# ---------------------------------------------------------------------------
# Active templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["report", "Quarterly_report-2", "r" * 64])
def test_a_template_target_is_its_name_under_planner_template(name):
    assert template_target(name) == f"planner.template.{name}"


@pytest.mark.parametrize("name", ["", "2report", "report.summary", "quarterly report", "r" * 65,
                                  "_report", None, 7])
def test_a_template_without_a_target_name_is_refused(name):
    with pytest.raises(ValueError, match="one dotted segment"):
        template_target(name)


def _in_effect(path, policies: dict[str, dict]) -> PolicyRegistry:
    """A registry where each target's baseline puts its policy in effect."""
    registry = PolicyRegistry(path)
    for target, policy in policies.items():
        registry._append(target, [("baseline", policy, None, None)], expected_seq=0,
                         actor="operator", reason="first version")
    return registry


def test_active_templates_build_the_version_in_effect_for_each_name(tmp_path, catalog):
    policies = {TARGET: {TARGET: V2.template_version},
                "planner.template.summary": {"planner.template.summary": SUMMARY.template_version}}
    with _in_effect(tmp_path / "registry.sqlite3", policies) as registry:
        ordinary = catalog.active_templates(registry, ("summary", "report"))
        local = catalog.active_local_templates(registry, ["summary", "report"])
        assert catalog.active_templates(registry, []) == []

    assert [(item.name, item.description) for item in ordinary] == [
        ("summary", "A short summary"), ("report", "A sectioned report")]
    assert [(item.name, item.description) for item in local] == [
        (item.name, item.description) for item in ordinary]
    assert all(type(item) is SubGraphTemplate for item in ordinary + local)
    assert [item.builder for item in local] == [SUMMARY.local_only(), V2.local_only()]
    assert [item.builder.template_version for item in ordinary] == [
        SUMMARY.template_version, V2.template_version]
    nodes, _ = ordinary[1].builder(Task("Summarize the quarter"))
    assert [node.id for node in nodes] == ["outline", "draft"]


def test_a_retired_template_is_left_out_and_an_unknown_one_is_refused(tmp_path, catalog):
    with _in_effect(tmp_path / "registry.sqlite3", {TARGET: {TARGET: V1.template_version}}) as registry:
        registry.retire(TARGET, expected_seq=1, actor="operator", reason="no longer offered")
        assert catalog.active_templates(registry, ["report"]) == []
        assert catalog.active_local_templates(registry, ["report"]) == []
        for active in (catalog.active_templates, catalog.active_local_templates):
            with pytest.raises(KeyError, match="'summary' has no version activated"):
                active(registry, ["report", "summary"])
            with pytest.raises(TypeError, match="not one string"):
                active(registry, "report")
            with pytest.raises(ValueError, match="one dotted segment"):
                active(registry, ["report.summary"])


@pytest.mark.parametrize("policy, error, message", [
    ({TARGET: V1.template_version, TARGET + ".max_sections": 3}, TemplateRecordError,
     "must hold only its template version"),
    ({TARGET: 3}, TemplateRecordError, "must hold only its template version"),
    ({TARGET: "v1"}, TemplateRecordError, "64 lowercase hexadecimal"),
    ({TARGET: sha256_hex("never stored")}, KeyError, "does not hold version"),
    ({TARGET: SUMMARY.template_version}, TemplateRecordError, "is named 'summary', not 'report'"),
], ids=["another-path", "not-text", "not-a-version", "not-in-catalog", "another-template"])
def test_a_policy_that_is_not_a_stored_version_of_the_template_is_refused(
    tmp_path, catalog, policy, error, message,
):
    with _in_effect(tmp_path / "registry.sqlite3", {TARGET: policy}) as registry:
        for active in (catalog.active_templates, catalog.active_local_templates):
            with pytest.raises(error, match=message):
                active(registry, ["report"])


# ---------------------------------------------------------------------------
# Pinned templates
# ---------------------------------------------------------------------------


def _write_notes(task, **params):
    return [Node(id="notes", label=f"Take notes on {task.goal}")], Registry()


def _recipe(store, templates) -> dict:
    return durable(store, templates)._workflow_runtime().recipe


@pytest.fixture(scope="module")
def saved_recipe(tmp_path_factory):
    """A recipe whose menu holds v1 of the report and a template written in code."""
    notes = SubGraphTemplate("notes", "Notes", LocalOnly(lambda: _write_notes, "notes", "1",
                                                         role="template_builder"))
    with SQLiteWorkflowStore(tmp_path_factory.mktemp("recipe") / "runs.db") as store:
        return _recipe(store, [SubGraphTemplate("report", "A sectioned report", V1.local_only()),
                               notes])


def test_pinned_templates_rebuild_the_saved_menu_exactly(tmp_path, catalog):
    # Names and descriptions come from the recipe, not the records.
    menu = [SubGraphTemplate("long-report", "Every section", V1.local_only()),
            SubGraphTemplate("short-report", "Outline and draft", V2.local_only())]
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        recipe = _recipe(store, menu)
        pinned = pinned_templates(recipe, catalog)
        assert [(item.name, item.description, item.builder) for item in pinned] == [
            (item.name, item.description, item.builder) for item in menu]
        assert _recipe(store, pinned) == recipe
        # An architect's own description, such as a router tier's, works too.
        architect = recipe["components"]["architect"]
        assert [item.builder for item in pinned_templates(architect, catalog)] == [
            V1.local_only(), V2.local_only()]
        router = Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                       router=WhiteRabbit(constrained=ConstrainedArchitect(
                           OfflineProvider(), menu, model="offline")))
        tier = router._workflow_runtime().recipe["components"]["router"]["constrained"]
        assert [item.builder for item in pinned_templates(tier, catalog)] == [
            V1.local_only(), V2.local_only()]


def _builder(**changes):
    """A change to the first template's saved builder."""
    def change(recipe):
        recipe["components"]["architect"]["templates"][0]["builder"].update(changes)
    return change


@pytest.mark.parametrize("change, error, message", [
    (lambda recipe: recipe["components"].update(
        architect={"type": "simple_architect", "version": 1}), TemplateRecordError,
     "not a ConstrainedArchitect"),
    (lambda recipe: recipe.update(components={}), TemplateRecordError, "not a ConstrainedArchitect"),
    (lambda recipe: None, TemplateRecordError,
     "'notes' in the recipe was not built from a template record"),
    (_builder(identity="template:summary"), TemplateRecordError, "is named 'report', not 'summary'"),
    (_builder(version=sha256_hex("never stored")), KeyError, "does not hold version"),
    (_builder(version="1"), TemplateRecordError, "64 lowercase hexadecimal"),
    (_builder(role="architect"), TemplateRecordError, "was not built from a template record"),
], ids=["simple-architect", "no-architect", "developer-builder", "renamed-record",
        "not-in-catalog", "not-a-version", "another-role"])
def test_pinned_templates_refuse_a_menu_they_cannot_rebuild(saved_recipe, catalog, change, error,
                                                            message):
    recipe = deepcopy(saved_recipe)
    change(recipe)
    with pytest.raises(error, match=message):
        pinned_templates(recipe, catalog)


def test_pinned_templates_refuse_a_recipe_without_a_template_menu(catalog):
    for config in (None, [], {"type": "constrained_architect", "templates": "report"},
                   {"type": "constrained_architect", "templates": [{"name": "report"}]}):
        with pytest.raises(TemplateRecordError):
            pinned_templates(config, catalog)


# ---------------------------------------------------------------------------
# The planning record
# ---------------------------------------------------------------------------


def test_a_plan_records_each_record_template_it_composed_on_the_graph_only():
    menu = [SubGraphTemplate("notes", "Notes", _write_notes), V1.to_template()]
    architect = ConstrainedArchitect(Selects("notes", "report", "notes", "report"), menu)
    graph, _ = architect.plan(Task("Summarize the quarter"))

    assert graph.planning == {"templates": [used(V1.template_version, index=1),
                                            used(V1.template_version, index=3)]}
    assert all(node.metadata == {} for node in graph.nodes)
    # Templates written in code record nothing, as before.
    graph, _ = ConstrainedArchitect(Selects("notes"), menu).plan(Task("Summarize the quarter"))
    assert graph.planning is None


def test_a_routed_plan_keeps_its_template_record_beside_the_routing_decision():
    swarm = Swarm(provider=Selects("report"), model="test-model", artifact_dir=None,
                  router=WhiteRabbit(autonomous=ConstrainedArchitect(Selects("report"),
                                                                     [V2.to_template()])))
    graph = swarm.plan(Task("Summarize the quarter"))

    assert graph.planning["templates"] == [used(V2.template_version)]
    assert graph.planning["routing"]["method"] == "fallback_no_classifier"


def test_a_durable_run_binds_a_record_builder_that_local_component_returns():
    binding = ComponentBinding("run-1", "architect", "planning",
                               call_factory=lambda provider, scope: provider)
    bound = bind_component(V1.local_only(), binding.child("template:report"),
                           role="template_builder")

    builder = local_component(bound)

    assert builder is not bound and builder.template_version == V1.template_version
    assert [node.id for node in builder(Task("Summarize the quarter"))[0]] == list(SECTIONS)
    # Anything else is returned unchanged.
    plain = V1.to_template().builder
    assert local_component(plain) is plain and local_component(None) is None
    # A durable menu of record builders records their versions, as an ordinary one does.
    durable = ConstrainedArchitect(OfflineProvider(), [SubGraphTemplate(
        "report", "A sectioned report", V1.local_only())], model="offline").bind_run(binding)
    assert durable._compose([{"template": "report"}], Task("Summarize the quarter"))[
        0].planning == {"templates": [used(V1.template_version)]}


def test_public_names_are_exported():
    for name in ("pinned_templates", "template_target"):
        assert name in smythe.__all__ and getattr(smythe, name) is getattr(smythe.templates, name)
