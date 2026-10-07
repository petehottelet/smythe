"""Repair plans: what an archive of durable runs calls for, planned without changing anything.

The research runs here come from ``test_workflow_lineage``: each reads
versioned sources from the fake external service over loopback HTTP, extracts
from each with a model node given a context snapshot, and publishes a brief
written by local code. A source catalog describes the service's sources from
its state, which counts no read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

import test_workflow_reuse as successors
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptancePolicy, ContextSource, CurrentState, LineagePolicy, OfflineProvider, ReadToolError,
    RepairPlan, SimpleArchitect, SourceStatus, SQLiteWorkflowStore, Swarm, Task, TextValidator,
    archive_heads,
)
from smythe._canonical import sha256_hex
from smythe._run_inspection import inspect_journal
from smythe.graph import ExecutionGraph
from smythe.lineage import load_lineage
from smythe.reuse import STATUSES, derived_status, evaluate
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import WorkflowNotFoundError, _canonical, _sha
from test_workflow_lineage import (
    QUESTION, TEXTS, profile, research, research_graph, serial, strict,
)

SCOPE = "research-team"
REVISED = "In 2026, monthly billing led renewals."
# The research publishes with MarkdownRenderer, as a record names it.
MARKDOWN = {"name": "markdown_renderer", "version": "1",
            "spec_sha256": sha256_hex({"type": "markdown_renderer", "version": 1})}
ZERO = {"nodes_reused": {}, "nodes_regenerated": {}, "read_calls": 0, "model_steps": 0,
        "publications": 0, "acceptance_evaluations": 0, "new_runs": 0,
        "may_incur_charges": False, "allowance_required": False}


class ServiceCatalog:
    """Describes each source from the fake service's state, which counts no read.

    *overrides* replace what it says about some sources, as if they changed.
    """

    def __init__(self, service: FixtureService, overrides=None) -> None:
        self.service, self.overrides, self.calls = service, dict(overrides or {}), []

    def describe(self, source_id):
        self.calls.append(source_id)
        if source_id in self.overrides:
            return self.overrides[source_id]
        item = self.service.snapshot()["sources"].get(source_id)
        return None if item is None else SourceStatus(version=str(item["version"]),
                                                      sha256=item["sha256"])


def revised(service: FixtureService) -> ServiceCatalog:
    """The renewals source at a second version with other text."""
    return ServiceCatalog(service, {"renewals": SourceStatus(
        version="2", sha256=hashlib.sha256(REVISED.encode()).hexdigest())})


@dataclass
class Archive:
    path: Path
    library: tuple
    root: Path
    runs: dict[str, str]

    @property
    def service(self) -> FixtureService:
        return self.library[0]

    def copy(self, directory: Path) -> Path:
        """A copy of the journal; the published files stay where the runs put them."""
        return Path(shutil.copyfile(self.path, directory / "runs.db"))

    def swarm(self, store, **options) -> Swarm:
        return research(store, self.library, self.root, **options)


def survey_only(library) -> ExecutionGraph:
    """The survey's read step and its extraction, without the renewals branch."""
    graph = research_graph(library)
    graph.nodes = [node for node in graph.nodes if node.id.endswith("survey")]
    return graph


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Three completed research runs, built once; the second one's published file is changed."""
    root = tmp_path_factory.mktemp("archive")
    service = FixtureService()
    records = {source_id: service.add_source(source_id, text) for source_id, text in TEXTS.items()}
    survey = records["survey"]
    with serve_http(service) as endpoint:
        library = (service, records, endpoint)
        with SQLiteWorkflowStore(root / "runs.db") as store:
            runs = {"full": research(store, library, root).execute(research_graph(library)),
                    "damaged": research(store, library, root).execute(research_graph(library)),
                    "survey": research(store, library, root, context_sources=[ContextSource(
                        survey.source_id, survey.content, str(survey.version))]).execute(
                        survey_only(library))}
            assert all(result.accepted for result in runs.values())
            published = current_artifact(store, runs["damaged"].execution_id).publication.path
        data = bytearray(published.read_bytes())
        data[-2] ^= 0x01
        published.write_bytes(bytes(data))
        yield Archive(root / "runs.db", library, root,
                      {name: result.execution_id for name, result in runs.items()})


def expected_state(store, run_id, catalog) -> CurrentState:
    """The current state of *run_id*, built from what the research configuration declares."""
    saved = store.get_checkpoint(run_id)["checkpoint"]
    agents = {sha256_hex({key: value for key, value in entry.items() if key != "id"})
              for entry in saved["registry"]["agents"]}
    return CurrentState(
        catalog=catalog, validators=strict().validator_identities,
        policy_sha256=strict().sha256,
        tools={"library": {"version": "1", "decoder_version": "fixture-source-v1"}},
        components={"brief-writer": "1"}, context_assembly_version=1, agents=agents,
        integrity={run_id: current_artifact(store, run_id).integrity["status"]},
        lineage=LineagePolicy(access_scope=SCOPE), renderer=MARKDOWN)


def items(plan: RepairPlan) -> dict[str, dict]:
    return {item["run_id"]: item for item in plan.to_dict()["items"]}


def rewrite_lineage(store, run_id: str, change) -> None:
    """Change the run's applied lineage record in place, as another release might have written it."""
    (operation,) = [item for item in store.list_operations(run_id, "lineage")
                    if item["applied_revision"] is not None]
    data = operation["result"]
    change(data["lineage"])
    stored = _canonical(data)
    with store._db:
        store._db.execute("UPDATE workflow_operations SET result_json=?, result_sha=? "
                          "WHERE operation_id=?", (stored, _sha(stored), operation["operation_id"]))


def link(store, successor: str, predecessor: str) -> None:
    """Name *predecessor* in *successor*'s lineage, as a repair's new run records it."""
    reference = {"store_id": store.store_id, "run_id": predecessor,
                 "lineage_sha256": load_lineage(store, predecessor).sha256}
    rewrite_lineage(store, successor, lambda record: record.update(predecessor=reference))


# ---------------------------------------------------------------------------
# What a plan holds
# ---------------------------------------------------------------------------


def test_a_plan_over_three_runs_equals_the_evaluator_decisions(built, tmp_path):
    runs = built.runs
    with SQLiteWorkflowStore(built.copy(tmp_path), read_only=True) as store:
        for catalog, expected in (
                (lambda: ServiceCatalog(built.service),
                 {"full": "reuse", "damaged": "rerender", "survey": "reuse"}),
                (lambda: revised(built.service),
                 {"full": "regenerate", "damaged": "regenerate", "survey": "reuse"})):
            listed = items(built.swarm(store).plan_repair(sources=catalog()))
            # Every head of the archive, in journal order.
            assert list(listed) == list(runs.values())
            assert {name: listed[run_id]["action"] for name, run_id in runs.items()} == expected
            for run_id, item in listed.items():
                decision = evaluate(load_lineage(store, run_id),
                                    expected_state(store, run_id, catalog()))
                assert {key: item[key] for key in ("action", "reasons", "nodes", "authority")} \
                    == decision.to_dict()
                assert item["lineage_sha256"] == load_lineage(store, run_id).sha256
    full = listed[runs["full"]]
    # Only the renewals read, the extractions given it and the brief run again.
    assert {node["id"]: node["action"] for node in full["nodes"]} == {
        "read-survey": "reuse", "extract-survey": "regenerate", "read-renewals": "regenerate",
        "extract-renewals": "regenerate", "synthesis": "regenerate"}
    # A changed premise is never fixed by rendering again.
    assert "artifact_changed" in listed[runs["damaged"]]["reasons"]


def test_each_item_predicts_its_work_and_derived_status(built, tmp_path):
    runs, service = built.runs, built.service
    with SQLiteWorkflowStore(built.copy(tmp_path), read_only=True) as store:
        before = items(built.swarm(store).plan_repair(sources=ServiceCatalog(service)))
        after = items(built.swarm(store).plan_repair(sources=revised(service)))
        recheck = items(built.swarm(store, acceptance=AcceptancePolicy(
            mode="strict", final_validators=[TextValidator({"min_chars": 2})])).plan_repair(
            sources=ServiceCatalog(service)))
    every = {"model": 2, "read": 2, "synthesis": 1}
    # Reusing a run keeps every node and does nothing else.
    assert before[runs["full"]]["work"] == ZERO | {"nodes_reused": every}
    assert before[runs["full"]]["status"] == "current"
    assert before[runs["survey"]]["work"] == ZERO | {
        "nodes_reused": {"model": 1, "read": 1, "synthesis": 1}}
    # Rendering again publishes the saved synthesis and decides on the new bytes; no node runs.
    assert before[runs["damaged"]]["work"] == ZERO | {
        "nodes_reused": every, "publications": 1, "acceptance_evaluations": 1}
    assert before[runs["damaged"]]["status"] == "invalid"
    # Regenerating starts a new run that keeps the reused nodes and runs the others.
    assert after[runs["full"]]["work"] == {
        "nodes_reused": {"read": 1}, "nodes_regenerated": {"model": 2, "read": 1, "synthesis": 1},
        "read_calls": 1, "model_steps": 2, "publications": 1, "acceptance_evaluations": 1,
        "new_runs": 1, "may_incur_charges": True, "allowance_required": True}
    assert after[runs["full"]]["status"] == "stale"
    assert after[runs["damaged"]]["status"] == "invalid"
    assert after[runs["survey"]]["status"] == "current"
    # A changed validator set needs only a new decision.
    assert recheck[runs["full"]]["action"] == "recheck"
    assert recheck[runs["full"]]["work"] == ZERO | {"nodes_reused": every,
                                                    "acceptance_evaluations": 1}
    assert recheck[runs["full"]]["status"] == "stale"


def test_a_withdrawn_source_restricts_the_runs_that_used_it(built, tmp_path):
    runs = built.runs
    withdrawn = SourceStatus(available=False, tombstone_reason="withdrawn by its publisher")
    with SQLiteWorkflowStore(built.copy(tmp_path), read_only=True) as store:
        plan = built.swarm(store).plan_repair(
            sources=ServiceCatalog(built.service, {"renewals": withdrawn}))
    listed = items(plan)
    assert (listed[runs["full"]]["action"], listed[runs["full"]]["status"]) == (
        "review", "restricted")
    assert listed[runs["full"]]["work"] == ZERO
    assert listed[runs["survey"]]["status"] == "current"
    assert {"source_id": "renewals", "version": None, "sha256": None, "available": False,
            "tombstoned": True} in plan.to_dict()["sources"]


def test_derived_statuses_are_computed_from_reasons():
    assert STATUSES == ("current", "stale", "invalid", "restricted", "superseded")
    assert derived_status(()) == "current"
    assert derived_status(["source_changed"]) == "stale"
    assert derived_status(["source_changed", "artifact_missing"]) == "invalid"
    assert derived_status(["decision_not_accepted"]) == "invalid"
    assert derived_status(["artifact_missing", "source_tombstoned"]) == "restricted"
    assert derived_status(["source_tombstoned"], superseded=True) == "superseded"
    assert derived_status(["run_in_progress"]) is None
    # Without a source catalog, freshness is unknown.
    assert derived_status((), sources_checked=False) is None
    assert derived_status(["artifact_changed"], sources_checked=False) == "invalid"


# ---------------------------------------------------------------------------
# The plan digest
# ---------------------------------------------------------------------------


def test_the_plan_digest_is_stable_and_changes_when_a_source_changes(built, tmp_path):
    service = built.service
    with SQLiteWorkflowStore(built.copy(tmp_path), read_only=True) as store:
        first = built.swarm(store).plan_repair(sources=ServiceCatalog(service))
        again = built.swarm(store).plan_repair(sources=ServiceCatalog(service))
        current = service.snapshot()["sources"]["survey"]
        renamed = built.swarm(store).plan_repair(sources=ServiceCatalog(
            service, {"survey": SourceStatus(version="1b", sha256=current["sha256"])}))
        changed = built.swarm(store).plan_repair(sources=revised(service))
    assert again == first and again.plan_sha256 == first.plan_sha256
    assert hash(again) == hash(first)
    data = first.to_dict()
    assert data["plan_sha256"] == first.plan_sha256 == sha256_hex(
        {key: value for key, value in data.items() if key != "plan_sha256"})
    # The payload is canonical and holds no time and no source text.
    encoded = json.dumps(data)
    assert "_ns" not in encoded and "_at" not in encoded
    assert all(text not in encoded for text in TEXTS.values())
    # Same bytes under a new version change the plan, as they change the decisions,
    # and so do new bytes.
    assert len({first.plan_sha256, renamed.plan_sha256, changed.plan_sha256}) == 3
    assert (data["version"], data["store_id"], data["access_scope"]) == (1, first.store_id, SCOPE)
    assert [item["source_id"] for item in data["sources"]] == ["renewals", "survey"]


# ---------------------------------------------------------------------------
# Planning reads only
# ---------------------------------------------------------------------------


def test_planning_a_read_only_store_changes_nothing_and_calls_nothing(
        built, tmp_path, monkeypatch):
    path = built.copy(tmp_path)
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    reads = built.service.snapshot()["sources"]

    async def forbidden(*args, **kwargs):
        pytest.fail("Planning called a provider")

    monkeypatch.setattr(OfflineProvider, "complete", forbidden)
    catalog = ServiceCatalog(built.service)
    with SQLiteWorkflowStore(path, read_only=True) as store:
        plan = built.swarm(store).plan_repair(sources=catalog)
    assert list(items(plan)) == list(built.runs.values())
    # Each source is described once for the whole archive, and none is read.
    assert sorted(catalog.calls) == ["renewals", "survey"]
    after = built.service.snapshot()["sources"]
    assert {key: item["reads"] for key, item in after.items()} == {
        key: item["reads"] for key, item in reads.items()}
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before


def test_a_run_in_progress_needs_review_and_its_lease_is_left_alone(built, tmp_path):
    lease = "SELECT owner, epoch, expires FROM workflow_runs WHERE run_id=?"
    with SQLiteWorkflowStore(built.copy(tmp_path)) as store:
        run_id = built.swarm(store).plan(Task(QUESTION)).run_ref["run_id"]
        store.acquire_lease(run_id, "another-worker", ttl_s=600)
        held = tuple(store._db.execute(lease, (run_id,)).fetchone())
        plan = built.swarm(store).plan_repair(sources=ServiceCatalog(built.service))
        assert tuple(store._db.execute(lease, (run_id,)).fetchone()) == held
    listed = items(plan)
    assert list(listed) == [*built.runs.values(), run_id]
    item = listed[run_id]
    assert (item["action"], item["reasons"], item["status"]) == ("review", ["run_in_progress"], None)
    assert (item["lineage_sha256"], item["nodes"], item["work"]) == (None, [], ZERO)


def test_a_failed_or_blocked_run_needs_review_and_is_not_called_in_progress(built, tmp_path):
    with SQLiteWorkflowStore(built.copy(tmp_path)) as store:
        swarm = built.swarm(store)
        # A read of a source the service does not have fails the run.
        graph = research_graph(built.library)
        graph.nodes = [profile(built.library[2]).step("read-missing", {"source_id": "missing"})]
        with pytest.raises(ReadToolError):
            swarm.execute(graph)
        (failed,) = [run["run_id"] for run in store.list_runs()
                     if run["run_id"] not in built.runs.values()]
        assert store.get_checkpoint(failed)["checkpoint"]["status"] == "failed"
        blocked = swarm.plan(Task(QUESTION)).run_ref["run_id"]
        store.security_stop(blocked, "credential-leak")
        waiting = swarm.plan(Task(QUESTION)).run_ref["run_id"]
        plan = swarm.plan_repair(sources=ServiceCatalog(built.service))
        inspected = {run_id: inspect_journal(store, run_id)["lineage"]
                     for run_id in (failed, blocked, waiting)}
    listed = items(plan)
    for run_id, code in ((failed, "run_failed"), (blocked, "run_failed"),
                         (waiting, "run_in_progress")):
        item = listed[run_id]
        # Each needs review and has no record, status, nodes or work.
        assert (item["action"], item["reasons"], item["status"]) == ("review", [code], None)
        assert (item["lineage_sha256"], item["nodes"], item["work"]) == (None, [], ZERO)
        assert (inspected[run_id]["reasons"], inspected[run_id]["status"]) == ([code], None)
    assert derived_status(["run_failed"]) is None


# ---------------------------------------------------------------------------
# Archive heads and scopes
# ---------------------------------------------------------------------------


def lineage_swarm(store, **options) -> Swarm:
    """A durable run under the strict policy, in the research scope unless *options* say otherwise."""
    settings = {"model": "offline", "provider": OfflineProvider(), "architect": SimpleArchitect(),
                "run_store": store, "max_budget_usd": 1, "acceptance": strict(),
                "lineage": LineagePolicy(access_scope=SCOPE)}
    return Swarm(**{**settings, **options})


def test_accepted_successors_supersede_and_failed_ones_do_not(tmp_path):
    reject = AcceptancePolicy(mode="strict", final_validators=[TextValidator({"max_chars": 1})])
    catalog = ServiceCatalog(FixtureService())
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        def run(**options):
            return lineage_swarm(store, **options).execute(serial(1)).execution_id

        first, second, third = run(), run(), run()
        kept, failed = run(), run(acceptance=reject)
        other = run(lineage=LineagePolicy(access_scope="other-team"))
        plain = run(lineage=None)
        link(store, second, first)
        link(store, third, second)
        link(store, failed, kept)
        # Each accepted successor replaces its predecessor; the rejected one is not a head.
        assert archive_heads(store, SCOPE) == [third, kept]
        assert archive_heads(store, "other-team") == [other]
        assert archive_heads(store, "nobody") == []

        swarm = lineage_swarm(store)
        assert list(items(swarm.plan_repair(sources=catalog))) == [third, kept]
        # A superseded run can still be planned by its id; ids are planned once each.
        named = items(swarm.plan_repair([first, failed, first], sources=catalog))
        assert list(named) == [first, failed]
        assert named[first]["status"] == "superseded"
        assert named[failed]["status"] == "invalid"
        assert "decision_not_accepted" in named[failed]["reasons"]
        # Runs in another scope, or without lineage, are invisible.
        for run_id in (other, plain, "missing"):
            with pytest.raises(WorkflowNotFoundError, match="access scope"):
                swarm.plan_repair([run_id], sources=catalog)
        with pytest.raises(TypeError, match="sequence"):
            swarm.plan_repair(third, sources=catalog)


def test_planning_needs_a_durable_store_and_a_lineage_policy(tmp_path):
    catalog = ServiceCatalog(FixtureService())
    with pytest.raises(ValueError, match="run_store"):
        Swarm(model="offline", provider=OfflineProvider()).plan_repair(sources=catalog)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(ValueError, match="lineage"):
            lineage_swarm(store, lineage=None).plan_repair(sources=catalog)
        # The catalog is checked as the evaluator checks it.
        with pytest.raises(TypeError, match="describe"):
            lineage_swarm(store).plan_repair(sources=object())
        catalog.metered = True
        with pytest.raises(ValueError, match="metered"):
            lineage_swarm(store).plan_repair(sources=catalog)
        # An empty archive plans nothing.
        plan = lineage_swarm(store).plan_repair(sources=ServiceCatalog(FixtureService()))
    assert plan.to_dict()["items"] == [] and plan.to_dict()["sources"] == []


# ---------------------------------------------------------------------------
# Repairs in progress
# ---------------------------------------------------------------------------


def test_a_successor_that_has_not_completed_holds_its_predecessor_for_review(
        tmp_path, monkeypatch):
    # The three-source research of test_workflow_reuse, whose successors keep results.
    service, log = FixtureService(), []
    for source_id, text in successors.TEXTS.items():
        service.add_source(source_id, text)
    successors.install_replies(monkeypatch, log)
    with serve_http(service) as endpoint, SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = successors.research(store, endpoint, tmp_path)
        first = swarm.execute(successors.research_graph(endpoint)).execution_id
        service.change_source("churn", successors.CHANGED)
        catalog = successors.Catalog(service)
        earlier = swarm.plan_repair(sources=catalog)
        (before,) = earlier.to_dict()["items"]
        assert before["action"] == "regenerate" and "repairs_in_progress" not in before
        successor = successors.stopped_before_running(
            swarm, store, first, {node_id: [] for node_id in successors.KEPT}, monkeypatch)

        # The successor is not a head, and the run it repairs needs review
        # rather than a second successor.
        assert archive_heads(store, SCOPE) == [first]
        plan = swarm.plan_repair(sources=catalog)
        (item,) = plan.to_dict()["items"]
        assert (item["run_id"], item["action"]) == (first, "review")
        assert item["reasons"] == sorted([*before["reasons"], "repair_in_progress"])
        assert item["repairs_in_progress"] == [successor]
        # It keeps its own status and nodes, and predicts no work.
        assert (item["status"], item["nodes"]) == (before["status"], before["nodes"])
        assert item["work"] == ZERO and plan.plan_sha256 != earlier.plan_sha256
        # Named, the successor is a run in progress; inspection names no
        # successor until one has a record.
        (named,) = swarm.plan_repair([successor], sources=catalog).to_dict()["items"]
        assert (named["action"], named["reasons"]) == ("review", ["run_in_progress"])
        assert inspect_journal(store, first)["lineage"]["successors"] == []

        # Once resumed to completion, the successor supersedes its predecessor.
        assert swarm.resume(successor).accepted
        assert archive_heads(store, SCOPE) == [successor]
        (done,) = swarm.plan_repair(sources=catalog).to_dict()["items"]
        assert (done["run_id"], done["action"], done["status"]) == (successor, "reuse", "current")
        assert inspect_journal(store, first)["lineage"]["successors"] == [
            {"run_id": successor, "supersedes": True}]
