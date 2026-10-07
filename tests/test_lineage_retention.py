"""Restricting a source withdraws its text from every lineage run that used it.

The research runs here come from ``test_workflow_reuse``: each reads three
versioned sources from the fake external service over loopback HTTP, extracts
from each read with a model node and publishes a brief written by local code.
``restrict_source`` journals a tombstone of the withdrawn text in every
completed run whose lineage names it, on the journal's clock, and can purge
those runs' published documents. A restricted run never delivers, reuses,
re-evaluates or publishes again what it derived; a new version of the source
lets a successor regenerate it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from fakes import ManualClock
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptanceDecision, ContextSource, LineagePolicy, OfflineProvider, ReadToolError,
    SourceRecord, SQLiteWorkflowStore, WorkflowBindingError,
)
from smythe._run_inspection import inspect_journal
from smythe.actions import ActionNotAuthorized, approve, check_authority, prepare_action
from smythe.cli import EXIT_RUN_INDETERMINATE, main
from smythe.evidence import (
    SOURCE_TOMBSTONE_OPERATION, load_source_tombstones, source_tombstone_key,
)
from smythe.lineage import (
    PURGE_OPERATION, load_lineage, purged_artifacts, restrict_source, restricted_sources,
)
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import WorkflowLeaseError, WorkflowStateError
from test_action_approvals import CEILING, PROFILE, Host, delivered, intent
from test_workflow_reuse import (
    CHANGED, KEPT, TEXTS, Catalog, install_replies, research, research_graph, results,
    stopped_before_running, successor,
)

WITHDRAWN = "Withdrawn at its owner's request"
START = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
STAMP = "2026-10-01T09:00:00Z"
HOUR_NS = 3600 * 10**9


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(int(START.timestamp()) * 10**9)


@pytest.fixture
def store(tmp_path, clock):
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as value:
        yield value


@pytest.fixture
def library():
    service = FixtureService()
    for source_id, text in TEXTS.items():
        service.add_source(source_id, text)
    with serve_http(service) as endpoint:
        yield service, endpoint


@pytest.fixture
def replies(monkeypatch) -> list[str]:
    log: list[str] = []
    install_replies(monkeypatch, log)
    return log


def survey_only(endpoint):
    """The survey's read step and its extraction, which never read the churn source."""
    graph = research_graph(endpoint)
    graph.nodes = [node for node in graph.nodes if node.id.endswith("survey")]
    return graph


def tombstone(service, source_id="churn", *, version=None, at=STAMP) -> dict:
    """The tombstone a restriction journals for the service's current text of *source_id*."""
    return {"version": 1, "source_id": source_id, "source_version": version,
            "sha256": service.snapshot()["sources"][source_id]["sha256"], "text": None,
            "reference": None, "tombstone": WITHDRAWN, "removed_at": at}


def items(plan) -> dict[str, dict]:
    return {item["run_id"]: item for item in plan.to_dict()["items"]}


def run_ids(store) -> list[str]:
    return [run["run_id"] for run in store.list_runs()]


# ---------------------------------------------------------------------------
# What a restriction records
# ---------------------------------------------------------------------------


def test_a_restriction_marks_every_run_that_used_the_source_restricted(
        store, library, tmp_path, replies, clock):
    service, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first, second = (swarm.execute(research_graph(endpoint)).execution_id for _ in range(2))
    survey = swarm.execute(survey_only(endpoint)).execution_id
    revisions = {run_id: store.get_checkpoint(run_id)["revision"] for run_id in run_ids(store)}
    counted = {run_id: store.inspect_run(run_id)["call_count"] for run_id in run_ids(store)}
    # The owner revokes access to the churn source; the journal still holds what it derived.
    service.revoke_source("churn")

    restricted = restrict_source(store, "churn", reason=WITHDRAWN)

    record = tombstone(service)
    # In journal order, which a frozen clock leaves to the run ids.
    ordered = run_ids(store)
    assert restricted == [{"run_id": run_id, "tombstones": [record], "purged": []}
                          for run_id in ordered if run_id != survey]
    key = source_tombstone_key("churn", record["sha256"])
    for run_id in (first, second):
        # The same tombstone a removed source leaves, applied by a new completed checkpoint.
        operation = store.load_operation(run_id, key)
        assert (operation["kind"], operation["state"]) == (SOURCE_TOMBSTONE_OPERATION, "applied")
        saved = store.get_checkpoint(run_id)
        assert saved["revision"] == revisions[run_id] + 1
        assert saved["checkpoint"]["status"] == "completed"
        assert restricted_sources(store, run_id) == (SourceRecord.from_dict(record),)
        assert store.inspect_run(run_id)["call_count"] == counted[run_id]
        assert store.audit(run_id)["ok"]
        lineage = inspect_journal(store, run_id)["lineage"]
        assert (lineage["status"], lineage["reasons"]) == (
            "restricted", ["decision_not_accepted", "source_tombstoned"])
        assert lineage["tombstones"] == [{"source_id": "churn", "version": None,
                                          "sha256": record["sha256"], "reason": WITHDRAWN,
                                          "removed_at": STAMP}]
    # A run that never read the source is untouched.
    assert restricted_sources(store, survey) == ()
    assert store.get_checkpoint(survey)["revision"] == revisions[survey]
    assert store.list_operations(survey, SOURCE_TOMBSTONE_OPERATION) == []

    # The service still serves the withdrawn text, so the plan needs review, never reuse.
    listed = items(swarm.plan_repair(sources=Catalog(service)))
    for run_id in (first, second):
        assert (listed[run_id]["action"], listed[run_id]["status"]) == ("review", "restricted")
        assert "source_tombstoned" in listed[run_id]["reasons"]
    assert (listed[survey]["action"], listed[survey]["status"]) == ("reuse", "current")
    # Repair refuses to reuse any result the withdrawn text reached, before any run exists.
    for reuse in ({"read-churn": []}, {node_id: [] for node_id in (*KEPT, "read-churn")}):
        with pytest.raises(WorkflowStateError, match="'read-churn'.*a source it used was removed"):
            successor(swarm, first, reuse)
    assert run_ids(store) == ordered

    # Times come from the journal's clock; a later restriction keeps the earlier tombstones.
    clock.advance(3600)
    later = restrict_source(store, "survey", reason=WITHDRAWN)
    assert [item["run_id"] for item in later] == ordered
    assert later[0]["tombstones"] == [tombstone(service, "survey", at="2026-10-01T10:00:00Z")]
    assert [item.removed_at for item in restricted_sources(store, first)] == [
        STAMP, "2026-10-01T10:00:00Z"]


def test_a_version_restricts_only_the_references_recorded_with_it(
        store, library, tmp_path, replies):
    service, endpoint = library
    current = service.snapshot()["sources"]
    sources = [ContextSource(source_id, item["content"], str(item["version"]))
               for source_id, item in current.items()]
    swarm = research(store, endpoint, tmp_path, context_sources=sources)
    run_id = swarm.execute(survey_only(endpoint)).execution_id
    revision = store.get_checkpoint(run_id)["revision"]

    assert restrict_source(store, "survey", version="2", reason=WITHDRAWN) == []
    assert restrict_source(store, "unknown", reason=WITHDRAWN) == []
    assert store.get_checkpoint(run_id)["revision"] == revision

    (restricted,) = restrict_source(store, "survey", version="1", reason=WITHDRAWN)
    assert restricted["tombstones"] == [tombstone(service, "survey", version="1")]
    # The extraction was given the run's sources, so it is restricted with the read.
    assert {node["id"] for node in load_lineage(store, run_id).nodes
            if any(ref["source_id"] == "survey" for ref in node["inputs"]["sources"])} == {
        "read-survey", "extract-survey"}
    resumed = swarm.resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert not resumed.accepted


# ---------------------------------------------------------------------------
# A restricted run never delivers, re-evaluates or publishes again
# ---------------------------------------------------------------------------


def test_a_restricted_runs_completed_resume_never_delivers(
        store, library, tmp_path, replies, clock):
    _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    result = swarm.execute(survey_only(endpoint))
    run_id = result.execution_id
    assert result.accepted
    host = Host()
    action = prepare_action(store, run_id, slot="publish", intent=intent(delivered(store, run_id)),
                            profile=PROFILE)
    approval = approve(store, action.action_id,
                       host.issue(action.intent.digest(), subject="editor-1", tenant="acme",
                                  expires_at_ns=clock.now_ns + HOUR_NS),
                       host, ceiling_nanousd=CEILING, expires_at_ns=clock.now_ns + HOUR_NS)
    assert check_authority(store, action.action_id, host) == approval

    restrict_source(store, "survey", reason=WITHDRAWN)

    resumed = swarm.resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert (resumed.output, resumed.rejected_output) == ("", result.output)
    assert any("'survey'" in reason and WITHDRAWN in reason
               for reason in resumed.acceptance.reasons)
    current = current_artifact(store, run_id)
    assert not current.eligible and current.reasons == ("not_accepted",)
    # External delivery goes through the governed action path, which refuses it too.
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        check_authority(store, action.action_id, host)
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        prepare_action(store, run_id, slot="announce",
                       intent=intent(delivered(store, run_id), title="Another post"),
                       profile=PROFILE)
    # No new decision or publication can deliver what the run derived.
    revision = store.get_checkpoint(run_id)["revision"]
    with pytest.raises(WorkflowStateError, match="cannot be re-evaluated.*'survey'.*restricted"):
        swarm.reevaluate(run_id)
    with pytest.raises(WorkflowStateError, match="cannot be published again.*'survey'.*restricted"):
        swarm.republish(run_id)
    assert store.get_checkpoint(run_id)["revision"] == revision
    view = inspect_journal(store, run_id)
    assert view["delivery"]["decision"] == "indeterminate"
    assert view["next_action"]["action"] == "resolve_indeterminate"


def decisions_and_publications(store, run_id) -> list[tuple[str, str]]:
    return [(item["operation_key"], item["state"]) for kind in ("acceptance", "publication",
                                                                "lineage")
            for item in store.list_operations(run_id, kind)]


def test_a_restricted_run_is_never_decided_or_published_again(
        store, library, tmp_path, replies):
    _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    run_id = swarm.execute(survey_only(endpoint)).execution_id
    published = current_artifact(store, run_id).publication.path
    restrict_source(store, "survey", reason=WITHDRAWN)
    revision, recorded = store.load_run(run_id)["revision"], decisions_and_publications(
        store, run_id)
    withdrawn = "source 'survey' that the run's lineage records was withdrawn"

    # Its policy would accept the saved output again: the restriction alone refuses it.
    with pytest.raises(WorkflowStateError, match=f"cannot be re-evaluated: {withdrawn}"):
        swarm.reevaluate(run_id)
    # Its deleted document would otherwise be rendered and decided again.
    published.unlink()
    with pytest.raises(WorkflowStateError, match=f"cannot be published again: {withdrawn}"):
        swarm.republish(run_id)

    assert store.load_run(run_id)["revision"] == revision
    assert decisions_and_publications(store, run_id) == recorded
    assert not published.exists() and listing(published.parent) == [".smythe-run-owner.json"]


def listing(directory) -> list[str]:
    return sorted(item.name for item in directory.iterdir())


def test_purge_deletes_the_derived_documents_and_keeps_each_tombstone(
        store, library, tmp_path, replies, capsys):
    service, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    survey = swarm.execute(survey_only(endpoint)).execution_id
    # A damaged document published again: both of the run's files are derived from the source.
    original = current_artifact(store, first).publication
    original.path.write_bytes(b"damaged")
    again = swarm.republish(first)
    files = [original.path, again.path]
    kept = current_artifact(store, survey).publication.path
    assert all(path.exists() for path in (*files, kept)) and again.name == "document-p1.md"
    # A retention rule deletes the churn source at the service.
    withdrawn = tombstone(service)
    service.delete_source("churn")

    (restricted,) = restrict_source(store, "churn", reason=WITHDRAWN, purge=True)

    purged = [{"relative_path": f"{first}/{name}", "sha256": original.sha256,
               "reason": WITHDRAWN, "purged_at": STAMP}
              for name in ("document.md", "document-p1.md")]
    assert restricted == {"run_id": first, "tombstones": [withdrawn], "purged": purged}
    assert not any(path.exists() for path in files) and kept.exists()
    assert (original.path.parent / ".smythe-run-owner.json").exists()
    assert purged_artifacts(store, first) == purged
    assert {item["state"] for item in store.list_operations(first, PURGE_OPERATION)} == {"applied"}
    assert store.audit(first)["ok"]

    # Inspection still shows each tombstone with its digest and reason.
    view = inspect_journal(store, first)
    lineage = view["lineage"]
    assert lineage["status"] == "restricted" and lineage["artifact"]["integrity"] == "missing"
    assert lineage["tombstones"][0]["reason"] == WITHDRAWN and lineage["purged"] == purged
    assert view["next_action"]["action"] == "resolve_indeterminate"
    with pytest.raises(WorkflowStateError, match="cannot be published again.*restricted"):
        swarm.republish(first)
    assert not any(path.exists() for path in files)
    assert main(["runs", "--journal", str(store.path), "inspect", first]) == EXIT_RUN_INDETERMINATE
    shown = capsys.readouterr().out
    assert (f"  Tombstone churn: sha256 {withdrawn['sha256'][:12]}; {WITHDRAWN} "
            f"({STAMP})") in shown
    assert f"  Purged {first}/document.md: sha256 {original.sha256[:12]}; {WITHDRAWN}" in shown
    # No record holds the withdrawn text.
    records = [restricted, view, store.list_operations(first, SOURCE_TOMBSTONE_OPERATION),
               store.list_operations(first, PURGE_OPERATION), shown]
    assert TEXTS["churn"] not in json.dumps(records)

    # Restricting again keeps the first tombstones and purge records and writes nothing.
    revision = store.get_checkpoint(first)["revision"]
    assert restrict_source(store, "churn", reason="Another reason", purge=True) == [restricted]
    assert store.get_checkpoint(first)["revision"] == revision


# ---------------------------------------------------------------------------
# Access scope and regeneration
# ---------------------------------------------------------------------------


def test_a_restriction_in_one_scope_leaves_identical_artifacts_in_another_current(
        store, library, tmp_path, replies):
    service, endpoint = library
    team_a = research(store, endpoint, tmp_path, lineage=LineagePolicy(access_scope="team-a"))
    team_b = research(store, endpoint, tmp_path, lineage=LineagePolicy(access_scope="team-b"))
    first = team_a.execute(research_graph(endpoint)).execution_id
    other = team_b.execute(research_graph(endpoint)).execution_id
    theirs = current_artifact(store, other).publication
    document = theirs.path.read_bytes()
    assert current_artifact(store, first).publication.path.read_bytes() == document

    restricted = restrict_source(store, "churn", scope="team-a", reason=WITHDRAWN, purge=True)

    assert [item["run_id"] for item in restricted] == [first]
    assert restricted_sources(store, other) == ()
    assert store.list_operations(other, SOURCE_TOMBSTONE_OPERATION) == []
    assert theirs.path.read_bytes() == document and current_artifact(store, other).eligible
    assert team_b.resume(other).accepted
    assert items(team_b.plan_repair(sources=Catalog(service)))[other]["status"] == "current"

    # A new version of the source lets team A regenerate what the restriction took.
    service.change_source("churn", CHANGED)
    plan = items(team_a.plan_repair(sources=Catalog(service)))
    assert list(plan) == [first]
    assert (plan[first]["action"], plan[first]["status"]) == ("regenerate", "restricted")
    reuse = {node["id"]: node["reasons"] for node in plan[first]["nodes"]
             if node["action"] == "reuse"}
    assert set(reuse) == set(KEPT)
    second = successor(team_a, first, reuse)
    assert second.accepted and results(second.graph)["read-churn"] == CHANGED
    assert TEXTS["churn"] not in second.output
    # The successor holds the predecessor's tombstone, which its new text does not match.
    (inherited,) = store.list_operations(second.execution_id, SOURCE_TOMBSTONE_OPERATION)
    assert inherited["result"] == {"tombstone": restricted[0]["tombstones"][0]}
    assert restricted_sources(store, second.execution_id) == ()
    # Team A's repair never names team B's runs or files.
    named = json.dumps([plan, load_lineage(store, second.execution_id).to_dict(),
                        store.list_operations(second.execution_id, "reuse"),
                        store.list_operations(second.execution_id, SOURCE_TOMBSTONE_OPERATION)])
    assert other not in named and theirs.directory not in named
    assert current_artifact(store, second.execution_id).publication.directory \
        != theirs.directory


def test_a_successor_never_derives_again_from_text_withdrawn_from_its_predecessor(
        store, library, tmp_path, replies):
    service, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    withdrawn = tombstone(service)
    restrict_source(store, "churn", reason=WITHDRAWN)

    # A successor given the withdrawn text as a source is refused before it exists.
    given = research(store, endpoint, tmp_path,
                     context_sources=[ContextSource("churn", TEXTS["churn"], "1")])
    with pytest.raises(WorkflowStateError, match="'churn' was removed from run"):
        successor(given, first, {node_id: [] for node_id in KEPT})
    assert run_ids(store) == [first]

    # The service still serves the withdrawn text; the successor's read step is refused it.
    refused = "the response is the text of source 'churn', which was removed from this run"
    with pytest.raises(ReadToolError, match=refused):
        successor(swarm, first, {node_id: [] for node_id in KEPT})
    (second,) = [run_id for run_id in run_ids(store) if run_id != first]
    inherited = store.load_operation(second, source_tombstone_key("churn", withdrawn["sha256"]))
    assert inherited["state"] == "applied" and inherited["result"] == {"tombstone": withdrawn}
    (read,) = [call for call in store.inspect_run(second)["calls"]
               if call["key"]["scope_id"] == "node/read-churn"]
    assert read["result_state"] == "rejected"
    saved = store.get_checkpoint(second)["checkpoint"]
    churn = next(node for node in saved["graph"]["nodes"] if node["id"] == "read-churn")
    assert saved["status"] == "failed" and TEXTS["churn"] not in (churn["result"] or "")
    # Resuming reads it again and is refused it again: nothing derives from it.
    with pytest.raises(ReadToolError, match=refused):
        swarm.resume(second)
    assert store.get_checkpoint(second)["checkpoint"]["status"] == "failed"


@pytest.mark.parametrize("held", [False, True], ids=["free", "held-by-another-process"])
def test_a_restriction_reaches_a_successor_in_progress(
        store, library, tmp_path, replies, monkeypatch, clock, held):
    service, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    clock.advance(1)
    service.change_source("churn", CHANGED)
    # A repair's successor, lost after it committed the reuse of the survey's read and extraction.
    second = stopped_before_running(swarm, store, first, {node_id: [] for node_id in KEPT},
                                    monkeypatch)
    withdrawn = tombstone(service, "survey", at="2026-10-01T09:00:01Z")

    if held:
        # Another process holds the successor: the predecessor is restricted first.
        lease = store.acquire_lease(second, "another-process", ttl_s=60)
        with pytest.raises(WorkflowLeaseError):
            restrict_source(store, "survey", reason=WITHDRAWN)
        store.release_lease(lease)
        assert load_source_tombstones(store, second) == ()
    else:
        restricted = restrict_source(store, "survey", reason=WITHDRAWN)
        assert restricted == [{"run_id": run_id, "tombstones": [withdrawn], "purged": []}
                              for run_id in (first, second)]
        assert [item.to_dict() for item in load_source_tombstones(store, second)] == [withdrawn]
        assert store.get_checkpoint(second)["checkpoint"]["status"] == "planned"
    assert [item.to_dict() for item in restricted_sources(store, first)] == [withdrawn]
    labels = list(replies)

    # Resuming, the successor holds its predecessor's tombstone and never uses
    # the reused result the withdrawn text reached.
    with pytest.raises(WorkflowStateError,
                       match="'read-survey' cannot be reused: a source it used was removed"):
        swarm.resume(second)
    assert [item.to_dict() for item in load_source_tombstones(store, second)] == [withdrawn]
    assert store.inspect_run(second)["call_count"] == 0 and replies == labels
    # The run it repairs is restricted and waits for the repair: nothing is reused.
    (item,) = swarm.plan_repair(sources=Catalog(service)).to_dict()["items"]
    assert (item["run_id"], item["action"], item["status"]) == (first, "review", "restricted")
    assert {"repair_in_progress", "source_tombstoned"} <= set(item["reasons"])


def test_a_restriction_made_while_a_successor_runs_reaches_it_before_it_delivers(
        store, library, tmp_path, replies, monkeypatch, clock):
    service, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    clock.advance(1)
    service.change_source("churn", CHANGED)
    withdrawn = tombstone(service, "survey", at="2026-10-01T09:00:01Z")
    answer, restricted = OfflineProvider.complete, []

    async def complete(self, system, prompt, model):
        if "Your step: Extract the churn finding" in prompt and not restricted:
            # Another process withdraws the survey while the successor runs.
            with SQLiteWorkflowStore(store.path, clock_ns=clock) as other:
                try:
                    restricted.append(restrict_source(other, "survey", reason=WITHDRAWN))
                except WorkflowLeaseError as error:
                    restricted.append(error)
        return await answer(self, system, prompt, model)

    monkeypatch.setattr(OfflineProvider, "complete", complete)

    result = successor(swarm, first, {node_id: [] for node_id in KEPT})

    # The successor took its predecessor's tombstone before it decided, so
    # it delivers nothing derived from the withdrawn text.
    second = result.execution_id
    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE and not result.accepted
    assert any("'survey'" in reason and WITHDRAWN in reason
               for reason in result.acceptance.reasons)
    # The restriction reached the predecessor first, then waited for the successor's lease.
    (refused,) = restricted
    assert isinstance(refused, WorkflowLeaseError)
    assert [item.to_dict() for item in restricted_sources(store, first)] == [withdrawn]
    assert [item.to_dict() for item in restricted_sources(store, second)] == [withdrawn]
    assert not current_artifact(store, second).eligible
    assert swarm.resume(second).acceptance.decision is AcceptanceDecision.INDETERMINATE
    # Calling the restriction again finishes it and keeps each first tombstone.
    again = restrict_source(store, "survey", reason="Another reason")
    assert again == [{"run_id": run_id, "tombstones": [withdrawn], "purged": []}
                     for run_id in (first, second)]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("source_id", "options", "match"), [
    ("", {}, "source_id"),
    ("   ", {}, "source_id"),
    ("churn", {"reason": ""}, "reason"),
    ("churn", {"reason": "x" * 501}, "reason"),
    ("churn", {"reason": "two\nlines"}, "reason"),
    ("churn", {"reason": 7}, "reason"),
    ("churn", {"version": ""}, "version"),
    ("churn", {"scope": "two words"}, "scope"),
    ("churn", {"purge": "yes"}, "purge"),
])
def test_malformed_restrictions_are_refused(store, source_id, options, match):
    options = {"reason": WITHDRAWN, **options}
    with pytest.raises(WorkflowBindingError, match=match):
        restrict_source(store, source_id, **options)


def test_a_restriction_waits_for_a_run_another_process_holds(store, library, tmp_path, replies):
    _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    for _ in range(2):
        swarm.execute(survey_only(endpoint))
    # The later run in journal order is held, so the earlier one is restricted first.
    first, second = run_ids(store)
    lease = store.acquire_lease(second, "another-process", ttl_s=60)
    with pytest.raises(WorkflowLeaseError):
        restrict_source(store, "survey", reason=WITHDRAWN)
    assert len(restricted_sources(store, first)) == 1 and restricted_sources(store, second) == ()
    store.release_lease(lease)
    # Calling it again finishes the restriction; the first run's tombstone is kept once.
    restricted = restrict_source(store, "survey", reason="Another reason")
    assert [item["run_id"] for item in restricted] == [first, second]
    assert [item["tombstones"][0]["tombstone"] for item in restricted] == [WITHDRAWN,
                                                                           "Another reason"]
    assert all(len(store.list_operations(run_id, SOURCE_TOMBSTONE_OPERATION)) == 1
               for run_id in (first, second))
    with SQLiteWorkflowStore(store.path, read_only=True) as reader:
        for source_id in ("survey", "unknown"):
            with pytest.raises(WorkflowStateError, match="read-only"):
                restrict_source(reader, source_id, reason=WITHDRAWN)
    assert restrict_source(store, "survey", reason=WITHDRAWN) == restricted
