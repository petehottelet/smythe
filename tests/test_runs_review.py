"""Runs review packets list only what in saved runs needs a decision, and change nothing.

``smythe runs review`` reads ordinary checkpoints, or a durable journal opened
read-only, derives its items from the ``runs inspect`` projection and exits 0
whether or not anything needs a decision. Without a run id it reviews a page
of runs, newest first. Items share the Jobs review's schema.
"""

from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3

import pytest

from fakes import LOST_RESPONSE, Crash, ManualClock, NativeTransport, crash_when
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptanceDecision, AcceptancePolicy, FileCheckpointStore, OfflineProvider, SimpleArchitect,
    SQLiteWorkflowStore, Swarm, Task,
)
from smythe._run_inspection import RUN_REVIEW_KINDS, review_run
from smythe.cli import (
    EXIT_INTERRUPTED, EXIT_INVALID_INPUT, EXIT_JOB_STATE, EXIT_LOCAL_ERROR, EXIT_OK, build_parser,
    main,
)
from smythe.jobs.inspection import REVIEW_BLOCKS, REVIEW_VERSION
from smythe.lineage import restrict_source
from smythe.provider import ProviderAccountingError
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.reuse import REASONS
from smythe.validation import ValidatorIdentity
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import WorkflowConflictError
import test_action_approvals as action_approvals
import test_action_operator as operator_actions
from test_action_reconciliation import Reconciling, unknown
from test_jobs_review import holds_no_path
from test_runs_cli import (
    CONNECT, Broken, Fixed, action_journal, durable_run, edit_checkpoint, files, graph,
    journal_copy, journal_state, no_side_effects, output, policy, recorded_version, run,
    secrets_of,
)
from test_runs_cli import research, strict_journal  # noqa: F401  (fixtures)
from test_security_stop import stopped_state
from test_source_evidence import KEPT
from test_source_evidence import SOURCES as EVIDENCE_SOURCES
from test_source_evidence import artifact as evidence_artifact
from test_workflow_acceptance import Contains, durable, strict
from test_workflow_acceptance import replies as scripted_replies  # noqa: F401  (a fixture)
import test_workflow_read_tool as read_tool
from test_workflow_read_tool import service  # noqa: F401  (a fixture)
import test_workflow_reuse as reuse
from test_workflow_store_migration import build_journal, mark_version_1

WITHDRAWN = "Withdrawn at its owner's request"


def reviewed(capsys, option, store, *arguments):
    """Review through the CLI as one compact JSON document, checking the packet's schema."""
    assert main(["runs", option, str(store), "--json", "review", *arguments]) == EXIT_OK
    payload = output(capsys)
    assert payload["ok"] is True and payload["command"] == "review"
    packet = payload["review"]
    assert packet["review_version"] == REVIEW_VERSION == 1
    schema(packet)
    return packet


def schema(packet):
    """Every item has the shared schema, and every suggested command parses without a store."""
    for item in packet["items"]:
        assert list(item) == ["item_id", "kind", "blocks", "subject", "observed", "reasons",
                              "decision_needed", "resolve_with"]
        assert item["blocks"] in REVIEW_BLOCKS and item["kind"] in RUN_REVIEW_KINDS
        assert set(item["reasons"]) <= set(REASONS) and "run_id" in item["subject"]
        assert {"status", "error_class", "excerpt"} <= set(item["observed"])
        for argv in item["resolve_with"]:
            args = build_parser().parse_args(argv)
            assert (args.root_command, args.runs_command) == ("runs", argv[1])
            assert args.journal is None and args.checkpoints is None
    assert packet["totals"] == {kind: sum(item["kind"] == kind for item in packet["items"])
                                for kind in RUN_REVIEW_KINDS}
    order = [(REVIEW_BLOCKS.index(item["blocks"])) for item in packet["items"]]
    assert order == sorted(order)


def kinds(packet):
    return [item["kind"] for item in packet["items"]]


def durable_swarm(store, **options) -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def without_wal(path):
    """Leave a journal in rollback mode, so that reading it creates no WAL sidecars."""
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    return path


# ---------------------------------------------------------------------------
# Runs that need nothing
# ---------------------------------------------------------------------------


def test_runs_that_need_nothing_give_an_empty_packet(tmp_path, capsys, monkeypatch):
    run(tmp_path)
    accepted = run(tmp_path, policy("pass"))
    before = files(tmp_path)
    no_side_effects(monkeypatch)

    packet = reviewed(capsys, "--checkpoints", tmp_path)

    assert (packet["store"], packet["run_id"]) == ("checkpoints", None)
    assert packet["page"] == {"limit": 50, "offset": 0, "total": 2, "returned": 2,
                              "has_more": False}
    assert packet["items"] == [] and packet["totals"] == dict.fromkeys(RUN_REVIEW_KINDS, 0)
    single = reviewed(capsys, "--checkpoints", tmp_path, accepted.execution_id)
    assert (single["run_id"], single["page"], single["items"]) == (accepted.execution_id, None, [])
    assert main(["runs", "--checkpoints", str(tmp_path), "review"]) == EXIT_OK
    assert capsys.readouterr().out.splitlines() == [
        "Runs reviewed: 2 of 2 in ordinary checkpoints, newest first (offset 0, limit 50)",
        "Needs a decision: 0", "Nothing needs a decision."]
    assert files(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_checkpoints_give_no_items(tmp_path, capsys, version):
    record = run(tmp_path / "strict", policy("reject")).acceptance.to_dict()
    result = run(tmp_path / "legacy")

    def legacy(state):
        state["version"] = version
        if version == 1:
            del state["control"]  # version 1 predates run control state
        # A legacy checkpoint holds no record, whatever it carries under that key.
        state["delivery_acceptance"] = record

    edit_checkpoint(tmp_path / "legacy", result.execution_id, legacy)
    packet = reviewed(capsys, "--checkpoints", tmp_path / "legacy")
    assert packet["items"] == [] and packet["page"]["total"] == 1
    store = FileCheckpointStore(tmp_path / "legacy")
    assert review_run(store, result.execution_id, acceptance=policy("pass"))["items"] == []


def test_version_1_durable_checkpoints_give_no_items(strict_journal, tmp_path, capsys):  # noqa: F811
    path = journal_copy(strict_journal, tmp_path)
    for name in ("legacy", "gate-validator"):
        run_id = strict_journal[1][name].execution_id
        assert reviewed(capsys, "--journal", path, run_id)["items"] == []
        with SQLiteWorkflowStore(path, read_only=True) as store:
            assert review_run(store, run_id, acceptance=policy("reject"))["items"] == []
    journal, run_id = durable_run(tmp_path / "v1")
    assert reviewed(capsys, "--journal", journal)["items"] == []
    assert reviewed(capsys, "--journal", journal, run_id)["items"] == []
    assert recorded_version(journal) == [(1,)]


# ---------------------------------------------------------------------------
# Delivery acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome,settings,decision,entry", [
    ("reject", {}, "rejected", "acceptance=policy"),
    ("raise", {"on_invalid": "needs_review"}, "needs_review", "operator=OperatorDecision"),
    ("raise", {}, "indeterminate", "operator=OperatorDecision"),
], ids=["rejected", "needs-review", "indeterminate"])
def test_a_decision_that_withholds_the_output_names_its_validators_and_reasons(
    tmp_path, capsys, monkeypatch, outcome, settings, decision, entry,
):
    result = run(tmp_path, policy(outcome, **settings))
    saved = json.loads((tmp_path / f"{result.execution_id}.json").read_text(encoding="utf-8"))
    before = files(tmp_path)
    no_side_effects(monkeypatch)

    packet = reviewed(capsys, "--checkpoints", tmp_path, result.execution_id)

    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == (decision, "delivery")
    assert item["subject"] == {"run_id": result.execution_id}
    observed = item["observed"]
    assert (observed["status"], observed["decision"], observed["integrity"]) == (
        "completed", decision, "verified")
    assert observed["validators"] == [{**Fixed(outcome).identity.to_dict(),
                                       "outcome": "rejected" if outcome == "reject" else "error"}]
    assert observed["excerpt"] == "; ".join(saved["delivery_acceptance"]["reasons"])
    assert item["reasons"] == ["decision_not_accepted"]
    assert "Swarm.reevaluate(run_id, acceptance=policy)" in item["decision_needed"]
    assert entry in item["decision_needed"] and item["resolve_with"] == []
    # Stable while nothing changes; the output itself is never shown.
    assert reviewed(capsys, "--checkpoints", tmp_path, result.execution_id) == packet
    assert saved["output"] not in json.dumps(packet)
    assert files(tmp_path) == before
    # The review exits 0, while inspect keeps its own exit codes.
    assert main(["runs", "--checkpoints", str(tmp_path), "review", result.execution_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert f"[delivery] {decision}: {result.execution_id} (item {item['item_id'][:12]})" in text
    assert f"  Validator fixed-{outcome} 2: {observed['validators'][0]['outcome']}" in text


def test_a_re_evaluation_that_accepts_the_output_settles_the_item(tmp_path, capsys):
    result = run(tmp_path, policy("reject"))
    [first] = reviewed(capsys, "--checkpoints", tmp_path)["items"]
    swarm = Swarm(model="offline", provider=OfflineProvider(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path))
    swarm.reevaluate(result.execution_id, acceptance=policy("raise"))
    [second] = reviewed(capsys, "--checkpoints", tmp_path)["items"]
    assert second["kind"] == "indeterminate" and second["item_id"] != first["item_id"]
    swarm.reevaluate(result.execution_id, acceptance=policy("pass"))
    assert reviewed(capsys, "--checkpoints", tmp_path)["items"] == []


def test_a_crash_before_the_decision_commit_asks_for_a_resume_that_decides_again(
    tmp_path, capsys, monkeypatch, scripted_replies,  # noqa: F811
):
    script, calls = scripted_replies
    script["Write"] = ["a report without it"]
    image, acceptance = tmp_path / "image.db", strict(Contains())
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        planned = durable(store, acceptance=acceptance).plan(Task("Write"))
        run_id = planned.run_ref["run_id"]
        # The process is lost after synthesis, as the decision is committed.
        point = crash_when(monkeypatch, store, "complete_operation",
                           lambda lease, key, *args, **kwargs: key.startswith("acceptance/"),
                           image=image, connection=store._db)
        with pytest.raises(Crash):
            durable(store, acceptance=acceptance).execute(planned)
        point.restore()
    before = journal_state(without_wal(image))

    packet = reviewed(capsys, "--journal", image, run_id)

    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == ("decision_uncommitted", "completion")
    assert item["observed"]["status"] == "running" and item["observed"]["nodes"] == {"completed": 1}
    assert "decides on the output again" in item["decision_needed"]
    assert acceptance.sha256 in item["decision_needed"] and item["resolve_with"] == []
    assert reviewed(capsys, "--journal", image)["items"] == packet["items"]
    assert journal_state(image) == before

    # Resuming the image decides on the saved output, which it never assumed accepted.
    sent, Contains.checked = list(calls), []
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as store:
        result = durable(store, acceptance=acceptance).resume(run_id)
    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    assert Contains.checked == ["a report without it"] and calls == sent
    assert kinds(reviewed(capsys, "--journal", image, run_id)) == ["rejected"]


class Revised(Fixed):
    """The passing validator at a later version, as a changed validator registry holds it."""

    def __init__(self) -> None:
        super().__init__("pass")
        self.identity = ValidatorIdentity.for_config("fixed-pass", "3", {"outcome": "pass"})


def test_a_changed_validator_needs_the_explicit_re_evaluation(
        strict_journal, tmp_path, capsys):  # noqa: F811
    path = journal_copy(strict_journal, tmp_path)
    run_id = strict_journal[1]["accepted"].execution_id
    revised = AcceptancePolicy(mode="strict", final_validators=[Revised()])
    with SQLiteWorkflowStore(path, read_only=True) as store:
        assert review_run(store, run_id, acceptance=policy("pass"))["items"] == []
        [item] = review_run(store, run_id, acceptance=revised)["items"]
        [settings] = review_run(store, run_id, acceptance=policy("pass", on_invalid="needs_review"))[
            "items"]
        with pytest.raises(ValueError, match="strict AcceptancePolicy"):
            review_run(store, run_id, acceptance=AcceptancePolicy())
    assert (item["kind"], item["blocks"], item["reasons"]) == (
        "policy_changed", "delivery", ["validator_set_changed"])
    assert settings["reasons"] == ["policy_changed"] and settings["item_id"] != item["item_id"]
    assert item["observed"]["policy_sha256"] == policy("pass").sha256
    assert "Swarm.reevaluate(run_id, acceptance=policy)" in item["decision_needed"]
    # The CLI registers no validators, so it never reports a changed one.
    assert reviewed(capsys, "--journal", path, run_id)["items"] == []
    with SQLiteWorkflowStore(path) as store:
        # Resume refuses the changed policy; only the explicit entry point decides again.
        with pytest.raises(WorkflowConflictError, match="acceptance policy"):
            durable_swarm(store, acceptance=revised).resume(run_id)
        assert durable_swarm(store).reevaluate(run_id, acceptance=revised).accepted
    with SQLiteWorkflowStore(path, read_only=True) as store:
        assert review_run(store, run_id, acceptance=revised)["items"] == []
        assert kinds(review_run(store, run_id, acceptance=policy("pass"))) == ["policy_changed"]


def test_a_damaged_published_document_asks_for_republication_first(
    tmp_path, capsys, scripted_replies,  # noqa: F811
):
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        run_id = action_approvals.completed(store, tmp_path, scripted_replies)
        publication = current_artifact(store, run_id).publication
    publication.path.write_bytes(publication.path.read_bytes() + b" (edited)")

    [item] = reviewed(capsys, "--journal", path, run_id)["items"]

    # The damaged file makes the decision indeterminate; only republication repairs that.
    assert (item["kind"], item["reasons"]) == ("artifact_integrity", ["artifact_changed"])
    assert item["subject"] == {"run_id": run_id, "publication": publication.sequence}
    assert item["observed"]["integrity"] == "changed" and "republish(run_id)" in item["decision_needed"]
    assert holds_no_path(item, tmp_path)
    with SQLiteWorkflowStore(path) as store:
        action_approvals.swarm(store, tmp_path).republish(run_id)
    assert reviewed(capsys, "--journal", path, run_id)["items"] == []


# ---------------------------------------------------------------------------
# Durable accounting, reads and sources
# ---------------------------------------------------------------------------


def test_unknown_exposure_lists_each_call_and_its_nanousd(tmp_path, capsys, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    transport.schedule("execution", 0, LOST_RESPONSE)
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        swarm = Swarm(model="gpt-6-astra", architect=SimpleArchitect(), run_store=store,
                      provider=OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100),
                      max_budget_usd=1)
        with pytest.raises(ProviderAccountingError):
            swarm.execute(graph())
        (listed,) = store.list_runs()
        run_id = listed["run_id"]
        inspected = store.inspect_run(run_id)
    [call] = [call for call in inspected["calls"] if call["billing_state"] == "unknown"]
    before = journal_state(without_wal(path))
    no_side_effects(monkeypatch)

    packet = reviewed(capsys, "--journal", path)

    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == ("unknown_exposure", "admission")
    assert item["subject"] == {"run_id": run_id}
    assert item["observed"]["status"] == "blocked"
    assert item["observed"]["unknown_calls"] == [{
        "call_id": call["call_id"], "phase": "execution", "scope_id": "node/draft",
        "unknown_nanousd": call["unknown_nanousd"]}]
    assert item["observed"]["unknown_nanousd"] == inspected["unknown_nanousd"] > 0
    assert "run the task again as a new run" in item["decision_needed"]
    assert journal_state(path) == before
    assert main(["runs", "--journal", str(path), "review"]) == EXIT_OK
    assert (f"  Unknown exposure: {inspected['unknown_nanousd']} nanoUSD on calls "
            f"{call['call_id']}") in capsys.readouterr().out


def test_a_version_1_journal_is_reviewed_without_migrating(tmp_path, capsys, monkeypatch):
    path = tmp_path / "workflow.db"
    build_journal(path, ManualClock())
    with closing(CONNECT(path)) as db:
        db.execute("PRAGMA journal_mode=DELETE")
    mark_version_1(path)
    before = path.read_bytes()
    statements = []

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    no_side_effects(monkeypatch)
    monkeypatch.setattr(sqlite3, "connect", connect)
    packet = reviewed(capsys, "--journal", path)
    monkeypatch.setattr(sqlite3, "connect", CONNECT)

    compact = ["".join(sql.upper().split()) for sql in statements]
    assert statements and "BEGINIMMEDIATE" not in compact
    assert not any(sql.startswith(("PRAGMAJOURNAL_MODE=", "CREATE", "INSERT", "UPDATE", "DELETE",
                                   "ALTER")) for sql in compact)
    assert path.read_bytes() == before and recorded_version(path) == [(1,)]
    assert sorted(item.name for item in tmp_path.iterdir()) == ["workflow.db"]
    found = {(item["kind"], item["subject"]["run_id"]) for item in packet["items"]}
    # The fixture's completed run saved a checkpoint without a version, which is unreadable.
    assert found == {("budget_overrun", "overrun"), ("unknown_exposure", "unknown"),
                     ("unreadable", "completed")}
    assert kinds(packet)[-1] == "unreadable"
    unknown_item = next(item for item in packet["items"] if item["kind"] == "unknown_exposure")
    [call] = unknown_item["observed"]["unknown_calls"]
    assert (call["phase"], call["scope_id"]) == ("execution", "node/a")
    assert call["unknown_nanousd"] == unknown_item["observed"]["unknown_nanousd"] > 0
    unreadable = next(item for item in packet["items"] if item["kind"] == "unreadable")
    assert unreadable["observed"]["error_class"] == "RunInspectionError"
    assert "checkpoint version None" in unreadable["observed"]["excerpt"]


@pytest.mark.parametrize("kind", ["free", "metered"])
def test_a_read_sent_without_a_saved_response_is_an_item(
        tmp_path, capsys, monkeypatch, service, kind):  # noqa: F811
    profile = read_tool.profile_for(service.port, kind)
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        # The source answered; the process was lost before the response was saved.
        point = crash_when(monkeypatch, SQLiteWorkflowStore, "append_response")
        with pytest.raises(Crash):
            read_tool.build(store, profile).execute(read_tool.graph_for(profile))
        point.restore()
        (listed,) = store.list_runs()
    assert service.arrivals == ["brief"]

    packet = reviewed(capsys, "--journal", path)

    # The stopped step's failure was saved as well; resuming the run settles both.
    assert kinds(packet) == ["run_failed", "read_unanswered"]
    item = packet["items"][1]
    assert item["blocks"] == "completion"
    assert item["subject"] == {"run_id": listed["run_id"], "node_id": "fetch", "generation": 0,
                               "attempt": 0, "invocation": 0}
    assert (item["observed"]["status"], item["observed"]["profile"]) == ("dispatched", "source")
    assert item["observed"]["metered"] is (kind == "metered")
    expected = "reads again as a new call" if kind == "free" else "never sends it again"
    assert expected in item["decision_needed"] and item["resolve_with"] == []
    assert read_tool.BRIEF not in json.dumps(item)


def test_a_removed_source_and_the_decision_it_leaves_are_items(research, capsys):  # noqa: F811
    result = research.run(evidence_artifact(KEPT), "survey-2019", "renewals-2026")
    run_id, path = result.execution_id, research.path
    assert result.accepted and reviewed(capsys, "--journal", path, run_id)["items"] == []
    with SQLiteWorkflowStore(path) as store:
        research.swarm(store).remove_source(run_id, "renewals-2026", "deleted on request",
                                            removed_at="2026-10-01")

    packet = reviewed(capsys, "--journal", path, run_id)

    assert kinds(packet) == ["source_unavailable", "indeterminate"]
    source, decision = packet["items"]
    assert source["subject"] == {"run_id": run_id, "source_id": "renewals-2026"}
    assert source["observed"]["excerpt"] == "removed from the run: deleted on request"
    assert source["reasons"] == ["source_unavailable"]
    assert "was removed from the run after this decision" in decision["observed"]["excerpt"]
    assert "Swarm.reevaluate" in decision["decision_needed"]
    assert all(item.text not in json.dumps(packet) for item in EVIDENCE_SOURCES)


def test_a_restricted_run_names_its_withdrawn_sources_and_offers_nothing_it_refuses(
        tmp_path, capsys, monkeypatch):
    reuse.install_replies(monkeypatch, [])
    library = FixtureService()
    for source_id, text in reuse.TEXTS.items():
        library.add_source(source_id, text)
    path = tmp_path / "runs.db"
    # A frozen journal clock stamps the restriction and keeps the lease from lapsing.
    with serve_http(library) as endpoint, SQLiteWorkflowStore(path, clock_ns=ManualClock()) as store:
        run_id = reuse.research(store, endpoint, tmp_path).execute(
            reuse.research_graph(endpoint)).execution_id
        library.revoke_source("churn")
        [restricted] = restrict_source(store, "churn", reason=WITHDRAWN, purge=True)
    before = journal_state(without_wal(path))

    packet = reviewed(capsys, "--journal", path, run_id)

    # Withdrawn content is never re-evaluated or published again, so no item suggests either.
    [item] = packet["items"]
    assert (item["kind"], item["blocks"], item["reasons"]) == (
        "source_restricted", "delivery", ["source_tombstoned"])
    assert [source["source_id"] for source in item["observed"]["sources"]] == ["churn"]
    assert item["observed"]["sources"][0]["sha256"] == restricted["tombstones"][0]["sha256"]
    assert item["observed"]["purged"] == [entry["relative_path"] for entry in restricted["purged"]]
    assert item["observed"]["excerpt"] == WITHDRAWN
    assert "Swarm.plan_repair" in item["decision_needed"]
    assert "reevaluate" not in item["decision_needed"]
    assert all(text not in json.dumps(packet) for text in reuse.TEXTS.values())
    assert journal_state(path) == before


# ---------------------------------------------------------------------------
# Stops and failures
# ---------------------------------------------------------------------------


def test_a_security_stop_is_the_only_item(strict_journal, tmp_path, capsys):  # noqa: F811
    path = journal_copy(strict_journal, tmp_path)
    run_id = strict_journal[1]["rejected"].execution_id
    assert kinds(reviewed(capsys, "--journal", path, run_id)) == ["rejected"]
    with SQLiteWorkflowStore(path) as store:
        store.security_stop(run_id, "approver_compromised")

    [item] = reviewed(capsys, "--journal", path, run_id)["items"]

    assert (item["kind"], item["blocks"]) == ("security_stop", "admission")
    assert item["observed"]["status"] == "blocked" and item["resolve_with"] == []
    assert "run the task again as a new run" in item["decision_needed"]

    checkpoints = tmp_path / "checkpoints"
    _, execution_id, state = stopped_state("failed")
    FileCheckpointStore(checkpoints).save(execution_id, state)
    [ordinary] = reviewed(capsys, "--checkpoints", checkpoints)["items"]
    assert (ordinary["kind"], ordinary["subject"]) == ("security_stop", {"run_id": execution_id})


def test_a_failed_checkpoint_names_how_to_resume_it(tmp_path, capsys):
    acceptance = policy("pass")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run(tmp_path, acceptance, provider=Broken())
    (run_id,) = FileCheckpointStore(tmp_path).list_ids()

    [item] = reviewed(capsys, "--checkpoints", tmp_path, run_id)["items"]

    assert (item["kind"], item["blocks"]) == ("run_failed", "completion")
    assert item["observed"]["status"] == "failed" and item["observed"]["unresolved"] == []
    assert item["decision_needed"].startswith("The run stopped before it completed. Resume it with")
    assert acceptance.sha256 in item["decision_needed"]

    edit_checkpoint(tmp_path, run_id,
                    lambda state: state["budget"].update(accounting_error="usage was malformed"))
    [unresolved] = reviewed(capsys, "--checkpoints", tmp_path, run_id)["items"]
    assert unresolved["observed"]["unresolved"] == ["the budget records an accounting error"]
    assert "Reconcile provider charges" in unresolved["decision_needed"]
    assert unresolved["item_id"] != item["item_id"]


# ---------------------------------------------------------------------------
# External actions
# ---------------------------------------------------------------------------


def awaiting_approval(store, tmp_path, replies, host, clock):  # noqa: F811
    run_id = action_approvals.completed(store, tmp_path, replies)
    return (action_approvals.prepare(store, run_id),)


def lost_acknowledgement(store, tmp_path, replies, host, clock):  # noqa: F811
    """The service committed the write and lost its acknowledgement; reconciliation can retry."""
    connector = Reconciling(FixtureService(clock_ns=clock))
    return unknown(store, tmp_path, replies, host, clock, connector, committed=True)


def unsettled(store, tmp_path, replies, host, clock):  # noqa: F811
    return operator_actions.under_review(store, tmp_path, replies, host, clock, committed=True)


def duplicate_risk(store, tmp_path, replies, host, clock):  # noqa: F811
    action = operator_actions.under_review(store, tmp_path, replies, host, clock, max_attempts=3)[0]
    operator_actions.decide(store, host, action, "not_executed", operator_actions.evidence())
    return (action,)


@pytest.mark.parametrize("setup,kind,step", [
    (awaiting_approval, "action_approval", "approve"),
    (lost_acknowledgement, "action_outcome_unknown", "reconcile"),
    (unsettled, "action_outcome_unknown", "operator_review"),
    (duplicate_risk, "action_duplicate_risk", "approve_duplicate_risk"),
    (operator_actions._wrong, "action_compensation", "compensate"),
], ids=["awaiting-approval", "reconcile", "operator-review", "duplicate-risk", "compensation"])
def test_external_actions_that_need_a_decision(tmp_path, capsys, monkeypatch, scripted_replies,  # noqa: F811
                                               setup, kind, step):
    path, action, _, _ = action_journal(tmp_path, scripted_replies, setup=setup)
    before = journal_state(path)
    no_side_effects(monkeypatch)

    packet = reviewed(capsys, "--journal", path, action.run_id)

    # The run's output is accepted; only the action needs a decision.
    [item] = packet["items"]
    assert (item["kind"], item["blocks"]) == (kind, "delivery")
    assert item["subject"] == {"run_id": action.run_id, "action_id": action.action_id}
    assert item["observed"]["next_action"] == step
    assert (item["observed"]["slot"], item["observed"]["sequence"]) == ("publish", 1)
    entry = {"approve": "smythe.actions.approve", "reconcile": ".reconcile(",
             "operator_review": "smythe.actions.resolve", "compensate": "prepare_compensation",
             "approve_duplicate_risk": "duplicate_risk=True"}[step]
    assert entry in item["decision_needed"]
    if step == "reconcile":
        assert item["resolve_with"] == [["runs", "revoke-action", action.action_id, "--reason",
                                         "REASON"]]
    else:
        assert item["resolve_with"] == []
    assert not any(secret in json.dumps(packet) for secret in secrets_of(path))
    assert journal_state(path) == before


def test_revoking_an_actions_approvals_changes_its_item(tmp_path, capsys, scripted_replies):  # noqa: F811
    path, action, _, _ = action_journal(tmp_path, scripted_replies, setup=lost_acknowledgement)
    [item] = reviewed(capsys, "--journal", path)["items"]
    assert main(["runs", "--journal", str(path), "review"]) == EXIT_OK
    text = capsys.readouterr().out
    assert (f"  Resolve with: smythe runs revoke-action {action.action_id} --reason REASON"
            in text)
    assert text.splitlines()[-1] == ("To run a command, add the --journal option used here and "
                                     "replace REASON with your own reason.")

    [argv] = item["resolve_with"]
    command = [token if token != "REASON" else "no further writes" for token in argv]
    assert main([*command, "--journal", str(path), "--json"]) == EXIT_OK
    output(capsys)
    [after] = reviewed(capsys, "--journal", path)["items"]
    assert (after["kind"], after["resolve_with"]) == ("action_outcome_unknown", [])
    assert after["item_id"] != item["item_id"]


# ---------------------------------------------------------------------------
# Scans, output and refusals
# ---------------------------------------------------------------------------


def test_a_scan_reviews_the_newest_runs_first_within_its_limit(tmp_path, capsys):
    rejected = run(tmp_path, policy("reject")).execution_id
    clean = run(tmp_path).execution_id
    undecided = run(tmp_path, policy("raise")).execution_id
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "folder.json").mkdir()
    saved = [rejected, clean, "folder", "broken", undecided]  # oldest first
    for index, name in enumerate(saved):
        stamp = 1_700_000_000_000_000_000 + index * 10**9
        os.utime(tmp_path / f"{name}.json", ns=(stamp, stamp))

    first = reviewed(capsys, "--checkpoints", tmp_path, "--limit", "2")
    assert first["page"] == {"limit": 2, "offset": 0, "total": 5, "returned": 2, "has_more": True}
    # Completion comes before delivery; the newest run first within each.
    assert [(item["kind"], item["subject"]["run_id"]) for item in first["items"]] == [
        ("unreadable", "broken"), ("indeterminate", undecided)]
    rest = reviewed(capsys, "--checkpoints", tmp_path, "--offset", "2")
    assert [(item["kind"], item["subject"]["run_id"]) for item in rest["items"]] == [
        ("unreadable", "folder"), ("rejected", rejected)]
    every = reviewed(capsys, "--checkpoints", tmp_path)
    assert sorted(every["items"], key=lambda item: item["item_id"]) == sorted(
        first["items"] + rest["items"], key=lambda item: item["item_id"])
    assert len({item["item_id"] for item in every["items"]}) == 4
    folder = next(item for item in every["items"] if item["subject"]["run_id"] == "folder")
    assert folder["observed"]["excerpt"].startswith("Checkpoint 'folder' is not readable")
    assert holds_no_path(every, tmp_path)

    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "review", "--limit", "1"]) == EXIT_OK
    text = capsys.readouterr().out
    assert text.count("\n") == 1
    assert json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":")) == text.strip()
    assert main(["runs", "--checkpoints", str(tmp_path), "review", "--limit", "2"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Runs reviewed: 2 of 5 in ordinary checkpoints, newest first (offset 0, limit 2)"
    assert lines[-1] == "More runs: use --offset 2"


def test_review_options_and_refusals_have_stable_exit_codes(tmp_path, capsys):
    result = run(tmp_path)
    for arguments in (["--limit", "0"], ["--limit", "501"], ["--offset", "-1"], ["../escape"]):
        assert main(["runs", "--checkpoints", str(tmp_path), "--json", "review",
                     *arguments]) == EXIT_INVALID_INPUT
        assert output(capsys)["error"]["type"] == "ValueError"
    assert main(["runs", "--checkpoints", str(tmp_path), "--journal", str(tmp_path / "runs.db"),
                 "--json", "review"]) == EXIT_INVALID_INPUT
    assert "not both" in output(capsys)["error"]["message"]
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "review", "missing"]) == (
        EXIT_JOB_STATE)
    assert output(capsys)["error"]["type"] == "RunNotFoundError"
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "review", "broken"]) == (
        EXIT_LOCAL_ERROR)
    assert output(capsys)["error"]["type"] == "RunInspectionError"
    missing = tmp_path / "absent"
    for option, store in (("--checkpoints", missing / "checkpoints"), ("--journal", missing / "runs.db")):
        assert main(["runs", option, str(store), "--json", "review"]) == EXIT_LOCAL_ERROR
        assert output(capsys)["ok"] is False
    assert not missing.exists()
    args = build_parser().parse_args(["runs", "review", result.execution_id, "--journal", "j.db",
                                      "--limit", "3"])
    assert (args.runs_command, args.run_id, args.journal, args.limit, args.offset) == (
        "review", result.execution_id, "j.db", 3, 0)
    assert build_parser().parse_args(["runs", "review"]).run_id is None


def test_an_interrupted_review_writes_no_partial_packet(tmp_path, capsys, monkeypatch):
    run(tmp_path, policy("reject"))
    dumps = json.dumps

    def interrupted(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("ok") and value.get("command") == "review":
            raise KeyboardInterrupt
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr("smythe.cli.json.dumps", interrupted)
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "review"]) == EXIT_INTERRUPTED
    assert output(capsys) == {"ok": False, "command": "review",
                              "error": {"type": "KeyboardInterrupt", "message": "review interrupted"}}
    monkeypatch.setattr("smythe.cli.json.dumps", dumps)

    import smythe.cli as cli

    terminal_text, calls = cli._terminal_text, []

    def human(value, **kwargs):
        calls.append(value)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return terminal_text(value, **kwargs)

    monkeypatch.setattr(cli, "_terminal_text", human)
    assert main(["runs", "--checkpoints", str(tmp_path), "review"]) == EXIT_INTERRUPTED
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "error: review interrupted\n"


def test_human_review_escapes_stored_terminal_controls(capsys):
    from smythe.cli import _emit_run_review

    _emit_run_review({
        "review_version": 1, "store": "journal", "run_id": "run\x1b[2J", "page": None,
        "totals": {}, "items": [{
            "item_id": "0" * 64, "kind": "source_unavailable", "blocks": "delivery",
            "subject": {"run_id": "run\x1b[2J", "source_id": "src\x1b]0;title\x07"},
            "observed": {"status": "completed", "error_class": None, "excerpt": "gone\x1b[31m",
                         "validators": [{"name": "v\x1b[0m", "version": "1", "outcome": "error"}]},
            "reasons": [], "decision_needed": "Decide\x1b[0m.", "resolve_with": [],
        }],
    }, journal=True)
    human = capsys.readouterr().out
    assert "\x1b" not in human and "\x07" not in human
    assert human.splitlines()[0] == "Run: run\\u001b[2J (durable workflow journal)"
    assert "src\\u001b]0;title\\u0007" in human and "gone\\u001b[31m" in human
    assert "  Validator v\\u001b[0m 1: error" in human
