"""Runs inspection reads saved checkpoints and journals without changing them.

Execution, delivery acceptance and cost are separate blocks. Exit status 10
means a completed run's output was rejected or needs review, 11 that no
valid decision covers it, and 12 that accepted output has an external action
whose effect is unresolved. ``runs revoke-action`` is the one command that
writes, and only revocations.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import time

import pytest

from fakes import ManualClock
from fixture_connector import FixtureConnector
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptanceDecision,
    AcceptancePolicy,
    FileCheckpointStore,
    LocalOnly,
    OfflineProvider,
    ReadProfile,
    SimpleArchitect,
    SourceEvidenceValidator,
    SQLiteWorkflowStore,
    Swarm,
    Task,
    ValidationReport,
    ValidatorIdentity,
)
import smythe.cli as cli
from smythe.acceptance import text_sha256
from smythe.actions import inspect_actions
from smythe.cli import (
    EXIT_INVALID_INPUT, EXIT_JOB_STATE, EXIT_LOCAL_ERROR, EXIT_OK, EXIT_RUN_ACTION_UNRESOLVED,
    EXIT_RUN_INDETERMINATE, EXIT_RUN_NOT_ACCEPTED, build_parser, main,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.lineage import load_lineage
from smythe.provider import CompletionResult, Provider
from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT, LLMSupervisor, Supervisor
from smythe.validation import hard_finding
from smythe.workflow_artifact import current_artifact
import test_action_approvals as action_approvals
from test_action_journal import rows as action_rows
import test_action_operator as operator_actions
import test_repair_plan as repair_plans
from test_source_evidence import CLAIM, DROPPED, KEPT, STRONG
from test_source_evidence import SOURCES as EVIDENCE_SOURCES
from test_source_evidence import artifact as evidence_artifact
from test_workflow_acceptance import Contains, Raises, rewrite_checkpoint, strict
from test_workflow_acceptance import replies as scripted_replies  # noqa: F401  (a fixture)
import test_workflow_lineage as lineage_runs
from test_workflow_store_migration import build_journal, dump, mark_version_1

CONNECT = sqlite3.connect


class Fixed:
    """A final validator with a fixed outcome; ``raise`` raises instead of reporting."""

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.identity = ValidatorIdentity.for_config(f"fixed-{outcome}", "2", {"outcome": outcome})

    def validate(self, subject):
        if self.outcome == "raise":
            raise RuntimeError("validator crashed")
        findings = (hard_finding("refused", "fixed rejection"),) if self.outcome == "reject" else ()
        return ValidationReport("output", findings, validator=self.identity)


class Broken(Provider):
    """Fails every call, so the run stops with a failed checkpoint."""

    async def complete(self, system, prompt, model) -> CompletionResult:
        raise RuntimeError("provider unavailable")


def graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft the answer"),
        Node(id="final", label="Finish the answer", depends_on=["draft"]),
    ])


def policy(outcome: str, **settings) -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[Fixed(outcome)], **settings)


def run(directory: Path, acceptance: AcceptancePolicy | None = None, provider=None):
    swarm = Swarm(model="offline", provider=provider or OfflineProvider(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(directory), acceptance=acceptance)
    return swarm.execute(graph())


def output(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    return json.loads(captured.out, parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"))


def inspect(capsys, directory, run_id, *extra):
    code = main(["runs", "--checkpoints", str(directory), "--json", "inspect", run_id, *extra])
    value = output(capsys)
    assert value["ok"] is True and value["command"] == "inspect"
    return code, value["inspect"]


def files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def provider_classes(cls=Provider):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from provider_classes(subclass)


def no_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Runs inspection constructed a provider or wrote a store")

    for provider in provider_classes():
        if "__init__" in provider.__dict__:
            monkeypatch.setattr(provider, "__init__", forbidden)
    monkeypatch.setattr("smythe.cli.ProviderPool", forbidden)
    monkeypatch.setattr("smythe.cli.JobRunner", forbidden)
    monkeypatch.setattr(FileCheckpointStore, "save", forbidden)
    monkeypatch.setattr(FileCheckpointStore, "delete", forbidden)
    monkeypatch.setattr(SQLiteWorkflowStore, "_migrate_schema", forbidden)


def edit_checkpoint(directory: Path, run_id: str, change) -> None:
    path = directory / f"{run_id}.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    change(state)
    path.write_text(json.dumps(state), encoding="utf-8")


def test_exit_codes_are_new_and_the_options_are_inherited():
    taken = {getattr(cli, name) for name in dir(cli)
             if name.startswith("EXIT_") and not name.startswith("EXIT_RUN_")}
    assert (EXIT_RUN_NOT_ACCEPTED, EXIT_RUN_INDETERMINATE) == (10, 11)
    assert not {EXIT_RUN_NOT_ACCEPTED, EXIT_RUN_INDETERMINATE} & taken
    args = build_parser().parse_args(["runs", "inspect", "abc", "--journal", "runs.db", "--json"])
    assert (args.runs_command, args.run_id, args.journal, args.json) == ("inspect", "abc", "runs.db", True)
    assert args.checkpoints is None
    args = build_parser().parse_args(["runs", "--checkpoints", "saved", "list", "--limit", "3"])
    assert (args.checkpoints, args.limit, args.offset, args.json) == ("saved", 3, 0, False)


def test_legacy_checkpoint_shows_acceptance_as_not_recorded(tmp_path, capsys, monkeypatch):
    result = run(tmp_path)
    assert json.loads((tmp_path / f"{result.execution_id}.json").read_text())["version"] == 4
    no_side_effects(monkeypatch)
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_OK
    assert value["format"] == {"checkpoint_version": 4, "requires": []}
    assert value["execution"] == {"state": "completed", "blocked_reason": None, "nodes": {"completed": 2}}
    assert value["delivery"] == {"recorded": False, "decision": None, "reasons": [], "integrity": None,
                                 "output_delivered": True, "rejected_output": None, "record": None}
    assert value["decisions"] == {"current": None, "history": []}
    assert value["accounting"]["cost_is_complete"] is True
    assert value["accounting"]["unknown_exposure"] is None
    assert value["accounting"]["unresolved"] == []
    assert value["timing"]["clock"] == "monotonic" and value["timing"]["segments"] == 1
    assert value["timing"]["phases_ms"]["execution"] >= 0
    assert value["next_action"]["action"] == "none"

    assert main(["runs", "--checkpoints", str(tmp_path), "inspect", result.execution_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Delivery acceptance: not recorded" in text
    assert "Next action: none" in text


@pytest.mark.parametrize("outcome,settings,exit_code,decision,action", [
    ("pass", {}, EXIT_OK, "accepted", "none"),
    ("reject", {}, EXIT_RUN_NOT_ACCEPTED, "rejected", "review_output"),
    ("raise", {"on_invalid": "needs_review"}, EXIT_RUN_NOT_ACCEPTED, "needs_review", "review_output"),
    ("raise", {}, EXIT_RUN_INDETERMINATE, "indeterminate", "resolve_indeterminate"),
])
def test_strict_decisions_have_distinct_exit_codes(
    tmp_path, capsys, monkeypatch, outcome, settings, exit_code, decision, action,
):
    acceptance = policy(outcome, **settings)
    result = run(tmp_path, acceptance)
    saved = json.loads((tmp_path / f"{result.execution_id}.json").read_text())
    before = files(tmp_path)
    no_side_effects(monkeypatch)

    code, value = inspect(capsys, tmp_path, result.execution_id)

    assert code == exit_code
    assert files(tmp_path) == before
    assert value["format"] == {"checkpoint_version": 5, "requires": ["delivery_acceptance"]}
    delivery = value["delivery"]
    assert delivery["decision"] == decision and delivery["integrity"] == "verified"
    assert delivery["record"] == saved["delivery_acceptance"]
    assert delivery["record"]["output_sha256"] == text_sha256(saved["output"])
    assert delivery["record"]["policy_sha256"] == acceptance.sha256
    assert delivery["record"]["validators"][0]["validator"] == Fixed(outcome).identity.to_dict()
    assert value["decisions"] == {"current": 0, "history": []}
    assert value["next_action"]["action"] == action
    if decision == "accepted":
        assert delivery["output_delivered"] is True and delivery["rejected_output"] is None
    else:
        assert delivery["output_delivered"] is False
        assert delivery["rejected_output"] == {
            "field": "output", "sha256": text_sha256(saved["output"]),
            "bytes": len(saved["output"].encode("utf-8")),
        }
    # Inspection references the withheld text by digest; it never prints it.
    assert main(["runs", "--checkpoints", str(tmp_path), "inspect", result.execution_id]) == exit_code
    text = capsys.readouterr().out
    assert f"Delivery acceptance: {decision}" in text
    assert f"Validator fixed-{outcome} 2" in text
    assert saved["output"] not in text


def test_result_projection_matches_the_saved_run(tmp_path, capsys):
    for acceptance in (None, policy("pass"), policy("reject"), policy("raise")):
        result = run(tmp_path, acceptance)
        _, value = inspect(capsys, tmp_path, result.execution_id)
        projected = result.to_dict()
        delivery = projected["delivery"]
        if delivery["rejected_output"] is not None:
            # The result keeps withheld text in rejected_output; the checkpoint in output.
            assert delivery["rejected_output"]["field"] == "rejected_output"
            delivery["rejected_output"]["field"] = "output"
        assert delivery == value["delivery"]
        assert projected["next_action"] == value["next_action"]
        assert projected["execution"] == {"state": "completed", "nodes": value["execution"]["nodes"]}
        assert projected["unknown_exposure"] is None
        assert "acceptance" not in projected and "accepted" not in projected


def test_changed_output_is_indeterminate_whatever_the_record_says(tmp_path, capsys):
    result = run(tmp_path, policy("pass"))
    edit_checkpoint(tmp_path, result.execution_id,
                    lambda state: state.update(output=state["output"] + " (edited)"))
    before = files(tmp_path)

    code, value = inspect(capsys, tmp_path, result.execution_id)

    assert code == EXIT_RUN_INDETERMINATE
    assert files(tmp_path) == before
    delivery = value["delivery"]
    assert (delivery["decision"], delivery["integrity"]) == ("indeterminate", "mismatch")
    assert delivery["record"]["decision"] == "accepted"
    assert delivery["reasons"] == ["the saved output does not match the digest its record bound"]
    assert delivery["output_delivered"] is False
    assert value["next_action"]["action"] == "resolve_indeterminate"


def test_unfinished_strict_run_names_the_policy_that_must_resume_it(tmp_path, capsys, monkeypatch):
    acceptance = policy("pass")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run(tmp_path, acceptance, provider=Broken())
    (run_id,) = FileCheckpointStore(tmp_path).list_ids()
    no_side_effects(monkeypatch)

    code, value = inspect(capsys, tmp_path, run_id)

    assert code == EXIT_OK
    assert value["execution"]["state"] == "failed"
    delivery = value["delivery"]
    assert (delivery["recorded"], delivery["decision"], delivery["integrity"]) == (True, "not_evaluated", None)
    assert delivery["output_delivered"] is False and delivery["rejected_output"] is None
    assert value["next_action"]["action"] == "resume"
    assert acceptance.sha256 in value["next_action"]["detail"]


def test_inspection_shows_the_decision_history_and_exits_by_the_current_decision(
    tmp_path, capsys, monkeypatch,
):
    result = run(tmp_path, policy("reject"))
    swarm = Swarm(model="offline", provider=OfflineProvider(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path))
    accepted = swarm.reevaluate(result.execution_id, acceptance=policy("pass"))
    before = files(tmp_path)

    with monkeypatch.context() as patched:
        no_side_effects(patched)
        code, value = inspect(capsys, tmp_path, result.execution_id)
        assert main(["runs", "--checkpoints", str(tmp_path), "inspect", result.execution_id]) == code
        text = capsys.readouterr().out

    assert code == EXIT_OK and files(tmp_path) == before
    delivery = value["delivery"]
    assert (delivery["decision"], delivery["integrity"]) == ("accepted", "verified")
    assert delivery["record"] == accepted.acceptance.to_dict()
    assert value["decisions"] == {
        "current": 1, "history": [{"sequence": 0, "record": result.acceptance.to_dict()}]}
    assert value["next_action"]["action"] == "none"
    assert "Delivery acceptance: accepted" in text
    assert "Decision history: decision 1 is current; earlier decisions:" in text
    assert (f"  Decision 0: rejected under policy {policy('reject').sha256[:12]}: "
            "final validator 'fixed-reject' rejected the output") in text

    # A later decision that withholds the output sets the exit code again.
    swarm.reevaluate(result.execution_id, acceptance=policy("raise"))
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_RUN_INDETERMINATE
    assert value["decisions"]["current"] == 2
    assert [entry["record"]["decision"] for entry in value["decisions"]["history"]] == [
        "rejected", "accepted"]


@pytest.mark.parametrize("change,unresolved", [
    (lambda state: state["budget"].update(accounting_error="usage was malformed"),
     "the budget records an accounting error"),
    (lambda state: state["graph"]["nodes"][0]["metadata"].update(accounting_invalid=True),
     "invalid accounting on nodes ['draft']"),
    (lambda state: state["control"].update(response_error={"kind": "native"}),
     "a native response failure is unresolved"),
])
def test_unresolved_accounting_comes_before_resume(tmp_path, capsys, change, unresolved):
    result = run(tmp_path)

    def unfinished(state):
        state["status"] = "failed"
        change(state)

    edit_checkpoint(tmp_path, result.execution_id, unfinished)
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_OK
    assert value["accounting"]["unresolved"] == [unresolved]
    assert value["next_action"]["action"] == "reconcile_accounting"


def test_completed_run_with_unknown_cost_needs_reconciliation(tmp_path, capsys):
    result = run(tmp_path)
    edit_checkpoint(tmp_path, result.execution_id,
                    lambda state: state["graph"]["nodes"][1]["metadata"].update(cost_usd_unknown=True))
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_OK
    accounting = value["accounting"]
    assert (accounting["cost_is_complete"], accounting["unknown_cost_nodes"]) == (False, ["final"])
    assert value["next_action"]["action"] == "reconcile_accounting"


@pytest.mark.parametrize("change", [
    lambda state: state.update(version=6),
    lambda state: state.update(requires=["delivery_acceptance", "later_feature"]),
    lambda state: state.update(delivery_acceptance={"version": 2}),
    lambda state: state.update(delivery_acceptance_history=[{"sequence": 0}]),
])
def test_unreadable_checkpoint_fails_closed(tmp_path, capsys, change):
    result = run(tmp_path, policy("pass"))
    edit_checkpoint(tmp_path, result.execution_id, change)
    before = files(tmp_path)
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "inspect",
                 result.execution_id]) == EXIT_LOCAL_ERROR
    value = output(capsys)
    assert value["ok"] is False and value["error"]["type"] == "RunInspectionError"
    assert files(tmp_path) == before


def test_corrupt_json_is_a_local_error(tmp_path, capsys):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "inspect", "broken"]) == EXIT_LOCAL_ERROR
    assert output(capsys)["error"]["type"] == "RunInspectionError"


def unreadable_entries(directory: Path) -> None:
    """A directory named like a checkpoint, and a checkpoint nested too deeply to parse."""
    (directory / "folder.json").mkdir()
    (directory / "deep.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")


def test_entries_that_cannot_be_opened_or_parsed_are_local_errors(tmp_path, capsys):
    """An entry that raised OSError, or JSON deep enough to raise RecursionError,
    escaped as that exception rather than a report that the run is unreadable."""
    unreadable_entries(tmp_path)
    for run_id in ("folder", "deep"):
        assert main(["runs", "--checkpoints", str(tmp_path), "--json", "inspect", run_id]) == EXIT_LOCAL_ERROR
        error = output(capsys)["error"]
        assert error["type"] == "RunInspectionError"
        assert f"Checkpoint {run_id!r} is not readable" in error["message"]


def test_list_reports_entries_it_cannot_open_and_continues(tmp_path, capsys):
    """Listing used to stop at the first entry that raised OSError or RecursionError."""
    run_id = run(tmp_path).execution_id
    unreadable_entries(tmp_path)

    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "list"]) == EXIT_OK
    rows = {row["run_id"]: row for row in output(capsys)["list"]["runs"]}
    assert sorted(rows) == sorted(["deep", "folder", run_id])
    assert (rows[run_id]["state"], rows[run_id]["error"]) == ("completed", None)
    for unreadable in ("deep", "folder"):
        assert rows[unreadable]["state"] is None
        assert f"Checkpoint {unreadable!r} is not readable" in rows[unreadable]["error"]
    assert main(["runs", "--checkpoints", str(tmp_path), "list"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Runs: 3 of 3" in text and "folder  unreadable:" in text and "deep  unreadable:" in text


def test_unknown_and_invalid_run_ids(tmp_path, capsys):
    run(tmp_path)
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "inspect", "missing"]) == EXIT_JOB_STATE
    assert output(capsys)["error"]["type"] == "RunNotFoundError"
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "inspect", "../escape"]) == EXIT_INVALID_INPUT
    assert output(capsys)["error"]["type"] == "ValueError"


def test_missing_stores_are_not_created(tmp_path, capsys):
    directory, journal = tmp_path / "missing" / "checkpoints", tmp_path / "absent" / "runs.db"
    for option, path in (("--checkpoints", directory), ("--journal", journal)):
        for command in (["list"], ["inspect", "run-1"]):
            assert main(["runs", option, str(path), "--json", *command]) == EXIT_LOCAL_ERROR
            assert output(capsys)["ok"] is False
    assert not directory.parent.exists() and not journal.parent.exists()


def test_foreign_database_is_rejected_without_changes(tmp_path, capsys):
    path = tmp_path / "other.sqlite3"
    with closing(CONNECT(path)) as db, db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
    before = path.read_bytes()
    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_LOCAL_ERROR
    assert output(capsys)["error"]["type"] == "RunInspectionError"
    assert path.read_bytes() == before


def test_one_store_at_a_time(tmp_path, capsys):
    assert main(["runs", "--checkpoints", str(tmp_path), "--journal", str(tmp_path / "runs.db"),
                 "--json", "list"]) == EXIT_INVALID_INPUT
    assert "not both" in output(capsys)["error"]["message"]


def test_list_pages_runs_and_reports_unreadable_ones(tmp_path, capsys, monkeypatch):
    ids = sorted([run(tmp_path).execution_id, run(tmp_path, policy("reject")).execution_id])
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
    before = files(tmp_path)
    no_side_effects(monkeypatch)

    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "list"]) == EXIT_OK
    listed = output(capsys)["list"]
    assert (listed["store"], listed["total"], listed["returned"], listed["has_more"]) == ("checkpoints", 3, 3, False)
    rows = {row["run_id"]: row for row in listed["runs"]}
    assert rows["broken"]["error"] and rows["broken"]["state"] is None
    decisions = {rows[run_id]["decision"] for run_id in ids}
    assert decisions == {None, "rejected"}
    assert main(["runs", "--checkpoints", str(tmp_path), "--json", "list", "--limit", "1",
                 "--offset", "1"]) == EXIT_OK
    page = output(capsys)["list"]
    assert [row["run_id"] for row in page["runs"]] == [sorted(rows)[1]] and page["has_more"] is True
    for arguments in (["--limit", "0"], ["--limit", "501"], ["--offset", "-1"]):
        assert main(["runs", "--checkpoints", str(tmp_path), "--json", "list", *arguments]) == EXIT_INVALID_INPUT
        assert output(capsys)["error"]["type"] == "ValueError"
    assert files(tmp_path) == before


def durable_run(tmp_path) -> tuple[Path, str]:
    """A completed durable run in a version 1 journal without WAL sidecars."""
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1)
        run_id = swarm.execute(graph()).execution_id
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    mark_version_1(path)
    return path, run_id


def recorded_version(path: Path):
    with closing(CONNECT(path)) as db:
        return db.execute("SELECT version FROM workflow_meta").fetchall()


def test_durable_run_in_a_version_1_journal_is_read_without_migrating(tmp_path, capsys, monkeypatch):
    path, run_id = durable_run(tmp_path)
    before = path.read_bytes()
    statements = []

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    no_side_effects(monkeypatch)
    monkeypatch.setattr(sqlite3, "connect", connect)
    code = main(["runs", "--journal", str(path), "--json", "inspect", run_id])
    monkeypatch.setattr(sqlite3, "connect", CONNECT)
    value = output(capsys)["inspect"]

    assert code == EXIT_OK
    compact = ["".join(sql.upper().split()) for sql in statements]
    assert statements and "BEGINIMMEDIATE" not in compact
    assert not any(sql.startswith(("PRAGMAJOURNAL_MODE=", "CREATE", "INSERT", "UPDATE", "DELETE", "ALTER"))
                   for sql in compact)
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["runs.db"]
    assert recorded_version(path) == [(1,)]

    assert value["store"] == "journal"
    assert value["format"] == {"journal_version": 1, "checkpoint_version": 1,
                               "revision": value["format"]["revision"]}
    assert value["execution"]["state"] == "completed"
    assert value["execution"]["nodes"] == {"completed": 2}
    # A durable run without a strict policy has no record: shown as not recorded.
    assert value["delivery"]["recorded"] is False and value["delivery"]["decision"] is None
    assert value["delivery"]["output_delivered"] is True
    accounting = value["accounting"]
    assert accounting["cost_scope"] == "complete_text_workflow"
    assert accounting["workflow_accounting"]["unit"] == "nanoUSD"
    assert accounting["workflow_accounting"]["call_count"] == 2
    assert accounting["unknown_exposure"] == {"unit": "nanoUSD", "amount": 0, "calls": 0}
    assert value["timing"]["clock"] == "wall" and value["timing"]["segments"] >= 1
    assert value["timing"]["phases_ms"] is None
    assert value["next_action"]["action"] == "none"


def test_reading_a_wal_journal_leaves_its_database_file_unchanged(tmp_path, capsys):
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1)
        run_id = swarm.execute(graph()).execution_id
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    version = recorded_version(path)
    before = path.read_bytes()

    assert main(["runs", "--journal", str(path), "--json", "inspect", run_id]) == EXIT_OK
    assert output(capsys)["inspect"]["execution"]["state"] == "completed"
    # SQLite may leave its WAL sidecars; the database itself is not written.
    assert {item.name for item in tmp_path.iterdir()} <= {"runs.db", "runs.db-wal", "runs.db-shm"}
    assert path.read_bytes() == before and recorded_version(path) == version


def test_blocked_durable_runs_name_the_action_they_need(tmp_path, capsys, monkeypatch):
    path = tmp_path / "workflow.db"
    build_journal(path, ManualClock())
    with closing(CONNECT(path)) as db:
        db.execute("PRAGMA journal_mode=DELETE")
    mark_version_1(path)
    before = path.read_bytes()
    no_side_effects(monkeypatch)

    expected = {
        "unknown": ("blocked", "reconcile_accounting"),
        "overrun": ("blocked", "rerun"),
        "interrupted": ("running", "resume"),
        "accepted": ("running", "resume"),
        "rejected": ("running", "resume"),
    }
    for run_id, (state, action) in expected.items():
        assert main(["runs", "--journal", str(path), "--json", "inspect", run_id]) == EXIT_OK
        value = output(capsys)["inspect"]
        assert (value["execution"]["state"], value["next_action"]["action"]) == (state, action), run_id
        assert value["delivery"]["recorded"] is False and value["delivery"]["output_delivered"] is False

    assert main(["runs", "--journal", str(path), "--json", "inspect", "unknown"]) == EXIT_OK
    exposure = output(capsys)["inspect"]["accounting"]
    assert exposure["unknown_exposure"]["calls"] == 1 and exposure["unknown_exposure"]["amount"] > 0
    assert exposure["cost_is_complete"] is False
    assert exposure["unresolved"] == ["admission is closed by unknown_exposure"]
    assert main(["runs", "--journal", str(path), "inspect", "unknown"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Execution: blocked (unknown_exposure)" in text
    assert f"Unknown exposure: {exposure['unknown_exposure']['amount']} nanoUSD on 1 calls" in text
    assert "Next action: reconcile_accounting" in text

    # The fixture's completed run saved a checkpoint without a version, which
    # this build reports as unreadable instead of guessing what it holds.
    assert main(["runs", "--journal", str(path), "--json", "inspect", "completed"]) == EXIT_LOCAL_ERROR
    assert output(capsys)["error"]["type"] == "RunInspectionError"
    assert main(["runs", "--journal", str(path), "--json", "inspect", "missing"]) == EXIT_JOB_STATE
    assert output(capsys)["error"]["type"] == "RunNotFoundError"

    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_OK
    listed = output(capsys)["list"]
    assert listed["total"] == 6 and listed["store"] == "journal"
    errors = [row["run_id"] for row in listed["runs"] if row["error"]]
    assert errors == ["completed"]
    assert main(["runs", "--journal", str(path), "list"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Runs: 6 of 6" in text and "completed  unreadable:" in text
    assert "overrun  blocked  acceptance not recorded  next rerun" in text
    assert path.read_bytes() == before
    assert recorded_version(path) == [(1,)]


# ---------------------------------------------------------------------------
# Durable runs under a strict policy
# ---------------------------------------------------------------------------


DECISIONS = {
    "accepted": policy("pass"),
    "rejected": policy("reject"),
    "needs_review": policy("raise", on_invalid="needs_review"),
    "indeterminate": policy("raise"),
}


def gated() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft the answer"),
        Node(id="judge", label="Judge the draft", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])


@pytest.fixture(scope="module")
def strict_journal(tmp_path_factory):
    """A journal, built once, with a completed durable run for each strict
    decision, a run without a policy, a run with only a gate validator and an
    unfinished strict run, saved without WAL sidecars."""
    path = tmp_path_factory.mktemp("strict-journal") / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        def durable(**options):
            return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                         run_store=store, max_budget_usd=1, **options)

        results = {decision: durable(acceptance=acceptance).execute(graph())
                   for decision, acceptance in DECISIONS.items()}
        results["legacy"] = durable().execute(graph())
        validator = LocalOnly(lambda: Fixed("pass"), "fixed-pass", "2", role="validator")
        results["gate-validator"] = durable(verifier=validator).execute(gated())
        planned = durable(acceptance=policy("pass")).plan(Task("Write the answer"))
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    return path, results, planned.run_ref["run_id"]


def journal_copy(strict_journal, tmp_path) -> Path:
    return Path(shutil.copyfile(strict_journal[0], tmp_path / "runs.db"))


def journal_state(path: Path):
    """The journal's bytes and rows, and the files beside it."""
    return path.read_bytes(), dump(path), sorted(item.name for item in path.parent.iterdir())


def inspect_durable(capsys, path, run_id):
    code = main(["runs", "--journal", str(path), "--json", "inspect", run_id])
    value = output(capsys)
    assert value["ok"] is True and value["command"] == "inspect"
    return code, value["inspect"]


def saved_checkpoint(path: Path, run_id: str) -> dict:
    with SQLiteWorkflowStore(path, read_only=True) as store:
        return store.get_checkpoint(run_id)["checkpoint"]


@pytest.mark.parametrize("decision,exit_code,action", [
    ("accepted", EXIT_OK, "none"),
    ("rejected", EXIT_RUN_NOT_ACCEPTED, "review_output"),
    ("needs_review", EXIT_RUN_NOT_ACCEPTED, "review_output"),
    ("indeterminate", EXIT_RUN_INDETERMINATE, "resolve_indeterminate"),
])
def test_durable_strict_decisions_have_the_same_exit_codes(
    strict_journal, tmp_path, capsys, monkeypatch, decision, exit_code, action,
):
    path, results = journal_copy(strict_journal, tmp_path), strict_journal[1]
    result = results[decision]
    saved = saved_checkpoint(path, result.execution_id)
    before = journal_state(path)
    no_side_effects(monkeypatch)

    code, value = inspect_durable(capsys, path, result.execution_id)

    assert code == exit_code
    assert journal_state(path) == before
    assert value["format"]["checkpoint_version"] == saved["version"] == 2
    assert value["execution"]["state"] == "completed"
    delivery = value["delivery"]
    assert (delivery["recorded"], delivery["decision"], delivery["integrity"]) == (True, decision, "verified")
    assert delivery["record"] == saved["delivery_acceptance"] == result.acceptance.to_dict()
    assert delivery["record"]["output_sha256"] == text_sha256(saved["output"])
    assert delivery["record"]["policy_sha256"] == DECISIONS[decision].sha256
    outcome = {"accepted": "pass", "rejected": "reject"}.get(decision, "raise")
    assert delivery["record"]["validators"][0]["validator"] == Fixed(outcome).identity.to_dict()
    assert value["decisions"] == {"current": 0, "history": []}
    assert value["next_action"]["action"] == action
    if decision == "accepted":
        assert delivery["output_delivered"] is True and delivery["rejected_output"] is None
    else:
        assert delivery["output_delivered"] is False
        assert delivery["rejected_output"] == {
            "field": "output", "sha256": text_sha256(saved["output"]),
            "bytes": len(saved["output"].encode("utf-8")),
        }
    # Inspection reports what the run returned.
    projected = result.to_dict()
    if projected["delivery"]["rejected_output"] is not None:
        projected["delivery"]["rejected_output"]["field"] = "output"
    assert projected["delivery"] == delivery
    assert projected["next_action"] == value["next_action"]

    assert main(["runs", "--journal", str(path), "inspect", result.execution_id]) == exit_code
    text = capsys.readouterr().out
    assert f"Delivery acceptance: {decision}" in text
    assert f"Validator fixed-{outcome} 2" in text
    assert f"Output: sha256 {text_sha256(saved['output'])}" in text
    assert saved["output"] not in text
    assert journal_state(path) == before


@pytest.mark.parametrize("change,reason", [
    ("output", "the saved output does not match the digest its record bound"),
    ("output-and-record", "the journal holds no applied decision matching the saved record"),
])
def test_changed_durable_output_is_indeterminate_as_on_resume(strict_journal, tmp_path, capsys, change, reason):
    path = journal_copy(strict_journal, tmp_path)
    run_id = strict_journal[1]["accepted"].execution_id
    tampered = "text that no validator checked"

    def rewrite(checkpoint):
        checkpoint["output"] = tampered
        if change == "output-and-record":
            checkpoint["delivery_acceptance"]["output_sha256"] = text_sha256(tampered)

    rewrite_checkpoint(path, run_id, rewrite)
    before = journal_state(path)

    code, value = inspect_durable(capsys, path, run_id)

    assert code == EXIT_RUN_INDETERMINATE
    assert journal_state(path) == before
    delivery = value["delivery"]
    assert (delivery["decision"], delivery["integrity"]) == ("indeterminate", "mismatch")
    assert delivery["record"]["decision"] == "accepted"
    assert delivery["reasons"] == [reason]
    assert delivery["output_delivered"] is False
    assert delivery["rejected_output"]["sha256"] == text_sha256(tampered)
    assert value["next_action"]["action"] == "resolve_indeterminate"
    # Resuming the run reaches the same decision for the same reasons.
    with SQLiteWorkflowStore(path) as store:
        resumed = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                        run_store=store, max_budget_usd=1, acceptance=policy("pass")).resume(run_id)
    assert resumed.acceptance.decision.value == "indeterminate"
    assert list(resumed.acceptance.reasons) == delivery["reasons"]


def test_unfinished_durable_strict_run_names_the_policy_that_must_resume_it(
    strict_journal, tmp_path, capsys, monkeypatch,
):
    path, run_id = journal_copy(strict_journal, tmp_path), strict_journal[2]
    before = journal_state(path)
    no_side_effects(monkeypatch)

    code, value = inspect_durable(capsys, path, run_id)

    assert code == EXIT_OK
    assert journal_state(path) == before
    assert value["execution"]["state"] == "planned"
    delivery = value["delivery"]
    assert (delivery["recorded"], delivery["decision"], delivery["integrity"]) == (True, "not_evaluated", None)
    assert delivery["output_delivered"] is False and delivery["rejected_output"] is None
    assert delivery["record"]["policy_sha256"] == policy("pass").sha256
    assert value["next_action"]["action"] == "resume"
    assert policy("pass").sha256 in value["next_action"]["detail"]


@pytest.mark.parametrize("name,version", [("legacy", 1), ("gate-validator", 2)])
def test_durable_runs_without_a_strict_policy_have_no_record(strict_journal, tmp_path, capsys, name, version):
    path, result = journal_copy(strict_journal, tmp_path), strict_journal[1][name]
    before = journal_state(path)

    code, value = inspect_durable(capsys, path, result.execution_id)

    assert code == EXIT_OK
    assert journal_state(path) == before
    assert value["format"]["checkpoint_version"] == version
    assert value["delivery"] == {"recorded": False, "decision": None, "reasons": [], "integrity": None,
                                 "output_delivered": True, "rejected_output": None, "record": None}
    assert value["decisions"] == {"current": None, "history": []}
    assert value["next_action"]["action"] == "none"


def test_durable_inspection_shows_the_journal_decision_history(
    strict_journal, tmp_path, capsys, monkeypatch,
):
    path, original = journal_copy(strict_journal, tmp_path), strict_journal[1]["rejected"]
    run_id = original.execution_id
    with SQLiteWorkflowStore(path) as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1)
        accepted = swarm.reevaluate(run_id, acceptance=policy("pass"))
        undecided = swarm.reevaluate(run_id, acceptance=policy("raise"))
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    before = journal_state(path)

    with monkeypatch.context() as patched:
        no_side_effects(patched)
        code, value = inspect_durable(capsys, path, run_id)
        assert main(["runs", "--journal", str(path), "inspect", run_id]) == code
        text = capsys.readouterr().out

    assert code == EXIT_RUN_INDETERMINATE and journal_state(path) == before
    delivery = value["delivery"]
    assert (delivery["decision"], delivery["integrity"]) == ("indeterminate", "verified")
    assert delivery["record"] == undecided.acceptance.to_dict()
    assert value["decisions"] == {"current": 2, "history": [
        {"sequence": 0, "record": original.acceptance.to_dict()},
        {"sequence": 1, "record": accepted.acceptance.to_dict()},
    ]}
    assert "Decision history: decision 2 is current; earlier decisions:" in text
    assert f"  Decision 1: accepted under policy {policy('pass').sha256[:12]}: " in text
    # Listing reports the current decision.
    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_OK
    rows = {row["run_id"]: row["decision"] for row in output(capsys)["list"]["runs"]}
    assert rows[run_id] == "indeterminate"


@pytest.mark.parametrize("change,message", [
    (lambda state: state.update(version=1), "checkpoint version 1, but its recipe writes version 2"),
    (lambda state: state.update(version=3), "it reads versions 1 and 2"),
    (lambda state: state.pop("delivery_acceptance"), "no delivery acceptance block"),
    (lambda state: state["delivery_acceptance"].update(decision="not_evaluated"), "does not match its run"),
    (lambda state: state["delivery_acceptance"].update(policy_sha256="0" * 64), "does not match its run"),
    (lambda state: state["delivery_acceptance"].update(version=2), "delivery acceptance is invalid"),
])
def test_a_durable_checkpoint_that_resume_would_refuse_is_unreadable(
    strict_journal, tmp_path, capsys, change, message,
):
    path = journal_copy(strict_journal, tmp_path)
    run_id = strict_journal[1]["accepted"].execution_id
    rewrite_checkpoint(path, run_id, change)
    before = journal_state(path)

    assert main(["runs", "--journal", str(path), "--json", "inspect", run_id]) == EXIT_LOCAL_ERROR
    error = output(capsys)["error"]
    assert error["type"] == "RunInspectionError" and message in error["message"]
    assert journal_state(path) == before


def test_list_shows_durable_decisions(strict_journal, tmp_path, capsys, monkeypatch):
    path, (_, results, planned) = journal_copy(strict_journal, tmp_path), strict_journal
    before = journal_state(path)
    no_side_effects(monkeypatch)

    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_OK
    listed = output(capsys)["list"]
    rows = {row["run_id"]: (row["state"], row["decision"], row["error"]) for row in listed["runs"]}
    expected = {results[name].execution_id: ("completed", decision, None) for name, decision in (
        ("accepted", "accepted"), ("rejected", "rejected"), ("needs_review", "needs_review"),
        ("indeterminate", "indeterminate"), ("legacy", None), ("gate-validator", None))}
    assert rows == expected | {planned: ("planned", "not_evaluated", None)}
    assert main(["runs", "--journal", str(path), "list"]) == EXIT_OK
    text = capsys.readouterr().out
    assert f"{results['rejected'].execution_id}  completed  acceptance rejected  next review_output" in text
    assert journal_state(path) == before


# ---------------------------------------------------------------------------
# Durable research runs: context, reads, evidence and open issues
# ---------------------------------------------------------------------------


class Research:
    """A journal, a source service on loopback, and research runs over it."""

    def __init__(self, path: Path, port: int, replies: dict[str, str]) -> None:
        self.path, self.replies = path, replies
        self.profile = ReadProfile(
            name="source", version="1", url=f"http://127.0.0.1:{port}/sources/{{source_id}}",
            input_schema={"source_id": "string"}, metered=False, repeat_safe=True,
            decode=lambda body: json.loads(body)["content"], decoder_version="fixture-source-v1",
            allow_private_networks=True)

    def swarm(self, store) -> Swarm:
        return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                     run_store=store, max_budget_usd=1, read_tools=[self.profile],
                     context_sources=EVIDENCE_SOURCES, acceptance=AcceptancePolicy(
                         mode="strict", final_validators=[SourceEvidenceValidator(EVIDENCE_SOURCES)]))

    def run(self, output: str, *sources: str):
        """Read *sources*, each a skipped step when it fails, then write *output*."""
        self.replies["Write"] = output
        reads = [self.profile.step(f"fetch-{source_id}", {"source_id": source_id},
                                   failure_policy=FailurePolicy.SKIP) for source_id in sources]
        graph = ExecutionGraph([Topology.SERIAL], [
            *reads, Node(id="write", label="Write", depends_on=[read.id for read in reads])])
        with SQLiteWorkflowStore(self.path) as store:
            return self.swarm(store).execute(graph)


@pytest.fixture
def research(tmp_path, monkeypatch):
    service = FixtureService()
    for source in EVIDENCE_SOURCES:
        service.add_source(source.source_id, source.text)
    replies: dict[str, str] = {}

    async def complete(self, system, prompt, model):
        return CompletionResult(replies["Write"], cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    with serve_http(service) as endpoint:
        yield Research(tmp_path / "runs.db", endpoint.port, replies)


def test_a_research_run_shows_its_context_reads_evidence_and_open_issues(research, capsys,
                                                                         monkeypatch):
    result = research.run(evidence_artifact(KEPT), "survey-2019", "renewals-2026")
    assert result.accepted
    run_id, path = result.execution_id, research.path
    before = journal_state(path)
    no_side_effects(monkeypatch)

    code, value = inspect_durable(capsys, path, run_id)
    assert code == EXIT_OK and journal_state(path) == before
    context = value["context"]
    assert (context["assembly_version"], context["snapshot_version"],
            context["sources_revision"]) == (1, 1, 0)
    assert context["sources"] == [
        {"source_id": source.source_id, "version": source.version, "sha256": source.sha256,
         "text_bytes": len(source.text.encode()), "trust": "untrusted", "removed": None}
        for source in EVIDENCE_SOURCES]
    (snapshot,) = context["snapshots"]
    assert (snapshot["node_id"], snapshot["generation"]) == ("write", 0)
    assert context["tombstones"] == []

    reads = {call["node_id"]: call for call in value["tool_calls"]}
    assert sorted(reads) == ["fetch-renewals-2026", "fetch-survey-2019"]
    audit = reads["fetch-renewals-2026"]
    assert (audit["profile"], audit["profile_version"], audit["decoder_version"]) == (
        "source", "1", "fixture-source-v1")
    assert (audit["state"], audit["result_state"], audit["http_status"],
            audit["cost_nanousd"]) == ("settled", "applied", 200, 0)
    assert (audit["text_bytes"], audit["text_sha256"]) == (len(STRONG.text.encode()), STRONG.sha256)
    assert audit["url"].endswith("/sources/renewals-2026") and audit["error"] is None

    evidence = value["evidence"]
    assert evidence["state"] == "applied" and evidence["problem"] is None
    assert [(item["source_id"], item["available"]) for item in evidence["sources"]] == [
        ("survey-2019", True), ("renewals-2026", True)]
    assert evidence["claims"] == [{
        "claim_id": "billing-preference", "supporting": 1, "contradicting": 1,
        "sources": ["renewals-2026", "survey-2019"], "scope": "Customers on self-serve plans",
        "unresolved": ["Whom did the 2019 poll ask?"]}]
    assert value["issues"] == [{"kind": "unresolved_question", "subject": "billing-preference",
                                "detail": "Whom did the 2019 poll ask?"}]

    assert main(["runs", "--journal", str(path), "inspect", run_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Context: assembly version 1, snapshot version 1; sources revision 0; 2 sources" in text
    assert ("Read fetch-renewals-2026 (generation 0, attempt 0): source 1; settled, applied; "
            "HTTP 200") in text
    assert "Claim billing-preference: 1 supporting, 1 contradicting" in text
    assert "Issue unresolved_question (billing-preference): Whom did the 2019 poll ask?" in text
    # No source text, read body, quote or output is shown.
    for shown in (text, json.dumps(value)):
        assert all(source.text not in shown for source in EVIDENCE_SOURCES)
        assert CLAIM.contradicting[0].quote not in shown and KEPT.strip() not in shown


def test_inspection_names_failed_reads_rejections_and_removed_sources(research, capsys):
    # One read finds no such source, and the output's evidence is rejected.
    result = research.run(evidence_artifact(DROPPED), "renewals-2026", "missing")
    run_id, path = result.execution_id, research.path
    assert result.acceptance.decision is AcceptanceDecision.REJECTED

    code, value = inspect_durable(capsys, path, run_id)
    assert code == EXIT_RUN_NOT_ACCEPTED
    failed = next(call for call in value["tool_calls"] if call["node_id"] == "fetch-missing")
    assert (failed["result_state"], failed["http_status"], failed["text_bytes"]) == (
        "rejected", 404, None)
    kinds = [(issue["kind"], issue["subject"]) for issue in value["issues"]]
    assert ("read_failed", "fetch-missing") in kinds
    assert ("validator_rejected", "smythe.source-evidence") in kinds
    rejected = next(issue for issue in value["issues"] if issue["kind"] == "validator_rejected")
    assert "contradiction_missing" in rejected["detail"]

    with SQLiteWorkflowStore(path) as store:
        research.swarm(store).remove_source(run_id, "renewals-2026", "deleted on request",
                                            removed_at="2026-10-01")
    code, value = inspect_durable(capsys, path, run_id)
    assert code == EXIT_RUN_INDETERMINATE
    removal = {"reason": "deleted on request", "removed_at": "2026-10-01"}
    audit = next(source for source in value["context"]["sources"]
                 if source["source_id"] == "renewals-2026")
    assert (audit["removed"], audit["text_bytes"]) == (removal, None)
    assert [item["source_id"] for item in value["context"]["tombstones"]] == ["renewals-2026"]
    cited = next(source for source in value["evidence"]["sources"]
                 if source["source_id"] == "renewals-2026")
    assert (cited["available"], cited["removed"]) == (False, removal)
    assert {"kind": "source_unavailable", "subject": "renewals-2026",
            "detail": "removed from the run: deleted on request"} in value["issues"]
    assert any("was removed from the run after this decision" in reason
               for reason in value["delivery"]["reasons"])


def test_ordinary_runs_have_no_durable_research_blocks(tmp_path, capsys):
    result = run(tmp_path, policy("reject"))
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_RUN_NOT_ACCEPTED
    assert (value["context"], value["tool_calls"], value["evidence"]) == (None, None, None)
    assert value["lineage"] is None
    assert value["issues"] == [{"kind": "validator_rejected", "subject": "fixed-reject",
                                "detail": value["delivery"]["reasons"][0]}]


# ---------------------------------------------------------------------------
# Supervisor revisions
# ---------------------------------------------------------------------------


class Revises(Supervisor):
    """Adds a check after the final step, then proposes to drop the finished draft."""

    async def review(self, graph, node, *, task, revisions_remaining):
        if node.id == "draft":
            check = Node(id="check", label="Check the answer", depends_on=["final"])
            return Revision(add_nodes=(check,), reason="the answer needs a check")
        if node.id == "final":
            return Revision(drop_node_ids=("draft",), reason="the draft is done")
        return None


# What both kinds of run log for those two reviews, apart from the refusal's detail.
REVISIONS = [
    {"after_node": "draft", "applied": True, "change": "+1", "added": ["check"], "dropped": [],
     "rewired": [], "retained": ["draft"], "invalidated": [], "preserved_gates": [],
     "revisions_remaining_after": 1, "added_total": 1, "trigger": "node_completed",
     "reason": "the answer needs a check", "detail": None},
    {"after_node": "final", "applied": False, "change": "-1", "added": [], "dropped": ["draft"],
     "rewired": [], "retained": ["draft", "final"], "invalidated": [], "preserved_gates": [],
     "revisions_remaining_after": 1, "added_total": 1, "trigger": "node_completed",
     "reason": "the draft is done", "detail": None},
]


def test_inspection_shows_an_ordinary_run_s_revision_log(tmp_path, capsys, monkeypatch):
    swarm = Swarm(model="offline", provider=OfflineProvider(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path), supervisor=Revises(),
                  max_revisions=2)
    result = swarm.execute(graph())
    saved = json.loads((tmp_path / f"{result.execution_id}.json").read_text())
    no_side_effects(monkeypatch)

    code, value = inspect(capsys, tmp_path, result.execution_id)

    assert code == EXIT_OK
    assert value["revision_log"] == saved["control"]["revision_log"]
    applied, refused = value["revision_log"]
    assert applied == REVISIONS[0]
    assert refused == REVISIONS[1] | {"detail": refused["detail"]}
    assert refused["detail"].startswith("cannot drop node 'draft' with status completed")
    assert main(["runs", "--checkpoints", str(tmp_path), "inspect", result.execution_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Revisions: 2 logged" in text
    assert ("Revision after draft: applied (+1); added check; 1 revisions left, "
            "1 added nodes in the graph") in text
    assert "  Reason: the answer needs a check" in text
    assert "Revision after final: refused (-1); dropped draft; 1 revisions left" in text
    assert "  Detail: cannot drop node 'draft' with status completed" in text


def test_a_run_without_revisions_shows_an_empty_revision_log(tmp_path, capsys):
    result = run(tmp_path)
    code, value = inspect(capsys, tmp_path, result.execution_id)
    assert code == EXIT_OK and value["revision_log"] == []
    main(["runs", "--checkpoints", str(tmp_path), "inspect", result.execution_id])
    assert "Revision" not in capsys.readouterr().out


def test_inspection_derives_a_durable_run_s_revision_log_from_its_supervision_operations(
    tmp_path, capsys, monkeypatch,
):
    proposals = [
        {"change": True, "reason": "the answer needs a check",
         "add": [{"id": "check", "label": "Check the answer", "depends_on": ["final"]}]},
        {"change": True, "reason": "the draft is done", "drop": ["draft"]},
        {"change": False, "reason": "nothing left to change"},
    ]

    async def complete(self, system, prompt, model):
        if system == SUPERVISOR_SYSTEM_PROMPT:
            return CompletionResult(json.dumps(proposals.pop(0)), cost_usd=0)
        return CompletionResult("Done", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        provider = OfflineProvider()
        swarm = Swarm(model="offline", provider=provider, architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1, max_revisions=2,
                      supervisor=LLMSupervisor(provider, model="offline", only_terminal=False))
        result = swarm.execute(graph())
        run_id = result.execution_id
        operations = store.list_operations(run_id, "supervision")
        checkpoint = store.get_checkpoint(run_id)["checkpoint"]
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    before = journal_state(path)
    no_side_effects(monkeypatch)

    code, value = inspect_durable(capsys, path, run_id)

    assert code == EXIT_OK and journal_state(path) == before
    # The journal keeps no log: three reviews, the last of which changed nothing.
    assert [operation["operation_key"] for operation in operations] == [
        "supervision/draft/0", "supervision/final/0", "supervision/check/0"]
    assert "revision_log" not in checkpoint
    assert value["revision_log"] == REVISIONS
    # The trace of the run that made the decisions records the same effects.
    traced = [(span["status"], span["revision"]) for span in result.trace if "revision" in span]
    assert [status for status, _ in traced] == ["revision_applied", "revision_rejected"]
    for (_, record), entry in zip(traced, value["revision_log"]):
        assert record == {key: entry[key] for key in record}
    main(["runs", "--journal", str(path), "inspect", run_id])
    text = capsys.readouterr().out
    assert "Revision after draft: applied (+1); added check" in text
    assert "Revision after final: refused (-1); dropped draft" in text


class SupervisionJournal:
    """The reads the durable revision log makes, over hand-written supervision operations."""

    def __init__(self, operations, *, max_revisions, revisions_used):
        self.operations, self.max_revisions, self.revisions_used = operations, max_revisions, revisions_used

    def list_operations(self, run_id, kind):
        assert kind == "supervision"
        return self.operations

    def get_checkpoint(self, run_id):
        return {"checkpoint": {"revisions_used": self.revisions_used}}

    def load_run(self, run_id):
        return {"config": {"max_revisions": self.max_revisions}}


def supervision(node_id, remaining, graph, result, *, applied=True):
    from smythe.checkpoint import graph_to_dict

    return {"operation_key": f"supervision/{node_id}/0", "applied_revision": 1 if applied else None,
            "inputs": {"graph": graph_to_dict(graph), "node_id": node_id, "task": None,
                       "revisions_remaining": remaining},
            "result": result}


def reviewed(*completed: str, extra: bool = False) -> ExecutionGraph:
    """Two independent steps, ``a`` and ``b``, and the added ``x`` when *extra*."""
    nodes = [Node(id=node_id, label=node_id) for node_id in ("a", "b")]
    if extra:
        nodes.append(Node(id="x", label="x", depends_on=["a"], metadata={"added_by_revision": True}))
    for node in nodes:
        if node.id in completed:
            node.status = NodeStatus.COMPLETED
    return ExecutionGraph([Topology.FORK_JOIN], nodes)


def test_the_durable_revision_log_is_derived_from_supervision_operations():
    from smythe.workflow import revision_log

    add_x = {"add_nodes": [{"id": "x", "label": "x", "depends_on": ["a"]}], "drop_node_ids": [],
             "rewire": {}, "reason": "a needs a follow-up"}
    operations = [
        supervision("a", 2, reviewed("a"), {"revision": add_x}),
        supervision("b", 1, reviewed("a", "b", extra=True),
                    {"revision": None, "local_error_type": "RuntimeError"}),
        supervision("x", 1, reviewed("a", "b", "x", extra=True), {"revision": None}),
        # Saved but not yet applied by a checkpoint: resume decides it again.
        supervision("y", 1, reviewed("a", "b", "x", extra=True), {"revision": add_x}, applied=False),
    ]
    journal = SupervisionJournal(operations, max_revisions=2, revisions_used=1)

    applied, raised = revision_log(journal, "run")

    assert applied == {
        "after_node": "a", "applied": True, "change": "+1", "added": ["x"], "dropped": [],
        "rewired": [], "retained": ["a"], "invalidated": ["a"], "preserved_gates": [],
        "revisions_remaining_after": 1, "added_total": 1, "trigger": "node_completed",
        "reason": "a needs a follow-up", "detail": None}
    assert raised == {
        "after_node": "b", "applied": False, "change": "unknown", "added": [], "dropped": [],
        "rewired": [], "retained": ["a", "b"], "invalidated": [], "preserved_gates": [],
        "revisions_remaining_after": 1, "added_total": 1, "trigger": "node_completed",
        "reason": None, "detail": "supervisor raised RuntimeError"}

    # An allowance that does not step down by none or one cannot say what happened.
    operations[2]["inputs"]["revisions_remaining"] = 3
    assert [entry["applied"] for entry in revision_log(journal, "run")] == [True, None]

    # Only the latest decisions are kept.
    refused = {"add_nodes": [], "drop_node_ids": ["a"], "rewire": {}, "reason": "again"}
    journal.operations = [supervision(f"n{index}", 1, reviewed("a"), {"revision": refused})
                          for index in range(40)]
    journal.revisions_used = 1
    log = revision_log(journal, "run")
    assert len(log) == 32 and log[0]["after_node"] == "n8"
    assert {entry["applied"] for entry in log} == {False}


def test_a_durable_review_that_proposed_an_empty_revision_is_not_logged():
    from smythe.workflow import revision_log

    nothing = {"add_nodes": [], "drop_node_ids": [], "rewire": {}, "reason": "nothing to change"}
    add_x = {"add_nodes": [{"id": "x", "label": "x", "depends_on": ["a"]}], "drop_node_ids": [],
             "rewire": {}, "reason": "a needs a follow-up"}
    journal = SupervisionJournal([supervision("a", 2, reviewed("a"), {"revision": nothing})],
                                 max_revisions=2, revisions_used=0)

    assert revision_log(journal, "run") == []

    # The allowance still steps across the review that changed nothing.
    journal.operations.append(supervision("b", 2, reviewed("a", "b"), {"revision": add_x}))
    journal.revisions_used = 1
    assert [(entry["after_node"], entry["applied"], entry["change"])
            for entry in revision_log(journal, "run")] == [("b", True, "+1")]


class Approves(Supervisor):
    """Reviews every step and proposes an empty revision, which changes nothing."""

    async def review(self, graph, node, *, task, revisions_remaining):
        return Revision(reason="looks fine, nothing to change")


def test_a_review_that_proposed_an_empty_revision_is_logged_by_neither_kind_of_run(
    tmp_path, capsys,
):
    checkpoints = tmp_path / "checkpoints"
    swarm = Swarm(model="offline", provider=OfflineProvider(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(checkpoints), supervisor=Approves(),
                  max_revisions=2)
    ordinary = swarm.execute(graph())
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1, max_revisions=2,
                      supervisor=LocalOnly(Approves, "approves", "1", role="supervisor"))
        durable = swarm.execute(graph())
        operations = store.list_operations(durable.execution_id, "supervision")
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)

    # The journal keeps each empty proposal; neither run traced a revision.
    assert [operation["result"]["revision"] for operation in operations] == [
        {"add_nodes": [], "drop_node_ids": [], "rewire": {}, "reason": "looks fine, nothing to change"},
    ] * 2
    for result in (ordinary, durable):
        assert [span for span in result.trace if "revision" in span] == []
    code, value = inspect(capsys, checkpoints, ordinary.execution_id)
    assert code == EXIT_OK and value["revision_log"] == []
    code, value = inspect_durable(capsys, path, durable.execution_id)
    assert code == EXIT_OK and value["revision_log"] == []
    main(["runs", "--journal", str(path), "inspect", durable.execution_id])
    assert "Revision" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Durable runs with artifact lineage
# ---------------------------------------------------------------------------


def test_a_lineage_run_shows_a_damaged_artifact_and_its_lineage_without_text(
        tmp_path, capsys, monkeypatch):
    service = FixtureService()
    records = {key: service.add_source(key, text) for key, text in lineage_runs.TEXTS.items()}
    path = tmp_path / "runs.db"
    with serve_http(service) as endpoint, SQLiteWorkflowStore(path) as store:
        library = (service, records, endpoint)
        result = lineage_runs.research(store, library, tmp_path).execute(
            lineage_runs.research_graph(library))
        run_id, record = result.execution_id, load_lineage(store, result.execution_id)
        published = current_artifact(store, run_id).publication.path
    published.write_bytes(published.read_bytes()[:-1] + b"?")
    before = journal_state(path)
    no_side_effects(monkeypatch)

    code, value = inspect_durable(capsys, path, run_id)

    # The damaged document makes the decision indeterminate, with its existing exit code.
    assert code == EXIT_RUN_INDETERMINATE and journal_state(path) == before
    served = [{"source_id": key, "version": "1", "sha256": records[key].sha256}
              for key in sorted(records)]
    assert value["lineage"] == {
        "access_scope": "research-team", "recorded": True, "lineage_sha256": record.sha256,
        "coverage": "complete", "nodes": {"model": 2, "read": 2, "synthesis": 1},
        "undeclared": [], "sources": served,
        "artifact": {"relative_path": f"{run_id}/document.md", "sha256": record.artifact["sha256"],
                     "size_bytes": record.artifact["size_bytes"], "integrity": "changed"},
        "predecessor": None, "successors": [], "tombstones": [], "purged": [],
        "status": "invalid", "reasons": ["artifact_changed", "decision_not_accepted"]}
    assert main(["runs", "--journal", str(path), "inspect", run_id]) == code
    text = capsys.readouterr().out
    assert (f"Lineage: complete coverage; scope research-team; nodes model 2, read 2, "
            f"synthesis 1; record sha256 {record.sha256}") in text
    assert f"  Source renewals version 1: sha256 {records['renewals'].sha256[:12]}" in text
    assert "; integrity changed" in text and "  Predecessor: none; successors: none" in text
    assert "  Derived status: invalid (artifact_changed, decision_not_accepted)" in text
    # Ids, versions and digests only: no source text, and no agent or model result digest.
    hidden = [digest for node in record.nodes if node["kind"] == "model"
              for digest in (node["inputs"]["agent_sha256"], node["result_sha256"])]
    for shown in (text, json.dumps(value)):
        assert all(source not in shown for source in lineage_runs.TEXTS.values())
        assert len(hidden) == 4 and not any(digest in shown for digest in hidden)
    # Listing does not read the lineage.
    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_OK
    assert output(capsys)["list"]["runs"][0]["decision"] == "indeterminate"
    assert journal_state(path) == before


def test_lineage_names_predecessors_and_the_successors_that_supersede_them(tmp_path, capsys):
    path = tmp_path / "runs.db"
    with SQLiteWorkflowStore(path) as store:
        def lineage_run(**options):
            return repair_plans.lineage_swarm(store, **options).execute(graph()).execution_id

        replaced, successor = lineage_run(), lineage_run()
        undecided = lineage_run(acceptance=policy("raise"))
        plain = lineage_run(lineage=None)
        repair_plans.link(store, successor, replaced)
        repair_plans.link(store, undecided, replaced)
        recorded, store_id = load_lineage(store, replaced).sha256, store.store_id

    code, value = inspect_durable(capsys, path, replaced)
    assert code == EXIT_OK
    lineage = value["lineage"]
    assert lineage["successors"] == [{"run_id": successor, "supersedes": True},
                                     {"run_id": undecided, "supersedes": False}]
    assert (lineage["status"], lineage["reasons"]) == ("superseded", [])
    code, value = inspect_durable(capsys, path, successor)
    assert code == EXIT_OK
    assert value["lineage"]["predecessor"] == {"store_id": store_id, "run_id": replaced,
                                               "lineage_sha256": recorded}
    # Without a source catalog, a run's freshness is not derived.
    assert (value["lineage"]["status"], value["lineage"]["coverage"]) == (None, "partial")
    assert value["lineage"]["undeclared"] == ["draft", "final"]
    assert main(["runs", "--journal", str(path), "inspect", replaced]) == EXIT_OK
    text = capsys.readouterr().out
    assert (f"  Predecessor: none; successors: {successor}, {undecided} (does not supersede)"
            in text)
    assert "  Derived status: superseded" in text
    assert main(["runs", "--journal", str(path), "inspect", successor]) == EXIT_OK
    text = capsys.readouterr().out
    assert f"  Predecessor: {replaced}; successors: none" in text
    assert "  Undeclared nodes: draft, final" in text
    assert "  Derived status: none; freshness needs a source catalog" in text
    # A run without lineage has no lineage block.
    assert inspect_durable(capsys, path, plain)[1]["lineage"] is None


# ---------------------------------------------------------------------------
# Durable runs with external actions
# ---------------------------------------------------------------------------


def action_journal(tmp_path, scripted, *, committed=False, setup=None):
    """A journal, without WAL sidecars, whose accepted run has one action; returns the path, the
    action, its host and clock. *setup* makes the action; by default its outcome is unknown and
    reconciliation could not settle it."""
    path, clock, host = tmp_path / "runs.db", ManualClock(time.time_ns()), action_approvals.Host()
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        if setup is None:
            action = operator_actions.under_review(store, tmp_path, scripted, host, clock, committed=committed)[0]
        else:
            action = setup(store, tmp_path, scripted, host, clock)[0]
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    return path, action, host, clock


def secrets_of(path: Path) -> list[str]:
    """What inspection must never show: assertions, error text, notes and parameters."""
    return [row["assertion"] for row in action_rows(path, "workflow_action_approvals")] + [
        "timed out", "nothing can show whether the write took effect", operator_actions.NOTE, "Launch notes"]


def test_exit_code_12_is_new():
    taken = {getattr(cli, name) for name in dir(cli)
             if name.startswith("EXIT_") and not name.startswith("EXIT_RUN_")}
    assert EXIT_RUN_ACTION_UNRESOLVED == 12 and EXIT_RUN_ACTION_UNRESOLVED not in taken
    args = build_parser().parse_args(["runs", "revoke-action", "abc", "--reason", "why", "--journal", "j.db"])
    assert (args.runs_command, args.target, args.reason, args.journal) == ("revoke-action", "abc", "why", "j.db")


def test_inspection_shows_external_actions_read_only_and_exits_12_while_one_is_unresolved(
        tmp_path, capsys, monkeypatch, scripted_replies):  # noqa: F811
    path, action, host, _ = action_journal(tmp_path, scripted_replies)
    before = journal_state(path)
    statements = []

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    def forbidden(*args, **kwargs):
        pytest.fail("Runs inspection constructed a connector or an action runner")

    no_side_effects(monkeypatch)
    monkeypatch.setattr(FixtureConnector, "__init__", forbidden)
    monkeypatch.setattr("smythe.actions.ActionRunner.__init__", forbidden)
    monkeypatch.setattr(sqlite3, "connect", connect)
    code = main(["runs", "--journal", str(path), "--json", "inspect", action.run_id])
    monkeypatch.setattr(sqlite3, "connect", CONNECT)
    value = output(capsys)["inspect"]

    assert code == EXIT_RUN_ACTION_UNRESOLVED
    compact = ["".join(sql.upper().split()) for sql in statements]
    assert statements and "BEGINIMMEDIATE" not in compact
    assert not any(sql.startswith(("PRAGMAJOURNAL_MODE=", "CREATE", "INSERT", "UPDATE", "DELETE", "ALTER"))
                   for sql in compact)
    assert journal_state(path) == before
    # The output is accepted, and the run itself needs nothing more: only the action does.
    assert (value["delivery"]["decision"], value["next_action"]["action"]) == ("accepted", "none")
    actions = value["actions"]
    assert list(actions) == ["accounting_scope", "unit", "confirmed_nanousd", "reserved_nanousd",
                             "unknown_nanousd", "unresolved", "items"]
    write = action.profile.write_ceiling_nanousd
    assert (actions["accounting_scope"], actions["unit"], actions["unknown_nanousd"], actions["unresolved"]) == (
        "external_actions", "nanoUSD", write, [action.action_id])
    (entry,) = actions["items"]
    assert (entry["action_id"], entry["slot"], entry["state"], entry["duplicate_risk"], entry["next_action"]) == (
        action.action_id, "publish", "outcome_unknown", False, "operator_review")
    (approval,) = entry["approvals"]
    assert approval["principal"] == {"issuer": host.issuer, "subject": "editor-1", "tenant": "acme"}
    assert set(approval) == {"approval_id", "verdict", "intent_sha256", "principal", "verifier", "ceiling_nanousd",
                             "expires_at_ns", "consumed_by_attempt", "revoked_at_ns", "duplicate_risk"}
    assert [(record["kind"], len(record["payload_sha256"])) for record in entry["records"]] == [("finding", 64)]
    assert all(set(record) == {"record_id", "kind", "payload_sha256", "trusted", "cost_nanousd"}
               for record in entry["records"])
    with SQLiteWorkflowStore(path, read_only=True) as reader:
        assert actions == inspect_actions(reader, action.run_id)

    assert main(["runs", "--journal", str(path), "inspect", action.run_id]) == EXIT_RUN_ACTION_UNRESOLVED
    text = capsys.readouterr().out
    assert f"External actions: 1; 0 confirmed, 0 reserved, {write} unknown nanoUSD; 1 unresolved" in text
    assert (f"  Action publish/1 ({action.action_id}): outcome_unknown; 1 attempts; 0 confirmed, "
            f"{write} unknown nanoUSD; next operator_review") in text
    assert (f"    Approval {approval['approval_id']}: editor-1 of acme ({host.issuer}); approved; "
            f"expires at {approval['expires_at_ns']} ns") in text
    assert "    Records: finding 1" in text
    for shown in (text, json.dumps(value)):
        assert not any(secret in shown for secret in secrets_of(path))
    # Listing reports the run as before, with no new exit code.
    assert main(["runs", "--journal", str(path), "--json", "list"]) == EXIT_OK
    assert output(capsys)["list"]["runs"][0]["decision"] == "accepted"
    assert journal_state(path) == before


def test_exit_codes_10_and_11_take_precedence_and_a_resolved_action_exits_0(
        tmp_path, capsys, scripted_replies):  # noqa: F811
    path, action, host, clock = action_journal(tmp_path, scripted_replies, committed=True)

    def decided(acceptance):
        with SQLiteWorkflowStore(path, clock_ns=clock) as store:
            action_approvals.swarm(store, tmp_path).reevaluate(action.run_id, acceptance=acceptance)
        return inspect_durable(capsys, path, action.run_id)

    for acceptance, decision, code in ((strict(Contains("absent")), "rejected", EXIT_RUN_NOT_ACCEPTED),
                                       (strict(Raises()), "indeterminate", EXIT_RUN_INDETERMINATE),
                                       (strict(Contains("report")), "accepted", EXIT_RUN_ACTION_UNRESOLVED)):
        exit_code, value = decided(acceptance)
        assert (exit_code, value["delivery"]["decision"]) == (code, decision)
        assert value["actions"]["unresolved"] == [action.action_id]

    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        operator_actions.decide(store, host, action, "adopt_observed", operator_actions.evidence())
    code, value = inspect_durable(capsys, path, action.run_id)
    assert (code, value["actions"]["unresolved"]) == (EXIT_OK, [])
    assert value["actions"]["items"][0]["state"] == "reconciled"
    assert not any(secret in json.dumps(value) for secret in secrets_of(path))


@pytest.mark.parametrize("setup,state,step,code", [
    (operator_actions._wrong, "compensation_required", "compensate", EXIT_RUN_ACTION_UNRESOLVED),
    (operator_actions._unverified, "dispatched", "operator_review", EXIT_RUN_ACTION_UNRESOLVED),
    (operator_actions.awaiting_readback, "dispatched", "readback", EXIT_OK),
], ids=["compensation-required", "unverified", "awaiting-readback"])
def test_a_wrong_or_unverified_effect_exits_12_and_a_pending_readback_does_not(
        tmp_path, capsys, scripted_replies, setup, state, step, code):  # noqa: F811
    path, action, _, _ = action_journal(tmp_path, scripted_replies, setup=setup)
    before = journal_state(path)

    exit_code, value = inspect_durable(capsys, path, action.run_id)

    assert (exit_code, value["delivery"]["decision"]) == (code, "accepted")
    (entry,) = value["actions"]["items"]
    assert (entry["state"], entry["next_action"]) == (state, step)
    assert value["actions"]["unresolved"] == ([] if code == EXIT_OK else [action.action_id])
    assert journal_state(path) == before


def test_revoke_action_is_a_fail_safe_control_and_an_abandoned_action_stays_unresolved(
        tmp_path, capsys, scripted_replies):  # noqa: F811
    path, action, host, clock = action_journal(tmp_path, scripted_replies)
    (approval_id,) = [row["approval_id"] for row in action_rows(path, "workflow_action_approvals")]
    reason = f"Approver left {action_approvals.ORGANIZATION}"
    revoke = ["runs", "--journal", str(path), "--json", "revoke-action", action.action_id, "--reason", reason]
    # Another worker's live lease does not stop the revocation.
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        store.acquire_lease(action.run_id, "another-worker", ttl_s=3600)

    assert main(revoke) == EXIT_OK
    assert output(capsys)["revoke-action"] == {"target": action.action_id, "revoked": [approval_id]}
    (row,) = action_rows(path, "workflow_action_approvals")
    assert row["revoked_reason"] == "Approver left org-[redacted]" and row["revoked_at_ns"] is not None
    assert main(revoke) == EXIT_OK
    assert output(capsys)["revoke-action"]["revoked"] == []
    assert main(revoke[:3] + revoke[4:]) == EXIT_OK
    assert capsys.readouterr().out == f"Revoked 0 approvals of {action.action_id}; none was in force\n"
    code, value = inspect_durable(capsys, path, action.run_id)
    assert code == EXIT_RUN_ACTION_UNRESOLVED
    assert value["actions"]["items"][0]["approvals"][0]["revocation"] == "after_dispatch"
    assert "Approver left" not in json.dumps(value)

    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        operator_actions.decide(store, host, action, "abandon")
    code, value = inspect_durable(capsys, path, action.run_id)
    assert (code, value["actions"]["unresolved"]) == (EXIT_RUN_ACTION_UNRESOLVED, [action.action_id])
    assert (value["actions"]["items"][0]["state"], value["actions"]["items"][0]["next_action"]) == (
        "unresolved", None)


def test_revoke_action_refuses_what_it_cannot_revoke_and_never_upgrades_a_journal(tmp_path, capsys):
    def revoke(*options, target="no-such-action"):
        return main(["runs", *options, "--json", "revoke-action", target, "--reason", "stop"])

    path, _ = durable_run(tmp_path)
    before = path.read_bytes()
    assert revoke("--journal", str(path)) == EXIT_JOB_STATE
    assert "no external actions" in output(capsys)["error"]["message"]
    assert path.read_bytes() == before and recorded_version(path) == [(1,)]

    current = tmp_path / "current.db"
    SQLiteWorkflowStore(current).close()
    assert revoke("--journal", str(current)) == EXIT_JOB_STATE
    assert output(capsys)["error"]["type"] == "RunNotFoundError"
    assert revoke("--checkpoints", str(tmp_path)) == EXIT_INVALID_INPUT
    assert "needs --journal" in output(capsys)["error"]["message"]
    assert revoke("--journal", str(tmp_path / "missing.db")) == EXIT_LOCAL_ERROR
    assert output(capsys)["error"]["type"] == "RunInspectionError"
    assert not (tmp_path / "missing.db").exists()
    assert main(["runs", "--journal", str(current), "--json", "revoke-action", "x", "--reason", ""]) == (
        EXIT_INVALID_INPUT)
    assert output(capsys)["error"]["type"] == "WorkflowValidationError"
