"""Runs inspection reads saved checkpoints and journals without changing them.

Execution, delivery acceptance and cost are separate blocks. Exit status 10
means a completed run's output was rejected or needs review, and 11 that no
valid decision covers it.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from fakes import ManualClock
from smythe import (
    AcceptancePolicy,
    FileCheckpointStore,
    LocalOnly,
    OfflineProvider,
    SimpleArchitect,
    SQLiteWorkflowStore,
    Swarm,
    Task,
    ValidationReport,
    ValidatorIdentity,
)
import smythe.cli as cli
from smythe.acceptance import text_sha256
from smythe.cli import (
    EXIT_INVALID_INPUT, EXIT_JOB_STATE, EXIT_LOCAL_ERROR, EXIT_OK, EXIT_RUN_INDETERMINATE,
    EXIT_RUN_NOT_ACCEPTED, build_parser, main,
)
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import CompletionResult, Provider
from smythe.validation import hard_finding
from test_workflow_acceptance import rewrite_checkpoint
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
    lambda state: state.update(requires=["delivery_acceptance", "attempt_allowance"]),
    lambda state: state.update(delivery_acceptance={"version": 2}),
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
    assert value["next_action"]["action"] == "none"


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
