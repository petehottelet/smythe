"""The optimize CLI reviews, activates, rolls back and retires policies, and reports targets.

Campaigns come from a fake evaluator in a local ledger, with no provider
calls. A registry change names the head it expects and writes nothing when it
is refused or conflicts; history reads both files without changing them; and
`optimize inspect --json` is unchanged by any of it.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
import shutil
import sqlite3

import pytest

from smythe.cli import EXIT_INVALID_INPUT, EXIT_OK, EXIT_OPTIMIZE_STATE, main
from smythe.optimize import (
    OptimizationNeedsAttention,
    OptimizationRunner,
    PolicyRegistry,
    TrialContext,
    TrialOutcome,
)
from smythe.optimize.contracts import Candidate, ExperimentContract, MetricObjective, MutableFieldRule
from smythe.optimize.ledger import ExperimentLedger

TARGET = "planner.template.report"
FIELD = TARGET + ".max_sections"
OTHER = "planner.template.summary.max_sections"
# Each campaign compares an incumbent section count with one challenger's.
CAMPAIGNS = {"promoted": (FIELD, 4, 6), "rejected": (FIELD, 4, 3), "chained": (FIELD, 6, 7),
             "stopped": (FIELD, 4, 5), "elsewhere": (OTHER, 4, 6)}
TOUCHING = ("promoted", "rejected", "chained", "stopped")
COST, CEILING = 250, 1_000
CHANGE = ("--actor", "release-operator", "--reason", "reviewed for new report runs")


def _contract(field: str) -> ExperimentContract:
    return ExperimentContract(
        name="report_sections",
        objectives=(MetricObjective("quality", "maximize", primary=True),),
        mutable_fields=(field,),
        mutable_field_rules={field: MutableFieldRule("integer", minimum=1, maximum=12)},
        required_gates=("valid",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=2,
        max_parallel_candidates=1,
        max_trials=14,
        max_wall_seconds=600,
        max_budget_microusd=14 * CEILING,
        per_trial_reservation_microusd=CEILING,
        confidence=0.8,
        min_improvement=0.1,
        base_seed=17,
    )


async def _sections(context: TrialContext) -> TrialOutcome:
    """More sections score higher; a five-section trial loses its outcome."""
    (sections,) = context.candidate.policy.values()
    if sections == 5:
        raise ConnectionError("response lost after dispatch")
    return TrialOutcome(metrics={"quality": float(sections)}, gates={"valid": True},
                        actual_cost_microusd=COST)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A closed ledger holding every campaign, and an empty closed registry."""
    folder = tmp_path_factory.mktemp("saved")
    ledger_path, registry_path = folder / "optimize.sqlite3", folder / "registry.sqlite3"
    with ExperimentLedger(ledger_path, durability="normal") as ledger:
        for name, (field, incumbent, challenger) in CAMPAIGNS.items():
            contract = _contract(field)
            first = Candidate(contract, {field: incumbent}, f"{incumbent} sections")
            second = Candidate(contract, {field: challenger}, f"{challenger} sections", parent=first)
            runner = OptimizationRunner(contract, ledger, _sections, evaluator_hash="sha256:" + "e" * 64,
                                        bootstrap_resamples=10, campaign_id=name)
            try:
                asyncio.run(runner.run(first, (second,)))
            except OptimizationNeedsAttention:
                assert name == "stopped"
    PolicyRegistry(registry_path).close()
    return ledger_path, registry_path


@pytest.fixture
def files(saved, tmp_path):
    copies = tuple(tmp_path / source.name for source in saved)
    for source, copy in zip(saved, copies):
        shutil.copyfile(source, copy)
    return copies


def _dump(path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return list(connection.iterdump())


def optimize(capsys, files, *arguments, as_json=True):
    """Run one optimize command against *files*; return its exit code and output."""
    ledger, registry = files
    code = main(["optimize", "--ledger", str(ledger), *arguments, "--registry", str(registry),
                 *(("--json",) if as_json else ())])
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if as_json else captured.out + captured.err


def _approve(capsys, files, campaign, *options):
    code, output = optimize(capsys, files, "review", campaign, "--verdict", "approve",
                            "--reviewer", "report-owner", *options)
    assert code == EXIT_OK, output
    return output["review"]


def _activate(capsys, files, review_id, expected_seq):
    code, output = optimize(capsys, files, "activate", TARGET, review_id,
                            "--expected-seq", str(expected_seq), *CHANGE)
    assert code == EXIT_OK, output
    return output["activate"]


def _state(files):
    ledger, registry = files
    return ledger.read_bytes(), registry.read_bytes(), _dump(registry)


def test_review_activate_history_rollback_and_retire_end_to_end(files, tmp_path, capsys):
    ledger = files[0]
    ledger_bytes = ledger.read_bytes()
    evidence = tmp_path / "evidence" / "holdout-review.txt"
    evidence.parent.mkdir()
    evidence.write_bytes(b"holdout gain holds on the long-report pack\n")

    # Only a promoted decision can be approved; the refusal writes nothing.
    before = _state(files)
    code, output = optimize(capsys, files, "review", "rejected", "--verdict", "approve",
                            "--reviewer", "report-owner")
    assert code == EXIT_OPTIMIZE_STATE
    assert output["ok"] is False and output["command"] == "review"
    assert output["error"]["type"] == "RegistryError"
    assert "was not promoted" in output["error"]["message"]
    assert _state(files) == before
    code, output = optimize(capsys, files, "review", "rejected", "--verdict", "reject",
                            "--reviewer", "report-owner")
    assert code == EXIT_OK and output["review"]["verdict"] == "reject"

    review = _approve(capsys, files, "promoted", "--evidence", str(evidence),
                      "--note", "holdout gain holds")
    assert (review["campaign_id"], review["promoted"], review["verdict"], review["reviewer"],
            review["note"]) == ("promoted", True, "approve", "report-owner", "holdout gain holds")
    assert review["evidence_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert (review["policy"], review["incumbent_policy"]) == ({FIELD: 6}, {FIELD: 4})
    # The evidence is recorded by digest, never by path or content.
    assert "holdout-review" not in json.dumps(review) and "long-report" not in json.dumps(review)

    activated = _activate(capsys, files, review["review_id"], 0)
    assert activated["target"] == TARGET
    assert [(event["seq"], event["action"], event["policy"], event["review_id"])
            for event in activated["events"]] == [
        (1, "baseline", {FIELD: 4}, None), (2, "activate", {FIELD: 6}, review["review_id"])]
    assert activated["events"][1]["policy_hash"] == review["policy_hash"]

    code, output = optimize(capsys, files, "history", TARGET)
    assert code == EXIT_OK
    report = output["history"]
    assert report["target"] == TARGET
    assert report["active"] == {"target": TARGET, "seq": 2, "policy_hash": review["policy_hash"],
                                "policy": {FIELD: 6}, "review_id": review["review_id"],
                                "source_seq": 2}
    assert report["history"] == activated["events"]
    assert report["campaigns_not_in_ledger"] == []
    campaigns = {item["campaign_id"]: item for item in report["campaigns"]}
    # 'elsewhere' tunes another target, so it and its cost are left out.
    assert set(campaigns) == set(TOUCHING)
    with ExperimentLedger(ledger, read_only=True) as reader:
        snapshots = {name: reader.snapshot(name) for name in TOUCHING}
    for name, snapshot in snapshots.items():
        assert campaigns[name]["cost"] == snapshot["cost"]
    totals = report["optimization_cost"]
    for field in ("confirmed_microusd", "reserved_microusd", "unknown_exposure_microusd",
                  "total_exposure_microusd"):
        assert totals[field] == sum(snapshot["cost"][field] for snapshot in snapshots.values())
    assert totals["campaigns"] == 4 and totals["confirmed_microusd"] > 0
    # The stopped campaign's lost outcome keeps its whole ceiling apart from spend.
    assert campaigns["stopped"]["unknown_trials"] == totals["unknown_trials"] == 1
    assert campaigns["stopped"]["cost"]["unknown_exposure_microusd"] == CEILING
    assert totals["unknown_exposure_microusd"] == CEILING
    assert campaigns["stopped"]["decision"] is None
    verdicts = {name: (item["decision"]["promoted"], item["decision"]["verdict"])
                for name, item in campaigns.items() if item["decision"] is not None}
    assert verdicts == {"promoted": (True, "improved"), "rejected": (False, "inconclusive"),
                        "chained": (True, "improved")}
    assert [(item["verdict"], item["reviewer"], item["evidence_sha256"])
            for item in campaigns["promoted"]["reviews"]] == [
        ("approve", "report-owner", review["evidence_sha256"])]
    assert [item["verdict"] for item in campaigns["rejected"]["reviews"]] == ["reject"]

    # A campaign run against the policy in effect replaces it.
    chained = _approve(capsys, files, "chained")
    assert [event["seq"] for event in _activate(capsys, files, chained["review_id"], 2)["events"]] == [3]
    code, output = optimize(capsys, files, "rollback", TARGET, "--to-seq", "1", "--expected-seq", "3",
                            *CHANGE)
    assert code == EXIT_OK
    (event,) = output["rollback"]["events"]
    assert (event["seq"], event["action"], event["restores_seq"], event["policy"]) == (
        4, "rollback", 1, {FIELD: 4})
    code, output = optimize(capsys, files, "retire", TARGET, "--expected-seq", "4", *CHANGE)
    assert code == EXIT_OK
    (event,) = output["retire"]["events"]
    assert (event["seq"], event["action"], event["policy"]) == (5, "retire", None)

    code, output = optimize(capsys, files, "history", TARGET)
    assert code == EXIT_OK
    report = output["history"]
    assert report["active"] is None
    assert [event["action"] for event in report["history"]] == [
        "baseline", "activate", "activate", "rollback", "retire"]
    assert {item["campaign_id"]: len(item["reviews"]) for item in report["campaigns"]} == {
        "promoted": 1, "rejected": 1, "chained": 1, "stopped": 0}
    # No registry command writes the ledger.
    assert ledger.read_bytes() == ledger_bytes


def test_activate_prints_the_target_seq_and_policy_and_leaves_configuration_alone(files, capsys):
    review = _approve(capsys, files, "promoted")
    code, text = optimize(capsys, files, "activate", TARGET, review["review_id"], "--expected-seq", "0",
                          *CHANGE, as_json=False)
    assert code == EXIT_OK
    assert f"Target: {TARGET}" in text
    assert "Seq 1: baseline" in text and "Seq 2: activate" in text
    assert json.dumps({FIELD: 6}) in text and review["policy_hash"] in text
    assert "Configuration was not changed" in text

    code, text = optimize(capsys, files, "history", TARGET, as_json=False)
    assert code == EXIT_OK
    assert f"Target: {TARGET}" in text and "In effect: seq 2" in text
    for name in TOUCHING:
        assert f"Campaign {name}:" in text
    assert "Campaign elsewhere" not in text
    assert "not netted against inference savings" in text
    assert f"Unknown exposure: 1 unknown trial holding ${CEILING // 1_000_000}.{CEILING:06d}" in text

    code, text = optimize(capsys, files, "review", "chained", "--verdict", "approve",
                          "--reviewer", "report-owner", as_json=False)
    assert code == EXIT_OK and "Review: review_v1_" in text and "Verdict: approve" in text


def test_a_conflicting_activate_exits_eight_and_writes_nothing(files, capsys):
    promoted = _approve(capsys, files, "promoted")
    _activate(capsys, files, promoted["review_id"], 0)
    chained = _approve(capsys, files, "chained")
    before = _state(files)
    for stale in ("0", "1", "3"):
        code, output = optimize(capsys, files, "activate", TARGET, chained["review_id"],
                                "--expected-seq", stale, *CHANGE)
        assert code == EXIT_OPTIMIZE_STATE
        assert output["error"]["type"] == "RegistryConflictError"
        assert f"not expected_seq {stale}" in output["error"]["message"]
        assert _state(files) == before
    # Refused state also exits 8: the promoted campaign's incumbent is no longer in effect.
    code, output = optimize(capsys, files, "activate", TARGET, promoted["review_id"],
                            "--expected-seq", "2", *CHANGE)
    assert code == EXIT_OPTIMIZE_STATE and output["error"]["type"] == "RegistryError"
    code, output = optimize(capsys, files, "rollback", TARGET, "--to-seq", "2", "--expected-seq", "2",
                            *CHANGE)
    assert code == EXIT_OPTIMIZE_STATE and "already in effect" in output["error"]["message"]
    assert _state(files) == before


def test_history_reads_both_files_without_changing_them(files, capsys):
    ledger, registry = files
    _activate(capsys, files, _approve(capsys, files, "promoted")["review_id"], 0)
    before = _state(files)
    for as_json in (True, False):
        code, _ = optimize(capsys, files, "history", TARGET, as_json=as_json)
        assert code == EXIT_OK
        assert _state(files) == before
    sidecars = [path for store in files for path in store.parent.glob(store.name + "-*")]
    assert sidecars == []
    # An open writer is refused, and neither file changes.
    with PolicyRegistry(registry):
        code, output = optimize(capsys, files, "history", TARGET)
    assert code == EXIT_OPTIMIZE_STATE and output["error"]["type"] == "RegistryError"
    assert "closed, checkpointed policy registry" in output["error"]["message"]
    assert ledger.read_bytes() == before[0] and _dump(registry) == before[2]


def test_history_names_activated_campaigns_that_another_ledger_holds(files, tmp_path, capsys):
    _activate(capsys, files, _approve(capsys, files, "promoted")["review_id"], 0)
    other = tmp_path / "other.sqlite3"
    ExperimentLedger(other).close()
    code, output = optimize(capsys, (other, files[1]), "history", TARGET)
    assert code == EXIT_OK
    report = output["history"]
    assert report["campaigns"] == [] and report["campaigns_not_in_ledger"] == ["promoted"]
    assert report["optimization_cost"] == {
        "campaigns": 0, "confirmed_microusd": 0, "reserved_microusd": 0,
        "unknown_exposure_microusd": 0, "total_exposure_microusd": 0, "unknown_trials": 0}
    code, text = optimize(capsys, (other, files[1]), "history", TARGET, as_json=False)
    assert "Not in this ledger: promoted" in text


def test_inspect_json_is_unchanged_by_registry_commands(files, capsys):
    ledger = files[0]

    def inspect() -> str:
        assert main(["optimize", "inspect", "promoted", "--ledger", str(ledger), "--json"]) == EXIT_OK
        return capsys.readouterr().out

    before = inspect()
    payload = json.loads(before)
    assert list(payload) == ["ok", "command", "inspect"]
    assert list(payload["inspect"]) == ["campaign_id", "evaluator_hashes", "ledger_snapshot"]
    with ExperimentLedger(ledger, read_only=True) as reader:
        assert payload["inspect"]["ledger_snapshot"] == reader.snapshot("promoted")
    review = _approve(capsys, files, "promoted")
    _activate(capsys, files, review["review_id"], 0)
    assert optimize(capsys, files, "history", TARGET)[0] == EXIT_OK
    assert optimize(capsys, files, "retire", TARGET, "--expected-seq", "2", *CHANGE)[0] == EXIT_OK
    assert inspect() == before


INVALID = {
    "a target that is not a dotted name": (
        ("activate", "../registry", "review_v1_x", "--expected-seq", "0", *CHANGE),
        "target must be a dotted policy name"),
    "a negative expected seq": (
        ("activate", TARGET, "review_v1_x", "--expected-seq", "-1", *CHANGE), "expected-seq must be"),
    "a rollback to seq 0": (
        ("rollback", TARGET, "--to-seq", "0", "--expected-seq", "2", *CHANGE), "to-seq must be"),
    "a blank actor": (
        ("retire", TARGET, "--expected-seq", "2", "--actor", " ", "--reason", "paused"),
        "actor must be a non-empty string"),
    "a reason with a line break": (
        ("retire", TARGET, "--expected-seq", "2", "--actor", "operator", "--reason", "two\nlines"),
        "reason cannot contain control characters"),
    "a blank reviewer": (
        ("review", "promoted", "--verdict", "approve", "--reviewer", " "), "reviewer must be"),
    "an overlong note": (
        ("review", "promoted", "--verdict", "approve", "--reviewer", "owner", "--note", "n" * 1025),
        "note must be at most 1024"),
    "an unsafe campaign id": (
        ("review", "../promoted", "--verdict", "approve", "--reviewer", "owner"), "campaign_id must be"),
    "a missing evidence file": (
        ("review", "promoted", "--verdict", "approve", "--reviewer", "owner", "--evidence", "MISSING"),
        "cannot read evidence file"),
    "a history target that is not a dotted name": (
        ("history", "planner..template"), "target must be a dotted policy name"),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_invalid_input_exits_two_before_any_file_is_opened_or_created(files, tmp_path, capsys, case):
    arguments, message = INVALID[case]
    arguments = tuple(str(tmp_path / "missing.txt") if item == "MISSING" else item for item in arguments)
    registry = tmp_path / "new" / "registry.sqlite3"
    ledger_bytes = files[0].read_bytes()
    code, output = optimize(capsys, (files[0], registry), *arguments)
    assert code == EXIT_INVALID_INPUT
    assert output["error"]["type"] == "ValueError" and message in output["error"]["message"]
    assert not registry.parent.exists()
    assert files[0].read_bytes() == ledger_bytes


def test_a_verdict_other_than_approve_or_reject_is_a_usage_error(files, capsys):
    with pytest.raises(SystemExit) as caught:
        optimize(capsys, files, "review", "promoted", "--verdict", "approved", "--reviewer", "owner")
    assert caught.value.code == EXIT_INVALID_INPUT


def test_changes_need_an_existing_registry_and_never_create_one(files, tmp_path, capsys):
    registry = tmp_path / "absent" / "registry.sqlite3"
    for arguments in (("activate", TARGET, "review_v1_x", "--expected-seq", "0", *CHANGE),
                      ("rollback", TARGET, "--to-seq", "1", "--expected-seq", "2", *CHANGE),
                      ("retire", TARGET, "--expected-seq", "2", *CHANGE)):
        code, output = optimize(capsys, (files[0], registry), *arguments)
        assert code == EXIT_OPTIMIZE_STATE and output["error"]["type"] == "RegistryError"
        assert "no policy registry" in output["error"]["message"]
    code, output = optimize(capsys, (files[0], registry), "history", TARGET)
    assert code == EXIT_OPTIMIZE_STATE and output["error"]["type"] == "FileNotFoundError"
    assert not registry.parent.exists()
    # A missing ledger refuses a review before a registry is created.
    code, output = optimize(capsys, (tmp_path / "absent.sqlite3", registry), "review", "promoted",
                            "--verdict", "approve", "--reviewer", "owner")
    assert code == EXIT_OPTIMIZE_STATE and output["error"]["type"] == "FileNotFoundError"
    assert not registry.parent.exists()
