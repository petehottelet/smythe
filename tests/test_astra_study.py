"""Offline checks for the calibrated follow-up study; never calls a provider."""

from copy import deepcopy
import asyncio
import hashlib
import json

import pytest

from benchmarks import astra_study as study
from benchmarks.astra_campaign import load_task_pack
from smythe.task import task_to_dict

# Native Responses transport is stubbed; the actual durable workflow and
# SQLite accounting still execute, including quote/receipt reconciliation.
from test_astra_runtime import native  # noqa: F401


def test_followup_makes_reasoning_type_explicit_without_changing_original_pack():
    case = next(c for c in load_task_pack().tasks if c.task_id == "pilot-relays")
    original = case.task_json
    task = task_to_dict(study.study_task(case))
    assert any("/reasoning" in c and "nonempty string" in c for c in task["constraints"])
    assert case.task_json == original
    assert "rubric" not in task and "checks" not in task


def test_task_contract_does_not_leak_planner_fields_or_the_correct_decision():
    for case in load_task_pack().tasks:
        task = task_to_dict(study.study_task(case))
        assert not any("max_retries" in c or "execution nodes" in c for c in task["constraints"])
        if case.task_id == "main-capacity-chain":
            assert '"overtime" or "regular"' in task["constraints"][-1]
        if case.task_id == "main-production-chain":
            assert '"expedite" or "normal"' in task["constraints"][-1]


def test_all_numeric_field_types_are_explicit_without_exposing_answers():
    for case in load_task_pack().tasks:
        task = study.study_task(case)
        contracts = [c for c in task.constraints if "must contain JSON numbers" in c]
        numbers = [c for c in case.checks if c["kind"] == "number"]
        assert bool(contracts) == bool(numbers)
        for check in numbers:
            assert "/" + "/".join(map(str, check["path"])) in contracts[0]
        assert not contracts or not any(char.isdigit() for char in contracts[0])
        if case.task_id.endswith("calendar") or case.task_id == "main-identifiers":
            assert not any("top-level JSON fields" in c for c in task.constraints)


def test_prior_amendments_cannot_drop_ancestor_spending(monkeypatch):
    old = {"directory": "old", "summary_sha256": "old", "confirmed_nanousd": 7}
    monkeypatch.setattr(study, "inspect_stage", lambda _: {"stage": "format-pilot", "confirmed_nanousd": 5})
    monkeypatch.setattr(study.pilot, "_read", lambda _: {"freeze_sha256": "new", "superseded_studies": [old]})
    with pytest.raises(ValueError, match="Include every earlier"):
        study._prior_stages(["latest"], "format-pilot")


def test_inspection_rejects_modified_freeze_before_touching_ledger(tmp_path):
    path = tmp_path / "study-freeze.json"
    path.write_text('{"freeze_sha256":"wrong","stage":"main"}', encoding="utf-8")
    with pytest.raises(ValueError, match="freeze content changed"):
        study.inspect_stage(tmp_path)


def test_amended_pilot_subtracts_all_earlier_spending(monkeypatch, tmp_path):
    allowance = {"pilot_nanousd": 60_000_000_000, "per_trial_nanousd": 5_000_000_000}
    original = {"freeze": {"allowances": allowance, "seed": 14173},
                "confirmed_nanousd": 780_595_500, "summary_sha256": "pilot"}
    receipt = {"directory": "prior", "summary_sha256": "prior", "confirmed_nanousd": 678_991_400}
    monkeypatch.setattr(study, "inspect_pilot", lambda _: original)
    monkeypatch.setattr(study, "_prior_stages", lambda *args: [receipt])
    value = study.freeze_stage(stage="format-pilot", directory=tmp_path / "next", original_pilot=tmp_path,
                               previous_stages=["prior"])
    assert value["stage_allowance_nanousd"] == 60_000_000_000 - 1_459_586_900
    assert value["prior_stage_cost_nanousd"] == 678_991_400
    assert value["superseded_studies"] == [receipt]


def test_changed_prior_receipt_blocks_execution_before_any_provider(monkeypatch):
    value = {"stage": "main", "superseded_studies": [{"directory": "prior", "confirmed_nanousd": 12}],
             "prior_stage_cost_nanousd": 12}
    value["freeze_sha256"] = study._hash(value)
    monkeypatch.setattr(study, "_prior_stages", lambda *args: [{"directory": "prior", "confirmed_nanousd": 13}])
    with pytest.raises(ValueError, match="Prior study spending"):
        study._verify_freeze(value)


def test_main_cannot_freeze_before_human_and_format_pilot(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "inspect_pilot", lambda _: {"freeze": {"allowances": {}}})
    with pytest.raises(ValueError, match="requires format-pilot"):
        study.freeze_stage(stage="main", directory=tmp_path, original_pilot=tmp_path)


@pytest.fixture
def calibration(tmp_path):
    samples, outcomes, ratings = [], [], []
    arms = ("astra-fixed", "astra-dynamic", "sol-fixed", "sol-dynamic")
    for i in range(6):
        output = f"Test sample {i}"
        sha = hashlib.sha256(output.encode()).hexdigest()
        identity = chr(65 + i)
        control = i == 5
        samples.append({"sample_id": identity, "output": output, "output_sha256": sha,
                        "origin": "negative-control" if control else "pilot", "run_id": str(i)})
        ratings.append({"sample_id": identity, "output_sha256": sha, "score": 1 if control else 4})
        if not control:
            outcomes.append({"run_id": str(i), "output_sha256": sha,
                             "trial": {"task_id": "pilot-membership" if i < 4 else "pilot-relays",
                                       "arm_id": arms[i % 4]}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(samples), encoding="utf-8")
    response = {"status": "submitted", "source": "local-user-review-form", "submitted_at": "2026-09-13",
                "sample_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "ratings": ratings}
    path = tmp_path / "response.json"
    path.write_text(json.dumps(response), encoding="utf-8")
    return path, manifest, {"outcomes": outcomes}, response


def test_balanced_human_ratings_and_rejected_control_bind_to_exact_outputs(calibration):
    path, manifest, original, _ = calibration
    result = study.inspect_human_calibration(path, manifest, original)
    assert result["status"] == "passed" and result["balanced_arms"] == 4


@pytest.mark.parametrize("kind", ["missing", "duplicate", "changed", "bool", "control", "disputed", "unsubmitted"])
def test_missing_or_disputed_human_evidence_cannot_open_main(calibration, kind):
    path, manifest, original, value = calibration
    response = deepcopy(value)
    if kind == "missing":
        response["ratings"].pop()
    elif kind == "duplicate":
        response["ratings"][1] = response["ratings"][0]
    elif kind == "changed":
        response["ratings"][0]["output_sha256"] = "0" * 64
    elif kind == "bool":
        response["ratings"][0]["score"] = True
    elif kind == "control":
        response["ratings"][-1]["score"] = 4
    elif kind == "disputed":
        response["ratings"][0]["score"] = 2
    else:
        response["status"] = "draft"
    path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError):
        study.inspect_human_calibration(path, manifest, original)


def test_main_gate_is_checked_at_dispatch_not_only_freeze():
    value = {"stage": "main", "human_calibration": None}
    value["freeze_sha256"] = study._hash(value)
    with pytest.raises(ValueError, match="calibration evidence"):
        study._verify_freeze(value)


def test_stage_balance_rejects_unresolved_billing_or_wrong_identity():
    class Store:
        def list_runs(self):
            return [{"run_id": "unbound", "budget_nanousd": 5_000_000_000}]
    with pytest.raises(ValueError, match="Unbound"):
        study._balance(Store(), {"schedule": [], "per_trial_nanousd": 5_000_000_000})


def test_summary_retains_execution_and_contract_failures():
    freeze = {"stage": "main", "freeze_sha256": "x", "schedule": [1, 2]}
    row = {"trial": {"trial_id": "one"}, "run_id": "id", "record_sha256": "h",
           "status": "failed", "deterministic_checks": {"deterministic_passed": False}}
    result = study._summary(freeze, [row], 12)
    assert result["planned_workflows"] == 2 and result["completed_workflows"] == 1
    assert result["execution_failures"] == result["deterministic_failures"] == 1
    assert result["confirmed_nanousd"] == 12 and result["claimable"] is False


def test_full_stage_executes_native_accounting_and_replays_without_rebuy(tmp_path, monkeypatch, native):  # noqa: F811
    from benchmarks.astra_campaign import prepare_campaign
    allowances = {"total_nanousd": 300_000_000_000, "pilot_nanousd": 60_000_000_000,
                  "main_nanousd": 200_000_000_000, "judge_nanousd": 40_000_000_000,
                  "per_trial_nanousd": 5_000_000_000}
    original = {"freeze": {"allowances": allowances, "seed": 14173},
                "confirmed_nanousd": 780_595_500, "summary_sha256": "prior-pilot"}
    monkeypatch.setattr(study, "inspect_pilot", lambda _: original)
    preparation = prepare_campaign()
    preparation["schedules"]["pilot"] = preparation["schedules"]["pilot"][:2]
    monkeypatch.setattr(study, "prepare_campaign", lambda **_: preparation)
    value = study.freeze_stage(stage="format-pilot", directory=tmp_path / "study", original_pilot=tmp_path)
    first = asyncio.run(study.run_stage(value))
    assert first["completed_workflows"] == 2 and first["execution_failures"] == 0
    assert first["deterministic_failures"] == 2  # Deliberately invalid fixture answers.
    assert first["confirmed_nanousd"] > 0 and len(native.requests) >= 6
    assert value["stage_allowance_nanousd"] == 60_000_000_000 - 780_595_500
    calls = len(native.requests)
    monkeypatch.setattr(study.pilot, "_swarm", lambda *args: pytest.fail("Completed trial was rebought"))
    assert asyncio.run(study.run_stage(value)) == first
    assert study.inspect_stage(tmp_path / "study") == first
    assert len(native.requests) == calls


def test_cli_requires_exact_freeze_approval_before_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    path.write_text('{"freeze_sha256":"expected"}', encoding="utf-8")
    monkeypatch.setattr(study, "run_stage", lambda _: pytest.fail("Unapproved dispatch"))
    with pytest.raises(SystemExit):
        study.main(["run", "--freeze", str(path), "--approved-freeze-sha256", "wrong"])
