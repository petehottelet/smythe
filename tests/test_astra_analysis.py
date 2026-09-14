"""Offline statistical checks; synthetic outcomes are never live evidence."""

from copy import deepcopy
import hashlib

import pytest

from benchmarks.astra_analysis import analyze_main, paired_interval, percentile
from benchmarks.astra_campaign import load_task_pack, prepare_campaign
from benchmarks.astra_evaluation import judge_request
from benchmarks.astra_study import GATES


@pytest.fixture
def complete():
    schedule = prepare_campaign()["schedules"]["main"]
    freeze = {"stage": "main", "freeze_sha256": "offline", "schedule": schedule,
              "gates": {**GATES, "bootstrap_repetitions": 101}}
    cases = {c.task_id: c for c in load_task_pack().tasks}
    rows, bindings, records = [], [], {}
    for trial in schedule:
        identity = trial["trial_id"]
        output = "synthetic output for " + identity
        sha = hashlib.sha256(output.encode()).hexdigest()
        dynamic = trial["strategy"] == "smythe_dynamic"
        rows.append({"trial": trial, "run_id": identity, "output": output, "output_sha256": sha,
                     "freeze_sha256": "offline", "status": "completed",
                     "wall_time_ns": 5_000_000_000 if dynamic else 10_000_000_000,
                     "accounting": {"confirmed_nanousd": 500_000_000 if dynamic else 1_000_000_000,
                                    "reserved_nanousd": 0, "unknown_nanousd": 0, "unknown_calls": 0},
                     "deterministic_checks": {"deterministic_passed": True}})
        scores = {"criteria": [{"id": c["id"], "score": 4, "reason": "fixture"} for c in cases[trial["task_id"]].rubric],
                  "material_defects": []}
        records[identity] = {"attempt": {"request": judge_request(cases[trial["task_id"]], output)},
                             "result": {"record_sha256": sha, "scores": scores}}
        bindings.append({"run_id": identity, "output_sha256": sha, "judgment_identity": identity,
                         "judgment_record_sha256": sha})
    return freeze, rows, {"records": records}, bindings


def test_known_paired_difference_resamples_ten_tasks_not_fifty_repetitions(complete):
    result = analyze_main(*complete)
    assert result["claimable"] is False and result["workflows"] == 200
    for comparison in result["within_model_comparisons"].values():
        assert comparison["metrics"]["cost_usd"] == {"mean_difference": -.5, "ci95": [-.5, -.5], "independent_tasks": 10}
        assert comparison["success_gate_passed"] and comparison["quality_noninferiority_gate_passed"]
    assert result["arms"]["astra-dynamic"]["median_wall_seconds"] == 5
    for contrast in [*result["model_comparisons"].values(), result["interaction"]]:
        assert contrast["metrics"]["cost_usd"]["ci95"] == [0, 0]


def test_model_by_strategy_interaction_has_the_declared_direction(complete):
    freeze, rows, judgments, bindings = complete
    for row in rows:
        if row["trial"]["arm_id"] == "astra-dynamic":
            row["accounting"]["confirmed_nanousd"] = 250_000_000
    result = analyze_main(freeze, rows, judgments, bindings)
    assert result["model_comparisons"]["dynamic"]["metrics"]["cost_usd"]["mean_difference"] == -.25
    assert result["interaction"]["metrics"]["cost_usd"]["ci95"] == [-.25, -.25]


def test_failed_runs_keep_their_cost_and_reduce_success(complete):
    freeze, rows, judgments, bindings = complete
    for row in rows:
        if row["trial"]["arm_id"] == "astra-dynamic" and row["trial"]["repetition"] == 1:
            row["status"] = "failed"
    result = analyze_main(freeze, rows, judgments, bindings)
    arm = result["arms"]["astra-dynamic"]
    assert arm["total_workflow_usd"] == 25 and arm["success_rate"] == .8
    assert arm["cost_per_accepted_usd"] == .625
    assert not result["within_model_comparisons"]["astra"]["success_gate_passed"]


def test_known_quality_regression_fails_predeclared_margin(complete):
    freeze, rows, judgments, bindings = complete
    for row in rows:
        if row["trial"]["arm_id"] == "astra-dynamic":
            for criterion in judgments["records"][row["run_id"]]["result"]["scores"]["criteria"]:
                criterion["score"] = 2
    result = analyze_main(freeze, rows, judgments, bindings)
    assert not result["within_model_comparisons"]["astra"]["quality_noninferiority_gate_passed"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "binding", "billing", "repetition"])
def test_incomplete_or_changed_evidence_is_rejected(complete, kind):
    freeze, rows, judgments, bindings = deepcopy(complete)
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[0] = rows[1]
    elif kind == "binding":
        bindings[0]["output_sha256"] = "different"
    elif kind == "billing":
        rows[0]["accounting"]["unknown_nanousd"] = 1
    else:
        rows[0]["trial"]["repetition"] = 7
    with pytest.raises(ValueError):
        analyze_main(freeze, rows, judgments, bindings)


def test_percentiles_and_cluster_randomness_are_reproducible():
    assert percentile([0, 10], .95) == 9.5
    assert paired_interval([1, 2, 3], seed=7, repetitions=101) == paired_interval([1, 2, 3], seed=7, repetitions=101)


def retained_unknown(complete):
    freeze, rows, judgments, bindings = deepcopy(complete)
    failed = next(r for r in rows if r["trial"]["arm_id"] == "sol-dynamic")
    failed.update({"status": "failed", "output": None, "output_sha256": None})
    failed["deterministic_checks"]["deterministic_passed"] = False
    failed["accounting"].update({"unknown_nanousd": 200_000_000, "unknown_calls": 1})
    bindings = [r for r in bindings if r["run_id"] != failed["run_id"]]
    return freeze, rows, judgments, bindings


def test_reserved_cost_keeps_failure_and_withholds_affected_cost_contrasts(complete):
    values = retained_unknown(complete)
    result = analyze_main(*values, allow_reserved_cost=True)
    failed = next(r for r in result["all_trials"] if r["cost_usd"] is None)
    assert not failed["accepted"] and failed["quality_mean"] == 0
    assert failed["cost_lower_usd"] == .5 and failed["cost_upper_usd"] == .7
    arm = result["arms"]["sol-dynamic"]
    assert arm["accepted"] == 49 and arm["total_workflow_usd"] is None
    assert arm["total_workflow_bounds_usd"] == [25, 25.2]
    assert arm["cost_per_accepted_bounds_usd"] == [25 / 49, 25.2 / 49]
    sol = result["within_model_comparisons"]["sol"]
    assert sol["metrics"]["cost_usd"] is None and sol["metrics"]["wall_seconds"] is not None
    assert sol["cost_difference_bounds_usd"] == pytest.approx([-.5, -.496])
    assert result["model_comparisons"]["dynamic"]["metrics"]["cost_usd"] is None
    assert result["interaction"]["metrics"]["cost_usd"] is None
    assert result["within_model_comparisons"]["astra"]["metrics"]["cost_usd"] is not None
    assert result["billing"]["cost_upper_nanousd"] - result["billing"]["cost_lower_nanousd"] == 200_000_000


@pytest.mark.parametrize("change", ["no-opt-in", "active", "completed", "no-hold", "malformed"])
def test_reserve_mode_cannot_silently_relax_billing_requirements(complete, change):
    values = retained_unknown(complete)
    failed = next(r for r in values[1] if r["output"] is None)
    if change == "active":
        failed["accounting"]["reserved_nanousd"] = 1
    elif change == "completed":
        failed["status"] = "completed"
    elif change == "no-hold":
        failed["accounting"]["unknown_nanousd"] = 0
    elif change == "malformed":
        failed["accounting"]["unknown_nanousd"] = -1
    with pytest.raises(ValueError):
        analyze_main(*values, allow_reserved_cost=change != "no-opt-in")


def test_truthy_text_cannot_enable_reserved_cost_analysis(complete):
    with pytest.raises(ValueError, match="explicit boolean"):
        analyze_main(*complete, allow_reserved_cost="false")
