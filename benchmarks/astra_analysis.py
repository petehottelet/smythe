"""Descriptive, task-clustered analysis of a complete saved Astra main study.

This module makes no provider calls and cannot promote a campaign to claimable.
Every scheduled workflow, including a failed one, remains in its arm totals.
"""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean, median

from benchmarks import astra_evaluation as evaluator
from benchmarks.astra_campaign import load_task_pack


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_interval(task_differences, *, seed, repetitions):
    """Resample whole task means; repetitions are never independent tasks."""
    values = list(task_differences)
    if not values:
        raise ValueError("Task differences cannot be empty")
    generator = random.Random(seed)
    samples = [mean(generator.choices(values, k=len(values))) for _ in range(repetitions)]
    return {"mean_difference": mean(values), "ci95": [percentile(samples, .025), percentile(samples, .975)],
            "independent_tasks": len(values)}


def analyze_main(freeze, outcomes, judgments, bindings, *, allow_reserved_cost=False):
    """Accept already-audited native records; validate the complete schedule."""
    if type(allow_reserved_cost) is not bool:
        raise ValueError("Reserved-cost mode must be an explicit boolean")
    schedule = freeze["schedule"]
    if freeze["stage"] != "main" or len(schedule) != 200 or len(outcomes) != 200:
        raise ValueError("The analysis requires all 200 main outcomes")
    expected = {r["trial_id"]: r for r in schedule}
    observed = {r["trial"]["trial_id"]: r for r in outcomes}
    if len(expected) != 200 or set(observed) != set(expected):
        raise ValueError("Missing or duplicate scheduled outcome")
    by_run = {r["run_id"]: r for r in bindings}
    if len(by_run) != len(bindings) or set(by_run) != {r["run_id"] for r in outcomes if r["output"] is not None}:
        raise ValueError("Every available output needs exactly one judge binding")
    cases = {c.task_id: c for c in load_task_pack().tasks}
    gates = freeze["gates"]
    arms, rows = defaultdict(list), []
    for outcome in outcomes:
        trial = outcome["trial"]
        if trial != expected[trial["trial_id"]] or outcome["freeze_sha256"] != freeze["freeze_sha256"]:
            raise ValueError("Outcome belongs to a different study")
        scores = None
        if outcome["output"] is not None:
            binding = by_run[outcome["run_id"]]
            saved = judgments["records"][binding["judgment_identity"]]
            if (binding["output_sha256"] != outcome["output_sha256"]
                    or binding["judgment_record_sha256"] != saved["result"]["record_sha256"]
                    or saved["attempt"]["request"] != evaluator.judge_request(cases[trial["task_id"]], outcome["output"])):
                raise ValueError("Judge binding differs from the scheduled output")
            scores = saved["result"]["scores"]
        accounting = outcome["accounting"]
        unresolved = bool(accounting["unknown_nanousd"] or accounting["unknown_calls"])
        if accounting["reserved_nanousd"] or unresolved and not allow_reserved_cost:
            raise ValueError("Unresolved billing prevents exact cost analysis")
        for key in ("confirmed_nanousd", "unknown_nanousd", "unknown_calls"):
            if type(accounting[key]) is not int or accounting[key] < 0:
                raise ValueError("Cost balances must be nonnegative integers")
        if unresolved and (outcome["status"] != "failed" or outcome["output"] is not None
                           or not accounting["unknown_nanousd"] or not accounting["unknown_calls"]):
            raise ValueError("Reserved-cost analysis requires a retained missing-output failure")
        quality_passed = bool(scores) and not scores["material_defects"] and all(
            r["score"] >= gates["criterion_minimum"] for r in scores["criteria"])
        accepted = (outcome["status"] == "completed" and quality_passed
                    and outcome["deterministic_checks"]["deterministic_passed"])
        row = {"trial": trial, "accepted": accepted, "execution_status": outcome["status"],
               "quality_mean": 0 if scores is None else mean(r["score"] for r in scores["criteria"]),
               "cost_usd": None if unresolved else accounting["confirmed_nanousd"] / 1e9,
               "cost_lower_usd": accounting["confirmed_nanousd"] / 1e9,
               "cost_upper_usd": (accounting["confirmed_nanousd"] + accounting["unknown_nanousd"]) / 1e9,
               "billing_status": "unknown-reserved" if unresolved else "known",
               "wall_seconds": None if outcome["wall_time_ns"] is None else outcome["wall_time_ns"] / 1e9}
        rows.append(row)
        arms[trial["arm_id"]].append(row)
    expected_arms = {"astra-fixed", "astra-dynamic", "sol-fixed", "sol-dynamic"}
    tasks = {r["task_id"] for r in schedule}
    if set(arms) != expected_arms or len(tasks) != 10:
        raise ValueError("Expected all four arms and ten independent tasks")
    for arm in arms.values():
        cells = defaultdict(list)
        for row in arm:
            cells[row["trial"]["task_id"]].append(row["trial"]["repetition"])
        if set(cells) != tasks or any(sorted(v) != [1, 2, 3, 4, 5] for v in cells.values()):
            raise ValueError("Every arm needs five repetitions of every task")
    distributions = {}
    for name, arm in sorted(arms.items()):
        accepted = sum(r["accepted"] for r in arm)
        unknown = sum(r["cost_usd"] is None for r in arm)
        total_lower = sum(r["cost_lower_usd"] for r in arm)
        total_upper = sum(r["cost_upper_usd"] for r in arm)
        total = None if unknown else total_lower
        timing = [r["wall_seconds"] for r in arm if r["wall_seconds"] is not None]
        distributions[name] = {"workflows": len(arm), "accepted": accepted,
            "success_rate": accepted / len(arm), "quality_mean": mean(r["quality_mean"] for r in arm),
            "total_workflow_usd": total, "cost_per_accepted_usd": None if not accepted or unknown else total / accepted,
            "total_workflow_bounds_usd": [total_lower, total_upper],
            "cost_per_accepted_bounds_usd": None if not accepted else [total_lower / accepted, total_upper / accepted],
            "median_cost_usd": None if unknown else median(r["cost_usd"] for r in arm),
            "median_cost_bounds_usd": [median(r["cost_lower_usd"] for r in arm), median(r["cost_upper_usd"] for r in arm)],
            "unknown_cost_workflows": unknown,
            "median_wall_seconds": median(timing) if timing else None,
            "p95_wall_seconds": percentile(timing, .95) if timing else None,
            "incomplete_latencies": len(arm) - len(timing)}
    comparisons = {}
    def contrast(weights):
        metrics = {}
        selected = [r for arm in weights for r in arms[arm]]
        for metric in ("cost_usd", "wall_seconds", "quality_mean", "accepted"):
            if any(r[metric] is None for r in selected):
                metrics[metric] = None
                continue
            differences = [sum(weight * mean(r[metric] for r in arms[arm] if r["trial"]["task_id"] == task)
                               for arm, weight in weights.items()) for task in sorted(tasks)]
            metrics[metric] = paired_interval(differences, seed=gates["bootstrap_seed"],
                                               repetitions=gates["bootstrap_repetitions"])
        return metrics

    def cost_bounds(weights):
        # These are identification bounds, not statistical confidence intervals.
        lower = upper = 0
        for arm, weight in weights.items():
            lo = mean(r["cost_lower_usd"] for r in arms[arm])
            hi = mean(r["cost_upper_usd"] for r in arms[arm])
            lower += weight * (lo if weight >= 0 else hi)
            upper += weight * (hi if weight >= 0 else lo)
        return [lower, upper]

    for model in ("astra", "sol"):
        weights = {model + "-dynamic": 1, model + "-fixed": -1}
        metrics = contrast(weights)
        success = all(distributions[model + suffix]["success_rate"] >= gates["minimum_success_rate"] for suffix in ("-fixed", "-dynamic"))
        quality = metrics["quality_mean"]["ci95"][0] >= -gates["quality_noninferiority_margin"]
        comparisons[model] = {"direction": "generated minus fixed; negative time/cost favors generated",
                              "metrics": metrics, "success_gate_passed": success,
                              "cost_difference_bounds_usd": cost_bounds(weights),
                              "quality_noninferiority_gate_passed": quality}
    return {"stage": "main", "claimable": False, "status": "analysis-awaiting-evidence-review",
            "sampling_unit": "task", "independent_tasks": 10, "workflows": 200,
            "gates": gates, "arms": distributions, "within_model_comparisons": comparisons,
            "model_comparisons": {strategy: {"direction": "Astra minus Sol",
                "cost_difference_bounds_usd": cost_bounds({"astra-" + strategy: 1, "sol-" + strategy: -1}), "metrics":
                contrast({"astra-" + strategy: 1, "sol-" + strategy: -1})} for strategy in ("fixed", "dynamic")},
            "interaction": {"direction": "(Astra generated minus fixed) minus (Sol generated minus fixed)",
                "cost_difference_bounds_usd": cost_bounds({"astra-dynamic": 1, "astra-fixed": -1, "sol-dynamic": -1, "sol-fixed": 1}),
                "metrics": contrast({"astra-dynamic": 1, "astra-fixed": -1, "sol-dynamic": -1, "sol-fixed": 1})},
            "task_exposure": freeze.get("task_exposure", "original task split"),
            "billing": {"unknown_cost_workflows": sum(r["cost_usd"] is None for r in rows),
                "cost_lower_nanousd": sum(r["accounting"]["confirmed_nanousd"] for r in outcomes),
                "cost_upper_nanousd": sum(r["accounting"]["confirmed_nanousd"] + r["accounting"]["unknown_nanousd"] for r in outcomes),
                "affected_cost_contrasts": "withheld until native usage is reconciled",
                "cost_bounds_kind": "confirmed charges through confirmed charges plus full held reservation; not confidence intervals"},
            "all_trials": rows, "limitations": ["Ten project-authored synthetic tasks.",
                "Anchored rubric means summarize ordinal scores; no cost-per-score ratios.",
                "Missing-output runs receive zero rubric score and remain in cost and success denominators.",
                "A separate evidence review and human review of disputed outcomes are required for claims."]}
