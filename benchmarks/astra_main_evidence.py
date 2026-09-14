"""Offline audit of complete Astra main records, inputs, source and judgments."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

from benchmarks import astra_runtime as runtime
from benchmarks.astra_analysis import analyze_main
from benchmarks.astra_campaign._json import canonical, digest, strict_json
from benchmarks.astra_combined import load_continued_main
from benchmarks.astra_evidence import inspect_judgments
from benchmarks.astra_judge_study import load_complete_main
from smythe.workflow_store import CallKey, SQLiteWorkflowStore


def review_source(archive, expected):
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or any(name not in names for name in expected):
            raise ValueError("Source archive has missing or duplicate members")
        for name, wanted in expected.items():
            if digest(source.read(name)) != wanted:
                raise ValueError(f"Frozen source differs: {name}")
    return hashlib.sha256(Path(archive).read_bytes()).hexdigest()


def review_request(call, trial, policy):
    """Check the actual dispatched request, not just its planned configuration."""
    request = strict_json(call["request_json"])
    provider = call["provider"]
    if (request["model"] != trial["model"] or request["max_output_tokens"] != policy["max_output_tokens"]
            or request["reasoning"] != {"effort": policy["reasoning_effort"]}
            or request["service_tier"] != policy["service_tier"]
            or any(k in request for k in ("tools", "temperature", "top_p", "logprobs"))
            or strict_json(call["tool_names_json"]) != {}
            or provider["kind"] != "openai_responses" or provider["endpoint_scope"] != policy["endpoint_scope"]):
        raise ValueError("Dispatched request differs from the frozen model or execution policy")
    receipt = call["receipt"]
    if call["billing_state"] == "unknown":
        if call["cost_nanousd"] is not None or receipt is not None and (
            receipt["requested_model"] != trial["model"] or receipt["endpoint_scope"] != policy["endpoint_scope"]
            or receipt["cost_nanousd"] is not None or receipt["actual_model"] is not None
            or receipt["service_tier"] is not None or receipt["cost_is_complete"] is not False
            or receipt["usage_is_complete"] is not False or not receipt["accounting_error"]
        ):
            raise ValueError("Unknown diagnostic receipt differs from the retained no-response failure")
        return
    if receipt is None or (
        receipt["requested_model"] != trial["model"] or receipt["actual_model"] != trial["model"]
        or receipt["service_tier"] != policy["service_tier"] or receipt["endpoint_scope"] != policy["endpoint_scope"]
        or receipt["cost_nanousd"] != call["cost_nanousd"]
    ):
        raise ValueError("Native response identity or charge differs from the frozen request")


def review_main(*, main_directory, judge_directory, bindings_path, source_archive,
                continuation_directory=None, continuation_source_archive=None):
    continuation = None
    if continuation_directory is None:
        if continuation_source_archive is not None:
            raise ValueError("A continuation source archive needs its evidence directory")
        freeze, outcomes = load_complete_main(main_directory)
    else:
        if continuation_source_archive is None:
            raise ValueError("Continuation evidence requires its frozen source archive")
        freeze, outcomes, continuation = load_continued_main(main_directory, continuation_directory)
        continuation["source_archive_sha256"] = review_source(continuation_source_archive,
            {**freeze["source_sha256"], "benchmarks/astra_continuation.py": continuation["continuation_source_sha256"]})
    source_sha = review_source(source_archive, freeze["source_sha256"])
    judged = inspect_judgments(judge_directory)
    bindings = runtime._read(bindings_path)
    analysis = analyze_main(freeze, outcomes, judged, bindings, allow_reserved_cost=continuation is not None)
    phase_cost, phase_unknown, phase_calls = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    graph_sizes = defaultdict(list)
    task_graph_sizes = defaultdict(lambda: defaultdict(list))
    verified_tasks, total, held, request_count = set(), 0, 0, 0
    directories = [main_directory] + ([] if continuation is None else [continuation_directory])
    for index, directory in enumerate(directories):
        with SQLiteWorkflowStore(Path(directory) / "workflow.sqlite3") as store:
            segment = [r for r in outcomes if bool(r.get("continuation_freeze_sha256")) == bool(index)]
            for outcome in segment:
                arm, task = outcome["trial"]["arm_id"], outcome["trial"]["task_id"]
                run = store.load_run(outcome["run_id"])
                expected = freeze["tasks"][task]
                if run["task"] != expected:
                    raise ValueError("Executed task differs from its frozen provider input")
                verified_tasks.add(task)
                checkpoint = outcome.get("checkpoint")
                if checkpoint is not None:
                    size = len(checkpoint["checkpoint"]["graph"]["nodes"])
                    graph_sizes[arm].append(size)
                    task_graph_sizes[task][arm].append(size)
                calls = outcome["accounting"]["calls"]
                cost = sum(c["cost_nanousd"] for c in calls if c["cost_nanousd"] is not None)
                unknown = sum(c.get("unknown_nanousd", 0) for c in calls)
                if (cost != outcome["accounting"]["confirmed_nanousd"]
                        or unknown != outcome["accounting"]["unknown_nanousd"]):
                    raise ValueError("Complete phase charges do not equal the workflow charge")
                total += cost
                held += unknown
                for call in calls:
                    native = store.lookup_call(outcome["run_id"], CallKey(**call["key"]))
                    review_request(native, outcome["trial"], freeze["policy"])
                    request_count += 1
                    phase = call["key"]["phase"]
                    if call["cost_nanousd"] is not None:
                        phase_cost[arm][phase] += call["cost_nanousd"]
                    phase_unknown[arm][phase] += call.get("unknown_nanousd", 0)
                    phase_calls[arm][phase] += 1
    if total + held > freeze["stage_allowance_nanousd"]:
        raise ValueError("Main study exceeded its remaining stage allowance")
    analysis["phase_cost_nanousd"] = {arm: dict(v) for arm, v in sorted(phase_cost.items())}
    analysis["phase_unknown_nanousd"] = {arm: dict(v) for arm, v in sorted(phase_unknown.items())}
    analysis["phase_generation_calls"] = {arm: dict(v) for arm, v in sorted(phase_calls.items())}
    analysis["graph_sizes"] = {arm: {"observed_graphs": len(values), "node_counts": values,
                                    "median_nodes": median(values), "minimum_nodes": min(values),
                                    "maximum_nodes": max(values)} for arm, values in sorted(graph_sizes.items())}
    analysis["task_graph_sizes"] = {task: {arm: values for arm, values in sorted(arms.items())}
                                     for task, arms in sorted(task_graph_sizes.items())}
    analysis["main_freeze_sha256"] = freeze["freeze_sha256"]
    evidence = {"status": "audited-main", "claimable": False,
                "main_freeze_sha256": freeze["freeze_sha256"], "source_archive_sha256": source_sha,
                "main_workflow_nanousd": total if not held else None,
                "main_workflow_lower_nanousd": total, "main_workflow_upper_nanousd": total + held,
                "prior_main_nanousd": freeze.get("prior_stage_cost_nanousd", 0),
                "original_stage_allocation_nanousd": freeze["campaign_allocations"]["main_nanousd"],
                "judgments": {k: v for k, v in judged.items() if k != "records"},
                "validated_outcomes": len(outcomes), "validated_task_inputs": len(verified_tasks),
                "validated_request_policies": request_count,
                "unknown_workflow_usage": sum(bool(r["accounting"]["unknown_nanousd"]) for r in outcomes),
                "held_unknown_nanousd": held, "known_measurement_defects": [],
                "human_calibration": freeze["human_calibration"],
                "disputed_output_run_ids": [r["run_id"] for r in outcomes if not r["deterministic_checks"]["deterministic_passed"]],
                "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "analysis_source_sha256": hashlib.sha256((Path(__file__).parent / "astra_analysis.py").read_bytes()).hexdigest()}
    if continuation is not None:
        if total != continuation["confirmed_nanousd"] or held != continuation["unknown_nanousd"]:
            raise ValueError("Segment and phase cost bounds differ")
        evidence["continuation"] = continuation
    # Record every rubric-rejected output for human review as well as strict
    # check failures. An audit alone cannot approve the headline.
    bad_trials = {r["trial"]["trial_id"] for r in analysis["all_trials"] if not r["accepted"]}
    evidence["disputed_output_run_ids"] = [r["run_id"] for r in outcomes if r["output"] is not None and r["trial"]["trial_id"] in bad_trials]
    evidence["missing_output_failure_run_ids"] = [r["run_id"] for r in outcomes if r["output"] is None]
    return analysis, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-directory", required=True)
    parser.add_argument("--continuation-directory")
    parser.add_argument("--continuation-source-archive")
    parser.add_argument("--judge-directory", required=True)
    parser.add_argument("--bindings-path", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    analysis, evidence = review_main(main_directory=args.main_directory, judge_directory=args.judge_directory,
        bindings_path=args.bindings_path, source_archive=args.source_archive,
        continuation_directory=args.continuation_directory, continuation_source_archive=args.continuation_source_archive)
    destination = runtime._safe_path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    runtime._write_new(destination / "analysis.json", analysis)
    evidence["analysis_sha256"] = hashlib.sha256((destination / "analysis.json").read_bytes()).hexdigest()
    runtime._write_new(destination / "review.json", evidence)
    print(canonical({k: v for k, v in evidence.items() if k not in {"human_calibration", "judgments"}}))


if __name__ == "__main__":
    main()
