"""Follow-up format pilot and human-calibrated, bounded Astra main study.

Earlier evidence is immutable. The follow-up states required field types and
decision labels in provider inputs; it never exposes factual answer values.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path
import time

from benchmarks import astra_runtime as pilot
from benchmarks.astra_campaign import check_output, load_task_pack, prepare_campaign, provider_task
from benchmarks.astra_campaign._json import canonical, strict_json
from benchmarks.astra_evaluation import text_contract
from smythe import Task
from smythe.task import task_to_dict
from smythe.workflow_store import SQLiteWorkflowStore

GATES = {"criterion_minimum": 3, "material_defects_allowed": 0,
         "minimum_success_rate": 0.90, "quality_noninferiority_margin": 0.25,
         "quality_margin_unit": "mean anchored 0..4 criterion score",
         "sampling_unit": "task", "bootstrap_seed": 14173,
         "bootstrap_repetitions": 10000, "human_calibration_required": True}


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def study_task(case):
    data = provider_task(case)
    # Graph construction policy belongs only in the architect's prompt. Putting
    # it in Task.constraints also asks fixed executors to emit graph metadata.
    if any(c["kind"] == "keys" and c["path"] == [] for c in case.checks):
        data["constraints"].append(
            "Use exactly the requested top-level JSON fields. Do not add workflow metadata "
            "or explanatory fields that the task does not request.")
    contract = text_contract(case)
    if contract:
        data["constraints"].append(contract)
    numeric_paths = ["/" + "/".join(map(str, c["path"])) for c in case.checks if c["kind"] == "number"]
    if numeric_paths:
        data["constraints"].append(
            "These JSON paths must contain JSON numbers, not quoted strings: "
            + ", ".join(numeric_paths) + ". Compute their values from the sources.")
    decision_labels = {"main-capacity-chain": ('"overtime"', '"regular"'),
                       "main-production-chain": ('"expedite"', '"normal"')}
    if case.task_id in decision_labels:
        data["constraints"].append(
            "The decision field must be exactly one of the JSON strings "
            + " or ".join(decision_labels[case.task_id])
            + ". Choose the applicable label using the source rule; add no prose to this field.")
    return Task(**data)


def inspect_pilot(directory):
    """Reconcile saved pilot outputs and native evidence without another call."""
    directory = Path(directory)
    binding = pilot._read(directory / "campaign.json")
    freeze = binding["freeze"]
    summary = pilot._read(directory / "pilot-summary.json")
    rows = prepare_campaign(seed=freeze["seed"])["schedules"]["pilot"]
    if binding != pilot._binding(freeze, rows) or summary["workflow_runs"] != 12:
        raise ValueError("Incomplete original pilot identity")
    cases = {c.task_id: c for c in load_task_pack().tasks}
    outcomes = []
    with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
        confirmed = pilot._clear_accounting(store, freeze, rows)
        for row in rows:
            outcome = pilot._read(directory / f"{pilot._run_id(freeze, row)}.outcome.json")
            pilot._validate_outcome(outcome, freeze, row, store, cases[row["task_id"]])
            outcomes.append(outcome)
    expected = [{"trial_id": r["trial"]["trial_id"], "run_id": r["run_id"],
                 "record_sha256": r["record_sha256"], "path": f"{r['run_id']}.outcome.json"}
                for r in outcomes]
    if summary["outcome_receipts"] != expected or summary["confirmed_nanousd"] != confirmed:
        raise ValueError("Pilot summary differs from native evidence")
    return {"freeze": freeze, "confirmed_nanousd": confirmed,
            "summary_sha256": _hash(summary), "outcomes": outcomes}


def inspect_human_calibration(response_path, manifest_path, original):
    response, manifest = pilot._read(response_path), pilot._read(manifest_path)
    digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if (response.get("status") != "submitted" or response.get("source") != "local-user-review-form"
            or response.get("sample_manifest_sha256") != digest or not response.get("submitted_at")):
        raise ValueError("A submitted, bound human calibration is required")
    if len(manifest) != 6 or len(response.get("ratings", [])) != 6:
        raise ValueError("Human calibration requires all six samples")
    samples = {r["sample_id"]: r for r in manifest}
    if len(samples) != 6:
        raise ValueError("Duplicate human sample IDs")
    outcomes = {r["run_id"]: r for r in original["outcomes"]}
    arms, controls, seen = set(), 0, set()
    for rating in response["ratings"]:
        identity = rating.get("sample_id")
        if identity not in samples or identity in seen:
            raise ValueError("Invalid human sample identity")
        seen.add(identity)
        sample = samples[identity]
        if (rating.get("output_sha256") != sample["output_sha256"]
                or hashlib.sha256(sample["output"].encode()).hexdigest() != sample["output_sha256"]
                or type(rating.get("score")) is not int or not 0 <= rating["score"] <= 4):
            raise ValueError("Invalid human rating or output binding")
        if sample["origin"] == "negative-control":
            controls += 1
            if rating["score"] >= 3:
                raise ValueError("Human calibration accepted the deliberately defective control")
        elif sample["origin"] == "pilot" and sample["run_id"] in outcomes:
            outcome = outcomes[sample["run_id"]]
            if outcome["output_sha256"] != sample["output_sha256"]:
                raise ValueError("Human sample does not match a real pilot answer")
            if outcome["trial"]["task_id"] == "pilot-membership":
                arms.add(outcome["trial"]["arm_id"])
            if rating["score"] < 3:
                raise ValueError("Human pilot judgment requires further calibration")
        else:
            raise ValueError("Unknown human calibration sample origin")
    if len(arms) != 4 or controls != 1:
        raise ValueError("Human sample must balance all four arms and one negative control")
    return {"status": "passed", "response_sha256": _hash(response),
            "manifest_sha256": digest, "balanced_arms": 4, "samples": 6,
            "method": "Human acceptance ratings: real samples >=3; planted-error control <3."}


def _prior_stages(directories, stage):
    receipts, seen, ancestors = [], set(), []
    for directory in directories:
        path = str(Path(directory).resolve())
        summary = inspect_stage(path)
        freeze = pilot._read(Path(path) / "study-freeze.json")
        if summary["stage"] != stage or freeze["freeze_sha256"] in seen:
            raise ValueError("Prior stages must be distinct direct studies of the same stage")
        ancestors.extend(freeze.get("superseded_studies", []))
        seen.add(freeze["freeze_sha256"])
        receipts.append({"directory": path, "summary_sha256": _hash(summary),
                         "confirmed_nanousd": summary["confirmed_nanousd"]})
    if any(ancestor not in receipts for ancestor in ancestors):
        raise ValueError("Include every earlier study; do not drop its spending from the allowance")
    return receipts


def freeze_stage(*, stage, directory, original_pilot, format_pilot=None,
                 human_response=None, human_manifest=None, judge_evidence=None,
                 previous_stages=()):
    if stage not in {"format-pilot", "main"}:
        raise ValueError("Unknown study stage")
    original = inspect_pilot(original_pilot)
    allowance = original["freeze"]["allowances"]
    previous = _prior_stages(previous_stages, stage)
    carried = sum(r["confirmed_nanousd"] for r in previous)
    calibration = None
    if stage == "main":
        from benchmarks.astra_evidence import inspect_calibration

        if any(v is None for v in (format_pilot, human_response, human_manifest, judge_evidence)):
            raise ValueError("Main requires format-pilot, human calibration and frozen judge evidence")
        revised = inspect_stage(format_pilot)
        if (revised["stage"] != "format-pilot" or revised["completed_workflows"] != 12
                or revised["execution_failures"] or revised["deterministic_failures"]):
            raise ValueError("The format pilot must complete all twelve output contracts")
        calibration = inspect_human_calibration(human_response, human_manifest, original)
        judge = pilot._read(judge_evidence)
        if judge.get("status") != "calibrated-pilot" or judge.get("negative_control_detected") is not True:
            raise ValueError("Judge must detect the negative control before main execution")
        if inspect_calibration(original=original, **judge["evidence_paths"]) != judge:
            raise ValueError("Judge calibration differs from complete native evidence")
        revised_freeze = pilot._read(Path(format_pilot) / "study-freeze.json")
        prior_pilot = original["confirmed_nanousd"] + revised_freeze.get("prior_stage_cost_nanousd", 0)
        if revised["confirmed_nanousd"] + prior_pilot > allowance["pilot_nanousd"]:
            raise ValueError("Combined pilot cost exceeds the authorized pilot allocation")
        allocation = allowance["main_nanousd"] - carried
    else:
        prior_pilot = original["confirmed_nanousd"] + carried
        allocation = allowance["pilot_nanousd"] - prior_pilot
    if allocation < allowance["per_trial_nanousd"]:
        raise ValueError("Prior spending leaves no full workflow reservation")
    cases = {c.task_id: c for c in load_task_pack().tasks}
    preparation = prepare_campaign(seed=original["freeze"]["seed"])
    schedule = preparation["schedules"]["main" if stage == "main" else "pilot"]
    source = pilot._source_hashes()
    for name in ("benchmarks/astra_study.py", "benchmarks/astra_evaluation.py",
                 "benchmarks/astra_evidence.py", "benchmarks/astra_analysis.py"):
        source[name] = hashlib.sha256((pilot.ROOT / name).read_bytes()).hexdigest()
    value = {"version": 1, "stage": stage, "claimable": False,
             "directory": pilot._destination(directory), "source_sha256": source,
             "dependencies": pilot._dependencies(), "policy": pilot.POLICY,
             "seed": original["freeze"]["seed"],
             "preparation_sha256": preparation["preparation_sha256"],
             "stage_allowance_nanousd": allocation, "per_trial_nanousd": allowance["per_trial_nanousd"],
             "prior_stage_cost_nanousd": carried, "superseded_studies": previous,
             "method": "planner-only-policy-and-explicit-field-types-v3",
             "task_exposure": "amended task reuse; not an untouched holdout" if stage == "main" and previous else "original task split",
             "campaign_allocations": allowance, "original_pilot_sha256": original["summary_sha256"],
             "prior_pilot_cost_nanousd": prior_pilot,
             "schedule": schedule, "tasks": {r["task_id"]: task_to_dict(study_task(cases[r["task_id"]])) for r in schedule},
             "gates": GATES, "human_calibration": calibration,
             "judge_evidence_sha256": None if judge_evidence is None else _hash(pilot._read(judge_evidence)),
             "format_pilot_sha256": None if format_pilot is None else _hash(pilot._read(Path(format_pilot) / "study-summary.json"))}
    value = strict_json(canonical(value))
    value["freeze_sha256"] = _hash(value)
    return value


def _verify_freeze(freeze):
    plain = dict(freeze)
    if plain.pop("freeze_sha256") != _hash(plain):
        raise ValueError("Study freeze content changed")
    previous = freeze.get("superseded_studies", [])
    if previous:
        actual = _prior_stages([r["directory"] for r in previous], freeze["stage"])
        if actual != previous or sum(r["confirmed_nanousd"] for r in actual) != freeze["prior_stage_cost_nanousd"]:
            raise ValueError("Prior study spending or evidence changed")
    if freeze["stage"] == "main" and (
        not freeze.get("human_calibration") or freeze["human_calibration"].get("status") != "passed"
        or not freeze.get("judge_evidence_sha256") or not freeze.get("format_pilot_sha256")
    ):
        raise ValueError("Main execution requires all frozen calibration evidence")
    if freeze["dependencies"] != pilot._dependencies() or freeze["policy"] != pilot.POLICY:
        raise ValueError("Frozen dependency or execution policy changed")
    if freeze["preparation_sha256"] != prepare_campaign(seed=freeze["seed"])["preparation_sha256"]:
        raise ValueError("Frozen task pack, rubric or schedule changed")
    for name, expected in freeze["source_sha256"].items():
        # Pilot runtime uses portable LF source hashes; new modules are written LF.
        from benchmarks.astra_campaign._json import digest
        if digest((pilot.ROOT / name).read_bytes()) != expected:
            raise ValueError(f"Measured study source changed: {name}")


def _balance(store, freeze):
    ids = {pilot._run_id(freeze, row) for row in freeze["schedule"]}
    confirmed = 0
    for row in store.list_runs():
        if row["run_id"] not in ids or row["budget_nanousd"] != freeze["per_trial_nanousd"]:
            raise ValueError("Unbound study ledger entry")
        audit = store.inspect_run(row["run_id"])
        if audit["confirmed_nanousd"] > freeze["per_trial_nanousd"]:
            raise ValueError("Trial cost exceeds its allowance")
        if audit["unknown_calls"] or audit["reserved_nanousd"] or audit["unknown_nanousd"]:
            raise ValueError("Unresolved study billing blocks admission")
        confirmed += audit["confirmed_nanousd"]
    if confirmed > freeze["stage_allowance_nanousd"]:
        raise ValueError("Study stage allowance exceeded")
    return confirmed


def _summary(freeze, outcomes, confirmed):
    return {"version": 1, "stage": freeze["stage"], "freeze_sha256": freeze["freeze_sha256"],
            "claimable": False, "completed_workflows": len(outcomes),
            "planned_workflows": len(freeze["schedule"]),
            "execution_failures": sum(r["status"] != "completed" for r in outcomes),
            "deterministic_failures": sum(not r["deterministic_checks"]["deterministic_passed"] for r in outcomes),
            "confirmed_nanousd": confirmed, "quality_evaluated": False,
            "outcomes": [{"trial": r["trial"], "run_id": r["run_id"], "record_sha256": r["record_sha256"]} for r in outcomes]}


def inspect_stage(directory):
    directory = Path(directory)
    freeze = pilot._read(directory / "study-freeze.json")
    plain = dict(freeze)
    if plain.pop("freeze_sha256") != _hash(plain):
        raise ValueError("Study freeze content changed")
    cases = {c.task_id: c for c in load_task_pack().tasks}
    outcomes = []
    with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
        confirmed = _balance(store, freeze)
        for row in freeze["schedule"]:
            path = directory / f"{pilot._run_id(freeze, row)}.outcome.json"
            if not path.exists():
                continue
            record = pilot._read(path)
            pilot._validate_outcome(record, freeze, row, store, cases[row["task_id"]])
            outcomes.append(record)
    actual = _summary(freeze, outcomes, confirmed)
    if actual != pilot._read(directory / "study-summary.json"):
        raise ValueError("Study summary differs from complete native trial evidence")
    return actual


async def run_stage(freeze):
    """Execute the frozen schedule once, with persistent per-trial identities."""
    freeze = strict_json(canonical(freeze))
    _verify_freeze(freeze)
    directory = Path(freeze["directory"])
    cases = {c.task_id: c for c in load_task_pack().tasks}
    with pilot._campaign_lock(directory):
        path = directory / "study-freeze.json"
        if path.exists():
            if pilot._read(path) != freeze:
                raise ValueError("Study directory belongs to another freeze")
        else:
            if set(p.name for p in directory.iterdir()) - {"campaign-lock.sqlite3", "campaign-lock.sqlite3-journal"}:
                raise ValueError("New study requires an empty directory")
            pilot._write_new(path, freeze)
        with SQLiteWorkflowStore(pilot._sqlite_path(directory / "workflow.sqlite3")) as store:
            outcomes = []
            for row in freeze["schedule"]:
                _verify_freeze(freeze)
                run_id = pilot._run_id(freeze, row)
                path = directory / f"{run_id}.outcome.json"
                if path.exists():
                    record = pilot._read(path)
                    pilot._validate_outcome(record, freeze, row, store, cases[row["task_id"]])
                    outcomes.append(record)
                    continue
                confirmed = _balance(store, freeze)
                if confirmed + freeze["per_trial_nanousd"] > freeze["stage_allowance_nanousd"]:
                    raise ValueError("Stage allowance cannot admit another full trial reservation")
                start_path = directory / f"{run_id}.started.json"
                resumed = start_path.exists()
                before = time.perf_counter_ns()
                swarm = pilot._swarm(store, row, freeze["per_trial_nanousd"],
                                     planning_instructions=pilot.PLANNING_CONSTRAINT)
                store.create_run(freeze["tasks"][row["task_id"]], swarm._workflow_runtime().recipe,
                                 freeze["per_trial_nanousd"], run_id=run_id)
                start = {"trial": row, "freeze_sha256": freeze["freeze_sha256"], "run_id": run_id}
                if resumed:
                    if pilot._read(start_path) != start:
                        raise ValueError("Study start receipt changed")
                else:
                    pilot._write_new(start_path, start)
                result, error = None, None
                try:
                    result = await swarm.aresume(run_id)
                except BaseException as caught:
                    error = caught
                record = pilot._outcome(store, freeze, row, run_id, result, error, before, resumed)
                if record["output"] is not None:
                    record["deterministic_checks"] = check_output(cases[row["task_id"]], record["output"])
                    record.pop("record_sha256")
                    record["record_sha256"] = pilot._sha(record)
                pilot._write_new(path, record)
                outcomes.append(record)
                if error is not None and not isinstance(error, Exception):
                    raise error
                _balance(store, freeze)
            summary = _summary(freeze, outcomes, _balance(store, freeze))
            path = directory / "study-summary.json"
            if path.exists():
                if pilot._read(path) != summary:
                    raise ValueError("Study summary changed")
            else:
                pilot._write_new(path, summary)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("stage", choices=["format-pilot", "main"])
    create.add_argument("--directory", required=True)
    create.add_argument("--original-pilot", required=True)
    create.add_argument("--format-pilot")
    create.add_argument("--human-response")
    create.add_argument("--human-manifest")
    create.add_argument("--judge-evidence")
    create.add_argument("--previous-stage", action="append", default=[])
    create.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--freeze", required=True)
    run.add_argument("--approved-freeze-sha256", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze_stage(stage=args.stage, directory=args.directory, original_pilot=args.original_pilot,
                             format_pilot=args.format_pilot, human_response=args.human_response,
                             human_manifest=args.human_manifest, judge_evidence=args.judge_evidence,
                             previous_stages=args.previous_stage)
        pilot._write_new(Path(args.output), value)
        print(canonical({"freeze_sha256": value["freeze_sha256"], "stage": value["stage"]}))
    elif args.command == "run":
        value = pilot._read(args.freeze)
        if args.approved_freeze_sha256 != value["freeze_sha256"]:
            parser.error("Execution requires approval of the exact saved freeze")
        print(canonical(asyncio.run(run_stage(value))))
    else:
        print(canonical(inspect_stage(args.directory)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
