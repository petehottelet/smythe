"""Frozen, single-attempt native Fable pilot and human-gated main execution.

Importing this module never loads credentials or sends a request. The original
preparation record remains immutable; this runner creates a new runtime freeze.
"""

import argparse
import asyncio
import hashlib
from importlib import metadata
from pathlib import Path
import sys
import time

from benchmarks import astra_runtime as evidence
from benchmarks import astra_study as study
from benchmarks.astra_campaign import check_output, load_task_pack
from smythe import Swarm, WorkflowGraphPolicy
from smythe.planner import LLMArchitect
from smythe.pricing_anthropic import PRICE_VERSION
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.task import task_to_dict
from smythe.workflow_binding import LocalOnly
from smythe.workflow_store import SQLiteWorkflowStore

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "benchmarks/fable_51_preparation_20260913.json"
PRIOR = ROOT / "benchmarks/results/astra_20260913_main/campaign-spending.json"
# Tighter allocations leave room for the separately measured Code Workflow arm.
ALLOCATIONS = {"pilot": 10_000_000_000, "main": 60_000_000_000,
               "ultracode-pilot": 5_000_000_000, "ultracode-main": 15_000_000_000,
               "judge": 10_000_000_000}
CAP = 5_000_000_000


def _sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "smythe").glob("*.py")]
    names += ["benchmarks/fable_runtime.py", "benchmarks/fable_51_benchmark_plan.md",
              "benchmarks/astra_runtime.py", "benchmarks/astra_study.py", "benchmarks/astra_evaluation.py"]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "benchmarks/astra_campaign").rglob("*")
              if p.suffix in {".py", ".json"}]
    return {name: _sha_bytes(ROOT / name) for name in sorted(set(names))}


def dependencies():
    result = {"python": sys.version}
    for name in ("anthropic", "httpx2", "pyyaml"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _write_same(path, value):
    if path.exists():
        if evidence._read(path) != value:
            raise ValueError(f"Immutable evidence changed: {path.name}")
    else:
        evidence._write_new(path, value)


def inspect(directory):
    directory = Path(directory)
    freeze = evidence._read(directory / "study-freeze.json")
    plain = dict(freeze)
    if plain.pop("freeze_sha256") != evidence._sha(plain):
        raise ValueError("Fable freeze changed")
    cases = {case.task_id: case for case in load_task_pack().tasks}
    records = []
    with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
        total = study._balance(store, freeze)
        for row in freeze["schedule"]:
            path = directory / f"{evidence._run_id(freeze, row)}.outcome.json"
            if path.exists():
                record = evidence._read(path)
                evidence._validate_outcome(record, freeze, row, store, cases[row["task_id"]])
                records.append(record)
    return {"freeze_sha256": freeze["freeze_sha256"], "confirmed_nanousd": total,
            "records": records, "complete": len(records) == len(freeze["schedule"])}


def human_gate(directory, response_path, manifest_path):
    """Bind actual user ratings to every required medium answer and flagged high answer."""
    result = inspect(directory)
    if not result["complete"] or len(result["records"]) != 12:
        raise ValueError("All twelve pilot attempts must be recorded")
    medium = [r for r in result["records"] if r["trial"]["effort"] == "medium"]
    if len(medium) != 6 or any(r["status"] != "completed" or not r["deterministic_checks"]["deterministic_passed"] for r in medium):
        raise ValueError("All six medium pilot workflows must pass before main")
    response, manifest = evidence._read(response_path), evidence._read(manifest_path)
    if (response.get("status") != "submitted" or response.get("source") != "local-user-review-form"
            or response.get("sample_manifest_sha256") != _sha_bytes(manifest_path) or not response.get("submitted_at")):
        raise ValueError("A bound, submitted human review is required")
    required = {r["run_id"]: r for r in result["records"] if r["trial"]["effort"] == "medium"
                or r["status"] != "completed" or not r["deterministic_checks"]["deterministic_passed"]}
    samples = {r["sample_id"]: r for r in manifest}
    if len(samples) != len(manifest):
        raise ValueError("Duplicate human review samples")
    seen = set()
    for rating in response.get("ratings", []):
        sample = samples.get(rating.get("sample_id"))
        if not sample or sample["run_id"] not in required or sample["run_id"] in seen:
            raise ValueError("Unexpected or repeated human rating")
        record = required[sample["run_id"]]
        if (sample.get("output_sha256") != record["output_sha256"]
                or rating.get("output_sha256") != record["output_sha256"]
                or sample.get("output") != record["output"]
                or type(rating.get("score")) is not int or not 0 <= rating["score"] <= 4):
            raise ValueError("Human output/rating binding differs from pilot")
        if record["trial"]["effort"] == "medium" and rating["score"] < 3:
            raise ValueError("Human medium review has not passed")
        seen.add(sample["run_id"])
    if seen != set(required):
        raise ValueError("Human review is incomplete")
    return {"status": "passed", "pilot_freeze_sha256": result["freeze_sha256"],
            "response_sha256": _sha_bytes(response_path), "manifest_sha256": _sha_bytes(manifest_path),
            "samples": len(seen), "pilot_confirmed_nanousd": result["confirmed_nanousd"]}


def freeze(*, directory, stage="pilot", model_access, validation, pilot_directory=None,
           human_response=None, human_manifest=None):
    if stage not in {"pilot", "main"}:
        raise ValueError("Native stage must be pilot or main")
    if Path(directory).name != stage:
        raise ValueError("Use the named stage inside one campaign directory")
    prep, prior = evidence._read(PREPARATION), evidence._read(PRIOR)
    if _sha_bytes(PRIOR) != prep["spending"]["prior_record_sha256"]:
        raise ValueError("Reconcile changed prior spending before preparing a new freeze")
    access, checks = evidence._read(model_access), evidence._read(validation)
    if access.get("model") != "claude-fable-5-1" or access.get("status") != "available":
        raise ValueError("Exact Fable metadata access receipt required")
    if checks.get("status") != "passed" or not checks.get("full_offline_suite") or not checks.get("ruff"):
        raise ValueError("Native offline contracts, full suite and Ruff must pass")
    if checks.get("source_sha256") != sources():
        raise ValueError("Validation must bind the exact runtime sources")
    calibration = None
    if stage == "main":
        if any(v is None for v in (pilot_directory, human_response, human_manifest)):
            raise ValueError("Main requires the completed pilot and actual human review")
        calibration = human_gate(pilot_directory, human_response, human_manifest)
    cases = {case.task_id: case for case in load_task_pack().tasks}
    rows = [row for row in prep["schedule"] if row["stage"] == stage]
    value = {"version": 1, "stage": stage, "claimable": False, "directory": str(Path(directory).resolve()),
             "preparation_sha256": _sha_bytes(PREPARATION), "prior_spending_sha256": _sha_bytes(PRIOR),
             "prior_total_upper_nanousd": prior["campaign_upper_nanousd"],
             "retained_old_unknown_nanousd": prior["unknown_nanousd"],
             "allocations_nanousd": ALLOCATIONS, "stage_allowance_nanousd": ALLOCATIONS[stage],
             "per_trial_nanousd": CAP, "price_version": PRICE_VERSION, "dependencies": dependencies(),
             "source_sha256": sources(), "model_access": access, "validation": checks,
             "schedule": rows, "tasks": {row["task_id"]: task_to_dict(study.study_task(cases[row["task_id"]])) for row in rows},
             "human_calibration": calibration, "human_required_before_main": True,
             "evaluation": prep["evaluation"],
             "policy": {**prep["request_policy"], **prep["execution_policy"]}, "gates": study.GATES}
    value["freeze_sha256"] = evidence._sha(value)
    return value


def campaign_balance(root):
    """Include every stage under the shared extension ceiling, including Code reserves."""
    total = 0
    for stage, allowance in ALLOCATIONS.items():
        path = root / stage
        if not path.exists():
            continue
        if stage in {"pilot", "main"}:
            if not (path / "study-freeze.json").exists():
                raise ValueError("Unbound native stage directory")
            frozen = evidence._read(path / "study-freeze.json")
            if frozen["stage_allowance_nanousd"] != allowance:
                raise ValueError("Saved stage allowance changed")
            with SQLiteWorkflowStore(path / "workflow.sqlite3") as store:
                total += study._balance(store, frozen)
        else:
            # Other runners must settle this shared exposure record before another
            # stage may spend. Merely lacking a cost field is never zero cost.
            ledger = evidence._read(path / "spending.json")
            amount = ledger.get("confirmed_nanousd")
            if (type(amount) is not int or not 0 <= amount <= allowance
                    or ledger.get("status") != "settled" or ledger.get("reserved_nanousd") != 0
                    or ledger.get("unknown_nanousd") != 0):
                raise ValueError("Other Fable stage has unresolved or excessive spending")
            total += amount
    if total > sum(ALLOCATIONS.values()):
        raise ValueError("Fable extension ceiling exceeded")
    return total


def verify(frozen):
    plain = dict(frozen)
    if plain.pop("freeze_sha256") != evidence._sha(plain):
        raise ValueError("Frozen Fable campaign changed")
    if frozen["source_sha256"] != sources() or frozen["dependencies"] != dependencies():
        raise ValueError("Runtime changed after freeze")
    if frozen["prior_spending_sha256"] != _sha_bytes(PRIOR) or frozen["preparation_sha256"] != _sha_bytes(PREPARATION):
        raise ValueError("Bound campaign records changed")
    if frozen["stage"] == "main" and (frozen.get("human_calibration") or {}).get("status") != "passed":
        raise ValueError("Human review required before main")


def swarm(store, row):
    provider = AnthropicMessagesProvider(reasoning_effort=row["effort"], max_output_tokens=8192, request_timeout_s=600)
    architect = (LocalOnly(evidence._FixedArchitect, identity="astra-fixed-research-analysis-writing", version="1")
        if row["strategy"] == "fixed_pipeline" else LLMArchitect(provider, planning_model=row["model"],
            max_retries=0, planning_instructions=evidence.PLANNING_CONSTRAINT))
    return Swarm(model=row["model"], provider=provider, architect=architect,
                 synthesizer=Synthesizer(SynthesisStrategy.DELIVERABLE), run_store=store,
                 max_budget_usd=5.0, parallel=True, max_concurrency=8, max_revisions=0,
                 graph_policy=WorkflowGraphPolicy(8, node_model=row["model"], max_retries=0, max_regenerations=0))


async def run(frozen):
    verify(frozen)
    directory = Path(frozen["directory"])
    cases = {case.task_id: case for case in load_task_pack().tasks}
    with evidence._campaign_lock(directory.parent):
        directory.mkdir(exist_ok=True)
        _write_same(directory.parent / "campaign-budget.json", {
            "allocations_nanousd": ALLOCATIONS, "total_nanousd": 100_000_000_000,
            "prior_spending_sha256": frozen["prior_spending_sha256"],
            "prior_total_upper_nanousd": frozen["prior_total_upper_nanousd"]})
        _write_same(directory / "study-freeze.json", frozen)
        with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
            records = []
            for row in frozen["schedule"]:
                verify(frozen)
                campaign_balance(directory.parent)
                run_id = evidence._run_id(frozen, row)
                path = directory / f"{run_id}.outcome.json"
                if path.exists():
                    record = evidence._read(path)
                    evidence._validate_outcome(record, frozen, row, store, cases[row["task_id"]])
                    records.append(record)
                    continue
                total = study._balance(store, frozen)
                if total + CAP > frozen["stage_allowance_nanousd"]:
                    raise ValueError("Stage cannot reserve another complete workflow")
                started = directory / f"{run_id}.started.json"
                if started.exists():
                    raise ValueError("Interrupted attempt requires local evidence recovery; automatic dispatch disabled")
                before = time.perf_counter_ns()
                execution = swarm(store, row)
                store.create_run(frozen["tasks"][row["task_id"]], execution._workflow_runtime().recipe, CAP, run_id=run_id)
                evidence._write_new(started, {"trial": row, "run_id": run_id, "freeze_sha256": frozen["freeze_sha256"]})
                result, error = None, None
                try:
                    result = await execution.aresume(run_id)
                except BaseException as caught:
                    error = caught
                record = evidence._outcome(store, frozen, row, run_id, result, error, before, False)
                if record["output"] is not None:
                    record["deterministic_checks"] = check_output(cases[row["task_id"]], record["output"])
                record.pop("record_sha256")
                record["record_sha256"] = evidence._sha(record)
                evidence._write_new(path, record)
                records.append(record)
                print(f"{len(records)}/{len(frozen['schedule'])} {row['trial_id']} {record['status']} "
                      f"contract={record['deterministic_checks']['deterministic_passed']} "
                      f"cost_nanousd={record['accounting']['confirmed_nanousd']}", flush=True)
                if error is not None and not isinstance(error, Exception):
                    raise error
                study._balance(store, frozen)
            summary = study._summary(frozen, records, study._balance(store, frozen))
            _write_same(directory / "study-summary.json", summary)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_args = sub.add_parser("inspect")
    inspect_args.add_argument("directory")
    run_args = sub.add_parser("run")
    run_args.add_argument("freeze")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        value = inspect(args.directory)
        print(f"{len(value['records'])} outcomes; {value['confirmed_nanousd']} nanoUSD; complete={value['complete']}")
    else:
        asyncio.run(run(evidence._read(args.freeze)))


if __name__ == "__main__":
    main()
