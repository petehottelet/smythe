"""Explicitly approved continuation with a retained prior unknown-cost reserve.

The original runner and failed workflow stay locked. This separate envelope
can execute only the unstarted suffix of the frozen schedule. It never retries
an earlier outcome and stops if another call has unresolved billing.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import time

from benchmarks import astra_runtime as runtime, astra_study as study
from benchmarks.astra_campaign import check_output, load_task_pack
from benchmarks.astra_campaign._json import canonical, strict_json
from smythe.workflow_store import SQLiteWorkflowStore

POLICY = {"retry_prior_workflows": False, "hold_full_unknown_reservation": True,
          "stop_on_new_unknown": True, "exact_cost_headlines": False}


def read_base(directory):
    base = runtime._read(Path(directory) / "study-freeze.json")
    plain = dict(base)
    if plain.pop("freeze_sha256") != runtime._sha(plain) or base["stage"] != "main":
        raise ValueError("Expected an intact frozen main study")
    return base


def audit_segment(directory, base, schedule):
    """Audit an exact contiguous prefix, including every unknown reservation."""
    directory = Path(directory)
    cases = {c.task_id: c for c in load_task_pack().tasks}
    outcomes, missing = [], False
    confirmed = unknown = 0
    unresolved = []
    with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
        for row in schedule:
            run_id = runtime._run_id(base, row)
            path = directory / f"{run_id}.outcome.json"
            if not path.exists():
                missing = True
                continue
            if missing:
                raise ValueError("Outcomes must be a contiguous schedule prefix")
            record = runtime._read(path)
            runtime._validate_outcome(record, base, row, store, cases[row["task_id"]])
            run = store.load_run(run_id)
            if run["task"] != base["tasks"][row["task_id"]] or run["budget_nanousd"] != base["per_trial_nanousd"]:
                raise ValueError("Executed task or allowance differs from its freeze")
            accounting = record["accounting"]
            if accounting["reserved_nanousd"]:
                raise ValueError("Active reservations prevent continuation")
            confirmed += accounting["confirmed_nanousd"]
            unknown += accounting["unknown_nanousd"]
            for call in accounting["calls"]:
                if call["billing_state"] == "unknown":
                    if record["status"] != "failed" or record["output"] is not None:
                        raise ValueError("Only a retained failed outcome may carry unresolved exposure")
                    envelope = store.load_evidence(call["call_id"], call["evidence_id"])
                    if (envelope["body"] or envelope["status_code"] is not None
                            or envelope["transport_error"] != "APIConnectionError"):
                        raise ValueError("This continuation covers only a no-response connection failure")
                    if call["unknown_nanousd"] != call["ceiling_nanousd"] or call["cost_nanousd"] is not None:
                        raise ValueError("The complete unknown reservation must remain held")
                    unresolved.append({"run_id": run_id, "call_id": call["call_id"],
                        "held_nanousd": call["unknown_nanousd"], "evidence_id": call["evidence_id"],
                        "evidence_sha256": record["call_evidence_sha256"][call["call_id"]]})
            outcomes.append(record)
        actual = {r["run_id"] for r in store.list_runs()}
        if actual != {r["run_id"] for r in outcomes}:
            raise ValueError("Unbound or unfinished ledger runs prevent continuation")
        if {p.name for p in directory.glob("*.outcome.json")} != {r["run_id"] + ".outcome.json" for r in outcomes}:
            raise ValueError("Unbound outcome files prevent continuation")
    if sum(r["held_nanousd"] for r in unresolved) != unknown:
        raise ValueError("Unknown call and campaign balances differ")
    return outcomes, {"completed_workflows": len(outcomes), "confirmed_nanousd": confirmed,
        "unknown_nanousd": unknown, "unresolved_calls": unresolved,
        "outcome_sha256": {r["run_id"]: r["record_sha256"] for r in outcomes}}


def admit(base, prior, continuation_confirmed):
    """Reserve the next full workflow after both known and held prior charges."""
    amounts = (prior["confirmed_nanousd"], prior["unknown_nanousd"], continuation_confirmed,
               base["per_trial_nanousd"], base["stage_allowance_nanousd"])
    for value in amounts:
        runtime._money(value, "continuation balance")
    if sum(amounts[:4]) > amounts[4]:
        raise ValueError("Confirmed charges, held exposure and the next reservation exceed the stage allowance")


def freeze_continuation(*, original_directory, directory):
    """Prepare a separate envelope; creating it grants no permission to run."""
    original = runtime._destination(original_directory)
    destination = runtime._destination(directory)
    if original == destination or Path(original) in Path(destination).parents or Path(destination) in Path(original).parents:
        raise ValueError("Continuation and original directories must be separate peers")
    base = read_base(original)
    study._verify_freeze(base)
    outcomes, prior = audit_segment(original, base, base["schedule"])
    if len(prior["unresolved_calls"]) != 1 or not 0 < len(outcomes) < len(base["schedule"]):
        raise ValueError("Expected one interrupted prefix and one retained unknown call")
    admit(base, prior, 0)
    value = {"version": 1, "method": "explicit-reserved-cost-continuation", "claimable": False,
        "original_directory": original, "directory": destination, "base_freeze_sha256": base["freeze_sha256"],
        "base": base, "prior": prior, "schedule": base["schedule"][len(outcomes):],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": dict(POLICY)}
    value["freeze_sha256"] = runtime._sha(value)
    return strict_json(canonical(value))


def verify_continuation(freeze):
    plain = dict(freeze)
    if plain.pop("freeze_sha256") != runtime._sha(plain):
        raise ValueError("Continuation freeze changed")
    if freeze["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Continuation source changed")
    if freeze["policy"] != POLICY:
        raise ValueError("Continuation reservation policy changed")
    base = read_base(freeze["original_directory"])
    if base != freeze["base"] or base["freeze_sha256"] != freeze["base_freeze_sha256"]:
        raise ValueError("Original study freeze changed")
    study._verify_freeze(base)
    outcomes, actual = audit_segment(freeze["original_directory"], base, base["schedule"])
    if actual != freeze["prior"] or freeze["schedule"] != base["schedule"][len(outcomes):]:
        raise ValueError("Prior evidence, held exposure or remaining schedule changed")
    return base


def _continuation_balance(store, base, schedule):
    ids = {runtime._run_id(base, row) for row in schedule}
    total = 0
    for run in store.list_runs():
        if run["run_id"] not in ids or run["budget_nanousd"] != base["per_trial_nanousd"]:
            raise ValueError("Unbound continuation run")
        audit = store.inspect_run(run["run_id"])
        if audit["unknown_calls"] or audit["unknown_nanousd"] or audit["reserved_nanousd"]:
            raise ValueError("New unresolved billing stops the continuation")
        if audit["confirmed_nanousd"] > base["per_trial_nanousd"]:
            raise ValueError("Continuation workflow exceeded its allowance")
        total += audit["confirmed_nanousd"]
    return total


async def run_continuation(freeze, *, approved_freeze_sha256, progress=None):
    if approved_freeze_sha256 != freeze.get("freeze_sha256"):
        raise ValueError("Explicit approval of the exact continuation freeze is required")
    freeze = strict_json(canonical(freeze))
    base = verify_continuation(freeze)
    directory = Path(freeze["directory"])
    cases = {c.task_id: c for c in load_task_pack().tasks}
    with runtime._campaign_lock(directory):
        saved = directory / "continuation-freeze.json"
        if saved.exists():
            if runtime._read(saved) != freeze:
                raise ValueError("Continuation directory belongs to another freeze")
        else:
            if set(p.name for p in directory.iterdir()) - {"campaign-lock.sqlite3", "campaign-lock.sqlite3-journal"}:
                raise ValueError("A continuation requires an empty directory")
            runtime._write_new(saved, freeze)
        outcomes = []
        with SQLiteWorkflowStore(runtime._sqlite_path(directory / "workflow.sqlite3")) as store:
            for row in freeze["schedule"]:
                verify_continuation(freeze)
                run_id = runtime._run_id(base, row)
                path = directory / f"{run_id}.outcome.json"
                if path.exists():
                    record = runtime._read(path)
                    runtime._validate_outcome(record, base, row, store, cases[row["task_id"]])
                    if record.get("continuation_freeze_sha256") != freeze["freeze_sha256"]:
                        raise ValueError("Outcome belongs to another continuation")
                    outcomes.append(record)
                    continue
                total = _continuation_balance(store, base, freeze["schedule"])
                admit(base, freeze["prior"], total)
                # This envelope never repairs or reruns a partially dispatched
                # continuation workflow. Such an interruption needs review.
                if (directory / f"{run_id}.started.json").exists() or any(r["run_id"] == run_id for r in store.list_runs()):
                    raise ValueError("An interrupted continuation trial requires separate review")
                before = time.perf_counter_ns()
                swarm = runtime._swarm(store, row, base["per_trial_nanousd"], planning_instructions=runtime.PLANNING_CONSTRAINT)
                store.create_run(base["tasks"][row["task_id"]], swarm._workflow_runtime().recipe,
                                 base["per_trial_nanousd"], run_id=run_id)
                runtime._write_new(directory / f"{run_id}.started.json", {
                    "trial": row, "run_id": run_id, "base_freeze_sha256": base["freeze_sha256"],
                    "continuation_freeze_sha256": freeze["freeze_sha256"]})
                result = error = None
                try:
                    result = await swarm.aresume(run_id)
                except BaseException as caught:
                    error = caught
                record = runtime._outcome(store, base, row, run_id, result, error, before, False)
                if record["output"] is not None:
                    record["deterministic_checks"] = check_output(cases[row["task_id"]], record["output"])
                record.pop("record_sha256")
                record["continuation_freeze_sha256"] = freeze["freeze_sha256"]
                record["record_sha256"] = runtime._sha(record)
                runtime._write_new(path, record)
                outcomes.append(record)
                if progress:
                    progress({"completed": len(outcomes), "planned": len(freeze["schedule"]),
                              "task": row["task_id"], "arm": row["arm_id"], "status": record["status"],
                              "checks": record["deterministic_checks"], "seconds": record["wall_time_ns"] / 1e9})
                if error is not None and not isinstance(error, Exception):
                    raise error
                _continuation_balance(store, base, freeze["schedule"])
            total = _continuation_balance(store, base, freeze["schedule"])
        summary = {"freeze_sha256": freeze["freeze_sha256"], "completed_workflows": len(outcomes),
            "confirmed_nanousd": total, "prior_confirmed_nanousd": freeze["prior"]["confirmed_nanousd"],
            "held_unknown_nanousd": freeze["prior"]["unknown_nanousd"], "claimable": False,
            "outcome_sha256": {r["run_id"]: r["record_sha256"] for r in outcomes}}
        saved = directory / "continuation-summary.json"
        if saved.exists():
            if runtime._read(saved) != summary:
                raise ValueError("Continuation summary changed")
        else:
            runtime._write_new(saved, summary)
        return summary


def run_approved(freeze, approved_freeze_sha256, progress=None):
    return asyncio.run(run_continuation(freeze, approved_freeze_sha256=approved_freeze_sha256, progress=progress))
