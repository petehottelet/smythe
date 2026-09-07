"""Hash-bound, allowance-gated Astra pilot execution; main and judging stay closed.

This benchmark module is separate from the historical preparation package.
Importing it does not inspect credentials, initialize an SDK, or execute work.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
from importlib import metadata
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import time
from uuid import uuid4

from benchmarks.astra_campaign import check_output, load_task_pack, prepare_campaign, provider_task
from benchmarks.astra_campaign._json import CampaignPlanError, canonical, digest, strict_json

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ("benchmarks/astra_runtime.py", "tests/test_astra_runtime.py")
MAX_MONEY = 2**63 - 1
MAX_RECORD_BYTES = 4 * 1024 * 1024
PLANNING_CONSTRAINT = (
    "Use at most eight execution nodes, all with the arm's execution model. "
    "Every node must explicitly set max_retries: 0 and max_regenerations: 0. "
    "Use no tools or outside sources. Produce one terminal node containing the "
    "complete requested bare JSON deliverable."
)
FIXED_STEPS = (
    ("research", "Extract and organize the supplied evidence relevant to the task.",
     "You are a research analyst. Use only the supplied fictional source pack."),
    ("analysis", "Analyze the evidence, calculate the required results, and check the task constraints.",
     "You are an analyst. Check the supplied evidence and prior research carefully."),
    ("writing", "Produce the complete final answer in exactly the requested bare JSON format.",
     "You are a writer. Deliver the requested answer using the supplied evidence and analysis."),
)
POLICY = {
    "endpoint": "responses", "reasoning_effort": "medium", "service_tier": "default",
    "endpoint_scope": "global", "max_output_tokens": 8192, "sdk_retries": 0,
    "request_timeout_s": 600, "tools": False, "external_search": False,
    "max_nodes": 8, "max_concurrency": 8, "workflow_concurrency": 1,
    "node_max_retries": 0, "max_regenerations": 0, "planning_repairs": 0,
    "max_revisions": 0, "synthesis": "deliverable", "sampling_parameters": [],
}
ALLOWANCE_KEYS = {"total_nanousd", "pilot_nanousd", "main_nanousd", "judge_nanousd",
                  "per_trial_nanousd"}
MAIN_BLOCKER = "Main execution requires a reviewed paid pilot and frozen human-calibrated acceptance gates."
JUDGE_BLOCKER = "Judge execution requires a separately frozen judge identity and durable accounting implementation."


class CampaignRuntimeError(CampaignPlanError):
    """The pilot cannot safely execute this configuration or durable state."""


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _money(value, name, *, positive=False):
    if type(value) is not int or not (int(positive) <= value <= MAX_MONEY):
        raise CampaignRuntimeError(f"{name} must be a strict {'positive' if positive else 'nonnegative'} nanoUSD integer")
    return value


def _budget_usd(nanousd):
    """Refuse a float conversion that would change the authorized ledger cap."""
    value = float(f"{nanousd // 10**9}.{nanousd % 10**9:09d}")
    amount = Decimal(str(value)).as_tuple()
    coefficient = int("".join(map(str, amount.digits)))
    exponent = amount.exponent + 9
    converted = coefficient * 10**exponent if exponent >= 0 else coefficient // 10**-exponent
    if converted != nanousd:
        raise CampaignRuntimeError("Per-trial allowance cannot be represented by the workflow USD cap exactly")
    return value


def _allowances(value):
    if value is None:
        return None
    if type(value) is not dict or value.keys() != ALLOWANCE_KEYS:
        raise CampaignRuntimeError("All total, pilot, main, judge and per-trial allowances are required")
    result = {key: _money(value[key], key, positive=key in {"total_nanousd", "pilot_nanousd", "per_trial_nanousd"})
              for key in sorted(ALLOWANCE_KEYS)}
    if sum(result[key] for key in ("pilot_nanousd", "main_nanousd", "judge_nanousd")) > result["total_nanousd"]:
        raise CampaignRuntimeError("Stage allowances exceed the total campaign allowance")
    if 12 * result["per_trial_nanousd"] > result["pilot_nanousd"]:
        raise CampaignRuntimeError("Twelve fixed trial caps exceed the pilot allowance")
    _budget_usd(result["per_trial_nanousd"])
    return result


def _source_hashes():
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "smythe/*.py"], cwd=ROOT,
        capture_output=True, check=True,
    )
    names = {name.decode("utf-8") for name in listing.stdout.split(b"\0") if name}
    if "smythe/workflow.py" not in names or "smythe/provider_responses.py" not in names:
        raise CampaignRuntimeError("Tracked runtime source inventory is incomplete")
    names.update((*RUNTIME_FILES, "pyproject.toml"))
    # Explicit files include the new harness before its first commit. Ignored
    # scratch files and Git status never decide whether a runtime may run.
    return {name: digest((ROOT / name).read_bytes()) for name in sorted(names)}


def _dependencies():
    versions = {"python": sys.version}
    for package in ("smythe", "openai", "httpx", "httpx2", "pyyaml"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _safe_path(path, *, file=False):
    """Reject links/reparse ancestry before an operation can follow them."""
    path = Path(os.path.abspath(path))
    for candidate in (*reversed(path.parents), path):
        if os.name == "nt" and candidate != candidate.parent:
            part = candidate.name
            stem = part.split(".", 1)[0].upper()
            if (part.rstrip(" .") != part or ":" in part
                    or stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                *(f"LPT{i}" for i in range(1, 10))}):
                raise CampaignRuntimeError("Campaign path has an aliased or reserved Windows component")
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise CampaignRuntimeError("Campaign paths cannot traverse symbolic links or reparse points")
        if candidate != path and not stat.S_ISDIR(info.st_mode):
            raise CampaignRuntimeError("Campaign path ancestry must contain directories")
        if candidate == path and not (stat.S_ISREG(info.st_mode) if file else stat.S_ISDIR(info.st_mode)):
            raise CampaignRuntimeError("Campaign path has an unexpected filesystem type")
    return path


def _sqlite_path(path):
    for suffix in ("", "-wal", "-shm", "-journal"):
        _safe_path(Path(str(path) + suffix), file=True)
    return path


def _destination(directory):
    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise CampaignRuntimeError("A campaign directory must be explicit")
    path = _safe_path(Path(directory))
    # Resolve only after existing ancestry is known not to contain links. This
    # binds one local destination, including Windows case/short-name aliases.
    return os.path.normcase(str(path.resolve()))


def freeze_runtime(*, allowances=None, directory=None, seed=14173):
    """Build reviewable local evidence. Missing allowances never authorize spend."""
    from smythe.pricing import PRICE_VERSION

    preparation = prepare_campaign(seed=seed)
    allocation = _allowances(allowances)
    if allocation is not None and directory is None:
        raise CampaignRuntimeError("A funded freeze requires its exact campaign directory before approval")
    destination = None if directory is None else _destination(directory)
    versions = _dependencies()
    blockers = [MAIN_BLOCKER, JUDGE_BLOCKER]
    if allocation is None:
        blockers.insert(0, "No total API spending ceiling and stage allocations have been authorized.")
    if versions["openai"] is None:
        blockers.insert(0, "Install the OpenAI SDK and freeze its version before pilot execution.")
    value = {
        "version": 1, "kind": "astra-pilot-runtime", "status": "offline-runtime-freeze",
        "claimable": False, "api_calls": 0, "seed": seed, "campaign_directory": destination,
        "preparation_sha256": preparation["preparation_sha256"],
        "protocol_sha256": preparation["protocol_sha256"],
        "schedule_sha256": preparation["schedule_sha256"],
        "source_sha256": _source_hashes(), "dependencies": versions,
        "price_version": PRICE_VERSION, "policy": strict_json(canonical(POLICY)),
        "prompts": {"planning_constraint": PLANNING_CONSTRAINT,
                    "fixed_steps": [list(step) for step in FIXED_STEPS]},
        "allowances": allocation, "main_enabled": False, "judge_enabled": False,
        "quality_evaluated": False, "blockers": blockers,
    }
    value["freeze_sha256"] = _sha(value)
    value["approval_token"] = "approve_astra_pilot_v1_" + value["freeze_sha256"]
    return value


def _verify_freeze(value, approval, *, directory=None):
    if type(value) is not dict:
        raise CampaignRuntimeError("Expected a runtime freeze object")
    if directory is not None and _destination(directory) != value.get("campaign_directory"):
        raise CampaignRuntimeError("Approval is bound to a different campaign directory")
    expected = freeze_runtime(allowances=value.get("allowances"), directory=value.get("campaign_directory"),
                              seed=value.get("seed"))
    if canonical(value) != canonical(expected):
        raise CampaignRuntimeError("Runtime freeze differs from current source, preparation, policy or dependencies")
    if value["allowances"] is None:
        raise CampaignRuntimeError("An explicit spending allowance is required")
    if value["dependencies"]["openai"] is None:
        raise CampaignRuntimeError("The frozen runtime has no OpenAI SDK")
    if type(approval) is not str or approval != value["approval_token"]:
        raise CampaignRuntimeError("Explicit approval of this exact pilot freeze is required")


def _read(path):
    path = _safe_path(path, file=True)
    with path.open("rb") as stream:
        content = stream.read(MAX_RECORD_BYTES + 1)
    if len(content) > MAX_RECORD_BYTES:
        raise CampaignRuntimeError("Campaign record exceeds its read bound")
    return strict_json(content.decode("utf-8"))


def _write_new(path, value):
    path = _safe_path(path, file=True)
    data = (canonical(value) + "\n").encode()
    if len(data) > MAX_RECORD_BYTES:
        raise CampaignRuntimeError("Campaign record exceeds its write bound")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _campaign_lock(directory):
    """A process-owned writer lock; it is independent of workflow transactions."""
    directory = _safe_path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _sqlite_path(directory / "campaign-lock.sqlite3")
    connection = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            raise CampaignRuntimeError("Another campaign worker holds the writer lock") from error
        yield
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


@contextmanager
def _failure_evidence(directory, freeze):
    try:
        yield
    except BaseException as error:
        # The ledger and any already published outcomes remain authoritative.
        # An unwritable diagnostic must never hide the original failure.
        try:
            _write_new(directory / f"failure-{uuid4().hex}.json", {
                "version": 1, "stage": "pilot", "claimable": False,
                "freeze_sha256": freeze["freeze_sha256"], "recorded_at_ns": time.time_ns(),
                "workflow_ledger": "workflow.sqlite3",
                "error": {"type": type(error).__name__, "message": str(error)[:2048]},
            })
        except (OSError, ValueError):
            pass
        raise


class _FixedArchitect:
    """Pure local topology construction; there are no hidden provider calls."""

    def plan(self, task):
        from smythe.agent import Agent, AgentProfile
        from smythe.graph import ExecutionGraph, Node, Topology
        from smythe.registry import Registry

        registry = Registry()
        nodes = []
        for index, (identity, label, persona) in enumerate(FIXED_STEPS):
            registry.register(Agent(id=identity, profile=AgentProfile(name=identity, persona=persona)))
            nodes.append(Node(id=identity, agent_id=identity, label=label,
                              depends_on=[FIXED_STEPS[index - 1][0]] if index else [],
                              max_retries=0, max_regenerations=0))
        return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes, task=task), registry


def _swarm(store, row, cap):
    from smythe import Swarm, WorkflowGraphPolicy
    from smythe.planner import LLMArchitect
    from smythe.provider_responses import OpenAIResponsesProvider
    from smythe.synthesizer import Synthesizer, SynthesisStrategy
    from smythe.workflow_binding import LocalOnly

    provider = OpenAIResponsesProvider(max_output_tokens=8192, reasoning_effort="medium", request_timeout_s=600)
    architect = (LocalOnly(_FixedArchitect, identity="astra-fixed-research-analysis-writing", version="1")
                 if row["strategy"] == "fixed_pipeline"
                 else LLMArchitect(provider, planning_model=row["model"], max_retries=0))
    return Swarm(model=row["model"], provider=provider, architect=architect,
                 synthesizer=Synthesizer(SynthesisStrategy.DELIVERABLE), run_store=store,
                 max_budget_usd=_budget_usd(cap), parallel=True, max_concurrency=8, max_revisions=0,
                 graph_policy=WorkflowGraphPolicy(8, node_model=row["model"], max_retries=0, max_regenerations=0))


def _run_id(freeze, row):
    return _sha([freeze["freeze_sha256"], row["trial_id"]])


def _task(case):
    from smythe import Task

    data = provider_task(case)
    data["constraints"].append(PLANNING_CONSTRAINT)
    return Task(**data)


def _binding(freeze, rows):
    return {"version": 1, "freeze": freeze,
            "trials": [{"trial": row, "run_id": _run_id(freeze, row)} for row in rows]}


def _clear_accounting(store, freeze, rows, *, recovering=None):
    expected_ids = {_run_id(freeze, row) for row in rows}
    confirmed = 0
    for run in store.list_runs():
        if run["run_id"] not in expected_ids:
            raise CampaignRuntimeError("Workflow ledger contains an unbound trial")
        if run["budget_nanousd"] != freeze["allowances"]["per_trial_nanousd"]:
            raise CampaignRuntimeError("Saved workflow allowance differs from the campaign")
        audit = store.inspect_run(run["run_id"])
        confirmed += audit["confirmed_nanousd"]
        if (audit["confirmed_nanousd"] > audit["budget_nanousd"]
                or run["run_id"] != recovering and (
                    audit["unknown_calls"] or audit["unknown_nanousd"] or audit["reserved_nanousd"])):
            raise CampaignRuntimeError("Unresolved or excessive workflow spending blocks new pilot trials")
    if confirmed > min(freeze["allowances"]["pilot_nanousd"], freeze["allowances"]["total_nanousd"]):
        raise CampaignRuntimeError("Confirmed pilot spending exceeds the campaign allocation")
    return confirmed


def _call_evidence(store, accounting):
    observed = {call["call_id"]: [] for call in accounting["calls"]}
    for event in accounting["events"]:
        if event["type"] != "evidence_saved":
            continue
        if event["call_id"] not in observed:
            raise CampaignRuntimeError("Saved evidence event refers to an unbound call")
        envelope = store.load_evidence(event["call_id"], event["data"]["evidence_id"])
        if (envelope["response_sha256"] != event["data"]["response_sha256"]
                or envelope["operation"] != event["data"]["operation"]):
            raise CampaignRuntimeError("Saved evidence event differs from its raw envelope")
        observed[event["call_id"]].append({key: item for key, item in envelope.items() if key != "body"})
    fingerprints = {}
    for call in accounting["calls"]:
        replay = store.load_replay(call["call_id"])
        for key in ("quote_evidence", "evidence"):
            if replay[key] is not None:
                # load_replay verifies the exact raw body hash and metadata.
                # Bind those bytes without duplicating provider raw evidence.
                replay[key] = {name: value for name, value in replay[key].items() if name != "body"}
        # Also bind invalid quotes and conflicting/late observations that were
        # retained but never selected as the call's accepted evidence.
        replay["observed_evidence"] = observed[call["call_id"]]
        fingerprints[call["call_id"]] = _sha(replay)
    return fingerprints


def _accounting_binding(accounting):
    return {**accounting, "events": [event for event in accounting["events"]
                                    if event["type"] not in {"lease_acquired", "lease_released"}]}


def _outcome(store, freeze, row, run_id, result, error, started, resumed):
    accounting = store.inspect_run(run_id)
    saved = store.get_checkpoint(run_id)
    output = result.output if result is not None else None
    if output is None and saved is not None:
        output = saved["checkpoint"].get("output")
    elapsed = time.perf_counter_ns() - started
    record = {
        "version": 1, "freeze_sha256": freeze["freeze_sha256"], "trial": row, "run_id": run_id,
        "status": "completed" if error is None else "failed", "claimable": False,
        "output": output, "output_sha256": None if output is None else hashlib.sha256(output.encode()).hexdigest(),
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)[:2048]},
        "wall_time_ns": elapsed if not resumed else None, "latency_complete": not resumed,
        "wall_time_scope": "Trial setup through workflow completion and ledger inspection; excludes evidence publication.",
        "cost_scope": "complete_text_workflow",
        "segment_wall_time_ns": elapsed, "resumed": resumed,
        "accounting": accounting, "call_evidence_sha256": _call_evidence(store, accounting), "checkpoint": saved,
        "trace": result.trace if result is not None else None,
        "deterministic_checks": {"deterministic_passed": False, "failed_checks": ["no-output"],
                                 "quality_evaluated": False, "accepted": None},
    }
    record["record_sha256"] = _sha(record)
    return record


def _validate_outcome(value, freeze, row, store, case=None):
    if type(value) is not dict:
        raise CampaignRuntimeError("Malformed saved pilot outcome")
    plain = dict(value)
    recorded_hash = plain.pop("record_sha256", None)
    if (recorded_hash != _sha(plain) or value.get("freeze_sha256") != freeze["freeze_sha256"]
            or value.get("trial") != row or value.get("run_id") != _run_id(freeze, row)
            or value.get("status") not in {"completed", "failed"}):
        raise CampaignRuntimeError("Saved pilot outcome has a different identity or content")
    run = store.load_run(value["run_id"])
    accounting = store.inspect_run(value["run_id"])
    saved = store.get_checkpoint(value["run_id"])
    # Lease observations may add events without changing accepted work. Every
    # accounting, request, raw receipt and checkpoint binding must still match.
    if (run["config_sha256"] != value["accounting"]["config_sha256"]
            or _accounting_binding(accounting) != _accounting_binding(value["accounting"])
            or saved != value.get("checkpoint")
            or _call_evidence(store, accounting) != value.get("call_evidence_sha256")):
        raise CampaignRuntimeError("Saved pilot outcome differs from current durable workflow evidence")
    output = None if saved is None else saved["checkpoint"].get("output")
    if (value.get("output") != output
            or value.get("output_sha256") != (None if output is None else hashlib.sha256(output.encode()).hexdigest())
            or value["status"] == "completed" and (saved is None or saved["checkpoint"]["status"] != "completed"
                                                     or run["status"] != "completed" or value.get("error") is not None)):
        raise CampaignRuntimeError("Saved pilot output differs from its completed checkpoint")
    checks = ({"deterministic_passed": False, "failed_checks": ["no-output"], "quality_evaluated": False, "accepted": None}
              if output is None else check_output(case, output))
    if value.get("deterministic_checks") != checks:
        raise CampaignRuntimeError("Saved deterministic checks differ from their output")


async def run_pilot(freeze, *, directory, approval=None):
    """Execute the 12 pilot trials in order, under an explicitly approved cap.

    Complete or failed outcomes are immutable and never rerun. A hard-crash
    interruption without an outcome resumes the same workflow identity and is
    marked incomplete for latency. Unknown billing blocks new trial admission.
    """
    if type(freeze) is not dict:
        raise CampaignRuntimeError("Expected a runtime freeze object")
    freeze = strict_json(canonical(freeze))
    _verify_freeze(freeze, approval, directory=directory)
    from smythe.task import task_to_dict
    from smythe.workflow_store import SQLiteWorkflowStore

    directory = Path(freeze["campaign_directory"])
    preparation = prepare_campaign(seed=freeze["seed"])
    rows = preparation["schedules"]["pilot"]
    cases = {case.task_id: case for case in load_task_pack().for_stage("pilot")}
    binding = _binding(freeze, rows)
    with _campaign_lock(directory):
        path = directory / "campaign.json"
        if _safe_path(path, file=True).exists():
            if _read(path) != binding:
                raise CampaignRuntimeError("Campaign directory belongs to a different runtime freeze")
        else:
            if set(item.name for item in directory.iterdir()) - {"campaign-lock.sqlite3", "campaign-lock.sqlite3-journal"}:
                raise CampaignRuntimeError("A new campaign requires an empty directory")
            _write_new(path, binding)
        with _failure_evidence(directory, freeze), SQLiteWorkflowStore(_sqlite_path(directory / "workflow.sqlite3")) as store:
            outcomes = []
            for row in rows:
                _verify_freeze(freeze, approval)
                run_id = _run_id(freeze, row)
                outcome_path = directory / f"{run_id}.outcome.json"
                if _safe_path(outcome_path, file=True).exists():
                    record = _read(outcome_path)
                    _validate_outcome(record, freeze, row, store, cases[row["task_id"]])
                    outcomes.append(record)
                    continue
                # First allow recovery of this exact unfinished trial; its
                # provider ledger refuses unknown redispatch itself. Admission
                # to a different trial requires all earlier balances to clear.
                existing = {entry["run_id"] for entry in store.list_runs()}
                _clear_accounting(store, freeze, rows, recovering=run_id if run_id in existing else None)
                start_path = directory / f"{run_id}.started.json"
                had_start = _safe_path(start_path, file=True).exists()
                resumed = had_start or run_id in existing
                if had_start:
                    if _read(start_path) != {"freeze_sha256": freeze["freeze_sha256"], "trial": row, "run_id": run_id}:
                        raise CampaignRuntimeError("Saved pilot start record belongs to a different trial")
                started = time.perf_counter_ns()
                swarm = _swarm(store, row, freeze["allowances"]["per_trial_nanousd"])
                recipe = swarm._workflow_runtime().recipe
                store.create_run(task_to_dict(_task(cases[row["task_id"]])), recipe,
                                 freeze["allowances"]["per_trial_nanousd"], run_id=run_id)
                if not had_start:
                    _write_new(start_path, {"freeze_sha256": freeze["freeze_sha256"], "trial": row, "run_id": run_id})
                result, error = None, None
                try:
                    result = await swarm.aresume(run_id)
                    _verify_freeze(freeze, approval)
                except BaseException as caught:
                    error = caught
                record = _outcome(store, freeze, row, run_id, result, error, started, resumed)
                if record["output"] is not None:
                    record["deterministic_checks"] = check_output(cases[row["task_id"]], record["output"])
                    record.pop("record_sha256")
                    record["record_sha256"] = _sha(record)
                _write_new(outcome_path, record)
                outcomes.append(record)
                if error is not None and not isinstance(error, Exception):
                    raise error
                _clear_accounting(store, freeze, rows)
            confirmed = _clear_accounting(store, freeze, rows)
            summary = {"version": 1, "stage": "pilot", "freeze_sha256": freeze["freeze_sha256"],
                       "claimable": False, "quality_evaluated": False, "accepted": None,
                       "workflow_runs": len(outcomes), "failed_workflows": sum(row["status"] == "failed" for row in outcomes),
                       "confirmed_nanousd": confirmed, "judge_nanousd": 0,
                       "main_enabled": False, "judge_enabled": False,
                       "outcome_receipts": [{"trial_id": item["trial"]["trial_id"], "run_id": item["run_id"],
                                             "record_sha256": item["record_sha256"],
                                             "path": f"{item['run_id']}.outcome.json"} for item in outcomes],
                       "blockers": [MAIN_BLOCKER, JUDGE_BLOCKER]}
            summary_path = directory / "pilot-summary.json"
            if _safe_path(summary_path, file=True).exists():
                if _read(summary_path) != summary:
                    raise CampaignRuntimeError("Saved pilot summary differs from its durable outcomes")
            else:
                _write_new(summary_path, summary)
            return {**summary, "outcomes": outcomes}


def run_main(*args, **kwargs):
    raise CampaignRuntimeError(MAIN_BLOCKER)


def run_judge(*args, **kwargs):
    raise CampaignRuntimeError(JUDGE_BLOCKER)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--directory", type=Path, help="exact destination bound into a funded pilot approval")
    for name in sorted(ALLOWANCE_KEYS):
        freeze.add_argument("--" + name.replace("_", "-"), type=int)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--freeze", type=Path, required=True)
    pilot.add_argument("--directory", type=Path, required=True)
    pilot.add_argument("--approve", required=True)
    commands.add_parser("main")
    commands.add_parser("judge")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            allocations = {key: getattr(args, key) for key in ALLOWANCE_KEYS}
            value = freeze_runtime(allowances=None if all(item is None for item in allocations.values()) else allocations,
                                   directory=args.directory)
            _write_new(args.out, value)
            print(canonical(value))
        elif args.command == "pilot":
            import asyncio

            print(canonical(asyncio.run(run_pilot(_read(args.freeze), directory=args.directory, approval=args.approve))))
        elif args.command == "main":
            run_main()
        else:
            run_judge()
    except (CampaignPlanError, OSError, sqlite3.Error) as error:
        print(canonical({"error": str(error), "claimable": False}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
