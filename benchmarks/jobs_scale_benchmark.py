"""Offline Jobs execution, hard-kill recovery, and explicit-reroll campaign.

Run from a checkout with ``python benchmarks/jobs_scale_benchmark.py --workdir
<new-directory> --output <new-result.json>``. The default executes 5,000 actual
operations. Every provider result is the same 1x1 PNG; this measures durable
execution correctness and local overhead, not image generation or API latency.
Work directories and evidence are retained and are never reused or deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.artifact_io import atomic_write_bytes, inspect_artifact
from smythe.jobs.models import ProviderKind
from smythe.jobs.providers import ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import SQLiteRunStore
from smythe.provider import Artifact, CompletionResult, Provider

REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ID = "jobs-scale"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PHASES = ("start", "resume", "reroll", "finished_resume")


def _require(condition: bool, message: str) -> None:
    # Assertions must still run when the campaign is invoked with python -O.
    if not condition:
        raise RuntimeError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, _json_bytes(value))


def _write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _manifest(count: int, concurrency: int) -> JobManifestV1:
    return JobManifestV1.from_dict({
        "version": 1,
        "name": "offline-jobs-scale-recovery",
        "profiles": [{"name": "fixture", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0", "options": {"artifacts_per_call": 1}}],
        "operations": [{"key": "pixel", "count": count, "profile": "fixture",
                        "prompt": "Return the deterministic offline one-pixel fixture.",
                        "artifact": {"mime_type": "image/png", "width": 1, "height": 1}}],
        "execution": {"max_concurrency": concurrency, "max_attempts": 2,
                      "max_budget_usd": "0", "output_directory": "outputs"},
    })


def _append_event(path: Path, event: dict) -> None:
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with path.open("ab", buffering=0) as stream:
        _require(stream.write(payload) == len(payload), "short provider evidence write")
        os.fsync(stream.fileno())


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _provider_summary(events: list[dict]) -> dict:
    active: set[str] = set()
    entered: dict[str, dict] = {}
    peak = 0
    for event in events:
        call_id = event["call_id"]
        if event["event"] == "entered":
            _require(call_id not in entered, "duplicate provider entry for one call")
            entered[call_id] = event
            active.add(call_id)
            peak = max(peak, len(active))
        else:
            _require(event["event"] == "returned" and call_id in active,
                     "provider return without an active entry")
            _require(event["attempt_id"] == entered[call_id]["attempt_id"]
                     and event["operation_id"] == entered[call_id]["operation_id"],
                     "provider event identity mismatch")
            active.remove(call_id)
    return {"entries": len(entered), "returns": len(entered) - len(active),
            "peak_active_provider_calls": peak, "interrupted_call_ids": sorted(active)}


class _FixtureProvider(Provider):
    def __init__(self, pool: "ProfileProviderPool", operation_id: str) -> None:
        self.pool = pool
        self.operation_id = operation_id

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.pool.provide(self.operation_id)


class ProfileProviderPool(ProviderPool):
    """Offline-only observer with durable entries before a deterministic barrier.

    Per-operation wrappers identify real journaled call/attempt IDs. No network
    adapter is constructed. The read connection never mutates runner state.
    """

    def __init__(self, root: Path, phase: str, config: dict) -> None:
        super().__init__()
        self.root, self.phase, self.config = root, phase, config
        self.connection = sqlite3.connect((root / "jobs.db").as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.entries = 0
        self.event_lock = asyncio.Lock()
        self.barrier_task: asyncio.Task | None = None

    def get(self, operation):
        self.preflight(operation)
        _require(operation.provider is ProviderKind.OFFLINE, "benchmark only permits offline")
        return _FixtureProvider(self, operation.operation_id)

    async def provide(self, operation_id: str) -> CompletionResult:
        row = self.connection.execute(
            """SELECT c.call_id, c.attempt_id, c.operation_id FROM calls c
               JOIN attempts a ON a.attempt_id = c.attempt_id
               WHERE a.run_id = ? AND a.operation_id = ? AND c.status = 'dispatched'
               ORDER BY a.attempt_number DESC LIMIT 1""", (RUN_ID, operation_id),
        ).fetchone()
        _require(row is not None, "provider entered without durable dispatch")
        identity = dict(row)
        self.entries += 1
        entry_number = self.entries
        await self._event("entered", identity)
        if self.phase == "start" and entry_number > self.config["completed_before_kill"]:
            if entry_number == self.config["kill_after_entries"]:
                self.barrier_task = asyncio.create_task(self._signal_barrier())
            await asyncio.Event().wait()
        # Yield once so provider overlap is observable; this is not an API delay.
        await asyncio.sleep(0)
        await self._event("returned", identity)
        return CompletionResult(text="offline fixture", artifacts=[Artifact(PNG, "image/png")],
                                cost_usd=0.0)

    async def _event(self, kind: str, identity: dict) -> None:
        # Serialize durable log writes without starving the real lease heartbeat.
        async with self.event_lock:
            await asyncio.to_thread(
                _append_event, self.root / f"{self.phase}-provider.jsonl",
                identity | {"event": kind, "phase": self.phase,
                            "pid": os.getpid(), "time_ns": time.time_ns()},
            )

    async def _signal_barrier(self) -> None:
        while True:
            completed = self.connection.execute(
                "SELECT COUNT(*) FROM operations WHERE run_id = ? AND status = 'succeeded'",
                (RUN_ID,),
            ).fetchone()[0]
            if completed == self.config["completed_before_kill"]:
                _write_json(self.root / "barrier.json", {
                    "pid": os.getpid(), "provider_entries": self.entries,
                    "completed_operations": completed,
                    "ready_time_ns": time.time_ns(),
                })
                return
            await asyncio.sleep(0.01)


async def _worker(root: Path, phase: str) -> None:
    config = _read_json(root / "config.json")
    started = time.perf_counter()
    with SQLiteRunStore(root / "jobs.db") as store:
        pool = ProfileProviderPool(root, phase, config)
        runner = JobRunner(store, provider_pool=pool, lease_ttl_s=config["lease_ttl_s"],
                           lease_heartbeat_s=config["lease_heartbeat_s"])
        try:
            # Observe the actual writer connection settings, not a different connection.
            durability = {"journal_mode": store._connection.execute(
                "PRAGMA journal_mode").fetchone()[0], "synchronous": store._connection.execute(
                "PRAGMA synchronous").fetchone()[0]}
            _write_json(root / f"{phase}-durability.json", durability)
            if phase == "start":
                plan = preflight_job(_manifest(config["count"], config["concurrency"]),
                                     manifest_root=root)
                result = await runner.start(plan, make_approval(plan), manifest_root=root,
                                            run_id=RUN_ID)
            elif phase == "reroll":
                result = await runner.reroll(RUN_ID, _read_json(root / "reroll-ids.json"),
                                            reason="offline campaign explicit crash recovery",
                                            acknowledge_unknown=True)
            else:
                result = await runner.resume(RUN_ID)
            _write_json(root / f"{phase}-result.json", {
                "status": result["status"], "counts": result["counts"],
                "execution_metrics": result["execution_metrics"],
                "worker_wall_s": time.perf_counter() - started,
            })
        finally:
            pool.connection.close()


def _launch(root: Path, phase: str) -> tuple[subprocess.Popen, Any]:
    log = (root / f"{phase}-worker.log").open("wb")
    try:
        # Windows venv python.exe can be a redirector: killing that wrapper
        # does not prove the actual worker died. Launch the real interpreter,
        # retaining this environment's installed packages via its import paths.
        executable = getattr(sys, "_base_executable", sys.executable) if os.name == "nt" else sys.executable
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(str(Path(path).resolve())
                                                  for path in sys.path if path)
        process = subprocess.Popen(
            [executable, str(Path(__file__).resolve()), "--worker", phase,
             "--workdir", str(root)], cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            env=environment,
        )
        return process, log
    except BaseException:
        log.close()
        raise


def _run_phase(root: Path, phase: str, timeout_s: float) -> dict:
    started = time.perf_counter()
    process, log = _launch(root, phase)
    try:
        return_code = process.wait(timeout=timeout_s)
        _require(return_code == 0,
                 f"{phase} worker exited {return_code}; inspect {root / (phase + '-worker.log')}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        log.close()
    result = _read_json(root / f"{phase}-result.json")
    result["parent_wall_s"] = time.perf_counter() - started
    result["provider"] = _provider_summary(_events(root / f"{phase}-provider.jsonl"))
    return result


def _kill_phase(root: Path, timeout_s: float) -> dict:
    started = time.perf_counter()
    process, log = _launch(root, "start")
    try:
        while not (root / "barrier.json").exists():
            _require(process.poll() is None, "start worker exited before crash barrier; inspect log")
            _require(time.perf_counter() - started < timeout_s, "crash barrier timed out")
            time.sleep(0.02)
        barrier = _read_json(root / "barrier.json")
        _require(barrier["pid"] == process.pid, "barrier does not belong to owned worker")
        killed_at_ns = time.time_ns()
        process.kill()  # TerminateProcess on Windows, SIGKILL on POSIX: no runner cleanup.
        return_code = process.wait(timeout=10)
        _require(return_code != 0, "hard-killed worker unexpectedly exited successfully")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        log.close()
    return {"parent_wall_s": time.perf_counter() - started, "barrier": barrier,
            "kill_time_ns": killed_at_ns, "exit_code": return_code,
            "execution_metrics": None,
            "kill_mechanism": "TerminateProcess" if os.name == "nt" else "SIGKILL",
            "provider": _provider_summary(_events(root / "start-provider.jsonl"))}


def _snapshot(root: Path) -> dict:
    with SQLiteRunStore(root / "jobs.db", read_only=True) as store:
        return store.snapshot(RUN_ID)


def _accepted_receipts(root: Path, snapshot: dict) -> dict[str, dict]:
    pointers = {op["operation_id"]: op["accepted_attempt_id"] for op in snapshot["operations"]
                if op["status"] == "succeeded"}
    receipts = {}
    for artifact in snapshot["artifacts"]:
        _require(artifact["accepted"] == 1, "unexpected unaccepted fixture artifact")
        operation_id = artifact["operation_id"]
        _require(operation_id not in receipts, "multiple accepted artifacts for one operation")
        _require(pointers.get(operation_id) == artifact["attempt_id"], "accepted pointer mismatch")
        data = (root / "outputs" / RUN_ID / artifact["relative_path"]).read_bytes()
        inspection = inspect_artifact(data, "image/png")
        _require(data == PNG, "artifact bytes differ from the fixture")
        for field in ("sha256", "size_bytes", "mime_type", "width", "height"):
            _require(artifact[field] == getattr(inspection, field), f"artifact {field} mismatch")
        receipts[operation_id] = {key: artifact[key] for key in (
            "artifact_id", "attempt_id", "relative_path", "sha256", "size_bytes",
            "mime_type", "width", "height")}
    _require(set(receipts) == set(pointers), "accepted operation missing artifact receipt")
    return receipts


def _source_hashes() -> dict[str, str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "smythe/*.py"], cwd=REPOSITORY,
    ).decode("utf-8").split("\0")
    paths = sorted({"benchmarks/jobs_scale_benchmark.py", "tests/test_jobs_scale_benchmark.py",
                    *(path for path in tracked if path)})
    return {relative: hashlib.sha256((REPOSITORY / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for relative in paths}


def _validate_zero_cost(snapshot: dict) -> None:
    _require(all(snapshot["cost"][field] == 0 for field in (
        "approved_microusd", "confirmed_microusd", "exposure_microusd", "reserved_microusd")),
        "zero-cost fixture run ledger drifted")


def _validate_call_ledger(calls: list[dict], accepted: dict, interrupted_call_ids: set[str]) -> None:
    _require(Counter(call["status"] for call in calls)
             == Counter({"succeeded": len(accepted), "unknown_outcome": len(interrupted_call_ids)}),
             "final call status partition is incorrect")
    _require({call["attempt_id"] for call in calls if call["status"] == "succeeded"}
             == {receipt["attempt_id"] for receipt in accepted.values()},
             "succeeded calls do not match accepted attempts")
    _require({call["call_id"] for call in calls if call["status"] == "unknown_outcome"}
             == interrupted_call_ids, "interrupted call uncertainty was lost")
    for call in calls:
        _require(all(call[field] == 0 for field in (
            "ceiling_microusd", "confirmed_microusd", "exposure_microusd")),
            "zero-cost fixture call ledger drifted")
        expected_flags = (0, 1) if call["status"] == "unknown_outcome" else (1, 0)
        _require((call["cost_is_complete"], call["cost_is_estimate"]) == expected_flags,
                 "call cost certainty flags are incorrect")


def _retained_evidence(root: Path) -> dict:
    return {path.name: {"size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(root.glob("*")) if path.is_file()}


def run_campaign(workdir: Path, *, count: int = 5000, concurrency: int = 8,
                 lease_ttl_s: float = 30.0, timeout_s: float = 7200) -> dict:
    """Retain explicit failure evidence as well as verified successful observations."""
    existed = workdir.exists()
    started = time.perf_counter()
    try:
        return _run_campaign(workdir, count=count, concurrency=concurrency,
                             lease_ttl_s=lease_ttl_s, timeout_s=timeout_s)
    except (Exception, KeyboardInterrupt) as exc:
        if not existed and workdir.is_dir():
            provenance_path = workdir / "provenance.json"
            _write_json(workdir / "failure.json", {
                "schema": "smythe.jobs-scale-recovery.v1", "campaign_status": "failed",
                "evidence_status": "diagnostic_incomplete", "comparative_claimable": False,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "configuration": {"count": count, "concurrency": concurrency,
                                  "lease_ttl_s": lease_ttl_s, "lease_heartbeat_s": 1.0,
                                  "timeout_s": timeout_s},
                "provenance": _read_json(provenance_path) if provenance_path.exists() else None,
                "failure": {"exception_type": type(exc).__name__,
                            "message": str(exc) or "operator interrupted campaign"},
                "wall_s": {"total": time.perf_counter() - started},
                "retained_evidence": _retained_evidence(workdir),
            })
        raise


def _run_campaign(workdir: Path, *, count: int, concurrency: int,
                  lease_ttl_s: float, timeout_s: float) -> dict:
    """Execute one campaign in a new directory and return verified observations."""
    import PIL

    if not 1 <= concurrency <= 64 or not 2 * concurrency + 2 <= count <= 5000:
        raise ValueError("require 1 <= concurrency <= 64 and 2*concurrency+2 <= count <= 5000")
    if not 2 <= lease_ttl_s <= 300 or not 10 <= timeout_s <= 86400:
        raise ValueError("require lease TTL 2..300 s and phase timeout 10..86400 s")
    root = workdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter()
    source_hashes = _source_hashes()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                                       text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY,
                                    text=True).splitlines()
    config = {"count": count, "concurrency": concurrency, "max_attempts": 2,
              "lease_ttl_s": lease_ttl_s, "lease_heartbeat_s": 1.0,
              "kill_after_entries": count // 2,
              "completed_before_kill": count // 2 - concurrency}
    _write_json(root / "config.json", config)
    provenance = {"git_revision": revision, "git_status_at_start": dirty,
                  "source_sha256": source_hashes,
                  "source_hash_policy": "SHA-256 of source bytes with CRLF normalized to LF; "
                  "Git-tracked smythe Python files plus the explicit harness and its tests. "
                  "Retained evidence files use raw byte hashes."}
    _write_json(root / "provenance.json", provenance)
    phases = {"start": _kill_phase(root, timeout_s)}
    validation_started = time.perf_counter()
    killed = _snapshot(root)
    before = _accepted_receipts(root, killed)
    interrupted = {op["operation_id"] for op in killed["operations"] if op["status"] == "running"}
    pending = {op["operation_id"] for op in killed["operations"] if op["status"] == "pending"}
    _require(len(before) == config["completed_before_kill"], "wrong completed count at kill")
    _require(len(interrupted) == concurrency and len(pending) == count - count // 2,
             "crash did not leave the specified running/pending partition")
    _require(phases["start"]["provider"]["entries"] == count // 2,
             "wrong provider-entry count at crash")
    _require(len(phases["start"]["provider"]["interrupted_call_ids"]) == concurrency,
             "provider interrupted-call count differs from running operations")
    crash_validation_s = time.perf_counter() - validation_started
    with SQLiteRunStore(root / "jobs.db", read_only=True) as store:
        lease = store.get_run_lease(RUN_ID)
    _require(lease is not None, "hard kill did not retain the runner lease")
    wait_started = time.perf_counter()
    while time.time_ns() <= lease.expires_at_ns:
        time.sleep(min(0.05, max(0.001, (lease.expires_at_ns - time.time_ns()) / 1e9)))
    lease_wait = {"wall_s": time.perf_counter() - wait_started,
                  "expires_at_ns": lease.expires_at_ns, "resume_allowed_at_ns": time.time_ns(),
                  "expiry_method": "elapsed real wall-clock time; no database edits"}

    phases["resume"] = _run_phase(root, "resume", timeout_s)
    validation_started = time.perf_counter()
    resumed = _snapshot(root)
    after_resume = _accepted_receipts(root, resumed)
    unknown = {op["operation_id"] for op in resumed["operations"]
               if op["status"] == "unknown_outcome"}
    _require(unknown == interrupted, "resume changed the interrupted operation set")
    _require(len(after_resume) == count - concurrency, "resume did not finish all pending work")
    _require(all(after_resume.get(key) == value for key, value in before.items()),
             "resume changed a previously accepted pointer or artifact")
    resume_entries = [e for e in _events(root / "resume-provider.jsonl") if e["event"] == "entered"]
    _require(Counter(e["operation_id"] for e in resume_entries) == Counter(pending),
             "resume redispatched an unknown/completed operation or missed pending work")
    _write_json(root / "reroll-ids.json", sorted(unknown))
    resume_validation_s = time.perf_counter() - validation_started

    phases["reroll"] = _run_phase(root, "reroll", timeout_s)
    phases["finished_resume"] = _run_phase(root, "finished_resume", timeout_s)
    validation_started = time.perf_counter()
    final = _snapshot(root)
    accepted = _accepted_receipts(root, final)
    _require(final["status"] == "completed" and final["counts"] == {"succeeded": count},
             "final job is not completely accepted")
    _require(len(accepted) == count, "incorrect final artifact receipt count")
    _require(all(accepted.get(key) == value for key, value in after_resume.items()),
             "reroll changed an already accepted pointer or artifact")
    _require(phases["finished_resume"]["provider"]["entries"] == 0,
             "finished resume entered a provider")
    all_events = [event for phase in PHASES for event in _events(root / f"{phase}-provider.jsonl")]
    entries = [event for event in all_events if event["event"] == "entered"]
    expected = Counter({operation_id: 1 + (operation_id in interrupted) for operation_id in accepted})
    _require(Counter(e["operation_id"] for e in entries) == expected,
             "unexpected repeated or missing provider operation entry")
    _require(len({e["call_id"] for e in entries}) == len(entries)
             and len({e["attempt_id"] for e in entries}) == len(entries),
             "call or attempt dispatched more than once")
    attempts = {item["attempt_id"]: item for item in final["attempts"]}
    lineage = []
    for op in final["operations"]:
        last = attempts[op["accepted_attempt_id"]]
        if op["operation_id"] in interrupted:
            parent = attempts[last["parent_attempt_id"]]
            _require(last["attempt_number"] == 2 and parent["status"] == "unknown_outcome"
                     and parent["operation_id"] == op["operation_id"], "reroll lineage lost")
            lineage.append({"operation_id": op["operation_id"],
                            "parent_attempt_id": parent["attempt_id"],
                            "accepted_attempt_id": last["attempt_id"]})
        else:
            _require(last["attempt_number"] == 1 and last["parent_attempt_id"] is None,
                     "previously succeeded operation was rerolled")
    with closing(sqlite3.connect((root / "jobs.db").as_uri() + "?mode=ro", uri=True)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        columns = ("call_id", "attempt_id", "operation_id", "status", "ceiling_microusd",
                   "confirmed_microusd", "exposure_microusd", "cost_is_complete", "cost_is_estimate")
        calls = [dict(zip(columns, row)) for row in connection.execute(
            "SELECT " + ", ".join(columns) + " FROM calls")]
        _require(integrity == "ok" and not connection.execute("PRAGMA foreign_key_check").fetchall(),
                 "SQLite integrity/foreign-key validation failed")
    _require({(c["call_id"], c["attempt_id"], c["operation_id"]) for c in calls}
             == {(e["call_id"], e["attempt_id"], e["operation_id"]) for e in entries},
             "durable call journal does not match actual provider entries")
    _validate_call_ledger(calls, accepted, set(phases["start"]["provider"]["interrupted_call_ids"]))
    for snapshot in (killed, resumed, final):
        _validate_zero_cost(snapshot)
    durability = {phase: _read_json(root / f"{phase}-durability.json") for phase in PHASES}
    _require(all(item == {"journal_mode": "wal", "synchronous": 2}
                 for item in durability.values()), "campaign did not use WAL/FULL")
    for phase in PHASES:
        _require(phases[phase]["provider"]["peak_active_provider_calls"] <= concurrency,
                 "concurrency bound exceeded")
        if phase != "start":
            _require(phases[phase]["execution_metrics"]["peak_active_calls"] <= concurrency,
                     "runner active-operation concurrency bound exceeded")
    _require(_source_hashes() == source_hashes, "campaign source changed during execution")
    final_validation_s = time.perf_counter() - validation_started
    return {
        "schema": "smythe.jobs-scale-recovery.v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "campaign_status": "completed",
        "evidence_status": "offline_correctness_observation", "comparative_claimable": False,
        "scope": "One local campaign; deterministic 1x1 PNGs; no API calls or speedup claim.",
        "configuration": config,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version, "sqlite": sqlite3.sqlite_version,
                        "pillow": PIL.__version__, "logical_cpu_count": os.cpu_count()},
        "provenance": provenance | {"sources_unchanged_during_campaign": True},
        "phases": phases, "lease_expiry": lease_wait, "durability": durability,
        "ledger": {"after_kill": killed["cost"], "after_resume": resumed["cost"],
                   "final": final["cost"]},
        "state_counts": {"after_kill": killed["counts"], "after_resume": resumed["counts"],
                         "final": final["counts"]},
        "wall_s": {"crash_validation": crash_validation_s,
                   "resume_validation": resume_validation_s, "final_validation": final_validation_s,
                   "total": time.perf_counter() - total_started},
        "verification": {"passed": True, "operations": count, "accepted_artifacts": len(accepted),
                         "accepted_before_kill": len(before), "pending_completed_on_resume": len(pending),
                         "unknown_preserved_on_resume": len(unknown), "explicit_rerolls": len(unknown),
                         "provider_entries": len(entries), "remote_api_calls": 0,
                         "previously_succeeded_redispatches": 0, "unknown_auto_redispatches": 0,
                         "completed_resume_provider_entries": 0, "sqlite_integrity": integrity,
                         "zero_cost_ledger_verified": True,
                         "unknown_call_cost_flags_preserved": True,
                         "durable_call_status_counts": dict(Counter(c["status"] for c in calls)),
                         "artifact_bytes": sum(a["size_bytes"] for a in accepted.values()),
                         "fixture_bytes": len(PNG), "fixture_sha256": hashlib.sha256(PNG).hexdigest(),
                         "artifact_mime_type": "image/png", "artifact_dimensions": [1, 1],
                         "unique_artifact_content_hashes": len({a["sha256"] for a in accepted.values()}),
                         "accepted_receipts_sha256": _digest(accepted),
                         "preserved_precrash_receipts_sha256": _digest(before),
                         "reroll_lineage": sorted(lineage, key=lambda item: item["operation_id"])},
        "metric_scopes": {
            "runner_peak_active_calls": "Active operation scope, including dispatch and artifact I/O.",
            "peak_active_provider_calls": "Entered fixture coroutine until its durable returned event.",
            "start_peak": "Runner metrics unavailable: hard kill prevents final metrics publication.",
            "phase_parent_wall_s": "Subprocess launch through exit, including imports and JSON evidence.",
            "timing_overhead": "Includes per-entry/return JSONL fsync and benchmark observation.",
        },
        "retained_evidence": _retained_evidence(root),
        "sqlite_disk_bytes": {name: (root / name).stat().st_size if (root / name).exists() else 0
                              for name in ("jobs.db", "jobs.db-wal", "jobs.db-shm")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--lease-ttl-s", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=7200)
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        asyncio.run(_worker(args.workdir.resolve(), args.worker))
        return
    if args.output is None:
        parser.error("--output is required for a campaign")
    if args.output.exists():
        parser.error("--output already exists; retained evidence cannot be overwritten")
    if args.output.resolve().is_relative_to(args.workdir.resolve()):
        parser.error("--output must be outside --workdir to prevent evidence-path collisions")
    if args.workdir.exists():
        parser.error("--workdir must be a new directory; stale evidence cannot be reused")
    try:
        result = run_campaign(args.workdir, count=args.count, concurrency=args.concurrency,
                              lease_ttl_s=args.lease_ttl_s, timeout_s=args.timeout_s)
    except (Exception, KeyboardInterrupt):
        failure_path = args.workdir / "failure.json"
        if failure_path.exists():
            _write_exclusive_json(args.output, _read_json(failure_path))
        raise
    _write_exclusive_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "verified": result["verification"],
                      "wall_s": result["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
