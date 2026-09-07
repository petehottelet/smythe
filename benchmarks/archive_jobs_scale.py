"""Archive a finished Jobs scale campaign and independently review copied evidence.

The producer must have exited and all owned workers must be closed. A terminal
external result is mandatory. Never invoke this against the running campaign.
SQLite is opened only after ZIP verification and extraction to a temporary copy.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import subprocess
import sys
import tempfile
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPECTED_REVISION = "4bb7c0295cec658c7118d7bf9305764dae9b1c57"
PHASES = ("start", "resume", "reroll", "finished_resume")
REVIEW_CHECKS = (
    "archive_verified",
    "ledger_reconciled",
    "accepted_pointers_preserved",
    "artifacts_verified",
    "attempt_lineage_verified",
    "real_kill_verified",
    "lease_expiry_verified",
    "zero_cost_verified",
)
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid integer: {name}")
    return value


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _member_name(name):
    _require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and ":" not in name
        and "\x00" not in name,
        "Unsafe archive path",
    )
    path = PurePosixPath(name.rstrip("/"))
    _require(
        not path.is_absolute()
        and all(part not in ("", ".", "..") for part in name.rstrip("/").split("/")),
        "Escaping archive path",
    )
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    _require(
        name in (path.as_posix(), path.as_posix() + "/")
        and all(
            part.rstrip(". ") == part
            and part.split(".")[0].upper() not in reserved
            and not any(ord(char) < 32 for char in part)
            for part in path.parts
        ),
        "Nonportable or ambiguous archive path",
    )
    return path


def _safe_path(path, *, missing=False):
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            details = item.lstat()
        except FileNotFoundError:
            _require(missing, f"Missing input: {item}")
            continue
        _require(
            not stat.S_ISLNK(details.st_mode)
            and not getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
            f"Links, junctions and reparse points are forbidden: {item}",
        )
    return path


def _identity(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_mode


def _read(path):
    path = _safe_path(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"Regular file required: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        _require(
            _identity(os.fstat(stream.fileno())) == _identity(before),
            "Input replaced while opening",
        )
        data = stream.read()
        _require(
            _identity(os.fstat(stream.fileno())) == _identity(before), "Input changed while reading"
        )
    _require(
        _identity(_safe_path(path).lstat()) == _identity(before), "Input replaced while reading"
    )
    return data


def _tree(root):
    root = _safe_path(root)
    _require(root.is_dir(), "Campaign must be a directory")
    entries = {}

    def visit(directory):
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            path = _safe_path(path)
            _require(path.resolve().is_relative_to(root.resolve()), "Campaign path escaped root")
            details = path.lstat()
            name = path.relative_to(root).as_posix()
            _member_name(name)
            _require(
                stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode),
                "Special files are forbidden",
            )
            entries[name] = (path, _identity(details), stat.S_ISDIR(details.st_mode))
            if stat.S_ISDIR(details.st_mode):
                visit(path)

    visit(root)
    _require(len({name.casefold() for name in entries}) == len(entries), "Case-colliding paths")
    return entries


def _frozen_sources(source_root, record, campaign):
    """Validate both the explicit checkout and its committed source blobs."""
    source_root = _safe_path(source_root)
    provenance = record.get("provenance")
    _require(
        isinstance(provenance, dict) and provenance.get("git_revision") == EXPECTED_REVISION,
        "Result does not bind the expected frozen source revision",
    )
    saved = _json(_read(campaign / "provenance.json"))
    _require(
        all(provenance.get(key) == value for key, value in saved.items()),
        "Provenance file/result mismatch",
    )
    head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(head == EXPECTED_REVISION, "Supply the frozen checkout, not current main")
    tracked = (
        subprocess.check_output(
            ["git", "-C", str(source_root), "ls-files", "-z", "--", "smythe/*.py"]
        )
        .decode()
        .split("\0")
    )
    names = sorted(
        {
            "benchmarks/jobs_scale_benchmark.py",
            "tests/test_jobs_scale_benchmark.py",
            *(name for name in tracked if name),
        }
    )
    declared = provenance.get("source_sha256")
    _require(
        isinstance(declared, dict) and set(declared) == set(names),
        "Incomplete frozen source inventory",
    )
    commands = "".join(f"{EXPECTED_REVISION}:{name}\n" for name in names).encode()
    blobs = io.BytesIO(
        subprocess.check_output(
            ["git", "-C", str(source_root), "cat-file", "--batch"], input=commands
        )
    )
    sources = {}
    for name in names:
        _member_name(name)
        header = blobs.readline().split()
        _require(len(header) == 3 and header[1] == b"blob", "Frozen source blob missing")
        committed = blobs.read(int(header[2]))
        _require(blobs.read(1) == b"\n", "Malformed Git blob stream")
        path = _safe_path(source_root / name)
        _require(path.resolve().is_relative_to(source_root.resolve()), "Source escaped checkout")
        data = _read(path)
        _require(
            _sha(data.replace(b"\r\n", b"\n"))
            == declared[name]
            == _sha(committed.replace(b"\r\n", b"\n")),
            f"Frozen source mismatch: {name}",
        )
        sources[f"source/{name}"] = path
    return sources


def verify_archive(path, members):
    """Verify every member, including exact bytes and safe unique paths."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _require(
            len(names) == len(set(names)) == len({name.casefold() for name in names}),
            "Duplicate ZIP members",
        )
        _require(set(names) == set(members), "ZIP member inventory mismatch")
        for info in infos:
            _member_name(info.filename)
            _require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlink forbidden")
            expected = members[info.filename]
            _require(info.is_dir() == (expected["kind"] == "directory"), "ZIP member kind mismatch")
            data = archive.read(info)
            _require(
                len(data) == expected["size_bytes"] and _sha(data) == expected["sha256"],
                "ZIP member bytes mismatch",
            )


def _extract(archive_path, destination, members):
    verify_archive(archive_path, members)
    with zipfile.ZipFile(archive_path) as archive:
        for name, details in members.items():
            path = destination.joinpath(*_member_name(name).parts)
            _require(
                path.resolve().is_relative_to(destination.resolve()),
                "Extraction escaped temporary root",
            )
            if details["kind"] == "directory":
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(archive.read(name))


def _phase_events(root, phase):
    path = root / f"{phase}-provider.jsonl"
    events = [_json(line) for line in _read(path).splitlines()] if path.exists() else []
    active, entries, peak, previous = {}, {}, 0, 0
    for event in events:
        _require(event["phase"] == phase, "Provider phase mismatch")
        now = _integer(event["time_ns"], "provider timestamp", 1)
        _require(now >= previous, "Provider log time moved backwards")
        previous = now
        _integer(event["pid"], "provider PID", 1)
        for key in ("call_id", "attempt_id", "operation_id"):
            _require(isinstance(event[key], str) and event[key], "Invalid provider identity")
        key = event["call_id"]
        if event["event"] == "entered":
            _require(key not in entries, "Repeated provider call entry")
            entries[key] = event
            active[key] = event
            peak = max(peak, len(active))
        else:
            _require(
                event["event"] == "returned" and key in active, "Provider return without entry"
            )
            _require(
                all(
                    event[field] == active[key][field]
                    for field in ("attempt_id", "operation_id", "pid")
                ),
                "Provider return identity mismatch",
            )
            del active[key]
    return (
        events,
        entries,
        {
            "entries": len(entries),
            "returns": len(entries) - len(active),
            "peak_active_provider_calls": peak,
            "interrupted_call_ids": sorted(active),
        },
    )


def _review_copy(root, record):
    """Reconstruct historical sets from final copied DB/events, not old snapshots."""
    from PIL import Image, __version__ as pillow_version

    config = record["configuration"]
    count = _integer(config["count"], "count", 1)
    concurrency = _integer(config["concurrency"], "concurrency", 1)
    _require(config == _json(_read(root / "config.json")), "Configuration file/result mismatch")
    _require(
        config["max_attempts"] == 2
        and config["completed_before_kill"] == count // 2 - concurrency
        and config["kill_after_entries"] == count // 2,
        "Wrong crash workload",
    )
    logs, entered, summaries = {}, {}, {}
    for phase in PHASES:
        logs[phase], entered[phase], summaries[phase] = _phase_events(root, phase)
        _require(
            summaries[phase] == record["phases"][phase]["provider"], "Provider summary mismatch"
        )
        _require(
            summaries[phase]["peak_active_provider_calls"] <= concurrency,
            "Provider concurrency exceeded",
        )
        _require(
            _json(_read(root / f"{phase}-durability.json"))
            == record["durability"][phase]
            == {"journal_mode": "wal", "synchronous": 2},
            "WAL/FULL settings mismatch",
        )
        if phase != "start":
            worker_result = _json(_read(root / f"{phase}-result.json"))
            _require(
                all(
                    record["phases"][phase].get(key) == value
                    for key, value in worker_result.items()
                ),
                "Worker result mismatch",
            )
            metrics = worker_result["execution_metrics"]
            _require(
                _integer(metrics["operations_started"], "operations started") == len(entered[phase])
                and _integer(metrics["peak_active_calls"], "active operation peak") <= concurrency,
                "Worker operation metrics mismatch",
            )
        _read(root / f"{phase}-worker.log")  # Retained even when empty.
    start = record["phases"]["start"]
    barrier = _json(_read(root / "barrier.json"))
    kill = _integer(start["kill_time_ns"], "kill timestamp", 1)
    _require(
        start["barrier"] == barrier and barrier["ready_time_ns"] <= kill, "Barrier/kill mismatch"
    )
    _require(
        type(start["exit_code"]) is int
        and start["exit_code"] != 0
        and start["kill_mechanism"] in ("TerminateProcess", "SIGKILL"),
        "Missing hard-kill evidence",
    )
    _require(
        {event["pid"] for event in logs["start"]} == {barrier["pid"]},
        "Killed PID does not match provider",
    )
    _require(
        barrier["provider_entries"] == count // 2
        and barrier["completed_operations"] == count // 2 - concurrency,
        "Incorrect kill barrier counts",
    )
    _require(
        all(event["time_ns"] <= barrier["ready_time_ns"] for event in logs["start"]),
        "Start event occurred after barrier",
    )
    expiry = record["lease_expiry"]
    _require(
        kill < expiry["expires_at_ns"] <= kill + int(config["lease_ttl_s"] * 1e9)
        and expiry["resume_allowed_at_ns"] > expiry["expires_at_ns"]
        and expiry["wall_s"] >= 0
        and expiry["expiry_method"] == "elapsed real wall-clock time; no database edits",
        "Lease expiry mismatch",
    )
    _require(
        all(event["time_ns"] >= expiry["resume_allowed_at_ns"] for event in logs["resume"]),
        "Resume dispatched before expiry",
    )
    _require(not logs["finished_resume"], "Completed resume dispatched work")

    database = root / "jobs.db"
    # mode=ro honors copied WAL files. immutable=1 would incorrectly ignore WAL.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        _require(
            connection.execute("PRAGMA integrity_check").fetchall()[0][0] == "ok",
            "SQLite integrity failure",
        )
        _require(
            not connection.execute("PRAGMA foreign_key_check").fetchall(),
            "SQLite foreign-key failure",
        )
        _require(
            connection.execute("PRAGMA user_version").fetchone()[0] == 3,
            "Wrong frozen ledger schema version",
        )
        rows = {
            table: [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for table in (
                "runs",
                "operations",
                "attempts",
                "calls",
                "artifacts",
                "events",
                "run_leases",
            )
        }
    _require(len(rows["runs"]) == 1 and not rows["run_leases"], "Unexpected run or active lease")
    run = rows["runs"][0]
    _require(
        run["run_id"] == "jobs-scale"
        and run["status"] == "completed"
        and run["output_directory"] == "outputs",
        "Final run incomplete or wrong output root",
    )
    for name in ("operations", "attempts", "calls", "artifacts", "events"):
        _require(all(row["run_id"] == run["run_id"] for row in rows[name]), "Foreign run in ledger")
    operations = {row["operation_id"]: row for row in rows["operations"]}
    attempts = {row["attempt_id"]: row for row in rows["attempts"]}
    calls = {row["call_id"]: row for row in rows["calls"]}
    all_entries = {key: value for phase in PHASES for key, value in entered[phase].items()}
    _require(
        len(all_entries) == sum(len(entered[phase]) for phase in PHASES) == count + concurrency,
        "Call entries duplicated or missing",
    )
    _require(
        len(operations) == len(rows["operations"]) == count
        and len(attempts) == count + concurrency,
        "Operation/attempt count mismatch",
    )
    _require(
        set(calls) == set(all_entries) and len(calls) == len(rows["calls"]),
        "Provider entries differ from durable calls",
    )
    _require(
        {call["attempt_id"] for call in calls.values()} == set(attempts)
        and len({call["attempt_id"] for call in calls.values()}) == len(calls),
        "Orphan or reused attempt identity",
    )
    _require(
        Counter(call["status"] for call in calls.values())
        == record["verification"]["durable_call_status_counts"],
        "Recorded call-state counts mismatch",
    )
    interrupted_calls = set(summaries["start"]["interrupted_call_ids"])
    interrupted = {calls[key]["operation_id"] for key in interrupted_calls}
    before_ids = {event["operation_id"] for event in logs["start"] if event["event"] == "returned"}
    pending_ids = set(operations) - before_ids - interrupted
    _require(
        len(interrupted) == concurrency
        and len(before_ids) == count // 2 - concurrency
        and len(pending_ids) == count - count // 2,
        "Historical partition mismatch",
    )
    _require(
        {event["operation_id"] for event in entered["resume"].values()} == pending_ids
        and len(entered["resume"]) == len(pending_ids),
        "Resume reissued accepted/unknown work",
    )
    _require(
        {event["operation_id"] for event in entered["reroll"].values()} == interrupted
        and len(entered["reroll"]) == concurrency,
        "Reroll selection mismatch",
    )
    _require(
        _json(_read(root / "reroll-ids.json")) == sorted(interrupted),
        "Explicit reroll list mismatch",
    )
    returned = {
        event["call_id"]: event
        for phase in PHASES
        for event in logs[phase]
        if event["event"] == "returned"
    }
    for key, call in calls.items():
        entry = all_entries[key]
        _require(
            all(call[field] == entry[field] for field in ("attempt_id", "operation_id")),
            "Call/entry identity mismatch",
        )
        _require(
            attempts[call["attempt_id"]]["lease_epoch"] == PHASES.index(entry["phase"]) + 1,
            "Attempt used a different phase lease",
        )
        _require(
            call["created_at_ns"] <= call["dispatched_at_ns"] <= entry["time_ns"],
            "Dispatch timestamp mismatch",
        )
        unknown = key in interrupted_calls
        _require(
            call["status"] == ("unknown_outcome" if unknown else "succeeded"), "Call state mismatch"
        )
        _require(
            (call["cost_is_complete"], call["cost_is_estimate"]) == ((0, 1) if unknown else (1, 0)),
            "Unknown cost flags lost",
        )
        if unknown:
            _require(
                key not in returned and call["completed_at_ns"] >= expiry["resume_allowed_at_ns"],
                "Unknown call timing mismatch",
            )
        else:
            _require(
                entry["time_ns"] <= returned[key]["time_ns"] <= call["completed_at_ns"],
                "Completion timestamp mismatch",
            )
        for field in ("ceiling_microusd", "confirmed_microusd", "exposure_microusd"):
            _require(_integer(call[field], field) == 0, "Nonzero call charges")
    for stage in (run, *record["ledger"].values()):
        for field in (
            "approved_microusd",
            "confirmed_microusd",
            "exposure_microusd",
            "reserved_microusd",
        ):
            _require(_integer(stage[field], field) == 0, "Nonzero run charges")
    with Image.open(io.BytesIO(PNG)) as image:
        image.load()
        _require(image.size == (1, 1) and image.format == "PNG", "Invalid reference fixture")
    receipts, files = {}, set()
    for artifact in rows["artifacts"]:
        operation = operations[artifact["operation_id"]]
        _require(
            artifact["accepted"] == 1
            and operation["status"] == "succeeded"
            and artifact["attempt_id"] == operation["accepted_attempt_id"],
            "Accepted pointer mismatch",
        )
        relative = _member_name(artifact["relative_path"])
        name = "outputs/jobs-scale/" + relative.as_posix()
        _require(
            name not in files and artifact["operation_id"] not in receipts,
            "Repeated artifact path/operation",
        )
        data = _read(root.joinpath(*PurePosixPath(name).parts))
        _require(
            data == PNG
            and artifact["sha256"] == _sha(data)
            and artifact["size_bytes"] == len(data)
            and artifact["mime_type"] == "image/png"
            and artifact["width"] == artifact["height"] == 1,
            "Artifact bytes or metadata mismatch",
        )
        files.add(name)
        receipts[artifact["operation_id"]] = {
            key: artifact[key]
            for key in (
                "artifact_id",
                "attempt_id",
                "relative_path",
                "sha256",
                "size_bytes",
                "mime_type",
                "width",
                "height",
            )
        }
    _require(set(receipts) == set(operations) and len(files) == count, "Missing accepted artifacts")
    _require(
        {
            f"outputs/{name}"
            for name, (_, _, directory) in _tree(root / "outputs").items()
            if not directory
        }
        == files,
        "Missing or orphan output files",
    )
    before = {key: receipts[key] for key in before_ids}
    _require(
        _sha(_json_bytes(before)) == record["verification"]["preserved_precrash_receipts_sha256"],
        "Prekill receipt digest changed",
    )
    _require(
        _sha(_json_bytes(receipts)) == record["verification"]["accepted_receipts_sha256"],
        "Final receipt digest changed",
    )
    actual_fixture = {
        "fixture_bytes": len(PNG),
        "fixture_sha256": _sha(PNG),
        "artifact_bytes": count * len(PNG),
        "artifact_dimensions": [1, 1],
        "artifact_mime_type": "image/png",
        "unique_artifact_content_hashes": 1,
    }
    _require(
        all(record["verification"].get(key) == value for key, value in actual_fixture.items()),
        "Recorded fixture fields mismatch",
    )
    reroll_first = min(event["time_ns"] for event in logs["reroll"])
    for artifact in rows["artifacts"]:
        if artifact["operation_id"] in before_ids:
            _require(artifact["created_at_ns"] <= kill, "Prekill artifact was replaced later")
        elif artifact["operation_id"] in pending_ids:
            _require(
                expiry["resume_allowed_at_ns"] <= artifact["created_at_ns"] < reroll_first,
                "Resume artifact was replaced during reroll",
            )
    lineage = []
    for key, operation in operations.items():
        _require(
            _json(operation["spec_json"])["provider"] == "offline",
            "Non-offline operation in fixture",
        )
        attempt = attempts[operation["accepted_attempt_id"]]
        _require(
            attempt["operation_id"] == key and attempt["status"] == "succeeded",
            "Accepted attempt mismatch",
        )
        if key in interrupted:
            parent = attempts[attempt["parent_attempt_id"]]
            _require(
                attempt["attempt_number"] == operation["attempt_count"] == 2
                and parent["operation_id"] == key
                and parent["status"] == "unknown_outcome"
                and parent["attempt_number"] == 1,
                "Reroll parent lineage mismatch",
            )
            lineage.append(
                {
                    "operation_id": key,
                    "parent_attempt_id": parent["attempt_id"],
                    "accepted_attempt_id": attempt["attempt_id"],
                }
            )
        else:
            _require(
                attempt["attempt_number"] == operation["attempt_count"] == 1
                and attempt["parent_attempt_id"] is None,
                "Accepted operation was reissued",
            )
    _require(
        sorted(lineage, key=lambda item: item["operation_id"])
        == record["verification"]["reroll_lineage"],
        "Recorded lineage mismatch",
    )
    events = rows["events"]
    for event in events:
        event["payload"] = _json(event["payload_json"])
    for event_type, expected in (
        ("call_dispatched", set(calls)),
        ("call_completed", set(calls) - interrupted_calls),
        ("unknown_outcome", interrupted_calls),
    ):
        selected = [event for event in events if event["event_type"] == event_type]
        _require(
            len(selected) == len(expected)
            and {event["payload"]["call_id"] for event in selected} == expected,
            f"Durable event inventory mismatch: {event_type}",
        )
        for event in selected:
            call = calls[event["payload"]["call_id"]]
            time_field = (
                "dispatched_at_ns" if event_type == "call_dispatched" else "completed_at_ns"
            )
            _require(
                event["operation_id"] == call["operation_id"]
                and event["created_at_ns"] == call[time_field],
                "Durable event/call chronology mismatch",
            )
            if event_type == "call_completed":
                _require(
                    event["payload"]["cost_microusd"] == 0
                    and event["payload"]["cost_is_complete"] is True,
                    "Completion event cost mismatch",
                )
    rerolls = [event for event in events if event["event_type"] == "reroll_queued"]
    _require(
        len(rerolls) == concurrency
        and {event["operation_id"] for event in rerolls} == interrupted
        and all(event["payload"]["acknowledged_unknown"] is True for event in rerolls),
        "Explicit acknowledgement missing",
    )
    acquired = sorted(
        (event for event in events if event["event_type"] == "run_lease_acquired"),
        key=lambda event: event["created_at_ns"],
    )
    _require(
        len(acquired) == 4
        and acquired[0]["created_at_ns"] < kill
        and acquired[1]["created_at_ns"] >= expiry["resume_allowed_at_ns"],
        "Lease reacquisition chronology mismatch",
    )
    _require(
        [event["payload"]["epoch"] for event in acquired] == [1, 2, 3, 4],
        "Lease fencing lineage mismatch",
    )
    released = [event for event in events if event["event_type"] == "run_lease_released"]
    _require(
        len(released) == 3 and {event["payload"]["epoch"] for event in released} == {2, 3, 4},
        "Killed lease cleaned up or completed lease not released",
    )
    stages = {
        "after_kill": {
            "succeeded": len(before_ids),
            "running": len(interrupted),
            "pending": len(pending_ids),
        },
        "after_resume": {"succeeded": count - concurrency, "unknown_outcome": concurrency},
        "final": {"succeeded": count},
    }
    _require(record["state_counts"] == stages, "Recorded stage arithmetic mismatch")
    return {
        "pillow_version": pillow_version,
        "operations": count,
        "accepted_artifacts": len(receipts),
        "calls": len(calls),
        "events": len(events),
        "explicit_rerolls": len(lineage),
        "state_counts": stages,
        "accepted_receipts_sha256": _sha(_json_bytes(receipts)),
        "preserved_precrash_receipts_sha256": _sha(_json_bytes(before)),
    }


def archive_campaign(campaign, result, source_root, output):
    """Package a finished producer result; never open SQLite in the input tree."""
    campaign, result = _safe_path(campaign), _safe_path(result)
    output = _safe_path(output, missing=True)
    _require(not output.exists(), "Publication directory already exists")
    _require(
        not output.resolve().is_relative_to(campaign.resolve())
        and not campaign.resolve().is_relative_to(output.resolve())
        and not result.resolve().is_relative_to(campaign.resolve())
        and not result.resolve().is_relative_to(output.resolve()),
        "Input/output locations overlap",
    )
    raw = _read(result)
    record = _json(raw)
    _require(
        record.get("schema") == "smythe.jobs-scale-recovery.v1"
        and record.get("campaign_status") in ("completed", "failed"),
        "A finished external producer result is required",
    )
    original = _tree(campaign)
    files = {"result.json": result}
    for name, (path, _, directory) in original.items():
        files[f"campaign/{name}" + ("/" if directory else "")] = path
    source_error = None
    try:
        files.update(_frozen_sources(source_root, record, campaign))
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        source_error = str(error)
    review_root = Path(__file__).absolute().parents[1]
    for name in (
        "benchmarks/archive_jobs_scale.py",
        "benchmarks/jobs_scale_chart.py",
        "benchmarks/render_readme_charts.py",
        "tests/test_jobs_scale_archive.py",
    ):
        files[f"review-source/{name}"] = _safe_path(review_root / name)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "evidence.zip"
    members = {}
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, path in sorted(files.items()):
            _member_name(name)
            directory = name.endswith("/")
            data = b"" if directory else _read(path)
            if (
                name.startswith("source/")
                and _sha(data.replace(b"\r\n", b"\n"))
                != record["provenance"]["source_sha256"][name[len("source/") :]]
            ):
                source_error = f"Frozen source changed before archive: {name}"
            details = path.lstat()
            archive.writestr(name, data)
            members[name] = {
                "kind": "directory" if directory else "file",
                "size_bytes": len(data),
                "sha256": _sha(data),
                "original_mtime_ns": details.st_mtime_ns,
            }
    verify_archive(archive_path, members)
    current = _tree(campaign)
    _require(
        {name: values[1:] for name, values in original.items()}
        == {name: values[1:] for name, values in current.items()},
        "Campaign changed during archiving",
    )
    for name, path in files.items():
        if not name.endswith("/"):
            _require(
                _sha(_read(path)) == members[name]["sha256"],
                "Original bytes changed during archive verification",
            )
    _require(_read(result) == raw, "Result changed during archive")
    (output / "result.json").write_bytes(raw)
    inventory = {
        "schema": "smythe.jobs-scale-archive.v1",
        "record_sha256": _sha(raw),
        "archive": {
            "path": "evidence.zip",
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha(_read(archive_path)),
        },
        "source_revision": record.get("provenance", {}).get("git_revision")
        if record.get("provenance")
        else None,
        "source_hash_policy": "Archive members use raw bytes; source identity also requires LF-normalized equality to the frozen Git blobs.",
        "members": members,
    }
    (output / "inventory.json").write_bytes(_json_bytes(inventory))
    review = {
        "schema": "smythe.jobs-scale-review.v1",
        "status": "failed",
        "scope": "offline_correctness_observation",
        "observation_claimable": False,
        "record_sha256": _sha(raw),
        "known_measurement_defects": [],
        "checks": dict.fromkeys(REVIEW_CHECKS, False),
        "source_revision": inventory["source_revision"],
        "archive_sha256": inventory["archive"]["sha256"],
        "inventory_sha256": _sha(_json_bytes(inventory)),
        "review_source_sha256": {
            name: details["sha256"]
            for name, details in members.items()
            if name.startswith("review-source/")
        },
        "review_runtime": {"python": sys.version, "sqlite": sqlite3.sqlite_version},
        "method": "Every original archive member verified by raw SHA/size. SQLite is inspected only from an extracted copy. Historical accepted sets are reconstructed from final ledger events/provider logs and bound receipt subsets, not historical database snapshots.",
    }
    review["checks"]["archive_verified"] = True
    try:
        _require(source_error is None, f"Frozen source verification failed: {source_error}")
        _require(
            record["campaign_status"] == "completed",
            "Producer retained a failed/incomplete campaign",
        )
        _require(
            set(record["retained_evidence"])
            == {
                name
                for name, (_, _, directory) in original.items()
                if not directory and "/" not in name
            },
            "Incomplete producer top-level inventory",
        )
        for name, detail in record["retained_evidence"].items():
            actual = members.get(f"campaign/{name}")
            _require(
                actual is not None
                and actual["sha256"] == detail["sha256"]
                and actual["size_bytes"] == detail["size_bytes"],
                f"Producer inventory mismatch: {name}",
            )
        with tempfile.TemporaryDirectory(prefix="smythe-jobs-archive-review-") as temporary:
            extracted = Path(temporary)
            _require(
                extracted.parent.resolve() == Path(tempfile.gettempdir()).resolve()
                and extracted.name.startswith("smythe-jobs-archive-review-"),
                "Unexpected temporary extraction root",
            )
            _extract(archive_path, extracted, members)
            review["reconciled"] = _review_copy(extracted / "campaign", record)
        review["checks"] = dict.fromkeys(REVIEW_CHECKS, True)
        review["checks"]["frozen_source_verified"] = True
        review["status"] = "passed"
        review["observation_claimable"] = True
        # The public chart accepts only this complete 5,000-operation profile.
        from benchmarks.jobs_scale_chart import validate_jobs_scale

        validate_jobs_scale(record, review, record_sha256=_sha(raw))
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError, ImportError) as error:
        review["status"] = "failed"
        review["observation_claimable"] = False
        review["known_measurement_defects"] = [f"{type(error).__name__}: {error}"]
    (output / "review.json").write_bytes(_json_bytes(review))
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review = archive_campaign(args.campaign, args.result, args.source_root, args.out)
    print(
        json.dumps(
            {
                "status": review["status"],
                "output": str(args.out),
                "record_sha256": review["record_sha256"],
                "known_measurement_defects": review["known_measurement_defects"],
            }
        )
    )
    if review["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
