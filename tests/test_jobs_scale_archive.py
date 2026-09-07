"""Synthetic archive/review fixtures; no live provider or production result."""

from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace
import zipfile

import pytest

from benchmarks import archive_jobs_scale as archive


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive._json_bytes(value))


@pytest.fixture
def failed_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "start-worker.log").write_bytes(b"retained failure\r\n\x00\xff")
    (campaign / "empty").mkdir()
    record = {
        "schema": "smythe.jobs-scale-recovery.v1",
        "campaign_status": "failed",
        "provenance": None,
        "failure": {"message": "synthetic worker failure"},
    }
    result = tmp_path / "original-result.json"
    result.write_bytes(json.dumps(record, indent=2).replace("\n", "\r\n").encode() + b"\r\n")
    return campaign, result


def test_failed_outcome_preserves_exact_bytes_and_never_opens_database(
    failed_campaign, tmp_path, monkeypatch
):
    campaign, result = failed_campaign
    before = {
        p.relative_to(campaign).as_posix(): p.read_bytes()
        for p in campaign.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        archive.sqlite3, "connect", lambda *a, **kw: pytest.fail("Source database opened")
    )
    output = tmp_path / "bundle"
    review = archive.archive_campaign(campaign, result, tmp_path, output)
    assert review["status"] == "failed" and review["observation_claimable"] is False
    assert review["checks"]["archive_verified"] is True
    assert (output / "result.json").read_bytes() == result.read_bytes()
    inventory = json.loads((output / "inventory.json").read_bytes())
    assert inventory["record_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert (
        inventory["archive"]["sha256"]
        == hashlib.sha256((output / "evidence.zip").read_bytes()).hexdigest()
    )
    archive.verify_archive(output / "evidence.zip", inventory["members"])
    with zipfile.ZipFile(output / "evidence.zip") as zipped:
        assert zipped.read("result.json") == result.read_bytes()
        assert set(review["review_source_sha256"]) == {
            "review-source/benchmarks/archive_jobs_scale.py",
            "review-source/benchmarks/jobs_scale_chart.py",
            "review-source/benchmarks/render_readme_charts.py",
            "review-source/tests/test_jobs_scale_archive.py",
        }
        for name, digest in review["review_source_sha256"].items():
            assert archive._sha(zipped.read(name)) == digest
        for name, data in before.items():
            assert zipped.read("campaign/" + name) == data
        assert "campaign/empty/" in zipped.namelist()
    assert before == {
        p.relative_to(campaign).as_posix(): p.read_bytes()
        for p in campaign.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("damage", ["running", "output_exists", "output_inside", "result_inside"])
def test_archive_refuses_unfinished_or_overlapping_paths(failed_campaign, tmp_path, damage):
    campaign, result = failed_campaign
    output = tmp_path / "bundle"
    if damage == "running":
        record = json.loads(result.read_bytes())
        record["campaign_status"] = "running"
        put(result, record)
    elif damage == "output_exists":
        output.mkdir()
        (output / "keep").write_bytes(b"preserve")
    elif damage == "output_inside":
        output = campaign / "bundle"
    else:
        moved_result = campaign / "result.json"
        moved_result.write_bytes(result.read_bytes())
        result = moved_result
    with pytest.raises(ValueError):
        archive.archive_campaign(campaign, result, tmp_path, output)
    if damage == "output_exists":
        assert (output / "keep").read_bytes() == b"preserve"
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/file",
        "a\\file",
        "a/../file",
        "a//file",
        "a//",
        "NUL.txt",
        "dir/file. ",
        "a/\x00bad",
    ],
)
def test_unsafe_archive_names_rejected(name):
    with pytest.raises(ValueError):
        archive._member_name(name)


def test_symlink_input_rejected(failed_campaign, tmp_path):
    campaign, result = failed_campaign
    link = campaign / "linked"
    try:
        link.symlink_to(result)
    except OSError:
        pytest.skip("Host does not permit creating test symlinks")
    with pytest.raises(ValueError, match="Links|junctions|reparse"):
        archive.archive_campaign(campaign, result, tmp_path, tmp_path / "bundle")


def test_reparse_directory_rejected_without_following_it(tmp_path, monkeypatch):
    root = tmp_path / "junction"
    root.mkdir()
    original = Path.lstat

    def reparse(path):
        result = original(path)
        if path == root:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ValueError, match="reparse"):
        archive._safe_path(root)


@pytest.mark.parametrize("damage", ["hash", "size", "extra", "duplicate", "symlink"])
def test_archive_member_verification_fails_closed(tmp_path, damage):
    path = tmp_path / "bad.zip"
    data = b"original"
    expected = {"file": {"kind": "file", "size_bytes": len(data), "sha256": archive._sha(data)}}
    with zipfile.ZipFile(path, "w") as zipped:
        info = zipfile.ZipInfo("file")
        if damage == "symlink":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zipped.writestr(info, data)
        if damage == "extra":
            zipped.writestr("extra", b"unlisted")
        elif damage == "duplicate":
            with pytest.warns(UserWarning):
                zipped.writestr("file", data)
    if damage == "hash":
        expected["file"]["sha256"] = "0" * 64
    elif damage == "size":
        expected["file"]["size_bytes"] += 1
    with pytest.raises(ValueError):
        archive.verify_archive(path, expected)


def test_source_verification_checks_checkout_and_committed_blobs(tmp_path, monkeypatch):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    source.mkdir()
    campaign.mkdir()
    names = [
        "benchmarks/jobs_scale_benchmark.py",
        "smythe/example.py",
        "tests/test_jobs_scale_benchmark.py",
    ]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"value = 1\r\n")
    hashes = dict.fromkeys(names, archive._sha(b"value = 1\n"))
    provenance = {
        "git_revision": archive.EXPECTED_REVISION,
        "source_sha256": hashes,
        "git_status_at_start": [],
    }
    put(campaign / "provenance.json", provenance)

    def git(command, **kwargs):
        if "rev-parse" in command:
            return archive.EXPECTED_REVISION + "\n"
        if "ls-files" in command:
            return b"smythe/example.py\x00"
        assert "cat-file" in command
        data = b"value = 1\n"
        return b"".join(
            b"a" * 40 + b" blob " + str(len(data)).encode() + b"\n" + data + b"\n" for _ in names
        )

    monkeypatch.setattr(archive.subprocess, "check_output", git)
    assert len(archive._frozen_sources(source, {"provenance": provenance}, campaign)) == 3
    (source / "smythe/example.py").write_bytes(b"changed = 2\n")
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        archive._frozen_sources(source, {"provenance": provenance}, campaign)


@pytest.fixture
def reconstructed_campaign(tmp_path):
    """Twelve synthetic rows exercise the same reconstruction without a 5k run."""
    root = tmp_path / "synthetic"
    root.mkdir()
    count, concurrency, run_id = 12, 2, "jobs-scale"
    config = {
        "count": count,
        "concurrency": concurrency,
        "max_attempts": 2,
        "lease_ttl_s": 30.0,
        "lease_heartbeat_s": 1.0,
        "kill_after_entries": 6,
        "completed_before_kill": 4,
    }
    put(root / "config.json", config)
    phases, provider_logs = {}, {}
    operations, attempts, calls, artifacts, events, receipts = [], [], [], [], [], {}

    def event(kind, time, op=None, **payload):
        events.append(
            {
                "run_id": run_id,
                "operation_id": op,
                "event_type": kind,
                "payload_json": json.dumps(payload),
                "created_at_ns": time,
            }
        )

    ranges = {
        "start": range(6),
        "resume": range(6, 12),
        "reroll": range(4, 6),
        "finished_resume": [],
    }
    bases = {
        "start": 1_000_000_000,
        "resume": 13_000_000_000,
        "reroll": 16_000_000_000,
        "finished_resume": 20_000_000_000,
    }
    for epoch, phase in enumerate(archive.PHASES, 1):
        event(
            "run_lease_acquired",
            [500_000_000, 12_000_000_002, 15_000_000_000, 20_000_000_000][epoch - 1],
            epoch=epoch,
        )
        if epoch > 1:
            event(
                "run_lease_released",
                [0, 14_000_000_000, 17_000_000_000, 21_000_000_000][epoch - 1],
                epoch=epoch,
            )
        log = []
        for index in ranges[phase]:
            op = f"op-{index}"
            attempt_id, call_id = f"a-{phase}-{index}", f"c-{phase}-{index}"
            stamp = bases[phase] + index * 100
            unknown = phase == "start" and index in (4, 5)
            identity = {
                "phase": phase,
                "pid": 101 + epoch - 1,
                "operation_id": op,
                "attempt_id": attempt_id,
                "call_id": call_id,
            }
            log.append(identity | {"event": "entered", "time_ns": stamp})
            if not unknown:
                log.append(identity | {"event": "returned", "time_ns": stamp + 10})
            completed = 12_000_000_010 + index if unknown else stamp + 20
            attempts.append(
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "operation_id": op,
                    "status": "unknown_outcome" if unknown else "succeeded",
                    "attempt_number": 2 if phase == "reroll" else 1,
                    "parent_attempt_id": f"a-start-{index}" if phase == "reroll" else None,
                    "lease_epoch": epoch,
                }
            )
            calls.append(
                {
                    "run_id": run_id,
                    "call_id": call_id,
                    "attempt_id": attempt_id,
                    "operation_id": op,
                    "status": "unknown_outcome" if unknown else "succeeded",
                    "created_at_ns": stamp - 2,
                    "dispatched_at_ns": stamp - 1,
                    "completed_at_ns": completed,
                    "ceiling_microusd": 0,
                    "confirmed_microusd": 0,
                    "exposure_microusd": 0,
                    "cost_is_complete": int(not unknown),
                    "cost_is_estimate": int(unknown),
                }
            )
            event("call_dispatched", stamp - 1, op, call_id=call_id)
            event(
                "unknown_outcome" if unknown else "call_completed",
                completed,
                op,
                call_id=call_id,
                cost_microusd=0,
                cost_is_complete=not unknown,
            )
            if phase == "reroll":
                event("reroll_queued", 15_000_000_010 + index, op, acknowledged_unknown=True)
            if not unknown:
                relative = f"{op}/{attempt_id}.png"
                destination = root / "outputs/jobs-scale" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.PNG)
                artifact = {
                    "run_id": run_id,
                    "artifact_id": f"art-{index}",
                    "attempt_id": attempt_id,
                    "operation_id": op,
                    "relative_path": relative,
                    "sha256": archive._sha(archive.PNG),
                    "size_bytes": len(archive.PNG),
                    "mime_type": "image/png",
                    "width": 1,
                    "height": 1,
                    "accepted": 1,
                    "created_at_ns": completed,
                }
                artifacts.append(artifact)
                receipts[op] = {
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
                operations.append(
                    {
                        "run_id": run_id,
                        "operation_id": op,
                        "status": "succeeded",
                        "accepted_attempt_id": attempt_id,
                        "attempt_count": 2 if phase == "reroll" else 1,
                        "spec_json": '{"provider":"offline"}',
                    }
                )
        provider_logs[phase] = log
        if log:
            (root / f"{phase}-provider.jsonl").write_bytes(
                b"".join(json.dumps(row).encode() + b"\n" for row in log)
            )
        summary = {
            "entries": len(ranges[phase]),
            "returns": len(ranges[phase]) - (2 if phase == "start" else 0),
            "peak_active_provider_calls": 2 if phase == "start" else int(bool(log)),
            "interrupted_call_ids": ["c-start-4", "c-start-5"] if phase == "start" else [],
        }
        phases[phase] = {"provider": summary}
        if phase != "start":
            result = {
                "status": "needs_attention" if phase == "resume" else "completed",
                "counts": {"succeeded": 10, "unknown_outcome": 2}
                if phase == "resume"
                else {"succeeded": count},
                "execution_metrics": {
                    "operations_started": len(ranges[phase]),
                    "peak_active_calls": 2 if log else 0,
                },
                "worker_wall_s": 1.0,
            }
            put(root / f"{phase}-result.json", result)
            phases[phase].update(result)
        put(root / f"{phase}-durability.json", {"journal_mode": "wal", "synchronous": 2})
        (root / f"{phase}-worker.log").write_bytes(b"")
    barrier = {
        "pid": 101,
        "provider_entries": 6,
        "completed_operations": 4,
        "ready_time_ns": 9_000_000_000,
    }
    put(root / "barrier.json", barrier)
    phases["start"].update(
        barrier=barrier, kill_time_ns=10_000_000_000, exit_code=1, kill_mechanism="TerminateProcess"
    )
    cost = dict.fromkeys(
        ("approved_microusd", "confirmed_microusd", "exposure_microusd", "reserved_microusd"), 0
    )
    runs = [{"run_id": run_id, "status": "completed", "output_directory": "outputs"} | cost]
    events.sort(key=lambda row: row["created_at_ns"])
    for number, row in enumerate(events, 1):
        row["sequence"] = number
    with closing(sqlite3.connect(root / "jobs.db")) as connection:
        for table, rows in {
            "runs": runs,
            "operations": operations,
            "attempts": attempts,
            "calls": calls,
            "artifacts": artifacts,
            "events": events,
            "run_leases": [],
        }.items():
            columns = list(rows[0]) if rows else ["run_id"]
            connection.execute(
                f"CREATE TABLE {table} ("
                + ",".join(
                    f"{key} " + ("INTEGER" if rows and type(rows[0][key]) is int else "TEXT")
                    for key in columns
                )
                + ")"
            )
            connection.executemany(
                f"INSERT INTO {table} VALUES (" + ",".join("?" for _ in columns) + ")",
                [tuple(row[key] for key in columns) for row in rows],
            )
        connection.execute("PRAGMA user_version=3")
        connection.commit()
    put(root / "reroll-ids.json", ["op-4", "op-5"])
    record = {
        "schema": "smythe.jobs-scale-recovery.v1",
        "campaign_status": "completed",
        "configuration": config,
        "phases": phases,
        "durability": {
            phase: {"journal_mode": "wal", "synchronous": 2} for phase in archive.PHASES
        },
        "lease_expiry": {
            "expires_at_ns": 12_000_000_000,
            "resume_allowed_at_ns": 12_000_000_001,
            "wall_s": 2.0,
            "expiry_method": "elapsed real wall-clock time; no database edits",
        },
        "ledger": {key: dict(cost) for key in ("after_kill", "after_resume", "final")},
        "state_counts": {
            "after_kill": {"succeeded": 4, "running": 2, "pending": 6},
            "after_resume": {"succeeded": 10, "unknown_outcome": 2},
            "final": {"succeeded": 12},
        },
        "verification": {
            "durable_call_status_counts": dict(Counter(row["status"] for row in calls)),
            "preserved_precrash_receipts_sha256": archive._sha(
                archive._json_bytes({f"op-{i}": receipts[f"op-{i}"] for i in range(4)})
            ),
            "accepted_receipts_sha256": archive._sha(archive._json_bytes(receipts)),
            "fixture_bytes": len(archive.PNG),
            "fixture_sha256": archive._sha(archive.PNG),
            "artifact_bytes": count * len(archive.PNG),
            "artifact_dimensions": [1, 1],
            "artifact_mime_type": "image/png",
            "unique_artifact_content_hashes": 1,
            "reroll_lineage": [
                {
                    "operation_id": f"op-{i}",
                    "parent_attempt_id": f"a-start-{i}",
                    "accepted_attempt_id": f"a-reroll-{i}",
                }
                for i in (4, 5)
            ],
        },
    }
    return root, record


def test_synthetic_ledger_reconstructs_precrash_resume_and_reroll_sets(reconstructed_campaign):
    root, record = reconstructed_campaign
    result = archive._review_copy(root, record)
    assert result["operations"] == result["accepted_artifacts"] == 12
    assert result["calls"] == 14 and result["explicit_rerolls"] == 2
    assert result["state_counts"] == record["state_counts"]


@pytest.mark.parametrize(
    "damage",
    [
        "artifact",
        "pointer",
        "lineage",
        "charge",
        "event",
        "unknown_flags",
        "digest",
        "lease",
        "kill_pid",
        "state_counts",
        "orphan_file",
    ],
)
def test_independent_review_rejects_corrupt_reconstruction(reconstructed_campaign, damage):
    root, record = reconstructed_campaign
    if damage == "artifact":
        next((root / "outputs").rglob("*.png")).write_bytes(b"changed")
    elif damage in ("pointer", "lineage", "charge", "event", "unknown_flags"):
        queries = {
            "pointer": "UPDATE operations SET accepted_attempt_id='foreign' WHERE operation_id='op-1'",
            "lineage": "UPDATE attempts SET parent_attempt_id=NULL WHERE attempt_id='a-reroll-4'",
            "charge": "UPDATE calls SET confirmed_microusd=1 WHERE call_id='c-resume-6'",
            "event": "DELETE FROM events WHERE event_type='unknown_outcome'",
            "unknown_flags": "UPDATE calls SET cost_is_complete=1 WHERE call_id='c-start-4'",
        }
        with closing(sqlite3.connect(root / "jobs.db")) as connection:
            connection.execute(queries[damage])
            connection.commit()
    elif damage == "digest":
        record["verification"]["preserved_precrash_receipts_sha256"] = "0" * 64
    elif damage == "lease":
        record["lease_expiry"]["resume_allowed_at_ns"] = 1
    elif damage == "kill_pid":
        record["phases"]["start"]["barrier"]["pid"] = 999
    elif damage == "state_counts":
        record["state_counts"]["after_kill"]["running"] = 0
    else:
        (root / "outputs/orphan.png").write_bytes(archive.PNG)
    with pytest.raises((ValueError, KeyError)):
        archive._review_copy(root, record)


def test_archive_review_opens_only_extracted_copy(reconstructed_campaign, tmp_path, monkeypatch):
    campaign, record = reconstructed_campaign
    record["provenance"] = {"git_revision": archive.EXPECTED_REVISION, "source_sha256": {}}
    record["retained_evidence"] = {
        p.name: {"sha256": archive._sha(p.read_bytes()), "size_bytes": p.stat().st_size}
        for p in campaign.iterdir()
        if p.is_file()
    }
    result = tmp_path / "terminal-result.json"
    put(result, record)
    monkeypatch.setattr(archive, "_frozen_sources", lambda *args: {})
    original = archive.sqlite3.connect
    opened = []

    def connect(path, **kwargs):
        opened.append(str(path))
        assert str(campaign).replace("\\", "/") not in str(path)
        assert "smythe-jobs-archive-review-" in str(path) and "?mode=ro" in str(path)
        return original(path, **kwargs)

    monkeypatch.setattr(archive.sqlite3, "connect", connect)
    before = archive._sha((campaign / "jobs.db").read_bytes())
    review = archive.archive_campaign(campaign, result, tmp_path, tmp_path / "bundle")
    assert len(opened) == 1
    assert review["reconciled"]["operations"] == 12
    assert (
        review["status"] == "failed" and not review["observation_claimable"]
    )  # Reduced synthetic scope cannot qualify.
    assert archive._sha((campaign / "jobs.db").read_bytes()) == before


def test_incomplete_producer_inventory_retained_but_cannot_pass(
    reconstructed_campaign, tmp_path, monkeypatch
):
    campaign, record = reconstructed_campaign
    record["provenance"] = {"git_revision": archive.EXPECTED_REVISION, "source_sha256": {}}
    record["retained_evidence"] = {
        path.name: {"sha256": archive._sha(path.read_bytes()), "size_bytes": path.stat().st_size}
        for path in campaign.iterdir()
        if path.is_file() and path.name != "config.json"
    }
    result = tmp_path / "terminal-result.json"
    put(result, record)
    monkeypatch.setattr(archive, "_frozen_sources", lambda *args: {})
    monkeypatch.setattr(
        archive, "_review_copy", lambda *args: pytest.fail("Incomplete inventory admitted")
    )
    output = tmp_path / "bundle"
    review = archive.archive_campaign(campaign, result, tmp_path, output)
    assert review["status"] == "failed" and not review["observation_claimable"]
    assert "Incomplete producer top-level inventory" in review["known_measurement_defects"][0]
    with zipfile.ZipFile(output / "evidence.zip") as zipped:
        assert zipped.read("campaign/config.json") == (campaign / "config.json").read_bytes()
