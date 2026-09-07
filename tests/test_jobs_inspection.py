"""Operator inspection preserves ledger state and bounds artifact reads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.inspection import inspect_job, list_jobs
from smythe.jobs.store import SQLiteRunStore


@pytest.fixture
def recorded(tmp_path):
    manifest = JobManifestV1.from_dict({
        "version": 1,
        "name": "inspection-fixture",
        "profiles": [{"name": "default", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [{"key": "glyph", "count": 2, "prompt": "Original prompt",
                        "profile": "default"}],
        "execution": {"max_concurrency": 2, "max_attempts": 1,
                      "max_budget_usd": "0", "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    database = tmp_path / "jobs.sqlite3"
    paths = []
    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path)
        for index, op in enumerate(store.pending_operations(run_id)):
            attempt = store.begin_attempt(run_id, op["operation_id"])
            permit = store.prepare_call(attempt["attempt_id"], 0)
            store.mark_call_dispatched(permit.call_id)
            content = f"artifact-{index}".encode()
            relative = f"artifacts/{index}.bin"
            path = tmp_path / "outputs" / run_id / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            paths.append(path)
            store.complete_call(
                permit.call_id, cost_microusd=0, cost_is_complete=True,
                cost_is_estimate=False, result_text=f"Response {index}", artifacts=[{
                    "relative_path": relative, "mime_type": "application/octet-stream",
                    "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }],
            )
        store.finalize_run(run_id)
    return database, run_id, paths


def test_inspection_reads_recorded_prompt_response_cost_and_artifact(recorded):
    database, run_id, paths = recorded
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)
        assert list_jobs(store)["runs"][0]["run_id"] == run_id
    assert database.read_bytes() == before
    assert report["status"] == "completed"
    assert report["counts"] == {"succeeded": 2}
    assert report["cost"]["confirmed_microusd"] == 0
    assert report["operations"][0]["spec"]["prompt"] == "Original prompt"
    assert report["operations"][0]["result_text"] == "Response 0"
    assert len(report["calls"]) == 2
    assert report["artifact_integrity"]["counts"] == {"verified": 2}
    assert report["artifact_integrity"]["bytes_hashed"] == sum(p.stat().st_size for p in paths)
    assert report["inspection_version"] == 1
    assert report["inspected_at_ns"] > report["created_at_ns"]
    json.dumps(report)  # Structured CLI output must remain serializable.


def test_inspection_page_preserves_whole_run_counts(recorded):
    database, run_id, _ = recorded
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id, limit=1, offset=1, events_limit=2)
    assert report["counts"] == {"succeeded": 2}
    assert len(report["operations"]) == len(report["attempts"]) == len(report["artifacts"]) == 1
    assert report["pagination"] == {
        "limit": 1, "offset": 1, "total": 2, "returned": 1, "has_more": False,
    }
    assert len(report["events"]) == 2
    assert report["event_pagination"]["has_more"] is True


@pytest.mark.parametrize("change,expected", [("missing", "missing"), ("size", "changed"),
                                             ("hash", "changed")])
def test_current_disk_finding_does_not_rewrite_acceptance(recorded, change, expected):
    database, run_id, paths = recorded
    if change == "missing":
        paths[0].unlink()
    elif change == "size":
        paths[0].write_bytes(b"different length")
    else:
        paths[0].write_bytes(b"x" * paths[0].stat().st_size)
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)
    assert report["artifacts"][0]["integrity"]["status"] == expected
    assert report["artifacts"][0]["accepted"] == 1
    assert report["status"] == "completed"
    assert database.read_bytes() == before


def _fake_store(tmp_path, **updates):
    data = b"fixture"
    path = tmp_path / "outputs" / "run" / "artifact.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    snapshot = {
        "run_id": "run", "manifest_root": str(tmp_path), "output_directory": "outputs",
        "artifacts": [{"relative_path": "artifact.bin", "size_bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}],
    }
    snapshot.update(updates)
    return SimpleNamespace(inspection_snapshot=lambda *a, **kw: copy.deepcopy(snapshot)), path, snapshot


@pytest.mark.parametrize("path", ["../outside", "..\\outside", "/etc/passwd",
                                  "C:\\outside", "C:outside", "\\\\host\\share",
                                  "artifact.bin:stream", ".", "", "a\x00b"])
def test_artifact_paths_do_not_escape_output_directory(tmp_path, path, monkeypatch):
    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"][0]["relative_path"] = path
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("unsafe file opened"))
    assert inspect_job(store, "run")["artifacts"][0]["integrity"]["status"] == "unsafe"


@pytest.mark.parametrize("field,value", [("output_directory", "../outside"),
                                        ("output_directory", "C:\\outside"),
                                        ("run_id", "../outside"),
                                        ("manifest_root", "."),
                                        ("manifest_root", b"bad")])
def test_unsafe_ledger_roots_are_findings_not_reads(tmp_path, field, value, monkeypatch):
    store, _, _ = _fake_store(tmp_path, **{field: value})
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("unsafe root opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


def test_manifest_output_dot_is_supported(tmp_path):
    store, path, _ = _fake_store(tmp_path, output_directory=".")
    target = tmp_path / "run" / "artifact.bin"
    target.parent.mkdir()
    target.write_bytes(path.read_bytes())
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"verified": 1}


def test_symbolic_link_is_never_hashed(tmp_path, monkeypatch):
    store, path, _ = _fake_store(tmp_path)
    original = tmp_path / "original.bin"
    original.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(original)
    except OSError:
        pytest.skip("Creating a symbolic link requires Windows Developer Mode")
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("symbolic link opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


def test_hash_work_is_bounded_across_artifacts(tmp_path, monkeypatch):
    import smythe.jobs.inspection as inspection

    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"].append(dict(snapshot["artifacts"][0]))
    monkeypatch.setattr(inspection, "MAX_INSPECTION_HASH_BYTES", 7)
    report = inspect_job(store, "run")
    assert report["artifact_integrity"]["counts"] == {"verified": 1, "not_checked": 1}
    assert report["artifact_integrity"]["bytes_hashed"] == 7


def test_unreadable_file_is_preserved_as_a_finding(tmp_path, monkeypatch):
    store, _, _ = _fake_store(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "open", denied)
    report = inspect_job(store, "run")
    assert report["artifact_integrity"]["counts"] == {"unreadable": 1}


@pytest.mark.parametrize("field,value", [("size_bytes", True), ("size_bytes", -1),
                                        ("size_bytes", 33 * 1024 * 1024),
                                        ("sha256", "not-a-hash")])
def test_invalid_artifact_metadata_is_not_used_to_read_files(tmp_path, field, value, monkeypatch):
    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"][0][field] = value
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("invalid record opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}
