"""Jobs inspection commands are scriptable, local, and read-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from smythe.cli import (
    EXIT_INVALID_INPUT, EXIT_JOB_FAILED, EXIT_JOB_STATE, EXIT_LOCAL_ERROR, EXIT_OK,
    build_parser, main,
)
from smythe.jobs.loading import load_manifest
from smythe.jobs.preflight import make_approval, preflight_job
from smythe.jobs.store import SQLiteRunStore
from test_jobs_cli import _manifest


def output(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    return json.loads(captured.out, parse_constant=lambda value: pytest.fail(f"Nonfinite JSON: {value}"))


def seed_job(tmp_path, capsys, *, rejected=False):
    manifest = _manifest(tmp_path, mime_type="image/jpeg" if rejected else "image/png", attempts=2)
    path = tmp_path / "jobs.sqlite3"
    assert main(["jobs", "plan", str(manifest), "--json"]) == EXIT_OK
    token = output(capsys)["plan"]["approval"]["token"]
    code = main(["jobs", "run", str(manifest), "--approve", token, "--store", str(path), "--json"])
    assert code == (EXIT_JOB_FAILED if rejected else EXIT_OK)
    run = output(capsys)["run"]
    return SimpleNamespace(path=path, run=run, manifest=manifest)


@pytest.fixture
def job(tmp_path, capsys):
    return seed_job(tmp_path, capsys)


def no_providers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Read command initialized a provider or runner")

    monkeypatch.setattr("smythe.cli.ProviderPool", forbidden)
    monkeypatch.setattr("smythe.jobs.providers.ProviderPool", forbidden)
    monkeypatch.setattr("smythe.cli.JobRunner", forbidden)


def test_parser_defaults_and_inherited_output_options():
    defaults = build_parser().parse_args(["jobs", "list"])
    assert (defaults.limit, defaults.offset, defaults.status) == (50, 0, None)
    args = build_parser().parse_args([
        "jobs", "--store", "saved.sqlite", "--json", "inspect", "run-1", "--operation", "glyph",
    ])
    assert args.json is True and args.store == "saved.sqlite" and args.operation == "glyph"
    assert (args.limit, args.offset, args.events_limit) == (50, 0, 100)


@pytest.mark.parametrize("command", ["list", "inspect", "status", "export"])
def test_read_commands_never_initialize_providers_or_change_store(job, capsys, monkeypatch, command):
    no_providers(monkeypatch)
    before = hashlib.sha256(job.path.read_bytes()).hexdigest()
    modified = job.path.stat().st_mtime_ns
    args = ["jobs", "--store", str(job.path), "--json", command]
    if command != "list":
        args.append(job.run["run_id"])
    assert main(args) == EXIT_OK
    value = output(capsys)
    assert value["ok"] is True and value["command"] == command
    assert job.run["run_id"] in json.dumps(value[command])
    assert hashlib.sha256(job.path.read_bytes()).hexdigest() == before
    assert job.path.stat().st_mtime_ns == modified


@pytest.mark.parametrize("command", ["list", "inspect", "status", "export"])
def test_missing_read_store_is_not_created(tmp_path, capsys, monkeypatch, command):
    no_providers(monkeypatch)
    path = tmp_path / "missing" / "jobs.sqlite3"
    args = ["jobs", command] + ([] if command == "list" else ["run-1"])
    assert main([*args, "--store", str(path), "--json"]) == EXIT_LOCAL_ERROR
    value = output(capsys)
    assert value["ok"] is False and value["command"] == command
    assert not path.parent.exists()


def test_foreign_sqlite_database_is_rejected_without_schema_changes(tmp_path, capsys):
    path = tmp_path / "other.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    original = path.read_bytes()
    for command in ("list", "inspect"):
        args = ["jobs", command] + ([] if command == "list" else ["any-run"])
        assert main([*args, "--store", str(path), "--json"]) == EXIT_LOCAL_ERROR
        assert output(capsys)["ok"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("arguments", [
    ["--limit", "0"], ["--limit", "501"], ["--offset", "-1"],
    ["--offset", str(1 << 63)], ["--status", "not-a-status"],
])
def test_invalid_list_filters_have_json_errors_without_mutating_store(job, capsys, arguments):
    before = job.path.read_bytes()
    assert main(["jobs", "list", *arguments, "--store", str(job.path), "--json"]) == EXIT_INVALID_INPUT
    assert output(capsys)["error"]["type"] == "ValueError"
    assert job.path.read_bytes() == before


def test_list_pagination_and_status_filter(job, capsys):
    manifest, root = load_manifest(job.manifest)
    plan = preflight_job(manifest, manifest_root=root)
    with SQLiteRunStore(job.path) as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="approved-a")
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="approved-b")
    assert main(["jobs", "list", "--limit", "1", "--offset", "1", "--status", "approved",
                 "--store", str(job.path), "--json"]) == EXIT_OK
    listed = output(capsys)["list"]
    serialized = json.dumps(listed)
    assert "approved-a" in serialized and "approved-b" not in serialized
    assert job.run["run_id"] not in serialized


@pytest.mark.parametrize("selector", ["key", "id"])
def test_inspect_selects_an_operation_by_key_or_id(job, capsys, selector):
    operation = job.run["operations"][0]
    selected = operation["operation_key" if selector == "key" else "operation_id"]
    assert main(["jobs", "inspect", job.run["run_id"], "--operation", selected,
                 "--store", str(job.path), "--json"]) == EXIT_OK
    payload = output(capsys)
    assert payload["ok"] is True and payload["command"] == "inspect"
    assert operation["operation_id"] in json.dumps(payload["inspect"])


def test_missing_run_and_operation_have_stable_error_outputs(job, capsys):
    assert main(["jobs", "inspect", "missing-run", "--store", str(job.path), "--json"]) == EXIT_JOB_STATE
    assert output(capsys)["error"]["type"] == "JobNotFoundError"
    assert main(["jobs", "inspect", job.run["run_id"], "--operation", "missing-operation",
                 "--store", str(job.path), "--json"]) == EXIT_INVALID_INPUT
    assert output(capsys)["ok"] is False


def test_inspecting_failed_job_succeeds_but_status_exit_is_unchanged(tmp_path, capsys):
    job = seed_job(tmp_path, capsys, rejected=True)
    operation = job.run["operations"][0]["operation_key"]
    assert main(["jobs", "reroll", job.run["run_id"], operation, "--reason", "MIME check",
                 "--store", str(job.path), "--json"]) == EXIT_JOB_FAILED
    rerolled = output(capsys)["reroll"]
    assert rerolled["operations"][0]["attempt_count"] == 2
    assert main(["jobs", "inspect", job.run["run_id"], "--store", str(job.path), "--json"]) == EXIT_OK
    inspected = output(capsys)["inspect"]
    assert len(inspected["attempts"]) == 2 and len(inspected["calls"]) == 2
    assert inspected["attempts"][1]["parent_attempt_id"] == inspected["attempts"][0]["attempt_id"]
    assert main(["jobs", "status", job.run["run_id"], "--store", str(job.path), "--json"]) == EXIT_JOB_FAILED
    assert output(capsys)["status"]["counts"] == {"rejected": 1}


def test_human_inspection_shows_costs_call_states_attempts_and_artifact_findings(job, capsys):
    assert main(["jobs", "inspect", job.run["run_id"], "--store", str(job.path)]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.err == ""
    assert job.run["run_id"] in captured.out and "completed" in captured.out
    for label in ("Confirmed", "Exposure", "Reserved", "Call state", "Attempt", "Artifact"):
        assert label in captured.out


def test_human_output_escapes_stored_terminal_controls(capsys):
    from smythe.cli import _emit_job_inspection

    _emit_job_inspection({
        "run_id": "safe-run", "name": "unsafe\x1b[2J\nname", "status": "completed", "counts": {},
        "cost": {"approved_microusd": 0, "confirmed_microusd": 0, "exposure_microusd": 0,
                 "reserved_microusd": 0, "cost_is_complete": True, "cost_contains_estimates": False},
        "pagination": {"returned": 0, "total": 0, "offset": 0, "limit": 50, "has_more": False},
        "event_pagination": {"returned": 0, "total": 0}, "operations": [], "attempts": [],
        "calls": [], "artifacts": [],
    })
    human = capsys.readouterr().out
    assert "\x1b" not in human and "unsafe\\u001b[2J\\u000aname" in human


def test_human_errors_escape_terminal_controls(job, capsys):
    assert main(["jobs", "inspect", job.run["run_id"], "--operation", "bad\x1b[2J",
                 "--store", str(job.path)]) == EXIT_INVALID_INPUT
    captured = capsys.readouterr()
    assert captured.out == "" and "\x1b" not in captured.err


def test_inspect_operation_and_event_pages_are_explicit(job, capsys):
    assert main(["jobs", "inspect", job.run["run_id"], "--limit", "1", "--offset", "0",
                 "--events-limit", "1", "--store", str(job.path), "--json"]) == EXIT_OK
    inspected = output(capsys)["inspect"]
    assert inspected["pagination"]["limit"] == 1 and inspected["pagination"]["offset"] == 0
    assert inspected["pagination"]["returned"] == len(inspected["operations"]) == 1
    assert inspected["event_pagination"]["limit"] == 1
    assert len(inspected["events"]) == 1 and inspected["event_pagination"]["has_more"]


def test_inspect_writes_explicit_html_and_still_emits_one_json_document(job, tmp_path, capsys):
    destination = tmp_path / "reports" / "inspection.html"
    before = job.path.read_bytes()
    assert main(["jobs", "inspect", job.run["run_id"], "--out", str(destination),
                 "--store", str(job.path), "--json"]) == EXIT_OK
    payload = output(capsys)
    assert payload["inspect"]["report_path"] == str(destination.resolve())
    html = destination.read_text(encoding="utf-8")
    assert "<html" in html.lower() and job.run["run_id"] in html
    assert job.path.read_bytes() == before
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("command", ["inspect", "export"])
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_output_cannot_replace_store_or_sqlite_sidecars(job, capsys, command, suffix):
    before = job.path.read_bytes()
    assert main(["jobs", command, job.run["run_id"], "--out", str(job.path) + suffix,
                 "--store", str(job.path), "--json"]) == EXIT_INVALID_INPUT
    assert output(capsys)["ok"] is False
    assert job.path.read_bytes() == before


def test_report_write_failure_has_a_local_error_and_no_partial_file(job, tmp_path, capsys):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("keep", encoding="utf-8")
    assert main(["jobs", "inspect", job.run["run_id"], "--out", str(blocker / "report.html"),
                 "--store", str(job.path), "--json"]) == EXIT_LOCAL_ERROR
    assert output(capsys)["ok"] is False
    assert blocker.read_text(encoding="utf-8") == "keep"


def test_module_cli_stdout_is_a_single_json_inspection(job):
    completed = subprocess.run(
        [sys.executable, "-m", "smythe.cli", "jobs", "inspect", job.run["run_id"],
         "--store", str(job.path), "--json"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stderr == "" and len(completed.stdout.splitlines()) == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True and payload["command"] == "inspect"
