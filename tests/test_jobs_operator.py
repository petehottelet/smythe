"""Owned worker startup decisions and read-only attachment contracts."""

import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from smythe.jobs import make_approval, preflight_job
from smythe.jobs.loading import load_manifest
from smythe.jobs import operator
from smythe.jobs.providers import ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import RunStoreError, SQLiteRunStore
from test_jobs_cli import _manifest


@pytest.fixture
def journal(tmp_path):
    manifest, root = load_manifest(_manifest(tmp_path))
    plan = preflight_job(manifest, manifest_root=root)
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="run")
        yield store


class Child:
    pid = 123

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pytest.fail("A stored PID or child must not be killed")

    terminate = kill


def fake_launch(monkeypatch, journal, *, behavior="ready"):
    calls = []
    monkeypatch.setattr(operator, "_select_process_options", lambda deadline: operator._process_options())

    def popen(command, **kwargs):
        directory = Path(command[-1]).parent
        request = operator._read_json(directory / "request.json")
        calls.append((command, kwargs, request, directory))
        if behavior in {"ready", "wrong_nonce", "stale", "bad_pid"}:
            lease = journal.acquire_run_lease("run", "test-worker")
            ready = operator._record(request, state="ready", pid=456, lease_owner_id=lease.owner_id,
                                     lease_epoch=lease.epoch, python_executable=sys.executable,
                                     python_prefix=sys.prefix)
            if behavior == "wrong_nonce":
                ready["nonce"] = "f" * 32
            elif behavior == "bad_pid":
                ready["pid"] = True
            elif behavior == "stale":
                journal.release_run_lease("run", lease.owner_id, lease=lease)
            operator._atomic_json(directory / "ready.json", ready)
        elif behavior == "failure":
            operator._atomic_json(directory / "finished.json", operator._record(
                request, state="failed", error="Provider preflight refused configuration", exit_code=7,
                error_type="ProviderConfigurationError", pid=456,
            ))
        return Child(7 if behavior == "exited" else None)

    monkeypatch.setattr(operator.subprocess, "Popen", popen)
    return calls


def test_launch_requires_matching_live_lease_and_reports_actual_worker_pid(journal, monkeypatch):
    calls = fake_launch(monkeypatch, journal)
    launched = operator.launch_worker(journal.path, "run")
    command, options, request, directory = calls[0]
    assert launched["detached"] is True and launched["status"] == "started"
    assert launched["worker_pid"] == 456 != Child.pid
    assert "nonce" not in launched and launched["run_id"] == "run"
    assert command[:6] == [sys.executable, "-P", "-u", "-m", "smythe.jobs.operator", "--worker"]
    assert options["stdin"] == subprocess.DEVNULL and options["stderr"] == subprocess.STDOUT
    assert options["shell"] is False and options["close_fds"] is True
    assert options["stdout"].closed
    assert options["env"]["PYTHONPATH"]
    decision = operator._read_json(directory / "decision.json")
    assert decision["action"] == "start" and decision["nonce"] == request["nonce"]
    assert request["clear_pause"] is False and request["pause_generation"] is None
    assert (directory / "worker.log").is_file()
    if os.name != "nt":
        assert options["start_new_session"] is True
        assert directory.stat().st_mode & 0o777 == 0o700
        assert (directory / "worker.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("behavior", ["wrong_nonce", "stale", "bad_pid", "failure", "exited"])
def test_unproven_startup_aborts_without_start_decision(journal, monkeypatch, behavior):
    calls = fake_launch(monkeypatch, journal, behavior=behavior)
    with pytest.raises(operator.WorkerStartupError) as error:
        operator.launch_worker(journal.path, "run")
    directory = calls[0][3]
    assert operator._read_json(directory / "decision.json")["action"] == "abort"
    assert operator._read_json(directory / "launcher-failure.json")["state"] == "failed"
    assert error.value.launch["run_id"] == "run" and "worker.log" in str(error.value)
    assert error.value.launch["startup_authorized"] is False
    assert "nonce" not in error.value.launch.get("worker", {})
    assert journal._call_rows("run") == []


def test_startup_timeout_permanently_aborts_a_late_child(journal, monkeypatch):
    calls = fake_launch(monkeypatch, journal, behavior="silent")
    tick = [0.0]
    monkeypatch.setattr(operator.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(operator.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    with pytest.raises(operator.WorkerStartupError, match="deadline"):
        operator.launch_worker(journal.path, "run", startup_timeout_s=.1)
    request, directory = calls[0][2:]
    with pytest.raises(operator._StartupAborted, match="aborted|expired"):
        asyncio.run(operator._wait_for_start(directory, request))
    assert operator._read_json(directory / "decision.json")["action"] == "abort"
    assert journal._call_rows("run") == []


def test_start_decision_cannot_be_accepted_after_slow_read_crosses_deadline(monkeypatch, tmp_path):
    tick = [0.0]
    request = {"version": 1, "run_id": "run", "launch_id": "a" * 32, "nonce": "b" * 32,
               "startup_timeout_s": .1, "expires_at_ns": time.time_ns() + int(1e9)}
    monkeypatch.setattr(operator.time, "monotonic", lambda: tick[0])

    def slow_read(*args, **kwargs):
        tick[0] = .2
        return operator._record(request, action="start")

    monkeypatch.setattr(operator, "_read_json", slow_read)
    with pytest.raises(operator._StartupAborted, match="while reading"):
        asyncio.run(operator._wait_for_start(tmp_path, request))


def test_launch_collision_never_changes_previous_files(journal, monkeypatch):
    fixed = "a" * 32
    monkeypatch.setattr(operator, "uuid4", lambda: SimpleNamespace(hex=fixed))
    root = operator._root(journal.path, "run")
    root.mkdir(parents=True)
    previous = root / fixed
    previous.mkdir()
    marker = previous / "worker.log"
    marker.write_bytes(b"previous accepted worker log")
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Child spawned"))
    with pytest.raises(operator.WorkerStartupError):
        operator.launch_worker(journal.path, "run")
    assert list(previous.iterdir()) == [marker]
    assert marker.read_bytes() == b"previous accepted worker log"


def test_exact_run_ids_have_distinct_portable_private_directories(tmp_path):
    identifiers = ["run", "RUN", "run.", "CON", "NUL", "COM1", "LPT1"]
    locations = {}
    for run_id in identifiers:
        directory = operator._root(tmp_path / "jobs.db", run_id)
        assert directory.name == hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        directory.mkdir(parents=True)
        (directory / "identity.txt").write_text(run_id, encoding="utf-8")
        locations[run_id] = directory
    assert len({path.name.casefold() for path in locations.values()}) == len(identifiers)
    for run_id, directory in locations.items():
        assert (directory / "identity.txt").read_text(encoding="utf-8") == run_id


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), 301, "30"])
def test_invalid_startup_bound_has_no_filesystem_side_effect(tmp_path, value):
    with pytest.raises(ValueError):
        operator.launch_worker(tmp_path / "missing.db", "run", startup_timeout_s=value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["../run", "a/b", "a\\b", "", "a" * 129, True])
def test_invalid_identity_has_no_filesystem_side_effect(tmp_path, value):
    with pytest.raises(ValueError):
        operator.launch_worker(tmp_path / "missing.db", value)
    assert list(tmp_path.iterdir()) == []


def test_private_directory_failure_retains_created_run_identity(journal, monkeypatch):
    monkeypatch.setattr(operator, "_private_directory", lambda *args, **kwargs: (_ for _ in ()).throw(
        PermissionError("cannot create private directory")))
    with pytest.raises(operator.WorkerStartupError) as error:
        operator.launch_worker(journal.path, "run")
    assert error.value.launch["run_id"] == "run"
    assert journal.get_run("run")["status"] == "approved"


def test_detached_resume_records_parent_observed_pause_generation(journal, monkeypatch):
    calls = fake_launch(monkeypatch, journal)
    monkeypatch.setattr(SQLiteRunStore, "get_control", lambda *args: {"pause_generation": 4})
    operator.launch_worker(journal.path, "run", clear_pause=True)
    request = calls[0][2]
    assert request["clear_pause"] is True and request["pause_generation"] == 4


@pytest.mark.parametrize("options", [
    {"clear_pause": 1}, {"pause_generation": 0}, {"clear_pause": True, "pause_generation": True},
    {"clear_pause": True, "pause_generation": -1}, {"clear_pause": True, "pause_generation": 1 << 63},
])
def test_invalid_resume_intent_does_not_create_launch_files(tmp_path, options):
    with pytest.raises(ValueError):
        operator.launch_worker(tmp_path / "missing.db", "run", **options)
    assert list(tmp_path.iterdir()) == []


def test_invalid_saved_pause_generation_is_rejected_before_launch(journal, monkeypatch):
    monkeypatch.setattr(SQLiteRunStore, "get_control", lambda *args: {"pause_generation": True})
    with pytest.raises(RunStoreError, match="generation"):
        operator.launch_worker(journal.path, "run", clear_pause=True)
    assert not operator._root(journal.path, "run").parent.exists()


def request_for(journal, *, expired=False):
    directory = operator._root(journal.path, "run") / ("a" * 32)
    directory.mkdir(parents=True)
    request = {"version": 1, "run_id": "run", "launch_id": "a" * 32, "nonce": "b" * 32,
               "store_path": str(journal.path), "startup_timeout_s": 30.0,
               "clear_pause": False, "pause_generation": None,
               "expires_at_ns": time.time_ns() + (-1 if expired else int(30e9))}
    operator._exclusive_json(directory / "request.json", request)
    operator._atomic_json(directory.parent / "latest.json", {"version": 1, "run_id": "run",
                                                            "launch_id": request["launch_id"]})
    return request, directory


def test_worker_emits_lease_receipt_then_waits_before_any_provider_dispatch(journal):
    request, directory = request_for(journal)

    async def scenario():
        worker = asyncio.create_task(operator._run_worker(request, directory))
        try:
            async with asyncio.timeout(15):
                while not (directory / "ready.json").exists():
                    await asyncio.sleep(.01)
                ready = operator._read_json(directory / "ready.json")
                lease = journal.get_run_lease("run")
                assert lease.owner_id == ready["lease_owner_id"] and lease.epoch == ready["lease_epoch"]
                assert ready["pid"] == os.getpid() and ready["python_prefix"] == sys.prefix
                assert journal._call_rows("run") == []
                operator._decision(directory, request, "start")
                result = await worker
                assert result["status"] == "completed" and result["counts"] == {"succeeded": 1}
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("clear_pause,newer_pause,expected", [
    (False, False, "paused"), (True, False, "completed"), (True, True, "paused"),
])
def test_worker_respects_bound_resume_intent_and_newer_pause(journal, clear_pause, newer_pause, expected):
    request, directory = request_for(journal)
    observed = journal.request_pause("run", reason="first operator request")["pause_generation"]
    request.update(clear_pause=clear_pause, pause_generation=observed if clear_pause else None)
    operator._atomic_json(directory / "request.json", request)
    if newer_pause:
        journal.request_pause("run", reason="request after detached resume intent")

    async def scenario():
        worker = asyncio.create_task(operator._run_worker(request, directory))
        try:
            async with asyncio.timeout(15):
                while not (directory / "ready.json").exists():
                    await asyncio.sleep(.01)
                assert journal._call_rows("run") == []
                assert journal.get_control("run")["pause_requested"] is (expected == "paused")
                operator._decision(directory, request, "start")
                result = await worker
                assert result["status"] == expected
                assert result["counts"] == ({"pending": 1} if expected == "paused" else {"succeeded": 1})
                assert len(journal._call_rows("run")) == (0 if expected == "paused" else 1)
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_expired_worker_fails_before_provider_construction(journal, monkeypatch):
    request, directory = request_for(journal, expired=True)
    monkeypatch.setattr(ProviderPool, "__init__", lambda *args, **kwargs: pytest.fail("Provider initialized"))
    assert operator.worker_main(directory / "request.json") == 7
    receipt = operator._read_json(directory / "finished.json")
    assert receipt["state"] == "aborted" and receipt["nonce"] == request["nonce"]
    assert journal.get_run_lease("run") is None and journal._call_rows("run") == []


def test_second_worker_cannot_replace_receipt_or_reuse_launch(journal, monkeypatch):
    _, directory = request_for(journal)

    async def run(*args):
        return {"status": "completed", "counts": {"succeeded": 1}}

    monkeypatch.setattr(operator, "_run_worker", run)
    assert operator.worker_main(directory / "request.json") == 0
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        operator.worker_main(directory / "request.json")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_attachment_is_read_only_provider_free_and_bounded(journal, monkeypatch):
    lease = journal.acquire_run_lease("run", "worker")
    before = list(journal._connection.iterdump())
    tick = [0.0]
    monkeypatch.setattr(operator.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(operator.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds))
    monkeypatch.setattr(ProviderPool, "__init__", lambda *args, **kwargs: pytest.fail("Provider initialized"))
    monkeypatch.setattr(JobRunner, "__init__", lambda *args, **kwargs: pytest.fail("Runner initialized"))
    updates = []
    result = operator.attach_job(journal.path, "run", timeout_s=.1, poll_interval_s=.05,
                                 on_update=updates.append)
    assert result["attachment"] == {"state": "timeout", "lease_active": True,
                                    "latest_launch_matches_lease": None}
    assert result["worker"] is None and len(updates) == 2
    assert list(journal._connection.iterdump()) == before
    assert journal.get_run_lease("run") == lease
    assert not operator._root(journal.path, "run").parent.exists()


def test_interrupting_attachment_preserves_lease_and_all_saved_state(journal, monkeypatch):
    lease = journal.acquire_run_lease("run", "worker")
    before = list(journal._connection.iterdump())

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(operator.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        operator.attach_job(journal.path, "run")
    assert journal.get_run_lease("run") == lease
    assert list(journal._connection.iterdump()) == before


def test_attachment_reports_failed_startup_without_mutating_approved_run(journal):
    request, directory = request_for(journal)
    operator._atomic_json(directory / "launcher-failure.json", operator._record(
        request, state="failed", error="child exited before readiness", error_type="WorkerStartupError",
        startup_authorized=False,
    ))
    before = list(journal._connection.iterdump())
    result = operator.attach_job(journal.path, "run")
    assert result["attachment"]["state"] == "worker_failed"
    assert result["status"] == "approved" and result["worker"]["state"] == "failed"
    assert "nonce" not in result["worker"]
    assert list(journal._connection.iterdump()) == before


def test_attachment_detects_expired_startup_even_if_launcher_never_wrote_failure(journal):
    request_for(journal, expired=True)
    before = list(journal._connection.iterdump())
    result = operator.attach_job(journal.path, "run", timeout_s=0)
    assert result["attachment"]["state"] == "worker_failed"
    assert result["worker"]["state"] == "startup_expired" and result["status"] == "approved"
    assert list(journal._connection.iterdump()) == before


@pytest.mark.parametrize("failed", [False, True])
def test_attachment_keeps_latest_launch_separate_from_another_workers_live_lease(journal, failed):
    request, directory = request_for(journal)
    current = journal.acquire_run_lease("run", "actual-owner")
    if failed:
        receipt = operator._record(request, state="failed", pid=777, exit_code=7,
                                   error_type="RunLeaseError", error="another worker owns the lease")
        filename = "finished.json"
    else:
        receipt = operator._record(request, state="ready", pid=777, lease_owner_id="old-owner",
                                   lease_epoch=current.epoch, python_executable=sys.executable,
                                   python_prefix=sys.prefix)
        filename = "ready.json"
    operator._atomic_json(directory / filename, receipt)
    before = list(journal._connection.iterdump())
    result = operator.attach_job(journal.path, "run", timeout_s=0)
    assert result["attachment"]["state"] == "timeout" and result["attachment"]["lease_active"]
    assert result["attachment"]["latest_launch_matches_lease"] is (None if failed else False)
    assert result["lease"]["owner_id"] == "actual-owner" and result["lease"]["epoch"] == current.epoch
    assert "pid" not in result["lease"] and result["worker"]["pid"] == 777
    assert list(journal._connection.iterdump()) == before


@pytest.mark.parametrize("options", [
    {"timeout_s": True}, {"timeout_s": -1}, {"timeout_s": 3601},
    {"timeout_s": float("nan")}, {"poll_interval_s": 0}, {"poll_interval_s": 61},
])
def test_invalid_attachment_bounds_do_not_create_store(tmp_path, options):
    with pytest.raises(ValueError):
        operator.attach_job(tmp_path / "missing.db", "run", **options)
    assert list(tmp_path.iterdir()) == []


def test_worker_request_rejects_duplicate_json_fields_and_foreign_launch_path(journal, tmp_path):
    request, directory = request_for(journal)
    foreign = tmp_path / "request.json"
    foreign.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RunStoreError, match="outside"):
        operator.worker_main(foreign)
    (directory / "request.json").write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(RunStoreError, match="JSON"):
        operator.worker_main(directory / "request.json")
    assert not (directory / "claimed.json").exists()


def test_read_worker_state_rejects_foreign_nonce(journal):
    request, directory = request_for(journal)
    value = operator._record(request, state="ready", pid=123)
    value["nonce"] = "f" * 32
    operator._atomic_json(directory / "ready.json", value)
    with pytest.raises(RunStoreError, match="match"):
        operator.read_worker_state(journal.path, "run")


def test_python_environment_keeps_venv_paths_without_changing_parent(monkeypatch, tmp_path):
    before = deepcopy(os.environ)
    monkeypatch.setattr(operator.sys, "path", [str(tmp_path), "", str(tmp_path)])
    environment = operator._python_environment()
    assert environment["PYTHONPATH"] == str(tmp_path.resolve())
    assert dict(os.environ) == dict(before)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", str(1 << 63)])
def test_receipt_json_rejects_nonfinite_and_out_of_bounds_nested_numbers(tmp_path, literal):
    path = tmp_path / "receipt.json"
    path.write_text('{"nested":[{"value":' + literal + '}]}', encoding="utf-8")
    with pytest.raises(RunStoreError, match="JSON"):
        operator._read_json(path)


@pytest.mark.parametrize("kind,change", [
    ("ready", {"state": None}), ("ready", {"state": "invented"}),
    ("ready", {"recorded_at_ns": True}), ("ready", {"recorded_at_ns": -1}),
    ("ready", {"pid": True}), ("ready", {"lease_epoch": 0}),
    ("ready", {"lease_owner_id": ""}), ("ready", {"python_prefix": None}),
    ("finished", {"state": {}}), ("finished", {"exit_code": True}),
    ("finished", {"run_status": []}), ("finished", {"counts": {"succeeded": True}}),
    ("launcher-failure", {"state": "finished"}), ("launcher-failure", {"error_type": None}),
])
def test_readonly_receipt_validation_rejects_malformed_states_and_fields(journal, kind, change):
    request, directory = request_for(journal)
    if kind == "ready":
        fields = {"state": "ready", "pid": 1, "lease_owner_id": "owner", "lease_epoch": 1,
                  "python_executable": sys.executable, "python_prefix": sys.prefix}
    elif kind == "finished":
        fields = {"state": "finished", "pid": 1, "exit_code": 0,
                  "run_status": "completed", "counts": {"succeeded": 1}}
    else:
        fields = {"state": "failed", "error_type": "OSError", "error": "startup failed",
                  "startup_authorized": False}
    value = dict(operator._record(request, **fields), **change)
    if change.get("state", "present") is None:
        del value["state"]
    operator._atomic_json(directory / f"{kind}.json", value)
    before = list(journal._connection.iterdump())
    with pytest.raises(RunStoreError, match="receipt"):
        operator.attach_job(journal.path, "run", timeout_s=0)
    assert list(journal._connection.iterdump()) == before


def test_receipt_nonce_non_ascii_is_a_controlled_error(journal):
    request, directory = request_for(journal)
    value = operator._record(request, state="ready", nonce="\N{SNOWMAN}")
    operator._atomic_json(directory / "ready.json", value)
    with pytest.raises(RunStoreError, match="match"):
        operator.read_worker_state(journal.path, "run")


def test_startup_interrupt_writes_abort_and_never_signals_child(journal, monkeypatch):
    calls = fake_launch(monkeypatch, journal, behavior="silent")

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(operator.time, "sleep", interrupt)
    with pytest.raises(operator.WorkerStartupInterrupted) as error:
        operator.launch_worker(journal.path, "run")
    assert error.value.launch["startup_authorized"] is False
    request, directory = calls[0][2:]
    assert operator._read_json(directory / "decision.json")["action"] == "abort"
    assert journal._call_rows("run") == []
    with pytest.raises(operator._StartupAborted):
        asyncio.run(operator._wait_for_start(directory, request))


@pytest.mark.parametrize("failure", ["interrupt", "write", "fsync", "unreadable"])
def test_start_authorization_visible_before_error_is_never_claimed_aborted(journal, monkeypatch, failure):
    calls = fake_launch(monkeypatch, journal)
    original_decision = operator._decision
    original_exclusive = operator._exclusive_json
    original_read = operator._read_json

    def decision(directory, request, action):
        if action == "start" and failure == "interrupt":
            original_decision(directory, request, action)
            raise KeyboardInterrupt
        if action == "start" and failure in {"write", "fsync", "unreadable"}:
            path = directory / "decision.json"
            # Model a flushed complete write that a child can read, followed
            # by a write/fsync failure before the launcher receives success.
            if failure == "fsync":
                original_fsync = operator.os.fsync

                def fail_fsync(descriptor):
                    raise OSError("decision fsync failed after bytes became visible")

                monkeypatch.setattr(operator.os, "fsync", fail_fsync)
                try:
                    original_exclusive(path, operator._record(request, action="start"))
                finally:
                    monkeypatch.setattr(operator.os, "fsync", original_fsync)
            else:
                path.write_bytes(operator._json_bytes(operator._record(request, action="start")))
                if failure == "unreadable":
                    def read(target, **kwargs):
                        if target == path:
                            raise OSError("decision cannot be read")
                        return original_read(target, **kwargs)
                    monkeypatch.setattr(operator, "_read_json", read)
                raise OSError("decision write reported failure after bytes became visible")
        return original_decision(directory, request, action)

    monkeypatch.setattr(operator, "_decision", decision)
    error_type = operator.WorkerStartupInterrupted if failure == "interrupt" else operator.WorkerStartupError
    with pytest.raises(error_type) as error:
        operator.launch_worker(journal.path, "run")
    expected = None if failure == "unreadable" else True
    assert error.value.launch["startup_authorized"] is expected
    assert "may already be running" in str(error.value)
    request, directory = calls[0][2:]
    assert original_read(directory / "decision.json")["action"] == "start"
    assert original_read(directory / "launcher-failure.json")["startup_authorized"] is expected
    monkeypatch.setattr(operator, "_read_json", original_read)
    # Independent child-side check demonstrates why calling this an abort
    # would be false even though the launcher raised an exception.
    asyncio.run(operator._wait_for_start(directory, request))


def test_deadline_crossing_during_start_write_is_not_reported_as_success(journal, monkeypatch):
    calls = fake_launch(monkeypatch, journal)
    original = operator._decision
    now = [time.time_ns()]
    monkeypatch.setattr(operator.time, "time_ns", lambda: now[0])

    def decision(directory, request, action):
        result = original(directory, request, action)
        if action == "start":
            now[0] = request["expires_at_ns"] + 1
        return result

    monkeypatch.setattr(operator, "_decision", decision)
    with pytest.raises(operator.WorkerStartupError, match="while recording") as error:
        operator.launch_worker(journal.path, "run")
    assert error.value.launch["startup_authorized"] is True
    request, directory = calls[0][2:]
    assert operator._read_json(directory / "decision.json")["action"] == "start"
    with pytest.raises(operator._StartupAborted, match="expired"):
        asyncio.run(operator._wait_for_start(directory, request))


def test_process_options_detach_from_terminal_and_hide_windows():
    options = operator._process_options()
    if os.name == "nt":
        expected = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_BREAKAWAY_FROM_JOB)
        assert options["creationflags"] == expected
        assert options["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
        assert options["startupinfo"].wShowWindow == 0
    else:
        assert options == {"start_new_session": True}


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation policy")
@pytest.mark.parametrize("denied", [False, True])
def test_windows_probe_is_isolated_no_work_and_bounded(monkeypatch, denied):
    calls = []
    monkeypatch.setattr(operator.time, "monotonic", lambda: 10)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if denied:
            raise __import__("ctypes").WinError(5)

    monkeypatch.setattr(operator.subprocess, "run", run)
    if denied:
        with pytest.raises(operator._BreakawayDenied):
            operator._select_process_options(15)
        options = operator._process_options()
    else:
        options = operator._select_process_options(15)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [sys.executable, "-I", "-S", "-c", "pass"]
    assert kwargs["timeout"] == 5 and kwargs["check"] is True
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["shell"] is False and kwargs["close_fds"] is True
    assert "env" not in kwargs  # Isolated Python ignores inherited startup configuration.
    assert kwargs["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB
    assert options["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB
    assert options["creationflags"] & subprocess.DETACHED_PROCESS
    assert options["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation policy")
@pytest.mark.parametrize("kind", ["filename", "other_code", "timeout", "nonzero"])
def test_windows_probe_other_failures_are_not_fallbacks(monkeypatch, kind):
    import ctypes

    if kind == "filename":
        error = PermissionError(13, "private file denied", "private.log", 5)
    elif kind == "other_code":
        error = ctypes.WinError(1314)
    elif kind == "timeout":
        error = subprocess.TimeoutExpired("probe", 1)
    else:
        error = subprocess.CalledProcessError(1, "probe")

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(operator.subprocess, "run", run)
    with pytest.raises(type(error)) as caught:
        operator._select_process_options(time.monotonic() + 1)
    assert caught.value is error


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation policy")
def test_actual_worker_access_denied_is_never_retried_or_authorized(journal, monkeypatch):
    import ctypes

    attempts = []
    monkeypatch.setattr(operator, "_select_process_options", lambda deadline: operator._process_options())

    def popen(*args, **kwargs):
        attempts.append(args)
        raise ctypes.WinError(5)

    monkeypatch.setattr(operator.subprocess, "Popen", popen)
    with pytest.raises(operator.WorkerStartupError) as caught:
        operator.launch_worker(journal.path, "run")
    assert len(attempts) == 1 and caught.value.launch["startup_authorized"] is False
    directory = Path(caught.value.launch["receipt_path"])
    assert operator._read_json(directory / "decision.json")["action"] == "abort"
    assert journal._call_rows("run") == []


@pytest.mark.skipif(os.name != "nt", reason="Windows breakaway policy")
def test_denied_breakaway_refuses_worker_and_authorization(journal, monkeypatch):
    import ctypes

    monkeypatch.setattr(operator.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(
        ctypes.WinError(5)))
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Worker spawned"))
    with pytest.raises(operator.WorkerStartupError, match="did not start a worker") as caught:
        operator.launch_worker(journal.path, "run")
    assert caught.value.launch["startup_authorized"] is False
    directory = Path(caught.value.launch["receipt_path"])
    assert operator._read_json(directory / "decision.json")["action"] == "abort"
    assert caught.value.launch["unsupported_reason"] == "windows_breakaway_denied"
    assert not (directory / "ready.json").exists() and journal._call_rows("run") == []


def test_process_probe_cannot_spend_past_startup_deadline(journal, monkeypatch):
    tick = [0.0]
    monkeypatch.setattr(operator.time, "monotonic", lambda: tick[0])

    def select(deadline):
        tick[0] = deadline
        return {}

    monkeypatch.setattr(operator, "_select_process_options", select)
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Worker spawned"))
    with pytest.raises(operator.WorkerStartupError, match="after process probe") as caught:
        operator.launch_worker(journal.path, "run")
    assert caught.value.launch["startup_authorized"] is False
    assert journal._call_rows("run") == []



def test_stop_persists_request_without_constructing_or_signaling_a_worker(journal, monkeypatch):
    lease = journal.acquire_run_lease("run", "worker")
    monkeypatch.setattr(ProviderPool, "__init__", lambda *args, **kwargs: pytest.fail("Provider initialized"))
    monkeypatch.setattr(JobRunner, "__init__", lambda *args, **kwargs: pytest.fail("Runner initialized"))
    result = operator.stop_job(journal.path, "run", reason="operator maintenance", timeout_s=0)
    assert result["stop_request"]["pause_generation"] == 1
    assert result["control"]["pause_requested"] is True
    assert result["control"]["pause_reason"] == "operator maintenance"
    assert result["attachment"]["state"] == "timeout"
    assert journal.get_run_lease("run") == lease and journal._call_rows("run") == []


def test_stop_observation_interrupt_leaves_the_committed_request(journal, monkeypatch):
    def attach(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(operator, "attach_job", attach)
    result = operator.stop_job(journal.path, "run")
    assert result["attachment"]["state"] == "disconnected"
    assert result["stop_request"]["pause_generation"] == 1
    assert journal.get_control("run")["pause_requested"] is True


@pytest.mark.parametrize("options", [
    {"timeout_s": float("nan")}, {"timeout_s": True}, {"poll_interval_s": 0},
    {"reason": ""}, {"reason": " "}, {"reason": "x" * 2049}, {"reason": 1},
])
def test_invalid_stop_options_do_not_create_store(tmp_path, options):
    with pytest.raises(ValueError):
        operator.stop_job(tmp_path / "missing.db", "run", **options)
    assert list(tmp_path.iterdir()) == []


def test_stop_missing_store_is_not_created(tmp_path):
    with pytest.raises((OSError, RunStoreError)):
        operator.stop_job(tmp_path / "missing.db", "run")
    assert list(tmp_path.iterdir()) == []


def test_attachment_worker_finishing_after_snapshot_is_not_reported_as_lease_failure(journal, monkeypatch):
    lease = journal.acquire_run_lease("run", "healthy-worker")
    operation = journal.pending_operations("run")[0]
    attempt = journal.begin_attempt("run", operation["operation_id"], lease=lease)
    permit = journal.prepare_call(attempt["attempt_id"], 0, lease=lease)
    journal.mark_call_dispatched(permit.call_id, lease=lease)
    original = SQLiteRunStore.inspection_snapshot
    finished = False

    def snapshot(self, *args, **kwargs):
        nonlocal finished
        value = original(self, *args, **kwargs)
        if not finished:
            finished = True
            journal.complete_call(permit.call_id, cost_microusd=0, cost_is_complete=True,
                                  cost_is_estimate=False, artifacts=[], result_text="done", lease=lease)
            journal.finalize_run("run", lease=lease)
            journal.release_run_lease("run", lease.owner_id, lease=lease)
        return value

    monkeypatch.setattr(SQLiteRunStore, "inspection_snapshot", snapshot)
    monkeypatch.setattr(SQLiteRunStore, "get_run_lease", lambda *args: pytest.fail("Split lease read"))
    updates = []
    result = operator.attach_job(journal.path, "run", timeout_s=10, poll_interval_s=.05,
                                 on_update=updates.append)
    assert result["attachment"]["state"] == "stopped" and result["status"] == "completed"
    assert updates[0]["status"] == "running" and updates[0]["attachment"]["lease_active"]
    assert all(item["attachment"]["state"] != "lease_expired" for item in updates)


def test_attachment_uses_observation_clock_before_receipt_read_delay(journal, monkeypatch):
    original = SQLiteRunStore.inspection_snapshot

    def snapshot(self, *args, **kwargs):
        return dict(original(self, *args, **kwargs), status="running", observed_at_ns=50,
                    lease={"run_id": "run", "owner_id": "worker", "epoch": 1,
                           "acquired_at_ns": 1, "heartbeat_at_ns": 1, "expires_at_ns": 100})

    monkeypatch.setattr(SQLiteRunStore, "inspection_snapshot", snapshot)
    monkeypatch.setattr(operator.time, "time_ns", lambda: 200)
    result = operator.attach_job(journal.path, "run", timeout_s=0)
    assert result["attachment"]["state"] == "timeout" and result["attachment"]["lease_active"]
