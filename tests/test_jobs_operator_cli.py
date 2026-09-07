"""Detached CLI launch and read-only attach, including terminal independence."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from smythe.cli import EXIT_APPROVAL, EXIT_INTERRUPTED, EXIT_INVALID_INPUT, EXIT_OK, main
from smythe.jobs import make_approval, preflight_job
from smythe.jobs.loading import load_manifest
from smythe.jobs import operator
from smythe.jobs.store import SQLiteRunStore
from test_jobs_cli import _manifest


def planned(tmp_path):
    path = _manifest(tmp_path)
    manifest, root = load_manifest(path)
    plan = preflight_job(manifest, manifest_root=root)
    return path, plan, make_approval(plan)


def payload(capsys):
    return json.loads(capsys.readouterr().out)


def test_run_detach_persists_approved_run_before_launch_and_closes_parent_store(tmp_path, monkeypatch, capsys):
    manifest, plan, approval = planned(tmp_path)
    database = tmp_path / "jobs.db"
    requests = []

    def launch(store_path, run_id, **kwargs):
        with SQLiteRunStore(store_path) as store:
            saved = store.get_run(run_id)
            assert saved["approval_token"] == approval.token and saved["plan_hash"] == plan.plan_hash
            assert saved["status"] == "approved" and store.get_run_lease(run_id) is None
            assert store._call_rows(run_id) == []
        requests.append(kwargs)
        return {"run_id": run_id, "detached": True, "status": "started", "worker_pid": 42,
                "log_path": str(tmp_path / "worker.log")}

    monkeypatch.setattr(operator, "launch_worker", launch)
    assert main(["jobs", "run", str(manifest), "--approve", approval.token, "--detach",
                 "--store", str(database), "--json"]) == EXIT_OK
    result = payload(capsys)["run"]
    assert result["detached"] is True and result["worker_pid"] == 42
    assert requests == [{"startup_timeout_s": 30.0}]


def test_detach_rejects_invalid_approval_and_timeout_before_creating_run(tmp_path, monkeypatch, capsys):
    manifest, _, approval = planned(tmp_path)
    database = tmp_path / "jobs.db"
    monkeypatch.setattr(operator, "launch_worker", lambda *args, **kwargs: pytest.fail("Worker launched"))
    assert main(["jobs", "run", str(manifest), "--approve", "wrong", "--detach",
                 "--store", str(database), "--json"]) == EXIT_APPROVAL
    assert payload(capsys)["ok"] is False and not database.exists()
    assert main(["jobs", "run", str(manifest), "--approve", approval.token, "--detach",
                 "--startup-timeout-s", "nan", "--store", str(database), "--json"]) == EXIT_INVALID_INPUT
    assert payload(capsys)["ok"] is False and not database.exists()


def test_resume_detach_is_explicit_pause_resume_intent(tmp_path, monkeypatch, capsys):
    calls = []

    def launch(store_path, run_id, **kwargs):
        calls.append((store_path, run_id, kwargs))
        return {"run_id": run_id, "detached": True, "status": "started", "worker_pid": 42,
                "log_path": str(tmp_path / "worker.log")}

    monkeypatch.setattr(operator, "launch_worker", launch)
    database = tmp_path / "jobs.db"
    assert main(["jobs", "resume", "run", "--detach", "--store", str(database), "--json"]) == EXIT_OK
    assert payload(capsys)["resume"]["detached"] is True
    assert calls == [(str(database), "run", {"startup_timeout_s": 30.0, "clear_pause": True})]


def test_launch_error_exposes_recoverable_run_and_log_path(tmp_path, monkeypatch, capsys):
    def launch(*args, **kwargs):
        raise operator.WorkerStartupError("worker exited", {
            "run_id": "run", "launch_id": "a" * 32, "log_path": str(tmp_path / "private.log"),
        })

    monkeypatch.setattr(operator, "launch_worker", launch)
    assert main(["jobs", "resume", "run", "--detach", "--json"]) == 7
    failure = payload(capsys)
    assert failure["ok"] is False and failure["error"]["launch"]["run_id"] == "run"
    assert failure["error"]["launch"]["log_path"].endswith("private.log")


def test_startup_interrupt_keeps_one_json_document_and_recovery_identity(tmp_path, monkeypatch, capsys):
    def launch(*args, **kwargs):
        raise operator.WorkerStartupInterrupted({
            "run_id": "run", "launch_id": "a" * 32, "log_path": str(tmp_path / "private.log"),
        })

    monkeypatch.setattr(operator, "launch_worker", launch)
    assert main(["jobs", "resume", "run", "--detach", "--json"]) == EXIT_INTERRUPTED
    failure = payload(capsys)
    assert failure["ok"] is False and failure["error"]["launch"]["run_id"] == "run"


def test_attach_interrupt_disconnects_without_invoking_worker_controls(monkeypatch, capsys):
    def attach(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(operator, "attach_job", attach)
    monkeypatch.setattr(operator, "launch_worker", lambda *args, **kwargs: pytest.fail("Worker launched"))
    assert main(["jobs", "attach", "run", "--json"]) == EXIT_INTERRUPTED
    assert payload(capsys)["attach"] == {"run_id": "run", "attachment": {"state": "disconnected"}}


def test_json_attach_emits_one_document_even_with_multiple_observations(monkeypatch, capsys):
    def attach(*args, on_update, **kwargs):
        value = {"run_id": "run", "status": "completed", "counts": {"succeeded": 1},
                 "attachment": {"state": "stopped", "lease_active": False}, "worker": None}
        on_update(dict(value, status="running"))
        on_update(value)
        return value

    monkeypatch.setattr(operator, "attach_job", attach)
    assert main(["jobs", "attach", "run", "--json"]) == EXIT_OK
    assert payload(capsys)["attach"]["status"] == "completed"


@pytest.mark.parametrize("state", ["worker_failed", "lease_expired"])
def test_attach_worker_failure_is_detectable_from_exit_status(monkeypatch, capsys, state):
    monkeypatch.setattr(operator, "attach_job", lambda *args, **kwargs: {
        "run_id": "run", "status": "approved", "counts": {"pending": 1},
        "attachment": {"state": state, "lease_active": False}, "worker": None,
    })
    assert main(["jobs", "attach", "run", "--json"]) == 7
    assert payload(capsys)["attach"]["attachment"]["state"] == state


@pytest.mark.parametrize("command", ["status", "inspect", "attach", "stop", "reroll", "export"])
def test_detach_is_not_accepted_for_observation_or_reroll(command):
    args = ["jobs", command, "run", "--detach"]
    if command == "reroll":
        args += ["item", "--reason", "test"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_stop_cli_persists_reason_and_reports_bounded_observation(tmp_path, capsys):
    _, plan, approval = planned(tmp_path)
    database = tmp_path / "jobs.db"
    with SQLiteRunStore(database) as store:
        store.create_run(plan, approval, manifest_root=tmp_path, run_id="run")
    assert main(["jobs", "stop", "run", "--reason", "maintenance", "--timeout-s", "0",
                 "--store", str(database), "--json"]) == EXIT_OK
    result = payload(capsys)["stop"]
    assert result["stop_request"]["pause_generation"] == 1
    assert result["control"]["pause_requested"] is True and result["control"]["pause_reason"] == "maintenance"
    assert result["attachment"]["state"] == "not_running" and result["status"] == "approved"


def test_stop_cli_interrupt_reports_the_committed_request(monkeypatch, capsys):
    monkeypatch.setattr(operator, "stop_job", lambda *args, **kwargs: {
        "run_id": "run", "stop_request": {"pause_generation": 3, "pause_requested": True},
        "attachment": {"state": "disconnected"},
    })
    assert main(["jobs", "stop", "run", "--json"]) == EXIT_INTERRUPTED
    assert payload(capsys)["stop"]["stop_request"]["pause_generation"] == 3


def wait_until(predicate, *, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError("Owned offline worker did not reach its expected barrier")


def test_real_process_worker_survives_launcher_exit_and_retains_python_environment(tmp_path):
    manifest, _, approval = planned(tmp_path)
    database = tmp_path / "jobs.db"
    gate = tmp_path / "release-worker"
    customization = tmp_path / "customization"
    customization.mkdir()
    # Deterministic offline call barrier: the parent CLI must exit before this
    # fixture allows the actual child to complete. No provider key is used.
    (customization / "sitecustomize.py").write_text(
        "import asyncio, os\n"
        "from pathlib import Path\n"
        "from smythe.provider import OfflineProvider\n"
        "original = OfflineProvider.complete\n"
        "async def complete(self, *args, **kwargs):\n"
        "    gate = Path(os.environ['SMYTHE_OPERATOR_TEST_GATE'])\n"
        "    entered = gate.with_suffix('.entered')\n"
        "    pending = entered.with_name(entered.name + '.' + str(os.getpid()) + '.tmp')\n"
        "    pending.write_text(str(os.getpid()))\n"
        "    pending.replace(entered)\n"
        "    while not gate.exists():\n"
        "        await asyncio.sleep(.025)\n"
        "    return await original(self, *args, **kwargs)\n"
        "OfflineProvider.complete = complete\n",
        encoding="utf-8",
    )
    environment = operator._python_environment()
    environment["PYTHONPATH"] = str(customization) + os.pathsep + environment["PYTHONPATH"]
    environment["SMYTHE_OPERATOR_TEST_GATE"] = str(gate)
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(key, None)
    launched = None
    process = subprocess.Popen(
        [sys.executable, "-P", "-m", "smythe.cli", "jobs", "run", str(manifest),
         "--approve", approval.token, "--detach", "--startup-timeout-s", "20",
         "--store", str(database), "--json"],
        cwd=tmp_path, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, text=True,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
        launched = json.loads(stdout)["run"]
        assert process.poll() == 0 and launched["detached"] is True
        entered = gate.with_suffix(".entered")
        wait_until(entered.exists)
        assert int(entered.read_text()) == launched["worker_pid"]
        assert launched["worker_pid"] != process.pid
        directory = Path(launched["receipt_path"])
        ready = operator._read_json(directory / "ready.json")
        assert Path(ready["python_prefix"]) == Path(sys.prefix)
        assert not (directory / "finished.json").exists()
        stopped = operator.stop_job(database, launched["run_id"], timeout_s=0)
        assert stopped["control"]["pause_requested"] is True
        assert stopped["attachment"]["state"] == "timeout"
        assert stopped["counts"] == {"running": 1}
        assert not (directory / "finished.json").exists()  # The admitted call is still draining.
        gate.write_text("finish", encoding="utf-8")
        finished = wait_until(lambda: operator._read_json(directory / "finished.json", missing_ok=True))
        assert finished["state"] == "finished" and finished["exit_code"] == 0
        with SQLiteRunStore(database, read_only=True) as store:
            snapshot = store.snapshot(launched["run_id"])
            assert snapshot["counts"] == {"succeeded": 1}
            assert len(snapshot["artifacts"]) == 1 and store.get_run_lease(launched["run_id"]) is None
        attached = operator.attach_job(database, launched["run_id"], timeout_s=0)
        assert attached["attachment"]["state"] == "stopped"
    finally:
        gate.write_text("finish", encoding="utf-8")
        if process.poll() is None:
            # This waits only for the owned launcher; no PID-based termination.
            process.wait(timeout=35)
        if launched is not None:
            directory = Path(launched["receipt_path"])
            wait_until(lambda: (directory / "finished.json").exists(), timeout=35)
