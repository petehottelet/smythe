"""Detached Jobs workers and read-only attachment to their durable state.

The launcher creates no provider calls. A worker proves lease ownership before
the launcher grants its one-use startup decision. Receipts identify a launch;
stored PIDs are diagnostic data and are never used to signal a process.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from threading import Thread
import time
import traceback
from uuid import uuid4

from smythe.jobs.store import MAX_SQLITE_INTEGER, OperationStatus, RunStatus, RunStoreError, SQLiteRunStore


OPERATOR_VERSION = 1
MAX_RECEIPT_BYTES = 16_384
DEFAULT_STARTUP_TIMEOUT_S = 30.0
MAX_STARTUP_TIMEOUT_S = 300.0
DEFAULT_ATTACH_TIMEOUT_S = 30.0
MAX_ATTACH_TIMEOUT_S = 3600.0
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IDENTITY = re.compile(r"[0-9a-f]{32}\Z")
_FINISHED = {"completed", "partial", "failed", "budget_overrun", "needs_attention", "paused"}
_FAILED = {"partial", "failed", "budget_overrun", "needs_attention"}


class WorkerStartupError(RunStoreError):
    """Launch failed; metadata identifies any already-visible authorization."""

    def __init__(self, message: str, launch: dict) -> None:
        self.message = message
        self.launch = dict(launch)
        super().__init__(f"{message}{_authorization_note(launch)}; run {launch['run_id']}; "
                         f"worker log: {launch['log_path']}")


class WorkerStartupInterrupted(KeyboardInterrupt):
    """Interrupted launch with the durable run and private log still locatable."""

    def __init__(self, launch: dict) -> None:
        self.launch = dict(launch)
        super().__init__(f"Worker startup interrupted{_authorization_note(launch)}; run {launch['run_id']}; "
                         f"worker log: {launch['log_path']}")


def _authorization_note(launch):
    if launch.get("startup_authorized") is True:
        return "; startup was authorized and the worker may already be running"
    if "startup_authorized" in launch and launch["startup_authorized"] is None:
        return "; startup authorization is unknown and the worker may already be running"
    return ""


class _StartupAborted(RunStoreError):
    pass


def _seconds(value, name, *, minimum, maximum):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum} seconds")
    return float(value)


def _run_id(value):
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must be a safe 1-128 character Jobs identifier")
    return value


def validate_startup_timeout(value) -> float:
    """Validate CLI launch options before creating a durable run."""
    return _seconds(value, "startup_timeout_s", minimum=.1, maximum=MAX_STARTUP_TIMEOUT_S)


def _identity(value, name):
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise RunStoreError(f"Invalid worker {name}")
    return value


def _root(store_path: Path, run_id: str) -> Path:
    # SQLite IDs are case-sensitive; filesystem names may be case-insensitive,
    # strip trailing dots, or reserve DOS device names. Keep exact identity in
    # receipts while using a fixed portable component for its private files.
    component = hashlib.sha256(_run_id(run_id).encode("utf-8")).hexdigest()
    return store_path.parent / f".{store_path.name}.workers" / component


def _reject_link(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RunStoreError("Worker files must not use symlinks or directory junctions")


def _private_windows_directory(path: Path) -> None:
    # Python 3.11/3.12 do not implement POSIX mode=0700 as a Windows ACL.
    # Set an inheritable, protected DACL for the object owner and SYSTEM.
    import ctypes
    from ctypes import wintypes

    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(wintypes.DWORD)]
    convert.restype = wintypes.BOOL
    apply = security.SetFileSecurityW
    apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    apply.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    if not convert("D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not apply(str(path), 0x80000004, descriptor):  # protected DACL
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def _private_directory(path: Path, *, exist_ok=False) -> None:
    path.mkdir(mode=0o700, exist_ok=exist_ok)
    _reject_link(path)
    if not path.is_dir():
        raise RunStoreError("Worker launch path is not a directory")
    if os.name == "nt":
        _private_windows_directory(path)
    else:
        path.chmod(0o700)


def _json_bytes(value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > MAX_RECEIPT_BYTES:
        raise RunStoreError("Worker receipt is too large")
    return data


def _exclusive_json(path: Path, value: dict) -> None:
    data = _json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _exclusive_json(temporary, value)
        # A Windows reader can briefly hold the old pointer open. Receipts use
        # distinct ready/finished files, so only the latest-launch pointer is
        # normally replaced while readers are active.
        deadline = time.monotonic() + 1
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, missing_ok=False):
    try:
        _reject_link(path)
        with path.open("rb") as stream:
            data = stream.read(MAX_RECEIPT_BYTES + 1)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if len(data) > MAX_RECEIPT_BYTES:
        raise RunStoreError("Worker receipt is too large")
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = item
        return result

    def constant(_):
        raise ValueError("nonfinite JSON value")

    def floating(text):
        number = float(text)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    def integer(text):
        number = int(text)
        if abs(number) > MAX_SQLITE_INTEGER:
            raise ValueError("JSON integer is out of bounds")
        return number

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant,
                           parse_float=floating, parse_int=integer)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise RunStoreError("Invalid worker receipt JSON") from error
    if type(value) is not dict:
        raise RunStoreError("Worker receipt must be an object")
    return value


def _bound_receipt(value: dict, request: dict) -> dict:
    if (type(value.get("version")) is not int or value["version"] != OPERATOR_VERSION
            or value.get("run_id") != request["run_id"]
            or value.get("launch_id") != request["launch_id"]
            or type(value.get("nonce")) is not str
            or _IDENTITY.fullmatch(value["nonce"]) is None
            or not hmac.compare_digest(value["nonce"], request["nonce"])):
        raise RunStoreError("Worker receipt does not match this launch")
    if not _integer(value.get("recorded_at_ns"), minimum=1):
        raise RunStoreError("Worker receipt timestamp is invalid")
    return value


def _integer(value, *, minimum=0, maximum=MAX_SQLITE_INTEGER):
    return type(value) is int and minimum <= value <= maximum


def _text(value, *, minimum=1, maximum=4096):
    return type(value) is str and minimum <= len(value) <= maximum


def _validated_receipt(value: dict, request: dict, kind: str) -> dict:
    """Validate the complete public receipt before exposing any saved fields."""
    value = _bound_receipt(value, request)
    expected = {"version", "run_id", "launch_id", "nonce", "recorded_at_ns", "state"}
    state = value.get("state")
    if type(state) is not str:
        raise RunStoreError(f"Invalid worker {kind} receipt state")
    valid = False
    if kind == "ready":
        expected |= {"pid", "lease_epoch", "lease_owner_id", "python_executable", "python_prefix"}
        valid = (state == "ready" and _integer(value.get("pid"), minimum=1)
                 and _integer(value.get("lease_epoch"), minimum=1)
                 and _text(value.get("lease_owner_id"), maximum=256)
                 and _text(value.get("python_executable")) and _text(value.get("python_prefix")))
    elif kind == "launcher-failure":
        expected |= {"error_type", "error", "startup_authorized"}
        valid = state == "failed" and (value.get("startup_authorized") is None
                                      or type(value.get("startup_authorized")) is bool)
    elif kind == "finished":
        expected |= {"pid", "exit_code"}
        valid = (_integer(value.get("pid"), minimum=1)
                 and _integer(value.get("exit_code"), maximum=255))
        if state == "finished":
            expected |= {"run_status", "counts"}
            counts = value.get("counts")
            valid = (valid and type(value.get("run_status")) is str
                     and value["run_status"] in {item.value for item in RunStatus}
                     and type(counts) is dict
                     and all(key in {item.value for item in OperationStatus} and _integer(count)
                             for key, count in counts.items()))
        else:
            expected |= {"error_type", "error"}
            valid = valid and state in {"failed", "aborted"}
    if "error" in expected:
        valid = (valid and _text(value.get("error_type"), maximum=256)
                 and _text(value.get("error"), minimum=0, maximum=1000))
    if not valid or value.keys() != expected:
        raise RunStoreError(f"Invalid worker {kind} receipt")
    return value


def _record(request, **fields):
    return {"version": OPERATOR_VERSION, "run_id": request["run_id"],
            "launch_id": request["launch_id"], "nonce": request["nonce"],
            "recorded_at_ns": time.time_ns(), **fields}


def _decision(directory: Path, request: dict, action: str) -> bool:
    try:
        _exclusive_json(directory / "decision.json", _record(request, action=action))
        return True
    except FileExistsError:
        return False  # A start decision is never replaced by a later abort.


def _authorization_after_error(directory, request, *, attempted):
    if not attempted:
        return False
    try:
        decision = _bound_receipt(_read_json(directory / "decision.json"), request)
        if type(decision.get("action")) is str and decision["action"] in {"start", "abort"}:
            return decision["action"] == "start"
    except (OSError, RunStoreError):
        pass
    # A failed write/fsync/read can leave a complete authorization that the
    # child already read. Failure to prove it is not proof it never existed.
    return None


def _process_options():
    if os.name == "nt":
        # A terminal/job object's lifetime must not own the actual worker.
        # If its job prohibits breakaway, Popen fails before any child starts.
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        return {"creationflags": (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_BREAKAWAY_FROM_JOB),
                "startupinfo": startup}
    return {"start_new_session": True}


def _python_environment():
    environment = os.environ.copy()
    # Keep this installation/venv's import context without letting -m prepend
    # an unrelated manifest directory. Do not persist credentials in receipts.
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(str(Path(item).resolve()) for item in sys.path if type(item) is str and item)
    )
    return environment


def _public_launch(request: dict, directory: Path, **fields) -> dict:
    return {"run_id": request["run_id"], "launch_id": request["launch_id"],
            "store_path": request["store_path"], "log_path": str(directory / "worker.log"),
            "receipt_path": str(directory), **fields}


def launch_worker(store_path, run_id, *, startup_timeout_s=DEFAULT_STARTUP_TIMEOUT_S,
                  poll_interval_s=.05, clear_pause=False, pause_generation=None) -> dict:
    """Launch an already-created, approved run and wait for its lease receipt.

    The returned PID is reported by the Python worker itself; it can differ
    from Popen.pid for a Windows venv redirector. No process is killed here.
    Failures before authorization deny dispatch. Errors at the authorization
    boundary report whether a grant is visible or its state remains unknown.
    """
    timeout = validate_startup_timeout(startup_timeout_s)
    interval = _seconds(poll_interval_s, "poll_interval_s", minimum=.01, maximum=1)
    if type(clear_pause) is not bool:
        raise ValueError("clear_pause must be a boolean")
    if pause_generation is not None and (
        type(pause_generation) is not int or not 0 <= pause_generation <= MAX_SQLITE_INTEGER
    ):
        raise ValueError("pause_generation must be a nonnegative integer")
    if not clear_pause and pause_generation is not None:
        raise ValueError("pause_generation requires explicit resume intent")
    database, run_id = Path(store_path).resolve(), _run_id(run_id)
    with SQLiteRunStore(database, read_only=True) as store:
        store.get_run(run_id)
        if clear_pause and pause_generation is None:
            pause_generation = store.get_control(run_id)["pause_generation"]
            if not _integer(pause_generation):
                raise RunStoreError("Saved pause generation is invalid")
    if not sys.executable or not Path(sys.executable).is_file():
        raise RunStoreError("Detached Jobs requires the current Python interpreter")
    root = _root(database, run_id)
    launch_id = uuid4().hex
    directory = root / launch_id
    request = {"version": OPERATOR_VERSION, "run_id": run_id, "launch_id": launch_id,
               "nonce": uuid4().hex, "store_path": str(database),
               "startup_timeout_s": timeout,
               "clear_pause": clear_pause, "pause_generation": pause_generation,
               "expires_at_ns": time.time_ns() + int(timeout * 1e9)}
    launch = _public_launch(request, directory)
    process = None
    created_directory = False
    authorization_attempted = False
    deadline = time.monotonic() + timeout
    try:
        _private_directory(root.parent, exist_ok=True)
        _private_directory(root, exist_ok=True)
        _private_directory(directory)
        created_directory = True
        _exclusive_json(directory / "request.json", request)
        _atomic_json(root / "latest.json", {"version": OPERATOR_VERSION, "run_id": run_id,
                                           "launch_id": launch_id})
        descriptor = os.open(directory / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(descriptor, "wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-P", "-u", "-m", "smythe.jobs.operator", "--worker",
                 str(directory / "request.json")], cwd=str(database.parent),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                shell=False, close_fds=True, env=_python_environment(), **_process_options(),
            )
        while True:
            failed = _read_json(directory / "finished.json", missing_ok=True)
            if failed is not None:
                failed = _validated_receipt(failed, request, "finished")
                raise WorkerStartupError(f"Worker startup failed: {failed.get('error', failed['state'])}",
                                         dict(launch, worker={key: value for key, value in failed.items()
                                                              if key != "nonce"}))
            ready = _read_json(directory / "ready.json", missing_ok=True)
            if ready is not None:
                ready = _validated_receipt(ready, request, "ready")
                with SQLiteRunStore(database, read_only=True) as store:
                    lease = store.get_run_lease(run_id)
                if (lease is None or lease.owner_id != ready["lease_owner_id"]
                        or lease.epoch != ready["lease_epoch"] or lease.expires_at_ns <= time.time_ns()):
                    raise RunStoreError("Worker readiness lease is no longer current")
                if time.monotonic() >= deadline or time.time_ns() >= request["expires_at_ns"]:
                    raise WorkerStartupError("Worker startup deadline expired", launch)
                authorization_attempted = True
                if not _decision(directory, request, "start"):
                    raise RunStoreError("Worker startup already has a decision")
                if time.monotonic() >= deadline or time.time_ns() >= request["expires_at_ns"]:
                    raise WorkerStartupError("Worker startup deadline expired while recording authorization", launch)
                return dict(launch, detached=True, status="started", startup_authorized=True,
                            worker_pid=ready["pid"],
                            lease_epoch=ready["lease_epoch"])
            returncode = process.poll()
            if returncode is not None:
                raise WorkerStartupError(f"Worker exited before readiness (exit {returncode})", launch)
            if time.monotonic() >= deadline:
                raise WorkerStartupError("Worker startup deadline expired", launch)
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    except BaseException as error:
        authorized = _authorization_after_error(directory, request, attempted=authorization_attempted)
        launch = dict(launch, startup_authorized=authorized)
        if created_directory:
            try:
                _decision(directory, request, "abort")
                _atomic_json(directory / "launcher-failure.json", _record(
                    request, state="failed", error_type=type(error).__name__, error=str(error)[:1000],
                    startup_authorized=authorized,
                ))
            except (OSError, RunStoreError):
                # No start decision means the child still expires closed if
                # a disk failure also prevents writing the abort evidence.
                pass
        if isinstance(error, KeyboardInterrupt):
            raise WorkerStartupInterrupted(launch) from error
        if isinstance(error, WorkerStartupError):
            error.launch["startup_authorized"] = authorized
            if authorized is not False:
                raise WorkerStartupError(error.message, error.launch) from error
            raise
        if isinstance(error, SystemExit):
            raise
        raise WorkerStartupError(str(error), launch) from error
    finally:
        if process is not None:
            # Reap only our own child handle in library use. The daemon never
            # keeps an exiting CLI alive or cancels its detached worker.
            Thread(target=process.wait, name="smythe-worker-reaper", daemon=True).start()


def _load_request(path: Path) -> tuple[dict, Path]:
    value = _read_json(path)
    expected = {"version", "run_id", "launch_id", "nonce", "store_path", "startup_timeout_s",
                "expires_at_ns", "clear_pause", "pause_generation"}
    if (value.keys() != expected or type(value["version"]) is not int
            or value["version"] != OPERATOR_VERSION or type(value["store_path"]) is not str
            or not _integer(value["expires_at_ns"], minimum=1)
            or type(value["clear_pause"]) is not bool
            or (value["pause_generation"] is not None and (
                type(value["pause_generation"]) is not int
                or not 0 <= value["pause_generation"] <= MAX_SQLITE_INTEGER))
            or (value["clear_pause"] and value["pause_generation"] is None)
            or (not value["clear_pause"] and value["pause_generation"] is not None)):
        raise RunStoreError("Invalid worker launch request")
    _run_id(value["run_id"])
    _identity(value["launch_id"], "launch_id")
    _identity(value["nonce"], "nonce")
    _seconds(value["startup_timeout_s"], "startup_timeout_s", minimum=.1,
             maximum=MAX_STARTUP_TIMEOUT_S)
    database = Path(value["store_path"]).resolve()
    directory = _root(database, value["run_id"]) / value["launch_id"]
    if path.absolute() != directory / "request.json":
        raise RunStoreError("Worker request is outside its private launch directory")
    for candidate in (directory.parent.parent, directory.parent, directory):
        _reject_link(candidate)
    return value, directory


async def _wait_for_start(directory, request):
    deadline = time.monotonic() + request["startup_timeout_s"]
    while time.monotonic() < deadline and time.time_ns() < request["expires_at_ns"]:
        try:
            decision = _read_json(directory / "decision.json", missing_ok=True)
        except RunStoreError:
            # The exclusive decision file is small, but its creator can be
            # interrupted mid-write. A partial decision can never authorize.
            decision = None
        if decision is not None:
            decision = _bound_receipt(decision, request)
            if decision.get("action") == "start":
                if time.monotonic() < deadline and time.time_ns() < request["expires_at_ns"]:
                    return
                raise _StartupAborted("Worker startup expired while reading its start decision")
            raise _StartupAborted("Launcher aborted worker startup")
        await asyncio.sleep(.05)
    raise _StartupAborted("Worker startup expired before a start decision")


async def _run_worker(request, directory):
    from smythe.jobs.runner import JobRunner

    class DetachedRunner(JobRunner):
        async def _execute(self, run_id, plan, root, *, lease):
            if time.time_ns() >= request["expires_at_ns"]:
                raise _StartupAborted("Worker startup expired before readiness")
            _atomic_json(directory / "ready.json", _record(
                request, state="ready", pid=os.getpid(), lease_owner_id=lease.owner_id,
                lease_epoch=lease.epoch, python_executable=sys.executable, python_prefix=sys.prefix,
            ))
            await _wait_for_start(directory, request)
            return await super()._execute(run_id, plan, root, lease=lease)

    with SQLiteRunStore(request["store_path"]) as store:
        runner = DetachedRunner(store)
        return await runner.resume(request["run_id"], clear_pause=request["clear_pause"],
                                   pause_generation=request["pause_generation"])


def worker_main(request_path) -> int:
    """Internal module entry point; public callers use launch_worker()."""
    request, directory = _load_request(Path(request_path).absolute())
    # An accidental second invocation must not overwrite the first worker's
    # readiness/final result or dispatch with the same launch identity.
    _exclusive_json(directory / "claimed.json", _record(request, pid=os.getpid()))
    try:
        if time.time_ns() >= request["expires_at_ns"]:
            raise _StartupAborted("Worker startup request expired")
        decision = _read_json(directory / "decision.json", missing_ok=True)
        if decision is not None and _bound_receipt(decision, request).get("action") == "abort":
            raise _StartupAborted("Launcher already aborted worker startup")
        result = asyncio.run(_run_worker(request, directory))
        code = int(result["status"] in _FAILED)
        _atomic_json(directory / "finished.json", _record(
            request, state="finished", pid=os.getpid(), exit_code=code,
            run_status=result["status"], counts=result["counts"],
        ))
        return code
    except BaseException as error:
        _atomic_json(directory / "finished.json", _record(
            request, state="aborted" if isinstance(error, _StartupAborted) else "failed",
            pid=os.getpid(), exit_code=7, error_type=type(error).__name__, error=str(error)[:1000],
        ))
        traceback.print_exc()
        return 7


def read_worker_state(store_path, run_id) -> dict | None:
    """Read the latest launch, which may differ from the current lease's worker."""
    database, run_id = Path(store_path).resolve(), _run_id(run_id)
    root = _root(database, run_id)
    for candidate in (root.parent, root):
        if not candidate.exists():
            return None
        _reject_link(candidate)
    pointer = _read_json(root / "latest.json", missing_ok=True)
    if pointer is None:
        return None
    if (type(pointer.get("version")) is not int or pointer["version"] != OPERATOR_VERSION
            or pointer.get("run_id") != run_id):
        raise RunStoreError("Invalid latest worker pointer")
    launch_id = _identity(pointer.get("launch_id"), "launch_id")
    request, directory = _load_request(root / launch_id / "request.json")
    for kind in ("finished", "launcher-failure", "ready"):
        receipt = _read_json(directory / f"{kind}.json", missing_ok=True)
        if receipt is not None:
            receipt = _validated_receipt(receipt, request, kind)
            break
    else:
        state = "startup_expired" if time.time_ns() >= request["expires_at_ns"] else "starting"
        return _public_launch(request, directory, state=state)
    return _public_launch(request, directory, **{
        key: value for key, value in receipt.items()
        if key not in {"version", "run_id", "launch_id", "nonce"}
    })


def attach_job(store_path, run_id, *, timeout_s=DEFAULT_ATTACH_TIMEOUT_S,
               poll_interval_s=1.0, on_update=None) -> dict:
    """Observe a job until it stops or the bounded attachment time expires.

    Closing or interrupting this reader does not stop, pause, recover, or
    construct providers for the worker. A PID is never treated as a signal
    target or proof that an expired lease's worker is still alive.
    """
    timeout = _seconds(timeout_s, "timeout_s", minimum=0, maximum=MAX_ATTACH_TIMEOUT_S)
    interval = _seconds(poll_interval_s, "poll_interval_s", minimum=.05, maximum=60)
    database, run_id = Path(store_path).resolve(), _run_id(run_id)
    deadline = time.monotonic() + timeout
    previous = None
    with SQLiteRunStore(database, read_only=True) as store:
        while True:
            snapshot = store.inspection_snapshot(run_id, limit=50, events_limit=20)
            lease = snapshot["lease"]
            worker = read_worker_state(database, run_id)
            live = lease is not None and lease["expires_at_ns"] > snapshot["observed_at_ns"]
            launch_matches = None
            if worker is not None and "lease_owner_id" in worker:
                launch_matches = bool(live and lease["owner_id"] == worker["lease_owner_id"]
                                      and lease["epoch"] == worker["lease_epoch"])
            reason = None
            if not live:
                if snapshot["status"] in _FINISHED:
                    reason = "stopped"
                elif worker is not None and worker["state"] in {"failed", "aborted", "startup_expired"}:
                    reason = "worker_failed"
                elif snapshot["status"] == "running":
                    reason = "lease_expired"
                elif worker is None or worker["state"] != "starting":
                    reason = "not_running"
            if reason is None and time.monotonic() >= deadline:
                reason = "timeout"
            snapshot = dict(snapshot, worker=worker,
                            attachment={"state": reason or "following", "lease_active": live,
                                        "latest_launch_matches_lease": launch_matches})
            encoded = json.dumps({key: value for key, value in snapshot.items()
                                  if key != "observed_at_ns"}, sort_keys=True)
            if on_update is not None and encoded != previous:
                on_update(snapshot)
            previous = encoded
            if reason is not None:
                return snapshot
            time.sleep(min(interval, max(0, deadline - time.monotonic())))


def stop_job(store_path, run_id, *, reason="operator requested pause",
             timeout_s=DEFAULT_ATTACH_TIMEOUT_S, poll_interval_s=1.0, on_update=None) -> dict:
    """Persist a graceful pause request, then observe its bounded drain.

    Already admitted calls may finish. An observation timeout or interruption
    leaves the request in SQLite; neither event cancels or signals a worker.
    """
    timeout = _seconds(timeout_s, "timeout_s", minimum=0, maximum=MAX_ATTACH_TIMEOUT_S)
    interval = _seconds(poll_interval_s, "poll_interval_s", minimum=.05, maximum=60)
    if type(reason) is not str or not reason.strip() or len(reason) > 2048:
        raise ValueError("stop reason must be nonempty text of at most 2048 characters")
    database, run_id = Path(store_path).resolve(), _run_id(run_id)
    # A typo must not create a new empty database. The writable open performs
    # any supported schema migration only after confirming the named run.
    with SQLiteRunStore(database, read_only=True) as store:
        store.get_run(run_id)
    with SQLiteRunStore(database) as store:
        requested = store.request_pause(run_id, reason=reason)
    try:
        snapshot = attach_job(database, run_id, timeout_s=timeout, poll_interval_s=interval,
                              on_update=on_update)
    except KeyboardInterrupt:
        return {"run_id": run_id, "stop_request": requested,
                "attachment": {"state": "disconnected"}}
    return dict(snapshot, stop_request=requested)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m smythe.jobs.operator")
    parser.add_argument("--worker", required=True, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    return worker_main(arguments.worker)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
