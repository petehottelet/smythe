"""Shared deterministic fakes: SDK transports, clocks, crashes and child processes.

Import this module by its own name (``from fakes import NativeTransport``), as
with ``helpers``. Importing it as ``tests.fakes`` as well would load a second
copy whose classes differ. It imports only the standard library and smythe;
the provider SDKs are never imported, so it works without the optional extras.

Durable runs accept only the exact provider types ``OpenAIResponsesProvider``,
``AnthropicMessagesProvider`` and a stateless ``OfflineProvider``:
``describe_provider`` compares ``type(provider)``. A ``Provider`` subclass, or
an ``OfflineProvider`` given scripted responses, is rejected before any call.
Durable fakes therefore replace the SDK client returned by ``_get_client``.
Request preparation, quoting, pricing, decoding and journaling stay real; only
the wire is fake. ``NativeTransport`` covers Responses and ``MessagesTransport``
covers Messages; both log every call with its request identity and accept a
failure schedule keyed by phase and ordinal.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from contextlib import closing
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock

from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT
from smythe.synthesizer import MERGE_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
NS_PER_SECOND = 1_000_000_000


def _nanoseconds(seconds: int | float) -> int:
    if type(seconds) not in (int, float):
        raise TypeError("Seconds must be an int or float")
    return seconds * NS_PER_SECOND if type(seconds) is int else round(seconds * NS_PER_SECOND)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class ManualClock:
    """A wall and monotonic clock that moves only when the test moves it.

    Calling the clock returns wall nanoseconds, so an instance can be passed
    as ``clock_ns=`` to ``SQLiteWorkflowStore`` and ``SQLiteRunStore``, or
    patched over ``smythe.optimize.ledger._lease_now_ns``. ``monotonic()`` and
    ``monotonic_ns()`` advance with ``advance()`` but ignore ``jump_wall()``.
    """

    def __init__(self, start_ns: int = 1_000_000_000_000, *, monotonic_ns: int = 0) -> None:
        if type(start_ns) is not int or type(monotonic_ns) is not int:
            raise TypeError("Clock readings are integer nanoseconds")
        self.now_ns = start_ns
        self._monotonic_ns = monotonic_ns

    @classmethod
    def after(cls, seconds: int | float) -> ManualClock:
        """A clock frozen ``seconds`` after the real time, e.g. past a lost lease."""
        return cls(time.time_ns() + _nanoseconds(seconds))

    def __call__(self) -> int:
        return self.now_ns

    def time_ns(self) -> int:
        return self.now_ns

    def monotonic_ns(self) -> int:
        return self._monotonic_ns

    def monotonic(self) -> float:
        return self._monotonic_ns / NS_PER_SECOND

    def advance(self, seconds: int | float) -> None:
        """Move both clocks forward by the same amount."""
        delta = _nanoseconds(seconds)
        if delta < 0:
            raise ValueError("A clock cannot advance backwards; use jump_wall")
        self.now_ns += delta
        self._monotonic_ns += delta

    def advance_past(self, deadline_ns: int) -> None:
        """Move both clocks to one nanosecond after a wall deadline."""
        if deadline_ns >= self.now_ns:
            delta = deadline_ns + 1 - self.now_ns
            self.now_ns += delta
            self._monotonic_ns += delta

    def expire_lease(self, ttl_s: int | float) -> None:
        """Advance just past a lease of ``ttl_s`` seconds taken at this reading."""
        self.advance_past(self.now_ns + _nanoseconds(ttl_s))

    def jump_wall(self, seconds: int | float) -> None:
        """Step only the wall clock, as a system time adjustment does."""
        self.now_ns += _nanoseconds(seconds)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class Gate:
    """Holds a fake call at one point until the test releases it, without sleeping.

    ``entered`` is set once a call reaches the gate. Coroutines wait with
    ``await gate.wait()`` and threads with ``gate.block()``. ``release()`` may
    be called from any thread or event loop and frees current and later
    callers. Every wait is bounded so a forgotten release cannot hang a test.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self._released = threading.Event()
        self._lock = threading.Lock()
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    @property
    def released(self) -> bool:
        return self._released.is_set()

    async def wait(self, timeout: float = 30.0) -> None:
        event = asyncio.Event()
        with self._lock:
            if self._released.is_set():
                event.set()
            else:
                self._waiters.append((asyncio.get_running_loop(), event))
        self.entered.set()
        await asyncio.wait_for(event.wait(), timeout)

    def block(self, timeout: float = 30.0) -> None:
        self.entered.set()
        if not self._released.wait(timeout):
            raise TimeoutError("Gate was not released")

    def release(self) -> None:
        with self._lock:
            self._released.set()
            waiters, self._waiters = self._waiters, []
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # The waiting loop has already closed.
                pass

    async def reached(self, timeout: float = 10.0) -> bool:
        """Wait, without blocking the event loop, until a call reaches the gate."""
        return await asyncio.to_thread(self.entered.wait, timeout)


# ---------------------------------------------------------------------------
# SDK transport fakes
# ---------------------------------------------------------------------------


class APIStatusError(Exception):
    """Mimics an SDK status error, which carries the raw HTTP response."""

    def __init__(self, status: int, body: dict, headers: dict) -> None:
        super().__init__(f"Error code: {status}")
        self.response = SimpleNamespace(content=json.dumps(body).encode(), status_code=status,
                                        headers=headers)


# The journal records the SDK class name, and both SDKs name one per status.
_STATUS_ERRORS = {
    status: type(name, (APIStatusError,), {"__doc__": f"The SDK's class for HTTP {status}."})
    for status, name in {
        400: "BadRequestError", 401: "AuthenticationError", 403: "PermissionDeniedError",
        404: "NotFoundError", 409: "ConflictError", 413: "RequestTooLargeError",
        422: "UnprocessableEntityError", 429: "RateLimitError", 500: "InternalServerError",
        529: "OverloadedError",
    }.items()
}


@dataclass(frozen=True)
class HttpError:
    """A scheduled HTTP status error from the provider."""

    status: int
    error_type: str
    message: str = "Provider error"


RATE_LIMITED = HttpError(429, "rate_limit_error", "Rate limit reached")
SERVER_ERROR = HttpError(500, "api_error")
# The request never reaches the provider; no response exists.
TRANSPORT_ERROR = "transport_error"
# The provider produces and bills the response, but the caller never receives it.
LOST_RESPONSE = "lost_response"
# The response reports no usage, so its charge is unknown.
MALFORMED_USAGE = "malformed_usage"
# The provider repeats the previous response of the same operation verbatim.
DUPLICATE = "duplicate"
# The input-token count is not a valid integer.
INVALID_COUNT = "invalid_count"
_FAULTS = frozenset({TRANSPORT_ERROR, LOST_RESPONSE, MALFORMED_USAGE, DUPLICATE, INVALID_COUNT})


@dataclass(frozen=True)
class WireRecord:
    """One client call as the fake provider saw it."""

    operation: str  # "response" for generation or "input_tokens" for counting
    phase: str
    ordinal: int  # 0-based position among calls with this operation and phase
    request_id: str | None
    response_id: str | None
    outcome: str  # delivered, duplicate, http_<status>, lost_response, not_delivered, invalid_count
    payload: dict = field(repr=False)


def phase_of(system: str) -> str:
    """Name the workflow phase a system prompt belongs to."""
    if system == PLANNING_SYSTEM_PROMPT:
        return "planning"
    if system.startswith("You are a task classifier."):
        return "routing"
    if system == SUPERVISOR_SYSTEM_PROMPT:
        return "supervision"
    if system == MERGE_SYSTEM_PROMPT:
        return "synthesis"
    return "execution"


DEFAULT_PLAN = {"topology": ["serial"], "nodes": [
    {"id": "draft", "label": "Draft"},
    {"id": "judge", "label": "Check", "depends_on": ["draft"],
     "verifies": "draft", "max_regenerations": 1},
]}


class _Transport:
    """Shared request log, scripted replies and failure schedule."""

    provider_class: type
    request_id_header: str

    def __init__(self, *, plan: dict | None = None) -> None:
        self.plan = DEFAULT_PLAN if plan is None else plan
        self.requests: list[dict] = []  # generation payloads that reached the provider
        self.counts: list[dict] = []  # count payloads that reached the provider
        self.log: list[WireRecord] = []
        self.clients: list[SimpleNamespace] = []
        self.overrides: dict[str, dict] = {}  # phase -> fields merged into the response body
        self.replies: dict[str, Callable[[str], str]] = {
            "planning": lambda prompt: json.dumps(self.plan),
            "routing": lambda prompt: "autonomous",
            "supervision": lambda prompt: '{"change":false,"reason":"Plan remains complete"}',
            "synthesis": lambda prompt: "Final accepted output.",
            "execution": lambda prompt: (
                "PASS" if "Your step: Check" in prompt else "Draft accepted output."),
        }
        self._scripts: dict[str, deque[str]] = {}
        self._schedule: dict[tuple[str, str, int | None], tuple] = {}
        self._ordinals: Counter[tuple[str, str]] = Counter()
        self._last_body: dict[str, tuple[bytes, str]] = {}
        self._sequence = 0

    def install(self, monkeypatch):
        """Serve this transport to every provider of the class; return self."""
        transport = self

        def get_client(provider):
            # A session-bound provider keeps its own client, as the SDK does.
            if provider._client is not None:
                return provider._client
            client = transport._client()
            transport.clients.append(client)
            return client

        monkeypatch.setattr(self.provider_class, "_get_client", get_client)
        return self

    def script(self, phase: str, *texts: str) -> None:
        """Reply to the next calls of ``phase`` with these texts, in order."""
        self._scripts.setdefault(phase, deque()).extend(texts)

    def schedule(self, phase: str, ordinal: int | None, *steps,
                 operation: str = "response") -> None:
        """Apply ``steps`` to one call of ``phase``, or to every call when ordinal is None.

        Steps are ``Gate`` instances, which hold the call after the provider
        receives it, and at most one fault: an ``HttpError`` or one of the
        fault names defined in this module.
        """
        if operation not in {"response", "input_tokens"}:
            raise ValueError("Operation must be 'response' or 'input_tokens'")
        faults = [step for step in steps if not isinstance(step, Gate)]
        if len(faults) > 1 or any(not isinstance(step, HttpError) and step not in _FAULTS
                                  for step in faults):
            raise ValueError("Schedule gates and at most one known fault")
        self._schedule[operation, phase, ordinal] = steps

    def clear(self, phase: str, ordinal: int | None, *, operation: str = "response") -> None:
        self._schedule.pop((operation, phase, ordinal), None)

    def _steps(self, operation, phase, ordinal):
        steps = self._schedule.get((operation, phase, ordinal), self._schedule.get(
            (operation, phase, None), ()))
        gates = [step for step in steps if isinstance(step, Gate)]
        faults = [step for step in steps if not isinstance(step, Gate)]
        return gates, faults[0] if faults else None

    def _identity(self, operation):
        self._sequence += 1
        kind = "count" if operation == "input_tokens" else "generation"
        return self._sequence, f"req_{kind}_{self._sequence}"

    async def _call(self, operation: str, payload: dict):
        phase = phase_of(self._system(payload))
        ordinal = self._ordinals[operation, phase]
        self._ordinals[operation, phase] += 1
        gates, fault = self._steps(operation, phase, ordinal)
        sequence, request_id = self._identity(operation)

        def record(outcome, request_id=None, response_id=None):
            self.log.append(WireRecord(operation, phase, ordinal, request_id, response_id,
                                       outcome, payload))

        if fault == TRANSPORT_ERROR:
            record("not_delivered")
            raise ConnectionError("Connection failed before the request was sent")
        (self.counts if operation == "input_tokens" else self.requests).append(payload)
        for gate in gates:
            await gate.wait()
        if isinstance(fault, HttpError):
            record(f"http_{fault.status}", request_id)
            error = _STATUS_ERRORS.get(fault.status, APIStatusError)
            raise error(fault.status, self._error_body(fault, request_id),
                        {self.request_id_header: request_id})
        outcome = "delivered"
        if fault == DUPLICATE and operation in self._last_body:
            body, request_id = self._last_body[operation]
            outcome = "duplicate"
        elif operation == "input_tokens":
            invalid = fault == INVALID_COUNT
            body = json.dumps({"input_tokens": False if invalid else self.input_tokens}).encode()
            outcome = "invalid_count" if invalid else outcome
        else:
            raw = self._response(sequence, payload, self._text(phase, payload))
            if fault == MALFORMED_USAGE:
                raw["usage"] = None
            raw.update(self.overrides.get(phase, {}))
            body = json.dumps(raw).encode()
        self._last_body[operation] = body, request_id
        response_id = json.loads(body).get("id") if operation == "response" else None
        if fault == LOST_RESPONSE:
            record("lost_response", request_id, response_id)
            raise ConnectionError("Connection lost before the response arrived")
        record(outcome, request_id, response_id)
        return self._wire(body, request_id)

    def _text(self, phase, payload):
        queued = self._scripts.get(phase)
        if queued:
            return queued.popleft()
        return self.replies[phase](self._prompt(payload))


class NativeTransport(_Transport):
    """Stateful OpenAI Responses wire fake for ``OpenAIResponsesProvider``.

    Each generation bills 100 input and 10 output tokens: 1,500,000 nanoUSD
    on ``gpt-6-astra``. Request IDs arrive in ``x-request-id``.
    """

    provider_class = OpenAIResponsesProvider
    request_id_header = "x-request-id"
    input_tokens = 100

    def _client(self):
        return SimpleNamespace(
            base_url="https://api.openai.com/v1/", max_retries=0, close=AsyncMock(),
            responses=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=AsyncMock(side_effect=self._generate)),
                input_tokens=SimpleNamespace(
                    with_raw_response=SimpleNamespace(count=AsyncMock(side_effect=self._count))),
            ),
        )

    async def _generate(self, **payload):
        return await self._call("response", payload)

    async def _count(self, **payload):
        return await self._call("input_tokens", payload)

    @staticmethod
    def _system(payload):
        return payload["instructions"]

    @staticmethod
    def _prompt(payload):
        return payload["input"][-1]["content"]

    def _wire(self, body, request_id):
        return SimpleNamespace(content=body, status_code=200,
                               headers={self.request_id_header: request_id})

    @staticmethod
    def _error_body(fault, request_id):
        return {"error": {"message": fault.message, "type": fault.error_type, "param": None,
                          "code": fault.error_type}}

    @staticmethod
    def _response(sequence, payload, text):
        return {
            "id": f"resp_{sequence}", "model": payload["model"], "status": "completed",
            "service_tier": "default", "usage": {
                "input_tokens": 100, "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            }, "output": [{"type": "message", "status": "completed", "role": "assistant",
                           "content": [{"type": "output_text", "text": text}]}],
        }


class MessagesTransport(_Transport):
    """Stateful Anthropic Messages wire fake for ``AnthropicMessagesProvider``.

    Each generation bills 2,535,000 nanoUSD on the supported model, including
    cache reads and writes. Request IDs arrive in ``request-id``.
    """

    provider_class = AnthropicMessagesProvider
    request_id_header = "request-id"
    input_tokens = 170

    def _client(self):
        return SimpleNamespace(
            base_url="https://api.anthropic.com/", max_retries=0, close=AsyncMock(),
            messages=SimpleNamespace(with_raw_response=SimpleNamespace(
                create=AsyncMock(side_effect=self._generate),
                count_tokens=AsyncMock(side_effect=self._count))),
        )

    async def _generate(self, **payload):
        return await self._call("response", payload)

    async def _count(self, **payload):
        return await self._call("input_tokens", payload)

    @staticmethod
    def _system(payload):
        return payload["system"]

    @staticmethod
    def _prompt(payload):
        return payload["messages"][-1]["content"]

    def _wire(self, body, request_id):
        return SimpleNamespace(http_response=SimpleNamespace(content=body), status_code=200,
                               headers={self.request_id_header: request_id})

    @staticmethod
    def _error_body(fault, request_id):
        return {"type": "error", "error": {"type": fault.error_type, "message": fault.message},
                "request_id": request_id}

    @staticmethod
    def _response(sequence, payload, text):
        return {"id": f"msg_{sequence}", "type": "message", "role": "assistant",
                "model": payload["model"], "stop_reason": "end_turn",
                "content": [{"type": "thinking", "thinking": "private"},
                            {"type": "text", "text": text}],
                "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
                          "cache_creation_input_tokens": 30,
                          "cache_creation": {"ephemeral_5m_input_tokens": 10,
                                             "ephemeral_1h_input_tokens": 20},
                          "service_tier": "standard"}}


# ---------------------------------------------------------------------------
# Crash images
# ---------------------------------------------------------------------------


class Crash(BaseException):
    """Stops a run where a killed process would, skipping its cleanup handlers."""


def copy_database(connection: sqlite3.Connection, path: str | Path, *,
                  timeout: float = 10.0) -> Path:
    """Copy a live database's last committed state, as a killed process leaves it.

    A separate read-only connection to the same file takes the copy, so it
    works inside or outside ``connection``'s open transaction and leaves that
    transaction running; its uncommitted writes are not copied. SQLite cannot
    back up a connection that is mid-write: the backup would wait forever.
    An in-memory database has no file to reopen, so it is copied directly and
    only outside a transaction. Raises ``sqlite3.OperationalError`` when
    another writer keeps the file locked for ``timeout`` seconds.
    """
    path = Path(path)
    source = connection.execute("PRAGMA database_list").fetchone()[2]
    if not source:
        if connection.in_transaction:
            raise ValueError("An in-memory database cannot be copied inside a transaction")
        with closing(sqlite3.connect(path)) as target:
            connection.backup(target)
        return path
    reader = sqlite3.connect(Path(source).as_uri() + "?mode=ro", uri=True, timeout=timeout,
                             isolation_level=None)
    with closing(reader), closing(sqlite3.connect(path)) as target:
        # Pin one committed snapshot first. A backup step that finds the file
        # busy retries without limit; this read gives up after ``timeout``.
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM sqlite_master").fetchone()
        reader.backup(target)
    return path


@dataclass
class CrashPoint:
    """A patched method that raises ``Crash`` from the first due call onward."""

    calls: list[tuple[tuple, dict]]
    image: Path | None
    restore: Callable[[], None]

    @property
    def fired(self) -> bool:
        return bool(self.calls)


def crash_when(monkeypatch, target: Any, name: str, due: Callable[..., bool] | None = None, *,
               image: str | Path | None = None,
               connection: sqlite3.Connection | None = None) -> CrashPoint:
    """Raise ``Crash`` from ``target.name`` once ``due(*args, **kwargs)`` holds.

    The patched method never runs the original at or after that call, because
    a killed process makes no further writes. With ``image``, the first crash
    first copies the last committed state of ``connection``'s database there,
    so a crash inside a transaction leaves the state before it. Reopen that
    copy with a clock past the lost owner's lease to resume as a new process
    would.
    """
    if image is not None and connection is None:
        raise ValueError("A crash image needs the database connection to copy")
    original = getattr(target, name)
    point = CrashPoint([], None if image is None else Path(image),
                       lambda: monkeypatch.setattr(target, name, original))

    def crash(*args, **kwargs):
        if point.calls or due is None or due(*args, **kwargs):
            if not point.calls and image is not None:
                copy_database(connection, image)
            point.calls.append((args, kwargs))
            raise Crash(f"Simulated process loss in {name}")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, crash)
    return point


# ---------------------------------------------------------------------------
# Child processes
# ---------------------------------------------------------------------------

_CREDENTIAL_VARIABLES = frozenset({"GITHUB_PERSONAL_ACCESS_TOKEN", "SMYTHE_MCP_URL"})


def offline_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A child environment without provider credentials, importing this checkout.

    The socket guard covers only the test process, so children must stay
    offline on their own: every ``*_API_KEY`` variable is removed.
    ``PYTHONPATH`` puts the repository root first, then this interpreter's
    import path, which includes ``tests`` so children can import ``fakes``.
    """
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().endswith("_API_KEY")
                   and key.upper() not in _CREDENTIAL_VARIABLES}
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), *(path for path in sys.path if path)])
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.update(extra or {})
    return environment


def child_executable() -> str:
    # The Windows venv python.exe can be a redirector: killing that wrapper
    # does not kill the worker, so launch the base interpreter it starts.
    return getattr(sys, "_base_executable", sys.executable) if os.name == "nt" else sys.executable


def launch_child(code: str, *args, cwd: str | Path = ROOT, env: dict[str, str] | None = None,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) -> subprocess.Popen:
    """Start ``python -c code *args`` as an offline child this test owns.

    Pair it with ``kill_and_reap`` in a ``finally`` block. Children that
    print a lot should pass files for ``stdout``/``stderr``, since pipes are
    read only when the child is reaped.
    """
    return subprocess.Popen(
        [child_executable(), "-c", code, *(str(arg) for arg in args)], cwd=cwd,
        env=offline_env() if env is None else env, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def write_barrier(path: str | Path, payload: dict) -> None:
    """Publish JSON atomically, so a reader never sees a partial barrier."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def wait_for_barrier(process: subprocess.Popen, path: str | Path, *, timeout: float = 30.0,
                     interval: float = 0.02) -> dict:
    """Return the child's barrier JSON; fail if it exits first or times out."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            pass
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=10)
            raise AssertionError(f"Child exited with {process.returncode} before {path.name}: "
                                 f"{stdout!r} {stderr!r}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"Child did not write {path.name} within {timeout} seconds")
        time.sleep(interval)


@dataclass(frozen=True)
class Reaped:
    returncode: int
    stdout: bytes
    stderr: bytes
    killed: bool  # The child was still running and was killed.


def kill_and_reap(process: subprocess.Popen, *, timeout: float = 10.0) -> Reaped:
    """Hard-kill a running child and collect its exit status and output.

    ``kill`` is TerminateProcess on Windows and SIGKILL on POSIX, so the child
    runs no cleanup handlers. Safe to call again, e.g. from ``finally``.
    """
    killed = process.poll() is None
    if killed:
        process.kill()
    stdout, stderr = process.communicate(timeout=timeout)
    return Reaped(process.returncode, stdout or b"", stderr or b"", killed)
