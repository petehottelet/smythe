"""Exact SQLite accounting and crash boundaries for native text workflows.

This is a separate database format from Jobs and Autotune. Monetary columns
contain canonical decimal nanoUSD strings; Python integers do all arithmetic
inside write transactions. Raw evidence is available only through explicit
load methods, never the routine inspection or event projections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from smythe.pricing import PRICE_VERSION, conservative_quote, price_native_response

STORE_VERSION = 1
STORE_KIND = "smythe.workflow"
OFFLINE_PRICE_VERSION = "offline-zero-v1"
_MONEY = re.compile(r"0|[1-9][0-9]*")
_PHASES = {"routing", "planning", "execution", "verification", "supervision", "synthesis"}


class WorkflowError(RuntimeError):
    """Workflow journal failure."""


class WorkflowValidationError(WorkflowError, ValueError):
    """Invalid caller data was rejected before a durable transition."""


class WorkflowConflictError(WorkflowError):
    """An immutable identity was reused with different content."""


class WorkflowLeaseError(WorkflowError):
    """The caller does not own a current fenced run lease."""


class WorkflowBudgetError(WorkflowError):
    """A durable exposure or overrun latch prevents admission."""


class WorkflowStateError(WorkflowError):
    """The requested transition is not allowed from the recorded state."""


class WorkflowNotFoundError(WorkflowError, KeyError):
    """The requested durable object does not exist."""


class WorkflowCorruptionError(WorkflowError):
    """Persisted journal data violates its format or accounting invariants."""


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise WorkflowValidationError(f"{name} must be a nonempty identifier without control characters")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkflowValidationError(f"{name} must be a nonnegative integer")
    return value


def _money(value: object, name: str = "nanoUSD") -> str:
    try:
        return str(_integer(value, name))
    except ValueError as exc:
        raise WorkflowValidationError("Unrepresentable " + name) from exc


def _amount(value: object) -> int:
    if type(value) is not str or _MONEY.fullmatch(value) is None:
        raise WorkflowCorruptionError("Invalid canonical nanoUSD in journal")
    try:
        return int(value)
    except ValueError as exc:
        raise WorkflowCorruptionError("Unrepresentable nanoUSD in journal") from exc


def _canonical(value: Any) -> str:
    def validate(item, active):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            import math
            if math.isfinite(item):
                return
        if type(item) not in (dict, list, tuple):
            raise WorkflowValidationError("Journal values must be finite JSON data")
        if id(item) in active:
            raise WorkflowValidationError("Journal JSON cannot contain cycles")
        active.add(id(item))
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise WorkflowValidationError("Journal JSON keys must be strings")
            children = item.values()
        else:
            children = item
        for child in children:
            validate(child, active)
        active.remove(id(item))
    try:
        validate(value, set())
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise WorkflowValidationError("Journal JSON exceeds supported finite serialization") from exc


def _loads(value: str | bytes, *, stored=False):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result
    def reject(value):
        raise ValueError("nonfinite value")
    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=reject)
        _canonical(result)
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        kind = WorkflowCorruptionError if stored else WorkflowValidationError
        raise kind("Invalid strict JSON") from exc


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _credential_free(value):
    """Reject credential fields in recipes, without examining source text."""
    forbidden = {"api_key", "apikey", "authorization", "access_token", "refresh_token",
                 "client_secret", "password", "secret_key", "private_key"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("-", "_") in forbidden:
                raise WorkflowValidationError("Workflow recipes cannot contain credentials")
            _credential_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _credential_free(item)


@dataclass(frozen=True, slots=True)
class CallKey:
    phase: str
    scope_id: str
    generation: int = 0
    invocation: int = 0
    attempt: int = 0
    turn: int = 0

    def __post_init__(self):
        if type(self.phase) is not str or self.phase not in _PHASES:
            raise WorkflowValidationError("Unknown workflow phase")
        _text(self.scope_id, "scope_id")
        for name in ("generation", "invocation", "attempt", "turn"):
            _integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class LeaseToken:
    run_id: str
    owner_id: str
    epoch: int
    expires_at_ns: int

    def __post_init__(self):
        _text(self.run_id, "run_id")
        _text(self.owner_id, "owner_id")
        if _integer(self.epoch, "epoch") == 0:
            raise WorkflowValidationError("Lease epoch must be positive")
        _integer(self.expires_at_ns, "expires_at_ns")


@dataclass(frozen=True, slots=True)
class CallPermit:
    call_id: str
    run_id: str
    ceiling_nanousd: int


@dataclass(frozen=True, slots=True)
class DispatchPermit:
    call_id: str
    run_id: str
    token: str
    request_sha256: str
    lease_epoch: int


class SQLiteWorkflowStore:
    """A fenced, exact local workflow journal with explicit raw evidence access."""

    def __init__(self, path: str | Path, *, read_only=False, clock_ns=time.time_ns):
        self.path = Path(path).resolve()
        self.read_only = read_only
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.path.as_uri() + "?mode=ro" if read_only else str(self.path)
        self._db = sqlite3.connect(target, uri=read_only, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        try:
            tables = {row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:
                required = {"workflow_meta", "workflow_runs", "workflow_calls", "workflow_evidence",
                            "workflow_checkpoints", "workflow_operations", "workflow_invocations", "workflow_events"}
                if not required <= tables:
                    raise WorkflowCorruptionError("Database is not a Smythe workflow journal")
                meta = self._db.execute("SELECT kind, version FROM workflow_meta").fetchall()
                if len(meta) != 1 or tuple(meta[0]) != (STORE_KIND, STORE_VERSION):
                    raise WorkflowCorruptionError("Unsupported workflow journal version")
            elif read_only:
                raise WorkflowCorruptionError("Database has no workflow schema")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA busy_timeout=5000")
            if read_only:
                self._db.execute("PRAGMA query_only=ON")
            else:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=FULL")
                if not tables:
                    self._create_schema()
            self._store_id = self._db.execute("SELECT store_id FROM workflow_meta").fetchone()[0]
            if type(self._store_id) is not str or re.fullmatch("[0-9a-f]{32}", self._store_id) is None:
                raise WorkflowCorruptionError("Invalid persistent workflow store identity")
        except sqlite3.DatabaseError as exc:
            self._db.close()
            raise WorkflowCorruptionError("Unreadable workflow journal schema") from exc
        except BaseException:
            self._db.close()
            raise

    def __enter__(self):
        return self

    @property
    def store_id(self):
        return self._store_id

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            self._db.close()

    def _create_schema(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS workflow_meta(kind TEXT PRIMARY KEY,version INTEGER NOT NULL,store_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workflow_runs(
          run_id TEXT PRIMARY KEY,task_json TEXT NOT NULL,config_json TEXT NOT NULL,binding_sha TEXT NOT NULL,
          budget TEXT,confirmed TEXT NOT NULL,reserved TEXT NOT NULL,unknown TEXT NOT NULL,
          status TEXT NOT NULL,blocked_reason TEXT,revision INTEGER NOT NULL DEFAULT 0,
          owner TEXT,epoch INTEGER NOT NULL DEFAULT 0,expires INTEGER NOT NULL DEFAULT 0,
          created INTEGER NOT NULL,updated INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS workflow_calls(
          call_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),key_json TEXT NOT NULL,
          request_json TEXT NOT NULL,tool_names_json TEXT NOT NULL,provider_json TEXT NOT NULL,
          price_version TEXT NOT NULL,binding_sha TEXT NOT NULL,request_sha TEXT NOT NULL,
          state TEXT NOT NULL,billing_state TEXT NOT NULL,result_state TEXT NOT NULL,
          quote_id TEXT,evidence_id TEXT,ceiling TEXT,cost TEXT,reserved TEXT NOT NULL,unknown TEXT NOT NULL,
          dispatch_token_sha TEXT,dispatch_epoch INTEGER,receipt_json TEXT,receipt_sha TEXT,
          result_json TEXT,result_sha TEXT,decoder_version TEXT,
          error TEXT,created INTEGER NOT NULL,updated INTEGER NOT NULL,UNIQUE(run_id,key_json));
        CREATE TABLE IF NOT EXISTS workflow_evidence(
          evidence_id TEXT PRIMARY KEY,call_id TEXT NOT NULL REFERENCES workflow_calls(call_id),
          operation TEXT NOT NULL,body BLOB NOT NULL,response_sha TEXT NOT NULL,request_sha TEXT NOT NULL,
          status_code INTEGER,request_id TEXT,transport_error TEXT,metadata_sha TEXT NOT NULL,created INTEGER NOT NULL,
          UNIQUE(call_id,operation,metadata_sha));
        CREATE TABLE IF NOT EXISTS workflow_checkpoints(
          run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),revision INTEGER NOT NULL,
          checkpoint_json TEXT NOT NULL,checkpoint_sha TEXT NOT NULL,consumed_json TEXT NOT NULL,
          operations_json TEXT NOT NULL,kind TEXT NOT NULL,created INTEGER NOT NULL,PRIMARY KEY(run_id,revision));
        CREATE TABLE IF NOT EXISTS workflow_operations(
          operation_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),operation_key TEXT NOT NULL,
          kind TEXT NOT NULL,state TEXT NOT NULL,inputs_json TEXT NOT NULL,inputs_sha TEXT NOT NULL,
          result_json TEXT,result_sha TEXT,consumed_json TEXT NOT NULL,created INTEGER NOT NULL,updated INTEGER NOT NULL,
          UNIQUE(run_id,operation_key));
        CREATE TABLE IF NOT EXISTS workflow_invocations(
          run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),scope_json TEXT NOT NULL,operation_key TEXT NOT NULL,
          ordinal INTEGER NOT NULL,PRIMARY KEY(run_id,scope_json,operation_key),UNIQUE(run_id,scope_json,ordinal));
        CREATE TABLE IF NOT EXISTS workflow_events(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
          call_id TEXT,event_type TEXT NOT NULL,payload_json TEXT NOT NULL,created INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS workflow_calls_by_run ON workflow_calls(run_id,state);
        """)
        if self._db.execute("SELECT COUNT(*) FROM workflow_meta").fetchone()[0] == 0:
            self._db.execute("INSERT OR IGNORE INTO workflow_meta VALUES (?,?,?)", (STORE_KIND, STORE_VERSION, uuid4().hex))

    @contextmanager
    def _transaction(self, *, write=True):
        if write and self.read_only:
            raise WorkflowStateError("Workflow store is read-only")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def _event(self, db, run_id, kind, *, call_id=None, payload=None):
        db.execute("INSERT INTO workflow_events(run_id,call_id,event_type,payload_json,created) VALUES (?,?,?,?,?)",
                   (run_id, call_id, kind, _canonical(payload or {}), self._clock_ns()))

    def _run(self, db, run_id):
        row = db.execute("SELECT * FROM workflow_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise WorkflowNotFoundError(run_id)
        for key in ("confirmed", "reserved", "unknown"):
            _amount(row[key])
        if row["budget"] is not None:
            _amount(row["budget"])
        task, config = _loads(row["task_json"], stored=True), _loads(row["config_json"], stored=True)
        if (task is not None and type(task) is not dict) or type(config) is not dict:
            raise WorkflowCorruptionError("Workflow task/config schema is invalid")
        if row["status"] not in ("running", "blocked", "completed") or type(row["revision"]) is not int or row["revision"] < 0:
            raise WorkflowCorruptionError("Workflow control state is invalid")
        if _sha(_canonical([task, config, row["budget"]])) != row["binding_sha"]:
            raise WorkflowCorruptionError("Workflow run binding is corrupt")
        return row

    def _lease(self, db, lease):
        if not isinstance(lease, LeaseToken):
            raise WorkflowLeaseError("Expected a lease token")
        row = self._run(db, lease.run_id)
        if (row["owner"] != lease.owner_id or row["epoch"] != lease.epoch
                or row["expires"] <= self._clock_ns()):
            raise WorkflowLeaseError("Workflow lease expired or was superseded")
        return row

    def _call(self, db, call_id):
        row = db.execute("SELECT * FROM workflow_calls WHERE call_id=?", (call_id,)).fetchone()
        if row is None:
            raise WorkflowNotFoundError(call_id)
        self._call_record(row)
        return row

    def _owned_call(self, db, lease, call_id):
        run = self._lease(db, lease)
        call = self._call(db, call_id)
        if call["run_id"] != lease.run_id:
            raise WorkflowConflictError("Call belongs to another workflow")
        return run, call

    def _admit(self, run):
        if run["blocked_reason"]:
            kind = WorkflowBudgetError if run["blocked_reason"] == "budget_overrun" else WorkflowStateError
            raise kind("Workflow admission is closed: " + run["blocked_reason"])
        if run["status"] == "completed":
            raise WorkflowStateError("Completed workflow cannot dispatch new work")

    @staticmethod
    def _balances(run):
        return tuple(_amount(run[name]) for name in ("confirmed", "reserved", "unknown"))

    def create_run(self, task, config, budget_nanousd, *, run_id=None):
        run_id = _text(run_id or uuid4().hex, "run_id")
        if task is not None and type(task) is not dict or type(config) is not dict:
            raise WorkflowValidationError("Task/config must be JSON objects")
        task_json, config_json = _canonical(task), _canonical(config)
        _credential_free(config)
        budget = None if budget_nanousd is None else _money(budget_nanousd)
        binding = _sha(_canonical([task, config, budget]))
        now = self._clock_ns()
        with self._transaction() as db:
            existing = db.execute("SELECT binding_sha FROM workflow_runs WHERE run_id=?", (run_id,)).fetchone()
            if existing:
                if existing[0] != binding:
                    raise WorkflowConflictError("Workflow run identity has different inputs")
            else:
                db.execute("""INSERT INTO workflow_runs(run_id,task_json,config_json,binding_sha,budget,
                  confirmed,reserved,unknown,status,created,updated) VALUES (?,?,?,?,?,'0','0','0','running',?,?)""",
                           (run_id, task_json, config_json, binding, budget, now, now))
                self._event(db, run_id, "run_created")
        return self.load_run(run_id)

    def load_run(self, run_id):
        with self._transaction(write=False) as db:
            row = self._run(db, run_id)
            return {"run_id": run_id, "task": _loads(row["task_json"], stored=True),
                    "config": _loads(row["config_json"], stored=True), "binding_sha256": row["binding_sha"],
                    "store_id": self.store_id, "config_sha256": _sha(row["config_json"]),
                    "task_sha256": _sha(row["task_json"]),
                    "budget_nanousd": None if row["budget"] is None else _amount(row["budget"]),
                    "revision": row["revision"], "status": row["status"], "blocked_reason": row["blocked_reason"]}

    def list_runs(self):
        """Discover resume IDs without exposing Task, configuration or output."""
        with self._transaction(write=False) as db:
            result = []
            for candidate in db.execute("SELECT run_id FROM workflow_runs ORDER BY created,run_id"):
                row = self._run(db, candidate[0])
                result.append({"run_id": row["run_id"], "status": row["status"], "revision": row["revision"],
                               "config_sha256": _sha(row["config_json"]), "task_sha256": _sha(row["task_json"]),
                               "budget_nanousd": None if row["budget"] is None else _amount(row["budget"]),
                               "confirmed_nanousd": _amount(row["confirmed"]), "reserved_nanousd": _amount(row["reserved"]),
                               "unknown_nanousd": _amount(row["unknown"]), "blocked_reason": row["blocked_reason"]})
            return result

    def acquire_lease(self, run_id, owner_id, ttl_s=30.0):
        _text(owner_id, "owner_id")
        duration = self._ttl(ttl_s)
        now = self._clock_ns()
        with self._transaction() as db:
            run = self._run(db, run_id)
            active = run["owner"] is not None and run["expires"] > now
            if active and run["owner"] != owner_id:
                raise WorkflowLeaseError("Workflow already has an active lease")
            epoch = run["epoch"] if active else run["epoch"] + 1
            db.execute("UPDATE workflow_runs SET owner=?,epoch=?,expires=?,updated=? WHERE run_id=?",
                       (owner_id, epoch, now + duration, now, run_id))
            self._event(db, run_id, "lease_acquired", payload={"epoch": epoch})
        return LeaseToken(run_id, owner_id, epoch, now + duration)

    @staticmethod
    def _ttl(ttl_s):
        import math
        try:
            valid = type(ttl_s) in (int, float) and math.isfinite(ttl_s) and 0 < ttl_s <= 86400
        except OverflowError:
            valid = False
        if not valid:
            raise WorkflowValidationError("Lease duration must be positive and at most one day")
        return max(1, int(ttl_s * 1_000_000_000))

    def heartbeat(self, lease, ttl_s=30.0):
        duration = self._ttl(ttl_s)
        with self._transaction() as db:
            self._lease(db, lease)
            expires = self._clock_ns() + duration
            db.execute("UPDATE workflow_runs SET expires=? WHERE run_id=?", (expires, lease.run_id))
        return LeaseToken(lease.run_id, lease.owner_id, lease.epoch, expires)

    def release_lease(self, lease):
        with self._transaction() as db:
            self._lease(db, lease)
            db.execute("UPDATE workflow_runs SET owner=NULL,expires=0 WHERE run_id=?", (lease.run_id,))
            self._event(db, lease.run_id, "lease_released", payload={"epoch": lease.epoch})

    def allocate_invocation(self, lease, phase, scope_id, generation, operation_key):
        key = CallKey(phase, scope_id, generation)
        scope = _canonical([key.phase, key.scope_id, key.generation])
        _text(operation_key, "operation_key")
        with self._transaction() as db:
            self._lease(db, lease)
            row = db.execute("SELECT ordinal FROM workflow_invocations WHERE run_id=? AND scope_json=? AND operation_key=?",
                             (lease.run_id, scope, operation_key)).fetchone()
            if row:
                return row[0]
            ordinal = db.execute("SELECT COALESCE(MAX(ordinal),-1)+1 FROM workflow_invocations WHERE run_id=? AND scope_json=?",
                                 (lease.run_id, scope)).fetchone()[0]
            db.execute("INSERT INTO workflow_invocations VALUES (?,?,?,?)", (lease.run_id, scope, operation_key, ordinal))
            return ordinal

    @staticmethod
    def _request_binding(request_json, tool_names_json, provider, price_version):
        request, names = _loads(request_json), _loads(tool_names_json)
        if type(request) is not dict or type(names) is not dict or type(provider) is not dict:
            raise WorkflowValidationError("Request, tool map and provider must be JSON objects")
        if _canonical(request) != request_json or _canonical(names) != tool_names_json:
            raise WorkflowValidationError("Prepared request must use canonical JSON")
        kind = provider.get("kind")
        if type(kind) is not str or kind not in {"openai_responses", "offline"}:
            raise WorkflowValidationError("Unsupported durable provider kind")
        for version in ("adapter_version", "decoder_version"):
            _text(provider.get(version), version)
        _text(price_version, "price_version")
        if kind == "openai_responses":
            if price_version != PRICE_VERSION:
                raise WorkflowValidationError("Unsupported native price version")
            if request.get("model") not in ("gpt-6-astra", "gpt-5.6-sol") or request.get("service_tier") != "default":
                raise WorkflowValidationError("Unsupported native billing identity")
            if provider.get("endpoint", "https://api.openai.com/v1") != "https://api.openai.com/v1":
                raise WorkflowValidationError("Durable native endpoint must be global")
            if provider.get("endpoint_scope", "global") != "global":
                raise WorkflowValidationError("Durable native endpoint scope must be global")
            cap = _integer(request.get("max_output_tokens"), "max_output_tokens")
            if not 0 < cap <= 128_000:
                raise WorkflowValidationError("max_output_tokens must be within 1..128000")
            if request.get("tools") or names:
                raise WorkflowValidationError("Durable text workflows do not support tool side effects")
            # Reuse the local wire validator so quote admission cannot diverge
            # from the adapter's supported text-only request configuration.
            from smythe.provider_responses import OpenAIResponsesProvider, PreparedRequest
            try:
                OpenAIResponsesProvider._validate_prepared(PreparedRequest(request_json, tool_names_json))
            except (TypeError, ValueError, KeyError) as exc:
                raise WorkflowValidationError("Native request is outside supported text workflow scope") from exc
        elif price_version != OFFLINE_PRICE_VERSION:
            raise WorkflowValidationError("Offline calls require the explicit zero-cost price version")
        provider_json = _canonical(provider)
        _credential_free(provider)
        binding = _sha(_canonical([request_json, tool_names_json, provider_json, price_version]))
        return provider_json, binding

    def prepare_call(self, lease, key, *, request_json, tool_names_json="{}", provider, price_version):
        if not isinstance(key, CallKey):
            raise WorkflowValidationError("Expected CallKey")
        provider_json, binding = self._request_binding(request_json, tool_names_json, provider, price_version)
        key_json = _canonical(asdict(key))
        with self._transaction() as db:
            run = self._lease(db, lease)
            existing = db.execute("SELECT * FROM workflow_calls WHERE run_id=? AND key_json=?", (lease.run_id, key_json)).fetchone()
            if existing:
                if existing["binding_sha"] != binding:
                    raise WorkflowConflictError("Logical call has different immutable request bindings")
                return self._call_record(existing)
            self._admit(run)
            call_id, now = uuid4().hex, self._clock_ns()
            db.execute("""INSERT INTO workflow_calls(call_id,run_id,key_json,request_json,tool_names_json,
              provider_json,price_version,binding_sha,request_sha,state,billing_state,result_state,reserved,unknown,created,updated)
              VALUES (?,?,?,?,?,?,?,?,?,'prepared','unquoted','pending','0','0',?,?)""",
                       (call_id, lease.run_id, key_json, request_json, tool_names_json, provider_json,
                        price_version, binding, _sha(request_json), now, now))
            self._event(db, lease.run_id, "call_prepared", call_id=call_id, payload={"key": asdict(key)})
            return self._call_record(self._call(db, call_id))

    def lookup_call(self, run_id, key):
        if not isinstance(key, CallKey):
            raise WorkflowValidationError("Expected CallKey")
        with self._transaction(write=False) as db:
            self._run(db, run_id)
            row = db.execute("SELECT * FROM workflow_calls WHERE run_id=? AND key_json=?", (run_id, _canonical(asdict(key)))).fetchone()
            return self._call_record(row) if row else None

    @staticmethod
    def _call_record(row):
        request = _loads(row["request_json"], stored=True)
        names = _loads(row["tool_names_json"], stored=True)
        provider = _loads(row["provider_json"], stored=True)
        key = _loads(row["key_json"], stored=True)
        try:
            CallKey(**key)
            _, binding = SQLiteWorkflowStore._request_binding(row["request_json"], row["tool_names_json"], provider, row["price_version"])
        except (WorkflowValidationError, TypeError, KeyError) as exc:
            raise WorkflowCorruptionError("Call bindings are invalid") from exc
        if binding != row["binding_sha"] or _sha(row["request_json"]) != row["request_sha"]:
            raise WorkflowCorruptionError("Call request binding is corrupt")
        if type(request) is not dict or type(names) is not dict:
            raise WorkflowCorruptionError("Call request is not an object")
        for column in ("receipt", "result"):
            value = row[column + "_json"]
            if (None if value is None else _sha(value)) != row[column + "_sha"]:
                raise WorkflowCorruptionError("Call " + column + " content hash is corrupt")
        state, billing, result_state = row["state"], row["billing_state"], row["result_state"]
        expected_billing = {"prepared": "unquoted", "quoted": "quoted", "reserved": "reserved",
                            "dispatched": "reserved", "response_saved": "reserved", "settled": "known",
                            "unknown": "unknown", "released": "released"}
        if state not in expected_billing or billing != expected_billing[state]:
            raise WorkflowCorruptionError("Call state and billing state disagree")
        if result_state not in ("pending", "accepted", "rejected", "applied"):
            raise WorkflowCorruptionError("Invalid native result state")
        ceiling = None if row["ceiling"] is None else _amount(row["ceiling"])
        cost = None if row["cost"] is None else _amount(row["cost"])
        held, unknown = _amount(row["reserved"]), _amount(row["unknown"])
        if ((row["quote_id"] is None) != (ceiling is None)
                or state not in ("prepared", "released") and ceiling is None
                or (billing == "known") != (cost is not None)
                or held != (ceiling if billing == "reserved" else 0)
                or unknown != (ceiling if billing == "unknown" else 0)):
            raise WorkflowCorruptionError("Call balances disagree with its state")
        dispatched = state in ("dispatched", "response_saved", "settled", "unknown")
        if (dispatched != (row["dispatch_token_sha"] is not None)
                or dispatched and (type(row["dispatch_epoch"]) is not int or row["dispatch_epoch"] < 1)
                or row["evidence_id"] is not None and not dispatched
                or state in ("response_saved", "settled") and row["evidence_id"] is None):
            raise WorkflowCorruptionError("Call dispatch evidence is inconsistent")
        if (result_state != "pending" and billing != "known"
                or (result_state in ("accepted", "applied")) != (row["result_json"] is not None)):
            raise WorkflowCorruptionError("Call semantic acceptance is inconsistent")
        return {
            "call_id": row["call_id"], "run_id": row["run_id"], "key": key,
            "request_json": row["request_json"], "tool_names_json": row["tool_names_json"],
            "provider": provider, "price_version": row["price_version"],
            "binding_sha256": row["binding_sha"], "request_sha256": row["request_sha"],
            "state": row["state"], "billing_state": row["billing_state"], "result_state": row["result_state"],
            "quote_id": row["quote_id"], "evidence_id": row["evidence_id"],
            "ceiling_nanousd": None if row["ceiling"] is None else _amount(row["ceiling"]),
            "cost_nanousd": None if row["cost"] is None else _amount(row["cost"]),
            "reserved_nanousd": _amount(row["reserved"]), "unknown_nanousd": _amount(row["unknown"]),
            "receipt": None if row["receipt_json"] is None else _loads(row["receipt_json"], stored=True),
            "decoded_result": None if row["result_json"] is None else _loads(row["result_json"], stored=True),
            "decoder_version": row["decoder_version"] or provider["decoder_version"], "error": row["error"],
        }

    @staticmethod
    def _envelope(envelope):
        def get(key, default=None):
            return envelope.get(key, default) if isinstance(envelope, Mapping) else getattr(envelope, key, default)
        body = get("body")
        if type(body) is not bytes:
            raise WorkflowValidationError("Raw response body must be bytes")
        operation = get("operation")
        if operation not in ("input_tokens", "response"):
            raise WorkflowValidationError("Invalid raw evidence operation")
        request_sha = get("request_sha256")
        if type(request_sha) is not str or re.fullmatch("[0-9a-f]{64}", request_sha) is None:
            raise WorkflowValidationError("Invalid evidence request hash")
        status = get("status_code")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise WorkflowValidationError("Invalid HTTP status")
        request_id, error = get("request_id"), get("transport_error")
        for value, name in ((request_id, "request_id"), (error, "transport_error")):
            if value is not None:
                _text(value, name)
        response_sha = _sha(body)
        metadata_sha = _sha(_canonical([operation, request_sha, response_sha, status, request_id, error]))
        return (operation, body, response_sha, request_sha, status, request_id, error, metadata_sha)

    def _insert_evidence(self, db, call, values):
        if values[3] != call["request_sha"]:
            raise WorkflowConflictError("Evidence belongs to a different request")
        row = db.execute("SELECT evidence_id FROM workflow_evidence WHERE call_id=? AND operation=? AND metadata_sha=?",
                         (call["call_id"], values[0], values[-1])).fetchone()
        if row:
            return row[0]
        evidence_id = uuid4().hex
        db.execute("INSERT INTO workflow_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (evidence_id, call["call_id"], *values, self._clock_ns()))
        self._event(db, call["run_id"], "evidence_saved", call_id=call["call_id"],
                    payload={"evidence_id": evidence_id, "operation": values[0], "response_sha256": values[2]})
        return evidence_id

    @staticmethod
    def _evidence_record(row):
        result = {"evidence_id": row["evidence_id"], "body": bytes(row["body"]),
                  "request_sha256": row["request_sha"], "operation": row["operation"],
                  "status_code": row["status_code"], "request_id": row["request_id"],
                  "transport_error": row["transport_error"], "response_sha256": row["response_sha"]}
        try:
            values = SQLiteWorkflowStore._envelope(result)
        except WorkflowValidationError as exc:
            raise WorkflowCorruptionError("Malformed persisted evidence") from exc
        if values[2] != row["response_sha"] or values[-1] != row["metadata_sha"]:
            raise WorkflowCorruptionError("Raw evidence hash is corrupt")
        return result

    def _evidence(self, db, call, evidence_id, operation):
        row = db.execute("SELECT * FROM workflow_evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        if row is None:
            raise WorkflowNotFoundError(evidence_id)
        if row["call_id"] != call["call_id"] or row["operation"] != operation or row["request_sha"] != call["request_sha"]:
            raise WorkflowConflictError("Evidence identity does not match this call")
        return self._evidence_record(row)

    def append_quote_evidence(self, lease, call_id, envelope):
        values = self._envelope(envelope)
        if values[0] != "input_tokens":
            raise WorkflowValidationError("Expected input-token evidence")
        with self._transaction() as db:
            _, call = self._owned_call(db, lease, call_id)
            return self._insert_evidence(db, call, values)

    def _quote(self, call, evidence):
        if evidence["transport_error"] or evidence["status_code"] not in (None, 200):
            raise WorkflowValidationError("Input count did not return a successful response")
        raw = _loads(evidence["body"])
        if type(raw) is not dict:
            raise WorkflowValidationError("Input count body must be an object")
        provider = _loads(call["provider_json"], stored=True)
        request = _loads(call["request_json"], stored=True)
        count = _integer(raw.get("input_tokens"), "input_tokens")
        if provider["kind"] == "offline":
            if (raw != {"provider_kind": "offline", "version": 1, "input_tokens": 0}
                    or type(raw.get("version")) is not int):
                raise WorkflowValidationError("Invalid offline quote evidence")
            ceiling = 0
        else:
            quote = conservative_quote(count, request["max_output_tokens"], request["model"])
            digits = quote.as_tuple()
            coefficient = int("".join(map(str, digits.digits)))
            exponent = digits.exponent + 9
            if exponent < 0:
                raise WorkflowValidationError("Quote cannot be represented as nanoUSD")
            ceiling = coefficient * 10 ** exponent
        return {"quote_id": evidence["evidence_id"], "call_id": call["call_id"], "input_tokens": count,
                "ceiling_nanousd": ceiling, "price_version": call["price_version"],
                "request_sha256": call["request_sha"]}

    def accept_quote(self, lease, call_id, evidence_id):
        with self._transaction() as db:
            _, call = self._owned_call(db, lease, call_id)
            quote = self._quote(call, self._evidence(db, call, evidence_id, "input_tokens"))
            if call["quote_id"] is not None:
                if call["quote_id"] != evidence_id or _amount(call["ceiling"]) != quote["ceiling_nanousd"]:
                    raise WorkflowConflictError("Call already has a different accepted quote")
                return quote
            if call["state"] not in ("prepared", "released") or call["dispatch_token_sha"]:
                raise WorkflowStateError("Call cannot accept a quote in this state")
            db.execute("UPDATE workflow_calls SET quote_id=?,ceiling=?,state='quoted',billing_state='quoted',updated=? WHERE call_id=?",
                       (evidence_id, _money(quote["ceiling_nanousd"]), self._clock_ns(), call_id))
            return quote

    def reserve_call(self, lease, call_id, quote_id):
        with self._transaction() as db:
            run, call = self._owned_call(db, lease, call_id)
            self._admit(run)
            if call["quote_id"] != quote_id or quote_id is None:
                raise WorkflowConflictError("Reservation quote does not match the call")
            ceiling = _amount(call["ceiling"])
            if call["state"] == "reserved":
                return CallPermit(call_id, lease.run_id, ceiling)
            if call["state"] not in ("quoted", "released"):
                raise WorkflowStateError("Call cannot reserve again after dispatch")
            confirmed, reserved, unknown = self._balances(run)
            if run["budget"] is not None and confirmed + reserved + unknown + ceiling > _amount(run["budget"]):
                raise WorkflowBudgetError("Call quote exceeds the remaining workflow budget")
            db.execute("UPDATE workflow_runs SET reserved=?,updated=? WHERE run_id=?",
                       (_money(reserved + ceiling), self._clock_ns(), lease.run_id))
            db.execute("UPDATE workflow_calls SET reserved=?,state='reserved',billing_state='reserved',updated=? WHERE call_id=?",
                       (_money(ceiling), self._clock_ns(), call_id))
            self._event(db, lease.run_id, "call_reserved", call_id=call_id, payload={"ceiling_nanousd": str(ceiling)})
            return CallPermit(call_id, lease.run_id, ceiling)

    def claim_dispatch(self, lease, call_id, request_sha256):
        with self._transaction() as db:
            run, call = self._owned_call(db, lease, call_id)
            self._admit(run)
            if request_sha256 != call["request_sha"]:
                raise WorkflowConflictError("Dispatch request does not match the reserved request")
            if call["state"] != "reserved" or call["dispatch_token_sha"] is not None:
                raise WorkflowStateError("Dispatch has already been claimed or was never reserved")
            token = secrets.token_hex(32)
            db.execute("UPDATE workflow_calls SET state='dispatched',dispatch_token_sha=?,dispatch_epoch=?,updated=? WHERE call_id=?",
                       (_sha(token), lease.epoch, self._clock_ns(), call_id))
            self._event(db, lease.run_id, "dispatch_claimed", call_id=call_id)
            return DispatchPermit(call_id, lease.run_id, token, request_sha256, lease.epoch)

    def append_response(self, permit, envelope):
        if not isinstance(permit, DispatchPermit):
            raise WorkflowValidationError("Expected dispatch permit")
        values = self._envelope(envelope)
        if values[0] != "response":
            raise WorkflowValidationError("Expected generation response evidence")
        conflict = False
        with self._transaction() as db:
            call = self._call(db, permit.call_id)
            if (call["run_id"] != permit.run_id or call["dispatch_token_sha"] != _sha(permit.token)
                    or call["dispatch_epoch"] != permit.lease_epoch or call["request_sha"] != permit.request_sha256):
                raise WorkflowConflictError("Response permit does not match the dispatched call")
            evidence_id = self._insert_evidence(db, call, values)
            if call["evidence_id"] not in (None, evidence_id):
                conflict = True
                db.execute("UPDATE workflow_runs SET status='blocked',blocked_reason='evidence_conflict' WHERE run_id=?", (permit.run_id,))
                self._event(db, permit.run_id, "evidence_conflict", call_id=permit.call_id)
            elif call["evidence_id"] is None:
                db.execute("UPDATE workflow_calls SET evidence_id=?,state=CASE WHEN state='dispatched' THEN 'response_saved' ELSE state END,updated=? WHERE call_id=?",
                           (evidence_id, self._clock_ns(), permit.call_id))
        if conflict:
            raise WorkflowConflictError("Conflicting raw responses retained; workflow admission is closed")
        return evidence_id

    def _update_latch(self, db, run_id):
        run = self._run(db, run_id)
        if run["blocked_reason"] not in (None, "unknown_exposure", "budget_overrun"):
            return
        balances = self._balances(run)
        overrun = run["budget"] is not None and sum(balances) > _amount(run["budget"])
        unknown = db.execute("SELECT 1 FROM workflow_calls WHERE run_id=? AND billing_state='unknown' LIMIT 1", (run_id,)).fetchone()
        reason = "budget_overrun" if overrun or run["blocked_reason"] == "budget_overrun" else "unknown_exposure" if unknown else None
        db.execute("UPDATE workflow_runs SET status=?,blocked_reason=? WHERE run_id=?",
                   ("blocked" if reason else "running", reason, run_id))

    def _unknown(self, db, run, call, reason):
        if call["billing_state"] == "known":
            raise WorkflowStateError("Known billing cannot be replaced with unknown exposure")
        if call["state"] not in ("dispatched", "response_saved", "unknown"):
            raise WorkflowStateError("Only a dispatched call can have unknown exposure")
        confirmed, reserved, unknown = self._balances(run)
        held, previous = _amount(call["reserved"]), _amount(call["unknown"])
        exposure = _amount(call["ceiling"])
        db.execute("UPDATE workflow_runs SET reserved=?,unknown=?,updated=? WHERE run_id=?",
                   (_money(reserved - held), _money(unknown - previous + exposure), self._clock_ns(), call["run_id"]))
        db.execute("UPDATE workflow_calls SET state='unknown',billing_state='unknown',reserved='0',unknown=?,error=?,updated=? WHERE call_id=?",
                   (_money(exposure), reason, self._clock_ns(), call["call_id"]))
        self._update_latch(db, call["run_id"])

    def mark_unknown(self, lease, call_id, reason):
        _text(reason, "reason")
        with self._transaction() as db:
            run, call = self._owned_call(db, lease, call_id)
            self._unknown(db, run, call, reason)

    def settle_call(self, lease, call_id, evidence_id):
        with self._transaction() as db:
            run, call = self._owned_call(db, lease, call_id)
            evidence = self._evidence(db, call, evidence_id, "response")
            if call["evidence_id"] != evidence_id:
                raise WorkflowConflictError("Response is not the selected evidence")
            if call["state"] not in ("response_saved", "unknown", "settled"):
                raise WorkflowStateError("Call has no dispatched response to settle")
            provider = _loads(call["provider_json"], stored=True)
            request = _loads(call["request_json"], stored=True)
            try:
                raw = _loads(evidence["body"])
            except WorkflowValidationError:
                raw = None
            if provider["kind"] == "offline":
                valid = (type(raw) is dict and raw.get("provider_kind") == "offline"
                         and type(raw.get("version")) is int and raw["version"] == 1 and type(raw.get("text")) is str
                         and not evidence["transport_error"] and evidence["status_code"] in (None, 200))
                cost = 0 if valid else None
                receipt = {"version": 1, "provider_kind": "offline", "price_version": OFFLINE_PRICE_VERSION,
                           "pricing_scope": "zero_external_api_cost", "cost_nanousd": "0" if valid else None,
                           "cost_usd": "0" if valid else None, "cost_is_complete": valid}
            else:
                native = price_native_response(raw, requested_model=request["model"])
                cost, receipt = native.cost_nanousd, native.safe_summary()
                if evidence["transport_error"] or evidence["status_code"] not in (None, 200):
                    cost = None
                    receipt.update(cost_is_complete=False, cost_nanousd=None, cost_usd=None,
                                   accounting_error="Transport or HTTP response was not successful")
            response_id = raw.get("id") if type(raw) is dict else None
            if type(response_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", response_id) is None:
                response_id = None
            request_id = evidence["request_id"]
            if request_id is not None and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", request_id) is None:
                request_id = None
            receipt.update(request_sha256=call["request_sha"], response_sha256=evidence["response_sha256"],
                           evidence_id=evidence_id, call_id=call_id, run_id=lease.run_id,
                           request_id=request_id, response_id=response_id)
            receipt_json = _canonical(receipt)
            if call["billing_state"] == "known":
                if call["receipt_json"] != receipt_json or _amount(call["cost"]) != cost:
                    raise WorkflowConflictError("Known settlement differs from saved billing")
            elif cost is None:
                self._unknown(db, run, call, "unpriced_response")
                db.execute("UPDATE workflow_calls SET receipt_json=?,receipt_sha=? WHERE call_id=?", (receipt_json, _sha(receipt_json), call_id))
            else:
                confirmed, reserved, unknown = self._balances(run)
                db.execute("UPDATE workflow_runs SET confirmed=?,reserved=?,unknown=?,updated=? WHERE run_id=?",
                           (_money(confirmed + cost), _money(reserved - _amount(call["reserved"])),
                            _money(unknown - _amount(call["unknown"])), self._clock_ns(), lease.run_id))
                db.execute("UPDATE workflow_calls SET state='settled',billing_state='known',cost=?,reserved='0',unknown='0',receipt_json=?,receipt_sha=?,error=NULL,updated=? WHERE call_id=?",
                           (_money(cost), receipt_json, _sha(receipt_json), self._clock_ns(), call_id))
                if cost > _amount(call["ceiling"]):
                    db.execute("UPDATE workflow_runs SET status='blocked',blocked_reason='budget_overrun' WHERE run_id=? AND blocked_reason IN ('unknown_exposure','budget_overrun')",
                               (lease.run_id,))
                    db.execute("UPDATE workflow_runs SET status='blocked',blocked_reason='budget_overrun' WHERE run_id=? AND blocked_reason IS NULL",
                               (lease.run_id,))
                self._update_latch(db, lease.run_id)
                self._event(db, lease.run_id, "call_settled", call_id=call_id, payload={"cost_nanousd": str(cost)})
            result = self._call_record(self._call(db, call_id))
            current = self._run(db, lease.run_id)
            return {**result, "run_status": current["status"], "blocked_reason": current["blocked_reason"],
                    "admission_closed": current["blocked_reason"] is not None}

    def accept_result(self, lease, call_id, decoded_result, decoder_version):
        if type(decoded_result) is not dict:
            raise WorkflowValidationError("Decoded result must be a JSON object")
        result_json = _canonical(decoded_result)
        with self._transaction() as db:
            _, call = self._owned_call(db, lease, call_id)
            provider = _loads(call["provider_json"], stored=True)
            if decoder_version != provider["decoder_version"]:
                raise WorkflowConflictError("Decoder version differs from prepared call")
            if call["billing_state"] != "known":
                raise WorkflowStateError("Cannot accept output without known billing")
            if call["result_state"] != "pending":
                if call["result_json"] != result_json or call["result_state"] == "rejected":
                    raise WorkflowConflictError("Call already has a different semantic result")
            else:
                db.execute("UPDATE workflow_calls SET result_state='accepted',result_json=?,result_sha=?,decoder_version=?,updated=? WHERE call_id=?",
                           (result_json, _sha(result_json), decoder_version, self._clock_ns(), call_id))
            return self._call_record(self._call(db, call_id))

    def reject_result(self, lease, call_id, reason):
        _text(reason, "reason")
        with self._transaction() as db:
            _, call = self._owned_call(db, lease, call_id)
            if call["billing_state"] != "known":
                raise WorkflowStateError("Semantic rejection requires settled billing")
            if call["result_state"] != "pending":
                if call["result_state"] != "rejected" or call["error"] != reason:
                    raise WorkflowConflictError("Call already has a different semantic result")
            else:
                db.execute("UPDATE workflow_calls SET result_state='rejected',error=?,updated=? WHERE call_id=?",
                           (reason, self._clock_ns(), call_id))
            return self._call_record(self._call(db, call_id))

    def release_before_dispatch(self, lease, call_id, reason):
        _text(reason, "reason")
        with self._transaction() as db:
            run, call = self._owned_call(db, lease, call_id)
            if call["state"] == "released":
                return
            if call["state"] not in ("prepared", "quoted", "reserved") or call["dispatch_token_sha"]:
                raise WorkflowStateError("A dispatched call cannot release its reservation")
            _, reserved, _ = self._balances(run)
            db.execute("UPDATE workflow_runs SET reserved=? WHERE run_id=?", (_money(reserved - _amount(call["reserved"])), lease.run_id))
            db.execute("UPDATE workflow_calls SET reserved='0',state='released',billing_state='released',error=? WHERE call_id=?", (reason, call_id))

    def load_evidence(self, call_id, evidence_id):
        with self._transaction(write=False) as db:
            call = self._call(db, call_id)
            row = db.execute("SELECT operation FROM workflow_evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            if row is None:
                raise WorkflowNotFoundError(evidence_id)
            return self._evidence(db, call, evidence_id, row[0])

    def load_replay(self, call_id):
        with self._transaction(write=False) as db:
            call = self._call(db, call_id)
            return {**self._call_record(call),
                    "quote_evidence": self._evidence(db, call, call["quote_id"], "input_tokens") if call["quote_id"] else None,
                    "evidence": self._evidence(db, call, call["evidence_id"], "response") if call["evidence_id"] else None}

    @staticmethod
    def _ids(values, name):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise WorkflowValidationError(name + " must be an ID sequence")
        result = [_text(value, name) for value in values]
        if len(set(result)) != len(result):
            raise WorkflowValidationError(name + " cannot contain duplicate IDs")
        return sorted(result)

    @staticmethod
    def _operation_record(row):
        inputs = _loads(row["inputs_json"], stored=True)
        result = None if row["result_json"] is None else _loads(row["result_json"], stored=True)
        consumed = _loads(row["consumed_json"], stored=True)
        if (type(inputs) is not dict or result is not None and type(result) is not dict
                or _sha(row["inputs_json"]) != row["inputs_sha"]
                or (None if row["result_json"] is None else _sha(row["result_json"])) != row["result_sha"]
                or row["state"] not in ("started", "completed", "applied")
                or (row["state"] == "started") != (result is None)):
            raise WorkflowCorruptionError("Invalid durable operation binding")
        try:
            if SQLiteWorkflowStore._ids(consumed, "consumed_call_ids") != consumed:
                raise WorkflowValidationError("Noncanonical IDs")
        except WorkflowValidationError as exc:
            raise WorkflowCorruptionError("Invalid operation call IDs") from exc
        return {"operation_id": row["operation_id"], "operation_key": row["operation_key"],
                "kind": row["kind"], "state": row["state"], "inputs": inputs, "result": result,
                "inputs_sha256": row["inputs_sha"], "result_sha256": row["result_sha"],
                "consumed_call_ids": consumed}

    def load_operation(self, run_id, operation_key):
        with self._transaction(write=False) as db:
            self._run(db, run_id)
            row = db.execute("SELECT * FROM workflow_operations WHERE run_id=? AND operation_key=?", (run_id, operation_key)).fetchone()
            return self._operation_record(row) if row else None

    def begin_operation(self, lease, operation_key, kind, inputs):
        _text(operation_key, "operation_key")
        _text(kind, "kind")
        if type(inputs) is not dict:
            raise WorkflowValidationError("Operation inputs must be a JSON object")
        inputs_json = _canonical(inputs)
        with self._transaction() as db:
            self._lease(db, lease)
            row = db.execute("SELECT * FROM workflow_operations WHERE run_id=? AND operation_key=?", (lease.run_id, operation_key)).fetchone()
            if row:
                record = self._operation_record(row)
                if row["inputs_json"] != inputs_json or row["kind"] != kind:
                    raise WorkflowConflictError("Operation identity has different frozen inputs")
                return record
            operation_id, now = uuid4().hex, self._clock_ns()
            db.execute("""INSERT INTO workflow_operations(operation_id,run_id,operation_key,kind,state,
              inputs_json,inputs_sha,consumed_json,created,updated) VALUES (?,?,?,?,'started',?,?,'[]',?,?)""",
                       (operation_id, lease.run_id, operation_key, kind, inputs_json, _sha(inputs_json), now, now))
            return self._operation_record(db.execute("SELECT * FROM workflow_operations WHERE operation_id=?", (operation_id,)).fetchone())

    def _consumable_calls(self, db, run_id, call_ids):
        for call_id in call_ids:
            call = self._call(db, call_id)
            if call["run_id"] != run_id:
                raise WorkflowConflictError("Consumed call belongs to another workflow")
            if call["billing_state"] != "known" or call["result_state"] not in ("accepted", "rejected", "applied"):
                raise WorkflowStateError("Cannot consume a call before billing and semantic settlement")

    def complete_operation(self, lease, operation_key, result, consumed_call_ids=()):
        if type(result) is not dict:
            raise WorkflowValidationError("Operation result must be a JSON object")
        result_json = _canonical(result)
        call_ids = self._ids(consumed_call_ids, "consumed_call_ids")
        with self._transaction() as db:
            self._lease(db, lease)
            row = db.execute("SELECT * FROM workflow_operations WHERE run_id=? AND operation_key=?", (lease.run_id, operation_key)).fetchone()
            if row is None:
                raise WorkflowNotFoundError(operation_key)
            self._operation_record(row)
            if row["state"] != "started":
                if row["result_json"] != result_json or row["consumed_json"] != _canonical(call_ids):
                    raise WorkflowConflictError("Operation already has a different committed result")
                return self._operation_record(row)
            self._consumable_calls(db, lease.run_id, call_ids)
            db.execute("UPDATE workflow_operations SET state='completed',result_json=?,result_sha=?,consumed_json=?,updated=? WHERE operation_id=?",
                       (result_json, _sha(result_json), _canonical(call_ids), self._clock_ns(), row["operation_id"]))
            return self._operation_record(db.execute("SELECT * FROM workflow_operations WHERE operation_id=?", (row["operation_id"],)).fetchone())

    def save_checkpoint(self, lease, expected_revision, checkpoint, *, consumed_call_ids=(),
                        consumed_operation_ids=(), transition_kind="checkpoint"):
        _integer(expected_revision, "expected_revision")
        _text(transition_kind, "transition_kind")
        if type(checkpoint) is not dict:
            raise WorkflowValidationError("Checkpoint must be a JSON object")
        checkpoint_json = _canonical(checkpoint)
        call_ids = set(self._ids(consumed_call_ids, "consumed_call_ids"))
        operation_ids = self._ids(consumed_operation_ids, "consumed_operation_ids")
        with self._transaction() as db:
            run = self._lease(db, lease)
            if run["revision"] != expected_revision:
                raise WorkflowConflictError("Checkpoint revision was superseded")
            for operation_id in operation_ids:
                row = db.execute("SELECT * FROM workflow_operations WHERE operation_id=?", (operation_id,)).fetchone()
                if row is None:
                    raise WorkflowNotFoundError(operation_id)
                operation = self._operation_record(row)
                if row["run_id"] != lease.run_id:
                    raise WorkflowConflictError("Consumed operation belongs to another workflow")
                if operation["state"] == "started":
                    raise WorkflowStateError("Cannot consume an unfinished operation")
                call_ids.update(operation["consumed_call_ids"])
            self._consumable_calls(db, lease.run_id, call_ids)
            revision = expected_revision + 1
            db.execute("INSERT INTO workflow_checkpoints VALUES (?,?,?,?,?,?,?,?)",
                       (lease.run_id, revision, checkpoint_json, _sha(checkpoint_json), _canonical(sorted(call_ids)),
                        _canonical(operation_ids), transition_kind, self._clock_ns()))
            for call_id in call_ids:
                # Rejected native output remains rejected on replay; its charge
                # is still consumed by the checkpoint's immutable ID list.
                db.execute("UPDATE workflow_calls SET result_state='applied' WHERE call_id=? AND result_state='accepted'", (call_id,))
            for operation_id in operation_ids:
                db.execute("UPDATE workflow_operations SET state='applied' WHERE operation_id=?", (operation_id,))
            if checkpoint.get("status") == "completed":
                if run["blocked_reason"] or db.execute("""SELECT 1 FROM workflow_calls WHERE run_id=? AND
                  (state IN ('reserved','dispatched','response_saved','unknown') OR
                  (billing_state='known' AND result_state IN ('pending','accepted'))) LIMIT 1""", (lease.run_id,)).fetchone():
                    raise WorkflowStateError("Cannot complete a workflow with unresolved calls or unapplied output")
                db.execute("UPDATE workflow_runs SET status='completed' WHERE run_id=?", (lease.run_id,))
            db.execute("UPDATE workflow_runs SET revision=?,updated=? WHERE run_id=?", (revision, self._clock_ns(), lease.run_id))
            self._event(db, lease.run_id, "checkpoint_saved", payload={"revision": revision, "kind": transition_kind})
            return {"run_id": lease.run_id, "revision": revision, "checkpoint": _loads(checkpoint_json),
                    "checkpoint_sha256": _sha(checkpoint_json), "consumed_call_ids": sorted(call_ids),
                    "consumed_operation_ids": operation_ids, "transition_kind": transition_kind}

    @staticmethod
    def _checkpoint_record(row):
        checkpoint = _loads(row["checkpoint_json"], stored=True)
        if type(checkpoint) is not dict or _sha(row["checkpoint_json"]) != row["checkpoint_sha"]:
            raise WorkflowCorruptionError("Checkpoint content hash is corrupt")
        calls, operations = _loads(row["consumed_json"], stored=True), _loads(row["operations_json"], stored=True)
        try:
            if (SQLiteWorkflowStore._ids(calls, "call_ids") != calls
                    or SQLiteWorkflowStore._ids(operations, "operation_ids") != operations):
                raise WorkflowValidationError("Invalid ID ordering")
        except WorkflowValidationError as exc:
            raise WorkflowCorruptionError("Invalid checkpoint consumption IDs") from exc
        return {"run_id": row["run_id"], "revision": row["revision"], "checkpoint": checkpoint,
                "checkpoint_sha256": row["checkpoint_sha"], "consumed_call_ids": calls,
                "consumed_operation_ids": operations, "transition_kind": row["kind"]}

    def get_checkpoint(self, run_id):
        with self._transaction(write=False) as db:
            run = self._run(db, run_id)
            row = db.execute("SELECT * FROM workflow_checkpoints WHERE run_id=? ORDER BY revision DESC LIMIT 1", (run_id,)).fetchone()
            if row is None:
                if run["revision"] != 0:
                    raise WorkflowCorruptionError("Workflow checkpoint revision is missing")
                return None
            if row["revision"] != run["revision"]:
                raise WorkflowCorruptionError("Workflow checkpoint revision is inconsistent")
            return self._checkpoint_record(row)

    def _audit(self, db, run_id):
        run = self._run(db, run_id)
        totals = [0, 0, 0]
        calls = db.execute("SELECT * FROM workflow_calls WHERE run_id=? ORDER BY created,call_id", (run_id,)).fetchall()
        for call in calls:
            record = self._call_record(call)
            cost = record["cost_nanousd"]
            held, unknown = record["reserved_nanousd"], record["unknown_nanousd"]
            if record["state"] not in ("prepared", "quoted", "reserved", "dispatched", "response_saved", "settled", "unknown", "released"):
                raise WorkflowCorruptionError("Invalid call state")
            if record["result_state"] not in ("pending", "accepted", "rejected", "applied"):
                raise WorkflowCorruptionError("Invalid call result state")
            if record["billing_state"] not in ("unquoted", "quoted", "reserved", "known", "unknown", "released"):
                raise WorkflowCorruptionError("Invalid billing state")
            if record["billing_state"] == "known":
                if cost is None or held or unknown or record["state"] != "settled" or not record["receipt"]:
                    raise WorkflowCorruptionError("Known call accounting is inconsistent")
                if str(cost) != str(record["receipt"].get("cost_nanousd")):
                    raise WorkflowCorruptionError("Known receipt disagrees with exact cost")
                totals[0] += cost
            elif cost is not None:
                raise WorkflowCorruptionError("Unsettled call has a fabricated cost")
            if held and record["state"] not in ("reserved", "dispatched", "response_saved"):
                raise WorkflowCorruptionError("Reservation exists outside an admitted call")
            if unknown and record["billing_state"] != "unknown":
                raise WorkflowCorruptionError("Unknown exposure is not latched on its call")
            if record["billing_state"] == "unknown" and record["state"] != "unknown":
                raise WorkflowCorruptionError("Unknown call state is inconsistent")
            if record["result_state"] != "pending" and record["billing_state"] != "known":
                raise WorkflowCorruptionError("Output is accepted without known billing")
            totals[1] += held
            totals[2] += unknown
            if call["quote_id"]:
                evidence = self._evidence(db, call, call["quote_id"], "input_tokens")
                try:
                    quote = self._quote(call, evidence)
                except WorkflowValidationError as exc:
                    raise WorkflowCorruptionError("Accepted quote evidence is invalid") from exc
                if record["ceiling_nanousd"] != quote["ceiling_nanousd"]:
                    raise WorkflowCorruptionError("Reserved quote was altered")
            if call["evidence_id"]:
                self._evidence(db, call, call["evidence_id"], "response")
            for evidence in db.execute("SELECT * FROM workflow_evidence WHERE call_id=?", (call["call_id"],)):
                self._evidence_record(evidence)
        if tuple(totals) != self._balances(run):
            raise WorkflowCorruptionError("Run aggregate balances disagree with call ledger")
        unknown_calls = sum(call["billing_state"] == "unknown" for call in calls)
        if unknown_calls and not run["blocked_reason"]:
            raise WorkflowCorruptionError("Unknown exposure has no admission latch")
        if run["budget"] is not None and sum(totals) > _amount(run["budget"]) and not run["blocked_reason"]:
            raise WorkflowCorruptionError("Budget overrun has no admission latch")
        checkpoints = db.execute("SELECT * FROM workflow_checkpoints WHERE run_id=? ORDER BY revision", (run_id,)).fetchall()
        if [row["revision"] for row in checkpoints] != list(range(1, run["revision"] + 1)):
            raise WorkflowCorruptionError("Checkpoint history is incomplete")
        for row in checkpoints:
            checkpoint = self._checkpoint_record(row)
            self._consumable_calls(db, run_id, checkpoint["consumed_call_ids"])
            for operation_id in checkpoint["consumed_operation_ids"]:
                operation = db.execute("SELECT * FROM workflow_operations WHERE operation_id=? AND run_id=?", (operation_id, run_id)).fetchone()
                if operation is None or self._operation_record(operation)["state"] != "applied":
                    raise WorkflowCorruptionError("Checkpoint operation is missing or not applied")
        for operation in db.execute("SELECT * FROM workflow_operations WHERE run_id=?", (run_id,)):
            value = self._operation_record(operation)
            self._consumable_calls(db, run_id, value["consumed_call_ids"])
        return {"ok": True, "call_count": len(calls), "confirmed_nanousd": totals[0],
                "reserved_nanousd": totals[1], "unknown_nanousd": totals[2], "unknown_calls": unknown_calls,
                "revision": run["revision"]}

    def audit(self, run_id):
        with self._transaction(write=False) as db:
            return self._audit(db, run_id)

    def inspect_run(self, run_id):
        with self._transaction(write=False) as db:
            audit = self._audit(db, run_id)
            run = self._run(db, run_id)
            calls = []
            for row in db.execute("SELECT * FROM workflow_calls WHERE run_id=? ORDER BY created,call_id", (run_id,)):
                record = self._call_record(row)
                calls.append({key: record[key] for key in (
                    "call_id", "key", "binding_sha256", "request_sha256", "state", "billing_state", "result_state",
                    "quote_id", "evidence_id", "ceiling_nanousd", "cost_nanousd", "reserved_nanousd", "unknown_nanousd")})
            return {**audit, "store_id": self.store_id, "run_id": run_id, "status": run["status"],
                    "blocked_reason": run["blocked_reason"], "budget_nanousd": None if run["budget"] is None else _amount(run["budget"]),
                    "config_sha256": _sha(run["config_json"]), "task_sha256": _sha(run["task_json"]), "calls": calls,
                    "events": [{"sequence": row["sequence"], "call_id": row["call_id"], "type": row["event_type"],
                                "data": _loads(row["payload_json"], stored=True)}
                               for row in db.execute("SELECT * FROM workflow_events WHERE run_id=? ORDER BY sequence", (run_id,))]}

    def recover(self, lease):
        with self._transaction() as db:
            self._lease(db, lease)
            self._audit(db, lease.run_id)
            calls = db.execute("SELECT * FROM workflow_calls WHERE run_id=? ORDER BY created,call_id", (lease.run_id,)).fetchall()
            for call in calls:
                if call["state"] == "dispatched" and call["evidence_id"] is None:
                    self._unknown(db, self._run(db, lease.run_id), call, "recovered_after_dispatch")
            plan = {"undispatched": [], "settle": [], "decode": [], "apply": [], "unknown": [], "applied": []}
            for call in db.execute("SELECT * FROM workflow_calls WHERE run_id=? ORDER BY created,call_id", (lease.run_id,)):
                if call["state"] in ("prepared", "quoted", "reserved", "released"):
                    bucket = "undispatched"
                elif call["result_state"] == "applied":
                    bucket = "applied"
                elif call["result_state"] in ("accepted", "rejected"):
                    bucket = "apply"
                elif call["billing_state"] == "known":
                    bucket = "decode"
                elif call["evidence_id"]:
                    bucket = "settle"
                else:
                    bucket = "unknown"
                plan[bucket].append(call["call_id"])
            run = self._run(db, lease.run_id)
            return {"run_id": lease.run_id, "revision": run["revision"], "blocked_reason": run["blocked_reason"], **plan}
