"""SQLite dispatch journal for durable, selectively rerunnable jobs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from smythe.jobs.artifact_io import MAX_ARTIFACT_BYTES, MAX_IMAGE_PIXELS
from smythe.jobs.preflight import JobApprovalV1, JobPlanV1, verify_approval


STORE_VERSION = 2
DEFAULT_RUN_LEASE_TTL_S = 30.0
MAX_SQLITE_INTEGER = (1 << 63) - 1
MAX_ARTIFACTS_PER_CALL = 10
MAX_LIST_RUNS = 500
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Jobs predates an explicit format-kind table. Verify its complete table
# shape and user_version before opening it for inspection or migration.
_SCHEMA_COLUMNS = {
    "runs": "run_id manifest_hash plan_hash manifest_json plan_json approval_json approval_token manifest_root output_directory max_concurrency status approved_microusd confirmed_microusd exposure_microusd reserved_microusd created_at_ns updated_at_ns",
    "operations": "run_id operation_id operation_key spec_json status attempt_count max_attempts accepted_attempt_id result_text error updated_at_ns",
    "attempts": "attempt_id run_id operation_id attempt_number parent_attempt_id reason status result_text error started_at_ns completed_at_ns",
    "calls": "call_id attempt_id run_id operation_id status idempotency_key ceiling_microusd confirmed_microusd exposure_microusd cost_is_complete cost_is_estimate provider_request_id error created_at_ns dispatched_at_ns completed_at_ns",
    "artifacts": "artifact_id attempt_id run_id operation_id relative_path mime_type sha256 size_bytes width height accepted created_at_ns",
    "events": "sequence run_id operation_id event_type payload_json created_at_ns",
    "run_leases": "run_id owner_id acquired_at_ns heartbeat_at_ns expires_at_ns",
}


class RunStatus(str, Enum):
    APPROVED = "approved"
    RUNNING = "running"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BUDGET_OVERRUN = "budget_overrun"


class OperationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class RunStoreError(RuntimeError):
    """Base error for job persistence and state transitions."""


class JobNotFoundError(RunStoreError, KeyError):
    """Raised when a run identifier is unknown."""


class InvalidTransitionError(RunStoreError):
    """Raised when a state transition would violate the dispatch journal."""


class JobBudgetError(RunStoreError):
    """Raised before dispatch when the approved exposure would be exceeded."""


class _BudgetAdmissionClosed(JobBudgetError):
    """Internal signal whose newly written overrun latch must be committed."""


class RunLeaseError(RunStoreError):
    """Raised when another worker owns an unexpired run lease."""


@dataclass(frozen=True, slots=True)
class CallPermit:
    """A durable local dispatch permit.

    ``idempotency_key`` identifies a logical call in Smythe's local journal.
    It is not an assertion that a provider endpoint supports remote
    idempotency, and Jobs does not transmit it unless an adapter explicitly
    implements a documented provider mechanism.
    """

    call_id: str
    attempt_id: str
    idempotency_key: str
    ceiling_microusd: int


@dataclass(frozen=True, slots=True)
class RunLease:
    """One renewable, crash-expiring ownership claim for a durable run."""

    run_id: str
    owner_id: str
    acquired_at_ns: int
    heartbeat_at_ns: int
    expires_at_ns: int


def _lease_duration_ns(ttl_s: float) -> int:
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)):
        raise TypeError("lease ttl_s must be a number")
    ttl = float(ttl_s)
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("lease ttl_s must be finite and positive")
    return max(1, int(ttl * 1_000_000_000))


def _lease_owner(owner_id: str) -> str:
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("lease owner_id must be a non-empty string")
    owner = owner_id.strip()
    if len(owner) > 256:
        raise ValueError("lease owner_id must be at most 256 characters")
    return owner


def _safe_run_id(run_id: object) -> str:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or _RUN_ID_RE.fullmatch(run_id) is None
    ):
        raise ValueError(
            "run_id must be a 1-128 character safe path component beginning "
            "with an alphanumeric character"
        )
    return run_id


def _microusd(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0 or value > MAX_SQLITE_INTEGER:
        raise ValueError(
            f"{field} must be between 0 and {MAX_SQLITE_INTEGER}"
        )
    return value


def _run_lease(row: sqlite3.Row) -> RunLease:
    return RunLease(
        run_id=row["run_id"],
        owner_id=row["owner_id"],
        acquired_at_ns=row["acquired_at_ns"],
        heartbeat_at_ns=row["heartbeat_at_ns"],
        expires_at_ns=row["expires_at_ns"],
    )


def _read_json_object(value: object, field: str) -> dict[str, Any]:
    """Decode stored JSON without allowing ambiguous or nonfinite output."""
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = item
        return result

    def constant(_value):
        raise ValueError("nonfinite JSON constant")

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    try:
        if type(value) is not str:
            raise ValueError("stored JSON is not text")
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
        if type(result) is not dict:
            raise ValueError("stored JSON root is not an object")
        finite(result)
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise RunStoreError("Invalid stored JSON in " + field) from exc


def _validate_read_row(row, table: str) -> None:
    """SQLite affinity is not a type guarantee; never emit BLOBs as JSON."""
    integers = {
        "max_concurrency", "approved_microusd", "confirmed_microusd", "exposure_microusd",
        "reserved_microusd", "created_at_ns", "updated_at_ns", "attempt_count", "max_attempts",
        "attempt_number", "started_at_ns", "completed_at_ns", "ceiling_microusd", "cost_is_complete",
        "cost_is_estimate", "dispatched_at_ns", "size_bytes", "width", "height", "accepted", "sequence",
        "cost_contains_estimates", "count", "acquired_at_ns", "heartbeat_at_ns", "expires_at_ns",
    }
    nullable = {
        "runs": set(),
        "operations": {"accepted_attempt_id", "result_text", "error"},
        "attempts": {"parent_attempt_id", "reason", "result_text", "error", "completed_at_ns"},
        "calls": {"provider_request_id", "error", "dispatched_at_ns", "completed_at_ns"},
        "artifacts": {"width", "height"},
        "events": {"operation_id"},
        "run_leases": set(),
    }
    flags = {"cost_is_complete", "cost_is_estimate", "accepted", "cost_contains_estimates"}
    for field, value in dict(row).items():
        if value is None and field in nullable[table]:
            continue
        if field in integers:
            if type(value) is not int or not 0 <= value <= MAX_SQLITE_INTEGER or field in flags and value not in (0, 1):
                raise RunStoreError("Invalid stored integer in " + table + "." + field)
        elif type(value) is not str:
            raise RunStoreError("Invalid stored text in " + table + "." + field)


class SQLiteRunStore:
    """Incremental local run state with a durable pre-dispatch boundary."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        if type(read_only) is not bool:
            raise TypeError("read_only must be a boolean")
        self.path = Path(path).resolve()
        self.read_only = read_only
        self._lock = threading.RLock()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro" if read_only else self.path,
                uri=read_only, isolation_level=None, check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise RunStoreError("Cannot open Jobs database") from exc
        self._connection.row_factory = sqlite3.Row
        try:
            # Connection-local settings do not alter database bytes. In
            # particular, never use immutable=1: inspectors must see live WAL.
            if read_only:
                self._connection.execute("PRAGMA query_only = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            existing = self._validate_schema(allow_empty=not read_only)
            self._connection.execute("PRAGMA foreign_keys = ON")
            if not read_only:
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
                if not existing or self._connection.execute("PRAGMA user_version").fetchone()[0] < STORE_VERSION:
                    self._create_schema()
        except sqlite3.Error as exc:
            self._connection.close()
            raise RunStoreError("Invalid or unreadable Jobs database schema") from exc
        except BaseException:
            self._connection.close()
            raise

    def _validate_schema(self, *, allow_empty: bool) -> bool:
        with self._read_transaction() as cursor:
            tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = cursor.execute("PRAGMA user_version").fetchone()[0]
            if not tables and version == 0 and allow_empty:
                return False
            supported_versions = (1, STORE_VERSION) if not self.read_only else (STORE_VERSION,)
            if version not in supported_versions:
                raise RunStoreError("Unsupported Jobs database version")
            expected = dict(_SCHEMA_COLUMNS)
            if version == 1 and not self.read_only:
                expected.pop("run_leases")  # Preserve the existing writable v1 upgrade.
            if not expected.keys() <= tables:
                raise RunStoreError("Database does not contain the required Jobs schema")
            for table, columns in expected.items():
                found = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
                if not set(columns.split()) <= found:
                    raise RunStoreError("Jobs database has an incomplete " + table + " table")
        return True

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteRunStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                manifest_hash TEXT NOT NULL,
                plan_hash TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                approval_json TEXT NOT NULL,
                approval_token TEXT NOT NULL,
                manifest_root TEXT NOT NULL,
                output_directory TEXT NOT NULL,
                max_concurrency INTEGER NOT NULL,
                status TEXT NOT NULL,
                approved_microusd INTEGER NOT NULL,
                confirmed_microusd INTEGER NOT NULL DEFAULT 0,
                exposure_microusd INTEGER NOT NULL DEFAULT 0,
                reserved_microusd INTEGER NOT NULL DEFAULT 0,
                created_at_ns INTEGER NOT NULL,
                updated_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS operations (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                operation_id TEXT NOT NULL,
                operation_key TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL,
                accepted_attempt_id TEXT,
                result_text TEXT,
                error TEXT,
                updated_at_ns INTEGER NOT NULL,
                PRIMARY KEY (run_id, operation_id),
                UNIQUE (run_id, operation_key)
            );

            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                parent_attempt_id TEXT,
                reason TEXT,
                status TEXT NOT NULL,
                result_text TEXT,
                error TEXT,
                started_at_ns INTEGER NOT NULL,
                completed_at_ns INTEGER,
                FOREIGN KEY (run_id, operation_id)
                    REFERENCES operations(run_id, operation_id) ON DELETE CASCADE,
                UNIQUE (run_id, operation_id, attempt_number)
            );

            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
                run_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                ceiling_microusd INTEGER NOT NULL,
                confirmed_microusd INTEGER NOT NULL DEFAULT 0,
                exposure_microusd INTEGER NOT NULL DEFAULT 0,
                cost_is_complete INTEGER NOT NULL DEFAULT 1,
                cost_is_estimate INTEGER NOT NULL DEFAULT 0,
                provider_request_id TEXT,
                error TEXT,
                created_at_ns INTEGER NOT NULL,
                dispatched_at_ns INTEGER,
                completed_at_ns INTEGER
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
                run_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                accepted INTEGER NOT NULL,
                created_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                operation_id TEXT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_leases (
                run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
                owner_id TEXT NOT NULL,
                acquired_at_ns INTEGER NOT NULL,
                heartbeat_at_ns INTEGER NOT NULL,
                expires_at_ns INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS operations_by_status
                ON operations(run_id, status);
            CREATE INDEX IF NOT EXISTS attempts_by_operation
                ON attempts(run_id, operation_id, attempt_number);
            CREATE INDEX IF NOT EXISTS calls_by_status
                ON calls(run_id, status);
            CREATE INDEX IF NOT EXISTS artifacts_by_operation
                ON artifacts(run_id, operation_id, accepted);
            CREATE INDEX IF NOT EXISTS run_leases_by_expiry
                ON run_leases(expires_at_ns);
            PRAGMA user_version = 2;
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        if self.read_only:
            raise RunStoreError("Jobs database is read-only")
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                yield cursor
            except _BudgetAdmissionClosed:
                # `_assert_budget_open` may have reconstructed an overrun from
                # durable call rows. Commit that terminal latch while still
                # rejecting the attempted admission to the caller.
                cursor.execute("COMMIT")
                raise
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Cursor]:
        """Hold one WAL snapshot across a multi-query read projection."""

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN")
            try:
                yield cursor
            except sqlite3.Error as exc:
                cursor.execute("ROLLBACK")
                raise RunStoreError("Cannot read Jobs database") from exc
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")

    def create_run(
        self,
        plan: JobPlanV1,
        approval: JobApprovalV1,
        *,
        manifest_root: str | Path,
        run_id: str | None = None,
    ) -> str:
        verify_approval(plan, approval)
        approved_microusd = _microusd(
            approval.approved_max_cost_microusd,
            field="approved_max_cost_microusd",
        )
        run_id = _safe_run_id(uuid4().hex if run_id is None else run_id)
        now = time.time_ns()
        plan_json = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))
        approval_json = json.dumps(
            approval.to_dict(), sort_keys=True, separators=(",", ":")
        )
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id, manifest_hash, plan_hash, manifest_json, plan_json,
                    approval_json, approval_token, manifest_root, output_directory,
                    max_concurrency, status, approved_microusd,
                    created_at_ns, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    plan.manifest_hash,
                    plan.plan_hash,
                    plan.manifest_json,
                    plan_json,
                    approval_json,
                    approval.token,
                    str(Path(manifest_root).resolve()),
                    plan.output_directory,
                    plan.max_concurrency,
                    RunStatus.APPROVED.value,
                    approved_microusd,
                    now,
                    now,
                ),
            )
            cursor.executemany(
                """
                INSERT INTO operations (
                    run_id, operation_id, operation_key, spec_json, status,
                    max_attempts, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        operation.operation_id,
                        operation.operation_key,
                        json.dumps(
                            operation.to_dict(), sort_keys=True, separators=(",", ":")
                        ),
                        OperationStatus.PENDING.value,
                        operation.max_attempts,
                        now,
                    )
                    for operation in plan.operations
                ],
            )
            self._event(cursor, run_id, None, "run_created", plan.to_dict(), now)
        return run_id

    def acquire_run_lease(
        self,
        run_id: str,
        owner_id: str,
        *,
        ttl_s: float = DEFAULT_RUN_LEASE_TTL_S,
    ) -> RunLease:
        """Atomically acquire or renew exclusive ownership of a run.

        A different owner may take over only after the previous heartbeat has
        expired. The lease is stored in SQLite so separate CLI processes obey
        the same ownership boundary.
        """

        owner = _lease_owner(owner_id)
        duration_ns = _lease_duration_ns(ttl_s)
        now = time.time_ns()
        expires = now + duration_ns
        with self._transaction() as cursor:
            if cursor.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                raise JobNotFoundError(run_id)
            current = cursor.execute(
                "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
            if (
                current is not None
                and current["owner_id"] != owner
                and current["expires_at_ns"] > now
            ):
                raise RunLeaseError(
                    f"run {run_id!r} is leased by {current['owner_id']!r}"
                )

            same_active_owner = (
                current is not None
                and current["owner_id"] == owner
                and current["expires_at_ns"] > now
            )
            acquired_at = current["acquired_at_ns"] if same_active_owner else now
            previous_owner = current["owner_id"] if current is not None else None
            cursor.execute(
                """
                INSERT INTO run_leases (
                    run_id, owner_id, acquired_at_ns, heartbeat_at_ns, expires_at_ns
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    acquired_at_ns = excluded.acquired_at_ns,
                    heartbeat_at_ns = excluded.heartbeat_at_ns,
                    expires_at_ns = excluded.expires_at_ns
                """,
                (run_id, owner, acquired_at, now, expires),
            )
            row = cursor.execute(
                "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            self._event(
                cursor,
                run_id,
                None,
                "run_lease_renewed" if same_active_owner else "run_lease_acquired",
                {
                    "owner_id": owner,
                    "previous_owner_id": previous_owner,
                    "expires_at_ns": expires,
                },
                now,
            )
            return _run_lease(row)

    def heartbeat_run_lease(
        self,
        run_id: str,
        owner_id: str,
        *,
        ttl_s: float = DEFAULT_RUN_LEASE_TTL_S,
    ) -> RunLease:
        """Extend a live lease, failing if it expired or changed owners."""

        owner = _lease_owner(owner_id)
        duration_ns = _lease_duration_ns(ttl_s)
        now = time.time_ns()
        expires = now + duration_ns
        with self._transaction() as cursor:
            self._assert_run_lease(cursor, run_id, owner, now)
            cursor.execute(
                """UPDATE run_leases SET heartbeat_at_ns = ?, expires_at_ns = ?
                   WHERE run_id = ? AND owner_id = ?""",
                (now, expires, run_id, owner),
            )
            row = cursor.execute(
                "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            return _run_lease(row)

    def release_run_lease(self, run_id: str, owner_id: str) -> bool:
        """Release a run lease. Missing leases are an idempotent no-op."""

        owner = _lease_owner(owner_id)
        now = time.time_ns()
        with self._transaction() as cursor:
            current = cursor.execute(
                "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
            if current is None:
                return False
            if current["owner_id"] != owner:
                raise RunLeaseError(
                    f"run {run_id!r} is leased by {current['owner_id']!r}"
                )
            cursor.execute("DELETE FROM run_leases WHERE run_id = ?", (run_id,))
            self._event(
                cursor,
                run_id,
                None,
                "run_lease_released",
                {"owner_id": owner},
                now,
            )
            return True

    def get_run_lease(self, run_id: str) -> RunLease | None:
        """Return the persisted lease, including an expired lease if present."""

        with self._read_transaction() as cursor:
            if cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise JobNotFoundError(run_id)
            row = cursor.execute(
                "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _run_lease(row) if row is not None else None

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(run_id)
        return dict(row)

    def pending_operations(self, run_id: str) -> list[dict[str, Any]]:
        with self._read_transaction() as cursor:
            if cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise JobNotFoundError(run_id)
            rows = cursor.execute(
                """
                SELECT * FROM operations
                WHERE run_id = ? AND status = ?
                ORDER BY operation_key
                """,
                (run_id, OperationStatus.PENDING.value),
            ).fetchall()
        return [dict(row) | {"spec": json.loads(row["spec_json"])} for row in rows]

    def list_runs(
        self, *, limit: int = 50, offset: int = 0, status: RunStatus | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return one bounded page, newest creation first, then run ID.

        Pagination is offset based and each call observes a single WAL
        snapshot. New runs inserted between pages can shift their offsets.
        Prompts, operation specifications, results and artifact rows are not
        loaded. ``limit`` is 1..500; ``offset`` is a nonnegative SQLite integer.
        """
        return self.list_run_page(limit=limit, offset=offset, status=status)["runs"]

    def list_run_page(
        self, *, limit: int = 50, offset: int = 0, status: RunStatus | str | None = None,
    ) -> dict[str, Any]:
        """Return bounded summaries and has_more from the same read snapshot."""
        if type(limit) is not int or not 1 <= limit <= MAX_LIST_RUNS:
            raise ValueError(f"limit must be an integer between 1 and {MAX_LIST_RUNS}")
        if type(offset) is not int or not 0 <= offset <= MAX_SQLITE_INTEGER:
            raise ValueError("offset must be a nonnegative SQLite integer")
        if isinstance(status, RunStatus):
            status = status.value
        if status is not None and (type(status) is not str or status not in {item.value for item in RunStatus}):
            raise ValueError("status must be a Jobs run status")
        where = "WHERE status = ?" if status is not None else ""
        parameters = (status, limit + 1, offset) if status is not None else (limit + 1, offset)
        try:
            with self._read_transaction() as cursor:
                runs = cursor.execute(f"""
                    SELECT run_id,status,created_at_ns,updated_at_ns,
                        approved_microusd,confirmed_microusd,exposure_microusd,reserved_microusd
                    FROM runs {where} ORDER BY created_at_ns DESC,run_id ASC LIMIT ? OFFSET ?
                """, parameters).fetchall()
                summaries = []
                for run in runs[:limit]:
                    _validate_read_row(run, "runs")
                    detail = cursor.execute("""SELECT json_extract(plan_json,'$.name') AS name,
                        json_type(plan_json,'$.name') AS name_type,typeof(plan_json) AS plan_storage,
                        EXISTS(SELECT 1 FROM calls WHERE calls.run_id=runs.run_id AND cost_is_estimate=1)
                            AS cost_contains_estimates FROM runs WHERE run_id=?""", (run["run_id"],)).fetchone()
                    if detail["name_type"] != "text" or detail["plan_storage"] != "text" or run["status"] not in {item.value for item in RunStatus}:
                        raise RunStoreError("Jobs run summary contains invalid name or status")
                    for field in ("created_at_ns", "updated_at_ns", "approved_microusd", "confirmed_microusd",
                                  "exposure_microusd", "reserved_microusd"):
                        try:
                            _microusd(run[field], field=field)
                        except (TypeError, ValueError) as exc:
                            raise RunStoreError("Jobs run summary contains invalid numeric data") from exc
                    counts = {row["status"]: row["count"] for row in cursor.execute(
                        "SELECT status,COUNT(*) AS count FROM operations WHERE run_id=? GROUP BY status", (run["run_id"],))}
                    if not counts.keys() <= {item.value for item in OperationStatus}:
                        raise RunStoreError("Jobs run has invalid operation status")
                    summaries.append({
                        "run_id": run["run_id"], "name": detail["name"], "status": run["status"],
                        "created_at_ns": run["created_at_ns"], "updated_at_ns": run["updated_at_ns"],
                        "operation_count": sum(counts.values()), "counts": counts,
                        "cost": {key: run[key] for key in ("approved_microusd", "confirmed_microusd",
                                 "exposure_microusd", "reserved_microusd")} | {
                            "cost_is_complete": run["exposure_microusd"] == 0,
                            "cost_contains_estimates": bool(detail["cost_contains_estimates"]),
                        },
                    })
                return {"runs": summaries, "limit": limit, "offset": offset,
                        "returned": len(summaries), "has_more": len(runs) > limit}
        except sqlite3.Error as exc:
            raise RunStoreError("Cannot read Jobs run summaries") from exc

    def begin_attempt(
        self,
        run_id: str,
        operation_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        now = time.time_ns()
        with self._transaction() as cursor:
            self._assert_budget_open(cursor, run_id, now)
            operation = cursor.execute(
                """SELECT * FROM operations
                   WHERE run_id = ? AND operation_id = ?""",
                (run_id, operation_id),
            ).fetchone()
            if operation is None:
                raise JobNotFoundError(f"{run_id}:{operation_id}")
            if operation["status"] != OperationStatus.PENDING.value:
                raise InvalidTransitionError(
                    f"operation {operation_id} is {operation['status']}, not pending"
                )
            consumed_attempt_count = operation["attempt_count"] + 1
            if consumed_attempt_count > operation["max_attempts"]:
                raise InvalidTransitionError(
                    f"operation {operation_id} exhausted its attempt allowance"
                )
            previous = cursor.execute(
                """SELECT attempt_id, attempt_number FROM attempts
                   WHERE run_id = ? AND operation_id = ?
                   ORDER BY attempt_number DESC LIMIT 1""",
                (run_id, operation_id),
            ).fetchone()
            # ``attempt_count`` is the retry allowance consumed by provider-bound
            # work. Safe recovery before dispatch restores that allowance, but
            # historical attempt numbers remain immutable and therefore advance
            # independently.
            attempt_number = previous["attempt_number"] + 1 if previous is not None else 1
            attempt_id = uuid4().hex
            cursor.execute(
                """
                INSERT INTO attempts (
                    attempt_id, run_id, operation_id, attempt_number,
                    parent_attempt_id, reason, status, started_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    operation_id,
                    attempt_number,
                    previous["attempt_id"] if previous is not None else None,
                    reason,
                    OperationStatus.RUNNING.value,
                    now,
                ),
            )
            cursor.execute(
                """
                UPDATE operations SET status = ?, attempt_count = ?, error = NULL,
                    updated_at_ns = ? WHERE run_id = ? AND operation_id = ?
                """,
                (
                    OperationStatus.RUNNING.value,
                    consumed_attempt_count,
                    now,
                    run_id,
                    operation_id,
                ),
            )
            cursor.execute(
                """UPDATE runs SET status = ?, updated_at_ns = ?
                   WHERE run_id = ? AND status != ?""",
                (
                    RunStatus.RUNNING.value,
                    now,
                    run_id,
                    RunStatus.BUDGET_OVERRUN.value,
                ),
            )
            self._event(
                cursor,
                run_id,
                operation_id,
                "attempt_started",
                {"attempt_id": attempt_id, "attempt_number": attempt_number},
                now,
            )
        return {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "operation_id": operation_id,
        }

    def prepare_call(self, attempt_id: str, ceiling_microusd: int) -> CallPermit:
        ceiling_microusd = _microusd(
            ceiling_microusd, field="ceiling_microusd"
        )
        now = time.time_ns()
        with self._transaction() as cursor:
            attempt = cursor.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise JobNotFoundError(attempt_id)
            if attempt["status"] != OperationStatus.RUNNING.value:
                raise InvalidTransitionError("call requires a running attempt")
            run = self._assert_budget_open(cursor, attempt["run_id"], now)
            exposure = (
                run["confirmed_microusd"]
                + run["exposure_microusd"]
                + run["reserved_microusd"]
                + ceiling_microusd
            )
            if exposure > run["approved_microusd"]:
                raise JobBudgetError(
                    f"call would expose {exposure} micro-USD above the approved "
                    f"{run['approved_microusd']} micro-USD ceiling"
                )
            call_id = uuid4().hex
            idempotency_key = hashlib.sha256(
                (
                    "smythe.jobs.call.v1:"
                    f"{attempt['run_id']}:{attempt['operation_id']}:"
                    f"{attempt['attempt_number']}:1"
                ).encode()
            ).hexdigest()
            cursor.execute(
                """
                INSERT INTO calls (
                    call_id, attempt_id, run_id, operation_id, status,
                    idempotency_key, ceiling_microusd, created_at_ns
                ) VALUES (?, ?, ?, ?, 'prepared', ?, ?, ?)
                """,
                (
                    call_id,
                    attempt_id,
                    attempt["run_id"],
                    attempt["operation_id"],
                    idempotency_key,
                    ceiling_microusd,
                    now,
                ),
            )
            cursor.execute(
                """UPDATE runs SET reserved_microusd = reserved_microusd + ?,
                   updated_at_ns = ? WHERE run_id = ?""",
                (ceiling_microusd, now, attempt["run_id"]),
            )
            self._event(
                cursor,
                attempt["run_id"],
                attempt["operation_id"],
                "call_prepared",
                {"call_id": call_id, "ceiling_microusd": ceiling_microusd},
                now,
            )
        return CallPermit(call_id, attempt_id, idempotency_key, ceiling_microusd)

    def mark_call_dispatched(
        self, call_id: str, *, provider_request_id: str | None = None
    ) -> None:
        now = time.time_ns()
        with self._transaction() as cursor:
            call = self._call(cursor, call_id)
            if call["status"] != "prepared":
                raise InvalidTransitionError("only a prepared call can be dispatched")
            self._assert_budget_open(cursor, call["run_id"], now)
            cursor.execute(
                """UPDATE calls SET status = 'dispatched', provider_request_id = ?,
                   dispatched_at_ns = ? WHERE call_id = ?""",
                (provider_request_id, now, call_id),
            )
            self._event(
                cursor,
                call["run_id"],
                call["operation_id"],
                "call_dispatched",
                {"call_id": call_id},
                now,
            )

    def complete_call(
        self,
        call_id: str,
        *,
        cost_microusd: int,
        cost_is_complete: bool,
        cost_is_estimate: bool,
        artifacts: Sequence[dict[str, Any]],
        result_text: str,
        accepted: bool = True,
        error: str | None = None,
    ) -> None:
        cost_microusd = _microusd(cost_microusd, field="cost_microusd")
        if not isinstance(cost_is_complete, bool):
            raise TypeError("cost_is_complete must be a boolean")
        if not isinstance(cost_is_estimate, bool):
            raise TypeError("cost_is_estimate must be a boolean")
        if cost_is_complete and cost_is_estimate:
            raise ValueError("an estimated cost cannot be marked complete")
        if len(artifacts) > MAX_ARTIFACTS_PER_CALL:
            raise ValueError(
                f"a call may persist at most {MAX_ARTIFACTS_PER_CALL} artifacts"
            )
        for artifact in artifacts:
            size = _microusd(artifact.get("size_bytes"), field="artifact size_bytes")
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte persistence limit"
                )
            width = artifact.get("width")
            height = artifact.get("height")
            if (width is None) != (height is None):
                raise ValueError("artifact width and height must both be present or absent")
            if width is not None:
                if (
                    isinstance(width, bool)
                    or isinstance(height, bool)
                    or not isinstance(width, int)
                    or not isinstance(height, int)
                    or width < 1
                    or height < 1
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError("artifact dimensions exceed the persistence limits")
        now = time.time_ns()
        with self._transaction() as cursor:
            call = self._call(cursor, call_id)
            if call["status"] != "dispatched":
                raise InvalidTransitionError("only a dispatched call can complete")
            attempt = cursor.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (call["attempt_id"],)
            ).fetchone()
            run = cursor.execute(
                "SELECT * FROM runs WHERE run_id = ?", (call["run_id"],)
            ).fetchone()
            if run is None:  # pragma: no cover - protected by foreign keys
                raise JobNotFoundError(call["run_id"])
            if cost_is_complete:
                confirmed = cost_microusd
                exposure = 0
            else:
                confirmed = 0
                exposure = max(cost_microusd, call["ceiling_microusd"])
            call_status = "succeeded" if accepted else "rejected"
            operation_status = (
                OperationStatus.SUCCEEDED if accepted else OperationStatus.REJECTED
            )
            cursor.execute(
                """
                UPDATE calls SET status = ?, confirmed_microusd = ?,
                    exposure_microusd = ?, cost_is_complete = ?,
                    cost_is_estimate = ?, error = ?, completed_at_ns = ?
                WHERE call_id = ?
                """,
                (
                    call_status,
                    confirmed,
                    exposure,
                    int(cost_is_complete),
                    int(cost_is_estimate),
                    error,
                    now,
                    call_id,
                ),
            )
            cursor.execute(
                """
                UPDATE runs SET
                    reserved_microusd = ?,
                    confirmed_microusd = ?,
                    exposure_microusd = ?,
                    updated_at_ns = ? WHERE run_id = ?
                """,
                (
                    _microusd(
                        run["reserved_microusd"] - call["ceiling_microusd"],
                        field="run reserved cost",
                    ),
                    _microusd(
                        run["confirmed_microusd"] + confirmed,
                        field="run confirmed cost",
                    ),
                    _microusd(
                        run["exposure_microusd"] + exposure,
                        field="run exposure",
                    ),
                    now,
                    call["run_id"],
                ),
            )
            for artifact in artifacts:
                cursor.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id, attempt_id, run_id, operation_id,
                        relative_path, mime_type, sha256, size_bytes,
                        width, height, accepted, created_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.get("artifact_id", uuid4().hex),
                        attempt["attempt_id"],
                        call["run_id"],
                        call["operation_id"],
                        artifact["relative_path"],
                        artifact["mime_type"],
                        artifact["sha256"],
                        artifact["size_bytes"],
                        artifact.get("width"),
                        artifact.get("height"),
                        int(accepted),
                        now,
                    ),
                )
            cursor.execute(
                """UPDATE attempts SET status = ?, result_text = ?, error = ?,
                   completed_at_ns = ? WHERE attempt_id = ?""",
                (operation_status.value, result_text, error, now, attempt["attempt_id"]),
            )
            cursor.execute(
                """UPDATE operations SET status = ?, accepted_attempt_id = ?,
                   result_text = ?, error = ?, updated_at_ns = ?
                   WHERE run_id = ? AND operation_id = ?""",
                (
                    operation_status.value,
                    attempt["attempt_id"] if accepted else None,
                    result_text,
                    error,
                    now,
                    call["run_id"],
                    call["operation_id"],
                ),
            )
            updated_run = cursor.execute(
                "SELECT * FROM runs WHERE run_id = ?", (call["run_id"],)
            ).fetchone()
            assert updated_run is not None
            self._latch_budget_overrun(
                cursor,
                updated_run,
                now,
                evidenced=(
                    confirmed + exposure > call["ceiling_microusd"]
                    or updated_run["confirmed_microusd"]
                    + updated_run["exposure_microusd"]
                    > updated_run["approved_microusd"]
                ),
            )
            self._event(
                cursor,
                call["run_id"],
                call["operation_id"],
                "call_completed" if accepted else "artifact_rejected",
                {
                    "call_id": call_id,
                    "cost_microusd": cost_microusd,
                    "cost_is_complete": cost_is_complete,
                },
                now,
            )

    def fail_pre_dispatch(
        self, call_id: str, error: str, *, retryable: bool = False
    ) -> None:
        now = time.time_ns()
        with self._transaction() as cursor:
            call = self._call(cursor, call_id)
            if call["status"] != "prepared":
                raise InvalidTransitionError("call was already dispatched")
            self._fail_safe_call(cursor, call, error, now, retryable=retryable)

    def mark_unknown_outcome(self, call_id: str, error: str) -> None:
        now = time.time_ns()
        with self._transaction() as cursor:
            call = self._call(cursor, call_id)
            if call["status"] != "dispatched":
                raise InvalidTransitionError(
                    "unknown outcome requires a dispatched provider call"
                )
            self._unknown_call(cursor, call, error, now)

    def recover_inflight(
        self,
        run_id: str,
        *,
        lease_owner_id: str | None = None,
    ) -> dict[str, list[str]]:
        self.get_run(run_id)
        owner = _lease_owner(lease_owner_id) if lease_owner_id is not None else None
        safe: list[str] = []
        unknown: list[str] = []
        now = time.time_ns()
        with self._transaction() as cursor:
            if owner is not None:
                self._assert_run_lease(cursor, run_id, owner, now)
            else:
                active_lease = cursor.execute(
                    """SELECT owner_id FROM run_leases
                       WHERE run_id = ? AND expires_at_ns > ?""",
                    (run_id, now),
                ).fetchone()
                if active_lease is not None:
                    raise RunLeaseError(
                        f"run {run_id!r} is leased by {active_lease['owner_id']!r}; "
                        "recovery requires that lease owner"
                    )

            # A crash can land after the attempt transaction commits but before
            # its call is prepared. Such an attempt has no possible upstream
            # effect and is as safe to retry as a prepared-but-undispatched call.
            orphan_attempts = cursor.execute(
                """
                SELECT attempt.* FROM attempts AS attempt
                WHERE attempt.run_id = ? AND attempt.status = 'running'
                  AND NOT EXISTS (
                      SELECT 1 FROM calls AS call
                      WHERE call.attempt_id = attempt.attempt_id
                  )
                ORDER BY attempt.operation_id, attempt.attempt_number
                """,
                (run_id,),
            ).fetchall()
            for attempt in orphan_attempts:
                self._recover_orphan_attempt(cursor, attempt, now)
                safe.append(attempt["operation_id"])

            calls = cursor.execute(
                "SELECT * FROM calls WHERE run_id = ? AND status IN ('prepared','dispatched')",
                (run_id,),
            ).fetchall()
            for call in calls:
                if call["status"] == "prepared":
                    self._fail_safe_call(
                        cursor,
                        call,
                        "recovered before dispatch",
                        now,
                        retryable=True,
                    )
                    safe.append(call["operation_id"])
                else:
                    self._unknown_call(cursor, call, "process stopped after dispatch", now)
                    unknown.append(call["operation_id"])
            if unknown:
                cursor.execute(
                    """UPDATE runs SET status = CASE
                           WHEN status = ? THEN status ELSE ? END,
                           updated_at_ns = ? WHERE run_id = ?""",
                    (
                        RunStatus.BUDGET_OVERRUN.value,
                        RunStatus.NEEDS_ATTENTION.value,
                        now,
                        run_id,
                    ),
                )
        return {"safe_to_retry": safe, "unknown_outcome": unknown}

    def queue_reroll(
        self,
        run_id: str,
        operation_keys: Iterable[str],
        *,
        acknowledge_unknown: bool = False,
        reason: str,
        lease_owner_id: str | None = None,
    ) -> list[str]:
        keys = list(dict.fromkeys(operation_keys))
        if not keys:
            raise ValueError("at least one operation key is required")
        owner = _lease_owner(lease_owner_id) if lease_owner_id is not None else None
        queued: list[str] = []
        now = time.time_ns()
        with self._transaction() as cursor:
            self._assert_budget_open(cursor, run_id, now)
            if owner is not None:
                self._assert_run_lease(cursor, run_id, owner, now)
            else:
                active_lease = cursor.execute(
                    """SELECT owner_id FROM run_leases
                       WHERE run_id = ? AND expires_at_ns > ?""",
                    (run_id, now),
                ).fetchone()
                if active_lease is not None:
                    raise RunLeaseError(
                        f"run {run_id!r} is leased by {active_lease['owner_id']!r}; "
                        "reroll requires that lease owner"
                    )
            for key in keys:
                operation = cursor.execute(
                    """SELECT * FROM operations WHERE run_id = ?
                       AND (operation_key = ? OR operation_id = ?)""",
                    (run_id, key, key),
                ).fetchone()
                if operation is None:
                    raise JobNotFoundError(f"{run_id}:{key}")
                status = OperationStatus(operation["status"])
                allowed = {OperationStatus.FAILED, OperationStatus.REJECTED}
                if status is OperationStatus.UNKNOWN_OUTCOME and acknowledge_unknown:
                    allowed.add(OperationStatus.UNKNOWN_OUTCOME)
                if status not in allowed:
                    raise InvalidTransitionError(
                        f"operation {operation['operation_key']} in state {status.value} "
                        "cannot be rerolled"
                    )
                if operation["attempt_count"] >= operation["max_attempts"]:
                    raise InvalidTransitionError(
                        f"operation {operation['operation_key']} exhausted attempts"
                    )
                cursor.execute(
                    """UPDATE operations SET status = ?, error = NULL,
                       updated_at_ns = ? WHERE run_id = ? AND operation_id = ?""",
                    (
                        OperationStatus.PENDING.value,
                        now,
                        run_id,
                        operation["operation_id"],
                    ),
                )
                self._event(
                    cursor,
                    run_id,
                    operation["operation_id"],
                    "reroll_queued",
                    {"reason": reason, "acknowledged_unknown": acknowledge_unknown},
                    now,
                )
                queued.append(operation["operation_key"])
            cursor.execute(
                """UPDATE runs SET status = ?, updated_at_ns = ?
                   WHERE run_id = ? AND status != ?""",
                (
                    RunStatus.APPROVED.value,
                    now,
                    run_id,
                    RunStatus.BUDGET_OVERRUN.value,
                ),
            )
        return queued

    def finalize_run(self, run_id: str) -> str:
        now = time.time_ns()
        with self._transaction() as cursor:
            run = cursor.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise JobNotFoundError(run_id)
            if self._latch_budget_overrun(cursor, run, now):
                return RunStatus.BUDGET_OVERRUN.value
            rows = cursor.execute(
                """SELECT status, COUNT(*) AS count FROM operations
                   WHERE run_id = ? GROUP BY status""",
                (run_id,),
            ).fetchall()
            counts = {row["status"]: row["count"] for row in rows}
            if counts.get(OperationStatus.UNKNOWN_OUTCOME.value, 0):
                status = RunStatus.NEEDS_ATTENTION
            elif counts.get(OperationStatus.PENDING.value, 0) or counts.get(
                OperationStatus.RUNNING.value, 0
            ):
                status = RunStatus.RUNNING
            elif set(counts) <= {OperationStatus.SUCCEEDED.value}:
                status = RunStatus.COMPLETED
            elif counts.get(OperationStatus.SUCCEEDED.value, 0):
                status = RunStatus.PARTIAL
            else:
                status = RunStatus.FAILED
            cursor.execute(
                "UPDATE runs SET status = ?, updated_at_ns = ? WHERE run_id = ?",
                (status.value, now, run_id),
            )
            self._event(cursor, run_id, None, "run_finalized", {"status": status.value}, now)
        return status.value

    def snapshot(self, run_id: str, *, include_events: bool = False) -> dict[str, Any]:
        with self._read_transaction() as cursor:
            run_row = cursor.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                raise JobNotFoundError(run_id)
            operations = cursor.execute(
                "SELECT * FROM operations WHERE run_id = ? ORDER BY operation_key",
                (run_id,),
            ).fetchall()
            attempts = cursor.execute(
                """SELECT * FROM attempts WHERE run_id = ?
                   ORDER BY operation_id, attempt_number""",
                (run_id,),
            ).fetchall()
            artifacts = cursor.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY relative_path",
                (run_id,),
            ).fetchall()
            events = (
                cursor.execute(
                    "SELECT * FROM events WHERE run_id = ? ORDER BY sequence", (run_id,)
                ).fetchall()
                if include_events
                else []
            )
            call_rows = cursor.execute(
                "SELECT cost_is_estimate FROM calls WHERE run_id = ?", (run_id,)
            ).fetchall()
        return self._snapshot_projection(run_row, operations, attempts, artifacts, events, call_rows)

    @staticmethod
    def _snapshot_projection(run_row, operations, attempts, artifacts, events, call_rows):
        run = dict(run_row)
        _validate_read_row(run, "runs")
        for table, rows in (("operations", operations), ("attempts", attempts), ("artifacts", artifacts),
                            ("events", events), ("calls", call_rows)):
            for row in rows:
                _validate_read_row(row, table)
        name = run["name"] if "name" in run else _read_json_object(run["plan_json"], "runs.plan_json").get("name")
        if type(name) is not str:
            raise RunStoreError("Invalid stored job name")
        counts: dict[str, int] = {}
        for operation in operations:
            counts[operation["status"]] = counts.get(operation["status"], 0) + 1
        exposure = run["exposure_microusd"]
        return {
            "version": STORE_VERSION,
            "run_id": run["run_id"],
            "name": name,
            "status": run["status"],
            "manifest_hash": run["manifest_hash"],
            "plan_hash": run["plan_hash"],
            "manifest_root": run["manifest_root"],
            "output_directory": run["output_directory"],
            "max_concurrency": run["max_concurrency"],
            "counts": counts,
            "cost": {
                "approved_microusd": run["approved_microusd"],
                "confirmed_microusd": run["confirmed_microusd"],
                "exposure_microusd": exposure,
                "reserved_microusd": run["reserved_microusd"],
                "cost_is_complete": exposure == 0,
                "cost_contains_estimates": any(
                    row["cost_is_estimate"] for row in call_rows
                ),
            },
            "operations": [
                {
                    "operation_id": row["operation_id"],
                    "operation_key": row["operation_key"],
                    "status": row["status"],
                    "attempt_count": row["attempt_count"],
                    "max_attempts": row["max_attempts"],
                    "accepted_attempt_id": row["accepted_attempt_id"],
                    "result_text": row["result_text"],
                    "error": row["error"],
                    "spec": _read_json_object(row["spec_json"], "operations.spec_json"),
                }
                for row in operations
            ],
            "attempts": [dict(row) for row in attempts],
            "artifacts": [dict(row) for row in artifacts],
            "events": [
                dict(row) | {"payload": _read_json_object(row["payload_json"], "events.payload_json")}
                for row in events
            ],
            "created_at_ns": run["created_at_ns"],
            "updated_at_ns": run["updated_at_ns"],
        }

    def inspection_snapshot(
        self, run_id: str, *, operation: str | None = None, limit: int = 50,
        offset: int = 0, events_limit: int = 100,
    ) -> dict[str, Any]:
        """Page operations and their lineage within one live WAL snapshot.

        Summary counts/costs always describe the full run. Attempts, calls and
        artifacts contain the selected operations' complete local history;
        events contain the latest N for the run or exact operation filter,
        returned in chronological order. This does not read artifact files.
        """
        if type(limit) is not int or not 1 <= limit <= MAX_LIST_RUNS:
            raise ValueError(f"limit must be an integer between 1 and {MAX_LIST_RUNS}")
        if type(offset) is not int or not 0 <= offset <= MAX_SQLITE_INTEGER:
            raise ValueError("offset must be a nonnegative SQLite integer")
        if type(events_limit) is not int or not 1 <= events_limit <= 1000:
            raise ValueError("events_limit must be an integer between 1 and 1000")
        if operation is not None and (type(operation) is not str or not operation or len(operation) > 512):
            raise ValueError("operation must be a nonempty operation key or ID")
        try:
            with self._read_transaction() as cursor:
                run = cursor.execute("""SELECT run_id,json_extract(plan_json,'$.name') AS name,
                    json_type(plan_json,'$.name') AS name_type,typeof(plan_json) AS plan_storage,
                    status,manifest_hash,plan_hash,manifest_root,output_directory,max_concurrency,
                    approved_microusd,confirmed_microusd,exposure_microusd,reserved_microusd,
                    created_at_ns,updated_at_ns FROM runs WHERE run_id=?""", (run_id,)).fetchone()
                if run is None:
                    raise JobNotFoundError(run_id)
                if (run["name_type"] != "text" or run["plan_storage"] != "text"
                        or type(run["name"]) is not str or run["status"] not in {item.value for item in RunStatus}):
                    raise RunStoreError("Jobs run summary contains invalid name or status")
                for field in ("created_at_ns", "updated_at_ns", "approved_microusd", "confirmed_microusd",
                              "exposure_microusd", "reserved_microusd"):
                    try:
                        _microusd(run[field], field=field)
                    except (TypeError, ValueError) as exc:
                        raise RunStoreError("Jobs run summary contains invalid numeric data") from exc
                counts = {row["status"]: row["count"] for row in cursor.execute(
                    "SELECT status,COUNT(*) AS count FROM operations WHERE run_id=? GROUP BY status", (run_id,))}
                if not counts.keys() <= {item.value for item in OperationStatus}:
                    raise RunStoreError("Jobs run has invalid operation status")
                where, parameters = "run_id=?", (run_id,)
                selected_id = None
                if operation is not None:
                    matches = cursor.execute("""SELECT operation_id FROM operations
                        WHERE run_id=? AND (operation_key=? OR operation_id=?) LIMIT 2""",
                        (run_id, operation, operation)).fetchall()
                    if not matches:
                        raise ValueError("No operation matches this key or ID")
                    if len(matches) != 1:
                        raise ValueError("Operation selector matches more than one operation")
                    selected_id = matches[0]["operation_id"]
                    where, parameters = "run_id=? AND operation_id=?", (run_id, selected_id)
                total = 1 if selected_id is not None else sum(counts.values())
                operations = cursor.execute(f"""SELECT * FROM operations WHERE {where}
                    ORDER BY operation_key,operation_id LIMIT ? OFFSET ?""", (*parameters, limit, offset)).fetchall()
                ids = tuple(row["operation_id"] for row in operations)
                attempts, calls, artifacts = [], [], []
                if ids:
                    selected = ",".join("?" for _ in ids)
                    lineage = (run_id, *ids)
                    attempts = cursor.execute(f"""SELECT * FROM attempts WHERE run_id=?
                        AND operation_id IN ({selected}) ORDER BY operation_id,attempt_number""", lineage).fetchall()
                    calls = cursor.execute(f"""SELECT * FROM calls WHERE run_id=?
                        AND operation_id IN ({selected}) ORDER BY created_at_ns,call_id""", lineage).fetchall()
                    artifacts = cursor.execute(f"""SELECT * FROM artifacts WHERE run_id=?
                        AND operation_id IN ({selected}) ORDER BY relative_path,artifact_id""", lineage).fetchall()
                event_total = cursor.execute(f"SELECT COUNT(*) FROM events WHERE {where}", parameters).fetchone()[0]
                events = cursor.execute(f"""SELECT * FROM events WHERE {where}
                    ORDER BY sequence DESC LIMIT ?""", (*parameters, events_limit)).fetchall()[::-1]
                estimates = cursor.execute("SELECT EXISTS(SELECT 1 FROM calls WHERE run_id=? AND cost_is_estimate=1)",
                                           (run_id,)).fetchone()[0]
                result = self._snapshot_projection(run, operations, attempts, artifacts, events,
                                                   [{"cost_is_estimate": estimates}])
                for call in calls:
                    _validate_read_row(call, "calls")
                result.update(counts=counts, calls=[dict(row) for row in calls], operation_filter=operation,
                              pagination={"limit": limit, "offset": offset, "total": total,
                                          "returned": len(operations), "has_more": offset + len(operations) < total},
                              event_pagination={"limit": events_limit, "total": event_total,
                                                "returned": len(events), "has_more": len(events) < event_total})
                return result
        except sqlite3.Error as exc:
            raise RunStoreError("Cannot read Jobs inspection snapshot") from exc

    def manifest_record(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        _validate_read_row(run, "runs")
        return {
            "manifest_json": run["manifest_json"],
            "manifest_root": run["manifest_root"],
            "approval": _read_json_object(run["approval_json"], "runs.approval_json"),
        }

    def _call_rows(self, run_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(
                "SELECT * FROM calls WHERE run_id = ?", (run_id,)
            ).fetchall()

    def _call(self, cursor: sqlite3.Cursor, call_id: str) -> sqlite3.Row:
        row = cursor.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
        if row is None:
            raise JobNotFoundError(call_id)
        return row

    def _assert_budget_open(
        self,
        cursor: sqlite3.Cursor,
        run_id: str,
        now: int,
    ) -> sqlite3.Row:
        """Reject admission after an overrun, deriving the latch from calls."""

        run = cursor.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise JobNotFoundError(run_id)
        summary_overrun = (
            run["confirmed_microusd"] + run["exposure_microusd"]
            > run["approved_microusd"]
        )
        if self._latch_budget_overrun(
            cursor, run, now, evidenced=summary_overrun
        ):
            raise _BudgetAdmissionClosed(
                f"run {run_id!r} is latched in budget_overrun; no further calls "
                "may be admitted"
            )
        return run

    def _latch_budget_overrun(
        self,
        cursor: sqlite3.Cursor,
        run: sqlite3.Row,
        now: int,
        *,
        evidenced: bool | None = None,
    ) -> bool:
        """Make overrun irreversible, using persisted call accounting evidence."""

        already_latched = run["status"] == RunStatus.BUDGET_OVERRUN.value
        if already_latched:
            return True
        total = 0
        call_overrun = False
        approved = _microusd(run["approved_microusd"], field="approved cost")
        if evidenced is None:
            rows = cursor.execute(
                """SELECT ceiling_microusd, confirmed_microusd, exposure_microusd
                   FROM calls WHERE run_id = ?""",
                (run["run_id"],),
            ).fetchall()
            for row in rows:
                ceiling = _microusd(row["ceiling_microusd"], field="call ceiling")
                confirmed = _microusd(
                    row["confirmed_microusd"], field="call confirmed cost"
                )
                exposure = _microusd(
                    row["exposure_microusd"], field="call exposure"
                )
                amount = confirmed + exposure
                total += amount
                call_overrun = call_overrun or amount > ceiling
            evidenced = call_overrun or total > approved
        else:
            total = run["confirmed_microusd"] + run["exposure_microusd"]
            call_overrun = evidenced and total <= approved
        if evidenced:
            cursor.execute(
                "UPDATE runs SET status = ?, updated_at_ns = ? WHERE run_id = ?",
                (RunStatus.BUDGET_OVERRUN.value, now, run["run_id"]),
            )
            self._event(
                cursor,
                run["run_id"],
                None,
                "budget_overrun_latched",
                {
                    "approved_microusd": approved,
                    "durable_call_cost_microusd": total,
                    "call_ceiling_exceeded": call_overrun,
                },
                now,
            )
        return evidenced

    @staticmethod
    def _assert_run_lease(
        cursor: sqlite3.Cursor,
        run_id: str,
        owner_id: str,
        now: int,
    ) -> sqlite3.Row:
        row = cursor.execute(
            "SELECT * FROM run_leases WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunLeaseError(f"run {run_id!r} has no active lease")
        if row["owner_id"] != owner_id:
            raise RunLeaseError(f"run {run_id!r} is leased by {row['owner_id']!r}")
        if row["expires_at_ns"] <= now:
            raise RunLeaseError(f"run {run_id!r} lease for {owner_id!r} expired")
        return row

    def _recover_orphan_attempt(
        self,
        cursor: sqlite3.Cursor,
        attempt: sqlite3.Row,
        now: int,
    ) -> None:
        operation = cursor.execute(
            """SELECT * FROM operations
               WHERE run_id = ? AND operation_id = ?""",
            (attempt["run_id"], attempt["operation_id"]),
        ).fetchone()
        if operation is None:
            raise JobNotFoundError(
                f"{attempt['run_id']}:{attempt['operation_id']}"
            )
        if operation["status"] != OperationStatus.RUNNING.value:
            raise InvalidTransitionError(
                f"orphan attempt {attempt['attempt_id']} belongs to an operation "
                f"in state {operation['status']}"
            )
        error = "recovered before call preparation"
        cursor.execute(
            """UPDATE attempts SET status = 'failed', error = ?, completed_at_ns = ?
               WHERE attempt_id = ?""",
            (error, now, attempt["attempt_id"]),
        )
        cursor.execute(
            """UPDATE operations SET status = 'pending',
               attempt_count = MAX(0, attempt_count - 1), error = NULL,
               updated_at_ns = ? WHERE run_id = ? AND operation_id = ?""",
            (now, attempt["run_id"], attempt["operation_id"]),
        )
        self._event(
            cursor,
            attempt["run_id"],
            attempt["operation_id"],
            "attempt_recovered_before_call",
            {"attempt_id": attempt["attempt_id"]},
            now,
        )

    def _fail_safe_call(
        self,
        cursor: sqlite3.Cursor,
        call: sqlite3.Row,
        error: str,
        now: int,
        *,
        retryable: bool,
    ) -> None:
        cursor.execute(
            """UPDATE calls SET status = 'failed', error = ?, completed_at_ns = ?
               WHERE call_id = ?""",
            (error, now, call["call_id"]),
        )
        cursor.execute(
            """UPDATE runs SET reserved_microusd = reserved_microusd - ?,
               updated_at_ns = ? WHERE run_id = ?""",
            (call["ceiling_microusd"], now, call["run_id"]),
        )
        attempt = cursor.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (call["attempt_id"],)
        ).fetchone()
        cursor.execute(
            """UPDATE attempts SET status = 'failed', error = ?, completed_at_ns = ?
               WHERE attempt_id = ?""",
            (error, now, call["attempt_id"]),
        )
        if retryable:
            cursor.execute(
                """UPDATE operations SET status = 'pending',
                   attempt_count = MAX(0, attempt_count - 1), error = NULL,
                   updated_at_ns = ? WHERE run_id = ? AND operation_id = ?""",
                (now, call["run_id"], call["operation_id"]),
            )
        else:
            cursor.execute(
                """UPDATE operations SET status = 'failed', error = ?,
                   updated_at_ns = ? WHERE run_id = ? AND operation_id = ?""",
                (error, now, call["run_id"], call["operation_id"]),
            )
        self._event(
            cursor,
            call["run_id"],
            call["operation_id"],
            "call_failed_before_dispatch",
            {"call_id": call["call_id"], "attempt_id": attempt["attempt_id"]},
            now,
        )

    def _unknown_call(
        self, cursor: sqlite3.Cursor, call: sqlite3.Row, error: str, now: int
    ) -> None:
        ceiling = call["ceiling_microusd"]
        run = cursor.execute(
            "SELECT * FROM runs WHERE run_id = ?", (call["run_id"],)
        ).fetchone()
        if run is None:  # pragma: no cover - protected by foreign keys
            raise JobNotFoundError(call["run_id"])
        cursor.execute(
            """UPDATE calls SET status = 'unknown_outcome', exposure_microusd = ?,
               cost_is_complete = 0, cost_is_estimate = 1, error = ?,
               completed_at_ns = ? WHERE call_id = ?""",
            (ceiling, error, now, call["call_id"]),
        )
        cursor.execute(
            """UPDATE runs SET reserved_microusd = ?,
               exposure_microusd = ?, status = CASE
                   WHEN status = ? THEN status ELSE ? END,
               updated_at_ns = ? WHERE run_id = ?""",
            (
                _microusd(
                    run["reserved_microusd"] - ceiling,
                    field="run reserved cost",
                ),
                _microusd(
                    run["exposure_microusd"] + ceiling,
                    field="run exposure",
                ),
                RunStatus.BUDGET_OVERRUN.value,
                RunStatus.NEEDS_ATTENTION.value,
                now,
                call["run_id"],
            ),
        )
        cursor.execute(
            """UPDATE attempts SET status = ?, error = ?, completed_at_ns = ?
               WHERE attempt_id = ?""",
            (OperationStatus.UNKNOWN_OUTCOME.value, error, now, call["attempt_id"]),
        )
        cursor.execute(
            """UPDATE operations SET status = ?, error = ?, updated_at_ns = ?
               WHERE run_id = ? AND operation_id = ?""",
            (
                OperationStatus.UNKNOWN_OUTCOME.value,
                error,
                now,
                call["run_id"],
                call["operation_id"],
            ),
        )
        self._event(
            cursor,
            call["run_id"],
            call["operation_id"],
            "unknown_outcome",
            {"call_id": call["call_id"], "error": error},
            now,
        )

    @staticmethod
    def _event(
        cursor: sqlite3.Cursor,
        run_id: str,
        operation_id: str | None,
        event_type: str,
        payload: Any,
        now: int,
    ) -> None:
        cursor.execute(
            """INSERT INTO events (
                   run_id, operation_id, event_type, payload_json, created_at_ns
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                operation_id,
                event_type,
                json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
                now,
            ),
        )
