"""Append-only SQLite ledger for durable, budget-visible Autotune campaigns."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

from smythe._sqlite import enable_wal
from smythe.optimize.contracts import Candidate, ExperimentContract, canonical_json_bytes


LEDGER_VERSION = 4
_WRITER_FUNCTION = "smythe_autotune_writer_version"
_OWNED_TABLES = ("ledger_meta", "campaigns", "candidates", "trials", "trial_events",
                 "promotion_decisions", "campaign_lease_epochs", "campaign_leases",
                 "trial_dispatch_owners")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HOLDOUT_NONCE_BYTES = 32
_HOLDOUT_COMMITMENT_DOMAIN = b"smythe-autotune-holdout-v1\0"
_SQLITE_INT_MAX = 2**63 - 1

# Python cannot make an in-process secret cryptographically inaccessible to code
# that can introspect this module.  This unexported identity is nevertheless a
# useful capability boundary: generic callers and public inspection APIs cannot
# accidentally retrieve holdout material, while the trusted engine must opt in
# explicitly.  The documented security boundary still trusts the runner process
# and the operator who can read the SQLite database.
_ENGINE_HOLDOUT_CAPABILITY = object()


class ExperimentLedgerError(RuntimeError):
    """Base class for durable experiment-ledger failures."""


class LedgerConflictError(ExperimentLedgerError):
    """Raised when an idempotency key is reused with different bytes."""


class CampaignNotFoundError(ExperimentLedgerError, KeyError):
    """Raised when a campaign identifier is unknown."""


class CandidateNotFoundError(ExperimentLedgerError, KeyError):
    """Raised when a candidate has not been registered in a campaign."""


class LedgerBudgetError(ExperimentLedgerError):
    """Raised before a trial reservation would exceed its immutable contract."""


class TrialNotFoundError(ExperimentLedgerError, KeyError):
    """Raised when a deterministic trial key is unknown."""


class TrialStateError(ExperimentLedgerError):
    """Raised when a trial event would violate its append-only state machine."""


class UnknownTrialError(TrialStateError):
    """Raised when code attempts to reuse an ambiguously billed trial."""


class CampaignLeaseError(ExperimentLedgerError):
    """Campaign ownership is unavailable, invalid, or no longer live."""


class CampaignLeaseConflict(CampaignLeaseError):
    """Another live runner owns this campaign."""


@dataclass(frozen=True, slots=True)
class CampaignLease:
    """An immutable local ownership token; persisted expiry is authoritative."""

    campaign_id: str
    owner_id: str
    epoch: int
    acquired_at_ns: int
    heartbeat_at_ns: int
    expires_at_ns: int

    def __post_init__(self) -> None:
        for name in ("campaign_id", "owner_id"):
            value = getattr(self, name)
            if type(value) is not str or _safe_id(value, name) != value:
                raise CampaignLeaseError(f"Invalid lease {name}")
        for name in ("epoch", "acquired_at_ns", "heartbeat_at_ns", "expires_at_ns"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _SQLITE_INT_MAX:
                raise CampaignLeaseError(f"Invalid lease {name}")
        if (self.epoch < 1 or not
                self.acquired_at_ns <= self.heartbeat_at_ns < self.expires_at_ns):
            raise CampaignLeaseError("Invalid lease epoch or timestamp ordering")


def _lease_duration_ns(ttl_s: float) -> int:
    if type(ttl_s) not in (int, float):
        raise ValueError("lease TTL must be finite and positive")
    try:
        valid = math.isfinite(ttl_s) and ttl_s > 0
        duration = int(ttl_s * 1_000_000_000) if valid else 0
    except (OverflowError, ValueError):
        duration = 0
    if not 0 < duration <= _SQLITE_INT_MAX:
        raise ValueError("lease TTL must fit positive signed-64-bit nanoseconds")
    return duration


def _lease_now_ns() -> int:
    now = time.time_ns()
    if type(now) is not int or not 0 <= now <= _SQLITE_INT_MAX:
        raise CampaignLeaseError("Invalid lease clock")
    return now


class TrialStatus(StrEnum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


def _nonempty(value: object, field_name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} must be at most {maximum} characters")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} cannot contain control characters")
    return normalized


def _safe_id(value: object, field_name: str) -> str:
    normalized = _nonempty(value, field_name, maximum=128)
    if normalized in {".", ".."} or _SAFE_ID_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must be a safe 1-128 character identifier beginning "
            "with an alphanumeric character"
        )
    return normalized


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value > _SQLITE_INT_MAX:
        raise ValueError(f"{field_name} must fit in a signed 64-bit integer")
    return value


def _seed(value: object) -> int:
    seed = _nonnegative_int(value, "seed")
    return seed


def _duration(value: object, field_name: str = "duration_ms") -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and non-negative")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not finite or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase sha256:<64 hex characters> identifier"
        )
    return value


def _metrics(values: Mapping[str, int | float]) -> dict[str, int | float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("metrics must be a non-empty mapping")
    normalized: dict[str, int | float] = {}
    for key, value in values.items():
        name = _nonempty(key, "metric name", maximum=128)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {name!r} must be a number")
        try:
            finite = math.isfinite(float(value))
        except OverflowError as exc:
            raise ValueError(f"metric {name!r} must be finite") from exc
        if not finite:
            raise ValueError(f"metric {name!r} must be finite")
        normalized[name] = value
    return normalized


def _gates(values: Mapping[str, bool]) -> dict[str, bool]:
    if not isinstance(values, Mapping):
        raise TypeError("gates must be a mapping")
    normalized: dict[str, bool] = {}
    for key, value in values.items():
        name = _nonempty(key, "gate name", maximum=128)
        if not isinstance(value, bool):
            raise TypeError(f"gate {name!r} must be a boolean")
        normalized[name] = value
    return normalized


def _hashes(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("artifact_hashes must be an iterable of hash strings")
    normalized = tuple(_sha256(value, "artifact hash") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("artifact_hashes cannot contain duplicates")
    return normalized


def _json_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _deterministic_id(prefix: str, value: object) -> str:
    return prefix + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _deep_json_copy(value: object) -> Any:
    """Return a fully detached, strictly JSON-compatible mutable value."""

    return json.loads(canonical_json_bytes(value))


def _deep_freeze_json(value: Any) -> Any:
    """Recursively freeze a value previously normalized as strict JSON."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _holdout_commitment(nonce: bytes) -> str:
    digest = hashlib.sha256(_HOLDOUT_COMMITMENT_DOMAIN + nonce).hexdigest()
    return f"sha256:{digest}"


def _campaign_binding(
    contract: ExperimentContract,
    incumbent_candidate_id: str,
    candidates: Sequence[Candidate],
    plan_hash: str,
) -> tuple[str, str, str, str, str, tuple[tuple[str, str, str], ...]]:
    """Normalize a complete, ordered campaign plan before any ledger write."""

    if not isinstance(contract, ExperimentContract):
        raise TypeError("contract must be an ExperimentContract")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of Candidate values")
    inventory = tuple(candidates)
    if not inventory:
        raise ValueError("candidates must contain the complete campaign inventory")
    if any(not isinstance(candidate, Candidate) for candidate in inventory):
        raise TypeError("candidates must contain only Candidate values")
    if len(inventory) > contract.max_candidates:
        raise LedgerConflictError("campaign inventory exceeds contract.max_candidates")

    incumbent = _safe_id(incumbent_candidate_id, "incumbent_candidate_id")
    normalized_plan_hash = _sha256(plan_hash, "plan_hash")
    ids = tuple(_safe_id(candidate.candidate_id, "candidate_id") for candidate in inventory)
    hashes = tuple(_sha256(candidate.policy_hash, "policy_hash") for candidate in inventory)
    if len(set(ids)) != len(ids):
        raise ValueError("campaign candidate IDs must be unique")
    if len(set(hashes)) != len(hashes):
        raise ValueError("campaign policy hashes must be unique")
    if ids[0] != incumbent or ids.count(incumbent) != 1:
        raise ValueError("the incumbent must be the first and only matching candidate")

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for candidate, candidate_id, policy_hash in zip(
        inventory, ids, hashes, strict=True
    ):
        if not hmac.compare_digest(candidate.contract_hash, contract.contract_hash):
            raise LedgerConflictError(
                f"candidate {candidate_id!r} belongs to a different contract"
            )
        if candidate.parent is not None and candidate.parent not in seen:
            raise LedgerConflictError(
                f"candidate {candidate_id!r} parent must precede it in the sealed inventory"
            )
        payload = candidate.to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("Candidate.to_dict() must return a mapping")
        rows.append((candidate_id, policy_hash, _json_text(payload)))
        seen.add(candidate_id)

    return (
        incumbent,
        normalized_plan_hash,
        _json_text(list(ids)),
        _json_text(list(hashes)),
        _json_text([json.loads(row[2]) for row in rows]),
        tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One materialized trial state derived from immutable ledger rows."""

    trial_key: str
    campaign_id: str
    candidate_id: str
    split: str
    phase: str
    seed: int
    evaluator_hash: str
    ceiling_microusd: int
    status: TrialStatus
    metrics: dict[str, int | float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    actual_cost_microusd: int | None = None
    duration_ms: float | None = None
    artifact_hashes: tuple[str, ...] = ()
    error: str | None = None
    prepared_at_ns: int | None = None
    dispatched_at_ns: int | None = None
    terminal_at_ns: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.trial_key, "trial_key", maximum=128)
        _safe_id(self.campaign_id, "campaign_id")
        _safe_id(self.candidate_id, "candidate_id")
        _nonempty(self.split, "split", maximum=128)
        _nonempty(self.phase, "phase", maximum=128)
        _seed(self.seed)
        _sha256(self.evaluator_hash, "evaluator_hash")
        _nonnegative_int(self.ceiling_microusd, "ceiling_microusd")
        object.__setattr__(self, "status", TrialStatus(self.status))
        if self.status is TrialStatus.COMPLETED:
            object.__setattr__(self, "metrics", _metrics(self.metrics))
            object.__setattr__(self, "gates", _gates(self.gates))
            if self.actual_cost_microusd is None:
                raise ValueError("completed trial requires actual_cost_microusd")
            _nonnegative_int(self.actual_cost_microusd, "actual_cost_microusd")
            _duration(self.duration_ms, "completed trial duration_ms")
            object.__setattr__(self, "artifact_hashes", _hashes(self.artifact_hashes))
            if self.error is not None:
                raise ValueError("completed trial cannot contain an error")
        elif self.status is TrialStatus.UNKNOWN:
            object.__setattr__(self, "metrics", {})
            object.__setattr__(self, "gates", {})
            object.__setattr__(self, "artifact_hashes", ())
            if self.error is None:
                raise ValueError("unknown trial requires an error")
            object.__setattr__(self, "error", _nonempty(self.error, "error", maximum=4096))
            if self.actual_cost_microusd is not None or self.duration_ms is not None:
                raise ValueError("unknown trial cannot claim confirmed cost or duration")
        else:
            if self.metrics or self.gates or self.artifact_hashes:
                raise ValueError("non-terminal trial cannot contain result data")
            if self.actual_cost_microusd is not None or self.duration_ms is not None:
                raise ValueError("non-terminal trial cannot contain cost or duration")
            if self.error is not None:
                raise ValueError("non-terminal trial cannot contain an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_key": self.trial_key,
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "split": self.split,
            "phase": self.phase,
            "seed": self.seed,
            "evaluator_hash": self.evaluator_hash,
            "ceiling_microusd": self.ceiling_microusd,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "gates": dict(self.gates),
            "actual_cost_microusd": self.actual_cost_microusd,
            "duration_ms": self.duration_ms,
            "artifact_hashes": list(self.artifact_hashes),
            "error": self.error,
            "prepared_at_ns": self.prepared_at_ns,
            "dispatched_at_ns": self.dispatched_at_ns,
            "terminal_at_ns": self.terminal_at_ns,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Append-only promotion or rejection decision with trial evidence."""

    campaign_id: str
    candidate_id: str
    promoted: bool
    reason: str
    trial_keys: tuple[str, ...] = ()
    assessment: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str = field(init=False)
    _payload_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _safe_id(self.campaign_id, "campaign_id"))
        object.__setattr__(self, "candidate_id", _safe_id(self.candidate_id, "candidate_id"))
        if not isinstance(self.promoted, bool):
            raise TypeError("promoted must be a boolean")
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason", maximum=4096))
        if isinstance(self.trial_keys, (str, bytes)):
            raise TypeError("trial_keys must be an iterable of strings")
        keys = tuple(_nonempty(key, "trial_key", maximum=128) for key in self.trial_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("trial_keys cannot contain duplicates")
        object.__setattr__(self, "trial_keys", keys)
        if not isinstance(self.assessment, Mapping):
            raise TypeError("assessment must be a mapping")
        assessment = _deep_json_copy(self.assessment)
        if not isinstance(assessment, dict):  # defensive; Mapping normalizes to object
            raise TypeError("assessment must normalize to a JSON object")
        object.__setattr__(self, "assessment", _deep_freeze_json(assessment))
        payload = {
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "promoted": self.promoted,
            "reason": self.reason,
            "trial_keys": list(keys),
            "assessment": assessment,
        }
        payload_json = _json_text(payload)
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(
            self,
            "decision_id",
            "decision_v1_" + hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        # Parsing the cached canonical payload returns mutable dict/list containers
        # at every depth without exposing the immutable identity-bearing state.
        result = json.loads(self._payload_json)
        assert isinstance(result, dict)
        return result


class ExperimentLedger:
    """WAL SQLite ledger with explicit append-only durability semantics."""

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        durability: str = "full",
    ) -> None:
        self.path = Path(path).resolve()
        self.read_only = read_only
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a boolean")
        if durability not in {"full", "normal"}:
            raise ValueError("durability must be 'full' or 'normal'")
        self.durability = durability
        self._lock = threading.RLock()
        self._closed = True
        self._read_only_fingerprint: tuple[int, int, int, int] | None = None
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self._read_only_fingerprint = self._database_fingerprint()
            self._assert_read_only_quiescent("before opening")
            try:
                connection = sqlite3.connect(
                    self.path.as_uri() + "?mode=ro&immutable=1",
                    uri=True,
                    isolation_level=None,
                    check_same_thread=False,
                )
            except BaseException as exc:
                try:
                    self._assert_read_only_quiescent("after failed open")
                except ExperimentLedgerError as quiescence_error:
                    raise quiescence_error from exc
                raise
            self._connection = connection
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                check_same_thread=False,
            )
        try:
            self._connection.row_factory = sqlite3.Row
            with self._lock:
                if read_only:
                    # `immutable=1` trusts the caller's quiescence assertion.  The
                    # second check closes the largest check/open race; the close
                    # check below catches overlapping writers and checkpoints.
                    self._assert_read_only_quiescent("immediately after opening")
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._connection.execute("PRAGMA busy_timeout = 5000")
                if read_only:
                    self._connection.execute("PRAGMA query_only = ON")
                    self._verify_version()
                else:
                    self._connection.create_function(_WRITER_FUNCTION, 0, lambda: LEDGER_VERSION)
                    enable_wal(self._connection, validate_before_write=self._guard_existing_version_before_write)
                    synchronous = "FULL" if durability == "full" else "NORMAL"
                    self._connection.execute(f"PRAGMA synchronous = {synchronous}")
                    self._create_schema()
        except BaseException as exc:
            self._connection.close()
            if read_only:
                try:
                    self._assert_read_only_quiescent("after failed initialization")
                except ExperimentLedgerError as quiescence_error:
                    raise quiescence_error from exc
            raise
        self._closed = False

    def __enter__(self) -> "ExperimentLedger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True
        if self.read_only:
            # No finite sequence of filesystem checks can eliminate a malicious
            # writer that starts and checkpoints entirely between checks.  This
            # conservative close-time sidecar/fingerprint validation detects the
            # practical race without pretending to provide a filesystem lock.
            self._assert_read_only_quiescent("after closing")

    def _database_fingerprint(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _assert_read_only_quiescent(self, stage: str) -> None:
        sidecars = [
            candidate
            for candidate in (Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
            if candidate.exists()
        ]
        if sidecars:
            names = ", ".join(item.name for item in sidecars)
            raise ExperimentLedgerError(
                "read-only inspection requires a closed, checkpointed ledger; "
                f"found SQLite sidecar(s) {stage}: {names}"
            )
        if (
            self._read_only_fingerprint is not None
            and self._database_fingerprint() != self._read_only_fingerprint
        ):
            raise ExperimentLedgerError(
                "read-only inspection requires an unchanged, closed, checkpointed "
                f"ledger; database changed {stage}"
            )

    def _guard_existing_version_before_write(self) -> None:
        table = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ledger_meta'"
        ).fetchone()
        if table is None:
            return
        rows = self._connection.execute("SELECT version FROM ledger_meta").fetchall()
        if len(rows) != 1 or rows[0]["version"] not in (3, LEDGER_VERSION):
            observed = [row["version"] for row in rows]
            raise ExperimentLedgerError(
                f"unsupported experiment ledger version {observed}; "
                f"create a fresh version-{LEDGER_VERSION} ledger"
            )

    def _create_schema(self) -> None:
        schema = """
            CREATE TABLE IF NOT EXISTS ledger_meta (
                version INTEGER PRIMARY KEY,
                created_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                contract_hash TEXT NOT NULL,
                contract_json TEXT NOT NULL,
                incumbent_candidate_id TEXT NOT NULL,
                plan_hash TEXT NOT NULL CHECK (length(plan_hash) = 71),
                ordered_candidate_ids_json TEXT NOT NULL,
                ordered_policy_hashes_json TEXT NOT NULL,
                candidate_inventory_json TEXT NOT NULL,
                holdout_nonce BLOB NOT NULL CHECK (
                    typeof(holdout_nonce) = 'blob' AND length(holdout_nonce) = 32
                ),
                holdout_commitment TEXT NOT NULL UNIQUE CHECK (
                    length(holdout_commitment) = 71
                ),
                created_at_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidates (
                campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                candidate_id TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL,
                PRIMARY KEY (campaign_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS trials (
                trial_key TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                split TEXT NOT NULL,
                phase TEXT NOT NULL,
                seed INTEGER NOT NULL,
                evaluator_hash TEXT NOT NULL,
                ceiling_microusd INTEGER NOT NULL,
                prepared_json TEXT NOT NULL,
                prepared_at_ns INTEGER NOT NULL,
                FOREIGN KEY (campaign_id, candidate_id)
                    REFERENCES candidates(campaign_id, candidate_id),
                UNIQUE (campaign_id, candidate_id, split, phase, seed)
            );

            CREATE TABLE IF NOT EXISTS trial_events (
                event_id TEXT PRIMARY KEY,
                trial_key TEXT NOT NULL REFERENCES trials(trial_key),
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('dispatched', 'completed', 'unknown')
                ),
                payload_json TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL,
                UNIQUE (trial_key, event_type)
            );

            CREATE TABLE IF NOT EXISTS promotion_decisions (
                decision_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_ns INTEGER NOT NULL,
                FOREIGN KEY (campaign_id, candidate_id)
                    REFERENCES candidates(campaign_id, candidate_id),
                UNIQUE (campaign_id)
            );

            CREATE INDEX IF NOT EXISTS trials_by_campaign
                ON trials(campaign_id, candidate_id, split, phase, seed);
            CREATE INDEX IF NOT EXISTS trial_events_by_trial
                ON trial_events(trial_key, created_at_ns);
            CREATE INDEX IF NOT EXISTS decisions_by_campaign
                ON promotion_decisions(campaign_id, created_at_ns);

            CREATE TABLE IF NOT EXISTS campaign_lease_epochs (
                campaign_id TEXT PRIMARY KEY REFERENCES campaigns(campaign_id),
                last_epoch INTEGER NOT NULL CHECK(typeof(last_epoch) = 'integer' AND last_epoch > 0)
            );
            CREATE TABLE IF NOT EXISTS campaign_leases (
                campaign_id TEXT PRIMARY KEY REFERENCES campaigns(campaign_id),
                owner_id TEXT NOT NULL,
                epoch INTEGER NOT NULL CHECK(typeof(epoch) = 'integer' AND epoch > 0),
                acquired_at_ns INTEGER NOT NULL CHECK(typeof(acquired_at_ns) = 'integer' AND acquired_at_ns >= 0),
                heartbeat_at_ns INTEGER NOT NULL CHECK(typeof(heartbeat_at_ns) = 'integer' AND heartbeat_at_ns >= acquired_at_ns),
                expires_at_ns INTEGER NOT NULL CHECK(typeof(expires_at_ns) = 'integer' AND expires_at_ns > heartbeat_at_ns)
            );
            CREATE TABLE IF NOT EXISTS trial_dispatch_owners (
                trial_key TEXT PRIMARY KEY REFERENCES trials(trial_key),
                owner_id TEXT NOT NULL,
                epoch INTEGER NOT NULL CHECK(typeof(epoch) = 'integer' AND epoch > 0)
            );
            """
        cursor = self._connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            self._guard_existing_version_before_write()
            exists = cursor.execute("SELECT 1 FROM sqlite_master WHERE name='ledger_meta'").fetchone()
            version = cursor.execute("SELECT version FROM ledger_meta").fetchone()[0] if exists else None
            if version == LEDGER_VERSION:
                self._verify_barrier(cursor)
            else:
                if version == 3:
                    self._verify_legacy_tables(cursor)
                for statement in schema.split(";"):
                    if statement.strip():
                        cursor.execute(statement)
                for sql in self._barrier_sql().values():
                    cursor.execute(sql)
                if version is None:
                    cursor.execute("INSERT INTO ledger_meta VALUES (?, ?)", (LEDGER_VERSION, time.time_ns()))
                else:
                    cursor.execute("UPDATE ledger_meta SET version = ?", (LEDGER_VERSION,))
                self._verify_barrier(cursor)
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    @staticmethod
    def _barrier_sql() -> dict[str, str]:
        return {
            f"autotune_v4_{table}_{operation.lower()}":
                f"CREATE TRIGGER autotune_v4_{table}_{operation.lower()} BEFORE {operation} ON {table} "
                f"BEGIN SELECT CASE WHEN {_WRITER_FUNCTION}() != 4 "
                "THEN RAISE(ABORT, 'Autotune schema v4 writer required') END; END"
            for table in _OWNED_TABLES for operation in ("INSERT", "UPDATE", "DELETE")
        }

    @staticmethod
    def _verify_legacy_tables(cursor: sqlite3.Cursor) -> None:
        expected = {
            "ledger_meta": {"version", "created_at_ns"},
            "campaigns": {"campaign_id", "contract_hash", "contract_json", "incumbent_candidate_id",
                          "plan_hash", "ordered_candidate_ids_json", "ordered_policy_hashes_json",
                          "candidate_inventory_json", "holdout_nonce", "holdout_commitment", "created_at_ns"},
            "candidates": {"campaign_id", "candidate_id", "policy_hash", "candidate_json", "created_at_ns"},
            "trials": {"trial_key", "campaign_id", "candidate_id", "split", "phase", "seed",
                       "evaluator_hash", "ceiling_microusd", "prepared_json", "prepared_at_ns"},
            "trial_events": {"event_id", "trial_key", "event_type", "payload_json", "created_at_ns"},
            "promotion_decisions": {"decision_id", "campaign_id", "candidate_id", "payload_json", "created_at_ns"},
        }
        for table, columns in expected.items():
            if {row["name"] for row in cursor.execute(f"PRAGMA table_info({table})")} != columns:
                raise ExperimentLedgerError(f"Invalid experiment ledger table {table}")
        if cursor.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ExperimentLedgerError("Experiment ledger has invalid foreign keys")

    @classmethod
    def _verify_barrier(cls, cursor: sqlite3.Cursor) -> None:
        cls._verify_legacy_tables(cursor)
        expected_columns = {
            "campaign_lease_epochs": {"campaign_id", "last_epoch"},
            "campaign_leases": set(CampaignLease.__dataclass_fields__),
            "trial_dispatch_owners": {"trial_key", "owner_id", "epoch"},
        }
        for table, columns in expected_columns.items():
            if {row["name"] for row in cursor.execute(f"PRAGMA table_info({table})")} != columns:
                raise ExperimentLedgerError(f"Invalid ownership table {table}")
        for name, sql in cls._barrier_sql().items():
            row = cursor.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
            if row is None or " ".join(row["sql"].split()) != " ".join(sql.split()):
                raise ExperimentLedgerError(f"Missing or altered schema v4 writer barrier: {name}")

    def _verify_version(self) -> None:
        try:
            rows = self._connection.execute("SELECT version FROM ledger_meta").fetchall()
        except sqlite3.Error as exc:
            raise ExperimentLedgerError("not a Smythe experiment ledger") from exc
        if len(rows) != 1 or rows[0]["version"] not in (3, LEDGER_VERSION):
            observed = [row["version"] for row in rows]
            raise ExperimentLedgerError(
                f"unsupported experiment ledger version {observed}"
            )
        cursor = self._connection.cursor()
        try:
            self._verify_legacy_tables(cursor)
            if rows[0]["version"] == LEDGER_VERSION:
                self._verify_barrier(cursor)
        finally:
            cursor.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            if self.read_only:
                raise sqlite3.OperationalError("attempt to write a readonly experiment ledger")
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                rows = cursor.execute("SELECT version FROM ledger_meta").fetchall()
                if len(rows) != 1 or rows[0]["version"] != LEDGER_VERSION:
                    raise ExperimentLedgerError("Writable operations require experiment ledger schema v4")
                yield cursor
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")
            finally:
                cursor.close()

    @staticmethod
    def make_trial_key(
        campaign_id: str,
        candidate_id: str,
        split: str,
        phase: str,
        seed: int,
    ) -> str:
        payload = {
            "campaign_id": _safe_id(campaign_id, "campaign_id"),
            "candidate_id": _safe_id(candidate_id, "candidate_id"),
            "split": _nonempty(split, "split", maximum=128),
            "phase": _nonempty(phase, "phase", maximum=128),
            "seed": _seed(seed),
        }
        return _deterministic_id("trial_v1_", payload)

    @staticmethod
    def _lease_from_row(cursor: sqlite3.Cursor, row: sqlite3.Row) -> CampaignLease:
        try:
            lease = CampaignLease(**dict(row))
        except (TypeError, ValueError) as exc:
            raise CampaignLeaseError("Malformed campaign lease") from exc
        counter = cursor.execute("SELECT last_epoch FROM campaign_lease_epochs WHERE campaign_id=?",
                                 (lease.campaign_id,)).fetchone()
        if counter is None or type(counter[0]) is not int or counter[0] != lease.epoch:
            raise CampaignLeaseError("Campaign lease epoch counter is inconsistent")
        return lease

    @classmethod
    def _require_lease(cls, cursor: sqlite3.Cursor, lease: CampaignLease,
                       *, campaign_id: str | None = None) -> CampaignLease:
        if type(lease) is not CampaignLease:
            raise CampaignLeaseError("An explicit CampaignLease token is required")
        lease.__post_init__()
        if campaign_id is not None and campaign_id != lease.campaign_id:
            raise CampaignLeaseError("Lease belongs to a different campaign")
        row = cursor.execute("SELECT * FROM campaign_leases WHERE campaign_id=?",
                             (lease.campaign_id,)).fetchone()
        if row is None:
            raise CampaignLeaseError("Campaign lease was released or is missing")
        current = cls._lease_from_row(cursor, row)
        now = _lease_now_ns()
        if ((current.owner_id, current.epoch) != (lease.owner_id, lease.epoch)
                or now < current.heartbeat_at_ns or now >= current.expires_at_ns):
            raise CampaignLeaseError("Campaign lease is expired or belongs to another owner/epoch")
        return current

    @contextmanager
    def _leased_transaction(self, lease: CampaignLease, *, campaign_id: str | None = None,
                            trial_key: str | None = None) -> Iterator[sqlite3.Cursor]:
        with self._transaction() as cursor:
            if trial_key is not None:
                campaign_id = self._require_trial(cursor, trial_key)["campaign_id"]
            self._require_lease(cursor, lease, campaign_id=campaign_id)
            yield cursor
            # Validation may be expensive; expiry still matters before commit.
            self._require_lease(cursor, lease, campaign_id=campaign_id)

    def acquire_campaign_lease(self, campaign_id: str, owner_id: str, *, ttl_s: float = 30.0) -> CampaignLease:
        campaign = _safe_id(campaign_id, "campaign_id")
        owner = _safe_id(owner_id, "owner_id")
        duration = _lease_duration_ns(ttl_s)
        with self._transaction() as cursor:
            self._require_campaign(cursor, campaign)
            now = _lease_now_ns()
            row = cursor.execute("SELECT * FROM campaign_leases WHERE campaign_id=?", (campaign,)).fetchone()
            if row is not None:
                current = self._lease_from_row(cursor, row)
                if now < current.expires_at_ns:
                    raise CampaignLeaseConflict(f"Campaign {campaign!r} is owned by a live runner")
            counter = cursor.execute("SELECT last_epoch FROM campaign_lease_epochs WHERE campaign_id=?",
                                     (campaign,)).fetchone()
            previous = 0 if counter is None else counter[0]
            if (type(previous) is not int or not 0 <= previous < _SQLITE_INT_MAX
                    or (counter is not None and previous == 0)):
                raise CampaignLeaseError("Invalid or exhausted campaign lease epoch")
            for dispatch in cursor.execute(
                "SELECT d.owner_id, d.epoch FROM trial_dispatch_owners d JOIN trials t "
                "ON t.trial_key=d.trial_key WHERE t.campaign_id=?", (campaign,),
            ).fetchall():
                if (type(dispatch["epoch"]) is not int or not 1 <= dispatch["epoch"] <= previous
                        or type(dispatch["owner_id"]) is not str
                        or _safe_id(dispatch["owner_id"], "dispatch owner") != dispatch["owner_id"]):
                    raise CampaignLeaseError("Campaign epoch counter contradicts immutable dispatch provenance")
            lease = CampaignLease(campaign, owner, previous + 1, now, now, now + duration)
            cursor.execute("INSERT INTO campaign_lease_epochs VALUES (?, ?) ON CONFLICT(campaign_id) "
                           "DO UPDATE SET last_epoch=excluded.last_epoch", (campaign, lease.epoch))
            cursor.execute("INSERT INTO campaign_leases VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(campaign_id) "
                           "DO UPDATE SET owner_id=excluded.owner_id, epoch=excluded.epoch, "
                           "acquired_at_ns=excluded.acquired_at_ns, heartbeat_at_ns=excluded.heartbeat_at_ns, "
                           "expires_at_ns=excluded.expires_at_ns",
                           (campaign, owner, lease.epoch, now, now, lease.expires_at_ns))
            self._require_lease(cursor, lease)
            return lease

    def heartbeat_campaign_lease(self, lease: CampaignLease, *, ttl_s: float = 30.0) -> CampaignLease:
        duration = _lease_duration_ns(ttl_s)
        with self._leased_transaction(lease) as cursor:
            current = self._require_lease(cursor, lease)
            now = _lease_now_ns()
            if not current.heartbeat_at_ns <= now < current.expires_at_ns:
                raise CampaignLeaseError("Campaign lease expired before heartbeat renewal")
            renewed = CampaignLease(current.campaign_id, current.owner_id, current.epoch,
                                    current.acquired_at_ns, now, now + duration)
            cursor.execute("UPDATE campaign_leases SET heartbeat_at_ns=?, expires_at_ns=? WHERE campaign_id=?",
                           (now, renewed.expires_at_ns, current.campaign_id))
            return renewed

    def assert_campaign_lease(self, lease: CampaignLease) -> None:
        with self._leased_transaction(lease):
            pass

    def release_campaign_lease(self, lease: CampaignLease) -> None:
        with self._transaction() as cursor:
            self._require_lease(cursor, lease)
            cursor.execute("DELETE FROM campaign_leases WHERE campaign_id=? AND owner_id=? AND epoch=?",
                           (lease.campaign_id, lease.owner_id, lease.epoch))

    @property
    def lease_supported(self) -> bool:
        with self._lock:
            rows = self._connection.execute("SELECT version FROM ledger_meta").fetchall()
            if len(rows) != 1 or rows[0][0] not in (3, LEDGER_VERSION):
                raise ExperimentLedgerError("Unsupported experiment ledger version")
            return rows[0][0] == LEDGER_VERSION

    def get_campaign_lease(self, campaign_id: str) -> CampaignLease | None:
        campaign = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN")
                self._require_campaign(cursor, campaign)
                if not self.lease_supported:
                    result = None
                else:
                    row = cursor.execute("SELECT * FROM campaign_leases WHERE campaign_id=?", (campaign,)).fetchone()
                    result = self._lease_from_row(cursor, row) if row is not None else None
                cursor.execute("COMMIT")
                return result
            except BaseException:
                if self._connection.in_transaction:
                    cursor.execute("ROLLBACK")
                raise
            finally:
                cursor.close()

    @staticmethod
    def _require_dispatch_owner(cursor: sqlite3.Cursor, trial_key: str, lease: CampaignLease) -> None:
        row = cursor.execute("SELECT * FROM trial_dispatch_owners WHERE trial_key=?", (trial_key,)).fetchone()
        if row is None or (row["owner_id"], row["epoch"]) != (lease.owner_id, lease.epoch):
            raise CampaignLeaseError("Trial dispatch belongs to another or an unbound historical owner")

    @classmethod
    def _require_admission(cls, cursor: sqlite3.Cursor, lease: CampaignLease) -> None:
        for row in cursor.execute("SELECT trial_key FROM trials WHERE campaign_id=?", (lease.campaign_id,)).fetchall():
            key = row[0]
            status = cls._status(cursor, key)
            if status is TrialStatus.UNKNOWN:
                raise UnknownTrialError("Campaign contains an unknown trial; new admission is forbidden")
            if status is TrialStatus.DISPATCHED:
                cls._require_dispatch_owner(cursor, key, lease)

    @staticmethod
    def _bind_dispatch_owner(cursor: sqlite3.Cursor, trial_key: str, lease: CampaignLease) -> None:
        cursor.execute("INSERT INTO trial_dispatch_owners VALUES (?, ?, ?)",
                       (trial_key, lease.owner_id, lease.epoch))

    def create_campaign(
        self,
        contract: ExperimentContract,
        incumbent_candidate_id: str,
        candidates: Sequence[Candidate],
        *,
        plan_hash: str,
        campaign_id: str | None = None,
    ) -> str:
        if not isinstance(contract, ExperimentContract):
            raise TypeError("contract must be an ExperimentContract")
        contract_payload = contract.to_dict()
        if not isinstance(contract_payload, Mapping):
            raise TypeError("ExperimentContract.to_dict() must return a mapping")
        contract_json = _json_text(contract_payload)
        contract_hash = _sha256(contract.contract_hash, "contract_hash")
        (
            incumbent,
            normalized_plan_hash,
            candidate_ids_json,
            policy_hashes_json,
            inventory_json,
            candidate_rows,
        ) = _campaign_binding(
            contract,
            incumbent_candidate_id,
            candidates,
            plan_hash,
        )
        identifier = (
            _safe_id(campaign_id, "campaign_id")
            if campaign_id is not None
            else _deterministic_id(
                "campaign_v3_",
                {"plan_hash": normalized_plan_hash},
            )
        )
        expected = (
            contract_hash,
            contract_json,
            incumbent,
            normalized_plan_hash,
            candidate_ids_json,
            policy_hashes_json,
            inventory_json,
        )
        with self._transaction() as cursor:
            existing = cursor.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (identifier,)
            ).fetchone()
            if existing is not None:
                observed = (
                    existing["contract_hash"],
                    existing["contract_json"],
                    existing["incumbent_candidate_id"],
                    existing["plan_hash"],
                    existing["ordered_candidate_ids_json"],
                    existing["ordered_policy_hashes_json"],
                    existing["candidate_inventory_json"],
                )
                if observed != expected:
                    raise LedgerConflictError(
                        f"campaign {identifier!r} already exists with a different binding"
                    )
                self._validated_holdout_values(existing)
                self._validated_campaign_inventory(cursor, existing)
                return identifier
            holdout_nonce = secrets.token_bytes(_HOLDOUT_NONCE_BYTES)
            holdout_commitment = _holdout_commitment(holdout_nonce)
            created_at_ns = time.time_ns()
            cursor.execute(
                """INSERT INTO campaigns (
                       campaign_id, contract_hash, contract_json,
                       incumbent_candidate_id, plan_hash,
                       ordered_candidate_ids_json, ordered_policy_hashes_json,
                       candidate_inventory_json, holdout_nonce,
                       holdout_commitment, created_at_ns
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    contract_hash,
                    contract_json,
                    incumbent,
                    normalized_plan_hash,
                    candidate_ids_json,
                    policy_hashes_json,
                    inventory_json,
                    holdout_nonce,
                    holdout_commitment,
                    created_at_ns,
                ),
            )
            cursor.executemany(
                """INSERT INTO candidates (
                       campaign_id, candidate_id, policy_hash,
                       candidate_json, created_at_ns
                   ) VALUES (?, ?, ?, ?, ?)""",
                [
                    (identifier, candidate_id, policy_hash, candidate_json, created_at_ns)
                    for candidate_id, policy_hash, candidate_json in candidate_rows
                ],
            )
        return identifier

    def open_campaign(
        self,
        campaign_id: str,
        *,
        contract: ExperimentContract | None = None,
        incumbent_candidate_id: str | None = None,
        plan_hash: str | None = None,
        candidates: Sequence[Candidate] | None = None,
    ) -> dict[str, Any]:
        identifier = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                row = self._require_campaign(cursor, identifier)
                self._validated_holdout_values(row)
                self._validated_campaign_inventory(cursor, row)
                if contract is not None:
                    if not isinstance(contract, ExperimentContract):
                        raise TypeError("contract must be an ExperimentContract")
                    if (
                        row["contract_hash"] != contract.contract_hash
                        or row["contract_json"] != _json_text(contract.to_dict())
                    ):
                        raise LedgerConflictError(
                            f"campaign {identifier!r} is bound to a different contract"
                        )
                if (
                    incumbent_candidate_id is not None
                    and row["incumbent_candidate_id"]
                    != _safe_id(incumbent_candidate_id, "incumbent_candidate_id")
                ):
                    raise LedgerConflictError(
                        f"campaign {identifier!r} is bound to a different incumbent"
                    )
                if plan_hash is not None and row["plan_hash"] != _sha256(
                    plan_hash, "plan_hash"
                ):
                    raise LedgerConflictError(
                        f"campaign {identifier!r} is bound to a different plan"
                    )
                if candidates is not None:
                    effective_contract = contract or ExperimentContract.from_dict(
                        json.loads(row["contract_json"])
                    )
                    binding = _campaign_binding(
                        effective_contract,
                        row["incumbent_candidate_id"],
                        candidates,
                        row["plan_hash"] if plan_hash is None else plan_hash,
                    )
                    observed = (
                        row["incumbent_candidate_id"],
                        row["plan_hash"],
                        row["ordered_candidate_ids_json"],
                        row["ordered_policy_hashes_json"],
                        row["candidate_inventory_json"],
                    )
                    if observed != binding[:5]:
                        raise LedgerConflictError(
                            f"campaign {identifier!r} is bound to a different inventory"
                        )
                return self._campaign_dict(row)
            finally:
                cursor.close()

    create_or_open_campaign = create_campaign

    @classmethod
    def _campaign_dict(cls, row: sqlite3.Row) -> dict[str, Any]:
        _, commitment = cls._validated_holdout_values(row)
        return {
            "campaign_id": row["campaign_id"],
            "contract_hash": row["contract_hash"],
            "contract": json.loads(row["contract_json"]),
            "incumbent_candidate_id": row["incumbent_candidate_id"],
            "plan_hash": row["plan_hash"],
            "ordered_candidate_ids": json.loads(row["ordered_candidate_ids_json"]),
            "ordered_policy_hashes": json.loads(row["ordered_policy_hashes_json"]),
            "candidate_inventory": json.loads(row["candidate_inventory_json"]),
            "holdout_seed_commitment": commitment,
            "created_at_ns": row["created_at_ns"],
        }

    def _get_holdout_seed_binding(
        self,
        campaign_id: str,
        *,
        plan_hash: str,
        candidates: Sequence[Candidate],
        capability: object,
    ) -> tuple[bytes, str]:
        """Return hidden seed material only to the trusted engine capability."""

        if capability is not _ENGINE_HOLDOUT_CAPABILITY:
            raise PermissionError("holdout seed material is engine-private")
        identifier = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                row = self._require_campaign(cursor, identifier)
                contract = ExperimentContract.from_dict(json.loads(row["contract_json"]))
                binding = _campaign_binding(
                    contract,
                    row["incumbent_candidate_id"],
                    candidates,
                    plan_hash,
                )
                observed = (
                    row["incumbent_candidate_id"],
                    row["plan_hash"],
                    row["ordered_candidate_ids_json"],
                    row["ordered_policy_hashes_json"],
                    row["candidate_inventory_json"],
                )
                if observed != binding[:5]:
                    raise LedgerConflictError(
                        f"campaign {identifier!r} plan is not exactly bound"
                    )
                self._validated_campaign_inventory(cursor, row)
                return self._validated_holdout_values(row)
            finally:
                cursor.close()

    def get_holdout_seed_commitment(self, campaign_id: str) -> str:
        """Return the public commitment to hidden holdout-seed material."""

        identifier = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                row = self._require_campaign(cursor, identifier)
                _, commitment = self._validated_holdout_values(row)
                return commitment
            finally:
                cursor.close()

    @staticmethod
    def _validated_holdout_values(row: sqlite3.Row) -> tuple[bytes, str]:
        nonce = row["holdout_nonce"]
        commitment = row["holdout_commitment"]
        if not isinstance(nonce, bytes) or len(nonce) != _HOLDOUT_NONCE_BYTES:
            raise ExperimentLedgerError("campaign holdout nonce is invalid")
        if not isinstance(commitment, str) or _SHA256_RE.fullmatch(commitment) is None:
            raise ExperimentLedgerError("campaign holdout commitment is invalid")
        expected = _holdout_commitment(nonce)
        if not hmac.compare_digest(commitment, expected):
            raise ExperimentLedgerError(
                "campaign holdout commitment does not match its nonce"
            )
        return nonce, commitment

    @staticmethod
    def _validated_campaign_inventory(
        cursor: sqlite3.Cursor,
        campaign: sqlite3.Row,
    ) -> tuple[dict[str, Any], ...]:
        """Validate both copies of the sealed inventory and return it in plan order."""

        try:
            ids = json.loads(campaign["ordered_candidate_ids_json"])
            hashes = json.loads(campaign["ordered_policy_hashes_json"])
            inventory = json.loads(campaign["candidate_inventory_json"])
            contract_payload = json.loads(campaign["contract_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ExperimentLedgerError("campaign plan binding is invalid") from exc
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(hashes, list)
            or not isinstance(inventory, list)
            or len(ids) != len(hashes)
            or len(ids) != len(inventory)
        ):
            raise ExperimentLedgerError("campaign candidate inventory is invalid")
        if (
            _json_text(ids) != campaign["ordered_candidate_ids_json"]
            or _json_text(hashes) != campaign["ordered_policy_hashes_json"]
            or _json_text(inventory) != campaign["candidate_inventory_json"]
        ):
            raise ExperimentLedgerError("campaign candidate inventory is not canonical")
        try:
            contract = ExperimentContract.from_dict(contract_payload)
        except (TypeError, ValueError) as exc:
            raise ExperimentLedgerError("campaign contract bytes are invalid") from exc
        if (
            contract.contract_hash != campaign["contract_hash"]
            or _json_text(contract.to_dict()) != campaign["contract_json"]
        ):
            raise ExperimentLedgerError("campaign contract binding is invalid")

        rows = cursor.execute(
            "SELECT * FROM candidates WHERE campaign_id = ?",
            (campaign["campaign_id"],),
        ).fetchall()
        by_id = {row["candidate_id"]: row for row in rows}
        if len(by_id) != len(ids) or set(by_id) != set(ids):
            raise ExperimentLedgerError(
                "campaign candidate rows do not exactly match the sealed inventory"
            )
        seen: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for candidate_id, policy_hash, payload in zip(ids, hashes, inventory, strict=True):
            try:
                candidate_id = _safe_id(candidate_id, "candidate_id")
                policy_hash = _sha256(policy_hash, "policy_hash")
                candidate = Candidate.from_dict(payload, contract=contract)
            except (TypeError, ValueError) as exc:
                raise ExperimentLedgerError("sealed candidate inventory is invalid") from exc
            if (
                candidate.candidate_id != candidate_id
                or candidate.policy_hash != policy_hash
                or (candidate.parent is not None and candidate.parent not in seen)
            ):
                raise ExperimentLedgerError("sealed candidate inventory binding drifted")
            row = by_id[candidate_id]
            candidate_json = _json_text(candidate.to_dict())
            if row["policy_hash"] != policy_hash or row["candidate_json"] != candidate_json:
                raise ExperimentLedgerError(
                    "campaign candidate row differs from its sealed inventory"
                )
            normalized.append(json.loads(candidate_json))
            seen.add(candidate_id)
        if ids[0] != campaign["incumbent_candidate_id"]:
            raise ExperimentLedgerError("campaign incumbent is not first in its inventory")
        return tuple(normalized)

    def register_candidate(self, campaign_id: str, candidate: Candidate) -> str:
        """Validate an already sealed candidate; campaign expansion is forbidden."""

        identifier = _safe_id(campaign_id, "campaign_id")
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be a Candidate")
        candidate_id = _safe_id(candidate.candidate_id, "candidate_id")
        policy_hash = _sha256(candidate.policy_hash, "policy_hash")
        payload = candidate.to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("Candidate.to_dict() must return a mapping")
        candidate_json = _json_text(payload)
        with self._transaction() as cursor:
            campaign = self._require_campaign(cursor, identifier)
            self._validated_campaign_inventory(cursor, campaign)
            if candidate.contract_hash != campaign["contract_hash"]:
                raise LedgerConflictError(
                    f"candidate {candidate_id!r} belongs to a different contract"
                )
            existing = cursor.execute(
                """SELECT * FROM candidates
                   WHERE campaign_id = ? AND candidate_id = ?""",
                (identifier, candidate_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["policy_hash"] != policy_hash
                    or existing["candidate_json"] != candidate_json
                ):
                    raise LedgerConflictError(
                        f"candidate {candidate_id!r} already has a different payload"
                    )
                return candidate_id
            raise LedgerConflictError(
                f"candidate {candidate_id!r} is not in the sealed campaign inventory"
            )

    def prepare_trial(
        self,
        campaign_id: str,
        candidate_id: str,
        split: str,
        phase: str,
        seed: int,
        *,
        lease: CampaignLease,
        evaluator_hash: str,
        ceiling_microusd: int,
    ) -> str:
        identifier = _safe_id(campaign_id, "campaign_id")
        candidate = _safe_id(candidate_id, "candidate_id")
        split_name = _nonempty(split, "split", maximum=128)
        phase_name = _nonempty(phase, "phase", maximum=128)
        normalized_seed = _seed(seed)
        evaluator = _sha256(evaluator_hash, "evaluator_hash")
        ceiling = _nonnegative_int(ceiling_microusd, "ceiling_microusd")
        trial_key = self.make_trial_key(
            identifier, candidate, split_name, phase_name, normalized_seed
        )
        prepared_payload = {
            "trial_key": trial_key,
            "campaign_id": identifier,
            "candidate_id": candidate,
            "split": split_name,
            "phase": phase_name,
            "seed": normalized_seed,
            "evaluator_hash": evaluator,
            "ceiling_microusd": ceiling,
        }
        prepared_json = _json_text(prepared_payload)
        with self._leased_transaction(lease, campaign_id=identifier) as cursor:
            self._require_candidate(cursor, identifier, candidate)
            existing = cursor.execute(
                "SELECT * FROM trials WHERE trial_key = ?", (trial_key,)
            ).fetchone()
            if existing is not None:
                if existing["prepared_json"] != prepared_json:
                    raise LedgerConflictError(
                        f"trial key {trial_key!r} already has a different preparation"
                    )
                if self._status(cursor, trial_key) is TrialStatus.UNKNOWN:
                    raise UnknownTrialError(
                        f"trial {trial_key!r} has an unknown outcome and cannot be reused"
                    )
                if self._status(cursor, trial_key) is TrialStatus.PREPARED:
                    self._require_admission(cursor, lease)
                return trial_key
            self._require_admission(cursor, lease)
            campaign = self._require_campaign(cursor, identifier)
            terminal = cursor.execute(
                "SELECT decision_id FROM promotion_decisions WHERE campaign_id = ?",
                (identifier,),
            ).fetchone()
            if terminal is not None:
                raise LedgerConflictError(
                    f"campaign {identifier!r} already has a terminal decision"
                )
            contract = json.loads(campaign["contract_json"])
            required_ceiling = contract["per_trial_reservation_microusd"]
            if ceiling != required_ceiling:
                raise LedgerConflictError(
                    f"trial ceiling must equal contract reservation {required_ceiling}"
                )
            trial_count = cursor.execute(
                "SELECT COUNT(*) AS count FROM trials WHERE campaign_id = ?",
                (identifier,),
            ).fetchone()["count"]
            if trial_count >= contract["max_trials"]:
                raise LedgerConflictError(f"campaign {identifier!r} reached max_trials")
            exposure = self._campaign_exposure(cursor, identifier)
            if exposure + ceiling > contract["max_budget_microusd"]:
                raise LedgerBudgetError(
                    f"trial reservation would expose {exposure + ceiling} microusd "
                    f"against a {contract['max_budget_microusd']} microusd budget"
                )
            cursor.execute(
                """INSERT INTO trials (
                       trial_key, campaign_id, candidate_id, split, phase, seed,
                       evaluator_hash, ceiling_microusd, prepared_json, prepared_at_ns
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trial_key,
                    identifier,
                    candidate,
                    split_name,
                    phase_name,
                    normalized_seed,
                    evaluator,
                    ceiling,
                    prepared_json,
                    time.time_ns(),
                ),
            )
        return trial_key

    def mark_trial_dispatched(self, trial_key: str, *, lease: CampaignLease) -> TrialRecord:
        key = _nonempty(trial_key, "trial_key", maximum=128)
        with self._leased_transaction(lease, trial_key=key) as cursor:
            self._require_trial(cursor, key)
            status = self._status(cursor, key)
            if status is TrialStatus.PREPARED:
                self._require_admission(cursor, lease)
                self._bind_dispatch_owner(cursor, key, lease)
                self._insert_event(cursor, key, "dispatched", {})
            elif status is TrialStatus.DISPATCHED:
                self._require_dispatch_owner(cursor, key, lease)
            elif status is TrialStatus.UNKNOWN:
                raise UnknownTrialError(
                    f"trial {key!r} has an unknown outcome and cannot be dispatched again"
                )
            elif status is TrialStatus.COMPLETED:
                raise TrialStateError(f"completed trial {key!r} cannot be dispatched again")
        return self.get_trial(key)

    def claim_trial_dispatch(self, trial_key: str, *, lease: CampaignLease) -> TrialRecord:
        """Atomically claim a prepared trial for exactly one dispatcher.

        Unlike :meth:`mark_trial_dispatched`, this operation is intentionally not
        idempotent.  A successful claim is the durable boundary after which an
        evaluator may have produced external effects, so a second worker must
        never interpret an existing ``dispatched`` event as its own claim.
        """

        key = _nonempty(trial_key, "trial_key", maximum=128)
        with self._leased_transaction(lease, trial_key=key) as cursor:
            trial = self._require_trial(cursor, key)
            status = self._status(cursor, key)
            if status is TrialStatus.UNKNOWN:
                raise UnknownTrialError(
                    f"trial {key!r} has an unknown outcome and cannot be dispatched again"
                )
            if status is not TrialStatus.PREPARED:
                raise TrialStateError(
                    f"trial {key!r} cannot be claimed from {status.value!r} state"
                )
            self._require_admission(cursor, lease)
            self._bind_dispatch_owner(cursor, key, lease)
            self._insert_event(cursor, key, "dispatched", {})
            return self._materialize(cursor, trial)

    def complete_trial(
        self,
        trial_key: str,
        *,
        lease: CampaignLease,
        metrics: Mapping[str, int | float],
        gates: Mapping[str, bool],
        actual_cost_microusd: int,
        duration_ms: float,
        artifact_hashes: Iterable[str] = (),
        evaluator_hash: str | None = None,
    ) -> TrialRecord:
        key = _nonempty(trial_key, "trial_key", maximum=128)
        normalized_metrics = _metrics(metrics)
        normalized_gates = _gates(gates)
        cost = _nonnegative_int(actual_cost_microusd, "actual_cost_microusd")
        normalized_duration = _duration(duration_ms)
        hashes = _hashes(artifact_hashes)
        payload = {
            "metrics": normalized_metrics,
            "gates": normalized_gates,
            "actual_cost_microusd": cost,
            "duration_ms": normalized_duration,
            "artifact_hashes": list(hashes),
        }
        payload_json = _json_text(payload)
        with self._leased_transaction(lease, trial_key=key) as cursor:
            trial = self._require_trial(cursor, key)
            if evaluator_hash is not None and trial["evaluator_hash"] != _sha256(
                evaluator_hash, "evaluator_hash"
            ):
                raise LedgerConflictError(f"trial {key!r} evaluator hash changed")
            status = self._status(cursor, key)
            if status is TrialStatus.COMPLETED:
                existing = self._event(cursor, key, "completed")
                if existing is None or existing["payload_json"] != payload_json:
                    raise LedgerConflictError(
                        f"completed trial {key!r} has different result bytes"
                    )
                return self._materialize(cursor, trial)
            if status is TrialStatus.UNKNOWN:
                raise UnknownTrialError(
                    f"trial {key!r} has an unknown outcome and cannot be completed"
                )
            if status is not TrialStatus.DISPATCHED:
                raise TrialStateError(f"trial {key!r} must be dispatched before completion")
            self._require_dispatch_owner(cursor, key, lease)
            self._insert_event(cursor, key, "completed", payload)
        return self.get_trial(key)

    def mark_trial_unknown(self, trial_key: str, error: str, *, lease: CampaignLease) -> TrialRecord:
        key = _nonempty(trial_key, "trial_key", maximum=128)
        message = _nonempty(error, "error", maximum=4096)
        payload = {"error": message}
        payload_json = _json_text(payload)
        with self._leased_transaction(lease, trial_key=key) as cursor:
            trial = self._require_trial(cursor, key)
            status = self._status(cursor, key)
            if status is TrialStatus.UNKNOWN:
                existing = self._event(cursor, key, "unknown")
                if existing is None or existing["payload_json"] != payload_json:
                    raise LedgerConflictError(
                        f"unknown trial {key!r} has different error bytes"
                    )
                return self._materialize(cursor, trial)
            if status is TrialStatus.COMPLETED:
                raise TrialStateError(f"completed trial {key!r} cannot become unknown")
            if status is not TrialStatus.DISPATCHED:
                raise TrialStateError(f"trial {key!r} must be dispatched before unknown")
            self._require_dispatch_owner(cursor, key, lease)
            self._insert_event(cursor, key, "unknown", payload)
        return self.get_trial(key)

    def append_trial(self, record: TrialRecord, *, lease: CampaignLease) -> TrialRecord:
        """Append a completed record only after its dispatch boundary exists."""
        if not isinstance(record, TrialRecord):
            raise TypeError("record must be a TrialRecord")
        if record.status is not TrialStatus.COMPLETED:
            raise ValueError("append_trial requires a completed TrialRecord")
        key = self.make_trial_key(
            record.campaign_id,
            record.candidate_id,
            record.split,
            record.phase,
            record.seed,
        )
        if key != record.trial_key:
            raise LedgerConflictError("TrialRecord.trial_key does not match its key fields")
        return self.complete_trial(
            key,
            lease=lease,
            metrics=record.metrics,
            gates=record.gates,
            actual_cost_microusd=record.actual_cost_microusd or 0,
            duration_ms=record.duration_ms or 0,
            artifact_hashes=record.artifact_hashes,
            evaluator_hash=record.evaluator_hash,
        )

    def get_trial(self, trial_key: str) -> TrialRecord:
        key = _nonempty(trial_key, "trial_key", maximum=128)
        with self._lock:
            cursor = self._connection.cursor()
            try:
                trial = self._require_trial(cursor, key)
                return self._materialize(cursor, trial)
            finally:
                cursor.close()

    def list_trials(
        self,
        campaign_id: str,
        *,
        candidate_id: str | None = None,
        split: str | None = None,
        phase: str | None = None,
    ) -> list[TrialRecord]:
        identifier = _safe_id(campaign_id, "campaign_id")
        clauses = ["campaign_id = ?"]
        values: list[object] = [identifier]
        if candidate_id is not None:
            clauses.append("candidate_id = ?")
            values.append(_safe_id(candidate_id, "candidate_id"))
        if split is not None:
            clauses.append("split = ?")
            values.append(_nonempty(split, "split", maximum=128))
        if phase is not None:
            clauses.append("phase = ?")
            values.append(_nonempty(phase, "phase", maximum=128))
        query = (
            "SELECT * FROM trials WHERE "
            + " AND ".join(clauses)
            + " ORDER BY candidate_id, split, phase, seed"
        )
        with self._lock:
            cursor = self._connection.cursor()
            try:
                self._require_campaign(cursor, identifier)
                rows = cursor.execute(query, values).fetchall()
                return [self._materialize(cursor, row) for row in rows]
            finally:
                cursor.close()

    @classmethod
    def _validate_decision_payload(
        cls,
        cursor: sqlite3.Cursor,
        campaign: sqlite3.Row,
        payload: Mapping[str, Any],
    ) -> None:
        """Require terminal evidence to be the exact completed plan prefix."""

        trial_keys = payload.get("trial_keys")
        if not isinstance(trial_keys, list) or not trial_keys:
            raise LedgerConflictError("promotion decision requires non-empty evidence")
        if len(set(trial_keys)) != len(trial_keys):
            raise LedgerConflictError("promotion decision evidence contains duplicates")
        candidate_id = _safe_id(payload.get("candidate_id"), "candidate_id")
        inventory = cls._validated_campaign_inventory(cursor, campaign)
        candidate_ids = tuple(item["candidate_id"] for item in inventory)
        incumbent_id = campaign["incumbent_candidate_id"]
        if candidate_id == incumbent_id or candidate_id not in candidate_ids:
            raise LedgerConflictError(
                "promotion decision candidate must be a sealed challenger"
            )

        assessment = payload.get("assessment")
        if not isinstance(assessment, Mapping):
            raise LedgerConflictError("promotion decision assessment is invalid")
        if assessment.get("optimization_plan_hash") != campaign["plan_hash"]:
            raise LedgerConflictError("promotion decision plan hash does not match")
        _, commitment = cls._validated_holdout_values(campaign)
        if assessment.get("holdout_seed_commitment") != commitment:
            raise LedgerConflictError(
                "promotion decision holdout commitment does not match"
            )

        promoted = payload.get("promoted")
        if not isinstance(promoted, bool):
            raise LedgerConflictError("promotion decision promoted flag is invalid")
        contract = json.loads(campaign["contract_json"])
        split_plan: dict[str, tuple[tuple[str, ...], int]] = {
            "development": (
                candidate_ids,
                contract["development_repetitions"],
            )
        }
        if assessment.get("stage") == "development":
            if promoted:
                raise LedgerConflictError("development-only decision cannot promote")
        else:
            confirmation = assessment.get("confirmation")
            holdout = assessment.get("holdout")
            if not isinstance(confirmation, Mapping) or not isinstance(
                confirmation.get("promote"), bool
            ):
                raise LedgerConflictError(
                    "terminal decision requires a confirmation assessment"
                )
            split_plan["confirmation"] = (
                (incumbent_id, candidate_id),
                contract["confirmation_repetitions"],
            )
            confirmation_promoted = confirmation["promote"]
            if confirmation_promoted:
                if not isinstance(holdout, Mapping) or not isinstance(
                    holdout.get("promote"), bool
                ):
                    raise LedgerConflictError(
                        "passed confirmation requires a holdout assessment"
                    )
                split_plan["holdout"] = (
                    (incumbent_id, candidate_id),
                    contract["holdout_repetitions"],
                )
                expected_promoted = bool(holdout["promote"])
            else:
                if holdout is not None:
                    raise LedgerConflictError(
                        "rejected confirmation cannot include holdout evidence"
                    )
                expected_promoted = False
            if promoted is not expected_promoted:
                raise LedgerConflictError(
                    "promotion flag disagrees with confirmation/holdout assessments"
                )

        rows = cursor.execute(
            "SELECT * FROM trials WHERE campaign_id = ?",
            (campaign["campaign_id"],),
        ).fetchall()
        if {row["trial_key"] for row in rows} != set(trial_keys):
            raise LedgerConflictError(
                "promotion decision evidence must exactly cover the campaign trial inventory"
            )
        rows_by_split: dict[str, list[sqlite3.Row]] = {}
        evaluator_hashes: set[str] = set()
        for row in rows:
            if cls._status(cursor, row["trial_key"]) is not TrialStatus.COMPLETED:
                raise TrialStateError(
                    f"evidence trial {row['trial_key']!r} is not completed"
                )
            rows_by_split.setdefault(row["split"], []).append(row)
            evaluator_hashes.add(row["evaluator_hash"])
            if row["ceiling_microusd"] != contract["per_trial_reservation_microusd"]:
                raise LedgerConflictError(
                    f"evidence trial {row['trial_key']!r} reservation drifted"
                )
        if set(rows_by_split) != set(split_plan):
            raise LedgerConflictError(
                "promotion decision evidence contains an unplanned or missing split"
            )
        if len(evaluator_hashes) != 1:
            raise LedgerConflictError("promotion decision mixes evaluator identities")

        plan_suffix = campaign["plan_hash"].removeprefix("sha256:")
        for split, (required_candidates, repetitions) in split_plan.items():
            split_rows = rows_by_split[split]
            if len(split_rows) != len(required_candidates) * repetitions:
                raise LedgerConflictError(
                    f"promotion decision has incomplete {split} evidence"
                )
            seeds_by_candidate: dict[str, set[int]] = {
                item: set() for item in required_candidates
            }
            for row in split_rows:
                observed_candidate = row["candidate_id"]
                if observed_candidate not in seeds_by_candidate:
                    raise LedgerConflictError(
                        f"{split} evidence belongs to an unrelated candidate"
                    )
                expected_role = (
                    "incumbent" if observed_candidate == incumbent_id else "challenger"
                )
                expected_phase = f"{expected_role}.plan_{plan_suffix}"
                if row["phase"] != expected_phase:
                    raise LedgerConflictError(
                        f"{split} evidence is not bound to the campaign plan"
                    )
                seeds_by_candidate[observed_candidate].add(row["seed"])
            seed_sets = tuple(seeds_by_candidate.values())
            if any(len(seeds) != repetitions for seeds in seed_sets) or any(
                seeds != seed_sets[0] for seeds in seed_sets[1:]
            ):
                raise LedgerConflictError(
                    f"promotion decision {split} evidence is not exactly paired"
                )

    def validate_decision_inventory(self, campaign_id: str) -> tuple[str, ...]:
        """Fail closed unless the campaign has zero or one exact terminal decision."""

        identifier = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            cursor = self._connection.cursor()
            try:
                campaign = self._require_campaign(cursor, identifier)
                self._validated_campaign_inventory(cursor, campaign)
                rows = cursor.execute(
                    """SELECT * FROM promotion_decisions
                       WHERE campaign_id = ? ORDER BY created_at_ns, decision_id""",
                    (identifier,),
                ).fetchall()
                if len(rows) > 1:
                    raise LedgerConflictError(
                        "campaign contains conflicting terminal decisions"
                    )
                for row in rows:
                    try:
                        payload = json.loads(row["payload_json"])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ExperimentLedgerError(
                            "campaign decision payload is invalid"
                        ) from exc
                    if not isinstance(payload, dict) or _json_text(payload) != row[
                        "payload_json"
                    ]:
                        raise ExperimentLedgerError(
                            "campaign decision payload is not canonical"
                        )
                    reconstructed = PromotionDecision(
                        campaign_id=payload.get("campaign_id"),
                        candidate_id=payload.get("candidate_id"),
                        promoted=payload.get("promoted"),
                        reason=payload.get("reason"),
                        trial_keys=tuple(payload.get("trial_keys", ())),
                        assessment=payload.get("assessment", {}),
                    )
                    if (
                        reconstructed.decision_id != row["decision_id"]
                        or reconstructed._payload_json != row["payload_json"]
                    ):
                        raise ExperimentLedgerError(
                            "campaign decision identity does not match its payload"
                        )
                    self._validate_decision_payload(cursor, campaign, payload)
                return tuple(row["decision_id"] for row in rows)
            finally:
                cursor.close()

    def append_decision(self, decision: PromotionDecision, *, lease: CampaignLease) -> str:
        if not isinstance(decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        # The canonical payload and ID were computed together before any caller
        # could retain mutable nested containers.  Never rebuild one from live
        # attributes while using the other as its append-only identity.
        payload_json = decision._payload_json
        decision_id = decision.decision_id
        payload = json.loads(payload_json)
        campaign_id = payload["campaign_id"]
        candidate_id = payload["candidate_id"]
        with self._leased_transaction(lease, campaign_id=campaign_id) as cursor:
            campaign = self._require_campaign(cursor, campaign_id)
            self._require_candidate(cursor, campaign_id, candidate_id)
            self._validate_decision_payload(cursor, campaign, payload)
            existing = cursor.execute(
                "SELECT * FROM promotion_decisions WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["decision_id"] == decision_id
                    and existing["payload_json"] == payload_json
                ):
                    return decision_id
                raise LedgerConflictError(
                    f"campaign {campaign_id!r} already has a different terminal decision"
                )
            cursor.execute(
                """INSERT INTO promotion_decisions (
                       decision_id, campaign_id, candidate_id,
                       payload_json, created_at_ns
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    decision_id,
                    campaign_id,
                    candidate_id,
                    payload_json,
                    time.time_ns(),
                ),
            )
        return decision_id

    append_promotion_decision = append_decision

    def list_decisions(self, campaign_id: str) -> list[dict[str, Any]]:
        identifier = _safe_id(campaign_id, "campaign_id")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM promotion_decisions
                   WHERE campaign_id = ? ORDER BY created_at_ns, decision_id""",
                (identifier,),
            ).fetchall()
        return [
            {
                "decision_id": row["decision_id"],
                **json.loads(row["payload_json"]),
                "created_at_ns": row["created_at_ns"],
            }
            for row in rows
        ]

    def snapshot(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.open_campaign(campaign_id)
        identifier = campaign["campaign_id"]
        with self._lock:
            candidate_rows = self._connection.execute(
                """SELECT * FROM candidates WHERE campaign_id = ?
                   ORDER BY candidate_id""",
                (identifier,),
            ).fetchall()
        trials = self.list_trials(identifier)
        decisions = self.list_decisions(identifier)
        counts = {status.value: 0 for status in TrialStatus}
        confirmed = 0
        reserved = 0
        unknown = 0
        for trial in trials:
            counts[trial.status.value] += 1
            if trial.status is TrialStatus.COMPLETED:
                confirmed += trial.actual_cost_microusd or 0
            elif trial.status is TrialStatus.UNKNOWN:
                unknown += trial.ceiling_microusd
            else:
                reserved += trial.ceiling_microusd
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "policy_hash": row["policy_hash"],
                "candidate": json.loads(row["candidate_json"]),
                "created_at_ns": row["created_at_ns"],
            }
            for row in candidate_rows
        ]
        return {
            "campaign": campaign,
            "spent_microusd": confirmed,
            "cost": {
                "confirmed_microusd": confirmed,
                "reserved_microusd": reserved,
                "unknown_exposure_microusd": unknown,
                "total_exposure_microusd": confirmed + reserved + unknown,
            },
            "trial_count": len(trials),
            "trial_counts": {key: value for key, value in counts.items() if value},
            "candidate_count": len(candidates),
            "candidates": candidates,
            "decision_count": len(decisions),
            "decisions": decisions,
        }

    @staticmethod
    def _require_campaign(cursor: sqlite3.Cursor, campaign_id: str) -> sqlite3.Row:
        row = cursor.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise CampaignNotFoundError(campaign_id)
        return row

    @staticmethod
    def _require_candidate(
        cursor: sqlite3.Cursor, campaign_id: str, candidate_id: str
    ) -> sqlite3.Row:
        row = cursor.execute(
            """SELECT * FROM candidates
               WHERE campaign_id = ? AND candidate_id = ?""",
            (campaign_id, candidate_id),
        ).fetchone()
        if row is None:
            raise CandidateNotFoundError(f"{campaign_id}:{candidate_id}")
        return row

    @staticmethod
    def _require_trial(cursor: sqlite3.Cursor, trial_key: str) -> sqlite3.Row:
        row = cursor.execute(
            "SELECT * FROM trials WHERE trial_key = ?", (trial_key,)
        ).fetchone()
        if row is None:
            raise TrialNotFoundError(trial_key)
        return row

    @staticmethod
    def _event(
        cursor: sqlite3.Cursor, trial_key: str, event_type: str
    ) -> sqlite3.Row | None:
        return cursor.execute(
            """SELECT * FROM trial_events
               WHERE trial_key = ? AND event_type = ?""",
            (trial_key, event_type),
        ).fetchone()

    @classmethod
    def _status(cls, cursor: sqlite3.Cursor, trial_key: str) -> TrialStatus:
        event_types = {
            row["event_type"]
            for row in cursor.execute(
                "SELECT event_type FROM trial_events WHERE trial_key = ?", (trial_key,)
            )
        }
        if "completed" in event_types and "unknown" in event_types:
            raise ExperimentLedgerError(
                f"trial {trial_key!r} has conflicting terminal events"
            )
        if "unknown" in event_types:
            return TrialStatus.UNKNOWN
        if "completed" in event_types:
            return TrialStatus.COMPLETED
        if "dispatched" in event_types:
            return TrialStatus.DISPATCHED
        return TrialStatus.PREPARED

    @classmethod
    def _campaign_exposure(cls, cursor: sqlite3.Cursor, campaign_id: str) -> int:
        exposure = 0
        for trial in cursor.execute(
            "SELECT * FROM trials WHERE campaign_id = ?", (campaign_id,)
        ).fetchall():
            status = cls._status(cursor, trial["trial_key"])
            if status is TrialStatus.COMPLETED:
                event = cls._event(cursor, trial["trial_key"], "completed")
                assert event is not None
                exposure += json.loads(event["payload_json"])["actual_cost_microusd"]
            else:
                exposure += trial["ceiling_microusd"]
        return exposure

    @staticmethod
    def _insert_event(
        cursor: sqlite3.Cursor,
        trial_key: str,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        payload_json = _json_text(payload)
        event_id = _deterministic_id(
            "trial_event_v1_",
            {"trial_key": trial_key, "event_type": event_type, "payload": payload},
        )
        cursor.execute(
            """INSERT INTO trial_events (
                   event_id, trial_key, event_type, payload_json, created_at_ns
               ) VALUES (?, ?, ?, ?, ?)""",
            (event_id, trial_key, event_type, payload_json, time.time_ns()),
        )

    @classmethod
    def _materialize(cls, cursor: sqlite3.Cursor, trial: sqlite3.Row) -> TrialRecord:
        dispatched = cls._event(cursor, trial["trial_key"], "dispatched")
        completed = cls._event(cursor, trial["trial_key"], "completed")
        unknown = cls._event(cursor, trial["trial_key"], "unknown")
        status = cls._status(cursor, trial["trial_key"])
        result: dict[str, Any] = {}
        error: str | None = None
        terminal_at_ns: int | None = None
        if completed is not None:
            result = json.loads(completed["payload_json"])
            terminal_at_ns = completed["created_at_ns"]
        elif unknown is not None:
            error = json.loads(unknown["payload_json"])["error"]
            terminal_at_ns = unknown["created_at_ns"]
        return TrialRecord(
            trial_key=trial["trial_key"],
            campaign_id=trial["campaign_id"],
            candidate_id=trial["candidate_id"],
            split=trial["split"],
            phase=trial["phase"],
            seed=trial["seed"],
            evaluator_hash=trial["evaluator_hash"],
            ceiling_microusd=trial["ceiling_microusd"],
            status=status,
            metrics=result.get("metrics", {}),
            gates=result.get("gates", {}),
            actual_cost_microusd=result.get("actual_cost_microusd"),
            duration_ms=result.get("duration_ms"),
            artifact_hashes=tuple(result.get("artifact_hashes", ())),
            error=error,
            prepared_at_ns=trial["prepared_at_ns"],
            dispatched_at_ns=dispatched["created_at_ns"] if dispatched else None,
            terminal_at_ns=terminal_at_ns,
        )


__all__ = [
    "CampaignLease",
    "CampaignLeaseError",
    "CampaignLeaseConflict",
    "CampaignNotFoundError",
    "CandidateNotFoundError",
    "ExperimentLedger",
    "ExperimentLedgerError",
    "LedgerConflictError",
    "LedgerBudgetError",
    "PromotionDecision",
    "TrialNotFoundError",
    "TrialRecord",
    "TrialStateError",
    "TrialStatus",
    "UnknownTrialError",
]
