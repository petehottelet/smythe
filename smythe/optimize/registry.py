"""Append-only SQLite registry of the policy version in effect for each target.

A target is a dotted policy name, such as ``planner.template.report``, with one
active pointer. Every change appends the target's next event: ``baseline`` or
``activate`` puts a policy in effect, ``rollback`` restores an earlier
``baseline`` or ``activate`` entry, and ``retire`` leaves no policy in effect.
A writer names the sequence number it last saw, so two operators cannot both
change a target from the same view.

The registry is its own file and never writes an Autotune ledger. Its review
records name ledger decisions by identity, never by path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smythe._sqlite import enable_wal
from smythe.optimize.contracts import _DOTTED_FIELD_RE
from smythe.optimize.ledger import (
    _SQLITE_INT_MAX,
    _create_private_file,
    _deep_freeze_json,
    _json_text,
    _nonempty,
)


REGISTRY_VERSION = 1
_WRITER_FUNCTION = "smythe_policy_registry_writer_version"
_ACTIONS = ("baseline", "activate", "rollback", "retire")
# The actions that put a policy of their own in effect.
_RESTORABLE = _ACTIONS[:2]
_TABLES = {
    "registry_meta": ("version", "created_at_ns"),
    # Written by review recording; an activate event must name one.
    "reviews": ("review_id", "campaign_id", "decision_id", "verdict", "reviewer_ref",
                "payload_json", "created_at_ns"),
    "activations": ("target", "seq", "action", "policy_hash", "policy_json", "review_id",
                    "restores_seq", "reason", "actor_ref", "created_at_ns"),
}
_SCHEMA = """
    CREATE TABLE registry_meta (
        version INTEGER PRIMARY KEY,
        created_at_ns INTEGER NOT NULL
    );
    CREATE TABLE reviews (
        review_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        verdict TEXT NOT NULL CHECK (verdict IN ('approve', 'reject')),
        reviewer_ref TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at_ns INTEGER NOT NULL
    );
    CREATE TABLE activations (
        target TEXT NOT NULL,
        seq INTEGER NOT NULL CHECK (typeof(seq) = 'integer' AND seq > 0),
        action TEXT NOT NULL CHECK (action IN ('baseline', 'activate', 'rollback', 'retire')),
        policy_hash TEXT,
        policy_json TEXT,
        review_id TEXT REFERENCES reviews(review_id),
        restores_seq INTEGER,
        reason TEXT NOT NULL,
        actor_ref TEXT NOT NULL,
        created_at_ns INTEGER NOT NULL,
        UNIQUE (target, seq),
        FOREIGN KEY (target, restores_seq) REFERENCES activations(target, seq),
        CHECK ((action = 'retire') = (policy_json IS NULL)
               AND (policy_json IS NULL) = (policy_hash IS NULL)),
        CHECK ((action = 'rollback') = (restores_seq IS NOT NULL)),
        CHECK (restores_seq IS NULL OR restores_seq < seq),
        CHECK (action != 'activate' OR review_id IS NOT NULL),
        CHECK (action != 'retire' OR review_id IS NULL)
    );
    CREATE INDEX reviews_by_decision ON reviews(decision_id)
"""


def _barrier() -> dict[str, str]:
    """Writer-version guards on inserts; updates and deletes are always refused."""

    triggers = {}
    for table in _TABLES:
        name = f"policy_registry_v1_{table}_"
        triggers[name + "insert"] = (
            f"CREATE TRIGGER {name}insert BEFORE INSERT ON {table} "
            f"BEGIN SELECT CASE WHEN {_WRITER_FUNCTION}() != 1 "
            "THEN RAISE(ABORT, 'Policy registry schema v1 writer required') END; END"
        )
        for operation in ("UPDATE", "DELETE"):
            triggers[name + operation.lower()] = (
                f"CREATE TRIGGER {name}{operation.lower()} BEFORE {operation} ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'Policy registry rows are append-only'); END"
            )
    return triggers


_BARRIER = _barrier()


class RegistryError(RuntimeError):
    """A policy registry cannot be opened, or refuses a change its history forbids."""


class RegistryConflictError(RegistryError):
    """The target's head moved past the caller's ``expected_seq``; nothing was written."""


class _Record:
    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        """Every field as plain JSON values, the policy as a mutable copy."""

        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        if result["policy"] is not None:
            result["policy"] = json.loads(_json_text(result["policy"]))
        return result


@dataclass(frozen=True, slots=True)
class ActivationEvent(_Record):
    """One immutable event in a target's history."""

    target: str
    seq: int
    action: str
    policy_hash: str | None
    policy: Mapping[str, Any] | None
    review_id: str | None
    restores_seq: int | None
    reason: str
    actor: str
    created_at_ns: int


@dataclass(frozen=True, slots=True)
class ActiveVersion(_Record):
    """The policy in effect for a target as of its head event ``seq``.

    ``source_seq`` is the baseline or activate entry that supplied the policy:
    the head itself, or the entry a rollback head restored.
    """

    target: str
    seq: int
    policy_hash: str
    policy: Mapping[str, Any]
    review_id: str | None
    source_seq: int


def _target(value: object) -> str:
    if not isinstance(value, str) or _DOTTED_FIELD_RE.fullmatch(value) is None:
        raise ValueError("target must be a dotted policy name such as 'planner.template.report'")
    return value


def _seq(value: object, field_name: str, *, minimum: int) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not minimum <= value <= _SQLITE_INT_MAX):
        raise ValueError(f"{field_name} must be an integer of at least {minimum}")
    return value


def _policy(target: str, policy: object) -> str:
    """Return the canonical JSON of a policy whose every path lies under *target*."""

    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("policy must be a non-empty mapping")
    for path in policy:
        if (not isinstance(path, str) or _DOTTED_FIELD_RE.fullmatch(path) is None
                or not (path == target or path.startswith(target + "."))):
            raise ValueError(f"policy path {path!r} is not target {target!r} or under it")
    return _json_text(policy)


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _in_effect(history: Sequence[ActivationEvent]) -> ActivationEvent | None:
    """The baseline or activate entry whose policy is in effect, if any."""

    if not history or history[-1].action == "retire":
        return None
    head = history[-1]
    return history[head.restores_seq - 1] if head.action == "rollback" else head


class PolicyRegistry:
    """WAL SQLite store of append-only activation events, separate from ledgers."""

    def __init__(self, path: str | Path | None = None, *, read_only: bool = False) -> None:
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a boolean")
        if path is None:
            path = Path.home() / ".smythe" / "policy-registry.sqlite3"
        self.path = Path(path).resolve()
        self.read_only = read_only
        self._lock = threading.RLock()
        self._closed = True
        self._fingerprint: tuple[int, int, int, int] | None = None
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self._fingerprint = self._stat()
            self._assert_quiescent("before opening")
            try:
                connection = sqlite3.connect(
                    self.path.as_uri() + "?mode=ro&immutable=1", uri=True,
                    isolation_level=None, check_same_thread=False,
                )
            except BaseException as exc:
                try:
                    self._assert_quiescent("after failed open")
                except RegistryError as quiescence_error:
                    raise quiescence_error from exc
                raise
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Reviews and activations are operator records: create them owner-only.
            _create_private_file(self.path)
            connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection = connection
        try:
            connection.row_factory = sqlite3.Row
            if read_only:
                # `immutable=1` trusts the quiescence check; repeat it now that
                # the file is open, and again at close.
                self._assert_quiescent("immediately after opening")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            if read_only:
                connection.execute("PRAGMA query_only = ON")
                self._verify(connection)
            else:
                connection.create_function(_WRITER_FUNCTION, 0, lambda: REGISTRY_VERSION)
                enable_wal(connection, validate_before_write=self._guard_before_write)
                connection.execute("PRAGMA synchronous = FULL")
                self._create_schema()
        except BaseException as exc:
            connection.close()
            if read_only:
                try:
                    self._assert_quiescent("after failed initialization")
                except RegistryError as quiescence_error:
                    raise quiescence_error from exc
            raise
        self._closed = False

    def __enter__(self) -> "PolicyRegistry":
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
            # Detects a writer that overlapped the read; not a filesystem lock.
            self._assert_quiescent("after closing")

    def _stat(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _assert_quiescent(self, stage: str) -> None:
        sidecars = [name for name in (f"{self.path.name}-wal", f"{self.path.name}-shm")
                    if self.path.with_name(name).exists()]
        if sidecars:
            raise RegistryError(
                "read-only inspection requires a closed, checkpointed policy registry; "
                f"found SQLite sidecar(s) {stage}: {', '.join(sidecars)}"
            )
        if self._fingerprint is not None and self._stat() != self._fingerprint:
            raise RegistryError(
                "read-only inspection requires an unchanged, closed, checkpointed "
                f"policy registry; database changed {stage}"
            )

    @staticmethod
    def _version(db: sqlite3.Connection | sqlite3.Cursor) -> None:
        try:
            rows = db.execute("SELECT version FROM registry_meta").fetchall()
        except sqlite3.Error as exc:
            raise RegistryError("not a Smythe policy registry") from exc
        if len(rows) != 1 or rows[0][0] != REGISTRY_VERSION:
            raise RegistryError(
                f"unsupported policy registry version {[row[0] for row in rows]}; "
                f"this release supports version {REGISTRY_VERSION}"
            )

    def _guard_before_write(self) -> None:
        names = {row[0] for row in self._connection.execute("SELECT name FROM sqlite_master")}
        if "registry_meta" in names:
            self._version(self._connection)
        elif names:
            # Never add registry tables to another database, such as a ledger.
            raise RegistryError("not a Smythe policy registry: the database holds other tables")

    def _create_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            self._guard_before_write()
            if cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'registry_meta'"
            ).fetchone() is None:
                for statement in _SCHEMA.split(";"):
                    cursor.execute(statement)
                for sql in _BARRIER.values():
                    cursor.execute(sql)
                cursor.execute("INSERT INTO registry_meta VALUES (?, ?)",
                               (REGISTRY_VERSION, time.time_ns()))
            self._verify(cursor)
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    @classmethod
    def _verify(cls, db: sqlite3.Connection | sqlite3.Cursor) -> None:
        cls._version(db)
        for table, columns in _TABLES.items():
            if tuple(row["name"] for row in db.execute(f"PRAGMA table_info({table})")) != columns:
                raise RegistryError(f"Invalid policy registry table {table}")
        for name, sql in _BARRIER.items():
            row = db.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
            ).fetchone()
            if row is None or " ".join(row["sql"].split()) != " ".join(sql.split()):
                raise RegistryError(f"Missing or altered policy registry writer barrier: {name}")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RegistryError("Policy registry has invalid references")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            if self.read_only:
                raise RegistryError("a read-only policy registry cannot be written")
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._version(cursor)
                yield cursor
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")
            finally:
                cursor.close()

    @staticmethod
    def _history(db: sqlite3.Connection | sqlite3.Cursor, target: str) -> list[ActivationEvent]:
        events: list[ActivationEvent] = []
        for row in db.execute(
            "SELECT * FROM activations WHERE target = ? ORDER BY seq", (target,)
        ).fetchall():
            text = row["policy_json"]
            policy = None
            if text is not None:
                try:
                    loaded = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    loaded = None
                if (not isinstance(loaded, dict) or _json_text(loaded) != text
                        or _digest(text) != row["policy_hash"]):
                    raise RegistryError(f"stored policy of {target!r} seq {row['seq']} is invalid")
                policy = _deep_freeze_json(loaded)
            event = ActivationEvent(
                row["target"], row["seq"], row["action"], row["policy_hash"], policy,
                row["review_id"], row["restores_seq"], row["reason"], row["actor_ref"],
                row["created_at_ns"],
            )
            # Sequence numbers are consecutive, and a rollback carries exactly
            # the policy and review of the entry it restores.
            restores = event.restores_seq
            restored = (events[restores - 1]
                        if type(restores) is int and 0 < restores <= len(events) else None)
            if event.seq != len(events) + 1 or event.action == "rollback" and (
                restored is None or restored.action not in _RESTORABLE
                or (restored.policy_hash, restored.review_id)
                != (event.policy_hash, event.review_id)
            ):
                raise RegistryError(f"stored history of {target!r} is inconsistent at seq {event.seq}")
            events.append(event)
        return events

    @staticmethod
    def _insert(cursor: sqlite3.Cursor, event: ActivationEvent) -> None:
        cursor.execute(
            """INSERT INTO activations (
                   target, seq, action, policy_hash, policy_json, review_id,
                   restores_seq, reason, actor_ref, created_at_ns
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.target, event.seq, event.action, event.policy_hash,
             None if event.policy is None else _json_text(event.policy), event.review_id,
             event.restores_seq, event.reason, event.actor, event.created_at_ns),
        )

    def _append(
        self,
        target: str,
        entries: Sequence[tuple[str, Mapping[str, Any] | None, str | None, int | None]],
        *,
        expected_seq: int,
        actor: str,
        reason: str,
    ) -> tuple[ActivationEvent, ...]:
        """Append ``(action, policy, review_id, restores_seq)`` entries atomically.

        Nothing is written unless ``expected_seq`` is the target's head (0 for
        a target without events) and every entry is a valid transition. A
        baseline must be a target's first event, an activate names a review,
        a rollback names only the earlier baseline or activate entry it
        restores and copies that entry's policy and review, and a retire needs
        a policy in effect.
        """

        name = _target(target)
        head = _seq(expected_seq, "expected_seq", minimum=0)
        actor_ref = _nonempty(actor, "actor", maximum=128)
        note = _nonempty(reason, "reason", maximum=1024)
        pending = []
        for action, policy, review_id, restores_seq in entries:
            if action not in _ACTIONS:
                raise ValueError(f"unknown registry action {action!r}")
            restorable = action in _RESTORABLE
            if restorable:
                policy = _policy(name, policy)
            elif policy is not None:
                raise ValueError(f"a {action} entry carries no policy of its own")
            if action == "rollback":
                restores_seq = _seq(restores_seq, "restores_seq", minimum=1)
            elif restores_seq is not None:
                raise ValueError(f"a {action} entry restores no entry")
            if review_id is not None:
                if not restorable:
                    raise ValueError(f"a {action} entry names no review of its own")
                review_id = _nonempty(review_id, "review_id", maximum=128)
            elif action == "activate":
                raise ValueError("an activate entry must name its review")
            pending.append((action, policy, review_id, restores_seq))
        if not pending:
            raise ValueError("entries must not be empty")

        with self._transaction() as cursor:
            history = self._history(cursor, name)
            if len(history) != head:
                raise RegistryConflictError(
                    f"target {name!r} is at seq {len(history)}, not expected_seq {head}"
                )
            appended = []
            for action, policy_json, review_id, restores_seq in pending:
                current = _in_effect(history)
                seq = len(history) + 1
                if action == "baseline" and history:
                    raise RegistryError(f"a baseline must be the first event of {name!r}")
                if action == "retire" and current is None:
                    raise RegistryError(f"target {name!r} has no policy in effect to retire")
                policy_hash = None if policy_json is None else _digest(policy_json)
                policy = None if policy_json is None else _deep_freeze_json(json.loads(policy_json))
                if action == "rollback":
                    restored = history[restores_seq - 1] if restores_seq < seq else None
                    if restored is None or restored.action not in _RESTORABLE:
                        raise RegistryError(
                            f"rollback of {name!r} must restore an earlier baseline or activate entry"
                        )
                    if restored is current:
                        raise RegistryError(f"entry {restores_seq} of {name!r} is already in effect")
                    policy_hash, policy, review_id = (
                        restored.policy_hash, restored.policy, restored.review_id)
                elif review_id is not None and cursor.execute(
                    "SELECT 1 FROM reviews WHERE review_id = ?", (review_id,)
                ).fetchone() is None:
                    raise RegistryError(f"unknown review {review_id!r}")
                event = ActivationEvent(name, seq, action, policy_hash, policy, review_id,
                                        restores_seq, note, actor_ref, time.time_ns())
                self._insert(cursor, event)
                history.append(event)
                appended.append(event)
            return tuple(appended)

    def resolve(self, target: str) -> ActiveVersion | None:
        """Return the policy in effect for *target*, or ``None`` if there is none."""

        history = self.history(target)
        entry = _in_effect(history)
        if entry is None:
            return None
        assert entry.policy_hash is not None and entry.policy is not None
        return ActiveVersion(entry.target, history[-1].seq, entry.policy_hash, entry.policy,
                             entry.review_id, entry.seq)

    def history(self, target: str) -> tuple[ActivationEvent, ...]:
        """Return every event of *target* in sequence order; empty when it has none."""

        name = _target(target)
        with self._lock:
            return tuple(self._history(self._connection, name))

    def rollback(
        self, target: str, *, to_seq: int, expected_seq: int, actor: str, reason: str
    ) -> ActivationEvent:
        """Restore an earlier baseline or activate entry as a new event.

        The entry must not be the one already in effect. A retired target can
        be restored this way.
        """

        restores = _seq(to_seq, "to_seq", minimum=1)
        (event,) = self._append(target, [("rollback", None, None, restores)],
                                expected_seq=expected_seq, actor=actor, reason=reason)
        return event

    def retire(self, target: str, *, expected_seq: int, actor: str, reason: str) -> ActivationEvent:
        """Leave *target* with no policy in effect; :meth:`resolve` then returns ``None``."""

        (event,) = self._append(target, [("retire", None, None, None)],
                                expected_seq=expected_seq, actor=actor, reason=reason)
        return event


__all__ = [
    "REGISTRY_VERSION",
    "ActivationEvent",
    "ActiveVersion",
    "PolicyRegistry",
    "RegistryConflictError",
    "RegistryError",
]
