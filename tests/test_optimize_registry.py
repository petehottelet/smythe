"""The policy registry keeps an append-only, compare-and-swap history per target.

Each target has one active pointer. Changes name the head they expect, so a
stale view writes nothing; rollback restores only an earlier baseline or
activate entry; and triggers refuse updates, deletes, inserts that would
replace a row, and inserts from another schema version. Reviews of Autotune
decisions from a fake-evaluator ledger are recorded and activated without
writing the ledger. Real child processes race for one head and are killed
inside a write transaction.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import os
import shutil
import sqlite3
import stat
import time

import pytest

from fakes import kill_and_reap, launch_child, wait_for_barrier
from smythe.optimize import (
    ActivationEvent,
    ActiveVersion,
    OptimizationRunner,
    PolicyRegistry,
    PolicyReview,
    RegistryConflictError,
    RegistryError,
    TrialContext,
    TrialOutcome,
    canonical_json_bytes,
)
from smythe.optimize.contracts import Candidate, ExperimentContract, MetricObjective, MutableFieldRule
from smythe.optimize.ledger import ExperimentLedger
from smythe.optimize.registry import _BARRIER

TARGET = "planner.template.report"
V1 = {TARGET: "v1"}
V2 = {TARGET: "v2"}
V3 = {TARGET: "v3"}
WRITER = "smythe_policy_registry_writer_version"


def _dump(path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return list(connection.iterdump())


def _review(registry: PolicyRegistry, review_id: str = "review-1") -> str:
    with registry._transaction() as cursor:
        cursor.execute(
            "INSERT INTO reviews (review_id, campaign_id, decision_id, verdict, reviewer_ref, "
            "payload_json, created_at_ns) VALUES (?, 'campaign', 'decision', 'approve', "
            "'reviewer', '{}', 1)",
            (review_id,),
        )
    return review_id


def _activate(registry, policy, *, expected_seq, review_id="review-1", target=TARGET):
    (event,) = registry._append(target, [("activate", policy, review_id, None)],
                                expected_seq=expected_seq, actor="operator", reason="reviewed")
    return event


def _activated(path) -> None:
    """A registry whose target has a baseline (seq 1) and an activation (seq 2)."""
    with PolicyRegistry(path) as registry:
        _review(registry)
        registry._append(TARGET, [("baseline", V1, None, None), ("activate", V2, "review-1", None)],
                         expected_seq=0, actor="operator", reason="first activation")


def test_each_change_appends_the_next_event_and_moves_the_active_pointer(tmp_path):
    path = tmp_path / "registry.sqlite3"
    with PolicyRegistry(path) as registry:
        assert registry.resolve(TARGET) is None and registry.history(TARGET) == ()
        _review(registry)
        baseline, activated = registry._append(
            TARGET, [("baseline", V1, "review-1", None), ("activate", V2, "review-1", None)],
            expected_seq=0, actor="operator", reason="first activation")
        assert (baseline.seq, baseline.action, activated.seq, activated.action) == (
            1, "baseline", 2, "activate")
        assert registry.resolve(TARGET) == ActiveVersion(
            TARGET, 2, activated.policy_hash, V2, "review-1", 2)

        rolled_back = registry.rollback(TARGET, to_seq=1, expected_seq=2, actor="operator",
                                        reason="v2 dropped a required section")
        assert (rolled_back.seq, rolled_back.restores_seq, rolled_back.policy) == (3, 1, V1)
        assert rolled_back.policy_hash == baseline.policy_hash
        assert registry.resolve(TARGET) == ActiveVersion(
            TARGET, 3, baseline.policy_hash, V1, "review-1", 1)

        _review(registry, "review-2")
        reactivated = _activate(registry, V3, expected_seq=3, review_id="review-2")
        retired = registry.retire(TARGET, expected_seq=4, actor="operator", reason="replaced")
        assert registry.resolve(TARGET) is None
        restored = registry.rollback(TARGET, to_seq=2, expected_seq=5, actor="operator",
                                     reason="restore v2")
        assert registry.resolve(TARGET).to_dict() == {
            "target": TARGET, "seq": 6, "policy_hash": activated.policy_hash, "policy": V2,
            "review_id": "review-1", "source_seq": 2}

        history = registry.history(TARGET)
        assert history == (baseline, activated, rolled_back, reactivated, retired, restored)
        assert [event.seq for event in history] == [1, 2, 3, 4, 5, 6]
        assert [event.to_dict() for event in history][3:5] == [
            {"target": TARGET, "seq": 4, "action": "activate",
             "policy_hash": reactivated.policy_hash, "policy": V3, "review_id": "review-2",
             "restores_seq": None, "reason": "reviewed", "actor": "operator",
             "created_at_ns": reactivated.created_at_ns},
            {"target": TARGET, "seq": 5, "action": "retire", "policy_hash": None,
             "policy": None, "review_id": None, "restores_seq": None, "reason": "replaced",
             "actor": "operator", "created_at_ns": retired.created_at_ns},
        ]
        assert all(isinstance(event, ActivationEvent) for event in history)
        # Each target has its own pointer and sequence.
        other = "planner.template.summary"
        registry._append(other, [("baseline", {other: "v1"}, None, None)],
                         expected_seq=0, actor="operator", reason="first")
        assert registry.history(TARGET) == history
        assert [event.seq for event in registry.history(other)] == [1]
    with PolicyRegistry(path, read_only=True) as reader:
        assert reader.history(TARGET) == history
    # Rows hold policies, references and notes, never the store's own location.
    assert not any(str(tmp_path) in line or tmp_path.name in line for line in _dump(path))


def test_stored_policy_hashes_match_autotune_candidate_policy_hashes(tmp_path):
    field = TARGET + ".max_sections"
    contract = ExperimentContract(
        name="report_template",
        objectives=(MetricObjective("quality", "maximize", primary=True),),
        mutable_fields=(field,),
        mutable_field_rules={field: MutableFieldRule("number", minimum=1, maximum=12)},
        required_gates=("valid",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=2,
        max_parallel_candidates=1,
        max_trials=14,
        max_wall_seconds=600,
        max_budget_microusd=0,
        per_trial_reservation_microusd=0,
        confidence=0.95,
        min_improvement=0.0,
        base_seed=1,
    )
    # The candidate normalizes 6 to 6.0; the registry hashes the same canonical policy.
    candidate = Candidate(contract, {field: 6}, "six sections cover every required topic")
    with PolicyRegistry(tmp_path / "registry.sqlite3") as registry:
        _review(registry)
        event = _activate(registry, candidate.policy, expected_seq=0)
    assert event.policy_hash == candidate.policy_hash


def test_a_stale_expected_seq_raises_a_conflict_and_writes_nothing(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    with PolicyRegistry(path) as registry:
        for stale in (0, 1, 3):
            with pytest.raises(RegistryConflictError, match=f"at seq 2, not expected_seq {stale}"):
                registry.rollback(TARGET, to_seq=1, expected_seq=stale, actor="operator",
                                  reason="stale view")
            with pytest.raises(RegistryConflictError):
                registry.retire(TARGET, expected_seq=stale, actor="operator", reason="stale view")
            with pytest.raises(RegistryConflictError):
                _activate(registry, V3, expected_seq=stale)
        assert list(registry._connection.iterdump()) == before
        assert registry.resolve(TARGET).seq == 2
    assert _dump(path) == before


@pytest.mark.parametrize("to_seq, refusal", [
    (2, "entry 2 of 'planner.template.report' is already in effect"),
    (3, "must restore an earlier baseline or activate entry"),   # a rollback entry
    (5, "must restore an earlier baseline or activate entry"),   # a retire entry
    (6, "must restore an earlier baseline or activate entry"),   # the head, a rollback
    (7, "must restore an earlier baseline or activate entry"),   # the next event
    (60, "must restore an earlier baseline or activate entry"),  # beyond the head
])
def test_rollback_restores_only_an_earlier_baseline_or_activate_entry(tmp_path, to_seq, refusal):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with PolicyRegistry(path) as registry:
        registry.rollback(TARGET, to_seq=1, expected_seq=2, actor="operator", reason="undo v2")
        registry.rollback(TARGET, to_seq=2, expected_seq=3, actor="operator", reason="redo v2")
        registry.retire(TARGET, expected_seq=4, actor="operator", reason="paused")
        registry.rollback(TARGET, to_seq=2, expected_seq=5, actor="operator", reason="resume v2")
        # The head (seq 6) restored entry 2, which is in effect again.
        assert registry.resolve(TARGET).source_seq == 2
        before = list(registry._connection.iterdump())
        with pytest.raises(RegistryError, match=refusal):
            registry.rollback(TARGET, to_seq=to_seq, expected_seq=6, actor="operator",
                              reason="refused")
        assert list(registry._connection.iterdump()) == before
        with pytest.raises(ValueError, match="to_seq"):
            registry.rollback(TARGET, to_seq=0, expected_seq=6, actor="operator", reason="no")
        restored = registry.rollback(TARGET, to_seq=1, expected_seq=6, actor="operator",
                                     reason="back to the baseline")
        assert (restored.seq, restored.restores_seq, restored.policy) == (7, 1, V1)


def test_after_retire_resolve_returns_none_until_an_entry_is_restored(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with PolicyRegistry(path) as registry:
        retired = registry.retire(TARGET, expected_seq=2, actor="operator", reason="withdrawn")
        assert (retired.seq, retired.action, retired.policy, retired.policy_hash) == (
            3, "retire", None, None)
        assert registry.resolve(TARGET) is None
        before = list(registry._connection.iterdump())
        with pytest.raises(RegistryError, match="no policy in effect to retire"):
            registry.retire(TARGET, expected_seq=3, actor="operator", reason="again")
        with pytest.raises(RegistryError, match="no policy in effect to retire"):
            registry.retire("planner.template.unknown", expected_seq=0, actor="operator",
                            reason="never activated")
        assert list(registry._connection.iterdump()) == before
        assert [event.action for event in registry.history(TARGET)] == [
            "baseline", "activate", "retire"]
    with PolicyRegistry(path, read_only=True) as reader:
        assert reader.resolve(TARGET) is None


@pytest.mark.parametrize("entries, error, message", [
    ([("activate", {"planner.template": "v3"}, "review-1", None)], ValueError, "not target"),
    ([("activate", {TARGET + "s": "v3"}, "review-1", None)], ValueError, "not target"),
    ([("activate", {}, "review-1", None)], ValueError, "non-empty mapping"),
    ([("activate", {TARGET: float("nan")}, "review-1", None)], ValueError, "finite"),
    ([("activate", V3, None, None)], ValueError, "must name its review"),
    ([("activate", V3, "review-9", None)], RegistryError, "unknown review 'review-9'"),
    ([("baseline", V3, None, None)], RegistryError, "baseline must be the first event"),
    ([("rollback", V1, None, 1)], ValueError, "carries no policy"),
    ([("rollback", None, "review-1", 1)], ValueError, "names no review"),
    ([("retire", None, None, 1)], ValueError, "restores no entry"),
    ([("promote", V3, None, None)], ValueError, "unknown registry action"),
    ([], ValueError, "must not be empty"),
    # A later invalid entry refuses the whole append.
    ([("activate", V3, "review-1", None), ("baseline", V1, None, None)],
     RegistryError, "baseline must be the first event"),
])
def test_append_refuses_an_invalid_transition_without_writing(tmp_path, entries, error, message):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with PolicyRegistry(path) as registry:
        before = list(registry._connection.iterdump())
        with pytest.raises(error, match=message):
            registry._append(TARGET, entries, expected_seq=2, actor="operator", reason="checked")
        assert list(registry._connection.iterdump()) == before


@pytest.mark.parametrize("target", ["", "planner..template", ".planner", "planner/template",
                                    "../registry", "1planner", None])
def test_targets_are_dotted_policy_names(tmp_path, target):
    with PolicyRegistry(tmp_path / "registry.sqlite3") as registry:
        with pytest.raises(ValueError, match="dotted policy name"):
            registry.resolve(target)
        with pytest.raises(ValueError, match="dotted policy name"):
            registry.retire(target, expected_seq=0, actor="operator", reason="invalid")


@pytest.mark.parametrize("actor, reason", [("", "why"), ("operator", " "), ("op\x00", "why")])
def test_every_change_records_an_actor_and_a_reason(tmp_path, actor, reason):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with PolicyRegistry(path) as registry:
        with pytest.raises(ValueError):
            registry.retire(TARGET, expected_seq=2, actor=actor, reason=reason)
        assert len(registry.history(TARGET)) == 2


TABLE_INSERTS = {
    "registry_meta": "INSERT INTO registry_meta VALUES (2, 1)",
    "reviews": "INSERT INTO reviews VALUES ('review-2', 'c', 'd', 'approve', 'r', '{}', 1)",
    "activations": ("INSERT INTO activations VALUES "
                    "('planner.template.report', 3, 'retire', NULL, NULL, NULL, NULL, 'r', 'a', 1)"),
}


@pytest.mark.parametrize("table", sorted(TABLE_INSERTS))
def test_raw_update_and_delete_raise_through_the_triggers(tmp_path, table):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        for statement in (f"UPDATE {table} SET created_at_ns = 2", f"DELETE FROM {table}"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                raw.execute(statement)
        # A client that is not a registry writer cannot insert either.
        with pytest.raises(sqlite3.OperationalError, match=WRITER):
            raw.execute(TABLE_INSERTS[table])
    with PolicyRegistry(path) as registry:
        # The current writer is refused the same way.
        for statement in (f"UPDATE {table} SET created_at_ns = 2", f"DELETE FROM {table}"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                registry._connection.execute(statement)
    assert _dump(path) == before


EVENT_AT_ROWID = ("INSERT OR REPLACE INTO activations "
                  "(rowid, target, seq, action, reason, actor_ref, created_at_ns) VALUES ({}, '{}', {}, "
                  "'retire', 'r', 'a', 1)")
SEQ_2_ROWID = "(SELECT rowid FROM activations WHERE seq = 2)"


@pytest.mark.parametrize("statement", [
    # REPLACE deletes the conflicting row without firing DELETE triggers.
    "INSERT OR REPLACE INTO activations VALUES "
    "('planner.template.report', 2, 'retire', NULL, NULL, NULL, NULL, 'r', 'a', 1)",
    "REPLACE INTO reviews VALUES ('review-1', 'c', 'd', 'reject', 'r', '{}', 1)",
    "INSERT OR REPLACE INTO registry_meta VALUES (1, 2)",
    "INSERT OR IGNORE INTO activations VALUES "
    "('planner.template.report', 1, 'retire', NULL, NULL, NULL, NULL, 'r', 'a', 1)",
    # A row named by rowid, under a new seq, target or review ID.
    EVENT_AT_ROWID.format(SEQ_2_ROWID, "planner.template.report", 3),
    EVENT_AT_ROWID.format(SEQ_2_ROWID, "planner.template.summary", 1),
    "INSERT OR REPLACE INTO reviews (rowid, review_id, campaign_id, decision_id, verdict, "
    "reviewer_ref, payload_json, created_at_ns) VALUES ((SELECT rowid FROM reviews "
    "WHERE review_id = 'review-1'), 'review-2', 'c', 'd', 'reject', 'r', '{}', 1)",
    # Rowids run 1, 2, 3 without gaps.
    EVENT_AT_ROWID.format(-1, "planner.template.report", 3),
    EVENT_AT_ROWID.format(9, "planner.template.report", 3),
    # An event that skips the target's next seq.
    "INSERT INTO activations VALUES "
    "('planner.template.report', 4, 'retire', NULL, NULL, NULL, NULL, 'r', 'a', 1)",
    "INSERT INTO activations VALUES "
    "('planner.template.summary', 2, 'retire', NULL, NULL, NULL, NULL, 'r', 'a', 1)",
])
def test_a_current_writer_cannot_replace_a_row_or_skip_an_event(tmp_path, statement):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        raw.create_function(WRITER, 0, lambda: 1)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(statement)
    assert _dump(path) == before
    with PolicyRegistry(path) as registry:
        assert [event.policy for event in registry.history(TARGET)] == [V1, V2]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            registry._connection.execute(statement)
        # Ordinary appends still pass.
        _review(registry, "review-2")
        assert registry.retire(TARGET, expected_seq=2, actor="operator", reason="paused").seq == 3


@pytest.mark.parametrize("index, statement", [
    # Replaces the target's activation (seq 2) with another target's first event.
    ("activations (reason, action)",
     "INSERT OR REPLACE INTO activations VALUES ('planner.template.summary', 1, 'activate', "
     "'sha256:x', '{}', 'review-1', NULL, 'first activation', 'a', 1)"),
    ("reviews (verdict)",
     "INSERT OR REPLACE INTO reviews VALUES ('review-2', 'c', 'd', 'approve', 'r', '{}', 1)"),
])
def test_a_current_writer_cannot_replace_a_row_through_an_added_unique_index(
    tmp_path, index, statement,
):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        raw.create_function(WRITER, 0, lambda: 1)
        raw.execute(f"CREATE UNIQUE INDEX added ON {index}")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(statement)
        raw.execute("DROP INDEX added")
    assert _dump(path) == before
    with PolicyRegistry(path) as registry:
        assert [event.policy for event in registry.history(TARGET)] == [V1, V2]


@pytest.mark.parametrize("table", sorted(TABLE_INSERTS))
def test_a_writer_of_another_schema_version_is_refused_by_the_barrier(tmp_path, table):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    for version in (0, 2):
        with closing(sqlite3.connect(path, isolation_level=None)) as other:
            other.create_function(WRITER, 0, lambda version=version: version)
            with pytest.raises(sqlite3.IntegrityError, match="schema v1 writer required"):
                other.execute(TABLE_INSERTS[table])
    assert _dump(path) == before
    # The same insert from the current writer passes the barrier.
    with closing(sqlite3.connect(path, isolation_level=None)) as current:
        current.create_function(WRITER, 0, lambda: 1)
        current.execute(TABLE_INSERTS[table])


def test_a_registry_of_another_version_is_refused_before_any_write(tmp_path):
    path = tmp_path / "registry.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE registry_meta (version INTEGER PRIMARY KEY, "
                           "created_at_ns INTEGER NOT NULL)")
        connection.execute("INSERT INTO registry_meta VALUES (2, 1)")
        connection.commit()
    before = _dump(path)
    with pytest.raises(RegistryError, match=r"unsupported policy registry version \[2\]"):
        PolicyRegistry(path)
    with pytest.raises(RegistryError, match=r"unsupported policy registry version \[2\]"):
        PolicyRegistry(path, read_only=True)
    assert _dump(path) == before
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_a_missing_or_altered_barrier_fails_closed_on_reopen(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        raw.execute("DROP TRIGGER policy_registry_v1_activations_delete")
    with pytest.raises(RegistryError, match="writer barrier: policy_registry_v1_activations_delete"):
        PolicyRegistry(path)
    with pytest.raises(RegistryError, match="writer barrier"):
        PolicyRegistry(path, read_only=True)


def _as_writer(path, *statements, lift=()) -> None:
    """Run ``(sql, parameters)`` statements as a client registered as the current writer.

    The insert guards named in *lift* are dropped for the statements and then
    recreated exactly, so the writer barrier still verifies on reopen.
    """
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        raw.create_function(WRITER, 0, lambda: 1)
        for name in lift:
            raw.execute(f"DROP TRIGGER {name}")
        for sql, parameters in statements:
            raw.execute(sql, parameters)
        for name in lift:
            raw.execute(_BARRIER[name])


def _policy_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event(seq, action, policy_json=None, review_id=None, restores_seq=None, *, policy_hash=None):
    if policy_json is not None and policy_hash is None:
        policy_hash = _policy_hash(policy_json)
    return ("INSERT INTO activations VALUES (?, ?, ?, ?, ?, ?, ?, 'stored', 'raw', 1)",
            (TARGET, seq, action, policy_hash, policy_json, review_id, restores_seq))


V1_JSON, V2_JSON, V3_JSON = (f'{{"{TARGET}":"{value}"}}' for value in ("v1", "v2", "v3"))
INVALID_POLICY = f"stored policy of {TARGET!r} seq 3 is invalid"


@pytest.mark.parametrize("statements, lift, message", [
    # Sequence numbers are consecutive.
    ([_event(4, "activate", V3_JSON, "review-1")], ("policy_registry_v1_activations_append",),
     "inconsistent at seq 4"),
    # A rollback carries exactly the policy and review of the entry it restores...
    ([_event(3, "rollback", V2_JSON, None, 1)], (), "inconsistent at seq 3"),
    ([_event(3, "rollback", V1_JSON, "review-1", 1)], (), "inconsistent at seq 3"),
    # ...and restores only a baseline or activate entry.
    ([_event(3, "rollback", V1_JSON, None, 1), _event(4, "rollback", V1_JSON, None, 3)], (),
     "inconsistent at seq 4"),
    # A stored policy is the canonical JSON of an object, under its own hash.
    ([_event(3, "activate", f'{{"{TARGET}": "v3"}}', "review-1")], (), INVALID_POLICY),
    ([_event(3, "activate", V3_JSON, "review-1", policy_hash=_policy_hash(V2_JSON))], (),
     INVALID_POLICY),
    ([_event(3, "activate", '["v3"]', "review-1")], (), INVALID_POLICY),
    ([_event(3, "activate", "v3", "review-1")], (), INVALID_POLICY),
])
def test_reads_refuse_a_stored_history_that_is_inconsistent(tmp_path, statements, lift, message):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    _as_writer(path, *statements, lift=lift)
    before = _dump(path)
    for read_only in (False, True):
        with PolicyRegistry(path, read_only=read_only) as registry:
            with pytest.raises(RegistryError, match=message):
                registry.resolve(TARGET)
            with pytest.raises(RegistryError, match=message):
                registry.history(TARGET)
            if not read_only:
                # A change reads the history first, so it writes nothing.
                with pytest.raises(RegistryError, match=message):
                    registry.retire(TARGET, expected_seq=2 + len(statements), actor="operator",
                                    reason="refused")
    assert _dump(path) == before


@pytest.mark.parametrize("statement, message", [
    (("ALTER TABLE reviews ADD COLUMN extra TEXT", ()), "Invalid policy registry table reviews"),
    (("ALTER TABLE activations RENAME COLUMN reason TO note", ()),
     "Invalid policy registry table activations"),
    (("ALTER TABLE registry_meta ADD COLUMN extra TEXT", ()),
     "Invalid policy registry table registry_meta"),
    # A client without foreign key enforcement can name a review that does not exist.
    (_event(3, "activate", V3_JSON, "review-9"), "Policy registry has invalid references"),
])
def test_opens_refuse_an_altered_table_or_an_invalid_reference(tmp_path, statement, message):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    _as_writer(path, statement)
    before = _dump(path)
    for read_only in (False, True):
        with pytest.raises(RegistryError, match=message):
            PolicyRegistry(path, read_only=read_only)
    assert _dump(path) == before


def test_a_registry_is_never_created_inside_another_database(tmp_path):
    path = tmp_path / "optimize.sqlite3"
    ExperimentLedger(path, durability="normal").close()
    before = _dump(path)
    with pytest.raises(RegistryError, match="holds other tables"):
        PolicyRegistry(path)
    with pytest.raises(RegistryError, match="not a Smythe policy registry"):
        PolicyRegistry(path, read_only=True)
    assert _dump(path) == before


def test_read_only_open_refuses_while_wal_sidecars_exist_and_never_writes(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    with PolicyRegistry(path) as writer:
        assert path.with_name(path.name + "-wal").exists()
        with pytest.raises(RegistryError, match="closed, checkpointed"):
            PolicyRegistry(path, read_only=True)
        writer.retire(TARGET, expected_seq=2, actor="operator", reason="withdrawn")

    files = sorted(item.name for item in tmp_path.iterdir())
    data, mtime = path.read_bytes(), path.stat().st_mtime_ns
    with PolicyRegistry(path, read_only=True) as reader:
        assert [event.action for event in reader.history(TARGET)] == [
            "baseline", "activate", "retire"]
        assert reader.resolve(TARGET) is None
        with pytest.raises(RegistryError, match="read-only"):
            reader.rollback(TARGET, to_seq=2, expected_seq=3, actor="operator", reason="no")
        with pytest.raises(sqlite3.OperationalError):
            reader._connection.execute("INSERT INTO registry_meta VALUES (0, 1)")
    assert sorted(item.name for item in tmp_path.iterdir()) == files
    assert (path.read_bytes(), path.stat().st_mtime_ns) == (data, mtime)

    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        PolicyRegistry(missing, read_only=True)
    assert not missing.exists()


def test_read_only_close_detects_a_writer_that_overlapped_the_read(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    reader = PolicyRegistry(path, read_only=True)
    wal = path.with_name(path.name + "-wal")
    wal.write_bytes(b"overlapping writer")
    with pytest.raises(RegistryError, match="after closing"):
        reader.close()
    wal.unlink()


def test_the_default_registry_lives_beside_other_smythe_state(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with PolicyRegistry() as registry:
        assert registry.path == (tmp_path / ".smythe" / "policy-registry.sqlite3").resolve()
        assert registry.resolve(TARGET) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_new_registry_and_its_sidecars_are_owner_only(tmp_path):
    previous = os.umask(0o022)
    try:
        path = tmp_path / "new-directory" / "registry.sqlite3"
        with PolicyRegistry(path) as registry:
            _review(registry)
            modes = {suffix: stat.S_IMODE(os.stat(f"{path}{suffix}").st_mode)
                     for suffix in ("", "-wal", "-shm")}
        assert modes == {"": 0o600, "-wal": 0o600, "-shm": 0o600}
    finally:
        os.umask(previous)


# ---------------------------------------------------------------------------
# Reviews of Autotune decisions, and their activation
# ---------------------------------------------------------------------------

FIELD = TARGET + ".max_sections"
# Each campaign compares an incumbent section count with one challenger's.
CAMPAIGNS = {"promoted": (4, 6), "rejected": (4, 3), "chained": (6, 7), "sibling": (4, 5)}


def _sections_contract() -> ExperimentContract:
    return ExperimentContract(
        name="report_sections",
        objectives=(MetricObjective("quality", "maximize", primary=True),),
        mutable_fields=(FIELD,),
        mutable_field_rules={FIELD: MutableFieldRule("integer", minimum=1, maximum=12)},
        required_gates=("valid",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=2,
        max_parallel_candidates=2,
        max_trials=14,
        max_wall_seconds=600,
        max_budget_microusd=0,
        per_trial_reservation_microusd=0,
        confidence=0.8,
        min_improvement=0.1,
        base_seed=17,
    )


async def _more_sections_score_higher(context: TrialContext) -> TrialOutcome:
    return TrialOutcome(metrics={"quality": float(context.candidate.policy[FIELD])},
                        gates={"valid": True}, actual_cost_microusd=0)


@pytest.fixture(scope="module")
def campaigns(tmp_path_factory):
    """A closed v4 ledger with one terminal decision per campaign, and their results."""
    path = tmp_path_factory.mktemp("campaigns") / "optimize.sqlite3"
    contract = _sections_contract()
    results = {}
    with ExperimentLedger(path, durability="normal") as ledger:
        for name, (incumbent, challenger) in CAMPAIGNS.items():
            first = Candidate(contract, {FIELD: incumbent}, f"{incumbent} sections")
            second = Candidate(contract, {FIELD: challenger}, f"{challenger} sections",
                               parent=first)
            runner = OptimizationRunner(contract, ledger, _more_sections_score_higher,
                                        evaluator_hash="sha256:" + "e" * 64,
                                        bootstrap_resamples=10, campaign_id=name)
            results[name] = asyncio.run(runner.run(first, (second,)))
        # A sealed campaign that has not decided yet.
        first = Candidate(contract, {FIELD: 4}, "4 sections")
        second = Candidate(contract, {FIELD: 8}, "8 sections", parent=first)
        ledger.create_campaign(contract, first.candidate_id, (first, second),
                               plan_hash="sha256:" + "a" * 64, campaign_id="undecided")
    assert [results[name].promoted for name in CAMPAIGNS] == [True, False, True, True]
    return path, results


@pytest.fixture(scope="module")
def empty_registry(tmp_path_factory):
    path = tmp_path_factory.mktemp("empty") / "registry.sqlite3"
    PolicyRegistry(path).close()
    return path


@pytest.fixture
def ledger_path(campaigns, tmp_path):
    path = tmp_path / "optimize.sqlite3"
    shutil.copyfile(campaigns[0], path)
    return path


@pytest.fixture
def registry_path(empty_registry, tmp_path):
    path = tmp_path / "registry.sqlite3"
    shutil.copyfile(empty_registry, path)
    return path


def _approve(registry, ledger, campaign="promoted", reviewer="reviewer"):
    return registry.record_review(ledger, campaign, reviewer=reviewer, verdict="approve")


def _activate_review(registry, review, *, expected_seq, target=TARGET):
    return registry.activate(target, review.review_id, expected_seq=expected_seq,
                             actor="release-operator", reason="approved by review")


def test_a_promoted_decision_is_reviewed_activated_and_rolled_back(
    campaigns, ledger_path, registry_path, tmp_path,
):
    promoted = campaigns[1]["promoted"]
    ledger_bytes = ledger_path.read_bytes()
    with PolicyRegistry(registry_path) as registry:
        rejection = registry.record_review(ledger_path, "promoted", reviewer="first reviewer",
                                           verdict="reject", note="wants a second task pack")
        approval = registry.record_review(ledger_path, "promoted", reviewer="second reviewer",
                                          verdict="approve", evidence_sha256="ab" * 32,
                                          note="holdout gain is large")
        assert isinstance(approval, PolicyReview)
        assert approval.to_dict() == {
            "review_id": approval.review_id,
            "campaign_id": "promoted",
            "plan_hash": promoted.optimization_plan_hash,
            "holdout_commitment": promoted.holdout_seed_commitment,
            "decision_id": promoted.decision_id,
            "decision_payload_sha256": promoted.decision_id.removeprefix("decision_v1_"),
            "promoted": True,
            "candidate_id": promoted.selected_candidate_id,
            "policy_hash": approval.policy_hash,
            "policy": {FIELD: 6},
            "incumbent_candidate_id": promoted.incumbent_candidate_id,
            "incumbent_policy_hash": approval.incumbent_policy_hash,
            "incumbent_policy": {FIELD: 4},
            "verdict": "approve",
            "reviewer": "second reviewer",
            "evidence_sha256": "ab" * 32,
            "note": "holdout gain is large",
            "created_at_ns": approval.created_at_ns,
        }
        contract = _sections_contract()
        assert approval.policy_hash == Candidate(contract, {FIELD: 6}, "six").policy_hash
        assert approval.incumbent_policy_hash == Candidate(contract, {FIELD: 4}, "four").policy_hash
        payload = {key: value for key, value in approval.to_dict().items()
                   if key not in ("review_id", "created_at_ns")}
        assert approval.review_id == (
            "review_v1_" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest())
        # Recording the same review again returns the stored one.
        assert registry.record_review(ledger_path, "promoted", reviewer="second reviewer",
                                      verdict="approve", evidence_sha256="ab" * 32,
                                      note="holdout gain is large") == approval
        # The rejecting review is kept beside the approval.
        assert registry.reviews("promoted") == (rejection, approval)
        assert registry.review(rejection.review_id) == rejection
        assert registry.reviews("rejected") == ()

        activated = _activate_review(registry, approval, expected_seq=0)
        baseline, event = registry.history(TARGET)
        assert (baseline.seq, baseline.action, baseline.policy, baseline.review_id) == (
            1, "baseline", {FIELD: 4}, None)
        assert baseline.policy_hash == approval.incumbent_policy_hash
        assert event == activated
        assert (event.seq, event.action, event.policy, event.review_id, event.actor) == (
            2, "activate", {FIELD: 6}, approval.review_id, "release-operator")
        assert registry.resolve(TARGET) == ActiveVersion(
            TARGET, 2, approval.policy_hash, {FIELD: 6}, approval.review_id, 2)

        # A campaign run against the policy in effect can replace it.
        chained = _approve(registry, ledger_path, "chained")
        _activate_review(registry, chained, expected_seq=2)
        assert registry.resolve(TARGET).policy == {FIELD: 7}

        # Rollback restores the first campaign's incumbent baseline.
        restored = registry.rollback(TARGET, to_seq=1, expected_seq=3, actor="release-operator",
                                     reason="seven sections overflow the page")
        assert restored.restores_seq == 1
        assert registry.resolve(TARGET) == ActiveVersion(
            TARGET, 4, approval.incumbent_policy_hash, {FIELD: 4}, None, 1)
    # Review, activation and rollback never write the ledger, and rows hold no paths.
    assert ledger_path.read_bytes() == ledger_bytes
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "optimize.sqlite3", "registry.sqlite3"]
    assert not any(str(tmp_path) in line or tmp_path.name in line
                   for line in _dump(registry_path))


def _downgrade_to_v3(path) -> None:
    """Rewrite a closed v4 ledger as schema v3, which had no ownership tables."""
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        for name in ExperimentLedger._barrier_sql():
            raw.execute(f"DROP TRIGGER {name}")
        for table in ("trial_dispatch_owners", "campaign_leases", "campaign_lease_epochs"):
            raw.execute(f"DROP TABLE {table}")
        raw.execute("UPDATE ledger_meta SET version = 3")


@pytest.mark.parametrize("source", ["v4 path", "v4 read-only", "v4 writable", "v3 path"])
def test_reviews_read_v3_and_v4_ledgers_without_writing_them(
    campaigns, ledger_path, registry_path, source,
):
    if source == "v3 path":
        _downgrade_to_v3(ledger_path)
    ledger_bytes = ledger_path.read_bytes()
    with PolicyRegistry(registry_path) as registry:
        if source.endswith("path"):
            review = _approve(registry, ledger_path)
        else:
            with ExperimentLedger(ledger_path, read_only=source == "v4 read-only") as ledger:
                review = _approve(registry, ledger)
        _activate_review(registry, review, expected_seq=0)
        assert registry.resolve(TARGET).policy == {FIELD: 6}
    assert ledger_path.read_bytes() == ledger_bytes
    promoted = campaigns[1]["promoted"]
    assert (review.decision_id, review.plan_hash) == (
        promoted.decision_id, promoted.optimization_plan_hash)
    with ExperimentLedger(ledger_path, read_only=True) as ledger:
        assert ledger.lease_supported is (source != "v3 path")


def test_a_ledger_path_with_an_open_writer_is_not_read(ledger_path, registry_path):
    with PolicyRegistry(registry_path) as registry, ExperimentLedger(ledger_path) as writer:
        before = list(registry._connection.iterdump())
        with pytest.raises(RegistryError, match="cannot be reviewed: read-only inspection "
                                                "requires a closed, checkpointed ledger"):
            _approve(registry, ledger_path)
        assert list(registry._connection.iterdump()) == before
        # The open ledger itself can be reviewed instead.
        assert _approve(registry, writer).decision_id == writer.list_decisions(
            "promoted")[0]["decision_id"]


@pytest.fixture(scope="module")
def reviewed(campaigns, tmp_path_factory):
    """A closed registry where 'promoted' is active and the other campaigns are reviewed."""
    path = tmp_path_factory.mktemp("reviewed") / "registry.sqlite3"
    with PolicyRegistry(path) as registry:
        reviews = {name: _approve(registry, campaigns[0], name)
                   for name in ("promoted", "chained", "sibling")}
        reviews["rejected"] = registry.record_review(campaigns[0], "rejected",
                                                     reviewer="reviewer", verdict="reject")
        _activate_review(registry, reviews["promoted"], expected_seq=0)
    return path, reviews


def _review_row(review, *, review_id=None, **changes):
    """The row of *review* with *changes*, as a client that skips review checks writes it."""
    payload = {key: value for key, value in review.to_dict().items()
               if key not in ("review_id", "created_at_ns")} | changes
    text = canonical_json_bytes(payload).decode("utf-8")
    review_id = review_id or "review_v1_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (review_id, payload["campaign_id"], payload["decision_id"], payload["verdict"],
            payload["reviewer"], text, 1)


def _forged_approval(reviews):
    # Approves the decision that was not promoted, under a matching identifier.
    return _review_row(reviews["rejected"], verdict="approve")


def _edited_review(reviews):
    # Changes a stored payload without changing its identifier.
    return _review_row(reviews["sibling"], review_id="review_v1_" + "1" * 64, note="edited")


def _store(make_row):
    def prepare(registry, ledger, reviews):
        with closing(sqlite3.connect(registry.path, isolation_level=None)) as raw:
            raw.create_function(WRITER, 0, lambda: 1)
            raw.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)", make_row(reviews))
    return prepare


def _retire(registry, ledger, reviews):
    registry.retire(TARGET, expected_seq=2, actor="operator", reason="paused")


def _tamper_ledger(registry, ledger, reviews):
    # Changes a sealed candidate row, so the campaign inventory no longer validates.
    with closing(sqlite3.connect(ledger, isolation_level=None)) as raw:
        raw.create_function("smythe_autotune_writer_version", 0, lambda: 4)
        raw.execute("UPDATE candidates SET candidate_json = replace(candidate_json, "
                    "'7 sections', 'seven sections') WHERE campaign_id = 'chained'")


def _activation(name, *, expected_seq=2, target=TARGET):
    return lambda registry, ledger, reviews: _activate_review(
        registry, reviews[name], expected_seq=expected_seq, target=target)


REFUSALS = {
    "approving a decision that was not promoted": (
        None, RegistryError, "was not promoted; only a promoted decision can be approved",
        lambda registry, ledger, reviews: _approve(registry, ledger, "rejected", "another")),
    "activating a rejecting review": (
        None, RegistryError, "rejects decision", _activation("rejected")),
    "activating a missing review": (
        None, RegistryError, "unknown review 'review_v1_0+'",
        lambda registry, ledger, reviews: registry.activate(
            TARGET, "review_v1_" + "0" * 64, expected_seq=2, actor="operator", reason="none")),
    "activating a stored approval of a decision that was not promoted": (
        _store(_forged_approval), RegistryError, "was not promoted",
        lambda registry, ledger, reviews: registry.activate(
            TARGET, _forged_approval(reviews)[0], expected_seq=2, actor="operator",
            reason="forged")),
    "activating a stored review that does not match its identifier": (
        _store(_edited_review), RegistryError, "stored review 'review_v1_1+' is invalid",
        lambda registry, ledger, reviews: registry.activate(
            TARGET, _edited_review(reviews)[0], expected_seq=2, actor="operator",
            reason="edited")),
    "activating a review of a campaign run against another incumbent": (
        None, RegistryError, "does not have the reviewed campaign's incumbent in effect",
        _activation("sibling")),
    "activating on a retired target": (
        _retire, RegistryError, "does not have the reviewed campaign's incumbent in effect",
        _activation("chained", expected_seq=3)),
    "activating a policy outside the target": (
        None, ValueError, "is not target 'planner.template.repo' or under it",
        _activation("chained", expected_seq=0, target="planner.template.repo")),
    "activating a policy above the target": (
        None, ValueError, "is not target", _activation("chained", target=FIELD + ".limit")),
    "activating from a stale head": (
        None, RegistryConflictError, "at seq 2, not expected_seq 1",
        _activation("chained", expected_seq=1)),
    "activating as a first activation over a head": (
        None, RegistryConflictError, "at seq 2, not expected_seq 0",
        _activation("chained", expected_seq=0)),
    "reviewing a ledger whose inventory no longer validates": (
        _tamper_ledger, RegistryError,
        "campaign 'chained' cannot be reviewed: campaign candidate row differs",
        lambda registry, ledger, reviews: _approve(registry, ledger, "chained", "another")),
    "reviewing a campaign without a terminal decision": (
        None, RegistryError, "campaign 'undecided' has no terminal decision",
        lambda registry, ledger, reviews: _approve(registry, ledger, "undecided")),
    "reviewing a missing campaign": (
        None, RegistryError, "campaign 'missing' cannot be reviewed",
        lambda registry, ledger, reviews: _approve(registry, ledger, "missing")),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_review_and_activation_refuse_without_writing(reviewed, ledger_path, tmp_path, case):
    prepare, error, message, refused = REFUSALS[case]
    path = tmp_path / "registry.sqlite3"
    shutil.copyfile(reviewed[0], path)
    with PolicyRegistry(path) as registry:
        if prepare is not None:
            prepare(registry, ledger_path, reviewed[1])
        before = list(registry._connection.iterdump())
        ledger_bytes = ledger_path.read_bytes()
        with pytest.raises(error, match=message):
            refused(registry, ledger_path, reviewed[1])
        assert list(registry._connection.iterdump()) == before
        in_effect = None if prepare is _retire else {FIELD: 6}
        assert getattr(registry.resolve(TARGET), "policy", None) == in_effect
    assert ledger_path.read_bytes() == ledger_bytes
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "optimize.sqlite3", "registry.sqlite3"]


@pytest.mark.parametrize("arguments, error, message", [
    ({"verdict": "approved"}, ValueError, "verdict must be 'approve' or 'reject'"),
    ({"verdict": None}, ValueError, "verdict must be"),
    ({"reviewer": " "}, ValueError, "reviewer must be a non-empty string"),
    ({"reviewer": "r" * 129}, ValueError, "reviewer must be at most 128"),
    ({"evidence_sha256": "AB" * 32}, ValueError, "evidence_sha256 must be 64 lowercase hex"),
    ({"evidence_sha256": "sha256:" + "ab" * 32}, ValueError, "evidence_sha256"),
    ({"note": "line\nbreak"}, ValueError, "note cannot contain control characters"),
    ({"note": "n" * 1025}, ValueError, "note must be at most 1024"),
    ({"campaign_id": "../promoted"}, ValueError, "campaign_id must be a safe"),
    ({"ledger": "missing.sqlite3"}, FileNotFoundError, "missing.sqlite3"),
    ({"ledger": 7}, TypeError, None),
])
def test_record_review_refuses_malformed_arguments(ledger_path, registry_path, tmp_path,
                                                   arguments, error, message):
    call = {"ledger": ledger_path, "campaign_id": "promoted", "reviewer": "reviewer",
            "verdict": "approve"} | arguments
    if call["ledger"] == "missing.sqlite3":
        call["ledger"] = tmp_path / "missing.sqlite3"
    ledger_bytes = ledger_path.read_bytes()
    with PolicyRegistry(registry_path) as registry:
        before = list(registry._connection.iterdump())
        with pytest.raises(error, match=message):
            registry.record_review(call.pop("ledger"), call.pop("campaign_id"), **call)
        assert list(registry._connection.iterdump()) == before
    assert ledger_path.read_bytes() == ledger_bytes
    assert not (tmp_path / "missing.sqlite3").exists()


def test_a_read_only_registry_records_and_activates_nothing(reviewed, ledger_path, tmp_path):
    path = tmp_path / "registry.sqlite3"
    shutil.copyfile(reviewed[0], path)
    data = path.read_bytes()
    with PolicyRegistry(path, read_only=True) as reader:
        assert reader.reviews("chained") == (reviewed[1]["chained"],)
        with pytest.raises(RegistryError, match="read-only"):
            _approve(reader, ledger_path, "chained", "another")
        with pytest.raises(RegistryError, match="read-only"):
            _activate_review(reader, reviewed[1]["chained"], expected_seq=2)
    assert path.read_bytes() == data


# ---------------------------------------------------------------------------
# Real child processes
# ---------------------------------------------------------------------------

# Child code: open the registry, wait for a go file, then activate over seq 2.
# The first child holds its write transaction after inserting, until released.
RACING_CHILD = r"""
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe.optimize.registry import PolicyRegistry, RegistryConflictError

path, folder, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
target = "planner.template.report"


def wait(flag):
    while not (folder / flag).exists():
        time.sleep(0.01)


insert = PolicyRegistry._insert


def held(cursor, event):
    insert(cursor, event)
    write_barrier(folder / f"{name}-inserted.json", {"seq": event.seq})
    wait("release")


PolicyRegistry._insert = staticmethod(held)
with PolicyRegistry(path) as registry:
    # The second child waits on the first child's write lock; leave a loaded
    # host room to report the conflict rather than "database is locked".
    registry._connection.execute("PRAGMA busy_timeout = 30000")
    write_barrier(folder / f"{name}-open.json", {})
    wait(f"{name}-go")
    write_barrier(folder / f"{name}-appending.json", {})
    try:
        (event,) = registry._append(target, [("activate", {target: name}, "review-1", None)],
                                    expected_seq=2, actor=name, reason="race")
        outcome = {"activated": event.seq}
    except RegistryConflictError as error:
        outcome = {"conflict": str(error)}
write_barrier(folder / f"{name}.json", outcome)
"""


def test_two_processes_with_the_same_expected_seq_activate_exactly_once(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    first = launch_child(RACING_CHILD, path, tmp_path, "first")
    second = launch_child(RACING_CHILD, path, tmp_path, "second")
    try:
        wait_for_barrier(first, tmp_path / "first-open.json", timeout=60)
        wait_for_barrier(second, tmp_path / "second-open.json", timeout=60)
        (tmp_path / "first-go").touch()
        # The first child has inserted seq 3 and holds its transaction open.
        assert wait_for_barrier(first, tmp_path / "first-inserted.json", timeout=60) == {"seq": 3}
        (tmp_path / "second-go").touch()
        wait_for_barrier(second, tmp_path / "second-appending.json", timeout=60)
        time.sleep(0.5)  # The second child waits for the write lock meanwhile.
        (tmp_path / "release").touch()
        outcomes = {name: wait_for_barrier(process, tmp_path / f"{name}.json", timeout=60)
                    for name, process in (("first", first), ("second", second))}
        assert first.wait(timeout=60) == 0 and second.wait(timeout=60) == 0
    finally:
        reaped = [kill_and_reap(first), kill_and_reap(second)]
    assert not any(item.killed for item in reaped), reaped
    assert outcomes == {"first": {"activated": 3},
                        "second": {"conflict": f"target {TARGET!r} is at seq 3, not expected_seq 2"}}
    assert not (tmp_path / "second-inserted.json").exists()
    with PolicyRegistry(path, read_only=True) as reader:
        history = reader.history(TARGET)
    assert [(event.seq, event.action, event.actor) for event in history] == [
        (1, "baseline", "operator"), (2, "activate", "operator"), (3, "activate", "first")]


# Child code: roll back over seq 2, but stop after the insert, before COMMIT.
KILLED_CHILD = r"""
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe.optimize.registry import PolicyRegistry

path, folder = Path(sys.argv[1]), Path(sys.argv[2])
target = "planner.template.report"
insert = PolicyRegistry._insert


def held(cursor, event):
    insert(cursor, event)
    head = cursor.execute("SELECT max(seq) FROM activations WHERE target = ?", (target,)).fetchone()[0]
    write_barrier(folder / "inserted.json", {"seq": event.seq, "head": head})
    time.sleep(600)


PolicyRegistry._insert = staticmethod(held)
with PolicyRegistry(path) as registry:
    registry.rollback(target, to_seq=1, expected_seq=2, actor="operator", reason="killed")
write_barrier(folder / "committed.json", {})
"""


def test_a_writer_killed_before_commit_leaves_the_previous_head(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _activated(path)
    before = _dump(path)
    process = launch_child(KILLED_CHILD, path, tmp_path)
    try:
        # Inside its transaction the child already sees its uncommitted seq 3.
        assert wait_for_barrier(process, tmp_path / "inserted.json", timeout=60) == {
            "seq": 3, "head": 3}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0
    assert not (tmp_path / "committed.json").exists()
    assert _dump(path) == before
    with PolicyRegistry(path) as registry:
        assert registry.resolve(TARGET).seq == 2
        assert registry.resolve(TARGET).policy == V2
        assert len(registry.history(TARGET)) == 2
        # The lost transaction left no lock and no partial event behind.
        event = registry.rollback(TARGET, to_seq=1, expected_seq=2, actor="operator",
                                  reason="after the crash")
        assert (event.seq, event.restores_seq) == (3, 1)
