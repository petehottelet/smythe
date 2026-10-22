"""Workflow journal version upgrades: read-only access, migration and refusal.

Each fixture is built through the public API and then marked as version 1.
Version 2 changes no table, so a version 1 journal differs from a new one only
in its recorded version.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3
from threading import Barrier

import pytest

from fakes import ManualClock, kill_and_reap, launch_child, wait_for_barrier
from smythe import workflow_store
from smythe.pricing import PRICE_VERSION
from smythe.workflow_binding import CallScope, WorkflowBindingError
from smythe.workflow_store import (
    STORE_VERSION, CallKey, SQLiteWorkflowStore, WorkflowCorruptionError, WorkflowLeaseError,
    WorkflowStateError, WorkflowValidationError,
)

CONNECT = sqlite3.connect
LEASE_TTL_S = 30
PROVIDER = {"kind": "openai_responses", "adapter_version": "test-v1", "decoder_version": "test-v1"}
STATES = {
    "completed": ("completed", None), "accepted": ("running", None), "rejected": ("running", None),
    "overrun": ("blocked", "budget_overrun"), "unknown": ("blocked", "unknown_exposure"),
    "interrupted": ("running", None),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def prepare(store, lease, phase, scope):
    request = {"model": "gpt-6-astra", "service_tier": "default", "max_output_tokens": 100,
               "instructions": "test", "reasoning": {"effort": "medium"}, "store": False,
               "truncation": "disabled", "input": [{"role": "user", "content": "source for " + scope}]}
    return store.prepare_call(lease, CallKey(phase, scope), request_json=canonical(request),
                              provider=PROVIDER, price_version=PRICE_VERSION)


def envelope(call, raw, operation="response", **metadata):
    return {"body": canonical(raw).encode(), "request_sha256": call["request_sha256"],
            "operation": operation, "status_code": 200, "request_id": "req_test", **metadata}


def response(outputs=2):
    return {"id": "resp_test", "model": "gpt-6-astra", "service_tier": "default", "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": outputs,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "saved output"}]}]}


def reserve(store, lease, call):
    evidence_id = store.append_quote_evidence(
        lease, call["call_id"], envelope(call, {"input_tokens": 100}, "input_tokens"))
    quote = store.accept_quote(lease, call["call_id"], evidence_id)
    store.reserve_call(lease, call["call_id"], quote["quote_id"])


def dispatch(store, lease, call):
    reserve(store, lease, call)
    return store.claim_dispatch(lease, call["call_id"], call["request_sha256"])


def settle(store, lease, call, raw=None, **metadata):
    permit = dispatch(store, lease, call)
    evidence_id = store.append_response(permit, envelope(call, raw or response(), **metadata))
    return store.settle_call(lease, call["call_id"], evidence_id)


def start(store, run_id):
    store.create_run({"goal": "task " + run_id}, {"recipe": run_id}, 10**12, run_id=run_id)
    return store.acquire_lease(run_id, "owner-" + run_id, ttl_s=LEASE_TTL_S)


def build_journal(path, clock):
    """Write runs in each terminal and blocking state; every lease then expires."""
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        # An accepted call applied by a checkpoint, then a completed run.
        lease = start(store, "completed")
        call = settle(store, lease, prepare(store, lease, "execution", "node/a"))
        store.accept_result(lease, call["call_id"], {"text": "accepted output"}, "test-v1")
        planned = store.begin_operation(lease, "plan", "planning", {"goal": "task completed"})
        store.complete_operation(lease, "plan", {"graph": {"nodes": ["a"]}}, [call["call_id"]])
        store.save_checkpoint(lease, 0, {"graph": "planned"}, consumed_operation_ids=[planned["operation_id"]])
        store.save_checkpoint(lease, 1, {"status": "completed"})
        store.release_lease(lease)
        # Decoded output that no checkpoint has consumed yet.
        lease = start(store, "accepted")
        call = settle(store, lease, prepare(store, lease, "execution", "node/a"))
        store.accept_result(lease, call["call_id"], {"text": "pending output"}, "test-v1")
        store.release_lease(lease)
        # A paid result the caller rejected, and a zero-cost provider refusal.
        lease = start(store, "rejected")
        call = settle(store, lease, prepare(store, lease, "verification", "node/judge"))
        store.reject_result(lease, call["call_id"], "verdict failed")
        refusal = {"error": {"message": "Rate limit reached", "type": "requests", "param": None,
                             "code": "rate_limit_exceeded"}}
        settle(store, lease, prepare(store, lease, "execution", "node/b"), refusal,
               status_code=429, transport_error="RateLimitError")
        store.release_lease(lease)
        # A charge above its quote closes admission.
        lease = start(store, "overrun")
        settle(store, lease, prepare(store, lease, "execution", "node/a"), response(outputs=1000))
        store.release_lease(lease)
        # A dispatched call whose response was lost holds unknown exposure.
        lease = start(store, "unknown")
        call = prepare(store, lease, "execution", "node/a")
        dispatch(store, lease, call)
        store.mark_unknown(lease, call["call_id"], "transport_lost")
        store.release_lease(lease)
        # A reservation whose owner stopped without releasing its lease.
        lease = start(store, "interrupted")
        reserve(store, lease, prepare(store, lease, "execution", "node/a"))
    clock.advance(LEASE_TTL_S + 1)


def copy(source, target):
    with closing(CONNECT(source)) as db, closing(CONNECT(target)) as copied:
        db.backup(copied)
    return target


def mark_version_1(path):
    with closing(CONNECT(path)) as db, db:
        db.execute("UPDATE workflow_meta SET version=1")


@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    """Version 1 journals, built once: every lease expired, or one still live."""
    folder, clock = tmp_path_factory.mktemp("journals"), ManualClock()
    build_journal(folder / "built.db", clock)
    expired, live = copy(folder / "built.db", folder / "expired.db"), copy(folder / "built.db", folder / "live.db")
    with SQLiteWorkflowStore(live, clock_ns=clock) as store:
        # A worker from an earlier release holds a run.
        store.acquire_lease("accepted", "earlier-release-worker", ttl_s=LEASE_TTL_S)
    mark_version_1(expired)
    mark_version_1(live)
    return {"expired": expired, "live": live}, clock.now_ns


def v1_journal(templates, tmp_path, kind="expired"):
    paths, now_ns = templates
    return copy(paths[kind], tmp_path / "workflow.db"), ManualClock(now_ns)


def dump(path):
    with closing(CONNECT(path)) as db:
        return list(db.iterdump())


def migrated(lines):
    """The dump of a migrated journal: only the recorded version changes."""
    return [line.replace("'smythe.workflow',1,", f"'smythe.workflow',{STORE_VERSION},")
            if line.startswith('INSERT INTO "workflow_meta"') else line for line in lines]


def recorded_version(path):
    with closing(CONNECT(path)) as db:
        return db.execute("SELECT version FROM workflow_meta").fetchall()


def assert_integrity(path):
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def snapshot(store):
    """Every record the public read API exposes, including raw evidence."""
    runs = store.list_runs()
    detail = {}
    for run in runs:
        run_id = run["run_id"]
        inspected = store.inspect_run(run_id)
        detail[run_id] = {
            "run": store.load_run(run_id), "inspect": inspected, "checkpoint": store.get_checkpoint(run_id),
            "sessions": store.lease_sessions(run_id), "operation": store.load_operation(run_id, "plan"),
            "replays": [store.load_replay(call["call_id"]) for call in inspected["calls"]],
        }
    return {"store_id": store.store_id, "runs": runs, "detail": detail}


def assert_fixture_states(state):
    runs = {run["run_id"]: run for run in state["runs"]}
    assert {run_id: (run["status"], run["blocked_reason"]) for run_id, run in runs.items()} == STATES
    calls = {run_id: [(call["state"], call["billing_state"], call["result_state"])
                      for call in detail["inspect"]["calls"]]
             for run_id, detail in state["detail"].items()}
    assert calls == {
        "completed": [("settled", "known", "applied")],
        "accepted": [("settled", "known", "accepted")],
        "rejected": [("settled", "known", "rejected"), ("settled", "known", "rejected")],
        "overrun": [("settled", "known", "pending")],
        "unknown": [("unknown", "unknown", "pending")],
        "interrupted": [("reserved", "reserved", "pending")],
    }
    ceiling = state["detail"]["unknown"]["inspect"]["calls"][0]["ceiling_nanousd"]
    assert ceiling > 0
    assert (runs["unknown"]["unknown_nanousd"], runs["unknown"]["reserved_nanousd"]) == (ceiling, 0)
    assert (runs["interrupted"]["reserved_nanousd"], runs["interrupted"]["unknown_nanousd"]) == (ceiling, 0)
    assert runs["overrun"]["confirmed_nanousd"] > ceiling
    assert runs["rejected"]["confirmed_nanousd"] == runs["completed"]["confirmed_nanousd"] > 0
    assert state["detail"]["completed"]["checkpoint"]["revision"] == 2
    assert state["detail"]["completed"]["operation"]["state"] == "applied"
    assert state["detail"]["interrupted"]["sessions"]["sessions"][0]["interrupted"] is True
    for detail in state["detail"].values():
        assert detail["inspect"]["ok"] is True


def test_writable_open_migrates_and_keeps_every_record_identity_and_balance(templates, tmp_path):
    path, clock = v1_journal(templates, tmp_path)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        before = snapshot(reader)
    assert_fixture_states(before)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == STORE_VERSION == 2
        assert snapshot(store) == before
    assert recorded_version(path) == [(2,)]
    # Every table except the version marker is unchanged, row for row.
    assert dump(path) == migrated(before_dump)
    assert_integrity(path)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert snapshot(store) == before
        # Saved runs continue on the migrated journal.
        lease = store.acquire_lease("accepted", "new-owner")
        plan = store.recover(lease)
        assert plan["apply"] == [before["detail"]["accepted"]["inspect"]["calls"][0]["call_id"]]
        store.save_checkpoint(lease, 0, {"status": "completed"}, consumed_call_ids=plan["apply"])
        assert store.load_run("accepted")["status"] == "completed"
        lease = store.acquire_lease("interrupted", "new-owner")
        assert lease.epoch == 2
        assert len(store.recover(lease)["undispatched"]) == 1


def test_read_only_open_reads_a_v1_journal_without_migrating(templates, tmp_path, monkeypatch):
    path, clock = v1_journal(templates, tmp_path, "live")
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    before = path.read_bytes()
    statements = []

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", connect)
    # Inspection works while a worker from an earlier release holds a lease.
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        state = snapshot(reader)
        assert reader.audit("accepted")["ok"] is True
        with pytest.raises(WorkflowStateError):
            reader.create_run(None, {}, None, run_id="new")
    monkeypatch.setattr(sqlite3, "connect", CONNECT)
    assert {run["run_id"] for run in state["runs"]} == set(STATES)
    compact = ["".join(sql.upper().split()) for sql in statements]
    assert "BEGINIMMEDIATE" not in compact
    assert not any(sql.startswith(("PRAGMAJOURNAL_MODE=", "CREATE", "INSERT", "UPDATE", "ALTER"))
                   for sql in compact)
    assert path.read_bytes() == before
    assert recorded_version(path) == [(1,)]


def test_live_lease_from_an_earlier_release_refuses_migration(templates, tmp_path):
    path, clock = v1_journal(templates, tmp_path, "live")
    before = dump(path)
    with pytest.raises(WorkflowLeaseError, match="live lease"):
        SQLiteWorkflowStore(path, clock_ns=clock)
    assert dump(path) == before
    assert recorded_version(path) == [(1,)]
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        assert reader.lease_sessions("accepted")["sessions"][-1]["ended_ns"] == clock.now_ns
    # Once that worker stops and its lease expires, the upgrade proceeds.
    clock.advance(LEASE_TTL_S + 1)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == 2
    assert dump(path) == migrated(before)


@pytest.mark.parametrize("failure", ["interrupt", "sql-error"])
def test_interrupted_migration_leaves_a_usable_v1_journal(templates, tmp_path, monkeypatch, failure):
    path, clock = v1_journal(templates, tmp_path)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        before = snapshot(reader)
    opened = []
    interrupt = KeyboardInterrupt("stopped before the migration committed")

    def deny_version_update(action, table, *_):
        if action == sqlite3.SQLITE_UPDATE and table == "workflow_meta":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        opened.append(db)
        if failure == "sql-error":
            db.set_authorizer(deny_version_update)
        return db

    original = SQLiteWorkflowStore._migrate_schema

    def interrupted(self, *args):
        original(self, *args)
        raise interrupt

    monkeypatch.setattr(sqlite3, "connect", connect)
    if failure == "interrupt":
        monkeypatch.setattr(SQLiteWorkflowStore, "_migrate_schema", interrupted)
        with pytest.raises(KeyboardInterrupt) as caught:
            SQLiteWorkflowStore(path, clock_ns=clock)
        assert caught.value is interrupt
    else:
        with pytest.raises(WorkflowCorruptionError) as caught:
            SQLiteWorkflowStore(path, clock_ns=clock)
        assert isinstance(caught.value.__cause__, sqlite3.DatabaseError)
    monkeypatch.undo()
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")
    assert dump(path) == before_dump
    assert recorded_version(path) == [(1,)]
    assert_integrity(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        assert snapshot(reader) == before
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == 2
        assert snapshot(store) == before
    assert dump(path) == migrated(before_dump)


def test_concurrent_writable_openers_migrate_once_and_share_one_identity(templates, tmp_path, monkeypatch):
    path, clock = v1_journal(templates, tmp_path)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        before = snapshot(reader)
    preflight, opened_together = Barrier(2), Barrier(2)
    failures, migrations = [], []
    original = SQLiteWorkflowStore._migrate_schema

    def counted(self, *args):
        migrations.append(self)
        return original(self, *args)

    def trace(sql):
        # Both openers validate the version 1 schema before either takes the write lock.
        if "".join(sql.upper().split()) == "PRAGMASYNCHRONOUS=FULL":
            try:
                preflight.wait(timeout=15)
            except BaseException as exc:
                failures.append(exc)

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs)
        db.set_trace_callback(trace)
        return db

    monkeypatch.setattr(SQLiteWorkflowStore, "_migrate_schema", counted)
    monkeypatch.setattr(sqlite3, "connect", connect)

    def open_writer():
        with SQLiteWorkflowStore(path, clock_ns=clock) as store:
            result = store.store_version, store.store_id
            opened_together.wait(timeout=15)
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(open_writer) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    monkeypatch.undo()
    assert not failures
    assert len(migrations) == 1
    assert results == [(2, before["store_id"])] * 2
    assert recorded_version(path) == [(2,)]
    assert dump(path) == migrated(before_dump)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert snapshot(reader) == before
    assert_integrity(path)


MIGRATING_CHILD = r"""
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe.workflow_store import SQLiteWorkflowStore

path, folder = Path(sys.argv[1]), Path(sys.argv[2])
migrate = SQLiteWorkflowStore._migrate_schema


def held(self, db):
    migrate(self, db)
    version = db.execute("SELECT version FROM workflow_meta").fetchone()[0]
    write_barrier(folder / "migrated.json", {"version": version})
    time.sleep(600)


SQLiteWorkflowStore._migrate_schema = held
SQLiteWorkflowStore(path).close()
write_barrier(folder / "committed.json", {})
"""


def test_process_killed_during_migration_leaves_the_v1_journal_intact(templates, tmp_path):
    path, clock = v1_journal(templates, tmp_path)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        before = snapshot(reader)
    # The child uses the real clock, long after the fixture's leases expired.
    process = launch_child(MIGRATING_CHILD, path, tmp_path)
    try:
        assert wait_for_barrier(process, tmp_path / "migrated.json") == {"version": 2}
        # Readers keep seeing version 1 until the migration commits.
        with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
            assert reader.store_version == 1
            assert snapshot(reader) == before
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0
    assert not (tmp_path / "committed.json").exists()
    assert recorded_version(path) == [(1,)]
    assert dump(path) == before_dump
    assert_integrity(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        assert snapshot(reader) == before
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == 2
        assert snapshot(store) == before
    assert dump(path) == migrated(before_dump)


@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read-only"])
@pytest.mark.parametrize("version", [0, STORE_VERSION + 1, 1.5, "v1"])
def test_unsupported_journal_versions_are_refused_without_changes(templates, tmp_path, read_only, version):
    path, clock = v1_journal(templates, tmp_path)
    with closing(CONNECT(path)) as db, db:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("UPDATE workflow_meta SET version=?", (version,))
    before = path.read_bytes()
    with pytest.raises(WorkflowCorruptionError, match="Unsupported workflow journal version"):
        SQLiteWorkflowStore(path, read_only=read_only, clock_ns=clock)
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, STORE_VERSION + 1])
def test_open_writer_refuses_writes_once_the_journal_version_changes(tmp_path, version):
    path = tmp_path / "workflow.db"
    with SQLiteWorkflowStore(path) as store:
        store.create_run(None, {}, None, run_id="saved")
        # Another release migrates the journal, or an older format is restored.
        with closing(CONNECT(path)) as db, db:
            db.execute("UPDATE workflow_meta SET version=?", (version,))
        before = dump(path)
        with pytest.raises(WorkflowCorruptionError, match="version changed"):
            store.create_run(None, {}, None, run_id="later")
        with pytest.raises(WorkflowCorruptionError, match="version changed"):
            store.acquire_lease("saved", "owner")
        assert store.load_run("saved")["status"] == "running"
    assert dump(path) == before


def test_tool_phase_is_shared_by_call_keys_and_call_scopes():
    phases = {"routing", "planning", "execution", "verification", "supervision", "synthesis", "tool"}
    assert workflow_store._PHASES == phases
    for phase in phases | {"tools", "Tool", ""}:
        try:
            CallKey(phase, "node/fetch")
        except WorkflowValidationError:
            key = False
        else:
            key = True
        try:
            CallScope("node/fetch", phase, {"node_id": "fetch"})
        except WorkflowBindingError:
            scope = False
        else:
            scope = True
        assert key == scope == (phase in phases), phase
