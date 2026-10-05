"""Workflow journal version upgrades and the external action tables.

Each fixture is built through the public API and then made a version 1
journal: version 2's external action tables are dropped and the recorded
version set to 1. Version 1 has every other table of version 2, so a version 1
journal differs from a new one only in those tables and its recorded version.
Action rows are written with SQL in the shape the action tables define, since
nothing in the library writes them yet.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
import re
import sqlite3
import stat
import subprocess
from threading import Barrier

import pytest

from fakes import (
    Crash, ManualClock, child_executable, crash_when, kill_and_reap, launch_child, offline_env,
    wait_for_barrier,
)
from smythe import (
    AcceptanceDecision, AcceptancePolicy, OfflineProvider, SimpleArchitect, Swarm, Task, TextValidator,
    workflow_store,
)
from smythe.actions import ActionIntent, ActionTarget, ConnectorProfile, ContentRef, action_key
from smythe.pricing import PRICE_VERSION
from smythe.workflow import current_decision
from smythe.workflow_binding import CallScope, WorkflowBindingError
from smythe.workflow_store import (
    STORE_VERSION, CallKey, SQLiteWorkflowStore, WorkflowCorruptionError, WorkflowLeaseError,
    WorkflowStateError, WorkflowValidationError, _canonical, _sha,
)

CONNECT = sqlite3.connect
LEASE_TTL_S = 30
PROVIDER = {"kind": "openai_responses", "adapter_version": "test-v1", "decoder_version": "test-v1"}
# The fixture's runs by id, besides the strict run whose delivery was accepted.
STATES = {
    "completed": ("completed", None), "accepted": ("running", None), "rejected": ("running", None),
    "overrun": ("blocked", "budget_overrun"), "unknown": ("blocked", "unknown_exposure"),
    "interrupted": ("running", None),
}
ACTION_TABLES = ("workflow_actions", "workflow_action_attempts", "workflow_action_approvals",
                 "workflow_action_records")


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


def add_delivered_run(path, clock):
    """Add a completed run whose strict delivery decision accepted its output."""
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(), run_store=store,
              max_budget_usd=1, artifact_dir=None,
              acceptance=AcceptancePolicy(mode="strict", final_validators=[TextValidator()]),
              ).execute(Task("Write the brief"))
    clock.advance(LEASE_TTL_S + 1)


def copy(source, target):
    with closing(CONNECT(source)) as db, closing(CONNECT(target)) as copied:
        db.backup(copied)
    return target


def mark_version_1(path):
    """Give a journal version 1's schema: no action tables, and version 1 recorded."""
    with closing(CONNECT(path)) as db, db:
        for table in ACTION_TABLES:
            db.execute(f"DROP TABLE {table}")
        db.execute("UPDATE workflow_meta SET version=1")


@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    """Version 1 journals, built once: every lease expired, or one still live."""
    folder, clock = tmp_path_factory.mktemp("journals"), ManualClock()
    build_journal(folder / "built.db", clock)
    add_delivered_run(folder / "built.db", clock)
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
    """A version 1 dump as its upgrade keeps it: only the recorded version changes."""
    return [line.replace("'smythe.workflow',1,", f"'smythe.workflow',{STORE_VERSION},")
            if line.startswith('INSERT INTO "workflow_meta"') else line for line in lines]


def without_actions(lines):
    """A dump without the action tables' schema and rows: version 1's tables."""
    return [line for line in lines
            if not re.match(r'(CREATE (TABLE|INDEX|TRIGGER) |INSERT INTO ")workflow_action', line)]


def schema(path):
    with closing(CONNECT(path)) as db:
        return sorted(db.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall(),
                      key=lambda row: (row[0], row[1]))


def assert_upgraded(path, before_dump):
    """Every version 1 row is kept, and the empty action tables match a new journal's."""
    after = dump(path)
    assert without_actions(after) == migrated(before_dump)
    assert [line for line in after if line.startswith('INSERT INTO "workflow_action')] == []
    fresh = path.parent / "fresh-schema.db"
    SQLiteWorkflowStore(fresh).close()
    assert schema(path) == schema(fresh)
    assert recorded_version(path) == [(STORE_VERSION,)]
    assert_integrity(path)


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
            "decision": current_decision(store, run_id),
            # Exactly what inspection returns, in its key order.
            "inspected_bytes": json.dumps(inspected).encode(),
        }
    return {"store_id": store.store_id, "runs": runs, "detail": detail}


def delivered_run(state):
    """The id of the fixture's run with a delivery decision."""
    decided = [run_id for run_id, detail in state["detail"].items() if detail["decision"]]
    assert len(decided) == 1
    return decided[0]


def assert_fixture_states(state):
    runs = {run["run_id"]: run for run in state["runs"]}
    delivered = delivered_run(state)
    assert state["detail"][delivered]["decision"]["record"].decision is AcceptanceDecision.ACCEPTED
    assert (runs[delivered]["status"], runs[delivered]["blocked_reason"]) == ("completed", None)
    assert {run_id: (run["status"], run["blocked_reason"])
            for run_id, run in runs.items() if run_id != delivered} == STATES
    calls = {run_id: [(call["state"], call["billing_state"], call["result_state"])
                      for call in detail["inspect"]["calls"]]
             for run_id, detail in state["detail"].items() if run_id != delivered}
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
    os.chmod(path, 0o600)
    mode = stat.S_IMODE(path.stat().st_mode)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        assert reader.store_version == 1
        before = snapshot(reader)
    assert_fixture_states(before)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == STORE_VERSION == 2
        # Inspection of runs without actions returns the same bytes as before.
        assert snapshot(store) == before
    for detail in before["detail"].values():
        assert "actions" not in detail["inspect"]
    # Every version 1 table is unchanged, row for row, apart from the version
    # marker, and the action tables are created empty.
    assert_upgraded(path, before_dump)
    assert stat.S_IMODE(path.stat().st_mode) == mode
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
    assert {run["run_id"] for run in state["runs"]} == set(STATES) | {delivered_run(state)}
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
    assert_upgraded(path, before)


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
    assert_upgraded(path, before_dump)


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
    assert_upgraded(path, before_dump)
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
    assert_upgraded(path, before_dump)


def test_a_crash_after_the_action_tables_are_created_rolls_the_upgrade_back(templates, tmp_path, monkeypatch):
    path, clock = v1_journal(templates, tmp_path)
    before_dump = dump(path)
    with SQLiteWorkflowStore(path, read_only=True, clock_ns=clock) as reader:
        before = snapshot(reader)
    created = []

    def final_check(self, db, *, allow_empty):
        # The upgrade's last step, inside its transaction, once its tables exist.
        if allow_empty:
            return False
        created.extend(row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'workflow_action%'"))
        return True

    point = crash_when(monkeypatch, SQLiteWorkflowStore, "_validate_schema_db", final_check)
    with pytest.raises(Crash):
        SQLiteWorkflowStore(path, clock_ns=clock)
    point.restore()
    assert point.fired and sorted(created) == sorted(ACTION_TABLES)
    assert dump(path) == before_dump
    assert recorded_version(path) == [(1,)]
    assert_integrity(path)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.store_version == 2
        assert snapshot(store) == before
    assert_upgraded(path, before_dump)


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


@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read-only"])
@pytest.mark.parametrize("change", ["version-2-missing-a-table", "version-1-with-an-action-table",
                                    "version-2-with-another-table"])
def test_journal_tables_must_be_exactly_those_of_its_version(tmp_path, read_only, change):
    path = tmp_path / "workflow.db"
    with SQLiteWorkflowStore(path) as store:
        store.create_run(None, {}, None, run_id="saved")
    if change == "version-1-with-an-action-table":
        mark_version_1(path)
    with closing(CONNECT(path)) as db, db:
        db.execute("PRAGMA journal_mode=DELETE")
        if change == "version-2-missing-a-table":
            db.execute("DROP TABLE workflow_action_records")
        elif change == "version-1-with-an-action-table":
            db.execute("CREATE TABLE workflow_actions(action_id TEXT)")
        else:
            db.execute("CREATE TABLE unrelated(value TEXT)")
    before = path.read_bytes()
    with pytest.raises(WorkflowCorruptionError, match="tables do not match its version"):
        SQLiteWorkflowStore(path, read_only=read_only)
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


# ---------------------------------------------------------------------------
# External action tables
# ---------------------------------------------------------------------------

NOW = ManualClock().now_ns
HOUR_NS = 3600 * 10**9
PROFILE = ConnectorProfile("fixture-posts", "1", ["publish", "delete"], write_ceiling_nanousd=2_000_000,
                           read_ceiling_nanousd=10_000, max_attempts=2, readback_reads=2)
PRINCIPAL = {"issuer": "host", "subject": "editor", "tenant": "acme"}
RECEIPT = {"key": "receipt-key", "object_version": 1, "cost_nanousd": 2_000_000, "body": "SECRET-RECEIPT"}
READBACK = {"content_sha256": "0" * 64, "body": "SECRET-PAYLOAD"}
NOTICE = {"notice": "SECRET-NOTICE"}


def intent(operation="publish", title="SECRET-PARAMETER"):
    return ActionIntent(operation, ActionTarget("acme", "posts"), parameters={"title": title},
                        content=ContentRef.for_bytes(b"accepted document", "text/markdown"),
                        connector=PROFILE.name, connector_version=PROFILE.version)


def insert(db, table, **row):
    db.execute(f"INSERT INTO {table}({','.join(row)}) VALUES ({','.join('?' * len(row))})",
               tuple(row.values()))


def add_action(db, store_id, action_id, slot, value, state, balances, compensates=None):
    insert(db, "workflow_actions", action_id=action_id, run_id="acting", slot=slot, sequence=0,
           action_key=action_key(store_id, "acting", slot, 0), intent_json=_canonical(value.to_dict()),
           intent_sha=value.digest(), profile_json=_canonical(PROFILE.to_dict()), state=state,
           ceiling="4020000", confirmed=balances[0], reserved=balances[1], unknown=balances[2],
           compensates_action_id=compensates, created=NOW, updated=NOW)


def add_approval(db, action_id, approval_id, value, **row):
    assertion = "SECRET-ASSERTION-" + approval_id
    insert(db, "workflow_action_approvals", approval_id=approval_id, action_id=action_id,
           intent_sha=value.digest(), verdict="approved", principal_json=_canonical(PRINCIPAL),
           verifier_json=_canonical({"identity": "fixture-hmac v1"}), assertion=assertion,
           assertion_sha=_sha(assertion), ceiling="4020000", expires_at_ns=NOW + HOUR_NS, created=NOW, **row)


def add_attempt(db, action_id, attempt_id, number, approval_id, state, *, epoch=None, receipt=None,
                cost=None, reserved="0", unknown="0"):
    receipt_json = None if receipt is None else _canonical(receipt)
    insert(db, "workflow_action_attempts", attempt_id=attempt_id, action_id=action_id, number=number,
           approval_id=approval_id, state=state,
           dispatch_token_sha=None if epoch is None else _sha("token-" + attempt_id), dispatch_epoch=epoch,
           receipt_json=receipt_json, receipt_sha=None if receipt is None else _sha(receipt_json),
           cost=cost, reserved=reserved, unknown=unknown, error="SECRET-ERROR from the service",
           created=NOW, dispatched=None if epoch is None else NOW, updated=NOW)


def add_record(db, action_id, record_id, kind, payload, *, trusted, cost=None):
    insert(db, "workflow_action_records", record_id=record_id, action_id=action_id, kind=kind,
           payload_json=_canonical(payload), payload_sha=_sha(_canonical(payload)), trusted=int(trusted),
           cost=cost, created=NOW)


@pytest.fixture(scope="module")
def action_template(tmp_path_factory):
    """A version 2 journal whose run ``acting`` has three external actions; ``quiet`` has none.

    ``published`` was refused once, then acknowledged and read back;
    ``pending`` has an outcome that may have taken effect; and ``undo``,
    which compensates ``published``, holds a reservation for its first attempt.
    """
    path, clock, before = tmp_path_factory.mktemp("actions") / "workflow.db", ManualClock(), {}
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        for run_id in ("acting", "quiet"):
            lease = start(store, run_id)
            call = settle(store, lease, prepare(store, lease, "execution", "node/a"))
            store.accept_result(lease, call["call_id"], {"text": "accepted output"}, "test-v1")
            store.release_lease(lease)
            before[run_id] = json.dumps(store.inspect_run(run_id)).encode()
        audit, store_id = store.audit("acting"), store.store_id
    published, pending, undo = intent(), intent(title="announcement"), intent("delete")
    with closing(CONNECT(path)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        add_action(db, store_id, "published", "publish", published, "observed", ("2010000", "0", "0"))
        add_approval(db, "published", "approval-1", published)
        add_attempt(db, "published", "attempt-1", 1, "approval-1", "refused", epoch=1, cost="0")
        add_attempt(db, "published", "attempt-2", 2, "approval-1", "acknowledged", epoch=1, receipt=RECEIPT,
                    cost="2000000")
        db.execute("UPDATE workflow_action_approvals SET consumed_by_attempt='attempt-2' "
                   "WHERE approval_id='approval-1'")
        add_record(db, "published", "readback-1", "readback", READBACK, trusted=True, cost="10000")
        add_record(db, "published", "notice-1", "notice", NOTICE, trusted=False)
        add_action(db, store_id, "pending", "announce", pending, "outcome_unknown", ("0", "0", "2000000"))
        add_approval(db, "pending", "approval-2", pending)
        add_approval(db, "pending", "approval-2b", pending, revoked_at_ns=NOW, revoked_reason="SECRET-REASON")
        add_attempt(db, "pending", "pending-1", 1, "approval-2", "outcome_unknown", epoch=2, unknown="2000000")
        add_action(db, store_id, "undo", "undo", undo, "not_dispatched", ("0", "2000000", "0"), "published")
        add_approval(db, "undo", "approval-3", undo)
        add_attempt(db, "undo", "undo-1", 1, "approval-3", "prepared", reserved="2000000")
    return path, store_id, before, audit


def action_journal(action_template, tmp_path):
    path, *rest = action_template
    return copy(path, tmp_path / "workflow.db"), *rest


def test_inspection_adds_a_safe_actions_block_only_for_runs_with_actions(action_template, tmp_path):
    path, store_id, before, audit = action_journal(action_template, tmp_path)
    with SQLiteWorkflowStore(path, read_only=True) as reader:
        # Action money never enters a run's balances, which equal its call ledger.
        assert reader.audit("acting") == audit
        inspected, quiet = reader.inspect_run("acting"), reader.inspect_run("quiet")
        assert reader.list_runs()[0]["confirmed_nanousd"] == audit["confirmed_nanousd"]
    assert json.dumps(quiet).encode() == before["quiet"]
    actions = inspected.pop("actions")
    # The block is added last; everything else is what the run inspected to before.
    assert json.dumps(inspected).encode() == before["acting"]
    assert "SECRET" not in json.dumps(actions)
    totals = {key: actions[key] for key in actions if key != "items"}
    assert totals == {"accounting_scope": "external_actions", "unit": "nanoUSD",
                      "confirmed_nanousd": 2_010_000, "reserved_nanousd": 2_000_000, "unknown_nanousd": 2_000_000}
    items = {item["action_id"]: item for item in actions["items"]}
    assert list(items) == ["pending", "published", "undo"]  # by slot, then sequence
    value = intent()
    assert items["published"] == {
        "action_id": "published", "slot": "publish", "sequence": 0,
        "action_key": action_key(store_id, "acting", "publish", 0), "intent_sha256": value.digest(),
        "operation": "publish", "target": {"tenant": "acme", "collection": "posts", "object_id": None},
        "connector": "fixture-posts", "connector_version": "1", "state": "observed",
        "compensates_action_id": None, "ceiling_nanousd": 4_020_000,
        "confirmed_nanousd": 2_010_000, "reserved_nanousd": 0, "unknown_nanousd": 0,
        "attempts": [
            {"attempt_id": "attempt-1", "number": 1, "approval_id": "approval-1", "state": "refused",
             "dispatch_epoch": 1, "receipt_sha256": None, "cost_nanousd": 0, "reserved_nanousd": 0,
             "unknown_nanousd": 0},
            {"attempt_id": "attempt-2", "number": 2, "approval_id": "approval-1", "state": "acknowledged",
             "dispatch_epoch": 1, "receipt_sha256": _sha(_canonical(RECEIPT)), "cost_nanousd": 2_000_000,
             "reserved_nanousd": 0, "unknown_nanousd": 0},
        ],
        "approvals": [
            {"approval_id": "approval-1", "verdict": "approved", "intent_sha256": value.digest(),
             "principal": PRINCIPAL, "verifier": {"identity": "fixture-hmac v1"},
             "ceiling_nanousd": 4_020_000, "expires_at_ns": NOW + HOUR_NS, "consumed_by_attempt": "attempt-2",
             "revoked_at_ns": None, "duplicate_risk": False},
        ],
        "records": [
            {"record_id": "readback-1", "kind": "readback", "payload_sha256": _sha(_canonical(READBACK)),
             "trusted": True, "cost_nanousd": 10_000},
            {"record_id": "notice-1", "kind": "notice", "payload_sha256": _sha(_canonical(NOTICE)),
             "trusted": False, "cost_nanousd": None},
        ],
    }
    pending, undo = items["pending"], items["undo"]
    assert (pending["state"], pending["unknown_nanousd"], pending["attempts"][0]["state"]) == (
        "outcome_unknown", 2_000_000, "outcome_unknown")
    assert [(approval["approval_id"], approval["revoked_at_ns"]) for approval in pending["approvals"]] == [
        ("approval-2", None), ("approval-2b", NOW)]
    assert (undo["compensates_action_id"], undo["operation"], undo["reserved_nanousd"]) == (
        "published", "delete", 2_000_000)
    assert undo["attempts"][0]["dispatch_epoch"] is None


def test_runs_without_actions_never_load_the_action_modules(tmp_path):
    code = (
        "import sys\n"
        "from smythe import SQLiteWorkflowStore\n"
        "with SQLiteWorkflowStore(sys.argv[1]) as store:\n"
        "    store.create_run(None, {}, None, run_id='run')\n"
        "    assert 'actions' not in store.inspect_run('run') and store.audit('run')['ok']\n"
        "assert not {'smythe.actions', 'smythe.workflow_actions'} & set(sys.modules)\n"
        "print('ok')\n"
    )
    completed = subprocess.run([child_executable(), "-c", code, str(tmp_path / "workflow.db")],
                               env=offline_env(), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_action_records_are_append_only(action_template, tmp_path):
    path = action_journal(action_template, tmp_path)[0]
    before = dump(path)
    for statement in ("UPDATE workflow_action_records SET trusted=1", "DELETE FROM workflow_action_records"):
        with closing(CONNECT(path)) as db:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(statement)
    assert dump(path) == before


def test_the_audit_detects_a_tampered_action_balance(action_template, tmp_path):
    path = action_journal(action_template, tmp_path)[0]
    with closing(CONNECT(path)) as db, db:
        db.execute("UPDATE workflow_actions SET confirmed='2000000' WHERE action_id='published'")
    with SQLiteWorkflowStore(path) as store:
        message = "External action balances disagree with its attempts and records"
        with pytest.raises(WorkflowCorruptionError, match=message):
            store.audit("acting")
        with pytest.raises(WorkflowCorruptionError, match=message):
            store.inspect_run("acting")
        with pytest.raises(WorkflowCorruptionError, match=message):
            store.recover(store.acquire_lease("acting", "owner"))
        # The audit covers one run's actions; another run is unaffected.
        assert store.audit("quiet")["ok"] is True
        with closing(CONNECT(path)) as db, db:
            db.execute("UPDATE workflow_actions SET confirmed='2010000' WHERE action_id='published'")
        assert store.audit("acting")["ok"] is True


ALLOW_RECORD_UPDATES = "DROP TRIGGER workflow_action_records_no_update"
TAMPERING = {
    "attempt-cost": (["UPDATE workflow_action_attempts SET cost='1999999' WHERE attempt_id='attempt-2'"],
                     "balances disagree"),
    "read-cost": ([ALLOW_RECORD_UPDATES, "UPDATE workflow_action_records SET cost='0' WHERE record_id='readback-1'"],
                  "balances disagree"),
    "reservation": (["UPDATE workflow_actions SET reserved='0' WHERE action_id='undo'"], "balances disagree"),
    "exposure": (["UPDATE workflow_action_attempts SET unknown='0' WHERE attempt_id='pending-1'"],
                 "balances disagree"),
    "amount-form": (["UPDATE workflow_actions SET confirmed='02010000' WHERE action_id='published'"],
                    "canonical nanoUSD"),
    "action-state": (["UPDATE workflow_actions SET state='published' WHERE action_id='published'"],
                     "identity or state"),
    "intent": ([("UPDATE workflow_actions SET intent_json=? WHERE action_id='published'",
                 (_canonical(intent(title="edited").to_dict()),))], "identity or state"),
    "action-key": (["UPDATE workflow_actions SET slot='republish' WHERE action_id='published'"],
                   "identity or state"),
    "attempt-state": (["UPDATE workflow_action_attempts SET state='sent' WHERE attempt_id='attempt-2'"],
                      "attempt is inconsistent"),
    "unclaimed-dispatch": (["UPDATE workflow_action_attempts SET dispatch_token_sha=NULL "
                            "WHERE attempt_id='attempt-2'"], "attempt is inconsistent"),
    "rejected-approval": (["UPDATE workflow_action_approvals SET verdict='rejected' WHERE approval_id='approval-3'"],
                          "attempt is inconsistent"),
    "approval-intent": ([("UPDATE workflow_action_approvals SET intent_sha=? WHERE approval_id='approval-3'",
                          (intent().digest(),))], "approval is inconsistent"),
    "assertion": (["UPDATE workflow_action_approvals SET assertion='forged' WHERE approval_id='approval-1'"],
                  "approval is inconsistent"),
    "record-payload": ([ALLOW_RECORD_UPDATES, "UPDATE workflow_action_records SET payload_json='{\"notice\":1}' "
                        "WHERE record_id='notice-1'"], "record is inconsistent"),
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_the_audit_detects_tampered_action_rows(action_template, tmp_path, case):
    path = action_journal(action_template, tmp_path)[0]
    statements, message = TAMPERING[case]
    with closing(CONNECT(path)) as db, db:
        for statement in statements:
            sql, *parameters = (statement,) if isinstance(statement, str) else statement
            db.execute(sql, *parameters)
    with SQLiteWorkflowStore(path, read_only=True) as reader:
        with pytest.raises(WorkflowCorruptionError, match=message):
            reader.audit("acting")
        with pytest.raises(WorkflowCorruptionError, match=message):
            reader.inspect_run("acting")
        assert reader.audit("quiet")["ok"] is True
