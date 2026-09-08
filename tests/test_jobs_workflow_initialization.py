"""Real SQLite connections exercise constructor visibility and rollback boundaries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
from threading import Barrier, Event

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.store import RunStoreError, SQLiteRunStore
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowCorruptionError


CONNECT = sqlite3.connect
STORES = [SQLiteRunStore, SQLiteWorkflowStore]


def _connection(store):
    return store._connection if isinstance(store, SQLiteRunStore) else store._db


def _identity(store):
    return store.schema_version if isinstance(store, SQLiteRunStore) else store.store_id


def _assert_integrity(path):
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def _tables(path):
    with closing(CONNECT(path)) as db:
        return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _dump(path):
    with closing(CONNECT(path)) as db:
        return tuple(db.iterdump())


def _seed(store, root):
    if isinstance(store, SQLiteWorkflowStore):
        store.create_run({"goal": "retained task", "context": {"source": "unaltered"}},
                         {"kind": "offline"}, 0, run_id="retained")
    else:
        manifest = JobManifestV1.from_dict({
            "version": 1, "name": "retained-job",
            "profiles": [{"name": "test", "provider": "offline", "model": "offline-image",
                          "max_cost_per_call_usd": "0"}],
            "operations": [{"key": "item", "count": 1, "prompt": "retained source", "profile": "test"}],
            "execution": {"max_concurrency": 1, "max_attempts": 1,
                          "max_budget_usd": "0", "output_directory": "outputs"},
        })
        plan = preflight_job(manifest, manifest_root=root)
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="retained")


def _legacy_jobs(path, root, version):
    with SQLiteRunStore(path) as store:
        _seed(store, root)
    with closing(CONNECT(path)) as db, db:
        db.execute("DROP TABLE run_controls")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_namespace")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_owner_id")
        if version < 3:
            db.execute("ALTER TABLE runs DROP COLUMN lease_epoch")
            db.execute("ALTER TABLE attempts DROP COLUMN lease_owner_id")
            db.execute("ALTER TABLE attempts DROP COLUMN lease_epoch")
            db.execute("ALTER TABLE run_leases DROP COLUMN epoch")
        if version == 1:
            db.execute("DROP TABLE run_leases")
        db.execute(f"PRAGMA user_version={version}")


def _install_connections(monkeypatch, *, trace=None, authorizer=None):
    opened = []

    class ObservedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            super().close()
            self.closed = True

    def connect(*args, **kwargs):
        db = CONNECT(*args, **kwargs, factory=ObservedConnection)
        opened.append(db)
        if trace is not None:
            db.set_trace_callback(trace)
        if authorizer is not None:
            db.set_authorizer(authorizer)
        return db

    monkeypatch.setattr(sqlite3, "connect", connect)
    return opened


@pytest.mark.parametrize("store_type", STORES)
def test_schema_is_invisible_until_all_ddl_and_identity_commit(tmp_path, monkeypatch, store_type):
    path = tmp_path / "fresh.db"
    entered, release, timed_out = Event(), Event(), Event()
    second = "OPERATIONS" if store_type is SQLiteRunStore else "WORKFLOW_RUNS"

    def trace(sql):
        normalized = " ".join(sql.upper().split()).replace(" (", "(")
        if normalized.startswith("CREATE TABLE IF NOT EXISTS " + second + "("):
            entered.set()
            if not release.wait(15):
                timed_out.set()

    opened = _install_connections(monkeypatch, trace=trace)

    def initialize():
        with store_type(path) as store:
            return _identity(store)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(initialize)
        try:
            assert entered.wait(15), "constructor did not reach its second table"
            visible = _tables(path)
        finally:
            release.set()
        identity = future.result(timeout=15)
    assert not timed_out.is_set()
    assert visible == set(), "a concurrent reader saw a partially initialized schema"
    assert identity
    assert all(db.closed for db in opened)
    _assert_integrity(path)
    with store_type(path, read_only=True) as reader:
        assert _identity(reader) == identity


@pytest.mark.parametrize("store_type", STORES)
@pytest.mark.parametrize("existing", [False, True], ids=["fresh", "delete-mode"])
def test_two_public_constructors_recheck_under_the_write_lock(tmp_path, monkeypatch, store_type, existing):
    path = tmp_path / "concurrent.db"
    previous = None
    if existing:
        with store_type(path) as store:
            _seed(store, tmp_path)
            previous = _identity(store)
        with closing(CONNECT(path)) as db:
            assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
        before = _dump(path)
    preflight = Barrier(2)
    opened_together = Barrier(2)
    trace_failures = []

    def trace(sql):
        if "".join(sql.upper().split()) == "PRAGMASYNCHRONOUS=FULL":
            try:
                preflight.wait(timeout=15)
            except BaseException as exc:
                trace_failures.append(exc)

    opened = _install_connections(monkeypatch, trace=trace)

    def initialize():
        with store_type(path) as store:
            identity = _identity(store)
            opened_together.wait(timeout=15)
            return identity

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(initialize) for _ in range(2)]
        identities = [future.result(timeout=30) for future in futures]
    assert not trace_failures
    assert identities[0] == identities[1]
    if existing:
        assert identities[0] == previous
        assert _dump(path) == before
    if store_type is SQLiteWorkflowStore:
        with closing(CONNECT(path)) as db:
            assert db.execute("SELECT COUNT(*) FROM workflow_meta").fetchone() == (1,)
    assert all(db.closed for db in opened)
    _assert_integrity(path)


@pytest.mark.parametrize("store_type", STORES)
def test_schema_changed_after_preflight_is_not_overwritten(tmp_path, monkeypatch, store_type):
    path = tmp_path / "recheck.db"
    entered, release, timed_out = Event(), Event(), Event()

    def trace(sql):
        if "".join(sql.upper().split()) == "PRAGMASYNCHRONOUS=FULL":
            entered.set()
            if not release.wait(15):
                timed_out.set()

    opened = _install_connections(monkeypatch, trace=trace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store_type, path)
        try:
            assert entered.wait(15)
            with closing(CONNECT(path)) as db, db:
                db.execute("CREATE TABLE unrelated(payload TEXT)")
                db.execute("INSERT INTO unrelated VALUES ('retained evidence')")
                db.execute("PRAGMA user_version=999")
            before = _dump(path)
        finally:
            release.set()
        unexpected = None
        try:
            with pytest.raises((RunStoreError, WorkflowCorruptionError)):
                unexpected = future.result(timeout=15)
        finally:
            if unexpected is not None:
                unexpected.close()
    assert not timed_out.is_set()
    assert all(db.closed for db in opened)
    assert _dump(path) == before
    assert _tables(path) == {"unrelated"}
    _assert_integrity(path)


@pytest.mark.parametrize("store_type", STORES)
def test_real_ddl_denial_rolls_back_every_table_and_closes_connection(tmp_path, monkeypatch, store_type):
    path = tmp_path / "denied.db"
    second = "operations" if store_type is SQLiteRunStore else "workflow_runs"

    def authorize(action, name, *_):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TABLE and name == second else sqlite3.SQLITE_OK

    opened = _install_connections(monkeypatch, authorizer=authorize)
    with pytest.raises((RunStoreError, WorkflowCorruptionError)) as caught:
        store_type(path)
    assert isinstance(caught.value.__cause__, sqlite3.DatabaseError)
    assert all(db.closed for db in opened)
    assert _tables(path) == set()
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
    _assert_integrity(path)


@pytest.mark.parametrize("store_type", STORES)
def test_baseexception_during_initialization_rolls_back_and_preserves_error(tmp_path, monkeypatch, store_type):
    path = tmp_path / "interrupted.db"
    failure = KeyboardInterrupt("stop during initialization")

    def interrupted(self, *_):
        _connection(self).execute("CREATE TABLE must_rollback(payload TEXT)")
        raise failure

    opened = _install_connections(monkeypatch)
    monkeypatch.setattr(store_type, "_create_schema", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        store_type(path)
    assert caught.value is failure
    assert all(db.closed for db in opened)
    assert _tables(path) == set()
    _assert_integrity(path)


@pytest.mark.parametrize("store_type", STORES)
@pytest.mark.parametrize("invalid", ["foreign", "unsupported", "missing-table", "bad-identity"])
def test_invalid_schema_is_rejected_before_journal_mode_changes(tmp_path, store_type, invalid):
    path = tmp_path / "invalid.db"
    if invalid == "foreign":
        with closing(CONNECT(path)) as db:
            db.execute("CREATE TABLE unrelated(value TEXT)")
    else:
        with store_type(path):
            pass
        with closing(CONNECT(path)) as db, db:
            if store_type is SQLiteRunStore:
                if invalid == "unsupported":
                    db.execute("PRAGMA user_version=999")
                elif invalid == "missing-table":
                    db.execute("DROP TABLE calls")
                else:
                    db.execute("ALTER TABLE attempts DROP COLUMN lease_epoch")
            elif invalid == "unsupported":
                db.execute("UPDATE workflow_meta SET version=999")
            elif invalid == "missing-table":
                db.execute("DROP TABLE workflow_evidence")
            else:
                db.execute("UPDATE workflow_meta SET store_id='invalid'")
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    before = path.read_bytes()
    with pytest.raises((RunStoreError, WorkflowCorruptionError)):
        store_type(path)
    assert path.read_bytes() == before
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)


@pytest.mark.parametrize("store_type", STORES)
def test_read_only_initialization_never_acquires_write_lock_or_sets_wal(tmp_path, monkeypatch, store_type):
    path = tmp_path / "readonly.db"
    with store_type(path) as store:
        identity = _identity(store)
    with closing(CONNECT(path)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    before = path.read_bytes()
    statements = []
    opened = _install_connections(monkeypatch, trace=statements.append)
    with store_type(path, read_only=True) as store:
        assert _identity(store) == identity
    assert all(db.closed for db in opened)
    compact = ["".join(sql.upper().split()) for sql in statements]
    assert "BEGINIMMEDIATE" not in compact
    assert not any(sql.startswith(("PRAGMAJOURNAL_MODE=", "CREATE", "INSERT", "UPDATE")) for sql in compact)
    assert path.read_bytes() == before


@pytest.mark.parametrize("store_type", STORES)
def test_read_only_missing_store_does_not_create_parent(tmp_path, store_type):
    path = tmp_path / "missing" / "readonly.db"
    with pytest.raises((RunStoreError, sqlite3.OperationalError)):
        store_type(path, read_only=True)
    assert not path.parent.exists()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_concurrent_jobs_migration_retains_one_identity_and_historical_events(tmp_path, monkeypatch, version):
    path = tmp_path / "legacy.db"
    _legacy_jobs(path, tmp_path, version)
    with closing(CONNECT(path)) as db:
        events = db.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        immutable_run = db.execute("SELECT manifest_json,plan_json,approval_json FROM runs").fetchall()
    preflight, completed = Barrier(2), Barrier(2)
    failures = []

    def trace(sql):
        if "".join(sql.upper().split()) == "PRAGMASYNCHRONOUS=FULL":
            try:
                preflight.wait(timeout=15)
            except BaseException as exc:
                failures.append(exc)

    opened = _install_connections(monkeypatch, trace=trace)

    def initialize():
        with SQLiteRunStore(path) as store:
            row = _connection(store).execute("SELECT artifact_owner_id FROM runs").fetchone()
            result = store.schema_version, row[0]
            completed.wait(timeout=15)
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(initialize) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert not failures
    assert results[0] == results[1]
    assert results[0][0] == 4 and len(results[0][1]) == 32
    assert all(db.closed for db in opened)
    with closing(CONNECT(path)) as db:
        assert db.execute("SELECT * FROM events ORDER BY sequence").fetchall() == events
        assert db.execute("SELECT manifest_json,plan_json,approval_json FROM runs").fetchall() == immutable_run
        assert db.execute("SELECT COUNT(*) FROM run_controls").fetchone() == (1,)
    _assert_integrity(path)


def test_jobs_migration_interruption_rolls_back_ddl_version_and_identity(tmp_path, monkeypatch):
    path = tmp_path / "migration.db"
    _legacy_jobs(path, tmp_path, 3)
    before = _dump(path)
    original = SQLiteRunStore._migrate_schema
    failure = KeyboardInterrupt("interrupted before migration commit")

    def interrupted(self, *args):
        original(self, *args)
        raise failure

    opened = _install_connections(monkeypatch)
    monkeypatch.setattr(SQLiteRunStore, "_migrate_schema", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        SQLiteRunStore(path)
    assert caught.value is failure
    assert all(db.closed for db in opened)
    assert _dump(path) == before
    _assert_integrity(path)
