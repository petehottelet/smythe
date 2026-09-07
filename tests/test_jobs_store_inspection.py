"""Read-only discovery sees live Jobs journals without advancing their state."""

import json
import sqlite3
from threading import Event, Thread

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.store import (
    JobNotFoundError, MAX_LIST_RUNS, MAX_SQLITE_INTEGER, RunStatus, RunStoreError, SQLiteRunStore,
)


def create_run(store, root, run_id="run", *, name="inspection-job", count=2):
    manifest = JobManifestV1.from_dict({
        "version": 1, "name": name,
        "profiles": [{"name": "test", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [{"key": "item", "count": count, "prompt": "PRIVATE PROMPT", "profile": "test"}],
        "execution": {"max_concurrency": 2, "max_attempts": 2,
                      "max_budget_usd": "0", "output_directory": "PRIVATE OUTPUT PATH"},
    })
    plan = preflight_job(manifest, manifest_root=root)
    store.create_run(plan, make_approval(plan), manifest_root=root, run_id=run_id)
    return run_id


@pytest.fixture
def journal(tmp_path):
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        create_run(store, tmp_path)
        yield store


def logical_dump(path):
    with sqlite3.connect(path) as db:
        return list(db.iterdump())


def test_read_only_open_and_all_read_methods_preserve_database(journal, monkeypatch):
    from smythe.provider_responses import OpenAIResponsesProvider
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda *args: pytest.fail("SDK constructed"))
    before = logical_dump(journal.path)
    database_bytes = journal.path.read_bytes()
    wal_path = journal.path.with_name(journal.path.name + "-wal")
    wal_bytes = wal_path.read_bytes() if wal_path.exists() else None
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        assert reader.read_only
        assert reader._connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert reader.list_runs()[0]["run_id"] == "run"
        assert reader.list_run_page()["returned"] == 1
        assert reader.get_run("run")["run_id"] == "run"
        assert reader.snapshot("run", include_events=True)["counts"] == {"pending": 2}
        assert reader.inspection_snapshot("run")["counts"] == {"pending": 2}
        assert len(reader.pending_operations("run")) == 2
        assert reader.manifest_record("run")["manifest_json"]
        assert reader.get_run_lease("run") is None
        assert reader._call_rows("run") == []
    assert logical_dump(journal.path) == before
    assert journal.path.read_bytes() == database_bytes
    if wal_bytes is not None:
        assert wal_path.read_bytes() == wal_bytes


def test_read_only_mutations_fail_before_any_sql_write(journal):
    before = logical_dump(journal.path)
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        statements = []
        reader._connection.set_trace_callback(statements.append)
        actions = [lambda: reader.acquire_run_lease("run", "owner"),
                   lambda: reader.finalize_run("run"),
                   lambda: reader.begin_attempt("run", reader.pending_operations("run")[0]["operation_id"]),
                   lambda: reader.recover_inflight("run")]
        for action in actions:
            with pytest.raises(RunStoreError, match="read-only"):
                action()
        assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "BEGIN IMMEDIATE")) for sql in statements)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._connection.execute("DELETE FROM runs")
    assert logical_dump(journal.path) == before


def test_read_only_missing_database_creates_neither_file_nor_parent(tmp_path):
    path = tmp_path / "missing" / "nested" / "jobs.db"
    with pytest.raises(RunStoreError, match="Cannot open"):
        SQLiteRunStore(path, read_only=True)
    assert not path.exists() and not path.parent.exists()


@pytest.mark.parametrize("kind", ["foreign", "corrupt", "empty", "missing_table", "missing_column", "unsupported_version"])
def test_read_only_rejects_foreign_or_invalid_schema_without_repair(tmp_path, kind):
    path = tmp_path / "invalid.db"
    if kind == "corrupt":
        path.write_bytes(b"not a sqlite database")
    elif kind in ("foreign", "empty"):
        with sqlite3.connect(path) as db:
            if kind == "foreign":
                db.execute("CREATE TABLE workflow_runs(run_id TEXT)")
                db.execute("PRAGMA user_version=2")
    else:
        with SQLiteRunStore(path) as store:
            create_run(store, tmp_path)
        with sqlite3.connect(path) as db:
            if kind == "missing_table":
                db.execute("DROP TABLE run_leases")
            elif kind == "missing_column":
                db.execute("ALTER TABLE runs DROP COLUMN approval_token")
            else:
                db.execute("PRAGMA user_version=999")
    before = path.read_bytes()
    with pytest.raises(RunStoreError):
        SQLiteRunStore(path, read_only=True)
    assert path.read_bytes() == before


def test_version_one_is_never_upgraded_by_reader_but_writable_upgrade_remains(tmp_path):
    path = tmp_path / "legacy.db"
    with SQLiteRunStore(path) as store:
        create_run(store, tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE run_controls")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_namespace")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_owner_id")
        db.execute("DROP TABLE run_leases")
        db.execute("ALTER TABLE runs DROP COLUMN lease_epoch")
        db.execute("ALTER TABLE attempts DROP COLUMN lease_owner_id")
        db.execute("ALTER TABLE attempts DROP COLUMN lease_epoch")
        db.execute("PRAGMA user_version=1")
    before = logical_dump(path)
    with pytest.raises(RunStoreError, match="version"):
        SQLiteRunStore(path, read_only=True)
    assert logical_dump(path) == before
    with SQLiteRunStore(path) as writer:
        assert writer._connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert writer.get_run_lease("run") is None
        assert writer.get_run("run")["run_id"] == "run"


def test_foreign_writable_database_is_not_overwritten(tmp_path):
    path = tmp_path / "workflow.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE workflow_runs(run_id TEXT)")
    before = path.read_bytes()
    with pytest.raises(RunStoreError):
        SQLiteRunStore(path)
    assert path.read_bytes() == before


def test_live_wal_commit_is_visible_on_the_next_read(journal):
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        assert reader.list_runs()[0]["counts"] == {"pending": 2}
        operation_id = journal.pending_operations("run")[0]["operation_id"]
        journal.begin_attempt("run", operation_id)
        current = reader.list_runs()[0]
        assert current["status"] == "running"
        assert current["counts"] == {"pending": 1, "running": 1}
        assert reader.snapshot("run")["counts"] == current["counts"]


@pytest.mark.parametrize("projection", ["list_runs", "snapshot", "inspection_snapshot"])
def test_multiquery_projection_is_one_snapshot_during_writer_commit(journal, projection):
    operation_id = journal.pending_operations("run")[0]["operation_id"]
    query_started, committed = Event(), Event()
    errors = []
    with SQLiteRunStore(journal.path, read_only=True) as reader, SQLiteRunStore(journal.path) as writer:
        def trace(sql):
            marker = "GROUP BY status" if projection == "list_runs" else "SELECT * FROM operations WHERE run_id"
            if marker in sql:
                query_started.set()
                if not committed.wait(10):
                    errors.append(AssertionError("writer did not commit within the barrier"))

        def write():
            try:
                assert query_started.wait(10)
                writer.begin_attempt("run", operation_id)
            except BaseException as exc:
                errors.append(exc)
            finally:
                committed.set()

        reader._connection.set_trace_callback(trace)
        thread = Thread(target=write)
        thread.start()
        try:
            previous = reader.list_runs()[0] if projection == "list_runs" else getattr(reader, projection)("run")
        finally:
            reader._connection.set_trace_callback(None)
            committed.set()
            thread.join(timeout=10)
        assert not thread.is_alive() and errors == []
        assert previous["status"] == "approved"
        assert previous["counts"] == {"pending": 2}
        assert reader.list_runs()[0]["counts"] == {"pending": 1, "running": 1}


def test_bounded_pages_status_and_tie_order_are_stable(journal):
    create_run(journal, journal.path.parent, "new", name="Newer", count=1)
    create_run(journal, journal.path.parent, "tie-b", name="Tie-B", count=3)
    create_run(journal, journal.path.parent, "tie-a", name="Tie-A", count=4)
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE runs SET created_at_ns=20 WHERE run_id LIKE 'tie-%'")
        db.execute("UPDATE runs SET created_at_ns=30 WHERE run_id='new'")
        db.execute("UPDATE runs SET created_at_ns=10 WHERE run_id='run'")
    journal.begin_attempt("run", journal.pending_operations("run")[0]["operation_id"])
    assert [run["run_id"] for run in journal.list_runs(limit=2)] == ["new", "tie-a"]
    assert [run["run_id"] for run in journal.list_runs(limit=2, offset=2)] == ["tie-b", "run"]
    assert journal.list_runs(offset=100) == []
    assert [run["run_id"] for run in journal.list_runs(status=RunStatus.RUNNING)] == ["run"]
    assert len(journal.list_runs(status="approved")) == 3
    assert journal.list_runs(status="failed") == []
    assert journal.list_runs(limit=2, offset=1)[0]["operation_count"] == 4


@pytest.mark.parametrize("kwargs", [{"limit": True}, {"limit": 0}, {"limit": -1}, {"limit": MAX_LIST_RUNS + 1},
    {"limit": 1.0}, {"limit": "1"}, {"offset": True}, {"offset": -1}, {"offset": MAX_SQLITE_INTEGER + 1},
    {"offset": 1.0}, {"offset": "0"}, {"status": ""}, {"status": "RUNNING"}, {"status": []}, {"status": 1}])
def test_invalid_page_options_fail_without_sql(journal, kwargs):
    statements = []
    journal._connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError):
            journal.list_runs(**kwargs)
    finally:
        journal._connection.set_trace_callback(None)
    assert statements == []


def test_listing_does_not_load_prompts_results_or_artifact_rows(journal):
    statements = []
    journal._connection.set_trace_callback(statements.append)
    try:
        summaries = journal.list_runs()
    finally:
        journal._connection.set_trace_callback(None)
    assert "PRIVATE" not in json.dumps(summaries)
    assert not any("FROM artifacts" in sql or "spec_json" in sql or "result_text" in sql or "SELECT *" in sql for sql in statements)
    # A corrupt plan outside the selected page cannot force an unbounded parse.
    create_run(journal, journal.path.parent, "older")
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE runs SET plan_json='bad JSON',created_at_ns=0 WHERE run_id='older'")
    assert journal.list_runs(limit=1)[0]["run_id"] == "run"
    with pytest.raises(RunStoreError):
        journal.list_runs(limit=2)


@pytest.mark.parametrize("column,value", [("plan_json", '{"name":123}'), ("status", "nonsense"),
    ("confirmed_microusd", -1), ("created_at_ns", -1)])
def test_corrupt_summary_data_fails_closed(journal, column, value):
    with sqlite3.connect(journal.path) as db:
        db.execute(f"UPDATE runs SET {column}=?", (value,))
    with pytest.raises(RunStoreError):
        journal.list_runs()


def test_summary_costs_and_estimates_match_existing_snapshot(journal):
    operation_id = journal.pending_operations("run")[0]["operation_id"]
    attempt = journal.begin_attempt("run", operation_id)
    permit = journal.prepare_call(attempt["attempt_id"], 0)
    journal.mark_call_dispatched(permit.call_id)
    journal.mark_unknown_outcome(permit.call_id, "offline process stopped")
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        summary = reader.list_runs()[0]
        snapshot = reader.snapshot("run")
        assert summary["cost"] == snapshot["cost"]
        assert summary["cost"]["cost_contains_estimates"] is True
        assert summary["counts"] == snapshot["counts"]


def complete_operation(store, operation_id, *, accepted=True):
    attempt = store.begin_attempt("run", operation_id)
    permit = store.prepare_call(attempt["attempt_id"], 0)
    store.mark_call_dispatched(permit.call_id)
    store.complete_call(permit.call_id, cost_microusd=0, cost_is_complete=True, cost_is_estimate=False,
                        artifacts=[{"relative_path": f"artifacts/{attempt['attempt_id']}.png", "mime_type": "image/png",
                                    "sha256": "a" * 64, "size_bytes": 1}],
                        result_text="PRIVATE RESULT", accepted=accepted)
    return permit


def test_inspection_pages_selected_lineage_and_recent_events(journal):
    operations = journal.pending_operations("run")
    first = operations[0]["operation_id"]
    complete_operation(journal, first, accepted=False)
    journal.queue_reroll("run", [first], reason="review rejected version")
    complete_operation(journal, first)
    complete_operation(journal, operations[1]["operation_id"])
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        full = reader.snapshot("run", include_events=True)
        page = reader.inspection_snapshot("run", limit=1, events_limit=2)
        assert page["pagination"] == {"limit": 1, "offset": 0, "total": 2, "returned": 1, "has_more": True}
        assert [item["operation_id"] for item in page["operations"]] == [first]
        assert len(page["attempts"]) == len(page["calls"]) == len(page["artifacts"]) == 2
        for kind in ("attempts", "calls", "artifacts"):
            assert {row["operation_id"] for row in page[kind]} == {first}
        assert page["counts"] == full["counts"] and page["cost"] == full["cost"]
        assert page["events"] == full["events"][-2:]
        assert page["event_pagination"] == {"limit": 2, "total": len(full["events"]), "returned": 2, "has_more": True}
        last = reader.inspection_snapshot("run", limit=1, offset=1)
        assert last["pagination"]["has_more"] is False
        assert last["operations"][0]["operation_id"] == operations[1]["operation_id"]
        empty = reader.inspection_snapshot("run", offset=2)
        assert empty["operations"] == empty["attempts"] == empty["calls"] == empty["artifacts"] == []
        assert empty["pagination"]["total"] == 2


def test_inspection_exact_operation_filter_accepts_key_or_id(journal):
    operation = journal.pending_operations("run")[0]
    complete_operation(journal, operation["operation_id"])
    by_id = journal.inspection_snapshot("run", operation=operation["operation_id"], events_limit=1)
    by_key = journal.inspection_snapshot("run", operation=operation["operation_key"], events_limit=1)
    assert by_id["operations"] == by_key["operations"]
    assert by_id["calls"] == by_key["calls"]
    assert by_id["pagination"]["total"] == 1
    assert by_id["counts"] == {"pending": 1, "succeeded": 1}
    assert by_id["event_pagination"]["total"] == 4
    assert by_id["events"][0]["operation_id"] == operation["operation_id"]
    assert by_id["events"][0]["event_type"] == "call_completed"
    assert by_id["operation_filter"] == operation["operation_id"]


def test_inspection_rejects_missing_or_ambiguous_operation(journal):
    with pytest.raises(JobNotFoundError):
        journal.inspection_snapshot("absent")
    with pytest.raises(ValueError, match="No operation"):
        journal.inspection_snapshot("run", operation="absent")
    operations = journal.pending_operations("run")
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE operations SET operation_key=? WHERE operation_id=?",
                   (operations[0]["operation_id"], operations[1]["operation_id"]))
    with pytest.raises(ValueError, match="more than one"):
        journal.inspection_snapshot("run", operation=operations[0]["operation_id"])


@pytest.mark.parametrize("kwargs", [{"limit": True}, {"limit": 0}, {"limit": 501}, {"offset": -1},
    {"offset": True}, {"offset": MAX_SQLITE_INTEGER + 1}, {"events_limit": 0}, {"events_limit": True},
    {"events_limit": 1001}, {"events_limit": 1.0}, {"operation": ""}, {"operation": []}])
def test_inspection_validates_bounds_before_queries(journal, kwargs):
    statements = []
    journal._connection.set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError):
            journal.inspection_snapshot("run", **kwargs)
    finally:
        journal._connection.set_trace_callback(None)
    assert statements == []


def test_inspection_does_not_parse_off_page_operation_specs(journal):
    operations = journal.pending_operations("run")
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE operations SET spec_json='invalid json' WHERE operation_id=?", (operations[1]["operation_id"],))
    page = journal.inspection_snapshot("run", limit=1)
    assert page["operations"][0]["operation_id"] == operations[0]["operation_id"]
    assert page["pagination"]["has_more"]


def test_maximum_list_page_has_exact_more_indicator(journal):
    with sqlite3.connect(journal.path) as db:
        fields = [row[1] for row in db.execute("PRAGMA table_info(runs)") if row[1] != "run_id"]
        columns = ",".join(fields)
        db.executemany(f"INSERT INTO runs(run_id,{columns}) SELECT ?,{columns} FROM runs WHERE run_id='run'",
                       [(f"copy-{number:03d}",) for number in range(501)])
    first = journal.list_run_page(limit=500)
    assert first["returned"] == 500 and first["has_more"] is True
    last = journal.list_run_page(limit=500, offset=500)
    assert last["returned"] == 2 and last["has_more"] is False
    assert set(row["run_id"] for row in first["runs"]).isdisjoint(row["run_id"] for row in last["runs"])


@pytest.mark.parametrize("table,column", [("operations", "spec_json"), ("events", "payload_json")])
def test_selected_json_corruption_is_a_store_error_for_inspection_and_export(journal, table, column):
    invalid_values = [
        '{"extra":NaN}', '{"extra":Infinity}', '{"extra":-Infinity}',
        '{"extra":[1e999]}', '{"extra":{"same":1,"same":2}}',
        '[]', 'null', '"text"', '{bad JSON', b'{"extra":"BLOB"}',
    ]
    with sqlite3.connect(journal.path) as db:
        original = db.execute(f"SELECT {column} FROM {table} LIMIT 1").fetchone()[0]
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        for value in invalid_values:
            with sqlite3.connect(journal.path) as db:
                db.execute(f"UPDATE {table} SET {column}=?", (value,))
            for read in (lambda: reader.inspection_snapshot("run"),
                         lambda: reader.snapshot("run", include_events=True)):
                with pytest.raises(RunStoreError, match=f"{table}.{column}"):
                    read()
        with sqlite3.connect(journal.path) as db:
            db.execute(f"UPDATE {table} SET {column}=?", (original,))
        # A failed read must also release its snapshot/transaction.
        assert reader.inspection_snapshot("run")["counts"] == {"pending": 2}


def test_nontext_job_name_is_rejected_by_all_public_projections(journal):
    invalid_values = ['{"name":[]}', '{"name":{}}', '{"name":1}',
                      '{"name":null}', b'{"name":"BLOB"}']
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        for value in invalid_values:
            with sqlite3.connect(journal.path) as db:
                db.execute("UPDATE runs SET plan_json=?", (value,))
            for read in (reader.list_runs, lambda: reader.inspection_snapshot("run"),
                         lambda: reader.snapshot("run", include_events=True)):
                with pytest.raises(RunStoreError):
                    read()


@pytest.mark.parametrize("table,column,value", [
    ("runs", "manifest_root", b"unusable path"),
    ("operations", "result_text", b"unusable result"),
    ("attempts", "result_text", b"unusable result"),
    ("artifacts", "relative_path", b"unusable path"),
    ("events", "event_type", b"unusable type"),
    ("calls", "provider_request_id", b"unusable ID"),
    ("calls", "cost_is_complete", 2),
])
def test_selected_scalar_corruption_cannot_escape_into_json(journal, table, column, value):
    operation = journal.pending_operations("run")[0]
    complete_operation(journal, operation["operation_id"])
    with sqlite3.connect(journal.path) as db:
        db.execute(f"UPDATE {table} SET {column}=?", (value,))
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        with pytest.raises(RunStoreError, match=f"{table}.{column}"):
            reader.inspection_snapshot("run")
        # Legacy status/export omit full call rows, but share all other guards.
        if table != "calls":
            with pytest.raises(RunStoreError, match=f"{table}.{column}"):
                reader.snapshot("run", include_events=True)


def test_valid_nested_json_remains_lossless_and_strictly_serializable(journal):
    value = {"prompt": "source data", "extra": [None, True, 1, 1.25, {"unicode": "雨"}]}
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE operations SET spec_json=?", (encoded,))
        db.execute("UPDATE events SET payload_json=?", (encoded,))
    with SQLiteRunStore(journal.path, read_only=True) as reader:
        for snapshot in (reader.inspection_snapshot("run"), reader.snapshot("run", include_events=True)):
            assert snapshot["operations"][0]["spec"] == value
            assert snapshot["events"][0]["payload"] == value
            json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
