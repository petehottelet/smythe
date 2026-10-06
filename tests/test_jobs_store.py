"""Durability and ambiguity tests for the Jobs v1 SQLite dispatch journal."""

from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import ExitStack, closing
from threading import Event, Thread

import pytest

from smythe.jobs import ApprovalError, JobManifestV1, make_approval, preflight_job
from smythe.jobs.store import (
    InvalidTransitionError,
    JobBudgetError,
    JobNotFoundError,
    OperationStatus,
    RunLeaseError,
    RunStatus,
    RunStoreError,
    SQLiteRunStore,
)


@pytest.fixture
def store_factory():
    """Close test-owned stores after all assertions and workers finish."""
    with ExitStack() as stack:
        def create(path):
            return stack.enter_context(SQLiteRunStore(path))

        yield create


def _plan(tmp_path, *, provider="openai_image", attempts=2, count=1, budget=None):
    ceiling = "0" if provider == "offline" else "0.10"
    if budget is None:
        budget = "0" if provider == "offline" else f"{0.10 * attempts * count:.2f}"
    manifest = JobManifestV1.from_dict(
        {
            "version": 1,
            "name": "store-test",
            "profiles": [
                {
                    "name": "default",
                    "provider": provider,
                    "model": "offline-image" if provider == "offline" else "gpt-image-1",
                    "max_cost_per_call_usd": ceiling,
                }
            ],
            "operations": [
                {
                    "key": "glyph",
                    "count": count,
                    "prompt": "Generate one glyph",
                    "profile": "default",
                }
            ],
            "execution": {
                "max_concurrency": 4,
                "max_attempts": attempts,
                "max_budget_usd": budget,
                "output_directory": "outputs",
            },
        }
    )
    plan = preflight_job(manifest, manifest_root=tmp_path)
    return plan, make_approval(plan)


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../outside", "..\\outside", "name/child", "name\\child"],
)
def test_store_rejects_unsafe_custom_run_id(tmp_path, store_factory, run_id):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="run_id"):
        store.create_run(plan, approval, manifest_root=tmp_path, run_id=run_id)


def _created_payload(store, run_id):
    events = store.snapshot(run_id, include_events=True)["events"]
    return next(event["payload"] for event in events if event["event_type"] == "run_created")


def test_a_used_approval_creates_no_second_run_and_names_the_first(tmp_path, store_factory):
    plan, approval = _plan(tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    store.create_run(plan, approval, manifest_root=tmp_path, run_id="first")
    before = list(store._connection.iterdump())

    for run_id in (None, "second", "first"):
        with pytest.raises(ApprovalError, match="run 'first'.*smythe jobs resume first"):
            store.create_run(plan, approval, manifest_root=tmp_path, run_id=run_id)
    # The token is derived from the plan and ceiling, so a fresh approval and
    # another connection to the same journal are refused as well.
    with SQLiteRunStore(store.path) as other:
        with pytest.raises(ApprovalError, match="first"):
            other.create_run(plan, make_approval(plan), manifest_root=tmp_path)

    assert list(store._connection.iterdump()) == before


def test_a_different_ceiling_is_a_different_approval(tmp_path, store_factory):
    plan, full = _plan(tmp_path, budget="0.30")
    exact = make_approval(plan, approved_max_cost_usd="0.20")
    store = store_factory(tmp_path / "jobs.db")

    assert full.token != exact.token
    first = store.create_run(plan, full, manifest_root=tmp_path)
    second = store.create_run(plan, exact, manifest_root=tmp_path)

    assert first != second
    assert "approval_reuse" not in _created_payload(store, second)


def test_acknowledged_approval_reuse_records_the_earlier_runs(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")

    for run_id in ("first", "second", "third"):
        assert store.create_run(
            plan, approval, manifest_root=tmp_path, run_id=run_id, allow_approval_reuse=True,
        ) == run_id

    first = _created_payload(store, "first")
    assert "approval_reuse" not in first
    assert _created_payload(store, "second")["approval_reuse"] == {
        "prior_run_count": 1, "prior_run_ids": ["first"],
    }
    third = _created_payload(store, "third")
    assert third.pop("approval_reuse") == {"prior_run_count": 2, "prior_run_ids": ["first", "second"]}
    assert third == first
    assert sorted(run["run_id"] for run in store.list_runs()) == ["first", "second", "third"]


def test_reuse_record_keeps_the_five_latest_earlier_runs_and_a_count(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")
    run_ids = [f"run-{index}" for index in range(8)]
    for run_id in run_ids:
        store.create_run(plan, approval, manifest_root=tmp_path, run_id=run_id, allow_approval_reuse=True)

    assert _created_payload(store, "run-7")["approval_reuse"] == {
        "prior_run_count": 7, "prior_run_ids": run_ids[2:7],
    }


def test_refusal_names_at_most_the_five_latest_runs(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")
    run_ids = [f"run-{index}" for index in range(7)]
    for run_id in run_ids:
        store.create_run(plan, approval, manifest_root=tmp_path, run_id=run_id, allow_approval_reuse=True)

    with pytest.raises(ApprovalError) as refused:
        store.create_run(plan, approval, manifest_root=tmp_path)

    message = str(refused.value)
    assert "7 runs, most recently 'run-2', 'run-3', 'run-4', 'run-5', 'run-6';" in message
    assert "'run-1'" not in message and "smythe jobs resume run-6" in message


@pytest.mark.parametrize("value", [1, 0, None, "yes"])
def test_approval_reuse_acknowledgment_must_be_a_boolean(tmp_path, store_factory, value):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")

    with pytest.raises(TypeError, match="allow_approval_reuse must be a boolean"):
        store.create_run(plan, approval, manifest_root=tmp_path, allow_approval_reuse=value)

    assert store.list_runs() == []


def test_concurrent_creators_with_one_approval_produce_exactly_one_run(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline")
    database = tmp_path / "jobs.db"
    first, second = store_factory(database), store_factory(database)
    second_began = Event()
    outcomes: dict[str, object] = {}
    thread: Thread | None = None

    def create(store, run_id):
        try:
            outcomes[run_id] = store.create_run(plan, approval, manifest_root=tmp_path, run_id=run_id)
        except BaseException as exc:
            outcomes[run_id] = exc

    def hold_first_after_its_check(statement: str) -> None:
        nonlocal thread
        if thread is None and statement.startswith("SELECT run_id FROM runs WHERE approval_token"):
            # The second creator begins while the first still holds its write
            # transaction between the reuse check and the insert.
            thread = Thread(target=create, args=(second, "b"))
            thread.start()
            second_began.wait(10)

    def note_second_begin(statement: str) -> None:
        if statement == "BEGIN IMMEDIATE":
            second_began.set()

    first._connection.set_trace_callback(hold_first_after_its_check)
    second._connection.set_trace_callback(note_second_begin)
    try:
        create(first, "a")
    finally:
        first._connection.set_trace_callback(None)
        if thread is not None:
            thread.join(timeout=30)
        second._connection.set_trace_callback(None)

    assert thread is not None and not thread.is_alive() and second_began.is_set()
    assert outcomes["a"] == "a"
    assert isinstance(outcomes["b"], ApprovalError) and "'a'" in str(outcomes["b"])
    assert [run["run_id"] for run in first.list_runs()] == ["a"]


@pytest.fixture
def default_umask():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_new_database_and_its_wal_files_are_owner_only(tmp_path, store_factory, default_umask):
    """Regression: a new jobs.sqlite3 was created world-readable (0644)."""
    database = tmp_path / ".smythe" / "jobs.sqlite3"
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(database)
    store.create_run(plan, approval, manifest_root=tmp_path)

    wal = database.with_name(database.name + "-wal")
    shm = database.with_name(database.name + "-shm")
    assert wal.exists() and shm.exists()
    assert {_mode(path) for path in (database, wal, shm)} == {0o600}
    assert _mode(database.parent) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_existing_database_keeps_its_permissions(tmp_path, default_umask):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    database = directory / "jobs.sqlite3"
    SQLiteRunStore(database).close()
    database.chmod(0o640)

    with SQLiteRunStore(database) as store:
        plan, approval = _plan(tmp_path, provider="offline")
        store.create_run(plan, approval, manifest_root=tmp_path)
        wal = database.with_name(database.name + "-wal")
        assert _mode(database) == 0o640
        assert _mode(wal) == 0o640
    assert _mode(directory) == 0o755


def test_complete_call_persists_artifact_and_cost(tmp_path, store_factory):
    plan, approval = _plan(tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 100_000)
    store.mark_call_dispatched(permit.call_id)

    store.complete_call(
        permit.call_id,
        cost_microusd=80_000,
        cost_is_complete=True,
        cost_is_estimate=False,
        artifacts=[
            {
                "relative_path": "artifacts/glyph.png",
                "mime_type": "image/png",
                "sha256": "a" * 64,
                "size_bytes": 123,
                "width": 128,
                "height": 128,
            }
        ],
        result_text="generated",
    )

    assert store.finalize_run(run_id) == RunStatus.COMPLETED.value
    snapshot = store.snapshot(run_id, include_events=True)
    assert snapshot["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    assert snapshot["cost"]["confirmed_microusd"] == 80_000
    assert snapshot["cost"]["reserved_microusd"] == 0
    assert snapshot["artifacts"][0]["sha256"] == "a" * 64
    assert any(event["event_type"] == "call_dispatched" for event in snapshot["events"])


def test_snapshot_uses_one_wal_read_view_during_concurrent_commit(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=1)
    database = tmp_path / "jobs.db"
    reader = store_factory(database)
    run_id = reader.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = reader.pending_operations(run_id)[0]["operation_id"]
    writer = store_factory(database)
    writer_committed = Event()
    writer_errors: list[BaseException] = []
    thread: Thread | None = None

    def trace(statement: str) -> None:
        nonlocal thread
        if statement.startswith("SELECT * FROM operations WHERE run_id"):
            # Start the writer only once the reader reaches the boundary.
            # A readiness deadline before this point counts unrelated reader
            # setup/scheduling time and can expire under parallel test load.
            thread = Thread(target=write_between_snapshot_queries)
            thread.start()
            if not writer_committed.wait(10):
                writer_errors.append(AssertionError("writer did not commit at the read barrier"))

    def write_between_snapshot_queries() -> None:
        try:
            writer.begin_attempt(run_id, operation_id)
        except BaseException as exc:  # pragma: no cover - surfaced below
            writer_errors.append(exc)
        finally:
            writer_committed.set()

    reader._connection.set_trace_callback(trace)
    try:
        snapshot = reader.snapshot(run_id)
    finally:
        reader._connection.set_trace_callback(None)
        if thread is not None:
            thread.join(timeout=10)

    assert thread is not None and not thread.is_alive()
    assert writer_errors == []
    assert snapshot["status"] == RunStatus.APPROVED.value
    assert snapshot["counts"] == {OperationStatus.PENDING.value: 1}
    assert writer.snapshot(run_id)["counts"] == {OperationStatus.RUNNING.value: 1}


def test_recovery_marks_dispatched_call_unknown_and_never_pending(tmp_path, store_factory):
    plan, approval = _plan(tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 100_000)
    store.mark_call_dispatched(permit.call_id)

    recovered = store.recover_inflight(run_id)

    assert recovered["unknown_outcome"] == [operation["operation_id"]]
    assert store.pending_operations(run_id) == []
    snapshot = store.snapshot(run_id)
    assert snapshot["status"] == RunStatus.NEEDS_ATTENTION.value
    assert snapshot["counts"] == {OperationStatus.UNKNOWN_OUTCOME.value: 1}
    assert snapshot["cost"]["exposure_microusd"] == 100_000


def test_prepared_call_recovery_is_safe_and_restores_attempt_allowance(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=1)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    store.prepare_call(attempt["attempt_id"], 0)

    recovered = store.recover_inflight(run_id)

    assert recovered["safe_to_retry"] == [operation["operation_id"]]
    pending = store.pending_operations(run_id)
    assert pending[0]["attempt_count"] == 0
    assert store.snapshot(run_id)["cost"]["reserved_microusd"] == 0

    retried = store.begin_attempt(run_id, operation["operation_id"])
    assert retried["attempt_number"] == 2
    store.prepare_call(retried["attempt_id"], 0)


def test_recovery_restores_running_attempt_that_never_prepared_a_call(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=1)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    first = store.begin_attempt(run_id, operation["operation_id"])

    recovered = store.recover_inflight(run_id)

    assert recovered == {
        "safe_to_retry": [operation["operation_id"]],
        "unknown_outcome": [],
    }
    assert store.pending_operations(run_id)[0]["attempt_count"] == 0
    second = store.begin_attempt(run_id, operation["operation_id"])
    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    store.prepare_call(second["attempt_id"], 0)


def test_run_lease_excludes_other_store_and_gates_recovery(tmp_path):
    plan, approval = _plan(tmp_path, provider="offline", attempts=1)
    database = tmp_path / "jobs.db"
    first_store = SQLiteRunStore(database)
    run_id = first_store.create_run(plan, approval, manifest_root=tmp_path)
    second_store = SQLiteRunStore(database)

    lease = first_store.acquire_run_lease(run_id, "worker-a", ttl_s=30)
    assert lease.owner_id == "worker-a"
    with pytest.raises(RunLeaseError, match="worker-a"):
        second_store.acquire_run_lease(run_id, "worker-b", ttl_s=30)
    with pytest.raises(RunLeaseError, match="requires a current lease token"):
        second_store.recover_inflight(run_id)
    with pytest.raises(RunLeaseError, match="matching current lease token"):
        second_store.recover_inflight(run_id, lease_owner_id="worker-b")

    assert first_store.recover_inflight(
        run_id, lease_owner_id="worker-a", lease=lease,
    ) == {"safe_to_retry": [], "unknown_outcome": []}
    renewed = first_store.heartbeat_run_lease(run_id, "worker-a", ttl_s=30, lease=lease)
    assert renewed.expires_at_ns >= lease.expires_at_ns
    assert first_store.release_run_lease(run_id, "worker-a", lease=lease) is True
    assert first_store.get_run_lease(run_id) is None
    assert second_store.acquire_run_lease(run_id, "worker-b").owner_id == "worker-b"

    first_store.close()
    second_store.close()


def test_unknown_requires_explicit_acknowledgement_before_reroll(tmp_path, store_factory):
    plan, approval = _plan(tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 100_000)
    store.mark_call_dispatched(permit.call_id)
    store.mark_unknown_outcome(permit.call_id, "connection lost after dispatch")

    with pytest.raises(InvalidTransitionError, match="cannot be rerolled"):
        store.queue_reroll(
            run_id,
            [operation["operation_key"]],
            acknowledge_unknown=False,
            reason="operator review",
        )

    lease = store.acquire_run_lease(run_id, "worker-a")
    with pytest.raises(RunLeaseError, match="requires a current lease token"):
        store.queue_reroll(
            run_id,
            [operation["operation_key"]],
            acknowledge_unknown=True,
            reason="competing operator",
        )
    queued = store.queue_reroll(
        run_id,
        [operation["operation_key"]],
        acknowledge_unknown=True,
        reason="operator accepts possible duplicate spend",
        lease_owner_id="worker-a",
        lease=lease,
    )
    store.release_run_lease(run_id, "worker-a", lease=lease)
    assert queued == [operation["operation_key"]]


def test_reroll_changes_only_selected_failed_operation(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2, count=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operations = store.pending_operations(run_id)
    failed = operations[0]
    attempt = store.begin_attempt(run_id, failed["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 0)
    store.mark_call_dispatched(permit.call_id)
    store.complete_call(
        permit.call_id,
        cost_microusd=0,
        cost_is_complete=True,
        cost_is_estimate=False,
        artifacts=[],
        result_text="invalid",
        accepted=False,
        error="missing artifact",
    )

    store.queue_reroll(
        run_id,
        [failed["operation_key"]],
        reason="replace rejected glyph",
    )

    pending = store.pending_operations(run_id)
    assert {item["operation_key"] for item in pending} == {
        failed["operation_key"],
        operations[1]["operation_key"],
    }


def _delivery_record(attempt_id, decision, subjects):
    return {"version": 1, "scope": "artifacts", "attempt_id": attempt_id, "decision": decision,
            "reasons": [] if decision == "accepted" else ["wrong size"], "subjects": subjects,
            "validators": []}


def test_complete_call_records_a_delivery_decision_that_matches_the_settlement(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 0)
    store.mark_call_dispatched(permit.call_id)
    artifact = {"relative_path": "artifacts/glyph.png", "mime_type": "image/png",
                "sha256": "a" * 64, "size_bytes": 123}
    subject = {"path": "artifacts/glyph.png", "sha256": "a" * 64, "size_bytes": 123,
               "published": True}

    def complete(record, accepted):
        store.complete_call(permit.call_id, cost_microusd=0, cost_is_complete=True,
                            cost_is_estimate=False, artifacts=[artifact], result_text="done",
                            accepted=accepted, delivery_acceptance=record)

    attempt_id = attempt["attempt_id"]
    with pytest.raises(ValueError, match="decision"):
        complete(_delivery_record(attempt_id, "rejected", [subject]), True)
    with pytest.raises(ValueError, match="decision"):
        complete(_delivery_record(attempt_id, "needs_review", [subject]), False)
    with pytest.raises(ValueError, match="subject"):
        complete(_delivery_record(attempt_id, "accepted", [dict(subject, sha256="b" * 64)]), True)
    with pytest.raises(ValueError, match="attempt"):
        complete(_delivery_record("other", "accepted", [subject]), True)
    with pytest.raises(ValueError, match="JSON"):
        complete(_delivery_record(attempt_id, "accepted", [subject]) | {"bad": float("nan")}, True)
    complete(_delivery_record(attempt_id, "accepted", [subject]), True)

    events = store.snapshot(run_id, include_events=True)["events"]
    [completed] = [event for event in events if event["event_type"] == "call_completed"]
    assert completed["payload"]["delivery_acceptance"] == _delivery_record(attempt_id, "accepted", [subject])
    assert store.schema_version == 4


def _settle_attempt(store, run_id, operation_id, outcome="rejected"):
    attempt = store.begin_attempt(run_id, operation_id)
    permit = store.prepare_call(attempt["attempt_id"], 0)
    store.mark_call_dispatched(permit.call_id)
    if outcome == "unknown_outcome":
        store.mark_unknown_outcome(permit.call_id, "connection lost after dispatch")
    else:
        store.complete_call(
            permit.call_id,
            cost_microusd=0,
            cost_is_complete=True,
            cost_is_estimate=False,
            artifacts=[],
            result_text="invalid",
            accepted=False,
            error="missing artifact",
        )


def _attempt_reasons(store, run_id):
    return [attempt["reason"] for attempt in store.snapshot(run_id)["attempts"]]


@pytest.mark.parametrize("outcome", ["rejected", "unknown_outcome"])
def test_reroll_reason_is_recorded_on_the_attempt_it_starts(tmp_path, store_factory, outcome):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    _settle_attempt(store, run_id, operation["operation_id"], outcome)
    store.queue_reroll(
        run_id,
        [operation["operation_key"]],
        acknowledge_unknown=outcome == "unknown_outcome",
        reason="replace rejected glyph",
    )

    store.begin_attempt(run_id, operation["operation_id"])

    assert _attempt_reasons(store, run_id) == [None, "replace rejected glyph"]


def test_attempts_without_a_newer_reroll_keep_no_reason(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = store.pending_operations(run_id)[0]["operation_id"]
    first = store.begin_attempt(run_id, operation_id)
    store.prepare_call(first["attempt_id"], 0)
    store.recover_inflight(run_id)
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason="operator review")
    store.begin_attempt(run_id, operation_id)
    store.recover_inflight(run_id)

    # A safe retry is a new attempt, but only the attempt a reroll started
    # carries that reroll's reason.
    store.begin_attempt(run_id, operation_id)

    assert _attempt_reasons(store, run_id) == [None, None, "operator review", None]


def test_attempt_after_two_rerolls_records_the_latest_reason(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=3)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = store.pending_operations(run_id)[0]["operation_id"]
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason="first review")
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason="second review")

    store.begin_attempt(run_id, operation_id)

    assert _attempt_reasons(store, run_id) == [None, "first review", "second review"]


def test_explicit_attempt_reason_takes_precedence_over_a_queued_reroll(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = store.pending_operations(run_id)[0]["operation_id"]
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason="operator review")

    store.begin_attempt(run_id, operation_id, reason="caller reason")

    assert _attempt_reasons(store, run_id) == [None, "caller reason"]


@pytest.mark.parametrize("reason", [None, 7, ["list"]])
def test_reroll_without_a_text_reason_leaves_the_attempt_reason_empty(tmp_path, store_factory, reason):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = store.pending_operations(run_id)[0]["operation_id"]
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason=reason)

    store.begin_attempt(run_id, operation_id)

    assert _attempt_reasons(store, run_id) == [None, None]


def test_corrupt_queued_reroll_event_refuses_the_attempt(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, provider="offline", attempts=2)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation_id = store.pending_operations(run_id)[0]["operation_id"]
    _settle_attempt(store, run_id, operation_id)
    store.queue_reroll(run_id, [operation_id], reason="operator review")
    store._connection.execute(
        "UPDATE events SET payload_json = '{bad JSON' WHERE event_type = 'reroll_queued'"
    )

    with pytest.raises(RunStoreError, match="events.payload_json"):
        store.begin_attempt(run_id, operation_id)

    assert len(store.snapshot(run_id)["attempts"]) == 1
    assert store.pending_operations(run_id)[0]["attempt_count"] == 1


def test_recorded_attempt_reasons_are_never_backfilled(tmp_path):
    plan, approval = _plan(tmp_path, provider="offline", attempts=3)
    database = tmp_path / "jobs.db"
    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, approval, manifest_root=tmp_path)
        operation_id = store.pending_operations(run_id)[0]["operation_id"]
        _settle_attempt(store, run_id, operation_id)
        store.queue_reroll(run_id, [operation_id], reason="earlier reroll")
        _settle_attempt(store, run_id, operation_id)
    # Earlier versions left every attempt reason empty.
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("UPDATE attempts SET reason = NULL")

    with SQLiteRunStore(database) as store:
        assert _attempt_reasons(store, run_id) == [None, None]
        store.queue_reroll(run_id, [operation_id], reason="later reroll")
        store.begin_attempt(run_id, operation_id)
        assert _attempt_reasons(store, run_id) == [None, None, "later reroll"]


_ADOPTED = {"relative_path": "artifacts/glyph_0_/attempt-0001/artifact-01.png",
            "mime_type": "image/png", "sha256": "a" * 64, "size_bytes": 123, "width": 1, "height": 1}


def _adoption_record(attempt_id, decision="accepted", artifact=_ADOPTED):
    return _delivery_record(attempt_id, decision, [
        {"path": artifact["relative_path"], "sha256": artifact["sha256"],
         "size_bytes": artifact["size_bytes"], "published": True}])


def test_adoption_records_published_bytes_and_keeps_the_unknown_call_exposed(tmp_path, store_factory):
    plan, approval = _plan(tmp_path)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt = store.begin_attempt(run_id, operation["operation_id"])
    permit = store.prepare_call(attempt["attempt_id"], 100_000)
    store.mark_call_dispatched(permit.call_id)
    store.mark_unknown_outcome(permit.call_id, "connection lost after dispatch")
    lease = store.acquire_run_lease(run_id, "worker-a")
    key, attempt_id = operation["operation_key"], attempt["attempt_id"]

    def adopt(record=None, artifacts=(_ADOPTED,), **options):
        options = {"reason": "provider shows one charged image", "lease": lease} | options
        return store.adopt_published_attempt(
            run_id, key, artifacts=list(artifacts),
            delivery_acceptance=_adoption_record(attempt_id) if record is None else record, **options)

    before = store.snapshot(run_id, include_events=True)
    with pytest.raises(RunLeaseError, match="requires a current lease token"):
        adopt(lease=None)
    for reason in ("", "  ", None, "x" * 2049):
        with pytest.raises(ValueError, match="reason"):
            adopt(reason=reason)
    with pytest.raises(ValueError, match="artifact"):
        adopt(artifacts=())
    with pytest.raises(ValueError, match="decision"):
        adopt(_adoption_record(attempt_id, "rejected"))
    with pytest.raises(ValueError, match="subject"):
        adopt(_adoption_record(attempt_id, artifact=dict(_ADOPTED, sha256="b" * 64)))
    with pytest.raises(ValueError, match="latest attempt"):
        adopt(_adoption_record("other"))
    with pytest.raises(JobNotFoundError):
        store.adopt_published_attempt(run_id, "missing[0]", reason="r", artifacts=[_ADOPTED],
                                      delivery_acceptance=_adoption_record(attempt_id), lease=lease)
    assert store.snapshot(run_id, include_events=True) == before

    adopted = adopt()

    snapshot = store.snapshot(run_id, include_events=True)
    assert snapshot["counts"] == {OperationStatus.SUCCEEDED.value: 1}
    [row] = snapshot["operations"]
    assert (row["accepted_attempt_id"], row["attempt_count"], row["error"]) == (attempt_id, 1, None)
    assert [item["status"] for item in snapshot["attempts"]] == ["unknown_outcome"]
    [call] = store._call_rows(run_id)
    assert (call["status"], call["confirmed_microusd"], call["exposure_microusd"]) == (
        "unknown_outcome", 0, 100_000)
    assert (snapshot["cost"]["exposure_microusd"], snapshot["cost"]["reserved_microusd"]) == (100_000, 0)
    [artifact] = snapshot["artifacts"]
    assert (artifact["attempt_id"], artifact["accepted"], artifact["sha256"]) == (attempt_id, 1, "a" * 64)
    event = snapshot["events"][-1]
    assert event["event_type"] == "unknown_outcome_adopted" and event["payload"] == adopted
    assert adopted == {"call_id": permit.call_id, "attempt_id": attempt_id,
                       "reason": "provider shows one charged image", "exposure_microusd": 100_000,
                       "artifacts": [{"relative_path": _ADOPTED["relative_path"], "sha256": "a" * 64}],
                       "delivery_acceptance": _adoption_record(attempt_id)}
    assert store.delivery_records(run_id) == {attempt_id: _adoption_record(attempt_id)}
    # The run completes while the call's exposure remains unresolved.
    assert store.finalize_run(run_id, lease=lease) == RunStatus.COMPLETED.value
    assert store.snapshot(run_id)["cost"]["cost_is_complete"] is False

    with pytest.raises(InvalidTransitionError, match="no unknown outcome"):
        adopt()
    assert store.schema_version == 4


@pytest.mark.parametrize("state", ["pending", "running", "rejected"])
def test_adoption_needs_an_operation_with_an_unknown_outcome(tmp_path, store_factory, state):
    plan, approval = _plan(tmp_path, provider="offline")
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    attempt_id = "none"
    if state == "rejected":
        _settle_attempt(store, run_id, operation["operation_id"])
    elif state == "running":
        attempt_id = store.begin_attempt(run_id, operation["operation_id"])["attempt_id"]
        store.mark_call_dispatched(store.prepare_call(attempt_id, 0).call_id)
    before = store.snapshot(run_id, include_events=True)

    with pytest.raises(InvalidTransitionError, match=f"state {state} has no unknown outcome"):
        store.adopt_published_attempt(run_id, operation["operation_key"], reason="review",
                                      artifacts=[_ADOPTED],
                                      delivery_acceptance=_adoption_record(attempt_id))
    assert store.snapshot(run_id, include_events=True) == before


def test_budget_overrun_is_derived_latched_and_blocks_every_admission(tmp_path, store_factory):
    plan, approval = _plan(tmp_path, attempts=1, count=4)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operations = store.pending_operations(run_id)

    first = store.begin_attempt(run_id, operations[0]["operation_id"])
    first_call = store.prepare_call(first["attempt_id"], 100_000)
    store.mark_call_dispatched(first_call.call_id)
    unprepared = store.begin_attempt(run_id, operations[1]["operation_id"])
    prepared_attempt = store.begin_attempt(run_id, operations[2]["operation_id"])
    prepared_call = store.prepare_call(prepared_attempt["attempt_id"], 100_000)

    # Actual durable evidence exceeds the first call's approved ceiling even
    # though the aggregate campaign ceiling has not yet been consumed.
    store.complete_call(
        first_call.call_id,
        cost_microusd=150_000,
        cost_is_complete=True,
        cost_is_estimate=False,
        artifacts=[],
        result_text="over ceiling",
    )
    assert store.get_run(run_id)["status"] == RunStatus.BUDGET_OVERRUN.value

    with pytest.raises(JobBudgetError, match="latched"):
        store.prepare_call(unprepared["attempt_id"], 100_000)
    with pytest.raises(JobBudgetError, match="latched"):
        store.mark_call_dispatched(prepared_call.call_id)
    with pytest.raises(JobBudgetError, match="latched"):
        store.begin_attempt(run_id, operations[3]["operation_id"])

    # Even if a stale writer corrupts the summary status, finalization and the
    # next admission reconstruct the terminal latch from immutable call rows.
    store._connection.execute(
        "UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,)
    )
    assert store.finalize_run(run_id) == RunStatus.BUDGET_OVERRUN.value
    assert store.get_run(run_id)["status"] == RunStatus.BUDGET_OVERRUN.value


# ---------------------------------------------------------------------------
# Integrity repair of succeeded operations
# ---------------------------------------------------------------------------

_RECORDED = [
    {"relative_path": f"artifacts/glyph_0_/attempt-0001/artifact-0{index}.png",
     "mime_type": "image/png", "sha256": letter * 64, "size_bytes": 123, "width": 1, "height": 1}
    for index, letter in ((1, "a"), (2, "b"))
]


def _settle_accepted(store, attempt, artifacts, *, lease, accepted=True, cost=0, ceiling=0):
    permit = store.prepare_call(attempt["attempt_id"], ceiling, lease=lease)
    store.mark_call_dispatched(permit.call_id, lease=lease)
    store.complete_call(permit.call_id, cost_microusd=cost, cost_is_complete=True,
                        cost_is_estimate=False, artifacts=[dict(item) for item in artifacts],
                        result_text="done", accepted=accepted, lease=lease)


def _succeeded(tmp_path, store_factory, *, attempts=2, provider="offline", cost=0):
    """A leased run whose one operation succeeded with two accepted artifacts."""
    plan, approval = _plan(tmp_path, provider=provider, attempts=attempts)
    store = store_factory(tmp_path / "jobs.db")
    run_id = store.create_run(plan, approval, manifest_root=tmp_path)
    operation = store.pending_operations(run_id)[0]
    lease = store.acquire_run_lease(run_id, "worker-a")
    attempt = store.begin_attempt(run_id, operation["operation_id"], lease=lease)
    _settle_accepted(store, attempt, _RECORDED, lease=lease, cost=cost,
                     ceiling=0 if provider == "offline" else 100_000)
    store.finalize_run(run_id, lease=lease)
    return store, run_id, operation, lease, store.snapshot(run_id)["artifacts"]


def _observed(artifact, status, detail="observed"):
    return {"artifact_id": artifact["artifact_id"], "relative_path": artifact["relative_path"],
            "sha256": artifact["sha256"], "status": status, "detail": detail}


def test_integrity_repair_queues_a_new_attempt_and_keeps_the_accepted_one(tmp_path, store_factory):
    store, run_id, operation, lease, artifacts = _succeeded(tmp_path, store_factory)
    accepted_id = artifacts[0]["attempt_id"]
    findings = [_observed(artifacts[0], "changed", "SHA-256 differs from the recorded artifact"),
                _observed(artifacts[1], "verified")]
    before = store.snapshot(run_id)

    queued = store.queue_integrity_repair(run_id, [operation["operation_key"]],
                                          reason="artifact-01.png was overwritten",
                                          findings=findings, lease=lease)

    assert queued == [operation["operation_key"]]
    snapshot = store.snapshot(run_id, include_events=True)
    [row] = snapshot["operations"]
    assert (row["status"], row["accepted_attempt_id"], row["attempt_count"]) == (
        OperationStatus.PENDING.value, accepted_id, 1)
    assert snapshot["status"] == RunStatus.APPROVED.value
    assert snapshot["artifacts"] == before["artifacts"] and snapshot["cost"] == before["cost"]
    event = snapshot["events"][-1]
    assert (event["event_type"], event["operation_id"]) == ("integrity_repair_queued",
                                                            operation["operation_id"])
    assert event["payload"] == {"version": 1, "reason": "artifact-01.png was overwritten",
                                "findings": findings}

    repaired = store.begin_attempt(run_id, operation["operation_id"], lease=lease)

    attempts = store.snapshot(run_id)["attempts"]
    assert [item["reason"] for item in attempts] == [None, "artifact-01.png was overwritten"]
    assert attempts[1]["parent_attempt_id"] == accepted_id
    # The accepted attempt stays named while its replacement runs.
    assert store.snapshot(run_id)["operations"][0]["accepted_attempt_id"] == accepted_id
    replacement = [dict(item, relative_path=item["relative_path"].replace("0001", "0002"))
                   for item in _RECORDED]
    _settle_accepted(store, repaired, replacement, lease=lease)

    after = store.snapshot(run_id)
    assert after["operations"][0]["accepted_attempt_id"] == repaired["attempt_id"]
    assert after["operations"][0]["attempt_count"] == 2
    # Artifact rows are insert-only and recorded accepted flags are unchanged.
    old = [item for item in after["artifacts"] if item["attempt_id"] == accepted_id]
    assert old == before["artifacts"]
    assert [item["accepted"] for item in after["artifacts"]] == [1, 1, 1, 1]
    assert store.finalize_run(run_id, lease=lease) == RunStatus.COMPLETED.value
    assert store.schema_version == 4


def test_a_rejected_repair_attempt_keeps_the_earlier_accepted_attempt(tmp_path, store_factory):
    store, run_id, operation, lease, artifacts = _succeeded(tmp_path, store_factory, attempts=3)
    accepted_id = artifacts[0]["attempt_id"]
    store.queue_integrity_repair(run_id, [operation["operation_id"]], reason="file deleted",
                                 findings=[_observed(item, "missing") for item in artifacts],
                                 lease=lease)
    repaired = store.begin_attempt(run_id, operation["operation_id"], lease=lease)

    _settle_accepted(store, repaired, [], lease=lease, accepted=False)

    [row] = store.snapshot(run_id)["operations"]
    assert row["status"] == OperationStatus.REJECTED.value
    assert row["accepted_attempt_id"] == accepted_id
    # A rejected repair is rerolled like any rejection; its reason is the reroll's.
    store.queue_reroll(run_id, [operation["operation_id"]], reason="second try", lease=lease)
    store.begin_attempt(run_id, operation["operation_id"], lease=lease)
    assert _attempt_reasons(store, run_id) == [None, "file deleted", "second try"]


@pytest.mark.parametrize("statuses,message", [
    (("verified", "verified"), "no missing or changed accepted file"),
    (("changed", "unreadable"), "is unreadable"),
    (("missing", "unsafe"), "is unsafe"),
    (("changed", "not_checked"), "is not_checked"),
])
def test_integrity_repair_needs_a_missing_or_changed_file_and_no_unknown_one(
    tmp_path, store_factory, statuses, message,
):
    store, run_id, operation, lease, artifacts = _succeeded(tmp_path, store_factory)
    before = store.snapshot(run_id, include_events=True)

    with pytest.raises(InvalidTransitionError, match=message):
        store.queue_integrity_repair(
            run_id, [operation["operation_key"]], reason="operator review", lease=lease,
            findings=[_observed(item, status) for item, status in zip(artifacts, statuses)])

    assert store.snapshot(run_id, include_events=True) == before


def test_integrity_repair_refuses_other_states_exhausted_attempts_and_stale_leases(
    tmp_path, store_factory,
):
    store, run_id, operation, lease, artifacts = _succeeded(tmp_path, store_factory, attempts=1)
    key = operation["operation_key"]
    findings = [_observed(item, "changed") for item in artifacts]

    def repair(**options):
        options = {"reason": "operator review", "findings": findings, "lease": lease} | options
        return store.queue_integrity_repair(run_id, [key], **options)

    before = store.snapshot(run_id, include_events=True)
    with pytest.raises(InvalidTransitionError, match="exhausted"):
        repair()
    with pytest.raises(RunLeaseError, match="requires a current lease token"):
        repair(lease=None)
    store.release_run_lease(run_id, "worker-a", lease=lease)
    current = store.acquire_run_lease(run_id, "worker-a")
    with pytest.raises(RunLeaseError, match="superseded"):
        repair()
    store.release_run_lease(run_id, "worker-a", lease=current)
    after = store.snapshot(run_id, include_events=True)
    assert after["operations"] == before["operations"] and after["artifacts"] == before["artifacts"]
    assert "integrity_repair_queued" not in {event["event_type"] for event in after["events"]}

    # Only a succeeded operation is repaired; other states use a reroll.
    other_run = store.create_run(*_plan(tmp_path, provider="offline"), manifest_root=tmp_path,
                                 allow_approval_reuse=True)
    pending = store.pending_operations(other_run)[0]
    with pytest.raises(InvalidTransitionError, match="state pending cannot be repaired"):
        store.queue_integrity_repair(other_run, [pending["operation_key"]], reason="review",
                                     findings=[])
    _settle_attempt(store, other_run, pending["operation_id"])
    with pytest.raises(InvalidTransitionError, match="state rejected cannot be repaired"):
        store.queue_integrity_repair(other_run, [pending["operation_key"]], reason="review",
                                     findings=[])
    with pytest.raises(JobNotFoundError):
        store.queue_integrity_repair(other_run, ["missing[0]"], reason="review", findings=[])


def test_integrity_repair_refuses_after_a_budget_overrun(tmp_path, store_factory):
    store, run_id, operation, lease, artifacts = _succeeded(
        tmp_path, store_factory, provider="openai_image", cost=150_000)
    assert store.get_run(run_id)["status"] == RunStatus.BUDGET_OVERRUN.value
    before = store.snapshot(run_id, include_events=True)

    with pytest.raises(JobBudgetError, match="latched"):
        store.queue_integrity_repair(run_id, [operation["operation_key"]], reason="review",
                                     findings=[_observed(item, "changed") for item in artifacts],
                                     lease=lease)

    assert store.snapshot(run_id, include_events=True) == before


def test_integrity_repair_findings_must_observe_each_accepted_file_as_recorded(
    tmp_path, store_factory,
):
    store, run_id, operation, lease, artifacts = _succeeded(tmp_path, store_factory)
    first, second = (_observed(item, "changed") for item in artifacts)
    before = store.snapshot(run_id, include_events=True)

    for findings, problem in (
        ([first], "every accepted artifact"),
        ([first, second, first], "more than once"),
        ([first, dict(second, sha256="c" * 64)], "every accepted artifact"),
        ([first, dict(second, relative_path="artifacts/other.png")], "every accepted artifact"),
        ([first, second, dict(first, artifact_id="f" * 32)], "not an accepted artifact"),
        ([first, dict(second, status="broken")], "status"),
        ([first, dict(second, detail=7)], "detail"),
        ([first, {key: second[key] for key in ("artifact_id", "status")}], "keys"),
        ([first, dict(second, extra=True)], "keys"),
        ("changed", "list"),
    ):
        with pytest.raises(ValueError, match=problem):
            store.queue_integrity_repair(run_id, [operation["operation_key"]], reason="review",
                                         findings=findings, lease=lease)
    for reason in ("", "  ", None, "x" * 2049):
        with pytest.raises(ValueError, match="reason"):
            store.queue_integrity_repair(run_id, [operation["operation_key"]], reason=reason,
                                         findings=[first, second], lease=lease)
    with pytest.raises(ValueError, match="operation key"):
        store.queue_integrity_repair(run_id, [], reason="review", findings=[], lease=lease)
    assert store.snapshot(run_id, include_events=True) == before


def test_integrity_repair_reopens_a_version_4_database_without_migration(tmp_path, monkeypatch):
    plan, approval = _plan(tmp_path, provider="offline")
    database = tmp_path / "jobs.db"

    def schema():
        with closing(sqlite3.connect(database)) as db:
            return (db.execute("PRAGMA user_version").fetchone()[0],
                    sorted(db.execute("SELECT type, name, sql FROM sqlite_master")))

    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, approval, manifest_root=tmp_path)
        operation_id = store.pending_operations(run_id)[0]["operation_id"]
        lease = store.acquire_run_lease(run_id, "worker-a")
        attempt = store.begin_attempt(run_id, operation_id, lease=lease)
        _settle_accepted(store, attempt, _RECORDED[:1], lease=lease)
        [artifact] = store.snapshot(run_id)["artifacts"]
        created = schema()
        store.queue_integrity_repair(run_id, [operation_id], reason="file deleted",
                                     findings=[_observed(artifact, "missing")], lease=lease)
        store.release_run_lease(run_id, "worker-a", lease=lease)

    def no_migration(*_args):
        raise AssertionError("a version 4 database was migrated")

    monkeypatch.setattr(SQLiteRunStore, "_migrate_schema", no_migration)
    assert schema() == created and created[0] == 4
    with SQLiteRunStore(database) as store:
        lease = store.acquire_run_lease(run_id, "worker-b")
        store.begin_attempt(run_id, operation_id, lease=lease)
        assert _attempt_reasons(store, run_id) == [None, "file deleted"]
    with SQLiteRunStore(database, read_only=True) as store:
        assert store.snapshot(run_id)["version"] == 4
    assert schema() == created
