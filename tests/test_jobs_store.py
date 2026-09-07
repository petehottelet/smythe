"""Durability and ambiguity tests for the Jobs v1 SQLite dispatch journal."""

from __future__ import annotations

from contextlib import ExitStack
from threading import Event, Thread

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.store import (
    InvalidTransitionError,
    JobBudgetError,
    OperationStatus,
    RunLeaseError,
    RunStatus,
    SQLiteRunStore,
)


@pytest.fixture
def store_factory():
    """Close test-owned stores after all assertions and workers finish."""
    with ExitStack() as stack:
        def create(path):
            return stack.enter_context(SQLiteRunStore(path))

        yield create


def _plan(tmp_path, *, provider="openai_image", attempts=2, count=1):
    ceiling = "0" if provider == "offline" else "0.10"
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
