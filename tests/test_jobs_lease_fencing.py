"""Lease takeover must fence old workers at every durable mutation boundary."""

from __future__ import annotations

from contextlib import closing
import asyncio
import base64
import sqlite3
from dataclasses import replace
from threading import Event, Thread

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.providers import ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import MAX_SQLITE_INTEGER, RunLeaseError, RunStoreError, SQLiteRunStore
from smythe.provider import Artifact, CompletionResult, Provider


class Clock:
    now = 1_000_000_000_000

    def __call__(self):
        return self.now

    def expire(self):
        self.now += 31_000_000_000


def plan_for(root, *, attachment=False):
    operation = {"key": "item", "prompt": "one pixel", "profile": "offline"}
    if attachment:
        (root / "input.png").write_bytes(PNG)
        operation["attachments"] = ["input.png"]
    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "lease-fencing",
        "profiles": [{"name": "offline", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [operation],
        "execution": {"max_concurrency": 1, "max_attempts": 2,
                      "max_budget_usd": "0", "output_directory": "outputs"},
    })
    return preflight_job(manifest, manifest_root=root)


@pytest.fixture
def journal(tmp_path):
    clock = Clock()
    plan = plan_for(tmp_path)
    with SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as first:
        first.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="run")
        with SQLiteRunStore(first.path, clock_ns=clock) as second:
            yield first, second, clock, plan.operations[0].operation_id


def dump(store):
    return list(store._connection.iterdump())


def complete(store, call_id, lease):
    store.complete_call(call_id, cost_microusd=0, cost_is_complete=True,
                        cost_is_estimate=False, artifacts=[{
                            "relative_path": f"artifacts/{call_id}.png", "mime_type": "image/png",
                            "sha256": "a" * 64, "size_bytes": 70, "width": 1, "height": 1,
                        }], result_text="done", lease=lease)


@pytest.mark.parametrize("omit_token", [False, True], ids=["old-token", "omitted-token"])
@pytest.mark.parametrize("action", ["begin", "prepare", "dispatch", "complete", "fail", "unknown",
                                    "finalize", "recover", "reroll", "heartbeat", "release"])
def test_takeover_fences_every_old_worker_write_without_side_effects(journal, action, omit_token):
    first, second, clock, operation_id = journal
    lease = first.acquire_run_lease("run", "worker-a")
    attempt = permit = None
    if action in {"prepare", "dispatch", "complete", "fail", "unknown", "reroll"}:
        attempt = first.begin_attempt("run", operation_id, lease=lease)
    if action in {"dispatch", "complete", "fail", "unknown", "reroll"}:
        permit = first.prepare_call(attempt["attempt_id"], 0, lease=lease)
    if action in {"complete", "unknown", "reroll"}:
        first.mark_call_dispatched(permit.call_id, lease=lease)
    if action == "reroll":
        first.mark_unknown_outcome(permit.call_id, "interrupted", lease=lease)
    clock.expire()
    replacement = second.acquire_run_lease("run", "worker-b")
    token = None if omit_token else lease
    actions = {
        "begin": lambda: first.begin_attempt("run", operation_id, lease=token),
        "prepare": lambda: first.prepare_call(attempt["attempt_id"], 0, lease=token),
        "dispatch": lambda: first.mark_call_dispatched(permit.call_id, lease=token),
        "complete": lambda: complete(first, permit.call_id, token),
        "fail": lambda: first.fail_pre_dispatch(permit.call_id, "failed", lease=token),
        "unknown": lambda: first.mark_unknown_outcome(permit.call_id, "unknown", lease=token),
        "finalize": lambda: first.finalize_run("run", lease=token),
        "recover": lambda: first.recover_inflight("run", lease=token),
        "reroll": lambda: first.queue_reroll("run", [operation_id], reason="retry",
                                             acknowledge_unknown=True, lease=token),
        "heartbeat": lambda: first.heartbeat_run_lease("run", "worker-a", lease=token),
        "release": lambda: first.release_run_lease("run", "worker-a", lease=token),
    }
    before = dump(second)
    with pytest.raises(RunLeaseError):
        actions[action]()
    assert dump(second) == before
    assert second.get_run_lease("run") == replacement


@pytest.mark.parametrize("prepared", [False, True])
def test_replacement_owner_must_recover_not_adopt_old_attempt(journal, prepared):
    first, second, clock, operation_id = journal
    old = first.acquire_run_lease("run", "same-owner")
    attempt = first.begin_attempt("run", operation_id, lease=old)
    permit = first.prepare_call(attempt["attempt_id"], 0, lease=old) if prepared else None
    clock.expire()
    new = second.acquire_run_lease("run", "same-owner")
    assert new.epoch == old.epoch + 1
    before = dump(second)
    with pytest.raises(RunLeaseError, match="earlier lease"):
        if permit:
            second.mark_call_dispatched(permit.call_id, lease=new)
        else:
            second.prepare_call(attempt["attempt_id"], 0, lease=new)
    assert dump(second) == before
    recovered = second.recover_inflight("run", lease=new)
    assert recovered == {"safe_to_retry": [operation_id], "unknown_outcome": []}
    retried = second.begin_attempt("run", operation_id, lease=new)
    assert retried["attempt_number"] == 2
    previous, current = second.snapshot("run")["attempts"]
    assert (current["lease_owner_id"], current["lease_epoch"]) == (new.owner_id, new.epoch)
    assert (previous["lease_owner_id"], previous["lease_epoch"]) == (old.owner_id, old.epoch)


def test_late_response_never_changes_unknown_or_replacement_acceptance(journal):
    first, second, clock, operation_id = journal
    old = first.acquire_run_lease("run", "worker-a")
    attempt = first.begin_attempt("run", operation_id, lease=old)
    permit = first.prepare_call(attempt["attempt_id"], 0, lease=old)
    first.mark_call_dispatched(permit.call_id, lease=old)
    clock.expire()
    new = second.acquire_run_lease("run", "worker-b")
    second.recover_inflight("run", lease=new)
    before = dump(second)
    for token in (old, new, None):
        with pytest.raises(RunLeaseError):
            complete(first, permit.call_id, token)
        assert dump(second) == before
    second.queue_reroll("run", [operation_id], reason="explicit duplicate acknowledgement",
                        acknowledge_unknown=True, lease=new)
    retried = second.begin_attempt("run", operation_id, lease=new)
    assert second.snapshot("run")["attempts"][-1]["parent_attempt_id"] == attempt["attempt_id"]
    new_call = second.prepare_call(retried["attempt_id"], 0, lease=new)
    second.mark_call_dispatched(new_call.call_id, lease=new)
    complete(second, new_call.call_id, new)
    second.finalize_run("run", lease=new)
    before = dump(second)
    with pytest.raises(RunLeaseError):
        complete(first, permit.call_id, old)
    assert dump(second) == before
    snapshot = second.snapshot("run")
    assert snapshot["operations"][0]["accepted_attempt_id"] == retried["attempt_id"]
    assert {row["call_id"]: row["status"] for row in second._call_rows("run")} == {
        permit.call_id: "unknown_outcome", new_call.call_id: "succeeded",
    }


def test_manual_use_only_before_first_lease_and_no_aba_release(journal):
    first, second, clock, operation_id = journal
    assert first.finalize_run("run") == "running"
    old = first.acquire_run_lease("run", "same-owner")
    assert first.release_run_lease("run", old.owner_id, lease=old)
    assert not first.release_run_lease("run", old.owner_id, lease=old)
    with pytest.raises(RunLeaseError):
        first.begin_attempt("run", operation_id)
    new = second.acquire_run_lease("run", "same-owner")
    assert new.epoch == old.epoch + 1
    for action in (lambda: first.heartbeat_run_lease("run", old.owner_id, lease=old),
                   lambda: first.release_run_lease("run", old.owner_id, lease=old)):
        with pytest.raises(RunLeaseError, match="superseded"):
            action()
    clock.expire()
    with pytest.raises(RunLeaseError, match="expired"):
        second.finalize_run("run", lease=new)
    with pytest.raises(RunLeaseError):
        second.finalize_run("run")


@pytest.mark.parametrize("epoch", [True, 1.0, -1, 0, MAX_SQLITE_INTEGER + 1])
def test_token_epoch_must_be_strict_bounded_integer(journal, epoch):
    first, _, _, operation_id = journal
    lease = first.acquire_run_lease("run", "owner")
    before = dump(first)
    with pytest.raises(RunLeaseError):
        first.begin_attempt("run", operation_id, lease=replace(lease, epoch=epoch))
    assert dump(first) == before


def test_epoch_exhaustion_fails_before_writes(journal):
    first, _, _, _ = journal
    first._connection.execute("UPDATE runs SET lease_epoch=?", (MAX_SQLITE_INTEGER,))
    before = dump(first)
    with pytest.raises(RunLeaseError, match="exhausted"):
        first.acquire_run_lease("run", "owner")
    assert dump(first) == before


@pytest.mark.parametrize("table,column", [("runs", "lease_epoch"), ("run_leases", "epoch")])
def test_persisted_epoch_must_be_integer_before_renewal(journal, table, column):
    first, _, _, _ = journal
    lease = first.acquire_run_lease("run", "owner")
    first._connection.execute(f"UPDATE {table} SET {column}=1.5")
    before = dump(first)
    with pytest.raises(TypeError, match="epoch must be an integer"):
        first.heartbeat_run_lease("run", "owner", lease=lease)
    assert dump(first) == before


def test_heartbeat_checks_clock_after_obtaining_write_lock(journal):
    first, second, clock, _ = journal
    lease = first.acquire_run_lease("run", "owner")
    started = Event()
    errors = []
    second._connection.execute("BEGIN IMMEDIATE")
    first._connection.set_trace_callback(lambda sql: started.set() if sql == "BEGIN IMMEDIATE" else None)

    def renew():
        try:
            first.heartbeat_run_lease("run", "owner", lease=lease)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=renew)
    thread.start()
    try:
        assert started.wait(5)
        clock.expire()
    finally:
        second._connection.rollback()
        thread.join(5)
        first._connection.set_trace_callback(None)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RunLeaseError)
    assert first.get_run_lease("run") == lease


def make_legacy(path, root, clock, *, leased=False, prepared=False):
    with SQLiteRunStore(path, clock_ns=clock) as store:
        plan = plan_for(root)
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id="run")
        if prepared:
            attempt = store.begin_attempt("run", plan.operations[0].operation_id)
            store.prepare_call(attempt["attempt_id"], 0)
        if leased:
            store.acquire_run_lease("run", "legacy-owner")
    # Reconstruct the historical v2 shape, including its lack of fencing columns.
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE run_controls")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_namespace")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_owner_id")
        db.execute("ALTER TABLE runs DROP COLUMN lease_epoch")
        db.execute("ALTER TABLE attempts DROP COLUMN lease_owner_id")
        db.execute("ALTER TABLE attempts DROP COLUMN lease_epoch")
        db.execute("ALTER TABLE run_leases DROP COLUMN epoch")
        db.execute("PRAGMA user_version=2")


def test_read_only_v2_inspection_does_not_migrate_bytes(tmp_path):
    path, clock = tmp_path / "legacy.db", Clock()
    make_legacy(path, tmp_path, clock, leased=True)
    before = path.read_bytes()
    with SQLiteRunStore(path, read_only=True) as reader:
        for method in (reader.snapshot, reader.inspection_snapshot):
            snapshot = method("run")
            assert snapshot["version"] == 2
            assert snapshot["lease_fencing_supported"] is False
        assert reader.get_run_lease("run").epoch == 0
    assert path.read_bytes() == before


def test_open_v2_reader_reports_migration_in_its_next_snapshot(tmp_path):
    path, clock = tmp_path / "legacy.db", Clock()
    make_legacy(path, tmp_path, clock)
    with SQLiteRunStore(path, read_only=True) as reader:
        assert reader.snapshot("run")["version"] == 2
        with SQLiteRunStore(path, clock_ns=clock):
            pass
        for method in (reader.snapshot, reader.inspection_snapshot):
            current = method("run")
            assert current["version"] == 4
            assert current["lease_fencing_supported"] is True


def test_v2_migration_rejects_live_legacy_lease_then_preserves_history(tmp_path):
    path, clock = tmp_path / "legacy.db", Clock()
    make_legacy(path, tmp_path, clock, leased=True, prepared=True)
    with closing(sqlite3.connect(path)) as db, db:
        before = list(db.iterdump())
    with pytest.raises(RunLeaseError, match="Stop legacy"):
        SQLiteRunStore(path, clock_ns=clock)
    with closing(sqlite3.connect(path)) as db, db:
        assert list(db.iterdump()) == before
    clock.expire()
    with SQLiteRunStore(path, clock_ns=clock) as store:
        assert store.snapshot("run")["lease_fencing_supported"] is True
        assert store.get_run("run")["lease_epoch"] == 1
        with pytest.raises(RunLeaseError):
            store.recover_inflight("run")
        new = store.acquire_run_lease("run", "new-owner")
        assert new.epoch == 2
        previous = store.snapshot("run")["attempts"][0]
        assert previous["lease_owner_id"] is None and previous["lease_epoch"] == 0
        with pytest.raises(RunLeaseError, match="earlier lease"):
            store.mark_call_dispatched(store._call_rows("run")[0]["call_id"], lease=new)
        assert store.recover_inflight("run", lease=new) == {
            "safe_to_retry": [previous["operation_id"]], "unknown_outcome": [],
        }


def test_migration_retains_ever_leased_fence_after_legacy_release(tmp_path):
    path, clock = tmp_path / "legacy.db", Clock()
    make_legacy(path, tmp_path, clock, leased=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DELETE FROM run_leases")  # Legacy release leaves its acquisition event.
    with SQLiteRunStore(path, clock_ns=clock) as store:
        assert store.get_run("run")["lease_epoch"] == 1
        with pytest.raises(RunLeaseError):
            store.finalize_run("run")
        assert store.acquire_run_lease("run", "new-owner").epoch == 2


def test_new_writer_rejects_schema_downgrade_and_mixed_reopen(journal):
    first, second, _, operation_id = journal
    second._connection.execute("PRAGMA user_version=2")
    before = dump(first)
    with pytest.raises(RunStoreError, match="version"):
        first.begin_attempt("run", operation_id)
    assert dump(first) == before
    with pytest.raises(RunStoreError, match="Mixed"):
        SQLiteRunStore(first.path)


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==")


class FixedPool(ProviderPool):
    def __init__(self, provider):
        self.provider = provider

    def preflight(self, operation):
        return None

    def get(self, operation):
        return self.provider


class PixelProvider(Provider):
    def __init__(self, *, block=False):
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def complete(self, system, prompt, model):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return CompletionResult(text="pixel", artifacts=[Artifact(PNG, "image/png")])

    async def chat(self, system, messages, model, tools=None):
        assert not tools
        return await self.complete(system, messages[0].content, model)


@pytest.mark.parametrize("phase", ["attachment", "provider"])
def test_runner_takeover_fences_dispatch_and_late_response(tmp_path, monkeypatch, phase):
    async def scenario():
        clock = Clock()
        plan = plan_for(tmp_path, attachment=phase == "attachment")
        with SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as first, \
                SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as second:
            provider = PixelProvider(block=phase == "provider")
            runner = JobRunner(first, provider_pool=FixedPool(provider), lease_heartbeat_s=10)
            loading, release_loading = Event(), Event()
            original_load = runner._load_attachments

            def load(*args):
                loading.set()
                assert release_loading.wait(10)
                return original_load(*args)

            if phase == "attachment":
                monkeypatch.setattr(runner, "_load_attachments", load)
            task = asyncio.create_task(runner.start(plan, make_approval(plan),
                                                   manifest_root=tmp_path, run_id="run"))
            try:
                if phase == "attachment":
                    assert await asyncio.to_thread(loading.wait, 10)
                else:
                    await asyncio.wait_for(provider.entered.wait(), 10)
                old = first.get_run_lease("run")
                clock.expire()
                new = second.acquire_run_lease("run", "replacement")
                if phase == "provider":
                    second.recover_inflight("run", lease=new)
                before = dump(second)
                release_loading.set()
                provider.release.set()
                with pytest.raises(RunLeaseError):
                    await asyncio.wait_for(task, 10)
                assert dump(second) == before
                assert provider.calls == (0 if phase == "attachment" else 1)
                assert second.get_run_lease("run") == new
                recovered = second.recover_inflight("run", lease=new)
                if phase == "attachment":
                    assert len(recovered["safe_to_retry"]) == 1
                second.release_run_lease("run", new.owner_id, lease=new)
                fresh = PixelProvider()
                resumed = await JobRunner(second, provider_pool=FixedPool(fresh)).resume("run")
                assert fresh.calls == (1 if phase == "attachment" else 0)
                assert resumed["counts"] == ({"succeeded": 1} if phase == "attachment" else {"unknown_outcome": 1})
                assert resumed["attempts"][0]["lease_epoch"] == old.epoch
            finally:
                release_loading.set()
                provider.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
