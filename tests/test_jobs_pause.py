"""Durable pause admission, graceful draining, and generation-bound resume."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from decimal import Decimal

import pytest

from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import (
    MAX_SQLITE_INTEGER, RunLeaseError, RunPauseRequested, RunStoreError, SQLiteRunStore,
)
from smythe.provider import Artifact, CompletionResult
from tests.test_jobs_lease_fencing import Clock, FixedPool, PixelProvider, PNG, dump


def plan_for(root, *, count=3, priced=False, attempts=2):
    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "durable-pause",
        "profiles": [{"name": "fixture", "provider": "openai_image" if priced else "offline",
                      "model": "gpt-image-1" if priced else "offline-image",
                      "max_cost_per_call_usd": "0.10" if priced else "0"}],
        "operations": [{"key": "item", "count": count, "prompt": "one pixel", "profile": "fixture"}],
        "execution": {"max_concurrency": 2, "max_attempts": attempts,
                      "max_budget_usd": str(Decimal("0.10") * count * attempts) if priced else "0",
                      "output_directory": "outputs"},
    })
    return preflight_job(manifest, manifest_root=root)


@pytest.fixture
def journal(tmp_path):
    clock = Clock()
    plan = plan_for(tmp_path, priced=True)
    with SQLiteRunStore(tmp_path / "jobs.db", clock_ns=clock) as first:
        first.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="run")
        with SQLiteRunStore(first.path, clock_ns=clock) as second:
            yield first, second, clock, plan


def finish(store, call_id, lease, *, cost=80_000):
    store.complete_call(call_id, cost_microusd=cost, cost_is_complete=True,
                        cost_is_estimate=False, artifacts=[{
                            "relative_path": f"artifacts/{call_id}.png", "mime_type": "image/png",
                            "sha256": "a" * 64, "size_bytes": 70,
                        }], result_text="done", lease=lease)


@pytest.mark.parametrize("stage", ["begin", "prepare", "dispatch"])
def test_pause_refuses_new_admission_and_preserves_safe_retry_allowance(journal, stage):
    first, second, _, plan = journal
    operation_id = plan.operations[0].operation_id
    lease = first.acquire_run_lease("run", "worker")
    attempt = first.begin_attempt("run", operation_id, lease=lease) if stage != "begin" else None
    permit = first.prepare_call(attempt["attempt_id"], 100_000, lease=lease) if stage == "dispatch" else None
    requested = second.request_pause("run", reason="operator pause")
    before = dump(first)
    with pytest.raises(RunPauseRequested):
        if stage == "begin":
            first.begin_attempt("run", operation_id, lease=lease)
        elif stage == "prepare":
            first.prepare_call(attempt["attempt_id"], 100_000, lease=lease)
        else:
            first.mark_call_dispatched(permit.call_id, lease=lease)
    assert dump(first) == before
    assert first.get_run_lease("run") == lease
    assert first.finalize_run("run", lease=lease) == "paused"
    paused = first.snapshot("run")
    assert paused["counts"] == {"pending": 3}
    assert paused["cost"]["reserved_microusd"] == 0
    assert all(item["attempt_count"] == 0 for item in paused["operations"])
    assert paused["control"]["drained_at_ns"] is not None
    first.release_run_lease("run", lease.owner_id, lease=lease)
    new = second.acquire_run_lease("run", "replacement", pause_generation=requested["pause_generation"])
    retried = second.begin_attempt("run", operation_id, lease=new)
    assert retried["attempt_number"] == (1 if stage == "begin" else 2)


def test_pause_does_not_release_reservations_or_classify_active_calls_until_drained(journal):
    first, second, _, plan = journal
    lease = first.acquire_run_lease("run", "worker")
    attempts = [first.begin_attempt("run", op.operation_id, lease=lease) for op in plan.operations]
    active = first.prepare_call(attempts[0]["attempt_id"], 100_000, lease=lease)
    prepared = first.prepare_call(attempts[1]["attempt_id"], 100_000, lease=lease)
    first.mark_call_dispatched(active.call_id, lease=lease)
    second.request_pause("run")
    assert first.finalize_run("run", lease=lease) == "running"
    draining = first.snapshot("run")
    assert draining["cost"]["reserved_microusd"] == 200_000
    assert draining["control"]["drained_at_ns"] is None
    assert {row["call_id"]: row["status"] for row in first._call_rows("run")} == {
        active.call_id: "dispatched", prepared.call_id: "prepared",
    }
    finish(first, active.call_id, lease)
    assert first.finalize_run("run", lease=lease) == "paused"
    paused = first.snapshot("run")
    assert paused["counts"] == {"succeeded": 1, "pending": 2}
    assert paused["cost"]["confirmed_microusd"] == 80_000
    assert paused["cost"]["exposure_microusd"] == paused["cost"]["reserved_microusd"] == 0
    assert {row["status"] for row in first._call_rows("run")} == {"succeeded", "failed"}


@pytest.mark.parametrize("outcome,status", [("unknown", "needs_attention"), ("overrun", "budget_overrun")])
def test_pause_preserves_unknown_and_budget_precedence(journal, outcome, status):
    first, second, _, plan = journal
    lease = first.acquire_run_lease("run", "worker")
    attempt = first.begin_attempt("run", plan.operations[0].operation_id, lease=lease)
    permit = first.prepare_call(attempt["attempt_id"], 100_000, lease=lease)
    first.mark_call_dispatched(permit.call_id, lease=lease)
    second.request_pause("run")
    if outcome == "unknown":
        first.mark_unknown_outcome(permit.call_id, "provider connection lost", lease=lease)
    else:
        finish(first, permit.call_id, lease, cost=150_000)
    cost = first.snapshot("run")["cost"]
    assert first.finalize_run("run", lease=lease) == status
    assert first.snapshot("run")["cost"] == cost
    assert first.get_control("run")["pause_requested"]


def test_newer_pause_survives_captured_resume_and_same_owner_renewal(journal):
    first, second, _, plan = journal
    lease = first.acquire_run_lease("run", "worker")
    old_intent = second.request_pause("run")["pause_generation"]
    newest = second.request_pause("run")["pause_generation"]
    renewed = first.acquire_run_lease("run", "worker", pause_generation=old_intent)
    assert renewed.epoch == lease.epoch
    assert first.get_control("run")["pause_requested"]
    assert first.get_control("run")["pause_generation"] == newest
    renewed = first.acquire_run_lease("run", "worker")
    assert renewed.epoch == lease.epoch and first.get_control("run")["pause_requested"]
    before = dump(first)
    with pytest.raises(RunLeaseError):
        second.acquire_run_lease("run", "other-worker", pause_generation=newest)
    assert dump(first) == before
    first.acquire_run_lease("run", "worker", pause_generation=newest)
    assert not first.get_control("run")["pause_requested"]
    first.begin_attempt("run", plan.operations[0].operation_id, lease=lease)


def test_dormant_pause_is_intent_until_worker_acknowledges_it(journal):
    first, second, _, _ = journal
    control = second.request_pause("run")
    assert control["pause_requested"] and control["drained_at_ns"] is None
    assert first.get_run("run")["status"] == "approved"
    with SQLiteRunStore(first.path, read_only=True) as reader:
        before = dump(first)
        assert reader.inspection_snapshot("run")["control"] == control
        with pytest.raises(RunStoreError, match="read-only"):
            reader.request_pause("run")
        assert dump(first) == before


@pytest.mark.parametrize("value", [True, -1, 1.5, MAX_SQLITE_INTEGER + 1])
def test_invalid_resume_generation_cannot_mutate_a_lease_or_control(journal, value):
    first, _, _, _ = journal
    before = dump(first)
    with pytest.raises((TypeError, ValueError)):
        first.acquire_run_lease("run", "worker", pause_generation=value)
    assert dump(first) == before


def test_control_corruption_and_generation_overflow_fail_closed(journal):
    first, _, _, plan = journal
    first._connection.execute("UPDATE run_controls SET pause_generation=?", (MAX_SQLITE_INTEGER,))
    before = dump(first)
    with pytest.raises(RunStoreError, match="exhausted"):
        first.request_pause("run")
    assert dump(first) == before
    first._connection.execute("DELETE FROM run_controls")
    before = dump(first)
    with pytest.raises(RunStoreError, match="control row"):
        first.begin_attempt("run", plan.operations[0].operation_id)
    assert dump(first) == before


class DrainingProvider(PixelProvider):
    def __init__(self):
        super().__init__(block=True)
        self.both_entered = asyncio.Event()

    async def complete(self, system, prompt, model):
        self.calls += 1
        if self.calls == 2:
            self.both_entered.set()
        await self.release.wait()
        return CompletionResult(text="pixel", artifacts=[Artifact(PNG, "image/png")])


@pytest.mark.parametrize("count,status", [(2, "completed"), (3, "paused")])
def test_runner_drains_admitted_calls_and_resume_preserves_accepted_artifacts(tmp_path, count, status):
    async def scenario():
        plan = plan_for(tmp_path, count=count)
        with SQLiteRunStore(tmp_path / "jobs.db") as first, SQLiteRunStore(tmp_path / "jobs.db") as second:
            provider = DrainingProvider()
            runner = JobRunner(first, provider_pool=FixedPool(provider))
            task = asyncio.create_task(runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="run"))
            try:
                await asyncio.wait_for(provider.both_entered.wait(), 30)
                lease = second.get_run_lease("run")
                second.request_pause("run")
                assert second.get_run_lease("run") == lease
                assert all(row["status"] == "dispatched" for row in second._call_rows("run"))
                assert not task.done()
                provider.release.set()
                result = await asyncio.wait_for(task, 30)
                assert result["status"] == status
                assert provider.calls == 2
                assert result["control"]["drained_at_ns"] is not None
                assert not second.get_run_lease("run")
                accepted = {row["operation_id"]: row["accepted_attempt_id"] for row in result["operations"]
                            if row["accepted_attempt_id"]}
                artifact_root = tmp_path / "outputs" / result["artifact_directory"]
                original_bytes = {row["relative_path"]: (artifact_root / row["relative_path"]).read_bytes()
                                  for row in result["artifacts"]}
                resumed = await runner.resume("run")
                assert resumed["status"] == "completed" and provider.calls == count
                assert not resumed["control"]["pause_requested"]
                assert {row["operation_id"]: row["accepted_attempt_id"] for row in resumed["operations"]
                        if row["operation_id"] in accepted} == accepted
                for path, data in original_bytes.items():
                    assert (artifact_root / path).read_bytes() == data
                await runner.resume("run")
                assert provider.calls == count
            finally:
                provider.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_default_resume_captures_intent_before_preflight_and_newer_stop_survives(tmp_path, monkeypatch):
    async def scenario():
        plan = plan_for(tmp_path, count=1)
        with SQLiteRunStore(tmp_path / "jobs.db") as store:
            store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="run")
            first_generation = store.request_pause("run")["pause_generation"]
            provider = PixelProvider()
            runner = JobRunner(store, provider_pool=FixedPool(provider))
            original = runner._validate_dispatch_inputs

            def newer_request(*args):
                original(*args)
                store.request_pause("run", reason="a newer operator stop")

            monkeypatch.setattr(runner, "_validate_dispatch_inputs", newer_request)
            paused = await runner.resume("run")
            assert provider.calls == 0 and paused["status"] == "paused"
            assert paused["control"]["pause_generation"] == first_generation + 1
            fresh = JobRunner(store, provider_pool=FixedPool(provider))
            assert (await fresh.resume("run", clear_pause=False))["status"] == "paused"
            assert provider.calls == 0
            assert (await fresh.resume("run"))["status"] == "completed"
            assert provider.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["begin_attempt", "prepare_call"])
def test_runner_pause_between_local_transactions_restores_attempt_allowance(tmp_path, monkeypatch, boundary):
    async def scenario():
        plan = plan_for(tmp_path, count=1, attempts=1)
        with SQLiteRunStore(tmp_path / "jobs.db") as store:
            provider = PixelProvider()
            runner = JobRunner(store, provider_pool=FixedPool(provider))
            original = getattr(store, boundary)
            requested = False

            def pause_after_commit(*args, **kwargs):
                nonlocal requested
                result = original(*args, **kwargs)
                if not requested:
                    requested = True
                    store.request_pause("run")
                return result

            monkeypatch.setattr(store, boundary, pause_after_commit)
            paused = await runner.start(plan, make_approval(plan), manifest_root=tmp_path, run_id="run")
            assert paused["status"] == "paused" and provider.calls == 0
            assert paused["operations"][0]["attempt_count"] == 0
            assert paused["cost"]["reserved_microusd"] == 0
            assert len(paused["attempts"]) == 1 and paused["attempts"][0]["status"] == "failed"
            resumed = await runner.resume("run")
            assert resumed["status"] == "completed" and provider.calls == 1
            assert resumed["attempts"][-1]["parent_attempt_id"] == paused["attempts"][0]["attempt_id"]
            assert resumed["attempts"][-1]["attempt_number"] == 2
            assert resumed["operations"][0]["attempt_count"] == 1

    asyncio.run(scenario())


def test_v3_reader_capability_follows_upgrade_and_live_v3_lease_blocks_it(tmp_path):
    path, clock = tmp_path / "legacy-v3.db", Clock()
    with SQLiteRunStore(path, clock_ns=clock) as store:
        plan = plan_for(tmp_path, count=1)
        store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id="run")
        lease = store.acquire_run_lease("run", "v3-worker")
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TABLE run_controls")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_namespace")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_owner_id")
        db.execute("PRAGMA user_version=3")
        db.commit()
    with SQLiteRunStore(path, read_only=True) as reader:
        before = path.read_bytes()
        legacy = reader.inspection_snapshot("run")
        assert legacy["version"] == 3 and legacy["lease_fencing_supported"]
        assert not legacy["control"]["pause_supported"]
        assert path.read_bytes() == before
        with pytest.raises(RunLeaseError, match="live leases"):
            SQLiteRunStore(path, clock_ns=clock)
        assert reader.snapshot("run")["version"] == 3
        clock.expire()
        with SQLiteRunStore(path, clock_ns=clock) as upgraded:
            assert upgraded.get_run_lease("run").epoch == lease.epoch
            assert upgraded.get_control("run")["pause_generation"] == 0
            assert reader.snapshot("run")["version"] == 4
            assert reader.inspection_snapshot("run")["control"]["pause_supported"]
