"""Campaign ownership fences, migration, and conservative offline recovery."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from smythe.optimize.contracts import (
    Candidate, ExperimentContract, MetricObjective, MutableFieldRule, ObjectiveDirection,
)
from smythe.optimize.engine import OptimizationNeedsAttention, OptimizationRunner, TrialOutcome
from smythe.optimize.ledger import (
    CampaignLease, CampaignLeaseConflict, CampaignLeaseError, ExperimentLedger,
    ExperimentLedgerError, PromotionDecision, TrialStatus, UnknownTrialError,
)
import smythe.optimize.ledger as ledger_module


PLAN = "sha256:" + "a" * 64
EVALUATOR = "sha256:" + "e" * 64


def _contract():
    return ExperimentContract(
        name="lease", objectives=(MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),),
        mutable_fields=("strength",), mutable_field_rules={"strength": MutableFieldRule("integer", minimum=1, maximum=8)},
        required_gates=("safe",), development_repetitions=1, confirmation_repetitions=3,
        holdout_repetitions=3, max_candidates=2, max_parallel_candidates=2,
        max_trials=14, max_wall_seconds=60, max_budget_microusd=140,
        per_trial_reservation_microusd=10, confidence=0.8, min_improvement=0.1, base_seed=17,
    )


def _candidates(contract):
    first = Candidate(contract=contract, policy={"strength": 1}, hypothesis="baseline")
    second = Candidate(contract=contract, policy={"strength": 2}, hypothesis="candidate", parent=first)
    return first, second


def _seal(ledger, campaign="campaign"):
    contract = _contract()
    candidates = _candidates(contract)
    ledger.create_campaign(contract, candidates[0].candidate_id, candidates,
                           plan_hash=PLAN, campaign_id=campaign)
    return candidates


def _prepare(ledger, lease, candidate, seed=17, phase="incumbent"):
    return ledger.prepare_trial(lease.campaign_id, candidate.candidate_id, "development",
                                phase + ".plan_" + PLAN[7:], seed, lease=lease,
                                evaluator_hash=EVALUATOR, ceiling_microusd=10)


def _complete(ledger, lease, key, quality=1):
    return ledger.complete_trial(key, lease=lease, metrics={"quality": quality},
                                  gates={"safe": True}, actual_cost_microusd=1,
                                  duration_ms=1, artifact_hashes=("sha256:" + "b" * 64,))


@pytest.fixture
def clock(monkeypatch):
    value = [100_000_000_000]
    monkeypatch.setattr(ledger_module, "_lease_now_ns", lambda: value[0])
    return value


def _evidence(ledger):
    return {table: [tuple(row) for row in ledger._connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("campaigns", "candidates", "trials", "trial_events", "promotion_decisions")}


def test_two_connections_cannot_settle_another_dispatch(tmp_path, clock):
    path = tmp_path / "owned.db"
    with ExperimentLedger(path, durability="normal") as first, ExperimentLedger(path, durability="normal") as other:
        candidate, _ = _seal(first)
        lease = first.acquire_campaign_lease("campaign", "first")
        key = _prepare(first, lease, candidate)
        first.claim_trial_dispatch(key, lease=lease)
        before = _evidence(first)
        with pytest.raises(CampaignLeaseConflict):
            other.acquire_campaign_lease("campaign", "other")
        foreign = replace(lease, owner_id="other")
        for action in (lambda: _complete(other, foreign, key, 9),
                       lambda: other.mark_trial_unknown(key, "foreign", lease=foreign),
                       lambda: other.mark_trial_dispatched(key, lease=foreign)):
            with pytest.raises(CampaignLeaseError):
                action()
        assert _evidence(first) == before
        assert _complete(first, lease, key, 8).metrics == {"quality": 8}


@pytest.mark.parametrize("unknown", [False, True])
def test_takeover_cannot_admit_prepared_sibling_or_rewrite_history(tmp_path, clock, unknown):
    with ExperimentLedger(tmp_path / "takeover.db", durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        first = ledger.acquire_campaign_lease("campaign", "owner", ttl_s=1)
        key = _prepare(ledger, first, candidate)
        sibling = _prepare(ledger, first, candidate, seed=18)
        ledger.claim_trial_dispatch(key, lease=first)
        if unknown:
            ledger.mark_trial_unknown(key, "lost", lease=first)
        before = _evidence(ledger)
        exposure = ledger.snapshot("campaign")["cost"]
        clock[0] += 1_000_000_000
        second = ledger.acquire_campaign_lease("campaign", "owner")
        assert second.epoch == first.epoch + 1
        for action in (lambda: ledger.claim_trial_dispatch(sibling, lease=second),
                       lambda: ledger.mark_trial_dispatched(sibling, lease=second),
                       lambda: _prepare(ledger, second, candidate, seed=19),
                       lambda: _complete(ledger, first, key),
                       lambda: ledger.release_campaign_lease(first)):
            with pytest.raises((CampaignLeaseError, UnknownTrialError)):
                action()
        if not unknown:
            with pytest.raises(CampaignLeaseError):
                _complete(ledger, second, key)
            with pytest.raises(CampaignLeaseError):
                ledger.mark_trial_unknown(key, "new interpretation", lease=second)
        assert _evidence(ledger) == before
        assert ledger.snapshot("campaign")["cost"] == exposure
        assert ledger.get_campaign_lease("campaign") == second


def test_prepared_takeover_and_stale_completed_idempotency(tmp_path, clock):
    with ExperimentLedger(tmp_path / "prepared.db", durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        first = ledger.acquire_campaign_lease("campaign", "first", ttl_s=1)
        key = _prepare(ledger, first, candidate)
        clock[0] += 1_000_000_000
        second = ledger.acquire_campaign_lease("campaign", "second")
        ledger.claim_trial_dispatch(key, lease=second)
        completed = _complete(ledger, second, key)
        before = _evidence(ledger)
        for action in (lambda: _complete(ledger, first, key),
                       lambda: ledger.append_trial(completed, lease=first),
                       lambda: _prepare(ledger, first, candidate)):
            with pytest.raises(CampaignLeaseError):
                action()
        assert ledger.append_trial(completed, lease=second) == completed
        ledger.release_campaign_lease(second)
        third = ledger.acquire_campaign_lease("campaign", "third")
        assert third.epoch == 3
        assert _complete(ledger, third, key) == completed
        assert _evidence(ledger) == before


@pytest.mark.parametrize("ttl", [True, False, 0, -1, float("inf"), float("nan"), "1", 10**1000, 1e-20])
def test_bad_ttl_cannot_change_ownership(tmp_path, ttl):
    with ExperimentLedger(tmp_path / "bad.db", durability="normal") as ledger:
        _seal(ledger)
        with pytest.raises(ValueError):
            ledger.acquire_campaign_lease("campaign", "owner", ttl_s=ttl)
        assert ledger.get_campaign_lease("campaign") is None
        assert not ledger._connection.execute("SELECT * FROM campaign_lease_epochs").fetchall()


def test_heartbeat_cannot_resurrect_lease_at_renewal_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_module, "_lease_now_ns", lambda: 100)
    with ExperimentLedger(tmp_path / "heartbeat.db", durability="normal") as ledger:
        _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "owner", ttl_s=1e-7)
        assert lease.expires_at_ns == 200
        values = iter([199, 199, 201, 201])
        monkeypatch.setattr(ledger_module, "_lease_now_ns", lambda: next(values))
        with pytest.raises(CampaignLeaseError):
            ledger.heartbeat_campaign_lease(lease, ttl_s=1e-7)
        assert ledger.get_campaign_lease("campaign") == lease


def test_write_clock_is_sampled_after_sqlite_lock_wait(tmp_path, clock):
    path = tmp_path / "waiting.db"
    with ExperimentLedger(path, durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "owner", ttl_s=1)
        key = _prepare(ledger, lease, candidate)
        entered = threading.Event()
        ledger._connection.set_trace_callback(lambda sql: entered.set() if sql == "BEGIN IMMEDIATE" else None)
        with closing(sqlite3.connect(path, isolation_level=None)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(ledger.claim_trial_dispatch, key, lease=lease)
                try:
                    assert entered.wait(5)
                    clock[0] += 1_000_000_000
                finally:
                    blocker.execute("COMMIT")
                with pytest.raises(CampaignLeaseError):
                    future.result(timeout=5)
        assert ledger.get_trial(key).status is TrialStatus.PREPARED
        assert not ledger._connection.execute("SELECT * FROM trial_dispatch_owners").fetchall()


def _decision(ledger, lease, candidates):
    keys = []
    for candidate, role in zip(candidates, ("incumbent", "challenger"), strict=True):
        key = _prepare(ledger, lease, candidate, phase=role)
        ledger.claim_trial_dispatch(key, lease=lease)
        _complete(ledger, lease, key)
        keys.append(key)
    return PromotionDecision(campaign_id=lease.campaign_id, candidate_id=candidates[1].candidate_id,
                             promoted=False, reason="offline rejection", trial_keys=tuple(keys),
                             assessment={"optimization_plan_hash": PLAN,
                                         "holdout_seed_commitment": ledger.get_holdout_seed_commitment(lease.campaign_id),
                                         "stage": "development"})


def test_decision_alias_and_idempotency_require_live_owner(tmp_path, clock):
    with ExperimentLedger(tmp_path / "decision.db", durability="normal") as ledger:
        candidates = _seal(ledger)
        first = ledger.acquire_campaign_lease("campaign", "first", ttl_s=1)
        decision = _decision(ledger, first, candidates)
        ledger.append_decision(decision, lease=first)
        before = _evidence(ledger)
        clock[0] += 1_000_000_000
        second = ledger.acquire_campaign_lease("campaign", "second")
        for method in (ledger.append_decision, ledger.append_promotion_decision):
            with pytest.raises(CampaignLeaseError):
                method(decision, lease=first)
            assert method(decision, lease=second) == decision.decision_id
        assert _evidence(ledger) == before


def test_expiry_during_decision_validation_rolls_back(tmp_path, clock, monkeypatch):
    with ExperimentLedger(tmp_path / "late-decision.db", durability="normal") as ledger:
        candidates = _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "owner", ttl_s=1)
        decision = _decision(ledger, lease, candidates)
        original = ledger._validate_decision_payload

        def expire(*args):
            original(*args)
            clock[0] += 1_000_000_000

        monkeypatch.setattr(ledger, "_validate_decision_payload", expire)
        before = _evidence(ledger)
        with pytest.raises(CampaignLeaseError):
            ledger.append_decision(decision, lease=lease)
        assert _evidence(ledger) == before


def _remove_ownership_schema(ledger):
    for name in ledger._barrier_sql():
        ledger._connection.execute(f"DROP TRIGGER {name}")
    for table in ("trial_dispatch_owners", "campaign_leases", "campaign_lease_epochs"):
        ledger._connection.execute(f"DROP TABLE {table}")
    ledger._connection.execute("UPDATE ledger_meta SET version=3")


def _legacy_database(path):
    with ExperimentLedger(path, durability="normal") as ledger:
        candidates = _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "fixture", ttl_s=3600)
        decision = _decision(ledger, lease, candidates)
        ledger.append_decision(decision, lease=lease)
        evidence = _evidence(ledger)
        _remove_ownership_schema(ledger)
    return evidence


def test_v3_readonly_and_writable_migration_preserve_evidence(tmp_path):
    path = tmp_path / "legacy.db"
    evidence = _legacy_database(path)
    before = path.read_bytes()
    with ExperimentLedger(path, read_only=True) as reader:
        assert not reader.lease_supported
        assert reader.get_campaign_lease("campaign") is None
        assert _evidence(reader) == evidence
    assert path.read_bytes() == before
    with ExperimentLedger(path, durability="normal") as migrated:
        assert migrated.lease_supported
        assert _evidence(migrated) == evidence
        assert not migrated._connection.execute("SELECT * FROM trial_dispatch_owners").fetchall()


@pytest.mark.parametrize("operation", ["UPDATE", "INSERT", "DELETE"])
def test_migration_blocks_retained_legacy_connection(tmp_path, operation):
    path = tmp_path / "old-writer.db"
    evidence = _legacy_database(path)
    old = sqlite3.connect(path, isolation_level=None)
    try:
        # Compile and cache an actual old UPDATE before the new triggers exist.
        statement = "UPDATE trial_events SET payload_json=payload_json"
        old.execute(statement)
        with ExperimentLedger(path, durability="normal") as migrated:
            if operation == "INSERT":
                statement = "INSERT INTO trial_events SELECT * FROM trial_events LIMIT 1"
            elif operation == "DELETE":
                statement = "DELETE FROM trial_events"
            with pytest.raises(sqlite3.OperationalError, match="smythe_autotune_writer_version"):
                old.execute(statement)
            assert _evidence(migrated) == evidence
    finally:
        old.close()


def test_missing_or_altered_marker_fails_closed_on_reopen(tmp_path):
    path = tmp_path / "broken.db"
    with ExperimentLedger(path, durability="normal") as ledger:
        _seal(ledger)
        ledger._connection.execute("DROP TRIGGER autotune_v4_trial_events_insert")
    with pytest.raises(ExperimentLedgerError, match="writer barrier"):
        ExperimentLedger(path, durability="normal")


@pytest.mark.asyncio
async def test_independent_runners_conflict_and_release_on_cancel(tmp_path):
    path = tmp_path / "runner.db"
    contract = _contract()
    candidates = _candidates(contract)
    entered = asyncio.Event()
    calls = []

    async def evaluate(context):
        calls.append(context.trial_key)
        entered.set()
        await asyncio.Event().wait()

    with ExperimentLedger(path, durability="normal") as first, ExperimentLedger(path, durability="normal") as other:
        runner = OptimizationRunner(contract, first, evaluate, evaluator_hash=EVALUATOR)
        task = asyncio.create_task(runner.run(candidates[0], candidates[1:]))
        await asyncio.wait_for(entered.wait(), 10)
        campaign = runner._build_plan(candidates[0], candidates[1:])["campaign_id"]
        before = _evidence(first)
        try:
            competing = OptimizationRunner(contract, other, evaluate, evaluator_hash=EVALUATOR)
            with pytest.raises(CampaignLeaseConflict):
                await competing.run(candidates[0], candidates[1:])
            with pytest.raises(CampaignLeaseConflict):
                await runner.run(candidates[0], candidates[1:])
            assert _evidence(first) == before
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert calls and len(calls) == 1
        assert first.get_campaign_lease(campaign) is None
        assert first.list_trials(campaign)[0].status is TrialStatus.UNKNOWN
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(candidates[0], candidates[1:])
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("migrate_legacy", [False, True])
async def test_completed_runner_replay_keeps_exact_evidence_and_identity(tmp_path, migrate_legacy):
    contract = _contract()
    candidates = _candidates(contract)
    calls = []

    async def evaluate(context):
        calls.append(context.trial_key)
        await asyncio.sleep(0)
        return TrialOutcome(metrics={"quality": context.candidate.policy["strength"] * 10},
                            gates={"safe": True}, actual_cost_microusd=1)

    path = tmp_path / "replay.db"
    with ExperimentLedger(path, durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR, bootstrap_resamples=10)
        first = await runner.run(candidates[0], candidates[1:])
        before = _evidence(ledger)
        if migrate_legacy:
            _remove_ownership_schema(ledger)
    with ExperimentLedger(path, durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR, bootstrap_resamples=10)
        second = await runner.run(candidates[0], candidates[1:])
        assert first.to_dict() == second.to_dict()
        assert first.campaign_id.startswith("optimization_v3_")
        assert len(calls) == 14
        assert _evidence(ledger) == before
        assert "owner_id" not in json.dumps(first.to_dict())
        assert ledger.get_campaign_lease(first.campaign_id) is None


def test_token_is_strict_and_frozen():
    token = CampaignLease("campaign", "owner", 1, 1, 1, 2)
    with pytest.raises(AttributeError):
        token.epoch = 2
    for field, bad in (("epoch", True), ("epoch", 0), ("expires_at_ns", 2**63),
                       ("owner_id", " owner "), ("heartbeat_at_ns", 3)):
        with pytest.raises((CampaignLeaseError, ValueError)):
            replace(token, **{field: bad})


@pytest.mark.parametrize("corruption", ["missing", "zero", "behind", "exhausted"])
def test_counter_corruption_cannot_reissue_old_authority(tmp_path, clock, corruption):
    with ExperimentLedger(tmp_path / "epoch.db", durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        first = ledger.acquire_campaign_lease("campaign", "same")
        ledger.release_campaign_lease(first)
        lease = ledger.acquire_campaign_lease("campaign", "same")
        key = _prepare(ledger, lease, candidate)
        ledger.claim_trial_dispatch(key, lease=lease)
        ledger.release_campaign_lease(lease)
        if corruption == "missing":
            ledger._connection.execute("DELETE FROM campaign_lease_epochs")
        else:
            ledger._connection.execute("PRAGMA ignore_check_constraints=ON")
            ledger._connection.execute("UPDATE campaign_lease_epochs SET last_epoch=?",
                                       ({"zero": 0, "behind": 1, "exhausted": 2**63 - 1}[corruption],))
        before = _evidence(ledger)
        with pytest.raises(CampaignLeaseError):
            ledger.acquire_campaign_lease("campaign", "same")
        with pytest.raises(CampaignLeaseError):
            _complete(ledger, lease, key)
        assert _evidence(ledger) == before


@pytest.mark.parametrize("fresh", [False, True])
def test_simultaneous_initialization_preserves_one_version_and_all_payloads(tmp_path, fresh):
    path = tmp_path / "migration-race.db"
    evidence = ({table: [] for table in ("campaigns", "candidates", "trials", "trial_events", "promotion_decisions")}
                if fresh else _legacy_database(path))
    gate = threading.Barrier(2)

    def migrate():
        gate.wait(timeout=5)
        with ExperimentLedger(path, durability="normal") as ledger:
            return (_evidence(ledger), ledger.lease_supported,
                    ledger._connection.execute("SELECT count(*) FROM ledger_meta").fetchone()[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: migrate(), range(2)))
    assert results == [(evidence, True, 1), (evidence, True, 1)]


def test_migration_failure_rolls_back_schema_and_raw_evidence(tmp_path, monkeypatch):
    path = tmp_path / "migration-rollback.db"
    evidence = _legacy_database(path)
    with closing(sqlite3.connect(path)) as old:
        schema = old.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    original = ExperimentLedger._verify_barrier

    def fail(cls, cursor):
        original(cursor)
        raise RuntimeError("injected final migration validation failure")

    monkeypatch.setattr(ExperimentLedger, "_verify_barrier", classmethod(fail))
    with pytest.raises(RuntimeError, match="injected"):
        ExperimentLedger(path, durability="normal")
    with closing(sqlite3.connect(path)) as old:
        assert old.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall() == schema
        assert old.execute("SELECT version FROM ledger_meta").fetchone() == (3,)
    with ExperimentLedger(path, read_only=True) as reader:
        assert _evidence(reader) == evidence


def test_actual_schema_version_is_checked_on_every_write(tmp_path, clock):
    with ExperimentLedger(tmp_path / "version.db", durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "owner")
        ledger._connection.execute("UPDATE ledger_meta SET version=99")
        with pytest.raises(ExperimentLedgerError, match="schema v4"):
            _prepare(ledger, lease, candidate)
        assert not ledger._connection.execute("SELECT * FROM trials").fetchall()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel_cleanup", [False, True])
async def test_repeated_cancellation_drains_callbacks_before_release(tmp_path, parallel_cleanup):
    contract = _contract()
    candidates = _candidates(contract)
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    expected = 2 if parallel_cleanup else 1
    active, draining = set(), set()

    async def evaluate(context):
        if parallel_cleanup and context.split == "development":
            return TrialOutcome(metrics={"quality": context.candidate.policy["strength"] * 10},
                                gates={"safe": True}, actual_cost_microusd=1)
        active.add(context.trial_key)
        if len(active) == expected:
            entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.add(context.trial_key)
            if len(draining) == expected:
                cleaning.set()
            await release.wait()

    with ExperimentLedger(tmp_path / "cancel.db", durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    lease_ttl_s=30, lease_heartbeat_s=0.05)
        task = asyncio.create_task(runner.run(candidates[0], candidates[1:]))
        await asyncio.wait_for(entered.wait(), 10)
        campaign = runner._build_plan(candidates[0], candidates[1:])["campaign_id"]
        try:
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            current = ledger.get_campaign_lease(campaign)
            assert current is not None
            ledger.assert_campaign_lease(current)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
        assert ledger.get_campaign_lease(campaign) is None
        assert len(active) == len(draining) == expected
        assert sum(trial.status is TrialStatus.UNKNOWN for trial in ledger.list_trials(campaign)) == expected
        assert not any(thread.name == "smythe-autotune-heartbeat" for thread in threading.enumerate())


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_cancellation", [False, True])
async def test_heartbeat_loss_keeps_dispatched_exposure_and_does_not_retry(tmp_path, monkeypatch, prior_cancellation):
    contract = _contract()
    candidates = _candidates(contract)
    entered = asyncio.Event()
    trigger_loss = threading.Event()
    original = ExperimentLedger.heartbeat_campaign_lease
    calls = []

    def heartbeat(self, lease, **kwargs):
        if trigger_loss.is_set():
            # Simulate the durable release/expiry boundary before reporting loss.
            self._connection.execute("DELETE FROM campaign_leases WHERE campaign_id=?", (lease.campaign_id,))
            raise CampaignLeaseError("injected ownership loss")
        return original(self, lease, **kwargs)

    monkeypatch.setattr(ExperimentLedger, "heartbeat_campaign_lease", heartbeat)

    async def evaluate(context):
        calls.append(context.trial_key)
        entered.set()
        await asyncio.Event().wait()

    with ExperimentLedger(tmp_path / "lost.db", durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    lease_ttl_s=30, lease_heartbeat_s=0.05)

        async def invoke():
            if prior_cancellation:
                caller = asyncio.current_task()
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.sleep(0)
                assert caller.cancelling() > 0
            return await runner.run(candidates[0], candidates[1:])

        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            trigger_loss.set()
            with pytest.raises(CampaignLeaseError, match="heartbeat"):
                await asyncio.wait_for(task, 10)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        campaign = runner._build_plan(candidates[0], candidates[1:])["campaign_id"]
        assert ledger.list_trials(campaign)[0].status is TrialStatus.DISPATCHED
        assert ledger.snapshot(campaign)["cost"]["reserved_microusd"] == 10
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(candidates[0], candidates[1:])
        assert len(calls) == 1


def test_owned_process_loss_preserves_accepted_evidence_and_unknown_exposure(tmp_path, monkeypatch):
    contract = _contract()
    candidates = _candidates(contract)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"contract": contract.to_dict(),
                                  "candidates": [candidate.to_dict() for candidate in candidates]}), encoding="utf-8")
    code = r'''
import asyncio, json, os, sys
from pathlib import Path
from smythe.optimize.contracts import Candidate, ExperimentContract
from smythe.optimize.engine import OptimizationRunner, TrialOutcome
from smythe.optimize.ledger import ExperimentLedger
folder=Path(sys.argv[1])
payload=json.loads((folder/'request.json').read_text(encoding='utf-8'))
contract=ExperimentContract.from_dict(payload['contract'])
candidates=[Candidate.from_dict(value, contract=contract) for value in payload['candidates']]
async def evaluate(context):
    with (folder/'calls.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'trial_key': context.trial_key})+'\n')
    if context.candidate.candidate_id == candidates[0].candidate_id:
        return TrialOutcome(metrics={'quality': 1}, gates={'safe': True}, actual_cost_microusd=1)
    (folder/'ready.json').write_text(json.dumps({'pid': os.getpid(), 'campaign': context.campaign_id}), encoding='utf-8')
    await asyncio.Event().wait()
with ExperimentLedger(folder/'process.db', durability='normal') as ledger:
    runner=OptimizationRunner(contract, ledger, evaluate, evaluator_hash='sha256:'+'e'*64, lease_ttl_s=3600)
    asyncio.run(runner.run(candidates[0], candidates[1:]))
'''
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(root), *(path for path in sys.path if path)])
    # The Windows venv redirector is not the worker we need to own and kill.
    executable = sys._base_executable if os.name == "nt" else sys.executable
    process = subprocess.Popen([executable, "-c", code, str(tmp_path)], cwd=root,
                               env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"owned child exited before receipt: {stdout!r} {stderr!r}")
            try:
                ready = json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
        assert ready is not None, "child did not reach the dispatched callback barrier"
        assert ready["pid"] == process.pid
        with ExperimentLedger(tmp_path / "process.db", durability="normal") as ledger:
            campaign = ready["campaign"]
            lease = ledger.get_campaign_lease(campaign)
            assert lease is not None
            before = _evidence(ledger)
            assert ledger.snapshot(campaign)["trial_counts"] == {"dispatched": 1, "completed": 1}
            with pytest.raises(CampaignLeaseConflict):
                ledger.acquire_campaign_lease(campaign, "competitor")
            process.kill()
            process.wait(timeout=10)
            monkeypatch.setattr(ledger_module, "_lease_now_ns", lambda: lease.expires_at_ns + 1)

            async def never_called(_context):
                pytest.fail("a dispatched predecessor must not be retried")

            runner = OptimizationRunner(contract, ledger, never_called, evaluator_hash=EVALUATOR)
            with pytest.raises(OptimizationNeedsAttention):
                asyncio.run(runner.run(candidates[0], candidates[1:]))
            assert _evidence(ledger) == before
            assert ledger.snapshot(campaign)["cost"] == {
                "confirmed_microusd": 1, "reserved_microusd": 10,
                "unknown_exposure_microusd": 0, "total_exposure_microusd": 11,
            }
            assert len((tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.asyncio
async def test_independent_original_failure_survives_heartbeat_cleanup_error(tmp_path, monkeypatch):
    import smythe.optimize.engine as engine_module

    failure = RuntimeError("original independent failure")
    original_close = engine_module._LeaseHeartbeat.close

    def close_with_failure(self):
        original_close(self)
        self.failure = CampaignLeaseError("cleanup heartbeat error")

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(engine_module._LeaseHeartbeat, "close", close_with_failure)
    monkeypatch.setattr(OptimizationRunner, "_run_owned", fail)
    contract = _contract()
    candidates = _candidates(contract)
    with ExperimentLedger(tmp_path / "original.db", durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, fail, evaluator_hash=EVALUATOR)
        with pytest.raises(RuntimeError) as caught:
            await runner.run(candidates[0], candidates[1:])
        assert caught.value is failure
        assert any("heartbeat" in note for note in failure.__notes__)


def test_renewed_expiry_is_authoritative_and_backwards_clock_fails_closed(tmp_path, clock):
    with ExperimentLedger(tmp_path / "renewed.db", durability="normal") as ledger:
        _seal(ledger)
        lease = ledger.acquire_campaign_lease("campaign", "owner", ttl_s=1)
        clock[0] += 100
        renewed = ledger.heartbeat_campaign_lease(lease, ttl_s=1)
        assert renewed.epoch == lease.epoch
        assert renewed.expires_at_ns == lease.expires_at_ns + 100
        clock[0] = lease.expires_at_ns + 50
        ledger.assert_campaign_lease(lease)
        clock[0] = renewed.heartbeat_at_ns - 1
        for action in (lambda: ledger.assert_campaign_lease(lease),
                       lambda: ledger.heartbeat_campaign_lease(lease),
                       lambda: ledger.release_campaign_lease(lease)):
            with pytest.raises(CampaignLeaseError):
                action()
        assert ledger.get_campaign_lease("campaign") == renewed
        clock[0] = renewed.expires_at_ns
        with pytest.raises(CampaignLeaseError):
            ledger.release_campaign_lease(lease)
        successor = ledger.acquire_campaign_lease("campaign", "successor")
        assert successor.epoch == renewed.epoch + 1


@pytest.mark.parametrize("state", [TrialStatus.PREPARED, TrialStatus.DISPATCHED, TrialStatus.UNKNOWN])
def test_migration_keeps_unfinished_state_and_never_invents_dispatch_ownership(tmp_path, state):
    path = tmp_path / "unfinished-v3.db"
    with ExperimentLedger(path, durability="normal") as ledger:
        candidate, _ = _seal(ledger)
        original = ledger.acquire_campaign_lease("campaign", "original")
        accepted = _prepare(ledger, original, candidate)
        ledger.claim_trial_dispatch(accepted, lease=original)
        accepted_record = _complete(ledger, original, accepted)
        unfinished = _prepare(ledger, original, candidate, seed=18)
        if state is not TrialStatus.PREPARED:
            ledger.claim_trial_dispatch(unfinished, lease=original)
        if state is TrialStatus.UNKNOWN:
            ledger.mark_trial_unknown(unfinished, "old unknown outcome", lease=original)
        evidence = _evidence(ledger)
        exposure = ledger.snapshot("campaign")["cost"]
        _remove_ownership_schema(ledger)
    with ExperimentLedger(path, read_only=True) as reader:
        assert reader.get_trial(unfinished).status is state
        assert _evidence(reader) == evidence
    with ExperimentLedger(path, durability="normal") as migrated:
        assert not migrated._connection.execute("SELECT * FROM trial_dispatch_owners").fetchall()
        assert _evidence(migrated) == evidence
        assert migrated.snapshot("campaign")["cost"] == exposure
        successor = migrated.acquire_campaign_lease("campaign", "successor")
        if state is TrialStatus.PREPARED:
            migrated.claim_trial_dispatch(unfinished, lease=successor)
            assert migrated.get_trial(unfinished).status is TrialStatus.DISPATCHED
        else:
            with pytest.raises((CampaignLeaseError, UnknownTrialError)):
                _prepare(migrated, successor, candidate, seed=19)
            with pytest.raises((CampaignLeaseError, UnknownTrialError)):
                _complete(migrated, successor, unfinished)
            if state is TrialStatus.DISPATCHED:
                with pytest.raises(CampaignLeaseError):
                    migrated.mark_trial_unknown(unfinished, "new interpretation", lease=successor)
            assert _evidence(migrated) == evidence
        assert migrated.get_trial(accepted) == accepted_record


@pytest.mark.asyncio
async def test_heartbeat_renews_while_statistics_block_the_event_loop(tmp_path, monkeypatch):
    contract = _contract()
    candidates = _candidates(contract)
    calculating, renewed = threading.Event(), threading.Event()
    original_heartbeat = ExperimentLedger.heartbeat_campaign_lease
    original_assess = OptimizationRunner._assess
    renewals = []

    def heartbeat(self, lease, **kwargs):
        result = original_heartbeat(self, lease, **kwargs)
        if calculating.is_set():
            renewals.append(result)
            renewed.set()
        return result

    def assess(self, *args, **kwargs):
        renewed.clear()
        calculating.set()
        try:
            assert renewed.wait(5), "heartbeat stopped during synchronous statistics"
        finally:
            calculating.clear()
        return original_assess(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentLedger, "heartbeat_campaign_lease", heartbeat)
    monkeypatch.setattr(OptimizationRunner, "_assess", assess)

    async def evaluate(context):
        return TrialOutcome(metrics={"quality": context.candidate.policy["strength"] * 10},
                            gates={"safe": True}, actual_cost_microusd=1)

    with ExperimentLedger(tmp_path / "statistics.db", durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    bootstrap_resamples=10, lease_heartbeat_s=0.05)
        result = await runner.run(candidates[0], candidates[1:])
        assert len(renewals) >= 2
        assert all(item.heartbeat_at_ns > item.acquired_at_ns for item in renewals)
        assert ledger.get_campaign_lease(result.campaign_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["constructor", "start", "task_factory"])
async def test_ownership_setup_failure_cannot_leave_a_running_campaign(tmp_path, monkeypatch, failure_phase):
    import smythe.optimize.engine as engine_module

    failure = RuntimeError("ownership setup failed")
    contract = _contract()
    candidates = _candidates(contract)
    calls = []

    def fail(*args, **kwargs):
        raise failure

    async def evaluate(context):
        calls.append(context.trial_key)
        pytest.fail("setup failure must precede evaluator admission")

    if failure_phase == "constructor":
        monkeypatch.setattr(engine_module, "_LeaseHeartbeat", fail)
    elif failure_phase == "start":
        monkeypatch.setattr(engine_module._LeaseHeartbeat, "start", fail)
    else:
        monkeypatch.setattr(engine_module.asyncio, "create_task", fail)
    before = asyncio.all_tasks()
    with ExperimentLedger(tmp_path / "setup.db", durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR)
        with pytest.raises(RuntimeError) as caught:
            await runner.run(candidates[0], candidates[1:])
        await asyncio.sleep(0)
        campaign = runner._build_plan(candidates[0], candidates[1:])["campaign_id"]
        assert caught.value is failure
        assert calls == []
        assert ledger.list_trials(campaign) == []
        assert ledger.get_campaign_lease(campaign) is None
    assert asyncio.all_tasks() == before
    assert not any(thread.name == "smythe-autotune-heartbeat" for thread in threading.enumerate())


def test_lease_inspection_uses_one_snapshot_across_a_concurrent_takeover(tmp_path, clock, monkeypatch):
    path = tmp_path / "inspection.db"
    with ExperimentLedger(path, durability="normal") as reader, ExperimentLedger(path, durability="normal") as writer:
        _seal(reader)
        first = reader.acquire_campaign_lease("campaign", "first", ttl_s=1)
        original = reader._lease_from_row
        acquired = []

        def interleave(cursor, row):
            clock[0] = first.expires_at_ns
            acquired.append(writer.acquire_campaign_lease("campaign", "second"))
            return original(cursor, row)

        monkeypatch.setattr(reader, "_lease_from_row", interleave)
        assert reader.get_campaign_lease("campaign") == first
        assert not reader._connection.in_transaction
        monkeypatch.setattr(reader, "_lease_from_row", original)
        assert reader.get_campaign_lease("campaign") == acquired[0]

        def fail(cursor, row):
            raise CampaignLeaseError("injected inspection validation failure")

        monkeypatch.setattr(reader, "_lease_from_row", fail)
        with pytest.raises(CampaignLeaseError, match="injected inspection"):
            reader.get_campaign_lease("campaign")
        assert not reader._connection.in_transaction
        monkeypatch.setattr(reader, "_lease_from_row", original)
        assert reader.get_campaign_lease("campaign") == acquired[0]
