"""Safety and staging tests for the in-process optimization runner."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from collections.abc import Mapping

import pytest

from smythe.optimize.contracts import (
    Candidate,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
    ObjectiveDirection,
)
from smythe.optimize.engine import (
    OptimizationError,
    OptimizationLimitError,
    OptimizationNeedsAttention,
    OptimizationRunner,
    TrialContext,
    TrialOutcome,
    _bound_phase,
    _split_seeds,
)
from smythe.optimize.ledger import CampaignLeaseConflict, ExperimentLedger, TrialStatus


EVALUATOR_HASH = "sha256:" + "e" * 64


def _contract(
    *,
    max_trials: int = 14,
    max_budget_microusd: int = 140,
    max_candidates: int = 2,
    required_gates: tuple[str, ...] = ("safe",),
    mutable_field_rules: Mapping[str, MutableFieldRule] | None = None,
) -> ExperimentContract:
    return ExperimentContract(
        name="engine_test",
        objectives=(
            MetricObjective(
                "quality",
                ObjectiveDirection.MAXIMIZE,
                primary=True,
            ),
            MetricObjective(
                "risk",
                ObjectiveDirection.MINIMIZE,
                hard_max=5.0,
                max_regression=1.0,
            ),
        ),
        mutable_fields=("strength",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=max_candidates,
        max_parallel_candidates=2,
        max_trials=max_trials,
        max_wall_seconds=60,
        max_budget_microusd=max_budget_microusd,
        per_trial_reservation_microusd=10,
        confidence=0.8,
        min_improvement=0.1,
        base_seed=17,
        required_gates=required_gates,
        mutable_field_rules=(
            {"strength": MutableFieldRule("integer", minimum=1, maximum=8)}
            if mutable_field_rules is None
            else mutable_field_rules
        ),
    )


def _candidates(contract: ExperimentContract) -> tuple[Candidate, Candidate]:
    incumbent = Candidate(
        contract=contract,
        policy={"strength": 1},
        hypothesis="incumbent",
    )
    challenger = Candidate(
        contract=contract,
        policy={"strength": 2},
        hypothesis="challenger",
        parent=incumbent.candidate_id,
    )
    return incumbent, challenger


def _runner(
    contract: ExperimentContract,
    ledger: ExperimentLedger,
    evaluator: Callable[[TrialContext], object],
    *,
    campaign_id: str | None = None,
    bootstrap_resamples: int = 100,
) -> OptimizationRunner:
    return OptimizationRunner(
        contract,
        ledger,
        evaluator,  # type: ignore[arg-type]
        evaluator_hash=EVALUATOR_HASH,
        bootstrap_resamples=bootstrap_resamples,
        campaign_id=campaign_id,
    )


@pytest.mark.asyncio
async def test_promotes_only_after_confirmation_and_holdout_then_resumes(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    calls: list[TrialContext] = []
    active = 0
    peak_active = 0

    async def evaluate(context: TrialContext) -> TrialOutcome:
        nonlocal active, peak_active
        calls.append(context)
        assert context.deadline_monotonic > time.monotonic()
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0)
        active -= 1
        quality = 10.0 if context.candidate == challenger else 0.0
        return TrialOutcome(
            metrics={"quality": quality, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "promotion.sqlite3", durability="normal") as ledger:
        runner = _runner(contract, ledger, evaluate)
        first = await runner.run(incumbent, (challenger,))
        assert first.promoted is True
        assert first.selected_candidate_id == challenger.candidate_id
        assert first.confirmation_assessment is not None
        assert first.confirmation_assessment.promote is True
        assert first.holdout_assessment is not None
        assert first.holdout_assessment.promote is True
        assert (
            first.holdout_seed_commitment
            == ledger.get_holdout_seed_commitment(first.campaign_id)
        )
        decision = ledger.list_decisions(first.campaign_id)[0]
        assert (
            decision["assessment"]["holdout_seed_commitment"]
            == first.holdout_seed_commitment
        )
        assert not hasattr(ledger, "get_holdout_seed_material")
        assert "holdout_nonce" not in json.dumps(first.to_dict(), sort_keys=True)
        assert first.ledger_snapshot["trial_counts"] == {"completed": 14}
        assert peak_active <= contract.max_parallel_candidates
        assert peak_active == 2

        call_count = len(calls)
        resumed = await runner.run(incumbent, (challenger,))
        assert resumed.to_dict() == first.to_dict()
        assert len(calls) == call_count == 14
        assert resumed.decision_id == first.decision_id

    split_seeds = {
        split: {context.seed for context in calls if context.split == split}
        for split in ("development", "confirmation", "holdout")
    }
    assert split_seeds["development"].isdisjoint(split_seeds["confirmation"])
    assert split_seeds["development"].isdisjoint(split_seeds["holdout"])
    assert split_seeds["confirmation"].isdisjoint(split_seeds["holdout"])
    assert split_seeds["holdout"] != set(_split_seeds(contract)[2])
    assert len({context.deadline_monotonic for context in calls}) == 1


def test_secret_holdout_material_changes_only_holdout_seeds():
    contract = _contract()
    first = _split_seeds(contract, holdout_seed_material=b"a" * 32)
    second = _split_seeds(contract, holdout_seed_material=b"b" * 32)

    assert first[:2] == second[:2]
    assert first[2] != second[2]
    with pytest.raises(ValueError, match="32 bytes"):
        _split_seeds(contract, holdout_seed_material=b"short")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract",
    [
        _contract(required_gates=()),
        _contract(mutable_field_rules={}),
    ],
)
async def test_runner_rejects_incomplete_executable_contract_before_mutation(
    tmp_path,
    contract: ExperimentContract,
):
    incumbent, challenger = _candidates(contract)

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        raise AssertionError("invalid contract must not invoke evaluator")

    with ExperimentLedger(tmp_path / f"invalid-{contract.contract_hash[-8:]}.sqlite3") as ledger:
        with pytest.raises(OptimizationError, match="executable optimization contracts"):
            await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        count = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gates",
    [
        {},
        {"safe": True, "extra": True},
        {"safe": True, " safe ": True},
    ],
)
async def test_outcome_gate_inventory_must_match_exactly_after_normalization(
    tmp_path,
    gates: dict[str, bool],
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        return TrialOutcome(
            metrics={"quality": 1.0, "risk": 1.0},
            gates=gates,
            actual_cost_microusd=1,
        )

    suffix = str(len(gates)) + str(len("".join(gates)))
    with ExperimentLedger(tmp_path / f"gates-{suffix}.sqlite3") as ledger:
        with pytest.raises(OptimizationNeedsAttention):
            await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        trials = ledger.list_trials(
            _runner(contract, ledger, evaluate)._build_plan(
                incumbent, (challenger,)
            )["campaign_id"]
        )
        assert len(trials) == 1
        assert trials[0].status is TrialStatus.UNKNOWN


@pytest.mark.asyncio
async def test_huge_integer_metric_after_dispatch_is_durably_unknown(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    calls = 0

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        nonlocal calls
        calls += 1
        return TrialOutcome(
            metrics={"quality": 10**1000, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "huge.sqlite3") as ledger:
        runner = _runner(contract, ledger, evaluate)
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        campaign_id = runner._build_plan(incumbent, (challenger,))["campaign_id"]
        assert ledger.list_trials(campaign_id)[0].status is TrialStatus.UNKNOWN
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        assert calls == 1


@pytest.mark.asyncio
async def test_huge_integer_cost_after_dispatch_is_durably_unknown(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        return TrialOutcome(
            metrics={"quality": 1.0, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=10**1000,
        )

    with ExperimentLedger(tmp_path / "huge-cost.sqlite3") as ledger:
        runner = _runner(contract, ledger, evaluate)
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        campaign_id = runner._build_plan(incumbent, (challenger,))["campaign_id"]
        assert ledger.list_trials(campaign_id)[0].status is TrialStatus.UNKNOWN


def test_bootstrap_resample_cap_is_enforced_before_running(tmp_path):
    contract = _contract()

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        raise AssertionError

    with ExperimentLedger(tmp_path / "resample-cap.sqlite3") as ledger:
        with pytest.raises(ValueError, match="not exceeding"):
            _runner(
                contract,
                ledger,
                evaluate,
                bootstrap_resamples=1_000_001,
            )


@pytest.mark.asyncio
async def test_holdout_rejection_never_promotes(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def evaluate(context: TrialContext) -> TrialOutcome:
        is_challenger = context.candidate == challenger
        quality = 10.0 if is_challenger else 0.0
        if context.split == "holdout" and is_challenger:
            quality = -10.0
        return TrialOutcome(
            metrics={"quality": quality, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "holdout.sqlite3", durability="normal") as ledger:
        result = await _runner(contract, ledger, evaluate).run(
            incumbent, (challenger,)
        )
        assert result.confirmation_assessment is not None
        assert result.confirmation_assessment.promote is True
        assert result.holdout_assessment is not None
        assert result.holdout_assessment.promote is False
        assert result.promoted is False
        assert "holdout rejected" in result.reason


@pytest.mark.asyncio
async def test_callback_failure_becomes_unknown_and_is_never_redispatched(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    calls = 0

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom\nwith control\x00text")

    with ExperimentLedger(tmp_path / "unknown.sqlite3", durability="normal") as ledger:
        runner = _runner(contract, ledger, evaluate)
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        trials = ledger.list_trials(runner._build_plan(incumbent, (challenger,))["campaign_id"])
        assert len(trials) == 1
        assert trials[0].status is TrialStatus.UNKNOWN
        assert "\n" not in (trials[0].error or "")
        assert "\x00" not in (trials[0].error or "")

        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_trials", "max_budget"),
    [(13, 140), (14, 139)],
)
async def test_full_paired_plan_limits_fail_before_campaign_mutation(
    tmp_path,
    max_trials: int,
    max_budget: int,
):
    contract = _contract(
        max_trials=max_trials,
        max_budget_microusd=max_budget,
    )
    incumbent, challenger = _candidates(contract)

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        raise AssertionError("preflight must not invoke evaluator")

    with ExperimentLedger(
        tmp_path / f"limit-{max_trials}-{max_budget}.sqlite3",
        durability="normal",
    ) as ledger:
        with pytest.raises(OptimizationLimitError):
            await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        count = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert count == 0


@pytest.mark.asyncio
async def test_custom_campaign_plan_drift_and_active_dispatch_fail_closed(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    campaign_id = "shared-campaign"
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocking(context: TrialContext) -> TrialOutcome:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return TrialOutcome(
            metrics={"quality": 10.0 if context.candidate == challenger else 0.0, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "shared.sqlite3", durability="normal") as ledger:
        owner = _runner(contract, ledger, blocking, campaign_id=campaign_id)
        owner_task = asyncio.create_task(owner.run(incumbent, (challenger,)))
        await started.wait()
        competing = _runner(contract, ledger, blocking, campaign_id=campaign_id)
        with pytest.raises(CampaignLeaseConflict):
            await competing.run(incumbent, (challenger,))
        dispatched = ledger.list_trials(campaign_id)
        assert len(dispatched) == 1
        assert dispatched[0].status is TrialStatus.DISPATCHED
        assert calls == 1
        owner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_task
        assert ledger.list_trials(campaign_id)[0].status is TrialStatus.UNKNOWN

    drift_path = tmp_path / "drift.sqlite3"
    with ExperimentLedger(drift_path, durability="normal") as ledger:
        original = _runner(contract, ledger, blocking, campaign_id="drift", bootstrap_resamples=100)
        plan = original._build_plan(incumbent, (challenger,))
        ledger.create_campaign(
            contract,
            incumbent.candidate_id,
            plan["all_candidates"],
            plan_hash=plan["plan_hash"],
            campaign_id="drift",
        )
        development_seed = _split_seeds(contract)[0][0]
        ledger.prepare_trial(
            "drift",
            incumbent.candidate_id,
            "development",
            _bound_phase("incumbent", plan["plan_hash"]),
            development_seed,
            lease=ledger.acquire_campaign_lease("drift", "fixture", ttl_s=3600),
            evaluator_hash=EVALUATOR_HASH,
            ceiling_microusd=contract.per_trial_reservation_microusd,
        )
        drifted = _runner(
            contract,
            ledger,
            blocking,
            campaign_id="drift",
            bootstrap_resamples=101,
        )
        with pytest.raises(OptimizationError, match="campaign binding drift"):
            await drifted.run(incumbent, (challenger,))


@pytest.mark.asyncio
async def test_precreated_custom_campaign_cannot_reveal_then_expand_plan(tmp_path):
    contract = _contract(
        max_candidates=3,
        max_trials=15,
        max_budget_microusd=150,
    )
    incumbent, first = _candidates(contract)
    second = Candidate(
        contract=contract,
        policy={"strength": 3},
        hypothesis="second challenger",
        parent=incumbent,
    )

    async def never_called(_context: TrialContext) -> TrialOutcome:
        raise AssertionError("plan drift must fail before evaluator dispatch")

    with ExperimentLedger(tmp_path / "sealed-superset.sqlite3") as ledger:
        original = _runner(
            contract,
            ledger,
            never_called,
            campaign_id="sealed-custom-campaign",
        )
        original_plan = original._build_plan(incumbent, (first,))
        ledger.create_campaign(
            contract,
            incumbent.candidate_id,
            original_plan["all_candidates"],
            plan_hash=original_plan["plan_hash"],
            campaign_id="sealed-custom-campaign",
        )

        assert not hasattr(ledger, "get_holdout_seed_material")
        with pytest.raises(OptimizationError, match="campaign binding drift"):
            await _runner(
                contract,
                ledger,
                never_called,
                campaign_id="sealed-custom-campaign",
            ).run(incumbent, (first, second))

        snapshot = ledger.snapshot("sealed-custom-campaign")
        assert snapshot["candidate_count"] == 2
        assert snapshot["trial_count"] == 0
