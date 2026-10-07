"""Safety and staging tests for the in-process optimization runner."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from smythe.optimize.contracts import (
    Candidate,
    EvaluatorManifest,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
    ObjectiveDirection,
    canonical_json_bytes,
    sha256_prefixed,
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
    _task_indices,
)
from smythe.optimize.ledger import (
    CampaignLeaseConflict,
    ExperimentLedger,
    HoldoutAlreadyUsedError,
    TrialStatus,
    holdout_identity,
)
from smythe.optimize.statistics import PROMOTION_METHOD
from smythe.validation import ValidatorIdentity


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
    **options: object,
) -> OptimizationRunner:
    return OptimizationRunner(
        contract,
        ledger,
        evaluator,  # type: ignore[arg-type]
        evaluator_hash=EVALUATOR_HASH,
        bootstrap_resamples=bootstrap_resamples,
        campaign_id=campaign_id,
        **options,  # type: ignore[arg-type]
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


# How a candidate is judged, what it may spend and what it may do.
RESERVED_ROOTS = (
    "acceptance",
    "validators",
    "verifier",
    "evaluator",
    "harness",
    "budget",
    "spend",
    "ceilings",
    "capabilities",
    "tools",
    "permissions",
    "authority",
    "holdout",
)


def _policy_contract(*paths: str, **limits: int) -> ExperimentContract:
    """The engine test contract with an integer rule for each mutable path."""

    return replace(
        _contract(**limits),
        mutable_fields=paths,
        mutable_field_rules={
            path: MutableFieldRule("integer", minimum=1, maximum=8) for path in paths
        },
    )


def _ledger_state(ledger: ExperimentLedger, path: Path) -> tuple[list[str], dict[str, bytes]]:
    """Every row the ledger holds, and the bytes of its database files."""

    files = sorted(path.parent.glob(path.name + "*"))
    return list(ledger._connection.iterdump()), {item.name: item.read_bytes() for item in files}


def _boundaries(
    contract: ExperimentContract, manifest: EvaluatorManifest, *candidates: Candidate
) -> bytes:
    return canonical_json_bytes(
        {
            "contract": contract.to_dict(),
            "contract_hash": contract.contract_hash,
            "evaluator": manifest.to_dict(),
            "evaluator_hash": manifest.evaluator_hash,
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
    )


async def _never_called(_context: TrialContext) -> TrialOutcome:
    raise AssertionError("a refused campaign must not invoke the evaluator")


@pytest.mark.asyncio
async def test_runner_refuses_a_mutable_reserved_path_before_any_ledger_write(tmp_path):
    paths = (
        *RESERVED_ROOTS,
        *(f"{root}.limit" for root in RESERVED_ROOTS),
        "acceptance.policy.mode",
        "Budget.max_usd",
        "TOOLS",
    )
    manifest = EvaluatorManifest(
        code_digest=EVALUATOR_HASH,
        validators=(ValidatorIdentity.for_config("safe", "1"),),
    )
    ledger_path = tmp_path / "reserved.sqlite3"

    with ExperimentLedger(ledger_path, durability="normal") as ledger:
        before = _ledger_state(ledger, ledger_path)
        for path in paths:
            contract = _policy_contract("strength", path)
            incumbent = Candidate(
                contract=contract, policy={"strength": 1, path: 1}, hypothesis="incumbent"
            )
            challenger = Candidate(
                contract=contract,
                policy={"strength": 1, path: 8},
                hypothesis="relax the bound",
                parent=incumbent,
            )
            boundaries = _boundaries(contract, manifest, incumbent, challenger)
            runner = OptimizationRunner(
                contract,
                ledger,
                _never_called,
                evaluator_hash=manifest.evaluator_hash,
                bootstrap_resamples=100,
            )
            with pytest.raises(OptimizationError, match="reserved") as refused:
                await runner.run(incumbent, (challenger,))
            assert path in str(refused.value)
            assert _ledger_state(ledger, ledger_path) == before
            # The contract, evaluator data and candidate inventory are untouched.
            assert runner.evaluator_hash == manifest.evaluator_hash
            assert _boundaries(contract, manifest, incumbent, challenger) == boundaries


@pytest.mark.asyncio
async def test_a_path_that_only_starts_like_a_reserved_name_stays_mutable(tmp_path):
    contract = _policy_contract("strength", "toolbar.layout", "budgeted")
    incumbent, challenger = _candidates(contract)

    with ExperimentLedger(tmp_path / "lookalike.sqlite3", durability="normal") as ledger:
        result = await _runner(contract, ledger, _challengers_win).run(incumbent, (challenger,))
    assert result.promoted is True


@pytest.mark.asyncio
async def test_a_stored_campaign_with_a_reserved_path_loads_but_does_not_resume(
    tmp_path, monkeypatch
):
    from smythe.optimize import engine

    contract = _policy_contract("strength", "budget.max_usd")
    incumbent, challenger = _candidates(contract)
    identity = holdout_identity(contract, EVALUATOR_HASH)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    ledger_path = tmp_path / "stored.sqlite3"
    with ExperimentLedger(ledger_path, durability="normal") as ledger:
        with monkeypatch.context() as earlier_release:
            earlier_release.setattr(engine, "_RESERVED_PATH_ROOTS", frozenset())
            first = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None
        evaluated = len(calls)

        # The ledger still reads the stored contract back for inspection and
        # for holdout accounting.
        stored = ledger.snapshot(first.campaign_id)["campaign"]["contract"]
        assert ExperimentContract.from_dict(stored) == contract
        assert ledger.holdout_uses(identity) == {challenger.policy_hash: (first.campaign_id,)}
        assert len(ledger.validate_decision_inventory(first.campaign_id)) == 1

        before = _ledger_state(ledger, ledger_path)
        with pytest.raises(OptimizationError, match="budget.max_usd"):
            await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert _ledger_state(ledger, ledger_path) == before
        assert len(calls) == evaluated


# Two challengers, each with one development trial, and two paired splits of
# three seeds; a reference adds five more.
_TWO_FIELD_LIMITS = {"max_candidates": 3, "max_trials": 20, "max_budget_microusd": 200}


def _two_field_contract() -> ExperimentContract:
    return replace(
        _contract(**_TWO_FIELD_LIMITS),
        mutable_fields=("strength", "style"),
        mutable_field_rules={
            "strength": MutableFieldRule("integer", minimum=1, maximum=8),
            "style": MutableFieldRule("string", choices=("plain", "bold")),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incumbent_policy", "changed_policy"),
    [
        ({"strength": 1, "style": "plain"}, {"strength": 2, "style": "bold"}),
        ({"strength": 1}, {"style": "bold"}),
    ],
    ids=["two-values", "different-path"],
)
async def test_single_change_refuses_a_challenger_that_changes_two_paths(
    tmp_path, incumbent_policy, changed_policy
):
    contract = _two_field_contract()
    incumbent = Candidate(contract=contract, policy=incumbent_policy, hypothesis="incumbent")
    one = Candidate(
        contract=contract,
        policy={**incumbent_policy, "strength": 3},
        hypothesis="one change",
        parent=incumbent,
    )
    two = Candidate(
        contract=contract, policy=changed_policy, hypothesis="two changes", parent=incumbent
    )
    ledger_path = tmp_path / "single.sqlite3"

    with ExperimentLedger(ledger_path, durability="normal") as ledger:
        before = _ledger_state(ledger, ledger_path)
        with pytest.raises(OptimizationError, match="exactly one policy path") as refused:
            await _runner(contract, ledger, _never_called, single_change=True).run(
                incumbent, (one, two)
            )
        message = str(refused.value)
        assert two.candidate_id in message
        assert one.candidate_id not in message
        assert "strength, style" in message
        assert _ledger_state(ledger, ledger_path) == before

        # The constraint is opt-in, and a reference is never promoted, so it
        # may differ in more than one path.
        assert _runner(contract, ledger, _never_called)._build_plan(incumbent, (one, two))
        assert _runner(contract, ledger, _never_called, single_change=True)._build_plan(
            incumbent, (one,), (two,)
        )


@pytest.mark.asyncio
async def test_single_change_runs_one_path_challengers_and_binds_the_plan(tmp_path):
    contract = _two_field_contract()
    incumbent = Candidate(
        contract=contract, policy={"strength": 1, "style": "plain"}, hypothesis="incumbent"
    )
    stronger = Candidate(
        contract=contract,
        policy={"strength": 2, "style": "plain"},
        hypothesis="stronger",
        parent=incumbent,
    )
    # Dropping a path the incumbent sets is also one change.
    unstyled = Candidate(
        contract=contract, policy={"strength": 1}, hypothesis="no style", parent=incumbent
    )
    challengers = (stronger, unstyled)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    payload = {
        "runner_version": 4,
        "statistics": {"method": PROMOTION_METHOD, "descriptive_bootstrap_resamples": 100},
        "contract_hash": contract.contract_hash,
        "incumbent_candidate_id": incumbent.candidate_id,
        "ordered_candidate_ids": [stronger.candidate_id, unstyled.candidate_id],
        "ordered_policy_hashes": [stronger.policy_hash, unstyled.policy_hash],
        "evaluator_hash": EVALUATOR_HASH,
        "ledger_durability": "normal",
        "required_trials": 15,
    }

    with ExperimentLedger(tmp_path / "single.sqlite3", durability="normal") as ledger:

        def plan(**options: object) -> dict:
            return _runner(contract, ledger, evaluate, **options)._build_plan(
                incumbent, challengers
            )

        assert plan()["plan_hash"] == sha256_prefixed(payload)
        assert plan(single_change=True)["plan_hash"] == sha256_prefixed(
            {**payload, "single_change": True}
        )
        assert plan(single_change=True)["campaign_id"] != plan()["campaign_id"]

        # A campaign sealed without the constraint does not reopen with it.
        unconstrained = plan(campaign_id="unconstrained")
        ledger.create_campaign(
            contract,
            incumbent.candidate_id,
            unconstrained["all_candidates"],
            plan_hash=unconstrained["plan_hash"],
            campaign_id="unconstrained",
        )
        with pytest.raises(OptimizationError, match="campaign binding drift"):
            await _runner(
                contract, ledger, evaluate, campaign_id="unconstrained", single_change=True
            ).run(incumbent, challengers)

        runner = _runner(contract, ledger, evaluate, campaign_id="single", single_change=True)
        result = await runner.run(incumbent, challengers)
        assert result.promoted is True
        assert result.selected_candidate_id == stronger.candidate_id
        assert result.optimization_plan_hash == plan(single_change=True)["plan_hash"]
        evaluated = len(calls)

        # Nor does a constrained campaign reopen without it; it still replays.
        with pytest.raises(OptimizationError, match="campaign binding drift"):
            await _runner(contract, ledger, evaluate, campaign_id="single").run(
                incumbent, challengers
            )
        replay = await runner.run(incumbent, challengers)
        assert replay.decision_id == result.decision_id
        assert len(calls) == evaluated


def test_single_change_leaves_the_default_plan_hash_unchanged(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    # Fixed bytes: a plan without the constraint keeps its earlier identity.
    golden = "sha256:37830969bc68da4e8b6a5cd8df722bef53c5c03cec600e4f63bb8a99b1c34fbc"

    with ExperimentLedger(tmp_path / "golden.sqlite3", durability="normal") as ledger:
        for options in ({}, {"single_change": False}):
            plan = _runner(contract, ledger, _challengers_win, **options)._build_plan(
                incumbent, (challenger,)
            )
            assert plan["plan_hash"] == golden
            assert plan["campaign_id"] == "optimization_v4_" + golden.removeprefix("sha256:")
        constrained = _runner(contract, ledger, _challengers_win, single_change=True)
        assert constrained._build_plan(incumbent, (challenger,))["plan_hash"] != golden


@pytest.mark.parametrize("value", [1, "yes", None])
def test_runner_rejects_a_non_boolean_single_change(tmp_path, value):
    with ExperimentLedger(tmp_path / "single-type.sqlite3", durability="normal") as ledger:
        with pytest.raises(TypeError, match="single_change must be a boolean"):
            _runner(_contract(), ledger, _challengers_win, single_change=value)


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
async def test_an_unknown_trial_keeps_no_account_identifier_from_its_error(tmp_path):
    organization = "org-" + "A1b2" * 6  # built here, so no tracked file holds an identifier
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def evaluate(_context: TrialContext) -> TrialOutcome:
        raise RuntimeError(f"429 in organization {organization} on requests per min")

    with ExperimentLedger(tmp_path / "redacted.sqlite3", durability="normal") as ledger:
        runner = _runner(contract, ledger, evaluate)
        with pytest.raises(OptimizationNeedsAttention):
            await runner.run(incumbent, (challenger,))
        (trial,) = ledger.list_trials(runner._build_plan(incumbent, (challenger,))["campaign_id"])
        assert trial.status is TrialStatus.UNKNOWN
        assert trial.error == ("evaluator raised RuntimeError after dispatch: 429 in organization "
                               "org-[redacted] on requests per min")


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


def _rewording(contract: ExperimentContract, incumbent: Candidate, text: str) -> Candidate:
    return Candidate(
        contract=contract,
        policy={"strength": 2},
        hypothesis=text,
        parent=incumbent.candidate_id,
    )


async def _challengers_win(context: TrialContext) -> TrialOutcome:
    quality = 10.0 if context.candidate.policy["strength"] >= 2 else 0.0
    return TrialOutcome(
        metrics={"quality": quality, "risk": 1.0},
        gates={"safe": True},
        actual_cost_microusd=1,
    )


@pytest.mark.asyncio
async def test_rewording_a_hypothesis_cannot_reroll_a_used_holdout(tmp_path):
    contract = _contract(max_candidates=3, max_trials=15, max_budget_microusd=150)
    incumbent, challenger = _candidates(contract)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    identity = holdout_identity(contract, EVALUATOR_HASH)
    with ExperimentLedger(tmp_path / "reroll.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None
        assert ledger.holdout_uses(identity) == {
            challenger.policy_hash: (first.campaign_id,)
        }
        evaluated = len(calls)

        reworded = _rewording(contract, incumbent, "Stronger settings help, reworded")
        assert reworded.candidate_id != challenger.candidate_id
        assert reworded.policy_hash == challenger.policy_hash
        other = Candidate(
            contract=contract,
            policy={"strength": 3},
            hypothesis="an unrelated policy",
            parent=incumbent.candidate_id,
        )
        attempts = [
            ((reworded,), None),
            ((reworded,), "fresh-campaign-id"),
            ((other, reworded), None),
        ]
        for challengers, campaign_id in attempts:
            runner = _runner(contract, ledger, evaluate, campaign_id=campaign_id)
            with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id) as refused:
                await runner.run(incumbent, challengers)
            assert challenger.policy_hash in str(refused.value)
            assert identity in str(refused.value)
        assert len(calls) == evaluated
        campaigns = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert campaigns == 1

        # The consuming campaign still replays idempotently, and an unused
        # policy under the same contract can still be tested.
        replay = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert replay.decision_id == first.decision_id
        assert len(calls) == evaluated
        fresh = await _runner(contract, ledger, evaluate).run(incumbent, (other,))
        assert fresh.holdout_assessment is not None
        assert ledger.holdout_uses(identity) == {
            challenger.policy_hash: (first.campaign_id,),
            other.policy_hash: (fresh.campaign_id,),
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "add_challenger"),
    [
        ({"max_wall_seconds": 61}, False),
        ({"max_parallel_candidates": 1}, False),
        ({"max_trials": 20, "max_budget_microusd": 200}, False),
        ({"max_candidates": 3, "max_trials": 15, "max_budget_microusd": 150}, True),
        ({"development_repetitions": 2, "max_trials": 16, "max_budget_microusd": 160}, False),
        ({"confirmation_repetitions": 4, "max_trials": 16, "max_budget_microusd": 160}, False),
        ({"base_seed": 18, "name": "renamed_engine_test"}, False),
        (
            {"mutable_field_rules": {"strength": MutableFieldRule("integer", minimum=1, maximum=9)}},
            False,
        ),
    ],
    ids=[
        "wall-time",
        "parallelism",
        "trial-and-budget-caps",
        "challenger-set",
        "development-repetitions",
        "confirmation-repetitions",
        "seed-and-name",
        "mutable-field-rules",
    ],
)
async def test_operational_contract_changes_cannot_reroll_a_used_holdout(
    tmp_path, changes, add_challenger
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    identity = holdout_identity(contract, EVALUATOR_HASH)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    with ExperimentLedger(tmp_path / "operational.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None
        evaluated = len(calls)

        variant = replace(contract, **changes)
        assert variant.contract_hash != contract.contract_hash
        assert holdout_identity(variant, EVALUATOR_HASH) == identity
        variant_incumbent, variant_challenger = _candidates(variant)
        assert variant_challenger.policy_hash == challenger.policy_hash
        challengers: tuple[Candidate, ...] = (variant_challenger,)
        if add_challenger:
            challengers = (
                Candidate(
                    contract=variant,
                    policy={"strength": 3},
                    hypothesis="an unrelated policy",
                    parent=variant_incumbent.candidate_id,
                ),
                variant_challenger,
            )
        with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id) as refused:
            await _runner(variant, ledger, evaluate).run(variant_incumbent, challengers)
        assert identity in str(refused.value)
        assert len(calls) == evaluated
        campaigns = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert campaigns == 1
        assert ledger.holdout_uses(identity) == {challenger.policy_hash: (first.campaign_id,)}


async def _challengers_win_every_gate(context: TrialContext) -> TrialOutcome:
    outcome = await _challengers_win(context)
    return TrialOutcome(
        metrics=outcome.metrics,
        gates={gate: True for gate in context.candidate.contract.required_gates},
        actual_cost_microusd=outcome.actual_cost_microusd,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "evaluator_hash"),
    [
        ({}, "sha256:" + "f" * 64),
        ({"confidence": 0.85}, EVALUATOR_HASH),
        ({"min_improvement": 0.2}, EVALUATOR_HASH),
        ({"holdout_repetitions": 4, "max_trials": 16, "max_budget_microusd": 160}, EVALUATOR_HASH),
        ({"required_gates": ("safe", "reviewed")}, EVALUATOR_HASH),
        (
            {
                "objectives": (
                    MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),
                    MetricObjective(
                        "risk", ObjectiveDirection.MINIMIZE, hard_max=4.0, max_regression=1.0
                    ),
                )
            },
            EVALUATOR_HASH,
        ),
    ],
    ids=[
        "evaluator",
        "confidence",
        "min-improvement",
        "holdout-repetitions",
        "required-gates",
        "objective-bound",
    ],
)
async def test_changing_the_holdout_evaluation_draws_a_new_holdout(
    tmp_path, changes, evaluator_hash
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    identity = holdout_identity(contract, EVALUATOR_HASH)
    with ExperimentLedger(tmp_path / "evaluation.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, _challengers_win).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None

        variant = replace(contract, **changes)
        variant_identity = holdout_identity(variant, evaluator_hash)
        assert variant_identity != identity
        variant_incumbent, variant_challenger = _candidates(variant)
        second = await OptimizationRunner(
            variant,
            ledger,
            _challengers_win_every_gate,
            evaluator_hash=evaluator_hash,
            bootstrap_resamples=100,
        ).run(variant_incumbent, (variant_challenger,))
        assert second.campaign_id != first.campaign_id
        assert second.holdout_assessment is not None
        assert ledger.holdout_uses(variant_identity) == {
            challenger.policy_hash: (second.campaign_id,)
        }
        assert ledger.holdout_uses(identity) == {challenger.policy_hash: (first.campaign_id,)}


@pytest.mark.asyncio
async def test_racing_holdout_claim_is_refused_before_any_holdout_dispatch(
    tmp_path, monkeypatch
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    reworded = _rewording(contract, incumbent, "the same policy, described differently")

    with ExperimentLedger(tmp_path / "race.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, _challengers_win).run(
            incumbent, (challenger,)
        )
        # Simulate a concurrent campaign that passed the early check before
        # the first campaign claimed the holdout.
        monkeypatch.setattr(OptimizationRunner, "_require_unused_holdouts", lambda *_: None)
        runner = _runner(contract, ledger, _challengers_win)
        with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id):
            await runner.run(incumbent, (reworded,))
        campaign_id = runner._build_plan(incumbent, (reworded,))["campaign_id"]
        trials = ledger.list_trials(campaign_id)
        assert {trial.split for trial in trials} == {"development", "confirmation"}
        assert all(trial.status is TrialStatus.COMPLETED for trial in trials)
        assert ledger.snapshot(campaign_id)["decision_count"] == 0


def _two_challengers(contract: ExperimentContract) -> tuple[Candidate, Candidate, Candidate]:
    incumbent, best = _candidates(contract)
    runner_up = Candidate(
        contract=contract,
        policy={"strength": 3},
        hypothesis="a weaker but viable policy",
        parent=incumbent.candidate_id,
    )
    return incumbent, best, runner_up


def _staged_evaluator(confirmation: float, holdout: float):
    """Development ranks strength 2 first; later splits use the given quality."""

    async def evaluate(context: TrialContext) -> TrialOutcome:
        strength = context.candidate.policy["strength"]
        quality = {1: 0.0, 2: 10.0, 3: 5.0}[strength]
        if strength == 2 and context.split != "development":
            quality = confirmation if context.split == "confirmation" else holdout
        return TrialOutcome(
            metrics={"quality": quality, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    return evaluate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("confirmation", "holdout", "verdicts", "promoted"),
    [
        (10.0, 10.0, ("improved", "improved", "improved"), True),
        (10.0, -10.0, ("improved", "worse", "worse"), False),
        (0.0, 10.0, ("inconclusive", None, "inconclusive"), False),
        (-10.0, 10.0, ("worse", None, "worse"), False),
    ],
    ids=["holdout-improved", "holdout-worse", "confirmation-inconclusive", "confirmation-worse"],
)
async def test_terminal_decision_keeps_every_development_score_and_its_verdict(
    tmp_path, confirmation, holdout, verdicts, promoted
):
    contract = _contract(max_candidates=3, max_trials=15, max_budget_microusd=150)
    incumbent, best, runner_up = _two_challengers(contract)
    confirmation_verdict, holdout_verdict, final_verdict = verdicts

    with ExperimentLedger(tmp_path / "verdict.sqlite3", durability="normal") as ledger:
        result = await _runner(
            contract, ledger, _staged_evaluator(confirmation, holdout)
        ).run(incumbent, (best, runner_up))
        assert result.promoted is promoted
        assert result.selected_candidate_id == best.candidate_id
        assert result.verdict == final_verdict
        assert result.confirmation_assessment.verdict == confirmation_verdict
        if holdout_verdict is None:
            assert result.holdout_assessment is None
        else:
            assert result.holdout_assessment.verdict == holdout_verdict

        assessment = ledger.list_decisions(result.campaign_id)[0]["assessment"]
        assert assessment["runner_version"] == 4
        assert assessment["verdict"] == final_verdict
        assert assessment["confirmation"]["verdict"] == confirmation_verdict
        if holdout_verdict is not None:
            assert assessment["holdout"]["verdict"] == holdout_verdict
        scores = assessment["development_scores"]
        assert [score["candidate_id"] for score in scores] == [
            best.candidate_id,
            runner_up.candidate_id,
        ]
        assert all(score["viable"] for score in scores)
        assert scores == result.to_dict()["development_scores"]
        assert assessment["development"] == scores[0]
        assert len(ledger.validate_decision_inventory(result.campaign_id)) == 1

        data = result.to_dict()
        assert data["verdict"] == final_verdict
        assert data["confirmation_assessment"]["verdict"] == confirmation_verdict


@pytest.mark.asyncio
async def test_development_only_campaign_verdict_is_inconclusive(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def unsafe_challenger(context: TrialContext) -> TrialOutcome:
        return TrialOutcome(
            metrics={"quality": 10.0, "risk": 1.0},
            gates={"safe": context.candidate == incumbent},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "development.sqlite3", durability="normal") as ledger:
        result = await _runner(contract, ledger, unsafe_challenger).run(
            incumbent, (challenger,)
        )
        assert result.selected_candidate_id is None
        assert result.confirmation_assessment is None
        assert (result.verdict, result.to_dict()["verdict"]) == ("inconclusive",) * 2
        assessment = ledger.list_decisions(result.campaign_id)[0]["assessment"]
        assert assessment["stage"] == "development"
        assert assessment["runner_version"] == 4
        # The decision records the campaign's verdict, as a paired decision does.
        assert assessment["verdict"] == "inconclusive"
        assert len(ledger.validate_decision_inventory(result.campaign_id)) == 1
        assert [score["candidate_id"] for score in assessment["development_scores"]] == [
            challenger.candidate_id
        ]


@pytest.mark.asyncio
async def test_replaying_a_completed_campaign_appends_a_byte_identical_decision(tmp_path):
    contract = _contract(max_candidates=3, max_trials=15, max_budget_microusd=150)
    incumbent, best, runner_up = _two_challengers(contract)
    calls: list[TrialContext] = []
    evaluate = _staged_evaluator(10.0, 10.0)

    async def counted(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await evaluate(context)

    def stored(ledger: ExperimentLedger) -> list[tuple[str, str]]:
        return [
            (row["decision_id"], row["payload_json"])
            for row in ledger._connection.execute(
                "SELECT decision_id, payload_json FROM promotion_decisions"
            )
        ]

    path = tmp_path / "replay.sqlite3"
    with ExperimentLedger(path, durability="normal") as ledger:
        first = await _runner(contract, ledger, counted).run(incumbent, (best, runner_up))
        before = stored(ledger)
    evaluated = len(calls)
    assert evaluated == 15
    assert [decision_id for decision_id, _ in before] == [first.decision_id]
    payload = json.loads(before[0][1])
    assert payload["assessment"]["runner_version"] == 4
    assert len(payload["assessment"]["development_scores"]) == 2
    assert payload["assessment"]["verdict"] == "improved"

    with ExperimentLedger(path, durability="normal") as ledger:
        replay = await _runner(contract, ledger, counted).run(incumbent, (best, runner_up))
        assert stored(ledger) == before
    assert len(calls) == evaluated
    assert replay.decision_id == first.decision_id
    assert replay.to_dict() == first.to_dict()


@pytest.mark.asyncio
async def test_campaign_from_runner_version_3_is_refused_as_binding_drift(
    tmp_path, monkeypatch
):
    from smythe.optimize import engine

    contract = _contract()
    incumbent, challenger = _candidates(contract)
    identity = holdout_identity(contract, EVALUATOR_HASH)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    current = engine.RUNNER_VERSION
    assert current == 4
    with ExperimentLedger(tmp_path / "upgrade.sqlite3", durability="normal") as ledger:
        with monkeypatch.context() as earlier_release:
            earlier_release.setattr(engine, "RUNNER_VERSION", 3)
            legacy_runner = _runner(contract, ledger, evaluate, campaign_id="explicit")
            legacy_plan = legacy_runner._build_plan(incumbent, (challenger,))
            legacy_auto = _runner(contract, ledger, evaluate)._build_plan(
                incumbent, (challenger,)
            )
            legacy = await legacy_runner.run(incumbent, (challenger,))
        assert legacy.holdout_assessment is not None
        assert ledger.list_decisions("explicit")[0]["assessment"]["runner_version"] == 3
        evaluated = len(calls)

        upgraded = _runner(contract, ledger, evaluate, campaign_id="explicit")
        plan = upgraded._build_plan(incumbent, (challenger,))
        assert plan["plan_hash"] != legacy_plan["plan_hash"]
        with pytest.raises(OptimizationError, match="campaign binding drift"):
            await upgraded.run(incumbent, (challenger,))

        # The runner version is not part of the holdout identity, so the
        # earlier campaign's holdout stays used.
        automatic = _runner(contract, ledger, evaluate)
        auto_plan = automatic._build_plan(incumbent, (challenger,))
        assert legacy_auto["campaign_id"].startswith("optimization_v3_")
        assert auto_plan["campaign_id"].startswith("optimization_v4_")
        with pytest.raises(HoldoutAlreadyUsedError, match="explicit"):
            await automatic.run(incumbent, (challenger,))
        assert len(calls) == evaluated
        assert ledger.holdout_uses(identity) == {challenger.policy_hash: ("explicit",)}
        assert ledger.snapshot("explicit")["decision_count"] == 1


@pytest.mark.asyncio
async def test_a_changed_task_pack_in_the_evaluator_manifest_draws_a_new_holdout(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    packs = {
        "development": "sha256:" + "1" * 64,
        "confirmation": "sha256:" + "2" * 64,
        "holdout": "sha256:" + "3" * 64,
    }
    manifest = EvaluatorManifest(code_digest=EVALUATOR_HASH, split_manifests=packs)
    respelled = EvaluatorManifest(
        code_digest=EVALUATOR_HASH, split_manifests=dict(reversed(packs.items()))
    )
    new_pack = EvaluatorManifest(
        code_digest=EVALUATOR_HASH,
        split_manifests={**packs, "holdout": "sha256:" + "4" * 64},
    )

    def runner(ledger: ExperimentLedger, evaluator_hash: str) -> OptimizationRunner:
        return OptimizationRunner(
            contract,
            ledger,
            _challengers_win,
            evaluator_hash=evaluator_hash,
            bootstrap_resamples=100,
        )

    identity = holdout_identity(contract, manifest.evaluator_hash)
    with ExperimentLedger(tmp_path / "manifest.sqlite3", durability="normal") as ledger:
        first = await runner(ledger, manifest.evaluator_hash).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None
        assert {
            trial.evaluator_hash for trial in ledger.list_trials(first.campaign_id)
        } == {manifest.evaluator_hash}

        # The same manifest spelled in another key order is the same evaluator.
        assert respelled.evaluator_hash == manifest.evaluator_hash
        replay = await runner(ledger, respelled.evaluator_hash).run(incumbent, (challenger,))
        assert replay.decision_id == first.decision_id

        # A changed holdout task pack is a different evaluation.
        new_identity = holdout_identity(contract, new_pack.evaluator_hash)
        assert new_identity != identity
        second = await runner(ledger, new_pack.evaluator_hash).run(incumbent, (challenger,))
        assert second.campaign_id != first.campaign_id
        assert second.holdout_assessment is not None
        assert ledger.holdout_uses(identity) == {challenger.policy_hash: (first.campaign_id,)}
        assert ledger.holdout_uses(new_identity) == {
            challenger.policy_hash: (second.campaign_id,)
        }


async def _record_contexts(
    path, contract: ExperimentContract, incumbent: Candidate, challenger: Candidate, **options
) -> list[TrialContext]:
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    with ExperimentLedger(path, durability="normal") as ledger:
        result = await _runner(contract, ledger, evaluate, **options).run(
            incumbent, (challenger,)
        )
    assert result.promoted is True
    return calls


def _assigned(
    calls: list[TrialContext], split: str, candidate: Candidate
) -> list[tuple[int, int | None]]:
    """(seed, task_index) for one candidate's trials in one split, in seed order."""

    return [
        (context.seed, context.task_index)
        for context in calls
        if context.split == split and context.candidate == candidate
    ]


@pytest.mark.asyncio
async def test_task_index_is_deterministic_and_never_repeats_within_a_split(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    pools = {"development": 2, "confirmation": 3, "holdout": 5}

    first = await _record_contexts(
        tmp_path / "first.sqlite3", contract, incumbent, challenger, task_pools=pools
    )
    again = await _record_contexts(
        tmp_path / "again.sqlite3", contract, incumbent, challenger, task_pools=pools
    )
    unpooled = await _record_contexts(
        tmp_path / "unpooled.sqlite3", contract, incumbent, challenger
    )

    assert {context.task_index for context in unpooled} == {None}
    for calls in (first, again):
        for split, pool in pools.items():
            assigned = _assigned(calls, split, challenger)
            seeds = tuple(seed for seed, _ in assigned)
            indices = tuple(index for _, index in assigned)
            # Each seed of a split names a different task inside its pool,
            # and the incumbent's paired trial uses the same task.
            assert len(set(indices)) == len(indices) == len(seeds)
            assert all(0 <= index < pool for index in indices)
            assert _assigned(calls, split, incumbent) == assigned
            assert indices == _task_indices(split, seeds, pool)
    # Development and confirmation seeds come from the contract, so a fresh
    # ledger repeats their tasks; holdout seeds come from each campaign's
    # sealed secret, and its tasks follow them.
    for split in ("development", "confirmation"):
        assert _assigned(again, split, challenger) == _assigned(first, split, challenger)


def test_task_indices_follow_a_fixed_derivation():
    seeds = (11, 22, 33)

    # Fixed values: a resumed campaign must keep drawing the same tasks.
    assert _task_indices("holdout", seeds, 10) == (1, 4, 9)
    assert _task_indices("confirmation", seeds, 10) == (3, 4, 5)
    assert _task_indices("holdout", seeds, None) == (None, None, None)
    assert sorted(_task_indices("development", (5, 6, 7, 8), 4)) == [0, 1, 2, 3]
    large = _task_indices("holdout", tuple(range(1_000)), 2**62)
    assert len(set(large)) == 1_000
    assert all(0 <= index < 2**62 for index in large)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pools",
    [{"holdout": 2}, {"confirmation": 2}, {"development": 1, "holdout": 2}],
    ids=["holdout", "confirmation", "two-splits"],
)
async def test_plan_with_more_repetitions_than_its_task_pool_is_refused(tmp_path, pools):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    async def never_called(_context: TrialContext) -> TrialOutcome:
        raise AssertionError("a refused plan must not invoke the evaluator")

    with ExperimentLedger(tmp_path / "pool.sqlite3", durability="normal") as ledger:
        runner = _runner(contract, ledger, never_called, task_pools=pools)
        with pytest.raises(OptimizationLimitError, match="task pool"):
            await runner.run(incumbent, (challenger,))
        count = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert count == 0


@pytest.mark.parametrize(
    ("pools", "error"),
    [
        ({"regression": 3}, ValueError),
        ({"holdout": 0}, ValueError),
        ({"holdout": True}, ValueError),
        ({"holdout": 3.0}, ValueError),
        ({"holdout": 2**63}, ValueError),
        ([("holdout", 3)], TypeError),
    ],
)
def test_runner_rejects_invalid_task_pools(tmp_path, pools, error):
    with ExperimentLedger(tmp_path / "invalid-pool.sqlite3") as ledger:
        with pytest.raises(error, match="task_pools"):
            _runner(_contract(), ledger, _challengers_win, task_pools=pools)


def test_task_pools_and_draw_cap_join_the_plan_hash_only_when_declared(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)

    with ExperimentLedger(tmp_path / "plan.sqlite3", durability="normal") as ledger:

        def plan_hash(**options: object) -> str:
            runner = _runner(contract, ledger, _challengers_win, **options)
            return runner._build_plan(incumbent, (challenger,))["plan_hash"]

        # Plans that declare neither setting keep their earlier bytes.
        legacy = sha256_prefixed(
            {
                "runner_version": 4,
                "statistics": {
                    "method": PROMOTION_METHOD,
                    "descriptive_bootstrap_resamples": 100,
                },
                "contract_hash": contract.contract_hash,
                "incumbent_candidate_id": incumbent.candidate_id,
                "ordered_candidate_ids": [challenger.candidate_id],
                "ordered_policy_hashes": [challenger.policy_hash],
                "evaluator_hash": EVALUATOR_HASH,
                "ledger_durability": "normal",
                "required_trials": 14,
            }
        )
        assert plan_hash() == legacy
        assert plan_hash(task_pools={}) == legacy
        pooled = plan_hash(task_pools={"holdout": 3})
        assert pooled != legacy
        assert plan_hash(task_pools={"holdout": 4}) != pooled
        assert plan_hash(task_pools={"holdout": 3, "development": 5}) != pooled

        assert plan_hash(max_holdout_draws=None) == legacy
        capped = plan_hash(max_holdout_draws=1)
        assert capped not in {legacy, pooled}
        assert plan_hash(max_holdout_draws=2) != capped


@pytest.mark.parametrize("cap", [0, -1, True, 1.0, 2**63])
def test_runner_rejects_an_invalid_holdout_draw_cap(tmp_path, cap):
    with ExperimentLedger(tmp_path / "invalid-cap.sqlite3", durability="normal") as ledger:
        with pytest.raises(ValueError, match="max_holdout_draws"):
            _runner(_contract(), ledger, _challengers_win, max_holdout_draws=cap)


def _second_policy(contract: ExperimentContract, incumbent: Candidate, strength: int) -> Candidate:
    return Candidate(
        contract=contract,
        policy={"strength": strength},
        hypothesis=f"strength {strength} helps",
        parent=incumbent.candidate_id,
    )


@pytest.mark.asyncio
async def test_holdout_draw_cap_refuses_another_campaign_even_for_a_different_policy(
    tmp_path,
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    second, third = (_second_policy(contract, incumbent, value) for value in (3, 4))
    identity = holdout_identity(contract, EVALUATOR_HASH)
    calls: list[TrialContext] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return await _challengers_win(context)

    with ExperimentLedger(tmp_path / "draws.sqlite3", durability="normal") as ledger:
        # A draw made without a cap still counts against a later cap.
        first = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert first.holdout_assessment is not None
        assert ledger.holdout_draw_campaigns(identity) == (first.campaign_id,)
        evaluated = len(calls)

        for campaign_id in (None, "fresh-campaign-id"):
            runner = _runner(
                contract, ledger, evaluate, campaign_id=campaign_id, max_holdout_draws=1
            )
            with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id) as refused:
                await runner.run(incumbent, (second,))
            assert identity in str(refused.value)
            assert "finite holdout task pool" in str(refused.value)
        assert len(calls) == evaluated
        campaigns = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert campaigns == 1

        # A larger cap admits one more draw, and a campaign that drew still
        # replays once the cap is reached.
        admitted = await _runner(contract, ledger, evaluate, max_holdout_draws=2).run(
            incumbent, (second,)
        )
        assert admitted.holdout_assessment is not None
        assert ledger.holdout_draw_campaigns(identity) == (
            first.campaign_id,
            admitted.campaign_id,
        )
        evaluated = len(calls)
        replay = await _runner(contract, ledger, evaluate, max_holdout_draws=2).run(
            incumbent, (second,)
        )
        assert replay.decision_id == admitted.decision_id
        assert len(calls) == evaluated
        with pytest.raises(HoldoutAlreadyUsedError, match="finite holdout task pool"):
            await _runner(contract, ledger, evaluate, max_holdout_draws=2).run(
                incumbent, (third,)
            )
        assert len(calls) == evaluated


@pytest.mark.asyncio
async def test_racing_holdout_draw_is_refused_before_any_holdout_dispatch(
    tmp_path, monkeypatch
):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    second = _second_policy(contract, incumbent, 3)

    with ExperimentLedger(tmp_path / "draw-race.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, _challengers_win, max_holdout_draws=1).run(
            incumbent, (challenger,)
        )
        # Simulate a campaign that passed the early check before the first
        # campaign drew the last holdout.
        monkeypatch.setattr(OptimizationRunner, "_require_unused_holdouts", lambda *_: None)
        runner = _runner(contract, ledger, _challengers_win, max_holdout_draws=1)
        with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id):
            await runner.run(incumbent, (second,))
        campaign_id = runner._build_plan(incumbent, (second,))["campaign_id"]
        trials = ledger.list_trials(campaign_id)
        assert {trial.split for trial in trials} == {"development", "confirmation"}
        assert all(trial.status is TrialStatus.COMPLETED for trial in trials)
        assert ledger.snapshot(campaign_id)["decision_count"] == 0


@pytest.mark.asyncio
async def test_task_pools_and_draw_cap_leave_existing_holdout_uses_unchanged(tmp_path):
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    reworded = _rewording(contract, incumbent, "the same policy under a cap")
    identity = holdout_identity(contract, EVALUATOR_HASH)

    with ExperimentLedger(tmp_path / "unchanged.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, _challengers_win).run(incumbent, (challenger,))
        uses = ledger.holdout_uses(identity)
        assert uses == {challenger.policy_hash: (first.campaign_id,)}

        # Neither setting enters the holdout identity, so a used policy stays
        # used: the policy refusal comes first, with its usual message.
        for options in (
            {"max_holdout_draws": 5},
            {"task_pools": {"holdout": 5}},
            {"task_pools": {"holdout": 5}, "max_holdout_draws": 5},
        ):
            with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id) as refused:
                await _runner(contract, ledger, _challengers_win, **options).run(
                    incumbent, (reworded,)
                )
            assert challenger.policy_hash in str(refused.value)
            assert "finite holdout task pool" not in str(refused.value)
        assert ledger.holdout_uses(identity) == uses
        assert ledger.holdout_draw_campaigns(identity) == (first.campaign_id,)


def _reference_arms(
    contract: ExperimentContract, incumbent: Candidate, *strengths: int
) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            contract=contract,
            policy={"strength": strength},
            hypothesis=f"a simpler reference with strength {strength}",
            parent=incumbent.candidate_id,
        )
        for strength in strengths
    )


def _arm_evaluator(quality: Mapping[int, object], calls: list[TrialContext] | None = None):
    """Score each policy by strength; a mapping value scores by split."""

    async def evaluate(context: TrialContext) -> TrialOutcome:
        if calls is not None:
            calls.append(context)
        value = quality[context.candidate.policy["strength"]]
        if isinstance(value, Mapping):
            value = value[context.split]
        return TrialOutcome(
            metrics={"quality": value, "risk": 1.0},
            gates={"safe": True},
            actual_cost_microusd=1,
        )

    return evaluate


# Two development trials, then four arms on three confirmation and three
# holdout seeds each.
_THREE_ARM_LIMITS = {"max_candidates": 4, "max_trials": 26, "max_budget_microusd": 260}


@pytest.mark.asyncio
async def test_three_arm_campaign_compares_the_challenger_with_every_reference(tmp_path):
    contract = _contract(**_THREE_ARM_LIMITS)
    incumbent, challenger = _candidates(contract)
    simpler, stronger = _reference_arms(contract, incumbent, 3, 4)
    calls: list[TrialContext] = []
    evaluate = _arm_evaluator({1: 0.0, 2: 10.0, 3: 4.0, 4: 20.0}, calls)
    pools = {"confirmation": 3, "holdout": 3}
    identity = holdout_identity(contract, EVALUATOR_HASH)

    with ExperimentLedger(tmp_path / "references.sqlite3", durability="normal") as ledger:
        result = await _runner(contract, ledger, evaluate, task_pools=pools).run(
            incumbent, (challenger,), references=(simpler, stronger)
        )
        # The default rule reports references without changing promotion.
        assert result.promoted is True
        assert result.verdict == "improved"
        assert [item["candidate_id"] for item in result.references] == [
            simpler.candidate_id,
            stronger.candidate_id,
        ]
        assert [item["verdict"] for item in result.references] == ["improved", "worse"]
        for item in result.references:
            assert item["confirmation"]["primary"]["sample_count"] == 3
            assert item["holdout"]["primary"]["sample_count"] == 3
            assert "reference.safe" in item["holdout"]["gates"]

        snapshot = ledger.snapshot(result.campaign_id)
        assert snapshot["campaign"]["ordered_candidate_ids"] == [
            incumbent.candidate_id,
            challenger.candidate_id,
            simpler.candidate_id,
            stronger.candidate_id,
        ]
        assert snapshot["trial_counts"] == {"completed": 26}
        assessment = snapshot["decisions"][0]["assessment"]
        assert assessment["reference_rule"] == "report"
        assert assessment["references"] == result.to_dict()["references"]
        assert assessment["verdict"] == "improved"
        assert len(ledger.validate_decision_inventory(result.campaign_id)) == 1

        # References run on exactly the incumbent's and challenger's seeds and
        # tasks, under their own phase, and never in development.
        for split in ("confirmation", "holdout"):
            expected = _assigned(calls, split, challenger)
            assert len(expected) == 3
            for arm in (incumbent, simpler, stronger):
                assert _assigned(calls, split, arm) == expected
        assert {
            context.phase for context in calls if context.candidate in (simpler, stronger)
        } == {_bound_phase("reference", result.optimization_plan_hash)}
        assert not [
            context
            for context in calls
            if context.split == "development" and context.candidate in (simpler, stronger)
        ]

        # Reference holdout trials draw from the pool but are not holdout use.
        assert ledger.holdout_uses(identity) == {
            challenger.policy_hash: (result.campaign_id,)
        }
        assert ledger.holdout_draw_campaigns(identity) == (result.campaign_id,)

        evaluated = len(calls)
        replay = await _runner(contract, ledger, evaluate, task_pools=pools).run(
            incumbent, (challenger,), references=(simpler, stronger)
        )
        assert len(calls) == evaluated
        assert replay.decision_id == result.decision_id
        assert replay.to_dict() == result.to_dict()

    # A fresh ledger seals a new holdout; the confirmation comparisons repeat.
    with ExperimentLedger(tmp_path / "again.sqlite3", durability="normal") as ledger:
        again = await _runner(contract, ledger, evaluate, task_pools=pools).run(
            incumbent, (challenger,), references=(simpler, stronger)
        )
    assert again.campaign_id == result.campaign_id
    assert [item["verdict"] for item in again.references] == ["improved", "worse"]
    assert [item["confirmation"] for item in again.references] == [
        item["confirmation"] for item in result.references
    ]


@pytest.mark.asyncio
async def test_a_policy_whose_holdout_was_used_can_be_a_reference_baseline(tmp_path):
    contract = _contract(**_THREE_ARM_LIMITS)
    incumbent, predecessor = _candidates(contract)
    successor = _second_policy(contract, incumbent, 3)
    evaluate = _arm_evaluator({1: 0.0, 2: 5.0, 3: 10.0})
    identity = holdout_identity(contract, EVALUATOR_HASH)

    with ExperimentLedger(tmp_path / "baseline.sqlite3", durability="normal") as ledger:
        first = await _runner(contract, ledger, evaluate).run(incumbent, (predecessor,))
        assert first.promoted is True
        with pytest.raises(HoldoutAlreadyUsedError, match=first.campaign_id):
            await _runner(contract, ledger, evaluate).run(
                incumbent, (_rewording(contract, incumbent, "the predecessor again"),)
            )

        # The used predecessor is a second baseline for its successor.
        second = await _runner(contract, ledger, evaluate).run(
            incumbent, (successor,), references=(predecessor,)
        )
        assert second.promoted is True
        assert second.references[0]["candidate_id"] == predecessor.candidate_id
        assert second.references[0]["verdict"] == "improved"
        assert ledger.holdout_uses(identity) == {
            predecessor.policy_hash: (first.campaign_id,),
            successor.policy_hash: (second.campaign_id,),
        }
        assert ledger.holdout_draw_campaigns(identity) == (
            first.campaign_id,
            second.campaign_id,
        )

        # Without reference arms, that comparison re-tests the successor in a
        # second campaign, which is refused.
        with pytest.raises(HoldoutAlreadyUsedError, match=second.campaign_id):
            await _runner(contract, ledger, evaluate).run(predecessor, (successor,))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference_quality", "rejected_at"),
    [
        (20.0, "confirmation"),
        ({"confirmation": 5.0, "holdout": 20.0}, "holdout"),
        (5.0, None),
    ],
    ids=["better-on-confirmation", "better-on-holdout", "not-better"],
)
async def test_must_not_be_worse_blocks_promotion_when_a_reference_is_better(
    tmp_path, reference_quality, rejected_at
):
    contract = _contract(max_candidates=3, max_trials=20, max_budget_microusd=200)
    incumbent, challenger = _candidates(contract)
    (reference,) = _reference_arms(contract, incumbent, 3)
    evaluate = _arm_evaluator({1: 0.0, 2: 10.0, 3: reference_quality})
    identity = holdout_identity(contract, EVALUATOR_HASH)

    with ExperimentLedger(tmp_path / "report.sqlite3", durability="normal") as ledger:
        reported = await _runner(contract, ledger, evaluate).run(
            incumbent, (challenger,), references=(reference,)
        )
    assert reported.promoted is True

    with ExperimentLedger(tmp_path / "rule.sqlite3", durability="normal") as ledger:
        result = await _runner(contract, ledger, evaluate).run(
            incumbent,
            (challenger,),
            references=(reference,),
            reference_rule="must_not_be_worse",
        )
        assert result.campaign_id != reported.campaign_id
        assert result.confirmation_assessment.promote is True
        assessment = ledger.list_decisions(result.campaign_id)[0]["assessment"]
        assert assessment["reference_rule"] == "must_not_be_worse"
        assert len(ledger.validate_decision_inventory(result.campaign_id)) == 1
        if rejected_at is None:
            assert result.promoted is True
            assert result.references[0]["verdict"] == "improved"
            return
        assert result.promoted is False
        assert result.reason == (
            f"{rejected_at} rejected candidate: reference {reference.candidate_id} "
            "is better on the primary objective"
        )
        # The challenger's own comparison with the incumbent still improved.
        assert result.verdict == "improved"
        assert result.references[0]["verdict"] == "worse"
        if rejected_at == "confirmation":
            # A blocked confirmation never reaches, or uses, the holdout.
            assert result.holdout_assessment is None
            assert result.references[0]["holdout"] is None
            assert {trial.split for trial in ledger.list_trials(result.campaign_id)} == {
                "development",
                "confirmation",
            }
            assert ledger.holdout_uses(identity) == {}
        else:
            assert result.holdout_assessment.promote is True
            assert ledger.holdout_uses(identity) == {
                challenger.policy_hash: (result.campaign_id,)
            }


@pytest.mark.asyncio
async def test_a_campaign_stopped_in_development_never_runs_its_references(tmp_path):
    contract = _contract(max_candidates=3, max_trials=20, max_budget_microusd=200)
    incumbent, challenger = _candidates(contract)
    (reference,) = _reference_arms(contract, incumbent, 3)
    calls: list[TrialContext] = []

    async def unsafe_challenger(context: TrialContext) -> TrialOutcome:
        calls.append(context)
        return TrialOutcome(
            metrics={"quality": 10.0, "risk": 1.0},
            gates={"safe": context.candidate != challenger},
            actual_cost_microusd=1,
        )

    with ExperimentLedger(tmp_path / "stopped.sqlite3", durability="normal") as ledger:
        runner = _runner(contract, ledger, unsafe_challenger)
        result = await runner.run(incumbent, (challenger,), references=(reference,))
        assert result.selected_candidate_id is None
        assert result.references == (
            {"candidate_id": reference.candidate_id, "policy_hash": reference.policy_hash},
        )
        assert {context.candidate.candidate_id for context in calls} == {
            incumbent.candidate_id,
            challenger.candidate_id,
        }
        assessment = ledger.list_decisions(result.campaign_id)[0]["assessment"]
        assert assessment["stage"] == "development"
        assert assessment["references"] == list(result.references)
        assert len(ledger.validate_decision_inventory(result.campaign_id)) == 1
        replay = await _runner(contract, ledger, unsafe_challenger).run(
            incumbent, (challenger,), references=(reference,)
        )
        assert replay.decision_id == result.decision_id
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_two_arm_campaigns_are_unchanged_by_reference_support(tmp_path):
    contract = _contract(**_THREE_ARM_LIMITS)
    incumbent, challenger = _candidates(contract)
    (reference,) = _reference_arms(contract, incumbent, 3)
    evaluate = _arm_evaluator({1: 0.0, 2: 10.0, 3: 5.0})
    two_arm_payload = {
        "runner_version": 4,
        "statistics": {"method": PROMOTION_METHOD, "descriptive_bootstrap_resamples": 100},
        "contract_hash": contract.contract_hash,
        "incumbent_candidate_id": incumbent.candidate_id,
        "ordered_candidate_ids": [challenger.candidate_id],
        "ordered_policy_hashes": [challenger.policy_hash],
        "evaluator_hash": EVALUATOR_HASH,
        "ledger_durability": "normal",
        "required_trials": 14,
    }

    with ExperimentLedger(tmp_path / "two-arm.sqlite3", durability="normal") as ledger:

        def plan(*references: Candidate, **options: object) -> dict:
            runner = _runner(contract, ledger, evaluate)
            return runner._build_plan(incumbent, (challenger,), references, **options)

        legacy = sha256_prefixed(two_arm_payload)
        assert plan()["plan_hash"] == legacy
        # A rule without references has nothing to apply to.
        assert plan(reference_rule="must_not_be_worse")["plan_hash"] == legacy
        referenced = plan(reference)
        assert referenced["required_trials"] == 14 + 3 + 3
        assert referenced["plan_hash"] == sha256_prefixed(
            {
                **two_arm_payload,
                "required_trials": 20,
                "ordered_reference_ids": [reference.candidate_id],
                "reference_rule": "report",
            }
        )
        assert plan(reference, reference_rule="must_not_be_worse")["plan_hash"] not in {
            legacy,
            referenced["plan_hash"],
        }

        result = await _runner(contract, ledger, evaluate).run(incumbent, (challenger,))
        assert result.optimization_plan_hash == legacy
        assert result.references == ()
        assert "references" not in result.to_dict()
        assessment = ledger.list_decisions(result.campaign_id)[0]["assessment"]
        assert "references" not in assessment
        assert "reference_rule" not in assessment
        assert {
            trial.phase.split(".")[0] for trial in ledger.list_trials(result.campaign_id)
        } == {"incumbent", "challenger"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ("incumbent", OptimizationError, "differ from the incumbent"),
        ("challenger", OptimizationError, "differ from the incumbent"),
        ("reworded-challenger", OptimizationError, "differ from the incumbent"),
        ("duplicate", OptimizationError, "reference candidate IDs"),
        ("not-a-candidate", TypeError, "references"),
        ("string", TypeError, "references"),
        ("rule", ValueError, "reference_rule"),
        ("max-candidates", OptimizationLimitError, "max_candidates"),
        ("max-trials", OptimizationLimitError, "requires 26 trials"),
    ],
)
async def test_invalid_reference_plans_are_refused_before_any_campaign(
    tmp_path, change, error, message
):
    limits = dict(_THREE_ARM_LIMITS)
    if change == "max-candidates":
        limits["max_candidates"] = 3
    if change == "max-trials":
        limits["max_trials"] = 25
    contract = _contract(**limits)
    incumbent, challenger = _candidates(contract)
    first, second = _reference_arms(contract, incumbent, 3, 4)
    references: object = {
        "incumbent": (first, incumbent),
        "challenger": (challenger,),
        "reworded-challenger": (_rewording(contract, incumbent, "the challenger policy"),),
        "duplicate": (first, first),
        "not-a-candidate": (first, {"strength": 4}),
        "string": "references",
    }.get(change, (first, second))
    options = {"reference_rule": "must_beat"} if change == "rule" else {}

    async def never_called(_context: TrialContext) -> TrialOutcome:
        raise AssertionError("a refused plan must not invoke the evaluator")

    with ExperimentLedger(tmp_path / "refused.sqlite3", durability="normal") as ledger:
        with pytest.raises(error, match=message):
            await _runner(contract, ledger, never_called).run(
                incumbent, (challenger,), references=references, **options
            )
        count = ledger._connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
        assert count == 0
