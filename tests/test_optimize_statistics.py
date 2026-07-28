"""Pure, deterministic promotion-statistics tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import time

import pytest

from smythe.optimize.contracts import MetricObjective, ObjectiveDirection
from smythe.optimize.statistics import (
    MAX_BOOTSTRAP_RESAMPLES,
    aggregate_mean,
    assess_promotion,
    bootstrap_confidence_interval,
    bootstrap_lower_bound,
    compare_metric,
    paired_improvements,
)


SEEDS = (101, 202, 303, 404, 505)


def _primary(*, direction=ObjectiveDirection.MAXIMIZE, **kwargs) -> MetricObjective:
    return MetricObjective(name="quality", direction=direction, primary=True, **kwargs)


def test_paired_improvement_normalizes_maximize_and_minimize_directions():
    baseline = (1.0, 4.0, 8.0)
    candidate = (2.5, 3.0, 8.5)
    seeds = SEEDS[:3]

    assert paired_improvements(
        baseline,
        candidate,
        ObjectiveDirection.MAXIMIZE,
        baseline_seeds=seeds,
        candidate_seeds=seeds,
    ) == (1.5, -1.0, 0.5)
    assert paired_improvements(
        baseline,
        candidate,
        ObjectiveDirection.MINIMIZE,
        baseline_seeds=seeds,
        candidate_seeds=seeds,
    ) == (-1.5, 1.0, -0.5)


def test_pairing_rejects_length_seed_and_order_mismatches():
    with pytest.raises(ValueError, match="equal length"):
        paired_improvements(
            (1.0, 2.0),
            (1.0,),
            ObjectiveDirection.MAXIMIZE,
            baseline_seeds=(1, 2),
            candidate_seeds=(1,),
        )
    with pytest.raises(ValueError, match="same order"):
        paired_improvements(
            (1.0, 2.0, 3.0),
            (2.0, 3.0, 4.0),
            ObjectiveDirection.MAXIMIZE,
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 3, 2),
        )
    with pytest.raises(ValueError, match="unique"):
        paired_improvements(
            (1.0, 2.0, 3.0),
            (2.0, 3.0, 4.0),
            ObjectiveDirection.MAXIMIZE,
            baseline_seeds=(1, 1, 3),
            candidate_seeds=(1, 1, 3),
        )


def test_bootstrap_interval_and_lower_bound_are_seed_deterministic():
    deltas = (0.25, 0.5, 0.75, 1.0, 1.25)
    first = bootstrap_confidence_interval(
        deltas, confidence=0.95, resamples=500, seed=1729
    )
    second = bootstrap_confidence_interval(
        deltas, confidence=0.95, resamples=500, seed=1729
    )

    assert first == second
    assert first[0] <= aggregate_mean(deltas) <= first[1]
    assert bootstrap_lower_bound(
        deltas, confidence=0.95, resamples=500, seed=1729
    ) == first[0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((1.0, float("nan"), 2.0), "finite"),
        ((1.0, float("inf"), 2.0), "finite"),
        ((), "must not be empty"),
    ],
)
def test_aggregate_mean_rejects_invalid_samples(values, message):
    with pytest.raises(ValueError, match=message):
        aggregate_mean(values)


def test_numeric_overflow_is_rejected_as_validation_error():
    huge_integer = 10**10_000
    with pytest.raises(ValueError, match="finite"):
        aggregate_mean((huge_integer,))
    with pytest.raises(ValueError, match="mean must be finite"):
        aggregate_mean((1e308, 1e308))
    with pytest.raises(ValueError, match="paired improvement.*finite"):
        paired_improvements(
            (-1e308,) * 3,
            (1e308,) * 3,
            ObjectiveDirection.MAXIMIZE,
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 2, 3),
        )


def test_bootstrap_parameter_validation_is_strict():
    with pytest.raises(ValueError, match="between zero and one"):
        bootstrap_confidence_interval((1.0,), confidence=1, resamples=10, seed=1)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_confidence_interval((1.0,), confidence=0.9, resamples=0, seed=1)
    with pytest.raises(TypeError, match="seed"):
        bootstrap_confidence_interval((1.0,), confidence=0.9, resamples=10, seed=True)
    with pytest.raises(ValueError, match="seed must be between"):
        bootstrap_confidence_interval(
            (1.0,), confidence=0.9, resamples=10, seed=10**10_000
        )
    with pytest.raises(ValueError, match="must not exceed"):
        bootstrap_confidence_interval(
            (1.0,),
            confidence=0.9,
            resamples=MAX_BOOTSTRAP_RESAMPLES + 1,
            seed=1,
        )
    with pytest.raises(ValueError, match="finite"):
        bootstrap_confidence_interval(
            (1.0, float("nan")), confidence=0.9, resamples=10, seed=1
        )


def test_statistics_apis_honor_cooperative_deadline():
    past = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        bootstrap_confidence_interval(
            (1.0,), confidence=0.9, resamples=10, seed=1, deadline=past
        )
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        bootstrap_lower_bound(
            (1.0,), confidence=0.9, resamples=10, seed=1, deadline=past
        )
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        compare_metric(
            _primary(),
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 2, 3),
            confidence=0.9,
            bootstrap_resamples=10,
            bootstrap_seed=1,
            deadline=past,
        )
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        assess_promotion(
            (_primary(),),
            {"quality": (1.0, 1.0, 1.0)},
            {"quality": (2.0, 2.0, 2.0)},
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 2, 3),
            gates={},
            min_improvement=0,
            confidence=0.9,
            bootstrap_resamples=10,
            bootstrap_seed=1,
            deadline=past,
        )


def test_compare_metric_requires_at_least_three_paired_finite_samples():
    with pytest.raises(ValueError, match="at least 3"):
        compare_metric(
            _primary(),
            (1.0, 2.0),
            (2.0, 3.0),
            baseline_seeds=(1, 2),
            candidate_seeds=(1, 2),
            confidence=0.9,
            bootstrap_resamples=50,
            bootstrap_seed=7,
        )
    with pytest.raises(ValueError, match="finite"):
        compare_metric(
            _primary(),
            (1.0, 2.0, 3.0),
            (2.0, float("inf"), 4.0),
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 2, 3),
            confidence=0.9,
            bootstrap_resamples=50,
            bootstrap_seed=7,
        )


def test_promotion_passes_all_gates_bounds_and_directional_regression_limit():
    objectives = (
        _primary(hard_min=1.5, hard_max=3.0),
        MetricObjective(
            name="latency",
            direction=ObjectiveDirection.MINIMIZE,
            hard_max=110.0,
            max_regression=1.0,
        ),
    )
    assessment = assess_promotion(
        objectives,
        baseline_metrics={"quality": (1.0,) * 5, "latency": (100.0,) * 5},
        candidate_metrics={"quality": (2.0,) * 5, "latency": (101.0,) * 5},
        baseline_seeds=SEEDS,
        candidate_seeds=SEEDS,
        gates={"artifacts_valid": True, "budget_ok": True},
        min_improvement=0.5,
        confidence=0.95,
        bootstrap_resamples=200,
        bootstrap_seed=99,
    )

    assert assessment.promote is True
    assert assessment.reasons == ()
    assert assessment.primary.mean_improvement == 1.0
    assert assessment.primary.lower_confidence_bound == 1.0
    assert assessment.secondary[0].mean_improvement == -1.0
    assert assessment.secondary[0].non_regression_passed is True
    assert assessment.hard_bounds_passed is True
    assert assessment.gates == (("artifacts_valid", True), ("budget_ok", True))


def test_promotion_uses_strict_primary_bound_and_reports_every_failed_policy():
    objectives = (
        _primary(hard_min=2.0),
        MetricObjective(
            name="latency",
            direction=ObjectiveDirection.MINIMIZE,
            hard_max=100.0,
            max_regression=0.5,
        ),
    )
    assessment = assess_promotion(
        objectives,
        baseline_metrics={"quality": (1.0,) * 5, "latency": (100.0,) * 5},
        candidate_metrics={"quality": (1.5,) * 5, "latency": (102.0,) * 5},
        baseline_seeds=SEEDS,
        candidate_seeds=SEEDS,
        gates={"budget_ok": False},
        min_improvement=0.5,
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=123,
    )

    assert assessment.promote is False
    assert assessment.primary.lower_confidence_bound == assessment.min_improvement
    assert assessment.all_gates_passed is False
    assert assessment.hard_bounds_passed is False
    assert assessment.secondary_non_regression_passed is False
    assert assessment.reasons == (
        "failed gates: budget_ok",
        "hard bounds failed: latency, quality",
        "secondary regression exceeded: latency",
        "primary lower confidence bound 0.5 does not exceed 0.5",
    )


def test_assessment_rejects_ambiguous_primary_and_mismatched_seeds():
    with pytest.raises(ValueError, match="exactly one"):
        assess_promotion(
            (MetricObjective("quality", ObjectiveDirection.MAXIMIZE),),
            {"quality": (1.0, 1.0, 1.0)},
            {"quality": (2.0, 2.0, 2.0)},
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 2, 3),
            gates={},
            min_improvement=0,
            confidence=0.9,
            bootstrap_resamples=20,
            bootstrap_seed=4,
        )
    with pytest.raises(ValueError, match="same order"):
        assess_promotion(
            (_primary(),),
            {"quality": (1.0, 1.0, 1.0)},
            {"quality": (2.0, 2.0, 2.0)},
            baseline_seeds=(1, 2, 3),
            candidate_seeds=(1, 3, 2),
            gates={},
            min_improvement=0,
            confidence=0.9,
            bootstrap_resamples=20,
            bootstrap_seed=4,
        )


def test_results_are_frozen():
    comparison = compare_metric(
        _primary(),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        baseline_seeds=(1, 2, 3),
        candidate_seeds=(1, 2, 3),
        confidence=0.9,
        bootstrap_resamples=20,
        bootstrap_seed=4,
    )
    with pytest.raises(FrozenInstanceError):
        comparison.mean_improvement = 0  # type: ignore[misc]
