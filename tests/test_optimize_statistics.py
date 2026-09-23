"""Pure, deterministic promotion-statistics tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import random
import time

import pytest

from smythe.optimize.contracts import MetricObjective, ObjectiveDirection
from smythe.optimize.statistics import (
    MAX_BOOTSTRAP_RESAMPLES,
    PROMOTION_METHOD,
    aggregate_mean,
    assess_promotion,
    bootstrap_confidence_interval,
    bootstrap_lower_bound,
    compare_metric,
    paired_improvements,
    paired_t_confidence_interval,
    paired_t_lower_bound,
    student_t_quantile,
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


# Reference quantiles computed independently at 50 digits (mpmath inverse of
# the regularized incomplete beta), rounded to 16 significant digits.
@pytest.mark.parametrize(
    ("probability", "df", "expected"),
    [
        (0.975, 1, 12.70620473617469),
        (0.975, 2, 4.302652729749462),
        (0.975, 4, 2.776445105197793),
        (0.95, 30, 1.697260886593957),
        (0.995, 10, 3.169272672616951),
        (0.9995, 3, 12.92397863668796),
        (0.75, 1, 1.0),
        (0.9, 7, 1.414923927650509),
        (0.975, 1_000, 1.962339080826408),
        (0.975, 9_999, 1.960201263621357),
        (0.975, 10_000, 1.960201239890626),
        (0.975, 1_000_000, 1.959966356814107),
    ],
)
def test_student_t_quantile_matches_reference_values(probability, df, expected):
    assert student_t_quantile(probability, df) == pytest.approx(expected, rel=1e-12)
    assert student_t_quantile(1 - probability, df) == pytest.approx(-expected, rel=1e-12)


# Extreme-tail references from the exact finite-sum Student-t tail for integer
# degrees of freedom, solved in 100- to 400-digit decimal arithmetic (they
# agree with SciPy's isf to about 2e-16), rounded to 16 significant digits.
# Tolerances are the accuracy regimes stated in student_t_quantile's docstring.
@pytest.mark.parametrize(
    ("probability", "df", "expected", "tolerance"),
    [
        # Numerical inversion below 10,000 degrees of freedom, even at 1e-300.
        (1e-300, 9_999, -38.35651906066025, 1e-13),
        # Cornish-Fisher from 10,000: the most extreme contract tail ...
        (2.0**-54, 10_000, -8.306845025331896, 1e-14),
        # ... and the documented loss of accuracy in more extreme tails.
        (1e-100, 10_000, -21.51697419391498, 1e-10),
        (1e-300, 10_000, -38.35638432100424, 1e-8),
    ],
    ids=["inversion-1e-300", "cornish-fisher-contract-tail", "cornish-fisher-1e-100",
         "cornish-fisher-1e-300"],
)
def test_student_t_quantile_meets_documented_accuracy_by_regime(
    probability, df, expected, tolerance
):
    assert student_t_quantile(probability, df) == pytest.approx(expected, rel=tolerance)


def test_student_t_quantile_known_table_values_and_symmetry():
    assert round(student_t_quantile(0.975, 4), 6) == 2.776445
    assert round(student_t_quantile(0.975, 1), 4) == 12.7062
    assert round(student_t_quantile(0.95, 30), 6) == 1.697261
    assert student_t_quantile(0.5, 3) == 0.0
    values = [student_t_quantile(p, 5) for p in (0.6, 0.8, 0.95, 0.999)]
    assert values == sorted(values)
    by_df = [student_t_quantile(0.975, df) for df in (1, 2, 5, 30, 10_000)]
    assert by_df == sorted(by_df, reverse=True)


@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (1, 5734161139222659.0),
        (2, 94906265.62425154),
        (3, 270823.8069996586),
        (4, 15247.02990221789),
        (30, 16.62211287939564),
        (4_000, 8.328651407063098),
    ],
)
def test_paired_t_uses_exact_tail_for_most_extreme_contract_confidence(df, expected):
    # The largest float below one is a valid contract confidence.  Its
    # one-sided tail (1 - c) / 2 is ~5.6e-17 and must not round to zero.
    confidence = math.nextafter(1.0, 0.0)
    deltas = [0.0] * df + [1.0]
    count = df + 1
    mean = 1.0 / count
    standard_error = math.sqrt((count - 1) / count / (count - 1) / count)
    lower, upper = paired_t_confidence_interval(deltas, confidence=confidence)
    assert (mean - lower) / standard_error == pytest.approx(expected, rel=1e-9)
    assert (upper - mean) / standard_error == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("probability", "df", "error"),
    [
        (0.0, 4, ValueError),
        (1.0, 4, ValueError),
        (float("nan"), 4, ValueError),
        (True, 4, TypeError),
        (0.9, 0, ValueError),
        (0.9, True, TypeError),
        (0.9, 2.0, TypeError),
    ],
)
def test_student_t_quantile_rejects_invalid_arguments(probability, df, error):
    with pytest.raises(error):
        student_t_quantile(probability, df)


def test_paired_t_bound_is_mean_minus_t_times_standard_error():
    deltas = (0.25, 0.5, 0.75, 1.0, 1.25)
    standard_error = math.sqrt(0.625 / 4) / math.sqrt(5)
    expected = 0.75 - 2.776445105197793 * standard_error

    assert paired_t_lower_bound(deltas, confidence=0.95) == pytest.approx(expected, rel=1e-12)
    comparison = compare_metric(
        _primary(),
        (1.0,) * 5,
        tuple(1.0 + delta for delta in deltas),
        baseline_seeds=SEEDS,
        candidate_seeds=SEEDS,
        confidence=0.95,
        bootstrap_resamples=200,
        bootstrap_seed=5,
    )
    assert comparison.method == PROMOTION_METHOD == "paired_student_t"
    assert comparison.degrees_of_freedom == 4
    assert comparison.critical_value == pytest.approx(2.776445105197793, rel=1e-12)
    assert comparison.standard_error == pytest.approx(standard_error, rel=1e-12)
    assert comparison.lower_confidence_bound == comparison.confidence_interval[0]
    assert comparison.lower_confidence_bound == pytest.approx(expected, rel=1e-12)
    low, high = comparison.confidence_interval
    assert (low + high) / 2 == pytest.approx(comparison.mean_improvement)
    # The deltas are exact binary fractions, so the descriptive bootstrap
    # sees exactly these values.
    assert comparison.descriptive_bootstrap_interval == bootstrap_confidence_interval(
        deltas, confidence=0.95, resamples=200, seed=5
    )


def test_paired_t_refuses_single_samples_and_degenerates_only_for_identical_deltas():
    for too_few in ((), (1.0,)):
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_lower_bound(too_few, confidence=0.95)
    with pytest.raises(ValueError, match="between zero and one"):
        paired_t_lower_bound((1.0, 2.0), confidence=1.0)
    assert paired_t_confidence_interval((0.5, 0.5, 0.5), confidence=0.95) == (0.5, 0.5)
    # A tiny spread is not treated as zero variance.
    assert paired_t_lower_bound((0.5, 0.5, 0.5 + 1e-9), confidence=0.95) < 0.5
    # Scaling keeps large but finite deltas from overflowing the variance.
    large = paired_t_lower_bound((1e200, 2e200, 3e200), confidence=0.95)
    assert large == pytest.approx(2e200 - 4.302652729749462 * 1e200 / math.sqrt(3))
    with pytest.raises(ValueError, match="finite"):
        paired_t_lower_bound((1.7e308, -1.7e308, 1.7e308), confidence=0.95)


def test_bootstrap_interval_is_descriptive_and_never_decides_promotion():
    # A small, skewed sample whose percentile bootstrap lower bound is
    # positive, while the paired t bound is not.
    baseline = (0.0, 0.0, 0.0)
    candidate = (0.1, 1.0, 2.0)
    assessment = assess_promotion(
        (_primary(),),
        {"quality": baseline},
        {"quality": candidate},
        baseline_seeds=(1, 2, 3),
        candidate_seeds=(1, 2, 3),
        gates={},
        min_improvement=0.0,
        confidence=0.95,
        bootstrap_resamples=2_000,
        bootstrap_seed=11,
    )
    assert assessment.primary.descriptive_bootstrap_interval[0] > 0
    assert assessment.primary.lower_confidence_bound < 0
    assert assessment.promote is False


@pytest.mark.parametrize("sample_count", [3, 5])
def test_null_false_promotion_rate_is_near_nominal(sample_count):
    # Regression: the percentile bootstrap bound promoted 7.5-9.2% of null
    # candidates at n=5 and 12-14% at n=3 against a nominal 2.5%.
    simulations = 2_000
    rng = random.Random(8_675_309 + sample_count)
    seeds = tuple(range(1, sample_count + 1))
    promoted = 0
    for _ in range(simulations):
        deltas = tuple(rng.gauss(0.0, 1.0) for _ in range(sample_count))
        assessment = assess_promotion(
            (_primary(),),
            {"quality": (0.0,) * sample_count},
            {"quality": deltas},
            baseline_seeds=seeds,
            candidate_seeds=seeds,
            gates={},
            min_improvement=0.0,
            confidence=0.95,
            bootstrap_resamples=1,
            bootstrap_seed=1,
        )
        promoted += assessment.promote
    rate = promoted / simulations
    assert 0.015 <= rate <= 0.035, rate
