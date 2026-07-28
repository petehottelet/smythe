"""Deterministic paired statistics and promotion policy for optimization runs."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from smythe.optimize.contracts import MetricObjective, ObjectiveDirection


MIN_PAIRED_SAMPLES = 3
MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_MAX_SEED = (1 << 63) - 1
_DEADLINE_CHECK_INTERVAL = 1_024
_DEADLINE_DRAW_CHECK_INTERVAL = 4_096


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One direction-normalized, paired metric comparison."""

    objective_name: str
    direction: ObjectiveDirection
    sample_count: int
    sample_seeds: tuple[int, ...]
    baseline_mean: float
    candidate_mean: float
    mean_improvement: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_interval: tuple[float, float]
    lower_confidence_bound: float
    hard_bounds_passed: bool
    non_regression_passed: bool


@dataclass(frozen=True, slots=True)
class PromotionAssessment:
    """Complete, immutable evidence for a promote-or-reject decision."""

    promote: bool
    primary: ComparisonResult
    secondary: tuple[ComparisonResult, ...]
    gates: tuple[tuple[str, bool], ...]
    all_gates_passed: bool
    hard_bounds_passed: bool
    secondary_non_regression_passed: bool
    min_improvement: float
    reasons: tuple[str, ...]

    @property
    def comparisons(self) -> tuple[ComparisonResult, ...]:
        return (self.primary, *self.secondary)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_mean(values: Iterable[float], *, name: str) -> float:
    try:
        result = fmean(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _seed(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _MAX_SEED:
        raise ValueError(f"{name} must be between 0 and {_MAX_SEED}")
    return value


def _normalized_deadline(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return _finite_number(deadline, name="deadline")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("statistics deadline exceeded")


def _finite_samples(
    values: Sequence[float],
    *,
    name: str,
    deadline: float | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    samples: list[float] = []
    for index, value in enumerate(values):
        if index % _DEADLINE_DRAW_CHECK_INTERVAL == 0:
            _check_deadline(deadline)
        samples.append(_finite_number(value, name=f"{name}[{index}]"))
    _check_deadline(deadline)
    return tuple(samples)


def _direction_value(direction: ObjectiveDirection) -> str:
    try:
        value = direction.value
    except AttributeError as exc:
        raise TypeError("direction must be an ObjectiveDirection") from exc
    if value not in {"maximize", "minimize"}:
        raise ValueError(f"unsupported objective direction: {value!r}")
    return value


def _paired_seeds(
    baseline_seeds: Sequence[int],
    candidate_seeds: Sequence[int],
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(baseline_seeds, (str, bytes)) or isinstance(
        candidate_seeds, (str, bytes)
    ):
        raise TypeError("sample seeds must be integer sequences")
    baseline = tuple(baseline_seeds)
    candidate = tuple(candidate_seeds)
    if len(baseline) != sample_count or len(candidate) != sample_count:
        raise ValueError("sample seeds must align one-to-one with metric samples")
    baseline = tuple(
        _seed(seed, name=f"baseline_seeds[{index}]")
        for index, seed in enumerate(baseline)
    )
    candidate = tuple(
        _seed(seed, name=f"candidate_seeds[{index}]")
        for index, seed in enumerate(candidate)
    )
    if len(set(baseline)) != len(baseline) or len(set(candidate)) != len(candidate):
        raise ValueError("sample seeds must be unique within each paired run")
    if baseline != candidate:
        raise ValueError("baseline and candidate sample seeds must match in the same order")
    return baseline


def paired_improvements(
    baseline: Sequence[float],
    candidate: Sequence[float],
    direction: ObjectiveDirection,
    *,
    baseline_seeds: Sequence[int],
    candidate_seeds: Sequence[int],
) -> tuple[float, ...]:
    """Return paired deltas where positive always means the candidate improved."""

    baseline_values = _finite_samples(baseline, name="baseline")
    candidate_values = _finite_samples(candidate, name="candidate")
    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline and candidate samples must have equal length")
    if not baseline_values:
        raise ValueError("paired samples must not be empty")
    _paired_seeds(
        baseline_seeds,
        candidate_seeds,
        sample_count=len(baseline_values),
    )
    direction_value = _direction_value(direction)
    if direction_value == "maximize":
        raw_deltas = (
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(
                baseline_values, candidate_values, strict=True
            )
        )
    else:
        raw_deltas = (
            baseline_value - candidate_value
            for baseline_value, candidate_value in zip(
                baseline_values, candidate_values, strict=True
            )
        )
    deltas: list[float] = []
    for index, delta in enumerate(raw_deltas):
        if not math.isfinite(delta):
            raise ValueError(f"paired improvement[{index}] must be finite")
        deltas.append(delta)
    return tuple(deltas)


def aggregate_mean(values: Sequence[float]) -> float:
    """Return a finite arithmetic mean with strict input validation."""

    samples = _finite_samples(values, name="values")
    if not samples:
        raise ValueError("values must not be empty")
    return _finite_mean(samples, name="values mean")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    result = (
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )
    if not math.isfinite(result):
        raise ValueError("bootstrap quantile must be finite")
    return result


def bootstrap_confidence_interval(
    paired_deltas: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    deadline: float | None = None,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the paired mean."""

    deadline_value = _normalized_deadline(deadline)
    _check_deadline(deadline_value)
    samples = _finite_samples(
        paired_deltas,
        name="paired_deltas",
        deadline=deadline_value,
    )
    if not samples:
        raise ValueError("paired_deltas must not be empty")
    confidence_value = _finite_number(confidence, name="confidence")
    if not 0 < confidence_value < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if resamples > MAX_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"resamples must not exceed {MAX_BOOTSTRAP_RESAMPLES}"
        )
    normalized_seed = _seed(seed, name="bootstrap seed")

    rng = random.Random(normalized_seed)
    sample_count = len(samples)
    means: list[float] = []
    for resample_index in range(resamples):
        if resample_index % _DEADLINE_CHECK_INTERVAL == 0:
            _check_deadline(deadline_value)

        def sampled_values() -> Iterable[float]:
            for draw_index in range(sample_count):
                if draw_index % _DEADLINE_DRAW_CHECK_INTERVAL == 0:
                    _check_deadline(deadline_value)
                yield samples[rng.randrange(sample_count)]

        means.append(
            _finite_mean(sampled_values(), name="bootstrap sample mean")
        )
    _check_deadline(deadline_value)
    means.sort()
    tail = (1 - confidence_value) / 2
    return _quantile(means, tail), _quantile(means, 1 - tail)


def bootstrap_lower_bound(
    paired_deltas: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    deadline: float | None = None,
) -> float:
    """Return the lower endpoint of the deterministic paired bootstrap interval."""

    return bootstrap_confidence_interval(
        paired_deltas,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        deadline=deadline,
    )[0]


def hard_thresholds_pass(candidate_mean: float, objective: MetricObjective) -> bool:
    """Check the aggregate candidate metric against its declared hard bounds."""

    value = _finite_number(candidate_mean, name=f"{objective.name} candidate mean")
    if objective.hard_min is not None and value < objective.hard_min:
        return False
    if objective.hard_max is not None and value > objective.hard_max:
        return False
    return True


def secondary_non_regression_passes(
    mean_improvement: float,
    objective: MetricObjective,
) -> bool:
    """Check a direction-normalized mean against a secondary regression allowance."""

    improvement = _finite_number(
        mean_improvement, name=f"{objective.name} mean improvement"
    )
    if objective.max_regression is None:
        return True
    return improvement >= -objective.max_regression


def compare_metric(
    objective: MetricObjective,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    baseline_seeds: Sequence[int],
    candidate_seeds: Sequence[int],
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    deadline: float | None = None,
) -> ComparisonResult:
    """Compare one metric using paired observations and seeded bootstrapping."""

    deadline_value = _normalized_deadline(deadline)
    _check_deadline(deadline_value)
    baseline_values = _finite_samples(
        baseline,
        name=f"{objective.name} baseline",
        deadline=deadline_value,
    )
    candidate_values = _finite_samples(
        candidate,
        name=f"{objective.name} candidate",
        deadline=deadline_value,
    )
    _check_deadline(deadline_value)
    if len(baseline_values) != len(candidate_values):
        raise ValueError(
            f"{objective.name} baseline and candidate samples must have equal length"
        )
    if len(baseline_values) < MIN_PAIRED_SAMPLES:
        raise ValueError(
            f"{objective.name} requires at least {MIN_PAIRED_SAMPLES} paired samples"
        )
    paired_seed_values = _paired_seeds(
        baseline_seeds,
        candidate_seeds,
        sample_count=len(baseline_values),
    )
    deltas = paired_improvements(
        baseline_values,
        candidate_values,
        objective.direction,
        baseline_seeds=paired_seed_values,
        candidate_seeds=paired_seed_values,
    )
    confidence_value = _finite_number(confidence, name="confidence")
    interval = bootstrap_confidence_interval(
        deltas,
        confidence=confidence_value,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        deadline=deadline_value,
    )
    candidate_mean = aggregate_mean(candidate_values)
    mean_improvement = aggregate_mean(deltas)
    return ComparisonResult(
        objective_name=objective.name,
        direction=objective.direction,
        sample_count=len(deltas),
        sample_seeds=paired_seed_values,
        baseline_mean=aggregate_mean(baseline_values),
        candidate_mean=candidate_mean,
        mean_improvement=mean_improvement,
        confidence_level=confidence_value,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_interval=interval,
        lower_confidence_bound=interval[0],
        hard_bounds_passed=hard_thresholds_pass(candidate_mean, objective),
        non_regression_passed=secondary_non_regression_passes(
            mean_improvement, objective
        ),
    )


def assess_promotion(
    objectives: Sequence[MetricObjective],
    baseline_metrics: Mapping[str, Sequence[float]],
    candidate_metrics: Mapping[str, Sequence[float]],
    *,
    baseline_seeds: Sequence[int],
    candidate_seeds: Sequence[int],
    gates: Mapping[str, bool],
    min_improvement: float,
    confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    deadline: float | None = None,
) -> PromotionAssessment:
    """Apply all statistical, hard-bound, gate, and non-regression requirements."""

    deadline_value = _normalized_deadline(deadline)
    _check_deadline(deadline_value)
    inventory = tuple(objectives)
    if not inventory:
        raise ValueError("at least one metric objective is required")
    if len({objective.name for objective in inventory}) != len(inventory):
        raise ValueError("metric objective names must be unique")
    primary_objectives = tuple(objective for objective in inventory if objective.primary)
    if len(primary_objectives) != 1:
        raise ValueError("exactly one metric objective must be primary")

    minimum = _finite_number(min_improvement, name="min_improvement")
    if minimum < 0:
        raise ValueError("min_improvement must be non-negative")
    if any(not isinstance(name, str) or not name.strip() for name in gates):
        raise ValueError("gate names must be non-empty strings")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise TypeError("gate results must be booleans")

    comparisons: dict[str, ComparisonResult] = {}
    for objective in inventory:
        _check_deadline(deadline_value)
        try:
            baseline = baseline_metrics[objective.name]
            candidate = candidate_metrics[objective.name]
        except KeyError as exc:
            raise ValueError(f"missing metric samples for {objective.name!r}") from exc
        comparisons[objective.name] = compare_metric(
            objective,
            baseline,
            candidate,
            baseline_seeds=baseline_seeds,
            candidate_seeds=candidate_seeds,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            deadline=deadline_value,
        )

    primary_objective = primary_objectives[0]
    primary = comparisons[primary_objective.name]
    secondary = tuple(
        comparisons[objective.name] for objective in inventory if not objective.primary
    )
    gate_results = tuple(sorted(gates.items()))
    all_gates_passed = all(value for _, value in gate_results)
    hard_bounds_passed = all(
        comparison.hard_bounds_passed for comparison in comparisons.values()
    )
    secondary_non_regression_passed = all(
        comparison.non_regression_passed for comparison in secondary
    )
    primary_improvement_passed = primary.lower_confidence_bound > minimum

    reasons: list[str] = []
    failed_gates = [name for name, passed in gate_results if not passed]
    if failed_gates:
        reasons.append("failed gates: " + ", ".join(failed_gates))
    failed_bounds = sorted(
        comparison.objective_name
        for comparison in comparisons.values()
        if not comparison.hard_bounds_passed
    )
    if failed_bounds:
        reasons.append("hard bounds failed: " + ", ".join(failed_bounds))
    failed_secondary = sorted(
        comparison.objective_name
        for comparison in secondary
        if not comparison.non_regression_passed
    )
    if failed_secondary:
        reasons.append("secondary regression exceeded: " + ", ".join(failed_secondary))
    if not primary_improvement_passed:
        reasons.append(
            f"primary lower confidence bound {primary.lower_confidence_bound:.12g} "
            f"does not exceed {minimum:.12g}"
        )

    promote = (
        all_gates_passed
        and hard_bounds_passed
        and secondary_non_regression_passed
        and primary_improvement_passed
    )
    return PromotionAssessment(
        promote=promote,
        primary=primary,
        secondary=secondary,
        gates=gate_results,
        all_gates_passed=all_gates_passed,
        hard_bounds_passed=hard_bounds_passed,
        secondary_non_regression_passed=secondary_non_regression_passed,
        min_improvement=minimum,
        reasons=tuple(reasons),
    )


__all__ = [
    "MAX_BOOTSTRAP_RESAMPLES",
    "MIN_PAIRED_SAMPLES",
    "ComparisonResult",
    "PromotionAssessment",
    "aggregate_mean",
    "assess_promotion",
    "bootstrap_confidence_interval",
    "bootstrap_lower_bound",
    "compare_metric",
    "hard_thresholds_pass",
    "paired_improvements",
    "secondary_non_regression_passes",
]
