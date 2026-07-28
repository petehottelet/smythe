"""Deterministic provider-concurrency simulation for the first Autotune campaign."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass


SIMULATOR_VERSION = "smythe.optimize.concurrency.v2"
MAX_SIMULATION_WORK_ITEMS = 1_000_000
MAX_SIMULATION_CONCURRENCY = 1_000_000
_MAX_SEED = (1 << 63) - 1
_DEADLINE_CHECK_INTERVAL = 64


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    # Equivalent zero-valued scenarios should not receive distinct hashes.
    return 0.0 if number == 0 else number


def _bounded_positive_int(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _normalized_deadline(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return _finite_float(deadline, name="deadline")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("concurrency simulation deadline exceeded")


@dataclass(frozen=True, slots=True)
class ConcurrencyScenario:
    """A nonlinear provider model with congestion and overload failures."""

    work_items: int = 250
    provider_capacity: int = 8
    base_latency_ms: float = 120.0
    jitter_ratio: float = 0.22
    base_error_rate: float = 0.004
    overload_error_rate: float = 0.20

    def __post_init__(self) -> None:
        work_items = _bounded_positive_int(
            self.work_items,
            name="work_items",
            maximum=MAX_SIMULATION_WORK_ITEMS,
        )
        provider_capacity = _bounded_positive_int(
            self.provider_capacity,
            name="provider_capacity",
            maximum=MAX_SIMULATION_CONCURRENCY,
        )
        object.__setattr__(self, "work_items", work_items)
        object.__setattr__(self, "provider_capacity", provider_capacity)
        for name in (
            "base_latency_ms",
            "jitter_ratio",
            "base_error_rate",
            "overload_error_rate",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        if self.base_latency_ms <= 0:
            raise ValueError("base_latency_ms must be positive")
        if not 0 <= self.jitter_ratio < 1:
            raise ValueError("jitter_ratio must be in [0, 1)")
        if not 0 <= self.base_error_rate < 1:
            raise ValueError("base_error_rate must be in [0, 1)")
        if not 0 <= self.overload_error_rate <= 1:
            raise ValueError("overload_error_rate must be in [0, 1]")

    @property
    def evaluator_hash(self) -> str:
        payload = {
            "version": SIMULATOR_VERSION,
            "scenario": {
                "work_items": self.work_items,
                "provider_capacity": self.provider_capacity,
                "base_latency_ms": self.base_latency_ms,
                "jitter_ratio": self.jitter_ratio,
                "base_error_rate": self.base_error_rate,
                "overload_error_rate": self.overload_error_rate,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def simulate_concurrency(
    policy: Mapping[str, object],
    *,
    scenario: ConcurrencyScenario,
    seed: int,
    split: str,
    deadline: float | None = None,
) -> dict[str, float]:
    """Return deterministic quality-of-service metrics for one policy trial.

    The simulator is deliberately nonlinear: throughput initially benefits
    from parallelism, while contention increases latency and concurrency above
    provider capacity sharply increases rate-limit-like failures. Holdout uses
    the same frozen scenario with independent seeds and a small latency shift.
    """

    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping")
    if not isinstance(scenario, ConcurrencyScenario):
        raise TypeError("scenario must be a ConcurrencyScenario")
    value = _bounded_positive_int(
        policy.get("max_concurrency"),
        name="policy.max_concurrency",
        maximum=MAX_SIMULATION_CONCURRENCY,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    if not isinstance(split, str):
        raise TypeError("split must be a string")
    if split not in {"development", "confirmation", "holdout"}:
        raise ValueError(f"unknown evaluation split: {split!r}")
    deadline_value = _normalized_deadline(deadline)
    _check_deadline(deadline_value)

    concurrency = min(value, scenario.work_items)
    rng = random.Random(seed)
    worker_available = [0.0] * concurrency
    heapq.heapify(worker_available)
    latencies: list[float] = []
    successes = 0
    errors = 0
    overload = max(0.0, (concurrency - scenario.provider_capacity) / scenario.provider_capacity)
    contention = 1.0 + 0.018 * max(0, concurrency - 1) + 0.85 * overload * overload
    split_latency = 1.08 if split == "holdout" else 1.0
    error_probability = min(
        0.98,
        scenario.base_error_rate + scenario.overload_error_rate * overload,
    )
    sigma = scenario.jitter_ratio
    lognormal_mean = -0.5 * sigma * sigma

    for item_index in range(scenario.work_items):
        if item_index % _DEADLINE_CHECK_INTERVAL == 0:
            _check_deadline(deadline_value)
        started_at = heapq.heappop(worker_available)
        try:
            service_ms = (
                scenario.base_latency_ms
                * rng.lognormvariate(lognormal_mean, sigma)
                * contention
                * split_latency
            )
        except OverflowError as exc:
            raise ValueError("simulation service time must be finite") from exc
        failed = rng.random() < error_probability
        if failed:
            errors += 1
            # A rate-limit response is usually observed sooner than a complete
            # generation, but it still occupies a concurrency slot briefly.
            service_ms *= 0.35
        else:
            successes += 1
        if not math.isfinite(service_ms) or service_ms <= 0:
            raise ValueError("simulation service time must be finite and positive")
        latencies.append(service_ms)
        completed_at = started_at + service_ms
        if not math.isfinite(completed_at):
            raise ValueError("simulation completion time must be finite")
        heapq.heappush(worker_available, completed_at)

    _check_deadline(deadline_value)
    wall_ms = max(worker_available)
    if not math.isfinite(wall_ms) or wall_ms <= 0:
        raise ValueError("simulation wall time must be finite and positive")
    wall_seconds = wall_ms / 1000
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("simulation wall time in seconds must be finite and positive")
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    metrics = {
        "throughput_ops_s": successes / wall_seconds,
        "p95_latency_ms": ordered[p95_index],
        "error_rate": errors / scenario.work_items,
        "successful_operations": float(successes),
        "wall_time_ms": wall_ms,
    }
    if any(not math.isfinite(metric) for metric in metrics.values()):
        raise ValueError("simulation metrics must be finite")
    return metrics


__all__ = [
    "MAX_SIMULATION_CONCURRENCY",
    "MAX_SIMULATION_WORK_ITEMS",
    "SIMULATOR_VERSION",
    "ConcurrencyScenario",
    "simulate_concurrency",
]
