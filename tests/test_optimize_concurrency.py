from __future__ import annotations

import time

import pytest

from smythe.optimize.concurrency import (
    MAX_SIMULATION_CONCURRENCY,
    MAX_SIMULATION_WORK_ITEMS,
    ConcurrencyScenario,
    simulate_concurrency,
)


def test_concurrency_simulator_is_deterministic_and_nonlinear():
    scenario = ConcurrencyScenario(work_items=500, provider_capacity=8)

    serial = simulate_concurrency(
        {"max_concurrency": 1}, scenario=scenario, seed=42, split="development"
    )
    balanced = simulate_concurrency(
        {"max_concurrency": 8}, scenario=scenario, seed=42, split="development"
    )
    overloaded = simulate_concurrency(
        {"max_concurrency": 24}, scenario=scenario, seed=42, split="development"
    )
    repeat = simulate_concurrency(
        {"max_concurrency": 8}, scenario=scenario, seed=42, split="development"
    )

    assert balanced == repeat
    assert balanced["throughput_ops_s"] > serial["throughput_ops_s"]
    assert overloaded["error_rate"] > balanced["error_rate"]
    assert overloaded["p95_latency_ms"] > balanced["p95_latency_ms"]
    assert scenario.evaluator_hash.startswith("sha256:")


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "8"])
def test_concurrency_simulator_rejects_invalid_policy(value):
    with pytest.raises((TypeError, ValueError)):
        simulate_concurrency(
            {"max_concurrency": value},
            scenario=ConcurrencyScenario(),
            seed=1,
            split="development",
        )


def test_holdout_is_distinct_but_reproducible():
    scenario = ConcurrencyScenario()
    development = simulate_concurrency(
        {"max_concurrency": 8}, scenario=scenario, seed=9, split="development"
    )
    holdout = simulate_concurrency(
        {"max_concurrency": 8}, scenario=scenario, seed=9, split="holdout"
    )

    assert holdout["p95_latency_ms"] > development["p95_latency_ms"]


def test_concurrency_simulator_rejects_huge_integers_without_overflow():
    huge = 10**10_000
    with pytest.raises(ValueError, match="max_concurrency must be between"):
        simulate_concurrency(
            {"max_concurrency": huge},
            scenario=ConcurrencyScenario(),
            seed=1,
            split="development",
        )
    with pytest.raises(ValueError, match="work_items must be between"):
        ConcurrencyScenario(work_items=MAX_SIMULATION_WORK_ITEMS + 1)
    with pytest.raises(ValueError, match="provider_capacity must be between"):
        ConcurrencyScenario(provider_capacity=MAX_SIMULATION_CONCURRENCY + 1)
    with pytest.raises(ValueError, match="base_latency_ms must be finite"):
        ConcurrencyScenario(base_latency_ms=huge)
    with pytest.raises(ValueError, match="seed must be between"):
        simulate_concurrency(
            {"max_concurrency": 1},
            scenario=ConcurrencyScenario(),
            seed=huge,
            split="development",
        )


def test_concurrency_simulator_honors_cooperative_deadline():
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        simulate_concurrency(
            {"max_concurrency": 1},
            scenario=ConcurrencyScenario(),
            seed=1,
            split="development",
            deadline=time.monotonic() - 1,
        )
    with pytest.raises(ValueError, match="deadline must be finite"):
        simulate_concurrency(
            {"max_concurrency": 1},
            scenario=ConcurrencyScenario(),
            seed=1,
            split="development",
            deadline=10**10_000,
        )


def test_concurrency_simulator_rejects_underflowed_service_time():
    with pytest.raises(ValueError, match="wall time.*must be finite and positive"):
        simulate_concurrency(
            {"max_concurrency": 1},
            scenario=ConcurrencyScenario(
                work_items=1,
                base_latency_ms=float.fromhex("0x0.0000000000001p-1022"),
            ),
            seed=1,
            split="development",
        )
