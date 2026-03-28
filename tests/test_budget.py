"""Tests for budget tracking and enforcement."""

import pytest

from smythe.budget import SentinelAlert, Sentinel
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class TokenCountingMockProvider(Provider):
    """Provider that returns configurable token counts."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50) -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        node_label = prompt.split("\n")[0]
        self.calls.append(node_label)
        return CompletionResult(
            text=f"result for: {node_label}",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )


def test_budget_tracks_cost():
    tracker = Sentinel(max_budget_usd=10.0, cost_per_token=0.000003)
    r1 = CompletionResult(text="hi", prompt_tokens=100, completion_tokens=50)
    r2 = CompletionResult(text="there", prompt_tokens=200, completion_tokens=100)

    tracker.record("node-1", r1)
    tracker.record("node-2", r2)

    expected = (150 + 300) * 0.000003
    assert abs(tracker.total_cost_usd - expected) < 1e-10
    assert len(tracker.breakdown()) == 2
    assert "node-1" in tracker.breakdown()
    assert "node-2" in tracker.breakdown()


def test_budget_raises_when_exhausted():
    tracker = Sentinel(max_budget_usd=0.0001, cost_per_token=0.000003)
    r = CompletionResult(text="big", prompt_tokens=10000, completion_tokens=10000)
    tracker.record("node-1", r)

    with pytest.raises(SentinelAlert) as exc_info:
        tracker.check("node-2")

    assert exc_info.value.node_id == "node-2"
    assert exc_info.value.spent > 0
    assert exc_info.value.limit == 0.0001


def test_no_budget_means_no_limit():
    tracker = Sentinel(max_budget_usd=None, cost_per_token=0.000003)
    r = CompletionResult(text="huge", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    for i in range(100):
        tracker.record(f"node-{i}", r)
        tracker.check(f"node-{i + 1}")

    assert tracker.total_cost_usd > 0


def test_executor_halts_on_budget():
    """Build a 3-node serial graph; budget runs out after node 2."""
    provider = TokenCountingMockProvider(prompt_tokens=1000, completion_tokens=500)
    cost_per_call = 1500 * 0.000003  # $0.0045 per node
    budget_limit = cost_per_call * 2  # exactly enough for 2 calls; check triggers before node 3

    budget = Sentinel(max_budget_usd=budget_limit, cost_per_token=0.000003)
    tracer = Tracer()
    registry = Registry()

    a = Node(label="Step A", id="a")
    b = Node(label="Step B", id="b", depends_on=["a"])
    c = Node(label="Step C", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    executor = Executor(provider=provider, registry=registry, tracer=tracer, budget=budget)

    with pytest.raises(SentinelAlert) as exc_info:
        executor.run(graph)

    assert exc_info.value.node_id == "c"
    assert len(provider.calls) == 2
    assert a.status == NodeStatus.COMPLETED
    assert b.status == NodeStatus.COMPLETED
    assert c.status == NodeStatus.PENDING


def test_budget_breakdown_per_node():
    tracker = Sentinel(max_budget_usd=1.0, cost_per_token=0.00001)
    tracker.record("alpha", CompletionResult(text="a", prompt_tokens=100, completion_tokens=50))
    tracker.record("beta", CompletionResult(text="b", prompt_tokens=200, completion_tokens=100))

    breakdown = tracker.breakdown()
    assert breakdown["alpha"] == 150 * 0.00001
    assert breakdown["beta"] == 300 * 0.00001


def test_budget_exhausted_error_message():
    err = SentinelAlert(spent=0.50, limit=0.50, node_id="node-x")
    assert "node-x" in str(err)
    assert "$0.50" in str(err)


def test_budget_reserve_prevents_overspend():
    """3 nodes want to reserve; budget only allows ~2."""
    tracker = Sentinel(max_budget_usd=0.012, cost_per_token=0.000003)
    estimated_cost = 2000 * 0.000003  # $0.006 per node

    tracker.reserve("node-a", estimated_cost)
    tracker.reserve("node-b", estimated_cost)

    with pytest.raises(SentinelAlert) as exc_info:
        tracker.reserve("node-c", estimated_cost)

    assert exc_info.value.node_id == "node-c"


def test_budget_record_replaces_reservation():
    """Actual cost should replace the estimated reservation."""
    tracker = Sentinel(max_budget_usd=1.0, cost_per_token=0.000003)
    estimated_cost = 2000 * 0.000003  # $0.006

    tracker.reserve("node-a", estimated_cost)
    assert abs(tracker.total_cost_usd - estimated_cost) < 1e-10

    actual_result = CompletionResult(text="x", prompt_tokens=100, completion_tokens=50)
    actual_cost = tracker.record("node-a", actual_result)

    expected_actual = 150 * 0.000003
    assert abs(actual_cost - expected_actual) < 1e-10
    assert abs(tracker.total_cost_usd - expected_actual) < 1e-10


def test_budget_release_on_failure():
    """Releasing a reservation should restore the spent budget."""
    tracker = Sentinel(max_budget_usd=0.01, cost_per_token=0.000003)
    estimated = 2000 * 0.000003

    tracker.reserve("node-a", estimated)
    assert abs(tracker.total_cost_usd - estimated) < 1e-10

    tracker.release("node-a")
    assert abs(tracker.total_cost_usd) < 1e-10

    tracker.reserve("node-b", estimated)
    assert abs(tracker.total_cost_usd - estimated) < 1e-10


def test_budget_release_nonexistent_is_safe():
    """Releasing a node that was never reserved should be a no-op."""
    tracker = Sentinel(max_budget_usd=1.0)
    tracker.release("ghost-node")
    assert tracker.total_cost_usd == 0.0
