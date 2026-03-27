"""Tests for the AsyncExecutor — concurrency, ordering, and deadlock detection."""

import asyncio
import time

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class SlowMockProvider(Provider):
    """Provider that sleeps to simulate latency, recording call order."""

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay
        self.call_order: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        node_label = prompt.split("\n")[0]
        await asyncio.sleep(self._delay)
        self.call_order.append(node_label)
        return CompletionResult(
            text=f"result for: {node_label}",
            prompt_tokens=10,
            completion_tokens=5,
        )


def _make_executor(provider: Provider | None = None) -> tuple[AsyncExecutor, Tracer]:
    tracer = Tracer()
    registry = Registry()
    p = provider or SlowMockProvider(delay=0.0)
    executor = AsyncExecutor(provider=p, registry=registry, tracer=tracer)
    return executor, tracer


@pytest.mark.asyncio
async def test_parallel_nodes_run_concurrently():
    delay = 0.15
    provider = SlowMockProvider(delay=delay)
    executor, _ = _make_executor(provider)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b")
    c = Node(label="C", id="c")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b, c])

    start = time.monotonic()
    await executor.run(graph)
    elapsed = time.monotonic() - start

    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)
    assert elapsed < delay * 2.5, f"Expected ~{delay}s but took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_serial_dependencies_respected():
    provider = SlowMockProvider(delay=0.01)
    executor, _ = _make_executor(provider)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    await executor.run(graph)

    assert provider.call_order == ["A", "B", "C"]
    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)


@pytest.mark.asyncio
async def test_fork_join_ordering():
    """Parallel roots complete before the join node runs."""
    provider = SlowMockProvider(delay=0.01)
    executor, _ = _make_executor(provider)

    r1 = Node(label="R1", id="r1")
    r2 = Node(label="R2", id="r2")
    j = Node(label="Join", id="j", depends_on=["r1", "r2"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[r1, r2, j])

    await executor.run(graph)

    assert "Join" == provider.call_order[-1]
    assert set(provider.call_order[:2]) == {"R1", "R2"}


@pytest.mark.asyncio
async def test_dependency_results_forwarded():
    provider = SlowMockProvider(delay=0.0)
    executor, _ = _make_executor(provider)

    a = Node(label="First step", id="a")
    b = Node(label="Second step", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    await executor.run(graph)

    assert "result for: First step" in a.result
    assert b.result is not None


@pytest.mark.asyncio
async def test_deadlock_detection():
    executor, _ = _make_executor()

    a = Node(label="A", id="a", depends_on=["b"])
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError, match="Deadlock"):
        await executor.run(graph)


# --- Failure policy tests ---

class FailingProvider(Provider):
    """Provider that fails a configurable number of times before succeeding.

    If ``fail_labels`` is set, only prompts whose first line matches
    one of the labels will be failed; all others succeed immediately.
    """

    def __init__(self, failures: int = 1, fail_labels: set[str] | None = None) -> None:
        self._failures = failures
        self._fail_labels = fail_labels
        self._attempts: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        if self._fail_labels is not None and label not in self._fail_labels:
            return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)
        self._attempts.setdefault(label, 0)
        self._attempts[label] += 1
        if self._attempts[label] <= self._failures:
            raise RuntimeError(f"Simulated failure #{self._attempts[label]}")
        return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)


@pytest.mark.asyncio
async def test_failure_policy_halt_raises():
    """HALT policy (default) propagates the exception."""
    provider = FailingProvider(failures=999)
    executor, _ = _make_executor(provider)

    node = Node(label="Doomed", id="d", failure_policy=FailurePolicy.HALT)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await executor.run(graph)
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_failure_policy_skip():
    """SKIP policy marks the node as SKIPPED and continues execution."""
    provider = FailingProvider(failures=999, fail_labels={"Flaky"})
    executor, _ = _make_executor(provider)

    a = Node(label="Flaky", id="a", failure_policy=FailurePolicy.SKIP)
    b = Node(label="Downstream", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    await executor.run(graph)

    assert a.status == NodeStatus.SKIPPED
    assert b.status == NodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_failure_policy_retry_succeeds():
    """RETRY policy retries up to max_retries and succeeds if the error clears."""
    provider = FailingProvider(failures=1)
    executor, _ = _make_executor(provider)

    node = Node(
        label="Retryable", id="r",
        failure_policy=FailurePolicy.RETRY,
        max_retries=3,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    await executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert "ok:" in node.result


@pytest.mark.asyncio
async def test_failure_policy_retry_exhausted():
    """RETRY policy raises after exhausting all retries."""
    provider = FailingProvider(failures=999)
    executor, _ = _make_executor(provider)

    node = Node(
        label="Always-fail", id="af",
        failure_policy=FailurePolicy.RETRY,
        max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await executor.run(graph)
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_cascading_failure_message_not_deadlock():
    """When nodes are blocked by upstream failure, the error should say so, not 'Deadlock'."""
    provider = FailingProvider(failures=999, fail_labels={"Root"})
    executor, _ = _make_executor(provider)

    a = Node(label="Root", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Child", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError, match="blocked by.*failed upstream"):
        await executor.run(graph)


@pytest.mark.asyncio
async def test_retry_count_means_additional_retries():
    """max_retries is interpreted as retries beyond initial attempt."""
    provider = FailingProvider(failures=2)
    executor, _ = _make_executor(provider)
    node = Node(
        label="EventuallyOk", id="eo",
        failure_policy=FailurePolicy.RETRY,
        max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    await executor.run(graph)
    assert node.status == NodeStatus.COMPLETED
