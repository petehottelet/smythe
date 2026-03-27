"""Tests for the AsyncExecutor — concurrency, ordering, and deadlock detection."""

import asyncio
import time

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class SlowMockProvider(Provider):
    """Provider that sleeps to simulate latency, recording call order."""

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay
        self.call_order: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> str:
        node_label = prompt.split("\n")[0]
        await asyncio.sleep(self._delay)
        self.call_order.append(node_label)
        return f"result for: {node_label}"


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
    # 3 nodes at 150ms each: serial would take ~450ms, parallel should be ~150ms
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
