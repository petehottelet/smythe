"""Tests for the serial Executor with failure policy support."""

import random

import pytest

from helpers import FailingProvider
from smythe.executor import Executor
from smythe.executor_base import NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class SuccessProvider(Provider):
    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        return CompletionResult(text=f"done: {label}", prompt_tokens=5, completion_tokens=5)


def _make_executor(provider: Provider | None = None) -> tuple[Executor, Tracer]:
    tracer = Tracer()
    registry = Registry()
    p = provider or SuccessProvider()
    return Executor(provider=p, registry=registry, tracer=tracer), tracer


def test_halt_raises_on_failure():
    executor, _ = _make_executor(FailingProvider(failures=999))
    node = Node(label="Fail", id="f", failure_policy=FailurePolicy.HALT)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        executor.run(graph)
    assert node.status == NodeStatus.FAILED


def test_skip_marks_skipped_and_continues():
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Flaky"}))
    a = Node(label="Flaky", id="a", failure_policy=FailurePolicy.SKIP)
    b = Node(label="Next", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    executor.run(graph)

    assert a.status == NodeStatus.SKIPPED
    assert b.status == NodeStatus.COMPLETED


def test_retry_succeeds_after_transient_failure():
    executor, _ = _make_executor(FailingProvider(failures=1))
    node = Node(
        label="Retryable", id="r",
        failure_policy=FailurePolicy.RETRY, max_retries=3,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert "ok:" in node.result


def test_finalization_failure_after_billing_is_not_retried():
    class CountingProvider(SuccessProvider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return await super().complete(system, prompt, model)

    provider = CountingProvider()
    executor, _ = _make_executor(provider)
    executor.finalize_node_result = lambda node, result: (_ for _ in ()).throw(
        OSError("disk full")
    )
    node = Node(
        label="Paid image", id="image",
        failure_policy=FailurePolicy.RETRY, max_retries=3,
    )
    later = Node(label="Must not start", id="later")

    with pytest.raises(NodeFinalizationError, match="disk full"):
        executor.run(
            ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[node, later])
        )

    assert provider.calls == 1
    assert node.status == NodeStatus.FAILED
    assert later.status == NodeStatus.PENDING


def test_retry_exhausted_raises():
    executor, _ = _make_executor(FailingProvider(failures=999))
    node = Node(
        label="HardFail", id="hf",
        failure_policy=FailurePolicy.RETRY, max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        executor.run(graph)
    assert node.status == NodeStatus.FAILED


def test_retry_count_means_additional_retries():
    """max_retries is interpreted as retries beyond the initial attempt."""
    executor, _ = _make_executor(FailingProvider(failures=2))
    node = Node(
        label="EventuallyOk", id="eo",
        failure_policy=FailurePolicy.RETRY, max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor.run(graph)
    assert node.status == NodeStatus.COMPLETED


def test_serial_halts_downstream_of_failed():
    """When an upstream node fails with HALT, its dependent should not execute."""
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Upstream"}))
    a = Node(label="Upstream", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Downstream", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError):
        executor.run(graph)

    assert a.status == NodeStatus.FAILED
    assert b.status == NodeStatus.PENDING
    assert b.result is None


def test_serial_halt_leaves_undispatched_skip_descendants_pending():
    """SKIP applies to the node's own attempt, not an earlier global HALT."""
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Upstream"}))
    a = Node(label="Upstream", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Downstream", id="b", depends_on=["a"], failure_policy=FailurePolicy.SKIP)
    c = Node(label="Final", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    with pytest.raises(RuntimeError):
        executor.run(graph)

    assert a.status == NodeStatus.FAILED
    assert b.status == NodeStatus.PENDING
    assert b.result is None
    assert c.status == NodeStatus.PENDING
    assert c.result is None


def test_default_policy_is_halt():
    node = Node(label="Default", id="d")
    assert node.failure_policy == FailurePolicy.HALT


def test_walk_invalid_dependency_raises_valueerror():
    """_walk raises ValueError (not KeyError) for missing dependency IDs."""
    executor, _ = _make_executor()
    node = Node(label="Orphan", id="o", depends_on=["ghost"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(ValueError, match="depends on unknown node 'ghost'"):
        executor.run(graph)


class HangingProvider(Provider):
    """Provider that sleeps far longer than any test timeout."""

    async def complete(self, system, prompt, model):
        import asyncio
        await asyncio.sleep(5.0)
        return CompletionResult(text="too late", prompt_tokens=1, completion_tokens=1)


def test_node_timeout_fails_node_serially():
    executor, _ = _make_executor(HangingProvider())

    node = Node(label="slow", id="slow", timeout_s=0.05)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(TimeoutError, match="'slow' timed out after 0.05s"):
        executor.run(graph)

    assert node.status == NodeStatus.FAILED


def test_node_without_timeout_completes():
    executor, _ = _make_executor(SuccessProvider())

    node = Node(label="fine", id="fine")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert node.timeout_s is None


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_walk_5000_node_chain_in_any_input_order(order):
    nodes = [
        Node(id=f"n{i}", label=f"Step {i}", depends_on=[f"n{i - 1}"] if i else [])
        for i in range(5_000)
    ]
    if order == "reverse":
        nodes.reverse()
    elif order == "shuffled":
        random.Random(20260907).shuffle(nodes)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)
    executor, _ = _make_executor()

    assert [node.id for node in executor._walk(graph)] == [f"n{i}" for i in range(5_000)]


def test_walk_preserves_branch_order_and_shared_dependencies():
    root = Node(id="root", label="Root")
    a = Node(id="a", label="A", depends_on=[root.id])
    b = Node(id="b", label="B", depends_on=[root.id])
    join = Node(id="join", label="Join", depends_on=[b.id, a.id, b.id])
    later = Node(id="later", label="Later")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[join, a, later, root, b])
    provider = OfflineProvider()
    executor, _ = _make_executor(provider)

    executor.run(graph)

    assert provider.calls == ["Root", "B", "A", "Join", "Later"]
    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes)


def test_walk_reports_first_missing_dependency_in_depth_first_order():
    child = Node(id="child", label="Child", depends_on=["parent", "missing"])
    parent = Node(id="parent", label="Parent", depends_on=["ghost"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[child, parent])
    executor, _ = _make_executor()

    with pytest.raises(ValueError, match="^Node 'parent' depends on unknown node 'ghost'$"):
        executor._walk(graph)


@pytest.mark.parametrize("count", [1, 2, 5_000])
def test_private_walk_retains_cycle_tolerance(count):
    nodes = [
        Node(id=f"n{i}", label="Step", depends_on=[f"n{i - 1}"] if i else [])
        for i in range(count)
    ]
    nodes[0].depends_on = [nodes[-1].id]
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=list(reversed(nodes)))
    executor, _ = _make_executor()

    assert executor._walk(graph) == nodes


def test_reverse_5000_node_chain_executes_offline_in_dependency_order():
    nodes = [
        Node(id=f"n{i}", label=f"Step {i}", depends_on=[f"n{i - 1}"] if i else [])
        for i in range(5_000)
    ]
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=list(reversed(nodes)))
    provider = OfflineProvider(responses=["done"])
    executor, tracer = _make_executor(provider)

    assert executor.run(graph) is graph

    assert provider.calls == [f"Step {i}" for i in range(5_000)]
    assert all(node.status is NodeStatus.COMPLETED and node.result == "done" for node in nodes)
    assert len(tracer.summary()) == 5_000
