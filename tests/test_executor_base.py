"""Tests for ExecutorBase shared helpers."""

from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class DummyProvider(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


def _make_base() -> ExecutorBase:
    return ExecutorBase(
        provider=DummyProvider(),
        registry=Registry(),
        tracer=Tracer(),
    )


def test_deps_satisfied_all_completed():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.COMPLETED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_skipped():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.SKIPPED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_failed_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.FAILED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_with_pending_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_no_deps():
    base = _make_base()
    a = Node(label="A", id="a")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    assert base.deps_satisfied(a, graph) is True
