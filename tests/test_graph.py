"""Tests for ExecutionGraph construction and validation."""

import pytest

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology


def test_roots_returns_nodes_without_deps():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[a, b])

    assert graph.roots() == [a]


def test_dependents():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["a"])
    graph = ExecutionGraph(topology=Topology.FORK_JOIN, nodes=[a, b, c])

    assert set(n.id for n in graph.dependents("a")) == {"b", "c"}


def test_is_ready_when_deps_completed():
    a = Node(label="A", id="a", status=NodeStatus.COMPLETED)
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[a, b])

    assert graph.is_ready(b)


def test_is_ready_false_when_deps_pending():
    a = Node(label="A", id="a", status=NodeStatus.PENDING)
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[a, b])

    assert not graph.is_ready(b)


def test_validate_detects_missing_dependency():
    b = Node(label="B", id="b", depends_on=["nonexistent"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[b])

    with pytest.raises(ValueError, match="unknown node"):
        graph.validate()


def test_validate_detects_cycle():
    a = Node(label="A", id="a", depends_on=["b"])
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[a, b])

    with pytest.raises(ValueError, match="cycle"):
        graph.validate()


def test_validate_passes_for_valid_dag():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["a", "b"])
    graph = ExecutionGraph(topology=Topology.SERIAL, nodes=[a, b, c])

    graph.validate()
