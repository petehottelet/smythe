"""Tests for ExecutionGraph construction and validation."""

import pytest

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology


def test_roots_returns_nodes_without_deps():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    assert graph.roots() == [a]


def test_dependents():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b, c])

    assert set(n.id for n in graph.dependents("a")) == {"b", "c"}


def test_is_ready_when_deps_completed():
    a = Node(label="A", id="a", status=NodeStatus.COMPLETED)
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    assert graph.is_ready(b)


def test_is_ready_false_when_deps_pending():
    a = Node(label="A", id="a", status=NodeStatus.PENDING)
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    assert not graph.is_ready(b)


def test_validate_detects_missing_dependency():
    b = Node(label="B", id="b", depends_on=["nonexistent"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[b])

    with pytest.raises(ValueError, match="unknown node"):
        graph.validate()


def test_validate_detects_cycle():
    a = Node(label="A", id="a", depends_on=["b"])
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(ValueError, match="cycle"):
        graph.validate()


def test_validate_passes_for_valid_dag():
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["a", "b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    graph.validate()


# --- __repr__ and __str__ ---


def test_repr():
    graph = ExecutionGraph(
        topology=[Topology.FORK_JOIN],
        nodes=[Node(label="X", id="x"), Node(label="Y", id="y")],
    )
    assert repr(graph) == "ExecutionGraph(topology=[<Topology.FORK_JOIN: 'fork_join'>], nodes=2)"


def test_str_serial_graph():
    a = Node(label="Summarize document", id="a", agent_id="SummaryAgent")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    output = str(graph)

    assert output == (
        'TaskGraph(topology="serial")\n'
        "└─ SummaryAgent: Summarize document"
    )


def test_str_fork_join_graph():
    r1 = Node(label="Research topic A", id="r1", agent_id="ResearcherA")
    r2 = Node(label="Research topic B", id="r2", agent_id="ResearcherB")
    r3 = Node(label="Research topic C", id="r3", agent_id="ResearcherC")
    j = Node(label="Merge findings", id="j", agent_id="JoinAgent", depends_on=["r1", "r2", "r3"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[r1, r2, r3, j])
    output = str(graph)

    assert 'TaskGraph(topology="fork-join")' in output
    assert "fork (parallel):" in output
    assert "ResearcherA: Research topic A" in output
    assert "ResearcherB: Research topic B" in output
    assert "ResearcherC: Research topic C" in output
    assert "join: JoinAgent: Merge findings" in output


def test_str_fork_join_with_serial_tail():
    r1 = Node(label="Find venues", id="r1", agent_id="VenueAgent")
    r2 = Node(label="Find bakeries", id="r2", agent_id="BakeryAgent")
    j = Node(label="Rank options", id="j", agent_id="RankerAgent", depends_on=["r1", "r2"])
    s = Node(label="Draft invitations", id="s", agent_id="InviteAgent", depends_on=["j"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN, Topology.SERIAL], nodes=[r1, r2, j, s])
    output = str(graph)

    assert 'TaskGraph(topology="fork-join \u2192 serial")' in output
    assert "fork (parallel):" in output
    assert "join:" in output
    assert "serial (depends on RankerAgent):" in output
    assert "InviteAgent: Draft invitations" in output


def test_str_unassigned_agents_fall_back_to_id():
    a = Node(label="Do work", id="node-1")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    output = str(graph)

    assert "node-1: Do work" in output


def test_str_compound_topology():
    r1 = Node(label="Analyze financials", id="r1", agent_id="FinanceAgent")
    r2 = Node(label="Assess IP", id="r2", agent_id="TechAgent")
    j = Node(label="Merge report", id="j", agent_id="JoinAgent", depends_on=["r1", "r2"])
    adv = Node(
        label="Challenge assumptions", id="adv", agent_id="RedTeamAgent",
        depends_on=["j"], metadata={"role": "adversarial"},
    )
    memo = Node(label="Final memo", id="memo", agent_id="MemoAgent", depends_on=["adv"])
    graph = ExecutionGraph(
        topology=[Topology.FORK_JOIN, Topology.ADVERSARIAL, Topology.SERIAL],
        nodes=[r1, r2, j, adv, memo],
    )
    output = str(graph)

    assert 'TaskGraph(topology="fork-join \u2192 adversarial \u2192 serial")' in output
    assert "fork (parallel):" in output
    assert "join:" in output
    assert "adversarial:" in output
    assert "RedTeamAgent: Challenge assumptions" in output
    assert "MemoAgent: Final memo" in output


# --- depth and agent_count ---


def test_depth_serial_chain():
    """A→B→C has depth 2 (two edges on the longest path)."""
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])
    assert graph.depth == 2


def test_depth_fork_join():
    """Parallel roots into a single join has depth 1."""
    r1 = Node(label="R1", id="r1")
    r2 = Node(label="R2", id="r2")
    j = Node(label="Join", id="j", depends_on=["r1", "r2"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[r1, r2, j])
    assert graph.depth == 1


def test_depth_empty_graph():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[])
    assert graph.depth == 0


def test_depth_single_node():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(label="A", id="a")])
    assert graph.depth == 0


def test_agent_count():
    a = Node(label="A", id="a", agent_id="agent-1")
    b = Node(label="B", id="b", agent_id="agent-2")
    c = Node(label="C", id="c", agent_id="agent-1")
    d = Node(label="D", id="d")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c, d])
    assert graph.agent_count == 2


def test_agent_count_no_agents():
    a = Node(label="A", id="a")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    assert graph.agent_count == 0


def test_validate_rejects_duplicate_node_ids():
    a = Node(label="First", id="dup")
    b = Node(label="Second", id="dup")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(ValueError, match="Duplicate node IDs"):
        graph.validate()
