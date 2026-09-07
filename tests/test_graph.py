"""Tests for ExecutionGraph construction and validation."""

import random

import pytest

from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology


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


def test_is_ready_when_dependency_was_intentionally_skipped():
    """SKIP failure policy resolves a dependency just like successful completion."""
    skipped = Node(label="Optional", id="optional", status=NodeStatus.SKIPPED)
    downstream = Node(label="Continue", id="next", depends_on=["optional"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[skipped, downstream])

    assert graph.is_ready(downstream)


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


def test_node_execution_controls_have_safe_independent_defaults():
    """New nodes should default to bounded retry semantics without shared lists."""
    first = Node(label="First")
    second = Node(label="Second")

    assert first.failure_policy is FailurePolicy.HALT
    assert first.max_retries == 1
    assert first.timeout_s is None
    assert first.max_tool_iterations is None
    assert first.attach_dep_artifacts is False

    first.required_capabilities.append("vision")
    assert second.required_capabilities == []


def _serial_nodes(count):
    return [
        Node(id=f"n{i}", label=f"Step {i}", depends_on=[f"n{i - 1}"] if i else [])
        for i in range(count)
    ]


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_5000_node_chain_validates_and_orders_without_recursion(order):
    nodes = _serial_nodes(5_000)
    if order == "reverse":
        nodes.reverse()
    elif order == "shuffled":
        random.Random(20260907).shuffle(nodes)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)

    graph.validate()

    assert graph.depth == 4_999
    assert [node.id for node in graph._topo_sort()] == [f"n{i}" for i in range(5_000)]


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_5000_node_fork_join_handles_shared_dependencies_in_any_order(order):
    root = Node(id="root", label="Root")
    leaves = [Node(id=f"leaf{i}", label="Leaf", depends_on=[root.id]) for i in range(4_998)]
    join = Node(id="join", label="Join", depends_on=[node.id for node in reversed(leaves)])
    nodes = [root, *leaves, join]
    if order == "reverse":
        nodes.reverse()
    elif order == "shuffled":
        random.Random(20260907).shuffle(nodes)
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes)

    graph.validate()
    positions = {node.id: index for index, node in enumerate(graph._topo_sort())}

    assert graph.depth == 2
    assert len(positions) == 5_000
    assert all(positions[root.id] < positions[leaf.id] < positions[join.id] for leaf in leaves)


def test_topological_order_preserves_dependency_order_and_shared_subtrees():
    root = Node(id="root", label="Root")
    a = Node(id="a", label="A", depends_on=[root.id])
    b = Node(id="b", label="B", depends_on=[root.id])
    join = Node(id="join", label="Join", depends_on=[b.id, a.id, b.id])
    later = Node(id="later", label="Later")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[join, a, later, root, b])

    assert graph._topo_sort() == [root, b, a, join, later]
    assert graph.depth == 2
    graph.validate()


def test_unvalidated_missing_dependencies_retain_rendering_and_depth_behavior():
    child = Node(id="child", label="Child", depends_on=["parent", "missing"])
    parent = Node(id="parent", label="Parent", depends_on=["ghost"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[child, parent])

    assert graph._topo_sort() == [parent, child]
    assert graph.depth == 2
    assert "depends on ghost" in str(graph)
    with pytest.raises(ValueError, match="Node 'child' depends on unknown node 'missing'"):
        graph.validate()


@pytest.mark.parametrize("count", [1, 2, 5_000])
def test_cycles_are_rejected_at_any_depth_but_private_walk_still_terminates(count):
    nodes = _serial_nodes(count)
    nodes[0].depends_on = [nodes[-1].id]
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=list(reversed(nodes)))

    with pytest.raises(ValueError, match="^Execution graph contains a cycle$"):
        graph.validate()
    with pytest.raises(ValueError, match="^Execution graph contains a cycle$"):
        _ = graph.depth
    assert graph._topo_sort() == nodes


def test_deep_reverse_chain_renders_the_complete_graph_and_depth():
    graph = ExecutionGraph(
        topology=[Topology.SERIAL], nodes=list(reversed(_serial_nodes(2_000))),
        estimated_cost_usd=0.0,
    )

    rendered = str(graph)

    assert "Depth: 1999" in rendered
    assert rendered.index("n0: Step 0") < rendered.index("n1999: Step 1999")
    assert rendered.count(": Step ") == 2_000
