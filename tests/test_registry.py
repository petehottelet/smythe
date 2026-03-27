"""Tests for the Agent Registry."""

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.registry import Registry


def test_register_and_retrieve():
    reg = Registry()
    agent = Agent(profile=AgentProfile(name="researcher"))
    reg.register(agent)

    assert reg.get(agent.id) is agent


def test_assign_fills_unassigned_nodes():
    reg = Registry()
    a = Node(label="Step 1", id="a")
    b = Node(label="Step 2", id="b")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    reg.assign(graph)

    assert a.agent_id is not None
    assert b.agent_id is not None
    assert len(reg.list_agents()) == 2


def test_assign_preserves_existing_assignment():
    reg = Registry()
    agent = Agent(profile=AgentProfile(name="specialist"))
    reg.register(agent)

    a = Node(label="Step 1", id="a", agent_id=agent.id)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])

    reg.assign(graph)

    assert a.agent_id == agent.id


def test_find_by_capabilities_exact_match():
    reg = Registry()
    agent = Agent(profile=AgentProfile(
        name="researcher",
        capabilities=["research", "summarize"],
    ))
    reg.register(agent)

    match = reg.find_by_capabilities(["research", "summarize"])
    assert match is agent


def test_find_by_capabilities_superset_match():
    reg = Registry()
    agent = Agent(profile=AgentProfile(
        name="polyglot",
        capabilities=["research", "summarize", "translate"],
    ))
    reg.register(agent)

    match = reg.find_by_capabilities(["research"])
    assert match is agent


def test_find_by_capabilities_no_match():
    reg = Registry()
    agent = Agent(profile=AgentProfile(
        name="writer",
        capabilities=["writing"],
    ))
    reg.register(agent)

    match = reg.find_by_capabilities(["research"])
    assert match is None


def test_find_by_capabilities_prefers_tightest_match():
    reg = Registry()
    generalist = Agent(profile=AgentProfile(
        name="generalist",
        capabilities=["research", "summarize", "translate", "code"],
    ))
    specialist = Agent(profile=AgentProfile(
        name="specialist",
        capabilities=["research", "summarize"],
    ))
    reg.register(generalist)
    reg.register(specialist)

    match = reg.find_by_capabilities(["research", "summarize"])
    assert match is specialist


def test_find_by_capabilities_deterministic_tiebreak():
    """Agents with the same fit are ordered alphabetically by name."""
    reg = Registry()
    alice = Agent(profile=AgentProfile(name="alice", capabilities=["code"]))
    bob = Agent(profile=AgentProfile(name="bob", capabilities=["code"]))
    reg.register(bob)
    reg.register(alice)

    match = reg.find_by_capabilities(["code"])
    assert match is alice


def test_assign_uses_capability_matching():
    reg = Registry()
    researcher = Agent(profile=AgentProfile(
        name="researcher",
        capabilities=["research"],
    ))
    reg.register(researcher)

    node = Node(label="Research task", id="r1", required_capabilities=["research"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    reg.assign(graph)

    assert node.agent_id == researcher.id


def test_assign_creates_generalist_when_no_capability_match():
    reg = Registry()
    node = Node(label="Unknown", id="u1", required_capabilities=["quantum-computing"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    reg.assign(graph)

    assert node.agent_id is not None
    assigned = reg.get(node.agent_id)
    assert assigned is not None
