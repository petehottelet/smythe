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
