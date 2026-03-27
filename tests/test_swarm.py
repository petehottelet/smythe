"""Tests for the Swarm top-level orchestrator."""

import pytest

from smythe import Swarm, Task
from smythe.graph import ExecutionGraph, Topology
from smythe.provider import Provider


class MockProvider(Provider):
    async def complete(self, system: str, prompt: str, model: str) -> str:
        return f"mock: {prompt[:40]}"


def test_swarm_construction():
    swarm = Swarm(max_budget_usd=1.00, model="claude-mythos", provider=MockProvider())
    assert swarm.model == "claude-mythos"
    assert swarm.max_budget_usd == 1.00


def test_plan_returns_assigned_graph():
    swarm = Swarm(provider=MockProvider())
    task = Task(goal="Do something")
    graph = swarm.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].agent_id is not None


def test_execute_with_task():
    swarm = Swarm(provider=MockProvider())
    task = Task(goal="Do something")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None
    assert len(result.trace) == 1


def test_execute_with_graph():
    swarm = Swarm(provider=MockProvider())
    task = Task(goal="Do something")
    graph = swarm.plan(task)
    result = swarm.execute(graph)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_execute_parallel():
    swarm = Swarm(provider=MockProvider(), parallel=True)
    task = Task(goal="Do something in parallel")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_model_stamped_on_nodes():
    swarm = Swarm(model="claude-mythos", provider=MockProvider())
    task = Task(goal="Check model stamping")
    graph = swarm.plan(task)

    assert graph.nodes[0].metadata["model"] == "claude-mythos"
