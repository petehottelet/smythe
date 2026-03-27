"""Tests for the Swarm top-level orchestrator."""

import pytest

from smythe import Swarm, Task
from smythe.graph import ExecutionGraph, Topology


def test_swarm_construction():
    swarm = Swarm(max_budget_usd=1.00, model="gpt-4o")
    assert swarm.model == "gpt-4o"
    assert swarm.max_budget_usd == 1.00


def test_plan_returns_assigned_graph():
    swarm = Swarm()
    task = Task(goal="Do something")
    graph = swarm.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].agent_id is not None


def test_execute_with_task_raises_not_implemented():
    """Until an LLM provider is wired up, execute should raise NotImplementedError."""
    swarm = Swarm()
    task = Task(goal="Do something")

    with pytest.raises(NotImplementedError, match="LLM dispatch not yet implemented"):
        swarm.execute(task)


def test_execute_with_graph_raises_not_implemented():
    """execute() also accepts a pre-planned graph."""
    swarm = Swarm()
    task = Task(goal="Do something")
    graph = swarm.plan(task)

    with pytest.raises(NotImplementedError, match="LLM dispatch not yet implemented"):
        swarm.execute(graph)
