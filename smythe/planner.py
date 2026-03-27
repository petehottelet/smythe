"""Planner — generates an execution graph from a task description."""

from __future__ import annotations

from smythe.graph import ExecutionGraph, Node, Topology
from smythe.task import Task


class Planner:
    """Decides *how* a task should be executed by generating a DAG.

    The default implementation produces a single-node serial graph.
    Future versions will use LLM-assisted planning informed by
    task structure and historical execution data.
    """

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate an execution graph for the given task.

        Override this method to implement custom planning strategies
        or LLM-driven topology selection.
        """
        node = Node(label=task.goal)
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        graph.validate()
        return graph
