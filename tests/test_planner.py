"""Tests for the default Planner."""

from smythe.planner import Planner
from smythe.graph import Topology
from smythe.task import Task


def test_default_planner_creates_serial_graph():
    planner = Planner()
    task = Task(goal="Write a summary")
    graph = planner.plan(task)

    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Write a summary"
