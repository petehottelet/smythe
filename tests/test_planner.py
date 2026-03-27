"""Tests for the planner hierarchy."""

from smythe.graph import ExecutionGraph, Node, Topology
from smythe.planner import DeterministicPlanner, SimplePlanner
from smythe.registry import Registry
from smythe.task import Task


def test_simple_planner_creates_serial_graph():
    planner = SimplePlanner()
    task = Task(goal="Write a summary")
    graph, registry = planner.plan(task)

    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].label == "Write a summary"
    assert isinstance(registry, Registry)


def test_simple_planner_is_deterministic():
    """SimplePlanner is a subclass of DeterministicPlanner."""
    assert issubclass(SimplePlanner, DeterministicPlanner)
    assert isinstance(SimplePlanner(), DeterministicPlanner)


class _ForkJoinPlanner(DeterministicPlanner):
    """Example user-defined deterministic planner for testing."""

    def __init__(self, num_branches: int = 3) -> None:
        self.num_branches = num_branches

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        branches = [
            Node(id=f"branch-{i}", label=f"Branch {i}")
            for i in range(self.num_branches)
        ]
        join = Node(
            id="join",
            label="Merge results",
            depends_on=[b.id for b in branches],
        )
        graph = ExecutionGraph(
            topology=[Topology.FORK_JOIN],
            nodes=[*branches, join],
        )
        graph.validate()
        return graph, Registry()


def test_deterministic_planner_subclass():
    """Users can build custom DeterministicPlanners with parameterized DAGs."""
    planner = _ForkJoinPlanner(num_branches=4)
    task = Task(goal="Do something in parallel")
    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 5
    assert graph.topology == [Topology.FORK_JOIN]
    join = next(n for n in graph.nodes if n.id == "join")
    assert len(join.depends_on) == 4


def test_deterministic_planner_no_provider_needed():
    """DeterministicPlanner operates without any LLM provider."""
    planner = _ForkJoinPlanner()
    task = Task(goal="No provider needed")
    graph, _ = planner.plan(task)
    assert len(graph.nodes) == 4
