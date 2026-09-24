"""Tests for distilling a successful run into a reusable template."""

from __future__ import annotations

import pytest

from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import SubGraphTemplate
from smythe.distill import DistillationError, distill_template
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.registry import Registry
from smythe.task import Task


def _completed_graph(with_agents: bool = True):
    registry = Registry()
    nodes = []
    for node_id, label, deps in (
        ("research", "Research the market", []),
        ("review", "Red-team the findings", ["research"]),
    ):
        agent_id = None
        if with_agents:
            agent = Agent(profile=AgentProfile(
                name=node_id.capitalize(),
                persona=f"You are the {node_id} specialist.",
                capabilities=[node_id],
            ))
            registry.register(agent)
            agent_id = agent.id
        node = Node(
            id=node_id, label=label, depends_on=deps, agent_id=agent_id,
            required_capabilities=[node_id],
        )
        node.status = NodeStatus.COMPLETED
        node.result = "some output"
        nodes.append(node)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)
    return graph, registry


def test_distills_a_usable_template():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="market-review", registry=registry)

    assert isinstance(template, SubGraphTemplate)
    assert template.name == "market-review"
    nodes, template_registry = template.builder(Task(goal="Assess widgets"))
    assert [n.id for n in nodes] == ["research", "review"]
    assert nodes[1].depends_on == ["research"]
    assert len(template_registry.list_agents()) == 2


def test_personas_travel_with_the_template():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    _, template_registry = template.builder(Task(goal="Anything"))
    personas = {a.profile.persona for a in template_registry.list_agents()}
    assert "You are the research specialist." in personas


def test_labels_are_pointed_at_the_new_task():
    """A learned shape must not re-answer the task it was learned from."""
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    nodes, _ = template.builder(Task(goal="Assess widget demand"))
    assert "Research the market" in nodes[0].label
    assert "Assess widget demand" in nodes[0].label


def test_goal_placeholder_is_substituted():
    graph, registry = _completed_graph()
    graph.nodes[0].label = "Research the market for {goal}"
    template = distill_template(graph, name="t", registry=registry)
    nodes, _ = template.builder(Task(goal="solar chargers"))
    assert nodes[0].label == "Research the market for solar chargers"
    assert "Apply this step to" not in nodes[0].label


def test_results_and_statuses_are_left_behind():
    """A template is a shape to fill, not a cached answer."""
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    nodes, _ = template.builder(Task(goal="New task"))
    assert all(n.result is None for n in nodes)
    assert all(n.status is NodeStatus.PENDING for n in nodes)


def test_template_is_isolated_from_later_graph_mutation():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    graph.nodes[0].label = "MUTATED"
    graph.nodes.pop()

    nodes, _ = template.builder(Task(goal="x"))
    assert len(nodes) == 2
    assert "MUTATED" not in nodes[0].label


def test_each_build_returns_independent_nodes():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    first, _ = template.builder(Task(goal="a"))
    second, _ = template.builder(Task(goal="b"))
    first[0].label = "changed"
    assert second[0].label != "changed"


def test_refuses_to_learn_from_an_unfinished_run():
    graph, registry = _completed_graph()
    graph.nodes[1].status = NodeStatus.FAILED
    with pytest.raises(DistillationError, match="unfinished"):
        distill_template(graph, name="t", registry=registry)


def test_can_be_forced_to_learn_from_a_partial_run():
    graph, registry = _completed_graph()
    graph.nodes[1].status = NodeStatus.FAILED
    template = distill_template(
        graph, name="t", registry=registry, require_success=False,
    )
    nodes, _ = template.builder(Task(goal="x"))
    assert len(nodes) == 2


def test_refuses_an_empty_graph():
    with pytest.raises(DistillationError, match="empty"):
        distill_template(
            ExecutionGraph(topology=[Topology.SERIAL], nodes=[]), name="t",
        )


def test_works_without_a_registry():
    graph, _ = _completed_graph(with_agents=False)
    template = distill_template(graph, name="t")
    nodes, registry = template.builder(Task(goal="x"))
    assert [n.id for n in nodes] == ["research", "review"]
    assert registry.list_agents() == []


def test_default_description_summarizes_the_shape():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    assert "serial" in template.description
    assert "2 steps" in template.description


def test_distilled_template_composes_with_the_constrained_architect():
    """The whole point: a learned shape becomes a menu item."""
    from smythe.constrained_planner import ConstrainedArchitect
    from smythe.provider import CompletionResult, Provider

    class PickFirst(Provider):
        async def complete(self, system, prompt, model):
            return CompletionResult(text='[{"template": "learned"}]')

    graph, registry = _completed_graph()
    template = distill_template(graph, name="learned", registry=registry)
    architect = ConstrainedArchitect(provider=PickFirst(), templates=[template])
    built, _ = architect.plan(Task(goal="A brand new task"))

    assert len(built.nodes) == 2
    assert any("A brand new task" in n.label for n in built.nodes)


@pytest.mark.parametrize("goal", [
    r"C:\Users\me\report.docx",
    r"count every \d+ in the log",
    r"\g<0> and \1 are literal",
])
def test_goal_is_substituted_literally(goal):
    """A goal is not a regex replacement template."""
    graph, registry = _completed_graph()
    graph.nodes[0].label = "Research {goal} thoroughly"
    template = distill_template(graph, name="t", registry=registry)
    nodes, _ = template.builder(Task(goal=goal))
    assert nodes[0].label == f"Research {goal} thoroughly"


def test_distilled_builder_accepts_params_however_they_arrive():
    graph, registry = _completed_graph()
    template = distill_template(graph, name="t", registry=registry)
    for nodes, _ in (
        template.builder(Task(goal="x")),
        template.builder(Task(goal="x"), {"depth": 2}),
        template.builder(Task(goal="x"), params={"depth": 2}),
        template.builder(Task(goal="x"), depth=2, audience="board"),
    ):
        assert [n.id for n in nodes] == ["research", "review"]


def test_distilled_template_tolerates_model_supplied_params():
    """ConstrainedArchitect calls builder(task, **params); that used to raise TypeError."""
    from smythe.constrained_planner import ConstrainedArchitect
    from smythe.provider import CompletionResult, Provider

    class PickWithParams(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(text='[{"template": "learned", "params": {"depth": 2}}]')

    graph, registry = _completed_graph()
    template = distill_template(graph, name="learned", registry=registry)
    provider = PickWithParams()
    architect = ConstrainedArchitect(provider=provider, templates=[template], max_retries=0)
    built, _ = architect.plan(Task(goal="A brand new task"))

    assert len(built.nodes) == 2
    assert provider.calls == 1


def test_model_params_named_like_the_builder_arguments_are_ignored():
    """Regression: a model param named ``task`` raised TypeError (multiple values
    for argument 'task'), which cost a paid planner retry."""
    from smythe.constrained_planner import ConstrainedArchitect
    from smythe.provider import CompletionResult, Provider

    class PickWithClashingParams(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(
                text='[{"template": "learned", "params": {"task": "x", "params": 5}}]'
            )

    graph, registry = _completed_graph()
    template = distill_template(graph, name="learned", registry=registry)
    provider = PickWithClashingParams()
    architect = ConstrainedArchitect(provider=provider, templates=[template], max_retries=0)
    built, _ = architect.plan(Task(goal="A brand new task"))

    assert provider.calls == 1
    assert [n.id for n in built.nodes] == ["learned-0-research", "learned-0-review"]
    assert all("A brand new task" in n.label for n in built.nodes)
