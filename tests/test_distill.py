"""Tests for distilling a successful run into a reusable template."""

from __future__ import annotations

import pytest

from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.distill import DistillationError, distill_template
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.synthesizer import SynthesisStrategy, Synthesizer
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
    r"D:\Uploads\me\report.docx",
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


WRITER = "You write the memo."
JUDGE = "You judge the memo."


def _completed_gated_graph():
    """A run whose judge gated the memo, with tuned execution policies."""
    registry = Registry()
    writer = Agent(profile=AgentProfile(name="Writer", persona=WRITER, capabilities=["writing"]))
    judge = Agent(profile=AgentProfile(name="Judge", persona=JUDGE, capabilities=["review"]))
    registry.register(writer)
    registry.register(judge)
    draft = Node(
        id="draft", label="Draft the memo", agent_id=writer.id,
        required_capabilities=["writing"], failure_policy=FailurePolicy.RETRY,
        max_retries=3, timeout_s=90.0, metadata={"role": "author"},
    )
    check = Node(
        id="check", label="Check the memo; answer PASS or FAIL", depends_on=["draft"],
        agent_id=judge.id, required_capabilities=["review"], verifies="draft",
        max_regenerations=2, failure_policy=FailurePolicy.SKIP, max_retries=0,
        attach_dep_artifacts=True, metadata={"role": "adversarial"},
    )
    for node, result in ((draft, "the memo"), (check, "PASS")):
        node.status = NodeStatus.COMPLETED
        node.result = result
        # Execution history, not structure.
        node.metadata.update(model="old-model", cost_usd=0.01)
    check.metadata["regenerations_used"] = 1
    graph = ExecutionGraph(topology=[Topology.ADVERSARIAL], nodes=[draft, check])
    return graph, registry


class _Selects(Provider):
    """A template-selection model that always picks one template."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def complete(self, system, prompt, model):
        return CompletionResult(text=f'[{{"template": "{self._name}"}}]')


class _PersonaScript(Provider):
    """Answers each agent from its own queue and counts its calls."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self._script = {persona: list(replies) for persona, replies in script.items()}
        self.runs: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        self.runs[system] = self.runs.get(system, 0) + 1
        queue = self._script[system]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return CompletionResult(text=text, prompt_tokens=1, completion_tokens=1)


def test_gates_and_execution_policies_survive_distillation():
    """A judge that loses 'verifies' stops gating, and its verdict becomes the output."""
    graph, registry = _completed_gated_graph()
    template = distill_template(graph, name="t", registry=registry)
    graph.nodes[1].verifies = None  # The template is a snapshot.

    (draft, check), _ = template.builder(Task(goal="Summarize the quarter"))

    assert check.verifies == "draft"
    assert check.depends_on == ["draft"]
    assert check.max_regenerations == 2
    assert check.attach_dep_artifacts is True
    assert draft.attach_dep_artifacts is False
    assert [(n.failure_policy, n.max_retries, n.timeout_s) for n in (draft, check)] == [
        (FailurePolicy.RETRY, 3, 90.0), (FailurePolicy.SKIP, 0, None),
    ]
    # The role describes the shape; model, cost and verification history stay behind.
    assert draft.metadata == {"role": "author"}
    assert check.metadata == {"role": "adversarial"}


def test_distilled_metadata_is_independent_between_builds():
    graph, registry = _completed_gated_graph()
    template = distill_template(graph, name="t", registry=registry)
    first, _ = template.builder(Task(goal="a"))
    first[1].metadata["role"] = "changed"
    second, _ = template.builder(Task(goal="b"))
    assert second[1].metadata == {"role": "adversarial"}


def test_composed_template_keeps_its_gate():
    graph, registry = _completed_gated_graph()
    template = distill_template(graph, name="memo", registry=registry)
    architect = ConstrainedArchitect(provider=_Selects("memo"), templates=[template], max_retries=0)

    built, _ = architect.plan(Task(goal="Summarize the quarter"))

    check = next(n for n in built.nodes if n.id == "memo-0-check")
    assert check.verifies == "memo-0-draft"
    assert check.depends_on == ["memo-0-draft"]
    assert check.max_regenerations == 2


def test_distilled_gate_still_gates_and_returns_its_target_as_the_deliverable():
    """Executing the learned shape must regenerate a rejected draft and
    return the approved memo, not the judge's verdict."""
    graph, registry = _completed_gated_graph()
    template = distill_template(graph, name="memo", registry=registry)
    provider = _PersonaScript({
        WRITER: ["first memo", "THE REVISED MEMO"],
        JUDGE: ["FAIL: cites no sources", "PASS"],
    })
    swarm = Swarm(
        provider=provider,
        architect=ConstrainedArchitect(provider=_Selects("memo"), templates=[template], max_retries=0),
        synthesizer=Synthesizer(SynthesisStrategy.DELIVERABLE),
        artifact_dir=None,
    )

    result = swarm.execute(Task(goal="Summarize the quarter"))

    assert result.output == "THE REVISED MEMO"
    assert provider.runs == {WRITER: 2, JUDGE: 2}


def test_refuses_a_gate_whose_target_is_not_in_the_graph():
    graph, registry = _completed_gated_graph()
    graph.nodes[1].verifies = "ghost"
    with pytest.raises(DistillationError, match="'check'.*'ghost'"):
        distill_template(graph, name="t", registry=registry)


def test_refuses_a_gate_that_verifies_itself():
    graph, registry = _completed_gated_graph()
    graph.nodes[1].verifies = "check"
    with pytest.raises(DistillationError, match="'check'.*itself"):
        distill_template(graph, name="t", registry=registry)
