"""Tests for the ConstrainedArchitect."""

import json

import pytest

from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.graph import ExecutionGraph, Node
from smythe.planner import ArchitectError
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.task import Task


class MockConstrainedProvider(Provider):
    """Provider that returns preconfigured responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.prompts_received: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.prompts_received.append(prompt)
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return CompletionResult(text=response, prompt_tokens=20, completion_tokens=30)


def _research_builder(task: Task) -> tuple[list[Node], Registry]:
    """Template that produces a 2-node research sub-graph."""
    nodes = [
        Node(id="search", label="Search the web"),
        Node(id="summarize", label="Summarize findings", depends_on=["search"]),
    ]
    registry = Registry()
    agent = Agent(profile=AgentProfile(name="Researcher", capabilities=["research"]))
    registry.register(agent)
    return nodes, registry


def _draft_builder(task: Task) -> tuple[list[Node], Registry]:
    """Template that produces a single-node drafting sub-graph."""
    nodes = [Node(id="write", label="Draft the document")]
    return nodes, Registry()


def _parallel_builder(task: Task, num_workers: int = 2) -> tuple[list[Node], Registry]:
    """Template with parameterized parallel workers."""
    workers = [Node(id=f"w-{i}", label=f"Worker {i}") for i in range(num_workers)]
    merge = Node(id="merge", label="Merge", depends_on=[w.id for w in workers])
    return [*workers, merge], Registry()


TEMPLATES = [
    SubGraphTemplate(name="research", description="Web research and summarization", builder=_research_builder),
    SubGraphTemplate(name="draft", description="Draft a document", builder=_draft_builder),
    SubGraphTemplate(name="parallel-work", description="Parallel workers with merge", builder=_parallel_builder),
]


def test_constrained_planner_selects_single_template():
    response = json.dumps([{"template": "research"}])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Research competitors")

    graph, registry = planner.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].id == "research-0-search"
    assert graph.nodes[1].id == "research-0-summarize"
    assert graph.nodes[1].depends_on == ["research-0-search"]


def test_constrained_planner_composes_multiple():
    response = json.dumps([
        {"template": "research"},
        {"template": "draft"},
    ])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Research then write")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 3
    draft_node = next(n for n in graph.nodes if "write" in n.id)
    assert "research-0-summarize" in draft_node.depends_on


def test_constrained_planner_rejects_unknown_template():
    response = json.dumps([{"template": "nonexistent"}])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES, max_retries=0)
    task = Task(goal="This should fail")

    with pytest.raises(ArchitectError, match="ConstrainedArchitect failed"):
        planner.plan(task)


def test_constrained_planner_passes_params():
    response = json.dumps([{"template": "parallel-work", "params": {"num_workers": 3}}])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Parallel task")

    graph, _ = planner.plan(task)

    worker_nodes = [n for n in graph.nodes if "w-" in n.id]
    assert len(worker_nodes) == 3


def test_constrained_planner_id_namespacing():
    """Same template used twice gets distinct ID prefixes."""
    response = json.dumps([
        {"template": "draft"},
        {"template": "draft"},
    ])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Draft twice")

    graph, _ = planner.plan(task)

    ids = [n.id for n in graph.nodes]
    assert "draft-0-write" in ids
    assert "draft-1-write" in ids
    assert len(set(ids)) == len(ids)


def test_constrained_planner_merges_registries():
    response = json.dumps([{"template": "research"}])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Research")

    _, registry = planner.plan(task)

    agents = registry.list_agents()
    assert len(agents) == 1
    assert agents[0].profile.name == "Researcher"


def test_constrained_planner_retries():
    provider = MockConstrainedProvider([
        "not json",
        json.dumps([{"template": "draft"}]),
    ])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES, max_retries=2)
    task = Task(goal="Retry test")

    graph, _ = planner.plan(task)
    assert len(graph.nodes) == 1
    assert len(provider.prompts_received) == 2
    assert "could not be parsed" in provider.prompts_received[1]


def test_constrained_planner_prompt_contains_menu():
    response = json.dumps([{"template": "draft"}])
    provider = MockConstrainedProvider([response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES)
    task = Task(goal="Check menu")

    planner.plan(task)

    prompt = provider.prompts_received[0]
    assert "research" in prompt
    assert "draft" in prompt
    assert "parallel-work" in prompt


def test_constrained_planner_does_not_mutate_reused_template_nodes():
    """Reusable template nodes should not be permanently rewritten across runs."""
    shared_nodes = [Node(id="write", label="Draft the document")]

    def _reused_builder(task: Task) -> tuple[list[Node], Registry]:
        return shared_nodes, Registry()

    templates = [
        SubGraphTemplate(
            name="reused",
            description="Returns same Node instance every call",
            builder=_reused_builder,
        )
    ]
    response = json.dumps([{"template": "reused"}])
    provider = MockConstrainedProvider([response, response])
    planner = ConstrainedArchitect(provider=provider, templates=templates)
    task = Task(goal="Draft")

    first_graph, _ = planner.plan(task)
    second_graph, _ = planner.plan(task)

    assert first_graph.nodes[0].id == "reused-0-write"
    assert second_graph.nodes[0].id == "reused-0-write"
    assert shared_nodes[0].id == "write"


def test_constrained_planner_catches_bad_params_type():
    """TypeError from builder (bad param type) should be caught and retried."""
    bad_response = json.dumps([{"template": "parallel-work", "params": {"num_workers": "not-int"}}])
    good_response = json.dumps([{"template": "draft"}])
    provider = MockConstrainedProvider([bad_response, good_response])
    planner = ConstrainedArchitect(provider=provider, templates=TEMPLATES, max_retries=2)
    task = Task(goal="Bad params recovery")

    graph, _ = planner.plan(task)
    assert len(graph.nodes) == 1
    assert len(provider.prompts_received) == 2
