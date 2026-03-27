"""Tests for the LLM-driven planner."""

import json
import tempfile

import pytest

from smythe.graph import ExecutionGraph, Topology
from smythe.memory import PlannerMemory
from smythe.planner import LLMPlanner, PlanningError
from smythe.provider import CompletionResult, Provider
from smythe.task import Task


FORK_JOIN_RESPONSE = json.dumps({
    "topology": ["fork_join", "serial"],
    "nodes": [
        {
            "id": "research-a",
            "label": "Research competitor A",
            "depends_on": [],
            "agent": {
                "name": "ResearcherA",
                "persona": "You are a market research analyst.",
                "capabilities": ["research", "analysis"],
            },
        },
        {
            "id": "research-b",
            "label": "Research competitor B",
            "depends_on": [],
            "agent": {
                "name": "ResearcherB",
                "persona": "You are a market research analyst.",
                "capabilities": ["research", "analysis"],
            },
        },
        {
            "id": "synthesize",
            "label": "Combine findings into report",
            "depends_on": ["research-a", "research-b"],
            "agent": {
                "name": "Synthesizer",
                "persona": "You are a report writer.",
                "capabilities": ["writing"],
            },
        },
    ],
})

SERIAL_RESPONSE = json.dumps({
    "topology": ["serial"],
    "nodes": [
        {
            "id": "step-1",
            "label": "Write the introduction",
            "depends_on": [],
            "agent": {
                "name": "Writer",
                "persona": "You are a technical writer.",
                "capabilities": ["writing"],
            },
        },
        {
            "id": "step-2",
            "label": "Proofread the introduction",
            "depends_on": ["step-1"],
            "agent": {
                "name": "Editor",
                "persona": "You are an editor.",
                "capabilities": ["editing"],
            },
        },
    ],
})

ADVERSARIAL_RESPONSE = json.dumps({
    "topology": ["fork_join", "adversarial", "serial"],
    "nodes": [
        {
            "id": "financial",
            "label": "Analyze financials",
            "depends_on": [],
            "agent": {"name": "FinAnalyst", "persona": "Financial analyst.", "capabilities": ["finance"]},
        },
        {
            "id": "tech",
            "label": "Assess technical IP",
            "depends_on": [],
            "agent": {"name": "TechAnalyst", "persona": "Tech diligence.", "capabilities": ["tech"]},
        },
        {
            "id": "merge",
            "label": "Merge findings",
            "depends_on": ["financial", "tech"],
            "agent": {"name": "Merger", "persona": "Report merger.", "capabilities": ["writing"]},
        },
        {
            "id": "red-team",
            "label": "Challenge assumptions",
            "depends_on": ["merge"],
            "agent": {"name": "RedTeam", "persona": "Devil's advocate.", "capabilities": ["critique"]},
            "metadata": {"role": "adversarial"},
        },
        {
            "id": "final",
            "label": "Produce final memo",
            "depends_on": ["red-team"],
            "agent": {"name": "MemoWriter", "persona": "Memo drafter.", "capabilities": ["writing"]},
        },
    ],
})


class MockPlanningProvider(Provider):
    """Provider that returns preconfigured responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.prompts_received: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.prompts_received.append(prompt)
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return CompletionResult(text=response, prompt_tokens=50, completion_tokens=100)


def test_plan_fork_join_task():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Research competitors and write a report")

    graph, registry = planner.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.FORK_JOIN, Topology.SERIAL]
    assert len(graph.nodes) == 3
    assert graph.nodes[2].id == "synthesize"
    assert set(graph.nodes[2].depends_on) == {"research-a", "research-b"}


def test_plan_serial_task():
    provider = MockPlanningProvider([SERIAL_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Write and proofread an introduction")

    graph, registry = planner.plan(task)

    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 2
    assert graph.nodes[1].depends_on == ["step-1"]


def test_plan_adversarial_task():
    provider = MockPlanningProvider([ADVERSARIAL_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Evaluate acquisition target with red-team review")

    graph, registry = planner.plan(task)

    assert graph.topology == [Topology.FORK_JOIN, Topology.ADVERSARIAL, Topology.SERIAL]
    assert len(graph.nodes) == 5
    red_team = next(n for n in graph.nodes if n.id == "red-team")
    assert red_team.depends_on == ["merge"]


def test_plan_creates_agents_with_personas():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Research competitors")

    graph, registry = planner.plan(task)

    agents = registry.list_agents()
    assert len(agents) == 3

    researcher = next(a for a in agents if a.profile.name == "ResearcherA")
    assert researcher.profile.persona == "You are a market research analyst."
    assert "research" in researcher.profile.capabilities


def test_plan_retries_on_malformed_json():
    provider = MockPlanningProvider([
        "this is not json",
        SERIAL_RESPONSE,
    ])
    planner = LLMPlanner(provider=provider, planning_model="test-model", max_retries=2)
    task = Task(goal="Write something")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2
    assert len(provider.prompts_received) == 2
    retry_prompt = provider.prompts_received[1]
    assert "Write something" in retry_prompt
    assert "could not be parsed" in retry_prompt


def test_plan_raises_after_max_retries():
    provider = MockPlanningProvider([
        "garbage",
        "still garbage",
        "yet more garbage",
    ])
    planner = LLMPlanner(provider=provider, planning_model="test-model", max_retries=2)
    task = Task(goal="This will fail")

    with pytest.raises(PlanningError, match="Failed to produce a valid plan"):
        planner.plan(task)

    assert len(provider.prompts_received) == 3


def test_json_extraction_strips_code_fences():
    fenced = '```json\n' + SERIAL_RESPONSE + '\n```'
    provider = MockPlanningProvider([fenced])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Test code fence stripping")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2


def test_json_extraction_strips_bare_fences():
    fenced = '```\n' + SERIAL_RESPONSE + '\n```'
    provider = MockPlanningProvider([fenced])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Test bare fence stripping")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2


def test_plan_includes_constraints_in_prompt():
    provider = MockPlanningProvider([SERIAL_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(
        goal="Plan a party",
        constraints=["Budget under $500", "Must be in Oakland"],
    )

    planner.plan(task)

    prompt = provider.prompts_received[0]
    assert "Budget under $500" in prompt
    assert "Must be in Oakland" in prompt


def test_plan_includes_history_in_prompt():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    memory = PlannerMemory(path=path)

    from smythe.memory import ExecutionOutcome
    import json as _json

    outcome = ExecutionOutcome(
        task_goal="Research competitors and write a report",
        task_constraints=[],
        topology=["fork_join", "serial"],
        node_count=3,
        total_cost_usd=0.15,
        total_duration_ms=4500,
        success=True,
        timestamp="2025-01-01T00:00:00Z",
    )
    with open(path, "w", encoding="utf-8") as f:
        from dataclasses import asdict
        f.write(_json.dumps(asdict(outcome)) + "\n")

    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model", memory=memory)
    task = Task(goal="Research competitors for a new product")

    planner.plan(task)

    prompt = provider.prompts_received[0]
    assert "past executions" in prompt.lower()
    assert "$0.15" in prompt

    import os
    os.unlink(path)


def test_estimated_cost_set_on_graph():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMPlanner(
        provider=provider,
        planning_model="test-model",
        cost_per_token=0.000003,
        avg_tokens_per_node=2000,
    )
    task = Task(goal="Research competitors")

    graph, _ = planner.plan(task)

    assert graph.estimated_cost_usd is not None
    expected = 3 * 2000 * 0.000003
    assert abs(graph.estimated_cost_usd - expected) < 1e-10


@pytest.mark.asyncio
async def test_llm_planner_aplan():
    """aplan() works directly from an async context without nested event loops."""
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMPlanner(provider=provider, planning_model="test-model")
    task = Task(goal="Research competitors async")

    graph, registry = await planner.aplan(task)

    assert isinstance(graph, ExecutionGraph)
    assert len(graph.nodes) == 3
    assert len(registry.list_agents()) == 3
