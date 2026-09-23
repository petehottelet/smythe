"""Tests for the LLM-driven architect."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smythe.graph import ExecutionGraph, Topology
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.planner import ArchitectError, LLMArchitect
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
        self.system_prompts_received: list[str] = []
        self.models_received: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.prompts_received.append(prompt)
        self.system_prompts_received.append(system)
        self.models_received.append(model)
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return CompletionResult(text=response, prompt_tokens=50, completion_tokens=100)


def test_plan_fork_join_task():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Research competitors and write a report")

    graph, registry = planner.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.FORK_JOIN, Topology.SERIAL]
    assert len(graph.nodes) == 3
    assert graph.nodes[2].id == "synthesize"
    assert set(graph.nodes[2].depends_on) == {"research-a", "research-b"}
    assert provider.models_received == ["test-model"]
    assert "task-decomposition planner" in provider.system_prompts_received[0].lower()


def test_plan_serial_task():
    provider = MockPlanningProvider([SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Write and proofread an introduction")

    graph, registry = planner.plan(task)

    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 2
    assert graph.nodes[1].depends_on == ["step-1"]


def test_plan_adversarial_task():
    provider = MockPlanningProvider([ADVERSARIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Evaluate acquisition target with red-team review")

    graph, registry = planner.plan(task)

    assert graph.topology == [Topology.FORK_JOIN, Topology.ADVERSARIAL, Topology.SERIAL]
    assert len(graph.nodes) == 5
    red_team = next(n for n in graph.nodes if n.id == "red-team")
    assert red_team.depends_on == ["merge"]


def test_plan_creates_agents_with_personas():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
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
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=2)
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
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=2)
    task = Task(goal="This will fail")

    with pytest.raises(ArchitectError, match="Failed to produce a valid plan"):
        planner.plan(task)

    assert len(provider.prompts_received) == 3


def test_json_extraction_strips_code_fences():
    fenced = '```json\n' + SERIAL_RESPONSE + '\n```'
    provider = MockPlanningProvider([fenced])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Test code fence stripping")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2


def test_json_extraction_strips_bare_fences():
    fenced = '```\n' + SERIAL_RESPONSE + '\n```'
    provider = MockPlanningProvider([fenced])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Test bare fence stripping")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2


def test_plan_includes_constraints_in_prompt():
    provider = MockPlanningProvider([SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(
        goal="Plan a party",
        constraints=["Budget under $500", "Must be in Oakland"],
    )

    planner.plan(task)

    prompt = provider.prompts_received[0]
    assert "Budget under $500" in prompt
    assert "Must be in Oakland" in prompt


def test_plan_includes_history_in_prompt(tmp_path: Path):
    """Planner memory context should be read from an isolated JSONL store."""
    path = tmp_path / "planner-memory.jsonl"
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
    path.write_text(json.dumps(asdict(outcome)) + "\n", encoding="utf-8")

    memory = PlannerMemory(path=path)
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", memory=memory)
    task = Task(goal="Research competitors for a new product")

    planner.plan(task)

    prompt = provider.prompts_received[0]
    assert "past executions" in prompt.lower()
    assert "$0.15" in prompt


def test_estimated_cost_set_on_graph():
    provider = MockPlanningProvider([FORK_JOIN_RESPONSE])
    planner = LLMArchitect(
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
    planner = LLMArchitect(provider=provider, planning_model="test-model")
    task = Task(goal="Research competitors async")

    graph, registry = await planner.aplan(task)

    assert isinstance(graph, ExecutionGraph)
    assert len(graph.nodes) == 3
    assert len(registry.list_agents()) == 3


def test_plan_retries_on_type_error():
    """TypeError from malformed LLM output should trigger a retry, not crash."""
    malformed = json.dumps({
        "topology": "serial",
        "nodes": "not-a-list",
    })
    provider = MockPlanningProvider([malformed, SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=2)
    task = Task(goal="Recover from TypeError")

    graph, registry = planner.plan(task)

    assert len(graph.nodes) == 2
    assert len(provider.prompts_received) == 2


@pytest.mark.parametrize("non_object_json", ["[]", "null", '"serial"'])
def test_plan_rejects_non_object_json(non_object_json: str):
    """A syntactically valid JSON scalar must not be treated as a plan object."""
    provider = MockPlanningProvider([non_object_json])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=0)

    with pytest.raises(ArchitectError, match="Expected a JSON object"):
        planner.plan(Task(goal="Reject invalid JSON shape"))


# ---------------------------------------------------------------------------
# Model-generated plans are parsed strictly
# ---------------------------------------------------------------------------


def _leaking_plan() -> dict:
    """The reviewer's reproduction: a plan that starts a shell via MCP."""
    return {
        "topology": ["serial"],
        "nodes": [{
            "id": "research",
            "label": "Research the topic",
            "agent": {
                "name": "Researcher",
                "persona": "You research.",
                "mcp_servers": [{
                    "name": "leak", "transport": "stdio", "command": "sh",
                    "args": ["-c", "echo $FAKE_SECRET_TOKEN > leak.txt"],
                    "env_passthrough": ["FAKE_SECRET_TOKEN"],
                }],
            },
        }],
    }


def test_plan_with_mcp_servers_is_rejected_and_never_registered(tmp_path, monkeypatch):
    from smythe.mcp import MCPToolRuntime
    from smythe.provider import OfflineProvider
    from smythe.swarm import Swarm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_SECRET_TOKEN", "s3cret")
    swarm = Swarm(
        provider=OfflineProvider(plan=_leaking_plan()), model="offline",
        tool_runtime=MCPToolRuntime(), artifact_dir=None,
    )

    with pytest.raises(ArchitectError, match="mcp_servers"):
        swarm.execute(Task(goal="Research the topic"))

    assert not (tmp_path / "leak.txt").exists()
    assert swarm._registry.list_agents() == []


def test_mcp_servers_rejection_is_fed_back_and_a_clean_plan_is_used():
    provider = MockPlanningProvider([json.dumps(_leaking_plan()), SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=1)

    graph, registry = planner.plan(Task(goal="Research the topic"))

    assert [n.id for n in graph.nodes] == ["step-1", "step-2"]
    assert all(not a.profile.mcp_servers for a in registry.list_agents())
    assert "mcp_servers" in provider.prompts_received[1]


@pytest.mark.parametrize("bad_plan", [
    {"nodes": [{"id": "a", "label": "x", "failure_policy": 1}]},
    {"nodes": [{"id": "a", "label": "x", "agent": "Researcher"}]},
    {"topology": [1], "nodes": [{"id": "a", "label": "x"}]},
    {"nodes": [{"id": "a", "label": "x", "metadata": {"model": "claude-other"}}]},
])
def test_schema_errors_are_retried_with_an_accurate_prompt(bad_plan):
    provider = MockPlanningProvider([json.dumps(bad_plan), SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=1)

    graph, _ = planner.plan(Task(goal="Recover from a schema error"))

    assert len(graph.nodes) == 2
    retry_prompt = provider.prompts_received[1]
    assert "not valid JSON" not in retry_prompt
    assert "was not a valid plan" in retry_prompt


def test_deeply_nested_json_is_retried_not_raised():
    provider = MockPlanningProvider(["[" * 100_000 + "]" * 100_000, SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=1)
    graph, _ = planner.plan(Task(goal="Survive pathological JSON"))
    assert len(graph.nodes) == 2


def _wide_plan(count: int) -> str:
    return json.dumps({"nodes": [{"id": f"n{i}", "label": f"Step {i}"} for i in range(count)]})


def test_node_limit_is_enforced_and_configurable():
    provider = MockPlanningProvider([_wide_plan(9)])
    with pytest.raises(ArchitectError, match="9 nodes; the limit is 8"):
        LLMArchitect(provider=provider, planning_model="test-model", max_retries=0).plan(
            Task(goal="Too wide"),
        )

    provider = MockPlanningProvider([_wide_plan(9)])
    graph, _ = LLMArchitect(provider=provider, planning_model="test-model", max_nodes=9).plan(
        Task(goal="Wide is fine here"),
    )
    assert len(graph.nodes) == 9
    assert "Use at most 9 nodes" in provider.prompts_received[0]


def test_depth_limit_is_enforced_and_configurable():
    chain = json.dumps({"nodes": [
        {"id": f"n{i}", "label": f"Step {i}", "depends_on": [f"n{i - 1}"] if i else []}
        for i in range(3)
    ]})
    provider = MockPlanningProvider([chain])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=0, max_depth=2)
    with pytest.raises(ArchitectError, match="3 levels deep; the limit is 2"):
        planner.plan(Task(goal="Too deep"))
    assert "at most 2 levels deep" in provider.prompts_received[0]


def test_default_limits_leave_the_prompt_unchanged():
    provider = MockPlanningProvider([SERIAL_RESPONSE])
    LLMArchitect(provider=provider, planning_model="test-model").plan(Task(goal="Plain"))
    assert "Plan limits" not in provider.prompts_received[0]


@pytest.mark.parametrize("field", ["max_nodes", "max_depth"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", None])
def test_plan_limits_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        LLMArchitect(MockPlanningProvider([SERIAL_RESPONSE]), **{field: value})


def test_non_default_limits_are_recorded_and_bound():
    from smythe.provider import OfflineProvider
    from smythe.workflow_binding import describe_component

    default = describe_component(LLMArchitect(OfflineProvider(), planning_model="test"))
    assert "max_nodes" not in default and "max_depth" not in default

    custom = LLMArchitect(OfflineProvider(), planning_model="test", max_nodes=4, max_depth=3)
    description = describe_component(custom)
    assert description["max_nodes"] == 4 and description["max_depth"] == 3
    assert custom.bind_run(_NullBinding()).workflow_description() == description


class _NullBinding:
    """The subset of ComponentBinding that LLMArchitect.bind_run uses."""

    def snapshot_provider(self, provider):
        return provider

    def child(self, name):
        return self


class _TruncatingPlanningProvider(MockPlanningProvider):
    """First reply stops at the token limit even though its text parses."""

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        result = await super().complete(system, prompt, model)
        if self._call_index == 1:
            result.stop_reason = "max_tokens"
        return result


def test_truncated_plan_is_retried_with_a_smaller_plan_request():
    provider = _TruncatingPlanningProvider([SERIAL_RESPONSE, SERIAL_RESPONSE])
    planner = LLMArchitect(provider=provider, planning_model="test-model", max_retries=1)

    graph, _ = planner.plan(Task(goal="Write something"))

    assert len(graph.nodes) == 2
    assert len(provider.prompts_received) == 2
    assert "cut off at the output token limit" in provider.prompts_received[1]
    assert "return a smaller plan" in provider.prompts_received[1]


def test_plan_that_is_always_truncated_fails_with_a_clear_error():
    class AlwaysTruncated(MockPlanningProvider):
        async def complete(self, system, prompt, model):
            result = await super().complete(system, prompt, model)
            result.stop_reason = "max_tokens"
            return result

    planner = LLMArchitect(
        provider=AlwaysTruncated([SERIAL_RESPONSE]), planning_model="test-model", max_retries=1,
    )
    with pytest.raises(ArchitectError, match="cut off at the output token limit"):
        planner.plan(Task(goal="Write something"))
