"""Tests for planner tool awareness — the available-agents inventory."""

import asyncio
import json

from smythe.agent import Agent, AgentProfile
from smythe.mcp import MCPServerSpec
from smythe.planner import LLMArchitect
from smythe.prompts import (
    MAX_INVENTORY_AGENTS,
    MAX_INVENTORY_TOOLS,
    build_agent_inventory,
    build_user_prompt,
)
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.task import Task


def _registry_with(*agents: Agent) -> Registry:
    registry = Registry()
    for a in agents:
        registry.register(a)
    return registry


def _tool_agent(name: str = "Researcher", tools: tuple[str, ...] = ("read_file",)) -> Agent:
    spec = MCPServerSpec(
        name="fs", transport="stdio", command="server", allowed_tools=tools,
    )
    return Agent(profile=AgentProfile(
        name=name, capabilities=["research"], mcp_servers=[spec],
    ))


def test_inventory_none_for_no_registry_or_empty():
    assert build_agent_inventory(None) is None
    assert build_agent_inventory(Registry()) is None


def test_inventory_lists_name_capabilities_and_tools():
    inventory = build_agent_inventory(_registry_with(_tool_agent()))
    assert "Researcher" in inventory
    assert "capabilities: research" in inventory
    assert "tools[fs]: read_file" in inventory


def test_inventory_marks_unlisted_servers_as_runtime_discovered():
    spec = MCPServerSpec(name="gh", transport="http", url="http://x")
    agent = Agent(profile=AgentProfile(name="Dev", mcp_servers=[spec]))
    inventory = build_agent_inventory(_registry_with(agent))
    assert "tools[gh]: (discovered at runtime)" in inventory


def test_inventory_elides_long_tool_lists():
    tools = tuple(f"tool_{i}" for i in range(MAX_INVENTORY_TOOLS + 5))
    inventory = build_agent_inventory(_registry_with(_tool_agent(tools=tools)))
    assert "+5 more" in inventory
    assert f"tool_{MAX_INVENTORY_TOOLS}" not in inventory


def test_inventory_elides_agent_overflow():
    agents = [
        Agent(profile=AgentProfile(name=f"agent-{i:03d}"))
        for i in range(MAX_INVENTORY_AGENTS + 3)
    ]
    inventory = build_agent_inventory(_registry_with(*agents))
    assert "...and 3 more agents" in inventory


def test_prompt_without_inventory_is_unchanged():
    task = Task(goal="Do a thing")
    assert build_user_prompt(task) == build_user_prompt(task, agent_inventory=None)
    assert "Available agents" not in build_user_prompt(task)


def test_prompt_with_inventory_instructs_required_capabilities():
    prompt = build_user_prompt(
        Task(goal="Do a thing"),
        agent_inventory="- Researcher | capabilities: research",
    )
    assert "## Available agents" in prompt
    assert "required_capabilities" in prompt
    assert "- Researcher | capabilities: research" in prompt


class CapturingPlanProvider(Provider):
    """Returns a valid plan and records the planning prompt it received."""

    PLAN = json.dumps({
        "topology": ["serial"],
        "nodes": [{"id": "n1", "label": "Do it", "depends_on": []}],
    })

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self.prompts.append(prompt)
        return CompletionResult(text=self.PLAN, prompt_tokens=10, completion_tokens=10)


def test_llm_architect_includes_inventory_when_registry_present():
    provider = CapturingPlanProvider()
    registry = _registry_with(_tool_agent())
    architect = LLMArchitect(provider=provider, registry=registry)

    asyncio.run(architect.aplan(Task(goal="Research something")))

    [prompt] = provider.prompts
    assert "## Available agents" in prompt
    assert "tools[fs]: read_file" in prompt


def test_llm_architect_prompt_unchanged_without_registry():
    provider = CapturingPlanProvider()
    architect = LLMArchitect(provider=provider)
    asyncio.run(architect.aplan(Task(goal="Research something")))
    [prompt] = provider.prompts
    assert "Available agents" not in prompt
