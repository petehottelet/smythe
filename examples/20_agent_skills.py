"""Python-backed skills: an Agent Skill becomes a tool that only the agent you name can call.

    python examples/20_agent_skills.py

A skills directory holds one Agent Skill, release-notes. Its SKILL.md asks
for Bash and Write and tells the model to run a script. Discovery reads that
one file: the skill lends both agents a capability tag, its declared tools
are recorded and never granted, and its instructions reach no prompt.
python_skill() makes the skill a tool backed by a Python function, and a
LocalToolRuntime offers that tool to the developer agent's id and to no one
else.

A generated plan routes both steps through the skill's tag. The model calls
the tool in the developer's drafting step; the editor holds the same tag and
is offered no tool. The tool's identity, which a template record can pin, is
printed at the end.

The run is offline and deterministic: a small scripted provider stands in
for the model and calls every tool it is offered, because OfflineProvider
does not call tools.

Docs: docs/skills.md#python-backed-skills
"""

import sys
import tempfile
from pathlib import Path

from smythe import (
    AgentSkillsProvider, CompletionResult, LocalToolRuntime, OfflineProvider, Swarm, Task, ToolCall,
    python_skill,
)
from smythe.agent import Agent, AgentProfile
from smythe.registry import Registry

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

SKILL_MD = """---
name: release-notes
description: Collect the changes merged since a release tag, grouped by area.
allowed-tools: Bash Write
---
# Release notes

Run scripts/collect.sh, then write the notes to NOTES.md.
"""

# An in-memory stand-in for a repository's merge history.
MERGED = {"v1.4.0": [("cli", "Add runs review"), ("docs", "Document Python-backed skills"),
                     ("cli", "Exit 12 while an action is unresolved")]}

PLAN = {"topology": ["serial"], "nodes": [
    {"id": "draft", "label": "Draft release notes for the changes since v1.4.0",
     "required_capabilities": ["release-notes", "coding"]},
    {"id": "edit", "label": "Edit the release notes for tone", "depends_on": ["draft"],
     "required_capabilities": ["release-notes", "editing"]},
]}


def collect_changes(arguments: dict) -> str:
    """The skill's Python implementation: group the changes merged since a tag."""
    since = arguments.get("since")
    if since not in MERGED:
        raise ValueError(f"unknown release tag {since!r}")
    areas: dict[str, list[str]] = {}
    for area, title in MERGED[since]:
        areas.setdefault(area, []).append(title)
    return "\n".join(f"- {area}: {'; '.join(titles)}" for area, titles in sorted(areas.items()))


def step_of(prompt: str) -> str:
    lines = prompt.splitlines()
    return next((line.removeprefix("Your step: ") for line in lines if line.startswith("Your step: ")),
                lines[0])


class ScriptedToolCaller(OfflineProvider):
    """Plans from PLAN, calls every tool it is offered, and answers from the results."""

    def __init__(self) -> None:
        super().__init__(plan=PLAN)
        self.offers: dict[str, list[str]] = {}

    async def chat(self, system, messages, model, tools=None):
        if messages[-1].tool_results:
            found = "\n".join(result.content for result in messages[-1].tool_results)
            return CompletionResult(f"Changes since v1.4.0:\n{found}", prompt_tokens=80,
                                    completion_tokens=40)
        self.offers[step_of(messages[0].content)] = [spec.name for spec in tools or ()]
        if not tools:
            return await super().chat(system, messages, model, tools)
        return CompletionResult(
            "", stop_reason="tool_use", prompt_tokens=60, completion_tokens=20,
            tool_calls=[ToolCall(f"call-{index}", spec.name, {"since": "v1.4.0"})
                        for index, spec in enumerate(tools)],
        )


root = Path(tempfile.mkdtemp(prefix="smythe-skills-demo-")) / "skills"
(root / "release-notes").mkdir(parents=True)
(root / "release-notes" / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

skills = AgentSkillsProvider({"developer": [root], "editor": [root]})
registry = Registry(skill_provider=skills)
registry.register(Agent(AgentProfile("Developer", persona="You maintain release tooling.",
                                     capabilities=["coding"]), id="developer"))
registry.register(Agent(AgentProfile("Editor", persona="You edit prose for clarity.",
                                     capabilities=["editing"]), id="editor"))

skill = skills.skill("release-notes")
tool = python_skill(skill, collect_changes, version="1", timeout_s=5, input_schema={
    "type": "object", "properties": {"since": {"type": "string"}}, "required": ["since"],
})
runtime = LocalToolRuntime([tool], agents={"developer": [tool.name]})
provider = ScriptedToolCaller()
swarm = Swarm(model="offline", provider=provider, registry=registry, tool_runtime=runtime,
              max_budget_usd=0.50, artifact_dir=None)

print("Running offline with a scripted provider (no keys, no cost).\n")
print("=== The skill ===")
print(f"  {skill.name}: {skill.description}")
print(f"  version:                {skill.version[:19]}...")
print(f"  declared allowed-tools: {', '.join(skill.declared_allowed_tools)} (recorded, never granted)")
for agent in registry.list_agents():
    print(f"  {agent.name} capabilities: {', '.join(registry.effective_capabilities(agent))}")

result = swarm.execute(Task("Publish the release notes for v1.5.0"))

print("\n=== Tools offered to each step ===")
for node in result.graph.nodes:
    if node.label in provider.offers:
        offered = ", ".join(provider.offers[node.label]) or "none"
        print(f"  {node.id} ({registry.get(node.agent_id).name}): {offered}")
print("\n=== Tool calls (from the trace) ===")
for span in result.trace:
    for call in span.get("tool_calls", []):
        print(f"  {call['tool']}  [{'ERROR' if call['is_error'] else 'ok'}]")
print("\n=== Drafting step ===")
print("  " + result.graph.nodes[0].result.replace("\n", "\n  "))
print("\n=== Tool identity for records ===")
for entry in runtime.describe():
    print(f"  {entry['name']}  version {entry['version']}  schema {entry['schema_sha256'][:12]}...")
print(f"\nFinal output: {result.output}")
