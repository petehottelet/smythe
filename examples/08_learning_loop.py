"""Learning loop: the Architect plans differently after remembering.

    python examples/08_learning_loop.py

Two similar tasks run through the same Swarm with PlannerMemory
enabled. Run 1's Architect picks an eight-node broadcast — expensive
and scattered. That outcome (topology, cost, success) is recorded.
When run 2 plans, the recalled outcome appears in the planning prompt
under "Relevant past executions", and the Architect returns a leaner
graph.

Offline, the provider is scripted to make the mechanism visible: it
returns the bloated plan when the prompt has no history and the lean
plan when it does — so what this example proves is the wiring itself
(record -> recall -> prompt -> different plan). With an API key set, a
real model sees the same recalled history and makes its own call.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from smythe import Swarm, Task
from smythe.memory import PlannerMemory
from smythe.provider import CompletionResult, OfflineProvider

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

HISTORY_MARKER = "## Relevant past executions"

NAIVE_PLAN = {
    "topology": ["broadcast_reduce"],
    "nodes": [
        {"id": "brief", "label": "Write the shared research brief", "depends_on": [],
         "agent": {"name": "BriefWriter", "persona": "You write research briefs."}},
        *[
            {"id": f"angle-{i}", "label": f"Research angle {i} of the market",
             "depends_on": ["brief"],
             "agent": {"name": f"Researcher{i}", "persona": "You research one angle."}}
            for i in range(1, 7)
        ],
        {"id": "reduce", "label": "Curate all angles into the brief",
         "depends_on": [f"angle-{i}" for i in range(1, 7)],
         "agent": {"name": "Curator", "persona": "You curate research."}},
    ],
}

LEAN_PLAN = {
    "topology": ["fork_join"],
    "nodes": [
        {"id": "market", "label": "Research the market landscape", "depends_on": [],
         "agent": {"name": "MarketResearcher", "persona": "You research markets."}},
        {"id": "rivals", "label": "Profile the leading competitors", "depends_on": [],
         "agent": {"name": "CompetitorAnalyst", "persona": "You profile competitors."}},
        {"id": "brief", "label": "Merge both tracks into the final brief",
         "depends_on": ["market", "rivals"],
         "agent": {"name": "Editor", "persona": "You write crisp briefs."}},
    ],
}


class LearningDemoProvider(OfflineProvider):
    """Scripted planner: bloated plan without history, lean plan with it."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_history: bool | None = None

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        from smythe.prompts import PLANNING_SYSTEM_PROMPT

        if system == PLANNING_SYSTEM_PROMPT:
            self.saw_history = HISTORY_MARKER in prompt
            plan = LEAN_PLAN if self.saw_history else NAIVE_PLAN
            return CompletionResult(
                text=json.dumps(plan), prompt_tokens=250, completion_tokens=400,
            )
        return await super().complete(system, prompt, model)


def pick_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from smythe.provider import AnthropicProvider
        return AnthropicProvider(), "claude-mythos"
    if os.environ.get("OPENAI_API_KEY"):
        from smythe.provider import OpenAIProvider
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        from smythe.provider import GeminiProvider
        return GeminiProvider(), "gemini-3-flash"
    print("No API key found - running offline with a scripted planner.")
    print("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY for real output.\n")
    return LearningDemoProvider(), "demo-model"


provider, model = pick_provider()
memory = PlannerMemory(
    path=Path(tempfile.mkdtemp(prefix="smythe-learning-demo-")) / "history.jsonl"
)
swarm = Swarm(provider=provider, model=model, memory=memory, parallel=True,
              max_budget_usd=1.00)

run1 = Task(
    goal=("Produce a competitive brief on portable espresso makers: "
          "market landscape, top competitors, and a one-page summary."),
    constraints=["Keep the final brief under 400 words"],
)
run2 = Task(
    goal=("Produce a competitive brief on portable cold-brew makers: "
          "market landscape, top competitors, and a one-page summary."),
    constraints=["Keep the final brief under 400 words"],
)

# --- Run 1: no history yet ---------------------------------------------
result1 = swarm.execute(run1)
graph1 = result1.graph
print("=== Run 1 (memory empty) ===")
print(f"Topology: {' -> '.join(t.value for t in graph1.topology)}"
      f" | Nodes: {len(graph1.nodes)} | Cost: ${result1.total_cost_usd:.4f}")

# --- What memory recorded ----------------------------------------------
outcome = json.loads(memory.path.read_text(encoding="utf-8").splitlines()[0])
outcome["total_cost_usd"] = round(outcome["total_cost_usd"], 6)
print("\n=== Recorded outcome ===")
print(json.dumps({k: outcome[k] for k in
                  ("task_goal", "topology", "node_count", "total_cost_usd", "success")},
                 indent=2))

# --- Run 2: a similar task, planned with recall -------------------------
result2 = swarm.execute(run2)
graph2 = result2.graph
print("\n=== Run 2 (memory recalled) ===")
if isinstance(provider, LearningDemoProvider):
    seen = "yes" if provider.saw_history else "NO (wiring broken!)"
    print(f'Planning prompt contained "{HISTORY_MARKER}": {seen}')
print(f"Topology: {' -> '.join(t.value for t in graph2.topology)}"
      f" | Nodes: {len(graph2.nodes)} | Cost: ${result2.total_cost_usd:.4f}")

print("\n=== Comparison ===")
print(f"Run 1: {len(graph1.nodes)} nodes, ${result1.total_cost_usd:.4f}"
      f"  ->  Run 2: {len(graph2.nodes)} nodes, ${result2.total_cost_usd:.4f}")
if len(graph2.nodes) < len(graph1.nodes):
    print("The recalled outcome changed the plan: fewer nodes, lower cost.")
