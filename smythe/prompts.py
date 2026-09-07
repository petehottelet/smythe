"""Planning prompt templates for LLM-driven task decomposition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smythe.task import Task, render_task

if TYPE_CHECKING:
    from smythe.registry import Registry

# Bounds on the available-agents section so planning-prompt size stays
# controlled even with large registries or tool-heavy agents.
MAX_INVENTORY_AGENTS = 40
MAX_INVENTORY_TOOLS = 15

PLANNING_SYSTEM_PROMPT = """\
You are a task-decomposition planner.  Given a user's goal and constraints, \
produce an execution plan as a JSON object.

## Available topologies

Choose one or more phases from this list.  Combine them as a JSON array \
to express compound execution patterns (e.g. ["fork_join", "adversarial", "serial"]).

- **serial** — sequential steps where each depends on the previous.  \
  Use for simple, linear workflows.
- **fork_join** — independent parallel branches that merge at a join node.  \
  Use when work can be researched, compared, or analyzed in parallel \
  (keywords: research, compare, analyze, investigate).
- **broadcast_reduce** — a setup step broadcasts context to many parallel \
  workers, then a reducer curates the results.  Use for generating \
  multiple variants or assets from a shared brief \
  (keywords: generate assets, create variants, produce alternatives).
- **adversarial** — a red-team review phase inserted after initial work.  \
  Use when claims should be stress-tested, audited, or challenged \
  (keywords: evaluate, diligence, review, audit, red-team).

## Output schema

Respond with **only** a JSON object — no prose, no markdown fences.

```
{
  "topology": ["fork_join", "serial"],
  "nodes": [
    {
      "id": "short-kebab-id",
      "label": "Human-readable description of what this step does",
      "depends_on": [],
      "agent": {
        "name": "AgentRoleName",
        "persona": "You are a ...",
        "capabilities": ["skill1", "skill2"]
      }
    }
  ]
}
```

## Rules

1. Every node must have a unique `id` (short, kebab-case).
2. `depends_on` lists node IDs that must complete before this node starts.  \
   The graph must be acyclic.
3. Keep graphs shallow — depth <= 5 levels.
4. Right-size the graph. **Start from one node and justify every \
   addition** — do not start from a pipeline and trim. Every node costs \
   money, latency, and a hand-off where detail is lost. \
   Apply this test to each node you are about to add: name the distinct \
   work product it contributes that no other node produces. If you \
   cannot name one, do not add it. \
   Use exactly one node when the task has a single deliverable and no \
   independently investigable parts — a transformation, a calculation, \
   a rewrite, a focused piece of writing. Splitting these produces \
   worse results than one competent pass, because each hand-off loses \
   specifics. \
   Add nodes when the work has genuinely separable parts: distinct \
   subjects that can be investigated without reference to each other, a \
   deliverable that must be attacked by an adversarial reviewer, or \
   stages where each consumes the previous stage's *output* rather than \
   merely following it in time. \
   Fan out in parallel only when the branches are truly independent; do \
   not split one analysis into thin slices. 8 nodes is the ceiling.
5. **The last node returns the deliverable, so it must produce all of \
   it.** Only that node's output is handed back — earlier nodes' work is \
   not shown alongside it. When the goal asks for several parts (an \
   argument *and* a critique of it; per-step arithmetic *and* a \
   recommendation; findings *and* a corrected version), the final node's \
   label must say it assembles the complete deliverable, listing those \
   parts. A label like "state what survives" produces only the \
   survivors and silently discards the rest of what was asked for. \
   Write "Assemble the final deliverable: the case, the critique, and \
   the surviving claims" instead.
6. Give each agent a meaningful `persona` that guides its behaviour.
7. Assign `capabilities` tags that describe the agent's expertise.
8. For fork-join: create parallel root nodes and a join node that depends on all of them.
9. For broadcast-reduce: create a setup node, parallel worker nodes depending on it, \
   and a reduce node depending on all workers.
10. For adversarial: insert a review node after the main work, before the final output.
11. **Only when the task states acceptance criteria**, you may add one \
   gating node that checks the deliverable against them:

   ```
   {"id": "check", "label": "Verify the memo meets every acceptance \
criterion; answer PASS or FAIL with reasons", "depends_on": ["memo"], \
"verifies": "memo", "max_regenerations": 1}
   ```

   `verifies` names the node being judged; a FAIL verdict re-runs that \
   node and everything downstream of it, up to `max_regenerations` \
   times. Use at most one such node, put it on the node that produces \
   the deliverable, and keep `max_regenerations` at 1 — each retry pays \
   for the subtree again. Omit it entirely when no criteria are stated.
"""

RETRY_PROMPT = """\
Your previous response was not valid JSON.  Please try again.

Return **only** the JSON object described in the system prompt — \
no markdown fences, no commentary, no explanation.  Just the raw JSON.
"""


def build_agent_inventory(registry: Registry | None) -> str | None:
    """One line per registered agent: name, capabilities, tool summary.

    Returns None when there is nothing to describe, so the planning
    prompt is byte-identical to the inventory-free prompt in that case.
    """
    if registry is None:
        return None
    agents = registry.list_agents()
    if not agents:
        return None

    lines: list[str] = []
    for agent in agents[:MAX_INVENTORY_AGENTS]:
        bits = [agent.profile.name]
        caps = registry.effective_capabilities(agent)
        if caps:
            bits.append("capabilities: " + ", ".join(sorted(caps)))
        for spec in getattr(agent.profile, "mcp_servers", []) or []:
            if spec.allowed_tools:
                tools = list(spec.allowed_tools)
                shown = tools[:MAX_INVENTORY_TOOLS]
                extra = len(tools) - len(shown)
                summary = ", ".join(shown) + (f", +{extra} more" if extra else "")
                bits.append(f"tools[{spec.name}]: {summary}")
            else:
                bits.append(f"tools[{spec.name}]: (discovered at runtime)")
        lines.append("- " + " | ".join(bits))

    if len(agents) > MAX_INVENTORY_AGENTS:
        lines.append(f"- ...and {len(agents) - MAX_INVENTORY_AGENTS} more agents")
    return "\n".join(lines)


def build_user_prompt(
    task: Task,
    history: list[dict[str, Any]] | None = None,
    agent_inventory: str | None = None,
) -> str:
    """Assemble the user prompt from a Task, optional history, and inventory."""
    parts: list[str] = []

    parts.append(f"## Task\n\n{render_task(task)}")

    if task.done_when:
        parts.append(
            "## Acceptance criteria\n\n"
            "The deliverable is not done until it meets all of these. "
            "Make some node in your plan accountable for each criterion "
            "listed under Done when above."
        )

    if history:
        history_lines: list[str] = []
        for outcome in history:
            topo = " → ".join(outcome.get("topology", []))
            cost = outcome.get("total_cost_usd", 0)
            duration = outcome.get("total_duration_ms", 0)
            success = "success" if outcome.get("success") else "failure"
            goal = outcome.get("task_goal", "")
            history_lines.append(
                f"- Goal: {goal!r} | Topology: {topo} | "
                f"Cost: ${cost:.2f} | Duration: {duration:.0f}ms | "
                f"Outcome: {success}"
            )
        parts.append(
            "## Relevant past executions\n\n"
            "Use these to inform your topology choice:\n"
            + "\n".join(history_lines)
        )

    if agent_inventory:
        parts.append(
            "## Available agents\n\n"
            "These agents already exist and their tools are real. Design the "
            "plan to exploit them: to route a node to one of these agents, "
            'set "required_capabilities" on the node to a subset of that '
            "agent's capabilities instead of inventing a new agent.\n\n"
            + agent_inventory
        )

    parts.append("Respond with only the JSON object.")

    return "\n\n".join(parts)
