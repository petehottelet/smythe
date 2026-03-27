"""Planning prompt templates for LLM-driven task decomposition."""

from __future__ import annotations

from typing import Any

from smythe.task import Task

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
4. Prefer 2–8 nodes.  A single-node graph is acceptable for trivial tasks.
5. Give each agent a meaningful `persona` that guides its behaviour.
6. Assign `capabilities` tags that describe the agent's expertise.
7. For fork-join: create parallel root nodes and a join node that depends on all of them.
8. For broadcast-reduce: create a setup node, parallel worker nodes depending on it, \
   and a reduce node depending on all workers.
9. For adversarial: insert a review node after the main work, before the final output.
"""

RETRY_PROMPT = """\
Your previous response was not valid JSON.  Please try again.

Return **only** the JSON object described in the system prompt — \
no markdown fences, no commentary, no explanation.  Just the raw JSON.
"""


def build_user_prompt(
    task: Task,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Assemble the user prompt from a Task and optional execution history."""
    parts: list[str] = []

    parts.append(f"## Goal\n\n{task.goal}")

    if task.constraints:
        constraints_text = "\n".join(f"- {c}" for c in task.constraints)
        parts.append(f"## Constraints\n\n{constraints_text}")

    if task.context:
        ctx_lines = "\n".join(f"- {k}: {v}" for k, v in task.context.items())
        parts.append(f"## Context\n\n{ctx_lines}")

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

    parts.append("Respond with only the JSON object.")

    return "\n\n".join(parts)
