"""Prompt templates for the ConstrainedArchitect."""

from __future__ import annotations

from typing import Any

from smythe.task import Task


CONSTRAINED_SYSTEM_PROMPT = """\
You are a workflow planner.  Given a user's goal and a menu of available \
sub-graph templates, select which templates to use and in what order.

## Rules

1. You may ONLY select templates from the menu below.  Do NOT invent \
   new templates or modify existing ones.
2. Return a JSON array of template selections.  Each selection is an \
   object with a required "template" key (the template name) and an \
   optional "params" object for template-specific parameters.
3. Templates are executed in the order listed.  The output of each \
   template flows as dependency context into the next.
4. Return **only** the JSON array — no prose, no markdown fences.

## Output schema

```
[
  {"template": "template-name", "params": {"key": "value"}},
  {"template": "another-template"}
]
```
"""

CONSTRAINED_RETRY_PROMPT = """\
Your previous response was not valid JSON.  Please try again.

Return **only** the JSON array of template selections — \
no markdown fences, no commentary.  Just the raw JSON array.
"""


def build_constrained_user_prompt(
    task: Task,
    templates: list[dict[str, Any]],
) -> str:
    """Assemble the user prompt from a Task and available templates."""
    parts: list[str] = []

    parts.append(f"## Goal\n\n{task.goal}")

    if task.constraints:
        constraints_text = "\n".join(f"- {c}" for c in task.constraints)
        parts.append(f"## Constraints\n\n{constraints_text}")

    menu_lines: list[str] = []
    for t in templates:
        menu_lines.append(f"- **{t['name']}**: {t['description']}")
    parts.append("## Available templates\n\n" + "\n".join(menu_lines))

    parts.append("Select the templates to use.  Return only the JSON array.")

    return "\n\n".join(parts)
