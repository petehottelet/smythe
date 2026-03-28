"""ConstrainedArchitect — LLM selects from a menu of pre-built sub-graph templates."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from smythe.constrained_prompts import (
    CONSTRAINED_RETRY_PROMPT,
    CONSTRAINED_SYSTEM_PROMPT,
    build_constrained_user_prompt,
)
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.planner import Architect, ArchitectError
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.task import Task


@dataclass
class SubGraphTemplate:
    """A reusable DAG fragment the constrained planner can select.

    Attributes:
        name: Unique template identifier shown in the LLM menu.
        description: Human-readable description of what this template does.
        builder: Callable that accepts a Task and optional params dict,
                 returning a list of Nodes and a Registry of agents.
    """

    name: str
    description: str
    builder: Callable[..., tuple[list[Node], Registry]]


class ConstrainedArchitect(Architect):
    """LLM selects and composes from a fixed menu of sub-graph templates.

    The LLM cannot invent nodes — it can only choose from pre-validated
    templates.  This dramatically shrinks the failure space compared to
    the fully autonomous LLMArchitect.

    Composition rules:
    - Node IDs are prefixed with ``{template_name}-{instance_index}-``
      to prevent collisions when the same template is used multiple times.
    - Templates are composed sequentially: leaf nodes of template N become
      dependencies of root nodes in template N+1.
    - Agent registries are merged; agent ID uniqueness is guaranteed by
      the UUID-based Agent.id generation.
    - The composed graph is validated after assembly.
    """

    def __init__(
        self,
        provider: Provider,
        templates: list[SubGraphTemplate],
        model: str = "claude-mythos",
        max_retries: int = 2,
    ) -> None:
        self._provider = provider
        self._templates = {t.name: t for t in templates}
        self._template_list = templates
        self._model = model
        self._max_retries = max_retries

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        return asyncio.run(self.aplan(task))

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        menu = [
            {"name": t.name, "description": t.description}
            for t in self._template_list
        ]
        user_prompt = build_constrained_user_prompt(task, menu)

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt == 0:
                prompt = user_prompt
            else:
                prompt = (
                    user_prompt
                    + "\n\n---\n\n"
                    + f"Your previous response could not be parsed: {last_error}\n\n"
                    + CONSTRAINED_RETRY_PROMPT
                )

            result = await self._provider.complete(
                CONSTRAINED_SYSTEM_PROMPT, prompt, model=self._model
            )

            try:
                selections = self._extract_selections(result.text)
                return self._compose(selections, task)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                continue

        raise ArchitectError(
            f"ConstrainedArchitect failed after {1 + self._max_retries} attempts: "
            f"{last_error}"
        )

    def _extract_selections(self, text: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response."""
        stripped = text.strip()
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL
        )
        if fence_match:
            stripped = fence_match.group(1).strip()

        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of template selections")
        return data

    def _compose(
        self,
        selections: list[dict[str, Any]],
        task: Task,
    ) -> tuple[ExecutionGraph, Registry]:
        """Build a single graph from ordered template selections.

        Each template's nodes are prefixed to avoid ID collisions.
        Sequential templates are wired so the later template's roots
        depend on the earlier template's leaf nodes.
        """
        if not selections:
            raise ValueError("No templates selected")

        all_nodes: list[Node] = []
        merged_registry = Registry()
        prev_leaf_ids: list[str] = []

        for idx, sel in enumerate(selections):
            template_name = sel.get("template", "")
            if template_name not in self._templates:
                valid = list(self._templates.keys())
                raise ValueError(
                    f"Unknown template {template_name!r}. "
                    f"Valid templates: {valid}"
                )

            template = self._templates[template_name]
            params = sel.get("params", {})

            nodes, registry = template.builder(task, **params)
            # Defensively clone template nodes so composition never mutates
            # reusable template internals across planner calls.
            nodes = [deepcopy(node) for node in nodes]

            prefix = f"{template_name}-{idx}"
            id_map: dict[str, str] = {}
            for node in nodes:
                old_id = node.id
                new_id = f"{prefix}-{old_id}"
                id_map[old_id] = new_id
                node.id = new_id

            for node in nodes:
                node.depends_on = [id_map.get(d, d) for d in node.depends_on]

            roots = [n for n in nodes if not n.depends_on]
            if prev_leaf_ids:
                for root in roots:
                    root.depends_on = list(prev_leaf_ids)

            dep_set = set()
            for n in nodes:
                dep_set.update(n.depends_on)
            leaf_ids = [n.id for n in nodes if n.id not in dep_set]
            prev_leaf_ids = leaf_ids

            all_nodes.extend(nodes)

            for agent in registry.list_agents():
                merged_registry.register(agent)

        graph = ExecutionGraph(
            topology=[Topology.SERIAL],
            nodes=all_nodes,
        )
        graph.validate()
        return graph, merged_registry
