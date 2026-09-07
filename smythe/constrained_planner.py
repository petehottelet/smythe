"""ConstrainedArchitect — LLM selects from a menu of pre-built sub-graph templates."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from smythe.budget import validate_completion_usage, validate_token_count
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
from smythe.workflow_binding import (
    ComponentBinding, WorkflowBindingError, bind_component,
    describe_component, provider_description, require_exact,
)


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
        model: str = "claude-opus-4-8",
        max_retries: int = 2,
        *,
        run_binding: ComponentBinding | None = None,
    ) -> None:
        self._provider = provider
        self._templates = {t.name: t for t in templates}
        self._template_list = templates
        self._model = model
        self._max_retries = max_retries
        self._run_binding = run_binding

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, ConstrainedArchitect)
        validate_token_count(self._max_retries, "max_retries")
        templates = []
        seen = set()
        for template in self._template_list:
            if (type(template) is not SubGraphTemplate or type(template.name) is not str
                    or not template.name or type(template.description) is not str
                    or template.name in seen):
                raise WorkflowBindingError("Templates require unique plain SubGraphTemplate descriptors")
            seen.add(template.name)
            description = describe_component(template.builder, role="template_builder")
            if description.get("type") != "local_only" or description.get("role") != "template_builder":
                raise WorkflowBindingError("Template builders require LocalOnly(role='template_builder')")
            templates.append({"name": template.name, "description": template.description,
                              "builder": description})
        return {"type": "constrained_architect", "version": 1,
                **provider_description(self._provider, self._model),
                "max_retries": self._max_retries, "templates": templates}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return (self._provider,)

    def bind_run(self, binding: ComponentBinding) -> ConstrainedArchitect:
        self.workflow_description()
        return ConstrainedArchitect(
            binding.snapshot_provider(self._provider),
            [SubGraphTemplate(t.name, t.description, bind_component(
                t.builder, binding.child(f"template:{t.name}"),
            )) for t in self._template_list], self._model, self._max_retries, run_binding=binding,
        )

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

            provider = (self._run_binding.for_call(self._provider, trigger="task", attempt=attempt)
                        if self._run_binding else self._provider)
            result = await provider.complete(
                CONSTRAINED_SYSTEM_PROMPT, prompt, model=self._model
            )
            validate_completion_usage(result)

            try:
                selections = self._extract_selections(result.text)
                return self._compose(selections, task)
            except WorkflowBindingError:
                raise
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
            if self._run_binding is not None:
                registry = bind_component(registry, self._run_binding.child(f"result:{idx}"))
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
