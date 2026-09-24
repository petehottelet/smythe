"""ConstrainedArchitect — LLM selects from a menu of pre-built sub-graph templates."""

from __future__ import annotations

import asyncio
import json
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
from smythe.planner import Architect, ArchitectError, _strip_code_fence
from smythe.provider import TRUNCATED_STOP_REASONS, Provider
from smythe.registry import Registry
from smythe.task import Task
from smythe.workflow_binding import (
    ComponentBinding, WorkflowBindingError, bind_component,
    describe_component, provider_description, require_exact,
)

# The model's selections and params decide the composed graph's size. The
# default admits eight templates the size of a default generated plan.
DEFAULT_MAX_NODES = 64


@dataclass
class SubGraphTemplate:
    """A reusable DAG fragment the constrained planner can select.

    Attributes:
        name: Unique template identifier shown in the LLM menu.
        description: Human-readable description of what this template does.
        builder: Callable invoked as ``builder(task, **params)``, where
                 ``params`` is the optional object the model supplied with
                 its selection, returning a list of Nodes and a Registry of
                 agents.  Params are model output: validate and bound them.
                 The architect's ``max_nodes`` cap is checked when the
                 builder returns, so it cannot limit what one call allocates.
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
    - The composed graph may have at most ``max_nodes`` nodes, counted
      after each builder call.  A selection over the cap is retried like a
      malformed one.  In a durable run, the composed plan must also pass
      the run's own graph checks and is retried the same way.
    """

    def __init__(
        self,
        provider: Provider,
        templates: list[SubGraphTemplate],
        model: str = "claude-opus-5-5",
        max_retries: int = 2,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        run_binding: ComponentBinding | None = None,
    ) -> None:
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError(f"max_nodes must be a positive integer, got {max_nodes!r}")
        self._provider = provider
        self._templates = {t.name: t for t in templates}
        self._template_list = templates
        self._model = model
        self._max_retries = max_retries
        self._max_nodes = max_nodes
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
                "max_retries": self._max_retries, "templates": templates,
                # Omitted at the default so existing recipes keep their identity.
                **({"max_nodes": self._max_nodes} if self._max_nodes != DEFAULT_MAX_NODES else {})}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return (self._provider,)

    def bind_run(self, binding: ComponentBinding) -> ConstrainedArchitect:
        self.workflow_description()
        return ConstrainedArchitect(
            binding.snapshot_provider(self._provider),
            [SubGraphTemplate(t.name, t.description, bind_component(
                t.builder, binding.child(f"template:{t.name}"),
            )) for t in self._template_list], self._model, self._max_retries,
            max_nodes=self._max_nodes, run_binding=binding,
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
        # A durable run journals each retry prompt, so a parse error keeps
        # 0.8.1's wording and a saved retry still replays after an upgrade.
        problem = "could not be parsed"
        for attempt in range(1 + self._max_retries):
            if attempt == 0:
                prompt = user_prompt
            else:
                prompt = (
                    user_prompt
                    + "\n\n---\n\n"
                    + f"Your previous response {problem}: {last_error}\n\n"
                    + CONSTRAINED_RETRY_PROMPT
                )

            provider = (self._run_binding.for_call(self._provider, trigger="task", attempt=attempt)
                        if self._run_binding else self._provider)
            result = await provider.complete(
                CONSTRAINED_SYSTEM_PROMPT, prompt, model=self._model
            )
            validate_completion_usage(result)

            try:
                if result.stop_reason in TRUNCATED_STOP_REASONS:
                    raise ValueError(
                        "the response was cut off at the output token limit; "
                        "return fewer selections"
                    )
                selections = self._extract_selections(result.text)
                graph, registry = self._compose(selections, task)
            except WorkflowBindingError:
                raise
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                last_error, problem = exc, "could not be parsed"
                continue
            # A template registry the run cannot bind is terminal (above); a
            # plan the durable run rejects is the model's choice, so repair it.
            if self._run_binding is not None and self._run_binding.plan_check is not None:
                try:
                    self._run_binding.plan_check(graph, registry)
                except (ValueError, KeyError, TypeError) as exc:
                    last_error, problem = exc, "was rejected"
                    continue
            return graph, registry

        raise ArchitectError(
            f"ConstrainedArchitect failed after {1 + self._max_retries} attempts: "
            f"{last_error}"
        )

    def _extract_selections(self, text: str) -> list[dict[str, Any]]:
        """Parse the JSON array of selections, raising ValueError on any schema error."""
        stripped = _strip_code_fence(text.strip())

        try:
            data = json.loads(stripped)
        except RecursionError:
            raise ValueError("The selection JSON is nested too deeply") from None
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of template selections")
        for index, selection in enumerate(data):
            if not isinstance(selection, dict):
                raise ValueError(
                    f"Selection at index {index} must be an object with a "
                    f"'template' key, got {type(selection).__name__}"
                )
            unknown = sorted(key for key in selection if key not in ("template", "params"))
            if unknown:
                raise ValueError(
                    f"Selection at index {index} has unsupported field(s) {unknown}; "
                    "allowed fields are ['params', 'template']"
                )
            if not isinstance(selection.get("template"), str):
                raise ValueError(f"Selection at index {index} needs a string 'template'")
            params = selection.get("params")
            if params is not None and not isinstance(params, dict):
                raise ValueError(f"'params' in selection at index {index} must be an object")
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
            params = sel.get("params") or {}

            nodes, registry = template.builder(task, **params)
            # Stop an oversized result before it is cloned and composed.
            nodes = list(nodes)
            total = len(all_nodes) + len(nodes)
            if total > self._max_nodes:
                raise ValueError(
                    f"Selection at index {idx} brings the graph to {total} nodes; "
                    f"the limit is {self._max_nodes}"
                )
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
                # A gate must judge the renamed node, or its verdict is discarded.
                if node.verifies is not None:
                    node.verifies = id_map.get(node.verifies, node.verifies)

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
