"""Architect — generates an execution graph from a task description."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod

from smythe.budget import Sentinel, validate_completion_usage, validate_token_count
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.loader import (
    MODEL_PLAN_MAX_DEPTH,
    MODEL_PLAN_MAX_NODES,
    build_graph_from_model_output,
)
from smythe.prompts import (
    PLANNING_SYSTEM_PROMPT,
    RETRY_PROMPT,
    build_agent_inventory,
    build_user_prompt,
)
from smythe.provider import TRUNCATED_STOP_REASONS, Provider
from smythe.registry import Registry
from smythe.task import Task
from smythe.workflow_binding import (
    ComponentBinding, WorkflowBindingError, describe_component, provider_description, require_exact,
)


class ArchitectError(Exception):
    """Raised when the Architect cannot produce a valid execution graph."""


def _strip_code_fence(text: str) -> str:
    """Return the body of the first ``` fence (minus a ``json`` tag), else ``text``.

    Uses plain searches: a regex with whitespace quantifiers around a lazy
    body backtracks cubically on a reply with an unclosed fence followed by
    a long run of whitespace, stalling the event loop for minutes.
    """
    start = text.find("```")
    end = text.find("```", start + 3) if start >= 0 else -1
    if end < 0:
        return text
    body = text[start + 3:end]
    return body.removeprefix("json").strip()


class Architect(ABC):
    """Base class for all architects."""

    @abstractmethod
    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Generate an execution graph and accompanying registry for a task."""

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Async variant — defaults to delegating to the sync plan().

        Subclasses that perform async I/O (e.g. LLM calls) should
        override this and make plan() the thin wrapper instead.
        """
        return self.plan(task)


class DeterministicArchitect(Architect):
    """Base class for architects that build DAGs with pure Python logic.

    Subclass this when you know the graph shape ahead of time and want
    zero LLM cost, zero latency, and 100% deterministic output.
    Users override ``plan()`` with programmatic node construction.
    """

    @abstractmethod
    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Build a graph from pure Python — no LLM calls."""


class SimpleArchitect(DeterministicArchitect):
    """Produces a single-node serial graph.  Useful as a fallback
    or for tasks that don't need LLM-driven decomposition.
    """

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, SimpleArchitect)
        return {"type": "simple_architect", "version": 1}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return ()

    def bind_run(self, binding: ComponentBinding) -> SimpleArchitect:
        self.workflow_description()
        return SimpleArchitect()

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        node = Node(label=task.goal)
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        graph.validate()
        return graph, Registry()


class LLMArchitect(Architect):
    """Decomposes tasks into multi-node DAGs via an LLM call.

    The Architect sends the task to the LLM with a structured prompt
    describing available topologies and the expected JSON output schema.
    The response is parsed into an ExecutionGraph with agent personas.

    The reply is model output, so it is parsed under a strict schema
    (:func:`smythe.loader.build_graph_from_model_output`): a plan cannot
    declare MCP servers or per-node models, and it must stay within
    ``max_nodes`` nodes and ``max_depth`` levels.  A reply that breaks
    the schema is retried like malformed JSON, up to ``max_retries``.

    In a durable run, each plan must also pass the run's own graph checks,
    including its ``WorkflowGraphPolicy``, before planning is saved; a
    plan they reject is retried the same way.
    """

    def __init__(
        self,
        provider: Provider,
        planning_model: str = "claude-opus-5-5",
        memory: object | None = None,
        max_retries: int = 2,
        cost_per_token: float = 0.000003,
        avg_tokens_per_node: int = 2000,
        registry: Registry | None = None,
        *,
        planning_instructions: str = "",
        max_nodes: int = MODEL_PLAN_MAX_NODES,
        max_depth: int = MODEL_PLAN_MAX_DEPTH,
        run_binding: ComponentBinding | None = None,
    ) -> None:
        self._provider = provider
        self._planning_model = planning_model
        self._memory = memory
        self._max_retries = max_retries
        self._cost_per_token = cost_per_token
        self._avg_tokens_per_node = avg_tokens_per_node
        # When set, the planning prompt includes an inventory of these
        # agents (and their tools) so plans can be designed around them.
        self._registry = registry
        if type(planning_instructions) is not str:
            raise ValueError("planning_instructions must be a string")
        self._planning_instructions = planning_instructions
        for name, value in (("max_nodes", max_nodes), ("max_depth", max_depth)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._run_binding = run_binding

    def _plan_limits(self) -> dict:
        """Non-default plan limits; defaults are omitted so existing recipes keep their identity."""
        limits = {}
        if self._max_nodes != MODEL_PLAN_MAX_NODES:
            limits["max_nodes"] = self._max_nodes
        if self._max_depth != MODEL_PLAN_MAX_DEPTH:
            limits["max_depth"] = self._max_depth
        return limits

    def workflow_description(self, **defaults) -> dict:
        require_exact(self, LLMArchitect)
        if self._memory is not None:
            raise WorkflowBindingError("Journaled workflows do not support live planner memory")
        if type(self._planning_instructions) is not str:
            raise WorkflowBindingError("planning_instructions must be a string")
        validate_token_count(self._max_retries, "max_retries")
        validate_token_count(self._avg_tokens_per_node, "avg_tokens_per_node")
        Sentinel(cost_per_token=self._cost_per_token)
        return {"type": "llm_architect", "version": 1,
                **provider_description(self._provider, self._planning_model),
                "max_retries": self._max_retries, "cost_per_token": self._cost_per_token,
                "avg_tokens_per_node": self._avg_tokens_per_node,
                **({"planning_instructions": self._planning_instructions}
                   if self._planning_instructions else {}),
                **self._plan_limits(),
                "registry": describe_component(self._registry, role="registry")}

    def workflow_providers(self) -> tuple[Provider, ...]:
        return (self._provider,)

    def bind_run(self, binding: ComponentBinding) -> LLMArchitect:
        self.workflow_description()
        return LLMArchitect(
            binding.snapshot_provider(self._provider), self._planning_model,
            max_retries=self._max_retries, cost_per_token=self._cost_per_token,
            avg_tokens_per_node=self._avg_tokens_per_node,
            registry=self._registry.bind_run(binding.child("registry")) if self._registry else None,
            planning_instructions=self._planning_instructions,
            max_nodes=self._max_nodes, max_depth=self._max_depth,
            run_binding=binding,
        )

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Sync wrapper — safe to call outside an event loop."""
        return asyncio.run(self.aplan(task))

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Core planning loop — awaits provider calls directly."""
        history = self._get_history(task)
        inventory = build_agent_inventory(self._registry)
        user_prompt = build_user_prompt(task, history, agent_inventory=inventory)
        if self._planning_instructions:
            user_prompt += ("\n\n## Execution graph policy\n"
                            "These instructions govern the plan, not the task's answer. "
                            "Do not add them to the deliverable schema.\n"
                            + self._planning_instructions)
        if self._plan_limits():
            # The system prompt states the default limits; say which apply.
            user_prompt += ("\n\n## Plan limits\n"
                            f"Use at most {self._max_nodes} nodes and keep the graph at most "
                            f"{self._max_depth} levels deep. These limits replace the "
                            "defaults stated in the rules.")

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt == 0:
                prompt = user_prompt
            else:
                prompt = (
                    user_prompt
                    + "\n\n---\n\n"
                    + f"Your previous response was rejected: {last_error}\n\n"
                    + RETRY_PROMPT
                )
            provider = (self._run_binding.for_call(self._provider, trigger="task", attempt=attempt)
                        if self._run_binding else self._provider)
            result = await provider.complete(
                PLANNING_SYSTEM_PROMPT, prompt, model=self._planning_model
            )
            validate_completion_usage(result)
            try:
                if result.stop_reason in TRUNCATED_STOP_REASONS:
                    # A cut-off plan can parse yet silently miss nodes.
                    raise ValueError(
                        "the response was cut off at the output token limit; "
                        "return a smaller plan"
                    )
                data = self._extract_json(result.text)
                graph, registry = build_graph_from_model_output(
                    data, max_nodes=self._max_nodes, max_depth=self._max_depth,
                )
                graph.estimated_cost_usd = self._estimate_cost(graph)
                if self._run_binding is not None and self._run_binding.plan_check is not None:
                    self._run_binding.plan_check(graph, registry)
                return graph, registry
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                continue

        raise ArchitectError(
            f"Failed to produce a valid plan after {1 + self._max_retries} attempts: "
            f"{last_error}"
        )

    def _get_history(self, task: Task) -> list[dict] | None:
        if self._memory is None:
            return None
        from smythe.memory import PlannerMemory as _PlannerMemory

        if not isinstance(self._memory, _PlannerMemory):
            return None
        outcomes = self._memory.recall(task)
        if not outcomes:
            return None
        return [
            {
                "task_goal": o.task_goal,
                "topology": o.topology,
                "total_cost_usd": o.total_cost_usd,
                "total_duration_ms": o.total_duration_ms,
                "success": o.success,
            }
            for o in outcomes
        ]

    def _estimate_cost(self, graph: ExecutionGraph) -> float:
        return len(graph.nodes) * self._avg_tokens_per_node * self._cost_per_token

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse JSON from an LLM response, stripping code fences if present."""
        stripped = _strip_code_fence(text.strip())

        try:
            data = json.loads(stripped)
        except RecursionError:
            raise ValueError("The plan JSON is nested too deeply") from None
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the top level")
        return data
