"""Planner — generates an execution graph from a task description."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod

from smythe.graph import ExecutionGraph, Node, Topology
from smythe.loader import build_graph_from_dict
from smythe.prompts import PLANNING_SYSTEM_PROMPT, RETRY_PROMPT, build_user_prompt
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.task import Task


class PlanningError(Exception):
    """Raised when the LLM planner cannot produce a valid execution graph."""


class Planner(ABC):
    """Base class for all planners."""

    @abstractmethod
    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Generate an execution graph and accompanying registry for a task."""

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Async variant — defaults to delegating to the sync plan().

        Subclasses that perform async I/O (e.g. LLM calls) should
        override this and make plan() the thin wrapper instead.
        """
        return self.plan(task)


class DeterministicPlanner(Planner):
    """Base class for planners that build DAGs with pure Python logic.

    Subclass this when you know the graph shape ahead of time and want
    zero LLM cost, zero latency, and 100% deterministic output.
    Users override ``plan()`` with programmatic node construction.
    """

    @abstractmethod
    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Build a graph from pure Python — no LLM calls."""


class SimplePlanner(DeterministicPlanner):
    """Produces a single-node serial graph.  Useful as a fallback
    or for tasks that don't need LLM-driven decomposition.
    """

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        node = Node(label=task.goal)
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        graph.validate()
        return graph, Registry()


class LLMPlanner(Planner):
    """Decomposes tasks into multi-node DAGs via an LLM call.

    The planner sends the task to the LLM with a structured prompt
    describing available topologies and the expected JSON output schema.
    The response is parsed into an ExecutionGraph with agent personas.
    """

    def __init__(
        self,
        provider: Provider,
        planning_model: str = "claude-mythos",
        memory: object | None = None,
        max_retries: int = 2,
        cost_per_token: float = 0.000003,
        avg_tokens_per_node: int = 2000,
    ) -> None:
        self._provider = provider
        self._planning_model = planning_model
        self._memory = memory
        self._max_retries = max_retries
        self._cost_per_token = cost_per_token
        self._avg_tokens_per_node = avg_tokens_per_node

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Sync wrapper — safe to call outside an event loop."""
        return asyncio.run(self.aplan(task))

    async def aplan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        """Core planning loop — awaits provider calls directly."""
        history = self._get_history(task)
        user_prompt = build_user_prompt(task, history)

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            if attempt == 0:
                prompt = user_prompt
            else:
                prompt = (
                    user_prompt
                    + "\n\n---\n\n"
                    + f"Your previous response could not be parsed: {last_error}\n\n"
                    + RETRY_PROMPT
                )
            result = await self._provider.complete(
                PLANNING_SYSTEM_PROMPT, prompt, model=self._planning_model
            )
            try:
                data = self._extract_json(result.text)
                graph, registry = build_graph_from_dict(data)
                graph.estimated_cost_usd = self._estimate_cost(graph)
                return graph, registry
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                continue

        raise PlanningError(
            f"Failed to produce a valid plan after {1 + self._max_retries} attempts: "
            f"{last_error}"
        )

    def _get_history(self, task: Task) -> list[dict] | None:
        if self._memory is None:
            return None
        from smythe.memory import PlannerMemory

        if not isinstance(self._memory, PlannerMemory):
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
        stripped = text.strip()
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL
        )
        if fence_match:
            stripped = fence_match.group(1).strip()

        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the top level")
        return data
