"""ExecutorBase — shared logic for serial and async executors."""

from __future__ import annotations

import asyncio
from typing import Any

from smythe.agent import Agent
from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class ExecutorBase:
    """Shared infrastructure for all executor variants.

    Provides constructor, prompt building, dependency lookup, and
    node-by-id helpers.  Subclasses implement ``run()`` and the
    provider call (sync or async).
    """

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._budget = budget

    @staticmethod
    def build_system_prompt(agent: Agent | None) -> str:
        if agent and agent.profile.persona:
            return agent.profile.persona
        return "You are a helpful assistant completing a step in a larger task."

    @staticmethod
    def build_user_prompt(node: Node, dep_results: dict[str, Any]) -> str:
        parts = [node.label]
        if dep_results:
            parts.append("\n\nContext from prior steps:")
            for dep_id, result in dep_results.items():
                parts.append(f"\n[{dep_id}]: {result}")
        return "\n".join(parts)

    @staticmethod
    def node_by_id(node_id: str, graph: ExecutionGraph) -> Node | None:
        return next((n for n in graph.nodes if n.id == node_id), None)

    def deps_satisfied(self, node: Node, graph: ExecutionGraph) -> bool:
        """True if every dependency is COMPLETED or SKIPPED."""
        for dep_id in node.depends_on:
            dep = self.node_by_id(dep_id, graph)
            if dep is None or dep.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
                return False
        return True

    async def acall_node(self, node: Node, graph: ExecutionGraph) -> CompletionResult:
        """Build the node's prompts and call the provider, enforcing timeout_s."""
        agent = self._registry.get(node.agent_id) if node.agent_id else None
        dep_results = self.gather_dep_results(node, graph)
        system = self.build_system_prompt(agent)
        prompt = self.build_user_prompt(node, dep_results)
        coro = self._provider.complete(system, prompt, model=node.metadata.get("model", ""))
        if node.timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=node.timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"Node {node.id!r} timed out after {node.timeout_s}s"
            ) from None

    def gather_dep_results(self, node: Node, graph: ExecutionGraph) -> dict[str, Any]:
        return {
            dep_id: dep_node.result
            for dep_id in node.depends_on
            if (dep_node := self.node_by_id(dep_id, graph)) is not None
        }
