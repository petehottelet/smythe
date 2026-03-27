"""AsyncExecutor — concurrent DAG execution via asyncio."""

from __future__ import annotations

import asyncio

from smythe.agent import Agent
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class AsyncExecutor:
    """Executes an assigned graph with maximum concurrency.

    Independent nodes (those whose dependencies are all completed)
    are launched in parallel via asyncio.gather().  The executor
    runs in waves until every node has completed or failed.
    """

    def __init__(self, provider: Provider, registry: Registry, tracer: Tracer) -> None:
        self._provider = provider
        self._registry = registry
        self._tracer = tracer

    async def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node, fanning out independent nodes concurrently."""
        while True:
            pending = [n for n in graph.nodes if n.status == NodeStatus.PENDING]
            if not pending:
                break
            ready = [n for n in pending if graph.is_ready(n)]
            if not ready:
                raise RuntimeError("Deadlock: pending nodes exist but none are ready")
            await asyncio.gather(*(self._execute_node(n, graph) for n in ready))
        return graph

    async def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider."""
        node.status = NodeStatus.RUNNING
        self._tracer.on_node_start(node)

        try:
            agent = self._registry.get(node.agent_id) if node.agent_id else None
            dep_results = {
                dep_id: dep_node.result
                for dep_id in node.depends_on
                if (dep_node := self._node_by_id(dep_id, graph)) is not None
            }
            system = self._build_system_prompt(agent)
            prompt = self._build_user_prompt(node, dep_results)
            node.result = await self._provider.complete(
                system, prompt, model=node.metadata.get("model", "")
            )
            node.status = NodeStatus.COMPLETED
        except Exception as exc:
            node.status = NodeStatus.FAILED
            node.result = str(exc)
            self._tracer.on_node_error(node, exc)
            raise
        finally:
            self._tracer.on_node_end(node)

    @staticmethod
    def _build_system_prompt(agent: Agent | None) -> str:
        if agent and agent.profile.persona:
            return agent.profile.persona
        return "You are a helpful assistant completing a step in a larger task."

    @staticmethod
    def _build_user_prompt(node: Node, dep_results: dict[str, str]) -> str:
        parts = [node.label]
        if dep_results:
            parts.append("\n\nContext from prior steps:")
            for dep_id, result in dep_results.items():
                parts.append(f"\n[{dep_id}]: {result}")
        return "\n".join(parts)

    @staticmethod
    def _node_by_id(node_id: str, graph: ExecutionGraph) -> Node | None:
        return next((n for n in graph.nodes if n.id == node_id), None)
