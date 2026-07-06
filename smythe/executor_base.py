"""ExecutorBase — shared logic for serial and async executors."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from smythe.agent import Agent
from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tools import ChatMessage, ToolLoopLimitError, ToolResult, ToolRuntime
from smythe.tracer import Tracer

DEFAULT_MAX_TOOL_ITERATIONS = 10


class ExecutorBase:
    """Shared infrastructure for all executor variants.

    Provides constructor, prompt building, dependency lookup, the
    tool-calling loop, and node-by-id helpers.  Subclasses implement
    ``run()`` and drive ``acall_node()`` (sync or async).
    """

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
        on_node_update: Callable[[Node], None] | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._budget = budget
        self._on_node_update = on_node_update
        self._tool_runtime = tool_runtime
        self._max_tool_iterations = max_tool_iterations

    def notify_update(self, node: Node) -> None:
        """Invoke the node-update hook (used for checkpointing) if one is set.

        Called whenever a node reaches a terminal status: COMPLETED,
        SKIPPED, or FAILED.
        """
        if self._on_node_update is not None:
            self._on_node_update(node)

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
        """Run the node's conversation (including any tool loop) under timeout_s.

        Owns cost recording: every provider call inside is billed via
        Sentinel.add_cost, so executors must not record costs again.
        """
        coro = self._run_node_conversation(node, graph)
        if node.timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=node.timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"Node {node.id!r} timed out after {node.timeout_s}s"
            ) from None

    async def _run_node_conversation(
        self, node: Node, graph: ExecutionGraph,
    ) -> CompletionResult:
        agent = self._registry.get(node.agent_id) if node.agent_id else None
        dep_results = self.gather_dep_results(node, graph)
        system = self.build_system_prompt(agent)
        prompt = self.build_user_prompt(node, dep_results)
        model = node.metadata.get("model", "")
        messages = [ChatMessage(role="user", content=prompt)]

        if self._tool_runtime is None:
            result = await self._provider.chat(system, messages, model)
            self._record_cost(node, result)
            return result

        async with self._tool_runtime.open(agent) as session:
            tools = list(session.tools) or None
            limit = node.max_tool_iterations or self._max_tool_iterations
            for _ in range(limit):
                if self._budget:
                    self._budget.check(node.id)
                result = await self._provider.chat(system, messages, model, tools=tools)
                self._record_cost(node, result)

                if result.stop_reason == "pause_turn" and not result.tool_calls:
                    # Provider paused a server-side loop; re-send to continue.
                    messages.append(ChatMessage(role="assistant", content=result.text))
                    continue
                if not result.tool_calls:
                    return result

                messages.append(ChatMessage(
                    role="assistant",
                    content=result.text,
                    tool_calls=list(result.tool_calls),
                ))
                tool_results: list[ToolResult] = []
                for tc in result.tool_calls:
                    started = time.monotonic()
                    try:
                        outcome = await session.call(tc)
                    except Exception as exc:
                        # Tool failures go back to the model, not up the stack —
                        # it can adapt or try another tool.
                        outcome = ToolResult(
                            tool_call_id=tc.id, content=str(exc), is_error=True,
                        )
                    duration_ms = (time.monotonic() - started) * 1000
                    self._tracer.on_tool_call(node, tc.name, duration_ms, outcome.is_error)
                    tool_results.append(outcome)
                messages.append(ChatMessage(role="user", tool_results=tool_results))

        raise ToolLoopLimitError(
            f"Node {node.id!r} hit max_tool_iterations={limit} without completing"
        )

    def _record_cost(self, node: Node, result: CompletionResult) -> None:
        if self._budget:
            node.metadata["cost_usd"] = self._budget.add_cost(node.id, result)

    def gather_dep_results(self, node: Node, graph: ExecutionGraph) -> dict[str, Any]:
        return {
            dep_id: dep_node.result
            for dep_id in node.depends_on
            if (dep_node := self.node_by_id(dep_id, graph)) is not None
        }
