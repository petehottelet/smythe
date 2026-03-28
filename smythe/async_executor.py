"""AsyncExecutor — concurrent DAG execution via asyncio."""

from __future__ import annotations

import asyncio

from smythe.budget import Sentinel
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class AsyncExecutor(ExecutorBase):
    """Executes an assigned graph with maximum concurrency.

    Independent nodes (those whose dependencies are all completed)
    are launched in parallel via asyncio.gather().  The executor
    runs in waves until every node has completed or failed.

    Budget safety: before each wave, estimated cost is *reserved* for
    every ready node so concurrent nodes cannot collectively overshoot
    the budget.  Reservations are reconciled with actual cost on
    completion, or released on failure.
    """

    DEFAULT_ESTIMATED_TOKENS = 2000

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
        estimated_tokens_per_node: int = DEFAULT_ESTIMATED_TOKENS,
    ) -> None:
        super().__init__(provider=provider, registry=registry, tracer=tracer, budget=budget)
        self._estimated_tokens_per_node = estimated_tokens_per_node

    async def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node, fanning out independent nodes concurrently."""
        first_error: Exception | None = None
        while True:
            pending = [n for n in graph.nodes if n.status == NodeStatus.PENDING]
            if not pending:
                break
            ready = [n for n in pending if graph.is_ready(n)]
            if not ready:
                if first_error is not None:
                    raise first_error
                failed = [n for n in graph.nodes if n.status == NodeStatus.FAILED]
                if failed:
                    raise RuntimeError(
                        f"Execution halted: {len(pending)} node(s) blocked by "
                        f"{len(failed)} failed upstream node(s)"
                    )
                raise RuntimeError("Deadlock: pending nodes exist but none are ready")

            if self._budget:
                estimated_cost = (
                    self._estimated_tokens_per_node * self._budget.cost_per_token
                )
                for n in ready:
                    self._budget.reserve(n.id, estimated_cost)

            try:
                await asyncio.gather(*(self._execute_node(n, graph) for n in ready))
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error
        return graph

    async def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider, respecting its failure policy."""
        last_exc: Exception | None = None
        attempts = 1 + max(node.max_retries, 0) if node.failure_policy == FailurePolicy.RETRY else 1

        for attempt in range(attempts):
            node.status = NodeStatus.RUNNING
            self._tracer.on_node_start(node)

            try:
                agent = self._registry.get(node.agent_id) if node.agent_id else None
                dep_results = self.gather_dep_results(node, graph)
                system = self.build_system_prompt(agent)
                prompt = self.build_user_prompt(node, dep_results)
                result = await self._provider.complete(
                    system, prompt, model=node.metadata.get("model", "")
                )
                node.result = result.text

                if self._budget:
                    cost = self._budget.record(node.id, result)
                    node.metadata["cost_usd"] = cost

                node.status = NodeStatus.COMPLETED
                self._tracer.on_node_end(node)
                return
            except Exception as exc:
                last_exc = exc
                self._tracer.on_node_error(node, exc)
                self._tracer.on_node_end(node)

        if self._budget:
            self._budget.release(node.id)

        node.result = str(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            return

        node.status = NodeStatus.FAILED
        raise last_exc  # type: ignore[misc]
