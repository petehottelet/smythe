"""Executor — walks the DAG and runs each node via an LLM provider."""

from __future__ import annotations

import asyncio

from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus


class Executor(ExecutorBase):
    """Executes an assigned graph by walking it in dependency order.

    Runs nodes serially via a Provider.  For concurrent execution
    of independent nodes, use AsyncExecutor instead.
    """

    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node in the graph, respecting dependency order."""
        first_error: Exception | None = None
        for node in self._walk(graph):
            try:
                self._execute_node(node, graph)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return graph

    def _walk(self, graph: ExecutionGraph) -> list[Node]:
        """Topological sort of nodes by dependency order."""
        visited: set[str] = set()
        order: list[Node] = []
        lookup = {n.id: n for n in graph.nodes}

        def visit(node: Node) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for dep_id in node.depends_on:
                visit(lookup[dep_id])
            order.append(node)

        for node in graph.nodes:
            visit(node)
        return order

    def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider, respecting its failure policy."""
        if not self.deps_satisfied(node, graph):
            if node.failure_policy == FailurePolicy.SKIP:
                node.status = NodeStatus.SKIPPED
                return
            node.status = NodeStatus.FAILED
            node.result = "Upstream dependency failed"
            raise RuntimeError(f"Node {node.id!r}: upstream dependency not satisfied")

        if self._budget:
            self._budget.check(node.id)

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
                result = asyncio.run(
                    self._provider.complete(system, prompt, model=node.metadata.get("model", ""))
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

        node.status = NodeStatus.FAILED
        node.result = str(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            return

        raise last_exc  # type: ignore[misc]
