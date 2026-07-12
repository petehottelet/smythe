"""Executor — walks the DAG and runs each node via an LLM provider."""

from __future__ import annotations

import asyncio
import time

from smythe.budget import BudgetEstimateRequired, SentinelAlert
from smythe.executor_base import ExecutorBase, NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus


class Executor(ExecutorBase):
    """Executes an assigned graph by walking it in dependency order.

    Runs nodes serially via a Provider.  For concurrent execution
    of independent nodes, use AsyncExecutor instead.
    """

    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node in the graph, respecting dependency order."""
        self.prepare_graph(graph)
        first_error: Exception | None = None
        for node in self._walk(graph):
            try:
                self._execute_node(node, graph)
            except (BudgetEstimateRequired, NodeFinalizationError, SentinelAlert):
                # Budget admission/reconciliation failures are global safety
                # stops, as is a post-billing finalization failure. Never let
                # a later independent node start after one.
                raise
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
                if dep_id not in lookup:
                    raise ValueError(
                        f"Node {node.id!r} depends on unknown node {dep_id!r}"
                    )
                visit(lookup[dep_id])
            order.append(node)

        for node in graph.nodes:
            visit(node)
        return order

    def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider, respecting its failure policy."""
        if node.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
            return  # already done — happens when resuming from a checkpoint

        if not self.deps_satisfied(node, graph):
            if node.failure_policy == FailurePolicy.SKIP:
                node.status = NodeStatus.SKIPPED
                return
            node.status = NodeStatus.FAILED
            node.result = "Upstream dependency failed"
            raise RuntimeError(f"Node {node.id!r}: upstream dependency not satisfied")

        if self._budget:
            self.reserve_node_budget(node)

        last_exc: Exception | None = None
        attempts = 1 + max(node.max_retries, 0) if node.failure_policy == FailurePolicy.RETRY else 1

        for attempt in range(attempts):
            delay = self.retry_delay_s(attempt)
            if delay:
                time.sleep(delay)
            node.status = NodeStatus.RUNNING
            self._tracer.on_node_start(node)

            try:
                # Cost recording happens inside acall_node (per provider call).
                result = asyncio.run(self.acall_node(node, graph))
                try:
                    self.finalize_node_result(node, result)
                except Exception as exc:
                    raise NodeFinalizationError(node.id, exc) from exc
                node.status = NodeStatus.COMPLETED
                self._tracer.on_node_end(node)
                self.notify_update(node)
                return
            except (NodeFinalizationError, SentinelAlert) as exc:
                self._tracer.on_node_error(node, exc)
                self._tracer.on_node_end(node)
                node.status = NodeStatus.FAILED
                node.result = str(exc)
                self.notify_update(node)
                raise
            except Exception as exc:
                last_exc = exc
                self._tracer.on_node_error(node, exc)
                self._tracer.on_node_end(node)

        if self._budget:
            self._budget.release(node.id)

        node.status = NodeStatus.FAILED
        node.result = str(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            self.notify_update(node)
            return

        self.notify_update(node)
        raise last_exc  # type: ignore[misc]
