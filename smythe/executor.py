"""Executor — walks the DAG and runs each node via an LLM provider."""

from __future__ import annotations

import asyncio
import time

from smythe.budget import BudgetValidationError, SentinelAlert
from smythe.executor_base import ExecutorBase, NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus
from smythe.provider import ProviderResponseError
from smythe.verifier import VerificationRecoveryError
from smythe.workflow_store import WorkflowError


class Executor(ExecutorBase):
    """Executes an assigned graph by walking it in dependency order.

    Runs nodes serially via a Provider.  For concurrent execution
    of independent nodes, use AsyncExecutor instead.
    """

    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute in dependency order; stop immediately on a terminal failure."""
        self.prepare_execution(graph)
        self.recover_verification(graph)
        visited: set[str] = set()
        # The walk is recomputed each step rather than taken once, so a
        # supervisor revision that adds or drops pending work is picked
        # up on the next iteration.
        while True:
            remaining = [
                n for n in self._walk(graph)
                if n.id not in visited or n.status is NodeStatus.PENDING
            ]
            if not remaining:
                break
            node = remaining[0]
            visited.add(node.id)
            try:
                # Node execution consumes SKIP and permitted retries. Any
                # exception that remains stops admission of all queued work.
                self._execute_node(node, graph)
            except BudgetValidationError as exc:
                self.mark_accounting_invalid(node, exc)
                self.notify_update(node)
                raise
            self.maybe_regenerate(node, graph)
            if self._supervisor is not None:
                asyncio.run(self.maybe_revise(node, graph))
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
            self._call_attempts[node.id] = attempt
            delay = self.retry_delay_s(attempt)
            if delay:
                time.sleep(delay)
            node.status = NodeStatus.RUNNING
            self.begin_verification(node, graph)
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
                self.complete_verification(node)
                self.notify_update(node)
                return
            except VerificationRecoveryError:
                # Keep the paid verdict and pending receipt intact. A local
                # control-write failure must never cause another provider call.
                raise
            except (BudgetValidationError, NodeFinalizationError, SentinelAlert,
                    ProviderResponseError, WorkflowError) as exc:
                # Invalid post-call accounting is terminal. Retain any held
                # reservation: an unusable bill is not evidence of a free call.
                if isinstance(exc, BudgetValidationError):
                    self.mark_accounting_invalid(node, exc)
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

        self.release_node_budget(node)

        node.status = NodeStatus.FAILED
        node.result = str(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            self.notify_update(node)
            return

        self.notify_update(node)
        raise last_exc  # type: ignore[misc]
