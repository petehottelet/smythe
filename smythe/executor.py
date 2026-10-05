"""Executor — walks the DAG and runs each node via an LLM provider."""

from __future__ import annotations

import asyncio
import inspect
import time

from smythe._failures import is_terminal
from smythe._redact import error_text
from smythe.budget import BudgetValidationError
from smythe.executor_base import ExecutorBase, NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, _dependency_order
from smythe.scheduling import RetryDeadlineError
from smythe.verifier import VerificationRecoveryError


class Executor(ExecutorBase):
    """Executes an assigned graph by walking it in dependency order.

    Runs nodes serially via a Provider.  For concurrent execution
    of independent nodes, use AsyncExecutor instead.
    """

    def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute in dependency order; stop immediately on a terminal failure."""
        self.prepare_execution(graph)
        with self._tracer.phase("verification"):
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
            with self._tracer.phase("verification"):
                self.maybe_regenerate(node, graph)
            if self._supervisor is not None:
                # Any pending node is held until the review finishes.
                held = any(n.status is NodeStatus.PENDING for n in graph.nodes)
                started = self._tracer.now()
                asyncio.run(self.maybe_revise(node, graph))
                if held:
                    self._tracer.on_wait("supervision", self._tracer.since(started))
        return graph

    def _walk(self, graph: ExecutionGraph) -> list[Node]:
        """Topological sort of nodes by dependency order."""
        return _dependency_order(graph.nodes, strict_missing=True)

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
            self._reserve_node(node)

        last_exc: Exception | None = None
        attempts = 1 + max(node.max_retries, 0) if node.failure_policy == FailurePolicy.RETRY else 1

        for attempt in range(attempts):
            self._call_attempts[node.id] = attempt
            try:
                delay = self._retry_wait_s(node, attempt, last_exc)
            except RetryDeadlineError as exc:
                last_exc = exc
                break
            if delay:
                started = self._tracer.now()
                waited = (self._sleep or time.sleep)(delay)
                if inspect.isawaitable(waited):
                    asyncio.run(asyncio.wait_for(waited, None))
                self._tracer.on_wait("retry_backoff", self._tracer.since(started), node)
            node.status = NodeStatus.RUNNING
            self.begin_verification(node, graph)
            self._tracer.on_node_start(node, attempt=attempt)

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
            except Exception as exc:
                if is_terminal(exc):
                    # Invalid post-call accounting is terminal. Retain any held
                    # reservation: an unusable bill is not evidence of a free call.
                    if isinstance(exc, BudgetValidationError):
                        self.mark_accounting_invalid(node, exc)
                    node.status = NodeStatus.FAILED
                    node.result = error_text(exc)
                    self._tracer.on_node_error(node, exc)
                    self._tracer.on_node_end(node)
                    self.notify_update(node)
                    raise
                last_exc = exc
                self._tracer.on_node_error(node, exc)
                self._end_failed_attempt(node)
                if not self._may_retry(exc):
                    break

        self.release_node_budget(node)

        node.status = NodeStatus.FAILED
        node.result = error_text(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            self.notify_update(node)
            return

        self.notify_update(node)
        raise last_exc  # type: ignore[misc]
