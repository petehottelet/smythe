"""AsyncExecutor — concurrent DAG execution via asyncio."""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Callable

from smythe.budget import Sentinel, SentinelAlert
from smythe.executor_base import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    ExecutorBase,
    NodeFinalizationError,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus
from smythe.provider import Provider
from smythe.registry import Registry
from smythe.tools import ToolRuntime
from smythe.tracer import Tracer


class AsyncExecutor(ExecutorBase):
    """Executes an assigned graph with maximum concurrency.

    Independent nodes (those whose dependencies are all completed)
    are launched as capacity becomes available.  Only the configured
    number of tasks is admitted to the event loop at once; a broadcast
    with thousands of nodes therefore does not create thousands of
    coroutines that merely wait behind a semaphore.

    Budget safety: estimated cost is reserved before each node is admitted,
    so concurrent admissions cannot collectively exceed the configured policy.
    Reservations are reconciled with reported cost or configured estimates on
    completion, or released on failure/cancellation.

    Concurrency safety: ``max_concurrency`` caps in-flight provider
    calls, so a wide broadcast doesn't fire every call
    at once and trip provider rate limits.  None means unlimited.
    """

    DEFAULT_ESTIMATED_TOKENS = 2000

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
        estimated_tokens_per_node: int = DEFAULT_ESTIMATED_TOKENS,
        max_concurrency: int | None = None,
        on_node_update: Callable[[Node], None] | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        artifact_dir: str | Path | None = "smythe_artifacts",
        retry_backoff_s: float = 0.0,
    ) -> None:
        super().__init__(
            provider=provider, registry=registry, tracer=tracer, budget=budget,
            on_node_update=on_node_update, tool_runtime=tool_runtime,
            max_tool_iterations=max_tool_iterations, artifact_dir=artifact_dir,
            retry_backoff_s=retry_backoff_s,
        )
        self._estimated_tokens_per_node = estimated_tokens_per_node
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._max_concurrency = max_concurrency

    async def run(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Execute every node, fanning out independent nodes concurrently."""
        self.prepare_graph(graph)
        pending = {
            node.id: node for node in graph.nodes if node.status == NodeStatus.PENDING
        }
        if not pending:
            return graph

        resolved = {
            node.id for node in graph.nodes
            if node.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
        }
        dependents: dict[str, list[Node]] = {node.id: [] for node in graph.nodes}
        unresolved: dict[str, int] = {}
        for node in graph.nodes:
            for dep_id in node.depends_on:
                dependents.setdefault(dep_id, []).append(node)
            if node.id in pending:
                unresolved[node.id] = sum(
                    dep_id not in resolved for dep_id in node.depends_on
                )

        ready = deque(
            node for node in graph.nodes
            if node.id in pending and unresolved[node.id] == 0
        )
        # None intentionally retains the documented unlimited mode.  With a
        # concrete cap, the active task set can never grow beyond that cap.
        concurrency = self._max_concurrency or max(1, len(pending))
        order = {node.id: i for i, node in enumerate(graph.nodes)}
        active: dict[asyncio.Task[None], Node] = {}

        try:
            while ready or active:
                admission_error: Exception | None = None
                while ready and len(active) < concurrency:
                    node = ready.popleft()
                    try:
                        self._reserve_node(node)
                    except Exception as exc:
                        # No provider call has started for this node. Stop
                        # admitting work, settle/cancel what is already in
                        # flight, then surface the admission failure.
                        ready.appendleft(node)
                        admission_error = exc
                        break
                    task = asyncio.create_task(
                        self._execute_node(node, graph),
                        name=f"smythe-node-{node.id}",
                    )
                    active[task] = node

                if admission_error is not None:
                    await self._cancel_and_settle(active)
                    raise admission_error
                if not active:
                    break

                done, _ = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED,
                )
                first_error: Exception | None = None
                for task in sorted(done, key=lambda item: order[active[item].id]):
                    node = active.pop(task)
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue

                    pending.pop(node.id, None)
                    resolved.add(node.id)
                    for child in dependents.get(node.id, []):
                        if child.id not in pending:
                            continue
                        unresolved[child.id] -= 1
                        if unresolved[child.id] == 0:
                            ready.append(child)

                if first_error is not None:
                    # HALT means no queued sibling may start after the error
                    # is observed. Cancel in-flight siblings and always await
                    # them so no provider task escapes the failed run.
                    await self._cancel_and_settle(active)
                    raise first_error
        except BaseException:
            await self._cancel_and_settle(active)
            raise

        if pending:
            failed = [n for n in graph.nodes if n.status == NodeStatus.FAILED]
            if failed:
                raise RuntimeError(
                    f"Execution halted: {len(pending)} node(s) blocked by "
                    f"{len(failed)} failed upstream node(s)"
                )
            raise RuntimeError("Deadlock: pending nodes exist but none are ready")
        return graph

    def _reserve_node(self, node: Node) -> None:
        """Reserve the best available per-call estimate before admission."""
        self.reserve_node_budget(
            node, default_estimated_tokens=self._estimated_tokens_per_node,
        )

    async def _cancel_and_settle(
        self, active: dict[asyncio.Task[None], Node],
    ) -> None:
        """Cancel and await active node tasks, releasing unused reservations."""
        if not active:
            return
        tasks = list(active)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            node = active.pop(task)
            if node.status in (NodeStatus.PENDING, NodeStatus.RUNNING):
                if self._budget:
                    self._budget.release(node.id)
                node.status = NodeStatus.PENDING
                node.result = None

    async def _execute_node(self, node: Node, graph: ExecutionGraph) -> None:
        """Run a single node through the provider, respecting its failure policy."""
        last_exc: Exception | None = None
        attempts = 1 + max(node.max_retries, 0) if node.failure_policy == FailurePolicy.RETRY else 1

        for attempt in range(attempts):
            delay = self.retry_delay_s(attempt)
            if delay:
                await asyncio.sleep(delay)
            node.status = NodeStatus.RUNNING
            self._tracer.on_node_start(node)

            try:
                # Cost recording happens inside acall_node (per provider call).
                result = await self.acall_node(node, graph)
                # Artifact writes are file I/O — keep them off the event
                # loop so a wide wave's completions don't serialize.
                finalize_task = asyncio.create_task(
                    asyncio.to_thread(self.finalize_node_result, node, result),
                )
                try:
                    await asyncio.shield(finalize_task)
                except asyncio.CancelledError:
                    # A sibling can fail after this response has already been
                    # billed. Finish the atomic write instead of abandoning a
                    # worker thread that could mutate the node after run()
                    # returns; this completed node remains a resumable result.
                    try:
                        await finalize_task
                    except Exception as exc:
                        raise NodeFinalizationError(node.id, exc) from exc
                except Exception as exc:
                    raise NodeFinalizationError(node.id, exc) from exc
                node.status = NodeStatus.COMPLETED
                self._tracer.on_node_end(node)
                self.notify_update(node)
                return
            except asyncio.CancelledError:
                if self._budget:
                    self._budget.release(node.id)
                node.status = NodeStatus.PENDING
                node.result = None
                self._tracer.on_node_end(node)
                raise
            except (NodeFinalizationError, SentinelAlert) as exc:
                # A reconciliation alert or post-billing persistence failure
                # is non-retryable here: another provider call would compound
                # the spend rather than repair the local failure.
                node.status = NodeStatus.FAILED
                node.result = str(exc)
                self._tracer.on_node_error(node, exc)
                self._tracer.on_node_end(node)
                self.notify_update(node)
                raise
            except Exception as exc:
                last_exc = exc
                self._tracer.on_node_error(node, exc)
                self._tracer.on_node_end(node)

        if self._budget:
            self._budget.release(node.id)

        node.result = str(last_exc)

        if node.failure_policy == FailurePolicy.SKIP:
            node.status = NodeStatus.SKIPPED
            self.notify_update(node)
            return

        node.status = NodeStatus.FAILED
        self.notify_update(node)
        raise last_exc  # type: ignore[misc]
