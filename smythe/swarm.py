"""Swarm — top-level orchestrator that ties planner, registry, executor, and synthesizer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from smythe.executor import Executor
from smythe.graph import ExecutionGraph
from smythe.planner import Planner
from smythe.provider import AnthropicProvider, OpenAIProvider, Provider
from smythe.registry import Registry
from smythe.synthesizer import Synthesizer
from smythe.task import Task
from smythe.tracer import Tracer


@dataclass
class SwarmResult:
    """The output of a swarm execution.

    Attributes:
        output: Synthesized final result.
        graph: The executed DAG (with per-node results and statuses).
        trace: Structured spans for observability and planner feedback.
    """

    output: str
    graph: ExecutionGraph
    trace: list[dict[str, Any]] = field(default_factory=list)


def _auto_detect_provider(model: str) -> Provider:
    """Infer the right provider from the model name."""
    lower = model.lower()
    if lower.startswith("claude"):
        return AnthropicProvider()
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIProvider()
    raise ValueError(
        f"Cannot auto-detect provider for model {model!r}. "
        "Pass an explicit provider= argument to Swarm()."
    )


class Swarm:
    """Entry point for submitting tasks.

    Coordinates the full lifecycle: plan -> assign -> execute -> synthesize.
    """

    def __init__(
        self,
        *,
        model: str = "claude-mythos",
        max_budget_usd: float | None = None,
        provider: Provider | None = None,
        planner: Planner | None = None,
        registry: Registry | None = None,
        synthesizer: Synthesizer | None = None,
        parallel: bool = False,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.parallel = parallel
        self._provider = provider or _auto_detect_provider(model)
        self._planner = planner or Planner()
        self._registry = registry or Registry()
        self._synthesizer = synthesizer or Synthesizer()

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate and assign an execution graph without running it.

        Returns the graph so you can inspect the planner's decisions
        before committing to execution.
        """
        graph = self._planner.plan(task)
        graph = self._registry.assign(graph)
        self._stamp_model(graph)
        return graph

    def execute(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Execute a task or a previously planned graph.

        Accepts either a Task (plans and executes in one call) or an
        ExecutionGraph returned by plan() for inspect-then-run workflows.
        If parallel=True, uses the async executor for concurrent nodes.
        """
        if self.parallel:
            return asyncio.run(self.execute_async(task_or_graph))
        return self._execute_sync(task_or_graph)

    async def execute_async(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Async execution using the parallel AsyncExecutor."""
        from smythe.async_executor import AsyncExecutor

        tracer = Tracer()

        if isinstance(task_or_graph, Task):
            graph = self.plan(task_or_graph)
        else:
            graph = task_or_graph

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer
        )
        graph = await executor.run(graph)

        output = self._synthesizer.synthesize(graph)

        return SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
        )

    def _execute_sync(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Serial execution using the standard Executor."""
        tracer = Tracer()

        if isinstance(task_or_graph, Task):
            graph = self.plan(task_or_graph)
        else:
            graph = task_or_graph

        executor = Executor(
            provider=self._provider, registry=self._registry, tracer=tracer
        )
        graph = executor.run(graph)

        output = self._synthesizer.synthesize(graph)

        return SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
        )

    def _stamp_model(self, graph: ExecutionGraph) -> None:
        """Tag every node with the swarm's model so the executor knows which to call."""
        for node in graph.nodes:
            if "model" not in node.metadata:
                node.metadata["model"] = self.model
