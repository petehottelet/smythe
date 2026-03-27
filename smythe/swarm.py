"""Swarm — top-level orchestrator that ties planner, registry, executor, and synthesizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smythe.executor import Executor
from smythe.graph import ExecutionGraph
from smythe.planner import Planner
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


class Swarm:
    """Entry point for submitting tasks.

    Coordinates the full lifecycle: plan -> assign -> execute -> synthesize.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-6",
        max_budget_usd: float | None = None,
        planner: Planner | None = None,
        registry: Registry | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
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
        return graph

    def execute(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Execute a task or a previously planned graph.

        Accepts either a Task (plans and executes in one call) or an
        ExecutionGraph returned by plan() for inspect-then-run workflows.
        """
        tracer = Tracer()

        if isinstance(task_or_graph, Task):
            graph = self.plan(task_or_graph)
        else:
            graph = task_or_graph

        executor = Executor(registry=self._registry, tracer=tracer)
        graph = executor.run(graph)

        output = self._synthesizer.synthesize(graph)

        return SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
        )
