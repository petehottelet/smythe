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
        model: str = "gpt-4o",
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

    def execute(self, task: Task) -> SwarmResult:
        """Plan, assign, execute, and synthesize a task end-to-end."""
        tracer = Tracer()

        graph = self._planner.plan(task)
        graph = self._registry.assign(graph)

        executor = Executor(registry=self._registry, tracer=tracer)
        graph = executor.run(graph)

        output = self._synthesizer.synthesize(graph)

        return SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
        )
