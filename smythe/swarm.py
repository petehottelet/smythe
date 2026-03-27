"""Swarm — top-level orchestrator that ties planner, registry, executor, and synthesizer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from smythe.budget import BudgetTracker
from smythe.executor import Executor
from smythe.graph import ExecutionGraph
from smythe.memory import PlannerMemory
from smythe.planner import LLMPlanner, Planner, SimplePlanner
from smythe.router import PlannerRouter
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
        total_cost_usd: Cumulative cost of all LLM calls during execution.
    """

    output: str
    graph: ExecutionGraph
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0


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
        planning_model: str | None = None,
        memory: PlannerMemory | None = None,
        router: PlannerRouter | None = None,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.parallel = parallel
        self._provider = provider or _auto_detect_provider(model)
        self._memory = memory
        self._registry = registry or Registry()
        self._synthesizer = synthesizer or Synthesizer()
        self._yaml_graph: ExecutionGraph | None = None
        self._router = router

        if planner is not None:
            self._planner = planner
        else:
            self._planner = LLMPlanner(
                provider=self._provider,
                planning_model=planning_model or model,
                memory=memory,
            )

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate and assign an execution graph without running it.

        Returns the graph so you can inspect the planner's decisions
        before committing to execution.
        """
        planner = self._select_planner(task)
        graph, planner_registry = planner.plan(task)

        for agent in planner_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        self._stamp_model(graph)
        return graph

    async def aplan(self, task: Task) -> ExecutionGraph:
        """Async variant of plan() — safe to call from a running event loop."""
        planner = await self._aselect_planner(task)
        graph, planner_registry = await planner.aplan(task)

        for agent in planner_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        self._stamp_model(graph)
        return graph

    def _select_planner(self, task: Task) -> Planner:
        """Pick the planner — use router if set, otherwise the default."""
        if self._router is not None:
            return self._router.route(task)
        return self._planner

    async def _aselect_planner(self, task: Task) -> Planner:
        """Async planner selection — uses aroute when a router is set."""
        if self._router is not None:
            return await self._router.aroute(task)
        return self._planner

    def execute(
        self, task_or_graph: Task | ExecutionGraph | None = None,
    ) -> SwarmResult:
        """Execute a task or a previously planned graph.

        Accepts either a Task (plans and executes in one call), an
        ExecutionGraph returned by plan(), or None when a YAML graph
        was loaded via from_yaml().
        If parallel=True, uses the async executor for concurrent nodes.
        """
        task_or_graph = self._resolve_input(task_or_graph)
        if self.parallel:
            return asyncio.run(self.execute_async(task_or_graph))
        return self._execute_sync(task_or_graph)

    async def execute_async(
        self, task_or_graph: Task | ExecutionGraph | None = None,
    ) -> SwarmResult:
        """Async execution using the parallel AsyncExecutor."""
        task_or_graph = self._resolve_input(task_or_graph)
        from smythe.async_executor import AsyncExecutor

        tracer = Tracer()
        budget = BudgetTracker(self.max_budget_usd)

        task = task_or_graph if isinstance(task_or_graph, Task) else None
        if task is not None:
            graph = await self.aplan(task)
        else:
            graph = task_or_graph

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget,
        )
        graph = await executor.run(graph)

        output = await self._synthesizer.asynthesize(
            graph,
            provider=self._provider,
            budget=budget,
            tracer=tracer,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
        )

        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)

        return result

    def _execute_sync(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Serial execution using the standard Executor."""
        tracer = Tracer()
        budget = BudgetTracker(self.max_budget_usd)

        task = task_or_graph if isinstance(task_or_graph, Task) else None
        if task is not None:
            graph = self.plan(task)
        else:
            graph = task_or_graph

        executor = Executor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget,
        )
        graph = executor.run(graph)

        output = self._synthesizer.synthesize(
            graph,
            provider=self._provider,
            budget=budget,
            tracer=tracer,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
        )

        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)

        return result

    def _resolve_input(
        self, task_or_graph: Task | ExecutionGraph | None,
    ) -> Task | ExecutionGraph:
        """Fall back to the YAML-loaded graph when no argument is given."""
        if task_or_graph is not None:
            return task_or_graph
        if self._yaml_graph is not None:
            return self._yaml_graph
        raise ValueError(
            "No task or graph provided, and no YAML graph loaded. "
            "Pass a Task or ExecutionGraph, or use Swarm.from_yaml()."
        )

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        model: str = "claude-mythos",
        max_budget_usd: float | None = None,
        provider: Provider | None = None,
        parallel: bool = False,
    ) -> Swarm:
        """Create a Swarm pre-loaded with a YAML-defined execution graph.

        The returned Swarm's internal registry is populated with agents
        defined in the YAML file.  Call execute() with the loaded graph
        to run it.
        """
        from smythe.loader import load_graph

        graph, registry = load_graph(path)
        instance = cls(
            model=model,
            max_budget_usd=max_budget_usd,
            provider=provider,
            registry=registry,
            planner=SimplePlanner(),
            parallel=parallel,
        )
        instance._yaml_graph = graph
        instance._stamp_model(graph)
        return instance

    def _stamp_model(self, graph: ExecutionGraph) -> None:
        """Tag every node with the swarm's model so the executor knows which to call."""
        for node in graph.nodes:
            if "model" not in node.metadata:
                node.metadata["model"] = self.model
