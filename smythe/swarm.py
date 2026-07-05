"""Swarm — top-level orchestrator that ties architect, registry, executor, and synthesizer."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from smythe.budget import Sentinel
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointStore,
    agents_from_list,
    build_state,
    graph_from_dict,
    reset_incomplete_nodes,
    task_from_dict,
)
from smythe.executor import Executor
from smythe.graph import ExecutionGraph
from smythe.memory import PlannerMemory
from smythe.planner import Architect, LLMArchitect, SimpleArchitect
from smythe.router import WhiteRabbit
from smythe.provider import AnthropicProvider, GeminiProvider, OpenAIProvider, Provider
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
        execution_id: Identifier for this execution; pass to
            ``swarm.resume()`` when a checkpoint store is configured.
    """

    output: str
    graph: ExecutionGraph
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    execution_id: str | None = None


def _auto_detect_provider(model: str) -> Provider:
    """Infer the right provider from the model name."""
    lower = model.lower()
    if lower.startswith("claude"):
        return AnthropicProvider()
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIProvider()
    if lower.startswith("gemini"):
        return GeminiProvider()
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
        architect: Architect | None = None,
        registry: Registry | None = None,
        synthesizer: Synthesizer | None = None,
        parallel: bool = False,
        planning_model: str | None = None,
        memory: PlannerMemory | None = None,
        router: WhiteRabbit | None = None,
        max_concurrency: int | None = 8,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.parallel = parallel
        self.max_concurrency = max_concurrency
        self._checkpoint_store = checkpoint_store
        self._provider = provider or _auto_detect_provider(model)
        self._memory = memory
        self._registry = registry or Registry()
        self._synthesizer = synthesizer or Synthesizer()
        self._yaml_graph: ExecutionGraph | None = None
        self._router = router

        if architect is not None:
            self._architect = architect
        else:
            self._architect = LLMArchitect(
                provider=self._provider,
                planning_model=planning_model or model,
                memory=memory,
            )

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate and assign an execution graph without running it.

        Returns the graph so you can inspect the architect's decisions
        before committing to execution.
        """
        architect = self._select_architect(task)
        graph, architect_registry = architect.plan(task)

        for agent in architect_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        self._stamp_model(graph)
        return graph

    async def aplan(self, task: Task) -> ExecutionGraph:
        """Async variant of plan() — safe to call from a running event loop."""
        architect = await self._aselect_architect(task)
        graph, architect_registry = await architect.aplan(task)

        for agent in architect_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        self._stamp_model(graph)
        return graph

    def _select_architect(self, task: Task) -> Architect:
        """Pick the architect — use router if set, otherwise the default."""
        if self._router is not None:
            return self._router.route(task)
        return self._architect

    async def _aselect_architect(self, task: Task) -> Architect:
        """Async architect selection — uses aroute when a router is set."""
        if self._router is not None:
            return await self._router.aroute(task)
        return self._architect

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
        budget = Sentinel(self.max_budget_usd)

        task = task_or_graph if isinstance(task_or_graph, Task) else None
        if task is not None:
            graph = await self.aplan(task)
        else:
            graph = task_or_graph
            graph.validate()

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at,
        )

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self.max_concurrency,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        try:
            graph = await executor.run(graph)
        except Exception:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
            )
            raise

        output = await self._synthesizer.asynthesize(
            graph,
            provider=self._provider,
            model=self.model,
            budget=budget,
            tracer=tracer,
        )

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            execution_id=execution_id,
        )

        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)

        return result

    def _execute_sync(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Serial execution using the standard Executor."""
        tracer = Tracer()
        budget = Sentinel(self.max_budget_usd)

        task = task_or_graph if isinstance(task_or_graph, Task) else None
        if task is not None:
            graph = self.plan(task)
        else:
            graph = task_or_graph
            graph.validate()

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at,
        )

        executor = Executor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        try:
            graph = executor.run(graph)
        except Exception:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
            )
            raise

        output = self._synthesizer.synthesize(
            graph,
            provider=self._provider,
            model=self.model,
            budget=budget,
            tracer=tracer,
        )

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            execution_id=execution_id,
        )

        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)

        return result

    def _save_checkpoint(
        self,
        execution_id: str,
        status: str,
        graph: ExecutionGraph,
        budget: Sentinel,
        task: Task | None,
        created_at: float,
        output: str | None = None,
    ) -> None:
        """Persist full execution state, if a checkpoint store is configured."""
        if self._checkpoint_store is None:
            return
        state = build_state(
            execution_id=execution_id,
            status=status,
            model=self.model,
            graph=graph,
            registry=self._registry,
            task=task,
            max_budget_usd=budget.max_budget_usd,
            node_costs=budget.breakdown(),
            output=output,
            created_at=created_at,
        )
        self._checkpoint_store.save(execution_id, state)

    def _checkpointer(
        self,
        execution_id: str,
        graph: ExecutionGraph,
        budget: Sentinel,
        task: Task | None,
        created_at: float,
    ):
        """Build the per-node on_node_update hook, or None when not checkpointing."""
        if self._checkpoint_store is None:
            return None

        def _on_node_update(_node) -> None:
            self._save_checkpoint(
                execution_id, "running", graph, budget, task, created_at,
            )

        return _on_node_update

    def resume(self, execution_id: str) -> SwarmResult:
        """Resume a checkpointed execution.  Sync wrapper around aresume()."""
        return asyncio.run(self.aresume(execution_id))

    async def aresume(self, execution_id: str) -> SwarmResult:
        """Resume from the last checkpoint of a prior execution.

        COMPLETED and SKIPPED nodes keep their recorded results and are
        not re-executed; RUNNING and FAILED nodes are reset to PENDING
        and re-run.  Cost accounting continues against the budget cap
        recorded in the checkpoint.  If the checkpointed execution
        already finished, its stored result is returned without
        re-executing anything.
        """
        if self._checkpoint_store is None:
            raise ValueError(
                "Cannot resume: this Swarm has no checkpoint_store configured."
            )
        state = self._checkpoint_store.load(execution_id)
        if state is None:
            raise KeyError(f"No checkpoint found for execution {execution_id!r}")
        version = state.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {version!r} is not supported "
                f"(expected {CHECKPOINT_VERSION})"
            )

        graph = graph_from_dict(state["graph"])

        if state.get("status") == "completed" and state.get("output") is not None:
            return SwarmResult(
                output=state["output"],
                graph=graph,
                trace=[],
                total_cost_usd=sum(state["budget"].get("node_costs", {}).values()),
                execution_id=execution_id,
            )

        from smythe.async_executor import AsyncExecutor

        for agent in agents_from_list(state.get("agents", [])):
            self._registry.register(agent)
        reset_incomplete_nodes(graph)
        graph.validate()
        self._stamp_model(graph)

        task = task_from_dict(state.get("task"))
        tracer = Tracer()
        budget = Sentinel(state.get("budget", {}).get("max_budget_usd"))
        budget.restore(state.get("budget", {}).get("node_costs", {}))
        created_at = state.get("created_at", time.time())

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self.max_concurrency,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        try:
            graph = await executor.run(graph)
        except Exception:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
            )
            raise

        output = await self._synthesizer.asynthesize(
            graph,
            provider=self._provider,
            model=self.model,
            budget=budget,
            tracer=tracer,
        )

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            execution_id=execution_id,
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
            architect=SimpleArchitect(),
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
