"""Swarm — top-level orchestrator that ties architect, registry, executor, and synthesizer."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from smythe.budget import BudgetValidationError, Sentinel
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
    CheckpointStore,
    agents_from_list,
    build_state,
    graph_from_dict,
    reset_incomplete_nodes,
    task_from_dict,
)
from smythe.executor import Executor
from smythe.executor_base import DEFAULT_MAX_TOOL_ITERATIONS, ExecutorBase
from smythe.graph import ExecutionGraph
from smythe.memory import PlannerMemory
from smythe.tools import ToolRuntime
from smythe.planner import Architect, LLMArchitect, SimpleArchitect
from smythe.router import WhiteRabbit
from smythe.provider import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIImageProvider,
    OpenAIProvider,
    Provider,
    ProviderResponseError,
    _response_error_marker,
)
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.synthesizer import Synthesizer
from smythe.task import Task, snapshot_task, task_snapshots_equal, task_to_dict
from smythe.tracer import Tracer
from smythe.verifier import Verifier, validate_verification_checkpoint, verification_pending
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_policy import WorkflowGraphPolicy, snapshot_graph_policy
from smythe.workflow_store import SQLiteWorkflowStore


@dataclass
class SwarmResult:
    """The output of a swarm execution.

    Attributes:
        output: Synthesized final result.
        graph: The executed DAG (with per-node results and statuses).
        trace: Structured spans for observability and planner feedback.
        total_cost_usd: Cumulative cost of all LLM calls during execution.
        cost_is_complete: False when at least one call had unknown USD cost.
        cost_contains_estimates: True when the total includes configured
            provider estimates or conservative request ceilings.
        execution_id: Identifier for this execution; pass to
            ``swarm.resume()`` when a checkpoint store is configured.
        cost_scope: Accounting boundary: ordinary execution/synthesis or the
            opt-in complete text workflow.
        workflow_accounting: Exact confirmed, reserved, and unknown nanoUSD
            balances and call counts for a managed text workflow.
    """

    output: str
    graph: ExecutionGraph
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    execution_id: str | None = None
    cost_is_complete: bool = True
    cost_contains_estimates: bool = False
    cost_scope: str = "execution_and_synthesis"
    workflow_accounting: dict[str, Any] | None = None


def _auto_detect_provider(model: str) -> Provider:
    """Infer the right provider from the model name."""
    lower = model.lower()
    if lower.startswith("claude"):
        return AnthropicProvider()
    if lower.startswith("gpt-image-"):
        return OpenAIImageProvider()
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
        model: str = "claude-opus-4-8",
        max_budget_usd: float | None = None,
        provider: Provider | None = None,
        architect: Architect | None = None,
        registry: Registry | None = None,
        synthesizer: Synthesizer | None = None,
        parallel: bool = False,
        planning_model: str | None = None,
        planning_provider: Provider | None = None,
        memory: PlannerMemory | None = None,
        router: WhiteRabbit | None = None,
        max_concurrency: int | None = 8,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_every_n_nodes: int = 1,
        tool_runtime: ToolRuntime | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        artifact_dir: str | Path | None = "smythe_artifacts",
        retry_backoff_s: float = 0.0,
        supervisor: Supervisor | None = None,
        max_revisions: int = 0,
        verifier: Verifier | None = None,
        run_store: SQLiteWorkflowStore | None = None,
        graph_policy: WorkflowGraphPolicy | None = None,
    ) -> None:
        Sentinel(max_budget_usd)  # Reject malformed policy before planning can call a provider.
        if graph_policy is not None and run_store is None:
            raise WorkflowBindingError("graph_policy requires a durable run_store")
        self._graph_policy = snapshot_graph_policy(graph_policy)
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.parallel = parallel
        self.max_concurrency = max_concurrency
        self.max_tool_iterations = max_tool_iterations
        self.artifact_dir = artifact_dir
        self.retry_backoff_s = retry_backoff_s
        self.supervisor = supervisor
        self.max_revisions = max_revisions
        self.verifier = verifier
        self._run_store = run_store
        # The running executor, so checkpoints can record the revision
        # allowance actually consumed rather than assuming zero.
        self._active_executor: Any = None
        self._checkpoint_store = checkpoint_store
        if (
            isinstance(checkpoint_every_n_nodes, bool)
            or not isinstance(checkpoint_every_n_nodes, int)
            or checkpoint_every_n_nodes < 1
        ):
            raise ValueError(
                "checkpoint_every_n_nodes must be >= 1, got "
                f"{checkpoint_every_n_nodes}"
            )
        self.checkpoint_every_n_nodes = checkpoint_every_n_nodes
        self._tool_runtime = tool_runtime
        self._provider = provider or _auto_detect_provider(model)
        self._memory = memory
        self._registry = registry or Registry()
        self._synthesizer = synthesizer or Synthesizer()
        self._yaml_graph: ExecutionGraph | None = None
        self._router = router

        if architect is not None:
            self._architect = architect
        else:
            # Planning is structured, low-creativity work: it does not
            # need the executor's model, and often should not pay for it.
            # A separate provider also lets planning run on a different
            # vendor entirely.
            self._architect = LLMArchitect(
                provider=planning_provider or self._provider,
                planning_model=planning_model or model,
                memory=memory,
                registry=self._registry,
            )

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate and assign an execution graph without running it.

        Returns the graph so you can inspect the architect's decisions
        before committing to execution.
        """
        if self._run_store is not None:
            return asyncio.run(self._workflow_runtime().plan(task))
        Sentinel(self.max_budget_usd)
        task = snapshot_task(task)
        architect = self._select_architect(snapshot_task(task))
        graph, architect_registry = architect.plan(snapshot_task(task))

        for agent in architect_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        graph.task = task
        self._stamp_model(graph)
        self._stamp_task_context(graph, task)
        return graph

    async def aplan(self, task: Task) -> ExecutionGraph:
        """Async variant of plan() — safe to call from a running event loop."""
        if self._run_store is not None:
            return await self._workflow_runtime().plan(task)
        Sentinel(self.max_budget_usd)
        task = snapshot_task(task)
        architect = await self._aselect_architect(snapshot_task(task))
        graph, architect_registry = await architect.aplan(snapshot_task(task))

        for agent in architect_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        graph.task = task
        self._stamp_model(graph)
        self._stamp_task_context(graph, task)
        return graph

    @staticmethod
    def _stamp_task_context(graph: ExecutionGraph, task: Task) -> None:
        """Refresh every node from the authoritative full Task snapshot."""
        ExecutorBase.stamp_task_context(graph.nodes, task)

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
        if self._run_store is not None:
            return asyncio.run(self._workflow_runtime().execute(
                task_or_graph, max_concurrency=self.max_concurrency if self.parallel else 1,
            ))
        if self.parallel:
            return asyncio.run(self.execute_async(task_or_graph))
        return self._execute_sync(task_or_graph)

    async def execute_async(
        self, task_or_graph: Task | ExecutionGraph | None = None,
    ) -> SwarmResult:
        """Async execution using the parallel AsyncExecutor."""
        task_or_graph = self._resolve_input(task_or_graph)
        if self._run_store is not None:
            return await self._workflow_runtime().execute(
                task_or_graph, max_concurrency=self.max_concurrency,
            )
        from smythe.async_executor import AsyncExecutor

        tracer = Tracer()
        budget = Sentinel(self.max_budget_usd)

        self._active_executor = None
        if isinstance(task_or_graph, Task):
            graph = await self.aplan(task_or_graph)
        else:
            graph = self._prepare_graph(task_or_graph)
        task = snapshot_task(graph.task) if graph.task is not None else None

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at,
        )

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self.max_concurrency,
            tool_runtime=self._tool_runtime,
            max_tool_iterations=self.max_tool_iterations,
            artifact_dir=self._run_artifact_dir(execution_id),
            retry_backoff_s=self.retry_backoff_s,
            supervisor=self.supervisor,
            max_revisions=self.max_revisions,
            task=task,
            verifier=self.verifier,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        self._active_executor = executor
        try:
            graph = await executor.run(graph)
            output = await self._synthesizer.asynthesize(
                graph,
                provider=self._provider,
                model=self.model,
                budget=budget,
                tracer=tracer,
            )
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            cost_is_complete=budget.cost_is_complete,
            cost_contains_estimates=budget.cost_contains_estimates,
            execution_id=execution_id,
        )

        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)

        return result

    def _execute_sync(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Serial execution using the standard Executor."""
        tracer = Tracer()
        budget = Sentinel(self.max_budget_usd)

        self._active_executor = None
        if isinstance(task_or_graph, Task):
            graph = self.plan(task_or_graph)
        else:
            graph = self._prepare_graph(task_or_graph)
        task = snapshot_task(graph.task) if graph.task is not None else None

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at,
        )

        executor = Executor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget,
            tool_runtime=self._tool_runtime,
            max_tool_iterations=self.max_tool_iterations,
            artifact_dir=self._run_artifact_dir(execution_id),
            retry_backoff_s=self.retry_backoff_s,
            supervisor=self.supervisor,
            max_revisions=self.max_revisions,
            task=task,
            verifier=self.verifier,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        self._active_executor = executor
        try:
            graph = executor.run(graph)
            output = self._synthesizer.synthesize(
                graph,
                provider=self._provider,
                model=self.model,
                budget=budget,
                tracer=tracer,
            )
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            cost_is_complete=budget.cost_is_complete,
            cost_contains_estimates=budget.cost_contains_estimates,
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
        accounting_error: str | None = None,
        response_error: dict | None = None,
    ) -> None:
        """Persist full execution state, if a checkpoint store is configured."""
        if self._checkpoint_store is None:
            return
        state = build_state(
            control={
                "revisions_used": getattr(
                    self._active_executor, "revisions_used", 0,
                ),
            },
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
        if accounting_error is not None:
            state["budget"]["accounting_error"] = accounting_error
        if response_error is not None:
            state["control"]["response_error"] = response_error
        self._checkpoint_store.save(execution_id, state)

    def _checkpointer(
        self,
        execution_id: str,
        graph: ExecutionGraph,
        budget: Sentinel,
        task: Task | None,
        created_at: float,
    ):
        """Build the optionally batched node-update checkpoint hook.

        The default interval of one preserves the strongest durability
        guarantee.  Wide jobs can choose a larger interval to avoid rewriting
        a multi-thousand-node graph after every completion; the initial,
        failed, and completed checkpoints are always written separately.
        """
        if self._checkpoint_store is None:
            return None

        completed_since_save = 0

        def _on_node_update(_node) -> None:
            nonlocal completed_since_save
            completed_since_save += 1
            if completed_since_save < self.checkpoint_every_n_nodes:
                return
            completed_since_save = 0
            self._save_checkpoint(
                execution_id, "running", graph, budget, task, created_at,
            )

        return _on_node_update

    def _control_checkpointer(self, execution_id, graph, budget, task, created_at):
        """Control transitions must be durable even with batched node saves."""
        if self._checkpoint_store is None:
            return None

        def _on_control_update() -> None:
            self._save_checkpoint(
                execution_id, "running", graph, budget, task, created_at,
            )

        return _on_control_update

    def resume(self, execution_id: str) -> SwarmResult:
        """Resume a checkpointed execution.  Sync wrapper around aresume()."""
        return asyncio.run(self.aresume(execution_id))

    async def aresume(self, execution_id: str) -> SwarmResult:
        """Resume from the last checkpoint of a prior execution.

        COMPLETED and SKIPPED nodes keep their recorded results and are
        not re-executed; RUNNING and FAILED nodes are reset to PENDING
        and re-run. Cost accounting continues against the budget policy
        recorded in the checkpoint.  If the checkpointed execution
        already finished, its stored result is returned without
        re-executing anything.
        """
        if self._run_store is not None:
            return await self._workflow_runtime().resume(
                execution_id, max_concurrency=self.max_concurrency,
            )
        if self._checkpoint_store is None:
            raise ValueError(
                "Cannot resume: this Swarm has no checkpoint_store configured."
            )
        state = self._checkpoint_store.load(execution_id)
        if state is None:
            raise KeyError(f"No checkpoint found for execution {execution_id!r}")
        version = state.get("version")
        if type(version) is not int or version not in SUPPORTED_CHECKPOINT_VERSIONS:
            raise ValueError(
                f"Checkpoint version {version!r} cannot be read by this "
                f"build of smythe, which writes version "
                f"{CHECKPOINT_VERSION} and reads "
                f"{list(SUPPORTED_CHECKPOINT_VERSIONS)}. Re-run the task "
                f"instead of resuming."
            )

        graph = graph_from_dict(state["graph"])
        if graph.run_ref is not None:
            raise ValueError("This graph requires its matching durable run_store")
        checkpoint_task = task_from_dict(state.get("task"))
        if (graph.task is not None and checkpoint_task is not None
                and not task_snapshots_equal(task_to_dict(graph.task), task_to_dict(checkpoint_task))):
            raise ValueError("Checkpoint graph Task conflicts with the top-level Task")
        task = graph.task if graph.task is not None else checkpoint_task
        if task is not None:
            task = snapshot_task(task)
            graph.task = snapshot_task(task)
            self._stamp_task_context(graph, task)

        budget_state = state.get("budget", {})
        if not isinstance(budget_state, dict):
            raise BudgetValidationError("Checkpoint budget must be an object")
        budget = Sentinel(budget_state.get("max_budget_usd"))
        budget.restore(
            budget_state.get("node_costs", {}),
            unknown_cost_nodes={
                node.id for node in graph.nodes
                if node.metadata.get("cost_usd_unknown")
            },
            estimated_cost_nodes={
                node.id for node in graph.nodes
                if node.metadata.get("cost_usd_is_estimate")
            },
        )
        invalid_nodes = [node.id for node in graph.nodes if node.metadata.get("accounting_invalid")]
        if invalid_nodes or "accounting_error" in budget_state:
            raise BudgetValidationError(
                "Cannot resume unresolved accounting for this workflow "
                f"(nodes: {invalid_nodes!r}); reconcile provider charges and repair the "
                "checkpoint before clearing its accounting markers."
            )
        if ("response_error" in state.get("control", {})
                or any("response_error" in node.metadata for node in graph.nodes)):
            raise ProviderResponseError(
                "Cannot resume an unresolved native response failure; reconcile its accounting "
                "and repair the saved output before clearing the response-error marker."
            )

        completed = state.get("status") == "completed" and state.get("output") is not None
        validate_verification_checkpoint(graph, version=version, completed=completed)
        if completed and not verification_pending(graph):
            return SwarmResult(
                output=state["output"],
                graph=graph,
                trace=[],
                total_cost_usd=budget.total_cost_usd,
                cost_is_complete=budget.cost_is_complete,
                cost_contains_estimates=budget.cost_contains_estimates,
                execution_id=execution_id,
            )

        from smythe.async_executor import AsyncExecutor

        for agent in agents_from_list(state.get("agents", [])):
            self._registry.register(agent)
        reset_incomplete_nodes(graph)
        graph.validate()
        self._stamp_model(graph)

        tracer = Tracer()
        created_at = state.get("created_at", time.time())

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self.max_concurrency,
            tool_runtime=self._tool_runtime,
            max_tool_iterations=self.max_tool_iterations,
            artifact_dir=self._run_artifact_dir(execution_id),
            retry_backoff_s=self.retry_backoff_s,
            supervisor=self.supervisor,
            max_revisions=self.max_revisions,
            # The revision cap bounds the run, not the attempt: carrying the
            # spent allowance forward stops a crash-resume cycle from
            # refilling it and looping past max_revisions.
            revisions_used=state.get("control", {}).get("revisions_used", 0),
            task=task,
            verifier=self.verifier,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at,
            ),
        )
        self._active_executor = executor
        try:
            graph = await executor.run(graph)
            output = await self._synthesizer.asynthesize(
                graph,
                provider=self._provider,
                model=self.model,
                budget=budget,
                tracer=tracer,
            )
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at,
            output=output,
        )

        result = SwarmResult(
            output=output,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            cost_is_complete=budget.cost_is_complete,
            cost_contains_estimates=budget.cost_contains_estimates,
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
        model: str = "claude-opus-4-8",
        max_budget_usd: float | None = None,
        provider: Provider | None = None,
        parallel: bool = False,
        run_store: SQLiteWorkflowStore | None = None,
        graph_policy: WorkflowGraphPolicy | None = None,
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
            run_store=run_store,
            graph_policy=graph_policy,
        )
        if instance._graph_policy is not None:
            instance._graph_policy.validate(graph, default_model=instance.model)
        instance._yaml_graph = graph
        instance._stamp_model(graph)
        return instance

    def _stamp_model(self, graph: ExecutionGraph) -> None:
        """Tag every node with the swarm's model so the executor knows which to call."""
        for node in graph.nodes:
            if "model" not in node.metadata:
                node.metadata["model"] = self.model

    def _prepare_graph(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Ready a caller-built graph for execution.

        Graphs built by hand skip plan()/from_yaml(), so nodes have no
        model stamped yet; providers reject an empty model name.  One
        shared helper so no execute path can forget it again.
        """
        if graph.run_ref is not None:
            raise ValueError("This graph requires its matching durable run_store")
        graph.validate()
        self._stamp_model(graph)
        if graph.task is not None:
            graph.task = snapshot_task(graph.task)
            self._stamp_task_context(graph, graph.task)
        return graph

    def _run_artifact_dir(self, execution_id: str) -> Path | None:
        """Per-execution artifact directory.

        Scoping by execution id keeps re-runs of the same graph (fixed
        node ids in YAML/hand-built graphs) from silently overwriting a
        previous run's files while old checkpoints still point at them.
        resume() reuses the original execution_id, so a resumed run
        lands in the same directory — which is the correct semantics.
        """
        if self.artifact_dir is None:
            return None
        return Path(self.artifact_dir) / execution_id[:12]

    def _workflow_runtime(self):
        """Pass explicit component dependencies into an isolated durable run."""
        from smythe.workflow import WorkflowRuntime

        return WorkflowRuntime(
            store=self._run_store, model=self.model, max_budget_usd=self.max_budget_usd,
            provider=self._provider, architect=self._architect, registry=self._registry,
            router=self._router, synthesizer=self._synthesizer, supervisor=self.supervisor,
            max_revisions=self.max_revisions, verifier=self.verifier,
            memory=self._memory, tool_runtime=self._tool_runtime,
            checkpoint_store=self._checkpoint_store,
            checkpoint_every_n_nodes=self.checkpoint_every_n_nodes,
            retry_backoff_s=self.retry_backoff_s,
            max_concurrency=self.max_concurrency,
            graph_policy=self._graph_policy,
        )
