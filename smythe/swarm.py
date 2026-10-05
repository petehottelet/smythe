"""Swarm — top-level orchestrator that ties architect, registry, executor, and synthesizer."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from smythe._run_inspection import result_view
from smythe.acceptance import (
    GENERATED_GATE_KEY,
    AcceptanceDecision,
    AcceptanceMode,
    AcceptancePolicy,
    AcceptanceSource,
    DeliveryAcceptance,
    _delivery_problems,
    evaluate_delivery,
    task_sha256,
    text_sha256,
    validate_acceptance_obligations,
)
from smythe.budget import BudgetValidationError, Sentinel
from smythe.context import ContextSource, snapshot_sources
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
    CheckpointStore,
    agents_from_list,
    build_state,
    delivery_acceptance_from_state,
    delivery_acceptance_history,
    drop_untrusted_mcp_servers,
    graph_from_dict,
    graph_to_dict,
    reset_incomplete_nodes,
    supersede_delivery_acceptance,
    task_from_dict,
    timing_segments,
)
from smythe.executor import Executor
from smythe.executor_base import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    ExecutorBase,
    check_pending_nodes,
    check_provider_support,
    validate_max_concurrency,
    validate_max_revisions,
    validate_retry_backoff_s,
)
from smythe.graph import ExecutionGraph, NodeStatus
from smythe.memory import PlannerMemory
from smythe.tools import ToolRuntime
from smythe.planner import Architect, DeterministicArchitect, LLMArchitect, SimpleArchitect
from smythe.constrained_planner import ConstrainedArchitect
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
from smythe.synthesizer import SynthesisStrategy, Synthesizer, deliverable_sources
from smythe.task import Task, snapshot_task, task_snapshots_equal, task_to_dict
from smythe.tracer import Tracer
from smythe.verifier import (
    Verifier,
    node_generation,
    validate_verification_checkpoint,
    verification_pending,
)
from smythe.workflow_binding import LocalOnly, WorkflowBindingError, _BoundLocal
from smythe.workflow_policy import WorkflowGraphPolicy, snapshot_graph_policy
from smythe.workflow_store import SQLiteWorkflowStore
from smythe.workflow_tools import READ_STEP_KEY, ReadProfile

logger = logging.getLogger("smythe.swarm")


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
        timing: Where the call's time went: elapsed time, phase totals and
            waits, with one segment per call that worked on the execution.
            None for a result built by hand.
        acceptance: The delivery acceptance record a strict policy made for
            the output, or None when no record exists, as for advisory and
            legacy runs.
        rejected_output: The delivered text when a record exists but did not
            accept it, or when a strict policy resumes a completed run that
            has no record; ``output`` is then ``""``. None otherwise.
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
    timing: dict[str, Any] | None = None
    acceptance: DeliveryAcceptance | None = None
    rejected_output: str | None = None

    @property
    def accepted(self) -> bool:
        """True only when the delivery acceptance record's decision is ``accepted``."""
        return isinstance(self.acceptance, DeliveryAcceptance) and self.acceptance.accepted

    def to_dict(self) -> dict[str, Any]:
        """This result as detached JSON data.

        The acceptance record appears under ``delivery_acceptance`` (None
        without one) and the graph in its checkpoint form. ``execution``,
        ``delivery``, ``unknown_exposure`` and ``next_action`` are the
        blocks ``smythe runs inspect`` reports for a saved run.
        """
        return json.loads(json.dumps({
            "execution_id": self.execution_id,
            "output": self.output,
            "delivery_acceptance": None if self.acceptance is None else self.acceptance.to_dict(),
            "rejected_output": self.rejected_output,
            "total_cost_usd": self.total_cost_usd,
            "cost_is_complete": self.cost_is_complete,
            "cost_contains_estimates": self.cost_contains_estimates,
            "cost_scope": self.cost_scope,
            "workflow_accounting": self.workflow_accounting,
            "timing": self.timing,
            "trace": self.trace,
            "graph": graph_to_dict(self.graph),
            **result_view(self),
        }, allow_nan=False))


class _RunTiming:
    """This call's timing segment, after any segments earlier calls saved.

    Each segment holds durations from one process's monotonic clock and
    wall-clock start and end times. Durations from different processes are
    never subtracted from each other; only the wall times relate segments.
    """

    def __init__(
        self, tracer: Tracer, *, resumed: bool = False,
        previous: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tracer = tracer
        self.resumed = resumed
        self.previous = previous or []

    def _segments(self, timing: dict[str, Any]) -> list[dict[str, Any]]:
        segment = {
            "started_at": self.tracer.started_at, "ended_at": self.tracer.wall_time(),
            "resumed": self.resumed,
        }
        segment.update((key, value) for key, value in timing.items() if key != "clock")
        return [*self.previous, segment]

    def checkpoint_block(self) -> dict[str, Any]:
        return {"version": 1, "clock": "monotonic", "segments": self._segments(self.tracer.timing())}

    def result(self) -> dict[str, Any]:
        timing = self.tracer.timing()
        return {**timing, "latency_complete": not self.resumed, "resumed": self.resumed,
                "segments": self._segments(timing)}


def _delivered(output: Any, acceptance: DeliveryAcceptance | None) -> tuple[Any, str | None]:
    """Split saved output into the delivered and the rejected slot.

    Without a record the output is delivered as saved. With one, the output
    must still match the record's digest, and only an accepted record
    delivers it; otherwise ``output`` is empty and the text is kept as
    ``rejected_output``. A ``not_evaluated`` record decided nothing, so its
    output is withheld too; one that binds no digest has none to check.
    """
    if acceptance is None:
        return output, None
    if not isinstance(output, str) or (
        acceptance.output_sha256 is not None and text_sha256(output) != acceptance.output_sha256
    ):
        raise ValueError("Checkpoint output does not match its delivery acceptance record")
    if acceptance.accepted:
        return output, None
    return "", output


def _synthesis_strategy(synthesizer: Any) -> str:
    """The strategy a run's synthesizer applies, or ``custom`` for any other synthesizer."""
    if type(synthesizer) is Synthesizer and type(synthesizer.strategy) is SynthesisStrategy:
        return synthesizer.strategy.value
    return "custom"


def _delivery_sources(graph: ExecutionGraph, strategy: str) -> tuple[list, bool]:
    """The nodes *strategy* synthesis delivered from, and whether ``DELIVERABLE`` fell back.

    Any synthesis but ``DELIVERABLE`` reads every completed node with a result.
    """
    if strategy == SynthesisStrategy.DELIVERABLE.value:
        return deliverable_sources(graph)
    return [node for node in graph.nodes
            if node.status is NodeStatus.COMPLETED and node.result is not None], False


def _builds_developer_gates(architect: Any) -> bool:
    """Whether *architect*'s gates come from developer code rather than a model.

    A ``DeterministicArchitect`` builds its graph in Python, a
    ``ConstrainedArchitect`` composes developer templates, and a durable
    run's ``LocalOnly`` architect plans in local code. Any other architect,
    such as ``LLMArchitect``, may generate its gates.
    """
    return isinstance(architect, (DeterministicArchitect, ConstrainedArchitect,
                                  LocalOnly, _BoundLocal))


def _planning_architects(router: Any, architect: Any) -> tuple[Any, ...]:
    """Every architect that could plan a run: *router*'s tiers, or *architect* without one.

    Empty for a router other than ``WhiteRabbit``, whose tiers are unknown.
    """
    if router is None:
        return (architect,)
    if type(router) is WhiteRabbit:
        return router._architects()
    return ()


def _check_gate_planning(policy: AcceptancePolicy | None, architects: tuple[Any, ...]) -> None:
    """Refuse, before planning, a strict run whose plan no gate could accept.

    Without final validators or ``allow_generated_gates`` only a gate from
    developer code accepts, so a run whose every architect generates its
    gates could never accept its output.
    """
    if (policy is not None and not policy.final_validators
            and not policy.allow_generated_gates and architects
            and not any(_builds_developer_gates(item) for item in architects)):
        raise ValueError(
            "A strict acceptance policy without a final validator accepts only output "
            "that a gate you supplied judged, and this run's plan would come from an "
            "architect that generates its gates. Add a final validator, set "
            "allow_generated_gates=True, or execute a graph you build."
        )


def _mark_generated_gates(graph: ExecutionGraph, architect: Any) -> None:
    """Mark each gate of a plan that *architect* generated with ``GENERATED_GATE_KEY``."""
    if not _builds_developer_gates(architect):
        for node in graph.nodes:
            if node.verifies:
                node.metadata[GENERATED_GATE_KEY] = True


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

    ``acceptance`` takes an :class:`~smythe.acceptance.AcceptancePolicy`. A
    strict policy decides delivery acceptance after synthesis, on exactly the
    delivered output, and records the decision with the completed
    checkpoint; output it does not accept is returned as ``rejected_output``.
    It decides only on verdicts produced in the run, so a fresh strict run
    refuses a graph whose judge completed before it. Before any node runs,
    it refuses a graph whose delivered output it could not decide
    (:func:`~smythe.acceptance.validate_acceptance_obligations`), and before
    planning a run whose plan could not meet that check.
    None, the default, and an advisory policy keep legacy behavior. With a
    durable ``run_store`` the same checks apply, the policy enters the run's
    recipe and the decision is journaled; see ``smythe.workflow.WorkflowRuntime``.
    Resuming a completed run never decides it again: :meth:`reevaluate`
    records a new decision for its saved output, under a fixed or changed
    validator, without running a node.

    ``read_tools`` declares the :class:`~smythe.workflow_tools.ReadProfile`
    reads that a durable run's read steps may name; it requires
    ``run_store``. An ordinary run refuses a read step before any node
    call, whether a caller's graph, an architect's or template's plan or a
    saved checkpoint declares it.

    ``context_sources``, a sequence of :class:`~smythe.context.ContextSource`
    that needs a durable ``run_store``, opts each new run into context
    assembly: the run records the sources once, and every node call is given
    a journaled context snapshot that marks sources and dependency outputs
    as untrusted data. :meth:`refresh_context` gives a run new sources.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
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
        verifier: Verifier | LocalOnly | None = None,
        run_store: SQLiteWorkflowStore | None = None,
        graph_policy: WorkflowGraphPolicy | None = None,
        acceptance: AcceptancePolicy | None = None,
        read_tools: tuple[ReadProfile, ...] | list[ReadProfile] = (),
        context_sources: Sequence[ContextSource] | None = None,
    ) -> None:
        Sentinel(max_budget_usd)  # Reject malformed policy before planning can call a provider.
        if graph_policy is not None and run_store is None:
            raise WorkflowBindingError("graph_policy requires a durable run_store")
        if read_tools is None:
            read_tools = ()
        if isinstance(read_tools, (str, bytes)) or not hasattr(read_tools, "__iter__"):
            raise WorkflowBindingError("read_tools must be a sequence of ReadProfile declarations")
        read_tools = tuple(read_tools)
        if read_tools:
            if run_store is None:
                raise WorkflowBindingError("read_tools requires a durable run_store")
            from smythe.workflow import _snapshot_read_tools

            _snapshot_read_tools(read_tools)
        if context_sources is not None and run_store is None:
            raise WorkflowBindingError("context_sources requires a durable run_store")
        self._context_sources = None if context_sources is None else snapshot_sources(context_sources)
        if isinstance(verifier, LocalOnly) and run_store is None:
            raise TypeError(
                "verifier: a LocalOnly validator judges gates only in a durable run_store run; "
                "an ordinary run takes a Verifier such as CallableVerifier"
            )
        if acceptance is not None and not isinstance(acceptance, AcceptancePolicy):
            raise TypeError(
                f"acceptance must be an AcceptancePolicy or None, not {type(acceptance).__name__}"
            )
        self._acceptance = acceptance
        # The undecided record a strict run's unfinished checkpoints carry.
        self._pending_record: tuple[Task | None, DeliveryAcceptance] | None = None
        # An ordinary run builds its executor only after routing and planning,
        # so check the executor's limits now. A durable run checks its own,
        # stricter limits before routing or planning on every entry point.
        if run_store is None:
            validate_retry_backoff_s(retry_backoff_s)
            validate_max_concurrency(max_concurrency)
            validate_max_revisions(max_revisions)
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
        self._read_tools = read_tools
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
        self._check_acceptance_support()
        self._yaml_graph: ExecutionGraph | None = None
        self._router = router
        # Each execute or resume call measures its time with a new tracer.
        self._tracer_factory: Callable[[], Tracer] = Tracer

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

    @property
    def acceptance(self) -> AcceptancePolicy | None:
        """The delivery acceptance policy this Swarm's runs follow, if any."""
        return self._acceptance

    def _strict_policy(self) -> AcceptancePolicy | None:
        policy = self._acceptance
        return policy if policy is not None and policy.mode is AcceptanceMode.STRICT else None

    def _check_acceptance_support(self) -> None:
        """Reject a strict policy that this Swarm's synthesis could never satisfy.

        Without final validators only a passed gate can accept, and only
        output that is one gated node's exact result, which only
        ``DELIVERABLE`` synthesis returns.
        """
        policy = self._strict_policy()
        if policy is None or policy.final_validators:
            return
        strategy = _synthesis_strategy(self._synthesizer)
        if strategy != SynthesisStrategy.DELIVERABLE.value:
            raise ValueError(
                f"A strict acceptance policy without a final validator can accept only "
                f"DELIVERABLE output that a passed gate judged; {strategy} synthesis needs "
                "a final validator"
            )

    def _pending_acceptance(self, task: Task | None) -> DeliveryAcceptance | None:
        """The undecided record a strict run's unfinished checkpoints carry."""
        policy = self._strict_policy()
        if policy is None:
            return None
        cached = self._pending_record
        if cached is not None and cached[0] is task and cached[1].policy_sha256 == policy.sha256:
            return cached[1]
        record = DeliveryAcceptance(
            AcceptanceDecision.NOT_EVALUATED, policy.sha256, task_sha256=task_sha256(task),
        )
        self._pending_record = (task, record)
        return record

    def _decide(
        self, graph: ExecutionGraph, output: Any, task: Task | None, tracer: Tracer,
    ) -> DeliveryAcceptance | None:
        """Decide delivery acceptance for exactly *output*; None without a strict policy.

        Final validators run here, after synthesis, with no provider call. A
        final rejection is terminal: nothing is regenerated.
        """
        policy = self._strict_policy()
        if policy is None:
            return None
        strategy = _synthesis_strategy(self._synthesizer)
        sources, used_fallback = _delivery_sources(graph, strategy)
        with tracer.phase("verification"):
            record = evaluate_delivery(
                policy, graph, output, strategy=strategy, sources=sources,
                used_fallback=used_fallback, task=task,
            )
        tracer.on_acceptance(record)
        return record

    def _finish(
        self,
        execution_id: str,
        graph: ExecutionGraph,
        budget: Sentinel,
        task: Task | None,
        created_at: float,
        timing: _RunTiming,
        output: Any,
        acceptance: DeliveryAcceptance | None,
    ) -> SwarmResult:
        """Save a finished run with its decision and build its result.

        Every execute and resume path ends here. Output a strict policy did
        not accept is saved and returned as ``rejected_output``, never as
        ``output``.
        """
        self._save_checkpoint(
            execution_id, "completed", graph, budget, task, created_at, timing,
            output=output, acceptance=acceptance,
        )
        delivered, rejected_output = _delivered(output, acceptance)
        result = SwarmResult(
            output=delivered,
            graph=graph,
            trace=timing.tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            cost_is_complete=budget.cost_is_complete,
            cost_contains_estimates=budget.cost_contains_estimates,
            execution_id=execution_id,
            timing=timing.result(),
            acceptance=acceptance,
            rejected_output=rejected_output,
        )
        if self._memory is not None and task is not None:
            self._memory.record(task, graph, result)
        return result

    def _reverified(
        self, record: DeliveryAcceptance, output: Any, graph: ExecutionGraph, task: Task | None,
    ) -> DeliveryAcceptance:
        """Check a completed run's decision against what its checkpoint holds now.

        The saved output must still match the record's digest, the task and
        source generations it binds must be unchanged, every file it binds
        must still hold the recorded bytes, and a strict policy on this
        Swarm must be the one that decided. A mismatch gives an
        ``indeterminate`` record for the saved output, never acceptance.
        """
        if not isinstance(output, str):
            raise ValueError("Checkpoint output does not match its delivery acceptance record")
        problems = _delivery_problems(record, output, graph, task)
        policy = self._strict_policy()
        if policy is not None and policy.sha256 != record.policy_sha256:
            problems.append(
                "this Swarm's acceptance policy did not make the saved decision, which "
                "therefore does not carry over"
            )
        if not problems:
            return record
        return DeliveryAcceptance(
            decision=AcceptanceDecision.INDETERMINATE,
            policy_sha256=record.policy_sha256 if policy is None else policy.sha256,
            output_sha256=text_sha256(output), strategy=record.strategy,
            sources=record.sources, used_fallback=record.used_fallback,
            task_sha256=task_sha256(task), reasons=tuple(problems), artifacts=record.artifacts,
        )

    @staticmethod
    def _withheld(execution_id: str, output: Any) -> tuple[str, Any]:
        """A strict Swarm's slots for a completed run that has no decision.

        Nothing decided the saved output, so a strict policy does not deliver
        it: ``output`` is empty and the text is ``rejected_output``, with no
        record. The checkpoint is left as it was.
        """
        logger.warning(
            "Checkpoint %r completed with no delivery acceptance record, so this Swarm's "
            "strict policy withholds its output, which the result returns as "
            "rejected_output. To decide it, re-run the task under this policy, or apply "
            "smythe.acceptance.evaluate_delivery() to the saved output and graph.",
            execution_id,
        )
        return "", output

    def _check_resume_policy(
        self, execution_id: str, saved: DeliveryAcceptance | None, task: Task | None,
    ) -> None:
        """Refuse to finish a run under a delivery acceptance policy other than its own."""
        policy = self._strict_policy()
        if saved is None:
            if policy is not None:
                raise ValueError(
                    f"Checkpoint {execution_id!r} was written without strict delivery "
                    "acceptance, so this Swarm's strict policy cannot finish it. Resume it "
                    "without a strict policy, or re-run the task."
                )
            return
        if policy is None or policy.sha256 != saved.policy_sha256:
            raise ValueError(
                f"Checkpoint {execution_id!r} belongs to an unfinished run with strict delivery "
                f"acceptance under policy {saved.policy_sha256[:12]}; resume it with a Swarm "
                "whose AcceptancePolicy has that sha256, or re-run the task."
            )
        if saved.task_sha256 != task_sha256(task):
            raise ValueError(
                f"Checkpoint {execution_id!r} holds a task that its strict delivery acceptance "
                "record does not bind"
            )

    def _check_inherited_verdicts(self, graph: ExecutionGraph) -> None:
        """Refuse a strict run on a gate whose judge completed before it.

        A strict decision rests only on verdicts its own judges produced in
        the run it decides. A pending judge replaces any record it carries
        when it runs, but a completed judge does not run again: its saved
        final verdict (a consumed receipt, or ``advisory_verdict`` without
        regenerations) would decide a run that never read it, a pending
        receipt or a bound target generation would have the run read a reply
        written before it, and without either the gate could never decide.
        A resumed run keeps its own records and is not checked here.
        """
        if self._strict_policy() is None:
            return
        for node in graph.nodes:
            if node.verifies and node.status is NodeStatus.COMPLETED:
                raise ValueError(
                    f"Judge {node.id!r} completed before this run, so this run would not "
                    "produce its verdict, and a strict acceptance policy decides only on "
                    "verdicts produced in the run. Resume the run's checkpoint, or set the "
                    "judge back to pending so it judges its target in this run."
                )

    def plan(self, task: Task) -> ExecutionGraph:
        """Generate and assign an execution graph without running it.

        Returns the graph so you can inspect the architect's decisions
        before committing to execution. Under ``max_budget_usd``, a provider
        without the per-call ceiling it requires for the nodes' model does
        not stop planning: set each node's ``metadata["estimated_cost_usd"]``
        before ``execute(graph)``, which checks every node before its first
        call. An ``LLM_MERGE`` synthesis, which no node estimate covers, is
        still refused before planning.
        """
        if self._run_store is not None:
            return asyncio.run(self._workflow_runtime().plan(task))
        return self._plan(task, Tracer(), executing=False)

    def _plan_for_execution(self, task: Task, tracer: Tracer) -> ExecutionGraph:
        """Plan *task* for execute(); a subclass's plan() stays authoritative."""
        if type(self).plan is not Swarm.plan:
            # How the subclass plans is its own; its nodes are checked once
            # the plan exists.
            self._check_run_support()
            with tracer.phase("planning"):
                graph = self.plan(task)
            self._refuse_read_steps(graph)
            return graph
        return self._plan(task, tracer)

    async def _aplan_for_execution(self, task: Task, tracer: Tracer) -> ExecutionGraph:
        """Async _plan_for_execution(); a subclass's aplan() stays authoritative."""
        if type(self).aplan is not Swarm.aplan:
            self._check_run_support()
            with tracer.phase("planning"):
                graph = await self.aplan(task)
            self._refuse_read_steps(graph)
            return graph
        return await self._aplan(task, tracer)

    def _check_run_support(self) -> None:
        """Reject a run its provider cannot serve, whatever its plan will contain.

        A tool runtime offers tools to every node, and an ``LLM_MERGE``
        synthesis calls the run's model, so both are known before any plan.
        """
        if self._tool_runtime is not None:
            check_provider_support(self._provider, str(self.model), node_id=None,
                                   check_model=False, tools=True)
        self._check_synthesis_support(self.max_budget_usd)

    def _check_synthesis_support(self, max_budget_usd: float | None) -> None:
        # A synthesizer that is not a Synthesizer has no merge to check.
        check = getattr(self._synthesizer, "check_merge_support", None)
        if check is not None:
            check(provider=self._provider, model=self.model, max_budget_usd=max_budget_usd)

    def _check_planning_support(self, *, executing: bool) -> None:
        """Reject, before routing or planning, a run its provider cannot serve.

        A model-generated plan cannot set a node's model or cost estimate, so
        when every plan comes from ``LLMArchitect`` the run's model is every
        node's model and is checked now. Other architects build nodes in code
        and may set both, so their nodes are checked once the plan exists.

        The run's missing cost ceiling is refused only when *executing*: a
        caller of plan() can still set each returned node's estimate, which
        the node check before the first call accepts in its place. So is a
        strict policy without final validators when every architect that
        could plan generates its gates, which that policy does not count.
        """
        self._check_run_support()
        architects = _planning_architects(self._router, self._architect)
        if architects and all(type(item) is LLMArchitect for item in architects):
            check_provider_support(self._provider, str(self.model), node_id=None,
                                   max_budget_usd=self.max_budget_usd if executing else None)
        if executing:
            # The plan's gates will not count, so nothing could accept its output.
            _check_gate_planning(self._strict_policy(), architects)

    def _check_obligations(self, graph: ExecutionGraph) -> None:
        """Refuse, before any node runs, a graph whose output a strict policy could not decide."""
        policy = self._strict_policy()
        if policy is not None:
            validate_acceptance_obligations(
                graph, policy, strategy=_synthesis_strategy(self._synthesizer),
            )

    def _check_dispatch_support(self, graph: ExecutionGraph, budget: Sentinel) -> None:
        """Reject, before any node runs, nodes or a synthesis the provider cannot serve."""
        check_pending_nodes(self._provider, graph, max_budget_usd=budget.max_budget_usd,
                            tools=self._tool_runtime is not None)
        self._check_synthesis_support(budget.max_budget_usd)

    def _plan(self, task: Task, tracer: Tracer, *, executing: bool = True) -> ExecutionGraph:
        """Plan *task*, measuring routing and planning on *tracer*.

        *executing* is False for plan(), whose caller sees the graph first.
        """
        Sentinel(self.max_budget_usd)
        self._check_planning_support(executing=executing)
        task = snapshot_task(task)
        started = tracer.now()
        architect = self._select_architect(snapshot_task(task))
        if self._router is not None:
            tracer.add_phase("routing", tracer.since(started))
        with tracer.phase("planning"):
            graph, architect_registry = architect.plan(snapshot_task(task))
            return self._adopt_plan(graph, architect_registry, task, architect)

    async def aplan(self, task: Task) -> ExecutionGraph:
        """Async variant of plan() — safe to call from a running event loop."""
        if self._run_store is not None:
            return await self._workflow_runtime().plan(task)
        return await self._aplan(task, Tracer(), executing=False)

    async def _aplan(self, task: Task, tracer: Tracer, *, executing: bool = True) -> ExecutionGraph:
        """Async _plan()."""
        Sentinel(self.max_budget_usd)
        self._check_planning_support(executing=executing)
        task = snapshot_task(task)
        started = tracer.now()
        architect = await self._aselect_architect(snapshot_task(task))
        if self._router is not None:
            tracer.add_phase("routing", tracer.since(started))
        with tracer.phase("planning"):
            graph, architect_registry = await architect.aplan(snapshot_task(task))
            return self._adopt_plan(graph, architect_registry, task, architect)

    def _adopt_plan(
        self, graph: ExecutionGraph, architect_registry: Registry, task: Task, architect: Architect,
    ) -> ExecutionGraph:
        """Register the plan's agents and stamp its nodes for execution.

        Only ordinary runs plan here, so a plan with read steps, which an
        architect or template can build in code, is refused before any node
        call. Under a strict policy, each gate of a plan that *architect*
        generated rather than built from developer code is marked with
        ``GENERATED_GATE_KEY``, so that gate does not accept output on its own.
        """
        self._refuse_read_steps(graph)
        for agent in architect_registry.list_agents():
            self._registry.register(agent)

        graph = self._registry.assign(graph)
        graph.task = task
        self._stamp_model(graph)
        self._stamp_task_context(graph, task)
        if self._strict_policy() is not None:
            _mark_generated_gates(graph, architect)
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
                task_or_graph, max_concurrency=self._execution_concurrency(),
            ))
        if self.parallel:
            return asyncio.run(self.execute_async(task_or_graph))
        return self._execute_sync(task_or_graph)

    def _execution_concurrency(self) -> int | None:
        """The node concurrency for every execute and resume path.

        ``parallel=False`` means one node at a time wherever the graph runs,
        including the async executor behind ``execute_async()`` and resume.
        """
        return self.max_concurrency if self.parallel else 1

    async def execute_async(
        self, task_or_graph: Task | ExecutionGraph | None = None,
    ) -> SwarmResult:
        """Async execution on the AsyncExecutor, safe inside a running loop.

        Admits up to ``max_concurrency`` ready nodes at once when
        ``parallel=True``, and one node at a time otherwise.
        """
        task_or_graph = self._resolve_input(task_or_graph)
        if self._run_store is not None:
            return await self._workflow_runtime().execute(
                task_or_graph, max_concurrency=self._execution_concurrency(),
            )
        from smythe.async_executor import AsyncExecutor

        tracer = self._tracer_factory()
        timing = _RunTiming(tracer)
        budget = Sentinel(self.max_budget_usd)

        self._active_executor = None
        if isinstance(task_or_graph, Task):
            graph = await self._aplan_for_execution(task_or_graph, tracer)
        else:
            graph = self._prepare_graph(task_or_graph)
        self._check_obligations(graph)
        task = snapshot_task(graph.task) if graph.task is not None else None

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at, timing,
        )

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self._execution_concurrency(),
            tool_runtime=self._tool_runtime,
            max_tool_iterations=self.max_tool_iterations,
            artifact_dir=self._run_artifact_dir(execution_id),
            retry_backoff_s=self.retry_backoff_s,
            supervisor=self.supervisor,
            max_revisions=self.max_revisions,
            task=task,
            verifier=self.verifier,
            on_node_update=self._checkpointer(
                execution_id, graph, budget, task, created_at, timing,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at, timing,
            ),
            acceptance=self._acceptance,
        )
        self._active_executor = executor
        try:
            self._check_dispatch_support(graph, budget)
            graph = await executor.run(graph)
            with tracer.phase("synthesis"):
                output = await self._synthesizer.asynthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        return self._finish(
            execution_id, graph, budget, task, created_at, timing, output, acceptance,
        )

    def _execute_sync(self, task_or_graph: Task | ExecutionGraph) -> SwarmResult:
        """Serial execution using the standard Executor."""
        tracer = self._tracer_factory()
        timing = _RunTiming(tracer)
        budget = Sentinel(self.max_budget_usd)

        self._active_executor = None
        if isinstance(task_or_graph, Task):
            graph = self._plan_for_execution(task_or_graph, tracer)
        else:
            graph = self._prepare_graph(task_or_graph)
        self._check_obligations(graph)
        task = snapshot_task(graph.task) if graph.task is not None else None

        execution_id = uuid4().hex
        created_at = time.time()
        self._save_checkpoint(
            execution_id, "running", graph, budget, task, created_at, timing,
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
                execution_id, graph, budget, task, created_at, timing,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at, timing,
            ),
            acceptance=self._acceptance,
        )
        self._active_executor = executor
        try:
            self._check_dispatch_support(graph, budget)
            graph = executor.run(graph)
            with tracer.phase("synthesis"):
                output = self._synthesizer.synthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        return self._finish(
            execution_id, graph, budget, task, created_at, timing, output, acceptance,
        )

    def _save_checkpoint(
        self,
        execution_id: str,
        status: str,
        graph: ExecutionGraph,
        budget: Sentinel,
        task: Task | None,
        created_at: float,
        timing: _RunTiming,
        output: str | None = None,
        accounting_error: str | None = None,
        response_error: dict | None = None,
        acceptance: DeliveryAcceptance | None = None,
    ) -> None:
        """Persist full execution state, if a checkpoint store is configured.

        A strict run's states are version 5: they carry *acceptance*, the
        decision, or until there is one an undecided record that binds the
        run's policy and task.
        """
        if self._checkpoint_store is None:
            return
        with timing.tracer.phase("checkpoint"):
            if acceptance is None:
                acceptance = self._pending_acceptance(task)
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
                timing=timing.checkpoint_block(),
                token_estimate_nodes=budget.token_estimate_nodes,
                delivery_acceptance=acceptance,
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
        timing: _RunTiming,
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
                execution_id, "running", graph, budget, task, created_at, timing,
            )

        return _on_node_update

    def _control_checkpointer(self, execution_id, graph, budget, task, created_at, timing):
        """Control transitions must be durable even with batched node saves."""
        if self._checkpoint_store is None:
            return None

        def _on_control_update() -> None:
            self._save_checkpoint(
                execution_id, "running", graph, budget, task, created_at, timing,
            )

        return _on_control_update

    def refresh_context(self, execution_id: str, sources: Sequence[ContextSource]) -> int:
        """Give an unfinished durable run new sources.  Sync wrapper around arefresh_context()."""
        return asyncio.run(self.arefresh_context(execution_id, sources))

    async def arefresh_context(self, execution_id: str, sources: Sequence[ContextSource]) -> int:
        """Record *sources* as a durable run's next source revision; returns that revision.

        A run with context assembly keeps the sources it recorded, so sources
        that changed while it was paused reach it only through this call.
        Every node of the run is pending again at its next generation, and
        :meth:`resume` re-evaluates each with a new context snapshot; nothing
        is called here. See ``smythe.workflow.WorkflowRuntime.refresh_context``.
        """
        if self._run_store is None:
            raise ValueError("refresh_context needs a durable run_store")
        return await self._workflow_runtime().refresh_context(execution_id, sources)

    def _load_checkpoint(self, execution_id: str) -> dict[str, Any]:
        """The saved state of *execution_id*, at a checkpoint version this build reads."""
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
        return state

    def _restore_checkpoint(
        self, state: dict[str, Any],
    ) -> tuple[DeliveryAcceptance | None, ExecutionGraph, Task | None, Sentinel]:
        """A saved state's delivery acceptance record, graph, task and restored costs.

        Refuses a durable run's graph, conflicting task copies and accounting
        that resume cannot continue: invalid usage, an accounting error or an
        unresolved native response failure.
        """
        acceptance = delivery_acceptance_from_state(state)

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
        # Checkpoints written before this list existed name no token estimates.
        token_estimate_nodes = budget_state.get("token_estimate_nodes", [])
        if not isinstance(token_estimate_nodes, list) or not all(
            isinstance(node_id, str) for node_id in token_estimate_nodes
        ):
            raise BudgetValidationError("Checkpoint token_estimate_nodes must be a list of node IDs")
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
            token_estimate_nodes=set(token_estimate_nodes),
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
        return acceptance, graph, task, budget

    def reevaluate(
        self, execution_id: str, *, acceptance: AcceptancePolicy | None = None,
    ) -> SwarmResult:
        """Decide a completed run's saved output again.  Sync wrapper around areevaluate()."""
        return asyncio.run(self.areevaluate(execution_id, acceptance=acceptance))

    async def areevaluate(
        self, execution_id: str, *, acceptance: AcceptancePolicy | None = None,
    ) -> SwarmResult:
        """Record a new delivery acceptance decision for a completed run's saved output.

        *acceptance*, by default this Swarm's policy, must be strict. Its final
        validators run on the saved output, with no provider call, and every
        gate's recorded final verdict is read, as when the run completed. No
        node runs, and the saved costs, budget, retry and regeneration
        counters and node generations are left as they are. Only a completed
        run whose checkpoint holds a decided record can be re-evaluated: the
        saved output must still match the digest that record binds, and its
        task, source generations and delivered files must be unchanged;
        otherwise ValueError is raised and nothing is written, so changed
        output never becomes accepted. A run without a record has nothing that
        binds its output: run the task again under a strict policy.

        The new decision becomes the checkpoint's record, and the record it
        replaces is kept, with its sequence, in the checkpoint's bounded
        ``delivery_acceptance_history``. Identical bytes under an unchanged
        policy still get a new decision. Resuming the run returns the current
        decision, checked as usual: under a strict policy other than the one
        that made it, ``indeterminate``. :meth:`resume` never makes a new
        decision for a completed run; this method is the way to one.

        A durable ``run_store`` run journals the decision as a new
        ``acceptance`` operation and checkpoint; its recipe may differ from
        this Swarm's only in its acceptance policy and gate verifier. See
        ``smythe.workflow.WorkflowRuntime.reevaluate``.
        """
        policy = self._acceptance if acceptance is None else acceptance
        if policy is not None and not isinstance(policy, AcceptancePolicy):
            raise TypeError(
                f"acceptance must be an AcceptancePolicy or None, not {type(policy).__name__}"
            )
        if policy is None or policy.mode is not AcceptanceMode.STRICT:
            raise ValueError(
                "reevaluate needs a strict AcceptancePolicy: pass acceptance=, or build the "
                "Swarm with one"
            )
        if self._run_store is not None:
            return await self._workflow_runtime(acceptance=policy).reevaluate(execution_id)
        if self._checkpoint_store is None:
            raise ValueError("Cannot re-evaluate: this Swarm has no checkpoint_store configured.")
        tracer = self._tracer_factory()
        started = tracer.now()
        state = self._load_checkpoint(execution_id)
        saved = deepcopy(state)
        record, graph, task, budget = self._restore_checkpoint(state)
        output = state.get("output")
        completed = state.get("status") == "completed" and output is not None
        validate_verification_checkpoint(graph, version=state["version"], completed=completed)
        if not completed or verification_pending(graph):
            raise ValueError(
                f"Checkpoint {execution_id!r} is not a completed run, so it has no output to "
                "re-evaluate; resume it to finish it."
            )
        if record is None:
            raise ValueError(
                f"Checkpoint {execution_id!r} has no delivery acceptance record, so nothing "
                "bound its output when it completed; run the task again under a strict policy."
            )
        # A malformed history is refused before anything is decided.
        delivery_acceptance_history(state)
        sources, used_fallback = _delivery_sources(graph, record.strategy)
        problems = _delivery_problems(record, output, graph, task)
        if ([AcceptanceSource(node.id, node_generation(node)) for node in sources]
                != list(record.sources) or used_fallback != record.used_fallback):
            problems.append("the saved graph no longer delivers the sources its record bound")
        if problems:
            raise ValueError(
                f"Checkpoint {execution_id!r} cannot be re-evaluated: " + "; ".join(problems)
            )
        timing = _RunTiming(tracer, resumed=True, previous=timing_segments(state))
        tracer.add_phase("recovery", tracer.since(started))
        with tracer.phase("verification"):
            decided = evaluate_delivery(
                policy, graph, output, strategy=record.strategy, sources=sources,
                used_fallback=used_fallback, task=task,
            )
        tracer.on_acceptance(decided)
        updated = supersede_delivery_acceptance(saved, decided)
        with tracer.phase("checkpoint"):
            updated["timing"] = timing.checkpoint_block()
            self._checkpoint_store.save(execution_id, updated)
        delivered, rejected_output = _delivered(output, decided)
        return SwarmResult(
            output=delivered,
            graph=graph,
            trace=tracer.summary(),
            total_cost_usd=budget.total_cost_usd,
            cost_is_complete=budget.cost_is_complete,
            cost_contains_estimates=budget.cost_contains_estimates,
            execution_id=execution_id,
            timing=timing.result(),
            acceptance=decided,
            rejected_output=rejected_output,
        )

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
        re-executing anything; a strict run's result carries its current
        delivery acceptance record, the latest that :meth:`reevaluate` made if
        any, after the saved output, task, sources and policy are checked
        against it, and any mismatch makes it ``indeterminate``.
        A strict policy withholds the output of a completed run that has no
        record, returning it as ``rejected_output`` with ``acceptance=None``.
        An unfinished strict run resumes only under the same strict policy,
        and a strict policy never finishes a run that began without one.
        Resumed work runs one node at a time
        unless this Swarm has ``parallel=True``; a durable ``run_store``
        run keeps the concurrency its first execution recorded and resumes
        only under the acceptance policy its recipe records.
        """
        if self._run_store is not None:
            return await self._workflow_runtime().resume(
                execution_id, max_concurrency=self._execution_concurrency(),
            )
        if self._checkpoint_store is None:
            raise ValueError(
                "Cannot resume: this Swarm has no checkpoint_store configured."
            )
        tracer = self._tracer_factory()
        recovery_started = tracer.now()
        state = self._load_checkpoint(execution_id)
        version = state["version"]
        acceptance, graph, task, budget = self._restore_checkpoint(state)

        completed = state.get("status") == "completed" and state.get("output") is not None
        validate_verification_checkpoint(graph, version=version, completed=completed)
        timing = _RunTiming(tracer, resumed=True, previous=timing_segments(state))
        if completed and not verification_pending(graph):
            if acceptance is not None:
                acceptance = self._reverified(acceptance, state["output"], graph, task)
            output, rejected_output = _delivered(state["output"], acceptance)
            if acceptance is None and self._strict_policy() is not None:
                output, rejected_output = self._withheld(execution_id, state["output"])
            tracer.add_phase("recovery", tracer.since(recovery_started))
            return SwarmResult(
                output=output,
                graph=graph,
                trace=[],
                total_cost_usd=budget.total_cost_usd,
                cost_is_complete=budget.cost_is_complete,
                cost_contains_estimates=budget.cost_contains_estimates,
                execution_id=execution_id,
                timing=timing.result(),
                acceptance=acceptance,
                rejected_output=rejected_output,
            )
        # A checkpoint saved before ordinary runs refused read steps may hold one.
        self._refuse_read_steps(graph)
        # Finishing a run under another policy would decide its output
        # differently, so only the policy it began with may finish it.
        self._check_resume_policy(execution_id, acceptance, task)

        from smythe.async_executor import AsyncExecutor

        restored = agents_from_list(state.get("agents", []))
        dropped = drop_untrusted_mcp_servers(restored, version=version)
        if dropped:
            logger.warning(
                "Checkpoint %r predates smythe 0.8.1, when a model-generated plan could "
                "declare MCP servers; resuming without %s. Re-run the task to use them.",
                execution_id, dropped,
            )
        for agent in restored:
            self._registry.register(agent)
        reset_incomplete_nodes(graph)
        graph.validate()
        self._stamp_model(graph)

        created_at = state.get("created_at", time.time())

        executor = AsyncExecutor(
            provider=self._provider, registry=self._registry, tracer=tracer,
            budget=budget, max_concurrency=self._execution_concurrency(),
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
                execution_id, graph, budget, task, created_at, timing,
            ),
            on_control_update=self._control_checkpointer(
                execution_id, graph, budget, task, created_at, timing,
            ),
            acceptance=self._acceptance,
        )
        self._active_executor = executor
        tracer.add_phase("recovery", tracer.since(recovery_started))
        try:
            self._check_dispatch_support(graph, budget)
            graph = await executor.run(graph)
            with tracer.phase("synthesis"):
                output = await self._synthesizer.asynthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise

        return self._finish(
            execution_id, graph, budget, task, created_at, timing, output, acceptance,
        )

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
        model: str = "claude-opus-5-5",
        max_budget_usd: float | None = None,
        provider: Provider | None = None,
        parallel: bool = False,
        run_store: SQLiteWorkflowStore | None = None,
        graph_policy: WorkflowGraphPolicy | None = None,
        acceptance: AcceptancePolicy | None = None,
        read_tools: tuple[ReadProfile, ...] | list[ReadProfile] = (),
    ) -> Swarm:
        """Create a Swarm pre-loaded with a YAML-defined execution graph.

        The returned Swarm's internal registry is populated with agents
        defined in the YAML file.  Call execute() with the loaded graph
        to run it.  A delivery acceptance policy comes only from the
        ``acceptance`` argument, never from the file, and so do the read
        profiles that the file's read steps name.
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
            acceptance=acceptance,
            read_tools=read_tools,
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

    @staticmethod
    def _refuse_read_steps(graph: ExecutionGraph) -> None:
        """Refuse, before any node call, an ordinary run of a graph with read steps."""
        read_steps = [node.id for node in graph.nodes if READ_STEP_KEY in node.metadata]
        if read_steps:
            raise ValueError(
                f"Nodes {read_steps} declare read steps, which run only in a durable workflow: "
                "pass run_store and read_tools to Swarm"
            )

    def _prepare_graph(self, graph: ExecutionGraph) -> ExecutionGraph:
        """Ready a caller-built graph for execution.

        Graphs built by hand skip plan()/from_yaml(), so nodes have no
        model stamped yet; providers reject an empty model name.  One
        shared helper so no execute path can forget it again.
        """
        if graph.run_ref is not None:
            raise ValueError("This graph requires its matching durable run_store")
        graph.validate()
        self._refuse_read_steps(graph)
        self._check_inherited_verdicts(graph)
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

    def _workflow_runtime(self, *, behavior_settings=None, acceptance=None):
        """Pass explicit component dependencies into an isolated durable run.

        ``behavior_settings`` configures behavior versions that a new run
        records (see ``smythe.workflow.RECIPE_BEHAVIOR_FIELDS``), and
        ``acceptance`` a policy other than this Swarm's.
        """
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
            graph_policy=self._graph_policy, behavior_settings=behavior_settings,
            acceptance=self._acceptance if acceptance is None else acceptance,
            read_tools=self._read_tools,
            context_sources=self._context_sources,
        )
