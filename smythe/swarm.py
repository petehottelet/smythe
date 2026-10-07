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

from smythe._failures import RunControl, SecurityStop
from smythe._run_inspection import result_view
from smythe.acceptance import (
    GENERATED_GATE_KEY,
    AcceptanceDecision,
    AcceptanceMode,
    AcceptancePolicy,
    AcceptanceSource,
    DeliveryAcceptance,
    OperatorDecision,
    _delivery_problems,
    _operator_record,
    evaluate_delivery,
    task_sha256,
    text_sha256,
    validate_acceptance_obligations,
)
from smythe.budget import BudgetValidationError, Sentinel, _validate_usd
from smythe.context import ContextSource, snapshot_sources
from smythe.evidence import SourceRecord
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    SUPPORTED_CHECKPOINT_VERSIONS,
    CheckpointStore,
    agents_from_list,
    attempt_allowance_from_state,
    build_state,
    delivery_acceptance_from_state,
    delivery_acceptance_history,
    delivery_acceptance_sequence,
    drop_untrusted_mcp_servers,
    graph_from_dict,
    graph_to_dict,
    model_profiles_from_state,
    reset_incomplete_nodes,
    revision_log_from_state,
    security_stop_from_state,
    supersede_delivery_acceptance,
    task_from_dict,
    timing_segments,
)
from smythe.executor import Executor
from smythe.executor_base import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    ExecutorBase,
    check_pending_nodes,
    check_profile_support,
    check_provider_support,
    check_supervisor_profiles,
    validate_max_concurrency,
    validate_max_revisions,
    validate_retry_backoff_s,
)
from smythe.graph import SYNTHESIS_NODE_ID, ExecutionGraph, NodeStatus
from smythe.lineage import LineagePolicy, snapshot_lineage_policy
from smythe.memory import PlannerMemory
from smythe.tools import ToolRuntime
from smythe.planner import (
    Architect, DeterministicArchitect, LLMArchitect, SimpleArchitect, _record_coordination,
)
from smythe.constrained_planner import ConstrainedArchitect
from smythe.router import RoutingDecision, WhiteRabbit
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
from smythe.reuse import RepairPlan, RepairResult, SourceCatalog, _journal_work, repair_plan
from smythe.scheduling import (
    ModelProfile, RetryPolicy, node_profile, profile_map,
)
from smythe.scheduling import CapacityPolicy, _capacity_limit
from smythe.scheduling import TerminationPolicy
from smythe.supervisor import Supervisor, _model_reviewer
from smythe.synthesizer import SynthesisStrategy, Synthesizer, deliverable_sources
from smythe.task import Task, snapshot_task, task_snapshots_equal, task_to_dict
from smythe.tracer import Tracer
from smythe.verifier import (
    Verifier,
    node_generation,
    validate_verification_checkpoint,
    verification_pending,
)
from smythe.workflow_artifact import MarkdownRenderer, PublishedArtifact
from smythe.workflow_binding import LocalOnly, WorkflowBindingError, _BoundLocal
from smythe.workflow_policy import WorkflowGraphPolicy, snapshot_graph_policy
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowBudgetError, WorkflowConflictError
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


def _planning_calls(router: Any, architect: Any) -> list[tuple[Any, str]]:
    """The provider and model of every call routing or planning could make.

    A ``WhiteRabbit``'s classifier and the tiers it could return, or
    *architect* without a router. Of the built-in architects only
    ``LLMArchitect`` and ``ConstrainedArchitect`` call a provider.
    """
    calls: list[tuple[Any, str]] = []
    architects: tuple[Any, ...] = (architect,) if router is None else ()
    if isinstance(router, WhiteRabbit):
        if router._classifier_provider is not None:
            calls.append((router._classifier_provider, router._classifier_model))
        architects = router._architects()
    for item in architects:
        if isinstance(item, LLMArchitect):
            calls.append((item._provider, item._planning_model))
        elif isinstance(item, ConstrainedArchitect):
            calls.append((item._provider, item._model))
    return calls


def _records_routing(router: Any) -> bool:
    """Whether *router* routes as WhiteRabbit does, so its decision can be recorded.

    Any other router, including a subclass that overrides route() or
    aroute(), is called as before and records no decision.
    """
    return (isinstance(router, WhiteRabbit) and type(router).route is WhiteRabbit.route
            and type(router).aroute is WhiteRabbit.aroute)


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
    validator or, for output left for review, an authorized operator's,
    without running a node.

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

    ``renderer``, which needs a durable ``run_store``, publishes each run's
    delivered output as a local document: a
    :class:`~smythe.workflow_artifact.MarkdownRenderer`, or a deterministic
    renderer declared with ``LocalOnly(..., role="renderer")``, renders the
    synthesized text after synthesis, and the document is published once
    into ``artifact_dir/<run id>`` without replacing any file. The document
    is the run's output, so a strict policy decides on its exact bytes.
    :meth:`republish` publishes it again when its file is missing or changed.

    ``retry_policy``, a :class:`~smythe.scheduling.RetryPolicy`, makes the
    executor the only retry owner of an ordinary run's node calls: only the
    failure classes it lists are retried, waits follow ``Retry-After`` up to
    a cap, and a deadline bounds each node's attempts and waits. Every
    provider the run can call must declare ``sdk_retries=0``: its own, a
    ``WhiteRabbit`` classifier's, each ``LLMArchitect`` or
    ``ConstrainedArchitect`` that could plan it, and an ``LLMSupervisor``'s
    when ``max_revisions`` allows a review. Otherwise the run is refused
    before routing, planning or any node call. A policy with
    ``persist_attempts`` is saved with the run's checkpoints, with each
    node's attempts, so resume continues the allowance instead of refilling it.
    A durable run records the policy in its recipe and counts each node's
    attempts from its journal, so its resume continues them too.

    ``termination``, a :class:`~smythe.scheduling.TerminationPolicy`, has the
    async executor drain the nodes in flight when a failure stops the run:
    no queued node starts and no retry is sent, the nodes in flight may finish
    their current attempts, tool turns included, and save their results
    until ``drain_timeout_s``, and the rest are cancelled before the
    original error is raised. The failed checkpoint
    records how the drain ended under ``control["drain"]``, for information
    only. Serial ``execute()`` has nothing in flight to drain. A durable run
    records the policy in its recipe and sends no call during the drain: a
    call already sent and answered within it keeps its known charge, one
    still unanswered at the deadline keeps unknown exposure, and resume keeps
    the run's policy.

    ``lineage``, a :class:`~smythe.lineage.LineagePolicy`, opts each durable
    run into artifact lineage under a declared access scope. It needs a
    ``run_store`` and a strict ``acceptance`` policy, and is refused without
    them when the Swarm is built. The run's recipe records it, so the run
    resumes only under the same scope and reuse mode.

    ``reserve_for_checks=True`` holds, under ``max_budget_usd`` and before
    any node runs, the estimate that each pending verifier and an
    ``LLM_MERGE`` synthesis will reserve (``Sentinel.hold``), so other nodes
    cannot spend that allowance: a node whose reservation would reach into
    it is refused with ``SentinelAlert`` before its call. Each verifier and the
    merge converts its own hold, a regeneration holds again for the
    verifiers it resets, and holds are never spend. Holds are not saved;
    resume holds again. Durable runs refuse it.

    ``profiles``, a sequence of :class:`~smythe.scheduling.ModelProfile`
    with distinct names, are the provider and model routes an ordinary run
    allows. A node whose ``metadata["profile"]`` names one calls its provider
    with its model and reserves its ceiling; developer graphs, architects,
    templates and supervisors can set the key, and a generated plan or an
    ``LLMSupervisor`` proposal cannot. Every other node and synthesis call the
    run's provider. Before routing or planning, and before any node call,
    each profile must fit: its provider declares ``True`` every feature it
    requires, supports its model, does not price the model as retired, and
    can call tools in a run with a ``tool_runtime``. While ``max_revisions``
    lets an ``LLMSupervisor`` without a model of its own review, every
    profile must call the supervisor's provider. A node that names a profile
    the run does not allow, or sets another model than its profile's, is
    refused as well, all with ``NoAllowedProfileError``. A revision that adds
    such a node is refused before it applies, like any invalid revision.
    Checkpoints of a run with profiled nodes are version 5, and resume routes
    them to the resuming Swarm's profiles of the same names. Durable runs
    refuse profiles.

    ``capacity_policy``, a :class:`~smythe.scheduling.CapacityPolicy`, limits
    each provider type and model's calls on an ordinary run: every node call,
    tool turn and retry, and an ``LLM_MERGE`` synthesis, waits for its turn
    (``min_interval_s`` and ``Retry-After`` cooldowns), and the async
    executor runs at most ``max_in_flight`` nodes of a model at once,
    passing over a ready node that has no free slot instead of stopping the
    run. Capacity waits are reported under ``timing["waiting_ms"]["capacity"]``.
    Nothing about it is saved. Durable runs refuse it.

    ``run_control``, a :class:`~smythe._failures.RunControl`, stops this
    Swarm's runs for security once it is revoked: ``plan``, ``execute`` and
    ``resume`` refuse to start, and a run in progress admits no node and
    sends no call after it, including supervision and synthesis. The async
    executor cancels the calls in flight at once, without a ``termination``
    drain, and a call that was sent and not answered keeps its reservation
    as unknown cost. The run raises ``SecurityStop``, as it does when a tool
    runtime raises one. An ordinary run's failed checkpoint records the stop
    under ``control["security_stop"]`` as version 5, and a durable run's
    journal blocks it with ``security_stop``; neither ever resumes.
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
        renderer: MarkdownRenderer | LocalOnly | None = None,
        retry_policy: RetryPolicy | None = None,
        lineage: LineagePolicy | None = None,
        reserve_for_checks: bool = False,
        profiles: Sequence[ModelProfile] = (),
        capacity_policy: CapacityPolicy | None = None,
        termination: TerminationPolicy | None = None,
        run_control: RunControl | None = None,
    ) -> None:
        Sentinel(max_budget_usd)  # Reject malformed policy before planning can call a provider.
        if run_control is not None and not isinstance(run_control, RunControl):
            raise TypeError(
                f"run_control must be a RunControl or None, not {type(run_control).__name__}"
            )
        self.run_control = run_control
        self._profiles = profile_map(profiles)
        if self._profiles and run_store is not None:
            raise WorkflowBindingError(
                "profiles route ordinary runs only; durable text workflows refuse them"
            )
        if type(reserve_for_checks) is not bool:
            raise TypeError(
                f"reserve_for_checks must be True or False, not {type(reserve_for_checks).__name__}"
            )
        self.reserve_for_checks = reserve_for_checks
        if capacity_policy is not None and not isinstance(capacity_policy, CapacityPolicy):
            raise TypeError(
                f"capacity_policy must be a CapacityPolicy or None, "
                f"not {type(capacity_policy).__name__}"
            )
        if capacity_policy is not None and run_store is not None:
            raise WorkflowBindingError(
                "capacity_policy is not supported by durable text workflows, whose journal "
                "admits each call when it is made"
            )
        self.capacity_policy = capacity_policy
        if renderer is not None and run_store is None:
            raise WorkflowBindingError("renderer requires a durable run_store")
        if renderer is not None and artifact_dir is None:
            raise WorkflowBindingError("renderer needs an artifact_dir to publish into")
        self._renderer = renderer
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError(
                f"retry_policy must be a RetryPolicy or None, not {type(retry_policy).__name__}"
            )
        self.retry_policy = retry_policy
        if termination is not None and not isinstance(termination, TerminationPolicy):
            raise TypeError(
                f"termination must be a TerminationPolicy or None, not {type(termination).__name__}"
            )
        self.termination = termination
        if graph_policy is not None and run_store is None:
            raise WorkflowBindingError("graph_policy requires a durable run_store")
        if lineage is not None and run_store is None:
            raise WorkflowBindingError("lineage requires a durable run_store")
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
        self._lineage = snapshot_lineage_policy(lineage)
        if self._lineage is not None and self._strict_policy() is None:
            raise WorkflowBindingError(
                "lineage requires a strict acceptance policy, whose decision its record binds"
            )
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

    def _check_resume_retry_policy(self, execution_id: str, allowance: tuple | None) -> None:
        """Refuse to finish a run whose saved attempt allowance this Swarm's policy would not keep."""
        policy = self.retry_policy
        if policy is not None and not policy.persist_attempts:
            policy = None
        saved = allowance[0] if allowance is not None else None
        if saved is None and policy is not None:
            raise ValueError(
                f"Checkpoint {execution_id!r} was written without persisted attempts, so this "
                "Swarm's retry policy cannot bound them. Resume it under a retry policy "
                "without persist_attempts, or re-run the task."
            )
        if saved != policy:
            raise ValueError(
                f"Checkpoint {execution_id!r} records attempts used under the retry policy "
                f"{saved.to_dict()}; resume it with a Swarm whose retry_policy equals it, or "
                "re-run the task."
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
        still refused before planning. With a ``WhiteRabbit`` router, the
        graph's ``planning["routing"]`` records the
        :class:`~smythe.router.RoutingDecision` that chose its architect, and
        ``planning["coordination"]`` any parallel siblings that
        :func:`~smythe.planner.lint_contributions` flags.
        """
        self._check_run_control()
        if self._run_store is not None:
            return asyncio.run(self._workflow_runtime().plan(task))
        return self._plan(task, Tracer(), executing=False)

    def _check_run_control(self) -> None:
        """Refuse to start or resume any run once this Swarm's run control is revoked."""
        if self.run_control is not None:
            self.run_control.check()

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
        So is a retry policy, which needs a provider whose SDK sends each
        call once, as does an ``LLMSupervisor``'s provider, and so is each
        model profile a node may name.
        """
        if self._tool_runtime is not None or self.retry_policy is not None:
            check_provider_support(self._provider, str(self.model), node_id=None,
                                   check_model=False, tools=self._tool_runtime is not None,
                                   retry_owner=self.retry_policy is not None)
        self._check_synthesis_support(self.max_budget_usd)
        self._check_retry_owners()
        self._admit_profiles()

    def _admit_profiles(self) -> None:
        """Refuse, with ``NoAllowedProfileError``, a profile that could not serve its nodes.

        While ``max_revisions`` lets an ``LLMSupervisor`` review, so is one it
        could not review its nodes with (see ``check_supervisor_profiles``).
        """
        for profile in self._profiles.values():
            check_profile_support(profile, tools=self._tool_runtime is not None)
        if self.max_revisions > 0:
            check_supervisor_profiles(self.supervisor, self._profiles)

    def _check_retry_owners(
        self, graph: ExecutionGraph | None = None, *, planning: bool = False,
    ) -> None:
        """Under a retry policy, refuse every call the run can make that an SDK may send again.

        Nodes and synthesis call the run's provider, and profiled nodes their
        profile's. An ``LLMSupervisor`` that ``max_revisions`` lets review,
        on its own or inside an ``EvidenceTriggeredSupervisor``,
        calls its own provider, with its model or else the reviewed node's:
        the run's model before a plan exists, then each of *graph*'s. With
        *planning*, so are the calls routing and planning could make (see
        ``_planning_calls``).
        """
        if self.retry_policy is None:
            return
        calls = [(self._provider, self.model)]
        calls += [(profile.provider, profile.model) for profile in self._profiles.values()]
        if planning:
            calls += _planning_calls(self._router, self._architect)
        supervisor = _model_reviewer(self.supervisor)
        if supervisor is not None and self.max_revisions > 0:
            if supervisor._model:
                models = {supervisor._model}
            elif graph is None:
                models = {self.model}
            else:
                models = {node.metadata.get("model", "") for node in graph.nodes}
            calls += [(supervisor._provider, model) for model in sorted(models, key=str)]
        for provider, model in calls:
            check_provider_support(provider, str(model), node_id=None, check_model=False,
                                   retry_owner=True)

    def _check_synthesis_support(self, max_budget_usd: float | None) -> None:
        # A synthesizer that is not a Synthesizer has no merge to check.
        check = getattr(self._synthesizer, "check_merge_support", None)
        if check is not None:
            check(provider=self._provider, model=self.model, max_budget_usd=max_budget_usd)
        if self.capacity_policy is not None:
            # The run's model, which a merge calls, has its capacity hint read now.
            _capacity_limit(self.capacity_policy, self._provider, str(self.model))

    def _merge_capacity(self, executor: ExecutorBase) -> dict[str, Any]:
        """The run's capacity limiter as synthesis arguments, for the built-in Synthesizer."""
        limiter = getattr(executor, "_capacity", None)
        if limiter is None or type(self._synthesizer) is not Synthesizer:
            return {}
        return {"capacity": limiter}

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
        self._check_retry_owners(planning=True)
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
        """Reject, before any node runs, nodes or a synthesis the provider cannot serve.

        Under a retry policy, synthesis and supervision must send each call
        once too, even when no node is left to run. Each model profile must
        fit, as before planning.
        """
        self._admit_profiles()
        check_pending_nodes(self._provider, graph, max_budget_usd=budget.max_budget_usd,
                            tools=self._tool_runtime is not None,
                            retry_owner=self.retry_policy is not None, profiles=self._profiles)
        self._check_synthesis_support(budget.max_budget_usd)
        self._check_retry_owners(graph)

    def _hold_check_budget(
        self, executor: ExecutorBase, graph: ExecutionGraph, budget: Sentinel,
    ) -> None:
        """With ``reserve_for_checks``, hold what pending verifiers and a merge will reserve."""
        if not self.reserve_for_checks or budget.max_budget_usd is None:
            return
        executor.hold_verifier_budget(graph)
        estimate = getattr(self._synthesizer, "merge_estimate_usd", None)
        amount = None if estimate is None else estimate(budget, provider=self._provider,
                                                        model=self.model)
        if amount is not None:
            budget.hold(SYNTHESIS_NODE_ID, amount)

    @staticmethod
    def _release_holds(budget: Sentinel) -> None:
        """Free allowance no call will use: after synthesis, or when a run fails."""
        for key in budget.holds:
            budget.release_hold(key)

    def _plan(self, task: Task, tracer: Tracer, *, executing: bool = True) -> ExecutionGraph:
        """Plan *task*, measuring routing and planning on *tracer*.

        *executing* is False for plan(), whose caller sees the graph first.
        """
        Sentinel(self.max_budget_usd)
        self._check_planning_support(executing=executing)
        task = snapshot_task(task)
        started = tracer.now()
        architect, routing = self._select_architect(snapshot_task(task))
        if self._router is not None:
            tracer.add_phase("routing", tracer.since(started))
        with tracer.phase("planning"):
            graph, architect_registry = architect.plan(snapshot_task(task))
            return self._adopt_plan(graph, architect_registry, task, architect, routing)

    async def aplan(self, task: Task) -> ExecutionGraph:
        """Async variant of plan() — safe to call from a running event loop."""
        self._check_run_control()
        if self._run_store is not None:
            return await self._workflow_runtime().plan(task)
        return await self._aplan(task, Tracer(), executing=False)

    async def _aplan(self, task: Task, tracer: Tracer, *, executing: bool = True) -> ExecutionGraph:
        """Async _plan()."""
        Sentinel(self.max_budget_usd)
        self._check_planning_support(executing=executing)
        task = snapshot_task(task)
        started = tracer.now()
        architect, routing = await self._aselect_architect(snapshot_task(task))
        if self._router is not None:
            tracer.add_phase("routing", tracer.since(started))
        with tracer.phase("planning"):
            graph, architect_registry = await architect.aplan(snapshot_task(task))
            return self._adopt_plan(graph, architect_registry, task, architect, routing)

    def _adopt_plan(
        self, graph: ExecutionGraph, architect_registry: Registry, task: Task, architect: Architect,
        routing: RoutingDecision | None = None,
    ) -> ExecutionGraph:
        """Register the plan's agents and stamp its nodes for execution.

        Only ordinary runs plan here, so a plan with read steps, which an
        architect or template can build in code, is refused before any node
        call. Under a strict policy, each gate of a plan that *architect*
        generated rather than built from developer code is marked with
        ``GENERATED_GATE_KEY``, so that gate does not accept output on its own.
        A *routing* decision becomes ``graph.planning["routing"]``, and the
        groups the contribution lint finds, if any, ``graph.planning["coordination"]``.
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
        if routing is not None:
            # The Swarm writes only this key; the architect may have set others.
            graph.planning = {**(graph.planning or {}), "routing": routing.to_dict()}
            # Agents are assigned, so the advisory lint can compare them.
            _record_coordination(graph, self._registry)
        return graph

    @staticmethod
    def _stamp_task_context(graph: ExecutionGraph, task: Task) -> None:
        """Refresh every node from the authoritative full Task snapshot."""
        ExecutorBase.stamp_task_context(graph.nodes, task)

    def _select_architect(self, task: Task) -> tuple[Architect, RoutingDecision | None]:
        """Pick the architect — use router if set, otherwise the default — and its decision."""
        if self._router is None:
            return self._architect, None
        if _records_routing(self._router):
            return self._router.decide(task)
        return self._router.route(task), None

    async def _aselect_architect(self, task: Task) -> tuple[Architect, RoutingDecision | None]:
        """Async architect selection — uses adecide or aroute when a router is set."""
        if self._router is None:
            return self._architect, None
        if _records_routing(self._router):
            return await self._router.adecide(task)
        return await self._router.aroute(task), None

    def execute(
        self, task_or_graph: Task | ExecutionGraph | None = None,
    ) -> SwarmResult:
        """Execute a task or a previously planned graph.

        Accepts either a Task (plans and executes in one call), an
        ExecutionGraph returned by plan(), or None when a YAML graph
        was loaded via from_yaml().
        If parallel=True, uses the async executor for concurrent nodes.
        """
        self._check_run_control()
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
        self._check_run_control()
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
            retry_policy=self.retry_policy,
            profiles=tuple(self._profiles.values()), default_model=self.model,
            capacity_policy=self.capacity_policy,
            termination=self.termination,
            run_control=self.run_control,
        )
        self._active_executor = executor
        try:
            self._check_dispatch_support(graph, budget)
            self._hold_check_budget(executor, graph, budget)
            graph = await executor.run(graph)
            executor.check_run_control()
            with tracer.phase("synthesis"):
                output = await self._synthesizer.asynthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                    **self._merge_capacity(executor),
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise
        finally:
            self._release_holds(budget)

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
            retry_policy=self.retry_policy,
            profiles=tuple(self._profiles.values()), default_model=self.model,
            capacity_policy=self.capacity_policy,
            run_control=self.run_control,
        )
        self._active_executor = executor
        try:
            self._check_dispatch_support(graph, budget)
            self._hold_check_budget(executor, graph, budget)
            graph = executor.run(graph)
            executor.check_run_control()
            with tracer.phase("synthesis"):
                output = self._synthesizer.synthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                    **self._merge_capacity(executor),
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise
        finally:
            self._release_holds(budget)

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
        run's policy and task. So are the states of a run whose retry policy
        persists attempts, which record it and each node's attempts used.
        A run's supervisor decisions, once there are any, add the bounded
        ``revision_log`` to ``control``, which keeps the version.
        A run that drained after a failure records how under ``drain``,
        which is informational and leaves the version unchanged. A run that a
        security stop ended records it under ``security_stop``, which makes
        the state version 5, so that no build resumes it.
        """
        if self._checkpoint_store is None:
            return
        with timing.tracer.phase("checkpoint"):
            if acceptance is None:
                acceptance = self._pending_acceptance(task)
            control = {"revisions_used": getattr(self._active_executor, "revisions_used", 0)}
            if self.retry_policy is not None and self.retry_policy.persist_attempts:
                control["retry_policy"] = self.retry_policy.to_dict()
                control["attempts_used"] = getattr(self._active_executor, "attempts_used", {})
            revision_log = getattr(self._active_executor, "revision_log", None)
            if revision_log:
                control["revision_log"] = revision_log
            drain = getattr(self._active_executor, "drain_record", None)
            if drain is not None:
                control["drain"] = drain
            stop = getattr(self._active_executor, "security_stop", None)
            if stop is not None:
                control["security_stop"] = stop
            state = build_state(
                control=control,
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

    def republish(self, execution_id: str) -> PublishedArtifact:
        """Publish a completed durable run's document again.  Sync wrapper around arepublish()."""
        return asyncio.run(self.arepublish(execution_id))

    async def arepublish(self, execution_id: str) -> PublishedArtifact:
        """Publish a completed run's document again when its file is missing or changed.

        The saved document's bytes go to a new file in the run's directory,
        which becomes the run's current publication; no file is replaced, no
        provider is called, and the delivery decision, which binds the same
        bytes, is unchanged. Returns the current record. A run with a lineage
        policy instead renders its saved synthesis again into a file named for
        the new publication, decides the new bytes again under this Swarm's
        policy and supersedes its lineage record; an intact document is then
        refused, as there is nothing to repair. See
        ``smythe.workflow.WorkflowRuntime.republish``.
        """
        if self._run_store is None:
            raise ValueError("republish needs a durable run_store")
        return await self._workflow_runtime().republish(execution_id)

    async def _arerender(self, execution_id: str, *, plan_sha256: str,
                         reason: str) -> PublishedArtifact:
        """Apply a repair plan's ``rerender`` item: republish a lineage run, recording the plan."""
        return await self._workflow_runtime().republish(
            execution_id, plan_sha256=plan_sha256, reason=reason)

    async def _arecheck(self, execution_id: str, *, plan_sha256: str, reason: str) -> SwarmResult:
        """Apply a repair plan's ``recheck`` item: decide a lineage run again under this Swarm's policy."""
        return await self._workflow_runtime().reevaluate(
            execution_id, plan_sha256=plan_sha256, reason=reason)

    def remove_source(self, execution_id: str, source_id: str, reason: str, *,
                      removed_at: str | None = None) -> SourceRecord:
        """Remove a durable run's source.  Sync wrapper around aremove_source()."""
        return asyncio.run(self.aremove_source(execution_id, source_id, reason,
                                               removed_at=removed_at))

    async def aremove_source(self, execution_id: str, source_id: str, reason: str, *,
                             removed_at: str | None = None) -> SourceRecord:
        """Remove one of a durable run's current sources; return its tombstone.

        The tombstone keeps the source's id, version and digest, why and when
        it was removed, and none of its text, which the run never uses or
        returns again: an unfinished run needs :meth:`refresh_context` before
        it resumes, and a completed run's earlier decision becomes
        ``indeterminate`` until :meth:`reevaluate` decides again. Nothing is
        called. See ``smythe.workflow.WorkflowRuntime.remove_source``.
        """
        if self._run_store is None:
            raise ValueError("remove_source needs a durable run_store")
        return await self._workflow_runtime().remove_source(
            execution_id, source_id, reason, removed_at=removed_at)

    def plan_repair(self, execution_ids: Sequence[str] | None = None, *,
                    sources: SourceCatalog) -> RepairPlan:
        """Plan what durable runs' accepted artifacts call for now, changing nothing.

        Each run in this Swarm's lineage access scope, by default each head
        of the archive (:func:`~smythe.reuse.archive_heads`), is evaluated
        (:func:`~smythe.reuse.evaluate`) against what a new run of this
        Swarm would use: its strict policy's validators, its read profiles,
        synthesizer, context assembly and agents, and the sources that
        *sources*, a zero-cost :class:`~smythe.reuse.SourceCatalog`,
        describes. The journal is opened again read-only, so no lease is
        taken, nothing is written, no provider is called and no source is
        read. Returns a :class:`~smythe.reuse.RepairPlan`; its
        ``plan_sha256`` is a staleness guard, not an authorization. See
        ``smythe.reuse.repair_plan``.
        """
        if self._run_store is None:
            raise ValueError("plan_repair needs a durable run_store")
        if self._lineage is None:
            raise ValueError("plan_repair needs a lineage policy, whose access scope it plans in")
        recipe = self._workflow_runtime().recipe
        with SQLiteWorkflowStore(self._run_store.path, read_only=True) as store:
            return repair_plan(store, recipe, sources, execution_ids)

    def repair(self, plan: RepairPlan, *, plan_sha256: str, max_total_budget_usd: float,
               reason: str, sources: SourceCatalog) -> RepairResult:
        """Apply a repair plan.  Sync wrapper around arepair()."""
        return asyncio.run(self.arepair(plan, plan_sha256=plan_sha256, reason=reason,
                                        max_total_budget_usd=max_total_budget_usd,
                                        sources=sources))

    async def arepair(self, plan: RepairPlan, *, plan_sha256: str, max_total_budget_usd: float,
                      reason: str, sources: SourceCatalog) -> RepairResult:
        """Apply a repair plan's items in order, each once, under an explicit allowance.

        *plan* comes from :meth:`plan_repair`, approved by *plan_sha256*. It
        is made again from *sources* for the same runs and must still have
        that digest, or ``WorkflowConflictError`` is raised. Repair needs a
        ``run_store`` and a lineage policy (``ValueError``) and refuses a
        supervisor (``WorkflowBindingError``). The ``regenerate`` items' new
        runs, at ``max_budget_usd`` each, must fit *max_total_budget_usd*,
        and an unbounded ``max_budget_usd`` is refused when any may incur
        charges (``WorkflowBudgetError``). Refusals precede any write or call.

        ``recheck`` decides the run again and ``rerender`` publishes its
        document again, binding the digest and *reason*; ``regenerate``
        starts a successor run that keeps the nodes marked ``reuse``, leaving
        the repaired run as it is; ``reuse`` and ``review`` change nothing.
        The first item that raises or is not ``accepted`` stops the repair,
        unretried. A successor left unfinished is a repair in progress that
        :meth:`resume` finishes. Returns a :class:`~smythe.reuse.RepairResult`;
        see the lineage guide.
        """
        from smythe.workflow import _nanousd, _repair_record

        if self._run_store is None:
            raise ValueError("repair needs a durable run_store; ordinary runs have no lineage")
        if self._lineage is None:
            raise ValueError("repair needs a lineage policy, whose access scope it repairs in")
        if self.supervisor is not None:
            raise WorkflowBindingError(
                "repair refuses a supervisor, which could change the graph a successor reuses")
        if not isinstance(plan, RepairPlan):
            raise TypeError(f"plan must be a RepairPlan, not {type(plan).__name__}")
        _repair_record(plan_sha256, reason)
        if plan.plan_sha256 != plan_sha256:
            raise ValueError("plan_sha256 is not the digest of the plan given")
        allowance = _nanousd(_validate_usd(max_total_budget_usd, "max_total_budget_usd"))
        self._check_run_control()
        items = plan.to_dict()["items"]
        if self.plan_repair([item["run_id"] for item in items],
                            sources=sources).plan_sha256 != plan_sha256:
            raise WorkflowConflictError(
                "The repair plan is stale: planned again now, it calls for other work")
        new = [item["work"] for item in items if item["action"] == "regenerate"]
        if self.max_budget_usd is None and any(work["may_incur_charges"] for work in new):
            raise WorkflowBudgetError(
                "A repair whose new runs may incur charges needs max_budget_usd for each run")
        runs = sum(work["new_runs"] for work in new)
        if self.max_budget_usd is not None and runs * _nanousd(self.max_budget_usd) > allowance:
            raise WorkflowBudgetError(
                f"The repair starts {runs} new runs of up to {self.max_budget_usd} USD each, "
                f"more than max_total_budget_usd={max_total_budget_usd}")
        entries, successors, decisions, stopped = [], [], {}, False
        for item in items:
            entry = {"run_id": item["run_id"], "action": item["action"]}
            if stopped or item["action"] in ("reuse", "review"):
                outcome = {"reuse": "kept", "review": "reported"}.get(item["action"])
                entries.append({**entry, "outcome": "unapplied" if stopped else outcome,
                                "work": _journal_work(None, None)})
                continue
            entry.update(await self._repair_item(item, plan_sha256, reason))
            decision = entry.pop("record", None)
            if "successor_run_id" in entry:
                successors.append(entry["successor_run_id"])
            if decision is not None:
                decisions[entry.get("successor_run_id", item["run_id"])] = decision
                entry["decision"] = decision.decision.value
            entry["outcome"] = "applied" if decision is not None and decision.accepted else "failed"
            stopped = entry["outcome"] == "failed"
            entries.append(entry)
        return RepairResult(plan_sha256, entries, successors, decisions)

    async def _repair_item(self, item: dict[str, Any], plan_sha256: str,
                           reason: str) -> dict[str, Any]:
        """Apply one ``recheck``, ``rerender`` or ``regenerate`` item, once.

        Returns its decision as ``record`` (absent when it raised, with its
        ``error``), the run a ``regenerate`` item started, and its ``work``.
        """
        from smythe.workflow import current_decision

        store, run_id, action = self._run_store, item["run_id"], item["action"]
        runtime, done = self._workflow_runtime(), {}
        before = _journal_work(store, None if action == "regenerate" else run_id)
        try:
            if action == "recheck":
                done["record"] = (await self._arecheck(
                    run_id, plan_sha256=plan_sha256, reason=reason)).acceptance
            elif action == "rerender":
                await self._arerender(run_id, plan_sha256=plan_sha256, reason=reason)
                done["record"] = current_decision(store, run_id)["record"]
            else:
                done["record"] = (await runtime._run_successor(
                    run_id, lineage_sha256=item["lineage_sha256"],
                    reuse={node["id"]: node["reasons"] for node in item["nodes"]
                           if node["action"] == "reuse"},
                    plan_sha256=plan_sha256, reason=reason,
                    max_concurrency=self._execution_concurrency())).acceptance
        except Exception as error:  # The item failed; nothing retries it.
            done["error"] = f"{type(error).__name__}: {error}"
        # A successor exists once its run was created, even when it then failed.
        target = getattr(runtime, "run_id", None) if action == "regenerate" else run_id
        if action == "regenerate" and target is not None:
            done["successor_run_id"] = target
        after = _journal_work(store, target)
        done["work"] = {unit: after[unit] - before[unit] for unit in after}
        done["work"]["new_runs"] = int("successor_run_id" in done)
        return done

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

        Refuses a run a security stop ended with its ``SecurityStop``, a
        durable run's graph, conflicting task copies, model profiles its
        ``requires`` list does not match, and accounting that resume cannot
        continue: invalid usage, an accounting error or an unresolved native
        response failure.
        """
        acceptance = delivery_acceptance_from_state(state)
        model_profiles_from_state(state)
        stop = security_stop_from_state(state)
        if stop is not None:
            raise SecurityStop(stop["reason_code"])

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
        operator: OperatorDecision | None = None,
    ) -> SwarmResult:
        """Decide a completed run's saved output again.  Sync wrapper around areevaluate()."""
        return asyncio.run(self.areevaluate(execution_id, acceptance=acceptance, operator=operator))

    async def areevaluate(
        self, execution_id: str, *, acceptance: AcceptancePolicy | None = None,
        operator: OperatorDecision | None = None,
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
        this Swarm's only in its acceptance policy, its gate verifier and its
        lineage policy's ``reuse``. A run
        with a lineage policy also journals the new decision's lineage record,
        which supersedes the earlier one. See
        ``smythe.workflow.WorkflowRuntime.reevaluate``.

        *operator*, an :class:`~smythe.acceptance.OperatorDecision`, records an
        operator's decision instead of a policy's, and takes no *acceptance*.
        Only a current decision of ``needs_review`` or ``indeterminate`` can be
        resolved this way; any other stands, and ValueError is raised before
        the assertion is checked. The operator's approval verifier must accept
        its assertion for exactly this run's current decision, or its
        ``ApprovalRejected`` propagates and nothing is written. The new record,
        version 2, keeps what the current one binds and names the operator.
        """
        if operator is not None and acceptance is not None:
            raise ValueError("reevaluate takes acceptance= or operator=, not both")
        if operator is not None and not isinstance(operator, OperatorDecision):
            raise TypeError(f"operator must be an OperatorDecision, not {type(operator).__name__}")
        policy = self._acceptance if acceptance is None else acceptance
        if policy is not None and not isinstance(policy, AcceptancePolicy):
            raise TypeError(
                f"acceptance must be an AcceptancePolicy or None, not {type(policy).__name__}"
            )
        if operator is None and (policy is None or policy.mode is not AcceptanceMode.STRICT):
            raise ValueError(
                "reevaluate needs a strict AcceptancePolicy: pass acceptance=, or build the "
                "Swarm with one"
            )
        if self._run_store is not None:
            return await self._workflow_runtime(acceptance=policy).reevaluate(
                execution_id, operator=operator)
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
            if operator is not None:
                decided = _operator_record(operator, execution_id,
                                           delivery_acceptance_sequence(saved), record)
            else:
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
        Likewise, a run whose retry policy persists attempts resumes only
        under an equal policy, which starts each node from the attempts its
        generation already used, and such a policy never finishes a run saved
        without one. Resumed work runs one node at a time
        unless this Swarm has ``parallel=True``; a durable ``run_store``
        run keeps the concurrency its first execution recorded and resumes
        only under the acceptance policy its recipe records. A run that a
        security stop ended never resumes: its ``SecurityStop`` is raised
        before anything runs, whatever this Swarm's ``run_control``.
        """
        self._check_run_control()
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
        allowance = attempt_allowance_from_state(state)

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
        self._check_resume_retry_policy(execution_id, allowance)

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
            retry_policy=self.retry_policy,
            # Like revisions_used: a crash must not refill a node's attempts.
            attempts_used=allowance[1] if allowance is not None else None,
            profiles=tuple(self._profiles.values()), default_model=self.model,
            capacity_policy=self.capacity_policy,
            termination=self.termination,
            run_control=self.run_control,
        )
        # Informational: decisions after resume extend the log the run saved.
        executor.revision_log = revision_log_from_state(state)
        self._active_executor = executor
        tracer.add_phase("recovery", tracer.since(recovery_started))
        try:
            self._check_dispatch_support(graph, budget)
            # Holds are not saved, so a resumed run holds again.
            self._hold_check_budget(executor, graph, budget)
            graph = await executor.run(graph)
            executor.check_run_control()
            with tracer.phase("synthesis"):
                output = await self._synthesizer.asynthesize(
                    graph,
                    provider=self._provider,
                    model=self.model,
                    budget=budget,
                    tracer=tracer,
                    **self._merge_capacity(executor),
                )
            acceptance = self._decide(graph, output, task, tracer)
        except BaseException as exc:
            self._save_checkpoint(
                execution_id, "failed", graph, budget, task, created_at, timing,
                accounting_error=str(exc) if isinstance(exc, BudgetValidationError) else None,
                response_error=_response_error_marker(exc),
            )
            raise
        finally:
            self._release_holds(budget)

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
        lineage: LineagePolicy | None = None,
        profiles: Sequence[ModelProfile] = (),
    ) -> Swarm:
        """Create a Swarm pre-loaded with a YAML-defined execution graph.

        The returned Swarm's internal registry is populated with agents
        defined in the YAML file.  Call execute() with the loaded graph
        to run it.  A delivery acceptance policy comes only from the
        ``acceptance`` argument, never from the file, and so do the read
        profiles that the file's read steps name, the ``lineage`` policy and
        the model ``profiles`` that its nodes' ``metadata.profile`` names.
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
            lineage=lineage,
            profiles=profiles,
        )
        if instance._graph_policy is not None:
            instance._graph_policy.validate(graph, default_model=instance.model)
        instance._yaml_graph = graph
        instance._stamp_model(graph)
        return instance

    def _stamp_model(self, graph: ExecutionGraph) -> None:
        """Tag every node with the swarm's model so the executor knows which to call.

        A profiled node gets its profile's model instead. Every node is checked
        before any is tagged: one that names a profile this Swarm does not
        allow, or that sets another model than its profile's, raises
        ``NoAllowedProfileError``.
        """
        models = []
        for node in graph.nodes:
            profile = node_profile(node, self._profiles)
            models.append(self.model if profile is None else profile.model)
        for node, model in zip(graph.nodes, models):
            if "model" not in node.metadata:
                node.metadata["model"] = model

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
            context_sources=self._context_sources, renderer=self._renderer,
            artifact_dir=self.artifact_dir, retry_policy=self.retry_policy,
            lineage=self._lineage,
            reserve_for_checks=self.reserve_for_checks,
            run_control=self.run_control, termination=self.termination,
        )
