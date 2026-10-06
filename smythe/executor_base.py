"""ExecutorBase — shared logic for serial and async executors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from smythe._errors import PicklableError
from smythe._failures import QUALITY_DEFECT, classify_failure
from smythe._http_rejection import _http_rejection_receipt
from smythe._publish import publish_unique
from smythe.acceptance import AcceptanceMode, AcceptancePolicy, _unmet_obligations, task_sha256
from smythe.agent import Agent
from smythe.budget import (
    BudgetEstimateRequired,
    BudgetValidationError,
    Sentinel,
    validate_completion_usage,
    validate_token_count,
)
from smythe.context import (
    CONTEXT_SNAPSHOT_KEY, UNTRUSTED_DATA_NOTE, ContextSnapshot, text_sha256, untrusted_block,
)
from smythe.evidence import claim_exchange
from smythe.graph import (
    REVISION_ADDED_KEY, ExecutionGraph, Node, NodeStatus, Revision, RevisionError, _deliverable_ids,
)
from smythe.provider import (
    CompletionResult, NativeOutputDefectError, Provider, ProviderAccountingError,
    ProviderResponseError, _declared_features, _missing_cost_ceiling, _native_receipt,
    _raise_if_incomplete, _settle_response_error, _native_response_errors,
    _settle_response_group,
)
from smythe.provider_responses import RawResponseEnvelope
from smythe.registry import Registry
from smythe.scheduling import (
    PROFILE_KEY, ModelProfile, NoAllowedProfileError, RetryDeadlineError, RetryPolicy,
    UncertainDispatchError, _attempt_records, _retry_after_s, node_profile, profile_map,
)
from smythe.scheduling import CapacityPolicy, _CapacityLimiter
from smythe.task import _prompt_json, render_task, snapshot_task
from smythe.tools import TOOL_ERRORS_KEY, ChatMessage, ToolLoopLimitError, ToolResult, ToolRuntime
from smythe.tracer import (
    REVIEW_TRIGGER, REVISION_LOG_LIMIT, Tracer, read_revision_log, revision_context,
    revision_log_entry, revision_record,
)
from smythe.verifier import (
    ADVISORY_VERDICT_KEY, TokenVerifier, Verifier, VerificationRecoveryError, node_generation,
    read_verdict, rejection_reason, verification_integer, validate_verification_receipt,
)
from smythe.workflow_provider import ProviderRequestRejectedError
from smythe.workflow_store import _PRICED_KINDS, WorkflowError, WorkflowStateError, _price_table
from smythe.workflow_tools import READ_STEP_KEY

if TYPE_CHECKING:
    from smythe.supervisor import Supervisor
    from smythe.task import Task

DEFAULT_MAX_TOOL_ITERATIONS = 10


def _raise_workflow_error(error: BaseException) -> None:
    """Keep journal failures terminal when a context wraps them in a group."""
    if isinstance(error, WorkflowError):
        raise error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            try:
                _raise_workflow_error(child)
            except WorkflowError as found:
                raise found from error


def validate_retry_backoff_s(value: float) -> float:
    """Accept a finite, non-negative retry backoff in seconds.

    A bool is not a number here. NaN would make every retry delay NaN, and
    infinity (or an integer too large for a float) would make a retry sleep
    forever or overflow when the delay is drawn.
    """
    message = f"retry_backoff_s must be a finite non-negative number, got {value!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(message)
    return value


def validate_max_revisions(value: int) -> int:
    """Accept a non-negative cap on supervisor revisions."""
    if value < 0:
        raise ValueError(f"max_revisions must be >= 0, got {value}")
    return value


def validate_max_concurrency(value: int | None) -> int | None:
    """Accept a positive node concurrency cap, or None for unlimited."""
    if value is not None and value < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {value}")
    return value


def check_provider_support(
    provider: Provider, model: str, *, node_id: str | None, max_budget_usd: float | None = None,
    estimate: object = None, check_model: bool = True, tools: bool = False,
    attachments: bool = False, retry_owner: bool = False,
) -> None:
    """Raise, before any call, what calling *model* would raise for this configuration.

    *node_id* names the node making the call, or is None for every node of
    a generated plan. With *max_budget_usd* set and no *estimate*, a provider
    that requires an inclusive per-call ceiling and has none raises
    ``BudgetEstimateRequired``, as the executor would when it reserves the
    node's budget. A model (when *check_model*), tool calling or image
    attachments the provider declares unsupported raise ``ValueError``; a
    feature it does not declare never raises. With *retry_owner*, for a run
    under a retry policy, so does a provider that does not declare
    ``sdk_retries=0``.
    """
    name = type(provider).__name__
    if max_budget_usd is not None and estimate is None and _missing_cost_ceiling(provider, model):
        raise BudgetEstimateRequired(node_id, model, name)
    features = _declared_features(provider, model)
    if retry_owner and features.sdk_retries != 0:
        raise ValueError(
            f"retry_policy must be the only retry owner, but {name} declares "
            f"sdk_retries={features.sdk_retries!r} for model {model!r}, so its SDK may send a "
            "failed call again. Construct it with max_retries=0, or declare sdk_retries=0 "
            "in feature_support()"
        )
    if check_model and features.model_supported is False:
        where = "a generated plan's nodes" if node_id is None else f"node {node_id!r}"
        raise ValueError(f"{name} does not support model {model!r}, which {where} would use")
    if tools and features.tools is False:
        raise ValueError(
            f"tool_runtime needs a provider that can call tools; {name} cannot with model {model!r}"
        )
    if attachments and features.attachments is False:
        raise ValueError(
            f"attach_dep_artifacts on node {node_id!r} needs a provider that accepts image "
            f"attachments; {name} does not with model {model!r}"
        )


def check_pending_nodes(
    provider: Provider, graph: ExecutionGraph, *, max_budget_usd: float | None, tools: bool,
    retry_owner: bool = False, profiles: dict[str, ModelProfile] | None = None,
) -> None:
    """Check every node that has not completed or been skipped, before any of them runs.

    A node that names one of *profiles* is checked against its profile's
    provider and model, with the profile's ceiling standing in for an
    estimate of its own; one that names any other profile raises
    ``NoAllowedProfileError``.
    """
    for node in graph.nodes:
        if node.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
            continue
        _exchanges_claims(node)
        profile = node_profile(node, profiles or {})
        estimate = node.metadata.get("estimated_cost_usd")
        if profile is not None and estimate is None:
            estimate = profile.max_cost_per_call_usd
        check_provider_support(
            provider if profile is None else profile.provider,
            str(node.metadata.get("model", "")) if profile is None else profile.model,
            node_id=node.id, max_budget_usd=max_budget_usd, estimate=estimate,
            tools=tools, attachments=node.attach_dep_artifacts, retry_owner=retry_owner,
        )


def _retired(provider: Provider, model: str) -> bool:
    """Whether *provider* prices *model* with a versioned table that retires it by today (UTC).

    The table is the current one of the durable provider kind its
    ``workflow_descriptor()`` names; a provider without one prices nothing.
    """
    describe = getattr(provider, "workflow_descriptor", None)
    try:
        kind = describe().get("kind") if callable(describe) else None
    except (AttributeError, TypeError, ValueError):  # Such as a scripted OfflineProvider.
        return False
    if type(kind) is not str or kind not in _PRICED_KINDS:
        return False
    table = _price_table(kind)
    return table.prices(model) and not table.admits(model, datetime.now(timezone.utc).date())


def check_profile_support(profile: ModelProfile, *, tools: bool = False) -> None:
    """Raise ``NoAllowedProfileError``, before any call, for a profile that cannot serve its nodes.

    Its provider must not declare its model unsupported, must declare
    ``True`` every feature the profile requires, and must not price the model
    with a table that retires it. With *tools*, a run offers tools to every
    node, so a provider that declares it cannot call them is refused too.
    """
    provider, model, name = profile.provider, profile.model, type(profile.provider).__name__
    features = _declared_features(provider, model)
    if features.model_supported is False:
        raise NoAllowedProfileError(
            f"Model profile {profile.name!r}: {name} does not support model {model!r}"
        )
    missing = sorted(feature for feature in profile.requires
                     if getattr(features, feature) is not True)
    if missing:
        raise NoAllowedProfileError(
            f"Model profile {profile.name!r} requires {missing}, which {name} does not "
            f"declare for model {model!r}"
        )
    if tools and features.tools is False:
        raise NoAllowedProfileError(
            f"Model profile {profile.name!r} cannot serve a run with tool_runtime: {name} "
            f"cannot call tools with model {model!r}"
        )
    if _retired(provider, model):
        raise NoAllowedProfileError(
            f"Model profile {profile.name!r} names {model!r}, which is retired and admits no "
            "new calls"
        )


def check_supervisor_profiles(
    supervisor: Supervisor | None, profiles: dict[str, ModelProfile],
) -> None:
    """Raise ``NoAllowedProfileError``, before any call, for a review sent to the wrong provider.

    Without a model of its own, an ``LLMSupervisor`` reviews a node with that
    node's model on its own provider, and a profiled node's model is its
    profile's. Each of *profiles* must therefore call the supervisor's
    provider. Callers check only while the supervisor may still review.
    So does one inside an ``EvidenceTriggeredSupervisor``.
    """
    from smythe.supervisor import _model_reviewer

    supervisor = _model_reviewer(supervisor)
    if supervisor is None or supervisor._model:
        return
    for profile in profiles.values():
        if profile.provider is not supervisor._provider:
            raise NoAllowedProfileError(
                f"LLMSupervisor reviews a node with that node's model, but model profile "
                f"{profile.name!r} calls {profile.model!r} on {type(profile.provider).__name__}, "
                "not on the supervisor's provider; give the supervisor its own model with "
                "LLMSupervisor(provider, model=...)"
            )


class NodeFinalizationError(PicklableError, RuntimeError):
    """Raised after a billed response cannot be persisted safely.

    This error is non-retryable at the executor layer: repeating the provider
    call would buy the same output again merely because local finalization
    failed.
    """

    def __init__(self, node_id: str, cause: Exception) -> None:
        self.node_id = node_id
        self.cause = cause
        super().__init__(f"Node {node_id!r} finalization failed: {cause}")


class ArtifactIntegrityError(ValueError):
    """A dependency's artifact file no longer holds the bytes its record bound.

    Raised before the provider call of a node that attaches its dependencies'
    images, so it never sends bytes that differ from the ``sha256`` recorded
    when the file was written.
    """


def _safe_filename_component(raw: str) -> str:
    """Make an arbitrary node id safe as a filename component.

    Node ids from YAML and graphs built in code have no character
    validation (only model-generated ids are restricted), so they can
    carry path separators (traversal out of the artifact dir) or
    Windows-illegal characters (OSError after an already-billed provider
    call).  When sanitization changes anything, a short hash of the
    original keeps distinct ids distinct.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._") or "node"
    if safe != raw:
        safe = f"{safe}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}"
    return safe

# When False, root nodes ignore the task_context stamped by Swarm.plan()
# and see only their planned label - exists for benchmark ablations.
INCLUDE_TASK_CONTEXT = True

# Appended to a terminal node's prompt so the deliverable survives the
# chain: without it, final nodes tend to reference or summarize upstream
# findings instead of reproducing them, and the specifics are lost
# (benchmarks/README.md documents the failure mode this addresses).
#
# The second paragraph exists because the first was not enough. A node
# labelled "state the surviving claims" obeyed its label and emitted only
# the survivors, dropping the case and the critique the goal had also
# asked for - the label scopes the model's behaviour more strongly than a
# general instruction to be self-contained. The shape suite scored those
# runs 1-2 out of 10, because most of what the task asked for was never
# returned. The fix is to say plainly that the step name describes a
# contribution, not the extent of the output.
TERMINAL_DELIVERABLE_NOTE = (
    "You are the final step in this workflow: your output is the "
    "deliverable, and the context above will not be shown alongside it. "
    "Make your output self-contained - carry forward the concrete "
    "findings, evidence, and specifics from the context rather than "
    "referencing or summarizing them.\n\n"
    "Your step name describes what you contribute, not how much you "
    "output. Deliver everything the overall task asked for, including "
    "the parts produced in earlier steps: if the task asked for an "
    "argument and then a critique of it, return both, not just your "
    "own increment."
)

# How node prompts are assembled. Version 2 gives TERMINAL_DELIVERABLE_NOTE to
# the nodes DELIVERABLE synthesis returns: terminal nodes that are not
# verifiers, where a judge's edge to the node it judges does not count.
# Version 1 counted that edge, so in a gated graph the judge received the note
# and the node it judged did not. Durable workflows keep version 1 unless their
# saved recipe records another (see smythe.workflow.RECIPE_BEHAVIOR_FIELDS),
# because a run's saved requests must not change when it resumes.
PROMPT_ASSEMBLY_VERSION = 2
SUPPORTED_PROMPT_ASSEMBLY_VERSIONS = (1, 2)

# What a dependent sees in place of a SKIPPED dependency's result. A node
# skipped after a failure keeps its error text in ``node.result`` for traces
# and checkpoints, but that text is not step output: passed on as context,
# the terminal note above would ask the model to carry it into the
# deliverable.
SKIPPED_DEPENDENCY_RESULT = "[skipped: this step did not complete]"

# A node whose metadata sets this key to "claims" is given, for each dependency
# whose output carries source evidence with claims, the compact record
# smythe.evidence.claim_exchange makes in place of that output.
EXCHANGE_KEY = "exchange"


def _exchanges_claims(node: Node) -> bool:
    """Whether *node* exchanges claims; another ``exchange`` value raises ValueError."""
    if EXCHANGE_KEY not in node.metadata:
        return False
    if node.metadata[EXCHANGE_KEY] != "claims":
        raise ValueError(f"Node {node.id!r} metadata {EXCHANGE_KEY!r} must be 'claims', not "
                         f"{node.metadata[EXCHANGE_KEY]!r}")
    return True


# Under a strict acceptance policy, a regeneration moves each discarded
# generation's result and artifact records into a list under this node
# metadata key instead of dropping them.
REJECTED_GENERATIONS_KEY = "rejected_generations"
# A record keeps at most this many bytes of the result's UTF-8 text inline;
# with an artifact_dir, a sidecar file holds a longer result in full.
REJECTED_EXCERPT_BYTES = 1024
# Ordinary checkpoints rewrite the whole graph on every save, so the records
# of all nodes together may add at most this many bytes to one checkpoint.
REJECTED_GENERATIONS_MAX_BYTES = 64 * 1024


def _checkpoint_bytes(record: Any) -> int:
    """At least the bytes *record* adds to a checkpoint ``FileCheckpointStore`` writes.

    That store indents JSON by two spaces and escapes non-ASCII text. A record
    sits six levels deep, so each line gains twelve spaces and, on Windows, a
    carriage return; 64 bytes cover the separators and the node's list key.
    """
    text = json.dumps(record, indent=2, default=str)
    return len(text) + 16 * (text.count("\n") + 1) + 64


def _retained_bytes(graph: ExecutionGraph) -> int:
    records = (node.metadata.get(REJECTED_GENERATIONS_KEY) for node in graph.nodes)
    return sum(_checkpoint_bytes(record) for listed in records if isinstance(listed, list)
               for record in listed)


class ExecutorBase:
    """Shared infrastructure for all executor variants.

    Provides constructor, prompt building, dependency lookup, the
    tool-calling loop, and node-by-id helpers.  Subclasses implement
    ``run()`` and drive ``acall_node()`` (sync or async).

    With a strict ``acceptance`` policy, every gate's final verdict is read
    and recorded for delivery acceptance: an exhausted gate's last verdict, a
    gate without regenerations under ``advisory_verdict``, and a verdict that
    cannot be read or a verifier that raises as an undecided outcome that
    ends the gate without regeneration. Without one, gates behave as before,
    except that a durable run's registered gate validator that cannot decide
    ends its gate undecided in the same way.

    ``prompt_assembly`` selects how node prompts are assembled (see
    ``PROMPT_ASSEMBLY_VERSION``); version 1 reproduces the earlier placement
    of ``TERMINAL_DELIVERABLE_NOTE``.

    ``read_call_factory(node, attempt)`` returns the journaled read for a
    node that declares a read step (``metadata["read_step"]``), which then
    runs without a model call. Without a factory such a node raises
    ``ValueError``: read steps run only in durable workflows.

    ``context_assembler``, an async ``(node, graph, dep_results)`` callable
    returning a :class:`~smythe.context.ContextSnapshot`, opts into rendering
    each node prompt from its snapshot (:meth:`build_context_prompt`), with
    tool results marked as untrusted data as well. Without one, prompts keep
    their earlier bytes.

    ``retry_policy``, a :class:`~smythe.scheduling.RetryPolicy`, opts into
    classified retries with ``Retry-After``, a backoff cap and a deadline
    shared by each node's attempts and waits; a run under one refuses,
    before any node call, a provider that does not declare
    ``sdk_retries=0``, its own, any model profile's or, while revisions
    remain, an ``LLMSupervisor``'s. ``sleep(seconds)`` replaces the wait before a retry
    (``time.sleep`` in the serial executor, ``asyncio.sleep`` in the async
    one) and is awaited when it returns an awaitable; ``rng`` replaces the
    ``random`` module that draws backoff jitter. Under a policy with
    ``persist_attempts``, each attempt is recorded in :attr:`attempts_used`
    and saved through the control hook before its call, and
    ``attempts_used``, as a checkpoint saved it, seeds the attempts each node
    has already used in its current ``execution_generation``. The seed applies
    to the executor's first run only: a later run is a new run, with a fresh
    allowance for every node. Under a policy with
    ``retain_uncertain_dispatch``, a node call that is sent and never
    answered, because the node's timeout, a cancellation or an interruption
    ended it, keeps its reservation as the node's unknown, estimated cost
    (``Sentinel.retain_unknown``) instead of releasing it.
    A durable journal records each attempt as its
    call, so journal-managed calls take a policy only with
    ``attempts_spent(node)``, which returns the attempts the node's current
    generation has already used each time the node starts; nothing is saved
    through the control hook for an attempt.

    ``profiles``, a sequence of :class:`~smythe.scheduling.ModelProfile`,
    routes each node that names one in ``metadata["profile"]`` to that
    profile's provider and model, and reserves the profile's ceiling for it
    unless the node sets ``estimated_cost_usd``. A node that names any other
    profile, or sets another model than its profile's, raises
    ``NoAllowedProfileError`` before its call, and a revision that adds such
    a node is refused before it applies. Every other node calls *provider*
    with its own model, and a node a supervisor adds without a profile takes
    the model of the first such node that has one, or else ``default_model``.
    While revisions remain, a run refuses with ``NoAllowedProfileError``,
    before any call, an ``LLMSupervisor`` without a model of its own and a
    profile that calls another provider than the supervisor's.

    ``capacity_policy``, a :class:`~smythe.scheduling.CapacityPolicy`, makes
    every provider call of a node, including tool turns and retries, wait
    for its provider type and model's turn: ``min_interval_s`` after the
    previous call's start and after any ``Retry-After`` cooldown, on the
    tracer's clock, with ``sleep`` replacing the wait. The async executor
    also admits at most ``max_in_flight`` nodes of a model at once. Each run
    starts with fresh limits, and every provider hint the run's pending nodes
    need is read before any node call.
    """

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        tracer: Tracer,
        budget: Sentinel | None = None,
        on_node_update: Callable[[Node], None] | None = None,
        tool_runtime: ToolRuntime | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        artifact_dir: str | Path | None = "smythe_artifacts",
        retry_backoff_s: float = 0.0,
        supervisor: Supervisor | None = None,
        max_revisions: int = 0,
        task: Task | None = None,
        verifier: Verifier | None = None,
        revisions_used: int = 0,
        on_control_update: Callable[[], None] | None = None,
        provider_call_factory: Callable[[Node, str, int, int], Provider] | None = None,
        on_supervision_update: Callable[[Node, bool], None] | None = None,
        *,
        prompt_assembly: int = PROMPT_ASSEMBLY_VERSION,
        read_call_factory: Callable[[Node, int], Any] | None = None,
        context_assembler: Callable[
            [Node, ExecutionGraph, dict[str, Any]], Awaitable[ContextSnapshot]
        ] | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], Any] | None = None,
        rng: Any = None,
        attempts_used: dict[str, dict[str, int]] | None = None,
        attempts_spent: Callable[[Node], int] | None = None,
        profiles: tuple[ModelProfile, ...] | list[ModelProfile] = (),
        default_model: str | None = None,
        capacity_policy: CapacityPolicy | None = None,
        acceptance: AcceptancePolicy | None = None,
    ) -> None:
        if context_assembler is not None and not callable(context_assembler):
            raise TypeError("context_assembler must be callable or None")
        self._profiles = profile_map(profiles)
        self._default_model = default_model
        self._context_assembler = context_assembler
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError(
                f"retry_policy must be a RetryPolicy or None, not {type(retry_policy).__name__}"
            )
        if sleep is not None and not callable(sleep):
            raise TypeError("sleep must be callable or None")
        if rng is not None and not callable(getattr(rng, "uniform", None)):
            raise TypeError("rng must have a uniform(low, high) method, or be None")
        self._retry_policy, self._sleep, self._rng = retry_policy, sleep, rng
        # When each node's retry deadline passes, on the tracer's clock.
        self._deadlines: dict[str, float] = {}
        if attempts_spent is not None and (not callable(attempts_spent) or retry_policy is None):
            raise TypeError("attempts_spent must be callable, and needs a retry_policy")
        # A journal's count of each node's attempts, which no checkpoint duplicates.
        self._journal_attempts = attempts_spent
        self._persist_attempts = (retry_policy is not None and retry_policy.persist_attempts
                                  and attempts_spent is None)
        if attempts_used is not None and not self._persist_attempts:
            raise ValueError("attempts_used needs a retry_policy with persist_attempts")
        # Like revisions_used, a per-run allowance: each node's generation and
        # the attempts it has spent there, restored on resume. The seed binds
        # only the first run; a reused executor's next run starts empty.
        self._attempts_used = _attempt_records(attempts_used or {})
        self._attempts_seed = dict(self._attempts_used)
        self._retain_dispatch = retry_policy is not None and retry_policy.retain_uncertain_dispatch
        # Under retain_uncertain_dispatch, the nodes whose call was sent and
        # has not answered: in flight, or ended by a timeout or cancellation.
        self._dispatched: set[str] = set()
        if acceptance is not None and not isinstance(acceptance, AcceptancePolicy):
            raise TypeError(
                f"acceptance must be an AcceptancePolicy or None, not {type(acceptance).__name__}"
            )
        if type(prompt_assembly) is not int or prompt_assembly not in SUPPORTED_PROMPT_ASSEMBLY_VERSIONS:
            raise ValueError(
                f"prompt_assembly must be one of {list(SUPPORTED_PROMPT_ASSEMBLY_VERSIONS)}, "
                f"not {prompt_assembly!r}"
            )
        self._prompt_assembly = prompt_assembly
        self._strict = acceptance is not None and acceptance.mode is AcceptanceMode.STRICT
        self._acceptance = acceptance if self._strict else None
        self._provider = provider
        self._provider_call_factory = provider_call_factory
        self._read_call_factory = read_call_factory
        self._workflow_managed = (
            provider_call_factory is not None or getattr(provider, "workflow_managed", False) is True
        )
        if retry_policy is not None and self._workflow_managed and attempts_spent is None:
            raise ValueError(
                "retry_policy with journal-managed provider calls needs attempts_spent, "
                "the attempts the journal records"
            )
        if self._profiles and self._workflow_managed:
            raise ValueError("profiles are not supported with journal-managed provider calls")
        if capacity_policy is not None and not isinstance(capacity_policy, CapacityPolicy):
            raise TypeError(
                f"capacity_policy must be a CapacityPolicy or None, "
                f"not {type(capacity_policy).__name__}"
            )
        if capacity_policy is not None and self._workflow_managed:
            raise ValueError("capacity_policy is not supported with journal-managed provider calls")
        # Each run's limiter, made when the run starts; None without a policy.
        self._capacity_policy = capacity_policy
        self._capacity: _CapacityLimiter | None = None
        self._call_attempts: dict[str, int] = {}
        self._reserved_node_ids: set[str] = set()
        self._settled_response_errors: list[tuple[str, Exception]] = []
        # Settled errors whose node marker a retry policy cleared for another attempt.
        self._cleared_response_errors: list[Exception] = []
        self._response_settlement_failures: dict[str, Exception] = {}
        self._registry = registry
        self._tracer = tracer
        self._budget = budget
        self._on_node_update = on_node_update
        self._on_control_update = on_control_update
        self._on_supervision_update = on_supervision_update
        self._tool_runtime = tool_runtime
        self._max_tool_iterations = max_tool_iterations
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._prepared_graph: ExecutionGraph | None = None
        self._node_lookup: dict[str, Node] = {}
        self._terminal_ids: set[str] = set()
        self._retry_backoff_s = validate_retry_backoff_s(retry_backoff_s)
        self._max_revisions = validate_max_revisions(max_revisions)
        self._supervisor = supervisor
        self._default_task = snapshot_task(task) if task is not None else None
        self._task = snapshot_task(self._default_task) if self._default_task is not None else None
        # Restored from the checkpoint on resume: the revision cap is a
        # per-run guarantee, so a crash must not refill the allowance.
        self._revisions_used = revisions_used
        self._revision_log: list[dict[str, Any]] = []
        # What prompted the current review, for its trace span and log entry.
        self._review_trigger = REVIEW_TRIGGER
        self._verifier = verifier or TokenVerifier()

    @property
    def revisions_used(self) -> int:
        """How many supervisor revisions this executor has applied."""
        return self._revisions_used

    @property
    def revision_log(self) -> list[dict[str, Any]]:
        """The run's latest supervisor decisions, applied or refused, oldest first.

        Each entry is a :func:`~smythe.tracer.revision_log_entry`, and at most
        ``REVISION_LOG_LIMIT`` are kept. Resume sets the log a checkpoint
        saved, which later decisions extend.
        """
        return read_revision_log(self._revision_log)

    @revision_log.setter
    def revision_log(self, entries: list[dict[str, Any]]) -> None:
        self._revision_log = read_revision_log(entries)

    @property
    def attempts_used(self) -> dict[str, dict[str, int]]:
        """Under a policy with ``persist_attempts``, each node's attempts in its generation."""
        return {node_id: {"execution_generation": generation, "attempts": used}
                for node_id, (generation, used) in self._attempts_used.items()}

    async def maybe_revise(self, node: Node, graph: ExecutionGraph) -> bool:
        """Let the supervisor revise the plan after *node* completed.

        Returns True when the graph changed, so the caller can refresh
        any scheduling state it derived from it. Ordinary supervisor
        errors and invalid changes are contained and traced. Invalid
        provider accounting is terminal because continuing would spend
        more while the supervisor's charge remains unresolved. Under a
        strict acceptance policy, a revision that breaks a delivery
        acceptance obligation is an invalid change too.

        Each decision that the trace records, applied or refused, also
        joins :attr:`revision_log` with its effect on the graph (see
        :func:`~smythe.tracer.revision_context`). For a supervisor with
        evidence triggers, the triggers that fire are traced before its
        review and named as the decision's ``trigger``.
        """
        if self._supervisor is None or node.status is not NodeStatus.COMPLETED:
            return False
        remaining = self._max_revisions - self._revisions_used
        if remaining <= 0:
            return False

        started = self._tracer.now()
        fired = self._fired_triggers(graph, node)
        self._review_trigger = ",".join(trigger.name for trigger in fired) or REVIEW_TRIGGER
        if fired:
            self._tracer.on_review_trigger(node, fired)
        try:
            revision = await self._supervisor.review(
                graph, node, task=snapshot_task(self._task) if self._task is not None else None,
                revisions_remaining=remaining,
            )
        except BaseExceptionGroup as exc:
            if _native_response_errors(exc):
                _settle_response_group(
                    exc, lambda error: self._record_response_error(node, error, phase="supervision"),
                )
            _raise_workflow_error(exc)
            self._trace_revision(node, graph, None, applied=False, detail=str(exc))
            self.notify_supervision_update(node, False)
            return False
        except WorkflowError:
            raise
        except ProviderResponseError as exc:
            self._record_response_error(node, exc, phase="supervision")
            self.notify_update(node)
            raise
        except BudgetValidationError as exc:
            self.mark_accounting_invalid(node, exc)
            self._trace_revision(
                node, graph, None, applied=False, detail=f"supervisor accounting invalid: {exc}",
            )
            self.notify_update(node)
            raise
        except Exception as exc:
            self._trace_revision(
                node, graph, None, applied=False, detail=f"supervisor raised: {exc}",
            )
            self.notify_supervision_update(node, False)
            return False
        finally:
            self._tracer.add_phase("supervision", self._tracer.since(started))
        if revision is None or revision.is_empty:
            self.notify_supervision_update(node, False)
            return False

        delivered = _deliverable_ids(graph.nodes)
        try:
            for added in revision.add_nodes:
                # An added node is routed like any other: refuse one no profile can serve.
                node_profile(added, self._profiles)
            self._apply_revision(graph, revision)
        except (RevisionError, NoAllowedProfileError) as exc:
            self._trace_revision(node, graph, revision, applied=False, detail=str(exc))
            self.notify_supervision_update(node, False)
            return False

        self._revisions_used += 1
        self._inherit_execution_context(revision.add_nodes, graph)
        self.prepare_graph(graph)
        self._trace_revision(node, graph, revision, applied=True, delivered=delivered)
        self.notify_supervision_update(node, True)
        return True

    def _fired_triggers(self, graph: ExecutionGraph, node: Node) -> tuple:
        """The evidence triggers that call for a review after *node*; none without any.

        Read from the graph before the review, as the supervisor reads them.
        """
        from smythe.supervisor import ReviewTriggers

        triggers = getattr(self._supervisor, "triggers", None)
        if not isinstance(triggers, ReviewTriggers):
            return ()
        try:
            return triggers.fired(graph, node)
        except Exception:  # The review reads them again and its error is contained.
            return ()

    def _trace_revision(
        self, node: Node, graph: ExecutionGraph, revision: Revision | None, *, applied: bool,
        detail: str = "", delivered: set[str] | None = None,
    ) -> None:
        """Trace a supervisor decision on *graph*, as it stands after it, and log it.

        *delivered* holds the graph's deliverable nodes before an applied
        *revision*; a refused one left the graph unchanged.
        """
        context = revision_context(
            graph, revisions_remaining_after=self._max_revisions - self._revisions_used,
            delivered_before=delivered,
            added=[added.id for added in revision.add_nodes] if applied and revision else (),
            trigger=self._review_trigger,
        )
        self._tracer.on_revision(node, revision, applied=applied, detail=detail, context=context)
        entry = revision_log_entry(revision_record(node, revision, context), applied=applied,
                                   reason=getattr(revision, "reason", None), detail=detail)
        self._revision_log = [*self._revision_log, entry][-REVISION_LOG_LIMIT:]

    def _apply_revision(self, graph: ExecutionGraph, revision: Revision) -> None:
        """Apply *revision*; under a strict policy, refuse one that breaks an obligation.

        The revised graph must meet every delivery acceptance obligation the
        graph met before it (see ``validate_acceptance_obligations``). Without
        final validators, a step added after a gated node, another delivered
        node, or a gate the revision adds itself would leave the delivered
        output without a check that can accept it, so such a revision raises
        RevisionError and the graph is left exactly as it was.
        """
        policy = self._acceptance
        if policy is None:
            graph.apply_revision(revision)
            return
        before = set(_unmet_obligations(graph, policy))
        nodes, edges = list(graph.nodes), {n.id: n.depends_on for n in graph.nodes}
        graph.apply_revision(revision)
        broken = [problem for problem in _unmet_obligations(graph, policy) if problem not in before]
        if broken:
            graph.nodes = nodes
            for kept in nodes:
                kept.depends_on = edges[kept.id]
            for added in revision.add_nodes:
                added.metadata.pop(REVISION_ADDED_KEY, None)
            raise RevisionError(
                "a strict acceptance policy refuses this revision: " + "; ".join(broken)
            )

    def notify_supervision_update(self, node: Node, revision_applied: bool) -> None:
        """Persist a consumed supervisor decision after its graph disposition."""
        if self._on_supervision_update is not None:
            try:
                self._on_supervision_update(node, revision_applied)
            except WorkflowError:
                raise
            except Exception as exc:
                raise WorkflowError("Could not persist supervisor disposition") from exc

    def maybe_regenerate(self, node: Node, graph: ExecutionGraph) -> bool:
        """Apply a durable rejection when the caller has no active descendants."""
        intent = self.prepare_regeneration(node, graph)
        if intent is None:
            return False
        if any(n.status is NodeStatus.RUNNING for n in graph.nodes
               if n.id in intent["affected_generations"]):
            raise VerificationRecoveryError("Active descendants must settle before regeneration")
        self.apply_regeneration(node, graph, intent)
        return True

    def notify_control_update(self, node: Node) -> None:
        """Persist a control transition without batching or changing old hooks."""
        try:
            if self._on_control_update is not None:
                self._on_control_update()
            else:
                self.notify_update(node)
        except WorkflowError:
            raise
        except Exception as exc:
            raise VerificationRecoveryError("Could not persist verification control") from exc

    def begin_verification(self, node: Node, graph: ExecutionGraph) -> None:
        """Bind the judge's request to the version of its target it observes.

        Under a strict policy a new attempt also drops the judge's recorded
        ``advisory_verdict``, so the verdict recorded is always the one this
        attempt produces, never one the judge carried into the run.
        """
        node_generation(node)
        if node.verifies and (node.max_regenerations > 0 or self._strict):
            target = self._cached_node_by_id(node.verifies, graph)
            node.metadata["verification_target_generation"] = (
                node_generation(target) if target is not None else None
            )
            if self._strict:
                node.metadata.pop(ADVISORY_VERDICT_KEY, None)

    def complete_verification(self, node: Node) -> None:
        """Stamp pending before any COMPLETED checkpoint can become visible."""
        if node.status is not NodeStatus.COMPLETED or not node.verifies or node.max_regenerations <= 0:
            return
        node.metadata["verification_receipt"] = {
            "version": 1, "state": "pending", "judge_generation": node_generation(node),
            "target_id": node.verifies,
            "target_generation": node.metadata.get("verification_target_generation"),
        }
        self.notify_control_update(node)

    def _affected_nodes(self, target: Node, graph: ExecutionGraph) -> list[Node]:
        children: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        for candidate in graph.nodes:
            for dep_id in candidate.depends_on:
                children.setdefault(dep_id, []).append(candidate.id)
            # A judge also observes its target when a hand-built graph omitted
            # the dependency edge. Its old verdict must never survive a reset.
            if candidate.verifies:
                children.setdefault(candidate.verifies, []).append(candidate.id)
        stack = [target.id]
        seen: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(children.get(node_id, ()))
        return [n for n in graph.nodes if n.id in seen]

    def prepare_regeneration(self, node: Node, graph: ExecutionGraph) -> dict | None:
        """Persist one rejection intent before cancellation or graph mutation."""
        existing = node.metadata.get("regeneration_intent")
        if existing is not None:
            self._validate_regeneration(node, graph, existing)
            return existing
        if node.status is not NodeStatus.COMPLETED or not node.verifies:
            return None
        if node.max_regenerations <= 0:
            if self._strict:
                self._record_advisory_verdict(node, graph)
            return None
        receipt = node.metadata.get("verification_receipt")
        # Completed legacy/manual nodes are not silently rejudged.
        if receipt is None:
            return None
        validate_verification_receipt(node, graph)
        if receipt.get("state") == "consumed":
            return None
        if receipt.get("state") != "pending" or receipt.get("judge_generation") != node_generation(node):
            raise VerificationRecoveryError(f"Stale verification receipt on {node.id!r}")
        target = self._cached_node_by_id(node.verifies, graph)
        used = verification_integer(node.metadata.get("regenerations_used", 0), "regenerations_used")
        valid_target = target is not None and target.status is NodeStatus.COMPLETED
        if valid_target and receipt.get("target_generation") != node_generation(target):
            raise VerificationRecoveryError(f"Verifier {node.id!r} observed an obsolete target")
        if receipt.get("target_id") != node.verifies:
            raise VerificationRecoveryError(f"Verifier {node.id!r} target identity changed")
        if self._strict:
            verdict = self._read_final_verdict(node, receipt, target if valid_target else None, used)
            if verdict is None:
                return None
        else:
            if not valid_target or used >= node.max_regenerations:
                node.metadata["verification_receipt"] = dict(receipt, state="consumed", reason="unavailable or exhausted")
                self.notify_control_update(node)
                return None
            verdict = self._verifier.verdict(node, target)
            # Only an exact True passes; a truthy "false" from a custom verifier fails.
            if verdict.passed is True:
                node.metadata["verification_receipt"] = dict(receipt, state="consumed", passed=True, reason=verdict.reason)
                self.notify_control_update(node)
                return None
            if (getattr(self._verifier, "_invalid_ends_gate", False) is True
                    and getattr(verdict, "invalid", False) is True):
                # A registered validator that could not decide ends the gate
                # undecided, as under a strict policy: no regeneration is spent.
                node.metadata["verification_receipt"] = dict(
                    receipt, state="consumed", outcome="error", passed=False,
                    reason=rejection_reason(verdict),
                )
                self.notify_control_update(node)
                return None
        intent = {
            "version": 1, "target_id": target.id, "target_generation": node_generation(target),
            "judge_generation": node_generation(node), "reason": rejection_reason(verdict),
            "regenerations_used": used + 1,
            "affected_generations": {n.id: node_generation(n) for n in self._affected_nodes(target, graph)},
        }
        node.metadata["regenerations_used"] = used + 1
        node.metadata["regeneration_intent"] = intent
        self.notify_control_update(node)
        return intent

    def _judge(self, node: Node, target: Node) -> tuple[str, str, Any]:
        """Strict mode: the verdict's outcome, reason and the verdict itself.

        A verifier that raises gives an ``error`` outcome rather than
        stopping the run, so the gate ends undecided.
        """
        try:
            verdict = self._verifier.verdict(node, target)
        except Exception as exc:
            return "error", f"verifier raised {type(exc).__name__}", None
        outcome, reason = read_verdict(verdict)
        return outcome, reason, verdict

    def _read_final_verdict(
        self, node: Node, receipt: dict, target: Node | None, used: int,
    ) -> Any:
        """Strict mode: consume the receipt unless a rejection may still regenerate.

        Returns the rejecting verdict while regenerations remain. Otherwise
        the receipt records the outcome: a pass, an exhausted gate's failing
        final verdict, or an ``error`` (an unavailable target, an unreadable
        verdict or a raising verifier), which consumes no regeneration.
        """
        if target is None:
            outcome, reason = "error", "verification target unavailable"
        else:
            outcome, reason, verdict = self._judge(node, target)
            if outcome == "rejected" and used < node.max_regenerations:
                return verdict
        fields: dict[str, Any] = {"outcome": outcome, "passed": outcome == "passed", "reason": reason}
        if outcome == "rejected":
            fields["exhausted"] = True
        node.metadata["verification_receipt"] = dict(receipt, state="consumed", **fields)
        self.notify_control_update(node)
        return None

    def _record_advisory_verdict(self, node: Node, graph: ExecutionGraph) -> None:
        """Strict mode: read a gate without regenerations once and record its verdict.

        A record already bound to the judge's generation is kept only because
        the judge has not run again since it was made: each attempt drops the
        record when it begins, so this is a resumed run's saved verdict.
        """
        generation = node_generation(node)
        recorded = node.metadata.get(ADVISORY_VERDICT_KEY)
        if (isinstance(recorded, dict) and type(recorded.get("judge_generation")) is int
                and recorded["judge_generation"] == generation):
            return
        target = self._cached_node_by_id(node.verifies, graph)
        if target is None or target.status is not NodeStatus.COMPLETED:
            outcome, reason = "error", "verification target unavailable"
        elif node.metadata.get("verification_target_generation") != node_generation(target):
            outcome, reason = "error", "the judge observed an obsolete target"
        else:
            outcome, reason, _ = self._judge(node, target)
        node.metadata[ADVISORY_VERDICT_KEY] = {
            "version": 1, "judge_generation": generation, "target_id": node.verifies,
            "target_generation": node_generation(target) if target is not None else None,
            "outcome": outcome, "passed": outcome == "passed", "reason": reason,
        }
        self.notify_control_update(node)

    def _validate_regeneration(self, node: Node, graph: ExecutionGraph, intent: dict) -> None:
        if (not isinstance(intent, dict) or type(intent.get("version")) is not int
                or intent["version"] != 1 or not isinstance(intent.get("reason"), str)
                or not isinstance(intent.get("target_id"), str)):
            raise VerificationRecoveryError("Invalid regeneration intent")
        target = self._cached_node_by_id(intent.get("target_id"), graph)
        if target is None or node.verifies != target.id:
            raise VerificationRecoveryError("Regeneration target identity changed")
        affected = intent.get("affected_generations")
        if not isinstance(affected, dict) or set(affected) != {n.id for n in self._affected_nodes(target, graph)}:
            raise VerificationRecoveryError("Regeneration affected-node inventory changed")
        used = verification_integer(intent.get("regenerations_used"), "intent regenerations_used")
        saved_used = verification_integer(node.metadata.get("regenerations_used"), "regenerations_used")
        if used < 1 or used > node.max_regenerations or saved_used != used:
            raise VerificationRecoveryError("Regeneration allowance differs from the saved intent")
        for candidate in graph.nodes:
            if candidate.id in affected:
                old = verification_integer(affected[candidate.id], "source generation")
                if node_generation(candidate) not in (old, old + 1):
                    raise VerificationRecoveryError("Regeneration generation identity changed")
                if (node_generation(candidate) == old + 1
                        and (candidate.status is not NodeStatus.PENDING or candidate.result is not None)):
                    raise VerificationRecoveryError("Regeneration intent would overwrite a newer result")
                if candidate is not node and "regeneration_intent" in candidate.metadata:
                    raise VerificationRecoveryError("Overlapping regeneration intents require reconciliation")
        target_generation = verification_integer(intent.get("target_generation"), "intent target generation")
        judge_generation = verification_integer(intent.get("judge_generation"), "intent judge generation")
        if target_generation != affected[target.id] or judge_generation != affected[node.id]:
            raise VerificationRecoveryError("Regeneration source identity is inconsistent")

    def apply_regeneration(self, node: Node, graph: ExecutionGraph, intent: dict) -> None:
        """Commit an idempotent reset after every affected worker has settled.

        Under a strict policy each discarded generation is first recorded
        under ``rejected_generations`` (see :meth:`_rejected_records`).
        """
        self._validate_regeneration(node, graph, intent)
        target = self._cached_node_by_id(intent["target_id"], graph)
        reset = [n for n in graph.nodes if n.id in intent["affected_generations"]]
        if any(n.metadata.get("accounting_invalid") for n in reset):
            raise BudgetValidationError("Cannot regenerate unresolved provider accounting")
        if any("response_error" in n.metadata for n in reset):
            raise ProviderResponseError("Cannot regenerate an unresolved native response failure")
        retained = self._rejected_records(node, reset, intent, graph) if self._strict else {}
        for candidate in reset:
            candidate.status = NodeStatus.PENDING
            candidate.result = None
            candidate.metadata["execution_generation"] = intent["affected_generations"][candidate.id] + 1
            for key in ("artifacts", "artifacts_discarded", "verification_receipt",
                        "verification_target_generation", ADVISORY_VERDICT_KEY,
                        CONTEXT_SNAPSHOT_KEY):
                candidate.metadata.pop(key, None)
            if candidate.id in retained:
                listed = candidate.metadata.get(REJECTED_GENERATIONS_KEY)
                candidate.metadata[REJECTED_GENERATIONS_KEY] = [
                    *(listed if isinstance(listed, list) else ()), retained[candidate.id],
                ]
        node.metadata.pop("regeneration_intent", None)
        self.notify_control_update(node)
        self._tracer.on_regeneration(
            node, target, reason=intent["reason"], attempt=intent["regenerations_used"],
            limit=node.max_regenerations, reset_ids=[n.id for n in reset],
        )
        for candidate in reset:
            self.notify_update(candidate)
        # Before the reset work runs again, so it cannot spend its checks' allowance.
        self._hold_pending_verifiers(reset)

    def _rejected_records(
        self, judge: Node, reset: list[Node], intent: dict, graph: ExecutionGraph,
    ) -> dict[str, dict]:
        """Strict mode: a record of each generation this reset discards, by node id.

        A record names the generation, its status, the judge and its reason,
        the result's SHA-256 and UTF-8 size, an ``excerpt`` of at most
        ``REJECTED_EXCERPT_BYTES``, a ``sidecar`` file holding a longer
        result in full when there is an ``artifact_dir``, and the
        generation's artifact records, whose files strict publication keeps.
        Once the graph's records would pass ``REJECTED_GENERATIONS_MAX_BYTES``
        a record keeps no excerpt, and one that still does not fit is not
        kept. A generation that a replayed reset already discarded, or that
        produced nothing, gets no record. Sidecars are written before any
        node changes, so a failed write leaves the graph as it was.
        """
        used = _retained_bytes(graph)
        records: dict[str, dict] = {}
        written: list[Path] = []
        try:
            for candidate in reset:
                generation = intent["affected_generations"][candidate.id]
                listed = candidate.metadata.get(REJECTED_GENERATIONS_KEY)
                if (node_generation(candidate) != generation
                        or (candidate.result is None and not candidate.metadata.get("artifacts")
                            and "artifacts_discarded" not in candidate.metadata)
                        or isinstance(listed, list) and any(
                            isinstance(item, dict) and item.get("generation") == generation
                            for item in listed)):
                    continue
                record, data = self._rejected_record(candidate, judge, intent["reason"], generation)
                if len(data) > REJECTED_EXCERPT_BYTES and self._artifact_dir is not None:
                    name = f"{_safe_filename_component(candidate.id)}_g{generation}_rejected.txt"
                    written.append(publish_unique(self._artifact_dir, name, data))
                    record["sidecar"] = str(written[-1].resolve())
                size = _checkpoint_bytes(record)
                if used + size > REJECTED_GENERATIONS_MAX_BYTES:
                    del record["excerpt"]
                    size = _checkpoint_bytes(record)
                if used + size > REJECTED_GENERATIONS_MAX_BYTES:
                    if "sidecar" in record:
                        written.pop().unlink(missing_ok=True)
                    continue
                used += size
                records[candidate.id] = record
        except OSError as exc:
            for path in written:
                path.unlink(missing_ok=True)
            raise VerificationRecoveryError("Could not retain a rejected generation") from exc
        return records

    @staticmethod
    def _rejected_record(
        candidate: Node, judge: Node, reason: str, generation: int,
    ) -> tuple[dict, bytes]:
        """The record of *candidate*'s discarded generation and its result's bytes."""
        result = candidate.result
        text = None if result is None else result if isinstance(result, str) else str(result)
        data = b"" if text is None else text.encode("utf-8", "surrogatepass")
        record: dict[str, Any] = {
            "version": 1, "generation": generation, "status": candidate.status.value,
            "judge_id": judge.id, "reason": reason[:300],
            "result_sha256": None if text is None else hashlib.sha256(data).hexdigest(),
            "result_bytes": len(data),
            "excerpt": data[:REJECTED_EXCERPT_BYTES].decode("utf-8", "ignore"),
        }
        artifacts = candidate.metadata.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            record["artifacts"] = json.loads(json.dumps(artifacts, default=str))
        if "artifacts_discarded" in candidate.metadata:
            record["artifacts_discarded"] = candidate.metadata["artifacts_discarded"]
        return record, data

    def recover_verification(self, graph: ExecutionGraph) -> None:
        """Finish saved control transitions before initial scheduling."""
        # Validate every saved intent before changing any node. In particular,
        # malformed overlapping intents must not partially invalidate a graph.
        for node in graph.nodes:
            if "regeneration_intent" in node.metadata:
                self._validate_regeneration(node, graph, node.metadata["regeneration_intent"])
        for node in graph.nodes:
            intent = node.metadata.get("regeneration_intent")
            if intent is not None:
                self.apply_regeneration(node, graph, intent)
        for node in graph.nodes:
            self.maybe_regenerate(node, graph)

    def _inherit_execution_context(
        self, new_nodes: tuple[Node, ...], graph: ExecutionGraph,
    ) -> None:
        """Give supervisor-added nodes the run's model and task context.

        Nodes minted mid-run never passed through ``Swarm.plan``, so
        they carry none of the metadata the executor depends on; without
        this a revision would produce nodes that call the provider with
        an empty model name. A node that names no profile calls the run's
        provider, so a profiled node's model, which belongs to its profile's
        provider, is not taken; one that names a profile takes that
        profile's model, which it calls. A profile it cannot be routed to
        raises ``NoAllowedProfileError`` before any node is changed.
        """
        if not new_nodes:
            return
        model = next(
            (n.metadata["model"] for n in graph.nodes
             if n.metadata.get("model") and PROFILE_KEY not in n.metadata),
            self._default_model,
        )
        context = next(
            (
                n.metadata["task_context"]
                for n in graph.nodes
                if n.metadata.get("task_context")
            ),
            None,
        )
        profiles = [node_profile(node, self._profiles) for node in new_nodes]
        for node, profile in zip(new_nodes, profiles):
            if profile is not None:
                node.metadata.setdefault("model", profile.model)
            elif model is not None:
                node.metadata.setdefault("model", model)
            if self._task is None and context is not None:
                node.metadata.setdefault("task_context", context)
        if self._task is not None:
            self.stamp_task_context(new_nodes, self._task)

    @staticmethod
    def stamp_task_context(nodes: list[Node] | tuple[Node, ...], task: Task) -> None:
        """Keep constraints/data/criteria even when the label repeats the goal."""
        # Serialize source data once per variant, not once per fan-out node.
        full_context = render_task(task)
        same_goal_context = render_task(task, include_goal=False)
        for node in nodes:
            context = same_goal_context if node.label.strip() == task.goal.strip() else full_context
            if context:
                node.metadata["task_context"] = context
            else:
                # A goal-only SimpleArchitect retains its original bare prompt.
                node.metadata.pop("task_context", None)

    def retry_delay_s(self, attempt: int, *, error: BaseException | None = None) -> float:
        """Full-jitter exponential backoff before retry `attempt` (1-based).

        Returns 0 when backoff is disabled (the default) so existing
        RETRY behavior and test timing are unchanged; opt in via
        ``retry_backoff_s`` for rate-limited workloads (429s at wide
        parallel image fan-out). Under a retry policy, the ``Retry-After``
        of *error*, the failure being retried, replaces the backoff when it
        is a transient status error and the policy honors the header, and
        no wait exceeds ``backoff_max_s``.
        """
        if attempt <= 0:
            return 0.0
        policy = self._retry_policy
        if policy is not None and policy.honor_retry_after and error is not None:
            asked = _retry_after_s(error)
            if asked is not None:
                return min(asked, policy.backoff_max_s)
        if not self._retry_backoff_s:
            return 0.0
        if policy is None:
            ceiling = self._retry_backoff_s * (2 ** (attempt - 1))
        else:
            try:
                ceiling = min(math.ldexp(self._retry_backoff_s, attempt - 1), policy.backoff_max_s)
            except OverflowError:
                ceiling = policy.backoff_max_s
        return (self._rng or random).uniform(0, ceiling)

    def _retry_wait_s(self, node: Node, attempt: int, error: BaseException | None) -> float:
        """The wait before *attempt*; a retry policy's deadline starts at the first.

        The first attempt in this process has no *error* before it: attempt
        0, or a resumed node's first unspent attempt. Raises
        ``RetryDeadlineError``, from *error*, when the wait would end at or
        after the node's deadline, so that retry is not made. The trace
        still records the refused attempt, with no duration and no call, as
        it records an attempt the deadline ends: its span says why the node
        stopped.
        """
        policy = self._retry_policy
        if policy is None:
            return self.retry_delay_s(attempt)
        if error is None:
            self._deadlines.pop(node.id, None)
            if policy.deadline_s is not None:
                self._deadlines[node.id] = self._tracer.now() + policy.deadline_s
            return 0.0
        delay = self.retry_delay_s(attempt, error=error)
        deadline = self._deadlines.get(node.id)
        if deadline is not None and self._tracer.now() + delay >= deadline:
            refused = self._deadline_error(node)
            self._tracer.on_node_start(node, attempt=attempt)
            self._tracer.on_node_error(node, refused)
            self._end_failed_attempt(node)
            raise refused from error
        return delay

    def _deadline_error(self, node: Node) -> RetryDeadlineError:
        return RetryDeadlineError(
            f"Node {node.id!r} reached its retry deadline of "
            f"{self._retry_policy.deadline_s}s before an attempt succeeded"
        )

    def _attempts_spent(self, node: Node) -> int:
        """The attempts *node* already used in its current generation, under ``persist_attempts``.

        With ``attempts_spent``, the journal's count, read each time the node starts.
        """
        if self._journal_attempts is not None:
            return self._journal_attempts(node)
        generation, used = self._attempts_used.get(node.id, (None, 0))
        return used if self._persist_attempts and generation == node_generation(node) else 0

    def _spend_attempt(self, node: Node, attempt: int) -> None:
        """Under ``persist_attempts``, save *attempt* as spent before it is sent.

        The save is not batched. When it fails, the attempt is not counted and
        its error stops the node without the call.
        """
        if not self._persist_attempts:
            return
        before = self._attempts_used.get(node.id)
        self._attempts_used[node.id] = (node_generation(node), attempt + 1)
        try:
            if self._on_control_update is not None:
                self._on_control_update()
            else:
                self.notify_update(node)
        except BaseException:
            if before is None:
                del self._attempts_used[node.id]
            else:
                self._attempts_used[node.id] = before
            raise

    @staticmethod
    def _allowance_spent(node: Node, attempts: int) -> RuntimeError:
        return RuntimeError(
            f"Node {node.id!r} already used all {attempts} attempts its retry policy allows "
            "in this generation"
        )

    def _may_retry(self, error: Exception) -> bool:
        """Whether a failed attempt may be retried: under a policy, only a listed class."""
        policy = self._retry_policy
        return policy is None or classify_failure(error) in policy.retry_classes

    def _attempt_timeout(self, node: Node) -> tuple[float | None, bool]:
        """The attempt's timeout, and whether the retry deadline set it.

        Under a deadline the timeout is ``timeout_s`` or the time left,
        whichever is shorter; with no time left the attempt is not started.
        """
        deadline = self._deadlines.get(node.id)
        if deadline is None:
            return node.timeout_s, False
        remaining = deadline - self._tracer.now()
        if remaining <= 0:
            raise self._deadline_error(node)
        if node.timeout_s is not None and node.timeout_s < remaining:
            return node.timeout_s, False
        return remaining, True

    def _end_failed_attempt(self, node: Node) -> None:
        """Close the span of an attempt that raised, recording it as failed.

        The node itself stays in flight until its failure policy settles
        it, so a retry backoff or a cancellation still sees it running.
        """
        status = node.status
        node.status = NodeStatus.FAILED
        try:
            self._tracer.on_node_end(node)
        finally:
            node.status = status

    def _reserve_node(self, node: Node) -> None:
        """Reserve this executor's pre-call estimate for one node."""
        self.reserve_node_budget(node)

    def _node_route(self, node: Node) -> tuple[Provider, Any, ModelProfile | None]:
        """The provider and model *node* calls, and the profile that chose them, if any.

        Raises ``NoAllowedProfileError`` for a profile this executor was not
        given, or for a node that sets another model than its profile's.
        """
        profile = node_profile(node, self._profiles)
        if profile is None:
            return self._provider, node.metadata.get("model", ""), None
        return profile.provider, profile.model, profile

    def _node_estimate(
        self, node: Node, default_estimated_tokens: int | None,
    ) -> tuple[float | None, bool]:
        """The node's pre-call estimate, and whether its provider requires an explicit one.

        A profiled node's estimate is its profile's ceiling unless it sets its own.
        """
        estimate = node.metadata.get("estimated_cost_usd")
        provider, model, profile = self._node_route(node)
        model = str(model)
        requires_explicit = provider.requires_explicit_budget_estimate(model)
        if estimate is None:
            estimate = (provider.budget_estimate_usd(model) if profile is None
                        else profile.max_cost_per_call_usd)
        if estimate is None and not requires_explicit and default_estimated_tokens is not None:
            tokens = validate_token_count(default_estimated_tokens, "default_estimated_tokens")
            try:
                estimate = tokens * self._budget.cost_per_token
            except OverflowError as exc:
                raise BudgetValidationError("Token-derived estimate must remain finite") from exc
        return estimate, requires_explicit

    def reserve_node_budget(
        self,
        node: Node,
        *,
        default_estimated_tokens: int | None = None,
    ) -> None:
        """Reserve a defensible pre-call cost for one node.

        Node metadata wins, then the provider's model-aware estimate.  A
        generic token fallback is allowed only for providers that declare it
        safe.  Serial execution passes no fallback to preserve its historical
        pre-call behavior for ordinary text providers while still reserving
        image calls. A node with a budget hold (see
        :meth:`hold_verifier_budget`) converts it into its reservation, and
        without an estimate of its own reserves the held amount.
        """
        if self._budget is None or self._workflow_managed:
            return

        estimate, requires_explicit = self._node_estimate(node, default_estimated_tokens)
        if estimate is None and requires_explicit:
            if self._budget.max_budget_usd is not None:
                raise BudgetEstimateRequired(
                    node.id, str(node.metadata.get("model", "")), type(self._provider).__name__,
                )
            # Unlimited execution may proceed without a price, but must not
            # consume a fake text-token reservation and report it as complete.
            # The provider result will mark its USD cost unknown.
            return

        holds = self._budget.holds
        if estimate is None:
            estimate = holds.get(node.id)
        if estimate is None:
            self._budget.check(node.id)
            return
        self._budget.reserve(
            node.id, estimate, hard_ceiling=requires_explicit, from_hold=node.id in holds,
        )
        self._reserved_node_ids.add(node.id)

    # The token estimate a verifier's hold falls back on when neither the
    # node nor its provider gives one; AsyncExecutor's estimated_tokens_per_node
    # replaces it. hold_verifier_budget() turns on holding for regenerations.
    _estimated_tokens_per_node = 2000
    _hold_verifiers = False

    def hold_verifier_budget(self, graph: ExecutionGraph) -> None:
        """Hold each pending verifier's estimate so that other nodes cannot spend it.

        A hold is what the async executor would reserve for the verifier, and
        the verifier converts it into its reservation when it is admitted.
        From then on, a regeneration holds the estimate of each verifier it
        makes pending again, so a rerun converts a hold as well. Raises
        ``SentinelAlert`` when the budget cannot cover a hold. Durable runs,
        whose journal admits each call, hold nothing.
        """
        if self._budget is None or self._workflow_managed:
            return
        self._hold_verifiers = True
        self._hold_pending_verifiers(graph.nodes)

    def _hold_pending_verifiers(self, nodes: list[Node]) -> None:
        if not self._hold_verifiers:
            return
        held = self._budget.holds
        for node in nodes:
            if node.verifies and node.status is NodeStatus.PENDING and node.id not in held:
                estimate = self._node_estimate(node, self._estimated_tokens_per_node)[0]
                if estimate is not None:
                    self._budget.hold(node.id, estimate)

    def notify_update(self, node: Node) -> None:
        """Invoke the node-update hook (used for checkpointing) if one is set.

        Called whenever a node reaches a terminal status: COMPLETED,
        SKIPPED, or FAILED.
        """
        if self._on_node_update is not None:
            try:
                self._on_node_update(node)
            except WorkflowError:
                raise
            except Exception as exc:
                if self._workflow_managed:
                    raise WorkflowError("Could not persist managed node completion") from exc
                if node.status is NodeStatus.COMPLETED and "verification_receipt" in node.metadata:
                    raise VerificationRecoveryError("Could not persist completed verifier") from exc
                raise

    def mark_accounting_invalid(self, node: Node, error: BudgetValidationError) -> None:
        """Persist the need for reconciliation before this run can resume."""
        node.metadata["accounting_invalid"] = True
        node.metadata["accounting_error"] = str(error)
        if self._workflow_managed and not isinstance(error, ProviderResponseError):
            node.metadata["workflow_accounting_invalid"] = True

    def prepare_graph(self, graph: ExecutionGraph) -> None:
        """Cache immutable graph structure used throughout an execution.

        Node statuses, results, and metadata change while a graph runs, but
        node IDs and dependency edges do not.  Building these indexes once
        keeps wide DAGs from repeatedly scanning every node while assembling
        dependency prompts or deciding whether a node is terminal.
        """
        if self._prepared_graph is not graph:
            # Reusing an executor must not carry a previous graph's source
            # material into a later taskless graph. Only the constructor Task
            # is a default; the current graph binding is scoped to that graph.
            task = graph.task if graph.task is not None else self._default_task
            self._task = snapshot_task(task) if task is not None else None
            if self._task is not None:
                graph.task = snapshot_task(self._task)
                self.stamp_task_context(graph.nodes, self._task)
        self._prepared_graph = graph
        self._node_lookup = {node.id: node for node in graph.nodes}
        if self._prompt_assembly == 1:
            depended_on = {dep_id for node in graph.nodes for dep_id in node.depends_on}
            self._terminal_ids = {node.id for node in graph.nodes} - depended_on
        else:
            self._terminal_ids = _deliverable_ids(graph.nodes)

    def prepare_execution(self, graph: ExecutionGraph) -> None:
        """Bind a new run even when a caller reuses the same graph object."""
        if not self._workflow_managed and any("response_error" in node.metadata for node in graph.nodes):
            raise ProviderResponseError(
                "Native response failure requires explicit accounting reconciliation and local repair"
            )
        if self._workflow_managed:
            for node in graph.nodes:
                if ("workflow_accounting_invalid" in node.metadata
                        or node.metadata.get("accounting_invalid") and "response_error" not in node.metadata):
                    raise BudgetValidationError("Unjournaled accounting requires reconciliation")
        if self._max_revisions > self._revisions_used:
            check_supervisor_profiles(self._supervisor, self._profiles)
        if self._retry_policy is not None:
            pending = [node for node in graph.nodes
                       if node.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)]
            routes = [self._node_route(node) for node in pending]
            calls = [(self._provider, model) for model in sorted(
                {str(model) for _, model, profile in routes if profile is None})]
            # Every profile, not only those pending nodes name: a supervisor may add one.
            calls += [(profile.provider, profile.model) for profile in self._profiles.values()]
            from smythe.supervisor import _model_reviewer

            # A supervisor that may still revise reviews with its own model or
            # else the reviewed node's, which can be any node of the graph.
            supervisor = _model_reviewer(self._supervisor)
            if supervisor is not None and self._max_revisions > self._revisions_used:
                models = {supervisor._model} if supervisor._model else {
                    str(node.metadata.get("model", "")) for node in graph.nodes}
                calls += [(supervisor._provider, model) for model in sorted(models)]
            for provider, model in calls:
                check_provider_support(provider, model, node_id=None, check_model=False,
                                       retry_owner=True)
        if self._capacity_policy is not None:
            self._capacity = _CapacityLimiter(
                self._capacity_policy, self._tracer, self._sleep or asyncio.sleep,
            )
            # Read every hint the pending nodes need, so a bad one stops the run before any call.
            # A profiled node's limit is its profile's provider and model.
            for node in graph.nodes:
                if node.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
                    provider, model, _ = self._node_route(node)
                    self._capacity.bucket(provider, str(model))
        self._reserved_node_ids.clear()
        self._dispatched.clear()
        self._call_attempts.clear()
        self._deadlines.clear()
        self._attempts_used, self._attempts_seed = self._attempts_seed, {}
        self._settled_response_errors.clear()
        self._cleared_response_errors.clear()
        self._response_settlement_failures.clear()
        self._prepared_graph = None
        self.prepare_graph(graph)

    def _ensure_graph_prepared(self, graph: ExecutionGraph) -> None:
        if self._prepared_graph is not graph:
            self.prepare_graph(graph)

    def _cached_node_by_id(self, node_id: str, graph: ExecutionGraph) -> Node | None:
        self._ensure_graph_prepared(graph)
        return self._node_lookup.get(node_id)

    def _is_terminal(self, node: Node, graph: ExecutionGraph) -> bool:
        self._ensure_graph_prepared(graph)
        return node.id in self._terminal_ids

    @staticmethod
    def build_system_prompt(agent: Agent | None) -> str:
        if agent and agent.profile.persona:
            return agent.profile.persona
        return "You are a helpful assistant completing a step in a larger task."

    @staticmethod
    def build_user_prompt(
        node: Node, dep_results: dict[str, Any], *, is_terminal: bool = False,
    ) -> str:
        task_context = (
            node.metadata.get("task_context") if INCLUDE_TASK_CONTEXT else None
        )
        if task_context:
            parts = [f"Overall task:\n{task_context}", f"\nYour step: {node.label}"]
        else:
            parts = [node.label]
        if dep_results:
            parts.append("\n\nContext from prior steps:")
            for dep_id, result in dep_results.items():
                parts.append(f"\n[{dep_id}]: {result}")
            if is_terminal and TERMINAL_DELIVERABLE_NOTE:
                parts.append("\n" + TERMINAL_DELIVERABLE_NOTE)
        return "\n".join(parts)

    @staticmethod
    def build_context_prompt(
        snapshot: ContextSnapshot, task: Task | None, dep_results: dict[str, Any], *,
        is_terminal: bool = False,
    ) -> str:
        """Render a node prompt from its context snapshot (context assembly version 1).

        The task's requirements and the step come first, as instructions.
        The task's context data, each source and each dependency's output
        follow as untrusted blocks (``smythe.context.untrusted_block``) after
        a note that they are material, not instructions. The task and every
        dependency output must be the ones whose digests the snapshot holds,
        or ``WorkflowStateError`` is raised before anything is sent. A
        terminal node with sources or dependencies also gets
        ``TERMINAL_DELIVERABLE_NOTE``.
        """
        where = f"the context snapshot of node {snapshot.node_id!r}"
        if task_sha256(task) != snapshot.task_sha256:
            raise WorkflowStateError(f"The task differs from {where}")
        outputs: dict[str, str] = {}
        for digest in snapshot.dependencies:
            value = dep_results.get(digest.node_id)
            text = value if isinstance(value, str) else str(value)
            if digest.node_id not in dep_results or text_sha256(text) != digest.output_sha256:
                raise WorkflowStateError(f"The output of step {digest.node_id!r} differs from {where}")
            outputs[digest.node_id] = text
        if outputs.keys() != dep_results.keys():
            raise WorkflowStateError(f"The node's dependencies differ from {where}")
        parts = [f"Your step: {snapshot.step}"]
        blocks = []
        if task is not None:
            parts.insert(0, "Overall task:\n" + render_task(task, include_context=False))
            if task.context:
                blocks.append(untrusted_block("task context (JSON)", _prompt_json(task.context)))
        blocks.extend(untrusted_block(source.label, source.text) for source in snapshot.sources)
        blocks.extend(untrusted_block(f"output of step {json.dumps(node_id)}", text)
                      for node_id, text in outputs.items())
        if blocks:
            parts += [UNTRUSTED_DATA_NOTE, *blocks]
        if is_terminal and (outputs or snapshot.sources) and TERMINAL_DELIVERABLE_NOTE:
            parts.append(TERMINAL_DELIVERABLE_NOTE)
        return "\n\n".join(parts)

    @staticmethod
    def node_by_id(node_id: str, graph: ExecutionGraph) -> Node | None:
        return next((n for n in graph.nodes if n.id == node_id), None)

    def deps_satisfied(self, node: Node, graph: ExecutionGraph) -> bool:
        """True if every dependency is COMPLETED or SKIPPED."""
        for dep_id in node.depends_on:
            dep = self._cached_node_by_id(dep_id, graph)
            if dep is None or dep.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED):
                return False
        return True

    async def acall_node(self, node: Node, graph: ExecutionGraph) -> CompletionResult:
        """Run the node's conversation (including any tool loop) under timeout_s.

        Owns cost recording: every provider call inside is billed via
        Sentinel.add_cost, so executors must not record costs again. Under
        a retry deadline the timeout is at most the time left, and reaching
        it raises ``RetryDeadlineError``. A ``TimeoutError`` the conversation
        raises itself, such as an HTTP client's own timeout, is an ordinary
        timeout even then. Under ``retain_uncertain_dispatch``, a timeout
        while a provider call is unanswered raises ``UncertainDispatchError``
        (the deadline's still raises ``RetryDeadlineError``), and however the
        attempt ends, such a call keeps its reservation as unknown cost.
        """
        timeout, by_deadline = self._attempt_timeout(node)
        coro = self._run_node_conversation(node, graph)
        deadline_timer = None
        try:
            if timeout is None:
                return await coro
            if not by_deadline:
                return await asyncio.wait_for(coro, timeout=timeout)
            # A scope of its own tells the deadline's expiry apart from a
            # TimeoutError raised inside the conversation.
            async with asyncio.timeout(timeout) as deadline_timer:
                return await coro
        except BaseExceptionGroup as exc:
            if not _native_response_errors(exc):
                _raise_workflow_error(exc)
                self._raise_unresolved_response(node)
            _settle_response_group(
                exc, lambda error: self._record_response_error(node, error, phase="tool"),
            )
        except ProviderResponseError as exc:
            # Provider and tool-call boundaries already settled their errors.
            # A tool context can also fail during connection or teardown.
            self._record_response_error(node, exc, phase="tool")
            raise
        except WorkflowError:
            raise
        except TimeoutError as exc:
            # wait_for or the deadline's timer can wrap a native CancelledError
            # subclass on timeout. Its unknown bill remains terminal,
            # regardless of retry policy.
            if isinstance(exc.__cause__, ProviderResponseError):
                raise exc.__cause__
            if deadline_timer is not None and deadline_timer.expired():
                raise self._deadline_error(node) from None
            if node.id in self._dispatched:
                raise UncertainDispatchError(
                    f"Node {node.id!r} timed out after {node.timeout_s}s with its provider call "
                    "unanswered; the call's outcome and charge are unknown"
                ) from None
            raise TimeoutError(
                f"Node {node.id!r} timed out after {node.timeout_s}s"
            ) from None
        except Exception:
            self._raise_unresolved_response(node)
            raise
        finally:
            self._retain_uncertain_dispatch(node)

    async def _run_node_conversation(
        self, node: Node, graph: ExecutionGraph,
    ) -> CompletionResult:
        if READ_STEP_KEY in node.metadata:
            return await self._run_read_step(node)
        agent = self._registry.get(node.agent_id) if node.agent_id else None
        dep_results = self.gather_dep_results(node, graph)
        system = self.build_system_prompt(agent)
        if self._context_assembler is None:
            prompt = self.build_user_prompt(
                node, dep_results, is_terminal=self._is_terminal(node, graph),
            )
        else:
            snapshot = await self._context_assembler(node, graph, dep_results)
            prompt = self.build_context_prompt(
                snapshot, self._task, dep_results, is_terminal=self._is_terminal(node, graph),
            )
        model = self._node_route(node)[1]
        attachments = (
            self.load_dep_image_artifacts(node, graph)
            if node.attach_dep_artifacts
            else []
        )
        messages = [ChatMessage(role="user", content=prompt, attachments=attachments)]

        if self._tool_runtime is None:
            # A retry after truncated output follows a billed call.
            self._readmit_node_budget(node)
            result = await self._provider_chat(node, system, messages, model)
            self._record_cost(node, result)
            # Only after the charge is recorded: incomplete output is billed.
            _raise_if_incomplete(result, where=f"Node {node.id!r}")
            return result

        collected_artifacts: list = []
        # Counted per attempt, for supervisor triggers that read the node.
        tool_errors = 0
        node.metadata.pop(TOOL_ERRORS_KEY, None)
        async with self._tool_runtime.open(agent) as session:
            tools = list(session.tools) or None
            limit = node.max_tool_iterations or self._max_tool_iterations
            for turn in range(limit):
                self._readmit_node_budget(node)
                result = await self._provider_chat(node, system, messages, model, tools=tools, turn=turn)
                self._record_cost(node, result)
                # A truncated, refused or filtered turn may carry half-written
                # or blocked tool calls, so it is rejected (after billing)
                # before any tool runs.
                _raise_if_incomplete(result, where=f"Node {node.id!r}")
                # Artifacts on intermediate turns are already billed —
                # carry them to the final result so they get persisted.
                if result.artifacts:
                    collected_artifacts.extend(result.artifacts)

                if result.stop_reason == "pause_turn" and not result.tool_calls:
                    # Provider paused a server-side loop; re-send to continue.
                    messages.append(ChatMessage(
                        role="assistant", content=result.text,
                        provider_continuation=result.provider_continuation,
                    ))
                    continue
                if not result.tool_calls:
                    if len(collected_artifacts) != len(result.artifacts):
                        result.artifacts = list(collected_artifacts)
                    return result

                messages.append(ChatMessage(
                    role="assistant",
                    content=result.text,
                    tool_calls=list(result.tool_calls),
                    provider_continuation=result.provider_continuation,
                ))
                tool_results: list[ToolResult] = []
                for tc in result.tool_calls:
                    started = time.monotonic()
                    try:
                        outcome = await session.call(tc)
                    except BaseExceptionGroup as exc:
                        if _native_response_errors(exc):
                            raise  # acall_node settles grouped accounting.
                        _raise_workflow_error(exc)
                        outcome = ToolResult(tool_call_id=tc.id, content=str(exc), is_error=True)
                    except ProviderResponseError as exc:
                        self._record_response_error(node, exc, phase="tool")
                        raise
                    except BudgetValidationError:
                        # A tool may itself call a paid provider. Numeric
                        # accounting failures must not become retryable model
                        # feedback that dispatches another provider turn.
                        raise
                    except WorkflowError:
                        raise
                    except Exception as exc:
                        # Tool failures go back to the model, not up the stack —
                        # it can adapt or try another tool.
                        outcome = ToolResult(
                            tool_call_id=tc.id, content=str(exc), is_error=True,
                        )
                    duration_ms = (time.monotonic() - started) * 1000
                    self._tracer.on_tool_call(node, tc.name, duration_ms, outcome.is_error)
                    if outcome.is_error:
                        tool_errors += 1
                        node.metadata[TOOL_ERRORS_KEY] = tool_errors
                    if self._context_assembler is not None:
                        # What a tool returns is material, like a source.
                        outcome = ToolResult(
                            outcome.tool_call_id, is_error=outcome.is_error,
                            content=untrusted_block(f"result of tool {json.dumps(tc.name)}",
                                                    str(outcome.content)),
                        )
                    tool_results.append(outcome)
                messages.append(ChatMessage(role="user", tool_results=tool_results))

        self._raise_unresolved_response(node)
        raise ToolLoopLimitError(
            f"Node {node.id!r} hit max_tool_iterations={limit} without completing"
        )

    async def _run_read_step(self, node: Node) -> CompletionResult:
        """Run a declared read step through its journaled read; no model is called.

        The trace records it as a tool call named ``read:<profile>``, with
        its duration and whether it failed, but not its input or text.
        """
        if self._read_call_factory is None:
            raise ValueError(
                f"Node {node.id!r} declares a read step, which runs only in a durable "
                "workflow: pass run_store and read_tools to Swarm"
            )
        step = node.metadata[READ_STEP_KEY]
        name = step.get("profile") if isinstance(step, dict) else None
        read = self._read_call_factory(node, self._call_attempts.get(node.id, 0))
        started, failed = time.monotonic(), True
        try:
            result = await read.read()
            failed = False
        finally:
            self._tracer.on_tool_call(node, f"read:{name}", (time.monotonic() - started) * 1000, failed)
        self._record_cost(node, result, phase="tool")
        return result

    def _readmit_node_budget(self, node: Node) -> None:
        """Admit another call for a node whose reservation a billed call used.

        Reconciling a charge consumes the node's reservation, so a later call
        for the same node (a tool turn, or a retry after truncated output)
        reserves the same estimate its first call did.  A bare check would
        let concurrent reservations or an explicit estimate be overrun.
        """
        if self._budget and not self._workflow_managed and node.id not in self._reserved_node_ids:
            self._reserve_node(node)

    async def _provider_chat(self, node, system, messages, model, tools=None, *, turn=0):
        self._raise_unresolved_response(node)
        phase = "verification" if node.verifies else "execution"
        provider = self._node_route(node)[0]
        if self._provider_call_factory is not None:
            provider = self._provider_call_factory(node, phase, self._call_attempts.get(node.id, 0), turn)
            if getattr(provider, "workflow_managed", False) is not True:
                raise WorkflowError("The provider call factory must return a journal-managed provider")
        try:
            async with self._capacity_call(node, provider, model):
                if tools is None:
                    return await self._send(node, provider.chat(system, messages, model))
                return await self._send(
                    node, provider.chat(system, messages, model, tools=tools))
        except BaseExceptionGroup as exc:
            if not _native_response_errors(exc):
                _raise_workflow_error(exc)
            _settle_response_group(
                exc, lambda error: self._record_response_error(node, error, phase=phase),
            )
        except ProviderResponseError as exc:
            # A classified error is raised without a cause: a run that fails
            # with it must not save the settled error's unresolved marker.
            rejection = self._settle_rejection(node, exc, phase)
            if rejection is not None:
                raise rejection from None
            self._record_response_error(node, exc, phase=phase)
            defect = self._release_known_charge(node, exc)
            if defect is not None:
                raise defect from None
            raise

    async def _send(self, node, call):
        """Await a provider *call*, marking *node* dispatched until the provider answers.

        Only under ``retain_uncertain_dispatch``. A result or an exception the
        provider raises, such as its own error or its client's timeout,
        answers the call. A cancellation, from the node's timeout or
        otherwise, or an interruption leaves the node marked, and the attempt
        then keeps its reservation (:meth:`_retain_uncertain_dispatch`).
        """
        if not self._retain_dispatch:
            return await call
        self._dispatched.add(node.id)
        try:
            result = await call
        except Exception:
            self._dispatched.discard(node.id)
            raise
        self._dispatched.discard(node.id)
        return result

    def _retain_uncertain_dispatch(self, node: Node) -> None:
        """Keep the reservation of *node*'s sent and unanswered call as unknown cost.

        The reservation becomes the node's recorded cost, flagged
        ``cost_usd_unknown`` and, with an amount, ``cost_usd_is_estimate``, so
        a checkpoint keeps it and resume restores it; the next call reserves
        again. Does nothing unless such a call is marked (see :meth:`_send`).
        """
        if node.id not in self._dispatched:
            return
        self._dispatched.discard(node.id)
        if self._budget is None:
            return
        kept = self._budget.retain_unknown(node.id)
        self._reserved_node_ids.discard(node.id)
        node.metadata["cost_usd_unknown"] = True
        if kept is not None:
            node.metadata["cost_usd_is_estimate"] = True
        node.metadata["cost_usd"] = self._budget.breakdown()[node.id]
    def _capacity_call(self, node: Node, provider: Any, model: Any):
        """Wait for one of *node*'s calls to take its turn; nothing without a capacity policy."""
        if self._capacity is None:
            return nullcontext()
        return self._capacity.call(provider, str(model), node)

    def _settle_rejection(self, node, error, phase):
        """Under a retry policy, settle a native plain HTTP rejection at zero cost.

        A durable journal's rules decide (``_http_rejection_receipt``): a 401,
        403, 404, 413 or 429 response with the provider's plain error body.
        Its zero-cost receipt is recorded as the call's charge, and the
        ``ProviderRequestRejectedError`` to raise, whose class follows the
        status, is returned. Any other error returns None and keeps its
        unknown exposure.
        """
        envelope = getattr(error, "envelope", None)
        if (self._retry_policy is None or not isinstance(error, ProviderAccountingError)
                or isinstance(error, asyncio.CancelledError)
                or not isinstance(envelope, RawResponseEnvelope)):
            return None
        try:
            raw = envelope.json()
        except ValueError:
            return None
        recorded = error.receipt if isinstance(error.receipt, dict) else {}
        receipt = _http_rejection_receipt(
            {"status_code": envelope.status_code, "transport_error": envelope.transport_error},
            raw, recorded.get("price_version"),
        )
        if receipt is None:
            return None
        receipt.update(request_sha256=envelope.request_sha256,
                       response_sha256=envelope.response_sha256, request_id=envelope.request_id)
        self._record_cost(node, CompletionResult("", cost_usd=0.0, native_receipt=receipt),
                          phase=phase)
        detail = receipt["error_code"] or receipt["error_type"]
        return ProviderRequestRejectedError(
            f"Provider rejected the request with HTTP {receipt['http_status']}"
            + (f" ({detail})" if detail else "") + "; it was settled at zero cost",
            status_code=receipt["http_status"], envelope=envelope, receipt=receipt,
        )

    def _release_known_charge(self, node, error):
        """Under a policy listing ``quality_defect``, release a settled refusal or truncation.

        A native reply whose ``stop_reason`` is ``refusal`` or ``max_tokens``
        and whose known charge was just recorded has its node marker cleared,
        so the next attempt can dispatch, and the ``NativeOutputDefectError``
        to raise is returned. Any other error, or a policy without that class,
        returns None and stays terminal.
        """
        policy = self._retry_policy
        if (policy is None or QUALITY_DEFECT not in policy.retry_classes
                or error.billing_result is None or isinstance(error, BudgetValidationError)
                or error.stop_reason not in ("refusal", "max_tokens")
                or node.metadata.get("accounting_invalid")):
            return None
        node.metadata.pop("response_error", None)
        self._cleared_response_errors.append(error)
        return NativeOutputDefectError(error.stop_reason, where=f"Node {node.id!r}")

    def _raise_unresolved_response(self, node):
        if self._workflow_managed and "workflow_accounting_invalid" in node.metadata:
            raise BudgetValidationError("Unjournaled accounting requires reconciliation")
        if "response_error" not in node.metadata and not node.metadata.get("accounting_invalid"):
            return
        if node.id in self._response_settlement_failures:
            raise self._response_settlement_failures[node.id]
        cleared = self._cleared_response_errors
        for node_id, error in reversed(self._settled_response_errors):
            # A cleared error was settled and released for another attempt.
            if node_id == node.id and not any(error is done for done in cleared):
                raise error
        if self._workflow_managed and "response_error" in node.metadata:
            # A new managed run re-enters the same immutable journal call.
            # Only that facade can decide whether saved evidence is usable.
            return
        if node.metadata.get("accounting_invalid"):
            raise BudgetValidationError("Unresolved provider accounting requires reconciliation")
        raise ProviderResponseError("Unresolved native response requires reconciliation and repair")

    @property
    def _accounting_budget(self):
        """A managed projection can observe receipts without mutating the ledger."""
        if not self._workflow_managed or getattr(self._budget, "workflow_managed", False) is True:
            return self._budget
        return None

    def release_node_budget(self, node: Node) -> None:
        """Release the node's reservation, except a sent, unanswered call's, which is kept."""
        self._retain_uncertain_dispatch(node)
        if self._budget is not None and not self._workflow_managed:
            self._budget.release(node.id)

    def _sync_managed_cost(self, node: Node) -> None:
        if not self._workflow_managed:
            return
        # Count a logical call once, including a replay after checkpoint lag.
        costs = {}
        for entry in node.metadata.get("native_receipts", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("receipt"), dict):
                raise WorkflowError("Invalid saved managed receipt")
            receipt = entry["receipt"]
            run_id, call_id = receipt.get("workflow_run_id"), receipt.get("workflow_call_id")
            amount = receipt.get("cost_nanousd")
            if isinstance(amount, str) and re.fullmatch(r"0|[1-9][0-9]*", amount):
                amount = int(amount)
            if (receipt.get("workflow_charge_recorded") is True
                    and isinstance(run_id, str) and isinstance(call_id, str)
                    and type(amount) is int and amount >= 0):
                costs[run_id, call_id] = amount
        if costs:
            try:
                node.metadata["cost_usd"] = sum(costs.values()) / 1_000_000_000
            except OverflowError as exc:
                raise WorkflowError("Managed receipt exceeds the display projection range") from exc

    def _record_response_error(self, node, error, *, phase):
        if any(node_id == node.id and settled is error
               for node_id, settled in self._settled_response_errors):
            return
        # Hold the actual objects: re-used Python ids must not hide another
        # failure, and a distinct teardown failure carries its own bill.
        self._settled_response_errors.append((node.id, error))
        try:
            _settle_response_error(
                error, budget=self._accounting_budget, node_id=node.id, metadata=node.metadata, phase=phase,
            )
        except Exception as exc:
            self._response_settlement_failures[node.id] = exc
            raise
        finally:
            self._sync_managed_cost(node)
        if error.billing_result is not None:
            self._reserved_node_ids.discard(node.id)

    def _record_cost(self, node: Node, result: CompletionResult, *, phase: str | None = None) -> None:
        # Custom providers can mutate CompletionResult after construction.
        # Reject malformed usage even when this executor has no Sentinel.
        validate_completion_usage(result)
        _native_receipt(
            node.metadata, result.native_receipt,
            phase=phase or ("verification" if node.verifies else "execution"),
        )
        budget = self._accounting_budget
        if budget is not None:
            cost = budget.add_cost(node.id, result)
            self._reserved_node_ids.discard(node.id)
            if result.cost_usd_unknown:
                node.metadata["cost_usd_unknown"] = True
            if result.cost_usd_is_estimate:
                node.metadata["cost_usd_is_estimate"] = True
            node.metadata["cost_usd"] = cost
        self._sync_managed_cost(node)

    def finalize_node_result(self, node: Node, result: CompletionResult) -> None:
        """Store the node's text result, persisting any artifacts to disk.

        Artifact bytes never land on the node itself — checkpoints are
        plain JSON and planner memory is JSONL, so binary payloads would
        break or bloat both.  Files go under ``artifact_dir`` with a
        filesystem-safe name; each file's absolute path, MIME type,
        ``sha256`` and ``size_bytes`` are recorded in
        ``node.metadata["artifacts"]``, which ``gather_dep_results``
        surfaces to dependent nodes.  ``node.result`` stays the
        provider's text verbatim (downstream JSON parsers rely on it) —
        only when a node produced artifacts and no text does the result
        become a path listing.  With ``artifact_dir=None``, artifacts
        are dropped and counted in ``metadata["artifacts_discarded"]``.

        Under a strict acceptance policy a regenerated node's files are
        named for its generation (``draw_g1_00.png``; the first generation
        keeps ``draw_00.png``) and published without replacing any file:
        a taken name gets a numbered suffix. A rejected generation's files
        therefore keep the bytes that were judged.
        """
        node.result = result.text
        artifacts = result.artifacts
        if not artifacts:
            return
        if self._artifact_dir is None:
            node.metadata["artifacts_discarded"] = len(artifacts)
            if not result.text.strip():
                node.result = (
                    f"[{len(artifacts)} artifact(s) discarded: artifact_dir is None]"
                )
            return
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_filename_component(node.id)
        generation = node_generation(node) if self._strict else 0
        prefix = stem if generation == 0 else f"{stem}_g{generation}"
        records = []
        for i, art in enumerate(artifacts):
            name = f"{prefix}_{i:02d}{art.suffix}"
            data = bytes(art.data) if isinstance(art.data, (bytearray, memoryview)) else art.data
            if self._strict:
                path = publish_unique(self._artifact_dir, name, data)
            else:
                path = self._artifact_dir / name
                self._atomic_write_bytes(path, data)
            records.append({
                "path": str(path.resolve()), "mime_type": art.mime_type,
                "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            })
        node.metadata["artifacts"] = records
        if not result.text.strip():
            node.result = "Generated artifacts:\n" + "\n".join(
                r["path"] for r in records
            )

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """Replace *path* atomically after fully writing a same-dir temp file.

        Artifact generation is commonly checkpointed immediately after this
        method returns.  Writing directly to the final path could therefore
        leave a checkpoint pointing at a truncated file after a crash.  A
        flushed temporary file plus ``os.replace`` gives readers either the
        prior complete artifact or the new complete artifact, never a partial
        write.  Same-directory placement preserves atomic rename semantics.
        """
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    MAX_ATTACHED_IMAGES = 12
    MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024  # per image

    def load_dep_image_artifacts(self, node: Node, graph: ExecutionGraph) -> list:
        """Load dependencies' image artifacts as multimodal attachments.

        Used when ``node.attach_dep_artifacts`` is set — the vision-judge
        pattern: an ArtDirector node that must *see* the images its
        dependencies generated, not just their paths.  Missing or
        oversized files are skipped (the path listing in the prompt
        still names them); count is capped so a wide fan-in can't blow
        the context window.  A file whose bytes no longer match the
        ``sha256`` its record carries raises ``ArtifactIntegrityError``;
        a record without one, written before digests were recorded, is
        attached as read.
        """
        from smythe.provider import Artifact

        attachments: list[Artifact] = []
        for dep_id in node.depends_on:
            dep = self._cached_node_by_id(dep_id, graph)
            if dep is None:
                continue
            for record in dep.metadata.get("artifacts", []):
                if len(attachments) >= self.MAX_ATTACHED_IMAGES:
                    return attachments
                mime = record.get("mime_type", "")
                if not mime.startswith("image/"):
                    continue
                try:
                    data = Path(record["path"]).read_bytes()
                except OSError:
                    continue
                recorded = record.get("sha256")
                if isinstance(recorded, str) and hashlib.sha256(data).hexdigest() != recorded:
                    raise ArtifactIntegrityError(
                        f"Artifact {Path(record['path']).name!r} of node {dep.id!r} changed after "
                        "it was recorded; its bytes no longer match the recorded sha256"
                    )
                if len(data) > self.MAX_ATTACHMENT_BYTES:
                    continue
                attachments.append(Artifact(data=data, mime_type=mime))
        return attachments

    def gather_dep_results(self, node: Node, graph: ExecutionGraph) -> dict[str, Any]:
        """Collect dependency results for prompt building.

        Artifact paths live in dep metadata (not in the result text, so
        JSON results stay parseable); they are appended here so
        downstream nodes can reference the files.  A SKIPPED dependency
        contributes only ``SKIPPED_DEPENDENCY_RESULT``: its recorded
        result is an error message, not output.  A node that exchanges
        claims (``EXCHANGE_KEY``) is given a dependency's compact claim
        record, when its output carries one, in place of the output.
        """
        claims = _exchanges_claims(node)
        dep_results: dict[str, Any] = {}
        for dep_id in node.depends_on:
            dep_node = self._cached_node_by_id(dep_id, graph)
            if dep_node is None:
                continue
            if dep_node.status is NodeStatus.SKIPPED:
                dep_results[dep_id] = SKIPPED_DEPENDENCY_RESULT
                continue
            value = dep_node.result
            if claims:
                value = claim_exchange(value) or value
            records = dep_node.metadata.get("artifacts") or []
            if records and isinstance(value, str) and records[0]["path"] not in value:
                listing = "\n".join(
                    f"- {r['path']} ({r['mime_type']})" for r in records
                )
                value = f"{value}\n\nArtifact files from this step:\n{listing}"
            dep_results[dep_id] = value
        return dep_results
