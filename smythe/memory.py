"""PlannerMemory — persistent execution history for learning-informed planning."""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smythe._failures import _http_status, classify_failure, status_failure_class
from smythe._redact import error_text, redact_paths
from smythe.acceptance import AcceptanceDecision, DeliveryAcceptance, text_sha256
from smythe.task import Task, snapshot_task

# Delivery acceptance decisions an outcome can record.
_ACCEPTANCE_VALUES = frozenset(decision.value for decision in AcceptanceDecision)
# Fields a row holds only for a run that left verification or acceptance
# evidence. Other rows leave them out and keep the keys they had before.
_EVIDENCE_FIELDS = ("execution_id", "verification", "rejected_output_sha256", "structure_sha256")
# Fields only a failed run's row holds, and only when set.
_FAILURE_FIELDS = ("failure_category", "failure_type", "failure_excerpt")
# The most characters of a provider or service error's message a row keeps.
_MAX_FAILURE_EXCERPT = 200
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAILURE_CLASS = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _is_duration(value: Any) -> bool:
    """True for a finite, non-negative int or float."""
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _flag(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_name(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _is_digest(value: Any) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _is_gate_entry(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and all(type(entry.get(key)) is str for key in ("gate", "target", "state"))
        and type(entry.get("passed")) is bool and _is_count(entry.get("regenerations_used"))
    )


def _is_failure_class(value: Any) -> bool:
    return type(value) is str and _FAILURE_CLASS.fullmatch(value) is not None


def _is_type_name(value: Any) -> bool:
    return type(value) is str and value.isidentifier()


def _is_excerpt(value: Any) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_FAILURE_EXCERPT


def _gate_verification(graph: Any) -> list[dict[str, Any]] | None:
    """One entry per gate that holds a verification receipt, or None without any."""
    entries = []
    for node in graph.nodes:
        receipt = node.metadata.get("verification_receipt")
        entry = {
            "gate": node.id,
            "target": node.verifies,
            "state": receipt.get("state") if isinstance(receipt, dict) else None,
            # A gate passed only when its receipt records a pass: an exhausted
            # gate, or one whose target was unavailable, is consumed without one.
            "passed": isinstance(receipt, dict) and receipt.get("passed") is True,
            "regenerations_used": node.metadata.get("regenerations_used", 0),
        }
        if _is_gate_entry(entry):
            entries.append(entry)
    return entries or None


def _structure(graph: Any) -> str | None:
    from smythe.graph import structure_sha256

    try:
        return structure_sha256(graph)
    except (TypeError, ValueError):
        return None  # A shape that is not JSON data has no digest; record the rest.


def _failure_excerpt(exc: BaseException) -> str | None:
    """A provider or service error's message, redacted and capped; None for other errors.

    Other errors can quote task text or local data, so their rows keep only
    the failure class and exception type.
    """
    from smythe.provider import IncompleteOutputError, ProviderResponseError
    from smythe.workflow_provider import ProviderRequestRejectedError
    from smythe.workflow_tools import ReadToolError

    if not isinstance(exc, (
        ProviderResponseError, IncompleteOutputError, ProviderRequestRejectedError, ReadToolError,
    )) and status_failure_class(_http_status(exc)) is None:
        return None
    try:
        text = " ".join(redact_paths(error_text(exc)).split())
    except Exception:  # An unprintable error keeps its class and type.
        return None
    if len(text) > _MAX_FAILURE_EXCERPT:
        text = text[:_MAX_FAILURE_EXCERPT - 1] + "…"
    return text or None


@dataclass
class ExecutionOutcome:
    """Record of a single task execution for planner feedback.

    ``total_duration_ms`` sums recorded node-span times; it is not wall time.
    A node outcome's ``duration_ms`` is the time of that node's final attempt.

    ``elapsed_ms`` is the elapsed time of the call that recorded the outcome.
    ``latency_complete`` is False and ``resumed`` True when that call resumed
    a checkpoint, so ``elapsed_ms`` covers only part of the run. The cost
    flags copy the result's. ``acceptance`` is the decision of the result's
    delivery acceptance record, or ``"not_evaluated"`` when the result has
    no record; ``success`` still means every node completed. These fields
    are None when not recorded, as in history written before they existed.

    A run that left verification or acceptance evidence, meaning a gate's
    verification receipt, a delivery acceptance record or rejected output,
    also records its ``execution_id`` and the
    :func:`~smythe.graph.structure_sha256` of its graph. ``verification``
    has one ``{gate, target, state, passed, regenerations_used}`` entry per
    gate with a receipt; ``passed`` is True only when the receipt records a
    pass, so an exhausted gate or one whose target was unavailable did not
    pass. ``rejected_output_sha256`` is the SHA-256 of the result's
    ``rejected_output``, never its text. These four are None, and left out
    of the history file, when not recorded.

    A failure row, which :meth:`PlannerMemory.record_failure` writes for a
    run that raised, records ``failure_category``, the exception's failure
    class as the executors classify it, ``failure_type``, its class name,
    and for a provider or service error only ``failure_excerpt``: at most
    200 characters of its message, with account identifiers redacted, the
    home folder written ``~`` and other absolute paths ``<path>``.
    ``success`` keeps its meaning there, so a run whose synthesis failed
    after every node completed records True; planning prompts still show
    every failure row's outcome as a failure. These three are None, and left
    out of the history file, for any other row.
    """

    task_goal: str
    task_constraints: list[str]
    topology: list[str]
    node_count: int
    total_cost_usd: float
    total_duration_ms: float
    success: bool
    node_outcomes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    task_context: dict[str, Any] = field(default_factory=dict)
    task_done_when: list[str] = field(default_factory=list)
    elapsed_ms: float | None = None
    latency_complete: bool | None = None
    resumed: bool | None = None
    cost_is_complete: bool | None = None
    cost_contains_estimates: bool | None = None
    acceptance: str | None = None
    execution_id: str | None = None
    verification: list[dict[str, Any]] | None = None
    rejected_output_sha256: str | None = None
    structure_sha256: str | None = None
    failure_category: str | None = None
    failure_type: str | None = None
    failure_excerpt: str | None = None


def _row(saved_task: Any, graph: Any, trace: Any, costs: Any, timing: Any,
         **fields: Any) -> dict[str, Any]:
    """A history row, without the optional fields it does not set.

    *costs* holds the run's cost and cost flags, as a result or a budget does.
    """
    from smythe.graph import NodeStatus

    total_duration = sum(s.get("duration_ms", 0) for s in trace)
    # Each attempt records its own span. A node's duration is its final
    # attempt's, so a retried node reports the attempt that settled it.
    final_spans = {s.get("node_id"): s for s in trace}

    node_outcomes: list[dict[str, Any]] = []
    for node in graph.nodes:
        node_outcome: dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "status": node.status.value if hasattr(node.status, "value") else str(node.status),
        }
        final_span = final_spans.get(node.id)
        if final_span:
            node_outcome["duration_ms"] = final_span.get("duration_ms", 0)
        cost = node.metadata.get("cost_usd")
        if cost is not None:
            node_outcome["cost_usd"] = cost
        node_outcomes.append(node_outcome)

    all_completed = all(
        n.status == NodeStatus.COMPLETED for n in graph.nodes
    )
    timing = timing if isinstance(timing, dict) else {}
    elapsed = timing.get("elapsed_ms")

    outcome = ExecutionOutcome(
        task_goal=saved_task.goal if hasattr(saved_task, "goal") else str(saved_task),
        task_constraints=list(getattr(saved_task, "constraints", [])),
        topology=[t.value for t in graph.topology],
        node_count=len(graph.nodes),
        total_cost_usd=getattr(costs, "total_cost_usd", 0.0),
        total_duration_ms=total_duration,
        success=all_completed,
        node_outcomes=node_outcomes,
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_context=saved_task.context if isinstance(saved_task, Task) else {},
        task_done_when=list(getattr(saved_task, "done_when", [])),
        elapsed_ms=elapsed if _is_duration(elapsed) else None,
        latency_complete=_flag(timing.get("latency_complete")),
        resumed=_flag(timing.get("resumed")),
        cost_is_complete=_flag(getattr(costs, "cost_is_complete", None)),
        cost_contains_estimates=_flag(getattr(costs, "cost_contains_estimates", None)),
        **fields,
    )

    data = asdict(outcome)
    for name in (*_EVIDENCE_FIELDS, *_FAILURE_FIELDS):
        if data[name] is None:
            del data[name]
    return data


class PlannerMemory:
    """Persistent store of execution outcomes.

    Stores one ExecutionOutcome per JSONL line.  The planner queries
    this history to inform topology decisions for new tasks.

    By default only runs that finish are recorded. With
    ``record_failures=True`` a ``Swarm`` also records each run that raises,
    through :meth:`record_failure`.
    """

    def __init__(self, path: str | Path | None = None, *, record_failures: bool = False) -> None:
        if type(record_failures) is not bool:
            raise TypeError(
                f"record_failures must be True or False, not {type(record_failures).__name__}"
            )
        if path is None:
            path = Path.home() / ".smythe" / "history.jsonl"
        self._path = Path(path)
        self._lock = threading.Lock()
        self._record_failures = record_failures

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record_failures(self) -> bool:
        """Whether a Swarm records the runs that fail, as well as those that finish."""
        return self._record_failures

    def record(
        self,
        task: Any,
        graph: Any,
        result: Any,
    ) -> None:
        """Append an execution outcome to the history file."""
        # Capture nested context before inspecting mutable execution results.
        # Legacy duck-typed callers keep their existing goal/constraint path.
        saved_task = snapshot_task(task) if isinstance(task, Task) else task

        trace = getattr(result, "trace", [])
        # Only a record decides; a bare string on a duck-typed result does not.
        record = getattr(result, "acceptance", None)
        acceptance = (
            record.decision.value if isinstance(record, DeliveryAcceptance)
            else AcceptanceDecision.NOT_EVALUATED.value
        )
        verification = _gate_verification(graph)
        rejected = getattr(result, "rejected_output", None)
        rejected_sha256 = text_sha256(rejected) if isinstance(rejected, str) else None
        execution_id = structure = None
        if (verification is not None or rejected_sha256 is not None
                or isinstance(record, DeliveryAcceptance)):
            execution_id = getattr(result, "execution_id", None)
            structure = _structure(graph)

        self._append(_row(
            saved_task, graph, trace, result, getattr(result, "timing", None),
            acceptance=acceptance,
            execution_id=execution_id if _is_name(execution_id) else None,
            verification=verification,
            rejected_output_sha256=rejected_sha256,
            structure_sha256=structure,
        ))

    def record_failure(
        self,
        task: Any,
        graph: Any,
        exc: BaseException,
        *,
        execution_id: str | None = None,
        timing: dict[str, Any] | None = None,
        trace: list[dict[str, Any]] | None = None,
        budget: Any = None,
    ) -> None:
        """Append the outcome of a run that raised *exc* to the history file.

        A ``Swarm`` calls this, when ``record_failures`` is True, for a run
        that raises during execution, synthesis or acceptance, after saving
        its failed checkpoint. The row records the failure fields that
        :class:`ExecutionOutcome` describes, the graph's structure digest,
        any gate verification entries, ``execution_id`` when given and
        ``acceptance`` ``"not_evaluated"``. *timing* and *trace* take the
        forms ``SwarmResult.timing`` and ``SwarmResult.trace`` have, and
        *budget* the run's ``Sentinel``, whose spend and cost flags the row
        records; without them, those fields are left unrecorded or zero.
        """
        saved_task = snapshot_task(task) if isinstance(task, Task) else task
        self._append(_row(
            saved_task, graph, trace or [], budget, timing,
            acceptance=AcceptanceDecision.NOT_EVALUATED.value,
            execution_id=execution_id if _is_name(execution_id) else None,
            verification=_gate_verification(graph),
            structure_sha256=_structure(graph),
            failure_category=classify_failure(exc),
            failure_type=type(exc).__name__,
            failure_excerpt=_failure_excerpt(exc),
        ))

    def _append(self, data: dict[str, Any]) -> None:
        line = (json.dumps(data) + "\n").encode("utf-8")
        with self._lock:
            # History holds task context, so a directory or file created here
            # is owner-only. Existing ones keep whatever mode they have.
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                self._path,
                os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0),
                0o600,
            )
            with open(descriptor, "a+b") as f:
                end = f.seek(0, os.SEEK_END)
                if end:
                    # A crash can leave a partial last line; never append to it.
                    f.seek(end - 1)
                    if f.read(1) != b"\n":
                        line = b"\n" + line
                f.write(line)

    def recall(self, task: Any, k: int = 3) -> list[ExecutionOutcome]:
        """Find the k most relevant past outcomes by keyword overlap."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        if k == 0:
            return []

        query_words = self._tokenize(self._task_text(task))
        if not query_words:
            return []

        with self._lock:
            if not self._path.exists():
                return []
            lines = self._path.read_text(encoding="utf-8").splitlines()

        scored: list[tuple[float, ExecutionOutcome]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                outcome = ExecutionOutcome(**{
                    k_: data[k_]
                    for k_ in ExecutionOutcome.__dataclass_fields__
                    if k_ in data
                })
                if not isinstance(outcome.task_goal, str):
                    continue
                if not isinstance(outcome.task_context, dict):
                    continue
                if any(
                    not isinstance(values, list)
                    or any(not isinstance(value, str) for value in values)
                    for values in (outcome.task_constraints, outcome.task_done_when, outcome.topology)
                ):
                    continue
                if type(outcome.success) is not bool:
                    continue
                if any(
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in (outcome.total_cost_usd, outcome.total_duration_ms)
                ):
                    continue
                if outcome.elapsed_ms is not None and not _is_duration(outcome.elapsed_ms):
                    continue
                if any(value is not None and type(value) is not bool for value in (
                    outcome.latency_complete, outcome.resumed,
                    outcome.cost_is_complete, outcome.cost_contains_estimates,
                )):
                    continue
                if outcome.acceptance is not None and (
                    type(outcome.acceptance) is not str
                    or outcome.acceptance not in _ACCEPTANCE_VALUES
                ):
                    continue
                if outcome.execution_id is not None and not _is_name(outcome.execution_id):
                    continue
                if any(value is not None and not _is_digest(value) for value in (
                    outcome.rejected_output_sha256, outcome.structure_sha256,
                )):
                    continue
                if outcome.verification is not None and (
                    not isinstance(outcome.verification, list)
                    or not all(_is_gate_entry(entry) for entry in outcome.verification)
                ):
                    continue
                if any(value is not None and not valid(value) for value, valid in (
                    (outcome.failure_category, _is_failure_class),
                    (outcome.failure_type, _is_type_name),
                    (outcome.failure_excerpt, _is_excerpt),
                )):
                    continue
            except (ValueError, TypeError, KeyError, OverflowError):
                # JSON integer limits and unformattable huge integers are
                # malformed history rows, not reasons to abort planning.
                continue
            stored_words = self._tokenize(
                " ".join([
                    outcome.task_goal, *outcome.task_constraints, *outcome.task_done_when,
                    json.dumps(outcome.task_context, ensure_ascii=False),
                ])
            )
            overlap = len(query_words & stored_words)
            if overlap > 0:
                scored.append((overlap, outcome))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [outcome for _, outcome in scored[:k]]

    def clear(self) -> None:
        """Remove all stored history."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    @staticmethod
    def _task_text(task: Any) -> str:
        """Return searchable task text, including source context and criteria."""
        if not hasattr(task, "goal"):
            return str(task)
        saved_task = snapshot_task(task) if isinstance(task, Task) else task
        goal = str(saved_task.goal)
        constraints = getattr(saved_task, "constraints", [])
        constraint_text = [str(item) for item in constraints]
        criteria = [str(item) for item in getattr(saved_task, "done_when", [])]
        context = saved_task.context if isinstance(saved_task, Task) else {}
        return " ".join([
            goal, *constraint_text, *criteria, json.dumps(context, ensure_ascii=False),
        ])

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase word tokens, filtering short/stop words."""
        stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                "and", "or", "but", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "it", "its", "this", "that", "i"}
        words = set(re.findall(r"[a-z]+", text.lower()))
        return {w for w in words if len(w) > 2 and w not in stop}
