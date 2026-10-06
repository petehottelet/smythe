"""Tracer — structured observability for every node execution."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from smythe._failures import classify_failure
from smythe._redact import error_text, redact_account_identifiers
from smythe.graph import REVISION_ADDED_KEY, ExecutionGraph, Node, NodeStatus, _deliverable_ids

logger = logging.getLogger("smythe.tracer")

# What prompted a supervisor review: each review follows a completed node.
REVIEW_TRIGGER = "node_completed"
# The revision log an ordinary checkpoint keeps (see revision_log_entry): its
# latest entries, at most this many ids in each of an entry's id lists, and at
# most this many characters of a supervisor's reason or a refusal's detail.
REVISION_LOG_LIMIT = 32
REVISION_LOG_MAX_IDS = 16
REVISION_LOG_TEXT_CHARS = 200
_REVISION_ID_LISTS = ("added", "dropped", "rewired", "retained", "invalidated", "preserved_gates")


# Phases reported by Tracer.timing(). Routing, planning, recovery and
# synthesis are stages a call may not run; the rest accumulate during it.
PHASES = (
    "routing", "planning", "recovery", "execution", "verification",
    "synthesis", "supervision", "checkpoint", "waiting",
)
_STAGES = frozenset({"routing", "planning", "recovery", "synthesis"})
WAIT_REASONS = ("retry_backoff", "admission", "supervision", "capacity", "drain")
_OUTCOMES = {
    NodeStatus.COMPLETED: "completed", NodeStatus.FAILED: "failed",
    NodeStatus.SKIPPED: "skipped", NodeStatus.PENDING: "cancelled",
}


def _seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
        math.isfinite(value) and value >= 0
    ):
        raise ValueError(f"Duration must be finite, non-negative seconds, got {value!r}")
    return value


@dataclass
class Span:
    """A single trace span corresponding to one node execution.

    ``start_time`` and ``end_time`` are wall-clock seconds. The optional
    monotonic bounds, when both are set, give ``duration_ms`` instead, so a
    wall-clock adjustment during the attempt cannot distort it. ``attempt``
    is the 0-based attempt index, ``phase`` is ``execution``,
    ``verification`` or ``synthesis``, and ``outcome`` is ``completed``,
    ``failed``, ``timed_out``, ``skipped`` or ``cancelled``. A failed
    attempt's ``metadata["failure_class"]`` names how it failed.
    """

    node_id: str
    label: str
    agent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    monotonic_start: float | None = None
    monotonic_end: float | None = None
    attempt: int | None = None
    phase: str | None = None
    outcome: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.monotonic_start is not None and self.monotonic_end is not None:
            return (self.monotonic_end - self.monotonic_start) * 1000
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Collects structured execution traces.

    Every node start/end/error emits a Span that can be inspected
    after execution for debugging and planner feedback.

    ``clock`` is a monotonic clock in seconds (default ``time.perf_counter``)
    and ``wall_clock`` a wall clock in seconds (default ``time.time``).
    Monotonic readings are process-local: values from different processes
    are never compared. ``started_at`` is the wall-clock creation time.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self.spans: list[Span] = []
        self._active: dict[str, Span] = {}
        self._clock = clock or time.perf_counter
        self._wall_clock = wall_clock or time.time
        self._started = self._clock()
        self.started_at = self._wall_clock()
        self._phases: dict[str, float] = {}
        self._waits: dict[str, float] = dict.fromkeys(WAIT_REASONS, 0.0)
        # Running span totals, so a checkpoint per node does not rescan them.
        self._totaled: tuple[list[Span], int] = (self.spans, 0)
        self._span_ms: dict[str, float] = {}

    def now(self) -> float:
        """Read this tracer's monotonic clock, in seconds."""
        return self._clock()

    def wall_time(self) -> float:
        """Read this tracer's wall clock, in seconds since the epoch."""
        return self._wall_clock()

    def since(self, started: float) -> float:
        """Monotonic seconds from ``started``, a reading of now(), to now.

        A clock that steps backwards, or rounding in a derived reading, gives
        0.0 rather than a negative duration, so a measurement taken while
        work runs can always be recorded and never fails that work.
        """
        seconds = self._clock() - started
        return seconds if math.isfinite(seconds) and seconds > 0 else 0.0

    def add_phase(self, name: str, seconds: float) -> None:
        """Add measured time to a phase; waits are recorded with on_wait()."""
        if name not in PHASES or name == "waiting":
            raise ValueError(f"Unknown timing phase {name!r}")
        self._phases[name] = self._phases.get(name, 0.0) + _seconds(seconds)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure a block as time spent in ``name``, even when it raises."""
        started = self._clock()
        try:
            yield
        finally:
            self.add_phase(name, self.since(started))

    def on_wait(self, reason: str, seconds: float, node: Node | None = None) -> None:
        """Record time that ready work spent waiting, such as retry backoff."""
        self._waits[reason] = self._waits.get(reason, 0.0) + _seconds(seconds)
        logger.debug(
            "Wait (%s) for %s: %.1fms", reason, node.id if node else "ready work", seconds * 1000,
        )

    def on_node_start(
        self, node: Node, *, attempt: int | None = None, phase: str | None = None,
    ) -> None:
        span = Span(
            node_id=node.id,
            label=node.label,
            agent_id=node.agent_id,
            start_time=self._wall_clock(),
            status="running",
            monotonic_start=self._clock(),
            attempt=attempt,
            phase=phase or ("verification" if node.verifies else "execution"),
        )
        self._active[node.id] = span
        logger.debug("Node started: %s (%s)", node.id, node.label)

    def on_node_end(self, node: Node) -> None:
        span = self._active.pop(node.id, None)
        if span:
            span.monotonic_end = self._clock()
            span.end_time = self._wall_clock()
            span.status = node.status.value
            if not (span.outcome == "timed_out" and node.status is NodeStatus.FAILED):
                span.outcome = _OUTCOMES.get(node.status)
            if "native_receipts" in node.metadata:
                span.metadata["native_receipts"] = list(node.metadata["native_receipts"])
            self.spans.append(span)
            logger.debug(
                "Node finished: %s (%s) — %.1fms",
                node.id,
                node.label,
                span.duration_ms,
            )

    def on_tool_call(
        self, node: Node, tool_name: str, duration_ms: float, is_error: bool,
    ) -> None:
        """Record one tool call on the active span for *node*."""
        span = self._active.get(node.id)
        if span:
            span.metadata.setdefault("tool_calls", []).append({
                "tool": tool_name,
                "duration_ms": round(duration_ms, 1),
                "is_error": is_error,
            })
        logger.debug(
            "Tool call: %s on node %s — %.1fms%s",
            tool_name, node.id, duration_ms, " (error)" if is_error else "",
        )

    def on_revision(
        self,
        node: Node,
        revision: Any,
        *,
        applied: bool,
        detail: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a supervisor revision, whether it was applied or refused.

        Refusals are recorded too: a supervisor that keeps proposing
        invalid changes is a finding, and silently dropping those
        attempts would hide it. The span's ``revision`` record
        (:func:`revision_record`) adds *context*, the decision's effect
        from :func:`revision_context`, when given.
        """
        now = self._wall_clock()
        span = Span(
            node_id=f"supervisor:{node.id}",
            label=getattr(revision, "reason", "") or "plan review",
            agent_id=None,
            start_time=now,
            end_time=now,
            status="revision_applied" if applied else "revision_rejected",
        )
        if detail:
            span.error = detail
        span.metadata["revision"] = revision_record(node, revision, context)
        self.spans.append(span)
        logger.info(
            "Plan revision %s after node %s: %s",
            "applied" if applied else "rejected", node.id, span.label,
        )

    def on_regeneration(
        self,
        verifier_node: Node,
        target: Node,
        *,
        reason: str,
        attempt: int,
        limit: int,
        reset_ids: list[str],
    ) -> None:
        """Record a failed verification sending work back for another try."""
        now = self._wall_clock()
        span = Span(
            node_id=f"verifier:{verifier_node.id}",
            label=reason or f"{target.id} failed verification",
            agent_id=None,
            start_time=now,
            end_time=now,
            status="regeneration",
        )
        span.metadata["regeneration"] = {
            "verifier": verifier_node.id,
            "target": target.id,
            "attempt": attempt,
            "limit": limit,
            "reset": sorted(reset_ids),
        }
        self.spans.append(span)
        logger.info(
            "Verification failed (%s/%s): %s sent %s back — %s",
            attempt, limit, verifier_node.id, target.id, reason or "no reason given",
        )

    def on_acceptance(self, record: Any) -> None:
        """Record the delivery acceptance decision made on the synthesized output.

        The span's status is ``acceptance_<decision>`` and its label the first
        reason; the record itself stays in the span's metadata, so
        ``summary()`` keeps its keys.
        """
        now = self._wall_clock()
        decision = record.decision.value
        span = Span(
            node_id="acceptance:delivery",
            label=record.reasons[0] if record.reasons else f"delivery {decision}",
            agent_id=None,
            start_time=now,
            end_time=now,
            status=f"acceptance_{decision}",
        )
        span.metadata["acceptance"] = record.to_dict()
        self.spans.append(span)
        logger.info("Delivery acceptance: %s — %s", decision, span.label)

    def on_node_error(self, node: Node, exc: Exception) -> None:
        """Record a failed attempt's error and failure class on its active span.

        The class is in ``span.metadata["failure_class"]``, which
        ``summary()`` leaves out, so the summary keeps its keys. The span and
        the log keep the error's message with account identifiers redacted.
        """
        error = error_text(exc)
        span = self._active.get(node.id)
        if span:
            span.error = error
            span.outcome = "timed_out" if isinstance(exc, TimeoutError) else "failed"
            span.metadata["failure_class"] = classify_failure(exc)
        logger.warning("Node error: %s (%s) — %s", node.id, node.label, error)

    def timing(self) -> dict[str, Any]:
        """Process-local elapsed time, phase totals and waits, in milliseconds.

        ``elapsed_ms`` runs from this tracer's creation to now on the monotonic
        clock. ``summed_node_ms`` adds every span's duration, as PlannerMemory's
        ``total_duration_ms`` does. In ``phases_ms``, ``execution`` and
        ``verification`` add their node attempts' spans to any measured
        control time; the others hold measured time. Stages that did not run
        (routing, planning, recovery, synthesis) are None. Phases and waits
        are sums, so concurrent work can make them exceed ``elapsed_ms``.
        """
        spans, counted = self._totaled
        if spans is not self.spans or counted > len(self.spans):
            # The span list was replaced: total it again.
            counted, self._span_ms = 0, {}
        for span in self.spans[counted:]:
            duration = round(span.duration_ms, 1)
            self._span_ms["all"] = self._span_ms.get("all", 0.0) + duration
            if span.phase in ("execution", "verification"):
                self._span_ms[span.phase] = self._span_ms.get(span.phase, 0.0) + duration
        self._totaled = (self.spans, len(self.spans))
        phases = {name: self._phases.get(name) for name in PHASES}
        for name in ("execution", "verification"):
            if name in self._span_ms:
                phases[name] = (phases[name] or 0.0) + self._span_ms[name] / 1000
        phases["waiting"] = sum(self._waits.values())
        return {
            "clock": "monotonic",
            "elapsed_ms": round((self._clock() - self._started) * 1000, 1),
            "summed_node_ms": round(self._span_ms.get("all", 0.0), 1),
            "phases_ms": {
                name: None if seconds is None and name in _STAGES
                else round((seconds or 0.0) * 1000, 1)
                for name, seconds in phases.items()
            },
            "waiting_ms": {
                reason: round(seconds * 1000, 1) for reason, seconds in self._waits.items()
            },
        }

    def summary(self) -> list[dict[str, Any]]:
        """Return spans as plain dicts for serialization / planner feedback."""
        out: list[dict[str, Any]] = []
        for s in self.spans:
            entry: dict[str, Any] = {
                "node_id": s.node_id,
                "label": s.label,
                "agent_id": s.agent_id,
                "status": s.status,
                "duration_ms": round(s.duration_ms, 1),
                "error": s.error,
            }
            if "tool_calls" in s.metadata:
                entry["tool_calls"] = s.metadata["tool_calls"]
            if "revision" in s.metadata:
                entry["revision"] = s.metadata["revision"]
            if "regeneration" in s.metadata:
                entry["regeneration"] = s.metadata["regeneration"]
            if "native_receipts" in s.metadata:
                entry["native_receipts"] = s.metadata["native_receipts"]
            out.append(entry)
        return out


def revision_record(
    node: Node, revision: Any, context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a supervisor decision after *node* proposed and, with *context*, what it left.

    ``change`` and the ``added``, ``dropped`` and ``rewired`` ids describe
    *revision*, which is None when the review raised; *context*, from
    :func:`revision_context`, adds the decision's effect on the graph.
    """
    return {
        "after_node": node.id,
        "change": getattr(revision, "summary", lambda: "unknown")(),
        "added": [n.id for n in getattr(revision, "add_nodes", ())],
        "dropped": list(getattr(revision, "drop_node_ids", ())),
        "rewired": sorted(getattr(revision, "rewire", {})),
        **(context or {}),
    }


def revision_context(
    graph: ExecutionGraph, *, revisions_remaining_after: int | None,
    delivered_before: Iterable[str] | None = None, added: Iterable[str] = (),
    trigger: str = REVIEW_TRIGGER,
) -> dict[str, Any]:
    """The effect of a supervisor decision, read from *graph* after it.

    ``retained`` lists the completed nodes: a revision never changes finished
    work, so the decision keeps every result. ``invalidated`` lists those
    whose result the run delivered before the decision (*delivered_before*,
    the graph's deliverable nodes then; by default now) and no longer does,
    because a step the revision added or rewired depends on them: ``DELIVERABLE``
    synthesis returns that step's output in their place. ``preserved_gates``
    lists the gates the graph kept, every verifier but those in *added*, the
    ids an applied revision added. ``revisions_remaining_after`` is the run's
    remaining revision allowance, ``added_total`` counts the revision-added
    nodes the graph holds, and ``trigger`` names what prompted the review.
    """
    delivered = _deliverable_ids(graph.nodes)
    before = delivered if delivered_before is None else set(delivered_before)
    added = set(added)
    completed = [n for n in graph.nodes if n.status is NodeStatus.COMPLETED]
    return {
        "retained": [n.id for n in completed],
        "invalidated": [n.id for n in completed if n.id in before and n.id not in delivered],
        "preserved_gates": [n.id for n in graph.nodes if n.verifies and n.id not in added],
        "revisions_remaining_after": revisions_remaining_after,
        "added_total": sum(1 for n in graph.nodes if n.metadata.get(REVISION_ADDED_KEY)),
        "trigger": trigger,
    }


def _log_text(value: Any) -> str | None:
    """Model or error text as the log keeps it: redacted, then cut short; None when empty."""
    if not value:
        return None
    return redact_account_identifiers(str(value))[:REVISION_LOG_TEXT_CHARS]


def revision_log_entry(
    record: Mapping[str, Any], *, applied: bool | None, reason: Any = None, detail: Any = None,
) -> dict[str, Any]:
    """A bounded copy of a revision *record* (see :func:`revision_record`) for the revision log.

    The entry adds whether the decision was ``applied`` (None when that is not
    known), the supervisor's ``reason`` and, for a refusal, its ``detail``.
    Both may be model text, which the log keeps only as data: account
    identifiers are redacted and at most ``REVISION_LOG_TEXT_CHARS``
    characters are kept. Each id list keeps its first ``REVISION_LOG_MAX_IDS``
    ids, and ``omitted_ids``, present only then, counts the ids left out.
    """
    entry: dict[str, Any] = {
        "after_node": str(record["after_node"]), "applied": applied, "change": record["change"],
    }
    omitted = 0
    for key in _REVISION_ID_LISTS:
        ids = [str(item) for item in record.get(key, ())]
        omitted += max(0, len(ids) - REVISION_LOG_MAX_IDS)
        entry[key] = ids[:REVISION_LOG_MAX_IDS]
    for key in ("revisions_remaining_after", "added_total", "trigger"):
        entry[key] = record.get(key)
    entry["reason"], entry["detail"] = _log_text(reason), _log_text(detail)
    if omitted:
        entry["omitted_ids"] = omitted
    return entry


_REVISION_LOG_KEYS = frozenset({
    "after_node", "applied", "change", *_REVISION_ID_LISTS, "revisions_remaining_after",
    "added_total", "trigger", "reason", "detail",
})


def _readable_entry(entry: Any) -> bool:
    """Whether *entry* has every key of a revision log entry, each of its type."""
    def either(kind: type, value: Any) -> bool:
        return value is None or type(value) is kind

    return (
        isinstance(entry, dict) and _REVISION_LOG_KEYS <= entry.keys()
        and all(type(entry[key]) is str for key in ("after_node", "change", "trigger"))
        and all(type(entry[key]) is list and all(type(item) is str for item in entry[key])
                for key in _REVISION_ID_LISTS)
        and either(bool, entry["applied"]) and either(int, entry.get("omitted_ids"))
        and either(int, entry["revisions_remaining_after"]) and either(int, entry["added_total"])
        and either(str, entry["reason"]) and either(str, entry["detail"])
    )


def read_revision_log(value: Any) -> list[dict[str, Any]]:
    """The latest ``REVISION_LOG_LIMIT`` entries of a saved revision log, detached.

    The log is informational, so one that is missing or malformed, such as a
    hand-edited entry without a key, reads as empty instead of stopping a
    resume or an inspection. Keys an entry holds beyond these are kept.
    """
    if not isinstance(value, list) or not all(_readable_entry(entry) for entry in value):
        return []
    return deepcopy(value[-REVISION_LOG_LIMIT:])
