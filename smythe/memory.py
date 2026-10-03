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

from smythe.task import Task, snapshot_task

# Delivery acceptance decisions an outcome can record.
_ACCEPTANCE_VALUES = frozenset({
    "not_evaluated", "accepted", "rejected", "needs_review", "indeterminate",
})


def _is_duration(value: Any) -> bool:
    """True for a finite, non-negative int or float."""
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _flag(value: Any) -> bool | None:
    return value if type(value) is bool else None


@dataclass
class ExecutionOutcome:
    """Record of a single task execution for planner feedback.

    ``total_duration_ms`` sums recorded node-span times; it is not wall time.
    A node outcome's ``duration_ms`` is the time of that node's final attempt.

    ``elapsed_ms`` is the elapsed time of the call that recorded the outcome.
    ``latency_complete`` is False and ``resumed`` True when that call resumed
    a checkpoint, so ``elapsed_ms`` covers only part of the run. The cost
    flags copy the result's. ``acceptance`` is ``"not_evaluated"`` for new
    records. These fields are None when not recorded, as in history written
    before they existed.
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


class PlannerMemory:
    """Persistent store of execution outcomes.

    Stores one ExecutionOutcome per JSONL line.  The planner queries
    this history to inform topology decisions for new tasks.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".smythe" / "history.jsonl"
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        task: Any,
        graph: Any,
        result: Any,
    ) -> None:
        """Append an execution outcome to the history file."""
        from smythe.graph import NodeStatus

        # Capture nested context before inspecting mutable execution results.
        # Legacy duck-typed callers keep their existing goal/constraint path.
        saved_task = snapshot_task(task) if isinstance(task, Task) else task

        trace = getattr(result, "trace", [])
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
        timing = getattr(result, "timing", None)
        timing = timing if isinstance(timing, dict) else {}
        elapsed = timing.get("elapsed_ms")

        outcome = ExecutionOutcome(
            task_goal=saved_task.goal if hasattr(saved_task, "goal") else str(saved_task),
            task_constraints=list(getattr(saved_task, "constraints", [])),
            topology=[t.value for t in graph.topology],
            node_count=len(graph.nodes),
            total_cost_usd=getattr(result, "total_cost_usd", 0.0),
            total_duration_ms=total_duration,
            success=all_completed,
            node_outcomes=node_outcomes,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_context=saved_task.context if isinstance(saved_task, Task) else {},
            task_done_when=list(getattr(saved_task, "done_when", [])),
            elapsed_ms=elapsed if _is_duration(elapsed) else None,
            latency_complete=_flag(timing.get("latency_complete")),
            resumed=_flag(timing.get("resumed")),
            cost_is_complete=_flag(getattr(result, "cost_is_complete", None)),
            cost_contains_estimates=_flag(getattr(result, "cost_contains_estimates", None)),
            # No run evaluates delivery acceptance yet; None means not recorded.
            acceptance="not_evaluated",
        )

        line = (json.dumps(asdict(outcome)) + "\n").encode("utf-8")
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
