"""PlannerMemory — persistent execution history for learning-informed planning."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExecutionOutcome:
    """Record of a single task execution for planner feedback."""

    task_goal: str
    task_constraints: list[str]
    topology: list[str]
    node_count: int
    total_cost_usd: float
    total_duration_ms: float
    success: bool
    node_outcomes: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


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

        trace = getattr(result, "trace", [])
        total_duration = sum(s.get("duration_ms", 0) for s in trace)

        node_outcomes: list[dict[str, Any]] = []
        for node in graph.nodes:
            node_outcome: dict[str, Any] = {
                "id": node.id,
                "label": node.label,
                "status": node.status.value if hasattr(node.status, "value") else str(node.status),
            }
            matching_span = next(
                (s for s in trace if s.get("node_id") == node.id), None
            )
            if matching_span:
                node_outcome["duration_ms"] = matching_span.get("duration_ms", 0)
            cost = node.metadata.get("cost_usd")
            if cost is not None:
                node_outcome["cost_usd"] = cost
            node_outcomes.append(node_outcome)

        all_completed = all(
            n.status == NodeStatus.COMPLETED for n in graph.nodes
        )

        outcome = ExecutionOutcome(
            task_goal=task.goal if hasattr(task, "goal") else str(task),
            task_constraints=list(getattr(task, "constraints", [])),
            topology=[t.value for t in graph.topology],
            node_count=len(graph.nodes),
            total_cost_usd=getattr(result, "total_cost_usd", 0.0),
            total_duration_ms=total_duration,
            success=all_completed,
            node_outcomes=node_outcomes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(outcome)) + "\n")

    def recall(self, task: Any, k: int = 3) -> list[ExecutionOutcome]:
        """Find the k most relevant past outcomes by keyword overlap."""
        if not self._path.exists():
            return []

        query_words = self._tokenize(
            getattr(task, "goal", str(task))
        )
        if not query_words:
            return []

        with self._lock:
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
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            stored_words = self._tokenize(outcome.task_goal)
            overlap = len(query_words & stored_words)
            if overlap > 0:
                scored.append((overlap, outcome))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [outcome for _, outcome in scored[:k]]

    def clear(self) -> None:
        """Remove all stored history."""
        if self._path.exists():
            self._path.unlink()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase word tokens, filtering short/stop words."""
        stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                "and", "or", "but", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "it", "its", "this", "that", "i"}
        words = set(re.findall(r"[a-z]+", text.lower()))
        return {w for w in words if len(w) > 2 and w not in stop}
