"""Tracer — structured observability for every node execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from smythe.graph import Node

logger = logging.getLogger("smythe.tracer")


@dataclass
class Span:
    """A single trace span corresponding to one node execution."""

    node_id: str
    label: str
    agent_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Collects structured execution traces.

    Every node start/end/error emits a Span that can be inspected
    after execution for debugging and planner feedback.
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._active: dict[str, Span] = {}

    def on_node_start(self, node: Node) -> None:
        span = Span(
            node_id=node.id,
            label=node.label,
            agent_id=node.agent_id,
            start_time=time.time(),
            status="running",
        )
        self._active[node.id] = span
        logger.debug("Node started: %s (%s)", node.id, node.label)

    def on_node_end(self, node: Node) -> None:
        span = self._active.pop(node.id, None)
        if span:
            span.end_time = time.time()
            span.status = node.status.value
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
    ) -> None:
        """Record a supervisor revision, whether it was applied or refused.

        Refusals are recorded too: a supervisor that keeps proposing
        invalid changes is a finding, and silently dropping those
        attempts would hide it.
        """
        now = time.time()
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
        span.metadata["revision"] = {
            "after_node": node.id,
            "change": getattr(revision, "summary", lambda: "unknown")(),
            "added": [n.id for n in getattr(revision, "add_nodes", ())],
            "dropped": list(getattr(revision, "drop_node_ids", ())),
            "rewired": sorted(getattr(revision, "rewire", {})),
        }
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
        now = time.time()
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

    def on_node_error(self, node: Node, exc: Exception) -> None:
        span = self._active.get(node.id)
        if span:
            span.error = str(exc)
        logger.warning("Node error: %s (%s) — %s", node.id, node.label, exc)

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
