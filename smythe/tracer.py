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
            self.spans.append(span)
            logger.debug(
                "Node finished: %s (%s) — %.1fms",
                node.id,
                node.label,
                span.duration_ms,
            )

    def on_node_error(self, node: Node, exc: Exception) -> None:
        span = self._active.get(node.id)
        if span:
            span.error = str(exc)
        logger.warning("Node error: %s (%s) — %s", node.id, node.label, exc)

    def summary(self) -> list[dict[str, Any]]:
        """Return spans as plain dicts for serialization / planner feedback."""
        return [
            {
                "node_id": s.node_id,
                "label": s.label,
                "agent_id": s.agent_id,
                "status": s.status,
                "duration_ms": round(s.duration_ms, 1),
                "error": s.error,
            }
            for s in self.spans
        ]
