"""Task definition — the unit of work submitted to a Swarm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A goal-oriented unit of work that the planner decomposes into an execution graph.

    Attributes:
        goal: Natural-language description of the desired outcome.
        constraints: Optional hard requirements the execution must satisfy.
        context: Arbitrary key-value context forwarded to agents.
    """

    goal: str
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValueError("Task requires a non-empty goal")
