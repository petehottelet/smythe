"""Task definition — the unit of work submitted to a Swarm."""

from __future__ import annotations

from collections.abc import Mapping
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
        if not isinstance(self.goal, str):
            raise TypeError("Task goal must be a string")
        self.goal = self.goal.strip()
        if not self.goal:
            raise ValueError("Task requires a non-empty goal")

        if isinstance(self.constraints, (str, bytes)):
            raise TypeError("Task constraints must be an iterable of strings")
        try:
            constraints = list(self.constraints)
        except TypeError as exc:
            raise TypeError(
                "Task constraints must be an iterable of strings"
            ) from exc
        if any(not isinstance(item, str) for item in constraints):
            raise TypeError("Task constraints must contain only strings")
        self.constraints = [item.strip() for item in constraints if item.strip()]

        if not isinstance(self.context, Mapping):
            raise TypeError("Task context must be a mapping")
        self.context = dict(self.context)
