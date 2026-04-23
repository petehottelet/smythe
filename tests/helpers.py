"""Shared test utilities for the smythe test suite."""

from __future__ import annotations

import json

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.planner import DeterministicArchitect
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry


# ---------------------------------------------------------------------------
# Reusable mock providers
# ---------------------------------------------------------------------------


class MockProvider(Provider):
    """Returns a canned response that includes a truncated prompt echo."""

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return CompletionResult(text=f"mock: {prompt[:40]}", prompt_tokens=10, completion_tokens=5)


class FailingProvider(Provider):
    """Provider that fails a configurable number of times before succeeding.

    If ``fail_labels`` is set, only prompts whose first line matches
    one of the labels will be failed; all others succeed immediately.
    """

    def __init__(self, failures: int = 1, fail_labels: set[str] | None = None) -> None:
        self._failures = failures
        self._fail_labels = fail_labels
        self._attempts: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        if self._fail_labels is not None and label not in self._fail_labels:
            return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)
        self._attempts.setdefault(label, 0)
        self._attempts[label] += 1
        if self._attempts[label] <= self._failures:
            raise RuntimeError(f"Simulated failure #{self._attempts[label]}")
        return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)


class ClassifierMockProvider(Provider):
    """Returns a fixed classification string — useful for WhiteRabbit tests."""

    def __init__(self, classification: str) -> None:
        self._classification = classification

    async def complete(self, system, prompt, model):
        return CompletionResult(text=self._classification, prompt_tokens=5, completion_tokens=2)


SIMPLE_PLAN_JSON = json.dumps({
    "topology": ["serial"],
    "nodes": [
        {
            "id": "step-1",
            "label": "Do the thing",
            "depends_on": [],
            "agent": {
                "name": "Worker",
                "persona": "You are a helpful worker.",
                "capabilities": ["general"],
            },
        },
    ],
})


class PlanningMockProvider(Provider):
    """Returns valid plan JSON on the first call, then mock text on subsequent calls."""

    def __init__(self) -> None:
        self._call_count = 0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self._call_count += 1
        if self._call_count == 1:
            return CompletionResult(text=SIMPLE_PLAN_JSON, prompt_tokens=50, completion_tokens=100)
        return CompletionResult(text=f"mock: {prompt[:40]}", prompt_tokens=10, completion_tokens=5)


class FixedArchitect(DeterministicArchitect):
    """Returns a single-node graph labelled with a tag — useful for router tests."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def plan(self, task):
        node = Node(label=f"{self.tag}: {task.goal}")
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        graph.validate()
        return graph, Registry()


# ---------------------------------------------------------------------------
# Helper to build completed graphs for synthesizer tests
# ---------------------------------------------------------------------------


def make_completed_graph(*results: str) -> ExecutionGraph:
    """Build a graph whose nodes are already COMPLETED with the given results."""
    nodes = []
    for i, r in enumerate(results):
        n = Node(id=f"n{i}", label=f"Step {i}")
        n.status = NodeStatus.COMPLETED
        n.result = r
        nodes.append(n)
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)
