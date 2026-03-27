"""Tests for PlannerMemory — persistence and keyword-based recall."""

import json
import os
import tempfile
from dataclasses import asdict

import pytest

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.swarm import SwarmResult
from smythe.task import Task


def _make_memory() -> tuple[PlannerMemory, str]:
    """Create a PlannerMemory backed by a temp file."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)
    return PlannerMemory(path=path), path


def _write_outcome(path: str, outcome: ExecutionOutcome) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(outcome)) + "\n")


def _make_outcome(goal: str, **kwargs) -> ExecutionOutcome:
    defaults = dict(
        task_goal=goal,
        task_constraints=[],
        topology=["serial"],
        node_count=1,
        total_cost_usd=0.01,
        total_duration_ms=100,
        success=True,
        node_outcomes=[],
        timestamp="2025-01-01T00:00:00Z",
    )
    defaults.update(kwargs)
    return ExecutionOutcome(**defaults)


def test_record_and_recall():
    memory, path = _make_memory()
    try:
        task = Task(goal="Research competitors and write a report")
        graph = ExecutionGraph(
            topology=[Topology.SERIAL],
            nodes=[Node(label="Do research", id="r1", status=NodeStatus.COMPLETED)],
        )
        result = SwarmResult(output="done", graph=graph, trace=[], total_cost_usd=0.05)

        memory.record(task, graph, result)

        recalled = memory.recall(Task(goal="Research competitors for new product"))
        assert len(recalled) == 1
        assert recalled[0].task_goal == "Research competitors and write a report"
        assert recalled[0].success is True
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_recall_relevance_ordering():
    memory, path = _make_memory()
    try:
        _write_outcome(path, _make_outcome("Bake a chocolate cake with frosting"))
        _write_outcome(path, _make_outcome("Research market competitors in healthcare"))
        _write_outcome(path, _make_outcome("Research financial competitors and analysis"))

        results = memory.recall(Task(goal="Research competitors"))
        assert len(results) >= 2
        goals = [r.task_goal for r in results]
        assert goals[0] in (
            "Research market competitors in healthcare",
            "Research financial competitors and analysis",
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_recall_returns_k_results():
    memory, path = _make_memory()
    try:
        for i in range(10):
            _write_outcome(path, _make_outcome(f"Research topic number {i}"))

        results = memory.recall(Task(goal="Research topic"), k=3)
        assert len(results) == 3

        results_all = memory.recall(Task(goal="Research topic"), k=20)
        assert len(results_all) == 10
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_recall_empty_memory():
    memory, path = _make_memory()
    try:
        results = memory.recall(Task(goal="Anything"))
        assert results == []
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_clear_wipes_history():
    memory, path = _make_memory()
    try:
        _write_outcome(path, _make_outcome("Some past task about research"))

        assert len(memory.recall(Task(goal="research"))) > 0

        memory.clear()

        assert memory.recall(Task(goal="research")) == []
        assert not os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_file_persistence():
    memory, path = _make_memory()
    try:
        _write_outcome(path, _make_outcome("Research market competitors"))

        memory2 = PlannerMemory(path=path)
        results = memory2.recall(Task(goal="Research competitors"))
        assert len(results) == 1
        assert results[0].task_goal == "Research market competitors"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_recall_skips_corrupt_lines():
    """Corrupt or partial JSONL lines are silently skipped."""
    memory, path = _make_memory()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"partial": true}\n')
            f.write("this is not json at all\n")
            f.write("\n")
            f.write(json.dumps(asdict(_make_outcome("Research competitors in market"))) + "\n")

        results = memory.recall(Task(goal="Research competitors"))
        assert len(results) == 1
        assert results[0].task_goal == "Research competitors in market"
    finally:
        if os.path.exists(path):
            os.unlink(path)
