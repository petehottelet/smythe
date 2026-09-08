"""Tests for PlannerMemory — persistence and keyword-based recall."""

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.planner import LLMArchitect
from smythe.prompts import build_user_prompt
from smythe.provider import CompletionResult, Provider
from smythe.swarm import SwarmResult
from smythe.task import Task
from smythe.tracer import Span, Tracer


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


def test_recall_uses_constraints_for_relevance():
    memory, path = _make_memory()
    try:
        _write_outcome(
            path,
            _make_outcome(
                "Create campaign",
                task_constraints=["Include accessible alt text"],
            ),
        )
        _write_outcome(
            path,
            _make_outcome(
                "Create campaign",
                task_constraints=["Optimize printed materials"],
            ),
        )

        results = memory.recall(
            Task(
                goal="Create campaign",
                constraints=["Require accessible alt text"],
            ),
            k=1,
        )

        assert results[0].task_constraints == ["Include accessible alt text"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_recall_zero_returns_no_results_without_reading_file():
    memory = PlannerMemory(path="missing-history.jsonl")

    assert memory.recall(Task(goal="Research topic"), k=0) == []


@pytest.mark.parametrize("k", [-1, 1.5, True])
def test_recall_rejects_invalid_result_limit(k):
    memory = PlannerMemory(path="missing-history.jsonl")

    with pytest.raises(ValueError, match="non-negative integer"):
        memory.recall(Task(goal="Research topic"), k=k)


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

_INVALID_NUMBERS = [
    pytest.param(None, id="null"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("0.05", id="numeric-string"),
    pytest.param([], id="list"),
    pytest.param({}, id="object"),
    pytest.param(-1, id="negative-int"),
    pytest.param(-0.1, id="negative-float"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(10**400, id="huge-int"),
]


def _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value):
    path = tmp_path / "history.jsonl"
    bad = _make_outcome("Research signal target", **{field: value})
    first = _make_outcome("Research signal first")
    second = _make_outcome("Research signal second", success=False)
    for outcome in (bad, first, second):
        _write_outcome(path, outcome)
    before = path.read_bytes()

    recalled = PlannerMemory(path).recall(Task("Research signal target"), k=2)

    assert [asdict(outcome) for outcome in recalled] == [asdict(first), asdict(second)]
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["total_cost_usd", "total_duration_ms"])
@pytest.mark.parametrize("value", _INVALID_NUMBERS)
def test_recall_skips_malformed_numeric_history_before_ranking(tmp_path, field, value):
    _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value)


@pytest.mark.parametrize("field,value", [
    ("topology", None), ("topology", "serial"), ("topology", 1),
    ("topology", {}), ("topology", ["serial", 1]), ("topology", [True]),
    ("success", None), ("success", 0), ("success", 1),
    ("success", "false"), ("success", []), ("success", {}),
])
def test_recall_skips_malformed_topology_and_success_before_ranking(tmp_path, field, value):
    _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value)


@pytest.mark.parametrize("field", ["total_cost_usd", "total_duration_ms"])
def test_recall_skips_oversized_integer_literals_without_rewriting_history(tmp_path, field):
    path = tmp_path / "history.jsonl"
    invalid = asdict(_make_outcome("Research signal target", **{field: "OVERSIZED"}))
    raw = json.dumps(invalid).replace('"OVERSIZED"', "9" * 5000)
    good = _make_outcome("Research signal")
    path.write_text(raw + "\n" + json.dumps(asdict(good)) + "\n", encoding="utf-8")
    before = path.read_bytes()

    assert PlannerMemory(path).recall(Task("Research signal target")) == [good]
    assert path.read_bytes() == before


@pytest.mark.parametrize("topology,cost,duration,success", [
    ([], 0, 0, False),
    (["custom phase"], 2, 100, True),
    (["fork_join", "serial"], 0.15, 4500.25, False),
    ([""], -0.0, -0.0, True),
    (["serial"], 1e308, 1e308, True),
])
def test_recall_preserves_valid_legacy_values_exactly(tmp_path, topology, cost, duration, success):
    path = tmp_path / "history.jsonl"
    outcome = _make_outcome(
        "Research signal", topology=topology, total_cost_usd=cost,
        total_duration_ms=duration, success=success,
        node_outcomes=[{"id": "old", "metadata": {"retained": [1, 2.0, False]}}],
    )
    data = asdict(outcome)
    del data["task_context"], data["task_done_when"]
    data["future_extension"] = {"retained": "in original file"}
    path.write_bytes((json.dumps(data, separators=(",", ":")) + "\r\n").encode("utf-8"))
    before = path.read_bytes()

    [restored] = PlannerMemory(path).recall(Task("Research signal"))

    # Canonical JSON distinguishes valid integer/float values and negative zero.
    assert json.dumps(asdict(restored), sort_keys=True) == json.dumps(asdict(outcome), sort_keys=True)
    history = [asdict(restored)]
    snapshot = deepcopy(history)
    build_user_prompt(Task("Research signal"), history)
    assert history == snapshot
    assert path.read_bytes() == before


class _HistoryPlanningProvider(Provider):
    def __init__(self):
        self.prompts = []

    async def complete(self, system, prompt, model):
        self.prompts.append(prompt)
        return CompletionResult(text=json.dumps({
            "topology": ["serial"],
            "nodes": [{"id": "deliverable", "label": "Return the complete answer"}],
        }))


@pytest.mark.parametrize("field,value", [
    ("total_cost_usd", "invalid"), ("total_duration_ms", "invalid"),
    ("topology", 1), ("total_cost_usd", float("nan")),
])
def test_real_planner_formats_only_valid_recalled_history(tmp_path, field, value):
    path = tmp_path / "history.jsonl"
    _write_outcome(path, _make_outcome("Research signal invalid", **{field: value}))
    _write_outcome(path, _make_outcome("Research signal accepted", success=False))
    before = path.read_bytes()
    provider = _HistoryPlanningProvider()
    planner = LLMArchitect(provider, planning_model="offline-test", memory=PlannerMemory(path))

    graph, _ = planner.plan(Task("Research signal"))

    assert len(graph.nodes) == 1
    assert len(provider.prompts) == 1
    assert "Research signal accepted" in provider.prompts[0]
    assert "Research signal invalid" not in provider.prompts[0]
    assert "Outcome: failure" in provider.prompts[0]
    assert path.read_bytes() == before


def test_recall_preserves_history_file_read_errors(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    _write_outcome(path, _make_outcome("Research signal"))
    failure = PermissionError("history cannot be read")
    original = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == path:
            raise failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(PermissionError) as caught:
        PlannerMemory(path).recall(Task("Research signal"))
    assert caught.value is failure


def test_parallel_history_labels_summed_node_time(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal in parallel")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.COMPLETED),
        Node(id="b", label="Research B", status=NodeStatus.COMPLETED),
        Node(id="join", label="Assemble findings", depends_on=["a", "b"], status=NodeStatus.COMPLETED),
    ])
    graph.validate()
    tracer = Tracer()
    tracer.spans = [
        Span("a", "Research A", start_time=100, end_time=103, status="completed"),
        Span("b", "Research B", start_time=101, end_time=104, status="completed"),
        Span("join", "Assemble findings", start_time=104, end_time=104, status="completed"),
    ]
    trace = tracer.summary()
    assert [span["duration_ms"] for span in trace] == [3000, 3000, 0]
    envelope_ms = (max(span.end_time for span in tracer.spans)
                   - min(span.start_time for span in tracer.spans)) * 1000
    assert envelope_ms == 4000
    before_graph, before_trace = asdict(graph), deepcopy(trace)
    PlannerMemory(path).record(task, graph, SwarmResult("done", graph, trace=trace, total_cost_usd=0.03))
    before = path.read_bytes()

    [outcome] = PlannerMemory(path).recall(task)
    history = [asdict(outcome)]
    before_history = deepcopy(history)
    prompt = build_user_prompt(task, history)

    assert outcome.total_duration_ms == 6000
    assert [row["duration_ms"] for row in outcome.node_outcomes] == [3000, 3000, 0]
    assert "Summed node time: 6000ms" in prompt
    assert "Duration:" not in prompt
    assert "4000ms" not in prompt
    assert "total_duration_ms" in json.loads(before)
    assert "wall_time_ms" not in json.loads(before)
    assert path.read_bytes() == before
    assert asdict(graph) == before_graph
    assert trace == before_trace
    assert history == before_history


def test_prompt_without_history_keeps_existing_text():
    assert build_user_prompt(Task("Research signal")) == (
        "## Task\n\nResearch signal\n\nRespond with only the JSON object."
    )
