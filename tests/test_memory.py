"""Tests for PlannerMemory — persistence and keyword-based recall."""

import asyncio
import json
import logging
import os
import stat
import tempfile
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from fakes import Gate, ManualClock
from smythe import _redact
from smythe._failures import classify_failure
from smythe.acceptance import AcceptancePolicy, DeliveryAcceptance, text_sha256
from smythe.checkpoint import FileCheckpointStore
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.graph import (
    ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology, structure_sha256,
)
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.planner import LLMArchitect
from smythe.prompts import build_user_prompt
from smythe.provider import (
    CompletionResult, OutputTruncatedError, Provider, ProviderAccountingCancelledError,
)
from smythe.registry import Registry
from smythe.swarm import Swarm, SwarmResult
from smythe.synthesizer import MERGE_SYSTEM_PROMPT, SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.tools import ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec
from smythe.tracer import Span, Tracer
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding

_OUTCOME_FIELDS = (
    "elapsed_ms", "latency_complete", "resumed", "cost_is_complete",
    "cost_contains_estimates", "acceptance",
)
# The keys a history row had before verification and acceptance evidence.
_PREVIOUS_KEYS = frozenset({
    "task_goal", "task_constraints", "topology", "node_count", "total_cost_usd",
    "total_duration_ms", "success", "node_outcomes", "timestamp", "task_context",
    "task_done_when", *_OUTCOME_FIELDS,
})
_EVIDENCE_FIELDS = ("execution_id", "verification", "rejected_output_sha256", "structure_sha256")
_FAILURE_FIELDS = ("failure_category", "failure_type", "failure_excerpt")
_GATE = {"gate": "judge", "target": "draft", "state": "consumed", "passed": False,
         "regenerations_used": 1}


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


def _clocked(swarm: Swarm, clock: ManualClock) -> Swarm:
    swarm._tracer_factory = lambda: Tracer(
        clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9,
    )
    return swarm


def test_parallel_run_records_summed_node_time_and_elapsed_time_separately(tmp_path):
    """Two research nodes overlap for 3s, then the join runs for 1s."""
    clock, gate, started = ManualClock(), Gate(), []

    class OverlappingProvider(Provider):
        async def complete(self, system, prompt, model):
            step = next(line for line in prompt.splitlines() if line.startswith("Your step: "))
            started.append(step)
            if step == "Your step: Assemble findings":
                clock.advance(1)
            else:
                await gate.wait()
            return CompletionResult(text=f"done: {step}", prompt_tokens=1, completion_tokens=1)

    path = tmp_path / "history.jsonl"
    task = Task("Research signal in parallel")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], task=task, nodes=[
        Node(id="a", label="Research A"), Node(id="b", label="Research B"),
        Node(id="join", label="Assemble findings", depends_on=["a", "b"]),
    ])
    swarm = _clocked(Swarm(model="m", provider=OverlappingProvider(), parallel=True,
                           memory=PlannerMemory(path), artifact_dir=None), clock)

    async def run():
        execution = asyncio.create_task(swarm.execute_async(graph))
        for _ in range(1000):
            if len(started) == 2:
                break
            await asyncio.sleep(0)
        clock.advance(3)
        gate.release()
        return await execution

    result = asyncio.run(run())
    record = json.loads(path.read_text(encoding="utf-8"))
    [outcome] = PlannerMemory(path).recall(task)
    prompt = build_user_prompt(task, [asdict(outcome)])

    assert sorted(started) == ["Your step: Assemble findings", "Your step: Research A",
                               "Your step: Research B"]
    assert (result.timing["summed_node_ms"], result.timing["elapsed_ms"]) == (7000, 4000)
    assert outcome.total_duration_ms == 7000
    assert outcome.elapsed_ms == 4000
    assert (outcome.latency_complete, outcome.resumed) == (True, False)
    assert (outcome.cost_is_complete, outcome.cost_contains_estimates) == (True, False)
    assert outcome.acceptance == "not_evaluated"
    assert "Summed node time: 7000ms" in prompt
    assert "4000" not in prompt
    assert "wall_time_ms" not in record


def test_resumed_run_records_that_its_elapsed_time_is_partial(tmp_path):
    clock = ManualClock()

    class Script(Provider):
        def __init__(self, crash_on=None):
            self.crash_on = crash_on

        async def complete(self, system, prompt, model):
            step = next(line for line in prompt.splitlines() if line.startswith("Your step: "))
            clock.advance(1)
            if step == self.crash_on:
                raise RuntimeError("process died")
            return CompletionResult(text=f"done: {step}", prompt_tokens=1, completion_tokens=1)

    path, store = tmp_path / "history.jsonl", FileCheckpointStore(tmp_path / "checkpoints")
    task = Task("Research signal after a crash")
    graph = ExecutionGraph(topology=[Topology.SERIAL], task=task, nodes=[
        Node(id="a", label="Research A"),
        Node(id="b", label="Assemble findings", depends_on=["a"]),
    ])
    with pytest.raises(RuntimeError, match="process died"):
        Swarm(model="m", provider=Script(crash_on="Your step: Assemble findings"),
              memory=PlannerMemory(path), checkpoint_store=store, artifact_dir=None).execute(graph)
    [execution_id] = store.list_ids()

    result = _clocked(Swarm(model="m", provider=Script(), memory=PlannerMemory(path),
                            checkpoint_store=store, artifact_dir=None), clock).resume(execution_id)
    [outcome] = PlannerMemory(path).recall(task)

    assert outcome.elapsed_ms == result.timing["elapsed_ms"] == 1000
    assert (outcome.latency_complete, outcome.resumed) == (False, True)
    assert outcome.acceptance == "not_evaluated"


def test_record_without_timing_leaves_elapsed_time_unrecorded(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal by hand")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.COMPLETED),
    ])

    class BareResult:
        trace: list = []
        total_cost_usd = 0.0

    PlannerMemory(path).record(task, graph, SwarmResult("done", graph))
    PlannerMemory(path).record(task, graph, BareResult())
    built, bare = PlannerMemory(path).recall(task, k=2)

    assert (built.elapsed_ms, built.latency_complete, built.resumed) == (None, None, None)
    assert (built.cost_is_complete, built.cost_contains_estimates) == (True, False)
    assert (bare.cost_is_complete, bare.cost_contains_estimates) == (None, None)
    assert built.acceptance == bare.acceptance == "not_evaluated"


@pytest.mark.parametrize("decision", [
    "accepted", "rejected", "needs_review", "indeterminate", "not_evaluated",
])
def test_record_writes_the_delivery_acceptance_decision(tmp_path, decision):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal with a decision")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.COMPLETED, result="done"),
    ])
    decided = decision != "not_evaluated"
    record = DeliveryAcceptance(
        decision=decision, policy_sha256="c" * 64,
        output_sha256=text_sha256("done") if decided else None,
        strategy="deliverable" if decided else None,
        reasons=() if decision in ("accepted", "not_evaluated") else ("validator rejected it",),
    )
    output = "done" if decision == "accepted" else ""

    PlannerMemory(path).record(task, graph, SwarmResult(output, graph, acceptance=record))
    [outcome] = PlannerMemory(path).recall(task)

    assert outcome.acceptance == decision
    # Delivery acceptance is recorded beside execution success, not folded into it.
    assert outcome.success is True


def test_record_reads_a_decision_only_from_an_acceptance_record(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal by hand")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.COMPLETED),
    ])

    class LooseResult:
        trace: list = []
        total_cost_usd = 0.0
        acceptance = "accepted"

    PlannerMemory(path).record(task, graph, LooseResult())
    [outcome] = PlannerMemory(path).recall(task)

    assert outcome.acceptance == "not_evaluated"


def test_legacy_rows_recall_with_outcome_fields_unrecorded(tmp_path):
    path = tmp_path / "history.jsonl"
    legacy = asdict(_make_outcome("Research signal"))
    for name in _OUTCOME_FIELDS:
        del legacy[name]
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    [outcome] = PlannerMemory(path).recall(Task("Research signal"))

    assert [getattr(outcome, name) for name in _OUTCOME_FIELDS] == [None] * 6


@pytest.mark.parametrize("acceptance", [
    "not_evaluated", "accepted", "rejected", "needs_review", "indeterminate",
])
def test_recall_keeps_valid_outcome_fields(tmp_path, acceptance):
    path = tmp_path / "history.jsonl"
    outcome = _make_outcome(
        "Research signal", elapsed_ms=4000.5, latency_complete=False, resumed=True,
        cost_is_complete=False, cost_contains_estimates=True, acceptance=acceptance,
    )
    _write_outcome(path, outcome)

    assert PlannerMemory(path).recall(Task("Research signal")) == [outcome]


@pytest.mark.parametrize("field,value", [
    ("elapsed_ms", "4000"), ("elapsed_ms", -1), ("elapsed_ms", float("nan")),
    ("elapsed_ms", float("inf")), ("elapsed_ms", True),
    pytest.param("elapsed_ms", 10**400, id="elapsed_ms-huge-int"),
    ("elapsed_ms", []), ("latency_complete", 1), ("latency_complete", "true"),
    ("resumed", 0), ("resumed", "no"), ("cost_is_complete", 1),
    ("cost_contains_estimates", "false"), ("acceptance", "approved"),
    ("acceptance", ""), ("acceptance", 1), ("acceptance", True),
])
def test_recall_skips_malformed_outcome_fields_before_ranking(tmp_path, field, value):
    _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value)


def test_retried_node_duration_comes_from_its_final_attempt(tmp_path):
    """Regression: a retried node reported its first, failed attempt's time."""
    path = tmp_path / "history.jsonl"
    task = Task("Research signal with retries")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.COMPLETED),
        Node(id="b", label="Assemble findings", depends_on=["a"], status=NodeStatus.COMPLETED),
    ])
    tracer = Tracer()
    tracer.spans = [
        Span("a", "Research A", start_time=100, end_time=105, status="failed", error="timeout"),
        Span("a", "Research A", start_time=105, end_time=106, status="failed", error="timeout"),
        Span("a", "Research A", start_time=106, end_time=108, status="completed"),
        Span("b", "Assemble findings", start_time=108, end_time=111, status="completed"),
    ]
    trace = tracer.summary()

    PlannerMemory(path).record(task, graph, SwarmResult("done", graph, trace=trace))
    [outcome] = PlannerMemory(path).recall(task)

    assert [(row["id"], row["duration_ms"]) for row in outcome.node_outcomes] == [
        ("a", 2000), ("b", 3000),
    ]
    # The summed total still counts every recorded attempt.
    assert outcome.total_duration_ms == 11000
    assert "Summed node time: 11000ms" in build_user_prompt(task, [asdict(outcome)])


def test_prompt_without_history_keeps_existing_text():
    assert build_user_prompt(Task("Research signal")) == (
        "## Task\n\nResearch signal\n\nRespond with only the JSON object."
    )


def _record_goal(memory: PlannerMemory, goal: str, context: dict | None = None) -> None:
    graph = ExecutionGraph(
        topology=[Topology.SERIAL],
        nodes=[Node(label="Do it", id="n1", status=NodeStatus.COMPLETED)],
    )
    task = Task(goal=goal, context=context or {})
    memory.record(task, graph, SwarmResult(output="done", graph=graph, trace=[]))


@pytest.fixture
def default_umask():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_record_creates_owner_only_history_and_directory(tmp_path, default_umask):
    """Regression: history holding task.context was created 0644."""
    directory = tmp_path / ".smythe"
    memory = PlannerMemory(directory / "history.jsonl")

    _record_goal(memory, "Research alpha", {"customer": "secret"})

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(memory.path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_record_keeps_existing_history_permissions(tmp_path, default_umask):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    path = directory / "history.jsonl"
    _write_outcome(path, _make_outcome("Research alpha"))
    path.chmod(0o640)

    _record_goal(PlannerMemory(path), "Research beta")

    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_record_after_a_torn_last_line_keeps_the_next_record(tmp_path):
    """Regression: a crash-truncated last line swallowed record "gamma"."""
    path = tmp_path / "history.jsonl"
    memory = PlannerMemory(path)
    _record_goal(memory, "Research alpha")
    with open(path, "ab") as stream:
        stream.write(b'{"task_goal": "Research beta", "task_constr')

    _record_goal(memory, "Research gamma")
    _record_goal(memory, "Research delta")

    goals = {outcome.task_goal for outcome in memory.recall(Task("Research"), k=10)}
    assert goals == {"Research alpha", "Research gamma", "Research delta"}
    lines = path.read_bytes().split(b"\n")
    assert lines[1] == b'{"task_goal": "Research beta", "task_constr'
    assert len(lines) == 5 and lines[-1] == b""


class _Steps(Provider):
    """Answers each step from the queue of the first label prefix it starts with.

    The last queued answer repeats; an exception in the queue is raised.
    """

    def __init__(self, script: dict[str, list]) -> None:
        self._script = {prefix: list(answers) for prefix, answers in script.items()}

    async def complete(self, system, prompt, model):
        step = next(line for line in prompt.splitlines() if line.startswith("Your step: "))
        label = step[len("Your step: "):]
        queue = next(answers for prefix, answers in self._script.items() if label.startswith(prefix))
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        return CompletionResult(text=answer, prompt_tokens=1, completion_tokens=1)


class _Forbids:
    """A final validator that rejects delivered text containing ``marker``."""

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.identity = ValidatorIdentity.for_config("forbids", "1", {"marker": marker})

    def validate(self, subject):
        if self.marker in subject.text():
            return ValidationReport("output", (
                hard_finding("forbidden_text", "output contains the forbidden marker"),
            ), validator=self.identity)
        return ValidationReport("output", validator=self.identity)


class _SelectsTemplate(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text='[{"template": "memo"}]')


def _gated(task: Task, *, max_regenerations: int = 1) -> ExecutionGraph:
    """A draft that a judge gates."""
    return ExecutionGraph(topology=[Topology.SERIAL], task=task, nodes=[
        Node(id="draft", label="Draft the memo"),
        Node(id="judge", label="Judge the memo", depends_on=["draft"], verifies="draft",
             max_regenerations=max_regenerations),
    ])


def _history_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("policy", [None, AcceptancePolicy()], ids=["no-policy", "advisory"])
@pytest.mark.parametrize("verdicts,passed,used", [
    (["FAIL: the memo is thin"], False, 2),
    (["FAIL: the memo is thin", "PASS"], True, 1),
    (["PASS"], True, 0),
], ids=["exhausted", "passed-after-regeneration", "passed"])
def test_gate_verdict_is_recorded_beside_success(tmp_path, policy, verdicts, passed, used):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal behind a gate")
    swarm = Swarm(model="m", provider=_Steps({"Draft": ["A draft."], "Judge": verdicts}),
                  memory=PlannerMemory(path), artifact_dir=None, acceptance=policy)

    result = swarm.execute(_gated(task, max_regenerations=2))
    [row] = _history_rows(path)
    [outcome] = PlannerMemory(path).recall(task)

    # An exhausted gate lets the run finish: success still means every node
    # completed, and the gate's entry shows that its final verdict did not pass.
    assert outcome.success is True
    assert outcome.acceptance == "not_evaluated"
    assert outcome.verification == [dict(_GATE, passed=passed, regenerations_used=used)]
    assert outcome.execution_id == result.execution_id
    assert outcome.structure_sha256 == structure_sha256(result.graph)
    assert outcome.rejected_output_sha256 is None
    assert set(row) == _PREVIOUS_KEYS | {"execution_id", "verification", "structure_sha256"}


def test_gate_whose_target_was_unavailable_is_recorded_as_not_passed(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal behind a gate")
    graph = ExecutionGraph(topology=[Topology.SERIAL], task=task, nodes=[
        Node(id="draft", label="Draft the memo", failure_policy=FailurePolicy.SKIP, max_retries=0),
        Node(id="judge", label="Judge the memo", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])
    provider = _Steps({"Draft": [RuntimeError("draft failed")], "Judge": ["PASS"]})

    Swarm(model="m", provider=provider, memory=PlannerMemory(path), artifact_dir=None,
          ).execute(graph)
    [outcome] = PlannerMemory(path).recall(task)

    # The judge said PASS, but about nothing: the receipt records no pass.
    assert outcome.success is False
    assert outcome.verification == [dict(_GATE, passed=False, regenerations_used=0)]


@pytest.mark.parametrize("draft,decision", [
    ("A draft with DEFECT-4c1e in its second paragraph.", "rejected"),
    ("A clean draft.", "accepted"),
])
def test_strict_run_records_its_decision_and_only_a_digest_of_rejected_output(
    tmp_path, draft, decision,
):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal under a strict policy")
    policy = AcceptancePolicy(mode="strict", final_validators=[_Forbids("DEFECT")])
    swarm = Swarm(model="m", provider=_Steps({"Draft": [draft], "Judge": ["PASS"]}),
                  memory=PlannerMemory(path), artifact_dir=None, acceptance=policy)

    result = swarm.execute(_gated(task))
    [row] = _history_rows(path)
    [outcome] = PlannerMemory(path).recall(task)

    assert outcome.acceptance == result.acceptance.decision.value == decision
    assert outcome.success is True
    # The gate passed the draft; the final validator decided its delivery.
    assert outcome.verification == [dict(_GATE, passed=True, regenerations_used=0)]
    assert outcome.execution_id == result.execution_id
    assert outcome.structure_sha256 == structure_sha256(result.graph)
    if decision == "rejected":
        assert "DEFECT-4c1e" in result.rejected_output
        assert outcome.rejected_output_sha256 == text_sha256(result.rejected_output)
        assert b"DEFECT-4c1e" not in path.read_bytes()
        assert set(row) == _PREVIOUS_KEYS | set(_EVIDENCE_FIELDS)
    else:
        assert result.rejected_output is None and outcome.rejected_output_sha256 is None
        assert "rejected_output_sha256" not in row


def test_strict_run_without_a_gate_records_its_acceptance_as_evidence(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal under a strict policy without a gate")
    graph = ExecutionGraph(topology=[Topology.SERIAL], task=task,
                           nodes=[Node(id="draft", label="Draft the memo")])
    policy = AcceptancePolicy(mode="strict", final_validators=[_Forbids("DEFECT")])
    swarm = Swarm(model="m", provider=_Steps({"Draft": ["A clean draft."]}),
                  memory=PlannerMemory(path), artifact_dir=None, acceptance=policy)

    result = swarm.execute(graph)
    [row] = _history_rows(path)
    [outcome] = PlannerMemory(path).recall(task)

    # No gate left a receipt and nothing was rejected: the acceptance record
    # alone makes the run's evidence worth keeping.
    assert "verification_receipt" not in result.graph.nodes[0].metadata
    assert result.rejected_output is None
    assert outcome.acceptance == result.acceptance.decision.value == "accepted"
    assert outcome.execution_id == result.execution_id
    assert outcome.structure_sha256 == structure_sha256(result.graph)
    assert (outcome.verification, outcome.rejected_output_sha256) == (None, None)
    assert set(row) == _PREVIOUS_KEYS | {"execution_id", "structure_sha256"}


def test_strict_exhausted_gate_records_the_rejection_and_its_failing_verdict(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal under a strict policy")
    provider = _Steps({"Draft": ["A thin draft, DEFECT-9b2d."], "Judge": ["FAIL: thin"]})
    swarm = Swarm(model="m", provider=provider, memory=PlannerMemory(path), artifact_dir=None,
                  acceptance=AcceptancePolicy(mode="strict"))

    result = swarm.execute(_gated(task))
    [outcome] = PlannerMemory(path).recall(task)

    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert (receipt["outcome"], receipt["exhausted"]) == ("rejected", True)
    assert (outcome.success, outcome.acceptance) == (True, "rejected")
    assert outcome.verification == [dict(_GATE, passed=False, regenerations_used=1)]
    assert outcome.rejected_output_sha256 == text_sha256(result.rejected_output)
    assert b"DEFECT-9b2d" not in path.read_bytes()


@pytest.mark.parametrize("policy", [None, AcceptancePolicy()], ids=["no-policy", "advisory"])
@pytest.mark.parametrize("gate", [False, True], ids=["no-gate", "unread-gate"])
def test_run_without_gate_receipts_or_a_strict_policy_writes_the_previous_keys(
    tmp_path, policy, gate,
):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal without evidence")
    # A gate without regenerations is never read without a strict policy.
    graph = _gated(task, max_regenerations=0) if gate else ExecutionGraph(
        topology=[Topology.SERIAL], task=task, nodes=[Node(id="draft", label="Draft the memo")],
    )
    swarm = Swarm(model="m", provider=_Steps({"Draft": ["A draft."], "Judge": ["FAIL"]}),
                  memory=PlannerMemory(path), artifact_dir=None, acceptance=policy)

    swarm.execute(graph)
    PlannerMemory(path).record(task, graph, SwarmResult("done", graph))
    rows = _history_rows(path)
    outcomes = PlannerMemory(path).recall(task, k=2)

    assert [set(row) for row in rows] == [_PREVIOUS_KEYS, _PREVIOUS_KEYS]
    assert [[getattr(outcome, name) for name in _EVIDENCE_FIELDS] for outcome in outcomes] == [
        [None] * 4, [None] * 4,
    ]


def test_recall_keeps_valid_evidence_fields(tmp_path):
    path = tmp_path / "history.jsonl"
    outcome = _make_outcome(
        "Research signal", acceptance="rejected", execution_id="run-1",
        verification=[_GATE, dict(_GATE, gate="second", passed=True, regenerations_used=0)],
        rejected_output_sha256=text_sha256("rejected"), structure_sha256="0" * 64,
    )
    _write_outcome(path, outcome)

    assert PlannerMemory(path).recall(Task("Research signal")) == [outcome]


@pytest.mark.parametrize("field,value", [
    ("execution_id", 1), ("execution_id", ""), ("execution_id", "  "), ("execution_id", ["run"]),
    ("structure_sha256", "abc"), ("structure_sha256", "A" * 64), ("structure_sha256", 1),
    ("rejected_output_sha256", "the rejected text"), ("rejected_output_sha256", "0" * 63),
    ("rejected_output_sha256", True),
    ("verification", {}), ("verification", "judge"), ("verification", [None]),
    ("verification", [_GATE, "judge"]), ("verification", [dict(_GATE, gate=1)]),
    ("verification", [dict(_GATE, target=None)]), ("verification", [dict(_GATE, state=False)]),
    ("verification", [dict(_GATE, passed="true")]), ("verification", [dict(_GATE, passed=1)]),
    ("verification", [dict(_GATE, regenerations_used=-1)]),
    ("verification", [dict(_GATE, regenerations_used=True)]),
    ("verification", [dict(_GATE, regenerations_used=1.0)]),
    ("verification", [{key: value for key, value in _GATE.items() if key != "passed"}]),
])
def test_recall_skips_malformed_evidence_fields_before_ranking(tmp_path, field, value):
    _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value)


def test_legacy_rows_recall_beside_a_malformed_evidence_row(tmp_path):
    path = tmp_path / "history.jsonl"
    previous = asdict(_make_outcome("Research signal previous", acceptance="not_evaluated"))
    oldest = asdict(_make_outcome("Research signal oldest"))
    malformed = asdict(_make_outcome("Research signal malformed",
                                     verification=[dict(_GATE, passed=None)]))
    for name in _EVIDENCE_FIELDS:
        del previous[name], oldest[name]
    for name in _OUTCOME_FIELDS:
        del oldest[name]
    path.write_text("".join(json.dumps(row) + "\n" for row in (previous, malformed, oldest)),
                    encoding="utf-8")

    recalled = PlannerMemory(path).recall(Task("Research signal"), k=5)

    assert [outcome.task_goal for outcome in recalled] == [
        "Research signal previous", "Research signal oldest",
    ]
    assert all(getattr(outcome, name) is None for outcome in recalled for name in _EVIDENCE_FIELDS)


def test_evidence_fields_leave_history_prompts_unchanged(tmp_path):
    path, legacy_path = tmp_path / "history.jsonl", tmp_path / "legacy.jsonl"
    task = Task("Research signal behind a gate")
    result = Swarm(model="m", provider=_Steps({"Draft": ["A draft."], "Judge": ["FAIL: thin"]}),
                   memory=PlannerMemory(path), artifact_dir=None).execute(_gated(task))
    [row] = _history_rows(path)
    previous = {key: value for key, value in row.items() if key in _PREVIOUS_KEYS}
    legacy_path.write_text(json.dumps(previous) + "\n", encoding="utf-8")
    before = path.read_bytes()

    [outcome] = PlannerMemory(path).recall(task)
    [legacy] = PlannerMemory(legacy_path).recall(task)
    prompt = build_user_prompt(task, [asdict(outcome)])
    planner_prompts = []
    for history in (path, legacy_path):
        provider = _HistoryPlanningProvider()
        LLMArchitect(provider, planning_model="offline-test",
                     memory=PlannerMemory(history)).plan(task)
        planner_prompts.append(provider.prompts)

    assert outcome.verification and outcome.structure_sha256 and outcome.execution_id
    assert prompt == build_user_prompt(task, [asdict(legacy)])
    for value in (result.execution_id, outcome.structure_sha256, "verification", "consumed"):
        assert value not in prompt
    assert planner_prompts[0] == planner_prompts[1]
    assert len(planner_prompts[0]) == 1 and "Outcome: success" in planner_prompts[0][0]
    assert path.read_bytes() == before


def test_runs_of_one_template_on_different_goals_share_a_structure_digest(tmp_path):
    path = tmp_path / "history.jsonl"

    def build(task):
        return [
            Node(id="draft", label=f"Draft a memo: {task.goal}"),
            Node(id="judge", label=f"Judge the memo: {task.goal}", depends_on=["draft"],
                 verifies="draft", max_regenerations=1),
        ], Registry()

    architect = ConstrainedArchitect(
        _SelectsTemplate(), [SubGraphTemplate("memo", "A judged memo", build)], max_retries=0,
    )
    results = [
        Swarm(model="m", provider=_Steps({"Draft": ["A memo."], "Judge": ["PASS"]}),
              architect=architect, memory=PlannerMemory(path), artifact_dir=None,
              ).execute(Task(goal))
        for goal in ("Summarize the quarter", "Review the annual budget")
    ]
    first, second = _history_rows(path)

    assert first["task_goal"] != second["task_goal"]
    assert ([node["label"] for node in first["node_outcomes"]]
            != [node["label"] for node in second["node_outcomes"]])
    assert first["execution_id"] == results[0].execution_id != second["execution_id"]
    assert first["structure_sha256"] == second["structure_sha256"]
    assert first["structure_sha256"] == structure_sha256(results[1].graph)


# The keys of a failure row for a run without gates, whose error is not a provider's.
_FAILURE_KEYS = _PREVIOUS_KEYS | {
    "execution_id", "structure_sha256", "failure_category", "failure_type",
}


def _answer(text: str = "A draft.", **kwargs) -> CompletionResult:
    return CompletionResult(text=text, prompt_tokens=1, completion_tokens=1, **kwargs)


class _Answers(Provider):
    """Answers every step with ``answer`` and an LLM_MERGE synthesis with ``merge``.

    An exception is raised instead. When tools are offered, every reply asks
    for another tool call.
    """

    def __init__(self, answer, merge=None) -> None:
        self.answer, self.merge = answer, merge

    async def complete(self, system, prompt, model):
        answer = self.merge if system == MERGE_SYSTEM_PROMPT else self.answer
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def chat(self, system, messages, model, tools=None):
        if tools:
            return _answer("", tool_calls=[ToolCall(id="c1", name="calc.add", arguments={})],
                           stop_reason="tool_use")
        return await super().chat(system, messages, model, tools)


class _AddSession(ToolSession):
    @property
    def tools(self):
        return [ToolSpec(name="calc.add", description="Add numbers", input_schema={"type": "object"})]

    async def call(self, tool_call):
        return ToolResult(tool_call_id=tool_call.id, content="42")


class _AddRuntime(ToolRuntime):
    def open(self, agent):
        @asynccontextmanager
        async def session():
            yield _AddSession()

        return session()


class RateLimitError(Exception):
    """An SDK-style status error whose message quotes the provider's response."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


def _draft(task: Task | None) -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.SERIAL], task=task,
                          nodes=[Node(id="draft", label="Draft the memo")])


# Each failure: the Swarm's settings, the failure class and exception type that
# history records, and whether every node completed.
_FAILURES = {
    "budget-overrun": (
        lambda: dict(provider=_Answers(_answer(cost_usd=0.05)), max_budget_usd=0.01),
        "terminal_accounting", "BudgetReconciliationError", False),
    "provider-refusal": (
        lambda: dict(provider=_Answers(_answer("", stop_reason="refusal"))),
        "provider_rejected", "OutputRefusedError", False),
    "tool-loop-limit": (
        lambda: dict(provider=_Answers(_answer()), tool_runtime=_AddRuntime(),
                     max_tool_iterations=1),
        "quality_defect", "ToolLoopLimitError", False),
    "halt-node-failure": (
        lambda: dict(provider=_Answers(RuntimeError("the draft step failed"))),
        "transient", "RuntimeError", False),
    "synthesis-failure": (
        lambda: dict(provider=_Answers(_answer(), merge=_answer("A half", stop_reason="max_tokens")),
                     synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE)),
        "provider_rejected", "OutputTruncatedError", True),
}


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("case", _FAILURES)
def test_an_opted_in_memory_records_one_row_for_each_failed_run(tmp_path, case, parallel):
    settings, category, failure_type, completed = _FAILURES[case]
    path = tmp_path / "history.jsonl"
    task = Task("Research signal that fails")
    graph = _draft(task)
    swarm = Swarm(model="m", memory=PlannerMemory(path, record_failures=True), artifact_dir=None,
                  parallel=parallel, **settings())

    with pytest.raises(Exception) as caught:
        swarm.execute(graph)
    [row] = _history_rows(path)
    [outcome] = PlannerMemory(path).recall(task)

    assert type(caught.value).__name__ == outcome.failure_type == failure_type
    assert outcome.failure_category == classify_failure(caught.value) == category
    # success keeps its meaning: every node completed, as when only synthesis failed.
    assert outcome.success is completed
    assert [node["status"] for node in outcome.node_outcomes] == [
        "completed" if completed else "failed"
    ]
    assert outcome.acceptance == "not_evaluated"
    assert (outcome.resumed, outcome.latency_complete) == (False, True)
    assert outcome.elapsed_ms is not None
    assert outcome.structure_sha256 == structure_sha256(graph)
    assert outcome.execution_id
    if case == "budget-overrun":
        assert (outcome.total_cost_usd, outcome.cost_is_complete) == (0.05, True)
    if case in ("provider-refusal", "synthesis-failure"):
        # A provider's error keeps its message, which here holds nothing to redact.
        assert outcome.failure_excerpt == str(caught.value)
        assert set(row) == _FAILURE_KEYS | {"failure_excerpt"}
    else:
        assert set(row) == _FAILURE_KEYS


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home" / "Example Person"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "async"])
def test_a_provider_error_excerpt_is_redacted_and_capped(tmp_path, fake_home, parallel):
    organization = "org-" + "A1b2" * 6
    key_file, bundle = fake_home / "keys" / "service.json", tmp_path / "certs" / "bundle.pem"
    message = (f"Rate limit reached for {organization} using {key_file} and {bundle}: limit 20"
               + " on requests per minute" * 20)
    path = tmp_path / "history.jsonl"
    task = Task("Research signal that is rate limited")
    swarm = Swarm(model="m", provider=_Answers(RateLimitError(message)), artifact_dir=None,
                  parallel=parallel, memory=PlannerMemory(path, record_failures=True))

    with pytest.raises(RateLimitError):
        swarm.execute(_draft(task))
    [outcome] = PlannerMemory(path).recall(task)
    stored = path.read_text(encoding="utf-8")

    assert (outcome.failure_category, outcome.failure_type) == ("transient", "RateLimitError")
    assert outcome.failure_excerpt.startswith(
        f"Rate limit reached for org-[redacted] using {os.path.join('~', 'keys', 'service.json')}"
        " and <path>: limit 20 on requests per minute on requests"
    )
    assert len(outcome.failure_excerpt) == 200 and outcome.failure_excerpt.endswith("…")
    for private in (organization, str(fake_home), fake_home.as_posix(), str(tmp_path),
                    tmp_path.as_posix(), "Example Person", tmp_path.name, "bundle.pem"):
        assert private not in outcome.failure_excerpt
    for private in (organization, "Example Person", tmp_path.name, "bundle.pem"):
        assert private not in stored


def test_an_error_that_is_not_a_providers_keeps_no_excerpt(tmp_path):
    organization = "org-" + "A1b2" * 6
    path = tmp_path / "history.jsonl"
    swarm = Swarm(model="m", provider=_Answers(RuntimeError(f"tool failed for {organization}")),
                  artifact_dir=None, memory=PlannerMemory(path, record_failures=True))

    with pytest.raises(RuntimeError):
        swarm.execute(_draft(Task("Research signal")))
    [row] = _history_rows(path)

    assert (row["failure_category"], row["failure_type"]) == ("transient", "RuntimeError")
    assert "failure_excerpt" not in row
    assert organization not in path.read_text(encoding="utf-8")


def test_path_redaction_replaces_the_home_folder_and_other_absolute_paths(fake_home):
    native = str(fake_home / "keys" / "service.json")
    assert _redact.redact_paths(f"read {native}") == f"read {os.path.join('~', 'keys', 'service.json')}"
    assert _redact.redact_paths(f"read {fake_home.as_posix()}/keys") == "read ~/keys"
    backslash = "\\"
    for absolute in (
        "D:" + backslash + backslash.join(["data", "run", "bundle.pem"]),
        "E:/cache/run/out.json",
        backslash * 2 + backslash.join(["fileserver", "share", "run.txt"]),
        "/srv/app/keys/service.json",
        "file:///srv/app/key.json",
    ):
        assert _redact.redact_paths(f"open '{absolute}' failed") == "open '<path>' failed"
        assert _redact.redact_paths(f"open {absolute}: failed") == "open <path>: failed"
        assert _redact.redact_paths(f"open {absolute} failed") == "open <path>"
    for kept in ("see https://api.example.com/v1/models/list", "relative/dir/file.txt",
                 "./local/file.txt", "~/project/out.json", "ratio 1/2/3", "org-chart"):
        assert _redact.redact_paths(kept) == kept


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "async"])
def test_a_failed_history_write_is_logged_and_the_runs_own_exception_raised(
    tmp_path, caplog, parallel,
):
    history = tmp_path / "history.jsonl"
    history.mkdir()  # A directory: no row can be appended to it.
    failure = RuntimeError("the draft step failed")
    swarm = Swarm(model="m", provider=_Answers(failure), artifact_dir=None, parallel=parallel,
                  memory=PlannerMemory(history, record_failures=True))

    with caplog.at_level(logging.WARNING, logger="smythe.swarm"):
        with pytest.raises(RuntimeError) as caught:
            swarm.execute(_draft(Task("Research signal")))

    assert caught.value is failure
    [logged] = [record for record in caplog.records if "planner memory" in record.getMessage()]
    assert (logged.name, logged.levelno) == ("smythe.swarm", logging.WARNING)
    assert isinstance(logged.exc_info[1], OSError)
    assert list(history.iterdir()) == []


class _RecordOnly:
    """A memory of the kind Swarm has always accepted: it implements only record()."""

    def __init__(self) -> None:
        self.recorded = []

    def record(self, task, graph, result):
        self.recorded.append(result)


class _RecordsFailures(_RecordOnly):
    def __init__(self, record_failures) -> None:
        super().__init__()
        self.record_failures = record_failures
        self.failures = []

    def record_failure(self, task, graph, exc, **evidence):
        self.failures.append((exc, sorted(evidence)))


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "async"])
def test_only_a_memory_that_opts_in_records_a_failed_run(tmp_path, parallel):
    path = tmp_path / "history.jsonl"
    failure = RuntimeError("the draft step failed")
    default, record_only, mock = PlannerMemory(path), _RecordOnly(), Mock()
    truthy, opted = _RecordsFailures(record_failures=1), _RecordsFailures(record_failures=True)

    def fail(memory, graph):
        with pytest.raises(RuntimeError) as caught:
            Swarm(model="m", provider=_Answers(failure), memory=memory, artifact_dir=None,
                  parallel=parallel).execute(graph)
        assert caught.value is failure

    for memory in (default, record_only, mock, truthy, opted):
        fail(memory, _draft(Task("Research signal")))
    # A graph executed without a task is not recorded, as a finished one is not.
    fail(opted, _draft(None))

    assert default.record_failures is False and not path.exists()
    assert record_only.recorded == [] and truthy.failures == []
    mock.record.assert_not_called()
    mock.record_failure.assert_not_called()
    assert opted.recorded == []
    assert opted.failures == [(failure, ["budget", "execution_id", "timing", "trace"])]


@pytest.mark.parametrize("interrupt", [
    KeyboardInterrupt, SystemExit, asyncio.CancelledError,
    lambda: ProviderAccountingCancelledError("Native generation was cancelled after dispatch"),
], ids=["keyboard-interrupt", "system-exit", "cancelled", "cancelled-after-dispatch"])
def test_interrupts_exits_and_cancellation_are_not_recorded(tmp_path, interrupt):
    path = tmp_path / "history.jsonl"
    error = interrupt()
    swarm = Swarm(model="m", provider=_Answers(error), artifact_dir=None,
                  memory=PlannerMemory(path, record_failures=True))

    with pytest.raises(BaseException) as caught:
        swarm.execute(_draft(Task("Research signal")))

    assert caught.value is error
    assert not path.exists()


def test_a_resumed_run_that_fails_records_that_it_resumed(tmp_path):
    path, store = tmp_path / "history.jsonl", FileCheckpointStore(tmp_path / "checkpoints")
    task = Task("Research signal after a crash")
    graph = ExecutionGraph(topology=[Topology.SERIAL], task=task, nodes=[
        Node(id="a", label="Research A"),
        Node(id="b", label="Assemble findings", depends_on=["a"]),
    ])
    provider = _Steps({"Research": ["Found it."], "Assemble": [RuntimeError("process died")]})

    def fail(call):
        with pytest.raises(RuntimeError, match="process died"):
            call(Swarm(model="m", provider=provider, checkpoint_store=store, artifact_dir=None,
                       memory=PlannerMemory(path, record_failures=True)))

    fail(lambda swarm: swarm.execute(graph))
    [execution_id] = store.list_ids()
    fail(lambda swarm: swarm.resume(execution_id))
    first, resumed = _history_rows(path)
    outcomes = PlannerMemory(path).recall(task, k=2)

    assert first["execution_id"] == resumed["execution_id"] == execution_id
    assert (first["resumed"], first["latency_complete"]) == (False, True)
    assert (resumed["resumed"], resumed["latency_complete"]) == (True, False)
    assert [node["status"] for node in resumed["node_outcomes"]] == ["completed", "failed"]
    assert [(outcome.failure_type, outcome.resumed) for outcome in outcomes] == [
        ("RuntimeError", False), ("RuntimeError", True),
    ]


def test_record_failure_writes_only_the_fields_it_has(tmp_path):
    path = tmp_path / "history.jsonl"
    task = Task("Research signal by hand")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="Research A", status=NodeStatus.FAILED),
    ])

    PlannerMemory(path).record_failure(task, graph, TimeoutError("Node 'a' timed out after 5s"))
    [row] = _history_rows(path)

    assert set(row) == _PREVIOUS_KEYS | {"structure_sha256", "failure_category", "failure_type"}
    assert (row["failure_category"], row["failure_type"]) == ("transient", "TimeoutError")
    assert (row["success"], row["total_cost_usd"], row["total_duration_ms"]) == (False, 0.0, 0)
    assert [row[name] for name in _OUTCOME_FIELDS] == [None] * 5 + ["not_evaluated"]


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_record_failures_must_be_true_or_false(value):
    with pytest.raises(TypeError, match="record_failures must be True or False"):
        PlannerMemory("missing-history.jsonl", record_failures=value)


def test_recall_returns_valid_failure_rows(tmp_path):
    path = tmp_path / "history.jsonl"
    outcome = _make_outcome(
        "Research signal", success=False, acceptance="not_evaluated", execution_id="run-1",
        structure_sha256="0" * 64, failure_category="provider_rejected",
        failure_type="OutputRefusedError", failure_excerpt="the model refused to respond",
    )
    _write_outcome(path, outcome)

    assert PlannerMemory(path).recall(Task("Research signal")) == [outcome]


@pytest.mark.parametrize("field,value", [
    ("failure_category", 1), ("failure_category", ""), ("failure_category", "Budget overrun"),
    ("failure_category", ["transient"]), ("failure_type", 1), ("failure_type", ""),
    ("failure_type", "Runtime Error"), ("failure_type", "1Error"), ("failure_excerpt", 1),
    ("failure_excerpt", ""), ("failure_excerpt", "x" * 201), ("failure_excerpt", ["refused"]),
])
def test_recall_skips_malformed_failure_fields_before_ranking(tmp_path, field, value):
    _assert_bad_row_is_skipped_without_changing_ranking(tmp_path, field, value)


def test_failure_rows_leave_history_prompts_unchanged(tmp_path):
    path, legacy_path = tmp_path / "history.jsonl", tmp_path / "legacy.jsonl"
    task = Task("Research signal that fails")
    with pytest.raises(RateLimitError):
        Swarm(model="m", provider=_Answers(RateLimitError("Rate limit reached on requests")),
              artifact_dir=None, memory=PlannerMemory(path, record_failures=True),
              ).execute(_draft(task))
    [row] = _history_rows(path)
    previous = {key: value for key, value in row.items() if key in _PREVIOUS_KEYS}
    legacy_path.write_text(json.dumps(previous) + "\n", encoding="utf-8")
    before = path.read_bytes()

    [outcome] = PlannerMemory(path).recall(task)
    [legacy] = PlannerMemory(legacy_path).recall(task)
    prompt = build_user_prompt(task, [asdict(outcome)])
    planner_prompts = []
    for history in (path, legacy_path):
        provider = _HistoryPlanningProvider()
        LLMArchitect(provider, planning_model="offline-test",
                     memory=PlannerMemory(history)).plan(task)
        planner_prompts.append(provider.prompts)

    assert all(getattr(outcome, name) for name in _FAILURE_FIELDS)
    assert prompt == build_user_prompt(task, [asdict(legacy)])
    for value in ("transient", "RateLimitError", "Rate limit", outcome.execution_id):
        assert value not in prompt
    assert planner_prompts[0] == planner_prompts[1]
    assert len(planner_prompts[0]) == 1 and "Outcome: failure" in planner_prompts[0][0]
    assert path.read_bytes() == before


def test_a_run_that_failed_after_every_node_completed_reaches_the_planner_as_a_failure(
    tmp_path,
):
    path, legacy_path = tmp_path / "history.jsonl", tmp_path / "legacy.jsonl"
    task = Task("Research signal that fails at synthesis")
    settings = _FAILURES["synthesis-failure"][0]
    with pytest.raises(OutputTruncatedError):
        Swarm(model="m", memory=PlannerMemory(path, record_failures=True), artifact_dir=None,
              **settings()).execute(_draft(task))
    [row] = _history_rows(path)
    # The same run as a row without failure fields, recorded as one that failed.
    failed = {key: value for key, value in row.items() if key in _PREVIOUS_KEYS}
    legacy_path.write_text(json.dumps({**failed, "success": False}) + "\n", encoding="utf-8")
    before = path.read_bytes()

    planner_prompts = []
    for history in (path, legacy_path):
        provider = _HistoryPlanningProvider()
        LLMArchitect(provider, planning_model="offline-test",
                     memory=PlannerMemory(history)).plan(task)
        planner_prompts.extend(provider.prompts)

    # success keeps its meaning in the row; the planner is told the run failed.
    assert (row["success"], row["failure_category"]) == (True, "provider_rejected")
    assert planner_prompts[0] == planner_prompts[1]
    assert "Outcome: failure" in planner_prompts[0]
    assert "Outcome: success" not in planner_prompts[0]
    assert path.read_bytes() == before
