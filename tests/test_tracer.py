"""Dedicated tests for Tracer and Span lifecycle."""

import pytest

from fakes import ManualClock
from smythe.graph import Node, NodeStatus
from smythe.tracer import Span, Tracer


def test_span_duration_ms():
    """Span.duration_ms should reflect end_time - start_time in milliseconds."""
    span = Span(node_id="n1", label="Test", start_time=1.0, end_time=1.5)
    assert span.duration_ms == 500.0


def test_span_defaults():
    """Span should have sensible defaults for optional fields."""
    span = Span(node_id="n1", label="Test")
    assert span.agent_id is None
    assert span.start_time == 0.0
    assert span.end_time == 0.0
    assert span.status == ""
    assert span.error is None
    assert span.metadata == {}


def test_tracer_start_and_end():
    """on_node_start followed by on_node_end should produce a completed span."""
    tracer = Tracer()
    node = Node(id="a", label="Step A", agent_id="agent-1")
    node.status = NodeStatus.COMPLETED

    tracer.on_node_start(node)
    tracer.on_node_end(node)

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.node_id == "a"
    assert span.label == "Step A"
    assert span.agent_id == "agent-1"
    assert span.status == "completed"
    assert span.duration_ms >= 0


def test_tracer_end_without_start():
    """on_node_end with no prior on_node_start should be a no-op."""
    tracer = Tracer()
    node = Node(id="x", label="Ghost")
    tracer.on_node_end(node)
    assert len(tracer.spans) == 0


def test_tracer_error_records_message():
    """on_node_error should set the error field on the active span."""
    tracer = Tracer()
    node = Node(id="b", label="Broken")

    tracer.on_node_start(node)
    tracer.on_node_error(node, RuntimeError("boom"))

    assert tracer._active["b"].error == "boom"


def test_tracer_error_without_start():
    """on_node_error with no prior on_node_start should be a no-op (no crash)."""
    tracer = Tracer()
    node = Node(id="z", label="Phantom")
    tracer.on_node_error(node, ValueError("nope"))


def test_tracer_summary_shape():
    """summary() should return dicts with the expected keys."""
    tracer = Tracer()
    node = Node(id="c", label="Task C")
    node.status = NodeStatus.COMPLETED

    tracer.on_node_start(node)
    tracer.on_node_end(node)

    summaries = tracer.summary()
    assert len(summaries) == 1
    s = summaries[0]
    assert set(s.keys()) == {"node_id", "label", "agent_id", "status", "duration_ms", "error"}
    assert s["node_id"] == "c"
    assert s["error"] is None


def test_tracer_multiple_spans():
    """Tracer should correctly track multiple sequential spans."""
    tracer = Tracer()
    for i in range(3):
        node = Node(id=f"n{i}", label=f"Step {i}")
        node.status = NodeStatus.COMPLETED
        tracer.on_node_start(node)
        tracer.on_node_end(node)

    assert len(tracer.spans) == 3
    assert [s.node_id for s in tracer.spans] == ["n0", "n1", "n2"]


def test_tracer_error_then_end_preserves_error():
    """If on_node_error is called before on_node_end, the error is preserved in the span."""
    tracer = Tracer()
    node = Node(id="e", label="ErrorThenEnd")
    node.status = NodeStatus.FAILED

    tracer.on_node_start(node)
    tracer.on_node_error(node, RuntimeError("kaboom"))
    tracer.on_node_end(node)

    assert len(tracer.spans) == 1
    assert tracer.spans[0].error == "kaboom"
    assert tracer.spans[0].status == "failed"


def test_tracer_records_tool_calls_in_serializable_summary():
    """Tool telemetry should survive span finalization and summary export."""
    tracer = Tracer()
    node = Node(id="tool-user", label="Search docs", status=NodeStatus.COMPLETED)

    tracer.on_node_start(node)
    tracer.on_tool_call(node, "docs.search", duration_ms=12.345, is_error=False)
    tracer.on_tool_call(node, "docs.fetch", duration_ms=4.0, is_error=True)
    tracer.on_node_end(node)

    assert tracer.summary()[0]["tool_calls"] == [
        {"tool": "docs.search", "duration_ms": 12.3, "is_error": False},
        {"tool": "docs.fetch", "duration_ms": 4.0, "is_error": True},
    ]


def test_span_metadata_is_independent():
    """Each Span's metadata dict should be independent."""
    s1 = Span(node_id="1", label="A")
    s2 = Span(node_id="2", label="B")
    s1.metadata["key"] = "val"
    assert "key" not in s2.metadata


# ---------------------------------------------------------------------------
# Monotonic span bounds, attempt and outcome, and process-local timing
# ---------------------------------------------------------------------------


def _manual_tracer() -> tuple[ManualClock, Tracer]:
    clock = ManualClock()
    return clock, Tracer(clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9)


def _finish(tracer: Tracer, node: Node, status: NodeStatus) -> None:
    node.status = status
    tracer.on_node_end(node)


def test_span_records_monotonic_bounds_attempt_phase_and_outcome():
    clock, tracer = _manual_tracer()
    node = Node(id="a", label="A")

    tracer.on_node_start(node, attempt=2)
    clock.advance(1.5)
    _finish(tracer, node, NodeStatus.COMPLETED)

    [span] = tracer.spans
    assert (span.attempt, span.phase, span.outcome) == (2, "execution", "completed")
    assert span.monotonic_end - span.monotonic_start == 1.5
    assert (span.start_time, span.end_time) == (1000.0, 1001.5)
    assert span.duration_ms == 1500


def test_span_duration_uses_the_monotonic_clock_across_a_wall_clock_step():
    clock, tracer = _manual_tracer()
    node = Node(id="a", label="A")

    tracer.on_node_start(node)
    clock.advance(2)
    clock.jump_wall(-3600)
    _finish(tracer, node, NodeStatus.COMPLETED)

    [span] = tracer.spans
    assert span.end_time - span.start_time == -3598
    assert span.duration_ms == 2000
    assert tracer.summary()[0]["duration_ms"] == 2000


def test_span_phase_follows_the_node_role_unless_given():
    _, tracer = _manual_tracer()
    judge = Node(id="j", label="Judge", verifies="w")
    merge = Node(id="__synthesis__", label="Synthesis merge")

    tracer.on_node_start(judge)
    tracer.on_node_start(merge, phase="synthesis")
    _finish(tracer, judge, NodeStatus.COMPLETED)
    _finish(tracer, merge, NodeStatus.COMPLETED)

    assert [span.phase for span in tracer.spans] == ["verification", "synthesis"]


def test_span_outcome_distinguishes_failure_timeout_and_cancellation():
    _, tracer = _manual_tracer()
    failed, timed_out, cancelled = (Node(id=name, label=name) for name in ("f", "t", "c"))
    for node in (failed, timed_out, cancelled):
        tracer.on_node_start(node)
    tracer.on_node_error(failed, RuntimeError("boom"))
    tracer.on_node_error(timed_out, TimeoutError("Node 't' timed out after 1s"))
    _finish(tracer, failed, NodeStatus.FAILED)
    _finish(tracer, timed_out, NodeStatus.FAILED)
    _finish(tracer, cancelled, NodeStatus.PENDING)

    assert [(span.status, span.outcome) for span in tracer.spans] == [
        ("failed", "failed"), ("failed", "timed_out"), ("pending", "cancelled"),
    ]


def test_summary_keys_are_unchanged_by_timing_fields():
    _, tracer = _manual_tracer()
    node = Node(id="c", label="Task C")
    tracer.on_node_start(node, attempt=0, phase="execution")
    tracer.on_wait("retry_backoff", 1.0, node)
    _finish(tracer, node, NodeStatus.COMPLETED)

    [entry] = tracer.summary()
    assert set(entry) == {"node_id", "label", "agent_id", "status", "duration_ms", "error"}


def test_timing_reports_phase_totals_waits_and_elapsed_time():
    clock, tracer = _manual_tracer()
    work = Node(id="w", label="Work")
    judge = Node(id="j", label="Judge", verifies="w")

    tracer.on_node_start(work, attempt=0)
    clock.advance(3)
    _finish(tracer, work, NodeStatus.COMPLETED)
    tracer.on_node_start(judge, attempt=0)
    clock.advance(1)
    _finish(tracer, judge, NodeStatus.COMPLETED)
    with tracer.phase("synthesis"):
        clock.advance(0.5)
    tracer.add_phase("checkpoint", 0.25)
    tracer.on_wait("retry_backoff", 2)
    tracer.on_wait("admission", 0.5, work)

    assert tracer.timing() == {
        "clock": "monotonic",
        "elapsed_ms": 4500.0,
        "summed_node_ms": 4000.0,
        "phases_ms": {
            "routing": None, "planning": None, "recovery": None,
            "execution": 3000.0, "verification": 1000.0, "synthesis": 500.0,
            "supervision": 0.0, "checkpoint": 250.0, "waiting": 2500.0,
        },
        "waiting_ms": {"retry_backoff": 2000.0, "admission": 500.0, "supervision": 0.0},
    }


def test_overlapping_spans_sum_beyond_elapsed_time():
    clock, tracer = _manual_tracer()
    a, b = Node(id="a", label="A"), Node(id="b", label="B")

    tracer.on_node_start(a)
    tracer.on_node_start(b)
    clock.advance(3)
    _finish(tracer, a, NodeStatus.COMPLETED)
    _finish(tracer, b, NodeStatus.COMPLETED)

    timing = tracer.timing()
    assert timing["summed_node_ms"] == timing["phases_ms"]["execution"] == 6000
    assert timing["elapsed_ms"] == 3000


def test_phase_is_recorded_when_its_block_raises():
    clock, tracer = _manual_tracer()

    with pytest.raises(RuntimeError, match="planner failed"):
        with tracer.phase("planning"):
            clock.advance(2)
            raise RuntimeError("planner failed")

    assert tracer.timing()["phases_ms"]["planning"] == 2000


@pytest.mark.parametrize("call", [
    lambda tracer: tracer.add_phase("thinking", 1),
    lambda tracer: tracer.add_phase("waiting", 1),
    lambda tracer: tracer.add_phase("planning", -1),
    lambda tracer: tracer.add_phase("planning", float("nan")),
    lambda tracer: tracer.add_phase("planning", True),
    lambda tracer: tracer.on_wait("admission", float("inf")),
    lambda tracer: tracer.on_wait("admission", -0.5),
])
def test_invalid_phase_or_duration_is_rejected(call):
    with pytest.raises(ValueError):
        call(Tracer())


def test_default_clocks_still_produce_wall_timestamps():
    tracer = Tracer()
    node = Node(id="a", label="A")
    tracer.on_node_start(node)
    _finish(tracer, node, NodeStatus.COMPLETED)

    [span] = tracer.spans
    assert span.start_time > 1_000_000_000
    assert span.monotonic_end >= span.monotonic_start
    assert tracer.timing()["elapsed_ms"] >= 0
