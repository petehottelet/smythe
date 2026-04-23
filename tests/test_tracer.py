"""Dedicated tests for Tracer and Span lifecycle."""

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


def test_span_metadata_is_independent():
    """Each Span's metadata dict should be independent."""
    s1 = Span(node_id="1", label="A")
    s2 = Span(node_id="2", label="B")
    s1.metadata["key"] = "val"
    assert "key" not in s2.metadata
