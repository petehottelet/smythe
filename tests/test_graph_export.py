"""Tests for graph export: Mermaid, DOT, and JSON renderings."""

import json

from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology


def _graph() -> ExecutionGraph:
    a = Node(id="research", label="Research the topic", agent_id="agent-1")
    b = Node(id="review", label='Red-team "everything"', depends_on=["research"])
    a.status = NodeStatus.COMPLETED
    a.metadata["cost_usd"] = 0.0012
    b.status = NodeStatus.FAILED
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])


def test_to_mermaid_structure_and_edges():
    out = _graph().to_mermaid()
    assert out.startswith("flowchart TD")
    assert 'research["agent-1: Research the topic"]' in out
    assert "research --> review" in out


def test_to_mermaid_status_classes():
    out = _graph().to_mermaid()
    assert "class research done" in out
    assert "class review failed" in out


def test_to_mermaid_escapes_label_breakers():
    out = _graph().to_mermaid()
    assert '"everything"' not in out.split("\n")[2]  # quotes escaped in review label
    assert "#quot;everything#quot;" in out


def test_to_mermaid_is_deterministic():
    assert _graph().to_mermaid() == _graph().to_mermaid()


def test_pending_nodes_get_no_class():
    g = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label="L")])
    out = g.to_mermaid()
    assert "classDef" not in out
    assert "class " not in out.replace("classDef", "")


def test_to_dot_structure():
    out = _graph().to_dot()
    assert out.startswith("digraph smythe {")
    assert '"research" -> "review";' in out
    assert out.rstrip().endswith("}")


def test_to_dot_escapes_quotes():
    assert '\\"everything\\"' in _graph().to_dot()


def test_to_json_roundtrips_through_json_dumps():
    data = _graph().to_json()
    parsed = json.loads(json.dumps(data))
    assert parsed["topology"] == ["serial"]
    research = parsed["nodes"][0]
    assert research["status"] == "completed"
    assert research["cost_usd"] == 0.0012
    assert parsed["nodes"][1]["depends_on"] == ["research"]


def test_to_mermaid_theme_prepends_house_header():
    out = _graph().to_mermaid(theme=True)
    first, second = out.split("\n")[:2]
    assert first.startswith("%%{init:")
    assert "Georgia" in first
    assert second == "flowchart TD"
    # Default stays clean for snapshot stability.
    assert _graph().to_mermaid().startswith("flowchart TD")
