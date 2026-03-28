"""Tests for the YAML DAG loader."""

import os
import tempfile

import pytest

from smythe.graph import ExecutionGraph, Topology
from smythe.loader import load_graph, load_graph_from_string


SERIAL_YAML = """\
topology: serial

nodes:
  - id: step-1
    label: "Do the thing"
"""

FORK_JOIN_YAML = """\
topology: [fork_join, serial]

nodes:
  - id: research-a
    label: "Research competitor A"

  - id: research-b
    label: "Research competitor B"

  - id: synthesize
    label: "Combine findings into report"
    depends_on: [research-a, research-b]
"""

AGENT_YAML = """\
topology: serial

nodes:
  - id: analyst
    label: "Analyze data"
    agent:
      name: DataAnalyst
      persona: "You are a senior data analyst."
      capabilities: [analysis, statistics]

  - id: writer
    label: "Write report"
    depends_on: [analyst]
    agent:
      name: ReportWriter
      persona: "You are a technical writer."
      capabilities: [writing]
"""

BAD_TOPOLOGY_YAML = """\
topology: warp_drive

nodes:
  - id: x
    label: "Go fast"
"""

MISSING_DEP_YAML = """\
topology: serial

nodes:
  - id: step-1
    label: "First"
    depends_on: [nonexistent]
"""


def test_load_serial_graph():
    graph, registry = load_graph_from_string(SERIAL_YAML)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "step-1"
    assert graph.nodes[0].label == "Do the thing"


def test_load_fork_join_graph():
    graph, registry = load_graph_from_string(FORK_JOIN_YAML)

    assert graph.topology == [Topology.FORK_JOIN, Topology.SERIAL]
    assert len(graph.nodes) == 3

    synth = next(n for n in graph.nodes if n.id == "synthesize")
    assert set(synth.depends_on) == {"research-a", "research-b"}


def test_load_with_agent_personas():
    graph, registry = load_graph_from_string(AGENT_YAML)

    assert len(graph.nodes) == 2
    analyst_node = graph.nodes[0]
    writer_node = graph.nodes[1]

    assert analyst_node.agent_id is not None
    assert writer_node.agent_id is not None

    analyst_agent = registry.get(analyst_node.agent_id)
    assert analyst_agent is not None
    assert analyst_agent.profile.name == "DataAnalyst"
    assert analyst_agent.profile.persona == "You are a senior data analyst."
    assert "analysis" in analyst_agent.profile.capabilities

    writer_agent = registry.get(writer_node.agent_id)
    assert writer_agent is not None
    assert writer_agent.profile.name == "ReportWriter"


def test_load_invalid_topology():
    with pytest.raises(ValueError, match="Unknown topology"):
        load_graph_from_string(BAD_TOPOLOGY_YAML)


def test_load_missing_dependency():
    with pytest.raises(ValueError, match="unknown node"):
        load_graph_from_string(MISSING_DEP_YAML)


def test_load_from_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(FORK_JOIN_YAML)
        path = f.name

    try:
        graph, registry = load_graph(path)
        assert graph.topology == [Topology.FORK_JOIN, Topology.SERIAL]
        assert len(graph.nodes) == 3
    finally:
        os.unlink(path)


def test_load_topology_as_string():
    """Topology can be specified as a plain string instead of a list."""
    graph, _ = load_graph_from_string(SERIAL_YAML)
    assert graph.topology == [Topology.SERIAL]


def test_load_node_without_label_uses_id():
    yaml_str = """\
topology: serial

nodes:
  - id: my-node
"""
    graph, _ = load_graph_from_string(yaml_str)
    assert graph.nodes[0].label == "my-node"


def test_load_non_dict_node_raises():
    """Bare strings in the nodes list should produce a clear error."""
    yaml_str = """\
topology: serial

nodes:
  - just-a-string
  - id: valid-node
    label: "Valid"
"""
    with pytest.raises(ValueError, match="Node at index 0 must be a mapping"):
        load_graph_from_string(yaml_str)


def test_load_non_list_nodes_raises():
    """nodes: as a string instead of a list should produce a clear error."""
    yaml_str = """\
topology: serial

nodes: "not a list"
"""
    with pytest.raises(ValueError, match="'nodes' must be a list"):
        load_graph_from_string(yaml_str)


def test_load_failure_policy():
    yaml_str = """\
topology: serial
nodes:
  - id: step-1
    label: "Retry step"
    failure_policy: retry
    max_retries: 3
  - id: step-2
    label: "Skipable"
    failure_policy: skip
    depends_on: [step-1]
"""
    from smythe.graph import FailurePolicy
    graph, _ = load_graph_from_string(yaml_str)
    assert graph.nodes[0].failure_policy == FailurePolicy.RETRY
    assert graph.nodes[0].max_retries == 3
    assert graph.nodes[1].failure_policy == FailurePolicy.SKIP


def test_load_invalid_failure_policy():
    yaml_str = """\
topology: serial
nodes:
  - id: step-1
    label: "Bad"
    failure_policy: explode
"""
    with pytest.raises(ValueError, match="Unknown failure_policy"):
        load_graph_from_string(yaml_str)


def test_load_required_capabilities():
    yaml_str = """\
topology: serial
nodes:
  - id: research
    label: "Research task"
    required_capabilities: [research, summarize]
"""
    graph, _ = load_graph_from_string(yaml_str)
    assert graph.nodes[0].required_capabilities == ["research", "summarize"]


def test_load_depends_on_non_list_raises():
    yaml_str = """\
topology: serial
nodes:
  - id: step-1
    label: "Bad deps"
    depends_on: "not-a-list"
"""
    with pytest.raises(ValueError, match="'depends_on' on node 'step-1' must be a list"):
        load_graph_from_string(yaml_str)


def test_load_metadata_non_dict_raises():
    yaml_str = """\
topology: serial
nodes:
  - id: step-1
    label: "Bad meta"
    metadata: "not-a-dict"
"""
    with pytest.raises(ValueError, match="'metadata' on node 'step-1' must be a mapping"):
        load_graph_from_string(yaml_str)


def test_load_required_capabilities_non_list_raises():
    yaml_str = """\
topology: serial
nodes:
  - id: step-1
    label: "Bad caps"
    required_capabilities: "not-a-list"
"""
    with pytest.raises(ValueError, match="'required_capabilities' on node 'step-1' must be a list"):
        load_graph_from_string(yaml_str)
