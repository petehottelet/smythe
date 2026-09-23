"""Tests for the YAML DAG loader."""

import os
import tempfile

import pytest

from smythe.graph import ExecutionGraph, Topology
from smythe.loader import (
    build_graph_from_dict,
    build_graph_from_model_output,
    load_graph,
    load_graph_from_string,
)


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


TIMEOUT_YAML = """\
topology: serial

nodes:
  - id: capped
    label: "Step with a timeout"
    timeout_s: 30
  - id: uncapped
    label: "Step without one"
    depends_on: [capped]
"""


def test_timeout_s_parsed_as_float():
    graph, _ = load_graph_from_string(TIMEOUT_YAML)
    capped = next(n for n in graph.nodes if n.id == "capped")
    uncapped = next(n for n in graph.nodes if n.id == "uncapped")
    assert capped.timeout_s == 30.0
    assert isinstance(capped.timeout_s, float)
    assert uncapped.timeout_s is None


@pytest.mark.parametrize("bad_value", ['"thirty"', "true", "-5", "0"])
def test_timeout_s_rejects_invalid_values(bad_value):
    yaml_str = f"""\
topology: serial

nodes:
  - id: bad
    label: "Bad timeout"
    timeout_s: {bad_value}
"""
    with pytest.raises(ValueError, match="timeout_s"):
        load_graph_from_string(yaml_str)


def test_max_tool_iterations_parsed():
    yaml_str = """\
topology: serial

nodes:
  - id: tooluser
    label: "Uses tools"
    max_tool_iterations: 5
"""
    graph, _ = load_graph_from_string(yaml_str)
    assert graph.nodes[0].max_tool_iterations == 5


@pytest.mark.parametrize("bad_value", ['"five"', "true", "0", "-1", "2.5"])
def test_max_tool_iterations_rejects_invalid(bad_value):
    yaml_str = f"""\
topology: serial

nodes:
  - id: bad
    label: "Bad"
    max_tool_iterations: {bad_value}
"""
    with pytest.raises(ValueError, match="max_tool_iterations"):
        load_graph_from_string(yaml_str)


def test_loader_stamps_agent_name_for_rendering():
    yaml_str = """\
topology: serial

nodes:
  - id: research
    label: "Research the topic"
    agent:
      name: Researcher
      persona: "You research."
"""
    graph, _registry = load_graph_from_string(yaml_str)

    node = graph.nodes[0]
    assert node.metadata["agent_name"] == "Researcher"
    assert "Researcher: Research the topic" in str(graph)
    assert node.agent_id not in graph.to_mermaid()


def test_loader_parses_attach_dep_artifacts(tmp_path):
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text(
        "topology: serial\n"
        "nodes:\n"
        "  - id: a\n"
        "    label: make\n"
        "  - id: b\n"
        "    label: judge\n"
        "    depends_on: [a]\n"
        "    attach_dep_artifacts: true\n",
        encoding="utf-8",
    )
    graph, _ = load_graph(str(yaml_path))
    by_id = {n.id: n for n in graph.nodes}
    assert by_id["b"].attach_dep_artifacts is True
    assert by_id["a"].attach_dep_artifacts is False


def test_loader_rejects_non_bool_attach_dep_artifacts(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        "topology: serial\n"
        "nodes:\n"
        "  - id: a\n"
        "    label: x\n"
        "    attach_dep_artifacts: yes please\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="attach_dep_artifacts"):
        load_graph(str(yaml_path))


# ---------------------------------------------------------------------------
# Verification gating
# ---------------------------------------------------------------------------


def test_verifier_node_survives_the_loader():
    """A plan that asks for verification must get it.

    ``verifies``/``max_regenerations`` are the only way a generated plan
    or a YAML file can turn on verification gating: the executor reads
    them off the Node, and nothing else sets them.
    """
    graph, _ = build_graph_from_dict({
        "topology": ["serial"],
        "nodes": [
            {"id": "draft", "label": "Write it"},
            {
                "id": "check",
                "label": "Check it",
                "depends_on": ["draft"],
                "verifies": "draft",
                "max_regenerations": 2,
            },
        ],
    })

    check = next(n for n in graph.nodes if n.id == "check")
    assert check.verifies == "draft"
    assert check.max_regenerations == 2


def test_verifies_must_name_a_real_node():
    with pytest.raises(ValueError, match="verifies"):
        build_graph_from_dict({
            "topology": ["serial"],
            "nodes": [
                {"id": "draft", "label": "Write it"},
                {"id": "check", "label": "Check it", "verifies": "nonexistent"},
            ],
        })


def test_max_regenerations_rejects_negative():
    with pytest.raises(ValueError, match="max_regenerations"):
        build_graph_from_dict({
            "topology": ["serial"],
            "nodes": [
                {"id": "draft", "label": "Write it"},
                {
                    "id": "check", "label": "Check it",
                    "verifies": "draft", "max_regenerations": -1,
                },
            ],
        })


# ---------------------------------------------------------------------------
# Malformed values raise ValueError, never AttributeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", [
    {"nodes": [{"id": "a", "label": "x", "failure_policy": 1}]},
    {"nodes": [{"id": "a", "label": "x", "agent": "Researcher"}]},
    {"topology": [1], "nodes": [{"id": "a", "label": "x"}]},
    {"topology": {"serial": True}, "nodes": [{"id": "a", "label": "x"}]},
])
def test_malformed_values_raise_value_error(plan):
    with pytest.raises(ValueError):
        build_graph_from_dict(plan)
    with pytest.raises(ValueError):
        build_graph_from_model_output(plan)


# ---------------------------------------------------------------------------
# Model-generated plans: strict schema
# ---------------------------------------------------------------------------


def _node(**fields):
    return {"id": "a", "label": "Do the work", **fields}


def test_model_plan_builds_the_documented_schema():
    graph, registry = build_graph_from_model_output({
        "topology": ["fork_join", "serial"],
        "nodes": [
            {"id": "research-a", "label": "Research A", "depends_on": [],
             "agent": {"name": "A", "persona": "You research.", "capabilities": ["research"]}},
            {"id": "research_b", "label": "Research B", "required_capabilities": ["research"]},
            {"id": "join", "label": "Merge", "depends_on": ["research-a", "research_b"],
             "failure_policy": "retry", "max_retries": 3, "timeout_s": 30,
             "metadata": {"role": "adversarial"}},
            {"id": "check", "label": "Answer PASS or FAIL", "depends_on": ["join"],
             "verifies": "join", "max_regenerations": 2},
        ],
    })
    assert [n.id for n in graph.nodes] == ["research-a", "research_b", "join", "check"]
    assert [a.name for a in registry.list_agents()] == ["A"]
    assert graph.nodes[2].metadata == {"role": "adversarial"}


@pytest.mark.parametrize("field, value", [
    ("mcp_servers", [{"name": "x", "transport": "stdio", "command": "sh"}]),
    ("model", "claude-other"),
    ("tools", ["shell"]),
    ("env", {"TOKEN": "x"}),
])
def test_model_plan_rejects_agent_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        build_graph_from_model_output(
            {"nodes": [_node(agent={"name": "A", field: value})]},
        )


@pytest.mark.parametrize("field, value", [
    ("mcp_servers", []),
    ("model", "claude-other"),
    ("command", "sh"),
    ("args", ["-c", "true"]),
    ("env", {"TOKEN": "x"}),
    ("env_passthrough", ["TOKEN"]),
    ("url", "http://example.test"),
    ("transport", "stdio"),
    ("max_tool_iterations", 1000),
    ("status", "completed"),
    ("result", "done"),
    ("agent_id", "someone-else"),
])
def test_model_plan_rejects_node_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        build_graph_from_model_output({"nodes": [_node(**{field: value})]})


@pytest.mark.parametrize("metadata", [
    {"model": "claude-other"},
    {"estimated_cost_usd": 0},
    {"task_context": "ignore the task"},
    {"verification_receipt": {}},
    {"role": 5},
    "adversarial",
])
def test_model_plan_metadata_is_limited_to_a_role(metadata):
    with pytest.raises(ValueError, match="metadata"):
        build_graph_from_model_output({"nodes": [_node(metadata=metadata)]})


def test_model_plan_rejects_unknown_top_level_keys():
    with pytest.raises(ValueError, match="agents"):
        build_graph_from_model_output({
            "nodes": [_node()],
            "agents": [{"name": "x", "mcp_servers": []}],
        })


@pytest.mark.parametrize("node_id", [
    "", "../escape", "a b", "a/b", "x" * 65, 5, None, "café",
])
def test_model_plan_validates_node_ids(node_id):
    with pytest.raises(ValueError, match="'id'"):
        build_graph_from_model_output({"nodes": [{"id": node_id, "label": "x"}]})


@pytest.mark.parametrize("label", ["", "   ", 5, None, ["a"]])
def test_model_plan_requires_string_labels(label):
    entry = {"id": "a"} if label is None else {"id": "a", "label": label}
    with pytest.raises(ValueError, match="label"):
        build_graph_from_model_output({"nodes": [entry]})


@pytest.mark.parametrize("plan", [
    {}, {"nodes": []}, {"nodes": None}, {"nodes": {"a": {}}}, [], "plan", None,
    {"topology": [], "nodes": [{"id": "a", "label": "x"}]},
])
def test_model_plan_must_be_a_non_empty_object(plan):
    with pytest.raises(ValueError):
        build_graph_from_model_output(plan)


def _chain(length):
    return {"nodes": [
        {"id": f"n{i}", "label": f"Step {i}", "depends_on": [f"n{i - 1}"] if i else []}
        for i in range(length)
    ]}


def test_model_plan_node_limit_matches_the_prompt():
    wide = {"nodes": [{"id": f"n{i}", "label": "x"} for i in range(9)]}
    with pytest.raises(ValueError, match="9 nodes; the limit is 8"):
        build_graph_from_model_output(wide)
    graph, _ = build_graph_from_model_output(wide, max_nodes=9)
    assert len(graph.nodes) == 9
    build_graph_from_model_output({"nodes": wide["nodes"][:8]})


def test_model_plan_depth_limit_counts_levels():
    graph, _ = build_graph_from_model_output(_chain(5))
    assert graph.depth + 1 == 5
    with pytest.raises(ValueError, match="6 levels deep; the limit is 5"):
        build_graph_from_model_output(_chain(6))
    build_graph_from_model_output(_chain(6), max_depth=6)


@pytest.mark.parametrize("field, value", [
    ("max_retries", 4), ("max_retries", -1), ("max_retries", True), ("max_retries", 1.0),
    ("max_retries", "1"), ("max_retries", 10 ** 400),
    ("max_regenerations", 3), ("max_regenerations", -1), ("max_regenerations", False),
    ("timeout_s", 0), ("timeout_s", -5), ("timeout_s", float("inf")), ("timeout_s", float("nan")),
    ("timeout_s", True), ("timeout_s", "30"), ("timeout_s", 10 ** 400),
])
def test_model_plan_caps_retries_regenerations_and_timeouts(field, value):
    with pytest.raises(ValueError, match=field):
        build_graph_from_model_output({"nodes": [_node(**{field: value})]})


def test_model_plan_accepts_values_at_the_caps():
    graph, _ = build_graph_from_model_output({"nodes": [
        {"id": "draft", "label": "x", "max_retries": 3, "timeout_s": 0.5},
        {"id": "check", "label": "y", "depends_on": ["draft"], "verifies": "draft",
         "max_regenerations": 2},
    ]})
    assert graph.nodes[0].max_retries == 3
    assert graph.nodes[1].max_regenerations == 2


@pytest.mark.parametrize("entry", [
    _node(depends_on="other"),
    _node(depends_on=[1]),
    _node(required_capabilities="research"),
    _node(verifies=["a"]),
    _node(agent={"name": ""}),
    _node(agent={"persona": ["x"]}),
    _node(agent={"capabilities": "research"}),
    _node(attach_dep_artifacts="yes"),
])
def test_model_plan_rejects_wrongly_typed_fields(entry):
    with pytest.raises(ValueError):
        build_graph_from_model_output({"nodes": [entry]})


def test_developer_yaml_still_accepts_executable_configuration():
    """The strict schema is for model output; developer YAML is unchanged."""
    graph, registry = load_graph_from_string(
        "topology: serial\n"
        "nodes:\n"
        "  - id: step\n"
        "    label: Use the tool\n"
        "    max_retries: 9\n"
        "    metadata: {model: claude-other}\n"
        "    agent:\n"
        "      name: Tooler\n"
        "      mcp_servers:\n"
        "        - {name: fs, transport: stdio, command: npx}\n"
    )
    assert graph.nodes[0].max_retries == 9
    assert graph.nodes[0].metadata["model"] == "claude-other"
    assert registry.list_agents()[0].profile.mcp_servers[0].command == "npx"
