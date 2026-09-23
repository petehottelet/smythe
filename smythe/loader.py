"""YAML loader — build ExecutionGraphs from declarative DAG files.

Developer-written YAML goes through :func:`build_graph_from_dict`, which
accepts every node and agent field, including MCP server declarations.
Model-generated plans go through :func:`build_graph_from_model_output`,
which accepts only the planning schema: a model must not be able to
declare commands, servers, environment variables, or model overrides.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology
from smythe.registry import Registry

# Limits for model-generated plans.  The node and depth defaults match
# what PLANNING_SYSTEM_PROMPT promises ("8 nodes is the ceiling",
# "depth <= 5 levels").  Retries and regenerations multiply spend, so a
# generated plan may not raise them past small fixed ceilings: three
# retries ride out transient provider errors, and two regenerations
# leave headroom over the prompt's recommended one while bounding a
# single gate to three runs of the subtree it judges.
MODEL_PLAN_MAX_NODES = 8
MODEL_PLAN_MAX_DEPTH = 5
MODEL_PLAN_MAX_RETRIES = 3
MODEL_PLAN_MAX_REGENERATIONS = 2

_MODEL_NODE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MODEL_PLAN_KEYS = frozenset({"topology", "nodes"})
_MODEL_NODE_KEYS = frozenset({
    "id", "label", "depends_on", "agent", "required_capabilities",
    "failure_policy", "max_retries", "timeout_s", "verifies",
    "max_regenerations", "attach_dep_artifacts", "metadata",
})
_MODEL_AGENT_KEYS = frozenset({"name", "persona", "capabilities"})
# Node metadata also carries executor control state (model, cost
# estimates, verification receipts), so a plan may set only the
# display-only adversarial role.
_MODEL_METADATA_KEYS = frozenset({"role"})


def load_graph(path: str | Path) -> tuple[ExecutionGraph, Registry]:
    """Load an ExecutionGraph and accompanying Registry from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    return load_graph_from_string(text)


def load_graph_from_string(yaml_str: str) -> tuple[ExecutionGraph, Registry]:
    """Parse a YAML string into an ExecutionGraph and Registry."""
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return build_graph_from_dict(data)


def build_graph_from_dict(data: dict) -> tuple[ExecutionGraph, Registry]:
    """Build an ExecutionGraph and Registry from a parsed dict.

    Shared by the YAML loader and the LLM planner.  The dict must have
    a ``topology`` key (string or list of strings) and a ``nodes`` list.
    Each node entry may include an ``agent`` sub-dict with name, persona,
    and capabilities.
    """
    topology = _parse_topology(data.get("topology", ["serial"]))

    registry = Registry()
    nodes: list[Node] = []

    nodes_raw = data.get("nodes", [])
    if not isinstance(nodes_raw, list):
        raise ValueError(
            f"'nodes' must be a list, got {type(nodes_raw).__name__}"
        )

    for i, entry in enumerate(nodes_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Node at index {i} must be a mapping, got {type(entry).__name__}"
            )
        node_id = entry.get("id")
        if not node_id:
            raise ValueError("Every node must have an 'id' field")

        label = entry.get("label", node_id)
        depends_on = entry.get("depends_on", [])
        metadata = entry.get("metadata", {})
        required_capabilities = entry.get("required_capabilities", [])

        if not isinstance(depends_on, list):
            raise ValueError(f"'depends_on' on node {node_id!r} must be a list")
        if not isinstance(metadata, dict):
            raise ValueError(f"'metadata' on node {node_id!r} must be a mapping")
        if not isinstance(required_capabilities, list):
            raise ValueError(f"'required_capabilities' on node {node_id!r} must be a list")

        fp_raw = entry.get("failure_policy", "halt")
        try:
            if not isinstance(fp_raw, str):
                raise ValueError
            failure_policy = FailurePolicy(fp_raw.lower())
        except ValueError:
            valid = [fp.value for fp in FailurePolicy]
            raise ValueError(
                f"Unknown failure_policy {fp_raw!r} on node {node_id!r}. "
                f"Valid values: {valid}"
            ) from None

        max_retries = entry.get("max_retries", 1)

        timeout_s = entry.get("timeout_s")
        if timeout_s is not None:
            if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
                raise ValueError(
                    f"'timeout_s' on node {node_id!r} must be a number, "
                    f"got {type(timeout_s).__name__}"
                )
            if timeout_s <= 0:
                raise ValueError(
                    f"'timeout_s' on node {node_id!r} must be positive, got {timeout_s}"
                )
            timeout_s = float(timeout_s)

        max_tool_iterations = entry.get("max_tool_iterations")
        if max_tool_iterations is not None:
            if isinstance(max_tool_iterations, bool) or not isinstance(max_tool_iterations, int):
                raise ValueError(
                    f"'max_tool_iterations' on node {node_id!r} must be an integer, "
                    f"got {type(max_tool_iterations).__name__}"
                )
            if max_tool_iterations < 1:
                raise ValueError(
                    f"'max_tool_iterations' on node {node_id!r} must be >= 1, "
                    f"got {max_tool_iterations}"
                )

        attach_dep_artifacts = entry.get("attach_dep_artifacts", False)
        if not isinstance(attach_dep_artifacts, bool):
            raise ValueError(
                f"'attach_dep_artifacts' on node {node_id!r} must be a boolean, "
                f"got {type(attach_dep_artifacts).__name__}"
            )

        verifies = entry.get("verifies")
        if verifies is not None and not isinstance(verifies, str):
            raise ValueError(
                f"'verifies' on node {node_id!r} must be a node id string, "
                f"got {type(verifies).__name__}"
            )

        max_regenerations = entry.get("max_regenerations", 0)
        if isinstance(max_regenerations, bool) or not isinstance(max_regenerations, int):
            raise ValueError(
                f"'max_regenerations' on node {node_id!r} must be an integer, "
                f"got {type(max_regenerations).__name__}"
            )
        if max_regenerations < 0:
            raise ValueError(
                f"'max_regenerations' on node {node_id!r} must be >= 0, "
                f"got {max_regenerations}"
            )

        node = Node(
            id=node_id,
            label=label,
            depends_on=depends_on,
            metadata=metadata,
            failure_policy=failure_policy,
            max_retries=max_retries,
            required_capabilities=required_capabilities,
            timeout_s=timeout_s,
            max_tool_iterations=max_tool_iterations,
            attach_dep_artifacts=attach_dep_artifacts,
            verifies=verifies,
            max_regenerations=max_regenerations,
        )

        agent_data = entry.get("agent")
        if agent_data:
            if not isinstance(agent_data, dict):
                raise ValueError(
                    f"'agent' on node {node_id!r} must be a mapping, "
                    f"got {type(agent_data).__name__}"
                )
            mcp_raw = agent_data.get("mcp_servers", [])
            if not isinstance(mcp_raw, list):
                raise ValueError(
                    f"'mcp_servers' on node {node_id!r} agent must be a list"
                )
            mcp_servers = []
            for server_entry in mcp_raw:
                if not isinstance(server_entry, dict):
                    raise ValueError(
                        f"Each mcp_servers entry on node {node_id!r} must be a mapping"
                    )
                from smythe.mcp import MCPConfigError, MCPServerSpec
                try:
                    mcp_servers.append(MCPServerSpec.from_dict(server_entry))
                except (MCPConfigError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid mcp_servers entry on node {node_id!r}: {exc}"
                    ) from exc

            profile = AgentProfile(
                name=agent_data.get("name", node_id),
                persona=agent_data.get("persona", ""),
                capabilities=agent_data.get("capabilities", []),
                mcp_servers=mcp_servers,
            )
            agent = Agent(profile=profile)
            registry.register(agent)
            node.agent_id = agent.id
            node.metadata["agent_name"] = agent.name

        nodes.append(node)

    known_ids = {node.id for node in nodes}
    for node in nodes:
        if node.verifies is not None and node.verifies not in known_ids:
            raise ValueError(
                f"Node {node.id!r} 'verifies' names unknown node "
                f"{node.verifies!r}"
            )
        if node.verifies == node.id:
            raise ValueError(f"Node {node.id!r} cannot 'verifies' itself")

    graph = ExecutionGraph(topology=topology, nodes=nodes)
    graph.validate()
    return graph, registry


def build_graph_from_model_output(
    data: object,
    *,
    max_nodes: int = MODEL_PLAN_MAX_NODES,
    max_depth: int = MODEL_PLAN_MAX_DEPTH,
) -> tuple[ExecutionGraph, Registry]:
    """Build a graph from a model-generated plan under a strict schema.

    Unlike :func:`build_graph_from_dict`, which serves developer-written
    YAML, this accepts only the fields the planning prompt defines.  A
    plan cannot declare MCP servers, commands, URLs, environment
    variables, or per-node or per-agent models, and unknown keys are
    rejected rather than ignored.  The schema is checked before any
    agent is created, and nothing is returned unless every check passes.

    ``max_depth`` counts levels: the number of nodes on the longest
    dependency chain, so a single node has depth 1.

    Raises:
        ValueError: The plan breaks the schema or a limit.  The message
            is written to be fed back to the model on a retry.
    """
    if not isinstance(data, dict):
        raise ValueError("The plan must be a JSON object")
    _reject_unknown_keys(data, _MODEL_PLAN_KEYS, "The plan")
    topology = data.get("topology", ["serial"])
    if not (
        isinstance(topology, str)
        or (isinstance(topology, list) and topology
            and all(isinstance(item, str) for item in topology))
    ):
        raise ValueError("'topology' must be a string or a non-empty list of strings")

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("'nodes' must be a non-empty list")
    if len(nodes) > max_nodes:
        raise ValueError(f"The plan has {len(nodes)} nodes; the limit is {max_nodes}")
    for index, entry in enumerate(nodes):
        _check_model_node(index, entry)

    graph, registry = build_graph_from_dict(data)
    levels = graph.depth + 1
    if levels > max_depth:
        raise ValueError(
            f"The plan is {levels} levels deep; the limit is {max_depth}"
        )
    return graph, registry


def _reject_unknown_keys(value: dict, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(
            f"{where} has unsupported field(s) {unknown}; "
            f"allowed fields are {sorted(allowed)}"
        )


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _check_bounded_int(value: object, name: str, node_id: str, ceiling: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= ceiling:
        raise ValueError(
            f"'{name}' on node {node_id!r} must be an integer from 0 to {ceiling}, "
            f"got {value!r}"
        )


def _check_model_node(index: int, entry: object) -> None:
    """Validate one generated node entry; raise ValueError when it is invalid."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"Node at index {index} must be an object, got {type(entry).__name__}"
        )
    node_id = entry.get("id")
    if not isinstance(node_id, str) or not _MODEL_NODE_ID.fullmatch(node_id):
        raise ValueError(
            f"Node at index {index} needs an 'id' of 1-64 letters, digits, "
            f"'-' or '_', got {node_id!r}"
        )
    _reject_unknown_keys(entry, _MODEL_NODE_KEYS, f"Node {node_id!r}")

    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Node {node_id!r} needs a non-empty string 'label'")
    for name in ("depends_on", "required_capabilities"):
        if name in entry and not _is_str_list(entry[name]):
            raise ValueError(f"'{name}' on node {node_id!r} must be a list of strings")
    for name in ("failure_policy", "verifies"):
        if name in entry and not isinstance(entry[name], str):
            raise ValueError(f"'{name}' on node {node_id!r} must be a string")
    if "max_retries" in entry:
        _check_bounded_int(entry["max_retries"], "max_retries", node_id, MODEL_PLAN_MAX_RETRIES)
    if "max_regenerations" in entry:
        _check_bounded_int(
            entry["max_regenerations"], "max_regenerations", node_id,
            MODEL_PLAN_MAX_REGENERATIONS,
        )
    if "timeout_s" in entry:
        timeout_s = entry["timeout_s"]
        try:
            finite = (not isinstance(timeout_s, bool) and isinstance(timeout_s, (int, float))
                      and math.isfinite(timeout_s))
        except OverflowError:  # an integer too large to become a float
            finite = False
        if not finite or timeout_s <= 0:
            raise ValueError(
                f"'timeout_s' on node {node_id!r} must be a finite positive number, "
                f"got {timeout_s!r}"
            )

    metadata = entry.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError(f"'metadata' on node {node_id!r} must be an object")
        _reject_unknown_keys(metadata, _MODEL_METADATA_KEYS, f"'metadata' on node {node_id!r}")
        if not all(isinstance(value, str) for value in metadata.values()):
            raise ValueError(f"'metadata' values on node {node_id!r} must be strings")

    agent = entry.get("agent")
    if agent is not None:
        if not isinstance(agent, dict):
            raise ValueError(
                f"'agent' on node {node_id!r} must be an object with "
                f"{sorted(_MODEL_AGENT_KEYS)}, got {type(agent).__name__}"
            )
        _reject_unknown_keys(agent, _MODEL_AGENT_KEYS, f"The agent on node {node_id!r}")
        if "name" in agent and (not isinstance(agent["name"], str) or not agent["name"].strip()):
            raise ValueError(f"The agent 'name' on node {node_id!r} must be a non-empty string")
        if "persona" in agent and not isinstance(agent["persona"], str):
            raise ValueError(f"The agent 'persona' on node {node_id!r} must be a string")
        if "capabilities" in agent and not _is_str_list(agent["capabilities"]):
            raise ValueError(
                f"The agent 'capabilities' on node {node_id!r} must be a list of strings"
            )


def _parse_topology(raw: str | list[str]) -> list[Topology]:
    """Convert a topology value (string or list of strings) to Topology enums."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(
            f"'topology' must be a string or a list of strings, got {type(raw).__name__}"
        )

    result: list[Topology] = []
    for item in raw:
        try:
            if not isinstance(item, str):
                raise ValueError
            result.append(Topology(item.strip().lower()))
        except ValueError:
            valid = [t.value for t in Topology]
            raise ValueError(
                f"Unknown topology {item!r}. Valid values: {valid}"
            ) from None
    return result
