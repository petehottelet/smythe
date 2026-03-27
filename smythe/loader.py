"""YAML loader — build ExecutionGraphs from declarative DAG files."""

from __future__ import annotations

from pathlib import Path

import yaml

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology
from smythe.registry import Registry


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
            failure_policy = FailurePolicy(fp_raw.lower())
        except ValueError:
            valid = [fp.value for fp in FailurePolicy]
            raise ValueError(
                f"Unknown failure_policy {fp_raw!r} on node {node_id!r}. "
                f"Valid values: {valid}"
            ) from None

        max_retries = entry.get("max_retries", 1)

        node = Node(
            id=node_id,
            label=label,
            depends_on=depends_on,
            metadata=metadata,
            failure_policy=failure_policy,
            max_retries=max_retries,
            required_capabilities=required_capabilities,
        )

        agent_data = entry.get("agent")
        if agent_data:
            profile = AgentProfile(
                name=agent_data.get("name", node_id),
                persona=agent_data.get("persona", ""),
                capabilities=agent_data.get("capabilities", []),
            )
            agent = Agent(profile=profile)
            registry.register(agent)
            node.agent_id = agent.id

        nodes.append(node)

    graph = ExecutionGraph(topology=topology, nodes=nodes)
    graph.validate()
    return graph, registry


def _parse_topology(raw: str | list[str]) -> list[Topology]:
    """Convert a topology value (string or list of strings) to Topology enums."""
    if isinstance(raw, str):
        raw = [raw]

    result: list[Topology] = []
    for item in raw:
        normalized = item.strip().lower()
        try:
            result.append(Topology(normalized))
        except ValueError:
            valid = [t.value for t in Topology]
            raise ValueError(
                f"Unknown topology {item!r}. Valid values: {valid}"
            ) from None
    return result
