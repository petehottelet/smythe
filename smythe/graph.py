"""Execution graph — the DAG that the planner generates and the executor walks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


def _escape_mermaid(text: str) -> str:
    """Escape characters that break Mermaid node labels."""
    return text.replace('"', "#quot;").replace("[", "(").replace("]", ")")


# House diagram style (docs/style.md): serif type, ivory nodes, hairline
# warm-gray edges, gentle curves. Opt in via to_mermaid(theme=True).
MERMAID_THEME = (
    '%%{init: {"theme":"base","themeVariables":{'
    '"fontFamily":"Georgia, \'Times New Roman\', serif","fontSize":"14px",'
    '"primaryColor":"#faf8f1","primaryTextColor":"#23221e",'
    '"primaryBorderColor":"#a89f8c","lineColor":"#a89f8c"},'
    '"flowchart":{"curve":"basis","nodeSpacing":48,"rankSpacing":58}}}%%'
)


class Topology(Enum):
    """High-level execution patterns the planner can select."""

    SERIAL = "serial"
    FORK_JOIN = "fork_join"
    BROADCAST_REDUCE = "broadcast_reduce"
    ADVERSARIAL = "adversarial"


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailurePolicy(Enum):
    """How to handle a node failure."""
    HALT = "halt"
    SKIP = "skip"
    RETRY = "retry"


@dataclass
class Node:
    """A single step in the execution graph.

    Attributes:
        id: Unique node identifier.
        label: Human-readable description of this step.
        agent_id: ID of the agent assigned to execute this node (set by planner/registry).
        depends_on: IDs of nodes that must complete before this one starts.
        result: Output produced after execution.
        status: Current lifecycle state.
        metadata: Arbitrary planner/executor annotations.
        failure_policy: How to handle failure (HALT, SKIP, or RETRY).
        max_retries: Number of retry attempts when failure_policy is RETRY.
        required_capabilities: Capability tags for agent assignment matching.
        timeout_s: Wall-clock limit for a single execution attempt, in
            seconds.  None (default) means no timeout.  A timed-out
            attempt fails and is handled by the node's failure policy.
        max_tool_iterations: Cap on tool-loop iterations for this node.
            None (default) uses the executor-level default.
    """

    label: str
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    agent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    status: NodeStatus = NodeStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_policy: FailurePolicy = FailurePolicy.HALT
    max_retries: int = 1
    required_capabilities: list[str] = field(default_factory=list)
    timeout_s: float | None = None
    max_tool_iterations: int | None = None


@dataclass
class ExecutionGraph:
    """A DAG of nodes representing the planned execution for a task.

    Attributes:
        topology: The high-level pattern this graph follows.
        nodes: Ordered list of execution nodes.
        estimated_cost_usd: Pre-execution cost estimate set by the planner.
    """

    topology: list[Topology]
    nodes: list[Node] = field(default_factory=list)
    estimated_cost_usd: float | None = None

    def roots(self) -> list[Node]:
        """Nodes with no dependencies — entry points for execution."""
        return [n for n in self.nodes if not n.depends_on]

    def dependents(self, node_id: str) -> list[Node]:
        """Nodes that directly depend on the given node."""
        return [n for n in self.nodes if node_id in n.depends_on]

    def is_ready(self, node: Node) -> bool:
        """True if all of a node's dependencies have completed or been skipped."""
        resolved = {
            n.id for n in self.nodes
            if n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
        }
        return all(dep in resolved for dep in node.depends_on)

    @property
    def depth(self) -> int:
        """Longest path in the DAG (number of edges on the critical path)."""
        if not self.nodes:
            return 0
        lookup = {n.id: n for n in self.nodes}
        cache: dict[str, int] = {}

        def _depth(node_id: str) -> int:
            if node_id in cache:
                return cache[node_id]
            node = lookup.get(node_id)
            if node is None or not node.depends_on:
                cache[node_id] = 0
                return 0
            d = 1 + max(_depth(dep) for dep in node.depends_on)
            cache[node_id] = d
            return d

        return max(_depth(n.id) for n in self.nodes)

    @property
    def agent_count(self) -> int:
        """Number of unique agents assigned to nodes."""
        return len({n.agent_id for n in self.nodes if n.agent_id is not None})

    def validate(self) -> None:
        """Check basic DAG invariants."""
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            from collections import Counter
            counts = Counter(n.id for n in self.nodes)
            dupes = [nid for nid, cnt in counts.items() if cnt > 1]
            raise ValueError(f"Duplicate node IDs: {dupes}")
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in ids:
                    raise ValueError(f"Node {node.id!r} depends on unknown node {dep!r}")
        if self._has_cycle():
            raise ValueError("Execution graph contains a cycle")

    def to_json(self) -> dict[str, Any]:
        """Serializable snapshot of the graph: topology, nodes, statuses, costs."""
        return {
            "topology": [t.value for t in self.topology],
            "estimated_cost_usd": self.estimated_cost_usd,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "agent_id": n.agent_id,
                    "depends_on": list(n.depends_on),
                    "status": n.status.value,
                    "cost_usd": n.metadata.get("cost_usd"),
                }
                for n in self.nodes
            ],
        }

    def to_mermaid(self, *, theme: bool = False) -> str:
        """Render the DAG as a Mermaid flowchart (top-down).

        Node statuses map to style classes so executed graphs are
        readable at a glance; output is deterministic for snapshot tests.
        With ``theme=True``, the house style header (serif type, ivory
        nodes, hairline edges — see docs/style.md) is prepended so
        generated diagrams match the hand-authored ones in the README.
        """
        lines = [MERMAID_THEME] if theme else []
        lines.append("flowchart TD")
        for n in self.nodes:
            label = _escape_mermaid(self._node_label(n))
            lines.append(f'    {n.id}["{label}"]')
        for n in self.nodes:
            for dep in n.depends_on:
                lines.append(f"    {dep} --> {n.id}")
        status_class = {
            NodeStatus.COMPLETED: "done",
            NodeStatus.FAILED: "failed",
            NodeStatus.SKIPPED: "skipped",
            NodeStatus.RUNNING: "running",
        }
        styled: dict[str, list[str]] = {}
        for n in self.nodes:
            cls = status_class.get(n.status)
            if cls:
                styled.setdefault(cls, []).append(n.id)
        if styled:
            lines.append("    classDef done fill:#eef0e4,stroke:#5a7742,color:#3a4d2b")
            lines.append("    classDef failed fill:#f3e0dd,stroke:#8c3b2e,color:#66291f")
            lines.append("    classDef skipped fill:#eceae3,stroke:#8a8578,color:#57534a")
            lines.append("    classDef running fill:#f5ead0,stroke:#9a7b2d,color:#5c4a1e")
            for cls in ("done", "failed", "skipped", "running"):
                if cls in styled:
                    lines.append(f"    class {','.join(styled[cls])} {cls}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render the DAG in Graphviz DOT format."""
        lines = ["digraph smythe {", "    rankdir=TB;"]
        for n in self.nodes:
            label = self._node_label(n).replace('"', '\\"')
            lines.append(f'    "{n.id}" [label="{label}"];')
        for n in self.nodes:
            for dep in n.depends_on:
                lines.append(f'    "{dep}" -> "{n.id}";')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExecutionGraph(topology={self.topology!r}, nodes={len(self.nodes)})"

    def __str__(self) -> str:
        topo_order = self._topo_sort()
        root_ids = {n.id for n in self.roots()}

        sections: list[tuple[str, list[Node]]] = []
        parallel_roots = [n for n in topo_order if n.id in root_ids]

        if len(parallel_roots) > 1:
            sections.append(("fork (parallel)", parallel_roots))
            remaining = [n for n in topo_order if n.id not in root_ids]

            join_nodes = [
                n for n in remaining
                if root_ids.issubset(set(n.depends_on))
            ]
            non_join = [n for n in remaining if n not in join_nodes]

            for jn in join_nodes:
                sections.append(("join", [jn]))

            adversarial = [
                n for n in non_join
                if n.metadata.get("role") == "adversarial"
            ]
            serial = [n for n in non_join if n not in adversarial]

            if adversarial:
                sections.append(("adversarial", adversarial))
            for sn in serial:
                dep_labels = self._dep_label(sn)
                sections.append((f"serial{dep_labels}", [sn]))
        else:
            for node in topo_order:
                dep_labels = self._dep_label(node)
                tag = "serial" if dep_labels else ""
                sections.append((tag + dep_labels, [node]))

        lines = [f'TaskGraph(topology="{self._topology_label()}")']
        for si, (section_label, section_nodes) in enumerate(sections):
            is_last_section = si == len(sections) - 1
            branch = "└─" if is_last_section else "├─"
            cont = "    " if is_last_section else "│   "

            if len(section_nodes) == 1 and not section_label.startswith("fork"):
                node = section_nodes[0]
                prefix = f"{section_label}: " if section_label else ""
                lines.append(f"{branch} {prefix}{self._node_label(node)}")
            else:
                lines.append(f"{branch} {section_label}:")
                for ni, node in enumerate(section_nodes):
                    is_last_node = ni == len(section_nodes) - 1
                    node_branch = "└─" if is_last_node else "├─"
                    lines.append(f"{cont}{node_branch} {self._node_label(node)}")

        if self.estimated_cost_usd is not None:
            lines.append(
                f"#\n# Estimated cost: ${self.estimated_cost_usd:.2f} | "
                f"Depth: {self.depth} | Agents: {self.agent_count}"
            )

        return "\n".join(lines)

    def _topology_label(self) -> str:
        """Human-readable topology, joining phases with arrows."""
        return " \u2192 ".join(t.value.replace("_", "-") for t in self.topology)

    @staticmethod
    def _node_label(node: Node) -> str:
        name = node.metadata.get("agent_name") or node.agent_id or node.id
        return f"{name}: {node.label}"

    def _dep_label(self, node: Node) -> str:
        if not node.depends_on:
            return ""
        dep_names: list[str] = []
        lookup = {n.id: n for n in self.nodes}
        for dep_id in node.depends_on:
            dep_node = lookup.get(dep_id)
            if dep_node:
                dep_names.append(
                    dep_node.metadata.get("agent_name") or dep_node.agent_id or dep_id
                )
            else:
                dep_names.append(dep_id)
        return " (depends on " + ", ".join(dep_names) + ")"

    def _topo_sort(self) -> list[Node]:
        visited: set[str] = set()
        order: list[Node] = []
        lookup = {n.id: n for n in self.nodes}

        def visit(node: Node) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for dep_id in node.depends_on:
                if dep_id in lookup:
                    visit(lookup[dep_id])
            order.append(node)

        for node in self.nodes:
            visit(node)
        return order

    def _has_cycle(self) -> bool:
        visited: set[str] = set()
        in_stack: set[str] = set()
        adj: dict[str, list[str]] = {n.id: n.depends_on for n in self.nodes}

        def dfs(nid: str) -> bool:
            visited.add(nid)
            in_stack.add(nid)
            for dep in adj.get(nid, []):
                if dep in in_stack:
                    return True
                if dep not in visited and dfs(dep):
                    return True
            in_stack.discard(nid)
            return False

        return any(nid not in visited and dfs(nid) for nid in adj)
