"""Execution graph — the DAG that the planner generates and the executor walks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


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
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in ids:
                    raise ValueError(f"Node {node.id!r} depends on unknown node {dep!r}")
        if self._has_cycle():
            raise ValueError("Execution graph contains a cycle")

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
        name = node.agent_id if node.agent_id else node.id
        return f"{name}: {node.label}"

    def _dep_label(self, node: Node) -> str:
        if not node.depends_on:
            return ""
        dep_names: list[str] = []
        lookup = {n.id: n for n in self.nodes}
        for dep_id in node.depends_on:
            dep_node = lookup.get(dep_id)
            dep_names.append(dep_node.agent_id or dep_id if dep_node else dep_id)
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
