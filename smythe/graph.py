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
    """

    label: str
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    agent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    status: NodeStatus = NodeStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionGraph:
    """A DAG of nodes representing the planned execution for a task.

    Attributes:
        topology: The high-level pattern this graph follows.
        nodes: Ordered list of execution nodes.
    """

    topology: Topology
    nodes: list[Node] = field(default_factory=list)

    def roots(self) -> list[Node]:
        """Nodes with no dependencies — entry points for execution."""
        return [n for n in self.nodes if not n.depends_on]

    def dependents(self, node_id: str) -> list[Node]:
        """Nodes that directly depend on the given node."""
        return [n for n in self.nodes if node_id in n.depends_on]

    def is_ready(self, node: Node) -> bool:
        """True if all of a node's dependencies have completed."""
        completed = {n.id for n in self.nodes if n.status == NodeStatus.COMPLETED}
        return all(dep in completed for dep in node.depends_on)

    def validate(self) -> None:
        """Check basic DAG invariants."""
        ids = {n.id for n in self.nodes}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in ids:
                    raise ValueError(f"Node {node.id!r} depends on unknown node {dep!r}")
        if self._has_cycle():
            raise ValueError("Execution graph contains a cycle")

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
