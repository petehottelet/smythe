"""Checkpointing — durable, resumable execution state.

After each node reaches a terminal status, the Swarm persists the full
execution state (graph, node results, agents, budget consumed) through a
CheckpointStore.  A crashed or interrupted execution can then be picked
up with ``swarm.resume(execution_id)``, re-running only the nodes that
never completed.

The state is a plain JSON document (version 1) so users can inspect or
repair checkpoints by hand.  See docs/checkpoint-format.md for the full
schema.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.registry import Registry
from smythe.task import Task

CHECKPOINT_VERSION = 1

_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _jsonable(value: Any) -> Any:
    """Return *value* if JSON-serializable, otherwise its str() form."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "agent_id": node.agent_id,
        "depends_on": list(node.depends_on),
        "result": _jsonable(node.result),
        "status": node.status.value,
        "metadata": {k: _jsonable(v) for k, v in node.metadata.items()},
        "failure_policy": node.failure_policy.value,
        "max_retries": node.max_retries,
        "required_capabilities": list(node.required_capabilities),
        "timeout_s": node.timeout_s,
    }


def node_from_dict(data: dict[str, Any]) -> Node:
    return Node(
        id=data["id"],
        label=data["label"],
        agent_id=data.get("agent_id"),
        depends_on=list(data.get("depends_on", [])),
        result=data.get("result"),
        status=NodeStatus(data.get("status", "pending")),
        metadata=dict(data.get("metadata", {})),
        failure_policy=FailurePolicy(data.get("failure_policy", "halt")),
        max_retries=data.get("max_retries", 1),
        required_capabilities=list(data.get("required_capabilities", [])),
        timeout_s=data.get("timeout_s"),
    )


def graph_to_dict(graph: ExecutionGraph) -> dict[str, Any]:
    return {
        "topology": [t.value for t in graph.topology],
        "estimated_cost_usd": graph.estimated_cost_usd,
        "nodes": [node_to_dict(n) for n in graph.nodes],
    }


def graph_from_dict(data: dict[str, Any]) -> ExecutionGraph:
    """Exact restore of a serialized graph, including statuses and results."""
    return ExecutionGraph(
        topology=[Topology(t) for t in data.get("topology", ["serial"])],
        nodes=[node_from_dict(n) for n in data.get("nodes", [])],
        estimated_cost_usd=data.get("estimated_cost_usd"),
    )


def agents_to_list(registry: Registry) -> list[dict[str, Any]]:
    return [
        {
            "id": agent.id,
            "name": agent.profile.name,
            "persona": agent.profile.persona,
            "capabilities": list(agent.profile.capabilities),
        }
        for agent in registry.list_agents()
    ]


def agents_from_list(data: list[dict[str, Any]]) -> list[Agent]:
    return [
        Agent(
            id=entry["id"],
            profile=AgentProfile(
                name=entry.get("name", entry["id"]),
                persona=entry.get("persona", ""),
                capabilities=list(entry.get("capabilities", [])),
            ),
        )
        for entry in data
    ]


def task_to_dict(task: Task | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "goal": task.goal,
        "constraints": list(task.constraints),
        "context": {k: _jsonable(v) for k, v in task.context.items()},
    }


def task_from_dict(data: dict[str, Any] | None) -> Task | None:
    if data is None:
        return None
    return Task(
        goal=data["goal"],
        constraints=list(data.get("constraints", [])),
        context=dict(data.get("context", {})),
    )


def reset_incomplete_nodes(graph: ExecutionGraph) -> list[str]:
    """Reset RUNNING and FAILED nodes to PENDING so resume re-runs them.

    COMPLETED and SKIPPED nodes keep their status and results.
    Returns the IDs of the nodes that were reset.
    """
    reset: list[str] = []
    for node in graph.nodes:
        if node.status in (NodeStatus.RUNNING, NodeStatus.FAILED):
            node.status = NodeStatus.PENDING
            node.result = None
            reset.append(node.id)
    return reset


class CheckpointStore(ABC):
    """Persistence interface for execution checkpoints.

    Implementations must make ``save()`` atomic per execution_id: a
    reader must never observe a partially written state.
    """

    @abstractmethod
    def save(self, execution_id: str, state: dict[str, Any]) -> None:
        """Persist the full state for an execution, replacing any prior state."""

    @abstractmethod
    def load(self, execution_id: str) -> dict[str, Any] | None:
        """Return the last saved state, or None if the id is unknown."""

    @abstractmethod
    def delete(self, execution_id: str) -> None:
        """Remove a checkpoint.  Deleting an unknown id is a no-op."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return all known execution ids, sorted."""


class FileCheckpointStore(CheckpointStore):
    """Filesystem-backed store: one JSON file per execution.

    Files live in ``~/.smythe/checkpoints/`` by default.  Writes go to a
    temp file and are moved into place with os.replace, so a crash
    mid-write never corrupts the previous checkpoint.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = (
            Path(directory)
            if directory is not None
            else Path.home() / ".smythe" / "checkpoints"
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, execution_id: str) -> Path:
        if not _EXECUTION_ID_RE.match(execution_id):
            raise ValueError(
                f"Invalid execution_id {execution_id!r}: must match "
                f"{_EXECUTION_ID_RE.pattern}"
            )
        return self._dir / f"{execution_id}.json"

    def save(self, execution_id: str, state: dict[str, Any]) -> None:
        path = self._path(execution_id)
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def load(self, execution_id: str) -> dict[str, Any] | None:
        path = self._path(execution_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def delete(self, execution_id: str) -> None:
        self._path(execution_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))


def build_state(
    *,
    execution_id: str,
    status: str,
    model: str,
    graph: ExecutionGraph,
    registry: Registry,
    task: Task | None,
    max_budget_usd: float | None,
    node_costs: dict[str, float],
    output: str | None = None,
    created_at: float | None = None,
) -> dict[str, Any]:
    """Assemble a version-1 checkpoint state document."""
    now = time.time()
    return {
        "version": CHECKPOINT_VERSION,
        "execution_id": execution_id,
        "status": status,
        "created_at": created_at if created_at is not None else now,
        "updated_at": now,
        "model": model,
        "task": task_to_dict(task),
        "graph": graph_to_dict(graph),
        "agents": agents_to_list(registry),
        "budget": {
            "max_budget_usd": max_budget_usd,
            "node_costs": dict(node_costs),
        },
        "output": output,
    }
