"""Checkpointing — durable, resumable execution state.

After each node reaches a terminal status, the Swarm persists the full
execution state (graph, node results, agents, budget consumed) through a
CheckpointStore.  A crashed or interrupted execution can then be picked
up with ``swarm.resume(execution_id)``, re-running only the nodes that
never completed.

The state is a plain JSON document (version 3) so users can inspect or
repair checkpoints by hand.  See docs/checkpoint-format.md for the full
schema.
"""

from __future__ import annotations

import json
from copy import deepcopy
import os
import re
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology, snapshot_run_ref
from smythe.registry import Registry
from smythe.task import Task, task_from_dict, task_snapshots_equal, task_to_dict

CHECKPOINT_VERSION = 3
# v3 adds mandatory verification dispositions. Older graphs without an
# ambiguous unfinished gating decision remain readable.
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, 3)

_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _jsonable(value: Any) -> Any:
    """Take a detached JSON snapshot, falling back to str() for opaque values."""
    try:
        return json.loads(json.dumps(value))
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
        "max_tool_iterations": node.max_tool_iterations,
        "attach_dep_artifacts": node.attach_dep_artifacts,
        "verifies": node.verifies,
        "max_regenerations": node.max_regenerations,
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
        max_tool_iterations=data.get("max_tool_iterations"),
        attach_dep_artifacts=data.get("attach_dep_artifacts", False),
        verifies=data.get("verifies"),
        max_regenerations=data.get("max_regenerations", 0),
    )


def graph_to_dict(graph: ExecutionGraph) -> dict[str, Any]:
    return _graph_snapshot(graph, task_to_dict(graph.task))


def _graph_snapshot(graph: ExecutionGraph, task_data: dict[str, Any] | None) -> dict[str, Any]:
    """Serialize graph state using an already captured Task representation."""
    return {
        "topology": [t.value for t in graph.topology],
        "estimated_cost_usd": graph.estimated_cost_usd,
        "task": task_data,
        "run_ref": snapshot_run_ref(graph.run_ref),
        "nodes": [node_to_dict(n) for n in graph.nodes],
    }


def graph_from_dict(data: dict[str, Any]) -> ExecutionGraph:
    """Exact restore of a serialized graph, including statuses and results."""
    return ExecutionGraph(
        topology=[Topology(t) for t in data.get("topology", ["serial"])],
        nodes=[node_from_dict(n) for n in data.get("nodes", [])],
        estimated_cost_usd=data.get("estimated_cost_usd"),
        task=task_from_dict(data.get("task")),
        run_ref=snapshot_run_ref(data.get("run_ref")),
    )


def agents_to_list(registry: Registry) -> list[dict[str, Any]]:
    out = []
    for agent in registry.list_agents():
        entry: dict[str, Any] = {
            "id": agent.id,
            "name": agent.profile.name,
            "persona": agent.profile.persona,
            "capabilities": list(agent.profile.capabilities),
        }
        if agent.profile.mcp_servers:
            # env_passthrough stores variable NAMES only; secret values
            # never touch the checkpoint (see smythe.mcp).
            entry["mcp_servers"] = [s.to_dict() for s in agent.profile.mcp_servers]
        out.append(entry)
    return out


def agents_from_list(data: list[dict[str, Any]]) -> list[Agent]:
    agents = []
    for entry in data:
        mcp_servers = []
        if entry.get("mcp_servers"):
            from smythe.mcp import MCPServerSpec

            mcp_servers = [MCPServerSpec.from_dict(s) for s in entry["mcp_servers"]]
        agents.append(Agent(
            id=entry["id"],
            profile=AgentProfile(
                name=entry.get("name", entry["id"]),
                persona=entry.get("persona", ""),
                capabilities=list(entry.get("capabilities", [])),
                mcp_servers=mcp_servers,
            ),
        ))
    return agents


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
    unique temporary file, are flushed, and are moved into place with
    os.replace. POSIX also flushes the containing directory after replacement.
    Independent writers publish whole snapshots; the last replacement wins.
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
        with self._lock:
            serialized = json.dumps(state, indent=2)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.", suffix=".tmp", delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(serialized)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, path)
                temporary_path = None
                if os.name != "nt":
                    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        # Preserve the persistence error. A stranded file is
                        # never grounds to remove another writer's temporary.
                        pass

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
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint state document.

    ``control`` carries run-level counters that enforce bounded-loop
    guarantees. They must be durable: an allowance that resets on
    resume is not a cap, and a crash would silently buy more work
    than the caller authorised.
    """
    task_data = task_to_dict(task)
    graph_task_data = task_data if graph.task is task else task_to_dict(graph.task)
    if (task_data is not None and graph_task_data is not None
            and not task_snapshots_equal(task_data, graph_task_data)):
        raise ValueError("Checkpoint graph Task conflicts with the top-level Task")
    task_data = task_data if task_data is not None else graph_task_data
    now = time.time()
    return {
        "version": CHECKPOINT_VERSION,
        "execution_id": execution_id,
        "status": status,
        "created_at": created_at if created_at is not None else now,
        "updated_at": now,
        "model": model,
        "task": deepcopy(task_data),
        "graph": _graph_snapshot(graph, deepcopy(task_data)),
        "agents": agents_to_list(registry),
        "budget": {
            "max_budget_usd": max_budget_usd,
            "node_costs": dict(node_costs),
        },
        "output": output,
        "control": dict(control or {"revisions_used": 0}),
    }
