"""Checkpointing — durable, resumable execution state.

After each node reaches a terminal status, the Swarm persists the full
execution state (graph, node results, agents, budget consumed) through a
CheckpointStore.  A crashed or interrupted execution can then be picked
up with ``swarm.resume(execution_id)``, re-running only the nodes that
never completed.

The state is a plain JSON document (version 4, or 5 with state that older
builds cannot read) so users can inspect or repair checkpoints by hand.  See
docs/checkpoint-format.md for the full schema.
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
from typing import Any, Iterable

from smythe.acceptance import AcceptanceDecision, DeliveryAcceptance, text_sha256
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology, snapshot_run_ref
from smythe.registry import Registry
from smythe.task import Task, task_from_dict, task_snapshots_equal, task_to_dict

CHECKPOINT_VERSION = 4
# v3 adds mandatory verification dispositions. Older graphs without an
# ambiguous unfinished gating decision remain readable. v4 changes no field:
# it certifies that every agent MCP server it records came from developer
# configuration, because builds that write v4 never take servers from model
# output. Earlier versions may hold servers a model-generated plan declared.
# v5 adds a top-level ``requires`` list naming the state features a reader must
# understand, such as a ``delivery_acceptance`` record. Only a state that
# carries such a feature is written as v5, so builds that read v4 refuse it,
# and a reader refuses any feature it does not know, instead of finishing or
# delivering a run it would misread.
STRICT_CHECKPOINT_VERSION = 5
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, 3, 4, 5)
MCP_TRUSTED_CHECKPOINT_VERSION = 4
# The version 5 state features this build reads.
KNOWN_CHECKPOINT_REQUIREMENTS = frozenset({"delivery_acceptance"})

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


def drop_untrusted_mcp_servers(agents: list[Agent], *, version: int) -> list[str]:
    """Remove MCP servers restored from a checkpoint older than version 4.

    Before 0.8.1 a model-generated plan could declare MCP servers, and resume
    would start them as local programs.  Version 4 checkpoints record only
    developer-configured servers, so theirs are kept.  Returns the removed
    servers as ``"<agent id>/<server name>"``.
    """
    if version >= MCP_TRUSTED_CHECKPOINT_VERSION:
        return []
    dropped = [f"{agent.id}/{spec.name}" for agent in agents for spec in agent.profile.mcp_servers]
    for agent in agents:
        agent.profile.mcp_servers = []
    return dropped


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
    timing: dict[str, Any] | None = None,
    delivery_acceptance: DeliveryAcceptance | None = None,
    token_estimate_nodes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint state document.

    ``control`` carries run-level counters that enforce bounded-loop
    guarantees. They must be durable: an allowance that resets on
    resume is not a cap, and a crash would silently buy more work
    than the caller authorised.

    ``timing`` is an informational block of per-process timing segments.
    It is additive within version 4: readers ignore keys they do not use.

    ``token_estimate_nodes`` names the nodes whose recorded cost prices
    tokens at the blended rate. It is stored, sorted, as
    ``budget.token_estimate_nodes``, which is also additive within version 4.

    ``delivery_acceptance`` makes the state version 5, lists the feature in
    ``requires`` and stores the record under that key. A record that binds an
    output digest needs the matching ``output``, and a completed state's
    record must hold a decision.
    """
    task_data = task_to_dict(task)
    graph_task_data = task_data if graph.task is task else task_to_dict(graph.task)
    if (task_data is not None and graph_task_data is not None
            and not task_snapshots_equal(task_data, graph_task_data)):
        raise ValueError("Checkpoint graph Task conflicts with the top-level Task")
    task_data = task_data if task_data is not None else graph_task_data
    now = time.time()
    state = {
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
    if token_estimate_nodes is not None:
        state["budget"]["token_estimate_nodes"] = sorted(token_estimate_nodes)
    if timing is not None:
        state["timing"] = deepcopy(timing)
    if delivery_acceptance is not None:
        if not isinstance(delivery_acceptance, DeliveryAcceptance):
            raise TypeError("delivery_acceptance must be a DeliveryAcceptance record")
        bound = delivery_acceptance.output_sha256
        if bound is not None and (not isinstance(output, str) or text_sha256(output) != bound):
            raise ValueError("The delivery acceptance record does not match the checkpoint output")
        if status == "completed" and delivery_acceptance.decision is AcceptanceDecision.NOT_EVALUATED:
            raise ValueError("A completed checkpoint's delivery acceptance record must hold a decision")
        state["version"] = STRICT_CHECKPOINT_VERSION
        state["requires"] = ["delivery_acceptance"]
        state["delivery_acceptance"] = delivery_acceptance.to_dict()
    return state


def checkpoint_requirements(state: dict[str, Any]) -> tuple[str, ...]:
    """The state features a checkpoint requires its reader to understand.

    Versions 1-4 require none. A version 5 state lists them in ``requires``;
    a missing or malformed list, or a feature this build does not know,
    raises ValueError for unfinished and completed states alike.
    """
    version = state.get("version")
    if type(version) is not int or version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Checkpoint version {version!r} is not one of {list(SUPPORTED_CHECKPOINT_VERSIONS)}"
        )
    if version < STRICT_CHECKPOINT_VERSION:
        return ()
    requires = state.get("requires")
    if not isinstance(requires, list) or any(not isinstance(name, str) for name in requires):
        raise ValueError(
            "A version 5 checkpoint must list the state features it needs under 'requires'"
        )
    unknown = sorted(set(requires) - KNOWN_CHECKPOINT_REQUIREMENTS)
    if unknown:
        raise ValueError(
            f"Checkpoint requires state features this build of smythe cannot read: {unknown}. "
            "Re-run the task instead of resuming."
        )
    return tuple(requires)


def delivery_acceptance_from_state(state: dict[str, Any]) -> DeliveryAcceptance | None:
    """The delivery acceptance record a checkpoint state carries, if any.

    Versions 1-4 predate delivery acceptance, so they have no record whatever
    keys they hold: a legacy run is never reported as accepted or as not yet
    evaluated. A version 5 state has a record exactly when its ``requires``
    list names ``delivery_acceptance``; it is stored under that key, and a
    completed state's record must hold a decision. The record is read as
    stored; comparing it with the saved output is left to the caller.
    Raises ValueError for an unsupported version, an unknown requirement, a
    missing or unlisted record, or an unreadable one.
    """
    requires = checkpoint_requirements(state)
    if state["version"] < STRICT_CHECKPOINT_VERSION:
        return None
    block = state.get("delivery_acceptance")
    if "delivery_acceptance" not in requires:
        if block is not None:
            raise ValueError(
                "Checkpoint carries a delivery_acceptance record that its 'requires' list omits"
            )
        return None
    if block is None:
        raise ValueError(
            "A checkpoint that requires delivery_acceptance must carry its delivery_acceptance record"
        )
    record = DeliveryAcceptance.from_dict(block)
    if state.get("status") == "completed" and record.decision is AcceptanceDecision.NOT_EVALUATED:
        raise ValueError("A completed checkpoint's delivery acceptance record must hold a decision")
    return record
