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
from smythe.scheduling import RetryPolicy, _attempt_records
from smythe.task import Task, task_from_dict, task_snapshots_equal, task_to_dict

CHECKPOINT_VERSION = 4
# v3 adds mandatory verification dispositions. Older graphs without an
# ambiguous unfinished gating decision remain readable. v4 changes no field:
# it certifies that every agent MCP server it records came from developer
# configuration, because builds that write v4 never take servers from model
# output. Earlier versions may hold servers a model-generated plan declared.
# v5 adds a top-level ``requires`` list naming the state features a reader must
# understand, such as a ``delivery_acceptance`` record or an
# ``attempt_allowance`` that resume must not refill. Only a state that
# carries such a feature is written as v5, so builds that read v4 refuse it,
# and a reader refuses any feature it does not know, instead of finishing or
# delivering a run it would misread.
STRICT_CHECKPOINT_VERSION = 5
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, 3, 4, 5)
MCP_TRUSTED_CHECKPOINT_VERSION = 4
# The version 5 state features this build reads. ``operator_decision`` is
# listed while the record or a kept earlier one is an operator's decision.
OPERATOR_DECISION_REQUIREMENT = "operator_decision"
KNOWN_CHECKPOINT_REQUIREMENTS = frozenset(
    {"delivery_acceptance", "attempt_allowance", OPERATOR_DECISION_REQUIREMENT})
# The control keys of an attempt allowance (see RetryPolicy.persist_attempts),
# which older builds would ignore and so refill.
_ALLOWANCE_KEYS = frozenset({"retry_policy", "attempts_used"})
# How many superseded delivery decisions a completed checkpoint keeps. The
# history is informational, so it is not a required state feature: a reader
# that does not know it still reads the current decision correctly.
DELIVERY_ACCEPTANCE_HISTORY_LIMIT = 16

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
    snapshot = {
        "topology": [t.value for t in graph.topology],
        "estimated_cost_usd": graph.estimated_cost_usd,
        "task": task_data,
        "run_ref": snapshot_run_ref(graph.run_ref),
        "nodes": [node_to_dict(n) for n in graph.nodes],
    }
    # Informational and additive: a graph without planning records keeps its
    # bytes, and readers that do not know the key ignore it.
    if graph.planning is not None:
        snapshot["planning"] = deepcopy(graph.planning)
    return snapshot


def graph_from_dict(data: dict[str, Any]) -> ExecutionGraph:
    """Exact restore of a serialized graph, including statuses and results."""
    planning = data.get("planning")
    if planning is not None and not isinstance(planning, dict):
        raise ValueError("Graph planning records must be a JSON object")
    return ExecutionGraph(
        topology=[Topology(t) for t in data.get("topology", ["serial"])],
        nodes=[node_from_dict(n) for n in data.get("nodes", [])],
        estimated_cost_usd=data.get("estimated_cost_usd"),
        task=task_from_dict(data.get("task")),
        run_ref=snapshot_run_ref(data.get("run_ref")),
        planning=deepcopy(planning),
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
    record must hold a decision. A ``control`` block with ``retry_policy``
    and ``attempts_used`` makes it version 5 too, listing ``attempt_allowance``.
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
    requires = []
    if delivery_acceptance is not None:
        if not isinstance(delivery_acceptance, DeliveryAcceptance):
            raise TypeError("delivery_acceptance must be a DeliveryAcceptance record")
        bound = delivery_acceptance.output_sha256
        if bound is not None and (not isinstance(output, str) or text_sha256(output) != bound):
            raise ValueError("The delivery acceptance record does not match the checkpoint output")
        if status == "completed" and delivery_acceptance.decision is AcceptanceDecision.NOT_EVALUATED:
            raise ValueError("A completed checkpoint's delivery acceptance record must hold a decision")
        requires.append("delivery_acceptance")
    if _ALLOWANCE_KEYS & state["control"].keys():
        requires.append("attempt_allowance")
    if delivery_acceptance is not None and delivery_acceptance.operator is not None:
        requires.append(OPERATOR_DECISION_REQUIREMENT)
    if requires:
        state["version"] = STRICT_CHECKPOINT_VERSION
        state["requires"] = requires
    if delivery_acceptance is not None:
        state["delivery_acceptance"] = delivery_acceptance.to_dict()
    return state


def timing_segments(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Timing segments earlier calls saved; none if the block is unreadable.

    Timing is informational, so an unreadable block never stops a resume.
    """
    block = state.get("timing")
    if not isinstance(block, dict) or type(block.get("version")) is not int or block["version"] != 1:
        return []
    segments = block.get("segments")
    if not isinstance(segments, list) or not all(isinstance(item, dict) for item in segments):
        return []
    try:
        return json.loads(json.dumps(segments, allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        return []


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
    missing or unlisted record, or an unreadable one, and for an operator
    decision that an unfinished state holds or ``requires`` does not list.
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
    return _listed(state, requires, record)


def _listed(
    state: dict[str, Any], requires: tuple[str, ...], record: DeliveryAcceptance,
) -> DeliveryAcceptance:
    """*record*, unless it is an operator decision that *state* may not hold."""
    if record.operator is not None and (OPERATOR_DECISION_REQUIREMENT not in requires
                                        or state.get("status") != "completed"):
        raise ValueError(
            "Only a completed checkpoint that lists operator_decision in 'requires' holds an "
            "operator decision"
        )
    return record


def attempt_allowance_from_state(
    state: dict[str, Any],
) -> tuple[RetryPolicy, dict[str, dict[str, int]]] | None:
    """The retry policy and attempts used that a checkpoint's attempt allowance records.

    A version 5 state has an allowance exactly when its ``requires`` list
    names ``attempt_allowance``: ``control.retry_policy``, a policy with
    ``persist_attempts``, and ``control.attempts_used``, each node's
    ``execution_generation`` and attempts spent there. Versions 1-4 have
    none, whatever keys they hold. Raises ValueError for an unknown
    requirement, or an allowance that is missing, unlisted or unreadable.
    """
    requires = checkpoint_requirements(state)
    if state["version"] < STRICT_CHECKPOINT_VERSION:
        return None
    control = state.get("control")
    present = isinstance(control, dict) and bool(_ALLOWANCE_KEYS & control.keys())
    if "attempt_allowance" not in requires:
        if present:
            raise ValueError("Checkpoint records an attempt allowance that its 'requires' list omits")
        return None
    if not present or not _ALLOWANCE_KEYS <= control.keys():
        raise ValueError(
            "A checkpoint that requires attempt_allowance must record control.retry_policy "
            "and control.attempts_used"
        )
    policy = RetryPolicy.from_dict(control["retry_policy"])
    if not policy.persist_attempts:
        raise ValueError("A checkpoint's attempt allowance must record a policy with persist_attempts")
    _attempt_records(control["attempts_used"])
    return policy, deepcopy(control["attempts_used"])


def delivery_acceptance_history(
    state: dict[str, Any],
) -> tuple[tuple[int, DeliveryAcceptance], ...]:
    """The decisions a completed checkpoint's current record superseded, oldest first.

    Each is ``(sequence, record)``: sequence 0 is the decision the run made
    when it completed, and each re-evaluation adds one, so the current
    record's sequence follows the last one listed
    (:func:`delivery_acceptance_sequence`). They are kept under
    ``delivery_acceptance_history``, at most
    ``DELIVERY_ACCEPTANCE_HISTORY_LIMIT`` of them, the oldest dropped first,
    and each decided the bytes the current record binds. A state without the
    key has none. Raises ValueError for a malformed history, or one that a
    state without a decided record carries.
    """
    data = state.get("delivery_acceptance_history")
    if data is None:
        return ()
    current = delivery_acceptance_from_state(state)
    if (state.get("status") != "completed" or current is None
            or current.decision is AcceptanceDecision.NOT_EVALUATED):
        raise ValueError(
            "Only a completed checkpoint with a delivery decision keeps earlier decisions"
        )
    if not isinstance(data, list) or not 1 <= len(data) <= DELIVERY_ACCEPTANCE_HISTORY_LIMIT:
        raise ValueError(
            f"delivery_acceptance_history must list 1 to {DELIVERY_ACCEPTANCE_HISTORY_LIMIT} "
            "earlier decisions"
        )
    history: list[tuple[int, DeliveryAcceptance]] = []
    for entry in data:
        if not isinstance(entry, dict) or set(entry) != {"sequence", "delivery_acceptance"}:
            raise ValueError(
                "Each earlier decision must hold its sequence and its delivery_acceptance"
            )
        sequence = entry["sequence"]
        if type(sequence) is not int or sequence < 0 or (history and sequence <= history[-1][0]):
            raise ValueError("Earlier decision sequences must be increasing non-negative integers")
        record = DeliveryAcceptance.from_dict(entry["delivery_acceptance"])
        if (record.decision is AcceptanceDecision.NOT_EVALUATED
                or record.output_sha256 != current.output_sha256):
            raise ValueError(
                "Each earlier decision must have decided the output the current one binds"
            )
        history.append((sequence, _listed(state, checkpoint_requirements(state), record)))
    return tuple(history)


def delivery_acceptance_sequence(state: dict[str, Any]) -> int | None:
    """The current decision's sequence: 0 until a re-evaluation; None without a decided record."""
    record = delivery_acceptance_from_state(state)
    if record is None or record.decision is AcceptanceDecision.NOT_EVALUATED:
        return None
    history = delivery_acceptance_history(state)
    return history[-1][0] + 1 if history else 0


def supersede_delivery_acceptance(
    state: dict[str, Any], record: DeliveryAcceptance,
) -> dict[str, Any]:
    """A copy of a completed checkpoint *state* whose current decision is *record*.

    The decision it replaces joins ``delivery_acceptance_history`` with its
    sequence (see :func:`delivery_acceptance_history`), ``requires`` lists
    ``operator_decision`` exactly while *record* or a kept earlier decision is
    an operator's, and ``updated_at`` is the current time; every other field
    is copied as saved. Raises ValueError
    unless *state* is a completed checkpoint with a decided record and
    *record* is a decision that binds the same output, which the state's
    ``output`` still matches.
    """
    current = delivery_acceptance_from_state(state)
    sequence = delivery_acceptance_sequence(state)
    if state.get("status") != "completed" or current is None or sequence is None:
        raise ValueError("Only a completed checkpoint's delivery decision can be superseded")
    if (not isinstance(record, DeliveryAcceptance)
            or record.decision is AcceptanceDecision.NOT_EVALUATED):
        raise ValueError("A new delivery decision must be a decided DeliveryAcceptance record")
    output = state.get("output")
    if (record.output_sha256 != current.output_sha256 or not isinstance(output, str)
            or text_sha256(output) != record.output_sha256):
        raise ValueError("A new delivery decision must bind the output its checkpoint holds")
    history = [*delivery_acceptance_history(state), (sequence, current)]
    kept = history[-DELIVERY_ACCEPTANCE_HISTORY_LIMIT:]
    updated = deepcopy(state)
    updated["delivery_acceptance"] = record.to_dict()
    updated["delivery_acceptance_history"] = [
        {"sequence": number, "delivery_acceptance": item.to_dict()} for number, item in kept
    ]
    requires = [name for name in updated["requires"] if name != OPERATOR_DECISION_REQUIREMENT]
    if record.operator or any(item.operator for _, item in kept):
        requires.append(OPERATOR_DECISION_REQUIREMENT)
    updated["requires"] = requires
    updated["updated_at"] = time.time()
    return updated
