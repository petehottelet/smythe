"""Read-only views of saved runs, and the status projection results share.

``smythe runs`` reads an ordinary run's checkpoint through a
:class:`~smythe.checkpoint.FileCheckpointStore`, or a durable run from a
:class:`~smythe.workflow_store.SQLiteWorkflowStore` opened read-only, and
reports execution, delivery acceptance and its decision history, a durable
run's published artifact, cost, timing and the next action as separate JSON
blocks. For a durable run it also reports its context sources and snapshots,
its journaled reads, the source evidence its output carried and the issues
left open, never source text, read bodies or output.
:meth:`smythe.swarm.SwarmResult.to_dict` builds its execution, delivery,
unknown-exposure and next-action blocks with the same functions. Nothing here
constructs a provider, migrates a journal or writes a file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from smythe._money import NANOUSD
from smythe._redact import redact_account_identifiers
from smythe.acceptance import (
    _OUTPUT_CHANGED,
    AcceptanceDecision,
    DeliveryAcceptance,
    _delivery_problems,
    task_sha256,
    text_sha256,
)
from smythe.budget import Sentinel
from smythe.checkpoint import (
    _EXECUTION_ID_RE,
    FileCheckpointStore,
    attempt_allowance_from_state,
    checkpoint_requirements,
    delivery_acceptance_from_state,
    delivery_acceptance_history,
    delivery_acceptance_sequence,
    graph_from_dict,
    task_from_dict,
    timing_segments,
)
from smythe.context import (
    CONTEXT_SNAPSHOT_KEY,
    CONTEXT_SNAPSHOT_VERSION,
    sources_from_operation,
    sources_key,
)
from smythe.evidence import load_source_evidence, load_source_tombstones
from smythe.graph import ExecutionGraph
from smythe.validation import _json_sha256
from smythe.workflow_artifact import CurrentArtifact, artifact_problems, current_artifact
from smythe.workflow_store import (
    CallKey,
    SQLiteWorkflowStore,
    WorkflowError,
    WorkflowNotFoundError,
)

MAX_LIST_RUNS = 500
# Decisions that withhold a completed run's output after a valid decision.
NOT_ACCEPTED = frozenset({AcceptanceDecision.REJECTED.value, AcceptanceDecision.NEEDS_REVIEW.value})


class RunInspectionError(Exception):
    """A saved run, or the store that holds it, cannot be read."""


class RunNotFoundError(LookupError):
    """The store holds no run with the requested id."""


def node_counts(graph: ExecutionGraph | None) -> dict[str, int]:
    """How many of the graph's nodes are in each status."""
    counts: dict[str, int] = {}
    for node in graph.nodes if graph is not None else ():
        counts[node.status.value] = counts.get(node.status.value, 0) + 1
    return dict(sorted(counts.items()))


def delivery_view(
    record: DeliveryAcceptance | None, text: Any, *, field: str, completed: bool,
    problems: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """How a run's output stands under its delivery acceptance record.

    *text* is the output the run saved, kept under *field*. *problems* are
    what no longer matches a decided record; any problem makes the decision
    ``indeterminate``, as resuming the run would. Output is delivered when
    a completed run has no record, or when the decision is ``accepted``;
    otherwise it is withheld and only referenced by digest and size.
    """
    if record is None:
        return {"recorded": False, "decision": None, "reasons": [], "integrity": None,
                "output_delivered": completed and isinstance(text, str),
                "rejected_output": None, "record": None}
    decision, reasons = record.decision.value, list(record.reasons)
    integrity = None
    if record.decision is not AcceptanceDecision.NOT_EVALUATED:
        integrity = "mismatch" if problems else "verified"
        if problems:
            decision, reasons = AcceptanceDecision.INDETERMINATE.value, list(problems)
    delivered = decision == AcceptanceDecision.ACCEPTED.value
    withheld = None
    if not delivered and isinstance(text, str):
        withheld = {"field": field, "sha256": text_sha256(text), "bytes": len(text.encode("utf-8"))}
    return {"recorded": True, "decision": decision, "reasons": reasons, "integrity": integrity,
            "output_delivered": delivered, "rejected_output": withheld, "record": record.to_dict()}


def decisions_view(
    current: int | None, history: list[tuple[int, DeliveryAcceptance]] | tuple = (),
) -> dict[str, Any]:
    """A run's decision history: the *current* decision's sequence, and each earlier decision.

    Sequence 0 is the decision the run made when it completed, and each
    re-evaluation adds one. *history* lists the decisions the current one
    superseded, oldest first, as ``(sequence, record)``.
    """
    return {"current": current,
            "history": [{"sequence": sequence, "record": record.to_dict()}
                        for sequence, record in history]}


def unknown_exposure(workflow_accounting: dict[str, Any] | None) -> dict[str, Any] | None:
    """A durable run's unknown exposure; None where the path does not track it."""
    if not isinstance(workflow_accounting, dict):
        return None
    return {"unit": workflow_accounting.get("unit"),
            "amount": workflow_accounting.get("unknown_nanousd"),
            "calls": workflow_accounting.get("unknown_calls")}


def _action(action: str, detail: str) -> dict[str, str]:
    return {"action": action, "detail": detail}


def next_action(
    state: str, delivery: dict[str, Any], *, cost_is_complete: bool,
    unresolved: list[str] | tuple[str, ...] = (), blocked_reason: str | None = None,
    resume: str = "", artifact: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The first thing a saved run needs, as a code and a sentence.

    Unresolved accounting comes first, because resume refuses it; then an
    unfinished run's resume (*resume* says how); then a completed run's
    published *artifact* whose file is missing or changed; then its withheld
    output, then its unknown cost.
    """
    if blocked_reason == "budget_overrun":
        return _action("rerun", "Spending passed the run's budget, so it admits no more calls; "
                                "run the task again as a new run.")
    if unresolved:
        return _action("reconcile_accounting",
                       "Reconcile provider charges before the run can continue: "
                       + "; ".join(unresolved) + ".")
    if state != "completed":
        return _action("resume", resume)
    if artifact is not None and any(reason.startswith("artifact_") for reason in
                                    artifact["delivery_eligibility"]["reasons"]):
        return _action("republish", "The published document is "
                                    f"{artifact['integrity']['status']}; publish it again with "
                                    "Swarm(run_store=...).republish(run_id), which never replaces "
                                    "a file.")
    decision = delivery["decision"]
    if decision in NOT_ACCEPTED:
        return _action("review_output", f"The decision is {decision}, so the output is withheld; "
                                        "review it before using it, or run the task again.")
    if decision == AcceptanceDecision.INDETERMINATE.value:
        return _action("resolve_indeterminate",
                       "No valid acceptance decision covers the output, which is withheld: "
                       + delivery["reasons"][0])
    if not cost_is_complete:
        return _action("reconcile_accounting", "The run finished, but part of its cost is "
                                               "unknown; reconcile it with the provider's billing.")
    if decision == AcceptanceDecision.ACCEPTED.value:
        return _action("none", "The run finished and its output was accepted.")
    return _action("none", "The run finished and delivered its output; no acceptance "
                           "decision was recorded.")


def result_view(result: Any) -> dict[str, Any]:
    """The execution, delivery, unknown-exposure and next-action blocks of a finished SwarmResult."""
    record = result.acceptance
    if record is not None and not record.accepted:
        text, field = result.rejected_output, "rejected_output"
    else:
        text, field = result.output, "output"
    problems = []
    if record is not None and record.decision is not AcceptanceDecision.NOT_EVALUATED and (
            not isinstance(text, str) or text_sha256(text) != record.output_sha256):
        problems.append(_OUTPUT_CHANGED)
    delivery = delivery_view(record, text, field=field, completed=True, problems=problems)
    return {
        "execution": {"state": "completed", "nodes": node_counts(result.graph)},
        "delivery": delivery,
        "unknown_exposure": unknown_exposure(result.workflow_accounting),
        "next_action": next_action("completed", delivery, cost_is_complete=result.cost_is_complete),
    }


def _number(value: Any) -> float | None:
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def _total(values: list[Any]) -> float | None:
    present = [number for number in map(_number, values) if number is not None]
    return round(sum(present), 1) if present else None


def _summed(segments: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    names: dict[str, list[Any]] = {}
    for segment in segments:
        block = segment.get(key)
        for name, value in (block.items() if isinstance(block, dict) else ()):
            names.setdefault(str(name), []).append(value)
    return {name: _total(values) for name, values in names.items()}


def checkpoint_timing(state: dict[str, Any]) -> dict[str, Any] | None:
    """Totals over an ordinary checkpoint's timing segments; None without any.

    Each segment's durations come from its own process's monotonic clock and
    are added, never subtracted; ``started_at`` and ``ended_at`` are the first
    and last segments' wall-clock bounds.
    """
    segments = timing_segments(state)
    if not segments:
        return None
    return {
        "clock": "monotonic", "segments": len(segments), "interrupted_segments": None,
        "elapsed_ms": _total([segment.get("elapsed_ms") for segment in segments]),
        "summed_node_ms": _total([segment.get("summed_node_ms") for segment in segments]),
        "phases_ms": _summed(segments, "phases_ms"),
        "waiting_ms": _summed(segments, "waiting_ms"),
        "started_at": _number(segments[0].get("started_at")),
        "ended_at": _number(segments[-1].get("ended_at")),
    }


def journal_timing(store: SQLiteWorkflowStore, run_id: str) -> dict[str, Any]:
    """Totals over a durable run's lease sessions, from journal wall-clock times."""
    record = store.lease_sessions(run_id)
    sessions = record["sessions"]
    return {
        "clock": "wall", "segments": len(sessions),
        "interrupted_segments": sum(1 for session in sessions if session["interrupted"]),
        "elapsed_ms": round(sum(session["ended_ns"] - session["started_ns"]
                                for session in sessions) / 1e6, 1),
        "summed_node_ms": None, "phases_ms": None, "waiting_ms": None,
        "started_at": record["created_ns"] / 1e9,
        "ended_at": sessions[-1]["ended_ns"] / 1e9 if sessions else None,
    }


def open_checkpoints(directory: str | Path) -> FileCheckpointStore:
    """A FileCheckpointStore over an existing directory, which is never created."""
    path = Path(directory)
    if not path.is_dir():
        raise RunInspectionError(f"Checkpoint directory does not exist: {path}")
    return FileCheckpointStore(path)


def open_journal(path: str | Path) -> SQLiteWorkflowStore:
    """An existing workflow journal opened read-only, so it is never migrated."""
    path = Path(path)
    if not path.is_file():
        raise RunInspectionError(f"Workflow journal does not exist: {path}")
    try:
        return SQLiteWorkflowStore(path, read_only=True)
    except WorkflowError as exc:
        raise RunInspectionError(f"Cannot read the workflow journal: {exc}") from exc


def _checkpoint_accounting(state: dict[str, Any], graph: ExecutionGraph) -> dict[str, Any]:
    """Restore the saved costs as resume does, and name what resume would refuse."""
    budget = state.get("budget", {})
    if not isinstance(budget, dict):
        raise ValueError("Checkpoint budget must be an object")
    token_estimates = budget.get("token_estimate_nodes", [])
    if not isinstance(token_estimates, list) or not all(isinstance(item, str) for item in token_estimates):
        raise ValueError("Checkpoint token_estimate_nodes must be a list of node IDs")
    unknown = sorted(node.id for node in graph.nodes if node.metadata.get("cost_usd_unknown"))
    sentinel = Sentinel(budget.get("max_budget_usd"))
    sentinel.restore(
        budget.get("node_costs", {}), unknown_cost_nodes=set(unknown),
        estimated_cost_nodes={node.id for node in graph.nodes if node.metadata.get("cost_usd_is_estimate")},
        token_estimate_nodes=set(token_estimates),
    )
    unresolved = []
    invalid = [node.id for node in graph.nodes if node.metadata.get("accounting_invalid")]
    if invalid:
        unresolved.append(f"invalid accounting on nodes {invalid!r}")
    if "accounting_error" in budget:
        unresolved.append("the budget records an accounting error")
    control = state.get("control", {})
    if ((isinstance(control, dict) and "response_error" in control)
            or any("response_error" in node.metadata for node in graph.nodes)):
        unresolved.append("a native response failure is unresolved")
    return {
        "cost_scope": "execution_and_synthesis",
        "total_cost_usd": sentinel.total_cost_usd,
        "max_budget_usd": sentinel.max_budget_usd,
        "cost_is_complete": sentinel.cost_is_complete,
        "cost_contains_estimates": sentinel.cost_contains_estimates,
        "cost_contains_token_estimates": sentinel.cost_contains_token_estimates,
        "unknown_cost_nodes": unknown,
        "workflow_accounting": None,
        "unknown_exposure": None,
        "unresolved": unresolved,
    }


def _checkpoint_view(execution_id: str, state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RunInspectionError(f"Checkpoint {execution_id!r} is not a JSON object")
    record = delivery_acceptance_from_state(state)
    status = state.get("status")
    if status not in ("running", "failed", "completed"):
        raise ValueError(f"unknown checkpoint status {status!r}")
    graph = graph_from_dict(state["graph"])
    saved_task = task_from_dict(state.get("task"))
    task = graph.task if graph.task is not None else saved_task
    output = state.get("output")
    problems = []
    if record is not None and record.decision is not AcceptanceDecision.NOT_EVALUATED:
        problems = _delivery_problems(record, output, graph, task)
    delivery = delivery_view(record, output, field="output", completed=status == "completed",
                             problems=problems)
    decisions = decisions_view(delivery_acceptance_sequence(state),
                               delivery_acceptance_history(state))
    accounting = _checkpoint_accounting(state, graph)
    resume = "Resume it with Swarm.resume(execution_id) from a Swarm with this checkpoint store"
    if record is not None:
        resume += f" and the strict acceptance policy whose sha256 is {record.policy_sha256}"
    allowance = attempt_allowance_from_state(state)
    if allowance is not None:
        resume += f" and the retry policy {allowance[0].to_dict()}"
    return {
        "run_id": execution_id,
        "store": "checkpoints",
        "format": {"checkpoint_version": state["version"],
                   "requires": list(checkpoint_requirements(state))},
        "execution": {"state": status, "blocked_reason": None, "nodes": node_counts(graph)},
        "delivery": delivery,
        "decisions": decisions,
        # Only durable runs publish a rendered document, journal reads and
        # record context snapshots and source evidence.
        "artifact": None,
        "context": None,
        "tool_calls": None,
        "evidence": None,
        "issues": issues_view(delivery, None, None),
        "accounting": accounting,
        "timing": checkpoint_timing(state),
        "next_action": next_action(
            status, delivery, cost_is_complete=accounting["cost_is_complete"],
            unresolved=accounting["unresolved"],
            resume=resume + ", once no process is running it.",
        ),
    }


def inspect_checkpoint(store: FileCheckpointStore, execution_id: str) -> dict[str, Any]:
    """Project one ordinary run's checkpoint without changing it."""
    if not isinstance(execution_id, str) or not _EXECUTION_ID_RE.match(execution_id):
        raise ValueError(f"Invalid execution id {execution_id!r}")
    try:
        state = store.load(execution_id)
    except (ValueError, RecursionError) as exc:
        # RecursionError: nesting deeper than the JSON parser accepts.
        raise RunInspectionError(f"Checkpoint {execution_id!r} is not readable JSON") from exc
    except OSError as exc:
        # FileNotFoundError is not raised: load() reports a missing file as None.
        raise RunInspectionError(f"Checkpoint {execution_id!r} is not readable: {exc}") from exc
    if state is None:
        raise RunNotFoundError(f"No checkpoint for run {execution_id!r}")
    try:
        return _checkpoint_view(execution_id, state)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise RunInspectionError(f"Checkpoint {execution_id!r} cannot be inspected: {exc}") from exc


def _journal_acceptance(
    store: SQLiteWorkflowStore, run: dict[str, Any], checkpoint: dict[str, Any],
    graph: ExecutionGraph, policy_sha256: str | None, artifact: CurrentArtifact | None,
) -> tuple[DeliveryAcceptance | None, list[str], dict[str, Any]]:
    """The delivery decision a durable checkpoint holds, what no longer matches it, and its history.

    Only the version 2 checkpoints of a run with a strict policy, whose
    digest is *policy_sha256*, hold a record; any other run is shown as not
    recorded, never as accepted. The record is read as resume reads it, and a
    decided one is checked as completed resume checks it: against the saved
    output, the task, the source generations and the journal's current
    decision, and for a run with a renderer against its current published
    *artifact*. The history lists every decision the journal applied.
    """
    from smythe.workflow import _decision_problems, _read_saved_record, decision_history

    if checkpoint["version"] == 1:
        return None, [], decisions_view(None)
    task = task_from_dict(run["task"])
    history = (decision_history(store, run["run_id"])
               if policy_sha256 is not None and checkpoint.get("status") == "completed" else [])
    record = _read_saved_record(store, run["run_id"], checkpoint, policy_sha256, task_sha256(task))
    decisions = decisions_view(history[-1]["sequence"] if history else None,
                               [(item["sequence"], item["record"]) for item in history[:-1]])
    if record is None or record.decision is AcceptanceDecision.NOT_EVALUATED:
        return record, [], decisions
    problems = _decision_problems(store, run["run_id"], graph, checkpoint.get("output"), record,
                                  task=task, policy_sha256=policy_sha256)
    if run["config"].get("renderer") is not None:
        problems += artifact_problems(artifact, checkpoint.get("output"))
    return record, problems, decisions


def _removal(record: Any) -> dict[str, Any] | None:
    return None if record is None else {"reason": record.tombstone, "removed_at": record.removed_at}


def context_view(store: SQLiteWorkflowStore, run_id: str, recipe: dict[str, Any],
                 checkpoint: dict[str, Any] | None, graph: ExecutionGraph | None) -> dict | None:
    """A durable run's context assembly: versions, current sources, snapshots and tombstones.

    None for a run without context assembly. Each source is named by id,
    version, digest and size, and a removed one by its tombstone's reason and
    time instead of its size; snapshots are named by node, generation and id.
    No source text is read out.
    """
    version = recipe.get("context_assembly", 0)
    if not version:
        return None
    tombstones = load_source_tombstones(store, run_id)
    removed = {(record.source_id, record.sha256): record for record in tombstones}
    binding = checkpoint.get("context_sources") if checkpoint is not None else None
    sources = []
    if isinstance(binding, dict) and type(binding.get("revision")) is int:
        operation = store.load_operation(run_id, sources_key(binding["revision"]))
        for source in (() if operation is None
                       else sources_from_operation(operation, binding["revision"])):
            tombstone = removed.get((source.source_id, source.sha256))
            sources.append({
                "source_id": source.source_id, "version": source.version, "sha256": source.sha256,
                "text_bytes": None if tombstone else len(source.text.encode("utf-8")),
                "trust": "untrusted", "removed": _removal(tombstone)})
    else:
        binding = None
    snapshots = []
    for node in graph.nodes if graph is not None else ():
        stamp = node.metadata.get(CONTEXT_SNAPSHOT_KEY)
        if isinstance(stamp, dict):
            snapshots.append({"node_id": node.id, "generation": stamp.get("generation"),
                              "snapshot_id": stamp.get("snapshot_id")})
    return {"assembly_version": version, "snapshot_version": CONTEXT_SNAPSHOT_VERSION,
            "sources_revision": None if binding is None else binding["revision"],
            "sources_sha256": None if binding is None else binding.get("sha256"),
            "sources": sources, "snapshots": snapshots,
            "tombstones": [record.to_dict() for record in tombstones]}


def tool_calls_view(store: SQLiteWorkflowStore, run_id: str,
                    calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A summary of each journaled read call, from the journal's call records.

    Each names its node, generation, attempt, profile and URL, its states,
    charge, HTTP status, failure and the response's SHA-256, and the size and
    SHA-256 of the text decoded from it, never the response body or text.
    """
    views = []
    for call in calls:
        key = call["key"]
        if key["phase"] != "tool":
            continue
        record = store.lookup_call(run_id, CallKey(**key))
        provider, receipt = record["provider"], record["receipt"] or {}
        request = json.loads(record["request_json"])
        decoded = record["decoded_result"]
        text = decoded.get("text") if isinstance(decoded, dict) else None
        url = request.get("url") if isinstance(request, dict) else None
        views.append({
            "node_id": key["scope_id"].removeprefix("node/"), "generation": key["generation"],
            "attempt": key["attempt"], "invocation": key["invocation"],
            "profile": provider.get("name"), "profile_version": provider.get("version"),
            "decoder_version": record["decoder_version"], "metered": provider.get("metered"),
            "repeat_safe": provider.get("repeat_safe"),
            "url": redact_account_identifiers(url) if isinstance(url, str) else None,
            "request_sha256": record["request_sha256"], "state": record["state"],
            "billing_state": record["billing_state"], "result_state": record["result_state"],
            "cost_nanousd": record["cost_nanousd"], "http_status": receipt.get("http_status"),
            "transport_error": receipt.get("transport_error"),
            "response_sha256": receipt.get("response_sha256"),
            "text_bytes": len(text.encode("utf-8", "surrogatepass")) if isinstance(text, str) else None,
            "text_sha256": text_sha256(text) if isinstance(text, str) else None,
            "error": record["error"],
        })
    return views


def evidence_view(store: SQLiteWorkflowStore, run_id: str) -> dict[str, Any] | None:
    """The source evidence a completed durable run delivered, summarized; None without any.

    Sources are their records' summaries, with ``available`` and ``removed``;
    each claim is its id, the number of passages for and against it, the
    sources they cite, its scope and its unresolved questions. Claim text,
    quotes and conclusions belong to the output and are left out.
    """
    loaded = load_source_evidence(store, run_id)
    if loaded is None:
        return None
    claims = [{
        "claim_id": claim["claim_id"], "supporting": len(claim["supporting"]),
        "contradicting": len(claim["contradicting"]),
        "sources": sorted({passage["source_id"]
                           for passage in claim["supporting"] + claim["contradicting"]}),
        "scope": claim["scope"], "unresolved": list(claim["unresolved"]),
    } for claim in loaded["claims"]]
    return {key: loaded[key] for key in ("operation_key", "state", "output_sha256",
                                         "context_sources", "problem", "sources")} | {"claims": claims}


def issues_view(delivery: dict[str, Any], evidence: dict[str, Any] | None,
                tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """What a run leaves open, one entry each, as ``kind``, ``subject`` and ``detail``.

    Kinds: ``evidence_problem`` when the output carried no readable evidence;
    ``source_unavailable`` for a cited source whose text the run does not
    hold; ``unresolved_question`` for each question a claim leaves open;
    ``read_failed`` for a read step with no accepted read at its generation,
    or a read with an unknown charge; and ``validator_rejected`` or
    ``validator_error`` for a final validator that did not pass.
    """
    issues = []

    def issue(kind: str, subject: Any, detail: Any) -> None:
        issues.append({"kind": kind, "subject": subject, "detail": detail})

    if evidence is not None:
        if evidence["problem"]:
            issue("evidence_problem", None, evidence["problem"])
        for source in evidence["sources"]:
            if not source["available"]:
                detail = ("removed from the run: " + source["removed"]["reason"] if source["removed"]
                          else "tombstoned: " + source["tombstone"] if source["tombstone"]
                          else "the run does not hold its text")
                issue("source_unavailable", source["source_id"], detail)
        for claim in evidence["claims"]:
            for question in claim["unresolved"]:
                issue("unresolved_question", claim["claim_id"], question)
    read = {(call["node_id"], call["generation"]) for call in tool_calls or ()
            if call["result_state"] in ("accepted", "applied")}
    for call in tool_calls or ():
        if call["billing_state"] == "unknown" or (
                call["result_state"] == "rejected" and (call["node_id"], call["generation"]) not in read):
            issue("read_failed", call["node_id"], call["error"] or call["state"])
    record = delivery["record"]
    for result in record["validators"] if record is not None else ():
        if result["outcome"] != "passed":
            name = result["validator"]["name"]
            detail = next((reason for reason in record["reasons"] if repr(name) in reason),
                          result["outcome"])
            issue("validator_" + result["outcome"], name, detail)
    return issues


def _journal_view(store: SQLiteWorkflowStore, run_id: str) -> dict[str, Any]:
    from smythe.workflow import _checkpoint_version

    inspected = store.inspect_run(run_id)
    run = store.load_run(run_id)
    saved = store.get_checkpoint(run_id)
    checkpoint = saved["checkpoint"] if saved is not None else None
    policy = run["config"].get("acceptance")
    policy_sha256 = None if policy is None else _json_sha256(policy, "acceptance policy")
    record, problems, graph, decisions = None, [], None, decisions_view(None)
    # The run's current publication, read through the one accessor resume uses.
    artifact = None
    if run["config"].get("renderer") is not None:
        artifact = current_artifact(store, run_id)
    if checkpoint is not None:
        # The run's recipe sets its checkpoint version: version 2 for a
        # strict policy or a gate validator, version 1 otherwise.
        version, expected = checkpoint.get("version"), _checkpoint_version(run["config"])
        if type(version) is not int or version not in (1, 2):
            raise RunInspectionError(
                f"Durable run {run_id!r} has checkpoint version {version!r}, which this build "
                "of smythe cannot read; it reads versions 1 and 2"
            )
        if version != expected:
            raise RunInspectionError(
                f"Durable run {run_id!r} has checkpoint version {version}, but its recipe "
                f"writes version {expected}"
            )
        graph = graph_from_dict(checkpoint["graph"])
        record, problems, decisions = _journal_acceptance(store, run, checkpoint, graph,
                                                          policy_sha256, artifact)
    status, blocked = inspected["status"], inspected["blocked_reason"]
    checkpoint_status = checkpoint.get("status") if checkpoint is not None else None
    if status in ("blocked", "completed"):
        state = status
    else:
        state = checkpoint_status if checkpoint_status in ("planned", "running", "failed") else "running"
    delivery = delivery_view(record, checkpoint.get("output") if checkpoint is not None else None,
                             field="output", completed=state == "completed", problems=problems)
    accounting_keys = ("confirmed_nanousd", "reserved_nanousd", "unknown_nanousd", "unknown_calls",
                       "call_count")
    workflow_accounting = {key: inspected[key] for key in accounting_keys} | {"unit": NANOUSD}
    budget = inspected["budget_nanousd"]
    accounting = {
        "cost_scope": "complete_text_workflow",
        "total_cost_usd": inspected["confirmed_nanousd"] / 1e9,
        "max_budget_usd": None if budget is None else budget / 1e9,
        "cost_is_complete": not inspected["unknown_calls"] and not inspected["reserved_nanousd"],
        "cost_contains_estimates": False,
        "cost_contains_token_estimates": False,
        "unknown_cost_nodes": None,
        "workflow_accounting": workflow_accounting,
        "unknown_exposure": unknown_exposure(workflow_accounting),
        "unresolved": [f"admission is closed by {blocked}"] if blocked else [],
    }
    resume = "Resume it with Swarm(run_store=...).resume(run_id) and the configuration it started with"
    if policy_sha256 is not None:
        resume += f", including the strict acceptance policy whose sha256 is {policy_sha256}"
    tool_calls = tool_calls_view(store, run_id, inspected["calls"])
    evidence = evidence_view(store, run_id)
    return {
        "run_id": run_id,
        "store": "journal",
        "format": {"journal_version": store.store_version,
                   "checkpoint_version": None if checkpoint is None else checkpoint["version"],
                   "revision": inspected["revision"]},
        "execution": {"state": state, "blocked_reason": blocked, "nodes": node_counts(graph),
                      "run_status": status, "checkpoint_status": checkpoint_status},
        "delivery": delivery,
        "decisions": decisions,
        "artifact": None if artifact is None else artifact.to_dict(),
        "context": context_view(store, run_id, run["config"], checkpoint, graph),
        "tool_calls": tool_calls,
        "evidence": evidence,
        "issues": issues_view(delivery, evidence, tool_calls),
        "accounting": accounting,
        "timing": journal_timing(store, run_id),
        "next_action": next_action(
            state, delivery, cost_is_complete=accounting["cost_is_complete"],
            unresolved=accounting["unresolved"], blocked_reason=blocked,
            resume=resume + ", once no worker holds its lease.",
            artifact=None if artifact is None else artifact.to_dict(),
        ),
    }


def inspect_journal(store: SQLiteWorkflowStore, run_id: str) -> dict[str, Any]:
    """Project one durable run from a journal opened read-only."""
    try:
        return _journal_view(store, run_id)
    except WorkflowNotFoundError as exc:
        raise RunNotFoundError(f"No durable run {run_id!r} in this journal") from exc
    except (WorkflowError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        raise RunInspectionError(f"Durable run {run_id!r} cannot be inspected: {exc}") from exc


def inspect_run(store: FileCheckpointStore | SQLiteWorkflowStore, run_id: str) -> dict[str, Any]:
    """Project one run from either kind of store."""
    if isinstance(store, SQLiteWorkflowStore):
        return inspect_journal(store, run_id)
    return inspect_checkpoint(store, run_id)


def list_runs(
    store: FileCheckpointStore | SQLiteWorkflowStore, *, limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    """One page of saved runs with their state, decision and next action.

    A run that cannot be read is listed with its error instead of stopping
    the listing.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_LIST_RUNS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LIST_RUNS}")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if isinstance(store, SQLiteWorkflowStore):
        try:
            ids = [run["run_id"] for run in store.list_runs()]
        except WorkflowError as exc:
            raise RunInspectionError(f"Cannot list the workflow journal: {exc}") from exc
    else:
        ids = store.list_ids()
    page = ids[offset:offset + limit]
    runs = []
    for run_id in page:
        try:
            view = inspect_run(store, run_id)
        except (RunInspectionError, RunNotFoundError, ValueError) as exc:
            runs.append({"run_id": run_id, "state": None, "decision": None, "next_action": None,
                         "error": str(exc)})
            continue
        runs.append({"run_id": run_id, "state": view["execution"]["state"],
                     "decision": view["delivery"]["decision"],
                     "next_action": view["next_action"]["action"], "error": None})
    return {"store": "journal" if isinstance(store, SQLiteWorkflowStore) else "checkpoints",
            "total": len(ids), "offset": offset, "limit": limit, "returned": len(runs),
            "has_more": offset + len(page) < len(ids), "runs": runs}
