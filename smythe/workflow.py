"""Run-scoped orchestration over an exact, durable text-call ledger."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from smythe._money import FLOOR, NANOUSD, to_units
from smythe.acceptance import (
    AcceptanceDecision, AcceptanceMode, AcceptancePolicy, AcceptanceSource, DeliveryAcceptance,
    ValidationSubject, _delivery_problems, _final_check, _operator_record, evaluate_delivery,
    task_sha256, text_sha256, validate_acceptance_obligations,
)
from smythe.budget import BudgetValidationError, Sentinel, validate_completion_usage
from smythe.checkpoint import (
    graph_from_dict, graph_to_dict, node_from_dict, node_to_dict, reset_incomplete_nodes,
)
from smythe.context import (
    CONTEXT_ASSEMBLY_VERSION, CONTEXT_SNAPSHOT_KEY, SUPPORTED_CONTEXT_ASSEMBLY_VERSIONS,
    ContextSnapshot, DependencyDigest, context_key, snapshot_sources, sources_from_operation,
    sources_key, sources_sha256, text_sha256 as output_sha256,
)
from smythe.evidence import (
    SOURCE_EVIDENCE_OPERATION, SOURCE_TOMBSTONE_OPERATION, SourceRecord, _removed_sources,
    _tombstone_operations, source_evidence_inputs, source_evidence_key, source_evidence_result,
    source_tombstone_key,
)
from smythe.executor_base import REJECTED_GENERATIONS_KEY, SUPPORTED_PROMPT_ASSEMBLY_VERSIONS, ExecutorBase
from smythe.graph import ExecutionGraph, NodeStatus, Revision, snapshot_run_ref
from smythe.lineage import LINEAGE_REUSE_KEY, snapshot_lineage_policy
from smythe.registry import Registry
from smythe.provider import ProviderResponseError
from smythe.task import Task, snapshot_task, task_from_dict, task_to_dict
from smythe.tracer import Tracer
from smythe.validation import ValidationOutcome, _json_sha256
from smythe.validators import SourceEvidenceValidator
from smythe.verifier import (
    ADVISORY_VERDICT_KEY, TOKEN_VERIFIER_IDENTITY, TokenVerifier, Verdict, Verifier,
    node_generation, validate_verification_checkpoint, verification_pending,
)
from smythe.workflow_artifact import (
    PUBLICATION_OPERATION, MarkdownRenderer, PublishedArtifact, artifact_problems, claim_directory,
    current_artifact, owner_bytes, publication_key, publications, publish_bytes, read_publication,
    read_published, render_document, renderer_settings,
)
from smythe.workflow_binding import (
    CallScope, ComponentBinding, LocalOnly, WorkflowBindingError, bind_component,
    describe_component, json_snapshot,
)
from smythe.workflow_provider import (
    WorkflowProviderContext, describe_provider, validate_workflow_model,
)
from smythe.workflow_policy import snapshot_graph_policy
from smythe.workflow_store import (
    CallKey, SQLiteWorkflowStore, WorkflowConflictError, WorkflowError, WorkflowLeaseError,
    WorkflowStateError, WorkflowValidationError, _is_http_rejection, _registered_price_version,
)
from smythe.workflow_tools import READ_STEP_KEY, JournaledReadTool, ReadProfile, reinvocable

# Journal keys that embed a node id stop at 512 characters. Capping durable node
# ids well below that leaves room for "supervision/<id>/<generation>".
MAX_NODE_ID_CHARS = 256

# Recipe compatibility. Plan, execute and resume compare a saved recipe with
# the current one exactly, field by field, except for two kinds of field:
#
# - Pricing metadata. A provider description's price_version and
#   supported_models follow the price registry, which a release extends. The
#   saved description may name any registered price version for its kind,
#   and its model list may differ; its other fields still compare exactly.
#   Saved calls keep the price version they recorded, and admission checks
#   each new call against the current version.
# - Behavior-version fields, which select a versioned behavior such as a
#   prompt-assembly version. This maps each field to (absent, default): the
#   value a recipe without the field means, and the value a new run records
#   unless WorkflowRuntime's behavior_settings argument configures another.
#   A recipe records the field only when its value differs from `absent`, so
#   existing recipes keep their bytes. Resume reads the value from the saved
#   recipe into WorkflowRuntime.behavior, so a changed default applies only to
#   new runs; a value configured explicitly must equal the saved one. The
#   settings are read-only after construction, which builds the recipe a new
#   run records from them.
#
# prompt_assembly selects how node prompts are assembled (see
# smythe.executor_base.PROMPT_ASSEMBLY_VERSION). Version 1, which recipes
# without the field mean and new runs still record by default, tells every
# node that no other node depends on that it is the final step, a judge
# included. Version 2 tells the nodes DELIVERABLE synthesis returns, as
# ordinary runs do; only behavior_settings records it, and no public setting
# passes that yet. A run keeps its version, so its saved requests still match
# when it resumes.
#
# context_assembly selects how node calls are given their context (see
# smythe.context). Version 0, which recipes without the field mean and new
# runs record by default, sends the prompts of every earlier release. Version
# 1, which context_sources selects, records each call's context snapshot and
# renders its prompt from it, with sources and dependency outputs marked as
# untrusted data.
RECIPE_BEHAVIOR_FIELDS = MappingProxyType({"prompt_assembly": (1, 1), "context_assembly": (0, 0)})
# The values this release supports for a behavior-version field. A configured
# value outside them is refused, and so is a saved recipe that records one,
# for example a recipe a newer release wrote.
_BEHAVIOR_VERSIONS = MappingProxyType({"prompt_assembly": SUPPORTED_PROMPT_ASSEMBLY_VERSIONS,
                                       "context_assembly": SUPPORTED_CONTEXT_ASSEMBLY_VERSIONS})
# Node metadata that belongs to a node's current generation. A refresh of a
# run's sources starts every node's next generation without it.
_GENERATION_KEYS = ("artifacts", "artifacts_discarded", "verification_receipt",
                    "verification_target_generation", ADVISORY_VERDICT_KEY, "regeneration_intent",
                    "workflow_supervision", CONTEXT_SNAPSHOT_KEY)
_PROVIDER_FIELDS = frozenset({"kind", "adapter_version", "decoder_version", "endpoint_scope",
                              "price_version", "supported_models", "config"})


def _canonical(value):
    return json.dumps(json_snapshot(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_pricing(value, *, saved=False):
    """Copy a recipe without its provider descriptions' pricing metadata.

    A saved description keeps its fields when its price version is no longer
    registered, so it cannot match a current one.
    """
    if isinstance(value, list):
        return [_without_pricing(item, saved=saved) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if (key == "provider" and isinstance(item, dict) and item.keys() == _PROVIDER_FIELDS
                and (not saved or _registered_price_version(item["kind"], item["price_version"]))):
            result[key] = {name: field for name, field in item.items()
                           if name not in ("price_version", "supported_models")}
        else:
            result[key] = _without_pricing(item, saved=saved)
    return result


def _snapshot_behavior_settings(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise WorkflowBindingError("behavior_settings must be a mapping")
    settings = json_snapshot(dict(settings))
    unknown = sorted(set(settings) - set(RECIPE_BEHAVIOR_FIELDS))
    if unknown:
        raise WorkflowBindingError("Unknown workflow behavior version: " + ", ".join(unknown))
    for name, value in settings.items():
        if not _supported_behavior(name, value):
            raise WorkflowBindingError(f"Unsupported workflow {name} version: {value!r}")
    return MappingProxyType(settings)


def _supported_behavior(name, value):
    versions = _BEHAVIOR_VERSIONS.get(name)
    # Versions are integers; a bool or float equal to one is not a version.
    return versions is None or (type(value) is int and value in versions)


def _snapshot_read_tools(read_tools):
    """The declared read profiles by name; each name once, built-in profiles only."""
    if read_tools is None:
        read_tools = ()
    if isinstance(read_tools, (str, bytes, Mapping)) or not hasattr(read_tools, "__iter__"):
        raise WorkflowBindingError("read_tools must be a sequence of ReadProfile declarations")
    profiles = {}
    for profile in read_tools:
        if type(profile) is not ReadProfile:
            raise WorkflowBindingError(
                f"read_tools takes ReadProfile declarations, not {type(profile).__name__}")
        if profile.name in profiles:
            raise WorkflowBindingError(f"read_tools declares {profile.name!r} more than once")
        profiles[profile.name] = profile
    return MappingProxyType(profiles)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _nanousd(value):
    # A budget floors to whole nanoUSD, whatever the caller's decimal context.
    return None if value is None else to_units(value, NANOUSD, FLOOR)


# The recipe's "verifier" for gates judged by a registered validator: the
# rules below that turn its report into a verdict. Change it with them.
VALIDATOR_VERIFIER_IDENTITY = "validator-v1"


class _ValidatorVerifier(Verifier):
    """Judges each gate by a registered validator's report on its target's exact text.

    A ``passed`` report passes and a ``rejected`` one fails. A validator that
    raises, returns anything but its own report or reports ``error``, and a
    target without text, give a failing verdict marked ``invalid``, which
    ends the gate without a verdict, with or without a strict policy.
    """

    # The executor ends a gate whose verdict is marked invalid instead of
    # regenerating: a validator that could not decide never spends a
    # regeneration or buys another judge call.
    _invalid_ends_gate = True

    def __init__(self, validator):
        self.validator, self.identity = validator, validator.identity

    def verdict(self, verifier_node, verified_node):
        if not isinstance(verified_node.result, str):
            return Verdict(False, f"{verified_node.id!r} has no text to validate", invalid=True)
        outcome, reason = _final_check(
            self.validator, self.identity, ValidationSubject.from_text(verified_node.result),
            kind="gate validator",
        )
        return Verdict(outcome is ValidationOutcome.PASSED, reason,
                       invalid=outcome is ValidationOutcome.ERROR)


def _checkpoint_version(recipe):
    """The durable checkpoint version a run with *recipe* writes.

    Version 2 holds a delivery acceptance block. Only a recipe with a strict
    policy or a gate validator writes it.
    """
    return 2 if "acceptance" in recipe or "validators" in recipe else 1


def _saved_record(state, policy_sha256, task_digest, reevaluated=()):
    """A version 2 checkpoint's record: undecided until completed, then the decision.

    *policy_sha256* is the run's strict policy, or None for a recipe without
    one, whose checkpoints hold no record; *task_digest* is the run's task. A
    completed checkpoint's decision may instead name one of the policies in
    *reevaluated*, which the run's re-evaluations decided under. Raises
    WorkflowStateError for a block that does not match the run.
    """
    if "delivery_acceptance" not in state:
        raise WorkflowStateError("Workflow checkpoint has no delivery acceptance block")
    data = state["delivery_acceptance"]
    if policy_sha256 is None:
        if data is not None:
            raise WorkflowStateError("Workflow checkpoint holds a record its recipe does not decide")
        return None
    try:
        record = DeliveryAcceptance.from_dict(data)
    except (TypeError, ValueError) as error:
        raise WorkflowStateError("Workflow checkpoint delivery acceptance is invalid") from error
    completed = state.get("status") == "completed"
    undecided = DeliveryAcceptance(AcceptanceDecision.NOT_EVALUATED, policy_sha256,
                                   task_sha256=task_digest)
    if (record.policy_sha256 not in {policy_sha256, *(reevaluated if completed else ())}
            or (record.decision is AcceptanceDecision.NOT_EVALUATED) is completed
            or not completed and record != undecided):
        raise WorkflowStateError("Workflow checkpoint delivery acceptance does not match its run")
    return record


def _synthesis_key(graph, *, scoped):
    """The synthesis operation's key: ``synthesis`` while every node is at generation 0.

    A recipe with a strict policy, a gate validator or context assembly
    (*scoped*) keys synthesis for any other graph state by a short digest of
    every node's id and generation, so synthesis for one graph state never
    replays another's output, whatever their generations add up to. Other
    recipes keep the one key their journals already hold.
    """
    generations = {node.id: node_generation(node) for node in graph.nodes} if scoped else {}
    if not any(generations.values()):
        return "synthesis"
    return "synthesis/" + _hash(generations)[:16]


def _decision_inputs(synthesis_key, policy_sha256, task_digest, output_sha256, strategy,
                     sources, used_fallback, sequence=0, tombstones=(), operator=None):
    """Everything a decision binds; its hash keys the acceptance operation.

    A re-evaluation's decision also binds its *sequence*, so deciding the
    same bytes under the same policy again is a new operation. A decision
    made while the run holds source tombstones binds their journal keys
    (*tombstones*, see :func:`_tombstone_keys`): the removals it was made
    under. A decision made before a removal therefore never matches one made
    after it. An operator's decision binds *operator* (see
    :func:`_operator_inputs`).
    """
    inputs = {
        "synthesis": synthesis_key, "output_sha256": output_sha256,
        "policy_sha256": policy_sha256, "task_sha256": task_digest,
        "strategy": strategy, "used_fallback": used_fallback,
        "sources": [source.to_dict() for source in sources],
    }
    if sequence:
        inputs["sequence"] = sequence
    if tombstones:
        inputs["tombstones"] = list(tombstones)
    if operator is not None:
        inputs["operator"] = operator
    return inputs


def _tombstone_keys(store, run_id):
    """The journal keys of the source tombstones a durable run committed, sorted."""
    return sorted(source_tombstone_key(record.source_id, record.sha256)
                  for record, _ in _tombstone_operations(store, run_id))


def _operator_inputs(record):
    """What an operator's decision binds beyond a policy's: its decision and operator; else None."""
    if record.operator is None:
        return None
    return {"decision": record.decision.value, **record.operator.to_dict()}


def _decision_key(inputs):
    """The acceptance operation's key: ``acceptance/<digest>``, with the sequence after it.

    A re-evaluation's key is ``acceptance/<sequence>/<digest>``.
    """
    sequence = inputs.get("sequence", 0)
    return f"acceptance/{sequence}/{_hash(inputs)}" if sequence else "acceptance/" + _hash(inputs)


def _decision_sequence(key):
    """The decision sequence an ``acceptance`` operation's key carries: 0 for the run's own."""
    parts = key.split("/")
    if len(parts) == 2 and parts[0] == "acceptance":
        return 0
    if (len(parts) == 3 and parts[0] == "acceptance" and parts[1].isascii()
            and parts[1].isdecimal() and parts[1] == str(int(parts[1])) and int(parts[1]) > 0):
        return int(parts[1])
    raise WorkflowStateError(f"Delivery decision key {key!r} is not one this release writes")


def decision_history(store, run_id):
    """A durable run's delivery decisions, oldest first.

    Each is a dict: ``sequence``, 0 for the decision the run made when it
    completed and one more for each re-evaluation; ``operation_key`` and
    ``inputs`` of its ``acceptance`` operation; ``revision``, the checkpoint
    that applied it; and ``record``, its :class:`DeliveryAcceptance`. Only
    decisions a checkpoint applied count, ordered by sequence and then by
    that checkpoint. Only reads are made, so a store opened read-only will do.
    """
    history = []
    for operation in store.list_operations(run_id, "acceptance"):
        if operation["applied_revision"] is None:
            continue
        try:
            record = DeliveryAcceptance.from_dict(operation["result"].get("delivery_acceptance"))
        except (TypeError, ValueError) as error:
            raise WorkflowStateError("Saved delivery acceptance is invalid") from error
        history.append({"sequence": _decision_sequence(operation["operation_key"]),
                        "operation_key": operation["operation_key"],
                        "revision": operation["applied_revision"],
                        "inputs": operation["inputs"], "record": record})
    return sorted(history, key=lambda decision: (decision["sequence"], decision["revision"]))


def current_decision(store, run_id):
    """A durable run's current delivery decision (see :func:`decision_history`), or None.

    The current decision is the latest one: a re-evaluation's, after one.
    Resuming a completed run, re-evaluating it and ``smythe runs inspect``
    all check the run's saved record against this decision.
    """
    history = decision_history(store, run_id)
    return history[-1] if history else None


def _reevaluation_policies(store, run_id):
    """The policies a durable run's re-evaluations decided under."""
    return {decision["record"].policy_sha256 for decision in decision_history(store, run_id)
            if decision["sequence"]}


def _read_saved_record(store, run_id, state, policy_sha256, task_digest):
    """A checkpoint's record (see :func:`_saved_record`), as every reader of a run reads it.

    A completed run's decision may be one that a re-evaluation made under
    its own policy, so resume, re-evaluation, republication, the current
    artifact and ``smythe runs inspect`` all accept those policies here.
    """
    reevaluated = (_reevaluation_policies(store, run_id)
                   if policy_sha256 is not None and state.get("status") == "completed" else ())
    return _saved_record(state, policy_sha256, task_digest, reevaluated)


def _held_record(record, output, problems, task_digest):
    """*record* when there are no *problems*, else ``indeterminate`` for *output* with them."""
    if not problems:
        return record
    return DeliveryAcceptance(
        decision=AcceptanceDecision.INDETERMINATE, policy_sha256=record.policy_sha256,
        output_sha256=text_sha256(output), strategy=record.strategy, sources=record.sources,
        used_fallback=record.used_fallback, task_sha256=task_digest, reasons=tuple(problems),
    )


def _removal_problems(store, run_id, considered):
    """The sources a completed run delivered from that were removed after its decision.

    *considered* holds the tombstone keys the decision binds (see
    :func:`_decision_inputs`), empty without a decision. Any other tombstone
    came after the decision, whichever checkpoint applied either. One reason
    each; only reads are made.
    """
    saved = store.get_checkpoint(run_id)
    binding = None if saved is None else saved["checkpoint"].get("context_sources")
    if not isinstance(binding, dict) or type(binding.get("revision")) is not int:
        return []
    operation = store.load_operation(run_id, sources_key(binding["revision"]))
    if operation is None:
        return []
    delivered = {(source.source_id, source.sha256)
                 for source in sources_from_operation(operation, binding["revision"])}
    return [f"source {record.source_id!r} was removed from the run after this decision "
            f"({record.tombstone}); re-evaluate the run"
            for record, _ in _tombstone_operations(store, run_id)
            if (record.source_id, record.sha256) in delivered
            and source_tombstone_key(record.source_id, record.sha256) not in considered]


def _decision_problems(store, run_id, graph, output, record, *, task, policy_sha256,
                       removed=True):
    """What no longer matches a completed run's decided *record*, one reason each.

    The saved *output* must match the record's digest, the run's *task* and
    the source generations must be those it binds, and the record must be
    the journal's :func:`current_decision`, made under the run's policy
    (*policy_sha256*) or, for a re-evaluation, the policy it names, and under
    tombstones the journal holds. With *removed*, a source the run delivered
    from that was removed after the decision
    (:meth:`WorkflowRuntime.remove_source`) is a problem too. An empty list
    means the record holds. Only reads are made, so a store opened read-only
    will do.
    """
    problems = _delivery_problems(record, output, graph, task)
    decision = current_decision(store, run_id)
    holds = decision is not None and decision["record"].to_dict() == record.to_dict()
    considered = []
    if holds:
        sequence = decision["sequence"]
        deciding = record.policy_sha256 if sequence else policy_sha256
        considered = decision["inputs"].get("tombstones", [])
        if type(considered) is not list or any(type(key) is not str for key in considered):
            considered, holds = [], False
        inputs = _decision_inputs(
            _synthesis_key(graph, scoped=True), deciding, task_sha256(task),
            record.output_sha256, record.strategy, record.sources, record.used_fallback,
            sequence=sequence, tombstones=considered, operator=_operator_inputs(record),
        )
        # A decision binds each tombstone once, sorted; the journal keeps them all.
        holds = (holds and record.policy_sha256 == deciding and decision["inputs"] == inputs
                 and decision["operation_key"] == _decision_key(inputs)
                 and (not considered or (considered == sorted(set(considered))
                                         and set(considered) <= set(_tombstone_keys(store, run_id)))))
    if not holds:
        considered = []
        problems.append("the journal holds no applied decision matching the saved record")
    if removed:
        problems += _removal_problems(store, run_id, considered)
    return problems


class WorkflowBudget:
    """Read-only compatibility projection; the journal alone admits and settles."""

    workflow_managed = True
    cost_per_token = 0.0

    def __init__(self, store, run_id, max_budget_usd):
        self.store, self.run_id, self.max_budget_usd = store, run_id, max_budget_usd
        self._node_calls = {}

    def snapshot(self):
        return self.store.inspect_run(self.run_id)

    @property
    def total_cost_usd(self):
        return self.snapshot()["confirmed_nanousd"] / 1_000_000_000

    @property
    def cost_is_complete(self):
        state = self.snapshot()
        return not state["unknown_calls"] and not state["reserved_nanousd"]

    @property
    def cost_contains_estimates(self):
        return False

    def check(self, node_id):
        state = self.store.load_run(self.run_id)
        if state["blocked_reason"]:
            raise WorkflowStateError("Workflow admission is closed: " + state["blocked_reason"])

    def reserve(self, *args, **kwargs):
        raise WorkflowStateError("Only the call journal may reserve workflow funds")

    def release(self, node_id):
        # A node failure cannot release a dispatched call's unknown exposure.
        pass

    def mark_unknown(self, node_id):
        pass  # The provider facade has already persisted the raw failure.

    def add_cost(self, node_id, result, **kwargs):
        validate_completion_usage(result)
        receipt = result.native_receipt or {}
        if (receipt.get("workflow_run_id") != self.run_id
                or receipt.get("workflow_charge_recorded") is not True):
            raise WorkflowStateError("Completion is not settled in this workflow")
        call_id = receipt.get("workflow_call_id")
        records = {call["call_id"]: call for call in self.snapshot()["calls"]}
        if call_id not in records or records[call_id]["billing_state"] != "known":
            raise WorkflowStateError("Completion has no confirmed journal charge")
        self._node_calls.setdefault(node_id, set()).add(call_id)
        return sum(records[key]["cost_nanousd"] for key in self._node_calls[node_id]) / 1e9

    record = add_cost

    def breakdown(self):
        records = {call["call_id"]: call for call in self.snapshot()["calls"]}
        return {node: sum(records[key]["cost_nanousd"] for key in calls) / 1e9
                for node, calls in self._node_calls.items()}


class WorkflowRuntime:
    """One isolated binding per plan/execute/resume entry, one persistent run.

    A strict ``acceptance`` policy decides delivery after synthesis, as a
    journal operation that the completed checkpoint consumes, and enters the
    recipe so that the run finishes only under that policy. As on ordinary
    runs, it marks the gates of a generated plan, refuses before planning a
    run no gate of its plan could accept, and refuses a graph that does not
    meet its acceptance obligations before any node runs. :meth:`reevaluate`
    records a new decision for a completed run under this runtime's policy,
    or an authorized operator's for output left for review.

    ``read_tools`` declares the :class:`~smythe.workflow_tools.ReadProfile`
    reads that the graph's read steps may name. The recipe records their
    descriptions only when there are any, and each read step runs as a
    journaled ``tool`` call without a model call.

    ``context_sources``, a sequence of :class:`~smythe.context.ContextSource`
    (possibly empty), opts a new run into context assembly version 1: its
    recipe records ``context_assembly``, its first checkpoint records the
    sources as revision 0, and each node call is given a journaled
    :class:`~smythe.context.ContextSnapshot`. A run that already recorded its
    sources keeps them, whatever sources a later runtime is given;
    :meth:`refresh_context` replaces them explicitly.

    ``renderer``, a :class:`~smythe.workflow_artifact.MarkdownRenderer` or a
    renderer declared with ``LocalOnly(..., role="renderer")``, publishes the
    run's delivered document into ``artifact_dir/<run id>`` after synthesis
    (see :mod:`smythe.workflow_artifact`); the recipe records its
    description only when there is one. Without a renderer, ``artifact_dir``
    is ignored, as before. :meth:`republish` publishes a completed run's
    document again when its file is missing or changed.

    ``lineage``, a :class:`~smythe.lineage.LineagePolicy`, needs a strict
    ``acceptance`` policy. The recipe records it only when it is configured,
    and compares it exactly on every entry, like ``graph_policy``.
    """

    def __init__(self, *, store, model, max_budget_usd, provider, architect, registry,
                 router, synthesizer, supervisor, max_revisions, verifier, memory,
                 tool_runtime, checkpoint_store, checkpoint_every_n_nodes, retry_backoff_s,
                 max_concurrency, graph_policy=None, behavior_settings=None, acceptance=None,
                 read_tools=(), context_sources=None, renderer=None, artifact_dir=None,
                 retry_policy=None, lineage=None,
                 reserve_for_checks=False):
        if type(store) is not SQLiteWorkflowStore:
            raise WorkflowBindingError(
                f"run_store must be SQLiteWorkflowStore itself, not {type(store).__name__}"
            )
        Sentinel(max_budget_usd)
        # Each unsupported argument is named, so a caller knows what to remove.
        if retry_policy is not None:
            raise WorkflowBindingError(
                "retry_policy is not supported by durable text workflows, whose journal "
                "settles rejected calls; use node failure policies and retry_backoff_s"
            )
        if reserve_for_checks:
            raise WorkflowBindingError(
                "reserve_for_checks is not supported by durable text workflows, whose journal "
                "admits each call against the run's budget when it is made"
            )
        if memory is not None:
            raise WorkflowBindingError(
                "memory is not supported by durable text workflows; planner memory "
                "needs an ordinary Swarm run without run_store"
            )
        if tool_runtime is not None:
            raise WorkflowBindingError(
                "tool_runtime is not supported by durable text workflows, which send "
                "plain text without tools; tools need an ordinary Swarm run"
            )
        if checkpoint_store is not None:
            raise WorkflowBindingError(
                "checkpoint_store is not supported by durable text workflows, which "
                "save their checkpoints in run_store"
            )
        # A gate validator is declared, so its identity and version enter the
        # recipe; other verifier code is unversioned and is refused.
        validator = type(verifier) is LocalOnly and verifier.role == "validator"
        if verifier is not None and type(verifier) is not TokenVerifier and not validator:
            raise WorkflowBindingError(
                "verifier must be the built-in TokenVerifier or a validator declared with "
                "LocalOnly(..., role='validator') in durable text workflows, "
                f"not {type(verifier).__name__}"
            )
        if acceptance is not None and not isinstance(acceptance, AcceptancePolicy):
            raise WorkflowBindingError(
                f"acceptance must be an AcceptancePolicy or None, not {type(acceptance).__name__}"
            )
        # Only a strict policy decides delivery; an advisory one records nothing.
        if acceptance is not None and acceptance.mode is not AcceptanceMode.STRICT:
            acceptance = None
        lineage = snapshot_lineage_policy(lineage)
        if lineage is not None and acceptance is None:
            raise WorkflowBindingError(
                "lineage requires a strict acceptance policy, whose decision its record binds"
            )
        if acceptance is not None and not acceptance.final_validators:
            from smythe.swarm import _synthesis_strategy

            strategy = _synthesis_strategy(synthesizer)
            if strategy != "deliverable":
                raise WorkflowBindingError(
                    "acceptance without a final validator can accept only DELIVERABLE output "
                    f"that a passed gate judged; {strategy} synthesis needs a final validator"
                )
        if renderer is not None:
            if type(renderer) is not MarkdownRenderer and not (
                    type(renderer) is LocalOnly and renderer.role == "renderer"):
                shown = (f"a LocalOnly {renderer.role}" if type(renderer) is LocalOnly
                         else type(renderer).__name__)
                raise WorkflowBindingError(
                    "renderer must be MarkdownRenderer or a renderer declared with "
                    f"LocalOnly(..., role='renderer'), not {shown}"
                )
            if not isinstance(artifact_dir, (str, os.PathLike)):
                raise WorkflowBindingError("renderer needs an artifact_dir to publish into")
            if acceptance is not None and not acceptance.final_validators:
                raise WorkflowBindingError(
                    "acceptance without a final validator accepts only a node's result that a "
                    "passed gate judged, and a rendered document is not one; add a final validator"
                )
        if type(max_revisions) is not int or max_revisions < 0:
            raise WorkflowBindingError("max_revisions must be a non-negative integer")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise WorkflowBindingError("Durable workflows require a positive concurrency cap")
        if (type(retry_backoff_s) not in (int, float) or not math.isfinite(retry_backoff_s)
                or retry_backoff_s < 0):
            raise WorkflowBindingError("retry_backoff_s must be finite and non-negative")
        self.store, self.model, self.max_budget_usd = store, model, max_budget_usd
        self.provider = provider
        self.sources = {"architect": architect, "registry": registry, "router": router,
                        "synthesizer": synthesizer, "supervisor": supervisor}
        self.max_revisions, self.verifier = max_revisions, verifier
        self.gate_validator = verifier if validator else None
        self.acceptance = acceptance
        self.undecided = self.saved_acceptance = None
        # The strict policy the bound run's recipe records, by digest.
        self.run_policy_sha256 = None
        self.max_concurrency = max_concurrency
        self.execution_concurrency = None
        self.retry_backoff_s = retry_backoff_s
        self.checkpoint_every_n_nodes = checkpoint_every_n_nodes
        self.graph_policy = snapshot_graph_policy(graph_policy)
        self.lineage = lineage
        self.read_tools = _snapshot_read_tools(read_tools)
        # The run's directory is <artifact_dir>/<run id>; no renderer, no publication.
        self.renderer = renderer
        self.artifact_dir = None if renderer is None else Path(artifact_dir).resolve()
        # Explicitly configured behavior versions (see RECIPE_BEHAVIOR_FIELDS),
        # which the recipe below records for a new run, and, once a run is
        # bound, the versions that run uses.
        settings = _snapshot_behavior_settings(behavior_settings)
        if context_sources is not None:
            context_sources = snapshot_sources(context_sources)
            version = settings.get("context_assembly", CONTEXT_ASSEMBLY_VERSION)
            if version == 0:
                raise WorkflowBindingError("context_sources need a context_assembly version above 0")
            settings = MappingProxyType({**settings, "context_assembly": version})
        elif settings.get("context_assembly", 0):
            context_sources = ()
        self._behavior_settings = settings
        # The sources a run that has recorded none records as revision 0;
        # None when this runtime configures no context assembly.
        self._supplied_sources = context_sources
        self.behavior = {}
        self.recipe = self._describe()
        self.checkpoint_version = _checkpoint_version(self.recipe)
        self.graph = None
        self.executor = None
        self.revision = 0
        self._revisions_used = 0
        # The bound run's source revision and sources, under context assembly.
        self.context_revision, self.context_sources = None, ()
        self._snapshot_ids = {}
        self._operation = ContextVar("workflow_operation", default=None)
        self._operation_keys = {}
        self._worker_keys = {}
        self._pending_operations = set()
        self._updates = 0
        self.trace = Tracer()

    @property
    def behavior_settings(self):
        """Read-only behavior versions configured for this runtime."""
        return self._behavior_settings

    @property
    def _context_version(self):
        """The bound run's context-assembly version; 0 when it assembles none."""
        return self.behavior.get("context_assembly", 0)

    def _describe(self):
        validate_workflow_model(self.provider, self.model)
        recipe = {
            "type": "text_workflow", "version": 1, "model": self.model,
            "provider": describe_provider(self.provider),
            "components": {name: describe_component(
                component, role=name, default_provider=self.provider, default_model=self.model,
            ) for name, component in self.sources.items()},
            "max_revisions": self.max_revisions, "retry_backoff_s": self.retry_backoff_s,
            "max_concurrency": self.max_concurrency,
            "verifier": "token-v1", "checkpoint_every_n_nodes": self.checkpoint_every_n_nodes,
        }
        # Omit the field for existing unbounded recipes: adding a null would
        # change their identity and prevent otherwise compatible recovery.
        if self.graph_policy is not None:
            recipe["graph_policy"] = self.graph_policy.to_dict()
        if self.lineage is not None:
            # Identity, like graph_policy: a run keeps its scope and reuse mode.
            recipe["lineage"] = self.lineage.to_dict()
        if self.gate_validator is not None:
            recipe["verifier"] = VALIDATOR_VERIFIER_IDENTITY
            recipe["validators"] = [describe_component(self.gate_validator, role="validator")]
        if self.acceptance is not None:
            # Validator identities, never code. "token-v1" also named the
            # reading rules before 0.8.1, so a strict run names the current ones.
            recipe["acceptance"] = self.acceptance.to_dict()
            if self.gate_validator is None:
                recipe["verifier"] = TOKEN_VERIFIER_IDENTITY
        if self.read_tools:
            # Profile descriptions, never decoder code; omitted when none.
            recipe["read_tools"] = [self.read_tools[name].workflow_descriptor()
                                    for name in sorted(self.read_tools)]
        if self.renderer is not None:
            # Identity and version, never code; omitted when nothing is published.
            recipe["renderer"] = describe_component(self.renderer, role="renderer")
        for name, (absent, default) in RECIPE_BEHAVIOR_FIELDS.items():
            value = self.behavior_settings.get(name, default)
            if _canonical(value) != _canonical(absent):
                recipe[name] = value
        return json_snapshot(recipe)

    def _adopt_recipe(self, saved, *, reevaluating=False):
        """Check a saved recipe against this runtime; bind its behavior versions.

        *reevaluating* tolerates a recipe whose acceptance policy and gate
        verifier differ: a re-evaluation decides under this runtime's policy
        and reads the verdicts the run's gates recorded without judging again.
        """
        saved, current = dict(saved), dict(self.recipe)
        policy = saved.get("acceptance")
        self.run_policy_sha256 = (None if policy is None
                                  else _json_sha256(policy, "acceptance policy"))
        if reevaluating:
            for name in ("acceptance", "verifier", "validators"):
                saved.pop(name, None)
                current.pop(name, None)
        behavior = {}
        for name, (absent, _) in RECIPE_BEHAVIOR_FIELDS.items():
            current.pop(name, None)
            behavior[name] = saved.pop(name, absent)
            if not _supported_behavior(name, behavior[name]):
                raise WorkflowConflictError(
                    f"Workflow {name} {behavior[name]!r} in the saved recipe is not supported "
                    "by this release"
                )
            if (name in self.behavior_settings
                    and _canonical(self.behavior_settings[name]) != _canonical(behavior[name])):
                raise WorkflowConflictError(f"Workflow {name} differs from the saved recipe")
        if _canonical(saved.get("acceptance")) != _canonical(current.get("acceptance")):
            raise WorkflowConflictError(
                "Workflow acceptance policy differs from the saved recipe; resume the run with "
                "the policy it started with"
            )
        if _canonical(saved.get("read_tools")) != _canonical(current.get("read_tools")):
            raise WorkflowConflictError(
                "Workflow read_tools differ from the saved recipe; resume the run with the read "
                "profiles it started with"
            )
        if _canonical(saved.get("renderer")) != _canonical(current.get("renderer")):
            raise WorkflowConflictError(
                "Workflow renderer differs from the saved recipe; resume the run with the renderer "
                "it started with"
            )
        if _canonical(saved.get("lineage")) != _canonical(current.get("lineage")):
            raise WorkflowConflictError(
                "Workflow lineage policy differs from the saved recipe; resume the run with the "
                "lineage policy it started with"
            )
        if _canonical(_without_pricing(saved, saved=True)) != _canonical(_without_pricing(current)):
            raise WorkflowConflictError("Workflow components or budget differ from the saved recipe")
        self.behavior = behavior

    def _reference(self):
        return {"version": 1, "store_id": self.store.store_id, "run_id": self.run_id,
                "recipe_sha256": self.run["config_sha256"]}

    def _bind(self):
        # Planners run _check_plan on each candidate inside their retry loops.
        binding = ComponentBinding(self.run_id, "workflow", "execution", self._component_call,
                                   default_provider=self.provider, default_model=self.model,
                                   plan_check=self._check_plan)
        self.execution_provider = binding.snapshot_provider(self.provider)
        phases = {"architect": "planning", "router": "routing", "registry": "execution",
                  "synthesizer": "synthesis", "supervisor": "supervision"}
        self.bound = {name: bind_component(component, binding.child(name, phase=phases[name]),
                                          role=name)
                      for name, component in self.sources.items()}
        sources = [self.execution_provider]
        for component in self.bound.values():
            if component is not None:
                sources.extend(component.workflow_providers())
        self.providers, self.provider_ids = {}, {}
        for source in sources:
            if id(source) not in self.provider_ids:
                key = f"provider-{len(self.providers)}"
                self.providers[key] = source
                self.provider_ids[id(source)] = key
        self.registry = self.bound["registry"]
        self.gate_verifier = self.verifier
        if self.gate_validator is not None:
            # A fresh validator per run, which must be the one the recipe names.
            validator = bind_component(self.gate_validator, binding.child("validator"),
                                       role="validator")
            if validator.identity.config_sha256 != self.recipe["validators"][0]["config_sha256"]:
                raise WorkflowBindingError("Gate validator configuration differs from the recipe")
            self.gate_verifier = _ValidatorVerifier(validator)
        self.bound_renderer = None
        if self.renderer is not None:
            self.bound_renderer = bind_component(self.renderer, binding.child("renderer"),
                                                 role="renderer")
            settings = renderer_settings(self.bound_renderer)
            self.renderer_filename, self.renderer_media_type = settings

    def _component_call(self, source, scope):
        operation_key = _hash(scope.trigger)
        invocation = self.store.allocate_invocation(
            self.context.lease, scope.phase, scope.component_id, scope.generation, operation_key,
        )
        key = CallKey(scope.phase, scope.component_id, scope.generation,
                      invocation, scope.attempt, scope.turn)
        keys = [key]
        # A zero-cost provider rejection is final for its call. Re-entering
        # the last call of an invocation (for example on resume after HALT)
        # moves to a new, durably allocated invocation instead of replaying
        # the rejection. A rejection that a later attempt or turn already
        # followed replays, so a retry reaches that recorded call instead of
        # buying its step again.
        while (_is_http_rejection(self.store.lookup_call(self.run_id, key))
               and not self.store.has_later_call(self.run_id, key)):
            invocation = self.store.allocate_invocation(
                self.context.lease, scope.phase, scope.component_id, scope.generation,
                f"{operation_key}/after-http-rejection/{len(keys)}",
            )
            key = CallKey(scope.phase, scope.component_id, scope.generation,
                          invocation, scope.attempt, scope.turn)
            keys.append(key)
        current = self._operation.get()
        if current is not None:
            self._operation_keys.setdefault(current, set()).update(keys)
        return self.context.for_call(self.provider_ids[id(source)], key)

    def _worker_call(self, node, phase, attempt, turn):
        trigger = {"node_id": node.id}
        if self._context_version:
            # The call names the snapshot it was rendered from, so a call
            # under a different snapshot is a different journal call.
            snapshot_id = self._snapshot_ids.get((node.id, node_generation(node)))
            if snapshot_id is None:
                raise WorkflowStateError(f"Node {node.id!r} has no context snapshot to call with")
            trigger["context_snapshot"] = snapshot_id
        scope = CallScope(f"node/{node.id}", phase, trigger,
                          generation=node_generation(node), attempt=attempt, turn=turn)
        result = self._component_call(self.execution_provider, scope)
        self._worker_keys.setdefault((node.id, node_generation(node)), set()).add(result.key)
        return result

    def _read_call(self, node, attempt):
        """Bind a read step's attempt to its journal call, in the ``tool`` phase.

        Re-entering the last call of an invocation that a free, repeat-safe
        read settled as rejected, such as one whose response was lost, moves
        to a new invocation and reads again; a rejection that a later attempt
        already followed replays. Other rejections replay, and a read with an
        unknown charge is never sent again.
        """
        step = node.metadata[READ_STEP_KEY]
        generation, scope_id = node_generation(node), f"node/{node.id}"
        operation_key = _hash({"node_id": node.id, "read_step": step})

        def allocate(suffix=""):
            invocation = self.store.allocate_invocation(
                self.context.lease, "tool", scope_id, generation, operation_key + suffix)
            return CallKey("tool", scope_id, generation, invocation, attempt, 0)

        keys = [allocate()]
        while (reinvocable(self.store.lookup_call(self.run_id, keys[-1]))
               and not self.store.has_later_call(self.run_id, keys[-1])):
            keys.append(allocate(f"/after-read-rejection/{len(keys)}"))
        self._worker_keys.setdefault((node.id, generation), set()).update(keys)
        return JournaledReadTool(self.context, self.read_tools[step["profile"]], keys[-1], step["input"])

    def _call_ids(self, keys):
        ids = []
        for key in keys:
            record = self.store.lookup_call(self.run_id, key)
            if record is not None and record["result_state"] in {"accepted", "rejected", "applied"}:
                ids.append(record["call_id"])
        return sorted(set(ids))

    @asynccontextmanager
    async def _session(self, run, *, reevaluating=False):
        self.run, self.run_id = run, run["run_id"]
        if run["budget_nanousd"] != _nanousd(self.max_budget_usd):
            raise WorkflowConflictError("Workflow components or budget differ from the saved recipe")
        self._adopt_recipe(run["config"], reevaluating=reevaluating)
        self.context_revision, self.context_sources, self._snapshot_ids = None, (), {}
        if self.run_policy_sha256 is not None:
            # The record a strict run's unfinished checkpoints carry.
            self.undecided = DeliveryAcceptance(
                AcceptanceDecision.NOT_EVALUATED, self.run_policy_sha256,
                task_sha256=task_sha256(task_from_dict(run["task"])),
            )
        self._bind()
        lease = self.store.acquire_lease(self.run_id, uuid4().hex, ttl_s=60)
        self.context = WorkflowProviderContext(self.store, lease, self.providers)
        self.budget = WorkflowBudget(self.store, self.run_id, self.max_budget_usd)
        self.revision = run["revision"]
        parent = asyncio.current_task()
        heartbeat_error = []

        async def heartbeat():
            try:
                while True:
                    await asyncio.sleep(15)
                    self.context.update_lease(self.store.heartbeat(self.context.lease, ttl_s=60))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                heartbeat_error.append(error)
                parent.cancel()

        pulse = asyncio.create_task(heartbeat())
        try:
            async with self.context:
                self.store.recover(self.context.lease)
                # Late raw evidence can resolve a previously unknown call.
                # Settle every saved envelope before any new admission; decoding
                # remains local to the original logical call's replay.
                for call in self.store.inspect_run(self.run_id)["calls"]:
                    if call["evidence_id"] and call["billing_state"] != "known":
                        self.store.settle_call(self.context.lease, call["call_id"], call["evidence_id"])
                yield
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            try:
                self.store.release_lease(self.context.lease)
            except WorkflowLeaseError:
                if not heartbeat_error:
                    raise
            if heartbeat_error:
                raise WorkflowLeaseError("Workflow lease renewal failed") from heartbeat_error[0]

    def _new_run(self, task):
        return self.store.create_run(task_to_dict(task), self.recipe, _nanousd(self.max_budget_usd))

    def _validate_graph_policy(self, graph):
        if self.graph_policy is not None:
            self.graph_policy.validate(graph, default_model=self.model)

    def _validate_graph(self, graph, *, fresh=False, registry=None):
        graph.validate()
        json_snapshot(graph_to_dict(graph))
        self._validate_graph_policy(graph)
        for node in graph.nodes:
            # The journal keys a node's calls by "node/<id>" and its supervision
            # decisions by "supervision/<id>/<generation>", each at most 512
            # characters without control characters. An id it rejects would fail
            # after planning is saved; the length cap leaves room for every key.
            try:
                if type(node.id) is not str or len(node.id) > MAX_NODE_ID_CHARS:
                    raise WorkflowValidationError("Node id is too long for the journal")
                CallKey("execution", f"node/{node.id}")
            except WorkflowValidationError as error:
                shown = str(node.id)
                shown = shown if len(shown) <= 60 else shown[:60] + "..."
                raise WorkflowBindingError(
                    f"Node id {shown!r} cannot key the workflow journal: node ids must be "
                    f"at most {MAX_NODE_ID_CHARS} characters, with no control characters"
                ) from error
            if fresh and (node.status is not NodeStatus.PENDING or node.result is not None):
                raise WorkflowBindingError("A new workflow requires pending nodes without prior results")
            if fresh and any(key in node.metadata for key in (
                "execution_generation", "regenerations_used", "verification_receipt",
                "regeneration_intent", "workflow_supervision", "native_receipts", "response_error",
                "accounting_invalid", "accounting_error", "workflow_accounting_invalid", "cost_usd",
                "cost_usd_unknown", "cost_usd_is_estimate", ADVISORY_VERDICT_KEY,
                REJECTED_GENERATIONS_KEY, CONTEXT_SNAPSHOT_KEY, LINEAGE_REUSE_KEY,
            )):
                raise WorkflowBindingError("A new workflow cannot inherit execution or accounting history")
            if self._context_version and graph.task is None and "task_context" in node.metadata:
                # Snapshots carry a run's Task, never text a graph supplies in its place.
                raise WorkflowBindingError(
                    f"Node {node.id!r} carries task_context metadata without a Task; context "
                    "assembly renders the run's Task, so give the graph a Task instead"
                )
            for name in ("max_retries", "max_regenerations"):
                value = getattr(node, name)
                if type(value) is not int or value < 0:
                    raise WorkflowBindingError(f"Node {name} must be a non-negative integer")
            if node.timeout_s is not None and (
                type(node.timeout_s) not in (int, float)
                or not math.isfinite(node.timeout_s) or node.timeout_s <= 0
            ):
                raise WorkflowBindingError("Node timeout must be finite and positive")
            if node.attach_dep_artifacts or node.metadata.get("attachments") or node.metadata.get("artifacts"):
                raise WorkflowBindingError(
                    f"Durable workflows currently support plain text nodes; node {node.id!r} "
                    "uses attach_dep_artifacts, attachments or artifacts"
                )
            if READ_STEP_KEY in node.metadata:
                self._check_read_step(node)
            validate_workflow_model(self.execution_provider, node.metadata.get("model", self.model))
            if node.agent_id is not None and self.registry.get(node.agent_id) is None and (
                registry is None or registry.get(node.agent_id) is None
            ):
                raise WorkflowBindingError(f"Node {node.id!r} names an unknown agent")

    def _check_read_step(self, node):
        """Check a read step against the profiles the run's recipe declares.

        A generated plan cannot declare one: model output may set no node
        metadata but ``role`` (``smythe.loader``). Developer graphs and
        template builders can, and their steps must name a declared profile
        with valid input.
        """
        step = node.metadata[READ_STEP_KEY]
        if type(step) is not dict or step.keys() != {"profile", "input"} or type(step["profile"]) is not str:
            raise WorkflowBindingError(
                f"Node {node.id!r} {READ_STEP_KEY} must be {{'profile': name, 'input': {{...}}}}")
        profile = self.read_tools.get(step["profile"])
        if profile is None:
            raise WorkflowBindingError(
                f"Node {node.id!r} reads with {step['profile']!r}, which this run's read_tools do "
                "not declare"
            )
        try:
            profile.request(step["input"])
        except WorkflowBindingError as error:
            raise WorkflowBindingError(f"Node {node.id!r} read step: {error}") from error
        if node.verifies is not None:
            raise WorkflowBindingError(f"Node {node.id!r} is a read step and cannot verify another node")

    def _check_plan(self, graph, registry):
        """Apply _plan's checks to a candidate plan without adopting it.

        A planner calls this before it returns, so the plan's agents are found
        in its own registry rather than registered with the run.
        """
        registry.workflow_description()
        self._validate_graph(graph, fresh=True, registry=registry)

    def _check_planning(self):
        """Refuse, before routing or planning, a strict run no gate of its plan could accept."""
        from smythe.swarm import _check_gate_planning, _planning_architects

        _check_gate_planning(self.acceptance, _planning_architects(
            self.sources["router"], self.sources["architect"]))

    def _check_obligations(self, graph):
        """Refuse, before any node runs, a graph whose output the strict policy could not decide."""
        if self.acceptance is not None:
            from smythe.swarm import _synthesis_strategy

            graph.validate()
            validate_acceptance_obligations(
                graph, self.acceptance, strategy=_synthesis_strategy(self.sources["synthesizer"]),
            )

    def _prepare_graph(self, graph, *, fresh=False):
        self._validate_graph(graph, fresh=fresh)
        for node in graph.nodes:
            node.metadata.setdefault("model", self.model)
        self.registry.assign(graph)
        self.registry.workflow_description()
        graph.run_ref = self._reference()
        if graph.task is not None:
            graph.task = snapshot_task(graph.task)
            ExecutorBase.stamp_task_context(graph.nodes, graph.task)
        self.graph = graph
        return graph

    def _save(self, status="running", *, output=None, acceptance=None):
        """Checkpoint the run; *acceptance* is the decision a completed one consumes."""
        if self.graph is None:
            return
        self._validate_graph_policy(self.graph)
        self.registry.workflow_description()
        generations = {node.id: node_generation(node) for node in self.graph.nodes}
        # A call claims its dispatch right after its reservation, without
        # yielding, so a reserved call seen here was left by a killed process
        # or a failed claim. Nodes call only at their current generation, so
        # one of a superseded generation, including one that a regeneration
        # applied on resume superseded, is never sent: release it. A completed
        # run sends nothing more, so it releases every reservation left.
        self.store.release_superseded_calls(self.context.lease, generations)
        if status == "completed":
            self.store.release_reserved_calls(self.context.lease, "run_completed")
        calls = []
        for node in self.graph.nodes:
            if node.status in {NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED}:
                calls.extend(self._call_ids(self._worker_keys.get((node.id, node_generation(node)), ())))
        # Regeneration can discard an accepted descendant during cancellation
        # after the previous batched snapshot. Consume that settled call with
        # the checkpoint that durably advances its node generation. The query
        # reads only those calls, so a save's cost does not grow with history.
        calls.extend(self.store.superseded_node_calls(self.run_id, generations))
        checkpoint = {
            "version": self.checkpoint_version, "status": status, "graph": graph_to_dict(self.graph),
            "registry": self.registry.workflow_description(), "output": output,
            "revisions_used": self.executor.revisions_used if self.executor else self._revisions_used,
            "max_concurrency": self.execution_concurrency,
        }
        if self.checkpoint_version == 2:
            record = acceptance or self.undecided
            checkpoint["delivery_acceptance"] = None if record is None else record.to_dict()
        if self._context_version:
            if self.context_revision is None:
                self._capture_sources()
            # The source set by digest; its text stays in its journal operation.
            checkpoint["context_sources"] = {"revision": self.context_revision,
                                             "sha256": sources_sha256(self.context_sources)}
        saved = self.store.save_checkpoint(
            self.context.lease, self.revision, checkpoint, consumed_call_ids=calls,
            consumed_operation_ids=sorted(self._pending_operations), transition_kind=status,
        )
        self.revision = saved["revision"]
        self._pending_operations.clear()

    def _node_update(self, node):
        self._clear_resolved_native_error(node)
        stamp = node.metadata.get(CONTEXT_SNAPSHOT_KEY)
        if (node.status is NodeStatus.COMPLETED and isinstance(stamp, dict)
                and stamp.get("generation") == node_generation(node)):
            # The calls behind the result, which its dependents' snapshots name.
            stamp["call_ids"] = self._call_ids(
                self._worker_keys.get((node.id, node_generation(node)), ()))
        self._updates += 1
        if self._updates >= self.checkpoint_every_n_nodes:
            self._updates = 0
            self._save()

    def _clear_resolved_native_error(self, node):
        # A completed node, or one skipped after zero-cost rejections, keeps
        # no error marker once every journal call it made is resolved. A
        # completed run requires every node to be free of these markers.
        if (node.status not in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
                or "response_error" not in node.metadata
                or "workflow_accounting_invalid" in node.metadata):
            return
        records = {call["call_id"]: call for call in self.store.inspect_run(self.run_id)["calls"]}
        ids = set()
        for entry in node.metadata.get("native_receipts", []):
            receipt = entry.get("receipt", {})
            if receipt.get("workflow_run_id") == self.run_id:
                ids.add(receipt.get("workflow_call_id"))
        ids.update(call["call_id"] for call in records.values()
                   if call["key"]["scope_id"] == f"node/{node.id}"
                   and call["key"]["generation"] == node_generation(node))

        def resolved(call_id):
            call = records.get(call_id)
            if call is None or call["billing_state"] != "known":
                return False
            # A zero-cost provider rejection needs no reconciliation.
            return call["result_state"] in {"accepted", "applied"} or (
                call["result_state"] == "rejected"
                and _is_http_rejection(self.store.lookup_call(self.run_id, CallKey(**call["key"]))))

        if ids and all(resolved(key) for key in ids):
            for key in ("response_error", "accounting_invalid", "accounting_error", "cost_usd_unknown"):
                node.metadata.pop(key, None)

    async def _operation_result(self, key, kind, inputs, function):
        operation = self.store.load_operation(self.run_id, key)
        if operation is None:
            operation = self.store.begin_operation(self.context.lease, key, kind, inputs)
        if operation["state"] == "started":
            token = self._operation.set(key)
            try:
                result = await function(operation["inputs"])
            finally:
                self._operation.reset(token)
            operation = self.store.complete_operation(
                self.context.lease, key, result,
                consumed_call_ids=self._call_ids(self._operation_keys.get(key, ())),
            )
        self._pending_operations.add(operation["operation_id"])
        return operation["result"]

    def _source_operation(self, revision, sources=None):
        """The source set operation at *revision*, begun with *sources* when there is none."""
        key = sources_key(revision)
        operation = self.store.load_operation(self.run_id, key)
        if operation is None and sources is not None:
            operation = self.store.begin_operation(self.context.lease, key, "context", {
                "revision": revision, "sources": [source.to_dict() for source in sources]})
        return operation

    def _read_sources(self, revision, operation):
        try:
            return sources_from_operation(operation, revision)
        except (TypeError, ValueError) as error:
            raise WorkflowStateError(f"Saved context source revision {revision} is invalid") from error

    def _bind_sources(self, revision, operation):
        """Adopt a source set operation as the run's sources; the next checkpoint consumes it."""
        recorded = self._read_sources(revision, operation)
        if operation["state"] == "started":
            operation = self.store.complete_operation(
                self.context.lease, sources_key(revision), {"sha256": sources_sha256(recorded)})
        self.context_revision, self.context_sources = revision, recorded
        self._pending_operations.add(operation["operation_id"])

    def _capture_sources(self):
        """Bind revision 0: the recorded one, or else the sources this runtime was given."""
        operation = self._source_operation(0, self._supplied_sources)
        if operation is None:
            raise WorkflowStateError(
                "Workflow never recorded its context sources; resume it with the "
                "context_sources it started with"
            )
        self._bind_sources(0, operation)

    def _restore_sources(self, record):
        if (not isinstance(record, dict) or record.keys() != {"revision", "sha256"}
                or type(record["revision"]) is not int or record["revision"] < 0):
            raise WorkflowStateError("Workflow checkpoint has no valid context sources record")
        operation = self._source_operation(record["revision"])
        if operation is None:
            raise WorkflowStateError("Workflow checkpoint names context sources the journal lacks")
        self._bind_sources(record["revision"], operation)
        if sources_sha256(self.context_sources) != record["sha256"]:
            raise WorkflowStateError("Workflow checkpoint context sources do not match the journal")

    async def _context_snapshot(self, node, graph, dep_results):
        """The node's snapshot for its generation, recorded once as a ``context`` operation.

        A node that already has one, including a call resumed after a crash
        or a retried attempt, gets the snapshot the journal kept: the run's
        sources and the dependency outputs it names are never read again.
        """
        generation = node_generation(node)
        key = context_key(node.id, generation)
        operation = self.store.load_operation(self.run_id, key)
        if operation is None:
            assembled = self._assemble_snapshot(node, graph, dep_results, generation)
            operation = self.store.begin_operation(self.context.lease, key, "context",
                                                   {"snapshot": assembled.to_dict()})
        try:
            snapshot = ContextSnapshot.from_dict(operation["inputs"].get("snapshot"))
        except (TypeError, ValueError) as error:
            raise WorkflowStateError(f"Saved context snapshot of node {node.id!r} is invalid") from error
        if operation["state"] == "started":
            operation = self.store.complete_operation(self.context.lease, key,
                                                      {"snapshot_id": snapshot.snapshot_id})
        if (operation["kind"] != "context"
                or operation["result"] != {"snapshot_id": snapshot.snapshot_id}
                or (snapshot.node_id, snapshot.generation) != (node.id, generation)
                or snapshot.context_assembly != self._context_version
                or snapshot.sources_revision != self.context_revision):
            raise WorkflowStateError(
                f"Saved context snapshot does not belong to node {node.id!r} at generation "
                f"{generation} of this run"
            )
        self._pending_operations.add(operation["operation_id"])
        self._snapshot_ids[node.id, generation] = snapshot.snapshot_id
        node.metadata[CONTEXT_SNAPSHOT_KEY] = {"generation": generation,
                                               "snapshot_id": snapshot.snapshot_id}
        return snapshot

    def _assemble_snapshot(self, node, graph, dep_results, generation):
        task = graph.task
        dependencies = []
        for dep_id, value in dep_results.items():
            dep = self.executor._cached_node_by_id(dep_id, graph)
            stamp = dep.metadata.get(CONTEXT_SNAPSHOT_KEY)
            calls = (stamp.get("call_ids", []) if isinstance(stamp, dict)
                     and stamp.get("generation") == node_generation(dep) else [])
            dependencies.append(DependencyDigest(
                dep_id, node_generation(dep), dep.status.value,
                output_sha256(value if isinstance(value, str) else str(value)), tuple(calls),
            ))
        agent = self.registry.get(node.agent_id) if node.agent_id else None
        return ContextSnapshot(
            node_id=node.id, generation=generation, step=node.label,
            goal=None if task is None else task.goal,
            constraints=() if task is None else task.constraints,
            acceptance_criteria=() if task is None else task.done_when,
            task_sha256=task_sha256(task), sources_revision=self.context_revision,
            sources=self.context_sources, dependencies=tuple(dependencies),
            agent_id=node.agent_id,
            capabilities=() if agent is None else sorted(set(agent.profile.capabilities)),
            context_assembly=self._context_version,
        )

    async def _plan(self, task):
        async def build(inputs):
            frozen = task_from_dict(inputs["task"])
            architect, routing = ((await self.bound["router"].adecide(snapshot_task(frozen)))
                                  if self.bound["router"] else (self.bound["architect"], None))
            graph, registry = await architect.aplan(snapshot_task(frozen))
            self._validate_graph_policy(graph)
            registry.workflow_description()
            for agent in registry.list_agents():
                self.registry.register(agent)
            graph.task = snapshot_task(frozen)
            if self.acceptance is not None:
                # The saved plan carries the marks, so a replay keeps them.
                from smythe.swarm import _mark_generated_gates

                _mark_generated_gates(graph, architect)
            if routing is not None:
                # The run writes only this key; the architect may have set others.
                graph.planning = {**(graph.planning or {}), "routing": routing.to_dict()}
            self._prepare_graph(graph, fresh=True)
            if routing is not None:
                # Agents are assigned now, so the advisory lint can compare them.
                from smythe.planner import _record_coordination

                _record_coordination(graph, self.registry)
            result = {"graph": graph_to_dict(graph), "registry": self.registry.workflow_description()}
            if routing is not None:
                result["routing"] = routing.to_dict()
            return result

        result = await self._operation_result("planning", "planning", {"task": task_to_dict(task)}, build)
        registry = Registry.from_workflow_description(result["registry"])
        # The graph snapshot carries its planning records, so a replay restores
        # them all; a result saved before routing records has none.
        graph = graph_from_dict(result["graph"])
        # Completed planning operations replay without invoking build(). Apply
        # the same graph policy before adopting or checkpointing that result.
        self._validate_graph_policy(graph)
        self.registry = registry
        self._prepare_graph(graph, fresh=True)
        self._save("planned")
        return self.graph

    async def plan(self, task):
        task = snapshot_task(task)
        async with self._session(self._new_run(task)):
            return await self._plan(task)

    async def execute(self, value, *, max_concurrency):
        if isinstance(value, Task):
            task = snapshot_task(value)
            self._check_planning()
            async with self._session(self._new_run(task)):
                await self._plan(task)
                return await self._execute(max_concurrency)
        if not isinstance(value, ExecutionGraph):
            raise TypeError("Expected Task or ExecutionGraph")
        # Capture caller-owned input before binding or waiting for a lease.
        for node in value.nodes:
            json_snapshot(node.metadata)
        value = graph_from_dict(json_snapshot(graph_to_dict(value)))
        reference = snapshot_run_ref(value.run_ref)
        if reference is None:
            # A refused graph creates no run.
            self._check_obligations(value)
            async with self._session(self._new_run(value.task)):
                self._prepare_graph(value, fresh=True)
                self._save("planned")
                return await self._execute(max_concurrency)
        if reference["store_id"] != self.store.store_id:
            raise WorkflowConflictError("Graph belongs to a different workflow store")
        run = self.store.load_run(reference["run_id"])
        if (reference["recipe_sha256"] != run["config_sha256"]
                or _canonical(task_to_dict(value.task)) != _canonical(run["task"])):
            raise WorkflowConflictError("Graph Task or recipe differs from its durable run")
        async with self._session(run):
            state = self._restore()
            if state["status"] == "planned":
                self._prepare_graph(value, fresh=True)
                self._save("planned")
            elif _canonical(graph_to_dict(value)) != _canonical(graph_to_dict(self.graph)):
                raise WorkflowConflictError("Graph has progressed; resume its execution_id")
            return await self._execute(max_concurrency, state=state)

    def _restore(self):
        saved = self.store.get_checkpoint(self.run_id)
        if saved is None:
            return None
        self.revision = saved["revision"]
        state = saved["checkpoint"]
        version = state.get("version")
        if type(version) is not int or version not in (1, 2):
            raise WorkflowStateError("Unsupported workflow checkpoint version")
        if version != self.checkpoint_version:
            raise WorkflowStateError(
                f"Workflow checkpoint version {version} does not match its recipe, which "
                f"writes version {self.checkpoint_version}"
            )
        if version == 2:
            self.saved_acceptance = self._saved_acceptance(state)
        if self._context_version:
            self._restore_sources(state.get("context_sources"))
        self._revisions_used = state.get("revisions_used", 0)
        registry = Registry.from_workflow_description(state["registry"])
        graph = graph_from_dict(state["graph"])
        if (graph.run_ref != self._reference()
                or _canonical(task_to_dict(graph.task)) != _canonical(self.run["task"])):
            raise WorkflowConflictError("Checkpoint graph does not match its durable run")
        self._validate_graph_policy(graph)
        self.registry = registry
        self._prepare_graph(graph)
        return state

    def _saved_acceptance(self, state):
        """A version 2 checkpoint's record: undecided until completed, then the decision.

        A completed run's decision may be a re-evaluation's, under its own policy.
        """
        if self.run_policy_sha256 is None:
            return _saved_record(state, None, None)
        return _read_saved_record(self.store, self.run_id, state, self.run_policy_sha256,
                                  self.undecided.task_sha256)

    async def resume(self, run_id, *, max_concurrency):
        async with self._session(self.store.load_run(run_id)):
            state = self._restore()
            if state is None:
                task = task_from_dict(self.run["task"])
                if task is None:
                    raise WorkflowStateError("Caller-built workflow has no saved initial graph")
                self._check_planning()
                await self._plan(task)
            return await self._execute(max_concurrency, state=state)

    async def refresh_context(self, run_id, sources):
        """Give an unfinished run new sources, to be re-evaluated when it resumes.

        Records *sources* (a sequence of :class:`~smythe.context.ContextSource`)
        as the run's next source revision and checkpoints every node pending
        at its next generation, so that resuming runs each node again with a
        new snapshot. Nothing is called now, and earlier snapshots, calls and
        charges stay in the journal. A run whose checkpoint is still
        ``planned`` has run nothing and only adopts the revision. Sources
        equal to the current ones change nothing. Returns the source revision
        the run now uses.

        Raises ``WorkflowStateError`` for a run without context assembly, a
        completed run, a run with no saved graph, and a node whose provider
        accounting is unresolved.
        """
        sources = snapshot_sources(sources)
        async with self._session(self.store.load_run(run_id)):
            if not self._context_version:
                raise WorkflowStateError("Workflow does not use context assembly; it has no sources")
            state = self._restore()
            if state is None:
                raise WorkflowStateError("Workflow has no saved graph yet; resume it first")
            if state["status"] == "completed":
                raise WorkflowStateError(
                    "A completed workflow keeps the sources it delivered with; start a new run "
                    "to use other sources"
                )
            if sources == self.context_sources:
                return self.context_revision
            self._refuse_removed(sources, "give the run other text for it")
            for node in self.graph.nodes:
                self._clear_resolved_native_error(node)
                if (any(key in node.metadata for key in ("response_error", "workflow_accounting_invalid"))
                        or node.metadata.get("accounting_invalid")):
                    raise WorkflowStateError(
                        f"Node {node.id!r} has unresolved provider accounting; resume the run to "
                        "settle it before refreshing its sources"
                    )
            # A refresh that stopped before its checkpoint may have recorded
            # other sources at the next revision; they were never used.
            revision = self.context_revision + 1
            while ((operation := self._source_operation(revision)) is not None
                   and self._read_sources(revision, operation) != sources):
                revision += 1
            self._bind_sources(revision, operation or self._source_operation(revision, sources))
            status = state["status"]
            if status != "planned":
                status = "running"
                for node in self.graph.nodes:
                    node.metadata["execution_generation"] = node_generation(node) + 1
                    node.status, node.result = NodeStatus.PENDING, None
                    for key in _GENERATION_KEYS:
                        node.metadata.pop(key, None)
            self.execution_concurrency = state.get("max_concurrency")
            self._save(status)
            return revision

    async def remove_source(self, run_id, source_id, reason, *, removed_at=None):
        """Remove one of a durable run's current sources; return its tombstone.

        The tombstone, a :class:`~smythe.evidence.SourceRecord`, keeps the
        source's id, version and digest, *reason* and *removed_at* (by
        default the current UTC time), and nothing of its text. It is
        journaled as a ``source_tombstone`` operation keyed
        :func:`~smythe.evidence.source_tombstone_key`, which a new checkpoint
        applies; nothing is called. From then on the run never uses the
        source's text again, and reading it from a cache is not a substitute:

        - ``load_context_sources`` and ``load_context_snapshot`` raise
          ``SourceRemovedError`` for a record that holds it, and
          ``load_source_evidence`` marks it ``removed`` and never returns it;
        - an unfinished run does not resume until :meth:`refresh_context`
          gives it other sources, which may not include the removed text;
        - a completed run's decision, made before the removal, no longer
          holds: resuming or inspecting the run gives ``indeterminate``, and
          :meth:`reevaluate` decides again, under a source evidence
          validator that holds the tombstone rather than the text.

        Earlier records keep what they hold: the journal's requests, read
        responses, snapshots and outputs, and a published document. Removing
        them is not supported yet. Removing a removed source again returns
        its first tombstone.

        Raises ``WorkflowBindingError`` for a reason that is not 1-500
        characters without control characters, and ``WorkflowStateError`` for
        a run without context assembly, a run that has recorded no sources,
        and an id that is not one of the run's current sources.
        """
        if (type(reason) is not str or not reason.strip() or len(reason) > 500
                or any(ord(char) < 32 or ord(char) == 127 for char in reason)):
            raise WorkflowBindingError(
                "reason must be 1-500 characters of text without control characters")
        async with self._session(self.store.load_run(run_id)):
            if not self._context_version:
                raise WorkflowStateError("Workflow does not use context assembly; it has no sources")
            state = self._restore()
            if state is None:
                raise WorkflowStateError("Workflow has recorded no sources yet")
            source = next((item for item in self.context_sources if item.source_id == source_id),
                          None)
            if source is None:
                raise WorkflowStateError(f"Source {source_id!r} is not one of the run's current sources")
            key = source_tombstone_key(source.source_id, source.sha256)
            operation = self.store.load_operation(self.run_id, key)
            if operation is None:
                stamp = removed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                tombstone = SourceRecord.from_context_source(source).tombstoned(reason, removed_at=stamp)
                operation = self.store.begin_operation(self.context.lease, key, SOURCE_TOMBSTONE_OPERATION,
                                                       {"tombstone": tombstone.to_dict()})
            if operation["state"] == "started":
                # A removal that stopped before completing is finished as it began.
                operation = self.store.complete_operation(self.context.lease, key, operation["inputs"])
            tombstone = SourceRecord.from_dict(operation["result"]["tombstone"])
            if operation["state"] != "applied":
                self._pending_operations.add(operation["operation_id"])
                self.execution_concurrency = state.get("max_concurrency")
                if state["status"] == "completed":
                    self._save("completed", output=state["output"], acceptance=self.saved_acceptance)
                else:
                    self._save(state["status"])
            return tombstone

    def _refuse_removed(self, sources, advice):
        """Refuse *sources* that hold the text of a source removed from the run."""
        removed = _removed_sources(self.store, self.run_id)
        for source in sources:
            tombstone = removed.get((source.source_id, source.sha256))
            if tombstone is not None:
                raise WorkflowStateError(
                    f"Source {source.source_id!r} was removed from this run "
                    f"({tombstone.tombstone}); {advice}"
                )

    async def reevaluate(self, run_id, *, operator=None):
        """Record a new delivery decision for a completed run's saved output.

        This runtime's strict policy decides: its final validators run on the
        output the run saved, with no provider call, and every gate's recorded
        final verdict is read. Nothing else changes: no node runs, and the
        run's calls, charges, budget, revision and regeneration counters,
        node generations and recipe stay as they are. The decision is a new
        ``acceptance`` operation whose key carries the next decision sequence,
        so identical bytes under an unchanged policy still get a new decision.
        A new completed checkpoint consumes it, which makes it the run's
        :func:`current_decision`; earlier decisions stay in the journal
        (:func:`decision_history`). A re-evaluation stopped before that
        checkpoint is finished by the next one under the same policy, unless
        a source was removed from the run after it decided: the next one then
        decides again, since a decision binds the run's tombstones.

        The saved recipe may differ from this runtime's only in its acceptance
        policy and gate verifier. Resuming the run still needs the policy its
        recipe records, and returns the current decision.

        *operator*, an :class:`~smythe.acceptance.OperatorDecision`, records
        that operator's decision instead, with or without a policy on this
        runtime: a version 2 record that keeps what the current decision binds
        and names the operator, in a new ``acceptance`` operation whose inputs
        also bind the decision and operator. Only a current ``needs_review``
        or ``indeterminate`` decision can be resolved; any other stands.

        Raises ``WorkflowBindingError`` without a strict policy or *operator*;
        ``WorkflowStateError`` for a run whose recipe has no strict policy, an
        unfinished run, unresolved accounting, a saved output, task, source
        generation or record that no longer matches the journal, so changed
        output is never accepted, a run with a renderer whose published
        document is missing or changed (:meth:`republish` publishes it again
        first), or a current decision an operator cannot resolve;
        ``WorkflowConflictError`` for a recipe that differs in anything else;
        and whatever the operator's verifier raises, such as
        ``ApprovalRejected``, before anything is journaled.
        """
        from smythe.swarm import _delivery_sources

        if self.acceptance is None and operator is None:
            raise WorkflowBindingError("reevaluate needs a strict acceptance policy")
        run = self.store.load_run(run_id)
        if "acceptance" not in run["config"]:
            raise WorkflowStateError(
                "Workflow ran without a strict acceptance policy, so it has no delivery decision "
                "to re-evaluate; run the task again under one"
            )
        # A runtime without a policy writes the checkpoint version the recipe does.
        self.checkpoint_version = _checkpoint_version(run["config"])
        async with self._session(run, reevaluating=True):
            state = self._restore()
            if state is None or state["status"] != "completed":
                raise WorkflowStateError(
                    "Only a completed workflow can be re-evaluated; resume it to finish it"
                )
            self._check_completed()
            output, record = state["output"], self.saved_acceptance
            if not isinstance(output, str):
                raise WorkflowStateError("Completed workflow output is not text")
            # A source removed after the decision is why a run is re-evaluated.
            problems = _decision_problems(
                self.store, self.run_id, self.graph, output, record,
                task=task_from_dict(self.run["task"]), policy_sha256=self.run_policy_sha256,
                removed=False,
            )
            sources, used_fallback = _delivery_sources(self.graph, record.strategy)
            if ([AcceptanceSource(node.id, node_generation(node)) for node in sources]
                    != list(record.sources) or used_fallback != record.used_fallback):
                problems.append("the saved graph no longer delivers the sources its record bound")
            if problems:
                raise WorkflowStateError("Workflow cannot be re-evaluated: " + "; ".join(problems))
            if self.renderer is not None:
                # A decision covers the document only while its published file holds it.
                damaged = artifact_problems(current_artifact(self.store, self.run_id), output)
                if damaged:
                    raise WorkflowStateError(
                        "Workflow cannot be re-evaluated: " + "; ".join(damaged)
                        + "; publish its document again with republish() first"
                    )
            if operator is None:
                decided = await self._decide(self._synthesis_key(), output, reevaluation=True)
            else:
                decided = await self._operator_decide(operator, record)
            self.execution_concurrency = state.get("max_concurrency")
            self._save("completed", output=output, acceptance=decided)
            self.saved_acceptance = decided
            return self._result(output, decided)

    def _check_completed(self):
        """Refuse a completed run that still holds unresolved control or accounting."""
        # A verdict's forced checkpoint can precede its ordinary node callback.
        # Resolve proven native replay markers before consuming that control.
        for node in self.graph.nodes:
            self._clear_resolved_native_error(node)
        validate_verification_checkpoint(self.graph, version=3, completed=True)
        if verification_pending(self.graph) or any(
            any(key in node.metadata for key in ("workflow_accounting_invalid", "response_error"))
            or node.metadata.get("accounting_invalid") for node in self.graph.nodes
        ):
            raise WorkflowStateError(
                "Completed workflow has an unresolved control or accounting marker"
            )

    async def _execute(self, max_concurrency, *, state=None):
        from smythe.async_executor import AsyncExecutor

        if state and state["status"] == "completed":
            self._check_completed()
            return self._result(state["output"], self._reverified(state["output"]))
        if self._context_version and self.context_revision is not None:
            # A removed source's text is never sent again, from a snapshot or the sources.
            self._refuse_removed(self.context_sources, "give the run other sources with "
                                 "refresh_context before resuming it")
        # A verdict's forced checkpoint can precede its ordinary node callback.
        # Resolve proven native replay markers before consuming that control.
        for node in self.graph.nodes:
            self._clear_resolved_native_error(node)
        if state is None or state["status"] == "planned":
            # No node has run, so a plan, edited or generated, is checked
            # now. A run that started met its obligations then.
            self._check_obligations(self.graph)
        if self.renderer is not None:
            # Refuse a directory the run cannot publish into before any call.
            self._claim_artifact_directory()
        self.execution_concurrency = (state.get("max_concurrency") if state else None) or max_concurrency
        if (type(self.execution_concurrency) is not int
                or not 1 <= self.execution_concurrency <= self.max_concurrency):
            raise WorkflowBindingError("Execution concurrency exceeds the saved workflow policy")
        validate_verification_checkpoint(self.graph, version=3, completed=False)
        reset_incomplete_nodes(self.graph)
        self.executor = AsyncExecutor(
            provider=self.execution_provider, registry=self.registry, tracer=self.trace,
            budget=self.budget, max_concurrency=self.execution_concurrency, artifact_dir=None,
            retry_backoff_s=self.retry_backoff_s, task=self.graph.task,
            verifier=self.gate_verifier, max_revisions=self.max_revisions,
            revisions_used=state["revisions_used"] if state else 0,
            supervisor=_WorkflowSupervisor(self) if self.bound["supervisor"] else None,
            provider_call_factory=self._worker_call, on_node_update=self._node_update,
            on_control_update=self._save, on_supervision_update=self._supervision_update,
            # The version the run's recipe records; 1 when it records none.
            prompt_assembly=self.behavior.get("prompt_assembly", 1),
            acceptance=self.acceptance, read_call_factory=self._read_call,
            context_assembler=self._context_snapshot if self._context_version else None,
        )
        self._save()
        try:
            # A crash can follow node acceptance but precede its review.
            self.executor.prepare_execution(self.graph)
            self.executor.recover_verification(self.graph)
            for node in tuple(self.graph.nodes):
                disposition = node.metadata.get("workflow_supervision", {})
                if (node.status is NodeStatus.COMPLETED
                        and (disposition.get("generation") != node_generation(node)
                             or disposition.get("state") != "applied")):
                    await self.executor.maybe_revise(node, self.graph)
            await self.executor.run(self.graph)

            async def synthesize(inputs):
                graph = graph_from_dict(inputs["graph"])
                output = await self.bound["synthesizer"].asynthesize(graph)
                if type(output) is not str:
                    raise WorkflowStateError("Synthesis must return text")
                return {"output": output}

            key = self._synthesis_key()
            final = await self._operation_result(
                key, "synthesis", {"graph": graph_to_dict(self.graph)}, synthesize,
            )
            output = final["output"]
            if self.renderer is not None:
                # The published document is the output delivered and decided.
                output = await self._publish_output(key, output)
            await self._record_source_evidence(output)
            acceptance = None if self.acceptance is None else await self._decide(key, output)
            # The completed checkpoint consumes the synthesis and the decision together.
            self._save("completed", output=output, acceptance=acceptance)
        except BaseException:
            self._save("failed")
            raise
        if self.renderer is not None and acceptance is not None:
            # As on completed resume, delivery needs the published file
            # intact, including one a lost process published before.
            acceptance = self._held(acceptance, output, artifact_problems(
                current_artifact(self.store, self.run_id), output))
        return self._result(output, acceptance)

    def _artifact_directory(self):
        """The bound run's own directory, ``<artifact_dir>/<run id>``."""
        return self.artifact_dir / self.run_id

    def _claim_artifact_directory(self):
        """Bind the run's directory to the run, and check it can publish without replacing."""
        claim_directory(self._artifact_directory(), owner_bytes(self.store.store_id, self.run_id))

    async def _publication(self, inputs_at, data_for, record_for):
        """Run the publication whose inputs are ``inputs_at(sequence)``; return its record.

        The sequence is that of the operation that already has those inputs,
        which then replays, or else the next free one. ``data_for()`` gives
        the bytes to publish, which ``record_for(sequence, name, data)``
        records once they are published. The run's directory must already be
        claimed in this session.
        """
        saved = publications(self.store, self.run_id)
        sequence = next((number for number, operation in enumerate(saved)
                         if operation["inputs"] == inputs_at(number)), len(saved))
        inputs = inputs_at(sequence)

        async def publish(_):
            data = data_for()
            name = publish_bytes(self._artifact_directory(), self.renderer_filename, data)
            return {"artifact": record_for(sequence, name, data).to_dict()}

        result = await self._operation_result(publication_key(sequence), PUBLICATION_OPERATION,
                                              inputs, publish)
        return read_publication({"kind": PUBLICATION_OPERATION, "inputs": inputs, "result": result},
                                sequence)

    async def _publish_output(self, synthesis_key, text):
        """Render the synthesized *text* and publish the document; return the document's text.

        A ``publication`` operation keyed ``publication/<sequence>`` records
        the file, bound to the synthesis it rendered. A process lost after
        publishing reuses the file it published; one lost after the operation
        committed renders the document again and checks that it has the
        recorded SHA-256, or else reads the published file. The text
        returned is exactly the published bytes; whether the file still holds
        them is checked when the run's result is built.
        """
        rendered = []

        def inputs_at(sequence):
            return {"sequence": sequence, "synthesis": synthesis_key,
                    "output_sha256": text_sha256(text), "renderer": self.recipe["renderer"]}

        def render():
            rendered.append(render_document(self.bound_renderer, text, self.graph.task))
            return rendered[-1]

        def record_for(sequence, name, data):
            return PublishedArtifact(
                sequence=sequence, directory=str(self._artifact_directory()), name=name,
                sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                media_type=self.renderer_media_type, renderer=self.recipe["renderer"],
                synthesis=synthesis_key, output_sha256=text_sha256(text),
            )

        record = await self._publication(inputs_at, render, record_for)
        data = rendered[-1] if rendered else render()
        if hashlib.sha256(data).hexdigest() != record.sha256:
            data = read_published(record)
            if data is None:
                raise WorkflowStateError(
                    f"The published document {record.name!r} is missing or changed, and the "
                    "renderer no longer makes its bytes from the saved synthesis"
                )
        return data.decode("utf-8")

    async def republish(self, run_id):
        """Publish a completed run's document again, as a new file; return the current record.

        When the current publication's file is missing or changed, the saved
        document's bytes, which must still have the published SHA-256, are
        published under the renderer's file name or the next free numbered
        name in the run's directory, never replacing a file, so a changed
        file stays as it is. The next ``publication`` operation records it and
        a new completed checkpoint applies it, which makes it current (see
        :func:`~smythe.workflow_artifact.current_artifact`). The delivery
        decision is unchanged: it binds the same bytes. Nothing is rendered,
        validated or called. A run whose current file still holds its bytes
        is left as it is, and its current record is returned.

        Raises ``WorkflowStateError`` for a run without a renderer, a run that
        has not completed or published, and a saved output that is not the
        published document.
        """
        async with self._session(self.store.load_run(run_id)):
            if self.renderer is None:
                raise WorkflowStateError("Workflow has no renderer, so it publishes no artifact")
            state = self._restore()
            if state is None or state["status"] != "completed":
                raise WorkflowStateError("Only a completed workflow can publish its document again")
            current, output = current_artifact(self.store, self.run_id), state["output"]
            if current is None:
                raise WorkflowStateError("Workflow has no published document to publish again")
            previous = current.publication
            if not isinstance(output, str) or text_sha256(output) != previous.sha256:
                raise WorkflowStateError("The saved output is not the published document")
            if current.integrity["status"] == "verified":
                return previous

            def inputs_at(sequence):
                return {"sequence": sequence, "republishes": previous.sequence,
                        "sha256": previous.sha256, "size_bytes": previous.size_bytes}

            def record_for(sequence, name, data):
                return PublishedArtifact(
                    sequence=sequence, directory=str(self._artifact_directory()), name=name,
                    sha256=previous.sha256, size_bytes=previous.size_bytes,
                    media_type=previous.media_type, renderer=previous.renderer,
                    synthesis=previous.synthesis, output_sha256=previous.output_sha256,
                )

            self._claim_artifact_directory()
            record = await self._publication(inputs_at, lambda: output.encode("utf-8"), record_for)
            self.execution_concurrency = state.get("max_concurrency")
            self._save("completed", output=output, acceptance=self.saved_acceptance)
            return record

    def _synthesis_key(self):
        """The synthesis operation's key for this run (see :func:`_synthesis_key`)."""
        return _synthesis_key(
            self.graph, scoped=self.checkpoint_version == 2 or bool(self._context_version),
        )

    async def _record_source_evidence(self, output):
        """Journal the source evidence *output* carries, when the strict policy checks it.

        A ``source_evidence`` operation, keyed by the output's digest and the
        run's source set, records the evidence read from the output, or why
        there is none. The completed checkpoint consumes it with the decision.
        A replay reads the recorded result; the evidence never decides
        acceptance, which the policy's validators do.
        """
        if self.acceptance is None or not any(
                isinstance(validator, SourceEvidenceValidator)
                for validator in self.acceptance.final_validators):
            return
        sources = None
        if self._context_version:
            sources = {"revision": self.context_revision,
                       "sha256": sources_sha256(self.context_sources)}
        inputs = source_evidence_inputs(output, sources)

        async def read(_):
            return source_evidence_result(output)

        await self._operation_result(source_evidence_key(inputs), SOURCE_EVIDENCE_OPERATION,
                                     inputs, read)

    def _unusable_evidence_validators(self):
        """Why each source evidence validator of the strict policy cannot check this run, by identity.

        Such a validator must check exactly the run's current source set:
        the same source ids, each with the same version and digest, and no
        text for a source removed from the run. One that checks other
        sources, such as the sources of an earlier revision, could accept
        evidence for sources the run's nodes were never given, so its
        outcome is ``error`` and the run is never accepted. A run without
        context assembly records no source set, and its validators check the
        sources they were given.
        """
        if not self._context_version:
            return {}
        current = {source.source_id: (source.version, source.sha256)
                   for source in self.context_sources}
        removed = _removed_sources(self.store, self.run_id)
        unusable = {}
        for validator, identity in zip(self.acceptance.final_validators,
                                       self.acceptance.validator_identities):
            if not isinstance(validator, SourceEvidenceValidator):
                continue
            checked = {record.source_id: (record.source_version, record.sha256)
                       for record in validator.sources}
            differences = [
                f"source {source_id!r} is not one of the run's sources" if source_id not in current
                else f"it lacks the run's source {source_id!r}" if source_id not in checked
                else f"source {source_id!r} has another version or digest"
                for source_id in sorted(current.keys() | checked.keys())
                if current.get(source_id) != checked.get(source_id)
            ] + [f"it holds the text of the removed source {record.source_id!r}"
                 for record in validator.sources
                 if record.text is not None and (record.source_id, record.sha256) in removed]
            if differences:
                unusable[identity] = (
                    f"final validator {identity.name!r} checks other sources than the run's "
                    f"current source set (revision {self.context_revision}): "
                    + "; ".join(differences[:3])
                )
        return unusable

    def _acceptance_inputs(self, synthesis_key, output_sha256, strategy, sources, used_fallback,
                           sequence=0, tombstones=()):
        """Everything a decision binds; its hash keys the acceptance operation."""
        return _decision_inputs(synthesis_key, self.acceptance.sha256, self.undecided.task_sha256,
                                output_sha256, strategy, sources, used_fallback, sequence,
                                tombstones)

    def _next_decision_sequence(self, inputs_at):
        """A re-evaluation's decision sequence: one past every decision the journal holds.

        A re-evaluation stopped before its checkpoint left its decision
        unapplied. When *inputs_at* that sequence give the same decision
        again, it is finished instead of taking a new sequence. They never
        do once a source was removed after it decided, since the inputs bind
        the run's tombstones: a validator then decides again.
        """
        operations = self.store.list_operations(self.run_id, "acceptance")
        latest = max((_decision_sequence(item["operation_key"]) for item in operations), default=0)
        unapplied = {item["operation_key"] for item in operations
                     if item["applied_revision"] is None}
        if latest and _decision_key(inputs_at(latest)) in unapplied:
            return latest
        return latest + 1

    async def _decide(self, synthesis_key, output, *, reevaluation=False):
        """Decide delivery acceptance for exactly *output* as a journal operation.

        Final validators run once, locally, and the decision is committed
        before the completed checkpoint consumes it. A run interrupted before
        that commit decides again on the same output; a committed decision is
        read back and never made again. A *reevaluation* decides under the
        next decision sequence (see :meth:`reevaluate`).
        """
        from smythe.swarm import _delivery_sources, _synthesis_strategy

        strategy = _synthesis_strategy(self.sources["synthesizer"])
        sources, used_fallback = _delivery_sources(self.graph, strategy)
        bound = [AcceptanceSource(node.id, node_generation(node)) for node in sources]
        # The removals the decision is made under; only a run with context
        # assembly can have any.
        tombstones = _tombstone_keys(self.store, self.run_id) if self._context_version else []

        def inputs_at(sequence):
            return self._acceptance_inputs(synthesis_key, text_sha256(output), strategy, bound,
                                           used_fallback, sequence, tombstones)

        sequence = self._next_decision_sequence(inputs_at) if reevaluation else 0
        inputs = inputs_at(sequence)

        async def decide(_):
            record = evaluate_delivery(
                self.acceptance, self.graph, output, strategy=strategy, sources=sources,
                used_fallback=used_fallback, task=self.graph.task,
                unusable=self._unusable_evidence_validators(),
            )
            return {"delivery_acceptance": record.to_dict()}

        result = await self._operation_result(_decision_key(inputs), "acceptance", inputs, decide)
        try:
            record = DeliveryAcceptance.from_dict(result.get("delivery_acceptance"))
        except (TypeError, ValueError) as error:
            raise WorkflowStateError("Saved delivery acceptance is invalid") from error
        if (record.decision is AcceptanceDecision.NOT_EVALUATED
                or record.task_sha256 != self.undecided.task_sha256
                or record.policy_sha256 != self.acceptance.sha256
                or self._acceptance_inputs(synthesis_key, record.output_sha256, record.strategy,
                                           record.sources, record.used_fallback,
                                           sequence, tombstones) != inputs):
            raise WorkflowStateError("Saved delivery acceptance does not match its output")
        self.trace.on_acceptance(record)
        return record

    async def _operator_decide(self, operator, record):
        """Journal *operator*'s decision on the current *record* as the next acceptance operation.

        The assertion is verified first, so nothing is journaled for one the
        verifier refuses. An operator decision stopped before its checkpoint
        is finished by the same decision again.
        """
        sequence = current_decision(self.store, self.run_id)["sequence"]
        decided = _operator_record(operator, self.run_id, sequence, record, WorkflowStateError)
        # Like a policy's decision, it binds the removals it was made under.
        tombstones = _tombstone_keys(self.store, self.run_id) if self._context_version else []

        def inputs_at(number):
            return _decision_inputs(self._synthesis_key(), decided.policy_sha256,
                                    self.undecided.task_sha256, decided.output_sha256,
                                    decided.strategy, decided.sources, decided.used_fallback,
                                    number, tombstones=tombstones,
                                    operator=_operator_inputs(decided))

        inputs = inputs_at(self._next_decision_sequence(inputs_at))

        async def write(_):
            return {"delivery_acceptance": decided.to_dict()}

        result = await self._operation_result(_decision_key(inputs), "acceptance", inputs, write)
        if result.get("delivery_acceptance") != decided.to_dict():
            raise WorkflowStateError("Saved delivery acceptance does not match its output")
        self.trace.on_acceptance(decided)
        return decided

    def _reverified(self, output):
        """Check a completed run's decision against its checkpoint and journal now.

        The saved output must match the record's digest, the task and source
        generations must be those it binds, and the record must be the
        journal's current decision (see :func:`_decision_problems`), which a
        re-evaluation may have made under another policy. A mismatch gives an
        ``indeterminate`` record for the saved output, never acceptance. Runs
        without a strict policy have no record.
        """
        record = self.saved_acceptance
        if record is None:
            return None
        if not isinstance(output, str):
            raise WorkflowStateError("Completed workflow output is not text")
        problems = _decision_problems(self.store, self.run_id, self.graph, output, record,
                                      task=task_from_dict(self.run["task"]),
                                      policy_sha256=self.run_policy_sha256)
        if self.renderer is not None:
            # The delivered document must also be published intact.
            problems += artifact_problems(current_artifact(self.store, self.run_id), output)
        return self._held(record, output, problems)

    def _held(self, record, output, problems):
        """*record* when nothing is wrong, else ``indeterminate`` for *output* with *problems*."""
        return _held_record(record, output, problems, self.undecided.task_sha256)

    def _supervision_update(self, node, applied):
        disposition = node.metadata.get("workflow_supervision")
        if disposition is None:
            return
        operation = self.store.load_operation(self.run_id, disposition["operation_key"])
        if operation is None or operation["state"] == "started":
            raise WorkflowStateError("Supervisor disposition has no durable result")
        disposition.update(state="applied", revision_applied=applied)
        self._pending_operations.add(operation["operation_id"])
        # Added agent assignments and model validation precede the checkpoint.
        self._prepare_graph(self.graph)
        self._save()

    def _result(self, output, acceptance=None):
        """The run's result; output a strict decision did not accept is ``rejected_output``."""
        from smythe.swarm import SwarmResult

        accounting = self.budget.snapshot()
        delivered = acceptance is None or acceptance.accepted
        return SwarmResult(
            output=output if delivered else "", graph=self.graph, trace=self.trace.summary(),
            execution_id=self.run_id,
            total_cost_usd=accounting["confirmed_nanousd"] / 1e9,
            cost_is_complete=not accounting["unknown_calls"] and not accounting["reserved_nanousd"],
            cost_contains_estimates=False, cost_scope="complete_text_workflow",
            # The unit tag names the *_nanousd amounts' unit; stored rows
            # and receipts do not carry it.
            workflow_accounting={key: accounting[key] for key in (
                "confirmed_nanousd", "reserved_nanousd", "unknown_nanousd", "unknown_calls", "call_count",
            )} | {"unit": NANOUSD},
            timing=self._timing(), acceptance=acceptance,
            rejected_output=None if delivered else output,
        )

    def _timing(self):
        """Timing from journal wall-clock timestamps, which relate processes.

        Each lease session is one segment. This call's monotonic clock is not
        used: an earlier session may have run in another process.
        """
        record = self.store.lease_sessions(self.run_id)
        segments = [{
            "started_at": session["started_ns"] / 1e9, "ended_at": session["ended_ns"] / 1e9,
            "elapsed_ms": round((session["ended_ns"] - session["started_ns"]) / 1e6, 1),
            "interrupted": session["interrupted"],
        } for session in record["sessions"]]
        resumed = len(segments) > 1
        return {
            "clock": "wall", "elapsed_ms": segments[-1]["elapsed_ms"] if segments else 0.0,
            "run_wall_ms": round((record["now_ns"] - record["created_ns"]) / 1e6, 1),
            "latency_complete": not resumed, "resumed": resumed, "segments": segments,
        }


class _WorkflowSupervisor:
    def __init__(self, runtime):
        self.runtime = runtime

    async def review(self, graph, node, *, task, revisions_remaining):
        runtime = self.runtime
        key = f"supervision/{node.id}/{node_generation(node)}"
        node.metadata["workflow_supervision"] = {
            "operation_key": key, "generation": node_generation(node), "state": "pending",
        }
        operation = runtime.store.load_operation(runtime.run_id, key)
        if operation is None:
            runtime.store.begin_operation(runtime.context.lease, key, "supervision", {
                "graph": graph_to_dict(graph), "node_id": node.id, "task": task_to_dict(task),
                "revisions_remaining": revisions_remaining,
            })
        runtime._save()

        async def review(inputs):
            frozen = graph_from_dict(inputs["graph"])
            frozen_node = next(item for item in frozen.nodes if item.id == inputs["node_id"])
            try:
                revision = await runtime.bound["supervisor"].review(
                    frozen, frozen_node, task=task_from_dict(inputs["task"]),
                    revisions_remaining=inputs["revisions_remaining"],
                )
            except BaseExceptionGroup as error:
                pending = list(error.exceptions)
                while pending:
                    nested = pending.pop()
                    if isinstance(nested, BaseExceptionGroup):
                        pending.extend(nested.exceptions)
                    elif (not isinstance(nested, Exception)
                          or isinstance(nested, (WorkflowError, ProviderResponseError, BudgetValidationError))):
                        raise
                return {"revision": None, "local_error_type": type(error).__name__}
            except (WorkflowError, ProviderResponseError, BudgetValidationError):
                raise
            except Exception as error:
                # Match the legacy no-change disposition for local review
                # failures, but persist it so recovery never repeats the review.
                return {"revision": None, "local_error_type": type(error).__name__}
            if revision is None:
                return {"revision": None}
            if not isinstance(revision, Revision):
                raise WorkflowStateError("Supervisor must return Revision or None")
            return {"revision": {
                "add_nodes": [node_to_dict(item) for item in revision.add_nodes],
                "drop_node_ids": list(revision.drop_node_ids),
                "rewire": {key: list(value) for key, value in revision.rewire.items()},
                "reason": revision.reason,
            }}

        result = await runtime._operation_result(key, "supervision", {}, review)
        revision = result["revision"]
        if revision is None:
            return None
        proposal = Revision(
            add_nodes=tuple(node_from_dict(item) for item in revision["add_nodes"]),
            drop_node_ids=tuple(revision["drop_node_ids"]),
            rewire={key: tuple(value) for key, value in revision["rewire"].items()},
            reason=revision["reason"],
        )
        # A checkpoint can lag an already admitted worker. Resetting RUNNING
        # to PENDING on recovery must not make that work rewritable history.
        targets = set(proposal.drop_node_ids) | set(proposal.rewire)
        generations = {item.id: node_generation(item) for item in graph.nodes if item.id in targets}
        # Calls are keyed by node id and generation. A new node that reuses the
        # id of a dropped node that made calls would conflict with, or replay,
        # that node's calls.
        present = {item.id for item in graph.nodes}
        reused = {f"node/{item.id}" for item in proposal.add_nodes if item.id not in present}
        for call in runtime.store.inspect_run(runtime.run_id)["calls"]:
            call_key = call["key"]
            if call_key["scope_id"] in reused:
                runtime.trace.on_revision(
                    node, proposal, applied=False,
                    detail="Revision adds a node whose id already has calls in the durable journal",
                )
                return None
            target = call_key["scope_id"].removeprefix("node/")
            if (call_key["phase"] in {"execution", "verification", "tool"}
                    and call_key["scope_id"].startswith("node/") and target in generations
                    and call_key["generation"] == generations[target]):
                runtime.trace.on_revision(
                    node, proposal, applied=False,
                    detail="Revision targets work already admitted in the durable journal",
                )
                return None
        # A node given its context snapshot is admitted too: its saved
        # snapshot names the dependencies a rewire would change.
        if runtime._context_version and any(
            runtime.store.load_operation(runtime.run_id, context_key(target, generation)) is not None
            for target, generation in generations.items()
        ):
            runtime.trace.on_revision(
                node, proposal, applied=False,
                detail="Revision targets work already admitted in the durable journal",
            )
            return None
        if runtime.graph_policy is not None:
            try:
                candidate = graph_from_dict(json_snapshot(graph_to_dict(graph)))
                additions = tuple(node_from_dict(json_snapshot(node_to_dict(item)))
                                  for item in proposal.add_nodes)
                candidate.apply_revision(Revision(
                    add_nodes=additions, drop_node_ids=proposal.drop_node_ids,
                    rewire=proposal.rewire, reason=proposal.reason,
                ))
                # Validate the exact context the executor will give new nodes,
                # without changing the live graph or the returned proposal.
                runtime.executor._inherit_execution_context(additions, candidate)
                runtime._validate_graph(candidate)
            except (ValueError, TypeError, KeyError) as error:
                runtime.trace.on_revision(node, proposal, applied=False, detail=str(error))
                return None
        return proposal
