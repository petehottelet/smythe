"""Reuse decisions: what a recorded artifact's lineage calls for now.

:func:`evaluate` compares one :class:`~smythe.lineage.ArtifactLineage` record
with a :class:`CurrentState`: the sources a developer's :class:`SourceCatalog`
describes, and the validators, acceptance policy, tools, components, context
assembly and agents a run would use now, with the published file's integrity.
It returns an :class:`ItemDecision`: an action for the item and for each node,
the stable reason codes in :data:`REASONS` behind them, and the authority any
kept result is kept under. It is a pure function. It reads no store or file,
calls nothing but the catalog's ``describe``, and gives equal decisions for
equal inputs, whatever their order.

:func:`repair_plan` applies it to the runs of a durable journal, by default
the heads that :func:`archive_heads` finds, and returns a :class:`RepairPlan`
with each run's decision, derived status and predicted work. Both only read
the journal: nothing is dispatched or written. ``Swarm.repair`` applies a
plan and returns a :class:`RepairResult`.
"""

from __future__ import annotations

# Imported at runtime, not only for type checking, so that
# typing.get_type_hints() can resolve the dataclass field annotations.
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import inspect
from types import MappingProxyType
from typing import Any, Protocol

from smythe._canonical import sha256_hex
from smythe.acceptance import AcceptanceDecision, DeliveryAcceptance, _calls_async
from smythe.context import CONTEXT_ASSEMBLY_VERSION
from smythe.evidence import load_source_tombstones
from smythe.lineage import (
    ArtifactLineage, LineagePolicy, _count, _frozen, _label, _named, _plain, _synthesizer, _text,
    _validator_set, load_lineage, validator_set_sha256,
)
from smythe.validation import ValidatorIdentity, _json_sha256, _require_keys, _require_sha256
from smythe.workflow_artifact import current_artifact
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowError, WorkflowNotFoundError

__all__ = ["ACTIONS", "REASONS", "REPAIR_PLAN_VERSION", "STATUSES", "CurrentState",
           "ItemDecision", "NodeDecision", "RepairPlan", "RepairResult", "SourceCatalog",
           "SourceStatus", "archive_heads", "derived_status", "evaluate", "repair_plan"]

# Every action, from least to most conservative. An item or node takes the most
# conservative action any of its reasons calls for.
ACTIONS = ("reuse", "recheck", "rerender", "regenerate", "review")

# Each reason code and the action it calls for; None where the lineage
# policy's ``undeclared_coverage`` decides.
REASONS = MappingProxyType({
    "lineage_missing": None, "coverage_partial": None, "coverage_undeclared": None,
    "source_unavailable": "review", "source_tombstoned": "review",
    "dependency_review": "review", "decision_not_accepted": "review",
    "artifact_unsafe": "review", "artifact_unreadable": "review", "artifact_unchecked": "review",
    "source_changed": "regenerate", "source_version_changed": "regenerate",
    "tool_changed": "regenerate", "decoder_changed": "regenerate",
    "component_changed": "regenerate", "context_assembly_changed": "regenerate",
    "agent_changed": "regenerate", "task_changed": "regenerate",
    "node_incomplete": "regenerate", "dependency_changed": "regenerate",
    "gate_pair": "regenerate", "reuse_disabled": "regenerate",
    "artifact_missing": "rerender", "artifact_changed": "rerender",
    "validator_set_changed": "recheck", "policy_changed": "recheck",
    "run_in_progress": "review", "repair_in_progress": "review",
    # Only the Jobs repair preview gives it: a provider call may have completed.
    "outcome_unknown": "review",
})

# The reason each published-file integrity status gives, if any.
_INTEGRITY = {"verified": None, "missing": "artifact_missing", "changed": "artifact_changed",
              "unsafe": "artifact_unsafe", "unreadable": "artifact_unreadable",
              "not_checked": "artifact_unchecked"}
# The lineage node kind of a verification gate; the nodes it depends on are what it judges.
_GATE = "gate"
_KEPT = ("reuse", "recheck", "rerender")

REPAIR_PLAN_VERSION = 1
# A run's lifecycle statuses, derived whenever they are needed and never stored.
STATUSES = ("current", "stale", "invalid", "restricted", "superseded")
# The reasons that make a run's accepted artifact invalid as it stands.
_INVALID = frozenset({"artifact_missing", "artifact_changed", "artifact_unsafe",
                      "artifact_unreadable", "decision_not_accepted"})


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceStatus:
    """What a source catalog says about one source now.

    An available source has the ``sha256`` of its current content and, when
    the catalog versions it, a ``version``. An unavailable one has neither
    obligation, and gives a ``tombstone_reason`` when it was withdrawn.
    """

    version: str | None = None
    sha256: str | None = None
    available: bool = True
    tombstone_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("source available must be True or False")
        if self.version is not None:
            _text(self.version, "source version")
        if self.sha256 is not None:
            _require_sha256(self.sha256, "source sha256")
        if self.tombstone_reason is not None:
            _text(self.tombstone_reason, "source tombstone_reason", 512)
        if self.available and (self.sha256 is None or self.tombstone_reason is not None):
            raise ValueError("an available source has a sha256 and no tombstone_reason")


class SourceCatalog(Protocol):
    """Developer code that describes each source as it is now.

    ``describe`` is synchronous and a zero-cost local read, such as a lookup
    in a local index. It returns a :class:`SourceStatus`, or None for a
    source it does not know. A catalog that declares ``metered`` true is
    refused.
    """

    def describe(self, source_id: str) -> SourceStatus | None: ...


def _mapping(value: Any, what: str, check: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    return MappingProxyType({_text(key, f"{what} key"): check(item, f"{what}[{key!r}]")
                             for key, item in value.items()})


def _items(value: Any, what: str) -> list[Any]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"{what} must be a list, tuple or set")
    return list(value)


def _tool(value: Any, what: str) -> Mapping[str, str]:
    data = _require_keys(dict(value) if isinstance(value, Mapping) else value, what,
                         frozenset({"version", "decoder_version"}))
    return MappingProxyType({key: _text(item, f"{what} {key}") for key, item in data.items()})


def _sha(value: Any, what: str) -> str:
    _require_sha256(value, what)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class CurrentState:
    """What a run would use now, for :func:`evaluate` to compare a record with.

    ``catalog`` describes sources. ``validators`` and ``policy_sha256`` are
    the final validators' identities and the acceptance policy's digest; the
    validators are kept as a sorted set. ``tools`` maps each read tool's
    name to its ``version`` and ``decoder_version``, ``components`` each
    component's name to its version, and ``agents`` holds the digest of each
    agent description. ``context_assembly_version`` is the context assembly
    a run would use, None for none. ``tasks`` optionally maps a recorded
    task's digest to the digest of the task that replaces it. ``integrity``
    maps a run id to its published file's status (``verified``, ``missing``,
    ``changed``, ``unsafe``, ``unreadable`` or ``not_checked``). ``lineage``
    is the configured policy, which gives the access scope. Every part is
    read-only.
    """

    catalog: SourceCatalog
    validators: Iterable[ValidatorIdentity]
    policy_sha256: str
    tools: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    components: Mapping[str, str] = field(default_factory=dict)
    context_assembly_version: int | None = CONTEXT_ASSEMBLY_VERSION
    agents: Iterable[str] = frozenset()
    tasks: Mapping[str, str] = field(default_factory=dict)
    integrity: Mapping[str, str] = field(default_factory=dict)
    lineage: LineagePolicy = field(default_factory=LineagePolicy)

    def __post_init__(self) -> None:
        describe = getattr(self.catalog, "describe", None)
        if isinstance(self.catalog, type) or not callable(describe) or _calls_async(describe):
            raise TypeError("catalog must be an object with a synchronous describe(source_id)")
        if getattr(self.catalog, "metered", False) is not False:
            raise ValueError("catalog must be a zero-cost local read; a metered catalog is refused")
        validators = _items(self.validators, "validators")
        if any(not isinstance(item, ValidatorIdentity) for item in validators):
            raise TypeError("validators must be ValidatorIdentity values")
        _require_sha256(self.policy_sha256, "policy_sha256")
        if self.context_assembly_version is not None:
            _count(self.context_assembly_version, "context_assembly_version", 1)
        if type(self.lineage) is not LineagePolicy:
            raise WorkflowBindingError("lineage must be the built-in LineagePolicy")
        detached = {
            "validators": tuple(_validator_set(validators)),
            "tools": _mapping(self.tools, "tools", _tool),
            "components": _mapping(self.components, "components", _text),
            "agents": frozenset(_sha(item, "agents") for item in _items(self.agents, "agents")),
            "tasks": _mapping(self.tasks, "tasks", _sha),
            "integrity": _mapping(self.integrity, "integrity", _integrity),
        }
        for key in detached["tasks"]:
            _require_sha256(key, "tasks key")
        for name, value in detached.items():
            object.__setattr__(self, name, value)

    @property
    def access_scope(self) -> str:
        return self.lineage.access_scope

    @property
    def validator_set_sha256(self) -> str:
        """:func:`~smythe.lineage.validator_set_sha256` of :attr:`validators`."""
        return validator_set_sha256(self.validators)


def _integrity(value: Any, what: str) -> str:
    if value not in _INTEGRITY:
        raise ValueError(f"{what} must be one of {list(_INTEGRITY)}")
    return value


@dataclass(frozen=True, slots=True)
class NodeDecision:
    """What one recorded node's result calls for, with its sorted reason codes."""

    id: str
    action: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "action": self.action, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class ItemDecision:
    """What one recorded artifact calls for under the current state.

    ``action`` is the most conservative action of the item and its nodes,
    and ``reasons`` every reason code behind any of them, sorted. ``nodes``
    follow the record's order. ``authority`` is what kept results are kept
    under: the access scope, the record's ``recipe_sha256``, the current
    ``policy_sha256`` and ``validator_set_sha256``, and the ``sources``
    (id, version and digest) that the nodes it keeps read.
    """

    action: str
    reasons: tuple[str, ...]
    nodes: tuple[NodeDecision, ...]
    # Compared, but left out of the hash: a read-only mapping cannot be hashed.
    authority: Mapping[str, Any] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        """The decision as JSON data, in a fixed order."""
        return {"action": self.action, "reasons": list(self.reasons),
                "nodes": [item.to_dict() for item in self.nodes],
                "authority": _plain(self.authority)}


def _describe(catalog: SourceCatalog, source_id: str) -> SourceStatus | None:
    status = catalog.describe(source_id)
    if inspect.iscoroutine(status):
        status.close()
    if status is not None and not isinstance(status, SourceStatus):
        raise TypeError(f"catalog describe({source_id!r}) must return a SourceStatus or None, "
                        f"not {type(status).__name__}")
    return status


def _action(reasons: Iterable[str], policy: LineagePolicy) -> str:
    return max((REASONS[code] or policy.undeclared_coverage for code in reasons),
               key=ACTIONS.index, default="reuse")


def _node_reasons(node: Mapping[str, Any], statuses: Mapping[str, SourceStatus | None],
                  state: CurrentState, task_changed: bool) -> set[str]:
    """The reason codes a node's own record gives, before its dependencies'."""
    inputs, reasons = node["inputs"], set()
    if node["result_sha256"] is None:
        reasons.add("node_incomplete")
    if node["coverage"] == "undeclared":
        reasons.add("coverage_undeclared")
    if task_changed:
        reasons.add("task_changed")
    for ref in inputs["sources"]:
        status = statuses[ref["source_id"]]
        if status is None or not status.available:
            reasons.add("source_tombstoned" if status is not None and status.tombstone_reason
                        else "source_unavailable")
            continue
        if status.version != ref["version"]:
            reasons.add("source_version_changed")
        if status.sha256 != ref["sha256"]:
            reasons.add("source_changed")
    if "tool" in inputs:
        tool, current = inputs["tool"], state.tools.get(inputs["tool"]["name"])
        if current is None or current["version"] != tool["version"]:
            reasons.add("tool_changed")
        if current is not None and current["decoder_version"] != tool["decoder_version"]:
            reasons.add("decoder_changed")
    if "component" in inputs and (state.components.get(inputs["component"]["name"])
                                  != inputs["component"]["version"]):
        reasons.add("component_changed")
    if "context" in inputs and (inputs["context"]["assembly_version"]
                                != state.context_assembly_version):
        reasons.add("context_assembly_changed")
    if "agent_sha256" in inputs and inputs["agent_sha256"] not in state.agents:
        reasons.add("agent_changed")
    return reasons


def _settle(own: Mapping[str, set[str]], children: Mapping[str, list[str]],
            policy: LineagePolicy) -> dict[str, set[str]]:
    """Each node's reasons with its ancestors': every node that depends on one
    that regenerates regenerates too, as a run resets a node's dependents."""
    reasons = {node_id: set(codes) for node_id, codes in own.items()}
    for code, action in (("dependency_changed", "regenerate"), ("dependency_review", "review")):
        stack = [child for node_id, codes in own.items() if _action(codes, policy) == action
                 for child in children[node_id]]
        seen: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id not in seen:
                seen.add(node_id)
                reasons[node_id].add(code)
                stack.extend(children[node_id])
    return reasons


def evaluate(lineage: ArtifactLineage | None, state: CurrentState) -> ItemDecision | None:
    """What *lineage*'s artifact calls for under *state*, or None for another scope.

    A record whose access scope is not the state's is invisible: None is
    returned before any digest is compared or any source described. A
    missing record (None) calls for the policy's ``undeclared_coverage``.
    Otherwise each node's sources, tool, component, context assembly, agent
    and task are compared with *state*, and a node that regenerates or needs
    review takes its dependents with it. When any node is computed again, a
    gate (a node of kind ``gate``) and the nodes it depends on are never
    reused. With nothing computed again, the published file's integrity,
    the acceptance decision and the validator set and policy decide. The
    actions and reason codes are documented in the lineage guide.
    """
    if lineage is not None and not isinstance(lineage, ArtifactLineage):
        raise TypeError("lineage must be an ArtifactLineage record or None")
    if not isinstance(state, CurrentState):
        raise TypeError("state must be a CurrentState")
    policy = state.lineage
    authority = {"access_scope": state.access_scope, "recipe_sha256": None,
                 "policy_sha256": state.policy_sha256,
                 "validator_set_sha256": state.validator_set_sha256, "sources": []}
    if lineage is None:
        return ItemDecision(policy.undeclared_coverage, ("lineage_missing",), (),
                            _frozen(authority))
    if lineage.access_scope != state.access_scope:
        return None
    nodes = lineage.nodes
    statuses: dict[str, SourceStatus | None] = {}
    for node in nodes:
        for ref in node["inputs"]["sources"]:
            if ref["source_id"] not in statuses:
                statuses[ref["source_id"]] = _describe(state.catalog, ref["source_id"])
    task = lineage.run["task_sha256"]
    task_changed = task is not None and state.tasks.get(task, task) != task
    own = {node["id"]: _node_reasons(node, statuses, state, task_changed) for node in nodes}
    children: dict[str, list[str]] = {node["id"]: [] for node in nodes}
    for node in nodes:
        for dependency in node["depends_on"]:
            children[dependency].append(node["id"])
    reasons = _settle(own, children, policy)
    if any(_action(codes, policy) != "reuse" for codes in reasons.values()):
        for node in nodes:
            if node["kind"] == _GATE:
                for member in (node["id"], *node["depends_on"]):
                    if not reasons[member]:
                        own[member].add("gate_pair")
        reasons = _settle(own, children, policy)
    actions = {node_id: _action(codes, policy) for node_id, codes in reasons.items()}
    item = {"coverage_partial"} if lineage.coverage == "partial" else set()
    integrity = _INTEGRITY[state.integrity.get(lineage.run["run_id"], "not_checked")]
    if lineage.artifact is not None and integrity is not None:
        item.add(integrity)
    acceptance = lineage.acceptance
    if acceptance["decision"] != "accepted":
        item.add("decision_not_accepted")
    if acceptance["validator_set_sha256"] != state.validator_set_sha256:
        item.add("validator_set_changed")
    if acceptance["policy_sha256"] != state.policy_sha256:
        item.add("policy_changed")
    action = max([_action(item, policy), *actions.values()], key=ACTIONS.index)
    if action in _KEPT:
        actions = dict.fromkeys(actions, action)
    if not policy.reuse and action != "recheck":
        for node_id, value in actions.items():
            if value in _KEPT:
                actions[node_id], reasons[node_id] = "regenerate", {"reuse_disabled"}
                item.add("reuse_disabled")
        action = max(action, "regenerate", key=ACTIONS.index)
    kept = {(ref["source_id"], ref["version"] or "", ref["sha256"]): ref
            for node in nodes if actions[node["id"]] in _KEPT for ref in node["inputs"]["sources"]}
    authority.update(recipe_sha256=lineage.run["recipe_sha256"],
                     sources=[dict(kept[key]) for key in sorted(kept)])
    return ItemDecision(
        action, tuple(sorted(item.union(*reasons.values()))),
        tuple(NodeDecision(node["id"], actions[node["id"]], tuple(sorted(reasons[node["id"]])))
              for node in nodes),
        _frozen(authority))


def derived_status(reasons: Iterable[str], *, superseded: bool = False,
                   sources_checked: bool = True) -> str | None:
    """A run's lifecycle status, from the reason codes found for it; computed, never stored.

    ``superseded`` when a completed successor whose decision is accepted
    replaced the run; else ``restricted`` when a source it used was
    withdrawn (``source_tombstoned``); else ``invalid`` when its published
    document no longer holds the decided bytes or its decision is not
    ``accepted``; else ``current`` without any reason and ``stale`` with one.
    A run that has not completed has no status (None), and neither has one
    of the last two kinds when its sources were not described
    (*sources_checked* false), because its freshness is then unknown.
    """
    reasons = set(reasons)
    if superseded:
        return "superseded"
    if "run_in_progress" in reasons:
        return None
    if "source_tombstoned" in reasons:
        return "restricted"
    if reasons & _INVALID:
        return "invalid"
    if not sources_checked:
        return None
    return "stale" if reasons else "current"


def _archive(store: Any, scope: str) -> tuple[dict[str, str], dict[str, tuple[str, bool | None]]]:
    """The runs in *scope* with their journal status, in journal order, and their links.

    A run is in a scope when its recipe records a lineage policy with that
    access scope. Each completed run in the scope whose lineage names a
    predecessor in the scope is linked to it, with whether its current
    decision is ``accepted``. A successor that has not completed, whose saved
    graph names the predecessor it repairs, is linked to it with None: the
    repair is in progress. A run whose record or decision cannot be read
    links nothing; reading that run itself reports why.
    """
    from smythe.workflow import current_decision

    runs = {}
    for run in store.list_runs():
        policy = store.load_run(run["run_id"])["config"].get("lineage")
        if isinstance(policy, dict) and policy.get("access_scope") == scope:
            runs[run["run_id"]] = run["status"]
    links: dict[str, tuple[str, bool | None]] = {}
    for run_id, status in runs.items():
        try:
            if status != "completed":
                saved = store.get_checkpoint(run_id)
                repair = (saved["checkpoint"]["graph"].get("planning") or {}).get(
                    "repair") if saved else None
                if isinstance(repair, dict) and repair["predecessor"]["run_id"] in runs:
                    links[run_id] = (repair["predecessor"]["run_id"], None)
                continue
            record = load_lineage(store, run_id)
            predecessor = (record.predecessor or {}).get("run_id") if record else None
            if predecessor in runs:
                decision = current_decision(store, run_id)
                links[run_id] = (predecessor, decision is not None
                                 and decision["record"].decision is AcceptanceDecision.ACCEPTED)
        except (ValueError, TypeError, KeyError, AttributeError, WorkflowError):
            continue
    return runs, links


def archive_heads(store: Any, scope: str) -> list[str]:
    """The ids of the durable runs in access *scope* that nothing supersedes, in journal order.

    A run is in a scope when its recipe records a :class:`LineagePolicy`
    with that ``access_scope``; runs in other scopes and runs without
    lineage are never listed. Predecessor links come from lineage records:
    a completed successor whose current decision is ``accepted`` supersedes
    its predecessor, and a successor whose decision is not supersedes nothing
    and is not a head itself, so its predecessor stays the head. A successor
    that has not completed, such as one a lost process left, is a repair in
    progress: it is not a head either, and resumes as any durable run does.
    Any other run that has not completed has no record yet, so it is a head.
    Only reads are made, so a store opened read-only will do.
    """
    return _heads(*_archive(store, scope))


def _heads(runs: Mapping[str, str], links: Mapping[str, tuple[str, bool | None]]) -> list[str]:
    superseded = {predecessor for predecessor, accepted in links.values() if accepted}
    return [run_id for run_id in runs
            if run_id not in superseded and links.get(run_id, (None, True))[1]]


class _Described:
    """A catalog that describes each source once and keeps what it said, for one plan."""

    metered = False

    def __init__(self, catalog: SourceCatalog) -> None:
        self.catalog, self.seen = catalog, {}

    def describe(self, source_id: str) -> SourceStatus | None:
        if source_id not in self.seen:
            self.seen[source_id] = _describe(self.catalog, source_id)
        return self.seen[source_id]


class _Withdrawn:
    """One run's view of a plan's catalog: text withdrawn from the run is not available to it.

    A source whose current text a tombstone in the run's journal withdrew is
    described as withdrawn, with the tombstone's reason; a newer text is
    described as the catalog has it, so the run can regenerate from it.
    """

    metered = False

    def __init__(self, catalog: SourceCatalog, tombstones: Iterable[Any]) -> None:
        self.catalog = catalog
        self.reasons = {(item.source_id, item.sha256): item.tombstone for item in tombstones}

    def describe(self, source_id: str) -> SourceStatus | None:
        status = self.catalog.describe(source_id)
        reason = None if status is None else self.reasons.get((source_id, status.sha256))
        return status if reason is None else SourceStatus(available=False, tombstone_reason=reason)


def _agents(entries: Iterable[Mapping[str, Any]]) -> set[str]:
    """The digests a lineage record gives the agents of these registry entries."""
    return {sha256_hex({key: value for key, value in entry.items() if key != "id"})
            for entry in entries}


def _state(recipe: Mapping[str, Any], catalog: SourceCatalog) -> CurrentState:
    """What a run made with *recipe*, the recipe a new run would record, would use now."""
    component = _synthesizer(recipe["components"]["synthesizer"])
    return CurrentState(
        catalog=catalog,
        validators=[ValidatorIdentity.from_dict(item)
                    for item in recipe["acceptance"]["final_validators"]],
        policy_sha256=_json_sha256(recipe["acceptance"], "acceptance policy"),
        tools={_label(item["name"]): {"version": _label(item["version"]),
                                      "decoder_version": _label(item["decoder_version"])}
               for item in recipe.get("read_tools", ())},
        components={component["name"]: component["version"]} if component else {},
        context_assembly_version=recipe.get("context_assembly") or None,
        agents=_agents(recipe["components"]["registry"].get("agents", ())),
        lineage=LineagePolicy.from_dict(recipe["lineage"]),
    )


def _kinds(nodes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(node["kind"] for node in nodes).items()))


def _work(record: ArtifactLineage | None, decision: ItemDecision,
          free_tools: frozenset[str]) -> dict[str, Any]:
    """What carrying out *decision* would take, predicted from *record*; nothing is called."""
    action = decision.action
    pairs = list(zip(record.nodes, decision.nodes)) if record is not None else []
    runs = int(action == "regenerate")
    kept = [node for node, item in pairs if action != "review" and item.action in _KEPT]
    again = [node for node, item in pairs if runs and item.action == "regenerate"]
    reads = [node for node in again if node["kind"] == "read"]
    # Gates count as model steps, and so does a synthesis that names no local component.
    models = sum(node["kind"] in ("model", _GATE)
                 or node["kind"] == "synthesis" and "component" not in node["inputs"]
                 for node in again)
    charged = any(node["inputs"].get("tool", {}).get("name") not in free_tools for node in reads)
    return {"nodes_reused": _kinds(kept), "nodes_regenerated": _kinds(again),
            "read_calls": len(reads), "model_steps": models,
            "publications": int(action == "rerender" or bool(
                runs and record is not None and record.artifact is not None)),
            "acceptance_evaluations": int(action in ("recheck", "rerender", "regenerate")),
            "new_runs": runs,
            "may_incur_charges": bool(runs) and (record is None or models > 0 or charged),
            "allowance_required": bool(runs)}


@dataclass(frozen=True)
class RepairPlan:
    """What each planned durable run's artifact calls for, before anything is dispatched.

    ``items`` hold one entry per run, in the order planned: its ``run_id``,
    the ``lineage_sha256`` of its record (None without one), its derived
    ``status`` (:func:`derived_status`), the ``action``, ``reasons``,
    ``nodes`` and ``authority`` of its :class:`ItemDecision`, its predicted
    ``work`` and, only while successors that have not completed repair it,
    their run ids in ``repairs_in_progress``. ``sources`` is what the
    catalog said about each source the records name, described once for the
    whole plan.
    :attr:`plan_sha256` is the digest of :meth:`to_dict` without itself,
    which holds no time: equal states give equal digests, and a changed
    decision or source description changes it. It guards against applying
    a stale plan; it authorizes nothing.
    """

    store_id: str
    access_scope: str
    items: tuple[Mapping[str, Any], ...]
    sources: tuple[Mapping[str, Any], ...] = ()
    version: int = REPAIR_PLAN_VERSION

    def __post_init__(self) -> None:
        for name in ("items", "sources"):
            object.__setattr__(self, name, _frozen(_plain(list(getattr(self, name)))))

    def to_dict(self) -> dict[str, Any]:
        """The plan as JSON data, ending with its ``plan_sha256``."""
        data = {"version": self.version, "store_id": self.store_id,
                "access_scope": self.access_scope, "sources": _plain(self.sources),
                "items": _plain(self.items)}
        return {**data, "plan_sha256": sha256_hex(data)}

    @property
    def plan_sha256(self) -> str:
        return self.to_dict()["plan_sha256"]

    def __hash__(self) -> int:
        # The generated hash would hash the read-only mappings, which cannot be hashed.
        return hash(self.plan_sha256)


def _source(source_id: str, status: SourceStatus | None) -> dict[str, Any]:
    if status is None:
        return {"source_id": source_id, "version": None, "sha256": None, "available": False,
                "tombstoned": False}
    return {"source_id": source_id, "version": status.version, "sha256": status.sha256,
            "available": status.available, "tombstoned": status.tombstone_reason is not None}


def repair_plan(store: Any, recipe: Mapping[str, Any], catalog: SourceCatalog,
                run_ids: Iterable[str] | None = None) -> RepairPlan:
    """Plan what durable runs' artifacts call for under *recipe*, without changing anything.

    *recipe* is the recipe a new run would record, and it must hold a
    lineage policy: its access scope chooses the runs, and its validators,
    acceptance policy, read profiles, synthesizer, context assembly and
    agents make the :class:`CurrentState`. An agent a run recorded that the
    recipe's registry does not define, such as one assigned when the run
    was planned, stays current. *catalog* describes each source once.
    *run_ids* defaults to :func:`archive_heads`; named runs are planned
    once each, in the order given, and a run outside the scope raises
    :class:`~smythe.workflow_store.WorkflowNotFoundError` as an unknown one
    does. A run that has not completed needs ``review``
    (``run_in_progress``), and so does a run that a successor which has not
    completed repairs (``repair_in_progress``): its item keeps its status and
    names the successor in ``repairs_in_progress``, since that repair
    resumes rather than starting again. Each completed run's published
    document is checked through
    :func:`~smythe.workflow_artifact.current_artifact`. A
    source text that a tombstone in a run's journal withdrew
    (:func:`~smythe.lineage.restrict_source`) is never available to that
    run: it needs ``review`` (``source_tombstoned``) until the catalog
    describes another text, which it may regenerate from, and its status
    stays ``restricted``. Only reads are made.
    """
    if not isinstance(recipe, Mapping) or recipe.get("lineage") is None:
        raise ValueError("A repair plan needs a lineage policy, whose access scope it plans in")
    state = _state(recipe, catalog)
    described = _Described(state.catalog)
    state = replace(state, catalog=described)
    configured = {entry["id"] for entry in recipe["components"]["registry"].get("agents", ())}
    free = frozenset(_label(item["name"]) for item in recipe.get("read_tools", ())
                     if item["metered"] is False)
    runs, links = _archive(store, state.access_scope)
    if run_ids is None:
        run_ids = _heads(runs, links)
    elif isinstance(run_ids, (str, bytes)) or not isinstance(run_ids, Iterable):
        raise TypeError("run_ids must be a sequence of run ids")
    planned = []
    for run_id in dict.fromkeys(run_ids):
        if run_id not in runs:
            raise WorkflowNotFoundError(
                f"No durable run {run_id!r} in access scope {state.access_scope!r}")
        record, restricted = None, False
        if runs[run_id] != "completed":
            decision = replace(evaluate(None, state), action="review", reasons=("run_in_progress",))
        else:
            record = load_lineage(store, run_id)
            recorded = store.get_checkpoint(run_id)["checkpoint"]["registry"]["agents"]
            current = current_artifact(store, run_id)
            tombstones = load_source_tombstones(store, run_id)
            # Text withdrawn from the run restricts it, whatever is available now.
            restricted = bool({(item.source_id, item.sha256) for item in tombstones}
                              & _named(record))
            decision = evaluate(record, replace(
                state, catalog=_Withdrawn(described, tombstones),
                agents=state.agents | _agents(
                    entry for entry in recorded if entry["id"] not in configured),
                integrity={} if current is None else {run_id: current.integrity["status"]}))
        superseded = any(accepted and predecessor == run_id
                         for predecessor, accepted in links.values())
        reasons = [*decision.reasons, *(["source_tombstoned"] if restricted else [])]
        status = derived_status(reasons, superseded=superseded)
        repairing = [successor for successor, (predecessor, accepted) in links.items()
                     if predecessor == run_id and accepted is None]
        if repairing:
            decision = replace(decision, action="review",
                               reasons=tuple(sorted({*decision.reasons, "repair_in_progress"})))
        planned.append({"run_id": run_id,
                        "lineage_sha256": None if record is None else record.sha256,
                        "status": status, **decision.to_dict(),
                        "work": _work(record, decision, free),
                        **({"repairs_in_progress": repairing} if repairing else {})})
    return RepairPlan(store.store_id, state.access_scope, planned,
                      [_source(source_id, described.seen[source_id])
                       for source_id in sorted(described.seen)])


# What applying a plan counts as each item's work, in this order.
_UNITS = ("new_runs", "read_calls", "model_calls", "reused_nodes", "publications",
          "acceptance_evaluations")


def _journal_work(store: Any, run_id: str | None) -> dict[str, int]:
    """The work in a run's journal: calls by kind, and reuse, publication and decision
    operations. None, for no run, has none."""
    if run_id is None:
        return dict.fromkeys(_UNITS, 0)
    phases = [call["key"]["phase"] for call in store.inspect_run(run_id)["calls"]]
    return {"new_runs": 0, "read_calls": phases.count("tool"),
            "model_calls": len(phases) - phases.count("tool"),
            **{unit: len(store.list_operations(run_id, kind)) for unit, kind in (
                ("reused_nodes", "reuse"), ("publications", "publication"),
                ("acceptance_evaluations", "acceptance"))}}


@dataclass(frozen=True)
class RepairResult:
    """What applying a :class:`RepairPlan` did, item by item (``Swarm.repair``).

    ``items`` hold one entry per plan item, in plan order: its ``run_id``,
    ``action`` and ``outcome``, and the ``work`` it did. The outcome is
    ``kept`` for ``reuse``, ``reported`` for ``review``, ``applied`` when the
    item's step ended with an ``accepted`` decision, ``failed`` when it
    raised or ended with another decision, and ``unapplied`` for each item
    after a failure. An item that decided names its ``decision``, a
    ``regenerate`` item the ``successor_run_id`` of the run it started, and
    one that raised its ``error``. ``work`` counts the ``new_runs`` an item
    started and the ``read_calls``, ``model_calls``, ``reused_nodes``,
    ``publications`` and ``acceptance_evaluations`` it added to the journal.
    ``successor_run_ids`` lists the runs started, in order, and
    ``decisions`` maps each run that decided to its new
    :class:`~smythe.acceptance.DeliveryAcceptance`. Every part is read-only.
    """

    plan_sha256: str
    items: tuple[Mapping[str, Any], ...]
    successor_run_ids: tuple[str, ...] = ()
    decisions: Mapping[str, DeliveryAcceptance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _frozen(_plain(list(self.items))))
        object.__setattr__(self, "successor_run_ids", tuple(self.successor_run_ids))
        object.__setattr__(self, "decisions", MappingProxyType(dict(self.decisions)))

    def __hash__(self) -> int:
        # The generated hash would hash the read-only mappings, which cannot be hashed.
        return hash((self.plan_sha256, self.successor_run_ids))

    @property
    def work(self) -> dict[str, int]:
        """Every item's work, summed."""
        return {unit: sum(item["work"][unit] for item in self.items) for unit in _UNITS}

    def to_dict(self) -> dict[str, Any]:
        """The result as JSON data."""
        return {"plan_sha256": self.plan_sha256, "items": _plain(self.items),
                "successor_run_ids": list(self.successor_run_ids),
                "decisions": {run_id: record.to_dict()
                              for run_id, record in self.decisions.items()},
                "work": self.work}
