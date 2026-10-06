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
"""

from __future__ import annotations

# Imported at runtime, not only for type checking, so that
# typing.get_type_hints() can resolve the dataclass field annotations.
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import Any, Protocol

from smythe.acceptance import _calls_async
from smythe.context import CONTEXT_ASSEMBLY_VERSION
from smythe.lineage import (
    ArtifactLineage, LineagePolicy, _count, _frozen, _plain, _text, _validator_set,
    validator_set_sha256,
)
from smythe.validation import ValidatorIdentity, _require_keys, _require_sha256
from smythe.workflow_binding import WorkflowBindingError

__all__ = ["ACTIONS", "REASONS", "CurrentState", "ItemDecision", "NodeDecision",
           "SourceCatalog", "SourceStatus", "evaluate"]

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
})

# The reason each published-file integrity status gives, if any.
_INTEGRITY = {"verified": None, "missing": "artifact_missing", "changed": "artifact_changed",
              "unsafe": "artifact_unsafe", "unreadable": "artifact_unreadable",
              "not_checked": "artifact_unchecked"}
# The lineage node kind of a verification gate; the nodes it depends on are what it judges.
_GATE = "gate"
_KEPT = ("reuse", "recheck", "rerender")


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
