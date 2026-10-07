"""Artifact lineage: the policy a durable run opts in with, and its record format.

:class:`LineagePolicy` enters a strict durable run's recipe under ``lineage``
only when it is configured, and is compared exactly, like ``graph_policy``,
so a run never changes its access scope or reuse mode while in flight.
:class:`ArtifactLineage` records how one run's accepted artifact was made, by
ids, versions and SHA-256 digests (:mod:`smythe._canonical`) and never text.
A run with a policy writes its record once, when it completes, as a
``lineage`` journal operation (:func:`build_lineage`, :func:`load_lineage`).
:func:`restrict_source` withdraws a source's text from every completed run
whose record names it, which restricts what those runs derived.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from smythe._canonical import sha256_hex
from smythe._publish import _fsync_directory, _regular_path
from smythe.acceptance import AcceptanceDecision, DeliveryAcceptance, task_sha256
from smythe.context import ContextSnapshot, ContextSource, context_key, text_sha256
from smythe.evidence import (
    SourceRecord, _journal_tombstone, claim_exchange, load_source_tombstones,
)
from smythe.executor_base import _exchanges_claims
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.validation import ValidatorIdentity, _require_keys, _require_sha256
from smythe.verifier import node_generation
from smythe.workflow_artifact import PublishedArtifact, publications, read_publication
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowStateError
from smythe.workflow_tools import READ_STEP_KEY

__all__ = [
    "LINEAGE_OPERATION", "LINEAGE_POLICY_VERSION", "LINEAGE_RECORD_VERSION", "LINEAGE_REUSE_KEY",
    "PURGE_OPERATION", "ArtifactLineage", "LineagePolicy", "build_lineage", "lineage_key",
    "load_lineage", "purged_artifacts", "restrict_source", "restricted_sources",
    "validator_set_sha256",
]

LINEAGE_POLICY_VERSION = 1
LINEAGE_RECORD_VERSION = 1
# The journal operation kind, and key prefix, of a run's lineage record.
LINEAGE_OPERATION = "lineage"
# Node metadata marking a result reused from an earlier run; a new graph cannot carry it.
LINEAGE_REUSE_KEY = "lineage_reuse"
_SCOPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_KIND = re.compile(r"[a-z][a-z0-9_-]{0,31}")


@dataclass(frozen=True, slots=True, kw_only=True)
class LineagePolicy:
    """How a durable run's artifact lineage is scoped and reused.

    ``access_scope`` names who may reuse the run's results: 1-64 ASCII
    letters, digits, ``.``, ``_``, ``:`` or ``-``, starting with a letter or
    digit. It is declared by configuration, not authenticated, and compared
    exactly before any digest, so equal content in another scope is never
    reused. ``reuse=False`` is the switch that turns reuse off: affected work
    is computed again while the lineage stays readable.
    ``undeclared_coverage`` is what a run whose lineage does not declare every
    node's inputs calls for: ``review`` (the default) or ``regenerate``.
    """

    access_scope: str = "local"
    reuse: bool = True
    undeclared_coverage: str = "review"
    version: int = LINEAGE_POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self) is not LineagePolicy:
            raise WorkflowBindingError("lineage must be the built-in LineagePolicy")
        if type(self.version) is not int or self.version != LINEAGE_POLICY_VERSION:
            raise WorkflowBindingError(f"Unsupported lineage policy version: {self.version!r}")
        if type(self.access_scope) is not str or _SCOPE.fullmatch(self.access_scope) is None:
            raise WorkflowBindingError(
                "lineage access_scope must be 1-64 ASCII letters, digits, '.', '_', ':' or '-', "
                "starting with a letter or digit"
            )
        if type(self.reuse) is not bool:
            raise WorkflowBindingError("lineage reuse must be True or False")
        if type(self.undeclared_coverage) is not str or self.undeclared_coverage not in (
                "review", "regenerate"):
            raise WorkflowBindingError(
                "lineage undeclared_coverage must be 'review' or 'regenerate'")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "access_scope": self.access_scope, "reuse": self.reuse,
                "undeclared_coverage": self.undeclared_coverage}

    @classmethod
    def from_dict(cls, data: Any) -> LineagePolicy:
        """Read a policy written by :meth:`to_dict`; its keys must be exactly those."""
        keys = {"version", "access_scope", "reuse", "undeclared_coverage"}
        if type(data) is not dict or data.keys() != keys:
            raise WorkflowBindingError("Lineage policy has an invalid schema")
        if type(data["version"]) is not int or data["version"] != LINEAGE_POLICY_VERSION:
            raise WorkflowBindingError(f"Unsupported lineage policy version: {data['version']!r}")
        return cls(**data)

    @property
    def sha256(self) -> str:
        """The SHA-256 of :meth:`to_dict` as canonical JSON."""
        return sha256_hex(self.to_dict())


def snapshot_lineage_policy(value: Any) -> LineagePolicy | None:
    """Check and detach a configured policy; None stays None."""
    if value is None:
        return None
    if type(value) is not LineagePolicy:
        raise WorkflowBindingError(
            f"lineage must be a LineagePolicy or None, not {type(value).__name__}")
    return LineagePolicy.from_dict(value.to_dict())


def validator_set_sha256(validators: Iterable[ValidatorIdentity]) -> str:
    """The digest of a set of validator identities, whatever their order or repeats."""
    return sha256_hex([item.to_dict() for item in _validator_set(validators)])


def _validator_set(validators: Iterable[ValidatorIdentity]) -> list[ValidatorIdentity]:
    return sorted(set(validators), key=lambda item: (item.name, item.version, item.config_sha256))


def _plain(value: Any) -> Any:
    """A detached copy with mappings as dicts and sequences as lists."""
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


# Each check below takes a value and what it is, for messages, and returns the
# value as plain JSON data or raises ValueError.
_Check = Callable[[Any, str], Any]


def _text(value: Any, what: str, limit: int = 256) -> str:
    if (not isinstance(value, str) or not value or len(value) > limit
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f"{what} must be 1-{limit} characters without control characters")
    return str.__str__(value)


def _id(value: Any, what: str) -> str:
    """An id: text that is not only whitespace."""
    if not _text(value, what).strip():
        raise ValueError(f"{what} must not be blank")
    return str.__str__(value)


def _long(value: Any, what: str) -> str:
    return _text(value, what, 512)


def _sha(value: Any, what: str) -> str:
    _require_sha256(value, what)
    return str.__str__(value)


def _count(value: Any, what: str, least: int = 0) -> int:
    if type(value) is not int or value < least:
        raise ValueError(f"{what} must be an integer of at least {least}")
    return value


def _path(value: Any, what: str) -> str:
    path = _long(value, what)
    if "\\" in path or ":" in path or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"{what} must be a relative POSIX path")
    return path


def _pattern(pattern: re.Pattern[str]) -> _Check:
    def check(value: Any, what: str) -> str:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"{what} {value!r} is not a valid label")
        return str.__str__(value)
    return check


def _choice(*values: str) -> _Check:
    def check(value: Any, what: str) -> str:
        if not isinstance(value, str) or value not in values:
            raise ValueError(f"{what} must be one of {list(values)}")
        return str.__str__(value)
    return check


def _nullable(inner: _Check) -> _Check:
    return lambda value, what: None if value is None else inner(value, what)


def _list(inner: _Check) -> _Check:
    def check(value: Any, what: str) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError(f"{what} must be a list")
        return [inner(item, f"{what}[{index}]") for index, item in enumerate(value)]
    return check


def _object(fields: dict[str, _Check], optional: dict[str, _Check] | None = None) -> _Check:
    """A check for an object with exactly *fields*, and any of *optional*, each checked."""
    checks = {**fields, **(optional or {})}

    def check(data: Any, what: str) -> dict[str, Any]:
        _require_keys(data, what, frozenset(fields), frozenset(optional or ()))
        return {key: checks[key](data[key], f"{what} {key}") for key in data}
    return check


_RUN = _object({"store_id": _id, "run_id": _id, "recipe_sha256": _sha,
                "task_sha256": _nullable(_sha)})
_ARTIFACT = _nullable(_object({
    "relative_path": _path, "sha256": _sha, "size_bytes": _count, "media_type": _text,
    "renderer": _object({"name": _text, "version": _text, "spec_sha256": _sha})}))
_VALIDATORS = _list(lambda value, what: ValidatorIdentity.from_dict(value))
_ACCEPTANCE = _object({
    "operation_key": _long, "record_sha256": _sha, "policy_sha256": _sha,
    "decision": _choice(*(item.value for item in AcceptanceDecision
                          if item is not AcceptanceDecision.NOT_EVALUATED)),
    "validator_set_sha256": _sha, "validators": _VALIDATORS})
_INPUTS = _object({"sources": _list(_object({"source_id": _id, "version": _nullable(_text),
                                             "sha256": _sha}))}, {
    "context": _object({"snapshot_key": _long,
                        "assembly_version": lambda value, what: _count(value, what, 1)}),
    "tool": _object({"name": _text, "version": _text, "decoder_version": _text}),
    "component": _object({"name": _text, "version": _text}),
    "agent_sha256": _sha,
    # The gate verifier that judged a gate: the recipe's verifier and gate validators.
    "verifier": _object({"name": _text, "validators": _VALIDATORS})})
_NODE = _object({
    "id": _id, "generation": _count, "kind": _pattern(_KIND), "depends_on": _list(_id),
    "result_sha256": _nullable(_sha), "inputs": _INPUTS,
    "coverage": _choice("declared", "undeclared")},
    {"reused_from": _object({"run_id": _id, "generation": _count})})
_PREDECESSOR = _nullable(_object({"store_id": _id, "run_id": _id, "lineage_sha256": _sha}))
_REPAIR = _nullable(_object({"plan_sha256": _sha, "reason": _text}))


def _node_problem(node: dict[str, Any]) -> str | None:
    """What makes a checked node inconsistent, or None."""
    depends_on, inputs = node["depends_on"], node["inputs"]
    if len(set(depends_on)) != len(depends_on) or node["id"] in depends_on:
        return "must depend on distinct other nodes"
    # A reference is its id, version and digest, whatever order its keys came in.
    sources = {(item["source_id"], item["version"], item["sha256"]) for item in inputs["sources"]}
    if len(sources) != len(inputs["sources"]):
        return "lists a source more than once"
    if node["coverage"] == "declared" and not inputs.keys() & {"context", "tool", "component"}:
        return "is declared without a recorded context, tool or component"
    if "verifier" in inputs and node["kind"] != "gate":
        return "names a gate verifier but is not a gate"
    if "reused_from" in node and node["result_sha256"] is None:
        return "is reused without a result"
    return None


@dataclass(frozen=True)
class ArtifactLineage:
    """How one durable run's accepted artifact was made, by ids, versions and digests.

    ``run`` names the run, its recipe and its task. ``artifact`` is the
    published document by its path relative to the artifact directory,
    digest, size, media type and renderer, or None for a run that published
    none. ``acceptance`` binds the delivery decision, its policy and its
    validator set (:func:`validator_set_sha256`). ``nodes`` each follow the
    nodes they depend on, each with the inputs it was given; a ``declared`` node
    names a context snapshot, read tool or component, a gate may name the gate
    ``verifier`` that judged it, and ``coverage`` is ``complete`` only when
    every node is declared, else ``partial``. A run
    that reused results names its ``predecessor`` in the same store, the
    ``repair`` that made it, and, on each reused node, ``reused_from``.

    Every part is read-only. :meth:`from_dict` rejects unknown versions and
    keys at every level; optional blocks are omitted, never null. Records
    compare by value and hash by :attr:`sha256`, so equal records are one
    set member or dictionary key.
    """

    run: Mapping[str, Any]
    access_scope: str
    artifact: Mapping[str, Any] | None
    acceptance: Mapping[str, Any]
    nodes: tuple[Mapping[str, Any], ...]
    coverage: str
    predecessor: Mapping[str, Any] | None = None
    repair: Mapping[str, Any] | None = None
    version: int = LINEAGE_RECORD_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != LINEAGE_RECORD_VERSION:
            raise ValueError(f"Unsupported lineage record version: {self.version!r}")
        checked = {
            "run": _RUN(_plain(self.run), "lineage run"),
            "access_scope": _pattern(_SCOPE)(self.access_scope, "lineage access_scope"),
            "artifact": _ARTIFACT(_plain(self.artifact), "lineage artifact"),
            "acceptance": _ACCEPTANCE(_plain(self.acceptance), "lineage acceptance"),
            "nodes": _list(_NODE)(_plain(self.nodes), "lineage nodes"),
            "coverage": _choice("complete", "partial")(self.coverage, "lineage coverage"),
            "predecessor": _PREDECESSOR(_plain(self.predecessor), "lineage predecessor"),
            "repair": _REPAIR(_plain(self.repair), "lineage repair"),
        }
        acceptance, nodes = checked["acceptance"], checked["nodes"]
        validators = _validator_set(acceptance["validators"])
        acceptance["validators"] = [item.to_dict() for item in validators]
        if acceptance["validator_set_sha256"] != validator_set_sha256(validators):
            raise ValueError("lineage acceptance validator_set_sha256 does not match its validators")
        for node in nodes:
            if "verifier" in node["inputs"]:
                verifier = node["inputs"]["verifier"]
                verifier["validators"] = [item.to_dict()
                                          for item in _validator_set(verifier["validators"])]
        ids = [node["id"] for node in nodes]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("lineage nodes must be a non-empty list of distinct nodes")
        listed: set[str] = set()
        for node in nodes:
            # Each node follows the nodes it depends on, so no dependency is a cycle.
            problem, later = _node_problem(node), set(node["depends_on"]) - listed
            if problem is None and later - set(ids):
                problem = "depends on a node the record does not list"
            elif problem is None and later:
                problem = "depends on a node listed after it"
            if problem is not None:
                raise ValueError(f"lineage node {node['id']!r} {problem}")
            listed.add(node["id"])
        declared = all(node["coverage"] == "declared" for node in nodes)
        if checked["coverage"] != ("complete" if declared else "partial"):
            raise ValueError("lineage coverage is 'complete' only when every node is declared")
        predecessor = checked["predecessor"] or {}
        if predecessor and (predecessor["store_id"] != checked["run"]["store_id"]
                            or predecessor["run_id"] == checked["run"]["run_id"]):
            raise ValueError("lineage predecessor must be another run in the same store")
        if any(node["reused_from"]["run_id"] != predecessor.get("run_id")
               for node in nodes if "reused_from" in node):
            raise ValueError("a reused lineage node must name the record's predecessor")
        for name, value in checked.items():
            object.__setattr__(self, name, _frozen(value))

    def to_dict(self) -> dict[str, Any]:
        """The record as detached JSON data; ``predecessor`` and ``repair`` only when present."""
        data = {"version": self.version, "run": _plain(self.run), "access_scope": self.access_scope,
                "artifact": _plain(self.artifact), "acceptance": _plain(self.acceptance),
                "nodes": _plain(self.nodes), "coverage": self.coverage}
        for name in ("predecessor", "repair"):
            if getattr(self, name) is not None:
                data[name] = _plain(getattr(self, name))
        return data

    @property
    def sha256(self) -> str:
        """The SHA-256 of :meth:`to_dict` as canonical JSON."""
        return sha256_hex(self.to_dict())

    def __hash__(self) -> int:
        # The generated hash would hash the read-only mappings, which cannot be hashed.
        return hash(self.sha256)

    @classmethod
    def from_dict(cls, data: Any) -> ArtifactLineage:
        """Read a record written by :meth:`to_dict`; raises ValueError for anything else."""
        _require_keys(data, "lineage record", frozenset({
            "version", "run", "access_scope", "artifact", "acceptance", "nodes", "coverage"}),
            frozenset({"predecessor", "repair"}))
        if data.get("predecessor", {}) is None or data.get("repair", {}) is None:
            raise ValueError("lineage predecessor and repair are omitted when absent, never null")
        return cls(**data)


def lineage_key(synthesis_key: str, sequence: int = 0) -> str:
    """The journal key of the lineage a run writes when it completes after *synthesis_key*.

    ``lineage`` follows the synthesis keyed ``synthesis``, and
    ``lineage/<digest>`` the one keyed ``synthesis/<digest>``, so a run whose
    nodes ran again never replays the lineage of an earlier graph state. The
    record that a later decision, with *sequence* above 0, supersedes it with
    adds ``/<sequence>``.
    """
    key = LINEAGE_OPERATION + synthesis_key.removeprefix("synthesis")
    return f"{key}/{sequence}" if sequence else key


def load_lineage(store: Any, run_id: str) -> ArtifactLineage | None:
    """The record of a durable run's latest applied ``lineage`` operation, or None.

    A record left unapplied by a process lost before its completed
    checkpoint is not returned. Only reads are made, so a store opened
    read-only will do. Raises ValueError for a saved record that is invalid.
    """
    applied = [item for item in store.list_operations(run_id, LINEAGE_OPERATION)
               if item["applied_revision"] is not None]
    return ArtifactLineage.from_dict(applied[-1]["result"].get("lineage")) if applied else None


def _fits(value: Any) -> bool:
    try:
        _text(value, "")
    except ValueError:
        return False
    return True


def _label(value: Any) -> str:
    """*value* as a record label: itself when it fits, else the digest of its text."""
    return value if _fits(value) else "sha256:" + text_sha256(str(value))


def _reference(source: ContextSource) -> dict[str, Any]:
    return {"source_id": source.source_id, "version": source.version, "sha256": source.sha256}


def _dependency_order(nodes: Iterable[Node]) -> list[Node]:
    """*nodes* with each after the nodes it depends on, otherwise in their given order."""
    ordered: list[Node] = []
    placed: set[str] = set()
    pending = list(nodes)
    while pending:
        waiting = []
        for node in pending:
            if set(node.depends_on) <= placed:
                ordered.append(node)
                placed.add(node.id)
            else:
                waiting.append(node)
        if len(waiting) == len(pending):
            raise ValueError("graph dependencies form a cycle or name a missing node")
        pending = waiting
    return ordered


def _read_text(call: Any, node: Node, generation: int, run_id: str) -> str | None:
    """The text of *call* when it is *node*'s accepted read at *generation* in this run."""
    if not isinstance(call, Mapping):
        return None
    key, decoded = call.get("key"), call.get("decoded_result")
    if (call.get("run_id") != run_id or not isinstance(key, Mapping)
            or (key.get("phase"), key.get("scope_id"), key.get("generation"))
            != ("tool", f"node/{node.id}", generation)
            or (call.get("provider") or {}).get("kind") != "read_tool"
            or call.get("result_state") not in ("accepted", "applied")
            or not isinstance(decoded, Mapping) or type(decoded.get("text")) is not str):
        return None
    return decoded["text"]


def _read_source(node: Node, text: str, sources: tuple[ContextSource, ...]) -> dict | None:
    """The reference of the source a read step returned, or None when it cannot be named.

    Text that is exactly one of the run's sources, under the ``source_id``
    the step's input names when it names one, is that source at its version.
    Other text read under a named ``source_id`` is that source at an unknown
    version.
    """
    named = (node.metadata[READ_STEP_KEY].get("input") or {}).get("source_id")
    named = named if _fits(named) and named.strip() else None
    digest = text_sha256(text)
    matches = [source for source in sources
               if source.sha256 == digest and named in (None, source.source_id)]
    if len(matches) == 1:
        return _reference(matches[0])
    return None if named is None else {"source_id": named, "version": None, "sha256": digest}


def _given(snapshot: ContextSnapshot, node: Node, results: Mapping[str, tuple]) -> bool:
    """Whether *snapshot* gave *node* exactly the dependency results already recorded.

    A node that exchanges claims is given the compact claim record of a
    result that carries one (:func:`~smythe.evidence.claim_exchange`).
    """
    given = {item.node_id: item for item in snapshot.dependencies}
    if given.keys() != set(node.depends_on):
        return False
    claims = _exchanges_claims(node)
    for dep_id, item in given.items():
        status, generation, result, text = results[dep_id]
        skipped = status is NodeStatus.SKIPPED
        exchanged = claim_exchange(text) if claims and not skipped else None
        if exchanged is not None:
            result = text_sha256(exchanged)
        if (item.generation != generation or item.status != ("skipped" if skipped else "completed")
                or not skipped and item.output_sha256 != result):
            return False
    return True


def _synthesizer(description: Mapping[str, Any]) -> dict[str, str] | None:
    """The identity of a synthesizer that calls no model, or None for any other."""
    if description.get("type") == "local_only":
        identity = {"name": description.get("identity"), "version": description.get("version")}
    elif description.get("type") == "synthesizer" and description.get("completion") is None:
        retains = "+retain_contradictions" if description.get("retain_contradictions") else ""
        identity = {"name": f"synthesizer/{description.get('strategy')}",
                    "version": f"{description.get('version')}{retains}"}
    else:
        return None
    return identity if all(_fits(value) for value in identity.values()) else None


def _gate_verifier(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The gate verifier a run with *recipe* judges its gates with, as a record names it.

    It is the recipe's ``verifier`` with the identities of its gate
    ``validators``, sorted, or None for a recipe that names no verifier.
    """
    if "verifier" not in recipe:
        return None
    validators = _validator_set(
        ValidatorIdentity(item["identity"], item["version"], item["config_sha256"])
        for item in recipe.get("validators", ()))
    return {"name": _label(recipe["verifier"]),
            "validators": [item.to_dict() for item in validators]}


def _matching_digests(record: ArtifactLineage) -> set[str]:
    """The digests a run's saved record may have when its journal assembles *record* again.

    A record written before records named their gates' verifier has the
    digest of *record* without them.
    """
    data = record.to_dict()
    for node in data["nodes"]:
        node["inputs"].pop("verifier", None)
    return {record.sha256, sha256_hex(data)}


def build_lineage(graph: ExecutionGraph, recipe: Mapping[str, Any],
                  snapshots: Mapping[str, ContextSnapshot], read_calls: Mapping[str, Any],
                  publication: PublishedArtifact | None,
                  acceptance: tuple[str, DeliveryAcceptance],
                  registry_description: Mapping[str, Any], *, synthesis: tuple[str, str],
                  sources: Iterable[ContextSource] = (),
                  reused: Mapping[str, tuple[str, Mapping[str, Any]]] = MappingProxyType({}),
                  predecessor: Mapping[str, Any] | None = None,
                  repair: Mapping[str, Any] | None = None,
                  validators: Iterable[ValidatorIdentity] | None = None) -> ArtifactLineage:
    """The lineage of a completed durable run, from that run's own records only.

    *graph* is the completed graph, whose ``run_ref`` names the run, and
    *recipe* the run's saved recipe. *snapshots* holds the context snapshot
    each node was given at its final generation, by node id, and
    *read_calls* the journal call record behind each read step's result.
    *publication* is the run's published document or None, *acceptance* the
    decision's operation key and record, *validators* the final validators of
    the policy that made it (by default the recipe's), *registry_description*
    the run's agents, *synthesis* the synthesis operation's key and output, and
    *sources* the run's context sources. A successor run gives its
    *predecessor* and *repair* blocks, and in *reused*, by node id, the
    predecessor run and its record of each node whose result it reused: such
    a node keeps that record's kind, inputs and coverage, and its
    ``reused_from`` names the run and the node's generation there. Nothing is
    read or written.

    Nodes are listed after their dependencies, then the synthesis, kind
    ``synthesis``, which depends on them all. A node is ``declared`` only
    when its inputs are known: a model or gate node given a snapshot whose
    dependency digests are the recorded results (for a node that exchanges
    claims, their compact claim records); a read step whose accepted read
    names its source; or a synthesis by local code or a built-in strategy
    that calls no model. Any other node is ``undeclared``, which makes the
    coverage ``partial``. Each gate names the recipe's gate ``verifier``.
    """
    run, sources = graph.run_ref, tuple(sources)
    agents = {entry["id"]: sha256_hex({key: value for key, value in entry.items() if key != "id"})
              for entry in registry_description["agents"]}
    verifier = _gate_verifier(recipe)
    entries: list[dict[str, Any]] = []
    results: dict[str, tuple] = {}
    for node in _dependency_order(graph.nodes):
        generation = node_generation(node)
        result = (text_sha256(str(node.result))
                  if node.status is NodeStatus.COMPLETED and node.result is not None else None)
        inputs: dict[str, Any] = {"sources": []}
        declared, kind = False, "gate" if node.verifies else "model"
        origin = reused.get(node.id)
        if origin is not None:
            # A reused result keeps the inputs its predecessor recorded for it.
            inputs, kind = _plain(origin[1]["inputs"]), origin[1]["kind"]
            declared = origin[1]["coverage"] == "declared"
        elif READ_STEP_KEY in node.metadata:
            kind, call = "read", read_calls.get(node.id)
            text = _read_text(call, node, generation, run["run_id"])
            if text is not None and text_sha256(text) == result:
                inputs["tool"] = {"name": _label(call["provider"].get("name")),
                                  "version": _label(call["provider"].get("version")),
                                  "decoder_version": _label(call.get("decoder_version"))}
                reference = _read_source(node, text, sources)
                inputs["sources"], declared = [reference] if reference else [], bool(reference)
        else:
            snapshot = snapshots.get(node.id)
            if snapshot is not None and (snapshot.node_id, snapshot.generation) == (node.id, generation):
                inputs["sources"] = [_reference(source) for source in snapshot.sources]
                inputs["context"] = {"snapshot_key": context_key(node.id, generation),
                                     "assembly_version": snapshot.context_assembly}
                declared = _given(snapshot, node, results)
            if node.agent_id in agents:
                inputs["agent_sha256"] = agents[node.agent_id]
            if kind == "gate" and verifier is not None:
                inputs["verifier"] = _plain(verifier)
        results[node.id] = (node.status, generation, result, node.result)
        entries.append({"id": node.id, "generation": generation, "kind": kind,
                        "depends_on": list(dict.fromkeys(node.depends_on)), "result_sha256": result,
                        "inputs": inputs, "coverage": "declared" if declared else "undeclared"})
        if origin is not None:
            entries[-1]["reused_from"] = {"run_id": origin[0], "generation": origin[1]["generation"]}
    key, output = synthesis
    ids = [entry["id"] for entry in entries]
    component = _synthesizer(recipe["components"]["synthesizer"])
    name, number = key, 1
    while name in ids:  # A graph node may already have the synthesis key as its id.
        number += 1
        name = f"{key}#{number}"
    entries.append({"id": name, "generation": 0, "kind": "synthesis", "depends_on": ids,
                    "result_sha256": text_sha256(output),
                    "inputs": {"sources": [], **({"component": component} if component else {})},
                    "coverage": "declared" if component else "undeclared"})
    artifact = None
    if publication is not None:
        renderer = publication.renderer
        artifact = {"relative_path": f"{run['run_id']}/{publication.name}",
                    "sha256": publication.sha256, "size_bytes": publication.size_bytes,
                    "media_type": publication.media_type,
                    "renderer": {"name": _label(renderer.get("identity", renderer.get("type"))),
                                 "version": _label(str(renderer.get("version"))),
                                 "spec_sha256": sha256_hex(renderer)}}
    operation_key, record = acceptance
    validators = list(validators) if validators is not None else [
        ValidatorIdentity.from_dict(item) for item in recipe["acceptance"]["final_validators"]]
    return ArtifactLineage(
        run={"store_id": run["store_id"], "run_id": run["run_id"],
             "recipe_sha256": run["recipe_sha256"], "task_sha256": task_sha256(graph.task)},
        access_scope=recipe["lineage"]["access_scope"], artifact=artifact,
        acceptance={"operation_key": operation_key, "record_sha256": sha256_hex(record.to_dict()),
                    "decision": record.decision.value, "policy_sha256": record.policy_sha256,
                    "validator_set_sha256": validator_set_sha256(validators),
                    "validators": [item.to_dict() for item in validators]},
        nodes=tuple(entries),
        coverage="complete" if all(entry["coverage"] == "declared" for entry in entries)
        else "partial",
        predecessor=predecessor, repair=repair,
    )


# ---------------------------------------------------------------------------
# Restricting a source: retention and access changes
# ---------------------------------------------------------------------------

# The journal operation kind of a published file deleted from a restricted
# run; its key is ``artifact-purge/<publication sequence>``.
PURGE_OPERATION = "artifact_purge"


def _named(record: ArtifactLineage | None) -> set[tuple[str, str]]:
    """The ``(source_id, sha256)`` of each source text *record*'s nodes name as inputs."""
    return set() if record is None else {(ref["source_id"], ref["sha256"])
                                         for node in record.nodes for ref in node["inputs"]["sources"]}


def restricted_sources(store: Any, run_id: str) -> tuple[SourceRecord, ...]:
    """The tombstones that restrict a durable run: each withdraws a source text its lineage names.

    A tombstone, which :func:`restrict_source` or
    :meth:`~smythe.workflow.WorkflowRuntime.remove_source` journals,
    withdraws one source text, by id and SHA-256, from one run. It restricts
    a run whose applied lineage record names that text as any node's input:
    whatever decision follows, what the run derived is never delivered,
    reused, re-evaluated or published again. A run without a record has
    none, and every tombstone of a run whose record cannot be read counts.
    Only reads are made.
    """
    tombstones = load_source_tombstones(store, run_id)
    if not tombstones:
        return ()
    try:
        named = _named(load_lineage(store, run_id))
    except ValueError:
        return tombstones
    return tuple(item for item in tombstones if (item.source_id, item.sha256) in named)


def purged_artifacts(store: Any, run_id: str) -> list[dict[str, Any]]:
    """The published files a restriction deleted from a durable run, in publication order.

    Each is ``{"relative_path", "sha256", "reason", "purged_at"}``: the
    file's path under the artifact directory, the SHA-256 its publication
    recorded, and why and when it was deleted. Only reads are made.
    """
    done = [item for item in store.list_operations(run_id, PURGE_OPERATION)
            if item["state"] != "started"]
    done.sort(key=lambda item: int(item["operation_key"].rpartition("/")[2]))
    return [item["result"]["purge"] for item in done]


def _stamp(store: Any) -> str:
    """The journal clock's time, in UTC to the second."""
    return datetime.fromtimestamp(store._clock_ns() // 1_000_000_000,
                                  timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _purge(store: Any, lease: Any, publication: PublishedArtifact, reason: str,
           stamp: str) -> dict[str, Any]:
    """Delete one file the leased run published, as an ``artifact_purge`` operation."""
    key = f"artifact-purge/{publication.sequence}"
    operation = store.load_operation(lease.run_id, key)
    if operation is None:
        operation = store.begin_operation(lease, key, PURGE_OPERATION, {"purge": {
            "relative_path": f"{lease.run_id}/{publication.name}", "sha256": publication.sha256,
            "reason": reason, "purged_at": stamp}})
    if operation["state"] == "started":
        directory = Path(publication.directory)
        try:
            # Never through a symbolic link or junction, and only a regular file.
            path = _regular_path(directory.parent, f"{directory.name}/{publication.name}")
            path.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass  # Already gone, as when a lost process deleted it before recording it.
        except ValueError as error:
            raise WorkflowStateError(
                f"Cannot purge {publication.name!r} of run {lease.run_id!r}: {error}") from error
        operation = store.complete_operation(lease, key, operation["inputs"])
    return operation


def _restrict(store: Any, run_id: str, texts: Mapping[str, str | None], source_id: str,
              reason: str, stamp: str, purge: bool) -> dict[str, Any]:
    """Tombstone *texts* (SHA-256 to version) in one completed run, and purge its files."""
    lease = store.acquire_lease(run_id, uuid4().hex, ttl_s=60)
    try:
        operations = [_journal_tombstone(store, lease, SourceRecord(
            source_id, source_version=version, sha256=sha256, tombstone=reason, removed_at=stamp))
            for sha256, version in sorted(texts.items())]
        if purge:
            operations += [_purge(store, lease, read_publication(operation, sequence), reason, stamp)
                           for sequence, operation in enumerate(publications(store, run_id))
                           if operation["state"] != "started"]
        pending = [item["operation_id"] for item in operations if item["state"] != "applied"]
        if pending:
            # A copy of the completed checkpoint applies them; nothing else changes.
            saved = store.get_checkpoint(run_id)
            store.save_checkpoint(lease, saved["revision"], saved["checkpoint"],
                                  consumed_operation_ids=sorted(pending),
                                  transition_kind="completed")
    finally:
        store.release_lease(lease)
    return {"run_id": run_id,
            "tombstones": [item["result"]["tombstone"] for item in operations
                           if item["kind"] != PURGE_OPERATION],
            "purged": purged_artifacts(store, run_id)}


def restrict_source(store: Any, source_id: str, *, version: str | None = None,
                    scope: str | None = None, reason: str,
                    purge: bool = False) -> list[dict[str, Any]]:
    """Withdraw a source's text from every completed lineage run whose record names it.

    *version*, when given, withdraws only the references recorded with that
    version. *scope*, when given, limits the runs to that access scope,
    compared before any record or digest is read, so a run in another scope
    is never touched however exactly its digests match; None covers every
    scope. Each completed run whose applied :class:`ArtifactLineage` names
    the text as a node's input journals a tombstone of it under the run's
    lease: the :class:`~smythe.evidence.SourceRecord` and
    ``source_tombstone`` operation that
    :meth:`~smythe.workflow.WorkflowRuntime.remove_source` writes, with the
    source's id, version and SHA-256, *reason* (1-500 characters) and
    ``removed_at`` from the journal's clock, and none of its text. A copy of
    the run's completed checkpoint applies it, and nothing is called. The
    run is then restricted (:func:`restricted_sources`).

    With *purge*, every file the run published is deleted, never through a
    symbolic link, and an ``artifact_purge`` operation records its path
    under the artifact directory, the SHA-256 its publication recorded,
    *reason* and the time. Tombstones and purge records always remain. Text
    the journal's checkpoints, snapshots and provider responses copied is
    not removed.

    Returns one entry per restricted run, in journal order: ``run_id``, the
    ``tombstones`` of the withdrawn text as JSON records, and the run's
    ``purged`` files (:func:`purged_artifacts`). A text a run already holds
    a tombstone of keeps its first one, so calling this again, for example
    after another process released a run, finishes a restriction without
    changing what it recorded. A run that has not completed has no record
    and is not changed.

    Raises ``WorkflowBindingError`` for a blank or overlong source id,
    version or reason, a scope that is not an access scope label, or a
    *purge* that is not True or False; ``WorkflowStateError``, before
    anything is written, for a run in scope whose record cannot be read,
    and for a store opened read-only; and ``WorkflowLeaseError`` for a run
    another process holds.
    """
    try:
        _id(source_id, "source_id")
        if version is not None:
            _text(version, "version")
        if not _text(reason, "reason", 500).strip():
            raise ValueError("reason must not be blank")
    except ValueError as error:
        raise WorkflowBindingError(str(error)) from error
    if scope is not None and (type(scope) is not str or _SCOPE.fullmatch(scope) is None):
        raise WorkflowBindingError("scope must be an access scope label, or None for every scope")
    if type(purge) is not bool:
        raise WorkflowBindingError("purge must be True or False")
    if store.read_only:
        raise WorkflowStateError("Workflow store is read-only")
    affected = []
    for run in store.list_runs():
        run_id = run["run_id"]
        policy = store.load_run(run_id)["config"].get("lineage") \
            if run["status"] == "completed" else None
        # The scope is compared before any record or digest is read.
        if not isinstance(policy, dict) or scope not in (None, policy.get("access_scope")):
            continue
        try:
            record = load_lineage(store, run_id)
        except ValueError as error:
            raise WorkflowStateError(
                f"Run {run_id!r} holds an invalid lineage record, so the sources it used are "
                "unknown") from error
        texts: dict[str, str | None] = {}
        for node in record.nodes if record is not None else ():
            for ref in node["inputs"]["sources"]:
                if ref["source_id"] == source_id and version in (None, ref["version"]):
                    # One tombstone per text; it keeps a version a reference recorded.
                    texts[ref["sha256"]] = texts.get(ref["sha256"]) or ref["version"]
        if texts:
            affected.append((run_id, texts))
    stamp = _stamp(store)
    return [_restrict(store, run_id, texts, source_id, reason, stamp, purge)
            for run_id, texts in affected]
