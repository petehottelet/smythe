"""Artifact lineage: the policy a durable run opts in with, and its record format.

:class:`LineagePolicy` enters a strict durable run's recipe under ``lineage``
only when it is configured, and is compared exactly, like ``graph_policy``,
so a run never changes its access scope or reuse mode while in flight.
:class:`ArtifactLineage` records how one run's accepted artifact was made, by
ids, versions and SHA-256 digests (:mod:`smythe._canonical`) and never text.
A run with a policy writes its record once, when it completes, as a
``lineage`` journal operation (:func:`build_lineage`, :func:`load_lineage`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any

from smythe._canonical import sha256_hex
from smythe.acceptance import AcceptanceDecision, DeliveryAcceptance, task_sha256
from smythe.context import ContextSnapshot, ContextSource, context_key, text_sha256
from smythe.evidence import claim_exchange
from smythe.executor_base import _exchanges_claims
from smythe.graph import ExecutionGraph, Node, NodeStatus
from smythe.validation import ValidatorIdentity, _require_keys, _require_sha256
from smythe.verifier import node_generation
from smythe.workflow_artifact import PublishedArtifact
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_tools import READ_STEP_KEY

__all__ = [
    "LINEAGE_OPERATION", "LINEAGE_POLICY_VERSION", "LINEAGE_RECORD_VERSION", "LINEAGE_REUSE_KEY",
    "ArtifactLineage", "LineagePolicy", "build_lineage", "lineage_key", "load_lineage",
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
_ACCEPTANCE = _object({
    "operation_key": _long, "record_sha256": _sha, "policy_sha256": _sha,
    "decision": _choice(*(item.value for item in AcceptanceDecision
                          if item is not AcceptanceDecision.NOT_EVALUATED)),
    "validator_set_sha256": _sha,
    "validators": _list(lambda value, what: ValidatorIdentity.from_dict(value))})
_INPUTS = _object({"sources": _list(_object({"source_id": _id, "version": _nullable(_text),
                                             "sha256": _sha}))}, {
    "context": _object({"snapshot_key": _long,
                        "assembly_version": lambda value, what: _count(value, what, 1)}),
    "tool": _object({"name": _text, "version": _text, "decoder_version": _text}),
    "component": _object({"name": _text, "version": _text}),
    "agent_sha256": _sha})
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
    names a context snapshot, read tool or component, and ``coverage`` is
    ``complete`` only when every node is declared, else ``partial``. A run
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


def lineage_key(synthesis_key: str) -> str:
    """The journal key of the lineage a run writes when it completes after *synthesis_key*.

    ``lineage`` follows the synthesis keyed ``synthesis``, and
    ``lineage/<digest>`` the one keyed ``synthesis/<digest>``, so a run whose
    nodes ran again never replays the lineage of an earlier graph state.
    """
    return LINEAGE_OPERATION + synthesis_key.removeprefix("synthesis")


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


def build_lineage(graph: ExecutionGraph, recipe: Mapping[str, Any],
                  snapshots: Mapping[str, ContextSnapshot], read_calls: Mapping[str, Any],
                  publication: PublishedArtifact | None,
                  acceptance: tuple[str, DeliveryAcceptance],
                  registry_description: Mapping[str, Any], *, synthesis: tuple[str, str],
                  sources: Iterable[ContextSource] = (),
                  reused: Mapping[str, tuple[str, Mapping[str, Any]]] = MappingProxyType({}),
                  predecessor: Mapping[str, Any] | None = None,
                  repair: Mapping[str, Any] | None = None) -> ArtifactLineage:
    """The lineage of a completed durable run, from that run's own records only.

    *graph* is the completed graph, whose ``run_ref`` names the run, and
    *recipe* the run's saved recipe. *snapshots* holds the context snapshot
    each node was given at its final generation, by node id, and
    *read_calls* the journal call record behind each read step's result.
    *publication* is the run's published document or None, *acceptance* the
    decision's operation key and record, *registry_description* the run's
    agents, *synthesis* the synthesis operation's key and output, and
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
    coverage ``partial``.
    """
    run, sources = graph.run_ref, tuple(sources)
    agents = {entry["id"]: sha256_hex({key: value for key, value in entry.items() if key != "id"})
              for entry in registry_description["agents"]}
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
    validators = [ValidatorIdentity.from_dict(item)
                  for item in recipe["acceptance"]["final_validators"]]
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
