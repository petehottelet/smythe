"""Artifact lineage: the policy a durable run opts in with, and its record format.

:class:`LineagePolicy` enters a strict durable run's recipe under ``lineage``
only when it is configured, and is compared exactly, like ``graph_policy``,
so a run never changes its access scope or reuse mode while in flight.
:class:`ArtifactLineage` records how one run's accepted artifact was made, by
ids, versions and SHA-256 digests (:mod:`smythe._canonical`) and never text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any

from smythe._canonical import sha256_hex
from smythe.acceptance import AcceptanceDecision
from smythe.validation import ValidatorIdentity, _require_keys, _require_sha256
from smythe.workflow_binding import WorkflowBindingError

__all__ = [
    "LINEAGE_POLICY_VERSION", "LINEAGE_RECORD_VERSION", "LINEAGE_REUSE_KEY", "ArtifactLineage",
    "LineagePolicy", "validator_set_sha256",
]

LINEAGE_POLICY_VERSION = 1
LINEAGE_RECORD_VERSION = 1
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


_RUN = _object({"store_id": _text, "run_id": _text, "recipe_sha256": _sha,
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
_INPUTS = _object({"sources": _list(_object({"source_id": _text, "version": _nullable(_text),
                                             "sha256": _sha}))}, {
    "context": _object({"snapshot_key": _long,
                        "assembly_version": lambda value, what: _count(value, what, 1)}),
    "tool": _object({"name": _text, "version": _text, "decoder_version": _text}),
    "component": _object({"name": _text, "version": _text}),
    "agent_sha256": _sha})
_NODE = _object({
    "id": _text, "generation": _count, "kind": _pattern(_KIND), "depends_on": _list(_text),
    "result_sha256": _nullable(_sha), "inputs": _INPUTS,
    "coverage": _choice("declared", "undeclared")},
    {"reused_from": _object({"run_id": _text, "generation": _count})})
_PREDECESSOR = _nullable(_object({"store_id": _text, "run_id": _text, "lineage_sha256": _sha}))
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
    validator set (:func:`validator_set_sha256`). ``nodes`` follow the
    graph's order, each with the inputs it was given; a ``declared`` node
    names a context snapshot, read tool or component, and ``coverage`` is
    ``complete`` only when every node is declared, else ``partial``. A run
    that reused results names its ``predecessor`` in the same store, the
    ``repair`` that made it, and, on each reused node, ``reused_from``.

    Every part is read-only. :meth:`from_dict` rejects unknown versions and
    keys at every level; optional blocks are omitted, never null.
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
        for node in nodes:
            problem = _node_problem(node)
            if problem is None and not set(node["depends_on"]) <= set(ids):
                problem = "depends on a node the record does not list"
            if problem is not None:
                raise ValueError(f"lineage node {node['id']!r} {problem}")
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

    @classmethod
    def from_dict(cls, data: Any) -> ArtifactLineage:
        """Read a record written by :meth:`to_dict`; raises ValueError for anything else."""
        _require_keys(data, "lineage record", frozenset({
            "version", "run", "access_scope", "artifact", "acceptance", "nodes", "coverage"}),
            frozenset({"predecessor", "repair"}))
        if data.get("predecessor", {}) is None or data.get("repair", {}) is None:
            raise ValueError("lineage predecessor and repair are omitted when absent, never null")
        return cls(**data)
