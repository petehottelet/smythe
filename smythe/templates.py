"""Versioned template records and an append-only template catalog.

:func:`distill_record` turns a graph that ran into a :class:`TemplateRecord`:
the blueprint that :func:`smythe.distill.distill_template` builds from, as
JSON data, with optional applicability, prerequisites, expected outputs,
failure examples, tests, source runs, a parent version and a label review.
A record holds instructions and structure only. It never carries MCP servers,
models, environment, URLs, tool grants, acceptance policies, budgets,
evaluator configuration or filesystem paths.

``template_version`` is the SHA-256 (:mod:`smythe._canonical`) of the fields
that decide what a template builds, and ``record_sha256`` covers the whole
record. :class:`TemplateCatalog` keeps one record per version in
append-only files named by those digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from smythe._canonical import canonical_bytes, sha256_hex
from smythe._publish import MAX_ARTIFACT_BYTES, _regular_path, atomic_publish_bytes
from smythe.acceptance import AcceptanceDecision
from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import SubGraphTemplate
from smythe.distill import (
    _STRUCTURAL_METADATA_KEYS, DistillationError, _describe, _profile_of, _specialize,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus
from smythe.registry import Registry
from smythe.task import Task
from smythe.validation import ValidatorIdentity
from smythe.workflow_binding import LocalOnly

__all__ = [
    "MAX_FAILURE_EXAMPLES", "TEMPLATE_RECORD_TYPE", "TEMPLATE_RECORD_VERSION", "TemplateCatalog",
    "TemplateRecord", "TemplateRecordError", "distill_record",
]

TEMPLATE_RECORD_TYPE = "smythe.template_record"
TEMPLATE_RECORD_VERSION = 1
# A record keeps failure examples as a category and an evidence digest each.
MAX_FAILURE_EXAMPLES = 16
# The fields that decide what a template builds; template_version hashes them.
_BEHAVIOR_KEYS = ("type", "version", "name", "description", "blueprint", "applicability",
                  "prerequisites", "expected_outputs")
_LIST_SECTIONS = frozenset({"expected_outputs", "failure_examples", "tests", "source_runs"})
_HEX = re.compile(r"[0-9a-f]{64}")
# Words a label review compares: four or more letters or digits.
_WORD = re.compile(r"[^\W_]{4,}")


class TemplateRecordError(ValueError):
    """A template record, or a catalog file that should hold one, is invalid."""


def _plain(value: Any) -> Any:
    """A detached copy with mappings as dicts and sequences as lists."""
    if isinstance(value, Mapping):
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


# Each check takes a value and what it is, for messages, and returns the value
# as plain JSON data or raises TemplateRecordError.
_Check = Callable[[Any, str], Any]


def _string(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TemplateRecordError(f"{what} must be a string")
    return str.__str__(value)


def _name(value: Any, what: str) -> str:
    if not _string(value, what).strip():
        raise TemplateRecordError(f"{what} must not be blank")
    return str.__str__(value)


def _hex(value: Any, what: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise TemplateRecordError(f"{what} must be 64 lowercase hexadecimal characters")
    return str.__str__(value)


def _count(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise TemplateRecordError(f"{what} must be a non-negative integer")
    return value


def _flag(value: Any, what: str) -> bool:
    if type(value) is not bool:
        raise TemplateRecordError(f"{what} must be true or false")
    return value


def _seconds(value: Any, what: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise TemplateRecordError(f"{what} must be a positive finite number")
    return value


def _validator(value: Any, what: str) -> dict[str, str]:
    try:
        identity = value if isinstance(value, ValidatorIdentity) else ValidatorIdentity.from_dict(value)
    except ValueError as error:
        raise TemplateRecordError(f"{what}: {error}") from None
    return identity.to_dict()


def _choice(*values: str) -> _Check:
    def check(value: Any, what: str) -> str:
        if not isinstance(value, str) or value not in values:
            raise TemplateRecordError(f"{what} must be one of {list(values)}")
        return str.__str__(value)
    return check


def _nullable(inner: _Check) -> _Check:
    return lambda value, what: None if value is None else inner(value, what)


def _list(inner: _Check, *, unique: bool = False) -> _Check:
    """A list check; a *unique* list is a set, sorted and each item kept once."""
    def check(value: Any, what: str) -> list[Any]:
        if not isinstance(value, list):
            raise TemplateRecordError(f"{what} must be a list")
        items = [inner(item, f"{what}[{index}]") for index, item in enumerate(value)]
        if unique:
            by_bytes = {canonical_bytes(item): item for item in items}
            items = [by_bytes[key] for key in sorted(by_bytes)]
        return items
    return check


def _keys(data: Any, what: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    if not isinstance(data, dict):
        raise TemplateRecordError(f"{what} must be an object")
    for problem, keys in (("is missing", set(required) - data.keys()),
                          ("has unknown", data.keys() - {*required, *optional})):
        if keys:
            raise TemplateRecordError(f"{what} {problem} keys: {sorted(keys, key=repr)}")


def _object(fields: dict[str, _Check], optional: dict[str, _Check] | None = None) -> _Check:
    """A check for an object with exactly *fields*, and any of *optional*, each checked."""
    checks = {**fields, **(optional or {})}

    def check(data: Any, what: str) -> dict[str, Any]:
        _keys(data, what, list(fields), list(optional or ()))
        return {key: checks[key](data[key], f"{what} {key}") for key in checks if key in data}
    return check


_NODE = _object({
    "id": _name, "label": _string, "depends_on": _list(_name), "capabilities": _list(_string),
    "profile": _nullable(_object({"name": _string, "persona": _string,
                                  "capabilities": _list(_string)})),
    "verifies": _nullable(_name), "max_regenerations": _count,
    "failure_policy": _choice(*(policy.value for policy in FailurePolicy)),
    "max_retries": _count, "timeout_s": _nullable(_seconds), "attach_dep_artifacts": _flag,
    "metadata": _object({}, {key: _string for key in _STRUCTURAL_METADATA_KEYS}),
})
# The optional sections, in record order. Each is omitted when empty.
_SECTIONS: dict[str, _Check] = {
    "applicability": _object({key: _list(_name, unique=True) for key in (
        "task_families", "keywords", "required_context_keys", "excluded_task_families")}),
    "prerequisites": _object({
        "capabilities": _list(_name, unique=True),
        "skills": _list(_object({"name": _name, "version": _name}), unique=True),
        "tools": _list(_object({"name": _name, "version": _name, "schema_sha256": _hex}),
                       unique=True),
        "validators": _list(_validator, unique=True)}),
    "expected_outputs": _list(_object({"node": _name, "description": _string,
                                       "validators": _list(_validator, unique=True)})),
    "failure_examples": _list(_object({"category": _name, "evidence_sha256": _hex})),
    "tests": _list(_object({"name": _name, "sha256": _hex})),
    "source_runs": _list(_object({
        "execution_id": _name, "structure_sha256": _hex,
        "acceptance": _choice(*(decision.value for decision in AcceptanceDecision))})),
    "parent_version": _hex,
    "review": _object({"label_findings": _list(_object({
        "node": _name, "contains_goal": _flag, "words": _list(_string)}))}),
}


def _empty(value: Any) -> bool:
    return not value or (isinstance(value, dict) and not any(value.values()))


@dataclass(frozen=True, kw_only=True, eq=False)
class TemplateRecord:
    """A versioned, serializable template: a blueprint plus optional sections.

    ``blueprint`` holds one object per node, as :func:`distill_record` builds
    it; the JSON form and each section are described in docs/learning.md.
    Lists inside ``applicability`` and ``prerequisites`` are sets, sorted
    with each item once, and an empty section is omitted.

    ``template_version`` is the SHA-256 of the type, version, name,
    description, blueprint, applicability, prerequisites and expected
    outputs as canonical JSON, so evidence, tests, source runs and review
    never change it. ``record_sha256`` covers the whole record. Every part
    is read-only. Records compare and hash by ``record_sha256``, so a
    timeout of ``90`` and one of ``90.0`` make different records.
    """

    name: str
    description: str
    blueprint: tuple[Mapping[str, Any], ...]
    applicability: Mapping[str, Any] | None = None
    prerequisites: Mapping[str, Any] | None = None
    expected_outputs: tuple[Mapping[str, Any], ...] = ()
    failure_examples: tuple[Mapping[str, Any], ...] = ()
    tests: tuple[Mapping[str, Any], ...] = ()
    source_runs: tuple[Mapping[str, Any], ...] = ()
    parent_version: str | None = None
    review: Mapping[str, Any] | None = None
    template_version: str = field(init=False)
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = {"name": _name(self.name, "template name"),
                  "description": _string(self.description, "template description"),
                  "blueprint": _list(_NODE)(_plain(self.blueprint), "template blueprint")}
        for key, check in _SECTIONS.items():
            value = getattr(self, key)
            value = None if value is None else check(_plain(value), f"template {key}")
            values[key] = None if _empty(value) else value
        nodes = values["blueprint"]
        ids = {node["id"] for node in nodes}
        if not nodes or len(ids) != len(nodes):
            raise TemplateRecordError("template blueprint must list at least one node, each id once")
        for node in nodes:
            if node["id"] in node["depends_on"] or node["verifies"] == node["id"]:
                raise TemplateRecordError(f"template node {node['id']!r} depends on or verifies itself")
            unknown = set(node["depends_on"]) - ids
            if node["verifies"] is not None and node["verifies"] not in ids:
                unknown.add(node["verifies"])
            if unknown:
                raise TemplateRecordError(
                    f"template node {node['id']!r} names unknown nodes {sorted(unknown)}")
        named = [item["node"] for item in values["expected_outputs"] or ()]
        named += [item["node"] for item in (values["review"] or {}).get("label_findings", ())]
        if set(named) - ids:
            raise TemplateRecordError(
                f"template expected outputs or review name unknown nodes {sorted(set(named) - ids)}")
        if len(values["failure_examples"] or ()) > MAX_FAILURE_EXAMPLES:
            raise TemplateRecordError(
                f"template failure_examples keeps at most {MAX_FAILURE_EXAMPLES} examples")
        for key, value in values.items():
            empty = () if key in _LIST_SECTIONS else None
            object.__setattr__(self, key, empty if value is None else _frozen(value))
        data = self.to_dict()
        object.__setattr__(self, "template_version",
                           sha256_hex({key: data[key] for key in _BEHAVIOR_KEYS if key in data}))
        object.__setattr__(self, "record_sha256", sha256_hex(data))
        if self.parent_version == self.template_version:
            raise TemplateRecordError("a template record cannot name itself as its parent")

    def to_dict(self) -> dict[str, Any]:
        """The record as detached JSON data; optional sections only when non-empty."""
        data = {"type": TEMPLATE_RECORD_TYPE, "version": TEMPLATE_RECORD_VERSION,
                "name": self.name, "description": self.description,
                "blueprint": _plain(self.blueprint)}
        for key in _SECTIONS:
            if getattr(self, key):
                data[key] = _plain(getattr(self, key))
        return data

    @classmethod
    def from_dict(cls, data: Any) -> TemplateRecord:
        """Read a record written by :meth:`to_dict`; raises TemplateRecordError for anything else."""
        if not isinstance(data, dict) or data.get("type") != TEMPLATE_RECORD_TYPE:
            raise TemplateRecordError(f"a template record has type {TEMPLATE_RECORD_TYPE!r}")
        if type(data.get("version")) is not int or data["version"] != TEMPLATE_RECORD_VERSION:
            raise TemplateRecordError(f"Unsupported template record version: {data.get('version')!r}")
        _keys(data, "template record", ("type", "version", "name", "description", "blueprint"),
              tuple(_SECTIONS))
        record = cls(**{key: value for key, value in data.items() if key not in ("type", "version")})
        empty = [key for key in _SECTIONS if key in data and not getattr(record, key)]
        if empty:
            raise TemplateRecordError(f"template record sections {empty} are omitted when empty")
        return record

    def __eq__(self, other: object) -> bool:
        # Equal values can differ as canonical JSON (90 and 90.0), so compare digests.
        if not isinstance(other, TemplateRecord):
            return NotImplemented
        return self.record_sha256 == other.record_sha256

    def __hash__(self) -> int:
        return hash(self.record_sha256)

    def to_template(self) -> SubGraphTemplate:
        """This record as a :class:`SubGraphTemplate` for ``ConstrainedArchitect``.

        Each build returns new pending nodes whose labels point at the new
        task, and a new registry holding one agent per recorded persona.
        """
        blueprint = _plain(self.blueprint)

        def builder(
            task: Task, params: dict | None = None, /, **model_params: object,
        ) -> tuple[list[Node], Registry]:
            # A distilled shape has no parameters. ConstrainedArchitect passes
            # the model's params as keywords, so accept and ignore them rather
            # than failing every selection that carries any. Positional-only
            # arguments keep a param named "task" or "params" from colliding.
            template_registry = Registry()
            nodes: list[Node] = []
            for entry in blueprint:
                agent_id = None
                profile = entry["profile"]
                if profile is not None:
                    # A copy, so changing one build's agent cannot change the template.
                    agent = Agent(profile=AgentProfile(**deepcopy(profile)))
                    template_registry.register(agent)
                    agent_id = agent.id
                nodes.append(
                    Node(
                        id=entry["id"],
                        label=_specialize(entry["label"], task),
                        depends_on=list(entry["depends_on"]),
                        agent_id=agent_id,
                        required_capabilities=list(entry["capabilities"]),
                        metadata=deepcopy(entry["metadata"]),
                        failure_policy=FailurePolicy(entry["failure_policy"]),
                        max_retries=entry["max_retries"],
                        timeout_s=entry["timeout_s"],
                        attach_dep_artifacts=entry["attach_dep_artifacts"],
                        verifies=entry["verifies"],
                        max_regenerations=entry["max_regenerations"],
                    )
                )
            return nodes, template_registry

        return SubGraphTemplate(name=self.name, description=self.description, builder=builder)

    def local_only(self) -> LocalOnly:
        """This record's builder declared for a durable run.

        Use it as the builder of a ``SubGraphTemplate`` given to a durable
        run's ``ConstrainedArchitect``. Its identity is ``template:<name>``
        and its version is :attr:`template_version`, so the run's recipe pins
        this exact template, and resuming with another version raises
        ``WorkflowConflictError``.
        """
        return LocalOnly(lambda: self.to_template().builder, f"template:{self.name}",
                         self.template_version, role="template_builder")


def _label_findings(blueprint: list[dict[str, Any]], goal: Any) -> list[dict[str, Any]]:
    """Nodes whose label repeats the goal, or two or more of its words."""
    if not isinstance(goal, str) or not goal.strip():
        return []

    def fold(text: str) -> str:
        return " ".join(text.casefold().split())

    goal = fold(goal)
    words = set(_WORD.findall(goal))
    findings = []
    for node in blueprint:
        if not isinstance(node["label"], str):
            continue  # The record refuses it.
        label = fold(node["label"])
        shared = sorted(words.intersection(_WORD.findall(label)))
        if goal in label or len(shared) > 1:
            findings.append({"node": node["id"], "contains_goal": goal in label, "words": shared})
    return findings


def distill_record(
    graph: ExecutionGraph,
    *,
    name: str,
    description: str | None = None,
    registry: Registry | None = None,
    require_success: bool = True,
    source_task: Task | None = None,
) -> TemplateRecord:
    """Build a template record from a graph that ran.

    The arguments are those of :func:`smythe.distill.distill_template`, and
    ``record.to_template()`` is the template it returns. ``source_task`` is
    the task the graph ran for, by default the graph's ``task``. Labels that
    contain its goal, or two or more of its words of four or more letters,
    are listed in ``review["label_findings"]`` for a reviewer. They are not
    changed: each build still points a label at the new task.

    Raises:
        DistillationError: For the reasons ``distill_template`` gives, or when
            the graph cannot be written as a record, for example because a
            node depends on a node the graph does not contain.
    """
    if not graph.nodes:
        raise DistillationError("cannot distill an empty graph")
    if require_success:
        unfinished = [
            n.id for n in graph.nodes if n.status is not NodeStatus.COMPLETED
        ]
        if unfinished:
            raise DistillationError(
                "cannot distill a graph with unfinished nodes: "
                f"{', '.join(sorted(unfinished))}"
            )

    # Snapshot structure now; the source graph must not be able to
    # mutate a template that was already handed to an architect.
    # Gates and policies are structure too: a judge without ``verifies``
    # no longer gates, and its verdict becomes the deliverable.
    blueprint = [
        {
            "id": node.id,
            "label": node.label,
            "depends_on": list(node.depends_on),
            "capabilities": list(node.required_capabilities),
            "profile": _profile_of(node, registry),
            "verifies": node.verifies,
            "max_regenerations": node.max_regenerations,
            "failure_policy": getattr(node.failure_policy, "value", node.failure_policy),
            "max_retries": node.max_retries,
            "timeout_s": node.timeout_s,
            "attach_dep_artifacts": node.attach_dep_artifacts,
            "metadata": {
                key: deepcopy(node.metadata[key])
                for key in _STRUCTURAL_METADATA_KEYS if key in node.metadata
            },
        }
        for node in graph.nodes
    ]
    # A gate on a missing node would judge nothing, or, once composed,
    # whichever node happens to carry that id.
    known_ids = {entry["id"] for entry in blueprint}
    for entry in blueprint:
        if entry["verifies"] == entry["id"]:
            raise DistillationError(f"cannot distill node {entry['id']!r}: it verifies itself")
        if entry["verifies"] is not None and entry["verifies"] not in known_ids:
            raise DistillationError(
                f"cannot distill node {entry['id']!r}: it verifies unknown "
                f"node {entry['verifies']!r}"
            )
    task = graph.task if source_task is None else source_task
    try:
        return TemplateRecord(
            name=name, description=description or _describe(graph), blueprint=blueprint,
            review={"label_findings": _label_findings(blueprint, getattr(task, "goal", None))},
        )
    except TemplateRecordError as error:
        raise DistillationError(f"cannot distill this graph: {error}") from error


class TemplateCatalog:
    """Template records kept as append-only files under ``root``.

    ``records/<record_sha256>.json`` holds a record's canonical JSON, so each
    record file is named by the SHA-256 of its bytes, and
    ``versions/<template_version>`` holds the ``record_sha256`` of the one
    record stored for that version. Files are published whole, never replace
    an existing file and are never deleted; new directories and files are
    owner-only where the platform supports it.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def put(self, record: TemplateRecord) -> str:
        """Store *record* and return its template version.

        Storing a record the catalog already holds writes nothing. A
        different record for a stored version, such as one with other source
        runs, raises :class:`TemplateRecordError` and the stored one is kept.
        """
        if not isinstance(record, TemplateRecord):
            raise TypeError("TemplateCatalog stores TemplateRecord values")
        data, digest, version = (canonical_bytes(record.to_dict()), record.record_sha256,
                                 record.template_version)
        conflict = f"the catalog holds a different record for template version {version}"
        try:
            if self._digest(version) != digest:
                raise TemplateRecordError(conflict)
        except KeyError:
            pass
        for directory in (self.root, self.root / "records", self.root / "versions"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The record is published before the version that names it, so an
        # interrupted put leaves at most a whole record that no version names.
        self._publish("records", f"{digest}.json", data,
                      f"template record {digest} does not match its digest")
        self._publish("versions", version, digest.encode("ascii"), conflict)
        return version

    def get(self, version: str) -> TemplateRecord:
        """The record stored for *version*, read and checked again.

        Raises:
            KeyError: No record is stored for *version*.
            TemplateRecordError: The stored files do not hold this version's
                record, for example because one was edited.
        """
        digest = self._digest(version)
        try:
            data = self._read("records", f"{digest}.json", MAX_ARTIFACT_BYTES)
        except KeyError:
            raise TemplateRecordError(f"template version {version} names a missing record") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise TemplateRecordError(f"template record {digest} does not match its digest")
        try:
            record = TemplateRecord.from_dict(json.loads(data))
        except (ValueError, RecursionError) as error:
            raise TemplateRecordError(f"template record {digest} is unreadable: {error}") from None
        if record.template_version != version or record.record_sha256 != digest:
            raise TemplateRecordError(f"template version {version} names a record of another version")
        return record

    def versions(self, name: str) -> list[str]:
        """The stored template versions of the records named *name*, in digest order."""
        try:
            with os.scandir(self.root / "versions") as entries:
                found = sorted(entry.name for entry in entries if _HEX.fullmatch(entry.name))
        except FileNotFoundError:
            return []
        return [version for version in found if self.get(version).name == name]

    def _publish(self, folder: str, name: str, content: bytes, problem: str) -> None:
        """Publish *content* unless the file already holds it; other bytes raise."""
        try:
            stored = self._read(folder, name, len(content))
        except KeyError:
            try:
                atomic_publish_bytes(self.root / folder / name, content)
                return
            except FileExistsError:  # Another put published it first.
                stored = self._read(folder, name, len(content))
        if stored != content:
            raise TemplateRecordError(problem)

    def _digest(self, version: str) -> str:
        if not isinstance(version, str) or _HEX.fullmatch(version) is None:
            raise TemplateRecordError("a template version is 64 lowercase hexadecimal characters")
        digest = self._read("versions", version, 64).decode("ascii", "replace")
        if _HEX.fullmatch(digest) is None:
            raise TemplateRecordError(f"template version {version} does not name a record digest")
        return digest

    def _read(self, folder: str, name: str, limit: int) -> bytes:
        try:
            path = _regular_path(self.root / folder, name)
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
        except FileNotFoundError:
            raise KeyError(name) from None
        except ValueError:
            raise TemplateRecordError(f"template catalog file {folder}/{name} is not a regular file") from None
        if len(data) > limit:
            raise TemplateRecordError(f"template catalog file {folder}/{name} is too large")
        return data
