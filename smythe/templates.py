"""Versioned template records and an append-only template catalog.

:func:`distill_record` turns a graph that ran into a :class:`TemplateRecord`:
the blueprint that :func:`smythe.distill.distill_template` builds from, as
JSON data, with optional applicability, prerequisites, expected outputs,
failure examples, tests, source runs, a parent version and a review.
A record holds instructions and structure only. No key carries MCP servers,
models, environment, URLs, tool grants, acceptance policies, budgets,
evaluator configuration or filesystem paths; free text is copied verbatim,
and the review flags text that repeats the source goal or holds a path.

``template_version`` is the SHA-256 (:mod:`smythe._canonical`) of the fields
that decide what a template builds, and ``record_sha256`` covers the whole
record. :class:`TemplateCatalog` keeps one record per version in
append-only files named by those digests.

A record's structure can come from model output, so
:func:`validate_template_record` checks one as an untrusted candidate, and
:meth:`TemplateCatalog.put` stores only a record that passes: no key that
grants authority, the model-plan limits unless a developer passes others, and
no weaker gate, output check or prerequisite than its parent's.

A policy registry target ``planner.template.<name>`` (:func:`template_target`)
holds the version of template *name* in effect.
:meth:`TemplateCatalog.active_templates` builds those versions for an
ordinary run, :meth:`TemplateCatalog.active_local_templates` declares them
for a durable run, whose recipe pins them, and :func:`pinned_templates`
rebuilds the versions a saved recipe pinned.

:func:`repeated_structures` finds the graph structures that accepted runs in
planner history repeat, and :func:`candidate_from_graph` builds a candidate
record for one, naming those runs and keeping each failed or rejected run's
category and evidence digest. A candidate is never stored or activated here.
:meth:`TemplateRecord.as_architect` builds a record's graph with no planning
call, and a durable run pins it by identity and version.
:class:`TemplateEnvironment` snapshots what a run would use, and
:func:`check_applicability` says whether a record still applies there; the
catalog's active templates can leave out those that do not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from smythe._canonical import canonical_bytes, sha256_hex
from smythe._errors import PicklableError
from smythe._publish import MAX_ARTIFACT_BYTES, _regular_path, atomic_publish_bytes
from smythe.acceptance import AcceptanceDecision
from smythe.agent import Agent, AgentProfile
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.distill import (
    _STRUCTURAL_METADATA_KEYS, DistillationError, _describe, _profile_of, _specialize,
)
from smythe.graph import (
    SYNTHESIS_NODE_ID, ExecutionGraph, FailurePolicy, Node, NodeStatus, _has_cycle_in,
    structure_sha256,
)
from smythe.loader import (
    MODEL_NODE_ID, MODEL_PLAN_MAX_DEPTH, MODEL_PLAN_MAX_GATES, MODEL_PLAN_MAX_NODES,
    MODEL_PLAN_MAX_REGENERATIONS, MODEL_PLAN_MAX_RETRIES, MODEL_PLAN_MIN_TIMEOUT_S,
    _build_model_graph,
)
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.planner import DeterministicArchitect
from smythe.registry import Registry
from smythe.task import Task
from smythe.validation import ValidatorIdentity
from smythe.workflow_binding import LocalOnly

__all__ = [
    "MAX_FAILURE_EXAMPLES", "TEMPLATE_CATALOG_LAYOUT", "TEMPLATE_RECORD_TYPE",
    "TEMPLATE_RECORD_VERSION", "TemplateCatalog", "TemplateLimits", "TemplateRecord",
    "TemplateRecordError", "TemplateValidationError", "distill_record", "pinned_templates",
    "template_target", "validate_template_record",
    "TemplateArchitect", "candidate_from_graph", "repeated_structures",
    "APPLICABILITY_REASONS", "ApplicabilityReport", "TemplateApplicabilityError",
    "TemplateEnvironment", "check_applicability",
]

logger = logging.getLogger("smythe.templates")

TEMPLATE_RECORD_TYPE = "smythe.template_record"
TEMPLATE_RECORD_VERSION = 1
# The catalog's file layout; layout.json records it, and a catalog without
# that file was written in layout 1.
TEMPLATE_CATALOG_LAYOUT = 1
_LAYOUT_FILE = "layout.json"
_LAYOUT = canonical_bytes({"type": "smythe.template_catalog", "layout": TEMPLATE_CATALOG_LAYOUT})
# A record keeps failure examples as a category and an evidence digest each.
MAX_FAILURE_EXAMPLES = 16
# The fields that decide what a template builds; template_version hashes them.
_BEHAVIOR_KEYS = ("type", "version", "name", "description", "blueprint", "applicability",
                  "prerequisites", "proven_with", "expected_outputs")
_LIST_SECTIONS = frozenset({"expected_outputs", "failure_examples", "tests", "source_runs"})
_HEX = re.compile(r"[0-9a-f]{64}")
# Words a label review compares: four or more letters or digits.
_WORD = re.compile(r"[^\W_]{4,}")
# A rooted or home path: a drive letter, a UNC share, a leading slash, ~ or
# ~user, or a home variable. A slash inside a word, a URL or ./ is not one.
_PATH = re.compile(r"(?<![\w.~/\\:<-])(?:[A-Za-z]:[\\/]|\\\\\w|/[\w.-]|~[\w.-]*[\\/])"
                   r"|\$HOME\b|%(?i:USERPROFILE|HOMEPATH)%")
# Keys that would let a template grant itself tools, servers, models,
# environment, commands or network access, or decide acceptance, evaluation
# or spend. A candidate holding one at any depth is refused, except where the
# record format itself uses the name for something that grants nothing: the
# tools prerequisites expect, and the acceptance decision a source run got.
_AUTHORITY_KEYS = frozenset({"mcp_servers", "model", "env", "command", "url", "tools",
                             "allowed_tools", "acceptance", "evaluator", "budget"})
_RECORD_FIELDS = frozenset({("prerequisites", "tools"), ("source_runs", int, "acceptance")})
# A template's name as one segment of its policy registry target.
_TARGET_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


class TemplateRecordError(ValueError):
    """A template record, or a catalog file that should hold one, is invalid."""


class TemplateValidationError(TemplateRecordError):
    """A template record refused as a candidate by :func:`validate_template_record`.

    ``node`` is the id of the blueprint node at fault, or None when the
    problem is not one node's. ``field`` names the field, such as
    ``"max_retries"`` or ``"profile.mcp_servers"``, or is None for data that
    is not a template record at all.
    """

    def __init__(self, message: str, *, node: str | None = None, field: str | None = None) -> None:
        super().__init__(message)
        self.node = node
        self.field = field


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
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # A lone surrogate: canonical JSON cannot hash it.
        raise TemplateRecordError(f"{what} must be text that UTF-8 can encode") from None
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
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:  # An integer too large to be a float.
        finite = False
    if not finite or value <= 0:
        raise TemplateRecordError(f"{what} must be a positive finite number")
    return value


def _validator(value: Any, what: str) -> dict[str, str]:
    try:
        identity = value if isinstance(value, ValidatorIdentity) else ValidatorIdentity.from_dict(value)
    except ValueError as error:
        raise TemplateRecordError(f"{what}: {error}") from None
    return {key: _string(item, f"{what} {key}") for key, item in identity.to_dict().items()}


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
# A tool as LocalToolRuntime.describe() lists it.
_TOOL = _object({"name": _name, "version": _name, "schema_sha256": _hex})
# The optional sections, in record order. Each is omitted when empty.
_SECTIONS: dict[str, _Check] = {
    "applicability": _object({key: _list(_name, unique=True) for key in (
        "task_families", "keywords", "required_context_keys", "excluded_task_families")}),
    "prerequisites": _object({
        "capabilities": _list(_name, unique=True),
        "skills": _list(_object({"name": _name, "version": _name}), unique=True),
        "tools": _list(_TOOL, unique=True),
        "validators": _list(_validator, unique=True)}),
    # Digests of the model ids and MCP allowlists a candidate's runs used.
    "proven_with": _object({"models_sha256": _hex, "mcp_allowlists_sha256": _hex}),
    "expected_outputs": _list(_object({"node": _name, "description": _string,
                                       "validators": _list(_validator, unique=True)})),
    "failure_examples": _list(_object({"category": _name, "evidence_sha256": _hex})),
    "tests": _list(_object({"name": _name, "sha256": _hex})),
    "source_runs": _list(_object({
        "execution_id": _name, "structure_sha256": _hex,
        "acceptance": _choice(*(decision.value for decision in AcceptanceDecision))})),
    "parent_version": _hex,
    "review": _object({}, {
        "label_findings": _list(_object({
            "node": _name, "contains_goal": _flag, "words": _list(_string)})),
        "path_findings": _list(_object({
            "node": _nullable(_name), "field": _choice("description", "label", "persona")}))}),
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
    description, blueprint, applicability, prerequisites, ``proven_with``
    and expected outputs as canonical JSON, so evidence, tests, source runs
    and review never change it. ``record_sha256`` covers the whole record.
    Every part is read-only. Records compare and hash by ``record_sha256``,
    so a timeout of ``90`` and one of ``90.0`` make different records.
    """

    name: str
    description: str
    blueprint: tuple[Mapping[str, Any], ...]
    applicability: Mapping[str, Any] | None = None
    prerequisites: Mapping[str, Any] | None = None
    proven_with: Mapping[str, Any] | None = None
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
        if _has_cycle_in({node["id"]: node["depends_on"] for node in nodes}):
            raise TemplateRecordError("template blueprint has a dependency cycle")
        review = values["review"] or {}
        if not all(review.values()):
            raise TemplateRecordError("template review findings are omitted when empty")
        for item in review.get("path_findings", ()):
            if (item["node"] is None) != (item["field"] == "description"):
                raise TemplateRecordError("a template path finding names a node exactly when "
                                          "its field is a label or persona")
        named = [item["node"] for item in values["expected_outputs"] or ()]
        named += [item["node"] for findings in review.values() for item in findings
                  if item["node"] is not None]
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
        The builder's ``template_version`` is this record's, which a plan
        composed from it records in ``graph.planning["templates"]``.
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

        builder.template_version = self.template_version  # type: ignore[attr-defined]
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

    def as_architect(self) -> TemplateArchitect:
        """A :class:`TemplateArchitect` that builds this template for any task without planning."""
        return TemplateArchitect(self)


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


def _path_findings(blueprint: list[dict[str, Any]], description: str) -> list[dict[str, Any]]:
    """The description, labels and personas that hold a rooted or home path."""
    texts = [(None, "description", description)]
    for node in blueprint:
        texts += [(node["id"], "label", node["label"]),
                  (node["id"], "persona", (node["profile"] or {}).get("persona"))]
    return [{"node": node, "field": field} for node, field, text in texts
            if isinstance(text, str) and _PATH.search(text)]


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
    are listed in ``review["label_findings"]`` for a reviewer, and the
    description, labels and personas that hold a rooted or home path, such
    as ``D:\\data\\...``, ``/srv/...`` or ``~/...``, in
    ``review["path_findings"]``. Text is copied verbatim, never changed: each
    build still points a label at the new task.

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
        record = TemplateRecord(name=name, description=description or "", blueprint=blueprint)
        # Summarized only once the record's checks pass: a graph with a
        # cycle or a label that is not text has no summary.
        description = description or _describe(graph)
        review = {"label_findings": _label_findings(blueprint, getattr(task, "goal", None)),
                  "path_findings": _path_findings(blueprint, description)}
        return replace(record, description=description,
                       review={key: value for key, value in review.items() if value})
    except TemplateRecordError as error:
        raise DistillationError(f"cannot distill this graph: {error}") from error


@dataclass(frozen=True, kw_only=True)
class TemplateLimits:
    """The limits :func:`validate_template_record` holds a record's blueprint to.

    The defaults are the limits on a generated plan
    (``smythe.loader.MODEL_PLAN_*``), because a record's structure can come
    from model output. Pass other limits only for a record whose structure
    you wrote or reviewed.
    """

    max_nodes: int = MODEL_PLAN_MAX_NODES
    max_depth: int = MODEL_PLAN_MAX_DEPTH
    max_retries: int = MODEL_PLAN_MAX_RETRIES
    max_regenerations: int = MODEL_PLAN_MAX_REGENERATIONS
    max_gates: int = MODEL_PLAN_MAX_GATES
    min_timeout_s: int | float = MODEL_PLAN_MIN_TIMEOUT_S

    def __post_init__(self) -> None:
        for name in ("max_nodes", "max_depth", "max_retries", "max_regenerations", "max_gates"):
            value, least = getattr(self, name), int(name in ("max_nodes", "max_depth"))
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")
        if type(self.min_timeout_s) not in (int, float) or not 0 <= self.min_timeout_s < math.inf:
            raise ValueError(
                f"min_timeout_s must be a finite number of at least 0, got {self.min_timeout_s!r}")


def _refuse(node: str | None, field: str, problem: str) -> TemplateValidationError:
    where = f"template node {node!r} {field}" if node is not None else f"template {field}"
    return TemplateValidationError(f"{where} {problem}", node=node, field=field)


def _check_authority(data: Mapping[str, Any]) -> None:
    """Refuse a key that would grant authority, at any depth of record *data*."""
    queue, seen = deque([((), data)]), set()
    while queue:
        path, value = queue.popleft()
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in seen:
                continue
            seen.add(id(value))
        if isinstance(value, Mapping):
            for key, item in value.items():
                shape = (*(int if isinstance(part, int) else part for part in path), key)
                if key in _AUTHORITY_KEYS and shape not in _RECORD_FIELDS:
                    node, path = None, (*path, key)
                    if path[0] == "blueprint" and isinstance(path[1], int):
                        entry = data["blueprint"][path[1]]
                        node = entry.get("id") if isinstance(entry, Mapping) else None
                        node, path = (node, path[2:]) if isinstance(node, str) else (None, path)
                    where = "".join(f"[{part}]" if isinstance(part, int) else f".{part}"
                                    for part in path).lstrip(".")
                    raise _refuse(node, where, "would grant authority a template never holds")
                queue.append(((*path, key), item))
        elif isinstance(value, (list, tuple)):
            queue.extend(((*path, index), item) for index, item in enumerate(value))


def _candidate(record: Any) -> TemplateRecord:
    """*record*, or the record its JSON data holds, with no key that grants authority."""
    if isinstance(record, TemplateRecord):
        _check_authority(record.to_dict())
        return record
    if not isinstance(record, Mapping):
        raise TypeError("a template candidate is a TemplateRecord or its JSON data")
    _check_authority(record)
    try:
        return TemplateRecord.from_dict(dict(record))
    except RecursionError:
        raise TemplateValidationError("template record data is nested too deeply") from None
    except TemplateRecordError as error:
        raise TemplateValidationError(str(error)) from None


def _check_limits(nodes: list[dict[str, Any]], limits: TemplateLimits) -> None:
    """Refuse a blueprint that a generated plan under *limits* could not have."""
    if len(nodes) > limits.max_nodes:
        raise _refuse(None, "blueprint", f"has {len(nodes)} nodes, above the limit of "
                                         f"{limits.max_nodes}")
    gates = 0
    for node in nodes:
        name = node["id"]
        if MODEL_NODE_ID.fullmatch(name) is None or name == SYNTHESIS_NODE_ID:
            raise _refuse(name, "id", f"must be 1-64 letters, digits, '-' or '_', and not "
                                      f"{SYNTHESIS_NODE_ID!r}")
        if not node["label"].strip():
            raise _refuse(name, "label", "must not be blank")
        if node["profile"] is not None and not node["profile"]["name"].strip():
            raise _refuse(name, "profile.name", "must not be blank")
        for field_name in ("max_retries", "max_regenerations"):
            cap = getattr(limits, field_name)
            if node[field_name] > cap:
                raise _refuse(name, field_name, f"is {node[field_name]}, above the limit of {cap}")
        if node["timeout_s"] is not None and node["timeout_s"] < limits.min_timeout_s:
            raise _refuse(name, "timeout_s", f"is {node['timeout_s']}, below the minimum of "
                                             f"{limits.min_timeout_s} seconds")
        if node["verifies"] is not None:
            gates += 1
            if gates > limits.max_gates:
                raise _refuse(name, "verifies", f"adds gate {gates}, above the limit of "
                                                f"{limits.max_gates}")
            # A gate that could run before its target would have its verdict discarded.
            if node["verifies"] not in node["depends_on"]:
                raise _refuse(name, "depends_on",
                              f"must list {node['verifies']!r}, the node it verifies")
    plan = []
    for node in nodes:
        entry = {"id": node["id"], "label": node["label"], "depends_on": node["depends_on"],
                 "required_capabilities": node["capabilities"],
                 "failure_policy": node["failure_policy"], "max_retries": node["max_retries"],
                 "max_regenerations": node["max_regenerations"],
                 "attach_dep_artifacts": node["attach_dep_artifacts"]}
        entry.update({key: node[key] for key in ("verifies", "timeout_s") if node[key] is not None})
        entry.update({"metadata": dict(node["metadata"])} if node["metadata"] else {})
        entry.update({"agent": node["profile"]} if node["profile"] is not None else {})
        plan.append(entry)
    try:
        _build_model_graph(
            {"nodes": plan}, max_nodes=limits.max_nodes, max_depth=limits.max_depth,
            max_retries=limits.max_retries, max_regenerations=limits.max_regenerations,
            max_gates=limits.max_gates, min_timeout_s=limits.min_timeout_s)
    except ValueError as error:
        raise TemplateValidationError(f"template blueprint breaks a model-plan rule: {error}",
                                      field="blueprint") from None


def _check_parent(
    record: TemplateRecord, nodes: list[dict[str, Any]], parent: TemplateRecord,
) -> None:
    """Refuse a candidate that guarantees less than *parent* does."""
    if record.parent_version not in (None, parent.template_version):
        raise _refuse(None, "parent_version", f"names {record.parent_version}, not "
                                              f"{parent.template_version}, the parent it is "
                                              "compared with")
    by_id = {node["id"]: node for node in nodes}
    parent_by_id = {node["id"]: node for node in _plain(parent.blueprint)}
    guarded = []
    for gate, old in parent_by_id.items():
        target = old["verifies"]
        if target is None:
            continue
        new = by_id.get(gate) or {"verifies": None}
        if new["verifies"] is None:
            raise _refuse(gate, "verifies", f"no longer gates {target!r}")
        if new["verifies"] != target:
            raise _refuse(gate, "verifies", f"gates {new['verifies']!r} instead of {target!r}")
        if new["max_regenerations"] < old["max_regenerations"]:
            raise _refuse(gate, "max_regenerations", f"is {new['max_regenerations']}, below the "
                                                     f"parent's {old['max_regenerations']}")
        guarded += [old, parent_by_id[target]]
    for old in guarded:
        # A gate or target that fails is skipped under 'skip', and the run
        # goes on without the verdict; 'halt' and 'retry' stop it.
        policy = by_id[old["id"]]["failure_policy"]
        if policy == FailurePolicy.SKIP.value and old["failure_policy"] != policy:
            raise _refuse(old["id"], "failure_policy",
                          f"relaxes the parent's {old['failure_policy']!r} to {policy!r}")

    def validators(of: TemplateRecord) -> dict[str, set[bytes]]:
        found: dict[str, set[bytes]] = {}
        for output in of.expected_outputs:
            found.setdefault(output["node"], set()).update(
                canonical_bytes(_plain(item)) for item in output["validators"])
        return found

    kept = validators(record)
    for node, required in validators(parent).items():
        for missing in sorted(required - kept.get(node, set())):
            raise _refuse(node, "expected_outputs.validators",
                          f"drops the validator {json.loads(missing)['name']!r}")
    for key, items in (parent.prerequisites or {}).items():
        kept_items = {canonical_bytes(_plain(item))
                      for item in (record.prerequisites or {}).get(key, ())}
        for item in items:
            if canonical_bytes(_plain(item)) not in kept_items:
                label = item if isinstance(item, str) else item["name"]
                raise _refuse(None, f"prerequisites.{key}", f"drops {label!r}, which the parent "
                                                            "lists")


def validate_template_record(
    record: TemplateRecord | Mapping[str, Any],
    *,
    parent: TemplateRecord | None = None,
    limits: TemplateLimits | None = None,
) -> TemplateRecord:
    """Check a template record, or its JSON data, as an untrusted candidate.

    Returns the record. Raises :class:`TemplateValidationError`, naming the
    node and field at fault, when the candidate:

    - holds, at any depth, a key that would grant authority: ``mcp_servers``,
      ``model``, ``env``, ``command``, ``url``, ``tools``, ``allowed_tools``,
      ``acceptance``, ``evaluator`` or ``budget``, except the record's own
      fields that grant nothing (the ``tools`` prerequisites expect and the
      ``acceptance`` decision a source run got), or is data that
      ``TemplateRecord.from_dict`` refuses;
    - breaks *limits*, by default the limits on a generated plan
      (:class:`TemplateLimits`): its blueprint, written as a generated plan,
      must pass the checks ``smythe.loader.build_graph_from_model_output``
      makes, and each gate must depend directly on the node it verifies;
    - given a *parent*, guarantees less than the parent does: a gate is
      dropped or verifies another node, a gate's ``max_regenerations`` is
      lowered, a gate or its target is relaxed to ``failure_policy`` skip,
      or an expected-output validator or a prerequisite is removed. The
      record's ``parent_version``, if set, must be the parent's version.

    Developer code is trusted, so ``distill_template`` and templates a
    developer builds are not checked; records are.
    """
    if limits is None:
        limits = TemplateLimits()
    if not isinstance(limits, TemplateLimits):
        raise TypeError("limits must be a TemplateLimits")
    if parent is not None and not isinstance(parent, TemplateRecord):
        raise TypeError("parent must be a TemplateRecord")
    record = _candidate(record)
    nodes = _plain(record.blueprint)
    _check_limits(nodes, limits)
    if parent is not None:
        _check_parent(record, nodes, parent)
    return record


def template_target(name: str) -> str:
    """The policy registry target ``planner.template.<name>`` of template *name*.

    Its policy maps the target to the template version in effect. The name
    must be one dotted segment: a letter, then up to 63 letters, digits,
    ``-`` or ``_``; any other raises ``ValueError``.
    """
    if not isinstance(name, str) or _TARGET_NAME.fullmatch(name) is None:
        raise ValueError(f"template name {name!r} must be one dotted segment to have a "
                         "policy registry target")
    return f"planner.template.{name}"


class TemplateCatalog:
    """Template records kept as append-only files under ``root``.

    ``records/<record_sha256>.json`` holds a record's canonical JSON, so each
    record file is named by the SHA-256 of its bytes, and
    ``versions/<template_version>`` holds the ``record_sha256`` of the one
    record stored for that version. ``layout.json`` holds the catalog's
    layout, :data:`TEMPLATE_CATALOG_LAYOUT`; a catalog without it was written
    in layout 1, and one in another layout is refused. Files are published
    whole, never replace an existing file and are never deleted; new
    directories and files are owner-only where the platform supports it.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def put(
        self,
        record: TemplateRecord | Mapping[str, Any],
        *,
        parent: TemplateRecord | None = None,
        limits: TemplateLimits | None = None,
    ) -> str:
        """Validate *record*, a record or its JSON data, store it, and return its version.

        :func:`validate_template_record` checks it first, under *limits*
        and against *parent*: by default the record this catalog holds for
        the candidate's ``parent_version``, when it holds one. A refused
        record raises :class:`TemplateValidationError` and writes nothing.

        Storing a record the catalog already holds writes no record or
        version file. A different record for a stored version, such as one
        with other source runs, raises :class:`TemplateRecordError` and the
        stored one is kept.
        """
        record = _candidate(record)
        if parent is None and record.parent_version is not None:
            try:
                parent = self.get(record.parent_version)
            except KeyError:
                pass  # A parent this catalog does not hold is not compared.
        record = validate_template_record(record, parent=parent, limits=limits)
        self._check_layout()
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
        # Last, so a put interrupted before it leaves a layout 1 catalog.
        self._publish("", _LAYOUT_FILE, _LAYOUT, "the template catalog layout file changed")
        return version

    def get(self, version: str) -> TemplateRecord:
        """The record stored for *version*, read and checked again.

        Raises:
            KeyError: No record is stored for *version*.
            TemplateRecordError: The stored files do not hold this version's
                record, for example because one was edited, or the catalog
                has another layout.
        """
        self._check_layout()
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
        self._check_layout()
        try:
            with os.scandir(self.root / "versions") as entries:
                found = sorted(entry.name for entry in entries if _HEX.fullmatch(entry.name))
        except FileNotFoundError:
            return []
        return [version for version in found if self.get(version).name == name]

    def active_templates(
        self, registry: Any, names: Sequence[str], *,
        environment: TemplateEnvironment | None = None, strict: bool = False,
    ) -> list[SubGraphTemplate]:
        """The version of each template in *names* that *registry* has in effect, for ordinary runs.

        *registry* is a ``smythe.optimize.PolicyRegistry``, read and never
        written. Each name's target (:func:`template_target`) must map
        itself, and nothing else, to a version this catalog holds of a
        record with that name. Returns ``record.to_template()`` for each, in
        the order of *names*; a retired template has no version in effect
        and is left out. A ``ConstrainedArchitect`` keeps the menu it is
        built with, so build a new one to offer later activations.

        With an *environment*, a version that does not apply there
        (:func:`check_applicability`) is left out too, and a warning on the
        ``smythe.templates`` logger gives the reasons; with *strict*, it
        raises :class:`TemplateApplicabilityError` instead.

        Raises:
            KeyError: A name has never had a version in effect, or the
                catalog does not hold the version in effect.
            TemplateRecordError: The policy in effect holds anything but a
                version of that template, or the stored record is invalid.
            TemplateApplicabilityError: With *strict*, a version in effect
                does not apply in *environment*.
        """
        return [record.to_template()
                for record in self._offered(registry, names, environment, strict)]

    def active_local_templates(
        self, registry: Any, names: Sequence[str], *,
        environment: TemplateEnvironment | None = None, strict: bool = False,
    ) -> list[SubGraphTemplate]:
        """:meth:`active_templates` declared for a durable run.

        Each builder is ``record.local_only()``, so the run's recipe pins the
        versions in effect when it starts, and a later activation or
        rollback never changes the run; :func:`pinned_templates` rebuilds
        them to resume it.
        """
        return [SubGraphTemplate(record.name, record.description, record.local_only())
                for record in self._offered(registry, names, environment, strict)]

    def _active(self, registry: Any, names: Sequence[str]) -> list[TemplateRecord]:
        if isinstance(names, str):
            raise TypeError("names must be a sequence of template names, not one string")
        records = []
        for name in names:
            target = template_target(name)
            active = registry.resolve(target)
            if active is None:
                if not registry.history(target):
                    raise KeyError(f"template {name!r} has no version activated in the registry")
                continue  # Retired: no version is in effect.
            version = active.policy.get(target)
            if set(active.policy) != {target} or not isinstance(version, str):
                raise TemplateRecordError(
                    f"the policy in effect for {target} must hold only its template version")
            records.append(self._named(name, version))
        return records

    def _offered(
        self, registry: Any, names: Sequence[str],
        environment: TemplateEnvironment | None = None, strict: bool = False,
    ) -> list[TemplateRecord]:
        records = self._active(registry, names)
        if environment is None:
            return records
        offered = []
        for record in records:
            report = check_applicability(record, environment)
            if report.valid:
                offered.append(record)
            elif strict:
                raise TemplateApplicabilityError(record.name, record.template_version, report)
            else:
                logger.warning("Template %r version %s is not offered: %s", record.name,
                               record.template_version, "; ".join(report.details))
        return offered

    def _named(self, name: str, version: str) -> TemplateRecord:
        """The record stored for *version*, which must be a version of template *name*."""
        try:
            record = self.get(version)
        except KeyError:
            raise KeyError(f"the catalog does not hold version {version} of template "
                           f"{name!r}") from None
        if record.name != name:
            raise TemplateRecordError(
                f"template version {version} is named {record.name!r}, not {name!r}")
        return record

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

    def _check_layout(self) -> None:
        try:
            stored = self._read("", _LAYOUT_FILE, 4096)
        except KeyError:
            return  # Written before the layout file existed: layout 1.
        if stored != _LAYOUT:
            try:
                layout = json.loads(stored).get("layout")
            except (ValueError, AttributeError):
                layout = None
            raise TemplateRecordError(f"the template catalog has layout {layout!r}; "
                                      f"this release reads layout {TEMPLATE_CATALOG_LAYOUT}")

    def _digest(self, version: str) -> str:
        if not isinstance(version, str) or _HEX.fullmatch(version) is None:
            raise TemplateRecordError("a template version is 64 lowercase hexadecimal characters")
        digest = self._read("versions", version, 64).decode("ascii", "replace")
        if _HEX.fullmatch(digest) is None:
            raise TemplateRecordError(f"template version {version} does not name a record digest")
        return digest

    def _read(self, folder: str, name: str, limit: int) -> bytes:
        where = f"{folder}/{name}".lstrip("/")
        try:
            path = _regular_path(self.root / folder, name)
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
        except FileNotFoundError:
            raise KeyError(name) from None
        except ValueError:
            raise TemplateRecordError(
                f"template catalog file {where} is not a regular file") from None
        if len(data) > limit:
            raise TemplateRecordError(f"template catalog file {where} is too large")
        return data


def pinned_templates(config: Mapping[str, Any], catalog: TemplateCatalog) -> list[SubGraphTemplate]:
    """The record templates a saved durable recipe pinned, rebuilt from *catalog*.

    *config* is a durable run's saved recipe,
    ``SQLiteWorkflowStore.load_run(run_id)["config"]``, whose architect is a
    ``ConstrainedArchitect``, or the description of such an architect in a
    recipe, such as a router tier's. Each template keeps the recipe's name
    and description, and its builder is ``record.local_only()`` for the
    exact version the recipe names. A ``ConstrainedArchitect`` given them
    and the run's other settings describes the same recipe, so resume
    continues the run on those versions, whichever version is in effect now.

    Raises:
        KeyError: The catalog does not hold a pinned version.
        TemplateRecordError: *config* holds no ``ConstrainedArchitect`` menu,
            a template's builder is not a record's ``local_only()``
            declaration, or the stored record does not match it.
    """
    architect: Any = config
    if isinstance(architect, Mapping) and architect.get("type") == "text_workflow":
        components = architect.get("components")
        architect = components.get("architect") if isinstance(components, Mapping) else None
    menu = (architect.get("templates") if isinstance(architect, Mapping)
            and architect.get("type") == "constrained_architect" else None)
    if not isinstance(menu, list):
        raise TemplateRecordError("the recipe's architect is not a ConstrainedArchitect with a "
                                  "template menu")
    templates = []
    for entry in menu:
        entry = entry if isinstance(entry, Mapping) else {}
        name, description, builder = (entry.get(key) for key in ("name", "description", "builder"))
        identity = builder.get("identity") if isinstance(builder, Mapping) else None
        if (not isinstance(name, str) or not isinstance(description, str)
                or not isinstance(identity, str) or not identity.startswith("template:")
                or (builder.get("type"), builder.get("role")) != ("local_only", "template_builder")):
            raise TemplateRecordError(f"template {name!r} in the recipe was not built from a "
                                      "template record")
        declaration = catalog._named(identity.removeprefix("template:"),
                                     builder.get("version")).local_only()
        if declaration.workflow_description() != dict(builder):
            raise TemplateRecordError(f"template {name!r} in the recipe does not match the "
                                      "stored record's declaration")
        templates.append(SubGraphTemplate(name, description, declaration))
    return templates


# Promotion from planner history. The decisions that make a run of a
# structure negative evidence, and the row fields a failure example's
# evidence digest covers: never text such as a goal or an error message.
_NOT_ACCEPTED = frozenset({"rejected", "needs_review", "indeterminate"})
_EVIDENCE_KEYS = ("execution_id", "structure_sha256", "acceptance", "verification",
                  "rejected_output_sha256", "failure_category", "failure_type")


def _min_accepted(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"min_accepted must be a positive integer, got {value!r}")
    return value


def _accepted(row: ExecutionOutcome) -> bool:
    """Whether *row* names a run of a structure whose output was accepted and that did not fail."""
    return (row.acceptance == AcceptanceDecision.ACCEPTED.value and row.failure_category is None
            and row.execution_id is not None and row.structure_sha256 is not None)


def repeated_structures(
    memory: PlannerMemory, *, min_accepted: int,
) -> dict[str, list[ExecutionOutcome]]:
    """The graph structures at least *min_accepted* accepted runs in *memory*'s history share.

    A run counts when its row records ``acceptance`` ``"accepted"``, an
    ``execution_id`` and a ``structure_sha256``, and no
    ``failure_category``; ``success`` never counts, and each execution id
    counts once. Returns each such structure digest, in the order its first
    accepted run was recorded, with the first row of each of its accepted
    runs. The history file is only read.
    """
    _min_accepted(min_accepted)
    groups: dict[str, dict[str, ExecutionOutcome]] = {}
    for row in memory.iter_outcomes():
        if _accepted(row):
            groups.setdefault(row.structure_sha256, {}).setdefault(row.execution_id, row)
    return {digest: list(runs.values()) for digest, runs in groups.items()
            if len(runs) >= min_accepted}


def candidate_from_graph(
    graph: ExecutionGraph,
    *,
    rows: Iterable[ExecutionOutcome],
    name: str,
    min_accepted: int = 1,
    description: str | None = None,
    registry: Registry | None = None,
    require_success: bool = True,
    source_task: Task | None = None,
    environment: TemplateEnvironment | None = None,
) -> TemplateRecord:
    """A candidate template record for *graph*, with the evidence *rows* hold for its structure.

    *rows* are planner history outcomes, such as ``memory.iter_outcomes()``;
    only those whose ``structure_sha256`` is *graph*'s are read. Each
    accepted run, as :func:`repeated_structures` counts them, becomes a
    source run. Each other row with a ``failure_category``, or with
    ``acceptance`` ``"rejected"``, ``"needs_review"`` or
    ``"indeterminate"``, becomes a failure example: that category, or else
    the decision, and the SHA-256 of the row's evidence as canonical JSON,
    its ``execution_id``, ``structure_sha256``, ``acceptance``,
    ``verification``, ``rejected_output_sha256``, ``failure_category`` and
    ``failure_type`` that are set. No text from a row, such as its goal or a
    failure excerpt, is kept. Rows with equal evidence give one example,
    and the latest :data:`MAX_FAILURE_EXAMPLES` are kept.

    With an *environment*, the one the runs used, the candidate records
    what it relied on, so :func:`check_applicability` can tell when that
    changes: as prerequisites, the capabilities that nodes without an agent
    profile require and an agent offered, and every skill, tool and
    validator; and as ``proven_with``, the digests of the model ids and MCP
    allowlists.

    The other arguments are those of :func:`distill_record`, whose review
    the candidate keeps. Nothing is stored or activated: store the
    candidate with :meth:`TemplateCatalog.put` and activate it through a
    reviewed decision.

    Raises:
        ValueError: *min_accepted* is not a positive integer.
        TypeError: A row is not an ``ExecutionOutcome``, or *environment*
            is not a :class:`TemplateEnvironment`.
        DistillationError: For the reasons ``distill_record`` gives, when
            *graph*'s structure has no digest, or when *rows* hold fewer
            than *min_accepted* accepted runs of it.
    """
    _min_accepted(min_accepted)
    if environment is not None and not isinstance(environment, TemplateEnvironment):
        raise TypeError("environment must be a TemplateEnvironment")
    try:
        structure = structure_sha256(graph)
    except (TypeError, ValueError) as error:
        raise DistillationError(f"cannot digest this graph's structure: {error}") from None
    runs: dict[str, dict[str, str]] = {}
    failures: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, ExecutionOutcome):
            raise TypeError("rows must be ExecutionOutcome values, such as "
                            "PlannerMemory.iter_outcomes() returns")
        if row.structure_sha256 != structure:
            continue
        if _accepted(row):
            runs.setdefault(row.execution_id, {"execution_id": row.execution_id,
                                               "structure_sha256": structure,
                                               "acceptance": row.acceptance})
        elif row.failure_category is not None or row.acceptance in _NOT_ACCEPTED:
            digest = sha256_hex({key: getattr(row, key) for key in _EVIDENCE_KEYS
                                 if getattr(row, key) is not None})
            failures.pop(digest, None)  # Kept at its latest place.
            failures[digest] = {"category": row.failure_category or row.acceptance,
                                "evidence_sha256": digest}
    if len(runs) < min_accepted:
        raise DistillationError(f"cannot build a candidate: the rows hold {len(runs)} accepted "
                                f"runs of this structure, fewer than {min_accepted}")
    record = distill_record(graph, name=name, description=description, registry=registry,
                            require_success=require_success, source_task=source_task)
    pinned: dict[str, Any] = {}
    if environment is not None:
        required = {capability for node in _plain(record.blueprint) if node["profile"] is None
                    for capability in node["capabilities"]}
        pinned = {"prerequisites": {
            "capabilities": list(required & environment.capabilities),
            "skills": [{"name": name, "version": version}
                       for name, version in environment.skills.items()],
            "tools": [dict(tool) for tool in environment.tools.values()],
            "validators": [identity.to_dict() for identity in environment.validators],
        }, "proven_with": environment._proven_with()}
    return replace(record, source_runs=list(runs.values()),
                   failure_examples=list(failures.values())[-MAX_FAILURE_EXAMPLES:], **pinned)


class TemplateArchitect(DeterministicArchitect):
    """Builds one template record's graph in local code, with no planning call.

    ``record.as_architect()`` returns one. Each plan is the graph a
    ``ConstrainedArchitect`` composes when its model selects the template
    once: node ids prefixed ``<name>-0-``, labels pointed at the task, a new
    registry of the record's agents, and ``graph.planning["templates"]``
    naming the record's version. A durable run takes :meth:`local_only`.
    """

    def __init__(self, record: TemplateRecord) -> None:
        if not isinstance(record, TemplateRecord):
            raise TypeError("a TemplateArchitect builds a TemplateRecord")
        self.record = record

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        # Composed as a selection is, so the two build the same graph.
        composer = ConstrainedArchitect(None, [self.record.to_template()],  # type: ignore[arg-type]
                                        max_nodes=len(self.record.blueprint))
        return composer._compose([{"template": self.record.name}], task)

    def local_only(self) -> LocalOnly:
        """This architect declared for a durable run, ``LocalOnly(role="architect")``.

        Its identity is ``template:<name>`` and its version the record's
        ``template_version``, so the run's recipe pins this exact template,
        and resuming with another version raises ``WorkflowConflictError``.
        """
        record = self.record
        return LocalOnly(lambda: TemplateArchitect(record), f"template:{record.name}",
                         record.template_version)


# Applicability: the reason codes check_applicability gives, sorted.
APPLICABILITY_REASONS = ("capability_removed", "mcp_allowlist_changed", "model_changed",
                         "skill_changed", "skill_removed", "tool_changed", "tool_removed",
                         "validator_set_changed")


@dataclass(frozen=True)
class ApplicabilityReport:
    """Whether a template record applies in an environment, and why not.

    ``reasons`` holds the sorted codes of :data:`APPLICABILITY_REASONS`
    behind any problem, each once, and ``details`` one sentence per problem.
    """

    valid: bool
    reasons: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


class TemplateApplicabilityError(PicklableError, ValueError):
    """A template version in effect does not apply in the environment a run would use.

    ``name`` and ``version`` identify it, and ``report`` says why. It
    survives pickle and copy with those attributes.
    """

    def __init__(self, name: str, version: str, report: ApplicabilityReport) -> None:
        super().__init__(f"template {name!r} version {version} does not apply: "
                         f"{'; '.join(report.details)}")
        self.name, self.version, self.report = name, version, report


def _items(values: Any, what: str) -> list[Any]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a list")
    return list(values)


def _names(values: Any, what: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a collection of names")
    return frozenset(_name(value, what) for value in values)


@dataclass(frozen=True, kw_only=True)
class TemplateEnvironment:
    """What a run would use now, for :func:`check_applicability` to compare a record with.

    ``capabilities`` are the capability tags its agents offer; ``skills``
    maps each skill's name to its version, for an Agent Skill the
    ``sha256:`` digest of its ``SKILL.md``; ``tools`` lists each tool as
    ``LocalToolRuntime.describe()`` does (name, version and
    ``schema_sha256``); ``mcp_allowlists`` maps each MCP server's name to
    its allowed tools, or None for all; ``validators`` holds the acceptance
    validators' identities; and ``models`` the model ids. :meth:`capture`
    reads them from live objects. Every part is read-only.
    """

    capabilities: Iterable[str] = frozenset()
    skills: Mapping[str, str] = field(default_factory=dict)
    tools: Iterable[Mapping[str, str]] = ()
    mcp_allowlists: Mapping[str, Sequence[str] | None] = field(default_factory=dict)
    validators: Iterable[ValidatorIdentity] = ()
    models: Iterable[str] = frozenset()

    def __post_init__(self) -> None:
        tools: dict[str, Mapping[str, str]] = {}
        for tool in _items(self.tools, "environment tools"):
            tool = _TOOL(_plain(tool), "environment tool")
            if tool["name"] in tools:
                raise ValueError(f"environment tool {tool['name']!r} is listed more than once")
            tools[tool["name"]] = MappingProxyType(tool)
        allowlists = {}
        for server, allowed in dict(self.mcp_allowlists).items():
            allowlists[_name(server, "MCP server name")] = (
                None if allowed is None else tuple(sorted(_names(allowed, "allowed MCP tool"))))
        validators = _items(self.validators, "environment validators")
        if not all(isinstance(item, ValidatorIdentity) for item in validators):
            raise TypeError("environment validators must be ValidatorIdentity values")
        for key, value in (
            ("capabilities", _names(self.capabilities, "environment capability")),
            ("skills", MappingProxyType({_name(name, "skill name"): _name(version, "skill version")
                                         for name, version in dict(self.skills).items()})),
            ("tools", MappingProxyType(tools)), ("mcp_allowlists", MappingProxyType(allowlists)),
            ("validators", frozenset(validators)),
            ("models", _names(self.models, "environment model")),
        ):
            object.__setattr__(self, key, value)

    @classmethod
    def capture(
        cls, *, registry: Registry | None = None, skill_provider: Any = None,
        tool_runtime: Any = None, validators: Iterable[Any] = (), models: Iterable[str] = (),
    ) -> TemplateEnvironment:
        """Snapshot what a run would use now.

        From *registry*: each agent's effective capabilities, leaving out a
        blank tag; each skill with a version that *skill_provider*, by
        default the registry's own, lists for the agent, since a change to
        an unversioned skill cannot be seen; and the ``allowed_tools`` of
        each MCP server on its profile, merged by server name, where None,
        every tool, wins. From *tool_runtime*, a ``LocalToolRuntime`` or a
        ``CompositeToolRuntime`` holding them, ``describe()``. *validators*
        are acceptance validators or their identities, such as
        ``AcceptancePolicy.validator_identities``, and *models* the model ids
        a run would use.
        """
        if skill_provider is None:
            skill_provider = getattr(registry, "_skill_provider", None)
        capabilities: set[str] = set()
        skills: dict[str, str] = {}
        allowlists: dict[str, list[str] | None] = {}
        for agent in registry.list_agents() if registry is not None else ():
            capabilities.update(tag for tag in registry.effective_capabilities(agent)
                                if isinstance(tag, str) and tag.strip())
            refs = skill_provider.list_agent_skills(agent.id) if skill_provider is not None else ()
            skills.update((ref.name, ref.version) for ref in refs
                          if isinstance(ref.version, str) and ref.version.strip())
            for server in agent.profile.mcp_servers:
                held = allowlists.get(server.name, [])
                allowlists[server.name] = (None if held is None or server.allowed_tools is None
                                           else sorted({*held, *server.allowed_tools}))
        return cls(capabilities=capabilities, skills=skills, tools=_described(tool_runtime),
                   mcp_allowlists=allowlists, models=models,
                   validators=[getattr(item, "identity", item) for item in validators])

    def _proven_with(self) -> dict[str, str]:
        """The ``proven_with`` section of a candidate whose runs used this environment."""
        return {"models_sha256": sha256_hex(sorted(self.models)),
                "mcp_allowlists_sha256": sha256_hex({
                    name: None if allowed is None else list(allowed)
                    for name, allowed in self.mcp_allowlists.items()})}


def _described(runtime: Any) -> list[Mapping[str, str]]:
    """``describe()`` of a local tool runtime, or of each one a composite runtime holds."""
    describe = getattr(runtime, "describe", None)
    if callable(describe):
        return list(describe())
    return [tool for child in getattr(runtime, "_runtimes", ()) for tool in _described(child)]


def check_applicability(
    record: TemplateRecord, environment: TemplateEnvironment,
) -> ApplicabilityReport:
    """Whether *record* applies in *environment*, and the reasons it does not.

    Each capability in the record's prerequisites must be one an agent
    offers (``capability_removed``). Each skill and tool there must be
    available at the same version, and a tool with the same input schema
    digest (``skill_removed``, ``skill_changed``, ``tool_removed``,
    ``tool_changed``). Each validator identity the prerequisites or an
    expected output names must be in use (``validator_set_changed``). A
    record with ``proven_with`` applies only where the run uses the same
    model ids (``model_changed``) and MCP allowlists
    (``mcp_allowlist_changed``) as the runs it was learned from. A record
    without prerequisites or ``proven_with`` applies anywhere.
    """
    if not isinstance(record, TemplateRecord) or not isinstance(environment, TemplateEnvironment):
        raise TypeError("check_applicability takes a TemplateRecord and a TemplateEnvironment")
    found: list[tuple[str, str]] = []
    needs = _plain(record.prerequisites or {})
    for capability in needs.get("capabilities", ()):
        if capability not in environment.capabilities:
            found.append(("capability_removed", f"no agent offers capability {capability!r}"))
    available = {
        "skill": {name: {"name": name, "version": version}
                  for name, version in environment.skills.items()},
        "tool": {name: dict(tool) for name, tool in environment.tools.items()},
    }
    for kind, current in available.items():
        for item in needs.get(f"{kind}s", ()):
            now = current.get(item["name"])
            if now is None:
                found.append((f"{kind}_removed", f"{kind} {item['name']!r} is not available"))
            elif now != item:
                found.append((f"{kind}_changed", f"{kind} {item['name']!r} is not the version "
                                                 "the template relies on"))
    named = [*needs.get("validators", ()),
             *(item for output in _plain(record.expected_outputs) for item in output["validators"])]
    for item in named:
        if ValidatorIdentity.from_dict(item) not in environment.validators:
            found.append(("validator_set_changed",
                          f"validator {item['name']!r} version {item['version']!r} with that "
                          "configuration is not in use"))
    pins = environment._proven_with()
    for key, code, what in (("models_sha256", "model_changed", "model ids"),
                            ("mcp_allowlists_sha256", "mcp_allowlist_changed",
                             "MCP server allowlists")):
        if record.proven_with and record.proven_with[key] != pins[key]:
            found.append((code, f"the run's {what} are not those the template was learned with"))
    return ApplicabilityReport(not found, tuple(sorted({code for code, _ in found})),
                               tuple(dict.fromkeys(detail for _, detail in found)))
