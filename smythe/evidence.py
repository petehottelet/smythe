"""Source evidence: versioned source and claim records, and how an artifact carries them.

Source evidence records what a deliverable's claims rest on. A
:class:`SourceRecord` identifies one source by its id and its version or
SHA-256 digest, and holds its retained text, or a reference to where the text
is kept, or a tombstone once the text is gone. A :class:`ClaimRecord` states
one claim, the passages that support and contradict it (each a source id and a
span quoted exactly from that source), its scope, the questions it leaves
unresolved and a proposed conclusion. :class:`SourceEvidence` holds an
artifact's source and claim records. Each record is versioned JSON:
``to_dict()`` is canonical and ``from_dict()`` rejects unknown versions and
keys. These records are distinct from the durable journal's response evidence,
which holds raw provider and read responses.

An artifact carries its evidence in one fenced block whose info string is
``source-evidence`` (:func:`render_source_evidence`); the rest of the artifact
is its body (:func:`read_source_evidence`).
:class:`~smythe.validators.SourceEvidenceValidator` checks the records
mechanically against the sources a run was given and the body. Whether a
passage really supports its claim is a judgement those checks do not make.

A durable text workflow whose strict acceptance policy has such a validator
journals the evidence its delivered output carries as a ``source_evidence``
operation (:func:`source_evidence_key`), and :func:`load_source_evidence`
reads it back, without source text unless the caller asks for it.

Removing a source from a durable run
(:meth:`~smythe.workflow.WorkflowRuntime.remove_source`) journals its
tombstone as a ``source_tombstone`` operation
(:func:`source_tombstone_key`), which :func:`load_source_tombstones` reads
back. From then on the run's records never yield the source's text.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import json
import re
from typing import Any

from smythe._canonical import sha256_hex
from smythe._redact import redact_account_identifiers
from smythe.context import (
    ContextSource, _identifier, sources_from_operation, sources_key, text_sha256,
)
from smythe.validation import _require_keys, _require_sha256

__all__ = [
    "EVIDENCE_BLOCK", "SOURCE_EVIDENCE_OPERATION", "SOURCE_EVIDENCE_VERSION",
    "SOURCE_TOMBSTONE_OPERATION", "ClaimPassage", "ClaimRecord", "SourceEvidence",
    "SourceRecord", "load_source_evidence", "load_source_tombstones", "read_source_evidence",
    "render_source_evidence", "source_evidence_inputs", "source_evidence_key",
    "source_evidence_result", "source_tombstone_key",
]

# The record version every source, claim and evidence record carries.
SOURCE_EVIDENCE_VERSION = 1
# The info string of the fenced block an artifact carries its evidence in.
EVIDENCE_BLOCK = "source-evidence"
# The durable journal operation kind that records a delivered output's evidence.
SOURCE_EVIDENCE_OPERATION = "source_evidence"
# The durable journal operation kind that records a source removed from a run.
SOURCE_TOMBSTONE_OPERATION = "source_tombstone"
RETAINED, REFERENCE, TOMBSTONED = "retained", "reference", "tombstoned"

_LINE = re.compile(r"[^\r\n]*")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")


def _text(value: Any, what: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not value.strip():
        raise ValueError(f"{what} must not be empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{what} must be valid Unicode") from error
    return str.__str__(value)


def _items(values: Any, kind: type, what: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a sequence of {kind.__name__}")
    values = tuple(values)
    if any(not isinstance(value, kind) for value in values):
        raise TypeError(f"{what} must be a sequence of {kind.__name__}")
    return values


def _record(data: Any, what: str, keys: frozenset[str]) -> dict[str, Any]:
    """Check a record's version before its keys, so a newer record is named as one."""
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be a JSON object")
    version = data.get("version")
    if type(version) is not int or version != SOURCE_EVIDENCE_VERSION:
        raise ValueError(f"Unsupported {what} version: {version!r}")
    return _require_keys(data, what, keys)


def _lists(data: dict[str, Any], what: str, *names: str) -> None:
    for name in names:
        if not isinstance(data[name], list):
            raise ValueError(f"{what} {name} must be a list")


def _loaded(build: Any, data: Any) -> Any:
    """Build a record from JSON data; a value of the wrong type raises ValueError too."""
    try:
        return build(data)
    except TypeError as error:
        raise ValueError(str(error)) from error


_SOURCE_KEYS = frozenset({"version", "source_id", "source_version", "sha256", "text",
                          "reference", "tombstone", "removed_at"})


@dataclass(frozen=True)
class SourceRecord:
    """One source a deliverable drew on: its identity and its text, a reference or a tombstone.

    ``sha256`` is the SHA-256 of the source text as UTF-8. A record with
    ``text`` retains the text, and its digest is computed from it. Without
    text, a record names where the text is kept (``reference``), or is a
    tombstone: ``tombstone`` says why the text is gone and ``removed_at``,
    optionally, when. A tombstone keeps the id, version and digest and
    nothing of the content. A record without text needs its version or
    digest, and a tombstone its digest.
    """

    source_id: str
    text: str | None = None
    source_version: str | None = None
    sha256: str | None = None
    reference: str | None = None
    tombstone: str | None = None
    removed_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if self.source_version is not None:
            object.__setattr__(self, "source_version",
                               _identifier(self.source_version, "source_version"))
        if self.sha256 is not None:
            _require_sha256(self.sha256, "source sha256")
        for name in ("reference", "tombstone"):
            object.__setattr__(self, name, _text(getattr(self, name), f"source {name}",
                                                 optional=True))
        if self.removed_at is not None:
            if self.tombstone is None:
                raise ValueError("removed_at belongs to a tombstone")
            object.__setattr__(self, "removed_at", _identifier(self.removed_at, "removed_at"))
        shown = repr(self.source_id)
        if self.text is not None:
            if not isinstance(self.text, str):
                raise TypeError("source text must be a string")
            try:
                self.text.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("source text must be valid Unicode") from error
            digest = text_sha256(self.text)
            if self.sha256 is not None and self.sha256 != digest:
                raise ValueError(f"source {shown} text does not match its sha256")
            if self.tombstone is not None:
                raise ValueError(f"tombstoned source {shown} cannot keep its text")
            object.__setattr__(self, "text", str.__str__(self.text))
            object.__setattr__(self, "sha256", digest)
        elif self.tombstone is not None:
            if self.sha256 is None:
                raise ValueError(f"tombstoned source {shown} must keep its sha256")
            if self.reference is not None:
                raise ValueError(f"tombstoned source {shown} cannot keep a reference")
        elif self.sha256 is None and self.source_version is None:
            raise ValueError(f"source {shown} needs its text, its sha256 or its version")

    @classmethod
    def from_context_source(cls, source: ContextSource) -> SourceRecord:
        """The retained record of a run's :class:`~smythe.context.ContextSource`."""
        if not isinstance(source, ContextSource):
            raise TypeError("from_context_source takes a ContextSource")
        return cls(source.source_id, source.text, source.version)

    @property
    def state(self) -> str:
        """``retained``, ``reference`` or ``tombstoned``."""
        if self.text is not None:
            return RETAINED
        return TOMBSTONED if self.tombstone is not None else REFERENCE

    def without_text(self) -> SourceRecord:
        """This record with its id, version, digest and reference, but no text."""
        return replace(self, text=None)

    def tombstoned(self, reason: str, *, removed_at: str | None = None) -> SourceRecord:
        """A tombstone for this source: why its text is gone and, optionally, when."""
        return SourceRecord(self.source_id, source_version=self.source_version,
                            sha256=self.sha256, tombstone=reason, removed_at=removed_at)

    def to_dict(self) -> dict[str, Any]:
        return {"version": SOURCE_EVIDENCE_VERSION, "source_id": self.source_id,
                "source_version": self.source_version, "sha256": self.sha256, "text": self.text,
                "reference": self.reference, "tombstone": self.tombstone,
                "removed_at": self.removed_at}

    def summary(self) -> dict[str, Any]:
        """:meth:`to_dict` with the text replaced by its UTF-8 size, and the record's state."""
        data = self.to_dict()
        text = data.pop("text")
        data["text_bytes"] = None if text is None else len(text.encode("utf-8"))
        data["state"] = self.state
        return data

    @classmethod
    def from_dict(cls, data: Any) -> SourceRecord:
        data = _record(data, "source record", _SOURCE_KEYS)
        return _loaded(lambda item: cls(
            item["source_id"], item["text"], item["source_version"], item["sha256"],
            item["reference"], item["tombstone"], item["removed_at"]), data)


@dataclass(frozen=True)
class ClaimPassage:
    """A span quoted exactly from one source, cited for or against a claim."""

    source_id: str
    quote: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "passage source_id"))
        object.__setattr__(self, "quote", _text(self.quote, "passage quote"))

    def to_dict(self) -> dict[str, str]:
        return {"source_id": self.source_id, "quote": self.quote}

    @classmethod
    def from_dict(cls, data: Any) -> ClaimPassage:
        _require_keys(data, "claim passage", frozenset({"source_id", "quote"}))
        return _loaded(lambda item: cls(item["source_id"], item["quote"]), data)


_CLAIM_KEYS = frozenset({"version", "claim_id", "text", "supporting", "contradicting", "scope",
                         "unresolved", "conclusion"})


@dataclass(frozen=True)
class ClaimRecord:
    """One claim, the passages for and against it, and what it leaves open.

    ``supporting`` and ``contradicting`` are :class:`ClaimPassage` values;
    ``scope`` says what the claim covers, ``unresolved`` lists the questions
    it leaves open and ``conclusion`` proposes one, each optional.
    """

    claim_id: str
    text: str
    supporting: tuple[ClaimPassage, ...] = ()
    contradicting: tuple[ClaimPassage, ...] = ()
    scope: str | None = None
    unresolved: tuple[str, ...] = ()
    conclusion: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "text", _text(self.text, "claim text"))
        for name in ("supporting", "contradicting"):
            object.__setattr__(self, name, _items(getattr(self, name), ClaimPassage,
                                                  f"claim {name}"))
        if (isinstance(self.unresolved, (str, bytes, Mapping))
                or not isinstance(self.unresolved, Iterable)):
            raise TypeError("claim unresolved must be a sequence of questions")
        object.__setattr__(self, "unresolved", tuple(
            _text(question, "unresolved question") for question in self.unresolved))
        for name in ("scope", "conclusion"):
            object.__setattr__(self, name, _text(getattr(self, name), f"claim {name}",
                                                 optional=True))

    def to_dict(self) -> dict[str, Any]:
        return {"version": SOURCE_EVIDENCE_VERSION, "claim_id": self.claim_id, "text": self.text,
                "supporting": [passage.to_dict() for passage in self.supporting],
                "contradicting": [passage.to_dict() for passage in self.contradicting],
                "scope": self.scope, "unresolved": list(self.unresolved),
                "conclusion": self.conclusion}

    @classmethod
    def from_dict(cls, data: Any) -> ClaimRecord:
        data = _record(data, "claim record", _CLAIM_KEYS)
        _lists(data, "claim record", "supporting", "contradicting", "unresolved")
        return _loaded(lambda item: cls(
            item["claim_id"], item["text"],
            tuple(ClaimPassage.from_dict(passage) for passage in item["supporting"]),
            tuple(ClaimPassage.from_dict(passage) for passage in item["contradicting"]),
            item["scope"], tuple(item["unresolved"]), item["conclusion"]), data)


@dataclass(frozen=True)
class SourceEvidence:
    """An artifact's source records and claim records, each id once.

    The records may cite a source they do not list, or name one the run was
    never given: :class:`~smythe.validators.SourceEvidenceValidator` reports
    that, so that the record of what an artifact claimed stays whole.
    """

    sources: tuple[SourceRecord, ...] = ()
    claims: tuple[ClaimRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _items(self.sources, SourceRecord, "evidence sources"))
        object.__setattr__(self, "claims", _items(self.claims, ClaimRecord, "evidence claims"))
        for what, ids in (("source", [item.source_id for item in self.sources]),
                          ("claim", [item.claim_id for item in self.claims])):
            if len(set(ids)) != len(ids):
                raise ValueError(f"evidence {what} ids must be distinct")

    @property
    def sha256(self) -> str:
        """The SHA-256 of :meth:`to_dict` as canonical JSON."""
        return sha256_hex(self.to_dict())

    def passages(self) -> Iterator[tuple[ClaimRecord, str, ClaimPassage]]:
        """Each claim's passages, as ``(claim, "supporting" or "contradicting", passage)``."""
        for claim in self.claims:
            for role in ("supporting", "contradicting"):
                for passage in getattr(claim, role):
                    yield claim, role, passage

    def to_dict(self) -> dict[str, Any]:
        return {"version": SOURCE_EVIDENCE_VERSION,
                "sources": [source.to_dict() for source in self.sources],
                "claims": [claim.to_dict() for claim in self.claims]}

    def summary(self) -> dict[str, Any]:
        """:meth:`to_dict` with each source's text replaced by its UTF-8 size."""
        data = self.to_dict()
        data["sources"] = [source.summary() for source in self.sources]
        return data

    @classmethod
    def from_dict(cls, data: Any) -> SourceEvidence:
        data = _record(data, "source evidence", frozenset({"version", "sources", "claims"}))
        _lists(data, "source evidence", "sources", "claims")
        return _loaded(lambda item: cls(
            tuple(SourceRecord.from_dict(source) for source in item["sources"]),
            tuple(ClaimRecord.from_dict(claim) for claim in item["claims"])), data)


def render_source_evidence(evidence: SourceEvidence) -> str:
    """*evidence* as the fenced ``source-evidence`` block an artifact carries.

    The block holds :meth:`SourceEvidence.to_dict` as indented JSON, so no
    line of it can close the fence early. Records meant for an artifact
    usually leave out source text (:meth:`SourceRecord.without_text`).
    """
    if not isinstance(evidence, SourceEvidence):
        raise TypeError("render_source_evidence takes a SourceEvidence")
    data = json.dumps(evidence.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    return f"```{EVIDENCE_BLOCK}\n{data}\n```"


def _lines(text: str) -> Iterator[tuple[int, int, int]]:
    """Each line's start, the end of its content and the end of its line break."""
    position, size = 0, len(text)
    while position < size:
        end = _LINE.match(text, position).end()
        stop = end + (2 if text.startswith("\r\n", end) else 1 if end < size else 0)
        yield position, end, stop
        position = stop


def _evidence_block(text: str) -> tuple[str | None, str]:
    """The JSON of *text*'s one evidence block and the text without it.

    Fences follow Markdown: an opening line of three or more backticks or
    tildes, indented at most three spaces, is closed by a line of at least as
    many of the same character. A fence line inside another fenced block is
    content.
    """
    blocks: list[tuple[int, int, int, int]] = []
    # The open fence's closing pattern, where its line starts and ends, and
    # whether it is an evidence block.
    fence: tuple[re.Pattern[str], int, int, bool] | None = None
    for start, end, stop in _lines(text):
        line = text[start:end]
        if fence is not None:
            closing, opened, content, evidence = fence
            if closing.fullmatch(line):
                if evidence:
                    blocks.append((opened, content, start, stop))
                fence = None
            continue
        opening = _FENCE.fullmatch(line)
        # A backtick fence's info string cannot hold a backtick.
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            marker = opening.group(1)
            closing = re.compile(f" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*")
            fence = (closing, start, stop, opening.group(2).strip(" \t") == EVIDENCE_BLOCK)
    if fence is not None and fence[3]:
        raise ValueError(f"the {EVIDENCE_BLOCK} block is not closed")
    if len(blocks) > 1:
        raise ValueError(f"the text has {len(blocks)} {EVIDENCE_BLOCK} blocks; it may have one")
    if not blocks:
        return None, text
    opened, content, closing, stop = blocks[0]
    return text[content:closing], text[:opened] + text[stop:]


def read_source_evidence(text: str) -> tuple[SourceEvidence | None, str]:
    """The source evidence an artifact's *text* carries, and the artifact's body.

    The body is the text without the evidence block. Text without a block
    gives ``(None, text)``. More than one block, a block that is not closed,
    or a block that is not one strict JSON :class:`SourceEvidence` record
    (a repeated key or a non-finite number is not strict) raises ValueError.
    """
    from smythe.validators import _strict_json

    if not isinstance(text, str):
        raise TypeError("read_source_evidence takes text")
    block, body = _evidence_block(text)
    if block is None:
        return None, body
    try:
        data = _strict_json(block)
    except RecursionError as error:
        raise ValueError(f"the {EVIDENCE_BLOCK} block is nested too deeply") from error
    except ValueError as error:
        raise ValueError(f"the {EVIDENCE_BLOCK} block is not strict JSON: {error}") from error
    try:
        return SourceEvidence.from_dict(data), body
    except RecursionError as error:
        raise ValueError(f"the {EVIDENCE_BLOCK} block is nested too deeply") from error


def source_evidence_inputs(output: str, context_sources: dict[str, Any] | None) -> dict[str, Any]:
    """What a ``source_evidence`` operation binds: the output's digest and the run's sources.

    *context_sources* is the ``{"revision", "sha256"}`` record of the source
    set a run with context assembly delivered with, and None for another run.
    """
    return {"output_sha256": text_sha256(output), "context_sources": context_sources}


def source_evidence_key(inputs: dict[str, Any]) -> str:
    """The journal key of a ``source_evidence`` operation: ``source-evidence/<sha256>``.

    The digest is of *inputs* (:func:`source_evidence_inputs`) as canonical JSON.
    """
    return "source-evidence/" + sha256_hex(inputs)


def source_evidence_result(output: str) -> dict[str, Any]:
    """The ``source_evidence`` operation result for a delivered *output*.

    ``source_evidence`` is the record :func:`read_source_evidence` reads, or
    None, and ``problem`` says why the output carries none.
    """
    try:
        evidence, _ = read_source_evidence(output)
    except ValueError as error:
        return {"source_evidence": None, "problem": redact_account_identifiers(str(error))[:300]}
    if evidence is None:
        return {"source_evidence": None, "problem": f"the output carries no {EVIDENCE_BLOCK} block"}
    return {"source_evidence": evidence.to_dict(), "problem": None}


def source_tombstone_key(source_id: str, sha256: str) -> str:
    """The journal key of a removed source's tombstone: ``source-tombstone/<sha256>``.

    The digest is of the source's id and the SHA-256 of its text as
    canonical JSON, so one text removed from a run has one tombstone.
    """
    return "source-tombstone/" + sha256_hex({"source_id": source_id, "sha256": sha256})


def _tombstone_operations(store: Any, run_id: str) -> list[tuple[SourceRecord, int | None]]:
    """Each tombstone a durable run committed, with the revision of the checkpoint that applied it.

    The revision is None for a tombstone that no checkpoint has applied yet.
    """
    found = []
    for operation in store.list_operations(run_id, SOURCE_TOMBSTONE_OPERATION):
        if operation["state"] == "started":
            continue
        key, result = operation["operation_key"], operation["result"]
        try:
            record = SourceRecord.from_dict(result.get("tombstone"))
            if (record.state != TOMBSTONED or result.keys() != {"tombstone"}
                    or operation["inputs"] != {"tombstone": record.to_dict()}
                    or key != source_tombstone_key(record.source_id, record.sha256)):
                raise ValueError("the record does not match its operation")
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"journal operation {key!r} is not a source tombstone") from error
        found.append((record, operation["applied_revision"]))
    return found


def load_source_tombstones(store: Any, run_id: str) -> tuple[SourceRecord, ...]:
    """The tombstones of the sources removed from a durable run, in the order they were applied.

    Each keeps the source's id, version and digest, why it was removed and
    when, and nothing of its text. Only reads are made. Raises ValueError for
    a record under a tombstone's kind that is not one.
    """
    return tuple(record for record, _ in _tombstone_operations(store, run_id))


def _removed_sources(store: Any, run_id: str) -> dict[tuple[str, str], SourceRecord]:
    """A durable run's tombstones, by the ``(source_id, sha256)`` of the text each removed."""
    return {(record.source_id, record.sha256): record for record in load_source_tombstones(store, run_id)}


def load_source_evidence(store: Any, run_id: str, *, include_text: bool = False) -> dict | None:
    """The source evidence a completed durable run delivered, or None if it recorded none.

    Returns the operation's key and state, the output digest, the run's
    source set (``context_sources``), the ``problem`` when the output carried
    no readable evidence, the claim records, and each source record's
    :meth:`~SourceRecord.summary` with ``available``: whether its text is
    retained, in the record or in the run's recorded sources with the same
    id, version and digest, and ``removed``: the reason and time of the
    tombstone of a source removed from the run, or None. A tombstoned or
    removed source is never available. Source text is left out unless
    *include_text* is True, which adds each available source's ``text``
    (None for one that is not).

    Raises ValueError when the journal's record under that key is not this
    run's source evidence.
    """
    saved = store.get_checkpoint(run_id)
    state = None if saved is None else saved["checkpoint"]
    if not state or state.get("status") != "completed" or not isinstance(state.get("output"), str):
        return None
    binding = state.get("context_sources")
    inputs = source_evidence_inputs(state["output"], binding)
    key = source_evidence_key(inputs)
    operation = store.load_operation(run_id, key)
    if operation is None:
        return None
    result = operation["result"]
    if (operation["kind"] != SOURCE_EVIDENCE_OPERATION or operation["inputs"] != inputs
            or not isinstance(result, dict) or result.keys() != {"source_evidence", "problem"}):
        raise ValueError(f"journal operation {key!r} is not this run's source evidence")
    evidence = (SourceEvidence() if result["source_evidence"] is None
                else SourceEvidence.from_dict(result["source_evidence"]))
    held: dict[str, ContextSource] = {}
    if isinstance(binding, dict) and type(binding.get("revision")) is int:
        recorded = store.load_operation(run_id, sources_key(binding["revision"]))
        if recorded is not None:
            held = {source.source_id: source
                    for source in sources_from_operation(recorded, binding["revision"])}
    removed = _removed_sources(store, run_id)
    sources = []
    for record in evidence.sources:
        source = held.get(record.source_id)
        digest = record.sha256
        if digest is None and source is not None and record.source_version in (None, source.version):
            digest = source.sha256
        # A removed source's text is never read again, from the record or the run.
        tombstone = removed.get((record.source_id, digest))
        text = None if tombstone is not None else record.text
        if (text is None and tombstone is None and record.state == REFERENCE
                and source is not None and record.sha256 in (None, source.sha256)
                and record.source_version in (None, source.version)):
            text = source.text
        item = record.summary()
        item["available"] = text is not None
        item["removed"] = None if tombstone is None else {"reason": tombstone.tombstone,
                                                          "removed_at": tombstone.removed_at}
        if include_text:
            item["text"] = text
        sources.append(item)
    return {"operation_key": key, "state": operation["state"],
            "output_sha256": inputs["output_sha256"], "context_sources": binding,
            "problem": result["problem"], "sources": sources,
            "claims": [claim.to_dict() for claim in evidence.claims]}
