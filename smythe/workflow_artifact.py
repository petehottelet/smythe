"""Local publication of a durable text workflow's delivered document.

A durable run given a ``renderer`` publishes its output after synthesis. The
renderer is registered, deterministic local code whose identity and version
the run's recipe records: the built-in :class:`MarkdownRenderer`, or a
component declared with ``LocalOnly(..., role="renderer")``. It turns the
synthesized text into a text document, and the document's UTF-8 bytes are
published once into the run's own directory, ``<artifact_dir>/<run id>/``,
which an owner marker binds to the run. No file there is ever replaced. The
document is the run's delivered output: a strict policy decides on exactly
those bytes, so its decision binds their SHA-256, and the run's checkpoint and
result hold the document's text.

Each publication is journaled as a ``publication`` operation keyed
``publication/<sequence>``, whose result is a :class:`PublishedArtifact`
record. :func:`current_artifact` is the one accessor for a run's current
publication, the delivery decision that covers it, and whether it is eligible
for delivery now: only with an ``accepted`` decision on the same bytes and a
file that still holds them. A missing or changed file is never eligible;
:meth:`~smythe.workflow.WorkflowRuntime.republish` publishes the saved bytes
again as a new file.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smythe._canonical import canonical_bytes
from smythe._publish import (
    MAX_ARTIFACT_BYTES, _regular_path, atomic_publish_bytes, claim_artifact_root,
)
from smythe.acceptance import (
    DeliveryAcceptance, _file_sha256, task_sha256, text_sha256,
)
from smythe.task import Task
from smythe.validation import _require_keys, _require_sha256
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowStateError

__all__ = [
    "ARTIFACT_RECORD_VERSION", "PUBLICATION_OPERATION", "CurrentArtifact", "MarkdownRenderer",
    "PublishedArtifact", "artifact_integrity", "current_artifact",
]

# The version every PublishedArtifact record carries.
ARTIFACT_RECORD_VERSION = 1
# The durable journal operation kind that records one publication.
PUBLICATION_OPERATION = "publication"
# How many numbered names a publication tries after the renderer's file name.
_MAX_NAMES = 1000
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.[a-z0-9]{1,16}")
# A published name: a renderer's file name, possibly numbered ("document-2.md").
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.[a-z0-9]{1,16}")
# Names Windows reserves for devices, whatever their extension.
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)),
                       *(f"LPT{n}" for n in range(1, 10))})
_MEDIA_TYPE = re.compile(r"text/[a-z0-9][a-z0-9.+-]{0,63}(; charset=utf-8)?")
_RECORD_KEYS = frozenset({"version", "sequence", "directory", "name", "sha256", "size_bytes",
                          "media_type", "renderer", "synthesis", "output_sha256"})
# What a publication's file is when it does not hold the published bytes.
_PROBLEMS = {"missing": "is missing", "changed": "no longer holds the bytes its record bound",
             "unsafe": "cannot be checked safely", "unreadable": "cannot be read"}


class MarkdownRenderer:
    """Render synthesized text as a Markdown document, deterministically.

    The document is a level-one heading holding the task's goal on one line,
    a blank line, then the text without trailing whitespace, and it ends with
    one newline. Without a task, or with an empty goal, it is the text alone.
    It is published as ``document.md``. The recipe records the renderer as
    ``{"type": "markdown_renderer", "version": 1}``.
    """

    filename = "document.md"
    media_type = "text/markdown; charset=utf-8"

    def render(self, output: str, task: Task | None) -> str:
        if not isinstance(output, str):
            raise TypeError("MarkdownRenderer renders text")
        goal = " ".join(task.goal.split()) if task is not None else ""
        heading = f"# {goal}\n\n" if goal else ""
        return heading + output.rstrip() + "\n"

    def workflow_description(self, **_: Any) -> dict[str, Any]:
        return {"type": "markdown_renderer", "version": 1}

    def workflow_providers(self) -> tuple[()]:
        return ()

    def bind_run(self, binding: Any) -> MarkdownRenderer:
        return MarkdownRenderer()


def renderer_settings(renderer: Any) -> tuple[str, str]:
    """A bound renderer's file name and media type.

    A ``LocalOnly`` renderer may set ``filename`` (default ``document.txt``)
    and ``media_type`` (default ``text/plain; charset=utf-8``): a simple name
    with an extension, and a ``text/*`` type whose only parameter is
    ``charset=utf-8``. Raises WorkflowBindingError for anything else.
    """
    component = getattr(renderer, "component", renderer)
    filename = getattr(component, "filename", "document.txt")
    media_type = getattr(component, "media_type", "text/plain; charset=utf-8")
    if (type(filename) is not str or _FILENAME.fullmatch(filename) is None
            or filename.split(".")[0].upper() in _RESERVED):
        raise WorkflowBindingError(
            f"renderer filename {filename!r} must be a simple name with an extension, such as "
            "'document.md'"
        )
    if type(media_type) is not str or _MEDIA_TYPE.fullmatch(media_type) is None:
        raise WorkflowBindingError(
            f"renderer media_type {media_type!r} must be a text/* type, such as "
            "'text/plain; charset=utf-8'"
        )
    return filename, media_type


def render_document(renderer: Any, output: str, task: Task | None) -> bytes:
    """The UTF-8 bytes of the document a bound *renderer* makes from *output*.

    ``render(output, task)`` must return text synchronously; the bytes must
    fit the publication limit. Raises WorkflowStateError otherwise.
    """
    document = renderer.render(output, task)
    if inspect.isawaitable(document):
        if inspect.iscoroutine(document):
            document.close()
        raise WorkflowStateError("Renderer returned an awaitable; render() must be synchronous")
    if type(document) is not str:
        raise WorkflowStateError(f"Renderer must return text, not {type(document).__name__}")
    try:
        data = document.encode("utf-8")
    except UnicodeEncodeError as error:
        raise WorkflowStateError("Renderer returned text that is not valid UTF-8") from error
    if len(data) > MAX_ARTIFACT_BYTES:
        raise WorkflowStateError(
            f"Rendered document exceeds the {MAX_ARTIFACT_BYTES}-byte publication limit")
    return data


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """One publication of a durable run's document, bound by its bytes.

    ``sequence`` is 0 for the run's first publication and one more for each
    republication. ``directory`` is the run's absolute artifact directory and
    ``name`` the file in it (:attr:`path`). ``sha256`` and ``size_bytes``
    bind the published bytes and ``media_type`` names their format.
    ``renderer`` is the recipe's description of the renderer that made them,
    ``synthesis`` the key of the synthesis operation it rendered, and
    ``output_sha256`` the digest of that synthesized text. :meth:`from_dict`
    rejects unknown versions and keys.
    """

    sequence: int
    directory: str
    name: str
    sha256: str
    size_bytes: int
    media_type: str
    renderer: dict[str, Any]
    synthesis: str
    output_sha256: str
    version: int = ARTIFACT_RECORD_VERSION

    def __post_init__(self) -> None:
        if self.version != ARTIFACT_RECORD_VERSION or type(self.version) is not int:
            raise ValueError(f"Unsupported published artifact record version {self.version!r}")
        for name in ("sequence", "size_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"published artifact {name} must be a non-negative integer")
        for name in ("directory", "name", "media_type", "synthesis"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"published artifact {name} must be a non-empty string")
        if not Path(self.directory).is_absolute() or _NAME.fullmatch(self.name) is None:
            raise ValueError("published artifact must name a file in an absolute directory")
        _require_sha256(self.sha256, "published artifact sha256")
        _require_sha256(self.output_sha256, "published artifact output_sha256")
        if not isinstance(self.renderer, dict):
            raise ValueError("published artifact renderer must be its recipe description")

    @property
    def path(self) -> Path:
        """Where the published file is."""
        return Path(self.directory) / self.name

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "sequence": self.sequence, "directory": self.directory,
                "name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes,
                "media_type": self.media_type, "renderer": self.renderer,
                "synthesis": self.synthesis, "output_sha256": self.output_sha256}

    @classmethod
    def from_dict(cls, data: Any) -> PublishedArtifact:
        """Read a record written by :meth:`to_dict`; raises ValueError for anything else."""
        _require_keys(data, "published artifact record", _RECORD_KEYS)
        return cls(sequence=data["sequence"], directory=data["directory"], name=data["name"],
                   sha256=data["sha256"], size_bytes=data["size_bytes"],
                   media_type=data["media_type"], renderer=data["renderer"],
                   synthesis=data["synthesis"], output_sha256=data["output_sha256"],
                   version=data["version"])


def owner_bytes(store_id: str, run_id: str) -> bytes:
    """The owner marker that binds a run's artifact directory to the run."""
    return canonical_bytes({"kind": "smythe.workflow.artifacts", "version": 1,
                            "store_id": store_id, "run_id": run_id})


def _holds(directory: Path, name: str, data: bytes) -> bool:
    """Whether *directory*'s regular file *name* holds exactly *data*."""
    try:
        observed = _file_sha256(_regular_path(directory, name), len(data))
    except (OSError, ValueError):
        return False
    return observed == hashlib.sha256(data).hexdigest()


def claim_directory(directory: Path, owner: bytes) -> None:
    """Bind a run's *directory* to the run's *owner* marker before anything is published there.

    The directory must be new or already the run's, and its filesystem must
    refuse to replace a file. Raises WorkflowStateError when the directory
    belongs to another run or holds files it does not own, and ``OSError``
    for a filesystem without hard links.
    """
    try:
        claim_artifact_root(directory, owner)
    except ValueError as error:
        raise WorkflowStateError(f"Cannot publish into {directory}: {error}") from error


def publish_bytes(directory: Path, filename: str, data: bytes) -> str:
    """Publish *data* in a run's claimed *directory* as *filename*; return the name it took.

    No file is replaced: when *filename* is taken, ``<stem>-2<suffix>`` is
    tried, then ``<stem>-3<suffix>``. A file that already holds exactly
    *data*, as one published by a process lost before it journaled the
    publication, is reused rather than duplicated.
    """
    base = Path(filename)
    for number in range(1, _MAX_NAMES + 2):
        name = filename if number == 1 else f"{base.stem}-{number}{base.suffix}"
        try:
            atomic_publish_bytes(directory / name, data)
        except FileExistsError:
            if _holds(directory, name, data):
                return name
            continue
        return name
    raise WorkflowStateError(f"Every name for {filename!r} is already taken in {directory}")


def artifact_integrity(record: PublishedArtifact) -> dict[str, str]:
    """Whether *record*'s file still holds its published bytes, as ``status`` and ``detail``.

    ``verified`` means a regular file, reached without symbolic links or
    junctions below the artifact directory's parent, with exactly the
    recorded size and SHA-256. Otherwise the status is ``missing``,
    ``changed``, ``unsafe`` or ``unreadable``. At most the recorded size plus
    one byte is read, and nothing is changed.
    """
    directory = Path(record.directory)
    try:
        path = _regular_path(directory.parent, f"{directory.name}/{record.name}")
        observed = _file_sha256(path, record.size_bytes)
    except FileNotFoundError:
        return {"status": "missing", "detail": "The published file is not present"}
    except ValueError as error:
        return {"status": "unsafe", "detail": str(error)}
    except OSError as error:
        return {"status": "unreadable",
                "detail": f"Cannot read the published file ({type(error).__name__})"}
    if observed != record.sha256:
        return {"status": "changed", "detail": "Size or SHA-256 differs from the published bytes"}
    return {"status": "verified", "detail": "Size and SHA-256 match the published bytes"}


def read_published(record: PublishedArtifact) -> bytes | None:
    """The bytes *record*'s file holds, or None unless they are exactly the published ones."""
    directory = Path(record.directory)
    try:
        path = _regular_path(directory.parent, f"{directory.name}/{record.name}")
        with path.open("rb") as stream:
            data = stream.read(record.size_bytes + 1)
    except (OSError, ValueError):
        return None
    return data if hashlib.sha256(data).hexdigest() == record.sha256 else None


def publication_key(sequence: int) -> str:
    """The journal key of a run's publication number *sequence*."""
    return f"publication/{sequence}"


def publications(store: Any, run_id: str) -> list[dict[str, Any]]:
    """A run's ``publication`` operations in sequence order; only reads are made."""
    found = []
    while (operation := store.load_operation(run_id, publication_key(len(found)))) is not None:
        found.append(operation)
    return found


def read_publication(operation: dict[str, Any], sequence: int) -> PublishedArtifact:
    """The record a completed publication operation holds, checked against its inputs."""
    inputs, result = operation["inputs"], operation["result"]
    try:
        if (operation["kind"] != PUBLICATION_OPERATION or not isinstance(result, dict)
                or result.keys() != {"artifact"}):
            raise ValueError("not a publication result")
        record = PublishedArtifact.from_dict(result["artifact"])
    except (TypeError, ValueError) as error:
        raise WorkflowStateError(f"Saved publication {sequence} is invalid") from error
    bound = ({"sequence": record.sequence, "republishes": inputs.get("republishes"),
              "sha256": record.sha256, "size_bytes": record.size_bytes}
             if "republishes" in inputs else
             {"sequence": record.sequence, "synthesis": record.synthesis,
              "output_sha256": record.output_sha256, "renderer": record.renderer})
    if record.sequence != sequence or bound != inputs:
        raise WorkflowStateError(f"Saved publication {sequence} does not match its operation")
    return record


def latest_publication(store: Any, run_id: str) -> PublishedArtifact | None:
    """The run's current publication: the one with the highest sequence a checkpoint applied."""
    latest = None
    for sequence, operation in enumerate(publications(store, run_id)):
        if operation["state"] == "applied":
            latest = read_publication(operation, sequence)
    return latest


def _current_decision(store: Any, run_id: str) -> DeliveryAcceptance | None:
    """The delivery decision that covers a durable run's delivered output now, or None.

    A run without a strict policy, or one that has not completed, has none.
    A completed run's decision is the record its checkpoint holds, made under
    the run's policy or, after a re-evaluation, under the policy that
    re-evaluation used. It is checked as resuming the run checks it: the
    saved output, task and source generations must be those it binds, and
    the journal must hold the same decision, applied. A record that no
    longer holds gives ``indeterminate`` under the record's policy, with the
    mismatches as its reasons. Only reads are made.
    """
    from smythe.checkpoint import graph_from_dict
    from smythe.task import task_from_dict
    from smythe.validation import _json_sha256
    from smythe.workflow import _decision_problems, _held_record, _read_saved_record

    run = store.load_run(run_id)
    policy = run["config"].get("acceptance")
    saved = store.get_checkpoint(run_id)
    if policy is None or saved is None or saved["checkpoint"].get("status") != "completed":
        return None
    state, task = saved["checkpoint"], task_from_dict(run["task"])
    policy_sha256 = _json_sha256(policy, "acceptance policy")
    record = _read_saved_record(store, run_id, state, policy_sha256, task_sha256(task))
    output = state.get("output")
    if not isinstance(output, str):
        raise WorkflowStateError("Completed workflow output is not text")
    problems = _decision_problems(store, run_id, graph_from_dict(state["graph"]), output, record,
                                  task=task, policy_sha256=policy_sha256)
    return _held_record(record, output, problems, record.task_sha256)


@dataclass(frozen=True, slots=True)
class CurrentArtifact:
    """A run's current publication, the delivery decision that covers it, and its eligibility.

    ``decision`` is None for a run without a strict policy or not yet
    completed. ``integrity`` is :func:`artifact_integrity` of the
    publication. ``reasons`` names every condition delivery is missing:
    ``no_decision``; ``not_accepted``; ``record_mismatch`` when the decision
    binds other bytes; and ``artifact_missing``, ``artifact_changed``,
    ``artifact_unsafe`` or ``artifact_unreadable`` from the integrity check.
    It is derived on every read and never stored.
    """

    publication: PublishedArtifact
    decision: DeliveryAcceptance | None
    integrity: dict[str, str]
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        """True only with an accepted decision on bytes the published file still holds."""
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {"publication": self.publication.to_dict(), "path": str(self.publication.path),
                "decision": None if self.decision is None else self.decision.decision.value,
                "integrity": dict(self.integrity),
                "delivery_eligibility": {"eligible": self.eligible, "reasons": list(self.reasons)}}


def current_artifact(store: Any, run_id: str) -> CurrentArtifact | None:
    """A durable run's current publication and the delivery decision that covers it.

    The current publication is the one with the highest sequence that a
    checkpoint applied: the first publication, or the latest
    republication. The decision is the one that covers the run's completed
    output (see :class:`CurrentArtifact`); it must bind the published bytes,
    which the file must still hold, for the artifact to be eligible for
    delivery. Returns None when the run has published nothing. Only reads are
    made, so a store opened read-only will do. Resuming a run, republishing
    it and ``smythe runs inspect`` all read a run's artifact through this
    accessor.
    """
    publication = latest_publication(store, run_id)
    if publication is None:
        return None
    decision = _current_decision(store, run_id)
    integrity = artifact_integrity(publication)
    reasons = []
    if decision is None:
        reasons.append("no_decision")
    else:
        if not decision.accepted:
            reasons.append("not_accepted")
        if decision.output_sha256 != publication.sha256:
            reasons.append("record_mismatch")
    if integrity["status"] != "verified":
        reasons.append("artifact_" + integrity["status"])
    return CurrentArtifact(publication, decision, integrity, tuple(reasons))


def artifact_problems(current: CurrentArtifact | None, output: Any) -> list[str]:
    """Why a completed run's published document no longer delivers *output*, one reason each.

    The current publication must hold exactly *output*'s UTF-8 bytes, and its
    file must still hold them. An empty list means it does.
    """
    if current is None:
        return ["the run has no published artifact"]
    publication, problems = current.publication, []
    if not isinstance(output, str) or text_sha256(output) != publication.sha256:
        problems.append("the published artifact does not hold the saved output")
    status = current.integrity["status"]
    if status != "verified":
        problems.append(f"the published artifact {publication.name!r} {_PROBLEMS[status]}")
    return problems
