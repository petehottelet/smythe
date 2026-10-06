"""Preview external actions that write a durable run's accepted content to a service.

An :class:`ActionIntent` says what one logical action writes and where, and
:func:`action_key` names the key every attempt sends. A
:class:`ConnectorProfile` declares what a connector's service guarantees.
Connectors and approval verifiers are trusted host code with synchronous
protocols, checked only structurally: a caller contract, not a sandbox.

:func:`prepare_action` records a logical action in the workflow journal, bound
to a completed run's accepted content; :func:`approve`, :func:`reject`,
:func:`revoke` and :func:`stop_actions` record verified approvals and
controls; :func:`check_authority` checks them again; and
``smythe.workflow_actions`` journals each attempt's
dispatch and outcome. :class:`ActionRunner` submits each attempt once through
a connector, reads the result back and reconciles an unknown outcome as the
connector's declarations allow, and :func:`record_notice` keeps a
service's callbacks as untrusted records. Only trusted host code calls these
functions, and ``import smythe`` does not load this module.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields, replace
from typing import Any, Protocol, Self
from uuid import uuid4

from smythe._canonical import canonical_bytes, sha256_hex
from smythe._redact import redact_account_identifiers
from smythe.acceptance import _calls_async, _is_async
from smythe.provider import Provider
from smythe.validation import (
    ValidationOutcome, ValidationReport, ValidatorIdentity, _require_keys, _require_sha256,
    advisory_finding, hard_finding,
)
from smythe.workflow_binding import WorkflowBindingError, json_snapshot
from smythe.workflow_store import (
    WorkflowBudgetError, WorkflowConflictError, WorkflowLeaseError, WorkflowStateError,
    WorkflowValidationError, _canonical, _text,
)

ACTION_CONTRACT_VERSION = "smythe.action.v1"
_CONTRACT_VERSIONS = (ACTION_CONTRACT_VERSION,)

# ``canceled`` closes an action that was rejected, expired or revoked before
# any dispatch.
ACTION_STATES = (
    "not_dispatched", "dispatched", "outcome_unknown", "observed", "reconciled",
    "compensation_required", "canceled",
)
# A finished action makes no further effect, so its slot can take a new action.
_FINISHED = ("observed", "reconciled", "canceled")
# No approval authorizes an action in these states.
_CLOSED = (*_FINISHED, "compensation_required")
# A run blocked for this reason authorizes no action.
_SECURITY_STOP = "security_stop"
_COMPENSATION = ("new_action", "unsupported")
# The validator whose report records the verdict of each readback read.
READBACK_VALIDATOR = ValidatorIdentity.for_config("smythe.action-readback", "1")
# The largest notice, as canonical JSON, that record_notice keeps.
_NOTICE_BYTES = 65_536


class ConnectorRefused(Exception):
    """The service definitely did not commit the write."""


class ConnectorAmbiguous(Exception):
    """The service may have committed the write: its answer was lost or unreadable."""


class ApprovalRejected(Exception):
    """An approval verifier refused an assertion."""


class ActionStateError(WorkflowStateError):
    """The requested transition is not allowed from the action's recorded state."""


class ActionNotAuthorized(ActionStateError):
    """Nothing authorizes the action now: its approval or its content no longer holds."""


def _bool(value: Any, what: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{what} must be True or False")


def _count(value: Any, what: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{what} must be an integer of at least {minimum}")


def _seconds(value: Any, what: str, *, positive: bool = False) -> float:
    try:
        seconds = float(value) if type(value) in (int, float) else math.nan
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or positive and seconds == 0:
        raise ValueError(f"{what} must be a finite, {'positive' if positive else 'non-negative'} "
                         "number of seconds")
    return seconds


def _media_type(value: Any) -> None:
    if "/" not in _text(value, "media_type"):
        raise ValueError("media_type must be a type/subtype such as text/plain")


def _object_target(target: Any, what: str) -> None:
    if not isinstance(target, ActionTarget):
        raise TypeError(f"{what} target must be an ActionTarget")
    if target.object_id is None:
        raise ValueError(f"{what} target must name an object_id")


class _Flat:
    """``to_dict``/``from_dict`` for a dataclass of JSON scalars; every key is required."""

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: Any) -> Self:
        _require_keys(data, cls.__name__, frozenset(item.name for item in fields(cls)))
        return cls(**data)


@dataclass(frozen=True)
class ActionTarget(_Flat):
    """Where an action writes: a tenant's collection and, once known, an object in it."""

    tenant: str
    collection: str
    object_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.tenant, "target tenant")
        _text(self.collection, "target collection")
        if self.object_id is not None:
            _text(self.object_id, "target object_id")


@dataclass(frozen=True)
class ContentRef(_Flat):
    """Content an action delivers: its SHA-256 as plain hex, size and media type."""

    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, "content sha256")
        _count(self.size_bytes, "content size_bytes")
        _media_type(self.media_type)

    @classmethod
    def for_bytes(cls, data: bytes, media_type: str) -> ContentRef:
        if not isinstance(data, bytes):
            raise TypeError("content must be bytes")
        return cls(hashlib.sha256(data).hexdigest(), len(data), media_type)


@dataclass(frozen=True, init=False)
class ActionIntent:
    """What one logical action does, independent of any attempt to do it.

    ``parameters`` is a JSON object, copied when the intent is made.
    :meth:`digest` hashes the canonical :meth:`to_dict` form, so reordering
    keys does not change it and changing any field does.
    """

    operation: str
    target: ActionTarget
    _parameters_json: str = field(repr=False)
    content: ContentRef | None
    connector: str
    connector_version: str
    contract_version: str

    def __init__(
        self, operation: str, target: ActionTarget, *, parameters: dict[str, Any] | None = None,
        content: ContentRef | None = None, connector: str, connector_version: str,
        contract_version: str = ACTION_CONTRACT_VERSION,
    ) -> None:
        if type(contract_version) is not str or contract_version not in _CONTRACT_VERSIONS:
            raise ValueError(f"Unknown action contract version {contract_version!r}")
        _text(operation, "action operation")
        if not isinstance(target, ActionTarget):
            raise TypeError("action target must be an ActionTarget")
        if content is not None and not isinstance(content, ContentRef):
            raise TypeError("action content must be a ContentRef or None")
        _text(connector, "connector")
        _text(connector_version, "connector_version")
        parameters = {} if parameters is None else parameters
        if type(parameters) is not dict:
            raise ValueError("action parameters must be a JSON object")
        try:
            snapshot = canonical_bytes(json_snapshot(parameters)).decode("utf-8")
        except (WorkflowBindingError, ValueError, RecursionError) as exc:
            raise ValueError(
                "action parameters must be finite, acyclic JSON data with string keys"
            ) from exc
        for name, value in (("operation", operation), ("target", target),
                            ("_parameters_json", snapshot), ("content", content),
                            ("connector", connector), ("connector_version", connector_version),
                            ("contract_version", contract_version)):
            object.__setattr__(self, name, value)

    @property
    def parameters(self) -> dict[str, Any]:
        """A fresh copy of the parameters."""
        return json.loads(self._parameters_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version, "operation": self.operation,
            "target": self.target.to_dict(), "parameters": self.parameters,
            "content": None if self.content is None else self.content.to_dict(),
            "connector": self.connector, "connector_version": self.connector_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ActionIntent:
        """Read :meth:`to_dict` data; unknown versions and keys raise ValueError."""
        _require_keys(data, "action intent", frozenset({
            "contract_version", "operation", "target", "parameters", "content", "connector",
            "connector_version"}))
        if data["contract_version"] not in _CONTRACT_VERSIONS:
            raise ValueError(f"Unknown action contract version {data['contract_version']!r}")
        if type(data["parameters"]) is not dict:
            # to_dict() always writes an object; only the constructor reads None as {}.
            raise ValueError("action parameters must be a JSON object")
        content = data["content"]
        return cls(
            data["operation"], ActionTarget.from_dict(data["target"]),
            parameters=data["parameters"],
            content=None if content is None else ContentRef.from_dict(content),
            connector=data["connector"], connector_version=data["connector_version"],
            contract_version=data["contract_version"],
        )

    def digest(self) -> str:
        """The SHA-256 of the canonical intent, as plain hex."""
        return sha256_hex(self.to_dict())


def action_key(store_id: str, run_id: str, slot: str, sequence: int) -> str:
    """The key a connector receives for every attempt of one logical action.

    It names the journal, the run, a host-chosen slot and the action's
    sequence number in that slot, so it never depends on the intent or an
    attempt and stays the same across retries and restarts.
    """
    for value, name in ((store_id, "store_id"), (run_id, "run_id"), (slot, "slot")):
        _text(value, name)
    _count(sequence, "sequence")
    return sha256_hex(["smythe.action-key", store_id, run_id, slot, sequence])


@dataclass(frozen=True)
class ConnectorProfile:
    """What a connector declares about the service it writes to.

    ``operations`` is stored sorted. Settings after it are keyword-only and
    have exact types, so a bool is not a number; seconds are finite and
    ceilings are integer nanoUSD. ``dedupe_window_s`` needs ``deduplicates``
    and ``None`` declares no expiry; ``visibility_bound_s=None`` is
    unbounded. :meth:`to_dict` omits defaults.
    """

    name: str
    version: str
    operations: tuple[str, ...]
    _: KW_ONLY
    deduplicates: bool = False
    dedupe_window_s: float | None = None
    lookup_by_key: bool = False
    lookup_by_object: bool = False
    visibility_bound_s: float | None = None
    cancellation: str = "unsupported"
    compensation: str = "unsupported"
    write_ceiling_nanousd: int = 0
    read_ceiling_nanousd: int = 0
    max_attempts: int = 1
    readback_reads: int = 1
    readback_interval_s: float = 0.0
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        def store(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        _text(self.name, "connector name")
        _text(self.version, "connector version")
        if type(self.operations) not in (list, tuple):
            raise ValueError("connector operations must be a list of names")
        operations = tuple(sorted(_text(item, "operation") for item in self.operations))
        if not operations or len(set(operations)) != len(operations):
            raise ValueError("connector operations must be distinct names, at least one")
        store("operations", operations)
        for name in ("deduplicates", "lookup_by_key", "lookup_by_object"):
            _bool(getattr(self, name), name)
        if self.dedupe_window_s is not None:
            if not self.deduplicates:
                raise ValueError("dedupe_window_s needs deduplicates=True")
            store("dedupe_window_s", _seconds(self.dedupe_window_s, "dedupe_window_s",
                                              positive=True))
        if self.visibility_bound_s is not None:
            store("visibility_bound_s", _seconds(self.visibility_bound_s, "visibility_bound_s"))
        if type(self.cancellation) is not str or self.cancellation != "unsupported":
            raise ValueError("cancellation must be 'unsupported'")
        if type(self.compensation) is not str or self.compensation not in _COMPENSATION:
            raise ValueError(f"compensation must be one of {_COMPENSATION}")
        _count(self.write_ceiling_nanousd, "write_ceiling_nanousd")
        _count(self.read_ceiling_nanousd, "read_ceiling_nanousd")
        _count(self.max_attempts, "max_attempts", 1)
        _count(self.readback_reads, "readback_reads")
        store("readback_interval_s", _seconds(self.readback_interval_s, "readback_interval_s"))
        store("timeout_s", _seconds(self.timeout_s, "timeout_s", positive=True))

    def to_dict(self) -> dict[str, Any]:
        data = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.default is MISSING or value != item.default:
                data[item.name] = list(value) if isinstance(value, tuple) else value
        return data

    @classmethod
    def from_dict(cls, data: Any) -> ConnectorProfile:
        required = frozenset({"name", "version", "operations"})
        _require_keys(data, "connector profile", required,
                      frozenset(item.name for item in fields(cls)) - required)
        return cls(**data)


@dataclass(frozen=True)
class ConnectorReceipt:
    """A service's answer to a write: the key, the committed object and its version.

    ``cost_nanousd`` is ``None`` when the service reports no cost;
    ``deduplicated`` means it answered from an earlier commit.
    """

    key: str
    target: ActionTarget
    object_version: int | None = None
    cost_nanousd: int | None = None
    deduplicated: bool = False

    def __post_init__(self) -> None:
        _text(self.key, "receipt key")
        _object_target(self.target, "receipt")
        for name in ("object_version", "cost_nanousd"):
            if getattr(self, name) is not None:
                _count(getattr(self, name), name)
        _bool(self.deduplicated, "deduplicated")

    def to_dict(self) -> dict[str, Any]:
        return {**_Flat.to_dict(self), "target": self.target.to_dict()}

    @classmethod
    def from_dict(cls, data: Any) -> ConnectorReceipt:
        _require_keys(data, "connector receipt", frozenset(item.name for item in fields(cls)))
        return cls(**{**data, "target": ActionTarget.from_dict(data["target"])})


def _receipt_of(key: str, intent: ActionIntent, receipt: Any) -> bool:
    """Whether *receipt* names the action's *key*, its target's tenant and collection and its object."""
    target = intent.target
    return (isinstance(receipt, ConnectorReceipt) and receipt.key == key
            and (receipt.target.tenant, receipt.target.collection) == (target.tenant, target.collection)
            and target.object_id in (None, receipt.target.object_id))


def _ns(seconds: float, *, up: bool = False) -> int:
    nanoseconds = seconds * 1_000_000_000
    return math.ceil(nanoseconds) if up else math.floor(nanoseconds)


def _window_open(profile: ConnectorProfile, first_claim_ns: int, now_ns: int) -> bool:
    """Whether a call sent at *now_ns* surely reaches the service inside the key's window.

    The window is ``dedupe_window_s`` (``None`` has no expiry), counted from
    the action's first dispatch claim, the earliest its key can have been
    committed. A call may take ``timeout_s`` to arrive: that is the margin.
    """
    window = profile.dedupe_window_s
    return window is None or now_ns + _ns(profile.timeout_s, up=True) < first_claim_ns + _ns(window)


def _absence_unproven(profile: ConnectorProfile, first_claim_ns: int, last_claim_ns: int,
                      now_ns: int) -> str | None:
    """Why an empty lookup sent at *now_ns* does not show that the key was never committed, or None.

    Only a declared ``visibility_bound_s`` that has passed since the latest
    dispatch claim plus the connector's timeout, while the key's window is
    still open, makes it proof.
    """
    bound = profile.visibility_bound_s
    if bound is None:
        return "the connector declares no visibility bound, so an empty lookup proves nothing"
    if now_ns < last_claim_ns + _ns(bound + profile.timeout_s, up=True):
        return "the visibility bound had not passed since the dispatch, so an empty lookup proves nothing"
    if not _window_open(profile, first_claim_ns, now_ns):
        return "the key's window may have closed, so an empty lookup proves nothing"
    return None


@dataclass(frozen=True)
class ObservedObject:
    """An object as a connector read it: its exact bytes and, if exposed, version and key.

    ``content_ref`` hashes the bytes; it never comes from the service.
    """

    target: ActionTarget
    content: bytes = field(repr=False)
    media_type: str
    version: int | None = None
    key: str | None = None

    def __post_init__(self) -> None:
        _object_target(self.target, "observed")
        if not isinstance(self.content, bytes):
            raise TypeError("observed content must be bytes")
        object.__setattr__(self, "content", bytes(self.content))
        _media_type(self.media_type)
        if self.version is not None:
            _count(self.version, "observed version")
        if self.key is not None:
            _text(self.key, "observed key")

    @property
    def content_ref(self) -> ContentRef:
        return ContentRef.for_bytes(self.content, self.media_type)


@dataclass(frozen=True)
class Principal(_Flat):
    """Who approved an action, as the host's verifier established it."""

    issuer: str
    subject: str
    tenant: str

    def __post_init__(self) -> None:
        for name in ("issuer", "subject", "tenant"):
            _text(getattr(self, name), f"principal {name}")


class WriteConnector(Protocol):
    """Trusted host code that writes to one service; each call makes one request, no retries.

    ``submit`` returns a receipt or raises :class:`ConnectorRefused` (nothing
    committed) or :class:`ConnectorAmbiguous` (maybe committed). ``lookup``
    finds an earlier submission by key in the target's tenant and ``read``
    returns an object's bytes; ``None`` alone does not prove nothing was written.
    """

    @property
    def profile(self) -> ConnectorProfile: ...

    def submit(self, key: str, intent: ActionIntent, payload: bytes | None,
               timeout_s: float) -> ConnectorReceipt: ...

    def lookup(self, key: str, target: ActionTarget,
               timeout_s: float) -> ConnectorReceipt | None: ...

    def read(self, target: ActionTarget, timeout_s: float) -> ObservedObject | None: ...


class ApprovalVerifier(Protocol):
    """Trusted host code that authenticates approvals; Smythe ships none.

    ``verify`` returns the principal that issued ``assertion`` for exactly
    ``intent_sha256`` or raises :class:`ApprovalRejected`;
    ``still_authorized`` says whether it may still approve ``intent``.
    ``identity`` names the verifier and its version.
    """

    @property
    def identity(self) -> str: ...

    def verify(self, assertion: str, intent_sha256: str, now_ns: int) -> Principal: ...

    def still_authorized(self, principal: Principal, intent: ActionIntent, now_ns: int) -> bool: ...


def _close(value: object) -> None:
    """Close a refused coroutine, so it is never reported as unawaited."""
    if inspect.iscoroutine(value):
        value.close()


def _check_host_code(value: object, role: str, methods: dict[str, bool]) -> None:
    """Refuse what cannot be synchronous host code; ``methods`` maps names to whether required."""
    if isinstance(value, Provider):
        raise TypeError(f"a Provider cannot be {role}")
    if inspect.isawaitable(value):
        _close(value)
        raise TypeError(f"an awaitable ({type(value).__name__}) cannot be {role}")
    if isinstance(value, type):
        raise TypeError(f"{role} must be an instance, not the class {value.__name__}")
    if _is_async(value):
        raise TypeError(f"a coroutine function cannot be {role}")
    for name, required in methods.items():
        method = getattr(value, name, None)
        if method is None and not required:
            continue
        if not callable(method):
            # An async property returned a coroutine in place of the method.
            _close(method)
            raise TypeError(f"{type(value).__name__} has no {name}() method")
        if _calls_async(method):
            raise TypeError(f"{type(value).__name__}.{name}() must be synchronous")


def check_connector(connector: object) -> ConnectorProfile:
    """Check that *connector* is synchronous host code, and return its profile.

    ``profile`` must be a :class:`ConnectorProfile`, so an async property is
    refused (its coroutine is closed). ``submit()`` must be synchronous, and
    so must ``lookup()`` and ``read()`` where present; ``lookup()`` is
    required when the profile declares ``lookup_by_key`` and ``read()`` when
    it declares ``lookup_by_object``. Other methods are not checked. Raises
    TypeError.
    """
    _check_host_code(connector, "a connector", {})
    profile = getattr(connector, "profile", None)
    if not isinstance(profile, ConnectorProfile):
        _close(profile)
        raise TypeError(
            f"connector profile must be a ConnectorProfile, not {type(profile).__name__}"
        )
    _check_host_code(connector, "a connector", {
        "submit": True, "lookup": profile.lookup_by_key, "read": profile.lookup_by_object,
    })
    return profile


def check_verifier(verifier: object) -> str:
    """Check that *verifier* is synchronous host code, and return its identity.

    ``verify()`` and ``still_authorized()`` must be synchronous; other methods
    are not checked. ``identity`` must be a string naming the verifier like
    any other name: nonempty, without control characters and at most 512
    characters. Raises TypeError.
    """
    _check_host_code(verifier, "an approval verifier", {"verify": True, "still_authorized": True})
    identity = getattr(verifier, "identity", None)
    if type(identity) is not str or not identity.strip():
        _close(identity)
        raise TypeError("approval verifier identity must be a non-empty string")
    try:
        return _text(identity, "approval verifier identity")
    except ValueError as exc:
        raise TypeError(str(exc)) from exc


@dataclass(frozen=True)
class PreparedAction:
    """A logical action a run's journal records, with the intent and profile it is bound to.

    ``sequence`` counts the actions of ``slot`` in the run from 1.
    ``ceiling_nanousd`` is the cost of ``max_attempts`` writes and
    ``readback_reads`` reads at the profile's ceilings, and an approval must
    cover it. No attempt, read or reconciliation lookup is made that would
    take the action's confirmed, reserved and unknown nanoUSD past the
    approval's ceiling, so lookups share the budget of the reads.
    """

    action_id: str
    run_id: str
    slot: str
    sequence: int
    action_key: str
    intent: ActionIntent
    profile: ConnectorProfile
    state: str
    ceiling_nanousd: int


@dataclass(frozen=True)
class ActionApproval:
    """A recorded approval: the principal who approved an action, its ceiling and expiry."""

    approval_id: str
    action_id: str
    principal: Principal
    ceiling_nanousd: int
    expires_at_ns: int


def _undeliverable(store: Any, run_id: str, content: ContentRef) -> str | None:
    """Why *content* is not a completed run's accepted, deliverable output now, or None.

    The run's current delivery decision must be ``accepted``; a run without
    one, such as an advisory or earlier run, never is, and none is assumed. A
    source the run delivered from that was removed after the decision makes
    the content unavailable. The decision must still hold for *content*'s
    exact SHA-256 and size, and a published document's file must still hold
    those bytes. Only reads are made.
    """
    from smythe.workflow import _removal_problems, current_decision
    from smythe.workflow_artifact import _current_decision, current_artifact

    run = store.load_run(run_id)
    if run["blocked_reason"] == _SECURITY_STOP:
        return "a security stop holds the run"
    if run["status"] != "completed":
        return "the run has not completed"
    latest = current_decision(store, run_id)
    if latest is None:
        return "the run has no recorded delivery acceptance"
    if not latest["record"].accepted:
        return f"the run's delivery acceptance is {latest['record'].decision.value}"
    considered = latest["inputs"].get("tombstones", [])
    if _removal_problems(store, run_id, considered if type(considered) is list else []):
        return "content unavailable: a source of the accepted output was removed from the run"
    current = current_artifact(store, run_id)
    held = _current_decision(store, run_id) if current is None else current.decision
    if held is None or not held.accepted:
        return "the run's accepted decision no longer holds"
    size = (len(store.get_checkpoint(run_id)["checkpoint"]["output"].encode("utf-8"))
            if current is None else current.publication.size_bytes)
    if (content.sha256, content.size_bytes) != (held.output_sha256, size):
        return "the content is not the run's accepted output"
    if current is not None and not current.eligible:
        return f"the published document is not eligible for delivery ({', '.join(current.reasons)})"
    return None


def _assertion(assertion: Any) -> str:
    if type(assertion) is not str or not assertion:
        raise WorkflowValidationError("assertion must be a non-empty string")
    return assertion


def _verified(verifier: Any, assertion: str, intent: ActionIntent, now_ns: int) -> Principal:
    """The principal *verifier* says issued *assertion* for *intent*, who may still approve it."""
    principal = verifier.verify(assertion, intent.digest(), now_ns)
    if not isinstance(principal, Principal):
        raise ApprovalRejected("The approval verifier did not return a Principal")
    if principal.tenant != intent.target.tenant:
        raise ApprovalRejected(f"A principal of tenant {principal.tenant!r} cannot approve an "
                               f"action in tenant {intent.target.tenant!r}")
    if verifier.still_authorized(principal, intent, now_ns) is not True:
        raise ApprovalRejected("The approval verifier no longer authorizes this principal")
    return principal


def prepare_action(store: Any, run_id: str, *, slot: str, intent: ActionIntent,
                   profile: ConnectorProfile) -> PreparedAction:
    """Record the logical action that delivers a completed run's accepted content.

    The intent must name *profile*'s connector and one of its operations, and
    its content must be the run's accepted output: the published document
    for a run with a renderer, whose file must still hold it. The run's
    current delivery decision must be ``accepted``. No lease is needed.

    The slot's first action has sequence 1. The same intent and profile again
    return the slot's latest action, whatever its state. Another intent or
    profile raises ``WorkflowConflictError`` while that action is unfinished,
    and starts the next sequence, with its own key, once it is observed,
    reconciled or canceled. Raises ``ActionNotAuthorized`` when the content
    is not deliverable.
    """
    from smythe.workflow_actions import prepare

    if not isinstance(intent, ActionIntent) or not isinstance(profile, ConnectorProfile):
        raise TypeError("prepare_action takes an ActionIntent and a ConnectorProfile")
    if ((intent.connector, intent.connector_version) != (profile.name, profile.version)
            or intent.operation not in profile.operations):
        raise WorkflowValidationError("The intent must name the profile's connector and one of "
                                      "its operations")
    if intent.content is None:
        raise WorkflowValidationError("An action delivers a run's accepted content; the intent "
                                      "names no content")
    _text(slot, "slot")
    problem = _undeliverable(store, run_id, intent.content)
    if problem:
        raise ActionNotAuthorized(f"Run {run_id!r} cannot prepare this action: {problem}")
    return prepare(store, run_id, slot, intent, profile)


def load_action(store: Any, action_id: str) -> PreparedAction:
    """The action *action_id* as the journal records it; only reads are made."""
    from smythe.workflow_actions import load

    return load(store, action_id)[0]


def approve(store: Any, action_id: str, assertion: str, verifier: Any, *, ceiling_nanousd: int,
            expires_at_ns: int) -> ActionApproval:
    """Record the verified approval of one action, up to a ceiling and until an expiry.

    *verifier* must verify *assertion* for exactly the action's intent
    digest, return a principal of the intent's tenant, and still authorize
    that principal. *ceiling_nanousd* must cover the action's
    ``ceiling_nanousd`` and *expires_at_ns* must be later than the journal's
    clock. The run's content must still be deliverable. An assertion is
    recorded once in a journal: the same approval again returns it, and an
    assertion recorded for anything else raises ``ApprovalRejected``. The
    newest approval supersedes earlier ones. The verifier is called outside
    any journal transaction, and no lease is needed.
    """
    from smythe.workflow_actions import load, record_verdict

    identity = check_verifier(verifier)
    _assertion(assertion)
    for value, name in ((ceiling_nanousd, "ceiling_nanousd"), (expires_at_ns, "expires_at_ns")):
        if type(value) is not int or value < 0:
            raise WorkflowValidationError(f"{name} must be a non-negative integer")
    action = load(store, action_id)[0]
    if action.state in _CLOSED:
        raise ActionStateError(f"Action {action_id!r} is {action.state} and cannot be approved")
    if ceiling_nanousd < action.ceiling_nanousd:
        raise WorkflowValidationError(
            f"ceiling_nanousd must cover the action's ceiling of {action.ceiling_nanousd} nanoUSD")
    now = store._clock_ns()
    if expires_at_ns <= now:
        raise WorkflowValidationError("expires_at_ns must be later than the journal's clock")
    principal = _verified(verifier, assertion, action.intent, now)
    problem = _undeliverable(store, action.run_id, action.intent.content)
    if problem:
        raise ActionNotAuthorized(f"Action {action_id!r} cannot be approved: {problem}")
    row = record_verdict(store, action, "approved", principal, identity, assertion,
                         ceiling_nanousd, expires_at_ns)
    return ActionApproval(row["approval_id"], action_id, principal, ceiling_nanousd, expires_at_ns)


def reject(store: Any, action_id: str, assertion: str, verifier: Any) -> PreparedAction:
    """Cancel an action that was never dispatched, by a verified principal of its tenant.

    The assertion is verified as for :func:`approve`. The rejection is
    recorded with the principal and the action is ``canceled`` for good:
    preparing the same intent in its slot returns it. An attempt begun but
    not yet claimed is released with its reservation, since nothing was sent.
    Rejecting a canceled action again changes nothing. Raises
    ``ActionStateError`` once the action has been dispatched.
    """
    from smythe.workflow_actions import load, record_verdict

    identity = check_verifier(verifier)
    _assertion(assertion)
    action = load(store, action_id)[0]
    if action.state == "canceled":
        return action
    if action.state != "not_dispatched":
        raise ActionStateError(f"Action {action_id!r} is {action.state} and cannot be rejected")
    principal = _verified(verifier, assertion, action.intent, store._clock_ns())
    record_verdict(store, action, "rejected", principal, identity, assertion, None, None)
    return load(store, action_id)[0]


def revoke(store: Any, action_or_approval_id: str, reason: str) -> tuple[str, ...]:
    """Revoke one approval, or every approval of one action; return the newly revoked IDs.

    A fail-safe control: it needs no principal and no lease. *reason* is
    stored with account identifiers redacted. A revoked approval never
    authorizes the action again; only a new approval can. Revoking twice
    keeps the first revocation.
    """
    from smythe.workflow_actions import revoke_approvals

    reason = _text(redact_account_identifiers(reason) if isinstance(reason, str) else reason,
                   "reason")
    return revoke_approvals(store, _text(action_or_approval_id, "action or approval id"), reason)


def stop_actions(store: Any, run_id: str, reason: str) -> tuple[str, ...]:
    """Stop every action of a run from being dispatched; return the approval IDs it revoked.

    A fail-safe control, like :func:`revoke`: it needs no principal and no
    lease. Every approval of the run's actions is revoked with *reason*,
    stored redacted, and an ``actions_stopped`` event lists them. Nothing
    clears it: no resume, recovery, lease or reopened journal makes a revoked
    approval authorize again, so each action is refused until it gets a new
    approval. A submission already claimed is not recalled; its outcome is
    recorded when it arrives, and reads and lookups that settle an outcome go
    on. Stopping again revokes the approvals recorded since.
    """
    from smythe.workflow_actions import stop_run

    reason = _text(redact_account_identifiers(reason) if isinstance(reason, str) else reason,
                   "reason")
    return stop_run(store, _text(run_id, "run_id"), reason)


def check_authority(store: Any, action_id: str, verifier: Any) -> ActionApproval:
    """The approval that authorizes an action now; raises ``ActionNotAuthorized`` otherwise.

    The action must not be finished or need compensation. Its newest
    approval must be unrevoked, unconsumed and unexpired on the journal's
    clock, bind the action's intent digest, cover its ceiling and have been
    recorded by this verifier. *verifier* must verify the stored assertion
    again for the same principal and still authorize it, and the run's
    content must still be deliverable (see :func:`prepare_action`). Only
    reads are made; the verifier is called outside any journal transaction.
    """
    from smythe.workflow_actions import _authorizing, load

    identity = check_verifier(verifier)
    action, approvals = load(store, action_id)

    def refused(reason: str) -> ActionNotAuthorized:
        return ActionNotAuthorized(f"Action {action_id!r} is not authorized: {reason}")

    now = store._clock_ns()
    approval = _authorizing(action_id, action.state, action.intent.digest(), action.ceiling_nanousd,
                            approvals, now)
    if json.loads(approval["verifier_json"]) != {"identity": identity}:
        raise refused("another approval verifier recorded its approval")
    try:
        principal = _verified(verifier, approval["assertion"], action.intent, now)
    except ApprovalRejected as error:
        raise refused(f"the approval verifier refuses it: {error}") from error
    if principal.to_dict() != json.loads(approval["principal_json"]):
        raise refused("the approval verifier names another principal")
    problem = _undeliverable(store, action.run_id, action.intent.content)
    if problem:
        raise refused(problem)
    return ActionApproval(approval["approval_id"], action_id, principal, int(approval["ceiling"]),
                          approval["expires_at_ns"])


def record_notice(store: Any, payload: Any) -> PreparedAction | None:
    """Keep a callback the host relays as an untrusted record; return the action to read back.

    *payload* is a JSON object of at most 64 KiB as canonical JSON. It names
    the action key it reports as ``key`` and the action's target as
    ``target``, an object with ``tenant`` and ``collection``; other fields are
    kept as they are. A notice whose key names an action of the journal is
    kept with that action, marked untrusted; any other is not kept. The action
    is returned, so that the host can call :meth:`ActionRunner.readback`, only
    when it is ``dispatched`` and the notice names its tenant and collection;
    otherwise None is. A notice never changes an action: only a read the
    runner makes can show its result. No lease is needed.
    """
    from smythe.workflow_actions import keep_notice

    if type(payload) is not dict:
        raise WorkflowValidationError("A notice is a JSON object")
    if len(_canonical(payload).encode("utf-8")) > _NOTICE_BYTES:
        raise WorkflowValidationError(f"A notice is at most {_NOTICE_BYTES} bytes of JSON")
    key, target = payload.get("key"), payload.get("target")
    action = keep_notice(store, key, payload) if type(key) is str else None
    if (action is None or action.state != "dispatched" or type(target) is not dict
            or (target.get("tenant"), target.get("collection"))
            != (action.intent.target.tenant, action.intent.target.collection)):
        return None
    return action


def _payload(store: Any, action: PreparedAction) -> bytes | None:
    """The bytes *action* delivers, read once; None unless they are exactly its bound content.

    They are the run's current published document, found through
    ``current_artifact``, or else its output text as UTF-8.
    """
    from smythe.workflow_artifact import current_artifact, read_published

    content = action.intent.content
    if content is None:
        return None
    current = current_artifact(store, action.run_id)
    data = (store.get_checkpoint(action.run_id)["checkpoint"]["output"].encode("utf-8")
            if current is None else read_published(current.publication))
    if data is None or (hashlib.sha256(data).hexdigest(), len(data)) != (content.sha256,
                                                                          content.size_bytes):
        return None
    return data


def _verdict(action: PreparedAction, receipt: ConnectorReceipt, observed: Any,
             error: str | None) -> ValidationReport:
    """The readback report on one read of the object *receipt* names.

    A read that fails, finds no object or finds a version older than the
    receipt's decides nothing: its outcome is ``error``. Otherwise the object
    must be the receipt's target, hold the bound content (the canonical JSON
    of the parameters for an intent without content) and, when the connector
    exposes it, carry the action's key.
    """
    target, content = receipt.target, action.intent.content
    path = f"{target.tenant}/{target.collection}/{target.object_id}"

    def undecided(code: str, message: str, **values: Any) -> ValidationReport:
        return ValidationReport(path, (advisory_finding(code, message, **values),),
                                ValidationOutcome.ERROR, READBACK_VALIDATOR)

    if not isinstance(observed, ObservedObject):
        return undecided("read_failed" if error else "not_visible", error or "The read found no object")
    findings = [] if observed.target == target else [hard_finding(
        "target_mismatch", "The object read is not the receipt's target",
        expected=target.to_dict(), observed=observed.target.to_dict())]
    if (not findings and None not in (observed.version, receipt.object_version)
            and observed.version < receipt.object_version):
        return undecided("stale_version", "The object read is older than the receipt's version",
                         expected=receipt.object_version, observed=observed.version)
    digest = (content.sha256 if content is not None
              else hashlib.sha256(canonical_bytes(action.intent.parameters)).hexdigest())
    for code, message, expected, seen in (
            ("content_mismatch", "The object does not hold the bound content", digest,
             observed.content_ref.sha256),
            ("key_mismatch", "The object was written under another key", action.action_key,
             action.action_key if observed.key is None else observed.key)):
        if seen != expected:
            findings.append(hard_finding(code, message, expected=expected, observed=seen))
    return ValidationReport(path, tuple(findings), validator=READBACK_VALIDATOR)


@dataclass(frozen=True)
class ActionProgress:
    """Where an action stands after an :class:`ActionRunner` step, and what it needs next.

    ``attempt`` is its latest attempt as ``inspect_run()`` shows it, or None.
    ``report`` is the verdict of the step's last read, or None when it made
    none. ``next_action`` is ``dispatch`` for an action not dispatched with
    attempts left, ``readback`` while an acknowledged attempt has reads left
    that fit its approval's ceiling, ``reconcile`` for an outcome the reads
    have not settled,
    ``operator_review`` once reconciliation has found it can do no more,
    ``compensate`` for ``compensation_required``, and None when the runner
    can do nothing more. ``refusal`` is None, or says why
    :meth:`ActionRunner.dispatch_pending` sent nothing for the action; its
    ``next_action`` is then None.
    """

    action_id: str
    state: str
    attempt: dict[str, Any] | None
    report: ValidationReport | None
    next_action: str | None
    refusal: str | None = None


class ActionRunner:
    """Dispatch a durable run's prepared actions, read back and reconcile their results, one at a time.

    The runner is the only retry owner: each attempt submits once, and only
    :meth:`reconcile` submits again, with the same key, where the service
    deduplicates it, within ``max_attempts``. *connector* must make one
    request per call; one exposing ``max_retries`` must set it to 0. Each
    step holds the run's lease as *owner_id* (a new name by default) for
    *lease_ttl_s*, renewed beyond the submission's timeout or the wait before
    the next read, and settles first what a stopped worker left in flight.
    *sleep* and *clock*, in seconds, space the reads and lookups. Dispatch is
    serial and independent of the asynchronous executor and its drain policy:
    a revocation refuses the next claim at once.
    """

    def __init__(self, store: Any, connector: Any, verifier: Any, *, owner_id: str | None = None,
                 lease_ttl_s: float = 60.0, sleep: Callable[[float], Any] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        check_connector(connector)
        if getattr(connector, "max_retries", 0) != 0:
            raise TypeError("A connector must not retry on its own: set its max_retries to 0")
        check_verifier(verifier)
        store._ttl(lease_ttl_s)
        self.store, self.connector, self.verifier = store, connector, verifier
        self.owner_id = (f"action-runner-{uuid4().hex}" if owner_id is None
                         else _text(owner_id, "owner_id"))
        self.lease_ttl_s, self._sleep, self._clock = lease_ttl_s, sleep, clock

    def _action(self, run_id: str, action_id: str) -> PreparedAction:
        """The action, once it is *run_id*'s and the connector's profile is the one it binds."""
        action = load_action(self.store, action_id)
        if action.run_id != run_id:
            raise WorkflowConflictError(f"Action {action_id!r} belongs to another workflow run")
        if check_connector(self.connector) != action.profile:
            raise WorkflowConflictError(
                f"The connector's profile is not the one action {action_id!r} was prepared with")
        return action

    @contextmanager
    def _leased(self, run_id: str) -> Iterator[Any]:
        from smythe.workflow_actions import recover_actions

        lease = self.store.acquire_lease(run_id, self.owner_id, ttl_s=self.lease_ttl_s)
        try:
            recover_actions(self.store, lease)
            yield lease
        finally:
            with suppress(WorkflowLeaseError):
                self.store.release_lease(lease)

    def _renew(self, lease: Any, seconds: float) -> None:
        """Hold the lease for *seconds* more than its usual duration, up to a day."""
        self.store.heartbeat(lease, min(86_400.0, self.lease_ttl_s + seconds))

    def dispatch(self, run_id: str, action_id: str) -> ActionProgress:
        """Make the action's next attempt: exactly one submission under its current approval.

        Under the run's lease, :func:`check_authority` and ``begin_attempt``
        check the approval and content. The payload is then read once and must
        be exactly the bound content, or the attempt is released and
        ``ActionNotAuthorized`` raised before anything is sent. After the
        claim, the connector's ``submit()`` is called once with the action's
        key and those bytes, and its receipt, refusal or ambiguous outcome is
        recorded; an unusable receipt is ambiguous. A receipt never shows the
        result: call :meth:`readback`. Raises what the journal's checks raise.
        """
        action = self._action(run_id, action_id)
        with self._leased(run_id) as lease:
            self._dispatch(lease, action)
        return self._progress(action_id)

    def _dispatch(self, lease: Any, action: PreparedAction) -> None:
        """Check the action's authority and content, then claim and submit one attempt."""
        from smythe import workflow_actions as journal

        approval = check_authority(self.store, action.action_id, self.verifier)
        attempt = journal.begin_attempt(self.store, lease, action.action_id, approval.approval_id)
        payload = _payload(self.store, action)
        if payload is None and action.intent.content is not None:
            problem = "the bytes read for delivery are not its bound content"
            journal.release_attempt(self.store, lease, attempt["attempt_id"], problem)
            raise ActionNotAuthorized(f"Action {action.action_id!r} is not authorized: {problem}")
        self._renew(lease, action.profile.timeout_s)
        permit = journal.claim_dispatch(self.store, lease, attempt["attempt_id"])
        self._submit(action, permit, payload)

    def dispatch_pending(self, run_id: str) -> tuple[ActionProgress, ...]:
        """Dispatch the run's actions that wait for dispatch, one at a time, under one lease.

        They are its ``not_dispatched`` actions with attempts left that were
        prepared with the connector's profile, by slot and sequence. Each is
        dispatched as :meth:`dispatch` does, so its authority is checked again
        before each claim, and in the claim's transaction: a :func:`revoke` or
        :func:`stop_actions` made while an earlier submission was in flight
        refuses every later claim, while that submission's outcome is still
        recorded. A refused action, such as one whose approval was revoked or
        whose run a security stop holds, is returned with ``refusal`` saying
        why, and nothing is sent for it. Returns one progress per action.
        """
        from smythe.workflow_actions import pending

        results, profile = [], check_connector(self.connector)
        with self._leased(run_id) as lease:
            for action in pending(self.store, run_id):
                if action.profile != profile:
                    continue
                try:
                    self._dispatch(lease, action)
                except (ActionStateError, WorkflowBudgetError) as error:
                    results.append(replace(self._progress(action.action_id), next_action=None,
                                           refusal=str(error)))
                else:
                    results.append(self._progress(action.action_id))
        return tuple(results)

    def _submit(self, action: PreparedAction, permit: Any, payload: bytes | None) -> None:
        """Submit once under a claimed *permit* and record the receipt, refusal or ambiguous outcome."""
        from smythe import workflow_actions as journal

        try:
            receipt = self.connector.submit(action.action_key, action.intent, payload,
                                            action.profile.timeout_s)
        except ConnectorRefused as error:
            journal.record_refused(self.store, permit, error)
        except Exception as error:  # anything else may follow a commit
            journal.record_ambiguous(self.store, permit, error)
        else:
            try:
                journal.record_receipt(self.store, permit, receipt)
            except WorkflowValidationError as error:
                journal.record_ambiguous(self.store, permit, error)

    def readback(self, run_id: str, action_id: str) -> ActionProgress:
        """Read the object an acknowledged attempt committed, to find whether it holds the intent.

        Under the run's lease, the connector's ``read()`` is called for the
        receipt's target at most ``readback_reads`` times per attempt in all,
        ``readback_interval_s`` apart, and only when the profile declares
        ``lookup_by_object``. Each read is a ``readback`` record charged the
        profile's ``read_ceiling_nanousd`` and holding its
        :data:`READBACK_VALIDATOR` report. A passed report makes the action
        ``observed``, or ``reconciled`` when an earlier attempt's outcome is
        still unknown, and a rejected one ``compensation_required``. No read
        is made that would take the action past the ceiling of the attempt's
        approval, which lookups by :meth:`reconcile` share. Reads that decide
        nothing leave it ``dispatched``, with ``reconcile`` next once none are
        left or none fits. Without an acknowledged attempt nothing is read.
        """
        from smythe import workflow_actions as journal

        action, report = self._action(run_id, action_id), None
        with self._leased(run_id) as lease:
            status = journal.reconcile_status(self.store, action_id)
            latest = status["attempts"][-1] if status["attempts"] else None
            if (status["action"].state == "dispatched" and status["receipt"] is not None
                    and latest["state"] == "acknowledged" and action.profile.lookup_by_object):
                attempt_id = latest["attempt_id"]
                report = self._read_back(lease, action, attempt_id, status["receipt"],
                                         status["reads"].get(attempt_id, 0), "dispatched",
                                         self._room(status, attempt_id))[0]
        return self._progress(action_id, report)

    def _pace(self, lease: Any, profile: ConnectorProfile, started: float | None) -> None:
        """Wait out ``readback_interval_s`` since the call made at *started*, holding the lease past the next."""
        wait = 0.0 if started is None else max(0.0, started + profile.readback_interval_s - self._clock())
        self._renew(lease, wait + profile.timeout_s)
        if wait:
            self._sleep(wait)

    def _read_back(self, lease: Any, action: PreparedAction, attempt_id: str, receipt: ConnectorReceipt,
                   done: int, undecided: str, room: int | None = None) -> tuple[Any, list[str], bool]:
        """Read the object *receipt* names until a read decides; *done* reads were made before.

        *undecided* is the action's state while reads decide nothing. With
        *room*, the nanoUSD the approval's ceiling has left, a read that would
        not fit is not made. Returns the last report, or None; the readback
        record IDs; and whether the ceiling stopped the reads.
        """
        from smythe import workflow_actions as journal

        profile, report, records, started = action.profile, None, [], None
        for number in range(done + 1, profile.readback_reads + 1):
            if room is not None:
                if room < profile.read_ceiling_nanousd:
                    return report, records, True
                room -= profile.read_ceiling_nanousd
            self._pace(lease, profile, started)
            started, observed, error = self._clock(), None, None
            try:
                observed = self.connector.read(receipt.target, profile.timeout_s)
            except Exception as exc:  # a failed read proves nothing
                error = journal._error(exc)
            report = _verdict(action, receipt, observed, error)
            state = {ValidationOutcome.PASSED: "observed",
                     ValidationOutcome.REJECTED: "compensation_required"}.get(report.outcome, undecided)
            seen = observed if isinstance(observed, ObservedObject) else None
            records.append(journal.record_readback(self.store, lease, attempt_id, {
                "attempt_id": attempt_id, "read": number, "error": error,
                "observed": seen and {"target": seen.target.to_dict(),
                                      "content": seen.content_ref.to_dict(),
                                      "version": seen.version, "key": seen.key},
                "report": report.to_dict()}, profile.read_ceiling_nanousd, state))
            if state != undecided:
                break
        return report, records, False

    def reconcile(self, run_id: str, action_id: str) -> ActionProgress:
        """Settle an outcome that dispatch left unknown, only as the connector's profile declares.

        Under the run's lease, after recovery: with ``lookup_by_key`` the key
        is looked up at most ``readback_reads`` times per attempt,
        ``readback_interval_s`` apart, and a receipt found is read back,
        making the action ``reconciled`` or ``compensation_required``. An
        empty lookup never causes a submission. It proves the write was not
        made (``not_executed``; the action is ``not_dispatched`` again) only
        once a declared ``visibility_bound_s`` plus ``timeout_s`` has passed
        since the latest dispatch claim while the key's window is open. With
        ``deduplicates`` alone, one same-key retry is made within
        ``max_attempts`` while the window is open and :func:`check_authority`
        passes, and its receipt is read back; with neither, nothing is sent.
        An acknowledged attempt with reads left is read back. What cannot be
        settled gets an ``unresolved`` finding, with ``operator_review`` next.
        Lookups and reads are charged ``read_ceiling_nanousd`` within the
        ceiling of the attempt's approval; times are the journal's clock.
        """
        from smythe import workflow_actions as journal

        action, report = self._action(run_id, action_id), None
        profile = action.profile
        with self._leased(run_id) as lease:
            status = journal.reconcile_status(self.store, action_id)
            state, latest = status["action"].state, status["attempts"][-1] if status["attempts"] else None
            if state == "outcome_unknown" and status["unsettled"]:
                attempt_id = status["unsettled"][-1]
                if profile.lookup_by_key:
                    report = self._look_up(lease, action, status, attempt_id)
                elif profile.deduplicates:
                    report = self._retry(lease, action, status, attempt_id)
                else:
                    journal.record_finding(self.store, lease, attempt_id, "unresolved",
                                           "the connector declares neither lookup by key nor deduplication, "
                                           "so nothing can show whether the write took effect")
            elif state == "dispatched" and latest["state"] == "acknowledged":
                report = self._confirm(lease, action, status, latest["attempt_id"], status["receipt"], [],
                                       "dispatched")
        return self._progress(action_id, report)

    def _room(self, status: dict[str, Any], attempt_id: str) -> int:
        """The nanoUSD the ceiling of *attempt_id*'s approval leaves for the action."""
        (approval_id,) = [item["approval_id"] for item in status["attempts"] if item["attempt_id"] == attempt_id]
        return status["ceilings"][approval_id] - status["spent"]

    def _look_up(self, lease: Any, action: PreparedAction, status: dict[str, Any],
                 attempt_id: str) -> ValidationReport | None:
        """Look the action's key up until a lookup finds its receipt or proves it absent."""
        from smythe import workflow_actions as journal

        profile, evidence, reason, started = action.profile, [], None, None
        room, found = self._room(status, attempt_id), status["found"].get(attempt_id)
        first, last = min(status["claims"].values()), max(status["claims"].values())
        for number in range(status["lookups"].get(attempt_id, 0) + 1,
                            profile.readback_reads + 1 if found is None else 0):
            if room < profile.read_ceiling_nanousd:
                reason = "the approval's ceiling leaves no room for another lookup"
                break
            self._pace(lease, profile, started)
            started, now, receipt, error = self._clock(), self.store._clock_ns(), None, None
            try:
                answer = self.connector.lookup(action.action_key, action.intent.target, profile.timeout_s)
            except Exception as exc:  # a failed lookup proves nothing
                error = journal._error(exc)
            else:
                if answer is None or _receipt_of(action.action_key, action.intent, answer):
                    receipt = answer
                else:
                    error = "The lookup answered with something other than the action's receipt"
            evidence.append(journal.record_lookup(self.store, lease, attempt_id, number, receipt, error,
                                                  profile.read_ceiling_nanousd))
            room -= profile.read_ceiling_nanousd
            if receipt is not None:
                found = receipt
                break
            reason = ("no lookup answered" if error is not None
                      else _absence_unproven(profile, first, last, now))
            if reason is None:
                journal.record_finding(self.store, lease, attempt_id, "not_executed",
                                       "an empty lookup after the visibility bound, inside the key's window, "
                                       "shows that the write was not made", evidence)
                return None
        if found is None:
            journal.record_finding(self.store, lease, attempt_id, "unresolved",
                                   reason or "no lookup is left, and none found the key", evidence)
            return None
        return self._confirm(lease, action, status, attempt_id, found, evidence, "outcome_unknown", room)

    def _retry(self, lease: Any, action: PreparedAction, status: dict[str, Any],
               attempt_id: str) -> ValidationReport | None:
        """Submit once more with the same key while the service still deduplicates it; read the receipt back."""
        from smythe import workflow_actions as journal

        profile, payload, permit = action.profile, None, None
        claimed = sum(item["state"] not in ("prepared", "released") for item in status["attempts"])
        problem = (None if _window_open(profile, min(status["claims"].values()), self.store._clock_ns())
                   else "the key's deduplication window may have closed, so a retry could take effect twice")
        if problem is None and claimed >= profile.max_attempts:
            problem = f"the action has made all {profile.max_attempts} of its attempts"
        if problem is None:
            try:
                approval = check_authority(self.store, action.action_id, self.verifier)
                payload = _payload(self.store, action)
                if payload is None and action.intent.content is not None:
                    raise ActionNotAuthorized("the bytes read for delivery are not its bound content")
                self._renew(lease, profile.timeout_s)
                permit = journal.claim_retry(self.store, lease, action.action_id, approval.approval_id)
            except (ActionStateError, WorkflowBudgetError) as error:
                problem = f"no retry is allowed: {error}"
        if problem is not None:
            journal.record_finding(self.store, lease, attempt_id, "unresolved", problem)
            return None
        self._submit(action, permit, payload)
        status = journal.reconcile_status(self.store, action.action_id)
        retry = status["attempts"][-1]
        if retry["state"] != "acknowledged":
            return None
        return self._confirm(lease, action, status, retry["attempt_id"], status["receipt"], [], "dispatched")

    def _confirm(self, lease: Any, action: PreparedAction, status: dict[str, Any], attempt_id: str,
                 receipt: ConnectorReceipt, evidence: list[str], undecided: str,
                 room: int | None = None) -> ValidationReport | None:
        """Read back the object *receipt* names; record an unresolved finding when no read decides."""
        from smythe import workflow_actions as journal

        report, full = None, False
        if action.profile.lookup_by_object:
            report, records, full = self._read_back(
                lease, action, attempt_id, receipt, status["reads"].get(attempt_id, 0), undecided,
                self._room(status, attempt_id) if room is None else room)
            evidence = [*evidence, *records]
        if report is None or report.outcome is ValidationOutcome.ERROR:
            journal.record_finding(
                self.store, lease, attempt_id, "unresolved",
                "the connector declares no lookup by object, so no read can confirm the receipt"
                if not action.profile.lookup_by_object
                else "the approval's ceiling leaves no room for another read" if full
                else "no read confirmed the receipt's object within the read bound", evidence)
        return report

    def _progress(self, action_id: str, report: ValidationReport | None = None) -> ActionProgress:
        from smythe.workflow_actions import reconcile_status

        status = reconcile_status(self.store, action_id)
        action, attempts, finding = status["action"], status["attempts"], status["finding"]
        profile, latest = action.profile, attempts[-1] if attempts else None
        claimed = sum(attempt["state"] not in ("prepared", "released") for attempt in attempts)
        # An unresolved finding stands until another attempt begins.
        review = ("operator_review" if finding is not None and finding.get("finding") == "unresolved"
                  and finding.get("attempts") == len(attempts) else "reconcile")
        next_action = {"outcome_unknown": review, "compensation_required": "compensate"}.get(action.state)
        if action.state == "not_dispatched" and claimed < profile.max_attempts:
            next_action = "dispatch"
        elif action.state == "dispatched":
            next_action = ("readback" if latest["state"] == "acknowledged" and profile.lookup_by_object
                           and status["reads"].get(latest["attempt_id"], 0) < profile.readback_reads
                           and self._room(status, latest["attempt_id"]) >= profile.read_ceiling_nanousd
                           else review)
        return ActionProgress(action_id, action.state, latest, report, next_action)


__all__ = [
    "ACTION_CONTRACT_VERSION", "ACTION_STATES", "READBACK_VALIDATOR", "ActionApproval",
    "ActionIntent", "ActionNotAuthorized", "ActionProgress", "ActionRunner", "ActionStateError",
    "ActionTarget", "ApprovalRejected", "ApprovalVerifier", "ConnectorAmbiguous",
    "ConnectorProfile", "ConnectorReceipt", "ConnectorRefused", "ContentRef", "ObservedObject",
    "PreparedAction", "Principal", "WriteConnector", "action_key", "approve", "check_authority",
    "check_connector", "check_verifier", "load_action", "prepare_action", "record_notice",
    "reject", "revoke", "stop_actions",
]
