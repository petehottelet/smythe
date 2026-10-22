"""Preview contract types for external actions that write accepted content to a service.

An :class:`ActionIntent` says what one logical action writes and where, and
:func:`action_key` names the key every attempt sends. A
:class:`ConnectorProfile` declares what a connector's service guarantees.
Connectors and approval verifiers are trusted host code with synchronous
protocols, checked only structurally: a caller contract, not a sandbox.
Nothing stores or dispatches actions yet, and ``import smythe`` does not load
this module.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import KW_ONLY, MISSING, dataclass, field, fields
from typing import Any, Protocol, Self

from smythe._canonical import canonical_bytes, sha256_hex
from smythe.acceptance import _calls_async, _is_async
from smythe.provider import Provider
from smythe.validation import _require_keys, _require_sha256
from smythe.workflow_binding import WorkflowBindingError, json_snapshot
from smythe.workflow_store import WorkflowStateError, _text

ACTION_CONTRACT_VERSION = "smythe.action.v1"
_CONTRACT_VERSIONS = (ACTION_CONTRACT_VERSION,)

# ``canceled`` closes an action that was rejected, expired or revoked before
# any dispatch.
ACTION_STATES = (
    "not_dispatched", "dispatched", "outcome_unknown", "observed", "reconciled",
    "compensation_required", "canceled",
)
_COMPENSATION = ("new_action", "unsupported")


class ConnectorRefused(Exception):
    """The service definitely did not commit the write."""


class ConnectorAmbiguous(Exception):
    """The service may have committed the write: its answer was lost or unreadable."""


class ApprovalRejected(Exception):
    """An approval verifier refused an assertion."""


class ActionStateError(WorkflowStateError):
    """The requested transition is not allowed from the action's recorded state."""


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


__all__ = [
    "ACTION_CONTRACT_VERSION", "ACTION_STATES", "ActionIntent", "ActionStateError",
    "ActionTarget", "ApprovalRejected", "ApprovalVerifier", "ConnectorAmbiguous",
    "ConnectorProfile", "ConnectorReceipt", "ConnectorRefused", "ContentRef", "ObservedObject",
    "Principal", "WriteConnector", "action_key", "check_connector", "check_verifier",
]
