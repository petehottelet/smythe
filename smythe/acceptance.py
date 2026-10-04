"""Delivery acceptance: the policy that decides a deliverable and its record.

Delivery acceptance is separate from execution and cost. A run can complete,
with its cost fully known, and still deliver output its policy did not accept.
A :class:`DeliveryAcceptance` record binds one decision to the exact delivered
bytes, the nodes and generations they came from, the task, the policy and the
validators that decided it. Results and checkpoints use the key
``delivery_acceptance`` for it.

Validators are trusted local code registered in-process: an object with a
:class:`~smythe.validation.ValidatorIdentity` and a synchronous
``validate(subject)`` that returns a
:class:`~smythe.validation.ValidationReport`. Registration refuses providers,
coroutine functions, awaitables and an ``async`` ``validate``. That check is a
caller contract, not a sandbox: it cannot detect a synchronous validator that
performs network I/O or paid calls, which such a validator must not do.
Policies and records store only validator names, versions and configuration
digests, never code.
"""

from __future__ import annotations

import hashlib
import inspect
# Imported at runtime, not only for type checking, so that
# typing.get_type_hints() can resolve the dataclass field annotations.
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from smythe.task import Task, task_to_dict
from smythe.validation import (
    ValidationOutcome,
    ValidationReport,
    ValidatorIdentity,
    _json_sha256,
    _require_keys,
    _require_sha256,
)

DELIVERY_ACCEPTANCE_VERSION = 1
SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS = (1,)
ACCEPTANCE_POLICY_VERSION = 1

# What a record's decision covers: the synthesized text output.
_SCOPES = frozenset({"output"})


class AcceptanceMode(StrEnum):
    """``advisory`` keeps legacy behavior and records no decision; ``strict`` decides delivery."""

    ADVISORY = "advisory"
    STRICT = "strict"


class AcceptanceDecision(StrEnum):
    """The delivery acceptance decision a record holds.

    ``not_evaluated`` means a record exists but nothing has been decided yet.
    ``accepted`` means every required check passed on the exact delivered
    bytes. ``rejected`` is a valid negative decision. ``needs_review`` occurs
    only when a policy selects it. ``indeterminate`` means no valid decision
    could be made, for example because a validator raised.
    """

    NOT_EVALUATED = "not_evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    INDETERMINATE = "indeterminate"


def text_sha256(text: str) -> str:
    """SHA-256 of *text* encoded as UTF-8, the bytes a text deliverable is checked as."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_sha256(task: Task | None) -> str | None:
    """SHA-256 of a task's canonical JSON snapshot, or None without a task."""
    data = task_to_dict(task)
    return None if data is None else _json_sha256(data, "task snapshot")


@dataclass(frozen=True, slots=True)
class ValidationSubject:
    """The exact bytes a validator inspects, bound to their SHA-256."""

    data: bytes
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("a validation subject holds bytes; use from_text() for a string")
        data = bytes(self.data)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sha256", hashlib.sha256(data).hexdigest())

    @classmethod
    def from_text(cls, text: str) -> ValidationSubject:
        """The UTF-8 bytes of *text*; ``sha256`` equals :func:`text_sha256`."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cls(text.encode("utf-8"))

    def text(self) -> str:
        """The bytes decoded as UTF-8; raises UnicodeDecodeError for other bytes."""
        return self.data.decode("utf-8")


class Validator(Protocol):
    """Trusted local code that decides whether delivered bytes are acceptable.

    ``validate`` is synchronous, makes no provider calls or network requests,
    and reports ``error`` rather than guessing when it cannot decide.
    """

    @property
    def identity(self) -> ValidatorIdentity: ...

    def validate(self, subject: ValidationSubject) -> ValidationReport: ...


def _is_async(function: Any) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function)


def check_validator(validator: object) -> ValidatorIdentity:
    """Check that *validator* can be registered, and return its identity.

    Raises TypeError for a ``Provider``, a class, a coroutine function, an
    awaitable (a coroutine is closed first), an object without a synchronous
    ``validate()`` or without a :class:`ValidatorIdentity`. This structural
    check cannot detect a synchronous ``validate()`` that performs I/O.
    """
    from smythe.provider import Provider

    if isinstance(validator, Provider):
        raise TypeError(
            "a Provider cannot be a validator: validators are local code that makes "
            "no provider calls"
        )
    if inspect.isawaitable(validator):
        if inspect.iscoroutine(validator):
            validator.close()
        raise TypeError(
            f"an awaitable ({type(validator).__name__}) cannot be a validator; register "
            "an object with a synchronous validate()"
        )
    if isinstance(validator, type):
        raise TypeError(f"register a validator instance, not the class {validator.__name__}")
    if _is_async(validator):
        raise TypeError("a coroutine function cannot be a validator; validate() must be synchronous")
    validate = getattr(validator, "validate", None)
    if not callable(validate):
        raise TypeError(f"validator {type(validator).__name__} has no validate() method")
    if _is_async(validate):
        raise TypeError(
            f"validator {type(validator).__name__} has an async validate(); it must be synchronous"
        )
    identity = getattr(validator, "identity", None)
    if not isinstance(identity, ValidatorIdentity):
        raise TypeError(
            f"validator {type(validator).__name__} identity must be a ValidatorIdentity, "
            f"not {type(identity).__name__}"
        )
    return identity


def _choice(enum: type[StrEnum], value: Any, what: str) -> Any:
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown {what}: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    """How a run decides whether its deliverable is accepted.

    ``advisory``, the default, keeps legacy behavior: gates advise, and no
    delivery acceptance record is made. ``strict`` decides delivery:
    ``final_validators`` check the exact delivered bytes, and with
    ``allow_gate_acceptance`` a passed gate may accept output that is
    byte-for-byte the gated node's result. An exhausted gate whose final
    verdict fails gives ``on_exhausted`` (``rejected`` or ``needs_review``); a
    missing, malformed or erroring verdict gives ``on_invalid``
    (``indeterminate`` or ``needs_review``).

    Validators are checked by :func:`check_validator` when the policy is
    built, and their identities are captured then. :meth:`to_dict` lists those
    identities sorted by name, never code, and :attr:`sha256` hashes it.
    """

    mode: AcceptanceMode = AcceptanceMode.ADVISORY
    final_validators: Sequence[Validator] = ()
    on_exhausted: AcceptanceDecision = AcceptanceDecision.REJECTED
    on_invalid: AcceptanceDecision = AcceptanceDecision.INDETERMINATE
    allow_gate_acceptance: bool = True
    validator_identities: tuple[ValidatorIdentity, ...] = field(init=False)

    def __post_init__(self) -> None:
        mode = _choice(AcceptanceMode, self.mode, "acceptance mode")
        if not isinstance(self.final_validators, (list, tuple)):
            raise TypeError("final_validators must be a list or tuple of validators")
        validators = tuple(self.final_validators)
        identities = tuple(check_validator(validator) for validator in validators)
        names = [identity.name for identity in identities]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"final_validators has duplicate validator names: {duplicates}")
        on_exhausted = _choice(AcceptanceDecision, self.on_exhausted, "on_exhausted decision")
        if on_exhausted not in (AcceptanceDecision.REJECTED, AcceptanceDecision.NEEDS_REVIEW):
            raise ValueError("on_exhausted must be 'rejected' or 'needs_review'")
        on_invalid = _choice(AcceptanceDecision, self.on_invalid, "on_invalid decision")
        if on_invalid not in (AcceptanceDecision.INDETERMINATE, AcceptanceDecision.NEEDS_REVIEW):
            raise ValueError("on_invalid must be 'indeterminate' or 'needs_review'")
        if type(self.allow_gate_acceptance) is not bool:
            raise ValueError("allow_gate_acceptance must be a bool")
        if mode is AcceptanceMode.ADVISORY and (
            validators or on_exhausted is not AcceptanceDecision.REJECTED
            or on_invalid is not AcceptanceDecision.INDETERMINATE
            or not self.allow_gate_acceptance
        ):
            raise ValueError(
                "an advisory policy records no delivery decision, so it takes no final "
                "validators or decision settings; use mode='strict'"
            )
        if mode is AcceptanceMode.STRICT and not validators and not self.allow_gate_acceptance:
            raise ValueError(
                "a strict policy without final validators or gate acceptance could never accept"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "final_validators", validators)
        object.__setattr__(self, "on_exhausted", on_exhausted)
        object.__setattr__(self, "on_invalid", on_invalid)
        object.__setattr__(self, "validator_identities", identities)

    def to_dict(self) -> dict[str, Any]:
        """The policy as canonical JSON data, with validator identities sorted by name."""
        return {
            "version": ACCEPTANCE_POLICY_VERSION,
            "mode": self.mode.value,
            "final_validators": [
                identity.to_dict()
                for identity in sorted(self.validator_identities, key=lambda item: item.name)
            ],
            "on_exhausted": self.on_exhausted.value,
            "on_invalid": self.on_invalid.value,
            "allow_gate_acceptance": self.allow_gate_acceptance,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 of :meth:`to_dict` as canonical JSON: the policy's identity."""
        return _json_sha256(self.to_dict(), "acceptance policy")


@dataclass(frozen=True, slots=True)
class AcceptanceSource:
    """A node whose result fed the delivered output, at the generation used."""

    node_id: str
    execution_generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("source node_id must be a non-empty string")
        if type(self.execution_generation) is not int or self.execution_generation < 0:
            raise ValueError("source execution_generation must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "execution_generation": self.execution_generation}

    @classmethod
    def from_dict(cls, data: Any) -> AcceptanceSource:
        _require_keys(data, "acceptance source", frozenset({"node_id", "execution_generation"}))
        return cls(data["node_id"], data["execution_generation"])


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    """One validator's identity and the outcome it reported."""

    validator: ValidatorIdentity
    outcome: ValidationOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.validator, ValidatorIdentity):
            raise ValueError("validator must be a ValidatorIdentity")
        object.__setattr__(self, "outcome", _choice(ValidationOutcome, self.outcome,
                                                    "validation outcome"))

    def to_dict(self) -> dict[str, Any]:
        return {"validator": self.validator.to_dict(), "outcome": self.outcome.value}

    @classmethod
    def from_dict(cls, data: Any) -> ValidatorResult:
        _require_keys(data, "validator result", frozenset({"validator", "outcome"}))
        if not isinstance(data["outcome"], str):
            raise ValueError(f"Unknown validation outcome: {data['outcome']!r}")
        return cls(ValidatorIdentity.from_dict(data["validator"]), data["outcome"])


_RECORD_KEYS = frozenset({
    "version", "decision", "reasons", "output_sha256", "strategy", "sources",
    "used_fallback", "task_sha256", "policy_sha256", "validators", "scope",
})


@dataclass(frozen=True, slots=True)
class DeliveryAcceptance:
    """A versioned delivery acceptance decision and what it is bound to.

    Every decision except ``not_evaluated`` binds ``output_sha256`` (the
    SHA-256 of the delivered bytes) and the synthesis ``strategy``; a decision
    other than ``accepted`` states at least one reason, and an ``accepted``
    record holds no validator outcome other than ``passed``. ``sources`` are
    the nodes the output came from with their ``execution_generation``,
    ``used_fallback`` is True when synthesis fell back to a node that was not
    a terminal deliverable, ``task_sha256`` is :func:`task_sha256` of the
    run's task, ``policy_sha256`` is the deciding policy's
    :attr:`AcceptancePolicy.sha256`, and ``scope`` names what was evaluated
    (``output``: the synthesized text).

    :meth:`from_dict` rejects unknown record versions, unknown decisions and
    unknown keys rather than guessing what a newer record means.
    """

    decision: AcceptanceDecision
    policy_sha256: str
    output_sha256: str | None = None
    strategy: str | None = None
    sources: tuple[AcceptanceSource, ...] = ()
    used_fallback: bool = False
    task_sha256: str | None = None
    validators: tuple[ValidatorResult, ...] = ()
    reasons: tuple[str, ...] = ()
    scope: str = "output"
    version: int = DELIVERY_ACCEPTANCE_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS:
            raise ValueError(
                f"Unsupported delivery acceptance record version {self.version!r}; this build "
                f"reads {list(SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS)}"
            )
        decision = _choice(AcceptanceDecision, self.decision, "delivery acceptance decision")
        object.__setattr__(self, "decision", decision)
        _require_sha256(self.policy_sha256, "policy_sha256")
        for name in ("output_sha256", "task_sha256"):
            if getattr(self, name) is not None:
                _require_sha256(getattr(self, name), name)
        if self.strategy is not None and (not isinstance(self.strategy, str) or not self.strategy):
            raise ValueError("strategy must be a non-empty string or None")
        if type(self.used_fallback) is not bool:
            raise ValueError("used_fallback must be a bool")
        # Check the type first: an unhashable scope cannot be looked up in a set.
        if not isinstance(self.scope, str) or self.scope not in _SCOPES:
            raise ValueError(f"Unknown delivery acceptance scope: {self.scope!r}")
        sources = self._items("sources", AcceptanceSource)
        validators = self._items("validators", ValidatorResult)
        reasons = self._items("reasons", str)
        if any(not reason for reason in reasons):
            raise ValueError("reasons must be non-empty strings")
        _unique([source.node_id for source in sources], "source node ids")
        _unique([result.validator.name for result in validators], "validator names")
        if decision is not AcceptanceDecision.NOT_EVALUATED:
            if self.output_sha256 is None:
                raise ValueError(f"a {decision.value} record must bind output_sha256")
            if self.strategy is None:
                raise ValueError(f"a {decision.value} record must name its synthesis strategy")
        if decision is AcceptanceDecision.ACCEPTED:
            if any(result.outcome is not ValidationOutcome.PASSED for result in validators):
                raise ValueError("an accepted record cannot hold a validator outcome other than passed")
        elif decision is not AcceptanceDecision.NOT_EVALUATED and not reasons:
            raise ValueError(f"a {decision.value} record needs at least one reason")

    def _items(self, name: str, kind: type) -> tuple[Any, ...]:
        value = getattr(self, name)
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, kind) for item in value):
            raise ValueError(f"{name} must be a sequence of {kind.__name__} values")
        items = tuple(value)
        object.__setattr__(self, name, items)
        return items

    @property
    def accepted(self) -> bool:
        """True only when the decision is ``accepted``."""
        return self.decision is AcceptanceDecision.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON data, stored under the ``delivery_acceptance`` key."""
        return {
            "version": self.version,
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "output_sha256": self.output_sha256,
            "strategy": self.strategy,
            "sources": [source.to_dict() for source in self.sources],
            "used_fallback": self.used_fallback,
            "task_sha256": self.task_sha256,
            "policy_sha256": self.policy_sha256,
            "validators": [result.to_dict() for result in self.validators],
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Any) -> DeliveryAcceptance:
        """Read a record written by :meth:`to_dict`; raises ValueError for anything else."""
        if not isinstance(data, dict):
            raise ValueError("A delivery acceptance record must be a JSON object")
        version = data.get("version")
        if type(version) is not int or version not in SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS:
            raise ValueError(
                f"Unsupported delivery acceptance record version {version!r}; this build "
                f"reads {list(SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS)}"
            )
        _require_keys(data, "delivery acceptance record", _RECORD_KEYS)
        if not isinstance(data["decision"], str):
            raise ValueError(f"Unknown delivery acceptance decision: {data['decision']!r}")
        for name in ("reasons", "sources", "validators"):
            if not isinstance(data[name], list):
                raise ValueError(f"delivery acceptance {name} must be a list")
        return cls(
            decision=data["decision"],
            policy_sha256=data["policy_sha256"],
            output_sha256=data["output_sha256"],
            strategy=data["strategy"],
            sources=tuple(AcceptanceSource.from_dict(item) for item in data["sources"]),
            used_fallback=data["used_fallback"],
            task_sha256=data["task_sha256"],
            validators=tuple(ValidatorResult.from_dict(item) for item in data["validators"]),
            reasons=tuple(data["reasons"]),
            scope=data["scope"],
            version=version,
        )


def _unique(values: list[str], what: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"delivery acceptance has duplicate {what}: {duplicates}")


__all__ = [
    "ACCEPTANCE_POLICY_VERSION",
    "DELIVERY_ACCEPTANCE_VERSION",
    "SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS",
    "AcceptanceDecision",
    "AcceptanceMode",
    "AcceptancePolicy",
    "AcceptanceSource",
    "DeliveryAcceptance",
    "ValidationSubject",
    "Validator",
    "ValidatorResult",
    "check_validator",
    "task_sha256",
    "text_sha256",
]
