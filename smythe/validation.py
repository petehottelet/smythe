"""Validation findings and reports shared by deterministic checks.

These types need no image library. ``smythe.assets`` re-exports them for image
validation, and delivery acceptance records the outcome and identity of the
validators that produce them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json_data(value: Any, where: str) -> Any:
    """A detached copy of *value* as JSON data; tuples become lists."""

    def copy(item: Any, active: set[int]) -> Any:
        if item is None or type(item) in (bool, int):
            return item
        if isinstance(item, str):
            return str.__str__(item)
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{where} must not contain NaN or infinity")
            return item
        if not isinstance(item, (list, tuple, dict)):
            raise TypeError(f"{where} must be JSON data, not {type(item).__name__}")
        if id(item) in active:
            raise ValueError(f"{where} must not contain cycles")
        active.add(id(item))
        try:
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise TypeError(f"{where} keys must be strings")
                return {str.__str__(key): copy(child, active) for key, child in item.items()}
            return [copy(child, active) for child in item]
        finally:
            active.discard(id(item))

    try:
        return copy(value, set())
    except RecursionError as exc:
        raise ValueError(f"{where} is nested too deeply") from exc


def _json_sha256(value: Any, where: str) -> str:
    """SHA-256 of *value* as canonical JSON: sorted keys, no spaces, UTF-8."""
    canonical = json.dumps(
        _json_data(value, where), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, where: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{where} must be 64 lowercase hexadecimal characters")


def _require_keys(
    data: Any, what: str, required: frozenset[str], optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Check that *data* is an object holding exactly the expected keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be a JSON object")
    missing = sorted(required - data.keys())
    unknown = sorted(data.keys() - required - optional)
    if missing:
        raise ValueError(f"{what} is missing keys: {missing}")
    if unknown:
        raise ValueError(f"{what} has unknown keys: {unknown}")
    return data


class FindingSeverity(StrEnum):
    """Whether a finding can fail deterministic acceptance."""

    HARD = "hard"
    ADVISORY = "advisory"


class ValidationOutcome(StrEnum):
    """What a validator decided about its subject.

    ``error`` means the validator could not decide, for example because its
    input could not be read. It is neither a pass nor a rejection.
    """

    PASSED = "passed"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidatorIdentity:
    """The name, version and configuration digest that identify a validator.

    ``config_sha256`` is the SHA-256 of the validator's configuration as
    canonical JSON; :meth:`for_config` computes it. Changing a validator's
    code or configuration should change its version or digest.
    """

    name: str
    version: str
    config_sha256: str

    def __post_init__(self) -> None:
        for label in ("name", "version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"validator {label} must be a non-empty string")
        _require_sha256(self.config_sha256, "validator config_sha256")

    @classmethod
    def for_config(cls, name: str, version: str, config: Any = None) -> ValidatorIdentity:
        """The identity of a validator configured by JSON *config* (``{}`` when None)."""
        return cls(name, version, _json_sha256({} if config is None else config, "validator config"))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "config_sha256": self.config_sha256}

    @classmethod
    def from_dict(cls, data: Any) -> ValidatorIdentity:
        _require_keys(data, "validator identity", frozenset({"name", "version", "config_sha256"}))
        return cls(data["name"], data["version"], data["config_sha256"])


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One stable, machine-readable validation observation."""

    code: str
    message: str
    severity: FindingSeverity = FindingSeverity.HARD
    expected: object | None = None
    observed: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("finding code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("finding message must be a non-empty string")
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        try:
            severity = FindingSeverity(self.severity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown finding severity: {self.severity!r}") from exc
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, Any]:
        """This finding as JSON data; tuples in ``expected`` and ``observed`` become lists."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "expected": _json_data(self.expected, "finding expected value"),
            "observed": _json_data(self.observed, "finding observed value"),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ValidationFinding:
        _require_keys(data, "validation finding", frozenset({"code", "message", "severity"}),
                      frozenset({"expected", "observed"}))
        try:
            expected = _json_data(data.get("expected"), "finding expected value")
            observed = _json_data(data.get("observed"), "finding observed value")
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return cls(data["code"], data["message"], data["severity"], expected, observed)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A validator's decision about one subject.

    Unless ``outcome`` is given, a report with no hard findings is ``passed``
    and one with any hard finding is ``rejected``; advisory findings never
    decide it. A validator that could not decide reports ``outcome="error"``,
    which never passes. A stated outcome must agree with the findings: a
    passed report has no hard findings and a rejected report has at least
    one. ``validator`` optionally identifies the validator that produced it.

    ``dataclasses.replace()`` derives an outcome that was not stated again
    from the new findings, and checks a stated outcome against them.
    :meth:`from_dict` states the outcome its data records.
    """

    path: str
    findings: tuple[ValidationFinding, ...] = ()
    outcome: ValidationOutcome | None = None
    validator: ValidatorIdentity | None = None
    # The outcome derived from the findings, or None when it was stated.
    # dataclasses.replace() passes it back beside the unchanged outcome, which
    # is how a copy tells a derived outcome from one its caller stated.
    _derived_outcome: ValidationOutcome | None = field(
        default=None, kw_only=True, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        if any(not isinstance(finding, ValidationFinding) for finding in findings):
            raise TypeError("findings must contain ValidationFinding values")
        object.__setattr__(self, "findings", findings)
        hard = any(finding.severity == FindingSeverity.HARD for finding in findings)
        stated = self.outcome
        if stated is not None and stated is self._derived_outcome:
            stated = None  # a derived outcome passed back unchanged is derived again
        if stated is None:
            outcome = ValidationOutcome.REJECTED if hard else ValidationOutcome.PASSED
        else:
            try:
                outcome = ValidationOutcome(stated)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown validation outcome: {stated!r}") from exc
            if outcome is ValidationOutcome.PASSED and hard:
                raise ValueError("a passed report cannot contain hard findings")
            if outcome is ValidationOutcome.REJECTED and not hard:
                raise ValueError("a rejected report needs at least one hard finding")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "_derived_outcome", outcome if stated is None else None)
        if self.validator is not None and not isinstance(self.validator, ValidatorIdentity):
            raise TypeError("validator must be a ValidatorIdentity or None")

    @property
    def hard_findings(self) -> tuple[ValidationFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.severity == FindingSeverity.HARD
        )

    @property
    def advisories(self) -> tuple[ValidationFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.severity == FindingSeverity.ADVISORY
        )

    @property
    def passed(self) -> bool:
        return self.outcome is ValidationOutcome.PASSED

    def to_dict(self) -> dict[str, Any]:
        """This report as JSON data."""
        if not isinstance(self.path, str):
            raise TypeError("report path must be a string to serialize")
        return {
            "path": self.path,
            "outcome": self.outcome.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "validator": None if self.validator is None else self.validator.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ValidationReport:
        _require_keys(data, "validation report", frozenset({"path", "outcome", "findings"}),
                      frozenset({"validator"}))
        if not isinstance(data["path"], str):
            raise ValueError("validation report path must be a string")
        if not isinstance(data["outcome"], str):
            raise ValueError(f"unknown validation outcome: {data['outcome']!r}")
        if not isinstance(data["findings"], list):
            raise ValueError("validation report findings must be a list")
        validator = data.get("validator")
        return cls(
            data["path"],
            tuple(ValidationFinding.from_dict(item) for item in data["findings"]),
            data["outcome"],
            None if validator is None else ValidatorIdentity.from_dict(validator),
        )


def hard_finding(
    code: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        message=message,
        severity=FindingSeverity.HARD,
        expected=expected,
        observed=observed,
    )


def advisory_finding(
    code: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        message=message,
        severity=FindingSeverity.ADVISORY,
        expected=expected,
        observed=observed,
    )


__all__ = [
    "FindingSeverity",
    "ValidationFinding",
    "ValidationOutcome",
    "ValidationReport",
    "ValidatorIdentity",
    "advisory_finding",
    "hard_finding",
]
