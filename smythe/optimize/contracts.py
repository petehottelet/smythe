"""Immutable, versioned contracts for bounded Autotune experiments.

This module defines data only.  It deliberately contains no engine, ledger,
provider, CLI, or statistics implementation.  Every wire representation is
strict JSON and every identity is derived from canonical JSON bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

CONTRACT_VERSION = 1
_MAX_INT = (1 << 63) - 1
_MAX_DEPTH = 32
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
_DOTTED_FIELD_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{0,63}"
    r"(?:\.[A-Za-z][A-Za-z0-9_-]{0,63}){0,15}$"
)
_CANDIDATE_ID_RE = re.compile(r"^cand_[0-9a-f]{64}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ContractValidationError(ValueError):
    """Raised when an optimization contract or candidate is unsafe."""


class ObjectiveDirection(str, Enum):
    """Whether lower or higher values are preferable for one metric."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MutableValueType(str, Enum):
    """JSON scalar type accepted for one mutable policy field."""

    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON for a JSON-safe value.

    Objects are key-sorted, whitespace-free, and rejected if they contain
    non-finite numbers, non-string keys, unsupported values, or excessive
    nesting.  Tuples and immutable mappings used internally are emitted as
    ordinary JSON arrays and objects.
    """

    normalized = _json_copy(value, context="value")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_prefixed(value: object) -> str:
    """Hash a JSON-safe value as ``sha256:<lowercase hex>``."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class MetricObjective:
    """One measured objective plus optional deterministic safety gates."""

    name: str
    direction: ObjectiveDirection
    primary: bool = False
    hard_min: float | None = None
    hard_max: float | None = None
    max_regression: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _dotted_name(self.name, "objective.name"))
        if isinstance(self.direction, str):
            try:
                object.__setattr__(self, "direction", ObjectiveDirection(self.direction))
            except ValueError as exc:
                raise ContractValidationError(
                    f"objective.direction is unsupported: {self.direction!r}"
                ) from exc
        if not isinstance(self.direction, ObjectiveDirection):
            raise ContractValidationError("objective.direction is unsupported")
        if not isinstance(self.primary, bool):
            raise ContractValidationError("objective.primary must be a boolean")

        hard_min = _optional_metric(self.hard_min, "objective.hard_min")
        hard_max = _optional_metric(self.hard_max, "objective.hard_max")
        max_regression = _optional_metric(
            self.max_regression,
            "objective.max_regression",
        )
        if hard_min is not None and hard_max is not None and hard_min > hard_max:
            raise ContractValidationError(
                "objective.hard_min must not exceed objective.hard_max"
            )
        if max_regression is not None and max_regression < 0:
            raise ContractValidationError(
                "objective.max_regression must be non-negative"
            )
        object.__setattr__(self, "hard_min", hard_min)
        object.__setattr__(self, "hard_max", hard_max)
        object.__setattr__(self, "max_regression", max_regression)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "primary": self.primary,
            "hard_min": self.hard_min,
            "hard_max": self.hard_max,
            "max_regression": self.max_regression,
        }

    @classmethod
    def from_dict(cls, data: object) -> "MetricObjective":
        item = _mapping(data, "objective")
        _strict_fields(
            item,
            allowed={
                "name",
                "direction",
                "primary",
                "hard_min",
                "hard_max",
                "max_regression",
            },
            required={"name", "direction"},
            context="objective",
        )
        return cls(
            name=item["name"],
            direction=item["direction"],
            primary=item.get("primary", False),
            hard_min=item.get("hard_min"),
            hard_max=item.get("hard_max"),
            max_regression=item.get("max_regression"),
        )


@dataclass(frozen=True, slots=True)
class MutableFieldRule:
    """Immutable value domain for one allowlisted mutable policy field."""

    value_type: MutableValueType
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[bool | int | float | str, ...] | None = None

    def __post_init__(self) -> None:
        value_type = self.value_type
        if isinstance(value_type, str):
            try:
                value_type = MutableValueType(value_type)
            except ValueError as exc:
                raise ContractValidationError(
                    f"mutable rule type is unsupported: {self.value_type!r}"
                ) from exc
        if not isinstance(value_type, MutableValueType):
            raise ContractValidationError("mutable rule type is unsupported")

        minimum = self.minimum
        maximum = self.maximum
        if value_type is MutableValueType.INTEGER:
            minimum = _optional_rule_integer(minimum, "mutable rule minimum")
            maximum = _optional_rule_integer(maximum, "mutable rule maximum")
        elif value_type is MutableValueType.NUMBER:
            minimum = _optional_metric(minimum, "mutable rule minimum")
            maximum = _optional_metric(maximum, "mutable rule maximum")
        elif minimum is not None or maximum is not None:
            raise ContractValidationError(
                "mutable rule minimum and maximum require an integer or number type"
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ContractValidationError(
                "mutable rule minimum must not exceed maximum"
            )

        choices = self.choices
        normalized_choices: tuple[bool | int | float | str, ...] | None = None
        if choices is not None:
            if isinstance(choices, (str, bytes)) or not isinstance(
                choices, (list, tuple)
            ):
                raise ContractValidationError("mutable rule choices must be an array")
            if not choices:
                raise ContractValidationError(
                    "mutable rule choices must not be empty when provided"
                )
            normalized_choices = tuple(
                _normalize_rule_value(
                    choice,
                    value_type,
                    context=f"mutable rule choices[{index}]",
                )
                for index, choice in enumerate(choices)
            )
            identities = [canonical_json_bytes(choice) for choice in normalized_choices]
            if len(set(identities)) != len(identities):
                raise ContractValidationError(
                    "mutable rule choices contains duplicate values"
                )
            for choice in normalized_choices:
                _validate_rule_bounds(
                    choice,
                    minimum=minimum,
                    maximum=maximum,
                    context="mutable rule choice",
                )

        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "choices", normalized_choices)

    def normalize_value(
        self,
        value: object,
        *,
        context: str = "mutable value",
    ) -> bool | int | float | str:
        """Validate and canonically normalize one candidate policy value."""

        normalized = _normalize_rule_value(value, self.value_type, context=context)
        _validate_rule_bounds(
            normalized,
            minimum=self.minimum,
            maximum=self.maximum,
            context=context,
        )
        if self.choices is not None and canonical_json_bytes(normalized) not in {
            canonical_json_bytes(choice) for choice in self.choices
        }:
            raise ContractValidationError(
                f"{context} is not one of the allowed choices"
            )
        return normalized

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "type": self.value_type.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices) if self.choices is not None else None,
        }

    @classmethod
    def from_dict(cls, data: object) -> "MutableFieldRule":
        item = _mapping(data, "mutable rule")
        fields = {"type", "minimum", "maximum", "choices"}
        _strict_fields(
            item,
            allowed=fields,
            required={"type"},
            context="mutable rule",
        )
        return cls(
            value_type=item["type"],
            minimum=item.get("minimum"),
            maximum=item.get("maximum"),
            choices=item.get("choices"),
        )


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Complete immutable bounds and evaluation policy for one search run."""

    name: str
    objectives: tuple[MetricObjective, ...]
    mutable_fields: tuple[str, ...]
    development_repetitions: int
    confirmation_repetitions: int
    holdout_repetitions: int
    max_candidates: int
    max_parallel_candidates: int
    max_trials: int
    max_wall_seconds: int
    max_budget_microusd: int
    per_trial_reservation_microusd: int
    confidence: float
    min_improvement: float
    base_seed: int
    required_gates: tuple[str, ...] = ()
    mutable_field_rules: Mapping[str, MutableFieldRule] = field(
        default_factory=dict
    )
    version: int = field(default=CONTRACT_VERSION, init=False)
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_name(self.name, "contract.name"))

        if not isinstance(self.objectives, (list, tuple)):
            raise ContractValidationError("contract.objectives must be an array")
        objectives = tuple(self.objectives)
        if not objectives or not all(
            isinstance(item, MetricObjective) for item in objectives
        ):
            raise ContractValidationError(
                "contract.objectives must contain MetricObjective values"
            )
        objective_names = [item.name for item in objectives]
        if len(set(objective_names)) != len(objective_names):
            raise ContractValidationError("contract.objectives contains duplicate names")
        if sum(item.primary for item in objectives) != 1:
            raise ContractValidationError(
                "contract.objectives must contain exactly one primary objective"
            )
        object.__setattr__(self, "objectives", objectives)

        if not isinstance(self.mutable_fields, (list, tuple)):
            raise ContractValidationError("contract.mutable_fields must be an array")
        mutable_fields = tuple(
            _dotted_name(item, "contract.mutable_fields")
            for item in self.mutable_fields
        )
        if not mutable_fields:
            raise ContractValidationError("contract.mutable_fields must not be empty")
        if len(set(mutable_fields)) != len(mutable_fields):
            raise ContractValidationError(
                "contract.mutable_fields contains duplicate paths"
            )
        object.__setattr__(self, "mutable_fields", mutable_fields)

        if not isinstance(self.required_gates, (list, tuple)):
            raise ContractValidationError("contract.required_gates must be an array")
        required_gates = tuple(
            normalize_gate_name(item, context="contract.required_gates")
            for item in self.required_gates
        )
        if len(set(required_gates)) != len(required_gates):
            raise ContractValidationError(
                "contract.required_gates contains duplicate normalized names"
            )
        object.__setattr__(self, "required_gates", required_gates)

        if not isinstance(self.mutable_field_rules, Mapping) or any(
            not isinstance(key, str) for key in self.mutable_field_rules
        ):
            raise ContractValidationError(
                "contract.mutable_field_rules must be an object"
            )
        mutable_field_rules: dict[str, MutableFieldRule] = {}
        for path, rule in self.mutable_field_rules.items():
            normalized_path = _dotted_name(path, "contract.mutable_field_rules path")
            if normalized_path not in mutable_fields:
                raise ContractValidationError(
                    f"contract.mutable_field_rules path {normalized_path!r} is not mutable"
                )
            if not isinstance(rule, MutableFieldRule):
                raise ContractValidationError(
                    "contract.mutable_field_rules values must be MutableFieldRule values"
                )
            mutable_field_rules[normalized_path] = rule
        object.__setattr__(
            self,
            "mutable_field_rules",
            MappingProxyType(mutable_field_rules),
        )

        development = _bounded_int(
            self.development_repetitions,
            "contract.development_repetitions",
            minimum=1,
        )
        confirmation = _bounded_int(
            self.confirmation_repetitions,
            "contract.confirmation_repetitions",
            minimum=3,
        )
        holdout = _bounded_int(
            self.holdout_repetitions,
            "contract.holdout_repetitions",
            minimum=3,
        )
        max_candidates = _bounded_int(
            self.max_candidates,
            "contract.max_candidates",
            minimum=1,
        )
        max_parallel = _bounded_int(
            self.max_parallel_candidates,
            "contract.max_parallel_candidates",
            minimum=1,
        )
        max_trials = _bounded_int(
            self.max_trials,
            "contract.max_trials",
            minimum=1,
        )
        max_wall = _bounded_int(
            self.max_wall_seconds,
            "contract.max_wall_seconds",
            minimum=1,
        )
        max_budget = _bounded_int(
            self.max_budget_microusd,
            "contract.max_budget_microusd",
            minimum=0,
        )
        per_trial = _bounded_int(
            self.per_trial_reservation_microusd,
            "contract.per_trial_reservation_microusd",
            minimum=0,
        )
        base_seed = _bounded_int(
            self.base_seed,
            "contract.base_seed",
            minimum=0,
        )
        if max_parallel > max_candidates:
            raise ContractValidationError(
                "contract.max_parallel_candidates must not exceed max_candidates"
            )
        minimum_trials = development + confirmation + holdout
        if max_trials < minimum_trials:
            raise ContractValidationError(
                "contract.max_trials cannot complete one development, "
                "confirmation, and holdout lifecycle"
            )
        minimum_reservation = minimum_trials * per_trial
        if minimum_reservation > max_budget:
            raise ContractValidationError(
                "contract.max_budget_microusd cannot reserve one complete "
                "candidate lifecycle"
            )

        confidence = _metric(self.confidence, "contract.confidence")
        if not 0.5 < confidence < 1.0:
            raise ContractValidationError(
                "contract.confidence must be greater than 0.5 and less than 1"
            )
        min_improvement = _metric(
            self.min_improvement,
            "contract.min_improvement",
        )
        if min_improvement < 0:
            raise ContractValidationError(
                "contract.min_improvement must be non-negative"
            )

        for field_name, value in (
            ("development_repetitions", development),
            ("confirmation_repetitions", confirmation),
            ("holdout_repetitions", holdout),
            ("max_candidates", max_candidates),
            ("max_parallel_candidates", max_parallel),
            ("max_trials", max_trials),
            ("max_wall_seconds", max_wall),
            ("max_budget_microusd", max_budget),
            ("per_trial_reservation_microusd", per_trial),
            ("base_seed", base_seed),
            ("confidence", confidence),
            ("min_improvement", min_improvement),
        ):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "contract_hash", sha256_prefixed(self.to_dict()))

    @property
    def primary_objective(self) -> MetricObjective:
        return next(item for item in self.objectives if item.primary)

    @property
    def minimum_full_evaluation_trials(self) -> int:
        return (
            self.development_repetitions
            + self.confirmation_repetitions
            + self.holdout_repetitions
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "name": self.name,
            "objectives": [item.to_dict() for item in self.objectives],
            "mutable_fields": list(self.mutable_fields),
            "required_gates": list(self.required_gates),
            "mutable_field_rules": {
                path: self.mutable_field_rules[path].to_dict()
                for path in self.mutable_fields
                if path in self.mutable_field_rules
            },
            "development_repetitions": self.development_repetitions,
            "confirmation_repetitions": self.confirmation_repetitions,
            "holdout_repetitions": self.holdout_repetitions,
            "max_candidates": self.max_candidates,
            "max_parallel_candidates": self.max_parallel_candidates,
            "max_trials": self.max_trials,
            "max_wall_seconds": self.max_wall_seconds,
            "max_budget_microusd": self.max_budget_microusd,
            "per_trial_reservation_microusd": (
                self.per_trial_reservation_microusd
            ),
            "confidence": self.confidence,
            "min_improvement": self.min_improvement,
            "base_seed": self.base_seed,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ExperimentContract":
        item = _mapping(data, "contract")
        fields = {
            "version",
            "name",
            "objectives",
            "mutable_fields",
            "development_repetitions",
            "confirmation_repetitions",
            "holdout_repetitions",
            "max_candidates",
            "max_parallel_candidates",
            "max_trials",
            "max_wall_seconds",
            "max_budget_microusd",
            "per_trial_reservation_microusd",
            "confidence",
            "min_improvement",
            "base_seed",
            "required_gates",
            "mutable_field_rules",
        }
        legacy_required = fields - {"required_gates", "mutable_field_rules"}
        _strict_fields(
            item,
            allowed=fields,
            required=legacy_required,
            context="contract",
        )
        if item["version"] != CONTRACT_VERSION:
            raise ContractValidationError(
                f"unsupported contract version: {item['version']!r}"
            )
        raw_objectives = item["objectives"]
        if not isinstance(raw_objectives, list):
            raise ContractValidationError("contract.objectives must be an array")
        raw_rules = item.get("mutable_field_rules", {})
        if not isinstance(raw_rules, Mapping) or any(
            not isinstance(key, str) for key in raw_rules
        ):
            raise ContractValidationError(
                "contract.mutable_field_rules must be an object"
            )
        return cls(
            name=item["name"],
            objectives=tuple(MetricObjective.from_dict(obj) for obj in raw_objectives),
            mutable_fields=item["mutable_fields"],
            development_repetitions=item["development_repetitions"],
            confirmation_repetitions=item["confirmation_repetitions"],
            holdout_repetitions=item["holdout_repetitions"],
            max_candidates=item["max_candidates"],
            max_parallel_candidates=item["max_parallel_candidates"],
            max_trials=item["max_trials"],
            max_wall_seconds=item["max_wall_seconds"],
            max_budget_microusd=item["max_budget_microusd"],
            per_trial_reservation_microusd=item[
                "per_trial_reservation_microusd"
            ],
            confidence=item["confidence"],
            min_improvement=item["min_improvement"],
            base_seed=item["base_seed"],
            required_gates=item.get("required_gates", ()),
            mutable_field_rules={
                path: MutableFieldRule.from_dict(rule)
                for path, rule in raw_rules.items()
            },
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """An immutable allowlisted policy patch bound to one contract lineage."""

    contract: ExperimentContract = field(repr=False, compare=False)
    policy: Mapping[str, object]
    hypothesis: str
    parent: str | Candidate | None = None
    version: int = field(default=CONTRACT_VERSION, init=False)
    contract_hash: str = field(init=False)
    policy_hash: str = field(init=False)
    candidate_hash: str = field(init=False)
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ExperimentContract):
            raise ContractValidationError(
                "candidate.contract must be an ExperimentContract"
            )
        if not isinstance(self.policy, Mapping):
            raise ContractValidationError("candidate.policy must be an object")
        policy = _json_copy(self.policy, context="candidate.policy")
        if not isinstance(policy, dict) or not policy:
            raise ContractValidationError("candidate.policy must not be empty")
        allowed = set(self.contract.mutable_fields)
        for path in policy:
            _dotted_name(path, "candidate.policy path")
            if path not in allowed:
                raise ContractValidationError(
                    f"candidate.policy path {path!r} is not mutable"
                )
            rule = self.contract.mutable_field_rules.get(path)
            if rule is not None:
                policy[path] = rule.normalize_value(
                    policy[path],
                    context=f"candidate.policy[{path!r}]",
                )

        if not isinstance(self.hypothesis, str):
            raise ContractValidationError("candidate.hypothesis must be a string")
        hypothesis = self.hypothesis.strip()
        if not hypothesis or len(hypothesis) > 2048:
            raise ContractValidationError(
                "candidate.hypothesis must contain 1 to 2048 characters"
            )
        if any(ord(char) < 32 and char not in "\n\t" for char in hypothesis):
            raise ContractValidationError(
                "candidate.hypothesis contains unsafe control characters"
            )

        parent = self.parent
        if isinstance(parent, Candidate):
            if not hmac.compare_digest(
                parent.contract_hash,
                self.contract.contract_hash,
            ):
                raise ContractValidationError(
                    "candidate.parent belongs to a different contract"
                )
            parent = parent.candidate_id
        elif parent is not None and (
            not isinstance(parent, str) or not _CANDIDATE_ID_RE.fullmatch(parent)
        ):
            raise ContractValidationError(
                "candidate.parent must be null or a cand_<sha256> identifier"
            )

        policy_hash = sha256_prefixed(policy)
        identity = {
            "version": self.version,
            "contract_hash": self.contract.contract_hash,
            "policy_hash": policy_hash,
            "hypothesis": hypothesis,
            "parent": parent,
        }
        candidate_hash = sha256_prefixed(identity)
        candidate_id = "cand_" + candidate_hash.removeprefix("sha256:")

        object.__setattr__(self, "policy", _freeze_json(policy))
        object.__setattr__(self, "hypothesis", hypothesis)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "contract_hash", self.contract.contract_hash)
        object.__setattr__(self, "policy_hash", policy_hash)
        object.__setattr__(self, "candidate_hash", candidate_hash)
        object.__setattr__(self, "candidate_id", candidate_id)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "contract_hash": self.contract_hash,
            "policy": _json_copy(self.policy, context="candidate.policy"),
            "hypothesis": self.hypothesis,
            "parent": self.parent,
            "policy_hash": self.policy_hash,
            "candidate_hash": self.candidate_hash,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        contract: ExperimentContract,
    ) -> "Candidate":
        item = _mapping(data, "candidate")
        fields = {
            "version",
            "contract_hash",
            "policy",
            "hypothesis",
            "parent",
            "policy_hash",
            "candidate_hash",
            "candidate_id",
        }
        _strict_fields(item, allowed=fields, required=fields, context="candidate")
        if item["version"] != CONTRACT_VERSION:
            raise ContractValidationError(
                f"unsupported candidate version: {item['version']!r}"
            )
        if not isinstance(item["contract_hash"], str) or not _HASH_RE.fullmatch(
            item["contract_hash"]
        ):
            raise ContractValidationError("candidate.contract_hash is invalid")
        if not hmac.compare_digest(item["contract_hash"], contract.contract_hash):
            raise ContractValidationError("candidate.contract_hash does not match")

        candidate = cls(
            contract=contract,
            policy=item["policy"],
            hypothesis=item["hypothesis"],
            parent=item["parent"],
        )
        for field_name in ("policy_hash", "candidate_hash", "candidate_id"):
            supplied = item[field_name]
            expected = getattr(candidate, field_name)
            if not isinstance(supplied, str) or not hmac.compare_digest(
                supplied,
                expected,
            ):
                raise ContractValidationError(
                    f"candidate.{field_name} does not match its contents"
                )
        return candidate


def _safe_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ContractValidationError(
            f"{context} must match {_NAME_RE.pattern!r}"
        )
    return value


def _dotted_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not _DOTTED_FIELD_RE.fullmatch(value):
        raise ContractValidationError(
            f"{context} must be a safe dotted field path"
        )
    return value


def _bounded_int(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{context} must be an integer")
    if value < minimum or value > _MAX_INT:
        raise ContractValidationError(
            f"{context} must be between {minimum} and {_MAX_INT}"
        )
    return value


def _optional_rule_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _bounded_int(value, context, minimum=-_MAX_INT)


def _metric(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{context} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ContractValidationError(f"{context} is out of range") from exc
    if not math.isfinite(number):
        raise ContractValidationError(f"{context} must be finite")
    return 0.0 if number == 0 else number


def _optional_metric(value: object, context: str) -> float | None:
    return None if value is None else _metric(value, context)


def _normalize_rule_value(
    value: object,
    value_type: MutableValueType,
    *,
    context: str,
) -> bool | int | float | str:
    if value_type is MutableValueType.BOOLEAN:
        if not isinstance(value, bool):
            raise ContractValidationError(f"{context} must be a boolean")
        return value
    if value_type is MutableValueType.STRING:
        if not isinstance(value, str):
            raise ContractValidationError(f"{context} must be a string")
        copied = _json_copy(value, context=context)
        assert isinstance(copied, str)
        return copied
    if value_type is MutableValueType.INTEGER:
        return _bounded_int(value, context, minimum=-_MAX_INT)
    return _metric(value, context)


def _validate_rule_bounds(
    value: bool | int | float | str,
    *,
    minimum: int | float | None,
    maximum: int | float | None,
    context: str,
) -> None:
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{context} is below the allowed minimum")
    if maximum is not None and value > maximum:
        raise ContractValidationError(f"{context} exceeds the allowed maximum")


def normalize_gate_name(value: object, *, context: str = "gate") -> str:
    """Return the canonical printable name used for a required result gate."""

    if not isinstance(value, str):
        raise ContractValidationError(f"{context} name must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in normalized)
    ):
        raise ContractValidationError(
            f"{context} names must be non-empty printable strings <= 128 chars"
        )
    return normalized


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ContractValidationError(f"{context} must be an object")
    return dict(value)


def _strict_fields(
    data: Mapping[str, object],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise ContractValidationError(
            f"{context} has unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise ContractValidationError(
            f"{context} is missing required fields: {sorted(missing)}"
        )


def _json_copy(value: object, *, context: str, depth: int = 0) -> JSONValue:
    if depth > _MAX_DEPTH:
        raise ContractValidationError(f"{context} is nested too deeply")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ContractValidationError(
                f"{context} contains invalid Unicode"
            ) from exc
        return value
    if isinstance(value, int):
        if not -_MAX_INT <= value <= _MAX_INT:
            raise ContractValidationError(f"{context} integer is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{context} contains a non-finite number")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractValidationError(f"{context} contains a non-string key")
        return {
            key: _json_copy(
                item,
                context=f"{context}.{key}",
                depth=depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_copy(
                item,
                context=f"{context}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    raise ContractValidationError(
        f"{context} contains unsupported value {type(value).__name__}"
    )


def _freeze_json(value: JSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


__all__ = [
    "CONTRACT_VERSION",
    "Candidate",
    "ContractValidationError",
    "ExperimentContract",
    "MetricObjective",
    "MutableFieldRule",
    "MutableValueType",
    "ObjectiveDirection",
    "canonical_json_bytes",
    "normalize_gate_name",
    "sha256_prefixed",
]
