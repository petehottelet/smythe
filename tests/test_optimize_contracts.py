"""Contract and identity tests for bounded Autotune experiments."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from smythe.optimize import (
    Candidate,
    ContractValidationError,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
    MutableValueType,
    ObjectiveDirection,
    canonical_json_bytes,
    sha256_prefixed,
)


def _objectives():
    return (
        MetricObjective(
            "quality.overall",
            ObjectiveDirection.MAXIMIZE,
            primary=True,
            hard_min=7,
        ),
        MetricObjective(
            "cost.microusd",
            ObjectiveDirection.MINIMIZE,
            hard_max=100_000,
            max_regression=5_000,
        ),
    )


def _contract(**overrides):
    values = {
        "name": "asset_autotune",
        "objectives": _objectives(),
        "mutable_fields": ("prompt.style", "runtime.concurrency", "flags.curate"),
        "development_repetitions": 1,
        "confirmation_repetitions": 3,
        "holdout_repetitions": 3,
        "max_candidates": 8,
        "max_parallel_candidates": 3,
        "max_trials": 40,
        "max_wall_seconds": 3_600,
        "max_budget_microusd": 1_000_000,
        "per_trial_reservation_microusd": 100_000,
        "confidence": 0.95,
        "min_improvement": 0.25,
        "base_seed": 42,
        "required_gates": ("safe", "licensed"),
    }
    values.update(overrides)
    if "mutable_field_rules" not in overrides:
        default_rules = {
            "prompt.style": MutableFieldRule(
                "string",
                choices=("precise", "minimal", "a", "b"),
            ),
            "runtime.concurrency": MutableFieldRule("integer", minimum=1, maximum=64),
            "flags.curate": MutableFieldRule("boolean"),
        }
        values["mutable_field_rules"] = {
            path: default_rules[path]
            for path in values["mutable_fields"]
            if path in default_rules
        }
    return ExperimentContract(**values)


def test_canonical_json_is_sorted_compact_utf8_and_hashable():
    payload = {"z": "café", "a": [True, 0.0, {"b": 2}]}
    encoded = canonical_json_bytes(payload)

    assert encoded == b'{"a":[true,0.0,{"b":2}],"z":"caf\xc3\xa9"}'
    assert sha256_prefixed(payload) == "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string-key"},
        {"value": object()},
        {"value": 1 << 80},
    ],
)
def test_canonical_json_rejects_non_json_or_nonportable_values(payload):
    with pytest.raises(ContractValidationError):
        canonical_json_bytes(payload)


def test_metric_objective_normalizes_direction_and_numbers():
    objective = MetricObjective(
        "latency.p95",
        "minimize",
        hard_min=0,
        hard_max=500,
        max_regression=25,
    )

    assert objective.direction is ObjectiveDirection.MINIMIZE
    assert objective.hard_min == 0.0
    assert objective.to_dict()["direction"] == "minimize"
    assert MetricObjective.from_dict(objective.to_dict()) == objective


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hard_min", True),
        ("hard_max", float("nan")),
        ("max_regression", float("inf")),
        ("max_regression", -0.01),
    ],
)
def test_metric_objective_rejects_boolean_nonfinite_or_negative_numbers(
    field,
    value,
):
    kwargs = {field: value}
    with pytest.raises(ContractValidationError):
        MetricObjective("quality", "maximize", **kwargs)


def test_metric_objective_rejects_inverted_bounds_and_unsafe_names():
    with pytest.raises(ContractValidationError, match="hard_min"):
        MetricObjective("quality", "maximize", hard_min=10, hard_max=5)
    with pytest.raises(ContractValidationError, match="safe dotted"):
        MetricObjective("../quality", "maximize")
    with pytest.raises(ContractValidationError, match="unsupported"):
        MetricObjective("quality", "sideways")
    with pytest.raises(ContractValidationError, match="boolean"):
        MetricObjective("quality", "maximize", primary=1)


def test_contract_round_trip_is_frozen_and_hash_bound():
    contract = _contract()
    restored = ExperimentContract.from_dict(contract.to_dict())

    assert restored == contract
    assert restored.contract_hash == sha256_prefixed(restored.to_dict())
    assert restored.primary_objective.name == "quality.overall"
    assert restored.minimum_full_evaluation_trials == 7
    assert isinstance(restored.objectives, tuple)
    assert isinstance(restored.mutable_fields, tuple)
    assert isinstance(restored.required_gates, tuple)
    assert restored.required_gates == ("safe", "licensed")
    assert restored.mutable_field_rules["runtime.concurrency"].maximum == 64
    with pytest.raises(TypeError):
        restored.mutable_field_rules["runtime.concurrency"] = MutableFieldRule("integer")
    with pytest.raises(FrozenInstanceError):
        restored.max_trials = 99


@pytest.mark.parametrize(
    "objectives",
    [
        (MetricObjective("quality", "maximize"),),
        (
            MetricObjective("quality", "maximize", primary=True),
            MetricObjective("cost", "minimize", primary=True),
        ),
    ],
)
def test_contract_requires_exactly_one_primary_objective(objectives):
    with pytest.raises(ContractValidationError, match="exactly one"):
        _contract(objectives=objectives)


def test_contract_rejects_duplicate_objectives_and_mutable_fields():
    duplicate_objectives = (
        MetricObjective("quality", "maximize", primary=True),
        MetricObjective("quality", "minimize"),
    )
    with pytest.raises(ContractValidationError, match="duplicate names"):
        _contract(objectives=duplicate_objectives)
    with pytest.raises(ContractValidationError, match="duplicate paths"):
        _contract(mutable_fields=("prompt.style", "prompt.style"))


def test_required_gate_inventory_is_normalized_unique_and_hash_bound():
    contract = _contract(required_gates=(" safe ", "licensed"))
    assert contract.required_gates == ("safe", "licensed")
    assert ExperimentContract.from_dict(contract.to_dict()) == contract
    assert _contract(required_gates=("safe",)).contract_hash != contract.contract_hash

    with pytest.raises(ContractValidationError, match="duplicate normalized"):
        _contract(required_gates=("safe", " safe "))
    with pytest.raises(ContractValidationError, match="printable"):
        _contract(required_gates=("bad\x00gate",))


def test_new_contract_fields_are_optional_when_reading_a_legacy_wire_payload():
    payload = _contract().to_dict()
    payload.pop("required_gates")
    payload.pop("mutable_field_rules")

    restored = ExperimentContract.from_dict(payload)

    assert restored.required_gates == ()
    assert dict(restored.mutable_field_rules) == {}


def test_mutable_field_rule_round_trip_and_value_normalization():
    integer = MutableFieldRule(
        MutableValueType.INTEGER,
        minimum=1,
        maximum=8,
        choices=(1, 2, 4, 8),
    )
    number = MutableFieldRule(
        "number",
        minimum=0,
        maximum=1,
        choices=(0.25, 0.5, 1.0),
    )

    assert MutableFieldRule.from_dict(integer.to_dict()) == integer
    assert number.normalize_value(0.5) == 0.5
    assert number.normalize_value(1) == 1.0
    assert number.value_type is MutableValueType.NUMBER


@pytest.mark.parametrize(
    "rule",
    [
        MutableFieldRule("integer", minimum=1),
        MutableFieldRule("number", maximum=2.5),
        MutableFieldRule("string", choices=("a", "b")),
        MutableFieldRule("boolean", choices=(True, False)),
    ],
)
def test_mutable_field_rule_supports_all_declared_scalar_types(rule):
    assert MutableFieldRule.from_dict(rule.to_dict()) == rule


def test_mutable_field_rules_reject_invalid_domains_and_unlisted_paths():
    with pytest.raises(ContractValidationError, match="unsupported"):
        MutableFieldRule("object")
    with pytest.raises(ContractValidationError, match="require an integer or number"):
        MutableFieldRule("string", minimum=1)
    with pytest.raises(ContractValidationError, match="must not exceed"):
        MutableFieldRule("number", minimum=2, maximum=1)
    with pytest.raises(ContractValidationError, match="duplicate"):
        MutableFieldRule("integer", choices=(1, 1))
    with pytest.raises(ContractValidationError, match="below"):
        MutableFieldRule("integer", minimum=2, choices=(1,))
    with pytest.raises(ContractValidationError, match="not mutable"):
        _contract(mutable_field_rules={"other": MutableFieldRule("integer")})


def test_contract_numeric_conversion_rejects_huge_integers_cleanly():
    with pytest.raises(ContractValidationError, match="out of range"):
        MetricObjective("quality", "maximize", hard_max=10**1000)
    with pytest.raises(ContractValidationError):
        MutableFieldRule("number", maximum=10**1000)


@pytest.mark.parametrize(
    "mutable_fields",
    [(), ("../prompt",), ("prompt..style",), ("prompt/style",), ("_private",)],
)
def test_contract_rejects_empty_or_unsafe_mutable_paths(mutable_fields):
    with pytest.raises(ContractValidationError):
        _contract(mutable_fields=mutable_fields)


@pytest.mark.parametrize("field", ["confirmation_repetitions", "holdout_repetitions"])
def test_contract_requires_at_least_three_confirmation_and_holdout_runs(field):
    with pytest.raises(ContractValidationError):
        _contract(**{field: 2})


@pytest.mark.parametrize(
    "field",
    [
        "development_repetitions",
        "max_candidates",
        "max_parallel_candidates",
        "max_trials",
        "max_wall_seconds",
        "max_budget_microusd",
        "per_trial_reservation_microusd",
        "base_seed",
    ],
)
def test_contract_rejects_booleans_as_integer_caps(field):
    with pytest.raises(ContractValidationError, match="integer"):
        _contract(**{field: True})


def test_contract_rejects_incoherent_parallel_trial_and_budget_caps():
    with pytest.raises(ContractValidationError, match="max_parallel"):
        _contract(max_candidates=2, max_parallel_candidates=3)
    with pytest.raises(ContractValidationError, match="max_trials"):
        _contract(max_trials=6)
    with pytest.raises(ContractValidationError, match="cannot reserve"):
        _contract(max_budget_microusd=699_999)


def test_contract_allows_a_zero_cost_offline_experiment():
    contract = _contract(
        max_budget_microusd=0,
        per_trial_reservation_microusd=0,
    )

    assert contract.max_budget_microusd == 0
    assert contract.per_trial_reservation_microusd == 0


@pytest.mark.parametrize("confidence", [True, 0.5, 1.0, float("nan")])
def test_contract_rejects_invalid_confidence(confidence):
    with pytest.raises(ContractValidationError):
        _contract(confidence=confidence)


@pytest.mark.parametrize("minimum", [True, -0.01, float("inf")])
def test_contract_rejects_invalid_minimum_improvement(minimum):
    with pytest.raises(ContractValidationError):
        _contract(min_improvement=minimum)


def test_contract_wire_format_is_strict_and_versioned():
    payload = _contract().to_dict()
    payload["surprise"] = True
    with pytest.raises(ContractValidationError, match="unknown fields"):
        ExperimentContract.from_dict(payload)

    payload = _contract().to_dict()
    payload["version"] = 2
    with pytest.raises(ContractValidationError, match="version"):
        ExperimentContract.from_dict(payload)


def test_contract_rejects_unsafe_name():
    with pytest.raises(ContractValidationError, match="contract.name"):
        _contract(name="../../experiment")


def test_candidate_identity_is_deterministic_across_mapping_order():
    contract = _contract()
    first = Candidate(
        contract,
        {"prompt.style": "precise", "runtime.concurrency": 4},
        "Improve quality without increasing cost.",
    )
    second = Candidate(
        contract,
        {"runtime.concurrency": 4, "prompt.style": "precise"},
        "Improve quality without increasing cost.",
    )

    assert first.policy_hash == second.policy_hash
    assert first.candidate_hash == second.candidate_hash
    assert first.candidate_id == second.candidate_id
    assert first.candidate_id.startswith("cand_")
    assert len(first.candidate_id) == 69


def test_candidate_policy_hypothesis_and_parent_are_identity_bound():
    contract = _contract()
    root = Candidate(contract, {"prompt.style": "precise"}, "Improve clarity")
    changed_policy = Candidate(contract, {"prompt.style": "minimal"}, "Improve clarity")
    changed_hypothesis = Candidate(
        contract,
        {"prompt.style": "precise"},
        "Improve consistency",
    )
    child = Candidate(
        contract,
        {"prompt.style": "precise"},
        "Improve clarity",
        parent=root,
    )

    assert changed_policy.policy_hash != root.policy_hash
    assert changed_hypothesis.policy_hash == root.policy_hash
    assert changed_hypothesis.candidate_id != root.candidate_id
    assert child.parent == root.candidate_id
    assert child.candidate_id != root.candidate_id


def test_candidate_policy_is_allowlisted_json_and_deeply_immutable():
    source = {
        "prompt.style": {"tone": "precise", "examples": ["a", "b"]},
        "flags.curate": True,
    }
    candidate = Candidate(
        _contract(mutable_field_rules={}),
        source,
        "Use a precise tone",
    )
    source["prompt.style"]["tone"] = "mutated"

    assert candidate.policy["prompt.style"]["tone"] == "precise"
    assert candidate.policy["prompt.style"]["examples"] == ("a", "b")
    with pytest.raises(TypeError):
        candidate.policy["prompt.style"] = "changed"
    with pytest.raises(TypeError):
        candidate.policy["prompt.style"]["tone"] = "changed"


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"unknown.field": 1},
        {"../prompt": 1},
        {"prompt.style": float("nan")},
        {"prompt.style": object()},
    ],
)
def test_candidate_rejects_empty_unlisted_unsafe_or_non_json_policy(policy):
    with pytest.raises(ContractValidationError):
        Candidate(_contract(), policy, "Try one change")


@pytest.mark.parametrize(
    "policy",
    [
        {"runtime.concurrency": True},
        {"runtime.concurrency": 0},
        {"runtime.concurrency": 65},
        {"prompt.style": "unlisted"},
        {"flags.curate": 1},
    ],
)
def test_candidate_validates_rule_bound_values_before_identity(policy):
    with pytest.raises(ContractValidationError):
        Candidate(_contract(), policy, "Try a bounded change")


def test_candidate_parent_must_share_the_contract():
    parent = Candidate(_contract(name="first"), {"prompt.style": "a"}, "First")
    with pytest.raises(ContractValidationError, match="different contract"):
        Candidate(
            _contract(name="second"),
            {"prompt.style": "b"},
            "Second",
            parent=parent,
        )
    with pytest.raises(ContractValidationError, match="candidate.parent"):
        Candidate(_contract(), {"prompt.style": "a"}, "Bad parent", parent="../x")


def test_candidate_wire_round_trip_verifies_all_hashes():
    contract = _contract()
    candidate = Candidate(contract, {"prompt.style": "precise"}, "Improve clarity")

    assert Candidate.from_dict(candidate.to_dict(), contract=contract) == candidate
    for field in ("policy_hash", "candidate_hash", "candidate_id"):
        tampered = candidate.to_dict()
        tampered[field] = "sha256:" + "0" * 64 if field != "candidate_id" else "cand_" + "0" * 64
        with pytest.raises(ContractValidationError, match=field):
            Candidate.from_dict(tampered, contract=contract)


def test_candidate_wire_rejects_contract_mismatch_and_unknown_fields():
    candidate = Candidate(
        _contract(name="first"),
        {"prompt.style": "precise"},
        "Improve clarity",
    )
    with pytest.raises(ContractValidationError, match="contract_hash"):
        Candidate.from_dict(candidate.to_dict(), contract=_contract(name="second"))

    payload = candidate.to_dict()
    payload["surprise"] = True
    with pytest.raises(ContractValidationError, match="unknown fields"):
        Candidate.from_dict(payload, contract=candidate.contract)


def test_package_exports_the_programmatic_campaign_surface():
    import smythe.optimize as optimize

    assert optimize.ExperimentContract is ExperimentContract
    assert optimize.Candidate is Candidate
    assert callable(optimize.OptimizationRunner)
    assert callable(optimize.ExperimentLedger)
    assert callable(optimize.simulate_concurrency)
