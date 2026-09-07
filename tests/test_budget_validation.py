"""Strict numeric boundaries must reject bad accounting atomically."""

import math

import pytest

from smythe.budget import (
    BudgetReconciliationError,
    BudgetValidationError,
    Sentinel,
    SentinelAlert,
    validate_completion_usage,
    validate_token_count,
)
from smythe.provider import CompletionResult


INVALID_USD = [
    pytest.param(-0.01, id="negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("0.01", id="numeric-string"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(10**400, id="integer-overflow"),
]
INVALID_TOKENS = [
    pytest.param(-1, id="negative"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("3", id="numeric-string"),
    pytest.param(3.0, id="integer-float"),
    pytest.param(0.5, id="fraction"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(None, id="missing"),
]


def _state(budget):
    """Include reservations and quality flags, not only the visible total."""
    return (
        budget.total_cost_usd,
        budget.breakdown(),
        dict(budget._reservations),
        set(budget._hard_reservations),
        set(budget._unknown_cost_nodes),
        set(budget._estimated_cost_nodes),
    )


def _populated_budget():
    budget = Sentinel(max_budget_usd=10, cost_per_token=0.01)
    budget.record("existing", CompletionResult(text="ok", cost_usd=0.25))
    budget.record(
        "uncertain",
        CompletionResult(
            text="partial", cost_usd=0.1, cost_usd_unknown=True,
            cost_usd_is_estimate=True,
        ),
    )
    budget.reserve("existing", 0.5, hard_ceiling=True)
    budget.reserve("other", 0.75)
    return budget


@pytest.mark.parametrize("field", ["max_budget_usd", "cost_per_token"])
@pytest.mark.parametrize("value", INVALID_USD)
def test_invalid_configuration_is_rejected_at_construction(field, value):
    with pytest.raises(BudgetValidationError, match=field):
        Sentinel(**{field: value})


@pytest.mark.parametrize("field", ["max_budget_usd", "cost_per_token"])
@pytest.mark.parametrize("value", INVALID_USD)
def test_invalid_configuration_assignment_preserves_previous_value(field, value):
    budget = _populated_budget()
    previous = getattr(budget, field)
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match=field):
        setattr(budget, field, value)
    assert getattr(budget, field) == previous
    assert _state(budget) == before


def test_none_is_only_valid_for_unlimited_budget_and_unspecified_actual_cost():
    budget = Sentinel(max_budget_usd=None, cost_per_token=0)
    assert budget.record("free", CompletionResult(text="ok")) == 0
    with pytest.raises(BudgetValidationError, match="cost_per_token"):
        Sentinel(cost_per_token=None)
    with pytest.raises(BudgetValidationError, match="estimated_cost"):
        budget.reserve("invalid", None)


@pytest.mark.parametrize("value", INVALID_USD)
def test_invalid_reservation_preserves_all_accounting(value):
    budget = _populated_budget()
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="estimated_cost"):
        budget.reserve("new", value, hard_ceiling=True)
    assert _state(budget) == before


@pytest.mark.parametrize("method", ["record", "add_cost"])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("value", INVALID_USD)
def test_invalid_actual_usd_retains_reservation_prior_cost_and_flags(method, unknown, value):
    budget = _populated_budget()
    before = _state(budget)
    result = CompletionResult(
        text="ok", cost_usd=0.1, cost_usd_unknown=unknown, cost_usd_is_estimate=True,
    )
    # Revalidate after construction: providers can mutate returned objects.
    result.cost_usd = value
    with pytest.raises(BudgetValidationError, match="cost_usd"):
        getattr(budget, method)("existing", result)
    assert _state(budget) == before
    assert math.isfinite(budget.total_cost_usd)


@pytest.mark.parametrize("method", ["record", "add_cost"])
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("value", INVALID_TOKENS)
def test_invalid_tokens_are_rejected_even_when_explicit_usd_wins(method, field, value):
    budget = _populated_budget()
    before = _state(budget)
    result = CompletionResult(text="ok", cost_usd=0.1)
    setattr(result, field, value)
    with pytest.raises(BudgetValidationError, match=field):
        getattr(budget, method)("existing", result)
    assert _state(budget) == before


@pytest.mark.parametrize("unknown", [False, True])
def test_invalid_tokens_without_explicit_cost_cannot_bypass_validation(unknown):
    budget = _populated_budget()
    before = _state(budget)
    result = CompletionResult(text="ok", cost_usd_unknown=unknown)
    result.completion_tokens = -1
    with pytest.raises(BudgetValidationError, match="completion_tokens"):
        budget.record("existing", result)
    assert _state(budget) == before


@pytest.mark.parametrize("value", INVALID_TOKENS)
def test_token_estimate_helper_preserves_integer_type_boundary(value):
    with pytest.raises(BudgetValidationError, match="estimated_tokens_per_node"):
        validate_token_count(value, "estimated_tokens_per_node")


def test_valid_zero_integer_and_explicit_usage_remain_supported():
    assert validate_token_count(0, "tokens") == 0
    assert validate_token_count(10**400, "tokens") == 10**400
    result = CompletionResult(text="ok", prompt_tokens=2, completion_tokens=3, cost_usd=1)
    assert validate_completion_usage(result) == (5, 1.0)
    budget = Sentinel(max_budget_usd=0, cost_per_token=0)
    budget.reserve("free", 0, hard_ceiling=True)
    assert budget.record("free", CompletionResult(text="free", prompt_tokens=10**400)) == 0
    budget.max_budget_usd = None
    budget.cost_per_token = 1
    assert budget.record("priced", result) == 1.0


@pytest.mark.parametrize("value", INVALID_USD + [pytest.param(None, id="missing")])
def test_restore_rejects_entire_batch_before_replacing_existing_cost(value):
    budget = _populated_budget()
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="Checkpoint cost"):
        budget.restore(
            {"existing": 0.4, "valid-new": 0.2, "bad": value},
            unknown_cost_nodes={"valid-new"}, estimated_cost_nodes={"valid-new"},
        )
    assert _state(budget) == before


def test_valid_restore_merges_costs_and_flags_with_pending_reservations():
    budget = _populated_budget()
    budget.restore(
        {"existing": 0.4, "new": 0.2},
        unknown_cost_nodes={"new"}, estimated_cost_nodes={"new"},
    )
    assert budget.breakdown() == {"existing": 0.4, "uncertain": 0.1, "new": 0.2}
    assert budget._reservations == {"existing": 0.5, "other": 0.75}
    assert budget._hard_reservations == {"existing"}
    assert budget._unknown_cost_nodes == {"uncertain", "new"}
    assert budget._estimated_cost_nodes == {"uncertain", "new"}
    assert budget.total_cost_usd == pytest.approx(1.95)


@pytest.mark.parametrize("value", [None, [], [("node", 0.1)], "{}", 0, False])
def test_restore_rejects_non_mapping_costs_atomically(value):
    budget = _populated_budget()
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="Checkpoint costs must be a mapping"):
        budget.restore(value)
    assert _state(budget) == before


def test_restore_retains_valid_incurred_cost_above_budget():
    budget = Sentinel(max_budget_usd=1)
    budget.restore({"already-spent": 2})
    assert budget.breakdown() == {"already-spent": 2}
    with pytest.raises(SentinelAlert):
        budget.check("next")


def test_reservation_aggregate_overflow_is_atomic():
    budget = Sentinel()
    budget.reserve("first", 1e308, hard_ceiling=True)
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="Total cost"):
        budget.reserve("second", 1e308, hard_ceiling=True)
    assert _state(budget) == before


@pytest.mark.parametrize("method", ["record", "add_cost"])
@pytest.mark.parametrize("tokens", [2, 10**400], ids=["float-overflow", "int-overflow"])
def test_token_pricing_overflow_retains_reservation(method, tokens):
    budget = Sentinel(cost_per_token=1e308)
    budget.reserve("pending", 0.5, hard_ceiling=True)
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="Token-derived cost"):
        getattr(budget, method)("pending", CompletionResult(text="ok", prompt_tokens=tokens))
    assert _state(budget) == before


@pytest.mark.parametrize("operation", ["add", "record", "restore"])
def test_reconciliation_or_restore_aggregate_overflow_is_atomic(operation):
    budget = Sentinel()
    budget.record("existing", CompletionResult(text="ok", cost_usd=1e308))
    budget.reserve("pending", 0.5, hard_ceiling=True)
    before = _state(budget)
    with pytest.raises(BudgetValidationError, match="Total cost"):
        if operation == "add":
            budget.add_cost("existing", CompletionResult(text="ok", cost_usd=1e308))
        elif operation == "record":
            budget.record("pending", CompletionResult(text="ok", cost_usd=1e308))
        else:
            budget.restore({"pending": 1e308})
    assert _state(budget) == before


@pytest.mark.parametrize("settlement", ["release", "record"])
def test_settling_large_reservation_preserves_small_recorded_cost(settlement):
    budget = Sentinel()
    budget.record("small", CompletionResult(text="ok", cost_usd=0.25))
    budget.reserve("large", 1e100)
    if settlement == "release":
        budget.release("large")
    else:
        budget.record("large", CompletionResult(text="free", cost_usd=0))
    assert budget.total_cost_usd == 0.25


def test_exact_fit_remains_valid_through_all_reconciliations():
    budget = Sentinel(max_budget_usd=192 * 0.06)
    for index in range(192):
        budget.reserve(str(index), 0.06, hard_ceiling=True)
    for index in range(192):
        assert budget.record(str(index), CompletionResult(text="ok", cost_usd=0.06)) == 0.06
    assert budget.total_cost_usd == 192 * 0.06
    assert not budget._reservations
    with pytest.raises(SentinelAlert):
        budget.reserve("extra", 0.06)


def test_corrected_report_after_rejection_can_settle_held_reservation():
    budget = _populated_budget()
    result = CompletionResult(text="ok", cost_usd=0.2)
    result.cost_usd = float("nan")
    with pytest.raises(BudgetValidationError):
        budget.add_cost("existing", result)
    result.cost_usd = 0.2
    assert budget.add_cost("existing", result) == 0.45
    assert budget.total_cost_usd == pytest.approx(1.3)
    assert "existing" not in budget._reservations
    assert "existing" not in budget._hard_reservations


def test_valid_overrun_commits_actuals_before_raising():
    budget = Sentinel(max_budget_usd=1)
    budget.reserve("node", 0.1, hard_ceiling=True)
    with pytest.raises(BudgetReconciliationError):
        budget.record("node", CompletionResult(text="ok", cost_usd=0.2))
    assert budget.breakdown() == {"node": 0.2}
    assert budget.total_cost_usd == 0.2
    assert not budget._reservations
