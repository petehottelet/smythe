"""Offline native billing fixtures; no SDK object coercions or provider calls."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal, Inexact, localcontext
import json

import pytest

from smythe.pricing import (
    LONG_CONTEXT_THRESHOLD,
    PRICE_SOURCES,
    PRICE_VERSION,
    SUPPORTED_MODELS,
    conservative_quote,
    price_native_response,
)


def response(*, model="gpt-6-astra", inputs=100, outputs=40, cached=20, writes=30):
    return {
        "id": "resp_offline_fixture",
        "model": model,
        "service_tier": "default",
        "status": "completed",
        "usage": {
            "input_tokens": inputs,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes},
            "output_tokens": outputs,
            "output_tokens_details": {"reasoning_tokens": outputs // 2},
            "total_tokens": inputs + outputs,
        },
        "output": [{"type": "reasoning", "encrypted_content": "do-not-log-this"}],
    }


@pytest.mark.parametrize("model,rates", [
    ("gpt-6-astra", (10_000, 1_000, 12_500, 50_000)),
    ("gpt-5.6-sol", (4_000, 400, 5_000, 20_000)),
])
def test_category_table_prices_each_token_once(model, rates):
    receipt = price_native_response(response(model=model))
    assert receipt.cost_is_complete
    assert receipt.usage_is_complete
    assert receipt.usage_details_complete
    assert receipt.accounting_error is None
    counts = (50, 20, 30, 40)
    assert tuple(x[1] for x in receipt.charges) == counts
    assert tuple(x[2] for x in receipt.charges) == rates
    expected = sum(count * rate for count, rate in zip(counts, rates))
    assert receipt.cost_nanousd == expected
    assert receipt.cost_usd == Decimal(expected) / Decimal(1_000_000_000)
    assert receipt.ordinary_input_tokens == 50
    assert receipt.total_tokens == 140
    assert receipt.reasoning_tokens == 20


@pytest.mark.parametrize("model,rates", [
    ("gpt-6-astra", (10_000, 1_000, 12_500, 50_000)),
    ("gpt-5.6-sol", (4_000, 400, 5_000, 20_000)),
])
@pytest.mark.parametrize("inputs,long_context", [
    (271_999, False), (272_000, False), (272_001, True), (1_000_000, True),
])
def test_whole_request_context_boundary_includes_cache_categories(model, rates, inputs, long_context):
    receipt = price_native_response(response(model=model, inputs=inputs))
    expected_rates = rates if not long_context else (
        rates[0] * 2, rates[1] * 2, rates[2] * 2, rates[3] * 3 // 2,
    )
    assert receipt.long_context is long_context
    assert tuple(x[2] for x in receipt.charges) == expected_rates
    assert receipt.cost_nanousd == sum(
        count * rate for count, rate in zip((inputs - 50, 20, 30, 40), expected_rates)
    )
    assert LONG_CONTEXT_THRESHOLD == 272_000


def test_price_uses_effective_returned_model_instead_of_requested_model():
    raw = response(model="gpt-5.6-sol")
    receipt = price_native_response(raw, requested_model="gpt-6-astra")
    assert receipt.actual_model == "gpt-5.6-sol"
    assert receipt.requested_model == "gpt-6-astra"
    assert receipt.cost_usd == Decimal("0.001158")
    assert receipt.cost_usd == price_native_response(raw, requested_model="not-a-model").cost_usd


@pytest.mark.parametrize("field,value", [
    ("model", None), ("model", ""), ("model", "gpt-6-astra-2026-09-07"),
    ("model", "gpt-5.6"), ("model", "gpt-6-astra-new-snapshot"),
    ("model", "gpt-5.6-sol-pro"), ("model", " gpt-6-astra"), ("model", {}),
    ("service_tier", None), ("service_tier", "standard"), ("service_tier", "auto"),
    ("service_tier", "flex"), ("service_tier", "fast"), ("service_tier", "priority"),
    ("service_tier", "ultrafast"), ("service_tier", "scale"), ("service_tier", True),
])
def test_unknown_returned_identity_never_uses_requested_default(field, value):
    raw = response()
    raw[field] = value
    receipt = price_native_response(raw, requested_model="gpt-6-astra")
    assert receipt.cost_usd is None
    assert receipt.cost_nanousd is None
    assert not receipt.cost_is_complete
    assert receipt.usage_is_complete
    assert receipt.input_tokens == 100
    assert receipt.accounting_error


@pytest.mark.parametrize("field", ["model", "service_tier"])
def test_missing_returned_identity_remains_unknown(field):
    raw = response()
    del raw[field]
    assert price_native_response(raw, requested_model="gpt-6-astra").cost_usd is None


@pytest.mark.parametrize("scope", ["us", "eu", "regional", "", None, False, {}])
def test_non_global_endpoint_is_not_priced_as_global(scope):
    receipt = price_native_response(response(), endpoint_scope=scope)
    assert receipt.cost_usd is None
    assert "Endpoint scope" in receipt.accounting_error


_COUNT_PATHS = [
    ("input_tokens",), ("output_tokens",),
    ("input_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cache_write_tokens"),
    ("output_tokens_details", "reasoning_tokens"), ("total_tokens",),
]


def at_path(raw, path):
    container = raw["usage"]
    for part in path[:-1]:
        container = container[part]
    return container


@pytest.mark.parametrize("path", _COUNT_PATHS)
@pytest.mark.parametrize("value", [True, False, -1, "0", 0.0, 1.5, None, {}, [], float("nan"),
                                  float("inf"), Decimal("0")])
def test_native_counts_reject_coercible_or_invalid_json_values(path, value):
    raw = response()
    at_path(raw, path)[path[-1]] = value
    receipt = price_native_response(raw)
    assert receipt.cost_usd is None
    assert not receipt.usage_is_complete
    assert not receipt.cost_is_complete
    assert "JSON integer" in receipt.accounting_error
    # Invalid values do not leak into the JSON-safe summary.
    json.dumps(receipt.safe_summary(), allow_nan=False)


@pytest.mark.parametrize("path", _COUNT_PATHS[:4])
def test_missing_required_usage_is_unknown_even_when_other_categories_are_zero(path):
    raw = response(inputs=0, outputs=0, cached=0, writes=0)
    del at_path(raw, path)[path[-1]]
    receipt = price_native_response(raw)
    assert receipt.cost_usd is None
    assert "Missing" in receipt.accounting_error


@pytest.mark.parametrize("field,value", [
    ("usage", None), ("usage", []), ("input_tokens_details", None),
    ("input_tokens_details", []), ("output_tokens_details", None),
    ("output_tokens_details", "{}"),
])
def test_invalid_usage_containers_fail_closed(field, value):
    raw = response()
    if field == "usage":
        raw[field] = value
    else:
        raw["usage"][field] = value
    receipt = price_native_response(raw)
    assert not receipt.cost_is_complete
    assert receipt.accounting_error


@pytest.mark.parametrize("raw", [{}, None, [], "not JSON", 0, False])
def test_malformed_response_keeps_an_unknown_receipt(raw):
    receipt = price_native_response(raw, requested_model="gpt-6-astra")
    assert receipt.actual_model is None
    assert receipt.cost_usd is None
    assert not receipt.usage_is_complete
    assert receipt.diagnostics
    json.dumps(receipt.safe_summary(), allow_nan=False)


@pytest.mark.parametrize("optional", ["reasoning_tokens", "output_tokens_details", "total_tokens"])
def test_omitted_optional_usage_does_not_invalidate_token_charge(optional):
    raw = response()
    before = price_native_response(raw)
    if optional == "reasoning_tokens":
        del raw["usage"]["output_tokens_details"][optional]
    else:
        del raw["usage"][optional]
    after = price_native_response(raw)
    assert after.cost_is_complete
    assert after.cost_usd == before.cost_usd
    assert after.usage_is_complete
    assert not after.usage_details_complete
    assert not after.safe_summary()["usage_details_complete"]
    assert after.accounting_error is None
    assert any("Optional" in diagnostic for diagnostic in after.diagnostics)
    if optional != "total_tokens":
        assert after.reasoning_tokens is None


@pytest.mark.parametrize("inputs,cached,writes", [(0, 1, 0), (0, 0, 1), (10, 6, 5)])
def test_cache_categories_must_fit_inside_aggregate_input(inputs, cached, writes):
    receipt = price_native_response(response(inputs=inputs, cached=cached, writes=writes))
    assert receipt.cost_usd is None
    assert receipt.ordinary_input_tokens is None
    assert "exceed input_tokens" in receipt.accounting_error


def test_reasoning_is_subset_of_output_and_must_not_be_billed_twice():
    raw = response()
    expected = price_native_response(raw).cost_usd
    raw["usage"]["output_tokens_details"]["reasoning_tokens"] = 40
    assert price_native_response(raw).cost_usd == expected
    raw["usage"]["output_tokens_details"]["reasoning_tokens"] = 41
    assert "exceed output_tokens" in price_native_response(raw).accounting_error


def test_reported_total_must_match_and_is_not_used_as_an_extra_charge():
    raw = response()
    raw["usage"]["total_tokens"] = 141
    assert "does not equal" in price_native_response(raw).accounting_error


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed", "cancelled", None])
def test_accounting_retains_charge_independently_of_semantic_output_status(status):
    raw = response()
    raw["status"] = status
    receipt = price_native_response(raw)
    assert receipt.cost_is_complete
    assert receipt.response_status == status
    assert receipt.status == status


def test_explicit_zero_cost_is_known_but_missing_usage_is_never_fabricated_zero():
    receipt = price_native_response(response(inputs=0, outputs=0, cached=0, writes=0))
    assert receipt.cost_nanousd == 0
    assert receipt.cost_usd == Decimal(0)
    assert receipt.cost_is_complete
    assert price_native_response({}).cost_usd is None


def test_sub_microdollar_price_is_exact_and_not_rounded_to_zero():
    receipt = price_native_response(response(model="gpt-5.6-sol", inputs=1, outputs=0,
                                            cached=1, writes=0))
    assert receipt.cost_nanousd == 400
    assert receipt.cost_usd == Decimal("0.000000400")
    assert receipt.safe_summary()["cost_usd"] == "0.000000400"


def test_huge_counts_and_low_decimal_precision_do_not_round_or_overflow():
    inputs = 10**400 + 137
    raw = response(inputs=inputs, outputs=17, cached=11, writes=19)
    expected = (inputs - 30) * 20_000 + 11 * 2_000 + 19 * 25_000 + 17 * 75_000
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        receipt = price_native_response(raw)
        quote = conservative_quote(inputs, 17, "gpt-6-astra")
        assert receipt.cost_nanousd == expected
        assert receipt.cost_usd.as_tuple().digits == Decimal(expected).as_tuple().digits
        assert receipt.cost_usd.as_tuple().exponent == -9
        assert quote >= receipt.cost_usd
        summary = receipt.safe_summary()
        assert Decimal(summary["cost_usd"]) == receipt.cost_usd
        assert json.loads(json.dumps(summary))["cost_nanousd"] == expected


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
@pytest.mark.parametrize("inputs", [0, 1, 10, 272_000, 272_001])
def test_quotes_dominate_every_cache_partition_and_output_below_cap(model, inputs):
    cap = 31
    quote = conservative_quote(inputs, cap, model)
    # Exhaust all partitions for small inputs; boundary cases cover all extreme
    # points plus mixed categories of the linear cost function.
    partitions = (
        [(c, w) for c in range(inputs + 1) for w in range(inputs - c + 1)]
        if inputs <= 10 else
        [(0, 0), (inputs, 0), (0, inputs), (inputs // 2, inputs - inputs // 2),
         (inputs // 3, inputs // 3)]
    )
    for cached, writes in partitions:
        for outputs in (0, cap // 2, cap):
            receipt = price_native_response(response(model=model, inputs=inputs,
                outputs=outputs, cached=cached, writes=writes))
            assert quote >= receipt.cost_usd
    highest = price_native_response(response(model=model, inputs=inputs, outputs=cap,
                                            cached=0, writes=inputs))
    assert quote == highest.cost_usd


@pytest.mark.parametrize("field", ["input_tokens", "max_output_tokens"])
@pytest.mark.parametrize("value", [True, False, -1, "0", 0.0, None, float("inf")])
def test_quotes_validate_counts_before_any_provider_call(field, value):
    args = {"input_tokens": 1, "max_output_tokens": 1, "model": "gpt-6-astra"}
    args[field] = value
    with pytest.raises(ValueError, match="JSON integer"):
        conservative_quote(**args)


@pytest.mark.parametrize("kwargs", [
    {"model": "gpt-6-astra-unknown"}, {"model": None}, {"model": {}},
    {"service_tier": "auto"}, {"service_tier": "priority"}, {"service_tier": "standard"},
    {"endpoint_scope": "eu"},
])
def test_quote_rejects_unsupported_identity(kwargs):
    args = {"input_tokens": 1, "max_output_tokens": 1, "model": "gpt-6-astra"} | kwargs
    with pytest.raises(ValueError):
        conservative_quote(**args)


def test_receipt_is_detached_immutable_safe_json_and_has_price_provenance():
    raw = response()
    original = deepcopy(raw)
    receipt = price_native_response(raw, requested_model="gpt-6-astra")
    assert raw == original
    with pytest.raises(FrozenInstanceError):
        receipt.input_tokens = 123
    raw["usage"]["input_tokens"] = 999
    summary = receipt.safe_summary()
    serialized = json.dumps(summary, allow_nan=False)
    assert "do-not-log-this" not in serialized
    assert "encrypted_content" not in serialized
    assert "resp_offline_fixture" not in serialized
    assert summary["price_version"] == PRICE_VERSION
    assert summary["sources"] == list(PRICE_SOURCES)
    assert summary["usage"]["input_tokens"] == 100
    summary["usage"]["input_tokens_details"]["cached_tokens"] = 999
    summary["sources"].clear()
    summary["categories"]["ordinary_input"]["tokens"] = 0
    assert receipt.safe_summary()["usage"]["input_tokens_details"]["cached_tokens"] == 20
    assert receipt.sources == PRICE_SOURCES
    assert receipt.safe_summary()["categories"]["ordinary_input"]["tokens"] == 50


@pytest.mark.parametrize("item", [
    {"type": "web_search_call"}, {"type": "file_search_call"},
    {"type": "image_generation_call"}, {"type": "code_interpreter_call"},
    {"type": "computer_call"}, {"type": "mcp_call"}, {"type": "future_billable_tool"},
    {"type": "audio"}, {"type": None}, {"type": []},
    {"type": "message", "content": [{"type": "output_audio"}]},
    {"type": "message", "content": [{"type": "output_image"}]},
    {"type": "message", "content": [{"type": "future_billable_modality"}]},
])
def test_unexpected_hosted_tools_or_modalities_cannot_claim_complete_price(item):
    raw = response()
    raw["output"].append(item)
    receipt = price_native_response(raw)
    assert receipt.usage_is_complete
    assert receipt.cost_usd is None
    assert "outside model text-token pricing" in receipt.accounting_error


@pytest.mark.parametrize("output", [
    [{"type": "function_call", "arguments": "{}"}],
    [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
    [{"type": "message", "content": [{"type": "refusal", "refusal": "declined"}]}],
    [{"type": "message", "content": [{"type": "output_text"}]}],
    [], None,
])
def test_known_text_charge_survives_adapter_semantic_output_validation(output):
    raw = response()
    raw["output"] = output
    assert price_native_response(raw).cost_is_complete
