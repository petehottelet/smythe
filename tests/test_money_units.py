"""Exact money units: the shared conversions, their call sites and the ledgers.

The ordinary Sentinel holds float USD, the durable journal integer nanoUSD,
and Jobs and the Autotune ledger integer microUSD. The same charge must
reconcile exactly across them, unknown cost must never read as zero, and a
reservation must never shrink when it changes unit.
"""

from contextlib import closing
from decimal import Decimal, DecimalException, Inexact, localcontext
import hashlib
import json
import math
import random
import sqlite3
import uuid

import pytest

from fakes import LOST_RESPONSE, NativeTransport
from smythe import OfflineProvider, SQLiteWorkflowStore, Swarm, Task
from smythe import pricing, pricing_anthropic
from smythe._money import (
    CEILING, EXACT, FLOOR, MICROUSD, NANOUSD, USD, MoneyUnitError, convert, from_units, to_units,
)
from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.jobs import JobManifestV1, make_approval, preflight_job
from smythe.jobs.models import (
    MAX_SIGNED_MICRO_USD, ManifestValidationError, micros_to_usd, normalize_usd, usd_string,
    usd_to_micros,
)
from smythe.jobs.store import SQLiteRunStore
from smythe.optimize.contracts import Candidate, ExperimentContract, MetricObjective, ObjectiveDirection
from smythe.optimize.ledger import ExperimentLedger
from smythe.provider import CompletionResult, ProviderAccountingError
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.workflow import _nanousd
from smythe.workflow_store import WorkflowValidationError

MAX63 = MAX_SIGNED_MICRO_USD  # (1 << 63) - 1
MICRODOLLAR = Decimal("0.000001")
OPENAI_VERSION = "openai-native-standard-global-2026-09-07-v1"
MESSAGES_VERSION = "anthropic-fable-5-1-standard-global-2026-09-13-v1"


# ---------------------------------------------------------------------------
# The conversions before they were shared, kept verbatim as references.
# ---------------------------------------------------------------------------


def legacy_pricing_usd(nanousd):
    value = Decimal(nanousd).as_tuple()
    return Decimal((value.sign, value.digits, value.exponent - 9))


def legacy_workflow_nanousd(value):
    if value is None:
        return None
    amount = Decimal(str(value)).as_tuple()
    coefficient = int("".join(map(str, amount.digits)))
    exponent = amount.exponent + 9
    return coefficient * 10 ** exponent if exponent >= 0 else coefficient // 10 ** -exponent


def legacy_quote_nanousd(quote):
    digits = quote.as_tuple()
    coefficient = int("".join(map(str, digits.digits)))
    exponent = digits.exponent + 9
    if exponent < 0:
        raise ValueError("Quote cannot be represented as nanoUSD")
    return coefficient * 10 ** exponent


def legacy_normalize_usd(value, *, field_name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ManifestValidationError(f"{field_name} must be a USD number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ManifestValidationError(f"{field_name} must be finite")
    try:
        amount = Decimal(str(value))
    except (DecimalException, ValueError) as exc:
        raise ManifestValidationError(f"{field_name} must be a USD number") from exc
    if not amount.is_finite():
        raise ManifestValidationError(f"{field_name} must be finite")
    if amount < 0:
        raise ManifestValidationError(f"{field_name} must be non-negative")
    try:
        quantized = amount.quantize(MICRODOLLAR)
    except (DecimalException, ValueError) as exc:
        raise ManifestValidationError(f"{field_name} must fit exact microdollar precision") from exc
    if quantized != amount:
        raise ManifestValidationError(f"{field_name} supports at most six decimal places")
    if quantized > Decimal(MAX63) * MICRODOLLAR:
        raise ManifestValidationError(f"{field_name} exceeds signed 63-bit microdollar capacity")
    return quantized


def legacy_usd_to_micros(value):
    return int(legacy_normalize_usd(value, field_name="USD amount") / MICRODOLLAR)


def legacy_micros_to_usd(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX63:
        raise ManifestValidationError("microdollar amount must be a non-negative signed 63-bit integer")
    return (Decimal(value) * MICRODOLLAR).quantize(MICRODOLLAR)


def outcome(function, value):
    """A comparable result: the exact Decimal form, or the error and its text."""
    try:
        result = function(value)
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return type(error).__name__, str(error)
    return result.as_tuple() if isinstance(result, Decimal) else result


NANO_COUNTS = [0, 1, 9, 10, 999, 1000, 1001, 999_999, 1_000_000, 1_500_000, 2**53 - 1, 2**53,
               2**53 + 1, MAX63, MAX63 + 1, 10**27, 10**28 - 1]
BUDGETS = [0, 0.0, -0.0, 1, 5.0, 0.1, 0.3, 1e-9, 9e-10, 1.5e-9, 1e-10, 12345.678901234,
           0.0000000009, 1e20, 1e-300, 1.7976931348623157e308, 2**53, MAX63, 3.14159265358979,
           0.123456789123, 100.0000000015]
USD_INPUTS = ["0", "-0", "-0.0", -0.0, 0, 0.0, 0.03, "0.03", "12.345678", "0.0000001",
              "0.0000005", "1.0000000", "1." + "0" * 40, "1." + "0" * 5000, "1." + "0" * 5000 + "1",
              "9223372036854.775807", "9223372036854.775808", str(2**63),
              "9999999999999999999999.9999994", "9999999999999999999999.9999995",
              "9999999999999999999999.999999", "1E+22", "1e999999", "1e-999999", -1, "nan",
              float("inf"), True, Decimal("0.000001"), Decimal("1E-7"), Decimal("5E+3"), 1e-7, 1e20,
              123456789, "abc", None, [], "12345678901234567890123.4",
              "0.12345678901234567890123456789"]
MICRO_COUNTS = [0, 1, 999_999, 1_000_000, 2**53, MAX63, MAX63 + 1, -1, True, 1.0, "1"]


# ---------------------------------------------------------------------------
# The shared conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amount,unit,expected", [
    (Decimal(0), NANOUSD, 0),
    (Decimal("1E-9"), NANOUSD, 1),
    (Decimal("0.000001"), MICROUSD, 1),
    (Decimal("9223372036.854775807"), NANOUSD, MAX63),
    (Decimal("9223372036854.775807"), MICROUSD, MAX63),
    (Decimal(MAX63), USD, MAX63),
    (0.0015, NANOUSD, 1_500_000),
    (0.3, NANOUSD, 300_000_000),
    (5, MICROUSD, 5_000_000),
    (-0.0, NANOUSD, 0),
    (Decimal("-0"), MICROUSD, 0),
    (Decimal("0E+999999999"), NANOUSD, 0),
    (Decimal("1." + "0" * 5000), MICROUSD, 1_000_000),
    (Decimal("5E+3"), MICROUSD, 5_000_000_000),
])
def test_exact_conversion_of_boundary_amounts(amount, unit, expected):
    assert to_units(amount, unit) == to_units(amount, unit, EXACT) == expected
    assert to_units(amount, unit, CEILING) == to_units(amount, unit, FLOOR) == expected


@pytest.mark.parametrize("amount,unit,floor,ceiling", [
    (Decimal("1E-10"), NANOUSD, 0, 1),
    (Decimal("0.0000000015"), NANOUSD, 1, 2),
    (Decimal("1.0000000001"), NANOUSD, 1_000_000_000, 1_000_000_001),
    (Decimal("0.0000001"), MICROUSD, 0, 1),
    (Decimal("9223372036854.7758069"), MICROUSD, MAX63 - 1, MAX63),
    (Decimal("1E-999999999"), NANOUSD, 0, 1),
    (1e-10, NANOUSD, 0, 1),
    # A float is read by the decimal its repr shows, not its binary expansion.
    (0.1 + 0.2, NANOUSD, 300_000_000, 300_000_001),
])
def test_rounding_names_its_direction_and_exact_refuses_a_fraction(amount, unit, floor, ceiling):
    assert to_units(amount, unit, FLOOR) == floor
    assert to_units(amount, unit, CEILING) == ceiling
    with pytest.raises(MoneyUnitError, match="finer than one"):
        to_units(amount, unit)


def test_ceiling_never_rounds_down_and_floor_never_rounds_up():
    generator = random.Random(20261003)
    amounts = [Decimal(generator.randrange(0, 10**30)).scaleb(-generator.randrange(0, 25))
               for _ in range(2000)]
    for amount in amounts:
        for unit in (NANOUSD, MICROUSD):
            one = from_units(1, unit)
            high = from_units(to_units(amount, unit, CEILING), unit)
            low = from_units(to_units(amount, unit, FLOOR), unit)
            assert low <= amount <= high and high - low <= one
            assert (high == amount) is (low == amount)


@pytest.mark.parametrize("amount", [
    None, True, "1", [], Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), float("nan"),
    float("inf"), -1, -1e-9, Decimal("-1E-9"),
])
def test_unknown_or_invalid_amounts_never_convert(amount):
    for rounding in (EXACT, CEILING, FLOOR):
        with pytest.raises(MoneyUnitError):
            to_units(amount, NANOUSD, rounding)


def test_units_and_roundings_must_be_named_exactly():
    for unit in ("nanousd", "USD ", None, MICROUSD.encode()):
        with pytest.raises(MoneyUnitError, match="unit"):
            to_units(Decimal(1), unit)
        with pytest.raises(MoneyUnitError, match="unit"):
            from_units(1, unit)
    for rounding in ("round", None, "EXACT", 0):
        with pytest.raises(MoneyUnitError, match="rounding"):
            to_units(Decimal(1), NANOUSD, rounding)
        with pytest.raises(MoneyUnitError, match="rounding"):
            convert(1, NANOUSD, MICROUSD, rounding)


def test_huge_amounts_are_refused_without_building_huge_integers():
    with pytest.raises(MoneyUnitError, match="too large"):
        to_units(Decimal("1E+999999999"), NANOUSD)
    # The largest float still converts.
    assert to_units(1.7976931348623157e308, NANOUSD, FLOOR) == 17976931348623157 * 10**301


def test_conversions_ignore_the_callers_decimal_context():
    expected = [to_units(Decimal("12345.678901234"), NANOUSD),
                to_units(Decimal("1.0000000005"), NANOUSD, CEILING),
                from_units(MAX63, NANOUSD), from_units(MAX63, MICROUSD)]
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        assert [to_units(Decimal("12345.678901234"), NANOUSD),
                to_units(Decimal("1.0000000005"), NANOUSD, CEILING),
                from_units(MAX63, NANOUSD), from_units(MAX63, MICROUSD)] == expected


@pytest.mark.parametrize("count,unit,text", [
    (0, NANOUSD, "0E-9"), (1_500_000, NANOUSD, "0.001500000"), (MAX63, NANOUSD, "9223372036.854775807"),
    (0, MICROUSD, "0.000000"), (1, MICROUSD, "0.000001"), (MAX63, MICROUSD, "9223372036854.775807"),
    (7, USD, "7"),
])
def test_whole_units_become_exact_usd_with_the_units_decimal_places(count, unit, text):
    amount = from_units(count, unit)
    assert str(amount) == text and amount.as_tuple().exponent == -{USD: 0, MICROUSD: 6, NANOUSD: 9}[unit]
    assert to_units(amount, unit) == count


@pytest.mark.parametrize("count", [-1, True, 1.0, None, "1", Decimal(1)])
def test_only_non_negative_integers_are_whole_units(count):
    with pytest.raises(MoneyUnitError, match="non-negative integer"):
        from_units(count, NANOUSD)
    with pytest.raises(MoneyUnitError, match="non-negative integer"):
        convert(count, NANOUSD, MICROUSD, CEILING)


@pytest.mark.parametrize("nanousd,floor,ceiling", [
    (0, 0, 0), (1, 0, 1), (999, 0, 1), (1000, 1, 1), (1001, 1, 2), (1_500_000, 1500, 1500),
    (MAX63, MAX63 // 1000, MAX63 // 1000 + 1),
])
def test_nano_to_micro_bridge_never_rounds_a_reservation_down(nanousd, floor, ceiling):
    assert convert(nanousd, NANOUSD, MICROUSD, FLOOR) == floor
    assert convert(nanousd, NANOUSD, MICROUSD, CEILING) == ceiling
    assert convert(ceiling, MICROUSD, NANOUSD) >= nanousd
    if nanousd % 1000:
        with pytest.raises(MoneyUnitError, match="nanoUSD amount is finer than one microUSD"):
            convert(nanousd, NANOUSD, MICROUSD)
    else:
        assert convert(nanousd, NANOUSD, MICROUSD) == floor


def test_micro_to_nano_is_exact_and_round_trips():
    for micros in (0, 1, 999_999, MAX63):
        nanos = convert(micros, MICROUSD, NANOUSD, FLOOR)
        assert nanos == micros * 1000 == convert(micros, MICROUSD, NANOUSD)
        assert convert(nanos, NANOUSD, MICROUSD) == micros
        assert from_units(nanos, NANOUSD) == from_units(micros, MICROUSD)


# ---------------------------------------------------------------------------
# The replaced call sites keep every value, receipt and message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", NANO_COUNTS)
def test_receipt_usd_amounts_are_unchanged(count):
    assert pricing._usd(count).as_tuple() == legacy_pricing_usd(count).as_tuple()
    receipt = pricing_anthropic.ClaudeReceipt(None, "claude-fable-5-1", None, None, None, None, (),
                                              count, None)
    assert receipt.cost_usd.as_tuple() == Decimal(count).scaleb(-9).as_tuple()


@pytest.mark.parametrize("value", BUDGETS)
def test_durable_budget_floor_is_unchanged(value):
    assert _nanousd(value) == legacy_workflow_nanousd(value)
    with localcontext() as context:
        context.prec = 2
        assert _nanousd(value) == legacy_workflow_nanousd(value)
    assert _nanousd(None) is None


@pytest.mark.parametrize("model", sorted(pricing.SUPPORTED_MODELS))
@pytest.mark.parametrize("inputs,cap", [(0, 1), (1, 1), (100, 10), (272_000, 128_000),
                                        (272_001, 128_000), (10**9, 128_000)])
def test_durable_quote_ceilings_are_unchanged(tmp_path, model, inputs, cap):
    quote = pricing.conservative_quote(inputs, cap, model, price_version=OPENAI_VERSION)
    call = {"provider_json": json.dumps({"kind": "openai_responses"}),
            "request_json": json.dumps({"model": model, "max_output_tokens": cap}),
            "price_version": OPENAI_VERSION, "call_id": "call", "request_sha": "a" * 64}
    evidence = {"transport_error": None, "status_code": 200, "evidence_id": "evidence",
                "body": json.dumps({"input_tokens": inputs}).encode()}
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        assert store._quote(call, evidence)["ceiling_nanousd"] == legacy_quote_nanousd(quote)


def test_durable_quote_finer_than_a_nanousd_keeps_its_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "conservative_quote", lambda *args, **kwargs: Decimal("1E-10"))
    call = {"provider_json": json.dumps({"kind": "openai_responses"}),
            "request_json": json.dumps({"model": "gpt-6-astra", "max_output_tokens": 1}),
            "price_version": OPENAI_VERSION, "call_id": "call", "request_sha": "a" * 64}
    evidence = {"transport_error": None, "status_code": 200, "evidence_id": "evidence",
                "body": b'{"input_tokens": 1}'}
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(WorkflowValidationError, match="^Quote cannot be represented as nanoUSD$"):
            store._quote(call, evidence)


@pytest.mark.parametrize("value", USD_INPUTS, ids=[str(index) for index in range(len(USD_INPUTS))])
def test_jobs_amounts_and_messages_are_unchanged(value):
    assert outcome(lambda item: normalize_usd(item, field_name="f"), value) == outcome(
        lambda item: legacy_normalize_usd(item, field_name="f"), value)
    assert outcome(usd_to_micros, value) == outcome(legacy_usd_to_micros, value)


@pytest.mark.parametrize("value", MICRO_COUNTS)
def test_jobs_microdollar_amounts_are_unchanged(value):
    assert outcome(micros_to_usd, value) == outcome(legacy_micros_to_usd, value)


@pytest.mark.parametrize("value", ["0", "-0", 0.03, "12.345678", "9223372036854.775807",
                                   Decimal("1.5"), Decimal("0.0000005"), Decimal("2.5000005")])
def test_jobs_usd_strings_are_unchanged(value):
    assert usd_string(Decimal(value)) == format(Decimal(value).quantize(MICRODOLLAR), "f")
    if not isinstance(value, Decimal):
        assert usd_string(normalize_usd(value, field_name="f")) == format(
            legacy_normalize_usd(value, field_name="f").quantize(MICRODOLLAR), "f")


def test_jobs_amounts_do_not_depend_on_the_decimal_context():
    """Regression: a caller's lower decimal precision rejected valid amounts."""
    largest = "9223372036854.775807"
    with localcontext() as context:
        context.prec = 10
        context.traps[Inexact] = True
        assert normalize_usd(largest, field_name="f") == Decimal(largest)
        assert usd_to_micros(Decimal(largest)) == MAX63
        assert micros_to_usd(MAX63).as_tuple() == Decimal(largest).as_tuple()
        assert usd_string(micros_to_usd(MAX63)) == largest
        with pytest.raises(ManifestValidationError, match="at most six decimal places"):
            normalize_usd("0.0000001", field_name="f")


MESSAGES_RESPONSE = {
    "model": "claude-fable-5-1", "stop_reason": "end_turn", "content": [{"type": "text", "text": "x"}],
    "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
              "cache_creation_input_tokens": 30,
              "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20},
              "service_tier": "standard"},
}


def test_messages_receipt_amounts_do_not_depend_on_the_decimal_context():
    """Regression: a lower decimal precision rounded a Messages receipt's cost_usd."""
    expected = pricing_anthropic.price_messages_response(MESSAGES_RESPONSE).safe_summary()
    assert expected["cost_usd"] == "0.002535000"
    with localcontext() as context:
        context.prec = 5
        receipt = pricing_anthropic.price_messages_response(MESSAGES_RESPONSE)
        assert receipt.safe_summary() == expected
        assert str(receipt.cost_usd) == "0.002535000"


def _digest(summary):
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()


_RESPONSES_USAGE = {"input_tokens": 100, "output_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}


# Receipts are compared byte for byte when a durable call settles again, so
# these digests of receipts at fixed price versions must never change.
@pytest.mark.parametrize("raw,digest", [
    ({"model": "gpt-6-astra", "service_tier": "default", "status": "completed",
      "usage": _RESPONSES_USAGE},
     "ea8491b259aa985218b3c67e6224832dbb3c6641a333d160aeb64d374cfd64d8"),
    ({"model": "gpt-5.6-sol", "service_tier": "default", "status": "completed",
      "usage": {"input_tokens": 300_000, "output_tokens": 77, "total_tokens": 300_077,
                "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 7}}},
     "187726549abfd337088115115d04f8eeb86b5c7c741f256c7adc609f7f76c237"),
    ({"model": "gpt-6-astra", "service_tier": "default", "status": "completed",
      "usage": {"input_tokens": 0, "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}},
     "0795ea35c4dc73e23f26f5be1864a608fa9e89cfb40ac1e00efb085b426196bd"),
    ({"model": "gpt-6-astra", "service_tier": "default", "usage": None},
     "3e952f64b3add3d93fd83a823c1342cbdb0d2729088f9b0dfc1d2d5e9338f738"),
], ids=["astra", "sol-long-context", "zero", "unpriced"])
def test_native_receipts_keep_their_bytes(raw, digest):
    receipt = pricing.price_native_response(raw, requested_model="gpt-6-astra",
                                            price_version=OPENAI_VERSION)
    assert _digest(receipt.safe_summary()) == digest


@pytest.mark.parametrize("raw,digest", [
    (MESSAGES_RESPONSE, "252b4450a18fe4943c5994dde3d66a464aa2dd244ef2f90a0c87eff21f0564da"),
    ({"model": "claude-fable-5-1", "stop_reason": "end_turn", "content": [],
      "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "service_tier": "standard"}},
     "835892faea93718dd4e228634f4527c9104a521e66c46bd3cbf1db28d0b151bd"),
    ({"model": "claude-fable-5-1", "usage": {}},
     "ce80fc411b9c13e5658ec45ea918450d9524bf9b9948007ab77a2105f49299be"),
], ids=["fable", "zero", "unpriced"])
def test_messages_receipts_keep_their_bytes(raw, digest):
    receipt = pricing_anthropic.price_messages_response(raw, price_version=MESSAGES_VERSION)
    assert _digest(receipt.safe_summary()) == digest


def _offline_durable_run(path):
    provider = OfflineProvider(plan={"nodes": [{"id": "write", "label": "Write the report"}]})
    with SQLiteWorkflowStore(path) as store:
        result = Swarm(model="offline", provider=provider, run_store=store).execute(Task("Report"))
    with closing(sqlite3.connect(path)) as db:
        stored = [row[0] for row in db.execute("SELECT receipt_json FROM workflow_calls")]
    return result, stored


def test_durable_offline_receipts_keep_the_string_zero(tmp_path):
    result, stored = _offline_durable_run(tmp_path / "runs.db")
    assert result.total_cost_usd == 0 and result.workflow_accounting["confirmed_nanousd"] == 0
    assert stored and all(json.loads(receipt)["cost_nanousd"] == "0" for receipt in stored)
    assert all(json.loads(receipt)["cost_usd"] == "0" for receipt in stored)


# ---------------------------------------------------------------------------
# Unit tags in projections, and the cross-ledger fixture: journal, Jobs, Autotune
# ---------------------------------------------------------------------------


def _jobs_run(store, root, ceiling_usd):
    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "money-units",
        "profiles": [{"name": "image", "provider": "openai_image", "model": "gpt-image-1",
                      "max_cost_per_call_usd": ceiling_usd}],
        "operations": [{"key": "item", "count": 2, "prompt": "Draw", "profile": "image"}],
        "execution": {"max_concurrency": 1, "max_attempts": 1, "max_budget_usd": "1",
                      "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=root)
    run_id = store.create_run(plan, make_approval(plan), manifest_root=root)
    calls = []
    for operation in store.pending_operations(run_id):
        attempt = store.begin_attempt(run_id, operation["operation_id"])
        permit = store.prepare_call(attempt["attempt_id"], usd_to_micros(normalize_usd(
            ceiling_usd, field_name="ceiling")))
        store.mark_call_dispatched(permit.call_id)
        calls.append(permit.call_id)
    return run_id, calls


def _autotune_trials(ledger, ceiling_microusd):
    contract = ExperimentContract(
        name="money_units", objectives=(MetricObjective(
            name="quality", direction=ObjectiveDirection.MAXIMIZE, primary=True, hard_min=0),),
        mutable_fields=("prompt.logo",), development_repetitions=1, confirmation_repetitions=3,
        holdout_repetitions=3, max_candidates=1, max_parallel_candidates=1, max_trials=10,
        max_wall_seconds=600, max_budget_microusd=10 * ceiling_microusd,
        per_trial_reservation_microusd=ceiling_microusd, confidence=0.95, min_improvement=0.25,
        base_seed=100,
    )
    candidate = Candidate(contract=contract, policy={"prompt.logo": "plain"}, hypothesis="Baseline")
    campaign = ledger.create_campaign(contract, candidate.candidate_id, (candidate,),
                                      plan_hash="sha256:" + "a" * 64)
    ledger.register_candidate(campaign, candidate)
    lease = ledger.acquire_campaign_lease(campaign, uuid.uuid4().hex, ttl_s=3600)
    trials = []
    for seed in (100, 101):
        trial = ledger.prepare_trial(campaign, candidate.candidate_id, "development", "evaluate",
                                     seed, lease=lease, evaluator_hash="sha256:" + "e" * 64,
                                     ceiling_microusd=ceiling_microusd)
        ledger.mark_trial_dispatched(trial, lease=lease)
        trials.append(trial)
    return campaign, lease, trials


def test_one_charge_reconciles_exactly_across_the_journal_jobs_and_autotune(tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = Swarm(model="gpt-6-astra", max_budget_usd=1, run_store=store,
                       provider=OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100),
                       ).execute(graph)
        calls = store.inspect_run(result.execution_id)["calls"]
    accounting = result.workflow_accounting
    assert accounting["unit"] == NANOUSD and not accounting["unknown_nanousd"]
    # One known charge: 100 input and 10 output tokens on gpt-6-astra.
    nanousd = accounting["confirmed_nanousd"]
    assert nanousd == 1_500_000 * len(transport.requests) == sum(call["cost_nanousd"] for call in calls)
    microusd = convert(nanousd, NANOUSD, MICROUSD)

    with SQLiteRunStore(tmp_path / "jobs.db") as jobs:
        run_id, (charged, _) = _jobs_run(jobs, tmp_path, "0.10")
        jobs.complete_call(charged, cost_microusd=microusd, cost_is_complete=True,
                           cost_is_estimate=False, artifacts=[], result_text="drawn")
        jobs_cost = jobs.snapshot(run_id)["cost"]
    with ExperimentLedger(tmp_path / "autotune.db") as ledger:
        campaign, lease, (completed, _) = _autotune_trials(ledger, 100_000)
        ledger.complete_trial(completed, lease=lease, metrics={"quality": 1.0}, gates={},
                              actual_cost_microusd=microusd, duration_ms=1.0)
        autotune_cost = ledger.snapshot(campaign)["cost"]

    assert jobs_cost["unit"] == MICROUSD
    exact = {from_units(nanousd, NANOUSD), from_units(jobs_cost["confirmed_microusd"], MICROUSD),
             from_units(autotune_cost["confirmed_microusd"], MICROUSD)}
    assert exact == {Decimal("0.0015") * len(transport.requests)}
    # The float projections read back to the same exact amount.
    assert to_units(result.total_cost_usd, NANOUSD) == nanousd
    assert to_units(float(next(iter(exact))), MICROUSD) == microusd


def test_unknown_exposure_is_never_zero_in_any_ledger(tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    transport.schedule("execution", 0, LOST_RESPONSE)
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(ProviderAccountingError, match="unresolved billing"):
            Swarm(model="gpt-6-astra", max_budget_usd=1, run_store=store,
                  provider=OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100),
                  ).execute(graph)
        state = store.inspect_run(store.list_runs()[0]["run_id"])
    (unknown_call,) = [call for call in state["calls"] if call["billing_state"] == "unknown"]
    held = state["unknown_nanousd"]
    assert held == unknown_call["ceiling_nanousd"] > 0 and state["unknown_calls"] == 1
    # Exposure moves into micro-USD ledgers rounded up, never down.
    reservation = convert(held, NANOUSD, MICROUSD, CEILING)
    assert from_units(reservation, MICROUSD) >= from_units(held, NANOUSD) > 0

    with SQLiteRunStore(tmp_path / "jobs.db") as jobs:
        ceiling = str(from_units(reservation, MICROUSD))
        run_id, (lost, _) = _jobs_run(jobs, tmp_path, ceiling)
        jobs.mark_unknown_outcome(lost, "provider connection lost")
        jobs_cost = jobs.snapshot(run_id)["cost"]
    assert jobs_cost["exposure_microusd"] == reservation and not jobs_cost["cost_is_complete"]
    with ExperimentLedger(tmp_path / "autotune.db") as ledger:
        campaign, lease, (lost_trial, _) = _autotune_trials(ledger, reservation)
        ledger.mark_trial_unknown(lost_trial, "provider connection lost", lease=lease)
        assert ledger.snapshot(campaign)["cost"]["unknown_exposure_microusd"] == reservation

    # The float ledger: a charge the provider could not report stays unknown.
    sentinel = Sentinel()
    sentinel.add_cost("draft", CompletionResult(text="", cost_usd_unknown=True))
    assert not sentinel.cost_is_complete
    with pytest.raises(MoneyUnitError):
        to_units(None, MICROUSD, CEILING)


def test_projections_tag_units_that_stored_records_do_not_carry(tmp_path):
    result, stored = _offline_durable_run(tmp_path / "runs.db")
    assert result.workflow_accounting["unit"] == NANOUSD == "nanoUSD"
    assert stored and not any('"unit"' in receipt for receipt in stored)

    with SQLiteRunStore(tmp_path / "jobs.db") as jobs:
        run_id, _ = _jobs_run(jobs, tmp_path, "0.10")
        snapshot = jobs.snapshot(run_id)
        assert snapshot["cost"]["unit"] == MICROUSD == "microUSD"
        assert jobs.list_runs()[0]["cost"] == snapshot["cost"]
        assert jobs.inspection_snapshot(run_id)["cost"] == snapshot["cost"]
    with closing(sqlite3.connect(tmp_path / "jobs.db")) as db:
        columns = {row[1] for table in ("runs", "calls") for row in db.execute(f"PRAGMA table_info({table})")}
    assert "unit" not in columns
