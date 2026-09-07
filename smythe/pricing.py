"""Exact native Responses token pricing, pinned to a dated Standard/global table.

Only the returned model and service tier select a price. Requested identities
are retained for inspection but never fill gaps in a provider's billing data.
The table covers model tokens; hosted tool fees and other modalities require
their own accounting and are outside this module's supported request scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any


PRICE_VERSION = "openai-native-standard-global-2026-09-07-v1"
PRICE_CHECKED_ON = "2026-09-07"
LONG_CONTEXT_THRESHOLD = 272_000
PRICE_SOURCES = (
    "https://developers.openai.com/api/docs/pricing",
    "https://developers.openai.com/api/docs/guides/prompt-caching",
    "https://developers.openai.com/api/docs/models/gpt-6-astra",
    "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "https://developers.openai.com/api/reference/resources/responses/methods/create",
)


@dataclass(frozen=True, slots=True)
class _RateCard:
    # Integer billionths of one USD per token; no intermediate float rounding.
    ordinary: int
    cached: int
    cache_write: int
    output: int

    def for_input(self, input_tokens: int) -> tuple[int, int, int, int]:
        if input_tokens > LONG_CONTEXT_THRESHOLD:
            return self.ordinary * 2, self.cached * 2, self.cache_write * 2, self.output * 3 // 2
        return self.ordinary, self.cached, self.cache_write, self.output


# These exact IDs are the complete snapshot lists on the model pages above.
# Do not infer prices from a model-name prefix or an undocumented dated ID.
_RATE_CARDS = MappingProxyType({
    "gpt-6-astra": _RateCard(10_000, 1_000, 12_500, 50_000),
    "gpt-5.6-sol": _RateCard(4_000, 400, 5_000, 20_000),
})
SUPPORTED_MODELS = tuple(_RATE_CARDS)
_CATEGORIES = ("ordinary_input", "cached_input", "cache_write_input", "output")
_MISSING = object()


def _usd(nanousd: int) -> Decimal:
    """Convert integer nanoUSD exactly, independently of Decimal context limits."""
    value = Decimal(nanousd).as_tuple()
    return Decimal((value.sign, value.digits, value.exponent - 9))


@dataclass(frozen=True, slots=True)
class NativeReceipt:
    """Immutable accounting summary; response text and encrypted items stay out.

    Unknown billing has ``cost_usd is None`` and ``accounting_error``. A known
    token charge does not mean the response is usable: the adapter separately
    validates response status and output before delivering a completion.
    """

    actual_model: str | None
    service_tier: str | None
    endpoint_scope: str | None
    requested_model: str | None
    response_status: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    reported_total_tokens: int | None
    ordinary_input_tokens: int | None
    usage_is_complete: bool
    # usage_is_complete means required billing categories validate; optional
    # reasoning/total detail availability is exposed separately below.
    accounting_error: str | None
    diagnostics: tuple[str, ...]
    long_context: bool | None
    # Entries are (category, token count, nanoUSD per token, total nanoUSD).
    charges: tuple[tuple[str, int, int, int], ...] = ()
    cost_nanousd: int | None = None
    price_version: str = PRICE_VERSION
    sources: tuple[str, ...] = PRICE_SOURCES

    @property
    def cost_usd(self) -> Decimal | None:
        return None if self.cost_nanousd is None else _usd(self.cost_nanousd)

    @property
    def cost_is_complete(self) -> bool:
        return self.cost_nanousd is not None and self.accounting_error is None

    @property
    def usage_details_complete(self) -> bool:
        """Whether optional reasoning and reported-total details also validate."""
        return (
            self.usage_is_complete
            and self.reasoning_tokens is not None
            and self.reported_total_tokens is not None
        )

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def status(self) -> str | None:
        return self.response_status

    def safe_summary(self) -> dict[str, Any]:
        """Return detached JSON data with exact money and validated counts only."""
        return {
            "version": 1,
            "price_version": self.price_version,
            "price_checked_on": PRICE_CHECKED_ON,
            "pricing_scope": "model_text_tokens",
            "actual_model": self.actual_model,
            "service_tier": self.service_tier,
            "endpoint_scope": self.endpoint_scope,
            "requested_model": self.requested_model,
            "response_status": self.response_status,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "input_tokens_details": {
                    "cached_tokens": self.cached_tokens,
                    "cache_write_tokens": self.cache_write_tokens,
                },
                "output_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
                "reported_total_tokens": self.reported_total_tokens,
                "total_tokens": self.total_tokens,
                "ordinary_input_tokens": self.ordinary_input_tokens,
            },
            "usage_is_complete": self.usage_is_complete,
            "usage_details_complete": self.usage_details_complete,
            "cost_is_complete": self.cost_is_complete,
            "cost_usd": None if self.cost_usd is None else format(self.cost_usd, "f"),
            "cost_nanousd": self.cost_nanousd,
            "long_context": self.long_context,
            "long_context_threshold": LONG_CONTEXT_THRESHOLD,
            "categories": {
                category: {
                    "tokens": count,
                    "rate_nanousd_per_token": rate,
                    "cost_nanousd": charge,
                    "cost_usd": format(_usd(charge), "f"),
                }
                for category, count, rate, charge in self.charges
            },
            "accounting_error": self.accounting_error,
            "diagnostics": list(self.diagnostics),
            "sources": list(self.sources),
        }


def _string(value: object) -> str | None:
    return value if type(value) is str else None


def _count(value: object, name: str, errors: list[str], *, optional: bool = False) -> int | None:
    if value is _MISSING:
        if not optional:
            errors.append(f"Missing {name}")
        return None
    if type(value) is not int or value < 0:
        errors.append(f"{name} must be a non-negative JSON integer")
        return None
    return value


def _output_scope_errors(raw: Mapping[str, Any]) -> list[str]:
    """Reject output types that can carry charges outside the token-only table.

    Missing output is a semantic failure for the adapter to handle after
    settlement. Refusals and malformed text remain token-billed responses.
    Explicit unknown types cannot receive a complete token-only invoice.
    """
    errors: list[str] = []
    output = raw.get("output")
    if not isinstance(output, list):
        return errors
    for item in output:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if type(kind) is not str or kind not in {"message", "reasoning", "function_call"}:
            errors.append("Output item type is outside model text-token pricing")
            continue
        if kind == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                content_type = part.get("type")
                if type(content_type) is not str or content_type not in {"output_text", "refusal"}:
                    errors.append("Message content type is outside model text-token pricing")
    return errors


def price_native_response(
    raw: Mapping[str, Any],
    *,
    requested_model: str | None = None,
    endpoint_scope: str = "global",
) -> NativeReceipt:
    """Price validated native usage or retain an explicit unknown-cost receipt.

    A missing cache counter is not zero. Reasoning and total-token details are
    optional, but when supplied they must agree with aggregate output/usage.
    The long-context multiplier applies to every category of the full request.
    """
    errors: list[str] = []
    if not isinstance(raw, Mapping):
        errors.append("Native response must be a JSON object")
        raw = {}
    actual_model = _string(raw.get("model"))
    service_tier = _string(raw.get("service_tier"))
    scope = _string(endpoint_scope)
    card = _RATE_CARDS.get(actual_model) if actual_model is not None else None
    if actual_model is None:
        errors.append("Missing or invalid returned model")
    elif card is None:
        errors.append("Returned model is outside the versioned price table")
    if service_tier is None:
        errors.append("Missing or invalid returned service_tier")
    elif service_tier != "default":
        errors.append("Returned service_tier is outside the Standard price table")
    if scope != "global":
        errors.append("Endpoint scope is outside the global price table")
    errors.extend(_output_scope_errors(raw))

    usage_errors: list[str] = []
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        usage_errors.append("Missing or invalid usage object")
        usage = {}
    details = usage.get("input_tokens_details")
    if not isinstance(details, Mapping):
        usage_errors.append("Missing or invalid usage.input_tokens_details object")
        details = {}
    inputs = _count(usage.get("input_tokens", _MISSING), "usage.input_tokens", usage_errors)
    outputs = _count(usage.get("output_tokens", _MISSING), "usage.output_tokens", usage_errors)
    cached = _count(
        details.get("cached_tokens", _MISSING),
        "usage.input_tokens_details.cached_tokens", usage_errors,
    )
    writes = _count(
        details.get("cache_write_tokens", _MISSING),
        "usage.input_tokens_details.cache_write_tokens", usage_errors,
    )
    output_details = usage.get("output_tokens_details", _MISSING)
    if output_details is _MISSING:
        output_details = {}
    elif not isinstance(output_details, Mapping):
        usage_errors.append("Invalid usage.output_tokens_details object")
        output_details = {}
    reasoning = _count(
        output_details.get("reasoning_tokens", _MISSING),
        "usage.output_tokens_details.reasoning_tokens", usage_errors, optional=True,
    )
    total = _count(
        usage.get("total_tokens", _MISSING), "usage.total_tokens", usage_errors, optional=True,
    )
    ordinary = None
    if inputs is not None and cached is not None and writes is not None:
        if cached + writes > inputs:
            usage_errors.append("Cached plus cache-write tokens exceed input_tokens")
        else:
            ordinary = inputs - cached - writes
    if reasoning is not None and outputs is not None and reasoning > outputs:
        usage_errors.append("Reasoning tokens exceed output_tokens")
    if total is not None and inputs is not None and outputs is not None:
        if total != inputs + outputs:
            usage_errors.append("Reported total_tokens does not equal input_tokens + output_tokens")
    errors.extend(usage_errors)
    detail_diagnostics: list[str] = []
    if "reasoning_tokens" not in output_details:
        detail_diagnostics.append("Optional reasoning_tokens detail was not reported")
    if "total_tokens" not in usage:
        detail_diagnostics.append("Optional total_tokens detail was not reported")

    charges: tuple[tuple[str, int, int, int], ...] = ()
    cost = None
    if not errors:
        assert card is not None and inputs is not None and outputs is not None
        assert ordinary is not None and cached is not None and writes is not None
        counts = ordinary, cached, writes, outputs
        charges = tuple(
            (category, count, rate, count * rate)
            for category, count, rate in zip(_CATEGORIES, counts, card.for_input(inputs))
        )
        cost = sum(charge for _, _, _, charge in charges)
    return NativeReceipt(
        actual_model=actual_model,
        service_tier=service_tier,
        endpoint_scope=scope,
        requested_model=_string(requested_model),
        response_status=_string(raw.get("status")),
        input_tokens=inputs,
        output_tokens=outputs,
        cached_tokens=cached,
        cache_write_tokens=writes,
        reasoning_tokens=reasoning,
        reported_total_tokens=total,
        ordinary_input_tokens=ordinary,
        usage_is_complete=not usage_errors,
        accounting_error="; ".join(errors) if errors else None,
        diagnostics=tuple(errors + detail_diagnostics),
        long_context=None if inputs is None else inputs > LONG_CONTEXT_THRESHOLD,
        charges=charges,
        cost_nanousd=cost,
    )


def conservative_quote(
    input_tokens: int,
    max_output_tokens: int,
    model: str,
    *,
    service_tier: str = "default",
    endpoint_scope: str = "global",
) -> Decimal:
    """Bound token cost with every input charged at the highest applicable rate.

    The caller must supply a defensible input bound and an enforced output cap
    for this exact model/tier/endpoint. This does not quote hosted tool charges.
    """
    errors: list[str] = []
    inputs = _count(input_tokens, "input_tokens", errors)
    outputs = _count(max_output_tokens, "max_output_tokens", errors)
    card = _RATE_CARDS.get(model) if type(model) is str else None
    if card is None:
        errors.append("Quote model is outside the versioned price table")
    if type(service_tier) is not str or service_tier != "default":
        errors.append("Quote service_tier must be default")
    if type(endpoint_scope) is not str or endpoint_scope != "global":
        errors.append("Quote endpoint_scope must be global")
    if errors:
        raise ValueError("; ".join(errors))
    assert inputs is not None and outputs is not None and card is not None
    ordinary, cached, writes, output = card.for_input(inputs)
    return _usd(inputs * max(ordinary, cached, writes) + outputs * output)
