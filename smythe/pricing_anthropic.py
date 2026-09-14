"""Exact Standard/global Fable Messages pricing, independent of output decoding."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

MODEL = "claude-fable-5-1"
PRICE_VERSION = "anthropic-fable-5-1-standard-global-2026-09-13-v1"
SOURCES = (
    "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
    "https://platform.claude.com/docs/en/models/fable-5-1/overview",
)
RATES = {"ordinary_input": 10000, "cache_read": 250, "cache_write_5m": 12500,
         "cache_write_1h": 20000, "output": 50000}


def _count(value, name):
    if type(value) is not int or not 0 <= value <= 10**9:
        raise ValueError(f"Missing or invalid {name}")
    return value


@dataclass(frozen=True)
class ClaudeReceipt:
    actual_model: str | None
    requested_model: str
    service_tier: str | None
    response_status: str | None
    input_tokens: int | None
    output_tokens: int | None
    categories: tuple[tuple[str, int], ...]
    cost_nanousd: int | None
    accounting_error: str | None

    @property
    def cost_usd(self):
        return None if self.cost_nanousd is None else Decimal(self.cost_nanousd).scaleb(-9)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "version": 1, "provider_kind": "anthropic_messages", "price_version": PRICE_VERSION,
            "price_checked_on": "2026-09-13", "pricing_scope": "model_text_tokens",
            "actual_model": self.actual_model, "requested_model": self.requested_model,
            "service_tier": self.service_tier, "endpoint_scope": "global",
            "response_status": self.response_status,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                      "reasoning_tokens": None, "thinking_included_in_output": True},
            "categories": {name: {"tokens": count, "rate_nanousd_per_token": RATES[name],
                                  "cost_nanousd": count * RATES[name]}
                           for name, count in self.categories},
            "cost_nanousd": self.cost_nanousd,
            "cost_usd": None if self.cost_usd is None else format(self.cost_usd, "f"),
            "usage_is_complete": self.accounting_error is None,
            "cost_is_complete": self.cost_nanousd is not None,
            "accounting_error": self.accounting_error, "sources": list(SOURCES),
        }


def price_messages_response(raw, *, requested_model=MODEL):
    """Do not subtract cache tokens from Claude's ordinary input count."""
    raw = raw if type(raw) is dict else {}
    model = raw.get("model") if type(raw.get("model")) is str else None
    usage = raw.get("usage") if type(raw.get("usage")) is dict else {}
    tier = usage.get("service_tier")
    status = raw.get("stop_reason") if type(raw.get("stop_reason")) is str else None
    categories, inputs, outputs, cost, error = (), None, None, None, None
    try:
        if requested_model != MODEL or model != requested_model:
            raise ValueError("Unpriced or unexpected Messages model")
        if tier != "standard":
            raise ValueError("Missing or unsupported Messages service tier")
        if usage.get("inference_geo") not in (None, "global"):
            raise ValueError("Unsupported inference geography")
        ordinary = _count(usage.get("input_tokens"), "input_tokens")
        read = _count(usage.get("cache_read_input_tokens"), "cache_read_input_tokens")
        write = _count(usage.get("cache_creation_input_tokens"), "cache_creation_input_tokens")
        outputs = _count(usage.get("output_tokens"), "output_tokens")
        detail = usage.get("cache_creation")
        if detail is None and write == 0:
            short, long = 0, 0
        elif type(detail) is dict:
            short = _count(detail.get("ephemeral_5m_input_tokens"), "5m cache writes")
            long = _count(detail.get("ephemeral_1h_input_tokens"), "1h cache writes")
            if short + long != write:
                raise ValueError("Cache creation categories disagree with total")
        else:
            raise ValueError("Cache creation TTL accounting missing")
        server = usage.get("server_tool_use")
        if server is not None and (type(server) is not dict or any(
            type(value) is not int or value != 0 for value in server.values()
        )):
            raise ValueError("Server tool charges are outside token-only pricing")
        for item in raw.get("content", []) if type(raw.get("content")) is list else []:
            if type(item) is dict and item.get("type") not in {"text", "thinking", "redacted_thinking"}:
                raise ValueError("Response content is outside supported text/thinking pricing")
        inputs = ordinary + read + write
        categories = (("ordinary_input", ordinary), ("cache_read", read),
                      ("cache_write_5m", short), ("cache_write_1h", long), ("output", outputs))
        cost = sum(count * RATES[name] for name, count in categories)
    except ValueError as exc:
        error = str(exc)
    return ClaudeReceipt(model, requested_model, tier if type(tier) is str else None,
                         status, inputs, outputs, categories, cost, error)


def messages_quote_nanousd(input_tokens, max_tokens, model=MODEL):
    """Reserve uncached input with counting headroom and the entire output cap.

    Token counting is an estimate. A measured overrun closes workflow admission;
    it never disappears from settlement. The input reserve also covers 1h writes.
    """
    count = _count(input_tokens, "input_tokens")
    cap = _count(max_tokens, "max_tokens")
    if model != MODEL or not 1 <= cap <= 128000:
        raise ValueError("Unsupported Messages quote")
    return ((count * 11 + 9) // 10 + 1024) * 20000 + cap * RATES["output"]
