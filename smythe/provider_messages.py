"""Native text-only Fable Messages with durable raw evidence and no retries."""

import asyncio
from contextlib import asynccontextmanager
from decimal import Decimal
import math
import os

from smythe.pricing_anthropic import MODEL, PRICE_VERSION, messages_quote_nanousd, price_messages_response
from smythe.provider import (CompletionResult, Provider, ProviderAccountingCancelledError,
                             ProviderAccountingError, ProviderResponseError)
from smythe.provider_responses import (PreparedRequest, RawResponseEnvelope, ResponseQuote,
                                      ResponseQuoteError, _close_owned, _json_dump, _safe_id)
from smythe.tools import ChatMessage

_ENDPOINT = "https://api.anthropic.com"
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class AnthropicMessagesProvider(Provider):
    """Standard/global Fable 5.1, adaptive thinking, text-only requests."""

    def __init__(self, *, api_key=None, max_output_tokens=8192, reasoning_effort="medium",
                 request_timeout_s=600.0, max_cost_per_call_usd=None):
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 128000:
            raise ValueError("max_output_tokens must be within 1..128000")
        if type(reasoning_effort) is not str or reasoning_effort not in _EFFORTS:
            raise ValueError("Unsupported Messages effort")
        for name, value in (("request_timeout_s", request_timeout_s),
                            ("max_cost_per_call_usd", max_cost_per_call_usd)):
            if value is None and name == "max_cost_per_call_usd":
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if request_timeout_s == 0:
            raise ValueError("Request timeout must be positive")
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self._config = dict(max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort,
                            request_timeout_s=request_timeout_s, max_cost_per_call_usd=max_cost_per_call_usd)
        self._client, self._loop, self._closed = None, None, False

    def workflow_descriptor(self):
        return {"kind": "anthropic_messages", "adapter_version": "anthropic-messages-v1",
                "decoder_version": "anthropic-messages-v1", "endpoint_scope": "global",
                "price_version": PRICE_VERSION, "supported_models": [MODEL], "config": dict(self._config)}

    def snapshot_for_workflow(self):
        return AnthropicMessagesProvider(api_key=self._api_key, **self._config)

    def budget_estimate_usd(self, model):
        return self._config["max_cost_per_call_usd"]

    def requires_explicit_budget_estimate(self, model):
        return True

    def _get_client(self):
        if self._client is not None:
            return self._client
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=self._api_key, base_url=_ENDPOINT,
                              timeout=self._config["request_timeout_s"], max_retries=0)

    @asynccontextmanager
    async def session(self):
        bound = self.snapshot_for_workflow()
        bound._client = bound._get_client()
        bound._loop = asyncio.get_running_loop()
        try:
            yield bound
        finally:
            bound._closed = True
            await _close_owned(bound._client)

    def prepare(self, system, messages, model, tools=None):
        if tools or not isinstance(system, str) or not isinstance(messages, list) or not messages:
            raise ValueError("Messages supports nonempty text-only conversations")
        items = []
        for message in messages:
            if (not isinstance(message, ChatMessage) or message.role not in {"user", "assistant"}
                    or not isinstance(message.content, str) or message.attachments
                    or message.tool_calls or message.tool_results or message.provider_continuation is not None):
                raise ValueError("Unsupported Messages input")
            items.append({"role": message.role, "content": message.content})
        prepared = PreparedRequest(_json_dump({"model": model, "system": system, "messages": items,
                    "max_tokens": self._config["max_output_tokens"], "service_tier": "standard_only",
                    "output_config": {"effort": self._config["reasoning_effort"]}}))
        self._validate_prepared(prepared)
        return prepared

    @staticmethod
    def _validate_prepared(prepared):
        if not isinstance(prepared, PreparedRequest):
            raise TypeError("Expected PreparedRequest")
        p = prepared.payload
        if (set(p) != {"model", "system", "messages", "max_tokens", "service_tier", "output_config"}
                or p["model"] != MODEL or p["service_tier"] != "standard_only"
                or type(p["max_tokens"]) is not int or not 1 <= p["max_tokens"] <= 128000
                or type(p["system"]) is not str
                or p["output_config"] not in [{"effort": effort} for effort in _EFFORTS]
                or type(p["messages"]) is not list or not p["messages"] or prepared.tool_names
                or _json_dump(p) != prepared.payload_json):
            raise ValueError("Request is outside supported Fable Standard/global scope")
        for item in p["messages"]:
            if (type(item) is not dict or set(item) != {"role", "content"}
                    or item["role"] not in {"user", "assistant"} or type(item["content"]) is not str):
                raise ValueError("Messages input must contain text only")
        if p["messages"][-1]["role"] != "user":
            raise ValueError("Fable does not support assistant prefill")
        return p

    async def _raw_call(self, prepared, *, count):
        payload = self._validate_prepared(prepared)
        if self._closed or self._loop is not None and self._loop is not asyncio.get_running_loop():
            raise ValueError("Use Messages inside its owning session loop")
        client = self._get_client()
        if str(client.base_url).rstrip("/") != _ENDPOINT or type(client.max_retries) is not int or client.max_retries != 0:
            raise ValueError("Messages requires global endpoint and zero SDK retries")
        operation = "input_tokens" if count else "response"
        method = client.messages.with_raw_response.create
        if count:
            method = client.messages.with_raw_response.count_tokens
            payload = {key: value for key, value in payload.items()
                       if key not in {"service_tier", "max_tokens"}}
        try:
            try:
                raw = await method(**payload)
                # SDK 1.5 returns AsyncAPIResponse. The non-streaming request
                # has already buffered the HTTP body; do not parse it first.
                body = bytes(raw.http_response.content)
            except asyncio.CancelledError as exc:
                if count:
                    raise
                envelope = RawResponseEnvelope(prepared, b"", operation, transport_error="CancelledError")
                raise ProviderAccountingCancelledError("Messages dispatch cancelled; billing unresolved",
                    envelope=envelope, receipt=self._summary(envelope, self.extract_receipt(envelope))) from exc
            except Exception as exc:
                response = getattr(exc, "response", None)
                return RawResponseEnvelope(prepared, bytes(response.content) if response is not None else b"",
                    operation, getattr(response, "status_code", None),
                    _safe_id(response.headers.get("request-id")) if response is not None else None, type(exc).__name__)
            return RawResponseEnvelope(prepared, body, operation, raw.status_code,
                                       _safe_id(raw.headers.get("request-id")))
        finally:
            if self._client is None:
                await _close_owned(client)

    async def dispatch(self, prepared):
        return await self._raw_call(prepared, count=False)

    async def quote(self, prepared):
        envelope = await self._raw_call(prepared, count=True)
        try:
            if envelope.transport_error or envelope.status_code != 200:
                raise ValueError("Input counting failed")
            count = envelope.json().get("input_tokens")
            cap = prepared.payload["max_tokens"]
            ceiling = messages_quote_nanousd(count, cap, prepared.model)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ResponseQuoteError("Messages input count cannot establish a quote", envelope) from exc
        return ResponseQuote(prepared.request_sha256, count, cap, Decimal(ceiling).scaleb(-9), PRICE_VERSION, envelope)

    @staticmethod
    def extract_receipt(envelope):
        try:
            raw = envelope.json()
        except ValueError:
            raw = {}
        return price_messages_response(raw, requested_model=envelope.request.model)

    @staticmethod
    def _summary(envelope, receipt):
        result = receipt.safe_summary()
        try:
            response_id = _safe_id(envelope.json().get("id"))
        except ValueError:
            response_id = None
        result.update(request_sha256=envelope.request_sha256, response_sha256=envelope.response_sha256,
                      request_id=envelope.request_id, response_id=response_id)
        return result

    @staticmethod
    def decode(envelope):
        receipt = AnthropicMessagesProvider.extract_receipt(envelope)
        summary = AnthropicMessagesProvider._summary(envelope, receipt)
        if receipt.cost_usd is None or envelope.transport_error or envelope.status_code != 200:
            raise ProviderAccountingError("Messages response has unresolved billing", envelope=envelope, receipt=summary)
        result = CompletionResult("", prompt_tokens=receipt.input_tokens, completion_tokens=receipt.output_tokens,
                                  cost_usd=float(receipt.cost_usd), native_receipt=summary, response_envelope=envelope)
        raw = envelope.json()
        try:
            if (raw.get("type") != "message" or raw.get("role") != "assistant"
                    or raw.get("stop_reason") != "end_turn" or summary["response_id"] is None):
                raise ValueError("Messages response did not complete an assistant turn")
            content = raw.get("content")
            if type(content) is not list or not content:
                raise ValueError("Messages response has no content")
            text = []
            for block in content:
                if type(block) is not dict:
                    raise ValueError("Malformed Messages content")
                if block.get("type") in {"thinking", "redacted_thinking"}:
                    continue
                if block.get("type") != "text" or type(block.get("text")) is not str:
                    raise ValueError("Unsupported Messages output")
                text.append(block["text"])
            if not any(item.strip() for item in text):
                raise ValueError("Messages response has no usable text")
        except ValueError as exc:
            raise ProviderResponseError(str(exc), envelope=envelope, receipt=summary, billing_result=result) from exc
        result.text, result.stop_reason = "\n".join(text), "end_turn"
        return result

    async def complete(self, system, prompt, model):
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(self, system, messages, model, tools=None):
        return self.decode(await self.dispatch(self.prepare(system, messages, model, tools)))
