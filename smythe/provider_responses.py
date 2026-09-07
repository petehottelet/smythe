"""Native OpenAI Responses calls with explicit quotes and raw billing receipts.

Generation does not count inputs first. ``prepare`` is local; ``quote`` is an
optional HTTP operation. ``dispatch`` returns bytes before either pricing or
output decoding, providing the persistence boundary for a future run ledger.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import math
import os
import re
from typing import Any

from smythe.pricing import (
    PRICE_VERSION, NativeReceipt, conservative_quote, price_native_response,
)
from smythe.provider import (
    CompletionResult, Provider, ProviderAccountingCancelledError, ProviderAccountingError,
    ProviderResponseError,
)
from smythe.tools import ChatMessage, ToolCall, ToolSpec, wire_name

MODELS = frozenset({"gpt-6-astra", "gpt-5.6-sol"})
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_ENDPOINT = "https://api.openai.com/v1"
_NAMESPACE = "openai_responses"
_WIRE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


async def _close_owned(client) -> None:
    """Close on its creating loop without allowing cancellation to erase bytes."""
    closing = asyncio.create_task(client.close())
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not closing.cancelled():
        closing.exception()  # Retrieve cleanup failure without hiding response evidence.


def _json_dump(value: Any) -> str:
    """Detach strict JSON without coercing mapping keys or nonfinite numbers."""
    def check(item: Any, active: set[int]) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (dict, list):
            raise ValueError("Request values must be finite JSON data")
        if id(item) in active:
            raise ValueError("Request JSON cannot contain cycles")
        active.add(id(item))
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("Request JSON keys must be strings")
            children = item.values()
        else:
            children = item
        for child in children:
            check(child, active)
        active.remove(id(item))

    check(value, set())
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _json_load(body: bytes | str) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON value")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed

    try:
        result = json.loads(body, object_pairs_hook=pairs, parse_constant=reject_constant,
                            parse_float=finite_float)
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise ValueError("Response JSON is malformed") from exc
    if not isinstance(result, dict):
        raise ValueError("Response JSON must be an object")
    return result


@dataclass(frozen=True)
class PreparedRequest:
    """Immutable canonical request; ``payload`` returns a fresh detached copy."""

    payload_json: str = field(repr=False)
    tool_names_json: str = field(default="{}", repr=False)

    @property
    def payload(self) -> dict:
        return _json_load(self.payload_json)

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()

    @property
    def model(self) -> str:
        return self.payload["model"]

    @property
    def tool_names(self) -> dict:
        return _json_load(self.tool_names_json)


@dataclass(frozen=True)
class RawResponseEnvelope:
    """Raw HTTP evidence, deliberately excluded from routine representations."""

    request: PreparedRequest = field(repr=False)
    body: bytes = field(repr=False)
    operation: str = "response"
    status_code: int | None = None
    request_id: str | None = None
    transport_error: str | None = None

    @property
    def request_sha256(self) -> str:
        return self.request.request_sha256

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def json(self) -> dict:
        return _json_load(self.body)


@dataclass(frozen=True)
class ResponseQuote:
    """Input-count evidence and a worst-case ceiling for this exact request."""

    request_sha256: str
    input_tokens: int
    max_output_tokens: int
    ceiling_usd: Decimal
    price_version: str
    envelope: RawResponseEnvelope = field(repr=False)


class ResponseQuoteError(ValueError):
    """Input counting failed; no generation request was made by ``quote``."""

    def __init__(self, message: str, envelope: RawResponseEnvelope) -> None:
        self.envelope = envelope
        super().__init__(message)


class OpenAIResponsesProvider(Provider):
    """Stateless text/function Responses for Astra and Sol, Standard/global.

    ``max_cost_per_call_usd`` is an inclusive caller-supplied ceiling required
    by legacy capped Swarm execution. Native receipts are per-call evidence;
    this adapter does not create a durable, whole-workflow accounting ledger.
    """

    def __init__(
        self, *, api_key: str | None = None, max_output_tokens: int = 8192,
        reasoning_effort: str = "medium", max_cost_per_call_usd: float | None = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 128000:
            raise ValueError("max_output_tokens must be an integer between 1 and 128000")
        if not isinstance(reasoning_effort, str) or reasoning_effort not in _EFFORTS:
            raise ValueError("Unsupported reasoning_effort")
        for name, value, allow_zero in (
            ("max_cost_per_call_usd", max_cost_per_call_usd, True),
            ("request_timeout_s", request_timeout_s, False),
        ):
            if value is None and name == "max_cost_per_call_usd":
                continue
            try:
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and value >= 0 and (value != 0 or allow_zero))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._max_cost_per_call_usd = max_cost_per_call_usd
        self._request_timeout_s = request_timeout_s
        self._client = None
        self._session_loop = None
        self._session_closed = False

    def workflow_descriptor(self) -> dict:
        """Describe supported durable configuration without credentials or I/O."""
        return {
            "kind": "openai_responses", "adapter_version": "openai-responses-v1",
            "decoder_version": "openai-responses-v1", "endpoint_scope": "global",
            "price_version": PRICE_VERSION, "supported_models": sorted(MODELS),
            "config": {
                "reasoning_effort": self._reasoning_effort,
                "max_output_tokens": self._max_output_tokens,
                "max_cost_per_call_usd": self._max_cost_per_call_usd,
                "request_timeout_s": self._request_timeout_s,
            },
        }

    def snapshot_for_workflow(self) -> OpenAIResponsesProvider:
        """Detach configuration; a workflow owns its own client lifecycle."""
        return OpenAIResponsesProvider(api_key=self._api_key, **self.workflow_descriptor()["config"])

    @asynccontextmanager
    async def session(self):
        """Yield an independent provider pooling HTTP connections in this loop.

        The bound provider must be used inside this async context only. The
        original provider remains safe across separate synchronous calls.
        """
        bound = OpenAIResponsesProvider(
            api_key=self._api_key, max_output_tokens=self._max_output_tokens,
            reasoning_effort=self._reasoning_effort,
            max_cost_per_call_usd=self._max_cost_per_call_usd,
            request_timeout_s=self._request_timeout_s,
        )
        bound._client = bound._get_client()
        bound._session_loop = asyncio.get_running_loop()
        try:
            yield bound
        finally:
            bound._session_closed = True
            await _close_owned(bound._client)

    def budget_estimate_usd(self, model: str) -> float | None:
        return self._max_cost_per_call_usd

    def requires_explicit_budget_estimate(self, model: str) -> bool:
        return True

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("Install smythe[openai] with openai>=3.8.0") from exc
            version = tuple(int(part) for part in openai.__version__.split(".")[:3]
                            if part.isdigit())
            if version < (3, 8, 0):
                raise ImportError("OpenAIResponsesProvider requires openai>=3.8.0")
            # Explicitly override OPENAI_BASE_URL: these prices cover global
            # OpenAI Standard service, not compatible or regional endpoints.
            return openai.AsyncOpenAI(
                api_key=self._api_key, base_url=_ENDPOINT, max_retries=0,
                timeout=self._request_timeout_s,
            )
        return self._client

    def prepare(
        self, system: str, messages: list[ChatMessage], model: str,
        tools: list[ToolSpec] | None = None,
    ) -> PreparedRequest:
        """Validate and detach a request without creating a client or doing I/O."""
        if model not in MODELS:
            raise ValueError("OpenAIResponsesProvider requires gpt-6-astra or gpt-5.6-sol")
        if not isinstance(system, str) or not isinstance(messages, list) or not messages:
            raise ValueError("Provide string instructions and a non-empty message list")
        input_items: list[dict] = []
        pending_calls: set[str] = set()
        seen_calls: set[str] = set()
        for message in messages:
            if not isinstance(message, ChatMessage) or message.role not in {"user", "assistant"}:
                raise ValueError("Only user and assistant ChatMessage inputs are supported")
            if not isinstance(message.content, str) or message.attachments:
                raise ValueError("Responses currently supports text and function inputs only")
            continuation = message.provider_continuation
            if pending_calls and (message.role != "user" or not message.tool_results):
                raise ValueError("Outstanding native calls require tool results before another turn")
            if continuation is not None:
                if message.role != "assistant" or message.tool_results:
                    raise ValueError("Native continuation belongs to an assistant message")
                if (not isinstance(continuation, dict)
                        or continuation.get("provider") != _NAMESPACE
                        or continuation.get("model") != model):
                    raise ValueError("Native continuation provider/model does not match the request")
                output = continuation.get("output")
                if not isinstance(output, list) or not output:
                    raise ValueError("Native continuation must retain its output items")
                for item in output:
                    if not isinstance(item, dict) or item.get("type") not in {
                        "message", "reasoning", "function_call",
                    }:
                        raise ValueError("Unsupported native continuation item")
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id")
                        if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                            raise ValueError("Native continuation has invalid or duplicate call IDs")
                        seen_calls.add(call_id)
                        pending_calls.add(call_id)
                # Never reconstruct text or tool calls in addition to native
                # output: that duplicates assistant history and drops reasoning.
                input_items.extend(output)
                continue
            if message.tool_calls:
                raise ValueError("Assistant function calls require their native continuation")
            if message.role == "assistant":
                if message.tool_results:
                    raise ValueError("Tool results belong to a user message")
                input_items.append({"role": "assistant", "content": message.content})
                continue
            for result in message.tool_results:
                if (not isinstance(result.tool_call_id, str)
                        or result.tool_call_id not in pending_calls
                        or not isinstance(result.content, str)):
                    raise ValueError("Tool result must match one outstanding native call_id")
                pending_calls.remove(result.tool_call_id)
                input_items.append({
                    "type": "function_call_output", "call_id": result.tool_call_id,
                    "output": f"ERROR: {result.content}" if result.is_error else result.content,
                })
            if message.content or not message.tool_results:
                input_items.append({"role": "user", "content": message.content})
        if pending_calls:
            raise ValueError("Every native function call must have a matching tool result")
        payload = {
            "model": model, "instructions": system, "input": input_items,
            "reasoning": {"effort": self._reasoning_effort}, "service_tier": "default",
            "store": False, "truncation": "disabled", "max_output_tokens": self._max_output_tokens,
        }
        tool_names = {}
        if tools:
            definitions = []
            names: set[str] = set()
            for tool in tools:
                if not isinstance(tool, ToolSpec) or not isinstance(tool.description, str):
                    raise ValueError("Only ToolSpec function definitions are supported")
                name = wire_name(tool.name)
                if name in names:
                    raise ValueError("Tool names collide in the native wire format")
                names.add(name)
                tool_names[name] = tool.name
                if not isinstance(tool.input_schema, dict) or tool.input_schema.get("type") != "object":
                    raise ValueError("Function parameters must be an object JSON schema")
                definitions.append({"type": "function", "name": name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema, "strict": False})
            payload["tools"] = definitions
        prepared = PreparedRequest(_json_dump(payload), _json_dump(tool_names))
        self._validate_prepared(prepared)
        return prepared

    @staticmethod
    def _validate_prepared(prepared: PreparedRequest) -> dict:
        if not isinstance(prepared, PreparedRequest):
            raise TypeError("Expected PreparedRequest")
        payload = prepared.payload
        required = {"model", "instructions", "input", "reasoning", "service_tier",
                    "store", "truncation", "max_output_tokens"}
        if not required <= payload.keys() or payload.keys() - required - {"tools"}:
            raise ValueError("Prepared request contains unsupported or missing parameters")
        cap = payload["max_output_tokens"]
        if (payload["model"] not in MODELS or payload["service_tier"] != "default"
                or payload["store"] is not False or payload["truncation"] != "disabled"
                or type(cap) is not int or not 1 <= cap <= 128000
                or not isinstance(payload["instructions"], str)
                or not isinstance(payload["input"], list) or not payload["input"]
                or payload["reasoning"] not in [{"effort": effort} for effort in _EFFORTS]):
            raise ValueError("Prepared request is outside supported Standard/global configuration")
        if _json_dump(payload) != prepared.payload_json:
            raise ValueError("Prepared request must use canonical JSON")
        tools = payload.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError("Function tools must be a list")
        names = {}
        for tool in tools:
            if (not isinstance(tool, dict) or tool.get("type") != "function"
                    or tool.get("strict") is not False
                    or not isinstance(tool.get("name"), str)
                    or not _WIRE_NAME.fullmatch(tool["name"])
                    or not isinstance(tool.get("description"), str)
                    or not isinstance(tool.get("parameters"), dict)
                    or tool["parameters"].get("type") != "object"
                    or tool.keys() != {"type", "name", "description", "parameters", "strict"}
                    or tool["name"] in names):
                raise ValueError("Only non-strict function tools are supported")
            original_name = prepared.tool_names.get(tool["name"])
            if not isinstance(original_name, str) or wire_name(original_name) != tool["name"]:
                raise ValueError("Prepared function names must retain their neutral identities")
            names[tool["name"]] = original_name
        if names != prepared.tool_names:
            raise ValueError("Prepared function name map does not match declarations")
        pending_calls, seen_calls = set(), set()
        for item in payload["input"]:
            if not isinstance(item, dict):
                raise ValueError("Prepared inputs must be message or function items")
            kind = item.get("type")
            if kind in {None, "message"}:
                if item.get("role") not in {"user", "assistant"}:
                    raise ValueError("Prepared message role is unsupported")
                content = item.get("content")
                if isinstance(content, str):
                    continue
                if (not isinstance(content, list) or any(
                    not isinstance(part, dict) or part.get("type") not in {"input_text", "output_text"}
                    or not isinstance(part.get("text"), str) for part in content
                )):
                    raise ValueError("Prepared inputs support text only")
            elif kind == "function_call":
                call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
                if (not _safe_id(call_id) or call_id in seen_calls
                        or not isinstance(name, str) or not _WIRE_NAME.fullmatch(name)
                        or not isinstance(arguments, str)
                        or item.get("status") not in {None, "completed"}):
                    raise ValueError("Prepared native function call is malformed")
                _json_load(arguments)
                pending_calls.add(call_id)
                seen_calls.add(call_id)
            elif kind == "function_call_output":
                call_id = item.get("call_id")
                if not _safe_id(call_id) or call_id not in pending_calls or not isinstance(item.get("output"), str):
                    raise ValueError("Prepared tool output must match one outstanding call")
                pending_calls.remove(call_id)
            elif kind == "reasoning":
                if item.get("status") not in {None, "completed"}:
                    raise ValueError("Prepared reasoning item did not complete")
            else:
                raise ValueError("Prepared native input type is unsupported")
        if pending_calls:
            raise ValueError("Prepared native calls are missing tool results")
        return payload

    async def _raw_call(self, prepared: PreparedRequest, *, count: bool) -> RawResponseEnvelope:
        payload = self._validate_prepared(prepared)
        if self._session_closed or (
            self._session_loop is not None and self._session_loop is not asyncio.get_running_loop()
        ):
            raise ValueError("Bound Responses provider must be used in its active session loop")
        client = self._get_client()
        if (str(getattr(client, "base_url", "")).rstrip("/") != _ENDPOINT
                or type(getattr(client, "max_retries", None)) is not int
                or client.max_retries != 0):
            raise ValueError("Responses client must use the global OpenAI endpoint and max_retries=0")
        method = client.responses.with_raw_response.create
        if count:
            payload = {key: value for key, value in payload.items()
                       if key not in {"service_tier", "store", "max_output_tokens"}}
            method = client.responses.input_tokens.with_raw_response.count
        owned = self._client is None
        try:
            try:
                raw = await method(**payload)
            except asyncio.CancelledError as exc:
                envelope = RawResponseEnvelope(
                    prepared, b"", "input_tokens" if count else "response",
                    transport_error="CancelledError",
                )
                if count:
                    raise  # Counting cancellation has not dispatched generation.
                summary = self._summary(envelope, self.extract_receipt(envelope))
                raise ProviderAccountingCancelledError(
                    "Native generation was cancelled after dispatch; billing is unresolved",
                    envelope=envelope, receipt=summary,
                ) from exc
            except Exception as exc:
                # SDK HTTP errors still carry raw evidence; transport failures
                # without a response remain explicit unknown exposure.
                response = getattr(exc, "response", None)
                return RawResponseEnvelope(
                    prepared, bytes(response.content) if response is not None else b"",
                    "input_tokens" if count else "response",
                    getattr(response, "status_code", None),
                    _safe_id(response.headers.get("x-request-id")) if response is not None else None,
                    type(exc).__name__,
                )
            return RawResponseEnvelope(
                prepared, bytes(raw.content), "input_tokens" if count else "response",
                raw.status_code, _safe_id(raw.headers.get("x-request-id")),
            )
        finally:
            if owned:
                # Serial Swarm uses multiple asyncio.run loops. Each owned
                # HTTP pool therefore closes in the loop that created it.
                # Once bytes arrived, cancellation during close cannot erase
                # that evidence. Injected clients remain caller-owned.
                await _close_owned(client)

    async def dispatch(self, prepared: PreparedRequest) -> RawResponseEnvelope:
        """Make one generation attempt, returning bytes before validating them."""
        return await self._raw_call(prepared, count=False)

    async def quote(self, prepared: PreparedRequest) -> ResponseQuote:
        """Explicitly count inputs and price a conservative whole-call ceiling."""
        envelope = await self._raw_call(prepared, count=True)
        try:
            if envelope.transport_error or envelope.status_code != 200:
                raise ValueError("Input-count request failed")
            count = envelope.json().get("input_tokens")
            payload = prepared.payload
            cost = conservative_quote(count, payload["max_output_tokens"], payload["model"])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ResponseQuoteError("Input-count response cannot establish a quote", envelope) from exc
        return ResponseQuote(prepared.request_sha256, count, payload["max_output_tokens"],
                             cost, PRICE_VERSION, envelope)

    @staticmethod
    def extract_receipt(envelope: RawResponseEnvelope) -> NativeReceipt:
        """Pure raw-JSON pricing; callers can persist the envelope before this."""
        try:
            raw = envelope.json()
        except ValueError:
            raw = {}
        return price_native_response(raw, requested_model=envelope.request.model)

    @staticmethod
    def _summary(envelope: RawResponseEnvelope, receipt: NativeReceipt) -> dict:
        try:
            response_id = _safe_id(envelope.json().get("id"))
        except ValueError:
            response_id = None
        summary = receipt.safe_summary()
        summary.update(request_sha256=envelope.request_sha256,
                       response_sha256=envelope.response_sha256, request_id=envelope.request_id,
                       response_id=response_id)
        return summary

    @staticmethod
    def decode(envelope: RawResponseEnvelope, receipt: NativeReceipt | None = None) -> CompletionResult:
        """Return usable output or raise an error retaining its incurred bill."""
        extracted = OpenAIResponsesProvider.extract_receipt(envelope)
        mismatched_receipt = receipt is not None and (
            not isinstance(receipt, NativeReceipt) or receipt != extracted
        )
        receipt = extracted
        summary = OpenAIResponsesProvider._summary(envelope, receipt)
        if receipt.accounting_error or receipt.cost_usd is None:
            raise ProviderAccountingError(
                "Native response has unresolved billing evidence", envelope=envelope, receipt=summary,
            )
        try:
            billing = CompletionResult(
                text="", prompt_tokens=receipt.input_tokens, completion_tokens=receipt.output_tokens,
                cost_usd=float(receipt.cost_usd), native_receipt=summary, response_envelope=envelope,
            )
        except (ValueError, OverflowError) as exc:
            raise ProviderAccountingError(
                "Native bill cannot be represented by the legacy cost ledger",
                envelope=envelope, receipt=summary,
            ) from exc
        try:
            if mismatched_receipt:
                raise ValueError("Supplied native receipt does not match response evidence")
            raw = envelope.json()
            if (envelope.transport_error or envelope.status_code != 200
                    or raw.get("status") != "completed" or raw.get("error") is not None
                    or raw.get("incomplete_details") is not None):
                raise ValueError("Native response did not complete")
            if summary["response_id"] is None:
                raise ValueError("Native response has no response ID")
            if raw.get("model") != envelope.request.model:
                raise ValueError("Native response model differs from requested model")
            output = raw.get("output")
            if not isinstance(output, list) or not output:
                raise ValueError("Native response contains no output")
            texts, calls = [], []
            call_ids: set[str] = set()
            declared = {tool["name"] for tool in envelope.request.payload.get("tools", [])}
            for item in output:
                if not isinstance(item, dict):
                    raise ValueError("Native output item is malformed")
                kind = item.get("type")
                if kind == "reasoning":
                    if item.get("status") not in {None, "completed"}:
                        raise ValueError("Native reasoning item did not complete")
                    continue
                if kind == "message" and item.get("status") != "completed":
                    raise ValueError("Native output item did not complete")
                if kind == "message":
                    content = item.get("content")
                    if item.get("role") != "assistant" or not isinstance(content, list):
                        raise ValueError("Native assistant message is malformed")
                    for part in content:
                        if (not isinstance(part, dict) or part.get("type") != "output_text"
                                or not isinstance(part.get("text"), str)):
                            raise ValueError("Native message does not contain usable text")
                        texts.append(part["text"])
                elif kind == "function_call":
                    if item.get("status") not in {None, "completed"}:
                        raise ValueError("Native function call did not complete")
                    call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
                    if (not _safe_id(call_id) or call_id in call_ids
                            or not isinstance(name, str) or not _WIRE_NAME.fullmatch(name)
                            or name not in declared or not isinstance(arguments, str)):
                        raise ValueError("Native function call is malformed or undeclared")
                    parsed_arguments = _json_load(arguments)
                    call_ids.add(call_id)
                    calls.append(ToolCall(call_id, envelope.request.tool_names[name], parsed_arguments))
                else:
                    raise ValueError("Native output type is unsupported")
            if not calls and not any(text.strip() for text in texts):
                raise ValueError("Native response has no usable text or function call")
            continuation = json.loads(_json_dump({
                "provider": _NAMESPACE, "model": raw["model"], "output": output,
            }))
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ProviderResponseError(
                str(exc), envelope=envelope, receipt=summary, billing_result=billing,
            ) from exc
        billing.text = "\n".join(texts)
        billing.tool_calls = calls
        billing.stop_reason = "tool_use" if calls else "end_turn"
        billing.provider_continuation = continuation
        return billing

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(
        self, system: str, messages: list[ChatMessage], model: str,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        prepared = self.prepare(system, messages, model, tools)
        envelope = await self.dispatch(prepared)
        return self.decode(envelope)
