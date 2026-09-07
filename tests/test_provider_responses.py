"""Offline native Responses contract, including an optional real SDK transport."""

import asyncio
from dataclasses import FrozenInstanceError
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe.budget import BudgetValidationError
from smythe.provider import (
    Artifact, ProviderAccountingCancelledError, ProviderAccountingError, ProviderResponseError,
)
from smythe.provider_responses import (
    OpenAIResponsesProvider, PreparedRequest, ResponseQuoteError,
)
from smythe.tools import ChatMessage, ToolCall, ToolResult, ToolSpec

MODEL = "gpt-6-astra"
KEY = "sk-offline-not-a-real-key"


def response(**overrides):
    return {
        "id": "resp_fixture", "model": MODEL, "status": "completed", "service_tier": "default",
        "usage": {"input_tokens": 1000, "input_tokens_details": {
            "cached_tokens": 200, "cache_write_tokens": 300,
        }, "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 25},
            "total_tokens": 1040},
        "output": [{"type": "message", "id": "msg_fixture", "status": "completed",
                    "role": "assistant", "phase": "final_answer", "content": [
                        {"type": "output_text", "text": "The answer.", "annotations": []},
                    ]}],
        **overrides,
    }


def wire(body, status=200):
    return SimpleNamespace(content=body if isinstance(body, bytes) else json.dumps(body).encode(),
                           status_code=status, headers={"x-request-id": "req_fixture"})


def provider(body=None, *, count=1000, **kwargs):
    p = OpenAIResponsesProvider(api_key=KEY, **kwargs)
    generation = AsyncMock(return_value=wire(response() if body is None else body))
    counting = AsyncMock(return_value=wire({"input_tokens": count}))
    p._client = SimpleNamespace(
        base_url="https://api.openai.com/v1/", max_retries=0,
        responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=generation),
                                  input_tokens=SimpleNamespace(
                                      with_raw_response=SimpleNamespace(count=counting))),
    )
    return p, generation, counting


def prepared(p, *, tools=None):
    return p.prepare("System instructions.", [ChatMessage("user", "Question.")], MODEL, tools)


def tool():
    return ToolSpec("lookup.item", "Read one item", {
        "type": "object", "properties": {"name": {"type": "string"},
                                          "limit": {"type": "integer"}},
        "required": ["name"],
    })


def tool_output():
    return [
        {"type": "reasoning", "id": "rs_fixture", "summary": [],
         "encrypted_content": "opaque-secret", "future_reasoning_field": {"keep": True}},
        {"type": "message", "id": "msg_commentary", "status": "completed",
         "role": "assistant", "phase": "commentary",
         "content": [{"type": "output_text", "text": "Checking.", "annotations": []}]},
        {"type": "function_call", "id": "fc_item_id", "call_id": "call_tool_id",
         "status": "completed", "name": "lookup__item", "arguments": '{"name":"x"}'},
    ]


def test_complete_one_generation_no_automatic_quote():
    p, generation, counting = provider()
    result = asyncio.run(p.complete("System instructions.", "Question.", MODEL))
    assert result.text == "The answer."
    assert result.prompt_tokens == 1000 and result.completion_tokens == 40
    assert result.cost_usd == pytest.approx(.01095)
    assert result.native_receipt["cost_usd"] is not None
    assert result.native_receipt["response_id"] == "resp_fixture"
    assert result.response_envelope.body == wire(response()).content
    assert result.response_envelope.request_id == "req_fixture"
    assert generation.await_count == 1 and counting.await_count == 0
    assert generation.call_args.kwargs == prepared(p).payload
    assert generation.call_args.kwargs["max_output_tokens"] == 8192
    assert generation.call_args.kwargs["store"] is False
    assert not {"temperature", "top_p", "max_tokens", "include"} & generation.call_args.kwargs.keys()


def test_prepare_is_detached_immutable_and_local():
    p = OpenAIResponsesProvider(api_key=KEY)
    specification = tool()
    message = ChatMessage("user", "Before.")
    request = p.prepare("System", [message], MODEL, [specification])
    expected_hash = request.request_sha256
    message.content = "After."
    specification.input_schema["required"].append("limit")
    payload = request.payload
    payload["input"][0]["content"] = "Changed copy."
    assert request.payload["input"][0]["content"] == "Before."
    assert request.payload["tools"][0]["parameters"]["required"] == ["name"]
    assert request.payload["tools"][0]["strict"] is False
    assert request.request_sha256 == expected_hash and p._client is None
    with pytest.raises(FrozenInstanceError):
        request.payload_json = "{}"


def test_quote_is_explicit_and_binds_full_request():
    p, generation, counting = provider()
    request = prepared(p, tools=[tool()])
    quote = asyncio.run(p.quote(request))
    assert quote.ceiling_usd == Decimal("0.422100")
    assert quote.input_tokens == 1000 and quote.max_output_tokens == 8192
    assert quote.request_sha256 == request.request_sha256
    assert quote.envelope.operation == "input_tokens"
    assert generation.await_count == 0 and counting.await_count == 1
    assert counting.call_args.kwargs == {
        key: value for key, value in request.payload.items()
        if key not in {"service_tier", "store", "max_output_tokens"}
    }
    other = OpenAIResponsesProvider(max_output_tokens=100).prepare(
        "System instructions.", [ChatMessage("user", "Question.")], MODEL, [tool()],
    )
    assert other.request_sha256 != quote.request_sha256


@pytest.mark.parametrize("count", [False, 0.0, "0", "", -1, None])
def test_quote_rejects_raw_malformed_counts(count):
    p, generation, _ = provider(count=count)
    with pytest.raises(ResponseQuoteError) as caught:
        asyncio.run(p.quote(prepared(p)))
    assert type(caught.value.envelope.json()["input_tokens"]) is type(count)
    assert generation.await_count == 0


@pytest.mark.parametrize("path", ["input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens"])
@pytest.mark.parametrize("value", [False, 0.0, "0", "", -1, None])
def test_native_billing_rejects_raw_values_without_coercion(path, value):
    body = response()
    destination = body["usage"] if path in {"input_tokens", "output_tokens"} else body["usage"]["input_tokens_details"]
    destination[path] = value
    p, generation, _ = provider(body)
    with pytest.raises(ProviderAccountingError) as caught:
        asyncio.run(p.complete("S", "P", MODEL))
    error = caught.value
    assert isinstance(error, BudgetValidationError)
    assert error.billing_result is None
    assert error.envelope.body == wire(body).content
    assert generation.await_count == 1


@pytest.mark.parametrize("field", ["model", "service_tier", "usage"])
def test_missing_price_identity_or_usage_is_unknown(field):
    body = response()
    del body[field]
    p, _, _ = provider(body)
    with pytest.raises(ProviderAccountingError) as caught:
        asyncio.run(p.complete("S", "P", MODEL))
    assert caught.value.receipt["cost_usd"] is None


@pytest.mark.parametrize("field", ["cached_tokens", "cache_write_tokens"])
def test_missing_cache_field_is_never_zero(field):
    body = response()
    del body["usage"]["input_tokens_details"][field]
    p, _, _ = provider(body)
    with pytest.raises(ProviderAccountingError):
        asyncio.run(p.complete("S", "P", MODEL))


def test_optional_reasoning_detail_is_not_needed_to_price():
    body = response()
    del body["usage"]["output_tokens_details"]
    p, _, _ = provider(body)
    assert asyncio.run(p.complete("S", "P", MODEL)).cost_usd == pytest.approx(.01095)


@pytest.mark.parametrize("updates", [
    {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    {"status": "failed", "error": {"code": "server_error"}},
    {"output": []}, {"id": None}, {"status": None},
    {"output": [{"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "refusal", "refusal": "No."}]}]},
])
def test_unusable_output_retains_known_bill(updates):
    body = response(**updates)
    p, generation, _ = provider(body)
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(p.complete("S", "P", MODEL))
    error = caught.value
    assert not isinstance(error, ProviderAccountingError)
    assert error.billing_result.cost_usd == pytest.approx(.01095)
    assert error.billing_result.text == "" and not error.billing_result.tool_calls
    assert error.envelope.body == wire(body).content
    assert generation.await_count == 1


@pytest.mark.parametrize("body", [b"not-json", b"[]", b'{"usage":null,"usage":{}}', b'{"x":NaN}'])
def test_dispatch_returns_raw_before_any_validation(body):
    p, _, _ = provider(body)
    envelope = asyncio.run(p.dispatch(prepared(p)))
    assert envelope.body == body
    with pytest.raises(ProviderAccountingError) as caught:
        p.decode(envelope)
    assert caught.value.envelope is envelope


def test_tool_history_replays_all_items_once_and_uses_call_id():
    items = tool_output()
    p, generation, _ = provider(response(output=items))
    first = asyncio.run(p.chat("S", [ChatMessage("user", "P")], MODEL, [tool()]))
    assert first.tool_calls == [ToolCall("call_tool_id", "lookup.item", {"name": "x"})]
    assert first.provider_continuation["output"] == items
    assert "opaque-secret" not in repr(first)
    assert "opaque-secret" not in json.dumps(first.native_receipt)
    generation.return_value = wire(response())
    history = [ChatMessage("user", "P"), ChatMessage(
        "assistant", first.text, tool_calls=first.tool_calls,
        provider_continuation=first.provider_continuation,
    ), ChatMessage("user", tool_results=[ToolResult("call_tool_id", "Found.")])]
    request = p.prepare("S", history, MODEL, [tool()])
    assert request.payload["input"] == [
        {"role": "user", "content": "P"}, *items,
        {"type": "function_call_output", "call_id": "call_tool_id", "output": "Found."},
    ]
    first.provider_continuation["output"][0]["encrypted_content"] = "changed"
    assert request.payload["input"][1]["encrypted_content"] == "opaque-secret"
    assert asyncio.run(p.chat("S", history, MODEL, [tool()])).text == "The answer."


@pytest.mark.parametrize("arguments", [
    '{"bad":', "[]", "null", '{"x":NaN}', '{"x":1,"x":2}', '{"x":1e309}',
])
def test_malformed_function_arguments_are_terminal_with_known_bill(arguments):
    items = tool_output()
    items[-1]["arguments"] = arguments
    p, generation, _ = provider(response(output=items))
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(p.chat("S", [ChatMessage("user", "P")], MODEL, [tool()]))
    assert caught.value.billing_result.cost_usd == pytest.approx(.01095)
    assert not caught.value.billing_result.tool_calls and generation.await_count == 1


@pytest.mark.parametrize("message", [
    ChatMessage("system", "P"), ChatMessage("user", attachments=[Artifact(b"x")]),
    ChatMessage("assistant", tool_calls=[ToolCall("c", "lookup.item", {})]),
    ChatMessage("user", tool_results=[ToolResult("missing", "x")]),
    ChatMessage("assistant", provider_continuation={"provider": "another", "model": MODEL, "output": []}),
    ChatMessage("assistant", provider_continuation={"provider": "openai_responses", "model": "gpt-5.6-sol", "output": []}),
])
def test_unsupported_inputs_reject_before_http(message):
    p, generation, counting = provider()
    with pytest.raises(ValueError):
        asyncio.run(p.chat("S", [message], MODEL))
    generation.assert_not_awaited()
    counting.assert_not_awaited()


@pytest.mark.parametrize("kwargs", [
    {"max_output_tokens": False}, {"max_output_tokens": 0}, {"max_output_tokens": 128001},
    {"reasoning_effort": "none"}, {"max_cost_per_call_usd": True},
    {"max_cost_per_call_usd": -1}, {"max_cost_per_call_usd": float("nan")},
    {"request_timeout_s": 0}, {"request_timeout_s": float("inf")},
    {"max_cost_per_call_usd": 10 ** 1000}, {"reasoning_effort": []},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OpenAIResponsesProvider(**kwargs)


def test_legacy_budget_requires_inclusive_explicit_ceiling():
    assert OpenAIResponsesProvider().requires_explicit_budget_estimate(MODEL)
    assert OpenAIResponsesProvider().budget_estimate_usd(MODEL) is None
    assert OpenAIResponsesProvider(max_cost_per_call_usd=.5).budget_estimate_usd(MODEL) == .5


@pytest.mark.parametrize("url,retries", [
    ("https://example.com/v1", 0), ("https://eu.api.openai.com/v1", 0),
    ("https://api.openai.com/v1", 2),
])
def test_injected_client_cannot_change_pricing_scope_or_hide_retries(url, retries):
    p, generation, _ = provider()
    p._client.base_url, p._client.max_retries = url, retries
    with pytest.raises(ValueError, match="global OpenAI"):
        asyncio.run(p.complete("S", "P", MODEL))
    generation.assert_not_awaited()


def test_transport_error_is_unknown_exposure_without_sdk_retries():
    p, generation, _ = provider()
    generation.side_effect = RuntimeError("transport detail should not enter traces")
    with pytest.raises(ProviderAccountingError) as caught:
        asyncio.run(p.complete("S", "P", MODEL))
    assert caught.value.envelope.transport_error == "RuntimeError"
    assert caught.value.envelope.body == b""
    assert "transport detail" not in str(caught.value)
    assert generation.await_count == 1


def test_prepared_request_rejects_added_sampling_parameter():
    p, generation, _ = provider()
    payload = prepared(p).payload
    payload["temperature"] = 1
    altered = PreparedRequest(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        asyncio.run(p.dispatch(altered))
    generation.assert_not_awaited()


@pytest.mark.parametrize("item", [
    {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]},
    {"type": "computer_call", "call_id": "c", "action": {"type": "screenshot"}},
    {"role": "developer", "content": "unsupported"},
    {"role": "assistant", "content": [{"type": "input_audio", "data": "x"}]},
])
def test_constructed_prepared_request_rejects_unsupported_nested_input(item):
    p, generation, counting = provider()
    payload = prepared(p).payload
    payload["input"] = [item]
    altered = PreparedRequest(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        asyncio.run(p.dispatch(altered))
    with pytest.raises(ValueError):
        asyncio.run(p.quote(altered))
    generation.assert_not_awaited()
    counting.assert_not_awaited()


def test_function_names_preserve_literal_underscores():
    specification = ToolSpec("lookup__literal", "Read", {"type": "object"})
    items = tool_output()
    items[-1]["name"] = "lookup__literal"
    p, _, _ = provider(response(output=items))
    result = asyncio.run(p.chat("S", [ChatMessage("user", "P")], MODEL, [specification]))
    assert result.tool_calls[0].name == "lookup__literal"


@pytest.mark.parametrize("status", [None, "absent"])
def test_function_call_status_is_optional_in_real_wire_schema(status):
    items = tool_output()
    if status == "absent":
        del items[-1]["status"]
    else:
        items[-1]["status"] = status
    p, _, _ = provider(response(output=items))
    result = asyncio.run(p.chat("S", [ChatMessage("user", "P")], MODEL, [tool()]))
    assert result.tool_calls[0].id == "call_tool_id"


def test_swapped_receipt_cannot_override_the_actual_charge():
    p, generation, _ = provider()
    original = asyncio.run(p.dispatch(prepared(p)))
    receipt = p.extract_receipt(original)
    expensive = response()
    expensive["usage"]["input_tokens"] += 1000
    expensive["usage"]["total_tokens"] += 1000
    generation.return_value = wire(expensive)
    envelope = asyncio.run(p.dispatch(prepared(p)))
    with pytest.raises(ProviderResponseError, match="does not match") as caught:
        p.decode(envelope, receipt)
    assert caught.value.billing_result.cost_usd == pytest.approx(.02095)
    assert caught.value.envelope is envelope
    generation.return_value = wire(response(usage=None))
    unknown = asyncio.run(p.dispatch(prepared(p)))
    with pytest.raises(ProviderAccountingError) as caught:
        p.decode(unknown, receipt)
    assert caught.value.billing_result is None
    assert caught.value.receipt["cost_usd"] is None


def test_unrepresentable_legacy_float_keeps_exact_receipt_and_envelope():
    body = response()
    body["usage"]["input_tokens"] = 10 ** 320
    body["usage"]["total_tokens"] = 10 ** 320 + 40
    p, _, _ = provider(body)
    with pytest.raises(ProviderAccountingError, match="legacy cost ledger") as caught:
        asyncio.run(p.complete("S", "P", MODEL))
    assert Decimal(caught.value.receipt["cost_usd"]) > Decimal("1e308")
    assert caught.value.envelope.body == wire(body).content


def test_cancelled_generation_preserves_cancellation_and_unknown_accounting():
    p, generation, _ = provider()

    async def exercise():
        started = asyncio.Event()

        async def pending(**kwargs):
            started.set()
            await asyncio.Event().wait()

        generation.side_effect = pending
        task = asyncio.create_task(p.complete("S", "P", MODEL))
        await started.wait()
        task.cancel()
        with pytest.raises(ProviderAccountingCancelledError) as caught:
            await task
        assert isinstance(caught.value, asyncio.CancelledError)
        assert isinstance(caught.value, BudgetValidationError)
        assert caught.value.receipt["cost_usd"] is None
        assert caught.value.envelope.transport_error == "CancelledError"

    asyncio.run(exercise())
    assert generation.await_count == 1


def test_count_cancellation_does_not_claim_generation_was_dispatched():
    p, generation, counting = provider()
    counting.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(p.quote(prepared(p)))
    assert not isinstance(caught.value, ProviderAccountingCancelledError)
    generation.assert_not_awaited()


def test_default_client_explicitly_overrides_endpoint_environment(monkeypatch):
    import sys
    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(__version__="3.8.0", AsyncOpenAI=client))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    p = OpenAIResponsesProvider(api_key=KEY)
    p._get_client()
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["max_retries"] == 0


def test_real_sdk_raw_transport_contract(monkeypatch):
    """Runs in CI/openai extra and the isolated 3.8.0 contract environment."""
    openai = pytest.importorskip("openai", minversion="3.8.0")
    httpx2 = pytest.importorskip("httpx2")
    requests = []
    outputs = [
        {"object": "response.input_tokens", "input_tokens": 1000},
        response(output=tool_output()), response(),
    ]
    malformed = response()
    malformed["usage"]["input_tokens_details"]["cache_write_tokens"] = False
    outputs.append(malformed)

    def handle(request):
        assert request.url.host == "api.openai.com"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        requests.append((request.url.path, json.loads(request.content)))
        return httpx2.Response(200, json=outputs.pop(0), headers={"x-request-id": "req_sdk"})

    def no_network(*args, **kwargs):
        raise AssertionError("Real HTTP is disabled in native provider tests")

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", no_network)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://not-the-pricing-endpoint.invalid")

    async def exercise():
        async with openai.AsyncOpenAI(
            api_key=KEY, base_url="https://api.openai.com/v1", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False),
        ) as client:
            p = OpenAIResponsesProvider(api_key=KEY)
            p._client = client
            request = prepared(p, tools=[tool()])
            quote = await p.quote(request)
            assert quote.ceiling_usd == Decimal("0.4221")
            first = await p.chat("S", [ChatMessage("user", "P")], MODEL, [tool()])
            assert first.provider_continuation["output"] == tool_output()
            history = [ChatMessage("user", "P"), ChatMessage(
                "assistant", first.text, tool_calls=first.tool_calls,
                provider_continuation=first.provider_continuation,
            ), ChatMessage("user", tool_results=[ToolResult("call_tool_id", "Found.")])]
            assert (await p.chat("S", history, MODEL, [tool()])).text == "The answer."
            with pytest.raises(ProviderAccountingError) as caught:
                await p.complete("S", "P", MODEL)
            assert caught.value.envelope.json()["usage"]["input_tokens_details"]["cache_write_tokens"] is False

    asyncio.run(exercise())
    assert [path for path, _ in requests] == ["/v1/responses/input_tokens"] + ["/v1/responses"] * 3
    assert requests[2][1]["input"][1:4] == tool_output()
    assert requests[2][1]["input"][-1]["call_id"] == "call_tool_id"
    assert requests[1][1]["tools"][0]["strict"] is False
    assert not outputs


@pytest.mark.parametrize("pooled", [False, True])
def test_owned_real_sdk_clients_work_across_sync_loops_with_http_keepalive(monkeypatch, pooled):
    """Real HTTP pool, loopback server only; MockTransport cannot expose this bug."""
    openai = pytest.importorskip("openai", minversion="3.8.0")
    httpx2 = pytest.importorskip("httpx2")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    observed = []
    ports = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            ports.append(self.client_address[1])
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(response()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clients = []
    real_constructor = openai.AsyncOpenAI

    class LoopbackTransport(httpx2.AsyncBaseTransport):
        def __init__(self):
            self.pool = httpx2.AsyncHTTPTransport(trust_env=False)

        async def handle_async_request(self, request):
            assert request.url.host == "api.openai.com"
            local = httpx2.Request(
                request.method, f"http://127.0.0.1:{server.server_port}{request.url.path}",
                headers=request.headers, content=request.content,
            )
            return await self.pool.handle_async_request(local)

        async def aclose(self):
            await self.pool.aclose()

    def make_client(**kwargs):
        client = real_constructor(
            **kwargs, http_client=httpx2.AsyncClient(transport=LoopbackTransport(), trust_env=False),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)
    try:
        p = OpenAIResponsesProvider(api_key=KEY)
        if pooled:
            async def exercise():
                async with p.session() as bound:
                    for _ in range(3):
                        assert (await bound.complete("S", "P", MODEL)).text == "The answer."
                with pytest.raises(ValueError, match="active session loop"):
                    await bound.complete("S", "P", MODEL)
            asyncio.run(exercise())
            assert len(clients) == 1 and len(set(ports)) == 1
        else:
            for _ in range(3):
                assert asyncio.run(p.complete("S", "P", MODEL)).text == "The answer."
            assert len(clients) == 3
        assert all(client.is_closed() for client in clients)
        assert len(observed) == 3
        assert p._client is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
