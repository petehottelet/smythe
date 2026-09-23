"""Offline Fable pricing, exact evidence, dispatch and replay boundaries."""

import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe.pricing_anthropic import MODEL, messages_quote_nanousd, price_messages_response
from smythe.provider import ProviderAccountingError, ProviderAccountingCancelledError, ProviderResponseError
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import PreparedRequest, RawResponseEnvelope, _json_dump
from smythe.tools import ChatMessage
from smythe.workflow_provider import ProviderRequestRejectedError, WorkflowProviderContext, describe_provider
from smythe.workflow_store import (
    CallKey, SQLiteWorkflowStore, WorkflowBudgetError, WorkflowConflictError, WorkflowError,
)


def response(**changes):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": MODEL,
            "stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "Accepted output."}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
                      "cache_creation_input_tokens": 30,
                      "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20},
                      "service_tier": "standard"}, **changes}


def wire(raw):
    return SimpleNamespace(http_response=SimpleNamespace(
        content=raw if isinstance(raw, bytes) else json.dumps(raw).encode()),
        status_code=200, headers={"request-id": "req_test"})


class StatusError(Exception):
    """Mimics an SDK status error, which carries the raw HTTP response."""

    def __init__(self, status, error_type):
        super().__init__(f"Error code: {status}")
        body = {"type": "error", "error": {"type": error_type, "message": "Provider error"},
                "request_id": "req_status"}
        self.response = SimpleNamespace(content=json.dumps(body).encode(), status_code=status,
                                        headers={"request-id": "req_status"})


@pytest.fixture
def transport(monkeypatch):
    generation, count = AsyncMock(return_value=wire(response())), AsyncMock(return_value=wire({"input_tokens": 170}))
    client = SimpleNamespace(base_url="https://api.anthropic.com/", max_retries=0, close=AsyncMock(),
        messages=SimpleNamespace(with_raw_response=SimpleNamespace(create=generation, count_tokens=count)))
    monkeypatch.setattr(AnthropicMessagesProvider, "_get_client", lambda self: client)
    return SimpleNamespace(generation=generation, count=count, client=client)


@pytest.fixture
def journal(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "calls.sqlite3") as store:
        run = store.create_run({"goal": "test"}, {}, 1_000_000_000)
        lease = store.acquire_lease(run["run_id"], "tests", 120)
        yield store, lease


async def invoke(journal, *, prompt="Question", key=CallKey("execution", "node")):
    store, lease = journal
    async with WorkflowProviderContext(store, lease, {"main": AnthropicMessagesProvider(api_key="test-secret")}) as context:
        return await context.for_call("main", key).complete("System", prompt, MODEL)


def test_cache_buckets_and_thinking_are_billed_once():
    receipt = price_messages_response(response())
    assert receipt.cost_nanousd == 2_535_000
    assert receipt.input_tokens == 170 and receipt.output_tokens == 20
    assert receipt.safe_summary()["usage"]["thinking_included_in_output"]
    assert receipt.safe_summary()["categories"]["ordinary_input"]["tokens"] == 100


@pytest.mark.parametrize("field,value", [
    ("input_tokens", None), ("output_tokens", True), ("cache_read_input_tokens", -1),
    ("cache_creation_input_tokens", 31), ("cache_creation", None),
    ("service_tier", "priority"), ("service_tier", None), ("inference_geo", "us"),
    ("server_tool_use", {"web_search_requests": 1}),
])
def test_incomplete_or_unpriced_usage_fails_closed(field, value):
    raw = response()
    raw["usage"][field] = value
    result = price_messages_response(raw)
    assert result.cost_nanousd is None and result.accounting_error


def test_unknown_model_and_tools_not_given_token_only_bill():
    for raw in (response(model="claude-other"), response(content=[{"type": "server_tool_use"}]), {}):
        assert price_messages_response(raw).cost_nanousd is None


def test_zero_cache_creation_can_omit_ttl_detail():
    raw = response()
    raw["usage"]["cache_creation_input_tokens"] = 0
    del raw["usage"]["cache_creation"]
    assert price_messages_response(raw).cost_nanousd == 2_010_000


def test_quote_covers_output_cap_and_expensive_input_categories():
    assert messages_quote_nanousd(100, 8192) > 8192 * 50000 + 100 * 20000
    for count in (True, -1, "100", None):
        with pytest.raises(ValueError):
            messages_quote_nanousd(count, 8192)


def test_prepare_is_strict_detached_and_secret_free():
    provider = AnthropicMessagesProvider(api_key="test-secret")
    desc = describe_provider(provider)
    assert "test-secret" not in json.dumps(desc)
    p = provider.prepare("System", [ChatMessage(role="user", content="Test")], MODEL)
    assert p.payload["output_config"] == {"effort": "medium"}
    assert set(p.payload) == {"model", "system", "messages", "max_tokens", "service_tier", "output_config"}
    for key, value in (("tools", []), ("model", "claude-other"), ("max_tokens", True),
                       ("output_config", {"effort": "ultracode"}), ("service_tier", "auto")):
        payload = {**p.payload, key: value}
        with pytest.raises(ValueError):
            provider._validate_prepared(PreparedRequest(_json_dump(payload)))
    with pytest.raises(ValueError, match="prefill"):
        provider.prepare("System", [ChatMessage(role="assistant", content="Answer")], MODEL)


def test_saved_bytes_settled_before_decode_and_no_paid_replay(journal, transport, monkeypatch):
    store, lease = journal
    original = AnthropicMessagesProvider.decode

    def decode(envelope):
        call = store.lookup_call(lease.run_id, CallKey("execution", "node"))
        assert call["billing_state"] == "known"
        assert store.load_replay(call["call_id"])["evidence"]["body"] == envelope.body
        return original(envelope)

    monkeypatch.setattr(AnthropicMessagesProvider, "decode", staticmethod(decode))
    first = asyncio.run(invoke(journal))
    monkeypatch.setattr(AnthropicMessagesProvider, "_get_client", lambda _: pytest.fail("SDK during replay"))
    again = asyncio.run(invoke(journal))
    assert first.text == again.text == "Accepted output."
    assert again.native_receipt["workflow_replayed"]
    assert transport.generation.await_count == transport.count.await_count == 1
    assert store.inspect_run(lease.run_id)["confirmed_nanousd"] == 2_535_000
    with pytest.raises(WorkflowConflictError):
        asyncio.run(invoke(journal, prompt="Changed"))


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "pause_turn"])
def test_truncations_refusals_keep_their_known_charge(journal, transport, reason):
    transport.generation.return_value = wire(response(stop_reason=reason))
    with pytest.raises(ProviderResponseError) as error:
        asyncio.run(invoke(journal))
    assert error.value.billing_result.cost_usd == .002535
    assert journal[0].inspect_run(journal[1].run_id)["confirmed_nanousd"] == 2_535_000


@pytest.mark.parametrize("failure", [ConnectionError("offline"), asyncio.CancelledError(), b"malformed"])
def test_unknown_dispatch_keeps_reserve_and_blocks_next_call(journal, transport, failure):
    if isinstance(failure, BaseException):
        transport.generation.side_effect = failure
    else:
        transport.generation.return_value = wire(failure)
    with pytest.raises((ProviderAccountingError, ProviderAccountingCancelledError)):
        asyncio.run(invoke(journal))
    audit = journal[0].inspect_run(journal[1].run_id)
    assert audit["unknown_nanousd"] > 0 and audit["unknown_calls"]
    with pytest.raises(Exception):
        asyncio.run(invoke(journal, key=CallKey("execution", "next")))
    assert transport.generation.await_count == 1


def test_rate_limit_is_zero_cost_rejection_and_next_attempt_is_admitted(journal, transport):
    store, lease = journal
    transport.generation.side_effect = [StatusError(429, "rate_limit_error"), wire(response())]
    with pytest.raises(ProviderRequestRejectedError) as caught:
        asyncio.run(invoke(journal))
    assert not isinstance(caught.value, (ProviderResponseError, WorkflowError))
    assert caught.value.status_code == 429 and "rate_limit_error" in str(caught.value)
    assert caught.value.envelope.request_id == "req_status"
    record = store.lookup_call(lease.run_id, CallKey("execution", "node"))
    assert (record["billing_state"], record["result_state"], record["cost_nanousd"]) == ("known", "rejected", 0)
    assert store.load_run(lease.run_id)["blocked_reason"] is None
    retried = asyncio.run(invoke(journal, key=CallKey("execution", "node", attempt=1)))
    assert retried.text == "Accepted output."
    audit = store.inspect_run(lease.run_id)
    assert audit["confirmed_nanousd"] == sum(call["cost_nanousd"] for call in audit["calls"]) == 2_535_000
    assert audit["unknown_calls"] == audit["reserved_nanousd"] == 0
    assert transport.generation.await_count == 2


def test_durable_swarm_retries_rate_limited_node_with_exact_totals(tmp_path, transport):
    from smythe import Swarm
    from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology

    transport.generation.side_effect = [StatusError(429, "rate_limit_error"), wire(response())]
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY, max_retries=1)])
    with SQLiteWorkflowStore(tmp_path / "fable-retry.sqlite3") as store:
        swarm = Swarm(model=MODEL, provider=AnthropicMessagesProvider(api_key="test-secret"),
                      run_store=store, max_budget_usd=1)
        result = swarm.execute(graph)
        assert "Accepted output." in result.output
        assert result.total_cost_usd == .002535 and result.cost_is_complete
        accounting = store.inspect_run(result.execution_id)
        assert sorted((call["key"]["attempt"], call["cost_nanousd"], call["result_state"])
                      for call in accounting["calls"]) == [(0, 0, "rejected"), (1, 2_535_000, "applied")]
        assert accounting["confirmed_nanousd"] == 2_535_000 and store.audit(result.execution_id)["ok"]
    assert transport.generation.await_count == 2


@pytest.mark.parametrize("status,error_type", [(529, "overloaded_error"), (500, "api_error")])
def test_server_errors_including_overloaded_keep_unknown_exposure(journal, transport, status, error_type):
    transport.generation.side_effect = StatusError(status, error_type)
    with pytest.raises(ProviderAccountingError):
        asyncio.run(invoke(journal))
    audit = journal[0].inspect_run(journal[1].run_id)
    assert audit["unknown_calls"] == 1 and audit["unknown_nanousd"] > 0
    assert audit["blocked_reason"] == "unknown_exposure"
    with pytest.raises(WorkflowError):
        asyncio.run(invoke(journal, key=CallKey("execution", "node", attempt=1)))
    assert transport.generation.await_count == 1


def test_concurrent_duplicate_call_has_one_dispatch(journal, transport):
    async def duplicate():
        store, lease = journal
        async with WorkflowProviderContext(store, lease, {"p": AnthropicMessagesProvider()}) as context:
            call = context.for_call("p", CallKey("execution", "same"))
            return await asyncio.gather(*(call.complete("System", "Question", MODEL) for _ in range(3)))
    results = asyncio.run(duplicate())
    assert all(result.text == "Accepted output." for result in results)
    assert transport.generation.await_count == 1


def test_too_large_quote_precedes_generation(journal, transport):
    transport.count.return_value = wire({"input_tokens": 100000})
    with pytest.raises(WorkflowBudgetError):
        asyncio.run(invoke(journal))
    assert transport.generation.await_count == 0


def test_http_error_cannot_decode_as_known_success():
    p = AnthropicMessagesProvider().prepare("System", [ChatMessage(role="user", content="Question")], MODEL)
    envelope = RawResponseEnvelope(p, json.dumps(response()).encode(), status_code=200)
    for changed in (replace(envelope, status_code=500), replace(envelope, transport_error="Timeout")):
        with pytest.raises(ProviderAccountingError):
            AnthropicMessagesProvider.decode(changed)


def test_real_sdk_raw_response_contract_without_network():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    paths = []

    def handle(request):
        paths.append(request.url.path)
        body = {"input_tokens": 170} if request.url.path.endswith("count_tokens") else response()
        return httpx.Response(200, json=body, headers={"request-id": "req_sdk"})

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            provider = AnthropicMessagesProvider(api_key="not-real")
            provider._client = anthropic.AsyncAnthropic(api_key="not-real", max_retries=0,
                base_url="https://api.anthropic.com", http_client=http)
            prepared = provider.prepare("System", [ChatMessage(role="user", content="Question")], MODEL)
            quote = await provider.quote(prepared)
            raw = await provider.dispatch(prepared)
            assert quote.input_tokens == 170
            assert provider.decode(raw).text == "Accepted output."
            assert raw.request_id == "req_sdk" and raw.json() == response()

    asyncio.run(check())
    assert paths == ["/v1/messages/count_tokens", "/v1/messages"]
