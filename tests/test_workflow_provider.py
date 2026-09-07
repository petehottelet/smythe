"""No-network provider journaling, replay, and dispatch crash boundaries."""

import asyncio
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe.provider import (
    OfflineProvider, OpenAIProvider, ProviderAccountingCancelledError,
    ProviderAccountingError, ProviderResponseError,
)
from smythe.provider_responses import OpenAIResponsesProvider, RawResponseEnvelope, ResponseQuoteError
from smythe.tools import ChatMessage, ToolSpec
from smythe.workflow_provider import (
    WorkflowProviderContext, describe_provider, snapshot_provider, validate_workflow_model,
)
from smythe.workflow_store import (
    CallKey, SQLiteWorkflowStore, WorkflowBudgetError, WorkflowConflictError, WorkflowError,
    WorkflowLeaseError,
)

MODEL = "gpt-6-astra"
KEY = CallKey("execution", "node-a")


def response(**changes):
    return {
        "id": "resp_workflow", "model": MODEL, "status": "completed", "service_tier": "default",
        "usage": {"input_tokens": 100, "output_tokens": 10,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
        "output": [{"type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Accepted output."}]}],
        **changes,
    }


def wire(body):
    return SimpleNamespace(content=body if isinstance(body, bytes) else json.dumps(body).encode(),
                           status_code=200, headers={"x-request-id": "req_workflow"})


@pytest.fixture
def journal(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "workflow.db")
    run = store.create_run({"goal": "test"}, {}, 1_000_000_000)
    lease = store.acquire_lease(run["run_id"], "test-owner", 120)
    yield store, lease
    store.close()


@pytest.fixture
def transport(monkeypatch):
    generation = AsyncMock(return_value=wire(response()))
    count = AsyncMock(return_value=wire({"input_tokens": 100}))
    clients = []

    def create_client(self):
        if self._client is not None:
            return self._client
        client = SimpleNamespace(
            base_url="https://api.openai.com/v1/", max_retries=0, close=AsyncMock(),
            responses=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=generation),
                input_tokens=SimpleNamespace(with_raw_response=SimpleNamespace(count=count)),
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", create_client)
    return SimpleNamespace(generation=generation, count=count, clients=clients)


def native():
    return OpenAIResponsesProvider(api_key="test-secret-do-not-persist", max_output_tokens=100)


async def invoke(journal, source=None, *, key=KEY, prompt="Question", model=MODEL):
    store, lease = journal
    async with WorkflowProviderContext(store, lease, {"main": source or native()}) as context:
        return await context.for_call("main", key).complete("System", prompt, model)


def test_descriptor_snapshot_is_local_detached_and_secret_free(monkeypatch):
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK created"))
    source = native()
    descriptor = describe_provider(source)
    assert descriptor["config"]["max_output_tokens"] == 100
    assert "secret" not in json.dumps(descriptor) and "api_key" not in json.dumps(descriptor)
    clone = snapshot_provider(source)
    assert clone is not source and describe_provider(clone) == descriptor
    descriptor["config"]["max_output_tokens"] = 5
    assert describe_provider(clone)["config"]["max_output_tokens"] == 100
    plan = {"nodes": [{"label": "First"}]}
    offline = snapshot_provider(OfflineProvider(plan=plan))
    plan["nodes"][0]["label"] = "Changed"
    assert describe_provider(offline)["config"]["plan"]["nodes"][0]["label"] == "First"


@pytest.mark.parametrize("source", [
    OpenAIProvider(api_key="not-real"), OfflineProvider(responses=["script"]),
    OfflineProvider(artifacts_per_call=1), OfflineProvider(plan=[1]),
    OfflineProvider(echo_prefix=object()),
])
def test_preflight_rejects_unsupported_modes(source):
    with pytest.raises(ValueError):
        describe_provider(source)


def test_preflight_rejects_subclasses_and_unsupported_models():
    class Custom(OfflineProvider):
        pass

    with pytest.raises(ValueError):
        describe_provider(Custom())
    for model in ("gpt-4", "", None, False):
        with pytest.raises(ValueError):
            validate_workflow_model(native(), model)
    validate_workflow_model(native(), MODEL)
    validate_workflow_model(OfflineProvider(), "arbitrary-offline-label")


def test_native_call_is_reserved_and_raw_saved_before_decode(journal, transport, monkeypatch):
    store, lease = journal
    original = OpenAIResponsesProvider.decode

    def check_decode(envelope, receipt=None):
        record = store.lookup_call(lease.run_id, KEY)
        assert record["state"] == "settled" and record["billing_state"] == "known"
        assert store.load_replay(record["call_id"])["evidence"]["body"] == envelope.body
        return original(envelope, receipt)

    monkeypatch.setattr(OpenAIResponsesProvider, "decode", staticmethod(check_decode))
    result = asyncio.run(invoke(journal))
    assert result.text == "Accepted output." and result.cost_usd == .0015
    assert result.native_receipt["workflow_charge_recorded"] is True
    assert result.native_receipt["workflow_replayed"] is False
    assert result.native_receipt["workflow_run_id"] == lease.run_id
    record = store.lookup_call(lease.run_id, KEY)
    assert record["cost_nanousd"] == 1_500_000 and record["result_state"] == "accepted"
    assert "provider_continuation" not in record["decoded_result"]
    assert "response_envelope" not in record["decoded_result"]
    assert transport.count.await_count == transport.generation.await_count == 1
    assert len(transport.clients) == 1 and transport.clients[0].close.await_count == 1


def test_accepted_replay_no_sdk_no_count_no_generation_and_detached(journal, transport, monkeypatch):
    result = asyncio.run(invoke(journal))
    result.text = "Caller mutated output"
    result.native_receipt["cost_nanousd"] = 0
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on replay"))
    again = asyncio.run(invoke(journal))
    assert again.text == "Accepted output."
    assert again.native_receipt["workflow_replayed"] is True
    assert again.native_receipt["cost_nanousd"] == 1_500_000
    assert transport.count.await_count == transport.generation.await_count == 1


@pytest.mark.parametrize("change", ["prompt", "model", "configuration"])
def test_changed_binding_under_same_key_rejected_without_http(journal, transport, change):
    asyncio.run(invoke(journal))
    kwargs = ({"prompt": "Different"} if change == "prompt" else
              {"model": "gpt-5.6-sol"} if change == "model" else
              {"source": OpenAIResponsesProvider(max_output_tokens=101)})
    with pytest.raises(WorkflowConflictError):
        asyncio.run(invoke(journal, **kwargs))
    assert transport.count.await_count == transport.generation.await_count == 1


def test_parallel_same_key_dispatches_once_and_two_calls_share_pool(journal, transport):
    async def run():
        async with WorkflowProviderContext(*journal, {"main": native()}) as context:
            bound = context.for_call("main", KEY)
            first, duplicate = await asyncio.gather(
                bound.complete("System", "Question", MODEL),
                bound.complete("System", "Question", MODEL),
            )
            next_result = await context.for_call("main", replace(KEY, attempt=1)).complete(
                "System", "Repair", MODEL,
            )
            return first, duplicate, next_result

    first, duplicate, next_result = asyncio.run(run())
    assert first.native_receipt["workflow_replayed"] is False
    assert duplicate.native_receipt["workflow_replayed"] is True
    assert next_result.native_receipt["workflow_call_id"] != first.native_receipt["workflow_call_id"]
    assert transport.generation.await_count == transport.count.await_count == 2
    assert len(transport.clients) == 1 and transport.clients[0].close.await_count == 1


def test_offline_zero_cost_static_plan_and_echo_no_network(journal, monkeypatch):
    from smythe.prompts import PLANNING_SYSTEM_PROMPT

    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("offline SDK"))
    plan = {"nodes": [{"label": "A"}]}
    source = OfflineProvider(plan=plan)

    async def run():
        async with WorkflowProviderContext(*journal, {"main": source}) as context:
            a = await context.for_call("main", replace(KEY, phase="planning")).complete(
                PLANNING_SYSTEM_PROMPT, "Goal", MODEL,
            )
            b = await context.for_call("main", KEY).complete("System", "Question", MODEL)
            return a, b

    planning, execution = asyncio.run(run())
    assert json.loads(planning.text) == plan and execution.text == "offline: Question"
    for result in (planning, execution):
        assert result.cost_usd == 0.0 and result.total_tokens == 0
        assert result.native_receipt["workflow_charge_recorded"] is True
    assert source.calls == []  # Each run owns its detached provider.
    assert asyncio.run(invoke(journal, source, prompt="Question")).native_receipt["workflow_replayed"]


@pytest.mark.parametrize("body", [b"{broken", response(usage=None), response(usage={})])
def test_unknown_billing_keeps_raw_and_never_redispatches(journal, transport, body):
    transport.generation.return_value = wire(body)
    for _ in range(2):
        with pytest.raises(ProviderAccountingError) as caught:
            asyncio.run(invoke(journal))
        assert caught.value.envelope.body == wire(body).content
        assert caught.value.receipt["workflow_charge_recorded"] is True
    store, lease = journal
    record = store.lookup_call(lease.run_id, KEY)
    assert record["billing_state"] == "unknown"
    assert store.load_replay(record["call_id"])["evidence"]["body"] == wire(body).content
    assert transport.generation.await_count == transport.count.await_count == 1


@pytest.mark.parametrize("changes", [{"status": "incomplete"}, {"output": []}, {"id": None}])
def test_known_bad_output_is_paid_terminal_and_replayed_locally(journal, transport, changes):
    transport.generation.return_value = wire(response(**changes))
    for _ in range(2):
        with pytest.raises(ProviderResponseError) as caught:
            asyncio.run(invoke(journal))
        assert not isinstance(caught.value, ProviderAccountingError)
        assert caught.value.billing_result.cost_usd == .0015
    record = journal[0].lookup_call(journal[1].run_id, KEY)
    assert record["billing_state"] == "known" and record["result_state"] == "rejected"
    assert record["cost_nanousd"] == 1_500_000
    assert transport.generation.await_count == 1


def test_generation_cancellation_is_journaled_unknown_and_not_retried(journal, transport):
    async def cancel(**kwargs):
        raise asyncio.CancelledError()

    transport.generation.side_effect = cancel
    with pytest.raises(ProviderAccountingCancelledError) as caught:
        asyncio.run(invoke(journal))
    assert caught.value.envelope.transport_error == "CancelledError"
    assert caught.value.receipt["workflow_charge_recorded"] is True
    record = journal[0].lookup_call(journal[1].run_id, KEY)
    assert record["billing_state"] == "unknown"
    with pytest.raises(ProviderAccountingError):
        asyncio.run(invoke(journal))
    assert transport.generation.await_count == 1


def test_quote_malformed_or_cancelled_never_dispatches(journal, transport):
    transport.count.return_value = wire({"input_tokens": False})
    with pytest.raises(ResponseQuoteError) as caught:
        asyncio.run(invoke(journal))
    assert isinstance(caught.value, WorkflowError)
    assert caught.value.envelope.json() == {"input_tokens": False}
    record = journal[0].lookup_call(journal[1].run_id, KEY)
    assert record["state"] == "prepared" and record["ceiling_nanousd"] is None
    transport.count.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(invoke(journal))
    assert transport.generation.await_count == 0


def test_insufficient_budget_quotes_without_generation(tmp_path, transport):
    with SQLiteWorkflowStore(tmp_path / "small.db") as store:
        run = store.create_run(None, {}, 1)
        lease = store.acquire_lease(run["run_id"], "owner")
        with pytest.raises(WorkflowBudgetError):
            asyncio.run(invoke((store, lease)))
    assert transport.count.await_count == 1 and transport.generation.await_count == 0


def test_late_response_after_lease_loss_is_saved_without_acceptance(tmp_path, transport):
    clock = [1_000_000_000]
    with SQLiteWorkflowStore(tmp_path / "late.db", clock_ns=lambda: clock[0]) as store:
        run = store.create_run(None, {}, 1_000_000_000)
        lease = store.acquire_lease(run["run_id"], "old", 1)

        async def lose_lease(**kwargs):
            clock[0] += 2_000_000_000
            store.acquire_lease(run["run_id"], "new", 30)
            return wire(response())

        transport.generation.side_effect = lose_lease
        with pytest.raises(ProviderAccountingError) as caught:
            asyncio.run(invoke((store, lease)))
        assert isinstance(caught.value.__cause__, WorkflowLeaseError)
        record = store.lookup_call(run["run_id"], KEY)
        assert record["result_state"] == "pending"
        assert store.load_replay(record["call_id"])["evidence"]["body"] == wire(response()).content


def test_context_and_bound_key_are_scoped_and_frozen(journal):
    context = WorkflowProviderContext(*journal, {"main": OfflineProvider()})
    bound = context.for_call("main", KEY)
    with pytest.raises(FrozenInstanceError):
        bound.key = replace(KEY, turn=1)
    with pytest.raises(RuntimeError):
        asyncio.run(bound.complete("System", "Question", MODEL))
    with pytest.raises(WorkflowLeaseError):
        context.update_lease(replace(journal[1], epoch=journal[1].epoch + 1))

    async def run():
        async with context:
            await bound.complete("System", "Question", MODEL)
        with pytest.raises(RuntimeError):
            await bound.complete("System", "Question", MODEL)

    asyncio.run(run())


def test_unsupported_input_rejected_before_journal_or_http(journal, transport):
    async def run():
        async with WorkflowProviderContext(*journal, {"main": native()}) as context:
            bound = context.for_call("main", KEY)
            with pytest.raises(ValueError):
                await bound.chat("System", [ChatMessage("user", "Question")], MODEL,
                                 [ToolSpec("f", "Function", {"type": "object"})])
            with pytest.raises(ValueError):
                await bound.chat("System", [ChatMessage("assistant", "Question",
                                 provider_continuation={"output": []})], MODEL)

    asyncio.run(run())
    assert journal[0].lookup_call(journal[1].run_id, KEY) is None
    assert transport.clients == []


@pytest.mark.parametrize("boundary", ["settle_call", "accept_result"])
def test_saved_response_recovers_locally_after_journal_crash(journal, transport, monkeypatch, boundary):
    store, lease = journal
    original = getattr(store, boundary)

    def crash(*args, **kwargs):
        raise OSError("Simulated journal boundary interruption")

    monkeypatch.setattr(store, boundary, crash)
    with pytest.raises(ProviderAccountingError) as caught:
        asyncio.run(invoke(journal))
    assert caught.value.envelope.body == wire(response()).content
    record = store.lookup_call(lease.run_id, KEY)
    assert record["evidence_id"] is not None and record["result_state"] == "pending"
    monkeypatch.setattr(store, boundary, original)
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on recovery"))
    recovered = asyncio.run(invoke(journal))
    assert recovered.text == "Accepted output." and recovered.native_receipt["workflow_replayed"]
    assert store.lookup_call(lease.run_id, KEY)["cost_nanousd"] == 1_500_000
    assert transport.count.await_count == transport.generation.await_count == 1


def test_reserved_not_dispatched_call_resumes_without_recount(journal, transport, monkeypatch):
    store, lease = journal
    claim = store.claim_dispatch
    monkeypatch.setattr(store, "claim_dispatch", lambda *args: (_ for _ in ()).throw(OSError("Crash")))
    with pytest.raises(OSError):
        asyncio.run(invoke(journal))
    assert store.lookup_call(lease.run_id, KEY)["state"] == "reserved"
    assert transport.count.await_count == 1 and transport.generation.await_count == 0
    monkeypatch.setattr(store, "claim_dispatch", claim)
    assert asyncio.run(invoke(journal)).text == "Accepted output."
    assert transport.count.await_count == transport.generation.await_count == 1


def test_claim_without_response_is_never_resent(journal, transport, monkeypatch):
    store, lease = journal
    append = store.append_response
    monkeypatch.setattr(store, "append_response", lambda *args: (_ for _ in ()).throw(OSError("Disk")))
    with pytest.raises(ProviderAccountingError) as caught:
        asyncio.run(invoke(journal))
    assert caught.value.envelope.body == wire(response()).content
    record = store.lookup_call(lease.run_id, KEY)
    assert record["state"] == "dispatched" and record["evidence_id"] is None
    monkeypatch.setattr(store, "append_response", append)
    with pytest.raises(ProviderAccountingError, match="cannot be resent"):
        asyncio.run(invoke(journal))
    assert transport.generation.await_count == 1


def test_overrun_commits_full_charge_and_never_decodes_text(tmp_path, transport, monkeypatch):
    expensive = response(usage={"input_tokens": 100, "output_tokens": 1000,
                               "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}})
    transport.generation.return_value = wire(expensive)
    monkeypatch.setattr(OpenAIResponsesProvider, "decode", lambda *args: pytest.fail("Decoded overrun"))
    with SQLiteWorkflowStore(tmp_path / "overrun.db") as store:
        run = store.create_run(None, {}, 10_000_000)
        lease = store.acquire_lease(run["run_id"], "owner")
        with pytest.raises(ProviderResponseError) as caught:
            asyncio.run(invoke((store, lease)))
        assert caught.value.billing_result.cost_usd == .051
        record = store.lookup_call(run["run_id"], KEY)
        assert record["cost_nanousd"] == 51_000_000 and record["result_state"] == "pending"
        assert store.load_run(run["run_id"])["blocked_reason"] == "budget_overrun"
        assert store.load_replay(record["call_id"])["evidence"]["body"] == wire(expensive).content


def test_cancelled_dispatch_preserves_cancellation_when_lease_is_lost(tmp_path, transport):
    clock = [1_000_000_000]
    with SQLiteWorkflowStore(tmp_path / "cancel-late.db", clock_ns=lambda: clock[0]) as store:
        run = store.create_run(None, {}, 1_000_000_000)
        lease = store.acquire_lease(run["run_id"], "old", 1)

        async def cancel(**kwargs):
            clock[0] += 2_000_000_000
            store.acquire_lease(run["run_id"], "new", 30)
            raise asyncio.CancelledError()

        transport.generation.side_effect = cancel
        with pytest.raises(ProviderAccountingCancelledError) as caught:
            asyncio.run(invoke((store, lease)))
        assert caught.value.envelope.transport_error == "CancelledError"
        record = store.lookup_call(run["run_id"], KEY)
        assert store.load_replay(record["call_id"])["evidence"] is not None


def test_real_sdk_journal_counts_then_dispatches_once_and_replays(journal, monkeypatch):
    openai = pytest.importorskip("openai", minversion="3.8.0")
    httpx2 = pytest.importorskip("httpx2")
    requests, clients = [], []
    original_client = openai.AsyncOpenAI

    def handle(request):
        assert request.url.host == "api.openai.com"
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        body = {"input_tokens": 100} if request.url.path.endswith("input_tokens") else response()
        return httpx2.Response(200, json=body, headers={"x-request-id": "req_real_sdk"})

    def client(**kwargs):
        created = original_client(**kwargs, http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(handle), trust_env=False,
        ))
        clients.append(created)
        return created

    def no_network(*args, **kwargs):
        raise AssertionError("Network is disabled in workflow provider tests")

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", no_network)
    initial = asyncio.run(invoke(journal))
    replayed = asyncio.run(invoke(journal))
    assert initial.text == replayed.text == "Accepted output."
    assert replayed.native_receipt["workflow_replayed"]
    assert [item[0] for item in requests] == ["/v1/responses/input_tokens", "/v1/responses"]
    assert requests[1][1]["model"] == MODEL and requests[1][1]["max_output_tokens"] == 100
    record = journal[0].lookup_call(journal[1].run_id, KEY)
    assert json.loads(record["request_json"]) == requests[1][1]
    assert len(clients) == 1 and clients[0].is_closed()


def test_other_unknown_call_does_not_permanently_reject_saved_usable_output(
    journal, transport, monkeypatch,
):
    store, lease = journal
    source = native()

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def delayed(**kwargs):
            started.set()
            await finish.wait()
            return wire(response())

        transport.generation.side_effect = delayed
        async with WorkflowProviderContext(store, lease, {"main": source}) as context:
            task = asyncio.create_task(context.for_call("main", KEY).complete("System", "Question", MODEL))
            await started.wait()
            request = source.prepare("System", [ChatMessage("user", "Other")], MODEL)
            descriptor = describe_provider(source)
            other = store.prepare_call(
                lease, replace(KEY, scope_id="other"), request_json=request.payload_json,
                provider=descriptor, price_version=descriptor["price_version"],
            )
            quote_id = store.append_quote_evidence(lease, other["call_id"], RawResponseEnvelope(
                request, b'{"input_tokens":100}', "input_tokens", 200,
            ))
            store.accept_quote(lease, other["call_id"], quote_id)
            store.reserve_call(lease, other["call_id"], quote_id)
            permit = store.claim_dispatch(lease, other["call_id"], request.request_sha256)
            store.mark_unknown(lease, other["call_id"], "Dispatch response not yet saved")
            finish.set()
            with pytest.raises(ProviderResponseError, match="admission closed"):
                await task
            saved = store.lookup_call(lease.run_id, KEY)
            assert saved["billing_state"] == "known" and saved["result_state"] == "pending"
            evidence_id = store.append_response(permit, RawResponseEnvelope(
                request, wire(response()).content, status_code=200,
            ))
            assert not store.settle_call(lease, other["call_id"], evidence_id)["admission_closed"]
        monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on recovery"))
        return await invoke(journal)

    assert asyncio.run(run()).text == "Accepted output."
    assert transport.generation.await_count == 1
