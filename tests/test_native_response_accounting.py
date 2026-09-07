"""Native evidence must survive unusable output without another paid attempt."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import json

import pytest

from smythe import ProviderAccountingCancelledError, ProviderAccountingError, ProviderResponseError
from smythe.async_executor import AsyncExecutor
from smythe.budget import BudgetReconciliationError, BudgetValidationError, Sentinel, SentinelAlert
from smythe.checkpoint import FileCheckpointStore, graph_to_dict
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.mcp import MCPToolSession, _DispatchEntry
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.swarm import Swarm
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.tools import ChatMessage, ToolCall, ToolRuntime
from smythe.tracer import Tracer
from test_tool_loop import SimpleRuntime


SECRET = "encrypted-reasoning-and-raw-body-must-stay-private"
SAFE = {"response_id": "resp_test", "cost_nanousd": 100_000_000, "cost_is_complete": True}


def failed_response(*, unknown=False, cost=0.1):
    return (ProviderAccountingError if unknown else ProviderResponseError)(
        "Native response cannot be consumed", envelope=SimpleNamespace(body=SECRET.encode()),
        receipt={**SAFE, "cost_is_complete": not unknown},
        billing_result=None if unknown else CompletionResult("", cost_usd=cost),
    )


class Script(Provider):
    def __init__(self, *responses, ceiling=0.2):
        self.responses = list(responses)
        self.ceiling = ceiling
        self.calls = []

    def budget_estimate_usd(self, model):
        return self.ceiling

    def requires_explicit_budget_estimate(self, model):
        return True

    async def complete(self, system, prompt, model):
        return await self.chat(system, [ChatMessage("user", prompt)], model)

    async def chat(self, system, messages, model, tools=None):
        self.calls.append((system, list(messages), tools))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def graph_for(policy=FailurePolicy.RETRY, **node_kwargs):
    node = Node(id="a", label="A", failure_policy=policy, max_retries=3, **node_kwargs)
    sibling = Node(id="b", label="B", depends_on=["a"])
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[node, sibling])


def execute(graph, provider, budget, *, parallel=False, tracer=None, **kwargs):
    kind = AsyncExecutor if parallel else Executor
    executor = kind(provider=provider, registry=Registry(), tracer=tracer or Tracer(),
                    budget=budget, **kwargs)
    if parallel:
        asyncio.run(executor.run(graph))
    else:
        executor.run(graph)
    return executor


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("policy", list(FailurePolicy))
@pytest.mark.parametrize("unknown", [False, True])
def test_native_failure_is_terminal_and_preserves_exposure(parallel, policy, unknown):
    error = failed_response(unknown=unknown)
    provider, budget, graph, tracer = Script(error), Sentinel(1), graph_for(policy), Tracer()
    with pytest.raises(type(error)) as caught:
        execute(graph, provider, budget, parallel=parallel, tracer=tracer)
    assert caught.value is error
    assert caught.value.envelope.body == SECRET.encode()
    assert len(provider.calls) == 1
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert graph.nodes[1].status is NodeStatus.PENDING
    assert budget.total_cost_usd == pytest.approx(0.2 if unknown else 0.1)
    assert budget.breakdown() == ({} if unknown else {"a": 0.1})
    assert bool(graph.nodes[0].metadata.get("accounting_invalid")) is unknown
    assert budget.cost_is_complete is not unknown
    assert bool(graph.nodes[0].metadata.get("cost_usd_unknown")) is unknown
    receipts = graph.nodes[0].metadata["native_receipts"]
    assert len(receipts) == 1 and receipts[0]["phase"] == "execution"
    assert SECRET not in json.dumps(graph_to_dict(graph))
    assert SECRET not in json.dumps(tracer.summary())
    if unknown:
        with pytest.raises(SentinelAlert):
            budget.reserve("probe", 0.9)
    with pytest.raises(ProviderResponseError):
        execute(graph, provider, budget, parallel=parallel)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("parallel", [False, True])
def test_reconciliation_overrun_preserves_original_evidence_and_known_bill(parallel):
    error = failed_response(cost=0.3)
    graph, provider, budget = graph_for(), Script(error), Sentinel(1)
    with pytest.raises(BudgetReconciliationError) as caught:
        execute(graph, provider, budget, parallel=parallel)
    assert caught.value.__cause__ is error
    assert budget.breakdown() == {"a": 0.3}
    assert graph.nodes[0].metadata["cost_usd"] == 0.3
    assert len(graph.nodes[0].metadata["native_receipts"]) == 1
    assert graph.nodes[0].metadata["response_error"]
    assert len(provider.calls) == 1


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("stop", ["pause_turn", "tool_use"])
def test_both_assistant_branches_preserve_continuation_without_trace_leak(parallel, stop):
    continuation = {"provider": "openai_responses", "model": "gpt-6-astra", "output": [
        {"type": "reasoning", "encrypted_content": SECRET},
        {"type": "function_call", "call_id": "call_1", "name": "calc__add", "arguments": "{}"},
    ]}
    first = CompletionResult(
        "", stop_reason=stop, provider_continuation=continuation,
        response_envelope=SimpleNamespace(body=SECRET.encode()), native_receipt=SAFE,
        tool_calls=[ToolCall("call_1", "calc.add", {})] if stop == "tool_use" else [],
        cost_usd=0.1,
    )
    provider = Script(first, CompletionResult("answer", cost_usd=0.1, native_receipt=SAFE))
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A")])
    tracer, budget = Tracer(), Sentinel(1)
    execute(graph, provider, budget, parallel=parallel, tracer=tracer, tool_runtime=SimpleRuntime())
    assistant = provider.calls[1][1][1]
    assert assistant.provider_continuation is continuation
    assert assistant.provider_continuation["output"][0]["encrypted_content"] == SECRET
    assert budget.breakdown() == {"a": 0.2}
    assert len(graph.nodes[0].metadata["native_receipts"]) == 2
    assert SECRET not in repr(first) + repr(assistant)
    assert SECRET not in json.dumps(graph_to_dict(graph)) + json.dumps(tracer.summary())


@pytest.mark.parametrize("parallel", [False, True])
def test_inclusive_ceiling_allows_exact_fit_first_call_and_blocks_second(parallel):
    provider = Script(CompletionResult("partial", stop_reason="pause_turn", cost_usd=0.1))
    budget, graph = Sentinel(0.2), graph_for()
    with pytest.raises(SentinelAlert):
        execute(graph, provider, budget, parallel=parallel, tool_runtime=SimpleRuntime(tools=[]))
    assert len(provider.calls) == 1
    assert budget.breakdown() == {"a": 0.1}
    assert budget.total_cost_usd == 0.1


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_checkpoint_blocks_native_failure_even_with_cached_completed_output(tmp_path, parallel, unknown):
    provider, store = Script(failed_response(unknown=unknown)), FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, parallel=parallel, max_budget_usd=1, checkpoint_store=store)
    with pytest.raises(ProviderResponseError):
        swarm.execute(graph_for())
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    assert state["control"]["response_error"]
    assert SECRET not in json.dumps(state)
    state["status"], state["output"] = "completed", "stale cache"
    store.save(execution_id, state)
    with pytest.raises((ProviderResponseError, BudgetValidationError)):
        swarm.resume(execution_id)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("marker", [{}, None, "", False, 0])
@pytest.mark.parametrize("location", ["control", "node"])
def test_empty_native_error_markers_cannot_enable_cached_resume(tmp_path, marker, location):
    provider, store = Script(failed_response()), FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, checkpoint_store=store)
    with pytest.raises(ProviderResponseError):
        swarm.execute(graph_for())
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    del state["control"]["response_error"]
    del state["graph"]["nodes"][0]["metadata"]["response_error"]
    container = state["control"] if location == "control" else state["graph"]["nodes"][0]["metadata"]
    container["response_error"] = marker
    state["status"], state["output"] = "completed", "STALE"
    store.save(execution_id, state)
    with pytest.raises(ProviderResponseError):
        swarm.resume(execution_id)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("marker", [{}, None, "", False, 0])
def test_empty_native_marker_blocks_direct_executor(parallel, marker):
    graph, provider = graph_for(), Script()
    graph.nodes[0].metadata["response_error"] = marker
    with pytest.raises(ProviderResponseError):
        execute(graph, provider, Sentinel(), parallel=parallel)
    assert not provider.calls


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("replacement", [None, "ordinary", "group"])
def test_tool_context_cannot_suppress_a_settled_native_failure(parallel, replacement):
    error = failed_response()

    class Swallow(ToolRuntime):
        @asynccontextmanager
        async def open(self, agent):
            try:
                async with SimpleRuntime().open(agent) as session:
                    yield session
            except ProviderResponseError:
                if replacement == "ordinary":
                    raise RuntimeError("cleanup failed") from None
                if replacement == "group":
                    raise ExceptionGroup("cleanup failed", [RuntimeError("close")]) from None

    graph, provider, budget = graph_for(), Script(error), Sentinel(1)
    with pytest.raises(ProviderResponseError) as caught:
        execute(graph, provider, budget, parallel=parallel, tool_runtime=Swallow())
    assert caught.value is error
    assert len(provider.calls) == 1 and budget.breakdown() == {"a": 0.1}
    assert graph.nodes[0].status is NodeStatus.FAILED


@pytest.mark.parametrize("unknown", [False, True])
def test_synthesis_failure_settles_once_and_keeps_safe_phase_receipt(unknown):
    error = failed_response(unknown=unknown)
    provider, budget, tracer = Script(error), Sentinel(1), Tracer()
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A", status=NodeStatus.COMPLETED, result="ok")])
    synth = Synthesizer(SynthesisStrategy.LLM_MERGE, provider=provider, budget=budget, tracer=tracer)
    with pytest.raises(type(error)) as caught:
        synth.synthesize(graph)
    assert caught.value is error
    assert budget.total_cost_usd == pytest.approx(0.2 if unknown else 0.1)
    assert len(provider.calls) == 1
    [span] = tracer.summary()
    assert span["native_receipts"][0]["phase"] == "synthesis"
    assert SECRET not in json.dumps(span)


def test_synthesis_overrun_retains_receipt_and_raw_cause():
    error, budget, tracer = failed_response(cost=0.3), Sentinel(1), Tracer()
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A", status=NodeStatus.COMPLETED, result="ok")])
    synth = Synthesizer(SynthesisStrategy.LLM_MERGE, provider=Script(error), budget=budget, tracer=tracer)
    with pytest.raises(BudgetReconciliationError) as caught:
        synth.synthesize(graph)
    assert caught.value.__cause__ is error
    assert budget.breakdown() == {"__synthesis__": 0.3}
    assert tracer.summary()[0]["native_receipts"][0]["phase"] == "synthesis"


@pytest.mark.parametrize("parallel", [False, True])
def test_synthesis_error_checkpoint_blocks_rebuy_without_worker_marker(tmp_path, parallel):
    error = failed_response()
    provider = Script(CompletionResult("worker", cost_usd=0.1), error)
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, parallel=parallel, max_budget_usd=1, checkpoint_store=store,
                  synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE))
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A")])
    with pytest.raises(ProviderResponseError):
        swarm.execute(graph)
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    assert state["budget"]["node_costs"] == {"a": 0.1, "__synthesis__": 0.1}
    assert state["control"]["response_error"]
    assert not graph.nodes[0].metadata.get("response_error")
    with pytest.raises(ProviderResponseError):
        swarm.resume(execution_id)
    assert len(provider.calls) == 2


@pytest.mark.parametrize("parallel", [False, True])
def test_supervision_native_error_is_terminal_and_billed_in_its_phase(parallel):
    error = failed_response()

    class FailedReview(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            raise error

    graph, budget = graph_for(), Sentinel(1)
    provider = Script(CompletionResult("worker", cost_usd=0.1))
    with pytest.raises(ProviderResponseError) as caught:
        execute(graph, provider, budget, parallel=parallel, supervisor=FailedReview(), max_revisions=1)
    assert caught.value is error
    assert budget.breakdown() == {"a": 0.2}
    assert graph.nodes[0].metadata["native_receipts"][0]["phase"] == "supervision"
    assert graph.nodes[1].status is NodeStatus.PENDING
    assert len(provider.calls) == 1


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("where", ["call", "open", "close"])
def test_tool_native_error_never_becomes_model_feedback(parallel, where):
    error = failed_response()

    def handler(call):
        raise error

    class FailingRuntime(ToolRuntime):
        @asynccontextmanager
        async def open(self, agent):
            if where == "open":
                raise error
            async with SimpleRuntime(handler=handler).open(agent) as session:
                yield session
            if where == "close":
                raise error

    provider = Script(CompletionResult(
        "answer", cost_usd=0.05,
        tool_calls=[ToolCall("call_1", "calc.add", {})] if where == "call" else [],
    ))
    graph, budget = graph_for(), Sentinel(1)
    with pytest.raises(ProviderResponseError) as caught:
        execute(graph, provider, budget, parallel=parallel, tool_runtime=FailingRuntime())
    assert caught.value is error
    assert len(provider.calls) == (0 if where == "open" else 1)
    assert budget.breakdown()["a"] == pytest.approx(0.1 if where == "open" else 0.15)
    assert len(graph.nodes[0].metadata["native_receipts"]) == 1
    assert graph.nodes[0].metadata["native_receipts"][0]["phase"] == "tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ProviderResponseError, ProviderAccountingError, RuntimeError])
async def test_mcp_preserves_typed_billing_errors_but_contains_ordinary_errors(error_type):
    error = error_type("failure")

    class Session:
        async def call_tool(self, *args, **kwargs):
            raise error

    session = MCPToolSession([], {"calc.add": _DispatchEntry(Session(), "add", 1)})
    call = ToolCall("call_1", "calc.add", {})
    if error_type is RuntimeError:
        assert (await session.call(call)).is_error
    else:
        with pytest.raises(error_type) as caught:
            await session.call(call)
        assert caught.value is error


@pytest.mark.parametrize("parallel", [False, True])
def test_distinct_provider_and_teardown_errors_are_each_settled_once(parallel):
    body_error, close_error = failed_response(cost=0.05), failed_response(cost=0.1)

    class FailingClose(ToolRuntime):
        @asynccontextmanager
        async def open(self, agent):
            try:
                async with SimpleRuntime().open(agent) as session:
                    yield session
            finally:
                raise close_error

    graph, provider, budget = graph_for(), Script(body_error), Sentinel(1)
    with pytest.raises(ProviderResponseError) as caught:
        execute(graph, provider, budget, parallel=parallel, tool_runtime=FailingClose())
    assert caught.value is close_error
    assert caught.value.__context__ is body_error
    assert len(provider.calls) == 1
    assert budget.breakdown()["a"] == pytest.approx(0.15)
    assert [r["phase"] for r in graph.nodes[0].metadata["native_receipts"]] == ["execution", "tool"]


@pytest.mark.parametrize("parallel", [False, True])
def test_known_teardown_bill_does_not_consume_unknown_body_reservation(parallel):
    body_error, close_error = failed_response(unknown=True), failed_response(cost=0.1)

    class FailingClose(ToolRuntime):
        @asynccontextmanager
        async def open(self, agent):
            try:
                async with SimpleRuntime().open(agent) as session:
                    yield session
            finally:
                raise close_error

    graph, provider, budget = graph_for(), Script(body_error), Sentinel(1)
    with pytest.raises(ProviderResponseError) as caught:
        execute(graph, provider, budget, parallel=parallel, tool_runtime=FailingClose())
    assert caught.value is close_error
    assert budget.breakdown() == {"a": 0.1}
    assert budget.total_cost_usd == pytest.approx(0.3)
    assert budget.cost_is_complete is False
    assert graph.nodes[0].metadata["cost_usd_unknown"]
    with pytest.raises(SentinelAlert):
        budget.reserve("probe", 0.8)
    assert [r["phase"] for r in graph.nodes[0].metadata["native_receipts"]] == ["execution", "tool"]


@pytest.mark.parametrize("parallel", [False, True])
def test_unknown_no_ceiling_or_cap_is_still_incomplete(parallel):
    budget, graph = Sentinel(), graph_for()
    with pytest.raises(ProviderAccountingError):
        execute(graph, Script(failed_response(unknown=True), ceiling=None), budget, parallel=parallel)
    assert budget.breakdown() == {} and budget.total_cost_usd == 0
    assert not budget.cost_is_complete
    assert graph.nodes[0].metadata["cost_usd_unknown"]


def test_preserved_reservation_known_cost_is_atomic_and_retains_overrun():
    budget = Sentinel(0.25)
    budget.reserve("a", 0.2, hard_ceiling=True)
    budget.mark_unknown("a")
    invalid = CompletionResult("", cost_usd=0.01)
    invalid.cost_usd = float("nan")
    with pytest.raises(BudgetValidationError):
        budget.add_cost("a", invalid, preserve_reservation=True)
    assert budget.total_cost_usd == 0.2 and budget.breakdown() == {}
    with pytest.raises(BudgetReconciliationError):
        budget.add_cost("a", CompletionResult("", cost_usd=0.1), preserve_reservation=True)
    assert budget.breakdown() == {"a": 0.1}
    assert budget.total_cost_usd == pytest.approx(0.3)
    assert budget._reservations == {"a": 0.2} and budget._hard_reservations == {"a"}
    assert not budget.cost_is_complete


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("kind", ["known", "unknown", "numeric", "cancelled"])
def test_grouped_body_and_teardown_failures_remain_terminal(parallel, kind):
    body = {
        "known": failed_response(cost=0.05),
        "unknown": failed_response(unknown=True),
        "numeric": BudgetValidationError("invalid grouped usage"),
        "cancelled": ProviderAccountingCancelledError(
            "cancelled native call", receipt={"cost_is_complete": False},
        ),
    }[kind]
    teardown = failed_response(cost=0.1)

    class GroupedClose(ToolRuntime):
        @asynccontextmanager
        async def open(self, agent):
            try:
                async with SimpleRuntime().open(agent) as session:
                    yield session
            except BaseException as error:
                children = [error, teardown, OSError("ordinary teardown failure")]
                if kind == "cancelled":
                    children.append(asyncio.CancelledError("task-group cancellation"))
                raise BaseExceptionGroup("context exit failed", children) from None

    graph, provider, budget = graph_for(), Script(body), Sentinel(1)
    with pytest.raises(type(body)) as caught:
        execute(graph, provider, budget, parallel=parallel, tool_runtime=GroupedClose())
    assert caught.value is body
    assert len(provider.calls) == 1
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert graph.nodes[1].status is NodeStatus.PENDING
    assert budget.breakdown()["a"] == pytest.approx(0.15 if kind == "known" else 0.1)
    assert budget.total_cost_usd == pytest.approx(0.15 if kind == "known" else 0.3)
    assert budget.cost_is_complete is (kind == "known")
    assert len(graph.nodes[0].metadata["native_receipts"]) == (1 if kind == "numeric" else 2)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_mutated_native_error_bill_stays_unknown_and_preserves_exposure(parallel, grouped):
    error = failed_response()
    error.billing_result.cost_usd = float("nan")
    thrown = ExceptionGroup("bills", [error, failed_response(cost=0.1)]) if grouped else error
    graph, provider, budget = graph_for(), Script(thrown), Sentinel(1)
    with pytest.raises(BudgetValidationError):
        execute(graph, provider, budget, parallel=parallel)
    assert len(provider.calls) == 1
    assert not budget.cost_is_complete
    assert graph.nodes[0].metadata["accounting_invalid"]
    assert graph.nodes[0].metadata["cost_usd_unknown"]
    assert budget.total_cost_usd == pytest.approx(0.3 if grouped else 0.2)
    assert budget.breakdown() == ({"a": 0.1} if grouped else {})
    assert all(r["phase"] == "execution" for r in graph.nodes[0].metadata["native_receipts"])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["known", "unknown", "cancelled"])
async def test_native_failure_during_descendant_settlement_blocks_regeneration(kind):
    entered = asyncio.Event()
    error = (ProviderAccountingCancelledError("cancelled native call", receipt={})
             if kind == "cancelled" else failed_response(unknown=kind == "unknown"))
    calls = []

    class Consumer(Provider):
        def budget_estimate_usd(self, model):
            return 0.2

        def requires_explicit_budget_estimate(self, model):
            return True

        async def complete(self, system, prompt, model):
            name = prompt.splitlines()[0]
            calls.append(name)
            if name == "consumer":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise error
            if name == "judge":
                await entered.wait()
            return CompletionResult("FAIL" if name == "judge" else "draft-v1", cost_usd=0.05)

    graph = ExecutionGraph([Topology.FORK_JOIN], [
        Node("draft", id="draft"),
        Node("judge", id="judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
        Node("consumer", id="consumer", depends_on=["draft"], failure_policy=FailurePolicy.RETRY,
             max_retries=3),
    ])
    budget = Sentinel(1)
    executor = AsyncExecutor(provider=Consumer(), registry=Registry(), tracer=Tracer(), budget=budget)
    with pytest.raises(type(error)) as caught:
        await asyncio.wait_for(executor.run(graph), 2)
    assert caught.value is error
    assert sorted(calls) == ["consumer", "draft", "judge"]
    assert graph.nodes[1].metadata["regeneration_intent"]
    assert graph.nodes[2].status is NodeStatus.FAILED
    assert budget.total_cost_usd == pytest.approx(0.2 if kind == "known" else 0.3)
    assert budget.cost_is_complete is (kind == "known")


@pytest.mark.parametrize("parallel", [False, True])
def test_successful_worker_and_synthesis_expose_safe_receipts_only(tmp_path, parallel):
    continuation = {"provider": "openai_responses", "model": "gpt-6-astra", "output": [
        {"type": "reasoning", "encrypted_content": SECRET},
    ]}
    provider = Script(*(CompletionResult(
        text, cost_usd=0.1, native_receipt=SAFE, provider_continuation=continuation,
        response_envelope=SimpleNamespace(body=SECRET.encode()),
    ) for text in ("worker", "merged")))
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, parallel=parallel, max_budget_usd=1, checkpoint_store=store,
                  synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE))
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A")])
    result = swarm.execute(graph)
    assert result.output == "merged" and result.total_cost_usd == 0.2
    assert [s["native_receipts"][0]["phase"] for s in result.trace] == ["execution", "synthesis"]
    assert SECRET not in json.dumps(result.trace)
    assert SECRET not in json.dumps(store.load(result.execution_id))


class Interrupted(Script):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.error = None

    async def chat(self, *args, **kwargs):
        self.calls.append(args)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            self.error = ProviderAccountingCancelledError(
                "Native dispatch interrupted; bill unknown", receipt={"cost_is_complete": False},
                envelope=SimpleNamespace(body=SECRET.encode()),
            )
            raise self.error from exc


@pytest.mark.parametrize("parallel", [False, True])
def test_native_timeout_is_terminal_under_retry_and_keeps_reservation(parallel):
    provider, graph, budget = Interrupted(), graph_for(timeout_s=0.02), Sentinel(1)
    with pytest.raises(ProviderAccountingCancelledError) as caught:
        execute(graph, provider, budget, parallel=parallel)
    assert caught.value is provider.error
    assert len(provider.calls) == 1
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert graph.nodes[0].metadata["accounting_invalid"]
    assert budget.total_cost_usd == 0.2 and budget.breakdown() == {}


@pytest.mark.asyncio
async def test_cancel_after_dispatch_persists_unknown_exposure_and_blocks_resume(tmp_path):
    provider, graph = Interrupted(), graph_for()
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, max_budget_usd=1, checkpoint_store=store)
    task = asyncio.create_task(swarm.execute_async(graph))
    await asyncio.wait_for(provider.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(provider.calls) == 1
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert graph.nodes[0].metadata["accounting_invalid"]
    assert swarm._active_executor._budget.total_cost_usd == 0.2
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    assert state["graph"]["nodes"][0]["metadata"]["accounting_invalid"]
    assert SECRET not in json.dumps(state)
    with pytest.raises(BudgetValidationError):
        await swarm.aresume(execution_id)
    assert len(provider.calls) == 1
