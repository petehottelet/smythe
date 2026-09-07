"""Invalid accounting stops dispatch without discarding valid held exposure."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import BudgetValidationError, Sentinel
from smythe.checkpoint import FileCheckpointStore
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import LLMSupervisor
from smythe.swarm import Swarm
from smythe.tools import ToolCall
from smythe.tracer import Tracer


INVALID_MONEY = [True, False, "0.25", -1, float("nan"), float("inf"), -float("inf")]
INVALID_USAGE = [True, False, "2", -1, 1.5, float("nan"), float("inf")]


def run(mode, provider, nodes, budget, **kwargs):
    common = dict(provider=provider, registry=Registry(), tracer=Tracer(),
                  budget=budget, artifact_dir=None, **kwargs)
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes)
    if mode == "serial":
        return Executor(**common).run(graph)
    return asyncio.run(AsyncExecutor(**common, max_concurrency=1).run(graph))


class CountingProvider(Provider):
    def __init__(self):
        self.calls = []

    async def complete(self, system, prompt, model):
        self.calls.append(prompt.splitlines()[0])
        return CompletionResult("ok", cost_usd=0.1)


@pytest.mark.parametrize("value", INVALID_USAGE)
def test_parallel_token_estimate_rejects_non_integer_usage(value):
    with pytest.raises(BudgetValidationError):
        AsyncExecutor(CountingProvider(), Registry(), Tracer(),
                      estimated_tokens_per_node=value)


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("source", ["node", "provider", "legacy_provider_hint"])
@pytest.mark.parametrize("value", INVALID_MONEY)
def test_invalid_estimate_is_rejected_before_dispatch(mode, source, value):
    provider = CountingProvider()
    invalid = Node("invalid", id="invalid")
    if source == "node":
        invalid.metadata["estimated_cost_usd"] = value
    elif source == "provider":
        provider.budget_estimate_usd = lambda model: value
    else:
        provider.cost_estimate_per_call = value
    later = Node("later", id="later")
    budget = Sentinel(10)
    budget.restore({"earlier": 0.25})

    with pytest.raises(BudgetValidationError):
        run(mode, provider, [invalid, later], budget)

    assert provider.calls == []
    assert budget.total_cost_usd == 0.25
    assert budget.breakdown() == {"earlier": 0.25}
    assert invalid.metadata["accounting_invalid"] is True
    assert invalid.metadata["accounting_error"]
    assert later.status is NodeStatus.PENDING


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("policy", list(FailurePolicy))
@pytest.mark.parametrize("field,value", [
    *(('cost_usd', value) for value in INVALID_MONEY),
    *(('prompt_tokens', value) for value in INVALID_USAGE),
    *(('completion_tokens', value) for value in INVALID_USAGE),
])
def test_invalid_response_never_retries_skips_or_releases_ceiling(mode, policy, field, value):
    class InvalidResponse(CountingProvider):
        async def complete(self, system, prompt, model):
            self.calls.append(prompt.splitlines()[0])
            # Exercise the Sentinel boundary even when a custom provider
            # mutates a previously validated CompletionResult.
            result = CompletionResult("paid response", cost_usd=0.1)
            setattr(result, field, value)
            return result

    provider = InvalidResponse()
    invalid = Node("invalid", id="invalid", failure_policy=policy, max_retries=3,
                   metadata={"estimated_cost_usd": 0.5})
    later = Node("later", id="later")
    budget = Sentinel(10)
    budget.restore({"earlier": 0.25})

    with pytest.raises(BudgetValidationError):
        run(mode, provider, [invalid, later], budget)

    assert provider.calls == ["invalid"]
    assert invalid.status is NodeStatus.FAILED
    assert invalid.metadata["accounting_invalid"] is True
    assert later.status is NodeStatus.PENDING
    assert budget.breakdown() == {"earlier": 0.25}
    assert budget.total_cost_usd == 0.75
    # The failed response does not prove the provider's call was free.
    assert budget._reservations == {"invalid": 0.5}


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_provider_validation_error_is_preserved(mode):
    error = BudgetValidationError("invalid usage returned by provider")

    class InvalidProvider(CountingProvider):
        async def complete(self, system, prompt, model):
            self.calls.append(prompt.splitlines()[0])
            raise error

    provider = InvalidProvider()
    node = Node("invalid", id="invalid", failure_policy=FailurePolicy.RETRY,
                max_retries=3, metadata={"estimated_cost_usd": 0.5})
    budget = Sentinel(10)
    with pytest.raises(BudgetValidationError) as caught:
        run(mode, provider, [node], budget)
    assert caught.value is error
    assert provider.calls == ["invalid"]
    assert budget.total_cost_usd == 0.5
    assert node.metadata["accounting_error"] == str(error)


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_invalid_usage_is_terminal_without_a_sentinel(mode):
    class InvalidResponse(CountingProvider):
        async def complete(self, system, prompt, model):
            self.calls.append(prompt.splitlines()[0])
            result = CompletionResult("invalid")
            result.completion_tokens = True
            return result

    provider = InvalidResponse()
    invalid = Node("invalid", id="invalid", failure_policy=FailurePolicy.RETRY,
                   max_retries=3)
    later = Node("later", id="later")
    with pytest.raises(BudgetValidationError):
        run(mode, provider, [invalid, later], None)
    assert provider.calls == ["invalid"]
    assert invalid.metadata["accounting_invalid"] is True
    assert later.status is NodeStatus.PENDING


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_invalid_later_tool_turn_preserves_prior_billed_turn(mode):
    class Runtime:
        @asynccontextmanager
        async def open(self, agent):
            yield SimpleNamespace(tools=[])

    class InvalidLaterTurn(CountingProvider):
        async def chat(self, system, messages, model, tools=None):
            self.calls.append(len(messages))
            if len(self.calls) == 1:
                return CompletionResult("continue", stop_reason="pause_turn", cost_usd=0.25)
            result = CompletionResult("invalid")
            result.prompt_tokens = -1
            return result

    provider = InvalidLaterTurn()
    invalid = Node("invalid", id="invalid", failure_policy=FailurePolicy.RETRY,
                   max_retries=3, metadata={"estimated_cost_usd": 0.5})
    later = Node("later", id="later")
    budget = Sentinel(10)
    with pytest.raises(BudgetValidationError):
        run(mode, provider, [invalid, later], budget, tool_runtime=Runtime())
    assert provider.calls == [1, 2]
    assert budget.breakdown() == {"invalid": 0.25}
    assert budget.total_cost_usd == 0.25
    assert invalid.metadata["accounting_invalid"] is True
    assert later.status is NodeStatus.PENDING


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("mutation", [False, True])
def test_supervisor_accounting_error_stops_and_blocks_resume(mode, mutation, tmp_path):
    class InvalidSupervisorProvider(CountingProvider):
        error = None

        async def complete(self, system, prompt, model):
            self.calls.append("review")
            if mutation:
                result = CompletionResult('{"change": false}')
                result.prompt_tokens = -1
                return result
            try:
                return CompletionResult('{"change": false}', prompt_tokens=-1)
            except BudgetValidationError as exc:
                self.error = exc
                raise

    provider = CountingProvider()
    supervisor_provider = InvalidSupervisorProvider()
    store = FileCheckpointStore(tmp_path)
    first, later = Node("first", id="first"), Node("later", id="later", depends_on=["first"])
    graph = ExecutionGraph([Topology.SERIAL], [first, later])
    swarm = Swarm(provider=provider, model="test-model", parallel=mode == "parallel",
                  max_budget_usd=10, checkpoint_store=store, artifact_dir=None,
                  supervisor=LLMSupervisor(supervisor_provider, review_after={"first"}),
                  max_revisions=1)

    with pytest.raises(BudgetValidationError) as caught:
        swarm.execute(graph)

    if not mutation:
        assert caught.value is supervisor_provider.error
    assert provider.calls == ["first"]
    assert supervisor_provider.calls == ["review"]
    assert first.status is NodeStatus.COMPLETED
    assert later.status is NodeStatus.PENDING
    assert first.metadata["accounting_invalid"] is True
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    assert state["status"] == "failed"
    assert state["graph"]["nodes"][0]["metadata"]["accounting_invalid"] is True
    assert state["budget"]["node_costs"] == {"first": 0.1}
    with pytest.raises(BudgetValidationError, match="Cannot resume unresolved"):
        swarm.resume(execution_id)
    assert provider.calls == ["first"]
    assert supervisor_provider.calls == ["review"]


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_tool_accounting_error_is_not_sent_back_for_another_paid_turn(mode):
    error = BudgetValidationError("tool provider returned invalid usage")

    class Runtime:
        @asynccontextmanager
        async def open(self, agent):
            yield SimpleNamespace(tools=[], call=self.call)

        async def call(self, tool_call):
            raise error

    class CallsPaidTool(CountingProvider):
        async def chat(self, system, messages, model, tools=None):
            self.calls.append(len(messages))
            return CompletionResult("", cost_usd=0.25,
                                    tool_calls=[ToolCall("1", "paid_provider", {})])

    provider = CallsPaidTool()
    invalid = Node("invalid", id="invalid", failure_policy=FailurePolicy.RETRY,
                   max_retries=3, metadata={"estimated_cost_usd": 0.5})
    later = Node("later", id="later")
    budget = Sentinel(10)
    with pytest.raises(BudgetValidationError) as caught:
        run(mode, provider, [invalid, later], budget, tool_runtime=Runtime())
    assert caught.value is error
    assert provider.calls == [1]
    assert budget.breakdown() == {"invalid": 0.25}
    assert invalid.metadata["accounting_invalid"] is True
    assert later.status is NodeStatus.PENDING


@pytest.mark.asyncio
async def test_parallel_invalid_response_cancels_sibling_but_keeps_invalid_ceiling():
    started, cancelled = asyncio.Event(), asyncio.Event()

    class InvalidWithSibling(CountingProvider):
        async def complete(self, system, prompt, model):
            name = prompt.splitlines()[0]
            self.calls.append(name)
            if name == "sibling":
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            await started.wait()
            result = CompletionResult("paid response")
            result.cost_usd = float("nan")
            return result

    provider = InvalidWithSibling()
    invalid = Node("invalid", id="invalid", metadata={"estimated_cost_usd": 0.5})
    sibling = Node("sibling", id="sibling", metadata={"estimated_cost_usd": 0.4})
    queued = Node("queued", id="queued")
    budget = Sentinel(10)
    budget.restore({"earlier": 0.25})
    executor = AsyncExecutor(provider, Registry(), Tracer(), budget=budget,
                             artifact_dir=None, max_concurrency=2)
    graph = ExecutionGraph([Topology.FORK_JOIN], [invalid, sibling, queued])

    with pytest.raises(BudgetValidationError):
        await asyncio.wait_for(executor.run(graph), timeout=3)

    assert cancelled.is_set()
    assert provider.calls == ["invalid", "sibling"]
    assert invalid.status is NodeStatus.FAILED
    assert sibling.status is NodeStatus.PENDING
    assert queued.status is NodeStatus.PENDING
    assert budget.breakdown() == {"earlier": 0.25}
    assert budget.total_cost_usd == pytest.approx(0.75)
    assert budget._reservations == {"invalid": 0.5}
