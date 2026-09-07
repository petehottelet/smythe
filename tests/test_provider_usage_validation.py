"""Provider results and restored runs cannot bypass strict cost validation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe import (
    BudgetValidationError, CompletionResult, FileCheckpointStore, Sentinel,
    Swarm, Synthesizer, SynthesisStrategy, Task,
)
from smythe.checkpoint import graph_to_dict
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import AnthropicProvider, GeminiProvider, OfflineProvider, OpenAIProvider, Provider
from smythe.constrained_planner import ConstrainedArchitect
from smythe.planner import LLMArchitect, SimpleArchitect
from smythe.router import WhiteRabbit


@pytest.mark.parametrize("value", [True, False, "1", -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "cost_usd"])
def test_result_rejects_malformed_usage_at_construction(field, value):
    with pytest.raises(BudgetValidationError):
        CompletionResult("invalid", **{field: value})


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
def test_result_rejects_fractional_token_counts_even_with_explicit_cost(field):
    with pytest.raises(BudgetValidationError):
        CompletionResult("invalid", cost_usd=1.0, **{field: 1.5})


def test_result_accepts_zero_usage_and_finite_explicit_cost():
    result = CompletionResult("cached", cost_usd=0, prompt_tokens=0, completion_tokens=0)
    assert result.total_tokens == 0


@pytest.mark.parametrize("value", [True, "0.1"])
def test_legacy_provider_hint_keeps_type_for_strict_admission(value):
    provider = OfflineProvider()
    provider.cost_estimate_per_call = value
    assert provider.budget_estimate_usd("offline") is value


@pytest.mark.parametrize("value", [True, "1", -1, float("nan"), float("inf")])
def test_swarm_rejects_invalid_policy_before_provider_selection(value):
    with pytest.raises(BudgetValidationError):
        Swarm(model="unconfigured", max_budget_usd=value)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_planning_rechecks_mutated_policy_before_provider_work(asynchronous):
    provider = OfflineProvider()
    swarm = Swarm(model="offline", provider=provider)
    swarm.max_budget_usd = float("nan")
    with pytest.raises(BudgetValidationError):
        if asynchronous:
            asyncio.run(swarm.aplan(Task("test")))
        else:
            swarm.plan(Task("test"))
    assert provider.calls == []


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("value", [True, "1", -1, float("nan"), float("inf")])
def test_resume_validates_costs_before_completed_shortcut_or_dispatch(tmp_path, status, value):
    provider = OfflineProvider()
    store = FileCheckpointStore(tmp_path)
    graph = ExecutionGraph(
        topology=[Topology.SERIAL],
        nodes=[Node(id="a", label="A", status=NodeStatus.COMPLETED, result="saved")],
    )
    store.save("run", {
        "version": 2, "status": status, "graph": graph_to_dict(graph),
        "budget": {"max_budget_usd": 10, "node_costs": {"a": value}}, "output": "saved",
    })
    swarm = Swarm(model="offline", provider=provider, checkpoint_store=store)
    with pytest.raises(BudgetValidationError):
        swarm.resume("run")
    assert provider.calls == []


class InvalidSecondResult(Provider):
    def __init__(self):
        self.calls = 0

    def budget_estimate_usd(self, model):
        return 0.4

    async def complete(self, system, prompt, model):
        self.calls += 1
        return CompletionResult("result", cost_usd=0.25 if self.calls == 1 else float("nan"))


@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_post_call_usage_is_durable_and_cannot_unlock_spend_on_resume(tmp_path, parallel):
    provider = InvalidSecondResult()
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(
        model="test", provider=provider, parallel=parallel,
        max_budget_usd=1.0, checkpoint_store=store,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
    ])
    with pytest.raises(BudgetValidationError):
        swarm.execute(graph)
    assert provider.calls == 2
    execution_id = store.list_ids()[0]
    state = store.load(execution_id)
    assert state["budget"]["node_costs"] == {"a": 0.25}
    assert state["graph"]["nodes"][1]["metadata"]["accounting_invalid"] is True
    with pytest.raises(BudgetValidationError, match="reconcile provider charges"):
        swarm.resume(execution_id)
    assert provider.calls == 2


@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_synthesis_usage_blocks_resume_at_workflow_boundary(tmp_path, parallel):
    provider = InvalidSecondResult()
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(
        model="test", provider=provider, parallel=parallel, max_budget_usd=1.0,
        checkpoint_store=store, synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE),
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="a", label="A")])
    with pytest.raises(BudgetValidationError):
        swarm.execute(graph)
    execution_id = store.list_ids()[0]
    state = store.load(execution_id)
    assert state["budget"]["node_costs"] == {"a": 0.25}
    assert state["budget"]["accounting_error"]
    with pytest.raises(BudgetValidationError, match="reconcile provider charges"):
        swarm.resume(execution_id)
    assert provider.calls == 2


def test_synthesis_keeps_reservation_after_invalid_report():
    provider = InvalidSecondResult()
    provider.calls = 1
    budget = Sentinel(1.0)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A", status=NodeStatus.COMPLETED, result="ready"),
    ])
    synthesizer = Synthesizer(SynthesisStrategy.LLM_MERGE, provider=provider, budget=budget)
    with pytest.raises(BudgetValidationError):
        synthesizer.synthesize(graph)
    assert budget.total_cost_usd == 0.4
    assert budget.breakdown() == {}


@pytest.mark.parametrize("stage", ["planner", "constrained", "router"])
def test_planning_boundaries_revalidate_mutated_results_without_retry(stage):
    class MutatedResultProvider(Provider):
        calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            result = CompletionResult("unparseable")
            result.cost_usd = -1
            return result

    provider = MutatedResultProvider()
    task = Task("test")
    with pytest.raises(BudgetValidationError):
        if stage == "planner":
            LLMArchitect(provider, max_retries=3).plan(task)
        elif stage == "constrained":
            ConstrainedArchitect(provider, templates=[], max_retries=3).plan(task)
        else:
            WhiteRabbit(autonomous=SimpleArchitect(), classifier_provider=provider).route(task)
    assert provider.calls == 1


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("output", [False, True])
@pytest.mark.parametrize("value", [False, 0.0, "", None])
def test_adapter_preserves_malformed_falsy_usage_for_validation(kind, output, value):
    if kind == "openai":
        provider = OpenAIProvider(api_key="offline-test")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        setattr(usage, "completion_tokens" if output else "prompt_tokens", value)
        create = AsyncMock(return_value=SimpleNamespace(choices=[], usage=usage))
        provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    elif kind == "anthropic":
        provider = AnthropicProvider(api_key="offline-test")
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        setattr(usage, "output_tokens" if output else "input_tokens", value)
        create = AsyncMock(return_value=SimpleNamespace(content=[], usage=usage))
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    else:
        provider = GeminiProvider(api_key="offline-test")
        usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1)
        setattr(usage, "candidates_token_count" if output else "prompt_token_count", value)
        create = AsyncMock(return_value=SimpleNamespace(
            text="ok", function_calls=[], candidates=[], usage_metadata=usage,
        ))
        provider._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=create)))
    if value is None:
        result = asyncio.run(provider.complete("system", "prompt", "test-model"))
        assert (result.completion_tokens if output else result.prompt_tokens) == 0
    else:
        with pytest.raises(BudgetValidationError):
            asyncio.run(provider.complete("system", "prompt", "test-model"))


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_empty_accounting_error_is_still_a_durable_resume_barrier(tmp_path, status):
    provider = OfflineProvider()
    store = FileCheckpointStore(tmp_path)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A", status=NodeStatus.COMPLETED, result="saved"),
    ])
    store.save("run", {
        "version": 2, "status": status, "graph": graph_to_dict(graph), "output": "saved",
        "budget": {"max_budget_usd": 1, "node_costs": {"a": 0.25}, "accounting_error": ""},
    })
    swarm = Swarm(model="offline", provider=provider, checkpoint_store=store)
    with pytest.raises(BudgetValidationError, match="Cannot resume unresolved"):
        swarm.resume("run")
    assert provider.calls == []
