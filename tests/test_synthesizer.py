"""Tests for the Synthesizer and SynthesisStrategy."""

import json

import pytest

from helpers import make_completed_graph as _make_graph
from smythe.budget import Sentinel
from smythe.graph import ExecutionGraph, Topology
from smythe.provider import CompletionResult, Provider
from smythe.synthesizer import Synthesizer, SynthesisStrategy
from smythe.tracer import Tracer


class MockSynthProvider(Provider):
    def __init__(self, response: str = "Merged output.") -> None:
        self._response = response
        self.called = False

    async def complete(self, system, prompt, model):
        self.called = True
        return CompletionResult(text=self._response, prompt_tokens=30, completion_tokens=40)


def test_concatenate_strategy():
    synth = Synthesizer(strategy=SynthesisStrategy.CONCATENATE)
    graph = _make_graph("Hello", "World")
    result = synth.synthesize(graph)
    assert result == "Hello\n\nWorld"


def test_concatenate_is_default():
    synth = Synthesizer()
    graph = _make_graph("A", "B")
    assert synth.synthesize(graph) == "A\n\nB"


def test_structured_merge_json():
    synth = Synthesizer(strategy=SynthesisStrategy.STRUCTURED)
    graph = _make_graph(
        json.dumps({"name": "Alice"}),
        json.dumps({"age": 30}),
    )
    result = synth.synthesize(graph)
    parsed = json.loads(result)
    assert parsed == {"name": "Alice", "age": 30}


def test_structured_merge_non_json_fallback():
    synth = Synthesizer(strategy=SynthesisStrategy.STRUCTURED)
    graph = _make_graph("plain text", json.dumps({"key": "val"}))
    result = synth.synthesize(graph)
    parsed = json.loads(result)
    assert parsed["n0"] == "plain text"
    assert parsed["key"] == "val"


def test_llm_merge_calls_provider():
    provider = MockSynthProvider("Synthesized result")
    synth = Synthesizer(strategy=SynthesisStrategy.LLM_MERGE, provider=provider)
    graph = _make_graph("Output A", "Output B")
    result = synth.synthesize(graph)

    assert result == "Synthesized result"
    assert provider.called


def test_llm_merge_budget_tracked():
    provider = MockSynthProvider()
    budget = Sentinel(max_budget_usd=1.0)
    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=provider,
        budget=budget,
    )
    graph = _make_graph("A", "B")
    synth.synthesize(graph)

    assert budget.total_cost_usd > 0
    assert "__synthesis__" in budget.breakdown()


def test_llm_merge_traced():
    provider = MockSynthProvider()
    tracer = Tracer()
    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=provider,
        tracer=tracer,
    )
    graph = _make_graph("X", "Y")
    synth.synthesize(graph)

    spans = tracer.summary()
    synthesis_spans = [s for s in spans if s.get("node_id") == "__synthesis__"]
    assert len(synthesis_spans) == 1


def test_llm_merge_no_provider_falls_back():
    synth = Synthesizer(strategy=SynthesisStrategy.LLM_MERGE)
    graph = _make_graph("A", "B")
    result = synth.synthesize(graph)
    assert result == "A\n\nB"


def test_empty_graph_returns_empty():
    synth = Synthesizer(strategy=SynthesisStrategy.LLM_MERGE, provider=MockSynthProvider())
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[])
    assert synth.synthesize(graph) == ""


@pytest.mark.asyncio
async def test_asynthesize_concatenate():
    """Async synthesize should work with CONCATENATE strategy."""
    synth = Synthesizer(strategy=SynthesisStrategy.CONCATENATE)
    graph = _make_graph("Hello", "World")
    result = await synth.asynthesize(graph)
    assert result == "Hello\n\nWorld"


@pytest.mark.asyncio
async def test_asynthesize_llm_merge():
    """Async synthesize should work with LLM_MERGE strategy."""
    provider = MockSynthProvider("Async merged")
    synth = Synthesizer(strategy=SynthesisStrategy.LLM_MERGE, provider=provider)
    graph = _make_graph("A", "B")
    result = await synth.asynthesize(graph)
    assert result == "Async merged"
    assert provider.called


def test_llm_merge_runtime_context_no_instance_mutation():
    """Per-call runtime context should not overwrite synthesizer defaults."""
    runtime_provider = MockSynthProvider("runtime")
    runtime_budget = Sentinel(max_budget_usd=1.0)
    runtime_tracer = Tracer()
    synth = Synthesizer(strategy=SynthesisStrategy.LLM_MERGE)
    graph = _make_graph("A", "B")

    out = synth.synthesize(
        graph,
        provider=runtime_provider,
        budget=runtime_budget,
        tracer=runtime_tracer,
    )

    assert out == "runtime"
    assert synth._provider is None
    assert synth._budget is None
    assert synth._tracer is None


def test_llm_merge_budget_exhaustion_raises():
    """LLM_MERGE should propagate SentinelAlert when the budget is already exhausted."""
    from smythe.budget import SentinelAlert

    provider = MockSynthProvider()
    budget = Sentinel(max_budget_usd=0.0001)
    budget._spent = 0.0001  # simulate prior spending that exhausts the budget

    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=provider,
        budget=budget,
    )
    graph = _make_graph("A", "B")

    with pytest.raises(SentinelAlert):
        synth.synthesize(graph)

    assert not provider.called


def test_llm_merge_reserves_before_call():
    """A merge whose estimate exceeds the remaining policy is never sent."""
    from smythe.budget import SentinelAlert

    provider = MockSynthProvider()
    budget = Sentinel(max_budget_usd=0.005)
    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=provider,
        budget=budget,
    )

    with pytest.raises(SentinelAlert):
        synth.synthesize(_make_graph("A", "B"))

    assert not provider.called


@pytest.mark.asyncio
async def test_asynthesize_llm_merge_budget_exhaustion():
    """Async LLM_MERGE should also propagate SentinelAlert."""
    from smythe.budget import SentinelAlert

    provider = MockSynthProvider()
    budget = Sentinel(max_budget_usd=0.0001)
    budget._spent = 0.0001

    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=provider,
        budget=budget,
    )
    graph = _make_graph("X", "Y")

    with pytest.raises(SentinelAlert):
        await synth.asynthesize(graph, budget=budget)

    assert not provider.called


def test_llm_merge_provider_error_sets_failed_status():
    """If the provider raises during LLM_MERGE, the synthesis node should be FAILED."""
    class _ErrorProvider(Provider):
        async def complete(self, system, prompt, model):
            raise RuntimeError("provider crash")

    tracer = Tracer()
    synth = Synthesizer(
        strategy=SynthesisStrategy.LLM_MERGE,
        provider=_ErrorProvider(),
        tracer=tracer,
    )
    graph = _make_graph("A")

    with pytest.raises(RuntimeError, match="provider crash"):
        synth.synthesize(graph)

    error_spans = [s for s in tracer.summary() if s.get("error") is not None]
    assert len(error_spans) == 1
