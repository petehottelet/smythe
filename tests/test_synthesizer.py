"""Tests for the Synthesizer and SynthesisStrategy."""

import json

import pytest

from smythe.budget import BudgetTracker
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
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


def _make_graph(*results: str) -> ExecutionGraph:
    nodes = []
    for i, r in enumerate(results):
        n = Node(id=f"n{i}", label=f"Step {i}")
        n.status = NodeStatus.COMPLETED
        n.result = r
        nodes.append(n)
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)


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
    budget = BudgetTracker(max_budget_usd=1.0)
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
    runtime_budget = BudgetTracker(max_budget_usd=1.0)
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
