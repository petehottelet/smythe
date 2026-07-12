"""Full pipeline integration tests — router -> parallel execution -> synthesis."""

import json

import pytest

from helpers import ClassifierMockProvider, MockProvider
from smythe import Swarm, Task
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.planner import DeterministicArchitect, SimpleArchitect
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.synthesizer import Synthesizer, SynthesisStrategy


class _MergeMockProvider(Provider):
    """Echoes node outputs on execution calls, returns 'MERGED' on synthesis calls."""

    async def complete(self, system, prompt, model):
        if "synthesis" in system.lower() or "merge" in system.lower():
            return CompletionResult(text="MERGED OUTPUT", prompt_tokens=20, completion_tokens=10)
        return CompletionResult(text=f"result: {prompt[:30]}", prompt_tokens=10, completion_tokens=5)


class _ParallelArchitect(DeterministicArchitect):
    """Produces a 2-node parallel graph with a fan-in node."""

    def plan(self, task):
        a = Node(id="a", label="Branch A")
        b = Node(id="b", label="Branch B")
        c = Node(id="c", label="Fan-in", depends_on=["a", "b"])
        graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b, c])
        graph.validate()
        return graph, Registry()


def test_router_to_parallel_to_llm_merge():
    """Full pipeline: classifier routes to parallel architect, executes, LLM merges."""
    provider = _MergeMockProvider()
    router = WhiteRabbit(
        deterministic={"parallel": _ParallelArchitect()},
        autonomous=SimpleArchitect(),
        classifier_provider=ClassifierMockProvider("deterministic:parallel"),
        classifier_model="test",
    )
    swarm = Swarm(
        provider=provider,
        router=router,
        parallel=True,
        synthesizer=Synthesizer(strategy=SynthesisStrategy.LLM_MERGE),
    )
    task = Task(goal="Full pipeline test")
    result = swarm.execute(task)

    assert result.output == "MERGED OUTPUT"
    completed = [n for n in result.graph.nodes if n.status == NodeStatus.COMPLETED]
    assert len(completed) == 3
    # 3 node spans + 1 synthesis span
    assert len(result.trace) == 4


def test_serial_pipeline_with_concatenation():
    """Serial execution with default CONCATENATE synthesis."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Simple serial run")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].status == NodeStatus.COMPLETED
    assert len(result.trace) >= 1


def test_parallel_pipeline_with_structured_merge():
    """Parallel execution with STRUCTURED synthesis merges JSON outputs."""
    class _JsonProvider(Provider):
        async def complete(self, system, prompt, model):
            label = prompt.split("\n")[0]
            data = json.dumps({"source": label})
            return CompletionResult(text=data, prompt_tokens=5, completion_tokens=5)

    swarm = Swarm(
        provider=_JsonProvider(),
        architect=_ParallelArchitect(),
        parallel=True,
        synthesizer=Synthesizer(strategy=SynthesisStrategy.STRUCTURED),
    )
    task = Task(goal="JSON merge")
    result = swarm.execute(task)

    parsed = json.loads(result.output)
    assert isinstance(parsed, dict)
    assert "source" in parsed


def test_pipeline_budget_tracking():
    """End-to-end execution should track costs through the budget."""
    swarm = Swarm(
        provider=MockProvider(),
        architect=SimpleArchitect(),
        max_budget_usd=10.0,
    )
    task = Task(goal="Budget tracking")
    result = swarm.execute(task)

    assert result.total_cost_usd > 0
    assert result.cost_is_complete is True
    assert result.cost_contains_estimates is False


def test_pipeline_traces_all_nodes():
    """Trace should contain one span per executed node."""
    swarm = Swarm(
        provider=MockProvider(),
        architect=_ParallelArchitect(),
        parallel=True,
    )
    task = Task(goal="Trace all")
    result = swarm.execute(task)

    node_ids_in_trace = {s["node_id"] for s in result.trace}
    node_ids_in_graph = {n.id for n in result.graph.nodes}
    assert node_ids_in_graph.issubset(node_ids_in_trace)


@pytest.mark.asyncio
async def test_async_pipeline_end_to_end():
    """Fully async path: aplan + execute_async."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Async e2e")
    result = await swarm.execute_async(task)

    assert result.output != ""
    assert result.total_cost_usd >= 0
    assert result.cost_is_complete is True
    assert len(result.trace) >= 1
