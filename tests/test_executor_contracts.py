"""Shared execution outcomes without assuming identical scheduler order or loops."""

import asyncio

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class ContractProvider(Provider):
    cost_estimate_per_call = 0.01

    def __init__(self, failure):
        self.failure = failure
        self.calls = []
        self.loops = []

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        self.calls.append(label)
        self.loops.append(asyncio.get_running_loop())
        if label == "B" and self.failure:
            if self.failure != "retry" or self.calls.count("B") == 1:
                raise RuntimeError("contract failure")
        return CompletionResult(text=f"done:{label}", cost_usd=0.01)


def execute(parallel, failure=None):
    provider = ContractProvider(failure)
    trace, budget = Tracer(), Sentinel(max_budget_usd=1)
    nodes = [Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
             Node(id="c", label="C", depends_on=["a"]),
             Node(id="d", label="D", depends_on=["b", "c"])]
    if failure == "skip":
        nodes[1].failure_policy = FailurePolicy.SKIP
    if failure == "retry":
        nodes[1].failure_policy = FailurePolicy.RETRY
        nodes[1].max_retries = 1
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes)
    updates = []
    arguments = dict(provider=provider, registry=Registry(), tracer=trace, budget=budget,
                     on_node_update=lambda node: updates.append((node.id, node.status.value)),
                     artifact_dir=None)
    engine = AsyncExecutor(**arguments, max_concurrency=1) if parallel else Executor(**arguments)

    def run():
        return asyncio.run(engine.run(graph)) if parallel else engine.run(graph)

    if failure == "halt":
        with pytest.raises(RuntimeError, match="contract failure"):
            run()
    else:
        run()
        previous = list(provider.calls)
        run()
        assert provider.calls == previous, "Completed/skipped nodes must not be dispatched on resume"
    for label in ("B", "C", "D"):
        if label in provider.calls:
            assert provider.calls.index("A") < provider.calls.index(label)
    if "D" in provider.calls:
        assert provider.calls.index("B") < provider.calls.index("D")
        assert provider.calls.index("C") < provider.calls.index("D")
    return ([(node.id, node.status.value, node.result) for node in nodes],
            budget.total_cost_usd, updates, provider)


@pytest.mark.parametrize("failure", [None, "skip", "retry", "halt"])
def test_concurrency_one_preserves_terminal_results_costs_and_saved_updates(failure):
    serial, concurrent = execute(False, failure), execute(True, failure)
    assert serial[:3] == concurrent[:3]
    if failure == "halt":
        assert serial[3].calls == concurrent[3].calls == ["A", "B"]


def test_provider_event_loop_lifetimes_are_intentionally_different():
    serial, concurrent = execute(False)[3], execute(True)[3]
    assert len(set(serial.loops)) == 4
    assert len(set(concurrent.loops)) == 1


def test_depth_first_and_ready_queue_orders_are_distinct_but_dependency_safe():
    orders = []
    for parallel in (False, True):
        provider = ContractProvider(None)
        graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
            Node(id="join", label="Join", depends_on=["b", "a"]),
            Node(id="a", label="A", depends_on=["root"]),
            Node(id="later", label="Later"), Node(id="root", label="Root"),
            Node(id="b", label="B", depends_on=["root"]),
        ])
        arguments = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None)
        if parallel:
            asyncio.run(AsyncExecutor(**arguments, max_concurrency=1).run(graph))
        else:
            Executor(**arguments).run(graph)
        assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes)
        orders.append(provider.calls)
    assert orders[0] == ["Root", "B", "A", "Join", "Later"]
    assert orders[1] == ["Later", "Root", "A", "B", "Join"]
