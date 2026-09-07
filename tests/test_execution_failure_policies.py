"""Serial and concurrency-one execution share terminal failure semantics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import FileCheckpointStore
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.tracer import Tracer


@pytest.fixture(params=["serial", "parallel-c1"])
def mode(request):
    return request.param


def make_graph(policy, *, timeout_s=None):
    nodes = [
        Node("prior", id="prior"),
        Node("problem", id="problem", failure_policy=policy, max_retries=2,
             timeout_s=timeout_s),
        Node("queued", id="queued"),
        Node("skip-child", id="skip-child", depends_on=["problem"],
             failure_policy=FailurePolicy.SKIP),
        Node("grandchild", id="grandchild", depends_on=["skip-child"]),
    ]
    for node in nodes:
        node.metadata["estimated_cost_usd"] = 0.25
    return ExecutionGraph([Topology.FORK_JOIN], nodes)


class ScriptedProvider(Provider):
    def __init__(self, failures=()):
        self.failures = list(failures)
        self.calls = []

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        self.calls.append(label)
        if label == "problem" and self.failures:
            raise self.failures.pop(0)
        return CompletionResult(f"done: {label}", cost_usd=0.125)


def run(mode, provider, graph, budget, **kwargs):
    options = dict(provider=provider, registry=Registry(), tracer=Tracer(),
                   budget=budget, artifact_dir=None, **kwargs)
    if mode == "serial":
        return Executor(**options).run(graph)
    return asyncio.run(AsyncExecutor(**options, max_concurrency=1).run(graph))


def assert_halted(graph):
    prior, problem, *queued = graph.nodes
    assert prior.status is NodeStatus.COMPLETED
    assert prior.result == "done: prior"
    assert problem.status is NodeStatus.FAILED
    assert all(node.status is NodeStatus.PENDING and node.result is None for node in queued)


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3)])
def test_terminal_failure_stops_queued_work_and_preserves_original_exception(mode, policy, attempts):
    errors = [RuntimeError(f"attempt {i}") for i in range(attempts)]
    provider = ScriptedProvider(errors)
    graph, budget = make_graph(policy), Sentinel(10)

    with pytest.raises(RuntimeError) as caught:
        run(mode, provider, graph, budget)

    assert caught.value is errors[-1]
    assert provider.calls == ["prior"] + ["problem"] * attempts
    assert_halted(graph)
    assert graph.nodes[1].result == str(errors[-1])
    assert budget.breakdown() == {"prior": 0.125}
    assert budget.total_cost_usd == 0.125
    assert budget._reservations == {}


@pytest.mark.parametrize("policy,failures", [(FailurePolicy.SKIP, 1), (FailurePolicy.RETRY, 2)])
def test_consumed_failure_policy_allows_later_work(mode, policy, failures):
    provider = ScriptedProvider([RuntimeError("transient")] * failures)
    graph, budget = make_graph(policy), Sentinel(10)

    assert run(mode, provider, graph, budget) is graph

    attempts = 1 if policy is FailurePolicy.SKIP else failures + 1
    assert provider.calls == ["prior"] + ["problem"] * attempts + [
        "queued", "skip-child", "grandchild",
    ]
    expected_status = NodeStatus.SKIPPED if policy is FailurePolicy.SKIP else NodeStatus.COMPLETED
    assert graph.nodes[1].status is expected_status
    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes if node.id != "problem")
    expected_ids = {node.id for node in graph.nodes if node.status is NodeStatus.COMPLETED}
    assert budget.breakdown() == dict.fromkeys(expected_ids, 0.125)
    assert budget.total_cost_usd == len(expected_ids) * 0.125
    assert budget._reservations == {}


class HangingProvider(ScriptedProvider):
    def __init__(self):
        super().__init__()
        self.cancelled = 0

    async def complete(self, system, prompt, model):
        if prompt.splitlines()[0] != "problem":
            return await super().complete(system, prompt, model)
        self.calls.append("problem")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3),
                                           (FailurePolicy.SKIP, 1)])
def test_timeout_obeys_failure_policy_and_settles_before_return(mode, policy, attempts):
    provider = HangingProvider()
    graph, budget = make_graph(policy, timeout_s=0.01), Sentinel(10)

    if policy is FailurePolicy.SKIP:
        run(mode, provider, graph, budget)
        assert graph.nodes[1].status is NodeStatus.SKIPPED
        assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes[2:])
        expected_tail = ["queued", "skip-child", "grandchild"]
    else:
        with pytest.raises(TimeoutError, match="'problem' timed out after 0.01s"):
            run(mode, provider, graph, budget)
        assert_halted(graph)
        expected_tail = []

    assert provider.cancelled == attempts
    assert provider.calls == ["prior"] + ["problem"] * attempts + expected_tail
    assert budget.breakdown() == dict.fromkeys(["prior", *expected_tail], 0.125)
    assert budget.total_cost_usd == (1 + len(expected_tail)) * 0.125
    assert budget._reservations == {}


class EmptyTools:
    @asynccontextmanager
    async def open(self, agent):
        yield SimpleNamespace(tools=[])


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3),
                                           (FailurePolicy.SKIP, 1)])
def test_failure_retains_cost_of_completed_provider_turns(mode, policy, attempts):
    error = RuntimeError("later conversation turn failed")

    class PaidThenFails(ScriptedProvider):
        async def chat(self, system, messages, model, tools=None):
            if messages[0].content.splitlines()[0] != "problem":
                return await super().chat(system, messages, model, tools)
            self.calls.append("problem")
            if len(messages) == 1:
                return CompletionResult("continue", stop_reason="pause_turn", cost_usd=0.0625)
            raise error

    provider = PaidThenFails()
    graph, budget = make_graph(policy), Sentinel(10)
    if policy is FailurePolicy.SKIP:
        run(mode, provider, graph, budget, tool_runtime=EmptyTools())
        assert graph.nodes[1].status is NodeStatus.SKIPPED
        tail = ["queued", "skip-child", "grandchild"]
    else:
        with pytest.raises(RuntimeError) as caught:
            run(mode, provider, graph, budget, tool_runtime=EmptyTools())
        assert caught.value is error
        assert_halted(graph)
        tail = []

    assert provider.calls == ["prior"] + ["problem"] * (2 * attempts) + tail
    costs = {"prior": 0.125, "problem": attempts * 0.0625, **dict.fromkeys(tail, 0.125)}
    assert budget.breakdown() == costs
    assert budget.total_cost_usd == sum(costs.values())
    assert graph.nodes[1].metadata["cost_usd"] == costs["problem"]
    assert budget._reservations == {}


@pytest.mark.parametrize("policy,attempts", [(FailurePolicy.HALT, 1), (FailurePolicy.RETRY, 3)])
def test_halted_checkpoint_resumes_failed_and_pending_work_only(mode, policy, attempts, tmp_path):
    errors = [RuntimeError(f"attempt {i}") for i in range(attempts)]
    provider = ScriptedProvider(errors)
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, model="test-model", parallel=mode == "parallel-c1",
                  max_concurrency=1, max_budget_usd=10, checkpoint_store=store,
                  checkpoint_every_n_nodes=100, artifact_dir=None)
    graph = make_graph(policy)
    with pytest.raises(RuntimeError) as caught:
        swarm.execute(graph)
    assert caught.value is errors[-1]
    assert_halted(graph)
    assert provider.calls == ["prior"] + ["problem"] * attempts
    [execution_id] = store.list_ids()
    failed = store.load(execution_id)
    assert failed["status"] == "failed"
    assert [node["status"] for node in failed["graph"]["nodes"]] == [
        "completed", "failed", "pending", "pending", "pending",
    ]
    assert failed["budget"]["node_costs"] == {"prior": 0.125}

    # A fresh process-equivalent Swarm must not repurchase completed work or
    # treat the never-dispatched SKIP child as already resolved.
    resumed_provider = ScriptedProvider()
    resumed = Swarm(provider=resumed_provider, model="test-model", max_concurrency=1,
                    checkpoint_store=store, artifact_dir=None)
    result = resumed.resume(execution_id)
    assert resumed_provider.calls == ["problem", "queued", "skip-child", "grandchild"]
    assert all(node.status is NodeStatus.COMPLETED for node in result.graph.nodes)
    assert result.graph.nodes[0].result == "done: prior"
    assert result.total_cost_usd == 0.625
    completed = store.load(execution_id)
    assert completed["status"] == "completed"
    assert completed["budget"]["node_costs"] == dict.fromkeys(
        [node.id for node in graph.nodes], 0.125,
    )
    assert resumed.resume(execution_id).output == result.output
    assert resumed_provider.calls == ["problem", "queued", "skip-child", "grandchild"]
