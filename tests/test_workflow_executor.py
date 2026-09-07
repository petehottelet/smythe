"""Managed executor calls keep journal identities and never rebill a replay."""

import asyncio

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import BudgetValidationError, Sentinel
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.provider import CompletionResult, OfflineProvider, Provider, ProviderAccountingError, ProviderResponseError
from smythe.registry import Registry
from smythe.tracer import Tracer
from smythe.workflow_provider import WorkflowProviderContext
from smythe.workflow_store import CallKey, SQLiteWorkflowStore, WorkflowError, WorkflowLeaseError
from test_tool_loop import SimpleRuntime


def receipt(call_id="call-a", cost=100_000_000):
    return {"workflow_run_id": "run-a", "workflow_call_id": call_id,
            "workflow_charge_recorded": True, "workflow_replayed": False,
            "cost_nanousd": cost, "cost_is_complete": cost is not None}


def result(text="accepted", *, call_id="call-a", stop_reason="stop"):
    return CompletionResult(text, cost_usd=0.1, stop_reason=stop_reason,
                            native_receipt=receipt(call_id))


class Unbound(Provider):
    async def complete(self, *args):
        pytest.fail("Unbound provider dispatched")

    def budget_estimate_usd(self, model):
        pytest.fail("Legacy reservation queried")

    def requires_explicit_budget_estimate(self, model):
        pytest.fail("Legacy reservation queried")


class Managed(Provider):
    workflow_managed = True

    def __init__(self, outcome):
        self.outcome = outcome

    async def complete(self, *args):
        pytest.fail("Executor must use chat")

    async def chat(self, system, messages, model, tools=None):
        await asyncio.sleep(0)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class UntouchedBudget(Sentinel):
    def reserve(self, *args, **kwargs):
        pytest.fail("Executor must not reserve managed calls")

    def check(self, *args, **kwargs):
        pytest.fail("Executor must not admit managed calls")

    def add_cost(self, *args, **kwargs):
        pytest.fail("Executor must not charge a legacy Sentinel for managed calls")

    def mark_unknown(self, *args, **kwargs):
        pytest.fail("Executor must not mutate legacy accounting for managed calls")

    def release(self, *args, **kwargs):
        pytest.fail("Executor must not release journal reservations")


def graph_for(policy=FailurePolicy.RETRY):
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A", failure_policy=policy, max_retries=3),
        Node(id="b", label="B", depends_on=["a"]),
    ])


def execute(graph, factory, *, parallel=False, **kwargs):
    executor = (AsyncExecutor if parallel else Executor)(
        Unbound(), Registry(), Tracer(), provider_call_factory=factory,
        budget=kwargs.pop("budget", UntouchedBudget(0)), **kwargs,
    )
    if parallel:
        asyncio.run(executor.run(graph))
    else:
        executor.run(graph)
    return executor


@pytest.mark.parametrize("parallel", [False, True])
def test_worker_verifier_and_regenerated_calls_use_current_phase_and_generation(parallel):
    graph = graph_for()
    graph.nodes[1].verifies = "a"
    graph.nodes[1].max_regenerations = 1
    calls = []

    def factory(node, phase, attempt, turn):
        generation = node.metadata.get("execution_generation", 0)
        calls.append((node.id, phase, generation, attempt, turn))
        text = "FAIL" if node.id == "b" and generation == 0 else "PASS"
        return Managed(result(text, call_id=f"{node.id}-{generation}"))

    execute(graph, factory, parallel=parallel)
    assert calls == [("a", "execution", 0, 0, 0), ("b", "verification", 0, 0, 0),
                     ("a", "execution", 1, 0, 0), ("b", "verification", 1, 0, 0)]
    for node in graph.nodes:
        assert node.status is NodeStatus.COMPLETED
        assert node.metadata["cost_usd"] == 0.2
        assert {entry["receipt"]["workflow_call_id"] for entry in node.metadata["native_receipts"]} == {
            f"{node.id}-0", f"{node.id}-1",
        }


@pytest.mark.parametrize("parallel", [False, True])
def test_attempt_and_turn_are_explicit_and_restart_at_each_attempt(parallel):
    graph = graph_for()
    graph.nodes.pop()
    calls = []

    def factory(node, phase, attempt, turn):
        calls.append((node.id, phase, attempt, turn))
        if attempt == 0:
            return Managed(ValueError("pre-dispatch local preparation failed"))
        return Managed(result(stop_reason="pause_turn" if turn == 0 else "stop",
                              call_id=f"attempt-{attempt}-turn-{turn}"))

    execute(graph, factory, parallel=parallel, tool_runtime=SimpleRuntime(tools=[]))
    assert calls == [("a", "execution", 0, 0), ("a", "execution", 1, 0), ("a", "execution", 1, 1)]
    assert graph.nodes[0].metadata["cost_usd"] == 0.2


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("policy", list(FailurePolicy))
@pytest.mark.parametrize("grouped", [False, True])
def test_journal_errors_are_terminal_without_retry_skip_or_release(parallel, policy, grouped):
    error = WorkflowLeaseError("Fenced owner")
    outcome = ExceptionGroup("teardown", [OSError("noise"), error]) if grouped else error
    calls = []

    def factory(node, phase, attempt, turn):
        calls.append((node.id, attempt, turn))
        return Managed(outcome)

    graph = graph_for(policy)
    with pytest.raises(WorkflowLeaseError) as caught:
        execute(graph, factory, parallel=parallel)
    assert caught.value is error
    assert calls == [("a", 0, 0)]
    assert graph.nodes[0].status is NodeStatus.FAILED
    assert graph.nodes[1].status is NodeStatus.PENDING


@pytest.mark.parametrize("parallel", [False, True])
def test_factory_cannot_return_an_unmanaged_provider(parallel):
    with pytest.raises(WorkflowError, match="journal-managed"):
        execute(graph_for(), lambda *args: Unbound(), parallel=parallel)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_failed_native_receipts_are_retained_without_legacy_billing(parallel, unknown):
    native_receipt = receipt(cost=None if unknown else 100_000_000)
    error = (ProviderAccountingError if unknown else ProviderResponseError)(
        "Saved unusable response", receipt=native_receipt,
        billing_result=None if unknown else result(),
    )
    graph = graph_for()
    with pytest.raises(type(error)) as caught:
        execute(graph, lambda *args: Managed(error), parallel=parallel)
    assert caught.value is error
    assert graph.nodes[0].metadata["native_receipts"] == [
        {"phase": "execution", "receipt": native_receipt},
    ]
    assert graph.nodes[0].metadata["response_error"]
    if unknown:
        assert graph.nodes[0].metadata["accounting_invalid"]
    else:
        assert graph.nodes[0].metadata["cost_usd"] == 0.1


@pytest.mark.parametrize("parallel", [False, True])
def test_replay_can_reenter_old_native_marker_without_erasing_evidence(parallel):
    graph = graph_for()
    graph.nodes.pop()
    node = graph.nodes[0]
    node.metadata.update(response_error={"type": "ProviderAccountingError"}, accounting_invalid=True,
                         native_receipts=[{"phase": "execution", "receipt": receipt()}])
    calls = []

    def factory(*args):
        calls.append(args)
        return Managed(result())

    execute(graph, factory, parallel=parallel)
    assert len(calls) == 1
    assert node.status is NodeStatus.COMPLETED and node.result == "accepted"
    assert node.metadata["response_error"] and node.metadata["accounting_invalid"]
    assert node.metadata["cost_usd"] == 0.1  # repeated receipt, one logical bill


@pytest.mark.parametrize("parallel", [False, True])
def test_plain_unjournaled_accounting_does_not_bypass_reconciliation(parallel):
    graph = graph_for()
    graph.nodes[0].metadata["accounting_invalid"] = True
    with pytest.raises(BudgetValidationError):
        execute(graph, lambda *args: pytest.fail("dispatched"), parallel=parallel)


@pytest.mark.parametrize("parallel", [False, True])
def test_new_unjournaled_numeric_error_stays_blocked_despite_old_native_marker(parallel):
    graph = graph_for()
    graph.nodes[0].metadata.update(
        response_error={"type": "ProviderAccountingError"}, accounting_invalid=True,
        native_receipts=[{"phase": "execution", "receipt": receipt()}],
    )
    invalid = result()
    invalid.cost_usd = float("nan")
    with pytest.raises(BudgetValidationError):
        execute(graph, lambda *args: Managed(invalid), parallel=parallel)
    assert graph.nodes[0].metadata["workflow_accounting_invalid"] is True
    graph.nodes[0].status = NodeStatus.PENDING
    with pytest.raises(BudgetValidationError, match="Unjournaled"):
        execute(graph, lambda *args: pytest.fail("Replayed over unjournaled invalid usage"), parallel=parallel)


def test_parallel_attempt_identity_is_isolated_between_nodes():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A", failure_policy=FailurePolicy.RETRY, max_retries=1), Node(id="b", label="B"),
    ])
    calls = []

    def factory(node, phase, attempt, turn):
        calls.append((node.id, attempt, turn))
        return Managed(ValueError("prepare") if node.id == "a" and attempt == 0 else result())

    execute(graph, factory, parallel=True)
    assert sorted(calls) == [("a", 0, 0), ("a", 1, 0), ("b", 0, 0)]


class Review:
    def __init__(self, outcome):
        self.outcome = outcome

    async def review(self, graph, node, *, task, revisions_remaining):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("outcome,applied", [
    (None, False), (Revision(), False), (ValueError("local-only review failure"), False),
    (ExceptionGroup("ordinary", [ValueError("local")]), False),
    (Revision(drop_node_ids=("a",)), False),
    (Revision(add_nodes=(Node(id="new", label="New", depends_on=["a"]),)), True),
])
def test_supervision_disposition_follows_mutation_or_declined_review(outcome, applied):
    graph = graph_for()
    node = graph.nodes[0]
    node.status, node.result = NodeStatus.COMPLETED, "done"
    updates = []
    executor = Executor(Unbound(), Registry(), Tracer(), supervisor=Review(outcome), max_revisions=1,
                        on_supervision_update=lambda n, changed: updates.append(
                            (n.id, changed, [item.id for item in graph.nodes], executor.revisions_used)))
    assert asyncio.run(executor.maybe_revise(node, graph)) is applied
    assert updates == [("a", applied, ["a", "b", "new"] if applied else ["a", "b"], int(applied))]


@pytest.mark.parametrize("grouped", [False, True])
def test_supervision_journal_failure_has_no_disposition(grouped):
    error = WorkflowLeaseError("lost")
    graph = graph_for()
    graph.nodes[0].status = NodeStatus.COMPLETED
    executor = Executor(
        Unbound(), Registry(), Tracer(), max_revisions=1,
        supervisor=Review(ExceptionGroup("wrapped", [error]) if grouped else error),
        on_supervision_update=lambda *args: pytest.fail("Consumed failed review"),
    )
    with pytest.raises(WorkflowLeaseError) as caught:
        asyncio.run(executor.maybe_revise(graph.nodes[0], graph))
    assert caught.value is error


@pytest.mark.parametrize("parallel", [False, True])
def test_disposition_persistence_failure_is_terminal_after_applied_revision(parallel):
    graph = graph_for()
    calls = []
    cause = OSError("disk")

    def factory(node, *args):
        calls.append(node.id)
        return Managed(result())

    def persist(node, applied):
        assert applied and [n.id for n in graph.nodes] == ["a", "b", "new"]
        raise cause

    with pytest.raises(WorkflowError) as caught:
        execute(graph, factory, parallel=parallel, max_revisions=1,
                supervisor=Review(Revision(add_nodes=(Node(id="new", label="New", depends_on=["a"]),))),
                on_supervision_update=persist)
    assert caught.value.__cause__ is cause
    assert calls == ["a"] and graph.nodes[0].status is NodeStatus.COMPLETED


@pytest.mark.parametrize("parallel", [False, True])
def test_completed_node_checkpoint_failure_cannot_buy_a_retry(parallel):
    calls = []
    graph = graph_for()

    def factory(*args):
        calls.append(args)
        return Managed(result())

    def persist(node):
        raise WorkflowLeaseError("checkpoint lease lost")

    with pytest.raises(WorkflowLeaseError):
        execute(graph, factory, parallel=parallel, on_node_update=persist)
    assert len(calls) == 1 and graph.nodes[1].status is NodeStatus.PENDING


def test_managed_projection_observes_receipt_without_admission_calls():
    projected = []

    class Projection(UntouchedBudget):
        workflow_managed = True

        def add_cost(self, node_id, completion, **kwargs):
            projected.append((node_id, completion.native_receipt["workflow_call_id"]))
            return 0.1

    graph = graph_for()
    graph.nodes.pop()
    execute(graph, lambda *args: Managed(result()), budget=Projection(0))
    assert projected == [("a", "call-a")]
    assert graph.nodes[0].metadata["cost_usd"] == 0.1


def test_async_cancellation_never_releases_a_managed_reservation():
    async def scenario():
        entered = asyncio.Event()

        class Blocked(Managed):
            async def chat(self, *args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        executor = AsyncExecutor(Unbound(), Registry(), Tracer(), budget=UntouchedBudget(0),
                                 provider_call_factory=lambda *args: Blocked(None))
        graph = graph_for()
        task = asyncio.create_task(executor.run(graph))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert graph.nodes[0].status is NodeStatus.PENDING

    asyncio.run(scenario())


def test_actual_sqlite_facade_replays_completed_call_after_checkpoint_lag(tmp_path, monkeypatch):
    calls = []
    original = OfflineProvider.complete

    async def observed(self, *args, **kwargs):
        calls.append(args)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(OfflineProvider, "complete", observed)

    async def scenario():
        with SQLiteWorkflowStore(tmp_path / "workflow.db") as store:
            run = store.create_run(None, {}, 0)
            lease = store.acquire_lease(run["run_id"], "test-owner", 120)
            async with WorkflowProviderContext(store, lease, {"main": OfflineProvider()}) as context:
                graph = graph_for()
                graph.nodes.pop()
                graph.nodes[0].metadata["model"] = "offline"

                def factory(node, phase, attempt, turn):
                    return context.for_call("main", CallKey(phase, node.id, attempt=attempt, turn=turn))

                executor = AsyncExecutor(Unbound(), Registry(), Tracer(), budget=UntouchedBudget(0),
                                         provider_call_factory=factory)
                await executor.run(graph)
                text = graph.nodes[0].result
                graph.nodes[0].status, graph.nodes[0].result = NodeStatus.PENDING, None
                await executor.run(graph)
                assert graph.nodes[0].result == text
                receipts = graph.nodes[0].metadata["native_receipts"]
                assert [row["receipt"]["workflow_replayed"] for row in receipts] == [False, True]
                assert receipts[0]["receipt"]["workflow_call_id"] == receipts[1]["receipt"]["workflow_call_id"]
                assert graph.nodes[0].metadata["cost_usd"] == 0

    asyncio.run(scenario())
    assert len(calls) == 1
