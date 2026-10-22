"""Tests for the AsyncExecutor — concurrency, ordering, and deadlock detection."""

import asyncio

import pytest

from fakes import Gate, ManualClock
from helpers import FailingProvider
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.executor_base import NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.tracer import Tracer


class SlowMockProvider(Provider):
    """Provider that sleeps to simulate latency, recording call order."""

    def __init__(self, delay: float = 0.1) -> None:
        self._delay = delay
        self.call_order: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        node_label = prompt.split("\n")[0]
        await asyncio.sleep(self._delay)
        self.call_order.append(node_label)
        return CompletionResult(
            text=f"result for: {node_label}",
            prompt_tokens=10,
            completion_tokens=5,
        )


def _make_executor(
    provider: Provider | None = None, budget: Sentinel | None = None,
) -> tuple[AsyncExecutor, Tracer]:
    tracer = Tracer()
    registry = Registry()
    p = provider or SlowMockProvider(delay=0.0)
    executor = AsyncExecutor(provider=p, registry=registry, tracer=tracer, budget=budget)
    return executor, tracer


@pytest.mark.asyncio
async def test_parallel_nodes_run_concurrently():
    provider = ConcurrencyTrackingProvider()
    executor, _ = _make_executor(provider)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b")
    c = Node(label="C", id="c")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b, c])

    await executor.run(graph)

    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)
    assert provider.peak == 3


@pytest.mark.asyncio
async def test_serial_dependencies_respected():
    provider = SlowMockProvider(delay=0.01)
    executor, _ = _make_executor(provider)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    c = Node(label="C", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    await executor.run(graph)

    assert provider.call_order == ["A", "B", "C"]
    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)


@pytest.mark.asyncio
async def test_fork_join_ordering():
    """Parallel roots complete before the join node runs."""
    provider = SlowMockProvider(delay=0.01)
    executor, _ = _make_executor(provider)

    r1 = Node(label="R1", id="r1")
    r2 = Node(label="R2", id="r2")
    j = Node(label="Join", id="j", depends_on=["r1", "r2"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[r1, r2, j])

    await executor.run(graph)

    assert "Join" == provider.call_order[-1]
    assert set(provider.call_order[:2]) == {"R1", "R2"}


@pytest.mark.asyncio
async def test_dependency_results_forwarded():
    provider = SlowMockProvider(delay=0.0)
    executor, _ = _make_executor(provider)

    a = Node(label="First step", id="a")
    b = Node(label="Second step", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    await executor.run(graph)

    assert "result for: First step" in a.result
    assert b.result is not None


@pytest.mark.asyncio
async def test_deadlock_detection():
    executor, _ = _make_executor()

    a = Node(label="A", id="a", depends_on=["b"])
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError, match="Deadlock"):
        await executor.run(graph)


# --- Failure policy tests ---


@pytest.mark.asyncio
async def test_failure_policy_halt_raises():
    """HALT policy (default) propagates the exception."""
    provider = FailingProvider(failures=999)
    executor, _ = _make_executor(provider)

    node = Node(label="Doomed", id="d", failure_policy=FailurePolicy.HALT)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await executor.run(graph)
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_failure_policy_skip():
    """SKIP policy marks the node as SKIPPED and continues execution."""
    provider = FailingProvider(failures=999, fail_labels={"Flaky"})
    executor, _ = _make_executor(provider)

    a = Node(label="Flaky", id="a", failure_policy=FailurePolicy.SKIP)
    b = Node(label="Downstream", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    await executor.run(graph)

    assert a.status == NodeStatus.SKIPPED
    assert b.status == NodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_failure_policy_retry_succeeds():
    """RETRY policy retries up to max_retries and succeeds if the error clears."""
    provider = FailingProvider(failures=1)
    executor, _ = _make_executor(provider)

    node = Node(
        label="Retryable", id="r",
        failure_policy=FailurePolicy.RETRY,
        max_retries=3,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    await executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert "ok:" in node.result


@pytest.mark.asyncio
async def test_failure_policy_retry_exhausted():
    """RETRY policy raises after exhausting all retries."""
    provider = FailingProvider(failures=999)
    executor, _ = _make_executor(provider)

    node = Node(
        label="Always-fail", id="af",
        failure_policy=FailurePolicy.RETRY,
        max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await executor.run(graph)
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_cascading_failure_preserves_original_exception():
    """When nodes are blocked by upstream failure, the original error is raised."""
    provider = FailingProvider(failures=999, fail_labels={"Root"})
    executor, _ = _make_executor(provider)

    a = Node(label="Root", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Child", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await executor.run(graph)


@pytest.mark.asyncio
async def test_retry_count_means_additional_retries():
    """max_retries is interpreted as retries beyond initial attempt."""
    provider = FailingProvider(failures=2)
    executor, _ = _make_executor(provider)
    node = Node(
        label="EventuallyOk", id="eo",
        failure_policy=FailurePolicy.RETRY,
        max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    await executor.run(graph)
    assert node.status == NodeStatus.COMPLETED


# --- Budget reservation tests ---


@pytest.mark.asyncio
async def test_partial_reservation_rollback_on_budget_exceeded():
    """When reserve() fails mid-wave, earlier reservations are rolled back."""
    budget = Sentinel(max_budget_usd=0.010, cost_per_token=0.000003)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b")
    c = Node(label="C", id="c")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b, c])

    executor, _ = _make_executor(
        provider=SlowMockProvider(delay=0.0),
        budget=budget,
    )
    executor._estimated_tokens_per_node = 2000

    with pytest.raises(SentinelAlert):
        await executor.run(graph)

    assert budget.total_cost_usd == 0.0, (
        f"Expected $0 after rollback but got ${budget.total_cost_usd}"
    )
    assert budget._reservations == {}


@pytest.mark.asyncio
async def test_budget_accounting_after_successful_wave():
    """Budget tracks actual cost correctly after a parallel wave completes."""
    budget = Sentinel(max_budget_usd=1.0, cost_per_token=0.000003)

    a = Node(label="A", id="a")
    b = Node(label="B", id="b")
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[a, b])

    executor, _ = _make_executor(
        provider=SlowMockProvider(delay=0.0),
        budget=budget,
    )

    await executor.run(graph)

    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)
    assert budget.total_cost_usd > 0
    assert budget._reservations == {}


class ConcurrencyTrackingProvider(Provider):
    """Provider that records the peak number of in-flight calls."""

    def __init__(self, delay: float = 0.02) -> None:
        self._delay = delay
        self._in_flight = 0
        self.peak = 0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        self._in_flight += 1
        self.peak = max(self.peak, self._in_flight)
        await asyncio.sleep(self._delay)
        self._in_flight -= 1
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


@pytest.mark.asyncio
async def test_max_concurrency_caps_in_flight_calls():
    provider = ConcurrencyTrackingProvider()
    tracer = Tracer()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=tracer, max_concurrency=2,
    )

    nodes = [Node(label=f"N{i}", id=f"n{i}") for i in range(6)]
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes)

    await executor.run(graph)

    assert all(n.status == NodeStatus.COMPLETED for n in graph.nodes)
    assert provider.peak <= 2


@pytest.mark.asyncio
async def test_max_concurrency_bounds_admitted_node_tasks():
    """A wide graph must not create one waiting coroutine per node."""
    gate = asyncio.Event()

    class GatedProvider(Provider):
        async def complete(self, system, prompt, model):
            await gate.wait()
            return CompletionResult(text="ok")

    executor = AsyncExecutor(
        provider=GatedProvider(), registry=Registry(), tracer=Tracer(),
        max_concurrency=2,
    )
    graph = ExecutionGraph(
        topology=[Topology.FORK_JOIN],
        nodes=[Node(label=f"N{i}", id=f"n{i}") for i in range(100)],
    )
    run_task = asyncio.create_task(executor.run(graph))
    try:
        for _ in range(100):
            admitted = [
                task for task in asyncio.all_tasks()
                if task.get_name().startswith("smythe-node-")
            ]
            running = sum(
                node.status == NodeStatus.RUNNING for node in graph.nodes
            )
            if len(admitted) == 2 and running == 2:
                break
            await asyncio.sleep(0)
        assert len(admitted) == 2
        assert running == 2
    finally:
        gate.set()
        await run_task


@pytest.mark.asyncio
async def test_halt_cancels_and_awaits_siblings_without_starting_queue():
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    calls: list[str] = []

    class HaltProvider(Provider):
        async def complete(self, system, prompt, model):
            label = prompt.split("\n")[0]
            calls.append(label)
            if label == "fail":
                await sibling_started.wait()
                raise RuntimeError("stop now")
            if label == "sibling":
                sibling_started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    sibling_cancelled.set()
                    raise
            return CompletionResult(text="should not run")

    nodes = [
        Node(label="fail", id="fail"),
        Node(label="sibling", id="sibling"),
        Node(label="queued", id="queued"),
    ]
    executor = AsyncExecutor(
        provider=HaltProvider(), registry=Registry(), tracer=Tracer(),
        max_concurrency=2,
    )

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes))

    assert sibling_cancelled.is_set()
    assert calls == ["fail", "sibling"]
    assert nodes[0].status == NodeStatus.FAILED
    assert nodes[1].status == NodeStatus.PENDING
    assert nodes[2].status == NodeStatus.PENDING


@pytest.mark.asyncio
async def test_finalization_failure_after_billing_is_not_retried():
    class CountingProvider(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(text="paid response")

    provider = CountingProvider()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=Tracer(), max_concurrency=1,
    )
    executor.finalize_node_result = lambda node, result: (_ for _ in ()).throw(
        OSError("disk full")
    )
    node = Node(
        label="Paid image", id="image",
        failure_policy=FailurePolicy.RETRY, max_retries=3,
    )

    with pytest.raises(NodeFinalizationError, match="disk full"):
        await executor.run(
            ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
        )

    assert provider.calls == 1
    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_unbounded_concurrency_when_cap_is_none():
    provider = ConcurrencyTrackingProvider()
    tracer = Tracer()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=tracer, max_concurrency=None,
    )

    nodes = [Node(label=f"N{i}", id=f"n{i}") for i in range(4)]
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=nodes)

    await executor.run(graph)

    assert provider.peak == 4


def test_max_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="max_concurrency"):
        AsyncExecutor(
            provider=SlowMockProvider(),
            registry=Registry(),
            tracer=Tracer(),
            max_concurrency=0,
        )


@pytest.mark.asyncio
async def test_node_timeout_fails_node():
    provider = SlowMockProvider(delay=5.0)
    executor, _ = _make_executor(provider)

    node = Node(label="slow", id="slow", timeout_s=0.05)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(TimeoutError, match="'slow' timed out after 0.05s"):
        await executor.run(graph)

    assert node.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_node_timeout_respects_skip_policy():
    provider = SlowMockProvider(delay=0.2)
    executor, _ = _make_executor(provider)

    slow = Node(
        label="slow", id="slow", timeout_s=0.05, failure_policy=FailurePolicy.SKIP,
    )
    downstream = Node(label="after", id="after", depends_on=["slow"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[slow, downstream])

    await executor.run(graph)

    assert slow.status == NodeStatus.SKIPPED
    assert downstream.status == NodeStatus.COMPLETED


# ---------------------------------------------------------------------------
# Cost-aware reservations (parallel image budgeting)
# ---------------------------------------------------------------------------

def test_provider_cost_hint_drives_reservations():
    """Per-image pricing must be reflected in reservations, not token math."""
    import asyncio as _asyncio

    from smythe.budget import Sentinel, SentinelAlert

    class PricedProvider(Provider):
        cost_estimate_per_call = 0.04

        async def complete(self, system, prompt, model):
            return CompletionResult(text="img", cost_usd=0.04)

    nodes = [Node(id=f"n{i}", label=f"image {i}") for i in range(3)]
    graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)
    executor = AsyncExecutor(
        provider=PricedProvider(), registry=Registry(), tracer=Tracer(),
        budget=Sentinel(max_budget_usd=0.10), artifact_dir=None,
    )
    # 3 x $0.04 reservations exceed the $0.10 cap up front — the wave is
    # refused instead of overshooting mid-flight.
    with pytest.raises(SentinelAlert):
        _asyncio.run(executor.run(graph))


def test_node_estimated_cost_overrides_hint():
    import asyncio as _asyncio

    from smythe.budget import Sentinel

    class PricedProvider(Provider):
        cost_estimate_per_call = 9.99  # would blow any cap

        async def complete(self, system, prompt, model):
            return CompletionResult(text="img", cost_usd=0.01)

    node = Node(id="cheap", label="tiny image")
    node.metadata["estimated_cost_usd"] = 0.02
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor = AsyncExecutor(
        provider=PricedProvider(), registry=Registry(), tracer=Tracer(),
        budget=Sentinel(max_budget_usd=0.05), artifact_dir=None,
    )
    _asyncio.run(executor.run(graph))
    assert node.status == NodeStatus.COMPLETED


def test_unpriced_image_budget_fails_before_any_async_admission():
    """A generic text-token estimate must never admit paid image calls."""
    import asyncio as _asyncio

    from smythe.budget import BudgetEstimateRequired, Sentinel

    class UnpricedImageProvider(Provider):
        def __init__(self):
            self.calls = 0

        def requires_explicit_budget_estimate(self, model: str) -> bool:
            return True

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(text="img", cost_usd=0.05)

    provider = UnpricedImageProvider()
    nodes = [
        Node(id="first", label="image 1", metadata={"model": "image-model"}),
        Node(id="later", label="image 2", metadata={"model": "image-model"}),
    ]
    graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)
    executor = AsyncExecutor(
        provider=provider,
        registry=Registry(),
        tracer=Tracer(),
        budget=Sentinel(max_budget_usd=1.0),
        max_concurrency=2,
        artifact_dir=None,
    )

    with pytest.raises(BudgetEstimateRequired):
        _asyncio.run(executor.run(graph))
    assert provider.calls == 0
    assert all(node.status == NodeStatus.PENDING for node in nodes)


def test_unbudgeted_unpriced_image_does_not_consume_text_token_estimate():
    import asyncio as _asyncio

    from smythe.budget import Sentinel

    class UnpricedImageProvider(Provider):
        def requires_explicit_budget_estimate(self, model: str) -> bool:
            return True

        async def complete(self, system, prompt, model):
            return CompletionResult(
                text="img",
                prompt_tokens=100,
                completion_tokens=900,
                cost_usd=0.0,
                cost_usd_unknown=True,
            )

    budget = Sentinel(max_budget_usd=None, cost_per_token=1.0)
    node = Node(id="image", label="image", metadata={"model": "image-model"})
    executor = AsyncExecutor(
        provider=UnpricedImageProvider(),
        registry=Registry(),
        tracer=Tracer(),
        budget=budget,
        max_concurrency=1,
        artifact_dir=None,
    )

    _asyncio.run(
        executor.run(ExecutionGraph(topology=[Topology.SERIAL], nodes=[node]))
    )
    assert budget.total_cost_usd == 0.0
    assert not budget.cost_is_complete
    assert node.metadata["cost_usd_unknown"] is True


# --- Elapsed time, waits and attempts ---


def _manual_tracer() -> tuple[ManualClock, Tracer]:
    clock = ManualClock()
    return clock, Tracer(clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9)


class ClockedProvider(Provider):
    """Each call moves a manual clock forward by its label's duration."""

    def __init__(self, clock: ManualClock, seconds: dict[str, float]) -> None:
        self._clock = clock
        self._seconds = seconds

    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        self._clock.advance(self._seconds[label])
        return CompletionResult(text=f"ok: {label}", prompt_tokens=1, completion_tokens=1)


@pytest.mark.asyncio
async def test_overlapping_nodes_record_more_node_time_than_elapsed_time():
    """Two nodes run together for 3s, then a join runs for 1s: 7s of node time in 4s."""
    clock, tracer = _manual_tracer()
    gate = Gate()
    started: list[str] = []

    class OverlappingProvider(Provider):
        async def complete(self, system, prompt, model):
            label = prompt.split("\n")[0]
            started.append(label)
            if label == "Join":
                clock.advance(1)
            else:
                await gate.wait()
            return CompletionResult(text=f"ok: {label}", prompt_tokens=1, completion_tokens=1)

    executor = AsyncExecutor(
        provider=OverlappingProvider(), registry=Registry(), tracer=tracer, max_concurrency=2,
    )
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B"),
        Node(id="join", label="Join", depends_on=["a", "b"]),
    ])
    run = asyncio.create_task(executor.run(graph))
    for _ in range(1000):
        if len(started) == 2:
            break
        await asyncio.sleep(0)
    assert sorted(started) == ["A", "B"]
    clock.advance(3)
    gate.release()
    await run

    timing = tracer.timing()
    assert sorted(span.duration_ms for span in tracer.spans) == [1000, 3000, 3000]
    assert timing["summed_node_ms"] == timing["phases_ms"]["execution"] == 7000
    assert timing["elapsed_ms"] == 4000
    assert timing["phases_ms"]["waiting"] == 0


@pytest.mark.asyncio
async def test_queued_node_records_its_admission_wait():
    clock, tracer = _manual_tracer()
    executor = AsyncExecutor(
        provider=ClockedProvider(clock, {"A": 2, "B": 1}), registry=Registry(),
        tracer=tracer, max_concurrency=1,
    )
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B"),
    ])

    await executor.run(graph)

    timing = tracer.timing()
    assert timing["waiting_ms"]["admission"] == 2000
    assert timing["phases_ms"]["waiting"] == 2000
    assert timing["elapsed_ms"] == 3000
    assert timing["summed_node_ms"] == 3000


@pytest.mark.asyncio
async def test_retry_attempts_record_their_index_outcome_and_backoff_wait(monkeypatch):
    tracer = Tracer()
    executor = AsyncExecutor(
        provider=FailingProvider(failures=1), registry=Registry(), tracer=tracer,
    )
    monkeypatch.setattr(executor, "retry_delay_s", lambda attempt: 0.05 if attempt else 0.0)
    node = Node(label="Retryable", id="r", failure_policy=FailurePolicy.RETRY, max_retries=1)

    await executor.run(ExecutionGraph(topology=[Topology.SERIAL], nodes=[node]))

    assert [(span.attempt, span.outcome) for span in tracer.spans] == [
        (0, "failed"), (1, "completed"),
    ]
    waited = tracer.timing()["waiting_ms"]["retry_backoff"]
    # The event loop may wake up to one clock tick early.
    assert waited >= 30


@pytest.mark.asyncio
async def test_supervisor_review_that_holds_ready_work_is_a_supervision_wait():
    clock, tracer = _manual_tracer()

    class SlowReview(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            clock.advance(5)
            return None

    executor = AsyncExecutor(
        provider=ClockedProvider(clock, {"A": 1, "B": 1}), registry=Registry(),
        tracer=tracer, supervisor=SlowReview(), max_revisions=1,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
    ])

    await executor.run(graph)

    timing = tracer.timing()
    # Both reviews take 5s; only the first held B, which was ready to start.
    assert timing["phases_ms"]["supervision"] == 10000
    assert timing["waiting_ms"] == {"retry_backoff": 0, "admission": 0, "supervision": 5000}
    assert timing["elapsed_ms"] == 12000


@pytest.mark.parametrize(("work_s", "review_s"), [(0.3, 0.6), (0.6, 1.1)])
@pytest.mark.asyncio
async def test_review_that_holds_a_node_never_makes_its_admission_wait_negative(
    work_s, review_s,
):
    """B becomes ready at ``work_s`` and is held until the review ends.

    Removing the review from B's admission wait must not leave a rounding
    error below zero that fails the run.
    """
    clock, tracer = _manual_tracer()

    class SlowReview(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            clock.advance(review_s)
            return None

    executor = AsyncExecutor(
        provider=ClockedProvider(clock, {"A": work_s, "B": 0.1}), registry=Registry(),
        tracer=tracer, supervisor=SlowReview(), max_revisions=1,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
    ])

    await executor.run(graph)

    assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 2
    assert tracer.timing()["waiting_ms"] == {
        "retry_backoff": 0, "admission": 0, "supervision": review_s * 1000,
    }


@pytest.mark.asyncio
async def test_node_blocked_again_by_a_regeneration_records_no_admission_wait():
    """The consumer is ready once the judge first answers, then waits on the rerun.

    Slots are free throughout, so the time it spends blocked on the
    regenerated draft and judge is dependency time, not an admission wait.
    """
    clock, tracer = _manual_tracer()
    calls: list[str] = []

    class FailOnceJudge(ClockedProvider):
        async def complete(self, system, prompt, model):
            result = await super().complete(system, prompt, model)
            label = prompt.split("\n")[0]
            calls.append(label)
            if label == "judge":
                verdict = "FAIL" if calls.count("judge") == 1 else "PASS"
                return CompletionResult(text=verdict, prompt_tokens=1, completion_tokens=1)
            return result

    executor = AsyncExecutor(
        provider=FailOnceJudge(clock, {"draft": 2, "judge": 1, "consumer": 0.5}),
        registry=Registry(), tracer=tracer, max_concurrency=4,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
        Node(id="consumer", label="consumer", depends_on=["draft", "judge"]),
    ])

    await executor.run(graph)

    assert calls == ["draft", "judge", "draft", "judge", "consumer"]
    timing = tracer.timing()
    assert timing["elapsed_ms"] == timing["summed_node_ms"] == 6500
    assert timing["waiting_ms"] == {"retry_backoff": 0, "admission": 0, "supervision": 0}
    assert timing["phases_ms"]["waiting"] == 0


@pytest.mark.asyncio
async def test_node_rewired_behind_an_inserted_step_records_no_admission_wait():
    """B is ready after A, then a revision makes it wait for an inserted fix."""
    clock, tracer = _manual_tracer()

    class InsertFix(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            return Revision(
                add_nodes=(Node(id="fix", label="Fix", depends_on=["a"]),),
                rewire={"b": ("fix",)},
            )

    executor = AsyncExecutor(
        provider=ClockedProvider(clock, {"A": 1, "Fix": 4, "B": 1}), registry=Registry(),
        tracer=tracer, supervisor=InsertFix(), max_revisions=1, max_concurrency=4,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
    ])

    await executor.run(graph)

    assert [span.node_id for span in tracer.spans if span.phase] == ["a", "fix", "b"]
    timing = tracer.timing()
    assert timing["elapsed_ms"] == timing["summed_node_ms"] == 6000
    assert timing["waiting_ms"] == {"retry_backoff": 0, "admission": 0, "supervision": 0}
    assert timing["phases_ms"]["waiting"] == 0


@pytest.mark.asyncio
async def test_node_still_ready_after_a_revision_keeps_its_admission_wait():
    """C waits for the single slot from the start; a revision does not reset that wait."""
    clock, tracer = _manual_tracer()

    class AddExtra(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            return Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=["a"]),))

    executor = AsyncExecutor(
        provider=ClockedProvider(clock, {"A": 1, "C": 2, "Extra": 1}), registry=Registry(),
        tracer=tracer, supervisor=AddExtra(), max_revisions=1, max_concurrency=1,
    )
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
        Node(id="a", label="A"), Node(id="c", label="C"),
    ])

    await executor.run(graph)

    assert [span.node_id for span in tracer.spans if span.phase] == ["a", "c", "extra"]
    timing = tracer.timing()
    # C waits 1s behind A; Extra, ready at 1s, waits 2s behind C.
    assert timing["waiting_ms"] == {"retry_backoff": 0, "admission": 3000, "supervision": 0}
    assert timing["elapsed_ms"] == 4000


class SteppingClock:
    """A monotonic clock a test can step backwards, as a faulty clock might."""

    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


@pytest.mark.asyncio
async def test_clock_stepping_backwards_during_a_review_does_not_fail_the_run():
    clock = SteppingClock()

    class BackwardsReview(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            clock.value -= 5
            return None

    tracer = Tracer(clock=clock)
    executor = AsyncExecutor(
        provider=SlowMockProvider(delay=0.0), registry=Registry(), tracer=tracer,
        supervisor=BackwardsReview(), max_revisions=1,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
    ])

    await executor.run(graph)

    assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 2
    timing = tracer.timing()
    assert timing["phases_ms"]["supervision"] == 0
    assert timing["waiting_ms"] == {"retry_backoff": 0, "admission": 0, "supervision": 0}
