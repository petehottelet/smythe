"""Provider capacity limits: admission, call spacing, cooldowns and fair turns.

A ``CapacityPolicy`` limits each provider type and model: the async executor
runs at most ``max_in_flight`` of a model's nodes at once and passes over a
ready node with no free slot, still ready, instead of stopping the run; every
call, tool turn and retry starts at least ``min_interval_s`` after the
previous one and after any ``Retry-After`` cooldown. Where a test measures
time, its clock is manual and the injected ``sleep`` moves it, so every wait
is exact.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager

import pytest

import smythe.executor_base as executor_base
from fakes import RATE_LIMITED, Gate, HttpError, ManualClock
from smythe import CapacityLimit, CapacityPolicy, ModelProfile
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.checkpoint import CheckpointStore
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.registry import Registry
from smythe.scheduling import _capacity_limit, _CapacityLimiter
from smythe.swarm import Swarm
from smythe.synthesizer import MERGE_SYSTEM_PROMPT, SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.tools import ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec
from smythe.tracer import Tracer
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import SQLiteWorkflowStore

NO_WAITS = {"retry_backoff": 0, "admission": 0, "supervision": 0, "capacity": 0, "drain": 0}


def manual_tracer() -> tuple[ManualClock, Tracer]:
    clock = ManualClock()
    return clock, Tracer(clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9)


def node(node_id: str, model: str = "m", **fields) -> Node:
    return Node(id=node_id, label=node_id, metadata={"model": model}, **fields)


def graph(*nodes: Node) -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=list(nodes))


def limits(model: str = "m", **fields) -> CapacityPolicy:
    return CapacityPolicy(limits={model: CapacityLimit(**fields)})


def label_of(system: str, prompt: str) -> str:
    return "merge" if system == MERGE_SYSTEM_PROMPT else prompt.split("\n")[0]


def async_executor(provider, tracer=None, **kwargs) -> AsyncExecutor:
    return AsyncExecutor(provider=provider, registry=Registry(), tracer=tracer or Tracer(),
                         artifact_dir=None, **kwargs)


class Overlapping(Provider):
    """Yields inside every call and records the most calls in flight, per model and in all."""

    def __init__(self, hint: object = None) -> None:
        self.hint = hint
        self.calls: list[str] = []
        self.in_flight: Counter[str] = Counter()
        self.peak: Counter[str] = Counter()
        self.peak_total = 0

    def capacity_hint(self, model):
        return self.hint

    async def complete(self, system, prompt, model):
        self.calls.append(label_of(system, prompt))
        self.in_flight[model] += 1
        self.peak[model] = max(self.peak[model], self.in_flight[model])
        self.peak_total = max(self.peak_total, sum(self.in_flight.values()))
        try:
            for _ in range(3):
                await asyncio.sleep(0)
        finally:
            self.in_flight[model] -= 1
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


class Clocked(Provider):
    """Records each call's start on a manual clock; a call takes its label's seconds.

    *fail* maps a label to the error its first call raises.
    """

    def __init__(self, clock: ManualClock, seconds: dict | None = None,
                 fail: dict | None = None) -> None:
        self.clock, self.seconds, self.fail = clock, seconds or {}, dict(fail or {})
        self.starts: list[tuple[str, float]] = []

    async def complete(self, system, prompt, model):
        label = label_of(system, prompt)
        self.starts.append((label, self.clock.monotonic()))
        error = self.fail.pop(label, None)
        if error is not None:
            raise error
        self.clock.advance(self.seconds.get(label, 0))
        return CompletionResult(text=f"ok: {label}", prompt_tokens=1, completion_tokens=1)


# ---------------------------------------------------------------------------
# Admission: in-flight limits and deferral
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_flight_calls_per_model_never_exceed_the_limit_while_the_global_cap_is_higher():
    provider = Overlapping()
    executor = async_executor(provider, max_concurrency=8, capacity_policy=limits(max_in_flight=2))
    work = graph(*(node(f"m{i}") for i in range(6)), node("n0", "n"), node("n1", "n"))

    await executor.run(work)

    assert all(item.status is NodeStatus.COMPLETED for item in work.nodes)
    assert provider.peak["m"] == 2
    # The other model's nodes ran beside them: the limit held back only "m".
    assert provider.peak["n"] == 2 and provider.peak_total == 4
    # The "m" nodes without a slot were passed over, so "n" work started first.
    assert provider.calls[:4] == ["m0", "m1", "n0", "n1"]


@pytest.mark.asyncio
async def test_without_a_policy_admission_and_waits_are_unchanged():
    class NoHint(Overlapping):
        def capacity_hint(self, model):
            raise AssertionError("a run without a capacity policy reads no hint")

    provider = NoHint()
    clock, tracer = manual_tracer()
    executor = async_executor(provider, tracer, max_concurrency=8)
    work = graph(*(node(f"m{i}") for i in range(6)))

    await executor.run(work)

    assert provider.peak["m"] == 6
    assert provider.calls == [f"m{i}" for i in range(6)]
    assert executor._capacity is None
    assert tracer.timing()["waiting_ms"] == NO_WAITS


@pytest.mark.asyncio
async def test_a_deferred_node_waits_for_capacity_and_never_stops_the_run():
    clock, tracer = manual_tracer()
    provider = Clocked(clock, {"a": 2, "b": 1, "c": 1})
    executor = async_executor(provider, tracer, max_concurrency=4,
                              capacity_policy=limits(max_in_flight=1))
    work = graph(node("a"), node("b"), node("c", "n"))

    await executor.run(work)

    assert [item.status for item in work.nodes] == [NodeStatus.COMPLETED] * 3
    # b had a free slot under max_concurrency but none for "m" until a ended at 3s.
    assert provider.starts == [("a", 0), ("c", 2), ("b", 3)]
    assert tracer.timing()["waiting_ms"] == dict(NO_WAITS, capacity=3000)


@pytest.mark.asyncio
async def test_a_wait_for_a_free_slot_before_deferral_stays_an_admission_wait(monkeypatch):
    """c waits 1s for a slot under max_concurrency, then 2s more for a's slot on "m"."""
    refused = asyncio.Event()

    class Watched(_CapacityLimiter):
        def try_acquire(self, bucket):
            taken = _CapacityLimiter.try_acquire(bucket)
            if not taken:
                refused.set()
            return taken

    monkeypatch.setattr(executor_base, "_CapacityLimiter", Watched)
    clock, tracer = manual_tracer()
    gate = Gate()

    class HeldA(Clocked):
        async def complete(self, system, prompt, model):
            if label_of(system, prompt) == "a":
                await gate.wait()
            return await super().complete(system, prompt, model)

    provider = HeldA(clock, {"b": 1})
    executor = async_executor(provider, tracer, max_concurrency=2,
                              capacity_policy=limits(max_in_flight=1))
    run = asyncio.create_task(executor.run(graph(node("a"), node("b", "n"), node("c"))))
    await asyncio.wait_for(refused.wait(), 10)
    clock.advance(2)
    gate.release()
    await run

    assert provider.starts == [("b", 0), ("a", 3), ("c", 3)]
    assert tracer.timing()["waiting_ms"] == dict(NO_WAITS, admission=1000, capacity=2000)


@pytest.mark.asyncio
async def test_capacity_deferral_reserves_nothing_until_admission_and_keeps_holds():
    events: list[tuple[str, str]] = []

    class Recording(Sentinel):
        def reserve(self, node_id, estimated_cost, *, hard_ceiling=False, from_hold=False):
            events.append(("reserve", node_id))
            super().reserve(node_id, estimated_cost, hard_ceiling=hard_ceiling,
                            from_hold=from_hold)

        def release(self, node_id):
            events.append(("release", node_id))
            super().release(node_id)

    budget = Recording(max_budget_usd=1.0)
    budget.hold("checks", 0.25)
    clock, tracer = manual_tracer()
    executor = async_executor(Clocked(clock, {"a": 1}), tracer, budget=budget,
                              max_concurrency=4, capacity_policy=limits(max_in_flight=1))

    await executor.run(graph(node("a"), node("b")))

    assert tracer.timing()["waiting_ms"]["capacity"] == 1000
    assert [event for event in events if event[1] == "b"] == [("reserve", "b")]
    assert budget.holds == {"checks": 0.25}


@pytest.mark.asyncio
async def test_a_reservation_failure_still_stops_the_run_and_frees_every_slot():
    """Only capacity defers: a node the budget refuses stops admission as before."""
    provider = Overlapping()
    # Room for one 2,000-token reservation, not two.
    budget = Sentinel(max_budget_usd=0.01)
    executor = async_executor(provider, budget=budget, max_concurrency=4,
                              capacity_policy=limits(max_in_flight=1))

    with pytest.raises(SentinelAlert):
        await executor.run(graph(node("a"), node("b", "n")))

    assert provider.calls == []
    assert [executor._capacity.bucket(provider, model).in_flight for model in "mn"] == [0, 0]


# ---------------------------------------------------------------------------
# Call spacing, tool turns, retries and cooldowns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calls_start_min_interval_apart_on_a_manual_clock():
    clock, tracer = manual_tracer()
    provider = Clocked(clock)
    executor = async_executor(provider, tracer, max_concurrency=4, sleep=clock.advance,
                              capacity_policy=limits(min_interval_s=0.5))

    await executor.run(graph(node("a"), node("b"), node("c"), node("other", "n")))

    # Another model's calls are not spaced by "m"'s interval.
    assert provider.starts == [("a", 0), ("b", 0.5), ("c", 1.0), ("other", 1.0)]
    # b waited from 0 to 0.5s, and c, which started its turn at 0.5s, until 1s.
    assert tracer.timing()["waiting_ms"] == dict(NO_WAITS, capacity=1000)


class ToolTurns(Provider):
    """Asks for one tool call, then answers; yields inside every turn."""

    def __init__(self, clock: ManualClock | None = None) -> None:
        self.clock = clock
        self.turns: list[tuple[str, float | None]] = []
        self.in_flight = 0
        self.peak = 0

    async def complete(self, system, prompt, model):
        raise AssertionError("a tool loop calls chat()")

    async def chat(self, system, messages, model, tools=None):
        label = messages[0].content.split("\n")[0]
        self.turns.append((label, self.clock.monotonic() if self.clock else None))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        if len(messages) == 1:
            return CompletionResult(
                text="", prompt_tokens=1, completion_tokens=1, stop_reason="tool_use",
                tool_calls=[ToolCall(id=f"{label}-call", name="lookup", arguments={})],
            )
        return CompletionResult(text=f"ok: {label}", prompt_tokens=1, completion_tokens=1)


class Lookup(ToolRuntime):
    """One tool, whose call yields to the event loop before it answers."""

    def open(self, agent):
        class Session(ToolSession):
            @property
            def tools(self):
                return [ToolSpec(name="lookup", description="Look up", input_schema={})]

            async def call(self, tool_call):
                await asyncio.sleep(0)
                return ToolResult(tool_call_id=tool_call.id, content="found")

        @asynccontextmanager
        async def session():
            yield Session()

        return session()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,order", [
    (None, ["a", "b", "a", "b"]),
    (limits(max_in_flight=1), ["a", "a", "b", "b"]),
], ids=["no-policy", "one-in-flight"])
async def test_tool_turns_count_toward_the_in_flight_limit(policy, order):
    provider = ToolTurns()
    executor = async_executor(provider, max_concurrency=2, tool_runtime=Lookup(),
                              capacity_policy=policy)

    await executor.run(graph(node("a"), node("b")))

    # With one slot, a keeps it through its tool call, so b's first turn waits.
    assert [label for label, _ in provider.turns] == order


def test_every_tool_turn_and_retry_waits_its_turn_in_serial_execution():
    clock, tracer = manual_tracer()

    class FirstTurnFails(ToolTurns):
        async def chat(self, system, messages, model, tools=None):
            if not self.turns:
                self.turns.append(("failed", self.clock.monotonic()))
                raise ConnectionError("dropped")
            return await super().chat(system, messages, model, tools)

    provider = FirstTurnFails(clock)
    executor = Executor(provider=provider, registry=Registry(), tracer=tracer,
                        artifact_dir=None, tool_runtime=Lookup(), sleep=clock.advance,
                        capacity_policy=limits(min_interval_s=1))
    work = graph(node("a", failure_policy=FailurePolicy.RETRY, max_retries=1))

    executor.run(work)

    assert work.nodes[0].status is NodeStatus.COMPLETED
    assert provider.turns == [("failed", 0), ("a", 1), ("a", 2)]
    assert tracer.timing()["waiting_ms"] == dict(NO_WAITS, capacity=2000)


@pytest.mark.asyncio
@pytest.mark.parametrize("cooldown_max_s,start", [
    (60.0, 5), (2.0, 2), (0.0, 0),
], ids=["retry-after", "capped", "off"])
async def test_a_retry_after_cools_the_model_down_for_later_calls(cooldown_max_s, start):
    clock, tracer = manual_tracer()
    limited = HttpError(429, "rate_limit_error", retry_after_s=5).exception()
    provider = Clocked(clock, fail={"a": limited})
    executor = async_executor(provider, tracer, sleep=clock.advance,
                              capacity_policy=CapacityPolicy(cooldown_max_s=cooldown_max_s))
    work = graph(node("a", failure_policy=FailurePolicy.SKIP), node("b", depends_on=["a"]))

    await executor.run(work)

    assert [item.status for item in work.nodes] == [NodeStatus.SKIPPED, NodeStatus.COMPLETED]
    assert provider.starts == [("a", 0), ("b", start)]
    assert tracer.timing()["waiting_ms"]["capacity"] == start * 1000


@pytest.mark.asyncio
async def test_a_cooldown_reaches_a_call_already_waiting_its_turn():
    """b took the start after a's; a's 429 then holds b until the cooldown ends."""
    clock, tracer = manual_tracer()
    waits: list[float] = []

    async def sleep(seconds):
        waits.append(seconds)
        await asyncio.sleep(0)
        clock.advance(seconds)

    limited = HttpError(429, "rate_limit_error", retry_after_s=4).exception()

    class SlowFailure(Clocked):
        async def complete(self, system, prompt, model):
            if label_of(system, prompt) != "a":
                return await super().complete(system, prompt, model)
            self.starts.append(("a", self.clock.monotonic()))
            await asyncio.sleep(0)
            raise limited

    provider = SlowFailure(clock)
    executor = async_executor(provider, tracer, max_concurrency=2, sleep=sleep,
                              capacity_policy=limits(min_interval_s=1))
    work = graph(node("a", failure_policy=FailurePolicy.SKIP), node("b"))

    await executor.run(work)

    assert provider.starts == [("a", 0), ("b", 4)]
    assert waits == [1, 3]
    assert tracer.timing()["waiting_ms"]["capacity"] == 4000


@pytest.mark.asyncio
async def test_a_rate_limit_without_retry_after_sets_no_cooldown():
    clock, tracer = manual_tracer()
    provider = Clocked(clock, fail={"a": RATE_LIMITED.exception()})
    executor = async_executor(provider, tracer, sleep=clock.advance,
                              capacity_policy=CapacityPolicy())
    work = graph(node("a", failure_policy=FailurePolicy.SKIP), node("b", depends_on=["a"]))

    await executor.run(work)

    assert provider.starts == [("a", 0), ("b", 0)]


async def settle() -> None:
    """Let every task that can run go as far as it can."""
    for _ in range(10):
        await asyncio.sleep(0)


class Timers:
    """An injected sleep on a manual clock: a sleep ends once the test moves the clock to its end."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.sleeping: list[tuple[float, asyncio.Future]] = []

    async def sleep(self, seconds: float) -> None:
        ended = asyncio.get_running_loop().create_future()
        self.sleeping.append((self.clock.monotonic() + seconds, ended))
        await ended

    async def advance_to(self, when: float) -> None:
        """Move the clock to *when* and end every sleep due by then."""
        self.clock.advance(when - self.clock.monotonic())
        for end, ended in self.sleeping:
            if end <= when and not ended.done():
                ended.set_result(None)
        await settle()


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting,cancelled,starts,waited_ms", [
    (["b"], "b", [("a", 0), ("d", 1)], 1000),
    (["b", "c"], "b", [("a", 0), ("c", 1), ("d", 2)], 3000),
    (["b", "c"], "c", [("a", 0), ("b", 1), ("d", 2)], 3000),
], ids=["only-waiting", "first-in-line", "second-in-line"])
async def test_a_call_cancelled_while_it_waits_its_turn_claims_no_start(
    waiting, cancelled, starts, waited_ms,
):
    """Later calls are spaced from the calls that started, never from the cancelled one."""
    clock, tracer = manual_tracer()
    timers = Timers(clock)
    limiter = _CapacityLimiter(limits(min_interval_s=1), tracer, timers.sleep)
    provider = Clocked(clock)

    async def call(label):
        async with limiter.call(provider, "m"):
            await provider.complete("system", label, "m")

    await call("a")
    calls = {label: asyncio.create_task(call(label)) for label in waiting}
    await settle()
    gone = calls.pop(cancelled)
    gone.cancel()
    await settle()
    calls["d"] = asyncio.create_task(call("d"))
    await settle()
    for second in (1, 2, 3):
        await timers.advance_to(second)
    await asyncio.wait_for(asyncio.gather(*calls.values()), 10)

    assert gone.cancelled()
    assert provider.starts == starts
    assert tracer.timing()["waiting_ms"]["capacity"] == waited_ms


def test_a_sleep_that_ends_late_spaces_the_next_call_from_the_actual_start():
    clock, tracer = manual_tracer()

    def oversleep(seconds):
        clock.advance(seconds + 0.25)

    limiter = _CapacityLimiter(limits(min_interval_s=1), tracer, oversleep)
    provider = Clocked(clock)

    async def run():
        for label in "abc":
            async with limiter.call(provider, "m"):
                await provider.complete("system", label, "m")

    asyncio.run(run())

    assert provider.starts == [("a", 0), ("b", 1.25), ("c", 2.5)]


@pytest.mark.asyncio
async def test_calls_queued_behind_a_cooldown_each_record_their_wait():
    """Without an interval, b and c both start when the cooldown ends; each waited 4s."""
    clock, tracer = manual_tracer()
    timers = Timers(clock)
    limiter = _CapacityLimiter(CapacityPolicy(), tracer, timers.sleep)
    limited = HttpError(429, "rate_limit_error", retry_after_s=4).exception()
    provider = Clocked(clock, fail={"a": limited})

    async def call(label):
        async with limiter.call(provider, "m"):
            await provider.complete("system", label, "m")

    with pytest.raises(type(limited)):
        await call("a")
    calls = [asyncio.create_task(call(label)) for label in "bc"]
    await settle()
    await timers.advance_to(4)
    await asyncio.wait_for(asyncio.gather(*calls), 10)

    assert provider.starts == [("a", 0), ("b", 4), ("c", 4)]
    assert tracer.timing()["waiting_ms"]["capacity"] == 8000


@pytest.mark.asyncio
async def test_a_node_that_times_out_waiting_its_turn_retries_at_the_next_free_start():
    """c's first wait outlasts its timeout, and so would any wait past 2s, where c's turn is.

    Its retry starts at 2s, 1s after b, the last call that started.
    """
    clock, tracer = manual_tracer()
    sleeps = 0

    async def sleep(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2 or clock.monotonic() + seconds > 2:
            await asyncio.Event().wait()  # Until c's timeout cancels it.
        clock.advance(seconds)

    provider = Clocked(clock)
    executor = async_executor(provider, tracer, max_concurrency=3, sleep=sleep,
                              capacity_policy=limits(min_interval_s=1))
    work = graph(node("a"), node("b"),
                 node("c", timeout_s=0.25, failure_policy=FailurePolicy.RETRY, max_retries=1))

    await executor.run(work)

    assert [item.status for item in work.nodes] == [NodeStatus.COMPLETED] * 3
    assert provider.starts == [("a", 0), ("b", 1), ("c", 2)]
    assert sleeps == 3
    assert tracer.timing()["waiting_ms"]["capacity"] == 2000


# ---------------------------------------------------------------------------
# Round-robin turns across branches
# ---------------------------------------------------------------------------


def two_fan_outs() -> ExecutionGraph:
    """Hubs x and y have completed, and each released three ready leaves."""
    hubs = [node("x"), node("y")]
    for hub in hubs:
        hub.status, hub.result = NodeStatus.COMPLETED, f"{hub.id} done"
    leaves = [node(f"{hub.id}{i}", depends_on=[hub.id]) for hub in hubs for i in (1, 2, 3)]
    return graph(*hubs, *leaves)


@pytest.mark.asyncio
@pytest.mark.parametrize("round_robin,order", [
    (False, ["x1", "x2", "x3", "y1", "y2", "y3"]),
    (True, ["x1", "y1", "x2", "y2", "x3", "y3"]),
], ids=["graph-order", "round-robin"])
async def test_round_robin_admits_ready_nodes_in_turn_across_branches(round_robin, order):
    provider = Overlapping()
    executor = async_executor(provider, max_concurrency=1,
                              capacity_policy=CapacityPolicy(round_robin=round_robin))

    await executor.run(two_fan_outs())

    assert provider.calls == order


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_limit,first", [
    (None, ["x1", "y1", "x2"]), (1, ["x1", "y1", "y2"]),
], ids=["unlimited", "one-slot"])
async def test_round_robin_passes_over_a_branch_whose_model_is_full(slow_limit, first):
    provider = Overlapping()
    work = two_fan_outs()
    for leaf in work.nodes[2:5]:
        leaf.metadata["model"] = "slow"
    policy = CapacityPolicy(limits={"slow": CapacityLimit(max_in_flight=slow_limit)},
                            round_robin=True)
    executor = async_executor(provider, max_concurrency=3, capacity_policy=policy)

    await executor.run(work)

    assert provider.calls[:3] == first
    assert sorted(provider.calls) == ["x1", "x2", "x3", "y1", "y2", "y3"]
    if slow_limit:
        assert provider.peak["slow"] == slow_limit


# ---------------------------------------------------------------------------
# Provider hints and developer values
# ---------------------------------------------------------------------------


def test_developer_values_override_hints_field_by_field():
    hinted = Overlapping(hint={"max_in_flight": 4, "min_interval_s": 0.5})
    assert _capacity_limit(CapacityPolicy(), hinted, "m") == CapacityLimit(4, 0.5)
    policy = CapacityPolicy(limits={
        "m": CapacityLimit(max_in_flight=2),
        ("Overlapping", "m"): CapacityLimit(min_interval_s=0.0),
        ("OfflineProvider", "m"): CapacityLimit(max_in_flight=1),
    })
    # The pair sets the interval and the model the cap, so nothing comes from the hint.
    assert _capacity_limit(policy, hinted, "m") == CapacityLimit(2, 0.0)
    assert _capacity_limit(policy, hinted, "other") == CapacityLimit(4, 0.5)
    assert _capacity_limit(policy, OfflineProvider(), "m") == CapacityLimit(1, None)
    assert _capacity_limit(policy, OfflineProvider(), "other") == CapacityLimit()
    assert Provider.capacity_hint(OfflineProvider(), "m") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,peak", [
    (CapacityPolicy(), 1), (limits(max_in_flight=3), 3),
], ids=["hint", "developer"])
async def test_a_hint_limits_a_run_unless_the_policy_overrides_it(policy, peak):
    provider = Overlapping(hint={"max_in_flight": 1})
    executor = async_executor(provider, max_concurrency=8, capacity_policy=policy)

    await executor.run(graph(*(node(f"m{i}") for i in range(5))))

    assert provider.peak["m"] == peak


@pytest.mark.parametrize("hint", [
    {"max_in_flight": 0}, {"max_in_flight": True}, {"min_interval_s": float("nan")},
    {"max_inflight": 2}, [("max_in_flight", 2)], "fast",
])
def test_a_bad_hint_stops_the_run_before_any_call(hint):
    provider = Overlapping(hint=hint)
    executor = async_executor(provider, capacity_policy=CapacityPolicy())

    with pytest.raises(ValueError, match="capacity_hint"):
        asyncio.run(executor.run(graph(node("a"))))

    assert provider.calls == []



class BadHintFor(Overlapping):
    """Hints nothing for every model but ``bad``, whose hint is not valid."""

    def capacity_hint(self, model):
        return {"max_in_flight": 0} if model == "bad" else None


@pytest.mark.parametrize("parallel", [True, False], ids=["async", "serial"])
def test_a_later_node_s_bad_hint_stops_the_run_before_any_call(parallel):
    provider = BadHintFor()
    work = graph(node("a"), node("b", "bad", depends_on=["a"]))

    with pytest.raises(ValueError, match="capacity_hint"):
        if parallel:
            asyncio.run(async_executor(provider, capacity_policy=CapacityPolicy()).run(work))
        else:
            Executor(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                     capacity_policy=CapacityPolicy()).run(work)

    # The first node's call is never made: every hint is read before any call.
    assert provider.calls == []


@pytest.mark.asyncio
async def test_a_profiled_node_takes_the_limit_of_its_profile_s_provider_and_model():
    run_provider, routed = Overlapping(), Overlapping(hint={"max_in_flight": 1})
    executor = async_executor(run_provider, max_concurrency=8, capacity_policy=CapacityPolicy(),
                              profiles=[ModelProfile("fast", routed, "p", 0.01)])
    work = graph(*(Node(id=f"p{i}", label=f"p{i}", metadata={"profile": "fast"})
                   for i in range(4)))

    await executor.run(work)

    assert all(item.status is NodeStatus.COMPLETED for item in work.nodes)
    assert run_provider.calls == [] and len(routed.calls) == 4
    # The profile's provider hints one call at a time for its model.
    assert routed.peak["p"] == 1

def test_a_swarm_refuses_a_bad_hint_before_planning():
    provider = Overlapping(hint={"max_in_flight": -1})
    swarm = Swarm(model="m", provider=provider, capacity_policy=CapacityPolicy())

    with pytest.raises(ValueError, match="capacity_hint"):
        swarm.execute(Task("Plan this"))

    assert provider.calls == []


# ---------------------------------------------------------------------------
# Policy validation and where a policy is accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [
    lambda: CapacityLimit(max_in_flight=0),
    lambda: CapacityLimit(max_in_flight=1.5),
    lambda: CapacityLimit(max_in_flight=True),
    lambda: CapacityLimit(min_interval_s=-1),
    lambda: CapacityLimit(min_interval_s=float("inf")),
    lambda: CapacityLimit(min_interval_s=True),
    lambda: CapacityPolicy(limits=[("m", CapacityLimit())]),
    lambda: CapacityPolicy(limits={3: CapacityLimit()}),
    lambda: CapacityPolicy(limits={("m",): CapacityLimit()}),
    lambda: CapacityPolicy(limits={"m": {"max_in_flight": 1}}),
    lambda: CapacityPolicy(round_robin="yes"),
    lambda: CapacityPolicy(cooldown_max_s=float("nan")),
    lambda: CapacityPolicy(cooldown_max_s=-1),
])
def test_invalid_policies_are_refused(make):
    with pytest.raises(ValueError):
        make()


def test_a_policy_keeps_its_own_copy_of_its_limits():
    given = {"m": CapacityLimit(max_in_flight=1)}
    policy = CapacityPolicy(limits=given)
    given["m"] = CapacityLimit(max_in_flight=9)
    assert policy.limits["m"].max_in_flight == 1
    with pytest.raises(TypeError):
        policy.limits["m"] = CapacityLimit()
    assert policy == CapacityPolicy(limits={"m": CapacityLimit(max_in_flight=1)})


def test_executors_and_swarm_check_the_policy_and_durable_runs_refuse_it(tmp_path):
    for executor_type in (Executor, AsyncExecutor):
        with pytest.raises(TypeError, match="CapacityPolicy"):
            executor_type(provider=OfflineProvider(), registry=Registry(), tracer=Tracer(),
                          capacity_policy={"m": 1})
        with pytest.raises(ValueError, match="journal-managed"):
            executor_type(provider=OfflineProvider(), registry=Registry(), tracer=Tracer(),
                          provider_call_factory=lambda *args: None,
                          capacity_policy=CapacityPolicy())
    with pytest.raises(TypeError, match="CapacityPolicy"):
        Swarm(model="offline", provider=OfflineProvider(), capacity_policy="fast")
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(WorkflowBindingError, match="^capacity_policy "):
            Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                  capacity_policy=CapacityPolicy())


# ---------------------------------------------------------------------------
# Swarm runs and LLM_MERGE synthesis
# ---------------------------------------------------------------------------


@pytest.fixture
def limited_calls(monkeypatch):
    """Each call a limiter makes, as (limiter id, node id or "merge")."""
    seen: list[tuple[int, str]] = []
    original = _CapacityLimiter.call

    def call(self, provider, model, node=None):
        seen.append((id(self), "merge" if node is None else node.id))
        return original(self, provider, model, node)

    monkeypatch.setattr(_CapacityLimiter, "call", call)
    return seen


class MemoryStore(CheckpointStore):
    """Keeps each saved state as JSON data, as a file store would."""

    def __init__(self) -> None:
        self.states: dict[str, str] = {}

    def save(self, execution_id, state):
        self.states[execution_id] = json.dumps(state)

    def load(self, execution_id):
        saved = self.states.get(execution_id)
        return None if saved is None else json.loads(saved)

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


def merging_swarm(provider: Provider, **kwargs) -> Swarm:
    return Swarm(model="m", provider=provider, artifact_dir=None,
                 synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE),
                 capacity_policy=limits(max_in_flight=1), **kwargs)


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_swarm_sends_every_node_call_and_the_merge_through_one_limiter(
    limited_calls, parallel,
):
    result = merging_swarm(Overlapping(), parallel=parallel).execute(
        graph(node("a"), node("b")))

    assert result.output == "ok"
    assert [name for _, name in limited_calls] == ["a", "b", "merge"]
    assert len({limiter for limiter, _ in limited_calls}) == 1


def test_a_resumed_swarm_sends_its_calls_and_merge_through_a_limiter(limited_calls):
    class FailsOnce(Overlapping):
        async def complete(self, system, prompt, model):
            if label_of(system, prompt) == "b" and "b" not in self.calls:
                self.calls.append("b")
                raise RuntimeError("b failed")
            return await super().complete(system, prompt, model)

    store = MemoryStore()
    swarm = merging_swarm(FailsOnce(), checkpoint_store=store)
    with pytest.raises(RuntimeError, match="b failed"):
        swarm.execute(graph(node("a"), node("b", depends_on=["a"])))
    limited_calls.clear()

    result = swarm.resume(store.list_ids()[0])

    assert result.output == "ok"
    assert [name for _, name in limited_calls] == ["b", "merge"]
    assert len({limiter for limiter, _ in limited_calls}) == 1


def test_a_merge_waits_its_turn_after_the_last_node_call():
    clock, tracer = manual_tracer()
    provider = Clocked(clock)
    limiter = _CapacityLimiter(limits(min_interval_s=2), tracer, clock.advance)
    work = graph(node("a"))

    async def run():
        async with limiter.call(provider, "m"):
            await provider.complete("system", "a", "m")
        return await Synthesizer(SynthesisStrategy.LLM_MERGE).asynthesize(
            work, provider=provider, model="m", capacity=limiter)

    work.nodes[0].status, work.nodes[0].result = NodeStatus.COMPLETED, "a done"
    assert asyncio.run(run()) == "ok: merge"
    assert provider.starts == [("a", 0), ("merge", 2)]
    assert tracer.timing()["waiting_ms"]["capacity"] == 2000


def test_a_custom_synthesizer_is_called_as_before_under_a_capacity_policy():
    class Custom:
        def synthesize(self, graph, *, provider, model, budget, tracer):
            return "custom"

    swarm = Swarm(model="m", provider=Overlapping(), artifact_dir=None, synthesizer=Custom(),
                  capacity_policy=limits(max_in_flight=1))

    assert swarm.execute(graph(node("a"))).output == "custom"
