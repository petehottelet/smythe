"""A sent call that never answered keeps its reservation as unknown cost, when a policy asks.

Under ``RetryPolicy(retain_uncertain_dispatch=True)``, a node call that the
node's timeout, a cancellation or an interruption ends after it was sent, and
before the provider answered, may still be charged. Its reservation becomes the
node's recorded cost, flagged ``cost_usd_unknown`` and ``cost_usd_is_estimate``,
instead of being released, so a version 4 checkpoint keeps it and resume
restores it. Such a timeout is ``UncertainDispatchError``, class
``unknown_outcome``, which a policy retries only when it lists that class. A
failure before the call is sent, or an error the provider raises itself, still
releases the reservation, and without the option nothing changes.

Every call charges ``CALL`` against an estimate of ``ESTIMATE``; both are
binary fractions, so every total here is exact.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import pickle

import pytest

import smythe.async_executor as async_executor_module
from fakes import Crash, Gate, HttpError, ManualClock, crash_when
from smythe import RetryDeadlineError, RetryPolicy, UncertainDispatchError
from smythe._failures import classify_failure, is_terminal
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.checkpoint import CheckpointStore
from smythe.executor import Executor
from smythe.executor_base import ExecutorBase
from smythe.graph import SYNTHESIS_NODE_ID, ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import (
    CompletionResult, Provider, ProviderAccountingCancelledError, ProviderFeatures,
)
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.tools import ChatMessage, ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec
from smythe.tracer import Tracer

MODES = ["serial", "parallel"]
ESTIMATE = 0.25
CALL = 0.125
TIMEOUT = 0.02
HANG = object()  # A call that never answers.
RETAIN = RetryPolicy(retain_uncertain_dispatch=True)


class Script(Provider):
    """Answers each node by its label with that label's outcomes in turn, repeating the last.

    An outcome is reply text, charged ``CALL``; ``HANG``, a call that waits
    until it is cancelled; an exception to raise; or a coroutine function
    whose result or exception is the outcome. ``calls`` lists each call's
    label and ``cancelled`` each call cancelled in flight. It sends each call
    once and declares so.
    """

    def __init__(self, **outcomes) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []
        self.cancelled: list[str] = []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0, tools=True)

    def budget_estimate_usd(self, model):
        return None

    async def complete(self, system, prompt, model):
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    async def chat(self, system, messages, model, tools=None):
        name = messages[0].content.splitlines()[0]
        self.calls.append(name)
        script = self.outcomes.get(name, [f"{name} done"])
        outcome = script[min(self.calls.count(name), len(script)) - 1]
        if outcome is HANG:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return CompletionResult(outcome, cost_usd=CALL)


def node(node_id: str, **options) -> Node:
    options.setdefault("metadata", {"estimated_cost_usd": ESTIMATE})
    return Node(id=node_id, label=node_id, **options)


def make(mode, provider, budget, policy, **options):
    cls = Executor if mode == "serial" else AsyncExecutor
    return cls(provider=provider, registry=Registry(), tracer=options.pop("tracer", Tracer()),
               budget=budget, artifact_dir=None, retry_policy=policy, **options)


def execute(executor, graph: ExecutionGraph) -> None:
    if isinstance(executor, Executor):
        executor.run(graph)
    else:
        asyncio.run(executor.run(graph))


def run(mode, provider, nodes, budget, policy=RETAIN, **options):
    executor = make(mode, provider, budget, policy, **options)
    execute(executor, ExecutionGraph([Topology.SERIAL], list(nodes)))
    return executor


def spans(tracer: Tracer):
    return [(span.attempt, span.outcome, span.metadata.get("failure_class"))
            for span in tracer.spans]


def assert_retained(budget: Sentinel, node_: Node, amount: float) -> None:
    """*node_* records *amount* as unknown, estimated cost, and holds no reservation."""
    assert budget.breakdown()[node_.id] == amount
    assert node_.id not in budget._reservations
    assert budget.cost_is_complete is False and budget.cost_contains_estimates is True
    assert node_.metadata["cost_usd_unknown"] is True
    assert node_.metadata["cost_usd_is_estimate"] is True
    assert node_.metadata["cost_usd"] == amount


class Store(CheckpointStore):
    """Keeps each saved state as detached JSON data, as a file store would."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    def save(self, execution_id, state):
        self.states[execution_id] = json.loads(json.dumps(state))

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


def saved(store: Store) -> tuple[str, dict]:
    (execution_id,) = store.list_ids()
    return execution_id, store.load(execution_id)


def saved_node(state: dict, node_id: str) -> dict:
    return next(item for item in state["graph"]["nodes"] if item["id"] == node_id)


# ---------------------------------------------------------------------------
# The option and its error
# ---------------------------------------------------------------------------


def test_retaining_uncertain_dispatch_is_opt_in_and_recorded_only_when_set():
    assert RetryPolicy().retain_uncertain_dispatch is False
    assert "retain_uncertain_dispatch" not in RetryPolicy(persist_attempts=True).to_dict()
    policy = RetryPolicy(persist_attempts=True, retain_uncertain_dispatch=True)
    assert policy.to_dict()["retain_uncertain_dispatch"] is True
    assert RetryPolicy.from_dict(policy.to_dict()) == policy != RetryPolicy(persist_attempts=True)
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="retain_uncertain_dispatch"):
            RetryPolicy(retain_uncertain_dispatch=value)


def test_an_uncertain_dispatch_is_an_unknown_outcome_timeout_that_is_not_terminal():
    error = UncertainDispatchError("Node 'step' timed out after 5s with its provider call unanswered")

    assert isinstance(error, TimeoutError)
    assert classify_failure(error) == "unknown_outcome" and is_terminal(error) is False
    copied = pickle.loads(pickle.dumps(error))
    assert type(copied) is UncertainDispatchError and str(copied) == str(error)


# ---------------------------------------------------------------------------
# A timeout after the call was sent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_timeout_after_dispatch_keeps_the_reservation_as_unknown_cost(mode):
    provider = Script(step=[HANG])
    budget = Sentinel(10)
    tracer = Tracer()
    step = node("step", timeout_s=TIMEOUT)

    with pytest.raises(UncertainDispatchError, match="'step' timed out after") as caught:
        run(mode, provider, [step], budget, tracer=tracer)

    assert classify_failure(caught.value) == "unknown_outcome"
    assert provider.cancelled == ["step"] and step.status is NodeStatus.FAILED
    assert_retained(budget, step, ESTIMATE)
    assert budget.total_cost_usd == ESTIMATE and budget._reservations == {}
    assert spans(tracer) == [(0, "timed_out", "unknown_outcome")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", [None, RetryPolicy()], ids=["no-policy", "policy-without-option"])
def test_without_the_option_a_timeout_after_dispatch_still_releases(mode, policy):
    provider = Script(step=[HANG])
    budget = Sentinel(10)
    step = node("step", timeout_s=TIMEOUT)

    with pytest.raises(TimeoutError, match="'step' timed out after") as caught:
        run(mode, provider, [step], budget, policy)

    assert type(caught.value) is TimeoutError and classify_failure(caught.value) == "transient"
    assert provider.cancelled == ["step"]
    assert budget.breakdown() == {} and budget._reservations == {}
    assert budget.total_cost_usd == 0 and budget.cost_is_complete is True
    assert "cost_usd_unknown" not in step.metadata


@pytest.mark.parametrize("mode,kept", [("serial", None), ("parallel", 2000 * 0.000003)])
def test_a_call_with_no_reservation_is_marked_unknown_at_no_amount(mode, kept):
    """A serial node without an estimate reserves nothing, so only the marker is kept.

    The async executor reserves its 2,000-token default for the same node,
    and keeps that.
    """
    budget = Sentinel()
    step = node("step", timeout_s=TIMEOUT, metadata={})

    with pytest.raises(UncertainDispatchError):
        run(mode, Script(step=[HANG]), [step], budget)

    assert budget.breakdown() == {"step": kept or 0.0} and budget._reservations == {}
    assert budget.cost_is_complete is False
    assert budget.cost_contains_estimates is (kept is not None)
    assert step.metadata["cost_usd_unknown"] is True
    assert ("cost_usd_is_estimate" in step.metadata) is (kept is not None)


@pytest.mark.parametrize("mode", MODES)
def test_an_interruption_during_a_sent_call_keeps_its_reservation(mode):
    """An interrupt that stops the run mid-call, like ``Crash``, is not the provider's answer."""
    budget = Sentinel(10)
    step = node("step")

    with pytest.raises(Crash):
        run(mode, Script(step=[Crash("interrupted during the call")]), [step], budget)

    assert_retained(budget, step, ESTIMATE)


@pytest.mark.parametrize("mode", MODES)
def test_a_deadline_that_ends_a_sent_call_keeps_its_reservation_too(mode):
    provider = Script(step=[HANG])
    budget = Sentinel(10)
    step = node("step", failure_policy=FailurePolicy.RETRY, max_retries=3)
    policy = RetryPolicy(deadline_s=TIMEOUT, retain_uncertain_dispatch=True)

    with pytest.raises(RetryDeadlineError):
        run(mode, provider, [step], budget, policy)

    # The deadline's class is never retried, so one call was made.
    assert provider.calls == ["step"]
    assert_retained(budget, step, ESTIMATE)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("classes,retried", [
    pytest.param({"transient"}, False, id="default-classes"),
    pytest.param({"transient", "unknown_outcome"}, True, id="unknown-outcome-listed"),
])
def test_an_uncertain_dispatch_is_retried_only_when_unknown_outcome_is_listed(mode, classes,
                                                                              retried):
    provider = Script(step=[HANG, "step done"])
    # The kept reservation and the retry's own fit the budget exactly.
    budget = Sentinel(2 * ESTIMATE)
    tracer = Tracer()
    step = node("step", failure_policy=FailurePolicy.RETRY, max_retries=2, timeout_s=TIMEOUT)
    policy = RetryPolicy(retry_classes=classes, retain_uncertain_dispatch=True)

    if retried:
        run(mode, provider, [step], budget, policy, tracer=tracer)
        assert step.status is NodeStatus.COMPLETED and provider.calls == ["step", "step"]
        assert spans(tracer) == [(0, "timed_out", "unknown_outcome"), (1, "completed", None)]
        # The retry's charge adds to the kept exposure; neither replaces the other.
        assert_retained(budget, step, ESTIMATE + CALL)
    else:
        with pytest.raises(UncertainDispatchError):
            run(mode, provider, [step], budget, policy, tracer=tracer)
        assert provider.calls == ["step"]
        assert spans(tracer) == [(0, "timed_out", "unknown_outcome")]
        assert_retained(budget, step, ESTIMATE)
    assert budget.total_cost_usd <= budget.max_budget_usd


@pytest.mark.parametrize("mode", MODES)
def test_a_retry_after_an_uncertain_dispatch_must_fit_beside_the_kept_exposure(mode):
    provider = Script(step=[HANG, "step done"])
    budget = Sentinel(ESTIMATE + CALL)
    step = node("step", failure_policy=FailurePolicy.RETRY, max_retries=2, timeout_s=TIMEOUT)
    policy = RetryPolicy(retry_classes={"transient", "unknown_outcome"},
                         retain_uncertain_dispatch=True)

    with pytest.raises(SentinelAlert):
        run(mode, provider, [step], budget, policy)

    # The kept exposure leaves too little for the retry's reservation: no second call.
    assert provider.calls == ["step"]
    assert budget.breakdown() == {"step": ESTIMATE} and budget._reservations == {}


# ---------------------------------------------------------------------------
# A failure before the call was sent, or one the provider raised
# ---------------------------------------------------------------------------


class BlockedOpen(ToolRuntime):
    """A tool runtime whose session never opens, so the node's first call is never sent."""

    def open(self, agent):
        @asynccontextmanager
        async def session():
            await asyncio.Event().wait()
            yield  # pragma: no cover

        return session()


class HangingTool(ToolSession):
    @property
    def tools(self):
        return [ToolSpec("lookup", "Look up a fact", {"type": "object"})]

    async def call(self, tool_call):
        await asyncio.Event().wait()


class HangingTools(ToolRuntime):
    def open(self, agent):
        @asynccontextmanager
        async def session():
            yield HangingTool()

        return session()


async def asks_for_a_tool():
    return CompletionResult("", cost_usd=CALL, tool_calls=[ToolCall("1", "lookup", {})])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("outcomes,options,error,recorded", [
    pytest.param([HANG], {"tool_runtime": BlockedOpen()}, TimeoutError, {},
                 id="before-the-first-call"),
    pytest.param([asks_for_a_tool], {"tool_runtime": HangingTools()}, TimeoutError,
                 {"step": CALL}, id="during-a-tool-call"),
    pytest.param([ConnectionError("connection reset")], {}, ConnectionError, {},
                 id="provider-error"),
])
def test_a_failure_while_no_call_is_unanswered_still_releases(mode, outcomes, options, error,
                                                              recorded):
    provider = Script(step=outcomes)
    budget = Sentinel(10)
    step = node("step", timeout_s=TIMEOUT)

    with pytest.raises(error) as caught:
        run(mode, provider, [step], budget, **options)

    assert type(caught.value) is error
    assert provider.cancelled == []
    # Only a call the provider answered is recorded, at its known charge.
    assert budget.breakdown() == recorded and budget._reservations == {}
    assert budget.cost_is_complete is True and budget.cost_contains_estimates is False
    assert "cost_usd_unknown" not in step.metadata


# ---------------------------------------------------------------------------
# A cancelled sibling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy,kept", [(RETAIN, True), (RetryPolicy(), False)],
                         ids=["option", "without-option"])
def test_a_cancelled_in_flight_sibling_keeps_its_exposure_under_the_option(policy, kept):
    started = asyncio.Event()

    async def wait_for_sibling():
        await started.wait()
        raise RuntimeError("boom")

    async def start_and_hang():
        started.set()
        await asyncio.Event().wait()

    provider = Script(fails=[wait_for_sibling], inflight=[start_and_hang])
    budget = Sentinel(10)
    fails, inflight = node("fails"), node("inflight")
    executor = make("parallel", provider, budget, policy, max_concurrency=2)

    with pytest.raises(RuntimeError, match="boom"):
        execute(executor, ExecutionGraph([Topology.FORK_JOIN], [fails, inflight]))

    assert fails.status is NodeStatus.FAILED and inflight.status is NodeStatus.PENDING
    assert budget._reservations == {}
    if kept:
        assert_retained(budget, inflight, ESTIMATE)
        assert budget.breakdown() == {"inflight": ESTIMATE}
    else:
        assert budget.breakdown() == {} and budget.cost_is_complete is True
        assert "cost_usd_unknown" not in inflight.metadata


def test_a_cancelled_sibling_whose_call_was_not_sent_releases():
    started = asyncio.Event()

    class OpensForOneNode(ToolRuntime):
        """Opens the failing node's session; the other node's never opens."""

        def open(self, agent):
            @asynccontextmanager
            async def session():
                if started.is_set():
                    await asyncio.Event().wait()
                started.set()
                yield HangingTool()

            return session()

    async def fail():
        raise RuntimeError("boom")

    provider = Script(fails=[fail])
    budget = Sentinel(10)
    executor = make("parallel", provider, budget, RETAIN, max_concurrency=2,
                    tool_runtime=OpensForOneNode())

    with pytest.raises(RuntimeError, match="boom"):
        execute(executor, ExecutionGraph([Topology.FORK_JOIN], [node("fails"), node("blocked")]))

    assert provider.calls == ["fails"]
    assert budget.breakdown() == {} and budget._reservations == {}
    assert budget.cost_is_complete is True


# ---------------------------------------------------------------------------
# Holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["judge", SYNTHESIS_NODE_ID])
def test_a_reservation_converted_from_a_hold_is_kept_without_touching_other_holds(key):
    other = SYNTHESIS_NODE_ID if key == "judge" else "judge"
    budget = Sentinel(1.0)
    budget.hold(key, ESTIMATE)
    budget.hold(other, CALL)
    budget.reserve(key, ESTIMATE, from_hold=True)

    assert budget.retain_unknown(key) == ESTIMATE

    assert budget.holds == {other: CALL}
    assert budget.breakdown() == {key: ESTIMATE} and budget._reservations == {}
    assert budget.total_cost_usd == ESTIMATE
    assert budget.cost_is_complete is False and budget.cost_contains_estimates is True
    # The key can hold and convert again, as a regenerated verifier does.
    budget.hold(key, ESTIMATE)
    budget.reserve(key, ESTIMATE, from_hold=True)
    assert budget.total_cost_usd == 2 * ESTIMATE
    assert budget.holds == {other: CALL}


def test_retaining_without_a_reservation_marks_the_cost_unknown_only():
    budget = Sentinel(1.0)
    budget.record("step", CompletionResult("done", cost_usd=CALL))

    assert budget.retain_unknown("step") is None

    assert budget.breakdown() == {"step": CALL} and budget.total_cost_usd == CALL
    assert budget.cost_is_complete is False and budget.cost_contains_estimates is False


@pytest.mark.parametrize("mode", MODES)
def test_a_verifier_that_converted_its_hold_keeps_it_as_unknown_cost(mode):
    provider = Script(judge=[HANG])
    budget = Sentinel(1.0)
    draft = node("draft")
    judge = node("judge", depends_on=["draft"], verifies="draft", timeout_s=TIMEOUT)
    graph = ExecutionGraph([Topology.SERIAL], [draft, judge])
    executor = make(mode, provider, budget, RETAIN)
    executor.hold_verifier_budget(graph)
    assert budget.holds == {"judge": ESTIMATE}

    with pytest.raises(UncertainDispatchError):
        execute(executor, graph)

    # The judge converted its hold into its reservation, and kept that.
    assert budget.holds == {}
    assert budget.breakdown() == {"draft": CALL, "judge": ESTIMATE}
    assert_retained(budget, judge, ESTIMATE)
    assert budget.total_cost_usd == CALL + ESTIMATE


# ---------------------------------------------------------------------------
# Native providers keep their own accounting
# ---------------------------------------------------------------------------


async def bill_unknown_when_cancelled():
    """Hang until cancelled, then report that the native call's bill is unknown."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
        raise ProviderAccountingCancelledError(
            "Native dispatch interrupted; bill unknown", receipt={"cost_is_complete": False},
        ) from exc


@pytest.mark.parametrize("mode", MODES)
def test_a_native_unknown_bill_stays_terminal_with_its_live_reservation(mode):
    budget = Sentinel(10)
    step = node("step", failure_policy=FailurePolicy.RETRY, max_retries=2)
    policy = RetryPolicy(deadline_s=TIMEOUT, retain_uncertain_dispatch=True)

    with pytest.raises(ProviderAccountingCancelledError):
        run(mode, Script(step=[bill_unknown_when_cancelled]), [step], budget, policy)

    # The native settlement keeps the reservation and blocks resume; the
    # option neither converts it nor records it a second time.
    assert step.metadata["accounting_invalid"] is True
    assert budget._reservations == {"step": ESTIMATE} and budget.breakdown() == {}
    assert "cost_usd_is_estimate" not in step.metadata


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("resume_policy", [RETAIN, None], ids=["same-policy", "no-policy"])
def test_a_checkpoint_keeps_the_exposure_and_resume_restores_it(mode, resume_policy):
    store = Store()
    graph = ExecutionGraph([Topology.SERIAL], [
        node("step", timeout_s=TIMEOUT), node("after", depends_on=["step"]),
    ])
    first = Swarm(model="test-model", provider=Script(step=[HANG]), checkpoint_store=store,
                  artifact_dir=None, retry_policy=RETAIN, parallel=mode == "parallel")

    with pytest.raises(UncertainDispatchError):
        first.execute(graph)

    execution_id, state = saved(store)
    # Costs and node metadata are version 4 fields that every reader restores.
    assert state["status"] == "failed" and state["version"] == 4 and "requires" not in state
    assert state["budget"]["node_costs"] == {"step": ESTIMATE}
    metadata = saved_node(state, "step")["metadata"]
    assert (metadata["cost_usd"], metadata["cost_usd_unknown"],
            metadata["cost_usd_is_estimate"]) == (ESTIMATE, True, True)

    provider = Script()
    result = Swarm(model="test-model", provider=provider, checkpoint_store=store,
                   artifact_dir=None, retry_policy=resume_policy).resume(execution_id)

    assert provider.calls == ["step", "after"] and result.output == "after done"
    assert result.cost_is_complete is False and result.cost_contains_estimates is True
    assert result.total_cost_usd == ESTIMATE + 2 * CALL
    resumed = {item.id: item for item in result.graph.nodes}
    assert resumed["step"].metadata["cost_usd"] == ESTIMATE + CALL
    _, final = saved(store)
    assert final["status"] == "completed"
    assert final["budget"]["node_costs"] == {"step": ESTIMATE + CALL, "after": CALL}


# ---------------------------------------------------------------------------
# A rate limit, a tool loop and a delayed sibling through a crash and resume
# ---------------------------------------------------------------------------


DEADLINE = 10.0
BUDGET = 1.0
POLICY = RetryPolicy(deadline_s=DEADLINE, persist_attempts=True, retain_uncertain_dispatch=True)
CAPS = {"rate": 2, "tools": 2, "sibling": 1}  # max_retries by node


class Combined(Provider):
    """One SDK-free provider for three nodes; it sends each call once and declares so.

    ``rate`` is rate limited once with ``Retry-After: 2``, ``tools`` asks for a
    tool and then answers, and ``sibling`` waits at *gate*. ``started`` holds
    each call's node and the clock reading when it started.
    """

    def __init__(self, gate: Gate, clock: ManualClock) -> None:
        self.gate, self.clock = gate, clock
        self.started: list[tuple[str, float]] = []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0, tools=True)

    async def complete(self, system, prompt, model):
        return await self.chat(system, [ChatMessage(role="user", content=prompt)], model)

    def calls(self, name: str) -> int:
        return sum(1 for started, _ in self.started if started == name)

    async def chat(self, system, messages, model, tools=None):
        name = messages[0].content.splitlines()[0]
        self.started.append((name, self.clock.monotonic()))
        if name == "rate" and self.calls(name) == 1:
            raise HttpError(429, "rate_limit_error", "Rate limit reached", retry_after_s=2).exception()
        if name == "tools" and len(messages) == 1:
            return CompletionResult("", cost_usd=CALL, tool_calls=[ToolCall("1", "lookup", {})])
        if name == "sibling":
            await self.gate.wait()
        return CompletionResult(f"{name} done", cost_usd=CALL)


class Lookup(ToolRuntime):
    def open(self, agent):
        class Session(ToolSession):
            @property
            def tools(self):
                return [ToolSpec("lookup", "Look up a fact", {"type": "object"})]

            async def call(self, tool_call):
                return ToolResult(tool_call.id, "a fact")

        @asynccontextmanager
        async def session():
            yield Session()

        return session()


def combined_graph() -> ExecutionGraph:
    def capped(node_id, **options):
        return node(node_id, failure_policy=FailurePolicy.RETRY, max_retries=CAPS[node_id],
                    **options)

    return ExecutionGraph([Topology.FORK_JOIN], [
        capped("rate"), capped("tools", depends_on=["rate"]), capped("sibling"),
    ])


def test_a_rate_limit_a_tool_loop_and_a_delayed_sibling_stay_bounded_through_a_crash(
        monkeypatch):
    """The run is lost once the tool-loop node completes, while ``sibling`` is in flight.

    ``rate`` retries its 429 after exactly the two seconds ``Retry-After`` asks
    and completes; then ``tools`` completes its tool loop, and the run stops
    with ``Crash`` at that node's update. Stopping cancels ``sibling``, whose
    call was sent and never answered, so its reservation is kept as unknown
    cost and the failed checkpoint saves it. The resumed run makes only
    ``sibling``'s one remaining attempt: every node stays within its declared
    attempts, each attempt is a single provider call apart from tool-loop
    turns, the retry waited inside its deadline, spend stays within the
    budget, and the kept exposure is still reported.
    """
    clock = ManualClock()
    waits: list[float] = []

    def sleep(seconds):
        waits.append(seconds)
        clock.advance(seconds)

    class Clocked(AsyncExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, sleep=sleep, **kwargs)

    monkeypatch.setattr(async_executor_module, "AsyncExecutor", Clocked)
    store, tracers = Store(), []

    def build(provider: Provider, checkpoint_store: CheckpointStore = store) -> Swarm:
        swarm = Swarm(model="test-model", provider=provider, checkpoint_store=checkpoint_store,
                      artifact_dir=None, retry_policy=POLICY, parallel=True, max_concurrency=3,
                      max_budget_usd=BUDGET, tool_runtime=Lookup())
        swarm._tracer_factory = lambda: tracers.append(Tracer(clock=clock.monotonic)) or tracers[-1]
        return swarm

    # One retry owner: a provider that may retry inside its SDK is refused before any call.
    class Undeclared(Combined):
        def feature_support(self, model):
            return ProviderFeatures(tools=True)

    refused = Undeclared(Gate(), clock)
    with pytest.raises(ValueError, match="sdk_retries"):
        build(refused, Store()).execute(combined_graph())
    assert refused.started == []

    first = Combined(Gate(), clock)
    point = crash_when(monkeypatch, ExecutorBase, "notify_update",
                       lambda self, item: item.id == "tools" and item.status is NodeStatus.COMPLETED)
    with pytest.raises(Crash):
        build(first).execute(combined_graph())
    point.restore()
    run_tracer = tracers[-1]

    execution_id, state = saved(store)
    assert state["status"] == "failed" and state["version"] == 5
    assert {name: first.calls(name) for name in CAPS} == {"rate": 2, "tools": 2, "sibling": 1}
    # The 429 waited exactly its Retry-After, inside the deadline.
    assert waits == [2]
    rate_starts = [at for name, at in first.started if name == "rate"]
    assert rate_starts == [0, 2] and rate_starts[-1] < DEADLINE
    assert run_tracer.timing()["waiting_ms"]["retry_backoff"] == 2000
    assert state["control"]["attempts_used"] == {
        "rate": {"execution_generation": 0, "attempts": 2},
        "tools": {"execution_generation": 0, "attempts": 1},
        "sibling": {"execution_generation": 0, "attempts": 1},
    }
    # The sibling's call was sent and never answered: its reservation is kept.
    assert state["budget"]["node_costs"] == {"rate": CALL, "tools": 2 * CALL, "sibling": ESTIMATE}
    sibling = saved_node(state, "sibling")
    assert sibling["status"] == "pending"
    assert (sibling["metadata"]["cost_usd_unknown"], sibling["metadata"]["cost_usd_is_estimate"]) \
        == (True, True)

    assert sum(state["budget"]["node_costs"].values()) <= BUDGET

    # The checkpoint records the option with the policy, so only an equal policy resumes it.
    assert state["control"]["retry_policy"]["retain_uncertain_dispatch"] is True
    other = Combined(Gate(), clock)
    swarm = build(other)
    swarm.retry_policy = RetryPolicy(deadline_s=DEADLINE, persist_attempts=True)
    with pytest.raises(ValueError, match="retain_uncertain_dispatch"):
        swarm.resume(execution_id)
    assert other.started == []

    released = Gate()
    released.release()
    second = Combined(released, clock)
    result = build(second).resume(execution_id)

    # Completed nodes are not called again; the sibling makes its last attempt.
    assert [name for name, _ in second.started] == ["sibling"]
    _, final = saved(store)
    attempts = {node_id: record["attempts"]
                for node_id, record in final["control"]["attempts_used"].items()}
    assert attempts == {"rate": 2, "tools": 1, "sibling": 2}
    for node_id, used in attempts.items():
        assert used <= 1 + CAPS[node_id]
    # Every attempt is one provider call, except a tool loop's turns, which are one attempt.
    calls = {name: first.calls(name) + second.calls(name) for name in CAPS}
    assert calls == {"rate": 2, "tools": 2, "sibling": 2}
    # Spend stays within the budget, and the kept exposure is still reported.
    assert final["budget"]["node_costs"] == {
        "rate": CALL, "tools": 2 * CALL, "sibling": ESTIMATE + CALL,
    }
    assert result.total_cost_usd == 4 * CALL + ESTIMATE <= BUDGET
    assert result.cost_is_complete is False and result.cost_contains_estimates is True
    resumed = {item.id: item for item in result.graph.nodes}
    assert resumed["sibling"].status is NodeStatus.COMPLETED
    assert resumed["sibling"].metadata["cost_usd_unknown"] is True
