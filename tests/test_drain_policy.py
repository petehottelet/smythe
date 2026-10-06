"""An opt-in ``TerminationPolicy`` drains in-flight work after a failure.

Without one, the async executor cancels every in-flight sibling as soon as it
sees a failure. With ``TerminationPolicy(drain_timeout_s=...)`` it admits no
further node, lets the nodes already in flight finish until the drain deadline,
keeping and saving their results, then cancels the rest and raises the original
error. A cancellation, an interrupt and a security stop never drain.

The executor's ``sleep`` times the drain deadline here: a sleep that advances a
``ManualClock`` reaches the deadline at once, and one that releases a gated
sibling and then waits lets that sibling finish first. Every call charges
``CALL`` against an estimate of ``ESTIMATE``; both are binary fractions, so
every total is exact.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import json
import math

import pytest

import smythe.async_executor as async_executor_module
from fakes import Crash, Gate, ManualClock
from smythe import RetryPolicy, TerminationPolicy
from smythe._failures import SECURITY_STOP, classify_failure
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.checkpoint import CheckpointStore
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import (
    CompletionResult, Provider, ProviderAccountingCancelledError, ProviderFeatures,
)
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.swarm import Swarm
from smythe.tracer import WAIT_REASONS, Tracer
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import SQLiteWorkflowStore

ESTIMATE = 0.25
CALL = 0.125
DRAIN_S = 30.0
DRAIN = TerminationPolicy(drain_timeout_s=DRAIN_S)


class Script(Provider):
    """Runs each call as the coroutine function its node's label names; others answer at once.

    A step's return value replaces the reply text and an exception it raises
    fails the call. ``calls`` lists each call's label and ``cancelled`` each
    call cancelled in flight. Every answered call is charged ``CALL``, and the
    provider sends each call once.
    """

    def __init__(self, **steps) -> None:
        self.steps = steps
        self.calls: list[str] = []
        self.cancelled: list[str] = []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0)

    def budget_estimate_usd(self, model):
        return None

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        self.calls.append(label)
        text = f"{label} done"
        step = self.steps.get(label)
        if step is not None:
            try:
                text = await step() or text
            except asyncio.CancelledError:
                self.cancelled.append(label)
                raise
        return CompletionResult(text, cost_usd=CALL)


def node(node_id: str, **options) -> Node:
    options.setdefault("metadata", {"estimated_cost_usd": ESTIMATE})
    return Node(id=node_id, label=node_id, **options)


def halting(gate: Gate, error: BaseException | None = None):
    """A step that fails once a gated sibling is in flight."""

    async def fail():
        assert await gate.reached()
        raise error or RuntimeError("stop now")

    return fail


async def hang():
    await asyncio.Event().wait()


def entering(gate: Gate, step):
    """*step*, after marking *gate* reached, for a sibling that is never released."""

    async def call():
        gate.entered.set()
        return await step()

    return call


class DrainSleep:
    """The executor's sleep, which never returns: only the nodes in flight can end the drain.

    It records each wait, sets ``draining`` and releases *gate*, if any.
    """

    def __init__(self, gate: Gate | None = None) -> None:
        self.gate, self.waits = gate, []
        self.draining = asyncio.Event()

    async def __call__(self, seconds):
        self.waits.append(seconds)
        self.draining.set()
        if self.gate is not None:
            self.gate.release()
        await asyncio.Event().wait()


def expiring(clock: ManualClock, waits: list[float]):
    """The executor's sleep: each wait passes at once on *clock*."""

    def sleep(seconds):
        waits.append(seconds)
        clock.advance(seconds)

    return sleep


def make(provider, *, termination=DRAIN, budget=None, clock=None, updates=None, **options):
    options.setdefault("max_concurrency", 2)
    tracer = Tracer(clock=clock.monotonic) if clock is not None else Tracer()
    hook = None if updates is None else (lambda item: updates.append((item.id, item.status)))
    return AsyncExecutor(provider=provider, registry=Registry(), tracer=tracer,
                         budget=budget if budget is not None else Sentinel(10),
                         artifact_dir=None, on_node_update=hook, termination=termination,
                         **options)


def fork(*nodes: Node) -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], list(nodes))


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_a_drain_needs_a_finite_positive_timeout_and_the_policy_is_frozen():
    assert TerminationPolicy(drain_timeout_s=5) == TerminationPolicy(drain_timeout_s=5.0)
    assert TerminationPolicy(drain_timeout_s=5).drain_timeout_s == 5.0
    with pytest.raises(FrozenInstanceError):
        DRAIN.drain_timeout_s = 1
    for value in (0, -1, math.inf, math.nan, True, "5", None, 10 ** 400):
        with pytest.raises(ValueError, match="drain_timeout_s"):
            TerminationPolicy(drain_timeout_s=value)
    assert "drain" in WAIT_REASONS


def test_only_a_termination_policy_is_accepted_and_durable_runs_refuse_one(tmp_path):
    with pytest.raises(TypeError, match="termination must be a TerminationPolicy"):
        make(Script(), termination={"drain_timeout_s": 5})
    with pytest.raises(TypeError, match="termination must be a TerminationPolicy"):
        Swarm(model="test-model", provider=Script(), termination=5)
    store = SQLiteWorkflowStore(tmp_path / "runs.sqlite")
    try:
        with pytest.raises(WorkflowBindingError, match="^termination is not supported"):
            Swarm(model="test-model", provider=Script(), run_store=store, termination=DRAIN)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Without a policy nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_policy_a_failure_still_cancels_in_flight_siblings_at_once():
    gate = Gate()
    provider = Script(fail=halting(gate), sibling=gate.wait)
    sleep = DrainSleep(gate)
    fail, sibling, queued = node("fail"), node("sibling"), node("queued")
    budget = Sentinel(10)
    executor = make(provider, termination=None, sleep=sleep, budget=budget)

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(fail, sibling, queued))

    assert provider.calls == ["fail", "sibling"] and provider.cancelled == ["sibling"]
    assert sleep.waits == [] and not gate.released
    assert (fail.status, sibling.status, queued.status) == (
        NodeStatus.FAILED, NodeStatus.PENDING, NodeStatus.PENDING)
    assert budget.breakdown() == {} and budget._reservations == {}
    assert executor.drain_record is None
    assert executor._tracer.timing()["waiting_ms"]["drain"] == 0


# ---------------------------------------------------------------------------
# Draining
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sibling_in_flight_finishes_during_the_drain_and_is_saved():
    gate = Gate()
    provider = Script(fail=halting(gate), sibling=gate.wait)
    sleep = DrainSleep(gate)
    fail, sibling, queued = node("fail"), node("sibling"), node("queued")
    budget, updates = Sentinel(10), []
    executor = make(provider, sleep=sleep, budget=budget, updates=updates)

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(fail, sibling, queued))

    # The drain began after the failure and released the gate before its deadline.
    assert sleep.waits == [DRAIN_S]
    assert sibling.status is NodeStatus.COMPLETED and sibling.result == "sibling done"
    # The completion went through the checkpoint hook like any other.
    assert updates == [("fail", NodeStatus.FAILED), ("sibling", NodeStatus.COMPLETED)]
    assert budget.breakdown() == {"sibling": CALL} and budget._reservations == {}
    # Admission stopped at the failure: the queued node never started.
    assert provider.calls == ["fail", "sibling"] and provider.cancelled == []
    assert queued.status is NodeStatus.PENDING
    assert executor.drain_record == {"settled": ["sibling"], "cancelled": [],
                                     "deadline_hit": False}
    outcomes = {span.node_id: span.outcome for span in executor._tracer.spans}
    assert outcomes == {"fail": "failed", "sibling": "completed"}


@pytest.mark.asyncio
async def test_a_sibling_that_fails_during_the_drain_settles_and_the_first_error_is_raised():
    gate = Gate()

    async def fail_late():
        await gate.wait()
        raise RuntimeError("second failure")

    provider = Script(fail=halting(gate), sibling=fail_late)
    executor = make(provider, sleep=DrainSleep(gate))
    sibling = node("sibling")

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(node("fail"), sibling))

    assert sibling.status is NodeStatus.FAILED and sibling.result == "second failure"
    assert executor.drain_record == {"settled": ["sibling"], "cancelled": [],
                                     "deadline_hit": False}


@pytest.mark.asyncio
async def test_completions_during_a_drain_are_not_reviewed():
    class Recording(Supervisor):
        def __init__(self):
            self.reviewed = []

        async def review(self, graph, node, *, task, revisions_remaining):
            self.reviewed.append(node.id)
            return None

    gate = Gate()
    supervisor = Recording()
    executor = make(Script(fail=halting(gate), sibling=gate.wait), sleep=DrainSleep(gate),
                    supervisor=supervisor, max_revisions=1)
    sibling = node("sibling")

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(node("fail"), sibling))

    assert sibling.status is NodeStatus.COMPLETED
    assert supervisor.reviewed == []


@pytest.mark.asyncio
async def test_an_admission_failure_drains_the_nodes_already_in_flight():
    gate = Gate()
    provider = Script(slow=gate.wait)
    budget = Sentinel(10)
    slow, quick = node("slow"), node("quick")
    # Once quick finishes, both become ready: ``late`` is admitted, then the
    # budget refuses ``costly`` before late's task has begun.
    late = node("late", depends_on=["quick"])
    costly = node("costly", depends_on=["quick"], metadata={"estimated_cost_usd": 20.0})
    executor = make(provider, sleep=DrainSleep(gate), budget=budget, max_concurrency=4)

    with pytest.raises(SentinelAlert):
        await executor.run(fork(slow, quick, late, costly))

    # slow, in flight, finished during the drain; late never started.
    assert provider.calls == ["slow", "quick"]
    assert (slow.status, quick.status, late.status, costly.status) == (
        NodeStatus.COMPLETED, NodeStatus.COMPLETED, NodeStatus.PENDING, NodeStatus.PENDING)
    assert budget.breakdown() == {"slow": CALL, "quick": CALL} and budget._reservations == {}
    assert executor.drain_record == {"settled": ["slow"], "cancelled": [],
                                     "deadline_hit": False}


# ---------------------------------------------------------------------------
# The deadline
# ---------------------------------------------------------------------------


async def bill_unknown_when_cancelled():
    """Hang until cancelled, then report that the native call's bill is unknown."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
        raise ProviderAccountingCancelledError(
            "Native dispatch interrupted; bill unknown", receipt={"cost_is_complete": False},
        ) from exc


@pytest.mark.asyncio
@pytest.mark.parametrize("straggle,retry_policy,exposure", [
    (hang, None, "released"),
    (hang, RetryPolicy(retain_uncertain_dispatch=True), "kept"),
    (bill_unknown_when_cancelled, None, "native"),
], ids=["legacy", "legacy-retained", "native"])
async def test_the_deadline_cancels_stragglers_with_exposure_kept_per_provider(
    straggle, retry_policy, exposure,
):
    gate = Gate()
    clock, waits = ManualClock(), []
    provider = Script(fail=halting(gate), straggler=entering(gate, straggle))
    straggler, queued = node("straggler"), node("queued")
    budget = Sentinel(10)
    executor = make(provider, sleep=expiring(clock, waits), budget=budget, clock=clock,
                    retry_policy=retry_policy)

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(node("fail"), straggler, queued))

    assert waits == [DRAIN_S]
    assert executor.drain_record == {"settled": [], "cancelled": ["straggler"],
                                     "deadline_hit": True}
    assert executor._tracer.timing()["waiting_ms"]["drain"] == DRAIN_S * 1000
    assert "queued" not in provider.calls and queued.status is NodeStatus.PENDING
    if exposure == "released":
        assert straggler.status is NodeStatus.PENDING
        assert budget.breakdown() == {} and budget._reservations == {}
        assert budget.cost_is_complete is True
    elif exposure == "kept":
        assert straggler.status is NodeStatus.PENDING
        assert budget.breakdown() == {"straggler": ESTIMATE} and budget._reservations == {}
        assert straggler.metadata["cost_usd_unknown"] is True
        assert budget.cost_is_complete is False
    else:
        # The native settlement keeps its live reservation and blocks resume.
        assert straggler.status is NodeStatus.FAILED
        assert straggler.metadata["accounting_invalid"] is True
        assert budget._reservations == {"straggler": ESTIMATE} and budget.breakdown() == {}


# ---------------------------------------------------------------------------
# What never drains
# ---------------------------------------------------------------------------


class Revoked(RuntimeError):
    """Stands for a failure whose class is a security stop."""


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [Crash("interrupted"), Revoked("authority revoked")],
                         ids=["interrupt", "security-stop"])
async def test_an_interrupt_or_a_security_stop_never_drains(error, monkeypatch):
    real = classify_failure
    monkeypatch.setattr(async_executor_module, "classify_failure",
                        lambda exc: SECURITY_STOP if isinstance(exc, Revoked) else real(exc))
    gate = Gate()
    provider = Script(fail=halting(gate, error), sibling=gate.wait)
    sleep = DrainSleep(gate)
    sibling = node("sibling")
    executor = make(provider, sleep=sleep)

    with pytest.raises(type(error)):
        await executor.run(fork(node("fail"), sibling))

    assert sleep.waits == [] and provider.cancelled == ["sibling"]
    assert sibling.status is NodeStatus.PENDING and executor.drain_record is None


@pytest.mark.asyncio
async def test_cancelling_the_run_during_a_drain_cancels_the_nodes_in_flight_at_once():
    gate = Gate()
    provider = Script(fail=halting(gate), sibling=gate.wait)
    sleep = DrainSleep()  # The drain waits; nothing releases the sibling.
    sibling = node("sibling")
    executor = make(provider, sleep=sleep)
    run = asyncio.create_task(executor.run(fork(node("fail"), sibling)))
    await asyncio.wait_for(sleep.draining.wait(), 10)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert provider.cancelled == ["sibling"] and sibling.status is NodeStatus.PENDING
    assert executor.drain_record == {"settled": [], "cancelled": ["sibling"],
                                     "deadline_hit": False}


# ---------------------------------------------------------------------------
# Checkpoints and resume
# ---------------------------------------------------------------------------


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


def saved_nodes(state: dict) -> dict[str, dict]:
    return {item["id"]: item for item in state["graph"]["nodes"]}


@pytest.mark.parametrize("termination", [None, TerminationPolicy(drain_timeout_s=0.5)],
                         ids=["no-policy", "drain"])
def test_a_drained_run_saves_its_finished_sibling_and_resume_runs_only_the_rest(termination):
    """``done`` answers shortly after ``fail`` raises; ``slow`` never answers."""
    entered: dict[str, asyncio.Event] = {}

    def event(name):
        return entered.setdefault(name, asyncio.Event())

    async def fail():
        await event("done").wait()
        await event("slow").wait()
        event("failed").set()
        raise RuntimeError("stop now")

    async def done():
        event("done").set()
        await event("failed").wait()
        await asyncio.sleep(0.05)

    async def slow():
        event("slow").set()
        await asyncio.Event().wait()

    store = Store()
    graph = fork(node("fail"), node("done"), node("slow"), node("queued"))
    first = Script(fail=fail, done=done, slow=slow)
    swarm = Swarm(model="test-model", provider=first, checkpoint_store=store, artifact_dir=None,
                  parallel=True, max_concurrency=3, termination=termination)

    with pytest.raises(RuntimeError, match="stop now"):
        swarm.execute(graph)

    (execution_id,) = store.list_ids()
    state = store.load(execution_id)
    # The drain record is informational: the checkpoint stays version 4.
    assert state["status"] == "failed" and state["version"] == 4 and "requires" not in state
    nodes = saved_nodes(state)
    statuses = {node_id: item["status"] for node_id, item in nodes.items()}
    if termination is None:
        assert "drain" not in state["control"]
        assert statuses == {"fail": "failed", "done": "pending", "slow": "pending",
                            "queued": "pending"}
        assert state["budget"]["node_costs"] == {}
        rerun = ["done", "fail", "queued", "slow"]
    else:
        assert state["control"]["drain"] == {"settled": ["done"], "cancelled": ["slow"],
                                             "deadline_hit": True}
        assert statuses == {"fail": "failed", "done": "completed", "slow": "pending",
                            "queued": "pending"}
        assert nodes["done"]["result"] == "done done"
        assert state["budget"]["node_costs"] == {"done": CALL}
        assert state["timing"]["segments"][-1]["waiting_ms"]["drain"] > 0
        rerun = ["fail", "queued", "slow"]
    assert sorted(first.calls) == ["done", "fail", "slow"]

    second = Script()
    result = Swarm(model="test-model", provider=second, checkpoint_store=store,
                   artifact_dir=None, parallel=True).resume(execution_id)

    # Only the failed, cancelled and never-started nodes run again.
    assert sorted(second.calls) == rerun
    assert result.total_cost_usd == 4 * CALL
    final = store.load(execution_id)
    assert final["status"] == "completed" and "drain" not in final["control"]
