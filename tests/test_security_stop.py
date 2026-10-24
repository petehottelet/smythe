"""A security stop ends a run at once, keeps what its calls may cost, and never resumes.

Only developer code stops a run for security: ``RunControl.revoke(reason_code)``
or a ``SecurityStop`` raised by a tool runtime. The stop is never delayed: a
revoked control admits no node and dispatches no call, and the async executor
cancels the calls in flight at once, even during a drain or a pending supervisor
review, which is cancelled too. A legacy call that
was sent and never answered keeps its reservation as unknown cost without the
retention option. An ordinary run saves the stop in a version 5 checkpoint and a
durable run's journal blocks it with ``security_stop``; resume refuses both.

Every call charges ``CALL`` against an estimate of ``ESTIMATE``; both are binary
fractions, so every total is exact.
"""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
import threading
import time

import pytest

import smythe.checkpoint as checkpoints
import smythe.swarm as swarm_module
from fakes import Gate, ManualClock, NativeTransport, phase_of
from smythe import (
    AcceptancePolicy, OfflineProvider, OpenAIResponsesProvider, RunControl, SecurityStop,
    SimpleArchitect, SQLiteWorkflowStore, Swarm, Task, TerminationPolicy,
)
from smythe._failures import SECURITY_STOP, classify_failure, is_terminal
from smythe._run_inspection import inspect_journal
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import CheckpointStore, build_state, graph_from_dict, security_stop_from_state
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider, ProviderFeatures
from smythe.registry import Registry
from smythe.supervisor import EvidenceTriggeredSupervisor, LLMSupervisor, ReviewTriggers, Supervisor
from smythe.tools import ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec
from smythe.tracer import Tracer
from smythe.workflow_store import (
    WorkflowConflictError, WorkflowStateError, WorkflowValidationError,
)

ESTIMATE = 0.25
CALL = 0.125
DRAIN = TerminationPolicy(drain_timeout_s=30.0)
WALL = 1_700_000_000.0


class Script(Provider):
    """Runs each call as the coroutine function its node's label names; others answer at once.

    ``calls`` lists each call's label and ``cancelled`` each call cancelled in
    flight. Every answered call is charged ``CALL``, and the provider sends
    each call once. A label in ``tooling`` asks for the ``lookup`` tool on its
    first turn.
    """

    def __init__(self, *, tooling=(), **steps) -> None:
        self.steps, self.tooling = steps, set(tooling)
        self.calls: list[str] = []
        self.cancelled: list[str] = []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0, tools=True)

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

    async def chat(self, system, messages, model, tools=None):
        label = messages[0].content.splitlines()[0]
        if any(message.role == "assistant" for message in messages):
            self.calls.append(label)
            return CompletionResult(f"{label} done", cost_usd=CALL)
        result = await self.complete(system, messages[0].content, model)
        if label in self.tooling:
            result.text, result.stop_reason = "", "tool_use"
            result.tool_calls = [ToolCall(id="lookup-1", name="lookup", arguments={})]
        return result


class Tools(ToolRuntime):
    """Opens a session whose ``lookup`` tool runs *call* and answers ``found``."""

    def __init__(self, call) -> None:
        self.call = call

    def open(self, agent):
        runtime = self

        class Session(ToolSession):
            @property
            def tools(self):
                return [ToolSpec(name="lookup", description="Look it up",
                                 input_schema={"type": "object"})]

            async def call(self, tool_call):
                runtime.call()
                return ToolResult(tool_call_id=tool_call.id, content="found")

        class Opened:
            async def __aenter__(self):
                return Session()

            async def __aexit__(self, *_):
                return False

        return Opened()


def node(node_id: str, **options) -> Node:
    options.setdefault("metadata", {"estimated_cost_usd": ESTIMATE})
    return Node(id=node_id, label=node_id, **options)


def fork(*nodes: Node) -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], list(nodes))


def make(provider, control, *, mode="parallel", budget=None, clock=None, **options):
    tracer = (Tracer(clock=clock.monotonic, wall_clock=lambda: WALL) if clock is not None
              else Tracer(wall_clock=lambda: WALL))
    if mode == "serial":
        return Executor(provider=provider, registry=Registry(), tracer=tracer,
                        budget=budget if budget is not None else Sentinel(10),
                        artifact_dir=None, run_control=control, **options)
    options.setdefault("max_concurrency", 2)
    return AsyncExecutor(provider=provider, registry=Registry(), tracer=tracer,
                         budget=budget if budget is not None else Sentinel(10),
                         artifact_dir=None, run_control=control, **options)


async def execute(executor, graph):
    if isinstance(executor, AsyncExecutor):
        return await executor.run(graph)
    return await asyncio.to_thread(executor.run, graph)


def halting(gate: Gate, error: BaseException | None = None):
    """A step that fails once a gated sibling is in flight."""

    async def fail():
        assert await gate.reached()
        raise error or RuntimeError("stop now")

    return fail


async def hang():
    await asyncio.Event().wait()


def entering(gate: Gate, step):
    async def call():
        gate.entered.set()
        return await step()

    return call


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


def saved_nodes(state: dict) -> dict[str, dict]:
    return {item["id"]: item for item in state["graph"]["nodes"]}


def ordinary(provider, store, control, **options) -> Swarm:
    options.setdefault("parallel", True)
    swarm = Swarm(model="test-model", provider=provider, checkpoint_store=store,
                  artifact_dir=None, run_control=control, **options)
    swarm._tracer_factory = lambda: Tracer(wall_clock=lambda: WALL)
    return swarm


# ---------------------------------------------------------------------------
# The stop and the control
# ---------------------------------------------------------------------------


def test_a_security_stop_carries_its_code_and_is_terminal():
    stop = SecurityStop("key_revoked")

    assert stop.reason_code == "key_revoked" and "key_revoked" in str(stop)
    assert classify_failure(stop) == SECURITY_STOP == "security_stop"
    assert is_terminal(stop) is True
    assert isinstance(stop, Exception)


@pytest.mark.parametrize("code", [
    "", " ", "has space", "x" * 65, "-leading", "Error code: 401 - {'error': 'invalid key'}",
    "line\nbreak",
])
def test_a_reason_code_is_a_short_developer_code(code):
    with pytest.raises(ValueError, match="reason_code"):
        SecurityStop(code)
    control = RunControl()
    with pytest.raises(ValueError, match="reason_code"):
        control.revoke(code)
    assert control.revoked is False and control.reason_code is None


@pytest.mark.parametrize("code", [None, 5, b"code", ["code"]])
def test_a_reason_code_must_be_text(code):
    with pytest.raises(TypeError, match="reason_code"):
        SecurityStop(code)


@pytest.mark.parametrize("how", ["pickle", "copy", "deepcopy"])
def test_a_security_stop_survives_pickle_and_copy(how):
    stop = SecurityStop("tenant.suspended:2")
    restored = {"pickle": lambda error: pickle.loads(pickle.dumps(error)),
                "copy": copy.copy, "deepcopy": copy.deepcopy}[how](stop)

    assert type(restored) is SecurityStop and restored.reason_code == "tenant.suspended:2"
    assert str(restored) == str(stop) and restored.args == stop.args


def test_a_revoked_control_stays_revoked_with_its_first_code():
    control = RunControl()
    control.check()
    assert control.revoked is False and control.reason_code is None

    control.revoke("credential_leak")
    control.revoke("later_reason")

    assert control.revoked is True and control.reason_code == "credential_leak"
    with pytest.raises(SecurityStop) as raised:
        control.check()
    assert raised.value.reason_code == "credential_leak"


@pytest.mark.asyncio
async def test_a_revoke_from_another_thread_wakes_a_waiting_run():
    control = RunControl()
    waiting = asyncio.ensure_future(control.wait())
    await asyncio.sleep(0)
    assert not waiting.done()

    thread = threading.Thread(target=control.revoke, args=("operator_stop",))
    thread.start()
    await asyncio.wait_for(waiting, 10)
    thread.join()

    # Once revoked, a wait returns at once.
    await asyncio.wait_for(control.wait(), 1)


def test_only_a_run_control_is_accepted():
    with pytest.raises(TypeError, match="run_control must be a RunControl"):
        Swarm(model="test-model", provider=Script(), run_control="stop")
    with pytest.raises(TypeError, match="run_control must be a RunControl"):
        AsyncExecutor(provider=Script(), registry=Registry(), tracer=Tracer(), run_control=True)


# ---------------------------------------------------------------------------
# Admission and dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_a_revoke_before_a_run_makes_no_provider_call_and_saves_nothing(mode):
    provider, store, control = Script(), Store(), RunControl()
    control.revoke("account_frozen")
    swarm = ordinary(provider, store, control, parallel=mode == "parallel")

    with pytest.raises(SecurityStop, match="account_frozen"):
        swarm.execute(fork(node("first"), node("second")))
    with pytest.raises(SecurityStop, match="account_frozen"):
        swarm.plan(Task("Write a report"))

    assert provider.calls == [] and store.list_ids() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["serial", "parallel"])
async def test_a_revoke_before_dispatch_makes_no_provider_call(mode):
    provider, control, budget = Script(), RunControl(), Sentinel(10)
    first = node("first")
    executor = make(provider, control, mode=mode, budget=budget)
    control.revoke("account_frozen")

    with pytest.raises(SecurityStop, match="account_frozen"):
        await execute(executor, fork(first))

    assert provider.calls == []
    assert budget.breakdown() == {} and budget._reservations == {}
    assert executor.security_stop == {"reason_code": "account_frozen", "at": WALL}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["serial", "parallel"])
async def test_a_revoke_between_tool_turns_sends_no_further_turn(mode):
    control = RunControl()
    provider = Script(tooling={"lookup"})
    budget = Sentinel(10)
    looking = node("lookup")
    executor = make(provider, control, mode=mode, budget=budget,
                    tool_runtime=Tools(lambda: control.revoke("tool_saw_secret")))

    with pytest.raises(SecurityStop, match="tool_saw_secret"):
        await execute(executor, fork(looking))

    # The tool's result never went back to the provider.
    assert provider.calls == ["lookup"]
    assert looking.status is NodeStatus.FAILED
    assert budget.breakdown() == {"lookup": CALL} and budget._reservations == {}


# ---------------------------------------------------------------------------
# A security stop never drains
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_revoke_during_a_drain_cancels_at_once_and_never_waits_the_deadline():
    gate, control, clock = Gate(), RunControl(), ManualClock()
    provider = Script(fail=halting(gate), sibling=entering(gate, hang))
    waits: list[float] = []

    async def sleep(seconds):
        # The drain's deadline starts; the operator revokes while it runs.
        waits.append(seconds)
        control.revoke("incident_42")
        await asyncio.Event().wait()

    sibling, queued = node("sibling"), node("queued")
    budget = Sentinel(10)
    executor = make(provider, control, budget=budget, clock=clock, sleep=sleep,
                    termination=DRAIN)

    with pytest.raises(SecurityStop, match="incident_42") as raised:
        await asyncio.wait_for(executor.run(fork(node("fail"), sibling, queued)), 10)

    # The run's own failure is kept as the stop's cause.
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert waits == [30.0] and clock.monotonic() == 0
    assert executor._tracer.timing()["waiting_ms"]["drain"] == 0
    assert provider.cancelled == ["sibling"] and "queued" not in provider.calls
    assert executor.drain_record == {"settled": [], "cancelled": ["sibling"],
                                     "deadline_hit": False}
    # The sibling's call was sent and never answered: its reservation is kept.
    assert sibling.status is NodeStatus.PENDING
    assert budget.breakdown() == {"sibling": ESTIMATE} and budget._reservations == {}
    assert sibling.metadata["cost_usd_unknown"] is True
    assert sibling.metadata["cost_usd_is_estimate"] is True
    assert executor.security_stop == {"reason_code": "incident_42", "at": WALL}


@pytest.mark.asyncio
async def test_a_revoke_while_calls_are_in_flight_cancels_them_without_a_drain():
    gate, control = Gate(), RunControl()

    async def revoke_then_fail():
        assert await gate.reached()
        control.revoke("incident_7")
        await hang()

    provider = Script(watch=revoke_then_fail, sibling=entering(gate, hang))
    waits: list[float] = []

    async def sleep(seconds):
        waits.append(seconds)
        await hang()

    budget = Sentinel(10)
    executor = make(provider, control, budget=budget, sleep=sleep, termination=DRAIN)

    with pytest.raises(SecurityStop, match="incident_7"):
        await asyncio.wait_for(executor.run(fork(node("watch"), node("sibling"))), 10)

    assert waits == [] and executor.drain_record is None
    assert sorted(provider.cancelled) == ["sibling", "watch"]
    # Both calls were sent and never answered.
    assert budget.breakdown() == {"sibling": ESTIMATE, "watch": ESTIMATE}
    assert budget.cost_is_complete is False


@pytest.mark.asyncio
async def test_a_security_stop_from_a_tool_cancels_the_siblings_without_a_drain():
    gate, control = Gate(), RunControl()

    async def once_the_sibling_is_in_flight():
        assert await gate.reached()

    # ``lookup`` asks for its tool once ``sibling``'s call is in flight, and the tool stops the run.
    provider = Script(tooling={"lookup"}, lookup=once_the_sibling_is_in_flight,
                      sibling=entering(gate, hang))
    waits: list[float] = []

    async def sleep(seconds):
        waits.append(seconds)
        await hang()

    budget = Sentinel(10)
    sibling = node("sibling")
    executor = make(provider, control, budget=budget, sleep=sleep, termination=DRAIN,
                    tool_runtime=Tools(raising(SecurityStop("tool_flagged"))))

    with pytest.raises(SecurityStop, match="tool_flagged"):
        await asyncio.wait_for(executor.run(fork(node("lookup"), sibling)), 10)

    assert waits == [] and executor.drain_record is None
    assert provider.cancelled == ["sibling"]
    assert budget.breakdown() == {"lookup": CALL, "sibling": ESTIMATE}
    assert sibling.metadata["cost_usd_unknown"] is True
    assert executor.security_stop == {"reason_code": "tool_flagged", "at": WALL}
    # The control itself was never revoked: the tool stopped this run only.
    assert control.revoked is False


@pytest.mark.asyncio
async def test_without_a_stop_a_cancelled_legacy_call_still_releases_its_reservation():
    gate = Gate()
    provider = Script(fail=halting(gate), sibling=entering(gate, hang))
    budget = Sentinel(10)
    sibling = node("sibling")
    executor = make(provider, RunControl(), budget=budget)

    with pytest.raises(RuntimeError, match="stop now"):
        await executor.run(fork(node("fail"), sibling))

    assert provider.cancelled == ["sibling"] and sibling.status is NodeStatus.PENDING
    assert budget.breakdown() == {} and budget._reservations == {}
    assert "cost_usd_unknown" not in sibling.metadata
    assert executor.security_stop is None


# ---------------------------------------------------------------------------
# A security stop during a supervisor review
# ---------------------------------------------------------------------------


class HeldReview(Supervisor):
    """Reviews after ``first`` by waiting at *gate*; ``cancelled`` records a review cancelled there."""

    def __init__(self, gate: Gate) -> None:
        self.gate, self.cancelled = gate, False

    async def review(self, graph, node, *, task, revisions_remaining):
        if node.id != "first":
            return None
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return None


class HeldReviewer(Provider):
    """A reviewer's provider that holds each review call at *gate*, as a slow model review would."""

    def __init__(self, gate: Gate) -> None:
        self.gate, self.cancelled = gate, False

    async def complete(self, system, prompt, model):
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CompletionResult('{"change": false}', cost_usd=CALL)


def held_review(kind: str, gate: Gate):
    """A supervisor whose review after ``first`` waits at *gate*, and what records its cancellation."""
    if kind == "supervisor":
        supervisor = HeldReview(gate)
        return supervisor, supervisor
    reviewer = HeldReviewer(gate)
    return EvidenceTriggeredSupervisor(
        LLMSupervisor(reviewer, model="test-model"), ReviewTriggers(review_after={"first"}),
    ), reviewer


async def ended_before_release(running: asyncio.Future, gate: Gate) -> bool:
    """Whether *running* ends while *gate* still holds the review; the gate is released after."""
    try:
        await asyncio.wait({running}, timeout=5)
        return running.done() and not gate.released
    finally:
        gate.release()
        await asyncio.wait({running}, timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["supervisor", "evidence-triggered"])
async def test_a_revoke_during_a_supervisor_review_cancels_the_calls_in_flight_at_once(kind):
    in_flight, review_gate, control = Gate(), Gate(), RunControl()
    provider = Script(slow=entering(in_flight, hang))
    supervisor, review = held_review(kind, review_gate)
    waits: list[float] = []

    async def sleep(seconds):
        waits.append(seconds)
        await hang()

    budget = Sentinel(10)
    slow = node("slow")
    executor = make(provider, control, budget=budget, sleep=sleep, termination=DRAIN,
                    supervisor=supervisor, max_revisions=1)
    running = asyncio.ensure_future(
        executor.run(fork(node("first"), slow, node("queued", depends_on=["first"]))))
    assert await in_flight.reached() and await review_gate.reached()

    control.revoke("incident_9")

    # The run ends, its call in flight cancelled, while the review is still held.
    assert await ended_before_release(running, review_gate)
    with pytest.raises(SecurityStop, match="incident_9"):
        running.result()
    assert provider.cancelled == ["slow"] and review.cancelled is True
    # No call was sent after the revocation, and nothing drained.
    assert sorted(provider.calls) == ["first", "slow"]
    assert waits == [] and executor.drain_record is None
    # The call in flight was sent and never answered: its reservation is kept.
    assert budget.breakdown() == {"first": CALL, "slow": ESTIMATE}
    assert slow.metadata["cost_usd_unknown"] is True
    assert executor.security_stop == {"reason_code": "incident_9", "at": WALL}


@pytest.mark.asyncio
async def test_a_revoke_during_a_serial_review_stops_the_run_without_waiting_for_it():
    review_gate, control = Gate(), RunControl()
    provider = Script()
    supervisor = HeldReview(review_gate)
    executor = make(provider, control, mode="serial", supervisor=supervisor, max_revisions=1)
    running = asyncio.ensure_future(
        execute(executor, fork(node("first"), node("queued", depends_on=["first"]))))
    assert await review_gate.reached()

    control.revoke("incident_9")

    assert await ended_before_release(running, review_gate)
    with pytest.raises(SecurityStop, match="incident_9"):
        running.result()
    assert supervisor.cancelled is True and provider.calls == ["first"]
    assert executor.security_stop == {"reason_code": "incident_9", "at": WALL}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def raising(error):
    def call():
        raise error

    return call


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("error", [
    SecurityStop("tool_flagged"),
    ExceptionGroup("tool task group", [SecurityStop("tool_flagged")]),
], ids=["plain", "grouped"])
async def test_a_security_stop_raised_by_a_tool_propagates(mode, error):
    provider = Script(tooling={"lookup"})
    looking = node("lookup")
    executor = make(provider, None, mode=mode, tool_runtime=Tools(raising(error)))

    with pytest.raises(SecurityStop, match="tool_flagged"):
        await execute(executor, fork(looking))

    # The stop never became a tool result for the model: no second turn was sent.
    assert provider.calls == ["lookup"]
    assert looking.status is NodeStatus.FAILED
    assert executor.security_stop == {"reason_code": "tool_flagged", "at": WALL}


@pytest.mark.asyncio
async def test_another_tool_error_still_goes_back_to_the_model():
    provider = Script(tooling={"lookup"})
    executor = make(provider, None, mode="serial", tool_runtime=Tools(raising(ValueError("no"))))
    looking = node("lookup")

    await execute(executor, fork(looking))

    assert provider.calls == ["lookup", "lookup"] and looking.status is NodeStatus.COMPLETED
    assert executor.security_stop is None


# ---------------------------------------------------------------------------
# Ordinary checkpoints and resume
# ---------------------------------------------------------------------------


def stopped_run(store: Store, control: RunControl) -> tuple[Script, str, dict]:
    """A parallel run that a revoke stopped while ``slow``'s call was in flight."""
    gate = Gate()

    async def watch():
        assert await gate.reached()
        control.revoke("incident_42")
        await hang()

    provider = Script(watch=watch, slow=entering(gate, hang))
    graph = fork(node("done"), node("watch"), node("slow"), node("queued", depends_on=["watch"]))
    swarm = ordinary(provider, store, control, max_concurrency=3, termination=DRAIN)

    with pytest.raises(SecurityStop, match="incident_42"):
        swarm.execute(graph)
    execution_id, state = saved(store)
    return provider, execution_id, state


def test_an_ordinary_stop_is_saved_as_a_version_5_checkpoint():
    store, control = Store(), RunControl()
    provider, _, state = stopped_run(store, control)

    assert state["status"] == "failed" and state["version"] == 5
    assert state["requires"] == ["security_stop"]
    assert state["control"]["security_stop"] == {"reason_code": "incident_42", "at": WALL}
    assert "drain" not in state["control"]
    nodes = saved_nodes(state)
    # The legacy calls in flight keep their reservations as unknown cost.
    assert {node_id: item["status"] for node_id, item in nodes.items()} == {
        "done": "completed", "watch": "pending", "slow": "pending", "queued": "pending"}
    assert state["budget"]["node_costs"] == {"done": CALL, "slow": ESTIMATE, "watch": ESTIMATE}
    for node_id in ("slow", "watch"):
        assert nodes[node_id]["metadata"]["cost_usd_unknown"] is True
        assert nodes[node_id]["metadata"]["cost_usd_is_estimate"] is True
    assert "queued" not in provider.calls
    assert security_stop_from_state(state) == {"reason_code": "incident_42", "at": WALL}


@pytest.mark.parametrize("parallel", [False, True])
def test_ordinary_resume_is_refused_while_the_stop_is_saved(parallel):
    store, control = Store(), RunControl()
    _, execution_id, state = stopped_run(store, control)
    provider = Script()
    fresh = ordinary(provider, store, RunControl(), parallel=parallel)

    with pytest.raises(SecurityStop, match="incident_42"):
        fresh.resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


def test_a_stopped_checkpoint_cannot_be_reevaluated():
    store, control = Store(), RunControl()
    _, execution_id, state = stopped_run(store, control)

    with pytest.raises(SecurityStop, match="incident_42"):
        ordinary(Script(), store, None).reevaluate(
            execution_id, acceptance=AcceptancePolicy(mode="strict"))
    assert store.load(execution_id) == state


def stopped_state(status: str) -> tuple[Store, str, dict]:
    """A saved state with a stop marker, unfinished or completed."""
    store = Store()
    if status == "failed":
        _, execution_id, state = stopped_run(store, RunControl())
        return store, execution_id, state
    graph = fork(node("done", status=NodeStatus.COMPLETED, result="done done"))
    state = build_state(
        execution_id="stopped", status="completed", model="test-model", graph=graph,
        registry=Registry(), task=None, max_budget_usd=None, node_costs={"done": CALL},
        output="done done", control={"revisions_used": 0,
                                     "security_stop": {"reason_code": "late_stop", "at": WALL}},
    )
    store.save("stopped", state)
    return store, "stopped", store.load("stopped")


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_reader_that_knows_only_delivery_acceptance_refuses_the_stop(monkeypatch, status):
    store, execution_id, state = stopped_state(status)
    assert state["version"] == 5 and state["requires"] == ["security_stop"]
    monkeypatch.setattr(checkpoints, "KNOWN_CHECKPOINT_REQUIREMENTS",
                        frozenset({"delivery_acceptance"}))
    provider = Script()

    with pytest.raises(ValueError, match="security_stop"):
        ordinary(provider, store, None).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_reader_of_versions_1_to_4_refuses_the_stop(monkeypatch, status):
    store, execution_id, state = stopped_state(status)
    monkeypatch.setattr(swarm_module, "SUPPORTED_CHECKPOINT_VERSIONS", (1, 2, 3, 4))
    provider = Script()

    with pytest.raises(ValueError, match="version 5"):
        ordinary(provider, store, None).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_completed_checkpoint_with_a_stop_is_refused_too(status):
    store, execution_id, state = stopped_state(status)
    provider = Script()

    with pytest.raises(SecurityStop):
        ordinary(provider, store, None).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("edit,match", [
    pytest.param(lambda state: state.update(requires=[]), "omits", id="unlisted"),
    pytest.param(lambda state: state["control"].pop("security_stop"), "must record",
                 id="listed-without-a-stop"),
    pytest.param(lambda state: state["control"]["security_stop"].update(reason_code="a b"),
                 "reason_code", id="bad-code"),
    pytest.param(lambda state: state["control"]["security_stop"].update(at="today"),
                 "at", id="bad-time"),
    pytest.param(lambda state: state["control"]["security_stop"].update(extra=1),
                 "reason_code and at", id="extra-key"),
])
def test_a_malformed_stop_is_refused(edit, match):
    store, execution_id, state = stopped_state("completed")
    edit(state)
    store.save(execution_id, state)
    provider = Script()

    with pytest.raises(ValueError, match=match):
        security_stop_from_state(state)
    with pytest.raises(ValueError, match=match):
        ordinary(provider, store, None).resume(execution_id)
    assert provider.calls == []


def test_build_state_refuses_a_malformed_stop():
    graph = fork(node("done"))
    with pytest.raises(ValueError, match="reason_code"):
        build_state(execution_id="x", status="failed", model="m", graph=graph,
                    registry=Registry(), task=None, max_budget_usd=None, node_costs={},
                    control={"revisions_used": 0,
                             "security_stop": {"reason_code": "", "at": WALL}})


def test_a_run_without_a_stop_keeps_version_4():
    store = Store()
    ordinary(Script(), store, RunControl()).execute(fork(node("done")))
    _, state = saved(store)
    assert state["version"] == 4 and "requires" not in state
    assert "security_stop" not in state["control"]
    assert security_stop_from_state(state) is None


def test_inspecting_a_stopped_checkpoint_names_the_stop(tmp_path):
    from smythe._run_inspection import inspect_checkpoint
    from smythe.checkpoint import FileCheckpointStore

    files = FileCheckpointStore(tmp_path / "checkpoints")
    memory, execution_id, state = stopped_state("failed")
    files.save(execution_id, state)

    view = inspect_checkpoint(files, execution_id)

    assert view["format"]["requires"] == ["security_stop"]
    assert view["execution"]["blocked_reason"] == "security_stop"
    assert view["next_action"]["action"] == "rerun"
    assert "security stop" in view["next_action"]["detail"]
    assert view["accounting"]["cost_is_complete"] is False


# ---------------------------------------------------------------------------
# Durable runs
# ---------------------------------------------------------------------------


def durable_swarm(store, control, **options) -> Swarm:
    provider = OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)
    return Swarm(model="gpt-6-astra", provider=provider, max_budget_usd=1, run_store=store,
                 architect=SimpleArchitect(), parallel=True, run_control=control, **options)


def durable_graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [Node(id="draft", label="Draft"),
                                                 Node(id="notes", label="Notes")])


def test_a_durable_stop_blocks_the_run_keeps_its_exposure_and_refuses_resume(
        tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    gate, control = Gate(), RunControl()
    transport.schedule("execution", None, gate)
    path = tmp_path / "runs.db"

    async def run_and_revoke(store):
        running = asyncio.ensure_future(durable_swarm(store, control).execute_async(durable_graph()))
        # Both nodes' generation requests have reached the provider and are held there.
        for _ in range(1000):
            if len(transport.requests) == 2:
                break
            await asyncio.sleep(0.01)
        control.revoke("approver_compromised")
        return await asyncio.wait_for(running, 10)

    with SQLiteWorkflowStore(path) as store:
        with pytest.raises(SecurityStop, match="approver_compromised"):
            asyncio.run(run_and_revoke(store))
        (run,) = store.list_runs()
        run_id = run["run_id"]
        loaded = store.load_run(run_id)
        assert (loaded["status"], loaded["blocked_reason"]) == ("blocked", "security_stop")
        stop = store.load_security_stop(run_id)
        assert stop["reason_code"] == "approver_compromised" and type(stop["at_ns"]) is int
        inspected = store.inspect_run(run_id)
        # Both calls were dispatched and cancelled: their exposure stays unknown.
        assert inspected["unknown_calls"] == 2 and inspected["unknown_nanousd"] > 0
        assert inspected["reserved_nanousd"] == 0 and inspected["confirmed_nanousd"] == 0
        events = [event for event in inspected["events"] if event["type"] == "security_stop"]
        assert [event["data"] for event in events] == [{"reason_code": "approver_compromised"}]
        assert store.audit(run_id)["ok"]
        checkpoint = store.get_checkpoint(run_id)["checkpoint"]
        assert checkpoint["status"] == "failed"
        sent = len(transport.requests)
        assert sent == 2

    gate.release()
    with SQLiteWorkflowStore(path) as store:
        before = store.inspect_run(run_id)
        # A fresh control does not reopen the run.
        with pytest.raises(SecurityStop, match="approver_compromised"):
            durable_swarm(store, RunControl()).resume(run_id)
        with pytest.raises(SecurityStop, match="approver_compromised"):
            durable_swarm(store, None).execute(graph_from_dict(checkpoint["graph"]))
        assert len(transport.requests) == sent
        assert store.inspect_run(run_id)["events"] == before["events"]

    with SQLiteWorkflowStore(path, read_only=True) as store:
        view = inspect_journal(store, run_id)
    assert view["execution"]["blocked_reason"] == "security_stop"
    assert view["accounting"]["unknown_exposure"]["calls"] == 2
    assert view["next_action"]["action"] == "rerun"


def test_a_durable_run_under_a_revoked_control_never_starts(tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    control = RunControl()
    control.revoke("account_frozen")
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(SecurityStop, match="account_frozen"):
            durable_swarm(store, control).execute(durable_graph())
        assert store.list_runs() == []
    assert transport.requests == []


def test_a_revoke_during_a_durable_review_cancels_the_calls_in_flight_at_once(
        tmp_path, monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    in_flight, review_gate, control = Gate(), Gate(), RunControl()
    # The first node's call is answered and reviewed; the second's is held in flight.
    transport.schedule("execution", 1, in_flight)
    transport.schedule("supervision", None, review_gate)
    reviewer = LLMSupervisor(OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100),
                             model="gpt-6-astra", review_after={"draft", "notes"})

    async def run_and_revoke(store):
        running = asyncio.ensure_future(durable_swarm(
            store, control, supervisor=reviewer, max_revisions=1,
        ).execute_async(durable_graph()))
        # Both nodes' calls and the review's call have reached the provider.
        for _ in range(1000):
            if len(transport.requests) == 3 and review_gate.entered.is_set():
                break
            await asyncio.sleep(0.01)
        control.revoke("approver_compromised")
        try:
            # The gates hold both calls until they are released below and give up only after
            # 30 s, so a run that ends within this bound cancelled them. The bound leaves room
            # for the journal's commits that record the stop on a loaded host.
            await asyncio.wait({running}, timeout=15)
            assert running.done() and not review_gate.released and not in_flight.released
        finally:
            review_gate.release()
            in_flight.release()
            await asyncio.wait({running}, timeout=30)
        return running.result()

    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(SecurityStop, match="approver_compromised"):
            asyncio.run(run_and_revoke(store))
        (run,) = store.list_runs()
        loaded = store.load_run(run["run_id"])
        assert (loaded["status"], loaded["blocked_reason"]) == ("blocked", "security_stop")
        inspected = store.inspect_run(run["run_id"])
    phases = sorted(phase_of(request["instructions"]) for request in transport.requests)
    assert phases == ["execution", "execution", "supervision"]
    # The held node call and the held review call were both sent and cancelled.
    unknown = sorted(call["key"]["phase"] for call in inspected["calls"]
                     if call["billing_state"] == "unknown")
    assert unknown == ["execution", "supervision"]
    assert inspected["unknown_calls"] == 2 and inspected["reserved_nanousd"] == 0


@pytest.fixture
def journal(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


def offline_run(store) -> tuple[Swarm, str]:
    """A planned, unfinished offline durable run, with the Swarm that planned it."""
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1)
    return swarm, swarm.plan(Task("Report")).run_ref["run_id"]


def test_the_store_stop_is_lease_free_first_wins_and_survives_every_latch(journal):
    swarm, run_id = offline_run(journal)
    assert journal.load_security_stop(run_id) is None

    first = journal.security_stop(run_id, "credential_leak")
    again = journal.security_stop(run_id, "other_reason")

    assert first == again and first["reason_code"] == "credential_leak"
    run = journal.load_run(run_id)
    assert (run["status"], run["blocked_reason"]) == ("blocked", "security_stop")
    events = [event for event in journal.inspect_run(run_id)["events"]
              if event["type"] == "security_stop"]
    assert len(events) == 1
    # The admission latch keeps a reason it does not own.
    with journal._transaction() as db:
        journal._update_latch(db, run_id)
    assert journal.load_run(run_id)["blocked_reason"] == "security_stop"
    with pytest.raises(SecurityStop, match="credential_leak"):
        swarm.resume(run_id)
    assert journal.load_run(run_id)["blocked_reason"] == "security_stop"


@pytest.mark.parametrize("code", ["", "has space", "x" * 65, 7])
def test_the_store_stop_takes_only_a_short_code(journal, code):
    _, run_id = offline_run(journal)
    with pytest.raises(WorkflowValidationError, match="reason_code"):
        journal.security_stop(run_id, code)
    assert journal.load_run(run_id)["blocked_reason"] is None


def test_a_completed_durable_run_can_be_stopped_and_is_then_refused(journal):
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=journal, max_budget_usd=1)
    run_id = swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="first", label="First")])
                           ).execution_id
    assert swarm.resume(run_id).execution_id == run_id

    journal.security_stop(run_id, "late_revocation")

    assert journal.load_run(run_id)["blocked_reason"] == "security_stop"
    with pytest.raises(SecurityStop, match="late_revocation"):
        swarm.resume(run_id)
    assert journal.audit(run_id)["ok"]


def test_no_later_journal_transition_clears_a_stop(tmp_path):
    from test_workflow_store import Clock, dispatch, envelope, quote, response
    from test_workflow_store import prepare as prepare_call

    with SQLiteWorkflowStore(tmp_path / "workflow.sqlite", clock_ns=Clock()) as store:
        store.create_run({"goal": "goal"}, {"config": 1}, 10**12, run_id="run")
        lease = store.acquire_lease("run", "owner")
        conflicted = prepare_call(store, lease, "conflicted")
        conflicted_permit = dispatch(store, lease, conflicted)
        original = store.append_response(conflicted_permit, envelope(conflicted, response()))
        costly = prepare_call(store, lease, "costly")
        costly_permit = dispatch(store, lease, costly)
        held = prepare_call(store, lease, "held")
        store.reserve_call(lease, held["call_id"], quote(store, lease, held)["quote_id"])

        store.security_stop("run", "credential_leak")

        # A call reserved before the stop is never dispatched after it.
        with pytest.raises(WorkflowStateError, match="security_stop"):
            store.claim_dispatch(lease, held["call_id"], held["request_sha256"])
        # A conflicting late response is kept without replacing the stop's reason.
        with pytest.raises(WorkflowConflictError):
            store.append_response(conflicted_permit,
                                  envelope(conflicted, response(text="different")))
        store.settle_call(lease, conflicted["call_id"], original)
        # So is a known charge above the budget, which would close admission itself.
        evidence = store.append_response(costly_permit,
                                         envelope(costly, response(outputs=30_000_000)))
        settled = store.settle_call(lease, costly["call_id"], evidence)
        assert settled["blocked_reason"] == "security_stop" and settled["admission_closed"]
        store.recover(lease)
        run = store.load_run("run")
        assert (run["status"], run["blocked_reason"]) == ("blocked", "security_stop")
        with pytest.raises(WorkflowStateError, match="security_stop"):
            prepare_call(store, lease, "later")
        assert store.audit("run")["ok"]


# ---------------------------------------------------------------------------
# External actions
# ---------------------------------------------------------------------------


from test_workflow_acceptance import replies  # noqa: E402,F401


def test_a_security_stopped_run_refuses_action_preparation_and_approval(tmp_path, replies):  # noqa: F811
    from smythe.actions import ActionNotAuthorized, check_authority
    from test_action_approvals import Host, approved, completed, prepare

    clock = ManualClock(time.time_ns())
    host = Host()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        action = prepare(store, completed(store, tmp_path, replies))
        approval = approved(store, host, action, clock)
        assert check_authority(store, action.action_id, host) == approval

        store.security_stop(action.run_id, "approver_compromised")

        with pytest.raises(ActionNotAuthorized, match="a security stop holds the run"):
            check_authority(store, action.action_id, host)
        with pytest.raises(ActionNotAuthorized, match="a security stop holds the run"):
            prepare(store, action.run_id, slot="announce")
        with pytest.raises(ActionNotAuthorized, match="a security stop holds the run"):
            approved(store, host, action, clock)
        assert store.audit(action.run_id)["ok"]
