"""A durable run's opt-in drain lets its dispatched siblings settle after a failure.

Without a termination policy, a node that fails under ``HALT`` cancels its
running siblings at once: a sibling whose generation request was already sent
keeps unknown exposure at its quoted ceiling, and the run is blocked with
``unknown_exposure``. Under ``TerminationPolicy(drain_timeout_s=...)``, which
the run's recipe records, the run admits nothing more and waits for those
requests until the deadline: a sibling answered in time settles its known
charge, the failed checkpoint consumes its result, and resume never calls it
again. A sibling still unanswered at the deadline is cancelled and keeps its
unknown exposure, a security stop ends the drain at once, and a process lost
during the drain leaves its sent call to recovery, which marks it unknown.

The wire fake holds the sibling's generation request, or its input-token
count, at a gate, and refuses the halting node's first request with a 401,
a zero-cost rejection, once the sibling has reached the gate. Patching
``AsyncExecutor._drain_wait`` controls the drain's deadline: releasing the
gate first lets the sibling answer within it, and returning at once makes it
pass.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from fakes import (
    Crash, Gate, HttpError, ManualClock, NativeTransport, crash_when, kill_and_reap,
    launch_child, wait_for_barrier,
)
from smythe import (
    OfflineProvider, ProviderRequestRejectedError, RunControl, SecurityStop, SimpleArchitect,
    SQLiteWorkflowStore, Swarm, Task, TerminationPolicy,
)
from smythe.async_executor import AsyncExecutor
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import CallKey, WorkflowConflictError, WorkflowStateError

# Each NativeTransport generation bills 1,500,000 nanoUSD; its quote reserves this ceiling.
CEILING = 6_250_000
CHARGE = 1_500_000
DRAIN_S = 30.0
DRAIN = TerminationPolicy(drain_timeout_s=DRAIN_S)
UNAUTHORIZED = HttpError(401, "invalid_api_key", "Incorrect API key provided")


class Wire(NativeTransport):
    """``NativeTransport`` that holds the sibling at *held* and refuses the halting node once.

    With *held*, the sibling's request of operation *at* (its generation
    request, or its input-token count) reaches the provider and waits at the
    gate, and the halting node's first generation request waits until the
    sibling has reached it, then gets a 401 rejection. Without *held*, every
    call is answered.
    """

    def __init__(self, held: Gate | None = None, *, at: str = "response") -> None:
        super().__init__()
        self.held, self.at, self.refused = held, at, held is None

    async def _call(self, operation, payload):
        prompt = self._prompt(payload)
        if self.held is not None and operation == self.at and prompt.startswith("Sibling"):
            self.schedule("execution", self._ordinals[operation, "execution"], self.held,
                          operation=operation)
        elif operation == "response" and prompt.startswith("Halt") and not self.refused:
            self.refused = True
            assert await self.held.reached()
            # Nothing else is sent meanwhile: the ordinal is this request's.
            self.schedule("execution", self._ordinals[operation, "execution"], UNAUTHORIZED)
        return await super()._call(operation, payload)


def durable(store, *, termination=DRAIN, control=None, **options) -> Swarm:
    provider = OpenAIResponsesProvider(api_key="no-network", max_output_tokens=100)
    return Swarm(model="gpt-6-astra", provider=provider, run_store=store, parallel=True,
                 max_concurrency=2, max_budget_usd=1, termination=termination,
                 run_control=control, **options)


def graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [Node(id="halt", label="Halt"),
                                                 Node(id="sibling", label="Sibling")])


def calls(inspected, node_id):
    """A node's generation calls, as ``inspect_run`` summarizes them, oldest first."""
    return [call for call in inspected["calls"]
            if call["key"]["scope_id"] == f"node/{node_id}" and call["key"]["phase"] == "execution"]


def events(inspected, call_id):
    return [event["type"] for event in inspected["events"] if event["call_id"] == call_id]


def sent(wire, label):
    return sum(wire._prompt(payload).startswith(label) for payload in wire.requests)


def counted(wire, label):
    return sum(wire._prompt(payload).startswith(label) for payload in wire.counts)


def saved_nodes(saved):
    return {item["id"]: item for item in saved["checkpoint"]["graph"]["nodes"]}


def deadline(monkeypatch, then):
    """Replace the drain's deadline wait with *then*; return the waits it was asked for."""
    waits = []

    async def drain_wait(self, seconds):
        waits.append(seconds)
        await then(self, seconds)

    monkeypatch.setattr(AsyncExecutor, "_drain_wait", drain_wait)
    return waits


def answering(monkeypatch, held):
    """The drain releases the held sibling, then waits out its deadline."""
    wait = AsyncExecutor._drain_wait

    async def release(self, seconds):
        held.release()
        await wait(self, seconds)

    return deadline(monkeypatch, release)


def expiring(monkeypatch):
    """The drain's deadline passes at once."""

    async def passed(self, seconds):
        return None

    return deadline(monkeypatch, passed)


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------


def offline(store, termination) -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, parallel=True, termination=termination)


def test_the_recipe_records_a_policy_only_when_one_is_configured(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        legacy = offline(store, None).plan(Task("Report")).run_ref["run_id"]
        drained = offline(store, DRAIN).plan(Task("Report")).run_ref["run_id"]
        legacy_recipe = store.load_run(legacy)["config"]
        drained_recipe = store.load_run(drained)["config"]

    # A recipe without a policy keeps the fields, and so the identity, it had.
    assert "termination" not in legacy_recipe
    assert drained_recipe.pop("termination") == {"drain_timeout_s": DRAIN_S}
    assert drained_recipe == legacy_recipe


def test_a_run_keeps_its_policy_and_refuses_a_different_one_on_resume(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        legacy = offline(store, None).plan(Task("Report")).run_ref["run_id"]
        drained = offline(store, DRAIN).plan(Task("Report")).run_ref["run_id"]
        for swarm, run_id in ((offline(store, TerminationPolicy(drain_timeout_s=5)), drained),
                              (offline(store, DRAIN), legacy)):
            with pytest.raises(WorkflowConflictError, match="termination differs"):
                swarm.resume(run_id)
        assert [store.inspect_run(run_id)["call_count"] for run_id in (drained, legacy)] == [0, 0]

        # Resuming without a policy uses the saved one; the same policy is accepted.
        runtime = offline(store, None)._workflow_runtime()
        asyncio.run(runtime.resume(drained, max_concurrency=2))
        assert runtime.behavior["termination"] == {"drain_timeout_s": DRAIN_S}
        assert runtime.executor._termination == DRAIN
        offline(store, DRAIN).resume(drained)
        plain = offline(store, None)._workflow_runtime()
        asyncio.run(plain.resume(legacy, max_concurrency=2))
        assert plain.behavior["termination"] is None and plain.executor._termination is None
        assert store.load_run(drained)["status"] == store.load_run(legacy)["status"] == "completed"


def test_only_a_termination_policy_is_recorded(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = offline(store, None)
        swarm.termination = {"drain_timeout_s": DRAIN_S}
        with pytest.raises(WorkflowBindingError, match="^termination must be a TerminationPolicy"):
            swarm._workflow_runtime()
        # The recipe holds exactly the record a valid policy gives.
        for value in ({"drain_timeout_s": 0}, {"drain_timeout_s": 30},
                      {"drain_timeout_s": 30.0, "x": 1}, [30.0], 30.0):
            with pytest.raises(WorkflowBindingError, match="termination"):
                offline(store, None)._workflow_runtime(behavior_settings={"termination": value})
        settings = offline(store, None)._workflow_runtime(
            behavior_settings={"termination": {"drain_timeout_s": DRAIN_S}})
        assert settings.recipe["termination"] == {"drain_timeout_s": DRAIN_S}
        with pytest.raises(TypeError, match="termination must be a TerminationPolicy"):
            Swarm(model="offline", provider=OfflineProvider(), run_store=store, termination=30)
        assert store.list_runs() == []


# ---------------------------------------------------------------------------
# Draining a dispatched sibling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("every", [1, 100], ids=["each-node", "batched"])
def test_a_sibling_answered_within_the_drain_settles_known_and_resume_never_calls_it(
        tmp_path, monkeypatch, every):
    held, path = Gate(), tmp_path / "runs.db"
    first = Wire(held).install(monkeypatch)
    waits = answering(monkeypatch, held)
    with SQLiteWorkflowStore(path) as store:
        with pytest.raises(ProviderRequestRejectedError, match="HTTP 401"):
            durable(store, checkpoint_every_n_nodes=every).execute(graph())
        (run,) = store.list_runs()
        run_id = run["run_id"]
        inspected, saved = store.inspect_run(run_id), store.get_checkpoint(run_id)
    assert waits == [DRAIN_S]
    assert (sent(first, "Halt"), sent(first, "Sibling")) == (1, 1)
    (sibling,), (refused,) = calls(inspected, "sibling"), calls(inspected, "halt")
    # The sibling settled its known charge, and the refusal cost nothing.
    assert inspected["ok"] and inspected["blocked_reason"] is None
    assert (inspected["unknown_calls"], inspected["reserved_nanousd"]) == (0, 0)
    assert inspected["confirmed_nanousd"] == CHARGE
    assert (sibling["billing_state"], sibling["result_state"], sibling["cost_nanousd"]) == (
        "known", "applied", CHARGE)
    assert (refused["billing_state"], refused["result_state"], refused["cost_nanousd"]) == (
        "known", "rejected", 0)
    # The failed checkpoint keeps the sibling's result and consumes both calls.
    nodes = saved_nodes(saved)
    assert saved["checkpoint"]["status"] == "failed"
    assert (nodes["halt"]["status"], nodes["sibling"]["status"]) == ("failed", "completed")
    assert nodes["sibling"]["result"] == "Draft accepted output."
    assert {sibling["call_id"], refused["call_id"]} <= set(saved["consumed_call_ids"])

    second = Wire().install(monkeypatch)
    with SQLiteWorkflowStore(path) as store:
        # A Swarm without a policy resumes under the saved one.
        result = durable(store, termination=None, checkpoint_every_n_nodes=every).resume(run_id)
        after = store.inspect_run(run_id)
    # Only the refused node is called again; the sibling's result is kept.
    assert (sent(second, "Halt"), sent(second, "Sibling")) == (1, 0)
    assert after["status"] == "completed" and after["ok"]
    assert (after["confirmed_nanousd"], after["reserved_nanousd"], after["unknown_nanousd"]) == (
        2 * CHARGE, 0, 0)
    assert calls(after, "sibling") == [sibling]
    assert result.cost_is_complete and result.output


@pytest.mark.parametrize("termination", [None, DRAIN], ids=["no-policy", "deadline"])
def test_a_sibling_unanswered_when_the_run_stops_keeps_unknown_exposure_and_blocks_it(
        tmp_path, monkeypatch, termination):
    held = Gate()
    wire = Wire(held).install(monkeypatch)
    waits = expiring(monkeypatch)
    try:
        with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
            with pytest.raises(ProviderRequestRejectedError, match="HTTP 401"):
                durable(store, termination=termination).execute(graph())
            (run,) = store.list_runs()
            run_id = run["run_id"]
            inspected, saved = store.inspect_run(run_id), store.get_checkpoint(run_id)
            with pytest.raises(WorkflowStateError, match="unknown_exposure"):
                durable(store, termination=termination).resume(run_id)
            audit = store.audit(run_id)
    finally:
        held.release()
    # Without a policy nothing drains; under one, the deadline passed first.
    assert waits == ([] if termination is None else [DRAIN_S])
    (sibling,) = calls(inspected, "sibling")
    assert (sibling["state"], sibling["unknown_nanousd"], sibling["reserved_nanousd"]) == (
        "unknown", CEILING, 0)
    assert inspected["blocked_reason"] == "unknown_exposure"
    assert (inspected["unknown_calls"], inspected["unknown_nanousd"]) == (1, CEILING)
    assert inspected["confirmed_nanousd"] == 0 and audit["ok"]
    nodes = saved_nodes(saved)
    assert saved["checkpoint"]["status"] == "failed" and nodes["sibling"]["status"] == "failed"
    # Resume is refused before any call: the exposure has no reconciliation.
    assert (sent(wire, "Halt"), sent(wire, "Sibling")) == (1, 1)


def test_a_sibling_the_deadline_cancels_before_its_reservation_holds_nothing(tmp_path, monkeypatch):
    held, path = Gate(), tmp_path / "runs.db"
    first = Wire(held, at="input_tokens").install(monkeypatch)
    waits = expiring(monkeypatch)
    try:
        with SQLiteWorkflowStore(path) as store:
            with pytest.raises(ProviderRequestRejectedError, match="HTTP 401"):
                durable(store).execute(graph())
            (run,) = store.list_runs()
            run_id = run["run_id"]
            inspected = store.inspect_run(run_id)
    finally:
        held.release()
    assert waits == [DRAIN_S] and (sent(first, "Sibling"), counted(first, "Sibling")) == (0, 1)
    # Its input-token count was cancelled: nothing was reserved or sent.
    (sibling,) = calls(inspected, "sibling")
    assert (sibling["state"], sibling["reserved_nanousd"], sibling["unknown_nanousd"]) == (
        "prepared", 0, 0)
    assert inspected["blocked_reason"] is None and inspected["unknown_calls"] == 0

    second = Wire().install(monkeypatch)
    with SQLiteWorkflowStore(path) as store:
        result = durable(store).resume(run_id)
        after = store.inspect_run(run_id)
    assert (sent(second, "Halt"), sent(second, "Sibling")) == (1, 1)
    assert after["status"] == "completed" and after["ok"]
    assert (after["confirmed_nanousd"], after["reserved_nanousd"]) == (2 * CHARGE, 0)
    assert result.cost_is_complete


def test_a_security_stop_during_a_durable_drain_cancels_the_sibling_at_once(tmp_path, monkeypatch):
    held, control, path = Gate(), RunControl(), tmp_path / "runs.db"
    wire = Wire(held).install(monkeypatch)

    async def revoke(self, seconds):
        control.revoke("approver_compromised")
        await asyncio.Event().wait()  # Only the revocation can end this drain.

    waits = deadline(monkeypatch, revoke)

    async def run(store):
        return await asyncio.wait_for(durable(store, control=control).execute_async(graph()), 10)

    try:
        with SQLiteWorkflowStore(path) as store:
            with pytest.raises(SecurityStop, match="approver_compromised") as stopped:
                asyncio.run(run(store))
            (run_record,) = store.list_runs()
            run_id = run_record["run_id"]
            inspected, saved = store.inspect_run(run_id), store.get_checkpoint(run_id)
            with pytest.raises(SecurityStop, match="approver_compromised"):
                durable(store, control=RunControl()).resume(run_id)
    finally:
        held.release()
    assert waits == [DRAIN_S]
    assert isinstance(stopped.value.__cause__, ProviderRequestRejectedError)
    assert (inspected["status"], inspected["blocked_reason"]) == ("blocked", "security_stop")
    (sibling,) = calls(inspected, "sibling")
    assert (sibling["state"], sibling["unknown_nanousd"]) == ("unknown", CEILING)
    assert inspected["ok"] and saved["checkpoint"]["status"] == "failed"
    assert (sent(wire, "Halt"), sent(wire, "Sibling")) == (1, 1)


# ---------------------------------------------------------------------------
# A process lost during a drain
# ---------------------------------------------------------------------------


def test_a_process_lost_during_a_drain_before_its_dispatch_claim_completes_on_resume(
        tmp_path, monkeypatch):
    # The drain lets the sibling count its input and reserve its quote; the
    # process is lost before it claims the dispatch, so the image holds a
    # reserved call that was never sent.
    held, image = Gate(), tmp_path / "killed.db"
    first = Wire(held, at="input_tokens").install(monkeypatch)
    answering(monkeypatch, held)
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        def sibling(lease, call_id, _):
            record = store.lookup_call(lease.run_id, CallKey("execution", "node/sibling"))
            return record is not None and record["call_id"] == call_id

        point = crash_when(monkeypatch, store, "claim_dispatch", sibling, image=image,
                           connection=store._db)
        with pytest.raises(Crash):
            durable(store).execute(graph())
        run_id = store.list_runs()[0]["run_id"]
    assert point.fired
    assert (sent(first, "Halt"), sent(first, "Sibling"), counted(first, "Sibling")) == (1, 0, 1)

    second = Wire().install(monkeypatch)
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        (orphan,) = calls(recovered.inspect_run(run_id), "sibling")
        assert (orphan["state"], orphan["reserved_nanousd"], orphan["evidence_id"]) == (
            "reserved", CEILING, None)
        result = durable(recovered).resume(run_id)
        after = recovered.inspect_run(run_id)
    # Its node is current, so resume sends it under its saved quote and reservation.
    (current,) = calls(after, "sibling")
    history = events(after, current["call_id"])
    assert current["call_id"] == orphan["call_id"]
    assert (current["state"], current["result_state"]) == ("settled", "applied")
    assert history.count("call_reserved") == history.count("dispatch_claimed") == 1
    assert "call_released" not in history
    assert (sent(second, "Halt"), sent(second, "Sibling"), counted(second, "Sibling")) == (1, 1, 0)
    assert after["status"] == "completed" and after["ok"]
    assert (after["confirmed_nanousd"], after["reserved_nanousd"], after["unknown_nanousd"]) == (
        2 * CHARGE, 0, 0)
    assert result.cost_is_complete


# The child runs a drained run and holds it inside the drain, with the
# sibling's generation request sent and unanswered, until the test kills it.
CHILD = r"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fakes import Gate, write_barrier
from smythe import SQLiteWorkflowStore
from smythe.async_executor import AsyncExecutor
from test_workflow_drain import Wire, durable, graph

path, folder = Path(sys.argv[1]), Path(sys.argv[2])


async def drain_wait(self, seconds):
    write_barrier(folder / "draining.json", {"seconds": seconds})
    time.sleep(600)


AsyncExecutor._drain_wait = drain_wait
# Serve the fake transport to the provider, as pytest's monkeypatch does.
Wire(Gate()).install(SimpleNamespace(setattr=setattr))
with SQLiteWorkflowStore(path) as store:
    durable(store).execute(graph())
"""


def test_a_process_killed_during_a_drain_leaves_its_sent_call_unknown_after_recovery(
        tmp_path, monkeypatch):
    path = tmp_path / "runs.db"
    process = launch_child(CHILD, path, tmp_path)
    try:
        assert wait_for_barrier(process, tmp_path / "draining.json", timeout=60) == {
            "seconds": DRAIN_S}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0

    resumed = Wire().install(monkeypatch)
    # The killed child still holds the lease; a successor takes it once it lapses.
    with SQLiteWorkflowStore(path, clock_ns=ManualClock.after(120)) as store:
        (run,) = store.list_runs()
        run_id = run["run_id"]
        before = store.inspect_run(run_id)
        (dispatched,) = calls(before, "sibling")
        assert (dispatched["state"], dispatched["evidence_id"], dispatched["reserved_nanousd"]) == (
            "dispatched", None, CEILING)
        assert before["ok"] and before["blocked_reason"] is None
        lease = store.acquire_lease(run_id, uuid4().hex, ttl_s=60)
        plan = store.recover(lease)
        store.release_lease(lease)
        with pytest.raises(WorkflowStateError, match="unknown_exposure"):
            durable(store).resume(run_id)
        after, audit = store.inspect_run(run_id), store.audit(run_id)
    # Recovery cannot know whether the sent request was charged: it is unknown exposure.
    assert plan["unknown"] == [dispatched["call_id"]]
    assert plan["blocked_reason"] == "unknown_exposure"
    (lost,) = calls(after, "sibling")
    assert (lost["state"], lost["billing_state"], lost["unknown_nanousd"],
            lost["reserved_nanousd"]) == ("unknown", "unknown", CEILING, 0)
    assert audit["ok"] and after["blocked_reason"] == "unknown_exposure"
    assert (audit["unknown_calls"], audit["unknown_nanousd"], audit["reserved_nanousd"],
            audit["confirmed_nanousd"]) == (1, CEILING, 0, 0)
    assert sum(call["unknown_nanousd"] for call in after["calls"]) == after["unknown_nanousd"]
    assert resumed.requests == [] and resumed.counts == []
