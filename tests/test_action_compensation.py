"""Revocation takes precedence over dispatch, and never undoes an effect.

``stop_actions`` is a fail-safe control that revokes every approval a run's
actions hold. A submission already claimed is not recalled: its outcome is
recorded when it arrives. Every later claim is refused, and nothing a later
worker does clears the stop; only a new approval reopens an action.
``ActionRunner.dispatch_pending`` checks each action's authority again
before each claim. Revoking never undoes an effect, and inspection shows when
a revocation came.
"""

from __future__ import annotations

import inspect
import json
import threading
import time

import pytest

from fakes import Gate, ManualClock
from fixture_service import Fault, FixtureService
from smythe import SQLiteWorkflowStore, TerminationPolicy, workflow_actions
from smythe.actions import (
    ActionNotAuthorized,
    ActionRunner,
    approve,
    check_authority,
    load_action,
    prepare_action,
    revoke,
    stop_actions,
)
from test_action_approvals import HOUR_NS, ORGANIZATION, Host, completed, delivered, intent, sign, swarm
from test_action_journal import WRITE, balances, item, rows
from test_action_runner import Recording, ready, runner
from test_workflow_acceptance import replies  # noqa: F401


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(time.time_ns())


@pytest.fixture
def store(tmp_path, clock):
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as value:
        yield value


@pytest.fixture
def host() -> Host:
    return Host()


@pytest.fixture
def service(clock) -> FixtureService:
    return FixtureService(clock_ns=clock)


@pytest.fixture
def connector(service) -> Recording:
    return Recording(service)


def approve_now(store, host, clock, action):
    return approve(store, action.action_id, sign(host, action), host,
                   ceiling_nanousd=action.ceiling_nanousd, expires_at_ns=clock.now_ns + HOUR_NS)


def queued(store, tmp_path, replies, host, clock, connector, slots=("first", "second", "third")):  # noqa: F811
    """A completed run with one approved action in each slot, in dispatch order."""
    run_id = completed(store, tmp_path, replies)
    actions = []
    for slot in slots:
        action = prepare_action(store, run_id, slot=slot, intent=intent(delivered(store, run_id), title=slot),
                                profile=connector.profile)
        approve_now(store, host, clock, action)
        actions.append(action)
    return actions


def held(service, action, *, lost=False) -> Gate:
    """Hold the action's first delivery after the service commits it, until the gate is released."""
    gate = Gate()
    service.schedule(action.action_key, Fault(hold_after_commit=gate, drop_ack=lost))
    return gate


def in_thread(step, *args):
    """Run a runner step in a thread; return the thread and a dict that gets its ``result``."""
    outcome: dict = {}

    def work():
        try:
            outcome["result"] = step(*args)
        except BaseException as error:  # reported by the test, not lost in the thread
            outcome["error"] = error

    thread = threading.Thread(target=work)
    thread.start()
    return thread, outcome


def finished(thread, outcome):
    thread.join(30)
    assert not thread.is_alive()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def summary(progress) -> list[tuple]:
    return [(item.action_id, item.state, item.next_action, item.refusal) for item in progress]


def revocations(store, action) -> list:
    return [approval.get("revocation") for approval in item(store, action)["approvals"]]


def observed(store, connector, host, clock, action):
    """Dispatch the action and read its result back; it must be observed."""
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    return worker


# ---------------------------------------------------------------------------
# Revocation takes precedence over dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lost", [False, True], ids=["acknowledged", "acknowledgement-lost"])
def test_a_stop_during_a_held_submission_records_it_and_refuses_the_queued_actions(
        store, tmp_path, replies, host, clock, service, connector, lost):  # noqa: F811
    first, second, third = queued(store, tmp_path, replies, host, clock, connector)
    gate = held(service, first, lost=lost)
    worker = runner(store, connector, host, clock)
    started = clock.monotonic()

    thread, outcome = in_thread(worker.dispatch_pending, first.run_id)
    try:
        assert gate.entered.wait(10)
        # Another process stops the run's actions while the first submission is held, and the
        # worker still holds the run's lease: the stop needs none.
        with SQLiteWorkflowStore(store.path, clock_ns=clock) as control:
            revoked = stop_actions(control, first.run_id, f"The approver left {ORGANIZATION}")
    finally:
        gate.release()
    progress = finished(thread, outcome)

    approvals = rows(store.path, "workflow_action_approvals")
    assert revoked == tuple(row["approval_id"] for row in approvals)
    # The submission in flight is recorded as it arrives; the queued actions are refused before any claim.
    in_flight = progress[0]
    assert (in_flight.action_id, in_flight.refusal) == (first.action_id, None)
    assert (in_flight.state, in_flight.attempt["state"], in_flight.next_action) == (
        ("outcome_unknown", "outcome_unknown", "reconcile") if lost
        else ("dispatched", "acknowledged", "readback"))
    assert [entry[:3] for entry in summary(progress[1:])] == [
        (second.action_id, "not_dispatched", None), (third.action_id, "not_dispatched", None)]
    assert all("its approval was revoked" in entry.refusal for entry in progress[1:])
    assert connector.made("submit") == 1
    assert [row["action_id"] for row in rows(store.path, "workflow_action_attempts")] == [first.action_id]
    snapshot = service.snapshot()
    assert (len(snapshot["deliveries"]), snapshot["totals"]["committed"]) == (1, 1)
    # Nothing waited for the queue to drain.
    assert clock.monotonic() == started
    # The reason is kept redacted, and inspection shows only that and when the stop came.
    assert {row["revoked_reason"] for row in approvals} == {"The approver left org-[redacted]"}
    inspected = store.inspect_run(first.run_id)
    assert "The approver left" not in json.dumps(inspected)
    (stop,) = [event["data"] for event in inspected["events"] if event["type"] == "actions_stopped"]
    assert stop == {"revoked": list(revoked)}
    assert store.audit(first.run_id)["ok"]

    # Resuming the run, a new lease, recovery and a reopened journal never clear the stop.
    assert swarm(store, tmp_path).resume(first.run_id).execution_id == first.run_id
    store.release_lease(store.acquire_lease(first.run_id, "another-worker", ttl_s=60))
    with SQLiteWorkflowStore(store.path, clock_ns=clock) as reopened:
        again = runner(reopened, connector, host, clock).dispatch_pending(first.run_id)
        assert [entry[:3] for entry in summary(again)] == [
            (second.action_id, "not_dispatched", None), (third.action_id, "not_dispatched", None)]
        for action in (second, third):
            with pytest.raises(ActionNotAuthorized, match="its approval was revoked"):
                check_authority(reopened, action.action_id, host)
            with pytest.raises(ActionNotAuthorized, match="its approval was revoked"):
                runner(reopened, connector, host, clock).dispatch(action.run_id, action.action_id)
    assert connector.made("submit") == 1

    # The in-flight outcome can still be settled: reads send nothing.
    settled = (worker.reconcile if lost else worker.readback)(first.run_id, first.action_id)
    assert settled.state == ("reconciled" if lost else "observed")
    assert revocations(store, first) == ["after_effect"]
    assert revocations(store, second) == revocations(store, third) == ["before_dispatch"]

    # Only a new approval reopens an action, and only that one.
    approve_now(store, host, clock, third)
    renewed = worker.dispatch_pending(first.run_id)
    assert [entry[:3] for entry in summary(renewed)] == [
        (second.action_id, "not_dispatched", None), (third.action_id, "dispatched", "readback")]
    assert renewed[1].refusal is None
    assert connector.made("submit") == 2
    assert service.snapshot()["totals"]["committed"] == 2
    assert store.audit(first.run_id)["ok"]


def test_a_stop_after_the_authority_check_refuses_the_claim(
        store, tmp_path, replies, host, clock, service, connector, monkeypatch):  # noqa: F811
    first, second = queued(store, tmp_path, replies, host, clock, connector, slots=("first", "second"))
    begin = workflow_actions.begin_attempt

    def begin_then_stop(store_, lease, action_id, approval_id):
        attempt = begin(store_, lease, action_id, approval_id)
        stop_actions(store_, lease.run_id, "stopped between the check and the claim")
        return attempt

    monkeypatch.setattr(workflow_actions, "begin_attempt", begin_then_stop)

    progress = runner(store, connector, host, clock).dispatch_pending(first.run_id)

    # The claim checks the journal again in its own transaction: the begun attempt is released unsent.
    assert [entry[:3] for entry in summary(progress)] == [
        (first.action_id, "not_dispatched", None), (second.action_id, "not_dispatched", None)]
    assert all("its approval was revoked" in entry.refusal for entry in progress)
    assert connector.calls == [] and service.snapshot()["deliveries"] == []
    (attempt,) = rows(store.path, "workflow_action_attempts")
    assert (attempt["action_id"], attempt["state"], attempt["dispatched"]) == (first.action_id, "released", None)
    assert balances(item(store, first)) == (0, 0, 0)
    assert store.audit(first.run_id)["ok"]


def test_dispatch_pending_works_through_the_runs_actions_in_slot_order(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    first, second, third = queued(store, tmp_path, replies, host, clock, connector)
    unapproved = prepare_action(store, first.run_id, slot="fourth",
                                intent=intent(delivered(store, first.run_id), title="fourth"),
                                profile=connector.profile)
    other = Recording(service, readback_reads=3)
    elsewhere = prepare_action(store, first.run_id, slot="elsewhere",
                               intent=intent(delivered(store, first.run_id), title="elsewhere"),
                               profile=other.profile)
    approve_now(store, host, clock, elsewhere)
    revoke(store, second.action_id, "withdrawn")

    progress = runner(store, connector, host, clock).dispatch_pending(first.run_id)

    # An action of another connector's profile is left to a runner of that connector.
    assert [entry[:3] for entry in summary(progress)] == [
        (first.action_id, "dispatched", "readback"), (unapproved.action_id, "not_dispatched", None),
        (second.action_id, "not_dispatched", None), (third.action_id, "dispatched", "readback")]
    assert "it has no approval" in progress[1].refusal
    assert "its approval was revoked" in progress[2].refusal
    assert [call[1] for call in connector.calls] == [first.action_key, third.action_key]
    assert load_action(store, elsewhere.action_id).state == "not_dispatched"
    # Dispatched actions wait for their readback, so a second pass dispatches nothing more.
    again = runner(store, connector, host, clock).dispatch_pending(first.run_id)
    assert [entry.action_id for entry in again] == [unapproved.action_id, second.action_id]
    assert connector.made("submit") == 2


def test_action_dispatch_is_never_subject_to_the_model_call_drain_policy(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    # A drain lets model calls in flight finish under the async executor; the action runner takes none.
    assert "termination" not in inspect.signature(ActionRunner).parameters
    with pytest.raises(TypeError, match="termination"):
        ActionRunner(store, connector, host, termination=TerminationPolicy(drain_timeout_s=30.0))
    first, second = queued(store, tmp_path, replies, host, clock, connector, slots=("first", "second"))
    gate = held(service, first)
    sleeps: list[float] = []
    worker = ActionRunner(store, connector, host, sleep=sleeps.append, clock=clock.monotonic)

    thread, outcome = in_thread(worker.dispatch_pending, first.run_id)
    try:
        assert gate.entered.wait(10)
        stop_actions(store, first.run_id, "stop now")
    finally:
        gate.release()
    progress = finished(thread, outcome)

    # The stop refuses the queue at once: there is no drain deadline to wait for.
    assert sleeps == []
    assert [(entry.action_id, entry.refusal is None) for entry in progress] == [
        (first.action_id, True), (second.action_id, False)]
    assert connector.made("submit") == 1


# ---------------------------------------------------------------------------
# Revoking never undoes an effect
# ---------------------------------------------------------------------------


def test_revoking_an_observed_action_changes_nothing_and_inspection_shows_it_came_after_the_effect(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    observed(store, connector, host, clock, action)
    before, objects = item(store, action), service.snapshot()["objects"]
    clock.advance(1)

    assert revoke(store, action.action_id, "too late") == (approval.approval_id,)

    after = item(store, action)
    assert (after["state"], after["attempts"], after["records"], balances(after)) == (
        "observed", before["attempts"], before["records"], balances(before))
    (shown,) = after["approvals"]
    assert (shown["revoked_at_ns"], shown["revocation"]) == (clock.now_ns, "after_effect")
    assert {**shown, "revoked_at_ns": None} == {**before["approvals"][0], "revocation": "after_effect"}
    assert "revocation" not in before["approvals"][0]
    assert service.snapshot()["objects"] == objects and connector.made("submit") == 1
    assert store.audit(action.run_id)["ok"]


def test_inspection_tells_a_revocation_before_dispatch_from_one_after_an_unknown_outcome(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False, dedupe=False)
    connector = Recording(service)
    early, late = queued(store, tmp_path, replies, host, clock, connector, slots=("early", "late"))
    service.commit_then_drop_ack(late.action_key)
    revoke(store, early.action_id, "never sent")
    assert runner(store, connector, host, clock).dispatch(late.run_id, late.action_id).state == "outcome_unknown"

    revoke(store, late.action_id, "sent, outcome unknown")

    assert revocations(store, early) == ["before_dispatch"]
    assert revocations(store, late) == ["after_dispatch"]
    # The unknown write keeps its exposure.
    assert balances(item(store, late)) == (0, 0, WRITE)


# ---------------------------------------------------------------------------
# A security stop
# ---------------------------------------------------------------------------


def test_a_security_stop_refuses_dispatch_pending_before_any_claim(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    first, second = queued(store, tmp_path, replies, host, clock, connector, slots=("first", "second"))

    store.security_stop(first.run_id, "approver_compromised")

    progress = runner(store, connector, host, clock).dispatch_pending(first.run_id)
    assert [entry[:3] for entry in summary(progress)] == [
        (first.action_id, "not_dispatched", None), (second.action_id, "not_dispatched", None)]
    assert all("a security stop holds the run" in entry.refusal for entry in progress)
    assert rows(store.path, "workflow_action_attempts") == [] and connector.calls == []
    # A security stop is not an action stop: the approvals stand, and only the run's stop refuses them.
    assert all(row["revoked_at_ns"] is None for row in rows(store.path, "workflow_action_approvals"))
    assert store.audit(first.run_id)["ok"]
