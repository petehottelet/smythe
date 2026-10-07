"""External actions under competing workers, late events and every declared service guarantee.

Actions bind a durable run's journal record, which an ordinary run never has.
Two workers that race to dispatch one action, as threads with their own store
connections or from another process, make one dispatch claim and one effect
on the service: the run's lease fences the other, which gets
``WorkflowLeaseError``. Receipts and notices that arrive late, in any order,
never move an action back. Across deduplication, lookup, visibility and
key-window declarations, no automatic step makes a second effect: another
submission follows only a lookup that proved the first was never made, or is
a same-key retry that the service still deduplicates.
"""

from __future__ import annotations

import itertools
import json
import threading
import time

import pytest

from fakes import Gate, ManualClock, kill_and_reap, launch_child, wait_for_barrier, write_barrier
from fixture_connector import FixtureConnector, HttpFixtureConnector, stored_value
from fixture_service import FixtureService, Target, serve_http
from smythe import (
    FileCheckpointStore, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, workflow_actions,
)
from smythe.actions import (
    ActionProgress, ActionRunner, ActionStateError, ActionTarget, ConnectorAmbiguous, ConnectorReceipt,
    ContentRef, approve, prepare_action, record_notice,
)
from smythe.workflow_actions import record_receipt
from smythe.workflow_store import WorkflowLeaseError, WorkflowNotFoundError
from test_action_approvals import (
    HOUR_NS, PROFILE, Host, approved, completed, delivered, intent, prepare, sign, swarm,
)
from test_action_journal import SETTINGS, item, rows
from test_action_runner import relay, runner
from test_workflow_acceptance import Contains, replies, single, strict  # noqa: F401

NS = 1_000_000_000


def claims(path, action) -> list[dict]:
    """The action's attempts whose dispatch was claimed, oldest first."""
    return [row for row in rows(path, "workflow_action_attempts")
            if row["action_id"] == action.action_id and row["dispatch_token_sha"] is not None]


def records(path, action, kind) -> list[dict]:
    return [json.loads(row["payload_json"]) for row in rows(path, "workflow_action_records")
            if row["action_id"] == action.action_id and row["kind"] == kind]


def submissions(service, action) -> list[dict]:
    """What the service did with each delivery of the action's key: committed, deduplicated or refused."""
    return [entry for entry in service.snapshot()["deliveries"] if entry["request_key"] == action.action_key]


# ---------------------------------------------------------------------------
# Where actions run
# ---------------------------------------------------------------------------


def test_an_ordinary_run_has_no_journal_record_an_action_could_bind(tmp_path, replies):  # noqa: F811
    replies[0]["Write"] = ["the report"]
    folder = tmp_path / "state"
    checkpoints = FileCheckpointStore(folder)
    ordinary = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                     checkpoint_store=checkpoints, acceptance=strict(Contains("report")),
                     artifact_dir=None).execute(single())
    assert ordinary.accepted and ordinary.output == "the report"

    # What the run leaves is its checkpoint, holding the accepted output, and nothing else.
    run_id = ordinary.execution_id
    saved = checkpoints.load(run_id)
    assert (saved["status"], saved["output"]) == ("completed", "the report")
    assert [path.name for path in folder.iterdir()] == [f"{run_id}.json"]

    # A journal created beside it has no record of the run: prepare_action refuses its ID, and a
    # durable Swarm on that journal cannot take the run over by resuming it.
    journal = folder / "runs.db"
    with SQLiteWorkflowStore(journal) as store:
        with pytest.raises(WorkflowNotFoundError, match=run_id):
            prepare_action(store, run_id, slot="publish", profile=PROFILE,
                           intent=intent(ContentRef.for_bytes(b"the report", "text/plain")))
        with pytest.raises(WorkflowNotFoundError, match=run_id):
            swarm(store, tmp_path, renderer=None).resume(run_id)
        assert store.list_runs() == []
    assert rows(journal, "workflow_runs") == rows(journal, "workflow_actions") == []
    assert checkpoints.load(run_id) == saved


# ---------------------------------------------------------------------------
# Competing workers
# ---------------------------------------------------------------------------


def _wait_for(condition, what: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def test_two_threads_racing_to_dispatch_make_one_claim_and_one_effect(tmp_path, replies):  # noqa: F811
    path, clock, host = tmp_path / "runs.db", ManualClock(time.time_ns()), Host()
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        action = prepare(store, completed(store, tmp_path, replies, renderer=None))
        approved(store, host, action, clock)
    service, gate = FixtureService(clock_ns=clock), Gate()
    # The winner's write commits and its answer waits here until the loser has lost.
    service.hold(action.action_key, gate)
    start, outcomes = threading.Barrier(2), {}

    def worker(own: SQLiteWorkflowStore, name: str) -> None:
        racer = ActionRunner(own, FixtureConnector(service, **SETTINGS), host, owner_id=name)
        start.wait(60)
        try:
            outcomes[name] = racer.dispatch(action.run_id, action.action_id)
        except Exception as error:  # the test checks which error
            outcomes[name] = error

    # Each worker has its own connection to the journal.
    with SQLiteWorkflowStore(path, clock_ns=clock) as first, SQLiteWorkflowStore(path, clock_ns=clock) as second:
        threads = [threading.Thread(target=worker, args=(own, name))
                   for own, name in ((first, "worker-a"), (second, "worker-b"))]
        for thread in threads:
            thread.start()
        try:
            assert gate.entered.wait(60), outcomes
            # The winner holds its lease while its answer waits, so the first worker to return lost.
            _wait_for(lambda: outcomes, "the losing worker")
        finally:
            gate.release()
            for thread in threads:
                thread.join(60)

    (lost,) = [value for value in outcomes.values() if isinstance(value, Exception)]
    (won,) = [value for value in outcomes.values() if isinstance(value, ActionProgress)]
    assert isinstance(lost, WorkflowLeaseError) and "active lease" in str(lost)
    assert (won.state, won.attempt["number"], won.attempt["state"]) == ("dispatched", 1, "acknowledged")
    (claim,) = claims(path, action)
    assert claim["number"] == 1
    assert len(rows(path, "workflow_action_attempts")) == 1
    counts = service.snapshot()["requests"][action.action_key]
    assert (counts["received"], counts["committed"], counts["deduplicated"]) == (1, 1, 0)
    with SQLiteWorkflowStore(path, clock_ns=clock) as store:
        assert store.audit(action.run_id)["ok"]


# Child code: open the journal, say so, wait for the start signal and dispatch the action once.
CHILD = r"""
from pathlib import Path
import sys
import time

from fakes import write_barrier
from fixture_connector import HmacApprovalHost, HttpFixtureConnector
from fixture_service import Endpoint
from smythe import SQLiteWorkflowStore
from smythe.actions import ActionRunner, load_action
from smythe.workflow_store import WorkflowLeaseError

path, port, action_id, key = sys.argv[1], int(sys.argv[2]), sys.argv[3], bytes.fromhex(sys.argv[4])
ready, go = Path(sys.argv[5]), Path(sys.argv[6])
with SQLiteWorkflowStore(path) as store:
    action = load_action(store, action_id)
    racer = ActionRunner(store, HttpFixtureConnector(Endpoint("127.0.0.1", port), action.profile),
                         HmacApprovalHost(key=key), owner_id="worker-in-another-process")
    write_barrier(ready, {"ready": True})
    deadline = time.monotonic() + 60
    while not go.exists():
        if time.monotonic() > deadline:
            sys.exit("No start signal")
        time.sleep(0.002)
    try:
        progress = racer.dispatch(action.run_id, action_id)
    except WorkflowLeaseError as error:
        print(f"WorkflowLeaseError: {error}")
        sys.exit(3)
    print(f"{progress.state} {progress.attempt['state']}")
"""


def test_a_worker_in_another_process_and_a_local_worker_race_and_one_claims(tmp_path, replies):  # noqa: F811
    path, host = tmp_path / "runs.db", Host()
    with SQLiteWorkflowStore(path, clock_ns=ManualClock(time.time_ns())) as store:
        action = prepare(store, completed(store, tmp_path, replies, renderer=None))
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=time.time_ns() + HOUR_NS)
    service, gate = FixtureService(clock_ns=ManualClock()), Gate()
    service.hold(action.action_key, gate)
    ready, go, start, local = tmp_path / "ready.json", tmp_path / "go", threading.Event(), {}

    with serve_http(service) as endpoint, SQLiteWorkflowStore(path) as own:
        racer = ActionRunner(own, HttpFixtureConnector(endpoint, action.profile), host, owner_id="local-worker")

        def dispatch() -> None:
            start.wait(60)
            try:
                local["outcome"] = racer.dispatch(action.run_id, action.action_id)
            except Exception as error:  # the test checks which error
                local["outcome"] = error

        thread = threading.Thread(target=dispatch)
        process = launch_child(CHILD, path, endpoint.port, action.action_id, host.key.hex(), ready, go)
        try:
            wait_for_barrier(process, ready, timeout=60)
            thread.start()
            write_barrier(go, {})
            start.set()
            assert gate.entered.wait(60)
            # The winner waits at the gate holding its lease, so whoever finishes first lost.
            _wait_for(lambda: process.poll() is not None or local, "the losing worker")
            gate.release()
            thread.join(120)
            # A winning child still records its receipt and releases the lease; let it finish.
            process.wait(120)
            reaped = kill_and_reap(process)
        finally:
            gate.release()
            kill_and_reap(process)

    output = reaped.stdout.decode()
    assert not reaped.killed, reaped
    if reaped.returncode == 3:  # The other process lost the race.
        assert "WorkflowLeaseError" in output and "active lease" in output
        assert isinstance(local["outcome"], ActionProgress)
        assert (local["outcome"].state, local["outcome"].attempt["state"]) == ("dispatched", "acknowledged")
    else:
        assert (reaped.returncode, output.split()) == (0, ["dispatched", "acknowledged"]), reaped
        assert isinstance(local["outcome"], WorkflowLeaseError) and "active lease" in str(local["outcome"])
    (claim,) = claims(path, action)
    assert claim["number"] == 1 and len(rows(path, "workflow_action_attempts")) == 1
    counts = service.snapshot()["requests"][action.action_key]
    assert (counts["received"], counts["committed"], counts["deduplicated"]) == (1, 1, 0)
    with SQLiteWorkflowStore(path) as store:
        assert store.audit(action.run_id)["ok"]


# ---------------------------------------------------------------------------
# Receipts and notices out of order
# ---------------------------------------------------------------------------


class InFlight(FixtureConnector):
    """Puts each submission in flight with ``service.send``: its answer arrives only when the test delivers it.

    The runner gives up waiting, so each attempt's outcome is unknown.
    """

    def __init__(self, service: FixtureService, **settings) -> None:
        super().__init__(service, **settings)
        self.messages = []

    def submit(self, key, intent, payload, timeout_s):
        target = Target(intent.target.tenant, intent.target.collection)
        self.messages.append(self.service.send(key, stored_value(intent, payload),
                                               object_id=intent.target.object_id, target=target))
        raise ConnectorAmbiguous("No answer arrived within the timeout")


def answer(message) -> ConnectorReceipt:
    """The receipt a delivered message answered with, as a connector reports it."""
    receipt = message.result()
    return ConnectorReceipt(receipt.request_key, ActionTarget(
        receipt.target.tenant, receipt.target.collection, receipt.object_id),
        object_version=receipt.object_version, deduplicated=receipt.deduplicated)


def standing(store, action) -> tuple:
    """The action's state, its attempts and its balances, which only a result may change."""
    current = item(store, action)
    return (current["state"], current["attempts"], current["confirmed_nanousd"], current["reserved_nanousd"],
            current["unknown_nanousd"])


# How far an action has come: a late event may leave it where it is, never move it back.
PROGRESS = {"not_dispatched": 0, "dispatched": 1, "outcome_unknown": 1, "observed": 2, "reconciled": 2}
EVENTS = ("first answer", "retry answer", "notice", "forged notice", "settle")
# Every event at every position, and each pair of events in both orders.
SEQUENCES = [EVENTS[shift:] + EVENTS[:shift] for shift in range(5)] + [
    EVENTS[::-1][shift:] + EVENTS[::-1][:shift] for shift in range(5)]


@pytest.mark.parametrize("delivery_order", [[0, 1], [1, 0]], ids=["first-delivered-first", "retry-delivered-first"])
def test_receipts_and_notices_in_any_order_never_move_an_action_back(
        tmp_path, replies, monkeypatch, delivery_order):  # noqa: F811
    clock, host = ManualClock(time.time_ns()), Host()
    permits = []
    for name in ("claim_dispatch", "claim_retry"):
        claim = getattr(workflow_actions, name)
        monkeypatch.setattr(workflow_actions, name,
                            lambda *args, claim=claim: permits.append(claim(*args)) or permits[-1])
    # The service deduplicates without expiry and has no lookup by key, so reconciliation retries.
    settings = {**SETTINGS, "max_attempts": 3}
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        run_id = completed(store, tmp_path, replies, renderer=None)
        for number, sequence in enumerate(SEQUENCES):
            service = FixtureService(clock_ns=clock, lookup_by_key=False, send_notices=True)
            sender = InFlight(service, **settings)
            action = prepare_action(store, run_id, slot=f"case-{number}", intent=intent(delivered(store, run_id)),
                                    profile=sender.profile)
            approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                    expires_at_ns=clock.now_ns + HOUR_NS)
            del permits[:]
            sending = runner(store, sender, host, clock)
            assert sending.dispatch(run_id, action.action_id).state == "outcome_unknown"
            retried = sending.reconcile(run_id, action.action_id)
            assert (retried.state, retried.attempt["number"]) == ("outcome_unknown", 2)
            assert len(sender.messages) == len(permits) == 2 and service.snapshot()["deliveries"] == []

            # Both copies arrive in the chosen order; the service commits one and deduplicates the other.
            service.deliver(delivery_order)
            assert [entry["outcome"] for entry in submissions(service, action)] == ["committed", "deduplicated"]
            (genuine,) = service.notices()
            forged = service.forge_notice({"request_key": action.action_key, "object_id": "obj_9",
                                           "object_version": 9, "target": {"tenant": "acme", "collection": "posts"}})
            settled = runner(store, FixtureConnector(service, **settings), host, clock)
            events = {
                "first answer": lambda: record_receipt(store, permits[0], answer(sender.messages[0])),
                "retry answer": lambda: record_receipt(store, permits[1], answer(sender.messages[1])),
                "notice": lambda: record_notice(store, relay(genuine)),
                "forged notice": lambda: record_notice(store, relay(forged)),
                "settle": lambda: settled.reconcile(run_id, action.action_id),
            }

            for event in sequence:
                before = standing(store, action)
                if event == "settle":
                    assert events[event]().state == "reconciled", (sequence, event)
                elif event.endswith("answer"):
                    # A late receipt is kept as a record, whatever the attempt became meanwhile.
                    with pytest.raises((WorkflowLeaseError, ActionStateError), match="kept as a late receipt"):
                        events[event]()
                else:
                    # The action is never dispatched while an event arrives, so no notice prompts a read.
                    assert events[event]() is None, (sequence, event)
                after = standing(store, action)
                assert PROGRESS[after[0]] >= PROGRESS[before[0]], (sequence, event, before[0], after[0])
                if event != "settle":
                    assert after == before, (sequence, event)

            assert standing(store, action)[0] == "reconciled"
            # Each answer is kept as it arrived, in the order it arrived.
            relayed = sorted(("first answer", "retry answer"), key=sequence.index)
            assert [payload["attempt_id"] for payload in records(store.path, action, "late_receipt")] == [
                permits[("first answer", "retry answer").index(name)].attempt_id for name in relayed]
            assert len(records(store.path, action, "notice")) == 2
            # The settling retry was answered from the one commit.
            assert [entry["outcome"] for entry in submissions(service, action)] == [
                "committed", "deduplicated", "deduplicated"]
            assert service.snapshot()["totals"]["committed"] == 1
        assert store.audit(run_id)["ok"]


# ---------------------------------------------------------------------------
# No second effect under any declared guarantee
# ---------------------------------------------------------------------------

WINDOW = 10
SWEEP_SETTINGS = {"write_ceiling_nanousd": 2_000_000, "read_ceiling_nanousd": 10_000, "max_attempts": 3,
                  "readback_reads": 2, "timeout_s": 1}
GUARANTEES = {
    "deduplicates-with-lookup": {"dedupe": True, "lookup_by_key": True},
    "deduplicates-in-a-window-with-lookup": {"dedupe": True, "key_ttl_s": WINDOW, "lookup_by_key": True},
    "deduplicates-only": {"dedupe": True, "lookup_by_key": False},
    "deduplicates-in-a-window-only": {"dedupe": True, "key_ttl_s": WINDOW, "lookup_by_key": False},
    "lookup-only": {"dedupe": False, "lookup_by_key": True},
    "neither": {"dedupe": False, "lookup_by_key": False},
    "deduplicates-with-lookup-without-reads": {"dedupe": True, "lookup_by_key": True, "lookup_by_object": False},
}
VISIBILITY = (0, 3)  # seconds a commit stays invisible to lookups and reads
FAULTS = ("answer-lost", "request-lost")
PACES = (0, 5, 20)  # seconds the clock moves before each step: before the bound, inside the window, past it
STEPS = {"dispatch": ActionRunner.dispatch, "readback": ActionRunner.readback, "reconcile": ActionRunner.reconcile}


class Lossy(FixtureConnector):
    """Loses the first submission: its request before it reaches the service, or the service's answer."""

    def __init__(self, service: FixtureService, fault: str, **settings) -> None:
        super().__init__(service, **settings)
        self.fault, self.submitted = fault, 0

    def submit(self, key, intent, payload, timeout_s):
        self.submitted += 1
        if self.submitted == 1:
            if self.fault == "request-lost":
                raise ConnectorAmbiguous("The connection failed; the request may have been sent")
            self.service.commit_then_drop_ack(key)
        return super().submit(key, intent, payload, timeout_s)


@pytest.mark.parametrize("guarantee", GUARANTEES)
def test_no_second_effect_without_proven_absence_or_live_deduplication(tmp_path, replies, guarantee):  # noqa: F811
    clock, host = ManualClock(time.time_ns()), Host()
    with SQLiteWorkflowStore(tmp_path / "runs.db", clock_ns=clock) as store:
        run_id = completed(store, tmp_path, replies, renderer=None)
        for number, (visibility, fault, pace) in enumerate(itertools.product(VISIBILITY, FAULTS, PACES)):
            case = (guarantee, visibility, fault, pace)
            service = FixtureService(clock_ns=clock, visibility_delay_s=visibility, **GUARANTEES[guarantee])
            connector = Lossy(service, fault, **SWEEP_SETTINGS)
            profile = connector.profile
            action = prepare_action(store, run_id, slot=f"case-{number}", intent=intent(delivered(store, run_id)),
                                    profile=profile)
            approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                    expires_at_ns=clock.now_ns + HOUR_NS)
            worker = runner(store, connector, host, clock)

            progress = worker.dispatch(run_id, action.action_id)
            assert progress.state == "outcome_unknown", case
            # Every automatic step the runner names, until it names none or an operator's review.
            for _ in range(8):
                step = STEPS.get(progress.next_action)
                if step is None:
                    break
                clock.advance(pace)
                progress = step(worker, run_id, action.action_id)
            assert progress.next_action in (None, "operator_review"), case

            delivered_to = submissions(service, action)
            committed = [entry for entry in delivered_to if entry["outcome"] == "committed"]
            # The service never holds a second effect.
            assert len(committed) <= 1, case
            absence = [finding for finding in records(store.path, action, "finding")
                       if finding["finding"] == "not_executed"]
            # A lookup proves absence only when nothing was committed.
            assert not absence or fault == "request-lost", case
            # Each later submission followed proven absence, or was a same-key retry the service
            # still deduplicated inside the declared window.
            attempts = claims(store.path, action)
            assert connector.submitted == len(attempts), case
            first = attempts[0]["dispatched"]
            for attempt in attempts[1:]:
                proven = any(finding["attempts"] == attempt["number"] - 1 for finding in absence)
                window = profile.dedupe_window_s
                live = profile.deduplicates and (
                    window is None or attempt["dispatched"] + profile.timeout_s * NS < first + window * NS)
                assert proven or live, (case, attempt["number"])
            for index, entry in enumerate(delivered_to):
                if index and entry["outcome"] != "deduplicated":
                    # Only the first effect can follow a commit that was never made.
                    assert entry["outcome"] == "committed" and absence, case
                    assert all(earlier["outcome"] != "committed" for earlier in delivered_to[:index]), case
            # A settled action holds what the service shows.
            if progress.state in ("observed", "reconciled"):
                assert len(committed) == 1, case
            if fault == "answer-lost":
                assert len(committed) == 1, case
        assert store.audit(run_id)["ok"]
