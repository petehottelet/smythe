"""Reconciling an external write whose outcome is unknown, only as the connector's declarations allow.

A write whose acknowledgement was lost may or may not have taken effect.
``ActionRunner.reconcile`` looks the action's key up where the service offers
that, and a read of the receipt it finds settles the action. An empty lookup
shows that nothing was written only after a declared visibility bound, while
the key is still inside its window. A deduplicating service without lookups
gets one same-key retry while its window is open. Anything else stays
explicitly unresolved, its exposure kept, and nothing is submitted again.
"""

from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from fakes import Gate, ManualClock
from fixture_service import FixtureService
from smythe import SQLiteWorkflowStore, workflow_actions
from smythe.actions import (
    ActionEvidence,
    ActionNotAuthorized,
    ActionStateError,
    ActionTarget,
    ConnectorAmbiguous,
    ConnectorReceipt,
    check_authority,
    decision_sha256,
    inspect_actions,
    resolve,
    revoke,
)
from smythe.workflow_store import WorkflowLeaseError, WorkflowValidationError
from test_action_approvals import FAR, Host
from test_action_journal import SETTINGS, WRITE, balances, execute, item
from test_action_runner import Recording, ready, records, report, runner
from test_workflow_acceptance import replies  # noqa: F401

READ = SETTINGS["read_ceiling_nanousd"]
TIMEOUT = 30  # the profile's timeout_s: the margin on a key's window and a visibility bound


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


class Reconciling(Recording):
    """A recording connector that also logs lookups.

    ``lost`` submissions time out before they reach the service, so nothing
    is written although the outcome is unknown. ``bound`` replaces the
    visibility bound the service's configuration declares; ``None`` declares
    none, as an eventually consistent service would.
    """

    def __init__(self, service: FixtureService, *, bound: object = ..., **settings) -> None:
        super().__init__(service, **settings)
        self.lost = 0
        if bound is not ...:
            self.profile = replace(self.profile, visibility_bound_s=bound)

    def submit(self, key, intent, payload, timeout_s):
        if self.lost:
            self.lost -= 1
            self.calls.append(("submit", key, payload))
            raise ConnectorAmbiguous("The request timed out before any answer")
        return super().submit(key, intent, payload, timeout_s)

    def lookup(self, key, target, timeout_s):
        self.calls.append(("lookup", key))
        return super().lookup(key, target, timeout_s)


def unknown(store, tmp_path, replies, host, clock, connector, *, committed=True):  # noqa: F811
    """An approved action whose one dispatch left its outcome unknown, written or not."""
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    if committed:
        connector.service.commit_then_drop_ack(action.action_key)
    else:
        connector.lost = 1
    progress = runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)
    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "outcome_unknown", "outcome_unknown", "reconcile")
    assert balances(item(store, action)) == (0, 0, WRITE)
    return action, approval


def counts(service, action) -> tuple:
    """What the service did with the action's key: received, committed, deduplicated."""
    found = service.snapshot()["requests"].get(action.action_key)
    return (0, 0, 0) if found is None else (found["received"], found["committed"], found["deduplicated"])


def intact(store, action, before) -> dict:
    """The action now, after checking that reconciliation only appended to its history.

    Earlier attempts and records are unchanged; confirmed cost grew by what new
    records and attempts confirmed; nothing is reserved and no unknown
    exposure was released. The journal's audit passes.
    """
    after = item(store, action)
    assert after["attempts"][:len(before["attempts"])] == before["attempts"]
    assert after["records"][:len(before["records"])] == before["records"]
    added = (sum(record["cost_nanousd"] or 0 for record in after["records"][len(before["records"]):])
             + sum(attempt["cost_nanousd"] or 0 for attempt in after["attempts"][len(before["attempts"]):]))
    assert after["confirmed_nanousd"] == before["confirmed_nanousd"] + added
    assert after["reserved_nanousd"] == 0 and after["unknown_nanousd"] >= before["unknown_nanousd"]
    assert after["confirmed_nanousd"] + after["unknown_nanousd"] <= after["approvals"][0]["ceiling_nanousd"]
    assert store.audit(action.run_id)["ok"]
    return after


def findings(store, action) -> list[dict]:
    return [record["payload"] for record in records(store, action, "finding")]


# ---------------------------------------------------------------------------
# Lookup by key
# ---------------------------------------------------------------------------


LOOKUP = {
    "deduplicating": ({}, {}),
    "without-deduplication": ({"dedupe": False}, {}),
    "inside-a-key-window": ({"key_ttl_s": 3600}, {}),
    # The commit becomes visible between the first lookup and the second.
    "bounded-visibility": ({"visibility_delay_s": 5}, {"readback_interval_s": 5}),
}


@pytest.mark.parametrize("config", LOOKUP.values(), ids=LOOKUP.keys())
def test_a_lost_acknowledgement_is_reconciled_from_a_lookup_and_a_read(
        store, tmp_path, replies, host, clock, config):  # noqa: F811
    service_config, settings = config
    service = FixtureService(clock_ns=clock, **service_config)
    connector = Reconciling(service, **settings)
    assert connector.profile.lookup_by_key and connector.profile.lookup_by_object
    action, approval = unknown(store, tmp_path, replies, host, clock, connector)
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("reconciled", None)
    assert progress.report.passed and progress.attempt == before["attempts"][0]
    # One write reached the service, and nothing was sent again.
    assert counts(service, action) == (1, 1, 0) and len(service.snapshot()["deliveries"]) == 1
    assert connector.made("submit") == 1 and connector.made("read") == 1
    lookups = records(store, action, "lookup")
    assert len(lookups) == connector.made("lookup") == (2 if service.visibility_delay_s else 1)
    (object_id,) = service.snapshot()["objects"]
    found = lookups[-1]["payload"]
    assert (found["attempt_id"], found["lookup"], found["error"]) == (
        progress.attempt["attempt_id"], len(lookups), None)
    assert found["receipt"]["target"] == {"tenant": "acme", "collection": "posts", "object_id": object_id}
    assert all(record["payload"]["receipt"] is None for record in lookups[:-1])
    (read,) = records(store, action, "readback")
    assert report(read) == progress.report and read["payload"]["attempt_id"] == progress.attempt["attempt_id"]
    assert findings(store, action) == []
    # Each lookup and read is charged; the write keeps its exposure, since no receipt reports its cost.
    after = intact(store, action, before)
    assert balances(after) == ((len(lookups) + 1) * READ, 0, WRITE)
    assert after["attempts"][0]["state"] == "outcome_unknown"
    with pytest.raises(ActionNotAuthorized, match="it is reconciled"):
        check_authority(store, action.action_id, host)
    events = [(event["type"], event["data"].get("state")) for event in store.inspect_run(action.run_id)["events"]
              if event["type"] in ("action_lookup", "action_readback")]
    assert events == [("action_lookup", None)] * len(lookups) + [("action_readback", "reconciled")]
    # A reconciled action is settled: reconciling again calls nothing.
    calls = list(connector.calls)
    assert runner(store, connector, host, clock).reconcile(action.run_id, action.action_id) == replace(
        progress, report=None)
    assert connector.calls == calls


def test_a_lookup_receipt_that_reports_a_cost_confirms_the_attempts_exposure(
        store, tmp_path, replies, host, clock):  # noqa: F811
    class Costed(Reconciling):
        def lookup(self, key, target, timeout_s):
            found = super().lookup(key, target, timeout_s)
            return found and replace(found, cost_nanousd=1_500_000)

    service = FixtureService(clock_ns=clock)
    connector = Costed(service)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)

    assert runner(store, connector, host, clock).reconcile(action.run_id, action.action_id).state == "reconciled"

    after = item(store, action)
    (attempt,) = after["attempts"]
    assert (attempt["state"], attempt["cost_nanousd"], attempt["unknown_nanousd"]) == (
        "outcome_unknown", 1_500_000, 0)
    assert balances(after) == (1_500_000 + 2 * READ, 0, 0)
    assert store.audit(action.run_id)["ok"]


@pytest.mark.parametrize("answer", [
    lambda receipt: replace(receipt, target=ActionTarget("globex", "posts", receipt.target.object_id)),
    lambda receipt: replace(receipt, key="another-key"),
    lambda receipt: receipt.to_dict(),
], ids=["another-tenant", "another-key", "not-a-receipt"])
def test_a_lookup_that_answers_for_something_else_proves_nothing(
        store, tmp_path, replies, host, clock, answer):  # noqa: F811
    class Misanswering(Reconciling):
        def lookup(self, key, target, timeout_s):
            self.calls.append(("lookup", key))
            return answer(ConnectorReceipt(key, ActionTarget("acme", "posts", "obj_9"), object_version=1))

    service = FixtureService(clock_ns=clock)
    connector = Misanswering(service)
    # The write never reached the service, and the bound has passed: an empty answer would prove that.
    action, _ = unknown(store, tmp_path, replies, host, clock, connector, committed=False)
    clock.advance(TIMEOUT)
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    lookups = records(store, action, "lookup")
    assert [(record["payload"]["receipt"], record["payload"]["error"]) for record in lookups] == [
        (None, "The lookup answered with something other than the action's receipt")] * 2
    (finding,) = findings(store, action)
    assert (finding["finding"], finding["reason"]) == ("unresolved", "no lookup answered")
    assert connector.made("submit") == 1 and connector.made("read") == 0
    intact(store, action, before)


# ---------------------------------------------------------------------------
# Empty lookups
# ---------------------------------------------------------------------------


EMPTY = {
    # An eventually consistent service that declares no bound: no lookup can prove absence.
    "unbounded-and-written": (None, True, "the connector declares no visibility bound"),
    "unbounded-and-unwritten": (None, False, "the connector declares no visibility bound"),
    # A declared bound that has not passed yet.
    "inside-the-bound": (3600, True, "the visibility bound had not passed since the dispatch"),
}


@pytest.mark.parametrize("config", EMPTY.values(), ids=EMPTY.keys())
def test_empty_lookups_before_a_visibility_bound_never_resubmit(
        store, tmp_path, replies, host, clock, config):  # noqa: F811
    bound, committed, reason = config
    service = FixtureService(clock_ns=clock, visibility_delay_s=3600)
    connector = Reconciling(service, bound=bound, readback_interval_s=10)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector, committed=committed)
    before = item(store, action)
    started = clock.monotonic()

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.attempt["state"], progress.next_action, progress.report) == (
        "outcome_unknown", "outcome_unknown", "operator_review", None)
    # Two lookups, one interval apart, both empty; nothing is submitted or read.
    assert (connector.made("lookup"), connector.made("submit"), connector.made("read")) == (2, 1, 0)
    assert clock.monotonic() - started == pytest.approx(10)
    assert counts(service, action) == ((1, 1, 0) if committed else (0, 0, 0))
    lookups = records(store, action, "lookup")
    assert [record["payload"]["receipt"] for record in lookups] == [None, None]
    (finding,) = findings(store, action)
    assert finding == {"attempt_id": progress.attempt["attempt_id"], "finding": "unresolved",
                       "reason": f"{reason}, so an empty lookup proves nothing",
                       "evidence": [record["record_id"] for record in lookups], "attempts": 1}
    after = intact(store, action, before)
    assert balances(after) == (2 * READ, 0, WRITE)
    # The approval is unconsumed, yet nothing dispatches an action whose write may have taken effect.
    assert check_authority(store, action.action_id, host) == approval
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)

    # The lookups are bounded across calls, even once the commit is visible.
    clock.advance(7200)
    again = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)
    assert (again.state, again.next_action) == ("outcome_unknown", "operator_review")
    assert (connector.made("lookup"), connector.made("submit")) == (2, 1)
    assert len(findings(store, action)) == 1  # The same finding is not recorded twice.
    assert runner(store, connector, host, clock).readback(action.run_id, action.action_id).next_action == (
        "operator_review")
    assert counts(service, action) == ((1, 1, 0) if committed else (0, 0, 0))


def test_lookups_poll_until_the_visibility_bound_has_passed(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=5)
    connector = Reconciling(service, readback_interval_s=1)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector, committed=False)
    # The bound counts from the dispatch claim plus the connector's timeout: one second short.
    clock.advance(5 + TIMEOUT - 1)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    # The first empty lookup came a second early and proved nothing; the second, one interval later, did.
    assert (progress.state, progress.next_action) == ("not_dispatched", "dispatch")
    lookups = records(store, action, "lookup")
    assert [record["payload"]["lookup"] for record in lookups] == [1, 2]
    (finding,) = findings(store, action)
    assert (finding["finding"], finding["evidence"]) == (
        "not_executed", [record["record_id"] for record in lookups])
    assert connector.made("submit") == 1 and counts(service, action) == (0, 0, 0)


@pytest.mark.parametrize("delay", [0, 5], ids=["immediate-visibility", "five-second-bound"])
@pytest.mark.parametrize("committed", [False, True], ids=["unwritten", "written"])
def test_past_the_visibility_bound_a_lookup_settles_the_outcome(
        store, tmp_path, replies, host, clock, delay, committed):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=delay)
    connector = Reconciling(service)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector, committed=committed)
    before = item(store, action)
    clock.advance(delay + TIMEOUT)
    worker = runner(store, connector, host, clock)

    progress = worker.reconcile(action.run_id, action.action_id)

    assert connector.made("lookup") == 1 and connector.made("submit") == 1
    if committed:
        # The commit is visible by now, so the lookup finds it: reconciled, still one write.
        assert (progress.state, progress.next_action) == ("reconciled", None)
        assert counts(service, action) == (1, 1, 0)
        return
    (lookup,) = records(store, action, "lookup")
    (finding,) = findings(store, action)
    assert finding == {"attempt_id": before["attempts"][0]["attempt_id"], "finding": "not_executed",
                       "reason": "an empty lookup after the visibility bound, inside the key's window, "
                                 "shows that the write was not made",
                       "evidence": [lookup["record_id"]], "attempts": 1}
    assert (progress.state, progress.attempt["state"], progress.next_action, progress.report) == (
        "not_dispatched", "outcome_unknown", "dispatch", None)
    # The attempt keeps its exposure: no receipt reported a cost.
    after = intact(store, action, before)
    assert balances(after) == (READ, 0, WRITE)
    events = [event["data"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_finding"]
    assert events == [{"action_id": action.action_id, "record_id": records(store, action, "finding")[0][
        "record_id"], "finding": "not_executed", "state": "not_dispatched"}]
    # The same, unconsumed approval authorizes the next attempt, which makes exactly one effect.
    assert check_authority(store, action.action_id, host) == approval
    dispatched = worker.dispatch(action.run_id, action.action_id)
    assert (dispatched.attempt["number"], dispatched.attempt["state"]) == (2, "acknowledged")
    clock.advance(delay)
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    assert counts(service, action) == (1, 1, 0)
    assert [call[1] for call in connector.calls if call[0] == "submit"] == [action.action_key] * 2
    final = intact(store, action, after)
    assert [attempt["state"] for attempt in final["attempts"]] == ["outcome_unknown", "acknowledged"]
    assert balances(final) == (2 * READ, 0, 2 * WRITE)


def test_an_expired_key_window_never_counts_an_empty_lookup_as_absence(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, key_ttl_s=60)
    connector = Reconciling(service)
    assert connector.profile.dedupe_window_s == 60 and connector.profile.lookup_by_key
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)
    # Past the visibility bound, but the service has forgotten the key it committed.
    clock.advance(61)
    assert service.lookup_request(action.action_key, "acme") is None
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    assert findings(store, action)[0]["reason"] == (
        "the key's window may have closed, so an empty lookup proves nothing")
    assert (connector.made("lookup"), connector.made("submit")) == (2, 1)
    assert counts(service, action) == (1, 1, 0)
    intact(store, action, before)


# ---------------------------------------------------------------------------
# Deduplication without lookup
# ---------------------------------------------------------------------------


RETRY = {"keys-without-expiry": {}, "inside-the-key-window": {"key_ttl_s": 3600}}


@pytest.mark.parametrize("config", RETRY.values(), ids=RETRY.keys())
def test_a_deduplicating_service_gets_one_same_key_retry(
        store, tmp_path, replies, host, clock, config):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False, **config)
    connector = Reconciling(service)
    assert connector.profile.deduplicates and not connector.profile.lookup_by_key
    action, approval = unknown(store, tmp_path, replies, host, clock, connector)
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("reconciled", None) and progress.report.passed
    # The retry reached the service with the same key, which answered from the first commit.
    assert counts(service, action) == (2, 1, 1)
    assert [call[1] for call in connector.calls if call[0] == "submit"] == [action.action_key] * 2
    assert (connector.made("lookup"), connector.made("read")) == (0, 1)
    after = intact(store, action, before)
    first, retry = after["attempts"]
    assert (first["state"], retry["number"], retry["state"], retry["approval_id"]) == (
        "outcome_unknown", 2, "acknowledged", approval.approval_id)
    assert after["approvals"][0]["consumed_by_attempt"] == retry["attempt_id"]
    # Neither receipt reports a cost, so both writes keep their exposure.
    assert balances(after) == (READ, 0, 2 * WRITE)
    (read,) = records(store, action, "readback")
    assert read["payload"]["attempt_id"] == retry["attempt_id"] and report(read).passed
    assert records(store, action, "lookup") == [] and findings(store, action) == []
    states = [event["data"]["state"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_attempt"]
    assert states == ["prepared", "dispatched", "outcome_unknown", "dispatched", "acknowledged"]


@pytest.mark.parametrize("elapsed", [TIMEOUT + 1, 61], ids=["inside-the-margin", "after-expiry"])
def test_no_retry_once_the_key_window_may_have_closed(
        store, tmp_path, replies, host, clock, elapsed):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False, key_ttl_s=60)
    connector = Reconciling(service)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)
    # The retry must arrive inside the window: the connector's timeout is the margin.
    clock.advance(elapsed)
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    (finding,) = findings(store, action)
    assert finding["reason"] == ("the key's deduplication window may have closed, so a retry could take "
                                 "effect twice")
    assert connector.made("submit") == 1 and counts(service, action) == (1, 1, 0)
    intact(store, action, before)
    # The journal refuses the retry too.
    lease = store.acquire_lease(action.run_id, "another-worker", ttl_s=60)
    with pytest.raises(ActionStateError, match="may not deduplicate the retry"):
        workflow_actions.claim_retry(store, lease, action.action_id, before["approvals"][0]["approval_id"])
    assert item(store, action)["attempts"] == before["attempts"] and store.audit(action.run_id)["ok"]


def test_a_refused_retry_leaves_the_outcome_unknown_and_attempts_bound_the_retries(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False)
    connector = Reconciling(service)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)
    service.fail_before_commit(action.action_key, delivery=1)
    worker = runner(store, connector, host, clock)

    refused = worker.reconcile(action.run_id, action.action_id)

    # The refusal says nothing of the first attempt, which may have taken effect.
    assert (refused.state, refused.attempt["state"], refused.next_action) == (
        "outcome_unknown", "refused", "reconcile")
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        worker.dispatch(action.run_id, action.action_id)
    # Both of the profile's attempts are used, so no further retry is made.
    exhausted = worker.reconcile(action.run_id, action.action_id)
    assert (exhausted.state, exhausted.next_action) == ("outcome_unknown", "operator_review")
    assert findings(store, action)[-1]["reason"] == "the action has made all 2 of its attempts"
    assert connector.made("submit") == 2 and counts(service, action) == (2, 1, 0)
    assert balances(item(store, action)) == (0, 0, WRITE)
    assert store.audit(action.run_id)["ok"]


@pytest.mark.parametrize("stop", ["revoked", "security-stop", "single-attempt"])
def test_a_retry_that_is_not_allowed_leaves_the_action_unresolved(
        store, tmp_path, replies, host, clock, stop):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False)
    connector = Reconciling(service, max_attempts=1) if stop == "single-attempt" else Reconciling(service)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector)
    if stop == "revoked":
        revoke(store, approval.approval_id, "withdrawn")
    elif stop == "security-stop":
        execute(store.path, "UPDATE workflow_runs SET blocked_reason='security_stop' WHERE run_id=?",
                (action.run_id,))
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    reason = findings(store, action)[0]["reason"]
    assert reason == {"revoked": f"no retry is allowed: Action {action.action_id!r} is not authorized: its "
                                 "approval was revoked",
                      "security-stop": f"no retry is allowed: Action {action.action_id!r} is not "
                                       "authorized: a security stop holds the run",
                      "single-attempt": "the action has made all 1 of its attempts"}[stop]
    assert connector.made("submit") == 1 and counts(service, action) == (1, 1, 0)
    intact(store, action, before)


# ---------------------------------------------------------------------------
# Neither lookup nor deduplication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reads", [True, False], ids=["with-reads", "without-reads"])
def test_without_lookup_or_deduplication_the_outcome_stays_unresolved_at_once(
        store, tmp_path, replies, host, clock, reads):  # noqa: F811
    service = FixtureService(clock_ns=clock, dedupe=False, lookup_by_key=False, lookup_by_object=reads)
    connector = Reconciling(service)
    assert not (connector.profile.deduplicates or connector.profile.lookup_by_key)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector)
    before = item(store, action)
    calls = list(connector.calls)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action, progress.report) == (
        "outcome_unknown", "operator_review", None)
    assert connector.calls == calls and counts(service, action) == (1, 1, 0)
    (finding,) = findings(store, action)
    assert (finding["finding"], finding["evidence"], finding["reason"]) == (
        "unresolved", [], "the connector declares neither lookup by key nor deduplication, so nothing "
                          "can show whether the write took effect")
    # The exposure stays, and so does the unconsumed approval, which authorizes nothing new.
    after = intact(store, action, before)
    assert balances(after) == (0, 0, WRITE)
    assert check_authority(store, action.action_id, host) == approval
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)


# ---------------------------------------------------------------------------
# A worker paused between its claim and its request
# ---------------------------------------------------------------------------


LEASE = 60  # each runner's lease, which a submission holds for its timeout beyond that
CONTRADICTION = "a late receipt shows that this attempt, which was found not executed, took effect"


class Paused(Reconciling):
    """A connector whose submissions wait at ``gate``, after the dispatch claim and before the request leaves."""

    def __init__(self, service: FixtureService, **settings) -> None:
        super().__init__(service, **settings)
        self.gate = Gate()

    def submit(self, key, intent, payload, timeout_s):
        self.gate.block()
        return super().submit(key, intent, payload, timeout_s)


class Blind(Reconciling):
    """A connector whose lookups miss every commit, as a service past its declared bound would."""

    def lookup(self, key, target, timeout_s):
        self.calls.append(("lookup", key))
        return None


def paused_past_its_lease(store, tmp_path, replies, host, clock, *,  # noqa: F811
                          dedupe=False, visibility=0, **settings):
    """An approved action whose worker claimed its dispatch and stalled, past its lease, before sending.

    The service looks keys up and shows each commit *visibility* seconds after
    it is made, at once by default; *settings* change the profile. Worker A, in
    a thread with its own journal connection, waits inside ``submit()``.
    Returns the action, the service, worker B's connector and runner, and
    ``resume()``, which lets A's request leave and returns what A's dispatch
    raised.
    """
    service = FixtureService(clock_ns=clock, dedupe=dedupe, visibility_delay_s=visibility)
    connector = Reconciling(service, timeout_s=10, **settings)
    paused = Paused(service, timeout_s=10, **settings)
    assert connector.profile == paused.profile and connector.profile.deduplicates is dedupe
    assert connector.profile.lookup_by_key and connector.profile.visibility_bound_s == visibility
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    outcome: dict = {}

    def work():
        with SQLiteWorkflowStore(store.path, clock_ns=clock) as journal:
            try:
                runner(journal, paused, host, clock, owner_id="worker-a", lease_ttl_s=LEASE).dispatch(
                    action.run_id, action.action_id)
            except BaseException as error:  # reported by the test, not lost in the thread
                outcome["error"] = error

    thread = threading.Thread(target=work)
    thread.start()
    assert paused.gate.entered.wait(30)
    assert item(store, action)["attempts"][0]["state"] == "dispatched"
    # A's lease, held for the connector's timeout beyond its duration, lapses while it waits.
    clock.advance(LEASE + connector.profile.timeout_s + 1)

    def resume():
        paused.gate.release()
        thread.join(30)
        assert not thread.is_alive()
        return outcome.get("error")

    return action, service, connector, runner(store, connector, host, clock, owner_id="worker-b",
                                              lease_ttl_s=LEASE), resume


def found_absent(worker, action) -> None:
    """Worker B recovers the stalled attempt and an empty lookup shows that its write was not made."""
    progress = worker.reconcile(action.run_id, action.action_id)
    assert (progress.state, progress.next_action) == ("not_dispatched", "dispatch")


def second_effect(store, tmp_path, replies, host, clock, *, dedupe=False):  # noqa: F811
    """Worker B dispatches again and observes the result; then the stalled request leaves.

    Returns the action, the service and the action as inspection showed it
    before the stalled request left.
    """
    action, service, _, worker, resume = paused_past_its_lease(
        store, tmp_path, replies, host, clock, dedupe=dedupe)
    found_absent(worker, action)
    assert worker.dispatch(action.run_id, action.action_id).attempt["state"] == "acknowledged"
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    before = item(store, action)
    error = resume()
    assert isinstance(error, WorkflowLeaseError) and "kept as a late receipt record" in str(error)
    return action, service, before


def test_a_late_receipt_that_contradicts_a_not_executed_finding_leaves_the_action_unresolved(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, service, before = second_effect(store, tmp_path, replies, host, clock)

    # Without deduplication, the stalled request committed a second object.
    assert counts(service, action) == (2, 2, 0) and len(service.snapshot()["objects"]) == 2
    first, retry = before["attempts"]
    (late,) = records(store, action, "late_receipt")
    assert (late["payload"]["attempt_id"], late["payload"]["outcome"]) == (first["attempt_id"], "acknowledged")
    absent, contradicted = findings(store, action)
    assert (absent["attempt_id"], absent["finding"]) == (first["attempt_id"], "not_executed")
    assert contradicted == {"attempt_id": first["attempt_id"], "finding": "contradicted_absence",
                            "reason": CONTRADICTION, "evidence": [late["record_id"]], "attempts": 2}
    # Only records were appended; both writes keep their exposure.
    after = intact(store, action, before)
    assert (after["state"], after["attempts"]) == ("observed", before["attempts"])
    assert balances(after) == (2 * READ, 0, 2 * WRITE)
    events = [event["data"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_finding"]
    assert events[-1] == {"action_id": action.action_id, "record_id": records(store, action, "finding")[-1][
        "record_id"], "finding": "contradicted_absence", "state": "observed"}

    # The action is listed unresolved, with the second effect to compensate.
    view = inspect_actions(store, action.run_id)
    (shown,) = view["items"]
    assert view["unresolved"] == [action.action_id]
    assert (shown["state"], shown["contradiction"], shown["next_action"]) == ("observed", "second_effect", "compensate")
    with pytest.raises(ActionNotAuthorized, match="it is observed"):
        check_authority(store, action.action_id, host)
    assert retry["state"] == "acknowledged"


def test_a_deduplicating_service_answers_the_stalled_request_and_nothing_is_unresolved(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, service, before = second_effect(store, tmp_path, replies, host, clock, dedupe=True)

    # The service answered the stalled request from the commit that the second attempt made.
    assert counts(service, action) == (2, 1, 1) and len(service.snapshot()["objects"]) == 1
    (late,) = records(store, action, "late_receipt")
    assert late["payload"]["receipt"]["deduplicated"] is True
    assert [finding["finding"] for finding in findings(store, action)] == ["not_executed"]
    after = intact(store, action, before)
    assert after["state"] == "observed"
    view = inspect_actions(store, action.run_id)
    assert view["unresolved"] == []
    assert (view["items"][0]["contradiction"], view["items"][0]["next_action"]) == (None, None)


@pytest.mark.parametrize("begun", [False, True], ids=["not-dispatched", "an-attempt-begun"])
def test_a_late_receipt_before_the_next_dispatch_stops_it_and_reconciliation_reads_the_write(
        store, tmp_path, replies, host, clock, begun):  # noqa: F811
    action, service, connector, worker, resume = paused_past_its_lease(store, tmp_path, replies, host, clock)
    found_absent(worker, action)
    if begun:
        lease = store.acquire_lease(action.run_id, "worker-c", ttl_s=LEASE)
        approval_id = item(store, action)["approvals"][0]["approval_id"]
        attempt = workflow_actions.begin_attempt(store, lease, action.action_id, approval_id)
    before = item(store, action)

    assert isinstance(resume(), WorkflowLeaseError)

    # The write that was found not made took effect: the action's outcome is open again,
    # an attempt begun meanwhile is released unsent, and nothing dispatches another write.
    after = item(store, action)
    assert after["state"] == "outcome_unknown"
    assert [entry["state"] for entry in after["attempts"]] == ["outcome_unknown"] + ["released"] * begun
    assert after["records"][:len(before["records"])] == before["records"] and store.audit(action.run_id)["ok"]
    assert [finding["finding"] for finding in findings(store, action)] == ["not_executed", "contradicted_absence"]
    view = inspect_actions(store, action.run_id)
    assert view["unresolved"] == [action.action_id]
    assert (view["items"][0]["contradiction"], view["items"][0]["next_action"]) == (
        "contradicted_absence", "reconcile")
    if begun:
        with pytest.raises(ActionStateError, match="is released"):
            workflow_actions.claim_dispatch(store, lease, attempt["attempt_id"])
        store.release_lease(lease)
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        worker.dispatch(action.run_id, action.action_id)
    assert connector.made("submit") == 0 and counts(service, action) == (1, 1, 0)

    # Reconciliation finds the stalled request's write and reads it back: one effect.
    progress = worker.reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("reconciled", None) and progress.report.passed
    assert connector.made("submit") == 0 and counts(service, action) == (1, 1, 0)
    view = inspect_actions(store, action.run_id)
    assert view["unresolved"] == [] and view["items"][0]["contradiction"] == "contradicted_absence"
    assert store.audit(action.run_id)["ok"]


def test_no_lookup_or_decision_finds_an_attempt_not_executed_once_a_late_receipt_shows_its_write(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, service, connector, worker, resume = paused_past_its_lease(store, tmp_path, replies, host, clock)
    # Worker B recovers the stalled attempt but does not reconcile it before the receipt arrives.
    assert worker.readback(action.run_id, action.action_id).state == "outcome_unknown"
    assert isinstance(resume(), WorkflowLeaseError)
    assert findings(store, action) == []

    progress = runner(store, Blind(service, timeout_s=10), host, clock).reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    (finding,) = findings(store, action)
    assert (finding["finding"], finding["reason"]) == (
        "unresolved", "a late receipt shows that the write took effect, so an empty lookup proves nothing")
    (attempt,) = item(store, action)["attempts"]
    lease = store.acquire_lease(action.run_id, "worker-c", ttl_s=LEASE)
    with pytest.raises(ActionStateError, match="a late receipt shows that it took effect"):
        workflow_actions.record_finding(store, lease, attempt["attempt_id"], "not_executed", "a guess")
    store.release_lease(lease)
    attached = ActionEvidence.for_bytes(b"audit-log export", "No write under the action's key")
    assertion = host.issue(decision_sha256(store, action.action_id, "not_executed", attached),
                           subject="operator-1", tenant="acme", expires_at_ns=FAR)
    with pytest.raises(ActionStateError, match="a late receipt shows that attempt 1 took effect"):
        resolve(store, action.action_id, "not_executed", assertion=assertion, verifier=host, evidence=attached,
                note="Checked the audit log")
    assert [entry["finding"] for entry in findings(store, action)] == ["unresolved"]
    assert item(store, action)["state"] == "outcome_unknown" and counts(service, action) == (1, 1, 0)


LATER_ATTEMPT = {
    # The service shows the stalled request's commit to lookups only five seconds after it is made.
    "commit-not-yet-visible": (5, Reconciling),
    "lookups-miss-the-commit": (0, Blind),
}


@pytest.mark.parametrize("config", LATER_ATTEMPT.values(), ids=LATER_ATTEMPT.keys())
def test_a_late_commit_keeps_every_attempt_of_its_action_from_being_found_not_executed(
        store, tmp_path, replies, host, clock, config):  # noqa: F811
    visibility, looking = config
    action, service, connector, worker, resume = paused_past_its_lease(
        store, tmp_path, replies, host, clock, visibility=visibility, max_attempts=3)
    found_absent(worker, action)
    # Worker B's next attempt is lost before it reaches the service, so its outcome is unknown.
    connector.lost = 1
    assert worker.dispatch(action.run_id, action.action_id).state == "outcome_unknown"
    clock.advance(connector.profile.timeout_s + 2)
    assert isinstance(resume(), WorkflowLeaseError)
    first, second = item(store, action)["attempts"]
    assert [entry["finding"] for entry in findings(store, action)] == ["not_executed", "contradicted_absence"]
    # The visibility bound plus the timeout has now passed since the second attempt's claim.
    clock.advance(3)
    before = item(store, action)

    progress = runner(store, looking(service, timeout_s=10, max_attempts=3), host, clock,
                      lease_ttl_s=LEASE).reconcile(action.run_id, action.action_id)

    # The stalled request committed under the action's key, so no empty lookup proves any attempt absent.
    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    *_, finding = findings(store, action)
    assert (finding["attempt_id"], finding["finding"], finding["reason"]) == (
        second["attempt_id"], "unresolved",
        "a late receipt shows that the write took effect, so an empty lookup proves nothing")
    after = intact(store, action, before)
    assert [entry["state"] for entry in after["attempts"]] == ["outcome_unknown"] * 2
    view = inspect_actions(store, action.run_id)
    assert view["unresolved"] == [action.action_id]
    assert (view["items"][0]["contradiction"], view["items"][0]["next_action"]) == (
        "contradicted_absence", "operator_review")
    # Nothing dispatches a third attempt: one write reached the service.
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        worker.dispatch(action.run_id, action.action_id)
    assert counts(service, action) == (1, 1, 0) and len(service.snapshot()["objects"]) == 1
    # The journal refuses a not_executed finding for either attempt.
    lease = store.acquire_lease(action.run_id, "worker-c", ttl_s=LEASE)
    for attempt, shown in ((first, "it"), (second, "attempt 1")):
        with pytest.raises(ActionStateError, match=f"a late receipt shows that {shown} took effect"):
            workflow_actions.record_finding(store, lease, attempt["attempt_id"], "not_executed", "a guess")
    store.release_lease(lease)
    assert item(store, action)["state"] == "outcome_unknown" and store.audit(action.run_id)["ok"]


def test_a_not_executed_finding_leaves_the_outcome_unknown_while_another_attempt_is_unsettled(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, lookup_by_key=False)
    connector = Reconciling(service, max_attempts=3)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector, committed=False)
    # The same-key retry is lost too, so the outcomes of both attempts are unknown.
    connector.lost = 1
    worker = runner(store, connector, host, clock)
    assert worker.reconcile(action.run_id, action.action_id).state == "outcome_unknown"
    first, retry = item(store, action)["attempts"]
    assert (first["state"], retry["state"]) == ("outcome_unknown", "outcome_unknown")
    lease = store.acquire_lease(action.run_id, "host-reconciler", ttl_s=LEASE)

    workflow_actions.record_finding(store, lease, retry["attempt_id"], "not_executed", "the retry never arrived")

    # The first attempt may still have taken effect, so nothing dispatches the action.
    assert item(store, action)["state"] == "outcome_unknown"
    assert records(store, action, "finding")[-1]["payload"]["finding"] == "not_executed"
    store.release_lease(lease)
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        worker.dispatch(action.run_id, action.action_id)
    assert inspect_actions(store, action.run_id)["items"][0]["next_action"] == "reconcile"

    # Once the first attempt is found not executed as well, the next attempt makes the one write.
    lease = store.acquire_lease(action.run_id, "host-reconciler", ttl_s=LEASE)
    workflow_actions.record_finding(store, lease, first["attempt_id"], "not_executed", "it never arrived either")
    store.release_lease(lease)
    assert item(store, action)["state"] == "not_dispatched"
    assert worker.dispatch(action.run_id, action.action_id).attempt["state"] == "acknowledged"
    assert connector.made("submit") == 3 and counts(service, action) == (1, 1, 0)
    assert store.audit(action.run_id)["ok"]


# ---------------------------------------------------------------------------
# Accounting against the approval's ceiling
# ---------------------------------------------------------------------------


def test_lookups_and_reads_stop_at_the_approvals_ceiling(
        store, tmp_path, replies, host, clock):  # noqa: F811
    # One write and two reads: the ambiguous write's exposure leaves room for two lookups only.
    service = FixtureService(clock_ns=clock, visibility_delay_s=5)
    connector = Reconciling(service, max_attempts=1, readback_interval_s=5)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)
    assert action.ceiling_nanousd == WRITE + 2 * READ
    before = item(store, action)

    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)

    # The second lookup found the receipt, but no read fits under the ceiling.
    assert (progress.state, progress.next_action, progress.report) == (
        "outcome_unknown", "operator_review", None)
    assert (connector.made("lookup"), connector.made("read")) == (2, 0)
    assert records(store, action, "lookup")[-1]["payload"]["receipt"] is not None
    assert findings(store, action)[0]["reason"] == "the approval's ceiling leaves no room for another read"
    after = intact(store, action, before)
    assert balances(after) == (2 * READ, 0, WRITE) and sum(balances(after)) == action.ceiling_nanousd


@pytest.mark.parametrize("lookups", [2, 1], ids=["two-lookups-leave-no-read", "one-lookup-leaves-one-read"])
def test_an_attempt_made_after_a_not_executed_finding_reads_only_within_the_ceiling(
        store, tmp_path, replies, host, clock, lookups):  # noqa: F811
    # The lookups that showed the write was not made used reads the approval's ceiling covers.
    service = FixtureService(clock_ns=clock, visibility_delay_s=5)
    connector = Reconciling(service, readback_interval_s=1 if lookups == 2 else 5)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector, committed=False)
    assert approval.ceiling_nanousd == action.ceiling_nanousd == 2 * WRITE + 2 * READ
    # With two lookups, the first comes a second before the bound has passed.
    clock.advance(5 + TIMEOUT - (lookups - 1))
    worker = runner(store, connector, host, clock)
    assert worker.reconcile(action.run_id, action.action_id).next_action == "dispatch"
    assert connector.made("lookup") == lookups and findings(store, action)[0]["finding"] == "not_executed"

    dispatched = worker.dispatch(action.run_id, action.action_id)

    assert dispatched.attempt["state"] == "acknowledged"
    assert dispatched.next_action == ("readback" if lookups == 1 else "reconcile")
    # The new commit is not visible yet: a read that fits decides nothing, and none is made past the ceiling.
    progress = worker.readback(action.run_id, action.action_id)
    after = item(store, action)
    assert sum(balances(after)) <= approval.ceiling_nanousd
    assert balances(after) == (2 * READ, 0, 2 * WRITE) and connector.made("read") == 2 - lookups
    assert (progress.state, progress.next_action) == ("dispatched", "reconcile")

    # Even once the commit is visible, no read fits: the result goes to an operator, unread.
    clock.advance(10)
    reviewed = worker.reconcile(action.run_id, action.action_id)
    assert (reviewed.state, reviewed.attempt["state"], reviewed.next_action) == (
        "dispatched", "acknowledged", "operator_review")
    assert findings(store, action)[-1]["reason"] == "the approval's ceiling leaves no room for another read"
    assert worker.readback(action.run_id, action.action_id).next_action == "operator_review"
    assert (connector.made("read"), connector.made("submit")) == (2 - lookups, 2)
    assert counts(service, action) == (1, 1, 0)
    final = intact(store, action, after)
    assert balances(final) == balances(after) and store.audit(action.run_id)["ok"]


# ---------------------------------------------------------------------------
# Acknowledged but unverified results
# ---------------------------------------------------------------------------


def test_an_unverified_receipt_goes_to_operator_review_without_another_submission(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=60)
    connector = Reconciling(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).next_action == "reconcile"
    before = item(store, action)

    progress = worker.reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "dispatched", "acknowledged", "operator_review")
    (finding,) = findings(store, action)
    assert (finding["finding"], finding["reason"]) == (
        "unresolved", "no read confirmed the receipt's object within the read bound")
    assert (connector.made("submit"), connector.made("lookup"), connector.made("read")) == (1, 0, 2)
    assert worker.readback(action.run_id, action.action_id).next_action == "operator_review"
    intact(store, action, before)


def test_reconcile_reads_back_an_acknowledged_attempt_with_reads_left(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock)
    connector = Reconciling(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    assert worker.dispatch(action.run_id, action.action_id).next_action == "readback"

    progress = worker.reconcile(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("observed", None) and progress.report.passed
    assert findings(store, action) == []


@pytest.mark.parametrize("state", ["not_dispatched", "observed"])
def test_reconcile_leaves_a_settled_or_undispatched_action_alone(
        store, tmp_path, replies, host, clock, state):  # noqa: F811
    connector = Reconciling(FixtureService(clock_ns=clock))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    if state == "observed":
        worker.dispatch(action.run_id, action.action_id)
        worker.readback(action.run_id, action.action_id)
    before, calls = item(store, action), list(connector.calls)

    progress = worker.reconcile(action.run_id, action.action_id)

    assert progress.state == state and progress.report is None
    assert connector.calls == calls and item(store, action) == before


# ---------------------------------------------------------------------------
# The journal's reconciliation records
# ---------------------------------------------------------------------------


def test_the_journal_refuses_findings_and_lookups_that_do_not_apply(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock)
    connector = Reconciling(service)
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)
    (attempt,) = item(store, action)["attempts"]
    lease = store.acquire_lease(action.run_id, "worker", ttl_s=60)

    # An acknowledged attempt is never found absent, and its outcome is not looked up.
    with pytest.raises(ActionStateError, match="no not_executed finding applies"):
        workflow_actions.record_finding(store, lease, attempt["attempt_id"], "not_executed", "a guess")
    with pytest.raises(ActionStateError, match="only an unknown outcome is reconciled"):
        workflow_actions.record_lookup(store, lease, attempt["attempt_id"], 1, None, None, READ)
    with pytest.raises(ActionStateError, match="only an unknown outcome is retried"):
        workflow_actions.claim_retry(store, lease, action.action_id, approval.approval_id)
    with pytest.raises(WorkflowValidationError, match="A finding is one of"):
        workflow_actions.record_finding(store, lease, attempt["attempt_id"], "reconciled", "a guess")
    with pytest.raises(WorkflowValidationError, match="evidence names records of its action"):
        workflow_actions.record_finding(store, lease, attempt["attempt_id"], "unresolved", "x", ["missing"])
    assert findings(store, action) == [] and store.audit(action.run_id)["ok"]


def test_a_read_of_an_unknown_outcome_needs_a_receipt_a_lookup_found(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock)
    connector = Reconciling(service)
    action, _ = unknown(store, tmp_path, replies, host, clock, connector)
    (attempt,) = item(store, action)["attempts"]
    lease = store.acquire_lease(action.run_id, "worker", ttl_s=60)

    with pytest.raises(ActionStateError, match="whose receipt a lookup found"):
        workflow_actions.record_readback(store, lease, attempt["attempt_id"], {}, READ, "observed")
    with pytest.raises(WorkflowValidationError, match="another key or target"):
        workflow_actions.record_lookup(store, lease, attempt["attempt_id"], 1, replace(
            connector.lookup(action.action_key, action.intent.target, 30), key="another-key"), None, READ)
    assert records(store, action, "lookup") == [] and records(store, action, "readback") == []
