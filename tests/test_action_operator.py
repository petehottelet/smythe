"""Operator decisions on external actions whose outcome reconciliation left open.

``resolve`` records a decision that the host's verifier authenticated for one
action at one point in its history. ``adopt_observed`` closes the action as
reconciled on a readback that showed the bound content, or on attached
evidence, without submitting anything. ``not_executed`` needs evidence and
returns the action to dispatch with duplicate risk, which only a new approval
that accepts it can authorize. ``abandon`` closes the action as unresolved
with its exposure. Decisions only append records, and ``inspect_actions``
shows what each action needs next.
"""

from __future__ import annotations

from dataclasses import replace
import json
import time

import pytest

from fakes import ManualClock
from fixture_service import FixtureService
from smythe import SQLiteWorkflowStore, workflow_actions
from smythe.actions import (
    ACTION_DECISIONS,
    ActionEvidence,
    ActionNotAuthorized,
    ActionStateError,
    ApprovalRejected,
    approve,
    check_authority,
    decision_sha256,
    duplicate_risk_sha256,
    inspect_actions,
    load_action,
    prepare_action,
    prepare_compensation,
    resolve,
    revoke,
)
from smythe.workflow_store import WorkflowConflictError, WorkflowValidationError
from test_action_approvals import FAR, HOUR_NS, ORGANIZATION, Host, delivered, intent, sign
from test_action_journal import WRITE, balances, item, rows
from test_action_reconciliation import Reconciling, unknown
from test_action_runner import READ, Recording, codes, ready, records, report, runner
from test_workflow_acceptance import replies  # noqa: F401

NOTE = "Checked the service's audit log by hand"


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


class Lagging(Recording):
    """Reads return each object as it was one version earlier: the bound bytes, at a stale version."""

    def read(self, target, timeout_s):
        observed = super().read(target, timeout_s)
        return observed and replace(observed, version=observed.version - 1)


class Overwritten(Recording):
    """Reads find other bytes than the action wrote."""

    def read(self, target, timeout_s):
        observed = super().read(target, timeout_s)
        return observed and replace(observed, content=b"someone else's document")


class Missing(Recording):
    """Reads find no object."""

    def read(self, target, timeout_s):
        super().read(target, timeout_s)
        return None


class Superseded(Recording):
    """Reads find the version before the write, holding other bytes."""

    def read(self, target, timeout_s):
        observed = super().read(target, timeout_s)
        return observed and replace(observed, version=observed.version - 1, content=b"the previous draft")


class Rekeyed(Recording):
    """Reads find the bound bytes at a stale version, written under another key."""

    def read(self, target, timeout_s):
        observed = super().read(target, timeout_s)
        return observed and replace(observed, version=observed.version - 1, key="another-writers-key")


def opaque(clock, **settings) -> Reconciling:
    """A connector that declares neither lookup by key nor deduplication."""
    return Reconciling(FixtureService(clock_ns=clock, dedupe=False, lookup_by_key=False), **settings)


def evidence(summary="The service's audit log has no write under the action's key") -> ActionEvidence:
    return ActionEvidence.for_bytes(b"audit-log export: raw-body-not-for-the-journal", summary)


def assertion_for(store, host, action, decision, attached=None, *, subject="operator-1", tenant="acme") -> str:
    return host.issue(decision_sha256(store, action.action_id, decision, attached), subject=subject,
                      tenant=tenant, expires_at_ns=FAR)


def decide(store, host, action, decision, attached=None, **options):
    return resolve(store, action.action_id, decision, assertion=assertion_for(store, host, action, decision, attached),
                   verifier=host, evidence=attached, note=options.get("note", NOTE))


def unverified(store, tmp_path, replies, host, clock):  # noqa: F811
    """An acknowledged write whose reads found the bound bytes only at a stale version."""
    connector = Lagging(FixtureService(clock_ns=clock))
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).next_action == "reconcile"
    progress = worker.reconcile(action.run_id, action.action_id)
    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "dispatched", "acknowledged", "operator_review")
    assert [codes(report(record)) for record in records(store, action, "readback")] == [["stale_version"]] * 2
    return action, approval, connector


def awaiting_readback(store, tmp_path, replies, host, clock, connector=None):  # noqa: F811
    """An acknowledged write whose reads have not run yet."""
    connector = connector or Recording(FixtureService(clock_ns=clock))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    progress = runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)
    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "dispatched", "acknowledged", "readback")
    return action, connector


def under_review(store, tmp_path, replies, host, clock, *, committed=False, **settings):  # noqa: F811
    """An approved action whose one dispatch left its outcome unknown, with nothing able to settle it."""
    connector = opaque(clock, **settings)
    action, approval = unknown(store, tmp_path, replies, host, clock, connector, committed=committed)
    progress = runner(store, connector, host, clock).reconcile(action.run_id, action.action_id)
    assert (progress.state, progress.next_action) == ("outcome_unknown", "operator_review")
    return action, approval, connector


def appended_only(store, action, before, state) -> dict:
    """The action now, after checking that a decision changed its state and appended records, nothing else."""
    after = item(store, action)
    assert after["state"] == state
    assert after["records"][:len(before["records"])] == before["records"]
    unchanged = {key: value for key, value in after.items() if key not in ("state", "records")}
    assert unchanged == {key: value for key, value in before.items() if key not in ("state", "records")}
    assert store.audit(action.run_id)["ok"]
    return after


def payload(store, record_id) -> dict:
    (row,) = [row for row in rows(store.path, "workflow_action_records") if row["record_id"] == record_id]
    return json.loads(row["payload_json"])


def journal(store) -> dict:
    """Every action table's rows, to show that a refusal changed nothing."""
    return {table: rows(store.path, table) for table in (
        "workflow_actions", "workflow_action_attempts", "workflow_action_approvals", "workflow_action_records")}


# ---------------------------------------------------------------------------
# adopt_observed
# ---------------------------------------------------------------------------


def test_adopting_a_readback_reconciles_the_action_without_a_new_submission(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, _, connector = unverified(store, tmp_path, replies, host, clock)
    reads = [record["record_id"] for record in records(store, action, "readback")]
    before, calls = item(store, action), list(connector.calls)

    decided = decide(store, host, action, "adopt_observed")

    assert decided == replace(action, state="reconciled") == load_action(store, action.action_id)
    assert connector.calls == calls and connector.made("submit") == 1
    assert len(connector.service.snapshot()["deliveries"]) == 1
    after = appended_only(store, action, before, "reconciled")
    (record,) = after["records"][len(before["records"]):]
    assert (record["kind"], record["trusted"], record["cost_nanousd"]) == ("decision", True, None)
    recorded = payload(store, record["record_id"])
    digest = recorded.pop("decision_sha256")
    assert recorded == {
        "decision": "adopt_observed", "state": "dispatched", "attempts": 1, "evidence": reads,
        "principal": {"issuer": host.issuer, "subject": "operator-1", "tenant": "acme"},
        "verifier": host.identity, "note": NOTE}
    assert host.verified[-1] == digest
    # The receipt's unreported cost stays unknown exposure: state and accounting are separate.
    assert balances(after) == (2 * READ, 0, WRITE)
    events = [event["data"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_decision"]
    assert events == [{"action_id": action.action_id, "record_id": record["record_id"],
                       "decision": "adopt_observed", "state": "reconciled"}]
    # A reconciled action is settled: nothing authorizes it, and reconciling calls nothing.
    with pytest.raises(ActionNotAuthorized, match="it is reconciled"):
        check_authority(store, action.action_id, host)
    assert runner(store, connector, host, clock).reconcile(action.run_id, action.action_id).next_action is None
    assert connector.calls == calls
    view = inspect_actions(store, action.run_id)
    assert (view["unresolved"], view["items"][0]["next_action"]) == ([], None)


def test_adopting_an_unknown_outcome_needs_evidence_when_no_read_showed_the_content(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, _, connector = under_review(store, tmp_path, replies, host, clock, committed=True)
    assert records(store, action, "readback") == []
    before, untouched = item(store, action), journal(store)

    with pytest.raises(WorkflowValidationError, match="needs attached evidence or a readback"):
        decide(store, host, action, "adopt_observed")
    assert journal(store) == untouched

    attached = evidence("The service's console shows the post under the action's key")
    assert decide(store, host, action, "adopt_observed", attached).state == "reconciled"
    after = appended_only(store, action, before, "reconciled")
    kept, decision = after["records"][len(before["records"]):]
    assert (kept["kind"], decision["kind"]) == ("evidence", "decision")
    assert payload(store, kept["record_id"]) == attached.to_dict()
    assert payload(store, decision["record_id"])["evidence"] == [kept["record_id"]]
    assert connector.made("submit") == 1
    # The ambiguous attempt keeps its exposure.
    assert after["attempts"][0]["unknown_nanousd"] == WRITE and balances(after) == (0, 0, WRITE)


@pytest.mark.parametrize("reading,code", [(Missing, "not_visible"), (Superseded, "stale_version"),
                                          (Rekeyed, "stale_version")],
                         ids=["no-object", "older-other-bytes", "older-under-another-key"])
def test_adopting_needs_evidence_when_the_reads_did_not_show_the_bound_content(
        store, tmp_path, replies, host, clock, reading, code):  # noqa: F811
    action, connector = awaiting_readback(store, tmp_path, replies, host, clock,
                                          reading(FixtureService(clock_ns=clock)))
    # An acknowledged write with reads left is in progress, not unresolved.
    assert inspect_actions(store, action.run_id)["unresolved"] == []
    worker = runner(store, connector, host, clock)
    assert worker.readback(action.run_id, action.action_id).next_action == "reconcile"
    assert [codes(report(record)) for record in records(store, action, "readback")] == [[code]] * 2
    # With no read left to confirm it, the dispatched action is unresolved.
    assert inspect_actions(store, action.run_id)["unresolved"] == [action.action_id]
    assert worker.reconcile(action.run_id, action.action_id).next_action == "operator_review"
    view = inspect_actions(store, action.run_id)
    assert (view["unresolved"], view["items"][0]["state"]) == ([action.action_id], "dispatched")
    before, untouched = item(store, action), journal(store)

    with pytest.raises(WorkflowValidationError, match="needs attached evidence or a readback"):
        decide(store, host, action, "adopt_observed")
    assert journal(store) == untouched

    attached = evidence("The service's console shows the post under the action's key")
    assert decide(store, host, action, "adopt_observed", attached).state == "reconciled"
    after = appended_only(store, action, before, "reconciled")
    kept, decision = after["records"][len(before["records"]):]
    # The decision rests on the evidence alone, never on reads that did not show the content.
    assert payload(store, decision["record_id"])["evidence"] == [kept["record_id"]]
    assert connector.made("submit") == 1
    assert inspect_actions(store, action.run_id)["unresolved"] == []


# ---------------------------------------------------------------------------
# not_executed and duplicate risk
# ---------------------------------------------------------------------------


def test_not_executed_needs_a_new_approval_that_accepts_the_duplicate_risk(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, original, connector = under_review(store, tmp_path, replies, host, clock, max_attempts=3)
    before, untouched = item(store, action), journal(store)
    with pytest.raises(WorkflowValidationError, match="not_executed decision needs attached evidence"):
        decide(store, host, action, "not_executed")
    assert journal(store) == untouched

    decided = decide(store, host, action, "not_executed", evidence())

    assert decided.state == "not_dispatched"
    appended_only(store, action, before, "not_dispatched")
    view = inspect_actions(store, action.run_id)["items"][0]
    assert (view["duplicate_risk"], view["next_action"]) == (True, "approve_duplicate_risk")
    # The original approval is unconsumed and unrevoked, yet it no longer authorizes a dispatch.
    assert view["approvals"][0]["consumed_by_attempt"] is None and view["approvals"][0]["revoked_at_ns"] is None
    with pytest.raises(ActionNotAuthorized, match="does not accept the duplicate risk"):
        check_authority(store, action.action_id, host)
    # The journal refuses it as well, in the transaction that would begin an attempt.
    decided_journal, lease = journal(store), store.acquire_lease(action.run_id, "worker-1", ttl_s=60)
    with pytest.raises(ActionNotAuthorized, match="does not accept the duplicate risk"):
        workflow_actions.begin_attempt(store, lease, action.action_id, original.approval_id)
    store.release_lease(lease)
    assert journal(store) == decided_journal
    worker = runner(store, connector, host, clock)
    with pytest.raises(ActionNotAuthorized, match="does not accept the duplicate risk"):
        worker.dispatch(action.run_id, action.action_id)
    (refused,) = worker.dispatch_pending(action.run_id)
    assert "does not accept the duplicate risk" in refused.refusal
    assert connector.made("submit") == 1

    # An approval for the intent alone, or one that names the risk without accepting it, is refused.
    expiry = clock.now_ns + HOUR_NS
    with pytest.raises(ActionStateError, match="duplicate_risk=True"):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=expiry)
    with pytest.raises(ApprovalRejected, match="another intent"):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=expiry, duplicate_risk=True)
    assert rows(store.path, "workflow_action_approvals") == untouched["workflow_action_approvals"]

    def accept_risk():
        assertion = host.issue(duplicate_risk_sha256(store, action.action_id), subject="editor-1", tenant="acme",
                               expires_at_ns=FAR)
        return approve(store, action.action_id, assertion, host, ceiling_nanousd=action.ceiling_nanousd,
                       expires_at_ns=expiry, duplicate_risk=True)

    risky = accept_risk()
    assert check_authority(store, action.action_id, host) == risky
    assert rows(store.path, "workflow_action_approvals")[-1]["duplicate_risk"] == 1
    assert inspect_actions(store, action.run_id)["items"][0]["next_action"] == "dispatch"

    # The approval is bound to the attempts the decision found: after another lost
    # dispatch and decision, it no longer authorizes, and only a new one does.
    connector.lost = 1
    assert worker.dispatch(action.run_id, action.action_id).state == "outcome_unknown"
    assert worker.reconcile(action.run_id, action.action_id).next_action == "operator_review"
    decide(store, host, action, "not_executed", evidence("The retry is missing from the audit log too"))
    with pytest.raises(ActionNotAuthorized, match="the approval verifier refuses it"):
        check_authority(store, action.action_id, host)
    # The journal refuses it too: an attempt was made under it for the earlier decision.
    second_decision, lease = journal(store), store.acquire_lease(action.run_id, "worker-1", ttl_s=60)
    with pytest.raises(ActionNotAuthorized, match="accepted an earlier duplicate risk"):
        workflow_actions.begin_attempt(store, lease, action.action_id, risky.approval_id)
    store.release_lease(lease)
    assert journal(store) == second_decision
    renewed = accept_risk()
    assert check_authority(store, action.action_id, host) == renewed

    progress = worker.dispatch(action.run_id, action.action_id)
    assert (progress.state, progress.attempt["number"], progress.next_action) == ("dispatched", 3, "readback")
    # The decisions settled the earlier attempts, so a passed read observes the action.
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    assert connector.made("submit") == 3 and len(connector.service.snapshot()["deliveries"]) == 1
    final = item(store, action)
    assert [attempt["unknown_nanousd"] for attempt in final["attempts"]] == [WRITE, WRITE, WRITE]
    assert final["attempts"][0] == before["attempts"][0]
    assert store.audit(action.run_id)["ok"]


def test_duplicate_risk_is_accepted_only_where_a_decision_left_it(store, tmp_path, replies, host, clock):  # noqa: F811
    connector = Recording(FixtureService(clock_ns=clock))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    with pytest.raises(ActionStateError, match="no duplicate risk"):
        duplicate_risk_sha256(store, action.action_id)
    with pytest.raises(ActionStateError, match="no duplicate risk"):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=clock.now_ns + HOUR_NS, duplicate_risk=True)
    with pytest.raises(WorkflowValidationError, match="duplicate_risk"):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=clock.now_ns + HOUR_NS, duplicate_risk=1)


# ---------------------------------------------------------------------------
# abandon
# ---------------------------------------------------------------------------


def _outcome_unknown(store, tmp_path, replies, host, clock):  # noqa: F811
    return under_review(store, tmp_path, replies, host, clock, committed=True)[::2]


def _unverified(store, tmp_path, replies, host, clock):  # noqa: F811
    return unverified(store, tmp_path, replies, host, clock)[::2]


def _wrong(store, tmp_path, replies, host, clock):  # noqa: F811
    connector = Overwritten(FixtureService(clock_ns=clock))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).state == "compensation_required"
    return action, connector


@pytest.mark.parametrize("setup", [_outcome_unknown, _unverified, _wrong],
                         ids=["outcome-unknown", "unverified", "compensation-required"])
def test_abandoning_closes_the_action_as_unresolved_and_keeps_its_exposure(
        store, tmp_path, replies, host, clock, setup):  # noqa: F811
    action, connector = setup(store, tmp_path, replies, host, clock)
    before, calls = item(store, action), list(connector.calls)
    # An effect that is unknown, unverified or wrong is unresolved before any decision.
    assert inspect_actions(store, action.run_id)["unresolved"] == [action.action_id]

    assert decide(store, host, action, "abandon").state == "unresolved"

    after = appended_only(store, action, before, "unresolved")
    assert balances(after) == balances(before) and after["unknown_nanousd"] == WRITE
    (record,) = after["records"][len(before["records"]):]
    assert payload(store, record["record_id"])["evidence"] == []
    # An unresolved action is closed: nothing approves, authorizes, settles or compensates it.
    with pytest.raises(ActionStateError, match="unresolved and cannot be approved"):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
                expires_at_ns=clock.now_ns + HOUR_NS)
    with pytest.raises(ActionNotAuthorized, match="it is unresolved"):
        check_authority(store, action.action_id, host)
    worker = runner(store, connector, host, clock)
    for step in (worker.reconcile, worker.readback):
        assert step(action.run_id, action.action_id).next_action is None
    assert worker.dispatch_pending(action.run_id) == ()
    assert connector.calls == calls
    with pytest.raises(ActionStateError, match="only an effect that a read has shown"):
        prepare_compensation(store, action.action_id, operation="update")
    for decision in ACTION_DECISIONS:
        with pytest.raises(ActionStateError, match="is unresolved"):
            decide(store, host, action, decision, evidence())
    # Preparing the same intent returns it, and its slot stays closed to another intent.
    assert prepare_action(store, action.run_id, slot="publish", intent=action.intent,
                          profile=action.profile) == load_action(store, action.action_id)
    with pytest.raises(WorkflowConflictError):
        prepare_action(store, action.run_id, slot="publish",
                       intent=intent(delivered(store, action.run_id), title="Another title"), profile=action.profile)
    view = inspect_actions(store, action.run_id)
    assert view["unresolved"] == [action.action_id] and view["items"][0]["next_action"] is None


# ---------------------------------------------------------------------------
# Authority and refusals
# ---------------------------------------------------------------------------


def test_a_decision_needs_a_principal_the_verifier_vouches_for_exactly_this_decision(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, _, _ = under_review(store, tmp_path, replies, host, clock)
    attached, untouched = evidence(), journal(store)

    def refused(error, match, assertion, decision="not_executed", given=attached, verifier=host):
        with pytest.raises(error, match=match):
            resolve(store, action.action_id, decision, assertion=assertion, verifier=verifier, evidence=given,
                    note=NOTE)
        assert journal(store) == untouched

    forged = Host().issue(decision_sha256(store, action.action_id, "not_executed", attached), subject="operator-1",
                          tenant="acme", expires_at_ns=FAR)
    refused(ApprovalRejected, "signature is not this host's", forged)
    # The action's approval assertion binds its intent, never a decision.
    refused(ApprovalRejected, "another intent", sign(host, action))
    # An assertion for another decision, or other evidence, decides nothing else.
    refused(ApprovalRejected, "another intent", assertion_for(store, host, action, "abandon"))
    refused(ApprovalRejected, "another intent", assertion_for(store, host, action, "not_executed", evidence("other")))
    refused(ApprovalRejected, "tenant 'globex'",
            assertion_for(store, host, action, "not_executed", attached, tenant="globex"))
    host.authorized = False
    refused(ApprovalRejected, "no longer authorizes", assertion_for(store, host, action, "not_executed", attached))
    host.authorized = True
    host.revoke_subject("operator-1")
    refused(ApprovalRejected, "revoked", assertion_for(store, host, action, "not_executed", attached))
    refused(TypeError, "verify", assertion_for(store, host, action, "abandon"), "abandon", None, object())
    refused(WorkflowValidationError, "assertion", "", "abandon", None)
    refused(WorkflowValidationError, "decision must be one of", "x", "retry")
    refused(TypeError, "ActionEvidence", "x", "not_executed", attached.to_dict())
    with pytest.raises(WorkflowValidationError, match="note"):
        resolve(store, action.action_id, "abandon", assertion="x", verifier=host, note=" ")
    assert journal(store) == untouched


def test_an_action_that_changes_while_a_decision_is_verified_is_not_decided(
        store, tmp_path, replies, host, clock):  # noqa: F811
    class Racing(Host):
        race = None

        def verify(self, assertion, intent_sha256, now_ns):
            race, self.race = self.race, None
            if race is not None:
                race()
            return super().verify(assertion, intent_sha256, now_ns)

    racing = Racing()
    action, _, _ = under_review(store, tmp_path, replies, racing, clock)
    assertion = assertion_for(store, racing, action, "not_executed", evidence())
    racing.race = lambda: decide(store, racing, action, "abandon")

    with pytest.raises(WorkflowConflictError, match="changed while its decision was verified"):
        resolve(store, action.action_id, "not_executed", assertion=assertion, verifier=racing, evidence=evidence(),
                note=NOTE)
    assert load_action(store, action.action_id).state == "unresolved"
    assert [record["kind"] for record in item(store, action)["records"]].count("decision") == 1


def test_decisions_apply_only_to_open_outcomes_and_a_security_stop_refuses_them(
        store, tmp_path, replies, host, clock):  # noqa: F811
    connector = Recording(FixtureService(clock_ns=clock))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    for decision in ACTION_DECISIONS:
        with pytest.raises(ActionStateError, match=f"not_dispatched; no {decision} decision applies"):
            decide(store, host, action, decision, evidence())
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    # An acknowledged write is never found not executed.
    with pytest.raises(ActionStateError, match="dispatched; no not_executed decision applies"):
        decide(store, host, action, "not_executed", evidence())
    worker.readback(action.run_id, action.action_id)
    for decision in ACTION_DECISIONS:
        with pytest.raises(ActionStateError, match="observed"):
            decide(store, host, action, decision, evidence())

    stopped, _, _ = under_review(store, tmp_path, replies, host, clock)
    untouched = journal(store)
    store.security_stop(stopped.run_id, "approver_compromised")
    for decision in ACTION_DECISIONS:
        with pytest.raises(ActionNotAuthorized, match="a security stop holds the run"):
            decide(store, host, stopped, decision, evidence())
    assert journal(store) == untouched


def test_a_decision_in_flight_is_refused(store, tmp_path, replies, host, clock):  # noqa: F811
    connector = Recording(FixtureService(clock_ns=clock))
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    lease = store.acquire_lease(action.run_id, "worker-1", ttl_s=60)
    attempt = workflow_actions.begin_attempt(store, lease, action.action_id, approval.approval_id)
    workflow_actions.claim_dispatch(store, lease, attempt["attempt_id"])
    for decision in ("adopt_observed", "abandon"):
        with pytest.raises(ActionStateError, match="with an attempt in flight"):
            decide(store, host, action, decision, evidence())


# ---------------------------------------------------------------------------
# Evidence and inspection
# ---------------------------------------------------------------------------


def test_evidence_is_kept_as_a_digest_and_a_bounded_redacted_summary(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, _, _ = under_review(store, tmp_path, replies, host, clock)
    summary = f"Support case for {ORGANIZATION}: " + "no write found. " * 60
    attached = ActionEvidence.for_bytes(b"raw-body-not-for-the-journal", summary)
    assert attached.summary.startswith("Support case for org-[redacted]: ") and len(attached.summary) == 500
    assert ActionEvidence.from_dict(attached.to_dict()) == attached

    decide(store, host, action, "not_executed", attached, note=f"Approved by {ORGANIZATION} " + "x" * 600)

    kept = rows(store.path, "workflow_action_records")
    text = json.dumps(kept)
    assert ORGANIZATION not in text and "raw-body-not-for-the-journal" not in text
    (decision,) = [json.loads(row["payload_json"]) for row in kept if row["kind"] == "decision"]
    assert decision["note"].startswith("Approved by org-[redacted] ") and len(decision["note"]) == 500
    with pytest.raises(ValueError, match="64 lowercase"):
        ActionEvidence("not hex", "summary")
    with pytest.raises(ValueError, match="summary"):
        ActionEvidence("a" * 64, " ")
    with pytest.raises(TypeError, match="bytes"):
        ActionEvidence.for_bytes("text", "summary")
    assert ACTION_DECISIONS == ("adopt_observed", "not_executed", "abandon")


def test_inspection_shows_states_approvals_records_balances_and_next_actions(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action, approval, connector = under_review(store, tmp_path, replies, host, clock)
    run_id = action.run_id
    waiting = prepare_action(store, run_id, slot="announce",
                             intent=intent(delivered(store, run_id), title="Announcement"), profile=connector.profile)
    approved = prepare_action(store, run_id, slot="mirror",
                              intent=intent(delivered(store, run_id), title="Mirror"), profile=connector.profile)
    approve(store, approved.action_id, sign(host, approved), host, ceiling_nanousd=approved.ceiling_nanousd,
            expires_at_ns=clock.now_ns + HOUR_NS)
    revoked = prepare_action(store, run_id, slot="notify", intent=intent(delivered(store, run_id), title="Notice"),
                             profile=connector.profile)
    approve(store, revoked.action_id, sign(host, revoked), host, ceiling_nanousd=revoked.ceiling_nanousd,
            expires_at_ns=clock.now_ns + HOUR_NS)
    revoke(store, revoked.action_id, f"Access withdrawn from {ORGANIZATION}")
    with SQLiteWorkflowStore(store.path, read_only=True) as reader:
        view = inspect_actions(reader, run_id)

    assert view == {**store.inspect_run(run_id)["actions"], "unresolved": [action.action_id],
                    "items": view["items"]}
    assert (view["accounting_scope"], view["unit"]) == ("external_actions", "nanoUSD")
    steps = {entry["slot"]: (entry["state"], entry["duplicate_risk"], entry["next_action"]) for entry in view["items"]}
    assert steps == {"announce": ("not_dispatched", False, "approve"), "mirror": ("not_dispatched", False, "dispatch"),
                     "notify": ("not_dispatched", False, "approve"),
                     "publish": ("outcome_unknown", False, "operator_review")}
    published = next(entry for entry in view["items"] if entry["action_id"] == action.action_id)
    assert published == {**item(store, action), "duplicate_risk": False, "contradiction": None,
                         "next_action": "operator_review"}
    (shown,) = published["approvals"]
    assert (shown["approval_id"], shown["principal"]["subject"], shown["expires_at_ns"]) == (
        approval.approval_id, "editor-1", approval.expires_at_ns)
    assert [record["kind"] for record in published["records"]] == ["finding"]
    text = json.dumps(view)
    for secret in [row["assertion"] for row in rows(store.path, "workflow_action_approvals")] + [
            "Access withdrawn", "timed out", '"parameters"', "Announcement"]:
        assert secret not in text

    decide(store, host, action, "not_executed", evidence())
    after = inspect_actions(store, run_id)
    assert after["unresolved"] == []
    assert [entry["next_action"] for entry in after["items"] if entry["action_id"] == action.action_id] == [
        "approve_duplicate_risk"]
    assert waiting.action_id in [entry["action_id"] for entry in after["items"]]
