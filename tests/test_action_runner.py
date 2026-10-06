"""The action runner: one submission per attempt, readback of the requested state, untrusted notices.

``ActionRunner.dispatch`` holds the run's lease, checks the approval and the
content, reads the payload once, claims the dispatch and submits exactly once.
A receipt never shows the result: ``readback`` reads the committed object a
bounded number of times and records each verdict, so only an independent read
of the intended bytes, key and target makes an action observed. Notices the
host relays are kept as untrusted records and can only prompt a readback.
"""

from __future__ import annotations

import base64
from dataclasses import replace
import json
import time

import pytest

from fakes import ManualClock
from fixture_connector import FixtureConnector, stored_value
from fixture_service import FixtureService, Target
from smythe import SQLiteWorkflowStore, workflow_actions
from smythe.actions import (
    READBACK_VALIDATOR,
    ActionIntent,
    ActionNotAuthorized,
    ActionProgress,
    ActionRunner,
    ActionStateError,
    ActionTarget,
    approve,
    check_authority,
    prepare_action,
    record_notice,
)
from smythe.validation import ValidationOutcome, ValidationReport
from smythe.workflow_actions import record_receipt, record_refused
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import (
    WorkflowConflictError, WorkflowLeaseError, WorkflowValidationError, _canonical, _sha,
)
from test_action_approvals import HOUR_NS, ORGANIZATION, PROFILE, Host, completed, delivered, intent, sign
from test_action_journal import SETTINGS, WRITE, balances, document, execute, item, rows
from test_workflow_acceptance import replies  # noqa: F401

READ = PROFILE.read_ceiling_nanousd


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


class Recording(FixtureConnector):
    """A fixture connector that logs its calls; ``settings`` change the journal tests' profile."""

    def __init__(self, service: FixtureService, **settings) -> None:
        super().__init__(service, **{**SETTINGS, **settings})
        self.calls: list[tuple] = []

    def submit(self, key, intent, payload, timeout_s):
        self.calls.append(("submit", key, payload))
        return super().submit(key, intent, payload, timeout_s)

    def read(self, target, timeout_s):
        self.calls.append(("read", target))
        return super().read(target, timeout_s)

    def made(self, kind: str) -> int:
        return sum(call[0] == kind for call in self.calls)


@pytest.fixture
def connector(service) -> Recording:
    value = Recording(service)
    assert value.profile == PROFILE
    return value


def runner(store, connector, host, clock, **options) -> ActionRunner:
    """A runner whose sleeps move the manual clock."""
    return ActionRunner(store, connector, host, sleep=clock.advance, clock=clock.monotonic, **options)


def ready(store, tmp_path, replies, host, clock, connector, *, action_intent=None, **options):  # noqa: F811
    """A completed run's action prepared under *connector*'s profile and approved at its ceiling."""
    run_id = completed(store, tmp_path, replies, **options)
    action = prepare_action(store, run_id, slot="publish",
                            intent=action_intent or intent(delivered(store, run_id)),
                            profile=connector.profile)
    approval = approve(store, action.action_id, sign(host, action), host,
                       ceiling_nanousd=action.ceiling_nanousd, expires_at_ns=clock.now_ns + HOUR_NS)
    return action, approval


def records(store, action, kind) -> list[dict]:
    """The action's records of *kind*, with their payloads."""
    return [{**row, "payload": json.loads(row["payload_json"])}
            for row in rows(store.path, "workflow_action_records")
            if row["action_id"] == action.action_id and row["kind"] == kind]


def report(record) -> ValidationReport:
    return ValidationReport.from_dict(record["payload"]["report"])


def codes(value: ValidationReport) -> list[str]:
    return [finding.code for finding in value.findings]


def relay(notice: dict) -> dict:
    """A service notice as host code relays it: the key and target it reports, and the notice."""
    payload = notice["payload"]
    return {"key": payload.get("request_key"), "target": payload.get("target"), "notice": notice}


# ---------------------------------------------------------------------------
# Dispatch and readback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("renderer", ["published", "none"])
def test_one_submission_then_an_independent_read_observes_the_intended_state(
        store, tmp_path, replies, host, clock, service, connector, renderer):  # noqa: F811
    options = {} if renderer == "published" else {"renderer": None}
    action, _ = ready(store, tmp_path, replies, host, clock, connector, **options)
    sent = document(store, action)
    worker = runner(store, connector, host, clock)

    dispatched = worker.dispatch(action.run_id, action.action_id)

    # A receipt alone never shows the result: the action waits for its readback.
    assert dispatched == ActionProgress(action.action_id, "dispatched", dispatched.attempt, None, "readback")
    assert (dispatched.attempt["number"], dispatched.attempt["state"]) == (1, "acknowledged")
    assert connector.calls == [("submit", action.action_key, sent)]
    current = item(store, action)
    assert current["approvals"][0]["consumed_by_attempt"] == dispatched.attempt["attempt_id"]
    # The lease is released after each step.
    store.release_lease(store.acquire_lease(action.run_id, "another-worker", ttl_s=60))

    observed = worker.readback(action.run_id, action.action_id)

    assert (observed.state, observed.attempt, observed.next_action) == (
        "observed", dispatched.attempt, None)
    assert observed.report.passed and observed.report.validator == READBACK_VALIDATOR
    assert observed.report.validator.to_dict()["name"] == "smythe.action-readback"
    assert connector.made("read") == 1
    (read,) = records(store, action, "readback")
    (object_id,) = service.snapshot()["objects"]
    assert read["payload"] == {
        "attempt_id": dispatched.attempt["attempt_id"], "read": 1, "error": None,
        "observed": {"target": {"tenant": "acme", "collection": "posts", "object_id": object_id},
                     "content": action.intent.content.to_dict(), "version": 1,
                     "key": action.action_key},
        "report": observed.report.to_dict()}
    assert observed.report.path == f"acme/posts/{object_id}"
    assert (read["trusted"], read["cost"]) == (1, str(READ))
    assert report(read) == observed.report
    current = item(store, action)
    # The read is charged to the action; the uncosted write keeps its exposure.
    assert (current["state"], balances(current)) == ("observed", (READ, 0, WRITE))
    with pytest.raises(ActionNotAuthorized, match="it is observed"):
        check_authority(store, action.action_id, host)
    snapshot = service.snapshot()
    assert (snapshot["totals"]["received"], snapshot["totals"]["committed"]) == (1, 1)
    (stored,) = snapshot["objects"].values()
    assert base64.b64decode(stored["value"]["body_base64"]) == sent
    events = [(event["type"], event["data"].get("state")) for event in store.inspect_run(action.run_id)["events"]
              if event["type"].startswith("action_")]
    assert events == [("action_attempt", "prepared"), ("action_attempt", "dispatched"),
                      ("action_attempt", "acknowledged"), ("action_readback", "observed")]
    # Inspection shows records by digest only, never the delivered bytes.
    assert sent.decode("utf-8") not in json.dumps(store.inspect_run(action.run_id))
    assert store.audit(action.run_id)["ok"]
    # Reading back an observed action reads nothing more.
    assert worker.readback(action.run_id, action.action_id) == replace(observed, report=None)
    assert connector.made("read") == 1


def test_a_duplicated_network_delivery_yields_one_effect(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    service.deliver_duplicate(action.action_key)
    worker = runner(store, connector, host, clock)

    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).state == "observed"

    assert connector.made("submit") == 1
    counts = service.snapshot()["requests"][action.action_key]
    assert (counts["received"], counts["committed"], counts["deduplicated"]) == (2, 1, 1)
    assert service.snapshot()["totals"]["committed"] == 1


def test_another_writer_changing_the_object_before_readback_requires_compensation(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    dispatched = worker.dispatch(action.run_id, action.action_id)
    (object_id,) = service.snapshot()["objects"]
    service.submit("another-writer", {"title": "Edited elsewhere"}, object_id=object_id,
                   target=Target("acme", "posts"))

    progress = worker.readback(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("compensation_required", "compensate")
    assert progress.report.outcome is ValidationOutcome.REJECTED
    assert codes(progress.report) == ["content_mismatch", "key_mismatch"]
    assert progress.attempt == dispatched.attempt
    # The first contradicting read settles it; no further read is made.
    assert connector.made("read") == 1
    (read,) = records(store, action, "readback")
    assert read["payload"]["observed"]["version"] == 2
    with pytest.raises(ActionNotAuthorized, match="it is compensation_required"):
        check_authority(store, action.action_id, host)
    assert service.snapshot()["requests"][action.action_key]["committed"] == 1
    assert store.audit(action.run_id)["ok"]


class Misrouted(Recording):
    """A connector whose reads go to an object of another tenant."""

    def __init__(self, service, target: ActionTarget, **settings) -> None:
        super().__init__(service, **settings)
        self.misrouted_to = target

    def read(self, target, timeout_s):
        return super().read(self.misrouted_to, timeout_s)


def test_an_object_visible_under_another_tenant_is_not_observed(
        store, tmp_path, replies, host, clock, service):  # noqa: F811
    connector = Misrouted(service, ActionTarget("globex", "posts", "obj_1"))
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    # Another tenant holds identical bytes under the same key: the service keeps them apart.
    service.submit(action.action_key, stored_value(action.intent, document(store, action)),
                   target=Target("globex", "posts"))
    worker = runner(store, connector, host, clock)

    worker.dispatch(action.run_id, action.action_id)
    progress = worker.readback(action.run_id, action.action_id)

    assert progress.state != "observed"
    assert (progress.state, codes(progress.report)) == ("compensation_required", ["target_mismatch"])
    finding = progress.report.findings[0]
    assert (finding.expected["tenant"], finding.observed["tenant"]) == ("acme", "globex")
    assert item(store, action)["attempts"][0]["state"] == "acknowledged"
    # The service itself never shows the action's object to the other tenant.
    assert FixtureConnector(service, **SETTINGS).read(ActionTarget("globex", "posts", "obj_2"), 30) is None


def test_a_commit_invisible_within_the_read_bound_stays_unverified_and_is_never_resent(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=60)
    connector = Recording(service, readback_interval_s=1)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    started = clock.monotonic()

    progress = worker.readback(action.run_id, action.action_id)

    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "dispatched", "acknowledged", "reconcile")
    assert progress.report.outcome is ValidationOutcome.ERROR and not progress.report.passed
    assert codes(progress.report) == ["not_visible"]
    # Two reads, one interval apart; neither proves anything.
    assert [read["payload"]["read"] for read in records(store, action, "readback")] == [1, 2]
    assert clock.monotonic() - started == pytest.approx(1)
    assert balances(item(store, action)) == (2 * READ, 0, WRITE)

    # The bound holds across calls, even once the commit becomes visible.
    clock.advance(120)
    again = worker.readback(action.run_id, action.action_id)
    assert (again.state, again.report, again.next_action) == ("dispatched", None, "reconcile")
    assert connector.made("read") == 2 and len(records(store, action, "readback")) == 2
    # Nothing is sent again: the receipt consumed the approval.
    with pytest.raises(ActionNotAuthorized, match="its approval was consumed"):
        worker.dispatch(action.run_id, action.action_id)
    assert connector.made("submit") == 1
    snapshot = service.snapshot()
    assert (len(snapshot["deliveries"]), snapshot["totals"]["committed"]) == (1, 1)
    assert store.audit(action.run_id)["ok"]


def test_a_stale_read_decides_nothing_until_the_requested_version_is_visible(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=5)
    service.submit("seed", {"title": "Draft"}, target=Target("acme", "posts"))
    clock.advance(5)
    connector = Recording(service, readback_interval_s=5)
    run_id = completed(store, tmp_path, replies)
    update = ActionIntent("update", ActionTarget("acme", "posts", "obj_1"),
                          parameters={"title": "Launch notes"}, content=delivered(store, run_id),
                          connector=PROFILE.name, connector_version=PROFILE.version)
    action = prepare_action(store, run_id, slot="update", intent=update, profile=connector.profile)
    approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=action.ceiling_nanousd,
            expires_at_ns=clock.now_ns + HOUR_NS)
    worker = runner(store, connector, host, clock)

    worker.dispatch(run_id, action.action_id)
    progress = worker.readback(run_id, action.action_id)

    assert progress.state == "observed"
    first, second = records(store, action, "readback")
    # The first read still saw the version before the update: not a contradiction.
    assert (report(first).outcome, codes(report(first))) == (ValidationOutcome.ERROR, ["stale_version"])
    assert (report(first).findings[0].expected, report(first).findings[0].observed) == (2, 1)
    assert report(second).passed and second["payload"]["observed"]["version"] == 2


@pytest.mark.parametrize("reads,interval,delay", [
    (3, 4, 7),  # the reads take eight seconds in all, longer than the lease
    (2, 10, 5),  # one wait between reads is longer than the lease
], ids=["reads-outlast-the-lease", "a-wait-outlasts-the-lease"])
def test_readback_renews_the_lease_between_reads(
        store, tmp_path, replies, host, clock, reads, interval, delay):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=delay)
    connector = Recording(service, readback_reads=reads, readback_interval_s=interval)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock, lease_ttl_s=5)
    worker.dispatch(action.run_id, action.action_id)
    started = clock.monotonic()

    progress = worker.readback(action.run_id, action.action_id)

    assert progress.state == "observed"
    assert [report(read).outcome for read in records(store, action, "readback")] == [
        *[ValidationOutcome.ERROR] * (reads - 1), ValidationOutcome.PASSED]
    assert clock.monotonic() - started == pytest.approx((reads - 1) * interval)
    # The step's lease is released when it ends.
    store.release_lease(store.acquire_lease(action.run_id, "another-worker", ttl_s=60))


def test_a_failed_read_proves_nothing_and_its_error_is_kept_redacted(
        store, tmp_path, replies, host, clock, service):  # noqa: F811
    class Failing(Recording):
        def read(self, target, timeout_s):
            raise ConnectionError(f"SECRET-READ for {ORGANIZATION}")

    connector = Failing(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)

    progress = worker.readback(action.run_id, action.action_id)

    assert (progress.state, progress.next_action, codes(progress.report)) == (
        "dispatched", "reconcile", ["read_failed"])
    errors = [read["payload"]["error"] for read in records(store, action, "readback")]
    assert errors == ["ConnectionError: SECRET-READ for org-[redacted]"] * 2
    assert ORGANIZATION not in json.dumps([read["payload"] for read in records(store, action, "readback")])
    assert "SECRET" not in json.dumps(store.inspect_run(action.run_id))


def test_a_connector_without_reads_leaves_the_result_to_reconciliation(
        store, tmp_path, replies, host, clock):  # noqa: F811
    connector = Recording(FixtureService(clock_ns=clock, lookup_by_object=False))
    assert not connector.profile.lookup_by_object
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)

    assert worker.dispatch(action.run_id, action.action_id).next_action == "reconcile"
    progress = worker.readback(action.run_id, action.action_id)

    assert (progress.state, progress.report, progress.next_action) == ("dispatched", None, "reconcile")
    assert connector.made("read") == 0 and records(store, action, "readback") == []


# ---------------------------------------------------------------------------
# Refusals, ambiguity and recovery
# ---------------------------------------------------------------------------


def test_a_definite_refusal_leaves_the_action_dispatchable_under_the_same_approval(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval = ready(store, tmp_path, replies, host, clock, connector)
    service.fail_before_commit(action.action_key)
    worker = runner(store, connector, host, clock)

    refused = worker.dispatch(action.run_id, action.action_id)

    assert (refused.state, refused.attempt["state"], refused.next_action) == (
        "not_dispatched", "refused", "dispatch")
    current = item(store, action)
    assert current["approvals"][0]["consumed_by_attempt"] is None and balances(current) == (0, 0, 0)
    assert check_authority(store, action.action_id, host) == approval

    retried = worker.dispatch(action.run_id, action.action_id)

    assert (retried.attempt["number"], retried.attempt["state"]) == (2, "acknowledged")
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    assert [call[1] for call in connector.calls if call[0] == "submit"] == [action.action_key] * 2
    snapshot = service.snapshot()
    assert (snapshot["totals"]["refused"], snapshot["totals"]["committed"]) == (1, 1)


def test_refusals_that_use_every_attempt_leave_nothing_to_dispatch(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    for delivery in range(PROFILE.max_attempts):
        service.fail_before_commit(action.action_key, delivery=delivery)
        progress = worker.dispatch(action.run_id, action.action_id)

    assert (progress.state, progress.next_action) == ("not_dispatched", None)
    with pytest.raises(ActionStateError, match="has made all 2 of its attempts"):
        worker.dispatch(action.run_id, action.action_id)
    assert connector.made("submit") == 2


def test_an_ambiguous_submission_is_recorded_and_never_resent(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    service.commit_then_drop_ack(action.action_key)
    worker = runner(store, connector, host, clock)

    unknown = worker.dispatch(action.run_id, action.action_id)

    assert (unknown.state, unknown.attempt["state"], unknown.next_action) == (
        "outcome_unknown", "outcome_unknown", "reconcile")
    assert balances(item(store, action)) == (0, 0, WRITE)
    readback = worker.readback(action.run_id, action.action_id)
    assert (readback.state, readback.report) == ("outcome_unknown", None)
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        worker.dispatch(action.run_id, action.action_id)
    assert (connector.made("submit"), connector.made("read")) == (1, 0)
    assert service.snapshot()["totals"]["committed"] == 1


class Answering(Recording):
    """A connector whose submissions commit but answer with ``answer(receipt)``."""

    def __init__(self, service, answer) -> None:
        super().__init__(service)
        self.answer = answer

    def submit(self, key, intent, payload, timeout_s):
        return self.answer(super().submit(key, intent, payload, timeout_s))


@pytest.mark.parametrize("answer", [
    lambda receipt: replace(receipt, target=ActionTarget("globex", "posts", receipt.target.object_id)),
    lambda receipt: replace(receipt, key="another-key"),
    lambda receipt: receipt.to_dict(),
], ids=["another-tenant", "another-key", "not-a-receipt"])
def test_an_answer_that_is_not_the_actions_receipt_is_recorded_as_ambiguous(
        store, tmp_path, replies, host, clock, service, answer):  # noqa: F811
    connector = Answering(service, answer)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)

    progress = runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)

    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "outcome_unknown", "outcome_unknown", "reconcile")
    (row,) = rows(store.path, "workflow_action_attempts")
    assert row["receipt_json"] is None and "WorkflowValidationError" in row["error"]
    assert store.audit(action.run_id)["ok"]


def test_an_interrupted_submission_is_recovered_as_unknown_and_never_resent(
        store, tmp_path, replies, host, clock, service):  # noqa: F811
    class Stopped(BaseException):
        """The worker stops after the service commits, before it records the answer."""

    class Interrupted(Recording):
        def submit(self, key, intent, payload, timeout_s):
            super().submit(key, intent, payload, timeout_s)
            raise Stopped

    connector = Interrupted(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    with pytest.raises(Stopped):
        worker.dispatch(action.run_id, action.action_id)
    assert item(store, action)["attempts"][0]["state"] == "dispatched"

    # The interrupted step released its lease, so the next step recovers the attempt.
    progress = worker.readback(action.run_id, action.action_id)

    assert (progress.state, progress.attempt["state"], progress.next_action) == (
        "outcome_unknown", "outcome_unknown", "reconcile")
    assert rows(store.path, "workflow_action_attempts")[0]["error"] == "recovered after dispatch"
    assert (connector.made("submit"), service.snapshot()["totals"]["committed"]) == (1, 1)


def test_a_stopped_workers_lease_fences_the_runner_until_it_lapses(
        store, tmp_path, replies, host, clock, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    store.acquire_lease(action.run_id, "stopped-worker", ttl_s=60)
    worker = runner(store, connector, host, clock)

    with pytest.raises(WorkflowLeaseError, match="active lease"):
        worker.dispatch(action.run_id, action.action_id)
    assert connector.calls == []

    clock.advance(61)
    assert worker.dispatch(action.run_id, action.action_id).state == "dispatched"


# ---------------------------------------------------------------------------
# The payload is read once, before the claim
# ---------------------------------------------------------------------------


def _edit_before_dispatch(store, action, monkeypatch):
    current_artifact(store, action.run_id).publication.path.write_bytes(b"# Edited after approval\n")
    return r"not eligible for delivery \(artifact_changed\)"


def _edit_after_the_attempt_begins(store, action, monkeypatch):
    begin = workflow_actions.begin_attempt

    def begin_then_edit(*args, **kwargs):
        attempt = begin(*args, **kwargs)
        current_artifact(store, action.run_id).publication.path.write_bytes(b"# Edited after the check\n")
        return attempt

    monkeypatch.setattr(workflow_actions, "begin_attempt", begin_then_edit)
    return "the bytes read for delivery are not its bound content"


@pytest.mark.parametrize("edit", [_edit_before_dispatch, _edit_after_the_attempt_begins],
                         ids=["before-dispatch", "after-the-attempt-begins"])
def test_payload_bytes_edited_after_approval_are_refused_before_the_claim(
        store, tmp_path, replies, host, clock, service, connector, monkeypatch, edit):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    message = edit(store, action, monkeypatch)

    with pytest.raises(ActionNotAuthorized, match=message):
        runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)

    assert connector.calls == [] and service.snapshot()["deliveries"] == []
    current = item(store, action)
    assert (current["state"], balances(current)) == ("not_dispatched", (0, 0, 0))
    # An attempt begun before the bytes changed is released, unclaimed.
    assert [attempt["state"] for attempt in current["attempts"]] in ([], ["released"])
    assert all(row["dispatched"] is None for row in rows(store.path, "workflow_action_attempts"))
    assert store.audit(action.run_id)["ok"]


def test_the_bytes_checked_before_the_claim_are_the_bytes_sent(
        store, tmp_path, replies, host, clock, service, connector, monkeypatch):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    sent = document(store, action)
    claim = workflow_actions.claim_dispatch

    def claim_then_edit(*args, **kwargs):
        permit = claim(*args, **kwargs)
        current_artifact(store, action.run_id).publication.path.write_bytes(b"# Edited after the claim\n")
        return permit

    monkeypatch.setattr(workflow_actions, "claim_dispatch", claim_then_edit)
    worker = runner(store, connector, host, clock)

    worker.dispatch(action.run_id, action.action_id)

    # The file was not opened again: the service received the bytes that were checked.
    assert connector.calls == [("submit", action.action_key, sent)]
    assert worker.readback(action.run_id, action.action_id).state == "observed"


# ---------------------------------------------------------------------------
# Authority and identity
# ---------------------------------------------------------------------------


def test_a_security_stop_refuses_the_dispatch(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    execute(store.path, "UPDATE workflow_runs SET blocked_reason='security_stop' WHERE run_id=?",
            (action.run_id,))

    with pytest.raises(ActionNotAuthorized, match="a security stop holds the run"):
        runner(store, connector, host, clock).dispatch(action.run_id, action.action_id)
    assert connector.calls == [] and rows(store.path, "workflow_action_attempts") == []


def test_a_connector_with_another_profile_is_refused(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    renamed = Recording(service)
    renamed.profile = replace(connector.profile, version="2")
    others = [Recording(service, readback_reads=3), renamed]

    for other in others:
        worker = runner(store, other, host, clock)
        for step in (worker.dispatch, worker.readback):
            with pytest.raises(WorkflowConflictError, match="not the one action"):
                step(action.run_id, action.action_id)
        assert other.calls == []
    with pytest.raises(WorkflowConflictError, match="another workflow run"):
        runner(store, connector, host, clock).dispatch("another-run", action.action_id)
    assert rows(store.path, "workflow_action_attempts") == []


def test_a_runner_refuses_a_connector_that_retries_and_async_host_code(store, service, host):
    retrying = Recording(service)
    retrying.max_retries = 2
    with pytest.raises(TypeError, match="must not retry"):
        ActionRunner(store, retrying, host)

    class AsyncSubmit(Recording):
        async def submit(self, key, intent, payload, timeout_s):
            raise AssertionError("never awaited")

    with pytest.raises(TypeError, match=r"submit\(\) must be synchronous"):
        ActionRunner(store, AsyncSubmit(service), host)
    with pytest.raises(TypeError, match=r"has no verify\(\) method"):
        ActionRunner(store, Recording(service), object())
    with pytest.raises(WorkflowValidationError, match="Lease duration"):
        ActionRunner(store, Recording(service), host, lease_ttl_s=0)
    explicit = Recording(service)
    explicit.max_retries = 0
    assert ActionRunner(store, explicit, host, owner_id="worker-1").owner_id == "worker-1"


# ---------------------------------------------------------------------------
# Untrusted notices
# ---------------------------------------------------------------------------


def test_forged_notices_never_complete_an_action(store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, visibility_delay_s=60, send_notices=True)
    connector = Recording(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    acme = {"tenant": "acme", "collection": "posts"}
    worker = runner(store, connector, host, clock)

    # A notice under an unknown key names no action and is not kept.
    assert record_notice(store, {"key": "f" * 64, "target": acme}) is None
    # A matching key before dispatch is kept, untrusted, and prompts nothing.
    forged = relay(service.forge_notice({"request_key": action.action_key, "target": acme,
                                         "object_id": "obj_1", "object_version": 1}))
    assert record_notice(store, forged) is None
    assert item(store, action)["state"] == "not_dispatched"
    assert worker.readback(action.run_id, action.action_id).state == "not_dispatched"

    assert worker.dispatch(action.run_id, action.action_id).next_action == "readback"

    # Another tenant's notice for the same key prompts nothing.
    assert record_notice(store, {"key": action.action_key,
                                 "target": {"tenant": "globex", "collection": "posts"}}) is None
    assert record_notice(store, {"key": action.action_key}) is None
    # A notice that matches only prompts a readback, which reads the receipt's object itself.
    (genuine,) = [notice for notice in service.notices() if notice["notice_id"] != forged["notice"]["notice_id"]]
    for notice in (forged, relay(genuine)):
        prompted = record_notice(store, notice)
        assert (prompted.action_id, prompted.run_id, prompted.state) == (
            action.action_id, action.run_id, "dispatched")
        assert item(store, action)["state"] == "dispatched"
    progress = worker.readback(prompted.run_id, prompted.action_id)

    # The commit is not visible yet, so no notice made it observed.
    assert (progress.state, progress.next_action) == ("dispatched", "reconcile")
    kept = records(store, action, "notice")
    assert len(kept) == 5 and {(row["trusted"], row["cost"]) for row in kept} == {(0, None)}
    assert [row["payload"] for row in kept][:1] == [forged]
    shown = item(store, action)["records"]
    assert [record["kind"] for record in shown].count("notice") == 5
    assert all(not record["trusted"] for record in shown if record["kind"] == "notice")
    events = [event["data"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_notice"]
    assert [event["record_id"] for event in events] == [row["record_id"] for row in kept]
    assert service.snapshot()["totals"]["committed"] == 1
    assert store.audit(action.run_id)["ok"]


def test_a_notice_after_the_result_is_observed_changes_nothing(
        store, tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock, send_notices=True)
    connector = Recording(service)
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    worker = runner(store, connector, host, clock)
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).state == "observed"

    (notice,) = service.notices()
    assert record_notice(store, relay(notice)) is None

    assert item(store, action)["state"] == "observed"
    assert len(records(store, action, "notice")) == 1


@pytest.mark.parametrize("payload,message", [
    (["not", "an", "object"], "JSON object"),
    ({"key": "k", "value": float("nan")}, "finite"),
    ({"key": "k", "padding": "x" * 70_000}, "at most 65536 bytes"),
])
def test_a_malformed_notice_is_refused_and_kept_nowhere(
        store, tmp_path, replies, host, clock, connector, payload, message):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    if isinstance(payload, dict):
        payload["key"] = action.action_key

    with pytest.raises(WorkflowValidationError, match=message):
        record_notice(store, payload)
    assert records(store, action, "notice") == []


# ---------------------------------------------------------------------------
# Events out of order
# ---------------------------------------------------------------------------


def test_a_late_receipt_after_a_newer_attempt_is_kept_and_the_state_does_not_regress(
        store, tmp_path, replies, host, clock, service, connector, monkeypatch):  # noqa: F811
    action, _ = ready(store, tmp_path, replies, host, clock, connector)
    permits = []
    claim = workflow_actions.claim_dispatch
    monkeypatch.setattr(workflow_actions, "claim_dispatch",
                        lambda *args: permits.append(claim(*args)) or permits[-1])
    service.fail_before_commit(action.action_key)
    worker = runner(store, connector, host, clock)
    assert worker.dispatch(action.run_id, action.action_id).attempt["state"] == "refused"
    worker.dispatch(action.run_id, action.action_id)
    assert worker.readback(action.run_id, action.action_id).state == "observed"
    before = item(store, action)
    receipt = connector.lookup(action.action_key, action.intent.target, 30)

    # The first attempt's answer arrives after the second attempt's result was observed. Each
    # runner step held its own lease session, so both permits are from superseded sessions.
    with pytest.raises(WorkflowLeaseError, match="since superseded; its outcome is kept"):
        record_receipt(store, permits[0], receipt)
    with pytest.raises(WorkflowLeaseError, match="since superseded; its outcome is kept"):
        record_refused(store, permits[1], "refused late")

    after = item(store, action)
    assert (after["state"], after["attempts"], balances(after)) == (
        "observed", before["attempts"], balances(before))
    (first, second) = records(store, action, "late_receipt")
    assert (first["payload"]["attempt_id"], first["payload"]["outcome"]) == (
        permits[0].attempt_id, "acknowledged")
    assert first["payload"]["receipt"] == receipt.to_dict() and first["trusted"] == 1
    assert second["payload"]["outcome"] == "refused"
    assert _sha(_canonical(first["payload"])) == first["payload_sha"]
    assert store.audit(action.run_id)["ok"]
