"""Prepared external actions bound to accepted content, and host-verified approvals.

A completed durable run whose current delivery decision is ``accepted`` can
prepare an action that delivers exactly its accepted output. The host's
verifier authenticates each approval for one action's intent digest, and
``check_authority`` checks the approval and the content again when asked.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import sqlite3
import time

import pytest

from fakes import ManualClock
from fixture_connector import FixtureConnector, HmacApprovalHost
from fixture_service import FixtureService
from smythe import (
    MarkdownRenderer, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task,
)
from smythe._canonical import canonical_bytes
from smythe.actions import (
    APPROVAL_SUBJECT_VERSION,
    ActionApproval,
    ActionIntent,
    ActionNotAuthorized,
    ActionStateError,
    ActionRunner,
    ActionTarget,
    ApprovalRejected,
    ContentRef,
    PreparedAction,
    Principal,
    action_key,
    approve,
    check_authority,
    load_action,
    prepare_action,
    reject,
    revoke,
)
from smythe.cli import main
import smythe.workflow
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import (
    WorkflowConflictError, WorkflowNotFoundError, WorkflowStateError, WorkflowValidationError,
    _canonical, _sha,
)
from test_acceptance_operator_decision import durable_operator
from test_source_evidence import KEPT, artifact
from test_source_tombstones import durable as sourced, remove
from test_workflow_acceptance import Contains, Raises, replies, single, strict  # noqa: F401

HOUR_NS = 3600 * 10**9
FAR = 2**62
PROFILE = FixtureConnector(FixtureService(), write_ceiling_nanousd=2_000_000,
                           read_ceiling_nanousd=10_000, max_attempts=2, readback_reads=2).profile
# Two writes and two readback reads at the profile's ceilings.
CEILING = 4_020_000
ORGANIZATION = "org-" + "A1b2" * 6  # built here, so no tracked file holds an identifier


class Host(HmacApprovalHost):
    """The test approval host, recording each digest it verifies; ``authorized`` can withdraw it."""

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.verified: list[str] = []
        self.authorized = True

    def verify(self, assertion, subject_sha256, now_ns):
        self.verified.append(subject_sha256)
        return super().verify(assertion, subject_sha256, now_ns)

    def still_authorized(self, principal, intent, now_ns):
        return self.authorized and super().still_authorized(principal, intent, now_ns)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(time.time_ns())


@pytest.fixture
def journal(tmp_path):
    return tmp_path / "runs.db"


@pytest.fixture
def store(journal, clock):
    with SQLiteWorkflowStore(journal, clock_ns=clock) as value:
        yield value


@pytest.fixture
def host() -> Host:
    return Host()


def swarm(store, tmp_path, **options) -> Swarm:
    options.setdefault("acceptance", strict(Contains("report")))
    options.setdefault("renderer", MarkdownRenderer())
    options.setdefault("artifact_dir", tmp_path / "artifacts")
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def completed(store, tmp_path, replies, **options) -> str:  # noqa: F811
    """A completed run whose output is "the report"; returns its ID."""
    replies[0]["Write"] = ["the report"]
    return swarm(store, tmp_path, **options).execute(single()).execution_id


def delivered(store, run_id) -> ContentRef:
    """The run's output as an action delivers it: its published document, if it has one."""
    current = current_artifact(store, run_id)
    if current is not None:
        return ContentRef.for_bytes(current.publication.path.read_bytes(), "text/markdown")
    output = store.get_checkpoint(run_id)["checkpoint"]["output"]
    return ContentRef.for_bytes(output.encode("utf-8"), "text/plain")


def intent(content, *, title="Launch notes", tenant="acme") -> ActionIntent:
    return ActionIntent("publish", ActionTarget(tenant, "posts"), parameters={"title": title},
                        content=content, connector=PROFILE.name,
                        connector_version=PROFILE.version)


def prepare(store, run_id, slot="publish", **changes) -> PreparedAction:
    return prepare_action(store, run_id, slot=slot, intent=intent(delivered(store, run_id), **changes),
                          profile=PROFILE)


def sign(host, action, *, subject="editor-1", tenant="acme", expires_at_ns=FAR) -> str:
    """An assertion that *subject* approves *action*: its approval subject, the action's key and intent."""
    return host.issue(action.approval_sha256, subject=subject, tenant=tenant, expires_at_ns=expires_at_ns)


def approved(store, host, action, clock, assertion=None, *, hours=1) -> ActionApproval:
    return approve(store, action.action_id, assertion or sign(host, action), host,
                   ceiling_nanousd=CEILING, expires_at_ns=clock.now_ns + hours * HOUR_NS)


def actions_block(store, run_id) -> dict | None:
    return store.inspect_run(run_id).get("actions")


def approval_rows(journal) -> list[dict]:
    with closing(sqlite3.connect(journal)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM workflow_action_approvals ORDER BY rowid")]


# ---------------------------------------------------------------------------
# Preparing an action
# ---------------------------------------------------------------------------


def test_an_accepted_run_prepares_an_action_bound_to_its_published_document(
        store, tmp_path, replies):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    before = store.inspect_run(run_id)
    content = delivered(store, run_id)
    assert content.sha256 == current_artifact(store, run_id).publication.sha256
    # Another worker's live lease does not stop preparation.
    store.acquire_lease(run_id, "another-worker", ttl_s=60)

    action = prepare_action(store, run_id, slot="publish", intent=intent(content), profile=PROFILE)

    assert action == PreparedAction(
        action.action_id, run_id, "publish", 1, action_key(store.store_id, run_id, "publish", 1),
        intent(content), PROFILE, "not_dispatched", CEILING)
    assert load_action(store, action.action_id) == action
    inspected = store.inspect_run(run_id)
    (item,) = inspected.pop("actions")["items"]
    assert (item["action_id"], item["state"], item["ceiling_nanousd"], item["approvals"]) == (
        action.action_id, "not_dispatched", CEILING, [])
    assert item["intent_sha256"] == action.intent.digest()
    # Action money is its own scope: the run's calls and balances are as they were.
    for key in ("calls", "confirmed_nanousd", "reserved_nanousd", "unknown_nanousd", "status"):
        assert inspected[key] == before[key]
    assert store.audit(run_id)["ok"]


def test_a_run_without_a_renderer_binds_its_output(store, tmp_path, replies):  # noqa: F811
    run_id = completed(store, tmp_path, replies, renderer=None)
    assert current_artifact(store, run_id) is None
    content = delivered(store, run_id)
    assert content.size_bytes == len(b"the report")

    action = prepare_action(store, run_id, slot="publish", intent=intent(content), profile=PROFILE)
    assert action.intent.content == content and action.state == "not_dispatched"

    # The digest alone is not enough: the size must be the output's too.
    misstated = ContentRef(content.sha256, content.size_bytes + 1, "text/plain")
    with pytest.raises(ActionNotAuthorized, match="not the run's accepted output"):
        prepare_action(store, run_id, slot="other", intent=intent(misstated), profile=PROFILE)


REFUSED = {
    "advisory": (None, "has no recorded delivery acceptance"),
    "rejected": (strict(Contains("absent")), "delivery acceptance is rejected"),
    "needs-review": (strict(Raises(), on_invalid="needs_review"),
                     "delivery acceptance is needs_review"),
    "indeterminate": (strict(Raises()), "delivery acceptance is indeterminate"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_only_an_accepted_run_can_prepare_actions(store, tmp_path, replies, case):  # noqa: F811
    policy, reason = REFUSED[case]
    run_id = completed(store, tmp_path, replies, acceptance=policy)
    content = delivered(store, run_id)

    with pytest.raises(ActionNotAuthorized, match=reason):
        prepare_action(store, run_id, slot="publish", intent=intent(content), profile=PROFILE)
    assert actions_block(store, run_id) is None


def test_content_other_than_the_accepted_output_is_refused(store, tmp_path, replies):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    other = ContentRef.for_bytes(b"# A different document\n", "text/markdown")
    with pytest.raises(ActionNotAuthorized, match="not the run's accepted output"):
        prepare_action(store, run_id, slot="publish", intent=intent(other), profile=PROFILE)

    content = delivered(store, run_id)
    without = ActionIntent("publish", ActionTarget("acme", "posts"), connector=PROFILE.name,
                           connector_version=PROFILE.version)
    with pytest.raises(WorkflowValidationError, match="names no content"):
        prepare_action(store, run_id, slot="publish", intent=without, profile=PROFILE)
    elsewhere = ActionIntent("publish", ActionTarget("acme", "posts"), content=content,
                             connector="another-service", connector_version="1")
    with pytest.raises(WorkflowValidationError, match="profile's connector"):
        prepare_action(store, run_id, slot="publish", intent=elsewhere, profile=PROFILE)
    with pytest.raises(WorkflowValidationError, match="slot"):
        prepare_action(store, run_id, slot="", intent=intent(content), profile=PROFILE)
    with pytest.raises(TypeError, match="ConnectorProfile"):
        prepare_action(store, run_id, slot="publish", intent=intent(content),
                       profile=PROFILE.to_dict())
    with pytest.raises(WorkflowNotFoundError):
        prepare_action(store, "no-such-run", slot="publish", intent=intent(content), profile=PROFILE)
    assert actions_block(store, run_id) is None


def test_an_unfinished_run_cannot_prepare_actions(store, tmp_path, replies):  # noqa: F811
    planned = swarm(store, tmp_path).plan(Task("Write the brief")).run_ref["run_id"]
    content = ContentRef.for_bytes(b"the report", "text/plain")
    with pytest.raises(ActionNotAuthorized, match="has not completed"):
        prepare_action(store, planned, slot="publish", intent=intent(content), profile=PROFILE)
    assert actions_block(store, planned) is None


def test_preparing_again_is_idempotent_and_a_changed_intent_conflicts(
        store, tmp_path, replies, host):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    first = prepare(store, run_id)
    assert prepare(store, run_id) == first

    # A changed intent or profile in an open slot is a different action, never a retry.
    with pytest.raises(WorkflowConflictError, match="another intent or profile"):
        prepare(store, run_id, title="Edited title")
    with pytest.raises(WorkflowConflictError, match="another intent or profile"):
        prepare_action(store, run_id, slot="publish", intent=first.intent,
                       profile=FixtureConnector(FixtureService(), max_attempts=3).profile)
    # Another slot holds its own action and key.
    other = prepare(store, run_id, slot="announce")
    assert (other.sequence, other.slot) == (1, "announce")
    assert other.action_key != first.action_key and other.action_id != first.action_id

    # Once the action is finished, the same intent returns it, still closed,
    # and a changed intent starts the slot's next action with its own key.
    canceled = reject(store, first.action_id, sign(host, first), host)
    assert canceled.state == "canceled"
    assert prepare(store, run_id) == canceled
    second = prepare(store, run_id, title="Edited title")
    assert (second.sequence, second.state) == (2, "not_dispatched")
    assert second.action_key == action_key(store.store_id, run_id, "publish", 2)
    assert prepare(store, run_id, title="Edited title") == second
    with pytest.raises(WorkflowConflictError):
        prepare(store, run_id)
    items = actions_block(store, run_id)["items"]
    assert [(item["slot"], item["sequence"], item["state"]) for item in items] == [
        ("announce", 1, "not_dispatched"), ("publish", 1, "canceled"),
        ("publish", 2, "not_dispatched")]
    assert store.audit(run_id)["ok"]


def test_actions_can_be_prepared_only_after_an_operator_acceptance(
        store, tmp_path, replies, host, clock):  # noqa: F811
    run_id = completed(store, tmp_path, replies, renderer=None,
                       acceptance=strict(Raises(), on_invalid="needs_review"))
    with pytest.raises(ActionNotAuthorized, match="needs_review"):
        prepare(store, run_id)

    decided = swarm(store, tmp_path, renderer=None, acceptance=None).reevaluate(
        run_id, operator=durable_operator(host, store, run_id))
    assert decided.accepted and decided.acceptance.operator is not None

    action = prepare(store, run_id)
    assert action.intent.content.sha256 == decided.acceptance.output_sha256
    approval = approved(store, host, action, clock)
    assert check_authority(store, action.action_id, host) == approval


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def test_an_approval_is_verified_for_the_action_and_checked_again(
        store, tmp_path, replies, host, clock):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    action = prepare(store, run_id)
    assertion = sign(host, action)
    # Another worker's live lease does not stop approval.
    store.acquire_lease(run_id, "another-worker", ttl_s=60)

    approval = approved(store, host, action, clock, assertion)

    principal = Principal(host.issuer, "editor-1", "acme")
    assert approval == ActionApproval(approval.approval_id, action.action_id, principal, CEILING,
                                      clock.now_ns + HOUR_NS)
    # The verifier checks the action's approval subject: a versioned digest of its key and intent digest.
    assert APPROVAL_SUBJECT_VERSION == "smythe.action-approval.v1"
    assert action.approval_sha256 == hashlib.sha256(canonical_bytes({
        "type": "smythe.action-approval.v1", "action_key": action.action_key,
        "intent_sha256": action.intent.digest()})).hexdigest()
    assert host.verified == [action.approval_sha256]
    assert check_authority(store, action.action_id, host) == approval
    assert host.verified == [action.approval_sha256] * 2
    # The same approval again is recorded once.
    assert approved(store, host, action, clock, assertion) == approval
    assert len(approval_rows(store.path)) == 1
    item = actions_block(store, run_id)["items"][0]
    assert item["approvals"] == [{
        "approval_id": approval.approval_id, "verdict": "approved",
        "intent_sha256": action.intent.digest(), "principal": principal.to_dict(),
        "verifier": {"identity": host.identity}, "ceiling_nanousd": CEILING,
        "expires_at_ns": clock.now_ns + HOUR_NS, "consumed_by_attempt": None,
        "revoked_at_ns": None, "duplicate_risk": False}]
    assert store.audit(run_id)["ok"]


def _another_intent(host, store, run_id, action):
    other = prepare(store, run_id, slot="announce", title="Another post")
    return sign(host, other), "approves another"


def _intent_digest(host, store, run_id, action):
    # The bare intent digest would fit an identical intent of any other action.
    return host.issue(action.intent.digest(), subject="editor-1", tenant="acme", expires_at_ns=FAR), (
        "approves another")


def _forged(host, store, run_id, action):
    return sign(HmacApprovalHost(issuer=host.issuer), action), "signature"


def _other_tenant(host, store, run_id, action):
    return sign(host, action, tenant="globex"), "tenant 'globex' cannot approve"


def _revoked_subject(host, store, run_id, action):
    host.revoke_subject("editor-1")
    return sign(host, action), "revoked"


def _withdrawn(host, store, run_id, action):
    host.authorized = False
    return sign(host, action), "no longer authorizes"


MISBOUND = {
    "another-intent": _another_intent,
    "intent-digest": _intent_digest,
    "forged": _forged,
    "other-tenant": _other_tenant,
    "revoked-subject": _revoked_subject,
    "withdrawn": _withdrawn,
}


@pytest.mark.parametrize("case", sorted(MISBOUND))
def test_an_assertion_that_does_not_approve_this_action_is_refused(
        store, tmp_path, replies, host, clock, case):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    action = prepare(store, run_id)
    assertion, reason = MISBOUND[case](host, store, run_id, action)

    with pytest.raises(ApprovalRejected, match=reason):
        approved(store, host, action, clock, assertion)
    assert approval_rows(store.path) == []
    with pytest.raises(ActionNotAuthorized, match="no approval"):
        check_authority(store, action.action_id, host)


@pytest.mark.parametrize("ceiling,expiry,reason", [
    (CEILING - 1, HOUR_NS, "must cover the action's ceiling of 4020000"),
    (CEILING, 0, "later than the journal's clock"),
    (CEILING, -HOUR_NS, "later than the journal's clock"),
    (True, HOUR_NS, "non-negative integer"),
    (float(CEILING), HOUR_NS, "non-negative integer"),
])
def test_an_approval_must_cover_the_ceiling_and_expire_later(
        store, tmp_path, replies, host, clock, ceiling, expiry, reason):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))

    with pytest.raises(WorkflowValidationError, match=reason):
        approve(store, action.action_id, sign(host, action), host, ceiling_nanousd=ceiling,
                expires_at_ns=clock.now_ns + expiry)
    # Refused before the host is asked.
    assert host.verified == [] and approval_rows(store.path) == []


def test_an_assertion_approves_one_action_even_where_intents_match(
        store, tmp_path, replies, host, clock):  # noqa: F811
    # Two runs with the same accepted document prepare the same intent.
    first = prepare(store, completed(store, tmp_path, replies))
    second = prepare(store, completed(store, tmp_path, replies))
    assert first.intent.digest() == second.intent.digest() and first.run_id != second.run_id
    assertion = sign(host, first)
    approved(store, host, first, clock, assertion)

    # The verifier refuses it for the second action, whose approval subject differs.
    with pytest.raises(ApprovalRejected, match="approves another"):
        approved(store, host, second, clock, assertion)
    # The journal refuses it too, even from a verifier that ignores the subject.
    class Careless(Host):
        def verify(self, assertion, subject_sha256, now_ns):
            return super().verify(assertion, self._claims(assertion)["subject_sha256"], now_ns)

    with pytest.raises(ApprovalRejected, match="already recorded"):
        approved(store, Careless(key=host.key), second, clock, assertion)
    # Nor can the same assertion approve the first action again on other terms.
    with pytest.raises(ApprovalRejected, match="already recorded"):
        approved(store, host, first, clock, assertion, hours=2)
    with pytest.raises(ActionNotAuthorized, match="no approval"):
        check_authority(store, second.action_id, host)
    assert len(approval_rows(store.path)) == 1


def test_an_assertion_recorded_in_another_journal_does_not_approve_the_same_intent(
        tmp_path, replies, host, clock):  # noqa: F811
    service = FixtureService(clock_ns=clock)
    connector = FixtureConnector(service, write_ceiling_nanousd=2_000_000, read_ceiling_nanousd=10_000,
                                 max_attempts=2, readback_reads=2)
    with (SQLiteWorkflowStore(tmp_path / "first.db", clock_ns=clock) as first_store,
          SQLiteWorkflowStore(tmp_path / "second.db", clock_ns=clock) as second_store):
        # Two journals whose runs accepted the same document prepare the identical intent.
        first = prepare(first_store, completed(first_store, tmp_path / "first", replies))
        second = prepare(second_store, completed(second_store, tmp_path / "second", replies))
        assert first.intent == second.intent and first.action_key != second.action_key
        assertion = sign(host, first)
        approved(first_store, host, first, clock, assertion)

        # The assertion that one journal recorded approves no action of another.
        with pytest.raises(ApprovalRejected, match="approves another"):
            approved(second_store, host, second, clock, assertion)
        assert approval_rows(second_store.path) == []
        with pytest.raises(ActionNotAuthorized, match="it has no approval"):
            ActionRunner(second_store, connector, host).dispatch(second.run_id, second.action_id)
        assert service.snapshot()["deliveries"] == []
        # The journal that recorded it is authorized, as before.
        assert check_authority(first_store, first.action_id, host).action_id == first.action_id


def test_an_assertion_must_be_text(store, tmp_path, replies, host, clock):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))
    for assertion in ("", None, b"assertion"):
        with pytest.raises(WorkflowValidationError, match="assertion"):
            approve(store, action.action_id, assertion, host, ceiling_nanousd=CEILING,
                    expires_at_ns=clock.now_ns + HOUR_NS)
        with pytest.raises(WorkflowValidationError, match="assertion"):
            reject(store, action.action_id, assertion, host)
    with pytest.raises(TypeError):
        approve(store, action.action_id, sign(host, action), OfflineProvider(),
                ceiling_nanousd=CEILING, expires_at_ns=clock.now_ns + HOUR_NS)
    for refuse in (lambda: load_action(store, "no-such-action"),
                   lambda: approve(store, "no-such-action", sign(host, action), host,
                                   ceiling_nanousd=CEILING, expires_at_ns=clock.now_ns + HOUR_NS),
                   lambda: check_authority(store, "no-such-action", host)):
        with pytest.raises(WorkflowNotFoundError):
            refuse()
    assert host.verified == []


# ---------------------------------------------------------------------------
# Checking authority later
# ---------------------------------------------------------------------------


def _expired(store, tmp_path, host, clock, action, approval, assertion):
    clock.advance_past(approval.expires_at_ns)
    return "its approval expired"


def _host_revoked(store, tmp_path, host, clock, action, approval, assertion):
    host.revoke(assertion)
    return "the approval verifier refuses it: The approval was revoked"


def _host_withdrawn(store, tmp_path, host, clock, action, approval, assertion):
    host.authorized = False
    return "no longer authorizes this principal"


def _revoked_action(store, tmp_path, host, clock, action, approval, assertion):
    assert revoke(store, action.action_id, "the campaign was withdrawn") == (approval.approval_id,)
    return "its approval was revoked"


def _revoked_approval(store, tmp_path, host, clock, action, approval, assertion):
    assert revoke(store, approval.approval_id, "the editor left") == (approval.approval_id,)
    return "its approval was revoked"


def _reevaluated(store, tmp_path, host, clock, action, approval, assertion):
    again = swarm(store, tmp_path).reevaluate(action.run_id, acceptance=strict(Contains("absent")))
    assert not again.accepted
    return "the run's delivery acceptance is rejected"


def _changed_on_disk(store, tmp_path, host, clock, action, approval, assertion):
    current_artifact(store, action.run_id).publication.path.write_bytes(b"# Edited later\n")
    return r"not eligible for delivery \(artifact_changed\)"


def _removed_from_disk(store, tmp_path, host, clock, action, approval, assertion):
    current_artifact(store, action.run_id).publication.path.unlink()
    return r"not eligible for delivery \(artifact_missing\)"


LATER = {
    "expired": _expired,
    "host-revoked": _host_revoked,
    "host-withdrawn": _host_withdrawn,
    "revoked-action": _revoked_action,
    "revoked-approval": _revoked_approval,
    "reevaluated": _reevaluated,
    "changed-on-disk": _changed_on_disk,
    "removed-from-disk": _removed_from_disk,
}


@pytest.mark.parametrize("case", sorted(LATER))
def test_a_delayed_check_fails_once_the_approval_or_the_content_no_longer_holds(
        store, tmp_path, replies, host, clock, case):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))
    assertion = sign(host, action)
    approval = approved(store, host, action, clock, assertion)
    clock.advance(60)
    assert check_authority(store, action.action_id, host) == approval

    reason = LATER[case](store, tmp_path, host, clock, action, approval, assertion)

    with pytest.raises(ActionNotAuthorized, match=reason):
        check_authority(store, action.action_id, host)
    assert store.audit(action.run_id)["ok"]


def test_a_removed_source_makes_the_content_unavailable(
        store, tmp_path, replies, host, clock):  # noqa: F811
    replies[0]["Write"] = [artifact(KEPT)]
    publishing = {"renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts"}
    result = sourced(store, **publishing).execute(single())
    assert result.accepted
    action = prepare(store, result.execution_id)
    approval = approved(store, host, action, clock)
    assert check_authority(store, action.action_id, host) == approval

    remove(store, result.execution_id, **publishing)

    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        check_authority(store, action.action_id, host)
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        prepare(store, result.execution_id, slot="announce")
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        approved(store, host, action, clock)


@pytest.mark.parametrize("decided_by", ["policy", "operator"])
def test_a_decision_whose_inputs_are_not_an_object_authorizes_nothing(
        store, tmp_path, replies, host, clock, monkeypatch, decided_by):  # noqa: F811
    if decided_by == "policy":
        run_id = completed(store, tmp_path, replies)
    else:
        run_id = completed(store, tmp_path, replies, renderer=None,
                           acceptance=strict(Raises(), on_invalid="needs_review"))
        swarm(store, tmp_path, renderer=None, acceptance=None).reevaluate(
            run_id, operator=durable_operator(host, store, run_id))
    action = prepare(store, run_id)
    assertion = sign(host, action)
    approved(store, host, action, clock, assertion)
    sequence = smythe.workflow.current_decision(store, run_id)["sequence"]
    assert sequence == (0 if decided_by == "policy" else 1)
    read = smythe.workflow.decision_history

    def history(store, run_id):
        # The journal itself refuses such inputs; a history read elsewhere may hold them.
        decisions = read(store, run_id)
        decisions[-1]["inputs"] = sorted(decisions[-1]["inputs"].items())
        return decisions

    monkeypatch.setattr(smythe.workflow, "decision_history", history)
    reason = f"the inputs of the journal's decision {sequence} are not a JSON object"
    with pytest.raises(ActionNotAuthorized, match=reason):
        check_authority(store, action.action_id, host)
    with pytest.raises(ActionNotAuthorized, match=reason):
        approved(store, host, action, clock, assertion)
    with pytest.raises(ActionNotAuthorized, match=reason):
        prepare(store, run_id, slot="announce")
    assert [item["slot"] for item in actions_block(store, run_id)["items"]] == ["publish"]


def test_an_approval_row_written_without_approve_never_authorizes(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))
    other = prepare(store, action.run_id, slot="announce", title="Another post")
    principal = _canonical({"issuer": host.issuer, "subject": "editor-1", "tenant": "acme"})

    def insert(assertion, principal_json=principal):
        with closing(sqlite3.connect(store.path)) as db, db:
            db.execute(
                "INSERT INTO workflow_action_approvals(approval_id,action_id,intent_sha,verdict,"
                "principal_json,verifier_json,assertion,assertion_sha,ceiling,expires_at_ns,created) "
                "VALUES (?,?,?,'approved',?,?,?,?,?,?,?)",
                (_sha(assertion)[:32], action.action_id, action.intent.digest(), principal_json,
                 _canonical({"identity": host.identity}), assertion, _sha(assertion), str(CEILING),
                 clock.now_ns + HOUR_NS, clock.now_ns))

    # The journal's audit accepts each row's form; only the host can vouch for it.
    insert("bogus")
    assert store.audit(action.run_id)["ok"]
    with pytest.raises(ActionNotAuthorized, match="refuses it: The assertion is malformed"):
        check_authority(store, action.action_id, host)
    insert(sign(host, other))
    with pytest.raises(ActionNotAuthorized, match="refuses it: The assertion approves another"):
        check_authority(store, action.action_id, host)
    insert(sign(host, action, subject="editor-2"))
    with pytest.raises(ActionNotAuthorized, match="names another principal"):
        check_authority(store, action.action_id, host)
    # A row recorded by another verifier is not this one's to vouch for.
    stranger = type("Stranger", (Host,), {"identity": "another-host/1"})(key=host.key)
    with pytest.raises(ActionNotAuthorized, match="another approval verifier"):
        check_authority(store, action.action_id, stranger)


def test_the_newest_approval_decides(store, tmp_path, replies, host, clock):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))
    lasting = approved(store, host, action, clock, hours=24)
    brief = approved(store, host, action, clock, hours=1)
    assert check_authority(store, action.action_id, host) == brief

    clock.advance_past(brief.expires_at_ns)
    # The longer approval it superseded does not come back.
    with pytest.raises(ActionNotAuthorized, match="expired"):
        check_authority(store, action.action_id, host)
    renewed = approved(store, host, action, clock, hours=1)
    assert check_authority(store, action.action_id, host) == renewed
    assert lasting.approval_id not in (brief.approval_id, renewed.approval_id)


# ---------------------------------------------------------------------------
# Rejection and revocation
# ---------------------------------------------------------------------------


def test_a_rejection_keeps_the_action_closed_after_the_store_is_reopened(
        journal, tmp_path, replies, host):  # noqa: F811
    clock = ManualClock(time.time_ns())
    with SQLiteWorkflowStore(journal, clock_ns=clock) as store:
        action = prepare(store, completed(store, tmp_path, replies))
        # Only a verified principal of the action's tenant can reject it.
        for assertion, reason in ((sign(HmacApprovalHost(issuer=host.issuer), action), "signature"),
                                  (sign(host, action, tenant="globex"), "tenant")):
            with pytest.raises(ApprovalRejected, match=reason):
                reject(store, action.action_id, assertion, host)
        assert load_action(store, action.action_id).state == "not_dispatched"
        canceled = reject(store, action.action_id, sign(host, action), host)
        assert canceled == PreparedAction(*[getattr(action, name) for name in (
            "action_id", "run_id", "slot", "sequence", "action_key", "intent", "profile")],
            "canceled", CEILING)

    with SQLiteWorkflowStore(journal, clock_ns=clock) as store:
        assert load_action(store, action.action_id).state == "canceled"
        verified = len(host.verified)
        with pytest.raises(ActionStateError, match="canceled and cannot be approved"):
            approved(store, host, action, clock)
        with pytest.raises(ActionNotAuthorized, match="it is canceled"):
            check_authority(store, action.action_id, host)
        # Preparing the same intent returns the closed action, and rejecting it again changes nothing.
        assert prepare(store, action.run_id) == canceled
        assert reject(store, action.action_id, sign(host, action), host) == canceled
        assert len(host.verified) == verified
        (rejection,) = approval_rows(journal)
        assert (rejection["verdict"], rejection["ceiling"], rejection["expires_at_ns"]) == (
            "rejected", None, None)
        assert json.loads(rejection["principal_json"]) == {
            "issuer": host.issuer, "subject": "editor-1", "tenant": "acme"}
        assert store.audit(action.run_id)["ok"]


def test_revocation_needs_no_principal_and_redacts_its_reason(
        store, tmp_path, replies, host, clock):  # noqa: F811
    action = prepare(store, completed(store, tmp_path, replies))
    earlier = approved(store, host, action, clock)
    newest = approved(store, host, action, clock)
    store.acquire_lease(action.run_id, "another-worker", ttl_s=60)

    # Revoking a superseded approval leaves the newest one in force.
    assert revoke(store, earlier.approval_id, "superseded") == (earlier.approval_id,)
    assert check_authority(store, action.action_id, host) == newest
    reason = f"key leaked in organization {ORGANIZATION}"
    assert revoke(store, action.action_id, reason) == (newest.approval_id,)
    assert revoke(store, action.action_id, "again") == ()
    with pytest.raises(ActionNotAuthorized, match="was revoked"):
        check_authority(store, action.action_id, host)

    rows = {row["approval_id"]: row for row in approval_rows(store.path)}
    assert rows[earlier.approval_id]["revoked_reason"] == "superseded"
    assert rows[newest.approval_id]["revoked_reason"] == (
        "key leaked in organization org-[redacted]")
    assert rows[newest.approval_id]["revoked_at_ns"] == clock.now_ns
    # A revoked assertion never approves again; a new approval reopens the action.
    with pytest.raises(ApprovalRejected, match="already recorded"):
        approved(store, host, action, clock, rows[newest.approval_id]["assertion"])
    reopened = approved(store, host, action, clock)
    assert check_authority(store, action.action_id, host) == reopened

    with pytest.raises(WorkflowNotFoundError):
        revoke(store, "no-such-action", "reason")
    for bad in ("", "two\nlines", None):
        with pytest.raises(WorkflowValidationError, match="reason"):
            revoke(store, action.action_id, bad)
    assert store.audit(action.run_id)["ok"]


def test_a_read_only_store_checks_but_never_writes(journal, tmp_path, replies, host):  # noqa: F811
    clock = ManualClock(time.time_ns())
    with SQLiteWorkflowStore(journal, clock_ns=clock) as store:
        action = prepare(store, completed(store, tmp_path, replies))
        approval = approved(store, host, action, clock)
    before = journal.read_bytes()

    with SQLiteWorkflowStore(journal, read_only=True, clock_ns=clock) as reader:
        assert check_authority(reader, action.action_id, host) == approval
        assert load_action(reader, action.action_id) == action
        for write in (lambda: prepare(reader, action.run_id, slot="announce"),
                      lambda: approved(reader, host, action, clock),
                      lambda: reject(reader, action.action_id, sign(host, action), host),
                      lambda: revoke(reader, action.action_id, "reason")):
            with pytest.raises(WorkflowStateError, match="read-only"):
                write()
    assert journal.read_bytes() == before


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_inspection_never_contains_assertions(journal, tmp_path, replies, host, capsys):  # noqa: F811
    clock = ManualClock(time.time_ns())
    with SQLiteWorkflowStore(journal, clock_ns=clock) as store:
        run_id = completed(store, tmp_path, replies)
        kept, dropped = prepare(store, run_id), prepare(store, run_id, slot="announce")
        assertions = [sign(host, kept), sign(host, kept), sign(host, dropped)]
        approved(store, host, kept, clock, assertions[0])
        approved(store, host, kept, clock, assertions[1])
        revoke(store, kept.action_id, "REVOCATION-REASON")
        reject(store, dropped.action_id, assertions[2], host)
        inspected = json.dumps(store.inspect_run(run_id))
        listed = json.dumps(store.list_runs())

    assert main(["runs", "--journal", str(journal), "--json", "inspect", run_id]) == 0
    shown = capsys.readouterr().out
    assert json.loads(shown)["ok"] is True
    signatures = [assertion.split(".")[1] for assertion in assertions]
    for text in (inspected, listed, shown):
        for secret in (*assertions, *signatures, "REVOCATION-REASON"):
            assert secret not in text
    items = {item["action_id"]: item for item in json.loads(inspected)["actions"]["items"]}
    assert [approval["verdict"] for approval in items[dropped.action_id]["approvals"]] == ["rejected"]
    assert [approval["revoked_at_ns"] is not None
            for approval in items[kept.action_id]["approvals"]] == [True, True]
