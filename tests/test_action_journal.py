"""The durable dispatch journal of external actions: attempts, claims, outcomes and recovery.

A dispatcher holds the run's lease. It begins an attempt under the action's
approval, which reserves the write ceiling; claims its dispatch, which
commits the attempt as dispatched before anything is sent; makes one
submission; and records the receipt, a refusal or an ambiguous outcome.
A permit from a superseded lease session changes nothing, and recovery
treats every attempt in flight as abandoned.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import json
import sqlite3
import time

import pytest

from fakes import Crash, ManualClock, crash_when
from fixture_connector import FixtureConnector
from fixture_service import FixtureService
from smythe import MarkdownRenderer, SQLiteWorkflowStore, workflow_actions
from smythe.actions import (
    ActionNotAuthorized,
    ActionStateError,
    ActionTarget,
    ConnectorAmbiguous,
    ConnectorRefused,
    check_authority,
    prepare_action,
    reject,
    revoke,
)
from smythe.cli import main
from smythe.workflow_actions import (
    ActionDispatchPermit,
    begin_attempt,
    claim_dispatch,
    record_ambiguous,
    record_receipt,
    record_refused,
    recover_actions,
)
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import (
    WorkflowBudgetError, WorkflowConflictError, WorkflowCorruptionError, WorkflowLeaseError,
    WorkflowNotFoundError, WorkflowStateError, WorkflowValidationError, _canonical, _sha,
)
from test_action_approvals import (
    CEILING, ORGANIZATION, PROFILE, Host, approved, completed, delivered, intent, prepare, sign, swarm,
)
from test_source_evidence import KEPT, artifact
from test_source_tombstones import durable as sourced, remove
from test_workflow_acceptance import Contains, replies, single, strict  # noqa: F401

TTL = 60
WRITE = PROFILE.write_ceiling_nanousd
SETTINGS = {"write_ceiling_nanousd": WRITE, "read_ceiling_nanousd": 10_000, "max_attempts": 2,
            "readback_reads": 2}


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
def connector(service) -> FixtureConnector:
    value = FixtureConnector(service, **SETTINGS)
    assert value.profile == PROFILE
    return value


def ready(store, tmp_path, replies, host, clock, *, profile=PROFILE, **options):  # noqa: F811
    """A prepared and approved action of a completed run, and a lease on the run."""
    run_id = completed(store, tmp_path, replies, **options)
    action = prepare_action(store, run_id, slot="publish", intent=intent(delivered(store, run_id)),
                            profile=profile)
    approval = approved(store, host, action, clock)
    return action, approval, store.acquire_lease(run_id, "worker-1", ttl_s=TTL)


def document(store, action) -> bytes:
    """The bytes the action delivers: the run's published document, or else its output."""
    current = current_artifact(store, action.run_id)
    if current is not None:
        return current.publication.path.read_bytes()
    return store.get_checkpoint(action.run_id)["checkpoint"]["output"].encode("utf-8")


def dispatch(store, lease, action, approval, connector):
    """One attempt as a dispatcher makes it: begin, claim, one submission and its outcome."""
    attempt = workflow_actions.begin_attempt(store, lease, action.action_id, approval.approval_id)
    permit = workflow_actions.claim_dispatch(store, lease, attempt["attempt_id"])
    try:
        receipt = connector.submit(action.action_key, action.intent, document(store, action),
                                   action.profile.timeout_s)
    except ConnectorRefused as error:
        return workflow_actions.record_refused(store, permit, error)
    except ConnectorAmbiguous as error:
        return workflow_actions.record_ambiguous(store, permit, error)
    return workflow_actions.record_receipt(store, permit, receipt)


def item(store, action) -> dict:
    """The action as ``inspect_run`` projects it."""
    (found,) = [entry for entry in store.inspect_run(action.run_id)["actions"]["items"]
                if entry["action_id"] == action.action_id]
    return found


def balances(entry) -> tuple:
    return entry["confirmed_nanousd"], entry["reserved_nanousd"], entry["unknown_nanousd"]


def rows(path, table) -> list[dict]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def dump(path) -> str:
    with closing(sqlite3.connect(path)) as db:
        return "\n".join(db.iterdump())


def execute(path, statement, parameters=()) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(statement, parameters)


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------


def test_an_attempt_reserves_claims_and_records_its_receipt(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    assert check_authority(store, action.action_id, host) == approval

    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)

    assert attempt == {"attempt_id": attempt["attempt_id"], "number": 1, "approval_id": approval.approval_id,
                       "state": "prepared", "dispatch_epoch": None, "receipt_sha256": None,
                       "cost_nanousd": None, "reserved_nanousd": WRITE, "unknown_nanousd": 0}
    # Beginning again under the same approval returns the prepared attempt.
    assert begin_attempt(store, lease, action.action_id, approval.approval_id) == attempt
    current = item(store, action)
    assert (current["state"], balances(current)) == ("not_dispatched", (0, WRITE, 0))
    assert service.snapshot()["deliveries"] == []

    clock.advance(1)
    permit = claim_dispatch(store, lease, attempt["attempt_id"])

    assert (permit.attempt_id, permit.run_id, permit.epoch) == (attempt["attempt_id"], action.run_id,
                                                               lease.epoch)
    assert permit.token not in repr(permit) and permit.token not in dump(store.path)
    (row,) = rows(store.path, "workflow_action_attempts")
    assert (row["state"], row["dispatch_token_sha"], row["dispatched"]) == (
        "dispatched", _sha(permit.token), clock.now_ns)
    assert item(store, action)["state"] == "dispatched"

    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)
    priced = replace(receipt, cost_nanousd=1_500_000)
    recorded = record_receipt(store, permit, priced)

    assert recorded == {**attempt, "state": "acknowledged", "dispatch_epoch": lease.epoch,
                        "receipt_sha256": _sha(_canonical(priced.to_dict())), "cost_nanousd": 1_500_000,
                        "reserved_nanousd": 0}
    # The same receipt again changes nothing.
    assert record_receipt(store, permit, priced) == recorded
    current = item(store, action)
    # A receipt alone never shows the result: the action stays dispatched until a read does.
    assert (current["state"], balances(current)) == ("dispatched", (1_500_000, 0, 0))
    assert current["records"] == []
    # The effect consumes the approval, which authorizes nothing further.
    assert current["approvals"][0]["consumed_by_attempt"] == attempt["attempt_id"]
    with pytest.raises(ActionNotAuthorized, match="its approval was consumed"):
        check_authority(store, action.action_id, host)
    with pytest.raises(ActionStateError, match="is dispatched"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)
    snapshot = service.snapshot()
    assert snapshot["totals"]["committed"] == 1
    assert snapshot["requests"][action.action_key]["object_ids"] == [receipt.target.object_id]
    events = [event["data"] for event in store.inspect_run(action.run_id)["events"]
              if event["type"] == "action_attempt"]
    assert events == [{"action_id": action.action_id, "attempt_id": attempt["attempt_id"], "state": state}
                      for state in ("prepared", "dispatched", "acknowledged")]
    assert store.audit(action.run_id)["ok"]


@pytest.mark.parametrize("ceiling,cost,expected", [
    (WRITE, None, (0, 0, WRITE)),  # No reported cost: the write's exposure is kept.
    (WRITE, 1_200_000, (1_200_000, 0, 0)),
    (0, None, (0, 0, 0)),  # A write declared free costs nothing.
])
def test_a_receipt_confirms_its_cost_or_keeps_the_exposure(
        store, tmp_path, replies, host, clock, service, ceiling, cost, expected):  # noqa: F811
    connector = FixtureConnector(service, **{**SETTINGS, "write_ceiling_nanousd": ceiling})
    action, approval, lease = ready(store, tmp_path, replies, host, clock, profile=connector.profile)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    permit = claim_dispatch(store, lease, attempt["attempt_id"])
    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)

    recorded = record_receipt(store, permit, replace(receipt, cost_nanousd=cost))

    assert (recorded["cost_nanousd"], recorded["reserved_nanousd"], recorded["unknown_nanousd"]) == (
        expected[0] if cost is not None or ceiling == 0 else None, 0, expected[2])
    current = item(store, action)
    assert (current["state"], balances(current)) == ("dispatched", expected)
    assert current["approvals"][0]["consumed_by_attempt"] == attempt["attempt_id"]
    assert store.audit(action.run_id)["ok"]


def test_a_refused_attempt_gives_back_its_reservation_and_keeps_the_approval(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    service.fail_before_commit(action.action_key)

    refused = dispatch(store, lease, action, approval, connector)

    assert (refused["state"], refused["cost_nanousd"], refused["reserved_nanousd"]) == ("refused", 0, 0)
    current = item(store, action)
    assert (current["state"], balances(current)) == ("not_dispatched", (0, 0, 0))
    assert current["approvals"][0]["consumed_by_attempt"] is None
    assert check_authority(store, action.action_id, host) == approval

    # The same approval authorizes the next attempt, which sends the same key.
    acknowledged = dispatch(store, lease, action, approval, connector)

    assert (acknowledged["number"], acknowledged["state"]) == (2, "acknowledged")
    assert [attempt["state"] for attempt in item(store, action)["attempts"]] == ["refused", "acknowledged"]
    snapshot = service.snapshot()
    assert [delivery["request_key"] for delivery in snapshot["deliveries"]] == [action.action_key] * 2
    assert (snapshot["totals"]["refused"], snapshot["totals"]["committed"]) == (1, 1)
    assert store.audit(action.run_id)["ok"]


def test_refusals_count_toward_the_attempt_allowance(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    for delivery in range(PROFILE.max_attempts):
        service.fail_before_commit(action.action_key, delivery=delivery)
        assert dispatch(store, lease, action, approval, connector)["state"] == "refused"

    with pytest.raises(ActionStateError, match="has made all 2 of its attempts"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)

    current = item(store, action)
    assert (current["state"], balances(current), len(current["attempts"])) == (
        "not_dispatched", (0, 0, 0), 2)
    assert current["approvals"][0]["consumed_by_attempt"] is None
    assert service.snapshot()["totals"]["committed"] == 0


def test_an_ambiguous_attempt_keeps_its_exposure(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    service.commit_then_drop_ack(action.action_key)

    unknown = dispatch(store, lease, action, approval, connector)

    assert (unknown["state"], unknown["cost_nanousd"], unknown["reserved_nanousd"],
            unknown["unknown_nanousd"]) == ("outcome_unknown", None, 0, WRITE)
    current = item(store, action)
    assert (current["state"], balances(current)) == ("outcome_unknown", (0, 0, WRITE))
    # The write may have taken effect: nothing resends it or releases its exposure.
    assert current["approvals"][0]["consumed_by_attempt"] is None
    with pytest.raises(ActionStateError, match="is outcome_unknown"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)
    assert recover_actions(store, lease) == {"released": [], "outcome_unknown": []}
    store.release_lease(lease)
    successor = store.acquire_lease(action.run_id, "worker-2", ttl_s=TTL)
    assert recover_actions(store, successor) == {"released": [], "outcome_unknown": []}
    assert balances(item(store, action)) == (0, 0, WRITE)
    snapshot = service.snapshot()
    assert (snapshot["totals"]["received"], snapshot["totals"]["committed"]) == (1, 1)
    assert store.audit(action.run_id)["ok"]


def test_connector_error_text_is_kept_redacted_and_never_inspected(
        store, tmp_path, replies, host, clock, service, connector, capsys):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    attempts = []
    for record, error in ((record_refused, ConnectorRefused(f"SECRET-REFUSAL for {ORGANIZATION}")),
                          (record_ambiguous, f"SECRET-TIMEOUT for {ORGANIZATION}")):
        attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
        permit = claim_dispatch(store, lease, attempt["attempt_id"])
        attempts.append(record(store, permit, error)["attempt_id"])

    errors = {row["attempt_id"]: row["error"] for row in rows(store.path, "workflow_action_attempts")}
    assert errors == {attempts[0]: "ConnectorRefused: SECRET-REFUSAL for org-[redacted]",
                      attempts[1]: "SECRET-TIMEOUT for org-[redacted]"}
    shown = [json.dumps(store.inspect_run(action.run_id)), json.dumps(store.list_runs())]
    assert main(["runs", "--journal", str(store.path), "--json", "inspect", action.run_id]) == 0
    shown.append(capsys.readouterr().out)
    for text in shown:
        assert "SECRET" not in text and ORGANIZATION not in text


# ---------------------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------------------


def test_illegal_transitions_are_refused_and_change_nothing(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    other_run = completed(store, tmp_path, replies)
    other = store.acquire_lease(other_run, "worker-1", ttl_s=TTL)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    permit = claim_dispatch(store, lease, attempt["attempt_id"])
    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)
    before = dump(store.path)

    refusals = [
        # A dispatch is claimed once, and no attempt begins beside one in flight.
        (lambda: claim_dispatch(store, lease, attempt["attempt_id"]), ActionStateError, "is dispatched"),
        (lambda: begin_attempt(store, lease, action.action_id, approval.approval_id), ActionStateError,
         "is dispatched"),
        # Only the lease of the action's own run can begin or claim.
        (lambda: begin_attempt(store, other, action.action_id, approval.approval_id),
         WorkflowConflictError, "another workflow run"),
        (lambda: claim_dispatch(store, other, attempt["attempt_id"]), WorkflowConflictError,
         "another workflow run"),
        (lambda: begin_attempt(store, None, action.action_id, approval.approval_id), WorkflowLeaseError,
         "lease token"),
        (lambda: claim_dispatch(store, lease, "no-such-attempt"), WorkflowNotFoundError, "no-such-attempt"),
        (lambda: begin_attempt(store, lease, "no-such-action", approval.approval_id), WorkflowNotFoundError,
         "no-such-action"),
        # Only the permit of the claim records its outcome.
        (lambda: record_receipt(store, replace(permit, token="0" * 64), receipt), WorkflowConflictError,
         "does not match"),
        (lambda: record_refused(store, replace(permit, epoch=permit.epoch + 1), "refused"),
         WorkflowConflictError, "does not match"),
        (lambda: record_ambiguous(store, replace(permit, attempt_id="no-such-attempt"), "lost"),
         WorkflowNotFoundError, "no-such-attempt"),
        (lambda: record_refused(store, {"attempt_id": attempt["attempt_id"], "token": permit.token},
                                "refused"), WorkflowValidationError, "permit"),
        # A receipt must be a receipt for the action's own key and target.
        (lambda: record_receipt(store, permit, receipt.to_dict()), WorkflowValidationError,
         "ConnectorReceipt"),
        (lambda: record_receipt(store, permit, replace(receipt, key="another-key")),
         WorkflowValidationError, "another key or target"),
        (lambda: record_receipt(store, permit, replace(
            receipt, target=ActionTarget("globex", "posts", receipt.target.object_id))),
         WorkflowValidationError, "another key or target"),
        (lambda: record_receipt(store, permit, replace(
            receipt, target=ActionTarget("acme", "drafts", receipt.target.object_id))),
         WorkflowValidationError, "another key or target"),
    ]
    for refuse, error, message in refusals:
        with pytest.raises(error, match=message):
            refuse()
    assert dump(store.path) == before

    # Once an outcome is recorded, a different one for the same attempt changes nothing.
    unknown = record_ambiguous(store, permit, ConnectorAmbiguous("timed out"))
    assert record_ambiguous(store, permit, ConnectorAmbiguous("timed out")) == unknown
    with pytest.raises(ActionStateError, match="is outcome_unknown; its outcome is kept"):
        record_receipt(store, permit, receipt)
    current = item(store, action)
    assert current["attempts"] == [unknown]
    assert (current["state"], balances(current)) == ("outcome_unknown", (0, 0, WRITE))
    (late,) = current["records"]
    assert (late["kind"], late["trusted"], late["cost_nanousd"]) == ("late_receipt", True, None)
    assert current["approvals"][0]["consumed_by_attempt"] is None
    assert store.audit(action.run_id)["ok"]


def test_an_attempt_must_fit_the_approval_ceiling(store, tmp_path, replies, host, clock):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    # Spending recorded against the action, such as reads, counts toward its approval.
    spent = CEILING - WRITE + 1
    payload = _canonical({"read": 1})
    execute(store.path, "INSERT INTO workflow_action_records(record_id,action_id,kind,payload_json,payload_sha,"
            "trusted,cost,created) VALUES ('read-1',?,'readback',?,?,1,?,?)",
            (action.action_id, payload, _sha(payload), str(spent), clock.now_ns))
    execute(store.path, "UPDATE workflow_actions SET confirmed=? WHERE action_id=?", (str(spent), action.action_id))
    assert store.audit(action.run_id)["ok"]

    with pytest.raises(WorkflowBudgetError, match="could exceed its approval's ceiling"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)
    assert rows(store.path, "workflow_action_attempts") == []


def test_a_read_only_store_never_dispatches(tmp_path, replies, host):  # noqa: F811
    journal, clock = tmp_path / "runs.db", ManualClock(time.time_ns())
    with SQLiteWorkflowStore(journal, clock_ns=clock) as store:
        action, approval, lease = ready(store, tmp_path, replies, host, clock)
        prepared = begin_attempt(store, lease, action.action_id, approval.approval_id)
    before = dump(journal)

    with SQLiteWorkflowStore(journal, read_only=True, clock_ns=clock) as reader:
        permit = ActionDispatchPermit(prepared["attempt_id"], action.run_id, "0" * 64, lease.epoch)
        for write in (lambda: begin_attempt(reader, lease, action.action_id, approval.approval_id),
                      lambda: claim_dispatch(reader, lease, prepared["attempt_id"]),
                      lambda: record_refused(reader, permit, "refused"),
                      lambda: recover_actions(reader, lease)):
            with pytest.raises(WorkflowStateError, match="read-only"):
                write()
    assert dump(journal) == before


# ---------------------------------------------------------------------------
# Authority is checked again in each transaction
# ---------------------------------------------------------------------------


def _revoked(store, tmp_path, host, clock, action, approval):
    revoke(store, action.action_id, "the campaign was withdrawn")
    return ActionNotAuthorized, "its approval was revoked"


def _expired(store, tmp_path, host, clock, action, approval):
    clock.advance_past(approval.expires_at_ns)
    return ActionNotAuthorized, "its approval expired"


def _superseded(store, tmp_path, host, clock, action, approval):
    approved(store, host, action, clock)
    return ActionNotAuthorized, f"approval {approval.approval_id!r} is not its newest approval"


def _security_stop(store, tmp_path, host, clock, action, approval):
    execute(store.path, "UPDATE workflow_runs SET blocked_reason='security_stop' WHERE run_id=?",
            (action.run_id,))
    return ActionNotAuthorized, "a security stop holds the run"


def _reevaluated(store, tmp_path, host, clock, action, approval):
    assert not swarm(store, tmp_path).reevaluate(action.run_id, acceptance=strict(Contains("absent"))).accepted
    return ActionNotAuthorized, "the run's delivery acceptance is rejected"


def _changed_on_disk(store, tmp_path, host, clock, action, approval):
    current_artifact(store, action.run_id).publication.path.write_bytes(b"# Edited later\n")
    return ActionNotAuthorized, r"not eligible for delivery \(artifact_changed\)"


AUTHORITY = {
    "revoked": _revoked,
    "expired": _expired,
    "superseded": _superseded,
    "security-stop": _security_stop,
    "reevaluated": _reevaluated,
    "changed-on-disk": _changed_on_disk,
}


@pytest.mark.parametrize("case", sorted(AUTHORITY))
def test_beginning_an_attempt_checks_its_authority_again(
        store, tmp_path, replies, host, clock, case):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    store.release_lease(lease)  # A re-evaluation takes the run's lease.
    error, message = AUTHORITY[case](store, tmp_path, host, clock, action, approval)
    lease = store.acquire_lease(action.run_id, "worker-1", ttl_s=TTL)

    with pytest.raises(error, match=message):
        begin_attempt(store, lease, action.action_id, approval.approval_id)

    assert rows(store.path, "workflow_action_attempts") == []
    assert balances(item(store, action)) == (0, 0, 0)


@pytest.mark.parametrize("case", sorted(AUTHORITY))
def test_a_claim_without_a_valid_approval_or_content_releases_the_attempt(
        store, tmp_path, replies, host, clock, service, connector, case):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    store.release_lease(lease)
    error, message = AUTHORITY[case](store, tmp_path, host, clock, action, approval)
    lease = store.acquire_lease(action.run_id, "worker-1", ttl_s=TTL)

    with pytest.raises(error, match=message):
        claim_dispatch(store, lease, attempt["attempt_id"])

    current = item(store, action)
    assert current["attempts"] == [{**attempt, "state": "released", "reserved_nanousd": 0}]
    assert (current["state"], balances(current)) == ("not_dispatched", (0, 0, 0))
    (row,) = rows(store.path, "workflow_action_attempts")
    assert message.replace("\\", "") in row["error"]
    with pytest.raises(ActionStateError, match="is released"):
        claim_dispatch(store, lease, attempt["attempt_id"])
    assert service.snapshot()["deliveries"] == []
    assert store.audit(action.run_id)["ok"]


def test_a_newer_approval_releases_the_attempt_prepared_under_an_older_one(
        store, tmp_path, replies, host, clock, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    older = begin_attempt(store, lease, action.action_id, approval.approval_id)
    newer = approved(store, host, action, clock)

    attempt = begin_attempt(store, lease, action.action_id, newer.approval_id)

    assert (attempt["number"], attempt["approval_id"], attempt["reserved_nanousd"]) == (
        2, newer.approval_id, WRITE)
    current = item(store, action)
    assert [entry["state"] for entry in current["attempts"]] == ["released", "prepared"]
    assert balances(current) == (0, WRITE, 0)
    with pytest.raises(ActionStateError, match="is released"):
        claim_dispatch(store, lease, older["attempt_id"])
    permit = claim_dispatch(store, lease, attempt["attempt_id"])
    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)
    assert record_receipt(store, permit, receipt)["state"] == "acknowledged"
    assert store.audit(action.run_id)["ok"]


def test_a_canceled_action_begins_no_attempt(store, tmp_path, replies, host, clock):  # noqa: F811
    run_id = completed(store, tmp_path, replies)
    action = prepare(store, run_id)
    lease = store.acquire_lease(run_id, "worker-1", ttl_s=TTL)
    with pytest.raises(ActionNotAuthorized, match="it has no approval"):
        begin_attempt(store, lease, action.action_id, "no-such-approval")
    approval = approved(store, host, action, clock)
    reject(store, action.action_id, sign(host, action), host)

    with pytest.raises(ActionNotAuthorized, match="it is canceled"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)
    assert rows(store.path, "workflow_action_attempts") == []


def test_a_removed_source_stops_a_dispatch(store, tmp_path, replies, host, clock):  # noqa: F811
    replies[0]["Write"] = [artifact(KEPT)]
    publishing = {"renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts"}
    run_id = sourced(store, **publishing).execute(single()).execution_id
    action = prepare(store, run_id)
    approval = approved(store, host, action, clock)
    lease = store.acquire_lease(run_id, "worker-1", ttl_s=TTL)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    store.release_lease(lease)

    remove(store, run_id, **publishing)

    lease = store.acquire_lease(run_id, "worker-1", ttl_s=TTL)
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        claim_dispatch(store, lease, attempt["attempt_id"])
    assert [entry["state"] for entry in item(store, action)["attempts"]] == ["released"]
    with pytest.raises(ActionNotAuthorized, match="content unavailable"):
        begin_attempt(store, lease, action.action_id, approval.approval_id)
    assert balances(item(store, action)) == (0, 0, 0)


@pytest.mark.parametrize("reason", ["budget_overrun", "unknown_exposure"])
def test_model_call_latches_do_not_block_actions(
        store, tmp_path, replies, host, clock, connector, reason):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    # These latches close admission to model calls; actions keep their own accounting.
    execute(store.path, "UPDATE workflow_runs SET blocked_reason=? WHERE run_id=?", (reason, action.run_id))

    assert dispatch(store, lease, action, approval, connector)["state"] == "acknowledged"


# ---------------------------------------------------------------------------
# Lease fencing
# ---------------------------------------------------------------------------


def test_a_superseded_lease_session_cannot_claim_or_record(
        store, tmp_path, replies, host, clock, service, connector):  # noqa: F811
    action, approval, first = ready(store, tmp_path, replies, host, clock)
    attempt = begin_attempt(store, first, action.action_id, approval.approval_id)
    permit = claim_dispatch(store, first, attempt["attempt_id"])
    # The first worker stalls past its lease, and a second takes the run over.
    clock.advance(TTL + 1)
    second = store.acquire_lease(action.run_id, "worker-2", ttl_s=TTL)
    assert second.epoch == first.epoch + 1
    for stale in (lambda: begin_attempt(store, first, action.action_id, approval.approval_id),
                  lambda: claim_dispatch(store, first, attempt["attempt_id"]),
                  lambda: recover_actions(store, first)):
        with pytest.raises(WorkflowLeaseError, match="expired or was superseded"):
            stale()

    # The stalled worker's request still reaches the service, which commits it.
    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)
    with pytest.raises(WorkflowLeaseError, match="since superseded; its outcome is kept"):
        record_receipt(store, permit, receipt)

    current = item(store, action)
    assert current["attempts"][0]["state"] == "dispatched"
    (late,) = current["records"]
    assert (late["kind"], late["trusted"], late["cost_nanousd"]) == ("late_receipt", True, None)
    (row,) = rows(store.path, "workflow_action_records")
    assert json.loads(row["payload_json"]) == {
        "attempt_id": attempt["attempt_id"], "dispatch_epoch": first.epoch, "outcome": "acknowledged",
        "receipt": receipt.to_dict(), "error": None}
    # The successor recovers the attempt, which may have taken effect.
    assert recover_actions(store, second) == {"released": [], "outcome_unknown": [attempt["attempt_id"]]}
    with pytest.raises(WorkflowLeaseError, match="its outcome is kept"):
        record_refused(store, permit, ConnectorRefused("refused late"))

    current = item(store, action)
    assert (current["state"], balances(current)) == ("outcome_unknown", (0, 0, WRITE))
    assert [record["kind"] for record in current["records"]] == ["late_receipt"] * 2
    assert current["approvals"][0]["consumed_by_attempt"] is None
    late_events = [event for event in store.inspect_run(action.run_id)["events"]
                   if event["type"] == "action_late_receipt"]
    assert [event["data"]["record_id"] for event in late_events] == [
        record["record_id"] for record in current["records"]]
    assert service.snapshot()["totals"]["committed"] == 1
    assert store.audit(action.run_id)["ok"]


def test_an_expired_lease_that_no_one_took_over_still_records(
        store, tmp_path, replies, host, clock, connector):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    permit = claim_dispatch(store, lease, attempt["attempt_id"])
    clock.advance(TTL + 1)
    receipt = connector.submit(action.action_key, action.intent, document(store, action), 30)

    # No later session has started, so the epoch still fences nothing out.
    assert record_receipt(store, permit, receipt)["state"] == "acknowledged"
    assert item(store, action)["records"] == []


# ---------------------------------------------------------------------------
# Crashes at each boundary
# ---------------------------------------------------------------------------


def _event(state):
    return lambda store, connector: (store, "_event", lambda db, run_id, kind, **values: (
        kind == "action_attempt" and values["payload"]["state"] == state))


# Each crash point: what to patch, the attempt state recovery leaves, and whether the service committed.
BOUNDARIES = {
    "inside-begin": (_event("prepared"), None, 0),
    "before-claim": (lambda store, connector: (workflow_actions, "claim_dispatch", None), "released", 0),
    "inside-claim": (_event("dispatched"), "released", 0),
    "before-submission": (lambda store, connector: (connector, "submit", None), "outcome_unknown", 0),
    "before-receipt": (lambda store, connector: (workflow_actions, "record_receipt", None),
                       "outcome_unknown", 1),
    "inside-receipt": (_event("acknowledged"), "outcome_unknown", 1),
    "after-receipt": (lambda store, connector: (store, "release_lease", None), "acknowledged", 1),
}


@pytest.mark.parametrize("boundary", list(BOUNDARIES))
def test_a_crash_at_each_boundary_recovers_safely(
        store, tmp_path, replies, host, clock, service, connector, monkeypatch, boundary):  # noqa: F811
    point, recovered_state, committed = BOUNDARIES[boundary]
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    image = tmp_path / "image.db"
    target, name, due = point(store, connector)
    crash = crash_when(monkeypatch, target, name, due, image=image, connection=store._db)

    with pytest.raises(Crash):
        dispatch(store, lease, action, approval, connector)
        store.release_lease(lease)
    assert crash.fired
    crash.restore()

    # A new process opens what the lost one committed, once its lease has lapsed.
    clock.advance(TTL + 1)
    with SQLiteWorkflowStore(image, clock_ns=clock) as journal:
        assert journal.audit(action.run_id)["ok"]
        successor = journal.acquire_lease(action.run_id, "worker-2", ttl_s=TTL)
        result = recover_actions(journal, successor)
        assert journal.audit(action.run_id)["ok"]
        current = item(journal, action)
        assert [attempt["state"] for attempt in current["attempts"]] == (
            [] if recovered_state is None else [recovered_state])
        attempt_ids = [attempt["attempt_id"] for attempt in current["attempts"]]
        assert result == {"released": attempt_ids if recovered_state == "released" else [],
                          "outcome_unknown": attempt_ids if recovered_state == "outcome_unknown" else []}
        assert service.snapshot()["totals"]["committed"] == committed
        if recovered_state in (None, "released"):
            # Nothing was sent: the reservation and the attempt allowance are restored.
            assert (current["state"], balances(current)) == ("not_dispatched", (0, 0, 0))
            assert dispatch(journal, successor, action, approval, connector)["state"] == "acknowledged"
            assert journal.audit(action.run_id)["ok"]
        elif recovered_state == "outcome_unknown":
            # The write may have been sent: its exposure is kept and nothing resends it.
            assert (current["state"], balances(current)) == ("outcome_unknown", (0, 0, WRITE))
            assert current["approvals"][0]["consumed_by_attempt"] is None
            error = rows(image, "workflow_action_attempts")[0]["error"]
            assert error == "recovered after dispatch"
            with pytest.raises(ActionStateError, match="is outcome_unknown"):
                begin_attempt(journal, successor, action.action_id, approval.approval_id)
        else:
            assert (current["state"], balances(current)) == ("dispatched", (0, 0, WRITE))
            assert current["approvals"][0]["consumed_by_attempt"] == current["attempts"][0]["attempt_id"]
        assert recover_actions(journal, successor) == {"released": [], "outcome_unknown": []}
        assert journal.audit(action.run_id)["ok"]
    # Recovery sent nothing; only a dispatch after a safe release did.
    assert service.snapshot()["totals"]["committed"] == (
        1 if recovered_state in (None, "released") else committed)


# ---------------------------------------------------------------------------
# The audit of the dispatch journal
# ---------------------------------------------------------------------------


def _two_in_flight(path, action, attempt):
    execute(path, "INSERT INTO workflow_action_attempts(attempt_id,action_id,number,approval_id,state,reserved,"
            "unknown,created,updated) VALUES ('second',?,2,?,'prepared',?,'0',0,0)",
            (action.action_id, attempt["approval_id"], str(WRITE)))
    execute(path, "UPDATE workflow_actions SET reserved=? WHERE action_id=?", (str(2 * WRITE), action.action_id))
    return "attempt is inconsistent"


def _prepared_under_dispatch(path, action, attempt):
    execute(path, "UPDATE workflow_actions SET state='dispatched' WHERE action_id=?", (action.action_id,))
    return "attempt is inconsistent"


def _consumed_by_refusal(path, action, attempt):
    execute(path, "UPDATE workflow_action_attempts SET state='refused',dispatch_token_sha='x',dispatch_epoch=1,"
            "dispatched=0,reserved='0',cost='0' WHERE attempt_id=?", (attempt["attempt_id"],))
    execute(path, "UPDATE workflow_actions SET reserved='0' WHERE action_id=?", (action.action_id,))
    execute(path, "UPDATE workflow_action_approvals SET consumed_by_attempt=? WHERE action_id=?",
            (attempt["attempt_id"], action.action_id))
    return "approval is inconsistent"


def _exposure_beside_a_cost(path, action, attempt):
    execute(path, "UPDATE workflow_action_attempts SET state='acknowledged',dispatch_token_sha='x',"
            "dispatch_epoch=1,dispatched=0,reserved='0',cost='5',unknown=? WHERE attempt_id=?",
            (str(WRITE), attempt["attempt_id"]))
    execute(path, "UPDATE workflow_actions SET state='dispatched',reserved='0',confirmed='5',unknown=? "
            "WHERE action_id=?", (str(WRITE), action.action_id))
    return "attempt is inconsistent"


TAMPERING = {
    "two-in-flight": _two_in_flight,
    "prepared-under-dispatch": _prepared_under_dispatch,
    "consumed-by-refusal": _consumed_by_refusal,
    "exposure-beside-a-cost": _exposure_beside_a_cost,
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_the_audit_detects_an_inconsistent_dispatch_journal(
        store, tmp_path, replies, host, clock, case):  # noqa: F811
    action, approval, lease = ready(store, tmp_path, replies, host, clock)
    attempt = begin_attempt(store, lease, action.action_id, approval.approval_id)
    assert store.audit(action.run_id)["ok"]

    message = TAMPERING[case](store.path, action, attempt)

    for check in (lambda: store.audit(action.run_id), lambda: recover_actions(store, lease)):
        with pytest.raises(WorkflowCorruptionError, match=message):
            check()
