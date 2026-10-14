"""Store-side SQL for external actions in the workflow journal.

A version 2 journal keeps each run's external actions in four tables:

- ``workflow_actions``: one logical action, with its intent, connector
  profile, state and balances;
- ``workflow_action_attempts``: its attempts, numbered from 1, each under one
  of its approvals, with any dispatch claim and receipt;
- ``workflow_action_approvals``: approvals and rejections bound to its intent
  digest; and
- ``workflow_action_records``: append-only readbacks, lookups, notices, late
  receipts, findings, decisions and evidence.

Action money is the ``external_actions`` accounting scope: whole nanoUSD kept
on these rows, never in a run's balances, which stay equal to its call ledger.
:func:`audit_actions` runs inside a ``SQLiteWorkflowStore`` transaction; the
other functions open their own. Preparation, verdicts, revocations, a run's
action stop, notices and an operator's decisions with their evidence need no
run lease and are called by ``smythe.actions``. The dispatch
journal, :func:`begin_attempt` to :func:`recover_actions`, the runner's
releases and readback records, and reconciliation's lookups, same-key
retries and findings are fenced by the run's lease.
``import smythe`` does not load this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import secrets
from uuid import uuid4

from smythe._money import NANOUSD
from smythe._redact import redact_account_identifiers
from smythe.actions import (
    _CLOSED, _DECISIONS, _FINISHED, _NOTICES, _SECURITY_STOP, _SHOWN, ACTION_STATES, ActionIntent, ActionNotAuthorized,
    ActionStateError, ApprovalRejected, ConnectorProfile, ConnectorReceipt, PreparedAction, Principal,
    _held_digest, _next_action, _receipt_of, _undeliverable, _window_open, action_key,
)
from smythe.workflow_store import (
    WorkflowBudgetError, WorkflowConflictError, WorkflowCorruptionError, WorkflowLeaseError,
    WorkflowNotFoundError, WorkflowValidationError, _amount, _canonical, _loads, _sha,
)

ACCOUNTING_SCOPE = "external_actions"
ATTEMPT_STATES = ("prepared", "dispatched", "acknowledged", "refused", "outcome_unknown", "released")
APPROVAL_VERDICTS = ("approved", "rejected")
RECORD_KINDS = ("readback", "lookup", "notice", "late_receipt", "finding", "decision", "evidence")
# Attempt states after a dispatch claim, whose request may have reached the service.
_CLAIMED = frozenset({"dispatched", "acknowledged", "refused", "outcome_unknown"})
# Attempt states that can consume an approval: a refused attempt never does.
_EFFECTS = _CLAIMED - {"refused"}
_BALANCES = ("confirmed_nanousd", "reserved_nanousd", "unknown_nanousd")


def _check(valid, message):
    if not valid:
        raise WorkflowCorruptionError(message)


def _money(value):
    return None if value is None else _amount(value)


def _stored(text, read=None):
    """Canonical stored JSON, as ``read`` makes it into an object if given."""
    value = _loads(text, stored=True)
    if read is not None:
        try:
            value = read(value)
        except (TypeError, ValueError) as exc:
            raise WorkflowCorruptionError("Stored external action data is invalid") from exc
    _check(_canonical(value if read is None else value.to_dict()) == text,
           "Stored external action JSON is not canonical")
    return value


def audit_actions(store, db, run_id):
    """Check a run's external actions in ``db``'s transaction; return their safe projection.

    Each action's intent must hash to its recorded digest and name its
    profile's connector and one of its operations, and its key must be
    ``action_key(store_id, run_id, slot, sequence)``. Its balances must equal
    its attempts' costs, reservations and unknown exposure plus its records'
    costs. Every state, verdict and record kind must be known; attempts are
    numbered from 1, each under one of the action's approvals; at most one is
    prepared or dispatched, and its action is then not dispatched or
    dispatched to match; and approvals bind the action's intent digest and
    are consumed only by an attempt that may have taken effect. Raises
    ``WorkflowCorruptionError``.

    The projection never includes parameters, assertions, receipts, record
    payloads or error text: only identities, digests, states and balances. A
    revoked approval also shows its ``revocation`` (see :func:`_revocation`).
    """
    rows = db.execute("SELECT * FROM workflow_actions WHERE run_id=? ORDER BY slot,sequence",
                      (run_id,)).fetchall()
    run_actions = {row["action_id"] for row in rows}
    items = [_action(store, db, row, run_actions) for row in rows]
    return {"accounting_scope": ACCOUNTING_SCOPE, "unit": NANOUSD,
            **{name: sum(item[name] for item in items) for name in _BALANCES}, "items": items}


def _action(store, db, row, run_actions):
    action_id = row["action_id"]
    intent = _stored(row["intent_json"], ActionIntent.from_dict)
    profile = _stored(row["profile_json"], ConnectorProfile.from_dict)
    try:
        key = action_key(store.store_id, row["run_id"], row["slot"], row["sequence"])
    except ValueError:
        key = None
    _check(intent.digest() == row["intent_sha"] and key == row["action_key"]
           and (intent.connector, intent.connector_version) == (profile.name, profile.version)
           and intent.operation in profile.operations and row["state"] in ACTION_STATES
           and row["compensates_action_id"] in (run_actions - {action_id}) | {None},
           "External action identity or state is invalid")
    approvals = db.execute("SELECT * FROM workflow_action_approvals WHERE action_id=? ORDER BY rowid",
                           (action_id,)).fetchall()
    attempts = db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? ORDER BY number",
                          (action_id,)).fetchall()
    approved = {approval["approval_id"] for approval in approvals if approval["verdict"] == "approved"}
    sums, attempt_items, in_flight = [0, 0, 0], [], 0
    for number, attempt in enumerate(attempts, 1):
        item = _attempt_item(attempt)
        state, claimed, epoch = item["state"], attempt["dispatch_token_sha"] is not None, item["dispatch_epoch"]
        cost, held, unknown = item["cost_nanousd"], item["reserved_nanousd"], item["unknown_nanousd"]
        receipt = attempt["receipt_json"]
        if receipt is not None:
            _stored(receipt)
        in_flight += state in ("prepared", "dispatched")
        _check(attempt["number"] == number and state in ATTEMPT_STATES and attempt["approval_id"] in approved
               and claimed == (state in _CLAIMED) == (attempt["dispatched"] is not None)
               and ((type(epoch) is int and epoch > 0) if claimed else epoch is None)
               and (None if receipt is None else _sha(receipt)) == attempt["receipt_sha"]
               and (not held or state in ("prepared", "dispatched"))
               # Exposure stays on an attempt that may have taken effect at an unknown cost.
               and (not unknown or cost is None and state in ("outcome_unknown", "acknowledged"))
               and in_flight <= 1 and (state != "prepared" or row["state"] == "not_dispatched")
               and (state != "dispatched" or row["state"] == "dispatched"),
               "External action attempt is inconsistent")
        sums = [sums[0] + (cost or 0), sums[1] + held, sums[2] + unknown]
        attempt_items.append(item)
    approval_items = []
    for approval in approvals:
        principal = _stored(approval["principal_json"], Principal.from_dict)
        verifier, consumed = _stored(approval["verifier_json"]), approval["consumed_by_attempt"]
        _check(approval["intent_sha"] == row["intent_sha"] and approval["verdict"] in APPROVAL_VERDICTS
               and _sha(approval["assertion"]) == approval["assertion_sha"] and type(verifier) is dict
               and (consumed is None or any(attempt["attempt_id"] == consumed
                                            and attempt["approval_id"] == approval["approval_id"]
                                            and attempt["state"] in _EFFECTS for attempt in attempts))
               and all(approval[name] is None or type(approval[name]) is int
                       for name in ("expires_at_ns", "revoked_at_ns"))
               and (approval["revoked_at_ns"] is None) == (approval["revoked_reason"] is None)
               and approval["duplicate_risk"] in (0, 1), "External action approval is inconsistent")
        approval_items.append({
            "approval_id": approval["approval_id"], "verdict": approval["verdict"],
            "intent_sha256": approval["intent_sha"], "principal": principal.to_dict(), "verifier": verifier,
            "ceiling_nanousd": _money(approval["ceiling"]), "expires_at_ns": approval["expires_at_ns"],
            "consumed_by_attempt": consumed, "revoked_at_ns": approval["revoked_at_ns"],
            "duplicate_risk": approval["duplicate_risk"] == 1,
            **({} if approval["revoked_at_ns"] is None else {"revocation": _revocation(db, approval, attempts)})})
    record_items = []
    for record in db.execute("SELECT * FROM workflow_action_records WHERE action_id=? ORDER BY rowid",
                             (action_id,)):
        _stored(record["payload_json"])
        cost = _money(record["cost"])
        _check(record["kind"] in RECORD_KINDS and _sha(record["payload_json"]) == record["payload_sha"]
               and record["trusted"] in (0, 1), "External action record is inconsistent")
        sums[0] += cost or 0
        record_items.append({"record_id": record["record_id"], "kind": record["kind"],
                             "payload_sha256": record["payload_sha"], "trusted": record["trusted"] == 1,
                             "cost_nanousd": cost})
    balances = [_amount(row[name]) for name in ("confirmed", "reserved", "unknown")]
    _check(balances == sums, "External action balances disagree with its attempts and records")
    return {"action_id": action_id, "slot": row["slot"], "sequence": row["sequence"],
            "action_key": row["action_key"], "intent_sha256": row["intent_sha"],
            "operation": intent.operation, "target": intent.target.to_dict(),
            "connector": profile.name, "connector_version": profile.version, "state": row["state"],
            "compensates_action_id": row["compensates_action_id"], "ceiling_nanousd": _amount(row["ceiling"]),
            **dict(zip(_BALANCES, balances)),
            "attempts": attempt_items, "approvals": approval_items, "records": record_items}


def _revocation(db, approval, attempts):
    """When a revoked approval was revoked, as the journal shows it.

    ``before_dispatch``: no attempt under it may have taken effect.
    ``after_dispatch``: one had been dispatched, since a claim checks
    revocation in its own transaction, and its outcome is unknown.
    ``after_effect``: a receipt, the attempt's own or one a lookup found,
    shows that a write under it took effect, which revoking never undoes.
    """
    sent = {attempt["attempt_id"] for attempt in attempts
            if attempt["approval_id"] == approval["approval_id"] and attempt["state"] in _EFFECTS}
    if not sent:
        return "before_dispatch"
    shown = approval["consumed_by_attempt"] is not None or sent & _found(db, approval["action_id"]).keys()
    return "after_effect" if shown else "after_dispatch"


def _attempt_item(attempt):
    """An attempt's safe projection: identities, state, receipt digest and balances."""
    return {"attempt_id": attempt["attempt_id"], "number": attempt["number"],
            "approval_id": attempt["approval_id"], "state": attempt["state"],
            "dispatch_epoch": attempt["dispatch_epoch"], "receipt_sha256": attempt["receipt_sha"],
            "cost_nanousd": _money(attempt["cost"]), "reserved_nanousd": _amount(attempt["reserved"]),
            "unknown_nanousd": _amount(attempt["unknown"])}


def _row(db, action_id):
    row = db.execute("SELECT * FROM workflow_actions WHERE action_id=?", (action_id,)).fetchone()
    if row is None:
        raise WorkflowNotFoundError(action_id)
    return row


def _prepared(row):
    return PreparedAction(row["action_id"], row["run_id"], row["slot"], row["sequence"], row["action_key"],
                          _stored(row["intent_json"], ActionIntent.from_dict),
                          _stored(row["profile_json"], ConnectorProfile.from_dict), row["state"],
                          _amount(row["ceiling"]), row["compensates_action_id"])


def load(store, action_id):
    """An action, after its run's actions pass the audit; its approval rows, newest first; its risk.

    The risk is the number of attempts an operator's latest ``not_executed``
    decision found, or None without one (see :func:`_risk`).
    """
    if store.store_version < 2:
        raise WorkflowNotFoundError(action_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        row = _row(db, action_id)
        audit_actions(store, db, row["run_id"])
        return _prepared(row), db.execute("SELECT * FROM workflow_action_approvals WHERE action_id=? "
                                          "ORDER BY rowid DESC", (action_id,)).fetchall(), _risk(db, action_id)


def prepare(store, run_id, slot, intent, profile, compensates=None):
    """Record *intent* as the next action in a run's *slot*, or return the slot's identical one.

    Raises ``WorkflowConflictError`` while the slot's latest action is
    unfinished and has another intent or profile. With *compensates*, the
    action compensates that action of the run, which a read must show took
    effect, or ``ActionStateError`` is raised.
    """
    intent_json, profile_json = _canonical(intent.to_dict()), _canonical(profile.to_dict())
    ceiling = (profile.max_attempts * profile.write_ceiling_nanousd
               + profile.readback_reads * profile.read_ceiling_nanousd)
    with store._transaction() as db:
        store._run(db, run_id)
        original = None if compensates is None else _row(db, compensates)
        if original is not None and (original["run_id"] != run_id or original["state"] not in _SHOWN):
            raise ActionStateError(f"Action {compensates!r} has no effect of this run that a read has shown")
        latest = db.execute("SELECT * FROM workflow_actions WHERE run_id=? AND slot=? ORDER BY sequence DESC "
                            "LIMIT 1", (run_id, slot)).fetchone()
        if latest is not None and (latest["intent_json"], latest["profile_json"], latest["compensates_action_id"]) == (
                intent_json, profile_json, compensates):
            return _prepared(latest)
        if latest is not None and latest["state"] not in _FINISHED:
            raise WorkflowConflictError(f"Slot {slot!r} holds an unfinished action with another intent or "
                                        "profile; a changed intent is a new action, never a retry")
        sequence, action_id, now = 1 if latest is None else latest["sequence"] + 1, uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_actions(action_id,run_id,slot,sequence,action_key,intent_json,intent_sha,
          profile_json,state,ceiling,confirmed,reserved,unknown,compensates_action_id,created,updated)
          VALUES (?,?,?,?,?,?,?,?,'not_dispatched',?,'0','0','0',?,?,?)""",
                   (action_id, run_id, slot, sequence, action_key(store.store_id, run_id, slot, sequence),
                    intent_json, intent.digest(), profile_json, str(ceiling), compensates, now, now))
        return _prepared(_row(db, action_id))


def effect_target(store, action_id):
    """An action, and the object the read that settled it read, or None without such a read.

    The object is the target of the receipt that read checked: its attempt's
    own, or one a lookup found. Only reads are made, after the audit.
    """
    if store.store_version < 2:
        raise WorkflowNotFoundError(action_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        row = _row(db, action_id)
        audit_actions(store, db, row["run_id"])
        reads, receipt = _payloads(db, action_id, "readback"), None
        if reads:
            attempt_id = reads[-1][1].get("attempt_id")
            attempt = db.execute("SELECT receipt_json FROM workflow_action_attempts WHERE attempt_id=? "
                                 "AND action_id=?", (attempt_id, action_id)).fetchone()
            found = _found(db, action_id).get(attempt_id)
            text = attempt and attempt["receipt_json"] or found and _canonical(found)
            receipt = text and _stored(text, ConnectorReceipt.from_dict)
        return _prepared(row), receipt.target if receipt else None


def record_verdict(store, action, verdict, principal, identity, assertion, ceiling, expires_at_ns, risk=None):
    """Record a verified approval or rejection of *action*; return its row.

    A rejection cancels the action and releases, with its reservation, an
    attempt prepared but not claimed: nothing was sent. An approval with a
    *risk* accepts the duplicate risk an operator's ``not_executed`` decision
    found at that many attempts, which must still be the action's. An
    assertion is recorded once in a journal: the same verdict on the same
    action again returns its row, and anything else raises
    ``ApprovalRejected``.
    """
    assertion_sha, ceiling = _sha(assertion), None if ceiling is None else str(ceiling)
    duplicate_risk = int(risk is not None)
    with store._transaction() as db:
        row = _row(db, action.action_id)
        if row["intent_sha"] != action.intent.digest():
            raise WorkflowConflictError("The action's recorded intent changed")
        if row["state"] in _CLOSED or verdict == "rejected" and row["state"] != "not_dispatched":
            raise ActionStateError(f"Action {action.action_id!r} is {row['state']}; it cannot be {verdict}")
        if verdict == "approved" and _risk(db, action.action_id) != risk:
            raise WorkflowConflictError("The action's duplicate risk changed while it was approved; approve "
                                        "it again")
        existing = db.execute("SELECT * FROM workflow_action_approvals WHERE assertion_sha=?",
                              (assertion_sha,)).fetchone()
        if existing is not None:
            if ((existing["action_id"], existing["verdict"], existing["ceiling"], existing["expires_at_ns"],
                 existing["revoked_at_ns"], existing["duplicate_risk"]) != (
                    action.action_id, verdict, ceiling, expires_at_ns, None, duplicate_risk)):
                raise ApprovalRejected("This assertion is already recorded in the journal; each approval "
                                       "needs its own assertion")
            return existing
        approval_id, now = uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_action_approvals(approval_id,action_id,intent_sha,verdict,principal_json,
          verifier_json,assertion,assertion_sha,ceiling,expires_at_ns,duplicate_risk,created)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (approval_id, action.action_id, row["intent_sha"], verdict, _canonical(principal.to_dict()),
                    _canonical({"identity": identity}), assertion, assertion_sha, ceiling, expires_at_ns,
                    duplicate_risk, now))
        if verdict == "rejected":
            for attempt in db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? "
                                      "AND state='prepared'", (action.action_id,)).fetchall():
                _finish(store, db, attempt, "released", error="the action was rejected")
            db.execute("UPDATE workflow_actions SET state='canceled',updated=? WHERE action_id=?",
                       (now, action.action_id))
        return db.execute("SELECT * FROM workflow_action_approvals WHERE approval_id=?", (approval_id,)).fetchone()


def revoke_approvals(store, target_id, reason):
    """Revoke the approval *target_id*, or every approval of the action *target_id*.

    Returns the IDs of the approvals this call revoked; an earlier
    revocation is kept as it was.
    """
    with store._transaction() as db:
        if db.execute("SELECT 1 FROM workflow_action_approvals WHERE approval_id=?", (target_id,)).fetchone():
            column = "approval_id"
        else:
            column = "action_id"
            _row(db, target_id)
        where = f"WHERE {column}=? AND verdict='approved' AND revoked_at_ns IS NULL"
        revoked = tuple(item[0] for item in db.execute(
            f"SELECT approval_id FROM workflow_action_approvals {where} ORDER BY rowid", (target_id,)))
        db.execute(f"UPDATE workflow_action_approvals SET revoked_at_ns=?,revoked_reason=? {where}",
                   (store._clock_ns(), reason, target_id))
        return revoked


def stop_run(store, run_id, reason):
    """Revoke every approval of a run's actions and record an ``actions_stopped`` event.

    Returns the IDs of the approvals this call revoked, which the event
    lists; an earlier revocation is kept as it was.
    """
    with store._transaction() as db:
        store._run(db, run_id)
        where = ("WHERE verdict='approved' AND revoked_at_ns IS NULL AND action_id IN "
                 "(SELECT action_id FROM workflow_actions WHERE run_id=?)")
        revoked = tuple(item[0] for item in db.execute(
            f"SELECT approval_id FROM workflow_action_approvals {where} ORDER BY rowid", (run_id,)))
        db.execute(f"UPDATE workflow_action_approvals SET revoked_at_ns=?,revoked_reason=? {where}",
                   (store._clock_ns(), reason, run_id))
        store._event(db, run_id, "actions_stopped", payload={"revoked": list(revoked)})
        return revoked


def _authorizing(action_id, state, intent_sha, ceiling, approvals, now, approval_id=None, risk=None):
    """The approval row that authorizes an action now, as its journal rows alone say.

    *approvals* are the action's approval rows, newest first. The action must
    not be finished or need compensation, and its newest approval, which must
    be *approval_id* when one is named, must be unrevoked, unconsumed,
    unexpired at *now*, bind *intent_sha*, cover *ceiling* and, with a
    duplicate *risk* (see :func:`_risk`), accept it. Raises
    ``ActionNotAuthorized``.
    """
    approval = next((row for row in approvals if row["verdict"] == "approved"), None)
    problem = (
        f"it is {state}" if state in _CLOSED
        else "it has no approval" if approval is None
        else f"approval {approval_id!r} is not its newest approval"
        if approval_id not in (None, approval["approval_id"])
        else "its approval was revoked" if approval["revoked_at_ns"] is not None
        else "its approval was consumed" if approval["consumed_by_attempt"] is not None
        else "its approval expired" if type(approval["expires_at_ns"]) is not int
        or approval["expires_at_ns"] <= now
        else "its approval binds another intent" if approval["intent_sha"] != intent_sha
        else "its approval does not cover the action's ceiling"
        if approval["ceiling"] is None or _amount(approval["ceiling"]) < ceiling
        else "its approval does not accept the duplicate risk an operator's not_executed decision left"
        if risk is not None and approval["duplicate_risk"] != 1 else None)
    if problem:
        raise ActionNotAuthorized(f"Action {action_id!r} is not authorized: {problem}")
    return approval


@dataclass(frozen=True, slots=True)
class ActionDispatchPermit:
    """The claim on one attempt's dispatch that records its outcome.

    ``epoch`` is the run lease's epoch at the claim. The journal keeps only
    the SHA-256 of ``token``.
    """

    attempt_id: str
    run_id: str
    token: str = field(repr=False)
    epoch: int


def _attempt(db, attempt_id):
    row = db.execute("SELECT * FROM workflow_action_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
    if row is None:
        raise WorkflowNotFoundError(attempt_id)
    return row


def _deliverable(store, action_id):
    """Why the content of *action_id* cannot be delivered now, as ``ActionNotAuthorized``, or None.

    These checks read the run's decision, sources and published file, so
    they run before the write transaction that checks the journal again.
    """
    action = load(store, action_id)[0]
    problem = _undeliverable(store, action.run_id, action.intent.content)
    return problem and ActionNotAuthorized(f"Action {action_id!r} is not authorized: {problem}")


def _owned(store, db, lease, action_id):
    """The run and action rows, when *lease* is the current lease of the action's run."""
    run, action = store._lease(db, lease), _row(db, action_id)
    if action["run_id"] != lease.run_id:
        raise WorkflowConflictError("The action belongs to another workflow run")
    return run, action


def _authorized(store, db, run, action, approval_id):
    """The approval row, once the journal shows that *approval_id* authorizes an attempt.

    Checks a security stop, the run's completion and the approval (see
    :func:`_authorizing`) inside the caller's write transaction.
    """
    stopped = run["blocked_reason"] == _SECURITY_STOP
    if stopped or run["status"] != "completed":
        problem = "a security stop holds the run" if stopped else "the run has not completed"
        raise ActionNotAuthorized(f"Action {action['action_id']!r} is not authorized: {problem}")
    approvals = db.execute("SELECT * FROM workflow_action_approvals WHERE action_id=? ORDER BY rowid DESC",
                           (action["action_id"],)).fetchall()
    risk = _risk(db, action["action_id"])
    approval = _authorizing(action["action_id"], action["state"], action["intent_sha"],
                            _amount(action["ceiling"]), approvals, store._clock_ns(), approval_id, risk)
    # A duplicate-risk approval accepts the attempts one decision found. One that
    # an attempt within that count was made under answered an earlier decision.
    if risk is not None and db.execute(
            "SELECT 1 FROM workflow_action_attempts WHERE action_id=? AND approval_id=? AND number<=?",
            (action["action_id"], approval["approval_id"], risk)).fetchone():
        raise ActionNotAuthorized(
            f"Action {action['action_id']!r} is not authorized: its approval accepted an earlier "
            "duplicate risk, not the one the latest not_executed decision left")
    return approval


def _note(store, db, action, attempt_id, state):
    store._event(db, action["run_id"], "action_attempt",
                 payload={"action_id": action["action_id"], "attempt_id": attempt_id, "state": state})


def _finish(store, db, attempt, state, *, cost=None, unknown=0, error=None, receipt_json=None,
            action_state=None):
    """Move an attempt in flight to *state*, giving back its reservation, and update its action."""
    held, now, action = _amount(attempt["reserved"]), store._clock_ns(), _row(db, attempt["action_id"])
    db.execute("""UPDATE workflow_action_attempts SET state=?,cost=?,reserved='0',unknown=?,error=?,
      receipt_json=?,receipt_sha=?,updated=? WHERE attempt_id=?""",
               (state, None if cost is None else str(cost), str(unknown), error, receipt_json,
                None if receipt_json is None else _sha(receipt_json), now, attempt["attempt_id"]))
    balances = [_amount(action[name]) + change
                for name, change in (("confirmed", cost or 0), ("reserved", -held), ("unknown", unknown))]
    db.execute("UPDATE workflow_actions SET confirmed=?,reserved=?,unknown=?,state=?,updated=? WHERE action_id=?",
               (*map(str, balances), action_state or action["state"], now, action["action_id"]))
    _note(store, db, action, attempt["attempt_id"], state)


def _error(error):
    """Connector error text as the journal keeps it: account identifiers redacted, bounded."""
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return redact_account_identifiers(text)[:500]


def _admit(db, action_id, approval, attempts):
    """The action row and the write ceiling one more attempt reserves, once its attempts and approval allow it."""
    action = _row(db, action_id)
    profile = _stored(action["profile_json"], ConnectorProfile.from_dict)
    if sum(attempt["state"] in _CLAIMED for attempt in attempts) >= profile.max_attempts:
        raise ActionStateError(f"Action {action_id!r} has made all {profile.max_attempts} of its attempts")
    ceiling = profile.write_ceiling_nanousd
    if sum(_amount(action[name]) for name in ("confirmed", "reserved", "unknown")) + ceiling > _amount(
            approval["ceiling"]):
        raise WorkflowBudgetError(f"Another attempt of action {action_id!r} could exceed its "
                                  "approval's ceiling")
    return action, ceiling


def begin_attempt(store, lease, action_id, approval_id):
    """Prepare an action's next attempt under its approval and reserve its write ceiling.

    The caller holds the run's lease. The run's content must still be
    deliverable (see ``smythe.actions.prepare_action``). In the write
    transaction that makes the reservation, the journal must show no
    security stop, a completed run, an action that is ``not_dispatched`` and
    *approval_id* as its newest approval, unrevoked, unconsumed and
    unexpired. Fewer than ``max_attempts`` attempts may have been
    dispatched, and the action's confirmed, reserved and unknown nanoUSD plus
    the write ceiling must fit the approval's ceiling. The verifier is not
    called: call ``smythe.actions.check_authority`` first.

    Returns the attempt's projection (see :func:`audit_actions`). A prepared
    attempt under the same approval is returned again, and one under an
    earlier approval is released. Raises ``ActionNotAuthorized``,
    ``ActionStateError``, ``WorkflowBudgetError`` or ``WorkflowLeaseError``.
    """
    refusal = _deliverable(store, action_id)
    if refusal:
        raise refusal
    with store._transaction() as db:
        run, action = _owned(store, db, lease, action_id)
        if action["state"] in ("dispatched", "outcome_unknown"):
            raise ActionStateError(f"Action {action_id!r} is {action['state']}; an attempt begins only "
                                   "when the action is not dispatched")
        approval = _authorized(store, db, run, action, approval_id)
        attempts = db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? ORDER BY number",
                              (action_id,)).fetchall()
        for attempt in attempts:
            if attempt["state"] == "prepared":
                if attempt["approval_id"] == approval_id:
                    return _attempt_item(attempt)
                _finish(store, db, attempt, "released", error="a newer approval superseded its own")
        action, ceiling = _admit(db, action_id, approval, attempts)
        attempt_id, now = uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_action_attempts(attempt_id,action_id,number,approval_id,state,
          reserved,unknown,created,updated) VALUES (?,?,?,?,'prepared',?,'0',?,?)""",
                   (attempt_id, action_id, len(attempts) + 1, approval_id, str(ceiling), now, now))
        db.execute("UPDATE workflow_actions SET reserved=?,updated=? WHERE action_id=?",
                   (str(_amount(action["reserved"]) + ceiling), now, action_id))
        _note(store, db, action, attempt_id, "prepared")
        return _attempt_item(_attempt(db, attempt_id))


def claim_dispatch(store, lease, attempt_id):
    """Claim the dispatch of a prepared attempt; return the permit that records its outcome.

    The claim commits the attempt and its action as ``dispatched``, with the
    lease's epoch and the time, before the caller sends anything, so a crash
    after it leaves an attempt that recovery treats as possibly sent. Each
    attempt is claimed once. The checks of :func:`begin_attempt` run again;
    when the approval or the content no longer authorizes the attempt, it is
    released with its reservation and ``ActionNotAuthorized`` is raised.
    """
    if store.store_version < 2:
        raise WorkflowNotFoundError(attempt_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        action_id = _attempt(db, attempt_id)["action_id"]
    refusal, token = _deliverable(store, action_id), secrets.token_hex(32)
    with store._transaction() as db:
        attempt, (run, action) = _attempt(db, attempt_id), _owned(store, db, lease, action_id)
        if attempt["state"] != "prepared":
            raise ActionStateError(f"Attempt {attempt_id!r} is {attempt['state']}; only a prepared attempt "
                                   "can claim dispatch")
        try:
            if refusal:
                raise refusal
            _authorized(store, db, run, action, attempt["approval_id"])
        except ActionNotAuthorized as error:
            _finish(store, db, attempt, "released", error=str(error))
            refusal = error
        else:
            now = store._clock_ns()
            db.execute("""UPDATE workflow_action_attempts SET state='dispatched',dispatch_token_sha=?,
              dispatch_epoch=?,dispatched=?,updated=? WHERE attempt_id=?""",
                       (_sha(token), lease.epoch, now, now, attempt_id))
            db.execute("UPDATE workflow_actions SET state='dispatched',updated=? WHERE action_id=?",
                       (now, action_id))
            _note(store, db, action, attempt_id, "dispatched")
    if refusal:
        raise refusal
    return ActionDispatchPermit(attempt_id, lease.run_id, token, lease.epoch)


def _record(store, permit, state, receipt=None, error=None):
    """Record the outcome of a claimed attempt, or keep it as a late receipt record."""
    if not isinstance(permit, ActionDispatchPermit):
        raise WorkflowValidationError("Expected an action dispatch permit")
    receipt_json, late = None if receipt is None else _canonical(receipt.to_dict()), None
    with store._transaction() as db:
        attempt = _attempt(db, permit.attempt_id)
        action = _row(db, attempt["action_id"])
        if (action["run_id"], attempt["dispatch_token_sha"], attempt["dispatch_epoch"]) != (
                permit.run_id, _sha(permit.token), permit.epoch):
            raise WorkflowConflictError("The permit does not match the attempt's dispatch claim")
        if receipt is not None and not _receipt_of(
                action["action_key"], _stored(action["intent_json"], ActionIntent.from_dict), receipt):
            raise WorkflowValidationError("The receipt names another key or target than the action's; "
                                          "record the attempt as ambiguous")
        if store._run(db, permit.run_id)["epoch"] != permit.epoch:
            late = WorkflowLeaseError(f"Attempt {permit.attempt_id!r} was claimed under lease epoch "
                                      f"{permit.epoch}, since superseded; its outcome is kept as a late "
                                      "receipt record")
        elif attempt["state"] != "dispatched":
            if (attempt["state"], attempt["receipt_json"], attempt["error"]) == (state, receipt_json, error):
                return _attempt_item(attempt)  # The same outcome again.
            late = ActionStateError(f"Attempt {permit.attempt_id!r} is {attempt['state']}; its outcome is "
                                    "kept as a late receipt record")
        if late:
            payload = _canonical({"attempt_id": permit.attempt_id, "dispatch_epoch": permit.epoch,
                                  "outcome": state, "receipt": receipt and receipt.to_dict(), "error": error})
            record_id = uuid4().hex
            db.execute("""INSERT INTO workflow_action_records(record_id,action_id,kind,payload_json,payload_sha,
              trusted,created) VALUES (?,?,'late_receipt',?,?,1,?)""",
                       (record_id, action["action_id"], payload, _sha(payload), store._clock_ns()))
            store._event(db, permit.run_id, "action_late_receipt", payload={
                "action_id": action["action_id"], "attempt_id": permit.attempt_id, "record_id": record_id})
            _contradict(store, db, action, attempt, record_id, receipt)
        elif state == "acknowledged":
            # Without a reported cost, only a write declared free has a known one.
            held = _amount(attempt["reserved"])
            cost = receipt.cost_nanousd if receipt.cost_nanousd is not None else None if held else 0
            _finish(store, db, attempt, state, cost=cost, unknown=held if cost is None else 0,
                    receipt_json=receipt_json)
            db.execute("UPDATE workflow_action_approvals SET consumed_by_attempt=? WHERE approval_id=? "
                       "AND consumed_by_attempt IS NULL", (permit.attempt_id, attempt["approval_id"]))
        elif state == "refused":
            # A refused retry says nothing of the earlier attempt whose outcome is still unknown.
            _finish(store, db, attempt, state, cost=0, error=error,
                    action_state="outcome_unknown" if _unsettled(db, action["action_id"]) else "not_dispatched")
        else:
            _finish(store, db, attempt, state, unknown=_amount(attempt["reserved"]), error=error,
                    action_state="outcome_unknown")
        result = _attempt_item(_attempt(db, permit.attempt_id))
    if late:
        raise late
    return result


def record_receipt(store, permit, receipt):
    """Record the service's receipt for a dispatched attempt, which becomes ``acknowledged``.

    The receipt must name the action's key, its target's tenant and
    collection and, when the intent names an object, that object; otherwise
    ``WorkflowValidationError`` is raised and nothing changes. The receipt is
    kept with the external object's identity and consumes the attempt's
    approval. Its cost is confirmed; without one, a write whose ceiling is
    zero is confirmed free, and any other keeps its reservation as unknown
    exposure. The action stays ``dispatched`` until a read confirms the
    result. Returns the attempt's projection; the same receipt again returns
    it.

    No lease is needed, but the permit's lease epoch must still be the run's.
    A permit from a superseded lease session, or for an attempt whose outcome
    is already recorded or recovered, changes nothing: its outcome is kept as
    a ``late_receipt`` record, then ``WorkflowLeaseError`` or
    ``ActionStateError`` is raised. A late receipt that shows a commit of an
    attempt found not executed also appends a contradicting finding (see
    :func:`_contradict`).
    """
    if not isinstance(receipt, ConnectorReceipt):
        raise WorkflowValidationError("Expected a ConnectorReceipt")
    return _record(store, permit, "acknowledged", receipt=receipt)


def record_refused(store, permit, error):
    """Record that the service definitely did not commit a dispatched attempt.

    The attempt is ``refused`` at zero cost, its reservation is given back
    and the action is ``not_dispatched`` again, its approval unconsumed. A
    refused same-key retry (see :func:`claim_retry`) leaves the action
    ``outcome_unknown`` instead, since it says nothing of the earlier
    attempt. The refusal still counts toward ``max_attempts``. *error* is
    kept with account identifiers redacted. Permits are fenced as for
    :func:`record_receipt`.
    """
    return _record(store, permit, "refused", error=_error(error))


def record_ambiguous(store, permit, error):
    """Record that a dispatched attempt may or may not have taken effect.

    The attempt and its action become ``outcome_unknown``, and the write
    ceiling it reserved becomes unknown exposure. :func:`begin_attempt` then
    refuses the action: only reconciliation makes another attempt, a
    same-key retry (:func:`claim_retry`), or allows one, by finding that the
    write was not made (:func:`record_finding`). The exposure stays until a
    receipt that a lookup finds reports the write's cost (see
    :func:`record_lookup`). *error* is kept with account identifiers
    redacted. Permits are fenced as for :func:`record_receipt`.
    """
    return _record(store, permit, "outcome_unknown", error=_error(error))


def recover_actions(store, lease):
    """Settle the attempts a stopped worker left in flight; call it before dispatching.

    The caller holds the run's lease, which a stopped worker's must have
    expired for. Every attempt in flight is treated as abandoned. A prepared
    attempt was never sent: it is ``released`` with its reservation and does
    not count toward ``max_attempts``. A dispatched attempt without an
    outcome may have reached the service: it and its action become
    ``outcome_unknown``, with its write ceiling as unknown exposure and the
    reason ``recovered after dispatch``. Nothing is sent and no exposure is
    released. Returns the ``released`` and ``outcome_unknown`` attempt IDs.
    """
    result = {"released": [], "outcome_unknown": []}
    with store._transaction() as db:
        store._lease(db, lease)
        audit_actions(store, db, lease.run_id)
        for attempt in db.execute("""SELECT attempt.* FROM workflow_action_attempts AS attempt
          JOIN workflow_actions AS action USING(action_id) WHERE action.run_id=?
          AND attempt.state IN ('prepared','dispatched') ORDER BY action.slot,action.sequence""",
                                  (lease.run_id,)).fetchall():
            if attempt["state"] == "prepared":
                _finish(store, db, attempt, "released", error="recovered before dispatch")
                result["released"].append(attempt["attempt_id"])
            else:
                _finish(store, db, attempt, "outcome_unknown", unknown=_amount(attempt["reserved"]),
                        error="recovered after dispatch", action_state="outcome_unknown")
                result["outcome_unknown"].append(attempt["attempt_id"])
    return result


# The runner's journal: pending actions, releases before a claim, readback records and untrusted notices.


def pending(store, run_id):
    """A run's ``not_dispatched`` actions with attempts left, by slot and sequence.

    Only reads are made, after the run's actions pass the audit.
    """
    if store.store_version < 2:
        return []  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        store._run(db, run_id)
        audit_actions(store, db, run_id)
        result = []
        for row in db.execute("SELECT * FROM workflow_actions WHERE run_id=? AND state='not_dispatched' "
                              "ORDER BY slot,sequence", (run_id,)).fetchall():
            action = _prepared(row)
            states = db.execute("SELECT state FROM workflow_action_attempts WHERE action_id=?",
                                (action.action_id,)).fetchall()
            if sum(state[0] in _CLAIMED for state in states) < action.profile.max_attempts:
                result.append(action)
        return result


def release_attempt(store, lease, attempt_id, error):
    """Release a prepared attempt with its reservation, before its dispatch is claimed.

    The caller holds the run's lease. Nothing was sent, so the attempt does
    not count toward ``max_attempts``. *error* is kept redacted. Returns the
    attempt's projection; raises ``ActionStateError`` unless it is prepared.
    """
    with store._transaction() as db:
        attempt = _attempt(db, attempt_id)
        _owned(store, db, lease, attempt["action_id"])
        if attempt["state"] != "prepared":
            raise ActionStateError(f"Attempt {attempt_id!r} is {attempt['state']}; only a prepared attempt "
                                   "is released")
        _finish(store, db, attempt, "released", error=_error(error))
        return _attempt_item(_attempt(db, attempt_id))


def attempt_status(store, action_id):
    """An action, its attempts' projections, its latest attempt's receipt and that attempt's reads.

    The receipt is None unless the latest attempt has one; reads counts its
    ``readback`` records. Only reads are made, after the run's actions pass
    the audit.
    """
    if store.store_version < 2:
        raise WorkflowNotFoundError(action_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        row = _row(db, action_id)
        audit_actions(store, db, row["run_id"])
        attempts = db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? ORDER BY number",
                              (action_id,)).fetchall()
        latest = attempts[-1] if attempts else None
        receipt = latest and latest["receipt_json"] and _stored(latest["receipt_json"], ConnectorReceipt.from_dict)
        reads = 0
        for record in db.execute("SELECT payload_json FROM workflow_action_records WHERE action_id=? "
                                 "AND kind='readback'", (action_id,)):
            payload = _loads(record[0], stored=True)
            reads += latest is not None and type(payload) is dict and payload.get("attempt_id") == latest["attempt_id"]
        return _prepared(row), [_attempt_item(attempt) for attempt in attempts], receipt or None, reads


def record_readback(store, lease, attempt_id, payload, cost, state):
    """Append one read of an attempt's object as a ``readback`` record.

    The caller holds the run's lease. The attempt is acknowledged and its
    action still ``dispatched``; or both are ``outcome_unknown`` and a lookup
    found the attempt's receipt. *cost* is the read's nanoUSD, confirmed
    against the action; *state* is the action's state after the read:
    unchanged when the read decides nothing, ``observed`` or
    ``compensation_required``. An action with an unknown outcome that no
    finding settled becomes ``reconciled`` instead of ``observed``. Returns
    the record ID.
    """
    if (state not in ("dispatched", "outcome_unknown", "observed", "compensation_required")
            or type(cost) is not int or cost < 0):
        raise WorkflowValidationError("A read leaves its action unchanged, observed or compensation_required "
                                      "at a non-negative integer cost")
    payload_json = _canonical(payload)
    with store._transaction() as db:
        attempt = _attempt(db, attempt_id)
        _, action = _owned(store, db, lease, attempt["action_id"])
        undecided = {("acknowledged", "dispatched"): "dispatched",
                     ("outcome_unknown", "outcome_unknown"): "outcome_unknown"}.get((attempt["state"], action["state"]))
        if (undecided is None or state not in (undecided, "observed", "compensation_required")
                or undecided == "outcome_unknown" and attempt_id not in _found(db, action["action_id"])):
            raise ActionStateError(f"Attempt {attempt_id!r} is {attempt['state']} and its action {action['state']}; "
                                   "only an acknowledged attempt of a dispatched action, or an attempt whose receipt a "
                                   "lookup found, is read back")
        if state == "observed" and _unsettled(db, action["action_id"]):
            state = "reconciled"
        record_id, now = uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_action_records(record_id,action_id,kind,payload_json,payload_sha,
          trusted,cost,created) VALUES (?,?,'readback',?,?,1,?,?)""",
                   (record_id, action["action_id"], payload_json, _sha(payload_json), str(cost), now))
        db.execute("UPDATE workflow_actions SET confirmed=?,state=?,updated=? WHERE action_id=?",
                   (str(_amount(action["confirmed"]) + cost), state, now, action["action_id"]))
        store._event(db, lease.run_id, "action_readback",
                     payload={"action_id": action["action_id"], "record_id": record_id, "state": state})
    return record_id


def keep_notice(store, key, payload):
    """Keep *payload* as an untrusted ``notice`` record of the action keyed *key*; return that action.

    A notice that names no action of the journal is not kept, and None is
    returned. A notice with the payload digest of one the action keeps is not
    kept again. An action keeps at most ``_NOTICES`` notices; for each further
    one an ``action_notice_dropped`` event records its digest, not its body.
    Keeping a notice changes no state.
    """
    payload_json = _canonical(payload)
    if store.store_version < 2:
        return None
    digest = _sha(payload_json)
    with store._transaction() as db:
        row = db.execute("SELECT * FROM workflow_actions WHERE action_key=?", (key,)).fetchone()
        if row is None:
            return None
        where = "FROM workflow_action_records WHERE action_id=? AND kind='notice'"
        if db.execute(f"SELECT 1 {where} AND payload_sha=?", (row["action_id"], digest)).fetchone() is not None:
            return _prepared(row)
        if db.execute(f"SELECT count(*) {where}", (row["action_id"],)).fetchone()[0] >= _NOTICES:
            store._event(db, row["run_id"], "action_notice_dropped",
                         payload={"action_id": row["action_id"], "payload_sha256": digest})
            return _prepared(row)
        record_id = uuid4().hex
        db.execute("""INSERT INTO workflow_action_records(record_id,action_id,kind,payload_json,payload_sha,
          trusted,created) VALUES (?,?,'notice',?,?,0,?)""",
                   (record_id, row["action_id"], payload_json, digest, store._clock_ns()))
        store._event(db, row["run_id"], "action_notice",
                     payload={"action_id": row["action_id"], "record_id": record_id})
        return _prepared(row)


# Reconciliation: lookups by key, same-key retries and findings on unknown outcomes.

FINDINGS = ("not_executed", "unresolved")
# The finding a late receipt appends when it shows a commit of an attempt found not executed.
CONTRADICTED = "contradicted_absence"


def _payloads(db, action_id, kind):
    """The action's records of *kind*, oldest first, as (record ID, payload object) pairs."""
    rows = db.execute("SELECT record_id,payload_json FROM workflow_action_records WHERE action_id=? AND kind=? "
                      "ORDER BY rowid", (action_id, kind)).fetchall()
    pairs = [(row[0], _loads(row[1], stored=True)) for row in rows]
    return [(record_id, payload) for record_id, payload in pairs if type(payload) is dict]


def _found(db, action_id):
    """The receipt the first successful lookup of each attempt found, by attempt ID."""
    found = {}
    for _, payload in _payloads(db, action_id, "lookup"):
        if payload.get("receipt") is not None:
            found.setdefault(payload.get("attempt_id"), payload["receipt"])
    return found


def _absent(db, action_id):
    """The action's ``outcome_unknown`` attempts, oldest first, each with whether it was found not executed.

    A ``not_executed`` finding names the attempt it settles, and an operator's
    ``not_executed`` decision settles every attempt it found.
    """
    settled = {payload.get("attempt_id") for _, payload in _payloads(db, action_id, "finding")
               if payload.get("finding") == "not_executed"}
    found = _risk(db, action_id) or 0
    rows = db.execute("SELECT attempt_id,number FROM workflow_action_attempts WHERE action_id=? "
                      "AND state='outcome_unknown' ORDER BY number", (action_id,)).fetchall()
    return [(row[0], row[0] in settled or row[1] <= found) for row in rows]


def _unsettled(db, action_id):
    """IDs of the action's ``outcome_unknown`` attempts that nothing settled, oldest first.

    A ``not_executed`` finding or decision settles an attempt until a late
    receipt contradicts it (see :func:`_contradict`).
    """
    contradicted = _contradicted(db, action_id)
    return [attempt_id for attempt_id, absent in _absent(db, action_id)
            if not absent or attempt_id in contradicted]


def _contradicted(db, action_id):
    """IDs of the attempts found not executed whose commit a late receipt has shown since."""
    return {payload.get("attempt_id") for _, payload in _payloads(db, action_id, "finding")
            if payload.get("finding") == CONTRADICTED}


def _late_commits(db, action_id):
    """The receipt of each attempt's own commit that a late receipt shows, by attempt ID.

    Only an acknowledged answer that the service did not deduplicate shows
    one; the first such record of an attempt counts.
    """
    commits = {}
    for _, payload in _payloads(db, action_id, "late_receipt"):
        receipt = payload.get("receipt")
        if (payload.get("outcome") == "acknowledged" and type(receipt) is dict
                and receipt.get("deduplicated") is False):
            commits.setdefault(payload.get("attempt_id"), receipt)
    return commits


def _late_numbers(db, action_id):
    """The number of each attempt whose commit a late receipt shows, by attempt ID, oldest first.

    Once there is one, a write under the action's key is known, so no
    attempt of the action can be found not executed.
    """
    committed = _late_commits(db, action_id)
    return {row[0]: row[1] for row in db.execute(
        "SELECT attempt_id,number FROM workflow_action_attempts WHERE action_id=? ORDER BY number", (action_id,))
        if row[0] in committed}


def _contradiction(db, action_id, attempts):
    """How late receipts contradicted the action's ``not_executed`` findings or decisions, or None.

    ``second_effect`` once the action's receipts name more than one commit:
    its attempts' own, late ones and those its lookups found, each a target
    and version the service did not answer from an earlier commit. A receipt
    without a version counts as its attempt's own commit. Otherwise
    ``contradicted_absence``, while a ``contradicted_absence`` finding exists.
    """
    if not _contradicted(db, action_id):
        return None
    receipts = [(attempt["attempt_id"], _loads(attempt["receipt_json"], stored=True))
                for attempt in attempts if attempt["receipt_json"] is not None]
    receipts += [*_late_commits(db, action_id).items(), *_found(db, action_id).items()]
    commits = {(_canonical(receipt.get("target")),
                attempt_id if receipt.get("object_version") is None else receipt["object_version"])
               for attempt_id, receipt in receipts if type(receipt) is dict and receipt.get("deduplicated") is False}
    return "second_effect" if len(commits) > 1 else CONTRADICTED


def _contradict(store, db, action, attempt, late_id, receipt):
    """Append a contradicting finding when a late *receipt* shows a commit of an attempt found not executed.

    *attempt* is the attempt row the late receipt record *late_id* answers.
    The ``contradicted_absence`` finding is appended once per attempt, with
    that record as its evidence, and earlier records stay as they were. The
    attempt is no longer settled, so an action that is ``not_dispatched``
    again becomes ``outcome_unknown``, releasing an attempt begun but not
    claimed: nothing is dispatched until its outcome is settled.
    """
    action_id, attempt_id = action["action_id"], attempt["attempt_id"]
    if (receipt is None or receipt.deduplicated or attempt_id in _contradicted(db, action_id)
            or (attempt_id, True) not in _absent(db, action_id)):
        return
    count = db.execute("SELECT count(*) FROM workflow_action_attempts WHERE action_id=?",
                       (action_id,)).fetchone()[0]
    record_id = _append(store, db, action, "finding", {
        "attempt_id": attempt_id, "finding": CONTRADICTED,
        "reason": "a late receipt shows that this attempt, which was found not executed, took effect",
        "evidence": [late_id], "attempts": count})
    state = _row(db, action_id)["state"]
    if state == "not_dispatched":
        for begun in db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? AND state='prepared'",
                                (action_id,)).fetchall():
            _finish(store, db, begun, "released", error="a late receipt showed that an earlier attempt took effect")
        state = "outcome_unknown"
        db.execute("UPDATE workflow_actions SET state=?,updated=? WHERE action_id=?",
                   (state, store._clock_ns(), action_id))
    store._event(db, action["run_id"], "action_finding", payload={
        "action_id": action_id, "record_id": record_id, "finding": CONTRADICTED, "state": state})


def _risk(db, action_id):
    """The number of attempts an operator's latest ``not_executed`` decision found, or None without one.

    Another dispatch may then repeat an effect, so only an approval that
    accepts that duplicate risk authorizes it.
    """
    found = [payload["attempts"] for _, payload in _payloads(db, action_id, "decision")
             if payload.get("decision") == "not_executed" and type(payload.get("attempts")) is int]
    return found[-1] if found else None


def _shown(db, action):
    """IDs of the action's readback records whose object held its bound content at its target.

    The object must be in the intent's tenant and collection, be its object
    when it names one, and carry the action's key when the read showed a key.
    """
    target, digest, shown = action.intent.target, _held_digest(action.intent), []
    for record_id, payload in _payloads(db, action.action_id, "readback"):
        seen = payload.get("observed")
        where = seen.get("target") if type(seen) is dict else None
        if (type(where) is dict and type(seen.get("content")) is dict
                and seen["content"].get("sha256") == digest and seen.get("key") in (None, action.action_key)
                and (where.get("tenant"), where.get("collection")) == (target.tenant, target.collection)
                and target.object_id in (None, where.get("object_id"))):
            shown.append(record_id)
    return shown


def _append(store, db, action, kind, payload, cost=None):
    """Insert one trusted record of *action*, charging *cost* as confirmed nanoUSD; return its ID."""
    payload_json, record_id, now = _canonical(payload), uuid4().hex, store._clock_ns()
    db.execute("""INSERT INTO workflow_action_records(record_id,action_id,kind,payload_json,payload_sha,
      trusted,cost,created) VALUES (?,?,?,?,?,1,?,?)""",
               (record_id, action["action_id"], kind, payload_json, _sha(payload_json),
                None if cost is None else str(cost), now))
    if cost:
        db.execute("UPDATE workflow_actions SET confirmed=?,updated=? WHERE action_id=?",
                   (str(_amount(action["confirmed"]) + cost), now, action["action_id"]))
    return record_id


def reconcile_status(store, action_id):
    """What reconciling an action needs to know; only reads are made, after the audit.

    Returns a dict: the ``action``; its ``attempts`` as :func:`audit_actions`
    projects them; ``receipt``, the latest attempt's receipt or None;
    ``claims``, each claimed attempt's dispatch time on the journal clock;
    ``ceilings``, each approval's nanoUSD ceiling; ``spent``, the action's
    confirmed, reserved and unknown nanoUSD together; ``unsettled``, the
    ``outcome_unknown`` attempts no ``not_executed`` finding or decision
    settled, oldest first; ``lookups`` and ``reads``, the number of lookup and readback
    records of each attempt; ``found``, the receipt a lookup found for an
    attempt; ``finding``, the payload of the latest finding, or None;
    ``duplicate_risk``, as :func:`_risk` finds it; ``shown``, the
    readback records that showed the bound content (see :func:`_shown`);
    ``committed``, the attempts a late receipt shows took effect; and
    ``contradiction``, as :func:`_contradiction` finds it.
    Attempts and approvals are keyed by ID.
    """
    if store.store_version < 2:
        raise WorkflowNotFoundError(action_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        row = _row(db, action_id)
        audit_actions(store, db, row["run_id"])
        return _status(db, row)


def _status(db, row):
    """:func:`reconcile_status` of the action *row*, inside the caller's transaction."""
    action_id, action = row["action_id"], _prepared(row)
    attempts = db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? ORDER BY number",
                          (action_id,)).fetchall()
    latest = attempts[-1] if attempts else None
    receipt = latest and latest["receipt_json"] and _stored(latest["receipt_json"], ConnectorReceipt.from_dict)
    counts = {"lookup": {}, "readback": {}}
    for kind, tally in counts.items():
        for _, payload in _payloads(db, action_id, kind):
            tally[payload.get("attempt_id")] = tally.get(payload.get("attempt_id"), 0) + 1
    findings = _payloads(db, action_id, "finding")
    return {
        "action": action, "attempts": [_attempt_item(attempt) for attempt in attempts],
        "receipt": receipt or None,
        "claims": {attempt["attempt_id"]: attempt["dispatched"] for attempt in attempts
                   if attempt["dispatched"] is not None},
        "ceilings": {approval["approval_id"]: _amount(approval["ceiling"]) for approval in db.execute(
            "SELECT approval_id,ceiling FROM workflow_action_approvals WHERE action_id=? AND verdict='approved'",
            (action_id,))},
        "spent": sum(_amount(row[name]) for name in ("confirmed", "reserved", "unknown")),
        "unsettled": _unsettled(db, action_id), "lookups": counts["lookup"], "reads": counts["readback"],
        "found": {attempt_id: _stored(_canonical(found), ConnectorReceipt.from_dict)
                  for attempt_id, found in _found(db, action_id).items()},
        "finding": findings[-1][1] if findings else None,
        "duplicate_risk": _risk(db, action_id), "shown": _shown(db, action),
        "committed": sorted(_late_commits(db, action_id)), "contradiction": _contradiction(db, action_id, attempts),
    }


def _unknown_attempt(store, db, lease, attempt_id):
    """The attempt and its action rows, once both are ``outcome_unknown`` and *lease* is the run's."""
    attempt = _attempt(db, attempt_id)
    _, action = _owned(store, db, lease, attempt["action_id"])
    if (attempt["state"], action["state"]) != ("outcome_unknown", "outcome_unknown"):
        raise ActionStateError(f"Attempt {attempt_id!r} is {attempt['state']} and its action {action['state']}; "
                               "only an unknown outcome is reconciled")
    return attempt, action


def record_lookup(store, lease, attempt_id, number, receipt, error, cost):
    """Append one lookup by key for an attempt whose outcome is unknown, as a ``lookup`` record.

    The caller holds the run's lease, and the attempt and its action must be
    ``outcome_unknown``. *number* counts the attempt's lookups from 1;
    *receipt* is what the lookup found, or None, and *error* a failed
    lookup's text, kept redacted. A receipt must name the action's key,
    tenant, collection and object, as for :func:`record_receipt`; otherwise
    ``WorkflowValidationError`` is raised and nothing changes. *cost* is the
    lookup's nanoUSD, confirmed against the action. A receipt that reports a
    cost confirms it as the attempt's own and releases the attempt's unknown
    exposure; without one, the exposure stays. Returns the record ID.
    """
    if type(cost) is not int or cost < 0 or type(number) is not int or number < 1:
        raise WorkflowValidationError("A lookup has a number from 1 and a non-negative integer cost")
    if receipt is not None and not isinstance(receipt, ConnectorReceipt):
        raise WorkflowValidationError("Expected a ConnectorReceipt or None")
    with store._transaction() as db:
        attempt, action = _unknown_attempt(store, db, lease, attempt_id)
        if receipt is not None and not _receipt_of(
                action["action_key"], _stored(action["intent_json"], ActionIntent.from_dict), receipt):
            raise WorkflowValidationError("The lookup answered another key or target than the action's")
        record_id = _append(store, db, action, "lookup", {
            "attempt_id": attempt_id, "lookup": number, "receipt": receipt and receipt.to_dict(),
            "error": None if error is None else _error(error)}, cost)
        if receipt is not None and receipt.cost_nanousd is not None and attempt["cost"] is None:
            action, unknown = _row(db, action["action_id"]), _amount(attempt["unknown"])
            db.execute("UPDATE workflow_action_attempts SET cost=?,unknown='0',updated=? WHERE attempt_id=?",
                       (str(receipt.cost_nanousd), store._clock_ns(), attempt_id))
            db.execute("UPDATE workflow_actions SET confirmed=?,unknown=? WHERE action_id=?",
                       (str(_amount(action["confirmed"]) + receipt.cost_nanousd),
                        str(_amount(action["unknown"]) - unknown), action["action_id"]))
        store._event(db, lease.run_id, "action_lookup", payload={
            "action_id": action["action_id"], "record_id": record_id, "found": receipt is not None})
    return record_id


def claim_retry(store, lease, action_id, approval_id):
    """Claim one same-key retry of an action whose outcome is unknown; return its permit.

    The retry is the action's next attempt. It is committed as ``dispatched``
    under *approval_id*, with its write ceiling reserved, before anything is
    sent, so it is never ``prepared``. Its outcome is recorded with the
    permit, as for :func:`claim_dispatch`; a refusal leaves the action
    ``outcome_unknown``, since it says nothing of the earlier attempt. The
    action must be ``outcome_unknown`` with an attempt no finding settled,
    and its profile must deduplicate, with the key's window open on the
    journal's clock until the retry arrives (see ``smythe.actions``). The
    content, run, approval, ``max_attempts`` and the approval's ceiling are
    checked as for :func:`begin_attempt`. Raises ``ActionNotAuthorized``,
    ``ActionStateError``, ``WorkflowBudgetError`` or ``WorkflowLeaseError``.
    """
    refusal, token = _deliverable(store, action_id), secrets.token_hex(32)
    if refusal:
        raise refusal
    with store._transaction() as db:
        run, action = _owned(store, db, lease, action_id)
        if action["state"] != "outcome_unknown" or not _unsettled(db, action_id):
            raise ActionStateError(f"Action {action_id!r} is {action['state']}; only an unknown outcome is retried")
        approval = _authorized(store, db, run, action, approval_id)
        profile = _stored(action["profile_json"], ConnectorProfile.from_dict)
        attempts = db.execute("SELECT * FROM workflow_action_attempts WHERE action_id=? ORDER BY number",
                              (action_id,)).fetchall()
        now = store._clock_ns()
        first = min(attempt["dispatched"] for attempt in attempts if attempt["dispatched"] is not None)
        if not profile.deduplicates or not _window_open(profile, first, now):
            raise ActionStateError(f"Action {action_id!r} cannot be retried: its service may not deduplicate "
                                   "the retry by its key")
        action, ceiling = _admit(db, action_id, approval, attempts)
        attempt_id = uuid4().hex
        db.execute("""INSERT INTO workflow_action_attempts(attempt_id,action_id,number,approval_id,state,
          dispatch_token_sha,dispatch_epoch,reserved,unknown,created,dispatched,updated)
          VALUES (?,?,?,?,'dispatched',?,?,?,'0',?,?,?)""",
                   (attempt_id, action_id, len(attempts) + 1, approval_id, _sha(token), lease.epoch, str(ceiling),
                    now, now, now))
        db.execute("UPDATE workflow_actions SET state='dispatched',reserved=?,updated=? WHERE action_id=?",
                   (str(_amount(action["reserved"]) + ceiling), now, action_id))
        _note(store, db, action, attempt_id, "dispatched")
    return ActionDispatchPermit(attempt_id, lease.run_id, token, lease.epoch)


def record_finding(store, lease, attempt_id, finding, reason, evidence=()):
    """Append what reconciliation found about an attempt as a ``finding`` record.

    ``not_executed`` says the attempt never took effect: it and its action
    must be ``outcome_unknown``, and no late receipt may show a commit of any
    of the action's attempts. The action returns to ``not_dispatched`` under
    its unconsumed approval once no other attempt's outcome is unknown, and
    stays ``outcome_unknown`` until then; the attempt keeps its unknown
    exposure. ``unresolved`` says that reconciliation can do no more and
    changes no state: the attempt must be ``outcome_unknown`` with its action,
    or acknowledged with its action ``dispatched``. The caller holds the run's
    lease. *reason* is kept redacted and *evidence* names the action's records
    the finding rests on. Returns the record ID, or None for an unresolved
    finding without evidence when the action's latest finding is already
    unresolved on this attempt and no attempt has begun since.
    """
    if finding not in FINDINGS:
        raise WorkflowValidationError(f"A finding is one of {FINDINGS}")
    evidence = list(evidence)
    with store._transaction() as db:
        attempt = _attempt(db, attempt_id)
        _, action = _owned(store, db, lease, attempt["action_id"])
        states = (attempt["state"], action["state"])
        if states != ("outcome_unknown", "outcome_unknown") and (
                finding == "not_executed" or states != ("acknowledged", "dispatched")):
            raise ActionStateError(f"Attempt {attempt_id!r} is {attempt['state']} and its action {action['state']}; "
                                   f"no {finding} finding applies")
        shown = _late_numbers(db, action["action_id"]) if finding == "not_executed" else {}
        if shown:
            which = "it" if attempt_id in shown else f"attempt {next(iter(shown.values()))}"
            raise ActionStateError(f"Attempt {attempt_id!r} cannot be found not executed: a late receipt shows "
                                   f"that {which} took effect")
        known = {row[0] for row in db.execute("SELECT record_id FROM workflow_action_records WHERE action_id=?",
                                              (action["action_id"],))}
        if not all(type(item) is str and item in known for item in evidence):
            raise WorkflowValidationError("A finding's evidence names records of its action")
        count = db.execute("SELECT count(*) FROM workflow_action_attempts WHERE action_id=?",
                           (action["action_id"],)).fetchone()[0]
        latest = _payloads(db, action["action_id"], "finding")
        previous = latest[-1][1] if latest else {}
        if finding == "unresolved" and not evidence and (
                previous.get("finding"), previous.get("attempt_id"), previous.get("attempts")) == (
                    "unresolved", attempt_id, count):
            return None
        record_id = _append(store, db, action, "finding", {
            "attempt_id": attempt_id, "finding": finding, "reason": _error(reason), "evidence": evidence,
            "attempts": count})
        state = action["state"]
        if finding == "not_executed":
            # Another attempt whose outcome is still unknown may have taken effect.
            state = "outcome_unknown" if _unsettled(db, action["action_id"]) else "not_dispatched"
        db.execute("UPDATE workflow_actions SET state=?,updated=? WHERE action_id=?",
                   (state, store._clock_ns(), action["action_id"]))
        store._event(db, lease.run_id, "action_finding", payload={
            "action_id": action["action_id"], "record_id": record_id, "finding": finding, "state": state})
    return record_id


# Operator decisions on open outcomes, and the operator's view of a run's actions.

# An action in these states has an effect that is unknown, wrong or abandoned; see inspect().
_OPEN = ("outcome_unknown", "compensation_required", "unresolved")


def decide(store, action, attempts, decision, verified, evidence=None):
    """Append an operator's verified *decision* on *action* and apply it; return the action.

    The action must still have the state and the number of *attempts* that
    the decision's assertion bound, or ``WorkflowConflictError`` is raised,
    and a security stop on its run raises ``ActionNotAuthorized``. A
    ``not_executed`` decision on an action with an attempt whose commit a late
    receipt shows raises ``ActionStateError``.
    *evidence*, an ``ActionEvidence``, becomes an ``evidence`` record. The
    ``decision`` record holds *verified* (the principal, the verifier's
    identity, the digest the assertion approved and the note) with the
    decision, the state it was made in, *attempts* and, as ``evidence``, the
    records it rests on: that evidence and, for ``adopt_observed``, the
    readbacks that showed the bound content. No attempt, approval or balance
    changes, and an ``action_decision`` event names the record.
    """
    state = _DECISIONS[decision][1]
    with store._transaction() as db:
        row = _row(db, action.action_id)
        count = db.execute("SELECT count(*) FROM workflow_action_attempts WHERE action_id=?",
                           (action.action_id,)).fetchone()[0]
        if (row["state"], count, row["intent_sha"]) != (action.state, attempts, action.intent.digest()):
            raise WorkflowConflictError(f"Action {action.action_id!r} changed while its decision was verified; "
                                        "decide again")
        if store._run(db, row["run_id"])["blocked_reason"] == _SECURITY_STOP:
            raise ActionNotAuthorized(f"Action {action.action_id!r} cannot be decided: a security stop holds "
                                      "the run")
        shown = list(_late_numbers(db, action.action_id).values())
        if decision == "not_executed" and shown:
            raise ActionStateError(f"Action {action.action_id!r} cannot be found not executed: a late receipt "
                                   f"shows that attempt {shown[0]} took effect")
        cited = [] if evidence is None else [_append(store, db, row, "evidence", evidence.to_dict())]
        if decision == "adopt_observed":
            cited += _shown(db, action)
        record_id = _append(store, db, row, "decision", {
            **verified, "decision": decision, "state": action.state, "attempts": attempts, "evidence": cited})
        db.execute("UPDATE workflow_actions SET state=?,updated=? WHERE action_id=?",
                   (state, store._clock_ns(), action.action_id))
        store._event(db, row["run_id"], "action_decision", payload={
            "action_id": action.action_id, "record_id": record_id, "decision": decision, "state": state})
        return _prepared(_row(db, action.action_id))


def inspect(store, run_id):
    """A run's actions as :func:`audit_actions` projects them, with what each needs next.

    Each action adds ``duplicate_risk``, ``contradiction`` (see
    :func:`_contradiction`) and ``next_action``; ``unresolved`` lists the
    actions that are ``outcome_unknown``, ``compensation_required`` or
    ``unresolved``, ``dispatched`` with no read left to confirm them, or
    contradicted: with a second effect, or not yet settled by a read.
    One read transaction; a version 1 journal has no actions.
    """
    with store._transaction(write=False) as db:
        store._run(db, run_id)
        if store.store_version < 2:
            block = {"accounting_scope": ACCOUNTING_SCOPE, "unit": NANOUSD, **dict.fromkeys(_BALANCES, 0),
                     "items": []}
        else:
            block = audit_actions(store, db, run_id)
        unresolved, now = [], store._clock_ns()
        for item in block["items"]:
            row = _row(db, item["action_id"])
            status = _status(db, row)
            step, risk = _next_action(status), status["duplicate_risk"]
            if step == "dispatch":
                approvals = db.execute("SELECT * FROM workflow_action_approvals WHERE action_id=? "
                                       "ORDER BY rowid DESC", (row["action_id"],)).fetchall()
                try:
                    _authorizing(row["action_id"], row["state"], row["intent_sha"], _amount(row["ceiling"]),
                                 approvals, now, risk=risk)
                except ActionNotAuthorized:
                    step = "approve" if risk is None else "approve_duplicate_risk"
            contradiction = status["contradiction"]
            item.update(duplicate_risk=risk is not None, contradiction=contradiction, next_action=step)
            if (item["state"] in _OPEN or item["state"] == "dispatched" and step != "readback"
                    or contradiction == "second_effect"
                    or contradiction is not None and item["state"] not in ("observed", "reconciled")):
                unresolved.append(item["action_id"])
    items = block.pop("items")
    return {**block, "unresolved": unresolved, "items": items}
