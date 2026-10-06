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
other functions open their own, need no run lease and are called by
``smythe.actions``. ``import smythe`` does not load this module.
"""

from __future__ import annotations

from uuid import uuid4

from smythe._money import NANOUSD
from smythe.actions import (
    _CLOSED, _FINISHED, ACTION_STATES, ActionIntent, ActionStateError, ApprovalRejected,
    ConnectorProfile, PreparedAction, Principal, action_key,
)
from smythe.workflow_store import (
    WorkflowConflictError, WorkflowCorruptionError, WorkflowNotFoundError, _amount, _canonical,
    _loads, _sha,
)

ACCOUNTING_SCOPE = "external_actions"
ATTEMPT_STATES = ("prepared", "dispatched", "acknowledged", "refused", "outcome_unknown", "released")
APPROVAL_VERDICTS = ("approved", "rejected")
RECORD_KINDS = ("readback", "lookup", "notice", "late_receipt", "finding", "decision", "evidence")
# Attempt states after a dispatch claim, whose request may have reached the service.
_CLAIMED = frozenset({"dispatched", "acknowledged", "refused", "outcome_unknown"})
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
    numbered from 1, each under one of the action's approvals; and approvals
    bind the action's intent digest. Raises ``WorkflowCorruptionError``.

    The projection never includes parameters, assertions, receipts, record
    payloads or error text: only identities, digests, states and balances.
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
    sums, attempt_items = [0, 0, 0], []
    for number, attempt in enumerate(attempts, 1):
        state, claimed, epoch = attempt["state"], attempt["dispatch_token_sha"] is not None, attempt["dispatch_epoch"]
        cost, held, unknown = _money(attempt["cost"]), _amount(attempt["reserved"]), _amount(attempt["unknown"])
        receipt = attempt["receipt_json"]
        if receipt is not None:
            _stored(receipt)
        _check(attempt["number"] == number and state in ATTEMPT_STATES and attempt["approval_id"] in approved
               and claimed == (state in _CLAIMED) == (attempt["dispatched"] is not None)
               and ((type(epoch) is int and epoch > 0) if claimed else epoch is None)
               and (None if receipt is None else _sha(receipt)) == attempt["receipt_sha"]
               and (not held or state in ("prepared", "dispatched"))
               and (not unknown or state == "outcome_unknown"), "External action attempt is inconsistent")
        sums = [sums[0] + (cost or 0), sums[1] + held, sums[2] + unknown]
        attempt_items.append({
            "attempt_id": attempt["attempt_id"], "number": number, "approval_id": attempt["approval_id"],
            "state": state, "dispatch_epoch": epoch, "receipt_sha256": attempt["receipt_sha"],
            "cost_nanousd": cost, "reserved_nanousd": held, "unknown_nanousd": unknown})
    approval_items = []
    for approval in approvals:
        principal = _stored(approval["principal_json"], Principal.from_dict)
        verifier, consumed = _stored(approval["verifier_json"]), approval["consumed_by_attempt"]
        _check(approval["intent_sha"] == row["intent_sha"] and approval["verdict"] in APPROVAL_VERDICTS
               and _sha(approval["assertion"]) == approval["assertion_sha"] and type(verifier) is dict
               and (consumed is None or any(attempt["attempt_id"] == consumed
                                            and attempt["approval_id"] == approval["approval_id"]
                                            for attempt in attempts))
               and all(approval[name] is None or type(approval[name]) is int
                       for name in ("expires_at_ns", "revoked_at_ns"))
               and (approval["revoked_at_ns"] is None) == (approval["revoked_reason"] is None)
               and approval["duplicate_risk"] in (0, 1), "External action approval is inconsistent")
        approval_items.append({
            "approval_id": approval["approval_id"], "verdict": approval["verdict"],
            "intent_sha256": approval["intent_sha"], "principal": principal.to_dict(), "verifier": verifier,
            "ceiling_nanousd": _money(approval["ceiling"]), "expires_at_ns": approval["expires_at_ns"],
            "consumed_by_attempt": consumed, "revoked_at_ns": approval["revoked_at_ns"],
            "duplicate_risk": approval["duplicate_risk"] == 1})
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


def _row(db, action_id):
    row = db.execute("SELECT * FROM workflow_actions WHERE action_id=?", (action_id,)).fetchone()
    if row is None:
        raise WorkflowNotFoundError(action_id)
    return row


def _prepared(row):
    return PreparedAction(row["action_id"], row["run_id"], row["slot"], row["sequence"], row["action_key"],
                          _stored(row["intent_json"], ActionIntent.from_dict),
                          _stored(row["profile_json"], ConnectorProfile.from_dict), row["state"],
                          _amount(row["ceiling"]))


def load(store, action_id):
    """An action, after its run's actions pass the audit, and its approval rows, newest first."""
    if store.store_version < 2:
        raise WorkflowNotFoundError(action_id)  # Version 1 journals have no action tables.
    with store._transaction(write=False) as db:
        row = _row(db, action_id)
        audit_actions(store, db, row["run_id"])
        return _prepared(row), db.execute("SELECT * FROM workflow_action_approvals WHERE action_id=? "
                                          "ORDER BY rowid DESC", (action_id,)).fetchall()


def prepare(store, run_id, slot, intent, profile):
    """Record *intent* as the next action in a run's *slot*, or return the slot's identical one.

    Raises ``WorkflowConflictError`` while the slot's latest action is
    unfinished and has another intent or profile.
    """
    intent_json, profile_json = _canonical(intent.to_dict()), _canonical(profile.to_dict())
    ceiling = (profile.max_attempts * profile.write_ceiling_nanousd
               + profile.readback_reads * profile.read_ceiling_nanousd)
    with store._transaction() as db:
        store._run(db, run_id)
        latest = db.execute("SELECT * FROM workflow_actions WHERE run_id=? AND slot=? ORDER BY sequence DESC "
                            "LIMIT 1", (run_id, slot)).fetchone()
        if latest is not None and (latest["intent_json"], latest["profile_json"]) == (intent_json, profile_json):
            return _prepared(latest)
        if latest is not None and latest["state"] not in _FINISHED:
            raise WorkflowConflictError(f"Slot {slot!r} holds an unfinished action with another intent or "
                                        "profile; a changed intent is a new action, never a retry")
        sequence, action_id, now = 1 if latest is None else latest["sequence"] + 1, uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_actions(action_id,run_id,slot,sequence,action_key,intent_json,intent_sha,
          profile_json,state,ceiling,confirmed,reserved,unknown,created,updated)
          VALUES (?,?,?,?,?,?,?,?,'not_dispatched',?,'0','0','0',?,?)""",
                   (action_id, run_id, slot, sequence, action_key(store.store_id, run_id, slot, sequence),
                    intent_json, intent.digest(), profile_json, str(ceiling), now, now))
        return _prepared(_row(db, action_id))


def record_verdict(store, action, verdict, principal, identity, assertion, ceiling, expires_at_ns):
    """Record a verified approval or rejection of *action*; return its row. A rejection cancels it.

    An assertion is recorded once in a journal: the same verdict on the same
    action again returns its row, and anything else raises ``ApprovalRejected``.
    """
    assertion_sha, ceiling = _sha(assertion), None if ceiling is None else str(ceiling)
    with store._transaction() as db:
        row = _row(db, action.action_id)
        if row["intent_sha"] != action.intent.digest():
            raise WorkflowConflictError("The action's recorded intent changed")
        if row["state"] in _CLOSED or verdict == "rejected" and row["state"] != "not_dispatched":
            raise ActionStateError(f"Action {action.action_id!r} is {row['state']}; it cannot be {verdict}")
        existing = db.execute("SELECT * FROM workflow_action_approvals WHERE assertion_sha=?",
                              (assertion_sha,)).fetchone()
        if existing is not None:
            if ((existing["action_id"], existing["verdict"], existing["ceiling"], existing["expires_at_ns"],
                 existing["revoked_at_ns"]) != (action.action_id, verdict, ceiling, expires_at_ns, None)):
                raise ApprovalRejected("This assertion is already recorded in the journal; each approval "
                                       "needs its own assertion")
            return existing
        approval_id, now = uuid4().hex, store._clock_ns()
        db.execute("""INSERT INTO workflow_action_approvals(approval_id,action_id,intent_sha,verdict,principal_json,
          verifier_json,assertion,assertion_sha,ceiling,expires_at_ns,created) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                   (approval_id, action.action_id, row["intent_sha"], verdict, _canonical(principal.to_dict()),
                    _canonical({"identity": identity}), assertion, assertion_sha, ceiling, expires_at_ns, now))
        if verdict == "rejected":
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
