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
These functions run inside a ``SQLiteWorkflowStore`` transaction, and
``import smythe`` does not load this module.
"""

from __future__ import annotations

from smythe._money import NANOUSD
from smythe.actions import ACTION_STATES, ActionIntent, ConnectorProfile, Principal, action_key
from smythe.workflow_store import WorkflowCorruptionError, _amount, _canonical, _loads, _sha

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
