"""Read-only, bounded operator views of the Jobs ledger and local artifacts."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smythe._canonical import sha256_hex
from smythe._publish import _check_path, _integrity, _relative_parts
from smythe._redact import redact_account_identifiers
from smythe.jobs.artifact_io import MAX_ARTIFACT_BYTES
from smythe.jobs.delivery import annotate_delivery, validator_registry
from smythe.jobs.preflight import operation_input_digest
from smythe.jobs.store import RunStoreError, SQLiteRunStore


INSPECTION_VERSION = 1
MAX_INSPECTION_HASH_BYTES = 64 * 1024 * 1024
_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REVIEW_VERSION = 1
# Every kind of review item, in the order a review's totals list them.
REVIEW_KINDS = ("budget_overrun", "invalid_accounting", "pause_requested", "worker_failed",
                "lease_expired", "unknown_outcome", "rejected", "failed", "attempts_exhausted",
                "artifact_integrity", "delivery_ineligible")
# What an item's state holds back, in the order a review lists items.
REVIEW_BLOCKS = ("admission", "completion", "delivery")
MAX_REVIEW_EXCERPT = 300
# The operator's own text replaces it in a suggested command's --reason.
_REASON = "REASON"
_NEW_PLAN = "a new plan and approval can regenerate it"
# What each kind of review item asks the operator to decide.
_DECIDE = {
    "budget_overrun": "The run spent past its approved ceiling, so it admits no further call. "
                      "Work it has not finished needs a new plan and approval.",
    "pause_requested": "A stop request holds back new work. Once admitted calls have drained and "
                       "its reason is settled, resume the run, which clears the request.",
    "worker_failed": "The latest detached worker stopped without finishing the run. Read its log "
                     "beside the Jobs database, then resume the run; a call the worker had "
                     "dispatched becomes an unknown outcome.",
    "lease_expired": "The run is recorded as running, but no live worker holds it. Resume it to "
                     "recover the interrupted work; a call that was dispatched becomes an unknown "
                     "outcome.",
    "invalid_accounting": "The provider reported cost or usage that cannot be accounted for, so "
                          "the run dispatches nothing more while this is unresolved. Repair the "
                          "provider adapter and investigate the charge, then reroll, "
                          "acknowledging a possible duplicate call.",
    "unknown_outcome": "The call may have completed and been charged. Investigate the provider "
                       "account and the artifact destination, then adopt the bytes the attempt "
                       "published, reroll if a duplicate call is acceptable, or leave it "
                       "unresolved.",
    "changed": "An accepted file is missing or changed; its recorded acceptance is unchanged.",
    "repair": "An accepted file is missing or changed; its recorded acceptance is unchanged. "
              "Regenerate it with an integrity repair, which hashes every accepted file again "
              "first.",
    "unsafe": "An accepted file is unreadable or unsafe, such as a symbolic link, so no repair in "
              "this run replaces it. Restore the file, then review the run again.",
    "recheck": "Its accepted bytes were decided under checks that have changed since. Recheck "
               "them under the current checks, without a provider call.",
    "validators": "Its accepted bytes were decided under checks that have changed since. Recheck "
                  "them in Python with JobRunner(store, validators=[...]).recheck(): the plan "
                  "names registered validators, and the CLI registers none.",
    "review": "The latest decision about its accepted bytes did not accept them, or its record "
              "does not match them. A person decides: no repair in this run replaces bytes that "
              "verify, and a new plan and approval can regenerate them.",
}


def list_jobs(
    store: SQLiteRunStore,
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> dict[str, Any]:
    """List run summaries without loading prompts or constructing providers."""
    return store.list_run_page(limit=limit, offset=offset, status=status)


def _artifact_directory(snapshot: dict[str, Any]) -> str:
    """Use the recorded directory; only older snapshots omit that field."""
    run_id = snapshot["run_id"]
    if not isinstance(run_id, str) or _RUN_RE.fullmatch(run_id) is None:
        raise ValueError("Ledger run ID is not a safe path component")
    directory = snapshot["artifact_directory"] if "artifact_directory" in snapshot else run_id
    if not isinstance(directory, str) or _RUN_RE.fullmatch(directory) is None:
        raise ValueError("Ledger artifact directory is not a safe path component")
    return directory


def _artifact_root(snapshot: dict[str, Any]) -> tuple[Path, tuple[str, ...]]:
    root_value = snapshot["manifest_root"]
    if not isinstance(root_value, str):
        raise ValueError("Ledger manifest root must be a text path")
    root = Path(root_value)
    if not root.is_absolute():
        raise ValueError("Ledger manifest root must be an absolute path")
    root = root.resolve()
    parts = (*_relative_parts(snapshot["output_directory"], allow_dot=True), _artifact_directory(snapshot))
    _check_path(root, parts)
    return root, parts


def inspect_job(
    store: SQLiteRunStore,
    run_id: str,
    *,
    operation: str | None = None,
    limit: int = 50,
    offset: int = 0,
    events_limit: int = 100,
    validators: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Read one ledger view, then check a bounded amount of local artifact bytes.

    Run costs and status describe the entire ledger run. Operations and their
    attempt lineage are paged; recent events carry a separate limit. Files can
    change after the SQLite snapshot, so integrity findings are timestamped
    observations and never change recorded acceptance or run status. Each
    attempt carries its recorded ``delivery_acceptance`` decision and its
    rechecks as ``delivery_reevaluations``, and each operation a projected
    ``delivery_acceptance`` and a derived ``delivery_eligibility`` (see
    :func:`smythe.jobs.delivery.operation_delivery`), and its ``adoption``:
    the reason, attempt, call and kept exposure of bytes adopted for an
    unknown outcome, whose attempt and call stay unknown, or ``None``.
    Eligibility needs a decision made under the validators the plan names,
    or, given *validators*, under those registered now with the same names,
    as ``JobRunner(validators=...)`` registers them. Each operation also
    carries its ``input_digest``, computed from the stored
    plan (see :func:`smythe.jobs.preflight.operation_input_digest`), or ``None``
    when its stored specification, or the plan's stored attachment
    fingerprints it needs, cannot produce one.
    """
    registered = {name: identity for name, (identity, _validator)
                  in validator_registry(validators).items()}
    snapshot = store.inspection_snapshot(
        run_id, operation=operation, limit=limit, offset=offset, events_limit=events_limit
    )
    snapshot["inspection_version"] = INSPECTION_VERSION
    snapshot["inspected_at_ns"] = time.time_ns()
    _add_input_digests(store, run_id, snapshot.get("operations", ()))
    summary = check_artifact_integrity(
        snapshot, snapshot["artifacts"], byte_limit=MAX_INSPECTION_HASH_BYTES,
    )
    snapshot["artifact_integrity"] = summary | {
        "hash_byte_limit": MAX_INSPECTION_HASH_BYTES,
        "scope": "artifact bytes in the selected operation page; acceptance is unchanged",
    }
    records = {attempt["attempt_id"]: attempt["delivery_acceptance"]
               for attempt in snapshot.get("attempts", ())
               if attempt.get("delivery_acceptance") is not None}
    rechecks = {attempt["attempt_id"]: attempt.get("delivery_reevaluations") or []
                for attempt in snapshot.get("attempts", ())}
    return annotate_delivery(snapshot, records, rechecks, registered)


def review_job(
    store: SQLiteRunStore,
    run_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    validators: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """List only what in one run needs an operator's decision, changing nothing.

    One :meth:`SQLiteRunStore.inspection_snapshot` read pages the run's
    operations; the accepted files of the page's succeeded operations are
    hashed within the 64 MiB inspection limit, and the latest detached
    launch's receipts are read. Nothing is written, migrated or dispatched,
    and no provider is constructed. Each item has an ``item_id``, a ``kind``
    from :data:`REVIEW_KINDS`, what it ``blocks`` (:data:`REVIEW_BLOCKS`),
    its ``subject``, what was ``observed``, the ``decision_needed`` and, as
    ``resolve_with``, the ``smythe`` arguments, without ``--store``, that
    carry the decision out, in order; ``REASON`` stands for the operator's
    own reason. The ``item_id`` hashes the kind, run, subject and the state
    the item depends on, so it changes only with that state. Items about
    the whole run appear on every page, those about operations on theirs;
    blocking kinds come first, then operation keys. Stored text is a capped
    excerpt with account identifiers redacted and local directories
    replaced. *validators* judge delivery eligibility as for
    :func:`inspect_job`.
    """
    # Imported here: the runner imports this module.
    from smythe.jobs.operator import _idle_reason, read_worker_state
    from smythe.jobs.runner import _INVALID_ACCOUNTING_PREFIX, _PREVIEW_REASONS

    registered = {name: identity for name, (identity, _validator)
                  in validator_registry(validators).items()}
    snapshot = store.inspection_snapshot(run_id, limit=limit, offset=offset, events_limit=1)
    status, operations, attempts = snapshot["status"], snapshot["operations"], snapshot["attempts"]
    accepted = {item["accepted_attempt_id"] for item in operations if item["status"] == "succeeded"}
    summary = check_artifact_integrity(snapshot, [
        row for row in snapshot["artifacts"] if row["accepted"] and row["attempt_id"] in accepted
    ], byte_limit=MAX_INSPECTION_HASH_BYTES)
    records = {item["attempt_id"]: item["delivery_acceptance"] for item in attempts
               if item["delivery_acceptance"] is not None}
    annotate_delivery(snapshot, records, {item["attempt_id"]: item["delivery_reevaluations"]
                                          for item in attempts}, registered)
    worker = read_worker_state(store.path, run_id)
    paths = _local_paths(((snapshot["manifest_root"], "<manifest root>"),
                          (store.path.parent, "<store directory>"), (_home(), "~")))
    admits = status != "budget_overrun"
    items: list[dict[str, Any]] = []

    def observe(state, text=None, error_class=None, **extra):
        return {"status": state, "error_class": error_class, "excerpt": _excerpt(text, paths),
                **extra}

    def add(kind, blocks, subject, state, observed, decision, commands=(), reasons=()):
        items.append(_review_item(kind, blocks, run_id, subject, state, observed, decision,
                                  commands, reasons))

    def command(name, keys=(), *options):
        # Manifest keys start with a letter; a damaged ledger's key still parses after "--".
        if any(key.startswith("-") for key in keys):
            return ["jobs", name, run_id, *options, "--", *keys]
        return ["jobs", name, run_id, *keys, *options]

    run = {"run_id": run_id}
    if not admits:
        amounts = {key: snapshot["cost"][key]
                   for key in ("approved_microusd", "confirmed_microusd", "exposure_microusd")}
        add("budget_overrun", "admission", run, amounts, observe(status, **amounts),
            _DECIDE["budget_overrun"])
    control = snapshot["control"]
    if control["pause_requested"]:
        generation = control["pause_generation"]
        add("pause_requested", "admission", run, {"status": status, "pause_generation": generation},
            observe(status, control["pause_reason"], pause_generation=generation),
            _DECIDE["pause_requested"], [command("resume")])
    idle = _idle_reason(snapshot, worker)
    if idle == "worker_failed":
        add(idle, "completion", {"launch_id": worker["launch_id"]},
            {"state": worker["state"], "exit_code": worker.get("exit_code"), "status": status},
            observe(worker["state"], worker.get("error"), worker.get("error_type")),
            _DECIDE[idle], [command("resume")])
    elif idle == "lease_expired":
        lease = snapshot["lease"] or {}
        add(idle, "completion", run | {"lease_epoch": lease.get("epoch")},
            {"status": status, "expires_at_ns": lease.get("expires_at_ns")},
            observe(status, lease_epoch=lease.get("epoch")), _DECIDE[idle], [command("resume")])

    latest, by_id, calls, files = {}, {}, {}, {}
    for attempt in attempts:  # Ordered by attempt number within each operation.
        latest[attempt["operation_id"]] = by_id[attempt["attempt_id"]] = attempt
    for call in snapshot["calls"]:
        calls[call["attempt_id"]] = call
    for row in snapshot["artifacts"]:
        if "integrity" in row:
            files.setdefault(row["attempt_id"], []).append(row)
    for operation in operations:
        key, state = operation["operation_key"], operation["status"]
        attempt = (by_id.get(operation["accepted_attempt_id"]) if state == "succeeded"
                   else latest.get(operation["operation_id"])) or {}
        call = calls.get(attempt.get("attempt_id")) or {}
        subject = {"operation_key": key, "operation_id": operation["operation_id"],
                   "attempt_id": attempt.get("attempt_id"),
                   "attempt_number": attempt.get("attempt_number"), "call_id": call.get("call_id")}
        left = max(0, operation["max_attempts"] - operation["attempt_count"])
        recorded = {"status": state, "attempts_used": operation["attempt_count"],
                    "call_status": call.get("status")}
        # Why this run admits no new call for the operation, if it admits none.
        no_call = ("The run's budget overran" if not admits
                   else None if left else "No attempt is left")
        if state == "unknown_outcome":
            invalid = (operation["error"] or "").startswith(_INVALID_ACCOUNTING_PREFIX)
            kind = "invalid_accounting" if invalid else "unknown_outcome"
            exposure = call.get("exposure_microusd", 0)
            commands = [] if invalid else [command("adopt", [key], "--reason", _REASON)]
            if no_call is None:
                commands.append(
                    command("reroll", [key], "--acknowledge-unknown", "--reason", _REASON))
            add(kind, "admission" if invalid else "completion", subject,
                recorded | {"exposure_microusd": exposure},
                observe(state, operation["error"],
                        "invalid_provider_accounting" if invalid else None,
                        exposure_microusd=exposure, attempts_remaining=left),
                _DECIDE[kind] + ("" if no_call is None else
                                 f" {no_call}, so this run admits no reroll of it."), commands)
        elif state in ("rejected", "failed"):
            add(state if left else "attempts_exhausted", "completion", subject, recorded,
                observe(state, operation["error"], attempts_remaining=left),
                f"Reroll it with a reason, or leave it {state}." if no_call is None else
                f"{no_call}, so this run has no supported resolution; {_NEW_PLAN}.",
                [] if no_call else [command("reroll", [key], "--reason", _REASON)])
        elif state == "succeeded" and not operation["delivery_eligibility"]["eligible"]:
            rows = files.get(operation["accepted_attempt_id"], [])
            bad = [row for row in rows
                   if row["integrity"]["status"] not in ("verified", "not_checked")]
            if bad:
                found = sorted({row["integrity"]["status"] for row in bad})
                repairable = set(found) <= {"missing", "changed"}
                add("artifact_integrity", "delivery", subject,
                    recorded | {"artifacts": [[row["artifact_id"], row["sha256"],
                                               row["integrity"]["status"]] for row in bad]},
                    observe(state, bad[0]["integrity"]["detail"], attempts_remaining=left,
                            artifacts=[{"artifact_id": row["artifact_id"],
                                        "relative_path": row["relative_path"],
                                        "status": row["integrity"]["status"]} for row in bad]),
                    _DECIDE["unsafe"] if not repairable else _DECIDE["repair"] if no_call is None
                    else f"{_DECIDE['changed']} {no_call}, so {_NEW_PLAN}.",
                    [command("reroll", [key], "--repair-integrity", "--reason", _REASON)]
                    if repairable and no_call is None else [],
                    [f"artifact_{status}" for status in found])
            explained = {f"artifact_{row['integrity']['status']}" for row in rows}
            other = [reason for reason in operation["delivery_eligibility"]["reasons"]
                     if reason not in explained]
            if other:
                codes = sorted({_PREVIEW_REASONS.get(reason, reason) for reason in other})
                # The CLI registers no validators, so it rechecks only plans that name none.
                cli = not operation["spec"].get("validators")
                recheck = codes == ["validator_set_changed"]
                decided = operation["delivery_acceptance"] or {}
                add("delivery_ineligible", "delivery", subject,
                    recorded | {"reasons": other, "decision": decided.get("decision")},
                    observe(state, "; ".join(map(str, decided.get("reasons") or ())) or None,
                            decision=decided.get("decision"), eligibility_reasons=other),
                    _DECIDE["recheck" if cli else "validators"] if recheck else _DECIDE["review"],
                    [command("recheck", [key], "--reason", _REASON)] if recheck and cli else [],
                    codes)

    items.sort(key=lambda item: (REVIEW_BLOCKS.index(item["blocks"]),
                                 item["subject"].get("operation_key") or "",
                                 REVIEW_KINDS.index(item["kind"]), item["item_id"]))
    totals = dict.fromkeys(REVIEW_KINDS, 0)
    for item in items:
        totals[item["kind"]] += 1
    return {
        "review_version": REVIEW_VERSION, "run_id": run_id, "name": snapshot["name"],
        "status": status, "schema_version": snapshot["version"],
        "reviewed_at_ns": snapshot["observed_at_ns"], "operation_counts": snapshot["counts"],
        "cost": snapshot["cost"], "page": snapshot["pagination"], "totals": totals, "items": items,
        "artifact_integrity": summary | {"hash_byte_limit": MAX_INSPECTION_HASH_BYTES},
    }


def _review_item(kind, blocks, run_id, subject, state, observed, decision, commands=(),
                 reasons=()) -> dict[str, Any]:
    """One review item; its ``item_id`` hashes the kind, run, subject and the state it depends on.

    Jobs and runs reviews share this item schema (:data:`REVIEW_VERSION`).
    """
    return {
        "item_id": sha256_hex({"kind": kind, "run_id": run_id, "subject": subject, "state": state}),
        "kind": kind, "blocks": blocks, "subject": subject, "observed": observed,
        "reasons": list(reasons), "decision_needed": decision, "resolve_with": list(commands),
    }


def _home() -> Path | None:
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):
        return None


def _local_paths(values) -> list[tuple[re.Pattern[str], str]]:
    """Patterns for the local directories a review replaces in stored text, longest first."""
    found: dict[str, str] = {}
    for value, label in values:
        if isinstance(value, (str, Path)) and len(Path(value).parts) > 1:
            for text in (str(value), Path(value).as_posix()):
                found.setdefault(text, label)
    flags = re.IGNORECASE if os.name == "nt" else 0
    return [(re.compile(re.escape(text), flags), label)
            for text, label in sorted(found.items(), key=lambda item: -len(item[0]))]


def _excerpt(text: object, paths: Sequence[tuple[re.Pattern[str], str]]) -> str | None:
    """Stored text as a review keeps it: redacted, without local directories, and capped."""
    if text is None:
        return None
    text = redact_account_identifiers(str(text))
    for pattern, label in paths:
        text = pattern.sub(label, text)
    return text if len(text) <= MAX_REVIEW_EXCERPT else text[:MAX_REVIEW_EXCERPT - 1] + "…"


def _add_input_digests(store: SQLiteRunStore, run_id: str, operations) -> None:
    # Plan fingerprints are read only when a listed operation sends attachments.
    # The digest is informational: fingerprints that cannot be read leave the
    # digests of operations that send attachments None, not the inspection failed.
    fingerprints = []
    if any(type(item.get("spec")) is dict and item["spec"].get("attachments")
           for item in operations):
        try:
            fingerprints = store.plan_attachments(run_id)
        except RunStoreError:
            fingerprints = []
    for item in operations:
        try:
            item["input_digest"] = operation_input_digest(item["spec"], fingerprints)
        except (KeyError, TypeError, ValueError):
            item["input_digest"] = None


def check_artifact_integrity(
    snapshot: dict[str, Any],
    artifacts: list[dict[str, Any]],
    *,
    byte_limit: int | None,
) -> dict[str, Any]:
    """Set each artifact's ``integrity`` observation and summarize the findings.

    Files are hashed until *byte_limit* bytes have been read; later files are
    ``not_checked``. ``None`` checks every file. Recorded rows are unchanged.
    """
    hashed = 0
    findings: dict[str, int] = {}
    try:
        anchor, root_parts = _artifact_root(snapshot)
        root_error = None
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        root_error = str(exc)
    for artifact in artifacts:
        if root_error is not None:
            integrity = {
                "status": "unsafe", "detail": root_error, "checked_at_ns": time.time_ns()
            }
        else:
            remaining = MAX_ARTIFACT_BYTES if byte_limit is None else byte_limit - hashed
            integrity, consumed = _integrity(artifact, anchor, root_parts, remaining)
            hashed += consumed
        artifact["integrity"] = integrity
        status = integrity["status"]
        findings[status] = findings.get(status, 0) + 1
    return {"counts": findings, "bytes_hashed": hashed}
