"""Read-only, bounded operator views of the Jobs ledger and local artifacts."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smythe._publish import _check_path, _integrity, _relative_parts
from smythe.jobs.artifact_io import MAX_ARTIFACT_BYTES
from smythe.jobs.delivery import annotate_delivery, validator_registry
from smythe.jobs.preflight import operation_input_digest
from smythe.jobs.store import RunStoreError, SQLiteRunStore


INSPECTION_VERSION = 1
MAX_INSPECTION_HASH_BYTES = 64 * 1024 * 1024
_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


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
