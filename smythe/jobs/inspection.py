"""Read-only, bounded operator views of the Jobs ledger and local artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from smythe.jobs.artifact_io import MAX_ARTIFACT_BYTES
from smythe.jobs.store import SQLiteRunStore


INSPECTION_VERSION = 1
MAX_INSPECTION_HASH_BYTES = 64 * 1024 * 1024
_HASH_RE = re.compile(r"[0-9a-f]{64}")
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


def _relative_parts(value: object, *, allow_dot: bool = False) -> tuple[str, ...]:
    # Reject both platform syntaxes even when inspecting a relocated ledger.
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected a nonempty relative artifact path")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.drive or windows.root or posix.is_absolute():
        raise ValueError("Absolute artifact paths are not inspected")
    if ".." in posix.parts or any(":" in part for part in posix.parts):
        raise ValueError("Artifact path escapes its output directory")
    if not posix.parts and not allow_dot:
        raise ValueError("Artifact path has no filename")
    return posix.parts


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _check_path(anchor: Path, parts: tuple[str, ...]) -> Path:
    candidate = anchor
    for part in parts:
        candidate = candidate / part
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info):
            raise ValueError("Symbolic links and junctions are not inspected")
    if not candidate.resolve().is_relative_to(anchor):
        raise ValueError("Artifact path escapes its output directory")
    return candidate


def _artifact_root(snapshot: dict[str, Any]) -> tuple[Path, tuple[str, ...]]:
    root_value = snapshot["manifest_root"]
    if not isinstance(root_value, str):
        raise ValueError("Ledger manifest root must be a text path")
    root = Path(root_value)
    if not root.is_absolute():
        raise ValueError("Ledger manifest root must be an absolute path")
    root = root.resolve()
    run_id = snapshot["run_id"]
    if not isinstance(run_id, str) or _RUN_RE.fullmatch(run_id) is None:
        raise ValueError("Ledger run ID is not a safe path component")
    parts = (*_relative_parts(snapshot["output_directory"], allow_dot=True), run_id)
    _check_path(root, parts)
    return root, parts


def _integrity(
    artifact: dict[str, Any],
    anchor: Path,
    root_parts: tuple[str, ...],
    remaining_bytes: int,
) -> tuple[dict[str, Any], int]:
    """Check recorded bytes, without decoding or opening artifacts in a viewer."""
    checked_at_ns = time.time_ns()
    consumed = 0

    def finding(status: str, detail: str, used: int = 0):
        return {"status": status, "detail": detail, "checked_at_ns": checked_at_ns}, used

    try:
        path = _check_path(anchor, (*root_parts, *_relative_parts(artifact["relative_path"])))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return finding("unsafe", "Artifact is not a regular file")
        expected_size = artifact.get("size_bytes")
        expected_hash = artifact.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 <= expected_size <= MAX_ARTIFACT_BYTES
            or not isinstance(expected_hash, str)
            or _HASH_RE.fullmatch(expected_hash) is None
        ):
            return finding("unsafe", "Ledger has an invalid artifact size or SHA-256")
        if info.st_size != expected_size:
            return finding("changed", "File size differs from the recorded artifact")
        if expected_size > remaining_bytes:
            return finding("not_checked", "Inspection reached its 64 MiB hashing limit")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        digest = hashlib.sha256()
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(info, opened):
                return finding("changed", "File changed while inspection opened it")
            # Cap reads even if another process grows the file after the size check.
            while consumed < expected_size:
                chunk = stream.read(min(1024 * 1024, expected_size - consumed))
                if not chunk:
                    break
                consumed += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.lstat()
        if (
            consumed != expected_size
            or after.st_size != expected_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or not os.path.samestat(after, current)
            or current.st_mtime_ns != after.st_mtime_ns
        ):
            return finding("changed", "File changed during inspection", consumed)
        if digest.hexdigest() != expected_hash:
            return finding("changed", "SHA-256 differs from the recorded artifact", consumed)
        return finding("verified", "Size and SHA-256 match the recorded bytes", consumed)
    except FileNotFoundError:
        return finding("missing", "Recorded artifact is not present", consumed)
    except (ValueError, KeyError, RuntimeError) as exc:
        return finding("unsafe", str(exc), consumed)
    except OSError as exc:
        return finding("unreadable", f"Cannot read artifact ({type(exc).__name__})", consumed)


def inspect_job(
    store: SQLiteRunStore,
    run_id: str,
    *,
    operation: str | None = None,
    limit: int = 50,
    offset: int = 0,
    events_limit: int = 100,
) -> dict[str, Any]:
    """Read one ledger view, then check a bounded amount of local artifact bytes.

    Run costs and status describe the entire ledger run. Operations and their
    attempt lineage are paged; recent events carry a separate limit. Files can
    change after the SQLite snapshot, so integrity findings are timestamped
    observations and never change recorded acceptance or run status.
    """
    snapshot = store.inspection_snapshot(
        run_id, operation=operation, limit=limit, offset=offset, events_limit=events_limit
    )
    snapshot["inspection_version"] = INSPECTION_VERSION
    snapshot["inspected_at_ns"] = time.time_ns()
    hashed = 0
    findings: dict[str, int] = {}
    try:
        anchor, root_parts = _artifact_root(snapshot)
        root_error = None
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        root_error = str(exc)
    for artifact in snapshot["artifacts"]:
        if root_error is not None:
            integrity = {
                "status": "unsafe", "detail": root_error, "checked_at_ns": time.time_ns()
            }
        else:
            integrity, consumed = _integrity(
                artifact, anchor, root_parts, MAX_INSPECTION_HASH_BYTES - hashed
            )
            hashed += consumed
        artifact["integrity"] = integrity
        status = integrity["status"]
        findings[status] = findings.get(status, 0) + 1
    snapshot["artifact_integrity"] = {
        "counts": findings,
        "bytes_hashed": hashed,
        "hash_byte_limit": MAX_INSPECTION_HASH_BYTES,
        "scope": "artifact bytes in the selected operation page; acceptance is unchanged",
    }
    return snapshot
