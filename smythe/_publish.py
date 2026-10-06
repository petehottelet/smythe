"""Crash-safe publication of new files that never replaces an existing file.

It also holds the bounded check that a recorded file still holds the bytes
its record names, which Jobs inspection, resume and reports share.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

# The largest file published with atomic_publish_bytes, such as a Jobs artifact.
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
# How many numbered names publish_unique tries after the one it was given.
_MAX_ALTERNATIVES = 1000
_HASH_RE = re.compile(r"[0-9a-f]{64}")
# The errors os.link reports on a filesystem without hard links, such as FAT
# or exFAT: EPERM on Linux, ENOTSUP or EOPNOTSUPP on macOS and some network
# filesystems, ENOSYS, and EINVAL, to which Windows maps their error.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP,
                            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)})


def publish_unique(directory: Path, name: str, data: bytes) -> Path:
    """Publish *data* as *name* in *directory*, or under the next free numbered name.

    When ``draw_00.png`` is taken, for example by an earlier run or by a run
    interrupted after it published, the bytes go to ``draw_00-2.png``, then
    ``draw_00-3.png``. No existing file is replaced. Returns the path written.
    Strict runs publish into their ``artifact_dir``, which a missing-hard-link
    error names.
    """
    base = Path(name)
    for number in range(1, _MAX_ALTERNATIVES + 2):
        path = directory / (name if number == 1 else f"{base.stem}-{number}{base.suffix}")
        if os.path.lexists(path):
            continue
        try:
            publish_new_bytes(path, data, remedy="Choose an artifact_dir on a filesystem with hard links")
        except FileExistsError:
            continue  # Another writer took the name after the check.
        return path
    raise FileExistsError(f"Every name for {name!r} is already taken")


def publish_new_bytes(
    path: Path, data: bytes, *, remedy: str = "Publish to a directory on a filesystem with hard links",
) -> None:
    """Durably publish new bytes, atomically refusing an existing destination.

    The fully flushed temporary file is linked into place without replacement.
    Filesystems without hard-link support fail closed with an ``OSError``
    that keeps the link's errno and says so, ending with *remedy*. An
    existing destination, even one holding identical bytes, stays untouched
    and raises ``FileExistsError``.
    """
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno not in _NO_HARD_LINKS:
                raise
            raise OSError(exc.errno, (
                f"Filesystem does not support exclusive artifact publication: {path.parent} is "
                f"on a filesystem without hard links ({exc.strerror or exc}). {remedy}, such as "
                "NTFS, APFS or ext4; FAT and exFAT have none."
            )) from exc
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_publish_bytes(path: Path, data: bytes) -> None:
    """Durably publish new bytes, atomically refusing an existing destination.

    The bytes must fit :data:`MAX_ARTIFACT_BYTES`; then
    :func:`publish_new_bytes` links the fully flushed temporary file into
    place without replacement. Filesystems without hard-link support fail
    closed; :func:`claim_artifact_root` probes this before a directory is
    used. Existing destinations, including identical bytes, always remain
    untouched.
    """
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte persistence limit")
    publish_new_bytes(path, data)


def _regular_path(root: Path, relative: str) -> Path:
    windows, posix = PureWindowsPath(relative), PurePosixPath(relative.replace("\\", "/"))
    if (not relative or windows.drive or windows.root or posix.is_absolute() or ".." in posix.parts
            or not posix.parts or any(":" in part for part in posix.parts)):
        raise ValueError("Unsafe recorded artifact path")
    path = root
    for part in posix.parts:
        path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Artifact ownership does not follow symbolic links or junctions")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Artifact ownership requires regular files")
    return path


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


def claim_artifact_root(root: Path, owner_bytes: bytes, *, legacy: bool = False,
                        legacy_artifacts: Sequence[Mapping[str, Any]] = ()) -> None:
    """Bind a directory to one persistent run identity before dispatch.

    Legacy receipt paths remain in place. An existing conflicting or incomplete
    marker is never replaced. No-clobber publication also protects artifacts
    belonging to an unknown legacy ledger that has not claimed this root yet.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".smythe-run-owner.json"
    try:
        marker.lstat()
    except FileNotFoundError:
        if not legacy and any(root.iterdir()):
            raise ValueError("New artifact namespace already contains unowned files")
        for artifact in legacy_artifacts:
            if not artifact["accepted"]:
                continue
            path = _regular_path(root, artifact["relative_path"])
            size = artifact["size_bytes"]
            if type(size) is not int or not 0 <= size <= MAX_ARTIFACT_BYTES:
                raise ValueError("Invalid legacy artifact size")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                remaining = size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Legacy accepted artifact is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if stream.read(1) or digest.hexdigest() != artifact["sha256"]:
                    raise ValueError("Legacy accepted artifact differs from its receipt")
        try:
            atomic_publish_bytes(marker, owner_bytes)
        except FileExistsError:
            pass  # Another claimant won; exact ownership is checked below.
    marker = _regular_path(root, marker.name)
    with marker.open("rb") as stream:
        if stream.read(len(owner_bytes) + 1) != owner_bytes:
            raise ValueError("Artifact directory belongs to another run or has an incomplete owner marker")
    # Check the actual mounted filesystem even when a marker predates this
    # process. All probe paths are owned by this fresh temporary directory.
    with tempfile.TemporaryDirectory(prefix=".smythe-publication-", dir=root) as directory:
        probe = Path(directory) / "probe"
        atomic_publish_bytes(probe, b"publication probe")
        try:
            atomic_publish_bytes(probe, b"must not replace")
        except FileExistsError:
            if probe.read_bytes() != b"publication probe":
                raise ValueError("Filesystem did not preserve an existing artifact")
        else:
            raise ValueError("Filesystem does not support exclusive artifact publication")


def _fsync_directory(directory: Path) -> None:
    """Persist the directory entry after replace where the OS supports it."""

    if os.name == "nt":
        # Windows does not support opening directories with ``os.open``. The
        # file itself is flushed before replacement or exclusive publication;
        # the standard library cannot separately flush the directory entry.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
