"""Crash-safe publication of new files that never replaces an existing file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_new_bytes(path: Path, data: bytes) -> None:
    """Durably publish new bytes, atomically refusing an existing destination.

    The fully flushed temporary file is linked into place without replacement.
    Filesystems without hard-link support fail closed. An existing
    destination, even one holding identical bytes, stays untouched and raises
    ``FileExistsError``.
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
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
