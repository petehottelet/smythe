"""Crash-safe publication of new files that never replaces an existing file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# How many numbered names publish_unique tries after the one it was given.
_MAX_ALTERNATIVES = 1000


def publish_unique(directory: Path, name: str, data: bytes) -> Path:
    """Publish *data* as *name* in *directory*, or under the next free numbered name.

    When ``draw_00.png`` is taken, for example by an earlier run or by a run
    interrupted after it published, the bytes go to ``draw_00-2.png``, then
    ``draw_00-3.png``. No existing file is replaced. Returns the path written.
    """
    base = Path(name)
    for number in range(1, _MAX_ALTERNATIVES + 2):
        path = directory / (name if number == 1 else f"{base.stem}-{number}{base.suffix}")
        if os.path.lexists(path):
            continue
        try:
            publish_new_bytes(path, data)
        except FileExistsError:
            continue  # Another writer took the name after the check.
        return path
    raise FileExistsError(f"Every name for {name!r} is already taken")


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
