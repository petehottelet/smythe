"""Small, dependency-free helpers for portable benchmark records."""

from __future__ import annotations

import mimetypes
import platform
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def image_mime_type(path: str | Path) -> str:
    """Return the image MIME type implied by *path* without assuming PNG."""
    common = {
        ".gif": "image/gif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }
    known = common.get(Path(path).suffix.lower())
    if known:
        return known
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "application/octet-stream"


def portable_path(path: str | Path, *, root: Path = REPO_ROOT) -> str:
    """Use a POSIX repo-relative path when the artifact is inside the repo.

    External inputs remain absolute because rewriting them as a misleading
    relative path would make a benchmark record impossible to interpret.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_record_path(path: str | Path, *, root: Path = REPO_ROOT) -> Path:
    """Resolve a portable result-record path against its checkout root."""
    recorded = Path(path)
    return recorded if recorded.is_absolute() else root / recorded


def environment_snapshot(*packages: str) -> dict:
    """Capture interpreter and installed package versions for a result record."""
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == "smythe":
                # Benchmarks commonly run straight from a checkout rather
                # than an installed wheel. Preserve the source version instead
                # of emitting a misleading null in that normal workflow.
                try:
                    from smythe import __version__
                except ImportError:
                    versions[package] = None
                else:
                    versions[package] = __version__
            else:
                versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
