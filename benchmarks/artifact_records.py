"""Small, dependency-free helpers for portable benchmark records."""

from __future__ import annotations

import mimetypes
import os
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

# Provider error messages can embed account identifiers and masked keys.
# Records redact them before they are written; a repository test rejects them
# in every tracked file.
ACCOUNT_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\borg-[A-Za-z0-9]{20,}"), "org-[redacted]"),
    (re.compile(r"\bproj_[A-Za-z0-9]{20,}"), "proj_[redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_*-]{32,}"), "sk-[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), "AIza[redacted]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_-]{30,}"), "AQ.[redacted]"),
    (re.compile(r"\bgen-lang-client-[0-9]{6,}"), "gen-lang-client-[redacted]"),
    (re.compile(r"\bproject_number[:=] ?[0-9]{6,}"), "project_number:[redacted]"),
)


def redact_account_identifiers(value: Any) -> Any:
    """Return *value* with account identifiers redacted from every string in it."""
    if isinstance(value, str):
        for pattern, replacement in ACCOUNT_IDENTIFIER_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {
            redact_account_identifiers(key): redact_account_identifiers(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_account_identifiers(item) for item in value]
    return value


PRIVATE_EVIDENCE_OVERRIDE = "SMYTHE_ALLOW_PRIVATE_EVIDENCE_PATHS"


def _home_spellings() -> set[str]:
    home = Path.home()
    spellings = {str(home)}
    if os.name == "nt":
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.kernel32.GetShortPathNameW(str(home), buffer, len(buffer)):
                spellings.add(buffer.value)  # the 8.3 short form, whose last part ends in ~1
        except (AttributeError, OSError):
            pass
    return {spelling for spelling in spellings if spelling.strip("\\/")}


def redact_local_paths(text: str) -> str:
    """Replace the current user's home folder, in any common spelling, with ``~``.

    Forward, back and JSON-escaped separators all match, case-insensitively on
    Windows, so logs, tracebacks and serialized records can be scrubbed alike.
    """
    flags = re.IGNORECASE if os.name == "nt" else 0
    for spelling in sorted(_home_spellings(), key=len, reverse=True):
        parts = [part for part in re.split(r"[\\/]+", spelling) if part]
        lead = r"[\\/]+" if spelling[:1] in "\\/" else ""
        pattern = lead + r"[\\/]+".join(re.escape(part) for part in parts) + r"(?![A-Za-z0-9_.~-])"
        text = re.sub(pattern, "~", text, flags=flags)
    return text


def scrub_record(value: Any) -> Any:
    """Redact account identifiers and the home folder from every string in a record."""
    if isinstance(value, str):
        return redact_local_paths(redact_account_identifiers(value))
    if isinstance(value, dict):
        return {scrub_record(key): scrub_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_record(item) for item in value]
    return value


def evidence_directory(path: str | Path) -> Path:
    """Refuse a new evidence directory whose absolute path would expose private locations.

    Sealed campaign records bind absolute directories. One inside the home
    folder records the user name; one inside the private planning folder
    records its name. Set SMYTHE_ALLOW_PRIVATE_EVIDENCE_PATHS=1 only for tests
    and local experiments whose records will never be published.
    """
    resolved = Path(path).resolve()
    if os.environ.get(PRIVATE_EVIDENCE_OVERRIDE) == "1":
        return resolved
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise ValueError(
            "Evidence directories must be outside the home folder because sealed records "
            f"bind their absolute path; set {PRIVATE_EVIDENCE_OVERRIDE}=1 only for records "
            "that will never be published")
    if any(part.lower() == "00_project_files" for part in resolved.parts):
        raise ValueError("Evidence directories must be outside the private planning folder "
                         "because sealed records bind their absolute path")
    return resolved


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


def _source_control_snapshot() -> dict:
    """Distinguish a clean revision from measurements of an edited checkout.

    ``dirty`` reports changes to tracked files only. Untracked files, such as
    records an earlier run wrote, are counted separately so they cannot be
    mistaken for edited sources.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "untracked_files": None}
    lines = [line for line in status.splitlines() if line.strip()]
    untracked = sum(line.startswith("??") for line in lines)
    return {"revision": revision, "dirty": len(lines) > untracked, "untracked_files": untracked}


def environment_snapshot(*packages: str) -> dict:
    """Capture imported Smythe identity separately from installed metadata."""
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    installed_versions = dict(versions)
    source = None
    if "smythe" in versions:
        try:
            import smythe
        except ImportError:
            pass
        else:
            versions["smythe"] = smythe.__version__
            source = {
                "version": smythe.__version__,
                "module": portable_path(smythe.__file__),
            }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "installed_packages": installed_versions,
        "smythe_source": source,
        "source_control": _source_control_snapshot(),
    }
