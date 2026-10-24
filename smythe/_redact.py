"""Account-identifier and local-path redaction for text that a record keeps.

Provider and service error messages can embed account identifiers and masked
keys. Records redact them before they are written: a failed node's result
and trace span, a native response error marker, Jobs call errors, Autotune
trial errors and journaled read and evidence reasons. A repository test
rejects them in every tracked file. ``benchmarks.artifact_records``
re-exports the identifier names. ``redact_paths`` also removes local paths,
as planner history does from the failure excerpts it keeps. This module
imports nothing from smythe.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ACCOUNT_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\borg-[A-Za-z0-9]{20,}"), "org-[redacted]"),
    (re.compile(r"\bproj_[A-Za-z0-9]{20,}"), "proj_[redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_*-]{32,}"), "sk-[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), "AIza[redacted]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_-]{30,}"), "AQ.[redacted]"),
    (re.compile(r"\bgen-lang-client-[0-9]{6,}"), "gen-lang-client-[redacted]"),
    (re.compile(r"\bproject_number[:=] ?[0-9]{6,}"), "project_number:[redacted]"),
)

# An absolute path, up to the next space or quote: a file URL, one with a
# drive or UNC root, or a POSIX one of two or more parts that no letter,
# digit or path character precedes, so a URL's path and relative paths stay.
_ABSOLUTE_PATH = re.compile(
    r"(?:\bfile:/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[^\s\\/'\"]+[\\/]"
    r"|(?<![\w:/\\.~-])/[^\s/'\"]+/)[^\s'\"]*"
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


def error_text(error: object) -> str:
    """*error*'s message, as a record keeps it: with account identifiers redacted."""
    return redact_account_identifiers(str(error))


def redact_paths(text: str) -> str:
    """*text* with the home folder written ``~`` and every other absolute path ``<path>``.

    The home folder matches with either separator, and without case on
    Windows, so a user name that holds a space is removed whole.
    """
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = ""
    if home.strip("\\/"):
        flags = re.IGNORECASE if os.name == "nt" else 0
        for spelling in {home, home.replace("\\", "/")}:
            text = re.sub(re.escape(spelling) + r"(?![\w.~-])", "~", text, flags=flags)
    return _ABSOLUTE_PATH.sub("<path>", text)
