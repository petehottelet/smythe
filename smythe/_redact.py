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

# Where an absolute path starts: a file URL, a drive root, a UNC root (whose
# backslashes a repr or JSON text doubles) or a POSIX root of two or more
# parts. No letter, digit or path character precedes one, so a URL's path and
# relative paths stay, and a POSIX path after an HTTP method is a route.
_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "CONNECT", "TRACE")
_PATH_START = (
    r"(?:\bfile:/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?<![\w.~-])\\{2,4}[^\s\\/'\"`]+[\\/]|"
    + "".join(f"(?<!{method} )" for method in _HTTP_METHODS)
    + r"(?<![\w:/\\.~\]-])/[^\s/'\"`]+/)"
)
# A path that starts a quoted string runs to its closing quote, spaces included.
_QUOTED_PATH = re.compile(r"(['\"`])" + _PATH_START + r"(?:(?!\1)[^\r\n])*\1")
# Any other runs to a space or quote, and on past a space while the next word
# holds a separator or a dot or is capitalized, as the rest of a name with a
# space would be; a URL or an HTTP method and its route is never taken in.
_WORD = r"[^\s'\"`]"
_ABSOLUTE_PATH = re.compile(
    _PATH_START + _WORD + r"*(?: +(?![A-Za-z][\w+.-]*://|(?:" + "|".join(_HTTP_METHODS)
    + r") )(?=[A-ZÀ-ÖØ-Þ]|" + _WORD + r"*(?:[\\/]|\.\w))" + _WORD + r"+)*"
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

    The home folder matches with any run of separators between its names,
    as a repr or JSON text doubles backslashes, and without case on
    Windows, so a user name that holds a space is removed whole. A path that
    starts a quoted string is removed up to the closing quote. Unquoted, a
    path ends at a space unless the next word holds a separator or a dot or
    is capitalized, so a name with a space, such as another user's folder,
    is removed whole, and so is a neighbouring word such as ``and/or``, in
    preference to keeping part of a path; any other word stays, even one of
    the path's last name. URL paths and a route after an HTTP method, as in
    ``POST /v1/chat/completions``, are not local paths and stay.
    """
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = ""
    names = [name for name in re.split(r"[\\/]+", home) if name]
    lead = r"[\\/]+" if home[:1] in ("\\", "/") else ""
    # A home folder at a drive's root names no folder, so nothing is written ~.
    if len(names) >= (1 if lead else 2):
        pattern = lead + r"[\\/]+".join(re.escape(name) for name in names)
        text = re.sub(r"(?<![\w.~-])" + pattern + r"(?![\w.~-])", "~", text,
                      flags=re.IGNORECASE if os.name == "nt" else 0)
    text = _QUOTED_PATH.sub(lambda match: f"{match[1]}<path>{match[1]}", text)
    return _ABSOLUTE_PATH.sub("<path>", text)
