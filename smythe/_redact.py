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
# parts. A UNC root starts after any character, as after a compiler flag, so
# a repr of a relative path, whose separators it doubles, is redacted too. No
# letter, digit or path character precedes a drive or POSIX root, so a URL's
# path and relative paths stay, and a POSIX path after an HTTP method is a
# route. A method is one only as a word of its own, at the start of the text
# or after a space, an opening bracket or a quote: a path after OUTPUT is
# redacted.
_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "CONNECT", "TRACE")
_PATH_START = (
    r"(?:\bfile:/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\{2,4}[^\s\\/'\"`]+[\\/]|"
    + "".join(rf"(?<!(?<![^\s(\[{{'\"`]){method} )" for method in _HTTP_METHODS)
    + r"(?<![\w:/\\.~\]-])/[^\s/'\"`]+/)"
)
# A path that starts a quoted string runs to its closing quote, spaces included.
_QUOTED_PATH = re.compile(r"(['\"`])" + _PATH_START + r"(?:(?!\1)[^\r\n])*\1")
# Any other path's last name cannot be told from the words after it, so it
# runs on past each space to the end of the line, a quote, a URL, or an HTTP
# method and its route, and ends after a word that ends in a colon or a
# semicolon, as a message after a path often starts; a comma, which folder
# names hold, does not end it, nor does a method's name with no route after it.
_WORD = r"[^\s'\"`]"
_ENDS_PATH = ":;"
_ABSOLUTE_PATH = re.compile(
    _PATH_START + _WORD + r"*(?:(?<![" + _ENDS_PATH + r"]) +(?![A-Za-z][\w+.-]*://|(?:"
    + "|".join(_HTTP_METHODS) + r") /)" + _WORD + r"+)*"
)


def _path_mark(match: re.Match[str]) -> str:
    """``<path>`` for an unquoted path, keeping the colon or semicolon that ended it."""
    return "<path>" + match[0][-1] if match[0][-1] in _ENDS_PATH else "<path>"


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


def _redact_other_paths(text: str) -> str:
    """*text* with every absolute path written ``<path>``."""
    text = _QUOTED_PATH.sub(lambda match: f"{match[1]}<path>{match[1]}", text)
    return _ABSOLUTE_PATH.sub(_path_mark, text)


def redact_paths(text: str) -> str:
    """*text* with the home folder written ``~`` and every other absolute path ``<path>``.

    The home folder matches with any run of separators between its names,
    as a repr or JSON text doubles backslashes, and without case on
    Windows, so a user name that holds a space is removed whole. The names
    below it stay, up to a character other than a letter, a digit, ``.``,
    ``~``, ``-`` or ``_``. Elsewhere two backslashes start a UNC root after
    any character, as in a compiler flag, so a repr of a relative path,
    whose separators it doubles, is redacted too. A path that starts a
    quoted string is removed up to the closing quote. Unquoted, a name with
    a space, such as another user's folder, cannot be told from the words
    after it, so they are removed with the path, in preference to keeping
    part of it, up to the end of the line, a quote, a URL, or an HTTP method
    and its route. A word that ends in a colon or a semicolon ends the path
    and keeps its mark, as in ``<path>: access denied``; a comma does not,
    as folder names hold commas. URL paths and a route after an HTTP method
    written as a word of its own, as in ``POST /v1/chat/completions``, are
    not local paths and stay; a path after ``OUTPUT`` or ``TARGET`` is
    redacted.
    """
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = ""
    names = [name for name in re.split(r"[\\/]+", home) if name]
    lead = r"[\\/]+" if home[:1] in ("\\", "/") else ""
    # A home folder at a drive's root names no folder, so nothing is written ~.
    if len(names) < (1 if lead else 2):
        return _redact_other_paths(text)
    # Any character may come before it, as in a compiler flag or a URL. The
    # names below it are split off with it, since a repr's doubled
    # backslashes between them would read as a UNC root.
    pattern = lead + r"[\\/]+".join(re.escape(name) for name in names)
    pieces = re.split(pattern + r"(?![\w.~-])((?:[\\/]+[\w.~-]+)*)", text,
                      flags=re.IGNORECASE if os.name == "nt" else 0)
    # Every second piece holds the names below one appearance of the home folder.
    return "".join("~" + piece if index % 2 else _redact_other_paths(piece)
                   for index, piece in enumerate(pieces))
