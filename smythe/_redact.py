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
    r"(?:\bfile:/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\{2,4}[^\s\\/'\"]+[\\/]|"
    + "".join(rf"(?<!(?<![^\s(\[{{'\"`]){method} )" for method in _HTTP_METHODS)
    + r"(?<![\w:/\\.~-])/[^\s/'\"]+/)"
)


def _quoted_path(quote: str) -> str:
    """A path that starts a string quoted with *quote*, up to its closing quote.

    The closing quote is the first of the next four quote marks of its kind
    that no letter or digit follows; one that a letter follows, as in
    O'Brien, is part of a name. Looking no further keeps a long line's cost
    linear. Failing that, the string ends at the first quote mark of its
    kind, and, as names hold backticks, a backtick's word is taken in to its
    end.
    """
    rest = rf"[^{quote}\r\n]*"
    closed = rf"{rest}(?:{quote}(?=\w){rest}){{0,3}}{quote}(?!\w)"
    first = rf"{rest}{quote}" + (r"[^\s'\"]*" if quote == "`" else "")
    return rf"{quote}{_PATH_START}(?:{closed}|{first})"


# A path that starts a quoted string runs to its closing quote, spaces included.
_QUOTED_PATH = re.compile("|".join(_quoted_path(quote) for quote in ("'", '"', "`")))
# A URL's bracketed IPv6 host ends in ']', as a log line's tag does, and a path
# after a tag is redacted; the host and the slash after it are taken in and
# kept, so the URL's path reads as any other URL's does.
_IPV6_HOST = r"://\[[0-9A-Fa-f.]*:[0-9A-Fa-f:.]*(?:%[\w.~-]+)?\]/"
# Any other path's last name cannot be told from the words after it, so it
# runs on past each space to the end of the line, a quote, a URL, or an HTTP
# method and its route, and ends after a word that ends in a colon or a
# semicolon, as a message after a path often starts; a comma or a backtick,
# which names hold, does not end it, nor does a method's name with no route
# after it.
_WORD = r"[^\s'\"]"
_ENDS_PATH = ":;"
_ABSOLUTE_PATH = re.compile(
    "(" + _IPV6_HOST + ")|"
    + _PATH_START + _WORD + r"*(?:(?<![" + _ENDS_PATH + r"]) +(?![A-Za-z][\w+.-]*://|(?:"
    + "|".join(_HTTP_METHODS) + r") /)" + _WORD + r"+)*"
)


def _path_mark(match: re.Match[str]) -> str:
    """``<path>`` for an unquoted path, keeping the colon or semicolon that ended it.

    A URL's IPv6 host is kept as it is.
    """
    if match[1]:
        return match[0]
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


def redact_paths(text: str) -> str:
    """*text* with the home folder written ``~`` and every other absolute path ``<path>``.

    The home folder matches with any run of separators between its names,
    as a repr or JSON text doubles backslashes, and without case on
    Windows, so a user name that holds a space is removed whole, and the
    names below it stay. Elsewhere two backslashes start a UNC root after
    any character, as in a compiler flag, so a repr of a relative path,
    whose separators it doubles, is redacted too. A path that starts a
    quoted string is removed up to the closing quote, which no letter or
    digit follows, as one in ``O'Brien`` does. Unquoted, a name with a
    space, such as another user's folder, cannot be told from the words
    after it, so they are removed with the path, in preference to keeping
    part of it, up to the end of the line, a quote, a URL, or an HTTP method
    and its route. A word that ends in a colon or a semicolon ends the path
    and keeps its mark, as in ``<path>: access denied``; a comma or a
    backtick does not, as names hold them. URL paths, including one after a
    bracketed IPv6 host, and a route after an HTTP method written as a word
    of its own, as in ``POST /v1/chat/completions``, are not local paths and
    stay; a path after ``OUTPUT``, ``TARGET`` or a log tag such as
    ``[worker-1]`` is redacted.
    """
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = ""
    names = [name for name in re.split(r"[\\/]+", home) if name]
    # A POSIX home folder takes the one separator before it and leaves any
    # other to the path that comes before it; a UNC one takes its whole root.
    lead = ""
    if home[:1] in ("\\", "/"):
        lead = r"[\\/]+" if home[1:2] in ("\\", "/") else r"[\\/]"
    stand_in = "\\"
    # A home folder at a drive's root names no folder, so nothing is written ~.
    if len(names) >= (1 if lead else 2):
        pattern = lead + r"[\\/]+".join(re.escape(name) for name in names)
        # The names below it run up to the home folder again or to a name
        # before a colon, which can be a drive or a URL's scheme.
        below = rf"(?:(?![\\/]*{pattern})[\\/]+[\w.~-]+(?![\w.~:-]))*"
        # A character the text does not hold stands in for each backslash
        # there that another follows, so a repr's doubled backslashes between
        # those names do not read as a UNC root. It is turned back at the end.
        stand_in = next((chr(code) for code in range(0xE000, 0xF900) if chr(code) not in text),
                        stand_in)

        def home_folder(match: re.Match[str]) -> str:
            return "~" + re.sub(r"\\(?=\\)", lambda _: stand_in, match[1])

        # Any character may come before it, as in a compiler flag or a URL.
        text = re.sub(rf"{pattern}(?![\w.~-])({below})", home_folder, text,
                      flags=re.IGNORECASE if os.name == "nt" else 0)
    text = _QUOTED_PATH.sub(lambda match: f"{match[0][0]}<path>{match[0][0]}", text)
    return _ABSOLUTE_PATH.sub(_path_mark, text).replace(stand_in, "\\")
