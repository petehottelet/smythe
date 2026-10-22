"""Account-identifier redaction for text that a record keeps.

Provider and service error messages can embed account identifiers and masked
keys. Records redact them before they are written: a failed node's result
and trace span, a native response error marker, Jobs call errors, Autotune
trial errors and journaled read and evidence reasons. A repository test
rejects them in every tracked file. ``benchmarks.artifact_records``
re-exports these names. This module imports nothing from smythe.
"""

from __future__ import annotations

import re
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
