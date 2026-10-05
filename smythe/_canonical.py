"""Canonical JSON bytes and their SHA-256 digests.

Canonical JSON sorts object keys, has no insignificant whitespace, writes
non-ASCII text as UTF-8 and refuses NaN, infinity, non-string keys, cycles and
values that are not JSON data. Fields named ``*_sha256`` hold
:func:`sha256_hex` digests; :func:`sha256_prefixed` gives the
``sha256:<hex>`` form. This module imports nothing from smythe.
"""

from __future__ import annotations

import hashlib
import json


def _check_keys(value: object, active: set[int]) -> None:
    # json.dumps would write an int, float, bool or None key as a string.
    if not isinstance(value, (dict, list, tuple)):
        return
    if id(value) in active:
        raise ValueError("cycles are not JSON data")
    active.add(id(value))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("object keys must be strings")
        children = list(value.values())
    else:
        children = list(value)
    for child in children:
        _check_keys(child, active)
    active.discard(id(value))


def canonical_bytes(value: object) -> bytes:
    """*value* as canonical JSON in UTF-8; raises ValueError for anything else."""
    try:
        _check_keys(value, set())
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"value is not canonical JSON data: {exc}") from exc


def sha256_hex(value: object) -> str:
    """The SHA-256 of *value*'s canonical JSON as 64 lowercase hex characters."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_prefixed(value: object) -> str:
    """The SHA-256 of *value*'s canonical JSON as ``sha256:<hex>``."""
    return "sha256:" + sha256_hex(value)
