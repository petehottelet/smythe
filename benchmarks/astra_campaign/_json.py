"""Strict, bounded JSON for the offline campaign preparation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MAX_FILE_BYTES = 256 * 1024
TEXT_HASH_POLICY = "sha256-utf8-text-crlf-to-lf-v1"


class CampaignPlanError(ValueError):
    """A preparation artifact violates the frozen protocol."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(data: bytes) -> str:
    """Hash UTF-8 text with CRLF normalized to LF; no other rewriting."""
    text = data.decode("utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_json(text: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CampaignPlanError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise CampaignPlanError(f"Nonfinite JSON number: {value}")

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject,
                          parse_float=floating)
    except (ValueError, TypeError, RecursionError) as exc:
        raise CampaignPlanError(f"Invalid campaign JSON: {exc}") from exc


def read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise CampaignPlanError(f"Campaign file exceeds {MAX_FILE_BYTES} bytes")
        return raw
    except OSError as exc:
        raise CampaignPlanError(f"Cannot read campaign file: {path.name}") from exc


def read_json(path: Path) -> tuple[object, bytes]:
    raw = read_bytes(path)
    try:
        return strict_json(raw.decode("utf-8")), raw
    except UnicodeError as exc:
        raise CampaignPlanError(f"Campaign file is not UTF-8: {path.name}") from exc


def keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise CampaignPlanError(f"Invalid {name} fields")


def strings(value, name, *, minimum=1):
    if (type(value) is not list or len(value) < minimum
            or any(type(item) is not str or not item.strip() for item in value)):
        raise CampaignPlanError(f"{name} must contain nonempty strings")
