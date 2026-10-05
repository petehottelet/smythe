"""Built-in final validators for text and JSON deliverables.

Each validator interprets a specification, which is JSON data, with versioned
code. Its :class:`~smythe.validation.ValidatorIdentity` is its name, the
version of that code, and the SHA-256 of the specification with every default
filled in, so a policy or record that names the validator pins both. They are
:class:`~smythe.acceptance.Validator` objects: pass them to
``AcceptancePolicy(mode="strict", final_validators=[...])``.

:class:`TextValidator` checks length bounds and required Markdown sections in
order. :class:`JSONValidator` parses the output as strict JSON and runs checks
at JSON paths: presence, equality, a number within a tolerance, a nonempty
string, object and array sizes, exact or required keys, and citations from an
allowed set. Both run locally, make no provider call, and report a rejection,
never an error, for output that fails their checks, including bytes that are
not UTF-8.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from smythe.acceptance import ValidationSubject
from smythe.validation import (
    ValidationFinding,
    ValidationReport,
    ValidatorIdentity,
    _json_data,
    _require_keys,
    hard_finding,
)

TEXT_VALIDATOR_VERSION = "1"
JSON_VALIDATOR_VERSION = "1"

# Counts in JSON checks, and characters in text, accepted by a specification.
_MAX_COUNT = 10_000
_MAX_CHARS = 1_000_000_000
_MAX_SECTIONS = 256
_MAX_CHECKS = 256
_MAX_PATH = 16
_CHECK_ID = re.compile(r"[a-z][a-z0-9-]{1,63}")
_LINES = re.compile(r"\r\n|\r|\n")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
# Matched against a line without trailing spaces and tabs, so that no part of
# a heading is scanned more than once; output can hold megabyte lines.
_HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*))?")
# What a title ignores at its start, and, matched on the reversed title, at its end.
_TITLE_START = re.compile(r"[\s*_]*")
_TITLE_END = re.compile(r"[\s*_:]*")


def _count(value: Any, what: str, *, optional: bool = False, limit: int = _MAX_COUNT) -> int | None:
    if optional and value is None:
        return None
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(f"{what} must be an integer from 0 to {limit:,}")
    return value


def _strings(value: Any, what: str, *, limit: int = _MAX_COUNT) -> list[str]:
    if (not isinstance(value, list) or len(value) > limit
            or any(not isinstance(item, str) or not item.strip() for item in value)):
        raise ValueError(f"{what} must be a list of at most {limit} nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{what} must not repeat a value")
    return value


def _spec(spec: Any, what: str, defaults: dict[str, Any], required: frozenset[str] = frozenset()) -> dict:
    """A detached copy of JSON object *spec* with every omitted default filled in."""
    try:
        data = _json_data({} if spec is None else spec, what)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    _require_keys(data, what, required, frozenset(defaults))
    return {**defaults, **data}


def _report(identity: ValidatorIdentity, findings: list[ValidationFinding]) -> ValidationReport:
    return ValidationReport("output", tuple(findings), validator=identity)


def _decode(subject: ValidationSubject) -> tuple[str | None, ValidationFinding | None]:
    try:
        return subject.text(), None
    except UnicodeDecodeError:
        return None, hard_finding("not_utf8", "output is not UTF-8 text")


def _title(text: str) -> str:
    """A heading compared ignoring case, spacing, emphasis markers and a final colon."""
    start = _TITLE_START.match(text).end()
    end = len(text) - _TITLE_END.match(text[::-1]).end()
    return " ".join(text[start:end].split()).casefold()


def _heading_text(heading: re.Match[str]) -> str:
    """The text of a line matched by ``_HEADING``, without its closing run of #s."""
    title = heading.group(2) or ""
    # A closing run of #s counts only after a space or tab, or alone.
    unclosed = title.rstrip("#")
    if unclosed != title and (not unclosed or unclosed[-1] in " \t"):
        title = unclosed.rstrip(" \t")
    return title


def _section(name: str) -> str:
    """A required section's name, without the ``#`` marker if it is written as a heading line."""
    heading = _HEADING.fullmatch(name.strip())
    return _heading_text(heading) if heading else name


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """``(line, level, title)`` of each Markdown ATX heading outside fenced code."""
    found: list[tuple[int, int, str]] = []
    fence: re.Pattern[str] | None = None
    for index, line in enumerate(lines):
        if fence is not None:
            if fence.match(line):
                fence = None
            continue
        opening = _FENCE.match(line)
        if opening:
            marker = opening.group(1)
            fence = re.compile(f" {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
            continue
        heading = _HEADING.fullmatch(line.rstrip(" \t"))
        if heading:
            found.append((index, len(heading.group(1)), _title(_heading_text(heading))))
    return found


class TextValidator:
    """Checks a text deliverable's length and its required Markdown sections.

    The specification is a JSON object; every key is optional:

    - ``min_chars`` (default 1) and ``max_chars`` (default None, no limit)
      bound the number of characters after surrounding whitespace is removed,
      so the default requires nonempty output;
    - ``sections`` (default ``[]``) lists headings the output must contain. A
      heading is a Markdown ATX heading (``#`` to ``######``) outside fenced
      code, at any level, compared ignoring case, spacing, ``*`` and ``_``
      emphasis and a final colon. A section written as a heading line, such
      as ``## Sources``, loses its ``#`` marker and surrounding spaces when
      the validator is built, so :attr:`spec` lists ``Sources``;
    - ``ordered`` (default True) requires the sections in the listed order;
      other headings may appear between them;
    - ``min_section_chars`` (default 1) is the least number of characters,
      after surrounding whitespace is removed, between a required heading and
      the next heading at its level or above.

    A final step that returns only its own increment, such as a critique
    without the argument it critiques, fails a ``sections`` check. ``name``
    renames the validator, so that one policy can hold two of them.
    """

    def __init__(self, spec: dict[str, Any] | None = None, *, name: str = "smythe.text-spec") -> None:
        data = _spec(spec, "text validator spec", {
            "min_chars": 1, "max_chars": None, "sections": [], "ordered": True,
            "min_section_chars": 1,
        })
        _count(data["min_chars"], "min_chars", limit=_MAX_CHARS)
        _count(data["min_section_chars"], "min_section_chars", limit=_MAX_CHARS)
        if _count(data["max_chars"], "max_chars", optional=True, limit=_MAX_CHARS) is not None and (
                data["max_chars"] < data["min_chars"]):
            raise ValueError("max_chars must not be less than min_chars")
        _strings(data["sections"], "sections", limit=_MAX_SECTIONS)
        data["sections"] = [_section(section) for section in data["sections"]]
        titles = [_title(section) for section in data["sections"]]
        if not all(titles) or len(set(titles)) != len(titles):
            raise ValueError("sections must name distinct, nonempty headings")
        if type(data["ordered"]) is not bool:
            raise ValueError("ordered must be a bool")
        self._spec = data
        self._titles = titles
        self._title_set = frozenset(titles)
        self._identity = ValidatorIdentity.for_config(name, TEXT_VALIDATOR_VERSION, data)

    @property
    def identity(self) -> ValidatorIdentity:
        return self._identity

    @property
    def spec(self) -> dict[str, Any]:
        """A copy of the specification with every default filled in."""
        return json.loads(json.dumps(self._spec))

    def validate(self, subject: ValidationSubject) -> ValidationReport:
        text, undecodable = _decode(subject)
        if text is None:
            return _report(self._identity, [undecodable])
        spec, findings = self._spec, []
        length = len(text.strip())
        if length < spec["min_chars"]:
            findings.append(hard_finding(
                "text_too_short", f"output has {length} characters; at least "
                f"{spec['min_chars']} are required", expected=spec["min_chars"], observed=length))
        if spec["max_chars"] is not None and length > spec["max_chars"]:
            findings.append(hard_finding(
                "text_too_long", f"output has {length} characters; at most "
                f"{spec['max_chars']} are allowed", expected=spec["max_chars"], observed=length))
        lines = _LINES.split(text)
        headings = _headings(lines)
        positions: dict[str, list[int]] = {}
        for position, (_, _, title) in enumerate(headings):
            positions.setdefault(title, []).append(position)
        # Each required heading once, in the order it first appears.
        order = [title for title in positions if title in self._title_set]
        after, previous = -1, None
        for section, title in zip(spec["sections"], self._titles):
            matches = positions.get(title, [])
            later = [position for position in matches if position > after]
            if not matches:
                findings.append(hard_finding(
                    "section_missing", f"required section {section!r} is missing", expected=section))
                continue
            if spec["ordered"] and not later:
                findings.append(hard_finding(
                    "section_out_of_order", f"section {section!r} must follow {previous!r}",
                    expected=list(spec["sections"]), observed=list(order)))
                continue
            position = later[0] if spec["ordered"] else matches[0]
            if spec["ordered"]:
                after, previous = position, section
            start, level, _ = headings[position]
            end = next((line for line, other, _ in headings[position + 1:] if other <= level),
                       len(lines))
            body = len("\n".join(lines[start + 1:end]).strip())
            if body < spec["min_section_chars"]:
                findings.append(hard_finding(
                    "section_empty", f"section {section!r} has {body} characters; at least "
                    f"{spec['min_section_chars']} are required",
                    expected=spec["min_section_chars"], observed=body))
        return _report(self._identity, findings)


# Each JSON check kind's keys besides id, kind and path, with their defaults;
# _REQUIRED marks a key without one.
_REQUIRED = object()
_CHECK_KEYS: dict[str, dict[str, Any]] = {
    "present": {},
    "equals": {"expected": _REQUIRED},
    "number": {"expected": _REQUIRED, "tolerance": 0},
    "nonempty": {"min_chars": 1},
    "object": {"minimum": 1},
    "length": {"expected": _REQUIRED},
    "keys": {"expected": _REQUIRED},
    "required_keys": {"expected": _REQUIRED},
    "citations": {"allowed": _REQUIRED, "minimum": 1},
}


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _check(check: Any) -> dict[str, Any]:
    """Validate one JSON check and fill in its defaults."""
    kind = check.get("kind") if isinstance(check, dict) else None
    if not isinstance(kind, str) or kind not in _CHECK_KEYS:
        raise ValueError(f"unknown JSON check kind {kind!r}; expected one of {sorted(_CHECK_KEYS)}")
    keys = _CHECK_KEYS[kind]
    _require_keys(check, f"{kind} check",
                  frozenset({"id", "kind", "path"} | {key for key, value in keys.items() if value is _REQUIRED}),
                  frozenset(keys))
    data = {**{key: value for key, value in keys.items() if value is not _REQUIRED}, **check}
    if not isinstance(data["id"], str) or not _CHECK_ID.fullmatch(data["id"]):
        raise ValueError(f"check id {data['id']!r} must be 2 to 64 lowercase letters, digits or "
                         "hyphens, starting with a letter")
    path = data["path"]
    if not isinstance(path, list) or len(path) > _MAX_PATH or any(
            not (isinstance(part, str) and part or type(part) is int and part >= 0) for part in path):
        raise ValueError(f"check {data['id']!r} path must be a list of at most {_MAX_PATH} "
                         "nonempty keys and non-negative indexes")
    if kind == "number":
        if not _finite(data["expected"]) or not _finite(data["tolerance"]) or data["tolerance"] < 0:
            raise ValueError(f"check {data['id']!r} needs a finite expected number and a "
                             "non-negative, finite tolerance")
    for key in ("min_chars", "minimum"):
        if key in data:
            _count(data[key], f"check {data['id']!r} {key}")
    if kind == "length":
        _count(data["expected"], f"check {data['id']!r} expected length")
    if kind in ("keys", "required_keys"):
        _strings(data["expected"], f"check {data['id']!r} expected keys")
    if kind == "citations":
        _strings(data["allowed"], f"check {data['id']!r} allowed citations")
    return data


_TOO_DEEP = "nesting is too deep"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _strict_json(text: str) -> Any:
    """Parse *text* as JSON; duplicate keys and non-finite numbers raise ValueError."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def number(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"number {value} is not finite")
        return parsed

    def constant(value: str) -> Any:
        raise ValueError(f"{value} is not JSON")

    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def _at(value: Any, path: list[str | int]) -> Any:
    for part in path:
        if isinstance(value, list) and type(part) is int:
            value = value[part]
        elif isinstance(value, dict) and isinstance(part, str):
            value = value[part]
        else:
            raise KeyError(part)
    return value


def _passes(check: dict[str, Any], actual: Any) -> bool:
    kind = check["kind"]
    if kind == "present":
        return True
    if kind == "equals":
        return _canonical(actual) == _canonical(check["expected"])
    if kind == "number":
        return _finite(actual) and abs(actual - check["expected"]) <= check["tolerance"]
    if kind == "nonempty":
        return isinstance(actual, str) and len(actual.strip()) >= check["min_chars"]
    if kind == "object":
        return isinstance(actual, dict) and len(actual) >= check["minimum"]
    if kind == "length":
        return isinstance(actual, (list, dict)) and len(actual) == check["expected"]
    if kind == "keys":
        return isinstance(actual, dict) and set(actual) == set(check["expected"])
    if kind == "required_keys":
        return isinstance(actual, dict) and set(check["expected"]) <= set(actual)
    return (isinstance(actual, list) and all(isinstance(item, str) for item in actual)
            and len(set(actual)) == len(actual) >= check["minimum"]
            and set(actual) <= set(check["allowed"]))


def _observed(actual: Any) -> Any:
    """A short description of a checked value for a finding."""
    if actual is None or type(actual) in (bool, int, float):
        return actual
    if isinstance(actual, str):
        return actual if len(actual) <= 80 else f"a string of {len(actual)} characters"
    if isinstance(actual, dict):
        return {"keys": sorted(actual)[:32], "count": len(actual)}
    return {"items": len(actual)}


class JSONValidator:
    """Checks that a deliverable is strict JSON and passes checks at JSON paths.

    The specification is a JSON object with ``checks`` (default ``[]``), a
    list of check objects, and ``max_bytes`` (default 1,048,576), the largest
    output it parses. Output that is not UTF-8, is larger, or is not strict
    JSON (a repeated object key or a non-finite number is not) is rejected,
    and so is nesting too deep for the interpreter to parse or compare.
    Each check has an ``id`` (2 to 64 lowercase letters, digits or hyphens,
    starting with a letter), a ``kind`` and a ``path``: a list of object keys
    and array indexes from the document root, ``[]`` for the root itself. A
    path that does not resolve fails its check. The kinds:

    - ``present``: the path resolves;
    - ``equals``: the value equals ``expected`` as canonical JSON, so ``1``
      does not equal ``1.0`` or ``true``;
    - ``number``: a finite number within ``tolerance`` (default 0) of
      ``expected``;
    - ``nonempty``: a string with at least ``min_chars`` (default 1)
      characters after surrounding whitespace is removed;
    - ``object``: an object with at least ``minimum`` (default 1) keys;
    - ``length``: an array or object with exactly ``expected`` items;
    - ``keys``: an object whose keys are exactly ``expected``;
    - ``required_keys``: an object holding every key in ``expected``;
    - ``citations``: an array of distinct strings, each in ``allowed``, with
      at least ``minimum`` (default 1) of them.

    ``name`` renames the validator, so that one policy can hold two of them.
    """

    def __init__(self, spec: dict[str, Any] | None = None, *, name: str = "smythe.json-spec") -> None:
        data = _spec(spec, "JSON validator spec", {"checks": [], "max_bytes": 1_048_576})
        if type(data["max_bytes"]) is not int or data["max_bytes"] < 1:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(data["checks"], list) or len(data["checks"]) > _MAX_CHECKS:
            raise ValueError(f"checks must be a list of at most {_MAX_CHECKS} checks")
        data["checks"] = [_check(check) for check in data["checks"]]
        ids = [check["id"] for check in data["checks"]]
        if len(set(ids)) != len(ids):
            raise ValueError("check ids must be distinct")
        self._spec = data
        self._identity = ValidatorIdentity.for_config(name, JSON_VALIDATOR_VERSION, data)

    @property
    def identity(self) -> ValidatorIdentity:
        return self._identity

    @property
    def spec(self) -> dict[str, Any]:
        """A copy of the specification with every default filled in."""
        return json.loads(json.dumps(self._spec))

    def validate(self, subject: ValidationSubject) -> ValidationReport:
        if len(subject.data) > self._spec["max_bytes"]:
            return _report(self._identity, [hard_finding(
                "json_too_large", f"output has {len(subject.data)} bytes; at most "
                f"{self._spec['max_bytes']} are parsed",
                expected=self._spec["max_bytes"], observed=len(subject.data))])
        text, undecodable = _decode(subject)
        if text is None:
            return _report(self._identity, [undecodable])
        try:
            document = _strict_json(text)
        except (ValueError, RecursionError) as exc:
            return self._invalid(str(exc) if isinstance(exc, ValueError) else _TOO_DEEP)
        findings = []
        for check in self._spec["checks"]:
            shown = "$" + "".join(f"[{part!r}]" for part in check["path"])
            try:
                actual = _at(document, check["path"])
            except (KeyError, IndexError):
                findings.append(hard_finding(
                    "json_path_missing", f"check {check['id']!r}: {shown} is missing",
                    expected=check["path"]))
                continue
            try:
                passed = _passes(check, actual)
            except (TypeError, ValueError, OverflowError):
                passed = False
            except RecursionError:
                # Python 3.11 counts the frames above the parser against the
                # same limit, so a value it parsed can be too deep to compare.
                return self._invalid(_TOO_DEEP)
            if not passed:
                expected = {key: value for key, value in check.items() if key not in ("id", "path")}
                findings.append(hard_finding(
                    "json_check_failed", f"check {check['id']!r} ({check['kind']}) failed at {shown}",
                    expected=expected, observed=_observed(actual)))
        return _report(self._identity, findings)

    def _invalid(self, message: str) -> ValidationReport:
        return _report(self._identity, [hard_finding(
            "json_invalid", f"output is not strict JSON: {message}"[:200])])


__all__ = [
    "JSON_VALIDATOR_VERSION",
    "TEXT_VALIDATOR_VERSION",
    "JSONValidator",
    "TextValidator",
]
