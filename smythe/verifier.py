"""Verification that gates — a failed check sends work back, not forward.

Smythe could already *score* work: a red-team node, a vision judge.
What it could not do was act on the score.  A judge that finds a
misspelling in a generated ad, or a reviewer that rejects a draft,
recorded its verdict and the pipeline carried on regardless.

A verifier closes that loop.  Verification is an ordinary node — so it
is planned, budgeted, traced, and checkpointed like any other work —
that declares which node it judges:

    Node(id="judge", label="Check the draft", depends_on=["draft"],
         verifies="draft", max_regenerations=2)

When the judge fails its target, the executor resets that target and
everything downstream of it back to pending and runs them again, up to
``max_regenerations`` times.  That is select-from-N generalised: the
same machinery regenerates a defective image, an unsupported claim, or
a draft that missed a constraint.

Regeneration costs real money, so it is bounded per verifier and every
attempt goes through the normal budget reservation.
"""

from __future__ import annotations

import bisect
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from smythe.graph import ExecutionGraph, Node, NodeStatus


class VerificationRecoveryError(RuntimeError):
    """Verification control cannot be safely persisted or recovered."""


def verification_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise VerificationRecoveryError(f"{name} must be a non-negative integer")
    return value


def node_generation(node: Node) -> int:
    return verification_integer(node.metadata.get("execution_generation", 0), "execution_generation")


def verification_pending(graph: ExecutionGraph) -> bool:
    return any(
        "regeneration_intent" in node.metadata
        or (isinstance(node.metadata.get("verification_receipt"), dict)
            and node.metadata["verification_receipt"].get("state") == "pending")
        for node in graph.nodes
    )


def validate_verification_receipt(node: Node, graph: ExecutionGraph) -> None:
    receipt = node.metadata["verification_receipt"]
    if (
        not isinstance(receipt, dict) or type(receipt.get("version")) is not int
        or receipt["version"] != 1 or receipt.get("state") not in ("pending", "consumed")
        or node.status is not NodeStatus.COMPLETED
        or not node.verifies or node.max_regenerations <= 0
    ):
        raise VerificationRecoveryError(f"Invalid verification receipt on {node.id!r}")
    if verification_integer(receipt.get("judge_generation"), "judge generation") != node_generation(node):
        raise VerificationRecoveryError(f"Stale verification receipt on {node.id!r}")
    if receipt.get("target_id") != node.verifies:
        raise VerificationRecoveryError(f"Verifier {node.id!r} target identity changed")
    target = next((n for n in graph.nodes if n.id == node.verifies), None)
    if target is None:
        if receipt.get("target_generation") is not None:
            raise VerificationRecoveryError("Missing verification target has a generation")
    elif verification_integer(receipt.get("target_generation"), "target generation") != node_generation(target):
        raise VerificationRecoveryError(f"Verifier {node.id!r} observed an obsolete target")


def validate_verification_checkpoint(
    graph: ExecutionGraph, *, version: int, completed: bool,
) -> None:
    """Reject ambiguous legacy recovery without rejudging accepted work."""
    for node in graph.nodes:
        node_generation(node)
        used = verification_integer(node.metadata.get("regenerations_used", 0), "regenerations_used")
        # An intent contains its own source identities and is validated before
        # recovery mutates anything. It remains authoritative during replay.
        if "regeneration_intent" in node.metadata:
            continue
        if "verification_receipt" in node.metadata:
            validate_verification_receipt(node, graph)
        if not node.verifies or node.max_regenerations <= 0:
            continue
        receipt = node.metadata.get("verification_receipt")
        if node.status is not NodeStatus.COMPLETED or receipt is not None:
            continue
        if version < 3:
            if not completed and used < node.max_regenerations:
                raise VerificationRecoveryError(
                    f"Legacy checkpoint has no verification disposition for {node.id!r}; "
                    "reconcile the saved verdict before resuming or start a new workflow."
                )
            # A terminal legacy snapshot was saved after verdict consumption;
            # exhausted and advisory gates cannot purchase another generation.
            target = next((n for n in graph.nodes if n.id == node.verifies), None)
            node.metadata["verification_receipt"] = {
                "version": 1, "state": "consumed", "target_id": node.verifies,
                "judge_generation": node_generation(node),
                "target_generation": node_generation(target) if target is not None else None,
                "reason": "legacy completed or exhausted disposition",
            }
            continue
        raise VerificationRecoveryError(f"Completed verifier {node.id!r} has no durable receipt")


@dataclass(frozen=True)
class Verdict:
    """The outcome of a verification."""

    passed: bool
    reason: str = ""


class Verifier(ABC):
    """Turns a verifier node's output into a pass/fail decision."""

    @abstractmethod
    def verdict(self, verifier_node: Node, verified_node: Node) -> Verdict:
        """Judge *verified_node* from the output of *verifier_node*."""


# ---------------------------------------------------------------------------
# Reading a judge's reply
# ---------------------------------------------------------------------------
# The reply is untrusted model output, so reading it takes linear time and
# never raises.  A false PASS lets a defective deliverable through; a false
# FAIL costs at most ``max_regenerations`` paid retries.  Every rule below
# therefore errs toward FAIL.  docs/verifier.md states the rules exactly.

_MAX_VERDICT_CHARS = 200_000
_MAX_JSON_CANDIDATES = 64
_MAX_LABEL_CHARS = 80

_PASS = r"PASS(?:ED|ES)?"
_FAIL = r"FAIL(?:ED|S)?"
# Prose verdict words are uppercase: "does not fail any criterion" is prose.
_VERDICT_WORD = re.compile(rf"\b(?:{_PASS}|{_FAIL})\b")
# "PASS or FAIL", "PASS/FAIL" and "PASS-FAIL" name the choice without making it.
_JOIN = r"(?:[ \t]{0,3}(?:/|\||\bor\b)[ \t]{0,3}|-)"
_CHOICE = re.compile(rf"\b(?:{_PASS}{_JOIN}{_FAIL}|{_FAIL}{_JOIN}{_PASS})\b", re.IGNORECASE)
_CHOICE_MARK = "pass/fail"
_CHOICE_NOTE = re.compile(r"\(\s*pass/fail\s*\)")
_LABEL_HINT = re.compile(rf"verdict|{_CHOICE.pattern}", re.IGNORECASE)
_WHOLE_REPLY = re.compile(r"pass(?:ed|es)?|fail(?:ed|s)?", re.IGNORECASE)
_REPLY_DECORATION = " \t\r\n.!*_`'\"#>:"
_FIRST_LINE = re.compile(r"[^\n\r\v\f\x1c-\x1e\x85\u2028\u2029]*")
_JSON_BRACKET = re.compile(r"[{\[]")
_JSON_OPEN = re.compile(r'\{\s*"|\[\s*\{')
_PASSED_KEY = re.compile(r'"passed"\s*:')
_JSON_MARK = "{…}"  # stands in for JSON removed from the prose
_LIST_ITEM = re.compile(r"[ \t]*(?:[-+•*]|\d{1,3}[.)])[ \t]+")
_BLOCKQUOTE = re.compile(r"[ \t]*>")
_LETTERS = re.compile(r"[A-Za-z]+")
_WORD_RUN = re.compile(r"[A-Za-z'’]+")

_TRIM = " \t\u00a0*_`#>|✅✔✓☑\ufe0f"  # decoration around a verdict word or label
_DECORATION = frozenset(_TRIM)
_SPACES = frozenset(" \t\u00a0")
_CLOSERS = frozenset("\"')]”’*_`")
_SENTENCE_END = frozenset(".!?…")
_DASHES = frozenset("—–")
_SPACED_SEPARATORS = frozenset(".!:;,-")
_AFTER_SEPARATOR = frozenset(" \t\u00a0.!:;,-—–*_`)]\"'”’")
_CLAUSE_TAIL = re.compile(r'[^.!?;:,()\[\]"“”—–-]*')  # matched against reversed text
_CONDITION_GAP = frozenset(" \t\u00a0*_`,:;-–—([")
_LABEL_SEPARATORS = (":", "=", "—", "–", " - ", "|")
_LABEL_END = frozenset(":=-–—,")

_NEGATORS = frozenset({
    "not", "no", "never", "cannot", "cant", "nor", "neither", "hardly",
    "fail", "fails", "failed", "failing",
})
_NEGATION_HINTS = ("no", "n't", "n’t", "cant", "never", "neither", "hardly", "fail")
_CONDITIONS = frozenset({
    "if", "unless", "provided", "providing", "assuming", "pending", "once",
    "conditional", "conditionally", "contingent",
})
# A verdict label names the final verdict: "Verdict:", "Final answer:",
# "Overall result:", "PASS/FAIL:".  "Criterion 2:" and "Result:" do not.
_LABEL_HEADS = frozenset({"verdict", "final", "overall", _CHOICE_MARK})
_LABEL_WORDS = _LABEL_HEADS | frozenset({
    "my", "our", "the", "is", "gate", "judge", "result", "answer", "decision",
    "conclusion", "outcome", "assessment", "evaluation", "grade", "status",
    "summary", "judgment", "judgement",
})
# A line labelled "Verdict:" must state a verdict.
_VERDICT_LABELS = frozenset({"verdict", _CHOICE_MARK})


class _Conflict:
    """Stands in for a ``passed`` key repeated with different values."""

    __slots__ = ()


_CONFLICT = _Conflict()


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj = dict(pairs)
    values = [value for key, value in pairs if key == "passed"]
    if any(value is not values[0] for value in values[1:]):
        obj["passed"] = _CONFLICT
    return obj


_JSON_DECODER = json.JSONDecoder(object_pairs_hook=_json_object)


class _Evidence:
    """Everything a reply says, collected before a decision is made."""

    __slots__ = ("passes", "fails", "alarms", "words", "invalid")

    def __init__(self) -> None:
        self.passes: list[str] = []  # reasons given with PASS verdicts
        self.fails: list[str] = []  # reasons given with FAIL verdicts
        self.alarms: list[tuple[str, str]] = []  # (line, why) other grounds to fail
        self.words: set[str] = set()  # verdict words the prose uses
        self.invalid: str | None = None  # the first unreadable JSON verdict

    def alarm(self, line: str, why: str = "") -> None:
        self.alarms.append((line, why))


def _clip(text: str) -> str:
    return text[:200]


def _json_reason(value: object) -> str:
    if isinstance(value, str):
        return value[:200]
    try:
        return str(value)[:200]
    except (RecursionError, ValueError):
        return type(value).__name__


def _invalid_passed(passed: object) -> str:
    if passed is _CONFLICT:
        return "JSON verdict repeats 'passed' with different values"
    if isinstance(passed, (str, int, float, type(None))):
        shown = repr(passed)[:60]
    else:
        shown = type(passed).__name__
    return f"'passed' must be a JSON boolean, got {shown}"


def _record_json(value: object, evidence: _Evidence) -> None:
    """Record a top-level verdict object.  A nested object can only fail it."""
    pending: list[object] = [value]
    if isinstance(value, dict) and "passed" in value:
        passed = value["passed"]
        reason = _json_reason(value.get("reason", ""))
        if passed is True:
            evidence.passes.append(reason)
        elif passed is False:
            evidence.fails.append(reason)
        elif evidence.invalid is None:
            evidence.invalid = _invalid_passed(passed)
        pending = [child for key, child in value.items() if key != "passed"]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get("passed", True) is False or item.get("passed") is _CONFLICT:
                evidence.alarm("a nested JSON object says 'passed': false")
                return
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _read_json(text: str, evidence: _Evidence) -> str:
    """Record the reply's JSON verdicts and return its prose without JSON."""
    spans: list[tuple[int, int]] = []
    covered = attempts = 0
    for bracket in _JSON_BRACKET.finditer(text):
        start = bracket.start()
        if start < covered or not _JSON_OPEN.match(text, start):
            continue
        if attempts == _MAX_JSON_CANDIDATES:
            break
        attempts += 1
        try:
            value, covered = _JSON_DECODER.raw_decode(text, start)
        except (ValueError, RecursionError):
            continue
        spans.append((start, covered))
        _record_json(value, evidence)
    # Malformed, truncated or unexamined JSON cannot hide a verdict.
    starts = [start for start, _ in spans]
    for key in _PASSED_KEY.finditer(text):
        index = bisect.bisect_right(starts, key.start()) - 1
        if index < 0 or key.start() >= spans[index][1]:
            evidence.invalid = evidence.invalid or "a 'passed' key is not part of valid JSON"
            break
    pieces: list[str] = []
    last = 0
    for start, stop in spans:
        pieces += (text[last:start], _JSON_MARK)
        last = stop
    pieces.append(text[last:])
    return "".join(pieces)


def _is_fence(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("~~~"):
        return True
    return stripped.startswith("```") and "`" not in stripped.lstrip("`")


def _negated(line: str, start: int) -> bool:
    """Whether a negation precedes the word at *start* in the same clause."""
    window = line[max(0, start - 48):start][::-1]
    clause = window[:_CLAUSE_TAIL.match(window).end()][::-1].lower()
    if not any(hint in clause for hint in _NEGATION_HINTS):
        return False
    for word in _WORD_RUN.findall(clause)[-3:]:
        word = word.replace("’", "'")
        if word in _NEGATORS or word.endswith("n't"):
            return True
    return False


def _conditional(line: str, stop: int) -> bool:
    """Whether a condition such as "if" or "provided" follows the word at *stop*."""
    limit = min(len(line), stop + 48)
    i = stop
    while i < limit and line[i] in _CONDITION_GAP:
        i += 1
    word = _LETTERS.match(line, i, limit)
    return word is not None and word.group().lower() in _CONDITIONS


def _ends_verdict(line: str, stop: int) -> bool:
    """Whether the word ending at *stop* ends its line or precedes a separator."""
    i, spaced = stop, False
    while i < len(line) and line[i] in _DECORATION:
        spaced = spaced or line[i] in _SPACES
        i += 1
    if i == len(line) or line[i] in _DASHES:
        return True
    if line[i] == "(":
        return spaced
    if line[i] in _SPACED_SEPARATORS:
        following = line[i + 1:i + 2]
        return not following or following in _AFTER_SEPARATOR
    return False


def _starts_sentence(line: str, start: int) -> bool:
    """Whether the word at *start* begins its line or a new sentence."""
    i, spaced = start, False
    while i > 0 and line[i - 1] in _DECORATION:
        spaced = spaced or line[i - 1] in _SPACES
        i -= 1
    if i == 0:
        return True
    if not spaced:
        return False
    while i > 0 and line[i - 1] in _CLOSERS:
        i -= 1
    return i > 0 and line[i - 1] in _SENTENCE_END


def _label_words(prefix: str) -> list[str] | None:
    """The words of a verdict label such as ``**Final verdict:**``, or None."""
    if len(prefix) > _MAX_LABEL_CHARS:
        return None
    item = _LIST_ITEM.match(prefix)
    text = (prefix[item.end():] if item else prefix).strip(_TRIM)
    if text[-1:] in _LABEL_END:
        text = text[:-1].rstrip(_TRIM)
    words = _CHOICE_NOTE.sub(" ", text).lower().split()
    if 0 < len(words) <= 6 and all(word in _LABEL_WORDS for word in words):
        return words
    return None


def _is_verdict_label(prefix: str) -> bool:
    words = _label_words(prefix)
    return words is not None and any(word in _LABEL_HEADS for word in words)


def _verdict_label_value(line: str) -> str:
    """The text after a ``Verdict:`` or ``PASS/FAIL:`` label, if the line has one."""
    item = _LIST_ITEM.match(line)
    body = (line[item.end():] if item else line).lstrip(_TRIM)
    cuts = [(body.find(mark, 0, _MAX_LABEL_CHARS), mark) for mark in _LABEL_SEPARATORS]
    cuts = [(index, mark) for index, mark in cuts if index > 0]
    if not cuts:
        return ""
    cut, mark = min(cuts)
    words = _label_words(body[:cut])
    if words is None or not any(word in _VERDICT_LABELS for word in words):
        return ""
    return body[cut + len(mark):].strip(_TRIM)


def _read_line(line: str, quoted: bool, evidence: _Evidence) -> None:
    listed = _LIST_ITEM.match(line) is not None
    stated = False
    for index, word in enumerate(_VERDICT_WORD.finditer(line)):
        start, stop = word.span()
        passing = word.group().startswith("P")
        evidence.words.add("PASS" if passing else "FAIL")
        if not passing:
            evidence.alarm(line)  # an uppercase FAIL anywhere fails the reply
        elif _negated(line, start) or _conditional(line, stop):
            evidence.alarm(line, "negated or conditional PASS")
            continue
        if not _ends_verdict(line, stop):
            continue
        if not ((index == 0 and _is_verdict_label(line[:start]))
                or (not listed and _starts_sentence(line, start))):
            continue
        stated = True
        if not passing:
            evidence.fails.append(_clip(line))
        elif not quoted:  # quoted material never supplies a PASS
            evidence.passes.append("")
    if not stated and not quoted:
        value = _verdict_label_value(line)
        if value and _JSON_MARK not in value:
            evidence.alarm(line, "unreadable verdict")


def _read_prose(prose: str, evidence: _Evidence) -> None:
    fenced = False
    for line in prose.splitlines():
        if ("```" in line or "~~~" in line) and _is_fence(line):
            fenced = not fenced
            continue
        if ("PASS" not in line and "FAIL" not in line
                and not _LABEL_HINT.search(line, 0, _MAX_LABEL_CHARS + 16)):
            continue
        line, choices = _CHOICE.subn(_CHOICE_MARK, line)
        if choices:
            evidence.words.update(("PASS", "FAIL"))
        _read_line(line, fenced or _BLOCKQUOTE.match(line) is not None, evidence)


def _rejected(reason: str) -> Verdict:
    return Verdict(passed=False, reason=_clip(reason))


def _decide(evidence: _Evidence, first_line: str) -> Verdict:
    ambiguous = "ambiguous verdict (both PASS and FAIL): "
    if evidence.invalid is not None:
        return _rejected(evidence.invalid)
    objections = [reason for reason in evidence.fails if reason]
    objections += [line for line, _ in evidence.alarms]
    if evidence.passes and (evidence.fails or evidence.alarms):
        return _rejected(ambiguous + (objections[0] if objections else first_line))
    if evidence.passes:
        return Verdict(passed=True, reason=next((r for r in evidence.passes if r), ""))
    if evidence.fails:
        return Verdict(passed=False, reason=next((r for r in evidence.fails if r), ""))
    if evidence.words == {"PASS", "FAIL"}:
        return _rejected(ambiguous + first_line)
    if evidence.alarms:
        line, why = evidence.alarms[0]
        return _rejected(f"{why}: {line}" if why else line)
    if evidence.words:
        return _rejected(f"no standalone PASS or FAIL verdict: {first_line}")
    return _rejected(f"no PASS or FAIL verdict: {first_line}")


class TokenVerifier(Verifier):
    """Reads a verdict from the verifier node's text, failing closed.

    Accepts JSON verdict objects (``{"passed": false, "reason": "..."}``)
    anywhere in the reply, bare or in a code fence, and prose verdicts: the
    uppercase word PASS or FAIL (PASSED, PASSES, FAILED and FAILS also count)
    alone on a line, leading a line or a sentence, or after a label that
    names the final verdict (``Verdict: PASS``).  A PASS inside a sentence,
    a list item, quoted material or after another label is not a verdict.
    A reply that is only the verdict word may use any case.

    Anything else is a **fail**: empty or unreadable output, a ``passed``
    value that is not a JSON boolean (``"false"`` is not ``false``),
    verdicts that disagree, an uppercase FAIL anywhere in the prose, and a
    negated or conditional PASS.  A gate must not wave work through because
    its judge was unreadable.  The cost of failing closed is bounded: the
    executor regenerates at most ``max_regenerations`` times, then lets the
    run finish with the last output, and a gate with ``max_regenerations=0``
    is never read.  docs/verifier.md lists the exact rules.
    """

    def verdict(self, verifier_node: Node, verified_node: Node) -> Verdict:
        text = str(verifier_node.result or "").strip()
        if not text:
            return Verdict(passed=False, reason="verifier produced no output")
        if len(text) > _MAX_VERDICT_CHARS:
            return Verdict(
                passed=False,
                reason=f"verifier output is too long to read ({len(text)} characters)",
            )
        first_line = _FIRST_LINE.match(text).group()
        bare = text.strip(_REPLY_DECORATION)
        if _WHOLE_REPLY.fullmatch(bare):
            passed = bare[0] in "Pp"
            return Verdict(passed=passed, reason="" if passed else _clip(first_line))
        evidence = _Evidence()
        _read_prose(_read_json(text, evidence), evidence)
        return _decide(evidence, first_line[:150])


class CallableVerifier(Verifier):
    """Adapts a plain function into a Verifier.

    Useful for objective gates that need no model at all — image
    dimensions, JSON schema conformance, a required section heading.
    """

    def __init__(self, fn) -> None:
        self._fn = fn

    def verdict(self, verifier_node: Node, verified_node: Node) -> Verdict:
        result = self._fn(verifier_node, verified_node)
        if isinstance(result, Verdict):
            return result
        return Verdict(passed=bool(result))
