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

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from smythe.graph import Node

_FAIL_PATTERN = re.compile(r"\b(fail|failed|reject|rejected)\b", re.IGNORECASE)
_PASS_PATTERN = re.compile(r"\b(pass|passed|approve|approved|ok)\b", re.IGNORECASE)


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


class TokenVerifier(Verifier):
    """Reads a verdict from the verifier node's text.

    Accepts either strict JSON (``{"passed": false, "reason": "..."}``)
    or plain prose containing a PASS/FAIL keyword.  When the output says
    neither, the result is treated as a **pass**: an unreadable verdict
    must not be able to burn a run's regeneration budget in a loop.
    """

    def verdict(self, verifier_node: Node, verified_node: Node) -> Verdict:
        text = str(verifier_node.result or "").strip()
        if not text:
            return Verdict(passed=True, reason="verifier produced no output")

        parsed = self._parse_json(text)
        if parsed is not None:
            return parsed

        head = text[:600]
        fail = _FAIL_PATTERN.search(head)
        passed = _PASS_PATTERN.search(head)
        if fail and (not passed or fail.start() < passed.start()):
            return Verdict(passed=False, reason=head.splitlines()[0][:200])
        return Verdict(passed=True, reason="")

    @staticmethod
    def _parse_json(text: str) -> Verdict | None:
        cleaned = text
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) < 2:
                return None
            cleaned = parts[1].removeprefix("json").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "passed" not in data:
            return None
        return Verdict(
            passed=bool(data["passed"]),
            reason=str(data.get("reason", ""))[:200],
        )


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
