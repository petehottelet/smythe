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

from smythe.graph import ExecutionGraph, Node, NodeStatus

_FAIL_PATTERN = re.compile(r"\b(fail|failed|reject|rejected)\b", re.IGNORECASE)
_PASS_PATTERN = re.compile(r"\b(pass|passed|approve|approved|ok)\b", re.IGNORECASE)


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
