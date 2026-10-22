"""Delivery acceptance records for Jobs artifacts.

Every completed provider call is decided by validators that inspect the exact
response bytes. The built-in artifact check, ``smythe.jobs.artifact-spec``
version 1, decodes each artifact and compares its MIME type and dimensions with
the operation's artifact spec. Validators a manifest names by name and version
are trusted local code registered in-process; they run after it on the same
in-memory bytes. The decision is recorded in the call's completion event under
``delivery_acceptance``, in the same transaction that settles the call's cost,
so the Jobs schema is unchanged. Inspection and resume derive each operation's
delivery eligibility from that record, the current bytes on disk and the
validators its plan requires, and never write it back.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from smythe.acceptance import ValidationSubject, check_validator
from smythe.jobs.models import RESERVED_VALIDATOR_PREFIX
from smythe.validation import ValidationOutcome, ValidationReport, ValidatorIdentity

ARTIFACT_SPEC_VALIDATOR = "smythe.jobs.artifact-spec"
ARTIFACT_SPEC_VALIDATOR_VERSION = "1"
DELIVERY_RECORD_VERSION = 1
# A report keeps at most this many findings, hard findings first.
MAX_RECORDED_FINDINGS = 32
# The report path for findings about the whole response rather than one artifact.
RESPONSE_PATH = "response"

Registry = dict[str, tuple[ValidatorIdentity, Any]]


def validator_registry(validators: Sequence[Any] | None) -> Registry:
    """Check validators for registration and index them, with their identities, by name.

    Each validator passes :func:`smythe.acceptance.check_validator`, which
    refuses providers, classes, coroutine functions, awaitables and an async
    ``validate``. Names are unique and may not use the reserved ``smythe.``
    prefix. The identity is captured here, at registration.
    """
    if validators is None:
        return {}
    if not isinstance(validators, (list, tuple)):
        raise TypeError("validators must be a list or tuple of validator objects")
    registry: Registry = {}
    for validator in validators:
        identity = check_validator(validator)
        if identity.name.startswith(RESERVED_VALIDATOR_PREFIX):
            raise ValueError(
                f"validator name {identity.name!r} uses the reserved "
                f"{RESERVED_VALIDATOR_PREFIX!r} prefix"
            )
        if identity.name in registry:
            raise ValueError(f"validators has more than one validator named {identity.name!r}")
        registry[identity.name] = (identity, validator)
    return registry


def run_validator(
    identity: ValidatorIdentity,
    validator: Any,
    subject: ValidationSubject,
    path: str,
    filename: str,
) -> tuple[ValidationReport, list[str]]:
    """Run one registered validator on one artifact's bytes.

    Returns the report to record for *path* and the attempt-error reasons it
    gives. An exception, a result that is not this validator's
    ``ValidationReport``, or findings that are not JSON data make the outcome
    ``error``: the validator could not decide. Only the exception's type is
    recorded, never its message.
    """
    label = f"{filename}: validator {identity.name!r}"
    try:
        report = validator.validate(subject)
    except Exception as exc:
        return _error_report(identity, path), [f"{label} raised {type(exc).__name__}"]
    if not isinstance(report, ValidationReport):
        if inspect.iscoroutine(report):
            report.close()
        return (_error_report(identity, path),
                [f"{label} returned {type(report).__name__}, not a ValidationReport"])
    if report.validator is not None and report.validator != identity:
        return _error_report(identity, path), [f"{label} returned a report for another validator"]
    findings = report.findings
    if len(findings) > MAX_RECORDED_FINDINGS:
        findings = (report.hard_findings + report.advisories)[:MAX_RECORDED_FINDINGS]
    recorded = ValidationReport(path, findings, report.outcome, identity)
    try:
        recorded.to_dict()
    except (TypeError, ValueError):
        return _error_report(identity, path), [f"{label} returned findings that are not JSON data"]
    if recorded.outcome is ValidationOutcome.ERROR:
        return recorded, [f"{label} could not decide"]
    return recorded, [f"{filename}: {identity.name}: {finding.message}"
                      for finding in recorded.hard_findings]


def _error_report(identity: ValidatorIdentity, path: str) -> ValidationReport:
    return ValidationReport(path, (), ValidationOutcome.ERROR, identity)


def artifact_spec_identity(artifact: Any) -> ValidatorIdentity:
    """The built-in check's identity for one operation's artifact spec.

    *artifact* is an ``ArtifactSpecV1`` or its ``to_dict()`` form. The check's
    configuration is the MIME type and the optional dimensions it compares.
    """
    data = artifact if isinstance(artifact, Mapping) else artifact.to_dict()
    return ValidatorIdentity.for_config(
        ARTIFACT_SPEC_VALIDATOR, ARTIFACT_SPEC_VALIDATOR_VERSION,
        {"mime_type": data.get("mime_type"), "width": data.get("width"),
         "height": data.get("height")},
    )


def validator_result(
    identity: ValidatorIdentity, reports: Sequence[ValidationReport],
) -> dict[str, Any]:
    """One validator's identity, overall outcome and reports as JSON data.

    A rejection of any subject rejects. Otherwise an error on any subject, or
    no decodable subject to check, leaves the validator without a decision.
    """
    outcomes = {report.outcome for report in reports}
    if ValidationOutcome.REJECTED in outcomes:
        outcome = ValidationOutcome.REJECTED
    elif ValidationOutcome.ERROR in outcomes or not reports:
        outcome = ValidationOutcome.ERROR
    else:
        outcome = ValidationOutcome.PASSED
    return {"validator": identity.to_dict(), "outcome": outcome.value,
            "reports": [report.to_dict() for report in reports]}


def delivery_record(
    attempt_id: str,
    subjects: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> dict[str, Any]:
    """The delivery decision for one attempt's response.

    Any validator rejection rejects the response. Without one, a validator
    that could not decide leaves it ``indeterminate``; otherwise it is
    ``accepted``. ``subjects`` lists each artifact's recorded path, SHA-256,
    size and whether its bytes were published.
    """
    outcomes = {result["outcome"] for result in results}
    if ValidationOutcome.REJECTED.value in outcomes:
        decision = "rejected"
    elif ValidationOutcome.ERROR.value in outcomes:
        decision = "indeterminate"
    else:
        decision = "accepted"
    return {
        "version": DELIVERY_RECORD_VERSION,
        "scope": "artifacts",
        "attempt_id": attempt_id,
        "decision": decision,
        "reasons": list(reasons),
        "subjects": [dict(subject) for subject in subjects],
        "validators": [dict(result) for result in results],
    }


_DECISIONS = frozenset({"accepted", "rejected", "indeterminate"})


def _valid_record(record: Any, attempt_id: str) -> bool:
    """Whether *record* has the shape :func:`delivery_record` writes for this attempt."""
    return (
        type(record) is dict
        and record.get("decision") in _DECISIONS
        and record.get("attempt_id") == attempt_id
        and type(record.get("reasons")) is list
        and type(record.get("subjects")) is list
        and all(type(subject) is dict for subject in record["subjects"])
        and type(record.get("validators")) is list
        and all(type(result) is dict and type(result.get("validator")) is dict
                and type(result.get("outcome")) is str for result in record["validators"])
    )


def _required_validators(operation: Mapping[str, Any]) -> list[Any] | None:
    """The identities an operation's current plan requires, or None if unreadable."""
    try:
        spec = operation["spec"]
        required = [artifact_spec_identity(spec["artifact"]).to_dict(),
                    *spec.get("validators", [])]
        return sorted(required, key=lambda identity: identity["name"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def operation_delivery(
    operation: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Project one operation's delivery decision and derive its eligibility.

    The decision is the accepted attempt's, else that of the latest attempt
    that completed. An attempt completed by an earlier release has no record
    and is projected as a legacy built-in decision whose validator is
    unrecorded; it is never reported as accepted or as not evaluated.

    The operation is eligible for delivery only with a recorded ``accepted``
    decision, validators identical to the ones its plan requires now, and
    every accepted artifact verified on disk with the recorded SHA-256.
    *artifacts* maps attempt IDs to artifact rows carrying an ``integrity``
    observation. Eligibility is derived on every read and never stored.
    """
    accepted_id = operation.get("accepted_attempt_id")
    completed = [attempt for attempt in attempts
                 if attempt.get("status") in ("succeeded", "rejected")]
    attempt = next((item for item in completed if item.get("attempt_id") == accepted_id), None)
    if attempt is None and completed:
        attempt = max(completed, key=lambda item: item.get("attempt_number") or 0)
    if attempt is None:
        return None, {"eligible": False, "reasons": ["no_decision"]}
    attempt_id = attempt["attempt_id"]
    accepted = accepted_id is not None and accepted_id == attempt_id
    record = records.get(attempt_id)
    projection: dict[str, Any] = {"source": "recorded", "attempt_id": attempt_id,
                                  "decision": None, "accepted": accepted,
                                  "validators": None, "reasons": []}
    reasons = [] if accepted else ["not_accepted"]
    subjects = None
    if record is None:
        projection["source"] = "legacy_built_in"
        reasons.append("validator_unrecorded")
    elif not _valid_record(record, attempt_id):
        reasons.append("record_invalid")
    else:
        projection.update(
            decision=record["decision"], reasons=list(record["reasons"]),
            validators=[{"validator": result["validator"], "outcome": result["outcome"]}
                        for result in record["validators"]],
        )
        if record["decision"] != "accepted" and accepted:
            reasons.append("not_accepted")
        recorded = sorted((result["validator"] for result in record["validators"]),
                          key=lambda identity: str(identity.get("name")))
        if recorded != _required_validators(operation):
            reasons.append("validator_changed")
        subjects = {(subject.get("path"), subject.get("sha256"))
                    for subject in record["subjects"] if subject.get("published") is True}
    if accepted:
        delivered = [item for item in artifacts.get(attempt_id, ()) if item.get("accepted") == 1]
        if not delivered:
            reasons.append("artifact_missing")
        for artifact in delivered:
            if subjects is not None and (artifact.get("relative_path"),
                                         artifact.get("sha256")) not in subjects:
                reasons.append("record_mismatch")
            status = (artifact.get("integrity") or {}).get("status", "not_checked")
            if status != "verified":
                reasons.append(f"artifact_{status}")
    reasons = list(dict.fromkeys(reasons))
    return projection, {"eligible": not reasons, "reasons": reasons}


def annotate_delivery(snapshot: dict[str, Any], records: Mapping[str, Any]) -> dict[str, Any]:
    """Add ``delivery_acceptance`` and ``delivery_eligibility`` to each operation.

    *records* maps attempt IDs to recorded decisions. Artifact rows should
    already carry an ``integrity`` observation; one without it is unchecked.
    """
    attempts: dict[Any, list[Mapping[str, Any]]] = {}
    for attempt in snapshot.get("attempts", ()):
        attempts.setdefault(attempt.get("operation_id"), []).append(attempt)
    artifacts: dict[Any, list[Mapping[str, Any]]] = {}
    for artifact in snapshot.get("artifacts", ()):
        artifacts.setdefault(artifact.get("attempt_id"), []).append(artifact)
    for operation in snapshot.get("operations", ()):
        acceptance, eligibility = operation_delivery(
            operation, attempts.get(operation.get("operation_id"), ()), artifacts, records,
        )
        operation["delivery_acceptance"] = acceptance
        operation["delivery_eligibility"] = eligibility
    return snapshot


__all__ = [
    "ARTIFACT_SPEC_VALIDATOR",
    "ARTIFACT_SPEC_VALIDATOR_VERSION",
    "DELIVERY_RECORD_VERSION",
    "MAX_RECORDED_FINDINGS",
    "RESPONSE_PATH",
    "annotate_delivery",
    "artifact_spec_identity",
    "delivery_record",
    "operation_delivery",
    "run_validator",
    "validator_registry",
    "validator_result",
]
