"""Delivery acceptance records for Jobs artifacts.

Every completed provider call is decided by validators that inspect the exact
response bytes. The built-in artifact check, ``smythe.jobs.artifact-spec``
version 1, decodes each artifact and compares its MIME type and dimensions with
the operation's artifact spec. Validators a manifest names by name and version
are trusted local code registered in-process; they run after it on the same
in-memory bytes. The decision is recorded in the call's completion event under
``delivery_acceptance``, in the same transaction that settles the call's cost,
so the Jobs schema is unchanged.
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


__all__ = [
    "ARTIFACT_SPEC_VALIDATOR",
    "ARTIFACT_SPEC_VALIDATOR_VERSION",
    "DELIVERY_RECORD_VERSION",
    "MAX_RECORDED_FINDINGS",
    "RESPONSE_PATH",
    "artifact_spec_identity",
    "delivery_record",
    "run_validator",
    "validator_registry",
    "validator_result",
]
