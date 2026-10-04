"""Delivery acceptance records for Jobs artifacts.

Every completed provider call is decided by validators that inspect the exact
response bytes. The built-in artifact check, ``smythe.jobs.artifact-spec``
version 1, decodes each artifact and compares its MIME type and dimensions with
the operation's artifact spec. The decision is recorded in the call's
completion event under ``delivery_acceptance``, in the same transaction that
settles the call's cost, so the Jobs schema is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from smythe.validation import ValidationOutcome, ValidationReport, ValidatorIdentity

ARTIFACT_SPEC_VALIDATOR = "smythe.jobs.artifact-spec"
ARTIFACT_SPEC_VALIDATOR_VERSION = "1"
DELIVERY_RECORD_VERSION = 1
# The report path for findings about the whole response rather than one artifact.
RESPONSE_PATH = "response"


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

    A rejection of any subject rejects; otherwise an error on any subject
    leaves the validator without a decision.
    """
    outcomes = {report.outcome for report in reports}
    if ValidationOutcome.REJECTED in outcomes:
        outcome = ValidationOutcome.REJECTED
    elif ValidationOutcome.ERROR in outcomes:
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
    "RESPONSE_PATH",
    "artifact_spec_identity",
    "delivery_record",
    "validator_result",
]
