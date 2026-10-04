"""Validation findings, reports and validator identities, without an image library."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess

import pytest

from fakes import child_executable, offline_env
from smythe.validation import (
    FindingSeverity,
    ValidationFinding,
    ValidationOutcome,
    ValidationReport,
    ValidatorIdentity,
    advisory_finding,
    hard_finding,
)


def test_validation_types_import_without_an_image_library():
    code = (
        "import sys\n"
        "sys.modules['PIL'] = None\n"
        "import smythe.validation\n"
        "report = smythe.validation.ValidationReport('output')\n"
        "assert report.passed and not [m for m in sys.modules if m.startswith('PIL') and sys.modules[m]]\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [child_executable(), "-c", code], env=offline_env(), capture_output=True,
        text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_report_outcome_follows_hard_findings_when_not_stated():
    advisory = advisory_finding("palette_drift", "palette drifts from the brand")
    hard = hard_finding("blank_image", "image is blank")

    assert ValidationReport("a.png").outcome is ValidationOutcome.PASSED
    assert ValidationReport("a.png", (advisory,)).outcome is ValidationOutcome.PASSED
    rejected = ValidationReport("a.png", (advisory, hard))
    assert rejected.outcome is ValidationOutcome.REJECTED
    assert not rejected.passed
    assert rejected.hard_findings == (hard,) and rejected.advisories == (advisory,)


def test_error_outcome_never_passes_and_stated_outcomes_must_match_findings():
    error = ValidationReport("output", outcome="error")
    assert error.outcome is ValidationOutcome.ERROR and not error.passed

    hard = hard_finding("too_short", "output is too short")
    assert ValidationReport("output", (hard,), outcome="error").outcome == "error"
    assert ValidationReport("output", (hard,), outcome="rejected").passed is False
    assert ValidationReport("output", outcome="passed").passed is True
    with pytest.raises(ValueError, match="passed report"):
        ValidationReport("output", (hard,), outcome="passed")
    with pytest.raises(ValueError, match="rejected report"):
        ValidationReport("output", outcome="rejected")
    with pytest.raises(ValueError, match="outcome"):
        ValidationReport("output", outcome="approved")
    with pytest.raises(TypeError, match="ValidationFinding"):
        ValidationReport("output", ("not a finding",))
    with pytest.raises(TypeError, match="ValidatorIdentity"):
        ValidationReport("output", validator={"name": "x"})


def test_report_keeps_positional_construction_and_equality():
    finding = hard_finding("size_mismatch", "image is 512x512")
    assert ValidationReport("a.png", (finding,)) == ValidationReport("a.png", [finding])
    assert ValidationReport("a.png", (finding,)).findings == (finding,)
    assert ValidationReport("a.png", (finding,)).validator is None


def test_replace_derives_an_unstated_outcome_from_the_new_findings():
    hard = hard_finding("blank_image", "image is blank")
    passing = ValidationReport("a.png")

    rejected = dataclasses.replace(passing, findings=(hard,))
    assert rejected.outcome is ValidationOutcome.REJECTED and not rejected.passed
    assert rejected == ValidationReport("a.png", (hard,))
    assert rejected.to_dict() == ValidationReport("a.png", (hard,)).to_dict()
    assert dataclasses.replace(rejected, findings=()).outcome is ValidationOutcome.PASSED
    assert dataclasses.replace(passing, path="b.png").outcome is ValidationOutcome.PASSED
    # Replacing the outcome itself states it, and a stated outcome is checked.
    error = dataclasses.replace(passing, outcome="error")
    assert error.outcome is ValidationOutcome.ERROR
    assert dataclasses.replace(error, findings=(hard,)).outcome is ValidationOutcome.ERROR
    with pytest.raises(ValueError, match="rejected report"):
        dataclasses.replace(passing, outcome="rejected")
    with pytest.raises(ValueError, match="passed report"):
        dataclasses.replace(passing, findings=(hard,), outcome="passed")
    assert dataclasses.replace(rejected, outcome=None, findings=()).passed


def test_replace_still_checks_a_stated_outcome():
    hard = hard_finding("too_short", "output is too short")
    with pytest.raises(ValueError, match="passed report"):
        dataclasses.replace(ValidationReport("output", outcome="passed"), findings=(hard,))
    with pytest.raises(ValueError, match="rejected report"):
        dataclasses.replace(ValidationReport("output", (hard,), outcome="rejected"), findings=())
    error = ValidationReport("output", outcome="error")
    assert dataclasses.replace(error, findings=(hard,)).outcome is ValidationOutcome.ERROR
    # A report read from JSON states the outcome its data records.
    restored = ValidationReport.from_dict(ValidationReport("output").to_dict())
    with pytest.raises(ValueError, match="passed report"):
        dataclasses.replace(restored, findings=(hard,))


def test_how_an_outcome_was_set_is_not_part_of_a_reports_value():
    derived, stated = ValidationReport("a.png"), ValidationReport("a.png", outcome="passed")
    assert derived == stated and hash(derived) == hash(stated)
    assert repr(derived) == repr(stated)
    assert ValidationReport.from_dict(derived.to_dict()) == derived


def test_report_round_trips_through_json():
    identity = ValidatorIdentity.for_config("dimensions", "2", {"width": 512, "height": 512})
    report = ValidationReport("hero.png", (
        hard_finding("dimensions_mismatch", "wrong size", expected=[512, 512], observed=[64, 64]),
        advisory_finding("palette_drift", "palette drifts", expected={"max": 0.2}, observed=0.25),
    ), validator=identity)

    data = json.loads(json.dumps(report.to_dict()))

    assert data == {
        "path": "hero.png",
        "outcome": "rejected",
        "findings": [
            {"code": "dimensions_mismatch", "message": "wrong size", "severity": "hard",
             "expected": [512, 512], "observed": [64, 64]},
            {"code": "palette_drift", "message": "palette drifts", "severity": "advisory",
             "expected": {"max": 0.2}, "observed": 0.25},
        ],
        "validator": identity.to_dict(),
    }
    assert ValidationReport.from_dict(data) == report
    error = ValidationReport("output", outcome="error")
    assert ValidationReport.from_dict(error.to_dict()) == error
    assert error.to_dict()["validator"] is None


def test_finding_tuples_project_as_json_lists():
    finding = hard_finding("dimensions_mismatch", "wrong size", expected=(512, 512))
    assert finding.to_dict()["expected"] == [512, 512]
    restored = ValidationFinding.from_dict(finding.to_dict())
    assert restored.expected == [512, 512]
    assert restored.severity is FindingSeverity.HARD


@pytest.mark.parametrize("value", [object(), float("nan"), {1: "x"}])
def test_finding_values_must_be_json_data_to_serialize(value):
    with pytest.raises((TypeError, ValueError)):
        hard_finding("odd", "odd value", observed=value).to_dict()


@pytest.mark.parametrize("change", [
    lambda data: data.update(extra=1),
    lambda data: data.pop("outcome"),
    lambda data: data.update(outcome="maybe"),
    lambda data: data.update(outcome="passed"),
    lambda data: data.update(findings="none"),
    lambda data: data.update(path=7),
    lambda data: data.update(validator={"name": "x"}),
    lambda data: data["findings"][0].update(severity="fatal"),
    lambda data: data["findings"][0].update(code=""),
    lambda data: data["findings"][0].update(surprise=True),
])
def test_report_from_dict_rejects_malformed_data(change):
    data = ValidationReport("output", (hard_finding("x", "missing x"),)).to_dict()
    change(data)
    with pytest.raises(ValueError):
        ValidationReport.from_dict(data)


@pytest.mark.parametrize("value", [None, [], "report"])
def test_report_from_dict_rejects_non_objects(value):
    with pytest.raises(ValueError):
        ValidationReport.from_dict(value)


def test_validator_identity_hashes_canonical_config():
    identity = ValidatorIdentity.for_config("sections", "1", {"b": 1, "a": [1, 2]})
    reordered = ValidatorIdentity.for_config("sections", "1", {"a": [1, 2], "b": 1})

    assert identity.config_sha256 == hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    assert identity == reordered
    assert ValidatorIdentity.for_config("n", "1").config_sha256 == hashlib.sha256(b"{}").hexdigest()
    assert ValidatorIdentity.for_config("n", "1", {"heading": "Résumé"}).config_sha256 == (
        hashlib.sha256('{"heading":"Résumé"}'.encode()).hexdigest()
    )
    assert identity.to_dict() == {
        "name": "sections", "version": "1", "config_sha256": identity.config_sha256,
    }
    assert ValidatorIdentity.from_dict(identity.to_dict()) == identity


@pytest.mark.parametrize("args", [
    ("", "1", "0" * 64), ("name", " ", "0" * 64), ("name", 1, "0" * 64),
    ("name", "1", "0" * 63), ("name", "1", "A" * 64), ("name", "1", None),
])
def test_validator_identity_rejects_malformed_fields(args):
    with pytest.raises(ValueError):
        ValidatorIdentity(*args)


def test_validator_identity_rejects_non_json_config_and_unknown_keys():
    with pytest.raises((TypeError, ValueError)):
        ValidatorIdentity.for_config("n", "1", {"limit": float("inf")})
    with pytest.raises((TypeError, ValueError)):
        ValidatorIdentity.for_config("n", "1", {"when": object()})
    with pytest.raises(ValueError, match="keys"):
        ValidatorIdentity.from_dict({"name": "n", "version": "1", "config_sha256": "0" * 64, "x": 1})
    with pytest.raises(ValueError, match="keys"):
        ValidatorIdentity.from_dict({"name": "n", "version": "1"})
