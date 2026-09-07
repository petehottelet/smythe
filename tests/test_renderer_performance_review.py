"""The independent review rejects source/environment changes without a browser."""

import hashlib
import json

import pytest

from benchmarks import verify_renderer_performance_20260907 as verifier


def test_retained_v2_review_is_reproduced_exactly():
    result = (json.dumps(verifier.verify_campaign(), indent=2) + "\n").encode()
    retained = verifier.ROOT / "benchmarks/results/glyph_rain_regl_20260907_f1_review_v2.json"
    assert result == retained.read_bytes()
    assert len(verifier.verify_campaign()["sourceVerification"]["files"]) == 50


def test_actual_source_bytes_are_required(monkeypatch):
    original = verifier._sha

    def changed_source(path):
        if path.name == "rain.js":
            return hashlib.sha256(path.read_bytes() + b"\n").hexdigest()
        return original(path)

    monkeypatch.setattr(verifier, "_sha", changed_source)
    with pytest.raises(ValueError, match="Frozen source bytes mismatch"):
        verifier.verify_campaign()


@pytest.mark.parametrize("damage", ["renderer", "auxiliary", "device", "viewport", "overlap", "boolean_rep"])
def test_control_environment_and_chronology_are_independently_checked(monkeypatch, damage):
    original = verifier._load

    def changed_control(path):
        record = original(path)
        if path.name != "glyph_rain_regl_20260907_cadence_r2.json":
            return record
        if damage == "renderer":
            record["browser"]["gl"]["unmaskedRenderer"] = "SwiftShader"
        elif damage == "auxiliary":
            record["browser"]["systemInfo"]["gpu"]["auxAttributes"]["glRenderer"] = "ANGLE (Intel)"
        elif damage == "device":
            record["browser"]["systemInfo"]["gpu"]["devices"] = []
        elif damage == "viewport":
            record["startup"]["width"] = 1280
        elif damage == "overlap":
            record["attemptStartedAt"] = "2000-01-01T00:00:00.000Z"
        else:
            record["repetition"] = True
        return record

    monkeypatch.setattr(verifier, "_load", changed_control)
    with pytest.raises(ValueError, match="GL|physical device|Auxiliary|viewport|chronology|order"):
        verifier.verify_campaign()
