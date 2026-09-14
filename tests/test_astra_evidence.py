"""Offline tests for the independent native judge evidence audit."""

import json

import pytest

from benchmarks import astra_evidence as evidence
from benchmarks import astra_evaluation as evaluation
from test_astra_evaluation import case, transport, run  # noqa: F401


def test_audit_reconciles_cost_and_all_native_receipts(case, transport, tmp_path):  # noqa: F811
    saved = run(case, tmp_path)
    audited = evidence.inspect_judgments(tmp_path)
    assert audited["paid_judgments"] == 1 and audited["unresolved_attempts"] == 0
    assert audited["cost_upper_nanousd"] == saved["accounting"]["cost_upper_nanousd"]
    assert audited["records"][saved["identity"]]["result"] == saved
    assert transport["calls"] == ["countTokens", "generateContent"]


@pytest.mark.parametrize("part", ["quote", "raw", "result", "started", "orphan", "model", "identity", "charge"])
def test_audit_refuses_incomplete_tampered_or_mismatched_evidence(case, transport, tmp_path, part):  # noqa: F811
    saved = run(case, tmp_path)
    if part in {"quote", "raw", "result", "started"}:
        path = next(tmp_path.glob(f"*.{part}.json"))
        path.write_text('{}', encoding="utf-8")
    elif part == "orphan":
        (tmp_path / "unbound.raw.json").write_text('{}', encoding="utf-8")
    elif part == "model":
        path = next(tmp_path.glob("*.raw.json"))
        raw = json.loads(path.read_bytes())
        raw["modelVersion"] = "different"
        path.write_text(json.dumps(raw), encoding="utf-8")
    else:
        path = next(tmp_path.glob("*.result.json"))
        saved.pop("record_sha256")
        if part == "identity":
            saved["identity"] = "different"
        else:
            saved["accounting"]["cost_upper_nanousd"] = 0
        saved["record_sha256"] = evaluation._hash(saved)
        path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        evidence.inspect_judgments(tmp_path)


def test_unresolved_attempt_is_not_treated_as_zero_cost(case, transport, tmp_path):  # noqa: F811
    transport["interrupt"] = True
    with pytest.raises(OSError):
        run(case, tmp_path)
    with pytest.raises(ValueError, match="unresolved"):
        evidence.inspect_judgments(tmp_path)
