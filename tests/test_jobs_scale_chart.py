"""Synthetic fixtures test evidence gates; this file contains no campaign result."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest

from benchmarks.jobs_scale_chart import REVIEW_CHECKS, render_jobs_scale


@pytest.fixture
def evidence():
    counts = {"after_kill": {"succeeded": 2492, "running": 8, "pending": 2500},
              "after_resume": {"succeeded": 4992, "unknown_outcome": 8}, "final": {"succeeded": 5000}}
    cost = {key: 0 for key in ("approved_microusd", "confirmed_microusd", "exposure_microusd", "reserved_microusd")}
    return {
        "schema": "smythe.jobs-scale-recovery.v1", "campaign_status": "completed",
        "evidence_status": "offline_correctness_observation", "comparative_claimable": False,
        "configuration": {"count": 5000, "concurrency": 8, "max_attempts": 2,
                          "kill_after_entries": 2500, "completed_before_kill": 2492},
        "state_counts": counts, "ledger": {stage: dict(cost) for stage in counts},
        "provenance": {"sources_unchanged_during_campaign": True},
        "verification": {
            "passed": True, "operations": 5000, "accepted_artifacts": 5000, "accepted_before_kill": 2492,
            "pending_completed_on_resume": 2500, "unknown_preserved_on_resume": 8, "explicit_rerolls": 8,
            "provider_entries": 5008, "remote_api_calls": 0, "previously_succeeded_redispatches": 0,
            "unknown_auto_redispatches": 0, "completed_resume_provider_entries": 0,
            "unique_artifact_content_hashes": 1, "zero_cost_ledger_verified": True,
            "unknown_call_cost_flags_preserved": True, "sqlite_integrity": "ok",
            "artifact_dimensions": [1, 1], "artifact_mime_type": "image/png", "fixture_bytes": 68,
            "fixture_sha256": "a" * 64,
            "artifact_bytes": 340000, "durable_call_status_counts": {"succeeded": 5000, "unknown_outcome": 8},
            "reroll_lineage": [{"operation_id": f"op-{i}", "parent_attempt_id": f"old-{i}",
                                "accepted_attempt_id": f"new-{i}"} for i in range(8)],
        },
    }


def paths(tmp_path, evidence):
    record = tmp_path / "synthetic-unit-fixture.json"
    record.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    review = {"schema": "smythe.jobs-scale-review.v1", "status": "passed", "observation_claimable": True,
              "known_measurement_defects": [], "scope": "offline_correctness_observation",
              "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
              "checks": dict.fromkeys(REVIEW_CHECKS, True)}
    review_path = tmp_path / "synthetic-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8", newline="\n")
    return record, review_path


def test_synthetic_chart_reconciles_three_stages_without_comparative_claim(evidence, tmp_path):
    record, review = paths(tmp_path, evidence)
    document = render_jobs_scale(record, review)
    assert document == render_jobs_scale(record, review)
    root = ET.fromstring(document)
    text = " ".join(root.itertext())
    for expected in ("5,000", "8 explicit rerolls", "0 previously accepted", "0 calls on completed resume",
                     "2,500 pending; 8 interrupted calls still dispatched", "8 unknown outcomes retained",
                     "1 campaign; identical 1×1 PNG fixtures", "no comparative speed"):
        assert expected in text
    assert hashlib.sha256(record.read_bytes()).hexdigest() in text
    assert hashlib.sha256(review.read_bytes()).hexdigest() in document
    assert set(re.findall(r"#[0-9a-fA-F]{6}", document)) == {"#000000", "#ffffff"}
    groups = root.findall("{http://www.w3.org/2000/svg}g")
    assert [int(group.attrib["data-accepted"]) for group in groups] == [2492, 4992, 5000]
    for group in groups:
        outline, solid = group
        assert outline.attrib["x"] == solid.attrib["x"] == "40"
        assert float(outline.attrib["width"]) == 880
        assert float(solid.attrib["width"]) == round(880 * int(group.attrib["data-accepted"]) / 5000, 1)


@pytest.mark.parametrize("section,key,value", [
    (None, "campaign_status", "failed"), (None, "comparative_claimable", True),
    ("configuration", "count", 25), ("configuration", "concurrency", 16),
    ("verification", "passed", False), ("verification", "accepted_artifacts", 4999),
    ("verification", "previously_succeeded_redispatches", 1), ("verification", "unknown_auto_redispatches", 1),
    ("verification", "completed_resume_provider_entries", 1), ("verification", "remote_api_calls", 1),
    ("verification", "remote_api_calls", False), ("verification", "artifact_bytes", 1),
    ("verification", "reroll_lineage", []), ("verification", "sqlite_integrity", "corrupt"),
    ("verification", "artifact_dimensions", [True, 1]),
    ("verification", "fixture_sha256", "invalid"),
    ("verification", "durable_call_status_counts", {"succeeded": 5000.0, "unknown_outcome": 8}),
    ("state_counts", "after_kill", {"succeeded": 2492, "unknown_outcome": 8, "pending": 2500}),
    ("state_counts", "after_resume", {"succeeded": 4991, "unknown_outcome": 8}),
    ("state_counts", "final", {"succeeded": 5000, "failed": False}),
    ("provenance", "sources_unchanged_during_campaign", False),
])
def test_rejects_failed_incomplete_or_mismatched_synthetic_data(evidence, tmp_path, section, key, value):
    target = evidence if section is None else evidence[section]
    target[key] = value
    with pytest.raises(ValueError, match="Jobs scale evidence"):
        render_jobs_scale(*paths(tmp_path, evidence))


@pytest.mark.parametrize("damage", ["missing_check", "false_check", "wrong_hash", "not_approved", "nonzero_cost",
                                    "duplicate_lineage", "same_parent", "missing_stage", "raw_line_endings",
                                    "missing_defects", "known_defect", "comparative_scope"])
def test_requires_exact_raw_hash_review_and_durable_findings(evidence, tmp_path, damage):
    if damage == "nonzero_cost":
        evidence["ledger"]["final"]["exposure_microusd"] = 1
    elif damage == "duplicate_lineage":
        evidence["verification"]["reroll_lineage"][1] = evidence["verification"]["reroll_lineage"][0]
    elif damage == "same_parent":
        evidence["verification"]["reroll_lineage"][0]["accepted_attempt_id"] = "old-0"
    elif damage == "missing_stage":
        del evidence["state_counts"]["after_resume"]
    record, review_path = paths(tmp_path, evidence)
    review = json.loads(review_path.read_bytes())
    if damage == "missing_check":
        del review["checks"]["real_kill_verified"]
    elif damage == "false_check":
        review["checks"]["archive_verified"] = False
    elif damage == "wrong_hash":
        review["record_sha256"] = "0" * 64
    elif damage == "not_approved":
        review["observation_claimable"] = False
    elif damage == "missing_defects":
        del review["known_measurement_defects"]
    elif damage == "known_defect":
        review["known_measurement_defects"] = ["artifact inventory incomplete"]
    elif damage == "comparative_scope":
        review["scope"] = "comparative_performance"
    elif damage == "raw_line_endings":
        record.write_bytes(record.read_bytes().replace(b"\n", b"\r\n"))
    review_path.write_text(json.dumps(review), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="Jobs scale evidence"):
        render_jobs_scale(record, review_path)
def test_validation_import_does_not_load_current_runtime_or_drawing_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", (
            "from benchmarks.jobs_scale_chart import validate_jobs_scale; import sys; "
            "assert not any(n == 'smythe' or n.startswith('smythe.') for n in sys.modules); "
            "assert 'benchmarks.render_readme_charts' not in sys.modules; "
            "print('pure-validation-import')"
        )],
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert result.stdout.strip() == "pure-validation-import"
