"""Render a reviewed Jobs scale observation; no default or unpublished evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.render_readme_charts import BLACK, MONO, SERIF, TRAJAN, _bar, _svg, _text


REVIEW_CHECKS = (
    "archive_verified", "ledger_reconciled", "accepted_pointers_preserved",
    "artifacts_verified", "attempt_lineage_verified", "real_kill_verified",
    "lease_expiry_verified", "zero_cost_verified",
)
STAGES = ("after_kill", "after_resume", "final")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Jobs scale evidence: {message}")


def _integer(value: object, name: str) -> int:
    _require(type(value) is int and value >= 0, f"invalid {name}")
    return value


def validate_jobs_scale(record: dict, review: dict, *, record_sha256: str) -> dict:
    """Reconcile chart data and require a review bound to the exact raw bytes.

    The independent review is produced after archive/SQLite inspection. This
    validator checks its required findings and the plotted arithmetic; it does
    not substitute for inspecting the retained campaign evidence.
    """
    _require(isinstance(record, dict) and isinstance(review, dict), "objects required")
    _require(record.get("schema") == "smythe.jobs-scale-recovery.v1", "wrong record schema")
    _require(record.get("campaign_status") == "completed", "campaign incomplete")
    _require(record.get("evidence_status") == "offline_correctness_observation"
             and record.get("comparative_claimable") is False, "wrong observation scope")
    _require(not record.get("failure") and not record.get("known_measurement_defects"), "known defects")
    _require(review.get("schema") == "smythe.jobs-scale-review.v1"
             and review.get("status") == "passed"
             and review.get("observation_claimable") is True, "review not approved")
    _require(review.get("known_measurement_defects") == []
             and review.get("scope") == "offline_correctness_observation", "review scope/defects missing")
    _require(len(record_sha256) == 64 and all(c in "0123456789abcdef" for c in record_sha256)
             and review.get("record_sha256") == record_sha256, "review/raw hash mismatch")
    checks = review.get("checks")
    _require(isinstance(checks, dict) and all(checks.get(key) is True for key in REVIEW_CHECKS),
             "independent checks incomplete")
    config, verification = record.get("configuration"), record.get("verification")
    _require(isinstance(config, dict) and isinstance(verification, dict), "missing configuration/verification")
    for key, expected in {"count": 5000, "concurrency": 8, "max_attempts": 2,
                          "kill_after_entries": 2500, "completed_before_kill": 2492}.items():
        _require(_integer(config.get(key), key) == expected, f"wrong {key} workload")
    _require(verification.get("passed") is True, "verification failed")
    expected_values = {
        "operations": 5000, "accepted_artifacts": 5000, "accepted_before_kill": 2492,
        "pending_completed_on_resume": 2500, "unknown_preserved_on_resume": 8,
        "explicit_rerolls": 8, "provider_entries": 5008, "remote_api_calls": 0,
        "previously_succeeded_redispatches": 0, "unknown_auto_redispatches": 0,
        "completed_resume_provider_entries": 0, "unique_artifact_content_hashes": 1,
    }
    for key, expected in expected_values.items():
        _require(_integer(verification.get(key), key) == expected, f"inconsistent {key}")
    _require(verification.get("zero_cost_ledger_verified") is True
             and verification.get("unknown_call_cost_flags_preserved") is True
             and verification.get("sqlite_integrity") == "ok", "ledger checks incomplete")
    _require(verification.get("artifact_dimensions") == [1, 1]
             and all(type(v) is int for v in verification["artifact_dimensions"])
             and verification.get("artifact_mime_type") == "image/png", "wrong fixture")
    fixture_bytes = _integer(verification.get("fixture_bytes"), "fixture_bytes")
    fixture_hash = verification.get("fixture_sha256")
    _require(isinstance(fixture_hash, str) and len(fixture_hash) == 64
             and all(c in "0123456789abcdef" for c in fixture_hash), "invalid fixture hash")
    _require(fixture_bytes > 0 and _integer(verification.get("artifact_bytes"), "artifact_bytes")
             == fixture_bytes * 5000, "artifact byte totals disagree")
    _require(verification.get("durable_call_status_counts") == {"succeeded": 5000, "unknown_outcome": 8},
             "call-state totals disagree")
    for key, value in verification["durable_call_status_counts"].items():
        _integer(value, f"durable_call_status_counts.{key}")
    lineage = verification.get("reroll_lineage")
    _require(isinstance(lineage, list) and len(lineage) == 8, "reroll lineage incomplete")
    for key in ("operation_id", "parent_attempt_id", "accepted_attempt_id"):
        values = [row.get(key) if isinstance(row, dict) else None for row in lineage]
        _require(all(isinstance(value, str) and value for value in values)
                 and len(set(values)) == 8, f"invalid lineage {key}")
    _require(not {row["parent_attempt_id"] for row in lineage}
             & {row["accepted_attempt_id"] for row in lineage}, "reroll reused parent attempt")
    state_counts, ledger = record.get("state_counts"), record.get("ledger")
    _require(isinstance(state_counts, dict) and isinstance(ledger, dict), "missing stages")
    expected_counts = (
        {"succeeded": 2492, "running": 8, "pending": 2500},
        {"succeeded": 4992, "unknown_outcome": 8},
        {"succeeded": 5000},
    )
    for stage, expected in zip(STAGES, expected_counts):
        counts, cost = state_counts.get(stage), ledger.get(stage)
        _require(isinstance(counts, dict) and isinstance(cost, dict), f"missing {stage}")
        _require({key: _integer(value, f"{stage}.{key}") for key, value in counts.items() if value != 0}
                 == expected, f"{stage} counts disagree")
        # Validate zero-valued fields too: False/0.0 must not pass integer checks.
        for key, value in counts.items():
            _integer(value, f"{stage}.{key}")
        for key in ("approved_microusd", "confirmed_microusd", "exposure_microusd", "reserved_microusd"):
            _require(_integer(cost.get(key), f"{stage}.{key}") == 0, "nonzero fixture ledger")
    _require(record.get("provenance", {}).get("sources_unchanged_during_campaign") is True,
             "source changed during campaign")
    return {"counts": state_counts, "verification": verification, "operations": config["count"]}


def render_jobs_scale(record_path: str | Path, review_path: str | Path) -> str:
    """Render only explicitly supplied, reviewed actual campaign records.

    Raw evidence hashes are byte-exact. Git must preserve these archived JSON
    bytes; translating line endings requires a new matching independent review.
    """
    record_path, review_path = Path(record_path), Path(review_path)
    raw, review_raw = record_path.read_bytes(), review_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = validate_jobs_scale(json.loads(raw), json.loads(review_raw), record_sha256=digest)
    verification, count = data["verification"], data["operations"]
    body = _text(40, 40, "DURABLE EXECUTION", size=11, weight="700", tracking=2.2)
    body += _text(40, 80, "Jobs after a hard process kill", size=31, family=SERIF, weight="700")
    body += _text(40, 107, "One local campaign; concurrency 8; durable calls and accepted artifact pointers", size=13)
    body += f'<line x1="40" y1="128" x2="920" y2="128" stroke="{BLACK}" stroke-width="2"/>\n'
    body += _text(40, 204, f'{verification["accepted_artifacts"]:,}', size=64, family=TRAJAN, weight="700")
    body += _text(280, 164, "ACCEPTED ARTIFACTS", size=14, weight="700", tracking=.8)
    body += _text(280, 189, f'{verification["explicit_rerolls"]} explicit rerolls of interrupted operations', size=14)
    body += _text(280, 214, "0 previously accepted operations reissued; 0 calls on completed resume", size=13)
    for stage, label, y in zip(STAGES, ("After the kill", "After safe resume", "After explicit rerolls"), (264, 354, 444)):
        counts = data["counts"][stage]
        accepted = counts.get("succeeded", 0)
        body += _text(40, y, label, size=18, family=SERIF, weight="700")
        body += _text(920, y, f"{accepted:,} accepted / {count:,}", size=14, anchor="end")
        body += f'<g data-stage="{stage}" data-accepted="{accepted}" data-maximum="{count}">\n'
        body += _bar(40, y + 12, 880, "outline")
        body += _bar(40, y + 12, 880 * accepted / count, "solid") + '</g>\n'
        details = (f'{counts.get("pending", 0):,} pending; {counts.get("running", 0)} interrupted calls still dispatched'
                   if stage == "after_kill" else f'{counts.get("unknown_outcome", 0)} unknown outcomes retained'
                   if stage == "after_resume" else "Every accepted pointer retained; rerolls keep parent-attempt lineage")
        body += _text(40, y + 48, details, size=12)
    body += f'<line x1="40" y1="518" x2="920" y2="518" stroke="{BLACK}"/>\n'
    body += _text(40, 544, "1 campaign; identical 1×1 PNG fixtures; zero API calls and zero API cost", size=12)
    body += _text(40, 566, "Correctness observation; no comparative speed or model-quality claim", size=12)
    body += _text(40, 592, record_path.name, size=10, family=MONO)
    body += _text(40, 612, f"RAW SHA-256 {digest}", size=10, family=MONO)
    body += f'<!-- review-sha256: {hashlib.sha256(review_raw).hexdigest()} -->\n'
    return _svg(960, 636, body, label=(
        "One offline Jobs campaign: 5,000 accepted identical 1×1 PNG fixtures after a hard kill, "
        "safe resume and eight explicit rerolls. No previously accepted work reissued; zero calls "
        "on completed resume. No comparative speed or model-quality claim."))
