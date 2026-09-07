"""Small real-process regression coverage for the offline Jobs scale campaign."""

from __future__ import annotations

import json
import hashlib

import pytest
import benchmarks.jobs_scale_benchmark as campaign_module

pytest.importorskip("PIL.Image")

from benchmarks.jobs_scale_benchmark import (
    _provider_summary, _source_hashes, _validate_call_ledger, _validate_zero_cost,
    _write_exclusive_json, main, run_campaign,
)


def test_real_hard_kill_resume_and_explicit_reroll_preserve_accepted_operations(tmp_path):
    root = tmp_path / "campaign"
    result = run_campaign(root, count=12, concurrency=2, lease_ttl_s=30, timeout_s=180)
    verified = result["verification"]
    assert verified["passed"]
    assert verified["operations"] == verified["accepted_artifacts"] == 12
    assert verified["accepted_before_kill"] == 4
    assert verified["pending_completed_on_resume"] == 6
    assert verified["unknown_preserved_on_resume"] == verified["explicit_rerolls"] == 2
    assert verified["provider_entries"] == 14
    assert verified["previously_succeeded_redispatches"] == 0
    assert verified["unknown_auto_redispatches"] == 0
    assert verified["completed_resume_provider_entries"] == 0
    assert len(verified["reroll_lineage"]) == 2
    assert all(item["parent_attempt_id"] != item["accepted_attempt_id"]
               for item in verified["reroll_lineage"])
    assert verified["unique_artifact_content_hashes"] == 1
    assert verified["artifact_bytes"] == 12 * verified["fixture_bytes"]
    assert result["lease_expiry"]["resume_allowed_at_ns"] > result["lease_expiry"]["expires_at_ns"]
    assert result["phases"]["resume"]["provider"]["entries"] == 6
    assert result["phases"]["reroll"]["provider"]["entries"] == 2
    assert result["phases"]["finished_resume"]["execution_metrics"]["operations_started"] == 0
    assert result["phases"]["start"]["execution_metrics"] is None
    assert result["phases"]["start"]["provider"]["peak_active_provider_calls"] == 2
    assert result["comparative_claimable"] is False
    assert result["sqlite_disk_bytes"]["jobs.db"] > 0
    assert "smythe/jobs/runner.py" in result["provenance"]["source_sha256"]
    assert json.loads((root / "config.json").read_text())["count"] == 12
    with pytest.raises(FileExistsError):
        run_campaign(root, count=12, concurrency=2)


def test_provider_entry_evidence_rejects_duplicate_calls_and_unmatched_returns():
    entry = {"event": "entered", "call_id": "c1", "attempt_id": "a1", "operation_id": "op1"}
    with pytest.raises(RuntimeError, match="duplicate provider entry"):
        _provider_summary([entry, entry])
    with pytest.raises(RuntimeError, match="without an active entry"):
        _provider_summary([entry | {"event": "returned"}])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _provider_summary([entry, entry | {"event": "returned", "attempt_id": "other"}])


@pytest.mark.parametrize("count,concurrency,ttl", [(3, 2, 2), (5001, 8, 2), (12, 0, 2), (12, 2, 0.1)])
def test_invalid_campaign_configuration_does_not_create_directory(tmp_path, count, concurrency, ttl):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_campaign(root, count=count, concurrency=concurrency, lease_ttl_s=ttl)
    assert not root.exists()


def test_source_hashes_only_include_tracked_product_sources_and_explicit_harness(tmp_path, monkeypatch):
    for relative in ("smythe/runner.py", "smythe/tmp/venv/ignored.py",
                     "benchmarks/jobs_scale_benchmark.py", "tests/test_jobs_scale_benchmark.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    monkeypatch.setattr(campaign_module, "REPOSITORY", tmp_path)
    monkeypatch.setattr(campaign_module.subprocess, "check_output", lambda *a, **k: b"smythe/runner.py\0")
    hashes = _source_hashes()
    assert set(hashes) == {"smythe/runner.py", "benchmarks/jobs_scale_benchmark.py",
                           "tests/test_jobs_scale_benchmark.py"}
    assert hashes["smythe/runner.py"] == hashlib.sha256(b"pass\n").hexdigest()
    (tmp_path / "smythe/runner.py").unlink()
    with pytest.raises(FileNotFoundError):
        _source_hashes()


@pytest.mark.parametrize("error", [RuntimeError("injected worker failure"), KeyboardInterrupt()])
def test_failed_campaign_retains_explicit_diagnostic_record(tmp_path, monkeypatch, error):
    def fail(root, **kwargs):
        root.mkdir()
        (root / "provenance.json").write_text('{"git_revision": "test"}')
        (root / "start-worker.log").write_text("simulated worker failure")
        raise error

    monkeypatch.setattr(campaign_module, "_run_campaign", fail)
    root = tmp_path / "failure"
    with pytest.raises(type(error)):
        run_campaign(root)
    record = json.loads((root / "failure.json").read_text())
    assert record["campaign_status"] == "failed"
    assert record["evidence_status"] == "diagnostic_incomplete"
    assert record["configuration"]["lease_ttl_s"] == 30
    assert record["failure"]["exception_type"] == type(error).__name__
    assert record["provenance"]["git_revision"] == "test"
    assert "start-worker.log" in record["retained_evidence"]


def test_output_is_exclusive_and_cannot_collide_with_internal_evidence(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        _write_exclusive_json(output, {"replacement": True})
    assert output.read_text() == "existing evidence"
    root = tmp_path / "new-campaign"
    monkeypatch.setattr(campaign_module.sys, "argv", ["campaign", "--workdir", str(root),
                                                     "--output", str(root / "config.json")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["status", "attempt_id", "call_id", "cost", "certainty"])
def test_final_call_ledger_rejects_corrupted_recovery_evidence(mutation):
    success = {"call_id": "accepted-call", "attempt_id": "accepted-attempt", "status": "succeeded",
               "ceiling_microusd": 0, "confirmed_microusd": 0, "exposure_microusd": 0,
               "cost_is_complete": 1, "cost_is_estimate": 0}
    unknown = success | {"call_id": "unknown-call", "attempt_id": "unknown-attempt",
                         "status": "unknown_outcome", "cost_is_complete": 0, "cost_is_estimate": 1}
    accepted = {"operation": {"attempt_id": "accepted-attempt"}}
    _validate_call_ledger([success, unknown], accepted, {"unknown-call"})
    if mutation == "status":
        unknown["status"] = "dispatched"
    elif mutation == "attempt_id":
        success["attempt_id"] = "other-attempt"
    elif mutation == "call_id":
        unknown["call_id"] = "other-call"
    elif mutation == "cost":
        unknown["exposure_microusd"] = 1
    else:
        unknown["cost_is_complete"] = 1
    with pytest.raises(RuntimeError):
        _validate_call_ledger([success, unknown], accepted, {"unknown-call"})


def test_final_zero_cost_ledger_rejects_drift():
    ledger = dict.fromkeys(("approved_microusd", "confirmed_microusd",
                           "exposure_microusd", "reserved_microusd"), 0)
    _validate_zero_cost({"cost": ledger})
    with pytest.raises(RuntimeError, match="ledger drifted"):
        _validate_zero_cost({"cost": ledger | {"confirmed_microusd": 1}})
