"""Installed CLI tests for the durable Jobs v1 workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from smythe.cli import (
    EXIT_APPROVAL,
    EXIT_INVALID_INPUT,
    EXIT_JOB_FAILED,
    EXIT_OK,
    EXIT_PREFLIGHT,
    main,
)


def _manifest(tmp_path, *, mime_type: str = "image/png", attempts: int = 1):
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "cli-test",
                "profiles": [
                    {
                        "name": "default",
                        "provider": "offline",
                        "model": "offline-image",
                        "max_cost_per_call_usd": "0",
                        "options": {"artifacts_per_call": 1},
                    }
                ],
                "operations": [
                    {
                        "key": "glyph",
                        "prompt": "Generate one glyph",
                        "profile": "default",
                        "artifact": {
                            "mime_type": mime_type,
                            "width": 1,
                            "height": 1,
                        },
                    }
                ],
                "execution": {
                    "max_concurrency": 2,
                    "max_attempts": attempts,
                    "max_budget_usd": "0",
                    "output_directory": "outputs",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _json_output(capsys):
    return json.loads(capsys.readouterr().out)


def test_schema_and_validate_emit_machine_readable_json(tmp_path, capsys):
    manifest = _manifest(tmp_path)

    assert main(["jobs", "--json", "schema"]) == EXIT_OK
    schema = _json_output(capsys)
    assert schema["ok"] is True
    assert schema["schema"]["properties"]["version"] == {"const": 1}

    assert main(["jobs", "validate", str(manifest), "--json"]) == EXIT_OK
    validated = _json_output(capsys)
    assert validated["validate"]["name"] == "cli-test"
    assert validated["validate"]["operation_count"] == 1
    assert validated["validate"]["worst_case_cost_usd"] == "0.000000"


def test_committed_offline_manifest_is_a_valid_zero_cost_plan(capsys):
    manifest = Path(__file__).parents[1] / "examples" / "12_jobs_manifest.yaml"

    assert main(["jobs", "validate", str(manifest), "--json"]) == EXIT_OK
    validated = _json_output(capsys)["validate"]
    assert validated["name"] == "offline-glyph-job"
    assert validated["operation_count"] == 4
    assert validated["worst_case_cost_usd"] == "0.000000"


def test_plan_token_runs_job_then_status_resume_and_export(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    store = tmp_path / "jobs.sqlite3"

    assert main(["jobs", "--json", "plan", str(manifest)]) == EXIT_OK
    planned = _json_output(capsys)["plan"]
    token = planned["approval"]["token"]
    assert token.startswith("approve_v1_")

    assert main(
        [
            "jobs",
            "run",
            str(manifest),
            "--approve",
            token,
            "--store",
            str(store),
            "--json",
        ]
    ) == EXIT_OK
    run = _json_output(capsys)["run"]
    run_id = run["run_id"]
    assert run["status"] == "completed"
    assert run["counts"] == {"succeeded": 1}
    assert len(run["artifacts"]) == 1

    assert main(
        ["jobs", "status", run_id, "--events", "--store", str(store), "--json"]
    ) == EXIT_OK
    status = _json_output(capsys)["status"]
    assert status["status"] == "completed"
    assert any(event["event_type"] == "run_created" for event in status["events"])

    assert main(
        ["jobs", "resume", run_id, "--store", str(store), "--json"]
    ) == EXIT_OK
    resumed = _json_output(capsys)["resume"]
    assert resumed["status"] == "completed"
    assert resumed["execution_metrics"]["operations_started"] == 0

    destination = tmp_path / "exports" / "job.json"
    assert main(
        [
            "jobs",
            "export",
            run_id,
            "--out",
            str(destination),
            "--store",
            str(store),
            "--json",
        ]
    ) == EXIT_OK
    acknowledgement = _json_output(capsys)["export"]
    assert acknowledgement["path"] == str(destination.resolve())
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["manifest_root"] == "."
    assert exported["paths_relative_to"] == "artifact_root"
    assert exported["artifact_root"] == f"outputs/{run_id}"
    assert exported["events"]
    assert all(not artifact["relative_path"].startswith("/") for artifact in exported["artifacts"])
    for artifact in exported["artifacts"]:
        artifact_path = tmp_path / exported["artifact_root"] / artifact["relative_path"]
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]


def test_run_refuses_a_token_for_a_different_approval(tmp_path, capsys):
    manifest = _manifest(tmp_path)

    code = main(
        [
            "jobs",
            "run",
            str(manifest),
            "--approve",
            "approve_v1_wrong",
            "--store",
            str(tmp_path / "jobs.sqlite3"),
            "--json",
        ]
    )

    assert code == EXIT_APPROVAL
    error = _json_output(capsys)
    assert error["ok"] is False
    assert error["error"]["type"] == "ApprovalError"


def test_invalid_manifest_has_stable_nonzero_exit(tmp_path, capsys):
    manifest = tmp_path / "invalid.json"
    manifest.write_text('{"version": 99}', encoding="utf-8")

    assert main(["jobs", "validate", str(manifest), "--json"]) == EXIT_INVALID_INPUT
    error = _json_output(capsys)
    assert error["ok"] is False
    assert error["command"] == "validate"


def test_provider_preflight_has_distinct_exit_code(tmp_path, capsys, monkeypatch):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["profiles"][0] = {
        "name": "default",
        "provider": "openai_image",
        "model": "gpt-image-1",
        "max_cost_per_call_usd": "0.01",
    }
    payload["execution"]["max_budget_usd"] = "0.01"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main(["jobs", "validate", str(manifest), "--json"]) == EXIT_PREFLIGHT
    error = _json_output(capsys)
    assert error["ok"] is False
    assert error["error"]["type"] == "ProviderPreflightError"


def test_rejected_job_and_reroll_return_terminal_error_exit(tmp_path, capsys):
    manifest = _manifest(tmp_path, mime_type="image/jpeg", attempts=2)
    store = tmp_path / "jobs.sqlite3"

    assert main(["jobs", "plan", str(manifest), "--json"]) == EXIT_OK
    token = _json_output(capsys)["plan"]["approval"]["token"]
    assert main(
        [
            "jobs",
            "run",
            str(manifest),
            "--approve",
            token,
            "--store",
            str(store),
            "--json",
        ]
    ) == EXIT_JOB_FAILED
    failed = _json_output(capsys)["run"]
    operation_key = failed["operations"][0]["operation_key"]
    assert failed["counts"] == {"rejected": 1}

    assert main(
        [
            "jobs",
            "reroll",
            failed["run_id"],
            operation_key,
            "--reason",
            "wrong MIME",
            "--store",
            str(store),
            "--json",
        ]
    ) == EXIT_JOB_FAILED
    rerolled = _json_output(capsys)["reroll"]
    assert rerolled["counts"] == {"rejected": 1}
    assert rerolled["operations"][0]["attempt_count"] == 2
