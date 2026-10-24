"""Operator inspection preserves ledger state and bounds artifact reads."""

from __future__ import annotations

from contextlib import closing
import copy
import hashlib
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from smythe.jobs import JobManifestV1, make_approval, operation_input_digest, preflight_job
from smythe.jobs.inspection import inspect_job, list_jobs
from smythe.jobs.store import RunStoreError, SQLiteRunStore


@pytest.fixture
def recorded(tmp_path):
    manifest = JobManifestV1.from_dict({
        "version": 1,
        "name": "inspection-fixture",
        "profiles": [{"name": "default", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [{"key": "glyph", "count": 2, "prompt": "Original prompt",
                        "profile": "default"}],
        "execution": {"max_concurrency": 2, "max_attempts": 1,
                      "max_budget_usd": "0", "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    database = tmp_path / "jobs.sqlite3"
    paths = []
    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path)
        directory = store.snapshot(run_id)["artifact_directory"]
        for index, op in enumerate(store.pending_operations(run_id)):
            attempt = store.begin_attempt(run_id, op["operation_id"])
            permit = store.prepare_call(attempt["attempt_id"], 0)
            store.mark_call_dispatched(permit.call_id)
            content = f"artifact-{index}".encode()
            relative = f"artifacts/{index}.bin"
            path = tmp_path / "outputs" / directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            paths.append(path)
            store.complete_call(
                permit.call_id, cost_microusd=0, cost_is_complete=True,
                cost_is_estimate=False, result_text=f"Response {index}", artifacts=[{
                    "relative_path": relative, "mime_type": "application/octet-stream",
                    "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                }],
            )
        store.finalize_run(run_id)
    return database, run_id, paths


def test_inspection_reads_recorded_prompt_response_cost_and_artifact(recorded):
    database, run_id, paths = recorded
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)
        assert list_jobs(store)["runs"][0]["run_id"] == run_id
    assert database.read_bytes() == before
    assert report["status"] == "completed"
    assert report["counts"] == {"succeeded": 2}
    assert report["cost"]["confirmed_microusd"] == 0
    assert report["operations"][0]["spec"]["prompt"] == "Original prompt"
    assert report["operations"][0]["result_text"] == "Response 0"
    assert len(report["calls"]) == 2
    assert report["artifact_integrity"]["counts"] == {"verified": 2}
    assert report["artifact_integrity"]["bytes_hashed"] == sum(p.stat().st_size for p in paths)
    assert report["inspection_version"] == 1
    assert report["inspected_at_ns"] > report["created_at_ns"]
    json.dumps(report)  # Structured CLI output must remain serializable.


def test_inspection_page_preserves_whole_run_counts(recorded):
    database, run_id, _ = recorded
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id, limit=1, offset=1, events_limit=2)
    assert report["counts"] == {"succeeded": 2}
    assert len(report["operations"]) == len(report["attempts"]) == len(report["artifacts"]) == 1
    assert report["pagination"] == {
        "limit": 1, "offset": 1, "total": 2, "returned": 1, "has_more": False,
    }
    assert len(report["events"]) == 2
    assert report["event_pagination"]["has_more"] is True


@pytest.mark.parametrize("change,expected", [("missing", "missing"), ("size", "changed"),
                                             ("hash", "changed")])
def test_current_disk_finding_does_not_rewrite_acceptance(recorded, change, expected):
    database, run_id, paths = recorded
    if change == "missing":
        paths[0].unlink()
    elif change == "size":
        paths[0].write_bytes(b"different length")
    else:
        paths[0].write_bytes(b"x" * paths[0].stat().st_size)
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)
    assert report["artifacts"][0]["integrity"]["status"] == expected
    assert report["artifacts"][0]["accepted"] == 1
    assert report["status"] == "completed"
    assert database.read_bytes() == before


def _completed_run(tmp_path, count=2):
    """A run made by this release, whose calls record their delivery decisions."""
    import asyncio

    pytest.importorskip("PIL.Image")
    from smythe.jobs.runner import JobRunner

    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "delivery-fixture",
        "profiles": [{"name": "default", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0", "options": {"artifacts_per_call": 1}}],
        "operations": [{"key": "glyph", "count": count, "prompt": "One glyph",
                        "profile": "default", "artifact": {"mime_type": "image/png"}}],
        "execution": {"max_concurrency": 2, "max_attempts": 1, "max_budget_usd": "0",
                      "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    database = tmp_path / "jobs.sqlite3"
    with SQLiteRunStore(database) as store:
        result = asyncio.run(JobRunner(store).start(plan, make_approval(plan),
                                                    manifest_root=tmp_path))
    root = tmp_path / "outputs" / result["artifact_directory"]
    paths = {artifact["attempt_id"]: root / artifact["relative_path"]
             for artifact in result["artifacts"]}
    return database, result["run_id"], paths


def test_recorded_decisions_make_intact_accepted_outputs_eligible(tmp_path):
    database, run_id, _ = _completed_run(tmp_path)
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    for operation in report["operations"]:
        acceptance = operation["delivery_acceptance"]
        assert acceptance["source"] == "recorded" and acceptance["decision"] == "accepted"
        assert acceptance["accepted"] is True
        assert acceptance["attempt_id"] == operation["accepted_attempt_id"]
        assert [item["outcome"] for item in acceptance["validators"]] == ["passed"]
        assert acceptance["validators"][0]["validator"]["name"] == "smythe.jobs.artifact-spec"
        assert operation["delivery_eligibility"] == {"eligible": True, "reasons": []}
    for attempt in report["attempts"]:
        record = attempt["delivery_acceptance"]
        assert record["attempt_id"] == attempt["attempt_id"] and record["decision"] == "accepted"
        assert record["validators"][0]["reports"][0]["outcome"] == "passed"
    json.dumps(report)


@pytest.mark.parametrize("change,reason", [("hash", "artifact_changed"),
                                           ("missing", "artifact_missing")])
def test_changed_accepted_bytes_are_ineligible_and_acceptance_is_unchanged(tmp_path, change, reason):
    database, run_id, paths = _completed_run(tmp_path)
    changed_attempt, path = next(iter(paths.items()))
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    assert database.read_bytes() == before
    assert report["status"] == "completed" and report["counts"] == {"succeeded": 2}
    assert all(artifact["accepted"] == 1 for artifact in report["artifacts"])
    by_attempt = {op["accepted_attempt_id"]: op for op in report["operations"]}
    assert by_attempt.pop(changed_attempt)["delivery_eligibility"] == {
        "eligible": False, "reasons": [reason]}
    [intact] = by_attempt.values()
    assert intact["delivery_eligibility"]["eligible"] is True
    assert all(op["delivery_acceptance"]["decision"] == "accepted" for op in report["operations"])


def test_legacy_decisions_are_built_in_with_validator_unrecorded(recorded):
    database, run_id, _ = recorded
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    for operation in report["operations"]:
        assert operation["delivery_acceptance"] == {
            "source": "legacy_built_in", "attempt_id": operation["accepted_attempt_id"],
            "decision": None, "accepted": True, "validators": None, "reasons": [],
        }
        assert operation["delivery_eligibility"] == {
            "eligible": False, "reasons": ["validator_unrecorded"]}
    assert all(attempt["delivery_acceptance"] is None for attempt in report["attempts"])
    assert "not_evaluated" not in json.dumps(report)


def test_a_newer_builtin_check_makes_recorded_decisions_not_current(tmp_path, monkeypatch):
    import smythe.jobs.delivery as delivery

    database, run_id, _ = _completed_run(tmp_path, count=1)
    monkeypatch.setattr(delivery, "ARTIFACT_SPEC_VALIDATOR_VERSION", "2")
    with SQLiteRunStore(database, read_only=True) as store:
        [operation] = inspect_job(store, run_id)["operations"]

    assert operation["delivery_acceptance"]["decision"] == "accepted"
    assert operation["delivery_eligibility"] == {"eligible": False, "reasons": ["validator_changed"]}


def test_rejected_and_undecided_operations_are_not_eligible(tmp_path):
    from smythe.jobs.delivery import artifact_spec_identity, validator_result
    from smythe.validation import ValidationReport, hard_finding

    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "undecided",
        "profiles": [{"name": "default", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [{"key": "glyph", "count": 3, "prompt": "One glyph", "profile": "default"}],
        "execution": {"max_concurrency": 1, "max_attempts": 1, "max_budget_usd": "0",
                      "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    identity = artifact_spec_identity(plan.operations[0].artifact)
    finding = hard_finding("no_artifacts", "provider returned no artifacts")
    database = tmp_path / "jobs.sqlite3"
    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path)
        rejected, malformed, pending = store.pending_operations(run_id)
        for operation, validators in ((rejected, None), (malformed, "not a list")):
            attempt = store.begin_attempt(run_id, operation["operation_id"])
            permit = store.prepare_call(attempt["attempt_id"], 0)
            store.mark_call_dispatched(permit.call_id)
            report = ValidationReport("response", (finding,), validator=identity)
            record = {"version": 1, "scope": "artifacts", "attempt_id": attempt["attempt_id"],
                      "decision": "rejected", "reasons": [finding.message], "subjects": [],
                      "validators": validators or [validator_result(identity, [report])]}
            store.complete_call(permit.call_id, cost_microusd=0, cost_is_complete=True,
                                cost_is_estimate=False, artifacts=[], result_text="none",
                                accepted=False, error=finding.message,
                                delivery_acceptance=record)
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    by_id = {op["operation_id"]: op for op in report["operations"]}
    assert by_id[rejected["operation_id"]]["delivery_acceptance"]["decision"] == "rejected"
    assert by_id[rejected["operation_id"]]["delivery_eligibility"] == {
        "eligible": False, "reasons": ["not_accepted"]}
    assert by_id[malformed["operation_id"]]["delivery_acceptance"]["decision"] is None
    assert by_id[malformed["operation_id"]]["delivery_eligibility"] == {
        "eligible": False, "reasons": ["not_accepted", "record_invalid"]}
    assert by_id[pending["operation_id"]]["delivery_acceptance"] is None
    assert by_id[pending["operation_id"]]["delivery_eligibility"] == {
        "eligible": False, "reasons": ["no_decision"]}


@pytest.mark.parametrize("sha256,status,reasons", [
    ("a" * 64, "verified", []),
    ("b" * 64, "verified", ["record_mismatch"]),
    ("a" * 64, "not_checked", ["artifact_not_checked"]),
    ("a" * 64, "unsafe", ["artifact_unsafe"]),
])
def test_eligibility_needs_the_recorded_bytes_and_a_verified_file(sha256, status, reasons):
    from smythe.jobs.delivery import annotate_delivery, artifact_spec_identity

    spec = {"artifact": {"mime_type": "image/png"}}
    record = {"version": 1, "scope": "artifacts", "attempt_id": "a1", "decision": "accepted",
              "reasons": [], "validators": [{"validator": artifact_spec_identity(spec["artifact"])
                                            .to_dict(), "outcome": "passed", "reports": []}],
              "subjects": [{"path": "x.png", "sha256": "a" * 64, "size_bytes": 1,
                            "published": True}]}
    snapshot = {
        "operations": [{"operation_id": "op", "accepted_attempt_id": "a1", "spec": spec}],
        "attempts": [{"attempt_id": "a1", "operation_id": "op", "attempt_number": 1,
                      "status": "succeeded"}],
        "artifacts": [{"attempt_id": "a1", "operation_id": "op", "relative_path": "x.png",
                       "sha256": sha256, "accepted": 1, "integrity": {"status": status}}],
    }

    [operation] = annotate_delivery(snapshot, {"a1": record})["operations"]

    assert operation["delivery_eligibility"] == {"eligible": not reasons, "reasons": reasons}


def test_an_adopted_attempt_keeps_its_unknown_outcome_and_its_decision_is_projected():
    from smythe.jobs.delivery import annotate_delivery, artifact_spec_identity

    spec = {"artifact": {"mime_type": "image/png"}}
    record = {"version": 1, "scope": "artifacts", "attempt_id": "a2", "decision": "accepted",
              "reasons": [], "validators": [{"validator": artifact_spec_identity(spec["artifact"])
                                            .to_dict(), "outcome": "passed", "reports": []}],
              "subjects": [{"path": "x.png", "sha256": "a" * 64, "size_bytes": 1,
                            "published": True}]}
    snapshot = {
        "operations": [{"operation_id": "op", "status": "succeeded", "accepted_attempt_id": "a2",
                        "spec": spec}],
        "attempts": [{"attempt_id": "a1", "operation_id": "op", "attempt_number": 1,
                      "status": "rejected"},
                     {"attempt_id": "a2", "operation_id": "op", "attempt_number": 2,
                      "status": "unknown_outcome"}],
        "artifacts": [{"attempt_id": "a2", "operation_id": "op", "relative_path": "x.png",
                       "sha256": "a" * 64, "accepted": 1, "integrity": {"status": "verified"}}],
    }

    [operation] = annotate_delivery(snapshot, {"a1": dict(record, attempt_id="a1",
                                                          decision="rejected"),
                                               "a2": record})["operations"]

    assert operation["delivery_acceptance"]["attempt_id"] == "a2"
    assert operation["delivery_acceptance"]["decision"] == "accepted"
    assert operation["delivery_eligibility"] == {"eligible": True, "reasons": []}


def _reference_run(tmp_path):
    """An approved run whose first template sends a reference image and whose second sends none."""
    (tmp_path / "brand").mkdir()
    (tmp_path / "brand" / "logo.png").write_bytes(b"logo-bytes")
    manifest = JobManifestV1.from_dict({
        "version": 1, "name": "input-digest",
        "profiles": [{"name": "render", "provider": "gemini_image",
                      "model": "gemini-2.5-flash-image", "max_cost_per_call_usd": "0.05"}],
        "operations": [
            {"key": "hero", "count": 2, "prompt": "Render the hero", "profile": "render",
             "attachments": ["brand/logo.png"]},
            {"key": "plain", "prompt": "Render the hero", "profile": "render"},
        ],
        "execution": {"max_concurrency": 2, "max_attempts": 1, "max_budget_usd": "0.15",
                      "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    database = tmp_path / "jobs.sqlite3"
    with SQLiteRunStore(database) as store:
        run_id = store.create_run(plan, make_approval(plan), manifest_root=tmp_path)
    return database, run_id, plan


def test_inspection_reports_each_operations_input_digest_from_the_stored_plan(tmp_path):
    database, run_id, plan = _reference_run(tmp_path)
    expected = {operation.operation_id: operation_input_digest(operation, plan.attachments)
                for operation in plan.operations}
    before = database.read_bytes()
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)
        [selected] = inspect_job(store, run_id, operation="hero[1]")["operations"]
        paged = inspect_job(store, run_id, limit=1, offset=2)["operations"]

    assert database.read_bytes() == before
    digests = {operation["operation_id"]: operation["input_digest"]
               for operation in report["operations"]}
    assert digests == expected
    by_key = {operation["operation_key"]: operation["input_digest"]
              for operation in report["operations"]}
    assert by_key["hero[0]"] == by_key["hero[1]"] != by_key["plain[0]"]
    assert selected["input_digest"] == by_key["hero[1]"]
    assert [operation["input_digest"] for operation in paged] == [by_key["plain[0]"]]
    assert all("input_digest" not in operation["spec"] for operation in report["operations"])
    json.dumps(report)


def test_repeated_samples_without_attachments_share_an_input_digest(recorded):
    database, run_id, _ = recorded
    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    [digest] = {operation["input_digest"] for operation in report["operations"]}
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


def test_an_operation_whose_digest_cannot_be_computed_reports_none(tmp_path):
    valid = {"provider": "offline", "model": "offline-image", "options": {}, "prompt": "Draw",
             "attachments": [], "artifact": {"mime_type": "image/png"}}
    snapshot = {
        "run_id": "run", "manifest_root": str(tmp_path), "output_directory": "outputs",
        "artifacts": [],
        "operations": [
            {"operation_id": "valid", "spec": valid},
            {"operation_id": "no-model", "spec": {k: v for k, v in valid.items() if k != "model"}},
            {"operation_id": "unknown-provider", "spec": valid | {"provider": "other"}},
            {"operation_id": "unplanned-file", "spec": valid | {"attachments": ["unplanned.png"]}},
        ],
    }
    store = SimpleNamespace(inspection_snapshot=lambda *a, **kw: copy.deepcopy(snapshot),
                            plan_attachments=lambda run_id: [])

    digests = {operation["operation_id"]: operation["input_digest"]
               for operation in inspect_job(store, "run")["operations"]}

    assert digests["valid"].startswith("sha256:")
    assert {key: value for key, value in digests.items() if key != "valid"} == {
        "no-model": None, "unknown-provider": None, "unplanned-file": None}


@pytest.mark.parametrize("attachments", ["'not a list'", "json('[1]')",
                                         "json('[{\"relative_path\": \"brand/logo.png\"}]')"],
                         ids=["not-an-array", "not-objects", "missing-fields"])
def test_corrupt_stored_plan_attachments_give_null_input_digests(tmp_path, attachments):
    """An unreadable attachments array used to fail the whole inspection; the
    digest is informational, so only the digests it would feed are null."""
    database, run_id, plan = _reference_run(tmp_path)
    plain = operation_input_digest(plan.operations[-1], plan.attachments)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute(f"UPDATE runs SET plan_json=json_set(plan_json,'$.attachments',{attachments})")

    with SQLiteRunStore(database, read_only=True) as store:
        with pytest.raises(RunStoreError):
            store.plan_attachments(run_id)  # the store itself still fails closed
        report = inspect_job(store, run_id)

    digests = {operation["operation_key"]: operation["input_digest"]
               for operation in report["operations"]}
    assert digests == {"hero[0]": None, "hero[1]": None, "plain[0]": plain}


@pytest.mark.parametrize("spec", [
    "json_set(spec_json,'$.attachments',json('{\"brand/logo.png\": 1}'))",
    "json_set(spec_json,'$.attachments','brand/logo.png')",
    "json_set(spec_json,'$.attachments',json('[[\"brand/logo.png\"]]'))",
    "json_set(spec_json,'$.attachments',null)",
    "json_remove(spec_json,'$.attachments')",
], ids=["object", "text", "nested-array", "null", "missing"])
def test_corrupt_stored_operation_attachments_give_a_null_input_digest(tmp_path, spec):
    """Only the operation whose stored attachments are not an array of
    planned paths loses its digest; the inspection still succeeds."""
    database, run_id, plan = _reference_run(tmp_path)
    hero, _, plain = (operation_input_digest(operation, plan.attachments)
                      for operation in plan.operations)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute(f"UPDATE operations SET spec_json={spec} WHERE operation_key='hero[0]'")

    with SQLiteRunStore(database, read_only=True) as store:
        report = inspect_job(store, run_id)

    digests = {operation["operation_key"]: operation["input_digest"]
               for operation in report["operations"]}
    assert digests == {"hero[0]": None, "hero[1]": hero, "plain[0]": plain}
    json.dumps(report)


def _fake_store(tmp_path, **updates):
    data = b"fixture"
    path = tmp_path / "outputs" / "run" / "artifact.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    snapshot = {
        "run_id": "run", "manifest_root": str(tmp_path), "output_directory": "outputs",
        "artifacts": [{"relative_path": "artifact.bin", "size_bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}],
    }
    snapshot.update(updates)
    return SimpleNamespace(inspection_snapshot=lambda *a, **kw: copy.deepcopy(snapshot)), path, snapshot


@pytest.mark.parametrize("path", ["../outside", "..\\outside", "/etc/passwd",
                                  "C:\\outside", "C:outside", "\\\\host\\share",
                                  "artifact.bin:stream", ".", "", "a\x00b"])
def test_artifact_paths_do_not_escape_output_directory(tmp_path, path, monkeypatch):
    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"][0]["relative_path"] = path
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("unsafe file opened"))
    assert inspect_job(store, "run")["artifacts"][0]["integrity"]["status"] == "unsafe"


@pytest.mark.parametrize("field,value", [("output_directory", "../outside"),
                                        ("output_directory", "C:\\outside"),
                                        ("run_id", "../outside"),
                                        ("manifest_root", "."),
                                        ("manifest_root", b"bad")])
def test_unsafe_ledger_roots_are_findings_not_reads(tmp_path, field, value, monkeypatch):
    store, _, _ = _fake_store(tmp_path, **{field: value})
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("unsafe root opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


def test_manifest_output_dot_is_supported(tmp_path):
    store, path, _ = _fake_store(tmp_path, output_directory=".")
    target = tmp_path / "run" / "artifact.bin"
    target.parent.mkdir()
    target.write_bytes(path.read_bytes())
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"verified": 1}


def test_inspection_reads_recorded_artifact_directory_instead_of_raw_run_id(tmp_path):
    directory = "run-" + "a" * 32
    store, old_path, _ = _fake_store(tmp_path, artifact_directory=directory)
    target = tmp_path / "outputs" / directory / "artifact.bin"
    target.parent.mkdir()
    target.write_bytes(old_path.read_bytes())
    old_path.write_bytes(b"wrong raw-ID directory")
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"verified": 1}


@pytest.mark.parametrize("directory", [None, True, "", "../run", "C:\\outside", "run/file", "run:stream"])
def test_invalid_recorded_directory_never_falls_back_to_raw_run_id(tmp_path, directory, monkeypatch):
    store, _, _ = _fake_store(tmp_path, artifact_directory=directory)
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("invalid directory opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


def test_symbolic_link_is_never_hashed(tmp_path, monkeypatch):
    store, path, _ = _fake_store(tmp_path)
    original = tmp_path / "original.bin"
    original.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(original)
    except OSError:
        pytest.skip("Creating a symbolic link requires Windows Developer Mode")
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("symbolic link opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


def test_hash_work_is_bounded_across_artifacts(tmp_path, monkeypatch):
    import smythe.jobs.inspection as inspection

    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"].append(dict(snapshot["artifacts"][0]))
    monkeypatch.setattr(inspection, "MAX_INSPECTION_HASH_BYTES", 7)
    report = inspect_job(store, "run")
    assert report["artifact_integrity"]["counts"] == {"verified": 1, "not_checked": 1}
    assert report["artifact_integrity"]["bytes_hashed"] == 7


def test_unreadable_file_is_preserved_as_a_finding(tmp_path, monkeypatch):
    store, _, _ = _fake_store(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "open", denied)
    report = inspect_job(store, "run")
    assert report["artifact_integrity"]["counts"] == {"unreadable": 1}


@pytest.mark.parametrize("field,value", [("size_bytes", True), ("size_bytes", -1),
                                        ("size_bytes", 33 * 1024 * 1024),
                                        ("sha256", "not-a-hash")])
def test_invalid_artifact_metadata_is_not_used_to_read_files(tmp_path, field, value, monkeypatch):
    store, _, snapshot = _fake_store(tmp_path)
    snapshot["artifacts"][0][field] = value
    monkeypatch.setattr(os, "open", lambda *a, **kw: pytest.fail("invalid record opened"))
    assert inspect_job(store, "run")["artifact_integrity"]["counts"] == {"unsafe": 1}


# ---------------------------------------------------------------------------
# Eligibility under the current validators, after rechecks
# ---------------------------------------------------------------------------


def _palette(version):
    from smythe.validation import ValidatorIdentity

    return ValidatorIdentity.for_config("brand-palette", version, {"colors": ["#ff0000"]})


def _rechecked_eligibility(rechecks, registered=None, *, record=True):
    """Project one accepted operation whose plan names palette version 1."""
    from smythe.jobs.delivery import annotate_delivery, artifact_spec_identity

    artifact = {"mime_type": "image/png", "validators": [{"name": "brand-palette", "version": "1"}]}
    builtin = artifact_spec_identity(artifact).to_dict()
    results = [{"validator": builtin, "outcome": "passed", "reports": []},
               {"validator": _palette("1").to_dict(), "outcome": "passed", "reports": []}]
    recorded = {"version": 1, "scope": "artifacts", "attempt_id": "a1", "decision": "accepted",
                "reasons": [], "validators": results,
                "subjects": [{"path": "x.png", "sha256": "a" * 64, "size_bytes": 1,
                              "published": True}]}
    snapshot = {
        "operations": [{"operation_id": "op", "accepted_attempt_id": "a1",
                        "spec": {"artifact": artifact, "validators": [_palette("1").to_dict()]}}],
        "attempts": [{"attempt_id": "a1", "operation_id": "op", "attempt_number": 1,
                      "status": "succeeded"}],
        "artifacts": [{"attempt_id": "a1", "operation_id": "op", "relative_path": "x.png",
                       "sha256": "a" * 64, "accepted": 1, "integrity": {"status": "verified"}}],
    }
    [operation] = annotate_delivery(snapshot, {"a1": recorded} if record else {}, {"a1": rechecks},
                                    registered)["operations"]
    return operation["delivery_acceptance"], operation["delivery_eligibility"]


def _recheck_record(version, outcome, at_ns):
    from smythe.jobs.delivery import artifact_spec_identity

    builtin = artifact_spec_identity({"mime_type": "image/png"}).to_dict()
    passed = outcome == "accepted"
    return {"version": 1, "attempt_id": "a1", "reason": f"palette {version}", "outcome": outcome,
            "validators": [{"validator": builtin, "outcome": "passed", "reports": []},
                           {"validator": _palette(version).to_dict(),
                            "outcome": "passed" if passed else "rejected", "reports": []}],
            "findings": [] if passed else ["x.png: brand-palette: off palette"],
            "artifacts": [{"relative_path": "x.png", "sha256": "a" * 64}],
            "reevaluated_at_ns": at_ns}


def test_eligibility_uses_the_latest_decision_under_the_current_validators():
    current = {"brand-palette": _palette("2")}
    rejected, accepted = _recheck_record("2", "rejected", 1), _recheck_record("2", "accepted", 2)

    # The call decided under palette version 1, so version 2 requires a recheck.
    assert _rechecked_eligibility([]) == _rechecked_eligibility([], {"brand-palette": _palette("1")})
    assert _rechecked_eligibility([])[1] == {"eligible": True, "reasons": []}
    projection, eligibility = _rechecked_eligibility([], current)
    assert projection["source"] == "recorded"
    assert eligibility == {"eligible": False, "reasons": ["validator_changed"]}

    # The latest recheck under version 2 decides, whatever came before it.
    projection, eligibility = _rechecked_eligibility([rejected], current)
    assert (projection["source"], projection["decision"]) == ("reevaluated", "rejected")
    assert projection["reasons"] == ["x.png: brand-palette: off palette"]
    assert eligibility == {"eligible": False, "reasons": ["not_accepted"]}
    projection, eligibility = _rechecked_eligibility([rejected, accepted], current)
    assert (projection["source"], projection["decision"]) == ("reevaluated", "accepted")
    assert projection["validators"][1] == {"validator": _palette("2").to_dict(), "outcome": "passed"}
    assert eligibility == {"eligible": True, "reasons": []}
    assert _rechecked_eligibility([accepted, rejected], current)[1]["reasons"] == ["not_accepted"]

    # Under the plan's version 1, the call's own decision is still the latest that applies.
    projection, eligibility = _rechecked_eligibility([rejected])
    assert projection["source"] == "recorded" and eligibility["eligible"] is True

    # A legacy attempt without a recorded decision is decided by its recheck.
    projection, eligibility = _rechecked_eligibility([accepted], current, record=False)
    assert projection["source"] == "reevaluated" and eligibility["eligible"] is True
    assert _rechecked_eligibility([], current, record=False)[1]["reasons"] == ["validator_unrecorded"]


@pytest.mark.parametrize("change", [
    {"attempt_id": "a0"}, {"version": 2}, {"outcome": "maybe"}, {"artifacts": "x.png"},
    {"findings": None}, {"validators": [{"outcome": "passed"}]},
], ids=["other-attempt", "other-version", "outcome", "artifacts", "findings", "validators"])
def test_a_malformed_or_misplaced_recheck_never_decides(change):
    recheck = _recheck_record("2", "accepted", 1) | change

    projection, eligibility = _rechecked_eligibility([recheck], {"brand-palette": _palette("2")})

    assert projection["source"] == "recorded"
    assert eligibility == {"eligible": False, "reasons": ["validator_changed"]}


def test_a_recheck_of_other_bytes_is_not_eligible():
    recheck = _recheck_record("2", "accepted", 1)
    recheck["artifacts"] = [{"relative_path": "x.png", "sha256": "b" * 64}]

    _, eligibility = _rechecked_eligibility([recheck], {"brand-palette": _palette("2")})

    assert eligibility == {"eligible": False, "reasons": ["record_mismatch"]}
