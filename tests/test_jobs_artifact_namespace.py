"""Persistent artifact ownership and publication never replace accepted bytes."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import sqlite3
from contextlib import closing

import pytest
from PIL import Image

from smythe.cli import _portable_export
from smythe.jobs import make_approval
from smythe.jobs.artifact_io import atomic_publish_bytes, claim_artifact_root
from smythe.jobs.inspection import inspect_job
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import SQLiteRunStore
from smythe.provider import Artifact, CompletionResult
from tests.test_jobs_lease_fencing import FixedPool, PixelProvider, PNG
from tests.test_jobs_pause import plan_for


class ColoredProvider(PixelProvider):
    def __init__(self, color):
        super().__init__()
        stream = io.BytesIO()
        Image.new("RGBA", (1, 1), color).save(stream, format="PNG")
        self.png = stream.getvalue()

    async def complete(self, system, prompt, model):
        self.calls += 1
        return CompletionResult(text="pixel", artifacts=[Artifact(self.png, "image/png")])


def artifact_path(root, snapshot):
    return root / snapshot["output_directory"] / snapshot["artifact_directory"] / snapshot["artifacts"][0]["relative_path"]


@pytest.mark.parametrize("ids", [("Run", "run"), ("run.", "run"), ("CON", "con")])
def test_distinct_custom_ids_have_distinct_portable_artifact_directories(tmp_path, ids):
    async def scenario():
        plan = plan_for(tmp_path, count=1)
        with SQLiteRunStore(tmp_path / "jobs.db") as store:
            results = []
            for run_id, color in zip(ids, ("red", "blue"), strict=True):
                provider = ColoredProvider(color)
                result = await JobRunner(store, provider_pool=FixedPool(provider)).start(
                    plan, make_approval(plan), manifest_root=tmp_path, run_id=run_id,
                )
                assert result["status"] == "completed" and provider.calls == 1
                results.append(result)
            first, second = results
            assert first["artifact_directory"] != second["artifact_directory"]
            assert artifact_path(tmp_path, first).read_bytes() != artifact_path(tmp_path, second).read_bytes()
            for result in results:
                assert result["artifact_directory"] == "run-" + result["artifact_namespace"]
                assert inspect_job(store, result["run_id"])["artifact_integrity"]["counts"] == {"verified": 1}

    asyncio.run(scenario())


def test_two_stores_sharing_output_root_and_run_id_cannot_share_artifacts(tmp_path):
    async def scenario():
        plan = plan_for(tmp_path, count=1)
        results = []
        for index, color in enumerate(("red", "blue")):
            with SQLiteRunStore(tmp_path / f"jobs-{index}.db") as store:
                results.append(await JobRunner(store, provider_pool=FixedPool(ColoredProvider(color))).start(
                    plan, make_approval(plan), manifest_root=tmp_path, run_id="same-id",
                ))
        assert results[0]["artifact_owner_id"] != results[1]["artifact_owner_id"]
        for result in results:
            assert hashlib.sha256(artifact_path(tmp_path, result).read_bytes()).hexdigest() == result["artifacts"][0]["sha256"]

    asyncio.run(scenario())


def legacy_run(path, root, *, run_id="LegacyRun", accepted=True):
    plan = plan_for(root, count=2 if accepted else 1)
    with SQLiteRunStore(path) as store:
        store.create_run(plan, make_approval(plan), manifest_root=root, run_id=run_id)
        if accepted:
            operation = plan.operations[0]
            attempt = store.begin_attempt(run_id, operation.operation_id)
            call = store.prepare_call(attempt["attempt_id"], 0)
            store.mark_call_dispatched(call.call_id)
            relative = "artifacts/prior.png"
            prior = root / "outputs" / run_id / relative
            prior.parent.mkdir(parents=True, exist_ok=True)
            prior.write_bytes(PNG)
            store.complete_call(call.call_id, cost_microusd=0, cost_is_complete=True,
                                cost_is_estimate=False, result_text="prior pixel", artifacts=[{
                                    "relative_path": relative, "mime_type": "image/png", "size_bytes": len(PNG),
                                    "sha256": hashlib.sha256(PNG).hexdigest(), "width": 1, "height": 1,
                                }])
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TABLE run_controls")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_namespace")
        db.execute("ALTER TABLE runs DROP COLUMN artifact_owner_id")
        db.execute("PRAGMA user_version=3")
        db.commit()
    return plan


def test_legacy_read_export_and_resume_keep_original_paths_and_accepted_bytes(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_run(path, tmp_path)
    before = path.read_bytes()
    with SQLiteRunStore(path, read_only=True) as reader:
        old = inspect_job(reader, "LegacyRun")
        assert old["artifact_directory"] == "LegacyRun" and old["artifact_namespace"] is None
        assert old["artifact_owner_id"] is None
        assert old["artifact_integrity"]["counts"] == {"verified": 1}
        assert _portable_export(old)["artifact_root"] == "outputs/LegacyRun"
    assert path.read_bytes() == before
    with SQLiteRunStore(path) as store:
        provider = PixelProvider()
        result = asyncio.run(JobRunner(store, provider_pool=FixedPool(provider)).resume("LegacyRun"))
        assert result["status"] == "completed" and provider.calls == 1
        assert result["artifact_directory"] == "LegacyRun" and result["artifact_namespace"] is None
        assert result["artifact_owner_id"]
        accepted = old["operations"][0]["accepted_attempt_id"]
        assert result["operations"][0]["accepted_attempt_id"] == accepted
        assert (tmp_path / "outputs/LegacyRun/artifacts/prior.png").read_bytes() == PNG
        assert inspect_job(store, "LegacyRun")["artifact_integrity"]["counts"] == {"verified": 2}


def test_legacy_cross_store_marker_collision_fails_before_provider_entry(tmp_path):
    paths = [tmp_path / "first.db", tmp_path / "second.db"]
    for path in paths:
        legacy_run(path, tmp_path, run_id="shared", accepted=False)
    first_provider = PixelProvider()
    with SQLiteRunStore(paths[0]) as first:
        result = asyncio.run(JobRunner(first, provider_pool=FixedPool(first_provider)).resume("shared"))
    prior = artifact_path(tmp_path, result).read_bytes()
    with SQLiteRunStore(paths[1]) as second:
        provider = PixelProvider()
        with pytest.raises(ValueError, match="another run"):
            asyncio.run(JobRunner(second, provider_pool=FixedPool(provider)).resume("shared"))
        assert provider.calls == 0 and second.snapshot("shared")["counts"] == {"pending": 1}
    assert artifact_path(tmp_path, result).read_bytes() == prior


def test_unknown_legacy_ledger_destination_is_never_overwritten_after_dispatch(tmp_path):
    path = tmp_path / "legacy.db"
    plan = legacy_run(path, tmp_path, run_id="shared", accepted=False)
    assert plan.operations[0].operation_key == "item[0]"
    # Count expansion's brackets map to the existing portable item_0 directory.
    destination = tmp_path / "outputs/shared/artifacts/item_0/attempt-0001/artifact-01.png"
    destination.parent.mkdir(parents=True)
    prior_bytes = b"accepted bytes belonging to an unclaimed legacy ledger"
    destination.write_bytes(prior_bytes)
    with SQLiteRunStore(path) as store:
        provider = PixelProvider()
        result = asyncio.run(JobRunner(store, provider_pool=FixedPool(provider)).resume("shared"))
        assert provider.calls == 1 and result["status"] == "needs_attention"
        assert result["counts"] == {"unknown_outcome": 1}
        assert not result["artifacts"]
    assert destination.read_bytes() == prior_bytes


def test_corrupt_accepted_legacy_bytes_block_first_claim_before_dispatch(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_run(path, tmp_path)
    destination = tmp_path / "outputs/LegacyRun/artifacts/prior.png"
    destination.write_bytes(b"changed accepted bytes")
    with SQLiteRunStore(path) as store:
        provider = PixelProvider()
        with pytest.raises(ValueError, match="truncated|differs"):
            asyncio.run(JobRunner(store, provider_pool=FixedPool(provider)).resume("LegacyRun"))
        assert provider.calls == 0
    assert destination.read_bytes() == b"changed accepted bytes"
    assert not (destination.parents[1] / ".smythe-run-owner.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows path alias semantics")
def test_legacy_same_store_case_alias_is_rejected_before_first_claim(tmp_path):
    path = tmp_path / "aliases.db"
    plan = plan_for(tmp_path, count=1)
    with SQLiteRunStore(path) as store:
        for run_id in ("Run", "run"):
            store.create_run(plan, make_approval(plan), manifest_root=tmp_path, run_id=run_id)
        # Model the retained raw path identities assigned by legacy migration.
        store._connection.execute("UPDATE runs SET artifact_namespace=NULL")
        provider = PixelProvider()
        with pytest.raises(ValueError, match="aliases another run"):
            asyncio.run(JobRunner(store, provider_pool=FixedPool(provider)).resume("Run"))
        assert provider.calls == 0


@pytest.mark.parametrize("existing", [b"previous accepted artifact", b"new response"])
def test_atomic_publication_never_replaces_existing_bytes(tmp_path, existing):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(existing)
    with pytest.raises(FileExistsError):
        atomic_publish_bytes(destination, b"new response")
    assert destination.read_bytes() == existing
    assert list(tmp_path.iterdir()) == [destination]


def test_missing_hard_link_support_fails_before_provider_entry(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError("filesystem has no hard links")

    monkeypatch.setattr("smythe.jobs.artifact_io.os.link", unsupported)
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        plan, provider = plan_for(tmp_path, count=1), PixelProvider()
        with pytest.raises(OSError, match="no hard links"):
            asyncio.run(JobRunner(store, provider_pool=FixedPool(provider)).start(
                plan, make_approval(plan), manifest_root=tmp_path, run_id="run",
            ))
        assert provider.calls == 0 and store.snapshot("run")["counts"] == {"pending": 1}
        assert store.get_run_lease("run") is None


def test_incomplete_owner_marker_is_retained_and_rejected(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    marker = root / ".smythe-run-owner.json"
    marker.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="incomplete owner marker"):
        claim_artifact_root(root, b"expected", legacy=True, legacy_artifacts=[])
    assert marker.read_bytes() == b"incomplete"
