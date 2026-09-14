"""Fable campaign identity, matching, budgets and real-human gates; no network."""

import asyncio
import json

import pytest

from benchmarks import fable_runtime as runtime
from benchmarks.astra_campaign import load_task_pack
from benchmarks.astra_study import study_task
from smythe.task import task_to_dict
from smythe.workflow_store import SQLiteWorkflowStore


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def frozen(tmp_path, **kw):
    access = write(tmp_path / "access.json", {"model": "claude-fable-5-1", "status": "available"})
    checks = write(tmp_path / "checks.json", {"status": "passed", "full_offline_suite": "offline-test-fixture",
        "ruff": "offline-test-fixture", "source_sha256": runtime.sources()})
    return runtime.freeze(directory=tmp_path / kw.get("stage", "pilot"), model_access=access, validation=checks, **kw)


def test_pilot_matches_frozen_tasks_and_both_efforts(tmp_path):
    value = frozen(tmp_path)
    runtime.verify(value)
    assert len(value["schedule"]) == 12
    assert sum(row["effort"] == "medium" for row in value["schedule"]) == 6
    cases = {case.task_id: case for case in load_task_pack().tasks}
    for row in value["schedule"]:
        assert value["tasks"][row["task_id"]] == task_to_dict(study_task(cases[row["task_id"]]))
    assert sum(runtime.ALLOCATIONS.values()) == 100_000_000_000
    assert runtime.ALLOCATIONS["pilot"] + runtime.ALLOCATIONS["ultracode-pilot"] == 15_000_000_000
    assert runtime.ALLOCATIONS["main"] + runtime.ALLOCATIONS["ultracode-main"] == 75_000_000_000
    assert value["retained_old_unknown_nanousd"] == 169645000


def test_main_refuses_missing_human_review(tmp_path):
    with pytest.raises(ValueError, match="human review"):
        frozen(tmp_path, stage="main")


def test_runtime_changes_invalidate_freeze_before_dispatch(tmp_path, monkeypatch):
    value = frozen(tmp_path)
    monkeypatch.setattr(runtime, "sources", lambda: {})
    with pytest.raises(ValueError, match="Runtime changed"):
        asyncio.run(runtime.run(value))
    assert not (tmp_path / "pilot").exists()


def test_ultracode_unknown_exposure_blocks_native_stage(tmp_path):
    path = tmp_path / "ultracode-pilot"
    path.mkdir()
    write(path / "spending.json", {"status": "unknown", "confirmed_nanousd": 0,
                                  "reserved_nanousd": 0, "unknown_nanousd": 5_000_000_000})
    with pytest.raises(ValueError, match="unresolved"):
        runtime.campaign_balance(tmp_path)


def test_native_fable_factory_binds_all_phases_and_limits(tmp_path):
    row = frozen(tmp_path)["schedule"][0]
    with SQLiteWorkflowStore(tmp_path / "factory.sqlite3") as store:
        execution = runtime.swarm(store, row)
        recipe = execution._workflow_runtime().recipe
        serialized = json.dumps(recipe)
        assert "anthropic_messages" in serialized
        assert "claude-fable-5-1" in serialized and "gpt-6-astra" not in serialized
        assert "medium" in serialized and "api_key" not in serialized
