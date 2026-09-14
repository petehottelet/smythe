"""Offline main-judging orchestration checks; native calls are stubbed."""

import json
from copy import deepcopy

import pytest

from benchmarks import astra_judge_study as judge


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    outcomes = [{"run_id": str(i), "output": f"anonymous output {i}", "output_sha256": str(i),
                 "trial": {"task_id": "main-capacity-chain", "arm_id": f"hidden-{i}"}} for i in range(2)]
    freeze = {"freeze_sha256": "main", "campaign_allocations": {"judge_nanousd": 40_000_000_000}}
    monkeypatch.setattr(judge, "load_complete_main", lambda _: (freeze, deepcopy(outcomes)))
    monkeypatch.setattr(judge, "inspect_judgments", lambda _: {"paid_judgments": 10,
                        "cost_upper_nanousd": 124980000, "cost_lower_nanousd": 105442800})
    directory = tmp_path / "judge"
    directory.mkdir()
    (directory / "judge-freeze.json").write_text(json.dumps({"model": judge.MODEL,
        "model_version": judge.MODEL, "allowance_nanousd": 40_000_000_000}), encoding="utf-8")
    return {"main_directory": tmp_path, "judge_directory": directory, "bindings_path": tmp_path / "bindings.json"}


def test_inspection_never_buys_a_judgment(campaign, monkeypatch):
    monkeypatch.setattr(judge, "run_judgment", lambda *a, **k: pytest.fail("Unrequested paid call"))
    plan = judge.judge_main(**campaign)
    assert plan["live"] is False and plan["outputs"] == 2
    assert plan["prior_judge_upper_nanousd"] == 124980000
    assert not campaign["bindings_path"].exists()


def test_live_requires_credentials_without_changing_the_frozen_allowance(campaign, monkeypatch):
    monkeypatch.setattr(judge, "run_judgment", lambda *a, **k: pytest.fail("Unfunded call"))
    with pytest.raises(ValueError, match="API key"):
        judge.judge_main(**campaign, live=True)
    path = campaign["judge_directory"] / "judge-freeze.json"
    value = json.loads(path.read_bytes())
    value["allowance_nanousd"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="allocation"):
        judge.judge_main(**campaign, live=True, api_key="test-only")


def test_all_bindings_are_preserved_and_changed_bindings_rejected(campaign, monkeypatch):
    calls = []

    def score(case, output, **kwargs):
        calls.append((case.task_id, output, kwargs["allowance_nanousd"]))
        return {"identity": output, "record_sha256": output,
                "scores": {"criteria": [{"score": 4}], "material_defects": []}}

    monkeypatch.setattr(judge, "run_judgment", score)
    result = judge.judge_main(**campaign, live=True, api_key="test-only")
    assert result["status"] == "scored" and len(calls) == 2
    assert all(call[2] == 40_000_000_000 for call in calls)
    saved = json.loads(campaign["bindings_path"].read_bytes())
    assert [r["run_id"] for r in saved] == ["0", "1"]
    assert judge.judge_main(**campaign, live=True, api_key="test-only")["bindings_sha256"] == result["bindings_sha256"]
    saved[0]["output_sha256"] = "changed"
    campaign["bindings_path"].write_text(json.dumps(saved), encoding="utf-8")
    before = len(calls)
    with pytest.raises(ValueError, match="bindings differ"):
        judge.judge_main(**campaign, live=True, api_key="test-only")
    assert len(calls) == before


def test_missing_binding_parent_is_rejected_before_paid_work(campaign, monkeypatch):
    campaign["bindings_path"] = campaign["bindings_path"].parent / "missing" / "bindings.json"
    monkeypatch.setattr(judge, "run_judgment", lambda *a, **k: pytest.fail("Unsaveable paid call"))
    with pytest.raises(ValueError, match="parent must exist"):
        judge.judge_main(**campaign, live=True, api_key="test-only")


def test_missing_main_summary_cannot_start_judging(tmp_path):
    with pytest.raises(FileNotFoundError):
        judge.load_complete_main(tmp_path)


def test_continuation_judges_available_answers_and_keeps_missing_failure(campaign, monkeypatch):
    freeze, rows = judge.load_complete_main(campaign["main_directory"])
    rows.append({"run_id": "failed", "output": None})
    monkeypatch.setattr(judge, "load_continued_main", lambda *a: (freeze, rows,
                        {"unknown_nanousd": 169645000, "segments": [147, 53]}))
    monkeypatch.setattr(judge, "run_judgment", lambda *a, **k: pytest.fail("Unrequested paid call"))
    plan = judge.judge_main(**campaign, continuation_directory="separate")
    assert plan["no_output_failures"] == 1 and plan["outputs"] == 2
    assert plan["continuation"]["unknown_nanousd"] == 169645000
