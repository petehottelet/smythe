"""The offline selector comparison runs four arms on both task families, offline.

The harness runs in a child process without provider keys and with this
suite's network guard (``tests/conftest.py``) installed, recording every
socket use the guard refuses. Fixture runs establish mechanics only, so
these tests check what the record holds, never which arm wins.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from fakes import child_executable, offline_env

from benchmarks.astra_campaign import CampaignPlanError
from benchmarks.astra_campaign._json import digest
from benchmarks.astra_campaign.tasks import DATA_DIR, MAX_OUTPUT_BYTES
from benchmarks.offline_arms import ARMS, policy_unchanged, run_cell, williams_schedule
from benchmarks.run_selector_comparison import (
    load_heldout_pack, run_campaign, run_regression, shape_family, supplied_source_family,
)
from smythe import AcceptancePolicy, Task, TextValidator
from smythe.optimize import EvaluatorManifest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
HARNESS = ROOT / "benchmarks" / "run_selector_comparison.py"
RECORD = ROOT / "benchmarks" / "results" / "selector_comparison_offline.json"
FAMILIES = {"task-shapes", "supplied-source"}
VERDICTS = {"improved", "inconclusive", "worse"}
REQUIRED_GATES = {"accepted.nonzero", "acceptance.policy_unchanged", "accounting.complete",
                  "regression.no_new_failures"}

# Install the suite's network guard, run the harness, and save what the guard refused.
GUARDED_RUN = """
import json, runpy, sys
tests, harness, refused_path, *argv = sys.argv[1:]
guard, refused = [], []
def watch(event, args):
    if guard:
        try:
            guard[0](event, args)
        except RuntimeError:
            refused.append(event)
sys.addaudithook(watch)
sys.path.insert(0, tests)
import conftest
guard.append(conftest._offline_network)
sys.argv = [harness, *argv]
try:
    runpy.run_path(harness, run_name="__main__")
finally:
    with open(refused_path, "w", encoding="utf-8") as stream:
        json.dump(refused, stream)
"""


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("selector")
    out, refused = work / "record.json", work / "refused.json"
    proc = subprocess.run(
        [child_executable(), "-c", GUARDED_RUN, str(TESTS), str(HARNESS), str(refused),
         "--out", str(out)],
        cwd=ROOT, env=offline_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return (json.loads(out.read_text(encoding="utf-8")),
            json.loads(refused.read_text(encoding="utf-8")))


def _identities(validators):
    return sorted((item["name"], item["version"], item["config_sha256"]) for item in validators)


def _attempts(family):
    """Every attempt the family's record holds: sweep, regression cases and campaign trials."""
    for run in (*family["runs"], *family["campaign"]["trials"]):
        for attempt in run["attempts"]:
            yield run["arm"], attempt
    for arm, entry in family["regression"]["arms"].items():
        for case in entry["cases"]:
            for attempt in case["attempts"]:
                yield arm, attempt


def test_the_run_is_offline_and_labeled_as_mechanics(offline_run):
    record, refused = offline_run
    assert refused == []
    assert record["evidence_status"] == "offline-mechanics"
    assert record["claimable"] is False and record["paid_execution_allowed"] is False
    assert record["api_calls"] == 0 and record["mode"] == "offline"
    assert record["default_change"] is False and record["promotion"]["eligible"] is False
    assert record["cost_protocol"]["invoice_cost"] is False


def test_all_four_arms_complete_on_both_families_under_one_policy(offline_run):
    record, _ = offline_run
    assert record["arms"] == list(ARMS) and set(record["families"]) == FAMILIES
    for family in record["families"].values():
        tasks = {task["task_id"] for task in family["tasks"]}
        runs = family["runs"]
        assert sorted((run["task_id"], run["arm"]) for run in runs) == sorted(
            (task, arm) for task in tasks for arm in ARMS)
        assert all(run["accepted"] for run in runs)
        # Every block of the schedule runs each arm once.
        blocks = Counter(run["block"] for run in runs)
        assert set(blocks.values()) == {len(ARMS)}
        policy = family["acceptance_policy"]
        assert policy["mode"] == "strict"
        for _, attempt in _attempts(family):
            assert attempt["policy_sha256"] == policy["sha256"]
            if attempt["validators"]:
                assert _identities(attempt["validators"]) == _identities(policy["final_validators"])
        for entry in family["summary"].values():
            assert entry["cells"] == len(tasks) and entry["accepted"] == len(tasks)
            assert entry["false_acceptances"] == 0


def test_every_failed_attempt_is_recorded_and_charged(offline_run):
    record, _ = offline_run
    for name, outcome in (("task-shapes", "error"), ("supplied-source", "rejected")):
        family = record["families"][name]
        planted = [task["task_id"] for task in family["tasks"] if task["planted"]]
        assert len(planted) == 1
        runs = [run for run in family["runs"] if run["task_id"] == planted[0]]
        assert sorted(run["arm"] for run in runs) == sorted(ARMS)
        for run in runs:
            first, second = run["attempts"]
            assert first["planted"] and first["outcome"] == outcome
            assert second["planted"] is None and second["outcome"] == "accepted"
            assert run["failed_attempts"] == 1
            assert run["cost_microusd"] == first["cost_microusd"] + second["cost_microusd"]
            if outcome == "error":
                # Calls answered before the fault are charged; the failed call had no usage.
                assert first["error"]["type"] == "FixtureFault"
                assert first["provider_calls_failed"] == 1
                assert first["latency_source"] == "failed_checkpoint_timing"
                if run["arm"] == "generated_plan":
                    assert first["planning_calls"] == 1 and first["cost_microusd"] > 0
            else:
                assert first["decision"] == "rejected" and first["cost_microusd"] > 0
                assert first["adjudication"]["deterministic_passed"] is False
        assert {entry["failed_attempts"] for entry in family["summary"].values()} == {1}


def test_every_adaptive_attempt_is_charged_its_selector_cost_and_latency(offline_run):
    record, _ = offline_run
    for family in record["families"].values():
        methods = Counter()
        for arm, attempt in _attempts(family):
            selector = attempt["selector"]
            if arm != "adaptive_selection":
                assert selector is None
                assert attempt["cost_microusd"] == attempt["model_cost_microusd"]
                continue
            methods[selector["method"]] += 1
            assert type(selector["cost_microusd"]) is int
            assert isinstance(selector["latency_ms"], float) and selector["latency_ms"] >= 0
            assert selector["reasons"] and selector["tier"] in {
                "single", "deterministic:pipeline", "autonomous"}
            assert attempt["cost_microusd"] == (attempt["model_cost_microusd"]
                                                + selector["cost_microusd"])
            if selector["method"] == "classifier":
                assert selector["cost_microusd"] > 0 and selector["classifier_calls"] == 1
                assert selector["cost_source"] == "declared classifier usage"
            else:
                assert selector["method"] == "rule"
                assert selector["cost_microusd"] == 0 and selector["classifier_calls"] == 0
        # Both evidence sets exercise rules and the scripted classifier.
        assert set(methods) == {"rule", "classifier"}
        adaptive = family["summary"]["adaptive_selection"]["selection"]
        assert adaptive["cost_microusd"] > 0 and adaptive["latency_measured"] == (
            family["summary"]["adaptive_selection"]["attempts"])


def test_the_autotune_decision_has_a_verdict_and_reference_assessments(offline_run):
    record, _ = offline_run
    for family in record["families"].values():
        campaign = family["campaign"]
        assert campaign["roles"] == {"incumbent": "generated_plan",
                                     "challenger": "adaptive_selection",
                                     "references": ["single_worker", "fixed_template"]}
        decision = campaign["decision"]
        assert decision["verdict"] in VERDICTS
        assert decision["incumbent_arm"] == "generated_plan"
        assert decision["selected_arm"] == "adaptive_selection"
        assert decision["confirmation_assessment"]["verdict"] in VERDICTS
        references = decision["references"]
        assert [item["arm"] for item in references] == ["single_worker", "fixed_template"]
        for item in references:
            assert item["verdict"] in VERDICTS
            assert item["confirmation"]["primary"]["sample_count"] == 3
            assert item["confirmation"]["primary"]["objective_name"] == "cost_microusd"
        manifest = EvaluatorManifest.from_dict(campaign["evaluator_manifest"])
        assert manifest.evaluator_hash == campaign["evaluator_hash"]
        assert _identities(item.to_dict() for item in manifest.validators) == _identities(
            family["acceptance_policy"]["final_validators"])
        assert manifest.price_snapshot_version == record["cost_protocol"]["price_snapshot_version"]
        assert set(manifest.split_manifests) == {"development", "confirmation", "holdout"}
        assert campaign["holdout"]["held_out"] is False
        assert set(campaign["contract"]["required_gates"]) == REQUIRED_GATES
        for trial in campaign["trials"]:
            assert set(trial["gates"]) == REQUIRED_GATES and all(trial["gates"].values())
            if trial["role"] == "reference":
                assert trial["arm"] in ("single_worker", "fixed_template")
                assert trial["split"] != "development"


def test_the_regression_gate_pins_reference_and_defect_in_every_arm(offline_run):
    record, _ = offline_run
    for family in record["families"].values():
        regression = family["regression"]
        assert regression["mode"] == "no_new_failures" and set(regression["arms"]) == set(ARMS)
        for entry in regression["arms"].values():
            assert entry["passed"] is True
            observed = {case["case"]: case["observed"] for case in entry["cases"]}
            assert observed == {"reference-output": "accepted", "planted-defect": "rejected"}
            # The reference output also passes the answer key where the family has one.
            reference = entry["cases"][0]
            assert reference["case"] == "reference-output"
            assert reference["adjudication_passed"] is (
                True if family["adjudication"] else None)


def test_the_committed_record_is_labeled_and_never_cited_as_a_claim():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["benchmark"] == "selector-comparison"
    assert record["evidence_status"] == "offline-mechanics" and record["claimable"] is False
    assert record["api_calls"] == 0 and record["default_change"] is False
    assert set(record["families"]) == FAMILIES
    for name in ("README.md", "ROADMAP.md"):
        assert "selector_comparison" not in (ROOT / name).read_text(encoding="utf-8")


def test_a_raised_attempt_keeps_its_routing_decision_and_charge():
    family = shape_family()
    task = next(item for item in family.tasks if item.task_id == "mixed-audit")
    cell = run_cell(replace(task, planted="fault"), "adaptive_selection", family.policy)
    first, second = cell["attempts"]
    assert first["outcome"] == "error" and first["error"]["type"] == "FixtureFault"
    assert first["decision"] == "not_evaluated" and first["policy_sha256"] == family.policy.sha256
    # The failed checkpoint keeps the decision and its routing time.
    assert first["latency_source"] == "failed_checkpoint_timing"
    assert first["selector"]["method"] == "classifier" and first["selector"]["cost_microusd"] > 0
    assert isinstance(first["selector"]["latency_ms"], float)
    assert second["outcome"] == "accepted" and cell["accepted"] and cell["failed_attempts"] == 1
    assert cell["selector_cost_microusd"] == (first["selector"]["cost_microusd"]
                                              + second["selector"]["cost_microusd"])

    spent = run_cell(replace(task, planted="defect"), "single_worker", family.policy, allowance=1)
    assert spent["accepted"] is False and spent["failed_attempts"] == 1
    assert spent["attempts"][0]["outcome"] == "rejected"


def test_the_williams_schedule_counterbalances_four_arms():
    schedule = williams_schedule("family", ["a", "b", "c", "d"], seed=7)
    assert schedule == williams_schedule("family", ["a", "b", "c", "d"], seed=7)
    blocks = [[slot["arm"] for slot in schedule if slot["block"] == block] for block in range(4)]
    assert all(sorted(row) == sorted(ARMS) for row in blocks)
    for arm in ARMS:
        assert sorted(row.index(arm) for row in blocks) == [0, 1, 2, 3]
    follows = Counter((row[i], row[i + 1]) for row in blocks for i in range(3))
    assert len(follows) == 12 and set(follows.values()) == {1}
    assert len(williams_schedule("family", ["a", "b"], seed=7, repetitions=2)) == 16
    with pytest.raises(ValueError):
        williams_schedule("family", ["a"], ARMS[:3], seed=7)


def test_a_heldout_pack_is_verified_and_kept_outside_the_repository(tmp_path):
    with pytest.raises(CampaignPlanError, match="outside the repository"):
        load_heldout_pack(DATA_DIR)
    copy = tmp_path / "pack"
    shutil.copytree(DATA_DIR, copy)
    with pytest.raises(CampaignPlanError, match="not a held-out pack"):
        load_heldout_pack(copy)
    changed = copy / "tasks" / "main-calendar.json"
    changed.write_text(changed.read_text(encoding="utf-8").replace("Normalize all", "Rewrite all"),
                       encoding="utf-8")
    with pytest.raises(CampaignPlanError, match="hash mismatch"):
        load_heldout_pack(copy)

    manifest_path = copy / "pack-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"]["tasks/main-calendar.json"] = digest(changed.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pack = load_heldout_pack(copy)
    family = supplied_source_family(pack)
    assert [task.task_id for task in family.holdout] == sorted(
        case.task_id for case in pack.for_stage("main"))
    assert family.holdout_manifest["pack_manifest_sha256"] == pack.manifest_sha256
    assert family.holdout_manifest["pack_manifest_sha256"] != (
        family.split_manifest["pack_manifest_sha256"])
    assert {task.task_id for task in family.holdout}.isdisjoint(
        task.task_id for task in family.tasks)

    # A record that names held-out tasks is never written into the repository.
    proc = subprocess.run([child_executable(), str(HARNESS), "--heldout-pack", str(copy)],
                          cwd=ROOT, env=offline_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "outside the repository" in proc.stderr


def test_the_policy_gate_fails_an_attempt_decided_by_another_policy():
    family = shape_family()
    task = next(item for item in family.tasks if item.task_id == "trivial-transform")
    cell = run_cell(task, "single_worker", family.policy, allowance=1)
    assert policy_unchanged(cell, family.policy) is True
    looser = AcceptancePolicy(mode="strict", final_validators=[
        TextValidator({"min_chars": 1, "max_chars": MAX_OUTPUT_BYTES})])
    assert policy_unchanged(cell, looser) is False
    attempt = {**cell["attempts"][0], "policy_sha256": looser.sha256}
    assert policy_unchanged({**cell, "attempts": [attempt]}, family.policy) is False


def _with_regression_task(family, **changes):
    """*family* with its pinned regression task changed."""
    tasks = tuple(replace(task, **changes) if task.task_id == family.regression_task else task
                  for task in family.tasks)
    return replace(family, tasks=tasks)


def _regression_task(family):
    return next(task for task in family.tasks if task.task_id == family.regression_task)


@pytest.mark.parametrize("failure", ["answer-key-failed", "defect-accepted"])
def test_the_regression_gate_fails_when_a_pinned_case_regresses(failure):
    family = supplied_source_family()
    task = _regression_task(family)
    if failure == "answer-key-failed":
        # Valid JSON, so strict acceptance takes it, but the answer key does not.
        pinned = _with_regression_task(family, deliverable='{"answer": "wrong"}')
        regressed, held = "reference-output", "planted-defect"
    else:
        pinned = _with_regression_task(family, defect=task.deliverable)
        regressed, held = "planted-defect", "reference-output"
    regression = run_regression(pinned)
    assert set(regression["arms"]) == set(ARMS)
    for entry in regression["arms"].values():
        assert entry["passed"] is False
        cases = {case["case"]: case for case in entry["cases"]}
        assert cases[regressed]["observed"] == "accepted" and cases[regressed]["passed"] is False
        assert cases[held]["passed"] is True
        assert cases["reference-output"]["adjudication_passed"] is (failure != "answer-key-failed")
        for case in cases.values():
            assert case["adjudication_passed"] is (
                case["attempts"][-1]["adjudication"]["deterministic_passed"])

    # Without an answer key, a pinned accepted case needs only acceptance.
    shapes = shape_family()
    assert all(entry["passed"] and entry["cases"][0]["adjudication_passed"] is None
               for entry in run_regression(shapes)["arms"].values())


def _conversions():
    """Three date conversions on which adaptive selection passes confirmation."""
    shapes = shape_family()
    base = next(item for item in shapes.tasks if item.task_id == "trivial-transform")
    dates = ["March 3 2024", "7/4/1999", "22 Oct 2011", "Jan 1, 2000", "2023-12-31"]
    tasks = tuple(
        replace(base, task_id=f"convert-{count}", sha256=f"{count:064x}",
                task=Task(goal="Convert each of these dates to ISO 8601: "
                               + "; ".join(dates[:count]), constraints=["Output only JSON"]),
                deliverable=f"Fixture deliverable for {count} converted dates.")
        for count in (2, 3, 5))
    return replace(shapes, name="conversions", tasks=tasks, regression_task="convert-2",
                   split_manifest={"family": "conversions"})


def _exact_answer(expected):
    def adjudicate(output):
        return {"deterministic_passed": output == expected}
    return adjudicate


@pytest.mark.parametrize("failure", ["answer-key-failed", "defect-accepted"])
def test_a_failed_regression_gate_fails_every_trial_and_blocks_promotion(failure):
    # With its pinned cases holding, this family's challenger passes confirmation
    # (test_a_challenger_that_passes_confirmation_meets_the_holdout_split).
    family = _conversions()
    task = _regression_task(family)
    if failure == "answer-key-failed":
        pinned = _with_regression_task(
            family, adjudicate=_exact_answer(task.deliverable),
            deliverable="Fixture deliverable that converts none of the dates.")
    else:
        pinned = _with_regression_task(family, defect=task.deliverable)
    regression = run_regression(pinned)
    assert not any(entry["passed"] for entry in regression["arms"].values())

    campaign = asyncio.run(run_campaign(family, regression, seed=3))
    assert campaign["trials"]
    for trial in campaign["trials"]:
        assert trial["gates"]["regression.no_new_failures"] is False
        assert trial["gates"]["acceptance.policy_unchanged"] is True
    decision = campaign["decision"]
    assert decision["promoted"] is False and decision["confirmation_assessment"] is None
    assert decision["reason"] == "no candidate passed development gates and hard bounds"
    assert not any(score["viable"] for score in decision["development_scores"])


def test_a_challenger_that_passes_confirmation_meets_the_holdout_split():
    family = _conversions()
    campaign = asyncio.run(run_campaign(family, run_regression(family), seed=3))
    decision = campaign["decision"]
    # One worker answers what the generated plan spends a planning call on.
    assert decision["confirmation_assessment"]["promote"] is True
    assert decision["holdout_assessment"]["verdict"] in VERDICTS
    assert campaign["holdout"] == {
        "source": "development-stand-in", "held_out": False,
        "note": campaign["holdout"]["note"]}
    holdout = [trial for trial in campaign["trials"] if trial["split"] == "holdout"]
    assert Counter(trial["arm"] for trial in holdout) == dict.fromkeys(ARMS, 3)
    assert all(item["holdout"] is not None for item in decision["references"])
