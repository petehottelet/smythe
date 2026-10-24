"""The offline trigger ablation: routine review, evidence-triggered review and none.

The harness runs in a child process without provider keys and with this
suite's network guard installed (``GUARDED_RUN``), recording every socket
use the guard refuses. Fixture runs establish mechanics only, so these tests
check what the record holds and how the arms differ on fixtures planted for
that purpose, never a measured benefit.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from fakes import child_executable, offline_env
from test_selector_comparison import GUARDED_RUN

from benchmarks.offline_arms import run_cell
from benchmarks.run_trigger_ablation import (
    ANOMALIES, ARMS, REVISE, ScriptedReviewer, acceptance_policy, balanced_schedule, fixture_tasks,
    review_state, run_attempt, run_campaign, run_task,
)
from smythe import EvidenceTriggeredSupervisor, LLMSupervisor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.optimize import EvaluatorManifest
from smythe.supervisor import EVIDENCE_REVIEW_PROMPT, REVIEW_PROMPT, SUPERVISOR_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
HARNESS = ROOT / "benchmarks" / "run_trigger_ablation.py"
RECORD = ROOT / "benchmarks" / "results" / "trigger_ablation_offline.json"
VERDICTS = {"improved", "inconclusive", "worse"}
# The anomalies a revision can repair; a gate's rejection stands under strict acceptance.
REPAIRABLE = {"empty_output", "failed_step", "truncated_step"}


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("trigger")
    out, refused = work / "record.json", work / "refused.json"
    proc = subprocess.run(
        [child_executable(), "-c", GUARDED_RUN, str(TESTS), str(HARNESS), str(refused),
         "--out", str(out)],
        cwd=ROOT, env=offline_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return (json.loads(out.read_text(encoding="utf-8")),
            json.loads(refused.read_text(encoding="utf-8")))


def _cells(record):
    """Each sweep cell by (task id, repetition) and arm."""
    cells = {}
    for run in record["runs"]:
        cells.setdefault((run["task_id"], run["repetition"]), {})[run["arm"]] = run
    return cells


def _supervision(cell):
    [attempt] = cell["attempts"]
    return attempt["supervision"]


def test_the_run_is_offline_and_labeled_as_mechanics(offline_run):
    record, refused = offline_run
    assert refused == []
    assert record["benchmark"] == "trigger-ablation"
    assert record["evidence_status"] == "offline-mechanics"
    assert record["claimable"] is False and record["paid_execution_allowed"] is False
    assert record["api_calls"] == 0 and record["mode"] == "offline"
    assert record["default_change"] is False and record["promotion"]["eligible"] is False
    assert record["cost_protocol"]["invoice_cost"] is False


def test_routine_and_triggered_review_accept_the_same_outcomes(offline_run):
    record, _ = offline_run
    assert record["arms"] == list(ARMS)
    cells = _cells(record)
    tasks = {task["task_id"]: task for task in record["tasks"]}
    assert {task_id for task_id, _ in cells} == set(tasks)
    policy = record["acceptance_policy"]
    for (task_id, _), arms in cells.items():
        assert set(arms) == set(ARMS)
        assert arms["routine"]["accepted"] == arms["triggered"]["accepted"], task_id
        for cell in arms.values():
            [attempt] = cell["attempts"]
            assert attempt["policy_sha256"] == policy["sha256"]
            assert attempt["error"] is None
        anomaly = tasks[task_id]["anomaly"]
        if anomaly is None:
            assert all(cell["accepted"] for cell in arms.values())
        elif anomaly in REPAIRABLE:
            # Review repairs the run; without a supervisor strict acceptance rejects it.
            assert arms["routine"]["accepted"] and not arms["none"]["accepted"]
            assert any("required section 'Risks' is missing" in reason
                       for reason in arms["none"]["attempts"][0]["reasons"])
    comparison = record["comparison"]
    assert comparison["routine_and_triggered_accept_the_same_tasks"] is True
    assert comparison["accepted"]["routine"] == comparison["accepted"]["triggered"]
    assert comparison["accepted"]["none"] < comparison["accepted"]["routine"]
    assert sorted(comparison["accepted_only_with_supervision"]) == sorted(
        task_id for task_id, task in tasks.items() if task["anomaly"] in REPAIRABLE)
    for arm, entry in record["summary"].items():
        assert entry["accepted"] == comparison["accepted"][arm]
        assert entry["false_acceptances"] == 0


def test_the_triggered_arm_makes_strictly_fewer_supervision_calls(offline_run):
    record, _ = offline_run
    tasks = {task["task_id"]: task for task in record["tasks"]}
    totals = Counter()
    for (task_id, _), arms in _cells(record).items():
        calls = {arm: _supervision(cell)["calls"] for arm, cell in arms.items()}
        totals.update(calls)
        assert calls["none"] == 0
        assert calls["triggered"] <= calls["routine"], task_id
        if tasks[task_id]["anomaly"] is None:
            # A clean run fires no trigger, while routine review still asks the model.
            assert calls["triggered"] == 0 and calls["routine"] > 0
        for arm, cell in arms.items():
            [attempt] = cell["attempts"]
            supervision = attempt["supervision"]
            assert supervision["calls"] == len(supervision["reviews"])
            assert attempt["cost_microusd"] == (attempt["model_cost_microusd"]
                                                + supervision["cost_microusd"])
            assert (supervision["cost_microusd"] > 0) is (supervision["calls"] > 0)
    assert totals["triggered"] < totals["routine"]
    comparison = record["comparison"]
    assert comparison["supervision_calls"] == dict(totals)
    assert comparison["triggered_makes_fewer_calls"] is True
    for arm in ARMS:
        assert record["summary"][arm]["supervision"]["calls"] == totals[arm]


def test_each_planted_anomaly_fires_its_triggers_in_one_review(offline_run):
    record, _ = offline_run
    tasks = {task["task_id"]: task for task in record["tasks"]}
    assert {task["anomaly"] for task in tasks.values()} == {None, *ANOMALIES}
    for (task_id, _), arms in _cells(record).items():
        task = tasks[task_id]
        triggered, routine = _supervision(arms["triggered"]), _supervision(arms["routine"])
        # Routine review sends the usual prompt; a triggered review leads with its evidence.
        assert not any(review["with_evidence"] for review in routine["reviews"])
        assert routine["reviews_triggered"] == []
        if task["anomaly"] is None:
            assert triggered["reviews"] == [] and triggered["reviews_triggered"] == []
            continue
        assert task["expected_triggers"] == list(ANOMALIES[task["anomaly"]])
        [review] = triggered["reviews"]
        [fired] = triggered["reviews_triggered"]
        assert review["with_evidence"] is True and review["after_node"] == fired["after_node"]
        assert fired["triggers"] == task["expected_triggers"]
        if task["anomaly"] in REPAIRABLE or task["anomaly"] == "rejecting_gate":
            for supervision in (triggered, routine):
                assert supervision["revisions_applied"] == 1
                assert supervision["added_steps"] == [REVISE]
        else:
            # Contradicting evidence is reviewed, and the plan stands.
            assert triggered["proposals"] == routine["proposals"] == 0
        if task["anomaly"] == "rejecting_gate":
            # A revision cannot turn a gate's rejection into an accepted outcome.
            for cell in arms.values():
                assert cell["accepted"] is False
                assert any("gate 'check' has a failing final verdict" in reason
                           for reason in cell["attempts"][0]["reasons"])


def test_the_autotune_campaign_compares_routine_triggered_and_none(offline_run):
    record, _ = offline_run
    campaign = record["campaign"]
    assert campaign["roles"] == {"incumbent": "routine", "challenger": "triggered",
                                 "references": ["none"]}
    decision = campaign["decision"]
    assert decision["verdict"] in VERDICTS
    assert decision["incumbent_arm"] == "routine" and decision["selected_arm"] == "triggered"
    confirmation = decision["confirmation_assessment"]
    assert confirmation["primary"]["objective_name"] == "supervision_calls"
    [accepted] = [item for item in confirmation["secondary"]
                  if item["objective_name"] == "accepted_outcome_rate"]
    assert accepted["non_regression_passed"] is True
    [reference] = decision["references"]
    assert reference["arm"] == "none" and reference["verdict"] in VERDICTS
    manifest = EvaluatorManifest.from_dict(campaign["evaluator_manifest"])
    assert manifest.evaluator_hash == campaign["evaluator_hash"]
    assert [item.to_dict() for item in manifest.validators] == (
        record["acceptance_policy"]["final_validators"])
    assert_mechanics_only(record)
    task_ids = sorted(task["task_id"] for task in record["tasks"])
    for split in ("confirmation", "holdout"):
        trials = [trial for trial in campaign["trials"] if trial["split"] == split]
        # Each paired split runs every arm once on every fixture task.
        for arm in ARMS:
            assert sorted(trial["task_id"] for trial in trials if trial["arm"] == arm) == task_ids
    for trial in campaign["trials"]:
        assert set(trial["gates"]) == set(campaign["contract"]["required_gates"])
        assert all(trial["gates"].values())
        if trial["role"] == "reference":
            assert trial["arm"] == "none" and trial["split"] != "development"


def assert_mechanics_only(record):
    """The campaign's holdout is labeled a stand-in, and its decision promotes nothing."""
    campaign = record["campaign"]
    assert campaign["holdout"]["held_out"] is False
    assert campaign["holdout"]["label"] == "offline stand-in"
    assert "not an untouched holdout" in campaign["holdout"]["note"]
    status = campaign["decision_status"]
    assert status["mechanics_only"] is True and status["promotes"] is False
    assert "untouched holdout" in status["note"] and "promotes nothing" in status["note"]
    assert any("mechanics only" in reason and "promotes nothing" in reason
               for reason in record["promotion"]["reasons"])
    assert record["promotion"]["eligible"] is False and record["claimable"] is False


def test_the_committed_record_is_labeled_and_never_cited_as_a_claim():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["benchmark"] == "trigger-ablation"
    assert record["evidence_status"] == "offline-mechanics" and record["claimable"] is False
    assert record["api_calls"] == 0 and record["default_change"] is False
    assert record["comparison"]["routine_and_triggered_accept_the_same_tasks"] is True
    assert_mechanics_only(record)
    # The record was generated from a committed revision with no local changes.
    assert record["environment"]["source_control"]["dirty"] is False
    for name in ("README.md", "ROADMAP.md"):
        assert "trigger_ablation" not in (ROOT / name).read_text(encoding="utf-8")
    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Trigger ablation (offline mechanics)", 1)[1].split("\n## ", 1)[0]
    assert "offline stand-in" in section and "promotes nothing" in section


def _review(graph, node, *, evidence: bool) -> dict:
    """The scripted reviewer's call after *node*: a routine review, or one evidence triggered."""
    reviewer = ScriptedReviewer()
    supervisor = (EvidenceTriggeredSupervisor(LLMSupervisor(reviewer)) if evidence
                  else LLMSupervisor(reviewer, only_terminal=False))
    asyncio.run(supervisor.review(graph, node, task=None, revisions_remaining=2))
    [call] = reviewer.calls
    return call


def _graph(risks: str, *, revised: bool = False) -> tuple[ExecutionGraph, Node]:
    risks_node = Node(id="risks", label="risks", status=NodeStatus.COMPLETED, result=risks)
    nodes = [Node(id="facts", label="facts", status=NodeStatus.COMPLETED, result="## Findings\nok"),
             risks_node, Node(id="deliver", label="deliver", depends_on=["facts", "risks"])]
    if revised:
        nodes.append(Node(id=REVISE, label="revise", depends_on=["deliver"]))
    return ExecutionGraph([Topology.FORK_JOIN], nodes), risks_node


def test_the_scripted_reviewer_decides_from_the_run_state_not_the_evidence():
    graph, node = _graph("")
    plain = _review(graph, node, evidence=False)
    with_evidence = _review(graph, node, evidence=True)
    assert plain["with_evidence"] is False and with_evidence["with_evidence"] is True
    assert plain["reply"] == with_evidence["reply"]
    proposal = json.loads(plain["reply"])
    assert proposal["change"] is True and proposal["add"][0]["id"] == REVISE
    assert proposal["add"][0]["depends_on"] == ["deliver"]
    assert plain["after_node"] == "risks" and plain["change"] is True

    # Complete output, or a revision step already in the plan, needs no change.
    complete = _review(*_graph("## Risks\nok"), evidence=False)
    revised = _review(*_graph("", revised=True), evidence=False)
    assert complete["change"] is False and revised["change"] is False
    assert json.loads(revised["reply"]) == {
        "change": False, "reason": "a revision step already covers the gap"}


def test_the_review_state_follows_the_review_prompt():
    for marker in ("Completed work so far:", "Remaining planned steps (not yet executed):",
                   "The step that just finished was"):
        assert marker in REVIEW_PROMPT
    graph, node = _graph("")
    prompt = LLMSupervisor(ScriptedReviewer())._build_prompt(graph, node, None, 1)
    completed, pending, finished = review_state(EVIDENCE_REVIEW_PROMPT.format(
        evidence="- empty_or_short_output: 'risks' returned 0 characters", review=prompt))
    assert completed == {"facts": "## Findings\nok", "risks": ""}
    assert pending == ["deliver"] and finished == "risks"
    with pytest.raises(ValueError, match="not a review prompt"):
        review_state("Plan the work.")
    reviewer = ScriptedReviewer()
    with pytest.raises(ValueError, match="answers review prompts only"):
        asyncio.run(reviewer.complete("You plan.", prompt, "fixture"))
    asyncio.run(reviewer.complete(SUPERVISOR_SYSTEM_PROMPT, prompt, "fixture"))
    assert len(reviewer.calls) == 1


def test_a_supervised_attempt_charges_its_review_calls():
    tasks = fixture_tasks()
    task = next(task for task in tasks if task.anomaly == "empty_output")
    policy = acceptance_policy()
    # The shared runner runs a harness's own arms through its attempt runner.
    routine = run_cell(task, "routine", policy, allowance=1, attempt_runner=run_attempt)
    [attempt] = routine["attempts"]
    supervision = attempt["supervision"]
    assert routine["accepted"] and supervision["calls"] == 2
    assert [review["after_node"] for review in supervision["reviews"]] == ["risks", REVISE]
    assert supervision["tokens"] > 0 and supervision["cost_microusd"] > 0
    assert attempt["cost_microusd"] == attempt["model_cost_microusd"] + supervision["cost_microusd"]
    assert run_task(task, "routine", policy)["supervision_calls"] == 2

    clean = next(task for task in tasks if task.anomaly is None)
    unsupervised = run_task(clean, "none", policy)
    [attempt] = unsupervised["attempts"]
    assert unsupervised["supervision_calls"] == 0 and attempt["supervision"]["cost_microusd"] == 0
    assert attempt["cost_microusd"] == attempt["model_cost_microusd"]
    with pytest.raises(ValueError, match="Unknown arm"):
        run_attempt(clean, "adaptive_selection", policy, attempt=1)


def test_the_balanced_schedule_counterbalances_three_arms():
    task_ids = [f"task-{index}" for index in range(6)]
    schedule = balanced_schedule(task_ids, seed=5)
    assert schedule == balanced_schedule(task_ids, seed=5)
    rows = [[slot["arm"] for slot in schedule if slot["block"] == block] for block in range(6)]
    assert all(sorted(row) == sorted(ARMS) for row in rows)
    for arm in ARMS:
        assert Counter(row.index(arm) for row in rows) == {0: 2, 1: 2, 2: 2}
    follows = Counter((row[i], row[i + 1]) for row in rows for i in range(2))
    assert len(follows) == 6 and set(follows.values()) == {2}
    assert sorted(slot["task_id"] for slot in schedule if slot["position"] == 0) == task_ids
    assert len(balanced_schedule(task_ids[:2], seed=5, repetitions=3)) == 18
    with pytest.raises(ValueError, match="seed"):
        balanced_schedule(task_ids, seed=-1)


def test_a_campaign_without_routine_review_keeps_the_accepted_outcomes():
    # A smaller campaign, in process: two clean tasks and two planted anomalies.
    tasks = fixture_tasks()
    chosen = (*[task for task in tasks if task.anomaly is None][:2],
              *[task for task in tasks if task.anomaly in ("empty_output", "failed_step")])
    campaign = asyncio.run(run_campaign(chosen, acceptance_policy(), seed=1))
    decision = campaign["decision"]
    assert campaign["holdout"]["label"] == "offline stand-in"
    assert campaign["decision_status"]["promotes"] is False
    development = decision["development_scores"][0]
    assert development["viable"] is True
    assert development["secondary_non_regression"] == {"accepted_outcome_rate": True,
                                                        "cost_microusd": True}
    assert decision["confirmation_assessment"]["primary"]["sample_count"] == len(chosen)
    calls = Counter()
    for trial in campaign["trials"]:
        if trial["split"] == "confirmation":
            calls[trial["arm"]] += trial["supervision_calls"]
    assert calls["triggered"] < calls["routine"] and calls["none"] == 0


def test_fixture_tasks_have_distinct_identities_and_known_anomalies():
    tasks = fixture_tasks()
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert len({task.sha256 for task in tasks}) == len(tasks)
    assert Counter(task.anomaly for task in tasks)[None] >= 3
    with pytest.raises(ValueError, match="Unknown anomaly"):
        replace(tasks[0], anomaly="slow_step")
