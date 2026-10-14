"""The offline template evaluation runs fixture versions against a predecessor and one worker.

The harness runs in a child process without provider keys and with this
suite's network guard (``tests/conftest.py``) installed, recording every
socket use the guard refuses. Fixture runs establish mechanics only: each
fixture version is designed for one outcome, so these tests check what the
campaigns decide about it, never a template's quality.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from fakes import child_executable, offline_env

import benchmarks.template_eval as template_eval
from benchmarks.template_eval import (
    PACKS, POLICIES, REQUIRED_GATES, SINGLE_AGENT, SPLITS, TARGET, PackError, PinnedPack,
    TaskPack, TemplateEvaluator, arm_report, campaign_contract, cost_per_accepted_outcome,
    evaluator_manifest, fixture_catalog, holdout_pack, run_campaign, run_pass, run_task,
    split_manifest,
)
from smythe import AcceptancePolicy, TextValidator
from smythe._money import CEILING, MICROUSD, NANOUSD, convert
from smythe.optimize import EvaluatorManifest, ExperimentContract, sha256_prefixed
from smythe.optimize.ledger import holdout_identity

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
HARNESS = ROOT / "benchmarks" / "template_eval.py"
RECORD = ROOT / "benchmarks" / "results" / "template_eval_offline.json"
CANDIDATES = ("improvement", "harmful", "partial", "unexcluded")
VERDICTS = {"improved", "inconclusive", "worse"}

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
    work = tmp_path_factory.mktemp("template-eval")
    out, refused = work / "record.json", work / "refused.json"
    proc = subprocess.run(
        [child_executable(), "-c", GUARDED_RUN, str(TESTS), str(HARNESS), str(refused),
         "--out", str(out)],
        cwd=ROOT, env=offline_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return (json.loads(out.read_text(encoding="utf-8")),
            json.loads(refused.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def catalog_versions(tmp_path_factory):
    """The fixture catalog and each fixture label's template version."""
    return fixture_catalog(tmp_path_factory.mktemp("catalog"))


def _trials(campaign, role, split):
    return [trial for trial in campaign["trials"] if trial["role"] == role
            and trial["split"] == split]


def _accepted(campaign, role, split):
    return {trial["metrics"]["accepted_count"] for trial in _trials(campaign, role, split)}


def test_the_run_makes_no_network_call_and_is_labeled_as_mechanics(offline_run):
    record, refused = offline_run
    assert refused == []
    assert record["benchmark"] == "template-evaluation"
    assert record["evidence_status"] == "offline-mechanics" and record["claimable"] is False
    assert record["api_calls"] == 0 and record["paid_execution_allowed"] is False
    assert record["mode"] == "offline" and record["default_change"] is False
    assert record["promotion"]["eligible"] is False
    assert record["cost_protocol"]["invoice_cost"] is False
    assert set(record["campaigns"]) == set(CANDIDATES)
    for campaign in record["campaigns"].values():
        assert campaign["decision_status"]["activates"] is False


def test_every_campaign_binds_its_evaluator_and_seals_the_single_worker_reference(offline_run):
    record, _ = offline_run
    validators = sorted((item["name"], item["version"], item["config_sha256"])
                        for policy in record["acceptance"].values()
                        for item in policy["final_validators"])
    for campaign in record["campaigns"].values():
        assert campaign["policies"]["reference"] == {TARGET: SINGLE_AGENT}
        assert campaign["roles"]["references"] == [SINGLE_AGENT]
        assert campaign["reference_rule"] == "must_not_be_worse"
        assert campaign["decision"]["references"][0]["candidate_id"] == (
            campaign["candidates"]["reference"])
        manifest = EvaluatorManifest.from_dict(campaign["evaluator_manifest"])
        assert manifest.evaluator_hash == campaign["evaluator_hash"]
        assert manifest.price_snapshot_version == record["cost_protocol"]["price_snapshot_version"]
        assert sorted((item.name, item.version, item.config_sha256)
                      for item in manifest.validators) == validators
        for split in SPLITS:
            bound = campaign["split_manifests"][split]
            assert manifest.split_manifests[split] == sha256_prefixed(bound)
            assert bound["pack_sha256"] == record["packs"][split]["sha256"]
            assert bound["acceptance"] == {family: policy["sha256"]
                                           for family, policy in record["acceptance"].items()}
        contract = ExperimentContract.from_dict(campaign["contract"])
        assert set(contract.required_gates) == set(REQUIRED_GATES)
        assert campaign["holdout_identity"] == holdout_identity(contract, manifest.evaluator_hash)
        for trial in campaign["trials"]:
            assert set(trial["gates"]) == set(REQUIRED_GATES)
            assert trial["pack_sha256"] == record["packs"][trial["split"]]["sha256"]
            tasks = record["packs"][trial["split"]]["tasks"]
            assert trial["metrics"]["attempted_count"] == len(tasks)
            if trial["role"] == "reference":
                assert trial["split"] != "development" and trial["arm"] == SINGLE_AGENT
                assert not any(task["template_built"] for task in trial["tasks"])
    # The held-out family is one development and confirmation never contain.
    families = {split: {task["family"] for task in record["packs"][split]["tasks"]}
                for split in SPLITS}
    assert families["holdout"] - families["development"] - families["confirmation"]


def test_a_fixture_designed_improvement_is_promoted_and_not_worse_than_one_worker(offline_run):
    record, _ = offline_run
    campaign = record["campaigns"]["improvement"]
    decision = campaign["decision"]
    assert decision["promoted"] is True and decision["verdict"] == "improved"
    assert decision["selected_candidate_id"] == campaign["candidates"]["challenger"]
    assert decision["confirmation_assessment"]["promote"] is True
    assert decision["holdout_assessment"]["promote"] is True
    single_worker = decision["references"][0]
    assert single_worker["verdict"] in VERDICTS - {"worse"}
    for stage in ("confirmation", "holdout"):
        assert single_worker[stage]["verdict"] != "worse"
    # It recovers every planted failure, on the held-out family too.
    for split in SPLITS:
        attempted = len(record["packs"][split]["tasks"])
        assert _accepted(campaign, "challenger", split) == {attempted}
        assert max(_accepted(campaign, "incumbent", split)) < attempted
    assert campaign["holdout"]["loaded"] is True


def test_a_harmful_candidate_is_found_worse_on_the_held_out_family_and_not_promoted(offline_run):
    record, _ = offline_run
    campaign = record["campaigns"]["harmful"]
    decision = campaign["decision"]
    blueprint = record["templates"]["harmful"]["record"]["blueprint"]
    assert "timeline" not in {node["id"] for node in blueprint}
    # Development and confirmation briefings never need the section it drops.
    assert decision["development_scores"][0]["viable"] is True
    assert decision["confirmation_assessment"]["promote"] is True
    assert decision["holdout_assessment"]["verdict"] == "worse"
    assert decision["verdict"] == "worse" and decision["promoted"] is False
    assert decision["reason"].startswith("holdout rejected candidate")
    for trial in _trials(campaign, "challenger", "holdout"):
        for task in trial["tasks"]:
            if task["family"] == "incident-review":
                assert task["outcome"] == "rejected"
                assert any("'Timeline' is missing" in reason for reason in task["reasons"])


def test_a_candidate_that_beats_its_predecessor_but_loses_to_one_worker_is_not_promoted(
        offline_run):
    record, _ = offline_run
    campaign = record["campaigns"]["partial"]
    decision = campaign["decision"]
    assert decision["confirmation_assessment"]["promote"] is True
    assert decision["confirmation_assessment"]["verdict"] == "improved"
    single_worker = decision["references"][0]
    assert single_worker["confirmation"]["verdict"] == "worse"
    assert decision["promoted"] is False and decision["holdout_assessment"] is None
    assert decision["reason"] == (
        "confirmation rejected candidate: reference "
        f"{campaign['candidates']['reference']} is better on the primary objective")
    assert max(_accepted(campaign, "incumbent", "confirmation")) < min(
        _accepted(campaign, "challenger", "confirmation"))
    assert max(_accepted(campaign, "challenger", "confirmation")) < min(
        _accepted(campaign, "reference", "confirmation"))
    # A confirmation that fails never reaches, or loads, the holdout.
    assert not any(trial["split"] == "holdout" for trial in campaign["trials"])
    assert campaign["holdout"]["loaded"] is False


def test_a_candidate_applied_to_an_excluded_family_fails_its_applicability_gate(offline_run):
    record, _ = offline_run
    campaign = record["campaigns"]["unexcluded"]
    decision = campaign["decision"]
    assert decision["promoted"] is False
    assert decision["reason"] == "no candidate passed development gates and hard bounds"
    score = decision["development_scores"][0]
    assert score["gates"]["candidate.applicability.excluded_untouched"] is False
    assert score["gates"]["incumbent.applicability.excluded_untouched"] is True
    excluded = record["packs"]["development"]["excluded_families"]
    for role, touched in (("challenger", True), ("incumbent", False)):
        (trial,) = _trials(campaign, role, "development")
        assert trial["gates"]["applicability.excluded_untouched"] is not touched
        assert all(gate for name, gate in trial["gates"].items()
                   if name != "applicability.excluded_untouched")
        for task in trial["tasks"]:
            if task["family"] in excluded:
                assert task["template_built"] is touched
                assert task["outcome"] == ("rejected" if touched else "accepted")


def test_cost_per_accepted_outcome_is_a_ratio_of_sums_and_undefined_without_one(offline_run):
    def trial(cost, accepted):
        return {"metrics": {"attributable_cost_microusd": cost, "accepted_count": accepted}}

    # 40 / 4, not the mean of the per-trial ratios (10 and 10/3).
    assert cost_per_accepted_outcome([trial(10, 1), trial(30, 3)]) == 10.0
    assert cost_per_accepted_outcome([trial(10, 0), trial(30, 3)]) == round(40 / 3, 3)
    assert cost_per_accepted_outcome([trial(10, 0), trial(30, 0)]) == "undefined"
    assert cost_per_accepted_outcome([]) == "undefined"

    record, _ = offline_run
    for campaign in record["campaigns"].values():
        report = campaign["report"]
        assert report["arms"] == arm_report(campaign["trials"])
        # The campaign's own spend is every trial's cost, reported apart.
        assert report["campaign_cost_microusd"]["confirmed_microusd"] == sum(
            trial["metrics"]["attributable_cost_microusd"] for trial in campaign["trials"])
        assert report["campaign_cost_microusd"]["unknown_exposure_microusd"] == 0


def test_a_pass_counts_outcomes_and_charges_declared_usage_rounded_up(
        catalog_versions, monkeypatch):
    catalog, versions = catalog_versions
    pack = PACKS["development"].load()
    predecessor = catalog.get(versions["predecessor"])
    tasks = {task.task_id: run_task(task, predecessor) for task in pack.tasks}
    # A section step that fails is skipped, so its section is missing and acceptance rejects.
    assert {task_id: item["outcome"] for task_id, item in tasks.items()} == {
        "briefing-outage": "accepted", "briefing-vendor": "rejected",
        "briefing-audit": "rejected", "change-retry": "accepted"}
    assert tasks["briefing-vendor"]["node_statuses"] == {"completed": 3, "skipped": 1}
    assert tasks["briefing-vendor"]["template_versions"] == [predecessor.template_version]
    assert tasks["change-retry"]["template_applied"] is False
    assert tasks["change-retry"]["template_versions"] == []
    assert tasks["change-retry"]["selection_calls"] == 0

    trial = run_pass(pack, predecessor, seed=5)
    tokens = sum(item["tokens"] for item in tasks.values())
    assert trial["metrics"] == {
        "accepted_count": 2, "attempted_count": 4, "node_count": 13,
        "attributable_cost_microusd": convert(tokens * template_eval.NANOUSD_PER_TOKEN,
                                              NANOUSD, MICROUSD, CEILING)}
    assert all(trial["gates"].values())
    assert [task["task_id"] for task in trial["tasks"]] == [
        task.task_id for task in template_eval.pass_order(pack.tasks, 5)]

    # A price that leaves a fraction of a microUSD is rounded up, never down.
    monkeypatch.setattr(template_eval, "NANOUSD_PER_TOKEN", 2_501)
    priced = run_pass(pack, predecessor, seed=5)
    assert priced["metrics"]["attributable_cost_microusd"] == -(-tokens * 2_501 // 1_000)
    assert tokens * 2_501 % 1_000 != 0

    single = run_pass(pack, None, seed=5)
    assert single["metrics"]["accepted_count"] == 4 and single["metrics"]["node_count"] == 4
    assert not any(task["template_built"] or task["selection_calls"] for task in single["tasks"])


def test_the_policy_gate_fails_a_pass_judged_by_another_policy(catalog_versions, monkeypatch):
    catalog, versions = catalog_versions
    pack = PACKS["development"].load()
    record = catalog.get(versions["predecessor"])
    run = template_eval.run_task

    def judged_elsewhere(task, arm):
        item = run(task, arm)
        return {**item, "policy_sha256": "0" * 64} if task.family == "briefing" else item

    monkeypatch.setattr(template_eval, "run_task", judged_elsewhere)
    trial = run_pass(pack, record, seed=0)
    assert trial["gates"]["acceptance.policy_unchanged"] is False
    assert trial["gates"]["applicability.excluded_untouched"] is True


def test_held_out_tasks_load_only_in_the_holdout_split(catalog_versions):
    catalog, versions = catalog_versions
    pinned = PACKS["holdout"]
    for label, reaches_holdout in (("improvement", True), ("partial", False)):
        loads: list[int] = []

        def spy():
            loads.append(len(evaluator.trials))
            return holdout_pack()

        evaluator = TemplateEvaluator(catalog, {**PACKS, "holdout": PinnedPack(
            pinned.name, spy, pinned.sha256)})
        # The evaluator identity binds the holdout pack by its pin, without loading it.
        assert evaluator.manifest().evaluator_hash == evaluator_manifest(PACKS).evaluator_hash
        assert loads == []
        campaign = asyncio.run(run_campaign(catalog, versions["predecessor"], versions[label],
                                            evaluator=evaluator))
        splits = [trial["split"] for trial in evaluator.trials]
        if reaches_holdout:
            # Loaded once, as the first holdout trial ran, after every earlier split.
            assert loads == [splits.index("holdout")]
            assert set(splits[:loads[0]]) == {"development", "confirmation"}
            assert campaign["holdout"]["first_holdout_trial"] == loads[0]
        else:
            assert loads == [] and "holdout" not in splits
            assert campaign["holdout"]["loaded"] is False


def test_changing_a_pack_changes_the_evaluator_hash_and_the_holdout_identity(monkeypatch):
    for pinned in PACKS.values():
        assert pinned.load().manifest_sha256 == pinned.sha256
    base = evaluator_manifest(PACKS)
    contract = campaign_contract(("a" * 64, "b" * 64), seed=0)
    original = holdout_pack()
    changed = TaskPack(original.name, (
        *original.tasks[:-1], replace(original.tasks[-1], goal="Summarize another change.")),
        original.excluded_families)

    # A changed pack still under the old pin is refused when it loads.
    with pytest.raises(PackError, match="pinned SHA-256"):
        PinnedPack(original.name, lambda: changed, original.manifest_sha256).load()

    repinned = PinnedPack(original.name, lambda: changed, changed.manifest_sha256)
    assert repinned.load() is changed
    manifest = evaluator_manifest({**PACKS, "holdout": repinned})
    assert manifest.split_manifests["holdout"] != base.split_manifests["holdout"]
    assert manifest.split_manifests["development"] == base.split_manifests["development"]
    assert manifest.evaluator_hash != base.evaluator_hash
    assert holdout_identity(contract, manifest.evaluator_hash) != holdout_identity(
        contract, base.evaluator_hash)

    # So does a changed acceptance policy, through every split and the validators.
    looser = AcceptancePolicy(mode="strict", final_validators=[TextValidator({"min_chars": 1})])
    monkeypatch.setitem(POLICIES, "briefing", looser)
    loosened = evaluator_manifest(PACKS)
    assert split_manifest(PACKS["holdout"])["acceptance"]["briefing"] == looser.sha256
    assert set(loosened.split_manifests.values()).isdisjoint(base.split_manifests.values())
    assert loosened.validators != base.validators
    assert holdout_identity(contract, loosened.evaluator_hash) != holdout_identity(
        contract, base.evaluator_hash)


def test_the_committed_record_is_labeled_and_never_cited_as_a_claim():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["benchmark"] == "template-evaluation"
    assert record["evidence_status"] == "offline-mechanics" and record["claimable"] is False
    assert record["api_calls"] == 0 and record["default_change"] is False
    assert record["promotion"]["eligible"] is False
    assert record["environment"]["source_control"]["dirty"] is False
    assert {split: pack["sha256"] for split, pack in record["packs"].items()} == {
        split: pinned.sha256 for split, pinned in PACKS.items()}
    decisions = {label: campaign["decision"] for label, campaign in record["campaigns"].items()}
    assert {label: (decision["promoted"], decision["verdict"])
            for label, decision in decisions.items()} == {
        "improvement": (True, "improved"), "harmful": (False, "worse"),
        "partial": (False, "improved"), "unexcluded": (False, "inconclusive")}
    for name in ("README.md", "ROADMAP.md"):
        assert "template_eval" not in (ROOT / name).read_text(encoding="utf-8")
