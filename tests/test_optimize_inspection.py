"""Report collection preserves durable evidence and never reruns evaluations.

Target reports sum each campaign's own snapshot cost across every campaign
that sets the target, keeping unknown outcomes apart.
"""

from contextlib import closing
import hashlib
import json
import sqlite3

import pytest

from smythe.optimize.contracts import Candidate, ExperimentContract, MetricObjective, ObjectiveDirection
from smythe.optimize.inspection import collect_optimization_report, collect_target_report
from smythe.optimize.ledger import ExperimentLedger, ExperimentLedgerError, PromotionDecision
from smythe.optimize.registry import PolicyRegistry


PLAN = "sha256:" + "a" * 64
EVALUATOR = "sha256:" + "b" * 64
CAMPAIGN = "report-campaign"


def make_ledger(path, *, states=("completed", "prepared", "dispatched", "unknown"),
                decision=False, cost=137, ceiling=1000, budget=10000, label="recorded policy",
                campaign=CAMPAIGN, field="label", verdict=None):
    """Create only local ledger records, using no evaluator or provider.

    Each call adds one campaign whose candidates set *field*; a recorded
    *verdict* joins the decision's assessment.
    """
    contract = ExperimentContract(
        name="report_fixture", objectives=(MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),),
        mutable_fields=(field,), development_repetitions=1,
        confirmation_repetitions=3, holdout_repetitions=3, max_candidates=2,
        max_parallel_candidates=2, max_trials=20, max_wall_seconds=600,
        max_budget_microusd=budget, per_trial_reservation_microusd=ceiling,
        confidence=.95, min_improvement=.25, base_seed=100,
    )
    incumbent = Candidate(contract=contract, policy={field: label}, hypothesis=label)
    challenger = Candidate(contract=contract, policy={field: "challenger " + label},
                           hypothesis=label, parent=incumbent)
    keys = []
    with ExperimentLedger(path, durability="normal") as ledger:
        ledger.create_campaign(contract, incumbent.candidate_id, (incumbent, challenger),
                               plan_hash=PLAN, campaign_id=campaign)
        lease = ledger.acquire_campaign_lease(campaign, "fixture", ttl_s=3600)
        try:
            for index, state in enumerate(states):
                candidate = (incumbent, challenger)[index % 2]
                role = "incumbent" if index % 2 == 0 else "challenger"
                key = ledger.prepare_trial(campaign, candidate.candidate_id, "development",
                                           role + ".plan_" + "a" * 64, 100 + index // 2,
                                           lease=lease, evaluator_hash=EVALUATOR, ceiling_microusd=ceiling)
                keys.append(key)
                if state != "prepared":
                    ledger.claim_trial_dispatch(key, lease=lease)
            for key, state in zip(keys, states):
                if state == "completed":
                    ledger.complete_trial(key, lease=lease, metrics={"quality": 9.25}, gates={"shape": True},
                                          actual_cost_microusd=cost, duration_ms=2.5)
                elif state == "unknown":
                    ledger.mark_trial_unknown(key, "outcome not recorded", lease=lease)
            if decision:
                ledger.append_decision(PromotionDecision(
                    campaign_id=campaign, candidate_id=challenger.candidate_id,
                    promoted=False, reason="No challenger passed development gates",
                    trial_keys=tuple(keys), assessment={
                        "optimization_plan_hash": PLAN,
                        "holdout_seed_commitment": ledger.get_holdout_seed_commitment(campaign),
                        "stage": "development", "development_scores": [],
                        **({} if verdict is None else {"verdict": verdict}),
                    },
                ), lease=lease)
        finally:
            ledger.release_campaign_lease(lease)
    return keys


def mutate(path, sql, values=()):
    with closing(sqlite3.connect(path)) as db, db:
        # The marker excludes older binaries, not a raw-SQL operator fixture.
        db.create_function("smythe_autotune_writer_version", 0, lambda: 4)
        db.execute(sql, values)


def logical_dump(path):
    with closing(sqlite3.connect(path)) as db:
        return tuple(db.iterdump())


def test_states_exact_costs_nulls_and_whole_campaign_totals_survive_truncation(tmp_path):
    path = tmp_path / "report.db"
    make_ledger(path)
    before = path.read_bytes(), logical_dump(path)
    with ExperimentLedger(path, read_only=True) as ledger:
        report = collect_optimization_report(ledger, CAMPAIGN, trial_limit=2)
        all_details = collect_optimization_report(ledger, CAMPAIGN)
        assert not ledger._closed
    snapshot = report["ledger_snapshot"]
    assert snapshot["cost"] == {"confirmed_microusd": 137, "reserved_microusd": 2000,
                                "unknown_exposure_microusd": 1000, "total_exposure_microusd": 3137}
    assert snapshot["trial_counts"] == {"prepared": 1, "dispatched": 1, "completed": 1, "unknown": 1}
    assert report["trial_detail"] == {"limit": 2, "total": 4, "returned": 2, "truncated": True}
    assert report["evaluator_hashes"] == [EVALUATOR]
    for trial in all_details["trials"]:
        assert "seed" not in trial
        if trial["status"] != "completed":
            assert trial["actual_cost_microusd"] is None and trial["duration_ms"] is None
    assert (path.read_bytes(), logical_dump(path)) == before


def test_large_aggregate_is_not_restricted_to_signed_sqlite_integer(tmp_path):
    path = tmp_path / "large.db"
    make_ledger(path, states=("completed",) * 3, cost=9_000_000_000_000_000_000,
                ceiling=1_000_000_000_000_000_000, budget=9_000_000_000_000_000_000)
    with ExperimentLedger(path, read_only=True) as ledger:
        report = collect_optimization_report(ledger, CAMPAIGN)
    assert report["ledger_snapshot"]["cost"]["confirmed_microusd"] == 27_000_000_000_000_000_000


def test_canonical_hash_detachment_and_hostile_text_preserve_recorded_values(tmp_path):
    path = tmp_path / "text.db"
    label = '</script><svg onload="alert(1)">Ω\u202e'
    make_ledger(path, states=("completed", "completed"), decision=True, label=label)
    with ExperimentLedger(path, read_only=True) as ledger:
        report = collect_optimization_report(ledger, CAMPAIGN)
        assert report == collect_optimization_report(ledger, CAMPAIGN)
        payload = {key: value for key, value in report.items() if key != "evidence_sha256"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert report["evidence_sha256"] == expected
        assert len(expected) == 64 and not expected.startswith("sha256:")
        assert report["ledger_snapshot"]["decisions"][0]["assessment"]["development_scores"] == []
        report["ledger_snapshot"]["candidates"][0]["candidate"]["policy"]["label"] = "changed"
        fresh = collect_optimization_report(ledger, CAMPAIGN)
        assert label in [row["candidate"]["policy"]["label"] for row in fresh["ledger_snapshot"]["candidates"]]
        assert "changed" not in json.dumps(fresh)


def test_report_uses_canonical_campaign_identity(tmp_path):
    path = tmp_path / "identity.db"
    make_ledger(path, states=())
    with ExperimentLedger(path, read_only=True) as ledger:
        report = collect_optimization_report(ledger, "  " + CAMPAIGN + "  ")
    assert report["campaign_id"] == report["ledger_snapshot"]["campaign"]["campaign_id"] == CAMPAIGN


@pytest.mark.parametrize("limit", [None, True, False, 0, -1, 1001, 1.0, "2"])
def test_invalid_limit_is_caller_error_before_any_read(limit):
    class NoRead:
        read_only = True

        def validate_decision_inventory(self, *_):
            pytest.fail("invalid argument reached the ledger")

    with pytest.raises(ValueError, match="trial_limit"):
        collect_optimization_report(NoRead(), CAMPAIGN, trial_limit=limit)


@pytest.mark.parametrize("campaign", [None, "", "../escape", "a" * 129])
def test_invalid_campaign_is_caller_error(campaign):
    class NoRead:
        read_only = True

    with pytest.raises(ValueError, match="campaign_id"):
        collect_optimization_report(NoRead(), campaign)


def test_writable_ledger_is_refused_without_acquiring_a_lease(tmp_path):
    path = tmp_path / "writable.db"
    make_ledger(path, states=())
    with ExperimentLedger(path) as ledger:
        with pytest.raises(ValueError, match="read-only"):
            collect_optimization_report(ledger, CAMPAIGN)
        assert ledger.get_campaign_lease(CAMPAIGN) is None


@pytest.mark.parametrize("kind", ["decision-json", "decision-root", "decision-identity", "decision-type",
                                  "contract", "trial-cost", "trial-metrics", "trial-blob"])
def test_corrupt_saved_evidence_is_a_ledger_error_not_a_user_argument(tmp_path, kind):
    path = tmp_path / "corrupt.db"
    make_ledger(path, states=("completed", "completed"), decision=True)
    with closing(sqlite3.connect(path)) as db:
        decision = json.loads(db.execute("SELECT payload_json FROM promotion_decisions").fetchone()[0])
        completion = json.loads(db.execute("SELECT payload_json FROM trial_events WHERE event_type='completed' LIMIT 1")
                                .fetchone()[0])
    if kind == "decision-json":
        mutate(path, "UPDATE promotion_decisions SET payload_json='{' ")
    elif kind == "decision-root":
        mutate(path, "UPDATE promotion_decisions SET payload_json='[]'")
    elif kind == "decision-identity":
        mutate(path, "UPDATE promotion_decisions SET decision_id='wrong'")
    elif kind == "decision-type":
        decision["promoted"] = "yes"
        mutate(path, "UPDATE promotion_decisions SET payload_json=?",
               (json.dumps(decision, sort_keys=True, separators=(",", ":")),))
    elif kind == "contract":
        mutate(path, "UPDATE campaigns SET contract_json='null'")
    elif kind == "trial-blob":
        mutate(path, "UPDATE trial_events SET payload_json=? WHERE event_type='completed'", (b"not-json",))
    else:
        completion["actual_cost_microusd" if kind == "trial-cost" else "metrics"] = "invalid"
        mutate(path, "UPDATE trial_events SET payload_json=? WHERE event_type='completed'", (json.dumps(completion),))
    before = path.read_bytes()
    with ExperimentLedger(path, read_only=True) as ledger:
        with pytest.raises(ExperimentLedgerError):
            collect_optimization_report(ledger, CAMPAIGN)
    assert path.read_bytes() == before


def test_nonfinite_arbitrary_assessment_is_refused_without_rewriting(tmp_path, monkeypatch):
    path = tmp_path / "nonfinite.db"
    make_ledger(path, states=())
    with ExperimentLedger(path, read_only=True) as ledger:
        original = ledger.snapshot

        def invalid(campaign):
            value = original(campaign)
            value["extra"] = float("nan")
            return value

        monkeypatch.setattr(ledger, "snapshot", invalid)
        with pytest.raises(ExperimentLedgerError) as caught:
            collect_optimization_report(ledger, CAMPAIGN)
        assert isinstance(caught.value.__cause__, ValueError)


def _legacy_comparison(seeds):
    # Exact comparison shape written before the paired t promotion rule: the
    # percentile bootstrap interval was the promotion interval.
    return {"objective_name": "quality", "direction": "maximize", "sample_count": len(seeds),
            "sample_seeds": list(seeds), "baseline_mean": 9.25, "candidate_mean": 10.25,
            "mean_improvement": 1.0, "confidence_level": 0.95, "bootstrap_resamples": 2000,
            "bootstrap_seed": 42, "confidence_interval": [1.0, 1.0], "lower_confidence_bound": 1.0,
            "hard_bounds_passed": True, "non_regression_passed": True}


def _legacy_assessment(seeds):
    return {"promote": True, "primary": _legacy_comparison(seeds), "secondary": [],
            "gates": {"candidate.shape": True, "incumbent.shape": True}, "all_gates_passed": True,
            "hard_bounds_passed": True, "secondary_non_regression_passed": True,
            "min_improvement": 0.25, "reasons": []}


def test_decisions_recorded_under_the_bootstrap_rule_still_validate_inspect_and_report(tmp_path):
    from smythe.optimize.report import render_optimization_report

    path = tmp_path / "legacy.db"
    contract = ExperimentContract(
        name="legacy_fixture", objectives=(MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),),
        mutable_fields=("label",), development_repetitions=1,
        confirmation_repetitions=3, holdout_repetitions=3, max_candidates=2,
        max_parallel_candidates=2, max_trials=14, max_wall_seconds=600,
        max_budget_microusd=14_000, per_trial_reservation_microusd=1000,
        confidence=.95, min_improvement=.25, base_seed=100,
    )
    incumbent = Candidate(contract=contract, policy={"label": "current"}, hypothesis="current")
    challenger = Candidate(contract=contract, policy={"label": "new"}, hypothesis="new", parent=incumbent)
    plan = {"development": (100,), "confirmation": (201, 202, 203), "holdout": (301, 302, 303)}
    keys = []
    with ExperimentLedger(path, durability="normal") as ledger:
        ledger.create_campaign(contract, incumbent.candidate_id, (incumbent, challenger),
                               plan_hash=PLAN, campaign_id=CAMPAIGN)
        lease = ledger.acquire_campaign_lease(CAMPAIGN, "legacy-writer", ttl_s=3600)
        for split, seeds in plan.items():
            for candidate, role in ((incumbent, "incumbent"), (challenger, "challenger")):
                for seed in seeds:
                    key = ledger.prepare_trial(CAMPAIGN, candidate.candidate_id, split,
                                               role + ".plan_" + "a" * 64, seed, lease=lease,
                                               evaluator_hash=EVALUATOR, ceiling_microusd=1000)
                    ledger.claim_trial_dispatch(key, lease=lease)
                    ledger.complete_trial(key, lease=lease, metrics={"quality": 9.25 + (role == "challenger")},
                                          gates={"shape": True}, actual_cost_microusd=0, duration_ms=1)
                    keys.append(key)
        ledger.append_decision(PromotionDecision(
            campaign_id=CAMPAIGN, candidate_id=challenger.candidate_id, promoted=True,
            reason="confirmation and untouched holdout both passed promotion policy",
            trial_keys=tuple(keys), assessment={
                "optimization_plan_hash": PLAN, "runner_version": 3, "ledger_durability": "normal",
                "holdout_seed_commitment": ledger.get_holdout_seed_commitment(CAMPAIGN),
                "development": {"candidate_id": challenger.candidate_id, "viable": True},
                "confirmation": _legacy_assessment(plan["confirmation"]),
                "holdout": _legacy_assessment(plan["holdout"]),
            },
        ), lease=lease)
        ledger.release_campaign_lease(lease)

    with ExperimentLedger(path, read_only=True) as ledger:
        assert len(ledger.validate_decision_inventory(CAMPAIGN)) == 1
        report = collect_optimization_report(ledger, CAMPAIGN)
    decision = report["ledger_snapshot"]["decisions"][0]
    assert decision["promoted"] is True
    assert "method" not in decision["assessment"]["holdout"]["primary"]
    html = render_optimization_report(report)
    assert html.count("Percentile bootstrap lower bound (earlier rule") == 2
    assert "Paired Student-t" not in html


# Campaign listing and target reports

TARGET = "label"
COST_FIELDS = ("confirmed_microusd", "reserved_microusd", "unknown_exposure_microusd",
               "total_exposure_microusd")


def make_campaigns(path):
    """Three campaigns that set the target and one that sets another field."""
    make_ledger(path, campaign="first")
    make_ledger(path, campaign="second", states=("completed", "completed"), decision=True, cost=250,
                verdict="inconclusive")
    # An unknown outcome under a zero ceiling holds no micro-USD but is still unresolved.
    make_ledger(path, campaign="zero-ceiling", states=("unknown",), ceiling=0, budget=0)
    make_ledger(path, campaign="elsewhere", states=("completed", "unknown"), field="other", cost=999)


def empty_registry(path):
    PolicyRegistry(path).close()
    return path


def target_report(path, registry_path, target=TARGET):
    with ExperimentLedger(path, read_only=True) as ledger, \
            PolicyRegistry(registry_path, read_only=True) as registry:
        return collect_target_report(ledger, registry, target)


def test_list_campaigns_returns_each_campaign_oldest_first_without_writing(tmp_path):
    path = tmp_path / "campaigns.db"
    with ExperimentLedger(path) as ledger:
        assert ledger.list_campaigns() == []
    make_campaigns(path)
    before = path.read_bytes(), logical_dump(path)
    with ExperimentLedger(path, read_only=True) as ledger:
        listed = ledger.list_campaigns()
        assert listed == [ledger.open_campaign(item["campaign_id"]) for item in listed]
    assert {item["campaign_id"] for item in listed} == {"first", "second", "zero-ceiling", "elsewhere"}
    order = [(item["created_at_ns"], item["campaign_id"]) for item in listed]
    assert order == sorted(order)
    # Only the public commitment is listed, never the holdout secret.
    assert all(set(item) == set(listed[0]) and "holdout_nonce" not in item for item in listed)
    assert (path.read_bytes(), logical_dump(path)) == before


def test_list_campaigns_refuses_a_campaign_whose_sealed_inventory_drifted(tmp_path):
    path = tmp_path / "campaigns.db"
    make_campaigns(path)
    mutate(path, "UPDATE candidates SET candidate_json = replace(candidate_json, 'challenger', 'other') "
                 "WHERE campaign_id = 'second'")
    with ExperimentLedger(path, read_only=True) as ledger:
        with pytest.raises(ExperimentLedgerError, match="differs from its sealed inventory"):
            ledger.list_campaigns()


def test_target_cost_is_the_exact_sum_of_its_campaigns_snapshot_costs(tmp_path):
    path, registry_path = tmp_path / "campaigns.db", empty_registry(tmp_path / "registry.db")
    make_campaigns(path)
    before = path.read_bytes(), registry_path.read_bytes()
    report = target_report(path, registry_path)
    with ExperimentLedger(path, read_only=True) as ledger:
        snapshots = {name: ledger.snapshot(name) for name in ("first", "second", "zero-ceiling")}
    assert (path.read_bytes(), registry_path.read_bytes()) == before

    campaigns = {item["campaign_id"]: item for item in report["campaigns"]}
    # 'elsewhere' sets another field, so neither it nor its cost is in the report.
    assert set(campaigns) == set(snapshots)
    for name, snapshot in snapshots.items():
        assert campaigns[name]["cost"] == snapshot["cost"]
        assert campaigns[name]["unknown_trials"] == snapshot["trial_counts"].get("unknown", 0)
        assert campaigns[name]["trial_counts"] == snapshot["trial_counts"]
    totals = report["optimization_cost"]
    for field in COST_FIELDS:
        assert totals[field] == sum(snapshot["cost"][field] for snapshot in snapshots.values())
    assert totals == {"campaigns": 3, "confirmed_microusd": 137 + 2 * 250, "reserved_microusd": 2000,
                      "unknown_exposure_microusd": 1000, "total_exposure_microusd": 3637,
                      "unknown_trials": 2}
    # The zero-ceiling unknown outcome adds nothing to the micro-USD but stays counted.
    assert campaigns["zero-ceiling"]["cost"]["unknown_exposure_microusd"] == 0
    assert campaigns["zero-ceiling"]["unknown_trials"] == 1

    assert campaigns["first"]["decision"] is None
    decision = campaigns["second"]["decision"]
    recorded = snapshots["second"]["decisions"][0]
    assert decision == {
        "decision_id": recorded["decision_id"], "candidate_id": recorded["candidate_id"],
        "policy_hash": next(item["policy_hash"] for item in snapshots["second"]["candidates"]
                            if item["candidate_id"] == recorded["candidate_id"]),
        "promoted": False, "verdict": "inconclusive", "reason": recorded["reason"],
        "created_at_ns": recorded["created_at_ns"],
    }
    assert campaigns["second"]["plan_hash"] == PLAN
    assert campaigns["second"]["holdout_commitment"] == (
        snapshots["second"]["campaign"]["holdout_seed_commitment"])
    assert all(item["reviews"] == [] for item in campaigns.values())
    assert (report["target"], report["active"], report["history"], report["campaigns_not_in_ledger"]) == (
        TARGET, None, [], [])
    payload = {key: value for key, value in report.items() if key != "evidence_sha256"}
    assert report["evidence_sha256"] == hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()
    assert report == target_report(path, registry_path)


def test_target_cost_across_campaigns_is_not_bounded_by_a_sqlite_integer(tmp_path):
    path, registry_path = tmp_path / "large.db", empty_registry(tmp_path / "registry.db")
    for campaign in ("first", "second"):
        make_ledger(path, campaign=campaign, states=("completed",) * 3, cost=9_000_000_000_000_000_000,
                    ceiling=1_000_000_000_000_000_000, budget=9_000_000_000_000_000_000)
    totals = target_report(path, registry_path)["optimization_cost"]
    assert totals["confirmed_microusd"] == 54_000_000_000_000_000_000
    assert totals["total_exposure_microusd"] == 54_000_000_000_000_000_000


def test_a_target_touched_by_no_campaign_reports_none_and_no_cost(tmp_path):
    path, registry_path = tmp_path / "campaigns.db", empty_registry(tmp_path / "registry.db")
    make_campaigns(path)
    # A path above or beside the policy's field is not the target or under it.
    for target in ("labels", "label.inner", "unknown.target"):
        report = target_report(path, registry_path, target)
        assert report["campaigns"] == []
        assert report["optimization_cost"] == {"campaigns": 0, **dict.fromkeys(COST_FIELDS, 0),
                                               "unknown_trials": 0}


def test_an_unrecorded_verdict_is_none_and_an_invalid_one_stops_the_report(tmp_path):
    registry_path = empty_registry(tmp_path / "registry.db")
    older = tmp_path / "older.db"
    make_ledger(older, states=("completed", "completed"), decision=True)
    (campaign,) = target_report(older, registry_path)["campaigns"]
    assert campaign["decision"]["verdict"] is None
    invalid = tmp_path / "invalid.db"
    make_ledger(invalid, states=("completed", "completed"), decision=True, verdict="better")
    before = invalid.read_bytes()
    with pytest.raises(ExperimentLedgerError, match="Invalid retained Autotune report evidence") as caught:
        target_report(invalid, registry_path)
    assert "verdict" in str(caught.value.__cause__)
    assert invalid.read_bytes() == before


def test_a_target_report_needs_read_only_stores_and_a_dotted_target(tmp_path):
    path, registry_path = tmp_path / "campaigns.db", empty_registry(tmp_path / "registry.db")
    make_ledger(path)
    with ExperimentLedger(path, read_only=True) as reader, PolicyRegistry(registry_path) as writer:
        with pytest.raises(ValueError, match="read-only policy registry"):
            collect_target_report(reader, writer, TARGET)
    with ExperimentLedger(path) as writer, PolicyRegistry(registry_path, read_only=True) as reader:
        with pytest.raises(ValueError, match="read-only ledger"):
            collect_target_report(writer, reader, TARGET)
    with ExperimentLedger(path, read_only=True) as ledger, \
            PolicyRegistry(registry_path, read_only=True) as registry:
        for target in ("", "../label", "label.", None):
            with pytest.raises(ValueError, match="target must be a dotted policy name"):
                collect_target_report(ledger, registry, target)


def test_a_corrupt_campaign_that_touches_the_target_is_a_ledger_error(tmp_path):
    path, registry_path = tmp_path / "campaigns.db", empty_registry(tmp_path / "registry.db")
    make_campaigns(path)
    mutate(path, "UPDATE promotion_decisions SET decision_id = 'wrong'")
    before = path.read_bytes()
    with pytest.raises(ExperimentLedgerError, match="does not match its payload"):
        target_report(path, registry_path)
    assert path.read_bytes() == before
