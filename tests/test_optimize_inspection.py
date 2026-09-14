"""Report collection preserves durable evidence and never reruns evaluations."""

from contextlib import closing
import hashlib
import json
import sqlite3

import pytest

from smythe.optimize.contracts import Candidate, ExperimentContract, MetricObjective, ObjectiveDirection
from smythe.optimize.inspection import collect_optimization_report
from smythe.optimize.ledger import ExperimentLedger, ExperimentLedgerError, PromotionDecision


PLAN = "sha256:" + "a" * 64
EVALUATOR = "sha256:" + "b" * 64
CAMPAIGN = "report-campaign"


def make_ledger(path, *, states=("completed", "prepared", "dispatched", "unknown"),
                decision=False, cost=137, ceiling=1000, budget=10000, label="recorded policy"):
    """Create only local ledger records, using no evaluator or provider."""
    contract = ExperimentContract(
        name="report_fixture", objectives=(MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),),
        mutable_fields=("label",), development_repetitions=1,
        confirmation_repetitions=3, holdout_repetitions=3, max_candidates=2,
        max_parallel_candidates=2, max_trials=20, max_wall_seconds=600,
        max_budget_microusd=budget, per_trial_reservation_microusd=ceiling,
        confidence=.95, min_improvement=.25, base_seed=100,
    )
    incumbent = Candidate(contract=contract, policy={"label": label}, hypothesis=label)
    challenger = Candidate(contract=contract, policy={"label": "challenger " + label},
                           hypothesis=label, parent=incumbent)
    keys = []
    with ExperimentLedger(path, durability="normal") as ledger:
        ledger.create_campaign(contract, incumbent.candidate_id, (incumbent, challenger),
                               plan_hash=PLAN, campaign_id=CAMPAIGN)
        lease = ledger.acquire_campaign_lease(CAMPAIGN, "fixture", ttl_s=3600)
        try:
            for index, state in enumerate(states):
                candidate = (incumbent, challenger)[index % 2]
                role = "incumbent" if index % 2 == 0 else "challenger"
                key = ledger.prepare_trial(CAMPAIGN, candidate.candidate_id, "development",
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
                    campaign_id=CAMPAIGN, candidate_id=challenger.candidate_id,
                    promoted=False, reason="No challenger passed development gates",
                    trial_keys=tuple(keys), assessment={
                        "optimization_plan_hash": PLAN,
                        "holdout_seed_commitment": ledger.get_holdout_seed_commitment(CAMPAIGN),
                        "stage": "development", "development_scores": [],
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
