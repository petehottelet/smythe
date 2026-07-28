"""Durability, idempotency, and ambiguity tests for the Autotune ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import sqlite3
from threading import Barrier

import pytest

from smythe.optimize.contracts import (
    Candidate,
    ExperimentContract,
    MetricObjective,
    ObjectiveDirection,
)
from smythe.optimize.ledger import (
    ExperimentLedger,
    ExperimentLedgerError,
    LedgerBudgetError,
    LedgerConflictError,
    PromotionDecision,
    TrialStateError,
    TrialStatus,
    UnknownTrialError,
)


PLAN_HASH = "sha256:" + "a" * 64


def _contract(
    *,
    name: str = "ledger_test",
    max_candidates: int = 3,
    max_trials: int = 10,
    budget: int = 10_000,
) -> ExperimentContract:
    return ExperimentContract(
        name=name,
        objectives=(
            MetricObjective(
                name="quality",
                direction=ObjectiveDirection.MAXIMIZE,
                primary=True,
                hard_min=7,
            ),
        ),
        mutable_fields=("prompt.logo",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=max_candidates,
        max_parallel_candidates=1,
        max_trials=max_trials,
        max_wall_seconds=600,
        max_budget_microusd=budget,
        per_trial_reservation_microusd=1_000,
        confidence=0.95,
        min_improvement=0.25,
        base_seed=100,
    )


def _candidate(contract: ExperimentContract, value: str = "simplify") -> Candidate:
    return Candidate(
        contract=contract,
        policy={"prompt.logo": value},
        hypothesis=f"The {value} policy improves fidelity",
    )


def _seal(
    ledger: ExperimentLedger,
    contract: ExperimentContract,
    *candidates: Candidate,
    campaign_id: str | None = None,
    plan_hash: str = PLAN_HASH,
) -> str:
    assert candidates
    return ledger.create_campaign(
        contract,
        candidates[0].candidate_id,
        candidates,
        plan_hash=plan_hash,
        campaign_id=campaign_id,
    )


def _prepared(
    ledger: ExperimentLedger,
    campaign_id: str,
    candidate: Candidate,
    *,
    seed: int = 100,
    split: str = "development",
    phase: str = "evaluate",
) -> str:
    return ledger.prepare_trial(
        campaign_id,
        candidate.candidate_id,
        split,
        phase,
        seed,
        evaluator_hash="sha256:" + "e" * 64,
        ceiling_microusd=1_000,
    )


def test_campaign_is_hash_bound_idempotent_and_resume_safe(tmp_path):
    path = tmp_path / "autotune.db"
    contract = _contract()
    incumbent = _candidate(contract, "incumbent")
    challenger = _candidate(contract, "challenger")
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(
            ledger, contract, incumbent, challenger, campaign_id="campaign-one"
        )
        assert (
            _seal(ledger, contract, incumbent, challenger, campaign_id="campaign-one")
            == campaign_id
        )
        opened = ledger.open_campaign(
            campaign_id,
            contract=contract,
            incumbent_candidate_id=incumbent.candidate_id,
            plan_hash=PLAN_HASH,
            candidates=(incumbent, challenger),
        )
        assert opened["contract_hash"] == contract.contract_hash
        assert opened["plan_hash"] == PLAN_HASH
        assert opened["ordered_candidate_ids"] == [
            incumbent.candidate_id,
            challenger.candidate_id,
        ]

        with pytest.raises(LedgerConflictError, match="different binding"):
            other_contract = _contract(name="other_contract")
            _seal(
                ledger,
                other_contract,
                _candidate(other_contract, "incumbent"),
                campaign_id=campaign_id,
            )

    with ExperimentLedger(path) as reopened:
        assert (
            reopened.open_campaign(campaign_id)["incumbent_candidate_id"]
            == incumbent.candidate_id
        )


def test_holdout_seed_material_is_random_durable_committed_and_narrow(tmp_path):
    path = tmp_path / "autotune.db"
    contract = _contract()
    incumbent = _candidate(contract, "incumbent")
    with ExperimentLedger(path) as ledger:
        first = _seal(ledger, contract, incumbent, campaign_id="campaign-one")
        second = _seal(ledger, contract, incumbent, campaign_id="campaign-two")
        commitment = ledger.get_holdout_seed_commitment(first)
        second_commitment = ledger.get_holdout_seed_commitment(second)
        opened = ledger.open_campaign(first)
        snapshot = ledger.snapshot(first)

        assert not hasattr(ledger, "get_holdout_seed_material")
        with pytest.raises(PermissionError, match="engine-private"):
            ledger._get_holdout_seed_binding(
                first,
                plan_hash=PLAN_HASH,
                candidates=(incumbent,),
                capability=object(),
            )
        assert commitment != second_commitment
        assert commitment.startswith("sha256:") and len(commitment) == 71
        assert opened["holdout_seed_commitment"] == commitment
        assert "holdout_nonce" not in opened
        assert "holdout_seed_material" not in opened
        assert "holdout_nonce" not in snapshot["campaign"]
        assert "holdout_seed_material" not in snapshot["campaign"]

    with ExperimentLedger(path) as reopened:
        assert reopened.get_holdout_seed_commitment(first) == commitment


def test_concurrent_idempotent_campaign_creation_commits_one_holdout_nonce(tmp_path):
    path = tmp_path / "autotune.db"
    contract = _contract()
    incumbent = _candidate(contract, "incumbent")
    with ExperimentLedger(path):
        pass
    barrier = Barrier(2)

    def create() -> str:
        with ExperimentLedger(path) as ledger:
            barrier.wait()
            campaign_id = _seal(
                ledger, contract, incumbent, campaign_id="shared-campaign"
            )
            return ledger.get_holdout_seed_commitment(campaign_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        materials = list(pool.map(lambda _: create(), range(2)))

    assert len(materials[0]) == 71
    assert materials[0] == materials[1]


def test_candidate_registration_is_idempotent_and_contract_bound(tmp_path):
    contract = _contract()
    other_contract = _contract(name="other_contract")
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)

    assert ledger.register_candidate(campaign_id, candidate) == candidate.candidate_id
    assert ledger.register_candidate(campaign_id, candidate) == candidate.candidate_id
    with pytest.raises(LedgerConflictError, match="different contract"):
        ledger.register_candidate(campaign_id, _candidate(other_contract))

    conflicting = _candidate(contract)
    object.__setattr__(conflicting, "policy_hash", "sha256:" + "f" * 64)
    with pytest.raises(LedgerConflictError, match="different payload"):
        ledger.register_candidate(campaign_id, conflicting)


def test_trial_lifecycle_is_idempotent_and_snapshot_separates_cost_states(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)

    trial_key = _prepared(ledger, campaign_id, candidate)
    assert _prepared(ledger, campaign_id, candidate) == trial_key
    prepared_snapshot = ledger.snapshot(campaign_id)
    assert prepared_snapshot["trial_counts"] == {"prepared": 1}
    assert prepared_snapshot["cost"] == {
        "confirmed_microusd": 0,
        "reserved_microusd": 1_000,
        "unknown_exposure_microusd": 0,
        "total_exposure_microusd": 1_000,
    }

    assert ledger.mark_trial_dispatched(trial_key).status is TrialStatus.DISPATCHED
    assert ledger.mark_trial_dispatched(trial_key).status is TrialStatus.DISPATCHED
    completed = ledger.complete_trial(
        trial_key,
        metrics={"quality": 8.5},
        gates={"format": True},
        actual_cost_microusd=625,
        duration_ms=12.5,
        artifact_hashes=("sha256:" + "a" * 64,),
        evaluator_hash="sha256:" + "e" * 64,
    )
    assert completed.status is TrialStatus.COMPLETED
    assert completed.metrics == {"quality": 8.5}
    assert ledger.complete_trial(
        trial_key,
        metrics={"quality": 8.5},
        gates={"format": True},
        actual_cost_microusd=625,
        duration_ms=12.5,
        artifact_hashes=("sha256:" + "a" * 64,),
    ).trial_key == trial_key

    snapshot = ledger.snapshot(campaign_id)
    assert snapshot["spent_microusd"] == 625
    assert snapshot["cost"]["confirmed_microusd"] == 625
    assert snapshot["cost"]["reserved_microusd"] == 0
    assert snapshot["trial_counts"] == {"completed": 1}
    assert ledger.get_trial(trial_key) == ledger.list_trials(campaign_id)[0]


def test_dispatch_claim_is_atomic_across_ledger_connections(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(ledger, contract, candidate)
        ledger.register_candidate(campaign_id, candidate)
        trial_key = _prepared(ledger, campaign_id, candidate)

    barrier = Barrier(2)

    def claim() -> str:
        with ExperimentLedger(path) as ledger:
            barrier.wait()
            try:
                return ledger.claim_trial_dispatch(trial_key).status.value
            except TrialStateError:
                return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))

    assert sorted(results) == ["dispatched", "rejected"]
    with ExperimentLedger(path) as ledger:
        assert ledger.get_trial(trial_key).status is TrialStatus.DISPATCHED


def test_list_trials_materializes_every_row_before_nested_event_queries(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(ledger, contract, candidate)
        ledger.register_candidate(campaign_id, candidate)
        for seed in (100, 101, 102):
            trial_key = _prepared(ledger, campaign_id, candidate, seed=seed)
            ledger.claim_trial_dispatch(trial_key)
            ledger.complete_trial(
                trial_key,
                metrics={"quality": 8},
                gates={"format": True},
                actual_cost_microusd=0,
                duration_ms=1,
            )

        assert len(ledger.list_trials(campaign_id)) == 3
        assert ledger.snapshot(campaign_id)["trial_counts"] == {"completed": 3}


def test_read_only_connection_does_not_create_sqlite_sidecars(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(ledger, contract, candidate)
        ledger.register_candidate(campaign_id, candidate)
        _prepared(ledger, campaign_id, candidate)

    before_files = sorted(item.name for item in tmp_path.iterdir())
    before_mtime = path.stat().st_mtime_ns
    with ExperimentLedger(path, read_only=True) as ledger:
        assert ledger.snapshot(campaign_id)["trial_count"] == 1
    assert sorted(item.name for item in tmp_path.iterdir()) == before_files
    assert path.stat().st_mtime_ns == before_mtime


def test_read_only_connection_fails_closed_while_wal_writer_is_active(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as writer:
        campaign_id = _seal(writer, contract, candidate)
        assert path.with_name(f"{path.name}-wal").exists()
        with pytest.raises(ExperimentLedgerError, match="closed, checkpointed"):
            ExperimentLedger(path, read_only=True)

    with ExperimentLedger(path, read_only=True) as reader:
        assert reader.open_campaign(campaign_id)["campaign_id"] == campaign_id


def test_read_only_rechecks_sidecars_after_open_and_closes_on_race(
    tmp_path, monkeypatch
):
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        contract = _contract()
        _seal(ledger, contract, _candidate(contract))

    original_connect = sqlite3.connect
    opened_connections = []
    wal_path = path.with_name(f"{path.name}-wal")

    def racing_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        opened_connections.append(connection)
        wal_path.write_bytes(b"simulated writer race")
        return connection

    monkeypatch.setattr(sqlite3, "connect", racing_connect)
    with pytest.raises(ExperimentLedgerError, match="sidecar"):
        ExperimentLedger(path, read_only=True)
    assert len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened_connections[0].execute("SELECT 1")
    wal_path.unlink()


def test_read_only_rechecks_sidecars_when_closed(tmp_path):
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        contract = _contract()
        _seal(ledger, contract, _candidate(contract))

    reader = ExperimentLedger(path, read_only=True)
    wal_path = path.with_name(f"{path.name}-wal")
    wal_path.write_bytes(b"simulated overlapping writer")
    with pytest.raises(ExperimentLedgerError, match="after closing"):
        reader.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        reader._connection.execute("SELECT 1")
    wal_path.unlink()


def test_durability_mode_is_explicit_and_defaults_to_full(tmp_path):
    with ExperimentLedger(tmp_path / "full.db") as ledger:
        assert ledger.durability == "full"
        assert ledger._connection.execute("PRAGMA synchronous").fetchone()[0] == 2

    with ExperimentLedger(tmp_path / "normal.db", durability="normal") as ledger:
        assert ledger.durability == "normal"
        assert ledger._connection.execute("PRAGMA synchronous").fetchone()[0] == 1

    with pytest.raises(ValueError, match="durability"):
        ExperimentLedger(tmp_path / "invalid.db", durability="off")


@pytest.mark.parametrize("version", [1, 2])
def test_older_ledger_fails_closed_before_schema_mutation(tmp_path, version):
    path = tmp_path / f"version-{version}.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE ledger_meta (version INTEGER PRIMARY KEY, created_at_ns INTEGER)"
    )
    connection.execute(
        "INSERT INTO ledger_meta (version, created_at_ns) VALUES (?, 1)",
        (version,),
    )
    connection.execute("CREATE TABLE sentinel (value TEXT)")
    connection.commit()
    before = connection.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    connection.close()

    with pytest.raises(ExperimentLedgerError, match="fresh version-3"):
        ExperimentLedger(path)

    connection = sqlite3.connect(path)
    after = connection.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    connection.close()
    assert after == before


def test_unknown_trial_retains_ceiling_and_can_never_be_reused(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    path = tmp_path / "autotune.db"
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(ledger, contract, candidate)
        ledger.register_candidate(campaign_id, candidate)
        trial_key = _prepared(ledger, campaign_id, candidate)
        with pytest.raises(TrialStateError, match="dispatched"):
            ledger.complete_trial(
                trial_key,
                metrics={"quality": 8},
                gates={},
                actual_cost_microusd=0,
                duration_ms=1,
            )
        ledger.mark_trial_dispatched(trial_key)
        unknown = ledger.mark_trial_unknown(trial_key, "connection lost after dispatch")
        assert unknown.status is TrialStatus.UNKNOWN
        assert ledger.mark_trial_unknown(
            trial_key, "connection lost after dispatch"
        ).status is TrialStatus.UNKNOWN
        with pytest.raises(LedgerConflictError, match="different error"):
            ledger.mark_trial_unknown(trial_key, "different failure")

    with ExperimentLedger(path) as reopened:
        with pytest.raises(UnknownTrialError, match="cannot be reused"):
            _prepared(reopened, campaign_id, candidate)
        snapshot = reopened.snapshot(campaign_id)
        assert snapshot["cost"]["unknown_exposure_microusd"] == 1_000
        assert snapshot["cost"]["reserved_microusd"] == 0


def test_trial_results_reject_nonfinite_or_conflicting_payloads(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)
    trial_key = _prepared(ledger, campaign_id, candidate)
    ledger.mark_trial_dispatched(trial_key)

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            ledger.complete_trial(
                trial_key,
                metrics={"quality": value},
                gates={},
                actual_cost_microusd=0,
                duration_ms=1,
            )

    ledger.complete_trial(
        trial_key,
        metrics={"quality": 8},
        gates={},
        actual_cost_microusd=500,
        duration_ms=1,
    )
    with pytest.raises(LedgerConflictError, match="different result"):
        ledger.complete_trial(
            trial_key,
            metrics={"quality": 9},
            gates={},
            actual_cost_microusd=500,
            duration_ms=1,
        )


def test_trial_results_reject_integers_too_large_for_finite_metrics(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)
    trial_key = _prepared(ledger, campaign_id, candidate)
    ledger.claim_trial_dispatch(trial_key)

    with pytest.raises(ValueError, match="finite"):
        ledger.complete_trial(
            trial_key,
            metrics={"quality": 10**10_000},
            gates={},
            actual_cost_microusd=0,
            duration_ms=1,
        )
    with pytest.raises(ValueError, match="signed 64-bit"):
        ledger.complete_trial(
            trial_key,
            metrics={"quality": 8},
            gates={},
            actual_cost_microusd=2**63,
            duration_ms=1,
        )
    with pytest.raises(ValueError, match="finite"):
        ledger.complete_trial(
            trial_key,
            metrics={"quality": 8},
            gates={},
            actual_cost_microusd=0,
            duration_ms=10**10_000,
        )
    with pytest.raises(ValueError, match="signed 64-bit"):
        ledger.prepare_trial(
            campaign_id,
            candidate.candidate_id,
            "development",
            "evaluate",
            101,
            evaluator_hash="sha256:" + "e" * 64,
            ceiling_microusd=2**63,
        )
    assert ledger.get_trial(trial_key).status is TrialStatus.DISPATCHED


def test_promotion_decision_is_append_only_and_evidence_bound(tmp_path):
    contract = _contract()
    incumbent = _candidate(contract, "current")
    candidate = _candidate(contract, "challenger")
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, incumbent, candidate)
    incumbent_trial = _prepared(
        ledger,
        campaign_id,
        incumbent,
        seed=99,
        phase="incumbent.plan_" + "a" * 64,
    )
    ledger.mark_trial_dispatched(incumbent_trial)
    ledger.complete_trial(
        incumbent_trial,
        metrics={"quality": 8},
        gates={"format": True},
        actual_cost_microusd=500,
        duration_ms=1,
    )
    trial_key = _prepared(
        ledger,
        campaign_id,
        candidate,
        seed=99,
        phase="challenger.plan_" + "a" * 64,
    )
    ledger.mark_trial_dispatched(trial_key)
    ledger.complete_trial(
        trial_key,
        metrics={"quality": 9},
        gates={"format": True},
        actual_cost_microusd=500,
        duration_ms=1,
    )
    decision = PromotionDecision(
        campaign_id=campaign_id,
        candidate_id=candidate.candidate_id,
        promoted=False,
        reason="No challenger passed development gates",
        trial_keys=(incumbent_trial, trial_key),
        assessment={
            "optimization_plan_hash": PLAN_HASH,
            "holdout_seed_commitment": ledger.get_holdout_seed_commitment(campaign_id),
            "stage": "development",
            "development_scores": [],
        },
    )

    empty = PromotionDecision(
        campaign_id=campaign_id,
        candidate_id=candidate.candidate_id,
        promoted=False,
        reason="Missing evidence must fail closed",
        assessment=decision.to_dict()["assessment"],
    )
    with pytest.raises(LedgerConflictError, match="non-empty evidence"):
        ledger.append_decision(empty)

    unrelated_campaign = _seal(
        ledger,
        contract,
        incumbent,
        candidate,
        campaign_id="unrelated-campaign",
    )
    unrelated_trial = _prepared(
        ledger,
        unrelated_campaign,
        incumbent,
        seed=99,
        phase="incumbent.plan_" + "a" * 64,
    )
    ledger.claim_trial_dispatch(unrelated_trial)
    ledger.complete_trial(
        unrelated_trial,
        metrics={"quality": 8},
        gates={"format": True},
        actual_cost_microusd=0,
        duration_ms=1,
    )
    unrelated = PromotionDecision(
        campaign_id=campaign_id,
        candidate_id=candidate.candidate_id,
        promoted=False,
        reason="Cross-campaign evidence must fail closed",
        trial_keys=(incumbent_trial, unrelated_trial),
        assessment=decision.to_dict()["assessment"],
    )
    with pytest.raises(LedgerConflictError, match="exactly cover"):
        ledger.append_decision(unrelated)

    assert ledger.append_decision(decision) == decision.decision_id
    assert ledger.append_promotion_decision(decision) == decision.decision_id
    assert (
        _prepared(
            ledger,
            campaign_id,
            candidate,
            seed=99,
            phase="challenger.plan_" + "a" * 64,
        )
        == trial_key
    )
    with pytest.raises(LedgerConflictError, match="terminal decision"):
        _prepared(
            ledger,
            campaign_id,
            candidate,
            seed=100,
            phase="challenger.plan_" + "a" * 64,
        )
    conflicting = PromotionDecision(
        campaign_id=campaign_id,
        candidate_id=candidate.candidate_id,
        promoted=False,
        reason="A second terminal interpretation is forbidden",
        trial_keys=decision.trial_keys,
        assessment=decision.to_dict()["assessment"],
    )
    with pytest.raises(LedgerConflictError, match="different terminal decision"):
        ledger.append_decision(conflicting)
    snapshot = ledger.snapshot(campaign_id)
    assert snapshot["decision_count"] == 1
    assert snapshot["decisions"][0]["promoted"] is False
    assert snapshot["decisions"][0]["assessment"]["stage"] == "development"
    assert ledger.validate_decision_inventory(campaign_id) == (decision.decision_id,)


def test_promotion_decision_identity_deep_freezes_all_nested_evidence():
    assessment = {"nested": {"scores": [1, {"lower_bound": 0.4}]}}
    trial_keys = ["trial-one", "trial-two"]
    decision = PromotionDecision(
        campaign_id="campaign-one",
        candidate_id="candidate-one",
        promoted=True,
        reason="Evidence is immutable",
        trial_keys=trial_keys,
        assessment=assessment,
    )
    original_id = decision.decision_id

    assessment["nested"]["scores"][1]["lower_bound"] = -99
    trial_keys.append("trial-three")
    assert decision.to_dict()["assessment"]["nested"]["scores"][1] == {
        "lower_bound": 0.4
    }
    assert decision.trial_keys == ("trial-one", "trial-two")
    assert decision.decision_id == original_id

    with pytest.raises(TypeError):
        decision.assessment["new"] = "mutable"
    with pytest.raises(TypeError):
        decision.assessment["nested"]["scores"][1]["lower_bound"] = 2

    mutable_copy = decision.to_dict()
    mutable_copy["assessment"]["nested"]["scores"][1]["lower_bound"] = 7
    mutable_copy["trial_keys"].append("trial-four")
    assert decision.to_dict()["assessment"]["nested"]["scores"][1][
        "lower_bound"
    ] == 0.4
    assert decision.decision_id == original_id


def test_candidate_parent_must_precede_child_in_atomic_inventory(tmp_path):
    contract = _contract()
    parent = _candidate(contract, "parent")
    child = Candidate(
        contract=contract,
        policy={"prompt.logo": "child"},
        hypothesis="The child descends from the registered parent",
        parent=parent,
    )
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, parent, child)
    assert ledger.register_candidate(campaign_id, parent) == parent.candidate_id
    assert ledger.register_candidate(campaign_id, child) == child.candidate_id

    with pytest.raises(LedgerConflictError, match="parent must precede"):
        ledger.create_campaign(
            contract,
            child.candidate_id,
            (child, parent),
            plan_hash="sha256:" + "b" * 64,
            campaign_id="reversed-inventory",
        )
    assert ledger._connection.execute(
        "SELECT COUNT(*) FROM campaigns WHERE campaign_id = 'reversed-inventory'"
    ).fetchone()[0] == 0


def test_contract_candidate_trial_and_budget_limits_are_enforced(tmp_path):
    contract = _contract(max_candidates=1, max_trials=7, budget=7_000)
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)
    with pytest.raises(LedgerConflictError, match="sealed campaign inventory"):
        ledger.register_candidate(campaign_id, _candidate(contract, "alternate"))

    trial_key = _prepared(ledger, campaign_id, candidate)
    ledger.mark_trial_dispatched(trial_key)
    ledger.complete_trial(
        trial_key,
        metrics={"quality": 8},
        gates={},
        actual_cost_microusd=6_500,
        duration_ms=1,
    )
    with pytest.raises(LedgerBudgetError, match="budget"):
        _prepared(ledger, campaign_id, candidate, seed=101)


def test_prepare_rejects_ceiling_drift_from_immutable_contract(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)

    with pytest.raises(LedgerConflictError, match="contract reservation"):
        ledger.prepare_trial(
            campaign_id,
            candidate.candidate_id,
            "development",
            "evaluate",
            100,
            evaluator_hash="sha256:" + "e" * 64,
            ceiling_microusd=999,
        )


def test_evaluator_and_artifact_evidence_require_canonical_sha256(tmp_path):
    contract = _contract()
    candidate = _candidate(contract)
    ledger = ExperimentLedger(tmp_path / "autotune.db")
    campaign_id = _seal(ledger, contract, candidate)
    ledger.register_candidate(campaign_id, candidate)

    with pytest.raises(ValueError, match="sha256"):
        ledger.prepare_trial(
            campaign_id,
            candidate.candidate_id,
            "development",
            "evaluate",
            100,
            evaluator_hash="evaluator-v1",
            ceiling_microusd=1_000,
        )

    trial_key = _prepared(ledger, campaign_id, candidate)
    ledger.claim_trial_dispatch(trial_key)
    with pytest.raises(ValueError, match="sha256"):
        ledger.complete_trial(
            trial_key,
            metrics={"quality": 8},
            gates={},
            actual_cost_microusd=0,
            duration_ms=1,
            artifact_hashes=("looks-hashed",),
        )


def test_read_only_ledger_inspection_never_creates_or_mutates(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        ExperimentLedger(missing, read_only=True)
    assert not missing.exists()

    path = tmp_path / "autotune.db"
    contract = _contract()
    candidate = _candidate(contract)
    with ExperimentLedger(path) as ledger:
        campaign_id = _seal(ledger, contract, candidate)
        ledger.register_candidate(campaign_id, candidate)

    with ExperimentLedger(path, read_only=True) as ledger:
        assert ledger.snapshot(campaign_id)["candidate_count"] == 1
        with pytest.raises(sqlite3.OperationalError) as error:
            _seal(
                ledger,
                contract,
                candidate,
                campaign_id="read-only-mutation",
                plan_hash="sha256:" + "b" * 64,
            )
        assert "readonly" in str(error.value).lower()
