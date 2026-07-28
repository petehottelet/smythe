"""CLI tests for the first deterministic Autotune campaign."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import NoReturn

import pytest

from smythe.cli import (
    EXIT_INVALID_INPUT,
    EXIT_OK,
    EXIT_OPTIMIZE_LIMIT,
    EXIT_OPTIMIZE_STATE,
    MAX_OPTIMIZE_CANDIDATES,
    MAX_OPTIMIZE_CONCURRENCY,
    _candidate_concurrencies,
    _concurrency_campaign,
    build_parser,
    main,
)
from smythe.optimize.engine import OptimizationRunner
from smythe.optimize.concurrency import simulate_concurrency as reference_simulate_concurrency
from smythe.optimize.ledger import ExperimentLedger, LedgerBudgetError, TrialStatus


def _json_output(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def _run(
    ledger: Path,
    capsys: pytest.CaptureFixture[str],
    *options: str,
) -> dict[str, object]:
    assert main(
        [
            "optimize",
            "concurrency",
            "--ledger",
            str(ledger),
            "--json",
            *options,
        ]
    ) == EXIT_OK
    return _json_output(capsys)["concurrency"]


def test_concurrency_contract_is_bounded_exact_and_lineage_bound():
    args = build_parser().parse_args(
        [
            "optimize",
            "concurrency",
            "--candidate-concurrency",
            "2,4",
            "--candidate-concurrency",
            "8",
        ]
    )

    contract, incumbent, challengers, scenario = _concurrency_campaign(args)

    assert scenario.work_items == 250
    assert contract.max_candidates == 4
    assert contract.max_trials == 4 * 3 + 2 * (5 + 5)
    assert contract.max_budget_microusd == 0
    assert contract.per_trial_reservation_microusd == 0
    assert contract.required_gates == ("all_operations_accounted",)
    assert contract.mutable_field_rules["max_concurrency"].minimum == 1
    assert contract.mutable_field_rules["max_concurrency"].maximum == 250
    assert incumbent.policy == {"max_concurrency": 1}
    assert [candidate.policy["max_concurrency"] for candidate in challengers] == [
        2,
        4,
        8,
    ]
    assert all(candidate.parent == incumbent.candidate_id for candidate in challengers)
    assert [objective.name for objective in contract.objectives] == [
        "throughput_ops_s",
        "p95_latency_ms",
        "error_rate",
    ]
    assert contract.objectives[1].hard_max == 250.0
    assert contract.objectives[1].max_regression == 37.5
    assert contract.objectives[2].hard_max == 0.05


@pytest.mark.parametrize(
    "values, message",
    [
        (["0"], "positive"),
        (["1"], "incumbent"),
        (["2,2"], "duplicate"),
        (["2", "2"], "duplicate"),
        (["2,,4"], "empty"),
        (["many"], "integer"),
    ],
)
def test_candidate_concurrency_rejects_ambiguous_input(values, message):
    with pytest.raises(ValueError, match=message):
        _candidate_concurrencies(values)


def test_candidate_grid_and_campaign_work_are_hard_bounded():
    with pytest.raises(ValueError, match="must not exceed"):
        _candidate_concurrencies([str(MAX_OPTIMIZE_CONCURRENCY + 1)])
    too_many = [str(value) for value in range(2, MAX_OPTIMIZE_CANDIDATES + 3)]
    with pytest.raises(ValueError, match="at most"):
        _candidate_concurrencies(too_many)

    args = build_parser().parse_args(
        [
            "optimize",
            "concurrency",
            "--candidate-concurrency",
            "2",
            "--work-items",
            "1000000",
        ]
    )
    with pytest.raises(ValueError, match="maximum"):
        _concurrency_campaign(args)

    args = build_parser().parse_args(
        [
            "optimize",
            "concurrency",
            "--bootstrap-resamples",
            "100001",
        ]
    )
    with pytest.raises(ValueError, match="must not exceed"):
        _concurrency_campaign(args)


def test_default_campaign_is_offline_resume_safe_and_fully_evidenced(
    tmp_path,
    capsys,
    monkeypatch,
):
    ledger_path = tmp_path / "optimize.sqlite3"

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("offline optimize campaign attempted external execution")

    monkeypatch.setattr("smythe.cli.ProviderPool", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)

    first = _run(ledger_path, capsys)

    assert first["campaign_id"].startswith("optimization_v3_")
    assert first["optimization_plan_hash"].startswith("sha256:")
    assert first["evaluator_hash"].startswith("sha256:")
    assert first["contract_hash"].startswith("sha256:")
    assert first["ledger_durability"] == "normal"
    assert first["promoted"] is True
    assert first["selected_candidate"]["policy"] == {"max_concurrency": 8}
    assert first["recommended_patch"] == {"max_concurrency": 8}
    snapshot = first["ledger_snapshot"]
    commitment = snapshot["campaign"]["holdout_seed_commitment"]
    assert commitment.startswith("sha256:")
    assert "holdout_nonce" not in snapshot["campaign"]
    assert first["evidence"]["holdout_seed_commitment"] == commitment
    assert snapshot["trial_count"] == 38
    assert snapshot["trial_counts"] == {"completed": 38}
    assert snapshot["decision_count"] == 1
    assert snapshot["cost"] == {
        "confirmed_microusd": 0,
        "reserved_microusd": 0,
        "unknown_exposure_microusd": 0,
        "total_exposure_microusd": 0,
    }

    with ExperimentLedger(ledger_path, read_only=True) as ledger:
        trials = ledger.list_trials(first["campaign_id"])
    assert all(trial.status is TrialStatus.COMPLETED for trial in trials)
    assert all(
        set(trial.metrics)
        == {"throughput_ops_s", "p95_latency_ms", "error_rate"}
        for trial in trials
    )
    assert all(trial.gates == {"all_operations_accounted": True} for trial in trials)
    assert {trial.evaluator_hash for trial in trials} == {first["evaluator_hash"]}

    resumed = _run(ledger_path, capsys)
    assert resumed["campaign_id"] == first["campaign_id"]
    assert resumed["optimization_plan_hash"] == first["optimization_plan_hash"]
    assert resumed["ledger_snapshot"]["trial_count"] == 38
    assert resumed["ledger_snapshot"]["decision_count"] == 1


def test_rejection_never_emits_a_recommended_patch(tmp_path, capsys):
    rejected = _run(
        tmp_path / "rejected.sqlite3",
        capsys,
        "--candidate-concurrency",
        "2",
        "--min-improvement",
        "1000",
        "--bootstrap-resamples",
        "10",
        "--development-repetitions",
        "1",
        "--confirmation-repetitions",
        "3",
        "--holdout-repetitions",
        "3",
    )

    assert rejected["promoted"] is False
    assert "recommended_patch" not in rejected
    assert rejected["evidence"]["reason"]


def test_cli_simulator_runs_off_the_event_loop_thread(tmp_path, capsys, monkeypatch):
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []

    def observed_simulator(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        assert kwargs["deadline"] is not None
        return reference_simulate_concurrency(*args, **kwargs)

    monkeypatch.setattr("smythe.cli.simulate_concurrency", observed_simulator)
    _run(
        tmp_path / "threaded.sqlite3",
        capsys,
        "--candidate-concurrency",
        "2",
        "--bootstrap-resamples",
        "10",
        "--development-repetitions",
        "1",
        "--confirmation-repetitions",
        "3",
        "--holdout-repetitions",
        "3",
    )

    assert worker_threads
    assert caller_thread not in worker_threads


def test_no_viable_development_candidate_returns_null_without_patch(tmp_path, capsys):
    rejected = _run(
        tmp_path / "inviable.sqlite3",
        capsys,
        "--candidate-concurrency",
        "2",
        "--max-p95-latency-ms",
        "1",
        "--bootstrap-resamples",
        "10",
        "--development-repetitions",
        "1",
        "--confirmation-repetitions",
        "3",
        "--holdout-repetitions",
        "3",
    )

    assert rejected["promoted"] is False
    assert rejected["selected_candidate"] is None
    assert "recommended_patch" not in rejected
    assert rejected["evidence"]["reason"] == (
        "no candidate passed development gates and hard bounds"
    )
    assert rejected["ledger_snapshot"]["trial_count"] == 2


def test_auto_identity_includes_candidates_and_bootstrap_configuration(tmp_path, capsys):
    common = (
        "--development-repetitions",
        "1",
        "--confirmation-repetitions",
        "3",
        "--holdout-repetitions",
        "3",
    )
    parser = build_parser()

    async def unused_evaluator(_context):
        raise AssertionError("plan identity test must not dispatch a trial")

    def plan(ledger, *options):
        args = parser.parse_args(["optimize", "concurrency", *options, *common])
        contract, incumbent, candidates, scenario = _concurrency_campaign(args)
        runner = OptimizationRunner(
            contract,
            ledger,
            unused_evaluator,
            evaluator_hash=scenario.evaluator_hash,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        return runner._build_plan(incumbent, candidates)

    with ExperimentLedger(
        tmp_path / "identity.sqlite3",
        durability="normal",
    ) as ledger:
        baseline = plan(
            ledger,
            "--candidate-concurrency",
            "2",
            "--bootstrap-resamples",
            "10",
        )
        changed_bootstrap = plan(
            ledger,
            "--candidate-concurrency",
            "2",
            "--bootstrap-resamples",
            "11",
        )
        changed_candidates = plan(
            ledger,
            "--candidate-concurrency",
            "2,4",
            "--bootstrap-resamples",
            "10",
        )

    assert len(
        {
            baseline["campaign_id"],
            changed_bootstrap["campaign_id"],
            changed_candidates["campaign_id"],
        }
    ) == 3
    assert len(
        {
            baseline["plan_hash"],
            changed_bootstrap["plan_hash"],
            changed_candidates["plan_hash"],
        }
    ) == 3


def test_inspect_is_read_only_and_missing_campaign_has_stable_exit(tmp_path, capsys):
    ledger_path = tmp_path / "inspect.sqlite3"
    campaign = _run(
        ledger_path,
        capsys,
        "--candidate-concurrency",
        "2",
        "--bootstrap-resamples",
        "10",
        "--development-repetitions",
        "1",
        "--confirmation-repetitions",
        "3",
        "--holdout-repetitions",
        "3",
    )
    before_files = sorted(path.name for path in tmp_path.iterdir())
    before_mtime = ledger_path.stat().st_mtime_ns

    assert main(
        [
            "optimize",
            "inspect",
            campaign["campaign_id"],
            "--ledger",
            str(ledger_path),
            "--json",
        ]
    ) == EXIT_OK
    inspected = _json_output(capsys)["inspect"]
    assert inspected["campaign_id"] == campaign["campaign_id"]
    assert inspected["ledger_snapshot"] == campaign["ledger_snapshot"]
    assert inspected["evaluator_hashes"] == [campaign["evaluator_hash"]]
    assert sorted(path.name for path in tmp_path.iterdir()) == before_files
    assert ledger_path.stat().st_mtime_ns == before_mtime

    missing_path = tmp_path / "missing.sqlite3"
    assert main(
        [
            "optimize",
            "inspect",
            "missing-campaign",
            "--ledger",
            str(missing_path),
            "--json",
        ]
    ) == EXIT_OPTIMIZE_STATE
    error = _json_output(capsys)
    assert error["error"]["type"] == "FileNotFoundError"
    assert not missing_path.exists()


def test_invalid_input_and_limit_have_stable_exits(tmp_path, capsys, monkeypatch):
    ledger_path = tmp_path / "invalid.sqlite3"
    assert main(
        [
            "optimize",
            "concurrency",
            "--candidate-concurrency",
            "2,2",
            "--ledger",
            str(ledger_path),
            "--json",
        ]
    ) == EXIT_INVALID_INPUT
    assert _json_output(capsys)["error"]["type"] == "ValueError"
    assert not ledger_path.exists()

    async def over_limit(_args):
        raise LedgerBudgetError("campaign budget limit reached")

    monkeypatch.setattr("smythe.cli._run_optimize_concurrency", over_limit)
    assert main(
        [
            "optimize",
            "concurrency",
            "--ledger",
            str(ledger_path),
            "--json",
        ]
    ) == EXIT_OPTIMIZE_LIMIT
    assert _json_output(capsys)["error"]["type"] == "LedgerBudgetError"
