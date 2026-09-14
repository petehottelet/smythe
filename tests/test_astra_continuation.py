"""Continuation tests use the real ledger and an offline native transport."""

import asyncio
import hashlib

import pytest

from benchmarks import astra_continuation as continuation, astra_runtime as runtime, astra_study as study
from benchmarks.astra_campaign import load_task_pack, prepare_campaign
from smythe.task import task_to_dict
from test_astra_runtime import native  # noqa: F401


class APIConnectionError(ConnectionError):
    """Offline equivalent of a transport exception with no HTTP response."""


def disconnect():
    raise APIConnectionError("Offline connection loss")


@pytest.fixture
def interrupted(tmp_path, monkeypatch, native):  # noqa: F811
    monkeypatch.setattr(study, "_verify_freeze", lambda _: None)
    cases = {c.task_id: c for c in load_task_pack().tasks}
    schedule = prepare_campaign()["schedules"]["main"][:2]
    base = {"stage": "main", "directory": str(tmp_path / "original"),
        "schedule": schedule, "per_trial_nanousd": 5_000_000_000,
        "stage_allowance_nanousd": 200_000_000_000,
        "tasks": {r["task_id"]: task_to_dict(study.study_task(cases[r["task_id"]])) for r in schedule}}
    base["freeze_sha256"] = runtime._sha(base)
    native.before_generate = disconnect
    with pytest.raises(ValueError, match="Unresolved study billing"):
        asyncio.run(study.run_stage(base))
    return base, tmp_path / "next", native


def frozen(interrupted):
    base, destination, _ = interrupted
    return continuation.freeze_continuation(original_directory=base["directory"], directory=destination)


def test_native_failure_is_retained_and_only_the_unstarted_suffix_runs(interrupted):
    base, _, transport = interrupted
    value = frozen(interrupted)
    assert value["schedule"] == base["schedule"][1:]
    assert value["prior"]["completed_workflows"] == 1
    assert value["prior"]["unknown_nanousd"] > 0
    original = runtime._safe_path(base["directory"])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.glob("*.outcome.json")}
    summary = continuation.run_approved(value, value["freeze_sha256"])
    assert summary["completed_workflows"] == 1 and summary["confirmed_nanousd"] > 0
    assert summary["held_unknown_nanousd"] == value["prior"]["unknown_nanousd"]
    calls = len(transport.requests)
    assert continuation.run_approved(value, value["freeze_sha256"]) == summary
    assert len(transport.requests) == calls
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.glob("*.outcome.json")}
    outcomes, retained = continuation.audit_segment(original, base, base["schedule"])
    assert outcomes[0]["status"] == "failed" and outcomes[0]["output"] is None
    assert retained == value["prior"]


def test_approval_of_the_exact_continuation_is_required(interrupted):
    value = frozen(interrupted)
    transport = interrupted[2]
    before = len(transport.requests), len(transport.quotes)
    with pytest.raises(ValueError, match="Explicit approval"):
        continuation.run_approved(value, "wrong")
    assert (len(transport.requests), len(transport.quotes)) == before


def test_insufficient_remaining_allocation_counts_the_whole_unknown_reserve():
    base = {"per_trial_nanousd": 100, "stage_allowance_nanousd": 199}
    prior = {"confirmed_nanousd": 40, "unknown_nanousd": 50}
    with pytest.raises(ValueError, match="held exposure"):
        continuation.admit(base, prior, 10)
    base["stage_allowance_nanousd"] = 200
    continuation.admit(base, prior, 10)


@pytest.mark.parametrize("value", [-1, True, 1.5, "12", None])
def test_malformed_held_balances_cannot_authorize_dispatch(value):
    with pytest.raises(ValueError, match="strict"):
        continuation.admit({"per_trial_nanousd": 100, "stage_allowance_nanousd": 1000},
                           {"confirmed_nanousd": 0, "unknown_nanousd": value}, 0)


def test_changed_prior_record_blocks_before_a_new_quote(interrupted):
    value = frozen(interrupted)
    transport = interrupted[2]
    path = next(runtime._safe_path(value["original_directory"]).glob("*.outcome.json"))
    path.write_bytes(path.read_bytes().replace(b'"failed"', b'"completed"'))
    before = len(transport.quotes)
    with pytest.raises(ValueError, match="identity or content"):
        continuation.run_approved(value, value["freeze_sha256"])
    assert len(transport.quotes) == before


def test_a_second_connection_failure_stops_without_rebuy_on_replay(interrupted):
    value = frozen(interrupted)
    transport = interrupted[2]
    transport.before_generate = disconnect
    with pytest.raises(ValueError, match="New unresolved billing"):
        continuation.run_approved(value, value["freeze_sha256"])
    before = len(transport.requests), len(transport.quotes)
    with pytest.raises(ValueError, match="New unresolved billing"):
        continuation.run_approved(value, value["freeze_sha256"])
    assert (len(transport.requests), len(transport.quotes)) == before
    paths = list(runtime._safe_path(value["directory"]).glob("*.outcome.json"))
    assert len(paths) == 1 and runtime._read(paths[0])["status"] == "failed"


def test_original_and_continuation_cannot_alias(interrupted):
    base, _, transport = interrupted
    with pytest.raises(ValueError, match="separate peers"):
        continuation.freeze_continuation(original_directory=base["directory"], directory=base["directory"])
    assert not transport.requests


def test_changed_policy_is_rejected_even_with_a_rehashed_freeze(interrupted):
    value = frozen(interrupted)
    value["policy"]["hold_full_unknown_reservation"] = False
    value.pop("freeze_sha256")
    value["freeze_sha256"] = runtime._sha(value)
    with pytest.raises(ValueError, match="reservation policy"):
        continuation.run_approved(value, value["freeze_sha256"])
    assert not interrupted[2].requests


def test_unbound_outcome_file_blocks_admission(interrupted):
    value = frozen(interrupted)
    path = runtime._safe_path(value["original_directory"]) / "unbound.outcome.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unbound outcome files"):
        continuation.run_approved(value, value["freeze_sha256"])
    assert not interrupted[2].requests
