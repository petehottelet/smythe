"""The offline policy ablations: drain, capacity admission, budget holds and contradiction synthesis.

The harness runs in a child process without provider keys and with this
suite's network guard installed (``GUARDED_RUN``), recording every socket use
the guard refuses. Each ablation's two arms differ in one policy. Fixture runs
establish mechanics only, so these tests check what the record holds and how
the arms differ on fixtures planted for that purpose, never a measured
benefit; the scheduler overhead test is checked for structure, never for its
times.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import threading
import time as pytime

import pytest

from fakes import child_executable, offline_env
from test_selector_comparison import GUARDED_RUN

from benchmarks.run_policy_ablations import (
    ARMS, CONTRADICTION_ALLOWANCE, DRAIN_TIMEOUT_S, OVERHEAD_EXECUTORS, OVERHEAD_LATENCIES_MS,
    QUOTA_IN_FLIGHT, VirtualTime, ablation_arms, budget_holds, correlated_errors, one_change,
    run_capacity_arm, run_drain_ordinary_arm, run_hold_arm,
)

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
HARNESS = ROOT / "benchmarks" / "run_policy_ablations.py"
RECORD = ROOT / "benchmarks" / "results" / "policy_ablations_offline.json"
# Each ablation and the one policy its arms vary.
ABLATIONS = {
    "drain_ordinary": "termination",
    "drain_durable": "termination",
    "capacity_admission": "capacity_policy",
    "budget_holds": "reserve_for_checks",
    "contradiction_synthesis": "retain_contradictions",
}
# Runs of each budget-hold arm that must match exactly.
HOLD_REPEATS = 10


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("ablations")
    out, refused = work / "record.json", work / "refused.json"
    proc = subprocess.run(
        [child_executable(), "-c", GUARDED_RUN, str(TESTS), str(HARNESS), str(refused),
         "--overhead-repetitions", "1", "--out", str(out)],
        cwd=ROOT, env=offline_env(), capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return (json.loads(out.read_text(encoding="utf-8")),
            json.loads(refused.read_text(encoding="utf-8")))


def _arms(record, ablation, case):
    """The baseline and policy results of one case."""
    [found] = [item for item in record["ablations"][ablation]["cases"] if item["case"] == case]
    return found["results"]["baseline"], found["results"]["policy"]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_run_is_offline_and_labeled_as_mechanics(offline_run):
    record, refused = offline_run
    assert refused == []
    assert not any(name.upper().endswith("_API_KEY") for name in offline_env())
    assert record["benchmark"] == "policy-ablations"
    assert record["evidence_status"] == "offline-mechanics"
    assert record["claimable"] is False and record["paid_execution_allowed"] is False
    assert record["api_calls"] == 0 and record["mode"] == "offline"
    assert record["default_change"] is False and record["promotion"]["eligible"] is False
    assert record["arms"] == list(ARMS)


def test_each_ablation_varies_exactly_one_policy(offline_run):
    record, _ = offline_run
    assert set(record["ablations"]) == set(ABLATIONS)
    for name, ablation in record["ablations"].items():
        varied = ABLATIONS[name]
        assert ablation["varied"] == varied and ablation["cases"]
        for case in ablation["cases"]:
            baseline, policy = case["arms"]["baseline"], case["arms"]["policy"]
            assert case["differs_in"] == one_change(baseline, policy) == [varied], case["case"]
            # The baseline runs without the policy, and every other setting is shared.
            assert baseline[varied] in (None, False) and policy[varied] not in (None, False)
            assert ({key: value for key, value in baseline.items() if key != varied}
                    == {key: value for key, value in policy.items() if key != varied})
            assert set(case["results"]) == set(ARMS)
            for measure in case["comparison"].values():
                assert set(measure) == set(ARMS)


def test_every_record_keeps_failed_attempts_and_correlated_errors(offline_run):
    record, _ = offline_run
    # A drain's baseline: the planted failure, and the sibling it cancelled.
    baseline, policy = _arms(record, "drain_ordinary", "answers_within_drain")
    assert baseline["failures"] == [
        {"node": "fail", "attempt": 1, "outcome": "failed",
         "error": {"type": "RuntimeError", "message": "planted failure"},
         "failure_class": "transient", "caused_by": None},
        {"node": "sibling", "attempt": 1, "outcome": "cancelled", "error": None,
         "failure_class": None, "caused_by": "fail"},
    ]
    assert baseline["correlated_errors"] == [{
        "kind": "consequence", "cause": "the failure of 'fail'",
        "failures": [{"node": "sibling", "attempt": 1}], "count": 1}]
    assert [item["node"] for item in policy["failures"]] == ["fail"]
    assert policy["correlated_errors"] == []

    # A durable run: the rejected call, and the sibling it left with unknown exposure.
    baseline, _ = _arms(record, "drain_durable", "answered_within_drain")
    rejected, unknown = baseline["failures"]
    assert (rejected["node"], rejected["outcome"], rejected["failure_class"]) == (
        "halt", "rejected", "authority_denied")
    assert "HTTP 401" in rejected["error"]["message"]
    assert (unknown["node"], unknown["outcome"], unknown["caused_by"]) == (
        "sibling", "unknown_exposure", "halt")

    # Overload: every rejected call is kept, and they share one signature.
    for case in ("retries_cover_overload", "retries_run_out"):
        baseline, _ = _arms(record, "capacity_admission", case)
        rejections = [item for item in baseline["failures"] if item["outcome"] == "failed"]
        assert len(rejections) == baseline["rejected_429"] > 0
        [shared] = [group for group in baseline["correlated_errors"]
                    if group["kind"] == "shared_signature"]
        assert shared["count"] == baseline["rejected_429"]
        assert shared["signature"] == {"outcome": "failed", "error_type": "RateLimitError",
                                       "failure_class": "transient", "message": "Error code: 429"}
    # Out of retries, the stopped run cancels the calls still in flight.
    baseline, _ = _arms(record, "capacity_admission", "retries_run_out")
    [consequence] = [group for group in baseline["correlated_errors"]
                     if group["kind"] == "consequence"]
    assert consequence["count"] == baseline["cancelled"] > 0

    baseline, _ = _arms(record, "budget_holds", "fan_out_short_budget")
    [refusal] = baseline["failures"]
    assert (refusal["node"], refusal["outcome"], refusal["failure_class"]) == (
        "__synthesis__", "refused", "terminal_accounting")

    # A rejection repeated on every attempt is one correlated error, not two chances.
    baseline, _ = _arms(record, "contradiction_synthesis", "ferry-dropped")
    assert len(baseline["attempts"]) == baseline["failed_attempts"] == CONTRADICTION_ALLOWANCE
    [repeated] = baseline["correlated_errors"]
    assert repeated["kind"] == "shared_signature" and repeated["count"] == CONTRADICTION_ALLOWANCE
    assert "contradiction_missing" in repeated["signature"]["message"]


# ---------------------------------------------------------------------------
# What each policy changes on its fixtures
# ---------------------------------------------------------------------------


def test_drain_keeps_in_flight_work_and_never_delays_a_security_stop(offline_run):
    record, _ = offline_run
    baseline, policy = _arms(record, "drain_ordinary", "answers_within_drain")
    # Without a policy the sibling is cancelled at the failure, and a resume repeats its call.
    assert baseline["nodes"] == {"fail": "failed", "sibling": "pending", "queued": "pending"}
    assert baseline["drain"] is None and baseline["kept_cost_usd"] == {}
    assert baseline["resume"] == {"nodes_to_run": ["fail", "queued", "sibling"],
                                  "sent_calls_repeated": 1}
    assert baseline["elapsed_ms"] == 1000
    # Draining lets it answer and keeps its result and cost; admission still stops.
    assert policy["nodes"] == {"fail": "failed", "sibling": "completed", "queued": "pending"}
    assert policy["drain"] == {"settled": ["sibling"], "cancelled": [], "deadline_hit": False}
    assert policy["kept_cost_usd"] == {"sibling": 0.125} and policy["reserved_usd"] == 0
    assert policy["resume"] == {"nodes_to_run": ["fail", "queued"], "sent_calls_repeated": 0}
    assert policy["elapsed_ms"] == 3000 and policy["waits_ms"]["drain"] == 2000
    for result in (baseline, policy):
        assert "queued" not in [call["node"] for call in result["calls"]]

    baseline, policy = _arms(record, "drain_ordinary", "straggles_past_deadline")
    assert policy["drain"] == {"settled": [], "cancelled": ["sibling"], "deadline_hit": True}
    assert policy["waits_ms"]["drain"] == DRAIN_TIMEOUT_S * 1000
    assert policy["elapsed_ms"] == 1000 + DRAIN_TIMEOUT_S * 1000
    assert policy["resume"] == baseline["resume"]

    # A security stop cancels at once under either arm, keeping the sent call's
    # reservation as unknown cost.
    baseline, policy = _arms(record, "drain_ordinary", "security_stop")
    assert baseline == policy
    assert policy["drain"] is None and policy["elapsed_ms"] == 1000
    assert policy["error"]["type"] == "SecurityStop" and policy["cost_is_complete"] is False
    assert policy["kept_cost_usd"] == {"sibling": 0.25}


def test_a_durable_drain_settles_the_sent_call_a_cancellation_leaves_unknown(offline_run):
    record, _ = offline_run
    baseline, policy = _arms(record, "drain_durable", "answered_within_drain")
    [sibling] = baseline["journal"]["calls"]["sibling"]
    assert (sibling["state"], sibling["billing_state"]) == ("unknown", "unknown")
    assert baseline["journal"]["unknown_nanousd"] == sibling["unknown_nanousd"] > 0
    assert baseline["journal"]["blocked_reason"] == "unknown_exposure"
    assert baseline["resume"]["outcome"] == "refused"
    assert baseline["resume"]["sent"] == {"Halt": 0, "Sibling": 0}
    assert baseline["drain_waits_s"] == []

    [sibling] = policy["journal"]["calls"]["sibling"]
    assert (sibling["state"], sibling["billing_state"], sibling["result_state"]) == (
        "settled", "known", "applied")
    assert policy["journal"]["unknown_nanousd"] == 0
    assert policy["journal"]["blocked_reason"] is None
    assert policy["drain_waits_s"] == [DRAIN_TIMEOUT_S]
    # Resume calls only the refused node; the drained sibling's result is kept.
    assert policy["resume"]["outcome"] == "completed" and policy["resume"]["cost_is_complete"]
    assert policy["resume"]["sent"] == {"Halt": 1, "Sibling": 0}
    assert policy["resume"]["journal"]["status"] == "completed"
    for result in (baseline, policy):
        assert result["sent"] == {"Halt": 1, "Sibling": 1}

    # Past the deadline, the drain preserves the exposure it could not settle.
    baseline, policy = _arms(record, "drain_durable", "deadline_passes")
    assert policy["drain_waits_s"] == [DRAIN_TIMEOUT_S]
    for result in (baseline, policy):
        [sibling] = result["journal"]["calls"]["sibling"]
        assert sibling["state"] == "unknown" and sibling["unknown_nanousd"] > 0
        assert result["resume"]["outcome"] == "refused"


def test_capacity_admission_defers_work_instead_of_overloading_the_provider(offline_run):
    record, _ = offline_run
    for case in ("retries_cover_overload", "retries_run_out"):
        baseline, policy = _arms(record, "capacity_admission", case)
        assert baseline["peak_in_flight"] > QUOTA_IN_FLIGHT
        assert baseline["rejected_429"] > 0 and baseline["waits_ms"]["retry_backoff"] > 0
        # The limit holds the calls in flight to the quota: nothing is refused or
        # retried, and the other ready nodes wait for capacity instead.
        assert policy["peak_in_flight"] == QUOTA_IN_FLIGHT
        assert policy["rejected_429"] == 0 and policy["calls"] == policy["answered"] == 8
        assert policy["nodes"] == {"completed": 8} and policy["error"] is None
        assert policy["waits_ms"]["retry_backoff"] == 0 and policy["waits_ms"]["capacity"] > 0
    baseline, policy = _arms(record, "capacity_admission", "retries_cover_overload")
    assert baseline["nodes"] == {"completed": 8}
    assert policy["elapsed_ms"] < baseline["elapsed_ms"]
    baseline, _ = _arms(record, "capacity_admission", "retries_run_out")
    assert baseline["error"]["type"] == "RateLimitError" and baseline["nodes"]["failed"] > 0

    baseline, policy = _arms(record, "capacity_admission", "simulated_provider")
    assert (baseline["admitted_concurrency"], policy["admitted_concurrency"]) == (16, 8)
    assert [trial["seed"] for trial in baseline["trials"]] == [
        trial["seed"] for trial in policy["trials"]]
    assert policy["mean"]["error_rate"] < baseline["mean"]["error_rate"]


def test_budget_holds_refuse_before_spend_that_cannot_deliver(offline_run):
    record, _ = offline_run
    baseline, policy = _arms(record, "budget_holds", "fan_out_short_budget")
    # Without holds every worker is paid, then the merge finds the allowance spent.
    assert baseline["refused"] == {"node": "__synthesis__", "role": "synthesis",
                                   "called_before": 0}
    assert baseline["spend_usd"] == 1.0 and len(baseline["calls"]) == 8
    # With holds a worker is refused before its call, and nothing reaches the merge.
    assert policy["refused"]["role"] == "worker" and policy["refused"]["called_before"] == 0
    assert policy["spend_usd"] < baseline["spend_usd"]
    assert "synthesis" not in policy["calls"]
    assert not baseline["delivered"] and not policy["delivered"]

    baseline, policy = _arms(record, "budget_holds", "fan_out_covered_budget")
    # A hold is never spend: a budget that covers the run delivers the same output.
    for result in (baseline, policy):
        assert result["outcome"] == "completed" and result["spend_usd"] == 1.125
    assert baseline["output_sha256"] == policy["output_sha256"] is not None

    baseline, policy = _arms(record, "budget_holds", "regeneration_rerun")
    assert baseline["calls"] == ["draft", "judge", "draft"]
    assert baseline["refused"] == {"node": "judge", "role": "verifier", "called_before": 1}
    assert policy["calls"] == ["draft", "judge"]
    assert policy["refused"] == {"node": "draft", "role": "rerun", "called_before": 1}


def test_contradiction_synthesis_carries_every_contradiction_to_acceptance(offline_run):
    record, _ = offline_run
    for case in ("ferry-dropped", "ferry-partly"):
        baseline, policy = _arms(record, "contradiction_synthesis", case)
        assert not baseline["accepted"]
        assert all("contradiction_missing" in attempt["reasons"][0]
                   for attempt in baseline["attempts"])
        [attempt] = policy["attempts"]
        assert policy["accepted"] and attempt["adjudication"]["deterministic_passed"]
        assert attempt["adjudication"]["contradictions_retained"] == 2
    # With nothing left out, the option delivers the same bytes.
    for case in ("ferry-kept", "ferry-uncontested"):
        baseline, policy = _arms(record, "contradiction_synthesis", case)
        assert baseline["accepted"] and policy["accepted"]
        assert baseline["attempts"][0]["output_sha256"] == policy["attempts"][0]["output_sha256"]
    # A merge's reply is delivered as written, so strict acceptance still rejects it.
    baseline, policy = _arms(record, "contradiction_synthesis", "ferry-merge-dropped")
    assert not baseline["accepted"] and not policy["accepted"]
    ablation = record["ablations"]["contradiction_synthesis"]
    for arm in ARMS:
        assert ablation["summary"][arm]["false_acceptances"] == 0
        for case in ablation["cases"]:
            for attempt in case["results"][arm]["attempts"]:
                assert attempt["policy_sha256"] == ablation["acceptance_policy"]["sha256"]


def test_the_matched_graph_delivers_identical_bytes_and_reports_store_time_apart(offline_run):
    record, _ = offline_run
    overhead = record["scheduler_overhead"]
    assert overhead["outputs_identical"] is True
    assert {run["output_sha256"] for run in overhead["runs"]} == {overhead["output_sha256"]}
    assert overhead["latencies_ms"] == list(OVERHEAD_LATENCIES_MS)
    assert overhead["graph"]["nodes"] == ["research", "analyze", "write"]
    assert "machine-specific" in overhead["note"]
    for latency in OVERHEAD_LATENCIES_MS:
        runs = {run["executor"]: run for run in overhead["runs"] if run["latency_ms"] == latency}
        assert set(runs) == set(OVERHEAD_EXECUTORS)
        assert all(run["wire_calls"]["response"] == 3 for run in runs.values())
        assert all(set(run["nodes_ms"]) == set(overhead["graph"]["nodes"])
                   for run in runs.values())
        # Only the Swarms use stores; the durable one journals every call.
        assert runs["reference"]["store_ms"] == {"checkpoint": 0.0, "journal": 0.0}
        assert runs["ordinary"]["store_calls"]["checkpoint"] > 0
        assert "journal" not in runs["ordinary"]["store_calls"]
        assert runs["durable"]["store_calls"]["journal"] > 0
        assert runs["durable"]["store_calls"]["checkpoint"] > 0
        summary = overhead["summary"][str(latency)]
        assert summary["reference"]["median_overhead_ms"] == 0
        assert set(summary) == set(OVERHEAD_EXECUTORS)
        # The durable runtime also counts each node's input tokens on the wire; those
        # calls wait the declared latency too, and its overhead figures include it.
        assert runs["durable"]["wire_calls"] == {"response": 3, "input_tokens": 3}
        assert {executor: entry["extra_wire_latency_ms"] for executor, entry in summary.items()
                } == {"reference": 0, "ordinary": 0, "durable": 3 * latency}
    assert "extra_wire_latency_ms" in overhead["overhead"]


def test_every_policy_stays_experimental_and_off_by_default(offline_run):
    record, _ = offline_run
    names = [item["feature"] for item in record["features"]]
    assert {"TerminationPolicy", "CapacityPolicy", "Swarm(reserve_for_checks=True)",
            "Synthesizer(retain_contradictions=True)"} <= set(names)
    for item in record["features"]:
        assert item["status"] == "experimental" and item["default"] == "off"
        assert item["promotion"].startswith("not eligible")


def test_the_committed_record_is_labeled_and_never_cited_as_a_claim():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["benchmark"] == "policy-ablations"
    assert record["evidence_status"] == "offline-mechanics" and record["claimable"] is False
    assert record["api_calls"] == 0 and record["default_change"] is False
    assert all(case["differs_in"] == [ABLATIONS[name]]
               for name, ablation in record["ablations"].items() for case in ablation["cases"])
    assert record["scheduler_overhead"]["outputs_identical"] is True
    for name in ("README.md", "ROADMAP.md"):
        assert "policy_ablations" not in (ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The harness's own pieces
# ---------------------------------------------------------------------------


def test_arms_may_differ_only_in_the_varied_policy():
    arms = ablation_arms({"graph": "fork"}, "termination", None, {"drain_timeout_s": 1.0})
    assert arms == {"baseline": {"graph": "fork", "termination": None},
                    "policy": {"graph": "fork", "termination": {"drain_timeout_s": 1.0}}}
    with pytest.raises(ValueError, match="base configuration cannot set it"):
        ablation_arms({"termination": None}, "termination", None, {"drain_timeout_s": 1.0})
    with pytest.raises(ValueError, match="differ in 'termination' alone"):
        ablation_arms({"graph": "fork"}, "termination", None, None)
    assert one_change({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == ["b", "c"]
    assert one_change({"a": None}, {}) == ["a"]


def test_correlated_errors_group_consequences_and_repeated_signatures():
    error = {"type": "RateLimitError", "message": "Error code: 429"}
    failures = [
        {"node": "a", "attempt": 1, "outcome": "failed", "error": error,
         "failure_class": "transient", "caused_by": None},
        {"node": "b", "attempt": 1, "outcome": "failed", "error": error,
         "failure_class": "transient", "caused_by": None},
        {"node": "c", "attempt": 1, "outcome": "failed", "error": {"type": "RuntimeError",
                                                                   "message": "other"},
         "failure_class": "transient", "caused_by": None},
        {"node": "d", "attempt": 1, "outcome": "cancelled", "error": None,
         "failure_class": None, "caused_by": "a"},
    ]
    assert correlated_errors(failures) == [
        {"kind": "shared_signature",
         "signature": {"outcome": "failed", "error_type": "RateLimitError",
                       "failure_class": "transient", "message": "Error code: 429"},
         "failures": [{"node": "a", "attempt": 1}, {"node": "b", "attempt": 1}], "count": 2},
        {"kind": "consequence", "cause": "the failure of 'a'",
         "failures": [{"node": "d", "attempt": 1}], "count": 1},
    ]
    assert correlated_errors(failures[2:3]) == [] and correlated_errors([]) == []


def test_virtual_time_moves_only_to_the_next_wake_time():
    time = VirtualTime()
    woke: list[tuple[str, int, int]] = []

    def record(name: str) -> None:
        # Real time passes here; on a worker thread the clock would move on meanwhile.
        pytime.sleep(0.02)
        woke.append((name, time.now_ms(), threading.get_ident()))

    async def sleeper(name: str, seconds: float) -> None:
        await time.sleep(seconds)
        # Work handed to a thread runs at once, on the loop's own thread, so it
        # finishes before the clock moves again.
        await asyncio.to_thread(record, name)

    async def run() -> str:
        await asyncio.gather(sleeper("late", 1.5), sleeper("early", 0.25), sleeper("tie", 1.5))
        return "done"

    assert time.run(run()) == "done"
    loop_thread = threading.get_ident()
    assert woke == [("early", 250, loop_thread), ("late", 1500, loop_thread),
                    ("tie", 1500, loop_thread)]
    assert time.now_ms() == 1500

    async def stuck() -> None:
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="no virtual timer ends"):
        VirtualTime().run(stuck())


def test_manual_clock_arms_repeat_exactly():
    drain = ablation_arms({"executor": "AsyncExecutor", "max_concurrency": 2,
                           "max_budget_usd": 10.0, "node_estimate_usd": 0.25},
                          "termination", None, {"drain_timeout_s": DRAIN_TIMEOUT_S})["policy"]
    first = run_drain_ordinary_arm("answers_within_drain", drain)
    assert run_drain_ordinary_arm("answers_within_drain", drain) == first
    assert first["drain"]["settled"] == ["sibling"]
    capacity = {"nodes": 8, "max_concurrency": 8, "provider_quota_in_flight": QUOTA_IN_FLIGHT,
                "max_retries": 4, "retry_policy": {"backoff_max_s": 8.0}, "retry_backoff_s": 1.0,
                "max_budget_usd": 10.0, "capacity_policy": None}
    first = run_capacity_arm(capacity)
    assert run_capacity_arm(capacity) == first
    assert first["rejected_429"] > 0


def test_budget_hold_arms_repeat_exactly():
    # Each arm admits its nodes in one order, so every run makes the same calls
    # and spends the same, however the threads that finish each node are timed.
    cases = {case["case"]: case for case in budget_holds()["cases"]}
    for name, case in cases.items():
        for arm, config in case["arms"].items():
            runs = [case["results"][arm]] + [run_hold_arm(name, config)
                                             for _ in range(HOLD_REPEATS - 1)]
            assert [(run["calls"], run["spend_usd"]) for run in runs] == [
                (runs[0]["calls"], runs[0]["spend_usd"])] * HOLD_REPEATS, (name, arm)
            assert runs == [runs[0]] * HOLD_REPEATS, (name, arm)
    # Under holds, the short budget pays seven workers and refuses the eighth before its call.
    policy = cases["fan_out_short_budget"]["results"]["policy"]
    assert policy["calls"] == [f"w{index}" for index in range(7)]
    assert policy["spend_usd"] == 0.875 and policy["refused"]["node"] == "w7"
