"""Independent offline reconciliation; fixture data is never live evidence."""

import hashlib
import json
import zipfile

import pytest

from benchmarks import astra_main_evidence as evidence
from test_astra_analysis import complete  # noqa: F401


def request_fixture(trial, cost):
    policy = evidence.runtime.POLICY
    return {"request_json": json.dumps({"model": trial["model"], "max_output_tokens": policy["max_output_tokens"],
            "reasoning": {"effort": policy["reasoning_effort"]}, "service_tier": policy["service_tier"]}),
        "tool_names_json": "{}", "provider": {"kind": "openai_responses", "endpoint_scope": "global"},
        "cost_nanousd": cost, "billing_state": "unknown" if cost is None else "known",
        "receipt": None if cost is None else {"requested_model": trial["model"],
            "actual_model": trial["model"], "service_tier": "default", "endpoint_scope": "global", "cost_nanousd": cost}}


@pytest.fixture
def campaign(complete, tmp_path, monkeypatch):  # noqa: F811
    freeze, rows, judgments, bindings = complete
    freeze.update({"stage_allowance_nanousd": 200_000_000_000,
                   "policy": dict(evidence.runtime.POLICY),
                   "campaign_allocations": {"main_nanousd": 200_000_000_000},
                   "human_calibration": {"status": "passed"},
                   "tasks": {r["trial"]["task_id"]: {"goal": "fixture"} for r in rows},
                   "source_sha256": {"example.py": hashlib.sha256(b"# fixture\n").hexdigest()}})
    for row in rows:
        row["accounting"]["calls"] = [{"key": {"phase": "execution", "scope_id": "execution"},
                                      "cost_nanousd": row["accounting"]["confirmed_nanousd"]}]
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("example.py", b"# fixture\n")
    monkeypatch.setattr(evidence, "load_complete_main", lambda _: (freeze, rows))
    monkeypatch.setattr(evidence, "inspect_judgments", lambda _: judgments)
    monkeypatch.setattr(evidence.runtime, "_read", lambda _: bindings)
    state = {"task": {"goal": "fixture"}}

    class Store:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load_run(self, _):
            return state

        def lookup_call(self, run_id, key):
            row = next(r for r in rows if r["run_id"] == run_id)
            call = next(c for c in row["accounting"]["calls"] if c["key"]["scope_id"] == key.scope_id)
            return request_fixture(row["trial"], call["cost_nanousd"])

    monkeypatch.setattr(evidence, "SQLiteWorkflowStore", Store)
    return {"main_directory": tmp_path, "judge_directory": tmp_path,
            "bindings_path": tmp_path / "bindings.json", "source_archive": archive}, rows, state


def test_reconciles_every_input_phase_and_charge_without_promoting_a_claim(campaign):
    args, rows, _ = campaign
    rows[0]["checkpoint"] = {"checkpoint": {"graph": {"nodes": [{}, {}]}}}
    analysis, review = evidence.review_main(**args)
    assert review["validated_outcomes"] == 200 and review["validated_task_inputs"] == 10
    assert review["main_workflow_nanousd"] == 150_000_000_000
    assert not review["disputed_output_run_ids"] and review["claimable"] is False
    assert analysis["phase_generation_calls"]["astra-fixed"] == {"execution": 50}
    assert review["validated_request_policies"] == 200
    assert analysis["graph_sizes"][rows[0]["trial"]["arm_id"]]["node_counts"] == [2]
    assert analysis["task_graph_sizes"][rows[0]["trial"]["task_id"]][rows[0]["trial"]["arm_id"]] == [2]


def test_wrong_frozen_task_is_rejected(campaign):
    args, _, state = campaign
    state["task"] = {"goal": "different"}
    with pytest.raises(ValueError, match="Executed task differs"):
        evidence.review_main(**args)


def test_missing_phase_charge_is_rejected(campaign):
    args, rows, _ = campaign
    rows[0]["accounting"]["calls"] = []
    with pytest.raises(ValueError, match="phase charges"):
        evidence.review_main(**args)


def test_wrong_archived_source_is_rejected(campaign):
    args, _, _ = campaign
    with zipfile.ZipFile(args["source_archive"], "w") as source:
        source.writestr("example.py", "changed")
    with pytest.raises(ValueError, match="Frozen source differs"):
        evidence.review_main(**args)


def test_continuation_phase_audit_retains_unknown_as_a_bound(campaign, tmp_path, monkeypatch):
    args, rows, _ = campaign
    freeze, _ = evidence.load_complete_main(args["main_directory"])
    failed = next(r for r in rows if r["trial"]["arm_id"] == "sol-dynamic")
    failed.update({"output": None, "output_sha256": None, "status": "failed"})
    failed["deterministic_checks"]["deterministic_passed"] = False
    failed["accounting"].update({"unknown_nanousd": 200_000_000, "unknown_calls": 1})
    failed["accounting"]["calls"].append({"cost_nanousd": None, "unknown_nanousd": 200_000_000,
                                           "key": {"phase": "execution", "scope_id": "unknown"}})
    bindings = evidence.runtime._read(args["bindings_path"])
    bindings[:] = [r for r in bindings if r["run_id"] != failed["run_id"]]
    rows[-1]["continuation_freeze_sha256"] = "continued"
    source_body = b"# offline continuation fixture\n"
    summary = {"confirmed_nanousd": 150_000_000_000, "unknown_nanousd": 200_000_000,
               "continuation_source_sha256": hashlib.sha256(source_body).hexdigest()}
    monkeypatch.setattr(evidence, "load_continued_main", lambda *a: (freeze, rows, dict(summary)))
    source = tmp_path / "continuation-source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("example.py", b"# fixture\n")
        archive.writestr("benchmarks/astra_continuation.py", source_body)
    args.update(continuation_directory=tmp_path / "continued", continuation_source_archive=source)
    analysis, review = evidence.review_main(**args)
    assert review["main_workflow_nanousd"] is None
    assert review["main_workflow_lower_nanousd"] == 150_000_000_000
    assert review["main_workflow_upper_nanousd"] == 150_200_000_000
    assert review["unknown_workflow_usage"] == 1
    assert review["missing_output_failure_run_ids"] == [failed["run_id"]]
    assert review["disputed_output_run_ids"] == []
    assert analysis["phase_unknown_nanousd"]["sol-dynamic"]["execution"] == 200_000_000
    failed["accounting"]["calls"][-1]["unknown_nanousd"] = 0
    with pytest.raises(ValueError, match="phase charges"):
        evidence.review_main(**args)


def test_continuation_requires_its_source_archive_before_analysis(campaign):
    args, _, _ = campaign
    with pytest.raises(ValueError, match="frozen source archive"):
        evidence.review_main(**args, continuation_directory="continued")


@pytest.mark.parametrize("change", ["model", "cap", "effort", "tools", "sampling", "actual-model", "charge"])
def test_actual_native_request_or_receipt_mismatch_is_rejected(change):
    trial = {"model": "gpt-6-astra"}
    call = request_fixture(trial, 100)
    request = json.loads(call["request_json"])
    if change == "model":
        request["model"] = "gpt-5.6-sol"
    elif change == "cap":
        request["max_output_tokens"] = 4096
    elif change == "effort":
        request["reasoning"]["effort"] = "high"
    elif change == "tools":
        request["tools"] = [{"type": "web_search"}]
    elif change == "sampling":
        request["temperature"] = 0
    elif change == "actual-model":
        call["receipt"]["actual_model"] = "different"
    else:
        call["receipt"]["cost_nanousd"] = 99
    call["request_json"] = json.dumps(request)
    with pytest.raises(ValueError, match="frozen"):
        evidence.review_request(call, trial, evidence.runtime.POLICY)


def test_no_response_diagnostic_is_retained_without_inventing_returned_identity():
    trial = {"model": "gpt-5.6-sol"}
    call = request_fixture(trial, None)
    call["receipt"] = {"requested_model": trial["model"], "actual_model": None,
        "endpoint_scope": "global", "service_tier": None, "cost_nanousd": None,
        "cost_is_complete": False, "usage_is_complete": False, "accounting_error": "No HTTP response"}
    evidence.review_request(call, trial, evidence.runtime.POLICY)
    assert call["receipt"]["cost_nanousd"] is None and call["receipt"]["actual_model"] is None
    call["receipt"]["cost_nanousd"] = 0
    with pytest.raises(ValueError, match="Unknown diagnostic"):
        evidence.review_request(call, trial, evidence.runtime.POLICY)
