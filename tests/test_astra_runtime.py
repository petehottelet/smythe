"""Offline native-transport qualification of the separate, gated pilot runtime."""

from contextlib import closing
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks import astra_runtime as runtime
from benchmarks.astra_campaign import prepare_campaign
from benchmarks.astra_campaign._json import canonical
from smythe import Task
from smythe.planner import ArchitectError
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.task import task_to_dict
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_provider import WorkflowQuoteError
from smythe.workflow_store import SQLiteWorkflowStore

ALLOWANCES = {"total_nanousd": 200_000_000_000, "pilot_nanousd": 120_000_000_000,
              "main_nanousd": 60_000_000_000, "judge_nanousd": 20_000_000_000,
              "per_trial_nanousd": 10_000_000_000}


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(requests=[], quotes=[], plans=[], invalid_usage=False, clients=[],
                            require_parallel=False, active=0, maximum_active=0, parallel_gate=None,
                            before_generate=None, invalid_quote=False)
    monkeypatch.setattr(runtime, "_dependencies", lambda: {
        "python": "offline-contract-python", "openai": "offline-contract-sdk", "smythe": "0.6.0",
        "httpx": None, "httpx2": None, "pyyaml": "offline-contract-yaml",
    })

    def wire(body):
        return SimpleNamespace(content=json.dumps(body).encode(), status_code=200,
                               headers={"x-request-id": "offline-request"})

    async def count(**payload):
        state.quotes.append(payload)
        return wire({"input_tokens": False if state.invalid_quote else 100})

    async def generate(**payload):
        if state.before_generate is not None:
            hook, state.before_generate = state.before_generate, None
            hook()
        state.requests.append(payload)
        if payload["instructions"] == PLANNING_SYSTEM_PROMPT:
            plan = state.plans.pop(0) if state.plans else {
                "topology": ["serial"], "nodes": [
                    {"id": "analysis", "label": "Analyze supplied evidence", "max_retries": 0, "max_regenerations": 0},
                    {"id": "answer", "label": "Write final bare JSON", "depends_on": ["analysis"],
                     "max_retries": 0, "max_regenerations": 0},
                ],
            }
            text = plan if isinstance(plan, str) else json.dumps(plan)
        else:
            text = '{"offline_output":"This is a deliberately unevaluated fixture answer."}'
            if state.require_parallel:
                if state.parallel_gate is None:
                    state.parallel_gate = asyncio.Event()
                state.active += 1
                state.maximum_active = max(state.maximum_active, state.active)
                if state.active == 2:
                    state.parallel_gate.set()
                try:
                    await asyncio.wait_for(state.parallel_gate.wait(), timeout=30)
                finally:
                    state.active -= 1
        return wire({
            "id": f"resp_{len(state.requests)}", "model": payload["model"],
            "status": "completed", "service_tier": "default",
            "usage": None if state.invalid_usage else {
                "input_tokens": 100, "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
            "output": [{"type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
        })

    def client(self):
        if self._client is not None:
            return self._client
        value = SimpleNamespace(
            base_url="https://api.openai.com/v1/", max_retries=0, close=AsyncMock(),
            responses=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=AsyncMock(side_effect=generate)),
                input_tokens=SimpleNamespace(with_raw_response=SimpleNamespace(count=AsyncMock(side_effect=count))),
            ),
        )
        state.clients.append(value)
        return value

    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", client)
    return state


def freeze(directory="unexecuted-test-campaign"):
    return runtime.freeze_runtime(allowances=ALLOWANCES, directory=directory)


def run(value, directory):
    return asyncio.run(runtime.run_pilot(value, directory=directory, approval=value["approval_token"]))


def test_runtime_freeze_is_separate_and_preserves_the_original_preparation_receipt(native):
    before = prepare_campaign()
    value = freeze()
    receipt = json.loads((runtime.ROOT / "benchmarks/results/astra_preparation_20260907.json").read_text())
    assert prepare_campaign() == before == receipt
    assert value["preparation_sha256"] == receipt["preparation_sha256"]
    assert value["schedule_sha256"] == receipt["schedule_sha256"]
    assert set(runtime.RUNTIME_FILES) <= value["source_sha256"].keys()
    assert value["policy"]["planning_repairs"] == value["policy"]["node_max_retries"] == 0
    assert value["main_enabled"] is value["judge_enabled"] is value["claimable"] is False
    assert value["api_calls"] == 0 and not native.requests and not native.quotes
    unsigned = dict(value)
    unsigned.pop("approval_token")
    assert unsigned.pop("freeze_sha256") == runtime._sha(unsigned)
    value["policy"]["sampling_parameters"].append("mutated")
    assert freeze()["policy"]["sampling_parameters"] == []


@pytest.mark.parametrize("key", sorted(runtime.ALLOWANCE_KEYS))
@pytest.mark.parametrize("invalid", [None, True, False, -1, 1.0, "1", 2**63])
def test_every_allowance_is_a_strict_bounded_integer(native, key, invalid):
    value = dict(ALLOWANCES, **{key: invalid})
    with pytest.raises(runtime.CampaignRuntimeError):
        runtime.freeze_runtime(allowances=value)
    assert not native.requests


@pytest.mark.parametrize("change", [
    {"total_nanousd": 1}, {"pilot_nanousd": 1}, {"per_trial_nanousd": 0},
    {"main_nanousd": 200_000_000_000}, {"judge_nanousd": 200_000_000_000},
])
def test_stage_and_trial_caps_cannot_exceed_the_total(native, change):
    with pytest.raises(runtime.CampaignRuntimeError):
        runtime.freeze_runtime(allowances={**ALLOWANCES, **change})


def test_nonrepresentable_float_budget_is_rejected_without_inflation():
    with pytest.raises(runtime.CampaignRuntimeError, match="represented"):
        runtime._budget_usd(9_223_372_036_854_775_807)
    assert runtime._budget_usd(1_234_567_890) == 1.23456789


@pytest.mark.parametrize("kind", ["unfunded", "approval", "policy", "source", "main", "judge"])
def test_unapproved_and_changed_runtime_cannot_initialize_a_provider_or_create_directory(
    tmp_path, monkeypatch, native, kind,
):
    target = tmp_path / "not-created"
    value = runtime.freeze_runtime(directory=target) if kind == "unfunded" else freeze(target)
    approval = value["approval_token"]
    if kind == "approval":
        approval = "some-other-approval"
    elif kind == "policy":
        value["policy"]["max_nodes"] = 9
    elif kind == "source":
        value["source_sha256"]["smythe/workflow.py"] = "0" * 64
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("provider construction before admission"))
    with pytest.raises(runtime.CampaignRuntimeError):
        if kind in {"main", "judge"}:
            getattr(runtime, "run_" + kind)(value, directory=target, approval=approval)
        else:
            asyncio.run(runtime.run_pilot(value, directory=target, approval=approval))
    assert not target.exists()


def test_complete_pilot_uses_every_arm_native_ledger_and_frozen_request_then_never_rebuys(
    tmp_path, monkeypatch, native,
):
    value = freeze(tmp_path / "campaign")
    caller_owned = deepcopy(value)
    native.before_generate = lambda: caller_owned["allowances"].update(per_trial_nanousd=0)
    result = run(caller_owned, tmp_path / "campaign")
    assert caller_owned["allowances"]["per_trial_nanousd"] == 0
    assert result["workflow_runs"] == 12 and result["failed_workflows"] == 0
    assert len(native.requests) == len(native.quotes) == 36
    assert result["confirmed_nanousd"] == 37_800_000
    assert result["judge_nanousd"] == 0
    saved_summary = runtime._read(tmp_path / "campaign/pilot-summary.json")
    assert len(saved_summary["outcome_receipts"]) == 12 and "outcomes" not in saved_summary
    assert result["accepted"] is None and result["quality_evaluated"] is result["claimable"] is False
    assert [item["trial"] for item in result["outcomes"]] == prepare_campaign()["schedules"]["pilot"]
    assert all(item["latency_complete"] and item["wall_time_ns"] > 0 for item in result["outcomes"])
    for item in result["outcomes"]:
        assert item["output"] and item["checkpoint"] and item["trace"]
        assert item["deterministic_checks"]["accepted"] is None
        assert item["accounting"]["unknown_calls"] == item["accounting"]["reserved_nanousd"] == 0
        assert all(call["billing_state"] == "known" for call in item["accounting"]["calls"])
        assert all(call["result_state"] == "applied" for call in item["accounting"]["calls"])
    for request in native.requests:
        assert request["model"] in {"gpt-6-astra", "gpt-5.6-sol"}
        assert request["reasoning"] == {"effort": "medium"}
        assert request["max_output_tokens"] == 8192 and request["service_tier"] == "default"
        assert not {"tools", "temperature", "top_p", "logprobs"} & request.keys()
        prompt = canonical(request)
        assert runtime.PLANNING_CONSTRAINT in prompt
        assert '"rubric"' not in prompt and '"checks"' not in prompt
        assert "criterion-1" not in prompt and "Absent or contradicts" not in prompt
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("completed outcome reconstructed a provider"))
    assert run(value, tmp_path / "campaign") == result
    with pytest.raises(runtime.CampaignRuntimeError, match="different campaign directory"):
        run(value, tmp_path / "second-campaign")
    assert not (tmp_path / "second-campaign").exists()
    assert len(native.requests) == 36


def test_fixed_builder_is_fresh_local_exactly_three_ordered_steps():
    first, registry = runtime._FixedArchitect().plan(Task("Answer"))
    second, _ = runtime._FixedArchitect().plan(Task("Answer"))
    assert [node.id for node in first.nodes] == ["research", "analysis", "writing"]
    assert [node.depends_on for node in first.nodes] == [[], ["research"], ["analysis"]]
    assert all(node.max_retries == node.max_regenerations == 0 for node in first.nodes)
    assert [agent.id for agent in registry.list_agents()] == ["research", "analysis", "writing"]
    first.nodes[0].label = "mutation"
    assert second.nodes[0].label != "mutation"


def test_dynamic_branches_actually_enter_provider_concurrently(tmp_path, native):
    native.require_parallel = True
    native.plans = [{"topology": ["fork_join"], "nodes": [
        {"id": "left", "label": "Left", "max_retries": 0},
        {"id": "right", "label": "Right", "max_retries": 0},
        {"id": "answer", "label": "Answer", "depends_on": ["left", "right"], "max_retries": 0},
    ]}]
    row = {"model": "gpt-6-astra", "strategy": "smythe_dynamic"}
    with SQLiteWorkflowStore(tmp_path / "parallel.db") as store:
        swarm = runtime._swarm(store, row, ALLOWANCES["per_trial_nanousd"])
        assert swarm.parallel and swarm.max_concurrency == 8
        store.create_run(task_to_dict(Task("Answer")), swarm._workflow_runtime().recipe,
                         ALLOWANCES["per_trial_nanousd"], run_id="parallel")
        result = asyncio.run(swarm.aresume("parallel"))
        assert result.output
    assert native.maximum_active == 2


@pytest.mark.parametrize("violation", ["nodes", "model", "retries", "regenerations", "parse"])
def test_dynamic_policy_rejects_invalid_planning_before_any_execution(tmp_path, native, violation):
    node = {"id": "answer", "label": "Answer", "max_retries": 0, "max_regenerations": 0}
    plan = {"topology": ["serial"], "nodes": [node]}
    if violation == "nodes":
        plan["nodes"] = [{**node, "id": f"n{index}"} for index in range(9)]
    elif violation == "model":
        node["metadata"] = {"model": "gpt-5.6-sol"}
    elif violation == "retries":
        node["max_retries"] = 1
    elif violation == "regenerations":
        node["max_regenerations"] = 1
    else:
        plan = "invalid planning JSON"
    native.plans = [plan]
    row = {"model": "gpt-6-astra", "strategy": "smythe_dynamic"}
    with SQLiteWorkflowStore(tmp_path / "trial.db") as store:
        swarm = runtime._swarm(store, row, ALLOWANCES["per_trial_nanousd"])
        recipe = swarm._workflow_runtime().recipe
        assert recipe["graph_policy"] == {"version": 1, "max_nodes": 8, "node_model": "gpt-6-astra",
                                          "max_retries": 0, "max_regenerations": 0}
        assert recipe["components"]["architect"]["max_retries"] == 0
        store.create_run(task_to_dict(Task("Answer")), recipe, ALLOWANCES["per_trial_nanousd"], run_id="trial")
        with pytest.raises((WorkflowBindingError, ArchitectError)):
            swarm.resume("trial")
        evidence = store.inspect_run("trial")
        assert evidence["confirmed_nanousd"] == 1_500_000
        assert evidence["call_count"] == 1
        assert evidence["calls"][0]["key"]["phase"] == "planning"
        assert store.get_checkpoint("trial") is None
    assert len(native.requests) == 1


def test_failed_planning_remains_an_included_charged_outcome(tmp_path, native):
    native.plans = ["bad JSON"]
    result = run(freeze(tmp_path / "campaign"), tmp_path / "campaign")
    assert result["workflow_runs"] == 12 and result["failed_workflows"] == 1
    failed = next(item for item in result["outcomes"] if item["status"] == "failed")
    assert failed["error"]["type"] == "ArchitectError" and failed["output"] is None
    assert failed["accounting"]["call_count"] == 1
    assert failed["accounting"]["confirmed_nanousd"] > 0
    assert failed["deterministic_checks"]["accepted"] is None
    assert len(list((tmp_path / "campaign").glob("*.outcome.json"))) == 12


def test_unknown_usage_preserves_failure_and_blocks_next_trial_and_future_rebuy(tmp_path, native, monkeypatch):
    native.invalid_usage = True
    value = freeze(tmp_path / "campaign")
    directory = tmp_path / "campaign"
    with pytest.raises(runtime.CampaignRuntimeError, match="spending blocks"):
        run(value, directory)
    records = list(directory.glob("*.outcome.json"))
    assert len(records) == 1
    failed = runtime._read(records[0])
    assert failed["status"] == "failed" and failed["error"]
    assert failed["accounting"]["unknown_calls"] == 1
    assert failed["accounting"]["unknown_nanousd"] > 0
    diagnostics = list(directory.glob("failure-*.json"))
    assert len(diagnostics) == 1
    assert runtime._read(diagnostics[0])["error"]["type"] == "CampaignRuntimeError"
    assert len(native.requests) == 1
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("unknown outcome was rebought"))
    with pytest.raises(runtime.CampaignRuntimeError, match="spending blocks"):
        run(value, directory)
    assert len(native.requests) == 1


def test_interruption_after_workflow_completion_recovers_same_run_without_rebuy(tmp_path, native, monkeypatch):
    value = freeze(tmp_path / "campaign")
    directory = tmp_path / "campaign"
    write = runtime._write_new
    interrupted = False

    def interrupt_outcome(path, record):
        nonlocal interrupted
        if path.name.endswith(".outcome.json") and not interrupted:
            interrupted = True
            raise OSError("simulated process loss before outcome publication")
        return write(path, record)

    monkeypatch.setattr(runtime, "_write_new", interrupt_outcome)
    with pytest.raises(OSError, match="process loss"):
        run(value, directory)
    assert len(native.requests) == 3
    with SQLiteWorkflowStore(directory / "workflow.sqlite3", read_only=True) as store:
        first = store.list_runs()
        assert len(first) == 1 and first[0]["status"] == "completed"
    monkeypatch.setattr(runtime, "_write_new", write)
    result = run(value, directory)
    assert len(native.requests) == 36
    assert result["outcomes"][0]["run_id"] == first[0]["run_id"]
    assert result["outcomes"][0]["resumed"]
    assert result["outcomes"][0]["wall_time_ns"] is None
    assert result["outcomes"][0]["latency_complete"] is False


def test_campaign_writer_lock_refuses_a_competitor_before_provider_initialization(tmp_path, native, monkeypatch):
    value = freeze(tmp_path / "campaign")
    directory = tmp_path / "campaign"
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("competing provider"))
    with runtime._campaign_lock(directory):
        with pytest.raises(runtime.CampaignRuntimeError, match="writer lock"):
            run(value, directory)
    assert not (directory / "workflow.sqlite3").exists()


def test_different_approved_allowance_cannot_reuse_a_campaign_directory(tmp_path, native, monkeypatch):
    first = freeze(tmp_path / "campaign")
    rows = prepare_campaign()["schedules"]["pilot"]
    directory = tmp_path / "campaign"
    directory.mkdir()
    runtime._write_new(directory / "campaign.json", runtime._binding(first, rows))
    second = runtime.freeze_runtime(allowances={**ALLOWANCES, "total_nanousd": 210_000_000_000}, directory=directory)
    assert runtime._run_id(first, rows[0]) != runtime._run_id(second, rows[0])
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("changed campaign constructed a provider"))
    with pytest.raises(runtime.CampaignRuntimeError, match="different runtime freeze"):
        run(second, directory)


def test_unbound_saved_run_is_rejected_before_next_provider(tmp_path, native, monkeypatch):
    value = freeze(tmp_path / "campaign")
    rows = prepare_campaign()["schedules"]["pilot"]
    directory = tmp_path / "campaign"
    directory.mkdir()
    runtime._write_new(directory / "campaign.json", runtime._binding(value, rows))
    with SQLiteWorkflowStore(directory / "workflow.sqlite3") as store:
        store.create_run({}, {}, ALLOWANCES["per_trial_nanousd"], run_id="foreign-run")
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("unbound ledger dispatched"))
    with pytest.raises(runtime.CampaignRuntimeError, match="unbound trial"):
        run(value, directory)


def test_source_inventory_uses_tracked_runtime_and_explicit_files_not_scratch_or_git_status(tmp_path, monkeypatch):
    for name in (*runtime.RUNTIME_FILES, "pyproject.toml", "smythe/workflow.py", "smythe/provider_responses.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "add", "smythe/workflow.py", "smythe/provider_responses.py"],
                   cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    before = runtime._source_hashes()
    (tmp_path / "smythe/scratch.py").write_text("# ignored by source inventory\n")
    assert runtime._source_hashes() == before
    (tmp_path / "smythe/workflow.py").write_text("# uncommitted runtime change\n")
    assert runtime._source_hashes()["smythe/workflow.py"] != before["smythe/workflow.py"]


def test_outcome_publication_is_exclusive_and_cli_never_opens_main_or_judge(tmp_path, capsys):
    path = tmp_path / "record.json"
    runtime._write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        runtime._write_new(path, {"replacement": True})
    assert runtime._read(path) == {"original": True}
    assert runtime.main(["main"]) == runtime.main(["judge"]) == 1
    assert "human-calibrated" in capsys.readouterr().err


def test_freeze_cli_without_spending_amounts_writes_only_blocked_offline_evidence(tmp_path, native, capsys):
    path = tmp_path / "freeze.json"
    assert runtime.main(["freeze", "--out", str(path)]) == 0
    value = runtime._read(path)
    assert value["allowances"] is None and value["api_calls"] == 0
    assert value["claimable"] is False and not native.requests
    assert json.loads(capsys.readouterr().out) == value


def test_saved_outcome_integrity_rejects_tampering_before_ledger_reads(native):
    value = freeze()
    row = prepare_campaign()["schedules"]["pilot"][0]
    record = {"freeze_sha256": value["freeze_sha256"], "trial": deepcopy(row),
              "run_id": runtime._run_id(value, row), "status": "failed"}
    record["record_sha256"] = runtime._sha(record)
    record["status"] = "completed"
    with pytest.raises(runtime.CampaignRuntimeError, match="different identity or content"):
        runtime._validate_outcome(record, value, row, None)


def test_funded_freeze_requires_its_destination_before_approval(native):
    with pytest.raises(runtime.CampaignRuntimeError, match="exact campaign directory"):
        runtime.freeze_runtime(allowances=ALLOWANCES)
    assert not native.requests


@pytest.mark.parametrize("location", ["ancestor", "lock", "ledger", "sidecar", "receipt"])
def test_reparse_ancestry_and_fixed_paths_are_rejected_before_foreign_ledger_or_provider(
    tmp_path, monkeypatch, native, location,
):
    directory = tmp_path / "campaign"
    value = freeze(directory)
    row = prepare_campaign()["schedules"]["pilot"][0]
    directory.mkdir()
    runtime._write_new(directory / "campaign.json", runtime._binding(value, prepare_campaign()["schedules"]["pilot"]))
    blocked = {
        "ancestor": tmp_path, "lock": directory / "campaign-lock.sqlite3",
        "ledger": directory / "workflow.sqlite3", "sidecar": directory / "workflow.sqlite3-wal",
        "receipt": directory / f"{runtime._run_id(value, row)}.outcome.json",
    }[location]
    original_lstat, original_connect = Path.lstat, sqlite3.connect
    opened = []

    def lstat(path, *args, **kwargs):
        if path.absolute() == blocked:
            return SimpleNamespace(st_mode=stat.S_IFDIR if location == "ancestor" else stat.S_IFREG,
                                   st_file_attributes=0x400)
        return original_lstat(path, *args, **kwargs)

    def connect(path, *args, **kwargs):
        opened.append(Path(path))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(runtime, "_swarm", lambda *args: pytest.fail("linked campaign initialized provider"))
    with pytest.raises(runtime.CampaignRuntimeError, match="reparse"):
        run(value, directory)
    assert blocked not in opened and not native.requests


@pytest.fixture
def recorded_trial(tmp_path, native):
    directory = tmp_path / "campaign"
    value = freeze(directory)
    directory.mkdir()
    row = prepare_campaign()["schedules"]["pilot"][0]
    case = next(case for case in runtime.load_task_pack().for_stage("pilot") if case.task_id == row["task_id"])
    run_id = runtime._run_id(value, row)
    path = directory / "workflow.sqlite3"
    with SQLiteWorkflowStore(path) as store:
        started = time.perf_counter_ns()
        swarm = runtime._swarm(store, row, ALLOWANCES["per_trial_nanousd"])
        store.create_run(task_to_dict(runtime._task(case)), swarm._workflow_runtime().recipe,
                         ALLOWANCES["per_trial_nanousd"], run_id=run_id)
        result = swarm.resume(run_id)
        record = runtime._outcome(store, value, row, run_id, result, None, started, False)
        record["deterministic_checks"] = runtime.check_output(case, record["output"])
        record.pop("record_sha256")
        record["record_sha256"] = runtime._sha(record)
        runtime._validate_outcome(record, value, row, store, case)
    return path, value, row, record, case


@pytest.mark.parametrize("change", ["checkpoint", "response", "decoded-result", "outcome-output"])
def test_retained_outcome_reconciles_against_current_immutable_evidence(recorded_trial, change):
    path, value, row, record, case = recorded_trial
    with SQLiteWorkflowStore(path) as store:
        if change == "response":
            call_id = record["accounting"]["calls"][0]["call_id"]
            evidence = store.load_replay(call_id)["evidence"]
            evidence["body"] += b"\n"
            envelope = SQLiteWorkflowStore._envelope(evidence)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE workflow_evidence SET body=?,response_sha=?,metadata_sha=? WHERE evidence_id=?",
                           (envelope[1], envelope[2], envelope[-1], evidence["evidence_id"]))
        elif change == "checkpoint":
            checkpoint = deepcopy(record["checkpoint"]["checkpoint"])
            checkpoint["output"] = "altered saved output"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE workflow_checkpoints SET checkpoint_json=?,checkpoint_sha=? WHERE run_id=? AND revision=?",
                           (canonical(checkpoint), runtime._sha(checkpoint), record["run_id"], record["checkpoint"]["revision"]))
        elif change == "decoded-result":
            call_id = record["accounting"]["calls"][0]["call_id"]
            decoded = store.load_replay(call_id)["decoded_result"]
            decoded["text"] = "altered accepted result"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE workflow_calls SET result_json=?,result_sha=? WHERE call_id=?",
                           (canonical(decoded), runtime._sha(decoded), call_id))
        else:
            record["output"] = "invented output"
            record["output_sha256"] = runtime.hashlib.sha256(record["output"].encode()).hexdigest()
            record.pop("record_sha256")
            record["record_sha256"] = runtime._sha(record)
        # These are self-consistent edits, not a test of an invalid SQLite row
        # checksum. The retained outcome must reject changed historical content.
        assert store.audit(record["run_id"])["ok"]
        with pytest.raises(runtime.CampaignRuntimeError, match="evidence|output"):
            runtime._validate_outcome(record, value, row, store, case)


def test_additional_lease_observation_does_not_invalidate_accepted_evidence(recorded_trial):
    path, value, row, record, case = recorded_trial
    with SQLiteWorkflowStore(path) as store:
        lease = store.acquire_lease(record["run_id"], "read-completed-state")
        store.release_lease(lease)
        assert store.inspect_run(record["run_id"])["events"] != record["accounting"]["events"]
        runtime._validate_outcome(record, value, row, store, case)


@pytest.mark.parametrize("change", ["unaccepted-envelope", "evidence-event-deletion"])
def test_failed_quote_history_is_bound_even_without_an_accepted_quote(tmp_path, native, change):
    native.invalid_quote = True
    directory = tmp_path / "campaign"
    value = freeze(directory)
    directory.mkdir()
    row = prepare_campaign()["schedules"]["pilot"][0]
    case = next(case for case in runtime.load_task_pack().for_stage("pilot") if case.task_id == row["task_id"])
    run_id = runtime._run_id(value, row)
    path = directory / "workflow.sqlite3"
    with SQLiteWorkflowStore(path) as store:
        started = time.perf_counter_ns()
        swarm = runtime._swarm(store, row, ALLOWANCES["per_trial_nanousd"])
        store.create_run(task_to_dict(runtime._task(case)), swarm._workflow_runtime().recipe,
                         ALLOWANCES["per_trial_nanousd"], run_id=run_id)
        with pytest.raises(WorkflowQuoteError) as caught:
            swarm.resume(run_id)
        record = runtime._outcome(store, value, row, run_id, None, caught.value, started, False)
        call = record["accounting"]["calls"][0]
        assert call["quote_id"] is call["evidence_id"] is None
        assert len(native.quotes) == 1 and not native.requests
        runtime._validate_outcome(record, value, row, store, case)
        event = next(event for event in record["accounting"]["events"] if event["type"] == "evidence_saved")
        if change == "unaccepted-envelope":
            envelope = store.load_evidence(call["call_id"], event["data"]["evidence_id"])
            envelope["request_id"] = "changed-unaccepted-quote-header"
            values = SQLiteWorkflowStore._envelope(envelope)
            # Keep the body and event byte hash identical; only binding every
            # retained envelope detects this self-consistent metadata change.
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE workflow_evidence SET request_id=?,metadata_sha=? WHERE evidence_id=?",
                           (envelope["request_id"], values[-1], envelope["evidence_id"]))
        else:
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("DELETE FROM workflow_events WHERE run_id=? AND sequence=?", (run_id, event["sequence"]))
        assert store.audit(run_id)["ok"]
        with pytest.raises(runtime.CampaignRuntimeError, match="evidence"):
            runtime._validate_outcome(record, value, row, store, case)
