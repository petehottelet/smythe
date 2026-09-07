"""Offline crash, accounting and fencing checks for the workflow journal."""

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import sqlite3
import threading

import pytest

from smythe.pricing import PRICE_VERSION
from smythe.workflow_store import (
    CallKey, OFFLINE_PRICE_VERSION, SQLiteWorkflowStore, WorkflowBudgetError,
    WorkflowConflictError, WorkflowCorruptionError, WorkflowLeaseError,
    WorkflowStateError, WorkflowValidationError,
)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds * 1_000_000_000


@pytest.fixture
def journal(tmp_path):
    clock = Clock()
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite", clock_ns=clock)
    store.create_run({"goal": "private goal"}, {"secret": "private config"}, 10**12, run_id="run")
    lease = store.acquire_lease("run", "owner")
    yield store, lease, clock
    store.close()


def prepare(store, lease, scope="worker", *, offline=False, output_cap=100):
    provider = {"kind": "offline" if offline else "openai_responses",
                "adapter_version": "test-v1", "decoder_version": "test-v1"}
    request = {"model": "gpt-6-astra", "service_tier": "default", "max_output_tokens": output_cap,
               "instructions": "test", "reasoning": {"effort": "medium"}, "store": False,
               "truncation": "disabled", "input": [{"role": "user", "content": "PRIVATE REQUEST encrypted_content secret"}]}
    return store.prepare_call(lease, CallKey("execution", scope), request_json=canonical(request),
                              provider=provider, price_version=OFFLINE_PRICE_VERSION if offline else PRICE_VERSION)


def envelope(call, raw, operation="response", **metadata):
    return {"body": raw if isinstance(raw, bytes) else canonical(raw).encode(),
            "request_sha256": call["request_sha256"], "operation": operation,
            "status_code": 200, "request_id": "req_test", **metadata}


def quote(store, lease, call, *, count=100):
    raw = {"input_tokens": count}
    if call["provider"]["kind"] == "offline":
        raw = {"provider_kind": "offline", "version": 1, "input_tokens": 0}
    evidence_id = store.append_quote_evidence(lease, call["call_id"], envelope(call, raw, "input_tokens"))
    return store.accept_quote(lease, call["call_id"], evidence_id)


def dispatch(store, lease, call):
    q = quote(store, lease, call)
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    return store.claim_dispatch(lease, call["call_id"], call["request_sha256"])


def response(*, inputs=100, outputs=2, model="gpt-6-astra", text="PRIVATE OUTPUT"):
    return {"id": "resp_test", "model": model, "service_tier": "default", "status": "completed",
            "usage": {"input_tokens": inputs, "output_tokens": outputs,
                      "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 1}},
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}


def settled(store, lease, scope="worker", **kwargs):
    call = prepare(store, lease, scope)
    permit = dispatch(store, lease, call)
    evidence_id = store.append_response(permit, envelope(call, response(**kwargs)))
    return store.settle_call(lease, call["call_id"], evidence_id)


def accepted(store, lease, scope="worker"):
    call = settled(store, lease, scope)
    return store.accept_result(lease, call["call_id"], {"text": "PRIVATE DECODED"}, "test-v1")


def test_exact_charge_receipt_replay_and_inspection_are_separate(journal):
    store, lease, _ = journal
    call = accepted(store, lease)
    assert call["cost_nanousd"] == 1_060_000
    assert call["receipt"]["request_id"] == "req_test"
    assert call["receipt"]["response_id"] == "resp_test"
    assert store.settle_call(lease, call["call_id"], call["evidence_id"])["cost_nanousd"] == 1_060_000
    assert store.audit("run")["confirmed_nanousd"] == 1_060_000
    replay = store.load_replay(call["call_id"])
    assert b"PRIVATE OUTPUT" in replay["evidence"]["body"]
    assert replay["decoded_result"] == {"text": "PRIVATE DECODED"}
    safe = canonical(store.inspect_run("run"))
    for private in ("PRIVATE", "private goal", "private config", "encrypted_content", "body", "request_json"):
        assert private not in safe


def test_offline_is_explicit_zero_external_api_cost(journal):
    store, lease, _ = journal
    call = prepare(store, lease, offline=True)
    permit = dispatch(store, lease, call)
    evidence_id = store.append_response(permit, envelope(call, {"provider_kind": "offline", "version": 1, "text": "echo"}))
    record = store.settle_call(lease, call["call_id"], evidence_id)
    assert record["cost_nanousd"] == 0
    assert record["receipt"]["price_version"] == OFFLINE_PRICE_VERSION
    assert record["receipt"]["pricing_scope"] == "zero_external_api_cost"
    assert "actual_model" not in record["receipt"]
    assert store.audit("run")["confirmed_nanousd"] == 0


def test_store_identity_run_null_task_and_reopen(journal):
    store, lease, _ = journal
    identity = store.store_id
    first = store.load_run("run")
    assert first["task_sha256"] == hashlib.sha256(canonical({"goal": "private goal"}).encode()).hexdigest()
    assert first["config_sha256"] == hashlib.sha256(canonical({"secret": "private config"}).encode()).hexdigest()
    assert store.create_run(None, {}, None, run_id="taskless")["task"] is None
    with SQLiteWorkflowStore(store.path, read_only=True) as reader:
        assert reader.store_id == identity
        assert reader.inspect_run("run")["run_id"] == "run"
        with pytest.raises(WorkflowStateError):
            reader.release_lease(lease)


@pytest.mark.parametrize("field,value", [("attempt", True), ("turn", -1), ("generation", "1"), ("phase", []), ("scope_id", "")])
def test_call_keys_validate_before_database(field, value):
    kwargs = {"phase": "planning", "scope_id": "architect", field: value}
    with pytest.raises(WorkflowValidationError):
        CallKey(**kwargs)


@pytest.mark.parametrize("change", ["request", "tools", "adapter", "price"])
def test_same_logical_call_cannot_change_binding(journal, change):
    store, lease, _ = journal
    call = prepare(store, lease)
    kwargs = {"request_json": call["request_json"], "tool_names_json": "{}", "provider": dict(call["provider"]), "price_version": PRICE_VERSION}
    if change == "request":
        raw = json.loads(kwargs["request_json"])
        raw["input"] = "different"
        kwargs["request_json"] = canonical(raw)
    elif change == "tools":
        kwargs["tool_names_json"] = '{"x":"y"}'
    elif change == "adapter":
        kwargs["provider"]["adapter_version"] = "test-v2"
    else:
        kwargs["price_version"] = "different"
    with pytest.raises((WorkflowConflictError, WorkflowValidationError)):
        store.prepare_call(lease, CallKey("execution", "worker"), **kwargs)
    assert store.inspect_run("run")["call_count"] == 1


def test_detached_operations_and_inputs_cannot_be_rebound(journal):
    store, lease, _ = journal
    inputs = {"graph": {"siblings": ["before"]}}
    first = store.begin_operation(lease, "review-node-g0", "supervision", inputs)
    inputs["graph"]["siblings"].append("after")
    assert store.load_operation("run", "review-node-g0")["inputs"]["graph"]["siblings"] == ["before"]
    with pytest.raises(WorkflowConflictError):
        store.begin_operation(lease, "review-node-g0", "supervision", inputs)
    frozen = store.load_operation("run", "review-node-g0")["inputs"]
    assert store.begin_operation(lease, "review-node-g0", "supervision", frozen)["operation_id"] == first["operation_id"]
    result = store.complete_operation(lease, "review-node-g0", {"proposal": None})
    assert result["state"] == "completed"
    with pytest.raises(WorkflowConflictError):
        store.complete_operation(lease, "review-node-g0", {"proposal": "different"})


def test_checkpoint_cas_consumes_graph_handoff_and_calls_atomically(journal):
    store, lease, _ = journal
    call = accepted(store, lease)
    operation = store.begin_operation(lease, "plan-0", "planning", {"task": {"goal": "g"}})
    store.complete_operation(lease, "plan-0", {"graph": {"nodes": []}}, [call["call_id"]])
    checkpoint = store.save_checkpoint(lease, 0, {"graph": {"nodes": []}}, consumed_operation_ids=[operation["operation_id"]])
    assert checkpoint["revision"] == 1
    assert checkpoint["consumed_call_ids"] == [call["call_id"]]
    assert store.load_replay(call["call_id"])["result_state"] == "applied"
    assert store.load_operation("run", "plan-0")["state"] == "applied"
    assert store.get_checkpoint("run") == checkpoint
    with pytest.raises(WorkflowConflictError):
        store.save_checkpoint(lease, 0, {"graph": "stale"})
    assert store.get_checkpoint("run") == checkpoint
    assert store.audit("run")["revision"] == 1


def test_checkpoint_invalid_consumption_rolls_back_every_transition(journal):
    store, lease, _ = journal
    valid, pending = accepted(store, lease), prepare(store, lease, "pending")
    with pytest.raises(WorkflowStateError):
        store.save_checkpoint(lease, 0, {}, consumed_call_ids=[valid["call_id"], pending["call_id"]])
    assert store.get_checkpoint("run") is None
    assert store.load_replay(valid["call_id"])["result_state"] == "accepted"
    assert store.load_run("run")["revision"] == 0


@pytest.mark.parametrize("boundary,expected", [("prepared", "undispatched"), ("quoted", "undispatched"),
    ("reserved", "undispatched"), ("dispatched", "unknown"), ("response_saved", "settle"),
    ("settled", "decode"), ("accepted", "apply"), ("applied", "applied")])
def test_crash_boundary_recovery_never_repeats_claim(journal, boundary, expected):
    store, lease, clock = journal
    call = prepare(store, lease)
    order = ["prepared", "quoted", "reserved", "dispatched", "response_saved", "settled", "accepted", "applied"]
    stage = order.index(boundary)
    if stage >= 1:
        q = quote(store, lease, call)
    if stage >= 2:
        store.reserve_call(lease, call["call_id"], q["quote_id"])
    if stage >= 3:
        permit = store.claim_dispatch(lease, call["call_id"], call["request_sha256"])
    if stage >= 4:
        evidence_id = store.append_response(permit, envelope(call, response()))
    if stage >= 5:
        store.settle_call(lease, call["call_id"], evidence_id)
    if stage >= 6:
        store.accept_result(lease, call["call_id"], {"text": "ok"}, "test-v1")
    if stage >= 7:
        store.save_checkpoint(lease, 0, {}, consumed_call_ids=[call["call_id"]])
    clock.advance(31)
    recovered_lease = store.acquire_lease("run", "new-owner")
    with SQLiteWorkflowStore(store.path, clock_ns=clock) as reopened:
        plan = reopened.recover(recovered_lease)
        assert plan[expected] == [call["call_id"]]
        if stage >= 3:
            with pytest.raises((WorkflowStateError, WorkflowBudgetError)):
                reopened.claim_dispatch(recovered_lease, call["call_id"], call["request_sha256"])
        assert reopened.audit("run")["call_count"] == 1


def test_expired_owner_can_append_exact_late_evidence_but_cannot_settle(journal):
    store, lease, clock = journal
    call = prepare(store, lease)
    permit = dispatch(store, lease, call)
    clock.advance(31)
    new_owner = store.acquire_lease("run", "new-owner")
    store.recover(new_owner)
    assert store.inspect_run("run")["unknown_nanousd"] == 6_250_000
    raw = envelope(call, response())
    evidence_id = store.append_response(permit, raw)
    assert store.append_response(permit, raw) == evidence_id
    with pytest.raises(WorkflowLeaseError):
        store.settle_call(lease, call["call_id"], evidence_id)
    result = store.settle_call(new_owner, call["call_id"], evidence_id)
    assert result["billing_state"] == "known"
    assert result["blocked_reason"] is None
    assert store.inspect_run("run")["unknown_nanousd"] == 0
    with pytest.raises(WorkflowConflictError):
        store.append_response(replace(permit, token="wrong"), raw)


def test_conflicting_response_is_retained_and_latches_without_replacing_original(journal):
    store, lease, _ = journal
    call = prepare(store, lease)
    permit = dispatch(store, lease, call)
    original = store.append_response(permit, envelope(call, response()))
    with pytest.raises(WorkflowConflictError):
        store.append_response(permit, envelope(call, response(text="different")))
    assert store.load_replay(call["call_id"])["evidence_id"] == original
    assert store.inspect_run("run")["blocked_reason"] == "evidence_conflict"
    with closing(sqlite3.connect(store.path)) as db, db:
        assert db.execute("SELECT COUNT(*) FROM workflow_evidence WHERE operation='response'").fetchone()[0] == 2
    store.settle_call(lease, call["call_id"], original)
    assert store.inspect_run("run")["blocked_reason"] == "evidence_conflict"


@pytest.mark.parametrize("raw", [b"malformed", b'{"usage":{},"usage":{}}', {}, response(model="unpriced"), response(inputs=True)])
def test_malformed_billing_is_durable_unknown_and_blocks_admission(journal, raw):
    store, lease, _ = journal
    call = prepare(store, lease)
    permit = dispatch(store, lease, call)
    evidence_id = store.append_response(permit, envelope(call, raw))
    first = store.settle_call(lease, call["call_id"], evidence_id)
    second = store.settle_call(lease, call["call_id"], evidence_id)
    assert first == second
    assert first["cost_nanousd"] is None
    assert first["unknown_nanousd"] == 6_250_000
    assert store.load_evidence(call["call_id"], evidence_id)["body"] == envelope(call, raw)["body"]
    with pytest.raises(WorkflowStateError):
        prepare(store, lease, "blocked")
    with pytest.raises(WorkflowStateError):
        store.release_before_dispatch(lease, call["call_id"], "unsafe")


def test_known_overrun_commits_full_charge_before_closing_admission(journal):
    store, lease, _ = journal
    call = settled(store, lease, outputs=30_000_000)
    assert call["cost_nanousd"] == 1_500_000_960_000
    assert call["admission_closed"] and call["blocked_reason"] == "budget_overrun"
    assert store.audit("run")["confirmed_nanousd"] == call["cost_nanousd"]
    with pytest.raises(WorkflowBudgetError):
        prepare(store, lease, "later")


def test_atomic_parallel_reservations_do_not_oversubscribe(tmp_path):
    path = tmp_path / "race.sqlite"
    with SQLiteWorkflowStore(path) as store:
        store.create_run(None, {}, 6_250_000, run_id="run")
        lease = store.acquire_lease("run", "owner")
        calls = [prepare(store, lease, str(index)) for index in range(2)]
        quotes = [quote(store, lease, call) for call in calls]
        barrier = threading.Barrier(2)
        def reserve(index):
            with SQLiteWorkflowStore(path) as contender:
                barrier.wait(timeout=5)
                try:
                    contender.reserve_call(lease, calls[index]["call_id"], quotes[index]["quote_id"])
                    return True
                except WorkflowBudgetError:
                    return False
        with ThreadPoolExecutor(2) as pool:
            assert sorted(pool.map(reserve, range(2))) == [False, True]
        assert store.inspect_run("run")["reserved_nanousd"] == 6_250_000


def test_dispatch_claim_is_exclusive_even_with_same_request_and_lease(journal):
    store, lease, _ = journal
    call = prepare(store, lease)
    q = quote(store, lease, call)
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    barrier = threading.Barrier(2)
    def claim(_):
        with SQLiteWorkflowStore(store.path, clock_ns=store._clock_ns) as contender:
            barrier.wait(timeout=5)
            try:
                contender.claim_dispatch(lease, call["call_id"], call["request_sha256"])
                return True
            except WorkflowStateError:
                return False
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(claim, range(2))) == [False, True]


def test_reservation_money_above_sqlite_integer_range_remains_exact(journal):
    store, lease, _ = journal
    store.create_run(None, {}, 10**100, run_id="huge")
    huge_lease = store.acquire_lease("huge", "owner")
    call = prepare(store, huge_lease)
    q = quote(store, huge_lease, call, count=10**80)
    store.reserve_call(huge_lease, call["call_id"], q["quote_id"])
    assert store.audit("huge")["reserved_nanousd"] == 10**80 * 25_000 + 100 * 75_000
    with closing(sqlite3.connect(store.path)) as db, db:
        assert db.execute("SELECT typeof(reserved) FROM workflow_runs WHERE run_id='huge'").fetchone()[0] == "text"


@pytest.mark.parametrize("mutation", ["aggregate", "money", "request", "raw", "quote", "checkpoint", "operation"])
def test_corrupt_persisted_data_fails_audit(journal, mutation):
    store, lease, _ = journal
    call = accepted(store, lease)
    store.begin_operation(lease, "o", "planning", {})
    store.complete_operation(lease, "o", {})
    store.save_checkpoint(lease, 0, {})
    with closing(sqlite3.connect(store.path)) as db, db:
        statements = {
            "aggregate": "UPDATE workflow_runs SET confirmed='0'",
            "money": "UPDATE workflow_runs SET confirmed='01'",
            "request": "UPDATE workflow_calls SET request_json='{}'",
            "raw": "UPDATE workflow_evidence SET body=x'00' WHERE operation='response'",
            "quote": "UPDATE workflow_calls SET ceiling='1'",
            "checkpoint": "UPDATE workflow_checkpoints SET checkpoint_json='{}x'",
            "operation": "UPDATE workflow_operations SET inputs_json='[]'",
        }
        db.execute(statements[mutation])
    with pytest.raises(WorkflowCorruptionError):
        store.audit("run")
    with pytest.raises(WorkflowCorruptionError):
        store.recover(lease)
    assert call["cost_nanousd"] > 0


def test_foreign_database_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE runs(id TEXT)")
    before = path.read_bytes()
    with pytest.raises(WorkflowCorruptionError):
        SQLiteWorkflowStore(path)
    assert path.read_bytes() == before


def test_invocation_trigger_is_durable_and_idempotent(journal):
    store, lease, clock = journal
    assert store.allocate_invocation(lease, "supervision", "judge", 0, "node-a-g0") == 0
    assert store.allocate_invocation(lease, "supervision", "judge", 0, "node-b-g0") == 1
    clock.advance(31)
    current = store.acquire_lease("run", "new")
    assert store.allocate_invocation(current, "supervision", "judge", 0, "node-a-g0") == 0
    assert store.allocate_invocation(current, "supervision", "judge", 1, "node-a-g1") == 0
    with pytest.raises(WorkflowLeaseError):
        store.allocate_invocation(lease, "supervision", "judge", 0, "stale")


def test_release_before_dispatch_preserves_quote_but_never_releases_dispatched(journal):
    store, lease, _ = journal
    call = prepare(store, lease)
    q = quote(store, lease, call)
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    store.release_before_dispatch(lease, call["call_id"], "cancelled before send")
    assert store.audit("run")["reserved_nanousd"] == 0
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    store.claim_dispatch(lease, call["call_id"], call["request_sha256"])
    with pytest.raises(WorkflowStateError):
        store.release_before_dispatch(lease, call["call_id"], "cancelled after send")


def test_paid_rejected_native_result_remains_rejected_after_checkpoint(journal):
    store, lease, _ = journal
    call = settled(store, lease)
    store.reject_result(lease, call["call_id"], "not usable")
    store.save_checkpoint(lease, 0, {}, consumed_call_ids=[call["call_id"]])
    replay = store.load_replay(call["call_id"])
    assert replay["result_state"] == "rejected"
    assert replay["decoded_result"] is None
    assert store.settle_call(lease, call["call_id"], call["evidence_id"])["result_state"] == "rejected"
    assert store.audit("run")["confirmed_nanousd"] == 1_060_000


@pytest.mark.parametrize("value", [True, -1, 1.0, "1", float("nan"), float("inf")])
def test_budget_rejects_malformed_money_without_run_creation(journal, value):
    store, _, _ = journal
    with pytest.raises(WorkflowValidationError):
        store.create_run(None, {}, value, run_id="invalid")


@pytest.mark.parametrize("value", [True, -1, 1.0, "1", None])
def test_invalid_count_is_saved_before_rejection_and_cannot_dispatch(journal, value):
    store, lease, _ = journal
    call = prepare(store, lease)
    evidence_id = store.append_quote_evidence(lease, call["call_id"], envelope(call, {"input_tokens": value}, "input_tokens"))
    with pytest.raises(WorkflowValidationError):
        store.accept_quote(lease, call["call_id"], evidence_id)
    assert store.load_evidence(call["call_id"], evidence_id)["body"]
    with pytest.raises(WorkflowStateError):
        store.claim_dispatch(lease, call["call_id"], call["request_sha256"])
    assert store.audit("run")["reserved_nanousd"] == 0


def test_released_unquoted_call_can_be_quoted_after_recovery(journal):
    store, lease, _ = journal
    call = prepare(store, lease)
    store.release_before_dispatch(lease, call["call_id"], "not dispatched")
    q = quote(store, lease, call)
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    assert store.claim_dispatch(lease, call["call_id"], call["request_sha256"])


def test_quote_breach_latches_even_with_room_in_total_budget(journal):
    store, lease, _ = journal
    call = settled(store, lease, outputs=1000)
    assert call["cost_nanousd"] > call["ceiling_nanousd"]
    assert call["cost_nanousd"] < store.load_run("run")["budget_nanousd"]
    assert call["blocked_reason"] == "budget_overrun"


def test_completed_checkpoint_consumes_then_closes_new_admission(journal):
    store, lease, _ = journal
    call = accepted(store, lease)
    with pytest.raises(WorkflowStateError):
        store.save_checkpoint(lease, 0, {"status": "completed"})
    assert store.get_checkpoint("run") is None
    assert store.load_replay(call["call_id"])["result_state"] == "accepted"
    store.save_checkpoint(lease, 0, {"status": "completed"}, consumed_call_ids=[call["call_id"]])
    assert store.load_run("run")["status"] == "completed"
    assert prepare(store, lease)["call_id"] == call["call_id"]
    with pytest.raises(WorkflowStateError):
        prepare(store, lease, "new")


def test_completed_checkpoint_rejects_even_zero_dollar_active_offline_call(journal):
    store, lease, _ = journal
    call = prepare(store, lease, offline=True)
    dispatch(store, lease, call)
    with pytest.raises(WorkflowStateError):
        store.save_checkpoint(lease, 0, {"status": "completed"})
    assert store.load_run("run")["status"] == "running"


@pytest.mark.parametrize("column", ["receipt_json", "result_json"])
def test_saved_result_and_receipt_hashes_detect_replay_mutation(journal, column):
    store, lease, _ = journal
    call = accepted(store, lease)
    with closing(sqlite3.connect(store.path)) as db, db:
        db.execute(f"UPDATE workflow_calls SET {column}='{{}}'")
    with pytest.raises(WorkflowCorruptionError):
        store.load_replay(call["call_id"])


def test_unknown_exposure_does_not_disappear_when_sibling_settles(journal):
    store, lease, _ = journal
    unresolved, sibling = prepare(store, lease), prepare(store, lease, "sibling")
    dispatch(store, lease, unresolved)
    sibling_permit = dispatch(store, lease, sibling)
    store.mark_unknown(lease, unresolved["call_id"], "lost transport")
    evidence_id = store.append_response(sibling_permit, envelope(sibling, response()))
    store.settle_call(lease, sibling["call_id"], evidence_id)
    audit = store.inspect_run("run")
    assert audit["confirmed_nanousd"] == 1_060_000
    assert audit["unknown_nanousd"] == 6_250_000
    assert audit["reserved_nanousd"] == 0
    assert audit["blocked_reason"] == "unknown_exposure"


def test_checkpoint_cannot_consume_another_runs_call(journal):
    store, lease, _ = journal
    call = accepted(store, lease)
    store.create_run(None, {}, None, run_id="other")
    other = store.acquire_lease("other", "owner")
    with pytest.raises(WorkflowConflictError):
        store.save_checkpoint(other, 0, {}, consumed_call_ids=[call["call_id"]])
    assert store.get_checkpoint("other") is None


@pytest.mark.parametrize("credential", [{"config": {"api_key": "secret"}},
    {"clients": [{"headers": {"Authorization": "Bearer secret"}}]}, {"options": {"refresh-token": "secret"}}])
def test_nested_provider_credentials_are_rejected_before_prepare(journal, credential):
    store, lease, _ = journal
    call = prepare(store, lease)
    with pytest.raises(WorkflowValidationError):
        store.prepare_call(lease, CallKey("execution", "unsafe"), request_json=call["request_json"],
                           provider={**call["provider"], **credential}, price_version=PRICE_VERSION)
    with pytest.raises(WorkflowValidationError):
        store.create_run(None, credential, None, run_id="unsafe")
    assert store.inspect_run("run")["call_count"] == 1


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_offline_quote_version_is_type_sensitive(journal, version):
    store, lease, _ = journal
    call = prepare(store, lease, offline=True)
    evidence_id = store.append_quote_evidence(lease, call["call_id"], envelope(call,
        {"provider_kind": "offline", "version": version, "input_tokens": 0}, "input_tokens"))
    with pytest.raises(WorkflowValidationError):
        store.accept_quote(lease, call["call_id"], evidence_id)


def test_missing_journal_table_is_never_recreated_silently(journal):
    store, _, _ = journal
    with closing(sqlite3.connect(store.path)) as db, db:
        db.execute("DROP TABLE workflow_invocations")
    with pytest.raises(WorkflowCorruptionError):
        SQLiteWorkflowStore(store.path)
    with closing(sqlite3.connect(store.path)) as db, db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='workflow_invocations'").fetchone() is None


@pytest.mark.parametrize("changes", [{"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "private"}]}]},
    {"audio": {"voice": "alloy"}}, {"store": True}, {"truncation": "auto"}])
def test_native_request_scope_is_validated_before_admission(journal, changes):
    store, lease, _ = journal
    call = prepare(store, lease)
    request = {**json.loads(call["request_json"]), **changes}
    with pytest.raises(WorkflowValidationError):
        store.prepare_call(lease, CallKey("execution", "invalid"), request_json=canonical(request),
                           provider=call["provider"], price_version=PRICE_VERSION)
    assert store.inspect_run("run")["call_count"] == 1


def test_read_only_run_discovery_contains_resume_ids_and_exact_balances_only(journal, monkeypatch):
    from smythe.provider_responses import OpenAIResponsesProvider
    store, lease, _ = journal
    call = prepare(store, lease)
    q = quote(store, lease, call)
    store.reserve_call(lease, call["call_id"], q["quote_id"])
    store.create_run(None, {}, None, run_id="another")
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda *args: pytest.fail("SDK constructed"))
    with closing(sqlite3.connect(store.path)) as db, db:
        before = list(db.iterdump())
    with SQLiteWorkflowStore(store.path, read_only=True) as reader:
        runs = {row["run_id"]: row for row in reader.list_runs()}
    assert set(runs) == {"run", "another"}
    assert runs["run"]["reserved_nanousd"] == 6_250_000
    assert runs["run"]["confirmed_nanousd"] == 0
    assert runs["another"]["budget_nanousd"] is None
    assert "PRIVATE" not in canonical(runs) and "private goal" not in canonical(runs)
    assert "task" not in runs["run"] and "config" not in runs["run"]
    with closing(sqlite3.connect(store.path)) as db, db:
        assert list(db.iterdump()) == before


def test_offline_failed_status_cannot_imply_known_success(journal):
    store, lease, _ = journal
    call = prepare(store, lease, offline=True)
    permit = dispatch(store, lease, call)
    evidence_id = store.append_response(permit, envelope(call,
        {"provider_kind": "offline", "version": 1, "text": "not successful"}, status_code=500))
    record = store.settle_call(lease, call["call_id"], evidence_id)
    assert record["cost_nanousd"] is None
    assert record["billing_state"] == "unknown"
    assert store.inspect_run("run")["unknown_calls"] == 1
