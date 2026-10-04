"""Reserved durable calls that were never dispatched, across kills, regeneration and completion."""

import asyncio
import sqlite3

import pytest

from fakes import Crash, Gate, ManualClock, NativeTransport, crash_when
from smythe import SQLiteWorkflowStore, Swarm
from smythe.async_executor import AsyncExecutor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.workflow_store import CallKey, WorkflowStateError

# Each NativeTransport generation bills 1,500,000 nanoUSD; its quote reserves this ceiling.
CEILING = 6_250_000
CHARGE = 1_500_000


def durable(store, *, concurrency):
    return Swarm(model="gpt-6-astra",
                 provider=OpenAIResponsesProvider(api_key="no-network", max_output_tokens=100),
                 run_store=store, parallel=True, max_concurrency=concurrency, max_budget_usd=1)


def transport(monkeypatch, *, verdict, held=None):
    """Install a wire fake whose judge ("Check") replies ``verdict``.

    With ``held``, the consumer's first input-token count waits for that gate.
    """
    class Wire(NativeTransport):
        async def _call(self, operation, payload):
            if (held is not None and not held.released and operation == "input_tokens"
                    and self._prompt(payload).startswith("Consumer")):
                await held.wait()
            return await super()._call(operation, payload)

    wire = Wire().install(monkeypatch)
    wire.replies["execution"] = lambda prompt: verdict if prompt.startswith("Check") else "Accepted output."
    return wire


def sent(wire, label):
    return sum(wire._prompt(payload).startswith(label) for payload in wire.requests)


def call(store, run_id, node, generation, *, attempt=0, phase="execution"):
    record = store.lookup_call(run_id, CallKey(phase, f"node/{node}", generation, 0, attempt))
    return next(item for item in store.inspect_run(run_id)["calls"] if item["call_id"] == record["call_id"])


def events(store, run_id, call_id):
    return [(event["type"], event["data"]) for event in store.inspect_run(run_id)["events"]
            if event["call_id"] == call_id]


def released(store, run_id):
    return [(event["call_id"], event["data"]) for event in store.inspect_run(run_id)["events"]
            if event["type"] == "call_released"]


@pytest.mark.parametrize("boundary", ["claim_dispatch", "append_response"])
def test_kill_before_a_regeneration_leaves_only_never_dispatched_calls_releasable(
    tmp_path, monkeypatch, boundary,
):
    # The judge's failing verdict is saved, unapplied, while the consumer, which
    # depends on the judged draft, is between its reservation and its response.
    # Resuming applies the verdict, so the consumer's generation 0 is superseded.
    held, image = Gate(), tmp_path / "killed.db"
    first = transport(monkeypatch, verdict="FAIL: revise the draft", held=held)
    source = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="consumer", label="Consumer", depends_on=["draft"]),
        Node(id="judge", label="Check", depends_on=["draft"], verifies="draft", max_regenerations=1),
    ])
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        def consumer(run_id, call_id):
            record = store.lookup_call(run_id, CallKey("execution", "node/consumer"))
            return record is not None and record["call_id"] == call_id

        due = {"claim_dispatch": lambda lease, call_id, _: consumer(lease.run_id, call_id),
               "append_response": lambda permit, _: consumer(permit.run_id, permit.call_id)}[boundary]
        point = crash_when(monkeypatch, store, boundary, due, image=image, connection=store._db)
        verdicts = AsyncExecutor._regenerate_and_settle

        async def kill_before_verdict(self, node, graph, active):
            if node.id != "judge":
                return await verdicts(self, node, graph, active)
            # The judge's completed checkpoint is saved; let the consumer reach
            # the kill before this process applies the verdict.
            held.release()
            for _ in range(3000):
                if point.fired:
                    break
                await asyncio.sleep(0.01)
            raise Crash("process killed")

        monkeypatch.setattr(AsyncExecutor, "_regenerate_and_settle", kill_before_verdict)
        with pytest.raises(Crash):
            durable(store, concurrency=2).execute(source)
        run_id = store.list_runs()[0]["run_id"]
    monkeypatch.setattr(AsyncExecutor, "_regenerate_and_settle", verdicts)
    assert point.fired and (sent(first, "Draft"), sent(first, "Check")) == (1, 1)
    assert sent(first, "Consumer") == (boundary == "append_response")

    second = transport(monkeypatch, verdict="PASS")
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        orphan = call(recovered, run_id, "consumer", 0)
        assert orphan["state"] == ("reserved" if boundary == "claim_dispatch" else "dispatched")
        if boundary == "append_response":
            # The request may have reached the provider: it stays unknown
            # exposure, which closes admission, and is never released.
            with pytest.raises(WorkflowStateError, match="unknown_exposure"):
                durable(recovered, concurrency=2).resume(run_id)
            orphan = call(recovered, run_id, "consumer", 0)
            assert (orphan["state"], orphan["unknown_nanousd"]) == ("unknown", CEILING)
            assert released(recovered, run_id) == []
            assert second.requests == []
            return
        result = durable(recovered, concurrency=2).resume(run_id)
        accounting = recovered.inspect_run(run_id)
        orphan = call(recovered, run_id, "consumer", 0)
        assert (orphan["state"], orphan["billing_state"], orphan["reserved_nanousd"]) == ("released", "released", 0)
        assert orphan["evidence_id"] is None
        assert [kind for kind, _ in events(recovered, run_id, orphan["call_id"])] == [
            "call_prepared", "evidence_saved", "call_reserved", "call_released"]
        assert released(recovered, run_id) == [
            (orphan["call_id"], {"reason": "superseded_generation", "released_nanousd": str(CEILING)})]
        assert recovered.audit(run_id)["ok"]
    # The released request was never sent; generation 1 of each node was.
    assert (sent(second, "Draft"), sent(second, "Consumer"), sent(second, "Check")) == (1, 1, 1)
    assert accounting["status"] == "completed"
    assert (accounting["reserved_nanousd"], accounting["unknown_nanousd"]) == (0, 0)
    assert accounting["confirmed_nanousd"] == 5 * CHARGE
    assert result.cost_is_complete and result.workflow_accounting["reserved_nanousd"] == 0


def test_reserved_call_of_the_current_generation_is_dispatched_on_resume(tmp_path, monkeypatch):
    # The draft's regeneration is saved, then a kill follows the reservation
    # of its generation 1 call. That call is current, so resuming sends it
    # under its saved quote and reservation instead of releasing it.
    image = tmp_path / "killed.db"
    first = transport(monkeypatch, verdict="FAIL: revise the draft")
    source = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Check", depends_on=["draft"], verifies="draft", max_regenerations=1),
    ])
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        def regenerated(lease, call_id, _):
            record = store.lookup_call(lease.run_id, CallKey("execution", "node/draft", 1))
            return record is not None and record["call_id"] == call_id

        point = crash_when(monkeypatch, store, "claim_dispatch", regenerated, image=image,
                           connection=store._db)
        with pytest.raises(Crash):
            durable(store, concurrency=1).execute(source)
        run_id = store.list_runs()[0]["run_id"]
    assert point.fired and (sent(first, "Draft"), sent(first, "Check")) == (1, 1)

    second = transport(monkeypatch, verdict="PASS")
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        assert call(recovered, run_id, "draft", 1)["state"] == "reserved"
        result = durable(recovered, concurrency=1).resume(run_id)
        accounting = recovered.inspect_run(run_id)
        current = call(recovered, run_id, "draft", 1)
        history = [kind for kind, _ in events(recovered, run_id, current["call_id"])]
        assert released(recovered, run_id) == []
        assert recovered.audit(run_id)["ok"]
    assert (current["state"], current["result_state"]) == ("settled", "applied")
    assert history.count("call_reserved") == 1 and history.count("dispatch_claimed") == 1
    # Only the judge's new call was counted; the draft reused its saved quote.
    assert (len(second.counts), sent(second, "Draft"), sent(second, "Check")) == (1, 1, 1)
    assert accounting["status"] == "completed" and accounting["reserved_nanousd"] == 0
    assert accounting["confirmed_nanousd"] == 4 * CHARGE
    assert result.cost_is_complete


def test_completion_releases_a_reservation_that_a_retried_attempt_never_dispatched(
    tmp_path, monkeypatch,
):
    wire = transport(monkeypatch, verdict="PASS")
    source = ExecutionGraph([Topology.SERIAL], [
        Node(id="only", label="Only", failure_policy=FailurePolicy.RETRY, max_retries=1),
    ])
    with SQLiteWorkflowStore(tmp_path / "attempts.db") as store:
        claim, refused = store.claim_dispatch, []

        def busy_once(lease, call_id, request_sha256):
            if not refused:
                refused.append(call_id)
                raise sqlite3.OperationalError("database is locked")
            return claim(lease, call_id, request_sha256)

        monkeypatch.setattr(store, "claim_dispatch", busy_once)
        result = durable(store, concurrency=1).execute(source)
        run_id = result.execution_id
        accounting = store.inspect_run(run_id)
        abandoned = call(store, run_id, "only", 0, attempt=0)
        retried = call(store, run_id, "only", 0, attempt=1)
        assert store.audit(run_id)["ok"]
        assert released(store, run_id) == [
            (abandoned["call_id"], {"reason": "run_completed", "released_nanousd": str(CEILING)})]
    assert refused == [abandoned["call_id"]]
    assert (abandoned["state"], abandoned["evidence_id"]) == ("released", None)
    assert (retried["state"], retried["result_state"]) == ("settled", "applied")
    assert (len(wire.counts), len(wire.requests)) == (2, 1)
    assert accounting["status"] == "completed"
    assert (accounting["reserved_nanousd"], accounting["confirmed_nanousd"]) == (0, CHARGE)
    assert result.cost_is_complete
