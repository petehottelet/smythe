"""Complete native workflow accounting through public Swarm entry points."""

import asyncio
from collections import Counter
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smythe import Swarm, Task
from smythe.planner import LLMArchitect
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import ProviderAccountingError
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.router import WhiteRabbit
from smythe.supervisor import LLMSupervisor, SUPERVISOR_SYSTEM_PROMPT
from smythe.synthesizer import MERGE_SYSTEM_PROMPT, Synthesizer, SynthesisStrategy
from smythe.workflow_provider import ProviderRequestRejectedError
from smythe.workflow_store import SQLiteWorkflowStore
from smythe.workflow_store import WorkflowBudgetError, WorkflowConflictError, WorkflowError

MODEL = "gpt-6-astra"
PLAN = {"topology": ["serial"], "nodes": [
    {"id": "draft", "label": "Draft"},
    {"id": "judge", "label": "Check", "depends_on": ["draft"],
     "verifies": "draft", "max_regenerations": 1},
]}


def wire(body, identity):
    return SimpleNamespace(content=json.dumps(body).encode(), status_code=200,
                           headers={"x-request-id": identity})


class RateLimitError(Exception):
    """Mimics the SDK's 429 status error, which carries the raw HTTP response."""

    def __init__(self):
        super().__init__("Error code: 429")
        self.response = SimpleNamespace(content=json.dumps({"error": {
            "message": "Rate limit reached", "type": "requests", "param": None,
            "code": "rate_limit_exceeded"}}).encode(), status_code=429,
            headers={"x-request-id": "req_rate_limited"})


@pytest.fixture
def native_transport(monkeypatch):
    state = SimpleNamespace(requests=[], counts=[], clients=[], planning_outputs=[], overrides={},
                            invalid_worker_counts=False, rate_limited=Counter())

    async def count(**payload):
        state.counts.append(payload)
        invalid = state.invalid_worker_counts and payload["instructions"] not in {
            PLANNING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT, MERGE_SYSTEM_PROMPT,
        }
        return wire({"input_tokens": False if invalid else 100}, f"req_count_{len(state.counts)}")

    async def generate(**payload):
        state.requests.append(payload)
        system = payload["instructions"]
        if system == PLANNING_SYSTEM_PROMPT:
            phase = "planning"
            text = state.planning_outputs.pop(0) if state.planning_outputs else json.dumps(PLAN)
        elif system.startswith("You are a task classifier."):
            phase, text = "routing", "autonomous"
        elif system == SUPERVISOR_SYSTEM_PROMPT:
            phase, text = "supervision", '{"change":false,"reason":"Plan remains complete"}'
        elif system == MERGE_SYSTEM_PROMPT:
            phase, text = "synthesis", "Final accepted output."
        else:
            phase = "execution"
            prompt = payload["input"][-1]["content"]
            text = "PASS" if "Your step: Check" in prompt else "Draft accepted output."
        if state.rate_limited[phase]:
            state.rate_limited[phase] -= 1
            raise RateLimitError()
        raw = {
            "id": f"resp_{len(state.requests)}", "model": payload["model"], "status": "completed",
            "service_tier": "default", "usage": {
                "input_tokens": 100, "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            }, "output": [{"type": "message", "status": "completed", "role": "assistant",
                           "content": [{"type": "output_text", "text": text}]}],
        }
        if phase in state.overrides:
            raw.update(state.overrides[phase])
        return wire(raw, f"req_generation_{len(state.requests)}")

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


def swarm(store, *, rich=False, max_budget_usd=1, **options):
    provider = OpenAIResponsesProvider(api_key="dummy-no-network", max_output_tokens=100)
    if rich:
        options.update(
            router=WhiteRabbit(
                classifier_provider=provider, classifier_model=MODEL,
                autonomous=LLMArchitect(provider, planning_model=MODEL),
            ), supervisor=LLMSupervisor(provider, model=MODEL, only_terminal=False), max_revisions=1,
        )
    return Swarm(model=MODEL, provider=provider, max_budget_usd=max_budget_usd, run_store=store,
                 synthesizer=Synthesizer(SynthesisStrategy.LLM_MERGE), **options)


def test_every_builtin_paid_phase_is_counted_once_and_consumed(tmp_path, native_transport):
    with SQLiteWorkflowStore(tmp_path / "all-phases.db") as store:
        result = swarm(store, rich=True).execute(Task(
            "Write and check", context={"source": "Input document"}, done_when=["All checks pass"],
        ))
        accounting = store.inspect_run(result.execution_id)
        phases = Counter(call["key"]["phase"] for call in accounting["calls"])
        assert phases == {"routing": 1, "planning": 1, "execution": 1, "verification": 1,
                          "supervision": 2, "synthesis": 1}
        assert accounting["call_count"] == len(native_transport.requests) == 7
        assert len(native_transport.counts) == 7
        assert result.total_cost_usd == pytest.approx(.0105)
        assert accounting["confirmed_nanousd"] == 10_500_000
        assert accounting["reserved_nanousd"] == accounting["unknown_nanousd"] == 0
        assert result.cost_scope == "complete_text_workflow" and result.cost_is_complete
        assert not result.cost_contains_estimates
        assert all(call["result_state"] == "applied" for call in accounting["calls"])
        assert result.output == "Final accepted output."
        assert len(native_transport.clients) == 1 and native_transport.clients[0].close.await_count == 1


def test_separate_plan_execute_and_reopened_resume_do_not_rebuy(tmp_path, native_transport, monkeypatch):
    path = tmp_path / "handoff.db"
    with SQLiteWorkflowStore(path) as store:
        first = swarm(store)
        graph = first.plan(Task("Write and check", constraints=["Keep it short"]))
        execution_id = graph.run_ref["run_id"]
        assert len(native_transport.requests) == 1
        assert store.inspect_run(execution_id)["confirmed_nanousd"] == 1_500_000
        result = first.execute(graph)
        assert result.execution_id == execution_id and result.total_cost_usd == .006
        assert len(native_transport.requests) == 4
        assert len(native_transport.clients) == 2
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on completed resume"))
    with SQLiteWorkflowStore(path) as reopened:
        restored = swarm(reopened).resume(execution_id)
        assert restored.output == result.output and restored.total_cost_usd == .006
        assert restored.workflow_accounting["call_count"] == 4
    assert len(native_transport.requests) == len(native_transport.counts) == 4


def test_invalid_plan_repair_is_an_additional_paid_explicit_attempt(tmp_path, native_transport):
    native_transport.planning_outputs = ["invalid JSON"]
    with SQLiteWorkflowStore(tmp_path / "repair.db") as store:
        result = swarm(store).execute(Task("Write and check"))
        calls = store.inspect_run(result.execution_id)["calls"]
        planning = [call for call in calls if call["key"]["phase"] == "planning"]
        assert sorted(call["key"]["attempt"] for call in planning) == [0, 1]
        assert all(call["billing_state"] == "known" for call in planning)
        assert result.total_cost_usd == .0075 and len(native_transport.requests) == 5


@pytest.mark.parametrize("boundary", ["settle_call", "accept_result"])
def test_native_worker_saved_response_recovers_after_persistence_interruption(
    tmp_path, native_transport, monkeypatch, boundary,
):
    with SQLiteWorkflowStore(tmp_path / f"crash-{boundary}.db") as store:
        instance = swarm(store)
        graph = instance.plan(Task("Write and check"))
        original = getattr(store, boundary)
        crashed = []

        def interrupt(lease, call_id, *args, **kwargs):
            record = store.load_replay(call_id)
            if record["key"]["phase"] == "execution" and not crashed:
                crashed.append(call_id)
                raise OSError("Simulated local persistence interruption after raw response")
            return original(lease, call_id, *args, **kwargs)

        monkeypatch.setattr(store, boundary, interrupt)
        with pytest.raises(ProviderAccountingError):
            instance.execute(graph)
        assert crashed and len(native_transport.requests) == 2
        assert store.load_replay(crashed[0])["evidence"] is not None
        monkeypatch.setattr(store, boundary, original)
        resumed = instance.resume(graph.run_ref["run_id"])
        assert resumed.output == "Final accepted output."
        assert resumed.total_cost_usd == .006
        assert len(native_transport.requests) == 4  # Planning and saved draft were replayed.
        assert store.inspect_run(resumed.execution_id)["confirmed_nanousd"] == 6_000_000
        recovered_node = next(node for node in resumed.graph.nodes if node.id == "draft")
        assert not {"accounting_invalid", "accounting_error", "response_error"} & recovered_node.metadata.keys()


def test_concurrent_tasks_on_same_swarm_have_separate_ledgers(tmp_path, native_transport):
    with SQLiteWorkflowStore(tmp_path / "concurrent.db") as store:
        instance = swarm(store)

        async def run():
            return await asyncio.gather(
                instance.execute_async(Task("First", context={"source": "One"})),
                instance.execute_async(Task("Second", context={"source": "Two"})),
            )

        first, second = asyncio.run(run())
        assert first.execution_id != second.execution_id
        assert first.total_cost_usd == second.total_cost_usd == .006
        assert first.graph.task.context != second.graph.task.context
        assert len(native_transport.requests) == 8
        assert len(native_transport.clients) == 2


def test_unknown_native_worker_stays_blocked_without_redispatch(tmp_path, native_transport):
    native_transport.overrides["execution"] = {"usage": None}
    with SQLiteWorkflowStore(tmp_path / "unknown.db") as store:
        instance = swarm(store)
        graph = instance.plan(Task("Write and check"))
        with pytest.raises(ProviderAccountingError):
            instance.execute(graph)
        before = len(native_transport.requests)
        with pytest.raises((ProviderAccountingError, WorkflowError)):
            instance.resume(graph.run_ref["run_id"])
        assert len(native_transport.requests) == before == 2
        accounting = store.inspect_run(graph.run_ref["run_id"])
        assert accounting["confirmed_nanousd"] == 1_500_000
        assert accounting["unknown_calls"] == 1 and accounting["unknown_nanousd"] > 0


def test_planning_charge_reduces_budget_before_any_worker_dispatch(tmp_path, native_transport):
    # Each call quote is6,250,000nanoUSD, while actual planning costs1,500,000.
    # The first quote fits; another quote cannot fit the remaining5,500,000.
    with SQLiteWorkflowStore(tmp_path / "budget.db") as store:
        instance = swarm(store, max_budget_usd=.007)
        graph = instance.plan(Task("Write and check"))
        with pytest.raises(WorkflowBudgetError):
            instance.execute(graph)
        assert len(native_transport.requests) == 1
        accounting = store.inspect_run(graph.run_ref["run_id"])
        assert accounting["confirmed_nanousd"] == 1_500_000
        assert accounting["reserved_nanousd"] == accounting["unknown_nanousd"] == 0


def test_completed_cached_run_rejects_changed_configuration_before_sdk(tmp_path, native_transport, monkeypatch):
    with SQLiteWorkflowStore(tmp_path / "config.db") as store:
        result = swarm(store).execute(Task("Write and check"))
        monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on mismatch"))
        with pytest.raises(WorkflowConflictError):
            swarm(store, max_budget_usd=2).resume(result.execution_id)
        assert len(native_transport.requests) == 4


def test_managed_quote_failure_does_not_advance_retry_attempt(tmp_path, native_transport):
    plan = json.loads(json.dumps(PLAN))
    plan["nodes"][0].update(failure_policy="retry", max_retries=2)
    native_transport.planning_outputs = [json.dumps(plan)]
    native_transport.invalid_worker_counts = True
    with SQLiteWorkflowStore(tmp_path / "quote-retry.db") as store:
        instance = swarm(store)
        graph = instance.plan(Task("Write and check"))
        with pytest.raises(WorkflowError) as caught:
            instance.execute(graph)
        assert caught.value.envelope.json() == {"input_tokens": False}
        assert len(native_transport.requests) == 1
        workers = [call for call in store.inspect_run(graph.run_ref["run_id"])["calls"]
                   if call["key"]["phase"] == "execution"]
        assert len(workers) == 1 and workers[0]["key"]["attempt"] == 0
        assert workers[0]["state"] == "prepared"
        native_transport.invalid_worker_counts = False
        result = instance.resume(graph.run_ref["run_id"])
        assert result.total_cost_usd == .006 and len(native_transport.requests) == 4


def draft_calls(store, run_id):
    return sorted((call["key"]["invocation"], call["key"]["attempt"], call["cost_nanousd"], call["result_state"])
                  for call in store.inspect_run(run_id)["calls"] if call["key"]["scope_id"] == "node/draft")


def assert_exact_ledger(store, run_id):
    accounting = store.inspect_run(run_id)
    assert accounting["confirmed_nanousd"] == sum(call["cost_nanousd"] or 0 for call in accounting["calls"])
    assert accounting["reserved_nanousd"] == accounting["unknown_nanousd"] == accounting["unknown_calls"] == 0
    assert store.audit(run_id)["ok"]


def test_rate_limited_worker_retries_under_retry_policy_with_exact_totals(tmp_path, native_transport):
    plan = json.loads(json.dumps(PLAN))
    plan["nodes"][0].update(failure_policy="retry", max_retries=1)
    native_transport.planning_outputs = [json.dumps(plan)]
    native_transport.rate_limited["execution"] = 1
    with SQLiteWorkflowStore(tmp_path / "retry-429.db") as store:
        result = swarm(store).execute(Task("Write and check"))
        assert result.output == "Final accepted output."
        assert result.total_cost_usd == .006 and result.cost_is_complete
        assert result.workflow_accounting["call_count"] == len(native_transport.requests) == 5
        assert draft_calls(store, result.execution_id) == [(0, 0, 0, "rejected"), (0, 1, 1_500_000, "applied")]
        assert_exact_ledger(store, result.execution_id)
        draft = next(node for node in result.graph.nodes if node.id == "draft")
        assert not {"accounting_invalid", "response_error"} & draft.metadata.keys()


def test_rate_limited_halt_fails_node_and_resume_makes_a_new_attempt(tmp_path, native_transport, monkeypatch):
    native_transport.rate_limited["execution"] = 1
    path = tmp_path / "halt-429.db"
    with SQLiteWorkflowStore(path) as store:
        instance = swarm(store)
        graph = instance.plan(Task("Write and check"))
        run_id = graph.run_ref["run_id"]
        with pytest.raises(ProviderRequestRejectedError):
            instance.execute(graph)
        run = store.load_run(run_id)
        assert run["status"] == "running" and run["blocked_reason"] is None
        saved = store.get_checkpoint(run_id)["checkpoint"]
        draft = next(node for node in saved["graph"]["nodes"] if node["id"] == "draft")
        assert saved["status"] == "failed" and draft["status"] == "failed" and "HTTP 429" in draft["result"]
        assert not {"accounting_invalid", "response_error"} & draft["metadata"].keys()
        assert_exact_ledger(store, run_id)
        result = instance.resume(run_id)
        assert result.output == "Final accepted output." and result.total_cost_usd == .006
        assert len(native_transport.requests) == 5
        # The rejected attempt stays in the journal; resume used a new invocation.
        assert draft_calls(store, run_id) == [(0, 0, 0, "rejected"), (1, 0, 1_500_000, "applied")]
        assert_exact_ledger(store, run_id)
    monkeypatch.setattr(OpenAIResponsesProvider, "_get_client", lambda _: pytest.fail("SDK on completed resume"))
    with SQLiteWorkflowStore(path) as reopened:
        assert swarm(reopened).resume(run_id).total_cost_usd == .006


def test_rate_limit_held_as_unknown_by_080_journal_is_reclassified_on_resume(
    tmp_path, native_transport, monkeypatch,
):
    import smythe.workflow_store as workflow_store

    native_transport.rate_limited["execution"] = 1
    original = workflow_store._http_rejection_receipt
    # Reproduce a 0.8.0 journal, where the 429 became unknown exposure.
    monkeypatch.setattr(workflow_store, "_http_rejection_receipt", lambda *args: None)
    path = tmp_path / "old-429.db"
    with SQLiteWorkflowStore(path) as store:
        instance = swarm(store)
        graph = instance.plan(Task("Write and check"))
        run_id = graph.run_ref["run_id"]
        with pytest.raises(ProviderAccountingError):
            instance.execute(graph)
        assert store.load_run(run_id)["blocked_reason"] == "unknown_exposure"
    monkeypatch.setattr(workflow_store, "_http_rejection_receipt", original)
    with SQLiteWorkflowStore(path) as store:
        result = swarm(store).resume(run_id)
        assert result.output == "Final accepted output." and result.total_cost_usd == .006
        assert result.cost_is_complete and len(native_transport.requests) == 5
        assert draft_calls(store, run_id) == [(0, 0, 0, "rejected"), (1, 0, 1_500_000, "applied")]
        events = [event["type"] for event in store.inspect_run(run_id)["events"]]
        assert events.count("unknown_exposure_resolved") == 1
        draft = next(node for node in result.graph.nodes if node.id == "draft")
        assert not {"accounting_invalid", "accounting_error", "response_error",
                    "cost_usd_unknown"} & draft.metadata.keys()
        assert_exact_ledger(store, run_id)
        assert swarm(store).resume(run_id).output == "Final accepted output."
        assert len(native_transport.requests) == 5
