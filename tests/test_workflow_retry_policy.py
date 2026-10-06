"""A durable run under a ``RetryPolicy`` bounds each node's attempts by its journal.

The recipe records the policy, so a run resumes only under the policy it started
with. Each node starts from the attempts its journal shows its current
generation used: every settled rejection counts, on any invocation, so a crash
and resume continue the cap instead of refilling it. Recipes without a policy
keep their bytes and their earlier retries. The wire is ``NativeTransport``:
requests, quotes, pricing and journaling are real.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from fakes import RATE_LIMITED, Crash, Gate, HttpError, ManualClock, NativeTransport, crash_when
from smythe import RetryPolicy, SQLiteWorkflowStore, Swarm, workflow
from smythe.async_executor import AsyncExecutor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology
from smythe.provider import NativeOutputDefectError, ProviderFeatures, ProviderResponseError
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.scheduling import _retry_after_s
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_provider import ProviderRequestRejectedError
from smythe.workflow_store import WorkflowConflictError

MODEL = "gpt-6-astra"
# Each NativeTransport generation bills 100 input and 10 output tokens.
CHARGE = 1_500_000
# The recipe's record of RetryPolicy(): every field but persist_attempts.
RECORD = {"retry_classes": ["transient"], "honor_retry_after": True, "backoff_max_s": 60.0,
          "deadline_s": None}
ERROR_TYPES = {401: "authentication_error", 403: "permission_error", 404: "not_found_error",
               413: "request_too_large"}


@pytest.fixture
def wire(monkeypatch):
    return NativeTransport().install(monkeypatch)


def durable(store, retry_policy=None, **options):
    return Swarm(model=MODEL, provider=OpenAIResponsesProvider(api_key="no-network", max_output_tokens=100),
                 run_store=store, max_budget_usd=1, retry_policy=retry_policy, **options)


def draft(*, failure_policy=FailurePolicy.RETRY, max_retries=1):
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", failure_policy=failure_policy, max_retries=max_retries),
    ])


def draft_calls(store, run_id, generation=0):
    """Each of the draft's calls at *generation*: invocation, attempt, cost and result."""
    return sorted((call["key"]["invocation"], call["key"]["attempt"], call["cost_nanousd"],
                   call["result_state"]) for call in store.inspect_run(run_id)["calls"]
                  if call["key"]["scope_id"] == "node/draft" and call["key"]["generation"] == generation)


def outcomes(wire):
    """The outcome of each generation request the provider received, in order."""
    return [record.outcome for record in wire.log if record.operation == "response"]


def test_a_rate_limited_node_retries_under_the_policy_with_exact_totals(tmp_path, wire):
    wire.schedule("execution", 0, RATE_LIMITED)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store, RetryPolicy()).execute(draft())
        run_id = result.execution_id
        assert result.output == "Draft accepted output." and result.cost_is_complete
        assert store.load_run(run_id)["config"]["retry_policy"] == RECORD
        assert draft_calls(store, run_id) == [(0, 0, 0, "rejected"), (0, 1, CHARGE, "applied")]
        assert result.workflow_accounting["confirmed_nanousd"] == CHARGE
        assert store.audit(run_id)["ok"]
    assert outcomes(wire) == ["http_429", "delivered"]


def test_a_rate_limited_node_is_capped_and_resume_makes_no_call_beyond_the_cap(tmp_path, wire):
    wire.schedule("execution", None, RATE_LIMITED)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = durable(store, RetryPolicy())
        with pytest.raises(ProviderRequestRejectedError) as caught:
            swarm.execute(draft(max_retries=2))
        assert caught.value.status_code == 429
        (run,) = store.list_runs()
        capped = [(0, attempt, 0, "rejected") for attempt in range(3)]
        assert draft_calls(store, run["run_id"]) == capped
        # Every attempt is used, so the node fails again without a call.
        with pytest.raises(RuntimeError, match="already used all 3 attempts"):
            swarm.resume(run["run_id"])
        assert draft_calls(store, run["run_id"]) == capped
        assert store.load_run(run["run_id"])["blocked_reason"] is None
    assert outcomes(wire) == ["http_429"] * 3



def test_resume_never_sends_an_attempt_whose_call_was_rejected_again(tmp_path, wire):
    """An attempt that ends before its request leaves no rejected call to count."""
    gate = Gate()
    # Attempt 0 times out while its input tokens are counted, before any request.
    wire.schedule("execution", 0, gate, operation="input_tokens")
    # Attempt 1's request is rate limited.
    wire.schedule("execution", 0, RATE_LIMITED)
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY, max_retries=1,
             timeout_s=1),
    ])
    try:
        with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
            swarm = durable(store, RetryPolicy())
            with pytest.raises(ProviderRequestRejectedError):
                swarm.execute(graph)
            gate.release()
            (run,) = store.list_runs()
            # Both attempts were used, so the node fails again without a call.
            with pytest.raises(RuntimeError, match="already used all 2 attempts"):
                swarm.resume(run["run_id"])
    finally:
        gate.release()
    assert outcomes(wire) == ["http_429"]

@pytest.mark.parametrize("second", ["delivered", "http_429"])
def test_a_crash_after_the_first_rejected_attempt_resumes_within_the_cap(
    tmp_path, monkeypatch, wire, second,
):
    wire.schedule("execution", 0, RATE_LIMITED)
    if second == "http_429":
        wire.schedule("execution", 1, RATE_LIMITED)
    image = tmp_path / "killed.db"
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        # The process is lost after attempt 0's rejection, before attempt 1 is prepared.
        point = crash_when(monkeypatch, store, "prepare_call",
                           lambda lease, key, **_: key.scope_id == "node/draft" and key.attempt == 1,
                           image=image, connection=store._db)
        with pytest.raises(Crash):
            durable(store, RetryPolicy()).execute(draft(max_retries=1))
        (run,) = store.list_runs()
    assert point.fired and outcomes(wire) == ["http_429"]
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        swarm = durable(recovered, RetryPolicy())
        if second == "delivered":
            assert swarm.resume(run["run_id"]).output == "Draft accepted output."
            # Attempt 1 continues the first invocation; nothing refilled attempt 0.
            assert draft_calls(recovered, run["run_id"]) == [(0, 0, 0, "rejected"),
                                                             (0, 1, CHARGE, "applied")]
        else:
            with pytest.raises(ProviderRequestRejectedError):
                swarm.resume(run["run_id"])
            assert draft_calls(recovered, run["run_id"]) == [(0, 0, 0, "rejected"),
                                                             (0, 1, 0, "rejected")]
            with pytest.raises(RuntimeError, match="already used all 2 attempts"):
                swarm.resume(run["run_id"])
        assert recovered.audit(run["run_id"])["ok"]
    # The run, its crash and its resumes made no more calls than its two attempts.
    assert outcomes(wire) == ["http_429", second]


@pytest.mark.parametrize("status", sorted(ERROR_TYPES))
def test_authority_and_input_rejections_are_not_retried_under_the_policy(tmp_path, wire, status):
    wire.schedule("execution", 0, HttpError(status, ERROR_TYPES[status]))
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(ProviderRequestRejectedError) as caught:
            durable(store, RetryPolicy()).execute(draft(max_retries=2))
        assert caught.value.status_code == status
        (run,) = store.list_runs()
        assert draft_calls(store, run["run_id"]) == [(0, 0, 0, "rejected")]
        assert outcomes(wire) == [f"http_{status}"]
        # Without a policy, the node failure policy retries the same rejection.
        wire.schedule("execution", 1, HttpError(status, ERROR_TYPES[status]))
        result = durable(store).execute(draft(max_retries=2))
        assert draft_calls(store, result.execution_id) == [(0, 0, 0, "rejected"),
                                                           (0, 1, CHARGE, "applied")]
    assert outcomes(wire) == [f"http_{status}"] * 2 + ["delivered"]


def test_a_recipe_without_the_policy_keeps_its_bytes(tmp_path, monkeypatch, wire):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        legacy = durable(store)._workflow_runtime()
        bounded = durable(store, RetryPolicy(persist_attempts=True))._workflow_runtime()
        with monkeypatch.context() as earlier:
            # The behavior fields of the release before retry_policy.
            earlier.setattr(workflow, "RECIPE_BEHAVIOR_FIELDS", MappingProxyType(
                {"prompt_assembly": (1, 1), "context_assembly": (0, 0)}))
            before = durable(store)._workflow_runtime().recipe
        assert workflow._canonical(legacy.recipe) == workflow._canonical(before)
        assert {key: value for key, value in bounded.recipe.items() if key != "retry_policy"} == before
        assert bounded.recipe["retry_policy"] == RECORD

        # A run without the policy resumes with its earlier retries: a halted
        # node's rejected call is followed by one new call.
        wire.schedule("execution", 0, RATE_LIMITED)
        swarm = durable(store)
        with pytest.raises(ProviderRequestRejectedError):
            swarm.execute(draft(failure_policy=FailurePolicy.HALT))
        (run,) = store.list_runs()
        assert "retry_policy" not in store.load_run(run["run_id"])["config"]
        assert swarm.resume(run["run_id"]).output == "Draft accepted output."
        assert draft_calls(store, run["run_id"]) == [(0, 0, 0, "rejected"), (1, 0, CHARGE, "applied")]


def test_resume_refuses_a_different_explicit_policy_and_keeps_the_saved_one(tmp_path, wire):
    wire.schedule("execution", None, RATE_LIMITED)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(ProviderRequestRejectedError):
            durable(store, RetryPolicy()).execute(draft(failure_policy=FailurePolicy.HALT))
        (run,) = store.list_runs()
        run_id = run["run_id"]
        calls = store.inspect_run(run_id)["call_count"]
        for other in (RetryPolicy(retry_classes={"transient", "authority_denied"}),
                      RetryPolicy(deadline_s=30), RetryPolicy(honor_retry_after=False)):
            with pytest.raises(WorkflowConflictError, match="retry_policy differs from the saved recipe"):
                durable(store, other).resume(run_id)
        # persist_attempts is not part of the record: the journal keeps every attempt.
        for same in (RetryPolicy(persist_attempts=True), None):
            with pytest.raises(RuntimeError, match="already used all 1 attempts"):
                durable(store, same).resume(run_id)
        assert store.inspect_run(run_id)["call_count"] == calls
        # A run without a policy cannot gain one when it resumes.
        wire.clear("execution", None)
        legacy = durable(store).execute(draft()).execution_id
        with pytest.raises(WorkflowConflictError, match="retry_policy differs from the saved recipe"):
            durable(store, RetryPolicy()).resume(legacy)
        assert store.inspect_run(legacy)["call_count"] == 1
    assert outcomes(wire) == ["http_429", "delivered"]


def test_a_journaled_rejection_waits_its_backoff_not_retry_after(tmp_path, monkeypatch, wire):
    # The journal saves no response headers, so a Retry-After never reaches the policy.
    waits = []
    original = AsyncExecutor.retry_delay_s

    def recorded(self, attempt, *, error=None):
        waits.append((attempt, _retry_after_s(error) if error is not None else None,
                      original(self, attempt, error=error)))
        return waits[-1][-1]

    monkeypatch.setattr(AsyncExecutor, "retry_delay_s", recorded)
    wire.schedule("execution", 0, HttpError(429, "rate_limit_error", retry_after_s=30))
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store, RetryPolicy()).execute(draft())
        assert draft_calls(store, result.execution_id) == [(0, 0, 0, "rejected"),
                                                           (0, 1, CHARGE, "applied")]
    # retry_backoff_s is 0, so the backoff is no wait; the header asked for 30 seconds.
    assert waits == [(1, None, 0.0)]


REFUSAL = {"output": [{"type": "message", "status": "completed", "role": "assistant",
                       "content": [{"type": "refusal", "refusal": "I can't help."}]}]}


# A completed reply whose only text is blank: unusable, though neither refused nor truncated.
BLANK = {"output": [{"type": "message", "status": "completed", "role": "assistant",
                     "content": [{"type": "output_text", "text": " "}]}]}
DEFECT_CLASSES = {"transient", "quality_defect"}


def reply(wire, *bodies):
    """Merge each of *bodies* into the next generation reply, in order; later replies are unchanged."""
    respond, queued = wire._response, iter(bodies)

    def replaced(sequence, payload, text):
        return {**respond(sequence, payload, text), **next(queued, {})}

    wire._response = replaced


@pytest.mark.parametrize("listed", [False, True], ids=["default-classes", "quality-defect-listed"])
def test_a_refusal_with_a_known_charge_is_retried_only_when_its_class_is_listed(
    tmp_path, wire, listed,
):
    reply(wire, REFUSAL)
    classes = DEFECT_CLASSES if listed else {"transient"}
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = durable(store, RetryPolicy(retry_classes=classes))
        if listed:
            result = swarm.execute(draft())
            run_id = result.execution_id
            assert result.output == "Draft accepted output." and result.cost_is_complete
            draft_node = next(node for node in result.graph.nodes if node.id == "draft")
            assert not {"response_error", "accounting_invalid"} & draft_node.metadata.keys()
        else:
            with pytest.raises(ProviderResponseError) as caught:
                swarm.execute(draft())
            assert caught.value.stop_reason == "refusal"
            (run,) = store.list_runs()
            run_id = run["run_id"]
        # The refused reply's known charge is kept as a rejected call either way.
        expected = [(0, 0, CHARGE, "rejected")] + ([(0, 1, CHARGE, "applied")] if listed else [])
        assert draft_calls(store, run_id) == expected
        assert store.inspect_run(run_id)["confirmed_nanousd"] == CHARGE * len(expected)
        assert store.audit(run_id)["ok"]
    assert outcomes(wire) == ["delivered"] * len(expected)


@pytest.mark.parametrize("failure_policy", [FailurePolicy.RETRY, FailurePolicy.HALT],
                         ids=["retry", "halt"])
@pytest.mark.parametrize(("body", "classes"), [(REFUSAL, {"transient"}), (BLANK, DEFECT_CLASSES)],
                         ids=["refusal-default-classes", "unusable-quality-defect-listed"])
def test_a_paid_rejection_the_policy_does_not_retry_is_replayed_on_resume(
    tmp_path, wire, failure_policy, body, classes,
):
    reply(wire, body)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = durable(store, RetryPolicy(retry_classes=classes))
        with pytest.raises(ProviderResponseError) as stopped:
            swarm.execute(draft(failure_policy=failure_policy))
        (run,) = store.list_runs()
        run_id = run["run_id"]
        rejected = [(0, 0, CHARGE, "rejected")]
        assert draft_calls(store, run_id) == rejected
        # Resume enters the rejected attempt again, and the journal replays its reply.
        with pytest.raises(ProviderResponseError) as replayed:
            swarm.resume(run_id)
        assert str(replayed.value) == str(stopped.value)
        assert replayed.value.receipt["workflow_replayed"] is True
        assert draft_calls(store, run_id) == rejected
        assert store.inspect_run(run_id)["confirmed_nanousd"] == CHARGE
        assert store.audit(run_id)["ok"]
    assert outcomes(wire) == ["delivered"]


def test_refusals_the_policy_retries_are_capped_and_resume_makes_no_call(tmp_path, wire):
    reply(wire, REFUSAL, REFUSAL)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = durable(store, RetryPolicy(retry_classes=DEFECT_CLASSES))
        with pytest.raises(NativeOutputDefectError):
            swarm.execute(draft(max_retries=1))
        (run,) = store.list_runs()
        run_id = run["run_id"]
        refused = [(0, 0, CHARGE, "rejected"), (0, 1, CHARGE, "rejected")]
        assert draft_calls(store, run_id) == refused
        # The last attempt's refusal replays, and no attempt is left after it.
        with pytest.raises(NativeOutputDefectError):
            swarm.resume(run_id)
        assert draft_calls(store, run_id) == refused
        assert store.inspect_run(run_id)["confirmed_nanousd"] == 2 * CHARGE
        assert store.audit(run_id)["ok"]
    assert outcomes(wire) == ["delivered"] * 2


def test_a_crash_after_a_retried_refusal_resumes_within_the_cap(tmp_path, monkeypatch, wire):
    reply(wire, REFUSAL)
    image = tmp_path / "killed.db"
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        # The process is lost after attempt 0's refusal, before attempt 1 is prepared.
        point = crash_when(monkeypatch, store, "prepare_call",
                           lambda lease, key, **_: key.scope_id == "node/draft" and key.attempt == 1,
                           image=image, connection=store._db)
        with pytest.raises(Crash):
            durable(store, RetryPolicy(retry_classes=DEFECT_CLASSES)).execute(draft(max_retries=1))
        (run,) = store.list_runs()
    assert point.fired and outcomes(wire) == ["delivered"]
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as recovered:
        swarm = durable(recovered, RetryPolicy(retry_classes=DEFECT_CLASSES))
        # The refusal replays without a call, and attempt 1 is the only new one.
        assert swarm.resume(run["run_id"]).output == "Draft accepted output."
        assert draft_calls(recovered, run["run_id"]) == [(0, 0, CHARGE, "rejected"),
                                                         (0, 1, CHARGE, "applied")]
        assert recovered.inspect_run(run["run_id"])["confirmed_nanousd"] == 2 * CHARGE
        assert recovered.audit(run["run_id"])["ok"]
    assert outcomes(wire) == ["delivered"] * 2


def test_a_judge_s_refused_reply_is_replayed_on_resume(tmp_path, wire):
    graph = draft()
    graph.nodes.append(Node(id="judge", label="Check", depends_on=["draft"], verifies="draft"))
    reply(wire, {}, REFUSAL)  # The draft is accepted, and the judge's reply is refused.
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = durable(store, RetryPolicy())
        with pytest.raises(ProviderResponseError) as stopped:
            swarm.execute(graph)
        (run,) = store.list_runs()
        run_id = run["run_id"]

        def judged():
            return [(call["key"]["phase"], call["key"]["attempt"], call["result_state"])
                    for call in store.node_generation_calls(run_id, "judge", 0)]

        assert judged() == [("verification", 0, "rejected")]
        with pytest.raises(ProviderResponseError) as replayed:
            swarm.resume(run_id)
        assert str(replayed.value) == str(stopped.value)
        assert judged() == [("verification", 0, "rejected")]
        assert store.inspect_run(run_id)["confirmed_nanousd"] == 2 * CHARGE
    assert outcomes(wire) == ["delivered"] * 2


def test_a_regenerated_node_gets_a_fresh_allowance(tmp_path, wire):
    graph = draft(max_retries=1)
    graph.nodes.append(Node(id="judge", label="Check", depends_on=["draft"], verifies="draft",
                            max_regenerations=1))
    # Generation 0 of the draft is rejected once and judged failing; generation 1 too.
    wire.script("execution", "First draft.", "FAIL: revise the draft", "Second draft.", "PASS")
    wire.schedule("execution", 0, RATE_LIMITED)
    wire.schedule("execution", 3, RATE_LIMITED)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store, RetryPolicy()).execute(graph)
        assert result.output == "Second draft."
        for generation in (0, 1):
            assert draft_calls(store, result.execution_id, generation) == [
                (0, 0, 0, "rejected"), (0, 1, CHARGE, "applied")]


def test_a_provider_whose_sdk_may_retry_is_refused_before_any_call(tmp_path, monkeypatch, wire):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        # A run that started under the policy, then a provider that declares SDK retries.
        wire.schedule("execution", 0, RATE_LIMITED)
        with pytest.raises(ProviderRequestRejectedError):
            durable(store, RetryPolicy()).execute(draft(failure_policy=FailurePolicy.HALT))
        (run,) = store.list_runs()
        calls = store.inspect_run(run["run_id"])["call_count"]
        monkeypatch.setattr(OpenAIResponsesProvider, "feature_support",
                            lambda self, model: ProviderFeatures(sdk_retries=2))
        with pytest.raises(WorkflowBindingError, match="^retry_policy must be the only retry owner"):
            durable(store, RetryPolicy()).execute(draft())
        assert len(store.list_runs()) == 1
        # A resume that configures no policy keeps the saved one, and is refused too.
        with pytest.raises(WorkflowBindingError, match="^retry_policy must be the only retry owner"):
            durable(store).resume(run["run_id"])
        assert store.inspect_run(run["run_id"])["call_count"] == calls
    assert outcomes(wire) == ["http_429"]



def test_a_policy_that_keeps_unanswered_exposure_is_refused_before_any_call(tmp_path, wire):
    """The journal keeps every sent call's exposure itself, so the option is not taken."""
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        policy = RetryPolicy(retain_uncertain_dispatch=True)
        with pytest.raises(WorkflowBindingError, match="^retry_policy with retain_uncertain_dispatch"):
            durable(store, policy).execute(draft())
        with pytest.raises(WorkflowBindingError, match="retry_policy"):
            durable(store)._workflow_runtime(
                behavior_settings={"retry_policy": {**RECORD, "retain_uncertain_dispatch": True}})
        assert store.list_runs() == []
    assert outcomes(wire) == []

def test_a_policy_this_release_cannot_read_is_refused_before_any_call(tmp_path, wire):
    newer = {**RECORD, "jitter": "full"}  # A field a newer release might record.
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        for value in (newer, {**RECORD, "persist_attempts": True}, {**RECORD, "backoff_max_s": -1}):
            with pytest.raises(WorkflowBindingError, match="retry_policy"):
                durable(store)._workflow_runtime(behavior_settings={"retry_policy": value})
        assert store.list_runs() == []
        recipe = {**durable(store)._workflow_runtime().recipe, "retry_policy": newer}
        run_id = store.create_run(None, recipe, workflow._nanousd(1))["run_id"]
        for swarm in (durable(store), durable(store, RetryPolicy())):
            with pytest.raises(WorkflowConflictError, match="not supported by this release"):
                swarm.resume(run_id)
        assert store.inspect_run(run_id)["call_count"] == 0
    assert wire.log == []
