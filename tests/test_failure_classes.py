"""Every exception a node attempt raises has one failure class.

``classify_failure`` names how an attempt failed and ``is_terminal`` whether
the run stops at once, whatever the node's failure policy. Both executors take
that decision from ``is_terminal`` alone, and it stops on exactly the
exceptions they stopped on before failure classes existed. Without a retry
policy (``tests/test_retry_policy.py``), a class describes a failure; it never
changes what is retried.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import copy
import json
import pickle

import pytest

from fakes import _STATUS_ERRORS, MessagesTransport
from smythe._failures import FAILURE_CLASSES, classify_failure, is_terminal, status_failure_class
from smythe.async_executor import AsyncExecutor
from smythe.budget import (
    BudgetEstimateRequired, BudgetReconciliationError, BudgetValidationError, SentinelAlert,
)
from smythe.executor import Executor
from smythe.executor_base import ArtifactIntegrityError, NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.pricing_anthropic import MODEL as MESSAGES_MODEL
from smythe.provider import (
    CompletionResult, OutputRefusedError, OutputTruncatedError, Provider,
    ProviderAccountingCancelledError, ProviderAccountingError, ProviderResponseError,
)
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import OpenAIResponsesProvider, RawResponseEnvelope
from smythe.registry import Registry
from smythe.tools import ChatMessage, ToolLoopLimitError
from smythe.tracer import Tracer
from smythe.verifier import VerificationRecoveryError
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_provider import (
    ProviderRequestRejectedError, WorkflowProviderContext, WorkflowQuoteError,
)
from smythe.workflow_store import CallKey, SQLiteWorkflowStore, WorkflowError, WorkflowStateError
from smythe.workflow_tools import ReadToolError

# The exceptions both executors stopped on, whatever the node's failure
# policy, before failure classes existed.
TERMINAL_TYPES = (
    VerificationRecoveryError, BudgetValidationError, NodeFinalizationError, SentinelAlert,
    ProviderResponseError, WorkflowError,
)


class StatusOnly(Exception):
    """An SDK-free error that carries nothing but an HTTP status."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def rejected(status):
    return ProviderRequestRejectedError(
        f"Provider rejected the request with HTTP {status}; the journal settled it at zero cost",
        status_code=status,
    )


def sdk_error(status):
    """The SDK status error a wire fake raises for ``status``."""
    return _STATUS_ERRORS[status](status, {"error": {"type": "error"}}, {})


# Every exception the executors see: a factory, its class and whether it is terminal.
CASES = {
    "verification-recovery": (
        lambda: VerificationRecoveryError("Could not persist verification control"),
        "persistence", True),
    "node-finalization": (
        lambda: NodeFinalizationError("step", OSError(28, "disk full")), "persistence", True),
    "budget-validation": (
        lambda: BudgetValidationError("cost_usd must be a finite non-negative number"),
        "terminal_accounting", True),
    "sentinel-alert": (lambda: SentinelAlert(1.5, 1.0, "step"), "terminal_accounting", True),
    "budget-reconciliation": (
        lambda: BudgetReconciliationError(spent=1.5, limit=1.0, node_id="step", actual_cost=0.7,
                                          reserved_cost=0.5),
        "terminal_accounting", True),
    "provider-accounting": (
        lambda: ProviderAccountingError("Native response has unresolved billing evidence"),
        "unknown_outcome", True),
    "provider-accounting-cancelled": (
        lambda: ProviderAccountingCancelledError("Native generation was cancelled after dispatch"),
        "unknown_outcome", True),
    "provider-response": (
        lambda: ProviderResponseError("Native response did not complete"),
        "provider_rejected", True),
    "workflow": (lambda: WorkflowError("Invalid saved managed receipt"), "workflow", True),
    "workflow-state": (
        lambda: WorkflowStateError("Saved read result has no text"), "workflow", True),
    "workflow-quote": (
        lambda: WorkflowQuoteError("Managed input counting did not establish a usable quote", None),
        "workflow", True),
    "rejected-401": (lambda: rejected(401), "authority_denied", False),
    "rejected-403": (lambda: rejected(403), "authority_denied", False),
    "rejected-404": (lambda: rejected(404), "input_error", False),
    "rejected-413": (lambda: rejected(413), "input_error", False),
    "rejected-429": (lambda: rejected(429), "transient", False),
    "truncated": (lambda: OutputTruncatedError("max_tokens"), "provider_rejected", False),
    "refused": (lambda: OutputRefusedError("refusal"), "provider_rejected", False),
    "read-tool": (
        lambda: ReadToolError("Read step 'page' failed: HTTP 404 read response"),
        "provider_rejected", False),
    "tool-loop-limit": (
        lambda: ToolLoopLimitError("Node 'step' hit max_tool_iterations=10 without completing"),
        "quality_defect", False),
    "workflow-binding": (
        lambda: WorkflowBindingError("Read profile input must be a mapping"), "input_error", False),
    "estimate-required": (
        lambda: BudgetEstimateRequired("step", "gpt-image-2", "OpenAIImageProvider"),
        "input_error", False),
    "artifact-integrity": (
        lambda: ArtifactIntegrityError("A dependency's image changed after it was written"),
        "input_error", False),
    "timeout": (lambda: TimeoutError("Node 'step' timed out after 5s"), "transient", False),
    "connection": (lambda: ConnectionError("Connection reset"), "transient", False),
    "runtime": (lambda: RuntimeError("boom"), "transient", False),
    "status-only-429": (lambda: StatusOnly(429), "transient", False),
    "status-only-403": (lambda: StatusOnly(403), "authority_denied", False),
}

# The class of each SDK status error in tests/fakes.py.
SDK_CLASSES = {
    400: "input_error", 401: "authority_denied", 403: "authority_denied", 404: "input_error",
    409: "transient", 413: "input_error", 422: "input_error", 429: "transient",
    500: "transient", 529: "transient",
}
CASES.update({
    f"sdk-{status}": (lambda status=status: sdk_error(status), failure_class, False)
    for status, failure_class in SDK_CLASSES.items()
})


def test_failure_classes_are_the_named_set():
    assert FAILURE_CLASSES == {
        "input_error", "authority_denied", "quality_defect", "provider_rejected", "transient",
        "unknown_outcome", "terminal_accounting", "persistence", "workflow", "validator_error",
        "security_stop", "deadline",
    }


def test_every_fake_sdk_status_error_has_a_case():
    assert set(SDK_CLASSES) == set(_STATUS_ERRORS)


@pytest.mark.parametrize("case", CASES)
def test_each_exception_has_one_class_and_the_terminal_flag_it_always_had(case):
    make, failure_class, terminal = CASES[case]
    error = make()

    assert classify_failure(error) == failure_class
    assert failure_class in FAILURE_CLASSES
    assert is_terminal(error) is terminal
    assert is_terminal(error) is isinstance(error, TERMINAL_TYPES)


def test_a_terminal_base_wins_over_any_other_base():
    class LateRead(ReadToolError, WorkflowError):
        pass

    class RejectedAccounting(ProviderRequestRejectedError, BudgetValidationError):
        pass

    assert (classify_failure(LateRead("late")), is_terminal(LateRead("late"))) == ("workflow", True)
    error = RejectedAccounting("HTTP 429", status_code=429)
    assert (classify_failure(error), is_terminal(error)) == ("terminal_accounting", True)


def test_cancellation_and_interpreter_exits_are_never_terminal_node_failures():
    for error in (asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(1)):
        assert is_terminal(error) is False


@pytest.mark.parametrize("status,failure_class", [
    (400, "input_error"), (401, "authority_denied"), (402, "input_error"),
    (403, "authority_denied"), (404, "input_error"), (408, "transient"), (409, "transient"),
    (413, "input_error"), (422, "input_error"), (429, "transient"), (499, "transient"),
    (500, "transient"), (503, "transient"), (529, "transient"),
    (200, None), (302, None), (600, None), (True, None), ("429", None), (429.0, None), (None, None),
])
def test_http_statuses_have_classes(status, failure_class):
    assert status_failure_class(status) == failure_class


def test_an_exception_without_a_usable_status_is_transient():
    class BrokenResponse(Exception):
        @property
        def response(self):
            raise RuntimeError("the response was not kept")

    assert classify_failure(BrokenResponse("lost")) == "transient"
    assert classify_failure(StatusOnly("429")) == "transient"
    assert classify_failure(StatusOnly(True)) == "transient"


@pytest.mark.parametrize("status,failure_class", [
    (401, "authority_denied"), (403, "authority_denied"), (404, "input_error"),
    (413, "input_error"), (429, "transient"),
])
def test_a_journaled_rejection_derives_its_read_only_class_from_its_status(status, failure_class):
    error = rejected(status)

    assert error.failure_class == failure_class == classify_failure(error)
    with pytest.raises(AttributeError):
        error.failure_class = "transient"
    restored = pickle.loads(pickle.dumps(error))
    assert restored.failure_class == failure_class
    assert "failure_class" not in vars(restored)


# ---------------------------------------------------------------------------
# Executor decisions
# ---------------------------------------------------------------------------


class Raising(Provider):
    """Raises a new instance of each failure in turn, repeating the last unless told to succeed."""

    def __init__(self, *makes, then_succeed=False):
        self.makes, self.then_succeed, self.calls = makes, then_succeed, 0

    async def complete(self, system, prompt, model):
        self.calls += 1
        if self.calls <= len(self.makes):
            raise self.makes[self.calls - 1]()
        if self.then_succeed:
            return CompletionResult("done")
        raise self.makes[-1]()


def run(mode, provider, graph, tracer):
    options = dict(provider=provider, registry=Registry(), tracer=tracer, artifact_dir=None)
    if mode == "serial":
        return Executor(**options).run(graph)
    return asyncio.run(AsyncExecutor(**options, max_concurrency=1).run(graph))


def one_node(policy, max_retries=2):
    node = Node("step", id="step", failure_policy=policy, max_retries=max_retries)
    return node, ExecutionGraph([Topology.SERIAL], [node])


@pytest.mark.parametrize("policy", [FailurePolicy.RETRY, FailurePolicy.SKIP])
@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("case", CASES)
def test_both_executors_retry_skip_and_stop_on_the_same_failures_as_before(case, mode, policy):
    make, _, terminal = CASES[case]
    provider, tracer = Raising(make), Tracer()
    node, graph = one_node(policy)

    if terminal or policy is FailurePolicy.RETRY:
        with pytest.raises(type(make())):
            run(mode, provider, graph, tracer)
    else:
        assert run(mode, provider, graph, tracer) is graph

    retried = policy is FailurePolicy.RETRY and not terminal
    assert provider.calls == (3 if retried else 1)
    if isinstance(make(), VerificationRecoveryError):
        # A control-write failure keeps the node as it was, with no closed span.
        assert node.status is NodeStatus.RUNNING
        assert tracer.spans == []
        return
    skipped = policy is FailurePolicy.SKIP and not terminal
    assert node.status is (NodeStatus.SKIPPED if skipped else NodeStatus.FAILED)
    outcome = "timed_out" if isinstance(make(), TimeoutError) else "failed"
    assert [span.outcome for span in tracer.spans] == [outcome] * provider.calls


class Interrupted(asyncio.CancelledError, Exception):
    """A cancellation that is also an Exception, and not a terminal one."""


@pytest.mark.parametrize("policy", [FailurePolicy.RETRY, FailurePolicy.SKIP])
def test_a_cancellation_that_is_also_an_exception_still_cancels_a_concurrent_node(policy):
    """The node returns to pending and the cancellation propagates; no attempt follows."""
    provider, tracer = Raising(Interrupted), Tracer()
    node, graph = one_node(policy)
    executor = AsyncExecutor(provider=provider, registry=Registry(), tracer=tracer,
                             artifact_dir=None, max_concurrency=1)

    async def scenario():
        with pytest.raises(Interrupted):
            await executor.run(graph)

    asyncio.run(scenario())

    assert is_terminal(Interrupted()) is False
    assert provider.calls == 1
    assert (node.status, node.result) == (NodeStatus.PENDING, None)
    assert [span.outcome for span in tracer.spans] == ["cancelled"]
    assert "failure_class" not in tracer.spans[0].metadata


@pytest.mark.parametrize("case", CASES)
def test_regeneration_settlement_stops_on_the_same_failures_as_before(case):
    """A sibling that fails while a regeneration cancels it stops the run only when terminal."""
    make, _, terminal = CASES[case]
    calls = Counter()

    async def scenario():
        started = asyncio.Event()

        class LateFailure(Provider):
            async def complete(self, system, prompt, model):
                name = prompt.splitlines()[0]
                calls[name] += 1
                if name == "consumer" and calls[name] == 1:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        raise make() from None
                if name == "judge" and calls[name] == 1:
                    await started.wait()
                    return CompletionResult("FAIL")
                return CompletionResult("PASS" if name == "judge" else f"{name}-v{calls[name]}")

        graph = ExecutionGraph([Topology.FORK_JOIN], [
            Node("draft", id="draft"),
            Node("judge", id="judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
            Node("consumer", id="consumer", depends_on=["draft"]),
        ])
        executor = AsyncExecutor(provider=LateFailure(), registry=Registry(), tracer=Tracer(),
                                 artifact_dir=None, max_concurrency=3)
        await asyncio.wait_for(executor.run(graph), 30)
        return graph

    if terminal:
        with pytest.raises(type(make())):
            asyncio.run(scenario())
        assert calls == {"draft": 1, "judge": 1, "consumer": 1}
    else:
        graph = asyncio.run(scenario())
        assert calls == {"draft": 2, "judge": 2, "consumer": 2}
        assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 3


# ---------------------------------------------------------------------------
# Trace spans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("case", [case for case in CASES if case != "verification-recovery"])
def test_a_failed_attempt_span_records_its_failure_class(case, mode):
    make, failure_class, terminal = CASES[case]
    tracer = Tracer()
    node, graph = one_node(FailurePolicy.SKIP)

    if terminal:
        with pytest.raises(type(make())):
            run(mode, Raising(make), graph, tracer)
    else:
        run(mode, Raising(make), graph, tracer)

    [span] = tracer.spans
    assert span.metadata["failure_class"] == failure_class
    assert "failure_class" not in node.metadata


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_each_attempt_keeps_its_own_class_outside_the_summary(mode):
    tracer = Tracer()
    node, graph = one_node(FailurePolicy.RETRY)
    provider = Raising(lambda: rejected(401), lambda: sdk_error(529), then_succeed=True)

    run(mode, provider, graph, tracer)

    assert node.status is NodeStatus.COMPLETED and provider.calls == 3
    assert [span.metadata.get("failure_class") for span in tracer.spans] == [
        "authority_denied", "transient", None,
    ]
    for entry in tracer.summary():
        assert set(entry) == {"node_id", "label", "agent_id", "status", "duration_ms", "error"}
    assert "failure_class" not in node.metadata


# ---------------------------------------------------------------------------
# Native stop reasons
# ---------------------------------------------------------------------------

RESPONSES_MODEL = "gpt-6-astra"


def responses_envelope(**changes):
    prepared = OpenAIResponsesProvider().prepare(
        "System", [ChatMessage("user", "Question")], RESPONSES_MODEL)
    body = {
        "id": "resp_1", "model": RESPONSES_MODEL, "status": "completed", "service_tier": "default",
        "usage": {"input_tokens": 100, "output_tokens": 10,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}},
        "output": [{"type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Answer."}]}],
        **changes,
    }
    return RawResponseEnvelope(prepared, json.dumps(body).encode(), status_code=200,
                               request_id="req_1")


def messages_envelope(**changes):
    prepared = AnthropicMessagesProvider().prepare(
        "System", [ChatMessage("user", "Question")], MESSAGES_MODEL)
    body = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MESSAGES_MODEL,
        "stop_reason": "end_turn", "content": [{"type": "text", "text": "Answer."}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0, "service_tier": "standard"},
        **changes,
    }
    return RawResponseEnvelope(prepared, json.dumps(body).encode(), status_code=200,
                               request_id="req_1")


def refusal_output():
    return [{"type": "message", "status": "completed", "role": "assistant",
             "content": [{"type": "refusal", "refusal": "I can't help with that."}]}]


# Each message is the decode failure's text before stop reasons existed: a
# durable journal compares a rejected call's saved text with it byte for byte.
@pytest.mark.parametrize("changes,message,stop_reason", [
    ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
     "Native response did not complete", "max_tokens"),
    ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
     "Native response did not complete", "content_filter"),
    ({"status": "incomplete", "incomplete_details": {"reason": "a_future_reason"}},
     "Native response did not complete", "incomplete"),
    ({"status": "incomplete", "incomplete_details": None},
     "Native response did not complete", "incomplete"),
    ({"output": refusal_output()}, "Native message does not contain usable text", "refusal"),
    ({"status": "failed", "error": {"code": "server_error"}},
     "Native response did not complete", None),
    ({"output": []}, "Native response contains no output", None),
    ({"model": "gpt-5.6-sol"}, "Native response model differs from requested model", None),
], ids=["max-output-tokens", "content-filter", "other-reason", "no-details", "refusal",
        "failed", "no-output", "other-model"])
def test_responses_decode_sets_the_stop_reason_without_changing_the_message(
        changes, message, stop_reason):
    with pytest.raises(ProviderResponseError) as caught:
        OpenAIResponsesProvider.decode(responses_envelope(**changes))

    error = caught.value
    assert str(error) == message and error.args == (message,)
    assert error.stop_reason == stop_reason
    assert error.billing_result.cost_usd > 0
    assert (classify_failure(error), is_terminal(error)) == ("provider_rejected", True)


@pytest.mark.parametrize("stop,stop_reason", [
    ("max_tokens", "max_tokens"),
    ("model_context_window_exceeded", "model_context_window_exceeded"),
    ("refusal", "refusal"),
    ("pause_turn", "incomplete"),
    ("stop_sequence", None),
    (["max_tokens"], None),
], ids=["max-tokens", "context-window", "refusal", "pause-turn", "stop-sequence", "not-text"])
def test_messages_decode_sets_the_stop_reason_without_changing_the_message(stop, stop_reason):
    with pytest.raises(ProviderResponseError) as caught:
        AnthropicMessagesProvider.decode(messages_envelope(stop_reason=stop))

    error = caught.value
    message = "Messages response did not complete an assistant turn"
    assert str(error) == message and error.args == (message,)
    assert error.stop_reason == stop_reason
    assert error.billing_result.cost_usd > 0
    assert (classify_failure(error), is_terminal(error)) == ("provider_rejected", True)


def test_a_completed_turn_without_text_has_no_stop_reason():
    with pytest.raises(ProviderResponseError) as caught:
        AnthropicMessagesProvider.decode(messages_envelope(content=[]))

    assert str(caught.value) == "Messages response has no content"
    assert caught.value.stop_reason is None


def test_other_response_errors_report_no_stop_reason():
    assert ProviderResponseError("Unresolved native response").stop_reason is None
    assert ProviderAccountingError("Native bill is unresolved").stop_reason is None


@pytest.mark.parametrize("how", ["pickle", "copy", "deepcopy"])
def test_the_stop_reason_survives_pickle_and_copy(how):
    with pytest.raises(ProviderResponseError) as caught:
        OpenAIResponsesProvider.decode(responses_envelope(output=refusal_output()))

    restored = {"pickle": lambda error: pickle.loads(pickle.dumps(error)),
                "copy": copy.copy, "deepcopy": copy.deepcopy}[how](caught.value)
    assert (str(restored), restored.stop_reason) == (str(caught.value), "refusal")


def test_a_durable_refusal_keeps_its_journal_text_and_replays_its_stop_reason(
        tmp_path, monkeypatch):
    transport = MessagesTransport().install(monkeypatch)
    transport.overrides["execution"] = {"stop_reason": "refusal"}
    key = CallKey("execution", "node")
    with SQLiteWorkflowStore(tmp_path / "calls.sqlite3") as store:
        run_id = store.create_run({"goal": "test"}, {}, 1_000_000_000)["run_id"]
        lease = store.acquire_lease(run_id, "tests", 120)

        async def invoke():
            providers = {"main": AnthropicMessagesProvider()}
            async with WorkflowProviderContext(store, lease, providers) as context:
                return await context.for_call("main", key).complete(
                    "System", "Question", MESSAGES_MODEL)

        for _ in range(2):  # The second call decodes the saved response again.
            with pytest.raises(ProviderResponseError) as caught:
                asyncio.run(invoke())
            assert caught.value.stop_reason == "refusal"
            assert str(caught.value) == "Messages response did not complete an assistant turn"
            record = store.lookup_call(run_id, key)
            assert (record["result_state"], record["error"]) == ("rejected", str(caught.value))
        assert len(transport.requests) == 1
