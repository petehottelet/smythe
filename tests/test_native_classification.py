"""Native rejections on an ordinary run under a retry policy.

Without a policy, a failed request to ``OpenAIResponsesProvider`` or
``AnthropicMessagesProvider`` stops an ordinary run: a 429 leaves its bill
unknown, and a refused or truncated reply is a terminal response error. Under a
``RetryPolicy``, a 401, 403, 404, 413 or 429 response with the provider's plain
error body settles at zero cost, as a durable journal settles it, and becomes a
``ProviderRequestRejectedError`` whose class follows its status. A refused or
truncated reply whose charge is known is a ``quality_defect``: when the policy
lists that class, the charge is recorded once and the node may call again.
Only the wire is fake; request preparation, pricing and decoding are real.
"""

from __future__ import annotations

from collections import deque

import pytest

import smythe
from fakes import RATE_LIMITED, HttpError, MessagesTransport, NativeTransport
from smythe import (
    AnthropicMessagesProvider, BudgetValidationError, FileCheckpointStore,
    OpenAIResponsesProvider, ProviderAccountingError, ProviderRequestRejectedError,
    ProviderResponseError, RetryPolicy, Sentinel, Swarm,
)
from smythe import workflow_store
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.pricing_anthropic import MODEL as FABLE
from smythe.registry import Registry
from smythe.tracer import Tracer

# Each adapter's wire fake, model and the nanoUSD charge of one generation.
RESPONSES = ("responses", NativeTransport, OpenAIResponsesProvider, "gpt-6-astra", 1_500_000)
MESSAGES = ("messages", MessagesTransport, AnthropicMessagesProvider, FABLE, 2_535_000)
PROVIDERS = [pytest.param(RESPONSES, id="responses"), pytest.param(MESSAGES, id="messages")]
MODES = ["serial", "parallel"]

# Body changes that make the next reply of each adapter refused or truncated.
REFUSED = {
    "responses": {"output": [{"type": "message", "status": "completed", "role": "assistant",
                              "content": [{"type": "refusal", "refusal": "I can't help."}]}]},
    "messages": {"stop_reason": "refusal"},
}
TRUNCATED = {
    "responses": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    "messages": {"stop_reason": "max_tokens"},
}
# Replies that did not complete for another reason: no attempt follows them.
FILTERED = {
    "responses": {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
    "messages": {"stop_reason": "model_context_window_exceeded"},
}
DEFECTS = {"refusal": (REFUSED, "refusal"), "max-tokens": (TRUNCATED, "max_tokens")}
LISTED = RetryPolicy(retry_classes={"transient", "quality_defect"})


def reply_changes(transport, *changes):
    """Merge each change into the body of the next generation reply, in order."""
    queued, original = deque(changes), transport._response

    def response(sequence, payload, text):
        raw = original(sequence, payload, text)
        raw.update(queued.popleft() if queued else {})
        return raw

    transport._response = response


def make_swarm(monkeypatch, provider, policy, *, parallel=False, **options):
    _, transport_class, provider_class, model, _ = provider
    transport = transport_class().install(monkeypatch)
    swarm = Swarm(model=model, artifact_dir=None, max_budget_usd=5, parallel=parallel,
                  provider=provider_class(api_key="test-key", max_cost_per_call_usd=0.5),
                  retry_policy=policy, **options)
    tracers: list[Tracer] = []
    swarm._tracer_factory = lambda: tracers.append(Tracer()) or tracers[-1]
    return swarm, transport, tracers


def one_node(max_retries=2):
    node = Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY,
                max_retries=max_retries)
    return node, ExecutionGraph([Topology.SERIAL], [node])


def outcomes(transport):
    return [record.outcome for record in transport.log]


def attempts(tracers):
    return [(span.outcome, span.metadata.get("failure_class"))
            for tracer in tracers for span in tracer.spans if span.node_id == "draft"]


def receipts(node):
    return [entry["receipt"] for entry in node.metadata.get("native_receipts", [])]


def usd(nanousd):
    return nanousd / 1_000_000_000


# ---------------------------------------------------------------------------
# Plain HTTP rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_plain_rate_limit_settles_at_zero_cost_and_retries_with_exact_totals(
        monkeypatch, provider, mode):
    name, charge = provider[0], provider[-1]
    swarm, transport, tracers = make_swarm(monkeypatch, provider, RetryPolicy(),
                                           parallel=mode == "parallel")
    transport.schedule("execution", 0, RATE_LIMITED)
    node, graph = one_node()

    result = swarm.execute(graph)

    assert result.output == "Draft accepted output."
    assert outcomes(transport) == ["http_429", "delivered"]
    rejection, accepted = receipts(node)
    assert rejection["pricing_scope"] == "rejected_http_request"
    assert (rejection["http_status"], rejection["error_type"]) == (429, "rate_limit_error")
    assert (rejection["cost_nanousd"], rejection["cost_usd"], rejection["cost_is_complete"]) == (
        0, "0", True)
    assert rejection["request_id"] == transport.log[0].request_id
    assert accepted["cost_nanousd"] == charge
    # The rejection adds nothing, so the total is exactly the accepted call's charge.
    assert result.total_cost_usd == usd(charge)
    assert result.cost_is_complete and not result.cost_contains_estimates
    for marker in ("response_error", "accounting_invalid", "cost_usd_unknown"):
        assert marker not in node.metadata, (name, marker)
    assert attempts(tracers) == [("failed", "transient"), ("completed", None)]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("status,error_type,failure_class", [
    (401, "authentication_error", "authority_denied"),
    (403, "permission_error", "authority_denied"),
    (404, "not_found_error", "input_error"),
    (413, "request_too_large", "input_error"),
])
def test_a_plain_rejection_the_policy_does_not_list_fails_the_node_at_zero_cost(
        monkeypatch, provider, status, error_type, failure_class):
    swarm, transport, tracers = make_swarm(monkeypatch, provider, RetryPolicy())
    transport.schedule("execution", 0, HttpError(status, error_type))
    node, graph = one_node()

    with pytest.raises(ProviderRequestRejectedError) as caught:
        swarm.execute(graph)

    error = caught.value
    assert (error.status_code, error.failure_class) == (status, failure_class)
    assert str(error).startswith(f"Provider rejected the request with HTTP {status} (")
    assert outcomes(transport) == [f"http_{status}"]
    assert node.status is NodeStatus.FAILED
    assert [receipt["cost_nanousd"] for receipt in receipts(node)] == [0]
    assert "response_error" not in node.metadata and "accounting_invalid" not in node.metadata
    assert attempts(tracers) == [("failed", failure_class)]


def with_usage(transport):
    """Give every error body a usage object, which may describe billed work."""
    plain = transport._error_body

    def body(fault, request_id):
        return {**plain(fault, request_id), "usage": {"input_tokens": 100, "output_tokens": 10}}

    transport._error_body = body


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("fault,change", [
    pytest.param(HttpError(400, "invalid_request_error", "context_length_exceeded"), None,
                 id="http-400"),
    pytest.param(RATE_LIMITED, with_usage, id="429-with-usage"),
])
def test_a_rejection_that_may_follow_model_work_stays_unknown_and_terminal(
        monkeypatch, provider, fault, change):
    # Even a policy that lists unknown_outcome cannot retry an unresolved bill.
    policy = RetryPolicy(retry_classes={"transient", "input_error", "unknown_outcome"})
    swarm, transport, _ = make_swarm(monkeypatch, provider, policy)
    transport.schedule("execution", 0, fault)
    if change is not None:
        change(transport)
    node, graph = one_node()

    with pytest.raises(ProviderAccountingError) as caught:
        swarm.execute(graph)

    assert type(caught.value) is ProviderAccountingError
    assert outcomes(transport) == [f"http_{fault.status}"]
    assert node.status is NodeStatus.FAILED
    assert node.metadata["accounting_invalid"] is True and node.metadata["cost_usd_unknown"] is True
    assert "response_error" in node.metadata
    [receipt] = receipts(node)
    assert receipt["cost_is_complete"] is False
    assert receipt.get("pricing_scope") != "rejected_http_request"


# ---------------------------------------------------------------------------
# Refused and truncated replies with a known charge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("defect", DEFECTS)
def test_a_known_charge_refusal_or_truncation_retries_when_quality_defect_is_listed(
        monkeypatch, provider, defect, mode):
    name, charge = provider[0], provider[-1]
    swarm, transport, tracers = make_swarm(monkeypatch, provider, LISTED,
                                           parallel=mode == "parallel")
    reply_changes(transport, DEFECTS[defect][0][name])
    node, graph = one_node()

    result = swarm.execute(graph)

    assert result.output == "Draft accepted output."
    assert outcomes(transport) == ["delivered", "delivered"]
    # Each attempt's charge is recorded once, and the total is their exact sum.
    assert [receipt["cost_nanousd"] for receipt in receipts(node)] == [charge, charge]
    assert result.total_cost_usd == usd(charge) + usd(charge)
    assert result.cost_is_complete and not result.cost_contains_estimates
    assert "response_error" not in node.metadata and "accounting_invalid" not in node.metadata
    assert attempts(tracers) == [("failed", "quality_defect"), ("completed", None)]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("replies,stop_reason,policy", [
    pytest.param(REFUSED, "refusal", RetryPolicy(), id="refusal-default-classes"),
    pytest.param(TRUNCATED, "max_tokens",
                 RetryPolicy(retry_classes={"transient", "provider_rejected"}),
                 id="max-tokens-provider-rejected-listed"),
    pytest.param(FILTERED, None, LISTED, id="other-stop-reason-quality-defect-listed"),
])
def test_a_known_charge_failure_stays_terminal_unless_its_retry_is_listed(
        monkeypatch, provider, replies, stop_reason, policy):
    name, charge = provider[0], provider[-1]
    swarm, transport, tracers = make_swarm(monkeypatch, provider, policy)
    reply_changes(transport, replies[name])
    node, graph = one_node()

    with pytest.raises(ProviderResponseError) as caught:
        swarm.execute(graph)

    assert type(caught.value) is ProviderResponseError
    if stop_reason is not None:
        assert caught.value.stop_reason == stop_reason
    else:
        assert caught.value.stop_reason not in ("refusal", "max_tokens")
    assert outcomes(transport) == ["delivered"]
    assert node.status is NodeStatus.FAILED and "response_error" in node.metadata
    assert [receipt["cost_nanousd"] for receipt in receipts(node)] == [charge]
    assert attempts(tracers) == [("failed", "provider_rejected")]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("fault", ["rate-limited", "refused"])
def test_without_a_policy_a_native_failure_still_stops_the_run(monkeypatch, provider, fault):
    name = provider[0]
    swarm, transport, _ = make_swarm(monkeypatch, provider, None)
    if fault == "rate-limited":
        transport.schedule("execution", 0, RATE_LIMITED)
    else:
        reply_changes(transport, REFUSED[name])
    node, graph = one_node()

    error = ProviderAccountingError if fault == "rate-limited" else ProviderResponseError
    with pytest.raises(error) as caught:
        swarm.execute(graph)

    assert type(caught.value) is error
    assert len(transport.log) == 1 and node.status is NodeStatus.FAILED
    assert "response_error" in node.metadata
    assert all(receipt.get("pricing_scope") != "rejected_http_request"
               for receipt in receipts(node))


# ---------------------------------------------------------------------------
# Resume and settlement state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("fault", ["rate-limited", "refused"])
def test_a_settled_failure_leaves_the_run_resumable_with_each_charge_counted_once(
        tmp_path, monkeypatch, provider, fault):
    name, charge = provider[0], provider[-1]
    swarm, transport, _ = make_swarm(monkeypatch, provider, LISTED,
                                     checkpoint_store=FileCheckpointStore(tmp_path))
    if fault == "rate-limited":
        transport.schedule("execution", 0, RATE_LIMITED)
        error, first_charge = ProviderRequestRejectedError, 0
    else:
        reply_changes(transport, REFUSED[name])
        error, first_charge = smythe.NativeOutputDefectError, charge
    node, graph = one_node(max_retries=0)

    with pytest.raises(error) as caught:
        swarm.execute(graph)
    if fault == "refused":
        assert caught.value.stop_reason == "refusal"
        assert isinstance(caught.value, smythe.IncompleteOutputError)
    # The failed run saved no unresolved marker, so it resumes like any failure.
    assert "response_error" not in node.metadata
    [execution_id] = FileCheckpointStore(tmp_path).list_ids()

    result = swarm.resume(execution_id)

    assert result.output == "Draft accepted output."
    assert len(transport.log) == 2
    assert result.total_cost_usd == usd(first_charge) + usd(charge)
    assert result.cost_is_complete and not result.cost_contains_estimates


def test_a_released_refusal_is_not_reported_again_as_unresolved(monkeypatch):
    transport = NativeTransport().install(monkeypatch)
    reply_changes(transport, REFUSED["responses"])
    provider = OpenAIResponsesProvider(api_key="test-key", max_cost_per_call_usd=0.5)
    executor = Executor(provider=provider, registry=Registry(), tracer=Tracer(),
                        budget=Sentinel(5), artifact_dir=None, retry_policy=LISTED)
    node, graph = one_node()
    node.metadata["model"] = "gpt-6-astra"

    executor.run(graph)

    assert node.status is NodeStatus.COMPLETED and len(transport.log) == 2
    # Accounting trouble found later is reported as itself, never as the
    # refusal whose charge was settled and released for another attempt.
    node.metadata["accounting_invalid"] = True
    with pytest.raises(BudgetValidationError) as caught:
        executor._raise_unresolved_response(node)
    assert not isinstance(caught.value, ProviderResponseError)


def test_the_journal_and_ordinary_runs_share_one_set_of_rejection_rules():
    from smythe import _http_rejection

    for name in ("_REJECTION_STATUS_ERRORS", "_ERROR_BODY_FIELDS", "_ERROR_OBJECT_FIELDS",
                 "_http_rejection_receipt"):
        assert getattr(workflow_store, name) is getattr(_http_rejection, name), name
    assert workflow_store._http_rejection_receipt.__module__ == "smythe._http_rejection"
