"""Every exception a node attempt raises has one failure class.

``classify_failure`` names how an attempt failed and ``is_terminal`` whether
the run stops at once, whatever the node's failure policy. Both executors take
that decision from ``is_terminal`` alone, and it stops on exactly the
exceptions they stopped on before failure classes existed. A class describes
a failure; it never changes what is retried.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import pickle

import pytest

from fakes import _STATUS_ERRORS
from smythe._failures import FAILURE_CLASSES, classify_failure, is_terminal, status_failure_class
from smythe.async_executor import AsyncExecutor
from smythe.budget import (
    BudgetEstimateRequired, BudgetReconciliationError, BudgetValidationError, SentinelAlert,
)
from smythe.executor import Executor
from smythe.executor_base import ArtifactIntegrityError, NodeFinalizationError
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import (
    CompletionResult, OutputRefusedError, OutputTruncatedError, Provider,
    ProviderAccountingCancelledError, ProviderAccountingError, ProviderResponseError,
)
from smythe.registry import Registry
from smythe.tools import ToolLoopLimitError
from smythe.tracer import Tracer
from smythe.verifier import VerificationRecoveryError
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_provider import ProviderRequestRejectedError, WorkflowQuoteError
from smythe.workflow_store import WorkflowError, WorkflowStateError
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
        "security_stop",
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
