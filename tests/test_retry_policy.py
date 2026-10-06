"""An opt-in ``RetryPolicy`` makes the executor the only retry owner of a node's calls.

Under a policy only the listed failure classes are retried, a transient status
error's ``Retry-After`` sets the wait before its retry, no wait is longer than
``backoff_max_s``, and ``deadline_s`` bounds a node's attempts and waits
together. A run refuses, before any call, a provider whose SDK may send a failed
call again on its own. Waits go through an injected ``sleep`` that advances a
``ManualClock`` the tracer reads, so every wait and deadline here is exact and
nothing sleeps.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
import inspect
import json
import math
import sys
from types import SimpleNamespace

import pytest

from fakes import _STATUS_ERRORS, HttpError, ManualClock, phase_of
from smythe import RetryDeadlineError, RetryPolicy
from smythe._failures import classify_failure, is_terminal
from smythe.async_executor import AsyncExecutor
from smythe.checkpoint import FileCheckpointStore
from smythe.constrained_planner import ConstrainedArchitect, SubGraphTemplate
from smythe.constrained_prompts import CONSTRAINED_SYSTEM_PROMPT
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.planner import LLMArchitect, SimpleArchitect
from smythe.pricing_anthropic import MODEL as FABLE
from smythe.provider import (
    AnthropicProvider, CompletionResult, GeminiProvider, OfflineProvider, OpenAIImageProvider,
    OpenAIProvider, OutputTruncatedError, Provider, ProviderAccountingCancelledError,
    ProviderFeatures,
)
from smythe.provider_messages import AnthropicMessagesProvider
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.scheduling import _retry_after_s
from smythe.supervisor import LLMSupervisor
from smythe.swarm import Swarm
from smythe.synthesizer import SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.tools import ToolCall, ToolRuntime, ToolSession, ToolSpec
from smythe.tracer import Tracer
from smythe.workflow_store import SQLiteWorkflowStore

MODES = ["serial", "parallel"]
HANG = object()  # A call that never answers.


class Scripted(Provider):
    """Answers or fails each call in turn, repeating the last outcome; it sends each call once.

    An outcome is reply text, an ``HttpError`` raised as its SDK status error,
    ``HANG``, or a callable returning a ``CompletionResult``, an exception to
    raise, or an awaitable that gives either. ``started`` holds the clock
    reading at the start of each call.
    """

    def __init__(self, *outcomes, clock: ManualClock | None = None) -> None:
        self.outcomes, self.clock, self.started = outcomes, clock, []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0)

    async def complete(self, system, prompt, model):
        self.started.append(self.clock.monotonic() if self.clock else None)
        outcome = self.outcomes[min(len(self.started), len(self.outcomes)) - 1]
        if outcome is HANG:
            await asyncio.Event().wait()
        if isinstance(outcome, HttpError):
            raise outcome.exception()
        if callable(outcome):
            outcome = outcome()
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return CompletionResult(outcome, prompt_tokens=1, completion_tokens=1)


class Undeclared(Provider):
    """A custom provider that declares nothing, including its SDK retries."""

    async def complete(self, system, prompt, model):
        return CompletionResult("done")


class Upper:
    """Draws the top of every jitter range, so a wait equals its ceiling."""

    def uniform(self, low, high):
        return high


def make_executor(mode, provider, clock, policy=None, **options):
    """An executor whose waits advance *clock*, which its tracer reads."""
    waits: list[float] = []

    def sleep(seconds):
        waits.append(seconds)
        clock.advance(seconds)

    tracer = Tracer(clock=clock.monotonic)
    cls = Executor if mode == "serial" else AsyncExecutor
    executor = cls(provider=provider, registry=Registry(), tracer=tracer, artifact_dir=None,
                   retry_policy=policy, sleep=options.pop("sleep", sleep), **options)
    return executor, tracer, waits


def run(executor, *nodes):
    graph = ExecutionGraph([Topology.SERIAL], list(nodes))
    if isinstance(executor, Executor):
        return executor.run(graph)
    return asyncio.run(executor.run(graph))


def retrying(**options):
    return Node(id="step", label="Step", failure_policy=FailurePolicy.RETRY, **options)


def attempts(tracer):
    return [(span.attempt, span.outcome, span.metadata.get("failure_class"))
            for span in tracer.spans]


@pytest.fixture
def attempt_timeouts(monkeypatch):
    """Record the timeout of every ``asyncio.wait_for`` and ``asyncio.timeout``, in order.

    An attempt timed by its node's ``timeout_s`` runs under ``wait_for``, and
    one timed by the retry deadline under ``timeout``; the timers are unchanged.
    """
    timeouts: list[float | None] = []
    real_wait_for, real_timeout = asyncio.wait_for, asyncio.timeout

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    def timeout(delay):
        timeouts.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    monkeypatch.setattr(asyncio, "timeout", timeout)
    return timeouts


def forbid_calls(monkeypatch, *classes):
    """Record every call or client creation of these provider classes from here on."""
    calls: list[str] = []
    for cls in classes or (AnthropicProvider, OpenAIProvider, Undeclared):
        for name in ("complete", "chat", "_get_client"):
            def record(self, *args, _name=f"{cls.__name__}.{name}", **kwargs):
                calls.append(_name)
                raise AssertionError(f"{_name} ran before the refusal")
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, record)
    return calls


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_the_default_policy_retries_transient_failures_and_honors_retry_after():
    policy = RetryPolicy()
    assert policy.retry_classes == frozenset({"transient"})
    assert (policy.honor_retry_after, policy.backoff_max_s, policy.deadline_s) == (True, 60.0, None)
    with pytest.raises(FrozenInstanceError):
        policy.deadline_s = 5
    listed = RetryPolicy(retry_classes=["transient", "provider_rejected"], deadline_s=30)
    assert listed.retry_classes == frozenset({"transient", "provider_rejected"})
    assert listed == RetryPolicy(retry_classes={"provider_rejected", "transient"}, deadline_s=30.0)
    assert hash(RetryPolicy()) == hash(RetryPolicy())


@pytest.mark.parametrize("options", [
    {"retry_classes": "transient"}, {"retry_classes": None}, {"retry_classes": {"deadline"}},
    {"retry_classes": {"terminal_accounting"}}, {"retry_classes": {"rate_limit"}},
    {"retry_classes": {1}}, {"honor_retry_after": 1}, {"honor_retry_after": "yes"},
    {"backoff_max_s": -1}, {"backoff_max_s": math.nan}, {"backoff_max_s": math.inf},
    {"backoff_max_s": True}, {"backoff_max_s": "5"}, {"backoff_max_s": None},
    {"deadline_s": 0}, {"deadline_s": -1}, {"deadline_s": math.inf}, {"deadline_s": True},
    {"deadline_s": 10 ** 400},
], ids=lambda options: ",".join(f"{key}={value!r}"[:32] for key, value in options.items()))
def test_a_malformed_policy_is_refused_when_built(options):
    with pytest.raises(ValueError):
        RetryPolicy(**options)


def test_executor_options_are_checked_when_built():
    options = dict(provider=Scripted("done"), registry=Registry(), tracer=Tracer())
    with pytest.raises(TypeError, match="retry_policy"):
        AsyncExecutor(**options, retry_policy={"deadline_s": 5})
    with pytest.raises(TypeError, match="sleep"):
        Executor(**options, sleep=5)
    with pytest.raises(TypeError, match="rng"):
        Executor(**options, rng=object())
    # Durable runs keep their journal-derived retries: journal-managed calls
    # take a policy only with the journal's count of each node's attempts.
    with pytest.raises(ValueError, match="retry_policy"):
        AsyncExecutor(**options, retry_policy=RetryPolicy(),
                      provider_call_factory=lambda node, phase, attempt, turn: None)
    with pytest.raises(TypeError, match="attempts_spent"):
        AsyncExecutor(**options, attempts_spent=lambda node: 0)  # No policy to count for.
    with pytest.raises(TypeError, match="attempts_spent"):
        AsyncExecutor(**options, retry_policy=RetryPolicy(), attempts_spent=0)
    with pytest.raises(ValueError, match="attempts_used"):
        AsyncExecutor(**options, retry_policy=RetryPolicy(persist_attempts=True),
                      attempts_spent=lambda node: 0, attempts_used={})
    AsyncExecutor(**options, retry_policy=RetryPolicy(), attempts_spent=lambda node: 0,
                  provider_call_factory=lambda node, phase, attempt, turn: None)


# ---------------------------------------------------------------------------
# Which failures are retried
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("status", [401, 403, 404, 413])
def test_authority_and_input_rejections_are_not_retried_under_the_policy(mode, status):
    clock = ManualClock()
    provider = Scripted(HttpError(status, "error", retry_after_s=1), "done", clock=clock)
    executor, tracer, waits = make_executor(mode, provider, clock, RetryPolicy())
    node = retrying(max_retries=2)

    with pytest.raises(_STATUS_ERRORS[status]):
        run(executor, node)

    assert len(provider.started) == 1 and waits == []
    assert node.status is NodeStatus.FAILED
    expected = "authority_denied" if status in (401, 403) else "input_error"
    assert attempts(tracer) == [(0, "failed", expected)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("status", [401, 403, 404, 413])
def test_without_a_policy_retry_still_retries_every_rejection(mode, status):
    clock = ManualClock()
    provider = Scripted(HttpError(status, "error", retry_after_s=1), "done", clock=clock)
    executor, _, waits = make_executor(mode, provider, clock)
    node = retrying(max_retries=2)

    run(executor, node)

    assert node.status is NodeStatus.COMPLETED and len(provider.started) == 2
    # Without a policy no header is read: the default backoff of 0 means no wait.
    assert waits == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("classes,retried", [
    pytest.param({"transient"}, False, id="default-classes"),
    pytest.param({"transient", "provider_rejected"}, True, id="listed"),
])
def test_a_truncated_reply_is_retried_only_when_its_class_is_listed(mode, classes, retried):
    clock = ManualClock()
    provider = Scripted(lambda: CompletionResult("partial", stop_reason="max_tokens"), "done",
                        clock=clock)
    executor, tracer, _ = make_executor(mode, provider, clock, RetryPolicy(retry_classes=classes))
    node = retrying(max_retries=1)

    if retried:
        run(executor, node)
        assert node.status is NodeStatus.COMPLETED
    else:
        with pytest.raises(OutputTruncatedError):
            run(executor, node)
    assert len(provider.started) == (2 if retried else 1)
    assert attempts(tracer)[0] == (0, "failed", "provider_rejected")


class ToolCaller(Provider):
    """Calls the lookup tool once, then answers; it sends each call once."""

    def __init__(self) -> None:
        self.turns: list = []

    def feature_support(self, model):
        return ProviderFeatures(tools=True, sdk_retries=0)

    async def complete(self, system, prompt, model):
        raise NotImplementedError

    async def chat(self, system, messages, model, tools=None):
        self.turns.append(messages[-1])
        if len(self.turns) == 1:
            return CompletionResult("", tool_calls=[ToolCall("call-1", "lookup", {})],
                                    stop_reason="tool_use")
        return CompletionResult("answered")


class FailingSession(ToolSession):
    @property
    def tools(self):
        return [ToolSpec("lookup", "Look up a fact", {"type": "object"})]

    async def call(self, tool_call):
        raise RuntimeError("lookup failed")


class FailingTools(ToolRuntime):
    def open(self, agent):
        @asynccontextmanager
        async def session():
            yield FailingSession()

        return session()


@pytest.mark.parametrize("mode", MODES)
def test_a_tool_error_fed_back_to_the_model_is_not_a_retry(mode):
    clock = ManualClock()
    provider = ToolCaller()
    executor, tracer, waits = make_executor(mode, provider, clock, RetryPolicy(deadline_s=10),
                                            tool_runtime=FailingTools())
    node = retrying(max_retries=2)

    run(executor, node)

    assert node.status is NodeStatus.COMPLETED and node.result == "answered"
    assert attempts(tracer) == [(0, "completed", None)] and waits == []
    assert [(result.content, result.is_error) for result in provider.turns[1].tool_results] == [
        ("lookup failed", True),
    ]


# ---------------------------------------------------------------------------
# How long a retry waits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_rate_limit_is_retried_after_exactly_its_retry_after(mode):
    clock = ManualClock()
    provider = Scripted(HttpError(429, "rate_limit_error", retry_after_s=7), "done", clock=clock)
    # The header replaces the 1s backoff the policy would otherwise wait.
    executor, tracer, waits = make_executor(mode, provider, clock, RetryPolicy(),
                                            retry_backoff_s=1.0, rng=Upper())
    node = retrying(max_retries=2)

    run(executor, node)

    assert node.status is NodeStatus.COMPLETED
    assert waits == [7] and provider.started == [0, 7]
    assert attempts(tracer) == [(0, "failed", "transient"), (1, "completed", None)]
    assert tracer.timing()["waiting_ms"]["retry_backoff"] == 7000


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault,policy,wait", [
    pytest.param(HttpError(429, "rate_limit_error", retry_after_s=120),
                 RetryPolicy(backoff_max_s=30), 30, id="cut-to-the-cap"),
    pytest.param(HttpError(429, "rate_limit_error", retry_after_s=7),
                 RetryPolicy(honor_retry_after=False), 1, id="not-honored"),
    pytest.param(HttpError(403, "permission_error", retry_after_s=7),
                 RetryPolicy(retry_classes={"authority_denied"}), 1, id="not-a-transient-status"),
])
def test_retry_after_is_capped_and_read_only_where_it_applies(mode, fault, policy, wait):
    clock = ManualClock()
    provider = Scripted(fault, "done", clock=clock)
    executor, _, waits = make_executor(mode, provider, clock, policy, retry_backoff_s=1.0,
                                       rng=Upper())

    run(executor, retrying(max_retries=1))

    assert waits == [wait]


@pytest.mark.parametrize("status,headers,expected", [
    pytest.param(429, {"retry-after": "2"}, 2.0, id="seconds"),
    pytest.param(429, {"Retry-After": "1.5"}, 1.5, id="any-case"),
    pytest.param(500, {"retry-after-ms": "250", "retry-after": "9"}, 0.25, id="milliseconds-first"),
    pytest.param(529, {"retry-after": "0"}, 0.0, id="zero"),
    pytest.param(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None, id="date"),
    pytest.param(429, {"retry-after": "-1"}, None, id="negative"),
    pytest.param(429, {"retry-after": "inf"}, None, id="infinite"),
    pytest.param(429, {"retry-after": "nan"}, None, id="nan"),
    pytest.param(429, {}, None, id="absent"),
    pytest.param(403, {"retry-after": "2"}, None, id="authority-denied"),
    pytest.param(413, {"retry-after": "2"}, None, id="input-error"),
])
def test_retry_after_is_read_only_from_a_transient_status_error(status, headers, expected):
    assert _retry_after_s(_STATUS_ERRORS[status](status, {}, headers)) == expected


def test_retry_after_needs_a_status_and_a_readable_response():
    class NoResponse(Exception):
        status_code = 429

    assert _retry_after_s(RuntimeError("reset")) is None
    assert _retry_after_s(NoResponse("rate limited")) is None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy,expected", [
    pytest.param(None, [4, 8, 16, 32], id="no-policy-uncapped"),
    pytest.param(RetryPolicy(backoff_max_s=10), [4, 8, 10, 10], id="capped"),
])
def test_backoff_doubles_up_to_the_policy_cap(mode, policy, expected):
    clock = ManualClock()
    reset = lambda: ConnectionError("reset")  # noqa: E731
    provider = Scripted(reset, reset, reset, reset, "done", clock=clock)
    executor, _, waits = make_executor(mode, provider, clock, policy, retry_backoff_s=4.0,
                                       rng=Upper())

    run(executor, retrying(max_retries=4))

    assert waits == expected


@pytest.mark.parametrize("mode", MODES)
def test_an_awaitable_sleep_is_awaited(mode):
    clock = ManualClock()
    waits = []

    async def sleep(seconds):
        await asyncio.sleep(0)
        waits.append(seconds)

    provider = Scripted(HttpError(429, "rate_limit_error", retry_after_s=3), "done", clock=clock)
    executor, _, _ = make_executor(mode, provider, clock, RetryPolicy(), sleep=sleep)

    run(executor, retrying(max_retries=1))

    assert waits == [3]


# ---------------------------------------------------------------------------
# The shared deadline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_deadline_bounds_attempts_and_waits_together(mode):
    clock = ManualClock()
    provider = Scripted(HttpError(429, "rate_limit_error", retry_after_s=4), clock=clock)
    executor, tracer, waits = make_executor(mode, provider, clock, RetryPolicy(deadline_s=10))
    node = retrying(max_retries=10)

    with pytest.raises(RetryDeadlineError) as caught:
        run(executor, node)

    # A third wait would end at 12s, past the 10s deadline, so no fourth call is made.
    assert provider.started == [0, 4, 8] and waits == [4, 4]
    assert classify_failure(caught.value) == "deadline" and not is_terminal(caught.value)
    assert isinstance(caught.value, TimeoutError)
    assert classify_failure(caught.value.__cause__) == "transient"
    assert node.status is NodeStatus.FAILED and node.result == str(caught.value)
    assert "deadline" in node.result
    # The refused fourth attempt makes no call but is recorded.
    assert attempts(tracer) == [*((attempt, "failed", "transient") for attempt in range(3)),
                                (3, "timed_out", "deadline")]


@pytest.mark.parametrize("mode", MODES)
def test_a_retry_the_deadline_refuses_is_recorded_on_the_trace(mode):
    clock = ManualClock()
    provider = Scripted(HttpError(429, "rate_limit_error", retry_after_s=4), clock=clock)
    executor, tracer, _ = make_executor(mode, provider, clock, RetryPolicy(deadline_s=10))

    with pytest.raises(RetryDeadlineError) as caught:
        run(executor, retrying(max_retries=10))

    # The fourth attempt would start at 12s, past the 10s deadline: its span
    # says why the node stopped, with no duration, and no call is made.
    assert provider.started == [0, 4, 8]
    assert attempts(tracer) == [*((attempt, "failed", "transient") for attempt in range(3)),
                                (3, "timed_out", "deadline")]
    refused = tracer.spans[-1]
    assert refused.status == "failed" and refused.duration_ms == 0
    assert tracer.summary()[-1] == {
        "node_id": "step", "label": "Step", "agent_id": None, "status": "failed",
        "duration_ms": 0.0, "error": str(caught.value),
    }
    timing = tracer.timing()
    assert timing["summed_node_ms"] == 0 and timing["waiting_ms"]["retry_backoff"] == 8000


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("timeout_s", [None, 30.0])
def test_each_attempt_times_out_at_the_remaining_deadline(mode, timeout_s, attempt_timeouts):
    clock = ManualClock()
    provider = Scripted(HttpError(429, "rate_limit_error", retry_after_s=9.95), HANG, clock=clock)
    executor, tracer, waits = make_executor(mode, provider, clock, RetryPolicy(deadline_s=10))
    node = retrying(max_retries=5, timeout_s=timeout_s)

    with pytest.raises(RetryDeadlineError):
        run(executor, node)

    # The 9.95s wait left the second attempt 0.05s of the deadline.
    assert attempt_timeouts == [10, pytest.approx(0.05)]
    assert len(provider.started) == 2 and waits == [9.95]
    assert attempts(tracer) == [(0, "failed", "transient"), (1, "timed_out", "deadline")]
    assert node.status is NodeStatus.FAILED


@pytest.mark.parametrize("mode", MODES)
def test_a_node_timeout_inside_the_deadline_is_transient_and_retried(mode, attempt_timeouts):
    clock = ManualClock()
    provider = Scripted(HANG, "done", clock=clock)
    executor, tracer, _ = make_executor(mode, provider, clock, RetryPolicy(deadline_s=100))
    node = retrying(max_retries=1, timeout_s=0.05)

    run(executor, node)

    assert attempt_timeouts == [0.05, 0.05]
    assert node.status is NodeStatus.COMPLETED
    assert attempts(tracer) == [(0, "timed_out", "transient"), (1, "completed", None)]


async def time_out_on_its_own_timer():
    """Time out as an HTTP client's own request timer does, with an ``asyncio`` timeout."""
    async with asyncio.timeout(0):
        await asyncio.Event().wait()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", [
    pytest.param(None, id="no-policy"),
    pytest.param(RetryPolicy(), id="no-deadline"),
    pytest.param(RetryPolicy(deadline_s=100), id="deadline"),
])
@pytest.mark.parametrize("timeout_s", [None, 300.0])
@pytest.mark.parametrize("fault", [
    pytest.param(lambda: TimeoutError("request timed out after 30s"), id="raised"),
    pytest.param(time_out_on_its_own_timer, id="own-timer"),
])
def test_a_timeout_the_call_raises_itself_is_transient_and_retried(mode, policy, timeout_s, fault):
    clock = ManualClock()
    provider = Scripted(fault, "done", clock=clock)
    executor, tracer, waits = make_executor(mode, provider, clock, policy)
    # Under the deadline, a timeout_s of None or 300s leaves the deadline to time
    # each attempt, and it never runs out: the clock stays at 0 of its 100s.
    node = retrying(max_retries=2, timeout_s=timeout_s)

    run(executor, node)

    assert node.status is NodeStatus.COMPLETED and node.result == "done"
    assert provider.started == [0, 0] and waits == []
    assert attempts(tracer) == [(0, "timed_out", "transient"), (1, "completed", None)]


async def bill_unknown_when_cancelled():
    """Hang until cancelled, then report that the native call's bill is unknown."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError as exc:
        raise ProviderAccountingCancelledError(
            "Native dispatch interrupted; bill unknown", receipt={"cost_is_complete": False},
        ) from exc


@pytest.mark.parametrize("mode", MODES)
def test_an_unknown_bill_when_the_deadline_runs_out_stays_terminal(mode):
    clock = ManualClock()
    provider = Scripted(bill_unknown_when_cancelled, "done", clock=clock)
    executor, _, _ = make_executor(mode, provider, clock, RetryPolicy(deadline_s=0.05))
    node = retrying(max_retries=2)

    with pytest.raises(ProviderAccountingCancelledError):
        run(executor, node)

    assert len(provider.started) == 1
    assert node.status is NodeStatus.FAILED and node.metadata["accounting_invalid"]


@pytest.mark.parametrize("mode", MODES)
def test_without_a_deadline_a_node_keeps_its_own_timeout(mode, attempt_timeouts):
    clock = ManualClock()
    executor, _, _ = make_executor(mode, Scripted("done", clock=clock), clock, RetryPolicy())

    run(executor, retrying(timeout_s=30), Node(id="untimed", label="Untimed"))

    assert attempt_timeouts == [30]


# ---------------------------------------------------------------------------
# One retry owner
# ---------------------------------------------------------------------------


def with_client(provider, **client):
    provider._client = SimpleNamespace(**client)
    return provider


@pytest.mark.parametrize("make_provider,model,declared", [
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), "claude-opus-5-5", None,
                 id="anthropic-sdk-default"),
    pytest.param(lambda: AnthropicProvider(api_key="test-key", max_retries=0), "claude-opus-5-5", 0,
                 id="anthropic-zero"),
    pytest.param(lambda: AnthropicProvider(api_key="test-key", max_retries=3), "claude-opus-5-5", 3,
                 id="anthropic-three"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), "gpt-5", None,
                 id="openai-sdk-default"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key", max_retries=0), "gpt-5", 0,
                 id="openai-zero"),
    pytest.param(lambda: OpenAIImageProvider(api_key="test-key", max_retries=0), "gpt-image-1", 0,
                 id="openai-image-zero"),
    pytest.param(lambda: GeminiProvider(api_key="test-key"), "gemini-3-pro", 0, id="gemini"),
    pytest.param(OfflineProvider, "offline", 0, id="offline"),
    pytest.param(lambda: OpenAIResponsesProvider(api_key="test-key"), "gpt-6-astra", 0,
                 id="responses"),
    pytest.param(lambda: AnthropicMessagesProvider(api_key="test-key"), FABLE, 0, id="messages"),
    pytest.param(Undeclared, "custom", None, id="custom"),
    # A caller-owned client keeps its own setting, whatever the provider was given.
    pytest.param(lambda: with_client(AnthropicProvider(api_key="test-key", max_retries=0),
                                     max_retries=2), "claude-opus-5-5", 2,
                 id="client-with-retries"),
    pytest.param(lambda: with_client(OpenAIProvider(api_key="test-key"), max_retries=0), "gpt-5", 0,
                 id="client-without-retries"),
    pytest.param(lambda: with_client(OpenAIProvider(api_key="test-key", max_retries=0)), "gpt-5",
                 None, id="client-without-a-setting"),
    pytest.param(lambda: with_client(GeminiProvider(api_key="test-key")), "gemini-3-pro", None,
                 id="gemini-client"),
])
def test_providers_declare_the_retries_their_sdk_makes(make_provider, model, declared):
    assert make_provider().feature_support(model).sdk_retries == declared


@pytest.mark.parametrize("max_retries,expected", [
    pytest.param(None, {"api_key": "test-key"}, id="sdk-default"),
    pytest.param(0, {"api_key": "test-key", "max_retries": 0}, id="disabled"),
])
def test_anthropic_provider_passes_max_retries_to_the_sdk_only_when_set(monkeypatch, max_retries,
                                                                        expected):
    created = []
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(
        AsyncAnthropic=lambda **options: created.append(options) or SimpleNamespace(**options)))

    AnthropicProvider(api_key="test-key", max_retries=max_retries)._get_client()

    assert created == [expected]


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_anthropic_provider_refuses_a_malformed_max_retries(value):
    with pytest.raises(ValueError, match="max_retries"):
        AnthropicProvider(api_key="test-key", max_retries=value)


@pytest.mark.parametrize("entry", ["plan", "execute", "execute-async", "parallel", "graph"])
@pytest.mark.parametrize("make_provider,model", [
    pytest.param(lambda: AnthropicProvider(api_key="test-key"), "claude-opus-5-5",
                 id="anthropic-sdk-default"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key"), "gpt-5", id="openai-sdk-default"),
    pytest.param(lambda: OpenAIProvider(api_key="test-key", max_retries=2), "gpt-5",
                 id="openai-two"),
    pytest.param(Undeclared, "custom", id="custom-undeclared"),
])
def test_a_provider_whose_sdk_may_retry_is_refused_before_any_call(monkeypatch, make_provider,
                                                                   model, entry):
    calls = forbid_calls(monkeypatch)
    swarm = Swarm(model=model, provider=make_provider(), retry_policy=RetryPolicy(),
                  artifact_dir=None, parallel=entry == "parallel")
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])

    with pytest.raises(ValueError, match="retry_policy"):
        if entry == "plan":
            swarm.plan(Task("Report"))
        elif entry == "execute-async":
            asyncio.run(swarm.execute_async(Task("Report")))
        elif entry == "graph":
            swarm.execute(graph)
        else:
            swarm.execute(Task("Report"))

    assert calls == []


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_with_a_policy_refuses_hidden_sdk_retries_before_any_node_call(mode,
                                                                                   monkeypatch):
    calls = forbid_calls(monkeypatch)
    clock = ManualClock()
    executor, _, _ = make_executor(mode, AnthropicProvider(api_key="test-key"), clock,
                                   RetryPolicy())
    node = Node(id="draft", label="Draft", metadata={"model": "claude-opus-5-5"})

    with pytest.raises(ValueError, match="retry_policy"):
        run(executor, node)

    assert calls == [] and node.status is NodeStatus.PENDING


def test_a_provider_that_sends_each_call_once_runs_under_a_policy(monkeypatch):
    created, requests = [], []

    async def create(**request):
        requests.append(request)
        if len(requests) == 1:
            raise HttpError(429, "rate_limit_error", retry_after_s=0).exception()
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Drafted")],
                               stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=3, output_tokens=2))

    def client(**options):
        created.append(options)
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=client))
    swarm = Swarm(model="claude-opus-5-5", retry_policy=RetryPolicy(), artifact_dir=None,
                  provider=AnthropicProvider(api_key="test-key", max_retries=0))

    result = swarm.execute(ExecutionGraph([Topology.SERIAL], [retrying(max_retries=1)]))

    assert result.output == "Drafted" and len(requests) == 2
    assert all(options == {"api_key": "test-key", "max_retries": 0} for options in created)


REPLIES = {
    "planning": json.dumps({"topology": ["serial"], "nodes": [{"id": "draft", "label": "Draft"}]}),
    "selection": json.dumps([{"template": "draft"}]),
    "routing": "autonomous",
    "supervision": json.dumps({"change": False}),
}


class Answering(Provider):
    """Answers a call of any phase of a run and declares ``sdk_retries``; ``calls`` names each phase."""

    def __init__(self, sdk_retries: int | None = 0) -> None:
        self.sdk_retries, self.calls = sdk_retries, []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=self.sdk_retries)

    async def complete(self, system, prompt, model):
        phase = "selection" if system == CONSTRAINED_SYSTEM_PROMPT else phase_of(system)
        self.calls.append(phase)
        return CompletionResult(REPLIES.get(phase, "done"), prompt_tokens=1, completion_tokens=1)


def draft_template():
    return SubGraphTemplate("draft", "Write a draft",
                            lambda task, **params: ([Node(id="draft", label="Draft")], Registry()))


# Each Swarm option that gives a run a provider besides its own, and the phase
# of the call that provider answers.
OTHER_CALLERS = {
    "planning-provider": (lambda other: {"planning_provider": other}, "planning"),
    "llm-architect": (lambda other: {
        "architect": LLMArchitect(provider=other, planning_model="custom"),
    }, "planning"),
    "constrained-architect": (lambda other: {
        "architect": ConstrainedArchitect(provider=other, templates=[draft_template()],
                                          model="custom"),
    }, "selection"),
    "router-tier": (lambda other: {
        "router": WhiteRabbit(autonomous=LLMArchitect(provider=other, planning_model="custom")),
    }, "planning"),
    "router-classifier": (lambda other: {
        "router": WhiteRabbit(autonomous=SimpleArchitect(), classifier_provider=other,
                              classifier_model="custom"),
    }, "routing"),
    "supervisor": (lambda other: {
        "supervisor": LLMSupervisor(other, model="custom"), "max_revisions": 1,
    }, "supervision"),
    "supervisor-node-model": (lambda other: {
        "supervisor": LLMSupervisor(other), "max_revisions": 1,
    }, "supervision"),
}


def enter(swarm, entry, task_or_graph):
    if entry == "plan":
        return swarm.plan(task_or_graph)
    if entry in ("execute-async", "graph-async"):
        return asyncio.run(swarm.execute_async(task_or_graph))
    return swarm.execute(task_or_graph)


@pytest.mark.parametrize("entry", ["plan", "execute", "execute-async", "parallel"])
@pytest.mark.parametrize("caller", OTHER_CALLERS)
@pytest.mark.parametrize("sdk_retries", [None, 2], ids=["undeclared", "two"])
def test_every_provider_a_run_can_call_is_refused_before_any_call_when_its_sdk_may_retry(
    caller, sdk_retries, entry,
):
    provider, other = Answering(), Answering(sdk_retries)
    options, _ = OTHER_CALLERS[caller]
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  parallel=entry == "parallel", **options(other))

    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        enter(swarm, entry, Task("Report"))

    assert provider.calls == [] and other.calls == []


@pytest.mark.parametrize("entry", ["execute", "parallel"])
@pytest.mark.parametrize("caller", OTHER_CALLERS)
def test_a_run_whose_every_provider_sends_each_call_once_runs_under_a_policy(caller, entry):
    provider, other = Answering(), Answering(0)
    options, phase = OTHER_CALLERS[caller]
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  parallel=entry == "parallel", **options(other))

    result = swarm.execute(Task("Report"))

    assert result.output == "done" and other.calls == [phase]
    assert provider.calls[-1] == "execution"


@pytest.mark.parametrize("entry", ["graph", "graph-async", "parallel"])
@pytest.mark.parametrize("model", [None, "custom"], ids=["node-model", "own-model"])
def test_a_supervisor_whose_sdk_may_retry_is_refused_before_a_graph_runs(entry, model):
    provider, reviewer = Answering(), Answering(None)
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  parallel=entry == "parallel", supervisor=LLMSupervisor(reviewer, model=model),
                  max_revisions=1)
    node = Node(id="draft", label="Draft")

    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        enter(swarm, entry, ExecutionGraph([Topology.SERIAL], [node]))

    assert provider.calls == [] and reviewer.calls == [] and node.status is NodeStatus.PENDING


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", [None, "custom"], ids=["node-model", "own-model"])
@pytest.mark.parametrize("sdk_retries", [None, 2], ids=["undeclared", "two"])
def test_an_executor_with_a_policy_refuses_a_supervisor_whose_sdk_may_retry(mode, model,
                                                                            sdk_retries):
    provider, reviewer = Answering(), Answering(sdk_retries)
    executor, _, _ = make_executor(mode, provider, ManualClock(), RetryPolicy(),
                                   supervisor=LLMSupervisor(reviewer, model=model),
                                   max_revisions=1)
    node = Node(id="draft", label="Draft", metadata={"model": "custom"})

    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        run(executor, node)

    assert provider.calls == [] and reviewer.calls == [] and node.status is NodeStatus.PENDING


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sdk_retries,max_revisions,revisions_used,reviews", [
    pytest.param(0, 1, 0, ["supervision"], id="sends-each-call-once"),
    pytest.param(None, 0, 0, [], id="no-revisions"),
    pytest.param(None, 1, 1, [], id="revisions-spent"),
])
def test_an_executor_with_a_policy_runs_a_supervisor_that_sends_each_call_once_or_cannot_review(
    mode, sdk_retries, max_revisions, revisions_used, reviews,
):
    provider, reviewer = Answering(), Answering(sdk_retries)
    executor, _, _ = make_executor(mode, provider, ManualClock(), RetryPolicy(),
                                   supervisor=LLMSupervisor(reviewer), max_revisions=max_revisions,
                                   revisions_used=revisions_used)
    node = Node(id="draft", label="Draft", metadata={"model": "custom"})

    run(executor, node)

    assert node.status is NodeStatus.COMPLETED and provider.calls == ["execution"]
    assert reviewer.calls == reviews


def test_a_supervisor_that_can_make_no_review_is_not_refused():
    provider, reviewer = Answering(), Answering(None)
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  supervisor=LLMSupervisor(reviewer))

    assert swarm.execute(Task("Report")).output == "done"
    assert reviewer.calls == []


def failed_checkpoint(tmp_path, *, synthesizer=None):
    """Save the checkpoint of a run whose one node failed, or with *synthesizer*, whose merge did."""
    provider = Scripted(HttpError(403, "permission_error"))
    if synthesizer is not None:
        provider = Scripted("drafted", HttpError(403, "permission_error"))
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path), synthesizer=synthesizer)
    with pytest.raises(_STATUS_ERRORS[403]):
        swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")]))
    return FileCheckpointStore(tmp_path).list_ids()[0]


def test_resume_refuses_a_supervisor_whose_sdk_may_retry_before_any_call(tmp_path):
    execution_id = failed_checkpoint(tmp_path)
    provider, reviewer = Answering(), Answering(None)
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path),
                  supervisor=LLMSupervisor(reviewer), max_revisions=1)

    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        swarm.resume(execution_id)

    assert provider.calls == [] and reviewer.calls == []


def test_resume_refuses_a_merge_whose_sdk_may_retry_when_every_node_completed(tmp_path):
    merge = Synthesizer(SynthesisStrategy.LLM_MERGE)
    execution_id = failed_checkpoint(tmp_path, synthesizer=merge)
    provider = Answering(None)
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path), synthesizer=merge)

    # No node is left to run, so only the merge would call the provider.
    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        swarm.resume(execution_id)

    assert provider.calls == []


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------


def test_swarm_refuses_a_retry_policy_that_is_not_one():
    with pytest.raises(TypeError, match="retry_policy"):
        Swarm(model="offline", provider=OfflineProvider(), retry_policy={"deadline_s": 5})


@pytest.mark.parametrize("entry", ["execute", "execute-async", "parallel"])
def test_swarm_hands_its_policy_to_the_executor(entry):
    provider = Scripted(HttpError(403, "permission_error"), "done")
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  parallel=entry == "parallel")
    graph = ExecutionGraph([Topology.SERIAL], [retrying(max_retries=2)])

    with pytest.raises(_STATUS_ERRORS[403]):
        if entry == "execute-async":
            asyncio.run(swarm.execute_async(graph))
        else:
            swarm.execute(graph)

    assert len(provider.started) == 1


def test_resume_keeps_the_policy(tmp_path):
    provider = Scripted(HttpError(403, "permission_error"), HttpError(403, "permission_error"),
                        "done")
    swarm = Swarm(model="custom", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  checkpoint_store=FileCheckpointStore(tmp_path))
    graph = ExecutionGraph([Topology.SERIAL], [retrying(max_retries=2)])
    with pytest.raises(_STATUS_ERRORS[403]):
        swarm.execute(graph)
    execution_id = FileCheckpointStore(tmp_path).list_ids()[0]

    with pytest.raises(_STATUS_ERRORS[403]):
        swarm.resume(execution_id)

    # Without the policy the resumed node would retry the 403 and complete.
    assert len(provider.started) == 2


def test_durable_runs_record_the_policy_in_their_recipe(tmp_path):
    # tests/test_workflow_retry_policy.py covers the journal-derived allowance.
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                      run_store=store, max_budget_usd=1, retry_policy=RetryPolicy(deadline_s=30))
        result = swarm.execute(Task("Report"))
        assert result.output and result.cost_is_complete
        recorded = store.load_run(result.execution_id)["config"]["retry_policy"]
        assert recorded == {key: value for key, value in RetryPolicy(deadline_s=30).to_dict().items()
                            if key != "persist_attempts"}
