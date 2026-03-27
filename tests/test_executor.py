"""Tests for the serial Executor with failure policy support."""

import pytest

from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class FailingProvider(Provider):
    """Provider that fails a configurable number of times before succeeding.

    If ``fail_labels`` is set, only prompts whose first line matches
    one of the labels will be failed; all others succeed immediately.
    """

    def __init__(self, failures: int = 1, fail_labels: set[str] | None = None) -> None:
        self._failures = failures
        self._fail_labels = fail_labels
        self._attempts: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        if self._fail_labels is not None and label not in self._fail_labels:
            return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)
        self._attempts.setdefault(label, 0)
        self._attempts[label] += 1
        if self._attempts[label] <= self._failures:
            raise RuntimeError(f"Simulated failure #{self._attempts[label]}")
        return CompletionResult(text=f"ok: {label}", prompt_tokens=5, completion_tokens=5)


class SuccessProvider(Provider):
    async def complete(self, system, prompt, model):
        label = prompt.split("\n")[0]
        return CompletionResult(text=f"done: {label}", prompt_tokens=5, completion_tokens=5)


def _make_executor(provider: Provider | None = None) -> tuple[Executor, Tracer]:
    tracer = Tracer()
    registry = Registry()
    p = provider or SuccessProvider()
    return Executor(provider=p, registry=registry, tracer=tracer), tracer


def test_halt_raises_on_failure():
    executor, _ = _make_executor(FailingProvider(failures=999))
    node = Node(label="Fail", id="f", failure_policy=FailurePolicy.HALT)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        executor.run(graph)
    assert node.status == NodeStatus.FAILED


def test_skip_marks_skipped_and_continues():
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Flaky"}))
    a = Node(label="Flaky", id="a", failure_policy=FailurePolicy.SKIP)
    b = Node(label="Next", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    executor.run(graph)

    assert a.status == NodeStatus.SKIPPED
    assert b.status == NodeStatus.COMPLETED


def test_retry_succeeds_after_transient_failure():
    executor, _ = _make_executor(FailingProvider(failures=1))
    node = Node(
        label="Retryable", id="r",
        failure_policy=FailurePolicy.RETRY, max_retries=3,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert "ok:" in node.result


def test_retry_exhausted_raises():
    executor, _ = _make_executor(FailingProvider(failures=999))
    node = Node(
        label="HardFail", id="hf",
        failure_policy=FailurePolicy.RETRY, max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

    with pytest.raises(RuntimeError, match="Simulated failure"):
        executor.run(graph)
    assert node.status == NodeStatus.FAILED


def test_retry_count_means_additional_retries():
    """max_retries is interpreted as retries beyond the initial attempt."""
    executor, _ = _make_executor(FailingProvider(failures=2))
    node = Node(
        label="EventuallyOk", id="eo",
        failure_policy=FailurePolicy.RETRY, max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor.run(graph)
    assert node.status == NodeStatus.COMPLETED


def test_serial_halts_downstream_of_failed():
    """When an upstream node fails with HALT, its dependent should not execute."""
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Upstream"}))
    a = Node(label="Upstream", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Downstream", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    with pytest.raises(RuntimeError):
        executor.run(graph)

    assert a.status == NodeStatus.FAILED
    assert b.status in (NodeStatus.PENDING, NodeStatus.FAILED)
    assert b.result is None or "Upstream" in str(b.result)


def test_serial_skips_downstream_of_failed():
    """Downstream node with SKIP policy should be SKIPPED when upstream fails."""
    executor, _ = _make_executor(FailingProvider(failures=999, fail_labels={"Upstream"}))
    a = Node(label="Upstream", id="a", failure_policy=FailurePolicy.HALT)
    b = Node(label="Downstream", id="b", depends_on=["a"], failure_policy=FailurePolicy.SKIP)
    c = Node(label="Final", id="c", depends_on=["b"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b, c])

    with pytest.raises(RuntimeError):
        executor.run(graph)

    assert a.status == NodeStatus.FAILED
    assert b.status == NodeStatus.SKIPPED


def test_default_policy_is_halt():
    node = Node(label="Default", id="d")
    assert node.failure_policy == FailurePolicy.HALT
