"""Tests for the tool-calling loop in ExecutorBase.acall_node."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel, SentinelAlert
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tools import (
    ToolCall,
    ToolLoopLimitError,
    ToolResult,
    ToolRuntime,
    ToolSession,
    ToolSpec,
)
from smythe.tracer import Tracer

ADD_TOOL = ToolSpec(name="calc.add", description="Add numbers", input_schema={"type": "object"})


def tool_use(*calls: ToolCall, tokens: int = 1000) -> CompletionResult:
    return CompletionResult(
        text="", prompt_tokens=tokens // 2, completion_tokens=tokens - tokens // 2,
        tool_calls=list(calls), stop_reason="tool_use",
    )


def text(content: str = "final answer", tokens: int = 500) -> CompletionResult:
    return CompletionResult(
        text=content, prompt_tokens=tokens // 2, completion_tokens=tokens - tokens // 2,
    )


def call(cid: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=cid, name="calc.add", arguments=arguments)


class ScriptedToolProvider(Provider):
    """Plays back scripted chat results and records every chat invocation."""

    def __init__(self, script: list[CompletionResult]) -> None:
        self._script = list(script)
        self.chats: list[tuple[list, list | None]] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        raise AssertionError("the tool loop must use chat(), not complete()")

    async def chat(self, system, messages, model, tools=None):
        self.chats.append((list(messages), tools))
        if not self._script:
            return text("script exhausted")
        return self._script.pop(0)


class SimpleSession(ToolSession):
    def __init__(self, tools: list[ToolSpec], handler) -> None:
        self._tools = tools
        self._handler = handler

    @property
    def tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, tool_call: ToolCall) -> ToolResult:
        return self._handler(tool_call)


class SimpleRuntime(ToolRuntime):
    def __init__(self, tools: list[ToolSpec] | None = None, handler=None) -> None:
        self._tools = tools if tools is not None else [ADD_TOOL]
        self._handler = handler or (
            lambda tc: ToolResult(tool_call_id=tc.id, content="42")
        )
        self.opened = 0
        self.closed = 0

    def open(self, agent):
        runtime = self

        @asynccontextmanager
        async def _session():
            runtime.opened += 1
            try:
                yield SimpleSession(runtime._tools, runtime._handler)
            finally:
                runtime.closed += 1

        return _session()


def run_node(provider, runtime, node=None, budget=None, max_iter=10):
    node = node or Node(id="n1", label="Do the thing")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    tracer = Tracer()
    executor = Executor(
        provider=provider, registry=Registry(), tracer=tracer, budget=budget,
        tool_runtime=runtime, max_tool_iterations=max_iter,
    )
    executor.run(graph)
    return node, tracer


def test_no_runtime_is_a_single_plain_chat():
    provider = ScriptedToolProvider([text("done")])
    node, _ = run_node(provider, runtime=None)
    assert node.status == NodeStatus.COMPLETED
    assert node.result == "done"
    [(messages, tools)] = provider.chats
    assert tools is None
    assert len(messages) == 1


def test_tool_call_is_executed_and_result_fed_back():
    provider = ScriptedToolProvider([tool_use(call("c1", a=1, b=2)), text("3")])
    handler = lambda tc: ToolResult(tool_call_id=tc.id, content="3")  # noqa: E731
    node, _ = run_node(provider, SimpleRuntime(handler=handler))

    assert node.result == "3"
    assert len(provider.chats) == 2
    second_messages, tools = provider.chats[1]
    assert [t.name for t in tools] == ["calc.add"]
    assert second_messages[1].tool_calls[0].id == "c1"
    [tr] = second_messages[2].tool_results
    assert tr.tool_call_id == "c1"
    assert tr.content == "3"


def test_parallel_calls_return_in_one_result_message():
    provider = ScriptedToolProvider([
        tool_use(call("c1", a=1), call("c2", a=2)),
        text("both done"),
    ])
    node, _ = run_node(provider, SimpleRuntime())
    second_messages, _ = provider.chats[1]
    assert len(second_messages[2].tool_results) == 2


def test_tool_exception_becomes_error_result_not_node_failure():
    def exploding(tc):
        raise RuntimeError("tool blew up")

    provider = ScriptedToolProvider([tool_use(call()), text("recovered")])
    node, _ = run_node(provider, SimpleRuntime(handler=exploding))

    assert node.status == NodeStatus.COMPLETED
    [tr] = provider.chats[1][0][2].tool_results
    assert tr.is_error is True
    assert "tool blew up" in tr.content


def test_loop_limit_raises_and_session_closes():
    provider = ScriptedToolProvider([tool_use(call(f"c{i}")) for i in range(5)])
    runtime = SimpleRuntime()
    with pytest.raises(ToolLoopLimitError, match="max_tool_iterations=3"):
        run_node(provider, runtime, max_iter=3)
    assert runtime.closed == runtime.opened == 1


def test_per_node_iteration_override_beats_executor_default():
    provider = ScriptedToolProvider([tool_use(call(f"c{i}")) for i in range(5)])
    node = Node(id="n1", label="L", max_tool_iterations=1)
    with pytest.raises(ToolLoopLimitError, match="max_tool_iterations=1"):
        run_node(provider, SimpleRuntime(), node=node, max_iter=10)


def test_budget_accumulates_across_iterations_no_double_billing():
    cpt = 0.000003
    budget = Sentinel(max_budget_usd=1.0, cost_per_token=cpt)
    provider = ScriptedToolProvider([tool_use(call(), tokens=1000), text(tokens=500)])
    node, _ = run_node(provider, SimpleRuntime(), budget=budget)

    expected = 1500 * cpt
    assert budget.breakdown()["n1"] == pytest.approx(expected)
    assert budget.total_cost_usd == pytest.approx(expected)
    assert node.metadata["cost_usd"] == pytest.approx(expected)


def test_midloop_budget_cap_halts_runaway_loop():
    cpt = 0.000003
    budget = Sentinel(max_budget_usd=0.002, cost_per_token=cpt)
    provider = ScriptedToolProvider([tool_use(call(), tokens=1000) for _ in range(5)])
    with pytest.raises(SentinelAlert):
        run_node(provider, SimpleRuntime(), budget=budget)
    # exactly one iteration ran: 0.003 spent >= 0.002 cap blocked iteration two
    assert len(provider.chats) == 1


def test_timeout_covers_the_whole_loop():
    class SlowScripted(ScriptedToolProvider):
        async def chat(self, system, messages, model, tools=None):
            await asyncio.sleep(0.05)
            return await super().chat(system, messages, model, tools)

    provider = SlowScripted([tool_use(call()), tool_use(call("c2")), text()])
    node = Node(id="n1", label="L", timeout_s=0.08)
    with pytest.raises(TimeoutError, match="timed out"):
        run_node(provider, SimpleRuntime(), node=node)


def test_trace_records_tool_calls_with_error_flag():
    def flaky(tc):
        if tc.id == "bad":
            raise RuntimeError("nope")
        return ToolResult(tool_call_id=tc.id, content="ok")

    provider = ScriptedToolProvider([tool_use(call("good"), call("bad")), text()])
    _, tracer = run_node(provider, SimpleRuntime(handler=flaky))

    [span] = tracer.summary()
    calls = span["tool_calls"]
    assert [c["tool"] for c in calls] == ["calc.add", "calc.add"]
    assert [c["is_error"] for c in calls] == [False, True]


def test_pause_turn_resends_and_continues():
    paused = CompletionResult(text="partial", stop_reason="pause_turn")
    provider = ScriptedToolProvider([paused, text("done")])
    node, _ = run_node(provider, SimpleRuntime(tools=[]))
    assert node.result == "done"
    assert len(provider.chats) == 2


def test_empty_toolset_sends_no_tools():
    provider = ScriptedToolProvider([text("done")])
    node, _ = run_node(provider, SimpleRuntime(tools=[]))
    [(_, tools)] = provider.chats
    assert tools is None
    assert node.status == NodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_async_executor_runs_the_same_loop():
    provider = ScriptedToolProvider([tool_use(call()), text("async done")])
    runtime = SimpleRuntime()
    node = Node(id="n1", label="L")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=Tracer(),
        budget=Sentinel(max_budget_usd=1.0), tool_runtime=runtime,
    )
    await executor.run(graph)
    assert node.result == "async done"
    assert runtime.closed == 1
