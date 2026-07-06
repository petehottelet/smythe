"""Tests for the MCP client runtime — specs, discovery, calls, teardown."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from smythe.checkpoint import agents_from_list, agents_to_list  # noqa: E402
from smythe.agent import Agent, AgentProfile  # noqa: E402
from smythe.loader import load_graph_from_string  # noqa: E402
from smythe.mcp import (  # noqa: E402
    MCPConfigError,
    MCPServerSpec,
    MCPToolRuntime,
    MCPToolSession,
    discover_tools,
)
from smythe.registry import Registry  # noqa: E402
from smythe.tools import ToolCall  # noqa: E402

SERVER_SCRIPT = str(Path(__file__).with_name("mcp_test_server.py"))


def stdio_spec(**overrides) -> MCPServerSpec:
    kwargs = dict(
        name="srv",
        transport="stdio",
        command=sys.executable,
        args=(SERVER_SCRIPT,),
    )
    kwargs.update(overrides)
    return MCPServerSpec(**kwargs)


# ---------------------------------------------------------------------------
# Spec validation and serialization
# ---------------------------------------------------------------------------


def test_stdio_requires_command():
    with pytest.raises(MCPConfigError, match="requires 'command'"):
        MCPServerSpec(name="s", transport="stdio")


def test_http_requires_url():
    with pytest.raises(MCPConfigError, match="requires 'url'"):
        MCPServerSpec(name="s", transport="http")


def test_unknown_transport_rejected():
    with pytest.raises(MCPConfigError, match="transport"):
        MCPServerSpec(name="s", transport="websocket", url="ws://x")


def test_server_name_with_dot_rejected():
    with pytest.raises(MCPConfigError, match="server name"):
        MCPServerSpec(name="bad.name", transport="http", url="http://x")


def test_resolve_env_merges_literals_and_passthrough(monkeypatch):
    monkeypatch.setenv("SMYTHE_TEST_TOKEN", "sekrit")
    spec = stdio_spec(env={"MODE": "test"}, env_passthrough=("SMYTHE_TEST_TOKEN",))
    resolved = spec.resolve_env()
    assert resolved == {"MODE": "test", "SMYTHE_TEST_TOKEN": "sekrit"}


def test_resolve_env_missing_passthrough_names_the_variable(monkeypatch):
    monkeypatch.delenv("SMYTHE_MISSING_TOKEN", raising=False)
    spec = stdio_spec(env_passthrough=("SMYTHE_MISSING_TOKEN",))
    with pytest.raises(MCPConfigError, match="SMYTHE_MISSING_TOKEN"):
        spec.resolve_env()


def test_spec_dict_roundtrip_preserves_everything():
    spec = stdio_spec(
        env={"MODE": "x"},
        env_passthrough=("TOKEN_NAME",),
        allowed_tools=("add",),
        tool_timeout_s=5.0,
    )
    restored = MCPServerSpec.from_dict(spec.to_dict())
    assert restored == spec
    # names only — the dict must never contain a resolved secret value
    assert "TOKEN_NAME" in spec.to_dict()["env_passthrough"]


def test_checkpoint_agents_roundtrip_with_mcp_servers():
    registry = Registry()
    agent = Agent(profile=AgentProfile(
        name="tooluser",
        capabilities=["research"],
        mcp_servers=[stdio_spec(allowed_tools=("add",))],
    ))
    registry.register(agent)

    [restored] = agents_from_list(agents_to_list(registry))
    assert restored.profile.mcp_servers == agent.profile.mcp_servers


def test_loader_parses_agent_mcp_servers():
    yaml_str = f"""\
topology: serial

nodes:
  - id: n1
    label: "Use tools"
    agent:
      name: ToolUser
      mcp_servers:
        - name: srv
          transport: stdio
          command: "{sys.executable.replace(chr(92), '/')}"
          args: ["server.py"]
          allowed_tools: [add]
          env_passthrough: [MY_TOKEN]
"""
    graph, registry = load_graph_from_string(yaml_str)
    [agent] = registry.list_agents()
    [spec] = agent.profile.mcp_servers
    assert spec.name == "srv"
    assert spec.allowed_tools == ("add",)
    assert spec.env_passthrough == ("MY_TOKEN",)


def test_loader_rejects_invalid_mcp_entry():
    yaml_str = """\
topology: serial

nodes:
  - id: n1
    label: "Bad"
    agent:
      name: X
      mcp_servers:
        - name: srv
          transport: stdio
"""
    with pytest.raises(ValueError, match="Invalid mcp_servers entry"):
        load_graph_from_string(yaml_str)


# ---------------------------------------------------------------------------
# Discovery and calls over the SDK's in-memory transport
# ---------------------------------------------------------------------------


def _memory_server() -> FastMCP:
    srv = FastMCP("mem")

    @srv.tool()
    def add(a: int, b: int) -> int:
        return a + b

    @srv.tool()
    def shout(text: str) -> str:
        return text.upper()

    @srv.tool()
    def explode() -> str:
        raise RuntimeError("kaboom")

    return srv


@pytest.mark.asyncio
async def test_discovery_namespaces_and_filters():
    spec = MCPServerSpec(
        name="mem", transport="http", url="http://unused", allowed_tools=("add", "shout"),
    )
    async with create_connected_server_and_client_session(_memory_server()) as cs:
        listing = await cs.list_tools()
        tool_specs, dispatch = discover_tools(spec, cs, listing)

    assert sorted(t.name for t in tool_specs) == ["mem.add", "mem.shout"]
    assert "mem.explode" not in dispatch
    assert dispatch["mem.add"].remote_name == "add"


@pytest.mark.asyncio
async def test_call_success_error_and_unknown():
    spec = MCPServerSpec(name="mem", transport="http", url="http://unused")
    async with create_connected_server_and_client_session(_memory_server()) as cs:
        listing = await cs.list_tools()
        tool_specs, dispatch = discover_tools(spec, cs, listing)
        session = MCPToolSession(tool_specs, dispatch)

        ok = await session.call(ToolCall(id="c1", name="mem.add", arguments={"a": 2, "b": 3}))
        assert ok.is_error is False
        assert "5" in ok.content

        err = await session.call(ToolCall(id="c2", name="mem.explode", arguments={}))
        assert err.is_error is True

        unknown = await session.call(ToolCall(id="c3", name="mem.nope", arguments={}))
        assert unknown.is_error is True
        assert "Unknown tool" in unknown.content


@pytest.mark.asyncio
async def test_large_toolset_warns_without_allowlist(caplog):
    srv = FastMCP("big")
    for i in range(21):
        def make(i=i):
            def tool_fn() -> int:
                return i
            tool_fn.__name__ = f"tool_{i}"
            return tool_fn
        srv.tool()(make())

    spec = MCPServerSpec(name="big", transport="http", url="http://unused")
    async with create_connected_server_and_client_session(srv) as cs:
        listing = await cs.list_tools()
        with caplog.at_level("WARNING", logger="smythe.mcp"):
            discover_tools(spec, cs, listing)

    assert any("allowed_tools" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: real stdio subprocess through the executor tool loop
# ---------------------------------------------------------------------------

from test_tool_loop import ScriptedToolProvider, text, tool_use  # noqa: E402
from smythe.executor import Executor  # noqa: E402
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology  # noqa: E402
from smythe.tools import ToolCall as TC  # noqa: E402
from smythe.tracer import Tracer  # noqa: E402


def _run_with_stdio_server(provider, spec, node=None):
    profile = AgentProfile(name="tooluser", mcp_servers=[spec])
    agent = Agent(profile=profile)
    registry = Registry()
    registry.register(agent)

    node = node or Node(id="n1", label="Use the tools")
    node.agent_id = agent.id
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor = Executor(
        provider=provider, registry=registry, tracer=Tracer(),
        tool_runtime=MCPToolRuntime(),
    )
    executor.run(graph)
    return node


def test_stdio_end_to_end_add():
    provider = ScriptedToolProvider([
        tool_use(TC(id="c1", name="srv.add", arguments={"a": 2, "b": 3})),
        text("the sum is 5"),
    ])
    node = _run_with_stdio_server(provider, stdio_spec(allowed_tools=("add",)))

    assert node.status == NodeStatus.COMPLETED
    second_messages, tools = provider.chats[1]
    assert [t.name for t in tools] == ["srv.add"]
    [tr] = second_messages[2].tool_results
    assert tr.is_error is False
    assert "5" in tr.content


def test_stdio_env_passthrough_reaches_server(monkeypatch):
    monkeypatch.setenv("SMYTHE_E2E_TOKEN", "hunter2")
    provider = ScriptedToolProvider([
        tool_use(TC(id="c1", name="srv.get_env", arguments={"name": "SMYTHE_E2E_TOKEN"})),
        text("done"),
    ])
    node = _run_with_stdio_server(
        provider,
        stdio_spec(allowed_tools=("get_env",), env_passthrough=("SMYTHE_E2E_TOKEN",)),
    )
    assert node.status == NodeStatus.COMPLETED
    [tr] = provider.chats[1][0][2].tool_results
    assert "hunter2" in tr.content


def test_stdio_timeout_mid_tool_call_tears_down_cleanly():
    """timeout_s cancels the loop mid-call; teardown must not hang or leak."""
    provider = ScriptedToolProvider([
        tool_use(TC(id="c1", name="srv.slow", arguments={"seconds": 30})),
        text("never reached"),
    ])
    node = Node(id="n1", label="L", timeout_s=2.0)
    with pytest.raises(TimeoutError, match="timed out"):
        _run_with_stdio_server(
            provider, stdio_spec(allowed_tools=("slow",), tool_timeout_s=60.0), node=node,
        )
    assert node.status == NodeStatus.FAILED
