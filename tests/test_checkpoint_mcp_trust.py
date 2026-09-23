"""Resume must not start MCP servers that a pre-0.8.1 model plan could declare."""

import logging
from contextlib import asynccontextmanager

from smythe.agent import Agent, AgentProfile
from smythe.checkpoint import CHECKPOINT_VERSION, FileCheckpointStore, build_state
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.mcp import MCPServerSpec
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.tools import ToolRuntime, ToolSession


class EchoProvider(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text="summary", prompt_tokens=1, completion_tokens=1)


class RecordingRuntime(ToolRuntime):
    """Records the MCP servers of every agent it is asked to open."""

    def __init__(self):
        self.opened: list[list[MCPServerSpec]] = []

    def open(self, agent):
        runtime = self

        @asynccontextmanager
        async def session():
            runtime.opened.append(list(agent.profile.mcp_servers) if agent else [])
            yield _NoTools()

        return session()


class _NoTools(ToolSession):
    @property
    def tools(self):
        return []

    async def call(self, tool_call):
        raise AssertionError("no tool is offered")


def _save_checkpoint(store, *, version):
    """Persist a halted run whose pending node belongs to an MCP-equipped agent."""
    server = MCPServerSpec(
        name="srv", transport="stdio", command="python",
        args=("-c", "open('launched', 'w')"), env_passthrough=("SECRET_TOKEN",),
    )
    agent = Agent(id="summarizer", profile=AgentProfile(name="Summarizer", mcp_servers=[server]))
    registry = Registry()
    registry.register(agent)
    node = Node(id="n1", label="Summarize", agent_id=agent.id, status=NodeStatus.FAILED)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    state = build_state(
        execution_id="halted", status="failed", model="test-model", graph=graph,
        registry=registry, task=None, max_budget_usd=None, node_costs={},
    )
    state["version"] = version
    store.save("halted", state)
    return server


def test_pre_0_8_1_checkpoint_resumes_without_its_mcp_servers(tmp_path, caplog):
    store = FileCheckpointStore(tmp_path)
    _save_checkpoint(store, version=3)
    runtime = RecordingRuntime()
    swarm = Swarm(
        provider=EchoProvider(), model="test-model", checkpoint_store=store,
        tool_runtime=runtime, artifact_dir=None,
    )

    with caplog.at_level(logging.WARNING, logger="smythe.swarm"):
        result = swarm.resume("halted")

    assert result.output == "summary"
    assert runtime.opened == [[]], "a server from a pre-0.8.1 checkpoint reached the runtime"
    assert "summarizer/srv" in caplog.text
    saved = store.load("halted")
    assert saved["version"] == CHECKPOINT_VERSION == 4
    assert all("mcp_servers" not in entry for entry in saved["agents"])


def test_version_4_checkpoint_keeps_developer_mcp_servers(tmp_path):
    store = FileCheckpointStore(tmp_path)
    server = _save_checkpoint(store, version=4)
    runtime = RecordingRuntime()
    swarm = Swarm(
        provider=EchoProvider(), model="test-model", checkpoint_store=store,
        tool_runtime=runtime, artifact_dir=None,
    )

    swarm.resume("halted")

    assert runtime.opened == [[server]]
