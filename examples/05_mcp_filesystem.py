"""MCP tool use: an agent reads real files through an MCP server.

    pip install smythe[mcp]
    python examples/05_mcp_filesystem.py

A read-only filesystem MCP server (mcp_file_server.py, shipped next to
this script — no npx needed) serves a scratch directory. The agent's
node runs the tool loop: list the files, read the interesting one,
answer from its contents.

Offline mode scripts the model's tool calls deterministically so you
can watch the loop work without keys. With an API key set, the real
model decides which tools to call.
"""

import sys
import tempfile
from pathlib import Path

from _providers import pick_provider

from smythe import MCPServerSpec, MCPToolRuntime, Swarm
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tools import ToolCall

# --- a scratch directory with something worth finding -----------------
data_dir = Path(tempfile.mkdtemp(prefix="smythe-mcp-demo-"))
(data_dir / "notes.txt").write_text("The launch code is TRINITY.\n", encoding="utf-8")
(data_dir / "readme.md").write_text("Nothing to see here.\n", encoding="utf-8")

SERVER = MCPServerSpec(
    name="fs",
    transport="stdio",
    command=sys.executable,
    args=(str(Path(__file__).with_name("mcp_file_server.py")), str(data_dir)),
    allowed_tools=("list_files", "read_file"),
)


class ScriptedToolCaller(Provider):
    """Offline stand-in that 'decides' to call the tools, then answers."""

    def __init__(self) -> None:
        self._step = 0

    async def complete(self, system, prompt, model):
        raise AssertionError("tool loop uses chat()")

    async def chat(self, system, messages, model, tools=None):
        self._step += 1
        if self._step == 1:
            return CompletionResult(
                text="", stop_reason="tool_use", prompt_tokens=40, completion_tokens=20,
                tool_calls=[ToolCall(id="c1", name="fs.list_files", arguments={})],
            )
        if self._step == 2:
            return CompletionResult(
                text="", stop_reason="tool_use", prompt_tokens=60, completion_tokens=20,
                tool_calls=[ToolCall(id="c2", name="fs.read_file",
                                     arguments={"name": "notes.txt"})],
            )
        last = messages[-1].tool_results[0].content if messages[-1].tool_results else ""
        return CompletionResult(
            text=f"Found it in notes.txt: {last.strip()}",
            prompt_tokens=80, completion_tokens=30,
        )


provider, model = pick_provider()
from smythe.provider import OfflineProvider  # noqa: E402

if isinstance(provider, OfflineProvider):
    # OfflineProvider echoes text; tool calling needs the scripted caller.
    provider = ScriptedToolCaller()

agent = Agent(profile=AgentProfile(
    name="FileDetective",
    persona="You answer questions by reading the available files.",
    mcp_servers=[SERVER],
))
registry = Registry()
registry.register(agent)

node = Node(
    id="find-code",
    label="What is the launch code? Check the files available to you.",
    agent_id=agent.id,
)
graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

swarm = Swarm(
    provider=provider, model=model, registry=registry,
    tool_runtime=MCPToolRuntime(), max_budget_usd=0.50,
)
result = swarm.execute(graph)

print("=== Answer ===")
print(result.output)
print("\n=== Tool calls (from the trace) ===")
for span in result.trace:
    for tc in span.get("tool_calls", []):
        flag = "ERROR" if tc["is_error"] else "ok"
        print(f"  {tc['tool']}  {tc['duration_ms']}ms  [{flag}]")
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
