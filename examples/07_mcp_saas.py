"""MCP tool use against any SaaS MCP server over streamable HTTP (env-gated).

    pip install smythe[mcp]
    export SMYTHE_MCP_URL=https://mcp.linear.app/mcp     # or Notion, Asana, ...
    export SMYTHE_MCP_NAME=linear
    export SMYTHE_MCP_TOOLS=list_issues,get_issue        # allowlist (recommended)
    export SMYTHE_MCP_TASK="Summarize my open issues"
    python examples/07_mcp_saas.py

Works with any MCP server that speaks streamable HTTP. Authentication
varies by vendor — most hosted servers use OAuth; consult the server's
docs and, if it expects a bearer token via environment, add the
variable name to env_passthrough rather than putting the token in
config.
"""

import os
import sys

from _providers import pick_provider

from smythe import MCPServerSpec, MCPToolRuntime, Swarm
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import OfflineProvider
from smythe.registry import Registry

url = os.environ.get("SMYTHE_MCP_URL")
if not url:
    print("Set SMYTHE_MCP_URL (and optionally SMYTHE_MCP_NAME / SMYTHE_MCP_TOOLS /")
    print("SMYTHE_MCP_TASK) to run this example against a SaaS MCP server.")
    sys.exit(0)

provider, model = pick_provider()
if isinstance(provider, OfflineProvider):
    print("Set an LLM API key too — the model drives the tool calls here.")
    sys.exit(0)

name = os.environ.get("SMYTHE_MCP_NAME", "saas")
tools_env = os.environ.get("SMYTHE_MCP_TOOLS", "")
allowed = tuple(t.strip() for t in tools_env.split(",") if t.strip()) or None

SERVER = MCPServerSpec(
    name=name,
    transport="http",
    url=url,
    allowed_tools=allowed,
    tool_timeout_s=60.0,
)

agent = Agent(profile=AgentProfile(
    name="SaaSAssistant",
    persona="You accomplish tasks in external services via your tools.",
    mcp_servers=[SERVER],
))
registry = Registry()
registry.register(agent)

node = Node(
    id="task",
    label=os.environ.get("SMYTHE_MCP_TASK", "List what your tools can access."),
    agent_id=agent.id,
    timeout_s=180,
)
graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

swarm = Swarm(
    provider=provider, model=model, registry=registry,
    tool_runtime=MCPToolRuntime(), max_budget_usd=0.50,
)
result = swarm.execute(graph)

print("=== Result ===")
print(result.output)
for span in result.trace:
    for tc in span.get("tool_calls", []):
        print(f"  {tc['tool']}  {tc['duration_ms']}ms  error={tc['is_error']}")
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
