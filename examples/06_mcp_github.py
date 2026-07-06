"""MCP tool use against the real GitHub MCP server (env-gated).

    pip install smythe[mcp]
    export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...   # repo read scope is enough
    python examples/06_mcp_github.py

Requires an LLM API key (the model decides which tools to call) and
npx on PATH (the GitHub MCP server is an npm package).

Note the mandatory allowed_tools list: the GitHub server exposes on the
order of a hundred tools, and sending every schema on every loop
iteration is expensive and degrades tool selection. Allowlist the few
the task needs. The token travels via env_passthrough — its NAME is in
the config, its value never leaves your process environment.
"""

import os
import sys

from _providers import pick_provider

from smythe import MCPServerSpec, MCPToolRuntime, Swarm
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import OfflineProvider
from smythe.registry import Registry

if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
    print("Set GITHUB_PERSONAL_ACCESS_TOKEN to run this example.")
    sys.exit(0)

provider, model = pick_provider()
if isinstance(provider, OfflineProvider):
    print("Set an LLM API key too — the model drives the tool calls here.")
    sys.exit(0)

GITHUB = MCPServerSpec(
    name="gh",
    transport="stdio",
    command="npx",
    args=("-y", "@modelcontextprotocol/server-github"),
    env_passthrough=("GITHUB_PERSONAL_ACCESS_TOKEN",),   # name only — never the value
    allowed_tools=("search_repositories", "get_file_contents", "list_issues"),
    tool_timeout_s=60.0,
)

agent = Agent(profile=AgentProfile(
    name="RepoAnalyst",
    persona="You investigate GitHub repositories using your tools and report concisely.",
    mcp_servers=[GITHUB],
))
registry = Registry()
registry.register(agent)

node = Node(
    id="analyze",
    label=(
        "Look at the petehottelet/smythe repository on GitHub: summarize what "
        "it does from its README and list any open issues."
    ),
    agent_id=agent.id,
    timeout_s=180,
)
graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])

swarm = Swarm(
    provider=provider, model=model, registry=registry,
    tool_runtime=MCPToolRuntime(), max_budget_usd=0.50,
)
result = swarm.execute(graph)

print("=== Report ===")
print(result.output)
print("\n=== Tool calls ===")
for span in result.trace:
    for tc in span.get("tool_calls", []):
        print(f"  {tc['tool']}  {tc['duration_ms']}ms  error={tc['is_error']}")
print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
