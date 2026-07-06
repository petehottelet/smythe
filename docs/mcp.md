# MCP tool support

Agents consume [MCP](https://modelcontextprotocol.io/) servers as tool sources: declare servers on an `AgentProfile`, pass a `MCPToolRuntime` to the Swarm, and nodes assigned to that agent run a bounded tool loop — the model calls tools, results feed back, every call lands in the trace, and the budget meters every iteration.

```python
from smythe import MCPServerSpec, MCPToolRuntime, Swarm
from smythe.agent import Agent, AgentProfile

fs = MCPServerSpec(
    name="fs",                      # namespace: tools appear as fs.read_file
    transport="stdio",              # or "http" with url=...
    command="npx",
    args=("-y", "@modelcontextprotocol/server-filesystem", "./data"),
    allowed_tools=("read_file", "list_directory"),
)
agent = Agent(profile=AgentProfile(name="Researcher", mcp_servers=[fs]))
swarm = Swarm(tool_runtime=MCPToolRuntime(), ...)
```

The same shape works in YAML under `agent.mcp_servers`. Install the extra: `pip install smythe[mcp]`.

## Secrets: env_passthrough

Never put tokens in `env` — it holds literal, non-secret values (paths, flags) and is serialized as-is into YAML and checkpoints. Secrets go in `env_passthrough` as variable **names**:

```python
MCPServerSpec(
    name="gh", transport="stdio",
    command="npx", args=("-y", "@modelcontextprotocol/server-github"),
    env_passthrough=("GITHUB_PERSONAL_ACCESS_TOKEN",),
)
```

The value is read from the process environment when the session opens; a missing variable fails immediately with its name. Checkpoints store only the name, so resumed executions re-resolve from the resuming process's environment — no secret ever touches disk.

## Allowlists

`allowed_tools` is effectively mandatory for large servers. Every tool-loop iteration sends every tool schema to the model; the GitHub server exposes ~100 tools, which is a per-iteration token bomb that also degrades tool selection. The runtime warns when a server contributes more than 20 tools with no allowlist.

## Guardrails

- `max_tool_iterations` (Swarm default 10, per-node override) bounds the loop.
- The budget cap is checked between iterations — a runaway loop halts at the cap.
- `Node.timeout_s` covers the *whole* loop, and session teardown survives the cancellation (stdio subprocesses are reaped before the node fails).
- `tool_timeout_s` bounds each individual tool call; a slow or failing tool returns to the model as an error result it can adapt to — it does not fail the node.

## Sessions and performance

Sessions are opened per node execution and closed when the node finishes. The serial executor therefore spawns stdio servers once per tool-using node — prefer `Swarm(parallel=True)` for tool-heavy workloads. Session reuse across nodes within one run is planned.

## Side effects are at-least-once

Two paths re-execute tool calls that already ran: `failure_policy: RETRY` re-runs a node's whole loop after a provider error, and `swarm.resume()` re-runs incomplete nodes from scratch. If a tool has side effects (posting, writing, sending), idempotency is the tool's or your responsibility.

## Planner awareness and assignment

- `Swarm` passes its registry to the default `LLMArchitect`, so the planning prompt includes an inventory of existing agents and their tools, and instructs the model to route nodes to them via `required_capabilities`.
- `MCPSkillProvider` plugs into the registry's capability hydration, so the same tools ground assignment: allowlisted tools contribute capabilities statically; `await provider.prefetch(agents)` discovers tools for servers without allowlists.

## Threat model

Treat tool results as **untrusted input**. A malicious or compromised MCP server — or hostile content a benign tool fetches (a web page, an issue comment) — can inject instructions that the model may follow, including instructions to call other tools. Mitigations available today:

- **Allowlists** limit the blast radius: an agent that can only `read_file` cannot be prompted into deleting one.
- **Server choice is trust**: running `npx some-server` executes third-party code with your process's privileges. Pin versions, prefer official servers, and give stdio servers narrow working directories.
- **Budget caps, iteration limits, and timeouts** bound what a hijacked loop can spend or do.
- Tool call **arguments and results are not persisted in traces** (only tool name, duration, error flag), so secrets that pass through tools don't land in trace files.

Not yet provided (roadmap): human approval gates on tool calls, sandboxed tool execution, and trace redaction hooks. If a task processes adversarial content with side-effect-capable tools, don't rely on the model to resist injection — restrict the toolset instead.
