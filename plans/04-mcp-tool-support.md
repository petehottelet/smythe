# PRD — MCP Tool Support

Give Smythe agents the ability to use tools via the Model Context Protocol, so nodes can do real work (read files, query APIs, act on external systems) instead of only transforming text. Phase 1, item 2 of the roadmap; prerequisite for a meaningful benchmark (Phase 2).

---

## 1) Problem Statement

Smythe agents are passive: `Provider.complete(system, prompt)` sends text and receives text. An agent cannot read a file, search the web, query GitHub, or call any external system. This caps what the framework can honestly claim — the benchmark suite can only test research-synthesis and writing tasks, templates can't do anything operational, and any evaluator comparing against LangGraph or CrewAI (both of which have tool ecosystems) will notice immediately.

MCP is the tool-integration substrate the ecosystem converged on. Supporting it means every existing MCP server (filesystem, GitHub, Slack, databases, hundreds more) becomes a Smythe capability source without us writing per-service adapters.

Two gaps must close, in order:

1. **The provider layer has no tool calling.** All three vendors (Anthropic, OpenAI, Gemini) ship native tool/function-calling; Smythe's abstraction doesn't expose it, and its single-turn `complete()` can't carry the multi-turn conversation a tool loop requires.
2. **There is no tool runtime.** Nothing discovers tools, hands their schemas to the model, executes the calls the model makes, or feeds results back.

## 2) Goals

1. **Agents consume MCP servers as capability sources.** Declare servers on an `AgentProfile` (Python and YAML); the agent's node executions can call those servers' tools.
2. **Provider-agnostic.** One neutral tool/message contract; each provider maps it to its native wire format. No provider-specific code above the provider layer.
3. **Tool calls are first-class in observability.** Every tool call appears in the trace (name, duration, error status) attached to its node's span, like any other node action.
4. **Deterministic guardrails hold.** Tool loops are bounded (`max_tool_iterations`), every loop iteration's tokens count against the Sentinel budget, `timeout_s` covers the whole loop, and per-server tool allowlists limit exposure.
5. **Backwards compatible.** Agents without MCP servers behave exactly as today. Custom `Provider` subclasses that only implement `complete()` keep working for tool-less execution.
6. **Three working examples** against real servers: filesystem, GitHub, and one SaaS — plus an offline demo so the machinery is visible without credentials.

## 3) Non-Goals

- **Anthropic's server-side MCP connector** (`mcp_servers` param + `mcp-client-2025-11-20` beta). It offloads the loop but is Anthropic-only and URL-transport-only. We run the loop client-side for provider-agnosticism; the connector is a possible later optimization for the Anthropic path.
- **Streaming** tool-call or token output (separate roadmap item under provider hardening).
- **Approval gates on tool calls** — that's the Phase 3 HITL work; this PRD lands the hook point (the loop) it will attach to.
- **Tool-result caching / memoization** (backlog: node-level caching).
- **Building MCP servers** — we are a client only.
- **Sandboxing tool execution** — MCP servers run wherever the user launches them; the threat-model doc (§10) says so plainly.

## 4) User Stories

1. As a developer, I declare `mcp_servers` on an agent in YAML and its nodes can read the repository the server exposes — no Python changes.
2. As an operator, I set `allowed_tools` so a research agent can call `search` and `fetch` but not `delete_file`, even though the server offers all three.
3. As a debugger, I open a trace and see every tool call a node made — which tool, how long, whether it errored — inside the node's span.
4. As a cost-conscious user, I see loop iterations reflected in the budget: a node that made 6 model calls while using tools shows the cumulative cost, and the Sentinel cap still halts it.
5. As a framework maintainer, I run the whole tool-loop test suite offline — no network, no real MCP servers, no API keys.

## 5) Design

### 5.1 Layer 1 — Neutral tool & message types (`smythe/tools.py`)

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                    # namespaced: "<server>.<tool>", e.g. "fs.read_file"
    description: str
    input_schema: dict           # JSON Schema (object type)

@dataclass(frozen=True)
class ToolCall:
    id: str                      # provider-assigned call id, echoed in the result
    name: str
    arguments: dict

@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str                 # text form of the result
    is_error: bool = False

@dataclass
class ChatMessage:
    role: str                    # "user" | "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # assistant turns
    tool_results: list[ToolResult] = field(default_factory=list)  # user turns
```

`CompletionResult` gains two fields (defaults preserve compatibility):

```python
@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"    # "end_turn" | "tool_use" | "max_tokens" | ...
```

### 5.2 Layer 1 — Provider contract change

New method on `Provider`, alongside the existing `complete()`:

```python
async def chat(
    self, system: str, messages: list[ChatMessage], model: str,
    tools: list[ToolSpec] | None = None,
) -> CompletionResult:
    """Multi-turn, tool-aware completion. Base impl (no tools, single user
    message) delegates to complete() so custom providers keep working."""
```

Per-provider mapping (all verified against current vendor APIs):

| Concern | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Tool declaration | `tools=[{name, description, input_schema}]` | `tools=[{type:"function", function:{name, description, parameters}}]` | `config.tools=[{function_declarations:[...]}]` |
| Model requests call | `tool_use` content blocks, `stop_reason=="tool_use"` | `message.tool_calls`, `finish_reason=="tool_calls"` | `function_call` parts in candidate content |
| Results sent back | `tool_result` blocks in **one** user message (splitting them degrades parallel calling) | one `role:"tool"` message per call, keyed by `tool_call_id` | `function_response` parts |
| Loop continuation quirk | `stop_reason=="pause_turn"` → re-send as-is and continue | — | — |
| Arguments | parsed dict | JSON string → `json.loads` (guard malformed) | parsed dict |

Tool names must satisfy every provider's naming rules (`^[a-zA-Z0-9_-]{1,64}$` is the strictest), so the namespace separator is `__` on the wire (`fs__read_file`) with `.` in user-facing config and traces; the runtime maps between them.

`complete()` stays as the simple path and is reimplemented as `chat(system, [ChatMessage(role="user", content=prompt)], model)` inside the three built-in providers — one code path, no behavior change.

### 5.3 Layer 2 — The tool loop (`ExecutorBase.acall_node`)

`acall_node` already centralizes the provider call for both executors (added in Phase 0). It becomes the loop:

```
tools = runtime.tools_for(agent)            # [] when agent has no MCP servers
messages = [ChatMessage(user, node prompt + dep results)]
for i in range(max_tool_iterations):        # default 10, circuit breaker
    result = await provider.chat(system, messages, model, tools)
    budget.record_iteration(node, result)   # every iteration billed
    if not result.tool_calls:
        return result                       # normal completion
    messages.append(assistant turn with tool_calls)
    results = [await runtime.call(tc) for tc in result.tool_calls]  # traced
    messages.append(user turn with tool_results)
raise ToolLoopLimitError(node.id, max_tool_iterations)
```

- **Timeout:** the existing `node.timeout_s` `wait_for` wraps the *entire* loop — a node with tools has one wall-clock budget, not one per iteration.
- **Budget:** each iteration's tokens go through `Sentinel.record()`; the async executor's per-wave reservation is unchanged (the estimate is just likelier to be exceeded — the reconciliation already handles that).
- **Failure policy:** a tool that errors returns `ToolResult(is_error=True)` to the model (which can adapt), not an exception; a *provider* error or loop-limit hit is a node failure handled by the existing HALT/SKIP/RETRY machinery.
- **Tracing:** `Tracer.on_tool_call(node, tool_name, duration_ms, is_error)` appends `{tool, duration_ms, is_error}` entries to the active span's `metadata["tool_calls"]`; `summary()` includes them. Arguments and results are **not** recorded in traces v1 (they may contain secrets; redaction hooks are backlog).
- `max_tool_iterations` is configurable per node (`Node.max_tool_iterations`, YAML-loadable) with a Swarm-level default.

### 5.4 Layer 3 — MCP client (`smythe/mcp.py`, extra `smythe[mcp]`)

Built on the official `mcp` Python SDK. New optional dependency: `mcp>=1.0`.

```python
@dataclass(frozen=True)
class MCPServerSpec:
    name: str                                # namespace prefix
    transport: str                           # "stdio" | "http"
    command: str | None = None               # stdio
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str | None = None                   # streamable HTTP
    allowed_tools: tuple[str, ...] | None = None   # None = all
    tool_timeout_s: float = 30.0
```

`ToolRuntime` owns sessions for one node execution: an `AsyncExitStack` opens each needed server at loop start (stdio subprocess or HTTP session), `list_tools()` → namespaced `ToolSpec`s filtered by `allowed_tools`, `call_tool()` executes with `tool_timeout_s`, and everything closes when the node finishes.

**Session scope decision:** per-node-execution. The serial executor calls `asyncio.run()` per node, so any loop-spanning session would die with its event loop; per-node scope is the only shape that works identically in both executors. Cost: a stdio server process spawn per tool-using node. Accepted for v1; session reuse across nodes within one `execute_async()` run is a contained fast-follow (sessions keyed on the running loop, owned by the executor).

**Config surface:** `AgentProfile.mcp_servers: list[MCPServerSpec]` (default empty), YAML:

```yaml
agent:
  name: Researcher
  mcp_servers:
    - name: fs
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
      allowed_tools: [read_file, list_directory]
```

**Checkpoint interplay:** `mcp_servers` serialize with the agent in checkpoints (env values redacted to `"***"`; resume re-reads real values from the resuming process's environment). Resume re-runs incomplete nodes from scratch, so tool side effects are **at-least-once** — documented; idempotency is the tool/user's responsibility v1.

### 5.5 Capability-hydration synergy

`MCPSkillProvider` implements the existing `SkillProvider` protocol: `list_agent_skills(agent_id)` returns the agent's MCP tool names as `SkillRef`s. The registry's hydration machinery (caching, TTL, MERGE/REPLACE modes, capability mapper) then makes MCP tools drive **assignment** with zero new registry code — the same servers that power execution also ground `required_capabilities` matching. This closes a loop the OpenClaw PRD (plans/03) left open: capabilities backed by verifiable, executable tools.

## 6) Testing

All CI tests run offline:

- **Provider mapping:** mock each vendor SDK response shape (tool_use blocks / tool_calls / function_call parts) → assert neutral `ToolCall`s; round-trip results back → assert wire format (esp. Anthropic single-user-message rule, OpenAI JSON-string args).
- **Tool loop:** `ScriptedToolProvider` emitting canned tool_calls then text — termination, multi-call turns, error results fed back, `ToolLoopLimitError` at the cap, budget accumulation across iterations, `timeout_s` covering the whole loop, trace entries present.
- **MCP runtime:** in-process server over the `mcp` SDK's memory transport — discovery, namespacing, allowlist filtering, tool timeout, session teardown on node failure.
- **Integration** (env-var-gated, not CI): real filesystem server via `npx`, real API calls.
- **Fuzz:** malformed arguments JSON from OpenAI-shaped responses must never raise uncaught.

## 7) Milestones

1. **M1 — Provider tool contract.** `tools.py` types, `Provider.chat()`, all three providers mapped, `complete()` reimplemented on `chat()`. Tests green with mocks.
2. **M2 — Tool loop.** `acall_node` loop, `max_tool_iterations`, budget/trace/timeout integration, `ScriptedToolProvider` suite.
3. **M3 — MCP runtime.** `mcp.py`, `AgentProfile.mcp_servers`, YAML + checkpoint serialization, memory-transport tests, `smythe[mcp]` extra.
4. **M4 — Hydration.** `MCPSkillProvider` + tests.
5. **M5 — Examples & docs.** `examples/05_mcp_filesystem.py` (offline demo with an in-process toy MCP server + real-server instructions), GitHub and SaaS examples (env-gated), README section, threat-model note. CHANGELOG. Roadmap tick.

Each milestone is independently landable and keeps the suite green; M1+M2 are useful alone (native function-calling with user-supplied Python tools could ride the same contract later).

## 8) Risks

| Risk | Mitigation |
|---|---|
| Serial executor spawns a stdio server per node (slow) | Accepted v1; doc recommends `parallel=True` for tool workloads; session-reuse fast-follow |
| Vendor SDK response shapes drift | Mapping isolated per provider class; mocks encode current shapes; env-gated integration tests catch drift |
| Windows stdio subprocess quirks (npx, console encoding) | Filesystem example tested on Windows first (dev machine is Windows); memory transport keeps CI platform-independent |
| Tool results as prompt-injection vector | Documented threat model (§10 in docs); allowlists limit blast radius; approval gates (Phase 3) add the human check |
| Loop cost blowups | `max_tool_iterations` + Sentinel cap + `timeout_s`, all on by default paths |

## 9) Acceptance (from roadmap)

- [ ] Agents can be configured with MCP servers as a capability source (Python + YAML)
- [ ] MCP tool calls show up in traces like any other node action
- [ ] Three working examples against real MCP servers (filesystem, GitHub, one SaaS)
- [ ] Everything above holds identically across Anthropic, OpenAI, and Gemini providers
