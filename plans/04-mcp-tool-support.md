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

**Non-text results policy (v1):** MCP tools can return content lists containing text, images, and embedded resources. `ToolResult.content` is text-only: text parts are concatenated, and each non-text part becomes a placeholder marker (e.g. `[image: image/png, 42KB — not passed to the model]`). Conversion never raises — a screenshot-returning server must not crash a node. Rich content support is future work.

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
    budget.check(node.id)                   # mid-loop cap enforcement
    result = await provider.chat(system, messages, model, tools)
    budget.add_cost(node.id, result)        # accumulates; see below
    if not result.tool_calls:
        return result                       # normal completion
    messages.append(assistant turn with tool_calls)
    results = [await runtime.call(tc) for tc in result.tool_calls]  # traced
    messages.append(user turn with tool_results)
raise ToolLoopLimitError(node.id, max_tool_iterations)
```

- **Budget accounting moves into the loop.** Today the executors call `budget.record(node.id, result)` once, after `acall_node` returns — and `Sentinel.record()` *overwrites* `_node_costs[node_id]`, so calling it per iteration would corrupt the breakdown, while leaving the executor call in place would double-bill the final iteration. The design therefore: (a) adds `Sentinel.add_cost(node_id, result)` which **accumulates** into `_node_costs[node_id]` (first call also releases any outstanding reservation); (b) makes `acall_node` the sole owner of cost recording — the executors' post-call `budget.record()` is removed; (c) sets `node.metadata["cost_usd"]` to the node's **cumulative** cost across iterations. `record()` remains for backward compatibility but nothing in the execution path calls it.
- **Mid-loop cap enforcement.** `Sentinel.check(node.id)` runs before every iteration, so a runaway loop halts at the budget cap on the next iteration boundary instead of sailing past it for up to `max_tool_iterations` model calls. The async executor's per-wave reservation protocol is unchanged.
- **Timeout:** the existing `node.timeout_s` `wait_for` wraps the *entire* loop — a node with tools has one wall-clock budget, not one per iteration.
- **Failure policy:** a tool that errors returns `ToolResult(is_error=True)` to the model (which can adapt), not an exception; a *provider* error or loop-limit hit is a node failure handled by the existing HALT/SKIP/RETRY machinery. **Note that RETRY re-runs the whole loop** — tool calls that succeeded before the failure execute again. Tool side effects are at-least-once under RETRY exactly as they are under resume (§5.4); the two caveats are documented together.
- **Tracing:** `Tracer.on_tool_call(node, tool_name, duration_ms, is_error)` appends `{tool, duration_ms, is_error}` entries to the active span's `metadata["tool_calls"]`; `summary()` includes them. Arguments and results are **not** recorded in traces v1 (they may contain secrets; redaction hooks are backlog).
- **Sequential tool execution (v1).** Calls within one assistant turn run sequentially even though providers emit parallel calls expecting concurrency. Correctness is unaffected (all results still return in one turn, per the provider mapping rules); concurrent execution via `asyncio.gather` is a contained follow-up once per-server session thread-safety is verified.
- `max_tool_iterations` is configurable per node (`Node.max_tool_iterations`, YAML-loadable, serialized in checkpoints) with a Swarm-level default. Additive checkpoint fields like this one do **not** bump `CHECKPOINT_VERSION` — `node_from_dict` reads them with defaults, so older checkpoints stay loadable.

### 5.4 Layer 3 — MCP client (`smythe/mcp.py`, extra `smythe[mcp]`)

Built on the official `mcp` Python SDK. New optional dependency: `mcp>=1.0`.

```python
@dataclass(frozen=True)
class MCPServerSpec:
    name: str                                # namespace prefix
    transport: str                           # "stdio" | "http"
    command: str | None = None               # stdio
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None        # literal, NON-secret values only
    env_passthrough: tuple[str, ...] = ()    # var NAMES resolved from os.environ at session start
    url: str | None = None                   # streamable HTTP
    allowed_tools: tuple[str, ...] | None = None   # None = all (see size warning below)
    tool_timeout_s: float = 30.0
```

**Secrets never live in the spec.** `env` is for literal, non-secret values (paths, flags) and serializes as-is. Secrets go in `env_passthrough`: a list of variable *names* whose values are read from the running process's environment when the session opens. This keeps tokens out of YAML files (nothing to accidentally commit) and out of checkpoints — and it makes resume coherent: a resumed execution re-resolves the same names from the *resuming* process's environment, so tool servers work after resume without any secret ever being persisted. If a passthrough name is missing from the environment, session open fails with an error naming the variable.

**Large tool sets.** Every loop iteration sends every tool schema to the model; servers like GitHub's expose ~100 tools, which is a per-iteration token bomb that also degrades tool selection. `allowed_tools` is therefore effectively mandatory for large servers: the runtime logs a warning when a server contributes more than 20 tools with no allowlist, and the GitHub example ships with an explicit allowlist rather than presenting the unfiltered server as normal usage.

`ToolRuntime` owns sessions for one node execution: an `AsyncExitStack` opens each needed server at loop start (stdio subprocess or HTTP session), `list_tools()` → namespaced `ToolSpec`s filtered by `allowed_tools`, `call_tool()` executes with `tool_timeout_s`, and everything closes when the node finishes. **Teardown must be cancellation-safe:** `node.timeout_s` uses `asyncio.wait_for`, which cancels the loop mid-tool-call, and the exit stack must still terminate stdio subprocesses under that cancellation (shield the cleanup) — otherwise timed-out nodes leak server processes, a failure mode that is easiest to hit on Windows where the serial executor creates and destroys an event loop per node.

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
    - name: github
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env_passthrough: [GITHUB_PERSONAL_ACCESS_TOKEN]   # name only — value from os.environ
      allowed_tools: [search_repositories, get_file_contents, list_issues]
```

**Checkpoint interplay:** `mcp_servers` serialize with the agent in checkpoints — `command`, `args`, `url`, `allowed_tools`, literal `env`, and `env_passthrough` *names* are stored; secret *values* never are, because they only ever exist in the process environment (§ secrets above). A resumed execution reconstructs each spec verbatim and re-resolves passthrough names at session start. Resume re-runs incomplete nodes from scratch, so tool side effects are **at-least-once** — same caveat as RETRY (§5.3); idempotency is the tool/user's responsibility v1.

### 5.5 Capability-hydration synergy

`MCPSkillProvider` implements the existing `SkillProvider` protocol: `list_agent_skills(agent_id)` returns the agent's MCP tool names as `SkillRef`s. The registry's hydration machinery (caching, TTL, MERGE/REPLACE modes, capability mapper) then makes MCP tools drive **assignment** with zero new registry code — the same servers that power execution also ground `required_capabilities` matching. This closes a loop the OpenClaw PRD (plans/03) left open: capabilities backed by verifiable, executable tools.

### 5.6 Planner tool awareness

Hydration grounds *assignment*, and the loop grounds *execution* — but the `LLMArchitect` plans with no knowledge that either exists: `build_user_prompt(task, history)` carries no inventory of agents or tools, so the Architect cannot design plans *around* the toolset. Left unfixed, tool use only happens when assignment accidentally lands a tool-equipped agent on a node whose label happens to suit it — which undercuts the Phase 2 benchmark's central claim.

The fix is deliberately small: when the Swarm's registry contains agents with capabilities (static or hydrated), `build_user_prompt` gains an **available-agents section** — one line per agent: name, capability tags, and (for MCP-backed agents) a short tool summary such as `fs: read_file, list_directory`. The planning system prompt is extended to instruct the model to set `required_capabilities` on nodes that should land on those agents. No schema change to the plan JSON; `DeterministicArchitect` and `ConstrainedArchitect` are untouched (their authors already know their toolsets). Inventory text is capped (~40 agents / ~15 tools each, then elided with a count) so planning-prompt size stays bounded.

## 6) Testing

All CI tests run offline:

- **Provider mapping:** mock each vendor SDK response shape (tool_use blocks / tool_calls / function_call parts) → assert neutral `ToolCall`s; round-trip results back → assert wire format (esp. Anthropic single-user-message rule, OpenAI JSON-string args); non-text MCP content converts to placeholders without raising.
- **Tool loop:** `ScriptedToolProvider` emitting canned tool_calls then text — termination, multi-call turns, error results fed back, `ToolLoopLimitError` at the cap, `timeout_s` covering the whole loop, trace entries present.
- **Budget:** `add_cost` accumulates per node across iterations (breakdown shows the sum, not the last iteration); no double-billing after `acall_node` returns; mid-loop `check()` raises `SentinelAlert` on the iteration after the cap is crossed; async-executor reservation reconciliation with multi-iteration actuals.
- **MCP runtime:** in-process server over the `mcp` SDK's memory transport — discovery, namespacing, allowlist filtering, >20-tools-without-allowlist warning, tool timeout, `env_passthrough` resolution (and the missing-variable error), session teardown on node failure **and on `timeout_s` cancellation mid-tool-call** (no leaked subprocess).
- **Planner inventory:** with a populated registry, the planning prompt contains the agent/tool inventory; caps and elision applied; empty registry → no section (byte-identical prompt to today).
- **Checkpoint:** specs round-trip with `env_passthrough` names and without secret values; resumed run re-resolves and executes tools.
- **Integration** (env-var-gated, not CI): real filesystem server via `npx`, real API calls.
- **Fuzz:** malformed arguments JSON from OpenAI-shaped responses must never raise uncaught.

## 7) Milestones

1. **M1 — Provider tool contract.** `tools.py` types (incl. non-text placeholder conversion), `Provider.chat()`, all three providers mapped, `complete()` reimplemented on `chat()`. Tests green with mocks.
2. **M2 — Tool loop + budget rework.** `acall_node` loop, `max_tool_iterations`, `Sentinel.add_cost` accumulation with recording ownership moved out of the executors, mid-loop `check()`, trace/timeout integration, `ScriptedToolProvider` suite.
3. **M3 — MCP runtime.** `mcp.py` with `env_passthrough` and cancellation-safe teardown, `AgentProfile.mcp_servers`, YAML + checkpoint serialization, large-toolset warning, memory-transport tests, `smythe[mcp]` extra.
4. **M4 — Assignment & planning integration.** `MCPSkillProvider` hydration + the planner agent/tool inventory (§5.6), with tests for both.
5. **M5 — Examples & docs.** `examples/05_mcp_filesystem.py` (offline demo with an in-process toy MCP server + real-server instructions), GitHub example (with mandatory allowlist) and SaaS example (env-gated), README section, threat-model note. CHANGELOG. Roadmap tick.

Each milestone is independently landable and keeps the suite green; M1+M2 are useful alone (native function-calling with user-supplied Python tools could ride the same contract later).

## 8) Risks

| Risk | Mitigation |
|---|---|
| Serial executor spawns a stdio server per node (slow) | Accepted v1; doc recommends `parallel=True` for tool workloads; session-reuse fast-follow |
| Vendor SDK response shapes drift | Mapping isolated per provider class; mocks encode current shapes; env-gated integration tests catch drift |
| Windows stdio subprocess quirks (npx, console encoding, cancellation cleanup) | Filesystem example tested on Windows first (dev machine is Windows); shielded exit-stack teardown with a dedicated timeout-mid-tool-call test; memory transport keeps CI platform-independent |
| Tool results as prompt-injection vector | Documented threat model (§10 in docs); allowlists limit blast radius; approval gates (Phase 3) add the human check |
| Loop cost blowups | `max_tool_iterations` + mid-loop `Sentinel.check()` + `timeout_s`, all on by default paths |
| Secrets leaking via config or checkpoints | `env_passthrough` names-only design; secret values never serialized; literal `env` documented as non-secret |
| Large servers flood the context window | `allowed_tools` effectively mandatory for big servers; runtime warning above 20 unfiltered tools; GitHub example ships an allowlist |
| Plans don't exploit the toolset | Planner inventory (§5.6) makes tools visible at planning time; benchmark tasks assert tool-using topologies |

## 9) Acceptance (from roadmap)

- [ ] Agents can be configured with MCP servers as a capability source (Python + YAML)
- [ ] MCP tool calls show up in traces like any other node action
- [ ] Three working examples against real MCP servers (filesystem, GitHub, one SaaS)
- [ ] Everything above holds identically across Anthropic, OpenAI, and Gemini providers
