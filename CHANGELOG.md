# Changelog

All notable changes to **smythe** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 stability note below.

## Versioning policy (pre-1.0)

While the project is on a `0.x` line, the public API is **not yet stable**:

- `0.x` minor bumps (e.g. `0.1.0` -> `0.2.0`) MAY include backward-incompatible changes.
  Each minor release will document its breaking changes in this file.
- `0.x.y` patch bumps (e.g. `0.1.0` -> `0.1.1`) are **non-breaking** and contain only
  bug fixes, documentation, or internal improvements.
- The first stable release will be `1.0.0`. Until then, pin to a specific minor
  version (`smythe~=0.1.0`) if API stability matters to you.

---

## [Unreleased]

Future work tracked in [ROADMAP.md](ROADMAP.md).

---

## [0.2.0] - 2026-07-05

First PyPI release. The v0.2 line makes agents real: they use tools, survive
crashes, and everything they do is visible.

### Added

- **Provider tool contract** — neutral tool-calling types (`ToolSpec`, `ToolCall`,
  `ToolResult`, `ChatMessage`) and `Provider.chat()`, mapped to native tool use on
  Anthropic, OpenAI, and Gemini. First milestone of MCP tool support
  (plans/04-mcp-tool-support.md).
- **MCP tool support** — agents consume MCP servers as tool sources
  (`pip install smythe[mcp]`): `MCPServerSpec` (stdio + streamable HTTP) with
  `env_passthrough` secret handling (variable names in config, values resolved
  from the environment — never serialized), per-server `allowed_tools` with a
  large-toolset warning, per-call timeouts, and cancellation-safe teardown.
  `MCPSkillProvider` grounds capability-based assignment in real tools, and the
  `LLMArchitect` planning prompt now includes an available-agents/tools
  inventory so plans exploit the toolset. Examples: offline filesystem
  (bundled server), GitHub (allowlisted), and generic SaaS over HTTP.
  Docs and threat model: docs/mcp.md.
- **Tool-calling loop** — nodes whose Swarm has a `tool_runtime` run a bounded
  tool loop: `max_tool_iterations` circuit breaker (per node and per Swarm),
  mid-loop budget enforcement, per-call trace entries, tool failures fed back to
  the model as error results, and `timeout_s` covering the whole loop. Budget
  recording moved into the loop via the new accumulating `Sentinel.add_cost()`;
  `node.metadata["cost_usd"]` is now cumulative across a node's provider calls.
  `ToolRuntime` / `ToolSession` define the interface the MCP runtime implements
  next (plans/04 M2).
- **Graph export** — `ExecutionGraph.to_mermaid()` (with node-status styling),
  `to_dot()`, and `to_json()` (with per-node cost).
- **`OfflineProvider`** — deterministic, no-network provider; every example runs
  offline and CI smoke-tests them with API keys stripped.
- **OpenAI-compatible `base_url`** on `OpenAIProvider` (env: `OPENAI_BASE_URL`) for
  Ollama, LM Studio, vLLM, and other compatible endpoints.
- **Release workflow** — tag-triggered PyPI publishing via trusted publishing, plus
  README badges and a public [ROADMAP.md](ROADMAP.md).

- **Durable, resumable execution** — `Swarm(checkpoint_store=...)` persists the full
  execution state (graph, node results, agents, budget consumed) after every node.
  `swarm.resume(execution_id)` picks up from the last completed node; finished nodes
  are never re-executed and cost accounting continues against the original cap.
  Ships with `FileCheckpointStore` (one JSON file per execution, atomic writes) and a
  `CheckpointStore` ABC for custom backends. Format documented in
  docs/checkpoint-format.md; demonstrated in examples/04_resume_after_crash.py.

- **Per-node timeouts** — `Node.timeout_s` (also settable in YAML) caps the wall-clock
  time of a single execution attempt in both executors; timeouts are handled by the
  node's failure policy like any other error.
- **Concurrency cap** — `AsyncExecutor(max_concurrency=...)` bounds in-flight provider
  calls; exposed as `Swarm(max_concurrency=...)` with a default of 8.
- **`examples/` directory** — three runnable scripts (YAML quickstart, dynamic LLM
  planning, budget-capped parallel run) that work offline via a built-in demo provider.

### Changed

- README aligned with shipped behavior: recursive subgraph decomposition, approval
  gates, and performance-history routing are now explicitly labeled roadmap items.
- Executor and AsyncExecutor share a single provider-call path
  (`ExecutorBase.acall_node`), removing duplicated prompt-building logic.

### Fixed

- Deflaked `test_registry_cache_expires_after_ttl` (deterministic clock instead of
  `time.sleep`).

---

## [0.1.0] - 2026-03-28

Initial public release.

### Added — Core runtime

- **Task -> Architect -> Graph -> Executor -> Synthesizer pipeline.** The full
  orchestration loop from a `Task` through plan generation, parallel or serial
  execution, and output synthesis.
- **`ExecutionGraph` DAG model** with first-class topology
  (`SERIAL`, `FORK_JOIN`, `BROADCAST_REDUCE`), per-node status (`PENDING`,
  `RUNNING`, `COMPLETED`, `FAILED`, `SKIPPED`), failure policies
  (`HALT`, `SKIP`, `RETRY`), and dependency edges with cycle, duplicate-ID,
  and unknown-dependency validation.
- **Three-tier architect routing via `WhiteRabbit`** — deterministic
  (template-based), constrained (LLM with strict topology vocabulary), and
  autonomous (`LLMArchitect`, full freedom). Routes by classifier prompt or
  explicit override.
- **`Architect` implementations** — `SimpleArchitect` (single-node fallback),
  `DeterministicArchitect` (template-based), `ConstrainedArchitect`
  (LLM with restricted topology vocabulary), `LLMArchitect` (autonomous,
  with JSON-mode planning and context-preserving retries on malformed output).
- **`AsyncExecutor`** — concurrent DAG execution with topological wave
  scheduling, deadlock detection, and partial-reservation rollback on
  budget exhaustion mid-wave.
- **`Executor`** — serial DAG execution with the same failure-policy
  semantics as the async executor.
- **`Sentinel` budget guardrails** — reservation/record/release protocol
  for safe concurrent cost tracking, with hard USD caps and per-node
  cost attribution.
- **`Synthesizer`** with three strategies: `CONCATENATE` (zero-cost join),
  `STRUCTURED` (JSON shallow-merge), and `LLM_MERGE` (provider-backed
  intelligent synthesis with optional budget and tracer integration).
- **`Tracer`** — structured per-node spans with start/end/error hooks
  and a JSON-serializable summary for downstream observability.
- **`Registry` and `Agent`/`AgentProfile`** — persistent agent identities
  with capabilities, persona, and append-only execution history.
- **`PlannerMemory`** — JSONL-backed outcome store for the architect
  feedback loop (recall surface implemented; closing the loop into
  prompt context is on the roadmap).
- **YAML pipeline loader** (`Swarm.from_yaml`) — declare a graph and
  agent registry in YAML and execute it directly.
- **Skills system** (`SkillRef`, `SkillProvider`, `CapabilityMapper`)
  for capability hydration from external skill inventories.

### Added — LLM providers

- **`AnthropicProvider`** — async wrapper over the official `anthropic` SDK.
- **`OpenAIProvider`** — async wrapper over the official `openai` SDK.
- **`GeminiProvider`** — async wrapper over the official `google-genai` SDK,
  including support for `gemini-3-pro-image-preview` and other Gemini models.
- **Auto-detection** in `Swarm` — picks the right provider from the model
  name prefix (`claude*`, `gpt*`/`o1`/`o3`/`o4`, `gemini*`).
- **`OpenClawSkillProvider`** — adapter for OpenClaw `AgentSkills`,
  translating SDK skill objects into `SkillRef`s for capability hydration.

### Added — Testing & CI

- **240 passing tests, 3 skipped** across the full suite, including
  dedicated test files for router edge cases, tracer/span lifecycle,
  agent model invariants, and a full pipeline integration suite.
- **Shared test fixtures** in `tests/helpers.py` (mock providers,
  failing providers, classifier mocks, fixed architects, completed-graph
  builder).
- **GitHub Actions CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml))
  with `ruff` lint and a `pytest` matrix across Python 3.11 / 3.12 / 3.13.

### Fixed

- **Async exception masking** — `AsyncExecutor` cascading failures no
  longer mask the original exception with a `RuntimeError`.
- **Partial-reservation leak in `AsyncExecutor`** — if `Sentinel.reserve()`
  fails partway through a wave, all previously successful reservations in
  that wave are now released before the exception propagates.
- **Synthesizer model passthrough** — `LLM_MERGE` synthesis now receives the
  swarm's configured model instead of an empty string.
- **Direct-graph validation** — `Swarm.execute(graph)` (with a pre-built
  `ExecutionGraph` instead of a `Task`) now runs `graph.validate()` before
  execution.
- **Executor dependency guard** — `Executor._walk` now raises a clear
  `ValueError` (instead of a `KeyError`) when a node depends on an unknown
  node ID.
- **`LLMArchitect` retry robustness** — `aplan` now also recovers from
  `TypeError` during LLM-response parsing, not only `ValueError`.

### Documentation

- **`Readme.md`** — full pitch, four worked examples (fork-join, broadcast-reduce,
  YAML pipeline, agent registry), API reference for the public surface,
  installation instructions for each provider extra, and an "Async usage"
  section documenting `asyncio.run()` limitations and recommending the
  async APIs (`aplan`, `execute_async`).
- **`LICENSE`** — MIT.

### Known issues

- **`tests/test_skills_registry.py::test_registry_cache_expires_after_ttl`** is
  timing-flaky on Windows under suite load. The test uses a 50ms TTL with a
  60ms sleep, which is too tight for `time.sleep()` precision on Windows.
  Passes consistently in isolation. Tracked for fix in 0.1.1.

[Unreleased]: https://github.com/petehottelet/smythe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/petehottelet/smythe/releases/tag/v0.1.0
