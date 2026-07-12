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

### Added

- **Vision input — nodes can see images.** `ChatMessage.attachments`
  carries `Artifact` objects mapped to each provider's native
  multimodal format (Anthropic image blocks, OpenAI `image_url` data
  URIs, Gemini `inline_data` parts). `Node(attach_dep_artifacts=True)`
  feeds a node its dependencies' generated images as actual pixels, not
  paths — the art-director/vision-judge pattern (select-from-N
  curation). Capped at 12 images / 8 MB each; YAML and checkpoint
  support included; `OfflineProvider` acknowledges attachments so the
  path runs deterministically in CI. New example:
  `examples/11_vision_judge.py` — in its first real run the judge
  caught and rejected a candidate ad with a spelling error.

Future work tracked in [ROADMAP.md](ROADMAP.md).

---

## [0.5.0] - 2026-07-12

Parallel images, measured. This release adds the multimodal artifact
pipeline, hardens the parallel executor around it, and publishes the
first benchmark where generated parallel topology demonstrably wins:
6.6x wall-clock at concurrency 8, 25 images in 10.2 seconds at
concurrency 25, at identical cost to serial. No breaking API changes.

### Added

- **Image generation (Gemini "Nano Banana") support.** Provider calls
  can now return binary artifacts: `CompletionResult.artifacts` carries
  `Artifact` objects (bytes + mime type), `GeminiProvider` requests
  image response modalities automatically for `gemini-*-image*` models
  and extracts returned inline images, and executors persist artifacts
  under `Swarm(artifact_dir=...)` (default `smythe_artifacts/`) with
  paths recorded in `node.metadata["artifacts"]` — bytes never enter
  checkpoints or planner memory. `CompletionResult.cost_usd` lets a
  provider price a call explicitly (e.g. `GeminiProvider(
  cost_per_image_usd=...)` for per-image billing) instead of the
  Sentinel's blended token rate. `OfflineProvider(artifacts_per_call=N)`
  returns deterministic PNGs so image pipelines run offline in CI.
  New example: `examples/09_image_generation.py`.
- **GPT Image support.** Dedicated `OpenAIImageProvider`
  (`images.generate` endpoint; size/quality/format/compression/
  moderation controls, per-image cost) with mocked coverage and a
  deterministic offline example (`examples/10_gpt_image_generation.py`).
- **Image concurrency benchmark**
  (`benchmarks/run_image_benchmarks.py`): objective metrics only — wall
  time, throughput, efficiency, decode/format compliance, dHash
  diversity. First published sweep: 6.6x wall-clock speedup at
  concurrency 8, 81–88% parallel efficiency, 72/72 valid images
  (`benchmarks/image_benchmarks.md`).

### Changed — parallel/artifact hardening (post-review)

A multi-agent code review of the artifact pipeline confirmed 8 findings;
all are addressed:

- **Cost-aware parallel reservations.** The AsyncExecutor now reserves
  per-node estimates from (in priority order) `node.metadata
  ["estimated_cost_usd"]`, the provider's new `cost_estimate_per_call`
  hint (set by `cost_per_image_usd` on the image providers), then the
  token estimate — a wide image wave is refused up front instead of
  overshooting `max_budget_usd` mid-flight.
- **Execution-scoped artifact paths.** Artifacts land in
  `artifact_dir/<execution_id>/`, so re-running a graph with fixed node
  ids no longer overwrites the previous run's files; `resume()` reuses
  the original directory. Filenames are sanitized (node ids from
  YAML/LLM plans can carry path separators or Windows-illegal
  characters) and recorded as absolute paths so checkpoints survive a
  cwd change.
- **`node.result` is the provider text verbatim again.** Artifact paths
  live in `node.metadata["artifacts"]` and are surfaced to dependent
  nodes by `gather_dep_results` — JSON results stay parseable by
  STRUCTURED synthesis and downstream consumers.
- **Tool-loop artifacts survive.** Images returned on intermediate
  tool-calling turns (already billed) are carried to the final result
  and persisted instead of silently dropped.
- **Gemini: tools no longer collide with image modalities** (auto-detect
  is suppressed when tools are passed; explicit `response_modalities`
  still win), and `image_config` (e.g. `{"aspect_ratio": "16:9"}`) is
  passed through for true format control.
- Artifact writes run off the event loop (`asyncio.to_thread`) in the
  parallel executor; opt-in full-jitter retry backoff
  (`retry_backoff_s` on `Swarm`/executors) for rate-limited fan-outs;
  negative provider costs are clamped so a buggy provider can't refund
  the budget; both `execute()` paths share one `_prepare_graph` helper.

### Fixed

- **Hand-built graphs passed directly to `Swarm.execute()` never got a
  model stamped onto their nodes**, so real providers rejected the call
  with an empty model name ("model is required"). Only `plan()`,
  `from_yaml()`, and `resume()` stamped models; every offline example
  masked the bug because `OfflineProvider` ignores the model string.
  Both execute paths now stamp the swarm's model onto unstamped nodes.
- `smythe.__version__` was stale at `0.2.0`; it now matches the
  released package version.


---

## [0.4.0] - 2026-07-07

Evidence. This release is what the benchmark campaign found and fixed:
every number, loss, and null result is published in `benchmarks/`.
No breaking API changes.

### Fixed

- **Every node now receives the task payload.** The Architect saw the
  full task when planning, but generated node labels rarely reproduce
  its payload (source code, documents, data) — so specialists worked
  from a one-line label and the material never entered the graph, and
  verifier nodes hedged claims about artifacts they couldn't see.
  `Swarm.plan()` stamps the goal and constraints into each node's
  metadata. Measured: dynamic-topology code review went 1.7/10 →
  9.0/10 across this fix (roots first, then all nodes).
- **Terminal nodes are told their output is the deliverable.** Final
  nodes tended to reference or summarize upstream findings instead of
  reproducing them, so specifics were lost from the returned result.
- **The Architect right-sizes graphs.** Cost-aware node-count guidance
  in the planning prompt; single-artifact tasks dropped from 5-node
  fork-joins to 3-node serial-adversarial graphs (−45% cost on code
  review, quality flat).

### Added

- Benchmark harness: `--runs N` repeats with mean/range reporting, an
  independent judge model (different from the executor, to reduce
  self-preference bias), ablation flags for the executor fixes, and
  source documents for the acquisition-diligence task.
- Memory A/B benchmark (`benchmarks/run_memory_ab.py`) with observable
  recall wiring; first memory-on/off numbers published (null on a
  homogeneous task family — see `benchmarks/README.md`).
- Full benchmark results: the v2→v5 progression, raw records, and
  judge reasoning committed under `benchmarks/results/`.

---

## [0.3.0] - 2026-07-05

Flagship proof. The demo the README promises now exists, runs offline,
and its expected output is committed and drift-tested. No breaking API
changes.

### Added

- **Flagship demo** — `examples/acquisition_diligence/`: task intake, an
  Architect-generated `fork-join -> adversarial -> serial` topology, three
  parallel specialists, a red-team tier, and a final structured memo.
  Fixture mode (default, no keys) is deterministic even under parallel
  execution; any provider API key switches the same script to real mode.
  The expected graph (Mermaid), trace, and memo are committed under
  `expected/`, regenerable with `--write-artifacts`, and guarded by a
  drift test in CI.

### Fixed

- **Agent names in rendered graphs** — `TaskGraph` trees, Mermaid, and DOT
  exports labeled nodes with the assigned agent's random hex id instead of
  its name. Assignment (loader and registry) now stamps `agent_name` into
  node metadata and rendering prefers it; checkpointed graphs render the
  same after resume.
- **CI** — the workflow now installs the `dev` extra (previously it
  hand-picked pytest packages and missed the `mcp` SDK, failing the MCP
  example smoke test). MCP-dependent examples are skipped when the `mcp`
  package is absent so a plain `pip install -e .` checkout tests green.

### Changed

- **README** — restructured around install, a 60-second quickstart, and
  the flagship demo's real output; renamed `Readme.md` to `README.md`.
- **Roadmap** — benchmarks now precede recursive subgraph decomposition.
- GitHub Actions bumped to current majors (Node 20 deprecation).

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
