# Smythe Roadmap

Where the project is going, in order. Everything here converges on one
artifact: **a published, auditable benchmark where Smythe agents do real
tool-using and artifact-producing work at high fan-out, under bounded cost,
resumable after a crash, with honest numbers against fixed pipelines and other
frameworks.** Honest means we publish where Smythe loses, too.

Status: pre-1.0. Minor versions may break APIs (see [CHANGELOG.md](CHANGELOG.md) for the versioning policy).

## Shipped (v0.2 line)

- ✅ Durable, resumable execution — per-node checkpointing, `swarm.resume()`, pluggable stores ([docs](docs/checkpoint-format.md))
- ✅ Per-node timeouts and bounded parallel concurrency
- ✅ Graph export — `to_mermaid()`, `to_dot()`, `to_json()`
- ✅ `OfflineProvider` — evaluate the full pipeline with no API keys; all examples run offline
- ✅ Provider tool contract — neutral tool-calling types mapped to Anthropic, OpenAI, and Gemini native tool use
- ✅ **MCP tool support** — agents use MCP servers (stdio + streamable HTTP) through a bounded, budget-enforced tool loop; secrets via `env_passthrough`; capability hydration and planner tool awareness; examples for filesystem, GitHub, and SaaS servers ([docs/mcp.md](docs/mcp.md))
- ✅ OpenAI-compatible `base_url` (Ollama, LM Studio, vLLM)
- ✅ v0.2.0 on PyPI (`pip install smythe`)
- ✅ **Flagship demo** — the acquisition-diligence showcase: fixture mode (no keys) and real mode, with committed graph, trace, and expected output ([examples/acquisition_diligence/](examples/acquisition_diligence/))

## Shipped (v0.5 line)

- ✅ **Multimodal artifact pipeline** — image generation (Gemini, GPT Image)
  with per-image cost accounting, execution-scoped artifact persistence,
  and cost-aware parallel budget reservations
- ✅ **Vision input** — nodes see their dependencies' images
  (`attach_dep_artifacts`); the select-from-N art-director pattern
- ✅ **Image concurrency benchmark, published** — 6.6× wall-clock at k=8,
  25 images in 10.2 s at k=25, objective metrics only
  ([benchmarks/image_benchmarks.md](benchmarks/image_benchmarks.md))
- ✅ **Repo Doctor MVP** — offline-first release-readiness auditor built
  on smythe ([skills/repo-doctor/](skills/repo-doctor/))

## Shipped (v0.6 line)

- ✅ **Brand-locked asset factory** — eight exact-spec formats, shared logo
  vision input, deterministic typography, and vision-based consistency judging
- ✅ **Framework head-to-head** — an ecological comparison against idiomatic
  LangGraph and CrewAI implementations, with losses and confounds published
- ✅ **Judge variance measurement + bounded optimizer smoke test** — noisy
  candidates are reverted and every experiment is journaled

## Unreleased: trust at production fan-out

The current unreleased work hardens the guarantees users depend on before a
5,000-item job is credible:

1. **Fail-closed budget reservations** for image and other non-token outputs,
   using explicit inclusive per-call price ceilings
2. **Bounded scheduling and cancellation** so a wide ready wave does not create
   thousands of live coroutines or leave sibling provider calls running after
   a fatal failure
3. **Crash-safe artifacts and checkpoints** with atomic writes and configurable
   batched full snapshots. The batch size trades write amplification against
   crash granularity and possible duplicate spend: at most the unflushed tail
   of a batch is replayed.
4. **Portable benchmark evidence** — installable optional harness dependencies,
   repo-relative artifact references, protocol metadata, and offline CI coverage

## Next

1. **Productize the artifact-factory wedge** — typed job manifests, preflight
   estimate/approval, OCR and brand validators, partial rerolls, and select-from-N
   curation
2. **Operator surface** — `smythe plan`, `run`, `inspect`, and `replay`, with a
   rendered DAG, per-node prompt/response/cost/duration, artifacts, and the
   Architect's reasoning
3. **Scale ladder** — offline 5,000-item stress tests followed by bounded paid
   50/250/1,000-item trials with kill-and-resume and duplicate detection
4. **Benchmarks, continued** — a discriminating judge, human calibration,
   repeated k=25 cells, and held-out brand-consistency comparisons

## Later

- A constrained `smythe optimize` research loop: immutable evaluation contract,
  small mutable candidate surface, held-out tasks, repeated end-to-end trials,
  confidence-aware promotion, and append-only resumable history
- Recursive subgraph decomposition, with depth limits and shared
  budget/trace/failure machinery
- Human-in-the-loop approval gates (pause/approve/reject, state survives restart)
- Provider hardening: retry with backoff, streaming, response caching
- Template/starter library and a `smythe init` command
- Docs site, OpenTelemetry export, local-model capability metadata

## Out of scope (deliberately)

- A hosted SaaS or agent marketplace — the OSS core comes first
- Rewriting in Rust/Go — Python is the right language for this audience

## Contributing

Issues tagged `good first issue` are curated entry points; see [CONTRIBUTING.md](CONTRIBUTING.md). If you want to work on a roadmap item, open an issue first so we can align on design.
