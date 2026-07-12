# Smythe Roadmap

Where the project is going, in order. Everything here converges on one artifact: **a published, reproducible benchmark where Smythe agents do real tool-using work — recursively decomposed, resumable after a crash — with honest numbers against fixed pipelines and other frameworks.** Honest means we publish where Smythe loses, too.

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

## Now: quality benchmarks + head-to-heads

The parallel *performance* claim is demonstrated. Next to measure:
the quality tier (select-from-N curation quality-per-dollar; ad-suite
brand-consistency: shared brief vs. serial style stage vs. single
agent) and the head-to-head vs. LangGraph and CrewAI.

## Next

1. **Benchmarks, continued** — head-to-head vs. LangGraph and CrewAI (memory-on/off numbers are published: null on a homogeneous family; a harder memory task family with planted failure modes follows)
2. **Recursive subgraph decomposition** — a node can resolve to a nested graph, with depth limits and shared budget/trace/failure machinery
3. **Trace inspector** — `smythe inspect`: rendered DAG, per-node prompt/response/cost/duration, and the Architect's reasoning

## Later

- Human-in-the-loop approval gates (pause/approve/reject, state survives restart)
- Provider hardening: retry with backoff, streaming, response caching
- Template/starter library and a `smythe` CLI (`init`, `run`, `plan`, `inspect`, `replay`)
- Docs site, OpenTelemetry export, local-model capability metadata

## Out of scope (deliberately)

- A hosted SaaS or agent marketplace — the OSS core comes first
- Rewriting in Rust/Go — Python is the right language for this audience

## Contributing

Issues tagged `good first issue` are curated entry points; see [CONTRIBUTING.md](CONTRIBUTING.md). If you want to work on a roadmap item, open an issue first so we can align on design.
