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

## Now: benchmarks

The demo shows what Smythe does; benchmarks show whether it's worth it. The harness and first self-baseline numbers are published in [benchmarks/](benchmarks/) — including the losses, which already paid for themselves by exposing (and fixing) two real executor bugs. Everything else waits behind this.

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
