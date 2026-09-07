# Smythe documentation

Smythe generates an execution graph for a goal, exposes that graph for
inspection, and runs it inside a durable envelope of cost, concurrency,
verification, trace, artifact, and recovery controls.

## Start here

- [README](../README.md) — product overview, measured evidence, and quickstart
- [Architecture](architecture.md) — the two core abstractions and component boundaries
- [Examples](../examples/README.md) — offline-first feature tours plus explicitly gated live integrations
- [Benchmarks](../benchmarks/README.md) — evidence status, protocols, and raw records

## Plan and execute graphs

- [Architecture](architecture.md) — planning tiers, execution flow, and learning loop
- [YAML and jobs](jobs.md) — declarative artifact work and the installed CLI
- [Adaptive supervision](supervisor.md) — revise pending work from completed results
- [Verification](verifier.md) — enforce objective acceptance gates

## Operate durable work

- [Checkpoint format](checkpoint-format.md) — saved state and resume semantics
- [Jobs](jobs.md) — preflight, approval, attempts, recovery, rerolls, and exports
- [Optimization](optimize.md) — bounded concurrency experiments and evidence ledgers

## Connect models and tools

- [MCP](mcp.md) — tool discovery, allowlists, secrets, budgets, and timeouts
- [Style](style.md) — visual language for diagrams and public assets

## Artifact workflows

- [Framework comparison](../benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) — matched Smythe, LangGraph, and CrewAI evidence
- [Image benchmarks](../benchmarks/image_benchmarks.md) — image fan-out and exact-spec finishing
- [Glyph Rain benchmark](../benchmarks/glyph_screensaver_benchmark.md) — an artifact fan-out example with isolated 64-, 128-, 192-, and 256-node measurements
- [Glyph Rain screensaver](../screensaver/README.md) — shared glyph catalog, heavy strokes, green cores, and layered glow; verified Windows, universal Mac, and Linux downloads

## Project guides

- [Repository review](project-review-2026-09-06.md) — architecture assessment, reproduced defects, and hardening priorities
- [Coming soon](../ROADMAP.md#coming-soon) — acceptance criteria for runtime and measurement improvements
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)
- [Releasing](../RELEASING.md)
- [Security](../SECURITY.md)
