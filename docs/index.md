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
- [Execution policies](execution.md) — halt, retry, skip, timeouts, and queued work
- [YAML and jobs](jobs.md) — declarative artifact work and the installed CLI
- [Adaptive supervision](supervisor.md) — revise pending work from completed results
- [Verification](verifier.md) — enforce objective acceptance gates

## Operate durable work

- [Checkpoint format](checkpoint-format.md) — saved state and resume semantics
- [Cost guardrails](budgets.md) — strict usage validation, reservations, and failed-accounting recovery
- [Jobs](jobs.md) — preflight, approval, attempts, recovery, rerolls, and exports
- [Optimization](optimize.md) — bounded concurrency experiments and evidence ledgers

## Connect models and tools

- [GPT-6 Astra quickstart](../README.md#quickstart) — text-only planning and execution
- [Astra benchmark plan](../benchmarks/astra_benchmark_plan.md) — compatibility, complete usage accounting, and matched comparisons
- [MCP](mcp.md) — tool discovery, allowlists, secrets, budgets, and timeouts
- [Style](style.md) — visual language for diagrams and public assets

## Artifact workflows

- [Framework comparison](../benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) — matched Smythe, LangGraph, and CrewAI evidence
- [Image benchmarks](../benchmarks/image_benchmarks.md) — image fan-out and exact-spec finishing
- [Glyph Rain benchmark](../benchmarks/glyph_screensaver_benchmark.md) — an artifact fan-out example with isolated 64-, 128-, 192-, and 256-node measurements
- [Original SVG workflow](../benchmarks/svg_glyph_benchmark.md) — fresh geometry, complete style validation, catalog assembly, and repeated thread/process measurements
- [Glyph Rain screensaver](../screensaver/README.md) — verified Windows, universal Mac, and Linux downloads using the current 56 reference and 192 original SVG shapes, with a 10% original mix
- [Web explorer](../screensaver/svg-preview/README.md) — MIT-licensed reference renderer and base glyphs, 10% original-glyph mix, Matrix green rain, presets, VT323 pixel controls, and a Trajan Bold outline logo; browser interaction checks pass, performance remains unmeasured
- [Glyph Rain design plan](glyph-rain-plan.md) — measured original-glyph distributions, licensed renderer scope, and native porting criteria
- [Reference behavior plan](glyph-rain-parity-plan.md) — pinned defaults, options, and acceptance checks for the adapted effect

## Project guides

- [Repository review](project-review-2026-09-06.md) — architecture assessment, reproduced defects, and hardening priorities
- [Coming soon](../ROADMAP.md#coming-soon) — acceptance criteria for runtime and measurement improvements
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)
- [Releasing](../RELEASING.md)
- [Security](../SECURITY.md)
