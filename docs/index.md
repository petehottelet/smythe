# Smythe documentation

Smythe generates an execution graph for a goal, exposes that graph for
inspection, and runs it inside a durable envelope of cost, concurrency,
verification, trace, artifact, and recovery controls.

## Start here

- [README](../README.md) — product overview, measured evidence, and quickstart
- [Architecture](architecture.md) — the two core abstractions and component boundaries
- [Examples](../examples/README.md) — offline-first feature tours plus explicitly gated live integrations
- [Benchmarks](../benchmarks/README.md) — evidence status, protocols, and raw records
- [Smythe 0.7.0 verification](release-0.7.0.md) — published package checks, current native screensaver downloads, checksums, and retained evidence

## Plan and execute graphs

- [Architecture](architecture.md) — planning tiers, execution flow, and learning loop
- [Tasks and handoffs](tasks.md) — complete task snapshots across planning, inspected graphs, synthesis, memory, and resume
- [Execution policies](execution.md) — halt, retry, skip, timeouts, queued work, and 5,000-node deep-graph checks
- [YAML and jobs](jobs.md) — declarative artifact work and the installed CLI
- [Adaptive supervision](supervisor.md) — revise pending work from completed results
- [Verification](verifier.md) — objective gates, active-work invalidation, and recoverable regeneration

## Operate durable work

- [Durable text workflows](workflow-accounting.md) — phase-wide native costs, saved graph limits, exact request quotes, fenced ownership, and local response replay
- [Checkpoint format](checkpoint-format.md) — flushed atomic publication, saved state, and resume semantics
- [Cost guardrails](budgets.md) — strict usage validation, reservations, and failed-accounting recovery
- [Jobs](jobs.md) — preflight, approval, detached workers and host limits, durable pause/resume, fenced ownership, artifact namespaces, database upgrades, inspection, rerolls, and exports
- [Optimization](optimize.md) — bounded concurrency experiments and evidence ledgers

## Connect models and tools

- [GPT-6 Astra quickstart](../README.md#quickstart) — native text planning and execution within one saved $5 allowance
- [Native OpenAI Responses](openai-responses.md) — Astra/Sol function tools, exact token prices, request quotes, and retained failure receipts
- [Astra benchmark plan](../benchmarks/astra_benchmark_plan.md) — prepared task packs and balanced schedules, complete usage accounting, and matched comparisons
- [Astra pilot runner](../benchmarks/astra_runtime.md) — source-bound spending allocations, native workflow receipts, and recovery of the 12 calibration trials
- [MCP](mcp.md) — tool discovery, allowlists, secrets, budgets, and timeouts
- [Style](style.md) — visual language for diagrams and public assets

## Artifact workflows

- [Jobs at 5,000 operations](../benchmarks/jobs_scale_5000_20260907_results.md) — one reconciled offline recovery campaign, with all artifacts, journal entries, source hashes, and interrupted-attempt lineage retained
- [Framework comparison](../benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) — matched Smythe, LangGraph, and CrewAI evidence
- [Image benchmarks](../benchmarks/image_benchmarks.md) — image fan-out and exact-spec finishing
- [Glyph Rain benchmark](../benchmarks/glyph_screensaver_benchmark.md) — an artifact fan-out example with isolated 64-, 128-, 192-, and 256-node measurements
- [Original SVG workflow](../benchmarks/svg_glyph_benchmark.md) — fresh geometry, complete style validation, catalog assembly, and repeated thread/process measurements
- [Glyph Rain screensaver](../screensaver/README.md) — verified Windows, universal Mac, and Linux downloads using the current 56 reference and 192 original SVG shapes, with a 10% original mix
- [Web explorer](../screensaver/svg-preview/README.md) — MIT-licensed reference renderer and base glyphs, 10% original-glyph mix, Matrix green rain, presets, VT323 pixel controls, and a Trajan Bold outline logo
- [Glyph Rain design plan](glyph-rain-plan.md) — measured original-glyph distributions, licensed renderer scope, and native porting criteria
- [Reference behavior plan](glyph-rain-parity-plan.md) — pinned defaults, options, and acceptance checks for the adapted effect
- [Renderer timing protocol](../benchmarks/renderer_performance_20260907.md) — six independent Classic/3D sessions, raw callback and CPU-submission samples, and explicit backend qualification
- [Renderer measurements](../benchmarks/renderer_performance_20260907_results.md) — all six target-missing sessions, a separate blank-page cadence control, and a ten-minute travel/resize check

## Project guides

- [Repository review](project-review-2026-09-06.md) — architecture assessment, reproduced defects, and hardening priorities
- [Coming soon](../ROADMAP.md#coming-soon) — acceptance criteria for runtime and measurement improvements
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)
- [Releasing](../RELEASING.md) — package builds, release assets, and PyPI publication
- [Security](../SECURITY.md)
