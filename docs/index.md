# Smythe documentation

Smythe generates an execution graph for a goal, exposes that graph for
inspection, and runs it inside a durable envelope of cost, concurrency,
verification, trace, artifact, and recovery controls.

## Start here

- [README](../README.md) — product overview, measured evidence, and quickstart
- [Architecture](architecture.md) — the two core abstractions and component boundaries
- [Examples](../examples/README.md) — offline-first feature tours plus explicitly gated live integrations
- [Benchmarks](../benchmarks/README.md) — evidence status, protocols, and raw records
- [Smythe 0.9.0](release-0.9.0.md) — Noumenon re-run, separate screensaver and Repo Doctor repositories, and a pinned evidence archive; upgrading from 0.8.2
- [Smythe 0.8.2](release-0.8.2.md) — correctness release: what changed and how to upgrade from 0.8.1
- [Smythe 0.8.1](release-0.8.1.md) — security release: what changed and how to upgrade from 0.8.0
- [Smythe 0.8.0](release-0.8.0.md) — scope, compatibility changes and release verification
- [Smythe 0.7.0 verification](release-0.7.0.md) — published package checks and retained evidence

## Plan and execute graphs

- [Architecture](architecture.md) — planning tiers, execution flow, and validated planner history with summed node timing
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
- [Optimization](optimize.md) — bounded concurrency experiments, evidence ledgers, and campaign ownership, schema-v4 migration, and read-only HTML comparison reports
- [Concurrent journal initialization checks](../tests/test_jobs_workflow_initialization.py) — WAL contention and atomic schema fixes, with regression evidence

## Connect models and tools

- [GPT-6 Astra quickstart](../README.md#quickstart) — native text planning and execution within one saved $5 allowance
- [Native OpenAI Responses](openai-responses.md) — Astra/Sol function tools, exact token prices, request quotes, and retained failure receipts
- [Original Astra pilot results](../benchmarks/results/astra_20260913/README.md) — historical 24-workflow calibration snapshot with complete native receipts
- [What the Astra study found](../benchmarks/astra_findings.md) — scoped findings and an offline reproduction supplement
- [Astra/Sol main results](../benchmarks/results/astra_20260913_main/README.md) — 200 outcomes, completed human review, automatic scores, timing and reserved-cost bounds; claimable within the recorded scope
- [Astra benchmark plan](../benchmarks/astra_benchmark_plan.md) — completed execution and human review, fixed acceptance gates and separate follow-up experiments
- [Fable 5.1 extension](../benchmarks/fable_51_benchmark_plan.md) — native Messages runner, 12-pilot/100-main design, human review gates and a separate Ultracode comparison
- [Fable pilot results](../benchmarks/results/fable_20260914_pilot/README.md) — 12 native workflows and one Code Workflow pilot; raw receipts, exact native costs, blind judging, and pending human review
- [Fable Code Workflow results](../benchmarks/results/fable_code_20260914/README.md) — ten completed tasks, five retained diagnostics, native billing and blind-judge reasoning; native comparison awaits pilot ratings
- [Native Claude Messages](anthropic-messages.md) — exact Fable cache billing, conservative request quotes, and saved-response recovery
- [Astra method amendment](../benchmarks/astra_method_amendment_20260913.md) — planner-only instructions, explicit field types, 48 pilot workflows and preserved diagnostic charges
- [Astra campaign runners](../benchmarks/astra_runtime.md) — pilot, main, approved reserved-cost continuation, blind judging and offline evidence analysis
- [MCP](mcp.md) — tool discovery, allowlists, secrets, budgets, and timeouts
- [Style](style.md) — visual language for diagrams and public assets

## Artifact workflows

- [Current materials](current-materials.md) — direct links to revised contact sheets, previews, exports, reports, and the publication completion check
- [Glyph contour review](glyph-contour-review-2026-09-12.md) — reference measurements and the full 192-glyph v2 replacement catalog
- [192/256-glyph workflow results](../benchmarks/svg_v2_results.md) — 36 completed v2 workflows, timing and memory charts; [protocol](../benchmarks/svg_v2_protocol.md) and raw receipts
- [256-glyph contact sheets](../benchmarks/partitions/glyph_svg_v2_256/README.md) — the reviewed 192 plus 64 additional structures, with no changes to the live catalog
- [Jobs at 5,000 operations](../benchmarks/jobs_scale_5000_20260907_results.md) — one reconciled offline recovery campaign, with all artifacts, journal entries, source hashes, and interrupted-attempt lineage retained
- [Framework comparison](../benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) — matched Smythe, LangGraph, and CrewAI evidence
- [Image benchmarks](../benchmarks/image_benchmarks.md) — image fan-out and exact-spec finishing
- [Noumenon fan-out benchmark](../benchmarks/noumenon_benchmark.md) — Smythe's parallel-processing example, with isolated 64-, 128-, 192-, and 256-node measurements
- [Historical v1 SVG workflow](../benchmarks/svg_glyph_benchmark.md) — archived geometry, validation, assembly, and timing records; current artwork is linked above
- [Noumenon screensaver](https://github.com/petehottelet/noumenon) — the screensaver, web explorer, and Windows, macOS, and Linux source builds using 56 reference and 192 original SVG shapes, in their own repository; no precompiled distribution
- [Glyph reference measurements](data/glyph-style-summary.json) — source-population distributions; [measurement method](data/glyph-style-method.json)
- [Retired benchmark evidence](../benchmarks/archive/README.md) — superseded, diagnostic, and historical records, including the renderer timing study, in one SHA-256-pinned archive

## Project guides

- [Coming soon](../ROADMAP.md#coming-soon) — acceptance criteria for runtime and measurement improvements
- [Contributing](../CONTRIBUTING.md)
- [Packages and typing](distribution.md) — slim source archives, explicit library tests, and installed type checks
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md) — current releases; 0.1.0 through 0.8.1 are in the [changelog archive](../CHANGELOG-ARCHIVE.md)
- [Releasing](../RELEASING.md) — package builds, release assets, and PyPI publication
- [Security](../SECURITY.md)
