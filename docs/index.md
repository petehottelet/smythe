# Smythe documentation

Smythe generates an execution graph for a goal, exposes that graph for
inspection, and runs it inside a durable envelope of cost, concurrency,
verification, trace, artifact, and recovery controls.

## Start here

- [README](../README.md) — product overview, measured evidence, and quickstart
- [Architecture](architecture.md) — the two core abstractions and component boundaries
- [Support matrix](support-matrix.md) — which providers, verifiers, tools, stores and Jobs profiles each execution path accepts, with the test behind each cell
- [Examples](../examples/README.md) — offline-first feature tours plus explicitly gated live integrations
- [Benchmarks](../benchmarks/README.md) — evidence status, protocols, and raw records
- [Smythe 0.9.1](release-0.9.1.md) — patch release: missing Gemini images fail their node, Gemini 3.1 image defaults, and account identifiers and local paths kept out of benchmark records; upgrading from 0.9.0
- [Smythe 0.9.0](release-0.9.0.md) — Noumenon re-run, separate screensaver and Repo Doctor repositories, and a pinned evidence archive; upgrading from 0.8.2
- [Smythe 0.8.2](release-0.8.2.md) — correctness release: what changed and how to upgrade from 0.8.1
- [Smythe 0.8.1](release-0.8.1.md) — security release: what changed and how to upgrade from 0.8.0
- [Smythe 0.8.0](release-0.8.0.md) — scope, compatibility changes and release verification
- [Smythe 0.7.0 verification](release-0.7.0.md) — published package checks and retained evidence

## Plan and execute graphs

- [Architecture](architecture.md) — planning tiers, declarative routing rules and a single-worker tier, routing decision records, the contribution lint that records parallel workers given identical work, opt-in claim exchange that gives a worker its dependencies' claims in place of their full output, execution flow, and validated planner history with summed node timing
- [Learning from runs](learning.md) — distilled templates, versioned template records with applicability, prerequisites, evidence and a review of labels that repeat the source goal and of text holding a path, graph structure digests, validation of records as untrusted candidates that cannot grant authority or weaken their parent's gates, an append-only content-addressed template catalog, and pinning an exact template version in durable runs
- [Tasks and handoffs](tasks.md) — complete task snapshots across planning, inspected graphs, synthesis, memory, and resume, immutable context snapshots that keep sources as untrusted data in durable runs, and removing a source with a tombstone
- [Execution policies](execution.md) — halt, retry, skip, timeouts, an opt-in retry policy with one retry owner, `Retry-After`, capped backoff, a shared deadline, attempts that resume cannot refill and kept exposure for calls sent without an answer, an opt-in drain of in-flight work after a failure, opt-in capacity limits per provider type and model with call spacing, `Retry-After` cooldowns, deferred admission and round-robin turns across branches, queued work, provider checks before any call, developer-declared model profiles that route nodes to approved providers and models and can fall back to another profile on a retry, and 5,000-node deep-graph checks
- [YAML and jobs](jobs.md) — declarative artifact work and the installed CLI
- [Adaptive supervision](supervisor.md) — revise pending work from completed results, evidence-triggered review that calls a model only when a declared stage or an anomaly in the run's recorded state fires, and a bounded revision log of what each decision kept, invalidated and added
- [Verification](verifier.md) — objective gates, active-work invalidation, recoverable regeneration, and strict acceptance of the delivered output, with its obligations checked before any call and after every revision, explicit re-evaluation of a completed run's saved output, and authorized operator decisions on output left for review
- [Source evidence](source-evidence.md) — versioned source and claim records an artifact carries, mechanical checks that its quotes and listed contradictions hold against the run's current sources, unavailable and removed sources, synthesis that keeps listed contradictions, and evidence journaled in durable runs

## Operate durable work

- [Durable text workflows](workflow-accounting.md) — phase-wide native costs, saved graph limits, exact request quotes, fenced ownership, local response replay, recorded read steps with network limits, strict delivery acceptance journaled with each run and explicit re-evaluation under a changed validator, published documents whose acceptance binds their bytes, and journal upgrades
- [Artifact lineage](lineage.md) — keep an archive of accepted documents current as its sources, tools and validators change: the opt-in lineage policy for strict durable runs, with a declared access scope and the reuse switch, the versioned record of ids, versions and digests that each completed run writes, with its coverage, reuse decisions and their stable reason codes, read-only repair plans with archive heads, predicted work and derived statuses, in-place repair that decides a run again or publishes its damaged document again without running a node, applying a reviewed plan under an explicit allowance with successor runs that keep unaffected results, source restriction, which withdraws a source's text from every run that used it and can purge their published documents, cache and access rules, inspection, and what ordinary runs and Jobs support
- [Archive-maintenance example](../examples/18_archive_maintenance.py) — offline and keyless: four lineage runs share bundled sources; a revised source, a damaged brief and a bumped validator are repaired from a reviewed plan, and the result is compared with a full recomputation
- [Checkpoint format](checkpoint-format.md) — flushed atomic publication, saved state, and resume semantics
- [Inspect saved runs](runs.md) — read-only `smythe runs inspect` and `list` over checkpoints and durable journals: execution, delivery acceptance and its decision history, the supervisor's revision log, the routing decision that chose the planning tier, a research run's context, reads, evidence and open issues, a lineage summary, external actions, cost, timing, next action, exit codes 10 and 11 for withheld output and 12 for an unresolved external action, a read-only `review` of what in a run or the newest runs needs a decision and the entry point or command for each, and the `revoke-action` fail-safe control
- [Source-to-artifact example](../examples/16_source_to_artifact.py) — offline and keyless: plan, journaled reads of bundled sources, an interruption, resume, and inspection of the accepted document and its evidence
- [External actions](actions.md) — preview: deliver a strict durable run's accepted output to another service through your connector, under approvals your host verifies for the exact intent, with one key across attempts, independent readback, reconciliation of lost answers within what the connector declares, revocation, compensation, operator decisions and where actions run
- [Governed-publication example](../examples/19_governed_publication.py) — offline and keyless: an accepted announcement prepared, approved and published to an in-memory service, a lost answer reconciled without a second write, and inspection
- [Cost guardrails](budgets.md) — strict usage validation, reservations, an opt-in allowance held for verification and synthesis, per-call ceilings, and failed-accounting recovery
- [Jobs](jobs.md) — preflight, approval, detached workers and host limits, durable pause/resume, fenced ownership, artifact namespaces, delivery acceptance with registered validators and delivery eligibility, database upgrades, inspection, a read-only review of what in a run needs a decision and the command for each, rerolls, integrity repair of corrupted accepted outputs, adoption of published bytes for unknown outcomes, rechecks of accepted outputs under changed validators, a read-only repair preview, and exports
- [Optimization](optimize.md) — bounded concurrency experiments, reference arms, evidence ledgers, and campaign ownership, schema-v4 migration, read-only HTML comparison reports, and a policy registry with append-only activation history, rollback and retirement
- [Concurrent journal initialization checks](../tests/test_jobs_workflow_initialization.py) — WAL contention and atomic schema fixes, with regression evidence

## Connect models and tools

- [GPT-6 Astra and Claude Fable 5.1 quickstarts](../README.md#with-a-real-model) — native text planning and execution within one saved $50 allowance
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
- [Skills](skills.md) — capability tags from Agent Skills directories, MCP tools and OpenClaw, what a skill can and cannot change, static snapshots for durable runs, and Python-backed skills as tools offered only to the agents you name
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
