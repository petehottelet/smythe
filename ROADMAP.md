# Smythe Roadmap

Where the project is going, in order. Everything here converges on one
standard: **generated execution graphs that do real tool-using and
artifact-producing work at high fan-out, under bounded cost, with inspectable
plans, durable recovery, and claimable benchmark evidence.** Every public
performance claim links to its protocol and committed result record;
superseded campaigns remain available as diagnostic history.

Status: pre-1.0. Minor versions may break APIs (see [CHANGELOG.md](CHANGELOG.md) for the versioning policy).

## Shipped (v0.2 line)

- ✅ Durable, resumable execution — per-node checkpointing, `swarm.resume()`, pluggable stores ([docs](docs/checkpoint-format.md))
- ✅ Per-node timeouts and bounded parallel concurrency
- ✅ Graph export — `to_mermaid()`, `to_dot()`, `to_json()`
- ✅ `OfflineProvider` — evaluate the full pipeline with no API keys; the core
  suite and most examples run offline, while live MCP examples are explicitly
  environment-gated
- ✅ Provider tool contract — neutral tool-calling types mapped to Anthropic, OpenAI, and Gemini native tool use
- ✅ **MCP tool support** — agents use MCP servers (stdio + streamable HTTP) through a bounded, budget-enforced tool loop; secrets via `env_passthrough`; capability hydration and planner tool awareness; examples for filesystem, GitHub, and SaaS servers ([docs/mcp.md](docs/mcp.md))
- ✅ OpenAI-compatible `base_url` (Ollama, LM Studio, vLLM)
- ✅ v0.2.0 on PyPI (`pip install smythe`)
- ✅ **Acquisition-diligence example** — fixture mode (no keys) and real mode, with committed graph, trace, and expected output ([examples/acquisition_diligence/](examples/acquisition_diligence/))

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
- ✅ **Benchmark fault-finding loop** — an ecological LangGraph/CrewAI
  comparison exposed payload, assembly, and measurement defects; the fixes
  now feed the claimable task-shape and durability suites
- ✅ **Corrected framework comparison** — delivered-output measurement across
  matched fixed-pipeline Smythe, LangGraph, and CrewAI implementations, with
  Smythe leading blind quality, mean tokens, and mean wall time
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
5. **Durable artifact Jobs v1** — strict manifests, deterministic preflight,
   plan-and-ceiling approvals, a SQLite dispatch/event journal, conservative
   unknown outcomes, selective rerolls, portable exports, and an installed CLI
6. **Typed production asset contracts** — concept-versus-production brand
   rules, exact master/text compositing, atomic finishing, hash receipts, and
   deterministic validation gates separated from advisory judgments
7. **Glyph Rain fan-out example** — isolated 64-, 128-, 192-, and 256-node
   partitions with objectively validated unique tiles, realistic-latency
   concurrency sweeps through k=64, live Gemini and GPT Image lanes, and assembled
   preview/GIF/atlas/HTML deliverables plus web, Windows, macOS, and Linux X11
   ports with shared heavy glyph strokes and layered green trails; compiled
   native ports have build and render smoke checks
8. **Bounded Autotune v1**: immutable hash-bound contracts and allowlisted
   candidates, a zero-API-spend offline concurrency campaign, plan-bound async
   orchestration, atomic dispatch claims, paired confirmation and sealed
   per-campaign holdout, exact gate inventories, typed mutation domains,
   bounded statistical/CPU work, immutable budget-visible evidence, and
   read-only `inspect` alongside the `concurrency` command
   ([docs](docs/optimize.md))
9. **Result-aware control and reuse** — acceptance criteria reach planning and
   execution, declarative verification gates regenerate rejected subtrees,
   supervisors revise only pending work under a persisted revision cap, and
   successful graphs distill into constrained-planning templates
10. **Current task-shape evidence** — generated plans record 14% lower wall
    time, including planning, with observed quality in the fixed pipeline's
    measured band. Historical cost totals cover execution and synthesis only;
    complete-workflow cost comparisons require new measurements
    ([report](benchmarks/shape_suite.md))
11. **Current framework evidence** — on the same fixed three-stage semantic
    pipeline, Smythe records the highest blind quality, fewest mean tokens, and
    lowest mean wall time across Smythe, LangGraph, and CrewAI
    ([report](benchmarks/README.md#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12))

## Coming soon

The [September repository review](docs/project-review-2026-09-06.md) defines
the hardening work ahead of broader production claims:

| Priority | Change | Acceptance criteria |
|---|---|---|
| P1 | Complete workflow accounting | Planning, routing, supervision, execution, and synthesis reserve and reconcile spend in one ledger; result completeness covers every provider call. Publish fresh cost comparisons with native usage, declared prices, and all optimization trials retained. |
| P1 | Verification and active work | A rejected target cancels or invalidates every active descendant before regeneration; no final result or resumed checkpoint contains output based on a rejected version. |
| P1 | Serial halt | A fatal node failure prevents all later independent provider calls and retains the original exception and completed results. |
| P1 | Numeric cost validation | Reject NaN, infinity, negative values, and invalid token counts at admission, reconciliation, and checkpoint restore; invalid usage cannot reduce spend or unlock a budget. |
| P2 | Full task propagation | Preserve context, constraints, and acceptance criteria for single-node plans and inspected graph handoffs; supervisors, memory, and checkpoints receive the original task. |

### Product and scale

Native distribution work includes Developer ID signing and notarization for
macOS downloads, plus native Wayland screensaver integration. The current
Linux port targets X11; the macOS bundle uses an ad-hoc signature.

1. **Deterministic deliverable contracts** — make every requested output part
   explicit in the graph and mechanically verify complete assembly, removing
   the residual stochastic failure where a terminal node returns only its own
   increment instead of the full deliverable
2. **Evidence-triggered control** — replace routine model supervision calls
   with deterministic stage and anomaly triggers; reserve LLM review for a
   result that supplies evidence the pending plan should change
3. **Reviewable continual learning** — promote successful graph structures,
   failure lessons, capability descriptions, and reusable procedures into
   versioned supplemental state with diffs and rollback, extending
   `PlannerMemory` and `distill_template` without mutating the base planner
4. **Native Agent Skills support** — discover the Agent Skills standard
   directly, add Python-backed executable skills, and keep the existing
   OpenClaw adapter as one inventory source rather than the only packaged path
5. **Integrate asset policy with Jobs v1** — manifest-native production brand
   masters, OCR and perceptual brand validators, select-from-N curation, and
   deterministic export bundles built from accepted attempt pointers
6. **Operator inspection surface** — render the Jobs event/attempt timeline,
   costs, prompts, responses, validation findings, and artifacts without
   requiring users to inspect SQLite or raw JSON
7. **Detachable operator runtime** — keep approved long jobs active when a
   terminal disconnects, then list, inspect, attach, stop, and resume them from
   the installed CLI
8. **Scale ladder** — offline 5,000-item stress tests followed by bounded paid
   50/250/1,000-item trials with kill-and-resume and duplicate detection
9. **Benchmarks, continued** — a discriminating judge, human calibration,
   repeated k=25 cells, repeated glyph live cells, and held-out
   brand-consistency comparisons; save delivered text and judge reasoning so
   independent reviewers can rescore each quality result
10. **Autotune generalization**: add campaign-wide leases, calibrated
   sample-size and repeated-comparison guidance, richer reports, process-
   supervised evaluator isolation, and conservative live-evaluator adapters
   after the offline campaign

## Later

- Autonomous Autotune proposal strategies after the bounded runner and
  evidence protocol are calibrated on more than the concurrency workload
- Recursive subgraph decomposition, with depth limits and shared
  budget/trace/failure machinery
- Direct messaging between retained agent runs with bounded family-scoped
  routing and persisted delivery receipts
- Human-in-the-loop approval gates (pause/approve/reject, state survives restart)
- Provider hardening: retry with backoff, streaming, response caching
- Template/starter library and a `smythe init` command
- Docs site, OpenTelemetry export, local-model capability metadata

## Out of scope (deliberately)

- A hosted SaaS or agent marketplace — the OSS core comes first
- Rewriting in Rust/Go — Python is the right language for this audience

## Contributing

Issues tagged `good first issue` are curated entry points; see [CONTRIBUTING.md](CONTRIBUTING.md). If you want to work on a roadmap item, open an issue first so we can align on design.
