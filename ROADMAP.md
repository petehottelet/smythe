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

## Shipped (v0.7.0): trust at production fan-out

Version 0.7.0 is available on [PyPI](https://pypi.org/project/smythe/0.7.0/)
and [GitHub](https://github.com/petehottelet/smythe/releases/tag/v0.7.0).
It adds the following runtime guarantees and measured evidence:

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
   ports with layered green trails using 56 reference and 192 original SVG
   shapes with a 10% original mix. Native ports are distributed as source.
   Historical rendering checks cover Windows, Apple Silicon, Intel Mac, and
   Ubuntu 22.04/24.04; their checksums and receipts remain archived
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
12. **Original SVG workflow and web explorer** — independently authored filled
    contours, measured style gates, duplicate detection, and hash-bound contact
    sheets. The web preview now adapts the MIT-licensed m8e/Rezmason renderer
    and 56 visible base glyphs plus their blank slot, with a 10% original-glyph
    mix, Classic/3D/Operator presets, Matrix green rain, simple VT323 pixel
    controls, and a Trajan Bold outline logo on black. Browser interaction checks
    pass. The [six-session headless study](benchmarks/renderer_performance_20260907_results.md)
    and ten-minute stability check are complete; visible presentation and
    quantified reference parity remain pending.
    The [workflow protocol](benchmarks/svg_glyph_benchmark.md) measures fresh
    generation, validation, and assembly across repeated thread/process runs;
    browser frame timing is measured separately.
13. **Strict cost and usage validation** — reject malformed policy, estimates,
    provider usage, and restored charges before ledger mutation. Invalid
    accounting stops new work and blocks resume until reconciliation.
    [Cost guardrails](docs/budgets.md).
14. **Consistent terminal failures** — serial halt and exhausted retries stop
    later dispatch immediately, preserve the original exception, and retain
    completed work. [Execution policies](docs/execution.md).
15. **Durable verification decisions** — persist pending verdicts and
    regeneration intents, settle active descendants before resetting their
    generation, and resume unfinished control transitions before new calls.
    [Verification](docs/verifier.md).
16. **Complete task propagation** — carry detached goal, constraints, source
    context, and acceptance criteria through routing, planning, graph handoffs,
    execution, supervision, synthesis, memory, and recovery.
    [Tasks and handoffs](docs/tasks.md).
17. **Native Astra and Sol Responses** — explicit endpoint, reasoning effort,
    and service tier; preserved function-tool continuation; exact native token
    pricing; request-bound quotes; and retained billing evidence for unusable
    responses. [Provider guide](docs/openai-responses.md).
18. **Durable text-workflow accounting** — one exact ledger across every
    supported phase, request-bound reservations, fenced dispatch, saved native
    responses, and atomic graph/control consumption. Separate planning and
    execution retain their charges; recovery replays persisted responses locally.
    [Workflow guide](docs/workflow-accounting.md).
19. **Jobs operator inspection** — read-only run lists and paged inspection,
    exact cost balances, prompts, responses, attempt lineage, recent events,
    and local artifact integrity in a self-contained black-and-white HTML report.
    [Inspection commands](docs/jobs.md#list-and-inspect-runs).
20. **Deep graph execution** — iterative validation, cycle checks, dependency
    ordering, and depth calculation preserve existing traversal order. Offline
    regressions execute a reverse-ordered 5,000-node chain and verify deep
    revisions before mutation. [Execution guide](docs/execution.md#deep-graphs).
21. **Saved graph limits** — bind node count, effective execution models,
    retries, and regeneration to the durable workflow recipe. Plan replay,
    handoffs, recovery, and revisions enforce the same limits.
    [Graph policy guide](docs/workflow-accounting.md#freeze-graph-limits).
22. **Jobs ownership fencing** — bind every worker mutation to a live lease
    epoch so expired workers cannot alter recovered journal state. Schema
    upgrades preserve historical attempt provenance and reject live legacy
    leases. [Upgrade and ownership scope](docs/jobs.md#jobs-database-upgrades).
23. **Detached operator runtime** — keep approved work running after its
    launcher exits on supported hosts, follow it through read-only attachment, and request a durable
    drain/pause. Generation-bound resume preserves later stop requests.
    [Operator commands](docs/jobs.md#detached-execution-and-attachment).
24. **Artifact directory ownership** — persistent namespaces separate custom
    run IDs and databases sharing an output root. Directory claims precede
    dispatch; exclusive publication preserves existing bytes. Migrated runs
    retain their receipt paths. [Journal and artifacts](docs/jobs.md#the-dispatch-journal).
25. **Atomic file checkpoints** — independent save attempts use exclusive
    temporary files, flush complete snapshots, and atomically replace the
    checkpoint. Failure cleanup preserves other writers' files.
    [Persistence and ownership scope](docs/checkpoint-format.md).
26. **Observed recovery at 5,000 operations** — one offline schema-v3 campaign
    retained accepted artifacts after a hard kill, completed pending work, and
    rerolled eight explicitly acknowledged unknown outcomes. All 5,000 files,
    call identities, and zero-cost balances pass independent reconciliation.
    [Scope and retained evidence](benchmarks/jobs_scale_5000_20260907_results.md).

## Implemented after 0.7.0 (Unreleased)

- **Planner history validation** — skip malformed recalled fields without
  rewriting history or changing valid ranking. Prompts identify summed node
  time explicitly. [Learning loop](docs/architecture.md#learning-loop).
- **Autotune campaign ownership** — one live owner and epoch govern trial
  mutations and decisions. Heartbeats survive event-loop statistical work;
  cancellation drains evaluators before release. Transactional v3-to-v4
  migration preserves evidence and blocks already-open legacy writers.
  Explicit `lease=` tokens are a new low-level API requirement.
  [Ownership and migration scope](docs/optimize.md#campaign-ownership-unreleased).
- **Concurrent journal initialization** — bounded WAL retries and atomic
  schema creation prevent competing openers from seeing partial stores.
  Existing evidence and identities remain intact.
  [Implementation and verification plan](plans/sqlite-initialization-2026-09-07.md).

## Coming soon

**Astra status — 8 September 2026:** live benchmarks have not run. The
[79 offline pilot checks](benchmarks/results/astra_pilot_runtime_20260907/README.md)
validate software contracts, not model cost, speed, or quality. An agreed API
spending ceiling is needed before starting the 12-workflow pilot.

The [GPT-6 Astra campaign plan](benchmarks/astra_benchmark_plan.md) specifies
matched model and orchestration experiments. Native Responses and complete
text-workflow accounting are implemented. The offline preparation package
provides 13 task/source packs and balanced schedules for 12 pilot and 200 main
workflows. The [pilot runner](benchmarks/astra_runtime.md) binds the spending
envelope, source identity, native ledger, and recovery of each calibration
trial. Paid comparisons require the authorized campaign allowance, accounted
judge, completed pilot, frozen acceptance gates, and reviewed result records.

The [September repository review](docs/project-review-2026-09-06.md) defines
the hardening work ahead of broader production claims:

| Priority | Change | Acceptance criteria |
|---|---|---|
| P1 | Complete-workflow cost evidence | Publish fresh cost comparisons from the durable text ledger, with native usage, declared prices, failed attempts, and all optimization trials retained. |
| P2 | Web renderer parity and presentation | Follow the [completed headless study](benchmarks/renderer_performance_20260907_results.md) with measured visible-display cadence, GPU timing, and quantified reference tolerances. All six primary sessions missed their frozen pacing target; the separate blank-page control showed similar callback cadence. Archived Canvas v1 timings remain superseded diagnostics. |

### Product and scale

**Windows Defender review:** resolve the v0.7.0 `SmytheGlyphRain.scr`
detection before restoring precompiled Windows downloads. Keep quarantined
copies quarantined, obtain Microsoft analysis, investigate build provenance,
and publish the review outcome with exact artifact identities. The cause
is unresolved; [status and evidence](docs/windows-defender-2026-09-07.md).

The Windows, macOS, and Linux source ports contain the licensed base catalog
and current original SVGs. Earlier compiled catalog, rendering, and host
checks remain archived; precompiled distribution is suspended.
Next, the [Glyph Rain design plan](docs/glyph-rain-plan.md) brings the web exposure
pipeline, 3D navigation, and settings into native exploration modes. Preserve
normal screensaver input dismissal and verify each compiled control. The
original-glyph generation benchmark remains separate from renderer changes.

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
6. **Scale ladder** — extend the reviewed offline 5,000-operation recovery
   observation to bounded paid 50/250/1,000-item trials with kill-and-resume
   and duplicate detection; measure the current schema-v4 runtime separately
7. **Benchmarks, continued** — a discriminating judge, human calibration,
   repeated k=25 cells, repeated glyph live cells, and held-out
   brand-consistency comparisons; save delivered text and judge reasoning so
   independent reviewers can rescore each quality result
8. **Autotune generalization**: add calibrated sample-size and
   repeated-comparison guidance, richer reports, process-
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
