# Smythe benchmark evidence

**Smythe used 77% fewer mean tokens and 28% less mean wall time than CrewAI**
on the same fixed three-stage semantic pipeline. Against LangGraph it used
8% fewer tokens and 6% less wall time. The generated-topology task-shape
suite recorded **14% lower end-to-end wall time** than its fixed pipeline,
with observed quality in the same measured band.

Every public number links to harness source and a committed raw record. Offline
mechanics run in CI with deterministic providers and zero API cost.

## Evidence status

| Campaign | Status | Current result |
|---|---|---|
| [Original SVG workflow](svg_glyph_benchmark.md) | **Claimable** | 192 original glyphs in 4.03 s median including full validation and assembly; 2.95× process-c1 speed; all 30 workflows accepted |
| [Task-shape suite v3](shape_suite.md) | **Claimable for wall time and observed quality** | 14% lower end-to-end wall time; historical cost excludes planning |
| [Hard-kill durability v2](durability_benchmark.md) | **Claimable** | 8 duplicate dispatches after resume versus LangGraph's 32, across 3 reps |
| [Glyph Rain width-scaling sweep](glyph_screensaver_benchmark.md#width-scaling-from-64-to-256-nodes) | **Claimable** | 64, 128, 192, and 256 valid unique tiles at every concurrency; 40.37×–56.21× at concurrency 64; isolated outputs |
| [Image concurrency sweep](image_benchmarks.md) | **Claimable** | 6.6× wall-clock speedup at concurrency 8; 72/72 valid images |
| [Corrected framework head-to-head](#corrected-framework-head-to-head-langgraph-and-crewai-2026-07-12) | **Claimable for the fixed arms** | 77% fewer tokens and 28% less wall time than CrewAI; highest observed blind score |
| Original self-baselines and pre-correction framework record | Diagnostic | Preserved because they found payload, assembly, and measurement defects; superseded by corrected campaigns |
| [Control ablation](control_ablation.md) | Mechanism scope | Objective gates remain valuable; routine LLM supervision and judged-prose gating are not default quality paths |

The authoritative deliverable is `Swarm.execute(...).output`, which is what a
caller receives. The harness records the historical terminal-node join beside
it for diagnosis, but no current headline uses that older measurement.

## The three systems

Every task runs through all three, with the same provider and model:

| Baseline | What it is | What it represents |
|---|---|---|
| `single_agent` | One node, the whole goal | A bare LLM call with a good prompt |
| `fixed_pipeline` | Research → analyze → write, serial, identical for every task | A hardcoded pipeline framework workflow |
| `smythe_dynamic` | The `LLMArchitect` designs a task-specific graph | Smythe's generated topology |

Each fixed-pipeline implementation uses the same semantic stage goals and
personas through its native framework API.

## Metrics

Per (task, baseline) run: topology, node count, graph depth, cost
(USD), wall time (real mode only — offline wall time measures the OS
scheduler, not the work), and — with `--judge` — blind rubric scoring:
an LLM judge scores each output 1–10 per rubric criterion without
knowing which system produced it.

## Running

```bash
python benchmarks/run_benchmarks.py                       # offline, mechanics only
python benchmarks/run_benchmarks.py --out results/run.json
python benchmarks/run_benchmarks.py --judge               # real mode + quality scores
python benchmarks/run_benchmarks.py --task research-memo --baseline smythe_dynamic
```

Offline mode is automatic when no provider key is set (or forced with
`--offline`). A CI test re-runs the offline suite and diffs it against
the committed [results/offline_sample.json](results/offline_sample.json).

## Tasks

Five task definitions live in [tasks/](tasks/) as YAML: a goal, constraints,
and the rubric the judge scores against. They cover diligence, code review,
competitive analysis, product launches, and research memos.

## Diagnostic campaigns — self-baselines and the fix loop (2026-07-06)

Protocol: 3 runs per cell; executor `claude-opus-4-8` for every
baseline; judge `claude-sonnet-5` (a different model than the
executor, to reduce self-preference). Quality is the blind judge's
overall score (1–10), reported as mean [min–max]. Full records —
outputs, per-criterion scores, judge reasoning — are committed under
[results/](results/) as `v2_*`, `v3_*`, and `v4_*` (an earlier n=1,
self-judged pilot is `real_run.json`).

### The fix loop

The benchmark's first job was finding out why dynamic topology lost.
It found two executor bugs, each fixed in one commit, each fix
measured before the next:

| Dynamic-topology quality | v2: no fix | v3: roots get the task | v4: every node gets the task |
|---|---:|---:|---:|
| acquisition-diligence | 2.0 [1–3] | 8.3 [8–9] | **9.0 [9–9]** |
| code-review | 1.7 [1–2] | 4.3 [3–5] | **9.0 [9–9]** |
| competitive-analysis | 6.0 [5–7] | 5.3 [3–7] | 7.7 [7–8] |
| product-launch | 6.0 [6–6] | 6.7 [5–8] | 7.7 [7–8] |
| research-memo | 7.7 [7–8] | 7.0 [6–8] | 7.7 [7–8] |
| **Mean** | **4.7** | **6.3** | **8.2** |

- **Bug 1 (v2 → v3): the task payload never reached the nodes.** The
  Architect read the full task when planning, but node execution saw
  only the generated one-line labels — so on code-review no specialist
  ever saw the code, and on diligence no analyst ever saw the source
  documents. Judges described reviews that "treat every actual,
  verifiable bug as an unconfirmed hypothetical." Fix: `Swarm.plan()`
  stamps the task goal and constraints into root nodes.
- **Bug 2 (v3 → v4): verifiers couldn't see the artifact either.**
  With only roots fixed, the red-team and memo nodes received *claims
  about* an artifact they couldn't see, and hedged everything — one
  run demanded the source "be produced before sign-off" while it sat
  in the task. Dependency results carry analyses of the artifact, not
  the artifact. Fix: every node gets the task context.

### Topology right-sizing (v4 → v5)

v4 left dynamic graphs over-built: 5–7 nodes for single-deliverable
tasks, at 2× the fixed pipeline's cost. v5 adds cost-aware node-count
guidance to the planning prompt ("every node costs money and latency;
add one only when it contributes a distinct work product"). The
Architect responded selectively — code-review dropped from a 5-node
fork-join to a 3-node `serial → adversarial` (−45% cost), while
competitive-analysis kept its 6–7-node fan-out where it judges the
parallel branches genuinely independent. Mean dynamic cost fell ~14%
with quality flat within noise (8.2 → 8.0).

### Historical standings (v5; diagnostic)

| Task | Single agent | Fixed pipeline | Smythe dynamic |
|---|---:|---:|---:|
| acquisition-diligence | 8.7 [8–9] ($0.007) | **9.0** ($0.059) | 8.7 ($0.096) |
| code-review | **9.0** ($0.005) | **9.0** ($0.046) | 8.3 [7–9] ($0.040) |
| competitive-analysis | 6.7 [6–7] ($0.006) | **8.0** ($0.046) | 7.3 [6–8] ($0.122) |
| product-launch | 6.0 [6–6] ($0.005) | **8.0** ($0.046) | 7.7 [7–8] ($0.079) |
| research-memo | 7.3 [7–8] ($0.009) | **8.0** ($0.033) | **8.0** ($0.073) |
| **Mean** | **7.5** ($0.006) | **8.6** ($0.046) | **8.0** ($0.082) |

**Recorded outcome.** After the two fixes and the calibration, dynamic
topology sits near parity with a well-built fixed pipeline (8.0 vs
8.6) at ~1.8× its cost — up from far-worst (4.7) when this campaign
started. The headline claim — that *generated* topology beats a
*hardcoded* one — is not demonstrated on this task set: the fixed
pipeline is a strong baseline and still edges every column. What is
demonstrated is the loop this harness exists for: losses were
mechanistically diagnosed, fixed in the framework, and each fix
measured (code-review 1.7 → 4.3 → 9.0 across the payload fixes; −45%
cost from calibration). Run-to-run variance between v4 and v5 (±0.3
on several cells) is visible in the ranges — treat single-cell deltas
smaller than that as noise.

### Memory on/off

The learning-Architect claim, measured
([run_memory_ab.py](run_memory_ab.py), records in
[results/memory_ab.json](results/memory_ab.json)): five similar
competitive-brief tasks run in sequence, with and without
`PlannerMemory`, two repeats. Recall wiring is observable per record —
with memory on, positions 2–5 provably received prior outcomes in
their planning prompts; without, never.

**Recorded scope.** Quality was 6.9 (memory on) vs 7.0 (off) on the recalled
positions; node counts and cost equal. The interpretation matters:
recalled history can only help where there is a planning mistake to
correct, and after the v5 calibration the Architect already plans this
family consistently (3 nodes, every time). The mechanism demonstrably
*can* change plans — [examples/08_learning_loop.py](../examples/08_learning_loop.py)
shows a recorded failure steering the next plan — but on a
well-calibrated planner and a homogeneous task family, it has nothing
to fix. A harder test (task families with planted failure modes)
is the follow-up.

**Measurement scope:** three runs per cell is a variance hint, not statistics;
one judge, same vendor as the executor; tasks were authored by this
project. The later framework head-to-head uses a cross-vendor judge,
but its protocol differs and its table is not directly comparable.

## Corrected framework head-to-head: LangGraph and CrewAI (2026-07-12)

The corrected comparison ([run_framework_h2h.py](run_framework_h2h.py), raw
records in
[results/framework_h2h_rightsized.json](results/framework_h2h_rightsized.json))
judges each framework's delivered API output. The fixed implementations share
the same five tasks, semantic step goals, personas, three-stage pipeline, and
`gpt-5.4-mini` executor. Gemini judges outputs blind from a different vendor.
Each implementation runs through its native framework API. All 45 fixed-arm
runs completed without error.

| Fixed implementation | Blind quality / 10 | Mean tokens | Mean wall |
|---|---:|---:|---:|
| **Smythe** | **9.73 [9–10]** | **8,796** | **30.53s** |
| LangGraph | 9.53 [7–10] | 9,590 | 32.50s |
| CrewAI | 9.53 [7–10] | 38,427 | 42.48s |

Smythe recorded the highest observed blind score. It used 8% fewer mean tokens and 6%
less mean wall time than LangGraph, plus **77% lower mean token load** and 28%
less mean wall time than CrewAI. The chart and callouts in the project README
are rendered directly from this corrected record.

This is an ecological end-to-end comparison: task semantics and the executor
model are matched, while dependency context, message structure, and accounting
flow through each framework's native implementation. The result measures the
systems developers actually run.

Token provenance was checked in the September 2026 audit. For the fixed
Smythe arm, the provider's returned input and output tokens were multiplied
by the Sentinel's constant $0.000003/token and divided by that same constant
in the harness. This recovers the recorded provider token total; the fixed
architect makes no provider calls. LangGraph and CrewAI expose native usage
totals. The comparison establishes token load, not invoice savings. The
0.20-point observed score lead does not establish superior quality.

The optional dynamic arm's historical token count excludes planning and is
not a claimable total-workflow token comparison. New runs record usage
directly at the provider boundary, with per-call receipts including planning,
execution, and synthesis. The original record is preserved unchanged.

### Superseded original record

[results/framework_h2h.json](results/framework_h2h.json) used the historical
terminal-node-only Smythe measurement. It remains committed as diagnostic
evidence. The corrected record above measures `Swarm.execute(...).output`, the
same delivered-output boundary used for the LangGraph and CrewAI arms.

## Methodology commitments

1. **Same model everywhere.** No baseline gets a better model.
2. **Blind judging.** The judge never sees which system wrote the output.
3. **Every measured row remains available.** Diagnostic and superseded
   campaigns stay committed with their evidence status.
4. **Auditable and repeatable.** Harness source and raw records are committed;
   the model and protocol are stated alongside each table, and offline
   mechanics are deterministic. New result records also capture installed
   dependency versions. Paid model outputs remain stochastic, so reproducing
   the protocol does not promise identical scores or timings.

Install the complete optional harness environment from a fresh checkout:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,benchmarks]"
python benchmarks/run_benchmarks.py --offline
python benchmarks/run_image_benchmarks.py
```

Offline commands consume no API credits. Paid commands require an explicit
`--live` flag where supported; review the printed estimate before continuing.
The `benchmarks` extra installs provider SDKs, Pillow, LangGraph, and CrewAI.
Future result records use repo-relative POSIX artifact paths when outputs live
inside the checkout; historical records may retain absolute producer paths.

## Image pipeline — concurrency sweep (2026-07-12)

First results with entirely objective metrics (no LLM judge): **6.6×
wall-clock speedup at concurrency 8** (46.3s → 7.0s for 8 images) at
identical cost, 81–88% of ideal parallel efficiency, 72/72 images valid,
zero rate-limit events on one paid key. Full table, protocol, and
measurement scope (including an observed near-duplicate pair):
[image_benchmarks.md](image_benchmarks.md).

## Original SVG workflow

The [SVG campaign](svg_glyph_benchmark.md) measures 192 fresh filled-contour
glyphs through Smythe, complete numeric and silhouette validation, and a
delivered catalog. Thread and process backends use five concurrency settings
and three repetitions each. Every configuration remains in the record;
the headline selects the lowest median complete workflow time.

[Full contact sheet](partitions/glyph_svg_v1/catalog/contact-sheet.png) ·
[Raw campaign](results/glyph_svg_v1.json) ·
[Navigable web explorer](../screensaver/svg-preview/README.md).

This is local procedural generation. It has no simulated delay and makes no
provider API calls. Reference research and design calibration happen before
the campaign and are recorded separately. Browser rendering uses its own
timing and navigation protocol.

## Glyph screensaver fan-out

The [glyph screensaver workload](glyph_screensaver_benchmark.md) turns wide
artifact generation into something directly inspectable: one independent node
per original fictional cyber glyph, 192 calls total, followed by objective PNG
normalization and SHA-256 uniqueness checks.

```bash
python benchmarks/run_glyph_screensaver.py
```

The default lane is deterministic, local, and costs nothing. It sweeps
concurrency 1, 4, 8, and 16 with controlled asynchronous latency, records
generation and end-to-end timing separately, and assembles the fastest valid
192-tile run into:

- 192 normalized 128×128 PNG tiles;
- a 2048×1536 contact-sheet atlas;
- a 1920×1080 green digital-rain preview;
- a compact 640×360 looping GIF; and
- a self-contained animated 1920×1080 HTML canvas.

The marks come from an original deterministic stroke grammar (bars, stems,
hooks, enclosures, press diagonals, bowls, and diacritic dots on an ideograph
grid) rather than any copied font, logo, screenshot, or reference pixels.
Every output has a dimensions, frame, byte-size, and SHA-256 receipt. A
published realistic-latency profile re-runs the sweep at the live image lane's
measured 5.8 s per-call latency across concurrency 1–64. The optional GPT
Image lane executes one chosen concurrency and refuses to start without an API
key, explicit inclusive per-call ceiling, and a whole-run budget large enough
for every call:

```bash
python benchmarks/run_glyph_screensaver.py --live --concurrency 8 \
  --max-cost-per-call-usd 0.01 --max-budget-usd 1.92
```

Those values are examples of the guardrail shape, not current pricing advice.
Verify provider pricing immediately before any paid run. The offline sweep is
an executor benchmark; only a repeated live lane can support claims about an
external image API's latency or rate limits.

The matched [width-scaling records](glyph_screensaver_benchmark.md#width-scaling-from-64-to-256-nodes)
cover 64, 128, 192, and 256 nodes in separate output namespaces. Every width
produced a complete set of valid, SHA-256-unique tiles at every measured
concurrency; speedup at concurrency 64 ranged from 40.37× to 56.21×.

### Results — realistic-latency profile (2026-08-05)

One 192-node broadcast graph, simulated 5.8 s per-call latency (the live
image lane's measured serial mean), all 192 tiles valid and SHA-256-unique
at every concurrency
([record](results/glyph_screensaver_offline_realistic.json)):

| Concurrency | Wall | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 1,149.6 s | 1.0× | — |
| 4 | 285.7 s | 4.02× | 100% |
| 8 | 144.2 s | 7.97× | 100% |
| 16 | 74.1 s | 15.5× | 97% |
| 32 | 40.6 s | 28.3× | 88% |
| 64 | 20.5 s | 56.2× | 88% |

The default 250 ms profile
([record](results/glyph_screensaver_offline.json)) plateaus near 7× because
fsync'd per-tile artifact journaling (~70 ms/tile on the reference
Windows/NTFS machine) dominates its short latency envelope — that floor is
documented in the protocol rather than hidden by construction.

## Hard-kill durability

The [durability benchmark](durability_benchmark.md) runs an entirely offline
wide fan-out, terminates each worker process without cleanup, and measures
operation IDs dispatched again after restart. Dispatch and completion are
durably recorded as separate events, so in-flight exposure is not hidden by a
completion-time log. The 2026-08-05 v2 record re-establishes the framework
comparison under the conservative accounting: across three reps, Smythe
re-dispatched exactly one in-flight wave (8 of 64 calls, resume 4.6–6.0 s)
while LangGraph's superstep checkpointing re-dispatched all 32 previously
dispatched operations (resume 16–19 s), including the requests still in flight
when the process was killed, with per-node fan-out overhead near parity and
published as measured. This is a durability microbenchmark, not provider
invoice evidence or a universal framework claim.

```bash
python benchmarks/run_durability_benchmark.py --quick
```

## Control features: supervision and verification (2026-07-29)

Do the control tiers earn their cost? Four arms, two campaigns, 120 live
runs. Short answer: on judged prose, no. The supervisor was consulted 94
times across 30 runs and proposed a change **zero** times; gating fired
in 4 of 30 runs, used 34% more recorded execution cost, and left the number of bad runs
unchanged. Neither moved the floor, which was the pre-registered claim.

Historical control USD totals exclude planning and supervisor reviews;
they cannot establish whole-workflow cost.

Full writeup, including the two invalidated arms and why they are
published anyway: [control_ablation.md](control_ablation.md).

## Coming soon

- [GPT-6 Astra campaign](astra_benchmark_plan.md): separate model capability
  from orchestration effects, retain native usage and every attempted run,
  and compare blind quality against complete-workflow cost and time
- Full-workflow task-shape cost campaign using the corrected provider-call
  recorder, with planning usage included and input/output pricing separated
- Repeated, randomized campaign order and a larger external task set for
  quality and latency comparisons

- More reps on the `criteria` arm of the control ablation — stating
  acceptance criteria without enforcing them was the best-scoring arm
  (9.67, 1/15 floor runs) but has half the samples of the others
- Deterministic deliverable assembly, which is the failure mode that
  actually causes catastrophic runs and that none of the control
  features catch
- A memory task family with planted failure modes, so recall has
  mistakes to correct (the current family measures null — see above)
- Framework head-to-head re-run on the original v5 protocol
  (claude-opus-4-8 executor) so the tables become directly comparable
- A judge with better score discrimination (the current independent
  judge is ceiling-compressed at 9–10)
- Image pipeline continuations — select-from-N quality-per-dollar,
  repeated k=25 runs, and shared-brief vs. serial-style vs. single-agent
  brand-consistency comparisons (see
  [image_benchmarks.md](image_benchmarks.md))
