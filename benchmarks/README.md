# Smythe benchmarks

The claim under test: **a generated, task-specific execution graph
outperforms both a single agent and a fixed pipeline** — often enough,
and by enough, to justify the planning call. This harness exists to
measure that honestly, including where the claim fails.

> **Status: self-baselines, memory-on/off, image-pipeline, and framework
> head-to-head results are published below.** The core harness also runs
> end-to-end offline in CI (mechanics verified, deterministic, zero cost).

## The three systems

Every task runs through all three, with the same provider and model:

| Baseline | What it is | What it represents |
|---|---|---|
| `single_agent` | One node, the whole goal | A bare LLM call with a good prompt |
| `fixed_pipeline` | Research → analyze → write, serial, identical for every task | A hardcoded pipeline framework workflow |
| `smythe_dynamic` | The `LLMArchitect` designs a task-specific graph | Smythe's generated topology |

The fixed pipeline is deliberately reasonable — a strawman baseline
would make the comparison worthless.

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

Task definitions live in [tasks/](tasks/) as YAML: a goal, constraints,
and the rubric the judge scores against. Current set is small and will
grow to ~5 tasks spanning analysis, research, and review work. Adding a
task is a good first contribution — copy an existing YAML.

## Results — self-baselines and the fix loop (2026-07-06)

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

### Current standings (v5)

| Task | Single agent | Fixed pipeline | Smythe dynamic |
|---|---:|---:|---:|
| acquisition-diligence | 8.7 [8–9] ($0.007) | **9.0** ($0.059) | 8.7 ($0.096) |
| code-review | **9.0** ($0.005) | **9.0** ($0.046) | 8.3 [7–9] ($0.040) |
| competitive-analysis | 6.7 [6–7] ($0.006) | **8.0** ($0.046) | 7.3 [6–8] ($0.122) |
| product-launch | 6.0 [6–6] ($0.005) | **8.0** ($0.046) | 7.7 [7–8] ($0.079) |
| research-memo | 7.3 [7–8] ($0.009) | **8.0** ($0.033) | **8.0** ($0.073) |
| **Mean** | **7.5** ($0.006) | **8.6** ($0.046) | **8.0** ($0.082) |

**The honest read.** After the two fixes and the calibration, dynamic
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

**Result: null.** Quality 6.9 (memory on) vs 7.0 (off) on the recalled
positions; node counts and cost equal. The interpretation matters:
recalled history can only help where there is a planning mistake to
correct, and after the v5 calibration the Architect already plans this
family consistently (3 nodes, every time). The mechanism demonstrably
*can* change plans — [examples/08_learning_loop.py](../examples/08_learning_loop.py)
shows a recorded failure steering the next plan — but on a
well-calibrated planner and a homogeneous task family, it has nothing
to fix. A harder test (task families with planted failure modes)
is the follow-up.

**Caveats:** three runs per cell is a variance hint, not statistics;
one judge, same vendor as the executor; tasks were authored by this
project. The later framework head-to-head uses a cross-vendor judge,
but its protocol differs and its table is not directly comparable.

## Framework head-to-head: LangGraph and CrewAI (2026-07-12)

The long-promised comparison ([run_framework_h2h.py](run_framework_h2h.py),
raw records in [results/framework_h2h.json](results/framework_h2h.json)):
the **same semantic fixed pipeline** — shared step goals and personas
([harness.PIPELINE_SPECS](harness.py)) — implemented idiomatically in
each framework, on the same 5 tasks, same executor model
(`gpt-5.4-mini`), 3 reps, **judged blind by a different vendor**
(Gemini), which addresses the self-preference caveat in the v5 results
below. 60/60 runs completed, zero framework errors.

This is an **ecological framework comparison**, not a byte-identical
prompt microbenchmark. Each implementation packages dependency context,
messages, and framework scaffolding through its native APIs. Those
differences are part of the measured end-to-end systems, but they prevent
attributing every token or latency delta to scheduler overhead alone.

| System | Quality (blind) | Tokens | Wall |
|---|---:|---:|---:|
| smythe (fixed pipeline) | 9.67 [8–10] | **8,372** | **29.3s** |
| LangGraph (fixed) | 9.40 [8–10] | 8,782 | 30.8s |
| CrewAI (fixed) | 9.87 [9–10] | 39,696 | 45.6s |
| smythe (dynamic) | 9.27 [5–10] | 14,584 | 37.4s |

**The honest read.**
- **Smythe's observed end-to-end footprint is competitive with LangGraph** —
  within 5% on tokens and wall time for this shared semantic pipeline.
  Because framework-native prompt packaging differs, this is not an
  isolated measurement of orchestration overhead.
- **CrewAI consumed 4.7× the tokens** (its agent scaffolding — roles,
  backstories, internal formatting — is baked into every call) and 56%
  more wall time, for +0.2 quality that sits inside a known confound:
  longer outputs tend to score higher with LLM judges, and CrewAI's
  outputs were the longest.
- **Quality is ceiling-compressed** (everything 9–10 except one
  outlier): the Gemini judge is lenient despite strict instructions, so
  this table discriminates *efficiency* well and *quality* weakly.
- **Dynamic topology still doesn't beat the fixed pipeline on
  homogeneous text tasks** — consistent with the v5 finding, now
  replicated under a different executor and an independent-vendor
  judge. Its [5–10] range includes one bad generated plan; planning
  variance is the cost of generated topology. (Where dynamic *does*
  win — parallel image workloads, 6.6–14× wall-clock — is measured in
  [image_benchmarks.md](image_benchmarks.md).)
- Caveats: different executor model than the v5 self-baselines (tables
  are not directly comparable); n=3 per cell; token counts come from
  each framework's own accounting (sources recorded per run).

## Methodology commitments

1. **Same model everywhere.** No baseline gets a better model.
2. **Blind judging.** The judge never sees which system wrote the output.
3. **Losses get published.** If the fixed pipeline beats dynamic
   topology on a task class, that row ships in the table.
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
honest caveats (including an observed near-duplicate pair):
[image_benchmarks.md](image_benchmarks.md).

## Planned

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
