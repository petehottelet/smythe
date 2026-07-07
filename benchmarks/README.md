# Smythe benchmarks

The claim under test: **a generated, task-specific execution graph
outperforms both a single agent and a fixed pipeline** — often enough,
and by enough, to justify the planning call. This harness exists to
measure that honestly, including where the claim fails.

> **Status: first self-baseline numbers published** (see Results below).
> The harness also runs end-to-end offline in CI (mechanics verified,
> deterministic, zero cost). Head-to-head framework comparisons and
> memory-on/off numbers are still pending.

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

### Current standings (v4)

| Task | Single agent | Fixed pipeline | Smythe dynamic |
|---|---:|---:|---:|
| acquisition-diligence | 8.3 [8–9] ($0.007) | **9.0** ($0.057) | **9.0** ($0.098) |
| code-review | 8.7 [8–9] ($0.005) | **9.0** ($0.047) | **9.0** ($0.073) |
| competitive-analysis | 6.0 [6–6] ($0.006) | **8.0** ($0.043) | 7.7 ($0.159) |
| product-launch | 6.0 [6–6] ($0.006) | **8.0** ($0.045) | 7.7 ($0.076) |
| research-memo | 7.7 [7–8] ($0.008) | **8.0** ($0.033) | 7.7 ($0.072) |
| **Mean** | **7.3** ($0.006) | **8.4** ($0.045) | **8.2** ($0.096) |

**The honest read.** After the two fixes, dynamic topology went from
far-worst (4.7) to statistical parity with a well-built fixed pipeline
(8.2 vs 8.4) — at roughly twice the fixed pipeline's cost and 15× the
single agent's. The headline claim — that *generated* topology beats a
*hardcoded* one — is not yet demonstrated on this task set: the fixed
pipeline is a strong baseline and still edges the mean. What is
demonstrated is the loop this harness exists for: losses were
mechanistically diagnosed, fixed in the framework, and the recovery
measured (code-review 1.7 → 9.0). Where dynamic still trails, the
graphs are wider and costlier (competitive-analysis runs ~7 nodes for
a one-artifact deliverable) — topology-size calibration is the next
lever.

**Caveats:** three runs per cell is a variance hint, not statistics;
one judge, same vendor as the executor; tasks were authored by this
project. Memory-on vs memory-off and framework head-to-heads remain
open (see Planned).

## Methodology commitments

1. **Same model everywhere.** No baseline gets a better model.
2. **Blind judging.** The judge never sees which system wrote the output.
3. **Losses get published.** If the fixed pipeline beats dynamic
   topology on a task class, that row ships in the table.
4. **Reproducible.** Committed results include the command, model,
   provider, and date; offline mechanics are byte-deterministic.

## Planned

- Real-provider runs across ~5 tasks, results committed
- Memory-on vs. memory-off numbers for the learning Architect
  (`PlannerMemory`) — the README's outstanding promise
- Head-to-head vs. LangGraph and CrewAI equivalents of the fixed
  pipeline, once the internal numbers exist
