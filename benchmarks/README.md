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

## Results — first self-baseline run (2026-07-05)

Model: `claude-opus-4-8` for all baselines and the judge. Command:
`python benchmarks/run_benchmarks.py --judge --out results/real_run.json`.
Full records (outputs, per-criterion scores, judge reasoning):
[results/real_run.json](results/real_run.json). Quality is the blind
judge's overall score, 1–10.

| Task | Single agent | Fixed pipeline | Smythe dynamic |
|---|---:|---:|---:|
| acquisition-diligence | 3 ($0.005) | 4 ($0.026) | **7** ($0.055) |
| code-review | **9** ($0.005) | **9** ($0.041) | 3 ($0.053) |
| competitive-analysis | 7 ($0.007) | 7 ($0.041) | 5 ($0.096) |
| product-launch | 6 ($0.005) | 7 ($0.041) | **8** ($0.058) |
| research-memo | 8 ($0.008) | 8 ($0.049) | 8 ($0.067) |
| **Mean** | **6.6** ($0.006) | **7.0** ($0.040) | **6.2** ($0.066) |

**The honest read: dynamic topology won 2, tied 1, and lost 2 — at
roughly 10× the cost of a single agent.** On this task set and harness,
the headline claim is not yet supported. The per-task story is more
useful than the mean:

- **Where it won.** Tasks whose structure matches a generated topology.
  On acquisition-diligence the dynamic graph's adversarial tier produced
  the only output the judge considered decision-grade (7 vs 3/4); on
  product-launch the fork-join decomposition beat both baselines.
- **Where it lost, and why.** Code-review collapsed (3 vs 9): the
  harness takes the *terminal node's* output as the deliverable, and
  after a five-node chain the final node hedged every bug as an
  unconfirmed hypothetical — the judge noted it "treats every actual,
  verifiable bug as an unconfirmed hypothetical." Competitive-analysis
  failed the same way: the final node *referenced* a comparison matrix
  from an upstream node without reproducing it. The failure mode is
  context dilution through the node chain, not bad planning — which
  makes synthesis/terminal-output handling the next engineering target,
  and is exactly the kind of thing this harness exists to surface.

**Caveats, all real:** one run per cell (no variance estimate); one
judge, and the judge is the same model that produced the outputs;
the acquisition-diligence task supplies no source documents, so every
baseline is partly scored on how it handles a data vacuum. Treat these
as a first calibration point, not a verdict.

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
