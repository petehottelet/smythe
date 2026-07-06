# Smythe benchmarks

The claim under test: **a generated, task-specific execution graph
outperforms both a single agent and a fixed pipeline** — often enough,
and by enough, to justify the planning call. This harness exists to
measure that honestly, including where the claim fails.

> **Status: instrument built, numbers pending.** The harness runs
> end-to-end offline in CI (mechanics verified, deterministic, zero
> cost). Quality numbers require real-provider runs and will be
> committed to [results/](results/) with the exact command, model, and
> date. Until then, no quality claims.

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
