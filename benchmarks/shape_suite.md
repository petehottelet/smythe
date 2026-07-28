# The shape suite — and a measurement bug in our own harness

Every smythe benchmark published before this one used a task set whose
members all fit a single shape: research → analyse → write. On such a
set a *generated* topology cannot beat a *fixed* one — there is nothing
to adapt to, and the planning call is pure overhead. The published
near-parity result was therefore partly a fact about the task set.

This suite ([tasks_shapes/](tasks_shapes/),
[run_shape_suite.py](run_shape_suite.py)) uses five tasks whose natural
shapes deliberately differ: a one-step transform, four independent
profiles, a deep arithmetic chain, a case-then-attack, and a
three-defect audit.

Protocol: executor `gpt-5.4-mini` for every baseline, blind judging by
`gemini-pro-latest` (different vendor), 2 reps, 30 runs, zero errors.
Raw records: [results/shape_suite.json](results/shape_suite.json).

## The measurement bug (found first, and it matters)

The first run scored dynamic topology **6.6 vs the fixed pipeline's
9.4**, with `adversarial-claim` at 1–2/10. Diagnosing rather than
publishing that turned up the cause, and it was not the framework.

`harness.py` collects a run's deliverable by joining **terminal nodes
only**. The dynamic plan decomposed correctly — build-case →
red-team → surviving-claims, 14 KB of good work — but only the
601-character conclusion was terminal, so the judge scored it against a
rubric demanding the case, the critique *and* the survivors, and
correctly gave it 1/10. The same run judged on `result.output`, what
`Swarm.execute` actually returns: **10/10**.

That convention systematically penalises the topologies smythe
generates. A single-agent run has one node and loses nothing; a fixed
pipeline's last node is a self-contained deliverable by construction;
only a decomposed plan whose deliverable is cumulative gets most of its
work discarded before scoring. Measured across this suite:

| Baseline | Delivered | Terminal-only | Swing |
|---|---:|---:|---:|
| single agent | 8.0 | 8.5 | +0.5 |
| fixed pipeline | 8.3 | 9.6 | **+1.3** |
| smythe dynamic | 8.3 | 6.5 | **−1.8** |

The old convention inflated the fixed pipeline and deflated generated
topology, a combined swing of about 3 points — comparable to the gap
this project previously published as a framework result. **The v2–v5
campaign and the framework head-to-head both use this harness**, so
those tables carry the same confound and should be re-run before being
cited again.

## Results (delivered-output measurement)

| Baseline | Quality | Min | Nodes | Cost | USD / quality point |
|---|---:|---:|---:|---:|---:|
| single agent | 8.0 | 4 | 1.0 | $0.023 | **$0.00029** |
| fixed pipeline | 8.3 | 4 | 3.0 | $0.139 | $0.00167 |
| smythe dynamic | **8.3** | 4 | 3.8 | $0.107 | $0.00129 |

**Generated topology matches the fixed pipeline on quality at 23% lower
cost** — the first time in this project's record that dynamic has not
lost to a well-built fixed pipeline. But the honest headline is the
first row: **a single good LLM call is 4.5× more cost-efficient per
quality point than either multi-agent approach**, and only one point
behind on quality.

## Where shape actually decides the winner

The per-task table is the point of the suite:

| Task | Single agent | Fixed pipeline | Smythe dynamic |
|---|---:|---:|---:|
| trivial-transform | **10, 10** | 4, 4 | 4, 4 |
| parallel-profiles | 5, 4 | **10, 10** | **10, 10** (6–7 nodes) |
| deep-serial | 10, 10 | 10, 10 | 10, 10 |
| adversarial-claim | 10, 10 | 10, 10 | 10, 10 |
| mixed-audit | 6, 5 | 5, 10 | 5, 10 |

Shape dependence is real and measurable:

- **Decomposition wins decisively on the parallel task.** A single agent
  scores 4–5 profiling four independent technologies; both multi-node
  approaches score 10. This is the clearest evidence in the project that
  multi-agent structure buys real quality, not just latency.
- **Decomposition *loses* decisively on the trivial task.** The rubric
  demands a bare JSON array; multi-node output carries intermediate
  reasoning into the deliverable and both pipelines score 4 against the
  single agent's 10.
- **Dynamic did not right-size the trivial task.** It used 2 nodes where
  1 was correct and scored identically to the fixed pipeline. The
  Architect's cost-aware node-count guidance is not aggressive enough at
  the bottom of the range — an actionable defect, and the most useful
  thing this suite found about the framework itself.

## Honest caveats

n=2 per cell on five tasks; `mixed-audit` swings 5↔10 between reps for
both pipelines, so single-cell differences are noise. One judge, one
executor model. Tasks authored by this project. Cost figures come from
the blended-rate estimate, not provider invoices. Nothing here measures
the supervision or verification tiers added in this branch — that
experiment is next, and its pre-registered hypothesis is that
supervision lifts the *floor* (the 4s) rather than the mean.

## What this suite does not show

It does not show that smythe beats a bare LLM call on cost. On four of
five tasks a single agent is competitive or better, and it is far
cheaper. The case for generated topology rests on the parallel-profiles
row: work with genuinely independent parts, where a single call
conflates subjects and multi-node structure does not. That is a
narrower claim than "agent swarms are better," and it is the one the
evidence currently supports.
