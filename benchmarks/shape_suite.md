# The shape suite — task-shaped execution, measured

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
`gemini-pro-latest` (different vendor), zero errors. The suite has been
run three times; each campaign's raw records are kept, because two of
the three found bugs in smythe and the sequence is the finding:

| Campaign | Records | What it established |
|---|---|---|
| 1 (2 reps) | [shape_suite.json](results/shape_suite.json) | The terminal-join measurement bug |
| 2 (3 reps) | [shape_suite_v2.json](results/shape_suite_v2.json) | Right-sizing worked; `DELIVERABLE` synthesis regressed multi-part goals |
| 3 (3 reps) | [shape_suite_v3.json](results/shape_suite_v3.json) | Claimable wall time and observed quality; execution-stage cost only |

## Current evidence status

**Generated plans took 14% less end-to-end wall time than the fixed pipeline,
with observed quality in the same measured band.** The 15 runs per arm include
planning in their wall-clock timing.

The September 2026 audit found that the historical USD column excludes the
LLM architect's provider calls. It records execution and synthesis tokens at a
uniform $3 per million tokens. The 19% reduction is therefore an
**execution-stage cost estimate**, not total workflow savings or provider
invoice savings. The 20% cost-per-quality reduction has the same restricted
scope. Those figures no longer support end-to-end cost headlines.

New harness records count usage at the shared provider boundary, including
planning, execution, and synthesis. They retain per-call input/output tokens
and label the blended-rate estimate explicitly. The original records remain
unchanged; a new live campaign is required to establish full-workflow cost.

The corrected [offline accounting run](results/shape_suite_usage_offline.json)
passes all 15 task/arm combinations with 40 recorded provider responses,
including five planning calls. Its deterministic provider returns synthetic
usage; the actual API spend is $0. This record validates accounting mechanics,
not model quality or paid-provider efficiency.

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
campaign and the original `framework_h2h.json` record both use this harness**,
so those tables are diagnostic. The corrected
`framework_h2h_rightsized.json` rerun judges `Swarm.execute(...).output` and is
the current claimable framework comparison.

## The second bug: selecting a deliverable is not the same as producing one

Fixing the measurement changed the default synthesis strategy to
`DELIVERABLE`, which returns the terminal node's output instead of a
transcript of every node. That was validated on the homogeneous pipeline
suite, where the last node is a report step and therefore *does* contain
the deliverable. It does not generalise.

Re-running this suite after that change, generated topology collapsed
from 8.3 to **5.87**:

| Task | Smythe dynamic (regressed) |
|---|---:|
| adversarial-claim | 2, 2, 1 |
| deep-serial | 2, 4, 4 |

The plans were correct — `strong-case → red-team-critique →
surviving-claims` is the right shape. The failure was that a node
labelled "state what survives" obeyed its label and emitted the
survivors alone, so the case and the critique the goal *also* asked for
were never returned. Three of `adversarial-claim`'s four rubric criteria
describe content produced by non-terminal nodes, so most of the
deliverable was never present to be judged. The fixed pipeline *rose* to
9.27 under the same change, because its last step is a synthesis step by
construction.

Two changes fixed it. The terminal note now states that a step name
describes what a node contributes, not how much it outputs — that alone
recovered `deep-serial` to 10, 10 but left `adversarial-claim` at 1, 3,
because that task's constraint ("End with only the claims that survive")
instructs the node to be exclusive and a general instruction loses to a
specific one. So the planner became responsible for it: when a goal asks
for several parts, the final node's label must say it assembles the
complete deliverable and name them.

The lesson generalises beyond this fix. Both bugs were the same mistake
in different places — assuming some single node already holds the
deliverable, when nothing in the system guaranteed that. A convention
validated on one task shape silently misreports every other shape.

## Results (3 reps, 45 runs, zero errors)

Raw records: [results/shape_suite_v3.json](results/shape_suite_v3.json).

| Baseline | Quality | Min | Runs < 7 | Nodes | Execution cost estimate | End-to-end wall | Execution USD / quality point |
|---|---:|---:|---:|---:|---:|---:|---:|
| single agent | 8.73 | 5 | 4/15 | 1.00 | $0.032 | 4.5 s | **$0.00024** |
| fixed pipeline | 9.33 | 5 | 2/15 | 3.00 | $0.201 | 14.9 s | $0.00144 |
| smythe dynamic | **9.47** | **6** | 2/15 | 2.87 | $0.163 | 12.8 s | $0.00115 |

Measured judge variance on this setup is ±0.49. The observed scores, **9.47
and 9.33**, occupy the same measured quality band. Generated plans took
**14% less end-to-end wall time** and used **19% less estimated execution
cost**. Planning is included in wall time and excluded from the historical
cost column. These are observed campaign means from three repetitions per
task, not a statistical latency guarantee.

The pre-registered cost-per-quality hypothesis remains open at the complete
workflow boundary until planning usage is included in a new campaign.

## Where shape actually decides the winner

| Task | Single agent | Fixed pipeline | Smythe dynamic | Nodes used |
|---|---:|---:|---:|---:|
| trivial-transform | 10, 10, 10 | 10, 10, 10 | 10, 10, 10 | **1.0** |
| parallel-profiles | 5, 5, 5 | 10, 10, 10 | 10, 10, 10 | **5.3** |
| deep-serial | 10, 10, 10 | 10, 10, 10 | 10, 10, 10 | 2.0 |
| adversarial-claim | 10, 10, 10 | 10, 10, 10 | 10, 10, 10 | 3.0 |
| mixed-audit | 10, 6, 10 | 5, 10, 5 | 10, 6, 6 | 3.0 |

- **Decomposition wins decisively on the parallel task.** A single agent
  scores 5, 5, 5 profiling four independent technologies; both
  multi-node approaches score 10, 10, 10. A five-point gap, ten times
  judge variance. This remains the clearest evidence in the project that
  multi-agent structure buys real quality rather than just latency.
- **Right-sizing now works at both ends.** The previous campaign's most
  useful finding was that the Architect used 2 nodes on the trivial task
  where 1 was correct, and scored 4/10 for it. It now uses exactly 1 and
  scores 10, 10, 10 — while still using 5.3 on the parallel task. The
  spread between 1.0 and 5.3 nodes *is* the feature.
- **A single call sets the price floor; Smythe decides when structure earns
  more calls.** The one-node baseline is cheapest overall, but scores 5/10 on
  the parallel task where both multi-node approaches score 10/10. Smythe uses
  one node on the trivial task and expands only for work that benefits from
  decomposition.

## Measurement scope

n=3 per cell on five tasks. `mixed-audit` still swings (10, 6, 10 for a
single agent; 5, 10, 5 for the fixed pipeline), so single-cell
differences on that task are noise and it is the one task no arm has
solved. One judge, one executor model, tasks authored by this project.
Cost figures are execution and synthesis tokens at a blended $3 per million
tokens, excluding planning. They are not provider invoices.

Three campaigns have now been run against this suite and two of them
found bugs in smythe rather than facts about it. That is the suite
earning its keep, but it also means these numbers describe a system that
changed twice while being measured — they should be treated as current,
not as a long-standing track record.

## Claim boundary

This suite establishes the value of selecting execution shape, not a universal
rule that more agents are better. The result rests on both ends of the routing
decision: `parallel-profiles`, where one call conflates independent subjects,
and `trivial-transform`, where a fixed pipeline pays for three stages to do one
node's work. Supervision and verification are measured separately in the
[control ablation](control_ablation.md).
