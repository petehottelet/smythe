# Do supervision and verification earn their cost?

Smythe added three control features on the claim that they improve
outcomes: acceptance criteria (`done_when`), verification that gates and
regenerates (`verifies`), and a supervisor that revises the plan mid-run.
None had ever been measured. This is that measurement, and the answer is
mostly no.

Protocol: four arms against the [shape suite](shape_suite.md), executor
`gpt-5.4-mini`, blind judging by `gemini-pro-latest`, 3 reps per cell,
two full campaigns, zero run errors.

    plain        default planning; constraints only
    criteria     constraints restated as done_when, plus a review node
                 that runs and is paid for but cannot send work back
    gated        identical plan, and that review node gates for real
    supervised   plain, plus LLMSupervisor with max_revisions=2

`done_when` is the task's own `constraints` verbatim. The system already
sees the constraints in every arm, so no new information enters with the
criteria — the comparison isolates the *mechanism*, not the briefing.
Feeding the judge's rubric in as `done_when` would be teaching to the
test and is deliberately not done.

**Pre-registered hypothesis:** these features lift the *floor*, not the
mean — fewer catastrophic runs at similar average quality, bought with
more tokens. Pre-registered consequence: if the floor does not move, say
so and stop investing in them.

## Result: the floor did not move

Pooled across both campaigns (n=30 per arm; `criteria` is n=15, see
[Two campaigns](#two-campaigns-and-why-both-are-published)):

| Arm | Quality | Runs < 7 | Cost / run | Mechanism activity |
|---|---:|---:|---:|---|
| plain | 8.80 | 6/30 | **$0.0111** | — |
| criteria | 9.67 | 1/15 | $0.0138 | — |
| gated | 8.87 | 6/30 | $0.0149 | fired 4 times in 30 runs |
| supervised | 9.17 | 5/30 | $0.0119 | **94 reviews, 0 proposals** |

Floor runs: 6/30 plain, 6/30 gated, 5/30 supervised. The hypothesis
predicted these features would reduce that count. They did not.

## Why the quality column cannot decide anything

Scores on this suite are bimodal — 24 of 30 plain runs scored 10, the
rest scored 1, 2, 5, or 6. A cell mean is therefore decided by how many
rare catastrophic runs happen to land in it, not by the arm.

The same arm, run twice with identical configuration:

| Arm | Campaign 1 | Campaign 2 | Swing |
|---|---:|---:|---:|
| plain | 9.40 | 8.20 | **1.20** |
| gated | 9.00 | 8.73 | 0.27 |
| supervised | 9.00 | 9.33 | 0.33 |

The largest gap *between* arms was 0.40 in campaign 1 and 1.47 in
campaign 2. **`plain` disagrees with itself by more than campaign 1's
entire spread between arms.** Any ranking read off a single campaign is
reading noise. This is why both campaigns are published rather than the
prettier one.

## What is not noise

Three findings are structural rather than judged, and they survive:

**Supervision never intervenes.** Across 30 runs the supervisor was
consulted **94 times and proposed a change zero times.** Not rejected
proposals — no proposals. That is 94 paid provider calls buying nothing.
A direct probe confirms `LLMSupervisor` *can* fire: it correctly
diagnosed a truncated deliverable during debugging. It fires when the
deliverable is visibly incomplete, which is exactly the failure the
[assembly fix](shape_suite.md) removed. Supervision was substantially
compensating for a bug that no longer exists.

**Gating fires rarely and catches the wrong thing.** 4 regenerations in
30 runs (13%), for **34% more cost per run** than plain. Floor runs are
identical to plain, 6/30 both. Most importantly, gated runs still
produced 2s and 4s on `adversarial-claim` and `deep-serial` — the
assembly failure mode. A verifier asked "does this meet the criteria"
reads a fluent, well-formed partial deliverable and passes it. Gating
catches malformed output; it does not catch *absent* output.

**Cost ordering is stable.** gated ($0.0149) > criteria ($0.0138) >
supervised ($0.0119) > plain ($0.0111). Gating is the most expensive
arm in both campaigns.

## The one result worth following up

`criteria` scored 9.67 with 1/15 floor runs — the best of any arm. It
differs from `gated` only in that a failed verdict is *not* enforced;
same information, same graph, same token cost. If real, that says
stating acceptance criteria helps and enforcing them hurts.

It is n=15, one campaign, and `plain` swung 1.20 across campaigns. **This
is not established.** It is the one arm worth spending reps on.

## Two campaigns, and why both are published

Campaign 1 ([control_ablation.json](results/control_ablation.json))
carries a broken `criteria` arm that scored 3.40 with eleven 1s. The
cause was in the experiment, not the framework: the arm disarmed the
gate by clearing `verifies`, which turned the review node into an
ordinary terminal node, so `DELIVERABLE` synthesis returned the PASS/FAIL
verdict *as the deliverable*. The correct disarm is
`max_regenerations=0`, the documented advisory mode, which keeps
`verifies` set so synthesis still excludes the node.

Campaign 2 ([control_ablation_v2.json](results/control_ablation_v2.json))
fixes it. The invalid records are kept at
[control_ablation_v1_invalid.json](results/control_ablation_v1_invalid.json)
rather than deleted, and campaign 1's other three arms remain valid and
are pooled above.

An earlier attempt was discarded entirely: the planner emitted a gate in
only 2 of 15 `gated` runs, so that arm was mostly a rerun of `plain` and
appeared to show a 0.6-point lift from a mechanism absent from 13 of its
15 runs. The gate is now injected when the planner omits one, so every
gated run gates.

## Recommendation

Per the pre-registration: **stop investing in supervision and
verification gating as quality mechanisms.** On this workload supervision
never fires, gating fires 13% of the time and catches a failure mode that
is not the one that occurs, and neither moves the floor.

Both features should stay in the codebase — they are bounded, off by
default, cheap when unused, and gating is genuinely useful for the
objective checks it was designed around (image dimensions, schema
conformance) rather than for judged prose quality.

The residual failure is worth naming, because none of the three features
address it: catastrophic runs are **assembly failures**, where the final
node returns its own increment instead of the whole deliverable. It is
prompt-dependent and stochastic, currently ~1 run in 8. Making assembly
deterministic is where the next effort belongs, not another control tier
layered on top.

## Caveats

Five tasks, one judge, one executor model, tasks authored by this
project. Cost is the blended-rate estimate, not provider invoices.
`mixed-audit` is unsolved by every arm and contributes disproportionately
to the floor counts. The `criteria` arm has half the samples of the
others. Nothing here measures gating on the objective checks it was built
for — only on judged prose.
