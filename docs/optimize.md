# Bounded Autotune

Smythe's bounded Autotune package runs repeatable optimization campaigns
without giving a research agent an unbounded code, time, or spend surface.

The initial shipped campaign is a zero-cost concurrency search with a
deterministic evaluator. It combines immutable experiment contracts,
deterministic candidate identity, an async bounded runner, paired promotion
statistics, an append-only SQLite ledger, and an installed `smythe optimize`
CLI. Development and confirmation are reproducible from the contract; the
holdout is sealed with fresh durable campaign material and is reproducible
only within that campaign.

This design borrows the useful core of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): make one
targeted change, evaluate it against a fixed measurement program, retain the
result, and keep only improvements. It adds controls needed for agent and
artifact workflows, where trials may be noisy, billable, parallel, or
ambiguous after a connection failure.

## What exists today

| Module | Responsibility |
|---|---|
| `smythe.optimize.contracts` | Versioned contracts, objectives, allowlisted candidate patches, evaluator manifests, canonical JSON, and content-derived identities |
| `smythe.optimize.concurrency` | Deterministic offline concurrency simulator and evaluator fingerprint |
| `smythe.optimize.statistics` | Direction-normalized paired comparisons, paired Student-t promotion bounds, three-way verdicts, descriptive bootstrap intervals, and promotion policy |
| `smythe.optimize.ledger` | Durable campaign, candidate, trial-event, cost-exposure, and decision history |
| `smythe.optimize.engine` | Bounded development, confirmation, and holdout orchestration with conservative recovery |
| `smythe.optimize.inspection` and `smythe.optimize.report` | Read-only evidence collection and standalone HTML reports |

These modules are deliberately separable from proposal generation. A human,
an agent, or a deterministic grid can propose candidates, but every candidate
is reduced to the same small, immutable policy patch before evaluation.

The package is pre-1.0 and its API may change. The programmatic runner is
`OptimizationRunner` in `smythe.optimize.engine`.

The [0.7.0 release](https://github.com/petehottelet/smythe/blob/v0.7.0/docs/optimize.md)
uses ledger schema v3. Campaign ownership, schema v4, and HTML reports below
ship in **0.8.0**; 0.8.1 adds the paired Student-t promotion rule and holdout
sealing. Install `pip install "smythe==0.9.1"`.

## Experiment contracts and budgets

`ExperimentContract` freezes the measurement policy before a campaign starts.
It includes:

- exactly one primary objective, plus optional secondary objectives;
- maximize or minimize direction for every metric;
- hard minimum and maximum metric bounds;
- an optional maximum tolerated regression for each secondary metric;
- the only dotted policy fields candidates may change, plus their scalar
  types, numeric bounds, and optional enumerated choices;
- the exact Boolean gate inventory every evaluator outcome must return;
- development, confirmation, and holdout repetition counts;
- candidate, parallel-candidate, trial, wall-time, and spend caps;
- a per-trial reservation in integer micro-USD;
- the confidence level, minimum improvement, and base seed.

Contracts reject non-finite values, unsafe field paths, incoherent trial caps,
and a budget that cannot reserve one complete development, confirmation, and
holdout lifecycle. Canonical JSON produces a stable contract hash.

`Candidate` accepts only fields on the contract's mutation allowlist and
validates each value against its declared rule. Its policy, hypothesis,
parent, and contract lineage are bound into deterministic hashes and a
`cand_<sha256>` identifier. Campaign creation atomically seals the complete
ordered inventory; a declared parent must precede its child in that same
inventory under the same contract. Candidates cannot be appended later. This
makes a candidate a reviewable experiment input rather than permission for an
agent to edit arbitrary source files.

The ledger enforces candidate count, trial count, exact per-trial reservation,
and total cost exposure when records are added. It accounts in micro-USD to
avoid floating-point drift. Completed trials contribute confirmed cost;
prepared and dispatched trials retain their reservation; unknown outcomes
retain their full ceiling as unresolved exposure.

The runner checks that the complete paired plan fits the candidate, trial, and
budget caps before creating work. Executable contracts must declare at least
one gate and a value rule for every mutable field, so empty gate maps and
unbounded mutations cannot pass vacuously. It runs the declared number of
trials, bounds active evaluator series with `max_parallel_candidates`, and
applies `max_wall_seconds` to the campaign and each remaining evaluator call.
Confirmation and holdout compare the incumbent with only the one selected
challenger, limiting both cost and repeated testing; declared
[reference arms](#reference-arms) join those comparisons but are never
promoted.

Independent implementation ceilings bound otherwise-valid contracts to 1,024
candidates, 256 parallel candidate series, 10,000 repetitions per split, and
100,000 trials. Descriptive bootstrap work is capped at 1,000,000 resamples
and checks the same monotonic deadline cooperatively. The installed
concurrency command uses tighter limits: 64 challengers, 64 parallel series,
100 repetitions per split, 100,000 bootstrap resamples, one million work items
or concurrency units, and ten million simulated item-trials per campaign.

On resume, previously completed evaluator durations count against the new
deadline so restarting cannot reset the time cap. The runner sums those
durations, including trials that originally overlapped. The resulting resumed
limit is deliberately conservative and can be lower than actual elapsed wall
time for a parallel campaign.

## Confirmation and holdout protocol

The runner keeps search data separate from promotion evidence:

| Split | Purpose | Seed rule | Decision use |
|---|---|---|---|
| Development | Screen or rank candidate ideas cheaply | Deterministic seeds may be reused while searching | Never sufficient for promotion |
| Confirmation | Re-evaluate the selected candidate against the incumbent and any references | Candidate, incumbent, and references use the same ordered, unique seeds | Candidate must pass the complete promotion policy |
| Holdout | Test the confirmed candidate on evidence not used for selection | Derive a separate paired seed set from secret material created only after the candidate plan is frozen | Candidate must pass again before promotion |

`compare_metric()` requires at least three equal-length baseline/candidate
observations and exact ordered seed equality. Improvements are normalized so a
positive delta is always better: candidate minus baseline for a maximized
metric, and baseline minus candidate for a minimized metric.

`assess_promotion()` promotes only when all of the following are true:

- every deterministic gate passes;
- every candidate aggregate satisfies its hard metric bounds;
- every secondary metric stays within its declared regression allowance; and
- the primary metric's one-sided paired Student-t lower bound is strictly
  greater than `min_improvement`.

The lower bound is `mean - t * sd / sqrt(n)` over the `n` paired
improvements, where `t` is the Student-t quantile with `n - 1` degrees of
freedom at upper-tail probability `(1 - confidence) / 2`. At the default
confidence of 0.95 each confirmation or holdout test therefore admits a truly
null candidate about 2.5% of the time when paired improvements are roughly
normal; with five pairs, `t` is 2.776. The quantile is computed in pure Python
for every degree of freedom and every contract confidence. Identical paired
improvements have zero spread, so the bound equals their mean; one pair cannot
form a bound. Each recorded comparison saves its method (`paired_student_t`),
degrees of freedom, standard error, and critical value. The reported
`confidence_interval` is the matching two-sided t interval, so its lower
endpoint is the promotion bound.

Each assessment also records a three-way `verdict` read from both endpoints
of the primary interval:

| Verdict | Primary interval |
|---|---|
| `improved` | Lower endpoint strictly greater than `min_improvement` |
| `worse` | Upper endpoint strictly below zero |
| `inconclusive` | Anything else, including an interval that touches either threshold |

The verdict describes the primary evidence only. An `improved` candidate is
still rejected when a gate, hard bound, or secondary allowance fails, and an
`inconclusive` verdict claims neither an improvement nor a regression. A
campaign's verdict is its holdout verdict when the holdout ran and its
confirmation verdict otherwise; a campaign that stops in development forms no
paired interval, so its verdict is `inconclusive`. Every terminal decision
records the campaign's verdict as `verdict` in its assessment, including a
decision made in development, which records `inconclusive`. Identical paired
improvements form a zero-width interval, so even three pairs give a definite
verdict; that reflects the data, not measured precision.
`improvement_verdict(interval, min_improvement)` in
`smythe.optimize.statistics` applies the same rule.

Comparisons also save a seeded percentile bootstrap interval as
`descriptive_bootstrap_interval`. It is description only and never decides
promotion: with the small samples Autotune uses, that bootstrap bound admitted
about 8% of null candidates with five pairs and 12–14% with three, against a
nominal 2.5%. Decisions
recorded by 0.8.0 and earlier used that bootstrap bound; they still load,
inspect, and report, labeled with their original rule.

Each campaign
stores a random 32-byte holdout secret and publicly records only a
domain-separated SHA-256 commitment. Holdout seeds are HMAC-derived after the
complete candidate plan has been bound. The campaign row, plan hash, ordered
candidate IDs and policy hashes, canonical candidate inventory, candidate
rows, secret, and commitment are created in one SQLite transaction. There is
no empty or partially registered campaign state and no later inventory
expansion. Resuming the same ledger reproduces the holdout, while a fresh
ledger gets an independent holdout. Three pairs are only a
structural minimum, not a claim of useful statistical power. Choose repetition
counts from measured evaluator variance, and use a larger sample when the
expected improvement is small.

A challenger policy gets one sealed holdout per holdout identity in a ledger.
The holdout identity covers what defines the holdout evaluation and its
decision: the evaluator hash, every objective (name, direction, primary flag,
hard bounds, and regression allowance), the required gates, the holdout
repetition count, the confidence level, and the minimum improvement. Holdout
use is keyed by that identity and the challenger's policy hash, which is
computed from canonical policy content, so key order, `1` versus `1.0` in a
number field, and `-0.0` do not matter. The first holdout trial a campaign
prepares for a challenger records that use, atomically with the trial and
before any holdout evaluation.

A later campaign in the same ledger whose challengers include a policy that
already used its holdout under the same identity is refused with
`HoldoutAlreadyUsedError` before any evaluator call. Nothing outside the
identity draws a new holdout: not a reworded hypothesis, a changed or
reordered challenger set, a different incumbent, or a new `--campaign-id`,
and not a change to the contract name, mutable fields or their rules,
development or confirmation repetitions, base seed, or any candidate,
parallel-candidate, trial, wall-time, spend, or per-trial reservation cap. For
the CLI, that means `--candidate-concurrency`, `--max-wall-seconds`,
`--max-parallel-candidates`, `--development-repetitions`,
`--confirmation-repetitions`, `--bootstrap-resamples`, and
`--ledger-durability`. Changing a field inside the identity defines a
different holdout evaluation and draws a fresh holdout; for the CLI, that
means `--work-items`, `--provider-capacity`, `--base-latency-ms`,
`--max-p95-latency-ms`, `--max-error-rate`, `--holdout-repetitions`,
`--confidence`, or `--min-improvement`. The campaign that used the holdout can
still be resumed and replayed. Incumbent and [reference](#reference-arms)
holdout trials do not count as use.

Each use's identity is recomputed from its campaign's stored contract and its
holdout trial's evaluator hash, so holdout trials recorded by 0.8.0, which
kept no separate record, still count.
`holdout_identity(contract, evaluator_hash)` in `smythe.optimize.ledger`
returns an identity, and `ExperimentLedger.holdout_uses(identity)` maps each
policy that used it to the campaigns that did, in first-use order. A holdout
drawn from a finite task pack can also be limited across policies; see
[finite task pools](#finite-task-pools).

The runner evaluates every policy on development seeds, removes challengers
that fail a gate, hard metric bound, primary mean-improvement threshold, or
secondary non-regression limit, and deterministically selects the viable
challenger with the best primary mean. It then evaluates that challenger and
the incumbent on matching confirmation seeds. Holdout runs only if
confirmation passes, and promotion requires both assessments to pass. Every
terminal decision this release records keeps each challenger's development
score, in challenger order, under `development_scores`, beside the selected
challenger's score under `development`.

Do not repeatedly tune against the holdout set: once it influences a proposal,
it is no longer held out. The ledger refuses a second holdout for the same
policy and holdout identity, but it cannot stop a changed evaluation or a
fresh ledger from drawing another. There is no multiple-comparison correction
across development challengers: development compares each challenger's mean
with the incumbent's without an interval, and only the one selected challenger
is tested on confirmation and holdout at the contract confidence. Screening
many challengers therefore favors one that was lucky in development, and the
independent one-time holdout is an important guard against that selection
bias.

### Reference arms

One campaign can compare its challenger with more than one baseline. Pass
sealed comparison policies, such as the policy the incumbent replaced or a
simpler alternative, as `references`:

```python
result = await runner.run(
    incumbent,
    challengers,
    references=(predecessor, simpler),
    reference_rule="must_not_be_worse",
)
```

References are sealed in the candidate inventory after the challengers, so a
reference may name a challenger as its parent but not the reverse. Each must
differ from the incumbent and every challenger by candidate ID and policy
hash. References take no part in development or selection. Each runs beside
the incumbent and the selected challenger on the same confirmation seeds and,
when confirmation passes, the same holdout seeds and tasks, under its own
`reference.plan_<plan hash>` trial phase. The challenger is then assessed
against each reference exactly as against the incumbent, with the reference as
the baseline, so each reference's verdict reads from the challenger's side:
`worse` means the reference is better on the primary objective.

`reference_rule` decides what those comparisons do:

| Rule | Effect |
|---|---|
| `report` (default) | Records each comparison; promotion depends on the incumbent comparison alone |
| `must_not_be_worse` | A stage also fails when any reference's verdict there is `worse`; a confirmation that fails this way never reaches the holdout |

Each comparison uses the contract confidence, with no multiple-comparison
correction across references.

A reference is never promoted and is never the decision candidate. Its holdout
trials are therefore not holdout use: a policy that already used its holdout,
such as an earlier challenger, can still be a reference, and serving as one
uses nothing. Reference holdout trials still draw from the holdout pool and
count toward [`max_holdout_draws`](#finite-task-pools), so a finite held-out
pack cannot be re-drawn through references.

Each reference adds one confirmation series and one holdout series to the
plan's required trials and one candidate to `max_candidates`. The reference IDs
and rule join the plan hash only when references are declared, so two-arm
plans keep their hashes. The decision records `reference_rule` and a
`references` list giving each reference's ID, policy hash, confirmation and
holdout assessments, and final verdict; a campaign that stops in development
records only the IDs and policy hashes. `OptimizationResult.references`
returns the same entries. The ledger derives references from the sealed
inventory, the trial phases, and the decision: declared references must be
the last sealed candidates, in order, with trials only in the paired splits
under the reference phase, and no challenger may use that phase.

The ledger schema, contract hash, and holdout identity are unchanged. Earlier
releases count reference holdout trials as holdout use, which is stricter, and
refuse a decision with references when they validate it, as
`smythe optimize inspect --out` does; export reports for such campaigns with a
release that supports reference arms.

## Deterministic offline concurrency campaign

`ConcurrencyScenario` is the first reference evaluator. It simulates a fixed
number of provider-like work items with service-time jitter, contention,
capacity pressure, and overload failures. The only candidate field is
`max_concurrency`.

```python
from smythe.optimize.concurrency import ConcurrencyScenario, simulate_concurrency

scenario = ConcurrencyScenario(provider_capacity=8)
metrics = simulate_concurrency(
    {"max_concurrency": 8},
    scenario=scenario,
    seed=42,
    split="development",
)
```

The evaluator returns throughput, p95 service latency, error rate, successful
operation count, and simulated wall time. Parallelism initially improves
throughput, while contention raises latency and concurrency above provider
capacity sharply raises failures. The holdout split applies a small fixed
latency shift and uses seeds derived from the campaign's sealed material.

The scenario's `evaluator_hash` binds its values and simulator version. Store
that hash with each trial so resumed evidence cannot silently switch evaluator
definitions.

This workload makes concurrency optimization fast, free, and reproducible
within a durable campaign. It is a model, not a measurement of OpenAI, Gemini,
Anthropic, network, or local hardware performance. A policy that wins the
simulator must still be benchmarked against a separately bounded real workload
before making a production claim.

### Run the offline campaign

The default command compares a serial incumbent (`max_concurrency: 1`) with
challengers at 2, 4, 8, 12, and 16. It uses three development repetitions,
five confirmation repetitions, five holdout repetitions, and a zero-cost
contract:

```bash
smythe optimize concurrency --json
```

This command uses only the local simulator. It needs no API key, makes no
provider or network calls, and declares zero API spend.

Useful controls include `--candidate-concurrency`, `--work-items`,
`--provider-capacity`, `--base-latency-ms`, `--max-p95-latency-ms`,
`--max-error-rate`, the three `--*-repetitions` options,
`--max-parallel-candidates`, `--max-wall-seconds`, `--confidence`,
`--min-improvement`, and `--bootstrap-resamples` (descriptive interval only).
`--reference N[,N...]` adds [reference arms](#reference-arms) at those
concurrencies under the `report` rule; a reference cannot repeat a
challenger's value or the incumbent's `1`. Use `--ledger PATH` to
override the default `~/.smythe/optimize.sqlite3`. The CLI defaults to
`--ledger-durability normal`; choose `full` when the evidence must also survive
host power loss rather than only a process crash.

Output includes the contract, evaluator, optimization-plan, and public holdout
commitment identities; the campaign's three-way `verdict` beside `promoted`;
development and paired statistical evidence, including `evidence.references`
when the campaign has references; the selected candidate (or null when none is
development-viable); and a ledger snapshot. A completed campaign
exits with code 0 whatever its verdict. The secret holdout material is never
emitted. `recommended_patch` appears only when confirmation and holdout both
promote the candidate. Rejection never emits a patch to apply.

The optimization plan hash binds the contract, incumbent, ordered challenger
set, policy hashes, evaluator hash, runner version, promotion method,
descriptive bootstrap configuration, ledger durability, and required trial
count, plus any declared [task pools and holdout draw cap](#finite-task-pools)
and [reference arms](#reference-arms).
When `--campaign-id` is omitted, the durable campaign identity is
derived from that full plan. Repeating the same command reuses completed
trials; changing a candidate, contract, evaluator, bootstrap, or durability
setting creates a different automatic campaign. That campaign is refused if
one of its challenger policies already used its holdout under the same
holdout identity, which the challenger set and operational flags do not
change. The holdout commitment is created when a new fully bound campaign is
atomically created and is checked on every resume and decision. Supplying `--campaign-id` does not weaken the binding: reopening it
requires an exact plan and candidate-inventory match.

The promotion method is part of the plan, so a campaign created by 0.8.0 is
not resumed under the paired t rule: the same command without `--campaign-id`
derives a new campaign. Re-running an explicit `--campaign-id` created by 0.8.0
exits with code 8 (campaign binding drift); finish that campaign on 0.8.0 or
choose a new ID.
The method is not part of the holdout identity. If the earlier campaign
reached its holdout, that challenger policy's holdout is already used and the
new campaign is refused; inspect the earlier campaign's recorded decision
instead.

The runner version is part of the plan as well. This release uses runner
version 4, which records the three-way verdict and every development score in
terminal decisions, so its plan hashes and automatic campaign IDs
(`optimization_v4_…`) differ from those of earlier releases. The same command
without `--campaign-id` derives a new campaign; re-running an explicit
`--campaign-id` created by an earlier release exits with code 8 (campaign
binding drift). The runner version is not part of the holdout identity either,
so a holdout an earlier release used stays used.

`smythe optimize` exits with code 8 for campaign state errors (an unknown
campaign, a plan that conflicts with a saved campaign, a used holdout, work that
needs attention) and for local SQLite or filesystem failures, and with code 9
when a declared limit (wall time, trials, candidates or spend) stops or refuses
it. With `--json`, `error.type`
names the exact error, such as `HoldoutAlreadyUsedError`.

Inspect a campaign without mutating the ledger:

```bash
smythe optimize inspect CAMPAIGN_ID --json
```

Inspection opens SQLite read-only, reports the evaluator hashes and complete
ledger snapshot, and does not create a missing ledger. Immutable inspection
requires the writer to be closed and its WAL checkpointed. The reader checks
SQLite sidecars and the database fingerprint before and immediately after open
and again at close. This detects practical races, but it is not a filesystem
lock: an external writer that bypasses Smythe and starts and checkpoints wholly
between checks is outside the guarantee.

<a id="export-a-campaign-report-unreleased"></a>

### Export a campaign report

Add `--out` to inspect a closed campaign as a standalone HTML document:

```bash
smythe optimize inspect CAMPAIGN_ID --ledger saved.sqlite3 --out report.html
```

The report opens with the recorded promotion or rejection, its reason, trial
states, and exact USD balances. It separates confirmed completed-trial cost,
held reservations, unknown exposure, and the contract's budget limit. Unknown
outcomes remain visible even when their reserved cost is zero.

Saved development scores, confirmation and holdout comparisons, candidate
policies, and trial observations follow. Black-and-white interval plots show
the recorded improvement and confidence interval, sample count, and objective
direction. Each comparison names the rule that produced its promotion bound:
the paired Student-t bound, with its degrees of freedom, standard error, and
critical value, or the percentile bootstrap bound for decisions recorded by
0.8.0 and earlier. A saved descriptive bootstrap interval is labeled as not
used for promotion. Positive improvement means better. Raw metric bounds and
secondary mean-regression rules remain distinct from the primary confidence
threshold. Metric names are preserved; the contract does not declare physical
units.

The decision summary and each confirmation and holdout section show the
three-way verdict. A decision recorded before verdicts were saved shows one
derived from the saved primary interval endpoints and minimum improvement,
labeled as derived, or "Not recorded" when either is missing. A decision made
in development shows the `inconclusive` verdict it records; one recorded by
an earlier release shows none. A recorded verdict
that is not one of the three values, or that disagrees with its saved interval
or with the decision's final stage, stops export, as does any verdict other
than `inconclusive` on a decision without a paired stage.

A decision with [reference arms](#reference-arms) adds a References section:
its rule, a table that compares the incumbent and each reference with the
decision candidate on each paired stage (arm and candidate means of the
primary objective, the saved interval, and the verdict), and each reference's
final verdict. The policies list marks each reference. A reference that is not
a recorded candidate, or whose verdict disagrees with its saved interval or
final stage, stops export.

Export does not rerun an evaluator, recompute statistics, rank candidates, or
change a decision. Every retained development score is shown, and the
challenger selected for confirmation is marked. Decisions recorded by earlier
releases kept only the selected score once a campaign advanced beyond
development; the report says so. Custom assessments without recognized
comparisons remain available as text. Malformed plot values or inconsistent
retained decision evidence stop export.

The first 500 trial records appear in detail, with the returned and total counts
shown explicitly. Whole-campaign balances, counts, policies, and saved decisions
remain complete. Programmatic callers can set `trial_limit` from 1 to 1,000
with `collect_optimization_report`; the limit bounds displayed detail, not the
ledger work needed to validate the campaign.

The report includes a SHA-256 fingerprint of its finite canonical JSON evidence,
plus the contract, plan, evaluator, and policy identities. This identifies the
exported evidence, not the database file or an independently attested benchmark.
Evaluator hashes alone do not establish whether a campaign used simulation or
a live provider.

The source must be closed and checkpointed, as for ordinary inspection. Export
validates the saved decision inventory and completes the final read-only ledger
check before publishing. Existing output files and database or sidecar targets
are refused. Publication uses an exclusive atomic link, requires a filesystem
that supports that operation, and limits the report to 32 MiB. Failed publication
does not overwrite another file. The existing inspection JSON remains unchanged,
including when `--json` and `--out` are combined.

Reports contain escaped local evidence with native disclosure controls. They
load no scripts, remote resources, or artifacts and work offline on desktop and
mobile browsers. Recorded policies, trial errors, and decision details are part
of the exported document.

### Programmatic runner

Custom in-process campaigns use the same bounded engine:

```python
from smythe.optimize import OptimizationRunner

runner = OptimizationRunner(
    contract,
    ledger,
    evaluate,
    evaluator_hash="sha256:<64 lowercase hex>",
    bootstrap_resamples=2_000,  # descriptive interval only
)
result = await runner.run(incumbent, challengers)
```

The async evaluator receives an immutable `TrialContext`, including the
campaign's monotonic deadline and, when the runner declares
[task pools](#finite-task-pools), the task to run, and must return a
`TrialOutcome` containing
exactly the contract's metrics and gate inventory, signed-63-bit integer
micro-USD cost, and optional SHA-256 artifact hashes.
`OptimizationResult` carries the plan identity, development scores,
confirmation and holdout assessments, the campaign's three-way `verdict`, any
[reference](#reference-arms) comparisons under `references`, decision and
trial keys, and ledger snapshot. Each `PromotionAssessment` has its
own `verdict`.

### Evaluator manifests

`evaluator_hash` must change whenever what the evaluator measures changes.
`EvaluatorManifest` builds it from the parts that define an evaluation:

```python
from smythe.optimize import EvaluatorManifest
from smythe.validation import ValidatorIdentity

manifest = EvaluatorManifest(
    code_digest="sha256:<evaluator code>",
    split_manifests={
        "development": "sha256:<development task-pack manifest>",
        "confirmation": "sha256:<confirmation task-pack manifest>",
        "holdout": "sha256:<holdout task-pack manifest>",
    },
    validators=(ValidatorIdentity.for_config("required-sections", "1", {"min": 3}),),
    judge={"model": "judge-model", "rubric_version": "2"},
    price_snapshot_version="2026-10-01",
)
runner = OptimizationRunner(
    contract, ledger, evaluate, evaluator_hash=manifest.evaluator_hash
)
```

`code_digest` identifies the evaluator code. `split_manifests` binds the task
pack each split draws from, so development and held-out tasks are separate,
named task families rather than repeated runs of one prompt. `validators`
lists acceptance validator identities (name, version and configuration
digest), `judge` is any judge's JSON configuration, and
`price_snapshot_version` names the prices used for cost. Every digest uses
the `sha256:<64 lowercase hex>` form; a task pack whose manifest records a
plain hex SHA-256 is written with the `sha256:` prefix.

`evaluator_hash` is the `sha256:` digest of the manifest's canonical JSON
(`to_dict()`). Object key order and validator order do not change it, and
empty fields are left out, so a manifest that leaves the judge unset hashes
the same as one with an empty judge. Changing any part changes the hash. The
holdout identity includes the evaluator hash, so a changed holdout task pack,
validator, judge, or price snapshot defines a new holdout evaluation.
`EvaluatorManifest.from_dict()` reads the JSON form back and refuses unknown
fields. Runners still accept any raw `sha256:` evaluator hash; the manifest is
a convenience for computing one. Keep a held-out pack's tasks away from
whoever proposes candidates: the manifest records only its digest.

### Finite task pools

Seeds are unlimited, but a task pack is not. Declare how many distinct tasks
each split's evaluator draws from with `task_pools`:

```python
runner = OptimizationRunner(
    contract,
    ledger,
    evaluate,
    evaluator_hash=manifest.evaluator_hash,
    task_pools={"development": 12, "confirmation": 12, "holdout": 12},
)
```

Each `TrialContext` for a split with a pool then carries a `task_index` from
0 to one less than that split's pool size; the evaluator runs that task.
Within a split no two seeds share a task, and the incumbent and challenger
trials on one seed run the same task, so every pair compares the two
policies on one task. The indices are a deterministic shuffle keyed by the
split's seeds: a resumed campaign draws the same tasks, development and
confirmation draw the same tasks in every campaign under one contract, and
holdout tasks follow the campaign's sealed holdout seeds. Each split's pool
is indexed on its own; give confirmation its own pack if it must not repeat
development tasks. A split without a pool gets `task_index=None`, as before.

A plan whose repetitions for a split exceed that split's pool is refused with
`OptimizationLimitError` before the campaign is created, since it would have
to repeat a task. Pools that are declared join the plan hash, so changing one
starts a different automatic campaign; a plan that declares none keeps its
hash. Pool sizes are not part of the holdout identity. Requirements the
engine does not model, such as a regression set whose results must not get
worse, belong in the evaluator as required gates.

Fresh holdout seeds do not make a finite holdout pack fresh: every campaign
that reaches the holdout runs some of its tasks, whatever policy it tests.
`max_holdout_draws` caps how many campaigns may draw from one holdout
identity:

```python
runner = OptimizationRunner(
    contract,
    ledger,
    evaluate,
    evaluator_hash=manifest.evaluator_hash,
    task_pools={"holdout": 12},
    max_holdout_draws=1,
)
```

A campaign draws when it prepares its first holdout trial, for the incumbent,
a challenger, or a reference, and every campaign that did so under the
identity counts, including campaigns run without a cap. Once the cap is
reached, a new campaign under the identity is refused with
`HoldoutAlreadyUsedError` before it is created, even for a policy that never
used the holdout; a campaign that
already drew can still resume and replay. The ledger repeats the check in the
write transaction that records a campaign's first holdout trial, so two
campaigns racing for the last draw cannot both take it, and
`ExperimentLedger.prepare_trial(..., max_holdout_draws=n)` applies the same
rule to low-level callers. `ExperimentLedger.holdout_draw_campaigns(identity)`
lists the campaigns that drew, in first-draw order.

The cap is not part of the holdout identity, so raising or removing it never
reopens a policy's used holdout. It joins the plan hash when set, like task
pools. To test again after the pack is used up, author a new holdout pack:
its manifest digest changes the evaluator hash and so defines a new holdout.

<a id="campaign-ownership-unreleased"></a>

### Campaign ownership

One live lease owns each campaign. `OptimizationRunner.run()` acquires a
unique owner identity and increasing epoch, renews it in a background thread,
and releases it after evaluator cleanup. Another runner for the same campaign
receives `CampaignLeaseConflict` before invoking its evaluator. Independent
campaigns can run concurrently.

The default lease lasts 30 seconds and renews every 10 seconds. Programmatic
callers can set `lease_ttl_s` and `lease_heartbeat_s`; both must be finite,
positive durations, and renewal must occur more often than expiry. The CLI
uses the defaults and needs no additional flags. Renewal runs separately from
the event loop so synchronous statistical work can continue without blocking
the heartbeat.

Every trial transition and promotion decision checks the current owner,
epoch, and expiry inside its write transaction. Dispatch records retain their
owner permanently. A stale or foreign token cannot complete a dispatched
trial, mark it unknown, publish a decision, or release a successor's lease.
Expiry is checked after acquiring the SQLite write lock and again after
validation, before committing trial or decision changes.

Heartbeat failure cancels owned evaluator work. Cancellation drains that work
before stopping renewal and releasing ownership; repeated caller cancellation
does not abandon cleanup. Evaluators must still cooperate with cancellation.
Ownership cannot forcibly stop arbitrary synchronous code or undo a provider
request already sent.

The low-level ledger API now requires an explicit `lease=` token for
`prepare_trial`, both dispatch methods, `complete_trial`, `mark_trial_unknown`,
`append_trial`, and `append_decision`. Acquire it through
`acquire_campaign_lease(campaign_id, owner_id)`, renew through
`heartbeat_campaign_lease`, and release through `release_campaign_lease`.
`CampaignLease`, `CampaignLeaseError`, and `CampaignLeaseConflict` are public
exports. Use `OptimizationRunner` to manage this lifecycle automatically.
This required-token API change ships in 0.8.0; it is not part of the 0.7.0 API.

Leases fence Smythe's public mutation API on a local SQLite ledger. They do not
isolate untrusted code inside the runner process, protect against direct
database editing, or provide a distributed lock across copied database files.

### Holdout trust boundary

Candidate proposers receive contracts and public campaign evidence, not the
writable ledger or the engine's private holdout capability. Public inspection,
snapshots, and package exports expose the commitment only. The engine retrieves
secret material through an unexported identity capability after revalidating
the exact plan and candidate inventory.

This is an application boundary, not cryptographic isolation inside one Python
process. Code that can introspect trusted engine internals, an operator who can
read the SQLite file, or an operator who can attach a debugger can recover the
secret and is therefore inside the trusted computing base. Run an untrusted
proposal agent in a separate process or service with only the proposer-facing
inputs; do not hand it the ledger path or execute it in the runner process.

`asyncio` can time out a cooperative async evaluator, but cannot forcibly stop
synchronous Python that blocks the event-loop thread. The reference simulator
runs in a worker thread and checks its deadline during CPU loops; custom
evaluators must likewise cooperate, use bounded provider timeouts, or run in a
separately supervised process when hard termination is required.

## Append-only trial history

`ExperimentLedger` uses SQLite WAL mode with explicit `full` or `normal`
synchronous durability. The Python API defaults to `full`; the offline CLI
defaults to faster `normal`. Both preserve transaction atomicity across a
process crash, while `full` adds the stronger power-loss guarantee.

The public state-transition API atomically creates immutable campaign and
candidate-plan payloads, then adds trial preparations, trial events, and one
terminal promotion decision. Retrying an operation is idempotent where replay
is safe; reusing an identity with different bytes raises a conflict. Campaign
creation cannot leave an empty or partially registered plan. The atomic
dispatch claim is deliberately non-idempotent: the runner claims a prepared
trial exactly once before invoking its evaluator, so two dispatchers cannot
both claim the same trial.

The trial lifecycle is:

```text
prepared -> dispatched -> completed
                       \-> unknown
```

- `prepared` reserves the contract's full per-trial ceiling before external
  work may begin.
- `dispatched` records that the evaluator may have received the request.
- `completed` records finite metrics, Boolean gates, confirmed cost, duration,
  and optional artifact hashes.
- `unknown` records that a dispatched trial cannot be proven completed or
  unexecuted.

Snapshots separate confirmed spend, live reservations, and unknown exposure.
Promotion decisions are append-only records with a reason, non-empty completed
trial evidence, and a JSON-safe assessment payload. The ledger permits exactly
one terminal decision per campaign (an exact replay is idempotent), verifies
that development evidence covers the complete sealed inventory except declared
references, and requires the incumbent, the selected challenger, and each
reference to have the exact paired confirmation and, when confirmation passes,
holdout evidence prescribed by the contract. Decision
construction detaches and recursively freezes the assessment, then caches the
exact canonical bytes and identifier written to the ledger so later caller
mutation cannot change its evidence identity. Resume revalidates that decision
identity and full evidence inventory before reusing it.

Append-only describes transitions made through the public Python API; it is
not protection against an operator directly modifying the SQLite database.
Protect and back up the database as experiment evidence.

The database also stores each campaign's holdout secret. A writable open
creates a missing ledger file with owner-only permissions (`0600`), and
SQLite's `-wal` and `-shm` files inherit them. Existing ledger files
keep their current permissions; tighten them with `chmod 600` if a ledger
created by 0.8.0 or earlier should not be readable by other local users.

The ledger schema in 0.8.0 is version 4. It adds campaign leases, monotonic
owner epochs, and immutable dispatch ownership to v3's atomic plan sealing and
single terminal-decision constraint. Writable opening upgrades a v3 database
transactionally, preserving campaign and candidate payloads, trial events,
costs, decisions, holdout secrets, and plan identities. Read-only inspection
continues to support a closed v3 database without migration.

Stop v3 runners before upgrading. Migration installs a connection-version
barrier that rejects writes from already-open v3 connections, including cached
statements. A current writer also validates the barrier on reopening; failed
migration rolls back. Existing evidence is preserved rather than assigning
historical dispatches to a new owner. Versions 1 and 2 remain unsupported;
create a fresh ledger for those early development schemas.

## Unknown outcomes stop the campaign

An interruption before dispatch is safe: the prepared trial has not crossed
the external side-effect boundary. After dispatch, a timeout or process loss
is different. The provider may have finished and billed the work even when no
response was committed locally.

An active owner can record that failure as `unknown`. Once recorded, the
ledger:

- keeps the full reservation in unknown cost exposure;
- refuses to dispatch, complete, or reuse the same trial; and
- refuses to use a non-completed trial as promotion evidence.

A runner that has lost ownership cannot write that disposition. Its unresolved
`dispatched` trial keeps the full ceiling in `reserved_microusd`; a recorded
`unknown` trial keeps it in `unknown_exposure_microusd`. Both block new admission.

`OptimizationRunner` enforces the stop rule. An evaluator timeout, exception,
invalid outcome, or failed completion stops the campaign with
`OptimizationNeedsAttention`. It records the ambiguous trial as unknown when
ownership remains live and the ledger write succeeds. A failed disposition
write leaves the dispatch unresolved and reserved. Caller cancellation retains `CancelledError`;
heartbeat or ownership failure can raise `CampaignLeaseError` and leave the
dispatch unresolved. During parallel development the runner cancels and
awaits its sibling tasks, recording their dispositions only while it owns
the lease. It never silently retries a dispatched sibling.

On restart, a new runner must acquire ownership before using the campaign.
Completed trials are validated and reused. Prepared trials remain safe to
claim only when the campaign contains no unknown outcome or dispatch owned
by an earlier epoch. A previously dispatched trial with no durable terminal
event stops the runner as `OptimizationNeedsAttention`; it is not automatically
reclassified or dispatched again. Its full reservation remains in the ledger.
The low-level admission API enforces the same stop rule, including already
prepared sibling trials.

Investigate provider records and local artifacts before deciding how to
proceed; do not hide the ambiguity by using a new seed or phase name.

Lease expiry permits a new owner to acquire the campaign; it does not prove
that a previous external request was unexecuted or unbilled. There is no
automatic retry or reconciliation command for these ambiguous trials.

## Current boundaries

The following pieces remain future work:

- proposal interfaces for human or agent-generated candidates beyond the
  current deterministic concurrency grid;
- CLI adapters for general or provider-backed evaluators with conservative
  per-trial price ceilings;
- calibrated sample-size guidance and a multiple-comparison correction across
  development challengers; and
- cross-campaign comparison and human-approval surfaces.

The installed concurrency campaign is intentionally offline. Custom in-process
evaluators can use `OptimizationRunner`, but a live evaluator must supply its
own bounded side-effect and price-ceiling implementation.
