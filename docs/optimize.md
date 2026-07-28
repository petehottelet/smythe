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
| `smythe.optimize.contracts` | Versioned contracts, objectives, allowlisted candidate patches, canonical JSON, and content-derived identities |
| `smythe.optimize.concurrency` | Deterministic offline concurrency simulator and evaluator fingerprint |
| `smythe.optimize.statistics` | Direction-normalized paired comparisons, seeded bootstrap intervals, and promotion policy |
| `smythe.optimize.ledger` | Durable campaign, candidate, trial-event, cost-exposure, and decision history |
| `smythe.optimize.engine` | Bounded development, confirmation, and holdout orchestration with conservative recovery |

These modules are deliberately separable from proposal generation. A human,
an agent, or a deterministic grid can propose candidates, but every candidate
is reduced to the same small, immutable policy patch before evaluation.

The package is pre-1.0 and its API may change. The programmatic runner is
`OptimizationRunner` in `smythe.optimize.engine`.

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
Confirmation and holdout compare only the incumbent and the one selected
challenger, limiting both cost and repeated testing.

Independent implementation ceilings bound otherwise-valid contracts to 1,024
candidates, 256 parallel candidate series, 10,000 repetitions per split, and
100,000 trials. Bootstrap work is capped at 1,000,000 resamples and checks the
same monotonic deadline cooperatively. The installed concurrency command uses
tighter limits: 64 challengers, 64 parallel series, 100 repetitions per split,
100,000 bootstrap resamples, one million work items or concurrency units, and
ten million simulated item-trials per campaign.

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
| Confirmation | Re-evaluate the selected candidate against the incumbent | Candidate and incumbent use the same ordered, unique seeds | Candidate must pass the complete promotion policy |
| Holdout | Test the confirmed candidate on evidence not used for selection | Derive a separate paired seed set from secret material created only after the candidate plan is frozen | Candidate must pass again before promotion |

`compare_metric()` requires at least three equal-length baseline/candidate
observations and exact ordered seed equality. Improvements are normalized so a
positive delta is always better: candidate minus baseline for a maximized
metric, and baseline minus candidate for a minimized metric.

`assess_promotion()` promotes only when all of the following are true:

- every deterministic gate passes;
- every candidate aggregate satisfies its hard metric bounds;
- every secondary metric stays within its declared regression allowance; and
- the primary metric's seeded bootstrap lower confidence bound is strictly
  greater than `min_improvement`.

The bootstrap implementation uses Python's standard-library PRNG with an
explicit seed, so identical inputs produce identical evidence. Each campaign
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

The runner evaluates every policy on development seeds, removes challengers
that fail a gate, hard metric bound, primary mean-improvement threshold, or
secondary non-regression limit, and deterministically selects the viable
challenger with the best primary mean. It then evaluates that challenger and
the incumbent on matching confirmation seeds. Holdout runs only if
confirmation passes, and promotion requires both assessments to pass.

Do not repeatedly tune against the holdout set: once it influences a proposal,
it is no longer held out. The current policy also does not correct for
repeatedly testing many candidates, so the independent one-time holdout is an
important guard against selection bias.

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
`--min-improvement`, and `--bootstrap-resamples`. Use `--ledger PATH` to
override the default `~/.smythe/optimize.sqlite3`. The CLI defaults to
`--ledger-durability normal`; choose `full` when the evidence must also survive
host power loss rather than only a process crash.

Output includes the contract, evaluator, optimization-plan, and public holdout
commitment identities; development and paired statistical evidence; the
selected candidate (or null when none is development-viable); and a ledger
snapshot. The secret holdout material is never emitted. `recommended_patch`
appears only when confirmation and holdout both promote the candidate.
Rejection never emits a patch to apply.

The optimization plan hash binds the contract, incumbent, ordered challenger
set, policy hashes, evaluator hash, runner version, bootstrap configuration,
and required trial count. When `--campaign-id` is omitted, the durable campaign
identity is derived from that full plan. Repeating the same command reuses
completed trials; changing a candidate or bootstrap configuration creates a
different automatic campaign. The holdout commitment is created when a new
fully bound campaign is atomically created and is checked on every resume and
decision. Supplying `--campaign-id` does not weaken the binding: reopening it
requires an exact plan and candidate-inventory match.

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

### Programmatic runner

Custom in-process campaigns use the same bounded engine:

```python
from smythe.optimize import OptimizationRunner

runner = OptimizationRunner(
    contract,
    ledger,
    evaluate,
    evaluator_hash="sha256:<64 lowercase hex>",
    bootstrap_resamples=2_000,
)
result = await runner.run(incumbent, challengers)
```

The async evaluator receives an immutable `TrialContext`, including the
campaign's monotonic deadline, and must return a `TrialOutcome` containing
exactly the contract's metrics and gate inventory, signed-63-bit integer
micro-USD cost, and optional SHA-256 artifact hashes.
`OptimizationResult` carries the plan identity, development scores,
confirmation and holdout assessments, decision and trial keys, and ledger
snapshot.

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
that development evidence covers the complete sealed inventory, and requires
the selected challenger to have the exact paired confirmation and, when
confirmation passes, holdout evidence prescribed by the contract. Decision
construction detaches and recursively freezes the assessment, then caches the
exact canonical bytes and identifier written to the ledger so later caller
mutation cannot change its evidence identity. Resume revalidates that decision
identity and full evidence inventory before reusing it.

Append-only describes transitions made through the public Python API; it is
not protection against an operator directly modifying the SQLite database.
Protect and back up the database as experiment evidence.

The current ledger schema is version 3. It adds atomic full-plan sealing and a
single terminal-decision constraint. It deliberately rejects the unreleased
version-1 and version-2 Autotune schemas before mutating them; create a fresh
ledger rather than treating old development evidence as a v3 campaign.

## Unknown outcomes stop the campaign

An interruption before dispatch is safe: the prepared trial has not crossed
the external side-effect boundary. After dispatch, a timeout or process loss
is different. The provider may have finished and billed the work even when no
response was committed locally.

When the active runner observes that failure, it marks the trial `unknown`.
The ledger then:

- keeps the full reservation in unknown cost exposure;
- refuses to dispatch, complete, or reuse the same trial; and
- refuses to use a non-completed trial as promotion evidence.

`OptimizationRunner` enforces the stop rule. An evaluator timeout, exception,
invalid outcome, cancellation, or failure to commit a returned outcome after
dispatch marks that trial unknown and raises `OptimizationNeedsAttention`.
During parallel development the runner cancels and awaits its sibling tasks;
any sibling already past dispatch is also recorded conservatively rather than
silently retried.

On restart, completed trials are validated and reused, while prepared trials
remain safe to claim. A previously dispatched trial with no durable terminal
event stops the campaign as `OptimizationNeedsAttention`; it is not
automatically reclassified or dispatched again because another process may
still own it.

Investigate provider records and local artifacts before deciding how to
proceed; do not hide the ambiguity by using a new seed or phase name.

The runner does not yet hold a campaign-wide process lease. Run only one active
runner for a campaign; a competing process can observe the first process's
dispatched trial as ambiguous and stop without duplicating it.

## Current boundaries

The following pieces remain future work:

- proposal interfaces for human or agent-generated candidates beyond the
  current deterministic concurrency grid;
- CLI adapters for general or provider-backed evaluators with conservative
  per-trial price ceilings;
- campaign-wide leases for safe multi-process operation;
- calibrated sample-size guidance and repeated-comparison controls; and
- richer export, comparison, and human-approval surfaces.

The installed concurrency campaign is intentionally offline. Custom in-process
evaluators can use `OptimizationRunner`, but a live evaluator must supply its
own bounded side-effect and price-ceiling implementation.
