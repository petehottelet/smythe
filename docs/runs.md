# Inspect saved runs

`smythe runs` reports what a saved run did and whether its output may be
used, without resuming or changing it. It reads ordinary `Swarm` checkpoints
from a [`FileCheckpointStore`](checkpoint-format.md) directory, or durable
text workflows from a [`SQLiteWorkflowStore`](workflow-accounting.md) journal.
Execution, delivery acceptance and cost are reported separately: a run can
complete with its cost fully known and still have output that its
[acceptance policy](verifier.md#strict-delivery-acceptance) did not accept.

```bash
smythe runs list                                   # ~/.smythe/checkpoints
smythe runs --checkpoints ./checkpoints inspect 3f9c2d0e8b4a
smythe runs --journal smythe-runs.db --json inspect 3f9c2d0e8b4a
```

`--checkpoints DIRECTORY` (default `~/.smythe/checkpoints`) and
`--journal PATH` choose the store; give one or the other. `--json` prints one
compact JSON document, `{"ok": true, "command": "inspect", "inspect": {...}}`,
with structured errors in the same shape. Options may follow the subcommand.
`list` takes `--limit` (1-500, default 50) and `--offset`, and lists each
run's id, execution state, decision and next action; a run it cannot read is
listed with its error.

## Read-only

Inspection constructs no provider and needs no API key. A missing directory
or journal is reported and never created. A journal is opened read-only, so a
version 1 journal is [never upgraded](workflow-accounting.md#journal-upgrades)
and inspection works while a worker holds a run. Checkpoint files are only
read. Saved output is never printed: withheld output is identified by its
SHA-256 and size, and the text stays in the store.

## What `inspect` reports

| Block | Contents |
|---|---|
| `format` | Checkpoint version and the state features it `requires`; or journal version, durable checkpoint version and revision |
| `execution` | `state`, `blocked_reason`, and node counts by status. Ordinary states are `running`, `failed` and `completed`; durable runs add `planned` and `blocked`, with the journal's `run_status` and `checkpoint_status` |
| `delivery` | The delivery acceptance decision; see below |
| `accounting` | `cost_scope`, `total_cost_usd`, `max_budget_usd`, `cost_is_complete`, `cost_contains_estimates`, `unknown_cost_nodes`, the durable `workflow_accounting` balances, `unknown_exposure` and `unresolved` accounting that stops resume |
| `timing` | `elapsed_ms`, `phases_ms` and `waiting_ms` added over the saved [timing segments](checkpoint-format.md#timing-segments), with the segment count, first start and last end; null when none were saved. Durable runs report lease sessions on the journal's wall clock and no phase totals |
| `next_action` | The first thing the run needs, as an `action` code and a `detail` sentence |

Ordinary checkpoints record no reservations, so `unknown_exposure` is null for
them rather than zero: a call in flight when the process stopped, on a node
still `running` in the checkpoint, may have been charged without a recorded
cost. Durable runs report their exact `unknown_exposure` in nanoUSD with the
number of calls that hold it.

### Delivery acceptance

| Field | Meaning |
|---|---|
| `recorded` | Whether the run has a [delivery acceptance record](verifier.md#delivery-acceptance-records) |
| `decision` | The decision that holds for the saved output now: `not_evaluated`, `accepted`, `rejected`, `needs_review`, `indeterminate`, or null without a record |
| `reasons` | The reasons for that decision |
| `integrity` | `verified` when the saved output, task and source node generations still match what the record bound, and for a durable run the journal applied the same decision; `mismatch` otherwise; null when nothing was decided |
| `output_delivered` | Whether resuming the run returns its output as `output` |
| `rejected_output` | For withheld output: the `field` that holds it, its `sha256` and its size in `bytes` |
| `record` | The record as saved, including its validators' names, versions, configuration digests and outcomes |

Checkpoints before version 5 and runs without a strict policy have no record:
`decision` is null and the output is delivered as before. A legacy run is never
shown as accepted or as not yet evaluated. As on [resume](checkpoint-format.md#delivery-acceptance-version-5),
a decided record whose output, task or source generations no longer match
gives `indeterminate` with the mismatch as its reason, whatever it recorded;
an ordinary checkpoint does not hold its acceptance policy, so inspection does
not check its policy digest.

A durable run under a strict policy keeps its record in
[checkpoint version 2](verifier.md#strict-acceptance-in-durable-workflows),
and inspection reads it as resume does. The checkpoint version must be the
one the run's recipe writes, and the record must be undecided until the run
completes and belong to the policy the recipe records, or the run is reported
as unreadable. A decided record is checked as resuming the completed run
checks it: against the saved output, the task and the source generations,
and against the decision the journal applied, so a record changed to match
other output also gives `indeterminate`. Durable runs without a strict
policy, including runs with only a gate validator, have no record.

### Next actions

| `action` | When |
|---|---|
| `none` | The run completed, its output is delivered and its cost is known |
| `resume` | The run is unfinished; `detail` says how to resume it, including the strict policy's `sha256` that must resume a strict run, ordinary or durable |
| `reconcile_accounting` | Accounting is unresolved (invalid usage, an accounting error, an unresolved native response, or a durable run blocked by `unknown_exposure` or `evidence_conflict`), or a completed run's cost is incomplete |
| `rerun` | A durable run passed its budget and admits no more calls |
| `review_output` | The output was `rejected` or `needs_review` and is withheld |
| `resolve_indeterminate` | No valid decision covers the output, which is withheld |

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Inspected; the run has no decision that withholds its output. It may still be unfinished: read `execution.state` |
| 2 | Invalid argument, such as both `--checkpoints` and `--journal`, an invalid run id, or a `list` page out of range |
| 6 | The store has no run with that id |
| 7 | The store or the run cannot be read: missing, foreign, corrupt, written by a newer release, or a durable checkpoint that resume would refuse |
| 10 | The run completed and its output was `rejected` or `needs_review` |
| 11 | The decision is `indeterminate` |

`list` returns 0 whatever the runs' decisions. Codes 10 and 11 are used only
by `runs inspect`; the [Jobs codes](jobs.md#cli-exit-codes) are unchanged.

## From Python

`SwarmResult.to_dict()` includes the `execution`, `delivery`,
`unknown_exposure` and `next_action` blocks for a finished result, built the
same way. Its withheld output's `field` is `rejected_output`, where the result
keeps the text.
