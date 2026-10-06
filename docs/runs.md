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
run's id, execution state, decision and next action; a run it cannot read,
such as a checkpoint entry it cannot open or JSON nested too deeply to parse,
is listed with its error and the listing continues.

## Read-only

Inspection constructs no provider and needs no API key. A missing directory
or journal is reported and never created. A journal is opened read-only, so a
version 1 journal is [never upgraded](workflow-accounting.md#journal-upgrades)
and inspection works while a worker holds a run. Checkpoint files are only
read. Saved output is never printed: withheld output is identified by its
SHA-256 and size, and the text stays in the store.

`SQLiteWorkflowStore` keeps its journal in SQLite's WAL mode. Reading a WAL
journal, even read-only, can create its `-wal` and `-shm` sidecar files, and
they may stay beside the journal afterwards: SQLite uses them to read the
journal consistently while a worker writes it. The journal's database file,
its runs and its recorded version do not change.

## What `inspect` reports

| Block | Contents |
|---|---|
| `format` | Checkpoint version and the state features it `requires`; or journal version, durable checkpoint version and revision |
| `execution` | `state`, `blocked_reason`, and node counts by status. Ordinary states are `running`, `failed` and `completed`; durable runs add `planned` and `blocked`, with the journal's `run_status` and `checkpoint_status`. An ordinary run that a [security stop](execution.md#security-stops) ended has `blocked_reason` `security_stop` |
| `delivery` | The current delivery acceptance decision; see below |
| `decisions` | The [decision history](#decision-history): the current decision's sequence and the decisions before it |
| `revision_log` | The run's latest [supervisor decisions](#revision-log), applied or refused; empty without any |
| `artifact` | A durable run's [published document](#published-documents); null for a run that publishes none and for ordinary runs |
| `lineage` | A summary of a durable run's [lineage](#lineage): coverage, source references, predecessor and successors, document integrity and derived status; null without a lineage policy and for ordinary runs |
| `context` | A durable run's [context assembly](#research-runs): versions, current sources, snapshots and tombstones; null without context assembly and for ordinary runs |
| `tool_calls` | A durable run's [journaled reads](#research-runs), without response bodies or text; null for ordinary runs |
| `evidence` | The [source evidence](#research-runs) a completed durable run delivered, summarized; null without any and for ordinary runs |
| `issues` | What the run leaves open: unresolved questions, unavailable sources, failed reads and final validators that did not pass |
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
| `integrity` | `verified` when the saved output, task and source node generations still match what the record bound, and for a durable run the journal applied the same decision and any [published document](#published-documents) still holds the decided bytes; `mismatch` otherwise; null when nothing was decided |
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
completes and belong to the policy the recipe records, or to a policy a
re-evaluation decided under, or the run is reported as unreadable. A decided
record is checked as resuming the completed run checks it: against the saved
output, the task and the source generations, and against the journal's
current decision, so a record changed to match other output also gives
`indeterminate`. Durable runs without a strict policy, including runs with
only a gate validator, have no record.

### Decision history

A completed run can be decided again with
[`Swarm.reevaluate`](verifier.md#re-evaluation), under a policy or with an
[operator's decision](verifier.md#operator-decisions), which never runs a
node; an operator's record carries an `operator` block. The
`delivery` block and the exit code always follow the current decision, and
the `decisions` block lists what came before it:

| Field | Meaning |
|---|---|
| `current` | The current decision's sequence: 0 for the decision the run made when it completed, one more for each re-evaluation or operator decision; null without a decision |
| `history` | The decisions the current one superseded, oldest first, each with its `sequence` and `record` |

An ordinary checkpoint keeps the last 16 superseded records
([checkpoint format](checkpoint-format.md#decision-history)), so `history`
may start after sequence 0. A durable run's history comes from its journal:
every `acceptance` operation a checkpoint applied, ordered by sequence, the
last of which is the current decision. A run never decided again has
`current` 0 and an empty `history`, and the text report lists earlier
decisions only when there are any:

```text
Delivery acceptance: accepted
Decision history: decision 1 is current; earlier decisions:
  Decision 0: rejected under policy 8a51c0e2d9f4: final validator 'required-section' rejected the output: section_missing: no '## Sources' section
```

### Revision log

`revision_log` lists up to 32 of a run's latest supervisor decisions, oldest
first, each with what the revision proposed, whether it was `applied`, and
what it left: the completed results it `retained`, the delivered results it
`invalidated`, the gates it preserved, the revisions left and the
revision-added nodes in the graph
([what a revision changed](supervisor.md#what-a-revision-changed)). An
ordinary run's log is the one its checkpoint saves
([format](checkpoint-format.md#revision-log)). A durable run's journal keeps
none, so inspection derives it from the run's `supervision` operations; their
refusals carry no `detail`. The supervisor's `reason` and a refusal's
`detail` are at most 200 characters and are printed as data:

```text
Revisions: 2 logged
Revision after draft: applied (+1); added check; 1 revisions left, 1 added nodes in the graph
  Reason: the answer needs a check
Revision after final: refused (-1); dropped draft; 1 revisions left, 1 added nodes in the graph
  Reason: the draft is done
  Detail: cannot drop node 'draft' with status completed; only pending work may be revised
```

### Published documents

A durable run with a [renderer](workflow-accounting.md#published-documents)
reports its current publication under `artifact`, read through the same
accessor resume uses:

| Field | Meaning |
|---|---|
| `publication` | The `PublishedArtifact` record: `sequence`, `directory`, `name`, `sha256`, `size_bytes`, `media_type`, the `renderer` description, and the `synthesis` it rendered with that text's `output_sha256` |
| `path` | Where the published file is |
| `decision` | The delivery decision that covers it: the run's current decision, a re-evaluation's after one; or null |
| `integrity` | `status` and `detail` from rechecking the file's size and SHA-256: `verified`, `missing`, `changed`, `unsafe` or `unreadable` |
| `delivery_eligibility` | `eligible`, and `reasons` for every unmet condition: `no_decision`, `not_accepted`, `record_mismatch`, `artifact_missing`, `artifact_changed`, `artifact_unsafe`, `artifact_unreadable` |
| `superseded` | The earlier publications a republication superseded, oldest first, each as `sequence`, `name` and `sha256`; empty when the document was published once. The text output lists them on a `Supersedes:` line |

The file is only read. As on resume, a decided record whose published file is
missing or changed gives `indeterminate` with that reason, and the next action
is `republish`.

### Lineage

A durable run with a [lineage policy](lineage.md) reports a summary of its
[lineage record](lineage.md#the-record-a-run-writes) under `lineage`, by
ids, versions and digests, never source text, node inputs or results:

| Field | Meaning |
|---|---|
| `access_scope` | The scope the run's policy declares |
| `recorded` | Whether the run has written its record, which it does when it completes |
| `lineage_sha256`, `coverage` | The record's digest, and `complete` or `partial`; null before the record |
| `nodes`, `undeclared` | The record's nodes counted by kind, and the ids of the nodes whose inputs are undeclared |
| `sources` | Each source reference the nodes read, once, by `source_id`, `version` and `sha256` |
| `artifact` | The published document the record binds, by `relative_path`, `sha256` and `size_bytes`, and its `integrity`: the status of the run's current publication when it holds those bytes, `changed` when it holds others, `not_checked` without one; null for a run that publishes none |
| `predecessor` | The run this run's record names as its predecessor, with that run's record digest; null for none |
| `successors` | Each run in the same access scope whose record names this run as its predecessor, and whether it `supersedes` it: only a completed successor whose current decision is `accepted` does |
| `tombstones` | Each tombstone that withdraws a source text the record names, by `source_id`, `version`, `sha256`, `reason` and `removed_at` |
| `purged` | Each published file a [restriction](lineage.md#restrict-a-source) deleted, by `relative_path`, `sha256`, `reason` and `purged_at` |
| `status`, `reasons` | The run's [derived status](lineage.md#derived-status) and the reason codes behind it |

Freshness depends on the sources, which only the application's source
catalog can describe, so `status` here is `superseded`, `restricted` (a
source the record names was [restricted](lineage.md#restrict-a-source) or
[removed from the run](tasks.md#remove-a-source)),
`invalid` (the document no longer holds the decided bytes, or the current
decision is not `accepted`) or null, never `current` or `stale`; plan a
repair with [`Swarm.plan_repair`](lineage.md#plan-a-repair) to see those.
The summary reads every run in the run's access scope to find its
successors; `list` leaves it out. It adds no exit code. A restricted run's
next action is never `republish`, which it refuses.

```text
Lineage: complete coverage; scope research-team; nodes model 2, read 2, synthesis 1; record sha256 5f0c...
  Source renewals version 1: sha256 7d1e0f3a9b2c
  Source survey version 1: sha256 26e6faa9bc6d
  Artifact 3f9c2d0e8b4a/document.md: sha256 b84c1d2e3f40; integrity changed
  Predecessor: none; successors: none
  Derived status: invalid (artifact_changed, decision_not_accepted)
```

A restricted run adds a line for each tombstone and purged file:

```text
  Tombstone renewals version 1: sha256 7d1e0f3a9b2c; withdrawn by its owner (2026-10-01T09:00:00Z)
  Purged 3f9c2d0e8b4a/document.md: sha256 b84c1d2e3f40; withdrawn by its owner (2026-10-01T09:00:00Z)
```

### Research runs

A durable run that reads sources and delivers evidence reports what it was
given, what it read and what its output claims, without printing source
text, read bodies, quotes or output. The
[source-to-artifact example](../examples/16_source_to_artifact.py) prints
one such report.

`context`, for a run with [context assembly](tasks.md#context-snapshots):

| Field | Meaning |
|---|---|
| `assembly_version`, `snapshot_version` | The context-assembly version the recipe records and the version of its snapshot records |
| `sources_revision`, `sources_sha256` | The source revision the latest checkpoint names, and its digest |
| `sources` | Each current source's `source_id`, `version`, `sha256`, `text_bytes` and `trust` (`untrusted`); a [removed](tasks.md#remove-a-source) source has `removed` with its tombstone's `reason` and `removed_at`, and no size |
| `snapshots` | Each node's current snapshot by `node_id`, `generation` and `snapshot_id` |
| `tombstones` | Every source removed from the run, as tombstone records |

`tool_calls` lists each [journaled read](workflow-accounting.md#recorded-read-steps):
its `node_id`, `generation`, `attempt` and `invocation`; the `profile`,
`profile_version` and `decoder_version`, whether it is `metered` and
`repeat_safe`; its `url`, with account identifiers redacted, and
`request_sha256`; its `state`, `billing_state`, `result_state` and
`cost_nanousd`; the `http_status`, `transport_error` and `response_sha256`
of its response; the `text_bytes` and `text_sha256` of the text decoded from
it; and its rejection `error`. Bodies and text stay in the journal.

`evidence` summarizes [`load_source_evidence`](source-evidence.md#inspect-evidence):
the operation's key and state, the output digest, the run's source set, any
`problem`, each cited source's summary with `available` and `removed`, and
each claim's id, the number of `supporting` and `contradicting` passages,
the `sources` they cite, its `scope` and its `unresolved` questions.

`issues` lists what is open, each with a `kind`, a `subject` and a `detail`:

| `kind` | When |
|---|---|
| `evidence_problem` | The output carried no readable evidence |
| `source_unavailable` | A source the evidence cites has no text the run holds: removed, tombstoned or not retained |
| `unresolved_question` | A claim leaves a question open; the subject is the claim |
| `read_failed` | A read step has no accepted read at its generation, or a read's charge is unknown |
| `validator_rejected`, `validator_error` | A final validator in the decision record did not pass; the detail is its reason |

The text report prints each block on its own lines:

```text
Context: assembly version 1, snapshot version 1; sources revision 0; 2 sources, 1 snapshots
  Source survey-2019 version 2019: 79 bytes; sha256 26e6faa9bc6d
Read fetch-survey-2019 (generation 0, attempt 0): source 1; settled, applied; HTTP 200; 79 bytes of text; cost 0 nanoUSD
Evidence: 1 claims citing 2 sources (applied)
  Claim billing-preference: 1 supporting, 1 contradicting; sources renewals-2026, survey-2019
Issue unresolved_question (billing-preference): Whom did the 2019 poll ask?
```

A source removed after the run's decision makes that decision
`indeterminate`, with the removal as its reason and exit code 11, until a
re-evaluation decides again.

### Next actions

| `action` | When |
|---|---|
| `none` | The run completed, its output is delivered and its cost is known |
| `resume` | The run is unfinished; `detail` says how to resume it, including the strict policy's `sha256` that must resume a strict run, ordinary or durable, and the retry policy an ordinary run with [persisted attempts](execution.md#persisted-attempts) records |
| `republish` | A completed durable run's published document is missing or changed; `Swarm(run_store=...).republish(run_id)` publishes it again without replacing a file |
| `reconcile_accounting` | Accounting is unresolved (invalid usage, an accounting error, an unresolved native response, or a durable run blocked by `unknown_exposure` or `evidence_conflict`), or a completed run's cost is incomplete |
| `rerun` | A [security stop](execution.md#security-stops) ended the run, ordinary or durable, or a durable run passed its budget: it admits no more calls |
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

Codes 10 and 11 follow the run's current decision, after any
[re-evaluation](#decision-history). `list` returns 0 whatever the runs'
decisions. Codes 10 and 11 are used only
by `runs inspect`; the [Jobs codes](jobs.md#cli-exit-codes) are unchanged.

## From Python

`SwarmResult.to_dict()` includes the `execution`, `delivery`,
`unknown_exposure` and `next_action` blocks for a finished result, built the
same way. Its withheld output's `field` is `rejected_output`, where the result
keeps the text.
