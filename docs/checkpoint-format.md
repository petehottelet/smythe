# Checkpoint format (versions 4 and 5)

This guide describes the ordinary `checkpoint_store` path. Opt-in
[`run_store` workflows](workflow-accounting.md) use a separate SQLite journal
with call evidence and exact accounting. Their graph exports carry a strict
`run_ref` containing store/run IDs and the recipe hash; ordinary execution and
checkpoint resume reject that reference instead of dropping its ledger.

When a `Swarm` is constructed with a `checkpoint_store`, it persists the full
execution state after planning and once more when the run finishes or fails.
By default (`checkpoint_every_n_nodes=1`) it also saves after every node reaches
a terminal status. Wide graphs may set a larger interval to reduce full-snapshot
write amplification; after a process crash, at most the completed but unflushed
tail of that batch may replay. That replay can duplicate provider spend or tool
side effects, so intervals above one are appropriate only when the write-saving
tradeoff is worth the recovery window and affected operations are idempotent.
Verification decisions and regeneration transitions always force a save,
regardless of the node interval. The state is a single JSON document per execution, so you can inspect—or
repair—a checkpoint with any text editor.

With the default `FileCheckpointStore`, checkpoints live at
`~/.smythe/checkpoints/<execution_id>.json`. Each save writes its own exclusive
temporary file in that directory, flushes its bytes, and atomically replaces
the checkpoint. Independent store instances cannot overwrite each other's
temporary files. Failed writes before replacement preserve the previous
checkpoint and clean up only the current save's temporary file. Checkpoints
contain prompts and results, so each temporary file, and the checkpoint that
replaces the previous one, is readable and writable by its owner only (mode
0600 on POSIX).

POSIX systems also flush the directory entry after replacement. Windows flushes
the file; Python's standard library cannot separately flush its directory
entry. A directory-flush failure raises after replacement, so the new complete
checkpoint may already be visible. Hard process termination can leave an
unused temporary file; `list_ids()` excludes those files.

The last completed replacement determines the saved snapshot. This file store
does not coordinate competing workflow owners or guarantee their logical
ordering. Use the [durable text journal](workflow-accounting.md) or
[Jobs leases](jobs.md#jobs-database-upgrades) for their documented ownership
and recovery controls.

## Schema

```json
{
  "version": 4,
  "execution_id": "9f2c4a…",
  "status": "running | completed | failed",
  "created_at": 1751600000.0,
  "updated_at": 1751600042.5,
  "model": "claude-opus-5-5",
  "task": {
    "goal": "…",
    "constraints": ["…"],
    "done_when": ["…"],
    "context": {}
  },
  "control": {
    "revisions_used": 0
  },
  "graph": {
    "task": {
      "goal": "…",
      "constraints": ["…"],
      "done_when": ["…"],
      "context": {}
    },
    "topology": ["fork_join"],
    "estimated_cost_usd": 0.02,
    "nodes": [
      {
        "id": "research",
        "label": "Research the topic",
        "agent_id": "a1b2c3d4e5f6",
        "depends_on": [],
        "result": "…node output, null until executed…",
        "status": "completed",
        "metadata": {"model": "claude-opus-5-5", "cost_usd": 0.0003},
        "failure_policy": "halt",
        "max_retries": 1,
        "required_capabilities": [],
        "timeout_s": null,
        "max_tool_iterations": null,
        "attach_dep_artifacts": false,
        "verifies": null,
        "max_regenerations": 0
      }
    ],
    "planning": {
      "routing": {"version": 1, "tier": "constrained", "method": "classifier", "…": "…"}
    }
  },
  "agents": [
    {"id": "a1b2c3d4e5f6", "name": "Researcher", "persona": "…", "capabilities": ["research"]}
  ],
  "budget": {
    "max_budget_usd": 0.5,
    "node_costs": {"research": 0.0003},
    "token_estimate_nodes": ["research"]
  },
  "output": "…final synthesized output, null until the run completes…",
  "timing": {
    "version": 1,
    "clock": "monotonic",
    "segments": [
      {
        "started_at": 1751600000.0,
        "ended_at": 1751600042.5,
        "resumed": false,
        "elapsed_ms": 42500.0,
        "summed_node_ms": 61200.0,
        "phases_ms": {"routing": null, "planning": 3100.0, "execution": 52000.0, "…": "…"},
        "waiting_ms": {"retry_backoff": 0.0, "admission": 900.0, "supervision": 0.0, "capacity": 0.0}
      }
    ]
  }
}
```

Notes:

- `task` and `graph.task` preserve the submitted task through an inspected-graph
  handoff. Both are `null` for a caller-built graph that carries no task.
- Node `result` values that aren't JSON-serializable are stored as their `str()` form.
- `budget.max_budget_usd` is the cap the execution started with; resume honors it, not whatever the resuming Swarm was constructed with.
- `budget.node_costs` records costs, never reservations. A node whose sent call never answered under a retry policy with `retain_uncertain_dispatch` holds that call's reservation here as recorded cost, with `cost_usd_unknown` and `cost_usd_is_estimate` in its `metadata`, so resume restores the exposure without a new version ([calls without an answer](execution.md#calls-without-an-answer)).
- `budget.token_estimate_nodes` lists the nodes, including `__synthesis__` for an `LLM_MERGE` synthesis, whose recorded cost prices tokens at the blended `cost_per_token` rate. Resume restores them, so `Sentinel.cost_contains_token_estimates` still reports costs priced before the checkpoint. A value other than a list of node IDs stops resume with `BudgetValidationError`.
- `control.revisions_used` is how much of the supervisor's `max_revisions` allowance the run has already spent. Resume seeds the executor from it, so the cap bounds the **run**, not each attempt: a crash-resume cycle cannot refill the allowance and revise past the limit the caller set.
- `control.retry_policy` and `control.attempts_used` appear only for a run whose retry policy persists attempts, which makes the checkpoint version 5; see [attempt allowance](#attempt-allowance-version-5).
- `control.revision_log`, present once a supervisor made a decision, lists the run's latest supervisor decisions; see [revision log](#revision-log).
- A node that a revision added carries `"added_by_revision": true` in its `metadata` (from 0.8.2). `LLMSupervisor` counts these nodes against `max_total_added_nodes`, so resume does not reset the count ([what the cap limits](supervisor.md#safety)). Nodes added before 0.8.2 have no marker and are not counted.
- `task.done_when` carries the acceptance criteria forward. A resumed run that had forgotten them could not hold its own output to them.
- `timing` records how long each call that worked on the execution took; see [timing segments](#timing-segments).
- `graph.planning`, present only when a component recorded how the graph was planned, holds informational records keyed by namespace; see [graph planning records](#graph-planning-records).
- A version 5 checkpoint adds a top-level `requires` list and the state it names, such as a `delivery_acceptance` record, an attempt allowance or nodes that name model profiles; see [required state features](#required-state-features-version-5).
- A completed version 5 checkpoint that was [re-evaluated](verifier.md#re-evaluation) adds `delivery_acceptance_history`, the decisions its current record superseded; see [decision history](#decision-history).

## Timing segments

Each `execute()` or `resume()` call that saves the checkpoint writes one
segment to the top-level `timing` block and carries the earlier segments
forward, so a run that crashed and was resumed twice has three. A segment
holds:

- `started_at` and `ended_at`: wall-clock seconds, like `created_at`;
- `resumed`: whether the call resumed this checkpoint;
- `elapsed_ms`, `summed_node_ms`, `phases_ms` and `waiting_ms`: that call's
  [timing](execution.md#timing), measured on its own process's monotonic clock.

Monotonic readings from different processes are never subtracted from each
other, so durations are comparable only within a segment. The gap between two
segments, when no process was running the execution, shows only in their
wall-clock times, which clock adjustments can shift. The block is
informational: when it is missing or unreadable, resume starts a new list, and
it does not change the checkpoint version.

## Graph planning records

`graph.planning` is an object keyed by namespace, and each component writes
only its own key. A `Swarm` with a `WhiteRabbit` router writes `routing`, the
[routing decision](architecture.md#routing-decisions) that chose the
architect:

```json
"planning": {
  "routing": {
    "version": 1,
    "tier": "constrained",
    "method": "classifier",
    "alternatives": ["deterministic:faq", "constrained", "autonomous"],
    "reasons": ["the classifier chose constrained"],
    "policy_version": 1,
    "matched_token": "constrained",
    "usage": {"prompt_tokens": 412, "completion_tokens": 3, "cost_usd": null,
              "cost_usd_is_estimate": false, "cost_usd_unknown": false},
    "call_ids": []
  }
}
```

The key is absent when nothing recorded a plan, so such a graph is saved as
before. Resume restores the object unchanged and never needs it: the records
are informational, so the checkpoint stays version 4. A value other than an
object stops resume with `ValueError`. `smythe.RoutingDecision.from_dict()`
reads the `routing` record back and rejects a version, method or key it does
not know.

When the [contribution lint](architecture.md#contribution-lint) finds
parallel siblings in a routed plan that could only repeat each other's work, it
writes `coordination` beside `routing`, one list of node ids per group:
`{"version": 1, "duplicate_siblings": [["research-0", "research-1"]]}`. A plan
it does not flag has no `coordination` key.

## Revision log

Once a [supervisor](supervisor.md#revision-log) has made a decision, applied
or refused, `control.revision_log` lists the run's latest 32 decisions, oldest
first:

```json
"control": {
  "revisions_used": 1,
  "revision_log": [
    {
      "after_node": "draft",
      "applied": true,
      "change": "+1",
      "added": ["check"],
      "dropped": [],
      "rewired": [],
      "retained": ["draft"],
      "invalidated": [],
      "preserved_gates": [],
      "revisions_remaining_after": 1,
      "added_total": 1,
      "trigger": "node_completed",
      "reason": "the answer needs a check",
      "detail": null
    }
  ]
}
```

The keys are those of the trace's [revision record](supervisor.md#what-a-revision-changed)
with `applied`, the supervisor's `reason` and a refusal's `detail`, each at
most 200 characters with account identifiers redacted. Entries hold node ids,
never labels or results; each id list keeps its first 16, and `omitted_ids`,
present only then, counts the ids left out. Resume continues the saved log, so
it covers decisions made before a crash although the resumed run's trace does
not.

The log is informational: it does not change the checkpoint version or what
resume does, and it is not a [required state feature](#required-state-features-version-5).
A run without decisions saves no key. Resume and `smythe runs inspect` read a
missing log, or one whose entries lack a key or hold a value of the wrong type,
as empty. `smythe.checkpoint.revision_log_from_state(state)` reads it back.

## Verification state

Node metadata carries the generation and disposition of each gating verdict:

- `execution_generation` identifies the node's output generation; absent means `0`.
- `verification_target_generation` binds a judge's request to the target it inspected.
- `verification_receipt` records version `1`, state `pending` or `consumed`,
  `judge_generation`, `target_id`, and `target_generation`. A completed gated
  judge saves its pending receipt before ordinary completion can be checkpointed.
- `regeneration_intent` records version `1`, the target and judge generations,
  rejection reason, absolute `regenerations_used` count, and an
  `affected_generations` map from node IDs to their old generations.

Under a [strict acceptance policy](verifier.md#strict-delivery-acceptance),
which writes version 5, a consumed receipt also records the final verdict:
`outcome` (`passed`, `rejected` or `error`), `passed`, `reason`, and
`exhausted: true` for a failing verdict after the last regeneration. A gate
with `max_regenerations=0` records its verdict under `advisory_verdict` in the
judge's metadata, with version `1`, `judge_generation`, `target_id`,
`target_generation`, `outcome`, `passed` and `reason`; such a judge also
records `verification_target_generation`. Each attempt of the judge drops the
previous `advisory_verdict` when it starts, so a resumed run keeps the record
only for a judge that completed before the crash. A gate that a strict run's
generated plan contains carries `generated_gate: true`, and the saved mark
keeps it from [accepting output on its own](verifier.md#which-gates-count)
after resume.

The intent is saved before cancellation. After affected workers settle, a
second forced save records the reset: results and stale artifact references
are cleared, affected generations advance exactly once, and the intent is
removed. Costs and consumed regeneration allowances survive the reset.
Detached snapshot values prevent later metadata mutations from changing a
checkpoint already handed to a custom store.

Under a strict acceptance policy the reset first adds a record of each
discarded generation to the node's `rejected_generations` list
([rejected generations](verifier.md#rejected-generations)). A record has
version `1`, `generation`, `status`, `judge_id`, `reason` (at most 300
characters), `result_sha256` (`null` without a result), `result_bytes`, an
`excerpt` of at most 1,024 bytes of the result's UTF-8 text, a `sidecar` path
to the full text when the result is longer and the run has an
`artifact_dir`, and the generation's `artifacts` records or
`artifacts_discarded` count when it had files. The records of one checkpoint
add at most 64 KiB as `FileCheckpointStore` writes them; past that a record
has no `excerpt`, and one that still does not fit is left out. A replayed
reset records each generation once. A new durable workflow refuses a graph
whose nodes carry `rejected_generations`, as it refuses other execution
history.

## Required state features (version 5)

Version 5 is written only for a state that a build reading version 4 would
misread. Every version 5 checkpoint carries a top-level `requires` list naming
the state features its reader must understand:

```json
"requires": ["delivery_acceptance"]
```

This build knows four features: [`delivery_acceptance`](#delivery-acceptance-version-5),
[`attempt_allowance`](#attempt-allowance-version-5),
[`model_profiles`](#model-profiles-version-5), and `operator_decision`,
which a completed state lists while its record or a kept
[earlier decision](#decision-history) is an
[operator's decision](verifier.md#operator-decisions). A state lists those it
needs in that order. A reader refuses, with
`ValueError` and before any provider call, a version 5 checkpoint whose
`requires` is missing or is not a list of strings, or that lists a feature it
does not know, whether the run finished or not. A later feature can therefore
be added under version 5 without an older reader finishing or delivering a
run it would misread. `smythe.checkpoint.checkpoint_requirements(state)`
returns the list after these checks.

## Delivery acceptance (version 5)

A state that carries a [delivery acceptance record](verifier.md#delivery-acceptance-records)
is written as version 5, lists `delivery_acceptance` in `requires`, and keeps
the record in a top-level `delivery_acceptance` block. A run under a
[strict acceptance policy](verifier.md#strict-delivery-acceptance) writes
every checkpoint this way: while it is unfinished the record is
`not_evaluated` and binds the policy and task, and the completed checkpoint
holds the decision. Other runs make no record and keep writing version 4;
`build_state(..., delivery_acceptance=record)` writes version 5 and refuses a
record whose `output_sha256` does not match the state's `output`, or a
`completed` state whose record has no decision.

The record is required exactly when it is listed: a version 5 state that
lists `delivery_acceptance` without the block, or carries the block without
listing it, is refused, and so is a `completed` state whose record is
`not_evaluated`.

```json
"delivery_acceptance": {
  "version": 1,
  "decision": "accepted",
  "reasons": [],
  "output_sha256": "3f1c…",
  "strategy": "deliverable",
  "sources": [{"node_id": "report", "execution_generation": 1}],
  "used_fallback": false,
  "task_sha256": "9a0e…",
  "policy_sha256": "8a51…",
  "validators": [
    {
      "validator": {"name": "required-sections", "version": "1", "config_sha256": "c2d4…"},
      "outcome": "passed"
    }
  ],
  "scope": "output"
}
```

- `version` is the record format. A reader rejects a record version, a
  `decision` or a key it does not know instead of guessing what it means.
  A policy's decision is version 1. An operator's decision is version 2,
  which adds `operator`: the verified principal's `issuer`, `subject` and
  `tenant`, the approval `verifier`'s identity and the `evidence_sha256` the
  operator decided on. It is `accepted` or `rejected`, lists no validator
  outcomes, and is held only by a `completed` state that lists
  `operator_decision` in `requires`.
- `decision` is `not_evaluated`, `accepted`, `rejected`, `needs_review` or
  `indeterminate`. Every decision except `not_evaluated` binds
  `output_sha256`, the SHA-256 of the checkpoint's `output` encoded as UTF-8,
  and the synthesis `strategy`. Every decision except `accepted` and
  `not_evaluated` lists at least one entry in `reasons`.
- `sources` are the nodes the output came from, each with the
  `execution_generation` used. `used_fallback` is true when synthesis fell
  back to a node that was not a terminal deliverable.
- `task_sha256` is the SHA-256 of the task snapshot as canonical JSON, or
  `null` without a task. `policy_sha256` identifies the acceptance policy.
- `validators` lists each final validator's identity and outcome (`passed`,
  `rejected` or `error`). An `accepted` record holds only `passed` outcomes.
- `scope` is `output`: the decision covers the delivered output, meaning the
  synthesized text and any files its source nodes wrote.
- `artifacts`, present only when source nodes delivered files, lists each
  file whose bytes were verified: `node_id`, file `name`, `sha256`,
  `size_bytes` and `mime_type`.

Versions 1 through 4 have no record, whatever keys they hold. Resuming one
returns `acceptance` as `None`: a legacy run is never reported as accepted or
as not yet evaluated. A `Swarm` with a strict policy that resumes a completed
checkpoint without a record, including a version 5 one that does not list
`delivery_acceptance`, returns `output=""` with the saved text in
`rejected_output`, because nothing decided it
([strict delivery acceptance](verifier.md#strict-delivery-acceptance)).

Resuming a completed version 5 checkpoint checks that `output` still matches
`output_sha256`, that the task and every source node's generation are the ones
the record binds, that every file in `artifacts` still holds its recorded
bytes, and, when the resuming `Swarm` has a strict policy, that its
`sha256` is `policy_sha256`. The result's `acceptance` is the record, or, after
any mismatch, an `indeterminate` record for the saved output that names the
mismatch: changed output is never accepted. An `accepted` record delivers the
stored output; for any other decision `output` is `""` and the text is in
`rejected_output`. An unfinished version 5 checkpoint resumes only under a
strict policy whose `sha256` is its record's `policy_sha256`, and is refused
with `ValueError` before any provider call otherwise, because finishing it
under another policy would decide its output differently. A strict policy
likewise refuses an unfinished checkpoint written without one.

A completed checkpoint must hold a decided record: `build_state` refuses to
write a completed state whose record is `not_evaluated`, and resume refuses
such a state with `ValueError` before any provider call, leaving the
checkpoint unchanged. Its output is neither delivered nor returned.

### Decision history

[`Swarm.reevaluate`](verifier.md#re-evaluation), under a policy or with an
[operator's decision](verifier.md#operator-decisions), replaces a completed
checkpoint's `delivery_acceptance` with a new decision on the same output and
keeps the record it replaces in a top-level `delivery_acceptance_history`
list, oldest first:

```json
"delivery_acceptance_history": [
  {"sequence": 0, "delivery_acceptance": {"version": 1, "decision": "rejected", "...": "..."}}
]
```

Sequence 0 is the decision the run made when it completed and each
re-evaluation adds one, so the current record's sequence follows the last
entry. At most 16 superseded records are kept
(`smythe.checkpoint.DELIVERY_ACCEPTANCE_HISTORY_LIMIT`), the oldest dropped
first; sequences keep counting. Every entry is a decided record that binds
the same `output_sha256` as the current one. The list is informational, so
it is not a required state feature: resume reads only the current record,
which a reader that does not know the key still reads correctly. An
operator's decision is the exception: while the current record or a kept
entry is one, `requires` lists `operator_decision`, and once no kept record
is one, it no longer does.
Re-evaluation and [`smythe runs inspect`](runs.md#decision-history) refuse a
malformed history. `delivery_acceptance_history(state)` and
`delivery_acceptance_sequence(state)` in `smythe.checkpoint` read it back.
Re-evaluation changes nothing else in the checkpoint except `updated_at`
and one more [timing segment](#timing-segments).

## Attempt allowance (version 5)

A run under `RetryPolicy(persist_attempts=True)` ([persisted
attempts](execution.md#persisted-attempts)) writes every checkpoint as
version 5, lists `attempt_allowance` in `requires`, and records its policy
and each node's spent attempts in `control`:

```json
"control": {
  "revisions_used": 0,
  "retry_policy": {
    "retry_classes": ["transient"],
    "honor_retry_after": true,
    "backoff_max_s": 60.0,
    "deadline_s": null,
    "persist_attempts": true
  },
  "attempts_used": {"research": {"execution_generation": 0, "attempts": 2}}
}
```

- `retry_policy` holds the policy's fields (`RetryPolicy.to_dict()`); its
  `persist_attempts` is `true`. `retain_uncertain_dispatch` appears, as
  `true`, only when the policy sets it.
- `attempts_used` maps a node ID to the `execution_generation` it last
  attempted and how many attempts it started there. Each attempt is saved
  before its call, so a crash during a call still counts it. A node whose
  current generation differs from its record has used none.

The allowance is required exactly when it is listed: a version 5 state that
lists `attempt_allowance` without both keys, carries either key without
listing it, records a policy without `persist_attempts`, or holds a field or
count that is not valid is refused with `ValueError`, finished or not, before
any provider call. Versions 1 through 4 never have an allowance, whatever keys
they hold: a build that reads only those versions refuses version 5 rather
than resuming with a refilled allowance, and so does a version 5 reader that
does not know `attempt_allowance`.

An unfinished run resumes only under a `Swarm` whose `retry_policy` equals the
recorded one, and a policy with `persist_attempts` refuses an unfinished run
saved without an allowance. Other runs keep writing version 4 with
`control` holding only `revisions_used`. `smythe.checkpoint.attempt_allowance_from_state(state)`
returns the recorded `RetryPolicy` and `attempts_used` after these checks, or
`None` without an allowance.

## Model profiles (version 5)

A run whose graph has a node that names a [model
profile](execution.md#model-profiles) in `metadata.profile` writes every
checkpoint as version 5 and lists `model_profiles` in `requires`. The node
keeps the profile's name and model in its `metadata`:

```json
"metadata": {"profile": "sol", "model": "gpt-5.6-sol"}
```

A build that does not route profiles would call the run's provider for that
node, so builds that read only versions 1 through 4 refuse the checkpoint,
and so does a version 5 reader that does not know `model_profiles`. A
version 5 state whose nodes name profiles without listing `model_profiles`,
or that lists it with no node naming one, is refused with `ValueError`,
finished or not, before any provider call.

Resume routes each unfinished profiled node to the resuming `Swarm`'s
profile of the same name, and refuses with `NoAllowedProfileError`, before
any provider call, a node whose profile that Swarm does not allow or whose
saved model differs from its profile's. A completed run returns its saved
result without them. Runs without a profiled node keep writing version 4.
`smythe.checkpoint.model_profiles_from_state(state)` returns the profile each
node names, by node ID, after these checks.

## Resume semantics

`swarm.resume(execution_id)` (or `await swarm.aresume(...)`):

1. Loads the state and rejects unknown ids (`KeyError`) and unreadable versions (`ValueError`).
2. Restores the complete task, validates saved budget policy, charges, and verification state. Conflicting task copies, unresolved accounting markers, or ambiguous verification decisions stop recovery. A completed snapshot with output and no pending verification work returns its validated stored result and task; a version 5 snapshot returns its checked [delivery acceptance record](#delivery-acceptance-version-5) too. An unfinished version 5 snapshot stops here unless the resuming `Swarm` has the strict policy it began with, with an [attempt allowance](#attempt-allowance-version-5) the retry policy it records, and with [profiled nodes](#model-profiles-version-5) the profiles they name.
3. Otherwise restores the graph, re-registers the recorded agents, and resets `running` / `failed` nodes to `pending`. It finishes saved regeneration intents and consumes pending verdicts before dispatch. A rejected generation invalidates affected `completed` and `skipped` results too; other finished nodes keep their recorded results.
4. Restores per-node costs into the budget so the resumed run keeps counting against the original cap.
5. Executes the remaining nodes on the async executor, one at a time unless the resuming Swarm has `parallel=True` (then up to `max_concurrency`), synthesizes over the full graph, decides delivery acceptance under a strict policy, and writes the final checkpoint.

The trace on a resumed `SwarmResult` covers only the resumed portion; spans from before the crash are not reconstructed. Its `timing` likewise measures the resuming call, with `resumed` set and `latency_complete` false, and lists the earlier calls' saved segments before its own.

An invalid provider usage report stops the run and preserves its live
reservation. Node failures carry `accounting_invalid` and `accounting_error` in
their metadata; `budget.accounting_error` also covers synthesis and other
workflow-level failures. Before clearing these markers, reconcile provider charges,
repair `budget.node_costs`, and correct the malformed configuration or provider.
The marker prevents an omitted unresolved reservation from unlocking spend on
resume. See [Cost guardrails](budgets.md).

## Version compatibility

This build writes version `4` and reads versions `1` through `5`. It writes
version `5` only for a state with a [delivery acceptance
record](#delivery-acceptance-version-5), an [attempt
allowance](#attempt-allowance-version-5) or a node that names a [model
profile](#model-profiles-version-5), so builds that read only versions 1
through 4 refuse it instead of delivering output they cannot check,
refilling attempts the run has spent or calling the wrong provider. A version
5 reader also refuses any [required state feature](#required-state-features-version-5)
it does not know.
Version 2 added `control` and `task.done_when`; version 1 loads their original
defaults of zero revisions and no acceptance criteria.

The top-level `timing` block is optional and keeps version `4`. Earlier
builds ignore the key when they resume and leave it out of the checkpoints
they write; this build resumes checkpoints that lack it.

`budget.token_estimate_nodes` is optional in the same way. A checkpoint
without it restores no token estimates, as before the key existed, so
`cost_contains_token_estimates` then reports only calls made after resume.

So is `graph.planning`. Earlier builds read a saved graph key by key, so they
ignore it when they resume and leave it out of the checkpoints they write; a
graph without it restores `planning` as `None`.

So is `control.revision_log`. Earlier builds read only the `control` keys they
know, so they ignore the log when they resume and leave it out of the
checkpoints they write; this build resumes a checkpoint without it with an
empty log.

Version 4 adds no field. It records that every MCP server listed under
`agents` came from developer configuration: from 0.8.1 on, a model-generated
plan cannot declare MCP servers. Before 0.8.1 one could, and resume would start
the server as a local program. Resuming a version 1–3 checkpoint therefore
removes its agents' `mcp_servers` and logs a warning naming them; re-run the
task to use those tools. Older readers reject version 4.

The optional `graph.task` field is additive. Legacy graphs inherit the
checkpoint's top-level task when available. If both populated task copies
disagree, recovery raises `ValueError` before any provider call. Taskless
checkpoints remain supported. [Task snapshot values](tasks.md#snapshot-values).

Version 3 makes verification dispositions durable. Older readers reject it
instead of ignoring a rejection in progress. Completed legacy runs return
their stored output. Exhausted legacy gates resume without renewing their
allowance; advisory gates remain advisory. An incomplete version 1 or 2
checkpoint with a completed gated judge, unused regeneration allowance, and
no disposition is ambiguous: `VerificationRecoveryError` stops it before
provider work. Reconcile that saved verdict or start a new workflow.

Malformed generation identities, receipts, or regeneration intents also stop
recovery. An unsupported checkpoint version raises `ValueError` naming the
supported versions.

## Finding an execution id after a crash

`SwarmResult.execution_id` is only returned on success. After a crash, list what the store knows:

```python
from smythe import FileCheckpointStore

store = FileCheckpointStore()
print(store.list_ids())
```

`smythe runs list` lists the same ids from the command line with each run's
state, delivery decision and next action, and `smythe runs inspect <id>` shows
one checkpoint without changing it; see [Inspect saved runs](runs.md).
