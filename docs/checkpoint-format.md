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
    ]
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
        "waiting_ms": {"retry_backoff": 0.0, "admission": 900.0, "supervision": 0.0}
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
- `budget.token_estimate_nodes` lists the nodes, including `__synthesis__` for an `LLM_MERGE` synthesis, whose recorded cost prices tokens at the blended `cost_per_token` rate. Resume restores them, so `Sentinel.cost_contains_token_estimates` still reports costs priced before the checkpoint. A value other than a list of node IDs stops resume with `BudgetValidationError`.
- `control.revisions_used` is how much of the supervisor's `max_revisions` allowance the run has already spent. Resume seeds the executor from it, so the cap bounds the **run**, not each attempt: a crash-resume cycle cannot refill the allowance and revise past the limit the caller set.
- A node that a revision added carries `"added_by_revision": true` in its `metadata` (from 0.8.2). `LLMSupervisor` counts these nodes against `max_total_added_nodes`, so resume does not reset the count ([what the cap limits](supervisor.md#safety)). Nodes added before 0.8.2 have no marker and are not counted.
- `task.done_when` carries the acceptance criteria forward. A resumed run that had forgotten them could not hold its own output to them.
- `timing` records how long each call that worked on the execution took; see [timing segments](#timing-segments).
- A version 5 checkpoint adds a top-level `requires` list and the state it names, such as a `delivery_acceptance` record; see [required state features](#required-state-features-version-5).

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
only for a judge that completed before the crash.

The intent is saved before cancellation. After affected workers settle, a
second forced save records the reset: results and stale artifact references
are cleared, affected generations advance exactly once, and the intent is
removed. Costs and consumed regeneration allowances survive the reset.
Detached snapshot values prevent later metadata mutations from changing a
checkpoint already handed to a custom store.

## Required state features (version 5)

Version 5 is written only for a state that a build reading version 4 would
misread. Every version 5 checkpoint carries a top-level `requires` list naming
the state features its reader must understand:

```json
"requires": ["delivery_acceptance"]
```

This build knows one feature, `delivery_acceptance`. A reader refuses, with
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
- `scope` is `output`: the decision covers the synthesized text.

Versions 1 through 4 have no record, whatever keys they hold. Resuming one
returns `acceptance` as `None`: a legacy run is never reported as accepted or
as not yet evaluated.

Resuming a completed version 5 checkpoint checks that `output` still matches
`output_sha256`, that the task and every source node's generation are the ones
the record binds, and, when the resuming `Swarm` has a strict policy, that its
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

## Resume semantics

`swarm.resume(execution_id)` (or `await swarm.aresume(...)`):

1. Loads the state and rejects unknown ids (`KeyError`) and unreadable versions (`ValueError`).
2. Restores the complete task, validates saved budget policy, charges, and verification state. Conflicting task copies, unresolved accounting markers, or ambiguous verification decisions stop recovery. A completed snapshot with output and no pending verification work returns its validated stored result and task; a version 5 snapshot returns its checked [delivery acceptance record](#delivery-acceptance-version-5) too. An unfinished version 5 snapshot stops here unless the resuming `Swarm` has the strict policy it began with.
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
record](#delivery-acceptance-version-5), so builds that read only versions 1
through 4 refuse it instead of delivering output they cannot check. A version
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
