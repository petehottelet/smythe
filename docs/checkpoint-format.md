# Checkpoint format (version 3)

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

With the default `FileCheckpointStore`, checkpoints live at `~/.smythe/checkpoints/<execution_id>.json`. Writes are atomic (temp file + rename): a crash mid-write never corrupts the previous checkpoint.

## Schema

```json
{
  "version": 3,
  "execution_id": "9f2c4a…",
  "status": "running | completed | failed",
  "created_at": 1751600000.0,
  "updated_at": 1751600042.5,
  "model": "claude-opus-4-8",
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
        "metadata": {"model": "claude-opus-4-8", "cost_usd": 0.0003},
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
    "node_costs": {"research": 0.0003}
  },
  "output": "…final synthesized output, null until the run completes…"
}
```

Notes:

- `task` and `graph.task` preserve the submitted task through an inspected-graph
  handoff. Both are `null` for a caller-built graph that carries no task.
- Node `result` values that aren't JSON-serializable are stored as their `str()` form.
- `budget.max_budget_usd` is the cap the execution started with; resume honors it, not whatever the resuming Swarm was constructed with.
- `control.revisions_used` is how much of the supervisor's `max_revisions` allowance the run has already spent. Resume seeds the executor from it, so the cap bounds the **run**, not each attempt: a crash-resume cycle cannot refill the allowance and revise past the limit the caller set.
- `task.done_when` carries the acceptance criteria forward. A resumed run that had forgotten them could not hold its own output to them.

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

The intent is saved before cancellation. After affected workers settle, a
second forced save records the reset: results and stale artifact references
are cleared, affected generations advance exactly once, and the intent is
removed. Costs and consumed regeneration allowances survive the reset.
Detached snapshot values prevent later metadata mutations from changing a
checkpoint already handed to a custom store.

## Resume semantics

`swarm.resume(execution_id)` (or `await swarm.aresume(...)`):

1. Loads the state and rejects unknown ids (`KeyError`) and unreadable versions (`ValueError`).
2. Restores the complete task, validates saved budget policy, charges, and verification state. Conflicting task copies, unresolved accounting markers, or ambiguous verification decisions stop recovery. A completed snapshot with output and no pending verification work returns its validated stored result and task.
3. Otherwise restores the graph, re-registers the recorded agents, and resets `running` / `failed` nodes to `pending`. It finishes saved regeneration intents and consumes pending verdicts before dispatch. A rejected generation invalidates affected `completed` and `skipped` results too; other finished nodes keep their recorded results.
4. Restores per-node costs into the budget so the resumed run keeps counting against the original cap.
5. Executes the remaining nodes (always on the parallel executor), synthesizes over the full graph, and writes the final checkpoint.

The trace on a resumed `SwarmResult` covers only the resumed portion; spans from before the crash are not reconstructed.

An invalid provider usage report stops the run and preserves its live
reservation. Node failures carry `accounting_invalid` and `accounting_error` in
their metadata; `budget.accounting_error` also covers synthesis and other
workflow-level failures. Before clearing these markers, reconcile provider charges,
repair `budget.node_costs`, and correct the malformed configuration or provider.
The marker prevents an omitted unresolved reservation from unlocking spend on
resume. See [Cost guardrails](budgets.md).

## Version compatibility

This build writes version `3` and reads versions `1`, `2`, and `3`.
Version 2 added `control` and `task.done_when`; version 1 loads their original
defaults of zero revisions and no acceptance criteria.

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
