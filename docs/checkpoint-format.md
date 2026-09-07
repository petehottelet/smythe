# Checkpoint format (version 2)

When a `Swarm` is constructed with a `checkpoint_store`, it persists the full
execution state after planning and once more when the run finishes or fails.
By default (`checkpoint_every_n_nodes=1`) it also saves after every node reaches
a terminal status. Wide graphs may set a larger interval to reduce full-snapshot
write amplification; after a process crash, at most the completed but unflushed
tail of that batch may replay. That replay can duplicate provider spend or tool
side effects, so intervals above one are appropriate only when the write-saving
tradeoff is worth the recovery window and affected operations are idempotent.
The state is a single JSON document per execution, so you can inspect—or
repair—a checkpoint with any text editor.

With the default `FileCheckpointStore`, checkpoints live at `~/.smythe/checkpoints/<execution_id>.json`. Writes are atomic (temp file + rename): a crash mid-write never corrupts the previous checkpoint.

## Schema

```json
{
  "version": 2,
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

- `task` is `null` when a pre-built `ExecutionGraph` was executed instead of a `Task`.
- Node `result` values that aren't JSON-serializable are stored as their `str()` form.
- `budget.max_budget_usd` is the cap the execution started with; resume honors it, not whatever the resuming Swarm was constructed with.
- `control.revisions_used` is how much of the supervisor's `max_revisions` allowance the run has already spent. Resume seeds the executor from it, so the cap bounds the **run**, not each attempt: a crash-resume cycle cannot refill the allowance and revise past the limit the caller set.
- `task.done_when` carries the acceptance criteria forward. A resumed run that had forgotten them could not hold its own output to them.

## Resume semantics

`swarm.resume(execution_id)` (or `await swarm.aresume(...)`):

1. Loads the state and rejects unknown ids (`KeyError`) and unreadable versions (`ValueError`).
2. Validates saved budget policy and all charges, and rejects nodes marked `accounting_invalid` until operator reconciliation. If `status` is `completed` and `output` is present, returns the validated stored result without executing anything.
3. Otherwise restores the graph, re-registers the recorded agents, and resets `running` / `failed` nodes to `pending`. `completed` and `skipped` nodes keep their recorded results and are **not** re-executed.
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

This build writes version `2` and reads `SUPPORTED_CHECKPOINT_VERSIONS =
(1, 2)`. Version 2 added `control` and `task.done_when`; both are
additive, so a version 1 document still resumes — it loads with
`revisions_used: 0` and no acceptance criteria, which is exactly the
state it was written in. A version this build cannot read fails with a
`ValueError` naming the versions it does read, rather than resuming
against a schema it would misinterpret.

## Finding an execution id after a crash

`SwarmResult.execution_id` is only returned on success. After a crash, list what the store knows:

```python
from smythe import FileCheckpointStore

store = FileCheckpointStore()
print(store.list_ids())
```
