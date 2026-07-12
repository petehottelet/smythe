# Checkpoint format (version 1)

When a `Swarm` is constructed with a `checkpoint_store`, it persists the full execution state after planning, after every node reaches a terminal status (completed, skipped, or failed), and once more when the run finishes or fails. The state is a single JSON document per execution, so you can inspect — or repair — a checkpoint with any text editor.

With the default `FileCheckpointStore`, checkpoints live at `~/.smythe/checkpoints/<execution_id>.json`. Writes are atomic (temp file + rename): a crash mid-write never corrupts the previous checkpoint.

## Schema

```json
{
  "version": 1,
  "execution_id": "9f2c4a…",
  "status": "running | completed | failed",
  "created_at": 1751600000.0,
  "updated_at": 1751600042.5,
  "model": "claude-opus-4-8",
  "task": {
    "goal": "…",
    "constraints": ["…"],
    "context": {}
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
        "attach_dep_artifacts": false
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

## Resume semantics

`swarm.resume(execution_id)` (or `await swarm.aresume(...)`):

1. Loads the state and rejects unknown ids (`KeyError`) and unknown versions (`ValueError`).
2. If `status` is `completed` and `output` is present, returns the stored result without executing anything.
3. Otherwise restores the graph, re-registers the recorded agents, and resets `running` / `failed` nodes to `pending`. `completed` and `skipped` nodes keep their recorded results and are **not** re-executed.
4. Restores per-node costs into the budget so the resumed run keeps counting against the original cap.
5. Executes the remaining nodes (always on the parallel executor), synthesizes over the full graph, and writes the final checkpoint.

The trace on a resumed `SwarmResult` covers only the resumed portion; spans from before the crash are not reconstructed.

## Finding an execution id after a crash

`SwarmResult.execution_id` is only returned on success. After a crash, list what the store knows:

```python
from smythe import FileCheckpointStore

store = FileCheckpointStore()
print(store.list_ids())
```
