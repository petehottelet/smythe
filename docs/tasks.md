# Tasks and graph handoffs

A `Task` carries four inputs through the workflow: its goal, constraints,
source context, and acceptance criteria. Smythe snapshots them before routing
or planning and attaches the snapshot to `ExecutionGraph.task`.

Install the current checkout with `pip install -e .` to run this example.

```python
from smythe import OfflineProvider, SimpleArchitect, Swarm, Task

swarm = Swarm(provider=OfflineProvider(), architect=SimpleArchitect())
task = Task(
    goal="Review the invoice",
    constraints=["Preserve the currency and every line item"],
    context={"invoice": {"currency": "USD", "items": ["Hosting: 40"]}},
    done_when=["Return a reconciled total"],
)

graph = swarm.plan(task)
print(graph.task.context)
result = swarm.execute(graph)
```

The example runs offline. The same handoff works with a generated plan.
Inspection does not discard the source material or acceptance criteria.

## One task across stages

Routing, autonomous and constrained planning, execution, supervision, and
model-based synthesis receive the complete task. A node whose label already
matches the goal still receives its constraints, context, and acceptance
criteria. Nodes added by a supervisor inherit the same inputs.

Context appears in prompts as explicitly labeled JSON source data. Text inside
that data is separate from the workflow's instructions. The supplied goal,
constraints, and acceptance criteria define the requested work.

`plan(task)` followed by `execute(graph)` preserves the same task inputs as
`execute(task)`. Execution takes another detached snapshot from the inspected
graph. Later mutations to the caller's original task or nested source values
cannot change the submitted run.

Custom synthesizers read `graph.task`; their existing method signatures remain
valid. Caller-built and YAML graphs without a task retain their existing
behavior, including legacy node context metadata.

## Snapshot values

Snapshots preserve strings, booleans, finite numbers, nulls, nested mappings,
and arrays. Tuples become arrays. Unsupported Python objects and nonfinite
numbers become strings when the task is first normalized; subsequent stages
reuse those values. The `Task` constructor continues to accept Python context
objects.

Mapping keys must be strings. Cycles and non-string keys are rejected at the
snapshot boundary, before provider calls, rather than silently changing the
meaning of the source data. Shared references are allowed and copied into
independent snapshot values.

## Recovery and memory

Graph JSON includes the optional task. Checkpoints retain both the graph task
and their existing top-level task field. Recovery hydrates older graphs from
the top-level field; conflicting task copies stop recovery before dispatch.
The task is restored even when resume returns a stored completed result.
See [Checkpoint format](checkpoint-format.md).

When `PlannerMemory` is configured, successful inspected-graph runs record the
same task as direct runs. New outcome records include context and acceptance
criteria, and recall searches those fields. Older history records remain
readable with empty defaults for the added fields.
