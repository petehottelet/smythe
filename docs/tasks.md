# Tasks and graph handoffs

A `Task` carries four inputs through the workflow: its goal, constraints,
source context, and acceptance criteria. Smythe snapshots them before routing
or planning and attaches the snapshot to `ExecutionGraph.task`.

Smythe 0.7.0 carries the complete task through planning, execution, and recovery.
Install `pip install "smythe==0.9.1"` to run this example.

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

`done_when` tells planners and nodes what the work must achieve; it does not
decide whether the output is delivered. A [strict acceptance
policy](verifier.md#strict-delivery-acceptance) does that with final
validators and gates, and its record binds the SHA-256 of the task snapshot
(`smythe.acceptance.task_sha256`). Resuming a strict run whose saved task no
longer matches that digest never returns accepted output.

When `PlannerMemory` is configured, successful inspected-graph runs record the
same task as direct runs. New outcome records include context and acceptance
criteria, and recall searches those fields. Older history records remain
readable with empty defaults for the added fields.

## Context snapshots

A [durable text workflow](workflow-accounting.md) can also give its nodes
sources: documents, excerpts or records that the work draws on but that are
not part of the task. Pass them as `context_sources` to opt each new run into
context assembly:

```python
from smythe import (
    ContextSource, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task,
)

handbook = ContextSource("handbook", "Refunds are accepted within 30 days.", version="2026-09")
with SQLiteWorkflowStore("runs.db") as store:
    swarm = Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
        run_store=store, context_sources=[handbook],
    )
    result = swarm.execute(Task("Answer the refund question", done_when=["States the window"]))
```

The run records its sources once, as source revision 0 in a journal
operation keyed `context-sources/<revision>`, with its first checkpoint.
Before each node call it records a `ContextSnapshot` as a journal operation
keyed `context/<node>/<generation>`: the task's goal, constraints
and acceptance criteria, the step, every source with its version and SHA-256,
the SHA-256 of each dependency output the node sees with the journal calls
that produced it, the agent and capability tags the node runs as, and the
context-assembly version. The snapshot's id is the SHA-256 of its canonical
JSON, and the node's call names it, so a call under another snapshot is a
different journal call. Checkpoints hold only digests: the source set and
each node's snapshot id, never their text.

The node's prompt is rendered from its snapshot. The task's requirements and
the step are instructions. The task's context data, each source and each
dependency output follow, after a note that they are material and not
instructions, each in its own block:

```text
<<<BEGIN UNTRUSTED DATA 358c37c2c13db7fa: source "handbook", version "2026-09">>>
Refunds are accepted within 30 days.
<<<END UNTRUSTED DATA 358c37c2c13db7fa>>>
```

A block's tag is the start of the SHA-256 of its own text, so the text cannot
end its block early. Tool results get the same blocks when an executor
assembles context. Text in a source never changes what a run may do: its
capabilities, budget and acceptance policy come from the recipe, and a source
that asks for more leaves them byte-identical.

A run keeps the sources it recorded: sources that changed while it was
paused, or sources a later `Swarm` is given, reach none of its calls. A
resumed or retried call uses the snapshot the journal kept, after checking
the task and every dependency output against the snapshot's digests; a
mismatch raises `WorkflowStateError`. To give an unfinished run new sources,
refresh them explicitly:

```python
edited = ContextSource("handbook", "Refunds are accepted within 14 days.", version="2026-10")
revision = swarm.refresh_context(run_id, [edited])  # 1: the run's next source revision
result = swarm.resume(run_id)
```

`refresh_context` records the sources as the run's next revision and makes
every node pending at its next generation, without calling anything;
`resume` then runs each node again with a new snapshot. Earlier snapshots,
calls and charges stay in the journal, and refreshing with the sources the run
already uses changes nothing. A run whose checkpoint is still `planned` only
adopts the new revision. A completed run, or one without context assembly,
raises `WorkflowStateError`. `smythe.context.load_context_snapshot` and
`load_context_sources` read the records back; `ContextSnapshot.summary()`
leaves out source text.

Sources reach node calls, not planning. A recipe without `context_assembly`,
which every run without `context_sources` records, keeps the prompts and
journal records of earlier releases byte for byte. Durable runs still refuse
planner memory.
