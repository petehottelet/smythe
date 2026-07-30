# Adaptive supervision

The Architect plans once, before any work happens. Without a supervisor
the executor walks that plan to the end regardless of what the results
show — a plan that turns out to be wrong is still executed in full.

The failure mode is concrete: a benchmark run once produced a badly
generated plan, and nothing in the run could notice or correct it —
every node executed the wrong shape faithfully to the end. Planning
variance is the price of generated topology, and a static graph has no
mechanism to pay it back.

**Supervision has now been measured, and on judged prose it does not
pay.** Across 30 live runs of the shape suite the supervisor was
consulted 94 times and proposed a change **zero** times — 94 paid
provider calls that changed nothing. See
[benchmarks/control_ablation.md](../benchmarks/control_ablation.md).

It is not inert by construction: it fires when a deliverable is visibly
incomplete, which used to happen often enough to matter and now rarely
does. Turn it on for workloads where you expect plans to be wrong in
ways the results reveal — long-running research, tool-driven work whose
findings redirect it. Do not turn it on expecting a quality lift on a
well-specified writing task.

A **supervisor** closes the loop. After a node completes it reviews the
work so far against the goal and may revise the *unexecuted* remainder.

```python
from smythe import LLMSupervisor, Swarm

swarm = Swarm(
    model="claude-opus-4-8",
    supervisor=LLMSupervisor(provider),
    max_revisions=2,          # supervision is off unless this is > 0
)
result = swarm.execute(task)
```

## What a revision may change

A `Revision` composes three operations, all restricted to work that has
not happened yet:

| Operation | Use |
|---|---|
| `add_nodes` | Append a step that closes a gap the results exposed |
| `drop_node_ids` | Cancel planned work the results made unnecessary |
| `rewire` | Re-point a pending node, e.g. to insert a step ahead of it |

History is immutable. Completed, running, and failed nodes are never
dropped or re-pointed, so a revision can never invalidate a result the
run has already banked or the money already spent on it.

## Safety

Adaptive orchestration is an unbounded agent loop unless you bound it.
Four guardrails do that:

1. **`max_revisions`** caps how many times a run may be revised.
   Supervision is off by default (`max_revisions=0`).
2. **Full validation before mutation.** `ExecutionGraph.apply_revision`
   checks every rule — no unknown nodes, no orphaned dependents, no
   cycles, no touching finished work — and mutates nothing if any check
   fails. A malformed proposal costs a trace entry, not a corrupt run.
3. **Contained failure.** A supervisor that raises, returns junk, or
   proposes an invalid change is recorded and ignored. Supervision is an
   optional improvement; it can never fail a run that would otherwise
   succeed.
4. **Budget still rules.** Revision-added nodes are admitted through the
   same Sentinel reservation as planned ones. A supervisor cannot spend
   past `max_budget_usd` — it can only make the run stop sooner.

## Cost control

Reviewing after every node is usually wasteful. By default
`LLMSupervisor` reviews only after a node with no pending dependents
finishes. Be aware what that means in practice: on a simple serial
graph it is the *final* node, so the supervisor can append remedial
work but has nothing left to drop or rewire. Genuine stage-boundary
review — closing a fork before its join — is not yet implemented.
Until it is, target reviews explicitly:

```python
LLMSupervisor(provider, review_after={"draft"})   # only after this node
LLMSupervisor(provider, only_terminal=False)      # after every node
```

Keeping the supervising model out of the routine path is the whole
cost argument for adaptive orchestration: a review after every node in
a wide graph can cost more than the work it supervises.

## Writing your own

Any deterministic rule can be a supervisor — no model required.

```python
from smythe import Revision, Supervisor
from smythe.graph import Node

class AddVerifierWhenShort(Supervisor):
    """Append a verification step when the deliverable looks thin."""

    async def review(self, graph, node, *, task, revisions_remaining):
        if len(str(node.result)) > 400:
            return None
        return Revision(
            add_nodes=(Node(label="Verify and expand the deliverable",
                            depends_on=[node.id]),),
            reason="deliverable was suspiciously short",
        )
```

Nodes a supervisor adds inherit the run's model and task context
automatically — they never passed through `Swarm.plan`, so the executor
stamps them.

## Observability

Every review that changes something emits a trace span, and so does
every rejection:

```python
for span in result.trace:
    if "revision" in span:
        print(span["status"], span["label"], span["revision"])
# revision_applied  closes the evidence gap  {'added': ['verify'], ...}
```

Rejections are recorded deliberately. A supervisor that keeps proposing
invalid changes is a finding, and silently dropping those attempts would
hide it.
