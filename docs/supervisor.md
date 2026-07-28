# Adaptive supervision

The Architect plans once, before any work happens. Without a supervisor
the executor walks that plan to the end regardless of what the results
show — a plan that turns out to be wrong is still executed in full.

The benchmark record puts a number on that. In the published framework
head-to-head, `smythe_dynamic` scored **9.27 with a [5–10] range**: one
badly generated plan dragged a whole task down, and nothing in the run
could notice or correct it. Planning variance is the price of generated
topology, and a static graph has no mechanism to pay it back.

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

Reviewing after every node is usually wasteful. `LLMSupervisor` reviews
only after a node with no pending dependents finishes — the point where
a missing step actually becomes visible. Override it when you know
better:

```python
LLMSupervisor(provider, review_after={"draft"})   # only after this node
LLMSupervisor(provider, only_terminal=False)      # after every node
```

Keeping the supervising model out of the routine path is where adaptive
orchestration earns its keep: the published comparisons that beat
static planning on cost do so by *not* consulting an expensive model on
every step.

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
