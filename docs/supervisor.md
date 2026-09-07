# Adaptive supervision

Generated topology becomes result-aware with a supervisor. The Architect
creates the initial graph; completed work can then redirect the unexecuted
remainder without invalidating any result already banked.

Use supervision when completed results can redirect the pending plan:
long-running research, tool-driven investigation, and workflows with explicit
stage boundaries. For a well-specified task with no evidence-driven branch,
the generated plan can run directly. Explicit review points and fan-in
triggers concentrate model review where new evidence can change the plan.
[Measured control scope](../benchmarks/control_ablation.md).

A **supervisor** closes the loop. After a node completes it reviews the
work so far against the goal and may revise the *unexecuted* remainder.

```python
from smythe import LLMSupervisor, Swarm

swarm = Swarm(
    model="claude-opus-4-8",
    supervisor=LLMSupervisor(provider, review_after={"research"}),
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
3. **Contained proposal failure.** Ordinary supervisor errors and invalid
   revisions are recorded and ignored. Invalid provider accounting raises
   `BudgetValidationError`, stops the run, and blocks resume until the charge
   is reconciled. An optional review cannot bypass cost validation.
4. **Budget still rules.** Revision-added nodes are admitted through the
   same Sentinel reservation as planned ones. Model-based review calls remain
   outside the current execution ledger; [complete workflow accounting](budgets.md#current-scope)
   is tracked separately.

## Triggering reviews

Reviewing after every node is usually wasteful. By default
`LLMSupervisor` reviews when a multi-input join becomes ready and once when the
whole graph finishes. A parallel set of terminal leaves produces one final
review rather than one review per leaf. On a simple serial graph the default
review is the final node, so use `review_after` when an earlier result can
redirect the remaining steps:

```python
LLMSupervisor(provider, review_after={"draft"})   # only after this node
LLMSupervisor(provider, only_terminal=False)      # after every node
```

The strongest default is an evidence-bearing stage boundary named in
`review_after`. Deterministic supervisors can narrow this further by checking
an anomaly first and calling a model only when the result gives the remaining
plan something to reconsider.

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
