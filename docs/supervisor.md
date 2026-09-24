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
    model="claude-opus-5-5",
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

Verification gates are protected too. A revision cannot drop a verifier
node (one that sets `verifies`) or the node it judges, and cannot rewire a
verifier unless its new dependencies still include that node. Without
this, a revision could let the judged output through unverified, or let a
parallel judge finish before its target so that its verdict was discarded.
The review prompt includes worker output, so text in source data could
steer a model supervisor toward such a revision.

## Safety

Adaptive orchestration is an unbounded agent loop unless you bound it.
Five guardrails do that:

1. **`max_revisions`** caps how many times a run may be revised.
   Supervision is off by default (`max_revisions=0`).
2. **Full validation before mutation.** `ExecutionGraph.apply_revision`
   checks every rule — no unknown nodes, no orphaned dependents, no
   cycles, no touching finished work, no removing or bypassing a gate, no
   node id `__synthesis__` (reserved for the synthesizer's budget entry) —
   and mutates nothing if any check fails. A malformed proposal costs a
   trace entry, not a corrupt run. These rules apply to every supervisor,
   including one you write.
3. **Contained proposal failure.** Ordinary supervisor errors and invalid
   revisions are recorded and ignored. Invalid provider accounting raises
   `BudgetValidationError`, stops the run, and blocks resume until the charge
   is reconciled. An optional review cannot bypass cost validation.
4. **Budget still rules.** Revision-added nodes are admitted through the
   same budget admission as planned ones. In ordinary Swarm execution without
   `run_store`, successful model-based review calls remain outside the execution
   ledger. Supported [managed text workflows](workflow-accounting.md) bind
   supervision to the shared ledger, reserve request-bound quotes before review
   calls, and preserve their charges and unknown exposure across recovery.
   See [accounting scope](budgets.md#current-scope).
5. **Strictly read proposals.** `LLMSupervisor` applies a reply only when
   `change` is JSON `true` (not the string `"true"`) and every field has the
   type shown in its prompt; anything else means no change. So does an
   added node whose label is longer than 500 characters, and so does a
   review that stopped at the output token limit or that the provider
   refused or filtered ([stop reasons](execution.md)). A proposal that
   adds more than `max_added_nodes` nodes (default 3) is treated as no
   change, not truncated, and logged as a warning on the `smythe.supervisor`
   logger. `max_total_added_nodes` (default 8, the generated-plan node
   limit) bounds growth across the whole run the same way: a proposal that
   would leave more revision-added nodes in the graph is refused and
   logged. The caps apply to model proposals; a `Supervisor` you write
   yourself is your code and is not capped.

```python
LLMSupervisor(provider, max_added_nodes=1)         # at most one new step per revision
LLMSupervisor(provider, max_total_added_nodes=4)   # at most four across the run
```

A node that a revision added carries `"added_by_revision": true` in its
metadata, and `LLMSupervisor` counts those nodes. The count is saved with the
graph, so resuming a checkpoint or a durable run cannot refill the allowance.
A pending added node that a later revision drops never ran, so it stops
counting. Non-default caps are part of a durable run's recipe.

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
stamps them. Review receives the complete task, including source data and
acceptance criteria, after an inspected-graph handoff or resume too.
[Task snapshots](tasks.md).

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
