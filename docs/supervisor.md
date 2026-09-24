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

Verification gates are protected too. A revision cannot:

- drop a verifier node (one that sets `verifies`) or the node it judges;
- rewire a verifier unless its new dependencies include the node it judges;
- rewire other nodes so that a verifier that depended on the node it judges,
  directly or through other nodes, no longer depends on it. In a hand-built
  `draft → review → check` gate, rewiring `review` off `draft` is rejected.

These rules allow a rewire of the verifier that keeps the node it judges
among its dependencies, and a rewire elsewhere that leaves the verifier
depending on that node, such as inserting a step between them.
Without these rules, a revision could let the judged output through
unverified, or let a judge run before its target so that its verdict was
discarded. The review prompt includes worker output, so text in source data
could steer a model supervisor toward such a revision.

### Steps added after a gate

Gate protection keeps each gate judging its target. It does not stop a
revision from adding work after that target: a revision may add a step that
depends on a gated node, such as a `final` step after `draft`, or rewire a
pending step to depend on it. No existing gate judges that step. Under
`DELIVERABLE` synthesis, the default, a node that a non-verifier step depends
on is not terminal, so the run returns the new step's output in place of the
judged node's. Limiting `review_after` to nodes that run before the gate does
not prevent this, because a new step may depend on work that is still
pending. To keep the gated node's output as the run's output, run without
revisions (`max_revisions=0`, the default) or use a `Supervisor` of your own
that never proposes such a step.

## Safety

Adaptive orchestration is an unbounded agent loop unless you bound it.
Five guardrails do that:

1. **`max_revisions`** caps how many times a run may be revised.
   Supervision is off by default (`max_revisions=0`).
2. **Full validation before mutation.** `ExecutionGraph.apply_revision`
   checks every rule — no unknown nodes, no orphaned dependents, no
   cycles, no touching finished work, no dropping a gate or the node it
   judges, no cutting a gate off from that node, no node id `__synthesis__`
   (reserved for the synthesizer's budget entry) — and mutates nothing if
   any check fails. A malformed proposal costs a trace entry, not a corrupt
   run. These rules apply to every supervisor, including one you write. A
   [durable run](workflow-accounting.md#recovery-boundaries) also rejects a
   revision that adds a node under the id of a node that already made
   journaled calls.
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
   limit) limits the revision-added nodes in the graph the same way: a
   proposal that would leave more of them in the graph is refused and
   logged. The caps apply to model proposals; a `Supervisor` you write
   yourself is your code and is not capped.

```python
LLMSupervisor(provider, max_added_nodes=1)         # at most one new step per revision
LLMSupervisor(provider, max_total_added_nodes=4)   # at most four added steps in the graph
```

A node that a revision added carries `"added_by_revision": true` in its
metadata, and `LLMSupervisor` counts the marked nodes in the graph when it
reads a proposal. The marker is saved with the graph, so resuming a checkpoint
or a durable run does not reset the count. The cap limits the added nodes the
graph holds, not every node added during a run, so a run can add, and run,
more than the cap in total:

- A pending added node that a later revision drops stops counting, and that
  revision or a later one can add another in its place. This includes an
  added node that ran and was then reset to pending by a gate's regeneration.
- A durable run saves each review's decision before applying it. If the run
  stops in between, resume applies the saved decision without checking the
  cap again, so nodes that another review added first during recovery are not
  counted against it.

Non-default caps are part of a durable run's recipe.

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
