# Durable text workflows

`Swarm(run_store=SQLiteWorkflowStore(...))` binds routing, planning, execution,
verification, supervision, and synthesis to one persistent call ledger.
Separate `plan()` and `execute()` calls retain the same run identity and planning
charges. `resume(execution_id)` continues that run under its original policy.
Smythe 0.7.0 includes this managed text-workflow API.

## Start a bounded Astra workflow

Install `pip install "smythe[openai]==0.7.0"` and set
`OPENAI_API_KEY`.
This example makes paid generation calls under a $5 run allowance:

```python
from smythe import OpenAIResponsesProvider, SQLiteWorkflowStore, Swarm, Task

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        model="gpt-6-astra",
        provider=OpenAIResponsesProvider(
            reasoning_effort="medium", max_output_tokens=8192,
        ),
        run_store=store,
        max_budget_usd=5.00,
        parallel=True,
        max_concurrency=8,
    )
    graph = swarm.plan(Task(
        "Summarize this supplied source in three sentences.",
        context={"source": "A DAG expresses dependencies without directed cycles."},
    ))
    print(graph.run_ref["run_id"])
    result = swarm.execute(graph)
    print(result.output)
    print(result.workflow_accounting)
```

The [offline example](../examples/14_durable_text_workflow.py) exercises the same
handoff and cached recovery with zero provider API charges.

## Freeze graph limits

Use `WorkflowGraphPolicy` to keep an experiment or production task within a
declared graph size, execution model, retry limit, and regeneration limit:

```python
from smythe import WorkflowGraphPolicy

policy = WorkflowGraphPolicy(
    max_nodes=8,
    node_model="gpt-6-astra",
    max_retries=0,
    max_regenerations=0,
)
# Pass graph_policy=policy alongside run_store=store when constructing Swarm.
```

The policy is part of the saved workflow recipe. It applies to generated
plans, caller-built and YAML graphs, edited handoffs, recovered checkpoints,
and replayed planning decisions. A revision is checked on a detached candidate
before it can change the live graph. Oversized or otherwise disallowed plans
fail before node execution; any planning charges remain in the ledger.

`max_nodes` counts every node in the current graph, including verification,
completed, and skipped nodes. Optional limits set to `None` add no restriction.
`node_model` checks effective node models; configure the planner model separately
when an experiment requires the same model in both phases. Resume requires the
original policy and rejects an attempt to remove or weaken it. `graph_policy`
requires a durable `run_store`; omitting it preserves existing recipe identities.
`Node.max_retries` defaults to one. A zero-retry policy therefore requires the
architect to declare `max_retries: 0` on every node, including `HALT` nodes.
Include the graph limits in the planning instructions for a bounded experiment.

## Admission and cost

Each native request is frozen before the input-token count. The resulting quote
covers the full output cap and the most expensive applicable input category.
The journal reserves that exact ceiling transactionally against confirmed
charges, other reservations, and unknown exposure. Concurrent calls cannot each
spend the same remaining allowance. The provider's `max_cost_per_call_usd`
setting is a legacy execution estimate; managed workflows use request-bound
quotes and the shared run allowance instead.

The ledger stores integer nanoUSD, with no floating-point arithmetic for money.
A known charge replaces its own reservation once. An actual charge above its
admitted quote or run budget is retained and stops further admission. Missing
billing evidence remains unknown and holds exposure. The allowance governs
admission; it cannot undo an already incurred provider charge.

`result.total_cost_usd` is a compatibility projection of confirmed charges.
`workflow_accounting` reports exact `confirmed_nanousd`, `reserved_nanousd`,
`unknown_nanousd`, `unknown_calls`, and `call_count` separately.
`cost_scope="complete_text_workflow"` identifies this path. Its cost includes
all recorded phases and failed attempts, using the adapter's dated published
token prices. It does not represent a reconciled account invoice or hardware,
human, and external service costs. No historical benchmark is repriced.

## Recovery boundaries

Each call has a stable phase, component or node scope, generation, invocation,
attempt, and turn. Its request, provider configuration, price version, and
decoder version are immutable. A transaction grants dispatch permission once.
After that point, missing response evidence stops automatic replay of the HTTP
request, even if a connection failure leaves the remote outcome uncertain.

Raw bytes are committed before billing validation and decoding. A crash after
that commit resumes through local settlement and decoding. Accepted outputs
are replayed locally until a graph checkpoint consumes them. Planning saves
the exact parsed graph and agent identities before execution starts. Supervision
saves the graph it reviewed and its decision, then commits the resulting
control state and consumed operation together. Verification retains its existing
generation and regeneration receipts.

The SQLite journal uses WAL, full synchronization, fenced leases, and checkpoint
revisions. Heartbeats renew ownership during calls. A former owner can append
immutable late evidence for its exact dispatch, but cannot advance the run.
Recovery settles available late evidence before admitting new work.

**Unreleased after 0.7.0:** concurrent journal openers use bounded WAL retries
and create all tables and the persistent store identity in one transaction.
Competing openers reuse that identity. Failed initialization rolls back;
existing evidence and read-only inspection retain their behavior.

Resume requires the same component descriptions, model configuration, budget,
and concurrency policy. It does not refill allowances. An inspected pending
plan can be edited before execution; a stale graph from an already progressing
run must use `resume()`. A graph carrying durable provenance cannot enter the
ordinary unjournaled execution path.

## Supported scope

The initial managed path supports plain text with the exact built-in native
Responses provider for Astra and Sol, or stateless `OfflineProvider` plan/echo
fixtures. It binds the built-in simple, LLM, and constrained architects,
WhiteRabbit routing, static registries, LLM supervision, synthesis strategies,
and token verification. Every reachable router tier and template is checked
before routing can make a paid call. Provider instances are snapshotted per run.

Custom local architects, template builders, supervisors, and synthesizers use
`LocalOnly(factory, identity, version, role=...)`. The factory must return a fresh
component that performs no provider calls or external side effects. This is an
explicit caller contract, not a sandbox. Arbitrary paid custom components,
scripted offline response cursors, tools, image attachments, active capability
hydration, live planner memory, and a separate checkpoint store are rejected.
Those features remain available through the ordinary Swarm and artifact Jobs
APIs within their documented accounting scopes.

## Inspect a run

`store.list_runs()` finds run IDs, including runs interrupted during planning.
`store.inspect_run(execution_id)` returns safe call summaries, exact balances,
hashes, and events. `store.audit(execution_id)` verifies accounting and evidence
bindings. Open `SQLiteWorkflowStore(path, read_only=True)` for inspection without
schema changes or provider construction. Explicit `load_replay()` and
`load_evidence()` calls expose sensitive saved text and raw provider responses;
protect the database and its SQLite sidecar files accordingly. API keys are
excluded from stored provider descriptions. Ordinary summaries and traces do
not include raw responses or encrypted reasoning.

[Native provider](openai-responses.md) · [Cost guardrails](budgets.md) ·
[Astra campaign protocol](../benchmarks/astra_benchmark_plan.md).
