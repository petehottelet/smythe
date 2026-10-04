# Durable text workflows

`Swarm(run_store=SQLiteWorkflowStore(...))` binds routing, planning, execution,
verification, supervision, and synthesis to one persistent call ledger.
Separate `plan()` and `execute()` calls retain the same run identity and planning
charges. `resume(execution_id)` continues that run under its original policy.
Smythe 0.7.0 includes this managed text-workflow API.

**Added in 0.8.0:** [native Fable Messages](anthropic-messages.md) uses the same
journal with Anthropic-specific cache categories, quotes, and response decoding.

## Start a bounded Astra workflow

Install `pip install "smythe[openai]==0.9.1"` and set
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

With `parallel=True`, a failing node cancels calls still in flight, which can
block the run. See [admission and cost](#admission-and-cost).

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
before it can change the live graph. A disallowed graph fails before node
execution; any planning charges remain in the ledger. The planner's own limits
also apply: `LLMArchitect` retries a plan over 8 nodes or 5 levels, so pass
`architect=LLMArchitect(max_nodes=...)` for a policy above 8 nodes.

**Changed in 0.8.2:** the built-in LLM and constrained architects check each
generated plan against the policy and the run's other graph rules, such as
plain-text nodes and node ids the journal can key, before planning is saved.
The journal keys each node's calls and supervision decisions by its id, so an
id must be at most 256 characters with no control characters
(`smythe.workflow.MAX_NODE_ID_CHARS`). A caller-built graph with another id
fails before any node runs, and so does resuming a run whose saved graph has
such an id. A rejected plan goes back to the
model with the reason, like a schema error, and each repair is a new journaled,
paid call. When `max_retries` repairs are used up, planning raises
`ArchitectError`. Resuming replays the saved attempts to the same result
without buying them again, so start a new run. A run that 0.8.1 stopped after
such a rejection resumes: its saved plan replays, is rejected, and the planner
asks for a repair if `max_retries` allows one.

`max_nodes` counts every node in the current graph, including verification,
completed, and skipped nodes. Optional limits set to `None` add no restriction.
`node_model` checks effective node models; configure the planner model separately
when an experiment requires the same model in both phases. Resume requires the
original policy and rejects an attempt to remove or weaken it. `graph_policy`
requires a durable `run_store`; omitting it preserves existing recipe identities.
`Node.max_retries` defaults to one. A zero-retry policy therefore requires the
architect to declare `max_retries: 0` on every node, including `HALT` nodes.
Include the graph limits in the planning instructions for a bounded experiment.

**Added in 0.8.0:** pass graph-only requirements through
`LLMArchitect(planning_instructions="...")`. This text is recorded in the
workflow recipe and sent only to the planner. Keep answer requirements in
`Task.constraints`, which every executor receives. For example:

```python
from smythe.planner import LLMArchitect

architect = LLMArchitect(
    provider, planning_model="gpt-6-astra", max_retries=0,
    planning_instructions=(
        "Use at most eight nodes. Set max_retries: 0 and max_regenerations: 0 "
        "on every node. Produce one terminal deliverable."
    ),
)
```

The prompt guides construction; `WorkflowGraphPolicy` enforces the limits.
An empty `planning_instructions` preserves existing recipe identities.

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

**Changed in 0.8.1:** a request that the provider refuses before any model
work settles at zero cost with a rejected result, and the run stays open. A
generation response qualifies only when all of these hold:

- Its status is 401, 403, 404, 413 or 429: authentication, permission, an
  unknown model or endpoint, an oversized request, or a rate or quota limit.
- Its body is the provider's plain error object and nothing else: OpenAI's
  `{"error": {...}}` or Anthropic's `{"type": "error", "error": {...},
  "request_id": "..."}`. The error object may contain only `message`, `type`,
  `param` and `code`, each a string or null. A body with usage or any other
  field does not qualify.
- The exception recorded with the response, if any, is that status's SDK
  error, such as `RateLimitError` for a 429, or the SDK's general
  `APIStatusError`. A transport error, such as a timeout, does not qualify.

The node raises `ProviderRequestRejectedError`, with the status code and raw
evidence attached, and its failure policy decides what happens next: `RETRY`
makes a new journaled call, `SKIP` skips the node, and `HALT` fails it. A
rejected supervisor review counts as no change, like other supervisor errors.
Every other unsuccessful response holds unknown exposure, because the provider
may have processed and billed the request:

- transport failures with no response
- every other 4xx status, including 400 (Anthropic reports output blocked by
  its content filter as a 400), 408, 409, 422 and 499
- a qualifying status with any other body or error
- 5xx responses, including Anthropic's 529 overloaded status

Before retry `n`, `RETRY` waits a random time between 0 and
`retry_backoff_s * 2**(n - 1)` seconds. The default, `retry_backoff_s=0`,
retries immediately. Each new call starts with an input-token count. If the
provider rejects the count, for example with another 429, the node stops
whatever its failure policy. The count leaves no exposure, and resuming the
run repeats it. `retry_backoff_s` is part of the saved recipe, so resuming
with a different value raises `WorkflowConflictError`. A run created with the
default of 0, including a 0.8.0 run created that way, cannot gain a backoff
when it resumes.

Resuming a run starts each unfinished node at its first attempt again. A
saved accepted response replays without a new request, and so does a
rejection that a later attempt already followed. Replayed retries still wait
their backoff. A rejected call that nothing followed is sent again as a new
call. Resuming a halted run therefore gives the rejected step one new call,
and a `RETRY` node whose attempts were all rejected gets one new call, for its
final attempt.

In a parallel run, a node that fails cancels the other nodes that are still
running. A zero-cost rejection fails its node under `HALT`, the default, and
under `RETRY` once the final attempt is rejected. A node that `SKIP` skips
cancels nothing. A cancelled node whose generation request was already sent
holds unknown exposure at its quoted ceiling, so the run is blocked with
`unknown_exposure` and resuming it fails. Runs that execute one node at a time
(`parallel=False`, the default, or `max_concurrency=1`) have no other call in
flight. A node's `timeout_s` also cancels a call that was already sent, with
the same result, even in a serial run. A generated plan cannot set a timeout
below 60 seconds.

`result.total_cost_usd` is a compatibility projection of confirmed charges.
`workflow_accounting` reports exact `confirmed_nanousd`, `reserved_nanousd`,
`unknown_nanousd`, `unknown_calls`, and `call_count` separately.
`cost_scope="complete_text_workflow"` identifies this path. Its cost includes
all recorded phases and failed attempts, using the adapter's dated published
token prices. It does not represent a reconciled account invoice or hardware,
human, and external service costs. No historical benchmark is repriced.

## Price versions

Each native call records the price version it was admitted under, and its
journal binding covers that version. `smythe.pricing.PRICE_TABLES` and
`smythe.pricing_anthropic.PRICE_TABLES` register every version by name, and
`PRICE_VERSION` in each module names the current one. A release that changes
prices or the supported models adds a version and keeps the earlier ones
registered.

A new call must use the current version and a model that version prices. A
saved call is verified by recomputing its stored binding instead of applying
the current request rules: its version must still be registered and must
price its model. Lookup, inspection, audit and recovery therefore keep working
after such a release, and the call's quote and settlement use the rates of
the version it recorded. A stored receipt never changes, and settling the
same evidence again produces the same bytes. A journal that names a version
this release does not register, such as one written by a newer release, is
reported as corrupt.

A price table can give a model a retirement date (`PriceTable.retired_on`).
From that day, by the journal clock in UTC, a durable run admits no new call to
that model and does not send a saved call to it that was never sent. Preparing
such a call again, reserving it or claiming its dispatch raises
`WorkflowValidationError`, so a resumed run neither quotes nor sends it. The
call keeps its state and recorded price version, and
`release_before_dispatch` frees its reservation. Calls to the model that were
already sent remain readable and still settle and replay.

A run saved before such a release resumes. The provider descriptions in its
saved recipe may name an earlier registered price version and a different
supported model list; every other recipe field must still match exactly.
Each saved call replays under the price version it recorded, including a call
that was prepared or quoted but not yet sent, unless its model has since
retired, and every new call uses the current version. A run's total can
therefore combine charges at two price versions, and each call's receipt names
its own. The configured model must still be one the current version prices.

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
control state and consumed operation together. Calls are keyed by node id and
generation, so a revision that adds a node under the id of a node with
journaled calls is rejected and traced: the new node would conflict with, or
replay, the earlier node's calls. Verification retains its existing
generation and regeneration receipts.

The SQLite journal uses WAL, full synchronization, fenced leases, and checkpoint
revisions. Heartbeats renew ownership during calls. A former owner can append
immutable late evidence for its exact dispatch, but cannot advance the run.
Recovery settles available late evidence before admitting new work.
Each plan, execute or resume entry audits the whole journal once when it
starts. A checkpoint saved during the run reads only the calls it consumes
and, once a node has been regenerated, the keys of settled calls, so its cost
does not grow with the run's events, evidence or earlier checkpoints.

A 0.8.0 journal holds every saved 4xx error response as unknown exposure. On
resume, 0.8.1 settles a saved response that meets the zero-cost conditions
above, such as a plain 429, again at zero cost with a rejected result and
records an `unknown_exposure_resolved` event. If no other unknown call or
overrun blocks the run, admission reopens. Once the node completes or is
skipped, its saved `response_error`, `accounting_invalid`, `accounting_error`
and `cost_usd_unknown` markers are removed. Any other saved response, such as a
400, keeps its unknown exposure. Repeating the settlement changes nothing.

**Added in 0.8.0:** concurrent journal openers use bounded WAL retries
and create all tables and the persistent store identity in one transaction.
Competing openers reuse that identity. Failed initialization rolls back;
existing evidence and read-only inspection retain their behavior.

Resume requires the same component descriptions, model configuration, budget,
retry backoff, and concurrency policy, apart from the
[price versions](#price-versions) of saved provider descriptions. It does not
refill allowances. An
inspected pending plan can be edited before execution; a stale graph from an
already progressing run must use `resume()`. A graph carrying durable
provenance cannot enter the ordinary unjournaled execution path.

## Supported scope

The managed path supports plain text with three exact built-in providers:
native `OpenAIResponsesProvider` for `gpt-6-astra` and `gpt-5.6-sol`, native
`AnthropicMessagesProvider` for `claude-fable-5-1`, and stateless
`OfflineProvider` plan/echo fixtures. Both native providers use their global
endpoints. The path binds the built-in simple, LLM, and constrained architects,
WhiteRabbit routing, static registries, LLM supervision, synthesis strategies,
and token verification. Every reachable router tier and template is checked
before routing can make a paid call. Provider instances are snapshotted per run.

[Native Claude Messages](anthropic-messages.md) shares this journal and runs
the [Fable 5.1 benchmark extension](../benchmarks/fable_51_benchmark_plan.md).
The ordinary `AnthropicProvider` cannot enter this managed path, and neither
can subclasses of the supported providers.

Custom local architects, template builders, supervisors, and synthesizers use
`LocalOnly(factory, identity, version, role=...)`. The factory must return a fresh
component that performs no provider calls or external side effects. A template
builder must also return the same nodes, with the same ids, for the same task
and params ([why](architecture.md#planning-tiers)). This is an
explicit caller contract, not a sandbox. Arbitrary paid custom components,
scripted offline response cursors, tools, image attachments, active capability
hydration, live planner memory, and a separate checkpoint store are rejected.
Those features remain available through the ordinary Swarm and artifact Jobs
APIs within their documented accounting scopes. A rejected `run_store`,
`verifier`, `memory`, `tool_runtime` or `checkpoint_store` raises
`WorkflowBindingError` with a message that starts with that argument's name.
The [support matrix](support-matrix.md) lists each accepted and rejected
combination with the test behind it.

## Inspect a run

`store.list_runs()` finds run IDs, including runs interrupted during planning.
`store.inspect_run(execution_id)` returns safe call summaries, exact balances,
hashes, and events. `store.lease_sessions(execution_id)` returns the
journal-clock bounds of each lease session, which a durable result reports as
its [timing](execution.md#timing). `store.audit(execution_id)` verifies accounting and evidence
bindings. Open `SQLiteWorkflowStore(path, read_only=True)` for inspection without
schema changes or provider construction, including a journal from an earlier
release ([journal upgrades](#journal-upgrades)). Explicit `load_replay()` and
`load_evidence()` calls expose sensitive saved text and raw provider responses;
protect the database and its SQLite sidecar files accordingly. Since 0.8.1, a
new database file is created readable and writable by its owner only (mode
0600 on POSIX), and SQLite gives its `-wal` and `-shm` files the same mode.
Opening an existing file never changes its permissions. API keys are
excluded from stored provider descriptions. Ordinary summaries and traces do
not include raw responses or encrypted reasoning.

## Journal upgrades

The journal records its format version, which `store.store_version` reports.
This release writes version 2, which adds the `tool` call phase; its tables
are the same as version 1's. Smythe 0.9.x and earlier write version 1 and
cannot open a version 2 journal, even read-only: they raise
`WorkflowCorruptionError("Unsupported workflow journal version")`.

Back up each journal before this release opens it for writing, and keep the
backup for as long as 0.9.x may need to read it. Copy the file, and any
`-wal` file beside it, while no process has it open, or use SQLite's backup
API, which includes changes still in the `-wal` file:

```python
import sqlite3

source, backup = sqlite3.connect("smythe-runs.db"), sqlite3.connect("smythe-runs-v1.db")
source.backup(backup)
backup.close()
source.close()
```

The backup holds the same raw prompts and responses, so protect it like the
journal; it is created with your default file permissions.

- `SQLiteWorkflowStore(path, read_only=True)` opens a version 1 journal
  without changing it, and `store_version` is 1. Listing, inspection, audit
  and replay work as before, including while an earlier release's worker holds
  a run.
- `SQLiteWorkflowStore(path)` upgrades a version 1 journal in the transaction
  that opens it. The upgrade changes only the recorded version: every run,
  call, balance, reservation, receipt, evidence row, checkpoint and event keeps
  its bytes, and the store identity is unchanged. Concurrent openers upgrade
  it once. An interrupted upgrade, including a killed process, rolls back and
  leaves a version 1 journal.
- Stop workers from earlier releases first. Those workers check the version
  only when they open the journal, so while any run holds an unexpired lease
  the upgrade raises `WorkflowLeaseError` and changes nothing. A released or
  expired lease, such as one a killed process left, does not block it.
- A store open for writing checks the version before each write. If another
  release changes the version while the store is open, its next write raises
  `WorkflowCorruptionError` and writes nothing.
- A journal whose version is newer than this release supports is refused in
  both modes, without changes.

To return to 0.9.x, restore the backup; runs recorded after the upgrade are
not in it.

[Native provider](openai-responses.md) · [Cost guardrails](budgets.md) ·
[Astra campaign protocol](../benchmarks/astra_benchmark_plan.md).
