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
makes a new journaled call, `SKIP` skips the node, and `HALT` fails it. Its
read-only `failure_class` is `authority_denied` for 401 or 403, `input_error`
for 404 or 413 and `transient` for 429 (see [failure
classes](execution.md#failure-classes)); it does not change what the failure
policy does. A rejected supervisor review counts as no change, like other
supervisor errors.
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
final attempt. A run under a [retry policy](#retry-policy) continues each
node's attempts instead of refilling them.

In a parallel run, a node that fails cancels the other nodes that are still
running. A zero-cost rejection fails its node under `HALT`, the default, and
under `RETRY` once the final attempt is rejected. A node that `SKIP` skips
cancels nothing. A cancelled node whose generation request was already sent
holds unknown exposure at its quoted ceiling, so the run is blocked with
`unknown_exposure` and resuming it fails. A [security stop](#security-stops)
cancels the running nodes the same way and blocks the run with `security_stop`
instead. Runs that execute one node at a time
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

### Retry policy

`Swarm(run_store=store, retry_policy=RetryPolicy(...))` puts a durable run's
node calls under a [retry policy](execution.md#retry-policy). A `RETRY` node
then calls again only after a failure whose class the policy lists,
`transient` by default: a 429 rejection is retried, and a 401, 403, 404 or 413
rejection is not. A refused or `max_tokens` reply keeps its known charge in
the journal, as a rejected call, and is retried only when the policy lists
`quality_defect`; otherwise it still stops the run. The recipe records the
policy's fields under `retry_policy`, except `persist_attempts`, and a recipe
without a policy keeps its bytes and the retries described above.

The journal is the record of each node's attempts. Each time a node starts,
it continues from the attempts its current generation has used: every call
of that generation that the journal settled as rejected counts, on whichever
invocation it was made. A crash, a halted run and every resume therefore
share the node's `1 + max_retries` attempts (one without `RETRY`) instead of
refilling them, and an attempt whose call was rejected is never sent again.
When the generation's latest call is a refused, truncated or unusable reply,
whose known charge the journal keeps, the node enters that attempt again
instead, and the journal replays the reply without a call: a reply the
policy does not retry stops the run again, as it does without a policy, and
one it retries is followed by the next attempt, within the cap. A node with
no attempt left fails without a call, with a `RuntimeError` that
names its allowance, and its failure policy settles it; a new run starts
fresh. A call that was accepted, reserved but never sent, or sent without a
known charge is not counted, so entering its attempt again replays, sends or
refuses it as before. A regenerated node starts a new generation with a fresh
allowance. `persist_attempts` makes no difference: the journal keeps every
attempt.

The journal keeps no response headers, so a durable run never reads a
`Retry-After`: the wait before a retry is the jittered backoff that doubles
from `retry_backoff_s`, at most `backoff_max_s`, and no wait with the default
`retry_backoff_s=0`. `deadline_s` is measured within one process and starts
again when the run resumes.

A run keeps the policy it started with. Resuming with a `Swarm` that sets no
`retry_policy` uses the saved one. A different policy, a policy for a run
that started without one, and a saved policy this release cannot read raise
`WorkflowConflictError` before any call. Every provider the run can call must
declare `sdk_retries=0`, as the native and offline providers do; otherwise
the run is refused with `WorkflowBindingError` before it is created or, on
resume, before any call. A policy with `retain_uncertain_dispatch` is refused
the same way: the journal keeps the exposure of every call it sends itself.

### Security stops

A durable run takes a [run control](execution.md#security-stops) as ordinary
runs do: `Swarm(run_store=store, run_control=control)`. The control is not
part of the recipe. Once it is revoked, or a tool runtime raises
`SecurityStop`, the run admits no node and sends no call, its synthesis is not
called, and the nodes in flight are cancelled at once. A cancelled call that
was already sent holds unknown exposure at its quoted ceiling, as after a
`HALT`. The run is then blocked in the journal before its failed checkpoint
is saved: its status becomes `blocked`, its `blocked_reason` becomes
`security_stop`, and a `security_stop` event records the reason code, never
provider text.

`store.security_stop(run_id, reason_code)` does the same for any run,
unfinished or completed, without a lease, from any process that can open the
journal. The first stop stands: stopping a stopped run again records nothing
and returns the first stop, and `store.load_security_stop(run_id)` returns it
as `{"reason_code", "at_ns"}`, on the journal's clock, or `None`. After a
stop no new call is prepared and no call is reserved or dispatched. Calls
already dispatched still settle when their evidence arrives, and keep unknown
exposure without it; neither an overrun, unknown exposure nor conflicting
evidence replaces the reason, and no method clears it. Resuming, executing,
re-evaluating or republishing a stopped run raises its `SecurityStop` before
any lease is taken, whatever the resuming Swarm's control. A successor run
never reuses a stopped run's results, and its output cannot be prepared or
approved as an [external action](actions.md), whose authority check refuses
it too. Earlier releases admit no new call for a stopped run either, since
any `blocked_reason` closes admission, but they still return a completed
run's saved result.

`store.inspect_run(run_id)` and `smythe runs inspect` show the reason and the
unknown exposure, and the next action is `rerun`. The journal has no way to
reconcile unknown model-call exposure, so a stopped run's exposure stays
recorded as unknown: once the cause is resolved, run the task again as a new
run.

### Money units

The journal stores each amount as the decimal text of a whole number of
nanoUSD. Stored receipts keep the form they were written in: an offline
receipt records its zero cost as the string `"0"`. `max_budget_usd` is rounded
down to whole nanoUSD. `workflow_accounting["unit"]` is `"nanoUSD"`, the unit
of its amounts; stored rows and receipts carry no unit field. See
[money units](budgets.md#money-units) for how these amounts compare with the
float USD and micro-USD ledgers.

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
request, even if a connection failure leaves the remote outcome uncertain. The
one exception is a [read step](#recorded-read-steps) whose profile declares it
free and safe to send again, which is read again as a new call.

A call that was reserved but never dispatched, as when a process is killed
between the reservation and the dispatch claim, keeps its reservation while
the run can still send it. When its node runs again at the same generation,
resuming sends it under that reservation and its saved quote, without counting
its input again. The run releases it once it can no longer be sent:

- when a checkpoint records a regeneration that moved its node to a later
  generation, including a regeneration that resuming applies, because a node
  calls only at its current generation;
- when the run completes, for a call that an interrupted attempt or review
  left behind, such as an attempt whose dispatch claim failed before a retry.

Each release returns the reservation to the run's allowance and records a
`call_released` event with the released amount and its reason,
`superseded_generation` or `run_completed`. `SQLiteWorkflowStore` exposes these
releases as `release_superseded_calls()` and `release_reserved_calls()`;
`release_before_dispatch()` releases one call and records the same event. A
call whose dispatch was claimed is never released: without response evidence,
recovery holds it as unknown exposure.

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
and, once a node has been regenerated, the keys of settled and reserved calls;
the completed checkpoint also reads the keys of reserved calls to release
them. Its cost therefore does not grow with the run's events, evidence or
earlier checkpoints.

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
retry backoff, concurrency policy, gate validator,
[acceptance policy](#strict-delivery-acceptance),
[renderer](#published-documents) and [lineage policy](lineage.md), apart from the
[price versions](#price-versions) of saved provider descriptions. It does not
refill allowances. An
inspected pending plan can be edited before execution; a stale graph from an
already progressing run must use `resume()`. A graph carrying durable
provenance cannot enter the ordinary unjournaled execution path.

Node prompts follow the prompt-assembly version that the run's recipe records.
Durable runs record none, which keeps the placement of every earlier release:
the instruction that a node is the final step goes to each node that no other
node depends on, including a judge that nothing depends on, and not to the
node that judge verifies. [Ordinary runs](verifier.md#the-final-step) place it
differently. No public setting opts a durable run into the ordinary placement
yet; the recipe field `prompt_assembly` is reserved for that opt-in. Resume
reads the version from the saved recipe, so a run in flight sends the requests
it saved, and a saved recipe whose version this release does not support
raises `WorkflowConflictError`.

A run started with `context_sources` records `context_assembly: 1` in its
recipe, the same way: each node call is then given a journaled context
snapshot, and its prompt marks sources and dependency outputs as untrusted
data ([context snapshots](tasks.md#context-snapshots)). Resume reads the
version from the saved recipe, so a `Swarm` without `context_sources` resumes
such a run with the sources it recorded, and one with `context_sources`
cannot resume a run whose recipe has no `context_assembly`. Such a run keys
synthesis by node generation, like a strict run, so a refresh of its sources
never replays an earlier synthesis. `Swarm.remove_source` journals a
`source_tombstone` operation for a source that must no longer be used; the
run then never sends or returns its text, an unfinished run resumes only
with other sources, and a completed run's earlier decision becomes
`indeterminate` until it is re-evaluated
([remove a source](tasks.md#remove-a-source)).

## Strict delivery acceptance

`Swarm(run_store=store, acceptance=AcceptancePolicy(mode="strict", ...))`
decides whether the synthesized output is accepted, as an ordinary run does
([strict delivery acceptance](verifier.md#strict-delivery-acceptance)), and
journals the decision:

- The recipe records the policy under `acceptance`, so the run resumes only
  under the same policy, and gates read with `TokenVerifier` record the
  verifier identity `token-v2` in place of `token-v1`. Recipes without a
  strict policy keep their bytes.
- [Which gates count](verifier.md#which-gates-count) follows the ordinary
  rule: the saved planning result marks each gate of a generated plan
  `generated_gate`, and a supervisor revision marks the nodes it adds. The
  [acceptance obligations](verifier.md#acceptance-obligations) are checked
  before any node call: for a graph you supply, before the run is created;
  for a generated or edited plan, before its first node runs, so a refused
  plan stays `planned` with its planning charge; and for each supervisor
  revision, before it applies. Without final validators or
  `allow_generated_gates`, a run whose architects all generate their gates
  is refused before the planning call.
- After the `synthesis` operation, an `acceptance` operation runs the final
  validators on the exact output and reads every gate's final verdict, with
  no provider call. It is keyed by the output digest and everything else its
  record binds, and the completed checkpoint consumes it together with the
  synthesis, in one transaction. When the policy has a
  `SourceEvidenceValidator`, a `source_evidence` operation keyed
  `source-evidence/<sha256>` first records the
  [source evidence](source-evidence.md#durable-runs) the output carries,
  bound to its digest and the run's source set, and the same checkpoint
  consumes it. A `SourceEvidenceValidator` built from other sources than
  the run's current source set does not run, and the decision is
  `indeterminate` by default.
- A process lost after synthesis and before the decision commits decides
  again on resume, on the same saved output. A process lost after the
  decision commits replays it, and one lost after the completed checkpoint
  returns it; neither runs a validator or calls a provider again.
- Checkpoints of runs with a strict policy or a gate validator are version 2,
  with a `delivery_acceptance` block; other runs keep version 1. Synthesis in
  such a run is keyed `synthesis` while every node is at generation 0, then
  `synthesis/<digest>`, a short digest of every node id and its generation,
  so each graph state has its own synthesis.
- Resuming a completed run checks the saved output, task, source generations
  and the journal's current decision against the record; a mismatch gives an
  `indeterminate` record and moves the text to `rejected_output`.
- Resuming never decides a completed run again, and resuming under another
  policy is refused. `Swarm(run_store=store).reevaluate(run_id,
  acceptance=policy)` is the explicit way to a new decision, for example
  under a fixed validator: it accepts a recipe that differs only in its
  acceptance policy and gate verifier, runs no node and calls no provider,
  journals the decision as a new `acceptance` operation keyed
  `acceptance/<sequence>/<digest>`, and applies it with a new completed
  checkpoint. Calls, charges, the budget, revision and regeneration counters,
  node generations and the recipe are unchanged, and earlier decisions stay
  in the journal. Output that no longer matches its record is refused. See
  [re-evaluation](verifier.md#re-evaluation);
  `smythe.workflow.current_decision(store, run_id)` returns the decision that
  resume and [`smythe runs inspect`](runs.md#decision-history) report.

The journal format is unchanged: operation kinds are free text, so this needs
no journal upgrade. `tests/test_workflow_process_loss.py` kills a real
process at each of these points and resumes the run, including a run with
priced native calls and a regenerated draft, whose confirmed charges, node
generations and regeneration count resume unchanged without a new call.

## Published documents

A durable run can publish its delivered output as a local document. Pass a
`renderer` with the `run_store`; the document goes to
`artifact_dir/<run id>/`:

```python
from smythe import (
    AcceptancePolicy, MarkdownRenderer, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore,
    Swarm, Task, TextValidator,
)
from smythe.workflow_artifact import current_artifact

with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
        run_store=store, renderer=MarkdownRenderer(), artifact_dir="briefs",
        acceptance=AcceptancePolicy(mode="strict", final_validators=[TextValidator()]),
    )
    result = swarm.execute(Task("Write the brief"))
    artifact = current_artifact(store, result.execution_id)
    print(artifact.publication.path, artifact.eligible)
```

- **The renderer is registered, deterministic local code.** The built-in
  `MarkdownRenderer` writes `document.md`: a level-one heading with the
  task's goal, a blank line, the synthesized text without trailing
  whitespace, and one final newline. Your own renderer is declared with
  `LocalOnly(factory, identity, version, role="renderer")`: the factory builds
  an object with a synchronous `render(output, task)` that returns text, and
  optional `filename` (default `document.txt`, a simple name with an
  extension) and `media_type` (default `text/plain; charset=utf-8`, a `text/*`
  type). It must make the same text from the same input, with no provider
  calls or side effects; this is a caller contract, not a sandbox. The recipe
  records the renderer's description, `{"type": "markdown_renderer",
  "version": 1}` or its `LocalOnly` identity and version, only when there is
  one, so existing recipes keep their bytes and a run resumes only with the
  renderer it started with. Without a renderer, `artifact_dir` is ignored, as
  before.
- **The document is the delivered output.** After the `synthesis` operation
  the renderer renders the synthesized text, and the document's UTF-8 bytes
  are published once. A strict policy's final validators check exactly those
  bytes, so the decision's `output_sha256` is the published file's SHA-256,
  and the run's result and completed checkpoint hold the document. A
  [source evidence](source-evidence.md#durable-runs) record is read from the
  document. Because a rendered document is not any node's result, a strict
  policy with a renderer needs a final validator.
- **Publication never replaces a file.** The run's directory is bound to the
  run by an owner marker before any node call; a directory owned by another
  run, or holding files the run does not own, is refused with
  `WorkflowStateError`, as is a filesystem without hard links. The document
  is published under the renderer's file name or, when that name is taken,
  the next free numbered name (`document-2.md`). A `publication` operation
  keyed `publication/<sequence>` records a `PublishedArtifact`: the sequence,
  the directory and file name, the SHA-256, size and media type of the bytes,
  the renderer's description, and the synthesis it rendered. The completed
  checkpoint consumes it with the decision.
- **Interrupted publication.** A process lost after publishing and before the
  operation commits reuses the file it published, which already holds the
  same bytes. A process lost after the operation commits renders the
  document again and checks that it has the recorded SHA-256; if the
  renderer no longer makes those bytes, the published file is read instead,
  and if that file is also missing or changed the run fails with
  `WorkflowStateError` rather than decide on other bytes.

`smythe.workflow_artifact.current_artifact(store, run_id)` is the one accessor
for a run's current publication: the one with the highest sequence that a
checkpoint applied, the delivery decision that covers it, and whether it is
eligible for delivery now. The decision is the run's current one, checked as
resume checks it, so after a [re-evaluation](verifier.md#re-evaluation) it is
the re-evaluation's, under the policy that made it. Its `reasons` name every
condition that is not met:
`no_decision` (the run has no strict policy or has not completed),
`not_accepted`, `record_mismatch` (the decision binds other bytes), and
`artifact_missing`, `artifact_changed`, `artifact_unsafe` (a symbolic link,
junction or non-regular file) or `artifact_unreadable` from rechecking the
file's size and SHA-256. Eligibility is derived on every read and never stored;
the accessor only reads, so a store opened read-only will do.

A strict run's result, whether the run completes now, including after an
interrupted publication, or completed earlier, and `smythe runs inspect` all
check the current publication: a missing or changed file gives an
`indeterminate` decision with the reason, the output moves to
`rejected_output`, and [inspection](runs.md#published-documents) exits with
code 11 and the next action `republish`. `swarm.republish(run_id)` publishes
the saved document again: its bytes, which must still have the published
SHA-256, go to the renderer's file name or, when that is taken, the next free
numbered name, so a changed file keeps its bytes, and the next `publication`
operation, applied by a new completed checkpoint, becomes current. It calls no
provider, runs no validator and leaves the decision as it is, since that binds
the same bytes. A run whose current file is intact is left as it is. A run
with a [lineage policy](lineage.md#repair-a-run-in-place) is repaired in place
instead: its saved synthesis is rendered again, published under the new
publication's own name (`document-p1.md`) and decided again, still without a
provider call. The accessor's `superseded` lists the earlier publications a
checkpoint applied, whose files stay where they are. Operation kinds are free
text, so publication needs no journal upgrade.

## Lineage records

A run with a [lineage policy](lineage.md) journals one more operation when it
completes. After the `acceptance` operation, a `lineage` operation records the
run's [lineage](lineage.md#the-record-a-run-writes): its published document,
its decision, and each node's result and inputs as ids, versions and digests.
It is keyed `lineage` while every node is at generation 0 and
`lineage/<digest>` after that, following the synthesis key, and the completed
checkpoint consumes it with the synthesis, publication and decision in one
transaction. It is assembled once, from targeted loads of the operations and
snapshots it names, so checkpoint saves read what they did before; it holds
no calls and no cost. `inspect_run()`, `list_runs()` and journal events never
return it, and `smythe.lineage.load_lineage(store, run_id)` reads it.
Resuming a completed run assembles it again and gives an `indeterminate`
decision when the saved record no longer matches. Each later decision, by
re-evaluation or republication, journals a `lineage` operation keyed by its
decision sequence, `lineage/<sequence>` (or `lineage/<digest>/<sequence>`),
whose inputs name the record it supersedes; the checkpoint that applies the
decision applies it too. Operation kinds are free text, so lineage needs no
journal upgrade.

## Reused results

A successor of a completed lineage run is a new run, under its own recipe and
budget, of the predecessor's task and completed graph: every node starts
pending without its result or history, the inherited planning records are
dropped, and each node either keeps the predecessor's result or runs again.
No public method starts a successor yet.

- **Reuse operations.** Before any node runs, each kept node completes from a
  `reuse` operation keyed `reuse/<node id>`. Its inputs name the predecessor
  (`store_id`, `run_id` and the `lineage_sha256` of its record), the node and
  its `generation` there, the `result_sha256`, the digests of the node's
  lineage record and of the source evidence its text carries (`records`),
  the reason codes the reuse was decided with and the repair plan's
  `plan_sha256`. Its result holds the text and those records. The next
  checkpoint consumes each operation once, with the completed node. A reuse
  operation carries no calls and no cost: the successor's accounting holds
  only the calls of the nodes that run again, and the predecessor's rows are
  never changed.
- **Checked before anything is created.** The predecessor must be in the
  same store, completed, with an applied lineage record in the successor's
  access scope, compared first, whose digest is the one named. A kept node's
  saved result must have the digest its record gives, its inputs must be
  declared, and no source its record or source evidence names may have been
  [removed](tasks.md#remove-a-source) from the predecessor. Gates, the nodes
  they judge or depend on, and nodes that depend on a node that runs again
  are never kept. The graph must also suit the successor's own
  configuration, which may differ from the predecessor's: its graph policy,
  the read profiles its read steps name, its node models and its agents. A
  refusal creates no run, so nothing is left to resume, and makes no call.
- **Same text, same prompts.** A kept result is the predecessor's text byte
  for byte, so the nodes that run again are given exactly what a full
  recomputation would give them.
- **Recovery.** A process lost after a reuse operation commits replays it on
  resume without reading the predecessor; one lost before it commits checks
  the predecessor again first. Refreshing an unfinished successor's sources
  runs its kept nodes again, like every other node, even when it was lost
  before it saved any of them; their reuse operations then stay unapplied.
  A kept result is never used once a source its record or source evidence
  names is removed from the successor itself.
- **Records.** The successor's graph holds one planning record, `repair`:
  the predecessor, the plan digest and the reason. Its
  [lineage](lineage.md#reuse-in-a-successor-run) names its `predecessor` and
  `repair`, and each kept node keeps its recorded inputs with `reused_from`.
  A new graph that carries a `repair` planning record, or `lineage_reuse`
  node metadata, is refused. `inspect_run()`, `list_runs()`, journal events
  and `smythe runs inspect` never return kept text.
- **Withdrawn text.** The successor's first checkpoint applies a copy of each
  of the predecessor's `source_tombstone` operations, so text
  [withdrawn](lineage.md#restrict-a-source) from the predecessor is withdrawn
  from the successor too.

## Restricted runs

`smythe.lineage.restrict_source` journals a `source_tombstone` operation in
each completed lineage run whose record names the withdrawn text, and, with
`purge=True`, an `artifact_purge` operation keyed
`artifact-purge/<publication sequence>` for each published file it deletes. A
copy of the run's completed checkpoint applies them under the run's lease;
they hold no calls, no cost and no source text, and the run's calls,
balances and earlier checkpoints are unchanged
([restrict a source](lineage.md#restrict-a-source)).

## Recorded read steps

A durable run can read a source as a step of its own, without a model call,
and journal the read like a model call: a saved response is reused on resume
instead of read again. Declare each read operation as a `ReadProfile`, pass
the profiles as `read_tools`, and add read steps to a graph you write:

```python
import json

from smythe import OpenAIResponsesProvider, ReadProfile, SQLiteWorkflowStore, Swarm
from smythe.graph import ExecutionGraph, Node, Topology

source = ReadProfile(
    name="source", version="1",
    url="https://docs.example.com/sources/{source_id}",
    input_schema={"source_id": "string"},
    metered=False, repeat_safe=True,
    decode=lambda body: json.loads(body)["content"], decoder_version="source-json-v1",
)
graph = ExecutionGraph([Topology.SERIAL], [
    source.step("fetch", {"source_id": "q3-brief"}),
    Node(id="summary", label="Summarize the brief in three sentences", depends_on=["fetch"]),
])
with SQLiteWorkflowStore("smythe-runs.db") as store:
    swarm = Swarm(model="gpt-6-astra", provider=OpenAIResponsesProvider(max_output_tokens=2048),
                  run_store=store, max_budget_usd=1.00, read_tools=[source])
    result = swarm.execute(graph)
```

A read step is a node whose `metadata["read_step"]` is
`{"profile": <name>, "input": {...}}`; `profile.step(node_id, input, ...)`
builds one after checking its input, and YAML graphs can declare the same
metadata. The step's result is the decoded text, which the steps that depend
on it receive as context, like any other step's result.

### Profiles

A profile is versioned data plus an optional local decoder:

- `url` is an `http` or `https` URL. `{field}` placeholders may appear only
  in its path or query, so input never chooses the host, and each takes the
  percent-encoded value of an input field. A URL with credentials, a
  fragment, spaces or non-ASCII text is refused.
- `input_schema` maps each input field to `"string"` or `"integer"`. A step
  must give exactly those fields; strings are 1 to 512 characters, without
  control characters, and not `.` or `..`.
- `metered` and `repeat_safe` have no defaults. `metered` says whether a
  read can be charged: a metered profile declares its per-call price as
  `ceiling_nanousd`, and a free one declares 0. `repeat_safe` says whether
  sending the read again is harmless.
- `decode(body)` turns a 200 response body into the step's text, and
  `decoder_version` names it; a custom decoder needs its own version. Without
  one, the body is read as strict UTF-8 (`utf8-text-v1`). The decoder is a
  caller contract, not a sandbox: it must be local code without side effects.
- `max_response_bytes` (default 1,000,000, at most 16 MiB), `timeout_s`
  (default 30 seconds, at most 600), `max_redirects` (default 0, at most 10),
  `allowed_hosts` (the URL's host is always allowed) and
  `allow_private_networks` (default `False`) limit each read.

A read sends one `GET`, asking for an uncompressed body, with no proxy. Unless
`allow_private_networks` is set, every address its host resolves to must be
public: loopback, private, link-local, shared and other special addresses are
refused. The request connects to the address that was checked, so a second
lookup cannot move it, and HTTPS still verifies the certificate for the host
name. A destination refused before anything is sent releases the call's
reservation, and the step fails. After the request is sent, a redirect beyond
`max_redirects`, to a host outside `allowed_hosts`, to a scheme other than
`http` or `https`, or from `https` to `http`, a body over
`max_response_bytes`, and a read that takes longer than `timeout_s` are
refused; the saved response then has an empty body and names the limit.
The time limit covers the whole read, however slowly the source sends: the
read stops and closes its connection once `timeout_s` seconds have passed.
Only a host name lookup can take longer, for as long as the system resolver
allows. Once connected, a read also stops as soon as its step stops waiting,
for example at the node's own `timeout_s`.

### Who can declare a read

Only graphs and templates you write. A generated plan cannot: model output may
set no node metadata except `role`, and `LLMSupervisor` adds steps with only
an id, a label and dependencies. A step must name a profile in the run's
`read_tools`, with valid input, and cannot judge another step.
`read_tools` requires a `run_store`. An ordinary run raises `ValueError` for a
read step before any node call: in a graph passed to `execute()` before it
runs, in a plan that an architect or template builds as soon as it exists,
and in a saved checkpoint before it resumes. A `tool_runtime` and model
function calling stay unsupported in durable runs.

The recipe records the profile descriptions under `read_tools`, without
decoder code, only when a run declares any, so existing recipes keep their
identity. Resuming with a changed profile raises `WorkflowConflictError` before
any call. Read text is data for the steps after it: it cannot add a step, a
profile, a tool or budget, or change the recipe.

### Journal, cost and retries

Each read is a call in the journal's `tool` phase, keyed by its node, its
generation, invocation and attempt, with the kind `read_tool` and the price
version `declared-read-v1`. The journal admits only the exact request the
recorded profile builds from the step's input. The quote is the profile's
declared ceiling; the call reserves it, claims its dispatch once, and saves
the raw response before settling and decoding it.

- A free read costs nothing, whatever its outcome, with one exception: a
  read that is not `repeat_safe` and received no response has an uncertain
  outcome, so it keeps unknown exposure, and the run is blocked with
  `unknown_exposure`.
- A metered read is charged its declared price for a 200 response. Any other
  outcome may still have been charged, so it keeps unknown exposure at that
  price and blocks the run. An uncertain metered read is never sent again
  automatically.

A read makes one request per journaled call and never retries inside it. A
free read whose response is not a usable 200, such as another status, a
refused limit or a lost connection, and any read whose decoder fails, settle
at their known cost as a rejected result, and the step raises `ReadToolError`
with the status or failure name, never the response body. The node's failure policy decides what happens next:
`RETRY` reads again as a new journaled call. Resuming re-enters the last
rejected call of a free, `repeat_safe` read as a new invocation, so a halted
step reads once more; other rejections replay.

### Interrupted reads

| The process stopped | On resume |
|---|---|
| Before the dispatch claim | The same call is sent under its reservation and saved quote; nothing had been sent. |
| After the claim, before the response was saved | A free, `repeat_safe` read settles at zero cost as rejected, records a `read_response_lost` event and reads again as a new invocation. Any other read keeps unknown exposure and blocks the run; it is never sent again. |
| After the response was saved | The response is settled and decoded locally, without reading again. |
| After the text was accepted | The saved text replays until a checkpoint consumes it. |

`tests/test_workflow_read_tool.py` kills a real process at each of these
points, against a fake service that counts every read it receives, and
resumes the run.

`inspect_run()`, `list_runs()` and journal events never include a read's URL,
input, response or text; a receipt records the declared pricing, the HTTP
status and any failure name. `load_replay()` and `load_evidence()` return
them explicitly.

## Supported scope

The managed path supports plain text with three exact built-in providers:
native `OpenAIResponsesProvider` for `gpt-6-astra` and `gpt-5.6-sol`, native
`AnthropicMessagesProvider` for `claude-fable-5-1`, and stateless
`OfflineProvider` plan/echo fixtures. Both native providers use their global
endpoints. The path binds the built-in simple, LLM, and constrained architects,
WhiteRabbit routing, static registries, LLM and
[evidence-triggered](supervisor.md#evidence-triggered-review) supervision, synthesis strategies,
token verification, gates judged by a
[registered validator](verifier.md#gate-validators-in-durable-workflows)
declared with `LocalOnly(..., role="validator")`,
[recorded read steps](#recorded-read-steps) of declared profiles, and
[published documents](#published-documents) from `MarkdownRenderer` or a
renderer declared with `LocalOnly(..., role="renderer")`. Every reachable router tier
and template is checked before routing can make a paid call. Provider
instances are snapshotted per run.

[Native Claude Messages](anthropic-messages.md) shares this journal and runs
the [Fable 5.1 benchmark extension](../benchmarks/fable_51_benchmark_plan.md).
The ordinary `AnthropicProvider` cannot enter this managed path, and neither
can subclasses of the supported providers.

Custom local architects, template builders, supervisors, synthesizers, gate
validators and renderers use `LocalOnly(factory, identity, version, role=...)`. The factory must return a fresh
component that performs no provider calls or external side effects. A template
builder must also return the same nodes, with the same ids, for the same task
and params ([why](architecture.md#planning-tiers)). This is an
explicit caller contract, not a sandbox. Arbitrary paid custom components,
scripted offline response cursors, model-driven tools, image attachments, active capability
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
bindings, including a run's [external action records](#external-action-records). Open `SQLiteWorkflowStore(path, read_only=True)` for inspection without
schema changes or provider construction, including a journal from an earlier
release ([journal upgrades](#journal-upgrades)). From the command line,
`smythe runs --journal <path> inspect <run id>` reports the same balances,
the run's state, its delivery decision, its
[published document](#published-documents) and its next action through a
read-only open; see
[Inspect saved runs](runs.md). Explicit `load_replay()` and
`load_evidence()` calls expose sensitive saved text and raw provider and read responses;
protect the database and its SQLite sidecar files accordingly. Since 0.8.1, a
new database file is created readable and writable by its owner only (mode
0600 on POSIX), and SQLite gives its `-wal` and `-shm` files the same mode.
Opening an existing file never changes its permissions. API keys are
excluded from stored provider descriptions. Ordinary summaries and traces do
not include raw responses or encrypted reasoning.

## External action records

A version 2 journal has four tables for [external actions](actions.md):
`workflow_actions` holds each logical action with its intent, connector
profile, state and balances; `workflow_action_attempts` its numbered attempts;
`workflow_action_approvals` the approvals and rejections bound to its intent
digest; and `workflow_action_records` its readbacks, lookups, notices, late
receipts, findings, decisions and evidence. Records are append-only: SQLite
refuses to update or delete them. In this release, `smythe.actions` writes
[prepared actions](actions.md#prepared-actions), including
[compensations](actions.md#compensation), whose `compensates_action_id`
names the action they undo, and their
[approvals, rejections and revocations](actions.md#approvals), including a
run's [action stop](actions.md#stopping-a-runs-actions), which also adds an
`actions_stopped` event; `smythe.workflow_actions` writes their attempts and
late receipts in the [dispatch journal](actions.md#dispatch-journal); the
[action runner](actions.md#readback) writes readback records, each charged
its read's cost, and its [reconciliation](actions.md#reconciliation) writes
lookup records, each charged its lookup's cost, and finding records; and
[`record_notice()`](actions.md#notices) keeps notices, marked untrusted.
Nothing writes decision or evidence records yet.

Action money is its own accounting scope, `external_actions`, in whole
nanoUSD. It never enters a run's balances or `workflow_accounting`, which stay
equal to the run's call ledger. `store.audit()` also checks a run's actions:
each intent hashes to its recorded digest and names its profile's connector
and one of its operations, each key is `action_key(store_id, run_id, slot,
sequence)`, every state, verdict and record kind is known, approvals bind
their action's intent digest, attempts are numbered from 1 under an approval,
at most one attempt is prepared or dispatched and the action's state agrees
with it, only an attempt that may have taken effect consumes an approval,
and an action's confirmed, reserved and unknown amounts equal the sums of its
attempts' costs, reservations and exposure and its records' costs. A
mismatch raises `WorkflowCorruptionError`.

An action's state and its accounting are kept apart. A write whose outcome
is unknown keeps its write ceiling as unknown exposure while reconciliation
looks it up, reads it back, retries it under deduplication or finds that it
was not made, and after the action is reconciled: only a receipt that
reports the write's cost turns that exposure into the attempt's confirmed
cost. Each lookup and read is confirmed at the profile's read ceiling, and
the action runner makes none that would take the action past the ceiling of
the attempt's approval, including the reads of an attempt dispatched after
reconciliation found that the write was not made, which share that budget
with its lookups. A reconciled action can therefore still carry unknown
exposure, and no attempt, cost or record is removed when its outcome is
settled.

For a run with actions, `inspect_run()` adds an `actions` block after its
other keys: the scope's totals and, for each action, its identity, intent
digest, operation and target, state and balances, with its attempts,
approvals (principal, verifier, ceiling, expiry, consumption, revocation time
and, for a revoked approval, whether the
[revocation](actions.md#revoking-after-dispatch) came before any dispatch,
after one or after its effect) and records by digest. It never includes
parameters, approval assertions, revocation reasons, receipts, record
payloads or error text. A run without actions inspects exactly as before.

## Journal upgrades

The journal records its format version, which `store.store_version` reports.
This release writes version 2, which adds the `tool` call phase of
[recorded read steps](#recorded-read-steps) and the
[external action tables](#external-action-records). Version 1 has every other
table. Each version requires its own tables: a journal missing one, or a
version 1 journal that already has an action table, is refused with
`WorkflowCorruptionError`. Tables that other tools, such as a backup or
replication tool, add to the file are kept, including through the upgrade.
Smythe 0.9.x and earlier write version 1 and cannot open a version 2 journal,
even read-only: they raise
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
  that opens it. The upgrade creates the external action tables, empty, and
  changes the recorded version; nothing else changes: every run, call,
  balance, reservation, receipt, evidence row, checkpoint and event keeps its
  bytes, the store identity is unchanged, and so is the file's mode.
  Concurrent openers upgrade it once. An interrupted upgrade, including a
  killed process, rolls back and leaves a version 1 journal without the new
  tables.
- Stop workers from earlier releases first. Those workers check the version
  only when they open the journal, so while any run holds an unexpired lease
  the upgrade raises `WorkflowLeaseError` and leaves the journal at version 1
  with its rows unchanged; the file may already have switched to SQLite's WAL
  journal mode, which earlier releases also read. A released or
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
