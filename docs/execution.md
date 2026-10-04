# Execution and failure policies

Serial and parallel execution apply the same node failure policies. A terminal
failure stops new provider dispatch, preserves the original exception, and
retains completed results and recorded charges.

| Policy | After a provider error |
|---|---|
| `HALT` | Fail the node and stop the run immediately |
| `RETRY` | Make up to `max_retries` additional attempts; stop if all fail |
| `SKIP` | Mark the node skipped and continue eligible work |

A skipped node's dependents receive `[skipped: this step did not complete]` in
place of its result. The error text stays in the node's `result`, the trace,
and checkpoints for diagnosis, but it never enters a dependent's prompt or the
synthesized output. Resumed runs apply the same rule to saved skipped nodes.

Each attempt records its own trace span. An attempt that fails with an error
ends its span with status `failed` and that error, whether the node then
retries, is skipped, or stops the run.

Incomplete output is a provider error for the SDK providers
(`AnthropicProvider`, `OpenAIProvider` and `GeminiProvider`). When a response
stops at its output limit (Anthropic `stop_reason` `max_tokens` or
`model_context_window_exceeded`, OpenAI `finish_reason` `length`, Gemini
`finish_reason` `MAX_TOKENS`), the executor records the call's charge and then
raises `OutputTruncatedError`. A response that the provider refused, filtered
or ended early raises `OutputRefusedError` the same way, with one of these
`stop_reason` values:

| `stop_reason` | Provider signal |
|---|---|
| `refusal` | Anthropic `stop_reason` `refusal`; OpenAI `message.refusal` set |
| `content_filter` | OpenAI `finish_reason` `content_filter`; Gemini `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT` or `IMAGE_RECITATION`; a Gemini prompt blocked with no candidates and a `prompt_feedback.block_reason` such as `SAFETY`, `PROHIBITED_CONTENT` or `OTHER` |
| `incomplete` | Any other Gemini `finish_reason` except `STOP` or an unspecified one, such as `OTHER` or `MALFORMED_FUNCTION_CALL`; a Gemini image request that finishes with no image |

Both errors subclass `IncompleteOutputError`. The node's failure policy
applies, a retry reserves the node's estimated cost again before it is sent,
and no tool call from the rejected turn runs. An `LLM_MERGE` synthesis raises
the same errors after recording its charge; synthesis has no failure policy,
so the run fails. `LLMArchitect` and `ConstrainedArchitect` retry such a
planning reply with the problem named, up to `max_retries`, and
`LLMSupervisor` makes no revision from such a review. Raising the provider's
`max_tokens` is the usual fix for truncation. Stop reasons outside these
values, including any a custom provider returns, count as complete. The native
`AnthropicMessagesProvider` and `OpenAIResponsesProvider` treat any response
that did not complete as a terminal response error instead.

A node timeout follows its failure policy. Invalid accounting, budget
admission or reconciliation failures, and failures persisting a billed result
are terminal regardless of that policy. See [Cost guardrails](budgets.md).

## Concurrency

`Swarm(parallel=False)`, the default, runs one node at a time on every path.
`execute()` uses the serial executor. `execute_async()`, `resume()`, and
`aresume()` use the async executor with a concurrency of one.
`Swarm(parallel=True)` admits up to `max_concurrency` ready nodes at once
(default 8; `None` means unlimited) on the same paths. A durable run
(`run_store`) records the concurrency of its first execution and keeps it when
resumed.

`Swarm` checks `max_concurrency`, `max_revisions`, and `retry_backoff_s` before
any routing or planning call. An ordinary Swarm applies the executor's rules
when it is constructed and raises `ValueError`. A durable Swarm applies the
durable run's stricter rules, which also require an integer concurrency cap, at
each `plan`, `execute`, or `resume` call and raises `WorkflowBindingError`.

## Provider checks before any call

An ordinary Swarm also checks what its provider declares through
`Provider.feature_support(model)` before it pays for anything. `plan()`,
`aplan()`, `execute()` and `execute_async()` make these checks before the
router or architect is called:

- A `tool_runtime` needs a provider that can call tools.
- An `LLM_MERGE` synthesizer, which calls the run's model after every node
  finishes, needs the per-call ceiling a capped call to that model requires
  and a model the provider accepts.
- When every plan comes from `LLMArchitect`, directly or through a
  `WhiteRabbit` whose tiers are all `LLMArchitect`, every planned node uses
  the run's model, because a generated plan cannot set a node's model or
  estimate. That model then needs the provider's support and, for
  `execute(Task)` and `execute_async(Task)`, the same ceiling.

`plan()` and `aplan()` refuse before planning only what no estimate on the
returned graph can fix: a tool runtime, which every node is offered, an
`LLM_MERGE` synthesis, whose ceiling can come only from the provider, and,
when `LLMArchitect` plans, a run model the provider does not accept. A
missing ceiling for the run's model is not refused there. Set
`metadata["estimated_cost_usd"]` on each node of the returned graph and call
`execute(graph)`, which accepts a node's estimate in place of the provider's
ceiling. `execute(Task)` and `execute_async(Task)` plan and run in one call,
leaving no chance to set estimates, so they refuse the missing ceiling before
routing or planning.

Before the first node call, `execute()`, `execute_async()` and `resume()`
check each node that has not completed or been skipped: its model, its
`attach_dep_artifacts` against a provider that cannot take image inputs, and
its ceiling. This covers caller-built and YAML graphs, plans that other
architects or an overridden `plan()` produce, and generated plans that ask
for image inputs. With a `checkpoint_store`, a rejection here saves a failed
checkpoint like any other failure, so a Swarm with a suitable provider can
resume a paid plan without planning again.

A missing ceiling raises `BudgetEstimateRequired`. Its `node_id` is `None`
when `execute(Task)` or `execute_async(Task)` refused a generated plan's
model before planning,
`"__synthesis__"` (`smythe.graph.SYNTHESIS_NODE_ID`) when an `LLM_MERGE`
synthesis was refused, and otherwise the refused node's id. Other
mismatches raise `ValueError`. Only a feature declared unsupported is
refused: custom providers declare nothing by default and run as before. See
the [support matrix](support-matrix.md) for each provider's declaration.

## Timing

`SwarmResult.timing` reports where a call's time went. Durations come from a
monotonic clock in the calling process, so a wall-clock adjustment cannot
distort them, and durations from different processes are never compared.

| Key | Meaning |
|---|---|
| `elapsed_ms` | Time from the start of the call to its result |
| `summed_node_ms` | Every recorded attempt's duration added together: the summed node time that `PlannerMemory` records as `total_duration_ms` |
| `phases_ms` | Time per phase, below |
| `waiting_ms` | Waiting time by reason, below |
| `resumed` | `true` when the call resumed a checkpoint |
| `latency_complete` | `false` when the call resumed a checkpoint, so `elapsed_ms` covers only part of the execution |
| `segments` | One entry per call that worked on the execution, oldest first; see [timing segments](checkpoint-format.md#timing-segments) |

| Phase | Time measured |
|---|---|
| `routing` | Router classification, when `execute(Task)` planned the graph in the same call and a router is set |
| `planning` | Architect planning, when `execute(Task)` planned the graph in the same call |
| `recovery` | Loading, checking and restoring the checkpoint on resume |
| `execution` | Attempts of work nodes |
| `verification` | Attempts of judge nodes, plus applying saved verdicts and regeneration resets, and a strict policy's final validators |
| `supervision` | Supervisor reviews |
| `synthesis` | Producing the output from the finished graph |
| `checkpoint` | Building and writing checkpoints |
| `waiting` | All waits in `waiting_ms` |

`routing`, `planning`, `recovery` and `synthesis` are `null` when that stage
did not run in the call. A graph planned by an earlier `plan()` call, or built
by hand, reports planning as not measured.

| Wait | Time measured |
|---|---|
| `retry_backoff` | Sleeping before a retry attempt (`retry_backoff_s`) |
| `admission` | A ready node waiting for a free concurrency slot in the async executor, which runs every parallel run and `execute_async()`, `resume()` and `aresume()`; serial `execute()` has no admission queue. When a regeneration or supervisor revision leaves a ready node waiting on new work, its wait starts again when it is next ready |
| `supervision` | Ready work that a supervisor review kept from starting: in the async executor, ready nodes with a free slot; in serial execution, the next pending node |

Phases and waits are totals. In a parallel run nodes overlap, and checkpoint
writes, verification and supervision happen during execution, so the totals
can add up to more than `elapsed_ms`. Comparing `summed_node_ms` with
`elapsed_ms` shows how much work overlapped.

Each attempt's trace span also records its 0-based `attempt`, its `phase`
(`execution`, `verification` or `synthesis`) and an `outcome`: `completed`,
`failed`, `timed_out`, `skipped` or `cancelled`. Spans measure their duration
on the monotonic clock and keep wall-clock `start_time` and `end_time`. These
fields are on `Tracer.spans`; the keys of `Tracer.summary()` and
`SwarmResult.trace` are unchanged. A tracer passed to an executor reports the
same measurements through `Tracer.timing()`, whose `elapsed_ms` starts when the
tracer is created. `Tracer(clock=..., wall_clock=...)` takes the monotonic and
wall clocks to use, for example fixed clocks in tests. Measuring never fails a
run: a phase or wait whose clock readings would give a negative duration, as
with a clock that steps backwards, records zero.

A durable [`run_store`](workflow-accounting.md) run can span several calls and
processes, so its `timing` comes from the journal's wall-clock event times
instead, and `clock` is `wall`. Each lease session is a segment with
`started_at`, `ended_at`, `elapsed_ms` and `interrupted`, which is `true` when
the session never released its lease, as when its process was killed; such a
session ends at its last journal event. `elapsed_ms` is this call's session,
and `run_wall_ms` runs from the run's creation to the result, including time
between calls. `resumed` is `true` and `latency_complete` `false` when an
earlier call held the run, such as the `plan()` call that produced a reviewed
plan. Durable results report no phase totals or waits.

## Terminal failures

In serial execution, every later node remains pending after a terminal
failure, including independent siblings and descendants with `SKIP` policies.
A queued node's policy does not override an earlier node's halt.

In parallel execution, calls already in flight can finish before a failure is
observed. The executor cancels and awaits the remaining active tasks, settles
completed work, and starts no queued sibling after observing the failure.
Cancellation does not establish that a remote provider issued no charge.

An explicit [resume](checkpoint-format.md#resume-semantics) preserves completed
nodes and their costs, resets failed or interrupted nodes, and continues the
pending graph. Invalid-accounting markers require reconciliation first.
Unfinished [verification transitions](verifier.md#recovery-and-concurrent-work)
are recovered before new work or cached-output return.

## Execution and acceptance

A run that completes has executed; whether its output is accepted is a
separate question that only a [strict acceptance
policy](verifier.md#strict-delivery-acceptance) (`Swarm(acceptance=...)`)
answers. Under one, a completed run whose output is rejected, needs review or
is undecided still returns normally, with `output=""`, the text in
`rejected_output` and the decision in `acceptance`; cost is reported as for
any other run. A gate verifier that raises does not stop a strict run: the
gate ends undecided and so does acceptance. Without a strict policy such a
verifier stops the run as a terminal failure, and results carry no decision.
An unfinished strict run resumes only under the policy it began with, and a
fresh strict run refuses a graph whose judge already completed with a recorded
verdict, since only verdicts read in the run decide it.

## Provider connections

Serial execution runs each node under its own `asyncio.run()`, as do
`Swarm.plan()`, `execute()`, and `resume()`. `AnthropicProvider`,
`OpenAIProvider`, `OpenAIImageProvider`, and `GeminiProvider` therefore keep
one SDK client per event loop. Calls on the same loop share its connection
pool, so a parallel run reuses connections across its fan-out. A client is
never used on another loop, and it closes on its own loop when that loop shuts
down (`asyncio.run()` does this before closing it).

## Deep graphs

Validation, cycle detection, dependency ordering, and depth calculation use
iterative traversal. A chain does not depend on Python's recursion limit or
the order in which its nodes appear. Serial execution preserves depth-first
dependency order, including the declared order of sibling dependencies.

Regression tests validate 5,000-node chains in forward, reverse, and shuffled
order, check a 5,000-node fork/join graph, and execute all 5,000 steps of a
reverse-ordered chain with an offline provider. Revision tests cover deep
rewiring and reject cycles or missing dependencies before changing the graph.
Reading `depth` on a cyclic graph raises `ValueError`.

These are correctness checks, not throughput measurements. Serial execution
still rechecks pending work after each step so supervision and verification
can change the remaining graph. Flat artifact fan-out has a separate
[Jobs scale and recovery protocol](../benchmarks/jobs_scale_benchmark.md).

[YAML example](../examples/01_pipeline.yaml) ·
[Crash and resume example](../examples/04_resume_after_crash.py) ·
[Architecture](architecture.md).
