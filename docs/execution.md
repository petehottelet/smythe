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
that did not complete as a terminal response error instead. Its
`ProviderResponseError.stop_reason` uses the same values:

| `stop_reason` | Native signal |
|---|---|
| `max_tokens` | Responses `incomplete_details.reason` `max_output_tokens`; Messages `stop_reason` `max_tokens` |
| `model_context_window_exceeded` | Messages `stop_reason` `model_context_window_exceeded` |
| `refusal` | A Responses message with a `refusal` part; Messages `stop_reason` `refusal` |
| `content_filter` | Responses `incomplete_details.reason` `content_filter` |
| `incomplete` | Any other incomplete Responses reply; Messages `stop_reason` `pause_turn` |

It is `None` for any other unusable response. The error's message never
includes it, so a durable run matches the text its journal saved for a
rejected call.

A node timeout follows its failure policy. Invalid accounting, budget
admission or reconciliation failures, and failures persisting a billed result
are terminal regardless of that policy. See [Cost guardrails](budgets.md).

## Failure classes

Every exception a node attempt raises has one failure class. The attempt's
trace span records it as `metadata["failure_class"]` on `Tracer.spans`; the
keys of `Tracer.summary()` and `SwarmResult.trace` are unchanged, and
checkpoints do not store it. Without a [retry policy](#retry-policy), a class
describes a failure and does not change what is retried: the exceptions
marked terminal stop the run whatever the node's failure policy, and that
policy decides every other one.

| Class | Exceptions | Terminal |
|---|---|---|
| `input_error` | HTTP 404 or 413 `ProviderRequestRejectedError`; any other 4xx status error not listed below; `WorkflowBindingError`, `BudgetEstimateRequired` and `ArtifactIntegrityError` | No |
| `authority_denied` | HTTP 401 or 403 `ProviderRequestRejectedError` or status error | No |
| `transient` | HTTP 429 `ProviderRequestRejectedError`; 408, 409, 429, 499 and 5xx status errors; timeouts; any exception this table does not list | No |
| `provider_rejected` | `ProviderResponseError`: the provider answered and the charge is known, but the response cannot be used. Also `OutputTruncatedError`, `OutputRefusedError` and `ReadToolError` | `ProviderResponseError` only |
| `quality_defect` | `ToolLoopLimitError` | No |
| `unknown_outcome` | `ProviderAccountingError`: a native call's charge is unresolved | Yes |
| `terminal_accounting` | `BudgetValidationError`, `SentinelAlert` and `BudgetReconciliationError` | Yes |
| `persistence` | `NodeFinalizationError` and `VerificationRecoveryError` | Yes |
| `workflow` | `WorkflowError` and its subclasses, such as `WorkflowQuoteError` | Yes |
| `validator_error` | None: the class of a validator outcome `error` | |
| `security_stop` | None yet: reserved for an explicit security stop | |
| `deadline` | `RetryDeadlineError`: a retry policy's `deadline_s` ran out | No |

A failed or skipped node keeps its last error's message as its result, which
checkpoints save and dependents receive, and its trace span and log line keep
it too. Provider and service messages can name an account, so each copy has
provider account identifiers and keys in known formats, such as
organization and project ids, replaced with placeholders like
`org-[redacted]`; the exception the run raises is left unchanged. Native
response error markers, Jobs call errors and Autotune trial errors are kept
the same way.

A status error is any exception that carries an HTTP status in `status_code`
or `response.status_code`, as the OpenAI and Anthropic SDK status errors do.
A durable rejection's read-only `ProviderRequestRejectedError.failure_class`
gives its class from its status code. A `VerificationRecoveryError` leaves its
attempt's span open, so no recorded span carries its class. The [support
matrix](support-matrix.md) lists who calls again after a failure on each path.

## Retry policy

Without a policy, `RETRY` calls again after every failure that is not
terminal, and the SDK client inside `AnthropicProvider` or `OpenAIProvider`
can send a failed request again before the executor sees the failure.
`Swarm(retry_policy=RetryPolicy(...))`, or `retry_policy=` on either
executor, makes the executor the only retry owner of an ordinary run's node
calls. A `RetryPolicy` is frozen and checks its fields when built:

| Field | Default | Meaning |
|---|---|---|
| `retry_classes` | `{"transient"}` | The [failure classes](#failure-classes) a `RETRY` node calls again for: any of `transient`, `input_error`, `authority_denied`, `provider_rejected`, `quality_defect` and `unknown_outcome`. Any other failure fails the node at once |
| `honor_retry_after` | `True` | Wait as long as a transient status error's `Retry-After` asks, in place of the backoff |
| `backoff_max_s` | `60.0` | The longest wait before any retry; a longer `Retry-After` is cut to it |
| `deadline_s` | `None` | Seconds that one node's attempts and waits share, from the start of its first attempt |
| `persist_attempts` | `False` | Save each attempt as spent before its call, so resume continues each node's allowance instead of refilling it (see [persisted attempts](#persisted-attempts)) |

Under the default policy, a 401 or 403 (`authority_denied`), a 404 or 413
(`input_error`), a refused or truncated reply (`provider_rejected`) and
`ToolLoopLimitError` (`quality_defect`) fail the node without another call,
while a rate limit, server error, timeout or connection failure is retried up
to `max_retries`. A terminal failure stops the run whatever the list says. A
tool error that goes back to the model as a tool result belongs to the same
attempt and is not a retry.

`Retry-After` is read from the `response.headers` of a status error whose
status is transient (408, 409, 429, 499 or 5xx): `retry-after-ms`, then
`retry-after`, as a number of seconds. An HTTP date is not read. Without such
a header, the wait is the full-jitter backoff that doubles from
`retry_backoff_s`, capped at `backoff_max_s`.

With `deadline_s`, each attempt's timeout is the node's `timeout_s` or the
time left, whichever is shorter, and a retry whose wait would end at or after
the deadline is not made. The node then fails with `RetryDeadlineError`, a
`TimeoutError` whose class is `deadline`, and its failure policy settles it
like any other failure. The retry not made still gets a trace span, with its
attempt number, outcome `timed_out`, class `deadline`, the error and no
duration, so the trace shows why the node stopped. A `TimeoutError` that the
call raises itself before then, such as an HTTP client's own request
timeout, stays a `transient` timeout and is retried. The deadline also
bounds the single attempt of a `HALT` or `SKIP` node. It is measured on the
tracer's monotonic clock.

A run under a policy needs every provider call sent once, planning and
supervision included. Each provider the run can call must declare
`ProviderFeatures.sdk_retries` as `0` through `feature_support()`: the run's
own, a `WhiteRabbit` classifier's, that of each `LLMArchitect` or
`ConstrainedArchitect` that could plan the run (directly, as a router tier,
or built from `planning_provider`), and an `LLMSupervisor`'s when
`max_revisions` lets it review. Otherwise the run is refused with
`ValueError` before routing, planning or any node call. `execute(graph)` and
`resume()` check the run's provider and the supervisor's before their first
call, even when only synthesis is left to run, and so do `Executor` and
`AsyncExecutor` run without a Swarm before their first node call:

| Provider | Declared `sdk_retries` |
|---|---|
| `OfflineProvider`, `OpenAIResponsesProvider`, `AnthropicMessagesProvider` | `0` |
| `GeminiProvider` | `0` with its own client; nothing with a caller-owned `_client` |
| `AnthropicProvider`, `OpenAIProvider`, `OpenAIImageProvider` | Its `max_retries`, or a caller-owned client's `max_retries`; nothing at the SDK default |
| Custom `Provider` | Nothing unless its `feature_support()` declares it |

So `AnthropicProvider(max_retries=0)` and `OpenAIProvider(max_retries=0)` run
under a policy, and both providers at their SDK defaults are refused. Durable
`run_store` runs refuse `retry_policy` with `WorkflowBindingError`. Without
`persist_attempts` a policy is not saved in checkpoints: a resumed run follows
the policy of the Swarm that resumes it, with a fresh allowance of attempts.
`Executor` and `AsyncExecutor` also take `sleep`, which replaces the wait
before a retry and is awaited when it returns an awaitable, and `rng`, which
draws the jitter, so tests can drive waits from a manual clock.

### Persisted attempts

`RetryPolicy(persist_attempts=True)` makes a node's allowance of attempts
(`1 + max_retries` for `RETRY`, one otherwise) hold across crashes and
resumes. Before each attempt's call the executor records it as spent and, with
a `checkpoint_store`, saves the checkpoint, whatever
`checkpoint_every_n_nodes` is. An attempt therefore counts once it starts,
including one that a crash or a cancellation interrupts; if that save fails,
the attempt is not counted and the node stops with the save's error before
its call.

The checkpoints of such a run are [version 5](checkpoint-format.md#attempt-allowance-version-5):
they record the policy and each node's attempts in its current
`execution_generation`. Resume starts each node from the attempts its
generation already used, so a run, its crash and its resume together make at
most `1 + max_retries` calls for a `RETRY` node. A node with no attempt left
fails at once with a `RuntimeError` naming its allowance, without a call, and
its failure policy settles it as usual: a `SKIP` node is skipped. A
regenerated node starts a new generation and gets a fresh allowance, so each
generation keeps its own bound. `deadline_s` is measured within one process,
so it starts again at the node's first attempt after a resume.

Resume refuses with `ValueError`, before any provider call, a run saved under a
policy other than the resuming `Swarm`'s `retry_policy`, and a run saved
without persisted attempts when the resuming policy persists them, since its
attempts so far are unknown. A completed run returns its saved result under
any policy. Each executor takes the saved allowance as `attempts_used=` and
reports its own as the `attempts_used` property, in the same shape.

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
- A `retry_policy` needs every provider the run can call, including a
  router classifier's, an architect's and an `LLMSupervisor`'s, to declare
  `sdk_retries=0` (see [retry policy](#retry-policy)).
- An `LLM_MERGE` synthesizer, which calls the run's model after every node
  finishes, needs the per-call ceiling a capped call to that model requires
  and a model the provider accepts.
- When every plan comes from `LLMArchitect`, directly or through a
  `WhiteRabbit` whose tiers are all `LLMArchitect`, every planned node uses
  the run's model, because a generated plan cannot set a node's model or
  estimate. That model then needs the provider's support and, for
  `execute(Task)` and `execute_async(Task)`, the same ceiling.

`plan()` and `aplan()` refuse before planning only what no estimate on the
returned graph can fix: a tool runtime, which every node is offered, a
provider whose SDK may retry under a retry policy, an
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
`attach_dep_artifacts` against a provider that cannot take image inputs, its
ceiling and, under a retry policy, the SDK retries its provider declares.
This covers caller-built and YAML graphs, plans that other
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
fresh strict run refuses a graph whose judge already completed, since only
verdicts its judges produce in the run decide it. A strict run is also
refused before any provider call when it could not decide its output, for
example a later step that no gate or final validator checks
([acceptance obligations](verifier.md#acceptance-obligations)).

For a saved run, `smythe runs inspect` reports execution, acceptance and cost
as separate blocks without changing the checkpoint or journal, and exits with
10 when a completed run's output was rejected or needs review and 11 when no
valid decision covers it. See [Inspect saved runs](runs.md).

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
