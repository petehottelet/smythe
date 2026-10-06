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
that did not complete as a terminal response error instead, unless a
[retry policy](#retry-policy) lists `quality_defect`. Its
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
| `input_error` | HTTP 404 or 413 `ProviderRequestRejectedError`; any other 4xx status error not listed below; `WorkflowBindingError`, `BudgetEstimateRequired`, `NoAllowedProfileError` and `ArtifactIntegrityError` | No |
| `authority_denied` | HTTP 401 or 403 `ProviderRequestRejectedError` or status error | No |
| `transient` | HTTP 429 `ProviderRequestRejectedError`; 408, 409, 429, 499 and 5xx status errors; timeouts other than those below; any exception this table does not list | No |
| `provider_rejected` | `ProviderResponseError`: the provider answered and the charge is known, but the response cannot be used. Also `OutputTruncatedError`, `OutputRefusedError` and `ReadToolError` | `ProviderResponseError` only |
| `quality_defect` | `ToolLoopLimitError`; `NativeOutputDefectError`, a native refusal or truncation with a known charge under a retry policy that lists this class | No |
| `unknown_outcome` | `ProviderAccountingError`: a native call's charge is unresolved. `UncertainDispatchError`: a node timed out with its call sent and unanswered, under a policy with `retain_uncertain_dispatch` | `ProviderAccountingError` only |
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
A zero-cost rejection's read-only `ProviderRequestRejectedError.failure_class`
gives its class from its status code, on a durable run and under a retry
policy alike. A `VerificationRecoveryError` leaves its
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
| `retain_uncertain_dispatch` | `False` | Keep the reservation of a node call that was sent and never answered as unknown cost, instead of releasing it (see [calls without an answer](#calls-without-an-answer)) |

Under the default policy, a 401 or 403 (`authority_denied`), a 404 or 413
(`input_error`), a refused or truncated reply (`provider_rejected`) and
`ToolLoopLimitError` (`quality_defect`) fail the node without another call,
while a rate limit, server error, timeout or connection failure is retried up
to `max_retries`. A terminal failure stops the run whatever the list says. A
tool error that goes back to the model as a tool result belongs to the same
attempt and is not a retry.

Under a policy, the native `OpenAIResponsesProvider` and
`AnthropicMessagesProvider` settle a 401, 403, 404, 413 or 429 response whose
body is only the provider's plain error object at zero cost, as a durable
journal does, and raise `ProviderRequestRejectedError`, whose class follows
its status: the default policy retries a native 429. A 400, any other status,
or a body with usage or another field still leaves the bill unknown and stops
the run. A native refusal (`stop_reason` `refusal`) or truncation
(`max_tokens`) has a known charge. Under a policy that lists
`quality_defect`, the executor records that charge once and raises
`NativeOutputDefectError`, an `IncompleteOutputError`, so the node's failure
policy applies; under any other policy, or for another stop reason, it still
stops the run. Neither error leaves an unresolved-response marker, so a run
that fails with one resumes like any other failed run. The native response's
`Retry-After` header is not kept, so a native 429 waits for the policy's
backoff.

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
own, each [model profile](#model-profiles)'s, a `WhiteRabbit` classifier's,
that of each `LLMArchitect` or `ConstrainedArchitect` that could plan the
run (directly, as a router tier, or built from `planning_provider`), and an
`LLMSupervisor`'s when `max_revisions` lets it review. Otherwise the run is
refused with `ValueError` before routing, planning or any node call.
`execute(graph)` and `resume()` check the run's provider, each profile's and
the supervisor's before their first call, even when only synthesis is left
to run, and so do `Executor` and `AsyncExecutor` run without a Swarm before
their first node call, for the providers their unfinished nodes call and
each profile's, since a supervisor may add a node that names one:

| Provider | Declared `sdk_retries` |
|---|---|
| `OfflineProvider`, `OpenAIResponsesProvider`, `AnthropicMessagesProvider` | `0` |
| `GeminiProvider` | `0` with its own client; nothing with a caller-owned `_client` |
| `AnthropicProvider`, `OpenAIProvider`, `OpenAIImageProvider` | Its `max_retries`, or a caller-owned client's `max_retries`; nothing at the SDK default |
| Custom `Provider` | Nothing unless its `feature_support()` declares it |

So `AnthropicProvider(max_retries=0)` and `OpenAIProvider(max_retries=0)` run
under a policy, and both providers at their SDK defaults are refused. A
durable `run_store` run records the policy in its recipe and counts each
node's attempts from its journal, so resume continues them; it waits its
backoff, never a `Retry-After` ([durable retry
policy](workflow-accounting.md#retry-policy)). On an ordinary run without
`persist_attempts` a policy is not saved in checkpoints: a resumed run follows
the policy of the Swarm that resumes it, with a fresh allowance of attempts.
`Executor` and `AsyncExecutor` also take `sleep`, which replaces the wait
before a retry and a [capacity](#provider-capacity) wait and is awaited when
it returns an awaitable, and `rng`, which draws the jitter, so tests can drive
waits from a manual clock.

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
reports its own as the `attempts_used` property, in the same shape. The saved
allowance applies to the executor's first run only: running another graph on
the same executor starts a new run, with a fresh allowance for every node.

A durable run's journal records every attempt as its call, so its executor
takes `attempts_spent=` instead: a callable that returns the attempts a
node's current generation has used, which each node reads when it starts.
An executor whose calls are journal-managed takes a policy only with it. See
[durable retry policy](workflow-accounting.md#retry-policy).

### Calls without an answer

By default, when a node's timeout or a cancellation ends a call that
`AnthropicProvider`, `OpenAIProvider`, `OpenAIImageProvider`,
`GeminiProvider` or a custom provider has sent, the executor releases the
call's reservation and records no cost, although the provider may still
charge for it. `RetryPolicy(retain_uncertain_dispatch=True)` keeps that
exposure instead. A call counts as sent from the moment the executor hands it
to the provider, and is answered once the provider returns a result or raises
an error of its own, including its client's own timeout. A call that the
node's `timeout_s`, the policy's deadline or a cancellation (a `HALT`
sibling's failure, a regeneration reset, or a run stopping on an exception
or interrupt) ends before then keeps its reservation as the node's recorded
cost, with `cost_usd_unknown` and `cost_usd_is_estimate` set on the node.
`total_cost_usd` is unchanged by the transfer, `cost_is_complete` becomes
`False` and `cost_contains_estimates` `True`: the reservation, whether a
node's estimate, a provider's ceiling or the async executor's token default
(`estimated_tokens_per_node`), estimates the charge and does not confirm
it. A serial node that reserved nothing is marked unknown at no amount. See
[cost guardrails](budgets.md#calls-sent-without-an-answer).

A timeout of such a call raises `UncertainDispatchError`, a `TimeoutError`
whose class is `unknown_outcome`, so the default policy does not call again:
another call could be charged as well. A policy that lists `unknown_outcome`
retries it, and each retry reserves again beside the kept exposure, so a
budget that cannot hold both stops the run with `SentinelAlert` before the
retry's call. When the deadline ends the call, the node still fails with
`RetryDeadlineError` and keeps the exposure. A timeout or cancellation
before the call is sent, such as during a tool call or before a tool session
opens, releases the reservation as before.

The kept cost is ordinary recorded cost: a version 4 checkpoint carries it,
and resume restores the amount and both flags, as releases before this
option do. With `persist_attempts` the checkpoint is version 5 and records
the option with the policy, so the run resumes only under an equal policy. A
process that is killed outright runs no handler, so a call in flight when it
dies is not recorded. Native providers keep their own settlement: an
unresolved native bill stays terminal with its live reservation.

## Concurrency

`Swarm(parallel=False)`, the default, runs one node at a time on every path.
`execute()` uses the serial executor. `execute_async()`, `resume()`, and
`aresume()` use the async executor with a concurrency of one.
`Swarm(parallel=True)` admits up to `max_concurrency` ready nodes at once
(default 8; `None` means unlimited) on the same paths. A durable run
(`run_store`) records the concurrency of its first execution and keeps it when
resumed.

When a failure stops a run, the async executor cancels the nodes in flight,
unless a [termination policy](#draining-in-flight-work) lets them finish first.

`Swarm` checks `max_concurrency`, `max_revisions`, and `retry_backoff_s` before
any routing or planning call. An ordinary Swarm applies the executor's rules
when it is constructed and raises `ValueError`. A durable Swarm applies the
durable run's stricter rules, which also require an integer concurrency cap, at
each `plan`, `execute`, or `resume` call and raises `WorkflowBindingError`.

### Provider capacity

`Swarm(capacity_policy=CapacityPolicy(...))`, or `capacity_policy=` on either
executor, limits the calls that each provider type and model takes on an
ordinary run. Without a policy none of this applies, and no provider hint is
read. A `CapacityLimit` has two fields, both `None` by default:

| Field | Meaning |
|---|---|
| `max_in_flight` | How many nodes using the model the async executor runs at once, so at most that many of their calls are in flight |
| `min_interval_s` | The least time, in seconds, between the starts of two calls to the model |

`CapacityPolicy.limits` maps a model name, or a `(provider type name, model)`
pair such as `("OpenAIResponsesProvider", "gpt-6-astra")`, to a limit. For each
field a pair's value wins, then the model's, then the provider's
`capacity_hint(model)`, which returns `None` (the default) or a dict with
either field. A field that nothing sets is not limited. A hint that is not
`None` or such a dict with valid values raises `ValueError`: for the run's
model before routing or planning, and for every model a pending node uses
before any node call.

```python
from smythe import CapacityLimit, CapacityPolicy, OpenAIResponsesProvider, Swarm

swarm = Swarm(
    model="gpt-6-astra", provider=OpenAIResponsesProvider(),
    parallel=True, max_concurrency=16,
    capacity_policy=CapacityPolicy(
        limits={"gpt-6-astra": CapacityLimit(max_in_flight=4, min_interval_s=0.25)},
        round_robin=True,
    ),
)
```

When a ready node's model has no free slot, the async executor passes over it:
the node stays ready, and ready work behind it, such as another model's nodes,
is admitted instead. It is admitted once a node of its model finishes, and it
never fails the run. While it waits it takes no slot under `max_concurrency`
and reserves no budget, so it leaves
[held allowance](budgets.md#held-allowance-for-checks) alone. A reservation
that the budget refuses still stops admission and the run, as without a
policy. The serial executor runs one node at a time, so it has no admission
limit to apply.

Every provider call of a node waits for its turn in both executors: its
first call, each tool turn and each retry. A call starts no sooner than
`min_interval_s` after the previous call to the same provider type and model
started, and not during a cooldown. A transient status error whose
`Retry-After` asks for a wait (read as under a [retry policy](#retry-policy))
starts a cooldown: no call to that provider type and model starts for that
long, at most `cooldown_max_s` (default 60; 0 turns cooldowns off). A call
already waiting its turn waits for the cooldown too. Calls take their turns in
the order they ask for them. The wait is measured on the tracer's clock and
made with the executor's `sleep` when one is given. It happens inside the
attempt, so it is part of the attempt's span and counts toward the node's
`timeout_s` and a retry policy's deadline. A call that a timeout, a deadline
or a cancellation stops while it waits gives up its turn without starting: the
next call is spaced from the last call that did start, so a retry waits only
for the next free turn. An `LLM_MERGE` synthesis by the built-in `Synthesizer`
takes its turn the same way; it runs after every node has settled, so it is
the only call in flight. Routing, planning and supervisor reviews are not
limited.

With `round_robin=True`, the async executor admits ready nodes in turn across
branches. A branch is the ready nodes that share a first dependency, and
nodes without dependencies form one branch. Each turn takes one node from
each branch, starting after the branch admitted last, so a wide fan-out
cannot hold back another branch's ready work. Without it, ready nodes are
admitted in the order they became ready.

Capacity waits are reported as `waiting_ms["capacity"]` (see [timing](#timing)).
Nothing about the policy is saved: a resumed run follows the resuming
`Swarm`'s policy, and each run starts with fresh limits. Durable `run_store`
runs refuse `capacity_policy` with `WorkflowBindingError`, and an executor
whose calls a journal manages refuses it with `ValueError`.

## Provider checks before any call

An ordinary Swarm also checks what its provider declares through
`Provider.feature_support(model)` before it pays for anything. `plan()`,
`aplan()`, `execute()` and `execute_async()` make these checks before the
router or architect is called:

- A `tool_runtime` needs a provider that can call tools.
- A `retry_policy` needs every provider the run can call, including a
  router classifier's, an architect's, an `LLMSupervisor`'s and each model
  profile's, to declare `sdk_retries=0` (see [retry policy](#retry-policy)).
- Each [model profile](#model-profiles) needs a provider that declares the
  features it requires and supports its model.
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

## Model profiles

`Swarm(profiles=...)` lists the providers and models that an ordinary run's
nodes may be routed to. Each is a `ModelProfile` with a distinct name:

```python
from smythe import AnthropicProvider, ModelProfile, OpenAIResponsesProvider, Swarm

sol = ModelProfile(
    "sol", OpenAIResponsesProvider(), "gpt-5.6-sol",
    max_cost_per_call_usd=0.05, requires={"exact_accounting"},
)
swarm = Swarm(model="claude-opus-5-5", provider=AnthropicProvider(),
              profiles=[sol], max_budget_usd=2)
```

A node whose `metadata["profile"]` names a profile calls that profile's
provider with its model, and its `metadata["model"]` is set to that model.
Before each call it reserves `max_cost_per_call_usd`, unless it sets
`metadata["estimated_cost_usd"]`; its held check under `reserve_for_checks`
and its persisted attempts use the same amount. Every other node calls the
run's provider with its own model or the run's. Synthesis calls the run's
provider with the run's model, and so does a node a supervisor adds without
a profile.

Only developer code names a profile: caller-built and YAML graphs
(`Swarm.from_yaml(..., profiles=...)`), architects written in code,
`ConstrainedArchitect` templates and the nodes a
[`Supervisor` you write](supervisor.md#writing-your-own) adds. Such an
addition is routed like any profiled node, and its saved `metadata["model"]`
is its profile's, so resume routes it the same way. A revision that adds a
node naming a profile the run does not allow, or setting another model than
its profile's, is refused before it applies and traced as
`revision_rejected`, like any invalid revision. A generated plan may set no
node metadata but `role`, and an `LLMSupervisor` proposes nodes without
metadata, so model output cannot choose a profile.

An `LLMSupervisor` without a `model` of its own reviews a node with that
node's model on its own provider, and a profiled node's model is its
profile's. So while `max_revisions` lets such a supervisor review, every
profile must call the supervisor's provider object; otherwise the run is
refused with `NoAllowedProfileError` before routing, planning or any call.
Give the supervisor a model, `LLMSupervisor(provider, model=...)`, to review
with that model whatever the node's profile.

`requires` lists the features that the profile's provider must declare
`True` for its model through `Provider.feature_support()`: `tools`,
`attachments`, `image_output` and `exact_accounting`. A feature the provider
does not declare does not count. A run raises `NoAllowedProfileError`, a
`ValueError`, before routing, planning or any node call for:

- a profile whose provider does not declare `True` a feature it requires,
  declares its model unsupported, or prices the model with a table whose
  retirement date for it has passed (UTC);
- a profile whose provider declares that it cannot call tools, in a run
  with a `tool_runtime`;
- a node that names a profile the run does not allow, including any profile
  when the Swarm has none, or that sets a `model` other than its profile's.

`execute(graph)` and `resume()` make the same checks before their first
call, and check each unfinished profiled node's image inputs and ceiling
against its profile's provider. Under a [retry policy](#retry-policy) each
profile's provider must declare `sdk_retries=0`. `Executor` and
`AsyncExecutor` take `profiles` too, fail a node that names any other
profile or sets another model than its profile's before its call, and
refuse a model-less `LLMSupervisor` with another provider's profile before
their first call while revisions remain.

A run with a profiled node writes
[version 5 checkpoints](checkpoint-format.md#model-profiles-version-5) that
list `model_profiles`, because a build that does not route profiles would
send those nodes to the run's provider. Resume routes them to the resuming
Swarm's profiles of the same names, and refuses a node whose profile that
Swarm does not allow or whose saved model differs from its profile's. A
completed run returns its saved result without them. Durable `run_store`
runs refuse `profiles`, and any node that names one, with
`WorkflowBindingError`.

### Profile fallback

A profile's `fallback` names another of the run's profiles, which a node's
retry calls in its place:

```python
from smythe import FailurePolicy, ModelProfile, OpenAIResponsesProvider, RetryPolicy, Swarm
from smythe.graph import Node

responses = OpenAIResponsesProvider()
sol = ModelProfile("sol", responses, "gpt-5.6-sol", max_cost_per_call_usd=0.05,
                   fallback="astra")
astra = ModelProfile("astra", responses, "gpt-6-astra", max_cost_per_call_usd=0.25)
swarm = Swarm(model="gpt-6-astra", provider=responses, profiles=[sol, astra],
              retry_policy=RetryPolicy(retry_classes={"transient", "quality_defect"}),
              max_budget_usd=2)
step = Node(id="summary", label="Summarize the findings", metadata={"profile": "sol"},
            failure_policy=FailurePolicy.RETRY, max_retries=2)
```

After an attempt on a profile fails with a `transient`, `provider_rejected`
or `quality_defect` [failure](#failure-classes), such as a rate limit, a
server error, a truncated reply or a tool loop that ran out, the node's next
attempt calls that profile's fallback. The fallback is a new attempt, not an
extra one: it needs a `RETRY` node with attempts left and, under a
[retry policy](#retry-policy), a listed class and time before the deadline,
and it waits the backoff or `Retry-After` that any retry waits. A node's
attempts on all its profiles together therefore stay within
`1 + max_retries`, and with `persist_attempts` within the allowance that
resume continues. A fallback's own fallback serves the attempt after it, and
later retries stay on the last profile of the chain. A retry after any other
failure calls the same profile again. That includes `unknown_outcome`, whose
call may still run and be charged. A node timeout that ends a call sent and
not yet answered has that class only under
`RetryPolicy(retain_uncertain_dispatch=True)` (see [calls without an
answer](#calls-without-an-answer)). Without that option the timeout is an
ordinary `transient` one, so the node's retry calls the fallback while the
first call may still run and be charged; set the option to keep such a
retry on its profile.

Each attempt's trace span records the profile it called as
`metadata["profile"]`, and an attempt that fell back records the profile it
replaced as `metadata["fallback_from"]`. Both are on `Tracer.spans`; the keys
of `Tracer.summary()` and `SwarmResult.trace` are unchanged. A fallback
attempt reserves its own profile's `max_cost_per_call_usd` before its call,
unless the node sets `estimated_cost_usd`. The reservation that the failed
attempt left is released first, so a budget that cannot hold the fallback's
ceiling stops the run with `SentinelAlert` before the call. Under a
[capacity policy](#provider-capacity), the fallback's calls take its
provider type and model's turns, and in the async executor the node's slot
moves to that model: the node gives back the slot it held, then waits until
the fallback's model has a free one.

The node's `metadata["profile"]` and `metadata["model"]` keep the profile it
names, and checkpoints save no choice of profile. The first attempt in each
process, a resumed node's included, calls the profile the node names, as does
a node that a regeneration or a cancellation runs again.

A fallback must be one of the run's profiles and require at least the
features of the profile it serves, and a chain must not return to a profile
it left; otherwise `Swarm`, `Executor` and `AsyncExecutor` raise `ValueError`
when built. Because a fallback is one of the run's profiles, the checks before
any call cover it, such as `sdk_retries=0` under a retry policy. Each
unfinished node is also checked against every profile it may fall back to,
including its image inputs and, under a capacity policy, each provider's
capacity hint.

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
| `capacity` | Under a [capacity policy](#provider-capacity): a ready node from when a full limit first passed it over until it is admitted (its wait before then is an admission wait), and each call waiting for its turn |
| `drain` | After a failure, the time the async executor waited for nodes in flight under a [termination policy](#draining-in-flight-work) |

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
Cancellation does not establish that a remote provider issued no charge: a
retry policy with `retain_uncertain_dispatch` keeps a cancelled call's
reservation as unknown cost (see [calls without an answer](#calls-without-an-answer)).

### Draining in-flight work

`Swarm(termination=TerminationPolicy(drain_timeout_s=30))`, or `termination=`
on `AsyncExecutor`, lets the nodes in flight finish their current attempts
before the run stops. After a failure that stops the run, such as a `HALT`
node's error, a `RETRY` node out of attempts or a terminal failure, and after
a ready node's budget reservation is refused at admission, the async executor:

1. Starts no queued node, including one admitted in the same pass as a
   refused node, whose work had not begun.
2. Lets each node in flight finish its current attempt, including any further
   tool turns that attempt takes, for up to `drain_timeout_s` seconds. Each
   such turn is a new provider call, reserved and charged under the budget as
   usual. A node that finishes keeps its result and cost and is checkpointed
   as any finished node is. Supervision and regeneration do not run after
   it, and a failure of its own does not replace the run's error.
   No retry starts: a `RETRY` node whose attempt fails during the drain, or
   that is waiting out a retry backoff when the drain begins, returns to
   pending with its unused reservation released, as a cancelled node does,
   so a resume runs it again.
3. Cancels the nodes still in flight at the deadline, as without a policy. A
   cancelled native call's unresolved bill stays terminal with its live
   reservation. A cancelled call to any other provider keeps its reservation
   as unknown cost only under a retry policy with
   `retain_uncertain_dispatch`; otherwise the reservation is released and no
   cost is recorded (see [calls without an answer](#calls-without-an-answer)).
4. Raises the original error.

A cancellation of the run, an interrupt such as `KeyboardInterrupt`, and a
failure of class `security_stop` cancel the nodes in flight at once whatever
the policy, and so does a cancellation during a drain. The serial executor
behind `execute()` with `parallel=False` runs one node at a time, so it has
nothing in flight to drain, nor does the async executor at a concurrency of
one. A durable `run_store` run refuses a policy with `WorkflowBindingError`
when the `Swarm` is built.

`AsyncExecutor.drain_record` reports how the last run drained: `settled` names
the nodes in flight whose work ended during the drain, `cancelled` those it
stopped, still in flight when it ended or before a retry, and `deadline_hit`
whether the deadline ended it. It is `None` when the run did not drain. A
`Swarm` saves it in the failed checkpoint as `control.drain`, for information
only: the checkpoint stays version 4, and resume keeps the nodes the drain
completed, runs the failed, cancelled and never-started ones, and follows the
resuming `Swarm`'s own policy. The time spent draining is the `drain` wait in
[timing](#timing). The executor's `sleep`, when given, times the drain's
deadline, so tests can end it from a manual clock.

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
valid decision covers it. A durable run with a
[lineage policy](lineage.md) also gets a [lineage summary](runs.md#lineage):
coverage, source references, predecessor and successors, the published
document's integrity and the run's derived status, with no source text and
no new exit code. See [Inspect saved runs](runs.md).

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
