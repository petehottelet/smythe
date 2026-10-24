# Cost guardrails

`Sentinel` validates cost policy, provider usage, reservations, and restored
charges before changing its ledger. Monetary values must be finite, nonnegative
numbers. Token counts must be nonnegative integers. Booleans, numeric strings,
NaN, infinity, and negative values raise `BudgetValidationError`.

`CompletionResult` validates usage when it is constructed. The ledger validates
it again when recording a call, including objects mutated by custom providers.
Explicit dollar costs do not bypass token validation. Valid zero-cost calls and
exact-fit reservations remain supported.

## Failed accounting

Malformed accounting stops serial and parallel execution under every node
failure policy. It cannot trigger a retry or release the affected reservation.
The runtime cancels active siblings, retains charges from completed calls, and
marks the affected node with `accounting_invalid` in its checkpoint. A saved
`budget.accounting_error` also records workflow-level failures such as synthesis.

Resume rejects an unresolved accounting marker before dispatching work. An
operator must reconcile provider charges, repair the checkpoint's cost records
and malformed configuration, and only then clear the marker. A completed-run
shortcut also validates saved costs before returning its result.

[Jobs](jobs.md#resume-and-unknown-outcomes) records invalid accounting as an
unknown outcome and stops queued dispatches across restarts. Already-dispatched
calls settle normally. An explicitly acknowledged reroll retains the earlier
unknown exposure rather than erasing a possible charge.

Valid reported charges that exceed a reservation are different: the ledger
records the actual charge, then raises `BudgetReconciliationError` and stops
execution. It never clamps an incurred charge to make a budget appear satisfied.

## Calls sent without an answer

On an ordinary run, a call that a node's timeout or a cancellation ends after
it was sent releases its reservation and records no cost, although the
provider may still charge for it. Under a
[retry policy](execution.md#calls-without-an-answer) with
`retain_uncertain_dispatch=True`, the ledger keeps that reservation as the
node's recorded cost instead (`Sentinel.retain_unknown(node_id)`), with
`cost_usd_unknown` and `cost_usd_is_estimate` on the node. `total_cost_usd`
does not change, since the reservation already counted against the budget,
and later admissions still see it. The run reports `cost_is_complete=False`
and `cost_contains_estimates=True`: a reservation estimates the charge and
does not confirm it. A node without a reservation, such as a serial node
without an estimate, is marked unknown at no amount. A verifier's
reservation converted from a [held allowance](#held-allowance-for-checks)
is kept the same way, and other holds are untouched.

Checkpoints record the kept amount in `budget.node_costs` and the flags in the
node's metadata, so resume restores the exposure and its flags, and a retry or
a resumed attempt adds its own charge to it. Native providers settle an
interrupted call themselves, as an unresolved bill that keeps its live
reservation and blocks resume.

## Per-call ceilings

Under `max_budget_usd`, calls to `OpenAIImageProvider`,
`OpenAIResponsesProvider`, `AnthropicMessagesProvider` and image-model
`GeminiProvider` cannot be reserved from a token estimate. Each needs an
inclusive `max_cost_per_call_usd` on the provider, or a node's own
`metadata["estimated_cost_usd"]`. A custom provider asks for the same with
`requires_explicit_budget_estimate()`.

An ordinary Swarm checks this before it pays for anything and raises
`BudgetEstimateRequired`. A generated plan cannot give a node its own
estimate, so when `execute(Task)` or `execute_async(Task)` has
`LLMArchitect` plan the run, the check runs before routing or planning, as
it does for an `LLM_MERGE` synthesis, whose ceiling only the provider can
supply. `plan()` and `aplan()` still return a generated plan: planning is
outside the ledger, and you can set `metadata["estimated_cost_usd"]` on each
node of the returned graph before calling `execute(graph)`. Every graph is
checked node by node before the first node call. See
[provider checks](execution.md#provider-checks-before-any-call).

## Held allowance for checks

Without a hold, every node competes for the same remaining allowance, so a
wide fan-out can spend what its verifiers or its `LLM_MERGE` synthesis need:
the run pays for the work and is then refused at the check or the merge.
`Swarm(reserve_for_checks=True)` sets that allowance aside. Under
`max_budget_usd`, before any node runs, the run holds what each pending
verifier and the merge will reserve: the node's `estimated_cost_usd`, else
the provider's estimate, else 2,000 tokens at `cost_per_token`. A budget
that cannot cover the holds stops the run with `SentinelAlert` before any
node call.

A hold counts toward every other node's admission. A node whose reservation
would reach into held allowance is refused with `SentinelAlert` before its
call, so a check or merge is never the step that finds its allowance spent.
Each verifier, and the merge, converts its own hold into its reservation
when it is admitted, so nothing is counted twice. A regeneration holds again
for each verifier it resets before the reset work runs; when the budget
cannot cover that hold, the run stops before the rerun.

Holds are never spend: `total_cost_usd`, `breakdown()`, run results and
checkpoints leave them out. A verifier or merge that fails releases its
reservation as before, and a run releases any hold still left after
synthesis or when it fails. Holds are not saved: `resume()` holds again from
its configuration, and checkpoints stay at version 4.

Holds narrow what other nodes may reserve, and a refused admission stops the
run; nothing is deferred. They bound reservations, not charges: a call that
costs more than its estimate can still reach into a hold, and so can a
serial `execute()` node without an estimate, which is only checked
(`Sentinel.check()` treats held allowance as spent). Set node estimates, or
run in parallel, where every node reserves before its call. A held verifier
without an estimate reserves its hold in serial runs too. A verifier that a
supervisor revision adds reserves when it is admitted, like any other node.
The option is off by default, and durable text workflows refuse it by name,
because their journal admits each call against the run's budget.

Code that drives a `Sentinel` directly can use the same allowance:
`hold(key, amount)` sets it aside, `release_hold(key)` frees it, `holds`
lists it, and `reserve(key, ..., from_hold=True)` converts a hold into the
key's reservation; a key with a hold must convert it.
`ExecutorBase.hold_verifier_budget(graph)` holds for a graph's pending
verifiers.

## Money units

Each ledger keeps money in one unit:

| Ledger | Unit |
|---|---|
| Ordinary runs: `Sentinel`, `CompletionResult.cost_usd`, `SwarmResult.total_cost_usd` | Float USD |
| [Durable text workflows](workflow-accounting.md#money-units) and native receipts | Integer nanoUSD |
| [Jobs](jobs.md) and the [Autotune ledger](optimize.md) | Integer micro-USD |

Conversions between units use integer arithmetic, so the caller's `decimal`
context cannot round them. Each has a fixed direction: a confirmed charge
converts exactly or is refused, a reservation or unknown exposure rounds up,
and a budget rounds down. A float is read as the decimal its `repr` shows, so
`0.3` is 300,000,000 nanoUSD. Jobs refuses a provider cost finer than one
micro-USD as invalid accounting instead of rounding it. Unknown cost has no
amount to convert and never becomes zero.

Durable results and Jobs snapshots name the unit of their integer amounts: a
durable result's `workflow_accounting["unit"]` is `"nanoUSD"`, and a Jobs
snapshot's `cost["unit"]` is `"microUSD"`. Stored rows and receipts carry no
unit field, so their bytes do not change.

## Current scope

The ordinary Swarm ledger covers execution and synthesis. Model-based routing,
planning, and successful supervision remain outside that ledger. The explicit
[OpenAI Responses provider](openai-responses.md) supplies native Astra/Sol token
prices and safe per-call receipts. Capped execution requires an inclusive
per-call ceiling, reserved again before each native tool turn. Known charges
from unusable responses are retained; unresolved billing keeps its reservation
and blocks ordinary resume.

Opt into [durable text workflows](workflow-accounting.md) with
`run_store=SQLiteWorkflowStore(...)` for one exact ledger across routing,
planning, execution, verification, supervision, and synthesis. That path uses
native request quotes, preserves unknown exposure across restarts, and replays
saved responses locally. Its returned accounting separates confirmed charges
from reserved and unknown exposure. Native prices are dated list prices.

Ordinary text calls without an explicit provider cost still use the configured
blended token estimate, `cost_per_token`.
`Sentinel.cost_contains_token_estimates` reports whether a total includes such
a call; `cost_contains_estimates` still covers only provider estimates and
conservative ceilings. `Sentinel.token_estimate_nodes` names those calls'
nodes, and ordinary checkpoints record them, so a resumed run keeps the flag
([checkpoint format](checkpoint-format.md#schema)). The [Astra cost campaign](../benchmarks/astra_benchmark_plan.md)
requires new records from the complete-workflow path.

`AnthropicProvider`, `OpenAIProvider` and `GeminiProvider` count a reply
without usage, or with an unset token count, as zero tokens, which the ledger
records as a known charge. Construct them with `missing_usage="unknown"` to mark
such a call's cost unknown instead (`cost_usd_unknown` on the result and the
node). The ledger then treats it like any unknown cost: a reservation held for
the call, as in parallel runs, is kept as its estimated charge
(`cost_contains_estimates=True`); without one, the run reports
`cost_is_complete=False`. A Gemini image's per-call ceiling or per-image cost
does not come from usage and is unaffected; `OpenAIImageProvider` has no such
option.

[Architecture](architecture.md) · [Checkpoint format](checkpoint-format.md) ·
[Offline budget example](../examples/03_parallel_budget.py).
