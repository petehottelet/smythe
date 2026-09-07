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
blended token estimate. The [Astra cost campaign](../benchmarks/astra_benchmark_plan.md)
requires new records from the complete-workflow path.

[Architecture](architecture.md) · [Checkpoint format](checkpoint-format.md) ·
[Offline budget example](../examples/03_parallel_budget.py).
