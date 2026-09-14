# Approved continuation after a connection failure

**Status: all 53 continuation workflows completed and passed their deterministic checks.**
The user approved the change with “Approve reserved-cost continuation.”
The envelope passed 13 offline regression checks before its separate source
freeze. Both segments now contain exactly 200 outcomes, including the
original failed workflow; its $0.169645 reservation remains unresolved.

The amended Astra/Sol study stopped after 147 of its 200 scheduled workflows.
The first 146 passed their deterministic checks. Workflow 147 failed when a
Sol execution call returned `APIConnectionError` without an HTTP response,
request ID, response body or usage receipt. Its confirmed earlier calls cost
$0.0318752. The unresolved call retains its entire **$0.169645 reservation**.

The [frozen execution protocol](astra_runtime.md#execution-and-recovery) says
“unknown billing blocks admission to later trials.” The existing runner
enforced that rule. The failed workflow and its native ledger remain intact.

## Approved change

1. Keep all 147 outcomes, including the failed workflow. Do not retry its call
   or replace its result with a successful repetition.
2. Hold $0.169645 against the original $200 main allocation until authoritative
   usage evidence resolves the charge. Keep the $300 total, $60 pilot, $40
   judge and $5 per-workflow caps unchanged.
3. Freeze a separate continuation for only the remaining 53 schedule entries.
   Preserve their original order, task inputs, models, graph policy, reasoning
   effort, output limits and zero-retry settings.
4. Admit an independent continuation workflow only when confirmed charges,
   the entire held exposure, and its full $5 reservation fit the original
   stage allowance. A new unresolved call stops the continuation again.
5. Preserve both ledgers and their source snapshots. Audit the exact union of
   147 earlier outcomes and 53 continuation outcomes; reject duplicates,
   omissions, changed inputs or unbound charges.
6. Count the failed workflow in the 200-run acceptance denominator, with zero
   quality for its missing deliverable. Score every available answer blindly.
7. Report affected costs as bounds, retaining the unknown charge explicitly.
   Do not publish an exact-cost headline until its native usage is reconciled.
   Label the operational interruption and this protocol amendment in the report.

This approval permits new, independent scheduled workflows under a reserved
campaign balance. It does not unlock or alter the failed workflow, change
previous scores, or silently relax the existing runner.

The prelaunch checks cover preserved failures, schedule identity, retained
unknown exposure, insufficient budget, duplicate/replay prevention and
stopping on another unresolved call. The continuation added $3.8220330 in
confirmed generation charges. Main-study charges are bounded at
$12.6076932–$12.7773382, excluding separately reported judging charges.
