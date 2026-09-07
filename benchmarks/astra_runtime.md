# Astra pilot execution

The [pilot runner](astra_runtime.py) executes the 12 scheduled Astra/Sol
workflows through native Responses and the durable text ledger. It binds
source files, dependencies, prices, prompts, graph limits, spending allocations,
and one campaign directory before execution.

**No paid campaign has run.** The runner has offline transport tests; live
execution awaits an authorized spending ceiling. The 200-workflow main study
and judge commands remain closed pending their calibration and accounting
requirements. The [experiment protocol](astra_benchmark_plan.md) defines those
publication gates.

## Inspect without spending

From the repository root, write an unfunded freeze to a new file:

```bash
python -m benchmarks.astra_runtime freeze --out astra-unfunded.json
```

This inspects local source and installed dependencies. It makes no provider
calls, records the missing spending allocation, and cannot authorize a pilot.
The [original preparation receipt](results/astra_preparation_20260907.json)
remains unchanged.

## Allocate and approve a pilot

Install the checkout with `pip install -e ".[openai]"`. Before freezing a live
run, agree on the total API allowance and separate pilot, main, judge, and
per-trial amounts. All five fields use strict integer nanoUSD: one dollar is
1,000,000,000 nanoUSD. Pilot + main + judge must fit within the total; twelve
per-trial allowances must fit within the pilot allocation. Unused trial amounts
are not reassigned to other trials.

Pass those agreed amounts to `freeze` using `--total-nanousd`,
`--pilot-nanousd`, `--main-nanousd`, `--judge-nanousd`, and
`--per-trial-nanousd`, plus `--directory` and a new `--out` file. The directory
must be dedicated to that campaign. Review the resulting policy, source
hashes, dependency versions, allocations, and destination before using its
`approval_token` with:

```bash
python -m benchmarks.astra_runtime pilot --freeze APPROVED_FREEZE.json --directory CAMPAIGN_DIRECTORY --approve APPROVAL_TOKEN
```

Set `OPENAI_API_KEY` only for live execution. The token records acceptance of
the exact freeze; it is not a secret credential. Changing code, dependencies,
policy, allocations, or the destination invalidates that freeze. Main and
judge allocations reserve room in the campaign envelope; they do not enable
those stages.

## Execution and recovery

Each arm uses its exact model, Responses, Standard/global pricing, medium
reasoning, an 8,192-token output cap, and zero SDK retries. One workflow runs at
a time. Generated graphs use at most eight execution nodes with concurrency
eight, zero node retries, zero regeneration, and zero revisions. The fixed arm
uses three local stages: research, analysis, and writing. Both receive the same
task sources and output constraints; evaluator rubrics stay out of requests.

The campaign writer lock prevents concurrent pilot workers. Each scheduled
trial receives a deterministic workflow identity before planning. Repeating
the same command preserves completed and failed outcomes. An interruption
before outcome publication resumes the existing workflow and marks its total
latency incomplete. Saved responses replay locally; unknown billing blocks
admission to later trials.

The output directory retains the SQLite ledger, immutable start and outcome
records, failure diagnostics, and a final pilot summary. Outcome validation
checks current accounting, requests, raw-response hashes, decoded results,
checkpoint output, and deterministic checks against the saved receipt.
Keep the directory and its SQLite sidecars together.

## What the pilot can establish

The pilot preserves full workflow charges, failures, output limits, and timing
segments for calibration. Its factual checks do not constitute quality scores
or accepted-task judgments: `quality_evaluated` is false, `accepted` is null,
and every result remains nonclaimable. Main execution requires a reviewed paid
pilot and frozen human-calibrated acceptance gates. Judging requires a frozen
judge identity and its own durable accounting implementation.

[Provider guide](../docs/openai-responses.md) ·
[Workflow accounting](../docs/workflow-accounting.md) ·
[Offline regression tests](../tests/test_astra_runtime.py).
