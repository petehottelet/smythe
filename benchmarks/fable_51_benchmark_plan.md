# Claude Fable 5.1 benchmark extension

**Status: preparation only; no Fable provider calls or results.**

Compare fixed and generated execution graphs with `claude-fable-5-1` through
the native Claude Messages API. This tests whether the Astra/Sol observations
extend to another provider. The [Astra/Sol records](results/astra_20260913_main/README.md)
remain a separate, completed and human-reviewed cohort. Do not append Fable
trials to that frozen 200-run campaign.

The [preparation record](fable_51_preparation_20260913.json) binds the proposed
112 trial positions, task inputs, rubrics, prior spending, and policy to local
source hashes. It is not an executable runner, paid-runtime freeze, or result.

## Experiment

| Stage | Design | Workflows | Use |
|---|---|---:|---|
| Pilot | Three existing pilot tasks × fixed/generated × medium/high effort | 12 | Verify request compatibility, accounting, output contracts, and blind judging; keep both effort settings visible |
| Main | Ten existing main tasks × fixed/generated × five repetitions, medium effort | 100 | Primary within-Fable comparison |

Every main task/repetition block runs both strategies. Order the 50 blocks
with seed 14173 and alternate strategy order: fixed first in 25 blocks,
generated first in 25. Each strategy has 50 outcomes. Retain every scheduled
attempt, including refusals, timeouts, truncated outputs, and failed graphs.
Pilot outcomes never enter main estimates.

Use the amended answer contracts in `astra_study.study_task`, the same source
packs, factual checks, rubric anchors, fixed three-stage pipeline, and
planner-only graph instructions. No reference answers enter provider inputs.
These are ten reused synthetic tasks, not an external or untouched holdout.

The primary effort setting is fixed at `medium` before seeing Fable outputs.
Matching that parameter name does not equate the providers' reasoning budgets
or tokenizers. Anthropic recommends starting Fable at `high`; the six high-effort
pilot runs are diagnostic and cannot establish its best quality or a reliable
effort comparison. A full high-effort main study needs its own frozen schedule.
[Effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort).

## Request and execution policy

- Native Claude Messages API; exact requested and returned model identities.
- Explicit `output_config.effort`; adaptive thinking remains enabled.
- `max_tokens: 8192` for the complete output, including thinking. Record
  truncation; do not silently raise the cap or replace unsuccessful trials.
- Standard/global service. No Batch, Fast mode, Priority, provider switching,
  server-side fallback, tools, search, attachments, or beta features.
- No temperature, top-p, forced tool calls, or assistant prefill. Use the same
  bare-JSON prompt contract as the earlier study; add no schema-enforcement
  advantage to one provider.
- No explicit cache breakpoints in this extension. Record actual cache usage;
  do not describe requests as cold-cache or claim an unmeasured cache benefit.
- Same Fable model and effort for planning and all execution nodes. Fixed
  research → analysis → writing topology; generated topology at most eight
  execution nodes. Concurrency eight within a graph; one workflow at a time.
- Zero SDK retries, node retries, regenerations, planning repairs, or revisions.
  Preserve the 600-second request timeout and full-workflow timing boundary.

Fable's always-on thinking and supported request controls are documented in
the [migration guide](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide).
Account compatibility must be checked before the paid pilot; a model listing
alone does not establish that generation will succeed.

## Native billing

Rates checked on 13 September 2026, in USD per million tokens:

| Category | Rate |
|---|---:|
| Ordinary input | $10.00 |
| Cache read | $0.25 |
| Five-minute cache write | $12.50 |
| One-hour cache write | $20.00 |
| Output, including billed thinking | $50.00 |

These are published list prices, not an invoice or predicted workflow cost.
Freeze and recheck the applicable prices before dispatch.
[Official model and prices](https://platform.claude.com/docs/en/models/fable-5-1/overview).

Claude's `input_tokens` excludes cached reads and writes. Total input is the
sum of ordinary input, cache reads, and cache writes; do not subtract cached
tokens from ordinary input as the OpenAI adapter does. Validate the five-minute
and one-hour write breakdown against total cache creation. Count billed
thinking inside output once. Missing or inconsistent usage remains unknown.
[Native cache fields](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

```text
cost_nanousd = 10000 × ordinary_input
             + 250 × cache_read
             + 12500 × cache_write_5m
             + 20000 × cache_write_1h
             + 50000 × billed_output
```

Preserve request/response bytes before parsing, request and response IDs,
requested/returned model and service metadata, each usage category, stop
reason, price version, and attempt status. Include planning and every failed
attempt. Store blind judging costs separately and in the campaign total.
Never publish credentials or authorization headers.

## Spending envelope

Keep the existing overall **$300 ceiling**, including all earlier Astra/Sol
work and held unknown charges. Its stage ceilings remain **$60 pilot, $200
main, and $40 judging**, with **$5 per workflow**. This extension uses a tighter
**$100 sublimit: $15 pilot, $75 main, and $10 judging**; it does not add $100
to the approved overall ceiling or move money between the existing stages.

The [prior spending record](results/astra_20260913_main/campaign-spending.json)
holds $19.4844496 at its upper bound, including the failed call's $0.169645
reserve and judge uncertainty. On that snapshot, spending the full extension
sublimit would place total exposure at $119.4844496. These are ceilings and
retained usage-derived bounds, not a prediction of Fable's bill.

Before each workflow, reserve up to $5 against its extension stage, original
stage, extension total, and combined campaign total. Admit only when every
balance covers that reservation; settle against actual native receipts.
Refresh prior spending before launch and prevent concurrent campaigns from
spending the same allowance. Do not promise all 112 runs if a ceiling is hit.
Stop new dispatch on any new unknown charge. The earlier reserved-cost
continuation authorized one specific Astra/Sol incident, not future incidents.

## Implementation and launch gates

1. **Credential and access.** Configure `ANTHROPIC_API_KEY` locally. Retrieve
   the exact model's metadata and retain a sanitized access receipt. This
   checkout currently has no Anthropic key in its environment or `.env`.
   Never change account retention settings automatically.
   [Model metadata endpoint](https://platform.claude.com/docs/en/api/models/retrieve).
2. **Native adapter.** Add a Messages adapter with explicit effort, zero
   retries, exact request counting/quotes, integer pricing, raw evidence,
   refusal/truncation handling, cancellation accounting, and offline replay.
   The existing `AnthropicProvider` reads ordinary input/output only and uses
   the generic cost fallback; it is unsuitable for this cost comparison.
3. **Durable integration.** Extend the provider descriptor, journal settlement,
   replay decoder, and workflow factory. `workflow_provider.py` currently
   accepts native OpenAI Responses and stateless offline providers only.
   Adding a model string to the Astra runner cannot satisfy this gate.
4. **Offline qualification.** Cover cache categories, malformed/missing usage,
   unexpected model/tier, output caps, refusals, known-charge parse failures,
   disconnects, cancellation, concurrency reservations, replay without calls,
   duplicate dispatch, and exhausted/shared budgets. Run modified test files,
   Ruff, and the full offline suite. Keep paid requests out of tests.
5. **Separate runtime freeze.** Bind the new implementation and dependency
   versions, task contracts, schedule, price card, judge identity, source
   archive, and refreshed shared spending. Do not alter old freezes or use
   the preparation JSON as a runtime approval token.
6. **Pilot review.** Run and retain all 12 pilot outcomes. Inspect both effort
   settings for contract failures, material defects and truncation. Require
   complete billing and all six medium-effort pilot outputs to pass before
   main dispatch. Blindly review at least one medium-effort answer from each
   strategy and every flagged pilot answer. Preserve existing calibration;
   obtain actual human ratings for the new samples.
7. **Main and publication.** Run the 100 medium-effort positions once, then
   reconcile every outcome and judge receipt. Inspect all disputed answers,
   reproduce analysis from a fresh evidence extraction, and publish only
   the claims the review supports.

If a pilot exposes a measurement defect, retain its evidence and costs and
write a separate amendment before further trials. Do not tune the primary
effort, prompts, rubric or token cap to obtain a better reported result.

## Evaluation and charts

Reuse the frozen Gemini judge prompt and anchored 0–4 criteria. Recheck its
returned identity and price before spending; a changed judge needs separate
calibration. Blind model, effort, and execution strategy. Acceptance requires
all deterministic checks, every criterion at least three, and no material
defect. Keep automatic and human judgments separate. Preserve the existing
90% arm-success gate and 0.25-point quality noninferiority margin.

Report completion and acceptance, all quality scores, total and phase costs,
cost per accepted output, mean/median/P95 latency, call and node counts,
graph depth and width, and peak concurrent calls. Undefined cost per accepted
output stays undefined when none are accepted. Retain unknown charges as
bounds and withhold affected exact comparisons.

The primary comparison is generated minus fixed **within Fable**. Use whole
tasks as the sampling units: ten task clusters, five repetitions, 10,000
bootstrap draws, seed 14173. Treat additional endpoints as descriptive;
do not turn an isolated favorable interval into a superiority claim.
Side-by-side Astra/Sol/Fable tables must label collection cohorts: these
models were not jointly randomized or run simultaneously. A direct model
ranking requires a new interleaved three-model campaign.

After review, render monochrome charts from committed records through
`python benchmarks/render_readme_charts.py`: all workflow timings and costs,
within-model paired differences, acceptance counts, and planning/execution
cost breakdowns. Show the high-effort pilot separately with its small sample
size. No results chart is generated from this preparation record.

## Scope of “all tests”

This extension covers the task-shape text campaign. The 192/256 SVG workflow
compiles authored geometry locally; the renderer measures browser graphics;
the offline suite checks software behavior. Replacing the model cannot
change those recorded workloads. Live Fable glyph generation, modern
framework comparisons, scheduler-only experiments, paid recovery, and
external-task quality studies each need a separate workload and protocol.
