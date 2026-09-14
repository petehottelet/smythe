# Astra campaign execution

The [pilot runner](astra_runtime.py) executes the 12 scheduled Astra/Sol
workflows through native Responses and the durable text ledger. It binds
source files, dependencies, prices, prompts, graph limits, spending allocations,
and one campaign directory before execution.

**Human calibration and the final 12-workflow pilot passed.**
The [original 24-workflow snapshot](results/astra_20260913/README.md) and
[method amendment](astra_method_amendment_20260913.md) retain the earlier
contracts, diagnostics and charges. Four pilots total 48 workflows and
$2.6596199; the stopped 31-workflow main diagnostic cost $1.7646655.
Native Gemini judging scored the original outputs and detected planted errors.
The approved ceiling is $300: $60 pilot, $200 main, $40 judging and $5 per
workflow. Every pilot draws from the same $60 allocation.

The [follow-up runner](astra_study.py) implements the corrected pilot and
200-workflow main schedule. The six submitted human ratings passed, and the
amended main study has [200 recorded and automatically scored outcomes](results/astra_20260913_main/README.md).
Its [approved 53-workflow continuation](astra_connection_continuation_20260913.md)
preserves one failed outcome and its unresolved reservation.

Human review accepted all eight flagged main answers at 4/4. The study is
claimable within its documented descriptive scope; the frozen automatic
classifications and affected cost bounds are retained.
The [experiment protocol](astra_benchmark_plan.md) defines the publication gates.
The original [79 offline checks](results/astra_pilot_runtime_20260907/README.md)
remain historical qualification, separate from these live results.

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
judge allocations reserve room in the campaign envelope; their separate
runners require the evidence and explicit inputs described below.

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
and every workflow receipt remains nonclaimable. Judgments and human ratings
are separate evidence; they never rewrite the original workflow result.

## Explicit-format follow-up and main study

The follow-up states existing nonempty-string and numeric requirements in
provider inputs, plus both allowed arithmetic decision labels. Graph-only
requirements use `LLMArchitect.planning_instructions`; all executors receive
the same clean task. It supplies no expected factual answer values and
preserves the original pack.
Freeze it in a new directory; the runner subtracts original pilot charges from
the authorized pilot allowance before admitting more work:

```bash
python -m benchmarks.astra_study freeze format-pilot --original-pilot ORIGINAL_PILOT --directory FORMAT_PILOT --output format-freeze.json
python -m benchmarks.astra_study run --freeze format-freeze.json --approved-freeze-sha256 EXACT_PRINTED_SHA256
python -m benchmarks.astra_study inspect FORMAT_PILOT
```

`freeze` and `inspect` make no provider calls. `run` spends against the saved
stage and per-workflow limits. Use the returned hash only after reviewing the
freeze under an approved campaign allowance. Completed and failed outcomes
are retained on replay; unresolved billing blocks new dispatch.

For the main freeze, supply the complete corrected pilot, a six-sample human
rating receipt bound to its anonymous sample manifest, and reconciled judge
calibration. Four sampled outputs must balance all four arms; the sample also
contains a planted-error control. A disagreement requires further review.

```bash
python -m benchmarks.astra_study freeze main --original-pilot ORIGINAL_PILOT --format-pilot FORMAT_PILOT --human-response HUMAN_RATINGS.json --human-manifest HUMAN_SAMPLES.json --judge-evidence JUDGE_CALIBRATION.json --directory MAIN_STUDY --output main-freeze.json
python -m benchmarks.astra_study run --freeze main-freeze.json --approved-freeze-sha256 EXACT_PRINTED_SHA256
```

The main runner validates native pilot and judge evidence before freezing its
200-row schedule. It binds source, dependencies, inputs, rubrics, acceptance
gates and calibration hashes. Old freezes require their archived source;
changing the current harness does not modify a completed pilot's records.
The legacy `astra_runtime` main/judge stubs remain closed; use the follow-up
modules for these stages.

For an amendment, pass each earlier directory of the same stage through
`--previous-stage`. The freeze deducts every recorded charge from that stage's
original allocation and binds each prior summary. Include earlier ancestors
as well as the most recent attempt; omission or changed evidence blocks
dispatch. Earlier records retain their original scores and source snapshots.
The current amended study reuses diagnosed tasks and makes no untouched-holdout claim.

## Reserved-cost continuation

The original runner stops new trials when billing is unknown. The September
study used a separately approved envelope after a no-response connection failure.
[`astra_continuation`](astra_continuation.py) binds the untouched earlier
outcomes, full unknown reserve, exact remaining schedule and its own source.
It requires approval of that freeze, never reruns an earlier trial, and stops
on any new unresolved call. The original caps and execution policy remain in force.

[`astra_combined`](astra_combined.py) audits both local ledgers and the exact
200-outcome union. Evidence directories can be relocated; stored source paths
are provenance. Unknown charges remain ranges, and affected exact cost
contrasts are withheld. Missing deliverables remain failed with zero quality.

For this campaign, add `--continuation-directory CONTINUATION` to the judging
command below. Add that option and `--continuation-source-archive CONTINUATION_SOURCE.zip`
to the evidence review command. The [completed report](results/astra_20260913_main/README.md#evidence-and-reproduction)
provides the complete archive and reproduction command.

## Blind judging and analysis

[`astra_evaluation.run_judgment`](astra_evaluation.py) evaluates a saved output
against its supplied sources and anchored rubric. Supply an explicit judge
directory, allowance in nanoUSD, API key and returned model version. The
function freezes Gemini's model, prompt, policy and price card. It counts and
reserves a request before generation, retains raw bytes before decoding,
and reuses an identical saved judgment without another paid call. Unknown
attempts stop admission. No credential is written to its records.

Native judge totals include reasoning once. When Gemini omits cached-input
detail, accounting retains a price interval and uses its upper bound for the
allowance. A missing aggregate usage receipt cannot become zero cost.
[`astra_evidence`](astra_evidence.py) reconciles every judge request, quote,
raw response, score, charge and pilot-output binding offline.

The complete main schedule has an explicit runner. Inspection makes no paid
calls; add `--live` and set `GOOGLE_API_KEY` to judge the saved outputs under
the original shared judge allowance:

```bash
python -m benchmarks.astra_judge_study --main-directory MAIN_STUDY --judge-directory SHARED_JUDGE_DIRECTORY --bindings-path main-judgments.json
```

Identical task/output judgments replay their saved receipts. A changed binding
or missing output directory stops before paid work. The judge sees task,
source, rubric and answer, without arm labels, cost or latency.

[`astra_analysis.analyze_main`](astra_analysis.py) requires all 200 scheduled
outcomes and their bound judgments. It reports every trial, arm success,
median and P95 wall time, total workflow cost and cost per accepted task.
Within-model differences, model contrasts and their interaction resample ten
whole tasks, preserving all five repetitions. Success and quality gates were
fixed before the first main execution and remain unchanged.
The analysis remains nonclaimable until a separate evidence review, including
human review of disputed outputs, clears publication.

Reconcile the completed study, every frozen task, all phase charges and its
archived source without provider calls:

```bash
python -m benchmarks.astra_main_evidence --main-directory MAIN_STUDY --judge-directory SHARED_JUDGE_DIRECTORY --bindings-path main-judgments.json --source-archive MAIN_SOURCE.zip --out REVIEW_DIRECTORY
```

This produces a complete analysis and a separate audit record. It never
approves a performance claim automatically.

The [published main study](results/astra_20260913_main/README.md#human-review)
now includes all eight submitted human ratings, bound to the archived output
hashes. Its publication review approves the stated descriptive scope; the
191/200 automatic classifications and affected cost bounds remain unchanged.
Reproducing the native audit above does not replace that separate human review.

[Provider guide](../docs/openai-responses.md) ·
[Workflow accounting](../docs/workflow-accounting.md) ·
[Offline regression tests](../tests/test_astra_runtime.py).
