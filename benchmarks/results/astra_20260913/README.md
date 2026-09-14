# Astra and Sol: live pilot results

**Historical snapshot:** this report records the first two pilots before
human calibration. The six ratings have since passed. The
[study amendment](../../astra_method_amendment_20260913.md) records later
diagnostics and the corrected main-study method. The counts and charges below
remain the original 24-workflow snapshot.

**24 live workflows completed.** The corrected 12-workflow pilot passed every
format and factual check. The original pilot retained one prompt/type mismatch.
The 200-workflow main comparison awaits six human calibration ratings.

These are calibration results, **not a claimable model or orchestration comparison**.
The two pilots use three synthetic tasks, one repetition per model/strategy arm.
The ten held-out main-study tasks have not been executed.

## Completed work

| Stage | Workflows | Execution failures | Format/factual passes | Workflow charges |
|---|---:|---:|---:|---:|
| Original pilot | 12 | 0 | 11/12 | $0.7805955 |
| Explicit-format follow-up | 12 | 0 | 12/12 | $0.6789914 |

**Workflow total: $1.4595869 across 86 generation calls**, including paid planning.
Token-count and model-access requests are separate from that generation count.
Gemini evaluated all 12 original answers and one deliberately flawed control.
Identical answers reused one saved judgment: 10 paid judge calls total.
Judge charges are **$0.1054428–$0.1249800**; total campaign charges are
**$1.5650297–$1.5845669** at the frozen list prices.

The approved ceiling is $300: $60 pilot, $200 main, $40 judging, with $5 per
workflow. Both pilots use the $60 allocation. Stage allowances are not transferred.

## What the pilot found

The original relay task asked for reasoning without requiring its JSON type.
Astra’s fixed pipeline returned a detailed object; the checker required a string.
The original failed check remains in the evidence. The follow-up adds explicit
string requirements to provider inputs wherever the existing checks require text.
It supplies no expected factual answers and leaves the original task pack intact.

The judge assigned 4/4 to every criterion on all 12 original outputs. It detected
false savings, guest-access and support claims in the deliberately flawed control,
with scores from 0 to 2. This establishes detection of those planted errors;
sensitivity to subtle quality differences remains unmeasured. Corrected pilot
outputs have deterministic validation only. Human calibration is not complete.

## Protocol and accounting

- Exact models: `gpt-6-astra` and `gpt-5.6-sol`, native Responses, Standard/global,
  medium reasoning, 8,192 output tokens, zero SDK retries, no tools or search.
- Fixed strategy: three local stages, research → analysis → writing.
  Generated strategy: paid planning plus at most eight execution nodes,
  concurrency eight, no retries, regeneration, or revisions.
- One workflow at a time on a Windows development host. Concurrent development
  and offline checks were not excluded from pilot timing; no speed claim uses it.
- Judge: `gemini-3.1-pro-preview`, returned model identity retained, medium
  thinking, Standard tier, no tools, anonymous source/task/output/rubric inputs.
  Model metadata reported version `3.1-pro-preview-01-2026`; generation returned
  the model alias. Both identities are retained; the provider controls alias updates.
  Native totals include thinking exactly once. Gemini omitted cached-input
  counters; the report gives a cost interval and budgets at its upper bound.
- OpenAI SDK 3.8.0 and HTTPX2 2.12.0 in an isolated environment. Frozen source,
  dependency versions, request counts, responses, graphs, usage and failures
  are retained with both pilots.

OpenAI list prices were rechecked on 13 September and match the
[frozen pricing table](../../astra_benchmark_plan.md#published-prices-and-cost-formula).
Judge pricing is frozen from [Google’s published schedule](https://ai.google.dev/gemini-api/docs/pricing).

## Every trial

All times are complete observed workflow seconds. All costs are USD.
Each cell contains **seconds / cost**; this table reports observations only.

| Task | Arm | Original pilot | Explicit-format pilot |
|---|---|---:|---:|
| relays | astra-fixed | 32.795 / $0.1075750 | 31.797 / $0.0906200 |
| relays | sol-fixed | 16.122 / $0.0270080 | 15.770 / $0.0267360 |
| relays | sol-dynamic | 30.521 / $0.0648550 | 30.657 / $0.0862200 |
| relays | astra-dynamic | 37.566 / $0.1547550 | 22.889 / $0.0674400 |
| membership | sol-fixed | 21.716 / $0.0241560 | 21.699 / $0.0283880 |
| membership | astra-dynamic | 52.664 / $0.2087025 | 49.096 / $0.1838750 |
| membership | astra-fixed | 25.741 / $0.0737500 | 27.061 / $0.0739800 |
| membership | sol-dynamic | 31.258 / $0.0578850 | 32.707 / $0.0631030 |
| calendar | astra-dynamic | 10.884 / $0.0256750 | 15.135 / $0.0249710 |
| calendar | sol-dynamic | 10.120 / $0.0085220 | 11.632 / $0.0069564 |
| calendar | sol-fixed | 8.038 / $0.0072320 | 17.061 / $0.0077720 |
| calendar | astra-fixed | 17.436 / $0.0204800 | 12.956 / $0.0189300 |

## Main-study gates

Before the main run: save six actual human ratings, review any disagreement,
and freeze the main schedule, source and calibration evidence. Execution
requires its frozen evidence. An accepted output passes every deterministic
check and every rubric criterion ≥3/4, with no material defect. The minimum success rate is 90% per arm.
The quality noninferiority margin is 0.25 on the mean anchored 0–4 criterion scale;
paired uncertainty resamples the ten whole tasks with seed 14173 and 10,000 draws.
These settings are declared before any held-out run.

The [follow-up runner](../../astra_study.py) implements the bounded main schedule.
The [analysis](../../astra_analysis.py) retains failures and all repetitions,
reports medians and tails, and keeps task-clustered intervals separate from
descriptive scores. It cannot mark a campaign claimable. Main judging, human
review of disputed outcomes, and a separate evidence review remain required.

## Verification

The full offline suite passed **3,861 tests with 6 skipped**
across 128 files; Ruff passed. The new Astra implementation has 60 focused
offline checks. [Qualification and source hashes](qualification.json) ·
[Complete test logs and JUnit records](offline-verification.zip).

## Evidence

- [Machine-readable summary and all 24 trial metrics](summary.json)
- [Complete pilot evidence archive](pilot-evidence.zip): SQLite native ledgers,
  immutable outcomes, both source snapshots, judge requests and raw responses.
- [Archive member hashes](archive-manifest.json) and [relocated evidence review](evidence-review.json)
- [Experiment protocol](../../astra_benchmark_plan.md) and [runner commands](../../astra_runtime.md)

Archive SHA-256: `6fee1bc87bf369c2de413661d89eaafaf04ed33737148313320ef04e9c496639`.

Extract to a new directory. From the repository root, reconcile the corrected
pilot without provider calls:

```bash
python -m benchmarks.astra_study inspect EXTRACTED_DIRECTORY/format-pilot
```

Frozen source snapshots describe the implementations used for each pilot;
later harness changes do not rewrite their records. No paid main result or
new performance headline is represented by this report.
