# Fable 5.1 with Claude Code Workflow

**10 scheduled workflows completed; 10/10 accepted by contract checks and blind judging.**
The revised Code cohort cost **$3.77506525** across
70 paid Messages requests. All requests have complete native billing.

This is the separate Ultracode execution study: ten supplied synthetic tasks,
one run per task, Claude Code 2.1.259, Agent SDK 0.2.152 and
`claude-fable-5-1` at medium effort. Code's generated Workflow orchestration
is retained with every answer. These descriptive results do not establish an
advantage over Smythe; the matched 100-run native Fable study awaits human
pilot ratings. No Code trial enters that native estimate.

## Revised ten-task cohort

| Task | Workflow time | Native cost | API calls | Workflow launches | Lowest judge criterion | Accepted |
|---|---:|---:|---:|---:|---:|---|
| `main-identifiers` | 52.15 s | $0.31697675 | 9 | 1 | 4/4 | Pass |
| `main-studio-copy` | 64.15 s | $0.4096715 | 7 | 1 | 4/4 | Pass |
| `main-library-hours` | 115.62 s | $0.637352 | 7 | 1 | 4/4 | Pass |
| `main-calendar` | 43.78 s | $0.188047 | 5 | 1 | 4/4 | Pass |
| `main-tunnel-links` | 56.67 s | $0.333389 | 8 | 1 | 4/4 | Pass |
| `main-delivery-copy` | 63.93 s | $0.38743225 | 7 | 1 | 4/4 | Pass |
| `main-capacity-chain` | 55.82 s | $0.344799 | 7 | 1 | 4/4 | Pass |
| `main-production-chain` | 50.32 s | $0.277902 | 5 | 1 | 4/4 | Pass |
| `main-field-links` | 67.08 s | $0.32270775 | 8 | 1 | 4/4 | Pass |
| `main-warehouse-routing` | 111.21 s | $0.556788 | 7 | 1 | 4/4 | Pass |

Mean workflow time: **68.07 seconds**; median:
**60.30 seconds**. Timing covers the SDK query through its
final event, including Code orchestration, child agents and automatic recovery.
It excludes process setup, publication and subsequent judging. Native Smythe
trials have their own documented full-workflow timing boundary.

Every request in the revised cohort used the exact Fable model, medium effort and Standard
service. Code manages its own cache; retained receipts separate ordinary
input, cache reads, cache writes and output. This is not a cold-cache test.
Only the Workflow tool was enabled, with no shell, file, web or hosted tools.
The prompt requested at most eight agents and no nested subagents. That
request is not a measured hard concurrency limit for Code's runtime.

## Retained diagnostic cohort

The first five main tasks exposed shorthand model identifiers that Code could
not resolve. The [method amendment](../../fable_code_model_id_amendment_20260914.md)
specified the exact model ID, then reran all ten tasks with the other conditions
held fixed. No earlier favorable row was selected for the revised cohort.

Those **5 diagnostic workflows cost $2.5184095**,
made 42 paid calls and launched Workflow 10 times.
5/5 final diagnostic answers passed contract checks and blind judging.
Every attempt and charge remains in the separate diagnostic archive.
The original `main-identifiers` diagnostic also issued five low-effort requests;
its other five requests used medium. The revised cohort
used medium for all requests. Per-request effort counts remain in the records.

**All Code main attempts cost $6.29347475**
against the same $15 allocation, including the five diagnostics. The
separate Code pilot remains in the [pilot report](../fable_20260914_pilot/README.md).
Each workflow reserved $5 before starting. The gateway reserved each request's
ceiling before forwarding; the SDK had an additional $4 stop threshold.
No new billing is unresolved.

## Blind evaluation and total spending

The frozen Gemini judge saw the tasks, sources, rubrics and answers without
model, strategy or cohort labels. All fifteen main answers were scored;
identical answers reused saved judgments. This added 13 paid judgments,
costing **$0.1478518–$0.178522**. Missing native cache
detail creates those bounds; its upper amount remains charged to the allowance.
Criterion scores and judge reasoning are retained for each answer.

Including the native pilot, Code pilot, both Code main cohorts and all judging,
the Fable extension has used **$8.52707745–$8.57736225**
of its $100 sublimit. The combined Astra/Sol/Fable exposure is at most
**$28.06181185** of the existing $300 ceiling,
including the earlier Astra reserve. These totals include every paid attempt.

## Evidence and verification

- [All task records, scores and cost categories](analysis.json)
- [Revised ten-task evidence](revised-evidence.zip)
- [Original five-task diagnostic evidence](diagnostic-evidence.zip)
- [Judging, schedules, frozen source and offline audit](evaluation-evidence.zip)
- [Archive and per-file hashes](archive-manifest.json)
- [Full Fable protocol](../../fable_51_benchmark_plan.md)

Archives are separated by cohort so the revised results and diagnostics can
be inspected independently. They contain benchmark evidence and reproduction
source. Local credentials, Code account settings and internal plans are excluded.
A fresh extraction reproduced all fifteen outcomes, every streamed invoice,
SQLite charge, output hash and blind-judge binding.

From a Smythe checkout, extract all three archives into one directory and run:

```bash
python /path/to/extracted/code/audit_code.py /path/to/extracted/revised/main-identifiers
```

This audits saved bytes without provider calls. Repeat for each task directory.
