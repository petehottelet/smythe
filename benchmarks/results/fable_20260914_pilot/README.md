# Fable 5.1 pilot

**12 native workflows completed; 12 passed the output-contract checks.**
All native requests settled with complete billing. The separate Code Workflow
pilot also completed and passed. The blind Gemini judge assigned 4/4 on every
criterion for all 13 answers, with no material defects.

Human review of the six native medium-effort answers remains pending. The
100-workflow native main study has not started. These pilot results qualify
the experiment and are excluded from its main estimates; they do not establish
a general quality, cost, or speed advantage.

## Native Messages pilot

Model: `claude-fable-5-1`, Standard/global. Three synthetic tasks, both execution
strategies, and medium/high effort. Each request has an 8,192-token output cap;
the generated graph allows at most eight nodes. Planning and all execution
calls are included. There are no SDK, graph, or revision retries.

| Strategy | Effort | Workflows | Passed | Total API cost | Mean workflow time | API calls |
|---|---|---:|---:|---:|---:|---:|
| Fixed pipeline | Medium | 3 | 3 | $0.265410 | 26.224 s | 9 |
| Fixed pipeline | High | 3 | 3 | $0.308990 | 27.353 s | 9 |
| Generated graph | Medium | 3 | 3 | $0.401950 | 28.494 s | 9 |
| Generated graph | High | 3 | 3 | $0.428380 | 27.757 s | 9 |

**Native pilot total: $1.404730, 36 API calls.** Every request, response, quote,
graph checkpoint, native receipt, and outcome is retained in the archive.
The timing boundary includes trial setup, execution and ledger inspection;
it excludes evidence publication.

## Separate Code Workflow pilot

Claude Code 2.1.259 with Agent SDK 0.2.152 completed `pilot-membership` using
Fable 5.1 at medium effort. Code invoked Workflow twice: its first generated
workflow used an unrecognized model alias, then Code relaunched it. Both
launches remain in the transcript. The final answer passed all contract checks
and received 4/4 on every blind-judge criterion.

The [Code model-identifier amendment](../../fable_code_model_id_amendment_20260914.md)
records the prompt correction for the separate main comparison. The original
pilot and diagnostic main attempts remain unchanged.

The eight paid API calls cost **$0.5658735**. A local gateway counted each
request, reserved its full ceiling before forwarding, and saved native
response bytes. It pinned Fable and Standard service, prohibited hosted tools,
and enforced a $5 workflow allowance. The SDK's own stop threshold was $4.
No billing exposure remains unresolved. This experiment measures Code's
Workflow runtime separately from Smythe's generated graphs.

## Evaluation and spending

The judge is the same frozen `gemini-3.1-pro-preview` configuration used for
the Astra/Sol study. It sees tasks, sources, rubrics and answers without model
or strategy labels. Identical calendar answers share a single saved judgment:
13 scored answers required 10 paid judgments.

Judge cost is **$0.1151474–$0.1347620**. Missing native cache detail accounts for
that interval; its upper bound is held against the allocation. Combined native
pilot, Code pilot and judging cost is **$2.0857509–$2.1053655**. Main-study costs
are outside this pilot subtotal. The extension remains inside its $100 sublimit
and the original $300 campaign ceiling, including prior Astra/Sol spending.

## Evidence

- [Analysis and all trial summaries](analysis.json)
- [Complete pilot evidence archive](evidence.zip)
- [Archive file hashes](archive-manifest.json)
- [Protocol, allocation and main-study gates](../../fable_51_benchmark_plan.md)
- [Qualified source and CI](https://github.com/petehottelet/smythe/actions/runs/34813499079)

The archive includes the frozen runtime source, access receipt, qualified
dependency versions, native SQLite journal, raw request/response bytes,
Code orchestration transcripts and source, and blind-judge reasoning.
Publication was checked for credential values. A fresh extraction reproduced
every native outcome and reconciled all native and judge charges.

After extracting the archive, inspect it without provider calls:

```bash
python -m benchmarks.fable_runtime inspect /path/to/extracted/native
```
