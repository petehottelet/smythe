# Astra benchmark plan

**Status: proposed; no paid Astra campaign has run.** This protocol defines a
new comparison. It does not update, reprice, or replace historical results.

## API and accounting prerequisites

Use the exact model ID `gpt-6-astra`. It supports text generation through
Chat Completions and Responses. Astra function calling requires Responses;
the current Smythe Chat Completions tool loop cannot run an Astra tool study.
For this campaign, use text-only requests with supplied source material.
Astra supports `low`, `medium`, `high`, `xhigh`, and `max` reasoning effort.
Remove unsupported sampling parameters, including `temperature`, `top_p`,
and `logprobs`. See the [model card](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [migration guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra).

The text-only README example passed an isolated published-package check on
**7 September 2026**, using `smythe==0.6.0`, Python 3.11.9, OpenAI SDK 3.8.0,
and HTTPX2 2.12.0. The exact example generated a mocked four-node fork/join plan
and completed all four nodes. Its five requests—one planning request and four
execution requests—passed through the real SDK's JSON serialization into an
in-memory HTTP transport. Every request contained exactly `model`, `messages`,
and `max_completion_tokens`, with model `gpt-6-astra` and the released default
cap of 4,096. No SDK warnings or live API calls occurred. The example uses
`Task.constraints`, which the published release supports.

The checked wheel's SHA-256 is
`c92bec189accdab68293b9e5be5e0385b58c93ff35ddcdc21e66fed22243bc99`;
the checked Python snippet's SHA-256 is
`b602a5fd1ea8206560486fe1f9758c1934db8226037e4d85e39240f4b988b6c6`.
This verifies the listed package/SDK combination, not account access, live
model behavior, or task quality. The 4,096-token cap covers visible output
and reasoning together; it does not establish that every task will fit.
`max_completion_tokens` is the current API parameter, while `max_tokens` is
deprecated. [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

Smythe's [execution ledger](../smythe/budget.py) and the existing
[benchmark usage wrapper](provider_usage.py) use a blended $3 per million
tokens when the provider supplies no explicit cost. Planning is separate from
the execution budget. Neither is a native Astra billing calculation. Omit a
numeric USD demonstration from the Astra Quickstart until native accounting
is implemented; do not present `max_budget_usd` as a ceiling on the full API bill.

Before a cost campaign, preserve endpoint-native usage for every request:
ordinary input, cache reads, cache writes, output, reasoning-token detail when
available, effective service tier, requested and returned model IDs, response
ID, and attempt status. Validate the endpoint's usage fields rather than
assuming that Chat Completions and Responses have identical schemas. Output
accounting must include billed reasoning tokens exactly once. Explicitly
configure reasoning effort and service tier in every adapter. Reconcile the
request ledger against provider usage; retain missing usage as unknown,
never zero. This work is a prerequisite, not part of the current README edit.

## Published prices and cost formula

The official Standard prices checked on **7 September 2026** are USD per
million tokens. Freeze a dated copy of the applicable schedule with the campaign.

| Model | Ordinary input | Cache read | Cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| `gpt-6-astra` | $10.00 | $1.00 | $12.50 | $50.00 |
| `gpt-5.6-sol` | $4.00 | $0.40 | $5.00 | $20.00 |

Requests exceeding 272,000 input tokens use 2× input and cache rates and 1.5×
output rates for the whole request. Service-tier and regional-processing
adjustments must be recorded when applicable. This campaign uses Standard
service, without Batch or Fast pricing. These are published list prices,
not an account-specific invoice. [Official pricing](https://developers.openai.com/api/docs/pricing)

For a short-context Standard Astra request, with total input `I`, cached-read
input `C`, cache-write input `W`, and billed output `O`:

```text
cost_usd = (10 × (I − C − W) + 1 × C + 12.5 × W + 50 × O) / 1,000,000
```

Validate `0 ≤ C + W ≤ I`. Cache writes are a distinct input category; do not
charge them again as ordinary input. Apply the corresponding long-context
rates when the request crosses the threshold. Record actual cache usage,
not an assumed discount. [Prompt-caching usage and billing](https://developers.openai.com/api/docs/guides/prompt-caching)

## Matched experiment

The primary design is **2 models × 2 execution strategies × 10 tasks × 5
repetitions = 200 complete workflow runs**:

| Factor | Fixed choices |
| --- | --- |
| Model | `gpt-6-astra`, `gpt-5.6-sol` |
| Strategy | Fixed research → analysis → writing pipeline; Smythe generated DAG |
| Task set | Two held-out tasks for each of the five [shape-suite categories](shape_suite.md) |
| Repetitions | Five per task, model, and strategy |

Use the same executor model for planning and execution within each arm.
Compare the two strategies within Astra to estimate its orchestration effect.
Compare models within the fixed pipeline to estimate the model effect.
Report the interaction: a faster model and a better topology are different
changes, and their benefits may not add together.

Run a separate 12-workflow pilot on three calibration tasks across the four
arms. Exclude it from confirmatory results. Freeze the held-out task pack,
source packs, prompts, rubrics, software versions, and policy before the main
campaign. Obtain a cost estimate and spending authorization before paid runs.

Use one API endpoint throughout the primary study, explicit `medium`
reasoning, a common output cap initially set to 8,192 tokens per call, and
identical supplied evidence and user-facing answer limits. Resolve any cap
changes during the pilot, then freeze them. Bound generated graphs to eight
execution nodes and concurrency to eight; count planning, synthesis, repair,
and retry work. Record every limit-triggered or truncated outcome. A graph
that cannot satisfy the frozen policy is a failed outcome, not an excluded
sample. No external search or tools are needed for the primary study.

Counterbalance model and strategy order within task/repetition blocks, with a
recorded random seed. Apply the same cache policy and record actual read/write
counts. Cache keys do not guarantee cold requests and caches cannot be
manually cleared, so label a first request as first observed, not proven cold.
Keep host load and request concurrency controlled. Record provider rate
limits, backoff, and SDK retries in both latency and spend.

## Outcomes and publication gates

Measure complete user-visible wall time, success rate, rubric quality, total
provider spend, and cost per accepted task. Total spend includes planning,
execution, synthesis, failed attempts, and repairs. Charge all attempted work
to the relevant arm. Report timeouts and unknown usage alongside successes.
Track judge spend separately from workflow spend, while disclosing the total
campaign expense.

Define acceptance with deterministic task checks and a frozen rubric. Blind
model and strategy labels for a fixed external judge; have a human review a
balanced sample and every disputed outcome. Keep complete outputs and judge
receipts. Set the quality noninferiority margin and minimum acceptable success
rate after the pilot but before held-out execution. Do not select them from
the final results.

Report all repetitions, medians, tails, paired differences, and task-clustered
uncertainty intervals. The task is the independent sampling unit; repeated
runs of one task do not create five new tasks. Report the ten-task scope
explicitly. A speed or cost headline requires the predeclared quality and
success gates to pass. Publish the full distribution, not only the fastest
run, and avoid cost-per-score ratios for an ordinal judge score.

Bind raw records to task/prompt hashes, graph artifacts, generator and harness
source hashes, dependencies, request parameters, price schedule, and provider
usage. A campaign becomes claimable only after a separate review confirms
complete cost scope, no known measurement defects, and preserved failures.
Missing billing evidence blocks cost headlines even when latency is usable.

## Separate follow-up studies

- **Scheduler effect:** execute the same frozen graphs and model at concurrency
  one and eight, preserving request content. Report graph planning separately
  and keep this diagnostic distinct from complete-workflow results.
- **Framework comparison:** match model, endpoint, effort, caps, tasks, and
  native usage accounting across Smythe, LangGraph, and CrewAI. If wire prompts
  differ, label it an idiomatic framework comparison, not an isolated scheduler
  test. The existing [framework harness](run_framework_h2h.py) needs these
  controls before use with Astra.
- **Tool workflows:** first implement and test an Astra Responses adapter,
  including tool-result continuation and retained reasoning state. Follow the
  [Astra tool restriction](https://developers.openai.com/api/docs/guides/reasoning#reasoning-effort)
  and the [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).
  Then freeze a separate tool task pack and count tool charges and failures.

Before any live run, offline contract tests should verify Astra provider
selection, the exact model ID and token-cap field, absence of unsupported
parameters, and complete usage-category reconciliation. Test ordinary,
cached-read, cache-write, long-context, missing-usage, retry, and partial-failure
receipts. A tool-enabled Astra request must fail clearly until a compatible
Responses path exists. No paid provider calls belong in the test suite.
