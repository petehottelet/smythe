# GPT Image parallel benchmark plan

Status: Phase 1 in progress (2026-07-11)

## Objective

Extend Smythe's parallel artifact path to GPT Image, establish a small honest live
benchmark, and then scale the same request contract to creative suites and large offline
batches. Credentials remain local and are never written to prompts, logs, checkpoints,
manifests, or artifacts.

## Phase 1 — guarded live proof

- Add a dedicated `OpenAIImageProvider` using `images.generate`; keep text completions
  and the existing Gemini image path isolated.
- Add mocked unit coverage plus a deterministic offline example.
- Preflight three independent Golden Gate Bridge prompts with the maintained image CLI.
- Live configuration: `gpt-image-2-2026-04-21`, `1536x1024`, medium quality, JPEG,
  `n=1`, concurrency 3, no automatic retries.
- Expected image-output cost: about $0.123 total, plus small text-input cost. Abort before
  dispatch if the fixed request count or settings differ.
- Record prompts, model snapshot, settings, per-job elapsed time, batch wall time,
  dimensions, MIME type, byte count, SHA-256, usage when exposed, and output paths.
- Report `sum(per-job latency) / batch wall time` as an observed concurrency factor, not
  as a controlled serial speedup.

Phase-1 gates: no secret output; model access confirmed; exactly three requests; all
outputs decode; all dimensions and formats match; total expected cost remains below the
declared cap; failures are retained rather than hidden.

## Phase 2 — reproducible request contract

Introduce a versioned per-image request spec with stable IDs, prompt hashes, immutable
request JSONL, conservative maximum-cost reservation, provider request IDs, usage-based
actual cost, atomic artifact writes, hashes, bounded retry classification, and offline
fixtures. Add append-only event/results JSONL and a compact run manifest containing only
environment-variable names—not values.

## Phase 3 — high-throughput execution

- Interactive lane: bounded queue/workers, adaptive IPM/TPM limiting, `Retry-After`,
  full-jitter backoff, circuit breaking, resumability, and dead-letter records.
- Bulk lane: OpenAI Batch API shards for offline work. Batch currently supports image
  generation, offers a separate higher-limit pool and a 50% discount, and completes
  within a stated 24-hour window. Use 500–1,000-item recovery shards rather than one
  monolith for a 5,000-image run.
- Replace per-node full-graph checkpoint rewrites with append-only per-item journaling
  plus periodic compact snapshots.

## Phase 4 — production creative suites

Generate a small approved set of master plates for landscape, square, and portrait
families; use reference-image edits for continuity; then deterministically compose exact
copy, logo, CTA, typography, safe areas, and legal text before exporting IAB sizes. Many
banner sizes cannot be generated directly because GPT Image 2 requires at least 655,360
pixels, dimensions divisible by 16, and aspect ratios no wider than 3:1.

## Phase 5 — evaluation and publication

Publish raw records and failures. Measure success/block/error rates, retries, p50/p95/p99
latency, throughput, observed concurrency factor, cost per image and accepted image,
decode/dimension validity, duplicates, prompt adherence, realism, diversity, OCR exactness,
palette/logo/safe-area fidelity, and blinded quality ratings. Use pinned snapshots, fixed
prompts, repetitions, paired comparisons, stratified judging, and confidence intervals.

## Critical review

Score: **9.2/10**.

The remaining uncertainty is explicit: the account's actual GPT Image entitlement and rate
tier, the local runtime repair, and missing Grau Industries brand assets. Those are live or
user-input gates, not reasons to widen Phase 1. The plan also avoids a common false shortcut:
creating every final banner independently with the image model, which would reduce brand
consistency and cannot satisfy several standard banner dimensions directly.

## Current official references

- https://developers.openai.com/api/docs/models/gpt-image-2
- https://developers.openai.com/api/docs/guides/image-generation
- https://developers.openai.com/api/docs/guides/batch
- https://developers.openai.com/api/docs/pricing
