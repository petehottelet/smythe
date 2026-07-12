# Parallel image generation — concurrency sweep (2026-07-12)

The first image-pipeline benchmark with entirely **objective** metrics:
wall time, per-node latency, cost, decode/format compliance, and
pairwise dHash diversity. No LLM judge anywhere in this table, so none
of the ±0.3 judge variance documented in [README.md](README.md) applies.

Harness: [run_image_benchmarks.py](run_image_benchmarks.py) (offline
mode runs free in any checkout). Raw records:
[results/image_k_sweep.json](results/image_k_sweep.json).

## Protocol

8 distinct Golden Gate Bridge scene prompts per run, broadcast through
`Swarm(parallel=True, max_concurrency=k)` on `gemini-2.5-flash-image`
($0.039/image), budget-capped with the cost-aware reservation protocol,
`RETRY` failure policy with full-jitter backoff armed. 3 repeats per
concurrency level; 72 live images total ($2.81).

## Results

| k | Wall (mean [range]) | Throughput | Efficiency vs ideal | Success | Cost/run |
|--:|---|---:|---:|---|---:|
| 1 | 46.3s [45.4–46.9] | 10.4 img/min | 1.00 (baseline) | 24/24 | $0.312 |
| 3 | 16.9s [16.6–17.1] | 28.4 img/min | **0.88** | 24/24 | $0.312 |
| 8 | 7.0s [6.3–7.6] | **68.8 img/min** | **0.81** | 24/24 | $0.312 |

- **6.6× wall-clock speedup at k=8** (46.3s → 7.0s) at identical cost —
  parallelism buys wall time for free; the budget question is what you
  generate, not how you schedule it.
- Efficiency declines gently with width (1.00 → 0.88 → 0.81), driven by
  the longest-call straggler in each wave, not by framework overhead
  (executor overhead measured ≲0.3s per run).
- **No rate limiting observed up to k=8** on a single paid-tier key:
  zero retries fired, 9/9 runs completed, recorded cost matched the
  per-image price exactly.
- Cross-provider context: a prior recorded GPT-Image run
  (`gpt-image-2`, 1536×1024 medium, k=3) showed the same executor
  efficiency (0.90) at ~6× the per-image latency (~39s vs ~6s) and
  near-identical price — see `plans/05-gpt-image-benchmark.md`.

## Objective quality checks

All 72 images decoded as valid 1024×1024 PNGs (100% format
compliance at the model's default size). Pairwise dHash diversity was
healthy overall (mean 24–31 bits across runs) **with one honest
exception**: one k=8 run produced a near-duplicate pair (min pairwise
distance 3 bits), and one k=1 run a close pair (7 bits), from *distinct*
scene prompts. Near-duplicate rate is therefore a real failure mode for
uncurated fan-out — the motivation for a select-from-N curation tier
(generate N, dedup + pick the best) rather than trusting one sample per
prompt.

## Caveats

- n=3 per cell: ranges shown; treat sub-second deltas as noise. Single
  day, single region, single key — no time-of-day or key variance.
- 1024×1024 only: `image_config` aspect-ratio control shipped in the
  framework but is not yet exercised live; format compliance at
  requested non-square sizes is an open cell.
- k>8 unmeasured; the rate-limit ceiling on one key is not yet found.
- Prompts authored by this project.

## Planned next

- Aspect-ratio compliance cell (`image_config` live).
- k=16/25 to find the single-key ceiling; multi-key pooling after.
- Select-from-N curation tier with a vision judge (quality-per-dollar
  curve), and the ad-suite brand-consistency comparison (shared brief
  vs. serial style stage vs. single agent).
