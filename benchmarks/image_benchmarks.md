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

## Follow-up cells (2026-07-12, same day)

**Aspect-ratio compliance** ([results/image_aspect_ratio.json](results/image_aspect_ratio.json)):
with `image_config={"aspect_ratio": "16:9"}`, 4/4 images returned native
widescreen **1344×768** — no letterboxing, uniform dimensions. (Note:
1344×768 is Gemini's 16:9 output bucket; its true ratio is 1.75 vs the
nominal 1.78.) This closes the earlier failure where wide-format
requests came back letterboxed inside a 1024×1024 square.

**k=25 ceiling probe** ([results/image_k25_ceiling.json](results/image_k25_ceiling.json)):
25 prompts, `max_concurrency=25`, single run — **25/25 images in 10.2s
wall ($0.975)**, 147 images/min throughput, zero rate-limit events on
one paid key. Concurrency factor 15.9 (64% of ideal 25): efficiency
declines at this width because per-call latency spreads 4.8–8.3s and
stragglers dominate the wall clock, not because anything queued or
failed. The single-key rate ceiling is therefore **above 25 concurrent
image requests** — multi-key pooling is not yet needed at this scale.
Diversity note: this run cycles 8 scene prompts across 25 nodes, so
pairwise dHash includes same-prompt pairs; min distance 7 bits means
even same-prompt regenerations differed materially.

## Caveats

- n=3 per cell: ranges shown; treat sub-second deltas as noise. Single
  day, single region, single key — no time-of-day or key variance.
- 1024×1024 only: `image_config` aspect-ratio control shipped in the
  framework but is not yet exercised live; format compliance at
  requested non-square sizes is an open cell.
- k>8 unmeasured; the rate-limit ceiling on one key is not yet found.
- Prompts authored by this project.

## The asset suite: 8 exact-spec launch assets (2026-07-12)

The README's broadcast-reduce example run for real
([run_asset_suite.py](run_asset_suite.py), records in
[results/asset_suite.json](results/asset_suite.json)): one brand brief,
eight assets with exact pixel specifications — hero 2400×1200, App
Store 1290×2796, print 2550×3300 at 300 dpi, and five more.

**Result: 8/8 specs met, 11.0s generation wall, $0.312.** Five aspect
buckets ran as concurrent parallel swarms (one Gemini `image_config`
per provider); a deterministic finishing pass (resize-to-cover +
center-crop) produced the exact dimensions, because no image model
emits 2400×1200 or 300-dpi print natively.

**Brand lock (same day, second run — 8/8, 13.2s + 6.4s logo, $0.351):**
assets for a brand must share its actual mark, not eight independent
hallucinations of one. The suite now confirms possession of a brand
logo first (`--logo path`) or generates one in a dedicated stage, and
every asset node receives the logo pixels as a reference image
(`attach_dep_artifacts=True` feeding the model's image-input channel).
Verified by inspection: the generated sun-over-horizon mark reproduced
faithfully across assets from *different* concurrent swarms — the
consistency mechanism is the shared reference image, not shared
sampling. Recorded in
[results/asset_suite.json](results/asset_suite.json) under
`brand_logo` (source: provided vs. generated, plus its own wall time
and cost).

**Measured, not eyeballed (third run — logo 6.3s, 8/8 specs in 12.1s,
$0.384 total):** with `--judge`, a vision node receives the official
logo plus all eight finished assets and scores brand consistency per
asset. Result: **overall 8/10**, per-asset 7–9, with genuinely
discriminating defect notes — it distinguished the *mark* (reproduced
faithfully everywhere) from the *secondary typography* ("Power from
the sun" weight/spacing drifts from the master logo), and flagged
perspective-distorted logo renderings on physical product surfaces.
Full pipeline: confirm-or-create logo → brand-locked parallel
generation across 5 concurrent aspect swarms → deterministic
exact-spec finishing → measured brand verification, in ~19 seconds
end to end. Caveats: n=1, one judge model, scores not yet calibrated
against human ratings.

Honesty metrics per asset: upscale factor and cropped fraction are
recorded — print ad upscaled 2.95× and App Store 2.08× beyond native
model resolution (the visible quality cost of oversized specs), while
five of eight assets finished at ≤1.4× (near-native). Known defect
class observed: small on-device brand text renders imperfectly at
hero size — the tiny-text fidelity limit of current image models, and
precisely the defect the vision-judge curation tier
(`examples/11_vision_judge.py`) is built to catch. This run measured
generation + finishing only; adding the judge as a reduce stage is the
ad-suite consistency benchmark, next.

## Two brands, one pipeline: text complexity drives consistency (2026-07-12)

The suite is brand-parameterized ([brands/](brands/)); the same
pipeline ran for two brands at identical cost and speed:

| Brand | Tagline | Judge overall | Dominant defect class |
|---|---|---:|---|
| Osiris | "Power from the sun" | **8/10** | secondary-typography drift |
| MetaCortex Datacentres | "Infrastructure for the intelligence age" | **3/10** | systematic tagline misspellings |

Same model, same brand-lock mechanism, same 8/8 format compliance,
~$0.38 each — a five-point consistency gap
([results/asset_suite_metacortex_datacentres.json](results/asset_suite_metacortex_datacentres.json)).
The judge attributed it to text: "Infrastrcture" appears in 7 of 8
MetaCortex assets, alongside "Datacnetres", "inteligence", and
gibberish glyphs in UI/panel text — all verified accurate by human
inspection of the images. Short, common tagline words render reliably;
long or uncommon words misrender *systematically*, not occasionally.

**Implication:** for text-heavy enterprise branding, generated
typography is not production-ready regardless of logo locking. The
data validates the master-plates approach (generate imagery, composite
exact copy deterministically) and gives the select-from-N curation
tier a measurable target. It also validates the judge itself: every
typo it reported exists in the pixels.

**The fix, measured (same day): 3/10 → 8/10.** With
`composite_tagline` in the brand config, generation prompts suppress
in-image typography ("leave clean negative space") and the finishing
pass composites the exact tagline deterministically (Pillow,
luminance-adaptive ink). Rerun: **overall 8/10, zero misspelling
defects** — the entire typo class eliminated in one run at identical
cost ($0.384); residual defects are the same logo-fidelity class as
Osiris at its 8/10. Honest residue: the negative-space instruction
sometimes yields a visible empty placeholder box the compositor
doesn't fill — placement-aware compositing is the refinement.
MetaCortex now matches Osiris despite a tagline that is 3x harder to
render, because it no longer renders it.

## Judge variance and the optimizer loop (2026-07-12)

**Judge variance, measured** ([run_judge_variance.py](run_judge_variance.py),
[results/judge_variance.json](results/judge_variance.json)): re-judging
identical pixels 5 times gives overall **7.6 ± 0.49 (range 7–8)**,
per-asset spread up to ±1. Consequences: single-run deltas under ~1
point are noise; the 3/10 → 8/10 compositing fix was ~10σ.

**The optimizer** ([run_optimizer.py](run_optimizer.py)) is the
autoresearch pattern applied to this suite: an LLM proposes one
targeted prompt-policy change per iteration, the full suite runs live,
and the candidate is scored as the mean of 3 independent judgings
(sd of mean ~0.28) — kept only if it beats the incumbent by >0.5
(~2σ) AND passes the un-gameable format gates (8/8 exact specs).
Every experiment is journaled
([results/optimizer_journal.jsonl](results/optimizer_journal.jsonl)).
Smoke run (~$0.90): baseline 7.67; iteration 1 proposed a sensible
logo-prompt simplification targeting the observed fidelity defects,
scored 7.67, and was **correctly reverted** — the keep rule held
against noise. Iteration cost ~$0.45; a 20-iteration overnight run is
~$9.50.

## Planned next

- Select-from-N curation tier with a vision judge (quality-per-dollar
  curve), and the ad-suite brand-consistency comparison (shared brief
  vs. serial style stage vs. single agent).
- Repeats for the k=25 cell (single run today) and k>25 if a workload
  ever needs it.
