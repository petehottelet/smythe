# Benchmark delivery audit — 13 September 2026

This inventory reconciles the README, roadmap, benchmark index and subsystem
protocols. Software tests and prepared experiments are distinguished from
completed measurement campaigns.

## Current glyph work

The [v2 protocol](../benchmarks/svg_v2_protocol.md) compares **192 and 256
authored glyphs** under the same compilation, four-size validation, all-pairs
distinctness and export workload. The 256-glyph set adds 64 structures while
preserving all 192 reviewed SVGs. This is separate from both the historical
v1 generation campaign and the older simulated-latency width sweep.

**Delivered locally:** [all 36 v2 workflows](../benchmarks/svg_v2_results.md) pass.
The best 256-glyph configuration completes in 8.06 seconds median;
both sizes have complete timing, memory and artifact evidence.

## Promised measurements still outstanding

| Commitment | What exists | What remains |
|---|---|---|
| [Astra review and follow-ups](../benchmarks/astra_benchmark_plan.md#separate-follow-up-studies) | The [primary 200-workflow comparison](../benchmarks/results/astra_20260913_main/README.md) has complete execution, automatic scoring and native audit, including its failed outcome and retained reserve | Complete human review of eight flagged answers; reconcile one missing native usage receipt. Run separately frozen scheduler, modern framework and tool studies; broaden the external task set. |
| [Complete-workflow task-shape cost](../benchmarks/shape_suite.md) | Wall time includes planning; historical cost uses execution/synthesis estimates | New native-ledger cost measurements including planning, input/output prices, failed attempts and judging. The cost-per-quality hypothesis is still open. |
| [Broader quality and latency evidence](../benchmarks/README.md#coming-soon) | Small project-authored suites and one independent judge | Larger external task set, repeated randomized execution, stronger score discrimination and human calibration with retained outputs/judge reasoning. Astra's prepared synthetic tasks alone do not fulfill the external-task promise. |
| [Additional criteria-arm repetitions](../benchmarks/control_ablation.md) | One valid 15-run criteria campaign; other arms have more samples | Replicate the criteria result and measure the unresolved complete-deliverable failure cases. |
| [Memory benefit on corrective tasks](../benchmarks/README.md#memory-onoff) | A completed memory on/off experiment with a null result; runtime history validation is implemented | A task family with planted mistakes that recall can correct. Runtime regression tests do not establish learning gains. |
| [Original-v5 framework comparison](../benchmarks/README.md#coming-soon) | Corrected fixed-pipeline comparison on a different executor | Same-protocol `claude-opus-4-8` rerun for historical comparability. This is a legacy commitment; a modern Astra comparison answers a different question. |
| [Image select-from-N and brand consistency](../benchmarks/image_benchmarks.md#planned-next) | Working curation example, exact-spec asset runs and judge-variance study | Quality-per-dollar curve; matched shared-brief, serial-style and single-agent brand-consistency arms. |
| [Repeated image and live glyph scaling](../benchmarks/image_benchmarks.md#planned-next) | A single k=25 image cell and earlier live glyph lanes | Repeated k=25 image trials and live glyph concurrency cells; retain failed calls and all paid usage. The new local SVG campaign does not measure API throughput. |
| [Paid scale ladder](../ROADMAP.md#product-and-scale) | One reconciled offline 5,000-operation recovery campaign on schema v3 | Bounded paid 50/250/1,000-item recovery trials and a separately measured current schema-v4 run. |
| [Visible renderer presentation and GPU timing](../benchmarks/renderer_performance_20260907_results.md) | Six completed headless primary runs, blank-page controls and a ten-minute soak; all primary pacing targets missed | New-glyph 1080p timing, visible display cadence, GPU execution timing and quantified reference parity. Native exposure/navigation parity and platform execution require separate checks. |

## Delivered measurements that should not remain “coming soon”

- The [Astra/Sol primary study](../benchmarks/results/astra_20260913_main/README.md)
  retains all 200 outcomes, with 191 accepted by the frozen automatic rule. Native
  phase costs and one retained unknown reservation are reconciled as bounds.
  Exact cost contrasts touching that failed call remain withheld.

Human review of eight flagged main answers remains pending. These records are
not claimable until that review is complete; automatic classifications are retained.

- The [64/128/192/256-node controlled width sweep](../benchmarks/glyph_screensaver_benchmark.md)
  exists. It uses the earlier tile method and simulated provider latency.
- The [192-glyph v1 workflow](../benchmarks/svg_glyph_benchmark.md) exists,
  including validation and assembly. Its artwork is historical.
- The [5,000-operation recovery observation](../benchmarks/jobs_scale_5000_20260907_results.md)
  is complete and independently reconciled, within its stated offline/schema-v3 scope.
- The [headless renderer campaign](../benchmarks/renderer_performance_20260907_results.md)
  is complete. Missing its target is a result, not an unrun experiment.

Native Wayland support, native navigation/settings parity, deterministic
deliverable contracts and signing are product work rather than delivered
performance evidence. Precompiled distribution remains paused.

## Priority

1. **Completed:** matched v2 glyph measurements, all repetitions, contact
   sheets, and timing/memory charts. Prepared locally for repository publication.
2. **Completed:** calibrated Astra main comparison, blind judging and native
   evidence review. Reconcile the one unresolved usage receipt before exact
   affected cost claims; keep all diagnostic history and the failed outcome.
3. Run visible renderer/GPU measurements with the current glyphs.
4. Extend live image, glyph and recovery scale measurements under separate
   spending ceilings, then broaden external-task and human quality evaluation.

The old-protocol framework rerun is lower priority than the modern matched
campaign. Keep it labeled as outstanding until it is run or explicitly retired.
