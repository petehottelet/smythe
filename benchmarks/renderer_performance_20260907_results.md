# REGL renderer: measured pacing and stability

**The six-session campaign completed with valid evidence and did not meet its
pacing target.** Classic and 3D produced 56.21–56.24 measured draws/second, with
an 18.10 ms P95 callback interval in every run. A separate blank-page control
showed almost the same callback cadence. These are headless browser diagnostics,
not a physical presentation, GPU execution-time, or cross-renderer speed claim.

[Frozen protocol](renderer_performance_20260907.md) ·
[Follow-up plan](results/glyph_rain_regl_20260907_followup1_plan.json) ·
[Raw aggregate](results/glyph_rain_regl_20260907_f1.json) ·
[Independent arithmetic/source review](results/glyph_rain_regl_20260907_f1_review_v2.json).

## Workload and environment

Three fresh sessions per preset ran in the frozen Classic/3D, 3D/Classic,
Classic/3D order. Each used five seconds of warmup and at least 60 actual
seconds of sampled callback intervals. The gate required **every repetition**
to reach at least 59.5 draws/second and a P95 interval no greater than 16.7 ms.
The thresholds were not changed after observing results.

The CSS viewport was 1920×1080 at DPR 1. The preset's 0.75 render scale produced
a **1440×810 drawing buffer**, with 80 columns, density 1, and a 60fps drawing
cap. The default mix selected the 192 original glyphs with 10% probability;
the other selections used 56 visible reference glyphs and the blank slot.
Matrix green, bloom, illumination, and all remaining preset settings stayed
unchanged. Startup was paused at simulation tick/time zero; the receipts bind
the complete resolved engine configuration and sampled simulation boundaries.

The host was Windows (kernel 10.0.26200), an AMD Ryzen 9 5950X with 32 logical CPUs,
and Chrome 152.0.7977.82 in headless mode through agent-browser 0.25.3.
The actual WebGL context reported **NVIDIA GeForce RTX 4090 through ANGLE/D3D11**,
matching the device inventory and auxiliary renderer; driver 32.0.15.9186.
The power profile was Ultimate Performance. Existing Chrome, Firefox,
Creative Cloud, NordVPN, cam_helper, and ChatGPT activity remained present.
No competing owned tests, builds, or benchmarks ran during the timing window. This was
an active desktop, with no idle-host claim.

The renderer/harness freeze is commit `41e4f9d`. During the campaign, HEAD moved
to `5c1939b` for a completion-tracker documentation update. All 46 frozen source
and harness hashes, the served assets, and the protocol remained identical.
The raw records preserve full launch arguments and backend evidence. The
enumerated but unused Microsoft fallback adapter does not describe the actual
NVIDIA context.

## Every primary repetition

| Order | Preset / repetition | Actual sample | Draws / second | P95 callback interval | P95 CPU submission | Target |
|---:|---|---:|---:|---:|---:|---|
| 1 | [Classic 1](results/glyph_rain_regl_20260907_f1_r1_classic.json) | 60.0072 s | 56.2433 | 18.10 ms | 0.20 ms | Not met |
| 2 | [3D 1](results/glyph_rain_regl_20260907_f1_r1_3d.json) | 60.0011 s | 56.2156 | 18.10 ms | 0.30 ms | Not met |
| 3 | [3D 2](results/glyph_rain_regl_20260907_f1_r2_3d.json) | 60.0057 s | 56.2447 | 18.10 ms | 0.20 ms | Not met |
| 4 | [Classic 2](results/glyph_rain_regl_20260907_f1_r2_classic.json) | 60.0011 s | 56.2156 | 18.10 ms | 0.30 ms | Not met |
| 5 | [Classic 3](results/glyph_rain_regl_20260907_f1_r3_classic.json) | 60.0058 s | 56.2446 | 18.10 ms | 0.20 ms | Not met |
| 6 | [3D 3](results/glyph_rain_regl_20260907_f1_r3_3d.json) | 60.0067 s | 56.2104 | 18.10 ms | 0.20 ms | Not met |

The displayed values are rounded; pass/fail used the unrounded records.
All six passed source, configuration, initialization, duration, raw-summary,
and browser-error validation. An independent Python standard-library check
recomputed durations, average rates, inclusive-linear P95s, schedule order,
and file identities from the raw records. Every original receipt remains
unchanged, including its diagnostic status and failed target result.
The [independent verifier](verify_renderer_performance_20260907.py) reproduces
the v2 review receipt byte-for-byte from the saved primary/control records and
50 actual frozen source files. It also checks control viewport, chronology,
and matching GL/device identities. The narrower
[initial review](results/glyph_rain_regl_20260907_f1_review.json), which compared
declared source hash maps, remains unchanged.

Callback intervals measure the timestamps selected for drawing. CPU submission
uses `performance.now()` around host simulation/camera work and GL submission,
including work accumulated between selected callbacks. It is neither operating
system CPU utilization nor elapsed GPU execution. Physical presentation,
GPU/process memory, GPU execution time, and power consumption were not measured.

## Blank-page cadence control

After the primary campaign, a separate
[control protocol](renderer_cadence_control_20260907.md) and
[source-bound plan](results/glyph_rain_regl_20260907_cadence_plan.json) were frozen.
Three new browser sessions used the same launch configuration, viewport, power
profile, and desktop workload declaration. Each used `about:blank` and an
unattached 2×2 WebGL context for backend identification, with no rain submission.

The control recorded every requestAnimationFrame timestamp and the subset
selected by the unchanged `timing.mjs` gate. Both streams span at least 60
seconds after five seconds of warmup. Every source hash, actual backend,
summary, and subset check passed.

| Repetition | Raw callbacks / second | Selected callbacks / second | Raw P95 interval | Selected P95 interval | Callbacks omitted by gate |
|---|---:|---:|---:|---:|---:|
| [1](results/glyph_rain_regl_20260907_cadence_r1.json) | 56.3279 | 56.2779 | 18.10 ms | 18.10 ms | 3 of 3,381 |
| [2](results/glyph_rain_regl_20260907_cadence_r2.json) | 56.3217 | 56.2384 | 18.10 ms | 18.10 ms | 5 of 3,381 |
| [3](results/glyph_rain_regl_20260907_cadence_r3.json) | 56.3545 | 56.2711 | 18.10 ms | 18.10 ms | 5 of 3,383 |

The blank page's source cadence closely matches the rain measurements. This
supports a browser/host cadence explanation in this environment. It does not
identify the cause, establish a GPU bottleneck, or turn the primary result into
a pass. The control is exploratory and has no performance gate.

## Retained failures and visual review

The [first attempt](results/glyph_rain_regl_20260907_r1_classic.json) stopped
before timing because Chrome required `--enable-automation` to return its
launch arguments. A metadata-only
[first preflight](results/glyph_rain_regl_20260907_metadata_preflight_f1.json)
then exposed a classifier error: an unused fallback adapter overrode the
actual NVIDIA context. Its original classification remains a superseded
diagnostic. The corrected
[second preflight](results/glyph_rain_regl_20260907_metadata_preflight_f2.json)
passed before any follow-up timing. Both fixes have focused regressions.
No primary workload, artwork, shader, or numerical target changed.

The [current browser review](partitions/glyph_rain_reference_v1/performance-preview-review-20260907.json)
passed 31 checks and five visual samples against the same renderer source.
It covers both catalogs, navigation, settings, lifecycle, and interruption.
The screenshot SHA-256 is
`7c27b2a2c16abed59672fc5cc2ce8adf56eba165b022660b302b522d2afb1af8`.

## Separate stability check

The [600.95-second soak](results/glyph_rain_regl_20260907_f1_soak.json) passed
**57 travel/resize cycles** across 1920×1080, 1280×720, 390×844, and 2560×1440.
The camera stayed finite and wrapped within the repeating world. Catalog
counts and the 90/10 mix stayed fixed; rain time advanced monotonically through
resize and travel. Blur released input, and pause stopped further drawing and
movement. Served assets, renderer source, and the soak harness stayed unchanged.

The [stability review and workload declaration](results/glyph_rain_regl_20260907_f1_soak_context.json)
records concurrent 5,000-operation Jobs recovery work, Jobs/operator correctness
checks, short renderer verification commands, and existing desktop activity.
The soak is stability evidence under competing load. Its retained JS heap
samples do not measure total process or GPU memory; it is not a pacing sample.

## Reproduction and next measurement

Use new output paths: the helpers refuse to replace retained records.
The [primary protocol](renderer_performance_20260907.md) declares all six
sessions and aggregation; the [control protocol](renderer_cadence_control_20260907.md)
declares the three blank-page sessions. Run focused offline guard checks with:

```bash
node screensaver/svg-preview/verify-lifecycle.mjs
node screensaver/svg-preview/verify-measurement.mjs
node screensaver/svg-preview/verify-browser-metadata.mjs
node screensaver/svg-preview/verify-raf-control.mjs
python benchmarks/verify_renderer_performance_20260907.py --out new-renderer-review.json
```

If the working renderer has since changed, pass `--source-root` pointing to a
checkout retaining the frozen primary and control source bytes. The verifier
rejects mismatched bytes instead of substituting newer source files.

The next performance step is a separately declared visible-browser campaign
with measured display cadence, followed by disjoint-safe GPU timing if supported.
Keep the same artwork and exposure settings while locating the pacing limit.
Full-resolution rendering and native-port parity require their own measurements.
