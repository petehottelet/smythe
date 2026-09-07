# REGL renderer performance protocol

**Protocol frozen before the first performance sample.** This campaign measures
the current [web explorer](../screensaver/svg-preview/README.md) after its
benchmark-completion scheduling fix. It does not inherit the archived Canvas
renderer's results or establish performance for the compiled native savers.
No provider calls or artwork generation are involved.

## Frozen workload

The primary campaign contains six independent browser sessions. Each samples
60 seconds after five seconds of warmup. Use one session at a time, on the same
idle host and browser build, with the settings sheet closed and normal idle
chrome. Run no tests, builds, browser reviews, recordings, or other benchmarks
during sampling. Record the actual power profile and competing workloads;
absence of a measurement is not proof that the host was idle.

| Setting | Classic | 3D |
|---|---|---|
| Preset | `classic` | `3d` |
| CSS viewport / device-pixel ratio | 1920 × 1080 / 1 | 1920 × 1080 / 1 |
| Render scale / drawing buffer | 0.75 / 1440 × 810 | 0.75 / 1440 × 810 |
| Grid / density | 80 × 80 / 1 | 80 × 80 / 1 |
| Frame-rate cap / simulation rate | 60 / nominal 60 Hz | 60 / nominal 60 Hz |
| Original catalog selection probability | 10% | 10% |
| Reference selection | 57 slots: 56 shapes and one blank | Same |
| Original selection | All 192 accepted original SVGs | Same |
| Palette / cursor | Matrix green 137°, 80% / `#A2FFD8` | Same |
| Camera input | None | None; preset automatic travel remains enabled |
| Initial camera / time / tick | `(0,0)` / 0 / 0 | `(0,0)` / 0 / 0 |

These are **1080p viewport measurements at 0.75 render scale**, not a
1920 × 1080 drawing-buffer workload. Classic draws a stationary screen-space
grid; 3D draws 6,400 quads at density one. Preserve each preset's remaining
values from [config.mjs](../screensaver/svg-preview/config.mjs) and its pinned
upstream defaults, including exposure, bloom, cycling, and travel. Store the
complete resolved configuration with each receipt. Do not reduce density,
columns, glyph mix, glow, or resolution to pass a target.

Declare full-resolution runs separately if undertaken: the same two presets,
three repetitions each, with only `resolution=1` changed, yielding a
1920 × 1080 drawing buffer. They have their own six-run gate. A favorable
0.75-scale result cannot substitute for a full-resolution result, or vice versa.

## Initialization and order

Use schedule seed **20260907** and this fixed order, reversing the preset order
in the middle block:

| Position | Repetition | Preset |
|---:|---:|---|
| 1 | 1 | Classic |
| 2 | 1 | 3D |
| 3 | 2 | 3D |
| 4 | 2 | Classic |
| 5 | 3 | Classic |
| 6 | 3 | 3D |

This order is a frozen schedule, not a randomization claim. The field itself
has no configurable random seed: its random values come from the frozen GLSL
hash functions and their coordinate/time inputs. Record that seed policy;
do not invent a user seed or replace the shader's random functions.

Before opening each scene, configure the fresh browser session for reduced
motion and the declared viewport. Wait for `GlyphRainPreview.version === "2"`,
then require paused playback, tick/time zero, camera zero, DPR one, and the
expected configuration and drawing-buffer dimensions. Verify served source
hashes while paused. `runBenchmark()` then starts the real animation clock and
the five-second warmup. Record the actual first/last sampled tick and time;
callback timing can change which simulation tick starts a sample. Reduced
motion is a reproducible initialization step, not a paused workload.

Do not use `stepForReview()`, `moveForReview()`, pixel readback, screenshots,
or synthetic frame timestamps during timing samples. User-driven travel is
tested separately by the movement/resize soak. The primary 3D timing condition
already includes the preset's automatic forward motion.

## What the numbers mean

[rain.js](../screensaver/svg-preview/rain.js) currently collects:

- **Draw-callback interval:** the difference between successive
  `requestAnimationFrame` timestamps selected by the drawing gate. It is neither
  a draw-completion timestamp nor the interval between physical presentations.
- **Average draws per second:** interval count divided by the elapsed time
  between first and last sampled draw callbacks.
- **CPU command-submission duration:** `performance.now()` elapsed time around
  camera updates, simulation steps, and WebGL submission, accumulated across
  callbacks between draws. It excludes later asynchronous GPU execution and
  other process CPU work; it is not an operating-system CPU-utilization measure.

Retain every raw interval and submission sample. Report sample count, mean,
median, P95, and maximum per repetition, with the existing linear-interpolated
quantile definition. Report each repetition's measured duration, average draw
rate, simulation-tick delta, and initial/final state. Do not pool frames across
repetitions to hide an unsuccessful run. Compare configurations using the
distribution of run-level summaries.

GPU execution, physical display presentation, energy, total process memory,
and total GPU memory remain **unmeasured**. An optional future GPU timer study
must use asynchronous timer queries and reject disjoint/incomplete queries;
it still would not measure physical presentation. No synchronous GPU waits
belong in the primary callback/submission study.

The existing helper launches a headless browser. Preserve that mode for the
primary campaign and label it explicitly. A headed run is a separate condition.
Identify the actual graphics backend: software rasterization results may be
reported as software-browser diagnostics, but cannot qualify a hardware-GPU
claim. Browser callback cadence may be virtualized or refresh-quantized;
neither headless execution nor an apparent 60 Hz cadence proves display timing.

## Validity and pass gates

Set gates before collecting data. A valid repetition must have:

1. A completed receipt, at least 60 seconds sampled after at least five seconds
   of warmup, nonempty finite nonnegative CPU samples, and positive finite frame
   intervals. Interval samples must reconcile to measured duration; CPU samples
   contain one entry for each sampled draw, including the first draw.
2. Matching initial/final preset, full configuration, viewport, DPR, render scale,
   catalog identities, and source/harness hashes. No settings changes, manual
   navigation, pause, visibility/focus loss, resize, context loss, or browser
   errors during the sample.
3. The frozen initialization/order, identified browser/graphics backend, and a
   declared headless/headed mode. Unexpected dimensions or an unavailable
   renderer are failed attempts, not alternate workload settings.

For each preset, the timing target passes only if **all three valid repetitions**
have **P95 draw-callback interval ≤16.7 ms** and **average draws/s ≥59.5**.
Both presets must pass for a combined primary-campaign pass. CPU submission is
reported separately without an invented GPU-completion gate. A hardware claim
also requires a verified hardware backend. Record target failures exactly;
do not round a failing P95 down to 16.7 ms.

A 60 FPS cap on a 144 Hz callback source can legitimately alternate two- and
three-refresh intervals, with three refreshes taking approximately 20.83 ms.
Record such cadence constraints rather than changing this gate after seeing
the data. Missing or failed repetitions block qualification. Keep every failed
attempt and its reason; any replacement is an additional recorded attempt,
not deletion of the original. A comparative improvement requires a separately
frozen baseline and candidate under matched settings and host conditions.

## Evidence and prerequisites

The [measurement helper](../screensaver/svg-preview/measure.mjs)
retains samples, local/served source hashes, source stability, settings, and
host/browser summaries, and refuses to overwrite receipts. Its preparation
and offline regressions now:

- Establish the paused zero-time startup above and wait for readiness. The
  previous immediate ready check could fail during a cold asset load and started
  the field before source verification.
- Freeze and validate the scenario and schedule, retain failed launch/timeout
  attempts as receipts, and aggregate all repetitions with the gates above.
- Capture the exact browser build/executable identity and launch arguments,
  browser-tool version, actual GL backend/device and available extensions,
  host OS/build/CPU, DPR, headless mode, and power/workload declarations.
  A generic WebGL renderer string alone does not establish hardware use.
- Record sample boundary simulation state and the actual engine configuration,
  including values inherited from upstream defaults. Reconcile raw samples,
  not just precomputed summary fields.

Use [source-files.mjs](../screensaver/svg-preview/source-files.mjs) for renderer
dependencies. Bind the exact bytes served for the host, timing/settings code,
all adapted shaders/passes, pinned REGL and matrix libraries, fonts/chrome,
both MSDF textures, and their provenance. Bind this protocol and every campaign
runner/aggregation module as harness sources. Preserve Git revision/dirty state
as context; content hashes remain authoritative for uncommitted or copied files.
Hash both before and after each session. Do not transfer old receipt hashes to
new source bytes or silently rewrite line endings inside an evidence record.

Proposed output names are
`benchmarks/results/glyph_rain_regl_20260907_r{1,2,3}_{classic,3d}.json`
and `benchmarks/results/glyph_rain_regl_20260907.json`, plus separately named
launch/failure records. The aggregate should remain diagnostic until independent
review confirms its samples, scope, complete attempt inventory, and gates.

Required local dependencies are Node.js 22 or later, the `agent-browser` executable and its
Chromium installation, and a static localhost HTTP server. REGL, matrix math,
fonts, and glyph textures are already vendored. No provider SDK, paid API, GPU
profiler, or native rebuild is required for the primary callback study.

Run one attempt with the scenario's URL and declared repetition number:

```bash
node screensaver/svg-preview/measure.mjs "http://127.0.0.1:8000/screensaver/svg-preview/?preset=classic" NEW_RECEIPT.json 60 5 1
```

Follow the six-position schedule above. Set `SMYTHE_MEASURE_POWER_PROFILE` and
`SMYTHE_MEASURE_WORKLOADS` to the observed host conditions before launching.
Each attempt, including a launch failure, gets its own new receipt path.
The helper does not retry or overwrite a failed attempt. Aggregate the six
declared receipts with
`node screensaver/svg-preview/aggregate-measurements.mjs NEW_AGGREGATE.json RECEIPT_1.json RECEIPT_2.json RECEIPT_3.json RECEIPT_4.json RECEIPT_5.json RECEIPT_6.json`.
The aggregator recomputes raw summaries and checks source, host/browser,
scenario, and chronological consistency. It does not grant claimable status.
Additional replacement attempts require an explicit follow-up campaign and
must retain the original failed campaign and all its records.

## Optimization candidates and visual preservation

Establish the post-fix baseline first. These are hypotheses to profile, not
measured improvements:

1. Disable unused depth/stencil attachments on postprocessing framebuffers;
   the fullscreen pipeline disables depth testing. Validate output before
   attributing resource or timing savings.
2. Consolidate redundant `regl.poll()` work across simulation and draw calls,
   preserving resize, paused redraw, and context lifecycle behavior.
3. For Classic/3D settings with thunder and ripples disabled, replace the
   per-tick constant effect pass with equivalent fixed data. Preserve the
   existing effect path for presets that use it.
4. Profile eliminating the final fullscreen copy by drawing the final palette
   stage to the presentation buffer. Preserve color precision, dimensions,
   readback semantics, and pause behavior before measuring it.

The current nominal sequence has four state passes and nineteen drawing/
postprocessing passes at one simulation step per draw. Extra simulation
catch-up steps add work. Bloom and 3D overdraw are plausible GPU costs, but
callback or submission measurements alone cannot identify GPU bottlenecks.
Reducing bloom, replacing glyphs, lowering the mix, or cutting resolution is a
workload change, not a style-preserving optimization.

Before comparing a candidate, check fixed ticks/cameras and all supported
presets against the frozen baseline on the same browser/backend. Use output
readback outside timing runs, keep difference images and exact/tolerance rules,
and review near glyph edges, bloom, cursor color, and depth wrapping. Do not
select a tolerance after seeing an unexplained difference.

## Current review and native scope

The scheduling fix changes `rain.js`, so earlier browser/style receipts no
longer bind the complete current renderer source. Preserve them as historical
checkpoints. Run a fresh [review.mjs](../screensaver/svg-preview/review.mjs)
after the renderer/harness freeze and before performance sampling, in a separate
session. It should recheck the actual output, interactions, and source identity.
The public screenshot need not be replaced if visual comparison confirms the
unchanged appearance; its old receipt still must not be presented as a current
full-source verification.

After timing, run the current [ten-minute soak](../screensaver/svg-preview/soak.mjs)
separately. It checks travel, resizing, pause/focus release, and finite state;
its JavaScript heap samples are not total graphics-memory measurements.

The compiled GDI+, Core Graphics, and Cairo savers already share the exact
catalogs but retain their native three-layer motion. Their next porting work
is the web exposure pipeline and explorer controls, with OS screensaver input
policy preserved. This campaign cannot establish native parity or the separate
native 40 FPS/720p target. Those need compiled-host timing and visual checks.
