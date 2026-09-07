# Blank-page callback cadence control

This exploratory protocol is frozen before its first control sample. It follows
the completed six-session [primary campaign](results/glyph_rain_regl_20260907_f1.json),
which missed the original pacing gates. Those gates and results remain unchanged.

Run three fresh headless Chrome sessions, repetitions 1, 2, 3, sequentially.
Use the primary campaign's browser launch helper and explicit automation flag,
1920×1080 CSS viewport, DPR 1, reduced-motion preference, power profile, and
observed desktop workload declaration. Do not close unrelated applications.
Record actual Chrome version/arguments, GL context and physical adapter inventory.

Each page is `about:blank`. Create an unattached 2×2 WebGL context only to identify
the backend; do not draw rain or submit an animation to GL. Import the exact
`timing.mjs` bytes through a data URL. For each real requestAnimationFrame callback,
record its timestamp and whether the unchanged 60fps gate selects it. Discard
five seconds of warmup. Continue until both raw and selected callback streams
span at least 60 actual seconds. Retain every timestamp, summary, browser error,
source hash and failed attempt. Existing output files cannot be overwritten.

Report average callback cadence, mean/P95/maximum interval, and selected versus
raw counts for each repetition. Recompute summaries from raw timestamps and
compare the selected stream with the primary rain results. The control has no
pass threshold, renderer-quality claim, or GPU-presentation measurement. Similar
blank-page and rain cadence supports a browser/host cadence explanation in this
environment; it does not prove the cause or turn the primary result into a pass.

Run with `node screensaver/svg-preview/raf-control.mjs NEW_OUTPUT_JSON REPETITION`.
The frozen plan binds this protocol and all control source files before sampling.
Outputs are `benchmarks/results/glyph_rain_regl_20260907_cadence_r1.json`, `r2.json`
and `r3.json` with the same full prefix. The later ten-minute renderer soak is
separate and may run alongside correctness tests; it is not a pacing sample.
