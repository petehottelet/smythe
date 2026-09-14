# Glyph Rain web explorer

The explorer adapts the MIT-licensed rain, bloom, and palette renderer from
[m8e/matrix-rain](https://github.com/m8e/matrix-rain) and
[Rezmason/matrix](https://github.com/Rezmason/matrix). Classic starts with a fixed
2D grid. The reference's base glyphs carry the effect; Smythe's original SVGs
appear in 10% of selections by default.

The [current browser review](../glyph-design-v2/browser-review.json) checks the
v2 catalog, all presets, mix endpoints, navigation, reset, and interruption.
It binds the current [still image](preview.png). Earlier performance receipts
remain historical evidence for their recorded renderer and v1 catalog.

The separate [192/256-glyph workflow benchmark](../../benchmarks/svg_v2_results.md)
measures the current authoring method and memory use. The live viewer still
uses 192 originals; the extra 64 remain in the benchmark contact sheets.

## README animation

[Animated GIF](preview.gif) · [Still image](preview.png) · [Capture record](preview-animation.json).

Both current previews show **100% of selections from the new 192-glyph Smythe
catalog**, using the actual Classic renderer and Matrix palette. The GIF is
900×506 pixels and loops at 20 playback frames per second. A short dissolve
joins the loop. Frame timing is an export setting, not a performance result.

## Run and controls

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open [the explorer](http://localhost:8000/screensaver/svg-preview/) or
[the new 192-glyph catalog](http://localhost:8000/screensaver/svg-preview/catalog.html).
The static preview requires a WebGL-capable browser, with no API key or build step.

| Input | Action |
|---|---|
| S / Settings | Open the settings sheet |
| Space | Pause or resume playback |
| R | Reset the viewpoint |
| F | Toggle fullscreen |
| Arrow up / down in 3D | Move forward / backward |
| Arrow left / right in 3D | Move sideways |
| Escape in settings | Discard the draft, close the sheet, and restore focus |

Settings provide Classic, 3D, and Operator presets. Classic is the default
2D effect; navigation belongs to the 3D preset. Adjust the original-glyph mix,
columns, motion, glow, render scale, glyph transforms, travel, and supported
colors as one draft. **Apply** commits the changes together and updates the
URL; Cancel leaves the running configuration intact. Only options supported
by the selected engine are exposed.

The default **Matrix green** grade uses a 137-degree body hue and mint
`#A2FFD8` leading glyphs. **Reference colors** retains the original body
palette; the leading glyph color is independently adjustable. The
[color comparison](../../benchmarks/partitions/glyph_rain_reference_v1/color-comparison.json)
records the supplied screenshot's sampled 136.92-degree body hue and the
previous preview's 108.75-degree hue. Each preset keeps its exposure curve.

VT323 gives the interface its simple pixel lettering. The SMYTHE logo uses
outlined Trajan Pro Bold, with generous black padding. Borderless text controls use
primary green `#37FF6E`, and bright `#9CFFBC`, following
[hottelet.com](https://www.hottelet.com/). Interface text and panels have no
glow or corner ornaments. These interface colors are separate from the rain's
137° body hue and `#A2FFD8` highlights.
[Font and logo provenance](fonts/provenance.json) records the bundled VT323
font and its OFL license; the Trajan font file is not bundled.
The [house style](../../docs/style.md#explorer-controls) preserves the README
charts' black-and-white palette and existing typography.

Browser checks cover all three presets, the default 10% mix and 0%/100%
endpoints, URL reload, batched Apply, Cancel/Escape focus restoration, paused
3D navigation and exact reset, Space, fullscreen, pointer/focus release, and
reduced-motion startup. A real tab switch preserved the frame count and rain
time while hidden. At 390 × 844, controls and the settings sheet remain inside
the viewport, with no horizontal overflow. Travel controls are disabled outside
the 3D preset and retain the correct state after Apply and reopen.

## Two catalogs, separate provenance

The licensed base catalog contains **56 visible classic glyphs plus the
reference's blank selection slot**. Each glyph selection draws from the
original catalog with a default probability of 10%; other selections use the
57 reference slots. The mix is weighted by catalog choice, so the larger
original catalog does not dominate the effect. The setting is adjustable
from 0% to 100%.

[Base SVGs and source](reference/README.md) ·
[Artwork provenance](reference/provenance.json) ·
[Artwork MIT notice](reference/LICENSE) ·
[Engine MIT notice](engine/LICENSE).

The imports are pinned to
[5ba9049](https://github.com/m8e/matrix-rain/tree/5ba90490453ceceb6812d6b1bc658a99a92411d0).
Credit to Rezmason and the reference project's contributors. Smythe adapts
licensed renderer code and base artwork; its additional 192 glyphs are
independently authored.

[New 192-glyph sheet](../glyph-design-v2/contact-sheet-128.png) ·
[16 px](../glyph-design-v2/contact-sheet-16.png) ·
[32 px](../glyph-design-v2/contact-sheet-32.png) ·
[64 px](../glyph-design-v2/contact-sheet-64.png) ·
[SVG files and manifest](../glyph-design-v2/README.md).

The gallery shows only these new original glyphs. The live effect keeps its
adjustable reference mix; use **Original glyphs: 100%** to see only v2.
The [4.03-second generation result](../../benchmarks/svg_glyph_benchmark.md)
belongs to the historical v1 workflow. The [current v2 report](../../benchmarks/svg_v2_results.md)
measures compilation, validation, and export of authored contours at 192 and
256 glyphs. It does not time the design work or a model generating new shapes.

[All current sheets, media, and benchmark reports](../../docs/current-materials.md).
The capture record retains its original source hashes. The later no-JavaScript
fallback-link correction changes no visible rain or glyph input; the
[materials check](../../docs/data/materials-review-2026-09-13.json) records that difference.

## Renderer and checks

The adapted REGL pipeline retains the reference's stationary grid cells,
traveling illumination, glyph changes, scalar bloom, and final palette mapping.
Classic uses 80 cells across the viewport's longer dimension and a default
render scale of 0.75. At 1920 × 1080 and DPR 1, those settings imply 24-pixel
cells and a 1440 × 810 drawing buffer. These are configuration-derived sizes,
not measured performance results.

The [reference provenance](reference/provenance.json) identifies the pinned
artwork. [Original-glyph acceptance](../glyph-design-v2/README.md) has its own
records; renderer checks and measurements are listed below.

Settings schema and URL checks run without a browser:

```bash
node screensaver/svg-preview/verify-settings.mjs
```

The [frozen performance protocol](../../benchmarks/renderer_performance_20260907.md)
defines three independent sessions each for Classic and 3D: five seconds of
warmup followed by 60 seconds of samples at a 1080p viewport and 0.75 render
scale. The helper binds source, catalogs, browser/backend, settings, and raw
samples; the aggregator recomputes every timing summary. Callback intervals
and CPU submission remain separate from GPU execution and physical presentation.
The [completed campaign and cadence control](../../benchmarks/renderer_performance_20260907_results.md)
retain all six valid sessions: 56.21–56.24 measured draws/second, with an
18.10 ms P95 callback interval in each run. Every run missed the declared
59.5 draws/second and 16.7 ms P95 targets. Three blank-page controls showed
similar source cadence. These headless, active-desktop diagnostics do not
measure physical presentation or GPU execution time.

The separate [600.95-second stability soak](../../benchmarks/renderer_performance_20260907_results.md#separate-stability-check)
passed 57 travel/resize cycles while Jobs and correctness workloads ran
concurrently. Camera bounds, catalog mix, rain-time continuity, pause, and
source identities held throughout the check.

Benchmark completion now restores exactly one animation loop. A regression
executes the renderer lifecycle through repeated measurements, pause/resume,
and timeout, rejecting duplicate scheduled callbacks. Run its offline checks:

```bash
node screensaver/svg-preview/verify-lifecycle.mjs
node screensaver/svg-preview/verify-measurement.mjs
node screensaver/svg-preview/verify-browser-metadata.mjs
node screensaver/svg-preview/verify-raf-control.mjs
```

### Archived Canvas v1 evidence

The earlier, all-original Canvas explorer and screenshot are preserved in
[renderer-v1](../../benchmarks/partitions/glyph_svg_v1/renderer-v1/). Its records
remain **superseded diagnostics** and do not describe the current REGL effect.

| V1 repetition | Average callbacks / s | P95 draw-completion interval | P95 CPU submission | Peak raw sprite cache |
|---|---:|---:|---:|---:|
| [1](../../benchmarks/results/glyph_svg_v1_renderer_r1.json) | 53.81 | 22.90 ms | 3.80 ms | 51.88 MiB |
| [2](../../benchmarks/results/glyph_svg_v1_renderer_r2.json) | 36.04 | 34.30 ms | 6.70 ms | 51.88 MiB |
| [3](../../benchmarks/results/glyph_svg_v1_renderer_r3.json) | 36.13 | 33.10 ms | 6.70 ms | 51.88 MiB |

Each old run sampled 60 seconds after five seconds of warmup at 1920 × 1080,
DPR 1, in headless Chromium on Windows build 26200 with a Ryzen 9 5950X.
None met the proposed 16.7 ms P95 interval target. The
[aggregate](../../benchmarks/results/glyph_svg_v1_renderer.json) retains every
run and the [pre-sampling helper failure](../../benchmarks/results/glyph_svg_v1_renderer_launch_diagnostic.json).

V1's [601.12-second soak](../../benchmarks/results/glyph_svg_v1_soak.json) passed
57 movement/resize cycles with stable state and a 51.88 MiB raw sprite-cache
peak. Offline tests and visual sampling ran concurrently. Its
[visual review](../../benchmarks/partitions/glyph_svg_v1/preview-review.json)
records the prior screenshot, interaction checks, and twelve declared views
with 70.15%–78.89% dark coverage. Those checks and cache figures apply only to
the archived implementation.

## Native source builds and next checks

The Windows, macOS, and Linux source ports in the
[native guide](../README.md#ports) use these same SVG outlines: 56 visible
reference glyphs, the blank slot, and 192 originals, with a 10% original mix.
Native GDI+, Core Graphics, and Cairo render the filled contours into cached
sprites. The [catalog record](../native-catalog.json) binds both source catalogs;
[historical verification](../README.md#native-verification) records earlier builds.
Precompiled packages are not distributed; the native guide lists build commands.
The native savers retain their three-layer motion. The REGL exposure pipeline,
3D exploration, and pixel settings are next for the native ports.

Next: measure reference-behavior tolerances, visible-browser/display cadence,
and GPU timing for the same preset and catalog configuration. Keep licensing,
source hashes, screenshots, and test receipts together.
