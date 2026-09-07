# Glyph Rain: style measurements and implementation plan

Status: the original 192-glyph SVG workflow is complete. The web explorer now
adapts the MIT-licensed m8e/Rezmason renderer and base artwork. Its
[renderer interaction checkpoint](../benchmarks/partitions/glyph_rain_reference_v1/pixel-preview-review.json)
passed before the final borderless styling.
Performance and quantified reference parity remain unmeasured. The
[explorer guide](../screensaver/svg-preview/README.md) documents this adaptation;
the [generation benchmark](../benchmarks/svg_glyph_benchmark.md) still measures
only Smythe's original catalog. Native downloads now render both current SVG
catalogs with their three-layer native renderers; web exposure and controls
remain separate porting work.

Build new, independently drawn SVG glyphs with the proportions, weight,
terminals, negative space, and visual rhythm distilled from the reference
sheet. Introduce them occasionally within the licensed classic base catalog.
The default effect is a fixed 2D grid; a selectable 3D preset supports arrow-key
travel. **Matrix green** is the default color grade; the pinned **Reference**
palette remains selectable. The original drawing brief and measured generation workflow remain
independent of the imported runtime artwork.

## Sources and measurement boundaries

The primary shape reference is column A of the
[Unofficial Matrix glyph database](https://docs.google.com/spreadsheets/d/1NRJP88EzQlj_ghBbtjkGi-NbluZzlWpAqVIAq1MDGJc/edit?gid=1663791447#gid=1663791447),
tab `Sheet1`, `gid=1663791447`, inspected on 7 September 2026. Rows 1–2 contain
the introduction and headings. **A3:A139 contains 137 image formulas.** Column E
marks 56 entries as appearing in the trilogy (`55 Y`, `1 Y*`) and 81 as `N`.
Column B contains alternate drawings and is outside the requested measurement
set. Preserve unknown labels; a question mark in the sheet is not an invitation
to invent an origin or character name.

The effect reference is [m8e/matrix-rain at commit
5ba9049](https://github.com/m8e/matrix-rain/tree/5ba90490453ceceb6812d6b1bc658a99a92411d0),
a fork of [Rezmason/matrix](https://github.com/Rezmason/matrix). Credit to
Rezmason and the project's contributors for the reference implementation.
Its [MIT license](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/LICENSE)
credits Copyright (c) 2018 Rezmason. Study its defaults, simulation structure, vector
sampling, bloom, palette, and depth projection. The sheet determines shape
characteristics; the repository explains the moving-light effect.
[Screensaver credits](../screensaver/README.md#credits-and-references).

Every number below is labeled as one of:

- **Measured:** extracted from source images or counted in source data.
- **Reference default:** a setting in the pinned project, with its original units.
- **Derived:** calculated from source settings, with the assumptions stated.
- **Proposed target:** a starting specification for Smythe, to verify during implementation.

The current preview includes adapted REGL rain, bloom, and palette code and
the reference's classic base artwork. The pinned MIT notice accompanies the
[engine](../screensaver/svg-preview/engine/LICENSE) and
[artwork](../screensaver/svg-preview/reference/LICENSE); the
[artwork provenance](../screensaver/svg-preview/reference/provenance.json)
records the original atlas and extracted curves. These licensed imports are
separate from the 192 original glyphs and excluded from their generation
benchmark. The earlier research-only/no-import plan is superseded.

## Glyph measurements

All **137 images** were retrieved and inspected, without subsampling. There
were no failed downloads, blank images, or duplicates by file hash, decoded
pixels, or tight-cropped binary silhouette. **136 images are 128 × 128 pixels;
A34 is 129 × 128.** Measurements use the actual dimensions of each image.

The [per-row measurements](data/glyph-style-measurements.csv),
[distribution summary](data/glyph-style-summary.json), and
[method record](data/glyph-style-method.json) contain numeric evidence and
source hashes. The 137 spreadsheet PNGs and their research contact sheets are
not included in the runtime. The separately licensed classic atlas imported
from the pinned repository is a different source population.

### Proportion, weight, and empty space

These are medians across each population. “Classic” and “new-only” below are
the sheet's column-E flags, not an independent historical classification.

| Measured property | All 137 | Classic 56 | New-only 81 |
|---|---:|---:|---:|
| Ink bounding width | 82 px | 84 px | 82 px |
| Ink bounding height | 92 px | 92 px | 92 px |
| Width / height | 0.923 | 0.935 | 0.913 |
| Cell width occupied | 64.06% | 65.63% | 64.06% |
| Cell height occupied | 71.88% | 71.88% | 71.88% |
| Cell area covered by ink | 24.18% | 25.49% | 23.88% |
| Tight bounding box covered by ink | 57.48% | 57.52% | 57.47% |
| Centerline width proxy | 18.00 px | 18.44 px | 17.09 px |
| Width proxy / cell height | 14.06% | 14.41% | 13.35% |
| Left/right reflection overlap (IoU) | 0.560 | 0.601 | 0.534 |
| Top/bottom reflection overlap (IoU) | 0.495 | 0.535 | 0.487 |

The classic interquartile bands are **0.870–1.000** for bounding aspect ratio,
**18.56–28.07%** for cell ink coverage, and **14.06–16.53%** for normalized
width proxy. New-only bands are **0.792–0.989**, **19.23–27.72%**, and
**12.50–14.06%**, respectively. The combined width-proxy P10–P90 band is
**10.94–17.19%** of cell height. This is a heavy silhouette with substantial
surrounding air, not a thin monoline drawing enlarged to fill its cell.

Aspect ratios span **0.239–6.429** because narrow marks and horizontal
punctuation are intentional. For example, A6/A28/A29 have ratios
0.352/0.260/0.239; A33/A35/A70 have ratios 6.429/2.182/4.556. Keep these
families distinct. Normalizing every glyph to the same tight bounds would
erase the reference's scale and spacing hierarchy.

Classic median left and right padding is **17.19% per edge**; new-only is
**17.97%**. Both groups have **14.06%** median top and bottom padding. Each
edge is measured independently. These values describe cell spacing, not a
requirement to center every asymmetric shape mathematically.

For multi-component glyphs, the nearest ink-pixel-center distance between
components has a median of **11.5 px classic (n=16)** and **9 px new-only
(n=25)**. Their P10–P90 bands are **7.81–27.35 px** and **6.91–13.69 px**.
This spacing proxy includes boundary-pixel centers; it is not the exact white
gap or the space between rain columns.

For glyphs with enclosed holes, the median white-region width proxy is
**14 px classic (n=6)** and **12 px new-only (n=35)**. P10–P90 bands are
**5.41–25 px** and **8–21.42 px**. This measures enclosed geometric voids,
including acute cutouts, without classifying their meaning. The per-glyph
stroke-width P90/P10 ratio has a median of **1.79 classic** and **1.86 new-only**;
junctions and tapers contribute, so this is not a prescribed font-stem ratio.

### Components, counters, direction, and symmetry

| Measured count | Classic | New-only |
|---|---:|---:|
| One connected component | 40/56 (71.4%) | 56/81 (69.1%) |
| Two connected components | 13/56 (23.2%) | 18/81 (22.2%) |
| Three connected components | 3/56 (5.4%) | 7/81 (8.6%) |
| No enclosed hole | 50/56 (89.3%) | 46/81 (56.8%) |
| One enclosed hole | 2/56 (3.6%) | 29/81 (35.8%) |
| Two enclosed holes | 4/56 (7.1%) | 5/81 (6.2%) |
| Three enclosed holes | 0 | 1/81 (1.2%) |

The newer set is much more counter-rich: **43.2%** contains enclosed holes,
versus **10.7%** of the classic set. Multi-component frequency is similar:
30.9% versus 28.6%. Treat enclosure and disconnected marks as independent
design controls, rather than one “complexity” setting.

Local centerline directions, pooled over qualifying samples:

| Direction window | Classic: 10,046 samples | New-only: 15,587 samples |
|---|---:|---:|
| Horizontal ±10° | 40.41% | 23.87% |
| Vertical ±10° | 21.89% | 29.20% |
| 45° / 135° diagonals ±12° | 12.84% | 18.09% |
| Other directions | 24.86% | 28.84% |

The classic set is 62.30% horizontal/vertical by this proxy; the new-only set
is 53.07%. “Other” includes curves and other straight angles; it is not a
measured curve percentage. Only **17/56 classic** and **12/81 new-only** shapes
reach mirror IoU ≥0.90 on either axis. Asymmetry is a defining trait.

Visual inspection of every specimen supports these concrete drawing rules:

- Use broad flat-ended bars, rectangular stems, oblique cuts, and controlled
  curved hooks in the classic profile. A smooth curve can join a square bar;
  a global rounded-cap treatment would remove that distinction.
- Use rounded shoulders, enclosed bowls, inward curls, bulb-like terminals,
  and linked loops in the expanded profile. Keep corners and terminal shape
  independently selectable from stroke direction.
- Preserve detached marks and their gaps. A74/A75/A76/A77/A78/A79/A81/A84
  contain checked examples of detached angled marks; six have two components
  and two have three. This is an example set, not a complete terminal census.
- Keep punctuation small within the common cell. A dot, paired bars, and a
  narrow upright contribute spacing and rhythm rather than filling space.
- Do not invent an “exact corner radius” from an antialiased raster. Selected
  rounded and cut terminals need explicit optical review at small and large
  sizes; their radius remains an authoring target until original SVGs exist.

### Visual families and the 192-glyph brief

The following are **observational art-direction groups**, not inferred script
labels from merged cells. The ranges cover every reference row. Generation
quotas are **proposed**, rounded to preserve the sample's family balance.

| Visual family | Reference rows | Observed count | New original glyphs |
|---|---|---:|---:|
| Numeral/operator forms and compact marks | A3:A34 | 32 | 45 |
| Horizontal bars, stems, and hooks | A35:A70 | 36 | 50 |
| Stacked detached marks | A71:A72 | 2 | 3 |
| Diagonal and lozenge forms | A73:A84 | 12 | 17 |
| Roofed curves | A85:A94 | 10 | 14 |
| Rounded loops and interlocking forms | A95:A122 | 28 | 39 |
| Mixed asymmetric structures | A123:A139 | 17 | 24 |
| Total | A3:A139 | 137 | 192 |

Across the new set, start with **135 one-component, 43 two-component, and 14
three-component glyphs**. Independently target **135 with no enclosed hole,
43 with one, 13 with two, and one with three**. These quotas approximate the
whole-sheet frequencies; they are not instructions to copy individual rows.
Keep classic-like and expanded-like family profiles separately measurable.

### Measurement method and precision

Composite transparency onto white; threshold grayscale at **<128/255**.
Measure tight ink bounds and area before glow. Count 8-connected ink components
with area ≥4 pixels. Detect enclosed background with a four-neighbor flood
fill, then count 4-connected enclosed regions with area ≥4 pixels.
Retain raw counts and threshold checks in the measurement record.

The width proxy is **twice the Euclidean distance to background along a
Zhang–Suen thinned centerline**. Preserve a medial point if thinning collapses
a compact component. Junctions, dots, corners, and tapers affect this statistic;
it is not a vector font's exact stem width. Direction uses local PCA within a
five-pixel radius, at least four samples, major eigenvalue ≥1, and an eigenvalue
ratio ≥4. Mirror IoU compares tight crops without interpreting character meaning.

Thresholds **96 and 160** preserve every image's component and hole counts.
Changing from 96 to 160 changes ink area by a median **2.98%** and a maximum
**9.99%**, relative to the 128 mask. Treat fine dimensions as raster estimates.
Use normalized distributions and optical review together; no single scalar
can prove an exact style match.

## Turning the measurements into original SVGs

The new catalog retains **192 original Smythe glyph IDs**. Historical
benchmark artifacts remain immutable; the new display catalog receives its own
version, hashes, and visual acceptance record. The 137 reference images are a
measurement set, not a list of drawings to reproduce.

### Shape grammar

Use filled outlines with deliberate weight. Each glyph should combine a small
number of recognizable writing gestures: a stem, crossbar, bowl, hook, diagonal,
detached mark, or open counter. The arrangement should suggest a character
without copying a reference character's complete silhouette.

Match distributions, not just the average glyph. A set containing only wide,
heavy, densely connected symbols will lose the narrow numerals, open forms,
punctuation, and dark gaps that give the reference its rhythm. Keep dense and
sparse shapes in distinct families. Preserve the reference's restrained use of
curves and asymmetric details instead of applying random distortion to every
stroke.

Proposed authoring rules:

| Property | Rule |
|---|---|
| Source format | One self-contained SVG per original glyph; explicit `viewBox`; no embedded raster images, font dependencies, scripts, filters, or external URLs. |
| Normalization | Use a common 100 × 100 coordinate system for new display assets. Preserve each glyph's internal proportions; never stretch a narrow form to fill the cell. |
| Geometry | Filled contours with explicit winding. Cubic curves are allowed where the family calls for a bowl, hook, or rounded transition. |
| Weight | Select the family's measured stroke distribution; vary structure and terminals deliberately, without thin calligraphic strokes or uniformly rounded line caps. |
| Negative space | Match measured counter and gap distributions. An opening must remain visible at the 16-pixel review size. |
| Handedness | Design asymmetry into the original shape. Do not mirror the entire reference atlas or reproduce its glyph order. |
| Detail budget | Prefer simple structural gestures. Reject tiny ornamental cuts that disappear at 16 pixels, excessive disconnected dots, and dense microtext. |
| Identity | A new glyph needs a distinct silhouette and internal structure; a changed filename, reflection, or small affine transform is not a new design. |
| Lighting | Store unlit geometry only. Color, bloom, cursor intensity, and depth attenuation belong to the renderer. |

### Generation sequence

1. **Calibrate 24 original glyphs.** Cover narrow, wide, sparse, dense, angular,
   curved, open-counter, closed-counter, and detached-mark families. View them
   at 16, 32, 64, and 128 pixels before expanding the catalog.
2. **Complete eight batches of 24.** The accepted calibration set is the first
   batch; generate seven more. Supply the numeric family brief and original
   geometric primitives. Do not supply traced paths, imported atlases, or a
   request to reconstruct a specific reference row. Keep per-glyph generation
   records and rejected candidates.
3. **Validate geometry first.** Require well-formed self-contained SVGs, finite
   coordinates, visible ink, valid bounds, and no external resources. Render
   every accepted file at all four review sizes.
4. **Validate the completed catalog.** Assign each glyph a classic-like or
   expanded-like profile before generation. Compare each profile's continuous
   metrics with its corresponding 56- or 81-image reference population:
   target the median inside the reference interquartile range and at least 80%
   of values inside its P10–P90 band. Compare component and hole frequencies
   within 10 percentage points. Evaluate these gates across the completed
   profile, not each small batch. The seven visual families guide composition
   and optical review; the two stacked-mark examples do not support percentile
   gates. These are the acceptance criteria implemented in
   [the catalog evaluator](../benchmarks/svg_glyph_measurements.py).
5. **Check distinctness.** Reject identical normalized masks. Flag near matches
   for review using intersection-over-union after translation and uniform-scale
   alignment, including a mirrored comparison. Start with an IoU flag at 0.85;
   simple marks require optical review because overlap alone cannot establish
   originality or duplication.
6. **Review the ensemble.** At normal playback size, the set should read as one
   writing system. Revise outliers that look like logos, emoji, a generic font,
   decorative runes, or random line piles. Use the shape brief and contact sheet,
   not a claimed universal numeric style score.
7. **Freeze and export.** Commit only accepted original SVGs, the catalog
   manifest, generator version, measurements, and receipts. Generate native
   path data from these SVGs; test all exports against the same source geometry.

### Current catalog and acceptance

The implemented catalog and its acceptance receipts are available as the
[complete contact sheet](../benchmarks/partitions/glyph_svg_v1/catalog/contact-sheet.png),
[numeric evaluation](../benchmarks/partitions/glyph_svg_v1/catalog/style-acceptance.json),
[optical review](../benchmarks/partitions/glyph_svg_v1/optical-review.json), and
[size audit](../benchmarks/partitions/glyph_svg_v1/size-review.json).
The [calibration history](../benchmarks/partitions/glyph_svg_v1/calibration-history.json)
retains the measured failures from earlier candidates. Design calibration was
completed before the timed campaign; its total elapsed time was not recorded.

The final catalog uses 78 classic-like and 114 expanded-like glyphs. Its
19 normalized shape metrics pass each profile's median and distribution gates;
component and hole frequencies pass their separate gates. All 18,336 aligned
and reflected pair comparisons fall below the 0.85 near-match threshold.
Thresholds 96/128/160 preserve topology. At 16/32/64/128 pixels, no glyph is
blank or loses a component or counter. Nine glyphs at 16 pixels and two at
64 pixels retain extra 1–3-pixel enclosed specks; the size audit lists them.

Raster edge sensitivity is reported separately: both profiles fall outside
the reference's antialiasing-area band. It depends on the rasterizer and is
not a geometry acceptance gate. Numeric matching and AI optical review do not
establish exact stylistic equivalence or independently prove originality.

## Reference effect: measured structure and defaults

The pinned repository's classic SVG atlas is **512 × 512**, with an **8 × 8**
grid of **64 × 64** cells. Its sequence contains **56 visible shapes plus one
blank**, unlike the sheet's 137 nonblank images. The classic atlas has 84
contours and 183 cubic segments after expanding shorthand curve commands.
Its median visible bounds occupy 70.47% of cell width and 74.71% of height.
These atlas measurements are a separate population; use the sheet measurements
above for the new shape brief.
[Atlas](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/svg%20sources/texture_simplified.svg).

### Motion and grid

| Reference parameter | Classic default | `3d` preset |
|---|---:|---:|
| Logical grid at density 1 | 80 × 80 | 80 × 80 |
| Glyph aspect / vertical spacing multipliers | 1 / 1 | 1 / 1 |
| Global animation speed | 1 | 1 |
| Fall-speed parameter | 0.3 | 0.5 |
| Per-column speed multiplier | 0.5–1.0 | 0.5–1.0 |
| Rain-length parameter | 0.75 | 0.30 |
| Symbol age increment per tick | 0.03 | 0.03 |
| Symbol-cycle frame skip | 1 | 1 |
| Extra glyph flip / rotation | None / 0° | None / 0° |
| Initial field | Already populated | Already populated |

[Configuration](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/config.js#L83).

The reference separates fixed cell positions, illumination, and symbol identity.
A periodic light ramp travels down each column, with independent phase and
speed. A smooth perturbation varies the spacing between bright runs without
reversing their order. A discontinuity marks the leading cell. Character
substitutions use a separate age counter.
[Rain simulation](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.raindrop.frag.glsl#L40).

**Derived, ignoring the spacing perturbation:** classic defaults imply 15–30
cells/second, a 75-cell repeat distance, and a 2.5–5-second cycle. The `3d`
preset implies 25–50 cells/second, 30 cells, and 0.6–1.2 seconds. With body
brightness clipping but without bloom, approximately 40.9 classic cells or 12
`3d` cells remain visibly lit per underlying period. These are mathematical
scales, not constant tail lengths measured from playback.

At 60 ticks/second, the 0.03 symbol-age increment yields about **1.8 selections
per second per cell**. The source counter advances by ticks, so that value is
not refresh-rate independent. Smythe must use elapsed time.
[Symbol simulation](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.symbol.frag.glsl#L31).

### Light and color

This section records the pinned upstream defaults. Smythe's deliberate Matrix
green color-grade adaptation is specified under [Current renderer adaptation](#current-renderer-adaptation).

| Reference parameter | Classic | `3d` |
|---|---:|---:|
| Body contrast | 1.10 | 1.50 |
| Body brightness offset | −0.50 | −0.90 |
| Cursor intensity | 2.0 | 2.0 |
| Bloom strength | 0.70 | 0.70 |
| Bloom high-pass threshold | 0.10 | 0.10 |
| Bloom pyramid levels | 5 | 5 |
| Bloom scale per level | 40%, 20%, 10%, 5%, 2.5% of render width/height | Same |
| Dither parameter | 0.05 | 0.05 |

Bloom acts on brightness before the final color mapping. Each level uses a
high-pass step and horizontal/vertical blur. The three-tap blur weights are
0.279, 0.442, and 0.279. Five differently weighted levels are combined; the
strength setting is not a simple alpha-opacity percentage. Draw a sharp core
as well as the halo, so counters remain open.
[Bloom stages](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/bloomPass.js#L6) ·
[Blur kernel](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/bloomPass.blur.frag.glsl#L10).

The classic body palette uses **108° hue, 90% saturation**, and lightness stops
of 0%, 20%, 70%, and 80%. Approximate colors before rendering are `#000000`,
`#176105`, `#89F76E`, and `#B0FA9E`. The independent cursor color is
HSL(87.12°, 100%, 73%), approximately `#C1FF75`, before its intensity multiplier.
It is pale yellow-green, not uniformly white. The palette contains 2,048
interpolated entries. The final dither subtracts less than 0.01667 brightness
units; the configured 0.05 is divided by three.
[Palette construction](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/palettePass.js#L9) ·
[Final mapping](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/palettePass.frag.glsl#L23).

### Depth and rendering cost

The reference's volumetric projection uses **90° vertical field of view**, near
plane **0.0001**, and far plane **1000**. Per-column depth repeats over a
normalized interval; the default forward-speed parameter 0.25 produces a
**derived four-second wrap period**. This travel is automatic. The inspected
project does not implement arrow-key camera navigation; its `camera.js` is for
webcam input.
[Projection](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/rainPass.js#L219) ·
[Depth mapping](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.vert.glsl#L29).

The SVG source becomes a 512 × 512 multichannel distance-field texture with a
four-pixel distance range. The shader recovers smooth edges using derivatives.
This explains scalable appearance; importing a low-resolution bitmap would
not reproduce that behavior.
[Distance-field generation](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/assets/msdf_command.txt#L1).

At density 1, the volumetric grid contains **6,400 quads, 12,800 triangles, and
38,400 vertices**. The normal pipeline has approximately **23 draw passes**,
including four state updates, the glyph pass, fifteen bloom-pyramid passes,
bloom combination, palette, and final copy. The render-resolution default is
0.75 × device-pixel ratio in each dimension: **1440 × 810** for a 1920 × 1080
viewport at DPR 1, or **2880 × 1620** at DPR 2. The configured **60 FPS** is a
target; this study did not measure upstream FPS, power draw, or total GPU memory.
[Geometry](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/rainPass.js#L26) ·
[Frame sizing and scheduling](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/main.js#L40).

## Current renderer adaptation

The current web preview adopts the pinned MIT-licensed REGL rain, bloom,
and palette passes. It replaces the earlier independently implemented Canvas
interpretation. The [behavior plan](glyph-rain-parity-plan.md) contains the
parameter-level comparison and acceptance criteria; the
[explorer guide](../screensaver/svg-preview/README.md) is the current control map.
Browser interaction checks pass. They cover presets, mix endpoints, settings,
paused navigation/reset, fullscreen, input release, mobile layout, reduced
motion, and hidden-tab suspension. New performance measurements and quantified
reference-parity checks remain pending.

| Setting | Current preview configuration |
|---|---|
| Default scene | Classic fixed 2D grid; 80 cells across the longer viewport dimension |
| Other presets | 3D with deliberate arrow-key travel; Operator visual preset |
| Base artwork | 56 licensed visible glyphs and the source sequence's blank slot |
| Original artwork | 192 independently generated SVGs, with their original IDs and hashes |
| Mix | Original catalog selected with 10% probability by default; otherwise select from the 57 base slots |
| Motion and light | Reference state simulation, separate glyph changes, scalar bloom, then palette mapping |
| Default color grade | Matrix green: 137° hue, 80% saturation, existing preset lightness stops, cursor `#A2FFD8` |
| Reference colors | Selectable; retain the chosen preset's original palette and cursor color |
| Render scale | Default 0.75 × viewport size × device-pixel ratio |
| Settings | Supported options only; draft changes apply together and are reflected in the URL |

The catalogs have different roles. Imported base artwork establishes the
reference appearance. The original shapes remain an independent measured
artifact workflow and appear occasionally in the default effect. A display
mix is not a new generation result, a claim of original authorship for the
base shapes, or evidence of rendering efficiency.

The default 57-slot base sequence includes an intentional blank. Preserve that
slot when weighting the base catalog; do not select uniformly across a joined
56+192 array. The original percentage controls catalog selection independently
of catalog size.

### Matrix green color grade

The default grade moves the body green to **137° hue and 80% saturation** while
preserving each preset's lightness values and stop positions. Classic retains
stops 0, 0.2, 0.7, and 0.8: approximately `#000000`, `#0A5C21`, `#75F098`, and
`#A3F5BA`. The cursor is mint **`#A2FFD8`**. This is a deliberate adaptation,
not an assertion that these are the reference repository's unmodified defaults.
The **Reference** palette restores the original preset colors, including its
cursor; the source audit above remains unchanged.

The [color comparison receipt](../benchmarks/partitions/glyph_rain_reference_v1/color-comparison.json)
measures the user's screenshot separately from the earlier preview. Excluding
the UI (`y < 685`), median body hue is **136.92° versus 108.75°**. A matched
mid-green mask gives 136.80° versus 108.51°. The bright-tip medians also support
mint-white highlights rather than the previous yellow-white. The receipt records
source paths and hashes, pixel masks, sample counts, and distributions; the
user's image is not copied into the repository.

Hue is measured; saturation is an art-direction choice. The existing HSL
lightness curve stays in place, while RGB luminance changes with the new
chroma. A new rendered comparison must verify appearance after bloom and
clipping. Use Reference when testing upstream color parity and Matrix green
when reviewing the requested grade.

### Archived renderer v1

The earlier Canvas implementation used 420 columns in a 140 × 80 × 60 world,
70 cells per column, direct vector close-ups, and a bounded sprite cache.
Its [source and screenshot archive](../benchmarks/partitions/glyph_svg_v1/renderer-v1/)
resolves the hashes in its original receipts.

Its three headless timing runs are superseded diagnostics: they recorded
36.04–53.81 average callbacks/s and missed the 16.7 ms P95 interval target.
The 601.12-second movement/resize soak and twelve-view darkness sample describe
that implementation only. The current REGL renderer requires new checks and
receipts; none of the old frame rates, cache bounds, or visual samples transfer.
The original SVG generation benchmark remains valid because its measured
input, output, and algorithm are unchanged.

## Controls and platform behavior

### Menu visual brief

Use simple VT323 pixel lettering for headings, labels, numerals, buttons,
inputs, and help text. The SMYTHE logo uses outlined Trajan Pro Bold and generous
black padding. Align labels and values without control frames. Interface text
and panels have no glow, corner ornaments, or stepped edges.
Follow the [explorer style](style.md#explorer-controls) and
[hottelet.com](https://www.hottelet.com/): primary green `#37FF6E`, bright
`#9CFFBC`, and black. The rain keeps its separate 137° grade, `#A2FFD8` leading
glyphs, and bloom. Chart typography and the black-and-white palette remain
unchanged.

Verify readable pixel lettering, borderless controls, a distinct focus indicator,
and touch targets of at least 44px. Focus, selection, and disabled states use
clear shapes and contrast. Verify the sheet at 390px and 1920px widths,
including scrolling, draft edits, Apply, Cancel/Escape, and restored focus.
These are visual acceptance targets, not measurements of the earlier menu.

### Input and native hosts

| Input | Current web control contract |
|---|---|
| S | Open settings; release held movement before focus enters the sheet |
| Space | Pause/resume playback |
| R | Reset the viewpoint |
| F | Toggle fullscreen |
| ↑ / ↓ in 3D | Move forward / backward |
| ← / → in 3D | Move sideways with perspective parallax |
| Escape in settings | Cancel the draft, close, and restore focus |
| Apply | Apply the entire settings draft, then update the URL |

Verify reduced-motion startup with no automatic travel, deliberate navigation,
keyboard focus, and settings cancellation. Held movement must clear on key
release, focus loss, hidden tabs, and pointer cancellation. Resizing must not
leave stale render buffers or an invalid camera. Presets and settings must
expose only capabilities the active engine implements.

Windows `/s` remains a normal screensaver; `/p` remains the settings preview.
Navigation belongs in an explicit `/w` explorer so normal input dismissal is
preserved. macOS needs a companion universal Explorer app using the same view;
the system screensaver host owns input dismissal. Linux exploration belongs
in a standalone X11 window; embedded/root modes leave keyboard ownership with
the screensaver manager. Native Wayland and Mac notarization remain separate work.
Current compiled downloads use the exact filled SVG catalogs and default
90% reference / 10% original selection, while retaining their native controls.
The [native catalog record](../screensaver/native-catalog.json) identifies both
sources and the [build receipts](../screensaver/README.md#native-verification)
record compiled glyph and host verification.

## Delivery and acceptance

The original 192-glyph generation and acceptance campaign is complete. The
remaining renderer work is:

1. Verify the adapted Classic, 3D, and Operator configurations against the
   pinned behavior, with explicit license notices and source provenance.
2. Verify mix endpoints and the default 10% weighting, including the base blank
   slot; retain independent provenance for both catalogs.
3. Test settings as a transaction: draft edits do not reset the effect, Apply
   commits supported values, Cancel/Escape restore focus, and URLs round-trip.
4. Test actual browser navigation, pause, reset, fullscreen, reduced motion,
   mobile layout, resize, hidden tabs, focus loss, and clean shutdown.
5. Capture the adapted renderer's actual output before replacing its public
   screenshot. Keep the earlier image with the v1 archive.
6. Run a new performance campaign and movement/resize soak against frozen
   source, catalog, and settings hashes.
7. Both catalog exports now ship on Windows, macOS, and Linux with compiled
   rendering checks. Next, port the web exposure pipeline and explorer controls,
   preserve OS screensaver policy, and verify the new compiled behavior.

Proposed measurement gates remain targets, not current achievements:

- Web: target 60 FPS at 1920 × 1080, with P95 completed-frame interval ≤16.7 ms
  after warmup. Declare render scale and DPR; a 0.75-scale buffer is not a
  full-resolution pixel workload.
- Native: target at least 40 FPS at 1280 × 720, with P95 interval ≤25 ms.
- Report CPU command submission separately from callback intervals. GPU
  completion, physical display presentation, total graphics memory, and power
  require their own measurements.
- Measure three 60-second runs after five seconds of warmup each. Keep every
  run, failure, device/OS/browser identifier, setting, and raw sample.
- Complete a ten-minute travel/resize soak with finite state, stable resources,
  stopped hidden/paused work, and released input. Scope any concurrent workload.

Native regression acceptance includes proportional depth/parallax, stable
column identities during repeated travel, correct pause/reset behavior, matching
SVG contours after export, and normal screensaver dismissal. Windows must load
and render the compiled `.scr` and exercise actual `/p` subprocess embedding.
Both Apple Silicon and Intel must execute the same universal artifact. Ubuntu
22.04 and 24.04 must execute the same Linux ELF with rendering, embedding,
resize, input, and shutdown checks. Current catalog and host receipts are linked
above; explorer controls and performance targets require their own new evidence.

Update screenshots, documentation, checksums, source provenance, and required
MIT notices together. Run the full offline suite, Ruff, and local Markdown link
validation before publication. Keep the README centered on generated execution
topology and its durable envelope; the mixed-artwork display remains separate
from the original-glyph workflow evidence.
