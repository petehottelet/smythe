# Glyph Rain: style measurements and implementation plan

Status: research and planned work. The current screensaver code, glyph catalog,
and compiled downloads remain unchanged by this plan.

Build new, independently drawn SVG glyphs with the proportions, weight,
terminals, negative space, and visual rhythm distilled from the reference
sheet. Place them in a persistent three-dimensional rain field that the user
can explore with the arrow keys. Match the visual grammar; create new symbols.

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
a fork of Rezmason/matrix. Study its defaults, simulation structure, vector
sampling, bloom, palette, and depth projection. The sheet determines shape
characteristics; the repository explains the moving-light effect.

Every number below is labeled as one of:

- **Measured:** extracted from source images or counted in source data.
- **Reference default:** a setting in the pinned project, with its original units.
- **Derived:** calculated from source settings, with the assumptions stated.
- **Proposed target:** a starting specification for Smythe, to verify during implementation.

No reference source code, SVG outlines, image atlas, or font file will ship in
Smythe. The MIT notice applies when copies or substantial portions of licensed
work are included. Studying methods and creating independent implementations
does not itself require adding that notice. Source artwork is expressive work;
tracing or translating its exact contours into a different format still copies
the drawing. Research links are retained as provenance.
[MIT terms](https://opensource.org/license/mit) ·
[Copyright Office: ideas, methods, and expression](https://www.copyright.gov/help/faq/faq-protect.html).

## Glyph measurements

All **137 images** were retrieved and inspected, without subsampling. There
were no failed downloads, blank images, or duplicates by file hash, decoded
pixels, or tight-cropped binary silhouette. **136 images are 128 × 128 pixels;
A34 is 129 × 128.** Measurements use the actual dimensions of each image.

The [per-row measurements](data/glyph-style-measurements.csv),
[distribution summary](data/glyph-style-summary.json), and
[method record](data/glyph-style-method.json) contain numeric evidence and
source hashes. Reference PNGs and contact sheets are research material and are
not included in the repository or runtime.

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

The next catalog will retain **192 original Smythe glyph IDs**. Historical
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
   gates. These are proposed acceptance criteria, not achieved measurements.
5. **Check distinctness.** Reject identical normalized masks. Flag near matches
   for review using intersection-over-union after translation and uniform-scale
   alignment, including a mirrored comparison. Start with an IoU flag at 0.85;
   simple marks require human review because overlap alone cannot establish
   originality or duplication.
6. **Review the ensemble.** At normal playback size, the set should read as one
   writing system. Revise outliers that look like logos, emoji, a generic font,
   decorative runes, or random line piles. Use the shape brief and contact sheet,
   not a claimed universal numeric style score.
7. **Freeze and export.** Commit only accepted original SVGs, the catalog
   manifest, generator version, measurements, and receipts. Generate native
   path data from these SVGs; test all exports against the same source geometry.

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

## Smythe rendering specification

Separate four systems: world geometry, glyph identity, illumination, and camera.
Moving the camera changes projection. It must not regenerate the glyph field
or reset the rain. The glyphs stay in their cells while a light front travels
down each column; sparse character substitutions run on an independent clock.

### World and camera

These are **proposed targets**, not values copied from the reference renderer.

| Parameter | Initial target | Reason / acceptance |
|---|---:|---|
| Persistent columns | 300 | Enough overlapping depth at 1920 × 1080; cull before drawing. Tune density from measured frame times and dark-space coverage. |
| World width × height × depth | 140 × 80 × 60 units | A bounded repeating volume supports continuous travel. |
| Vertical cells per column | 70 | 21,000 logical cells; only illuminated, visible cells become draws. |
| Cell pitch | 80/70 ≈ 1.143 units | Fixed world spacing, independent of viewport and refresh rate. |
| Near / far depth | 4 / 64 units | Clip and fade before a glyph crosses the camera. |
| Focal length | 0.9 × viewport height | Approximately 58.1° vertical field of view. |
| Glyph height | 0.85 world units | Leaves space between consecutive glyph cells. |
| Sideways speed | 12 world units/second | Key-hold duration controls distance. |
| Forward/backward speed | 18 world units/second | Continuous approach and recession, with no automatic forward drift. |
| Near fade interval | Depth 4–7 | Smoothly hide wrapping columns before crossing the near plane. |
| Far fade interval | Depth 46–64 | Blend recycled columns into the distance. |
| Position reset | X=0, Z=0 | Preserve glyph identities and rain phase. |

Use standard perspective: projected size is proportional to focal length divided
by camera-relative depth. At a 1080-pixel viewport, focal length is 972 pixels.
A glyph of height 0.85 projects to **82.62 pixels at depth 10**, **41.31 at 20**,
and **20.66 at 40**. A one-unit lateral camera move shifts a point **121.5 pixels
at depth 8** and **30.375 at depth 32**: the acceptance ratio is **4:1**.
These are analytical examples, not measured frame-rate results.

Track held keys independently of operating-system key repeat. Normalize diagonal
input so holding two keys does not multiply movement magnitude. Clear held keys
on focus loss, hidden tabs, pointer cancellation, and window deactivation.
Wrap positions without changing the stable column seed. Keep the same world and
camera through resize; only projection and render buffers change.

### Rain and light

| Property | Proposed starting target |
|---|---|
| Illumination speed | 16–30 cells/second per column (18.3–34.3 world units/second at the proposed pitch); independent phase. |
| Lit trail length | Usually 12–24 cells (13.7–27.4 world units), with occasional longer runs and real gaps between waves. |
| Character substitution | Independently sampled intervals of 0.35–0.9 seconds; time-based, never once per display frame. |
| Leading glyph | At most one hot cell per wave front; vary whether a column uses a pale head. Avoid a uniform row of white cursors. |
| Core and bloom | Render crisp cores separately from halos. Apply a brightness threshold before broad bloom; preserve counters and separation. |
| Palette | Black background; green bodies; yellow-green to pale-green highlights. Calibrate to the reference palette values, then judge in the complete scene. |
| Dark-space gate | Proposed 65–80% dark pixels in a paused 1080p frame, excluding UI; define dark as all RGB channels ≤16. Record several seeds and camera positions. |
| Highlight gate | Proposed <3% near-white pixels; define near-white as all RGB channels ≥210. Bright heads must remain localized. |
| Frame independence | Equivalent rain/camera state after 10 seconds at simulated 30, 60, and 144 FPS, within declared floating-point tolerance. |

The sheet contains flat glyph artwork. It cannot supply bloom radii, luminance
decay, frame rate, or motion speed. Derive those from the effect source and
playback measurements; never report them as measurements of column A.

### Vector rendering and performance

SVG supplies resolution-independent shape geometry; perspective itself does
not require SVG. Parse original paths once. Use a bounded sprite cache at several
physical pixel sizes for distant and middle-distance glyphs, with direct vector
rendering or a higher-resolution representation for close-ups. Re-rasterize at
the required size rather than enlarging a tiny bitmap.

Start with four physical-pixel cache sizes: **16, 32, 64, and 128**, with two
light roles. Fully populating 192 glyphs at all four sizes takes **31.875 MiB**
of raw RGBA pixels before padding, halos, or graphics-library overhead.
Set a proposed **64 MiB image-cache cap**, evict unused entries, and account
for render buffers and runtime overhead separately.
Do not allocate one SVG DOM element for every logical cell or rebuild paths
every frame. A future MSDF/GPU implementation is an alternative only if profiling
shows the simpler renderer misses the target. Implement its shaders independently.

Proposed performance gates:

- Web: target 60 FPS at 1920 × 1080, with P95 frame time ≤16.7 ms after warmup.
- Native: target at least 40 FPS at 1280 × 720, with P95 frame time ≤25 ms.
- Frame-time gates use intervals between completed frames, including scheduling
  delays. Report update/draw work duration separately. Browser callback timing
  alone does not prove physical presentation timing; identify the available
  measurement and report missed-frame estimates without claiming GPU timings.
- Input response: visible movement within 100 ms; key release/focus loss stops
  travel within the next rendered frame.
- Measure for 60 seconds after a 5-second warmup; report device, OS, renderer,
  resolution, device-pixel ratio, visible glyph count, median/P95 frame time,
  peak cache bytes, and allocation growth.
- Run a 10-minute travel-and-resize soak. Memory should plateau at the declared
  cache bound; hidden/minimized renderers should stop animation work.

These are targets. Publish efficiency numbers only after measuring the final
original-glyph build. Keep renderer FPS separate from the historical glyph
generation benchmark and its simulated provider latency.

## Controls and platform behavior

| Input | Behavior |
|---|---|
| ↑ / ↓ | Move forward / backward through the field. |
| ← / → | Move sideways with distance-dependent parallax. |
| Space | Pause/resume the rain; deliberate camera movement still works. |
| R | Reset the viewpoint without reshuffling glyphs or advancing paused time. |
| F | Toggle fullscreen in the web explorer. |
| Escape | Leave fullscreen or exit the native explorer according to platform convention. |
| Touch | Hold four labeled directional controls; cancel movement on release or lost pointer capture. |

Web reduced-motion mode starts with still rain and no automatic travel. A user
may deliberately move the viewpoint. Pause, reset, fullscreen, and control help
remain keyboard-accessible, with visible focus. Hide passive UI during idle
viewing; restore it on interaction or focus.

Windows `/s` remains a normal screensaver; `/p` remains the settings preview.
Navigation belongs in an explicit `/w` explorer so normal input dismissal is
preserved. macOS needs a companion universal Explorer app using the same view;
the system screensaver host owns input dismissal. Linux supports exploration
in a standalone X11 window; embedded/root modes leave keyboard ownership with
the screensaver manager. Native Wayland and Mac notarization remain separate work.

## Delivery and acceptance

1. Complete the measured shape brief and 24-glyph calibration set.
2. Generate and accept all 192 original SVGs; freeze their catalog and hashes.
3. Implement the persistent world and camera as a renderer-independent model.
4. Verify web interaction, reduced motion, touch, resize, and hidden-tab behavior.
5. Port the same geometry and interaction contract to Windows, macOS, and Linux.
6. Build and test compiled artifacts before replacing any current download.
7. Update the screenshot, documentation, checksums, and build provenance together.

Required regression evidence:

- A 4:1 parallax ratio at depths 8 and 32; projected size doubles when depth halves.
- Forward/backward movement changes depth and scale; lateral movement preserves
  depth and changes screen position. No flat-layer translation substitute.
- 10,000 movement steps remain in world bounds with stable column identities.
- With rain paused, camera reset restores the same pixels at the same viewport.
- Key release, focus loss, visibility change, and touch cancellation clear motion.
- SVG contours and holes survive all native geometry exports. Original SVGs
  remain the authoritative assets; bounded runtime rasterization caches are allowed.
- Windows compiled load/render/resize and actual `/p` subprocess embedding/exit.
- The same universal Mac artifact executes on Apple Silicon and Intel; verify
  both the screensaver view and companion Explorer app.
- The same Linux ELF passes on Ubuntu 22.04 and 24.04. Retain the 16 existing
  native checks and add the nine checks below, for 25 named receipt checks.
- Full offline tests, Ruff, local Markdown links, and a scan confirming that no
  upstream implementation or artwork entered the distributable files.

The nine added Linux checks are: left/right parallax; forward/backward scale;
normalized diagonal travel; release stops movement; focus loss clears held
keys; pause freezes rain while allowing travel; reset restores the paused
view; resize preserves world and camera state; and 10,000-step wrapping
preserves column identities. These are planned checks, not current receipts.

The README introduces the current screensaver before its evidence. Its benchmark
sequence is glyph generation and scaling → recovery → framework efficiency →
generated execution topology. The planned SVG/navigation work stays labeled as
planned until its own artifacts and verification are published.
