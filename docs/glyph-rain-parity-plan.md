# Glyph Rain reference parity plan

Restore the reference's fixed 2D code grid as the default, use its classic base
glyphs, and introduce Smythe's original glyphs occasionally. Preserve the
interactive 3D explorer as a selectable mode. Adapt the pinned MIT-licensed
REGL renderer, preserve its license and source hashes, and add a separate atlas
for Smythe's independently generated original artwork.

The default **Matrix green** color grade uses 137° hue, 80% saturation, and a
mint cursor. The original **Reference** palette remains selectable. This color
grade is a deliberate adaptation of the licensed renderer; the pinned defaults
below remain the source audit.

This is an implementation and acceptance plan, not a claim that parity has
shipped. The current SVG generation benchmark measures the original 192-glyph
catalog. Adding a base catalog or changing the renderer does not change that
benchmark's input, output, or claim scope.

## Source and scope

The audited reference is `m8e/matrix-rain` at commit
`5ba90490453ceceb6812d6b1bc658a99a92411d0`, inspected in the clean local clone
`smythe/tmp/matrix-rain-reference`. The primary sources are its
[configuration](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/config.js),
[rain state](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.raindrop.frag.glsl),
[glyph state](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.symbol.frag.glsl),
[rain drawing](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/rainPass.frag.glsl),
[pass ordering](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/main.js),
[bloom](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/js/regl/bloomPass.js),
and [palette mapping](https://github.com/m8e/matrix-rain/blob/5ba90490453ceceb6812d6b1bc658a99a92411d0/shaders/glsl/palettePass.frag.glsl).
The default REGL/WebGL path defines the initial comparison. WebGPU and special
display modes have separate capability requirements.

The starting Smythe implementation is
[model.mjs](../screensaver/svg-preview/model.mjs) and
[rain.js](../screensaver/svg-preview/rain.js). Its previous renderer and receipts
are preserved in [renderer-v1](../benchmarks/partitions/glyph_svg_v1/renderer-v1/).
Findings below come from source inspection. Proposed visual tolerances are
acceptance targets; they are not measurements already taken from a new renderer.

The adaptation lives in [engine](../screensaver/svg-preview/engine/). Its
[manifest](../screensaver/svg-preview/engine/manifest.json) records all 15 pass,
utility, and shader sources, the pinned upstream Git blob hashes, source SHA-256,
adapted SHA-256, and per-file changes. The illumination, intro, optional effect,
bloom kernels, palette shader, and color conversion retain their original
source bytes. The [MIT license](../screensaver/svg-preview/engine/LICENSE)
accompanies the files.

The bounded changes are a second MSDF atlas and catalog selection channel,
user camera offsets, host-controlled simulation time, complete shader readiness,
one-pixel minimum bloom buffers, and higher precision for atlas coordinates.
The host advances simulation at a fixed nominal 60Hz. A paused camera draw
reuses the latest compute buffers and frozen time; it cannot cycle glyphs or
advance illumination. Camera/webcam, image effects, WebGPU, and Looking Glass
are outside this adapted preview's supported subset.

## What makes the reference different

| Behavior | Pinned reference | Starting Smythe preview | Required change |
|---|---|---|---|
| Scene | Fixed 2D grid by default | 420 columns distributed through a 140 × 80 × 60 depth field | Default to 2D; put the existing navigation contract behind 3D selection |
| Scale | 80 cells across the viewport's longer dimension | Apparent size varies with depth from roughly 13 to 207 px at 1080px height | Use uniform 24px cell pitch at 1920 × 1080 in classic mode |
| Rain | Periodic illumination of stationary cells; successive fronts coexist | One wrapped head and a 12–24-cell tail per column | Use an ordered train of illumination fronts with variable spacing |
| Rhythm | Column-dependent speed; warped phase changes spacing without overtaking | A constant 16–30 cells/s head moving around 70 rows | Match 15–30 cells/s at classic defaults and remove the single-head restriction |
| Glyph choice | 57 classic source slots, including one blank | Uniform choice from 192 original glyphs | Select the base catalog most of the time; weight originals separately |
| Glyph cycling | Separate per-cell age, staggered initial age; nominal 1.8 changes/s at 60 callbacks/s | Independent 0.35–0.899s intervals, yielding approximately 1.11–2.86 changes/s | Match a shared nominal cadence with staggered cell phase |
| Light | Scalar body/cursor/glint channels; contrast and brightness before postprocessing | Two precolored green shadow fills, attenuated by depth and tail position | Separate exposure from color and distinguish the cursor channel |
| Glow | Scene-wide, five-scale bloom before green palette lookup | Broad and tight per-glyph Canvas shadows | Bloom scalar channels, combine exposure, then map color |
| Empty space | Brightness threshold plus blank source slot and spatial gaps | Only the selected tail cells are drawn | Match the reference's exposure distribution, not a fixed tail occupancy |
| Resolution | Drawing buffer is 0.75 × CSS size × DPR | CSS size × DPR, with DPR capped at 2 | Expose resolution explicitly and compare at matched drawing-buffer sizes |
| Controls | URL configuration and double-click fullscreen | Arrow/touch navigation, pause, reset, fullscreen; only `seed` in the URL | Preserve useful controls and add a documented supported configuration subset |

The reference's head spacing is not an arbitrary particle count. Its default
unwarped period is 75 underlying grid cells. An 80-cell column spans about
1.067 periods; the visible 45 rows of a 16:9 viewport span about 0.6 periods.
Warping allows variable gaps and multiple heads, but a visible column may also
contain zero or one head. Forcing two to four heads into every visible column
would increase density beyond the default reference.

The pinned shader moves fronts in the same column at the same velocity. The
column's velocity varies between columns; the phase warp changes the gaps.
Its phase remains monotone, so neighboring fronts cannot overtake. Preserve
the licensed phase shader and its ordering property when adding the original
atlas and interactive camera offsets.

## Actual default configuration

These values come from `js/config.js`, not the reference README. HSL and RGB
components use the 0–1 convention. They describe the upstream configuration,
not the new Matrix green default.

| Setting | Pinned default |
|---|---|
| Font / effect | `matrixcode` / `palette` |
| Scene | `volumetric=false`, `isometric=false`, `useHoloplay=false` |
| Grid / density | `numColumns=80`, `density=1` |
| Playback | `animationSpeed=1`, `fps=60`, `fallSpeed=0.3`, `forwardSpeed=0.25` |
| Rain | `raindropLength=0.75`, `brightnessDecay=1`, `loops=false`, `skipIntro=true` |
| Glyph changes | `cycleSpeed=0.03`, `cycleFrameSkip=1` |
| Glyph geometry | `glyphHeightToWidth=1`, `glyphVerticalSpacing=1`, `glyphEdgeCrop=0`, `glyphFlip=false`, `glyphRotation=0` |
| Body exposure | `baseBrightness=-0.5`, `baseContrast=1.1`, `brightnessOverride=0`, `brightnessThreshold=0` |
| Cursor | `isolateCursor=true`, `cursorColor=HSL(0.242,1,0.73)`, `cursorIntensity=2` |
| Glint | `isolateGlint=false`, `glintColor=HSL(0,0,1)`, `glintIntensity=1`, `glintBrightness=-1.5`, `glintContrast=2.5` |
| Bloom | `bloomStrength=0.7`, `bloomSize=0.4`, `highPassThreshold=0.1` |
| Dither / background | `ditherMagnitude=0.05`, `backgroundColor=HSL(0,0,0)` |
| Palette | `HSL(0.3,0.9,0)` at 0; `HSL(0.3,0.9,0.2)` at 0.2; `HSL(0.3,0.9,0.7)` at 0.7; `HSL(0.3,0.9,0.8)` at 0.8 |
| Textures | `baseTexture=null`, `glintTexture=null` |
| Optional effects | `hasThunder=false`, `isPolar=false`, `rippleTypeName=null`; ripple thickness 0.2, scale 30, speed 0.2 |
| Display / backend | `resolution=0.75`, `renderer=regl`, `useHalfFloat=false`, `suppressWarnings=false` |
| Other | `slant=0`, `useCamera=false`, `testFix=null`; `once` is absent and therefore inactive |

At 1920 × 1080, DPR 1, these defaults produce a 1440 × 810 drawing buffer.
The classic screen-space cell pitch remains 24 CSS px: 80 visible columns and
45 visible rows, cropped from an 80 × 80 simulation grid. At 1080 × 1920 the
visible dimensions exchange places. The cell pitch is set by the longer
dimension, not always by viewport width.

The body exposure remap is contrast 1.1 with an offset of −0.5. Before bloom,
the positive part starts above raw brightness 0.4545; this is a threshold in
brightness space, not a claim about the proportion of illuminated screen
pixels. Cursor exposure remains a separate channel, so a bright head is more
than a slightly lighter body glyph.

The reference bloom pyramid begins at 40% of drawing-buffer dimensions and
has five levels, each halving each dimension. At the reference 1080p/DPR-1
settings the levels are 576 × 324, 288 × 162, 144 × 81, 72 × 40, and 36 × 20.
Channels are thresholded before separable blur, combined with decreasing
scale weights, and added to the sharp exposure image. The palette has 2,048
samples interpolated in RGB after converting HSL stops. Dither subtracts at
most approximately `0.05/3` from exposure before color mapping.

Two README defaults are stale: `forwardSpeed` is **0.25**, not 1, and
`resolution` is **0.75**, not 1. `glintIntensity` appears in its README but is
not a mapped URL option. Conversely, mapped `glyphIntensity` is not consumed
by the audited drawing passes. Do not reproduce these documentation defects.

## This iteration: core parity and user-requested mixing

### P0 — fixed grid and ordered illumination

Implement one persistent cell identity per grid position. Store glyph state
separately from the column illumination state. Use a repeating, monotonically
ordered family of illumination fronts with column-specific offsets and speeds.
Vary their spacing without allowing their order to reverse. A head lights the
cell at the leading edge; it never transports the cell's geometry.

Acceptance checks:

1. At 1920 × 1080 and 1080 × 1920, classic mode has 24px pitch and a uniform
   cell size within 0.1 CSS px. At zero camera motion, a tracked cell's origin
   moves less than 0.01px over 10 seconds while its exposure changes.
2. A controlled 160-row diagnostic column shows multiple fronts. Across
   10,000 updates, front ordering never reverses and no front disappears by
   colliding with its neighbor. Test wrap transitions explicitly.
3. At `fallSpeed=0.3`, a tracked front moves 15–30 cells in one second,
   depending on its column's fixed speed factor. At zero fall speed, fronts
   remain still while glyph cycling continues. Negative fall speed reverses
   illumination without reversing glyph orientation.
4. Default head counts are observations of the phase field, not a forced
   minimum per visible column. Halving `raindropLength` approximately doubles
   the long-run number of fronts per 10,000 underlying rows; test a long sample
   and allow 5% finite-window variation.
5. Identical initial state and elapsed time produce identical cell identity and
   illumination state under 30/60/144 update schedules. Define a fixed
   simulation cadence or elapsed-time equivalent for smoothing and cycling.

The reference increments glyph age per animation callback. At its nominal
60Hz behavior, `cycleSpeed=0.03` corresponds to one change per approximately
0.556 seconds. Its display `fps` option does not eliminate all simulation
updates, and faster callback rates can change the observed cycling speed.
Match the nominal 60Hz appearance while keeping Smythe's elapsed-time
determinism. Document that intentional correction instead of claiming the
reference's refresh-rate dependence is reproduced.

### P0 — base glyphs with occasional originals

Use the exact, separately attributed classic MSDF atlas for the familiar
forms. Its 57-slot sequence has 56 visible outlines and intentionally blank
source slot 4. Preserve all 57 selection slots. The companion `BASE_GLYPHS`
vector catalog documents those 56 visible outlines. Keep the 192 `SVG_GLYPHS`
originals and their measured catalog unchanged; build a separate SDF atlas
from those exact SVG files for rendering.

Smythe default: **90% base choices, 10% original choices**, exposed as
an `originalMix` percentage control (`glyphMix=0.1` internally). This is a product choice interpreting “here and
there,” not an upstream setting. Select the catalog first, then a member; a
uniform draw from 56 + 192 entries would make the originals dominate.

Acceptance checks:

1. `originalMix=0` uses only the 57 base sequence slots, including its blank.
   `originalMix=100` uses only the 192 original glyphs. At 10%, a 100,000-choice
   deterministic audit contains 9.5–10.5% original choices.
2. Base blank choices occur with probability 1/57 within the base branch.
   Blank is represented as no ink, not a fabricated empty SVG or a failed
   glyph-validation record.
3. Selection changes happen at glyph-cycle boundaries, never on every draw.
   The base/original weighting is independent of viewport size, camera depth,
   glyph brightness, frame rate, and catalog entry count.
4. Verify every displayed path against its catalog and provenance hash.
   Base geometry remains unmodified; originals retain their exact benchmark
   SVG hashes. Renderer color and exposure do not alter source geometry.
5. Review 0%, 10%, and 100% contact views with the same lighting and cell pitch.
   The 10% mode must retain the familiar base character texture at first glance.

### P0 — scalar bloom and color

Render sharp glyph coverage into separate scalar body and cursor exposure
channels. Reserve a glint channel for a later glyph set that actually provides
glint artwork. Apply the contrast/offset, thresholded multiscale bloom, and
palette in that order. Blend the cursor contribution independently. Keep the
geometry sharp underneath the bloom; avoid recoloring preblurred sprites as
the primary exposure model.

Use **Matrix green** by default: body hue **137°**, saturation **80%**, preserving
each selected preset's lightness values and stop positions. For Classic, those
stops are 0, 0.2, 0.7, and 0.8, giving approximate RGB colors `#000000`,
`#0A5C21`, `#75F098`, and `#A3F5BA`. The leading-glyph color is **`#A2FFD8`**.
Keep **Reference** available with the selected preset's original palette and
cursor color. Color selection does not alter glyph geometry, illumination,
exposure controls, or bloom ordering.

The [screenshot color measurements](../benchmarks/partitions/glyph_rain_reference_v1/color-comparison.json)
record a median body hue of **136.92°** in the user's reference, versus
**108.75°** in the earlier preview. The reference rain crop excludes the UI at
`y ≥ 685`; matched mid-green masks give 136.80° versus 108.51°. Bright-tip
channel medians are mint-white in the reference and yellow-white in the earlier
preview. The receipt retains image hashes, masks, distributions, and a separate
UI comparison; it does not redistribute the user screenshot.

The measured hue sets the direction; 80% saturation is the chosen grade.
Preserving HSL lightness stops preserves that control curve, not exact RGB
luminance after changing hue and saturation. Screenshot noise, scaling,
clipping, and unrecorded display transforms limit exact color matching.

Acceptance checks:

1. With bloom and dither off, a controlled exposure ramp matches the selected
   palette's stops within 2/255 per RGB channel. Verify Matrix green and
   Reference separately. Distinguish body and cursor
   channels in the diagnostic output.
2. With all cells dark, output matches the black background. A single bright
   cell produces both a sharp core and a measurable halo outside its outline.
   `bloomStrength=0` removes the halo; `bloomSize=0` disables bloom cleanly.
3. With the same single-cell exposure, verify that the implementation adds
   blurred scalar exposure before palette lookup. A deliberately nonlinear
   test palette must distinguish this result from blurring final RGB colors.
4. Capture the pinned reference and adapted renderer at 1920 × 1080,
   DPR 1, resolution 0.75, with controls excluded, the **Reference** palette,
   and identical settings. The Matrix green grade is a separate color review.
   Sample 60 frames after a five-second warmup. Record brightness percentiles,
   dark-pixel fraction, cursor/body contrast, head count, and halo falloff.
   Proposed target: dark-pixel fraction within 5 percentage points and
   median halo half-maximum radius within 15% of the captured reference.
5. Random layouts prevent meaningful whole-frame pixel equality. Use
   controlled single-glyph ramps for pixel tests and distributions for the
   live effect. Publish the captures and definitions before claiming parity.

### P1 — controls, intro, and optional 3D

Ship classic 2D as the initial scene, plus an explicit 3D switch. Retain the
current arrow and touch navigation, Space pause, R camera reset, F fullscreen,
reduced-motion start, input clearing, and resize continuity. In 2D mode,
navigation controls should clearly offer entering 3D; they should not silently
introduce depth into the supposedly fixed-grid default.

The upstream `forwardSpeed` is automatic depth drift. It is separate from
Smythe's user-controlled forward motion. Preserve both as distinct controls
in 3D, including an auto-drift value of zero. Never overload the webcam
`camera` parameter to mean viewpoint navigation.

Acceptance checks:

1. Classic mode starts fully populated (`skipIntro=true`). With
   `skipIntro=false`, first-frame coverage is zero, independently staggered
   columns activate, and activated cells stay eligible thereafter. Replaying
   the same initial state and time reproduces the intro. The adapted shaders
   use fixed coordinate hashes; a configurable random seed is not exposed.
2. Pausing freezes illumination, glyph changes, and auto-drift. In 3D, held
   arrow input still moves the viewpoint. R returns the camera to its origin
   without resetting rain phase; a paused reset restores the previous image.
3. Verify 4:1 lateral parallax at view distances 0.2 and 0.8 in the adapted
   projection's units, scale growth during forward movement, and 10,000-step
   wrapping. Camera coordinates retain a 140-unit horizontal wrap and 60-unit
   depth wrap; they are converted to the reference's normalized projection.
   Losing focus,
   hiding the tab, or cancelling touch input clears all held navigation.
4. Resize retains cell identities where dimensions overlap, camera,
   and elapsed rain phase. Freeze all input during a recorded performance
   sample and invalidate samples on scene, size, catalog, or control changes.
5. Test classic and 3D independently at 1080p/DPR 1 with three fresh sessions,
   five-second warmup, and 60-second samples. Record real drawing-buffer size,
   callback intervals, CPU submission, cache/buffer accounting, and hashes.
   Retain the proposed 16.7ms P95 interval target; passing requires new evidence.
   Existing v1 browser measurements remain diagnostic historical records.

### P1 — pixel controls and outlined logo

Apply the [explorer style](style.md#explorer-controls) to the settings sheet,
logo bar, and playback controls. Use VT323 pixel lettering for headings,
labels, numerals, buttons, inputs, and help text. Align labels and values into
clear rows. Use a separate Trajan Pro Bold outline for the SMYTHE logo, surrounded
by generous black padding. Keep text controls borderless.

The interface uses primary `#37FF6E`, bright `#9CFFBC`, and black, following
[hottelet.com](https://www.hottelet.com/). It has no text or panel glow and no
corner ornaments or stepped edges. Selection and keyboard focus use a solid
fill or visible outline. The rain's 137° grade, `#A2FFD8` leading glyphs, and
bloom remain separate. The README charts retain their existing typography
and black-and-white palette.

Acceptance targets:

1. Interface text uses VT323; the logo is a Trajan Bold outline. Pixel lettering
   remains readable and labels and values stay aligned at 390px and 1920px
   viewport widths.
2. Text controls have no decorative frames; keyboard focus remains distinct.
   Controls have at least 44px touch targets and do not rely on
   hover to reveal their labels or current values.
3. Verify the exact UI palette, black logo padding, borderless controls, and absence
   of interface glow and corner ornaments. Focus, selection, and disabled
   states remain distinct from the animated rain behind the controls.
4. The visual treatment preserves draft/apply/cancel behavior, mobile scrolling,
   focus restoration, and native input keys. Capture the open menu and rain
   after the color and style changes; earlier screenshots describe earlier UI.

## Public URL inventory in the pinned reference

There are **40 mapped keys and 9 aliases: 49 accepted names**. This inventory
describes upstream parsing; it does not announce 49 implemented Smythe options.
Core support this iteration should cover classic/3D, mixing, grid density,
rain/cycle speeds, bloom, resolution, palette, pause/intro, and glyph orientation.
Unsupported options must be identified in the UI or documentation, not silently
presented as working modes.

| Accepted name(s) | Meaning and parser behavior |
|---|---|
| `version` | Preset name; unknown name falls back to classic |
| `font` | One of the ten catalog names below; recognized names are case-sensitive |
| `effect` | One of the ten effect names below; unknown name uses palette drawing |
| `camera` | Webcam input, mapped to `useCamera`; boolean |
| `numColumns`, `width` | Grid maximum dimension; integer, no upstream range clamp |
| `density` | Nonnegative float; affects volumetric columns, except debug effect `none` |
| `resolution` | Drawing-buffer multiplier; float, no upstream range clamp |
| `animationSpeed` | Global animation multiplier; float |
| `forwardSpeed` | Automatic volumetric approach speed; float |
| `cycleSpeed` | Glyph-age increment; float |
| `fallSpeed` | Illumination movement speed; float, including negative values |
| `raindropLength`, `dropLength` | Vertical phase scale; float |
| `slant`, `angle` | Rain/grid rotation in degrees, converted to radians |
| `bloomSize` | Bloom working-resolution multiplier, clamped 0–1 |
| `bloomStrength` | Bloom contribution, clamped 0–1 |
| `ditherMagnitude` | Exposure noise magnitude, clamped 0–1 |
| `url` | Image-effect source URL, stored as `bgURL` |
| `palette`, `paletteRGB` | Repeated RGB + stop-position groups of four numbers |
| `paletteHSL` | Repeated HSL + stop-position groups of four numbers |
| `stripeColors`, `stripeRGB`, `colors` | Repeated RGB triples |
| `stripeHSL` | Repeated HSL triples |
| `backgroundColor`, `backgroundRGB` | One RGB triple |
| `backgroundHSL` | One HSL triple |
| `cursorColor`, `cursorRGB` | One RGB triple |
| `cursorHSL` | One HSL triple |
| `glintColor`, `glintRGB` | One RGB triple |
| `glintHSL` | One HSL triple |
| `cursorIntensity` | Nonnegative float |
| `glyphIntensity` | Nonnegative float; parsed but unused in the audited render passes |
| `volumetric` | 3D mode; boolean |
| `glyphFlip` | Horizontal reflection; boolean |
| `glyphRotation` | Nonnegative degree value; shader supports general angles despite the 90° suggestion in comments |
| `loops` | Work-in-progress looping variant; boolean |
| `fps` | Requested display rate clamped 0–60; simulation callbacks can continue |
| `skipIntro` | Start already activated; boolean |
| `renderer` | `webgpu` requests WebGPU if available; otherwise REGL fallback |
| `suppressWarnings` | Suppress the hardware-acceleration notice; boolean |
| `once` | Execute a single animation callback; boolean |
| `isometric` | Alternate projection/rotation in volumetric mode; boolean |
| `testFix` | Diagnostic string; known branches `fwidth_10_1_2022_A` and `fwidth_10_1_2022_B` alter extension requests |

Upstream boolean parsing checks whether the lowercase string contains `true`;
numeric parsing drops NaN but is otherwise permissive. Our parser should use
explicit booleans, finite numbers, safe size limits, and useful error messages.
This is a deliberate robustness difference, not a visual parity gap.

Parameter precedence is defaults, then preset, then font texture metadata,
then mapped URL values. Setting any explicit `effect` also defaults the cursor
and glint colors to white unless those colors were supplied, and sets cursor
intensity to 2. Therefore `effect=palette` can differ from omitting `effect`.
Palette or color aliases that target the same setting resolve in URL entry
order. Texture selection comes from presets, not arbitrary texture URL keys.

Important internal settings are **not** URL options: `cycleFrameSkip`,
`baseBrightness`, `baseContrast`, `glintBrightness`, `glintContrast`,
`brightnessDecay`, `brightnessOverride`, `brightnessThreshold`,
`highPassThreshold`, `glyphEdgeCrop`, `glyphHeightToWidth`,
`glyphVerticalSpacing`, `isolateCursor`, `isolateGlint`, `glintIntensity`,
`baseTexture`, `glintTexture`, `hasThunder`, `isPolar`, the ripple settings,
`useHalfFloat`, and `useHoloplay`. Adding these to Smythe is a separately
documented extension; do not claim upstream URL compatibility for them.

### Every preset and alias

Unlisted values inherit the defaults above. Cycle values below are the final
object values; repeated keys in several upstream preset objects overwrite
earlier values.

| Preset | Font | Columns | Animation / fall / cycle | Drop scale | Distinguishing overrides |
|---|---|---:|---|---:|---|
| `classic` | matrixcode | 80 | 1 / 0.3 / 0.03 | 0.75 | Default 2D palette |
| `megacity` | megacity | 40 | 0.5 / 0.3 / 0.03 | 0.75 | Expanded classic catalog |
| `neomatrixology` | neomatrixology | 40 | 0.8 / 0.3 / 0.03 | 0.75 | Yellow-toned palette and cursor |
| `operator` | matrixcode | 108 | 1 / 0.6 / 0.01 | 1.5 | Frame skip 8; fixed body brightness 0.22; glyph aspect 1.35; crop 0.15; box ripple; bloom size 0.6, strength 0.75 |
| `nightmare` | gothic | 60 | 1 / 1.2 / 0.35 | 0.5 | No isolated cursor; thunder; 22.5° slant; decay 0.75; warm palette |
| `paradise` | coptic | 40 | 1 / 0.02 / 0.005 | 0.4 | Polar layout; circle ripple; decay 0.05; no isolated cursor; bloom strength 1 |
| `resurrections` | resurrections | 70 | 1 / 0.3 / 0.03 | 0.75 | Crop 0.1; body brightness −0.7, contrast 1.17; cyan-green palette; high pass 0 |
| `trinity` | resurrections | 60 | 1 / 0.3 / 0.01 | 0.3 | Volumetric; forward 0.2; density 0.75; pixel base texture, metal glints |
| `morpheus` | resurrections | 60 | 1 / 0.3 / 0.015 | 0.4 | Volumetric; forward 0.1; density 0.75; metal base texture, mesh glints; magenta base palette |
| `bugs` | resurrections | 60 | 1 / 0.3 / 0.01 | 0.3 | Volumetric; forward 0.4; density 0.75; metal base texture, sand glints; yellow base and blue highlights |
| `palimpsest` | huberfishA | 40 | 1 / 0.5 / 0.03 | 1.2 | No isolated cursor; frame skip 3; −11.25° slant; bloom strength 0.2; pale-to-blue palette |
| `twilight` | huberfishD | 50 | 1 / 0.1 / 0.03 | 0.9 | Bloom strength 0.1; cursor intensity 1.5; blue-purple-gold palette |
| `holoplay` | resurrections | 20 | 1 / 0.3 / 0.01 | 0.3 | REGL; Looking Glass; volumetric; density 3; forward 0; bloom 0; dither 0; metal glints |
| `3d` | matrixcode | 80 | 1 / 0.5 / 0.03 | 0.3 | Volumetric; body brightness −0.9, contrast 1.5 |

Aliases: `throwback` and `1999` → `operator`; `updated` and `2021` →
`resurrections`; `2003` → `classic`. There are 14 distinct preset definitions
and five aliases. Classic, Operator, and the selectable 3D mode are the selected preset
subset this iteration; the rest remain named backlog items until built and
verified.

### Every font and effect name

| Font name | Sequence length | Texture grid |
|---|---:|---|
| `matrixcode` | 57, including the blank slot | 8 × 8 |
| `megacity` | 64 | 8 × 8 |
| `resurrections` | 135 | 13 × 12 |
| `coptic` | 32 | 8 × 8 |
| `gothic` | 27 | 8 × 8 |
| `huberfishA` | 34 | 6 × 6 |
| `huberfishD` | 34 | 6 × 6 |
| `gtarg_tenretniolleh` | 36 | 6 × 6 |
| `gtarg_alientext` | 38 | 8 × 5 |
| `neomatrixology` | 12 | 4 × 4 |

Effect names: `palette` and `plain` use palette mapping; `customStripes`,
`stripes`, `pride`, `transPride`, and `trans` use stripe mapping; `image` uses an
image; `mirror` applies interactive distortion and optional webcam input;
`none` shows the rain diagnostic channels. There are ten names. Explicit
stripe colors override the flag palettes; without explicit colors, `pride`
uses the pride palette and the other stripe names use the trans palette.

## Optional capabilities after core parity

| Capability | Work and acceptance needed before claiming support |
|---|---|
| Additional fonts/presets | Separate artwork provenance for each catalog; exact resolved preset tests; exposure/glint review; representative screenshots |
| Stripes and custom images | Correct mapping after bloom; parse/validate colors; image loading and cross-origin failure states; separate image provenance |
| Webcam mirror | Explicit opt-in; video-only permission; permission-denied handling; stop tracks when disabled; five-click ripple history; no camera request during ordinary rain |
| Thunder, polar layouts, ripples | Independent effects; preset-specific tests; reduced-motion behavior and control exposure |
| Loop export / single-frame capture | Define exact loop period and matching first/last states; verify deterministic capture; upstream `loops` is labeled work in progress |
| Isometric projection | Separate 3D projection tests; verify glyph orientation, culling, and navigation expectations |
| WebGPU | Separate adapter/device-loss/fallback tests and shader validation; do not treat REGL parity as proof of a second backend |
| Looking Glass / holographic output | Device-specific calibration and multiview/quilt checks on hardware; upstream WebGPU has a multiview TODO, so backend capability is not interchangeable |
| Native Windows, macOS, Linux | Export the same base/original catalogs and implement the new exposure pipeline; build and run platform smoke tests before changing compiled downloads |

Core completion means the default scene, base-heavy mixture, illumination
rhythm, glyph cycling, bloom/palette order, and preserved 3D controls have
passed their checks. It does not mean every upstream preset, URL option,
webcam effect, display device, or native binary has reached parity.
