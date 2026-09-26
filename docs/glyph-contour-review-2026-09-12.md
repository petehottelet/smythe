# Glyph contour review — 12 September 2026

The published glyphs passed numerical gates but have visible drawing defects.
Those gates did not establish a consistent type style. This review supersedes
the earlier optical acceptance for future artwork selection; it does not
rewrite the measured v1 files or their timing records.

**Current deliverable:** [192 rebuilt glyphs](../benchmarks/noumenon/catalog/contact-sheet-128.png)
and [per-glyph measurements](../benchmarks/noumenon/catalog/measurements.json).
Current sheets, gallery, stills, animation, and native source exports use v2.
The gallery and README animation show only the new originals. The live
reference mix remains adjustable; historical benchmark evidence retains v1.

## Reference examined

Re-examined all 137 cached column-A images from the supplied spreadsheet and
verified every image against the SHA-256 in the committed
[per-row record](data/glyph-style-measurements.csv). Also inspected the entire
[56-glyph classic vector catalog](https://github.com/petehottelet/noumenon/blob/main/svg-preview/reference/README.md)
and selected exact vector cross-sections. Spreadsheet images remain research
inputs; none were traced, copied into the new contours, or redistributed here.
The licensed base catalog retains its
[credits and MIT notice](https://github.com/petehottelet/noumenon/blob/main/svg-preview/reference/README.md).

The spreadsheet contains **56 classic-flagged and 81 expanded forms**, not one
uniform face. The expanded forms deliberately include rounded ends and bulb
terminals. The classic set mostly uses cut bar ends and controlled tapering.
Because the rain mixes 90% classic reference with 10% generated characters,
the replacement originals should use the classic terminal language throughout.
Curved bowls remain welcome; a curved shoulder does not require a round cap.

## What the reference actually does

| Property | Reference evidence | Drawing rule for the revision |
|---|---|---|
| Weight | Classic median centerline width proxy: 18.44 px in a 128 px cell; middle 50%: 18.00–21.16 px | Keep heavy stems. Judge weight at the same cell size, without glow. |
| Ink mass | Classic median ink coverage: 25.49%; middle 50%: 18.56–28.07% | Keep open space around a compact, dark silhouette. Punctuation remains smaller. |
| Scale | Classic median bounds: 84 × 92 px within 128 × 128 px | Use a common cell and alignment; do not stretch every shape to a full square. |
| Bars | BASE-015's bars are 14.69 and 14.84 units thick on a 100-unit canvas | Broad, nearly equal bars with clean rectangular ends. |
| Elbows | BASE-026 has a 13.125-unit stem and 13.906-unit crossbars | Join strokes into one clean contour. Avoid short steps where their envelopes meet. |
| Hooks | BASE-011's curved stroke starts 19.688 units wide and ends 8.594 units wide | A deliberate taper is part of the style. A generic constant-width tube is insufficient. Both exposed end cuts are straight. |
| Diagonals | BASE-020 uses flat horizontal terminal edges about 20.469 units long | Cut terminals can be oblique to the stroke direction. Do not add a semicircular cap. |
| Counters | Only 6/56 classic-flagged sheet images contain holes; 35/81 expanded images do | Use counters for the character's structure. Do not drill extra holes to improve a metric. |
| Silhouette | Full-sheet visual inspection; examples BASE-011, 014, 026, 028, 031 | Broad sweeps, clear corners and deliberate openings. Small accidental chips do not belong. |

Vector measurements above are exact differences in the imported 100-unit
coordinates, rounded to three decimals. Raster population figures come from
the existing spreadsheet study; the sheet and imported atlas are separate
sources. The skeleton width proxy includes junctions and taper, so it is not
a font stem specification or a direct measurement of cap radius.

## Why our glyphs failed

1. **Circular punches became visible divots.** The v1 generator has explicit
   circular aperture corrections in 13 glyphs: 001, 087, 096, 099, 121, 129,
   151, 156, 162, 166, 173, 180, and 181. GLYPH-001's top-right bite is one.
2. **Extra cuts created artificial differences.** Later variants subtract
   rectangles and add extra terminals to distinguish otherwise similar
   recipes. That can change uniqueness scores without improving the drawing.
3. **Stroke endings were selected locally.** Diagonal primitives request
   round caps, while bars use flat caps. GLYPH-014 and GLYPH-015 expose the
   mismatch. Other shapes mix cap treatments within a single glyph.
4. **Curve weight swells mechanically.** The old curve primitive changes
   width from 0.62× at its ends to 1.30× at its middle, regardless of the
   particular shoulder or stroke. Overlapping pieces can leave bumps and steps.
5. **Cleanup happened after composition.** Buffer closing/opening rounds the
   resulting contour; fixed circular cuts are applied afterward. This cannot
   substitute for drawing the intended boundary correctly.
6. **Aggregate measurements hid local defects.** V1's 192 glyphs already have
   an 18.00 px median width proxy and 24.24% median ink coverage. Those values
   are close to the reference despite the visibly inconsistent terminals.
   The v1 catalog also assigns 114/192 glyphs to the expanded profile.

## Replacement rules

These are authored design targets, not additional measurements of the source:

- One exposed-terminal policy: straight cuts. Keep intentional curves in
  bowls and shoulders, without pill ends, bulb caps, or random terminal changes.
- Start upright stems at 14–17 units and crossbars at 12–15 units on the
  100-unit canvas. Draw hook mass and taper explicitly; assess the resulting
  optical weight against adjacent reference glyphs.
- Draw joined shapes as continuous outlines. Author intentional holes as
  interior contours. Use no punched aperture fixes, random notches,
  post-composition erosion/dilation, or counter scaling.
- Use 8 units as the starting minimum for deliberate short projections and
  openings. Widen an entire entrance when needed. Remove incidental details
  rather than smoothing them into a smaller bump.
- Leave room for clear counters and detached marks. Require their count to
  survive 16, 32, and 128 px rendering and threshold sensitivity at 128 px.
- Vary the structure, placement of major strokes, counter arrangement, and
  proportions. Never obtain a new glyph by nibbling its edge.
- Reject obvious arrow icons and pictograms. Clean geometry alone does not
  make a convincing character; the stroke arrangement must read as writing.
- Reject new additions that read directly as existing letters or numerals,
  including simple mirrored versions. Borrow the reference's weight, cuts,
  taper and spacing, then compose a different character structure. A tiny
  notch or added speck does not make a familiar letter into a new glyph.
- Review unlit black silhouettes first, native-size samples second, green
  glow last. Bloom cannot be the reason a bad contour passes.

## The complete v2 catalog

The [authoring module](../benchmarks/noumenon/glyph_design_v2.py) constructs all 192
outlines from [authored contour data](../benchmarks/noumenon/glyph_contours_v2.json).
It reuses the restricted SVG serializer and rasterizer, not v1 geometry.
All exposed terminals follow the cut-end policy. No circular punches,
random notches, or post-composition erosion/dilation are applied.

The initial 24 drawings established the direction. The other 168 combine
broad elbows, open shoulders, cut sweeps, and detached marks in horizontal
and vertical arrangements. Candidate selection compares whole structures,
including aligned reflections; it does not add small defects for uniqueness.
A second pass replaced the remaining immediate letter-like readings in
005, 007, 010, 016, and 021.

Inspect [the complete sheet](../benchmarks/noumenon/catalog/contact-sheet-128.png)
or [four enlarged sections](../benchmarks/noumenon/catalog/README.md).
All small-size sheets contain the same current 192 IDs. Geometry, silhouette,
raster, export, and browser checks are recorded in the review receipt, kept in
the [evidence archive](../benchmarks/archive/README.md). They do not
establish a measured generation speed or universal uniqueness across scripts.

| Current 192-glyph sample, 128 px cells | Measured value |
|---|---:|
| Median stroke-width proxy | 18.00 px |
| Median ink coverage | 20.47% |
| Median bounding dimensions | 88 × 90.5 px |
| Glyphs with intentional counters | 3 |
| Aligned/reflected pairs compared | 18,336 |
| Pairs flagged at 0.85 IoU | 0 |

These describe the current set; they are not a score for aesthetic quality.
The full pair comparison is retained in
[distinctness.json](../benchmarks/noumenon/catalog/distinctness.json).

### Letter and numeral rejection pass

| Candidate rejected | Reading to avoid | Revision 3 structure |
|---|---|---|
| 009 | Theta-like closed bowl and middle stroke | Offset upright, open shoulder and independent upper return |
| 018 | 8-like stacked bowls | Two stepped bars and a separate descending sweep |
| 019 | Capital A | Offset upper elbow and separate tapered lower sweep |
| 020 | Mirrored numeral/letter form | Open crossbar, detached square and separate lower crook |
| 023 | Mirrored S | Short upper stem and offset lower returning stroke |

All 24 candidates were reviewed for immediate familiar-character readings.
A supplementary comparison against Latin letters, digits and selected Greek
letters in three bold fonts helped identify candidates; visual review made the
decision. Silhouette overlap alone also produces false matches for simple
bars, so it is not an approval rule or a claim of uniqueness across scripts.
This rejection rule applies to the new generated additions. The licensed
reference catalog retains its original characters.

### Detached center bar in 017

The user rejected 017's angled inner stroke and proposed a separate horizontal
piece in the character's center. Revision 4 keeps the roof and descending
curve, then places a flat rectangular bar inside that opening. The bar is
25 × 13 units on the 100-unit canvas, with a minimum 9-unit gap from the
outer stroke. Both components remain separate at 16, 32 and 128 px.
[Inspect revised 017](../benchmarks/noumenon/catalog/glyph-017-detail.png).

The approved 017 SVG is preserved byte-for-byte in the full set. The review
record in the [evidence archive](../benchmarks/archive/README.md) binds the
completed catalog, checks, and the presentation assets of that date.

## Completion status and next work

1. **Complete:** 192 v2 outlines and 16/32/64/128 px test sheets.
2. **Complete:** current gallery, specimen plate, web texture, stills, GIF,
   layered Canvas view, and Windows/macOS/Linux source exports use v2.
3. **Complete:** preview-only materials show the new originals, with no old
   before/after rows or reference characters mixed into the glyph sheets.
4. **Complete:** the [192/256-glyph v2 campaign](../benchmarks/svg_v2_results.md)
   measures current contour compilation, validation and export across 36
   workflows. Design selection is outside timing; v1 evidence remains historical.
5. **Next:** qualify newly compiled native builds on all target platforms.
   Export parity is distinct from platform execution; distribution stays
   source-only. Historical compiled-build receipts remain in their archive.

The current set is available locally for inspection. Approval of 017 does
not stand in for user approval of every character or authorization to push
past the user's requested preview checkpoint.
