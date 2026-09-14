# Current Smythe materials

Use these links for the revised glyphs, previews, and benchmark reports.
The original catalog under `benchmarks/partitions/glyph_svg_v1/` is historical
evidence. Its contact sheet contains superseded designs.

## Glyphs and previews

| Material | Current version |
|---|---|
| Complete 192-glyph sheet | [All revised originals](../screensaver/glyph-design-v2/contact-sheet-128.png) |
| Complete 256-glyph sheet | [The same 192 plus 64 benchmark designs](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-128.png) |
| Individual SVGs and detail sheets | [192-glyph catalog](../screensaver/glyph-design-v2/README.md) · [256-glyph catalog](../benchmarks/partitions/glyph_svg_v2_256/README.md) |
| Small-size inspection | 192: [16 px](../screensaver/glyph-design-v2/contact-sheet-16.png), [32 px](../screensaver/glyph-design-v2/contact-sheet-32.png), [64 px](../screensaver/glyph-design-v2/contact-sheet-64.png). 256: [16 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-16.png), [32 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-32.png), [64 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-64.png). |
| README animation | [GIF](../screensaver/svg-preview/preview.gif) · [Capture record](../screensaver/svg-preview/preview-animation.json); shows only the revised 192 originals |
| README still | [PNG](../screensaver/svg-preview/preview.png); shows only the revised 192 originals |
| Selected specimens | [Vector sheet](../assets/glyph_rain/glyph_specimens.svg) |
| Interactive gallery and effect | [Run the web explorer](../screensaver/svg-preview/README.md); the original-only gallery and adjustable reference mix have separate roles |
| Windows, macOS, and Linux source exports | [Native catalog and hashes](../screensaver/native-catalog.json) · [Build instructions](../screensaver/README.md); 192 revised originals plus the licensed reference set |

The active screensaver uses 192 original designs. The 256-glyph set is a
benchmark extension whose first 192 SVG files match the active catalog exactly.
Reference characters remain separately credited. Precompiled distribution is
paused; source exports do not establish platform execution or feature parity.

## Benchmark material

| Material | Authoritative location |
|---|---|
| Glyph compilation, validation, and export | [192/256 v2 report](../benchmarks/svg_v2_results.md) and its linked raw trials |
| Astra/Sol text comparison | [200-workflow report and review status](../benchmarks/results/astra_20260913_main/README.md) |
| Fable 5.1 extension | [Preparation and launch prerequisites](../benchmarks/fable_51_benchmark_plan.md); no Fable results yet |
| All campaigns and outstanding work | [Evidence index](../benchmarks/README.md) · [Delivery audit](benchmark-delivery-audit-2026-09-13.md) |
| Public charts | [Monochrome SVG assets](../assets/benchmarks/) generated from committed records |

The glyph workflow compiles authored contours. The Astra/Sol and planned Fable
studies measure text workflows; they do not generate a replacement glyph set.
Model results alone do not require redrawing a contact sheet. A changed glyph
catalog requires new sheets, previews, exports, and matching artwork evidence.

## Completion check for every benchmark update

1. Reconcile all trials, failures, costs, unknown charges, judgments, and review
   status. Preserve the original input/output records and source identities.
2. Verify that current glyph IDs and SVG hashes agree across the catalog,
   contact sheets, browser data, selected specimens, and native source exports.
   If glyphs change, regenerate all affected 192/256 sheets and detail views.
3. Recapture the README animation and still when their visible glyphs or effect
   change. Retain capture settings and source hashes; an animation's playback
   rate is not renderer performance evidence.
4. Run `python benchmarks/render_readme_charts.py`. Check every changed chart
   against its committed inputs, evidence status, units, denominators, and
   uncertainty. Preserve the black-and-white chart style. Promote only reviewed,
   claimable campaigns that have no known measurement defects.
5. Update the README, documentation map, roadmap, unreleased changelog, examples,
   benchmark index, delivery audit, and affected subsystem guides together.
   Link current showcase images directly; label historical evidence at entry.
6. Validate Markdown and HTML asset links, inspect the contact sheets and
   changed charts, and verify native catalog exports. Run code/platform checks
   appropriate to any implementation changes. Keep actual native-platform
   qualification distinct from source consistency checks.
7. Push the reviewed changes and verify GitHub serves the intended files and
   commit. Use the [release checklist](../RELEASING.md) for an actual release;
   completing a benchmark does not itself publish a package version.

The [13 September materials check](data/materials-review-2026-09-13.json)
records the assets and checks for this update. Historical image and benchmark
bytes remain intact so earlier measurements can still be reproduced.
