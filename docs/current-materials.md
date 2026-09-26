# Current Smythe materials

Use these links for the revised glyphs, previews, and benchmark reports.
The original catalog under `benchmarks/partitions/glyph_svg_v1/` is historical
evidence. Its contact sheet contains superseded designs.

## Glyphs and previews

| Material | Current version |
|---|---|
| Complete 192-glyph sheet | [All revised originals](../benchmarks/noumenon/catalog/contact-sheet-128.png) |
| Complete 256-glyph sheet | [The same 192 plus 64 benchmark designs](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-128.png) |
| Individual SVGs and detail sheets | [192-glyph catalog](../benchmarks/noumenon/catalog/README.md) · [256-glyph catalog](../benchmarks/partitions/glyph_svg_v2_256/README.md) |
| Small-size inspection | 192: [16 px](../benchmarks/noumenon/catalog/contact-sheet-16.png), [32 px](../benchmarks/noumenon/catalog/contact-sheet-32.png), [64 px](../benchmarks/noumenon/catalog/contact-sheet-64.png). 256: [16 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-16.png), [32 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-32.png), [64 px](../benchmarks/partitions/glyph_svg_v2_256/catalog/contact-sheet-64.png). |
| Noumenon animation | [GIF](../assets/noumenon/noumenon-loop.gif); shows only the revised 192 originals |
| Selected specimens | [Vector sheet](../assets/noumenon/noumenon_specimens.svg) |
| Screensaver, web explorer, and native source exports | [Noumenon repository](https://github.com/petehottelet/noumenon); 192 revised originals plus the licensed reference set, with its own capture records and build instructions |

Noumenon uses the 192 original designs. The 256-glyph set is a benchmark
extension whose first 192 SVG files match the active catalog exactly.
Reference characters remain separately credited. Precompiled distribution is
paused; source exports do not establish platform execution or feature parity.

## Benchmark material

| Material | Authoritative location |
|---|---|
| Glyph compilation, validation, and export | [192/256 v2 report](../benchmarks/svg_v2_results.md) and its linked raw trials |
| Parallel glyph generation | [Noumenon fan-out report](../benchmarks/noumenon_benchmark.md) and its 64- to 256-node records |
| Astra/Sol text comparison | [200-workflow report and review status](../benchmarks/results/astra_20260913_main/README.md) |
| Fable 5.1 extension | [Native and Code pilots](../benchmarks/results/fable_20260914_pilot/README.md) and [ten-task Code study](../benchmarks/results/fable_code_20260914/README.md); human pilot ratings gate the native main study |
| All campaigns and outstanding work | [Evidence index](../benchmarks/README.md) · [Retired evidence](../benchmarks/archive/README.md) · [Roadmap](../ROADMAP.md#coming-soon) |
| Public charts | [Monochrome SVG assets](../assets/benchmarks/) generated from committed records |

The glyph workflow compiles authored contours. The Astra/Sol and Fable
studies measure text workflows; they do not generate a replacement glyph set.
Model results alone do not require redrawing a contact sheet. A changed glyph
catalog requires new sheets, previews, exports, and matching artwork evidence.

## Completion check for every benchmark update

1. Reconcile all trials, failures, costs, unknown charges, judgments, and review
   status. Preserve the original input/output records and source identities.
2. Verify that current glyph IDs and SVG hashes agree across the catalog,
   contact sheets, browser data and selected specimens. If glyphs change,
   regenerate all affected 192/256 sheets and detail views, and update the
   catalog copy that Noumenon pins.
3. Recapture the Noumenon animation when its visible glyphs or effect change.
   Retain capture settings and source hashes; an animation's playback rate is
   not renderer performance evidence.
4. Run `python benchmarks/render_readme_charts.py`. Check every changed chart
   against its committed inputs, evidence status, units, denominators, and
   uncertainty. Preserve the black-and-white chart style. Promote only reviewed,
   claimable campaigns that have no known measurement defects.
5. Update the README, documentation map, roadmap, unreleased changelog, examples,
   benchmark index and affected subsystem guides together.
   Link current showcase images directly; label historical evidence at entry.
6. Validate Markdown and HTML asset links and inspect the contact sheets and
   changed charts. Run code checks appropriate to any implementation changes.
7. Push the reviewed changes and verify GitHub serves the intended files and
   commit. Use the [release checklist](../RELEASING.md) for an actual release;
   completing a benchmark does not itself publish a package version.

The 13 September materials check and other retired records are in the
[evidence archive](../benchmarks/archive/README.md), byte for byte, so earlier
measurements can still be reproduced.
