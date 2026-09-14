# Authored SVG workflow v2: 192 and 256 glyphs

This protocol times the current contour method through Smythe. The 256-glyph
set preserves the first 192 SVGs exactly and adds 64 authored structures. The
active screensaver remains on the reviewed 192-glyph catalog.

## Frozen experiment

- Catalog sizes: **192 and 256**.
- Local backends: **thread and process**; requested concurrency **1, 4, 8**.
- Three repetitions per cell: **36 complete workflows**.
- Worker capacity: minimum of requested concurrency, eight, and logical CPUs.
- Seeded shuffled order within each repetition block; seed **20260913**.
- Fresh graph, provider, pool, and output folder per workflow. One independent
  Smythe task per glyph, zero retries, no planning model, no simulated latency.
- Write the schedule and actual source hashes before sampling. Check those
  hashes before and after every trial. Preserve every attempt.

## Included work

The total clock starts before graph, provider, resource sampler and worker-pool
setup. Each actual provider call constructs Shapely geometry from the fixed
outline recipe, samples cubic curves, unites components, orients counters,
serializes restricted SVG, and passes JSON back through Smythe.

Validation parses every SVG, rasterizes and measures it at **16, 32, 64 and
128 px**, checks component/counter stability and 128 px threshold sensitivity,
and verifies nonblank opaque output. Every catalog compares all normalized,
aligned and reflected silhouette pairs at **0.85 IoU**: **18,336 pairs** for
192 glyphs and **32,640 pairs** for 256. Exact and near duplicates fail.

Every successful workflow writes its SVGs, manifest, validation report and
128 px raster atlas. Pool shutdown and resource-sampling overhead remain
inside the total. Phase clocks cover setup, compilation, validation, assembly,
and shutdown; remaining measured overhead stays in the total.

Parent CPU time is reported once. Process worker task CPU is separate and
excludes worker startup/shutdown CPU. Thread task CPU is not added to the
parent process clock. When available, a 20 ms sampler records parent-plus-
descendant RSS; shared pages are counted per process and brief peaks can be
missed. Unavailable memory measurements are null, never zero.

## Scope and exclusions

The recipes are authored and selected before the campaign. **Design research,
candidate selection, and creative quality are not timed.** The new method
compiles complete authored contours; it does not invent the designs during
the measured provider call. No finished SVG or raster is used as a generation
cache. CLI/driver imports, environment capture, aggregate receipt publication,
labeled review sheets, browser textures and README assets are outside the timer.

This is a local CPU workload with **zero API calls and $0 provider API charges**.
Hardware, electricity and design labor are unpriced. It does not measure Astra,
image-generation quality, screensaver FPS, GPU time or native-port performance.
The v1 grammar and validation gates differ, so no v1-to-v2 speedup is claimed.

Run timing on the existing desktop after owned tests and design-selection work
finish. Existing desktop apps may remain open; record that condition without
an idle-host claim. Do not run other owned benchmarks, tests or builds during
the campaign.

## Evidence rules

All scheduled trials must finish successfully. Counts must match the requested
size; SVG, raster and full measurement hashes must match across repetitions
and concurrency settings. The first 192 hashes must match between catalog
sizes. A source change, missing trial, duplicate trial, partial catalog,
duplicate silhouette, failed export or inconsistent clock disqualifies claims.

Report all repetitions, medians and ranges. Speedup uses the complete-workflow
median against concurrency one of the same backend and size. Select the lowest
median among declared configurations, retaining overlapping ranges. Throughput
is catalog size divided by that median. Phase illustrations use one actual
median run, not a sum of independently selected phase medians. These checks
establish reproducibility and completion; they are not an aesthetic score.

The runner refuses existing output folders. A smoke campaign is diagnostic.

```powershell
pip install -e ".[glyphs]" psutil==7.2.2
python benchmarks/run_svg_v2_benchmark.py --out smythe_artifacts/svg_v2_reproduction
```

For a four-glyph plumbing check, add `--smoke`. Results are written to the new
folder with its frozen plan, individual receipts and assembled per-run catalogs.
