# V2 SVG catalog benchmark: 192 and 256 glyphs

**256 glyphs compiled, validated and exported in 8.06 seconds
median**, using process execution at concurrency 8.
That is **2.20×** its same-backend concurrency-one baseline.
The best 192-glyph configuration completed in **6.66 seconds median**.

All **36 workflows passed**: 18 at each size, with three repetitions per
backend/concurrency cell. Every workflow delivered its complete catalog;
SVG, pixel and four-size measurement hashes match across all repetitions.
The first 192 hashes also match between sizes and the current reviewed catalog.

**Evidence status: claimable for this local authored-contour workflow, with no
known measurement defects.** This compiles fixed authored designs. Design
research, candidate selection and creative work happen before the timer.
No model generates artwork during the measured calls.

[Frozen protocol](svg_v2_protocol.md) · [All raw trials](results/glyph_svg_v2_20260913.json) ·
[256-glyph sheets and SVGs](partitions/glyph_svg_v2_256/README.md) ·
[Archive verification](partitions/glyph_svg_v2_256/archive.json).

![All 36 workflow measurements](../assets/benchmarks/svg_v2_workflow.svg)

## Every tested configuration

| Glyphs | Backend | Concurrency | Median | Min–max | Speedup vs same-backend c1 | Glyphs/s | Median sampled peak RSS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 192 | Process | 1 | 13.691 s | 12.694–14.469 s | 1.00× | 14.02 | 240.5 MiB |
| 192 | Process | 4 | 6.659 s | 6.193–7.307 s | 2.06× | 28.83 | 469.0 MiB |
| 192 | Process | 8 | 7.053 s | 6.062–7.293 s | 1.94× | 27.22 | 774.3 MiB |
| 192 | Thread | 1 | 16.546 s | 16.201–17.004 s | 1.00× | 11.60 | 162.0 MiB |
| 192 | Thread | 4 | 25.127 s | 23.215–25.352 s | 0.66× | 7.64 | 159.6 MiB |
| 192 | Thread | 8 | 25.142 s | 24.775–28.365 s | 0.66× | 7.64 | 163.7 MiB |
| 256 | Process | 1 | 17.690 s | 17.483–18.336 s | 1.00× | 14.47 | 250.7 MiB |
| 256 | Process | 4 | 8.704 s | 8.532–11.305 s | 2.03× | 29.41 | 482.7 MiB |
| 256 | Process | 8 | 8.058 s | 7.292–11.155 s | 2.20× | 31.77 | 789.2 MiB |
| 256 | Thread | 1 | 22.461 s | 20.969–22.979 s | 1.00× | 11.40 | 172.8 MiB |
| 256 | Thread | 4 | 33.485 s | 33.060–33.917 s | 0.67× | 7.65 | 171.0 MiB |
| 256 | Thread | 8 | 34.128 s | 33.008–35.253 s | 0.66× | 7.50 | 179.0 MiB |

Each speedup uses concurrency one of the same backend and glyph count.
The lowest median is selected from the declared configurations; every range
and repetition remains visible. Comparing the best settings at each size,
256 glyphs are 33.3% more outputs in 1.21× the time. This is a
catalog-size observation, not a scaling law or a claim about all workloads.

## What the clock includes

One independent Smythe graph node compiles each glyph through a real local
provider call. The complete clock includes graph/pool setup, geometry
construction and SVG serialization, validation at 16/32/64/128 px, all-pairs
aligned/reflected comparisons, SVG/manifest/report/atlas export, worker
shutdown and resource sampling. There is no simulated latency or SVG cache.

The 192-glyph catalog compares 18,336 pairs per workflow; the 256-glyph
catalog compares 32,640. Both use the same 0.85 IoU threshold, small-size
component/counter stability and 128 px threshold-sensitivity checks. All
gates pass. These checks establish valid, stable, distinct artifacts within
the catalog; they do not score aesthetic quality or universal originality.

The first 192 authored SVGs are unchanged. The additional 64 use the same
cut terminals, broad strokes and deliberate spacing. The Noumenon
screensaver's live catalog remains 192 originals; the extension is a separate
benchmark and review artifact.

## Resource tradeoff

![Measured process-tree memory](../assets/benchmarks/svg_v2_memory.svg)

Process workers shorten completion time and require more memory. RSS is
about **483 MiB at process c4** for the 256-glyph set, with an **8.70-second
median**, versus **789 MiB and 8.06 seconds at c8**. Their observed timing
ranges overlap; c8 has the lowest median, while c4 uses less sampled memory.

RSS is sampled every 20 ms across the parent and descendants; it counts shared
pages in each process and can miss brief peaks. This is not GPU memory.
The raw records separately retain parent CPU and process-worker task CPU;
worker startup/shutdown CPU is excluded from the latter. Thread task CPU
is not added to the parent clock.

All runs made **zero API calls and incurred $0 provider API charges**.
Hardware, electricity and design labor are unpriced. No Astra, image-model
quality, API-throughput, GPU, animation-FPS or cross-platform claim follows.

## Host and provenance

Windows build 26200; AMD Ryzen 9 5950X, 16 cores / 32 logical CPUs; Python
3.11.9, Pillow 11.1.0, NumPy 2.2.6, SciPy 1.15.2, Shapely 2.1.2 and psutil
7.2.2. The worker cap was eight. Existing desktop apps remained open;
owned tests, design selection, builds and other benchmarks finished before
sampling. This is an active-desktop measurement, not an idle-host claim.

The measured checkout is edited from `8663cdf`, importing Smythe source
0.7.0; installed distribution metadata still identifies 0.2.0. The frozen
plan hashes the actual harness, generator, protocol and runtime files.
All hashes match before and after every trial. The protocol fixes the
36-trial schedule before sampling with seed 20260913.

[Measured source](partitions/glyph_svg_v2_256/measured-source.zip) preserves
those exact files. [Measured catalogs](partitions/glyph_svg_v2_256/measured-catalogs.zip)
contains both actual selected median-run exports. Every timed SVG and all
108 catalog/atlas/validation bindings were rehashed before this report.
All 8,064 per-glyph trial receipts are retained in the raw aggregate.

The [v1 campaign](svg_glyph_benchmark.md) uses a different grammar and
validation boundary. Its records remain historical; no v1-to-v2 speedup is
claimed. Reproduce this experiment with the command in the
[v2 protocol](svg_v2_protocol.md), choosing a new output folder.
