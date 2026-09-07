# Original SVG glyph workflow benchmark

This protocol measures fresh procedural SVG creation through Smythe, complete validation, and delivery of a usable catalog. It is separate from the historical 192-glyph stroke benchmark. Historical records and their simulated-latency results are unchanged.

The new generator authors original geometry. Reference glyph images inform aggregate style measurements; source images, copied paths, and imported character outlines are not benchmark inputs or deliverables. The web explorer separately imports the reference character set and mixes these 192 originals into it. That display composition and its derived GPU textures are outside this generation benchmark.

## Results: 7 September 2026

**192 original SVGs generated, fully validated, and assembled in 4.03 seconds
median.** The selected configuration admits 16 Smythe calls with eight process
workers. It is **2.95× faster** than process c1's 11.88-second median.
All 30 workflows completed all 192 nodes with accepted, unique glyphs and
identical SVG, pixel, and measurement hashes. The campaign is **claimable for
this local procedural workflow**, with no known measurement defects.

[Raw record](results/glyph_svg_v1.json) ·
[Complete contact sheet](partitions/glyph_svg_v1/catalog/contact-sheet.png) ·
[SVGs and manifest](partitions/glyph_svg_v1/catalog/).

![Complete SVG workflow and stage breakdown](../assets/benchmarks/svg_workflow.svg)

Each cell below is the median of three complete runs. Phase medians are
independent summaries; their sum need not equal the median complete workflow.

| Backend | Requested concurrency | Workers | Generation | Validation | Assembly | Complete workflow |
|---|---:|---:|---:|---:|---:|---:|
| Thread | 1 | 1 | 1.32 s | 9.57 s | 1.12 s | 11.65 s |
| Thread | 4 | 4 | 0.76 s | 9.98 s | 0.94 s | 12.20 s |
| Thread | 8 | 8 | 0.71 s | 10.16 s | 0.68 s | 11.70 s |
| Thread | 16 | 8 | 0.70 s | 10.49 s | 0.61 s | 11.81 s |
| Thread | 32 | 8 | 0.67 s | 10.30 s | 0.79 s | 11.42 s |
| Process | 1 | 1 | 1.62 s | 9.61 s | 0.57 s | 11.88 s |
| Process | 4 | 4 | 0.84 s | 3.10 s | 0.66 s | 4.64 s |
| Process | 8 | 8 | 0.90 s | 2.45 s | 0.56 s | 4.03 s |
| **Process** | **16** | **8** | **0.89 s** | **2.43 s** | **0.54 s** | **4.03 s** |
| Process | 32 | 8 | 0.89 s | 2.50 s | 0.54 s | 4.08 s |

Process c8 and c16 have overlapping ranges and differ by only 0.002 seconds
in their medians. C16 is the mechanical lowest-median selection; the record
does not establish a meaningful advantage over c8. The main measured gain
comes from parallelizing the complete validation work with process workers.

The published catalog comes from **process c16, repetition 3**, the actual
median run. Its measured stages are 0.8680 s generation, 2.4756 s validation,
0.5444 s assembly, and 0.1411 s setup/shutdown/remaining overhead. Complete
wall time is 4.0292 s. Parent CPU time was 1.5781 s; worker tasks used 13.0313
CPU seconds, excluding worker startup and shutdown. Process memory was not
measured because `psutil` was unavailable. The assembled run wrote 2,869,677
bytes; publication adds labels and provenance outside the workflow timer.

The host ran Windows build 26200, Python 3.11.9, Pillow 11.1.0, NumPy 2.2.6,
SciPy 1.15.2, and Shapely 2.1.2, with 32 logical CPUs and an eight-worker cap.
The record identifies imported Smythe source 0.6.0 and its edited checkout;
installed package metadata still said 0.2.0. Source hashes bind the actual
generator, measurement method, harness, reference summary, and optical review.

The complete campaign made **0 API calls and incurred $0 provider API charges**.
This does not price local hardware, electricity, or design calibration.

### Shape acceptance and separate rendering

All 19 normalized shape metrics pass the two profile gates. All 18,336
aligned/reflected comparisons pass the 0.85 near-match threshold, with no
duplicate silhouettes. The [size audit](partitions/glyph_svg_v1/size-review.json)
and [AI optical review](partitions/glyph_svg_v1/optical-review.json) cover all
192 glyphs at 16/32/64/128 pixels. Raster edge sensitivity remains a separate
diagnostic outside the reference band; see the
[full acceptance report](partitions/glyph_svg_v1/catalog/style-acceptance.json).
The [calibration history](partitions/glyph_svg_v1/calibration-history.json)
records earlier design candidates outside the timed campaign.

The separate 192-glyph static rasterization study recorded median batch times
of **0.546 s at 64px**, **0.979 s at 128px**, and **10.961 s at 512px**.
All three repetitions at each size produced identical pixel hashes. These are
static rasterization and hashing times, not animation frame rates.
[Browser measurement and navigation checks](../screensaver/svg-preview/README.md#renderer-and-checks)
have separate receipts. The [first renderer's archived measurements](partitions/glyph_svg_v1/renderer-v1/README.md)
cover its original-only Canvas implementation; they do not measure the revised
mixed-glyph WebGL renderer.

## Run

Install the repository's glyph dependencies, then run from the repository root:

```bash
pip install -e ".[glyphs]"
python benchmarks/run_svg_glyph_benchmark.py \
  --out smythe_artifacts/svg_glyph_v1_reproduction \
  --optical-review benchmarks/partitions/glyph_svg_v1/optical-review.json
```

The default campaign creates 192 glyphs for each combination of:

- Requested Smythe concurrency: **1,4,8,16,32**.
- Local execution backend: **thread** and **process**.
- Repetitions: **3 per configuration**, 30 complete workflows total.

Each run uses a fresh graph, provider, worker pool, and output directory. Worker capacity is `min(requested concurrency, worker cap, logical CPU count)`; the default worker cap is 8. Thus a c32 run on this setting can admit 32 provider calls while at most 8 workers execute glyph work. The record reports requested concurrency, configured workers, observed workers, and maximum in-flight provider calls separately.

A bounded smoke run exercises the workflow without becoming headline evidence:

```bash
python benchmarks/run_svg_glyph_benchmark.py --glyphs 24 --repeats 1 \
  --concurrencies 1,4 --executors thread --render-sizes 128 \
  --out smythe_artifacts/svg_glyph_v1_smoke
```

For a process-only campaign, pass `--executors process`. For another worker ceiling, pass `--worker-cap N`. Every sweep requires a c1 baseline.

Default raw output is ignored `smythe_artifacts/svg_glyph_v1/`, with a separate catalog/atlas/validation directory for every repeat and configuration. The default aggregate record is `benchmarks/partitions/glyph_svg_v1/results.json`. A custom `--out` places `results.json` there unless `--results` explicitly selects another location. The repeated artifact directories are local evidence and need not be committed. Publish the accepted uppercase `GLYPH-*` catalog once at its canonical location, separately from these repeated timing outputs; compare its SVG hashes with the result receipts. Existing output is rejected before work begins. Choose another partition to preserve a prior campaign. `--overwrite` explicitly allows replacement of the selected output paths; the harness never deletes historical records or changes the historical glyph runner.

## What executes

1. Build a real `ExecutionGraph` containing one independent node per glyph. Every node calls the local `SVGGenerationProvider` through `Swarm.execute_async`, with Smythe's budget accounting and bounded executor.
2. Inside each timed provider call, run `generate_glyph(index, attempt=0)` and serialize its new SVG payload. No completed glyph is generated before this boundary, loaded from a prior catalog, or served from a benchmark cache. No artificial latency is added.
3. Validate every completed output in a separately timed phase. `validate_svg` must explicitly pass with an empty error list. Render at 128×128, require opaque pixels and a nonblank measured silhouette, check ink/background and uniqueness, and run the complete `measure_glyph` method. Evaluate catalog profile gates and all-pairs aligned/reflected near matches against the committed numeric style brief. These measurements and comparisons are inside the timer.
4. Assemble per-glyph SVG files, a catalog manifest with hashes/seeds/profiles, a complete validation report, and a PNG atlas. Assembly happens for every successful run, so each comparison includes a delivered catalog.
5. Close the worker pool and collect resource measurements. Failed independent nodes do not prevent other nodes from being attempted. Execution, structural-validation, uniqueness, and assembly failures disqualify timing evidence. Numeric style gaps remain visible in their separate acceptance report; they do not turn a valid elapsed-time measurement into a style achievement.

The protocol uses one attempt per index. A structurally invalid candidate is recorded as a failure. A valid SVG with unmet style targets remains in the measured catalog and its failed style gates remain visible. It does not substitute a cached accepted glyph or remove failed work from the measured cost. A future regeneration protocol must version the protocol and account for every additional attempt.

## Timing and resource accounting

| Field | Included work |
|---|---|
| `setup_wall_s` | Resource sampler, graph, provider, swarm, and pool construction |
| `generation_wall_s` | Actual Smythe execution, worker startup/imports, fresh SVG creation, provider serialization, execution traces, and node finalization |
| `validation_wall_s` | Dispatch, parsing, structural validation, 128px rasterization, full style measurement, profile gates, aligned/reflected distinctness, hashing, and result transfer |
| `assembly_wall_s` | SVG file writes, catalog JSON, atlas construction, PNG encoding, and output hashes |
| `worker_shutdown_wall_s` | Pool shutdown and worker cleanup |
| `end_to_end_wall_s` | The complete sequence above, including resource-sampling overhead |

Driver library imports, CLI parsing, environment capture, and final aggregate-record serialization occur outside the workflow timer. There is no pre-generated glyph warmup. Each run includes its own worker startup and shutdown, so the comparison represents a complete bounded job rather than a permanently running service.

The authored grammar and its per-index shape parameters are frozen before the
campaign. Reference research, design calibration, optical review, contact-sheet
labeling, browser-data publication, and documentation are outside the timer.
Every measured call constructs new contours from those fixed parameters;
this is a local procedural workload, with no generative model in the loop.

Worker-call wall durations are retained as diagnostics. Their sum is not an end-to-end duration because calls overlap. For process workers, task CPU time is measured inside each worker and reported separately from parent-process CPU; worker startup/shutdown CPU is outside that task-CPU subtotal. For threads, the aggregate CPU figure is the parent process's complete CPU time. Thread task diagnostics use `thread_time_ns`; they are not added to the parent total. Summing process-wide CPU clocks inside overlapping threads would double-count work.

When `psutil` is available, a sampler records peak aggregate RSS across the benchmark process and its descendants at 20 ms intervals. This is a sampled peak, can miss shorter spikes, and counts shared pages in each process. If the sampler is unavailable, the record explicitly says memory was not measured. It never substitutes zero memory.

The execution backends perform local CPU work. They make **zero provider API calls**, use **zero API tokens**, and incur **$0 provider API charges**. Hardware, electricity, and infrastructure are unpriced; this is not a claim that local computation has no cost.

## Separate rasterization study

After a successful workflow campaign, rasterize the accepted SVG strings sequentially at 64, 128, and 512 pixels, with the same repetition count. Record wall time, CPU time, and a combined pixel hash for each batch. Repeated hashes must match.

These accepted SVG strings are inputs to a clearly separate rendering measurement. Their reuse is not counted as new generation. This test measures static SVG-to-pixel work plus hashing. It does **not** measure screensaver frame rate, browser composition, camera movement, GPU cost, or native presentation. Real-time renderer measurements require their own controlled viewport, glyph density, animation duration, and frame-time protocol.

## Comparison and evidence status

The run order rotates and alternates direction across repetitions. Every configuration generates the same indices and attempt seeds. A configuration summary includes median, minimum, and maximum generation, validation, assembly, and end-to-end times. Speedup uses the median complete workflow against c1 of the same backend. The selected workflow has the lowest median end-to-end time across the tested backends and concurrency settings. All samples remain visible; a single favorable run never becomes the headline.

A timing record is claimable only when every run succeeds, at least 3 complete repetitions exist per configuration, every output is structurally valid and unique, all repeated/concurrency SVG/pixel/measurement hashes agree, and any requested rasterization repetitions are deterministic. Missing work, generation failures, structural-validation failures, duplicate shapes, assembly errors, or cross-run differences keep timing evidence diagnostic. Duplicate shapes include exact normalized, reflected, and aligned silhouettes; near matches require optical review in the separate style gate. The claim scope remains this local procedural workload and measured host; it cannot establish model-quality superiority, API throughput, or real-time animation performance.

Timing eligibility and style achievement have separate fields: `timing_claimable`, `catalog_style_accepted`, and `readme_promotion_eligible`. README promotion requires valid timing evidence **and** a complete, accepted 192-glyph catalog. A smaller calibration campaign can measure elapsed time but cannot satisfy the finished-catalog gate.

Pass `--optical-review review.json` only after examining the actual catalog. The review must contain `passed: true`, any `reviewed_near_pairs` required by the style evaluator, and `glyph_svg_sha256`, a mapping from every glyph ID to its exact SVG SHA-256. The harness rejects a review whose mapping differs from the freshly generated catalog. Without completed numeric/distinctness gates and this separate, matching optical review, style acceptance remains pending.

The record includes protocol version, generator/measurement/harness/reference SHA-256, exact environment and imported Smythe identity, source-control state, seeds, per-call success/failure, glyph/atlas/catalog hashes, execution identifiers, and resource scope. Source hashes identify edited-checkout measurements without presenting them as a clean committed revision.

## Publish the measured catalog

Select one completed run from the recorded campaign. Publishing preserves its exact SVG bytes and rechecks every SVG hash, opaque 128px raster hash, validation identity, measurement hash, and atlas pixel. It never invokes the generator.

```bash
python benchmarks/publish_svg_catalog.py \
  --source smythe_artifacts/svg_glyph_v1/runs/r01-thread-c1 \
  --destination benchmarks/partitions/glyph_svg_v1/catalog \
  --results benchmarks/results/glyph_svg_v1.json \
  --preview screensaver/svg-preview/glyphs.js
```

Use the actual source directory and aggregate record from the campaign; the example paths do not select a result automatically. `--results` verifies that exactly one passed run identifies these source artifacts and binds its path, hash, execution ID, and configuration in the published manifest. Without it, the manifest still records the source directory and individual artifact hashes.

Publication contains 192 uppercase `GLYPH-*.svg` files, a labeled black-on-white contact sheet, a first-24 calibration sheet, numeric measurements, unchanged style acceptance status, and a manifest. The browser data contains these exact path strings and the SHA-256 of the published manifest. Individual PNGs are optional via `--individual-pngs`. Existing outputs require `--overwrite`; source artifacts are never overwritten. Publication does not upgrade diagnostic timing or incomplete style acceptance into a successful claim.

## Evidence status

The committed campaign passes timing eligibility, complete-catalog style
acceptance, and README promotion. The displayed 4.03-second result includes
fresh geometry, full validation, and a delivered catalog. It does not measure
LLM creativity, reference research, initial design work, native rendering,
browser presentation, or provider API throughput.
