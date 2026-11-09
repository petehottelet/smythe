# Rust Noumenon preview: local comparison

**Evidence status: diagnostic native preview; not a README performance headline.**

The native pipeline preserves the approved 192 glyphs and the 256-glyph extension.
All 54 fresh compilation trials produced SVG hashes identical across Rust and
both Python backends. Every native SVG was also compared byte-for-byte with a
fresh Python compilation. The original 192 glyphs remain the extension's prefix.

[Protocol](rust_noumenon_protocol.md) · [Raw record](results/rust_noumenon_20261107.json) ·
[Harness](run_rust_noumenon.py) · [Native guide](../docs/rust.md)

## Matched compilation and dispatch

Milliseconds, median of three runs, with observed minimum–maximum in parentheses.
Each cell includes fresh graph creation, worker startup and compilation through
its implementation's Swarm. Python processes include child imports. Parent imports,
shutdown, raster validation, durable receipts and exports are outside this clock.
The Python JSON envelope and Rust typed payload are part of their respective paths.
This is not an isolated geometry-function benchmark or a warm-pool comparison.

| Glyphs | Workers | Rust | Python threads | Python processes |
|---:|---:|---:|---:|---:|
| 192 | 1 | 49.70 (49.69–52.32) | 310.11 (308.67–318.75) | 906.72 (903.39–928.06) |
| 192 | 4 | 37.25 (35.41–37.62) | 338.88 (328.82–355.62) | 689.70 (671.09–705.71) |
| 192 | 8 | 26.87 (25.65–27.46) | 338.07 (325.01–349.60) | 784.72 (774.30–795.81) |
| 256 | 1 | 60.46 (59.73–60.83) | 411.49 (406.24–420.70) | 1012.38 (1010.12–1038.13) |
| 256 | 4 | 46.85 (44.76–46.94) | 448.47 (441.60–456.24) | 714.57 (703.60–733.73) |
| 256 | 8 | 33.73 (31.96–34.02) | 437.68 (436.74–446.29) | 782.35 (772.99–807.32) |

## Full workflows: separate diagnostics

**These clocks measure different workloads. Do not calculate an end-to-end
Rust/Python speedup from them.** Native full mode uses tiny-skia, native topology
and distinctness gates, durable batches, SVGs, a catalog and an atlas. Python v2
uses Pillow, additionally computes detailed style measurements, and exports its
existing catalog. Native CLI time includes process startup, output and terminal
report publication; the original Python clock measures its workflow function.
All full trials use eight workers. Seconds, median of three runs:

| Glyphs | Native CLI | Original Python v2 workflow |
|---:|---:|---:|
| 192 | 2.973 | 3.911 |
| 256 | 2.992 | 4.750 |

All six native catalogs independently passed the original Python four-size
measurement/topology gates and all-pairs distinctness gate outside the measured
intervals: 18,336 pairs for 192 glyphs and 32,640 for 256. Each completed resume
made zero provider calls. Native raster hashes use a separate renderer version;
matching SVG bytes does not imply identical anti-aliased edge pixels.

## Environment and limits

- Windows 11, AMD64, 32 logical CPUs.
- Python 3.12.10; rustc 1.91.0 (f8297e351 2025-10-28); cargo 1.91.0 (ea2d97820 2025-10-10).
- Rust release profile, thin LTO, one codegen unit; exact dependencies in Cargo.lock.
- The record freezes source hashes, input contours, binary SHA-256 and the randomized schedule.
- No API calls, synthetic provider latency or finished-glyph caches in measured trials.
- Existing desktop applications remained open; no idle-host or cross-platform performance claim.
- Owned tests and builds completed before the measured campaign.
- Complete framework parity, remote-provider recovery and screensaver FPS are outside this campaign.
- Historical Noumenon timings use different environments and are not comparison baselines.
