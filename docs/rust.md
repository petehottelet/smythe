# Rust preview: Noumenon first

The native preview runs the existing authored Noumenon glyph pipeline without a
Python runtime. It is the first part of the Rust rewrite, not a replacement for
the complete Python framework. Both implementations remain in this checkout.
The preview crates are versioned independently and are not published packages.

## Build and run

Install Rust 1.91 or later and build from the repository root:

```bash
cargo build --release --locked
cargo run --release --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs
cargo run --release --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs --resume
```

The binary is named `smythe`; using `cargo run` keeps it distinct from the Python
CLI on your PATH. Counts from 1 through 256 and worker counts from 1 through 64
are accepted. The default is the approved 192-glyph set and four workers.
Each fresh run requires a new output directory. All work is local, with zero
API calls and zero provider cost. `--mode compile` exports SVGs and a catalog
without raster validation and is intended for the compiler benchmark.

Full mode exports `GLYPH-000.svg` onward, `catalog.json`, `validation.json`,
`atlas.png`, and `report.json`, including spans for this invocation's dispatched
nodes. The individual SVGs retain the Python artwork,
IDs, viewBox, winding and serialization. The atlas is an inspection image.
This command does not install or modify the separate Noumenon screensaver or
its live provider integration. Its catalog/report schema is versioned separately
from the Python benchmark; existing SVG consumers can use the individual files.

## Execution and acceptance

`smythe-core` retains `Task`, `Node`, `ExecutionGraph`, `Topology`, `Swarm`,
`Provider`, `CompletionResult`, `Sentinel` and `Tracer`. The local executor
validates the DAG before dispatch, resolves dependencies in graph order, bounds
CPU work using a Rayon pool and reserves integer nanoUSD before dispatch.
Workers receive immutable dependencies. A coordinator owns scheduling and cost
state, and waits for siblings before returning a failure. The current scheduler
uses bounded waves, so a slow sibling can delay the next wave.

`smythe-noumenon` implements `Verifier`, the authored contour provider and the
CLI. Every provider call compiles a fresh contour; no finished SVG cache is used.
Cubic curves use the same 64 samples as Python. Disjoint filled parts avoid a
general polygon union. Overlapping parts are rejected explicitly; arbitrary SVG
input, new artwork generation and general geometry editing are outside this path.

Full mode renders at 16, 32, 64 and 128 pixels with tiny-skia, rejects blank or
topologically unstable glyphs, and compares every pair after crop normalization,
bounded translation and reflection at the 0.85 overlap threshold. Native pixel
hashes have their own renderer version. They are not Pillow pixel hashes.
The comparison harness also runs the original Python four-size style and
topology measurements and distinctness gate against the native SVGs.

## Recovery boundary

A process lock fences a run directory. Each immutable checkpoint contains up to
16 glyphs, their SVG bytes and validation receipts. The file is flushed before
atomic hard-link publication; derived SVGs can be reconstructed from that batch.
The filesystem must support hard links. A killed process leaves completed batches
reusable and incomplete batches eligible for regeneration.

Resume checks the compiled implementation and contour identities, reconstructs
the expected authored glyphs, checks saved measurements, and refuses corrupt
existing artifacts. It can restore missing derived SVGs. Once a completed report
exists, missing or changed final exports fail closed. Completed replay dispatches
zero provider calls, but still performs integrity validation. Worker count may
change; mode, count, sources and implementation must match. Resume is intentionally
not compatible across modified builds. Unpublished `.pending-*` files from an
interruption are ignored; they are never treated as checkpoints.

This is local process-interruption recovery. It does not claim power-loss or
network-filesystem durability, distributed leases, or exactly-once external
effects. Python's SQLite provider journal, ambiguous-call accounting, tool loops,
WhiteRabbit planner, Supervisor, learning policies, artifact lineage, Jobs and
MCP integrations remain Python-only. The local `Provider` contract cannot safely
represent an unsuccessful remote call that might have incurred a charge.

## Verify and compare

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
python -m ruff check .
python -m pytest tests/ -q
python benchmarks/run_rust_noumenon.py --smoke --out smythe_artifacts/native-smoke
python benchmarks/run_rust_noumenon.py --out smythe_artifacts/native-comparison
```

Fetch dependencies with `cargo fetch --locked` before running offline Cargo checks.
The Python comparison requires `pip install -e ".[dev]"`. The Rust tests include
all 256 Python reference SVG hashes, graph and budget rejection, corrupted
receipts, exclusive ownership, completed replay, and an actual killed child
process that resumes after one committed batch.

The [comparison protocol](../benchmarks/rust_noumenon_protocol.md) times the same
compilation boundary at both catalog sizes, across Rust, Python threads and
Python processes. Complete workflow timings are separate diagnostics because
the renderers, validation work and persistence policies differ. They do not
establish a full-framework or complete-workflow speedup. The
[local diagnostic results](../benchmarks/rust_noumenon_results.md) include the full
54-trial record and six independently validated native catalogs.
