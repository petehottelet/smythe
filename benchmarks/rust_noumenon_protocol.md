# Rust / Python Noumenon comparison v1

This offline protocol compares the native preview with Python Smythe on the
same 192 approved contours and 256-glyph extension. It generates fresh SVGs;
no provider network calls, simulated delays or finished-artifact caches occur.

## Matched compilation timing

Use Rust release mode, Python threads and Python processes at 1, 4 and 8
workers, both catalog sizes, and three repetitions: 54 trials. Shuffle the
complete schedule with seed 20261107 before execution. Every trial constructs
one broadcast graph and compiles one glyph per node through Smythe. The clock
includes graph construction, pool startup and execution. Parent imports and input
loading, pool shutdown, raster validation, file export, durable receipts and
independent verification are excluded on both sides. Python process startup
includes its child imports. Each implementation uses its own Swarm payload
representation: JSON envelopes in Python and typed values in Rust. This measures
the compile-and-dispatch path, not an isolated geometry function or a warm pool.
Keep medians and ranges for all cells;
compare identical sizes and worker counts, with both Python backends visible.

Native output must be byte-for-byte identical to freshly compiled Python SVGs.
All hashes must match across trials and worker counts, and the 192 originals
must match the extension's prefix. Missing or duplicate trials, source changes,
invalid clocks or output changes invalidate the comparison. Embed compiled Rust
source hashes in the binary and compare them to the checkout before admitting
a native trial. Freeze input/source hashes, Cargo.lock and binary SHA-256.

## Full workflow diagnostics

At the largest declared worker count, run three complete native workflows and
three existing Python v2 workflows at each size. Keep their clocks separate:
the native renderer uses tiny-skia, checks topology and distinctness, and writes
durable batches; Python uses pinned Pillow and additionally computes detailed
style measurements. These are different workloads. **Do not compute a complete
workflow speedup from these clocks or compare them to historical timings.**

Independently run the original Python SVG, four-size measurement and all-pairs
distinctness gates on every native catalog outside timing. Verify completed
native resume dispatches zero provider calls. Native pixel hashes are versioned
separately from Pillow hashes; matching source SVGs does not imply matching
anti-aliased edge pixels. The Python artwork and its historical records remain
unchanged. This campaign establishes a local native preview and is diagnostic;
it does not establish live generation, remote provider, universal recovery,
screensaver FPS, cross-platform or complete framework parity.

## Reproduction

```bash
cargo build --release --locked
python benchmarks/run_rust_noumenon.py --out smythe_artifacts/rust-comparison
```

Use `--smoke` for four glyphs and one repetition. Output directories must be new.
The Python environment needs the project's glyph dependencies. Finish owned
tests, builds and other benchmarks before timing; existing desktop applications
may remain open and must be reported. No idle-host claim follows. Store all
trial records, source identities and validation receipts; never overwrite an
earlier campaign or promote diagnostic numbers into README headlines.
