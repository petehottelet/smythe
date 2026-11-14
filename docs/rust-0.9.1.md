# 0.9.1

Rust 0.9.1 fixes concurrent retry recovery and admission deadlines, and retains
native benchmark evidence when a trial fails. It remains an experimental source
preview. The release tag is `rust-v0.9.1`; the historical Python `v0.9.1` tag
identifies a different implementation.

## Changes

- Retrying an eligible failed node while siblings are active preserves their
  outputs, known costs and dependent readiness. Completed work is reused on resume.
- Admission decisions include time spent serializing and publishing outputs,
  committing the journal and receiving external controls. Previously committed
  admissions still drain; deadlines do not forcibly interrupt callbacks.
- Native campaigns write `trials.jsonl` incrementally with source and binary
  provenance, trial starts, elapsed time, exit outcomes and sanitized failure
  classes. Earlier evidence survives a failed trial. Final source or binary
  changes invalidate the campaign; results remain diagnostic and nonclaimable.
- Benchmark-index links to retired Python harnesses resolve to their pinned
  historical source. The roadmap explicitly includes storage lifecycle
  qualification and remote timeout, cancellation, preflight and retry ownership.

## Compatibility

Journal schema 3 and existing run bindings are unchanged from Rust 0.9.0.
Earlier schema-3 journals remain readable. Schema-1/2 runs require fresh
directories, and Python stores are not native run stores. No paid provider,
generated planner, tool, MCP, supervisor or Jobs integration is introduced.

## Build and verify

Build the `rust-v0.9.1` source with Rust 1.91 or later and a C build toolchain
for bundled SQLite:

```bash
cargo fetch --locked
cargo build --release --locked --offline -p smythe-cli
cargo run --release --locked --offline --bin smythe -- --version
```

The release provides source archives. Workspace packages remain unpublished;
there are no prebuilt native binaries or crates.io packages. Local examples
need neither Python nor API keys. See the [source-build guide](distribution.md)
and the [native checks and benchmark commands](rust.md#verification-and-native-measurements).

The [roadmap](../ROADMAP.md#local-qualification) lists remaining raster,
storage lifecycle, tooling and installation qualification. This patch release
does not establish beta or 1.0 readiness.
