# Source builds and distribution

Smythe **0.9.0** is a Rust source preview. The workspace packages have
`publish = false`; there is no current crates.io package or Rust binary release
to install. Build from a checkout of `main` or an exact commit. Historical PyPI
packages contain the retired Python implementation.

## Choose a source

| Need | Source |
|---|---|
| Current Rust runtime and CLI | This repository's `main` branch; record the exact commit when reproducing behavior |
| Original Rust 0.2.0 snapshot | The immutable `rust-v0.2.0` tag; later fixes are on `main` |
| Historical Python implementation | [Revision 12f1470](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce) and the [historical packaging guide](https://github.com/petehottelet/smythe/blob/12f14704beff344228fffffce40d995daa1f7cce/docs/distribution.md) |
| Noumenon screensaver | The separate [Noumenon repository](https://github.com/petehottelet/noumenon) |
| Repo Doctor | The separate [repodoctor repository](https://github.com/petehottelet/repodoctor) |

## Build the CLI

Use Rust **1.91 or later** and a C build toolchain for bundled SQLite. From the
repository root:

```bash
cargo fetch --locked
cargo build --release --locked --offline -p smythe-cli
cargo run --release --locked --offline --bin smythe -- --help
```

`cargo fetch` downloads dependencies. Subsequent `--offline` commands require
that cache. The executable is `target/release/smythe` on Unix and
`target/release/smythe.exe` on Windows. No Python environment or API key is
required. Linux and Windows are the platforms covered by the current CI matrix.

Run a local graph, then inspect its journal:

```bash
cargo run --release --locked --offline --bin smythe -- run --graph examples/native_graph.toml --id example --store smythe_artifacts/native-runs
cargo run --release --locked --offline --bin smythe -- inspect example --store smythe_artifacts/native-runs
```

Use a fresh run ID for new work; use the same graph, ID and store with
`run --resume` to recover it. See the [native guide](rust.md) for controls,
output retrieval and the 256-glyph Noumenon example.

## Reproduce a build or measurement

Keep `Cargo.lock` with the source and record the exact Git revision and Rust
toolchain. Use `--locked` to reject dependency resolution changes. A version
label alone does not identify later fixes on `main`.

The [native measurement runner](rust.md#verification-and-native-measurements)
requires Git metadata and a clean committed tree, hashes committed blob bytes,
and records the built binary. An extracted source archive can supply build
inputs but does not meet that runner's provenance contract without the original
Git history. LF/CRLF conversion does not change committed source hashes.

## Compatibility and publication

Rust APIs and journal formats remain experimental. Schema-3 runs require fresh
directories when moving from schema 1 or 2. Python checkpoints and Jobs stores
are not Rust run stores. Keep old data with the version that wrote it.

The historical `v0.2.0` tag belongs to Python. The Rust `rust-v0.2.0` tag remains
unchanged, and stable Rust **1.0.0** is still ahead. Pushing source to GitHub
does not create a package or GitHub Release. See [Releasing](../RELEASING.md)
for the separate verification and publication requirements.

Keep generated stores, credentials and local outputs outside distributions.
The [MIT license](../LICENSE) and copyright notice accompany the software.
Run the [contributor checks](../CONTRIBUTING.md#development-setup) before
proposing implementation changes.
