# Contributing to Smythe

Smythe's active implementation is Rust. The current workspace is a 0.2.0 source
preview; stable Rust 1.0.0 remains ahead. Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
and report sensitive issues through [private security reporting](SECURITY.md).

## Development setup

Install Rust 1.91 or later and a C build toolchain for bundled SQLite, then:

```bash
cargo fetch --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --release --locked --offline
```

The full offline suite includes deterministic scheduling simulation and real
process-kill recovery. Paid calls do not belong in tests. Linux and Windows CI
run the suite and the explicit [release performance gates](docs/rust.md#verification-and-native-measurements).
Ruff is not applicable: active Python implementation and tests have been retired.

## Project conventions

Preserve the two product abstractions: task-specific execution graphs and their
durable execution envelope. Keep scheduling decisions in the pure coordinator;
clock, worker, control-watch and storage I/O belong in the shell. Add a meaningful
regression test for every defect. Keep public behavior, examples and documentation
aligned, and put unreleased changes under `[Unreleased]` in the changelog.
Released sections are immutable.

Report reproducible bugs and discuss substantial feature proposals in issues.
PRs should describe the behavior and relevant checks. Use descriptive branches
and commit messages without tool attribution. Preserve unrelated local work and
stage explicit task files.

## Evidence files

Keep retained benchmark records and existing evidence bytes intact. The Noumenon
GIF and catalog remain the primary benchmark example. Historical Python results
are not Rust performance claims. Reproduce old harnesses from
[source revision 12f1470](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce);
the original native comparison records identify revision 409d367.

New large binary evidence should have a retained source identity, size, SHA-256,
download location, backup and offline reproduction instructions. Checksums
detect damage; they do not recover missing bytes. Keep README artwork at its
existing paths. Private notes, credentials and local outputs must stay outside
commits and distributions.

The native measurement runner requires a clean committed tree and hashes Git
blob bytes. See the [measurement contract](docs/rust.md#verification-and-native-measurements)
before interpreting timings.
