# Rust alpha: journaled local execution

The native workspace is `0.2.0`. It runs local execution graphs and the
existing authored Noumenon glyph pipeline without Python. The Python 0.10 line
remains in this checkout while the native product is developed. These Rust
packages are unpublished and their interfaces are experimental.

The [README quickstart](../README.md#quickstart) is the native entry point.
For the existing model-driven workflows, see the
[Python 0.10 quickstart](python-quickstart.md) and
[Python examples](../examples/README.md#python-examples). Their planning,
provider, tool, supervision, and Jobs APIs are not Rust APIs.

## Build and run

Rust 1.91 or later and a C build toolchain for bundled SQLite are required.
Fetch dependencies once, then build offline:

```bash
cargo fetch --locked
cargo build --release --locked --offline -p smythe-cli
cargo run --release --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs
cargo run --release --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs --resume
```

The binary is named `smythe`; using `cargo run` distinguishes it from the Python
CLI on PATH. Noumenon accepts 1–256 glyphs and 1–64 workers, defaulting to 192
and four. A fresh run requires a new output directory. All current native
providers perform local work with zero API calls. `--mode compile` omits raster
validation. Full mode publishes SVGs, `catalog.json`, `validation.json`,
`atlas.png` and `report.json`.

An ordinary graph uses the same engine:

```bash
cargo run --release --bin smythe -- run --graph examples/native_graph.toml --id example --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- inspect example --store smythe_artifacts/native-runs --events
cargo run --release --bin smythe -- run --graph examples/native_graph.toml --id example --store smythe_artifacts/native-runs --resume
```

`run` accepts TOML or JSON and currently uses `OfflineProvider`, which returns
its goal, input and content-addressed references to dependency outputs. `--workers` defaults to four;
`--budget-nanousd` defaults to zero. Run IDs are generated when omitted; resume
requires an explicit ID. Output is JSON; `--json` is accepted explicitly. Ordinary
run, inspect and control envelopes use schema version 1. Exit code 0 means success,
1 means an execution or integrity error, and 2 means invalid CLI syntax.

## Workspace and scheduling

- `smythe-core`: graph validation, exact money and journal transitions; no I/O.
- `smythe`: `Swarm`, local `Provider`, `CompletionResult`, `Tracer`, SQLite store
  and immutable output blobs.
- `smythe-cli`: the `smythe` executable.
- `smythe-noumenon`: authored contours, compilation, verification and exports.
- `xtask`: native benchmark and provenance commands; not a shipped component.

One coordinator owns admissions and state transitions. A bounded Rayon pool
executes local work, and a ready child can start as soon as its dependencies are
accepted. An unrelated slow sibling does not impose a wave barrier. Outputs are
returned in graph order. Invalid graphs and unaffordable admissions fail before
that work dispatches. Successful in-flight siblings settle before an ordinary
provider failure is returned.

Ready admissions that fit both free slots and the cumulative allowance share a
transaction. Already available completions may also share a transaction, without
waiting for a slow sibling. Workers serialize and publish output blobs; only the
coordinator commits acceptance and enables dependent tasks. A publication or
serialization failure retains the successful call's known cost.

`NanoUsd` stores checked unsigned 64-bit amounts and serializes them as decimal
strings. Reservations have unique call identities; settlement is idempotent.
Known overruns remain recorded and stop new admissions. The local provider
contract requires failures to carry no charge or external effect; remote calls
are not supported by this contract. Every provider explicitly declares `Effect`.
External effects are refused before execution, and journal transitions cannot
release an external reservation through local failure or replay.

## Journal and recovery

Every local run uses the same state machine. Ephemeral execution uses an
in-memory SQLite store; `run` and Noumenon compilation use a disk journal.
The store worker owns its connection and acknowledges durable commits. Disk
stores use bundled SQLite with the WAL-reset fix, WAL mode and
`synchronous=NORMAL`. Commits survive application-process crashes; a power loss
or operating-system crash can lose recent commits. Money remains decimal strings in event payloads, avoiding SQLite's
signed integer limit for amounts.

Admissions and dispatch intent precede work. Output blobs are hashed, fully written and
published immutably before their journal receipts. Received, settled and accepted
states are distinct. On recovery, received evidence completes without another
provider call; interrupted pure local work may repeat. Sequence checks and
ownership generations reject stale writes. A process lock prevents two disk
coordinators from owning the same run. Corrupt journals and missing or changed
blobs fail closed.

Native journals now use schema 2. Earlier alpha schema-1 journals and preview
directories are refused; start a new run in a new directory. No automatic alpha
migration is provided. CLI result envelopes remain schema 1. Each event records
its run, node and attempt identities, a run or execution phase, transaction
boundary, and occurrence/recording times as UTC Unix milliseconds. The event ID
is derived from run ID and sequence. Sequence orders events even when clocks
move backward. `seams::Sources` injects clocks, fresh identities and storage
boundary observers for tests; production uses real clocks, UUIDs and a no-op
observer. Typed `EngineError` variants govern admission and conflict handling.

Noumenon compilation uses this journal inside its output directory. Validation
also retains immutable batches of up to 16 glyphs. A crash after compilation but
before batch publication can reuse the compilation outputs without invoking the
contour provider again. Missing derived SVGs can be reconstructed; changed
existing artifacts are refused. A completed report still requires valid final
exports. Resume binds the protocol, contour source digest, compiler semantics,
glyph count, mode, raster version and gate version; worker count may change.
Unrelated source or CLI edits do not invalidate a run. Source hashes remain
report provenance. Every recovered compilation output is compared with the
authored recipe, including outputs saved before any batch receipt existed.
Compilation and validation evidence must agree; conflicts are integrity errors.
The compilation journal owns compiled outputs, validation batches own their
validated measurements, and missing derived SVGs are reconstructible. A completed
compilation journal alone never proves full catalog acceptance.

```bash
cargo run --release --bin smythe -- control example pause --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example continue --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example stop --store smythe_artifacts/native-runs
```

Controls apply to unfinished runs. Their acknowledgement means the command is
persisted. Active local calls drain; pause and stop prevent new admissions. After
a paused coordinator exits, record `continue` and invoke `run --resume`. Stop is
terminal. This alpha does not forcibly interrupt a local Rust callback.

These are local process-crash guarantees, not power-loss, network-filesystem,
distributed-worker or remote-effect guarantees. Paid providers, tools, MCP,
planning, supervision and Jobs are not implemented in this alpha.

## Noumenon acceptance

The 192 authored glyphs remain the prefix of the 256-glyph catalog. SVG names,
artwork and reference hashes are retained. Cubic curves use 64 samples. Full mode
renders at 16, 32, 64 and 128 pixels with tiny-skia, checks topology and all-pairs
distinctness under crop normalization, translation and reflection at 0.85 IoU.
Native raster hashes are distinct from Pillow hashes. Native reports also include stroke widths, component spacing, counters, skeleton
orientation and symmetry. Analytic fixtures check their numerical definitions;
these measurements do not yet establish catalog-wide style acceptance. This command does not modify the screensaver or
its live provider integration.

## Verification and native measurements

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo test --release --locked --offline -p smythe-noumenon release_compile_256_has_bounded_journal_cost -- --ignored --nocapture
cargo run --release -p xtask -- provenance
cargo run --release -p xtask -- benchmark smythe_artifacts/native-campaign 3
```

The release compilation gate checks 256 glyphs against a generous platform
timeout and verifies grouped journal transactions. It is a regression alarm,
not a speed claim. Recovery tests include seeded completion schedules, event
prefixes, and actual process termination before/after SQL commits and blob
publication. A subprocess fixture is invoked by its parent test; the release
performance gate is run explicitly in CI.

The native runner requires a committed, clean tree, including untracked files;
ignored local output does not make it dirty. It rejects hidden Git index entries,
hashes committed Git blob bytes, builds with locked dependencies and verifies the
same clean revision and binary after measurement. Checkout LF/CRLF conversion
therefore does not change the source hashes. It records fresh 256-glyph CLI runs
at one, four and eight workers, alternating width order. These results remain
diagnostic; three samples do not establish p95 or a public speed claim. CPU and
peak-memory measurements are not yet collected.

The [original comparison](../benchmarks/rust_noumenon_results.md) is immutable
historical evidence for revision `409d367b3cb09a06461127c6d70f2ef52e16a0ce`, which
contains its producer code. It does not describe this journaled implementation's
performance. Its recorded checkout hashes include Windows line endings; new
native records hash committed contents. The historical comparison harness rejects
the journaled runtime because its compilation clock includes different durability
work. Python remains necessary only for the
historical comparison harness and the Python line's own tests during transition.
