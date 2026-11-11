# Rust alpha: journaled local execution

The native workspace is **0.9.0**, a Rust-only source preview. Its packages are
unpublished and interfaces are experimental. The eventual stable Rust release
will be 1.0.0. Historical Python source, tests and comparison harnesses remain at
[12f1470](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce).
They are not required to build or run the active repository.

See [source builds and distribution](distribution.md) for executable locations
and version identities, and the [roadmap](../ROADMAP.md) for the path to 1.0.
The local engine is still being qualified; the [known preview limitations](#known-preview-limitations)
below describe gaps that remain in the current implementation.

## Build and run

Rust 1.91 or later and a C build toolchain for bundled SQLite are required:

```bash
cargo fetch --locked
cargo build --release --locked --offline -p smythe-cli
cargo run --release --locked --offline --bin smythe -- run --graph examples/native_graph.toml --id example --store smythe_artifacts/native-runs
cargo run --release --locked --offline --bin smythe -- inspect example --store smythe_artifacts/native-runs
cargo run --release --locked --offline --bin smythe -- run --graph examples/native_graph.toml --id example --store smythe_artifacts/native-runs --resume
```

Graphs accept TOML or JSON. Nodes declare `id`, `task`, `depends_on`, and an
optional named `phase`. Topology labels are optional descriptions; dependencies
determine scheduling. Inspect derives node/edge counts, maximum level width,
depth and joins. Changing only the topology label does not invalidate resume.

`run` uses the zero-cost local `OfflineProvider`. Workers default to four
(range 1–64); a fresh budget defaults to zero nanoUSD. Fresh IDs are generated
when omitted; resume requires an ID. On resume, omit the budget to use the
journaled ceiling. An explicit conflicting ceiling is an error.

## Workspace and scheduling

- `smythe-core`: graph validation, exact accounting, transitions and the pure coordinator.
- `smythe`: `Swarm`, local `Provider`, `CompletionResult`, `Tracer`, storage and I/O seams.
- `smythe-cli`: the `smythe` executable.
- `smythe-noumenon`: contours, compilation, verification and exports.
- `xtask`: native measurements and provenance.

A pure coordinator consumes completions, controls, clock ticks and commit
acknowledgements. It emits journal appends and dispatch commands. The shell owns
Rayon workers, storage, clocks and control polling. A ready child runs as soon as
its dependencies are accepted. Admissions that fit the worker and allowance
limits share a transaction; available completions can share one too.

The hot path uses graph indexes, incremental readiness and accounting totals,
and rollback of touched entries. Storage returns compact acknowledgements and
incremental journal changes. It does not clone or replay the full projection
for each event. Listing uses a compact transactional summary.

Providers receive `Dependency<T>` handles. `reference()` returns SHA-256 and
byte length without reading data; `load()` verifies and deserializes the output.
The scheduler retains references, not every prior value. `SwarmResult.outputs`
is a graph-ordered list of node references; use `load(node)` or `load_all()`
explicitly. `dependency_bytes` counts referenced bytes at dispatch and
`loaded_dependency_bytes` counts actual handle reads for this invocation.

## Journal and recovery

Schema **3** records run, revision, node, attempt, scope, named phase, transaction
boundary, and occurrence/recording times. Sequence orders events despite clock
skew. Schema 1 and 2 journals are refused without modification; use a fresh
directory. CLI envelopes remain schema 1.

Schema 3 is amended in place: the current reader preserves the original fields
of older schema-3 journal projections, including every recorded prefix. It adds
resumable `quote_blocked` events and derives an attempt's `revision_base` from
invalidation. New `local_replay` events carry an optional checked base marker;
older events without it retain their original lifetime-limit meaning. Older
binaries are not forward-compatible with these events and attempt sequences;
use the amended reader for stores written by this implementation.

Already-terminal `RunFailed` journals with `quote_failed` or `attempt_limit`
remain failed. They need a new run ID; this amendment does not rewrite history
or provide a recovery override.

One process lock owns each run. Generation and sequence checks reject stale
writes. Every dispatch follows committed admission. Received evidence, settlement
and acceptance remain distinct. Recovery reuses valid receipts without another
provider call; interrupted pure work without a receipt may repeat within its
attempt limit. Explicit retry is required for failed or rejected pure attempts.

Small outputs (at most 64 KiB) and their receipts commit in one SQLite transaction.
Large outputs are published immutably beneath a two-hex-digit hash directory
before their receipt commits. Locked per-process staging directories protect
live writers and allow abandoned temporary files to be reclaimed. Every read
checks the hash and byte length.

Both memory and disk runs use SQLite. Disk stores use WAL with
`synchronous=NORMAL`: the guarantee is recovery after an application-process
crash. OS crashes, power loss, network filesystems, remote effects and distributed
workers are outside this contract. Noumenon exports use the same process-crash
contract; there is no separate fsync durability tier.

`NanoUsd` uses checked integers and JSON decimal strings. Reservation identities
include revision, node and attempt. Known spend survives output rejection.
An overrun closes admission without changing operator intent; complete graph
coverage still yields `completed`. Otherwise closure drains in-flight work.
A budget-blocked run remains nonterminal and can continue after its ceiling grows.
A failed cost quote also leaves the run nonterminal with status `quote_blocked`.
Already-admitted siblings drain and settle first. Correct the provider and
resume the same run: quotes are recomputed, accepted outputs are reused and
prior settlement is preserved. Repeated unsuccessful resumes add no duplicate
quote-block event or attempt. Provider diagnostic text is not journaled.

## Controls, limits and revisions

```bash
cargo run --release --bin smythe -- control example pause --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example continue --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example budget 1000000 --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example retry research --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- control example stop --store smythe_artifacts/native-runs
```

Acknowledgement means the command is recorded. Pause and stop prevent new
admissions while active callbacks drain; callbacks are not forcibly interrupted.
After the coordinator exits, record continue/retry as appropriate and invoke
`run --resume`. Operator stop cannot be undone. A new ceiling cannot be below
spent plus reserved money. Budget changes are outside the semantic run binding.
External effects are refused by the runtime and cannot use pure retry or replay.

Saved limits default to 10,000 nodes, 100,000 edges, three attempts per node
since its last invalidation,
64 MiB per output and 1 GiB total received output. Configure them with
`--max-nodes`, `--max-edges`, `--max-attempts-per-node`,
`--max-output-bytes`, and `--max-total-output-bytes`.
`--wall-deadline-ms` is an optional absolute UTC Unix-millisecond deadline;
once observed, backward clock movement cannot reopen admission. Resume preserves
limits. Oversized successful results are rejected while retaining their cost.

The event API supports quiescent graph changes: `Expanded` names its parent and
new revision and input manifest digest; `Revised` gives a reason and invalidation
roots. Descendants are invalidated transitively; unaffected accepted outputs are
reused. Duplicate identities, missing dependencies, cycles, stale revisions and
changes with unsettled work are refused. These events are foundations for later
planning and supervision, not a model-driven planner.

Attempt numbers remain monotonic across revisions. Invalidation resets only
the affected node's retry allowance, including descendants; ordinary retries,
resumes and expansions do not refill it. Three failures or interrupted attempts
within that allowance still exhaust the default limit. A separate cap on the
number of revisions belongs to future supervision and is not enforced here.

## CLI output and inspection

Normal commands emit JSON to stdout. Failures emit a schema-1 error envelope
containing `run_id` when known and `error.class` and `error.message`; stderr
also contains a human-readable diagnostic. `--json` is accepted explicitly.

The same public diagnostic appears in JSON and stderr, bounded to 1,024 UTF-8
bytes with `[truncated]` when shortened. Recognized causes explain cycles,
validation and ownership failures. Parser diagnostics include format, line and
column; schema names and single-edit field typos are retained. Other field names,
source snippets, argument values and arbitrary cause text are omitted. Unsupported
causes use the fixed class message. Provider callback errors remain suppressed
before journaling. Classes and exit codes remain the decision interface.

| Exit | Meaning |
|---|---|
| 0 | Success |
| 1 | Execution/provider failure or resumable `quote_blocked` |
| 2 | Invalid arguments or configuration |
| 3 | Paused or stopped |
| 4 | Budget blocked |
| 5 | Integrity failure |
| 6 | Existing or missing run/store |

`run` returns references by default. Use `--outputs inline` to materialize all
values, or retrieve one output:

```bash
cargo run --release --bin smythe -- output example research --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- list --store smythe_artifacts/native-runs
cargo run --release --bin smythe -- inspect example --events --after 0 --limit 100 --store smythe_artifacts/native-runs
```

Inspect and list open existing stores read-only and never create a missing store.
Plain inspect verifies the complete journal and reports shape, accepted coverage,
unresolved reservations, last progress and derived owner `live`/`absent` health.
Paged inspection uses sequence cursors; pages can split transaction groups.
Concatenating pages reconstructs the same event stream as full history.
An explicit `--limit` requires `--after`, which requires `--events`; omitting
the cursor is an argument error. With a cursor, the default limit is 1,000
and the allowed range is 1–10,000. Without a cursor, `--events` retains the
full inspection envelope and event history.

## Known preview limitations

These qualification gaps remain in the current schema-3 preview. The roadmap
lists the work still required before 1.0.

| Area | Current behavior |
|---|---|
| Local acceptance | Shared raster-reference approval, catalog style acceptance and full-workflow measurement qualification remain open |
| Native tooling | Tracked-file identifier scanning, local-link tooling and dependency advisory/license gates remain planned |
| Remote workflows | Paid providers, external-effect recovery, generated planning, tools and supervision remain future capabilities |

## Noumenon acceptance

```bash
cargo run --release --locked --offline --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs
cargo run --release --locked --offline --bin smythe -- noumenon --glyphs 256 --workers 8 --out smythe_artifacts/native-glyphs --resume
```

Noumenon accepts 1–256 glyphs and 1–64 workers, defaulting to 192 and four.
A fresh run needs a new output directory. The 192 authored glyphs remain the
prefix of the 256-glyph catalog with identical names and SVG reference hashes.
The README GIF remains the benchmark example; it is not a Rust timing recording.

Compilation and validation are separate graph runs in the shared journal, with
named compile/validation phases. Separate batch-receipt files have been retired.
Resume checks saved compilation and validation evidence in parallel, reconstructs
missing derived SVGs, and refuses changed artifacts. Semantic protocol, recipe,
compiler, raster and gate versions bind recovery; worker count may change.

Reports hash every Rust source beneath `src/` in the four runtime crates, their manifests,
the workspace manifest, lockfile and Clippy configuration. A completeness test
detects unlisted source files and checks the embedded hashes against their bytes.
These hashes are informational: source edits do not change semantic resume
bindings. Clean committed-source and binary provenance remains the separate
`xtask provenance` contract.

Full mode renders at 16, 32, 64 and 128 pixels, checks topology and all-pairs
distinctness under crop normalization, translation and reflection at 0.85 IoU,
and writes SVGs, catalog, atlas, validation and report files. Native raster
hashes differ from Pillow. Stroke, spacing, counters, skeleton orientation and
symmetry measurements have analytic fixtures but do not yet establish catalog-wide
style acceptance. `--mode compile` omits validation. Both modes make zero API calls.

SVG hashes have a committed reference. Native pixel, atlas and catalog hashes
are checked for consistency within runs, but are not yet compared with a shared,
approved raster manifest across Windows and Linux. That reference and catalog
style acceptance are separate remaining qualification requirements.

## Verification and native measurements

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --release --locked --offline
cargo test --release --locked --offline -p smythe --test scaling -- --ignored --nocapture
cargo test --release --locked --offline -p smythe-noumenon release_compile_256_has_bounded_journal_cost -- --ignored --nocapture
cargo run --release --locked --offline -p xtask -- provenance
cargo run --release --locked --offline -p xtask -- benchmark smythe_artifacts/native-campaign 3
```

Ruff is not applicable to the Rust-only active repository. Linux and Windows CI
run the offline suite, including 5,000 deterministic ChaCha8 scheduler seeds,
committed-prefix replay, controls, clock skew, quote-block recovery, explicit
retries, expansion, revision and crash-before-acknowledgement schedules. Separate
real-process tests kill writers around SQL commits and blob
publication. Release gates compare per-node time at 2,000 and 8,000 independent
nodes (at most 1.5× growth), execute a 5,000-node chain, and bound 256-node inline
disk overhead to 3× memory. Large-file disk timing is diagnostic.

The simulation checks accepted lineage, per-invalidation attempt allowance,
unique reservations, single settlement and recovery of recorded expansions.
Saved counterexamples cover dispatch before commit and a deliberately restored
lifetime attempt limit. Four journals captured from the earlier schema-3
implementation preserve original projections at every prefix. These local
checks do not qualify remote effects or complete [local qualification](../ROADMAP.md#local-qualification).

The measurement runner refuses uncommitted trees and hidden index entries,
hashes committed Git blob contents, builds locked dependencies and rechecks
source and binary identities after measurement. Checkout LF/CRLF conversion
does not alter its source hashes. Fresh 256-glyph trials alternate one, four and
eight workers. Results are diagnostic; CPU and peak-memory campaign measurements
are still pending. Regression gates are not public speed claims.

The [original comparison](../benchmarks/rust_noumenon_results.md) is historical
evidence for `409d367b3cb09a06461127c6d70f2ef52e16a0ce`. Its checkout hashes include
Windows line endings and two recorded mixed-ending exceptions. The historical
harness lives in Git history and has different clocks; those timings do not
describe the schema-3 engine. The source revision and scope are recorded in the
[historical comparison report](../benchmarks/rust_noumenon_results.md).

Paid providers, tools, MCP, model-driven planning, supervision and Jobs remain
future native capabilities. Prompt optimization is deferred until there is an architect.
