# Smythe roadmap

Smythe's destination is a Rust framework that turns goals into inspectable task
graphs, executes them within explicit limits, and delivers verified outputs
with durable recovery. The active workspace is the **0.9.0 source preview**.
The first stable Rust release will be **1.0.0** when its acceptance criteria
are met; this roadmap does not assign a release date.

## Rust alpha (0.9.0)

The [native guide](docs/rust.md) describes the implemented behavior:

| Area | Available now |
|---|---|
| Local execution | Supplied TOML/JSON graphs, pure Rust providers, dependency-driven scheduling and bounded workers |
| Accounting and control | Exact nanoUSD reservations, saved resource limits, explicit retries, budget changes, pause, continue and stop |
| Recovery | Schema-3 SQLite journals, committed admission before dispatch, resumable quote blocks, atomic inline outputs and receipts, immutable large blobs and process-crash recovery |
| Graph changes | Named phases, revision identities, quiescent expansion and revision, transitive invalidation, fresh retry allowance for invalidated nodes and reuse of unaffected outputs |
| Inspection | Output references and explicit loading, read-only run lists, verified inspection, event cursors, owner health and typed CLI errors |
| Noumenon | Shared compilation and validation runtime, the authored 192/256-glyph catalogs, SVG parity, exports and completed-run recovery |
| Verification | Linux and Windows CI, 5,000 deterministic scheduler seeds, real-process crash tests and explicit release performance gates |
| Provenance | A native measurement runner that refuses uncommitted trees and hashes committed Git blob contents |

The active implementation has no Python dependency. Model-driven planning,
remote providers, tools, MCP, supervision, Jobs and learning remain future
native capabilities. Graph-revision events provide a foundation for these
features; they do not supply a planner or supervisor by themselves.

## Local qualification

The project is qualifying the local engine. Passing recovery and scaling tests
establishes the covered behavior; the remaining correctness and acceptance
work still gates the next milestone. Quote recovery and per-invalidation retry
limits are implemented, with expanded simulation and schema-3 compatibility
fixtures. Existing terminal quote/attempt failures retain their recorded outcome;
see [compatibility](docs/rust.md#journal-and-recovery). Bounded sanitized CLI
diagnostics, enforced event-paging arguments and complete runtime source hashes
in Noumenon reports are implemented. Remaining [qualification gaps](docs/rust.md#known-preview-limitations)
still prevent a 1.0 readiness claim; version 0.9.0 remains a source preview.

The immediate priorities are:

1. Add native identifier/link checks and dependency advisory/license checks.
   This work can proceed independently of raster-reference approval.
2. Approve a shared native raster reference, complete catalog style acceptance,
   and measure full-workflow CPU and peak memory with committed provenance.

The Rust-only repository is already in place. Native evidence-tool replacements,
free-campaign reruns and installation qualification remain required work; the
source cutover does not establish beta readiness.

## Coming soon

The following milestones build on the same execution and accounting contracts.
Each integration must first work against deterministic fixtures or a local fake
service. Paid evaluations require a separately authorized budget.

| Order | Milestone | Acceptance criteria |
|---|---|---|
| 1 | Qualify the local engine | Close the correctness and acceptance gaps above; keep Linux/Windows recovery and scaling gates green; retain source, binary and output identities |
| 2 | Remote accounting and providers | Prove one complete workflow against a fake service, including lost responses, retained exposure, reconciliation and resume without duplicate dispatch or settlement; then extend adapter coverage |
| 3 | Planning and routing | Preserve the Architect and WhiteRabbit naming; validate generated graphs before execution; compare a single worker, deterministic templates and generated graphs including planning cost |
| 4 | Tools, MCP and context | Bound tool loops and reads; keep tool authority in developer configuration; treat retrieved content as untrusted; journal context and reusable read receipts |
| 5 | Verification and supervision | Judge the exact delivered output; preserve verification obligations through revisions; record what a revision keeps, invalidates and adds; keep retries and revisions bounded |
| 6 | Jobs and governed actions | Add inspectable artifact jobs, approval-bound external actions and reconciliation after lost acknowledgements; demonstrate recovery with a fake service before live use |
| 7 | Interfaces, evidence and distribution | Qualify the control API, MCP server, optional OpenTelemetry export, documentation examples, native free campaigns and clean installation; telemetry failure must not affect execution |

Prompt optimization follows the first native Architect. A new execution policy
becomes a default only after measurements show it preserves accepted outcomes.
The local, single-machine path must continue to work without a service cluster.
Optional OpenTelemetry export is part of the 1.0 interface milestone, with
execution independent of whether it is enabled.

## Noumenon and performance evidence

Noumenon remains the primary benchmark example. Keep the existing names, SVG
reference hashes, contact sheets and [README animation](assets/noumenon/noumenon-loop.gif).
The animation shows the catalog in use; it is not a recording of a Rust run or
a timing result. The screensaver itself lives in the
[Noumenon repository](https://github.com/petehottelet/noumenon).

Current regression gates cover 2,000/8,000-node per-node scaling, a 5,000-node
chain, inline disk overhead and 256-glyph compilation. Full acceptance also
validates all 256 glyphs and resumes a completed run without new provider calls.
These checks detect regressions; they are not public speed claims.

The [original Rust/Python comparison](benchmarks/rust_noumenon_results.md)
describes an earlier preview and remains diagnostic. It does not measure the
schema-3 engine. A new comparison must identify both producer revisions, use
matched work and clocks, and preserve every failure and output check. The
[measurement guide](docs/rust.md#verification-and-native-measurements) and
[evidence index](benchmarks/README.md) define the current scope.

## Requirements for 1.0

- Document the supported native API, CLI, provider combinations and stability
  policy, with examples that run from the released source.
- Complete the native execution paths above with offline acceptance, accounting,
  authority and recovery tests; identify any explicitly deferred integrations.
- Keep local quote/revision recovery and the expanded simulation qualified on
  supported platforms; establish an approved raster reference and catalog style acceptance.
- Establish a tested journal compatibility and upgrade policy. Schema-1/2 Rust
  preview runs currently require fresh directories; Python stores are historical.
- Qualify installation and distribution independently of a developer checkout,
  with reproducible source identities and platform-specific build checks.
- Retain Noumenon output parity and publish performance claims only from a
  campaign whose report marks them claimable within a stated scope.
- Keep private data out of artifacts and records, and document the limits of
  the trusted-operator security model.
- Complete native privacy and evidence tooling, dependency checks and release
  artifact qualification. Passing source-tree tests alone is insufficient.

## Later

Learning, Autotune, policy and template lifecycle, memory, distillation and
learned reuse are after 1.0. Reusing verified outputs within a run and bounded
revision remain in 1.0; deterministic planning templates do not imply learning.

External workers, optional durable backends, a live dashboard, recursive
decomposition and coordination of coding harnesses can follow demonstrated
workload needs.
They must preserve the same accounting, authority and recovery contracts.
Hosted services, marketplaces and general-purpose infrastructure are outside
the current core scope.

## Python history

Python releases through 0.10.0 remain in Git history, along with their provider,
Jobs, learning and optimization implementations. Their milestones are recorded
in the [changelog](CHANGELOG.md), [changelog archive](CHANGELOG-ARCHIVE.md),
[historical guides](docs/index.md#python-010-reference) and
[previous roadmap](https://github.com/petehottelet/smythe/blob/12f14704beff344228fffffce40d995daa1f7cce/ROADMAP.md).
Historical benchmark records remain available with their original scope.
Python release numbers do not identify Rust releases.

See [Contributing](CONTRIBUTING.md) for development checks and ways to propose
work against these milestones.
