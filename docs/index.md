# Smythe documentation

The active project is the **Rust 0.9.0 source preview**. It runs supplied local
task graphs and the Noumenon glyph pipeline with bounded concurrency, exact
accounting and journaled recovery. Python is not needed to build or run it.
The stable Rust release will be 1.0.0. Local qualification remains open;
remote providers and generated planning are still ahead. The roadmap separates
the 1.0 integrations from later learning capabilities.

The amended schema-3 engine supports resumable cost-quote blocks and separate
retry allowance after invalidation. Earlier schema-3 journals retain their
recorded outcomes; see [compatibility and recovery](rust.md#journal-and-recovery).
CLI diagnostics now retain safe causes and parser locations within a byte bound;
explicit page limits require a cursor. Noumenon report source coverage is checked
against all runtime sources. These additions do not complete local qualification.

## Start here

| Goal | Guide |
|---|---|
| Build and run the preview | [Source builds and distribution](distribution.md) · [README quickstart](../README.md#quickstart) |
| Run, inspect and resume a graph | [Native guide](rust.md#build-and-run) · [Examples](../examples/README.md#rust-alpha-examples) |
| Understand the crate boundaries and scheduler | [Workspace and scheduling](rust.md#workspace-and-scheduling) |
| Understand stored outputs and recovery | [Journal and recovery](rust.md#journal-and-recovery) |
| Check current recovery and inspection limitations | [Known preview limitations](rust.md#known-preview-limitations) · [Local qualification](../ROADMAP.md#local-qualification) |
| Change budgets, retry or stop work | [Controls, limits and revisions](rust.md#controls-limits-and-revisions) |
| Retrieve outputs or page through events | [CLI output and inspection](rust.md#cli-output-and-inspection) |
| Generate and validate Noumenon glyphs | [Noumenon acceptance](rust.md#noumenon-acceptance) |
| Run checks or record native measurements | [Verification and provenance](rust.md#verification-and-native-measurements) |

## Benchmarks and artwork

- [Current materials](current-materials.md): the retained Noumenon GIF, 192/256
  glyph sheets, SVGs and links to evidence.
- [Rust/Python comparison](../benchmarks/rust_noumenon_results.md): diagnostic
  measurements of the original preview, with its producer revision and limits.
- [Benchmark index](../benchmarks/README.md): protocols, reports and retained
  raw records. Historical Python results do not describe the current Rust engine.
- [Evidence archive](../benchmarks/archive/README.md): retained historical and
  superseded records with pinned identities.
- [Noumenon](https://github.com/petehottelet/noumenon): the separate screensaver
  project that renders the glyph catalog.

## Project guides

- [Roadmap](../ROADMAP.md): implemented capabilities, next milestones and 1.0 criteria.
- [Contributing](../CONTRIBUTING.md): setup, crate responsibilities and checks.
- [Releasing](../RELEASING.md): preview identities, verification and publication scope.
- [Changelog](../CHANGELOG.md): unreleased Rust changes and immutable Python releases.
- [Security](../SECURITY.md): supported code, reporting and the native threat model.
- [Code of Conduct](../CODE_OF_CONDUCT.md): community standards and reporting.
- [House style](style.md): diagrams, charts, badges and evidence presentation.
- [MIT license](../LICENSE): license terms and copyright notice.

## Python 0.10 reference

These guides describe the retired Python implementation. Their commands require
[source revision 12f1470](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce),
not the current checkout. In those guides, a provider's “native API” means its
own network interface, not a Rust integration. Retaining a guide does not imply
that its feature has been ported.

| Area | Historical guides |
|---|---|
| Setup and compatibility | [Python quickstart](python-quickstart.md), [examples](../examples/README.md#python-examples), [support matrix](support-matrix.md), [Python packaging](https://github.com/petehottelet/smythe/blob/12f14704beff344228fffffce40d995daa1f7cce/docs/distribution.md) |
| Graphs and execution | [Architecture](architecture.md), [tasks](tasks.md), [execution policies](execution.md), [budgets](budgets.md) |
| Persistence and inspection | [Checkpoints](checkpoint-format.md), [durable workflows](workflow-accounting.md), [run inspection](runs.md) |
| Verification and repair | [Verification](verifier.md), [supervision](supervisor.md), [source evidence](source-evidence.md), [lineage](lineage.md) |
| Models and tools | [OpenAI Responses](openai-responses.md), [Claude Messages](anthropic-messages.md), [MCP](mcp.md), [skills](skills.md) |
| Artifacts and policy changes | [Jobs](jobs.md), [external actions](actions.md), [learning](learning.md), [optimization](optimize.md) |

Release guides: [0.10.0](release-0.10.0.md), [0.9.1](release-0.9.1.md),
[0.9.0](release-0.9.0.md), [0.8.2](release-0.8.2.md),
[0.8.1](release-0.8.1.md), [0.8.0](release-0.8.0.md),
[0.7.0](release-0.7.0.md). Earlier release entries remain in the
[changelog archive](../CHANGELOG-ARCHIVE.md).
