# Releasing Smythe

The active project is the **Rust 0.9.0 source preview**. Packages remain
`publish = false`; there is no automatic package or GitHub Release publication
workflow. The old PyPI publishing workflow has been retired. Existing Python
releases and tags remain historical.

The [source-build guide](docs/distribution.md) is the installation contract for
this preview. The [roadmap](ROADMAP.md#requirements-for-10) defines the remaining
1.0 acceptance criteria; a version label alone does not establish readiness.

## Verification

Run the documented [native lint, offline suite, build and performance gates](docs/rust.md#verification-and-native-measurements).
Validate all 256 Noumenon glyphs and completed recovery. Confirm the tested
commit on main has successful Linux and Windows CI and inspect the exact
source and binary provenance. Review documentation links, benchmark scope and
retained display artwork.

Current CI qualifies the implemented regression tests and performance gates.
Before 1.0, also close the [local qualification gaps](ROADMAP.md#local-qualification),
approve the shared raster reference, complete catalog style acceptance and
native campaign measurements, and qualify installation independently of a
checkout. Native identifier/link and evidence tools, dependency checks, and
release-artifact privacy, license and provenance checks remain requirements;
do not record unavailable or unrun checks as passing.

Ruff and the historical Python suite are not applicable to this Rust-only tree.
Historical Python reproduction uses the
[retained source revision](https://github.com/petehottelet/smythe/tree/12f14704beff344228fffffce40d995daa1f7cce).

## Version identities

Keep workspace versions aligned in Cargo.toml and Cargo.lock. Rust's stable
release will be **1.0.0** when its functionality and acceptance gates are ready.
The existing `v0.2.0` tag belongs to Python history. The immutable
`rust-v0.2.0` tag identifies the original Rust preview; subsequent fixes do not
move that published tag. Main continues development of the 0.9.0 preview.
The 0.9.0 workspace version still denotes an experimental source preview;
it does not declare M1 complete or establish beta or 1.0 readiness. Historical
Python tags, including `v0.9.0`, are not Rust release identities.

A push does not publish a package or create a release. Release publication is a
separate, explicitly authorized action. Use the supplied release title exactly.
Keep unreleased changes under `[Unreleased]`; preserve historical changelog
sections and published references.

## Evidence

Retain benchmark reports, records and the Noumenon GIF. New results must identify
the committed producer revision and binary and state their measurement scope.
The optional local source archive is not a release or cutover gate. Historical
campaigns retain their recorded provenance, including line-ending exceptions.
Private materials never belong in a release archive or distribution.
