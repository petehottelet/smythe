# Smythe 0.7.0 verification

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.7.0) ·
[PyPI package](https://pypi.org/project/smythe/0.7.0/) ·
[Release checklist](https://github.com/petehottelet/smythe/issues/15)

Published 7 September 2026 from
`729fda3745f69f35937b831ee1734c235ce29c52`. The annotated `v0.7.0` tag,
GitHub release, and PyPI publishing run identify that source.

## Source and package checks

All nine [release CI jobs](https://github.com/petehottelet/smythe/actions/runs/34164206893)
passed. The full offline suites passed 3,273 tests with 24 skipped on each
Linux Python version, 3.11–3.13; Windows passed 3,282 with 15 skipped.
The macOS operator/checkpoint lane passed 177 with 12 skipped, and the
OpenAI SDK 3.8.0 contract lane passed 502. The final local Python 3.11.9 suite
passed 3,287 with 10 skipped in 1,633.63 seconds. Platform and optional
dependency coverage account for different totals.

Python 3.13's pytest summary reports 139 warnings, including SQLite
`ResourceWarning`s. The raw log also retains warnings outside that summary.

Candidate distributions passed strict Twine validation, complete package
member inspection, and an identical wheel rebuild from the source archive.
The [trusted publishing workflow](https://github.com/petehottelet/smythe/actions/runs/34165199495)
then built, retained, and published its own distributions. Both files downloaded
from PyPI match that workflow's retained artifacts and `package-hashes.txt`
byte for byte:

| PyPI distribution | SHA-256 |
|---|---|
| `smythe-0.7.0-py3-none-any.whl` | `2a3fc92ce8ee3755b270cef8f1219f9fb6dcb292118896aa142f551e61375eac` |
| `smythe-0.7.0.tar.gz` | `13a80a02b371decde608332931d179947e9eca13b556e4d127fab342e981e206` |

The Windows candidate build uses different source newlines from the Linux
publisher. Its separate hashes are preserved in the candidate verification
bundle; the table above identifies the actual PyPI downloads.

## Fresh PyPI installation

A new Python 3.11 environment installed `smythe[jobs,openai]==0.7.0` from
PyPI with OpenAI SDK 3.8.0. Dependency validation passed.

- The exact release README example completed planning and four execution
  nodes through five native request quotes and five generations. Synthetic
  SDK responses exercised the real SDK and Smythe accounting path, recording
  an exact $0.0075 fixture ledger total. Resuming the completed workflow made
  zero SDK calls.
- All 12 installed offline Jobs CLI checks passed. Four operations produced
  accepted artifacts with zero API charges. Read-only inspections preserved
  database bytes, and completed resume made zero additional calls.
- The published package description matches the release README's build
  transformation. Checks resolve 59 source links against the frozen checkout
  and fetch four live URLs. Documentation and artwork links use the release
  tag; the PyPI badge alone follows `main`.

These are package, API-contract, and recovery checks. They made no paid
provider calls and provide no model-quality result. The
[Astra campaign](../benchmarks/astra_runtime.md) has separate spending and
calibration gates.

## Screensaver downloads

The release includes the verified native **1.1** packages from source
`a0327aa58f80e6c7206f5c3eeeae1b48985953e6` and
[build 34123023804](https://github.com/petehottelet/smythe/actions/runs/34123023804).
They contain 56 visible reference glyphs, the reference blank slot, and all
192 original Smythe SVG shapes, with a 10% original mix.

| Native package | SHA-256 |
|---|---|
| `SmytheGlyphRain.scr` | `c7660ee87d6d0e7774b1ee0c044d66dabf8cd92ebd8b8d193a55e6621ac177ea` |
| `GlyphRain-macos-universal.zip` | `d6b3d8367efa23526b0458c4dc81b59e39044e82e5c5133ac2e26af042c820a8` |
| `SmytheGlyphRain-linux-x86_64.tar.gz` | `a42a258e5fa85ed0939bc5ba1f4cd1a74237786bc9566b7a420819ac4af178a5` |

All three packages, `SHA256SUMS`, `BUILD_INFO.json`, and the candidate
verification ZIP were downloaded again after publication and matched the
uploaded files. The native packages retain their own version and source
identity; attaching them to 0.7.0 does not imply a new build.

[Native rendering and host checks](../screensaver/README.md#native-verification)
cover Windows, Apple Silicon, Intel Mac, and Ubuntu 22.04/24.04. macOS remains
ad-hoc signed, and Linux requires X11. Native layered rain and the web
explorer's REGL effect, 3D travel, and settings have separate implementation
scope.

## Retained evidence

Download the [candidate verification ZIP](https://github.com/petehottelet/smythe/releases/download/v0.7.0/Smythe-0.7.0-verification.zip),
[postpublication verification ZIP](https://github.com/petehottelet/smythe/releases/download/v0.7.0/Smythe-0.7.0-postpublication-verification.zip),
and [postpublication summary](https://github.com/petehottelet/smythe/releases/download/v0.7.0/Smythe-0.7.0-postpublication-verification.json).
The postpublication archive contains 143 hash-verified members and is
290,767 bytes. Its SHA-256 is
`fae0635990edb61341610edbaa49a1f5b9f4827030b82118d9a07513f5646627`.

The release attachments and checklist preserve checker sources, hashes,
environment identities, raw SDK and CLI evidence, package inventories, and
CI output. Candidate qualification and postpublication download/install
verification are separate records. Failed checking attempts remain available,
including the initial checksum comparison against Windows CRLF checkout bytes
instead of the uploaded LF file.

The [5,000-operation recovery report](../benchmarks/jobs_scale_5000_20260907_results.md)
and [renderer study](../benchmarks/renderer_performance_20260907_results.md)
retain their own frozen sources, protocols, and measurement limits.
