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

## Historical screensaver packages

Precompiled screensaver packages were withdrawn on 7 September 2026. The
screensaver is now the source-only
[Noumenon repository](https://github.com/petehottelet/noumenon); build its
native ports from source there.

The release originally included native **1.1** packages from source
`a0327aa58f80e6c7206f5c3eeeae1b48985953e6` and
[build 34123023804](https://github.com/petehottelet/smythe/actions/runs/34123023804).
They contain 56 visible reference glyphs, the reference blank slot, and all
192 original Smythe SVG shapes, with a 10% original mix. All three packages,
their checksum list, build information and the candidate verification ZIP
matched the uploaded files when downloaded again after publication.

The packages' file names and SHA-256 checksums, their build information, and
the native rendering receipts for Windows, Apple Silicon, Intel Mac, and
Ubuntu 22.04/24.04 are retained, with this guide's original text, in the
[evidence archive](../benchmarks/archive/README.md).

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
and the renderer study in the [evidence archive](../benchmarks/archive/README.md)
retain their own frozen sources, protocols, and measurement limits.
