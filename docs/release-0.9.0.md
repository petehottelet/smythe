# Smythe 0.9.0

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.9.0) ·
[PyPI package](https://pypi.org/project/smythe/0.9.0/)

Smythe 0.9.0 narrows the repository to the framework and its evidence. The
glyph fan-out example is now Noumenon, re-measured under its unchanged
protocol and extended with live transparent-PNG and SVG lanes. The screensaver
apps and Repo Doctor live in their own repositories, and retired benchmark
records move into an archive pinned by SHA-256. The library adds
transparent-background requests to `OpenAIImageProvider`.

```bash
pip install "smythe[openai]==0.9.0"
```

## What changes

- **Transparent GPT Image output.**
  `OpenAIImageProvider(background="transparent")` asks models that support it
  for a transparent background and requires PNG or WebP output. `"opaque"`
  asks for an opaque background, and the default `"auto"` sends requests
  unchanged.
- **The Noumenon benchmark.** The glyph fan-out example is renamed Noumenon
  (`benchmarks/run_noumenon.py`) and was re-run: every tile valid and unique
  at every concurrency, 43.73× to 53.98× at concurrency 64 across 64 to 256
  nodes, and 52.32× at 192 nodes. Live GPT Image lanes generated all 192
  glyphs as transparent PNGs and traced them into SVGs that rasterize back to
  their masks exactly. [Report and records](../benchmarks/noumenon_benchmark.md).
- **Separate repositories.** The screensaver's web explorer and Windows,
  macOS and Linux ports are in the
  [Noumenon repository](https://github.com/petehottelet/noumenon). Repo Doctor
  is [repodoctor](https://github.com/petehottelet/repodoctor), which publishes
  its own skill ZIP.
- **Retired evidence is archived, not deleted.** `tools/evidence_archive.py`
  builds an archive from git history and verifies it against a committed
  pointer. The first archive, attached to this release, holds the first glyph
  fan-out campaign, the browser renderer timing study, superseded partitions,
  review records and the withdrawn native package receipts.
  [Archive and verification](../benchmarks/archive/README.md).
- **Changelog archive.** Sections through 0.8.1 moved, unchanged, to
  [`CHANGELOG-ARCHIVE.md`](../CHANGELOG-ARCHIVE.md).

Full behavior and fixes are in the [changelog](../CHANGELOG.md#090---2026-09-26).

## Upgrade from 0.8.2

The Python API changes only by addition, and checkpoints and durable runs are
unchanged. Check these if you work from the repository:

- **Benchmark paths.** `benchmarks/run_glyph_screensaver.py` is now
  `benchmarks/run_noumenon.py`, `benchmarks/glyph_screensaver_assets.py` is
  `benchmarks/noumenon_assets.py`, and default records are
  `benchmarks/results/noumenon_*.json`. The glyph generator, contours and
  catalog moved from `screensaver/` to `benchmarks/noumenon/`.
- **Screensaver sources** are no longer in this repository; build them from
  the Noumenon repository.
- **Repo Doctor** ships from the repodoctor repository; Smythe releases no
  longer attach the skill ZIP.
- **Benchmark records.** Environment snapshots mark a checkout `dirty` only
  when tracked files changed and count untracked files in `untracked_files`.
  A live Noumenon run that halts records what its completed calls charged.

## Verify a checkout or package

Install `.[dev,openai,anthropic]` from the tagged checkout and run:

```bash
python -m ruff check .
python -m mypy
python -m pytest tests/ -q
python examples/14_durable_text_workflow.py
```

The example uses fixture responses and makes no provider calls. To check the
evidence archive, download it from the release and run
`python tools/evidence_archive.py verify smythe-evidence-2026-09-25.zip --against-git`
from a checkout with tags fetched. The [distribution guide](distribution.md)
covers archive inventories, source-to-wheel reproduction, clean
installed-package checks and the source test profile. Package README links
resolve against `v0.9.0`. The release workflow retains the actual published
distributions and SHA-256 hashes; verify downloaded PyPI bytes against that
receipt as described in [Releasing](../RELEASING.md).
