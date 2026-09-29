# Smythe 0.9.1

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.9.1) ·
[PyPI package](https://pypi.org/project/smythe/0.9.1/)

Smythe 0.9.1 is a patch release. A Gemini image request that finishes without
an image now fails its node instead of completing with no artifact. The Gemini
image benchmarks and examples move to `gemini-3.1-flash-image` before
`gemini-2.5-flash-image` shuts down on October 2, 2026. Benchmark records no
longer carry provider account identifiers or local paths. The Python API is
unchanged.

```bash
pip install "smythe[gemini]==0.9.1"
```

## What changes

- **Missing Gemini images fail their node.** Gemini can end an image request
  normally with only an empty text part. `GeminiProvider` now reports that
  response as `incomplete`, so the node raises `OutputRefusedError` after its
  cost is recorded and a `RETRY` policy can make another call.
  [Execution policies](execution.md).
- **Gemini 3.1 image defaults.** The image benchmarks and examples 09 and 11
  use `gemini-3.1-flash-image`, recorded at Google's $0.067 per 1K image.
  Committed records keep the model they ran.
- **Benchmark records stay clean.** `redact_account_identifiers` in
  `benchmarks/artifact_records.py` removes OpenAI organization, project and
  key formats and Google API key and project formats from the records the
  Noumenon benchmark writes, and a repository test fails when any tracked file
  contains one. The Noumenon and image concurrency benchmarks replace the home
  folder with `~`, and the jobs scale campaign refuses an evidence folder
  inside the home folder. Retained records and test fixtures were rewritten
  without local paths, and the records that pin their digests were updated.
- **Diagnostic records from September 26, 2026.** Second runs of the live
  transparent and SVG Noumenon lanes, a zero-latency Noumenon profile, a
  192-glyph Gemini concurrency sweep up to 128 and a repeat of the image
  concurrency sweep. Their reports label them diagnostic; no headline result
  changes.

Full behavior and fixes are in the [changelog](../CHANGELOG.md#091---2026-09-29).

## Upgrade from 0.9.0

Checkpoints and durable runs are unchanged. Check these before upgrading:

- **Gemini image requests.** A request to an image model, or with
  `response_modalities` that include `IMAGE`, that returns no image now fails
  its node under the node's failure policy; `RETRY` makes another billed call.
  Text requests and image requests that return an image are unaffected.
- **Gemini image model.** Code that passes `gemini-2.5-flash-image` must move
  to another model, such as `gemini-3.1-flash-image`, before October 2, 2026.
- **Benchmark harnesses.** The jobs scale campaign refuses an evidence folder
  inside the home folder, because its journal and records bind absolute paths.
  Set `SMYTHE_ALLOW_PRIVATE_EVIDENCE_PATHS=1` only for records that will never
  be published. `benchmarks/artifact_records.py` adds `redact_local_paths`,
  `scrub_record` and `evidence_directory` for new harnesses.

## Verify a checkout or package

Install `.[dev,openai,anthropic]` from the tagged checkout and run:

```bash
python -m ruff check .
python -m mypy
python -m pytest tests/ -q
python examples/14_durable_text_workflow.py
```

The example uses fixture responses and makes no provider calls. The
[distribution guide](distribution.md) covers archive inventories, source-to-wheel
reproduction, clean installed-package checks and the source test profile.
Package README links resolve against `v0.9.1`. The release workflow retains
the actual published distributions and SHA-256 hashes; verify downloaded PyPI
bytes against that receipt as described in [Releasing](../RELEASING.md).
