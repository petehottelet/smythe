# Smythe 0.8.0

[GitHub release](https://github.com/petehottelet/smythe/releases/tag/v0.8.0) ·
[PyPI package](https://pypi.org/project/smythe/0.8.0/)

Smythe 0.8.0 adds native Claude Messages, owned Autotune campaigns, and smaller
distributions to its generated execution graphs and durable execution envelope.
It also publishes the completed Astra comparison with reproducible evidence.

```bash
pip install "smythe[openai]==0.8.0"
# For native Claude Messages:
pip install "smythe[anthropic]==0.8.0"
```

## What changes

- Native Claude Messages support with exact cache billing, conservative request
  quotes and saved-response recovery. Human pilot ratings still gate the Fable
  native main comparison; completed pilot and Code Workflow evidence is retained.
- Autotune campaign leases, fenced mutations, conservative recovery, and
  read-only HTML reports; planner-history validation and concurrent SQLite
  initialization checks.
- Compact source distributions, an explicit library test profile, installed
  consumer typing checks, and separately verified Repo Doctor release assets.
- External-network blocking in Python tests and bounded manual provider probes.
  Publishing requires a matching tag and successful main CI at that exact commit.
- The [Astra findings](../benchmarks/astra_findings.md), with all 200 outcomes,
  the original automatic acceptance rule, separate human judgments and retained
  unknown-cost bounds. Generated plans increased mean time on the study's reused
  task set; these records do not support a general speedup claim.

Full behavior and fixes are in the [changelog](../CHANGELOG-ARCHIVE.md#080---2026-09-21).

## Upgrade from 0.7

Low-level Autotune trial and decision methods now require explicit `lease=`
tokens. The CLI acquires and renews its lease automatically. Back up an existing
ledger and stop old runners before upgrading. Schema v4 transactionally migrates
v3 evidence and blocks writes by already-open legacy connections. Closed
read-only v3 inspection remains available. See the
[ownership and migration guide](optimize.md#campaign-ownership).

Smythe remains pre-1.0; minor versions may include API changes. This release
does not alter existing benchmark archives or retrospectively resolve missing
provider receipts. Screensaver distribution remains source-only.

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
Package README links resolve against `v0.8.0`. The release workflow retains
the actual published distributions and SHA-256 hashes; verify downloaded PyPI
bytes against that receipt as described in [Releasing](../RELEASING.md).
