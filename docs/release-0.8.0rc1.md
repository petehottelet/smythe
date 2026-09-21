# Smythe 0.8.0rc1 candidate

This checkout prepares a release candidate for review. **0.7.0 remains the
published release**; no 0.8.0 package or release tag has been published.

## What changes

- Native Claude Messages support with exact cache billing and durable saved
  responses. The Fable pilots and separate Code Workflow study retain their
  evidence; human pilot ratings still gate the native main comparison.
- Autotune campaign ownership, conservative recovery, and read-only HTML
  reports; stronger planner-history and concurrent SQLite initialization checks.
- Compact source distributions, an explicit library test profile, installed
  consumer typing checks, and separately packaged Repo Doctor assets.
- External-network blocking in Python tests and explicit bounded manual
  provider probes. Publishing requires a matching tag and successful CI at
  that exact main-branch commit.
- A product-focused README and completed Astra publication with an additive
  offline evidence reproduction. Historical measurements remain unchanged.

Full behavior and fixes are in [Unreleased](../CHANGELOG.md#unreleased).

## Compatibility

Low-level Autotune trial and decision methods now require explicit `lease=`
tokens. The CLI acquires and renews its lease automatically. Schema v4
transactionally migrates v3 evidence and blocks writes by already-open legacy
connections; closed read-only v3 inspection remains available. Read the
[ownership and migration guide](optimize.md#campaign-ownership-unreleased)
before using existing campaign databases with the candidate.

Smythe is pre-1.0, so this change belongs in a minor release. Freeze a backup
of an existing ledger before upgrading its runtime.

## Verify the candidate

From this checkout, install `.[dev,openai,anthropic]` and run:

```bash
python -m ruff check .
python -m mypy
python -m pytest tests/ -q
python examples/14_durable_text_workflow.py
```

Use [distribution verification](distribution.md) to build and audit a fresh
wheel/source pair, reproduce the wheel from the source archive and check an
installed package. Public README links in package metadata resolve against
`v0.8.0rc1`; that tag must exist before publishing the candidate to PyPI.

The release workflow enforces exact-commit main CI. Review the candidate,
merge the tested changes, then follow [the release process](../RELEASING.md)
for versioned notes, tags and publication. Keep screensaver downloads
source-only while precompiled distribution is paused.
