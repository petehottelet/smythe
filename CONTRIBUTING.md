# Contributing to smythe

Thanks for your interest in improving smythe. This document covers how to get
set up, the conventions the project follows, and what to expect when you open
a PR.

## Code of conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
Reports go to the address listed there.

## Ways to contribute

- **Bug reports** — open an issue using the bug template. A minimal reproduction
  helps a lot; a failing test helps even more.
- **Feature proposals** — open an issue using the feature template before
  starting significant work, so we can discuss scope and fit.
- **Documentation** — README clarifications, docstring improvements, and
  worked examples are always welcome.
- **Code** — see "Development setup" below.

## Development setup

The project supports Python 3.11, 3.12, and 3.13.

### Recommended: `uv`

[`uv`](https://docs.astral.sh/uv/) is the fastest path. From the repo root:

```bash
uv venv
uv pip install -e ".[dev,benchmarks]"
```

### Alternative: `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,benchmarks]"
```

The `dev` extra installs the test and lint tools. The `benchmarks` extra adds
the provider SDKs, Pillow, LangGraph, and CrewAI needed to reproduce every
published harness. Install individual provider extras instead when you only
need a smaller local development environment.

## Running tests and lint

```bash
ruff check smythe/ tests/ benchmarks/ examples/
pytest tests/ -q
```

CI runs the same commands across the supported Python matrix. PRs must be
green on both before merge.

## Project conventions

- **Style:** [`ruff`](https://docs.astral.sh/ruff/) governs formatting and
  lint. Run it before pushing.
- **Type hints:** all public APIs are fully type-hinted. New code should match.
- **Async first:** core executors are async-native. If you add a sync wrapper,
  keep the async path the source of truth.
- **No emojis** in commit messages, code, or docs unless explicitly requested.
- **No "created with X" attribution** in commit messages.
- **Tests live next to features.** New behavior needs a test. Bug fixes need
  a regression test that fails without the fix.
- **Shared test fixtures** live in [tests/helpers.py](tests/helpers.py).
  Prefer reusing them over re-defining mock providers.

## Pull request workflow

1. Open an issue first for non-trivial changes — it saves rework.
2. Fork, branch, commit. Keep commits focused; squash on merge if you accumulate
   noisy WIP commits.
3. Update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]` for any user-facing
   change.
4. Open the PR using the template. Fill in the summary, the testing notes, and
   any breaking-change callouts.
5. CI must be green. Reviewers may ask for changes; please respond inline rather
   than force-pushing over old comments where possible.

## Reporting security issues

Please do **not** open public issues for security-sensitive bugs. See
[SECURITY.md](SECURITY.md) for the disclosure process.

## Versioning

The project follows [SemVer](https://semver.org/) with a pre-1.0 stability
note documented at the top of [CHANGELOG.md](CHANGELOG.md). On a `0.x` line,
minor bumps may include breaking changes (each one called out in the changelog);
patch bumps are non-breaking.
