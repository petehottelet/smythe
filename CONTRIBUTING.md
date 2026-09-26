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
ruff check .
python -m mypy
pytest tests/ -q
```

CI runs these checks across the supported Python matrix. All required checks
must pass before merge. Python test processes block external socket connections
and DNS lookups, including when provider keys are present. Loopback SDK wire
fixtures and local MCP IPC remain available. Subprocess examples use their own
offline fixtures; the socket guard is not an operating-system sandbox.

Paid connectivity probes live outside `tests/`. To explicitly authorize one
short request after installing the chosen SDK, run:

```bash
python -m tools.provider_probe --provider openai --model YOUR_TEXT_MODEL --allow-paid
```

The probe supports `openai`, `anthropic`, and `gemini`. It uses a fixed prompt,
128 output tokens, no retries and a 30-second deadline. These bound the request,
not its exact dollar cost. It is never invoked by pytest or CI. Gemini uses the
[SDK retry and timeout controls](https://googleapis.github.io/python-genai/).

### Working with a slim checkout

For library work, fetch history lazily and omit large artwork and benchmark
trees from the checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/petehottelet/smythe.git
cd smythe
git sparse-checkout set smythe tests tools docs examples
pip install -e ".[dev]"
python -m pytest tests/ --distribution -q
```

`--distribution` selects the documented library test profile; it does not
claim to exercise omitted benchmarks or skills. The normal
test command requires the full checkout and fails on missing materials.
See [package contents and verification](docs/distribution.md).

For all current files with lazy historical blobs:

```bash
git clone --filter=blob:none https://github.com/petehottelet/smythe.git
```

Partial cloning defers historical blob downloads. Sparse checkout omits
current files; full checkout still downloads the current tree. Transfer and
disk use depend on the revision and subsequent commands.

### Evidence files

Keep committed benchmark records and existing evidence bytes in place.
New binary evidence over 1 MiB should use a retained release asset with a
small manifest recording its source revision, byte size, SHA-256 and download
URL. Include download verification and offline reproduction instructions,
and retain a backup: checksums detect changes but cannot recover deleted bytes.
Keep README display images available at their repository paths.

CI warns when an ordinary push or pull request introduces a binary file above
that threshold. Deliberate history rewrites need a separate history audit;
their prior commit may no longer be fetchable. Review
decides whether its size and storage location are justified; the warning is
not a hard limit. Internal implementation plans, project reviews and working
notes always belong in gitignored `00_project_files/`.

## Project conventions

- **Style:** [`ruff`](https://docs.astral.sh/ruff/) governs formatting and
  lint. Run it before pushing.
- **Type hints:** all public APIs are fully type-hinted. New code should match.
- **Execution:** serial and concurrent schedulers share provider, accounting
  and persistence machinery. Preserve their ordering, cancellation and
  recovery contracts when changing either path.
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
