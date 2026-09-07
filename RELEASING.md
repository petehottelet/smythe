# Releasing smythe to PyPI

Publishing uses PyPI **trusted publishing** (GitHub OIDC, no API token). The five names that must align exactly — a mismatch fails with `invalid-publisher`:

| What | Value |
|---|---|
| PyPI project name | `smythe` |
| `pyproject.toml` name | `smythe` |
| GitHub owner/repo | `petehottelet/smythe` |
| Workflow file | `.github/workflows/publish.yml` |
| Workflow environment | `pypi` |

## Publisher configuration

The repository publishes through `.github/workflows/publish.yml` and the
GitHub environment `pypi`. Keep the corresponding PyPI trusted publisher
aligned with the values above. The workflow runs when a GitHub release is
published or when a maintainer dispatches it manually.

## Per-release flow

1. Complete the release's code review, Ruff check, full offline suite, and
   platform checks. Confirm the tested commit is on `main`.
2. Bump `version` in `pyproject.toml` **and** `__version__` in `smythe/__init__.py`.
   Move the completed `Unreleased` entries into a dated section for that version;
   keep older released sections unchanged. Update installation and feature
   availability documentation. PyPI never accepts a re-upload of an existing version.
3. Build into a fresh output directory and validate the exact new artifacts:
   ```bash
   python -m pip install build twine
   python -m build --outdir release-dist
   python -m twine check release-dist/*
   # then install the wheel in a scratch venv and:
   python -c "import smythe; print(smythe.__version__)"
   ```
   The build-only README hook resolves local documentation and image links
   against `vX.Y.Z` for PyPI. The source README stays unchanged. The PyPI badge
   alone follows `main` with a generic label, so the badge update after
   publication also reaches the package description. Verify a wheel rebuilt
   from the sdist has identical package members and metadata; run
   `python -m pytest tests/test_pypi_readme.py` with Hatchling installed.
4. Open a release-checklist issue (template: "Release checklist") and record
   the tested commit, check URLs, package hashes, and release notes.
5. Push the version commit and verify its CI checks. Create tag `vX.Y.Z` at
   that exact commit and publish its GitHub release. This triggers `publish.yml`.
6. Attach the verified native screensaver packages with `SHA256SUMS` and
   `BUILD_INFO.json`. Their independent native version and source commit remain
   explicit in the build record; attaching them does not imply a new rebuild.
7. Verify the successful publish workflow. In a fresh environment, install
   `smythe==X.Y.Z` from PyPI, check its version, and exercise the installed CLI.
   Download the workflow's retained distributions and `package-hashes.txt`;
   confirm PyPI serves those exact wheel and sdist bytes. Build hosts can use
   different source newlines, so local candidate hashes do not substitute for
   the publication workflow's artifact hashes.
   Update the PyPI badge only after the published version is available.

## If publishing fails

- `invalid-publisher` / `invalid-pending-publisher`: one of the five names above doesn't match — check filename, environment name, owner/repo, and package name first.
- Permission error: the job must have `permissions: id-token: write`.
- Metadata error: reproduce locally with `python -m build && python -m twine check dist/*`.
- "Version already exists": bump, commit, tag a new release. Never reuse a published version.
