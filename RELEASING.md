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
published or when a maintainer dispatches it manually at a release tag.
Both paths require the tag to match the package version and successful
`ci.yml` push checks on `main` at the exact checked-out commit. A branch
dispatch, an untested commit or a failed CI run cannot publish.

Candidate branches can run the same platform matrix with
`gh workflow run ci.yml --ref BRANCH`. That manual qualification does not
replace the required successful push checks on `main` before publication.

The [0.9.0 release guide](docs/release-0.9.0.md) records its scope,
compatibility changes and verification commands.

## Per-release flow

1. Complete the release's code review, Ruff check, full offline suite, and
   platform checks. Confirm the tested commit is on `main`.
   Complete the [materials check](docs/current-materials.md#completion-check-for-every-benchmark-update)
   for contact sheets, media, charts, benchmark status, and documentation links.
2. Bump `version` in `pyproject.toml` **and** `__version__` in `smythe/__init__.py`.
   Move the completed `Unreleased` entries into a dated section for that version;
   keep older released sections unchanged. Update installation and feature
   availability documentation. PyPI never accepts a re-upload of an existing version.
3. Build into a fresh output directory and validate the exact new artifacts:
   ```bash
   python -m pip install build twine
   python -m build --outdir release-dist
   python -m twine check release-dist/*
   python tools/distribution.py --dist release-dist
   python tools/package_smoke.py --dist release-dist --work /path/to/fresh/package-check
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
6. Verify the successful publish workflow. In a fresh environment, install
   `smythe==X.Y.Z` from PyPI, check its version, and exercise the installed CLI.
   Download the workflow's retained distributions and `package-hashes.txt`;
   confirm PyPI serves those exact wheel and sdist bytes. Build hosts can use
   different source newlines, so local candidate hashes do not substitute for
   the publication workflow's artifact hashes.
   Update the PyPI badge only after the published version is available.

## Evidence assets

When a release retires benchmark evidence, build its archive from git history
and verify it before attaching it to the release its pointer names:

```bash
python tools/evidence_archive.py build SPEC.json --out /path/to/fresh/evidence --pointer benchmarks/archive/NAME.json
python tools/evidence_archive.py verify /path/to/fresh/evidence/NAME.zip --pointer benchmarks/archive/NAME.json --against-git
```

Commit the pointer with the release, upload the verified ZIP under the exact
name in its `release_asset` URL, and download it again to confirm the
SHA-256. Existing asset names are never overwritten.

Before a release, review the package's exact file manifest and run the
[distribution checks](docs/distribution.md). Internal plans and review notes
must remain outside every package and release asset.

For new binary benchmark evidence over 1 MiB, follow the
[evidence retention policy](CONTRIBUTING.md#evidence-files): retain source
identity, bytes, checksum, download URL, backup and offline reproduction
instructions. Existing committed evidence and README image paths stay intact.

## If publishing fails

- `invalid-publisher` / `invalid-pending-publisher`: one of the five names above doesn't match — check filename, environment name, owner/repo, and package name first.
- Permission error: the job must have `permissions: id-token: write`.
- Metadata error: reproduce locally with `python -m build && python -m twine check dist/*`.
- "Version already exists": bump, commit, tag a new release. Never reuse a published version.
