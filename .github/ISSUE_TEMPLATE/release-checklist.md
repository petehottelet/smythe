---
name: Release checklist
about: Steps for cutting a release to PyPI
title: "Release vX.Y.Z"
labels: roadmap
---

# Release Checklist

Follow [RELEASING.md](https://github.com/petehottelet/smythe/blob/main/RELEASING.md). Record the evidence for this release:

- Version and tested commit:
- Ruff, full offline suite, and platform check URLs/results:
- Wheel and sdist filenames with SHA-256 hashes:
- Installed-wheel/SDK quickstart receipt:
- Release notes and benchmark evidence links:
- Native asset build records and hashes, when attached:

## Preflight

- [ ] Code review, `ruff check .`, the full offline suite, and applicable platform checks pass for the tested commit on `main`.
- [ ] `pyproject.toml` and `smythe/__init__.py` declare the same new version; no published version is reused.
- [ ] Completed `Unreleased` entries move to a dated changelog section; older released sections remain unchanged.
- [ ] README, documentation map, roadmap, examples, subsystem guides, and release notes agree on installation, availability, commands, and evidence status.
- [ ] A fresh output directory contains only this release's wheel and sdist; `python -m twine check release-dist/*` passes.
- [ ] `tests/test_pypi_readme.py` passes with Hatchling installed; the sdist rebuild has identical wheel package members and metadata.
- [ ] PyPI README links resolve against the release tag; only the generically labeled PyPI badge follows `main`.
- [ ] The wheel installs, imports, reports the expected version, and exercises the installed CLI in a clean venv.
- [ ] The exact README quickstart has a retained installed-wheel/SDK check. Mocked usage is labeled synthetic, with zero paid calls; it is not a live model, cost, or quality result.
- [ ] Public benchmark claims cite eligible committed records. Astra paid execution still requires its campaign allowance and calibration gates; offline qualification does not unlock main or judging.

## GitHub

- [ ] The version commit is pushed and its CI checks pass; tag `vX.Y.Z` points to that tested commit.
- [ ] `.github/workflows/publish.yml` exists and uses the GitHub environment `pypi` with `id-token: write`.
- [ ] The GitHub release uses the matching tag and reviewed release notes.
- [ ] Attached native packages include `SHA256SUMS` and `BUILD_INFO.json`; their independent native version and source commit are explicit. Attaching an existing verified build does not imply a rebuild.

## PyPI

- [ ] Project `smythe` trusts `petehottelet/smythe`, workflow `publish.yml`, environment `pypi`.
- [ ] Release workflow publishes successfully.
- [ ] PyPI wheel and sdist hashes match the publication workflow's retained distributions and `package-hashes.txt`.
- [ ] `pip install smythe==X.Y.Z` succeeds from PyPI in a fresh venv; its version and installed CLI are checked.
- [ ] The PyPI badge is updated only after the new published version is available.
