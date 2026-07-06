---
name: Release checklist
about: Steps for cutting a release to PyPI
title: "Release vX.Y.Z"
labels: roadmap
---

# Release Checklist

## Preflight

- [ ] `pyproject.toml` version has been bumped (PyPI never allows re-uploading a version).
- [ ] `smythe/__init__.py` `__version__` matches.
- [ ] `CHANGELOG.md` has a section for this version.
- [ ] README install instructions mention `pip install smythe`.
- [ ] Tests pass locally.
- [ ] Build passes locally with `python -m build`.
- [ ] Package validates with `python -m twine check dist/*`.
- [ ] Built wheel installs and imports in a fresh venv.

## GitHub

- [ ] `.github/workflows/publish.yml` exists on the default branch.
- [ ] GitHub environment `pypi` exists (Settings → Environments).
- [ ] Release notes are drafted.
- [ ] Git tag matches the package version, e.g. `v0.2.0`.

## PyPI

- [ ] Pending/trusted publisher exists for `petehottelet/smythe`.
- [ ] Publisher workflow is `publish.yml`.
- [ ] Publisher environment is `pypi`.
- [ ] Release workflow publishes successfully.
- [ ] `pip install smythe` works from a clean venv after release.
