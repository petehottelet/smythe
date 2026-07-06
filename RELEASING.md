# Releasing smythe to PyPI

Publishing uses PyPI **trusted publishing** (GitHub OIDC, no API token). The five names that must align exactly — a mismatch fails with `invalid-publisher`:

| What | Value |
|---|---|
| PyPI project name | `smythe` |
| `pyproject.toml` name | `smythe` |
| GitHub owner/repo | `petehottelet/smythe` |
| Workflow file | `.github/workflows/publish.yml` |
| Workflow environment | `pypi` |

## One-time setup

1. **PyPI pending publisher** — configured (project `smythe`, workflow `publish.yml`, environment `pypi`). ✅
2. **GitHub environment** — Repository → Settings → Environments → New environment → `pypi`. No protection rules needed for the first release.
3. **Workflow on the default branch** — `publish.yml` triggers on GitHub *release published* and on manual `workflow_dispatch`; both require the workflow to exist on `main`, so merge the branch that carries it before releasing.

## Per-release flow

1. Bump `version` in `pyproject.toml` **and** `__version__` in `smythe/__init__.py` (PyPI never accepts a re-upload of an existing version). Add the CHANGELOG section.
2. Local check:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   # then install the wheel in a scratch venv and:
   python -c "import smythe; print(smythe.__version__)"
   ```
3. Open a release-checklist issue (template: "Release checklist") and work through it.
4. Create the GitHub release — tag `vX.Y.Z` matching the package version, target `main`. Publishing the release triggers `publish.yml`.
5. Verify from a clean venv: `pip install smythe`, import it, and `pip index versions smythe`.

## Note on the first publish

The GitHub release **v0.1.0 already exists** (April 2026) and predates the publish workflow, so it will not trigger anything. Two options for the first PyPI upload:

- **Recommended:** merge the v0.2 line, bump to `0.2.0`, and create release `v0.2.0` — the first PyPI version is then the substantially stronger current code.
- Quick path: run `publish.yml` via *workflow_dispatch* on `main`, which uploads whatever version `main` currently carries.

## If publishing fails

- `invalid-publisher` / `invalid-pending-publisher`: one of the five names above doesn't match — check filename, environment name, owner/repo, and package name first.
- Permission error: the job must have `permissions: id-token: write`.
- Metadata error: reproduce locally with `python -m build && python -m twine check dist/*`.
- "Version already exists": bump, commit, tag a new release. Never reuse a published version.
