# Packages, source checkouts and typing

These distribution improvements ship in [Smythe 0.8.0](release-0.8.0.md).
Historical 0.7.0 artifacts remain unchanged.

## Choose a distribution

| Need | Distribution |
|---|---|
| Use Smythe's Python API or CLI | Install the wheel with `pip install smythe` and the required provider extra |
| Build and test the library from source | Use the source distribution and its explicit library test profile |
| Reproduce benchmarks | Clone the repository or download its GitHub source archive |
| Build the Noumenon screensaver | Clone the [Noumenon repository](https://github.com/petehottelet/noumenon) |
| Audit another project with Repo Doctor | Use the separately verified skill ZIP and its exact runtime pin |

The wheel contains Smythe, its typing marker, license and package metadata.
The source distribution adds tests, build and audit tools, README, changelog
and build configuration. Benchmark archives, media and skills stay in the
repository. The `benchmarks` extra installs dependencies;
the benchmark scripts themselves require a checkout.

## Build and verify a source package

From the repository root:

```bash
python -m pip install build hatchling
python -m build --outdir dist
python tools/distribution.py --dist dist
python tools/package_smoke.py --dist dist --work /path/to/fresh/package-check
```

Use fresh output directories. The audit enforces an exact file inventory,
a 1,000,000-byte compressed source limit and a 400,000-byte wheel limit.
It checks metadata, CLI entry points, release-tag README links and matching
package bytes. The smoke check compares checkout and source-rebuilt wheels,
installs into a clean environment, runs an offline graph and CLI, verifies
consumer types, then runs the source archive's tests. It installs test/build
dependencies from the package index but makes no provider API calls.

When adding or removing library, test or maintenance files, stage those
specific paths and update the reviewed inventory:

```bash
python tools/distribution.py --write-manifest
```

Review `tools/distribution-files.json` before committing it. Unexpected files
inside allowed folders still fail the archive audit. Keep internal plans,
review notes and local measurements in gitignored `00_project_files/`.

## Test an extracted source distribution

From its extracted root:

```bash
python -m pip install ".[dev]" hatchling
python -m pytest tests/ --distribution -q
```

The explicit profile excludes repository-only test modules before importing
them and deselects the committed example-manifest case. Its complete inventory
and reasons are in [the profile manifest](../tests/distribution_profile.json).
Library behavior, budgets, recovery, journals, tools and CLI tests remain
enabled. Unknown missing imports still fail collection.

Normal `python -m pytest tests/ -q` requires the full repository and preserves
its coverage. It fails if required repository materials disappear. Nothing
switches test profiles automatically because a directory is missing.

## Python typing

The package includes `py.typed`. Public consumer checks verify `Task`, `Swarm`,
`SwarmResult`, graph types and awaited results; three deliberately invalid
calls must produce argument-type errors. They run against an installed wheel
outside the source tree, so successful imports alone cannot satisfy the gate.

Run `python -m mypy` to check the explicit strict module set in
`pyproject.toml`. Imported signatures remain visible; internal errors in
modules outside that set are not part of the blocking gate. Expand that set
only after resolving the next module's diagnostics.

## Repo Doctor archive

```bash
python tools/skill_archive.py --out /path/to/fresh/skill-dist
```

The builder reads committed files from one exact commit, excludes untracked
and working-tree changes, and refuses to overwrite existing candidates.
The ZIP carries the MIT license, source commit, exact runtime requirement and
per-file hashes. The adjacent JSON receipt records archive bytes and SHA-256.

CI installs the exact published runtime in `skills/repo-doctor/runtime.txt`
and runs the extracted skill against an offline fixture with provider keys
cleared. The release workflow attaches verified assets without replacing old
ones; manual workflow runs retain downloadable candidates.
[Installation and offline usage](../skills/repo-doctor/README.md).

For a smaller contributor checkout and evidence retention rules, see
[Contributing](../CONTRIBUTING.md#working-with-a-slim-checkout).
