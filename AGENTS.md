# Repository working agreement

These instructions apply to the entire repository.

## Product story

Smythe has two defining abstractions:

1. **Generated execution topology** — a goal becomes an inspectable,
   task-specific DAG.
2. **A durable execution envelope** — budgets, bounded concurrency, traces,
   verification, artifacts, and recovery govern that DAG.

Keep public documentation centered on those two ideas. The README is the
concise product page; `docs/index.md` is the documentation map; detailed
behavior belongs in the linked guide for that subsystem.

## Evidence and claims

- Treat committed benchmark records as the source of truth for numbers.
- A README headline may cite only a current campaign that its benchmark report
  marks as claimable and free of known measurement defects.
- Keep diagnostic and superseded campaigns in the repository, but label their
  evidence status and do not promote them as current results.
- Prefer exact, positive claims such as “19% lower measured cost” over broad
  superlatives. Preserve scope in the linked protocol.
- Render landing-page charts from committed records with
  `python benchmarks/render_readme_charts.py`.

## Documentation consistency

- Update the README, `docs/index.md`, ROADMAP, CHANGELOG, examples index, and
  subsystem guide together when a public feature, status, command, or benchmark
  changes.
- Keep shipped counts, model/provider names, CLI commands, and artifact links
  consistent across those surfaces.
- Put unreleased changes only under `## [Unreleased]` in `CHANGELOG.md`; released
  sections are immutable.
- Use the visual rules in `docs/style.md` for diagrams and landing-page assets.
- Validate local Markdown links after broad documentation edits.

## Code quality

- Read a file in full before making a wide-ranging edit to it.
- Add regression tests for every fixed defect and run each modified test file.
- Run `ruff check .` and the full offline test suite before handing off a code
  change. Paid provider calls never belong in tests.
- Keep public behavior in code, tests, documentation, and examples aligned.

## Git safety

- Preserve unrelated user changes.
- Stage only files changed for the current task; never use `git add .` or
  `git add -A`.
- Never use destructive reset, checkout, clean, or force-push operations.
