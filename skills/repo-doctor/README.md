# Repo Doctor

Repo Doctor audits a local repository and tells you how close it is to being a
credible open-source release: a 1-10 rating, a category scorecard with evidence,
hard-cap blockers, and an ordered `NEXT_STEPS.md`.

## Quickstart (offline, no API key)

The skill requires **`smythe==0.7.0`**, recorded in `runtime.txt`. Install that
exact runtime before using either the repository copy or a skill archive:

```bash
pip install "smythe==0.7.0"
```

From the smythe repo root:

```bash
python skills/repo-doctor/scripts/audit_repo.py . --output 00_project_files/repo-doctor
```

Writes `PROJECT_SCORECARD.md`, `NEXT_STEPS.md`, `graph.mmd` (the agent graph as
Mermaid), and `trace.json` (per-node execution spans) to the selected private output folder. Set
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` to run the same
command against a real model.

## Standalone ZIP (unreleased)

The distribution workflow builds `repo-doctor-skill-0.7.0.zip` and a matching
JSON checksum receipt from one exact commit. The ZIP includes this guide,
the skill, scripts, rubric, license, runtime pin and `MANIFEST.json` with
per-file hashes. CI verifies it against the pinned PyPI runtime outside the
repository with provider keys cleared. Release publication attaches the
verified ZIP and receipt; manual runs retain candidates as workflow artifacts.

Extract it, install the runtime above, then run:

```bash
python repo-doctor/scripts/audit_repo.py /path/to/project --output /path/to/project/00_project_files/repo-doctor
```

The default output folder is `00_project_files/repo-doctor/` beneath the
current working directory. The collector omits `00_project_files/` from
snapshots. Keep the output folder gitignored. With provider keys unset, the audit uses
the deterministic offline graph. No repository clone is needed for the ZIP.

## Why this uses smythe

Different repositories need different reviews: a PyPI-bound package needs
packaging and CI scrutiny, a prototype needs positioning and onboarding review,
and a framework with big claims needs benchmark evidence checked. Repo Doctor
expresses the audit as a smythe task, so the review runs as an agent graph —
intake, a parallel fork of specialists (packaging, README, tests/CI,
security/license, docs/examples), an adversarial red-team pass that challenges
inflated scores, then synthesis — under a hard budget cap, with the generated
topology and trace exported as part of the audit output. Offline mode runs the
same graph with a deterministic canned plan, so the orchestration is testable
without spending a token.
