# Repo Doctor

Repo Doctor audits a local repository and tells you how close it is to being a
credible open-source release: a 1-10 rating, a category scorecard with evidence,
hard-cap blockers, and an ordered `NEXT_STEPS.md`.

## Quickstart (offline, no API key)

From the smythe repo root:

```bash
python skills/repo-doctor/scripts/audit_repo.py . --output audit-output
```

Writes `PROJECT_SCORECARD.md`, `NEXT_STEPS.md`, `graph.mmd` (the agent graph as
Mermaid), and `trace.json` (per-node execution spans) to `audit-output/`. Set
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` to run the same
command against a real model.

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
