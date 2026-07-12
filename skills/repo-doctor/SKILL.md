---
name: repo-doctor
summary: Audit a local repository for open-source release readiness using smythe specialist agents.
description: Use this skill when the user asks to rate, audit, review, improve, package, publish, or generate next steps for a local software repository. It produces a scorecard, release-readiness review, and a concrete improvement plan. Do not use it for general code debugging unless the user asks for a repo-level project audit.
---

# Repo Doctor

## What this skill does

Audits a software repository for clarity, installability, packaging, tests, CI,
docs, security/license hygiene, examples, and release readiness, then writes a
scorecard and an ordered action plan. Reviews run as a smythe agent graph:
intake, a fork of specialist reviewers, an adversarial red-team pass, and a
final synthesis.

## Required input

- A local repository path. (GitHub URL support is planned; clone manually for now.)

## Optional input

- Target release channel (`--target`, default `pypi`).
- Strictness level (`--strictness`, default `alpha`).
- Output directory (`--output`, default `audit-output`).

## Workflow

1. Confirm or infer the project type and target release channel.
2. Collect a repository snapshot without executing project code:
   `python scripts/collect_repo_snapshot.py <path>`
3. Run the audit: `python scripts/audit_repo.py <path> --output <dir>`.
   Offline by default; set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
   to plan and execute with a real model.
4. Read `PROJECT_SCORECARD.md` and relay the rating, blockers, and next steps.
5. Show `graph.mmd` when the user wants to see the agent topology; `trace.json`
   holds per-node spans.
6. Score against `references/scoring_rubric.md` — apply its hard caps.

## Output requirements

Always include:

- Overall rating (1-10) and a short verdict.
- Category scorecard with evidence notes.
- Strengths and blockers.
- Concrete next steps with acceptance criteria.
- Release-readiness judgment (Ready / Ready after fixes / Not ready).

## Guardrails

- Do not reveal secrets; the snapshot redacts likely secret values and reports
  only file and redacted stub.
- Do not execute target repo code or install its dependencies.
- Do not modify the audited repo; artifacts go to the output directory only.
- Cite file evidence from the snapshot when available.
- Be direct, practical, and specific; make uncertainty visible.
