"""Audit a local repository for open-source release readiness with smythe.

    python audit_repo.py <path> [--output audit-output] [--target pypi] [--strictness alpha]

Collects a repo snapshot (never executing repo code), has smythe plan
and run a specialist review graph, and writes PROJECT_SCORECARD.md,
NEXT_STEPS.md, graph.mmd, and trace.json to the output directory.

Runs offline by default with a canned specialist-fork -> red-team ->
synthesis plan; set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY
to plan and execute against a real model instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_repo_snapshot import collect_snapshot  # noqa: E402

from smythe import Swarm, Task  # noqa: E402
from smythe.graph import ExecutionGraph  # noqa: E402
from smythe.prompts import PLANNING_SYSTEM_PROMPT  # noqa: E402
from smythe.provider import (  # noqa: E402
    AnthropicProvider,
    GeminiProvider,
    OfflineProvider,
    OpenAIProvider,
    Provider,
)

MAX_BUDGET_USD = 0.75
MAX_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Canned offline plan: intake -> specialist fork -> red-team -> synthesis
# ---------------------------------------------------------------------------

_LABELS = {
    "intake": "Identify the project type, ecosystem, maturity, and release target",
    "packaging": "Review packaging metadata and release configuration",
    "readme": "Review README positioning, onboarding, and quickstart",
    "testing-ci": "Review tests, CI coverage, and reproducibility",
    "security-license": "Review license, security policy, and secret hygiene",
    "docs-examples": "Review documentation depth and runnable examples",
    "red-team": "Challenge inflated scores and unsupported claims",
    "synthesis": "Synthesize the final rating, scorecard, and action plan",
}

_AGENTS = {
    "intake": ("RepoIntakeAgent", "You identify project type, maturity, and release targets."),
    "packaging": ("PackagingAgent", "You review packaging metadata and release configuration."),
    "readme": ("ReadmeAgent", "You evaluate README positioning and onboarding."),
    "testing-ci": ("TestingCIAgent", "You review tests, CI, and reproducibility."),
    "security-license": ("SecurityLicenseAgent", "You check license and security hygiene."),
    "docs-examples": ("DocsExamplesAgent", "You evaluate docs depth and runnable examples."),
    "red-team": ("RedTeamAgent", "You challenge inflated scores and weak claims."),
    "synthesis": ("SynthesisAgent", "You produce the final rating and ordered action plan."),
}

_SPECIALISTS = ("packaging", "readme", "testing-ci", "security-license", "docs-examples")


def _offline_plan() -> dict:
    def node(node_id: str, depends_on: list[str]) -> dict:
        name, persona = _AGENTS[node_id]
        return {
            "id": node_id,
            "label": _LABELS[node_id],
            "depends_on": depends_on,
            "agent": {"name": name, "persona": persona, "capabilities": ["review"]},
        }

    nodes = [node("intake", [])]
    nodes += [node(s, ["intake"]) for s in _SPECIALISTS]
    nodes.append(node("red-team", list(_SPECIALISTS)))
    nodes.append(node("synthesis", ["red-team"]))
    return {"topology": ["fork_join", "adversarial"], "nodes": nodes}


class RepoDoctorOfflineProvider(OfflineProvider):
    """OfflineProvider preloaded with the audit plan and per-step findings."""

    def __init__(self, findings: dict[str, str]) -> None:
        super().__init__(plan=_offline_plan())
        self._findings = findings

    async def complete(self, system, prompt, model):
        result = await super().complete(system, prompt, model)
        if system != PLANNING_SYSTEM_PROMPT:
            result.text = self._findings.get(self.calls[-1], result.text)
        return result


def _offline_findings(snapshot: dict, overall: float, caps: list[tuple[float, str]]) -> dict:
    """Deterministic specialist findings derived from the snapshot signals."""
    s = snapshot["signals"]
    d = snapshot["detected"]
    has_readme, packaged, _ = _facts(snapshot)

    def flag(key: str) -> str:
        return "present" if s[key] else "missing"

    def word(value: bool) -> str:
        return "present" if value else "missing"

    suspects = snapshot["risks"]["secret_suspects"]
    cap_text = "; ".join(reason for _, reason in caps) or "none"
    return {
        _LABELS["intake"]: (
            f"{snapshot['repo']['name']}: ecosystems {', '.join(d['ecosystems'])}; "
            f"release targets {', '.join(d['release_targets'])}."
        ),
        _LABELS["packaging"]: (
            f"Packaging metadata {word(packaged)}; "
            f"release workflow {flag('has_release_workflow')}."
        ),
        _LABELS["readme"]: (
            f"README {word(has_readme)}; changelog {flag('has_changelog')}."
        ),
        _LABELS["testing-ci"]: f"Tests {flag('has_tests')}; CI {flag('has_ci')}.",
        _LABELS["security-license"]: (
            f"License {flag('has_license')}; security policy {flag('has_security_policy')}; "
            f"{len(suspects)} likely secret(s) found and redacted."
        ),
        _LABELS["docs-examples"]: (
            f"Examples {flag('has_examples')}; benchmarks {flag('has_benchmarks')}."
        ),
        _LABELS["red-team"]: f"Hard caps triggered: {cap_text}. Score held to {overall}/10.",
        _LABELS["synthesis"]: f"{overall}/10 — {_verdict(overall)}",
    }


# ---------------------------------------------------------------------------
# Rubric scoring (see references/scoring_rubric.md)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "Positioning and purpose": 0.10,
    "README and onboarding": 0.15,
    "Installability and quickstart": 0.10,
    "Code structure": 0.10,
    "Tests and CI": 0.15,
    "Packaging and release readiness": 0.15,
    "Docs and examples": 0.10,
    "Security/license/community hygiene": 0.10,
    "Proof/benchmarks/adoption": 0.05,
}


def _facts(snapshot: dict) -> tuple[bool, bool, bool]:
    """(has_readme, packaged, has_secret_suspects) — shared by scoring and caps."""
    important = snapshot["files"]["important_files"]
    has_readme = any(n.upper().startswith("README") for n in important)
    packaged = any(
        n in important
        for n in ("pyproject.toml", "setup.py", "package.json", "Cargo.toml", "go.mod")
    )
    return has_readme, packaged, bool(snapshot["risks"]["secret_suspects"])


def score_categories(snapshot: dict) -> dict[str, tuple[int, str]]:
    """Heuristic 1-10 score per rubric category, with a one-line evidence note."""
    s = snapshot["signals"]
    has_readme, packaged, has_secrets = _facts(snapshot)

    def pick(flag: bool, hi: tuple[int, str], lo: tuple[int, str]) -> tuple[int, str]:
        return hi if flag else lo

    if s["has_tests"] and s["has_ci"]:
        tests = (8, "Tests and CI are both present.")
    elif s["has_tests"]:
        tests = (6, "Tests exist but no CI runs them.")
    else:
        tests = (3, "No test suite detected.")

    if packaged and s["has_release_workflow"]:
        packaging = (8, "Packaging metadata and a release workflow are present.")
    elif packaged:
        packaging = (6, "Packaging metadata present; no release workflow.")
    else:
        packaging = (3, "No packaging metadata found.")

    if s["has_license"] and s["has_security_policy"] and not has_secrets:
        hygiene = (8, "License and security policy present; no likely secrets.")
    elif s["has_license"] and not has_secrets:
        hygiene = (6, "License present; no security policy.")
    elif s["has_license"]:
        hygiene = (4, "Likely secrets detected in the tree (redacted in snapshot).")
    else:
        hygiene = (3, "No license found.")

    return {
        "Positioning and purpose": pick(
            has_readme, (7, "README states what the project is."),
            (3, "No README to position the project."),
        ),
        "README and onboarding": pick(
            has_readme, (7, "README present with onboarding content."), (2, "No README."),
        ),
        "Installability and quickstart": pick(
            packaged, (7, "Standard packaging metadata found."),
            (4, "No standard install path detected."),
        ),
        "Code structure": pick(
            s["has_tests"], (7, "Source and tests are separated."),
            (5, "Structure is readable but unproven by tests."),
        ),
        "Tests and CI": tests,
        "Packaging and release readiness": packaging,
        "Docs and examples": pick(
            s["has_examples"], (7, "Runnable examples are present."),
            (4, "No examples or demo directory."),
        ),
        "Security/license/community hygiene": hygiene,
        "Proof/benchmarks/adoption": pick(
            s["has_benchmarks"], (7, "Benchmark artifacts exist."),
            (3, "No benchmark or proof artifacts."),
        ),
    }


def hard_caps(snapshot: dict) -> list[tuple[float, str]]:
    """Rubric hard caps triggered by this snapshot."""
    has_readme, packaged, _ = _facts(snapshot)
    s = snapshot["signals"]
    caps = []
    if not has_readme:
        caps.append((5.0, "No README (capped at 5)."))
    if not packaged:
        caps.append((6.0, "No packaging metadata; not installable as released (capped at 6)."))
    if not s["has_license"]:
        caps.append((6.0, "No license (capped at 6)."))
    if not s["has_tests"]:
        caps.append((7.0, "No tests (capped at 7)."))
    if not s["has_ci"]:
        caps.append((8.0, "No CI (capped at 8)."))
    return caps


def overall_score(scores: dict[str, tuple[int, str]], caps: list[tuple[float, str]]) -> float:
    weighted = sum(scores[name][0] * weight for name, weight in _WEIGHTS.items())
    return round(min([weighted] + [cap for cap, _ in caps]), 1)


def _verdict(overall: float) -> str:
    if overall >= 9:
        return "Strong, well-evidenced project; keep raising the proof bar."
    if overall >= 8:
        return "Publishable and useful, with solid open-source hygiene."
    if overall >= 7:
        return "Credible project; missing proof, polish, or adoption readiness."
    if overall >= 5:
        return "Usable early project with clear gaps in tests, docs, packaging, or release flow."
    if overall >= 3:
        return "Prototype; basic onboarding, structure, or runnable proof is missing."
    return "Not meaningfully reviewable yet."


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

_GAP_STEPS = (
    ("has_license", "Add a LICENSE",
     "Without a license the project is not legally usable by anyone else.",
     "Pick MIT or Apache-2.0 and commit LICENSE at the repo root.",
     "LICENSE exists and is referenced from the packaging metadata."),
    ("has_tests", "Add a test suite",
     "Untested code cannot be trusted or refactored safely.",
     "Add tests/ covering the core happy path and one failure path.",
     "Tests run green locally with one documented command."),
    ("has_ci", "Add CI",
     "Without CI, green tests are a claim rather than a guarantee.",
     "Add .github/workflows/ci.yml running lint and tests on push.",
     "CI runs on every push and the badge/status is visible."),
    ("has_changelog", "Add a CHANGELOG",
     "Users cannot upgrade confidently without release history.",
     "Add CHANGELOG.md and record every released version.",
     "CHANGELOG.md lists the current version with dated entries."),
    ("has_security_policy", "Add a security policy",
     "Reporters need a private disclosure path before you publish.",
     "Add SECURITY.md naming a contact and response expectation.",
     "SECURITY.md exists and is linked from the repo settings."),
    ("has_examples", "Add runnable examples",
     "Examples are the fastest proof that the project actually works.",
     "Add examples/ with at least one copy-paste runnable script.",
     "A new user can run one example in under three minutes."),
    ("has_release_workflow", "Add a release workflow",
     "Manual releases are error-prone and hard to reproduce.",
     "Add a tag-triggered workflow that builds and publishes the package.",
     "Pushing a version tag produces a published release."),
)

_GAP_FILES = {
    "has_license": ("LICENSE", "Required for open-source use."),
    "has_security_policy": ("SECURITY.md", "Gives reporters a disclosure path."),
    "has_changelog": ("CHANGELOG.md", "Documents release history."),
    "has_ci": (".github/workflows/ci.yml", "Runs tests on every push."),
    "has_tests": ("tests/", "Proves the code works."),
    "has_examples": ("examples/", "Shows users a working entry point."),
}


def next_steps(snapshot: dict) -> list[tuple[str, str, str, str]]:
    s = snapshot["signals"]
    steps = [
        (title, why, how, done)
        for flag, title, why, how, done in _GAP_STEPS
        if not s[flag]
    ]
    if not steps:
        steps = [(
            "Publish and gather adoption proof",
            "Hygiene is in place; credibility now comes from users and benchmarks.",
            "Cut a release, announce it, and publish a reproducible benchmark.",
            "At least one external user has run the released version.",
        )]
    return steps[:5]


def suggested_files(snapshot: dict) -> list[tuple[str, str]]:
    s = snapshot["signals"]
    return [(path, reason) for flag, (path, reason) in _GAP_FILES.items() if not s[flag]]


def render_scorecard(
    snapshot: dict,
    scores: dict[str, tuple[int, str]],
    caps: list[tuple[float, str]],
    overall: float,
    synthesis: str,
    mode: str,
) -> str:
    name = snapshot["repo"]["name"]
    working = [f"{area}: {note}" for area, (score, note) in scores.items() if score >= 7]
    blockers = [reason for _, reason in caps] or [
        f"{area}: {note}" for area, (score, note) in scores.items() if score <= 4
    ]
    if overall >= 8:
        readiness = "Ready."
    elif overall >= 6:
        readiness = "Ready after fixes."
    else:
        readiness = "Not ready."

    lines = [
        f"# Project Audit: {name}", "",
        "## Overall Rating", "",
        f"**{overall}/10** — {_verdict(overall)}", "",
        "## Summary", "",
        f"{name} was audited {mode} against the Repo Doctor rubric "
        f"(references/scoring_rubric.md). Category scores are computed from repository "
        f"signals in the snapshot; {len(caps)} hard cap(s) applied.", "",
        "## Scorecard", "",
        "| Area | Score | Notes |",
        "|---|---:|---|",
    ]
    lines += [f"| {area} | {score} | {note} |" for area, (score, note) in scores.items()]
    lines += ["", "## What Is Working", ""]
    lines += [f"{i}. {item}" for i, item in enumerate(working, 1)] or [
        "1. Nothing scores 7 or higher yet."
    ]
    lines += ["", "## Main Blockers", ""]
    lines += [f"{i}. {item}" for i, item in enumerate(blockers, 1)] or [
        "1. No hard blockers detected."
    ]
    lines += ["", "## Highest-Impact Next Steps", ""]
    for i, (title, why, how, done) in enumerate(next_steps(snapshot), 1):
        lines += [
            f"### {i}. {title}", "",
            f"Why it matters: {why}",
            f"How to do it: {how}",
            f"Acceptance criteria: {done}", "",
        ]
    lines += ["## Suggested Files to Add or Update", ""]
    suggestions = suggested_files(snapshot)
    if suggestions:
        lines += ["| File | Action | Reason |", "|---|---|---|"]
        lines += [f"| `{path}` | Add | {reason} |" for path, reason in suggestions]
    else:
        lines.append("Nothing missing from the standard open-source file set.")
    lines += [
        "", "## Release Readiness", "", readiness, "",
        "## Final Verdict", "", synthesis or _verdict(overall), "",
    ]
    return "\n".join(lines)


def render_next_steps(snapshot: dict, overall: float) -> str:
    name = snapshot["repo"]["name"]
    lines = [
        "# Next Steps", "",
        "## Goal", "",
        f"Move {name} from {overall}/10 toward a publishable, well-evidenced release.", "",
    ]
    for i, (title, why, how, done) in enumerate(next_steps(snapshot), 1):
        lines += [
            f"## Priority {i}: {title}", "",
            f"- [ ] {how}", "",
            "Acceptance criteria:", "",
            f"- {done}", "",
        ]
    lines += [
        "## Done When", "",
        "- Every priority above is checked off.",
        f"- {name} re-audits at 8+ on the Repo Doctor rubric.", "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _pick_real_provider() -> tuple[Provider, str] | None:
    """A real provider when an API key is set, mirroring examples/_providers.py."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(), "claude-opus-4-8"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        return GeminiProvider(), "gemini-flash-latest"
    return None


def _final_result(graph: ExecutionGraph) -> str:
    """Result of the terminal (synthesis) node, whatever the planner named it."""
    depended = {dep for node in graph.nodes for dep in node.depends_on}
    terminals = [n for n in graph.nodes if n.id not in depended and n.result]
    return str(terminals[-1].result) if terminals else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="local repository directory to audit")
    parser.add_argument("--output", default="audit-output", help="directory for artifacts")
    parser.add_argument("--target", default="pypi", help="target release channel")
    parser.add_argument("--strictness", default="alpha", help="strictness level")
    args = parser.parse_args(argv)

    snapshot = collect_snapshot(Path(args.path))
    scores = score_categories(snapshot)
    caps = hard_caps(snapshot)
    overall = overall_score(scores, caps)

    real = _pick_real_provider()
    if real is not None:
        provider, model = real
        mode = f"with model {model}"
    else:
        provider = RepoDoctorOfflineProvider(_offline_findings(snapshot, overall, caps))
        model = "offline-model"
        mode = "offline (deterministic canned smythe graph)"

    swarm = Swarm(
        provider=provider,
        model=model,
        parallel=True,
        max_budget_usd=MAX_BUDGET_USD,
        max_concurrency=MAX_CONCURRENCY,
        artifact_dir=None,
    )
    task = Task(
        goal=(
            "Audit this repository for open-source release readiness. Produce a 1-10 "
            "rating, category scorecard, release blockers, and concrete next steps.\n\n"
            f"Repository snapshot (JSON):\n{json.dumps(snapshot)}"
        ),
        constraints=[
            "Do not execute repository code.",
            "Do not expose secrets; redact any likely secret values.",
            "Ground claims in the repository snapshot.",
            "Use adversarial review before finalizing any score of 7 or higher.",
            "Final output must be valid markdown.",
        ],
        context={"target_release": args.target, "strictness": args.strictness},
    )
    result = swarm.execute(task)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "PROJECT_SCORECARD.md": render_scorecard(
            snapshot, scores, caps, overall, _final_result(result.graph), mode,
        ),
        "NEXT_STEPS.md": render_next_steps(snapshot, overall),
        "graph.mmd": result.graph.to_mermaid(theme=True) + "\n",
        "trace.json": json.dumps(result.trace, indent=2) + "\n",
    }
    for filename, text in artifacts.items():
        (out / filename).write_text(text, encoding="utf-8")
        print(f"Wrote {out / filename}")
    print(f"\nOverall rating: {overall}/10 (cost ${result.total_cost_usd:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
