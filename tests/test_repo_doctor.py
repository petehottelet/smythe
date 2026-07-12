"""Tests for the Repo Doctor skill: snapshot collector and offline audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "skills" / "repo-doctor" / "scripts"))

import audit_repo  # noqa: E402
from collect_repo_snapshot import collect_snapshot  # noqa: E402

FAKE_TOKEN = "ghp_" + "a1B2c3D4e5" * 4  # planted GitHub-token shape, 40 chars


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A tiny Python package with one planted secret and no license/CI."""
    repo = tmp_path / "tinypkg"
    (repo / "src" / "tinypkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "tinypkg"\nversion = "0.1.0"\n', encoding="utf-8",
    )
    (repo / "README.md").write_text(
        f"# tinypkg\n\nA tiny package.\n\nDebug token: {FAKE_TOKEN}\n", encoding="utf-8",
    )
    (repo / "src" / "tinypkg" / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (repo / "tests" / "test_basic.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (repo / ".env").write_text(f"GITHUB_TOKEN={FAKE_TOKEN}\n", encoding="utf-8")
    return repo


def test_snapshot_schema_and_signals(fixture_repo: Path) -> None:
    snapshot = collect_snapshot(fixture_repo)

    assert set(snapshot) == {"repo", "detected", "files", "signals", "risks", "limits"}
    assert snapshot["repo"]["name"] == "tinypkg"
    assert snapshot["detected"]["ecosystems"] == ["python"]
    assert snapshot["detected"]["release_targets"] == ["pypi"]
    assert "pyproject.toml" in snapshot["files"]["important_files"]
    assert "src/tinypkg/__init__.py" in snapshot["files"]["tree_summary"]

    signals = snapshot["signals"]
    assert signals["has_tests"] is True
    assert signals["has_license"] is False
    assert signals["has_ci"] is False
    assert signals["has_examples"] is False
    assert snapshot["limits"]["files_read"] >= 2


def test_snapshot_redacts_planted_secret(fixture_repo: Path) -> None:
    snapshot = collect_snapshot(fixture_repo)
    serialized = json.dumps(snapshot)

    assert FAKE_TOKEN not in serialized
    assert "[REDACTED]" in snapshot["files"]["important_files"]["README.md"]
    suspect_files = {s["file"] for s in snapshot["risks"]["secret_suspects"]}
    assert {"README.md", ".env"} <= suspect_files
    for suspect in snapshot["risks"]["secret_suspects"]:
        assert suspect["redacted"] == "ghp_***"


def test_snapshot_is_deterministic(fixture_repo: Path) -> None:
    assert collect_snapshot(fixture_repo) == collect_snapshot(fixture_repo)


def test_snapshot_never_executes_repo_code(fixture_repo: Path) -> None:
    (fixture_repo / "setup.py").write_text(
        "raise SystemExit('snapshot executed repo code')\n", encoding="utf-8",
    )
    snapshot = collect_snapshot(fixture_repo)  # must not raise
    assert "setup.py" in snapshot["files"]["important_files"]


def test_hard_caps_bound_fixture_score(fixture_repo: Path) -> None:
    snapshot = collect_snapshot(fixture_repo)
    scores = audit_repo.score_categories(snapshot)
    caps = audit_repo.hard_caps(snapshot)
    overall = audit_repo.overall_score(scores, caps)

    assert any("license" in reason.lower() for _, reason in caps)
    assert overall <= 6.0  # no-license cap
    assert all(1 <= score <= 10 for score, _ in scores.values())


def test_offline_audit_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "audit-output"

    exit_code = audit_repo.main([str(REPO_ROOT), "--output", str(out)])

    assert exit_code == 0
    for name in ("PROJECT_SCORECARD.md", "NEXT_STEPS.md", "graph.mmd", "trace.json"):
        assert (out / name).is_file(), f"missing {name}"

    scorecard = (out / "PROJECT_SCORECARD.md").read_text(encoding="utf-8")
    assert "# Project Audit: smythe" in scorecard
    assert "## Overall Rating" in scorecard
    assert "/10**" in scorecard
    assert "| Tests and CI |" in scorecard

    graph = (out / "graph.mmd").read_text(encoding="utf-8")
    assert "flowchart TD" in graph
    assert "synthesis" in graph

    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    node_ids = {span["node_id"] for span in trace}
    assert {"intake", "red-team", "synthesis"} <= node_ids

    next_steps = (out / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert next_steps.startswith("# Next Steps")
    assert "## Done When" in next_steps
