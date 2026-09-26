"""Tracked files never carry provider account identifiers or keys."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchmarks.artifact_records import ACCOUNT_IDENTIFIER_PATTERNS

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("requires a git checkout")
    return [name for name in listed.decode("utf-8").split("\0") if name]


def _identifier_findings(root: Path, names: list[str]) -> list[str]:
    """Locate identifiers without repeating them, so the report cannot leak one."""
    findings = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        for pattern, replacement in ACCOUNT_IDENTIFIER_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{name}:{line}: {replacement}")
    return findings


def test_identifier_scan_reports_locations_without_repeating_identifiers(tmp_path):
    organization = "org-" + "A1b2" * 6
    (tmp_path / "record.json").write_text(
        f'{{"status": "failed",\n "error": "in organization {organization}"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "tile.png").write_bytes(b"\x89PNG\0" + organization.encode())

    findings = _identifier_findings(tmp_path, ["record.json", "tile.png", "deleted.txt"])

    assert findings == ["record.json:2: org-[redacted]"]


def test_tracked_files_contain_no_account_identifiers():
    findings = _identifier_findings(ROOT, _tracked_files())

    assert not findings, "Redact these before committing:\n" + "\n".join(findings)
