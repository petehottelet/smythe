"""Skill archives are reproducible, self-contained and runtime-bound."""

import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from tools.skill_archive import FILES, build, checked_members, verify


@pytest.fixture
def skill_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in FILES:
        path = root / "skills/repo-doctor" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("smythe==0.7.0\n" if name == "runtime.txt" else f"fixture: {name}\n")
    (root / "LICENSE").write_text("MIT fixture\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "--", "skills", "LICENSE"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture"], cwd=root, check=True)
    return root


def test_archive_uses_committed_bytes_and_is_reproducible(skill_repo, tmp_path):
    (skill_repo / "skills/repo-doctor/private.txt").write_text("never ship")
    first = build(tmp_path / "one", root=skill_repo)
    (skill_repo / "skills/repo-doctor/README.md").write_text("uncommitted change")
    second = build(tmp_path / "two", root=skill_repo)
    assert first.read_bytes() == second.read_bytes()
    manifest, members = checked_members(first)
    assert manifest["runtime"] == "smythe==0.7.0"
    assert len(manifest["source_commit"]) == 40
    assert members["README.md"] == b"fixture: README.md\n"
    assert "LICENSE" in members and "private.txt" not in members


def test_skill_archive_will_not_overwrite_an_existing_release_candidate(skill_repo, tmp_path):
    first = build(tmp_path / "out", root=skill_repo)
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        build(tmp_path / "out", root=skill_repo)
    assert first.read_bytes() == before


def test_archive_verifier_rejects_modified_content(skill_repo, tmp_path):
    archive = build(tmp_path / "out", root=skill_repo)
    with zipfile.ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    members["repo-doctor/README.md"] = b"tampered"
    with zipfile.ZipFile(archive, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        checked_members(archive)


def test_runtime_pin_is_verified_before_running_skill_code(skill_repo, tmp_path, monkeypatch):
    archive = build(tmp_path / "out", root=skill_repo)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "0.8.0\n")
    with pytest.raises(ValueError, match="exact pin"):
        verify(archive, Path("unused-python"))


def test_external_receipt_hashes_the_actual_archive(skill_repo, tmp_path):
    import hashlib

    archive = build(tmp_path / "out", root=skill_repo)
    receipt = json.loads(archive.with_suffix(".json").read_text())
    assert receipt["bytes"] == archive.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
