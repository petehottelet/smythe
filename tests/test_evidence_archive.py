"""The retired-evidence archive rebuilds from git and fails closed on any drift."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import zipfile

import pytest

from tools import evidence_archive

ROOT = Path(__file__).resolve().parents[1]


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {"results/old_a.json": b'{"a": 1}\n', "results/old_b.json": b'{"b": 2}\n',
             "partitions/old/tile.png": bytes(range(256)), "docs/kept.md": b"# kept\n"}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    def run(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                       cwd=root, check=True, capture_output=True)

    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "records")
    run("tag", "v1")
    return root


def _spec(**changes) -> dict:
    spec = {"name": "evidence-test", "ref": "v1", "release_asset": "https://example.invalid/evidence-test.zip",
            "groups": [{"id": "records", "title": "Old records", "status": "superseded",
                        "description": "Replaced by a re-run.", "patterns": ["results/old_*.json"]},
                       {"id": "tiles", "title": "Old tiles", "status": "diagnostic",
                        "description": "Diagnostic tiles.", "patterns": ["partitions/old/*"]}]}
    spec.update(changes)
    return spec


def test_archive_is_deterministic_and_verifies_against_git(tmp_path):
    root = _repo(tmp_path)
    pointer = evidence_archive.build(_spec(), tmp_path / "one", root=root)
    again = evidence_archive.build(_spec(), tmp_path / "two", root=root)
    assert pointer == again
    assert (pointer["files"], [g["files"] for g in pointer["groups"]]) == (3, [2, 1])
    archive = tmp_path / "one/evidence-test.zip"
    assert evidence_archive.verify(archive, pointer, against_git=True, root=root)["files"] == 3
    with zipfile.ZipFile(archive) as bundle:
        assert "evidence-test/files/docs/kept.md" not in bundle.namelist()
        assert bundle.read("evidence-test/files/partitions/old/tile.png") == bytes(range(256))


def test_tampered_members_and_mismatched_pointers_are_rejected(tmp_path):
    root = _repo(tmp_path)
    pointer = evidence_archive.build(_spec(), tmp_path / "out", root=root)
    archive = tmp_path / "out/evidence-test.zip"
    with pytest.raises(ValueError, match="SHA-256"):
        evidence_archive.verify(archive, {**pointer, "sha256": "0" * 64}, root=root)

    with zipfile.ZipFile(archive) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    members["evidence-test/files/results/old_a.json"] = b'{"a": 2}\n'
    forged = tmp_path / "forged.zip"
    evidence_archive._write_zip(forged, members)
    data = forged.read_bytes()
    relabeled = {**pointer, "bytes": len(data), "sha256": evidence_archive._sha(data)}
    with pytest.raises(ValueError, match="checksum mismatch: results/old_a.json"):
        evidence_archive.verify(forged, relabeled, root=root)

    members.pop("evidence-test/files/results/old_a.json")
    extra = tmp_path / "missing.zip"
    evidence_archive._write_zip(extra, members)
    data = extra.read_bytes()
    with pytest.raises(ValueError, match="inventory"):
        evidence_archive.verify(extra, {**pointer, "bytes": len(data), "sha256": evidence_archive._sha(data)},
                                root=root)


def test_unmatched_overlapping_and_unlabeled_groups_are_rejected(tmp_path):
    root = _repo(tmp_path)
    missing = _spec(groups=[{**_spec()["groups"][0], "patterns": ["results/none_*.json"]}])
    with pytest.raises(ValueError, match="matches no file"):
        evidence_archive.build(missing, tmp_path / "a", root=root)
    overlap = _spec(groups=[_spec()["groups"][0], {**_spec()["groups"][1], "patterns": ["results/*"]}])
    with pytest.raises(ValueError, match="more than one group"):
        evidence_archive.build(overlap, tmp_path / "b", root=root)
    unlabeled = _spec(groups=[{**_spec()["groups"][0], "status": "current"}])
    with pytest.raises(ValueError, match="Unknown evidence status"):
        evidence_archive.build(unlabeled, tmp_path / "c", root=root)


def test_committed_pointer_pins_a_labeled_release_asset():
    pointer = json.loads(evidence_archive.POINTER.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", pointer["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", pointer["manifest_sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", pointer["source"]["commit"])
    assert pointer["release_asset"] == ("https://github.com/petehottelet/smythe/releases/download/"
                                        f"v0.9.0/{pointer['archive']}")
    assert {g["status"] for g in pointer["groups"]} <= evidence_archive.STATUSES
    assert sum(g["files"] for g in pointer["groups"]) == pointer["files"]
    assert sum(g["bytes"] for g in pointer["groups"]) == pointer["total_bytes"]
    guide = (ROOT / "benchmarks/archive/README.md").read_text(encoding="utf-8")
    assert pointer["sha256"] in guide and pointer["release_asset"] in guide
