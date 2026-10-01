"""The additive Astra reproduction rejects changed or unsafe evidence."""

import zipfile

import pytest

from benchmarks.verify_astra_publication import digest, extract_verified


def fixture(tmp_path, name="main/receipt.json"):
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        member = zipfile.ZipInfo("placeholder")
        member.filename = name  # Preserve a foreign separator on Windows too.
        stream.writestr(member, b"evidence")
    manifest = {"archive_sha256": digest(archive.read_bytes()),
                "members": {name: {"bytes": 8, "sha256": digest(b"evidence")}}}
    return archive, manifest


def test_valid_extraction_preserves_archive_and_refuses_reuse(tmp_path):
    archive, manifest = fixture(tmp_path)
    before = archive.read_bytes()
    assert extract_verified(archive, manifest, tmp_path / "copy") == 1
    assert archive.read_bytes() == before
    assert (tmp_path / "copy/main/receipt.json").read_bytes() == b"evidence"
    with pytest.raises(FileExistsError):
        extract_verified(archive, manifest, tmp_path / "copy")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "main\\escape"])
def test_unsafe_members_never_escape_extraction(tmp_path, name):
    archive, manifest = fixture(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe|inventory mismatch"):
        extract_verified(archive, manifest, tmp_path / "copy")


def test_changed_archive_or_member_fails_verification(tmp_path):
    archive, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="Archive hash"):
        extract_verified(archive, manifest | {"archive_sha256": "bad"}, tmp_path / "a")
    manifest["members"]["main/receipt.json"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="Member hash"):
        extract_verified(archive, manifest, tmp_path / "b")


def test_committed_main_analysis_reproduces_exactly(tmp_path):
    import json
    from pathlib import Path

    from benchmarks.astra_main_evidence import review_main

    public = Path(__file__).resolve().parents[1] / "benchmarks/results/astra_20260913_main"
    manifest = json.loads((public / "archive-manifest.json").read_text(encoding="utf-8"))
    extracted = tmp_path / "extracted"
    extract_verified(public / "evidence.zip", manifest, extracted)
    computed, _ = review_main(
        main_directory=extracted / "main", continuation_directory=extracted / "main-continuation",
        judge_directory=extracted / "judging", bindings_path=extracted / "calibration/main-judgment-bindings.json",
        source_archive=extracted / "sources/typed-main-source.zip",
        continuation_source_archive=extracted / "sources/continuation-source.zip",
    )
    assert computed == json.loads((public / "analysis.json").read_text(encoding="utf-8"))
