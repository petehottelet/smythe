from __future__ import annotations

from pathlib import Path

import pytest

from smythe.jobs import JobManifestV1, ManifestValidationError
from smythe.jobs.loading import load_manifest
from smythe.jobs.models import MAX_MANIFEST_BYTES


def test_manifest_document_size_is_rejected_before_read(tmp_path, monkeypatch):
    source = tmp_path / "oversized.json"
    with source.open("wb") as stream:
        stream.truncate(MAX_MANIFEST_BYTES + 1)

    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: pytest.fail("oversized manifest was opened"),
    )
    with pytest.raises(ManifestValidationError, match="manifest exceeds"):
        load_manifest(source)


def test_manifest_document_must_be_utf8(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_bytes(b"\xff\xfe")

    with pytest.raises(ManifestValidationError, match="valid UTF-8"):
        load_manifest(source)


def test_manifest_from_json_enforces_the_same_document_cap():
    with pytest.raises(ManifestValidationError, match="manifest exceeds"):
        JobManifestV1.from_json(" " * (MAX_MANIFEST_BYTES + 1))


def test_manifest_from_json_rejects_duplicate_keys_directly():
    with pytest.raises(ManifestValidationError, match="duplicate key 'version'"):
        JobManifestV1.from_json('{"version": 1, "version": 1}')


@pytest.mark.parametrize(
    ("suffix", "document"),
    [
        (".json", '{"version": 1, "version": 1}'),
        (".yaml", "version: 1\nversion: 1\n"),
    ],
)
def test_manifest_loading_rejects_duplicate_keys_at_the_parser_boundary(
    tmp_path,
    suffix,
    document,
):
    source = tmp_path / f"manifest{suffix}"
    source.write_text(document, encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="duplicate key"):
        load_manifest(source)


def test_manifest_yaml_rejects_alias_expansion(tmp_path):
    source = tmp_path / "manifest.yaml"
    source.write_text(
        "profile: &profile\n  provider: offline\ncopy: *profile\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="aliases are forbidden"):
        load_manifest(source)
