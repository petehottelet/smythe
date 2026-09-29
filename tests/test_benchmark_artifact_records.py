import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import smythe
from benchmarks import artifact_records

from benchmarks.artifact_records import (
    evidence_directory,
    environment_snapshot,
    image_mime_type,
    portable_path,
    redact_account_identifiers,
    redact_local_paths,
    resolve_record_path,
    scrub_record,
)
import os


def _identifier(prefix: str, length: int) -> str:
    """Build identifier-shaped text at runtime so no literal is committed."""
    return prefix + ("A1b2" * length)[:length]


def test_image_mime_type_uses_the_actual_extension():
    assert image_mime_type("banner.jpg") == "image/jpeg"
    assert image_mime_type("logo.PNG") == "image/png"
    assert image_mime_type("reference.webp") == "image/webp"


def test_image_mime_type_has_safe_fallback():
    assert image_mime_type("extensionless") == "application/octet-stream"


def test_portable_path_is_repo_relative_and_uses_forward_slashes(tmp_path):
    root = tmp_path / "checkout"
    artifact = root / "smythe_artifacts" / "suite" / "hero.png"

    assert portable_path(artifact, root=root) == "smythe_artifacts/suite/hero.png"


def test_portable_path_preserves_external_location(tmp_path):
    root = tmp_path / "checkout"
    external = tmp_path / "brand-assets" / "logo.png"

    assert portable_path(external, root=root) == Path(external).resolve().as_posix()


def test_resolve_record_path_uses_checkout_root_for_relative_records(tmp_path):
    assert resolve_record_path("artifacts/hero.png", root=tmp_path) == (
        tmp_path / "artifacts" / "hero.png"
    )


def test_environment_snapshot_records_missing_packages_without_failing():
    snapshot = environment_snapshot("definitely-not-a-real-smythe-package")

    assert snapshot["python"]
    assert snapshot["platform"]
    assert snapshot["packages"]["definitely-not-a-real-smythe-package"] is None


def test_environment_snapshot_records_checkout_smythe_version():
    snapshot = environment_snapshot("smythe")

    assert snapshot["packages"]["smythe"] == smythe.__version__
    assert snapshot["smythe_source"]["version"] == smythe.__version__
    assert snapshot["smythe_source"]["module"].endswith("smythe/__init__.py")


def test_environment_snapshot_does_not_confuse_stale_install_with_source(monkeypatch):
    monkeypatch.setattr(artifact_records.metadata, "version", lambda _: "0.0.1-stale")
    snapshot = environment_snapshot("smythe")
    assert snapshot["packages"]["smythe"] == smythe.__version__
    assert snapshot["installed_packages"]["smythe"] == "0.0.1-stale"


def test_source_snapshot_records_revision_and_dirty_status(monkeypatch):
    replies = iter([SimpleNamespace(stdout="abc123\n"),
                    SimpleNamespace(stdout=" M file.py\n?? out.json\n")])
    monkeypatch.setattr(artifact_records.subprocess, "run", lambda *a, **kw: next(replies))
    assert artifact_records._source_control_snapshot() == {
        "revision": "abc123", "dirty": True, "untracked_files": 1}


def test_untracked_outputs_do_not_mark_the_sources_dirty(monkeypatch):
    replies = iter([SimpleNamespace(stdout="abc123\n"),
                    SimpleNamespace(stdout="?? benchmarks/results/a.json\n?? smoke.json\n")])
    monkeypatch.setattr(artifact_records.subprocess, "run", lambda *a, **kw: next(replies))
    assert artifact_records._source_control_snapshot() == {
        "revision": "abc123", "dirty": False, "untracked_files": 2}


def test_source_snapshot_remains_usable_without_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(artifact_records.subprocess, "run", missing_git)
    assert artifact_records._source_control_snapshot() == {
        "revision": None, "dirty": None, "untracked_files": None}


@pytest.mark.parametrize(
    ("identifier", "replacement"),
    [
        (_identifier("org-", 24), "org-[redacted]"),
        (_identifier("proj_", 24), "proj_[redacted]"),
        (_identifier("sk-proj-", 48), "sk-[redacted]"),
        (_identifier("sk-" + "ant-api03-", 48), "sk-[redacted]"),
        ("sk-proj-" + "*" * 40 + "Ab12", "sk-[redacted]"),
        (_identifier("AIza", 35), "AIza[redacted]"),
        (_identifier("AQ.", 40), "AQ.[redacted]"),
        ("gen-lang-client-" + "0123456789", "gen-lang-client-[redacted]"),
        ("project_number:" + "123456789012", "project_number:[redacted]"),
    ],
)
def test_redaction_replaces_each_account_identifier_format(identifier, replacement):
    assert redact_account_identifiers(f"consumer '{identifier}' was refused") == (
        f"consumer '{replacement}' was refused"
    )


def test_redaction_reaches_every_string_in_a_nested_record():
    organization = _identifier("org-", 24)
    record = {
        "errors": [{"error": f"429 in organization {organization} on images per min"}],
        "detail": (f"organization {organization}", 3, None),
        organization: 1.5,
    }

    redacted = redact_account_identifiers(record)

    assert organization not in json.dumps(redacted)
    assert redacted == {
        "errors": [{"error": "429 in organization org-[redacted] on images per min"}],
        "detail": ["organization org-[redacted]", 3, None],
        "org-[redacted]": 1.5,
    }


def test_redaction_leaves_ordinary_text_and_placeholder_keys_alone():
    text = "org-chart " + "sk-" + "offline-test project_number:42 gen-lang-client-7"

    assert redact_account_identifiers(text) == text


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home" / "Example Person"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_local_path_redaction_matches_separator_and_escape_spellings(fake_home):
    native = str(fake_home / "project" / "out.json")

    assert redact_local_paths(native) == os.path.join("~", "project", "out.json")
    assert redact_local_paths(fake_home.as_posix() + "/project") == "~/project"
    assert redact_local_paths(json.dumps(native)) == json.dumps(os.path.join("~", "project", "out.json"))
    assert redact_local_paths(str(fake_home) + "Extra") == str(fake_home) + "Extra"
    if os.name == "nt":
        assert redact_local_paths(str(fake_home).upper() + "\\x") == "~\\x"


def test_scrub_record_redacts_identifiers_and_the_home_folder(fake_home):
    organization = _identifier("org-", 24)
    record = {"error": f"429 in organization {organization}",
              "paths": [str(fake_home / "out" / "tile.png")], "count": 3}

    assert scrub_record(record) == {"error": "429 in organization org-[redacted]",
                                    "paths": [os.path.join("~", "out", "tile.png")], "count": 3}


def test_new_evidence_directories_must_not_expose_private_locations(fake_home, tmp_path, monkeypatch):
    monkeypatch.delenv(artifact_records.PRIVATE_EVIDENCE_OVERRIDE, raising=False)

    with pytest.raises(ValueError, match="outside the home folder"):
        evidence_directory(fake_home / "campaign")
    with pytest.raises(ValueError, match="private planning folder"):
        evidence_directory(tmp_path / "work" / "00_project_files" / "campaign")
    assert evidence_directory(tmp_path / "work" / "campaign") == (tmp_path / "work" / "campaign").resolve()
    monkeypatch.setenv(artifact_records.PRIVATE_EVIDENCE_OVERRIDE, "1")
    assert evidence_directory(fake_home / "campaign") == (fake_home / "campaign").resolve()
