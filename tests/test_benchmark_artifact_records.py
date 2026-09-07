from pathlib import Path
import subprocess
from types import SimpleNamespace

import smythe
from benchmarks import artifact_records

from benchmarks.artifact_records import (
    environment_snapshot,
    image_mime_type,
    portable_path,
    resolve_record_path,
)


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
    replies = iter([SimpleNamespace(stdout="abc123\n"), SimpleNamespace(stdout=" M file.py\n")])
    monkeypatch.setattr(artifact_records.subprocess, "run", lambda *a, **kw: next(replies))
    assert artifact_records._source_control_snapshot() == {"revision": "abc123", "dirty": True}


def test_source_snapshot_remains_usable_without_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(artifact_records.subprocess, "run", missing_git)
    assert artifact_records._source_control_snapshot() == {"revision": None, "dirty": None}
