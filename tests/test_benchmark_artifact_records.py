from pathlib import Path

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

    assert snapshot["packages"]["smythe"]
