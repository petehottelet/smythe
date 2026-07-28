"""Tests for dependency-free durable-job artifact inspection."""

from __future__ import annotations

import hashlib
import io

import pytest

Image = pytest.importorskip("PIL.Image")

from smythe.jobs import artifact_io  # noqa: E402
from smythe.jobs.artifact_io import (  # noqa: E402
    ArtifactInspectionError,
    atomic_write_bytes,
    inspect_artifact,
)


def _png_bytes(size: tuple[int, int] = (320, 180)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_inspect_png_uses_content_not_declared_mime():
    data = _png_bytes()

    observed = inspect_artifact(data, "application/octet-stream")

    assert observed.mime_type == "image/png"
    assert (observed.width, observed.height) == (320, 180)
    assert observed.size_bytes == len(data)
    assert observed.sha256 == hashlib.sha256(data).hexdigest()


def test_inspect_image_rejects_header_only_and_declared_garbage():
    header_only = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    with pytest.raises(ArtifactInspectionError, match="fully decoded"):
        inspect_artifact(header_only, "application/octet-stream")
    with pytest.raises(ArtifactInspectionError, match="fully decoded"):
        inspect_artifact(b"not-an-image", "image/png")


def test_inspect_unknown_binary_retains_declared_mime():
    observed = inspect_artifact(b"not-an-image", "application/pdf")

    assert observed.mime_type == "application/pdf"
    assert observed.width is None
    assert observed.height is None


def test_atomic_write_replaces_complete_file(tmp_path):
    destination = tmp_path / "run" / "artifact.bin"
    atomic_write_bytes(destination, b"first")
    atomic_write_bytes(destination, b"replacement")

    assert destination.read_bytes() == b"replacement"
    assert list(destination.parent.glob("*.tmp")) == []


def test_artifact_bytes_are_bounded_before_decode_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_io, "MAX_ARTIFACT_BYTES", 3)

    with pytest.raises(ArtifactInspectionError, match="persistence limit"):
        inspect_artifact(b"four", "application/octet-stream")
    with pytest.raises(ValueError, match="persistence limit"):
        atomic_write_bytes(tmp_path / "too-large.bin", b"four")


def test_image_pixels_are_bounded_from_header_before_full_decode(monkeypatch):
    monkeypatch.setattr(artifact_io, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ArtifactInspectionError, match="pixel limit"):
        inspect_artifact(_png_bytes((20, 20)), "image/png")


def test_animated_image_frame_count_is_bounded(monkeypatch):
    frames = [Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(3)]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=10,
        loop=0,
    )
    monkeypatch.setattr(artifact_io, "MAX_IMAGE_FRAMES", 2)

    with pytest.raises(ArtifactInspectionError, match="frame"):
        inspect_artifact(buffer.getvalue(), "image/gif")
