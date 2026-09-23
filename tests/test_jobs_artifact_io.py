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


def _spy_on_plugin_open(monkeypatch, plugin_class):
    calls = []
    original = plugin_class._open

    def spy(self):
        calls.append(plugin_class.__name__)
        return original(self)

    monkeypatch.setattr(plugin_class, "_open", spy)
    return calls


def test_declared_png_bytes_never_reach_other_pillow_plugins(monkeypatch):
    """Regression: EPS bytes labelled image/png reached EpsImagePlugin
    (Ghostscript), and TIFF was fully decoded before being rejected."""
    from PIL import EpsImagePlugin, TiffImagePlugin

    calls = _spy_on_plugin_open(monkeypatch, EpsImagePlugin.EpsImageFile)
    calls += _spy_on_plugin_open(monkeypatch, TiffImagePlugin.TiffImageFile)
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\nshowpage\n%%EOF\n"
    tiff = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(tiff, format="TIFF")

    for data in (eps, tiff.getvalue()):
        with pytest.raises(
            ArtifactInspectionError, match="not a recognizable PNG, JPEG, GIF, or WebP"
        ):
            inspect_artifact(data, "image/png")
    assert calls == []


def test_job_run_records_a_disallowed_provider_image_as_an_error(tmp_path, monkeypatch):
    """End to end: the run records a clear error instead of decoding EPS."""
    import asyncio

    from PIL import EpsImagePlugin

    from smythe.jobs import JobManifestV1, make_approval, preflight_job
    from smythe.jobs.providers import ProviderPool
    from smythe.jobs.runner import JobRunner
    from smythe.jobs.store import SQLiteRunStore
    from smythe.provider import Artifact, CompletionResult, Provider

    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\nshowpage\n%%EOF\n"

    class EpsProvider(Provider):
        async def complete(self, system, prompt, model):
            return CompletionResult(text="done", artifacts=[Artifact(eps, "image/png")])

    class Pool(ProviderPool):
        def get(self, operation):
            return EpsProvider()

        @staticmethod
        def validate_operation(operation):
            return None

        @staticmethod
        def preflight(operation, **_kwargs):
            return None

    calls = _spy_on_plugin_open(monkeypatch, EpsImagePlugin.EpsImageFile)
    manifest = JobManifestV1.from_dict({
        "version": 1,
        "name": "eps-as-png",
        "profiles": [{"name": "default", "provider": "offline", "model": "offline-image",
                      "max_cost_per_call_usd": "0"}],
        "operations": [{"key": "tile", "prompt": "One tile", "profile": "default",
                        "artifact": {"mime_type": "image/png", "width": 1, "height": 1}}],
        "execution": {"max_concurrency": 1, "max_attempts": 1, "max_budget_usd": "0",
                      "output_directory": "outputs"},
    })
    plan = preflight_job(manifest, manifest_root=tmp_path)
    with SQLiteRunStore(tmp_path / "jobs.db") as store:
        result = asyncio.run(
            JobRunner(store, provider_pool=Pool()).start(
                plan, make_approval(plan), manifest_root=tmp_path
            )
        )

    assert "succeeded" not in result["counts"]
    assert result["artifacts"] == []
    [attempt] = result["attempts"]
    assert "not a recognizable PNG, JPEG, GIF, or WebP image" in attempt["error"]
    assert calls == []


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"), ("WEBP", "image/webp")],
)
def test_supported_raster_formats_still_decode(image_format, mime_type):
    from PIL import features

    if image_format == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow was built without WebP support")
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(buffer, format=image_format)

    observed = inspect_artifact(buffer.getvalue(), "image/png")

    assert (observed.mime_type, observed.width, observed.height) == (mime_type, 6, 4)


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
