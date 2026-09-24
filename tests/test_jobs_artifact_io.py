"""Tests for dependency-free durable-job artifact inspection."""

from __future__ import annotations

import hashlib
import io
import struct
import threading
import warnings
from contextlib import contextmanager

import pytest

Image = pytest.importorskip("PIL.Image")

from smythe.jobs import artifact_io  # noqa: E402
from smythe.jobs.artifact_io import (  # noqa: E402
    ArtifactInspectionError,
    atomic_write_bytes,
    inspect_artifact,
)
from test_images import (  # noqa: E402
    READ_PAST_TERMINATOR_CASES,
    gif_hiding_a_frame,
    refuse_large_allocations,
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


def _gif(screen: tuple[int, int], frames: list[tuple[int, int, int, int, int]]) -> bytes:
    """Build a GIF from ``(left, top, width, height, disposal)`` frame descriptors.

    Each frame carries one LZW-coded pixel, so only 1x1 frames decode fully;
    larger frames exist only to be refused before Pillow reads them.
    """
    data = bytearray(b"GIF89a" + struct.pack("<HHBBB", *screen, 0x80, 0, 0))
    data += b"\x00\x00\x00\xff\xff\xff"  # two-color global color table
    for left, top, width, height, disposal in frames:
        data += b"\x21\xf9\x04" + bytes([disposal << 2, 0, 0, 0, 0])  # graphic control
        data += b"\x2c" + struct.pack("<HHHHB", left, top, width, height, 0)
        data += bytes([2, 2, 0x44, 0x01, 0])  # code size 2: clear, index 0, end
    return bytes(data + b"\x3b")


@pytest.mark.parametrize(
    ("frames", "interlude"),
    [
        # A 1x1 frame far outside the 1x1 screen grows the canvas to 9001x9001.
        ([(0, 0, 1, 1, 0), (9000, 9000, 1, 1, 0)], b""),
        # A 9001x9001 frame also has its disposal area allocated inside seek().
        ([(0, 0, 1, 1, 0), (0, 0, 9001, 9001, 2)], b""),
        # A first frame that large is allocated while Pillow opens the file.
        ([(0, 0, 9001, 9001, 2)], b""),
        # A comment block and a stray byte, which Pillow skips, hide nothing.
        ([(0, 0, 1, 1, 0), (9000, 9000, 1, 1, 0)], b"\x21\xfe\x03,,,\x02,,\x00\x07"),
    ],
    ids=["offset-frame", "disposal-frame", "first-frame", "after-comment"],
)
def test_gif_frames_cannot_grow_the_canvas_past_the_pixel_limit(monkeypatch, frames, interlude):
    """Regression: only the first frame's size was checked. A 74-byte GIF whose
    second frame grew the canvas to 9001x9001 was accepted with its first-frame
    size after a 718 MB decode."""
    from PIL import GifImagePlugin

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)  # A host may disable Pillow's limit.
    calls = _spy_on_plugin_open(monkeypatch, GifImagePlugin.GifImageFile)
    data = _gif((1, 1), frames)
    last_frame = data.rindex(b"\x21\xf9\x04")
    data = data[:last_frame] + interlude + data[last_frame:]

    with pytest.raises(ArtifactInspectionError, match="pixel limit"):
        inspect_artifact(data, "image/gif")
    assert calls == []


def test_gif_canvas_growth_counts_toward_the_aggregate_pixel_limit(monkeypatch):
    monkeypatch.setattr(artifact_io, "MAX_TOTAL_DECODED_PIXELS", 25_000)
    # The second frame grows the 1x1 canvas to 100x100: four frames decode 30,001 pixels.
    data = _gif((1, 1), [(0, 0, 1, 1, 0)] + [(99, 99, 1, 1, 0)] * 3)

    with pytest.raises(ArtifactInspectionError, match="aggregate pixel limit"):
        inspect_artifact(data, "image/gif")


def test_gif_reports_the_canvas_its_frames_grow_to():
    observed = inspect_artifact(_gif((1, 1), [(0, 0, 1, 1, 0), (99, 49, 1, 1, 0)]), "image/gif")

    assert (observed.mime_type, observed.width, observed.height) == ("image/gif", 100, 50)


@pytest.mark.parametrize(("extension", "second_frame"), READ_PAST_TERMINATOR_CASES)
def test_gif_blocks_pillow_reads_past_a_terminator_are_refused_before_decoding(
    monkeypatch, extension, second_frame
):
    """Regression: the pre-scan stopped at a terminator Pillow reads past, so
    a frame hidden behind it was never bounded. A 258-byte file reached a
    13000x13000 disposal allocation, or 60000x60000 with Pillow's own limit
    disabled."""
    from PIL import GifImagePlugin

    refused = refuse_large_allocations(monkeypatch)
    calls = _spy_on_plugin_open(monkeypatch, GifImagePlugin.GifImageFile)
    data = gif_hiding_a_frame(extension, second_frame=second_frame)

    for pillow_limit in (Image.MAX_IMAGE_PIXELS, None):
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
        with pytest.raises(
            ArtifactInspectionError, match="not a recognizable PNG, JPEG, GIF, or WebP"
        ):
            inspect_artifact(data, "image/gif")
    assert calls == []
    assert refused == []


def test_later_mpo_frames_are_bounded_before_they_are_decoded(monkeypatch):
    """Regression: each MPO frame declares its own size and only the first was
    checked, so a 1.4 KB file could allocate gigabytes before its format was
    refused."""
    from PIL import ImageFile

    buffer = io.BytesIO()
    frames = [Image.new("RGB", (8, 8), (index * 40, 0, 0)) for index in range(2)]
    frames[0].save(buffer, format="MPO", save_all=True, append_images=frames[1:])
    data = bytearray(buffer.getvalue())
    start_of_frame = b"\xff\xc0\x00\x11\x08"  # baseline SOF0 for 8-bit RGB
    second = data.index(start_of_frame, data.index(start_of_frame) + 1)
    data[second + 5 : second + 9] = struct.pack(">HH", 9000, 9000)  # height, width
    prepared = []
    original = ImageFile.ImageFile.load_prepare

    def spy(self):
        prepared.append(self.size)
        return original(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load_prepare", spy)

    with pytest.raises(ArtifactInspectionError, match="pixel limit"):
        inspect_artifact(bytes(data), "image/jpeg")
    assert (9000, 9000) not in prepared


def _webp_declaring(chunk: bytes, width: int, height: int) -> bytes:
    """Build the smallest WebP file whose first chunk declares this canvas."""
    if chunk == b"VP8 ":  # key-frame tag, start code, 14-bit sizes, filler
        payload = b"\x30\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height) + bytes(10)
    elif chunk == b"VP8L":  # signature, then 14-bit sizes minus one
        payload = b"\x2f" + ((width - 1) | (height - 1) << 14).to_bytes(4, "little")
    else:  # VP8X: flags, then 24-bit sizes minus one
        payload = bytes(4) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    body = b"WEBP" + chunk + struct.pack("<I", len(payload)) + payload + bytes(len(payload) % 2)
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("chunk", [b"VP8 ", b"VP8L", b"VP8X"])
def test_webp_canvas_is_bounded_before_libwebp_allocates_it(monkeypatch, chunk):
    """Regression: a 40-byte lossy WebP declaring 16383x16383 reached libwebp,
    which allocated about 2 GB before any size check, and a 26-byte lossless
    one slipped under the header parser's 30-byte minimum the same way."""
    from PIL import WebPImagePlugin

    calls = _spy_on_plugin_open(monkeypatch, WebPImagePlugin.WebPImageFile)

    with pytest.raises(ArtifactInspectionError, match="pixel limit"):
        inspect_artifact(_webp_declaring(chunk, 8193, 8193), "image/webp")
    assert calls == []


def test_concurrent_inspections_leave_the_process_warning_filters_alone(monkeypatch):
    """Regression: inspection escalated DecompressionBombWarning inside
    warnings.catch_warnings(), which swaps the process-global filter list.
    Overlapping worker-thread inspections left the "error" filter installed."""
    both_open = threading.Barrier(2, timeout=10)
    first_done = threading.Event()
    seen = threading.local()
    original_open = artifact_io.open_image

    @contextmanager
    def overlapping_open(source, **kwargs):
        # Hold both inspections inside their decode at once, then let the
        # first finish before the second, as two worker threads can.
        if not getattr(seen, "opened", False):
            seen.opened = True
            both_open.wait()
            if threading.current_thread().name == "second":
                assert first_done.wait(10)
        with original_open(source, **kwargs) as image:
            yield image

    monkeypatch.setattr(artifact_io, "open_image", overlapping_open)
    before = list(warnings.filters)
    errors = []

    def inspect(name):
        try:
            inspect_artifact(_png_bytes((4, 4)), "image/png")
        except BaseException as exc:  # Reported by the assertion below.
            errors.append(exc)
        finally:
            if name == "first":
                first_done.set()

    threads = [threading.Thread(target=inspect, args=(name,), name=name)
               for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == []
    assert warnings.filters == before


@pytest.mark.parametrize("gif_hex", [
    # The last image descriptor is cut short: Pillow raises struct.error.
    pytest.param(
        "4749463839610800080080000000000000000021ff0b4e45545343415045322e3003010000"
        "0021fe0268690021f90400050000002c000000000800080000080f0001081c48b0a0c18308"
        "132a4c1810002c3b",
        id="struct-error",
    ),
    # A truncated extension block: Pillow raises IndexError while seeking.
    pytest.param(
        "474946383961080008008000f900000000000021ff0b4e45545343415045322e3003010000"
        "0021fe0268690088f90400050000002c000000000800080000080f0001081c48b0a0c18308"
        "132a4c18100021",
        id="index-error",
    ),
])
def test_malformed_gif_blocks_fail_as_inspection_errors(gif_hex):
    """Pillow's GIF plugin raised these types past the decode step."""
    with pytest.raises(ArtifactInspectionError, match="could not be fully decoded"):
        inspect_artifact(bytes.fromhex(gif_hex), "image/gif")
