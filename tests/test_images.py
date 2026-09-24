"""Tests for the restricted Pillow opener shared by every image reader."""

from __future__ import annotations

import io
import struct

import pytest

Image = pytest.importorskip("PIL.Image")

from smythe._images import open_image, webp_dimensions  # noqa: E402


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


class _Unseekable(io.RawIOBase):
    """A readable stream without seek(), like a pipe."""

    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer):
        return self._data.readinto(buffer)


@pytest.mark.parametrize("chunk", [b"VP8 ", b"VP8L", b"VP8X"])
def test_oversized_webp_canvas_is_refused_before_libwebp_runs(tmp_path, monkeypatch, chunk):
    """Regression: Pillow's WebP plugin let libwebp allocate two canvas-sized
    buffers before open_image checked the size; 16383x16383 cost about 2 GB."""
    from PIL import WebPImagePlugin

    calls = []
    original = WebPImagePlugin.WebPImageFile._open

    def spy(self):
        calls.append(chunk)
        return original(self)

    monkeypatch.setattr(WebPImagePlugin.WebPImageFile, "_open", spy)
    data = _webp_declaring(chunk, 101, 101)
    path = tmp_path / "canvas.webp"
    path.write_bytes(data)

    for source in (data, path, io.BytesIO(data), _Unseekable(data)):
        with pytest.raises(Image.DecompressionBombError, match="101x101"):
            with open_image(source, max_pixels=10_000):
                pass
    assert calls == []


@pytest.mark.parametrize("lossless", [False, True], ids=["lossy", "lossless"])
def test_webp_within_the_limit_still_opens_from_every_source(tmp_path, lossless):
    from PIL import features

    if not features.check("webp"):
        pytest.skip("Pillow was built without WebP support")
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(buffer, format="WEBP", lossless=lossless)
    data = buffer.getvalue()
    path = tmp_path / "small.webp"
    path.write_bytes(data)

    assert data[12:16] == (b"VP8L" if lossless else b"VP8 ")
    assert webp_dimensions(data) == (6, 4)
    for source in (data, path, io.BytesIO(data), _Unseekable(data)):
        with open_image(source, max_pixels=6 * 4) as image:
            assert (image.format, image.size) == ("WEBP", (6, 4))
