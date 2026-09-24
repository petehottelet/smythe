"""Tests for the restricted Pillow opener shared by every image reader."""

from __future__ import annotations

import io
import os
import struct

import pytest

Image = pytest.importorskip("PIL.Image")

from smythe._images import gif_dimensions, open_image, webp_dimensions  # noqa: E402


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


def _spy_on_plugin_open(monkeypatch, plugin_class):
    calls = []
    original = plugin_class._open

    def spy(self):
        calls.append(plugin_class.__name__)
        return original(self)

    monkeypatch.setattr(plugin_class, "_open", spy)
    return calls


def refuse_large_allocations(monkeypatch):
    """Make Pillow refuse, and record, any image buffer over a million pixels."""
    refused = []
    for name in ("fill", "new"):

        def guard(mode, size, *rest, _real=getattr(Image.core, name), _name=name):
            if size[0] * size[1] > 1_000_000:
                refused.append(f"{_name}({mode!r}, {tuple(size)})")
                raise MemoryError("the test refused a large allocation")
            return _real(mode, size, *rest)

        monkeypatch.setattr(Image.core, name, guard)
    return refused


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


def gif_hiding_a_frame(extension: bytes, *, second_frame: bool = False) -> bytes:
    """A 1x1 GIF in which only Pillow finds a 13000x13000 frame.

    The GIF format reads the bytes after *extension* as a 200-byte comment.
    Pillow reads on past the extension's terminator: it takes the comment's
    introducer as a 33-byte sub-block, stops at the zero after it, and then
    finds an image descriptor. The graphic control block before the extension
    makes Pillow allocate that frame's disposal area.
    """
    data = bytearray(b"GIF89a" + struct.pack("<HH", 1, 1))
    if second_frame:  # a two-color table and an ordinary 1x1 first frame
        data += b"\x80\x00\x00" + b"\xff\x00\x00\x00\x00\xff"
        data += b"\x21\xf9\x04\x00\x00\x00\x00\x00"
        data += b"\x2c" + struct.pack("<HHHHB", 0, 0, 1, 1, 0) + b"\x02\x02\x44\x01\x00"
    else:
        data += b"\x00\x00\x00"
    data += b"\x21\xf9\x04\x08\x00\x00\x00\x00"  # graphic control: restore to background
    data += extension
    comment = bytearray(b"\x21\xfe\xc8" + bytes(200) + b"\x00")
    comment[35:46] = b"\x2c" + struct.pack("<HHHHB", 0, 0, 13000, 13000, 0) + b"\x02"
    return bytes(data + comment + b"\x3b")


# Extensions after which Pillow reads on past a data sub-block terminator: an
# empty first sub-block (any label but a comment's), and a NETSCAPE2.0 block
# before the first frame without its looping sub-block.
READ_PAST_TERMINATOR_CASES = [
    pytest.param(b"\x21\x01\x00", False, id="plain-text"),
    pytest.param(b"\x21\x01\x00", True, id="plain-text-second-frame"),
    pytest.param(b"\x21\xf9\x00", False, id="graphic-control"),
    pytest.param(b"\x21\xff\x00", False, id="application"),
    pytest.param(b"\x21\xff\x0bNETSCAPE2.0\x00", False, id="netscape-loop"),
]


@pytest.mark.parametrize("frames", [
    # A 1x1 frame far outside the 1x1 screen grows the canvas to 9001x9001.
    [(0, 0, 1, 1, 0), (9000, 9000, 1, 1, 0)],
    # A 9001x9001 frame also has its disposal area allocated inside seek().
    [(0, 0, 1, 1, 0), (0, 0, 9001, 9001, 2)],
    # A first frame that large is allocated while Pillow opens the file.
    [(0, 0, 9001, 9001, 2)],
], ids=["offset-frame", "disposal-frame", "first-frame"])
def test_gif_canvas_is_refused_before_pillow_parses_the_file(tmp_path, monkeypatch, frames):
    """Regression: open_image checked the first frame's canvas only after
    Image.open had allocated that frame's disposal area, and later frames not
    at all; a 37-byte GIF made Pillow allocate a 12000x12000 buffer."""
    from PIL import GifImagePlugin

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)  # A host may disable Pillow's limit.
    refused = refuse_large_allocations(monkeypatch)
    calls = _spy_on_plugin_open(monkeypatch, GifImagePlugin.GifImageFile)
    data = _gif((1, 1), frames)
    path = tmp_path / "canvas.gif"
    path.write_bytes(data)

    assert gif_dimensions(io.BytesIO(data)) == (9001, 9001)
    for source in (data, path, io.BytesIO(data), _Unseekable(data)):
        with pytest.raises(Image.DecompressionBombError, match="9001x9001"):
            with open_image(source, max_pixels=10_000):
                pass
    assert calls == []
    assert refused == []


@pytest.mark.parametrize(("extension", "second_frame"), READ_PAST_TERMINATOR_CASES)
def test_gif_blocks_pillow_reads_past_a_terminator_are_refused(
    tmp_path, monkeypatch, extension, second_frame
):
    """Regression: the pre-scan stopped at these terminators while Pillow read
    on, so a frame hidden in what the pre-scan took for a comment was never
    bounded. A 258-byte file made Pillow allocate a 13000x13000 buffer."""
    from PIL import GifImagePlugin, UnidentifiedImageError

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    refused = refuse_large_allocations(monkeypatch)
    data = gif_hiding_a_frame(extension, second_frame=second_frame)
    # Pillow alone reaches the hidden frame and allocates its disposal area.
    with pytest.raises(MemoryError):
        with Image.open(io.BytesIO(data)) as image:
            image.seek(image.n_frames - 1)
    assert refused and refused[-1].endswith("(13000, 13000))")
    refused.clear()
    calls = _spy_on_plugin_open(monkeypatch, GifImagePlugin.GifImageFile)
    path = tmp_path / "hidden.gif"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="terminator"):
        gif_dimensions(io.BytesIO(data))
    for source in (data, path, io.BytesIO(data), _Unseekable(data)):
        with pytest.raises(UnidentifiedImageError, match="not a recognizable"):
            with open_image(source):
                pass
    assert calls == []
    assert refused == []


def test_gif_dimensions_match_the_canvas_pillow_decodes():
    """Blocks Pillow reads as the GIF format does are accepted, and the bound is
    exactly the canvas Pillow grows to while decoding every frame."""
    frames = [Image.new("P", (5, 4), index) for index in range(3)]
    saved = io.BytesIO()
    frames[0].save(saved, format="GIF", save_all=True, append_images=frames[1:],
                   loop=0, comment=b"hi")
    grown = _gif((3, 2), [(0, 0, 1, 1, 0), (6, 1, 1, 1, 2), (1, 4, 1, 1, 3)])
    last = grown.rindex(b"\x21\xf9\x04")
    tolerated = (
        grown[:last]
        + b"\x21\xfe\x00"  # an empty comment, which Pillow also ends at its terminator
        + b"\x07"  # a stray byte, which Pillow skips
        + b"\x21\xff\x0bNETSCAPE2.0\x00"  # after the first frame, read as the format does
        + grown[last:-1]
        + b"\x21\x01\x00"  # an empty extension that nothing follows
    )

    for data, canvas in [(saved.getvalue(), (5, 4)), (grown, (7, 5)), (tolerated, (7, 5))]:
        assert gif_dimensions(io.BytesIO(data)) == canvas
        with open_image(data, max_pixels=canvas[0] * canvas[1]) as image:
            for index in range(image.n_frames):
                image.seek(index)
                image.load()
            assert image.size == canvas
    assert gif_dimensions(io.BytesIO(b"\x89PNG\r\n\x1a\n" + bytes(16))) is None


def test_a_path_is_opened_once_and_pillow_parses_that_stream(tmp_path, monkeypatch):
    """Regression: the header check and Pillow each opened the path, so the
    file Pillow parsed could differ from the one that was checked."""
    import builtins

    path = tmp_path / "small.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(path)
    handles = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if isinstance(file, (str, os.PathLike)) and os.fspath(file) == os.fspath(path):
            handles.append(handle)
        return handle

    monkeypatch.setattr(builtins, "open", recording_open)
    with open_image(path) as image:
        image.load()
        assert (image.format, image.size) == ("PNG", (3, 2))
        assert len(handles) == 1 and not handles[0].closed
    assert len(handles) == 1 and handles[0].closed
