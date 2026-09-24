"""Restricted Pillow decoding for image bytes Smythe did not produce itself.

Pillow tries every registered plugin when ``Image.open`` is called without
``formats=``. Several of those plugins parse complex containers (TIFF) or hand
data to external programs (EPS launches Ghostscript when it is installed), so
provider output, logo masters and verifier inputs are opened only with the
plugins for the raster formats Smythe actually supports: PNG, JPEG, GIF and
WebP. That covers :class:`smythe.assets.ImageFormat`, the Jobs manifest image
MIME types and the raster logo-master suffixes.
"""

from __future__ import annotations

import io
import os
import struct
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import IO, Any

ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP")
# Pillow's JPEG plugin reports multi-picture JPEG files as "MPO". They are
# identified and decoded by that same plugin, so they count as JPEG here.
_JPEG_VARIANTS = {"MPO"}
# Pillow's own warning threshold. Pillow only warns between this and twice
# this many pixels; Smythe refuses anything above it before decoding pixels.
DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024 * 1024 // 4 // 3
# Leading bytes that hold every canvas size webp_dimensions() reads.
_HEADER_BYTES = 30
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")

NOT_ALLOWED_MESSAGE = "not a recognizable PNG, JPEG, GIF, or WebP image"


def webp_dimensions(header: bytes) -> tuple[int, int] | None:
    """Return the canvas size declared by a WebP file's first chunk, or None.

    Pillow hands a WebP file to libwebp, which allocates two canvas-sized
    RGBA buffers before Pillow can check the size. libwebp takes the canvas
    from the first chunk, so callers can refuse an oversized one from these
    header bytes first.
    """
    if not header.startswith(b"RIFF") or header[8:12] != b"WEBP":
        return None
    chunk = header[12:16]
    if chunk == b"VP8X" and len(header) >= 30:
        width = 1 + int.from_bytes(header[24:27], "little")
        height = 1 + int.from_bytes(header[27:30], "little")
        return width, height
    if chunk == b"VP8L" and len(header) >= 25 and header[20] == 0x2F:
        bits = int.from_bytes(header[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 " and len(header) >= 30 and header[23:26] == b"\x9d\x01\x2a":
        # Lossy: 14-bit width and height follow the frame tag and start code.
        width = int.from_bytes(header[26:28], "little") & 0x3FFF
        height = int.from_bytes(header[28:30], "little") & 0x3FFF
        return width, height
    return None


def gif_dimensions(stream: IO[bytes]) -> tuple[int, int] | None:
    """Return the largest canvas Pillow reaches while seeking every GIF frame.

    *stream* is read from its current position; None means it does not start
    with a complete GIF header. A frame may extend past the logical screen.
    Pillow's GIF plugin then grows the canvas and allocates the frame's
    disposal area inside ``seek()`` (inside ``Image.open`` for the first
    frame), before any caller can check the size, and
    ``PIL.Image.MAX_IMAGE_PIXELS = None`` turns off Pillow's own check. This
    reads the blocks exactly as that plugin does, including skipping stray
    bytes between them, so every frame it can reach is counted.

    In two places Pillow reads on past a data sub-block terminator as if more
    sub-blocks followed: after an extension, other than a comment, whose
    first sub-block is empty, and after a NETSCAPE2.0 extension before the
    first frame whose looping sub-block is empty. Pillow and the GIF format
    then disagree about what the following bytes are, so if any follow,
    ValueError is raised instead of guessing.
    """
    # Mirrors GifImageFile._open, data() and _seek(), which parse blocks the
    # same way in Pillow 11.1 and 12.3.
    header = stream.read(13)
    if len(header) < 13 or not header.startswith(_GIF_SIGNATURES):
        return None
    width, height = struct.unpack("<HH", header[6:10])
    if header[10] & 0x80:  # global color table
        stream.read(3 << ((header[10] & 7) + 1))

    def sub_block() -> bytes | None:
        # One data sub-block; None at a terminator or the end of the stream.
        size = stream.read(1)
        return stream.read(size[0]) if size and size[0] else None

    first_frame = True
    while True:
        introducer = stream.read(1)
        if not introducer or introducer == b";":  # end of the stream, or trailer
            break
        if introducer == b"!":  # extension label, then data sub-blocks
            label = stream.read(1)
            block = sub_block()
            if label == b"\xfe":  # comment: Pillow stops at its terminator
                while block:
                    block = sub_block()
                continue
            if (label == b"\xff" and first_frame and block is not None
                    and block.startswith(b"NETSCAPE2.0")):
                block = sub_block()  # Pillow reads the looping sub-block separately
            if block is None and stream.read(1):
                raise ValueError("Pillow would read past the terminator of this GIF extension")
            while sub_block():
                pass
        elif introducer == b",":  # image descriptor
            descriptor = stream.read(9)
            if len(descriptor) < 9:
                break  # Pillow raises here, before allocating anything for it
            left, top, frame_width, frame_height, flags = struct.unpack("<HHHHB", descriptor)
            width = max(width, left + frame_width)
            height = max(height, top + frame_height)
            if flags & 0x80:  # local color table
                stream.read(3 << ((flags & 7) + 1))
            if not stream.read(1):  # LZW minimum code size; Pillow raises without it
                break
            first_frame = False
            while sub_block():  # image data
                pass
        # Pillow skips any other byte between blocks.
    return width, height


@contextmanager
def open_image(
    source: bytes | str | os.PathLike[str] | IO[bytes],
    *,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> Iterator[Any]:
    """Open *source* with only the allowed Pillow plugins.

    A path is opened once, and Pillow parses that same stream, which closes
    when the context exits. No pixels have been decoded when the image is
    yielded. Raises ``PIL.UnidentifiedImageError`` for any other or
    unrecognizable format and ``PIL.Image.DecompressionBombError`` when the
    declared size exceeds *max_pixels*. A WebP canvas, and the largest canvas
    any GIF frame requires (see :func:`gif_dimensions`), are checked before
    Pillow's plugin reads the file, whatever ``PIL.Image.MAX_IMAGE_PIXELS``
    allows; a GIF whose blocks Pillow would read past a terminator is
    unrecognizable. A missing path raises ``FileNotFoundError`` as usual.
    """
    from PIL import Image, UnidentifiedImageError

    with ExitStack() as owned:
        if isinstance(source, (bytes, bytearray, memoryview)):
            stream: IO[bytes] = io.BytesIO(bytes(source))
        elif isinstance(source, (str, os.PathLike)):
            stream = owned.enter_context(open(source, "rb"))
        else:
            stream = source
            try:
                stream.seek(0)  # Image.open also parses a stream from its start.
            except (AttributeError, io.UnsupportedOperation):
                stream = io.BytesIO(stream.read())
        header = stream.read(_HEADER_BYTES)
        stream.seek(0)
        try:
            canvas = webp_dimensions(header) or gif_dimensions(stream)
        except ValueError as exc:
            raise UnidentifiedImageError(f"image is {NOT_ALLOWED_MESSAGE}") from exc
        stream.seek(0)
        if canvas is not None and canvas[0] * canvas[1] > max_pixels:
            raise Image.DecompressionBombError(
                f"image is {canvas[0]}x{canvas[1]}, which exceeds the {max_pixels}-pixel limit"
            )
        try:
            image = Image.open(stream, formats=ALLOWED_IMAGE_FORMATS)
        except UnidentifiedImageError as exc:
            raise UnidentifiedImageError(f"image is {NOT_ALLOWED_MESSAGE}") from exc
        except Image.DecompressionBombWarning as exc:
            # Raised only when the caller escalated Pillow's warning to an error.
            raise Image.DecompressionBombError(str(exc)) from exc
        with image:
            format_name = (image.format or "").upper()
            if format_name not in ALLOWED_IMAGE_FORMATS and format_name not in _JPEG_VARIANTS:
                raise UnidentifiedImageError(f"image is {NOT_ALLOWED_MESSAGE}")
            width, height = image.size
            if width * height > max_pixels:
                raise Image.DecompressionBombError(
                    f"image is {width}x{height}, which exceeds the {max_pixels}-pixel limit"
                )
            yield image


__all__ = [
    "ALLOWED_IMAGE_FORMATS",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "NOT_ALLOWED_MESSAGE",
    "gif_dimensions",
    "open_image",
    "webp_dimensions",
]
