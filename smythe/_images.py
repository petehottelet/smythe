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
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def open_image(
    source: bytes | str | os.PathLike[str] | IO[bytes],
    *,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> Iterator[Any]:
    """Open *source* with only the allowed Pillow plugins.

    Only the header has been read when the image is yielded. Raises
    ``PIL.UnidentifiedImageError`` for any other or unrecognizable format and
    ``PIL.Image.DecompressionBombError`` when the declared size exceeds
    *max_pixels*; a WebP canvas is checked before Pillow's WebP plugin reads
    the file. A missing path raises ``FileNotFoundError`` as usual.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as stream:
            header = stream.read(_HEADER_BYTES)
    else:
        try:
            source.seek(0)  # Image.open also parses a stream from its start.
        except (AttributeError, io.UnsupportedOperation):
            source = io.BytesIO(source.read())
        header = source.read(_HEADER_BYTES)
    canvas = webp_dimensions(header)
    if canvas is not None and canvas[0] * canvas[1] > max_pixels:
        raise Image.DecompressionBombError(
            f"image is {canvas[0]}x{canvas[1]}, which exceeds the {max_pixels}-pixel limit"
        )
    try:
        image = Image.open(source, formats=ALLOWED_IMAGE_FORMATS)
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
    "open_image",
    "webp_dimensions",
]
