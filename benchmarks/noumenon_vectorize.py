"""Trace glyph tiles into validated even-odd SVG outlines with Pillow only.

A tile's glyph mask is ``alpha > ALPHA_THRESHOLD`` when the tile carries
transparency and ``luminance > LUMINANCE_THRESHOLD`` when it is opaque (a glyph
on black). The mask is traced along pixel edges into closed outlines, collinear
points are merged, and the outlines become one ``fill-rule="evenodd"`` path
whose viewBox matches the tile.

Validation re-reads the written SVG, rasterizes its outlines at pixel centers
with even-odd parity -- equivalent to filling each loop and XOR-combining the
fills -- and requires an intersection-over-union with the source mask of at
least ``IOU_THRESHOLD``. Pixel-edge tracing is exact, so a correct round trip
scores 1.0. No numpy, tracing library, or SVG renderer is required.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from xml.etree import ElementTree

if TYPE_CHECKING:
    from PIL.Image import Image as PillowImage

ALPHA_THRESHOLD = 127
LUMINANCE_THRESHOLD = 64
IOU_THRESHOLD = 0.98
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
VECTORIZATION_METHOD = "pixel-edge boundary tracing with collinear-point merging"

# Clockwise quarter turn in SVG's y-down coordinates: east -> south -> west -> north.
_RIGHT_TURN = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}
_PATH_TOKEN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z]")


@dataclass(frozen=True, slots=True)
class GlyphMask:
    """A binary glyph mask stored row-major: 1 for glyph, 0 for background."""

    width: int
    height: int
    bits: bytes
    source: str

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("mask dimensions must be positive")
        if len(self.bits) != self.width * self.height:
            raise ValueError("mask bits do not match the mask dimensions")
        if self.bits.translate(None, b"\x00\x01"):
            raise ValueError("mask bits must be 0 or 1")

    @property
    def pixel_count(self) -> int:
        return self.bits.count(1)


@dataclass(frozen=True, slots=True)
class SvgCheck:
    """The result of re-reading and rasterizing one written SVG."""

    valid: bool
    iou: float | None
    outline_count: int
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SvgReceipt:
    """Hash-bound facts about one vectorized tile."""

    path: str | None
    sha256: str | None
    byte_size: int
    width: int
    height: int
    mask_source: str
    mask_pixels: int
    outline_count: int
    vertex_count: int
    iou: float | None
    valid: bool
    errors: tuple[str, ...]


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise ImportError(
            "Glyph vectorization requires Pillow; install smythe[benchmarks]"
        ) from exc
    return Image, ImageStat


def _check_threshold(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 254:
        raise ValueError(f"{name} must be an integer from 0 to 254")


def glyph_mask(
    image: PillowImage,
    *,
    alpha_threshold: int = ALPHA_THRESHOLD,
    luminance_threshold: int = LUMINANCE_THRESHOLD,
) -> GlyphMask:
    """Binarize a Pillow image: alpha when it has transparency, else luminance."""

    _check_threshold("alpha_threshold", alpha_threshold)
    _check_threshold("luminance_threshold", luminance_threshold)
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    # Any pixel below full opacity means the tile carries transparency.
    if alpha.histogram()[255] < rgba.width * rgba.height:
        channel, threshold, source = alpha, alpha_threshold, "alpha"
    else:
        channel, threshold, source = rgba.convert("L"), luminance_threshold, "luminance"
    bits = channel.point(lambda value: 1 if value > threshold else 0).tobytes()
    return GlyphMask(rgba.width, rgba.height, bits, source)


def trace_outlines(mask: GlyphMask) -> list[list[tuple[int, int]]]:
    """Trace the mask's pixel-edge boundaries into closed loops of corner points.

    Every boundary edge runs with the glyph on its right: outer outlines turn
    clockwise and holes counter-clockwise in SVG's y-down space. Where two
    glyph pixels touch only at a corner the walk turns right, so diagonal
    neighbours stay separate outlines. Each boundary edge is used exactly once,
    which is why the loops' even-odd fill reproduces the mask exactly.
    """

    width, height, bits = mask.width, mask.height, mask.bits

    def filled(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and bits[y * width + x] == 1

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y in range(height):
        row = y * width
        if bits.find(1, row, row + width) < 0:
            continue
        for x in range(width):
            if bits[row + x] != 1:
                continue
            if not filled(x, y - 1):
                outgoing.setdefault((x, y), []).append((1, 0))
            if not filled(x + 1, y):
                outgoing.setdefault((x + 1, y), []).append((0, 1))
            if not filled(x, y + 1):
                outgoing.setdefault((x + 1, y + 1), []).append((-1, 0))
            if not filled(x - 1, y):
                outgoing.setdefault((x, y + 1), []).append((0, -1))

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    loops: list[list[tuple[int, int]]] = []
    for start, headings in outgoing.items():
        for first in headings:
            if (start, first) in visited:
                continue
            edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            vertex, heading = start, first
            while (vertex, heading) not in visited:
                visited.add((vertex, heading))
                edges.append((vertex, heading))
                vertex = (vertex[0] + heading[0], vertex[1] + heading[1])
                choices = outgoing[vertex]
                heading = choices[0] if len(choices) == 1 else _RIGHT_TURN[heading]
            loops.append(
                [
                    point
                    for index, (point, direction) in enumerate(edges)
                    if direction != edges[index - 1][1]
                ]
            )
    return loops


def outline_path_data(loops: Sequence[Sequence[tuple[int, int]]]) -> str:
    """Serialize axis-aligned corner loops as absolute M/H/V/Z commands."""

    commands: list[str] = []
    for loop in loops:
        x, y = loop[0]
        commands.append(f"M{x} {y}")
        for next_x, next_y in loop[1:]:
            if next_y == y:
                commands.append(f"H{next_x}")
            elif next_x == x:
                commands.append(f"V{next_y}")
            else:
                raise ValueError("outline segments must be axis-aligned")
            x, y = next_x, next_y
        commands.append("Z")
    return "".join(commands)


def svg_document(
    loops: Sequence[Sequence[tuple[int, int]]],
    *,
    width: int,
    height: int,
    fill: str,
) -> str:
    """Return one even-odd path whose viewBox matches the source tile."""

    return (
        f'<svg xmlns="{SVG_NAMESPACE}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<path fill="{fill}" fill-rule="evenodd" d="{outline_path_data(loops)}"/>'
        "</svg>\n"
    )


def parse_path_loops(d: str) -> list[list[tuple[float, float]]]:
    """Parse closed M/L/H/V/Z subpaths, absolute or relative, into point loops."""

    if _PATH_TOKEN.sub("", d).strip(" \t\r\n,"):
        raise ValueError("path data contains characters outside numbers and commands")
    tokens = _PATH_TOKEN.findall(d)
    loops: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None
    command: str | None = None
    x = y = start_x = start_y = 0.0
    index = 0

    def number() -> float:
        nonlocal index
        if index >= len(tokens) or tokens[index].isalpha():
            raise ValueError(f"path command {command!r} is missing a coordinate")
        value = float(tokens[index])
        index += 1
        if not math.isfinite(value):
            raise ValueError("path coordinates must be finite")
        return value

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in "Zz":
                if current is None:
                    raise ValueError("closepath has no open subpath")
                loops.append(current)
                current = None
                x, y = start_x, start_y
                continue
        elif command is None or command in "Zz":
            raise ValueError("path coordinates must follow a drawing command")
        if command in "Mm":
            if current is not None:
                raise ValueError("subpath is not closed before the next moveto")
            dx, dy = number(), number()
            x, y = (x + dx, y + dy) if command == "m" else (dx, dy)
            start_x, start_y = x, y
            current = [(x, y)]
            # Further coordinate pairs after a moveto are implicit linetos.
            command = "l" if command == "m" else "L"
        elif command in "LlHhVv":
            if current is None:
                raise ValueError(f"path command {command!r} has no current subpath")
            if command in "Ll":
                dx, dy = number(), number()
                x, y = (x + dx, y + dy) if command == "l" else (dx, dy)
            elif command in "Hh":
                value = number()
                x = x + value if command == "h" else value
            else:
                value = number()
                y = y + value if command == "v" else value
            current.append((x, y))
        else:
            raise ValueError(f"unsupported path command {command!r}")
    if current is not None:
        raise ValueError("subpath is not closed")
    return loops


def rasterize_evenodd(
    loops: Sequence[Sequence[tuple[float, float]]],
    width: int,
    height: int,
) -> bytes:
    """Fill closed loops with even-odd parity, sampling every pixel center.

    Parity is counted over all loops' crossings on each scanline, which equals
    filling every loop on its own and XOR-combining the fills.
    """

    crossings: list[list[float]] = [[] for _ in range(height)]
    for loop in loops:
        for index in range(len(loop)):
            x0, y0 = loop[index - 1]
            x1, y1 = loop[index]
            if y0 == y1:
                continue
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            slope = (x1 - x0) / (y1 - y0)
            # Half-open [y0, y1) keeps every scanline's crossing count even.
            for row in range(max(0, math.ceil(y0 - 0.5)), min(height, math.ceil(y1 - 0.5))):
                crossings[row].append(x0 + (row + 0.5 - y0) * slope)
    bits = bytearray(width * height)
    for row, xs in enumerate(crossings):
        if not xs:
            continue
        xs.sort()
        base = row * width
        for left, right in zip(xs[0::2], xs[1::2]):
            start = max(0, math.ceil(left - 0.5))
            stop = min(width, math.ceil(right - 0.5))
            if stop > start:
                bits[base + start : base + stop] = b"\x01" * (stop - start)
    return bytes(bits)


def mask_iou(first: bytes, second: bytes) -> float:
    """Intersection-over-union of two equal-size 0/1 masks."""

    if len(first) != len(second):
        raise ValueError("masks must have the same size")
    a = int.from_bytes(first, "big")
    b = int.from_bytes(second, "big")
    union = (a | b).bit_count()
    if not union:
        raise ValueError("intersection-over-union is undefined for two blank masks")
    return (a & b).bit_count() / union


def validate_svg(
    data: bytes,
    mask: GlyphMask,
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> SvgCheck:
    """Re-read a written SVG and compare its even-odd fill with the source mask."""

    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        return SvgCheck(False, None, 0, (f"SVG is not well-formed XML: {exc}",))
    errors: list[str] = []
    if root.tag != f"{{{SVG_NAMESPACE}}}svg":
        errors.append(f"root element {root.tag!r} is not an SVG element")
    view_box_text = root.get("viewBox", "")
    try:
        view_box = tuple(float(part) for part in re.split(r"[\s,]+", view_box_text.strip()))
    except ValueError:
        view_box = ()
    if view_box != (0.0, 0.0, float(mask.width), float(mask.height)):
        errors.append(
            f"viewBox {view_box_text!r} does not match the {mask.width}x{mask.height} tile"
        )
    for name, expected in (("width", mask.width), ("height", mask.height)):
        if root.get(name) != str(expected):
            errors.append(f"{name} {root.get(name)!r} does not match the tile ({expected})")

    loops: list[list[tuple[float, float]]] = []
    paths = root.findall(f"{{{SVG_NAMESPACE}}}path")
    if len(paths) != 1:
        errors.append(f"expected one path element, found {len(paths)}")
    else:
        if paths[0].get("fill-rule") != "evenodd":
            errors.append('path does not declare fill-rule="evenodd"')
        try:
            loops = parse_path_loops(paths[0].get("d", ""))
        except ValueError as exc:
            errors.append(f"path data is invalid: {exc}")
        else:
            if not loops:
                errors.append("path has no closed outlines")

    iou = None
    if not mask.pixel_count:
        errors.append("source glyph mask is blank")
    elif loops:
        measured = mask_iou(rasterize_evenodd(loops, mask.width, mask.height), mask.bits)
        iou = round(measured, 6)
        if measured < iou_threshold:
            errors.append(f"IoU {measured:.6f} is below the {iou_threshold} threshold")
    return SvgCheck(
        valid=not errors,
        iou=iou,
        outline_count=len(loops),
        errors=tuple(errors),
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def vectorize_tile(
    source: str | os.PathLike[str] | bytes,
    destination: str | os.PathLike[str],
    *,
    iou_threshold: float = IOU_THRESHOLD,
    alpha_threshold: int = ALPHA_THRESHOLD,
    luminance_threshold: int = LUMINANCE_THRESHOLD,
) -> SvgReceipt:
    """Trace one tile, write ``destination`` atomically, and validate the file.

    A blank mask fails and leaves no SVG at ``destination``, removing a stale
    one from an earlier run. The fill is the mean color of the glyph pixels,
    so the outline keeps the tile's hue.
    """

    Image, ImageStat = _pillow()
    stream = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
    with Image.open(stream) as loaded:
        loaded.load()
        rgba = loaded.convert("RGBA")
    mask = glyph_mask(
        rgba,
        alpha_threshold=alpha_threshold,
        luminance_threshold=luminance_threshold,
    )
    path = Path(destination)
    pixels = mask.pixel_count
    if not pixels:
        path.unlink(missing_ok=True)
        return SvgReceipt(
            path=None,
            sha256=None,
            byte_size=0,
            width=mask.width,
            height=mask.height,
            mask_source=mask.source,
            mask_pixels=0,
            outline_count=0,
            vertex_count=0,
            iou=None,
            valid=False,
            errors=("glyph mask is blank",),
        )
    loops = trace_outlines(mask)
    selector = Image.frombytes("L", rgba.size, mask.bits)
    mean = ImageStat.Stat(rgba.convert("RGB"), mask=selector).mean
    fill = "#" + "".join(f"{round(channel):02x}" for channel in mean[:3])
    _atomic_write_bytes(
        path,
        svg_document(loops, width=mask.width, height=mask.height, fill=fill).encode("utf-8"),
    )
    data = path.read_bytes()
    check = validate_svg(data, mask, iou_threshold=iou_threshold)
    return SvgReceipt(
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        width=mask.width,
        height=mask.height,
        mask_source=mask.source,
        mask_pixels=pixels,
        outline_count=check.outline_count,
        vertex_count=sum(len(loop) for loop in loops),
        iou=check.iou,
        valid=check.valid,
        errors=check.errors,
    )


__all__ = [
    "ALPHA_THRESHOLD",
    "IOU_THRESHOLD",
    "LUMINANCE_THRESHOLD",
    "SVG_NAMESPACE",
    "VECTORIZATION_METHOD",
    "GlyphMask",
    "SvgCheck",
    "SvgReceipt",
    "glyph_mask",
    "mask_iou",
    "outline_path_data",
    "parse_path_loops",
    "rasterize_evenodd",
    "svg_document",
    "trace_outlines",
    "validate_svg",
    "vectorize_tile",
]
