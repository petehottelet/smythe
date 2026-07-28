"""Deterministic cyber-glyph provider and digital-rain asset assembly.

The 64 glyphs are fictional procedural marks drawn from small bitmap cells;
they do not reproduce a font, logo, or source image. The assembled visuals use
the general visual vocabulary of green digital rain -- black ground, glowing
descending columns, bright heads, and varied trails -- without copying exact
reference pixels.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from smythe.provider import Artifact, CompletionResult, Provider


GLYPH_COUNT = 64
TILE_SIZE = 128
PREVIEW_SIZE = (1920, 1080)
GIF_SIZE = (640, 360)
GIF_FRAMES = 12
ATLAS_SIZE = (1024, 1024)
DEFAULT_SEED = 0x5A17_2026


@dataclass(frozen=True, slots=True)
class GlyphSpec:
    """One fictional 7x9 bitmap glyph plus animation characteristics."""

    id: str
    rows: tuple[int, ...]
    speed: float
    trail_length: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"glyph-[0-9]{2}", self.id):
            raise ValueError(f"invalid glyph id: {self.id!r}")
        if len(self.rows) != 9 or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < 128
            for row in self.rows
        ):
            raise ValueError("rows must contain nine 7-bit integers")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise TypeError("speed must be numeric")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if isinstance(self.trail_length, bool) or not isinstance(self.trail_length, int):
            raise TypeError("trail_length must be an integer")
        if self.trail_length < 2:
            raise ValueError("trail_length must be at least two")


@dataclass(frozen=True, slots=True)
class OutputReceipt:
    """Objective, hash-bound facts about one generated output."""

    path: str
    sha256: str
    byte_size: int
    format: str
    width: int
    height: int
    frames: int = 1


@dataclass(frozen=True, slots=True)
class GlyphSuiteReceipt:
    """Receipts for a complete tile set and its four assembled views."""

    tiles: tuple[OutputReceipt, ...]
    preview: OutputReceipt
    animation: OutputReceipt
    atlas: OutputReceipt
    html: OutputReceipt
    unique_tile_hashes: int


def _build_glyph_specs(seed: int = DEFAULT_SEED) -> tuple[GlyphSpec, ...]:
    specs: list[GlyphSpec] = []
    seen: set[tuple[int, ...]] = set()
    for index in range(GLYPH_COUNT):
        nonce = 0
        while True:
            rng = random.Random(seed ^ (index * 0x9E3779B1) ^ nonce)
            rows: list[int] = []
            for y in range(9):
                # Two independent procedural motifs create circuit-like marks.
                row = 0
                for x in range(7):
                    diagonal = (x + y + index) % (3 + index % 3) == 0
                    noise = rng.random() < 0.23
                    edge = x in {0, 6} and rng.random() < 0.28
                    if diagonal ^ noise or edge:
                        row |= 1 << x
                rows.append(row)
            # Encode several index bits into separated cells. This guarantees
            # identity without turning the bitmap into a recognizable numeral.
            for bit in range(6):
                if index & (1 << bit):
                    y = (bit * 2 + 1) % 9
                    x = (bit * 3 + 2) % 7
                    rows[y] |= 1 << x
            pattern = tuple(rows)
            active = sum(row.bit_count() for row in pattern)
            if pattern not in seen and 13 <= active <= 43:
                seen.add(pattern)
                break
            nonce += 1
        motion_rng = random.Random(seed + index * 104729)
        specs.append(
            GlyphSpec(
                id=f"glyph-{index:02d}",
                rows=pattern,
                speed=round(motion_rng.uniform(0.65, 2.35), 3),
                trail_length=motion_rng.randint(8, 26),
            )
        )
    return tuple(specs)


GLYPH_SPECS = _build_glyph_specs()
_SPEC_BY_ID = {spec.id: spec for spec in GLYPH_SPECS}
_PROMPT_ID_RE = re.compile(r"CYBER_GLYPH_ID=(glyph-[0-9]{2})(?:\b|$)")


def glyph_prompt(spec: GlyphSpec) -> str:
    """Return a stable prompt that lets a concurrent provider select a glyph."""

    return (
        f"CYBER_GLYPH_ID={spec.id}\n"
        "Render one fictional cyber glyph as a transparent green luminous tile. "
        "Do not use letters, existing symbols, logos, or branded marks."
    )


def _select_spec(prompt: str) -> GlyphSpec:
    match = _PROMPT_ID_RE.search(prompt)
    if match:
        return _SPEC_BY_ID[match.group(1)]
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return GLYPH_SPECS[int.from_bytes(digest[:4], "big") % GLYPH_COUNT]


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise ImportError(
            "Glyph asset rendering requires Pillow; install smythe[benchmarks]"
        ) from exc
    return Image, ImageDraw, ImageFilter, ImageOps


def _glyph_core_mask(spec: GlyphSpec, size: int):
    Image, ImageDraw, _, _ = _pillow()
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cell = max(1, size // 12)
    glyph_width = 7 * cell
    glyph_height = 9 * cell
    x0 = (size - glyph_width) // 2
    y0 = (size - glyph_height) // 2
    inset = max(0, cell // 5)
    radius = max(0, cell // 4)
    for y, row in enumerate(spec.rows):
        for x in range(7):
            if not row & (1 << x):
                continue
            box = (
                x0 + x * cell + inset,
                y0 + y * cell + inset,
                x0 + (x + 1) * cell - inset,
                y0 + (y + 1) * cell - inset,
            )
            draw.rounded_rectangle(box, radius=radius, fill=255)
            # Short links turn the isolated cells into circuit-like marks.
            if x < 6 and row & (1 << (x + 1)):
                cy = y0 + y * cell + cell // 2
                draw.line((box[2], cy, box[2] + 2 * inset + 1, cy), fill=225, width=1)
            if y < 8 and spec.rows[y + 1] & (1 << x):
                cx = x0 + x * cell + cell // 2
                draw.line((cx, box[3], cx, box[3] + 2 * inset + 1), fill=225, width=1)
    return mask


def render_glyph_tile(spec: GlyphSpec, *, size: int = TILE_SIZE) -> bytes:
    """Render one deterministic transparent PNG tile."""

    if not isinstance(spec, GlyphSpec):
        raise TypeError("spec must be a GlyphSpec")
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    Image, _, ImageFilter, _ = _pillow()
    core = _glyph_core_mask(spec, size)
    glow = core.filter(ImageFilter.GaussianBlur(max(1.0, size / 22)))
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    aura = Image.new("RGBA", (size, size), (32, 255, 92, 0))
    aura.putalpha(glow.point(lambda value: round(value * 0.42)))
    image.alpha_composite(aura)
    body = Image.new("RGBA", (size, size), (116, 255, 142, 0))
    body.putalpha(core)
    image.alpha_composite(body)
    # A small deterministic highlight gives each tile a luminous leading edge.
    highlight = core.crop((0, 0, size, max(1, size // 2)))
    white = Image.new("RGBA", (size, max(1, size // 2)), (224, 255, 232, 0))
    white.putalpha(highlight.point(lambda value: round(value * 0.34)))
    image.alpha_composite(white, (0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


class ProceduralGlyphProvider(Provider):
    """Offline provider returning one deterministic glyph PNG per prompt."""

    def __init__(self, *, latency_s: float = 0.0, tile_size: int = TILE_SIZE) -> None:
        if isinstance(latency_s, bool) or not isinstance(latency_s, (int, float)):
            raise TypeError("latency_s must be numeric")
        if latency_s < 0:
            raise ValueError("latency_s must be non-negative")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int):
            raise TypeError("tile_size must be an integer")
        if tile_size < 16:
            raise ValueError("tile_size must be at least 16")
        self.latency_s = float(latency_s)
        self.tile_size = tile_size
        self.calls: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        spec = _select_spec(prompt)
        self.calls.append(spec.id)
        return CompletionResult(
            text=json.dumps(
                {
                    "glyph_id": spec.id,
                    "procedural": True,
                    "tile_size": self.tile_size,
                },
                sort_keys=True,
            ),
            artifacts=[
                Artifact(
                    data=render_glyph_tile(spec, size=self.tile_size),
                    mime_type="image/png",
                )
            ],
            cost_usd=0.0,
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


def _encode_image(image, image_format: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def _image_receipt(path: Path) -> OutputReceipt:
    Image, _, _, _ = _pillow()
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        frames = int(getattr(image, "n_frames", 1))
        image_format = image.format or "UNKNOWN"
        width, height = image.size
    return OutputReceipt(
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        format=image_format,
        width=width,
        height=height,
        frames=frames,
    )


def _text_receipt(path: Path, *, width: int, height: int) -> OutputReceipt:
    data = path.read_bytes()
    return OutputReceipt(
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        format="HTML",
        width=width,
        height=height,
        frames=0,
    )


def normalize_tile(
    source: str | os.PathLike[str] | bytes,
    destination: str | os.PathLike[str],
    *,
    size: int = TILE_SIZE,
) -> OutputReceipt:
    """Normalize arbitrary raster input to a centered transparent square PNG."""

    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 16:
        raise ValueError("size must be at least 16")
    Image, _, _, ImageOps = _pillow()
    stream = io.BytesIO(source) if isinstance(source, bytes) else Path(source)
    with Image.open(stream) as loaded:
        loaded.load()
        tile = loaded.convert("RGBA")
    tile.putalpha(_extract_glyph_mask(tile))
    contained = ImageOps.contain(
        tile,
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    normalized = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    normalized.alpha_composite(
        contained,
        ((size - contained.width) // 2, (size - contained.height) // 2),
    )
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(normalized, "PNG", compress_level=6))
    return _image_receipt(path)


def _extract_glyph_mask(rgba):
    """Return a bounded foreground mask, including for opaque model output.

    GPT Image can return an opaque canvas even when transparency is requested.
    In that case, using alpha directly turns the whole tile into a rectangle.
    We instead measure distance from the corner-estimated background and reject
    masks that are empty or cover most of the tile.
    """

    Image, _, ImageFilter, _ = _pillow()
    alpha = rgba.getchannel("A")
    if alpha.getextrema() == (255, 255):
        from PIL import ImageChops

        rgb = rgba.convert("RGB")
        corners = (
            rgb.getpixel((0, 0)),
            rgb.getpixel((rgb.width - 1, 0)),
            rgb.getpixel((0, rgb.height - 1)),
            rgb.getpixel((rgb.width - 1, rgb.height - 1)),
        )
        background_color = tuple(
            round(sum(pixel[channel] for pixel in corners) / len(corners))
            for channel in range(3)
        )
        background = Image.new("RGB", rgb.size, background_color)
        difference = ImageChops.difference(rgb, background)
        red, green, blue = difference.split()
        alpha = ImageChops.lighter(red, ImageChops.lighter(green, blue)).point(
            lambda value: 0 if value < 20 else min(255, (value - 20) * 3)
        )
        alpha = alpha.filter(ImageFilter.GaussianBlur(0.5))

    histogram = alpha.histogram()
    visible_fraction = sum(histogram[17:]) / (alpha.width * alpha.height)
    if visible_fraction < 0.002:
        raise ValueError("tile has no separable glyph foreground")
    if visible_fraction > 0.60:
        raise ValueError(
            "tile foreground covers most of the canvas; refusing an opaque "
            "background or rectangular pseudo-glyph"
        )
    return alpha


def _load_tile_masks(tile_paths: Sequence[str | os.PathLike[str]], size: int):
    if len(tile_paths) != GLYPH_COUNT:
        raise ValueError(f"exactly {GLYPH_COUNT} tile paths are required")
    Image, _, _, _ = _pillow()
    masks = []
    for path in tile_paths:
        with Image.open(path) as tile:
            tile.load()
            rgba = tile.convert("RGBA")
        mask = _extract_glyph_mask(rgba).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        if mask.getbbox() is None:
            raise ValueError(f"tile has no visible pixels: {path}")
        masks.append(mask)
    return masks


def _render_rain_frame(
    tile_paths: Sequence[str | os.PathLike[str]],
    *,
    width: int,
    height: int,
    seed: int,
    frame_index: int,
    masks=None,
):
    Image, ImageDraw, ImageFilter, _ = _pillow()
    image = Image.new("RGB", (width, height), (0, 2, 1))
    cell = max(14, round(width / 96))
    glyph_size = max(12, cell - 2)
    masks = masks if masks is not None else _load_tile_masks(tile_paths, glyph_size)
    if len(masks) != GLYPH_COUNT:
        raise ValueError(f"exactly {GLYPH_COUNT} tile masks are required")
    rng = random.Random(seed)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    heads = Image.new("RGBA", image.size, (0, 0, 0, 0))

    column_count = math.ceil(width / cell) + 1
    for column in range(column_count):
        spec = GLYPH_SPECS[column % GLYPH_COUNT]
        trail = max(
            7,
            round(spec.trail_length * 1.6 * height / PREVIEW_SIZE[1]),
        )
        speed = max(1, round(spec.speed * cell * 0.68))
        cycle = height + (trail + 2) * cell
        start = rng.randrange(cycle)
        # Keep some part of every column visible. As a head moves below the
        # viewport, its trailing glyphs continue to drain before it wraps.
        head_y = (start + frame_index * speed) % cycle
        x = column * cell + rng.randint(-2, 2)
        base_glyph = rng.randrange(GLYPH_COUNT)
        for tail_index in range(trail, -1, -1):
            y = head_y - tail_index * cell
            if y < -glyph_size or y >= height:
                continue
            mask = masks[(base_glyph + tail_index * 7 + frame_index // 2) % GLYPH_COUNT]
            if tail_index == 0:
                glow = mask.filter(ImageFilter.GaussianBlur(max(1.0, glyph_size / 5)))
                halo = Image.new("RGBA", (glyph_size, glyph_size), (64, 255, 118, 0))
                halo.putalpha(glow.point(lambda value: round(value * 0.72)))
                heads.alpha_composite(halo, (x, y))
                head = Image.new("RGBA", (glyph_size, glyph_size), (225, 255, 232, 0))
                head.putalpha(mask)
                heads.alpha_composite(head, (x, y))
            else:
                proximity = 1 - tail_index / max(1, trail)
                green = round(68 + 182 * proximity)
                alpha = round(52 + 198 * proximity * proximity)
                body = Image.new("RGBA", (glyph_size, glyph_size), (34, green, 76, 0))
                body.putalpha(mask.point(lambda value, a=alpha: value * a // 255))
                overlay.alpha_composite(body, (x, y))

    glow = overlay.filter(ImageFilter.GaussianBlur(max(1.0, glyph_size / 8)))
    glow.putalpha(glow.getchannel("A").point(lambda value: round(value * 0.62)))
    image = Image.alpha_composite(image.convert("RGBA"), glow)
    image = Image.alpha_composite(image, overlay)
    image = Image.alpha_composite(image, heads)
    # Subtle dark scan lines add display texture without importing pixels.
    draw = ImageDraw.Draw(image)
    for y in range(2, height, 4):
        draw.line((0, y, width, y), fill=(0, 7, 3, 42), width=1)
    return image.convert("RGB")


def assemble_preview(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> OutputReceipt:
    """Assemble the full-resolution 1920x1080 still preview."""

    frame = _render_rain_frame(
        tile_paths,
        width=PREVIEW_SIZE[0],
        height=PREVIEW_SIZE[1],
        seed=seed,
        frame_index=29,
    )
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(frame, "PNG", compress_level=6))
    return _image_receipt(path)


def assemble_animation(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    frames: int = GIF_FRAMES,
) -> OutputReceipt:
    """Assemble a compact looping GIF proof of varied column motion."""

    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("frames must be an integer")
    if frames < 2:
        raise ValueError("frames must be at least two")
    cell = max(14, round(GIF_SIZE[0] / 78))
    masks = _load_tile_masks(tile_paths, max(12, cell - 2))
    rendered = [
        _render_rain_frame(
            tile_paths,
            width=GIF_SIZE[0],
            height=GIF_SIZE[1],
            seed=seed,
            frame_index=index * 3,
            masks=masks,
        )
        for index in range(frames)
    ]
    data = _encode_image(
        rendered[0],
        "GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=85,
        loop=0,
        optimize=True,
        disposal=2,
    )
    path = Path(destination)
    _atomic_write_bytes(path, data)
    return _image_receipt(path)


def assemble_atlas(
    tile_paths: Sequence[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
) -> OutputReceipt:
    """Assemble an 8x8 contact-sheet atlas of normalized tiles."""

    if len(tile_paths) != GLYPH_COUNT:
        raise ValueError(f"exactly {GLYPH_COUNT} tile paths are required")
    Image, ImageDraw, _, _ = _pillow()
    atlas = Image.new("RGB", ATLAS_SIZE, (0, 3, 1))
    draw = ImageDraw.Draw(atlas)
    for index, path_value in enumerate(tile_paths):
        with Image.open(path_value) as source:
            tile = source.convert("RGBA")
        x = (index % 8) * TILE_SIZE
        y = (index // 8) * TILE_SIZE
        atlas.paste(tile, (x, y), tile)
        draw.rectangle((x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1), outline=(7, 42, 18))
    path = Path(destination)
    _atomic_write_bytes(path, _encode_image(atlas, "PNG", compress_level=6))
    return _image_receipt(path)


def _html_document(seed: int) -> str:
    patterns = [list(spec.rows) for spec in GLYPH_SPECS]
    speeds = [spec.speed for spec in GLYPH_SPECS]
    trails = [spec.trail_length for spec in GLYPH_SPECS]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fictional Cyber Glyph Rain</title>
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#000}}
canvas{{display:block;width:100vw;height:100vh;background:#000}}
</style>
</head>
<body>
<canvas id="rain" width="1920" height="1080" aria-label="Animated fictional green cyber glyph rain"></canvas>
<script>
const patterns={json.dumps(patterns, separators=(",", ":"))};
const speeds={json.dumps(speeds, separators=(",", ":"))};
const trails={json.dumps(trails, separators=(",", ":"))};
let state={seed & 0xFFFFFFFF};
function rand(){{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296}}
const canvas=document.getElementById('rain'),ctx=canvas.getContext('2d');
const cell=25,columns=[];
for(let x=0;x<canvas.width+cell;x+=cell){{
  const g=Math.floor(rand()*64);
  columns.push({{x:x+(rand()*5-2),y:rand()*canvas.height-canvas.height,
    glyph:g,speed:speeds[g]*70,length:trails[g],phase:Math.floor(rand()*64)}});
}}
function glyph(pattern,x,y,size,color,alpha){{
  const unit=size/9;ctx.fillStyle=color;ctx.globalAlpha=alpha;
  for(let row=0;row<9;row++)for(let col=0;col<7;col++)if(pattern[row]&(1<<col))
    ctx.fillRect(x+col*unit+1,y+row*unit+1,Math.max(1,unit-1),Math.max(1,unit-1));
}}
let previous=performance.now();
function draw(now){{
  const dt=Math.min(.05,(now-previous)/1000);previous=now;
  ctx.globalAlpha=1;ctx.fillStyle='rgba(0,2,1,.30)';ctx.fillRect(0,0,canvas.width,canvas.height);
  for(const column of columns){{
    column.y+=column.speed*dt;
    if(column.y-column.length*cell>canvas.height){{column.y=-cell;column.glyph=(column.glyph+17)%64}}
    for(let tail=column.length;tail>=0;tail--){{
      const y=column.y-tail*cell;if(y<-cell||y>canvas.height)continue;
      const index=(column.glyph+column.phase+tail*7)%64,near=1-tail/column.length;
      if(tail===0){{ctx.shadowColor='#63ff8d';ctx.shadowBlur=14;glyph(patterns[index],column.x,y,cell-3,'#e3ffe9',1)}}
      else{{ctx.shadowBlur=0;glyph(patterns[index],column.x,y,cell-3,`rgb(28,${{Math.round(48+174*near)}},72)`,.14+.76*near*near)}}
    }}
  }}
  ctx.shadowBlur=0;ctx.globalAlpha=1;requestAnimationFrame(draw);
}}
requestAnimationFrame(draw);
</script>
</body>
</html>
"""


def assemble_html(
    destination: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> OutputReceipt:
    """Write a self-contained animated 1920x1080 HTML canvas screensaver."""

    path = Path(destination)
    _atomic_write_bytes(path, _html_document(seed).encode("utf-8"))
    return _text_receipt(path, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])


def build_glyph_screensaver_assets(
    output_dir: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
) -> GlyphSuiteReceipt:
    """Build all 64 tiles and the four deterministic flagship outputs."""

    root = Path(output_dir)
    tile_dir = root / "tiles"
    tile_receipts = tuple(
        normalize_tile(
            render_glyph_tile(spec),
            tile_dir / f"{spec.id}.png",
        )
        for spec in GLYPH_SPECS
    )
    unique = len({receipt.sha256 for receipt in tile_receipts})
    if unique != GLYPH_COUNT:
        raise RuntimeError(f"expected {GLYPH_COUNT} unique tiles, got {unique}")
    tile_paths = [receipt.path for receipt in tile_receipts]
    preview = assemble_preview(tile_paths, root / "glyph-rain-preview.png", seed=seed)
    animation = assemble_animation(tile_paths, root / "glyph-rain-loop.gif", seed=seed)
    atlas = assemble_atlas(tile_paths, root / "glyph-atlas.png")
    html = assemble_html(root / "glyph-rain.html", seed=seed)
    return GlyphSuiteReceipt(
        tiles=tile_receipts,
        preview=preview,
        animation=animation,
        atlas=atlas,
        html=html,
        unique_tile_hashes=unique,
    )


__all__ = [
    "ATLAS_SIZE",
    "DEFAULT_SEED",
    "GIF_FRAMES",
    "GIF_SIZE",
    "GLYPH_COUNT",
    "GLYPH_SPECS",
    "PREVIEW_SIZE",
    "TILE_SIZE",
    "GlyphSpec",
    "GlyphSuiteReceipt",
    "OutputReceipt",
    "ProceduralGlyphProvider",
    "assemble_animation",
    "assemble_atlas",
    "assemble_html",
    "assemble_preview",
    "build_glyph_screensaver_assets",
    "glyph_prompt",
    "normalize_tile",
    "render_glyph_tile",
]
