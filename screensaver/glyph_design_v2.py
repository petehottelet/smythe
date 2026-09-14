"""Current 192-glyph contour catalog; preserves the historical v1 evidence.

Contours are authored here without reading reference outlines. Flat terminals,
continuous bowls and deliberate counters replace v1's cut-and-patch grammar.
The existing restricted SVG serializer/rasterizer is reused, not its geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from shapely import Polygon, orient_polygons, union_all

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.svg_glyphs import _ring_path, render_svg, validate_svg  # noqa: E402


VERSION = "glyph-svg-v2-1"
COUNT = 192
# Each tuple contains complete filled outlines. Interior rings are deliberate
# counters. No circular punches, pressure oscillation, random cuts, caps, or
# morphological opening/closing are applied to the finished silhouette.
OUTLINES = tuple((item["recipe"], item["outline"]) for item in json.loads(
    (Path(__file__).with_name("glyph_contours_v2.json")).read_text(encoding="utf-8")))


def geometry(index: int, *, outlines=OUTLINES):
    """Construct authored contours with 64 samples per cubic, without repair."""
    if not 1 <= len(outlines) <= 256:
        raise ValueError("An authored catalog must contain 1 through 256 outlines")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(outlines):
        raise ValueError(f"index must be an integer from 0 through {len(outlines) - 1}")
    tokens = re.findall(r"[MLCZ]|-?\d+(?:\.\d+)?", outlines[index][1])
    rings, ring = [], []
    cursor = 0
    while cursor < len(tokens):
        command = tokens[cursor]
        cursor += 1
        if command == "Z":
            rings.append(ring)
            ring = []
            continue
        count = 6 if command == "C" else 2
        values = [float(v) for v in tokens[cursor:cursor + count]]
        cursor += count
        if command in {"M", "L"}:
            ring.append(tuple(values))
        elif command == "C":
            a, b, c, d = ring[-1], values[:2], values[2:4], values[4:]
            for step in range(1, 65):
                t, u = step / 64, 1 - step / 64
                ring.append(tuple(u**3*a[k] + 3*u*u*t*b[k] + 3*u*t*t*c[k] + t**3*d[k]
                                  for k in (0, 1)))
    # Ring containment defines authored counters; no size-based hole deletion.
    shapes = [Polygon(r) for r in rings]
    filled = []
    for shape in shapes:
        if not any(other.contains(shape) for other in shapes if other is not shape):
            holes = [list(other.exterior.coords) for other in shapes
                     if shape.contains(other) and other is not shape]
            filled.append(Polygon(shape.exterior.coords, holes))
    result = orient_polygons(union_all(filled))
    if result.is_empty or not result.is_valid:
        raise ValueError(f"Invalid authored silhouette {index}")
    return result


def generate_glyph(index: int, *, outlines=OUTLINES) -> dict:
    shape = geometry(index, outlines=outlines)
    parts = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    parts.sort(key=lambda part: (-part.area, part.bounds))
    paths = [" ".join([_ring_path(list(part.exterior.coords)),
                       *[_ring_path(list(hole.coords)) for hole in part.interiors]])
             for part in parts]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
           'width="128" height="128">'
           + ''.join(f'<path fill="#000000" fill-rule="nonzero" d="{p}"/>' for p in paths)
           + '</svg>\n')
    validation = validate_svg(svg)
    if not validation["passed"]:
        raise ValueError(validation["errors"])
    digest = hashlib.sha256(svg.encode()).hexdigest()
    return {"glyph_id": f"GLYPH-{index:03d}", "version": VERSION,
            "recipe": outlines[index][0], "terminal_policy": "flat-cut",
            "file": f"GLYPH-{index:03d}.svg", "svg_sha256": digest,
            "svg": svg, "sha256": digest}


def write_study(destination: Path, *, outlines=OUTLINES) -> dict:
    """Write a new review directory; never overwrite historical evidence."""
    from PIL import Image, ImageDraw, ImageFont

    if not 1 <= len(outlines) <= 256:
        raise ValueError("An authored catalog must contain 1 through 256 outlines")
    destination.mkdir(parents=True, exist_ok=False)
    count = len(outlines)
    glyphs = [generate_glyph(i, outlines=outlines) for i in range(count)]
    font = ImageFont.load_default(size=14)
    for size in (16, 32, 64, 128):
        tile_size = 128
        sheet = Image.new("RGB", (16 * 160, math.ceil(count / 16) * 180), "white")
        draw = ImageDraw.Draw(sheet)
        for i, glyph in enumerate(glyphs):
            tile = render_svg(glyph["svg"], size).convert("RGB")
            if size < tile_size:
                tile = tile.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            x, y = i % 16 * 160 + 16, i // 16 * 180 + 8
            sheet.paste(tile, (x, y))
            draw.text((x + 21, y + 138), glyph["glyph_id"], fill="black", font=font)
        sheet.save(destination / f"contact-sheet-{size}.png")
        if size == 128:
            sheet.save(destination / "contact-sheet.png")
            for page in range(math.ceil(count / 48)):
                sheet.crop((0, page * 540, 2560, min(sheet.height, (page + 1) * 540))).save(
                    destination / f"sheet-{page + 1}.png")
    for glyph in glyphs:
        (destination / (glyph["glyph_id"] + ".svg")).write_text(
            glyph["svg"], encoding="utf-8", newline="\n")
    manifest = {"version": VERSION, "status": "current-catalog",
                "count": count, "benchmark_claimable": False,
                "reference_artwork_included": False,
                "glyphs": [{k: v for k, v in g.items() if k != "svg"} for g in glyphs]}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    catalog = {"version": VERSION, "count": count, "canvas": [100, 100],
               "glyphs": [{"glyph_id": g["glyph_id"], "family": g["recipe"],
                           "paths": [p.attrib["d"] for p in ET.fromstring(g["svg"])],
                           "svg_sha256": g["sha256"]} for g in glyphs]}
    catalog["catalog_sha256"] = hashlib.sha256(json.dumps(
        catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    (destination / "glyphs.js").write_text(
        "// Generated from the current Smythe contour catalog.\n"
        + "globalThis.SVG_GLYPHS = " + json.dumps(catalog, separators=(",", ":")) + ";\n",
        encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(write_study(parser.parse_args().out), indent=2))
