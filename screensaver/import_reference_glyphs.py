"""Import pinned classic SVG artwork, without upstream implementation code.

Reproduce the checked-in files offline with:
    python screensaver/import_reference_glyphs.py --check

The initial import can read a pinned local clone with --source-repo PATH.
SVG coordinates use Decimal arithmetic; cubic curves are never flattened in
the exported artwork. Only the PNG inspection sheet samples the curves.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "screensaver/svg-preview"
VERSION = "m8e-classic-svg-v1"
COMMIT = "5ba90490453ceceb6812d6b1bc658a99a92411d0"
REPOSITORY = "https://github.com/m8e/matrix-rain"
UPSTREAM_REPOSITORY = "https://github.com/Rezmason/matrix"
ASSET = "svg sources/texture_simplified.svg"
SVG_SHA256 = "37bf6ef51382e4a29a236a7067772cca2eeaac26ff9682d34534f8d932fb8a89"
LICENSE_SHA256 = "65333257cebb87e8af86201a65f0e550d65b3cf5a468bfee2a69c9d2b5d21123"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4}
SCALE = Decimal(100) / Decimal(64)


def _number(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_contours(path: str) -> list[list[tuple]]:
    """Convert SVG M/L/H/V/C/S/Z, absolute or relative, to absolute M/L/C/Z.

    SVG fill implicitly closes open subpaths. Make those closures explicit
    while retaining all contour order, curve controls, and winding.
    """
    tokens = []
    end = 0
    for match in TOKEN.finditer(path):
        if path[end:match.start()].strip(" ,\t\r\n"):
            raise ValueError("invalid SVG path token")
        tokens.append(match.group())
        end = match.end()
    if path[end:].strip(" ,\t\r\n") or not tokens:
        raise ValueError("empty or invalid SVG path")
    contours, contour = [], []
    point = (Decimal(0), Decimal(0))
    start, last_control, previous = point, None, None
    position, command = 0, None

    def close():
        if contour:
            if contour[-1][0] != "Z":
                contour.append(("Z",))
            contours.append(contour.copy())
            contour.clear()

    while position < len(tokens):
        if tokens[position].isalpha():
            command = tokens[position]
            position += 1
            if command.upper() == "Z":
                if not contour:
                    raise ValueError("close without a subpath")
                close()
                point, last_control, previous, command = start, None, "Z", None
                continue
        if command is None or command.upper() not in ARITY:
            raise ValueError("unsupported or missing SVG path command")
        kind, relative = command.upper(), command.islower()
        count = ARITY[kind]
        raw = tokens[position:position+count]
        if len(raw) != count or any(value.isalpha() for value in raw):
            raise ValueError("incomplete SVG path command")
        values = [Decimal(value) for value in raw]
        position += count
        if not all(value.is_finite() for value in values):
            raise ValueError("non-finite SVG coordinate")

        def absolute(x, y):
            return (x+point[0], y+point[1]) if relative else (x, y)

        if kind == "M":
            close()
            point = absolute(*values)
            start = point
            contour.append(("M", *point))
            command = "l" if relative else "L"
            last_control = None
        else:
            if not contour:
                raise ValueError("drawing command without a subpath")
            if kind == "H":
                target = (values[0]+point[0] if relative else values[0], point[1])
                contour.append(("L", *target))
            elif kind == "V":
                target = (point[0], values[0]+point[1] if relative else values[0])
                contour.append(("L", *target))
            elif kind == "L":
                target = absolute(*values)
                contour.append(("L", *target))
            elif kind == "C":
                first = absolute(*values[:2])
                second = absolute(*values[2:4])
                target = absolute(*values[4:])
                contour.append(("C", *first, *second, *target))
            else:
                first = (2*point[0]-last_control[0], 2*point[1]-last_control[1]) if previous in {"C", "S"} else point
                second = absolute(*values[:2])
                target = absolute(*values[2:])
                contour.append(("C", *first, *second, *target))
            point = target
            last_control = second if kind in {"C", "S"} else None
        previous = kind
    close()
    return contours


def _cubic_extrema(points):
    """Include curve extrema, not just controls (controls may leave the cell)."""
    roots = {0.0, 1.0}
    for axis in (0, 1):
        p0, p1, p2, p3 = [float(point[axis]) for point in points]
        a, b, c = p3-3*p2+3*p1-p0, 2*(p2-2*p1+p0), p1-p0
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                roots.add(-c/b)
        else:
            discriminant = b*b-4*a*c
            if discriminant >= 0:
                roots.update(((-b+math.sqrt(discriminant))/(2*a),
                              (-b-math.sqrt(discriminant))/(2*a)))
    return [
        tuple(sum(float(point[axis])*weight for point, weight in zip(
            points, ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3),
        )) for axis in (0, 1))
        for t in roots if 0 <= t <= 1
    ]


def _normalized_path(contours, column, row):
    commands = []
    for contour in contours:
        point = None
        for command in contour:
            numbers = [
                (value-Decimal(64*(column if index % 2 == 0 else row)))*SCALE
                for index, value in enumerate(command[1:])
            ]
            points = list(zip(numbers[::2], numbers[1::2]))
            geometry = _cubic_extrema([point, *points]) if command[0] == "C" else points
            if any(value < -1e-8 or value > 100+1e-8 for pair in geometry for value in pair):
                raise ValueError("a contour crosses its source atlas cell")
            commands.append(command[0] + " ".join(_number(value) for value in numbers))
            if points:
                point = points[-1]
    return " ".join(commands)


def build_catalog(svg_bytes: bytes, license_bytes: bytes) -> dict:
    """Verify the pinned source bytes and preserve every occupied source cell."""
    if hashlib.sha256(svg_bytes).hexdigest() != SVG_SHA256:
        raise ValueError("source SVG differs from the pinned Git blob")
    if hashlib.sha256(license_bytes).hexdigest() != LICENSE_SHA256:
        raise ValueError("license differs from the pinned Git blob")
    root = ET.fromstring(svg_bytes)
    if root.tag != f"{{{SVG_NAMESPACE}}}svg" or root.get("viewBox") != "0 0 512 512":
        raise ValueError("unexpected source SVG viewport")
    if len(root) != 1 or root[0].tag != f"{{{SVG_NAMESPACE}}}path" or set(root[0].attrib) != {"d"}:
        raise ValueError("expected one untransformed compound path")
    contours = parse_contours(root[0].get("d"))
    cells = defaultdict(list)
    for contour in contours:
        x, y = contour[0][1:]
        column = int((x/64).to_integral_value(rounding=ROUND_FLOOR))
        row = int((y/64).to_integral_value(rounding=ROUND_FLOOR))
        cells[row*8+column].append(contour)
    occupied = set(range(57)) - {4}
    if len(contours) != 84 or set(cells) != occupied:
        raise ValueError("the classic atlas cell or contour inventory changed")
    glyphs = []
    for index, slot in enumerate(sorted(cells)):
        column, row = slot % 8, slot // 8
        glyphs.append({
            "glyph_id": f"BASE-{index:03d}",
            "source_sequence_index": slot,
            "source_cell": [column, row],
            "contour_count": len(cells[slot]),
            "paths": [_normalized_path(cells[slot], column, row)],
        })
    return {
        "version": VERSION,
        "source": {
            "repository": REPOSITORY,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "commit": COMMIT,
            "asset": ASSET,
            "sha256": SVG_SHA256,
            "license_sha256": LICENSE_SHA256,
            "view_box": [0, 0, 512, 512],
            "grid": [8, 8],
            "cell": [64, 64],
            "sequence_length": 57,
            "blank_sequence_indices": [4],
            "unused_sequence_indices": list(range(57, 64)),
        },
        "count": len(glyphs),
        "canvas": [100, 100],
        "fill_rule": "nonzero",
        "glyphs": glyphs,
    }


def glyph_svg(glyph: dict) -> str:
    """Serialize exact normalized geometry with attribution in SVG metadata."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
        f'<title>{glyph["glyph_id"]} — imported classic reference glyph</title>\n'
        f'<desc>Artwork from m8e/matrix-rain, fork of Rezmason/matrix, commit {COMMIT}. '
        'See LICENSE and README.md in this directory for attribution and stated origins. '
        f'Source atlas slot {glyph["source_sequence_index"]}; cell translated and uniformly '
        'scaled 100/64 without redrawing.</desc>\n'
        + "".join(f'<path fill="#000000" fill-rule="nonzero" d="{path}"/>\n' for path in glyph["paths"])
        + '</svg>\n'
    )


def _flatten_cubic(a, b, c, d, points, depth=0):
    """Inspection raster only: bound curve-to-polyline error to .01 units."""
    dx, dy = d[0]-a[0], d[1]-a[1]
    denominator = dx*dx+dy*dy

    def distance(point):
        t = min(1, max(0, ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denominator)) if denominator else 0
        return math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)

    if max(distance(b), distance(c)) <= .01 or depth == 16:
        points.append(d)
        return

    def midpoint(left, right):
        return ((left[0]+right[0])/2, (left[1]+right[1])/2)

    ab, bc, cd = midpoint(a, b), midpoint(b, c), midpoint(c, d)
    abc, bcd = midpoint(ab, bc), midpoint(bc, cd)
    middle = midpoint(abc, bcd)
    _flatten_cubic(a, ab, abc, middle, points, depth+1)
    _flatten_cubic(middle, bcd, cd, d, points, depth+1)


def render_glyph(glyph: dict, size: int = 128):
    """Render an inspection image; the exported SVG retains exact cubic curves."""
    import numpy as np
    from PIL import Image, ImageDraw

    winding = np.zeros((size*4, size*4), dtype=np.int16)
    for path in glyph["paths"]:
        for contour in parse_contours(path):
            points = []
            for command in contour:
                numbers = tuple(float(value) for value in command[1:])
                if command[0] in {"M", "L"}:
                    points.append(numbers)
                elif command[0] == "C":
                    _flatten_cubic(points[-1], numbers[:2], numbers[2:4], numbers[4:], points)
            area = sum(x*y2-x2*y for (x, y), (x2, y2) in zip(points, points[1:]+points[:1]))
            mask = Image.new("L", (size*4, size*4), 0)
            ImageDraw.Draw(mask).polygon([(x*size/25, y*size/25) for x, y in points], fill=1)
            winding += np.asarray(mask, dtype=np.int16) * (1 if area > 0 else -1)
    return Image.fromarray(np.where(winding != 0, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS,
    ).convert("RGB")


def contact_sheet(catalog: dict) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    sheet = Image.new("RGB", (8*144, 7*158+54), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=12)
    draw.text((12, 10), "CLASSIC REFERENCE ARTWORK - 56 VISIBLE GLYPHS", fill="black", font=font)
    draw.text((12, 29), "m8e/matrix-rain / Rezmason/matrix - original curves; source slot 4 is blank", fill="black", font=font)
    for index, glyph in enumerate(catalog["glyphs"]):
        x, y = (index % 8)*144+8, (index//8)*158+54
        sheet.paste(render_glyph(glyph), (x, y))
        draw.text((x+1, y+133), f'{glyph["glyph_id"]} / slot {glyph["source_sequence_index"]:02d}', fill="black", font=font)
    output = io.BytesIO()
    sheet.save(output, format="PNG")
    return output.getvalue()


def provenance(catalog: dict) -> dict:
    return {
        "version": VERSION,
        "source": catalog["source"],
        "copyright_notice": "Copyright (c) 2018 Rezmason",
        "repository_license": "MIT; exact pinned notice included in LICENSE",
        "artwork_origin_as_stated_upstream": (
            "Cleaned vector artwork from an archived SWF on the official The Matrix: "
            "Path of Neo promotional website. Upstream describes katakana-derived forms "
            "and characters from Susan Kare's Chicago typeface."
        ),
        "origin_documentation": f"{REPOSITORY}/blob/{COMMIT}/README.md#goals",
        "archived_promotional_source": "https://web.archive.org/web/20070914173039/http://www.atari.com:80/thematrixpathofneo/",
        "rights_scope": (
            "The repository MIT notice is preserved as published. It is not a separate "
            "clearance or assertion of ownership of every underlying film or typeface element."
        ),
        "transformation": {
            "source_contours": 84,
            "visible_glyphs": 56,
            "coordinate_method": "exact Decimal translation of each 64x64 cell, then uniform 100/64 scaling",
            "curves": "Cubic Bezier geometry retained; relative and shorthand commands expanded exactly",
            "closure": "Implicit fill closures serialized explicitly; nonzero winding retained",
            "artwork_redrawn": False,
            "mirrored": False,
            "upstream_implementation_code_copied": False,
            "inspection_png": "4x antialiasing; adaptive cubic sampling at 0.01 normalized-unit tolerance",
        },
        "exporter": "screensaver/import_reference_glyphs.py",
        "original_generated_catalog": "The separate 192 original SVG catalog is unchanged and is not part of this import.",
    }


def attribution_readme() -> str:
    return f"""# Classic reference artwork

These 56 visible glyphs are imported from
[m8e/matrix-rain]({REPOSITORY}), a fork of
[Rezmason/matrix]({UPSTREAM_REPOSITORY}), pinned to
`{COMMIT}`. This directory contains artwork, not upstream rendering code.

The original [classic atlas]({REPOSITORY}/blob/{COMMIT}/svg%20sources/texture_simplified.svg)
is included as `texture_simplified.svg`. It has a 512×512 viewBox and an 8×8 grid
of 64×64 cells. Its 57-slot active sequence includes blank slot 4 and 56 visible
glyphs; slots 57–63 are unused. `BASE-000.svg` through `BASE-055.svg` preserve
the visible cells in row-major source order. Each cell is translated to the
origin and uniformly scaled by 100/64. Cubic curves, contour order, winding,
spacing, and handedness are retained. No glyph has been redrawn or fitted to
its ink bounding box. `contact-sheet.png` shows the full imported catalog.

## Credit and stated origins

The upstream repository carries the MIT License with
**Copyright (c) 2018 Rezmason**. Its exact pinned notice is included in
[LICENSE](LICENSE). The source artwork and normalized derivatives retain this
attribution. Smythe's importer is independently implemented. The adapted
renderer is documented separately in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The upstream [Goals section]({REPOSITORY}/blob/{COMMIT}/README.md#goals)
states that the classic vectors were cleaned from an archived SWF on the
official *The Matrix: Path of Neo* promotional website. It identifies
katakana-derived forms and characters from Susan Kare's Chicago typeface.
That stated history is recorded here rather than assigning Smythe authorship
to these shapes. The repository MIT notice does not separately establish
clearance of every underlying film or typeface element.

Only the classic atlas is imported. The Coptic, Gothic, Huberfish, and
*Resurrections* assets and their separate stated origins are outside this
import. Smythe's 192 original SVGs remain a separate, unchanged catalog.

## Reproduction

Run `python screensaver/import_reference_glyphs.py --check` from the repository
root. The check verifies pinned source and license hashes, all 84 contours,
all 56 visible cells, the blank slot, normalized SVGs, metadata, and browser data.
It needs the repository's `glyphs` or `dev` extra for the inspection PNG.
`provenance.json` records source hashes and exact transformation details.
"""


def export_files(svg_bytes: bytes, license_bytes: bytes) -> dict[str, bytes]:
    catalog = build_catalog(svg_bytes, license_bytes)
    js = ("// Imported reference artwork; see reference/README.md and reference/LICENSE.\n"
          "globalThis.BASE_GLYPHS = " + json.dumps(catalog, separators=(",", ":")) + ";\n")
    files = {
        "base-glyphs.js": js.encode(),
        "reference/texture_simplified.svg": svg_bytes,
        "reference/LICENSE": license_bytes,
        "reference/provenance.json": (json.dumps(provenance(catalog), indent=2)+"\n").encode(),
        "reference/README.md": attribution_readme().encode(),
        "reference/catalog.json": (json.dumps(catalog, indent=2)+"\n").encode(),
        "reference/contact-sheet.png": contact_sheet(catalog),
    }
    for glyph in catalog["glyphs"]:
        files[f'reference/{glyph["glyph_id"]}.svg'] = glyph_svg(glyph).encode()
    return files


def read_inputs(out: Path, source_repo: Path | None = None) -> tuple[bytes, bytes]:
    if source_repo is None:
        return ((out / "reference/texture_simplified.svg").read_bytes(),
                (out / "reference/LICENSE").read_bytes())
    head = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError(f"expected reference clone at {COMMIT}, got {head}")
    return tuple(subprocess.check_output(["git", "-C", str(source_repo), "show", f"{COMMIT}:{path}"])
                 for path in (ASSET, "LICENSE"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, help="local pinned reference clone; no network access")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    args = parser.parse_args()
    files = export_files(*read_inputs(args.out, args.source_repo))
    mismatches = []
    for relative, content in files.items():
        target = args.out / relative
        if args.check:
            if not target.is_file() or target.read_bytes() != content:
                mismatches.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    if mismatches:
        raise SystemExit("reference exports differ: " + ", ".join(mismatches))
    print(f'{"Verified" if args.check else "Exported"} 56 visible reference glyphs; '
          f'84 contours; 57 source slots including blank 4; {len(files)} files')


if __name__ == "__main__":
    main()
