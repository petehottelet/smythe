"""Derive a true MSDF atlas from the frozen original SVG catalog.

Build: python screensaver/export_generated_sdf.py --msdfgen PATH_TO_MSDFGEN
Check source and texture bindings without the build tool: add --check instead.
The build tool and intermediate images stay outside published artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/partitions/glyph_svg_v1/catalog"
PREVIEW = ROOT / "screensaver/svg-preview"
VERSION = "original-glyph-msdf-v1"
COUNT, COLUMNS, ROWS, CELL, RANGE = 192, 16, 12, 128, 16
NAMESPACE = "http://www.w3.org/2000/svg"
TOOL_RELEASE = "https://github.com/Chlumsky/msdfgen/releases/tag/v1.13"
TOOL_ARCHIVE = "https://github.com/Chlumsky/msdfgen/releases/download/v1.13/msdfgen-1.13-win64.zip"
TOOL_ARCHIVE_SHA256 = "08615362a7e4dc822fe1f4ef6a33ae62d7da0e3b91b6334a2ad2930bf4092b0c"
TOOL_EXE_SHA256 = "603707f3d08017f0e6adff2166ca663de4aa34596cf563f2232059a369f8375a"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combined_svg(source: str) -> str:
    """Keep source coordinates and combine same-winding fills for MSDFgen input.

    Some MSDFgen SVG parsers use only the last path. The original catalog has
    positive exterior and negative counter windings, so one compound nonzero
    path preserves their union without dropping detached components.
    """
    root = ET.fromstring(source)
    if root.tag != f"{{{NAMESPACE}}}svg" or root.get("viewBox") != "0 0 100 100":
        raise ValueError("expected the original 100x100 SVG canvas")
    paths = []
    for path in root:
        if path.tag != f"{{{NAMESPACE}}}path" or set(path.attrib) != {"fill", "fill-rule", "d"}:
            raise ValueError("expected untransformed filled source paths")
        if path.get("fill") != "#000000" or path.get("fill-rule") != "nonzero":
            raise ValueError("expected the original nonzero black fill")
        paths.append(path.get("d"))
    if not paths:
        raise ValueError("empty source SVG")
    # Width/height describe the intermediate viewport, not a fit to the ink.
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
            'width="100" height="100"><path fill="#000000" fill-rule="nonzero" '
            f'd="{" ".join(paths)}"/></svg>\n')


def command(executable: Path, source: Path, output: Path) -> list[str]:
    return [str(executable), "msdf", "-svg", str(source),
            "-dimensions", str(CELL), str(CELL), "-scale", "1.28", "-translate", "0", "0",
            "-pxrange", str(RANGE), "-coloringstrategy", "distance", "-seed", "0",
            "-fillrule", "nonzero", "-scanline", "-nopreprocess", "-o", str(output)]


def decode_mask(image: Image.Image) -> np.ndarray:
    return np.median(np.asarray(image.convert("RGB")), axis=2) > 127


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right)/union) if union else 1.0


def build_atlas(executable: Path, source_directory: Path, output_directory: Path,
                *, workers: int = 4) -> dict:
    from benchmarks.svg_glyphs import render_svg

    executable = executable.resolve()
    tool_version = subprocess.check_output([str(executable), "-version"], text=True).strip()
    if "MSDFgen v1.13.0" not in tool_version:
        raise ValueError("this atlas recipe requires MSDFgen 1.13.0")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    source_directory, output_directory = source_directory.resolve(), output_directory.resolve()
    source_bytes = [(source_directory/f"GLYPH-{index:03d}.svg").read_bytes()
                    for index in range(COUNT)]
    catalog_bytes = (PREVIEW / "glyphs.js").read_bytes()
    catalog = json.loads(catalog_bytes.decode().split("=", 1)[1].strip().removesuffix(";"))
    if len(catalog["glyphs"]) != COUNT:
        raise ValueError("expected 192 original browser glyphs")

    scratch_parent = (ROOT / "smythe/tmp").resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smythe-msdf-", dir=scratch_parent) as temporary:
        scratch = Path(temporary).resolve()
        if not scratch.is_relative_to(scratch_parent):
            raise ValueError("temporary MSDF output left its dedicated scratch directory")

        def build_one(index):
            glyph_id = f"GLYPH-{index:03d}"
            source = source_bytes[index].decode()
            combined = combined_svg(source)
            paths = [path.get("d") for path in ET.fromstring(source)]
            if (catalog["glyphs"][index]["glyph_id"] != glyph_id or
                    catalog["glyphs"][index]["paths"] != paths):
                raise ValueError(f"browser/source geometry differs for {glyph_id}")
            svg_path, png_path = scratch/f"{glyph_id}.svg", scratch/f"{glyph_id}.png"
            svg_path.write_text(combined, encoding="utf-8", newline="\n")
            run = subprocess.run(command(executable, svg_path, png_path), capture_output=True,
                                 text=True, check=True)
            with Image.open(png_path) as file:
                tile = file.convert("RGB")
            if tile.size != (CELL, CELL):
                raise ValueError(f"wrong distance-field dimensions for {glyph_id}")
            decoded = decode_mask(tile)
            truth = np.asarray(render_svg(source, CELL).convert("L")) < 128
            fidelity = mask_iou(decoded, truth)
            if not decoded.any() or fidelity < .94:
                raise ValueError(f"decoded MSDF/source mismatch for {glyph_id}: IoU={fidelity:.6f}")
            return tile, {
                "glyph_id": glyph_id,
                "cell": [index % COLUMNS, index // COLUMNS],
                "source_svg_sha256": sha256(source_bytes[index]),
                "combined_input_svg_sha256": sha256(combined.encode()),
                "tile_png_sha256": sha256(png_path.read_bytes()),
                "source_raster_iou_at128": fidelity,
                "ink_pixels_at128": int(decoded.sum()),
                "tool_stdout": run.stdout.strip(),
                "tool_stderr": run.stderr.strip(),
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            built = list(pool.map(build_one, range(COUNT)))

    atlas = Image.new("RGB", (COLUMNS*CELL, ROWS*CELL))
    for index, (tile, _) in enumerate(built):
        atlas.paste(tile, ((index % COLUMNS)*CELL, (index//COLUMNS)*CELL))
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / "generated-sdf.png"
    atlas.save(target)
    records = [record for _, record in built]
    tool_sha = sha256(executable.read_bytes())
    receipt = {
        "version": VERSION,
        "method": "MSDF from original vector paths; not a raster-derived single-channel SDF",
        "source_kind": "Smythe original generated SVG artwork",
        "count": COUNT,
        "canvas": [100, 100],
        "grid": [COLUMNS, ROWS],
        "cell_pixels": [CELL, CELL],
        "texture_pixels": list(atlas.size),
        "distance_range_pixels": RANGE,
        "distance_range_semantics": "full range 16 atlas pixels; -8 outside to +8 inside; boundary 0.5",
        "channels": "RGB multi-channel; reconstruct signed distance from median(r,g,b)",
        "color_space": "linear data; no sRGB conversion",
        "pixel_orientation": "top-left origin; y increases downward; no yflip applied",
        "order": "row-major GLYPH-000 through GLYPH-191",
        "source_svg_directory": source_directory.relative_to(ROOT).as_posix(),
        "browser_catalog_sha256": sha256(catalog_bytes),
        "source_catalog_sha256": catalog["catalog_sha256"],
        "atlas_sha256": sha256(target.read_bytes()),
        "exporter_sha256": sha256(Path(__file__).read_bytes()),
        "tool": {
            "version": tool_version,
            "executable_sha256": tool_sha,
            "official_release": TOOL_RELEASE,
            "verified_archive_url": TOOL_ARCHIVE if tool_sha == TOOL_EXE_SHA256 else None,
            "verified_archive_sha256": TOOL_ARCHIVE_SHA256 if tool_sha == TOOL_EXE_SHA256 else None,
            "license": "https://github.com/Chlumsky/msdfgen/blob/v1.13/LICENSE.txt",
            "distributed_with_preview": False,
            "arguments": command(Path("msdfgen"), Path("INPUT.svg"), Path("OUTPUT.png"))[1:],
        },
        "validation": {
            "all_cells_nonblank": True,
            "all_browser_paths_match_source_svgs": True,
            "comparison": "128px median-channel threshold versus original SVG rasterizer threshold 128",
            "minimum_iou_gate": .94,
            "minimum_iou": min(record["source_raster_iou_at128"] for record in records),
            "median_iou": float(np.median([record["source_raster_iou_at128"] for record in records])),
            "scope": "Derived-texture fidelity; not a new glyph or workflow benchmark",
        },
        "glyphs": records,
    }
    (output_directory / "generated-sdf.json").write_text(
        json.dumps(receipt, indent=2)+"\n", encoding="utf-8", newline="\n",
    )
    return receipt


def verify_atlas(output_directory: Path = PREVIEW) -> dict:
    receipt = json.loads((output_directory / "generated-sdf.json").read_text())
    expected = {"version": VERSION, "count": COUNT, "grid": [COLUMNS, ROWS],
                "cell_pixels": [CELL, CELL], "texture_pixels": [COLUMNS*CELL, ROWS*CELL],
                "distance_range_pixels": RANGE}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("atlas geometry or encoding contract changed")
    if receipt["exporter_sha256"] != sha256(Path(__file__).read_bytes()):
        raise ValueError("atlas exporter changed; regenerate the derived texture")
    if receipt["browser_catalog_sha256"] != sha256((PREVIEW / "glyphs.js").read_bytes()):
        raise ValueError("browser glyph catalog changed")
    texture = output_directory / "generated-sdf.png"
    if receipt["atlas_sha256"] != sha256(texture.read_bytes()):
        raise ValueError("atlas image changed")
    if len(receipt["glyphs"]) != COUNT:
        raise ValueError("incomplete atlas glyph receipt")
    with Image.open(texture) as atlas:
        if atlas.mode != "RGB" or atlas.size != (COLUMNS*CELL, ROWS*CELL):
            raise ValueError("atlas image format mismatch")
        for index, glyph in enumerate(receipt["glyphs"]):
            name = f"GLYPH-{index:03d}"
            column, row = index % COLUMNS, index // COLUMNS
            if glyph["glyph_id"] != name or glyph["cell"] != [column, row]:
                raise ValueError("atlas glyph order changed")
            if glyph["source_svg_sha256"] != sha256((SOURCE/f"{name}.svg").read_bytes()):
                raise ValueError(f"source SVG changed: {name}")
            tile = atlas.crop((column*CELL, row*CELL, (column+1)*CELL, (row+1)*CELL))
            if int(decode_mask(tile).sum()) != glyph["ink_pixels_at128"]:
                raise ValueError(f"decoded atlas mask changed: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msdfgen", type=Path)
    parser.add_argument("--out", type=Path, default=PREVIEW)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        receipt = verify_atlas(args.out)
    elif args.msdfgen:
        receipt = build_atlas(args.msdfgen, SOURCE, args.out, workers=args.workers)
    else:
        parser.error("choose --check or provide --msdfgen")
    print(json.dumps({key: receipt[key] for key in
                      ("version", "method", "count", "texture_pixels", "validation")}, indent=2))


if __name__ == "__main__":
    main()
