"""Publish an exact measured SVG catalog without regenerating its glyphs.

python benchmarks/publish_svg_catalog.py --source smythe_artifacts/svg_glyph_v1/runs/RUN \
  --destination benchmarks/partitions/glyph_svg_v1/catalog \
  --preview screensaver/svg-preview/glyphs.js
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.artifact_records import portable_path, resolve_record_path  # noqa: E402

COUNT = 192
VERSION = "glyph-svg-v1"
DEFAULT_DESTINATION = Path("benchmarks/partitions/glyph_svg_v1/catalog")
AUTHORING_FIELDS = ("authoring_parameters", "recipe", "attempt", "version", "authoring_weight")


def _api():
    return importlib.import_module("benchmarks.svg_glyphs")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _load_json(path: Path) -> tuple[dict, bytes]:
    content = path.read_bytes()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return parsed, content


def _within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _guard_output(source: Path, destination: Path, preview: Path, overwrite: bool) -> None:
    if _within(destination, source) or _within(source, destination) or _within(preview, source):
        raise ValueError("Publication outputs must not overlap the measured source directory")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Destination must be a directory")
    if not overwrite and ((destination.exists() and any(destination.iterdir())) or preview.exists()):
        raise FileExistsError("Publication output exists; choose a new destination or pass --overwrite")
    reserved = {destination / name for name in (
        "manifest.json", "contact-sheet.png", "calibration-sheet.png",
        "style-measurements.json", "style-acceptance.json",
    )}
    reserved.update(destination / f"GLYPH-{index:03d}{suffix}"
                    for index in range(COUNT) for suffix in (".svg", ".png"))
    if preview in reserved or preview == destination:
        raise ValueError("Preview output collides with a canonical catalog artifact")


def _read_source(source: Path) -> tuple[list[dict], dict, dict]:
    """Verify every source artifact and raster before creating destination files."""
    from PIL import Image

    manifest, manifest_bytes = _load_json(source / "catalog.json")
    validation, validation_bytes = _load_json(source / "validation.json")
    records = manifest.get("glyphs")
    receipts = validation.get("glyphs")
    if manifest.get("glyph_count") != COUNT or not isinstance(records, list) or len(records) != COUNT:
        raise ValueError("Publication requires exactly 192 measured glyphs")
    if not isinstance(receipts, list) or len(receipts) != COUNT:
        raise ValueError("Validation report must contain all 192 glyph receipts")
    indices = [r.get("index") for r in records]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in indices) or set(indices) != set(range(COUNT)):
        raise ValueError("Catalog must contain each glyph index 0–191 exactly once")
    receipt_indices = [r.get("index") for r in receipts]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in receipt_indices) or set(receipt_indices) != set(range(COUNT)):
        raise ValueError("Validation must contain each glyph index 0–191 exactly once")
    by_index = {r["index"]: r for r in receipts}
    style = validation.get("catalog_style")
    if not isinstance(style, dict) or style.get("count") != COUNT:
        raise ValueError("Missing complete 192-glyph style report")
    items = []
    for record in sorted(records, key=lambda r: r["index"]):
        index = record["index"]
        glyph_id = f"GLYPH-{index:03d}"
        if record.get("glyph_id") != glyph_id:
            raise ValueError(f"Expected canonical glyph ID {glyph_id}")
        if record.get("file") != f"glyph-{index:03d}.svg":
            raise ValueError(f"Unexpected measured SVG filename for {glyph_id}")
        for key in ("family", "profile"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Missing {key} for {glyph_id}")
        if isinstance(record.get("seed"), bool) or not isinstance(record.get("seed"), int):
            raise ValueError(f"Invalid seed for {glyph_id}")
        svg_path = source / record["file"]
        if not _within(svg_path.resolve(), source):
            raise ValueError(f"Source SVG escapes the measured directory: {glyph_id}")
        svg_bytes = svg_path.read_bytes()
        if _hash(svg_bytes) != record.get("svg_sha256"):
            raise ValueError(f"SVG hash mismatch: {glyph_id}")
        svg = svg_bytes.decode("utf-8")
        structural = _api().validate_svg(svg)
        if structural.get("passed") is not True or structural.get("errors") != []:
            raise ValueError(f"Invalid measured SVG {glyph_id}: {structural}")
        image = _api().render_svg(svg, size=128).convert("RGBA")
        if image.size != (128, 128) or image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Expected an opaque 128px raster: {glyph_id}")
        pixel_hash = _hash(image.tobytes())
        if pixel_hash != record.get("pixel_sha256"):
            raise ValueError(f"Raster hash mismatch: {glyph_id}")
        receipt = by_index[index]
        if receipt.get("status") != "passed":
            raise ValueError(f"Validation receipt did not pass: {glyph_id}")
        for key in ("glyph_id", "family", "profile", "seed", "svg_sha256", "pixel_sha256"):
            if receipt.get(key) != record[key]:
                raise ValueError(f"Validation {key} mismatch: {glyph_id}")
        for key in AUTHORING_FIELDS:
            if receipt.get(key) != record.get(key):
                raise ValueError(f"Validation {key} mismatch: {glyph_id}")
        measurements = receipt.get("measurements")
        if not isinstance(measurements, dict) or measurements.get("blank") is not False:
            raise ValueError(f"Missing or blank measured silhouette: {glyph_id}")
        if measurements.get("pixel_sha256") != pixel_hash:
            raise ValueError(f"Measurement raster hash mismatch: {glyph_id}")
        root = ET.fromstring(svg)
        if root.attrib.get("viewBox") != "0 0 100 100":
            raise ValueError(f"Unsupported source viewBox: {glyph_id}")
        paths = [element.attrib["d"] for element in root]
        if not paths:
            raise ValueError(f"Missing filled SVG paths: {glyph_id}")
        items.append({**record, "svg_bytes": svg_bytes, "image": image,
                      "paths": paths, "measurements": measurements})
    for field in ("svg_sha256", "pixel_sha256"):
        if len({r[field] for r in items}) != COUNT:
            raise ValueError(f"Duplicate {field} in measured catalog")
    # The measured atlas is part of the authoritative run, not a substitute for
    # re-rasterizing each source SVG and checking its recorded pixels above.
    atlas_bytes = (source / "atlas.png").read_bytes()
    atlas = Image.open(io.BytesIO(atlas_bytes)).convert("RGBA")
    expected = Image.new("RGBA", (16 * 128, 12 * 128), "white")
    for index, record in enumerate(items):
        expected.paste(record["image"], ((index % 16) * 128, (index // 16) * 128))
    if atlas.size != expected.size or atlas.tobytes() != expected.tobytes():
        raise ValueError("Measured atlas pixels do not match the 192 source SVG rasters")
    provenance = {"kind": "measured-smythe-benchmark-run", "directory": portable_path(source),
                  "catalog": portable_path(source / "catalog.json"),
                  "validation": portable_path(source / "validation.json"),
                  "atlas": portable_path(source / "atlas.png"), "catalog_sha256": _hash(manifest_bytes),
                  "validation_sha256": _hash(validation_bytes), "atlas_sha256": _hash(atlas_bytes)}
    return items, style, provenance


def _bind_results(results_path: Path, provenance: dict) -> dict:
    """Bind publication to one completed run in an optional campaign record."""
    results, results_bytes = _load_json(results_path)
    matches = []
    for run in results.get("runs", []):
        assembly = run.get("assembly") or {}
        if assembly.get("catalog") and resolve_record_path(assembly["catalog"]).resolve() == resolve_record_path(provenance["catalog"]).resolve():
            matches.append(run)
    if len(matches) != 1:
        raise ValueError("Result record must identify exactly one selected source run")
    run = matches[0]
    if run.get("status") != "passed" or run.get("valid_glyphs") != COUNT:
        raise ValueError("Selected result run must have passed with all 192 valid glyphs")
    for field in ("catalog", "validation", "atlas"):
        assembly = run["assembly"]
        if not assembly.get(field) or resolve_record_path(assembly[field]).resolve() != resolve_record_path(provenance[field]).resolve():
            raise ValueError(f"Result artifact path mismatch: {field}")
        if assembly.get(field + "_sha256") != provenance[field + "_sha256"]:
            raise ValueError(f"Result artifact hash mismatch: {field}")
    return {"file": portable_path(results_path), "sha256": _hash(results_bytes),
            "protocol_version": results.get("protocol_version"),
            "timing_claimable": results.get("timing_claimable") is True,
            "readme_promotion_eligible": results.get("readme_promotion_eligible") is True,
            "run": {key: run.get(key) for key in ("executor", "concurrency", "repeat", "execution_id")}}


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _contact_sheet(items: list[dict], *, columns: int):
    from PIL import Image, ImageDraw, ImageFont

    width, row_height = 144, 164
    sheet = Image.new("RGB", (columns * width, math.ceil(len(items) / columns) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow10.0 did not accept a size argument.
        font = ImageFont.load_default()
    for index, record in enumerate(items):
        x, y = (index % columns) * width + 8, (index // columns) * row_height + 4
        sheet.paste(record["image"].convert("RGB"), (x, y))
        label = record["glyph_id"]
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (128 - (box[2] - box[0])) / 2, y + 133), label, fill="black", font=font)
    return sheet


def _write_bytes(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("xb") as handle:
            handle.write(content)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_catalog(*, source: Path, destination: Path = DEFAULT_DESTINATION,
                    preview: Path, individual_pngs: bool = False, overwrite: bool = False,
                    results_path: Path | None = None) -> dict:
    """Copy exact measured SVGs, produce review sheets, and bind the browser data."""
    source, destination, preview = (Path(p).resolve() for p in (source, destination, preview))
    _guard_output(source, destination, preview, overwrite)
    if results_path is not None:
        results_path = Path(results_path).resolve()
        if _within(results_path, destination) or results_path == preview:
            raise ValueError("Publication outputs must not overlap the source result record")
    items, style, provenance = _read_source(source)
    if results_path is not None:
        provenance["campaign"] = _bind_results(results_path, provenance)
    artifacts = {
        "contact-sheet.png": _png_bytes(_contact_sheet(items, columns=16)),
        "calibration-sheet.png": _png_bytes(_contact_sheet(items[:24], columns=8)),
        "style-measurements.json": _json_bytes({"count": COUNT, "glyphs": [
            {"glyph_id": r["glyph_id"], "family": r["family"], "profile": r["profile"],
             "seed": r["seed"], "svg_sha256": r["svg_sha256"], "measurements": r["measurements"]}
            for r in items]}),
        "style-acceptance.json": _json_bytes(style),
    }
    entries = []
    for record in items:
        name = record["glyph_id"] + ".svg"
        artifacts[name] = record["svg_bytes"]
        entry = {key: record[key] for key in (
            "index", "glyph_id", "family", "profile", "seed", "svg_sha256", "pixel_sha256",
        ) + AUTHORING_FIELDS if key in record} | {"file": name}
        if individual_pngs:
            png_name = record["glyph_id"] + ".png"
            artifacts[png_name] = _png_bytes(record["image"])
            entry.update(png_file=png_name, png_sha256=_hash(artifacts[png_name]))
        entries.append(entry)
    manifest = {
        "version": VERSION, "count": COUNT, "canvas": [100, 100],
        "method": "exact publication of measured SVG sources; no regeneration",
        "reference_artwork_included": False, "source_artifacts": provenance,
        "catalog_style_accepted": style.get("accepted") is True,
        "glyphs": entries,
        "review_artifacts": {name: {"sha256": _hash(artifacts[name]), "bytes": len(artifacts[name])}
                             for name in ("contact-sheet.png", "calibration-sheet.png",
                                          "style-measurements.json", "style-acceptance.json")},
    }
    manifest_bytes = _json_bytes(manifest)
    manifest_hash = _hash(manifest_bytes)
    payload = {"version": VERSION, "canvas": [100, 100], "catalog_sha256": manifest_hash,
               "glyphs": [{key: record[key] for key in ("glyph_id", "family", "profile", "paths")}
                          for record in items]}
    js_bytes = ("// Exact measured SVG paths; source catalog hash is part of the payload.\n"
                "globalThis.SVG_GLYPHS=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
                + ";\n").encode("utf-8")
    # All source verification and derived asset construction precedes writes.
    for name, content in artifacts.items():
        _write_bytes(destination / name, content, overwrite=overwrite)
    _write_bytes(destination / "manifest.json", manifest_bytes, overwrite=overwrite)
    _write_bytes(preview, js_bytes, overwrite=overwrite)
    for name, content in {**artifacts, "manifest.json": manifest_bytes}.items():
        if _hash((destination / name).read_bytes()) != _hash(content):
            raise OSError(f"Publication readback failed: {name}")
    if preview.read_bytes() != js_bytes:
        raise OSError("Preview data readback failed")
    return {"status": "published", "count": COUNT, "destination": str(destination),
            "manifest_sha256": manifest_hash, "preview": str(preview),
            "preview_sha256": _hash(js_bytes), "catalog_style_accepted": manifest["catalog_style_accepted"],
            "individual_pngs": individual_pngs, "generation_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="completed benchmark run artifact directory")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--preview", type=Path, required=True, help="explicit browser glyphs.js output path")
    parser.add_argument("--results", type=Path, help="bind the selected run to this aggregate benchmark record")
    parser.add_argument("--individual-pngs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = publish_catalog(source=args.source, destination=args.destination, preview=args.preview,
                                 individual_pngs=args.individual_pngs, overwrite=args.overwrite,
                                 results_path=args.results)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
