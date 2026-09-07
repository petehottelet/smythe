"""Canonical publication checks use measured files; they never regenerate glyphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw
import pytest

from benchmarks import publish_svg_catalog as publisher


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture
def measured_source(tmp_path, monkeypatch):
    source = tmp_path / "measured-run"
    source.mkdir()
    rendered_calls = []

    def render(svg, size=128):
        root = ET.fromstring(svg)
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", list(root)[0].attrib["d"])]
        x0, y0, x1, _, _, y1, _, _ = values
        image = Image.new("RGBA", (size, size), "white")
        scale = size / 100
        ImageDraw.Draw(image).rectangle((x0 * scale, y0 * scale, x1 * scale, y1 * scale), fill="black")
        rendered_calls.append(digest(svg.encode()))
        return image

    def never_generate(*args, **kwargs):
        raise AssertionError("Publishing must not regenerate glyphs")

    def validate(svg):
        root = ET.fromstring(svg)
        return {"passed": root.attrib.get("viewBox") == "0 0 100 100", "errors": []}

    monkeypatch.setattr(publisher, "_api", lambda: SimpleNamespace(
        generate_glyph=never_generate, render_svg=render, validate_svg=validate))
    records, receipts = [], []
    atlas = Image.new("RGBA", (16 * 128, 12 * 128), "white")
    for index in range(192):
        width, height = 18 + index % 16, 20 + 2 * (index // 16)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="128" height="128">'
               f'<path fill="#000000" fill-rule="nonzero" d="M20 20 L{20 + width} 20 L{20 + width} {20 + height} L20 {20 + height} Z"/>'
               '</svg>\n')
        image = render(svg)
        filename = f"glyph-{index:03d}.svg"
        (source / filename).write_bytes(svg.encode())
        record = {"index": index, "glyph_id": f"GLYPH-{index:03d}", "family": "numeral_operator",
                  "profile": "classic", "seed": 8000 + index, "file": filename,
                  "authoring_parameters": {"fixedweight": 18, "structure": index, "seed_offset": 0},
                  "recipe": "test rectangle", "attempt": 0, "version": "fixture-v1", "authoring_weight": 18,
                  "svg_sha256": digest(svg.encode()), "pixel_sha256": digest(image.tobytes())}
        records.append(record)
        receipts.append({**record, "status": "passed", "measurements": {
            "blank": False, "pixel_sha256": record["pixel_sha256"], "ink_pixels": width * height,
        }})
        atlas.paste(image, ((index % 16) * 128, (index // 16) * 128))
    save_json(source / "catalog.json", {"glyph_count": 192, "glyphs": records})
    save_json(source / "validation.json", {"glyphs": receipts, "catalog_style": {
        "count": 192, "accepted": False, "failed_gates": ["test-fixture.unreviewed"],
    }})
    atlas.save(source / "atlas.png")
    rendered_calls.clear()
    return source, rendered_calls


def publish(source, tmp_path, **options):
    return publisher.publish_catalog(source=source, destination=tmp_path / "canonical",
                                     preview=tmp_path / "preview/glyphs.js", **options)


def test_measured_bytes_preserved_and_preview_bound_to_manifest(measured_source, tmp_path):
    source, calls = measured_source
    result = publish(source, tmp_path)
    destination = Path(result["destination"])
    manifest_bytes = (destination / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert result["generation_calls"] == 0
    assert len(calls) == 192
    assert manifest["count"] == 192
    assert manifest["catalog_style_accepted"] is False
    assert manifest["source_artifacts"]["directory"] == source.as_posix()
    assert manifest["source_artifacts"]["catalog_sha256"] == digest((source / "catalog.json").read_bytes())
    assert b"\r\n" not in manifest_bytes
    assert b"\r\n" not in Path(result["preview"]).read_bytes()
    assert len(list(destination.glob("GLYPH-*.svg"))) == 192
    assert len(list(destination.glob("GLYPH-*.png"))) == 0
    for index, entry in enumerate(manifest["glyphs"]):
        assert entry["file"] == f"GLYPH-{index:03d}.svg"
        assert (destination / entry["file"]).read_bytes() == (source / f"glyph-{index:03d}.svg").read_bytes()
        assert entry["authoring_parameters"] == {"fixedweight": 18, "structure": index, "seed_offset": 0}
        assert entry["recipe"] == "test rectangle"
        assert entry["attempt"] == 0
        assert entry["version"] == "fixture-v1"
        assert entry["authoring_weight"] == 18
    with Image.open(destination / "contact-sheet.png") as image:
        assert image.size == (2304, 1968)
        assert image.convert("L").crop((8, 137, 136, 161)).getextrema()[0] < 128  # Visible label.
    with Image.open(destination / "calibration-sheet.png") as image:
        assert image.size == (1152, 492)
    javascript = Path(result["preview"]).read_text(encoding="utf-8")
    payload = json.loads(javascript.split("globalThis.SVG_GLYPHS=", 1)[1].removesuffix(";\n"))
    assert payload["catalog_sha256"] == digest(manifest_bytes) == result["manifest_sha256"]
    assert payload["version"] == "glyph-svg-v1"
    assert payload["canvas"] == [100, 100]
    assert len(payload["glyphs"]) == 192
    assert payload["glyphs"][0]["paths"][0].startswith("M20 20")
    acceptance = json.loads((destination / "style-acceptance.json").read_text())
    assert acceptance["accepted"] is False
    assert acceptance["failed_gates"] == ["test-fixture.unreviewed"]
    measurements = json.loads((destination / "style-measurements.json").read_text())
    assert len(measurements["glyphs"]) == 192
    for filename, receipt in manifest["review_artifacts"].items():
        assert digest((destination / filename).read_bytes()) == receipt["sha256"]


@pytest.mark.parametrize("damage,match", [
    ("missing", None), ("svg_hash", "SVG hash mismatch"),
    ("raster_hash", "Raster hash mismatch"), ("count", "exactly 192"),
    ("validation_identity", "Validation seed mismatch"),
    ("authoring", "Validation recipe mismatch"),
    ("measurement_hash", "Measurement raster hash mismatch"),
    ("atlas", "atlas pixels"),
])
def test_bad_source_fails_before_publication(measured_source, tmp_path, damage, match):
    source, _ = measured_source
    if damage == "missing":
        (source / "glyph-001.svg").unlink()
    elif damage == "svg_hash":
        with (source / "glyph-001.svg").open("ab") as file:
            file.write(b"\n")
    elif damage in {"raster_hash", "count"}:
        path = source / "catalog.json"
        data = json.loads(path.read_text())
        if damage == "count":
            data["glyph_count"] = 191
        else:
            data["glyphs"][1]["pixel_sha256"] = "0" * 64
        save_json(path, data)
    elif damage in {"validation_identity", "measurement_hash", "authoring"}:
        path = source / "validation.json"
        data = json.loads(path.read_text())
        if damage == "validation_identity":
            data["glyphs"][1]["seed"] += 1
        elif damage == "authoring":
            data["glyphs"][1]["recipe"] = "different recipe"
        else:
            data["glyphs"][1]["measurements"]["pixel_sha256"] = "0" * 64
        save_json(path, data)
    else:
        Image.new("RGBA", (2048, 1536), "white").save(source / "atlas.png")
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        publish(source, tmp_path)
    assert not (tmp_path / "canonical").exists()
    assert not (tmp_path / "preview/glyphs.js").exists()


def test_overwrite_guard_and_repeat_publication_are_deterministic(measured_source, tmp_path):
    source, calls = measured_source
    first = publish(source, tmp_path)
    calls.clear()
    with pytest.raises(FileExistsError, match="--overwrite"):
        publish(source, tmp_path)
    assert calls == []
    unrelated = tmp_path / "canonical/keep.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    second = publish(source, tmp_path, overwrite=True)
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert second["preview_sha256"] == first["preview_sha256"]
    assert unrelated.read_text() == "unrelated"


def test_explicit_individual_pngs_match_recorded_pixels(measured_source, tmp_path):
    source, _ = measured_source
    result = publish(source, tmp_path, individual_pngs=True)
    destination = Path(result["destination"])
    assert len(list(destination.glob("GLYPH-*.png"))) == 192
    manifest = json.loads((destination / "manifest.json").read_text())
    for entry in manifest["glyphs"]:
        path = destination / entry["png_file"]
        assert digest(path.read_bytes()) == entry["png_sha256"]
        with Image.open(path) as image:
            assert digest(image.convert("RGBA").tobytes()) == entry["pixel_sha256"]


@pytest.mark.parametrize("target", ["source", "manifest", "results"])
def test_publication_paths_cannot_clobber_sources_or_manifest(measured_source, tmp_path, target):
    source, calls = measured_source
    destination = source if target == "source" else tmp_path / "canonical"
    preview = tmp_path / "preview.js" if target == "source" else destination / "manifest.json"
    results_path = preview if target == "results" else None
    if target == "results":
        preview = tmp_path / "results.json"
        results_path = preview
    with pytest.raises(ValueError, match="overlap|collides"):
        publisher.publish_catalog(source=source, destination=destination, preview=preview, overwrite=True,
                                  results_path=results_path)
    assert calls == []


@pytest.mark.parametrize("damage", [None, "hash", "missing_run", "failed_run"])
def test_campaign_binding_checks_selected_run(measured_source, tmp_path, damage):
    source, _ = measured_source
    assembly = {}
    for field, filename in (("catalog", "catalog.json"), ("validation", "validation.json"), ("atlas", "atlas.png")):
        assembly[field] = (source / filename).as_posix()
        assembly[field + "_sha256"] = digest((source / filename).read_bytes())
    run = {"executor": "thread", "concurrency": 4, "repeat": 2, "execution_id": "measured-test-run",
           "status": "passed", "valid_glyphs": 192, "assembly": assembly}
    if damage == "hash":
        assembly["validation_sha256"] = "0" * 64
    if damage == "failed_run":
        run["status"] = "failed"
    record = {"protocol_version": "original-svg-workflow-v1", "timing_claimable": True,
              "readme_promotion_eligible": False, "runs": [] if damage == "missing_run" else [run]}
    results_path = tmp_path / "results.json"
    save_json(results_path, record)
    if damage:
        with pytest.raises(ValueError, match="Result artifact hash|exactly one|must have passed"):
            publish(source, tmp_path, results_path=results_path)
        assert not (tmp_path / "canonical").exists()
        return
    published = publish(source, tmp_path, results_path=results_path)
    manifest = json.loads((Path(published["destination"]) / "manifest.json").read_bytes())
    campaign = manifest["source_artifacts"]["campaign"]
    assert campaign["file"] == results_path.as_posix()
    assert campaign["sha256"] == digest(results_path.read_bytes())
    assert campaign["run"] == {key: run[key] for key in ("executor", "concurrency", "repeat", "execution_id")}
    assert campaign["timing_claimable"] is True
    assert campaign["readme_promotion_eligible"] is False
