"""Linux native catalog parity and executable rendering checks."""

from __future__ import annotations

import importlib.util
import ctypes.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from PIL import Image, ImageDraw

from screensaver.export_native_glyphs import build_payload, render_outputs
from screensaver.linux.smoke_linux import (
    compare_glyph_sheet, inspect_catalog, selection_counts, source_sheet_svg,
)


ROOT = Path(__file__).resolve().parents[1]


def test_linux_catalog_exports_exact_mixed_svg_contours():
    data = build_payload()
    header = ROOT / "screensaver/linux/glyph_data.h"
    assert header.read_bytes() == render_outputs()["screensaver/linux/glyph_data.h"]
    assert data["count"] == 249
    assert data["reference_count"] == 57
    assert data["original_count"] == 192
    assert data["blank_index"] == 4
    assert data["canvas"] == [100, 100]
    assert data["fill_rule"] == "nonzero"


def _catalog():
    receipt = json.loads((ROOT / "screensaver/native-catalog.json").read_text())
    return {
        "version": receipt["version"], "glyph_count": 249, "reference_count": 57,
        "reference_visible_count": 56, "original_count": 192, "original_offset": 57,
        "blank_index": 4, "canvas": [100, 100], "fill_rule": "nonzero",
        "default_original_mix": 0.1, "catalog_sha256": receipt["catalog_sha256"],
        "reference_sha256": hashlib.sha256((ROOT / "screensaver/svg-preview/reference/catalog.json").read_bytes()).hexdigest(),
        "original_sha256": hashlib.sha256((ROOT / "benchmarks/partitions/glyph_svg_v1/catalog/manifest.json").read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("field,value", [("glyph_count", 192), ("blank_index", 5),
                                       ("original_count", 191), ("fill_rule", "evenodd"),
                                       ("catalog_sha256", "bad"), ("reference_sha256", "bad"),
                                       ("original_sha256", "bad"), ("default_original_mix", .77)])
def test_catalog_rejects_stale_counts_wrong_fill_or_unbound_sources(field, value):
    catalog = _catalog()
    assert inspect_catalog(catalog)["catalog_sha256"] == catalog["catalog_sha256"]
    with pytest.raises(AssertionError, match="Compiled catalog mismatch"):
        inspect_catalog({**catalog, field: value})


def _shape_sheet(path):
    image = Image.new("L", (2048, 2048), 255)
    draw = ImageDraw.Draw(image)
    for index in range(249):
        if index != 4:
            x, y = index % 16 * 128, index // 16 * 128
            draw.rectangle((x+10, y+10, x+90, y+80), fill=0)
            draw.rectangle((x+20, y+20, x+40, y+40), fill=255)
    image.save(path)
    return image


@pytest.mark.parametrize("defect", ["missing", "counter_filled", "blank_filled", "shifted", "wrong_size"])
def test_source_silhouette_check_rejects_geometry_defects(tmp_path, defect):
    source, actual = tmp_path / "source.png", tmp_path / "actual.png"
    image = _shape_sheet(source)
    draw = ImageDraw.Draw(image)
    if defect == "missing":
        draw.rectangle((0, 0, 127, 127), fill=255)
    elif defect == "counter_filled":
        draw.rectangle((20, 20, 40, 40), fill=0)
    elif defect == "blank_filled":
        draw.rectangle((4*128+10, 10, 4*128+20, 20), fill=0)
    elif defect == "shifted":
        tile = image.crop((0, 0, 128, 128))
        draw.rectangle((0, 0, 127, 127), fill=255)
        image.paste(tile, (4, 0))
    else:
        image = image.crop((0, 0, 1024, 1024))
    image.save(actual)
    with pytest.raises(AssertionError):
        compare_glyph_sheet(actual, source)


def test_source_silhouette_check_covers_all_slots_and_both_families(tmp_path):
    path = tmp_path / "source.png"
    _shape_sheet(path)
    result = compare_glyph_sheet(path, path)
    assert result["glyph_count"] == len(result["glyphs"]) == 249
    assert result["minimum_measured_iou"] == 1
    assert result["glyphs"][4]["ink_pixels"] == 0
    assert result["glyphs"][57]["family"] == "original"


def test_source_silhouette_check_rejects_invisible_rgb_geometry(tmp_path):
    source, actual = tmp_path / "source.png", tmp_path / "transparent.png"
    image = _shape_sheet(source).convert("RGBA")
    image.putalpha(0)
    image.save(actual)
    with pytest.raises(AssertionError, match="must be opaque"):
        compare_glyph_sheet(actual, source)


@pytest.mark.parametrize("stdout", ["", "selected_reference=0 selected_original=0 selected_blank=0 mix=0.100000",
                                   "selected_reference=900 selected_original=100 selected_blank=901 mix=0.100000",
                                   "selected_reference=570 selected_original=1920 selected_blank=10 mix=0.100000",
                                   "selected_reference=1 selected_original=900 selected_blank=0 mix=1.000000",
                                   "selected_reference=900 selected_original=1 selected_blank=0 mix=0.000000"])
def test_selection_receipt_rejects_missing_invalid_or_size_weighted_mix(stdout):
    with pytest.raises(AssertionError):
        selection_counts(stdout)


def test_source_oracle_uses_published_svg_paths_not_generated_c_header():
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    sheet = ET.fromstring(source_sheet_svg())
    groups = sheet.findall("svg:g", namespace)
    assert len(groups) == 248
    assert not any(glyph.get("transform").startswith("translate(516 4)") for glyph in groups)
    first = json.loads((ROOT / "screensaver/svg-preview/reference/catalog.json").read_text())["glyphs"][0]
    assert groups[0][0].get("d") == " ".join(first["paths"])


def test_source_oracle_rejects_an_altered_published_svg(monkeypatch):
    original_read = Path.read_bytes

    def read_bytes(path):
        data = original_read(path)
        return data + b"\n" if path.name == "GLYPH-000.svg" else data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(AssertionError, match="Original source SVG hash mismatch"):
        source_sheet_svg()


def test_cross_platform_white_ink_64px_comparison_and_polarity(tmp_path):
    image = Image.new("L", (1024, 1024), 0)
    draw = ImageDraw.Draw(image)
    for index in range(249):
        if index != 4:
            x, y = index % 16 * 64, index // 16 * 64
            draw.rectangle((x+5, y+5, x+40, y+40), fill=255)
    path = tmp_path / "white-on-black.png"
    image.save(path)
    receipt = compare_glyph_sheet(path, path, tile_size=64, margin=0, white_ink=True)
    assert receipt["minimum_measured_iou"] == 1
    assert receipt["white_ink"] and receipt["inner_margin"] == 0
    with pytest.raises(AssertionError, match="blank slot"):
        compare_glyph_sheet(path, path, tile_size=64, margin=0)


@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux executable test")
def test_linux_elf_renders_and_embeds_under_xvfb(tmp_path):
    for command in ("cc", "pkg-config", "xvfb-run"):
        if shutil.which(command) is None:
            pytest.skip(f"Linux native smoke needs {command}")
    dependencies = subprocess.run(["pkg-config", "--exists", "x11", "cairo"], check=False)
    if dependencies.returncode:
        pytest.skip("Linux native smoke needs X11/Cairo development packages")
    if importlib.util.find_spec("PIL") is None:
        pytest.skip("Linux native smoke needs Pillow")
    if not ctypes.util.find_library("rsvg-2"):
        pytest.skip("Linux native source-SVG verification needs librsvg2-2")
    binary = tmp_path / "smythe-glyph-rain"
    subprocess.run(["sh", str(ROOT / "screensaver/linux/build_linux.sh"), str(binary)],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", sys.executable,
                    str(ROOT / "screensaver/linux/smoke_linux.py"), str(binary),
                    "--out", str(tmp_path / "verification")],
                   check=True, capture_output=True, text=True, timeout=90)
