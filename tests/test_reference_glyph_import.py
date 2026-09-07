"""Reference artwork import preserves pinned geometry, spacing, and provenance."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from screensaver.import_reference_glyphs import (
    COMMIT, DEFAULT_OUT, LICENSE_SHA256, SCALE, SVG_SHA256,
    _normalized_path, build_catalog, export_files, parse_contours, read_inputs, render_glyph,
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog(*read_inputs(DEFAULT_OUT))


def test_pinned_sources_and_all_visible_cells_are_present(catalog):
    svg, license_text = read_inputs(DEFAULT_OUT)
    assert hashlib.sha256(svg).hexdigest() == SVG_SHA256
    assert hashlib.sha256(license_text).hexdigest() == LICENSE_SHA256
    assert b"Copyright (c) 2018 Rezmason" in license_text
    assert catalog["source"]["commit"] == COMMIT
    assert catalog["count"] == 56
    assert catalog["canvas"] == [100, 100]
    assert catalog["fill_rule"] == "nonzero"
    assert catalog["source"]["blank_sequence_indices"] == [4]
    assert catalog["source"]["sequence_length"] == 57
    assert catalog["source"]["unused_sequence_indices"] == list(range(57, 64))
    assert {g["source_sequence_index"] for g in catalog["glyphs"]} == set(range(57))-{4}
    assert [g["glyph_id"] for g in catalog["glyphs"]] == [f"BASE-{i:03d}" for i in range(56)]
    assert sum(g["contour_count"] for g in catalog["glyphs"]) == 84


def test_normalized_geometry_round_trips_every_source_control_exactly(catalog):
    source = ET.fromstring(read_inputs(DEFAULT_OUT)[0])[0].get("d")
    originals = parse_contours(source)
    grouped = {}
    for contour in originals:
        x, y = contour[0][1:]
        grouped.setdefault((int(x//64), int(y//64)), []).append(contour)
    for glyph in catalog["glyphs"]:
        column, row = glyph["source_cell"]
        expected = grouped[(column, row)]
        normalized = parse_contours(" ".join(glyph["paths"]))
        assert len(normalized) == len(expected)
        for source_contour, exported_contour in zip(expected, normalized):
            assert len(source_contour) == len(exported_contour)
            for original, exported in zip(source_contour, exported_contour):
                assert original[0] == exported[0]
                restored = [value/SCALE+64*(column if i % 2 == 0 else row)
                            for i, value in enumerate(exported[1:])]
                assert tuple(restored) == original[1:]


def test_relative_moves_repeated_coordinates_and_close_follow_svg_semantics():
    assert parse_contours("m10 10 5 0 h5 v10 l-5 0 z m2 3 l1 0") == [
        [("M", 10, 10), ("L", 15, 10), ("L", 20, 10),
         ("L", 20, 20), ("L", 15, 20), ("Z",)],
        [("M", 12, 13), ("L", 13, 13), ("Z",)],
    ]


def test_smooth_cubic_reflects_only_a_preceding_cubic_control():
    assert parse_contours("M10 10 c2 0 4 2 6 2 s4 2 6 0 l2 0 s4 2 6 0Z") == [[
        ("M", 10, 10), ("C", 12, 10, 14, 12, 16, 12),
        ("C", 18, 12, 20, 14, 22, 12), ("L", 24, 12),
        ("C", 24, 12, 28, 14, 30, 12), ("Z",),
    ]]


def test_normalization_is_cell_based_not_ink_bounding_box_fit():
    path = parse_contours("M72 136h16v32h-16z")
    assert _normalized_path(path, 1, 2) == "M12.5 12.5 L37.5 12.5 L37.5 62.5 L12.5 62.5 Z"


def test_bezier_control_can_leave_cell_while_curve_stays_inside():
    contours = parse_contours("M10 60 C20 65 30 60 40 60 L40 50 L10 50Z")
    result = _normalized_path(contours, 0, 0)
    assert "101.5625" in result  # Retain the control; the curve itself is below64.
    with pytest.raises(ValueError, match="crosses"):
        _normalized_path(parse_contours("M10 60 C20 80 30 80 40 60Z"), 0, 0)


@pytest.mark.parametrize("path", ["", "M0", "L0 0", "M0 0Q1 2 3 4", "M0 0!L1 1", "M0 0Zz", "M0 0C1 2"])
def test_parser_fails_on_incomplete_or_unsupported_geometry(path):
    with pytest.raises(ValueError):
        parse_contours(path)


def test_source_or_license_changes_fail_closed():
    svg, license_text = read_inputs(DEFAULT_OUT)
    with pytest.raises(ValueError, match="source SVG"):
        build_catalog(svg.replace(b"23.1", b"23.2", 1), license_text)
    with pytest.raises(ValueError, match="license"):
        build_catalog(svg, license_text.replace(b"2018", b"2026"))


def test_reference_counters_and_detached_marks_survive_inspection_raster(catalog):
    eight = render_glyph(catalog["glyphs"][45], size=128)
    assert eight.getpixel((64, 32))[0] > 240
    assert eight.getpixel((64, 95))[0] > 240
    assert eight.getpixel((64, 63))[0] < 15
    colon = render_glyph(catalog["glyphs"][6], size=128)
    assert colon.getpixel((64, 33))[0] < 15
    assert colon.getpixel((64, 78))[0] > 240
    assert colon.getpixel((64, 92))[0] < 15


def test_exports_match_checked_in_files_and_keep_original_catalog_separate():
    files = export_files(*read_inputs(DEFAULT_OUT))
    assert len(files) == 63
    assert "glyphs.js" not in files
    assert not any("GLYPH-" in name for name in files)
    for name, expected in files.items():
        assert (DEFAULT_OUT / name).read_bytes() == expected, name
    js = files["base-glyphs.js"].decode().split("globalThis.BASE_GLYPHS = ", 1)[1]
    assert json.loads(js.removesuffix(";\n")) == json.loads(files["reference/catalog.json"])
    provenance = json.loads(files["reference/provenance.json"])
    assert "Path of Neo" in provenance["artwork_origin_as_stated_upstream"]
    assert "Susan Kare" in provenance["artwork_origin_as_stated_upstream"]
    assert "not a separate" in provenance["rights_scope"]
    assert provenance["transformation"]["upstream_implementation_code_copied"] is False


def test_cli_check_is_read_only_and_detects_changed_exports(tmp_path):
    files = export_files(*read_inputs(DEFAULT_OUT))
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    changed = tmp_path / "base-glyphs.js"
    changed.write_text("changed", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                                 "screensaver/import_reference_glyphs.py"),
               "--out", str(tmp_path), "--check"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "base-glyphs.js" in result.stderr
    assert changed.read_text() == "changed"
