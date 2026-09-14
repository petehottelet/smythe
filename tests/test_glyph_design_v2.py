"""Regression coverage for the current clean-contour catalog."""

import hashlib
import json

import numpy as np
import pytest
from shapely import LineString, Point, box

from benchmarks.svg_glyph_measurements import find_near_matches, measure_glyph
from benchmarks.svg_glyphs import render_svg, validate_svg
from screensaver.glyph_design_v2 import COUNT, generate_glyph, geometry, write_study


@pytest.mark.parametrize("index", range(COUNT))
def test_study_has_valid_contours_and_survives_small_rendering(index):
    glyph = generate_glyph(index)
    assert glyph == generate_glyph(index)
    assert validate_svg(glyph["svg"])["passed"]
    reference = measure_glyph(render_svg(glyph["svg"], 128))
    assert reference["threshold_topology_stable"]
    for size in (16, 32):
        sample = measure_glyph(render_svg(glyph["svg"], size))
        assert not sample["blank"]
        assert sample["components_raw"] == reference["components_raw"]
        assert sample["holes_raw"] == reference["holes_raw"]


def test_roof_hook_has_no_punched_divot_and_roof_bowl_has_no_join_step():
    hook = geometry(1)
    # The roof has one uninterrupted lower edge on each side of the stem.
    assert hook.covers(box(18, 16, 82, 29))
    assert hook.boundary.intersection(LineString([(64, 29), (82, 29)])).length == 18
    bowl = geometry(15)
    assert bowl.covers(box(20, 16, 36, 29))
    assert bowl.boundary.intersection(LineString([(20, 16), (20, 29)])).length == 13


def test_diagonal_and_curved_glyphs_both_have_straight_exposed_terminal_cuts():
    diagonal = geometry(14)
    terminal = LineString([(68, 14), (68, 27)])
    assert diagonal.boundary.intersection(terminal).length == 13
    assert not diagonal.covers(Point(69, 21))  # Would lie in a rounded cap.
    bowl = geometry(15)
    assert bowl.boundary.intersection(LineString([(79, 58), (89, 58)])).length == 10
    assert not bowl.covers(Point(84, 57))


def test_rejected_arrow_is_replaced_by_separate_character_strokes():
    glyph = generate_glyph(14)
    assert glyph["recipe"] == "slanted-roof-hook"
    shape = geometry(14)
    assert shape.geom_type == "MultiPolygon"
    assert len(shape.geoms) == 3
    # There is clear air between the slanted header and the descending hook.
    assert not shape.intersects(box(25, 36, 60, 40))
    assert not shape.covers(Point(47, 45))


def test_catalog_is_visually_distinct_and_has_no_fragment_or_pinprick_counter():
    hashes = set()
    silhouettes = set()
    for index in range(COUNT):
        glyph = generate_glyph(index)
        pixels = np.asarray(render_svg(glyph["svg"], 32).convert("L")) < 128
        hashes.add(hashlib.sha256(pixels.tobytes()).hexdigest())
        full = np.asarray(render_svg(glyph["svg"], 128).convert("L")) < 128
        yy, xx = np.where(full)
        crop = full[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        silhouettes.add(hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest())
        shape = geometry(index)
        parts = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
        assert min(p.area for p in parts) >= 195
        for part in parts:
            for interior in part.interiors:
                from shapely import Polygon
                assert Polygon(interior).area >= 150
    assert len(hashes) == COUNT
    assert len(silhouettes) == COUNT


def test_catalog_export_cannot_overwrite_or_invent_benchmark_evidence(tmp_path):
    out = tmp_path / "study"
    receipt = write_study(out)
    assert receipt["count"] == 192
    assert receipt["benchmark_claimable"] is False
    assert receipt["status"] == "current-catalog"
    before = (out / "manifest.json").read_bytes()
    assert json.loads(before) == receipt
    with pytest.raises(FileExistsError):
        write_study(out)
    assert (out / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("index", [True, -1, 192, 1.5, "14"])
def test_invalid_study_indices_are_rejected(index):
    with pytest.raises(ValueError):
        generate_glyph(index)


def test_approved_017_is_preserved_with_a_detached_center_bar():
    assert generate_glyph(17)["sha256"] == '329ab252f1f8e8c6870b8ccc2a39a763340a7134b79be814c5db788ba4e57573'
    shape = geometry(17)
    assert len(shape.geoms) == 2
    assert shape.geoms[0].distance(shape.geoms[1]) == 9


def test_catalog_has_no_aligned_or_reflected_repeats():
    glyphs = [generate_glyph(i) for i in range(COUNT)]
    result = find_near_matches([render_svg(g["svg"]) for g in glyphs],
                               [g["glyph_id"] for g in glyphs])
    assert result["compared_pairs"] == 18336
    assert not result["aligned_exact_pairs"]
    assert not result["near_matches"]


def test_current_web_catalog_and_source_exports_use_the_revised_contours():
    from pathlib import Path
    from screensaver.export_glyphs import _payload
    from screensaver.export_native_glyphs import ORIGINAL, REPO_ROOT
    assert ORIGINAL == Path("screensaver/glyph-design-v2")
    current = json.loads((REPO_ROOT / ORIGINAL / "catalog.json").read_text(encoding="utf-8"))
    assert _payload()["paths"] == [g["paths"] for g in current["glyphs"]]
    browser = (REPO_ROOT / "screensaver/svg-preview/glyphs.js").read_text(encoding="utf-8")
    assert json.loads(browser.split(" = ", 1)[1].strip().removesuffix(";")) == current
    assert [g["svg_sha256"] for g in current["glyphs"]] == [generate_glyph(i)["sha256"] for i in range(COUNT)]
