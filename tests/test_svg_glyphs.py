"""Validate SVG source geometry and the reproducible original catalog."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import ImageChops

from benchmarks.svg_glyphs import (
    FAMILY_QUOTAS, GLYPH_COUNT, generate_glyph, render_svg, validate_svg, write_catalog,
)


def test_catalog_is_original_filled_geometry_with_stable_families_and_ids():
    catalog = [generate_glyph(index) for index in range(GLYPH_COUNT)]
    assert Counter(g["family"] for g in catalog) == FAMILY_QUOTAS
    assert Counter(g["profile"] for g in catalog) == {"classic": 78, "expanded": 114}
    assert len({g["svg_sha256"] for g in catalog}) == 192
    assert len({g["family"] for g in catalog[:24]}) == len(FAMILY_QUOTAS)
    for index, glyph in enumerate(catalog):
        assert glyph["glyph_id"] == f"GLYPH-{index:03d}"
        assert validate_svg(glyph["svg"])["passed"], glyph["glyph_id"]
        assert generate_glyph(index)["svg"] == glyph["svg"]


def test_svg_rasterizer_preserves_counter_and_detached_mark_at_all_review_sizes():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
           'width="128" height="128"><path fill="#000000" fill-rule="nonzero" '
           'd="M10 10 L70 10 L70 90 L10 90 Z M30 30 L30 70 L50 70 L50 30 Z"/>'
           '<path fill="#000000" fill-rule="nonzero" '
           'd="M80 40 L95 40 L95 60 L80 60 Z"/></svg>')
    assert validate_svg(svg)["passed"]
    for size in (16, 32, 64, 128, 512):
        image = render_svg(svg, size)
        assert image.getpixel((int(size*.4), int(size*.5)))[0] > 200
        assert image.getpixel((int(size*.2), int(size*.5)))[0] < 50
        assert image.getpixel((int(size*.875), int(size*.5)))[0] < 50
        assert image.getpixel((int(size*.75), int(size*.5)))[0] > 200


@pytest.mark.parametrize("replacement", [
    '<image href="https://example.com/a.png"/>',
    '<script>alert(1)</script>',
    '<path fill="#000000" fill-rule="nonzero" d="M0 0 L101 0 L1 1 Z"/>',
    '<path fill="#000000" fill-rule="nonzero" d="M0 0 L10 10 L20 20 Z"/>',
    '<path fill="#000000" fill-rule="nonzero" d="M0 0 L10 10 L1e309 3 Z"/>',
    '<path fill="#000000" fill-rule="nonzero" d="M0 0 L50 0 L50 50"/>',
    '<path fill="#000000" fill-rule="nonzero" d="M10 10 L70 80 L70 10 L20 60 Z"/>',
])
def test_svg_validator_rejects_external_or_invalid_geometry(replacement):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
           'width="128" height="128">' + replacement + '</svg>')
    assert not validate_svg(svg)["passed"]


def test_generation_does_not_mutate_random_state_or_reuse_a_raster():
    import random
    state = random.getstate()
    glyph = generate_glyph(7)
    assert random.getstate() == state
    assert "<image" not in glyph["svg"] and "<text" not in glyph["svg"]
    assert glyph["svg_sha256"] == hashlib.sha256(glyph["svg"].encode()).hexdigest()
    assert ImageChops.difference(render_svg(glyph["svg"]).convert("RGB"),
                                 render_svg(generate_glyph(8)["svg"]).convert("RGB")).getbbox()


def test_catalog_export_refuses_to_replace_existing_evidence(tmp_path):
    destination = tmp_path / "catalog"
    result = write_catalog(destination, count=2)
    assert result["count"] == 2
    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_catalog(destination, count=3)
    assert (destination / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("index,attempt", [(True, 0), (-1, 0), (192, 0), (0, -1), (0, True)])
def test_invalid_generation_requests_are_rejected(index, attempt):
    with pytest.raises(ValueError):
        generate_glyph(index, attempt)


def test_recipe_receipt_identifies_the_actual_authored_structure_and_numeral_variant():
    glyph = generate_glyph(18)
    assert glyph["recipe"].endswith("/" + glyph["authoring_parameters"]["structure"])
    assert generate_glyph(63)["recipe"] == "numeral_operator/14"


def test_completed_original_catalog_passes_shape_and_distinctness_gates():
    from benchmarks.svg_glyph_measurements import evaluate_catalog, find_near_matches, measure_glyph

    glyphs = [generate_glyph(index) for index in range(GLYPH_COUNT)]
    images = [render_svg(glyph["svg"]) for glyph in glyphs]
    records = [{**glyph, "measurements": measure_glyph(image)}
               for glyph, image in zip(glyphs, images)]
    reference = json.loads((Path(__file__).resolve().parents[1] /
                            "docs/data/glyph-style-summary.json").read_text(encoding="utf-8"))
    distinctness = find_near_matches(images, [glyph["glyph_id"] for glyph in glyphs])
    report = evaluate_catalog(records, reference, distinctness)
    assert report["failed_gates"] == []
    assert distinctness["near_matches"] == []
    assert report["accepted"] is False  # Numeric tests cannot supply an optical review.
    for glyph, record in zip(glyphs, records):
        large = record["measurements"]
        for size in (16, 32, 64):
            small = measure_glyph(render_svg(glyph["svg"], size))
            assert not small["blank"], (glyph["glyph_id"], size)
            assert small["components_raw"] >= large["components_raw"], (glyph["glyph_id"], size)
            assert small["holes_raw"] >= large["holes_raw"], (glyph["glyph_id"], size)
            extra = small["holes_raw"] - large["holes_raw"]
            if extra:
                # A1–3px sampling speck is diagnostic; a visibly closed
                # entrance or split counter is an optical regression.
                assert max(sorted(small["hole_areas"])[:extra]) <= 3, (glyph["glyph_id"], size)
