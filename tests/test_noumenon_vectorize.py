"""Tests for the pure-Python Noumenon tile vectorizer."""

from __future__ import annotations

import hashlib
import io
import random

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from benchmarks.noumenon_assets import GLYPH_SPECS, render_glyph_tile  # noqa: E402
from benchmarks.noumenon_vectorize import (  # noqa: E402
    IOU_THRESHOLD,
    GlyphMask,
    glyph_mask,
    mask_iou,
    outline_path_data,
    parse_path_loops,
    rasterize_evenodd,
    svg_document,
    trace_outlines,
    validate_svg,
    vectorize_tile,
)


def _mask(*rows: str) -> GlyphMask:
    bits = bytes(1 if char == "#" else 0 for row in rows for char in row)
    return GlyphMask(len(rows[0]), len(rows), bits, "alpha")


def _round_trip(mask: GlyphMask) -> bytes:
    loops = parse_path_loops(outline_path_data(trace_outlines(mask)))
    return rasterize_evenodd(loops, mask.width, mask.height)


def _signed_area(loop) -> float:
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(loop, [*loop[1:], loop[0]])
    ) / 2


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_single_pixel_traces_one_clockwise_square():
    mask = _mask("...", ".#.", "...")

    loops = trace_outlines(mask)

    assert loops == [[(1, 1), (2, 1), (2, 2), (1, 2)]]
    assert outline_path_data(loops) == "M1 1H2V2H1Z"
    assert _round_trip(mask) == mask.bits


def test_collinear_edges_merge_into_rectangle_corners():
    mask = _mask("......", ".####.", ".####.", "......")

    assert trace_outlines(mask) == [[(1, 1), (5, 1), (5, 3), (1, 3)]]


def test_shape_with_hole_round_trips_exactly_with_opposite_windings():
    mask = _mask(
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    )

    loops = trace_outlines(mask)

    assert len(loops) == 2
    outer, hole = loops
    assert outer == [(0, 0), (5, 0), (5, 5), (0, 5)]
    assert set(hole) == {(1, 1), (4, 1), (4, 4), (1, 4)}
    # Clockwise outline and counter-clockwise hole in SVG's y-down space.
    assert _signed_area(outer) > 0 > _signed_area(hole)
    assert outline_path_data(loops).count("M") == 2
    assert _round_trip(mask) == mask.bits


def test_diagonal_neighbours_stay_separate_outlines():
    mask = _mask("#.", ".#")

    assert trace_outlines(mask) == [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(1, 1), (2, 1), (2, 2), (1, 2)],
    ]
    assert _round_trip(mask) == mask.bits


@pytest.mark.parametrize("seed", range(6))
def test_random_masks_round_trip_exactly(seed):
    rng = random.Random(seed)
    width, height = 23, 17
    bits = bytes(rng.random() < 0.45 for _ in range(width * height))
    mask = GlyphMask(width, height, bits, "alpha")

    assert _round_trip(mask) == bits


def test_evenodd_fill_xors_loops_regardless_of_winding():
    outer = [(0, 0), (6, 0), (6, 6), (0, 6)]
    same_direction_inner = [(2, 2), (4, 2), (4, 4), (2, 4)]
    overlapping = [(5, 5), (8, 5), (8, 8), (5, 8)]

    bits = rasterize_evenodd([outer, same_direction_inner, overlapping], 8, 8)

    expected = _mask(
        "######..",
        "######..",
        "##..##..",
        "##..##..",
        "######..",
        "#####.##",
        ".....###",
        ".....###",
    )
    assert bits == expected.bits


def test_path_parser_accepts_relative_and_implicit_linetos():
    square = [[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]]

    assert parse_path_loops("m1 1h2v2h-2z") == square
    assert parse_path_loops("M1,1 3,1 3,3 1,3 Z") == square
    assert parse_path_loops("M1 1L3 1L3 3L1 3Z") == square


@pytest.mark.parametrize(
    ("d", "message"),
    [
        ("M0 0H2V2H0", "subpath is not closed"),
        ("M0 0Q1 1 2 0Z", "unsupported path command 'Q'"),
        ("H2V2Z", "has no current subpath"),
        ("M0 0H2V2H0Z 5", "must follow a drawing command"),
        ("M0 0H#Z", "outside numbers and commands"),
        ("M0 Z", "missing a coordinate"),
    ],
)
def test_path_parser_rejects_malformed_or_open_outlines(d, message):
    with pytest.raises(ValueError, match=message):
        parse_path_loops(d)


def test_glyph_mask_uses_alpha_for_transparent_tiles():
    image = Image.new("RGBA", (3, 1), (0, 255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0, 127))
    image.putpixel((2, 0), (0, 255, 0, 128))

    mask = glyph_mask(image)

    assert mask.source == "alpha"
    assert mask.bits == b"\x00\x00\x01"


def test_glyph_mask_uses_luminance_for_opaque_tiles_on_black():
    image = Image.new("RGB", (3, 1), (0, 0, 0))
    image.putpixel((1, 0), (0, 109, 0))
    image.putpixel((2, 0), (0, 110, 0))
    assert image.convert("L").tobytes() == bytes([0, 64, 65])

    mask = glyph_mask(image)

    assert mask.source == "luminance"
    assert mask.bits == b"\x00\x00\x01"


def test_vectorize_tile_writes_valid_hash_bound_evenodd_svg(tmp_path):
    image = Image.new("RGBA", (40, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 35, 27), fill=(40, 250, 90, 255))
    draw.rectangle((12, 10, 27, 21), fill=(0, 0, 0, 0))
    destination = tmp_path / "ring.svg"

    receipt = vectorize_tile(_png(image), destination)

    data = destination.read_bytes()
    assert receipt.valid and receipt.errors == ()
    assert receipt.iou == 1.0
    assert receipt.outline_count == 2
    assert receipt.vertex_count == 8
    assert receipt.mask_source == "alpha"
    assert receipt.mask_pixels == 32 * 24 - 16 * 12
    assert receipt.sha256 == hashlib.sha256(data).hexdigest()
    assert receipt.byte_size == len(data)
    text = data.decode("utf-8")
    assert 'viewBox="0 0 40 32"' in text
    assert 'fill-rule="evenodd"' in text
    assert 'fill="#28fa5a"' in text


def test_vectorize_tile_traces_opaque_glyph_on_black_by_luminance(tmp_path):
    image = Image.new("RGB", (24, 24), (0, 0, 0))
    ImageDraw.Draw(image).ellipse((3, 3, 20, 20), fill=(30, 230, 80))

    receipt = vectorize_tile(_png(image), tmp_path / "dot.svg")

    assert receipt.valid
    assert receipt.mask_source == "luminance"
    assert receipt.iou == 1.0


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGBA", (16, 16), (0, 255, 0, 0)),
        Image.new("RGB", (16, 16), (0, 0, 0)),
    ],
    ids=["transparent", "opaque-black"],
)
def test_blank_tile_fails_without_writing_an_svg(tmp_path, image):
    destination = tmp_path / "blank.svg"
    destination.write_text("<svg/>", encoding="utf-8")  # stale earlier output

    receipt = vectorize_tile(_png(image), destination)

    assert not receipt.valid
    assert receipt.errors == ("glyph mask is blank",)
    assert receipt.path is None and receipt.iou is None
    assert not destination.exists()


def test_procedural_glyph_tiles_vectorize_exactly(tmp_path):
    for spec in GLYPH_SPECS[:4]:
        receipt = vectorize_tile(render_glyph_tile(spec), tmp_path / f"{spec.id}.svg")
        assert receipt.valid, receipt.errors
        assert receipt.iou == 1.0
        assert receipt.outline_count >= 1


def _svg(mask: GlyphMask) -> bytes:
    return svg_document(
        trace_outlines(mask), width=mask.width, height=mask.height, fill="#00ff00"
    ).encode("utf-8")


def test_validate_svg_rejects_a_drawing_that_misses_the_mask():
    drawn = _mask("##..", "##..", "....", "....")
    other = _mask("....", "....", "..##", "..##")

    check = validate_svg(_svg(drawn), other)

    assert not check.valid
    assert check.iou == 0.0
    assert any(f"below the {IOU_THRESHOLD} threshold" in error for error in check.errors)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b' fill-rule="evenodd"', b"", 'fill-rule="evenodd"'),
        (b'viewBox="0 0 4 4"', b'viewBox="0 0 8 8"', "viewBox"),
        (b"Z", b"", "subpath is not closed"),
        (b"</svg>", b"", "not well-formed XML"),
    ],
)
def test_validate_svg_rejects_structural_defects(old, new, message):
    mask = _mask("....", ".##.", ".##.", "....")
    data = _svg(mask).replace(old, new, 1)

    check = validate_svg(data, mask)

    assert not check.valid
    assert any(message in error for error in check.errors)


def test_mask_iou_counts_overlap_and_refuses_two_blank_masks():
    assert mask_iou(b"\x01\x01\x00\x00", b"\x01\x00\x01\x00") == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="undefined"):
        mask_iou(b"\x00\x00", b"\x00\x00")


def test_glyph_mask_rejects_malformed_bits():
    with pytest.raises(ValueError, match="0 or 1"):
        GlyphMask(2, 1, b"\x00\x02", "alpha")
    with pytest.raises(ValueError, match="dimensions"):
        GlyphMask(2, 2, b"\x00", "alpha")
