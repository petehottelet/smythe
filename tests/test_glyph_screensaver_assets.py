"""Tests for the deterministic cyber-glyph flagship visual assets."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from benchmarks.glyph_screensaver_assets import (  # noqa: E402
    ATLAS_SIZE,
    GIF_FRAMES,
    GIF_SIZE,
    GLYPH_CATALOG_SPECS,
    GLYPH_COUNT,
    GLYPH_SPECS,
    MAX_GLYPH_COUNT,
    PREVIEW_SIZE,
    TILE_SIZE,
    ProceduralGlyphProvider,
    assemble_atlas,
    assemble_html,
    assemble_preview,
    build_glyph_screensaver_assets,
    get_glyph_specs,
    glyph_prompt,
    normalize_tile,
    render_glyph_tile,
)


def _inspect_image(data_or_path) -> tuple[str, tuple[int, int], str, int]:
    source = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path
    with Image.open(source) as image:
        image.load()
        return image.format, image.size, image.mode, int(getattr(image, "n_frames", 1))


def test_catalog_contains_192_unique_fictional_stroke_specs():
    assert len(GLYPH_SPECS) == GLYPH_COUNT == 192
    assert len({spec.id for spec in GLYPH_SPECS}) == GLYPH_COUNT
    assert len({spec.strokes for spec in GLYPH_SPECS}) == GLYPH_COUNT
    assert all(2 <= len(spec.strokes) <= 9 for spec in GLYPH_SPECS)
    assert all(
        stroke[0] in {"l", "q", "d"}
        for spec in GLYPH_SPECS
        for stroke in spec.strokes
    )
    assert all(spec.speed > 0 and spec.trail_length >= 8 for spec in GLYPH_SPECS)


def test_extended_256_catalog_preserves_flagship_and_adds_unique_specs():
    extended = get_glyph_specs(MAX_GLYPH_COUNT)

    assert len(extended) == len(GLYPH_CATALOG_SPECS) == MAX_GLYPH_COUNT == 256
    assert extended[:GLYPH_COUNT] == GLYPH_SPECS
    assert len({spec.id for spec in extended}) == MAX_GLYPH_COUNT
    assert len({spec.strokes for spec in extended}) == MAX_GLYPH_COUNT
    assert extended[-1].id == "glyph-255"


def test_extended_catalog_assembles_16_by_16_atlas_and_html(tmp_path):
    tile_paths = []
    for spec in get_glyph_specs(MAX_GLYPH_COUNT):
        path = tmp_path / f"{spec.id}.png"
        path.write_bytes(render_glyph_tile(spec))
        tile_paths.append(path)

    atlas = assemble_atlas(tile_paths, tmp_path / "atlas.png")
    html = assemble_html(
        tmp_path / "glyph-rain.html",
        glyph_count=MAX_GLYPH_COUNT,
    )
    html_text = Path(html.path).read_text(encoding="utf-8")
    strokes = json.loads(re.search(r"const strokes=(.*);", html_text).group(1))

    assert (atlas.width, atlas.height) == (2048, 2048)
    assert len(strokes) == MAX_GLYPH_COUNT


def test_rendered_tiles_are_deterministic_unique_transparent_pngs():
    first_pass = [render_glyph_tile(spec) for spec in GLYPH_SPECS]
    second_pass = [render_glyph_tile(spec) for spec in GLYPH_SPECS]
    assert first_pass == second_pass
    assert len({hashlib.sha256(data).hexdigest() for data in first_pass}) == GLYPH_COUNT
    for data in first_pass:
        image_format, size, mode, frames = _inspect_image(data)
        assert (image_format, size, mode, frames) == (
            "PNG",
            (TILE_SIZE, TILE_SIZE),
            "RGBA",
            1,
        )


def test_procedural_provider_returns_one_matching_tile_per_concurrent_prompt():
    provider = ProceduralGlyphProvider(latency_s=0.001)

    async def run_all():
        return await asyncio.gather(
            *(
                provider.complete("", glyph_prompt(spec), "procedural-glyph-v1")
                for spec in GLYPH_SPECS
            )
        )

    results = asyncio.run(run_all())
    assert len(results) == GLYPH_COUNT
    assert len(provider.calls) == GLYPH_COUNT
    assert set(provider.calls) == {spec.id for spec in GLYPH_SPECS}
    hashes = set()
    for spec, result in zip(GLYPH_SPECS, results, strict=True):
        assert result.cost_usd == 0.0
        assert len(result.artifacts) == 1
        assert result.artifacts[0].mime_type == "image/png"
        assert json.loads(result.text)["glyph_id"] == spec.id
        assert _inspect_image(result.artifacts[0].data)[1] == (TILE_SIZE, TILE_SIZE)
        hashes.add(hashlib.sha256(result.artifacts[0].data).hexdigest())
    assert len(hashes) == GLYPH_COUNT


def test_normalize_tile_contains_rectangular_input_on_transparent_square(tmp_path):
    source = io.BytesIO()
    image = Image.new("RGB", (240, 80), (2, 4, 3))
    draw = ImageDraw.Draw(image)
    draw.line((80, 65, 120, 15, 160, 65), fill=(20, 240, 80), width=8)
    image.save(source, format="JPEG")
    destination = tmp_path / "normalized.png"
    receipt = normalize_tile(source.getvalue(), destination)

    assert receipt.format == "PNG"
    assert (receipt.width, receipt.height) == (TILE_SIZE, TILE_SIZE)
    assert receipt.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    image_format, size, mode, frames = _inspect_image(destination)
    assert (image_format, size, mode, frames) == (
        "PNG",
        (TILE_SIZE, TILE_SIZE),
        "RGBA",
        1,
    )
    with Image.open(destination) as normalized:
        alpha = normalized.getchannel("A")
        assert alpha.getextrema()[0] == 0
        assert alpha.getbbox() is not None


def test_normalize_tile_rejects_opaque_rectangular_pseudo_glyph(tmp_path):
    source = io.BytesIO()
    Image.new("RGB", (128, 128), (20, 240, 80)).save(source, format="PNG")

    with pytest.raises(ValueError, match="no separable glyph foreground"):
        normalize_tile(source.getvalue(), tmp_path / "invalid.png")


def test_complete_suite_has_valid_dimensions_animation_html_and_receipts(tmp_path):
    receipt = build_glyph_screensaver_assets(tmp_path)
    assert len(receipt.tiles) == GLYPH_COUNT
    assert receipt.unique_tile_hashes == GLYPH_COUNT
    assert len({tile.sha256 for tile in receipt.tiles}) == GLYPH_COUNT
    assert all(Path(tile.path).is_file() for tile in receipt.tiles)

    expected = (
        (receipt.preview, "PNG", PREVIEW_SIZE, 1),
        (receipt.animation, "GIF", GIF_SIZE, GIF_FRAMES),
        (receipt.atlas, "PNG", ATLAS_SIZE, 1),
    )
    for output, image_format, size, frames in expected:
        path = Path(output.path)
        assert path.is_file()
        assert output.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        observed_format, observed_size, _mode, observed_frames = _inspect_image(path)
        assert observed_format == image_format
        assert observed_size == size
        assert observed_frames == frames
        assert (output.width, output.height, output.frames) == (*size, frames)

    html_path = Path(receipt.html.path)
    html = html_path.read_text(encoding="utf-8")
    assert receipt.html.format == "HTML"
    assert (receipt.html.width, receipt.html.height, receipt.html.frames) == (
        *PREVIEW_SIZE,
        0,
    )
    assert receipt.html.sha256 == hashlib.sha256(html_path.read_bytes()).hexdigest()
    assert '<canvas id="rain" width="1920" height="1080"' in html
    assert "requestAnimationFrame" in html
    assert "const strokes=" in html
    assert "https://" not in html and "http://" not in html

    # Repeat the largest still and the executable canvas document to prove that
    # the seed and serialized program are stable without rebuilding the already
    # verified GIF and all 192 tiles a second time in CI.
    tile_paths = [tile.path for tile in receipt.tiles]
    repeat_preview = assemble_preview(tile_paths, tmp_path / "repeat-preview.png")
    repeat_html = assemble_html(tmp_path / "repeat.html")
    assert repeat_preview.sha256 == receipt.preview.sha256
    assert repeat_html.sha256 == receipt.html.sha256
