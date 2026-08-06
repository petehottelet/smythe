"""Regression tests for public benchmark chart evidence and palette."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.render_readme_charts import (
    render_framework_callouts,
    render_framework_comparison,
    render_glyph_pipeline,
    render_glyph_scaling,
    render_glyph_specimens,
    render_shape_efficiency,
)

ROOT = Path(__file__).parents[1]
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
MONOCHROME = {"#000000", "#ffffff"}


def test_framework_chart_uses_corrected_record_values():
    svg = render_framework_comparison()
    for value in ("9.73", "8,796", "30.53s", "9,590", "38,427", "42.48s"):
        assert value in svg
    assert "framework_h2h_rightsized.json" in svg
    assert "15 runs per framework" in svg


def test_framework_callouts_are_computed_and_use_trajan():
    svg = render_framework_callouts()
    assert "77%" in svg
    assert "LOWER MEAN TOKEN LOAD THAN CREWAI" in svg
    assert "8,796 vs 38,427 mean tokens" in svg
    assert "6%" in svg
    assert "Trajan Pro 3" in svg
    assert "framework_h2h_rightsized.json" in svg


def test_generated_public_charts_are_strictly_black_and_white():
    for renderer in (
        render_framework_comparison,
        render_framework_callouts,
        render_shape_efficiency,
        render_glyph_scaling,
        render_glyph_pipeline,
        render_glyph_specimens,
    ):
        colors = set(HEX_COLOR.findall(renderer()))
        assert colors == MONOCHROME


def test_committed_graph_assets_are_strictly_black_and_white():
    paths = [*ROOT.joinpath("assets").rglob("*.svg")]
    paths.extend(ROOT.joinpath("examples").rglob("*.mmd"))
    assert paths
    for path in paths:
        colors = set(HEX_COLOR.findall(path.read_text(encoding="utf-8")))
        assert colors <= MONOCHROME, f"{path} contains {colors - MONOCHROME}"


def test_readme_badges_use_bordered_black_and_white_assets():
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    badge_paths = re.findall(r'src="(assets/badges/[^"]+\.svg)"', readme)
    assert len(badge_paths) == 4
    for relative in badge_paths:
        svg = ROOT.joinpath(relative).read_text(encoding="utf-8")
        assert set(HEX_COLOR.findall(svg)) == MONOCHROME
        assert 'fill="#ffffff" stroke="#000000"' in svg


def test_glyph_diagrams_use_the_committed_vector_catalog():
    pipeline = render_glyph_pipeline()
    specimens = render_glyph_specimens()
    assert "192-node generated graph" in pipeline
    for glyph_id in ("GLYPH-000", "GLYPH-151"):
        assert glyph_id in specimens
    assert "12 / 192" in specimens


def test_glyph_scaling_chart_uses_all_four_isolated_records():
    svg = render_glyph_scaling()
    for value in (
        "64 nodes",
        "128 nodes",
        "192 nodes",
        "256 nodes",
        "40.37×",
        "44.51×",
        "56.21×",
        "49.56×",
        "64 / 128 / 192 / 256 valid unique tiles",
    ):
        assert value in svg
    assert "Scaling across 64, 128, 192 and 256 nodes" in svg


def test_readme_places_evidence_before_the_glyph_rain_example():
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    ordered_markers = (
        "## 60-second quickstart",
        "## Framework comparison",
        "assets/benchmarks/framework_comparison.svg",
        "## Architected planning beats fixed execution on efficiency",
        "assets/benchmarks/shape_efficiency.svg",
        "## Artifact fan-out scales from 64 to 256 nodes",
        "assets/benchmarks/glyph_scaling.svg",
        "## Example: Glyph Rain at 192-node fan-out",
        "assets/glyph_rain/glyph-rain-screenshot.png",
        "**Download:** [Windows `.scr`]",
        "assets/glyph_rain/glyph_pipeline.svg",
        "assets/glyph_rain/glyph_specimens.svg",
    )
    positions = [readme.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)


def test_public_mermaid_avoids_reserved_graph_node_id():
    public_markdown = [ROOT / "README.md", *ROOT.joinpath("docs").glob("*.md")]
    reserved_node = re.compile(r"^\s*graph\[", re.MULTILINE)
    for path in public_markdown:
        assert not reserved_node.search(path.read_text(encoding="utf-8")), path


def test_screensaver_is_one_word_in_public_landing_copy():
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    pipeline = render_glyph_pipeline()
    assert not re.search(r"screen saver", readme, re.IGNORECASE)
    assert "SCREEN SAVER" not in pipeline
