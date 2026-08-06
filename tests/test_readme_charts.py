"""Regression tests for public benchmark chart evidence and palette."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.render_readme_charts import (
    render_framework_callouts,
    render_framework_comparison,
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


def test_readme_badges_use_black_and_white_two_tone_fields():
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    badge_urls = re.findall(r'src="(https://img\.shields\.io/[^"]+)"', readme)
    assert len(badge_urls) == 4
    assert all("labelColor=000000" in url for url in badge_urls)
    assert all("ffffff" in url for url in badge_urls)
    assert "labelColor=ffffff" not in readme
