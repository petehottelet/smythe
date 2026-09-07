"""Regression tests for public benchmark chart evidence and palette."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import render_readme_charts
from benchmarks.render_readme_charts import (
    render_framework_callouts,
    render_framework_comparison,
    render_glyph_pipeline,
    render_glyph_scaling,
    render_glyph_specimens,
    render_recovery,
    render_shape_efficiency,
    render_svg_workflow,
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
    assert "28%" in svg
    assert "LOWER MEAN WALL TIME THAN CREWAI" in svg
    assert "Trajan Pro 3" in svg
    assert "framework_h2h_rightsized.json" in svg


def test_generated_public_charts_are_strictly_black_and_white():
    for renderer in (
        render_framework_comparison,
        render_framework_callouts,
        render_shape_efficiency,
        render_svg_workflow,
        render_recovery,
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


def test_committed_benchmark_charts_match_the_current_evidence_renderer():
    charts = {
        "framework_comparison.svg": render_framework_comparison,
        "framework_callouts.svg": render_framework_callouts,
        "shape_efficiency.svg": render_shape_efficiency,
        "glyph_scaling.svg": render_glyph_scaling,
        "svg_workflow.svg": render_svg_workflow,
        "recovery.svg": render_recovery,
    }
    for name, renderer in charts.items():
        path = ROOT / "assets" / "benchmarks" / name
        assert path.read_text(encoding="utf-8") == renderer(), (
            f"{name} is stale; run python benchmarks/render_readme_charts.py"
        )


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


def test_readme_connects_glyph_rain_to_benchmark_evidence():
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    ordered_markers = (
        "## Glyph Rain",
        "screensaver/svg-preview/preview.png",
        "**Native downloads:** [Windows `.scr`]",
        "## Benchmarks",
        "### Original SVG generation",
        "assets/benchmarks/svg_workflow.svg",
        "### Glyph generation and scaling",
        "assets/benchmarks/glyph_scaling.svg",
        "### Recovery after interruption",
        "assets/benchmarks/recovery.svg",
        "### Framework efficiency",
        "assets/benchmarks/framework_callouts.svg",
        "assets/benchmarks/framework_comparison.svg",
        "### Generated execution topology",
        "assets/benchmarks/shape_efficiency.svg",
    )
    positions = [readme.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)

    guide = ROOT.joinpath("screensaver/README.md").read_text(encoding="utf-8")
    for name in ("glyph_pipeline.svg", "glyph_specimens.svg"):
        assert f"../assets/glyph_rain/{name}" in guide


def test_shape_chart_promotes_complete_wall_time_and_not_partial_cost():
    svg = render_shape_efficiency()
    assert "Mean graph nodes" in svg
    assert "Planning included in wall time" in svg
    assert "lower end-to-end wall time" in svg
    assert "lower cost" not in svg


@pytest.mark.parametrize("defect", ["summary", "missing", "unmatched", "failed"])
def test_framework_charts_reject_invalid_evidence(monkeypatch, defect):
    record = deepcopy(render_readme_charts._load("framework_h2h_rightsized.json"))
    if defect == "summary":
        record["summary"][0]["tokens_mean"] = 1
    elif defect == "missing":
        record["records"].pop(0)
    elif defect == "unmatched":
        record["records"][0]["task"] = "unmatched-task"
    else:
        record["records"][0]["error"] = "failed"
    monkeypatch.setattr(render_readme_charts, "_load", lambda _: record)
    with pytest.raises(ValueError):
        render_framework_comparison()


@pytest.mark.parametrize("defect", ["speedup", "completed", "validation"])
def test_glyph_chart_rejects_invalid_run_metrics(monkeypatch, defect):
    record = deepcopy(render_readme_charts._load("glyph_screensaver_64_offline_realistic.json"))
    if defect == "speedup":
        record["runs"][-1]["speedup_vs_concurrency_1"] = 1000
    elif defect == "completed":
        record["runs"][-1]["completed_nodes"] = 63
    else:
        record["runs"][-1]["validation"]["passed"] = False
    monkeypatch.setattr(render_readme_charts, "_load", lambda _: record)
    with pytest.raises(ValueError):
        render_readme_charts._glyph_runs("test.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("quality", None), ("quality", float("nan")), ("quality", 11),
        ("wall_s", 0), ("wall_s", float("inf")), ("nodes", 0), ("nodes", 1.5),
    ],
)
def test_shape_chart_rejects_missing_or_invalid_raw_metrics(monkeypatch, field, value):
    record = deepcopy(render_readme_charts._load("shape_suite_v3.json"))
    row = next(row for row in record["records"] if row["baseline"] == "smythe_dynamic")
    row[field] = value
    monkeypatch.setattr(render_readme_charts, "_load", lambda _: record)
    with pytest.raises(ValueError):
        render_shape_efficiency()


@pytest.mark.parametrize("metric", ["quality_mean", "nodes_mean"])
def test_shape_chart_rejects_summaries_that_disagree_with_raw_metrics(monkeypatch, metric):
    record = deepcopy(render_readme_charts._load("shape_suite_v3.json"))
    summary = next(row for row in record["summary"] if row["baseline"] == "smythe_dynamic")
    summary[metric] = 1
    monkeypatch.setattr(render_readme_charts, "_load", lambda _: record)
    with pytest.raises(ValueError):
        render_shape_efficiency()


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
