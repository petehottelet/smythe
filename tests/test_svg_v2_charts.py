"""Landing figures recompute the full campaign and show resource tradeoffs."""

from copy import deepcopy
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

from benchmarks.svg_v2_charts import RECORD, checked_record, render_memory, render_workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return json.loads((ROOT / "benchmarks/results" / RECORD).read_bytes())


def test_current_figures_are_reproducible_monochrome_and_retain_all_trials():
    path = ROOT / "benchmarks/results" / RECORD
    svg = render_workflow(path)
    assert (ROOT / "assets/benchmarks/svg_v2_workflow.svg").read_text(encoding="utf-8") == svg
    assert len(ET.fromstring(svg).findall('.//{*}circle[@data-seconds]')) == 36
    assert "neither creative design nor screensaver FPS" in svg
    assert "min–max" in svg
    memory = render_memory(path)
    assert (ROOT / "assets/benchmarks/svg_v2_memory.svg").read_text(encoding="utf-8") == memory
    assert "RSS counts shared pages" in memory
    for figure in (svg, memory):
        assert set(re.findall(r'#[0-9a-fA-F]{6}', figure)) <= {"#000000", "#ffffff"}
        assert RECORD in figure
        assert "192" in figure and "256" in figure


def test_chart_ignores_favorable_forged_summary(evidence, tmp_path):
    evidence["summaries"] = [{"median_s": .000001, "speedup_vs_same_backend_c1": 999999}]
    path = tmp_path / "record.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    _, rows, _ = checked_record(path)
    assert len(rows) == 12
    assert all(row["median_s"] > .1 for row in rows)
    assert rows == checked_record(ROOT / "benchmarks/results" / RECORD)[1]
    labels = [node.text for node in ET.fromstring(render_workflow(path)).findall('.//{*}text')]
    assert not any("999999.00×" in label for label in labels)


@pytest.mark.parametrize("damage", ["diagnostic", "defect", "missing", "repeats", "clock", "source", "prefix"])
def test_chart_rejects_invalid_evidence(evidence, tmp_path, damage):
    data = deepcopy(evidence)
    if damage == "diagnostic":
        data["claimable"] = False
    elif damage == "defect":
        data["known_measurement_defects"] = ["unresolved"]
    elif damage == "missing":
        data["runs"].pop()
    elif damage == "repeats":
        data["protocol"]["repeats"] = 1
    elif damage == "clock":
        data["runs"][0]["generation_wall_s"] = True
    elif damage == "source":
        data["runs"][0]["source_stable"] = False
    else:
        for run in data["runs"]:
            if run["glyph_count"] == 256:
                run["glyphs"][0]["svg_sha256"] = "replacement"
    path = tmp_path / "record.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        render_workflow(path)


def test_unavailable_memory_cannot_be_drawn_as_zero(evidence, tmp_path):
    evidence["runs"][0]["memory"]["sampled_peak_process_tree_rss_bytes"] = None
    path = tmp_path / "record.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="memory evidence"):
        render_memory(path)
