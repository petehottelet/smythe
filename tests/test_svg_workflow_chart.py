"""Public claims must come from complete raw SVG workflow evidence."""

from copy import deepcopy
import math
from xml.etree import ElementTree as ET

import pytest

from benchmarks import render_readme_charts as charts


@pytest.fixture
def evidence(monkeypatch):
    glyphs = [{"index": i, "svg_sha256": f"svg-{i}", "pixel_sha256": f"pixels-{i}",
               "measurement_sha256": f"measurements-{i}"} for i in range(192)]
    runs = []
    for executor, concurrency, seconds in (("thread", 1, 12), ("thread", 4, 6),
                                            ("process", 1, 15), ("process", 4, 8)):
        for repeat, delta in enumerate((-1, 0, 1), start=1):
            elapsed = seconds + delta
            runs.append({"executor": executor, "concurrency": concurrency, "repeat": repeat,
                         "status": "passed", "errors": [], "provider_calls": 192,
                         "successful_provider_calls": 192, "failed_provider_calls": 0,
                         "completed_nodes": 192, "valid_glyphs": 192, "api_calls": 0,
                         "api_cost_usd": 0, "style_acceptance": {"accepted": True},
                         "glyphs": deepcopy(glyphs), "end_to_end_wall_s": elapsed,
                         "setup_wall_s": elapsed*.05, "worker_shutdown_wall_s": elapsed*.05,
                         "generation_wall_s": elapsed*.15, "validation_wall_s": elapsed*.6,
                         "assembly_wall_s": elapsed*.1})
    record = {"status": "passed", "claimable": True, "readme_promotion_eligible": True,
              "catalog_style_accepted": True, "known_measurement_defects": [],
              "protocol": {"glyph_count": 192, "repeats": 3, "simulated_latency_s": 0,
                           "cached_generation_outputs": False, "executors": ["thread", "process"],
                           "concurrencies": [1, 4], "worker_cap": 8}, "runs": runs,
              "fastest_median_workflow": {"end_to_end_wall_s": {"median": .001}}}
    monkeypatch.setattr(charts, "_load", lambda _: record)
    return record


def test_svg_chart_recomputes_medians_and_uses_complete_workflow(evidence):
    svg = charts.render_svg_workflow()
    assert "6.00s" in svg and "2.00×" in svg
    assert "0.001" not in svg  # A forged summary cannot improve the published claim.
    assert "min–max" in svg and "worker cap 8" in svg
    assert "hardware and electricity unpriced" in svg
    assert "glyph_svg_v1.json" in svg


def test_stage_breakdown_uses_one_actual_median_run_and_reconciles_to_total(evidence):
    # Independently taking medians of these phase clocks would invent a workflow.
    for run in evidence["runs"]:
        if run["executor"] == "thread" and run["concurrency"] == 4:
            generation, validation, assembly = ((3, 1, .1), (.75, 4, .25), (2, 3, .5))[run["repeat"] - 1]
            run.update(generation_wall_s=generation, validation_wall_s=validation, assembly_wall_s=assembly)
    svg = charts.render_svg_workflow()
    root = ET.fromstring(svg)
    namespace = {"s": "http://www.w3.org/2000/svg"}
    group = root.find("s:g[@id='workflow-phase-breakdown']", namespace)
    assert group is not None
    assert group.attrib["data-executor"] == "thread"
    assert group.attrib["data-concurrency"] == "4"
    assert group.attrib["data-repeat"] == "2"
    assert float(group.attrib["data-total-seconds"]) == 6
    segments = group.findall("s:rect[@data-phase]", namespace)
    seconds = {node.attrib["data-phase"]: float(node.attrib["data-seconds"]) for node in segments}
    assert seconds == {"generation": .75, "validation": 4, "assembly": .25, "other": 1}
    assert math.fsum(seconds.values()) == float(group.attrib["data-total-seconds"])
    assert math.fsum(float(node.attrib["width"]) for node in segments) == pytest.approx(880, abs=1e-5)
    assert "Thread c4, repetition 2; one measured 6.00s workflow" in svg
    assert "setup, worker shutdown, and remaining measured overhead" in svg
    # All configuration medians and ranges remain in the chart data.
    _, rows = charts._svg_workflow_rows()
    assert [(row["executor"], row["concurrency"], row["min"], row["median"], row["max"]) for row in rows] == [
        ("process", 1, 14, 15, 16), ("process", 4, 7, 8, 9),
        ("thread", 1, 11, 12, 13), ("thread", 4, 5, 6, 7),
    ]


@pytest.mark.parametrize("field,value", [
    ("generation_wall_s", None), ("validation_wall_s", float("nan")),
    ("assembly_wall_s", -1), ("setup_wall_s", True), ("worker_shutdown_wall_s", float("inf")),
    ("generation_wall_s", 0), ("validation_wall_s", 100), ("setup_wall_s", 100),
])
def test_stage_breakdown_rejects_missing_or_inconsistent_clocks_in_any_run(evidence, field, value):
    # The altered slow configuration must be validated even though it is not selected.
    evidence["runs"][0][field] = value
    with pytest.raises(ValueError, match="phase timing"):
        charts.render_svg_workflow()


def test_stage_breakdown_does_not_invent_an_average_run_for_even_repetitions(evidence):
    evidence["protocol"]["repeats"] = 4
    for original in list(evidence["runs"]):
        if original["repeat"] == 3:
            extra = deepcopy(original)
            extra["repeat"] = 4
            extra["end_to_end_wall_s"] += 1
            evidence["runs"].append(extra)
    with pytest.raises(ValueError, match="actual run at the median"):
        charts.render_svg_workflow()


@pytest.mark.parametrize("defect", [
    "style", "defects", "cached", "latency", "missing_repeat", "duplicate_repeat",
    "missing_config", "missing_glyph", "duplicate_shape", "changed_shape", "failed",
    "incomplete", "invalid_time", "api_calls",
])
def test_svg_chart_rejects_unusable_evidence(evidence, defect):
    if defect == "style":
        evidence["catalog_style_accepted"] = False
    elif defect == "defects":
        evidence["known_measurement_defects"] = ["invalid timer"]
    elif defect in {"cached", "latency"}:
        evidence["protocol"]["cached_generation_outputs" if defect == "cached" else "simulated_latency_s"] = 1
    elif defect == "missing_repeat":
        evidence["runs"].pop()
    elif defect == "duplicate_repeat":
        evidence["runs"][0]["repeat"] = 2
    elif defect == "missing_config":
        evidence["runs"] = evidence["runs"][:-3]
    elif defect == "missing_glyph":
        evidence["runs"][0]["glyphs"].pop()
    elif defect == "duplicate_shape":
        evidence["runs"][0]["glyphs"][1]["pixel_sha256"] = "pixels-0"
    elif defect == "changed_shape":
        evidence["runs"][0]["glyphs"][0]["svg_sha256"] = "changed"
    elif defect == "failed":
        evidence["runs"][0]["status"] = "failed"
    elif defect == "incomplete":
        evidence["runs"][0]["valid_glyphs"] = 191
    elif defect == "invalid_time":
        evidence["runs"][0]["end_to_end_wall_s"] = float("nan")
    else:
        evidence["runs"][0]["api_calls"] = 1
    with pytest.raises(ValueError):
        charts.render_svg_workflow()
