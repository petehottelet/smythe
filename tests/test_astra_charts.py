"""All Astra observations remain visible in deterministic monochrome charts."""

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from benchmarks.astra_analysis import analyze_main
from benchmarks.astra_charts import checked_record, render_differences, render_distributions
from test_astra_analysis import complete, retained_unknown  # noqa: F401


@pytest.fixture
def record(complete, tmp_path):  # noqa: F811
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analyze_main(*complete)), encoding="utf-8")
    review = {"status": "audited-main", "known_measurement_defects": [],
              "analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_name("review.json").write_text(json.dumps(review), encoding="utf-8")
    return path


def test_all_200_cost_and_time_points_and_inclusion_notes_are_rendered(record):
    svg = render_distributions(record)
    root = ET.fromstring(svg)
    points = [r for r in root.iter() if r.get("data-arm")]
    assert len(points) == 400
    assert "Accepted: 50/50" in svg and "All failures remain included" in svg
    assert "not an external benchmark or untouched holdout" in svg
    assert svg == render_distributions(record)
    paints = {r.get(k) for r in root.iter() for k in ("fill", "stroke") if r.get(k)}
    assert paints <= {"#000000", "#ffffff", "none"}


def test_paired_chart_labels_task_sampling_and_shows_all_four_intervals(record):
    svg = render_differences(record)
    root = ET.fromstring(svg)
    assert sum(r.tag.endswith("circle") for r in root.iter()) == 4
    assert "10 tasks, not 50 independent samples" in svg
    assert "Generated minus fixed" in svg and "10,000 draws" in svg


def test_unreviewed_or_modified_results_cannot_render(record):
    record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="bound evidence review"):
        checked_record(record)


def test_omitting_a_trial_is_rejected_even_with_a_matching_byte_hash(record):
    value = json.loads(record.read_bytes())
    value["all_trials"].pop()
    record.write_text(json.dumps(value), encoding="utf-8")
    review_path = record.with_name("review.json")
    review = json.loads(review_path.read_bytes())
    review["analysis_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="200 balanced"):
        checked_record(record)


def test_unknown_cost_is_a_range_and_never_an_invented_point(complete, tmp_path):  # noqa: F811
    value = analyze_main(*retained_unknown(complete), allow_reserved_cost=True)
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    review = {"status": "audited-main", "known_measurement_defects": [], "unknown_workflow_usage": 1,
              "analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_name("review.json").write_text(json.dumps(review), encoding="utf-8")
    svg = render_distributions(path)
    root = ET.fromstring(svg)
    points = [r for r in root.iter() if r.get("data-value")]
    ranges = [r for r in root.iter() if r.get("data-unknown")]
    assert len(points) == 399 and len(ranges) == 1
    assert ranges[0].get("data-lower") == "0.5" and ranges[0].get("data-upper") == "0.7"
    assert "no point is invented" in svg and "Accepted: 49/50" in svg
    paired = ET.fromstring(render_differences(path))
    assert sum(r.tag.endswith("circle") for r in paired.iter()) == 3
    assert "one unknown charge" in render_differences(path)
    review["unknown_workflow_usage"] = 0
    path.with_name("review.json").write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown cost counts"):
        checked_record(path)


def test_boolean_is_not_a_price_even_when_equal_to_numeric_bounds(record):
    value = json.loads(record.read_bytes())
    value["all_trials"][0].update(cost_usd=True, cost_lower_usd=1, cost_upper_usd=1)
    record.write_text(json.dumps(value), encoding="utf-8")
    path = record.with_name("review.json")
    review = json.loads(path.read_bytes())
    review["analysis_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="finite number"):
        checked_record(record)
