"""Recovery chart evidence must be complete, internally consistent, and matched."""

from __future__ import annotations

import hashlib
import json
import re
from xml.etree import ElementTree as ET

import pytest

from benchmarks import recovery_chart as chart


NS = {"svg": "http://www.w3.org/2000/svg"}


@pytest.fixture
def evidence():
    return json.loads(chart.RECORD_PATH.read_bytes())


def render_fixture(monkeypatch, tmp_path, record):
    path = tmp_path / "durability_kill_resume_v2.json"
    path.write_text(json.dumps(record), encoding="utf-8", newline="\n")
    monkeypatch.setattr(chart, "RECORD_PATH", path)
    return chart.render_recovery()


def test_current_recovery_chart_uses_all_six_committed_samples_and_zero_based_bars():
    source = chart.RECORD_PATH.read_bytes()
    document = chart.render_recovery()
    assert document == chart.render_recovery()
    root = ET.fromstring(document)
    text = " ".join(root.itertext())
    assert "75%" in text
    assert "8 vs 32 mean repeated dispatches" in text
    assert "3 repetitions per framework" in text
    assert "64-node fan-out; concurrency 8; hard process kill at 32 durable dispatches" in text
    assert "100 ms simulated calls" in text
    assert "Legacy Swarm" in text
    assert "provider billing unmeasured" in text
    assert chart.RECORD_PATH.name in text
    assert hashlib.sha256(source).hexdigest() in document
    assert set(re.findall(r"#[0-9a-fA-F]{6}", document)) == {"#000000", "#ffffff"}
    assert not any("opacity" in key for node in root.iter() for key in node.attrib)
    bars = root.findall("svg:g[@data-framework]", NS)
    assert len(bars) == 6
    for rep in (1, 2, 3):
        selected = {node.attrib["data-framework"]: node for node in bars if node.attrib["data-repetition"] == str(rep)}
        smythe, langgraph = (selected[name].find("svg:rect", NS) for name in ("smythe", "langgraph"))
        assert selected["smythe"].attrib["data-dispatches"] == "8"
        assert selected["langgraph"].attrib["data-dispatches"] == "32"
        assert smythe.attrib["x"] == langgraph.attrib["x"]  # Shared zero origin.
        assert float(smythe.attrib["width"]) / float(langgraph.attrib["width"]) == .25
        assert smythe.attrib["fill"] == "#000000"
        assert langgraph.attrib["fill"] == "#ffffff"
        assert langgraph.attrib["stroke"] == "#000000"


def test_chart_derives_reduction_and_mark_widths_from_raw_rows(evidence, monkeypatch, tmp_path):
    for row in evidence["cell_b_durability"]:
        if row["framework"] == "smythe":
            row.update(duplicate_dispatches=16, total_dispatches=80, total_completions=72,
                       replayed_operation_ids=[f"n{index:04d}" for index in range(16, 32)])
    document = render_fixture(monkeypatch, tmp_path, evidence)
    root = ET.fromstring(document)
    assert "50%" in " ".join(root.itertext())
    assert "16 vs 32" in " ".join(root.itertext())
    for node in root.findall("svg:g[@data-framework='smythe']/svg:rect", NS):
        assert float(node.attrib["width"]) == 130


@pytest.mark.parametrize("damage", [
    "missing_run", "duplicate_rep", "unexpected_rep", "unknown_framework", "incomplete_resume", "wrong_n",
    "wrong_kill", "wrong_inflight", "wrong_total", "wrong_completions", "wrong_duplicate_count",
    "duplicate_operation", "foreign_operation", "missing_operation", "row_persistence", "protocol_persistence",
    "missing_fsync", "different_protocol", "old_schema", "diagnostic", "defect", "boolean_count", "nan_time",
])
def test_chart_rejects_incomplete_or_mismatched_evidence(evidence, monkeypatch, tmp_path, damage):
    row = evidence["cell_b_durability"][0]
    protocol = evidence["protocol"]["cell_b"]
    if damage == "missing_run":
        evidence["cell_b_durability"].pop()
    elif damage == "duplicate_rep":
        evidence["cell_b_durability"][1]["rep"] = 0
    elif damage == "unexpected_rep":
        row["rep"] = 3
    elif damage == "unknown_framework":
        row["framework"] = "unmatched"
    elif damage == "incomplete_resume":
        row["resume_completed"] = 63
    elif damage == "wrong_n":
        row["n"] = 65
    elif damage == "wrong_kill":
        row["dispatches_at_kill"] = 31
    elif damage == "wrong_inflight":
        row["inflight_attempts_at_kill"] = 7
    elif damage == "wrong_total":
        row["total_dispatches"] = 71
    elif damage == "wrong_completions":
        row["total_completions"] = 63
    elif damage == "wrong_duplicate_count":
        row["duplicate_dispatches"] = 7
    elif damage == "duplicate_operation":
        row["replayed_operation_ids"][1] = row["replayed_operation_ids"][0]
    elif damage == "foreign_operation":
        row["replayed_operation_ids"][0] = "n9999"
    elif damage == "missing_operation":
        row["replayed_operation_ids"].pop()
    elif damage == "row_persistence":
        row["durability"] = "default"
    elif damage == "protocol_persistence":
        protocol["langgraph"] = "AsyncSqliteSaver, durability='async'"
    elif damage == "missing_fsync":
        protocol["call_event_fsync"] = False
    elif damage == "different_protocol":
        protocol["concurrency"] = 16
    elif damage == "old_schema":
        evidence["benchmark_schema"] = "smythe.durability-benchmark.v1"
    elif damage == "diagnostic":
        evidence["evidence_status"] = "diagnostic"
    elif damage == "defect":
        evidence["known_measurement_defects"] = ["incomplete event accounting"]
    elif damage == "boolean_count":
        row["duplicate_dispatches"] = True
    elif damage == "nan_time":
        row["resume_wall_s"] = float("nan")
    with pytest.raises(ValueError, match="Recovery|recovery"):
        render_fixture(monkeypatch, tmp_path, evidence)
