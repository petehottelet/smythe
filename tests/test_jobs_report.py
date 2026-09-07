"""Inert HTML, exact accounting, and operator context in Jobs reports."""

from copy import deepcopy
from decimal import localcontext
from html.parser import HTMLParser
import re

import pytest

from smythe.jobs.report import render_job_report


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags, self.text = [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)

    @property
    def content(self):
        return " ".join(self.text)


def payload():
    return {
        "inspection_version": 1, "run_id": "job-fixture", "name": "Glyph batch",
        "status": "needs_attention", "created_at_ns": 1_788_796_800_000_000_001,
        "updated_at_ns": 1_788_796_802_000_000_002, "inspected_at_ns": 1_788_796_803_000_000_003,
        "manifest_hash": "a" * 64, "plan_hash": "b" * 64,
        "manifest_root": "C:/Work/source", "output_directory": "artifacts/jobs", "max_concurrency": 4,
        "counts": {"succeeded": 27, "rejected": 2, "unknown_outcome": 1},
        "cost": {"approved_microusd": 20_000_000, "confirmed_microusd": 3_750_001,
                 "exposure_microusd": 250_000, "reserved_microusd": 0,
                 "cost_is_complete": False, "cost_contains_estimates": True},
        "operations": [{"operation_id": "op-a", "operation_key": "tile-17", "status": "succeeded",
                        "attempt_count": 2, "max_attempts": 3, "accepted_attempt_id": "attempt-b",
                        "result_text": "Selected output", "error": None,
                        "spec": {"prompt": "Draw an original glyph", "model": "offline"}}],
        "attempts": [
            {"attempt_id": "attempt-a", "operation_id": "op-a", "attempt_number": 1,
             "parent_attempt_id": None, "reason": "initial", "status": "rejected",
             "result_text": "First output", "error": "Width does not meet acceptance rule",
             "started_at_ns": 1_788_796_800_000_000_001, "completed_at_ns": 1_788_796_801_000_000_002},
            {"attempt_id": "attempt-b", "operation_id": "op-a", "attempt_number": 2,
             "parent_attempt_id": "attempt-a", "reason": "manual_reroll", "status": "succeeded",
             "result_text": "Second output", "error": None,
             "started_at_ns": 1_788_796_802_000_000_001, "completed_at_ns": 1_788_796_803_000_000_002},
        ],
        "calls": [{"call_id": "call-a", "attempt_id": "attempt-a", "operation_id": "op-a",
                   "status": "completed", "idempotency_key": "local-key", "ceiling_microusd": 250_000,
                   "confirmed_microusd": 123_456, "exposure_microusd": 0,
                   "cost_is_complete": 1, "cost_is_estimate": 0, "provider_request_id": "req_a",
                   "error": None, "created_at_ns": 1_788_796_800_000_000_001,
                   "dispatched_at_ns": 1_788_796_800_000_000_002,
                   "completed_at_ns": 1_788_796_801_000_000_001}],
        "artifacts": [{"artifact_id": "artifact-a", "operation_id": "op-a", "attempt_id": "attempt-b",
                       "relative_path": "glyphs/17.svg", "mime_type": "image/svg+xml", "sha256": "c" * 64,
                       "size_bytes": 1234, "width": 128, "height": 128, "accepted": 1,
                       "integrity": {"status": "changed", "detail": "SHA-256 differs from the recorded artifact",
                                     "checked_at_ns": 1_788_796_803_000_000_003}}],
        "artifact_integrity": {"counts": {"changed": 1}, "bytes_hashed": 1234,
                               "hash_byte_limit": 67_108_864, "scope": "selected operation page"},
        "events": [
            {"sequence": 17, "event_type": "call_dispatched", "operation_id": "op-a",
             "created_at_ns": 1_788_796_800_000_000_002, "payload": {"call_id": "call-a"}},
            {"sequence": 18, "event_type": "attempt_rejected", "operation_id": "op-a",
             "created_at_ns": 1_788_796_801_000_000_001, "payload": {"finding": "Width too small"}},
        ],
        "pagination": {"limit": 1, "offset": 16, "total": 30, "returned": 1, "has_more": True},
        "event_pagination": {"limit": 2, "total": 18, "returned": 2, "has_more": True},
        "operation_filter": "tile-*",
    }


def test_whole_run_costs_and_counts_are_not_recomputed_from_page():
    document = Document(render_job_report(payload()))
    assert "$3.750001" in document.content
    assert "$0.123456" in document.content
    assert "27" in document.content and "Unknown outcome" in document.content
    assert "Showing 17–17 of 30 matching operations." in document.content
    assert "cover the entire run" in document.content
    assert "More operations are available" in document.content
    assert "Operation filter:" in document.content and "tile-*" in document.content
    assert "Recorded cost complete: No. Includes estimates: Yes." in document.content


def test_every_attempt_reroll_parent_spec_response_and_call_is_present():
    document = Document(render_job_report(payload()))
    for expected in ("attempt-a", "attempt-b", "manual_reroll", "Parent attempt", "No parent",
                     "Draw an original glyph", "Selected output", "First output", "Second output",
                     "Width does not meet acceptance rule", "call-a", "req_a", "local-key"):
        assert expected in document.content
    assert "Complete specification" in document.content and "Prompt" in document.content
    assert "Attempt error / acceptance finding" in document.content
    assert sum(tag == "details" for tag, _ in document.tags) >= 10
    assert all("scope" in attrs for tag, attrs in document.tags if tag == "th")
    assert sum(tag == "caption" for tag, _ in document.tags) == 6


def test_attempts_and_calls_show_human_operation_key_and_keep_exact_ids():
    html = render_job_report(payload())
    document = Document(html)
    assert "tile-17 · Attempt 1" in document.content
    assert '<div class="cell-value">Attempt 1</div>' in html
    assert '<div class="cell-value"><code>tile-17</code></div>' in html
    assert "Operation ID" in document.content and "op-a" in document.content
    assert "Attempt ID" in document.content and "attempt-a" in document.content
    assert html.index("<summary>Prompt</summary>") < html.index("<summary>Complete specification</summary>")
    assert "<summary>Prompt</summary><pre tabindex=\"0\">Draw an original glyph</pre>" in html


def test_exact_microusd_uses_no_float_or_decimal_rounding():
    sample = payload()
    sample["cost"].update(approved_microusd=9_223_372_036_854_775_807,
                          confirmed_microusd=1, exposure_microusd=1_234_567, reserved_microusd=0)
    with localcontext() as context:
        context.prec = 2
        document = Document(render_job_report(sample))
    for exact in ("$9,223,372,036,854.775807", "$0.000001", "$1.234567", "$0.000000"):
        assert exact in document.content


@pytest.mark.parametrize("invalid", [True, False, 1.5, float("nan"), "1000000", -1])
def test_invalid_recorded_amount_is_not_coerced_to_money(invalid):
    sample = payload()
    sample["cost"]["confirmed_microusd"] = invalid
    document = Document(render_job_report(sample))
    assert "Invalid recorded amount" in document.content
    assert "$3.750001" not in document.content


def test_zero_pending_run_has_useful_empty_states_and_no_invented_usage():
    sample = {"name": "Approved job", "run_id": "new", "status": "approved", "counts": {"pending": 3},
              "cost": {"approved_microusd": 0, "confirmed_microusd": 0, "exposure_microusd": 0,
                       "reserved_microusd": 0, "cost_is_complete": True, "cost_contains_estimates": False}}
    document = Document(render_job_report(sample))
    assert "$0.000000" in document.content and "Pending" in document.content
    for expected in ("No operations match", "No attempts recorded", "No provider calls recorded",
                     "No artifacts recorded", "No events in this inspection window"):
        assert expected in document.content
    assert "token" not in document.content.lower()
    assert "Not recorded" in document.content


def test_html_and_attributes_from_every_source_are_inert():
    attack = '</pre><script src="https://evil.invalid/x">alert(1)</script><img src=x onerror="alert(2)">'
    sample = payload()
    sample.update(name=attack, run_id=attack, manifest_root=attack, output_directory=attack,
                  operation_filter=attack, manifest_hash=attack, plan_hash=attack, status=attack)
    sample["operations"][0].update(operation_key=attack, operation_id=attack, result_text=attack,
                                    error=attack, spec={attack: attack})
    sample["attempts"][0].update(attempt_id=attack, parent_attempt_id=attack, reason=attack,
                                 result_text=attack, error=attack)
    sample["calls"][0].update(call_id=attack, provider_request_id=attack, idempotency_key=attack, error=attack)
    sample["artifacts"][0].update(relative_path="javascript:alert(1)", mime_type=attack, sha256=attack)
    sample["artifacts"][0]["integrity"].update(status=attack, detail=attack)
    sample["events"][0].update(event_type=attack, operation_id=attack, payload={attack: attack})
    html = render_job_report(sample)
    document = Document(html)
    assert attack in document.content and "javascript:alert(1)" in document.content
    assert not {tag for tag, _ in document.tags} & {"script", "img", "svg", "iframe", "object", "embed", "a", "link", "form"}
    for _, attrs in document.tags:
        assert not any(key.lower().startswith("on") for key in attrs)
        assert not {"src", "href", "srcdoc", "action", "xlink:href"} & attrs.keys()
    assert "&lt;script" in html and "&lt;img" in html


def test_csp_and_assets_remain_self_contained_monochrome_and_keyboard_accessible():
    document = Document(render_job_report(payload()))
    policies = [attrs["content"] for tag, attrs in document.tags
                if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"]
    assert policies == ["default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"]
    html = render_job_report(payload())
    css = re.search(r"<style>(.*?)</style>", html, re.DOTALL).group(1)
    assert set(re.findall(r"#[0-9a-fA-F]{3,8}\b", css)) == {"#000000", "#ffffff"}
    assert "url(" not in css and "@import" not in css
    assert "focus-visible" in css and "::selection" in css
    assert "max-width: 700px" in css and "overflow-x: auto" in css
    assert any(attrs.get("role") == "region" and attrs.get("tabindex") == "0"
               for _, attrs in document.tags)
    assert all(attrs.get("lang") == "en" for tag, attrs in document.tags if tag == "html")


def test_mobile_rows_have_visible_labels_value_wrappers_and_full_width_records():
    html = render_job_report(payload())
    document = Document(html)
    cells = [attrs for tag, attrs in document.tags if tag == "td" or tag == "th" and attrs.get("scope") == "row"]
    assert cells and all(attrs.get("data-label") for attrs in cells)
    assert sum(attrs.get("class") == "cell-value" for _, attrs in document.tags) == len(cells)
    assert any(attrs.get("class") == "record-cell" for attrs in cells)
    assert all(attrs.get("role") == "table" for tag, attrs in document.tags if tag == "table")
    assert all(attrs.get("role") == "row" for tag, attrs in document.tags if tag == "tr")
    css = re.search(r"<style>(.*?)</style>", html, re.DOTALL).group(1)
    assert 'content: attr(data-label)' in css
    assert '.wide tbody .record-cell, .calls tbody .record-cell { display: block; }' in css
    assert '.wide details, .calls details { min-width: 0; }' in css


def test_header_counts_are_immediate_and_cost_flags_stay_with_cost_table():
    html = render_job_report(payload())
    header = html[html.index("<header>"):html.index("</header>")]
    header_text = Document(header).content
    assert "30 operations" in header_text and "Unknown outcome" in header_text
    assert "27" in header_text and "2" in header_text and "1" in header_text
    assert html.index("Recorded cost complete:") < html.index("<caption>All operations</caption>")


def test_recent_timeline_preserves_order_and_explains_its_window():
    document = Document(render_job_report(payload()))
    assert document.content.index("call_dispatched") < document.content.index("attempt_rejected")
    assert "Showing 2 of 18 recorded events." in document.content
    assert "Older events are outside this inspection window." in document.content
    assert "Width too small" in document.content
    times = [attrs for tag, attrs in document.tags if tag == "time"]
    assert any(attrs["datetime"].endswith(".000000003Z") for attrs in times)
    assert any(attrs["title"] == "1788796803000000003 ns" for attrs in times)


@pytest.mark.parametrize("status,label", [
    ("verified", "Verified"), ("missing", "Missing"), ("changed", "Changed"),
    ("unsafe", "Unsafe path"), ("unreadable", "Unreadable"), ("not_checked", "Not checked"),
])
def test_disk_integrity_never_relabels_original_acceptance(status, label):
    sample = payload()
    sample["artifacts"][0]["integrity"]["status"] = status
    sample["artifacts"][0]["accepted"] = 1
    document = Document(render_job_report(sample))
    assert label in document.content and "Recorded acceptance is unchanged" in document.content
    assert "content and image validation are not rerun" in document.content
    assert "67,108,864" in document.content and "1,234" in document.content
    assert "c" * 64 in document.content
    assert "128 × 128 pixels" in document.content
    assert not any(tag in {"img", "svg", "a"} for tag, _ in document.tags)
    assert sample["artifacts"][0]["accepted"] == 1


def test_renderer_is_deterministic_and_does_not_mutate_input():
    sample = payload()
    original = deepcopy(sample)
    assert render_job_report(sample) == render_job_report(sample)
    assert sample == original


@pytest.mark.parametrize("field,value", [
    ("operations", ["not an operation"]), ("attempts", None), ("calls", {}),
    ("cost", []), ("counts", False), ("pagination", []), ("event_pagination", "1"),
])
def test_malformed_record_shapes_fail_explicitly(field, value):
    sample = payload()
    sample[field] = value
    with pytest.raises(ValueError):
        render_job_report(sample)


def test_empty_page_keeps_nonzero_whole_run_totals():
    sample = payload()
    sample.update(operations=[], attempts=[], calls=[], artifacts=[])
    sample["pagination"].update(offset=99, returned=0, has_more=False)
    document = Document(render_job_report(sample))
    assert "Showing 0 of 30 matching operations." in document.content
    assert "$3.750001" in document.content and "27" in document.content
    assert "No operations match this inspection page." in document.content
