"""Pure, self-contained HTML for a bounded Jobs inspection snapshot.

Recorded text is always escaped. Artifact paths are text, never resources to
load. The renderer does not inspect files, run validation, or contact providers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from typing import Any


_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
_CSS = """
:root { color-scheme: light; color: #000000; background: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Avenir Next', 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px; line-height: 1.55; }
main { max-width: 1180px; margin: 0 auto; padding: 56px 40px 80px; }
header { border-top: 3px solid #000000; padding-top: 24px; margin-bottom: 48px; }
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; font-weight: 400;
  line-height: 1.15; text-wrap: balance; overflow-wrap: anywhere; }
h1 { font-size: clamp(2rem, 5vw, 3.75rem); letter-spacing: -.025em; margin: 0 0 18px; }
h2 { font-size: 1.85rem; margin: 0 0 12px; }
h3 { font-size: 1.35rem; margin: 28px 0 8px; }
p { max-width: 72ch; margin: 0 0 14px; overflow-wrap: anywhere; }
section { margin-top: 48px; padding-top: 20px; border-top: 1px solid #000000; }
section:first-of-type { margin-top: 0; }
.subtitle { font-size: 1.1rem; }
.meta, .note { font-size: .875rem; }
.state { font-weight: 700; }
.attention { text-decoration: underline; text-underline-offset: .2em; }
.overview { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: start; }
.overview > *, .cell-value { min-width: 0; }
.cost-group .note { margin-top: 14px; }
.table-scroll { max-width: 100%; overflow-x: auto; margin: 18px 0 0;
  scrollbar-color: #000000 #ffffff; scrollbar-width: thin; }
table { width: 100%; border-collapse: collapse; text-align: left; }
caption { text-align: left; font-weight: 700; padding: 0 0 10px; }
th, td { padding: 12px 12px 12px 0; border-bottom: 1px solid #000000;
  vertical-align: top; overflow-wrap: anywhere; }
th { font-weight: 600; }
thead th { font-size: .8rem; border-bottom: 2px solid #000000; }
tbody th { font-weight: 400; }
th:last-child, td:last-child { padding-right: 0; }
.wide { min-width: 640px; }
.calls { min-width: 850px; }
.costs td:last-child, .costs thead th:last-child { text-align: right; }
.amount { display: block; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
code, pre, .machine, .amount, time { font-family: ui-monospace, 'SFMono-Regular', Consolas,
  'Liberation Mono', monospace; font-size: .83rem; overflow-wrap: anywhere; }
code { white-space: pre-wrap; }
.machine { display: block; margin-top: 4px; }
pre { margin: 12px 0; white-space: pre-wrap; word-break: break-word;
  max-height: 32rem; overflow: auto; line-height: 1.55; scrollbar-color: #000000 #ffffff; }
details { min-width: 160px; }
details details { margin-top: 8px; }
summary { cursor: pointer; min-height: 44px; padding: 10px 2px;
  font-size: .875rem; font-weight: 600; }
summary:hover { text-decoration: underline; text-underline-offset: .2em; }
summary:focus-visible, .table-scroll:focus-visible, pre:focus-visible {
  outline: 2px solid #000000; outline-offset: 4px; }
dl { margin: 8px 0 16px; }
dt { font-size: .8rem; font-weight: 600; margin-top: 12px; }
dd { margin: 3px 0 0; overflow-wrap: anywhere; }
.empty { margin-top: 18px; padding: 16px 0; border-bottom: 1px solid #000000; }
.timeline { list-style: none; margin: 24px 0 0; padding: 0; }
.timeline li { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 28px;
  padding: 18px 0; border-bottom: 1px solid #000000; }
.timeline h3 { margin: 0 0 5px; font-family: inherit; font-size: 1rem; font-weight: 600; }
footer { margin-top: 48px; padding-top: 16px; border-top: 3px solid #000000; font-size: .8rem; }
::selection { color: #ffffff; background: #000000; }
@media (max-width: 700px) {
  main { padding: 28px 20px 48px; } header { margin-bottom: 32px; }
  section { margin-top: 36px; } .overview { grid-template-columns: 1fr; gap: 24px; }
  h2 { font-size: 1.55rem; } .timeline li { grid-template-columns: 1fr; gap: 8px; }
  .wide, .calls, .wide tbody, .calls tbody, .wide tbody tr, .calls tbody tr {
    display: block; width: 100%; min-width: 0; max-width: 100%; }
  .wide thead, .calls thead { position: absolute; width: 1px; height: 1px;
    overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .wide caption, .calls caption { display: block; }
  .wide tbody tr, .calls tbody tr { border-top: 1px solid #000000;
    padding: 12px 0 18px; margin-top: 12px; }
  .wide tbody th, .wide tbody td, .calls tbody th, .calls tbody td {
    display: grid; grid-template-columns: 7rem minmax(0, 1fr); gap: 12px;
    width: 100%; min-width: 0; padding: 8px 0; border-bottom: 0; }
  .wide tbody [data-label]::before, .calls tbody [data-label]::before {
    content: attr(data-label); font-size: .75rem; font-weight: 600; }
  .wide .cell-value, .calls .cell-value { grid-column: 2; }
  .wide tbody .record-cell, .calls tbody .record-cell { display: block; }
  .wide .record-cell::before, .calls .record-cell::before { display: block; margin-bottom: 4px; }
  .wide details, .calls details { min-width: 0; }
  .wide .amount, .calls .amount { white-space: normal; text-align: left; }
}
@media print {
  main { max-width: none; padding: 0; } .table-scroll, pre { overflow: visible; max-height: none; }
  .wide, .calls { min-width: 0; } th, td { font-size: 9pt; }
  section { break-inside: auto; } h2, caption { break-after: avoid; }
}
"""
_LABELS = {
    "approved": "Approved", "pending": "Pending", "running": "Running",
    "succeeded": "Succeeded", "completed": "Completed", "partial": "Partial",
    "failed": "Failed", "rejected": "Rejected", "unknown_outcome": "Unknown outcome",
    "needs_attention": "Needs attention", "budget_overrun": "Budget overrun",
    "prepared": "Prepared", "dispatched": "Dispatched", "verified": "Verified",
    "missing": "Missing", "changed": "Changed", "unsafe": "Unsafe path",
    "unreadable": "Unreadable", "not_checked": "Not checked",
}


def _text(value: Any, missing: str = "Not recorded") -> str:
    return escape(missing if value is None else str(value), quote=True)


def _code(value: Any) -> str:
    return f"<code>{_text(value)}</code>"


def _number(value: Any) -> str:
    return f"{value:,}" if type(value) is int and value >= 0 else "Not recorded"


def _money(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if type(value) is not int or value < 0:
        return "Invalid recorded amount"
    whole, fractional = divmod(value, 1_000_000)
    return f"${whole:,}.{fractional:06d}"


def _flag(value: Any) -> str:
    if value is True or type(value) is int and value == 1:
        return "Yes"
    if value is False or type(value) is int and value == 0:
        return "No"
    return "Not recorded"


def _state(value: Any) -> str:
    label = _LABELS.get(value, value) if isinstance(value, str) else value
    attention = value in {"failed", "rejected", "unknown_outcome", "needs_attention", "budget_overrun"} \
        if isinstance(value, str) else False
    return f'<span class="state{" attention" if attention else ""}">{_text(label)}</span>'


def _time(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if type(value) is not int or value < 0:
        return "Invalid recorded time"
    seconds, nanos = divmod(value, 1_000_000_000)
    try:
        date = datetime.fromtimestamp(seconds, timezone.utc)
    except (ValueError, OverflowError, OSError):
        return "Invalid recorded time"
    iso = date.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"
    return f'<time datetime="{iso}" title="{value} ns">{date:%Y-%m-%d %H:%M:%S} UTC</time>'


def _json(value: Any) -> str:
    try:
        return escape(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        return "Invalid recorded JSON"


def _details(label: str, value: Any, *, structured: bool = False, opened: bool = False) -> str:
    body = _json(value) if structured else _text(value, "No text recorded")
    if value == "":
        body = "(Empty text)"
    return (f'<details{" open" if opened else ""}><summary>{escape(label)}</summary>'
            f'<pre tabindex="0">{body}</pre></details>')


def _records(payload: dict, name: str) -> list[dict]:
    value = payload.get(name, [])
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f"{name} must be a list of record objects")
    return value


def _object(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    return value


def _table(caption: str, headers: list[str], rows: list[list[str]], *, wide: str = "") -> str:
    head = "".join(f'<th scope="col" role="columnheader">{escape(label)}</th>' for label in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            label = escape(headers[index], quote=True)
            record_class = ' class="record-cell"' if cell.startswith("<details>") else ""
            tag = "th" if index == 0 else "td"
            role = 'scope="row" role="rowheader"' if index == 0 else 'role="cell"'
            cells.append(f'<{tag} {role} data-label="{label}"{record_class}>'
                         f'<div class="cell-value">{cell}</div></{tag}>')
        body.append('<tr role="row">' + "".join(cells) + "</tr>")
    return (f'<div class="table-scroll" role="region" aria-label="{escape(caption)}" tabindex="0">'
            f'<table class="{wide}" role="table"><caption>{escape(caption)}</caption>'
            f'<thead role="rowgroup"><tr role="row">{head}</tr></thead>'
            f'<tbody role="rowgroup">{"".join(body)}</tbody></table></div>')


def _definitions(items: list[tuple[str, str]]) -> str:
    return "<dl>" + "".join(f"<dt>{escape(label)}</dt><dd>{value}</dd>" for label, value in items) + "</dl>"


def _operation_rows(operations: list[dict]) -> list[list[str]]:
    rows = []
    for operation in operations:
        identity = operation.get("operation_id")
        detail = _definitions([
            ("Operation ID", _code(identity)),
            ("Accepted attempt", _code(operation.get("accepted_attempt_id"))),
        ])
        specification = _object(operation.get("spec"), "Operation specification")
        detail += _details("Prompt", specification.get("prompt"))
        detail += _details("Complete specification", specification, structured=True)
        detail += _details("Selected response", operation.get("result_text"))
        if operation.get("error") is not None:
            detail += _details("Operation error", operation["error"], opened=True)
        rows.append([
            _code(operation.get("operation_key", identity)), _state(operation.get("status")),
            f'{_number(operation.get("attempt_count"))} / {_number(operation.get("max_attempts"))}',
            f"<details><summary>Inspect operation</summary>{detail}</details>",
        ])
    return rows


def render_job_report(payload: dict) -> str:
    """Render supplied inspection data as inert, escaped, standalone HTML."""
    if type(payload) is not dict:
        raise TypeError("Job report payload must be an object")
    operations, attempts, calls, artifacts, events = (
        _records(payload, key) for key in ("operations", "attempts", "calls", "artifacts", "events")
    )
    operation_keys = {operation.get("operation_id"): operation.get("operation_key", operation.get("operation_id"))
                      for operation in operations}
    attempt_numbers = {attempt.get("attempt_id"): attempt.get("attempt_number") for attempt in attempts}
    cost = _object(payload.get("cost"), "cost")
    counts = _object(payload.get("counts"), "counts")
    count_summary = " · ".join(f"{_number(value)} {_state(key)}" for key, value in counts.items())
    if counts and all(type(value) is int and value >= 0 for value in counts.values()):
        count_summary = f"{sum(counts.values()):,} operations · " + count_summary
    name = _text(payload.get("name"), "Untitled job")
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{escape(_CSP, quote=True)}">',
        f"<title>{name} — Smythe Jobs report</title><style>{_CSS}</style></head><body><main>",
        f"<header><h1>{name}</h1><p class=\"subtitle\">Jobs report · {_state(payload.get('status'))}</p>",
        f'<p class="run-counts">{count_summary or "No operations recorded."}</p>',
        f"<p>Run {_code(payload.get('run_id'))}</p>",
        f"<p class=\"meta\">Inspected {_time(payload.get('inspected_at_ns'))}</p></header>",
        '<section aria-labelledby="whole-run"><h2 id="whole-run">Whole run</h2>',
        '<p>Costs and operation counts cover the entire run, including operations outside this page.</p>',
        '<div class="overview"><div class="cost-group">',
        _table("Recorded cost (USD)", ["Account", "Amount"], [
            [label, f'<span class="amount">{_money(cost.get(key))}</span>'] for label, key in (
                ("Approved limit", "approved_microusd"), ("Confirmed cost", "confirmed_microusd"),
                ("Unresolved exposure", "exposure_microusd"), ("Reserved", "reserved_microusd"),
            )
        ], wide="costs"),
        f'<p class="note">Recorded cost complete: {_flag(cost.get("cost_is_complete"))}. '
        f'Includes estimates: {_flag(cost.get("cost_contains_estimates"))}.</p>',
        '<p class="note">Confirmed cost is recorded spend. Exposure and reservations are held amounts.</p>',
        '</div>',
    ]
    if counts:
        parts.append(_table("All operations", ["State", "Count"], [
            [_state(key), _number(value)] for key, value in counts.items()
        ]))
    else:
        parts.append('<p class="empty">No operations recorded.</p>')
    parts.extend([
        '</div>',
        '</section><section aria-labelledby="operations"><h2 id="operations">Operation detail</h2>',
    ])
    page = _object(payload.get("pagination"), "pagination")
    total, offset = page.get("total"), page.get("offset", 0)
    if type(total) is int and total >= 0 and type(offset) is int and offset >= 0:
        extent = f"{offset + 1:,}–{offset + len(operations):,}" if operations else "0"
        parts.append(f"<p>Showing {extent} of {total:,} matching operations.</p>")
    else:
        parts.append(f"<p>Showing {len(operations):,} operations in this view.</p>")
    if payload.get("operation_filter") is not None:
        parts.append(f'<p class="note">Operation filter: {_code(payload["operation_filter"])}</p>')
    if page.get("has_more") is True:
        parts.append('<p class="note">More operations are available in the next inspection page.</p>')
    parts.append(_table("Operations in this page", ["Operation", "State", "Attempts / limit", "Record"],
                        _operation_rows(operations), wide="wide") if operations
                 else '<p class="empty">No operations match this inspection page.</p>')
    parts.append('<h3>Attempts</h3><p class="note">Attempt history includes rerolls and parent attempts.</p>')
    selected = {op.get("accepted_attempt_id") for op in operations if op.get("accepted_attempt_id")}
    rows = []
    for attempt in attempts:
        detail = _definitions([
            ("Attempt ID", _code(attempt.get("attempt_id"))),
            ("Operation ID", _code(attempt.get("operation_id"))),
            ("Parent attempt", _code(attempt["parent_attempt_id"]) if attempt.get("parent_attempt_id") else "No parent"),
            ("Reason", _text(attempt.get("reason"))),
            ("Started", _time(attempt.get("started_at_ns"))),
            ("Completed", _time(attempt.get("completed_at_ns"))),
        ]) + _details("Attempt response", attempt.get("result_text"))
        if attempt.get("error") is not None:
            detail += _details("Attempt error / acceptance finding", attempt["error"], opened=True)
        rows.append([
            f'Attempt {_number(attempt.get("attempt_number"))}',
            _code(operation_keys.get(attempt.get("operation_id"), attempt.get("operation_id"))),
            _state(attempt.get("status")),
            "Yes" if attempt.get("attempt_id") in selected else "No",
            f"<details><summary>Inspect attempt</summary>{detail}</details>",
        ])
    parts.append(_table("Attempt history in this view", ["Attempt", "Operation", "State", "Selected", "Record"],
                        rows, wide="wide") if rows else '<p class="empty">No attempts recorded for these operations.</p>')
    parts.append('<h3>Provider calls</h3>')
    rows = []
    for call in calls:
        operation_key = operation_keys.get(call.get("operation_id"), call.get("operation_id"))
        attempt_number = attempt_numbers.get(call.get("attempt_id"))
        detail = _definitions([
            ("Attempt ID", _code(call.get("attempt_id"))),
            ("Operation key", _code(operation_key)),
            ("Operation ID", _code(call.get("operation_id"))),
            ("Provider request ID", _code(call.get("provider_request_id"))),
            ("Local idempotency key", _code(call.get("idempotency_key"))),
            ("Cost complete", _flag(call.get("cost_is_complete"))),
            ("Cost is an estimate", _flag(call.get("cost_is_estimate"))),
            ("Created", _time(call.get("created_at_ns"))),
            ("Dispatched", _time(call.get("dispatched_at_ns"))),
            ("Completed", _time(call.get("completed_at_ns"))),
        ])
        if call.get("error") is not None:
            detail += _details("Call error", call["error"], opened=True)
        rows.append([_code(call.get("call_id"))
                     + f'<span class="machine">{_text(operation_key)} · Attempt {_number(attempt_number)}</span>',
                     _state(call.get("status")), *[
            f'<span class="amount">{_money(call.get(key))}</span>' for key in (
                "ceiling_microusd", "confirmed_microusd", "exposure_microusd")
        ], f"<details><summary>Inspect call</summary>{detail}</details>"])
    parts.append(_table("Call accounting in this view (USD)", ["Call", "State", "Ceiling", "Confirmed", "Exposure", "Record"],
                        rows, wide="calls") if rows else '<p class="empty">No provider calls recorded for these operations.</p>')
    parts.append('<h3>Artifacts and disk integrity</h3><p class="note">Integrity is an observation of current disk bytes. '
                 'Recorded acceptance is unchanged; content and image validation are not rerun.</p>')
    integrity_summary = _object(payload.get("artifact_integrity"), "artifact_integrity")
    if "bytes_hashed" in integrity_summary:
        parts.append(f'<p class="note">Bytes hashed: {_number(integrity_summary.get("bytes_hashed"))}; '
                     f'limit: {_number(integrity_summary.get("hash_byte_limit"))}.</p>')
    rows = []
    for artifact in artifacts:
        integrity = _object(artifact.get("integrity"), "Artifact integrity")
        width, height = artifact.get("width"), artifact.get("height")
        dimensions = (f"{width:,} × {height:,} pixels"
                      if type(width) is int and width >= 0 and type(height) is int and height >= 0
                      else "Not recorded")
        detail = _definitions([
            ("Artifact ID", _code(artifact.get("artifact_id"))),
            ("Operation", _code(artifact.get("operation_id"))),
            ("Attempt", _code(artifact.get("attempt_id"))),
            ("MIME type", _code(artifact.get("mime_type"))),
            ("Recorded bytes", _number(artifact.get("size_bytes"))),
            ("Recorded dimensions", dimensions),
            ("Recorded SHA-256", _code(artifact.get("sha256"))),
            ("Disk observation", _text(integrity.get("detail"))),
            ("Checked", _time(integrity.get("checked_at_ns"))),
        ])
        rows.append([_code(artifact.get("relative_path")), _flag(artifact.get("accepted")),
                     _state(integrity.get("status", "not_checked")),
                     f"<details><summary>Inspect artifact</summary>{detail}</details>"])
    parts.append(_table("Recorded artifacts in this view", ["Recorded path", "Accepted", "Disk integrity", "Record"],
                        rows, wide="wide") if rows else '<p class="empty">No artifacts recorded for these operations.</p>')
    parts.append('</section><section aria-labelledby="events"><h2 id="events">Recent events</h2>')
    event_page = _object(payload.get("event_pagination"), "event_pagination")
    parts.append(f'<p>Showing {len(events):,} of {_number(event_page.get("total"))} recorded events.</p>')
    if event_page.get("has_more") is True:
        parts.append('<p class="note">Older events are outside this inspection window.</p>')
    if events:
        parts.append('<ol class="timeline">')
        for event in events:
            parts.append(f'<li><div>{_time(event.get("created_at_ns"))}<span class="machine">'
                         f'Sequence {_number(event.get("sequence"))}</span></div><div>'
                         f'<h3>{_text(event.get("event_type"))}</h3>'
                         f'<p class="note">Operation: {_code(event.get("operation_id"))}</p>'
                         f'{_details("Event data", event.get("payload"), structured=True)}</div></li>')
        parts.append('</ol>')
    else:
        parts.append('<p class="empty">No events in this inspection window.</p>')
    parts.append('</section><section aria-labelledby="provenance"><h2 id="provenance">Record provenance</h2>')
    parts.append(_definitions([
        ("Run ID", _code(payload.get("run_id"))),
        ("Manifest SHA-256", _code(payload.get("manifest_hash"))),
        ("Plan SHA-256", _code(payload.get("plan_hash"))),
        ("Manifest root", _code(payload.get("manifest_root"))),
        ("Output directory", _code(payload.get("output_directory"))),
        ("Maximum concurrency", _number(payload.get("max_concurrency"))),
        ("Created", _time(payload.get("created_at_ns"))),
        ("Last ledger update", _time(payload.get("updated_at_ns"))),
        ("Inspection version", _number(payload.get("inspection_version"))),
    ]))
    parts.append('</section><footer><p>Smythe Jobs · Static inspection report. '
                 'Includes recorded prompts, responses, errors, and local paths.</p></footer></main></body></html>')
    return "".join(parts)
