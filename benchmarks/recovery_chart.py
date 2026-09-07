"""Render the matched durability v2 recovery comparison from committed records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.render_readme_charts import BLACK, MONO, SERIF, TRAJAN, _bar, _svg, _text


RECORD_PATH = Path(__file__).resolve().parent / "results/durability_kill_resume_v2.json"
FRAMEWORKS = ("smythe", "langgraph")


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Recovery record has invalid {name}")
    return value


def validate_recovery(record: dict) -> dict[str, list[dict]]:
    """Require the complete, matched kill profile before emitting any claim."""
    if not isinstance(record, dict) or record.get("benchmark_schema") != "smythe.durability-benchmark.v2":
        raise ValueError("Recovery chart requires durability benchmark v2")
    if (record.get("claimable") is False or record.get("status") in {"failed", "invalid", "superseded"}
            or record.get("evidence_status") in {"diagnostic", "invalid", "superseded"}
            or record.get("known_measurement_defects") or record.get("invalidation_reason")):
        raise ValueError("Recovery evidence is not claimable")
    protocol = record.get("protocol", {}).get("cell_b", {})
    expected = {"n": 64, "latency_ms": 100, "concurrency": 8, "kill_at_dispatches": 32, "reps": 3}
    for field, value in expected.items():
        if _integer(protocol.get(field), f"protocol {field}", minimum=1) != value:
            raise ValueError(f"Recovery chart requires the matched v2 {field} profile")
    if (protocol.get("call_event_schema") != "smythe.durability-call-event.v1"
            or protocol.get("call_event_fsync") is not True
            or protocol.get("smythe") != "FileCheckpointStore, checkpoint_every_n_nodes=1"
            or protocol.get("langgraph") != "AsyncSqliteSaver, durability='sync'"):
        raise ValueError("Recovery chart requires durable dispatch logging and matched persistence")
    rows = record.get("cell_b_durability")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("Recovery chart requires three complete repetitions per framework")
    cells = {}
    n, kill = protocol["n"], protocol["kill_at_dispatches"]
    expected_ids = {f"n{index:04d}" for index in range(n)}
    for row in rows:
        if not isinstance(row, dict) or row.get("framework") not in FRAMEWORKS:
            raise ValueError("Recovery record contains an unknown framework")
        framework = row["framework"]
        rep = _integer(row.get("rep"), "repetition")
        if rep >= protocol["reps"] or (framework, rep) in cells:
            raise ValueError("Recovery record contains duplicate or unmatched repetitions")
        cells[framework, rep] = row
        for field in ("n", "dispatches_at_kill", "completions_at_kill", "inflight_attempts_at_kill",
                      "total_dispatches", "total_completions", "duplicate_dispatches", "resume_completed"):
            _integer(row.get(field), field)
        if row["n"] != n or row["resume_completed"] != n or row.get("error"):
            raise ValueError("Recovery record contains incomplete resumed work")
        if (row["dispatches_at_kill"] != kill
                or row["completions_at_kill"] + row["inflight_attempts_at_kill"] != kill
                or not 0 < row["inflight_attempts_at_kill"] <= protocol["concurrency"]):
            raise ValueError("Recovery record has an inconsistent kill point")
        duplicates = row["duplicate_dispatches"]
        if (duplicates > kill or row["total_dispatches"] != n + duplicates
                or row["total_completions"] != row["total_dispatches"] - row["inflight_attempts_at_kill"]
                or row["total_completions"] < n):
            raise ValueError("Recovery dispatch/completion totals disagree")
        replayed = row.get("replayed_operation_ids")
        if (not isinstance(replayed, list) or any(not isinstance(item, str) for item in replayed)
                or len(set(replayed)) != len(replayed) or len(replayed) != duplicates
                or not set(replayed).issubset(expected_ids)):
            raise ValueError("Recovery duplicate count disagrees with the replayed operation inventory")
        durability = "file/every=1" if framework == "smythe" else "sync"
        if row.get("durability") != durability:
            raise ValueError("Recovery framework used mismatched persistence")
        elapsed = row.get("resume_wall_s")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise ValueError("Recovery record has invalid resume timing")
    expected_cells = {(framework, rep) for framework in FRAMEWORKS for rep in range(protocol["reps"])}
    if set(cells) != expected_cells:
        raise ValueError("Recovery record is missing a matched repetition")
    groups = {framework: [cells[framework, rep] for rep in range(protocol["reps"])] for framework in FRAMEWORKS}
    if statistics.mean(row["duplicate_dispatches"] for row in groups["langgraph"]) <= 0:
        raise ValueError("Recovery comparison requires a positive repeated-dispatch baseline")
    return groups


def render_recovery() -> str:
    """Render each actual repetition and its derived mean dispatch reduction."""
    # Git checkouts can translate CRLF without changing any evidence. Bind the
    # chart to the LF source representation used by the committed JSON blob.
    source = RECORD_PATH.read_bytes().replace(b"\r\n", b"\n")
    record = json.loads(source)
    groups = validate_recovery(record)
    means = {framework: statistics.mean(row["duplicate_dispatches"] for row in rows)
             for framework, rows in groups.items()}
    reduction = 1 - means["smythe"] / means["langgraph"]
    direction = "fewer" if reduction > 0 else "more" if reduction < 0 else "equal"
    comparison = f"{direction.upper()} DUPLICATE DISPATCHES"
    body = _text(40, 42, "DURABLE EXECUTION", size=11, weight="700", tracking=2.2)
    body += _text(40, 82, "Work repeated after a hard kill", size=31, family=SERIF, weight="700")
    body += _text(40, 109, "64-node fan-out; concurrency 8; hard process kill at 32 durable dispatches", size=13)
    body += f'<line x1="40" y1="132" x2="920" y2="132" stroke="{BLACK}" stroke-width="2"/>\n'
    body += _text(40, 206, f"{abs(reduction):.0%}", size=64, family=TRAJAN, weight="700")
    body += _text(285, 175, comparison, size=14, weight="700", tracking=.7)
    body += _text(285, 201, f'{means["smythe"]:g} vs {means["langgraph"]:g} mean repeated dispatches after resume', size=14)
    body += _text(285, 225, "All 64 operations completed after resume in every run", size=12)
    body += f'<line x1="40" y1="251" x2="920" y2="251" stroke="{BLACK}"/>\n'
    maximum = max(row["duplicate_dispatches"] for rows in groups.values() for row in rows)
    for rep in range(3):
        x = 40 + 305 * rep
        body += _text(x, 286, f"Repetition {rep + 1}", size=18, family=SERIF, weight="700")
        body += _text(x, 308, "DUPLICATE DISPATCHES / LOWER IS BETTER", size=8.5, weight="700", tracking=.4)
        for index, (framework, label, style) in enumerate((
            ("smythe", "Smythe", "solid"), ("langgraph", "LangGraph", "outline"),
        )):
            y = 340 + 59 * index
            value = groups[framework][rep]["duplicate_dispatches"]
            body += _text(x, y, label, size=13, weight="600")
            body += _text(x + 260, y, str(value), size=14, anchor="end", family=MONO)
            body += f'<g data-framework="{framework}" data-repetition="{rep + 1}" data-dispatches="{value}">\n'
            body += _bar(x, y + 11, 260 * value / maximum, style)
            body += '</g>\n'
        if rep < 2:
            body += f'<line x1="{x + 282}" y1="273" x2="{x + 282}" y2="429" stroke="{BLACK}"/>\n'
    body += f'<line x1="40" y1="453" x2="920" y2="453" stroke="{BLACK}"/>\n'
    body += _text(40, 480, "3 repetitions per framework; 100 ms simulated calls; durable dispatch logging", size=11.5)
    body += _text(40, 502, "Legacy Swarm file checkpoints vs LangGraph synchronous SQLite checkpoints", size=11.5)
    body += _text(40, 524, "Offline process-kill profile; provider billing unmeasured", size=11.5)
    body += _text(920, 550, RECORD_PATH.name, size=11, anchor="end", family=MONO)
    body += _text(40, 550, "SOURCE RECORD", size=10, weight="700", tracking=1)
    body += f'<!-- source-sha256: {hashlib.sha256(source).hexdigest()} -->\n'
    return _svg(960, 572, body, label=(
        f'Smythe repeated {means["smythe"]:g} dispatches versus LangGraph {means["langgraph"]:g} '
        f'after a hard process kill: {abs(reduction):.0%} {direction} mean duplicate dispatches across three repetitions. '
        'All runs resumed 64 of 64 operations. Offline simulation; provider billing unmeasured.'))
