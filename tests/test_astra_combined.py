"""Union and reservation checks; fixtures contain no provider responses."""

from copy import deepcopy

import pytest

from benchmarks import astra_combined as combined, astra_runtime as runtime


@pytest.fixture
def segments(tmp_path, monkeypatch):
    original, later = tmp_path / "original", tmp_path / "later"
    original.mkdir()
    later.mkdir()
    schedule = [{"trial_id": str(i)} for i in range(200)]
    base = {"stage": "main", "schedule": schedule, "human_calibration": {"status": "passed"},
            "stage_allowance_nanousd": 200_000_000_000}
    base["freeze_sha256"] = runtime._sha(base)
    prior = {"completed_workflows": 147, "confirmed_nanousd": 1000, "unknown_nanousd": 200,
             "unresolved_calls": [{"run_id": "146", "held_nanousd": 200}]}
    envelope = {"base": base, "base_freeze_sha256": base["freeze_sha256"], "prior": prior,
                "schedule": schedule[147:], "policy": dict(combined.POLICY), "source_sha256": "fixture"}
    envelope["freeze_sha256"] = runtime._sha(envelope)
    rows = [{"run_id": str(i), "trial": row, "record_sha256": str(i)} for i, row in enumerate(schedule)]
    for row in rows[147:]:
        row["continuation_freeze_sha256"] = envelope["freeze_sha256"]
    current = {"confirmed_nanousd": 500, "unknown_nanousd": 0, "unresolved_calls": [],
               "outcome_sha256": {r["run_id"]: r["record_sha256"] for r in rows[147:]}}
    summary = {"freeze_sha256": envelope["freeze_sha256"], "completed_workflows": 53,
        "confirmed_nanousd": 500, "prior_confirmed_nanousd": 1000, "held_unknown_nanousd": 200,
        "claimable": False, "outcome_sha256": current["outcome_sha256"]}
    runtime._write_new(original / "study-freeze.json", base)
    runtime._write_new(later / "continuation-freeze.json", envelope)
    runtime._write_new(later / "continuation-summary.json", summary)
    state = {"prior": deepcopy(prior), "current": current, "earlier": rows[:147], "later": rows[147:]}

    def audit(directory, audited_base, audited_schedule):
        assert audited_base == base
        if directory == original:
            assert audited_schedule == schedule
            return state["earlier"], state["prior"]
        assert directory == later and audited_schedule == schedule[147:]
        return state["later"], state["current"]

    monkeypatch.setattr(combined, "audit_segment", audit)
    return original, later, state


def test_union_preserves_every_run_and_holds_full_cost_range(segments):
    original, later, _ = segments
    base, rows, summary = combined.load_continued_main(original, later)
    assert [r["trial"] for r in rows] == base["schedule"]
    assert summary["segments"] == [147, 53]
    assert summary["cost_lower_nanousd"] == 1500 and summary["cost_upper_nanousd"] == 1700
    assert summary["unknown_nanousd"] == 200 and not summary["claimable"]


@pytest.mark.parametrize("change", ["prior-reserve", "missing", "duplicate", "new-unknown", "binding", "summary", "freeze"])
def test_changed_or_incomplete_union_is_rejected(segments, change):
    original, later, state = segments
    if change == "prior-reserve":
        state["prior"]["unknown_nanousd"] = 0
    elif change == "missing":
        state["later"].pop()
    elif change == "duplicate":
        state["later"][-1] = state["later"][0]
    elif change == "new-unknown":
        state["current"]["unknown_nanousd"] = 1
    elif change == "binding":
        state["later"][0]["continuation_freeze_sha256"] = "changed"
    else:
        path = later / ("continuation-summary.json" if change == "summary" else "continuation-freeze.json")
        path.write_bytes(path.read_bytes().replace(b'"claimable":false', b'"claimable":true')
                         if change == "summary" else path.read_bytes().replace(b'"fixture"', b'"changed"'))
    with pytest.raises(ValueError):
        combined.load_continued_main(original, later)


def test_same_directory_cannot_supply_both_segments(segments):
    original, _, _ = segments
    with pytest.raises(ValueError, match="distinct"):
        combined.load_continued_main(original, original)
