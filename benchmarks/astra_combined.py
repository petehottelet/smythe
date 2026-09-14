"""Offline reconciliation of an interrupted main study and its approved suffix.

Directories can be relocated: immutable source paths are provenance, never
used to find evidence. Every outcome is audited against its local native ledger.
"""

from pathlib import Path

from benchmarks import astra_runtime as runtime
from benchmarks.astra_continuation import POLICY, audit_segment, read_base


def load_continued_main(main_directory, continuation_directory):
    original, directory = Path(main_directory), Path(continuation_directory)
    if original.resolve() == directory.resolve():
        raise ValueError("The two evidence segments must have distinct directories")
    base = read_base(original)
    envelope = runtime._read(directory / "continuation-freeze.json")
    plain = dict(envelope)
    if (plain.pop("freeze_sha256") != runtime._sha(plain)
            or envelope["base"] != base or envelope["base_freeze_sha256"] != base["freeze_sha256"]
            or envelope["policy"] != POLICY):
        raise ValueError("Continuation freeze, policy or original study differs")
    if len(base["schedule"]) != 200 or base["human_calibration"]["status"] != "passed":
        raise ValueError("Expected the complete human-calibrated 200-workflow schedule")
    earlier, prior = audit_segment(original, base, base["schedule"])
    if (not 0 < len(earlier) < 200 or prior != envelope["prior"]
            or len(prior["unresolved_calls"]) != 1
            or envelope["schedule"] != base["schedule"][len(earlier):]):
        raise ValueError("Earlier outcomes, retained reserve or continuation schedule differs")
    later, current = audit_segment(directory, base, envelope["schedule"])
    if (len(later) != len(envelope["schedule"]) or current["unknown_nanousd"]
            or current["unresolved_calls"]):
        raise ValueError("A complete continuation with no new unknown charge is required")
    if any(r.get("continuation_freeze_sha256") != envelope["freeze_sha256"] for r in later):
        raise ValueError("Outcome belongs to a different continuation")
    expected_summary = {"freeze_sha256": envelope["freeze_sha256"],
        "completed_workflows": len(later), "confirmed_nanousd": current["confirmed_nanousd"],
        "prior_confirmed_nanousd": prior["confirmed_nanousd"],
        "held_unknown_nanousd": prior["unknown_nanousd"], "claimable": False,
        "outcome_sha256": current["outcome_sha256"]}
    if runtime._read(directory / "continuation-summary.json") != expected_summary:
        raise ValueError("Continuation summary differs from its native evidence")
    outcomes = earlier + later
    if (len({r["run_id"] for r in outcomes}) != 200
            or [r["trial"] for r in outcomes] != base["schedule"]):
        raise ValueError("The segment union must contain every scheduled outcome exactly once")
    confirmed = prior["confirmed_nanousd"] + current["confirmed_nanousd"]
    if confirmed + prior["unknown_nanousd"] > base["stage_allowance_nanousd"]:
        raise ValueError("Confirmed charges and the entire reserve exceed the main allowance")
    summary = {"status": "complete-with-retained-unknown", "claimable": False,
        "workflows": 200, "segments": [len(earlier), len(later)],
        "confirmed_nanousd": confirmed, "unknown_nanousd": prior["unknown_nanousd"],
        "cost_lower_nanousd": confirmed, "cost_upper_nanousd": confirmed + prior["unknown_nanousd"],
        "continuation_freeze_sha256": envelope["freeze_sha256"],
        "continuation_source_sha256": envelope["source_sha256"],
        "unresolved_calls": prior["unresolved_calls"],
        "outcome_sha256": {r["run_id"]: r["record_sha256"] for r in outcomes}}
    return base, outcomes, summary
