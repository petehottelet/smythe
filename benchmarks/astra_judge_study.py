"""Judge a completed Astra main study, preserving shared campaign accounting."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from benchmarks import astra_runtime as runtime
from benchmarks.astra_campaign import load_task_pack
from benchmarks.astra_campaign._json import canonical
from benchmarks.astra_combined import load_continued_main
from benchmarks.astra_evaluation import MODEL, run_judgment
from benchmarks.astra_evidence import inspect_judgments
from benchmarks.astra_study import inspect_stage


def load_complete_main(directory):
    directory = Path(directory)
    summary = inspect_stage(directory)
    freeze = runtime._read(directory / "study-freeze.json")
    plain = dict(freeze)
    if plain.pop("freeze_sha256") != runtime._sha(plain):
        raise ValueError("Main freeze contents changed")
    if (summary["stage"] != "main" or summary["completed_workflows"] != 200
            or summary["planned_workflows"] != 200
            or freeze["human_calibration"]["status"] != "passed"):
        raise ValueError("Judging requires all 200 human-calibrated main outcomes")
    outcomes = [runtime._read(directory / f"{runtime._run_id(freeze, row)}.outcome.json")
                for row in freeze["schedule"]]
    return freeze, outcomes


def judge_main(*, main_directory, judge_directory, bindings_path, continuation_directory=None, api_key=None,
               live=False, progress=None):
    """Use one judge ledger for pilot and main; repeated invocations reuse receipts."""
    segment_summary = None
    if continuation_directory is None:
        freeze, outcomes = load_complete_main(main_directory)
    else:
        freeze, outcomes, segment_summary = load_continued_main(main_directory, continuation_directory)
    audited = inspect_judgments(judge_directory)
    config = runtime._read(Path(judge_directory) / "judge-freeze.json")
    allowance = freeze["campaign_allocations"]["judge_nanousd"]
    if config["allowance_nanousd"] != allowance or config["model"] != MODEL:
        raise ValueError("Judge allocation or identity differs from the approved campaign")
    available = [row for row in outcomes if row["output"] is not None]
    plan = {"main_freeze_sha256": freeze["freeze_sha256"], "outputs": len(available),
            "no_output_failures": len(outcomes) - len(available),
            "allowance_nanousd": allowance,
            "prior_judge_upper_nanousd": audited["cost_upper_nanousd"],
            "judge_model_version": config["model_version"], "live": live,
            "claimable": False}
    if segment_summary is not None:
        plan["continuation"] = segment_summary
    if not live:
        return plan
    if not api_key:
        raise ValueError("Live judging requires a configured Google API key")
    path = runtime._safe_path(bindings_path, file=True)
    if not path.parent.is_dir():
        raise ValueError("Judge binding destination parent must exist before paid work")
    if path.exists():
        saved = runtime._read(path)
        expected = {r["run_id"]: r["output_sha256"] for r in available}
        if (type(saved) is not list or len(saved) != len(expected)
                or {r.get("run_id"): r.get("output_sha256") for r in saved} != expected):
            raise ValueError("Saved main judgment bindings differ")
    cases = {c.task_id: c for c in load_task_pack().tasks}
    bindings = []
    # Order depends on anonymous task/output content, never model or strategy.
    available.sort(key=lambda r: hashlib.sha256(("astra-judge-14173" + r["trial"]["task_id"] + r["output"]).encode()).digest())
    for row in available:
        scored = run_judgment(cases[row["trial"]["task_id"]], row["output"],
            directory=judge_directory, allowance_nanousd=allowance,
            api_key=api_key, model_version=config["model_version"])
        binding = {"run_id": row["run_id"], "output_sha256": row["output_sha256"],
                   "judgment_identity": scored["identity"],
                   "judgment_record_sha256": scored["record_sha256"]}
        bindings.append(binding)
        if progress is not None:
            progress({"outputs_scored": len(bindings), "outputs_available": len(available),
                      "scores": [r["score"] for r in scored["scores"]["criteria"]],
                      "material_defects": len(scored["scores"]["material_defects"])})
    bindings.sort(key=lambda r: r["run_id"])
    if path.exists():
        if runtime._read(path) != bindings:
            raise ValueError("Saved main judgment bindings differ")
    else:
        runtime._write_new(path, bindings)
    final_audit = inspect_judgments(judge_directory)
    return {**plan, "status": "scored", "bindings_sha256": runtime._sha(bindings),
            "judge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "total_paid_judgments": final_audit["paid_judgments"],
            "total_judge_lower_nanousd": final_audit["cost_lower_nanousd"],
            "total_judge_upper_nanousd": final_audit["cost_upper_nanousd"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-directory", required=True)
    parser.add_argument("--continuation-directory")
    parser.add_argument("--judge-directory", required=True)
    parser.add_argument("--bindings-path", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    result = judge_main(main_directory=args.main_directory, judge_directory=args.judge_directory,
                        continuation_directory=args.continuation_directory,
                        bindings_path=args.bindings_path, live=args.live,
                        api_key=os.environ.get("GOOGLE_API_KEY") if args.live else None,
                        progress=lambda row: print(canonical(row), flush=True))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
