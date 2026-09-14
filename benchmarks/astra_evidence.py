"""Offline reconciliation of saved Astra workflows and blind judge receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks import astra_evaluation as evaluator
from benchmarks import astra_runtime as pilot
from benchmarks.astra_campaign import load_task_pack
from benchmarks.astra_campaign._json import strict_json


def inspect_judgments(directory):
    """Validate every attempt and return a conservative campaign charge bound."""
    directory = Path(directory)
    config = pilot._read(directory / "judge-freeze.json")
    if (config["model"] != evaluator.MODEL or config["price"] != evaluator.PRICE
            or config["source_sha256"] != hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest()):
        raise ValueError("Judge implementation or price freeze differs")
    started = {p.name.removesuffix(".started.json"): p for p in directory.glob("*.started.json")}
    for suffix in (".result.json", ".raw.json", ".quote.json"):
        if {p.name.removesuffix(suffix) for p in directory.glob("*" + suffix)} != set(started):
            raise ValueError("Judge has unresolved or unbound evidence")
    lower = upper = 0
    records = {}
    for identity, path in sorted(started.items()):
        attempt = pilot._read(path)
        expected = evaluator._hash({"request": attempt["request"], "criteria": attempt["criterion_ids"], "config": config})
        if identity != expected or attempt["identity"] != identity:
            raise ValueError("Judge request identity changed")
        record = pilot._read(directory / f"{identity}.result.json")
        unsigned = dict(record)
        if unsigned.pop("record_sha256") != evaluator._hash(unsigned):
            raise ValueError("Judge result hash changed")
        raw_path = pilot._safe_path(directory / f"{identity}.raw.json", file=True)
        body = raw_path.read_bytes()
        raw = strict_json(body.decode())
        quote_path = pilot._safe_path(directory / f"{identity}.quote.json", file=True)
        quote = quote_path.read_bytes()
        if (hashlib.sha256(quote).hexdigest() != attempt["quote_raw_sha256"]
                or strict_json(quote.decode())["totalTokens"] != attempt["quoted_input_tokens"]):
            raise ValueError("Judge token quote changed")
        accounting = evaluator.price_judge_response(raw)
        scores = evaluator.decode_scores(raw, attempt["criterion_ids"])
        if (record["identity"] != identity or record["request_sha256"] != evaluator._hash(attempt["request"])
                or record["raw_sha256"] != hashlib.sha256(body).hexdigest()
                or record["model_version"] != config["model_version"]
                or raw.get("modelVersion") != config["model_version"]
                or record["accounting"] != accounting or record["scores"] != scores
                or accounting["cost_upper_nanousd"] > attempt["reservation_nanousd"]):
            raise ValueError("Judge native evidence differs from its result")
        lower += accounting["cost_lower_nanousd"]
        upper += accounting["cost_upper_nanousd"]
        records[identity] = {"attempt": attempt, "result": record}
    if upper > config["allowance_nanousd"]:
        raise ValueError("Judge allowance exceeded")
    return {"paid_judgments": len(records), "cost_lower_nanousd": lower,
            "cost_upper_nanousd": upper, "unresolved_attempts": 0, "records": records,
            "judge_freeze_sha256": evaluator._hash(config)}


def inspect_calibration(*, original, directory, bindings_path, manifest_path, control_path):
    """Bind all pilot judgments and a deliberately defective control to inputs."""
    audit = inspect_judgments(directory)
    cases = {c.task_id: c for c in load_task_pack().tasks}
    bindings = pilot._read(bindings_path)
    by_run = {r["run_id"]: r for r in bindings}
    if len(bindings) != 12 or len(by_run) != 12 or set(by_run) != {r["run_id"] for r in original["outcomes"]}:
        raise ValueError("Every original pilot output must have a bound judgment")
    for outcome in original["outcomes"]:
        binding = by_run[outcome["run_id"]]
        saved = audit["records"][binding["judgment_identity"]]
        result = saved["result"]
        if (binding["output_sha256"] != outcome["output_sha256"]
                or binding["judgment_record_sha256"] != result["record_sha256"]
                or saved["attempt"]["request"] != evaluator.judge_request(cases[outcome["trial"]["task_id"]], outcome["output"])):
            raise ValueError("Pilot judgment differs from its anonymous source answer")
        if result["scores"]["material_defects"] or any(r["score"] < 3 for r in result["scores"]["criteria"]):
            raise ValueError("Pilot quality requires further calibration")
    controls = [r for r in pilot._read(manifest_path) if r["origin"] == "negative-control"]
    if len(controls) != 1:
        raise ValueError("Expected one planted-error calibration output")
    control, binding = controls[0], pilot._read(control_path)
    saved = audit["records"][binding["judgment_identity"]]
    if (control["sample_id"] != binding["sample_id"]
            or control["output_sha256"] != binding["output_sha256"]
            or hashlib.sha256(control["output"].encode()).hexdigest() != binding["output_sha256"]
            or saved["result"]["record_sha256"] != binding["record_sha256"]
            or saved["attempt"]["request"] != evaluator.judge_request(cases["pilot-membership"], control["output"])):
        raise ValueError("Negative control differs from its saved blind judgment")
    scores = saved["result"]["scores"]
    detected = bool(scores["material_defects"]) and any(r["score"] < 3 for r in scores["criteria"])
    return {"status": "calibrated-pilot" if detected else "calibration-failed",
            "human_calibrated": False, "claimable": False, "negative_control_detected": detected,
            "pilot_outputs_judged": 12, "paid_judgments": audit["paid_judgments"],
            "cost_lower_nanousd": audit["cost_lower_nanousd"], "cost_upper_nanousd": audit["cost_upper_nanousd"],
            "judge_freeze_sha256": audit["judge_freeze_sha256"],
            "pilot_summary_sha256": original["summary_sha256"],
            "bindings_sha256": evaluator._hash(bindings),
            "control_binding_sha256": evaluator._hash(binding),
            "evidence_paths": {"directory": str(Path(directory).resolve()),
                               "bindings_path": str(Path(bindings_path).resolve()),
                               "manifest_path": str(Path(manifest_path).resolve()),
                               "control_path": str(Path(control_path).resolve())}}
