"""Hash-bound offline preparation. This module cannot start a paid campaign."""

from __future__ import annotations

from pathlib import Path

from ._json import (
    CampaignPlanError, TEXT_HASH_POLICY, canonical, digest, keys, read_bytes, read_json,
)
from .schedule import ARMS, build_schedule
from .tasks import DATA_DIR, load_task_pack

BLOCKERS = (
    "No total API-spend ceiling or pilot/main/judge allocations have been authorized.",
    "The campaign runner must bind the persisted graph policy and phase-wide accounting.",
    "The fixed external judge and its durable accounting path have not been frozen.",
    "The paid pilot, human calibration, success gate and quality margin remain pending.",
)
SOURCE_FILES = ("__init__.py", "_json.py", "protocol.py", "schedule.py", "tasks.py")


def prepare_campaign(seed: int = 14173, directory: str | Path = DATA_DIR) -> dict:
    """Validate artifacts and return a reproducible, explicitly blocked run plan.

    Only local bounded files are read. No API keys, models, SDKs or execution
    stores are accessed. This return value is preparation evidence, not results.
    """
    root = Path(directory)
    pack = load_task_pack(root)
    protocol, raw = read_json(root / "protocol.json")
    keys(protocol, {"version", "status", "pack_manifest_sha256", "models", "strategies",
                    "pilot", "main", "request_policy", "execution_policy", "evaluation_policy",
                    "price_snapshot", "total_api_budget_nanousd"}, "protocol")
    if type(protocol["version"]) is not int or protocol["version"] != 1:
        raise CampaignPlanError("Unsupported protocol version")
    if (protocol["status"] != "offline-preparation-only"
            or protocol["pack_manifest_sha256"] != pack.manifest_sha256):
        raise CampaignPlanError("Protocol does not bind this preparation pack")
    if protocol["models"] != ["gpt-6-astra", "gpt-5.6-sol"] or protocol["strategies"] != [
        "fixed_pipeline", "smythe_dynamic",
    ]:
        raise CampaignPlanError("Protocol must preserve all four experimental arms")
    expected = {
        "pilot": {"tasks": 3, "repetitions": 1, "workflow_runs": 12, "included_in_main": False},
        "main": {"tasks": 10, "repetitions": 5, "workflow_runs": 200, "tasks_per_shape": 2},
        "request_policy": {"endpoint": "responses", "reasoning_effort": "medium",
                           "service_tier": "default", "endpoint_scope": "global",
                           "max_output_tokens": 8192, "sdk_retries": 0, "tools": False,
                           "external_search": False, "sampling_parameters": [],
                           "cache_policy": "Provider-managed; record actual usage; no cold-cache claim."},
        "execution_policy": {"max_nodes": 8, "max_concurrency": 8,
                             "workflow_concurrency": 1, "same_model_for_planning_and_execution": True,
                             "graph_policy_persisted": False, "node_retry_policy": None,
                             "regeneration_policy": None, "planning_repairs": None,
                             "synthesis_policy": None},
        "evaluation_policy": {"judge_model": None, "judge_accounting": None,
                              "quality_noninferiority_margin": None, "minimum_success_rate": None,
                              "pilot_completed": False, "human_calibration_completed": False,
                              "rubric_scale": [0, 1, 2, 3, 4], "sampling_unit": "task",
                              "preserve_failures": True, "cost_per_ordinal_score": False,
                              "judge_sees_arm_labels": False, "judge_sees_reference_material": True},
    }
    for field, value in expected.items():
        if canonical(protocol[field]) != canonical(value):
            raise CampaignPlanError(f"Preparation protocol changed: {field}")
    prices = {"version": "openai-native-standard-global-2026-09-07-v1", "checked_on": "2026-09-07",
              "unit": "nanousd_per_token", "long_context_threshold": 272000,
              "long_context_input_multiplier": 2, "long_context_output_multiplier": "1.5",
              "rates": {"gpt-6-astra": {"ordinary": 10000, "cached": 1000, "cache_write": 12500, "output": 50000},
                        "gpt-5.6-sol": {"ordinary": 4000, "cached": 400, "cache_write": 5000, "output": 20000}},
              "source": "https://developers.openai.com/api/docs/pricing",
              "scope": "Dated repository price snapshot; revalidate before authorizing paid execution."}
    if canonical(protocol["price_snapshot"]) != canonical(prices):
        raise CampaignPlanError("Preparation price snapshot changed")
    if protocol["total_api_budget_nanousd"] is not None:
        raise CampaignPlanError("This preparation artifact cannot authorize API spending")
    schedules = {stage: build_schedule(stage, seed, pack) for stage in ("pilot", "main")}
    source_hashes = {name: digest(read_bytes(Path(__file__).parent / name))
                     for name in SOURCE_FILES}
    result = {
        "version": 1, "status": "offline-preparation-only", "api_calls": 0,
        "paid_execution_allowed": False, "claimable": False, "blockers": list(BLOCKERS),
        "seed": seed, "schedule_algorithm": "sha256-order-williams-four-arm-v1",
        "protocol_sha256": digest(raw), "pack_manifest_sha256": pack.manifest_sha256,
        "text_hash_policy": TEXT_HASH_POLICY,
        "source_hash_scope": "Offline preparation package only; not a paid campaign runtime freeze.",
        "source_sha256": source_hashes, "data_sha256": dict(pack.file_hashes),
        "arms": [dict(zip(("arm_id", "model", "strategy"), arm)) for arm in ARMS],
        "schedules": schedules,
        "schedule_sha256": {stage: digest(canonical(rows).encode()) for stage, rows in schedules.items()},
        "independent_main_tasks": 10, "main_repetitions_per_task_arm": 5,
        "total_api_budget_nanousd": None,
    }
    result["preparation_sha256"] = digest(canonical(result).encode())
    return result
