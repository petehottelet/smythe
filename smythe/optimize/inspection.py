"""Read-only report projections of retained Autotune evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from smythe.optimize.ledger import ExperimentLedger, ExperimentLedgerError
from smythe.optimize.registry import PolicyRegistry, _target
from smythe.optimize.statistics import VERDICTS


DEFAULT_TRIAL_DETAIL_LIMIT = 500
MAX_TRIAL_DETAIL_LIMIT = 1000
_TRIAL_FIELDS = (
    "trial_key", "candidate_id", "split", "phase", "status", "metrics", "gates",
    "ceiling_microusd", "actual_cost_microusd", "duration_ms", "artifact_hashes",
    "error", "prepared_at_ns", "dispatched_at_ns", "terminal_at_ns",
)
_COST_FIELDS = ("confirmed_microusd", "reserved_microusd", "unknown_exposure_microusd",
                "total_exposure_microusd")
_REVIEW_FIELDS = ("review_id", "verdict", "reviewer", "evidence_sha256", "note", "created_at_ns")


def _fingerprinted(payload: dict[str, Any]) -> dict[str, Any]:
    """A detached copy of *payload* with the SHA-256 of its canonical JSON."""

    # Aggregated balances can exceed a single SQLite INTEGER. Preserve
    # Python integers; contract canonicalization imposes a per-value bound
    # that does not apply to these derived accounting totals.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    # Preserve insertion order for established CLI JSON output; only the
    # fingerprint representation sorts keys.
    detached = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    detached["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    return detached


def collect_optimization_report(
    ledger: ExperimentLedger,
    campaign_id: str,
    *,
    trial_limit: int = DEFAULT_TRIAL_DETAIL_LIMIT,
) -> dict[str, Any]:
    """Collect detached evidence without recomputing scores or closing the ledger.

    The caller must own a read-only ledger context and complete its close-time
    quiescence check before publishing. ``trial_limit`` bounds detail in the
    report, not the existing whole-campaign validation and accounting queries.
    """

    if type(trial_limit) is not int or not 1 <= trial_limit <= MAX_TRIAL_DETAIL_LIMIT:
        raise ValueError(f"trial_limit must be an integer between 1 and {MAX_TRIAL_DETAIL_LIMIT}")
    if getattr(ledger, "read_only", None) is not True:
        raise ValueError("Autotune reports require a read-only ledger")
    if type(campaign_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", campaign_id.strip()) is None:
        raise ValueError("campaign_id must be a valid campaign identifier")
    try:
        ledger.validate_decision_inventory(campaign_id)
        snapshot = ledger.snapshot(campaign_id)
        trials = ledger.list_trials(campaign_id)
        detail = []
        for trial in trials[:trial_limit]:
            record = trial.to_dict()
            detail.append({name: record[name] for name in _TRIAL_FIELDS})
        return _fingerprinted({
            "report_version": 1,
            "campaign_id": snapshot["campaign"]["campaign_id"],
            "evaluator_hashes": sorted({trial.evaluator_hash for trial in trials}),
            "ledger_snapshot": snapshot,
            "trials": detail,
            "trial_detail": {
                "limit": trial_limit, "total": len(trials), "returned": len(detail),
                "truncated": len(detail) < len(trials),
            },
        })
    except ExperimentLedgerError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        # Errors while reconstructing stored contract/decision/trial JSON are
        # ledger corruption, not invalid CLI arguments. Preserve their cause.
        raise ExperimentLedgerError("Invalid retained Autotune report evidence") from exc


def collect_target_report(
    ledger: ExperimentLedger,
    registry: PolicyRegistry,
    target: str,
) -> dict[str, Any]:
    """Collect a policy target's campaigns, decisions, activations and optimization cost.

    A campaign touches *target* when a candidate in its sealed inventory sets
    the target or a path under it. Each such campaign's decision inventory is
    validated, and its terminal decision, with the three-way verdict when one
    was recorded, and its reviews are listed beside the target's activation
    history. Confirmed, reserved and unknown micro-USD are summed as exact
    integers across those campaigns: what optimizing cost, never netted
    against any saving a policy brings. Unknown trials are also counted, so
    an unknown outcome under a zero ceiling still shows. Campaigns that an
    activation's review names but the ledger does not hold are listed, since
    their cost is missing. Both stores must be open read-only.
    """

    if getattr(ledger, "read_only", None) is not True:
        raise ValueError("Autotune reports require a read-only ledger")
    if getattr(registry, "read_only", None) is not True:
        raise ValueError("target reports require a read-only policy registry")
    name = _target(target)
    history = registry.history(name)
    active = registry.resolve(name)
    named = {event.review_id: registry.review(event.review_id)
             for event in history if event.review_id is not None}
    try:
        campaigns = []
        totals = dict.fromkeys(("campaigns", *_COST_FIELDS, "unknown_trials"), 0)
        for campaign in ledger.list_campaigns():
            policies = {item["candidate_id"]: item["policy_hash"]
                        for item in campaign["candidate_inventory"]}
            if not any(path == name or path.startswith(name + ".")
                       for item in campaign["candidate_inventory"] for path in item["policy"]):
                continue
            identifier = campaign["campaign_id"]
            ledger.validate_decision_inventory(identifier)
            snapshot = ledger.snapshot(identifier)
            decision = None
            for recorded in snapshot["decisions"]:
                verdict = recorded["assessment"].get("verdict")
                if verdict is not None and verdict not in VERDICTS:
                    raise ValueError(f"recorded verdict {verdict!r} is not one of {VERDICTS}")
                decision = {
                    "decision_id": recorded["decision_id"],
                    "candidate_id": recorded["candidate_id"],
                    "policy_hash": policies[recorded["candidate_id"]],
                    "promoted": recorded["promoted"],
                    "verdict": verdict,
                    "reason": recorded["reason"],
                    "created_at_ns": recorded["created_at_ns"],
                }
            sealed = (identifier, campaign["plan_hash"], campaign["holdout_seed_commitment"])
            unknown = snapshot["trial_counts"].get("unknown", 0)
            campaigns.append({
                "campaign_id": identifier,
                "plan_hash": campaign["plan_hash"],
                "holdout_commitment": campaign["holdout_seed_commitment"],
                "created_at_ns": campaign["created_at_ns"],
                "incumbent_policy_hash": policies[campaign["incumbent_candidate_id"]],
                "trial_counts": snapshot["trial_counts"],
                "cost": snapshot["cost"],
                "unknown_trials": unknown,
                "decision": decision,
                # A review names its campaign by ID, plan and commitment, so a
                # campaign of the same ID in another ledger is not this one.
                "reviews": [{field: getattr(review, field) for field in _REVIEW_FIELDS}
                            for review in registry.reviews(identifier)
                            if (review.campaign_id, review.plan_hash,
                                review.holdout_commitment) == sealed],
            })
            totals["campaigns"] += 1
            totals["unknown_trials"] += unknown
            for field in _COST_FIELDS:
                totals[field] += snapshot["cost"][field]
        held = {(item["campaign_id"], item["holdout_commitment"]) for item in campaigns}
        return _fingerprinted({
            "target_report_version": 1,
            "target": name,
            "active": None if active is None else active.to_dict(),
            "history": [event.to_dict() for event in history],
            "campaigns": campaigns,
            "campaigns_not_in_ledger": sorted({
                review.campaign_id for review in named.values()
                if (review.campaign_id, review.holdout_commitment) not in held}),
            "optimization_cost": totals,
        })
    except ExperimentLedgerError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        raise ExperimentLedgerError("Invalid retained Autotune report evidence") from exc
