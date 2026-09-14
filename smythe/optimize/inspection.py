"""Read-only report projections of retained Autotune evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from smythe.optimize.ledger import ExperimentLedger, ExperimentLedgerError


DEFAULT_TRIAL_DETAIL_LIMIT = 500
MAX_TRIAL_DETAIL_LIMIT = 1000
_TRIAL_FIELDS = (
    "trial_key", "candidate_id", "split", "phase", "status", "metrics", "gates",
    "ceiling_microusd", "actual_cost_microusd", "duration_ms", "artifact_hashes",
    "error", "prepared_at_ns", "dispatched_at_ns", "terminal_at_ns",
)


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
        payload = {
            "report_version": 1,
            "campaign_id": snapshot["campaign"]["campaign_id"],
            "evaluator_hashes": sorted({trial.evaluator_hash for trial in trials}),
            "ledger_snapshot": snapshot,
            "trials": detail,
            "trial_detail": {
                "limit": trial_limit, "total": len(trials), "returned": len(detail),
                "truncated": len(detail) < len(trials),
            },
        }
        # Aggregated balances can exceed a single SQLite INTEGER. Preserve
        # Python integers; contract canonicalization imposes a per-value bound
        # that does not apply to these derived accounting totals.
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        # Preserve snapshot insertion order for the established CLI JSON output;
        # only the fingerprint representation sorts keys.
        detached = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        detached["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
        return detached
    except ExperimentLedgerError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        # Errors while reconstructing stored contract/decision/trial JSON are
        # ledger corruption, not invalid CLI arguments. Preserve their cause.
        raise ExperimentLedgerError("Invalid retained Autotune report evidence") from exc
