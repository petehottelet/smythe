"""Safe, deterministic, in-process orchestration for optimization campaigns."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import re
import time
import threading
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from smythe.optimize.contracts import (
    Candidate,
    ExperimentContract,
    ObjectiveDirection,
    canonical_json_bytes,
    normalize_gate_name,
    sha256_prefixed,
)
from smythe.optimize.ledger import (
    _ENGINE_HOLDOUT_CAPABILITY,
    _lease_duration_ns,
    CampaignLease,
    CampaignLeaseError,
    ExperimentLedger,
    ExperimentLedgerError,
    LedgerBudgetError,
    LedgerConflictError,
    PromotionDecision,
    TrialRecord,
    TrialStateError,
    TrialStatus,
    UnknownTrialError,
)
from smythe.optimize.statistics import (
    MAX_BOOTSTRAP_RESAMPLES,
    ComparisonResult,
    PromotionAssessment,
    aggregate_mean,
    assess_promotion,
    hard_thresholds_pass,
)


RUNNER_VERSION = 3
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_SIGNED_63 = (1 << 63) - 1
_MAX_RUNNER_CANDIDATES = 1_024
_MAX_RUNNER_PARALLEL_CANDIDATES = 256
_MAX_RUNNER_REPETITIONS = 10_000
_MAX_RUNNER_TRIALS = 100_000


class OptimizationError(RuntimeError):
    """Base class for optimization-runner failures."""


class OptimizationNeedsAttention(OptimizationError):
    """A dispatched trial has an ambiguous outcome and must not be retried."""


class OptimizationLimitError(OptimizationError):
    """A declared wall-time, trial, per-trial, or campaign budget was reached."""


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Immutable input supplied to one in-process evaluator invocation."""

    campaign_id: str
    optimization_plan_hash: str
    trial_key: str
    candidate: Candidate
    split: str
    phase: str
    seed: int
    evaluator_hash: str
    ceiling_microusd: int
    deadline_monotonic: float


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Evaluator output recorded at the durable completion boundary."""

    metrics: Mapping[str, int | float]
    gates: Mapping[str, bool]
    actual_cost_microusd: int
    artifact_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if not isinstance(self.gates, Mapping):
            raise TypeError("gates must be a mapping")
        if isinstance(self.artifact_hashes, (str, bytes)):
            raise TypeError("artifact_hashes must be an iterable of hashes")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "gates", dict(self.gates))
        object.__setattr__(self, "artifact_hashes", tuple(self.artifact_hashes))


TrialEvaluator: TypeAlias = Callable[[TrialContext], Awaitable[TrialOutcome]]


@dataclass(slots=True)
class _DurationBudget:
    completed_keys: set[str]
    total_ms: float
    lock: asyncio.Lock


class _LeaseHeartbeat:
    """Renew independently of synchronous statistics on the event-loop thread."""

    def __init__(self, ledger, lease, ttl, interval, loop, task):
        self.failure: BaseException | None = None
        self._stop = threading.Event()

        def renew():
            while not self._stop.wait(interval):
                try:
                    ledger.heartbeat_campaign_lease(lease, ttl_s=ttl)
                except BaseException as error:
                    self.failure = error
                    loop.call_soon_threadsafe(task.cancel)
                    return

        self._thread = threading.Thread(target=renew, name="smythe-autotune-heartbeat", daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join()


async def _cancel_and_drain(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    draining = asyncio.gather(*tasks, return_exceptions=True)
    while not draining.done():
        try:
            await asyncio.shield(draining)
        except asyncio.CancelledError:
            continue
    draining.result()


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Terminal campaign result and its durable evidence references."""

    campaign_id: str
    optimization_plan_hash: str
    holdout_seed_commitment: str
    incumbent_candidate_id: str
    selected_candidate_id: str | None
    promoted: bool
    reason: str
    decision_id: str | None
    development_scores: tuple[dict[str, Any], ...]
    confirmation_assessment: PromotionAssessment | None
    holdout_assessment: PromotionAssessment | None
    trial_keys: tuple[str, ...]
    ledger_snapshot: dict[str, Any]

    def __post_init__(self) -> None:
        scores = _json_copy([dict(item) for item in self.development_scores])
        snapshot = _json_copy(self.ledger_snapshot)
        object.__setattr__(self, "development_scores", tuple(scores))
        object.__setattr__(self, "ledger_snapshot", snapshot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "optimization_plan_hash": self.optimization_plan_hash,
            "holdout_seed_commitment": self.holdout_seed_commitment,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "promoted": self.promoted,
            "reason": self.reason,
            "decision_id": self.decision_id,
            "development_scores": _json_copy(list(self.development_scores)),
            "confirmation_assessment": _assessment_dict(
                self.confirmation_assessment
            ),
            "holdout_assessment": _assessment_dict(self.holdout_assessment),
            "trial_keys": list(self.trial_keys),
            "ledger_snapshot": _json_copy(self.ledger_snapshot),
        }


class OptimizationRunner:
    """Run a bounded development/confirmation/holdout campaign in-process."""

    def __init__(
        self,
        contract: ExperimentContract,
        ledger: ExperimentLedger,
        evaluator: TrialEvaluator,
        *,
        evaluator_hash: str,
        bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        campaign_id: str | None = None,
        lease_ttl_s: float = 30.0,
        lease_heartbeat_s: float | None = None,
    ) -> None:
        if not isinstance(contract, ExperimentContract):
            raise TypeError("contract must be an ExperimentContract")
        if not isinstance(ledger, ExperimentLedger):
            raise TypeError("ledger must be an ExperimentLedger")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        if not isinstance(evaluator_hash, str) or not evaluator_hash.strip():
            raise ValueError("evaluator_hash must be a sha256 hash")
        normalized_hash = evaluator_hash.strip()
        if _HASH_RE.fullmatch(normalized_hash) is None:
            raise ValueError("evaluator_hash must have format sha256:<64 lowercase hex>")
        if (
            isinstance(bootstrap_resamples, bool)
            or not isinstance(bootstrap_resamples, int)
            or bootstrap_resamples < 1
            or bootstrap_resamples > MAX_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(
                "bootstrap_resamples must be a positive integer not exceeding "
                f"{MAX_BOOTSTRAP_RESAMPLES}"
            )
        if campaign_id is not None and (
            not isinstance(campaign_id, str) or not campaign_id.strip()
        ):
            raise ValueError("campaign_id must be a non-empty string when provided")

        self.contract = contract
        self.ledger = ledger
        self.evaluator = evaluator
        self.evaluator_hash = normalized_hash
        self.bootstrap_resamples = bootstrap_resamples
        self.campaign_id = campaign_id.strip() if campaign_id is not None else None
        _lease_duration_ns(lease_ttl_s)
        interval = lease_ttl_s / 3 if lease_heartbeat_s is None else lease_heartbeat_s
        _lease_duration_ns(interval)
        if interval >= lease_ttl_s:
            raise ValueError("lease heartbeat must be shorter than its TTL")
        self.lease_ttl_s = float(lease_ttl_s)
        self.lease_heartbeat_s = float(interval)

    async def run(
        self,
        incumbent: Candidate,
        candidates: Sequence[Candidate],
    ) -> OptimizationResult:
        """Execute one deterministic campaign, safely resuming completed trials."""

        caller = asyncio.current_task()
        initial_cancellations = caller.cancelling()
        self._validate_executable_contract()
        started = time.monotonic()
        plan = self._build_plan(incumbent, candidates)
        try:
            self.ledger.create_campaign(self.contract, incumbent.candidate_id,
                                        plan["all_candidates"], plan_hash=plan["plan_hash"],
                                        campaign_id=plan["campaign_id"])
        except (ExperimentLedgerError, ValueError) as exc:
            raise OptimizationError(f"campaign binding drift: {exc}") from exc
        lease = self.ledger.acquire_campaign_lease(plan["campaign_id"], uuid.uuid4().hex,
                                                   ttl_s=self.lease_ttl_s)
        task = None
        heartbeat = None
        error = None
        result = None
        started_heartbeat = False
        try:
            owned = self._run_owned(incumbent, plan, started, lease=lease)
            try:
                task = asyncio.create_task(owned)
            except BaseException:
                owned.close()
                raise
            heartbeat = _LeaseHeartbeat(self.ledger, lease, self.lease_ttl_s,
                                        self.lease_heartbeat_s, asyncio.get_running_loop(), task)
            heartbeat.start()
            started_heartbeat = True
            result = await asyncio.shield(task)
        except BaseException as caught:
            error = caught
            if task is not None:
                task.cancel()
                # Repeated caller cancellation must not abandon owned evaluators.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except BaseException:
                        pass
                if not task.cancelled():
                    task.exception()
        finally:
            if started_heartbeat:
                heartbeat.close()
            if heartbeat is not None and heartbeat.failure is not None:
                loss = CampaignLeaseError(f"Campaign lease heartbeat failed: {heartbeat.failure}")
                loss.__cause__ = heartbeat.failure
                if error is None or (isinstance(error, asyncio.CancelledError)
                                     and caller.cancelling() <= initial_cancellations):
                    error = loss
                else:
                    error.add_note(str(loss))
            try:
                self.ledger.release_campaign_lease(lease)
            except BaseException as release_error:
                if error is None:
                    error = release_error
                else:
                    error.add_note(f"Campaign lease release failed: {release_error}")
        if error is not None:
            raise error
        assert result is not None
        return result

    async def _run_owned(self, incumbent, plan, started, *, lease: CampaignLease):
        campaign_id = self._open_and_register(plan)
        holdout_seed_material, holdout_seed_commitment = (
            self._holdout_seed_binding(campaign_id, plan)
        )
        plan["holdout_seed_commitment"] = holdout_seed_commitment
        existing_trials = self.ledger.list_trials(campaign_id)
        completed = tuple(
            record for record in existing_trials if record.status is TrialStatus.COMPLETED
        )
        durable_ms = sum(record.duration_ms or 0.0 for record in completed)
        durable_seconds = durable_ms / 1_000
        deadline = started + self.contract.max_wall_seconds - durable_seconds
        duration_budget = _DurationBudget(
            completed_keys={record.trial_key for record in completed},
            total_ms=durable_ms,
            lock=asyncio.Lock(),
        )

        development_seeds, confirmation_seeds, holdout_seeds = _split_seeds(
            self.contract,
            holdout_seed_material=holdout_seed_material,
        )
        incumbent_development = await self._evaluate_series(
            campaign_id,
            plan["plan_hash"],
            incumbent,
            "development",
            _bound_phase("incumbent", plan["plan_hash"]),
            development_seeds,
            deadline,
            duration_budget,
            lease=lease,
        )
        challenger_development = await self._evaluate_development_candidates(
            campaign_id,
            plan["plan_hash"],
            plan["candidates"],
            development_seeds,
            deadline,
            duration_budget,
            lease=lease,
        )
        self._require_time(deadline)
        scores = tuple(
            self._development_score(
                incumbent_development,
                challenger_development[candidate.candidate_id],
                candidate,
            )
            for candidate in plan["candidates"]
        )
        viable = [item for item in scores if item["viable"]]
        selected_score = self._best_score(viable or list(scores))
        self._require_time(deadline)
        all_development_keys = _unique_keys(
            record.trial_key
            for records in (incumbent_development, *challenger_development.values())
            for record in records
        )

        if not viable:
            reason = "no candidate passed development gates and hard bounds"
            candidate_id = selected_score["candidate_id"]
            decision = PromotionDecision(
                campaign_id=campaign_id,
                candidate_id=candidate_id,
                promoted=False,
                reason=reason,
                trial_keys=all_development_keys,
                assessment={
                    "optimization_plan_hash": plan["plan_hash"],
                    "runner_version": RUNNER_VERSION,
                    "ledger_durability": self.ledger.durability,
                    "holdout_seed_commitment": holdout_seed_commitment,
                    "stage": "development",
                    "development_scores": list(scores),
                },
            )
            decision_id = self.ledger.append_decision(decision, lease=lease)
            return OptimizationResult(
                campaign_id=campaign_id,
                optimization_plan_hash=plan["plan_hash"],
                holdout_seed_commitment=holdout_seed_commitment,
                incumbent_candidate_id=incumbent.candidate_id,
                selected_candidate_id=None,
                promoted=False,
                reason=reason,
                decision_id=decision_id,
                development_scores=scores,
                confirmation_assessment=None,
                holdout_assessment=None,
                trial_keys=all_development_keys,
                ledger_snapshot=self.ledger.snapshot(campaign_id),
            )

        selected = next(
            item
            for item in plan["candidates"]
            if item.candidate_id == selected_score["candidate_id"]
        )
        confirmation_baseline, confirmation_candidate = await self._evaluate_pair(
            campaign_id,
            plan["plan_hash"],
            incumbent,
            selected,
            "confirmation",
            confirmation_seeds,
            deadline,
            duration_budget,
            lease=lease,
        )
        self._require_time(deadline)
        confirmation = self._assess(
            confirmation_baseline,
            confirmation_candidate,
            "confirmation",
            plan["plan_hash"],
            deadline,
        )
        self._require_time(deadline)
        holdout: PromotionAssessment | None = None
        evidence = list(all_development_keys)
        evidence.extend(record.trial_key for record in confirmation_baseline)
        evidence.extend(record.trial_key for record in confirmation_candidate)

        if confirmation.promote:
            holdout_baseline, holdout_candidate = await self._evaluate_pair(
                campaign_id,
                plan["plan_hash"],
                incumbent,
                selected,
                "holdout",
                holdout_seeds,
                deadline,
                duration_budget,
                lease=lease,
            )
            self._require_time(deadline)
            holdout = self._assess(
                holdout_baseline,
                holdout_candidate,
                "holdout",
                plan["plan_hash"],
                deadline,
            )
            self._require_time(deadline)
            evidence.extend(record.trial_key for record in holdout_baseline)
            evidence.extend(record.trial_key for record in holdout_candidate)

        promoted = confirmation.promote and holdout is not None and holdout.promote
        if promoted:
            reason = "confirmation and untouched holdout both passed promotion policy"
        elif not confirmation.promote:
            reason = "confirmation rejected candidate: " + "; ".join(
                confirmation.reasons
            )
        else:
            assert holdout is not None
            reason = "holdout rejected candidate: " + "; ".join(holdout.reasons)
        trial_keys = _unique_keys(evidence)
        decision = PromotionDecision(
            campaign_id=campaign_id,
            candidate_id=selected.candidate_id,
            promoted=promoted,
            reason=reason,
            trial_keys=trial_keys,
            assessment={
                "optimization_plan_hash": plan["plan_hash"],
                "runner_version": RUNNER_VERSION,
                "ledger_durability": self.ledger.durability,
                "holdout_seed_commitment": holdout_seed_commitment,
                "development": selected_score,
                "confirmation": _assessment_dict(confirmation),
                "holdout": _assessment_dict(holdout),
            },
        )
        decision_id = self.ledger.append_decision(decision, lease=lease)
        return OptimizationResult(
            campaign_id=campaign_id,
            optimization_plan_hash=plan["plan_hash"],
            holdout_seed_commitment=holdout_seed_commitment,
            incumbent_candidate_id=incumbent.candidate_id,
            selected_candidate_id=selected.candidate_id,
            promoted=promoted,
            reason=reason,
            decision_id=decision_id,
            development_scores=scores,
            confirmation_assessment=confirmation,
            holdout_assessment=holdout,
            trial_keys=trial_keys,
            ledger_snapshot=self.ledger.snapshot(campaign_id),
        )

    def _build_plan(
        self,
        incumbent: Candidate,
        candidates: Sequence[Candidate],
    ) -> dict[str, Any]:
        if not isinstance(incumbent, Candidate):
            raise TypeError("incumbent must be a Candidate")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("candidates must be a sequence of Candidate values")
        challengers = tuple(candidates)
        if not challengers:
            raise OptimizationError("at least one challenger candidate is required")
        if any(not isinstance(candidate, Candidate) for candidate in challengers):
            raise TypeError("candidates must contain only Candidate values")
        planned = (incumbent, *challengers)
        for candidate in planned:
            if candidate.contract_hash != self.contract.contract_hash:
                raise OptimizationError("all candidates must belong to the runner contract")
        ids = [candidate.candidate_id for candidate in challengers]
        hashes = [candidate.policy_hash for candidate in challengers]
        if len(set(ids)) != len(ids):
            raise OptimizationError("challenger candidate IDs must be unique")
        if len(set(hashes)) != len(hashes):
            raise OptimizationError("challenger policy hashes must be unique")
        if incumbent.candidate_id in ids:
            raise OptimizationError("a challenger cannot be the incumbent candidate")
        if incumbent.policy_hash in hashes:
            raise OptimizationError("a challenger cannot repeat the incumbent policy")
        if len(planned) > self.contract.max_candidates:
            raise OptimizationLimitError(
                "planned incumbent and challengers exceed max_candidates"
            )

        required_trials = (
            len(planned) * self.contract.development_repetitions
            + 2 * self.contract.confirmation_repetitions
            + 2 * self.contract.holdout_repetitions
        )
        required_reservation = (
            required_trials * self.contract.per_trial_reservation_microusd
        )
        if required_trials > self.contract.max_trials:
            raise OptimizationLimitError(
                f"paired plan requires {required_trials} trials but max_trials is "
                f"{self.contract.max_trials}"
            )
        if required_reservation > self.contract.max_budget_microusd:
            raise OptimizationLimitError(
                f"paired plan requires {required_reservation} microusd of reservation "
                f"but max_budget_microusd is {self.contract.max_budget_microusd}"
            )
        plan_payload = {
            "runner_version": RUNNER_VERSION,
            "statistics": {
                "method": "deterministic_paired_bootstrap",
                "bootstrap_resamples": self.bootstrap_resamples,
            },
            "contract_hash": self.contract.contract_hash,
            "incumbent_candidate_id": incumbent.candidate_id,
            "ordered_candidate_ids": ids,
            "ordered_policy_hashes": hashes,
            "evaluator_hash": self.evaluator_hash,
            "ledger_durability": self.ledger.durability,
            "required_trials": required_trials,
        }
        plan_hash = sha256_prefixed(plan_payload)
        campaign_id = self.campaign_id or (
            "optimization_v3_" + plan_hash.removeprefix("sha256:")
        )
        return {
            "incumbent": incumbent,
            "candidates": challengers,
            "all_candidates": planned,
            "required_trials": required_trials,
            "plan_hash": plan_hash,
            "campaign_id": campaign_id,
        }

    def _validate_executable_contract(self) -> None:
        if not self.contract.required_gates:
            raise OptimizationError(
                "executable optimization contracts must declare required_gates"
            )
        missing_rules = [
            path
            for path in self.contract.mutable_fields
            if path not in self.contract.mutable_field_rules
        ]
        if missing_rules:
            raise OptimizationError(
                "executable optimization contracts must bind every mutable field "
                "to a value rule; missing=" + ", ".join(missing_rules)
            )
        if self.contract.max_candidates > _MAX_RUNNER_CANDIDATES:
            raise OptimizationLimitError(
                f"contract.max_candidates exceeds runner cap {_MAX_RUNNER_CANDIDATES}"
            )
        if (
            self.contract.max_parallel_candidates
            > _MAX_RUNNER_PARALLEL_CANDIDATES
        ):
            raise OptimizationLimitError(
                "contract.max_parallel_candidates exceeds runner cap "
                f"{_MAX_RUNNER_PARALLEL_CANDIDATES}"
            )
        repetitions = (
            self.contract.development_repetitions,
            self.contract.confirmation_repetitions,
            self.contract.holdout_repetitions,
        )
        if any(value > _MAX_RUNNER_REPETITIONS for value in repetitions):
            raise OptimizationLimitError(
                "contract repetition count exceeds runner cap "
                f"{_MAX_RUNNER_REPETITIONS}"
            )
        if self.contract.max_trials > _MAX_RUNNER_TRIALS:
            raise OptimizationLimitError(
                f"contract.max_trials exceeds runner cap {_MAX_RUNNER_TRIALS}"
            )

    def _open_and_register(self, plan: Mapping[str, Any]) -> str:
        incumbent: Candidate = plan["incumbent"]
        campaign_id: str = plan["campaign_id"]
        planned_candidates: tuple[Candidate, ...] = plan["all_candidates"]
        expected = {candidate.candidate_id: candidate for candidate in planned_candidates}
        try:
            self.ledger.create_campaign(
                self.contract,
                incumbent.candidate_id,
                planned_candidates,
                plan_hash=plan["plan_hash"],
                campaign_id=campaign_id,
            )
            self.ledger.open_campaign(
                campaign_id,
                contract=self.contract,
                incumbent_candidate_id=incumbent.candidate_id,
                plan_hash=plan["plan_hash"],
                candidates=planned_candidates,
            )
        except (LedgerConflictError, ExperimentLedgerError, ValueError) as exc:
            raise OptimizationError(f"campaign binding drift: {exc}") from exc

        snapshot = self.ledger.snapshot(campaign_id)
        holdout_seed_material, holdout_seed_commitment = self._holdout_seed_binding(
            campaign_id, plan
        )
        if (
            snapshot.get("campaign", {}).get("holdout_seed_commitment")
            != holdout_seed_commitment
        ):
            raise OptimizationError("campaign holdout seed commitment drifted")
        observed_inventory = snapshot.get("campaign", {}).get("candidate_inventory")
        expected_inventory = [candidate.to_dict() for candidate in planned_candidates]
        if canonical_json_bytes(observed_inventory) != canonical_json_bytes(
            expected_inventory
        ):
            raise OptimizationError("campaign sealed candidate inventory drifted")
        observed_ids = {
            item["candidate_id"] for item in snapshot.get("candidates", [])
        }
        if observed_ids != set(expected):
            raise OptimizationError("campaign candidate rows do not match its sealed plan")
        for item in snapshot.get("candidates", []):
            candidate = expected[item["candidate_id"]]
            if canonical_json_bytes(item["candidate"]) != canonical_json_bytes(
                candidate.to_dict()
            ):
                raise OptimizationError(
                    f"campaign candidate {candidate.candidate_id!r} payload drifted"
                )
        trials = self.ledger.list_trials(campaign_id)
        if trials:
            ambiguous: list[TrialRecord] = []
            split_seeds = dict(
                zip(
                    ("development", "confirmation", "holdout"),
                    _split_seeds(
                        self.contract,
                        holdout_seed_material=holdout_seed_material,
                    ),
                    strict=True,
                )
            )
            for trial in trials:
                if trial.candidate_id not in expected:
                    raise OptimizationError(
                        f"trial {trial.trial_key!r} belongs to an unplanned candidate"
                    )
                if trial.evaluator_hash != self.evaluator_hash:
                    raise OptimizationError(
                        f"trial {trial.trial_key!r} evaluator hash drifted"
                    )
                expected_role = (
                    "incumbent"
                    if trial.candidate_id == incumbent.candidate_id
                    else "challenger"
                )
                if trial.phase != _bound_phase(expected_role, plan["plan_hash"]):
                    raise OptimizationError(
                        f"trial {trial.trial_key!r} optimization plan phase drifted"
                    )
                if trial.split not in split_seeds or trial.seed not in split_seeds[trial.split]:
                    raise OptimizationError(
                        f"trial {trial.trial_key!r} has an unplanned split or seed"
                    )
                if trial.status is TrialStatus.UNKNOWN:
                    raise OptimizationNeedsAttention(
                        f"trial {trial.trial_key!r} has an unknown outcome"
                    )
                if trial.status is TrialStatus.DISPATCHED:
                    ambiguous.append(trial)
                elif trial.status is TrialStatus.COMPLETED:
                    self._validate_completed_record(trial)
            if ambiguous:
                raise OptimizationNeedsAttention(
                    "campaign contains a dispatched trial owned by another or prior runner"
                )
        try:
            self.ledger.validate_decision_inventory(campaign_id)
        except (ExperimentLedgerError, ValueError) as exc:
            raise OptimizationError(f"campaign decision inventory drifted: {exc}") from exc
        for decision in snapshot.get("decisions", []):
                observed_hash = decision.get("assessment", {}).get(
                    "optimization_plan_hash"
                )
                if observed_hash != plan["plan_hash"]:
                    raise OptimizationError("campaign decision optimization plan drifted")
                observed_commitment = decision.get("assessment", {}).get(
                    "holdout_seed_commitment"
                )
                if observed_commitment != holdout_seed_commitment:
                    raise OptimizationError(
                        "campaign decision holdout seed commitment drifted"
                    )
        if (
            snapshot["cost"]["total_exposure_microusd"]
            > self.contract.max_budget_microusd
        ):
            raise OptimizationLimitError(
                "existing campaign exposure exceeds max_budget_microusd"
            )
        durable_seconds = sum(
            (trial.duration_ms or 0.0) / 1_000
            for trial in trials
            if trial.status is TrialStatus.COMPLETED
        )
        if durable_seconds > self.contract.max_wall_seconds:
            raise OptimizationLimitError(
                "existing completed trial duration exceeds max_wall_seconds"
            )
        return campaign_id

    def _holdout_seed_binding(
        self, campaign_id: str, plan: Mapping[str, Any]
    ) -> tuple[bytes, str]:
        try:
            material, commitment = self.ledger._get_holdout_seed_binding(
                campaign_id,
                plan_hash=plan["plan_hash"],
                candidates=plan["all_candidates"],
                capability=_ENGINE_HOLDOUT_CAPABILITY,
            )
        except ExperimentLedgerError as exc:
            raise OptimizationError(
                "campaign holdout seed binding is unavailable or invalid"
            ) from exc
        if not isinstance(material, bytes) or len(material) != 32:
            raise OptimizationError("campaign holdout seed material is invalid")
        if not isinstance(commitment, str) or _HASH_RE.fullmatch(commitment) is None:
            raise OptimizationError("campaign holdout seed commitment is invalid")
        return material, commitment

    async def _evaluate_development_candidates(
        self,
        campaign_id: str,
        plan_hash: str,
        candidates: Sequence[Candidate],
        seeds: Sequence[int],
        deadline: float,
        duration_budget: _DurationBudget,
        *,
        lease: CampaignLease,
    ) -> dict[str, tuple[TrialRecord, ...]]:
        semaphore = asyncio.Semaphore(self.contract.max_parallel_candidates)
        stop_event = asyncio.Event()

        async def evaluate(candidate: Candidate) -> tuple[TrialRecord, ...]:
            try:
                async with semaphore:
                    return await self._evaluate_series(
                        campaign_id,
                        plan_hash,
                        candidate,
                        "development",
                        _bound_phase("challenger", plan_hash),
                        seeds,
                        deadline,
                        duration_budget,
                        stop_event,
                        lease=lease,
                    )
            except BaseException:
                stop_event.set()
                raise

        tasks = [asyncio.create_task(evaluate(candidate)) for candidate in candidates]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            await _cancel_and_drain(tasks)
            raise
        return {
            candidate.candidate_id: result
            for candidate, result in zip(candidates, results, strict=True)
        }

    async def _evaluate_pair(
        self,
        campaign_id: str,
        plan_hash: str,
        incumbent: Candidate,
        challenger: Candidate,
        split: str,
        seeds: Sequence[int],
        deadline: float,
        duration_budget: _DurationBudget,
        *,
        lease: CampaignLease,
    ) -> tuple[tuple[TrialRecord, ...], tuple[TrialRecord, ...]]:
        semaphore = asyncio.Semaphore(self.contract.max_parallel_candidates)
        stop_event = asyncio.Event()

        async def evaluate(
            candidate: Candidate, role: str
        ) -> tuple[TrialRecord, ...]:
            try:
                async with semaphore:
                    return await self._evaluate_series(
                        campaign_id,
                        plan_hash,
                        candidate,
                        split,
                        _bound_phase(role, plan_hash),
                        seeds,
                        deadline,
                        duration_budget,
                        stop_event,
                        lease=lease,
                    )
            except BaseException:
                stop_event.set()
                raise

        tasks = (
            asyncio.create_task(evaluate(incumbent, "incumbent")),
            asyncio.create_task(evaluate(challenger, "challenger")),
        )
        try:
            baseline, candidate = await asyncio.gather(*tasks)
        except BaseException:
            await _cancel_and_drain(tasks)
            raise
        return baseline, candidate

    async def _evaluate_series(
        self,
        campaign_id: str,
        plan_hash: str,
        candidate: Candidate,
        split: str,
        phase: str,
        seeds: Sequence[int],
        deadline: float,
        duration_budget: _DurationBudget,
        stop_event: asyncio.Event | None = None,
        *,
        lease: CampaignLease,
    ) -> tuple[TrialRecord, ...]:
        records: list[TrialRecord] = []
        for seed in seeds:
            if stop_event is not None and stop_event.is_set():
                raise OptimizationNeedsAttention(
                    "parallel evaluation stopped before admitting another trial"
                )
            records.append(
                await self._evaluate_trial(
                    campaign_id,
                    plan_hash,
                    candidate,
                    split,
                    phase,
                    seed,
                    deadline,
                    duration_budget,
                    stop_event,
                    lease=lease,
                )
            )
        return tuple(records)

    async def _evaluate_trial(
        self,
        campaign_id: str,
        plan_hash: str,
        candidate: Candidate,
        split: str,
        phase: str,
        seed: int,
        deadline: float,
        duration_budget: _DurationBudget,
        stop_event: asyncio.Event | None,
        *,
        lease: CampaignLease,
    ) -> TrialRecord:
        self._require_time(deadline)
        try:
            trial_key = self.ledger.prepare_trial(
                campaign_id,
                candidate.candidate_id,
                split,
                phase,
                seed,
                evaluator_hash=self.evaluator_hash,
                ceiling_microusd=self.contract.per_trial_reservation_microusd,
                lease=lease,
            )
        except UnknownTrialError as exc:
            raise OptimizationNeedsAttention(str(exc)) from exc
        except LedgerBudgetError as exc:
            raise OptimizationLimitError(str(exc)) from exc
        except LedgerConflictError as exc:
            raise OptimizationLimitError(str(exc)) from exc

        record = self.ledger.get_trial(trial_key)
        if record.status is TrialStatus.COMPLETED:
            self._validate_completed_record(record)
            await self._enforce_limits(record, duration_budget, lease=lease)
            return record
        if record.status is TrialStatus.UNKNOWN:
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} has an unknown outcome and cannot be reused"
            )
        if record.status is TrialStatus.DISPATCHED:
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} is already dispatched; evaluator will not be invoked"
            )
        if record.status is not TrialStatus.PREPARED:
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} has unsupported state {record.status.value!r}"
            )

        self._require_time(deadline)
        if stop_event is not None and stop_event.is_set():
            raise OptimizationNeedsAttention(
                "parallel evaluation stopped before atomic dispatch claim"
            )
        try:
            self.ledger.claim_trial_dispatch(trial_key, lease=lease)
        except (TrialStateError, UnknownTrialError) as exc:
            raise OptimizationNeedsAttention(
                f"lost atomic dispatch claim for trial {trial_key!r}; evaluator not invoked"
            ) from exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._mark_unknown(trial_key, "wall-time expired after dispatch claim", lease=lease)
            raise OptimizationNeedsAttention(
                f"wall-time expired after dispatching trial {trial_key!r}"
            )
        context = TrialContext(
            campaign_id=campaign_id,
            optimization_plan_hash=plan_hash,
            trial_key=trial_key,
            candidate=candidate,
            split=split,
            phase=phase,
            seed=seed,
            evaluator_hash=self.evaluator_hash,
            ceiling_microusd=self.contract.per_trial_reservation_microusd,
            deadline_monotonic=deadline,
        )
        trial_started = time.monotonic()
        try:
            outcome = await asyncio.wait_for(self.evaluator(context), timeout=remaining)
        except asyncio.CancelledError as exc:
            try:
                self._mark_unknown(trial_key, "evaluator cancelled after dispatch", lease=lease)
            except CampaignLeaseError as loss:
                exc.add_note(str(loss))
                raise exc from loss
            raise
        except TimeoutError as exc:
            self._mark_unknown(trial_key, "evaluator exceeded remaining wall-time", lease=lease)
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} timed out after dispatch; outcome is unknown"
            ) from exc
        except Exception as exc:
            self._mark_unknown(
                trial_key,
                f"evaluator raised {type(exc).__name__} after dispatch: {exc}",
                lease=lease,
            )
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} evaluator failed after dispatch"
            ) from exc
        duration_ms = (time.monotonic() - trial_started) * 1_000
        try:
            normalized = self._validate_outcome(outcome)
        except (TypeError, ValueError, OverflowError) as exc:
            self._mark_unknown(
                trial_key, f"evaluator returned an invalid outcome: {type(exc).__name__}: {exc}",
                lease=lease,
            )
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} returned an invalid outcome"
            ) from exc
        try:
            completed = self.ledger.complete_trial(
                trial_key,
                metrics=normalized.metrics,
                gates=normalized.gates,
                actual_cost_microusd=normalized.actual_cost_microusd,
                duration_ms=duration_ms,
                artifact_hashes=normalized.artifact_hashes,
                evaluator_hash=self.evaluator_hash,
                lease=lease,
            )
        except Exception as exc:
            self._mark_unknown(
                trial_key,
                f"durable completion failed after evaluator returned: {type(exc).__name__}: {exc}",
                lease=lease,
            )
            raise OptimizationNeedsAttention(
                f"trial {trial_key!r} could not be durably completed"
            ) from exc
        await self._enforce_limits(completed, duration_budget, lease=lease)
        return completed

    def _validate_outcome(self, outcome: object) -> TrialOutcome:
        if not isinstance(outcome, TrialOutcome):
            raise TypeError("evaluator must return TrialOutcome")
        expected_metrics = {objective.name for objective in self.contract.objectives}
        if set(outcome.metrics) != expected_metrics:
            missing = sorted(expected_metrics - set(outcome.metrics))
            extra = sorted(set(outcome.metrics) - expected_metrics)
            raise ValueError(
                f"metrics must exactly match contract objectives; missing={missing}, extra={extra}"
            )
        metrics: dict[str, int | float] = {}
        for name, value in outcome.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[name] = value
        gates: dict[str, bool] = {}
        for name, passed in outcome.gates.items():
            normalized_name = normalize_gate_name(name, context="outcome gate")
            if normalized_name in gates:
                raise ValueError(
                    f"gate names collide after normalization: {normalized_name!r}"
                )
            if not isinstance(passed, bool):
                raise TypeError(f"gate {name!r} must be boolean")
            gates[normalized_name] = passed
        expected_gates = set(self.contract.required_gates)
        observed_gates = set(gates)
        if observed_gates != expected_gates:
            missing = sorted(expected_gates - observed_gates)
            extra = sorted(observed_gates - expected_gates)
            raise ValueError(
                "gates must exactly match contract required_gates; "
                f"missing={missing}, extra={extra}"
            )
        cost = outcome.actual_cost_microusd
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int)
            or cost < 0
            or cost > _MAX_SIGNED_63
        ):
            raise ValueError(
                "actual_cost_microusd must be a non-negative signed-63 integer"
            )
        hashes = tuple(outcome.artifact_hashes)
        if any(not isinstance(value, str) or _HASH_RE.fullmatch(value) is None for value in hashes):
            raise ValueError("artifact hashes must have format sha256:<64 lowercase hex>")
        if len(set(hashes)) != len(hashes):
            raise ValueError("artifact hashes cannot contain duplicates")
        return TrialOutcome(
            metrics=metrics,
            gates=gates,
            actual_cost_microusd=cost,
            artifact_hashes=hashes,
        )

    def _validate_completed_record(self, record: TrialRecord) -> None:
        try:
            self._validate_outcome(
                TrialOutcome(
                    metrics=record.metrics,
                    gates=record.gates,
                    actual_cost_microusd=record.actual_cost_microusd
                    if record.actual_cost_microusd is not None
                    else -1,
                    artifact_hashes=record.artifact_hashes,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise OptimizationNeedsAttention(
                f"completed trial {record.trial_key!r} violates the current result contract"
            ) from exc
        if record.actual_cost_microusd is None:
            raise OptimizationNeedsAttention(
                f"completed trial {record.trial_key!r} has no confirmed cost"
            )
        if record.actual_cost_microusd > record.ceiling_microusd:
            raise OptimizationLimitError(
                f"completed trial {record.trial_key!r} cost {record.actual_cost_microusd} "
                f"microusd above its {record.ceiling_microusd} ceiling"
            )

    async def _enforce_limits(
        self,
        record: TrialRecord,
        duration_budget: _DurationBudget,
        *,
        lease: CampaignLease,
    ) -> None:
        self.ledger.assert_campaign_lease(lease)
        if (
            record.actual_cost_microusd is not None
            and record.actual_cost_microusd > record.ceiling_microusd
        ):
            raise OptimizationLimitError(
                f"trial {record.trial_key!r} completed above its cost ceiling"
            )
        async with duration_budget.lock:
            if record.trial_key not in duration_budget.completed_keys:
                duration_budget.completed_keys.add(record.trial_key)
                duration_budget.total_ms += record.duration_ms or 0.0
            durable_seconds = duration_budget.total_ms / 1_000
            if durable_seconds > self.contract.max_wall_seconds:
                raise OptimizationLimitError(
                    f"cumulative completed duration {durable_seconds:.6g}s exceeds "
                    f"max_wall_seconds {self.contract.max_wall_seconds}"
                )

    def _mark_unknown(self, trial_key: str, error: str, *, lease: CampaignLease) -> None:
        normalized = "".join(
            character if ord(character) >= 32 else " " for character in str(error)
        ).strip()
        if not normalized:
            normalized = "unknown evaluator failure after dispatch"
        try:
            self.ledger.mark_trial_unknown(trial_key, normalized[:4096], lease=lease)
        except CampaignLeaseError:
            raise
        except Exception:
            pass

    @staticmethod
    def _require_time(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise OptimizationLimitError("campaign max_wall_seconds exhausted")

    def _development_score(
        self,
        baseline: Sequence[TrialRecord],
        challenger: Sequence[TrialRecord],
        candidate: Candidate,
    ) -> dict[str, Any]:
        candidate_metrics = _metric_samples(challenger, self.contract)
        baseline_metrics = _metric_samples(baseline, self.contract)
        means = {
            name: aggregate_mean(values)
            for name, values in candidate_metrics.items()
        }
        baseline_means = {
            name: aggregate_mean(values)
            for name, values in baseline_metrics.items()
        }
        improvements: dict[str, float] = {}
        for objective in self.contract.objectives:
            improvement = (
                means[objective.name] - baseline_means[objective.name]
                if objective.direction is ObjectiveDirection.MAXIMIZE
                else baseline_means[objective.name] - means[objective.name]
            )
            if not math.isfinite(improvement):
                raise OptimizationNeedsAttention(
                    f"development improvement for {objective.name!r} is non-finite"
                )
            improvements[objective.name] = improvement
        gates = _paired_gates(baseline, challenger)
        hard_bounds = {
            objective.name: hard_thresholds_pass(means[objective.name], objective)
            for objective in self.contract.objectives
        }
        primary = self.contract.primary_objective
        primary_improvement = improvements[primary.name] > self.contract.min_improvement
        secondary_non_regression = {
            objective.name: (
                True
                if objective.max_regression is None
                else improvements[objective.name] >= -objective.max_regression
            )
            for objective in self.contract.objectives
            if not objective.primary
        }
        viable = (
            all(gates.values())
            and all(hard_bounds.values())
            and primary_improvement
            and all(secondary_non_regression.values())
        )
        reasons: list[str] = []
        failed_gates = sorted(name for name, passed in gates.items() if not passed)
        failed_bounds = sorted(name for name, passed in hard_bounds.items() if not passed)
        if failed_gates:
            reasons.append("failed gates: " + ", ".join(failed_gates))
        if failed_bounds:
            reasons.append("hard bounds failed: " + ", ".join(failed_bounds))
        if not primary_improvement:
            reasons.append(
                f"primary mean improvement {improvements[primary.name]:.12g} "
                f"does not exceed {self.contract.min_improvement:.12g}"
            )
        failed_secondary = sorted(
            name for name, passed in secondary_non_regression.items() if not passed
        )
        if failed_secondary:
            reasons.append(
                "secondary regression exceeded: " + ", ".join(failed_secondary)
            )
        return {
            "candidate_id": candidate.candidate_id,
            "policy_hash": candidate.policy_hash,
            "primary_objective": primary.name,
            "primary_direction": primary.direction.value,
            "primary_mean": means[primary.name],
            "metric_means": means,
            "incumbent_metric_means": baseline_means,
            "mean_improvements": improvements,
            "gates": gates,
            "hard_bounds": hard_bounds,
            "primary_improvement_passed": primary_improvement,
            "secondary_non_regression": secondary_non_regression,
            "viable": viable,
            "reasons": reasons,
            "trial_keys": [
                *(record.trial_key for record in baseline),
                *(record.trial_key for record in challenger),
            ],
        }

    def _best_score(self, scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
        primary = self.contract.primary_objective
        multiplier = 1.0 if primary.direction is ObjectiveDirection.MINIMIZE else -1.0
        return min(
            scores,
            key=lambda item: (
                multiplier * float(item["primary_mean"]),
                item["candidate_id"],
            ),
        )

    def _assess(
        self,
        baseline: Sequence[TrialRecord],
        candidate: Sequence[TrialRecord],
        split: str,
        plan_hash: str,
        deadline: float,
    ) -> PromotionAssessment:
        try:
            return assess_promotion(
                self.contract.objectives,
                _metric_samples(baseline, self.contract),
                _metric_samples(candidate, self.contract),
                baseline_seeds=[record.seed for record in baseline],
                candidate_seeds=[record.seed for record in candidate],
                gates=_paired_gates(baseline, candidate),
                min_improvement=self.contract.min_improvement,
                confidence=self.contract.confidence,
                bootstrap_resamples=self.bootstrap_resamples,
                bootstrap_seed=_derived_seed(
                    self.contract,
                    f"bootstrap.{split}.{plan_hash}",
                    0,
                    set(),
                ),
                deadline=deadline,
            )
        except TimeoutError as exc:
            raise OptimizationLimitError(
                "campaign max_wall_seconds exhausted during statistical analysis"
            ) from exc


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _unique_keys(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _metric_samples(
    records: Sequence[TrialRecord],
    contract: ExperimentContract,
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for objective in contract.objectives:
        samples: list[float] = []
        for record in records:
            try:
                sample = float(record.metrics[objective.name])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise OptimizationNeedsAttention(
                    f"trial {record.trial_key!r} has an invalid metric "
                    f"{objective.name!r}"
                ) from exc
            if not math.isfinite(sample):
                raise OptimizationNeedsAttention(
                    f"trial {record.trial_key!r} has a non-finite metric "
                    f"{objective.name!r}"
                )
            samples.append(sample)
        result[objective.name] = tuple(samples)
    return result


def _paired_gates(
    baseline: Sequence[TrialRecord],
    candidate: Sequence[TrialRecord],
) -> dict[str, bool]:
    names = sorted(
        {
            name
            for record in (*tuple(baseline), *tuple(candidate))
            for name in record.gates
        }
    )
    aggregated: dict[str, bool] = {}
    for prefix, records in (("incumbent", baseline), ("candidate", candidate)):
        for name in names:
            aggregated[f"{prefix}.{name}"] = all(
                name in record.gates and record.gates[name] for record in records
            )
    return aggregated


def _bound_phase(role: str, plan_hash: str) -> str:
    return f"{role}.plan_{plan_hash.removeprefix('sha256:')}"


def _derived_seed(
    contract: ExperimentContract,
    label: str,
    index: int,
    used: set[int],
    *,
    seed_material: bytes | None = None,
) -> int:
    nonce = 0
    while True:
        payload = canonical_json_bytes(
            {
                "version": RUNNER_VERSION,
                "contract_hash": contract.contract_hash,
                "base_seed": contract.base_seed,
                "label": label,
                "index": index,
                "nonce": nonce,
            }
        )
        digest = (
            hashlib.sha256(payload).digest()
            if seed_material is None
            else hmac.new(
                seed_material,
                b"smythe.optimize.holdout-seed.v1\0" + payload,
                hashlib.sha256,
            ).digest()
        )
        seed = int.from_bytes(digest[:8], "big") & (2**63 - 1)
        if seed not in used:
            used.add(seed)
            return seed
        nonce += 1


def _split_seeds(
    contract: ExperimentContract,
    *,
    holdout_seed_material: bytes | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if holdout_seed_material is not None and (
        not isinstance(holdout_seed_material, bytes)
        or len(holdout_seed_material) != 32
    ):
        raise ValueError("holdout_seed_material must be exactly 32 bytes")
    used: set[int] = set()

    def derive(label: str, count: int) -> tuple[int, ...]:
        return tuple(_derived_seed(contract, label, index, used) for index in range(count))

    development = derive("development", contract.development_repetitions)
    confirmation = derive("confirmation", contract.confirmation_repetitions)
    holdout = tuple(
        _derived_seed(
            contract,
            "holdout",
            index,
            used,
            seed_material=holdout_seed_material,
        )
        for index in range(contract.holdout_repetitions)
    )
    return development, confirmation, holdout


def _comparison_dict(result: ComparisonResult) -> dict[str, Any]:
    return {
        "objective_name": result.objective_name,
        "direction": result.direction.value,
        "sample_count": result.sample_count,
        "sample_seeds": list(result.sample_seeds),
        "baseline_mean": result.baseline_mean,
        "candidate_mean": result.candidate_mean,
        "mean_improvement": result.mean_improvement,
        "confidence_level": result.confidence_level,
        "bootstrap_resamples": result.bootstrap_resamples,
        "bootstrap_seed": result.bootstrap_seed,
        "confidence_interval": list(result.confidence_interval),
        "lower_confidence_bound": result.lower_confidence_bound,
        "hard_bounds_passed": result.hard_bounds_passed,
        "non_regression_passed": result.non_regression_passed,
    }


def _assessment_dict(assessment: PromotionAssessment | None) -> dict[str, Any] | None:
    if assessment is None:
        return None
    return {
        "promote": assessment.promote,
        "primary": _comparison_dict(assessment.primary),
        "secondary": [_comparison_dict(item) for item in assessment.secondary],
        "gates": {name: passed for name, passed in assessment.gates},
        "all_gates_passed": assessment.all_gates_passed,
        "hard_bounds_passed": assessment.hard_bounds_passed,
        "secondary_non_regression_passed": (
            assessment.secondary_non_regression_passed
        ),
        "min_improvement": assessment.min_improvement,
        "reasons": list(assessment.reasons),
    }


__all__ = [
    "OptimizationError",
    "OptimizationLimitError",
    "OptimizationNeedsAttention",
    "OptimizationResult",
    "OptimizationRunner",
    "TrialContext",
    "TrialEvaluator",
    "TrialOutcome",
]
