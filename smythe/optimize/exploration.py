"""Bounded exploration: Autotune campaigns fed by an out-of-process proposer.

Each round asks a proposer program for challengers and runs them as one
:class:`~smythe.optimize.engine.OptimizationRunner` campaign against a fixed
incumbent. The loop stops at the first promotion, after a set number of
consecutive campaigns that did not promote, when the proposer proposes
nothing, or before a proposer call or campaign its budget cannot cover. It
never reviews or activates anything: a promoted decision waits for review in
the policy registry like any other.

The proposer sees :func:`proposer_view`, which holds no holdout metric, seed,
trial or assessment, and never the ledger or its path. Its charges are
reserved at its ceiling before it starts and recorded in the policy
registry's exploration tables.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smythe._redact import error_text, redact_paths
from smythe.optimize.contracts import Candidate, ContractValidationError, canonical_json_bytes
from smythe.optimize.engine import (
    _RESERVED_PATH_ROOTS,
    OptimizationError,
    OptimizationResult,
    OptimizationRunner,
)
from smythe.optimize.ledger import (
    _SQLITE_INT_MAX,
    CampaignNotFoundError,
    ExperimentLedger,
    HoldoutAlreadyUsedError,
    _safe_id,
)
from smythe.optimize.registry import PolicyRegistry

PROTOCOL = "smythe.exploration.v1"
_MAX_REPLY_BYTES = 1 << 20
_REAP_SECONDS = 5.0
# Inherited variables a proposer never receives: provider keys and other credentials.
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_ACCESS_KEY", "_ACCESS_KEY_ID", "_SECRET", "_SECRET_KEY",
                        "_TOKEN", "_PASSWORD", "_CREDENTIALS")
_CREDENTIAL_NAMES = frozenset({"SMYTHE_MCP_URL"})


class ProposerError(OptimizationError):
    """A proposer call failed after it may have spent; its charge is unknown."""


def _count(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (
            minimum <= value <= _SQLITE_INT_MAX):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ExplorationBudget:
    """What one exploration may run and spend, checked before each proposer call and campaign.

    ``max_total_exposure_microusd`` caps the confirmed, reserved and unknown
    micro-USD of this exploration's campaigns, counting the next campaign's
    full required reservation before it starts. ``max_proposer_microusd``
    caps proposer charges, counting the next call at its ceiling. With
    ``max_non_promotions``, the loop also stops after that many consecutive
    campaigns that did not promote.
    """

    max_campaigns: int
    max_total_exposure_microusd: int
    max_proposer_microusd: int
    max_non_promotions: int | None = None

    def __post_init__(self) -> None:
        _count(self.max_campaigns, "max_campaigns", 1)
        _count(self.max_total_exposure_microusd, "max_total_exposure_microusd", 0)
        _count(self.max_proposer_microusd, "max_proposer_microusd", 0)
        if self.max_non_promotions is not None:
            _count(self.max_non_promotions, "max_non_promotions", 1)


@dataclass(frozen=True, slots=True)
class ExplorationResult:
    """Why an exploration stopped, its campaigns, and what it exposed in micro-USD.

    ``detail`` explains the stop in words and is not recorded.
    """

    exploration_id: str
    stop_reason: str
    detail: str
    campaigns: tuple[OptimizationResult, ...]
    campaign_exposure_microusd: int
    proposer_settled_microusd: int
    proposer_unknown_microusd: int


def _proposer_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items()
            if not name.upper().endswith(_CREDENTIAL_SUFFIXES)
            and name.upper() not in _CREDENTIAL_NAMES}


def _kill(process: subprocess.Popen) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _exchange(process: subprocess.Popen, data: bytes, timeout: float) -> tuple[int, bytes] | None:
    """The program's exit status and output, or ``None`` once it is killed at the deadline.

    A watchdog kills the program at the deadline even while the request is
    still being written: on Windows, ``communicate`` writes standard input
    without a timeout, so a program that stops before reading a request
    larger than the pipe buffer would otherwise outlive its deadline.
    """

    lock = threading.Lock()
    state = {"done": False, "killed": False}

    def expire() -> None:
        with lock:
            if not state["done"]:
                state["killed"] = True
                _kill(process)

    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        stdout, _ = process.communicate(data, timeout=timeout)
    except subprocess.TimeoutExpired:
        expire()
        try:
            process.communicate(timeout=_REAP_SECONDS)
        except subprocess.TimeoutExpired:
            # A launcher was killed, but a program it started still holds the pipe.
            process.wait()
        return None
    finally:
        watchdog.cancel()
    with lock:
        # A watchdog kill can let communicate return the killed program's status and output.
        state["done"] = True
        return None if state["killed"] else (process.returncode, stdout)


class SubprocessProposer:
    """Run a proposer program once per round, exchanging one JSON object each way.

    The program reads the request from standard input and must write one
    JSON object to standard output and exit with status 0 within
    ``timeout_s`` seconds; at the deadline it is killed, even before it has
    read the request. It inherits the caller's environment without
    variables named like provider API keys or other credentials, its
    standard error is discarded, and it is never given the ledger or its
    path. ``ceiling_microusd`` is the most one call may spend. Give the
    program's executable itself: a launcher killed at the deadline can
    leave the program it started running.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        ceiling_microusd: int,
        timeout_s: float,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv or any(
                not isinstance(item, str) or not item for item in argv):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be a finite, positive number of seconds")
        self.argv = tuple(argv)
        self.ceiling_microusd = _count(ceiling_microusd, "ceiling_microusd", 0)
        self.timeout_s = float(timeout_s)
        self.cwd = cwd

    async def propose(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Send *request* and return the program's reply object.

        Raises :class:`ProposerError` when the program cannot start, is
        killed at its deadline, exits with another status, or replies with
        anything but a JSON object of at most 1 MiB.
        """

        data = canonical_json_bytes(request)
        try:
            process = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=self.cwd, env=_proposer_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ProposerError("the proposer could not be started") from exc
        exchange = asyncio.ensure_future(
            asyncio.to_thread(_exchange, process, data, self.timeout_s))
        try:
            output = await asyncio.shield(exchange)
        except asyncio.CancelledError:
            _kill(process)
            # Repeated cancellation must not abandon the reaping thread.
            while not exchange.done():
                try:
                    await asyncio.shield(exchange)
                except BaseException:
                    pass
            raise
        if output is None:
            raise ProposerError(
                f"the proposer did not reply within {self.timeout_s:g} seconds and was killed")
        status, stdout = output
        if status != 0:
            raise ProposerError(f"the proposer exited with status {status}")
        reply = None
        if len(stdout) <= _MAX_REPLY_BYTES:
            try:
                reply = json.loads(stdout.decode("utf-8"))
            except (ValueError, RecursionError):
                # Nesting too deep to decode is as invalid as malformed text.
                pass
        if not isinstance(reply, dict):
            raise ProposerError("the proposer reply is not a JSON object of at most 1 MiB")
        return reply


def proposer_view(ledger: ExperimentLedger, campaign_ids: Sequence[str]) -> dict[str, Any]:
    """Return the evidence of *campaign_ids* a proposer may see, as plain JSON.

    Each campaign lists its contract, its sealed candidates, its development
    and confirmation trials with their seeds, metrics and gates, and its
    terminal decision's development scores and confirmation assessments,
    with ``holdout`` only ``"passed"``, ``"failed"`` or ``"not_run"``.
    Holdout trials, seeds, metrics and assessments, the decision's reason
    once the holdout ran, the holdout commitment, trial errors, costs and
    times, and the ledger's location are left out. Each campaign's decision
    inventory is validated first.
    """

    campaigns = []
    for campaign_id in campaign_ids:
        ledger.validate_decision_inventory(campaign_id)
        campaign = ledger.open_campaign(campaign_id)
        decision = None
        for recorded in ledger.list_decisions(campaign_id):
            assessment = recorded["assessment"]
            ran = assessment.get("holdout") is not None
            decision = {
                "candidate_id": recorded["candidate_id"],
                "promoted": recorded["promoted"],
                "holdout": ("passed" if recorded["promoted"] else "failed") if ran else "not_run",
                "reason": None if ran else recorded["reason"],
                "development_scores": assessment.get("development_scores"),
                "confirmation": assessment.get("confirmation"),
                "references": [
                    {key: item.get(key) for key in ("candidate_id", "policy_hash", "confirmation")}
                    for item in assessment.get("references", ())
                ],
            }
        campaigns.append({
            "campaign_id": campaign["campaign_id"],
            "plan_hash": campaign["plan_hash"],
            "contract": campaign["contract"],
            "incumbent_candidate_id": campaign["incumbent_candidate_id"],
            "candidates": campaign["candidate_inventory"],
            "trials": [
                {"candidate_id": trial.candidate_id, "split": trial.split,
                 "role": trial.phase.split(".", 1)[0], "seed": trial.seed,
                 "status": trial.status.value, "metrics": trial.metrics, "gates": trial.gates}
                for trial in ledger.list_trials(campaign_id) if trial.split != "holdout"
            ],
            "decision": decision,
        })
    return {"view_version": 1, "campaigns": campaigns}


def _challengers(
    runner: OptimizationRunner, incumbent: Candidate, proposals: object
) -> tuple[tuple[Candidate, ...], dict[str, Any] | None]:
    """The round's challengers and campaign plan, checked without writing the ledger.

    Raises :class:`OptimizationError` or :class:`HoldoutAlreadyUsedError`
    for the first bound a proposal breaks.
    """

    if not isinstance(proposals, list):
        raise OptimizationError("the proposer reply has no proposals list")
    candidates = []
    for index, item in enumerate(proposals):
        if not isinstance(item, dict) or set(item) != {"policy", "hypothesis"}:
            raise OptimizationError(f"proposal {index} must hold exactly a policy and a hypothesis")
        reserved = sorted(path for path in item["policy"] if isinstance(path, str)
                          and path.split(".", 1)[0].lower() in _RESERVED_PATH_ROOTS
                          ) if isinstance(item["policy"], dict) else ()
        if reserved:
            raise OptimizationError(f"proposal {index} sets reserved paths: " + ", ".join(reserved))
        try:
            candidates.append(Candidate(runner.contract, item["policy"], item["hypothesis"],
                                        parent=incumbent))
        except ContractValidationError as exc:
            raise OptimizationError(f"proposal {index} is refused: {exc}") from exc
    if not candidates:
        return (), None
    plan = runner._build_plan(incumbent, candidates)
    try:
        runner.ledger.open_campaign(plan["campaign_id"])
    except CampaignNotFoundError:
        pass
    else:
        raise OptimizationError(f"the proposals repeat campaign {plan['campaign_id']}")
    runner._require_unused_holdouts(plan)
    return tuple(candidates), plan


async def explore(
    runner: OptimizationRunner,
    registry: PolicyRegistry,
    proposer: SubprocessProposer,
    incumbent: Candidate,
    budget: ExplorationBudget,
    *,
    exploration_id: str | None = None,
) -> ExplorationResult:
    """Run one *runner* campaign per round on the proposer's challengers until a stop.

    Every round's campaign compares the proposals with *incumbent*. The
    runner must name no ``campaign_id``, since each round is a new campaign.
    Proposals must satisfy the contract's allowlist and value rules, the
    runner's reserved paths, its ``single_change`` setting and its caps, and
    must not repeat an existing campaign or a used holdout. Returns at a
    stop; ``stop_reason`` is ``"promoted"``, ``"max_non_promotions"``,
    ``"max_campaigns"``, ``"max_total_exposure"``, ``"max_proposer_spend"``,
    ``"proposer_finished"``, ``"proposer_failed"`` or ``"proposal_refused"``.
    An exception from a campaign, such as an unknown trial outcome, is
    recorded as an ``"interrupted"`` stop and raised. Nothing is activated.
    A given ``exploration_id`` must be a safe identifier and is recorded
    without surrounding whitespace; by default it is ``exploration_<hex>``.
    """

    if not isinstance(runner, OptimizationRunner):
        raise TypeError("runner must be an OptimizationRunner")
    if not isinstance(registry, PolicyRegistry):
        raise TypeError("registry must be a PolicyRegistry")
    if not isinstance(proposer, SubprocessProposer):
        raise TypeError("proposer must be a SubprocessProposer")
    if not isinstance(budget, ExplorationBudget):
        raise TypeError("budget must be an ExplorationBudget")
    if runner.campaign_id is not None:
        raise ValueError("an exploring runner names no campaign_id; each round is a new campaign")
    runner._validate_executable_contract()
    contract = runner.contract
    if not isinstance(incumbent, Candidate) or incumbent.contract_hash != contract.contract_hash:
        raise OptimizationError("the incumbent must be a candidate of the runner's contract")
    # Every event is recorded under the ID as stored, without surrounding whitespace.
    identifier = ("exploration_" + uuid.uuid4().hex if exploration_id is None
                  else _safe_id(exploration_id, "exploration_id"))
    ceiling = proposer.ceiling_microusd
    registry._start_exploration(identifier, {
        "budget": {name: getattr(budget, name) for name in budget.__dataclass_fields__},
        "contract_hash": contract.contract_hash,
        "evaluator_hash": runner.evaluator_hash,
        "incumbent_candidate_id": incumbent.candidate_id,
        "proposer_ceiling_microusd": ceiling,
    })
    # The smallest campaign has one challenger and no references.
    smallest = 2 * contract.minimum_full_evaluation_trials * contract.per_trial_reservation_microusd
    results: list[OptimizationResult] = []
    spent = {"exposure": 0, "settled": 0, "unknown": 0}
    round_ = 1

    def record(kind: str, **fields: Any) -> None:
        registry._record_exploration(identifier, round_, kind, **fields)

    def stop(reason: str, detail: str = "") -> ExplorationResult:
        record("stopped", detail=reason)
        return ExplorationResult(identifier, reason, detail or reason, tuple(results),
                                 spent["exposure"], spent["settled"], spent["unknown"])

    try:
        while True:
            if len(results) >= budget.max_campaigns:
                return stop("max_campaigns")
            if spent["exposure"] + smallest > budget.max_total_exposure_microusd:
                return stop("max_total_exposure", "the smallest campaign would exceed the cap")
            if spent["settled"] + spent["unknown"] + ceiling > budget.max_proposer_microusd:
                return stop("max_proposer_spend")
            request = {
                "protocol": PROTOCOL, "round": round_, "ceiling_microusd": ceiling,
                "contract": contract.to_dict(), "incumbent": incumbent.to_dict(),
                "max_challengers": contract.max_candidates - 1,
                "single_change": runner.single_change,
                "evidence": proposer_view(runner.ledger, [item.campaign_id for item in results]),
            }
            record("proposer_reserved", microusd=ceiling)
            try:
                reply = await proposer.propose(request)
                declared = reply.get("spent_microusd")
                if isinstance(declared, bool) or not isinstance(declared, int) or not (
                        0 <= declared <= _SQLITE_INT_MAX):
                    raise ProposerError("the proposer reply declares no whole spent_microusd")
            except BaseException as error:
                text = (redact_paths(error_text(error))[:1024] if isinstance(error, ProposerError)
                        else "interrupted")
                record("proposer_unknown", microusd=ceiling, detail=text)
                spent["unknown"] += ceiling
                if not isinstance(error, ProposerError):
                    raise
                return stop("proposer_failed", text)
            record("proposer_settled", microusd=declared)
            spent["settled"] += declared
            if declared > ceiling:
                return stop("proposer_failed",
                            f"the proposer declared {declared} micro-USD above its {ceiling} ceiling")
            try:
                candidates, plan = _challengers(runner, incumbent, reply.get("proposals"))
            except (OptimizationError, HoldoutAlreadyUsedError) as error:
                return stop("proposal_refused", str(error))
            if plan is None:
                return stop("proposer_finished")
            required = plan["required_trials"] * contract.per_trial_reservation_microusd
            if spent["exposure"] + required > budget.max_total_exposure_microusd:
                return stop("max_total_exposure",
                            f"campaign {plan['campaign_id']} would reserve {required} micro-USD")
            record("campaign_started", microusd=required, campaign_id=plan["campaign_id"])
            result = await runner.run(incumbent, candidates)
            exposure = result.ledger_snapshot["cost"]["total_exposure_microusd"]
            spent["exposure"] += exposure
            results.append(result)
            record("campaign_ended", microusd=exposure, campaign_id=result.campaign_id,
                   detail="promoted" if result.promoted else "not_promoted")
            if result.promoted:
                return stop("promoted")
            # A promotion stops the loop, so every campaign so far did not promote.
            if budget.max_non_promotions is not None and len(results) >= budget.max_non_promotions:
                return stop("max_non_promotions")
            round_ += 1
    except BaseException as error:
        try:
            record("stopped", detail="interrupted")
        except Exception as failure:
            error.add_note(f"the exploration stop could not be recorded: {failure}")
        raise


__all__ = [
    "PROTOCOL",
    "ExplorationBudget",
    "ExplorationResult",
    "ProposerError",
    "SubprocessProposer",
    "explore",
    "proposer_view",
]
