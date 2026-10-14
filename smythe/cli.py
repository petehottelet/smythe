"""Installed command-line interface for Smythe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from smythe._run_inspection import (
    NOT_ACCEPTED,
    RunInspectionError,
    RunNotFoundError,
    inspect_run,
    list_runs,
    open_checkpoints,
    open_journal,
    review_run,
)
from smythe.jobs.loading import load_manifest
from smythe.jobs.models import JOB_MANIFEST_V1_JSON_SCHEMA, ManifestValidationError
from smythe.jobs.preflight import ApprovalError, PreflightError, make_approval, preflight_job
from smythe.jobs.providers import ProviderConfigurationError, ProviderPool
from smythe.jobs.runner import JobRunner
from smythe.jobs.store import (
    InvalidTransitionError,
    JobBudgetError,
    JobNotFoundError,
    RunLeaseError,
    RunStatus,
    RunStoreError,
    SQLiteRunStore,
)
from smythe.optimize.concurrency import ConcurrencyScenario, simulate_concurrency
from smythe.optimize.contracts import (
    Candidate,
    ContractValidationError,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
    MutableValueType,
    ObjectiveDirection,
)
from smythe.optimize.engine import (
    OptimizationError,
    OptimizationLimitError,
    OptimizationNeedsAttention,
)
from smythe.optimize.ledger import (
    CampaignNotFoundError,
    ExperimentLedger,
    ExperimentLedgerError,
    LedgerBudgetError,
    LedgerConflictError,
    TrialStateError,
)
from smythe.optimize.registry import RegistryError


EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_INVALID_INPUT = 2
EXIT_PREFLIGHT = 3
EXIT_APPROVAL = 4
EXIT_BUDGET = 5
EXIT_JOB_STATE = 6
EXIT_LOCAL_ERROR = 7
EXIT_OPTIMIZE_STATE = 8
EXIT_OPTIMIZE_LIMIT = 9
# runs inspect: the run completed, but its output was not accepted.
EXIT_RUN_NOT_ACCEPTED = 10
EXIT_RUN_INDETERMINATE = 11
# runs inspect: the output was accepted, but an external action's effect is unresolved.
EXIT_RUN_ACTION_UNRESOLVED = 12
EXIT_INTERRUPTED = 130

_FAILED_STATUSES = {"needs_attention", "partial", "failed", "budget_overrun"}

MAX_OPTIMIZE_CANDIDATES = 64
MAX_OPTIMIZE_CONCURRENCY = 1_000_000
MAX_OPTIMIZE_WORK_ITEMS = 1_000_000
MAX_OPTIMIZE_PROVIDER_CAPACITY = 1_000_000
MAX_OPTIMIZE_REPETITIONS = 100
MAX_OPTIMIZE_PARALLEL_CANDIDATES = 64
MAX_OPTIMIZE_WALL_SECONDS = 86_400
MAX_OPTIMIZE_BOOTSTRAP_RESAMPLES = 100_000
MAX_OPTIMIZE_SIMULATED_OPERATIONS = 10_000_000
# optimize commands that read or change the policy registry.
_REGISTRY_COMMANDS = ("review", "activate", "rollback", "retire", "history")


def _default_store() -> Path:
    return Path.home() / ".smythe" / "jobs.sqlite3"


def _default_optimize_ledger() -> Path:
    return Path.home() / ".smythe" / "optimize.sqlite3"


def _default_policy_registry() -> Path:
    return Path.home() / ".smythe" / "policy-registry.sqlite3"


def _default_checkpoints() -> Path:
    return Path.home() / ".smythe" / "checkpoints"


def _add_output_options(parser: argparse.ArgumentParser, *, inherited: bool) -> None:
    default = argparse.SUPPRESS if inherited else str(_default_store())
    parser.add_argument(
        "--store",
        default=default,
        help="SQLite job store (default: ~/.smythe/jobs.sqlite3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if inherited else False,
        help="emit one compact JSON document on stdout",
    )


def _add_optimize_output_options(
    parser: argparse.ArgumentParser,
    *,
    inherited: bool,
) -> None:
    default = argparse.SUPPRESS if inherited else str(_default_optimize_ledger())
    parser.add_argument(
        "--ledger",
        default=default,
        help="SQLite optimization ledger (default: ~/.smythe/optimize.sqlite3)",
    )
    parser.add_argument(
        "--ledger-durability",
        choices=("full", "normal"),
        default=argparse.SUPPRESS if inherited else "normal",
        help=(
            "SQLite durability: normal is fast and process-crash atomic; "
            "full also protects against power loss (default: normal)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if inherited else False,
        help="emit one compact JSON document on stdout",
    )


def _add_registry_options(parser: argparse.ArgumentParser, *, ledger: bool = False,
                          change: bool = False) -> None:
    """A registry command's --registry, its --ledger if it reads one, and --json.

    *change* adds the head the change expects and who makes it and why.
    """
    if change:
        parser.add_argument("--expected-seq", type=int, required=True, metavar="SEQ",
                            help="the target's head seq this change was decided from (0 for none)")
        parser.add_argument("--actor", required=True, help="declared name of whoever makes the change")
        parser.add_argument("--reason", required=True, help="reason recorded with the change")
    parser.add_argument("--registry", default=str(_default_policy_registry()),
                        help="SQLite policy registry (default: ~/.smythe/policy-registry.sqlite3)")
    if ledger:
        parser.add_argument("--ledger", default=argparse.SUPPRESS,
                            help="SQLite optimization ledger (default: ~/.smythe/optimize.sqlite3)")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit one compact JSON document on stdout")


def _add_runs_output_options(parser: argparse.ArgumentParser, *, inherited: bool) -> None:
    parser.add_argument(
        "--checkpoints",
        default=argparse.SUPPRESS if inherited else None,
        metavar="DIRECTORY",
        help="ordinary Swarm checkpoint directory (default: ~/.smythe/checkpoints)",
    )
    parser.add_argument(
        "--journal",
        default=argparse.SUPPRESS if inherited else None,
        metavar="PATH",
        help="durable workflow journal (SQLiteWorkflowStore) to read instead of checkpoints",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if inherited else False,
        help="emit one compact JSON document on stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smythe")
    root_commands = parser.add_subparsers(dest="root_command", required=True)
    jobs = root_commands.add_parser("jobs", help="run durable artifact jobs")
    _add_output_options(jobs, inherited=False)
    commands = jobs.add_subparsers(dest="jobs_command", required=True)

    schema = commands.add_parser("schema", help="print the v1 manifest JSON Schema")
    _add_output_options(schema, inherited=True)

    validate = commands.add_parser("validate", help="validate and preflight a manifest")
    validate.add_argument("manifest")
    _add_output_options(validate, inherited=True)

    plan = commands.add_parser("plan", help="expand a manifest and issue an approval token")
    plan.add_argument("manifest")
    plan.add_argument(
        "--max-spend-usd",
        default=None,
        help="approved ceiling; defaults to the manifest maximum",
    )
    _add_output_options(plan, inherited=True)

    run = commands.add_parser("run", help="execute an exactly approved manifest")
    run.add_argument("manifest")
    run.add_argument("--approve", required=True, help="exact token printed by jobs plan")
    run.add_argument(
        "--max-spend-usd",
        default=None,
        help="must match the ceiling used by jobs plan",
    )
    run.add_argument("--new-run", action="store_true",
                     help="start another run with an approval already used by an earlier run")
    run.add_argument("--detach", action="store_true", help="start a worker independent of this terminal")
    run.add_argument("--startup-timeout-s", type=float, default=30.0,
                     help="detached worker readiness timeout (0.1-300 seconds; default: 30)")
    _add_output_options(run, inherited=True)

    status = commands.add_parser("status", help="show durable run state")
    status.add_argument("run_id")
    status.add_argument("--events", action="store_true")
    _add_output_options(status, inherited=True)

    listing = commands.add_parser("list", help="list saved jobs without changing the store")
    listing.add_argument("--limit", type=int, default=50, help="page size (1-500; default: 50)")
    listing.add_argument("--offset", type=int, default=0, help="number of jobs to skip (default: 0)")
    listing.add_argument("--status", default=None, help="filter by run status: " + ", ".join(item.value for item in RunStatus))
    _add_output_options(listing, inherited=True)

    inspection = commands.add_parser("inspect", help="inspect job attempts, phases, costs, and artifacts")
    inspection.add_argument("run_id")
    inspection.add_argument("--operation", default=None, metavar="KEY_OR_ID", help="inspect one operation")
    inspection.add_argument("--limit", type=int, default=50, help="operation page size (default: 50)")
    inspection.add_argument("--offset", type=int, default=0, help="number of operations to skip (default: 0)")
    inspection.add_argument("--events-limit", type=int, default=100, help="maximum recent events (default: 100)")
    inspection.add_argument("--out", default=None, metavar="HTML_PATH", help="write a standalone HTML report")
    _add_output_options(inspection, inherited=True)

    review = commands.add_parser("review", help="list what in a run needs a decision, without changing the store")
    review.add_argument("run_id")
    review.add_argument("--limit", type=int, default=50, help="operation page size (1-500; default: 50)")
    review.add_argument("--offset", type=int, default=0, help="number of operations to skip (default: 0)")
    _add_output_options(review, inherited=True)

    resume = commands.add_parser("resume", help="resume only definitively safe work")
    resume.add_argument("run_id")
    resume.add_argument("--detach", action="store_true", help="resume in a detached worker")
    resume.add_argument("--startup-timeout-s", type=float, default=30.0,
                        help="detached worker readiness timeout (0.1-300 seconds; default: 30)")
    _add_output_options(resume, inherited=True)

    stop = commands.add_parser("stop", help="request a durable pause and let admitted calls drain")
    stop.add_argument("run_id")
    stop.add_argument("--reason", default="operator requested pause", help="saved reason for this stop request")
    stop.add_argument("--drain-timeout-s", type=float, default=None,
                      help="record calls still awaiting a provider this long after the request "
                           "as unknown outcomes (0-3600 seconds; default: wait for them)")
    stop.add_argument("--timeout-s", type=float, default=30.0,
                      help="drain observation duration (0-3600 seconds; default: 30)")
    stop.add_argument("--poll-interval-s", type=float, default=1.0,
                      help="observation interval (0.05-60 seconds; default: 1)")
    _add_output_options(stop, inherited=True)

    attach = commands.add_parser("attach", help="watch saved job state without controlling its worker")
    attach.add_argument("run_id")
    attach.add_argument("--timeout-s", type=float, default=30.0,
                        help="maximum attachment time (0-3600 seconds; default: 30)")
    attach.add_argument("--poll-interval-s", type=float, default=1.0,
                        help="read interval (0.05-60 seconds; default: 1)")
    _add_output_options(attach, inherited=True)

    reroll = commands.add_parser("reroll", help="rerun selected rejected operations")
    reroll.add_argument("run_id")
    # Required unless --dry-run; _parse_arguments() checks them and collects
    # the operation keys that follow an option.
    reroll.add_argument("operation_keys", nargs="*")
    reroll.add_argument("--reason", default=None)
    reroll.add_argument(
        "--acknowledge-unknown",
        action="store_true",
        help="acknowledge possible duplicate spend for unknown outcomes",
    )
    reroll.add_argument(
        "--repair-integrity",
        action="store_true",
        help="rerun succeeded operations whose accepted files are missing or changed",
    )
    reroll.add_argument(
        "--dry-run",
        action="store_true",
        help="show what the selected operations, or all of them, call for; changes nothing",
    )
    _add_output_options(reroll, inherited=True)

    adopt = commands.add_parser(
        "adopt", help="accept verified bytes an unknown outcome already published, without a call",
    )
    adopt.add_argument("run_id")
    adopt.add_argument("operation_key")
    adopt.add_argument("--reason", required=True)
    _add_output_options(adopt, inherited=True)

    recheck = commands.add_parser(
        "recheck", help="decide succeeded operations' accepted files again under the current checks",
    )
    recheck.add_argument("run_id")
    recheck.add_argument("operation_keys", nargs="+")
    recheck.add_argument("--reason", required=True)
    _add_output_options(recheck, inherited=True)

    export = commands.add_parser("export", help="export portable run JSON with events")
    export.add_argument("run_id")
    export.add_argument("--out", default=None, help="write atomically to this path")
    _add_output_options(export, inherited=True)

    optimize = root_commands.add_parser(
        "optimize",
        help="run bounded, evidence-gated optimization campaigns",
    )
    _add_optimize_output_options(optimize, inherited=False)
    optimize_commands = optimize.add_subparsers(
        dest="optimize_command",
        required=True,
    )

    concurrency = optimize_commands.add_parser(
        "concurrency",
        help="find a safe concurrency policy with the offline simulator",
    )
    concurrency.add_argument(
        "--campaign-id",
        default=None,
        help="explicit durable campaign identifier; otherwise derived from the full plan",
    )
    concurrency.add_argument(
        "--candidate-concurrency",
        action="append",
        default=None,
        metavar="N[,N...]",
        help="challenger concurrency (repeatable or comma-separated; default: 2,4,8,12,16)",
    )
    concurrency.add_argument(
        "--reference",
        action="append",
        default=None,
        metavar="N[,N...]",
        help=(
            "reference concurrency compared with the selected challenger on "
            "confirmation and holdout seeds and never promoted (repeatable or "
            "comma-separated)"
        ),
    )
    concurrency.add_argument("--work-items", type=int, default=250)
    concurrency.add_argument("--provider-capacity", type=int, default=8)
    concurrency.add_argument("--base-latency-ms", type=float, default=120.0)
    concurrency.add_argument("--max-p95-latency-ms", type=float, default=250.0)
    concurrency.add_argument("--max-error-rate", type=float, default=0.05)
    concurrency.add_argument("--development-repetitions", type=int, default=3)
    concurrency.add_argument("--confirmation-repetitions", type=int, default=5)
    concurrency.add_argument("--holdout-repetitions", type=int, default=5)
    concurrency.add_argument("--max-parallel-candidates", type=int, default=2)
    concurrency.add_argument("--max-wall-seconds", type=int, default=300)
    concurrency.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help=(
            "two-sided paired t confidence; promotion requires the one-sided "
            "lower bound at error (1 - confidence) / 2 to exceed --min-improvement"
        ),
    )
    concurrency.add_argument("--min-improvement", type=float, default=0.5)
    concurrency.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2_000,
        help="resamples for the descriptive bootstrap interval (not used for promotion)",
    )
    _add_optimize_output_options(concurrency, inherited=True)

    inspect = optimize_commands.add_parser(
        "inspect",
        help="inspect one campaign without mutating its ledger",
    )
    inspect.add_argument("campaign_id")
    inspect.add_argument("--out", default=None, metavar="HTML_PATH",
                         help="publish a new standalone report without replacing existing files")
    _add_optimize_output_options(inspect, inherited=True)

    review_decision = optimize_commands.add_parser(
        "review",
        help="record a verdict on a campaign's terminal decision in the policy registry",
    )
    review_decision.add_argument("campaign_id")
    review_decision.add_argument("--verdict", required=True, choices=("approve", "reject"))
    review_decision.add_argument("--reviewer", required=True,
                                 help="declared reviewer name; not authenticated")
    review_decision.add_argument("--evidence", default=None, metavar="FILE",
                                 help="file the reviewer examined; only its SHA-256 is recorded")
    review_decision.add_argument("--note", default="", help="one line recorded with the review")
    _add_registry_options(review_decision, ledger=True)

    activation = optimize_commands.add_parser(
        "activate",
        help="put an approved review's policy in effect for a target; never edits configuration",
    )
    activation.add_argument("target")
    activation.add_argument("review_id")
    _add_registry_options(activation, change=True)

    rollback = optimize_commands.add_parser(
        "rollback", help="restore an earlier baseline or activate entry of a target as a new event",
    )
    rollback.add_argument("target")
    rollback.add_argument("--to-seq", type=int, required=True, metavar="SEQ",
                          help="the baseline or activate entry to restore")
    _add_registry_options(rollback, change=True)

    retire = optimize_commands.add_parser("retire", help="leave a target with no policy in effect")
    retire.add_argument("target")
    _add_registry_options(retire, change=True)

    history = optimize_commands.add_parser(
        "history",
        help="report a target's activations and the decisions and cost of every campaign that "
             "touched it, without changing the registry or ledger",
    )
    history.add_argument("target")
    _add_registry_options(history, ledger=True)

    runs = root_commands.add_parser(
        "runs",
        help="inspect and review saved Swarm checkpoints and durable workflow runs without "
             "changing them, and revoke external action approvals",
    )
    _add_runs_output_options(runs, inherited=False)
    runs_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_inspection = runs_commands.add_parser(
        "inspect",
        help="show a run's execution, delivery acceptance, context, reads, evidence, open issues, "
             "lineage, external actions, cost, timing and next action",
    )
    run_inspection.add_argument("run_id")
    _add_runs_output_options(run_inspection, inherited=True)
    run_listing = runs_commands.add_parser("list", help="list saved runs with their state and decision")
    run_listing.add_argument("--limit", type=int, default=50, help="page size (1-500; default: 50)")
    run_listing.add_argument("--offset", type=int, default=0, help="number of runs to skip (default: 0)")
    _add_runs_output_options(run_listing, inherited=True)
    run_review = runs_commands.add_parser(
        "review", help="list what in a run, or in the newest runs, needs a decision, without changing them",
    )
    run_review.add_argument("run_id", nargs="?", default=None,
                            help="the run to review; without it, review a page of runs, newest first")
    run_review.add_argument("--limit", type=int, default=50, help="runs to review (1-500; default: 50)")
    run_review.add_argument("--offset", type=int, default=0, help="number of runs to skip (default: 0)")
    _add_runs_output_options(run_review, inherited=True)
    revocation = runs_commands.add_parser(
        "revoke-action",
        help="revoke every approval of an external action, or one approval, without a principal or lease",
    )
    revocation.add_argument("target", metavar="ACTION_OR_APPROVAL_ID")
    revocation.add_argument("--reason", required=True,
                            help="reason to record, with account identifiers redacted")
    _add_runs_output_options(revocation, inherited=True)
    return parser


def _candidate_concurrencies(
    values: Sequence[str] | None,
    *,
    maximum_value: int = MAX_OPTIMIZE_CONCURRENCY,
    role: str = "candidate",
) -> tuple[int, ...]:
    if values is None and role == "reference":
        return ()
    raw_values = ("2,4,8,12,16",) if values is None else tuple(values)
    parsed: list[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        for token in raw_value.split(","):
            item = token.strip()
            if not item:
                raise ValueError(f"{role} concurrency contains an empty value")
            try:
                value = int(item)
            except ValueError as exc:
                raise ValueError(
                    f"{role} concurrency must be an integer, got {item!r}"
                ) from exc
            if value < 1:
                raise ValueError(f"{role} concurrency must be positive")
            if value == 1:
                raise ValueError(f"{role} concurrency 1 is reserved for the incumbent")
            if value > maximum_value:
                raise ValueError(
                    f"{role} concurrency must not exceed {maximum_value}"
                )
            if value in seen:
                raise ValueError(f"duplicate {role} concurrency: {value}")
            seen.add(value)
            parsed.append(value)
            if len(parsed) > MAX_OPTIMIZE_CANDIDATES:
                kind = "challenger candidates" if role == "candidate" else "references"
                raise ValueError(f"at most {MAX_OPTIMIZE_CANDIDATES} {kind} are allowed")
    if not parsed:
        raise ValueError(f"at least one {role} concurrency is required")
    return tuple(parsed)


def _positive_int(
    value: int,
    name: str,
    *,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _probability(value: float, name: str, *, confidence: bool = False) -> float:
    valid = 0.5 < value < 1.0 if confidence else 0.0 <= value < 1.0
    if not valid:
        comparison = (
            "greater than 0.5 and less than 1"
            if confidence
            else "at least 0 and less than 1"
        )
        raise ValueError(f"{name} must be {comparison}")
    return value


def _concurrency_campaign(
    args: argparse.Namespace,
) -> tuple[
    ExperimentContract,
    Candidate,
    tuple[Candidate, ...],
    ConcurrencyScenario,
    tuple[Candidate, ...],
]:
    """Build the contract, incumbent, challengers, scenario and references."""

    work_items = _positive_int(
        args.work_items,
        "work-items",
        maximum=MAX_OPTIMIZE_WORK_ITEMS,
    )
    challenger_values = _candidate_concurrencies(
        args.candidate_concurrency,
        maximum_value=min(MAX_OPTIMIZE_CONCURRENCY, work_items),
    )
    reference_values = _candidate_concurrencies(
        args.reference,
        maximum_value=min(MAX_OPTIMIZE_CONCURRENCY, work_items),
        role="reference",
    )
    shared = sorted(set(reference_values) & set(challenger_values))
    if shared:
        raise ValueError(f"reference concurrency {shared[0]} is also a challenger")
    scenario = ConcurrencyScenario(
        work_items=work_items,
        provider_capacity=_positive_int(
            args.provider_capacity,
            "provider-capacity",
            maximum=MAX_OPTIMIZE_PROVIDER_CAPACITY,
        ),
        base_latency_ms=args.base_latency_ms,
    )
    if args.max_p95_latency_ms <= 0:
        raise ValueError("max-p95-latency-ms must be positive")
    max_error_rate = _probability(args.max_error_rate, "max-error-rate")
    confidence = _probability(args.confidence, "confidence", confidence=True)
    development = _positive_int(
        args.development_repetitions,
        "development-repetitions",
        maximum=MAX_OPTIMIZE_REPETITIONS,
    )
    confirmation = _positive_int(
        args.confirmation_repetitions,
        "confirmation-repetitions",
        maximum=MAX_OPTIMIZE_REPETITIONS,
    )
    holdout = _positive_int(
        args.holdout_repetitions,
        "holdout-repetitions",
        maximum=MAX_OPTIMIZE_REPETITIONS,
    )
    _positive_int(
        args.bootstrap_resamples,
        "bootstrap-resamples",
        maximum=MAX_OPTIMIZE_BOOTSTRAP_RESAMPLES,
    )
    if args.min_improvement < 0:
        raise ValueError("min-improvement must be non-negative")
    max_candidates = 1 + len(challenger_values) + len(reference_values)
    max_parallel = _positive_int(
        args.max_parallel_candidates,
        "max-parallel-candidates",
        maximum=MAX_OPTIMIZE_PARALLEL_CANDIDATES,
    )
    max_wall_seconds = _positive_int(
        args.max_wall_seconds,
        "max-wall-seconds",
        maximum=MAX_OPTIMIZE_WALL_SECONDS,
    )
    if max_parallel > max_candidates:
        raise ValueError(
            "max-parallel-candidates cannot exceed the incumbent plus challengers "
            "and references"
        )

    # The incumbent and every challenger receive development evidence. Only
    # the selected challenger and the incumbent advance, beside each
    # reference, so each later paired split runs 2 + references arms.
    max_trials = (1 + len(challenger_values)) * development + (
        2 + len(reference_values)
    ) * (confirmation + holdout)
    simulated_operations = max_trials * scenario.work_items
    if simulated_operations > MAX_OPTIMIZE_SIMULATED_OPERATIONS:
        raise ValueError(
            "campaign simulation would process "
            f"{simulated_operations} work items; maximum is "
            f"{MAX_OPTIMIZE_SIMULATED_OPERATIONS}"
        )
    contract = ExperimentContract(
        name="concurrency_autotune",
        objectives=(
            MetricObjective(
                name="throughput_ops_s",
                direction=ObjectiveDirection.MAXIMIZE,
                primary=True,
            ),
            MetricObjective(
                name="p95_latency_ms",
                direction=ObjectiveDirection.MINIMIZE,
                hard_max=args.max_p95_latency_ms,
                max_regression=args.max_p95_latency_ms * 0.15,
            ),
            MetricObjective(
                name="error_rate",
                direction=ObjectiveDirection.MINIMIZE,
                hard_max=max_error_rate,
            ),
        ),
        mutable_fields=("max_concurrency",),
        required_gates=("all_operations_accounted",),
        mutable_field_rules={
            "max_concurrency": MutableFieldRule(
                MutableValueType.INTEGER,
                minimum=1,
                maximum=work_items,
            )
        },
        development_repetitions=development,
        confirmation_repetitions=confirmation,
        holdout_repetitions=holdout,
        max_candidates=max_candidates,
        max_parallel_candidates=max_parallel,
        max_trials=max_trials,
        max_wall_seconds=max_wall_seconds,
        max_budget_microusd=0,
        per_trial_reservation_microusd=0,
        confidence=confidence,
        min_improvement=args.min_improvement,
        base_seed=20_260_715,
    )
    incumbent = Candidate(
        contract=contract,
        policy={"max_concurrency": 1},
        hypothesis="The current serial policy is the safety baseline",
    )
    challengers = tuple(
        Candidate(
            contract=contract,
            policy={"max_concurrency": value},
            hypothesis=(
                f"Concurrency {value} increases throughput within the declared "
                "latency and error constraints"
            ),
            parent=incumbent,
        )
        for value in challenger_values
    )
    references = tuple(
        Candidate(
            contract=contract,
            policy={"max_concurrency": value},
            hypothesis=f"Concurrency {value} is a reference for the selected challenger",
            parent=incumbent,
        )
        for value in reference_values
    )
    # Validation above is deliberately eager; this keeps bad CLI input from
    # creating a durable campaign before the runner starts.
    return contract, incumbent, challengers, scenario, references


def _plan(path: str) -> tuple[Any, Path]:
    manifest, root = load_manifest(path)
    plan = preflight_job(manifest, manifest_root=root)
    pool = ProviderPool()
    for operation in plan.operations:
        pool.validate_operation(operation)
        pool.get(operation)
    return plan, root


def _job_exit(snapshot: dict[str, Any]) -> int:
    return (
        EXIT_JOB_FAILED
        if snapshot.get("status") in _FAILED_STATUSES
        else EXIT_OK
    )


def _portable_export(snapshot: dict[str, Any]) -> dict[str, Any]:
    from smythe.jobs.inspection import _artifact_directory, _relative_parts

    exported = dict(snapshot)
    exported["manifest_root"] = "."
    exported["artifact_root"] = PurePosixPath(
        *_relative_parts(snapshot["output_directory"], allow_dot=True), _artifact_directory(snapshot),
    ).as_posix()
    exported["paths_relative_to"] = "artifact_root"
    return exported


def _write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _report_destination(path: str | Path, store_path: str | Path) -> Path:
    """An explicit export must never replace the database being inspected."""
    destination = Path(path).resolve()
    source = Path(store_path).resolve()
    if destination in {source, *(Path(str(source) + suffix) for suffix in ("-wal", "-shm", "-journal"))}:
        raise ValueError("Output path must not replace the Jobs store or its SQLite sidecars")
    return destination


def _write_html(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(html)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _optimization_report_destination(path: str | Path, ledger_path: str | Path) -> Path:
    """Protect lexical and resolved database/sidecar identities, without writes."""
    requested = Path(os.path.abspath(path))
    source = Path(os.path.abspath(ledger_path))
    protected = set()
    for base in (source, source.resolve()):
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = Path(str(base) + suffix)
            protected.update((candidate, candidate.resolve()))
    destination = requested.resolve()
    if requested in protected or destination in protected:
        raise ValueError("Output path must not replace the optimization ledger or its SQLite sidecars")
    for candidate in protected:
        try:
            if requested.samefile(candidate):
                raise ValueError("Output path aliases the optimization ledger or its SQLite sidecars")
        except FileNotFoundError:
            pass
    try:
        requested.lstat()
    except FileNotFoundError:
        return destination
    raise ValueError("Report output already exists; choose a new path")


def _terminal_text(value: Any, *, limit: int = 300) -> str:
    """Keep stored names and errors from executing terminal control sequences."""
    value = str(value)
    shortened = value[:limit] + ("…" if len(value) > limit else "")
    return "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} else char
        for char in shortened
    )


def _usd(microusd: int) -> str:
    whole, fraction = divmod(microusd, 1_000_000)
    return f"${whole}.{fraction:06d}"


def _emit_job_list(value: dict[str, Any]) -> None:
    print(f"Jobs: {value['returned']} (offset {value['offset']}, limit {value['limit']})")
    for run in value["runs"]:
        print(f"{_terminal_text(run['run_id'])}  {_terminal_text(run['status'])}  {_terminal_text(run['name'])}")
        cost = run["cost"]
        print(f"  Operations: {run['operation_count']}; confirmed {_usd(cost['confirmed_microusd'])}; "
              f"exposure {_usd(cost['exposure_microusd'])}; reserved {_usd(cost['reserved_microusd'])}")
    if not value["runs"]:
        print("No jobs matched.")
    if value["has_more"]:
        print(f"More jobs: use --offset {value['offset'] + value['returned']}")


def _emit_job_inspection(value: dict[str, Any]) -> None:
    print(f"Run: {_terminal_text(value['run_id'])}")
    print(f"Name: {_terminal_text(value['name'])}")
    print(f"Status: {_terminal_text(value['status'])}")
    print(f"Operations (whole run): {_terminal_text(json.dumps(value['counts'], sort_keys=True))}")
    cost = value["cost"]
    print(f"Confirmed: {_usd(cost['confirmed_microusd'])}; Exposure: {_usd(cost['exposure_microusd'])}; "
          f"Reserved: {_usd(cost['reserved_microusd'])}; Approved: {_usd(cost['approved_microusd'])}")
    print(f"Cost complete: {'yes' if cost['cost_is_complete'] else 'no'}; "
          f"includes estimates: {'yes' if cost['cost_contains_estimates'] else 'no'}")
    page = value["pagination"]
    print(f"Operation page: {page['returned']} of {page['total']} (offset {page['offset']}, limit {page['limit']})")
    for operation in value["operations"]:
        print(f"Operation {_terminal_text(operation['operation_key'])} "
              f"({_terminal_text(operation['operation_id'])}): {_terminal_text(operation['status'])}; "
              f"attempts {operation['attempt_count']}/{operation['max_attempts']}")
        print(f"  Input digest: {_terminal_text(operation.get('input_digest') or 'unavailable')}")
        adoption = operation.get("adoption")
        if adoption:
            print(f"  Adopted published bytes: {_terminal_text(adoption['reason'])}; attempt "
                  f"{_terminal_text(adoption.get('attempt_id'))} keeps its unknown outcome and "
                  f"exposure {_usd(adoption['exposure_microusd'])}")
        for attempt in value["attempts"]:
            if attempt["operation_id"] != operation["operation_id"]:
                continue
            print(f"  Attempt {attempt['attempt_number']} ({_terminal_text(attempt['attempt_id'])}): "
                  f"{_terminal_text(attempt['status'])}; parent: {_terminal_text(attempt['parent_attempt_id'] or 'none')}")
            if attempt.get("reason"):
                print(f"    Reason: {_terminal_text(attempt['reason'])}")
            if attempt.get("error"):
                print(f"    Error: {_terminal_text(attempt['error'])}")
            for call in value["calls"]:
                if call["attempt_id"] == attempt["attempt_id"]:
                    print(f"    Call state: {_terminal_text(call['status'])} ({_terminal_text(call['call_id'])}); "
                          f"confirmed {_usd(call['confirmed_microusd'])}; exposure {_usd(call['exposure_microusd'])}")
        for artifact in value["artifacts"]:
            if artifact["operation_id"] == operation["operation_id"]:
                print(f"  Artifact {_terminal_text(artifact['relative_path'])}: "
                      f"{_terminal_text(json.dumps(artifact['integrity'], sort_keys=True))}")
    if page["has_more"]:
        print(f"More operations: use --offset {page['offset'] + page['returned']}")
    events = value["event_pagination"]
    print(f"Recent events: {events['returned']} of {events['total']}")
    if "report_path" in value:
        print(f"HTML report: {_terminal_text(value['report_path'], limit=1000)}")


def _command_line(argv: Sequence[str], *, windows: bool | None = None) -> str:
    """Quote *argv* for the shell it is shown in: cmd.exe on Windows, a POSIX shell elsewhere.

    ``shlex.join`` quotes with single quotes, which cmd.exe keeps as part of
    an argument; ``subprocess.list2cmdline`` quotes as Windows programs read
    their command line. *windows* defaults to whether this host is Windows.
    """
    if windows is None:
        windows = os.name == "nt"
    return subprocess.list2cmdline(argv) if windows else shlex.join(argv)


def _shown_command(argv: Sequence[str], *, windows: bool | None = None) -> str:
    """A suggested ``smythe`` command line, quoted for the shell it is shown in."""
    return f"smythe {_terminal_text(_command_line(argv, windows=windows), limit=1000)}"


def _operation_span(page: dict[str, Any]) -> str:
    """Which of the run's operations a page holds, by position."""
    first, last, total = page["offset"] + 1, page["offset"] + page["returned"], page["total"]
    if not page["returned"]:
        return f"none of {total} operations at offset {page['offset']}"
    return f"operation {first} of {total}" if first == last else f"operations {first}-{last} of {total}"


def _review_lines(items: list[dict[str, Any]], label, extra, *,
                  windows: bool | None = None) -> list[str]:
    """Review items as printed: kind and subject, then what was observed, the decision and commands.

    *label* names an item's subject; *extra* gives an observation's added lines.
    Commands are quoted as ``_command_line`` quotes them for *windows*.
    """
    lines = []
    for item in items:
        observed = item["observed"]
        lines.append(f"[{item['blocks']}] {item['kind']}: {_terminal_text(label(item))} "
                     f"(item {item['item_id'][:12]})")
        state = ", ".join(_terminal_text(part) for part in (observed["status"], observed["error_class"])
                          if part)
        lines.append(f"  Observed: {state}")
        if observed["excerpt"]:
            lines.append(f"  Detail: {_terminal_text(observed['excerpt'])}")
        lines.extend(extra(observed))
        lines.append(f"  Decision: {_terminal_text(item['decision_needed'], limit=1000)}")
        lines.extend(f"  Resolve with: {_shown_command(argv, windows=windows)}"
                     for argv in item["resolve_with"])
    return lines


def _review_notes(items: list[dict[str, Any]], option: str | None, *,
                  printed: bool = False) -> list[str]:
    """The reminder after review items whose commands need the store *option* or a reason.

    *printed* says another command line was shown, which also needs the option.
    """
    commands = [argv for item in items for argv in item["resolve_with"]]
    # Without it, every command line printed here would use the default store.
    shown = bool(commands) or printed
    notes = ([f"add the {option} option used here"] if option and shown else []) + (
        ["replace REASON with your own reason"] if any("REASON" in argv for argv in commands) else [])
    return [f"To run a command, {' and '.join(notes)}."] if notes else []


def _emit_job_review(value: dict[str, Any], *, store_given: bool, windows: bool | None = None) -> None:
    """Print a review packet at once, so an interruption leaves no partial output.

    Suggested commands are quoted for this host's shell, or for cmd.exe when
    *windows* is true and a POSIX shell when it is false.
    """
    page, items = value["page"], value["items"]
    lines = [f"Run: {_terminal_text(value['run_id'])}", f"Name: {_terminal_text(value['name'])}",
             f"Status: {_terminal_text(value['status'])}",
             f"Operations (whole run): {_terminal_text(json.dumps(value['operation_counts'], sort_keys=True))}",
             f"Needs a decision: {len(items)} ({_operation_span(page)}, limit {page['limit']})"]
    lines += _review_lines(
        items, lambda item: (item["subject"].get("operation_key") or item["subject"].get("launch_id")
                             or value["run_id"]),
        lambda observed: [f"  Kept exposure: {_usd(observed['exposure_microusd'])}"]
        if "exposure_microusd" in observed else [], windows=windows)
    if not items:
        lines.append("Nothing on this page needs a decision.")
    if page["has_more"]:
        lines.append(f"More operations: use --offset {page['offset'] + page['returned']}")
    unchecked = value["artifact_integrity"]["counts"].get("not_checked", 0)
    if unchecked:
        lines.append(f"Accepted files not hashed within the 64 MiB limit: {unchecked}; "
                     f"{_shown_command(['jobs', 'reroll', value['run_id'], '--dry-run'], windows=windows)}"
                     " hashes them all.")
    lines += _review_notes(items, "--store" if store_given else None, printed=bool(unchecked))
    print("\n".join(lines))


def _run_review_extra(observed: dict[str, Any]) -> list[str]:
    """A runs review item's unknown exposure and validators, when it has them."""
    lines = []
    if observed.get("unknown_nanousd"):
        calls = ", ".join(call["call_id"] for call in observed.get("unknown_calls") or ())
        lines.append(f"  Unknown exposure: {observed['unknown_nanousd']} nanoUSD"
                     + (f" on calls {_terminal_text(calls, limit=1000)}" if calls else ""))
    lines.extend(f"  Validator {_terminal_text(result['name'])} {_terminal_text(result['version'])}: "
                 f"{result['outcome']}" for result in observed.get("validators") or ())
    return lines


def _emit_run_review(value: dict[str, Any], *, journal: bool) -> None:
    """Print a runs review packet at once, so an interruption leaves no partial output."""
    page, items = value["page"], value["items"]
    store = "durable workflow journal" if value["store"] == "journal" else "ordinary checkpoints"
    lines = [f"Run: {_terminal_text(value['run_id'])} ({store})" if page is None else
             f"Runs reviewed: {page['returned']} of {page['total']} in {store}, newest first "
             f"(offset {page['offset']}, limit {page['limit']})", f"Needs a decision: {len(items)}"]
    lines += _review_lines(items, lambda item: " ".join(
        f"{key} {name}" if key != "run_id" else str(name) for key, name in item["subject"].items()),
        _run_review_extra)
    if not items:
        lines.append("Nothing needs a decision.")
    if page is not None and page["has_more"]:
        lines.append(f"More runs: use --offset {page['offset'] + page['returned']}")
    lines += _review_notes(items, "--journal" if journal else None)
    print("\n".join(lines))


def _emit_repair_preview(value: dict[str, Any]) -> None:
    print(f"Run: {_terminal_text(value['run_id'])}")
    print(f"Status: {_terminal_text(value['status'])}; nothing was changed")
    budget = value["budget"]
    print(f"Budget: {_usd(budget['remaining_microusd'])} of {_usd(budget['approved_microusd'])} "
          f"left{'; overrun' if budget['overrun'] else ''}")
    for attachment in value["attachments"]:
        if attachment["status"] != "verified":
            print(f"Attachment {_terminal_text(attachment['relative_path'])}: {attachment['status']}")
    for item in value["operations"]:
        reasons = f" ({', '.join(item['reasons'])})" if item["reasons"] else ""
        left = item["attempts_remaining"]
        plan = "; new plan and approval required" if item["new_plan_required"] else ""
        print(f"{_terminal_text(item['operation_key'])}: {item['action']}{reasons}; "
              f"{left} attempt{'' if left == 1 else 's'} left{plan}")


def _yes(value: Any) -> str:
    return "yes" if value else "no"


def _emit_run_list(value: dict[str, Any]) -> None:
    print(f"Runs: {value['returned']} of {value['total']} (offset {value['offset']}, limit {value['limit']})")
    for run in value["runs"]:
        if run["error"] is not None:
            print(f"{_terminal_text(run['run_id'])}  unreadable: {_terminal_text(run['error'])}")
            continue
        decision, unresolved = run["decision"] or "not recorded", run["unresolved_actions"]
        marker = f"  {unresolved} unresolved action{'' if unresolved == 1 else 's'}" if unresolved else ""
        print(f"{_terminal_text(run['run_id'])}  {run['state']}  acceptance {decision}  "
              f"next {run['next_action']}{marker}")
    if not value["runs"]:
        print("No runs matched.")
    if value["has_more"]:
        print(f"More runs: use --offset {value['offset'] + value['returned']}")


def _emit_lineage(lineage: dict[str, Any]) -> None:
    """A lineage run's record, links, integrity and derived status, by ids and digests only."""
    scope = _terminal_text(lineage["access_scope"])
    if not lineage["recorded"]:
        print(f"Lineage: not recorded yet; scope {scope}")
    else:
        nodes = ", ".join(f"{kind} {count}" for kind, count in lineage["nodes"].items())
        print(f"Lineage: {lineage['coverage']} coverage; scope {scope}; nodes {_terminal_text(nodes)}; "
              f"record sha256 {lineage['lineage_sha256']}")
        if lineage["undeclared"]:
            print(f"  Undeclared nodes: {_terminal_text(', '.join(lineage['undeclared']))}")
        for source in lineage["sources"]:
            version = f" version {_terminal_text(source['version'])}" if source["version"] else ""
            print(f"  Source {_terminal_text(source['source_id'])}{version}: "
                  f"sha256 {source['sha256'][:12]}")
        artifact = lineage["artifact"]
        if artifact is not None:
            print(f"  Artifact {_terminal_text(artifact['relative_path'])}: sha256 "
                  f"{artifact['sha256'][:12]}; integrity {artifact['integrity']}")
        for item in lineage["tombstones"]:
            version = f" version {_terminal_text(item['version'])}" if item["version"] else ""
            when = f" ({_terminal_text(item['removed_at'])})" if item["removed_at"] else ""
            print(f"  Tombstone {_terminal_text(item['source_id'])}{version}: sha256 "
                  f"{item['sha256'][:12]}; {_terminal_text(item['reason'])}{when}")
        for item in lineage["purged"]:
            print(f"  Purged {_terminal_text(item['relative_path'])}: sha256 {item['sha256'][:12]}; "
                  f"{_terminal_text(item['reason'])} ({_terminal_text(item['purged_at'])})")
    predecessor = lineage["predecessor"]
    successors = ", ".join(
        _terminal_text(item["run_id"]) + ("" if item["supersedes"] else " (does not supersede)")
        for item in lineage["successors"])
    print(f"  Predecessor: {_terminal_text(predecessor['run_id']) if predecessor else 'none'}; "
          f"successors: {successors or 'none'}")
    reasons = f" ({', '.join(lineage['reasons'])})" if lineage["reasons"] else ""
    status = lineage["status"] or "none; freshness needs a source catalog (Swarm.plan_repair)"
    print(f"  Derived status: {status}{reasons}")


def _emit_research(value: dict[str, Any]) -> None:
    """A durable run's context, reads, source evidence and open issues, without any text."""
    context = value.get("context")
    if context is not None:
        print(f"Context: assembly version {context['assembly_version']}, snapshot version "
              f"{context['snapshot_version']}; sources revision {context['sources_revision']}; "
              f"{len(context['sources'])} sources, {len(context['snapshots'])} snapshots")
        for source in context["sources"]:
            version = f" version {_terminal_text(source['version'])}" if source["version"] else ""
            removed = source["removed"]
            state = (f"removed {_terminal_text(removed['removed_at'])}: {_terminal_text(removed['reason'])}"
                     if removed else f"{source['text_bytes']} bytes")
            print(f"  Source {_terminal_text(source['source_id'])}{version}: {state}; "
                  f"sha256 {source['sha256'][:12]}")
    for call in value.get("tool_calls") or ():
        status = call["http_status"] if call["http_status"] is not None else "none"
        text = f"; {call['text_bytes']} bytes of text" if call["text_bytes"] is not None else ""
        error = f"; {_terminal_text(call['error'])}" if call["error"] else ""
        print(f"Read {_terminal_text(call['node_id'])} (generation {call['generation']}, attempt "
              f"{call['attempt']}): {_terminal_text(call['profile'])} "
              f"{_terminal_text(call['profile_version'])}; {call['state']}, "
              f"{call['result_state']}; HTTP {status}{text}; cost {call['cost_nanousd']} nanoUSD{error}")
    evidence = value.get("evidence")
    if evidence is not None:
        print(f"Evidence: {len(evidence['claims'])} claims citing {len(evidence['sources'])} sources "
              f"({evidence['state']})")
        for claim in evidence["claims"]:
            print(f"  Claim {_terminal_text(claim['claim_id'])}: {claim['supporting']} supporting, "
                  f"{claim['contradicting']} contradicting; sources "
                  f"{_terminal_text(', '.join(claim['sources']) or 'none')}")
    for issue in value.get("issues") or ():
        subject = f" ({_terminal_text(issue['subject'])})" if issue["subject"] is not None else ""
        print(f"Issue {issue['kind']}{subject}: {_terminal_text(issue['detail'])}")


def _emit_actions(actions: dict[str, Any]) -> None:
    """A durable run's external actions by identity, state, balances, approvals and record kinds."""
    unit = actions["unit"]
    print(f"External actions: {len(actions['items'])}; {actions['confirmed_nanousd']} confirmed, "
          f"{actions['reserved_nanousd']} reserved, {actions['unknown_nanousd']} unknown {unit}; "
          f"{len(actions['unresolved'])} unresolved")
    for item in actions["items"]:
        risk = "; duplicate risk" if item["duplicate_risk"] else ""
        if item["contradiction"] is not None:
            risk += "; " + item["contradiction"].replace("_", " ")
        attempts = len(item["attempts"])
        print(f"  Action {_terminal_text(item['slot'])}/{item['sequence']} ({_terminal_text(item['action_id'])}): "
              f"{item['state']}; {attempts} attempt{'' if attempts == 1 else 's'}; {item['confirmed_nanousd']} "
              f"confirmed, {item['unknown_nanousd']} unknown {unit}{risk}; next {item['next_action'] or 'none'}")
        for approval in item["approvals"]:
            principal, expiry = approval["principal"], approval["expires_at_ns"]
            standing = (f"revoked {approval['revocation']}" if approval["revoked_at_ns"] is not None
                        else "consumed" if approval["consumed_by_attempt"] else approval["verdict"])
            print(f"    Approval {_terminal_text(approval['approval_id'])}: {_terminal_text(principal['subject'])} of "
                  f"{_terminal_text(principal['tenant'])} ({_terminal_text(principal['issuer'])}); {standing}"
                  + ("" if expiry is None else f"; expires at {expiry} ns"))
        kinds: dict[str, int] = {}
        for record in item["records"]:
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        if kinds:
            print("    Records: " + ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())))


def _emit_revisions(log: list[dict[str, Any]]) -> None:
    """A run's revision log: one line per decision, then its reason and refusal detail."""
    if log:
        print(f"Revisions: {len(log)} logged")
    for entry in log:
        state = {True: "applied", False: "refused"}.get(entry["applied"], "undetermined")
        parts = [f"{verb} {_terminal_text(', '.join(entry[key]))}" for verb, key in (
            ("added", "added"), ("dropped", "dropped"), ("rewired", "rewired"),
            ("invalidated", "invalidated")) if entry[key]]
        parts.append(f"{entry['revisions_remaining_after']} revisions left, "
                     f"{entry['added_total']} added nodes in the graph")
        print(f"Revision after {_terminal_text(entry['after_node'])}: {state} "
              f"({_terminal_text(entry['change'])}); " + "; ".join(parts))
        for label, key in (("Reason", "reason"), ("Detail", "detail")):
            if entry[key]:
                print(f"  {label}: {_terminal_text(entry[key])}")


def _emit_routing(routing: dict[str, Any] | None) -> None:
    """One line for the tier the router chose, how, and what its classifier call used."""
    if routing is None:
        return
    if not routing["readable"]:
        print("Routing: a decision this build cannot read")
        return
    method = routing["method"]
    how = (f"rule {_terminal_text(routing['rule'] or 'unnamed')}" if method == "rule"
           else "the classifier" if method == "classifier" else method.replace("_", " "))
    line = f"Routing: tier {_terminal_text(routing['tier'])}, chosen by {how}"
    classifier = routing["classifier"]
    usage = None if classifier is None else classifier["usage"]
    if usage is not None:
        cost = ("cost unknown" if usage["cost_usd_unknown"] or usage["cost_usd"] is None
                else f"${usage['cost_usd']:.6f}" + (" estimated" if usage["cost_usd_is_estimate"]
                                                   else ""))
        line += (f"; classifier call: {usage['prompt_tokens']} prompt and "
                 f"{usage['completion_tokens']} completion tokens, {cost}")
    elif classifier is not None:
        calls = len(classifier["call_ids"])
        line += f"; classifier charge in {calls} journal call{'' if calls == 1 else 's'}"
    print(line)


def _emit_run_inspection(value: dict[str, Any]) -> None:
    print(f"Run: {_terminal_text(value['run_id'])}")
    saved = value["format"]
    if value["store"] == "checkpoints":
        print(f"Store: ordinary checkpoint, version {saved['checkpoint_version']}")
    else:
        print(f"Store: durable workflow journal, version {saved['journal_version']}; "
              f"checkpoint revision {saved['revision']}")
    execution = value["execution"]
    blocked = f" ({_terminal_text(execution['blocked_reason'])})" if execution["blocked_reason"] else ""
    print(f"Execution: {execution['state']}{blocked}; nodes: {json.dumps(execution['nodes'], sort_keys=True)}")
    _emit_routing(value.get("routing"))
    delivery = value["delivery"]
    record = delivery["record"]
    if record is None:
        print("Delivery acceptance: not recorded")
    else:
        recorded = "" if record["decision"] == delivery["decision"] else f" (recorded {record['decision']})"
        print(f"Delivery acceptance: {delivery['decision']}{recorded}")
        for reason in delivery["reasons"]:
            print(f"  Reason: {_terminal_text(reason)}")
        if record["output_sha256"] is not None:
            print(f"  Output: sha256 {record['output_sha256']}; {_terminal_text(record['strategy'])} "
                  f"synthesis; integrity {delivery['integrity']}")
        for result in record["validators"]:
            identity = result["validator"]
            print(f"  Validator {_terminal_text(identity['name'])} {_terminal_text(identity['version'])} "
                  f"(config {identity['config_sha256'][:12]}): {result['outcome']}")
        print(f"  Policy: {record['policy_sha256']}")
    decisions = value["decisions"]
    if decisions["history"]:
        print(f"Decision history: decision {decisions['current']} is current; earlier decisions:")
        for entry in decisions["history"]:
            earlier = entry["record"]
            reason = f": {_terminal_text(earlier['reasons'][0])}" if earlier["reasons"] else ""
            print(f"  Decision {entry['sequence']}: {earlier['decision']} under policy "
                  f"{earlier['policy_sha256'][:12]}{reason}")
    _emit_revisions(value.get("revision_log") or [])
    withheld = delivery["rejected_output"]
    if withheld is not None:
        print(f"Withheld output: {withheld['bytes']} bytes in the saved {withheld['field']} field; "
              f"sha256 {withheld['sha256']}")
    artifact = value["artifact"]
    if artifact is not None:
        published, eligibility = artifact["publication"], artifact["delivery_eligibility"]
        print(f"Artifact: {_terminal_text(artifact['path'], limit=1000)} (publication "
              f"{published['sequence']}); {published['size_bytes']} bytes; "
              f"sha256 {published['sha256']}")
        print(f"  Integrity: {artifact['integrity']['status']}: "
              f"{_terminal_text(artifact['integrity']['detail'])}")
        if artifact["superseded"]:
            print("  Supersedes: " + ", ".join(
                f"publication {item['sequence']} ({_terminal_text(item['name'])})"
                for item in artifact["superseded"]))
        eligible = "eligible" if eligibility["eligible"] else (
            "not eligible (" + ", ".join(eligibility["reasons"]) + ")")
        print(f"  Delivery eligibility: {eligible}")
    if value.get("lineage") is not None:
        _emit_lineage(value["lineage"])
    _emit_research(value)
    if value.get("actions") is not None:
        _emit_actions(value["actions"])
    accounting = value["accounting"]
    print(f"Cost: ${accounting['total_cost_usd']:.6f} ({accounting['cost_scope']}); "
          f"complete: {_yes(accounting['cost_is_complete'])}; "
          f"includes estimates: {_yes(accounting['cost_contains_estimates'])}")
    exposure = accounting["unknown_exposure"]
    if exposure is None:
        print("Unknown exposure: not tracked for ordinary runs")
    else:
        print(f"Unknown exposure: {exposure['amount']} {exposure['unit']} on {exposure['calls']} calls")
    if accounting["unknown_cost_nodes"]:
        print(f"Unknown cost on nodes: {_terminal_text(', '.join(accounting['unknown_cost_nodes']))}")
    for item in accounting["unresolved"]:
        print(f"Unresolved: {_terminal_text(item)}")
    timing = value["timing"]
    if timing is None:
        print("Timing: not recorded")
    else:
        print(f"Timing: {timing['elapsed_ms']} ms elapsed over {timing['segments']} "
              f"segment(s), {timing['clock']} clock")
        if timing["phases_ms"]:
            phases = ", ".join(f"{name} {ms} ms" for name, ms in timing["phases_ms"].items() if ms)
            print(f"Phases: {_terminal_text(phases or 'none measured')}")
    action = value["next_action"]
    print(f"Next action: {action['action']}: {_terminal_text(action['detail'], limit=1000)}")


def _policy_text(entry: dict[str, Any]) -> str:
    """An event's or version's policy hash and canonical JSON, or that it sets none."""
    if entry["policy_hash"] is None:
        return "no policy in effect"
    policy = json.dumps(entry["policy"], sort_keys=True, ensure_ascii=False)
    return f"policy {entry['policy_hash']} {_terminal_text(policy, limit=1000)}"


def _event_line(event: dict[str, Any]) -> str:
    restores = "" if event["restores_seq"] is None else f" of seq {event['restores_seq']}"
    review = f"; review {event['review_id']}" if event["review_id"] else ""
    return (f"Seq {event['seq']}: {event['action']}{restores}, {_policy_text(event)}{review}; "
            f"{_terminal_text(event['actor'])}: {_terminal_text(event['reason'])}")


def _unknown_text(trials: int, microusd: int) -> str:
    """Unknown outcomes by count, so one under a zero ceiling never reads as nothing."""
    if not trials:
        return "none"
    return f"{trials} unknown trial{'' if trials == 1 else 's'} holding {_usd(microusd)}"


def _cost_text(cost: dict[str, Any]) -> str:
    return (f"confirmed {_usd(cost['confirmed_microusd'])}; reserved {_usd(cost['reserved_microusd'])}; "
            f"total exposure {_usd(cost['total_exposure_microusd'])}")


def _emit_registry(command: str, value: dict[str, Any]) -> None:
    """Print a policy review, a target change or a target report at once."""
    if command == "review":
        incumbent = {"policy_hash": value["incumbent_policy_hash"], "policy": value["incumbent_policy"]}
        lines = [f"Review: {value['review_id']}",
                 f"Campaign: {_terminal_text(value['campaign_id'])}; decision {value['decision_id']} "
                 f"({'promoted' if value['promoted'] else 'not promoted'})",
                 f"Verdict: {value['verdict']} by {_terminal_text(value['reviewer'])}",
                 f"Candidate {_policy_text(value)}", f"Incumbent {_policy_text(incumbent)}"]
        if value["evidence_sha256"]:
            lines.append(f"Evidence: sha256 {value['evidence_sha256']}")
        if value["note"]:
            lines.append(f"Note: {_terminal_text(value['note'])}")
    elif command != "history":
        lines = [f"Target: {value['target']}", *map(_event_line, value["events"]),
                 "Configuration was not changed; new work resolves the policy in effect "
                 "from the registry."]
    else:
        active, events = value["active"], value["history"]
        lines = [f"Target: {value['target']}",
                 "In effect: nothing" if active is None else
                 f"In effect: seq {active['seq']}, {_policy_text(active)} from seq {active['source_seq']}",
                 f"History: {len(events)} event{'' if len(events) == 1 else 's'}",
                 *("  " + _event_line(event) for event in events),
                 f"Campaigns touching the target: {len(value['campaigns'])}"]
        for campaign in value["campaigns"]:
            decision = campaign["decision"]
            state = "no terminal decision" if decision is None else (
                f"{'promoted' if decision['promoted'] else 'not promoted'}, verdict "
                f"{decision['verdict'] or 'not recorded'}; decision {decision['decision_id']}")
            lines += [f"Campaign {_terminal_text(campaign['campaign_id'])}: {state}",
                      f"  Cost: {_cost_text(campaign['cost'])}",
                      "  Unknown exposure: " + _unknown_text(
                          campaign["unknown_trials"], campaign["cost"]["unknown_exposure_microusd"])]
            lines += [f"  Review {review['review_id']}: {review['verdict']} by "
                      f"{_terminal_text(review['reviewer'])}" for review in campaign["reviews"]]
        total = value["optimization_cost"]
        lines += [f"Optimization cost of these campaigns, not netted against inference savings: "
                  f"{_cost_text(total)}",
                  "Unknown exposure: " + _unknown_text(total["unknown_trials"],
                                                       total["unknown_exposure_microusd"])]
        elsewhere = value["campaigns_not_in_ledger"]
        if elsewhere:
            lines.append(f"Not in this ledger: {', '.join(map(_terminal_text, elsewhere))} "
                         "(named by an activation; cost not included)")
    print("\n".join(lines))


def _emit(args: argparse.Namespace, command: str, value: Any) -> None:
    payload = {"ok": True, "command": command, command: value}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return
    if args.root_command == "optimize" and command in _REGISTRY_COMMANDS:
        _emit_registry(command, value)
        return
    if args.root_command == "runs":
        if command == "list":
            _emit_run_list(value)
        elif command == "review":
            _emit_run_review(value, journal=args.journal is not None)
        elif command == "revoke-action":
            revoked = value["revoked"]
            print(f"Revoked {len(revoked)} approvals of {_terminal_text(value['target'])}"
                  + (": " + ", ".join(map(_terminal_text, revoked)) if revoked else "; none was in force"))
        else:
            _emit_run_inspection(value)
    elif command == "validate":
        print(f"Valid job manifest: {value['name']}")
        print(f"Manifest: {value['manifest_hash']}")
        print(f"Plan: {value['plan_hash']}")
        print(f"Operations: {value['operation_count']}")
        print(f"Worst-case cost: ${value['worst_case_cost_usd']}")
    elif command == "plan":
        print(f"Plan: {value['plan']['plan_hash']}")
        print(f"Operations: {len(value['plan']['operations'])}")
        print(f"Worst-case cost: ${value['plan']['worst_case_cost_usd']}")
        print(f"Approval token: {value['approval']['token']}")
    elif command == "reroll" and getattr(args, "dry_run", False):
        _emit_repair_preview(value)
    elif command in {"run", "status", "resume", "reroll", "adopt", "recheck"}:
        job = value
        print(f"Run: {_terminal_text(job['run_id'])}")
        print(f"Status: {_terminal_text(job['status'])}")
        if job.get("detached"):
            print(f"Worker PID: {job['worker_pid']}")
            print(f"Worker log: {_terminal_text(job['log_path'], limit=1000)}")
            print("Use jobs attach to observe this run.")
        else:
            print(f"Operations: {json.dumps(job['counts'], sort_keys=True)}")
    elif command in {"attach", "stop"}:
        print(f"Run: {_terminal_text(value['run_id'])}")
        if "stop_request" in value:
            print(f"Stop request generation: {value['stop_request']['pause_generation']}")
        print(f"Attachment: {_terminal_text(value['attachment']['state'])}")
        if "status" in value:
            print(f"Status: {_terminal_text(value['status'])}")
            print(f"Operations: {json.dumps(value['counts'], sort_keys=True)}")
        if value.get("worker"):
            print(f"Worker log: {_terminal_text(value['worker']['log_path'], limit=1000)}")
    elif command == "export" and isinstance(value, dict) and "path" in value:
        print(f"Exported {value['run_id']} to {value['path']}")
    elif command == "list":
        _emit_job_list(value)
    elif command == "inspect" and args.root_command == "jobs":
        _emit_job_inspection(value)
    elif command == "review":
        _emit_job_review(value, store_given=args.store != str(_default_store()))
    else:
        print(json.dumps(value, indent=2, ensure_ascii=False))


def _emit_error(
    args: argparse.Namespace,
    command: str,
    error: BaseException,
) -> None:
    payload = {
        "ok": False,
        "command": command,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    from smythe.jobs.operator import WorkerStartupError, WorkerStartupInterrupted

    if isinstance(error, (WorkerStartupError, WorkerStartupInterrupted)):
        payload["error"]["launch"] = error.launch
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":") if args.json else None,
    )
    print(text if args.json else f"error: {_terminal_text(error, limit=1000)}",
          file=sys.stdout if args.json else sys.stderr)


async def _run_optimize_concurrency(args: argparse.Namespace) -> dict[str, Any]:
    # Kept local so the established Jobs CLI remains importable independently
    # of the optional optimization surface during rolling upgrades.
    from smythe.optimize.engine import OptimizationRunner, TrialOutcome

    contract, incumbent, challengers, scenario, references = _concurrency_campaign(args)

    async def evaluate(context: Any) -> Any:
        simulated = await asyncio.to_thread(
            simulate_concurrency,
            context.candidate.policy,
            scenario=scenario,
            seed=context.seed,
            split=context.split,
            deadline=context.deadline_monotonic,
        )
        accounted_operations = round(
            simulated["successful_operations"]
            + simulated["error_rate"] * scenario.work_items
        )
        return TrialOutcome(
            metrics={
                "throughput_ops_s": simulated["throughput_ops_s"],
                "p95_latency_ms": simulated["p95_latency_ms"],
                "error_rate": simulated["error_rate"],
            },
            gates={
                "all_operations_accounted": (
                    accounted_operations == scenario.work_items
                )
            },
            actual_cost_microusd=0,
            artifact_hashes=(),
        )

    with ExperimentLedger(
        args.ledger,
        durability=args.ledger_durability,
    ) as ledger:
        runner = OptimizationRunner(
            contract,
            ledger,
            evaluate,
            evaluator_hash=scenario.evaluator_hash,
            bootstrap_resamples=args.bootstrap_resamples,
            campaign_id=args.campaign_id,
        )
        result = await runner.run(incumbent, challengers, references=references)

    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in (incumbent, *challengers)
    }
    selected = (
        candidates_by_id.get(result.selected_candidate_id)
        if result.selected_candidate_id is not None
        else None
    )
    if result.selected_candidate_id is not None and selected is None:
        raise OptimizationNeedsAttention(
            "optimization result selected a candidate outside the immutable plan"
        )
    if result.promoted and selected is None:
        raise OptimizationNeedsAttention(
            "optimization result promoted without selecting a candidate"
        )
    result_data = result.to_dict()
    ledger_snapshot = result_data.pop("ledger_snapshot")
    evidence = {
        key: value
        for key, value in result_data.items()
        if key
        not in {
            "campaign_id",
            "optimization_plan_hash",
            "selected_candidate_id",
            "promoted",
            "verdict",
        }
    }
    payload: dict[str, Any] = {
        "campaign_id": result.campaign_id,
        "optimization_plan_hash": result.optimization_plan_hash,
        "evaluator_hash": scenario.evaluator_hash,
        "contract_hash": contract.contract_hash,
        "ledger_durability": args.ledger_durability,
        "promoted": result.promoted,
        "verdict": result.verdict,
        "selected_candidate": selected.to_dict() if selected is not None else None,
        "evidence": evidence,
        "ledger_snapshot": ledger_snapshot,
    }
    if result.promoted:
        assert selected is not None
        payload["recommended_patch"] = dict(selected.to_dict()["policy"])
    return payload


def _check_registry_arguments(args: argparse.Namespace) -> None:
    """Refuse malformed registry input before any file is opened or created."""
    from smythe.optimize.ledger import _nonempty, _safe_id
    from smythe.optimize.registry import _seq, _target

    command = args.optimize_command
    if command == "review":
        _safe_id(args.campaign_id, "campaign_id")
        _nonempty(args.reviewer, "reviewer", maximum=128)
        if args.note != "":
            _nonempty(args.note, "note", maximum=1024)
        return
    _target(args.target)
    if command != "history":
        _seq(args.expected_seq, "expected-seq", minimum=0)
        if command == "rollback":
            _seq(args.to_seq, "to-seq", minimum=1)
        _nonempty(args.actor, "actor", maximum=128)
        _nonempty(args.reason, "reason", maximum=1024)


def _evidence_sha256(path: str | None) -> str | None:
    """The plain hex SHA-256 of an evidence file; an unreadable file is invalid input."""
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot read evidence file {path!r}: {exc.strerror or exc}") from exc
    return digest.hexdigest()


def _dispatch_registry(args: argparse.Namespace) -> int:
    """Record a review, change a target, or report one; history opens both files read-only."""
    from smythe.optimize.inspection import collect_target_report
    from smythe.optimize.registry import PolicyRegistry, _reviewable

    command = args.optimize_command
    _check_registry_arguments(args)
    if command == "history":
        with ExperimentLedger(args.ledger, read_only=True) as ledger, \
                PolicyRegistry(args.registry, read_only=True) as registry:
            value = collect_target_report(ledger, registry, args.target)
    elif command == "review":
        evidence = _evidence_sha256(args.evidence)
        # Read-only like inspect; checked first so a missing ledger creates no registry.
        if not Path(args.ledger).is_file():
            raise FileNotFoundError(f"no optimization ledger at {args.ledger}")
        if not Path(args.registry).exists():
            # A review that would be refused must not leave a new, empty registry.
            _reviewable(args.ledger, args.campaign_id, args.verdict)
        with PolicyRegistry(args.registry) as registry:
            value = registry.record_review(
                args.ledger, args.campaign_id, reviewer=args.reviewer, verdict=args.verdict,
                evidence_sha256=evidence, note=args.note,
            ).to_dict()
    else:
        # A change acts on recorded reviews and history, so it never creates a registry.
        if not Path(args.registry).is_file():
            raise RegistryError(f"no policy registry at {args.registry}; a review creates one")
        change = {"expected_seq": args.expected_seq, "actor": args.actor, "reason": args.reason}
        with PolicyRegistry(args.registry) as registry:
            if command == "activate":
                event = registry.activate(args.target, args.review_id, **change)
            elif command == "rollback":
                event = registry.rollback(args.target, to_seq=args.to_seq, **change)
            else:
                event = registry.retire(args.target, **change)
            # Compare-and-swap makes every event after the expected head this change's own,
            # such as the baseline a first activation records.
            events = registry.history(event.target)[args.expected_seq:event.seq]
        value = {"target": event.target, "events": [item.to_dict() for item in events]}
    _emit(args, command, value)
    return EXIT_OK


def _dispatch_optimize(args: argparse.Namespace) -> int:
    command = args.optimize_command
    if command in _REGISTRY_COMMANDS:
        return _dispatch_registry(args)
    if command == "inspect":
        destination = _optimization_report_destination(args.out, args.ledger) if args.out is not None else None
        with ExperimentLedger(args.ledger, read_only=True) as ledger:
            if destination is not None:
                from smythe.optimize.inspection import collect_optimization_report

                report = collect_optimization_report(ledger, args.campaign_id)
                snapshot = report["ledger_snapshot"]
                evaluator_hashes = report["evaluator_hashes"]
            else:
                snapshot = ledger.snapshot(args.campaign_id)
                evaluator_hashes = sorted(
                    {
                        trial.evaluator_hash
                        for trial in ledger.list_trials(args.campaign_id)
                    }
                )
        if destination is not None:
            from smythe.jobs.artifact_io import atomic_publish_bytes
            from smythe.optimize.report import render_optimization_report

            # Close-time fingerprint/WAL validation must succeed before any
            # output directory or file is created. Recheck aliases immediately
            # before the atomic, no-clobber publication boundary.
            if _optimization_report_destination(args.out, args.ledger) != destination:
                raise ValueError("Report output path changed during inspection")
            try:
                html = render_optimization_report(report)
                atomic_publish_bytes(destination, html.encode("utf-8"))
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                raise ExperimentLedgerError("Cannot publish retained Autotune report evidence") from exc
        _emit(
            args,
            command,
            {
                "campaign_id": args.campaign_id,
                "evaluator_hashes": evaluator_hashes,
                "ledger_snapshot": snapshot,
            },
        )
        return EXIT_OK
    if command == "concurrency":
        payload = asyncio.run(_run_optimize_concurrency(args))
        _emit(args, command, payload)
        return EXIT_OK
    raise AssertionError(command)  # pragma: no cover - argparse guarantees this


def _dispatch_jobs(args: argparse.Namespace) -> int:
    command = args.jobs_command
    if command == "schema":
        _emit(args, command, JOB_MANIFEST_V1_JSON_SCHEMA)
        return EXIT_OK

    if command in {"validate", "plan", "run"}:
        if command == "run" and args.detach:
            from smythe.jobs.operator import validate_startup_timeout

            validate_startup_timeout(args.startup_timeout_s)
        plan, manifest_root = _plan(args.manifest)
        if command == "validate":
            _emit(
                args,
                command,
                {
                    "name": plan.name,
                    "manifest_hash": plan.manifest_hash,
                    "plan_hash": plan.plan_hash,
                    "operation_count": len(plan.operations),
                    "worst_case_cost_usd": str(plan.worst_case_cost_usd),
                    "max_budget_usd": str(plan.max_budget_usd),
                },
            )
            return EXIT_OK

        approval = make_approval(
            plan,
            approved_max_cost_usd=args.max_spend_usd,
        )
        if command == "plan":
            _emit(
                args,
                command,
                {"plan": plan.to_dict(), "approval": approval.to_dict()},
            )
            return EXIT_OK

        if not hmac.compare_digest(args.approve, approval.token):
            raise ApprovalError(
                "approval token does not match this manifest and spend ceiling"
            )
        if args.detach:
            from smythe.jobs.operator import launch_worker

            with SQLiteRunStore(args.store) as store:
                # Preserve start()'s local attachment/provider checks before
                # creating the approved run. The worker revalidates on resume.
                JobRunner(store)._validate_dispatch_inputs(plan, manifest_root)
                run_id = store.create_run(plan, approval, manifest_root=manifest_root,
                                          allow_approval_reuse=args.new_run)
            launched = launch_worker(args.store, run_id, startup_timeout_s=args.startup_timeout_s)
            _emit(args, command, launched)
            return EXIT_OK
        with SQLiteRunStore(args.store) as store:
            snapshot = asyncio.run(
                JobRunner(store).start(
                    plan,
                    approval,
                    manifest_root=manifest_root,
                    allow_approval_reuse=args.new_run,
                )
            )
        _emit(args, command, snapshot)
        return _job_exit(snapshot)

    if command == "resume" and args.detach:
        from smythe.jobs.operator import launch_worker

        launched = launch_worker(args.store, args.run_id, startup_timeout_s=args.startup_timeout_s,
                                 clear_pause=True)
        _emit(args, command, launched)
        return EXIT_OK

    if command in {"attach", "stop"}:
        from smythe.jobs.operator import attach_job, stop_job

        def update(value):
            if not args.json:
                print(f"{_terminal_text(value['status'])}: "
                      f"{json.dumps(value['counts'], sort_keys=True)}", file=sys.stderr)

        try:
            observe = stop_job if command == "stop" else attach_job
            snapshot = observe(args.store, args.run_id, timeout_s=args.timeout_s,
                               poll_interval_s=args.poll_interval_s, on_update=update,
                               **({"reason": args.reason, "drain_timeout_s": args.drain_timeout_s}
                                  if command == "stop" else {}))
        except KeyboardInterrupt:
            _emit(args, command, {"run_id": args.run_id, "attachment": {"state": "disconnected"}})
            return EXIT_INTERRUPTED
        _emit(args, command, snapshot)
        if snapshot["attachment"]["state"] == "disconnected":
            return EXIT_INTERRUPTED
        if command == "attach" and snapshot["attachment"]["state"] in {"worker_failed", "lease_expired"}:
            return EXIT_LOCAL_ERROR
        return _job_exit(snapshot)

    if command in {"list", "inspect"}:
        from smythe.jobs.inspection import inspect_job, list_jobs

        destination = _report_destination(args.out, args.store) if command == "inspect" and args.out else None
        with SQLiteRunStore(args.store, read_only=True) as store:
            if command == "list":
                payload = list_jobs(store, limit=args.limit, offset=args.offset, status=args.status)
            else:
                payload = inspect_job(store, args.run_id, operation=args.operation,
                                      limit=args.limit, offset=args.offset, events_limit=args.events_limit)
        if destination is not None:
            from smythe.jobs.report import render_job_report

            _write_html(destination, render_job_report(payload))
            payload = dict(payload, report_path=str(destination))
        _emit(args, command, payload)
        return EXIT_OK

    if command == "review":
        from smythe.jobs.inspection import review_job

        try:
            with SQLiteRunStore(args.store, read_only=True) as store:
                packet = review_job(store, args.run_id, limit=args.limit, offset=args.offset)
            # Each format is rendered whole before it is written.
            _emit(args, command, packet)
        except KeyboardInterrupt:
            _emit_error(args, command, KeyboardInterrupt("review interrupted"))
            return EXIT_INTERRUPTED
        return EXIT_OK

    if command == "reroll" and args.dry_run:
        with SQLiteRunStore(args.store, read_only=True) as store:
            preview = JobRunner(store).repair_preview(args.run_id, args.operation_keys or None)
        _emit(args, command, preview)
        return EXIT_OK

    if command == "export" and args.out:
        _report_destination(args.out, args.store)
    with SQLiteRunStore(args.store, read_only=command in {"status", "export"}) as store:
        if command == "status":
            snapshot = store.snapshot(args.run_id, include_events=args.events)
        elif command == "resume":
            snapshot = asyncio.run(JobRunner(store).resume(args.run_id))
        elif command == "reroll":
            snapshot = asyncio.run(
                JobRunner(store).reroll(
                    args.run_id,
                    args.operation_keys,
                    reason=args.reason,
                    acknowledge_unknown=args.acknowledge_unknown,
                    repair_integrity=args.repair_integrity,
                )
            )
        elif command == "adopt":
            snapshot = asyncio.run(
                JobRunner(store).adopt(args.run_id, args.operation_key, reason=args.reason)
            )
        elif command == "recheck":
            snapshot = asyncio.run(
                JobRunner(store).recheck(args.run_id, args.operation_keys, reason=args.reason)
            )
        elif command == "export":
            snapshot = _portable_export(
                store.snapshot(args.run_id, include_events=True)
            )
            if args.out:
                destination = _write_json(args.out, snapshot)
                _emit(
                    args,
                    command,
                    {"run_id": args.run_id, "path": str(destination)},
                )
            else:
                _emit(args, command, snapshot)
            return _job_exit(snapshot)
        else:  # pragma: no cover - argparse guarantees the command set
            raise AssertionError(command)
    _emit(args, command, snapshot)
    return _job_exit(snapshot)


def _revoke_action(args: argparse.Namespace) -> int:
    """Revoke as ``smythe.actions.revoke`` does, opening for writing only a journal that has actions."""
    from smythe.actions import revoke
    from smythe.workflow_store import SQLiteWorkflowStore, WorkflowError, WorkflowNotFoundError

    if args.journal is None:
        raise ValueError("revoke-action needs --journal: only durable runs have external actions")
    with open_journal(args.journal) as journal:
        # A version 1 journal has no actions, and opening it for writing would upgrade it.
        if journal.store_version < 2:
            raise RunNotFoundError("This workflow journal has no external actions")
    try:
        with SQLiteWorkflowStore(args.journal) as store:
            revoked = revoke(store, args.target, args.reason)
    except WorkflowNotFoundError as exc:
        raise RunNotFoundError(f"No external action or approval {args.target!r} in this journal") from exc
    except WorkflowError as exc:
        if isinstance(exc, ValueError):
            raise  # an invalid reason or id is invalid input
        raise RunInspectionError(f"Cannot revoke in the workflow journal: {exc}") from exc
    _emit(args, "revoke-action", {"target": args.target, "revoked": list(revoked)})
    return EXIT_OK


def _open_runs(args: argparse.Namespace) -> Any:
    if args.journal is not None:
        return open_journal(args.journal)
    return open_checkpoints(args.checkpoints if args.checkpoints is not None else _default_checkpoints())


def _review_runs(args: argparse.Namespace) -> int:
    """Review saved runs; an interruption prints only its structured error, never part of the packet."""
    try:
        store = _open_runs(args)
        try:
            packet = review_run(store, args.run_id, limit=args.limit, offset=args.offset)
        finally:
            if args.journal is not None:
                store.close()
        # Each format is rendered whole before it is written.
        _emit(args, "review", packet)
    except KeyboardInterrupt:
        _emit_error(args, "review", KeyboardInterrupt("review interrupted"))
        return EXIT_INTERRUPTED
    return EXIT_OK


def _dispatch_runs(args: argparse.Namespace) -> int:
    if args.journal is not None and args.checkpoints is not None:
        raise ValueError("Choose --checkpoints or --journal, not both")
    if args.runs_command == "revoke-action":
        return _revoke_action(args)
    if args.runs_command == "review":
        return _review_runs(args)
    store = _open_runs(args)
    try:
        if args.runs_command == "list":
            _emit(args, "list", list_runs(store, limit=args.limit, offset=args.offset))
            return EXIT_OK
        value = inspect_run(store, args.run_id)
    finally:
        if args.journal is not None:
            store.close()
    _emit(args, "inspect", value)
    decision = value["delivery"]["decision"]
    if decision in NOT_ACCEPTED:
        return EXIT_RUN_NOT_ACCEPTED
    if decision == "indeterminate":
        return EXIT_RUN_INDETERMINATE
    if decision == "accepted" and value["actions"] is not None and value["actions"]["unresolved"]:
        return EXIT_RUN_ACTION_UNRESOLVED
    return EXIT_OK


def _dispatch(args: argparse.Namespace) -> int:
    if args.root_command == "jobs":
        return _dispatch_jobs(args)
    if args.root_command == "optimize":
        return _dispatch_optimize(args)
    if args.root_command == "runs":
        return _dispatch_runs(args)
    raise AssertionError(args.root_command)  # pragma: no cover


def _parse_arguments(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Parse ``argv`` like ``parse_args``, with every operation key of ``jobs reroll``.

    A reroll's operation keys are optional, so that ``--dry-run`` can preview a
    whole run. Python 3.11 matches them as empty once an option follows the run
    ID and leaves the keys after that option unrecognized, so a reroll's
    leftover plain arguments are its operation keys on every Python version.
    """
    args, extras = parser.parse_known_args(argv)
    if args.root_command == "jobs" and args.jobs_command == "reroll":
        keys, unrecognized, separated = list(args.operation_keys or ()), [], False
        for token in extras:
            if token == "--" and not separated:
                separated = True
            elif separated or not token.startswith("-"):
                keys.append(token)
            else:
                unrecognized.append(token)
        args.operation_keys, extras = keys, unrecognized
        missing = [name for name, absent in (("operation_keys", not keys),
                                             ("--reason", args.reason is None)) if absent]
        if missing and not args.dry_run:
            parser.error("the following arguments are required: " + ", ".join(missing))
    if extras:
        parser.error(f"unrecognized arguments: {' '.join(extras)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Smythe CLI and return a stable process exit code."""
    parser = build_parser()
    args = _parse_arguments(parser, list(argv) if argv is not None else None)
    command = getattr(args, f"{args.root_command}_command", args.root_command)
    try:
        return _dispatch(args)
    except KeyboardInterrupt as exc:
        from smythe.jobs.operator import WorkerStartupInterrupted

        if not isinstance(exc, WorkerStartupInterrupted):
            raise
        _emit_error(args, command, exc)
        return EXIT_INTERRUPTED
    except ApprovalError as exc:
        _emit_error(args, command, exc)
        return EXIT_APPROVAL
    except JobBudgetError as exc:
        _emit_error(args, command, exc)
        return EXIT_BUDGET
    except LedgerBudgetError as exc:
        _emit_error(args, command, exc)
        return EXIT_OPTIMIZE_LIMIT
    except OptimizationLimitError as exc:
        _emit_error(args, command, exc)
        return EXIT_OPTIMIZE_LIMIT
    except (
        CampaignNotFoundError,
        LedgerConflictError,
        OptimizationNeedsAttention,
        RegistryError,
        TrialStateError,
    ) as exc:
        _emit_error(args, command, exc)
        return EXIT_OPTIMIZE_STATE
    except (JobNotFoundError, InvalidTransitionError, RunLeaseError, RunNotFoundError) as exc:
        _emit_error(args, command, exc)
        return EXIT_JOB_STATE
    except RunInspectionError as exc:
        _emit_error(args, command, exc)
        return EXIT_LOCAL_ERROR
    except (
        ProviderConfigurationError,
        ImportError,
    ) as exc:
        _emit_error(args, command, exc)
        return EXIT_PREFLIGHT
    except (
        ContractValidationError,
        ManifestValidationError,
        PreflightError,
        ValueError,
    ) as exc:
        _emit_error(args, command, exc)
        return EXIT_INVALID_INPUT
    except (
        ExperimentLedgerError,
        OptimizationError,
        RunStoreError,
        OSError,
        sqlite3.Error,
    ) as exc:
        _emit_error(args, command, exc)
        return (
            EXIT_OPTIMIZE_STATE
            if args.root_command == "optimize"
            else EXIT_LOCAL_ERROR
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
