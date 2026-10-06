"""Bounded execution for preflighted artifact jobs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from uuid import uuid4

from smythe.acceptance import ValidationSubject
from smythe.budget import BudgetValidationError, validate_completion_usage
from smythe.jobs.artifact_io import (
    MAX_ARTIFACT_BYTES,
    ArtifactInspection,
    ArtifactInspectionError,
    atomic_publish_bytes,
    claim_artifact_root,
    inspect_artifact,
)
from smythe.jobs.delivery import (
    RESPONSE_PATH,
    annotate_delivery,
    artifact_spec_identity,
    delivery_record,
    run_validator,
    validator_registry,
    validator_result,
)
from smythe.jobs.inspection import (
    _artifact_root,
    _check_path,
    _relative_parts,
    check_artifact_integrity,
)
from smythe.jobs.models import (
    ArtifactSpecV1,
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_ATTACHMENT_BYTES,
    MAX_CALL_TIMEOUT_S,
    JobManifestV1,
    ProviderKind,
    usd_to_micros,
)
from smythe.jobs.preflight import (
    JobApprovalV1,
    JobPlanV1,
    PlannedOperationV1,
    PreflightError,
    preflight_job,
    verify_approval,
)
from smythe.jobs.providers import ProviderPool
from smythe.jobs.store import (
    MAX_ARTIFACTS_PER_CALL,
    MAX_SQLITE_INTEGER,
    InvalidTransitionError,
    JobBudgetError,
    JobNotFoundError,
    OperationStatus,
    RunLease,
    RunLeaseError,
    RunPauseRequested,
    RunStatus,
    RunStoreError,
    SQLiteRunStore,
)
from smythe.provider import Artifact, CompletionResult
from smythe.tools import ChatMessage
from smythe.validation import ValidationFinding, ValidationReport, ValidatorIdentity, hard_finding


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MAX_TOTAL_ARTIFACT_BYTES_PER_CALL = 128 * 1024 * 1024
_INVALID_ACCOUNTING_PREFIX = "invalid provider accounting: "
_DRAIN_POLL_S = 1.0
# A repair preview's actions, from least to most conservative. Jobs keep no
# rendering input, so what the shared vocabulary would rerender regenerates.
_PREVIEW_ACTIONS = ("reuse", "recheck", "regenerate", "review")
# The shared reason code for each delivery eligibility reason with another name.
_PREVIEW_REASONS = {
    "not_accepted": "decision_not_accepted", "record_invalid": "decision_not_accepted",
    "record_mismatch": "decision_not_accepted", "validator_unrecorded": "validator_set_changed",
    "validator_changed": "validator_set_changed", "artifact_not_checked": "artifact_unchecked",
    "no_decision": "node_incomplete",
}
# The reason code for an operation that has not succeeded.
_STATE_REASONS = {"pending": "node_incomplete", "failed": "node_incomplete",
                  "rejected": "node_incomplete", "running": "run_in_progress",
                  "unknown_outcome": "outcome_unknown"}


@dataclass(frozen=True, slots=True)
class JobExecutionMetrics:
    operations_started: int
    peak_active_calls: int
    wall_s: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "operations_started": self.operations_started,
            "peak_active_calls": self.peak_active_calls,
            "wall_s": round(self.wall_s, 6),
        }


@dataclass(frozen=True, slots=True)
class _ArtifactCheck:
    """Published artifact receipts, attempt errors and the delivery record."""

    records: list[dict[str, Any]]
    errors: list[str]
    record: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.record["decision"] == "accepted"


@dataclass(slots=True)
class _Drain:
    """A pause request's drain deadline, in event-loop time.

    ``waits`` holds the timeouts of calls still awaiting their provider;
    an earlier deadline reschedules each one.
    """

    when: float | None = None
    waits: set[asyncio.Timeout] = field(default_factory=set)

    def tighten(self, when: float) -> None:
        if self.when is None or when < self.when:
            self.when = when
            for wait in self.waits:
                if not wait.expired():
                    wait.reschedule(when)


def _drain_bound(drain: _Drain) -> asyncio.Timeout:
    """A timeout for one provider wait that expires at the drain deadline.

    ``drain.when`` is an event-loop time, so a wait that starts after the
    deadline is set must take it as an absolute time, not a delay.
    """
    return asyncio.timeout_at(drain.when)


class JobRunner:
    """Execute only planned calls, with a durable boundary before dispatch.

    ``validators`` registers the trusted local validators that a plan's
    artifact specs name. Each runs, after the built-in artifact check, on
    the exact bytes of every decodable artifact. A plan that names a
    validator this runner has not registered with the same identity is
    refused before a run is created or resumed.
    """

    def __init__(
        self,
        store: SQLiteRunStore,
        *,
        provider_pool: ProviderPool | None = None,
        lease_ttl_s: float = 30.0,
        lease_heartbeat_s: float | None = None,
        call_timeout_s: float | None = None,
        max_wall_seconds: float | None = None,
        validators: Sequence[Any] = (),
    ) -> None:
        self._validators = validator_registry(validators)
        if (
            isinstance(lease_ttl_s, bool)
            or not isinstance(lease_ttl_s, (int, float))
            or not math.isfinite(float(lease_ttl_s))
            or lease_ttl_s <= 0
        ):
            raise ValueError("lease_ttl_s must be finite and positive")
        heartbeat = lease_ttl_s / 3 if lease_heartbeat_s is None else lease_heartbeat_s
        if (
            isinstance(heartbeat, bool)
            or not isinstance(heartbeat, (int, float))
            or not math.isfinite(float(heartbeat))
            or heartbeat <= 0
            or heartbeat >= lease_ttl_s
        ):
            raise ValueError(
                "lease_heartbeat_s must be finite, positive, and less than lease_ttl_s"
            )
        call_timeout = (
            None
            if call_timeout_s is None
            else _positive_finite_timeout(call_timeout_s, "call_timeout_s")
        )
        wall_timeout = (
            None
            if max_wall_seconds is None
            else _positive_finite_timeout(max_wall_seconds, "max_wall_seconds")
        )
        self.store = store
        if provider_pool is not None:
            self.providers = provider_pool
        else:
            self.providers = ProviderPool(
                # The approved plan can raise its per-call deadline above the
                # runner default. Keep the transport ceiling at least as wide
                # as the contract; outer wait_for enforces the exact plan.
                request_timeout_s=MAX_CALL_TIMEOUT_S,
                max_retries=0,
            )
        self.lease_ttl_s = float(lease_ttl_s)
        self.lease_heartbeat_s = float(heartbeat)
        self.call_timeout_s = call_timeout
        self.max_wall_seconds = wall_timeout
        self.last_metrics = JobExecutionMetrics(0, 0, 0.0)

    async def start(
        self,
        plan: JobPlanV1,
        approval: JobApprovalV1,
        *,
        manifest_root: str | Path,
        run_id: str | None = None,
        allow_approval_reuse: bool = False,
    ) -> dict[str, Any]:
        """Create and execute a run; a used approval needs ``allow_approval_reuse``."""
        if type(allow_approval_reuse) is not bool:
            raise TypeError("allow_approval_reuse must be a boolean")
        verify_approval(plan, approval)
        root = Path(manifest_root).resolve()
        if run_id is not None:
            _validate_run_id(run_id)
        self._validate_dispatch_inputs(plan, root)
        created_id = self.store.create_run(
            plan,
            approval,
            manifest_root=root,
            run_id=run_id,
            allow_approval_reuse=allow_approval_reuse,
        )
        _validate_run_id(created_id)
        return await self._execute_with_lease(created_id, plan, root)

    async def resume(
        self, run_id: str, *, clear_pause: bool = True, pause_generation: int | None = None,
    ) -> dict[str, Any]:
        _validate_run_id(run_id)
        if type(clear_pause) is not bool:
            raise TypeError("clear_pause must be a boolean")
        if pause_generation is not None and (
            type(pause_generation) is not int or not 0 <= pause_generation <= MAX_SQLITE_INTEGER
        ):
            raise ValueError("pause_generation must be a nonnegative SQLite integer")
        if not clear_pause and pause_generation is not None:
            raise ValueError("pause_generation requires clear_pause=True")
        # Capture explicit resume intent before preflight or worker launch. A
        # later pause generation must survive the lease-acquisition transaction.
        observed_pause = None
        if clear_pause:
            observed_pause = (self.store.get_control(run_id)["pause_generation"]
                              if pause_generation is None else pause_generation)
        record = self.store.manifest_record(run_id)
        root = Path(record["manifest_root"])
        manifest = JobManifestV1.from_json(record["manifest_json"])
        plan = self._preflight(manifest, root)
        approval = JobApprovalV1.from_dict(record["approval"])
        verify_approval(plan, approval)
        self._validate_dispatch_inputs(plan, root)
        result = await self._execute_with_lease(
            run_id, plan, root, recover=True, pause_generation=observed_pause,
        )
        return await asyncio.to_thread(self._delivery_view, result)

    async def reroll(
        self,
        run_id: str,
        operation_keys: list[str],
        *,
        reason: str,
        acknowledge_unknown: bool = False,
        repair_integrity: bool = False,
    ) -> dict[str, Any]:
        """Run selected operations again under the run's original approval.

        Failed and rejected operations are rerolled; an unknown outcome also
        needs ``acknowledge_unknown``. With ``repair_integrity`` the selected
        operations are succeeded ones instead: under the run's lease, the
        run's artifact directory is claimed, then every accepted file of each
        one's accepted attempt is hashed again just before
        :meth:`SQLiteRunStore.queue_integrity_repair` queues it, which refuses
        verified bytes with ``InvalidTransitionError``.
        """
        _validate_run_id(run_id)
        if type(repair_integrity) is not bool:
            raise TypeError("repair_integrity must be a boolean")
        if repair_integrity and acknowledge_unknown:
            raise ValueError("an integrity repair cannot acknowledge an unknown outcome")
        record = self.store.manifest_record(run_id)
        root = Path(record["manifest_root"])
        manifest = JobManifestV1.from_json(record["manifest_json"])
        plan = self._preflight(manifest, root)
        approval = JobApprovalV1.from_dict(record["approval"])
        verify_approval(plan, approval)
        self._validate_dispatch_inputs(plan, root)
        result = await self._execute_with_lease(
            run_id,
            plan,
            root,
            recover=True,
            reroll=(operation_keys, reason, acknowledge_unknown),
            repair=repair_integrity,
        )
        return await asyncio.to_thread(self._delivery_view, result)

    async def adopt(self, run_id: str, operation_key: str, *, reason: str) -> dict[str, Any]:
        """Accept the bytes an unknown outcome already published, without a provider call.

        The operation's latest attempt directory must hold exactly the files a
        response with the artifact count its profile requests is published
        as, and nothing else. Those bytes then pass the built-in artifact
        check and the plan's registered validators again before
        :meth:`SQLiteRunStore.adopt_published_attempt` records them under the
        run's lease. The call keeps its unknown outcome and its exposure. Any
        other state or bytes refuse adoption with ``InvalidTransitionError``.
        """
        _validate_run_id(run_id)
        record = self.store.manifest_record(run_id)
        root = Path(record["manifest_root"])
        plan = self._preflight(JobManifestV1.from_json(record["manifest_json"]), root)
        verify_approval(plan, JobApprovalV1.from_dict(record["approval"]))
        result = await self._execute_with_lease(run_id, plan, root, adopt=(operation_key, reason))
        return await asyncio.to_thread(self._delivery_view, result)

    async def recheck(
        self, run_id: str, operation_keys: Sequence[str], *, reason: str,
    ) -> dict[str, Any]:
        """Decide succeeded operations' accepted bytes again under this runner's validators.

        Nothing is dispatched and no provider is called. The plan's stored
        approval is not needed, so the validators the plan names may be
        registered here at another version or configuration; each must be
        registered under its name. Under the run's lease, every accepted
        file of each selected operation is hashed again and must be
        ``verified``. Each operation's files are then read once into memory,
        at most 10 files of 32 MiB, and must still hold the recorded bytes;
        the built-in artifact check and the registered validators decide that
        copy. :meth:`SQLiteRunStore.record_reevaluation` records the
        decisions; statuses, accepted attempts and ``accepted`` flags never
        change. Delivery eligibility then uses the latest decision made under
        the current validators. A missing or changed file, another state, an
        unregistered validator or another worker's lease refuses the recheck
        with nothing recorded.
        """
        _validate_run_id(run_id)
        if isinstance(operation_keys, (str, bytes)) or not isinstance(operation_keys, Sequence):
            raise TypeError("operation_keys must be a list of operation keys or IDs")
        keys = list(dict.fromkeys(operation_keys))
        if not keys:
            raise ValueError("at least one operation key is required")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
            raise ValueError("recheck reason must be nonempty text of at most 2048 characters")
        root = Path(self.store.manifest_record(run_id)["manifest_root"])
        result = await self._execute_with_lease(run_id, None, root, recheck=(keys, reason))
        return await asyncio.to_thread(self._delivery_view, result)

    def repair_preview(
        self, run_id: str, operation_keys: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Show what each operation calls for now, before anything is dispatched.

        The Jobs database is opened again read-only, so no lease is taken and
        nothing is written, even while a worker holds the run. Each selected
        operation, or every one, gets an ``action`` (``reuse``, ``recheck``,
        ``regenerate`` or ``review``), its sorted ``reasons`` in the reuse
        decisions' vocabulary (:data:`smythe.reuse.REASONS`), its
        ``attempts_remaining`` and ``new_plan_required``: whether its
        regeneration needs a new plan and approval, because it sends an
        attachment whose bytes changed since the plan, has no attempt left, or
        its run's budget overran. Accepted files are hashed again, and every
        planned attachment is fingerprinted again. The validators are this
        runner's, as for :meth:`recheck`. Nothing is repaired: an integrity
        repair, reroll, resume or recheck carries out an action, and no
        artifact of another run is offered.
        """
        _validate_run_id(run_id)
        if operation_keys is not None and (isinstance(operation_keys, (str, bytes))
                                           or not isinstance(operation_keys, Sequence)):
            raise TypeError("operation_keys must be a list of operation keys or IDs")
        if self.store.read_only:
            return self._preview(self.store, run_id, operation_keys)
        with SQLiteRunStore(self.store.path, read_only=True) as store:
            return self._preview(store, run_id, operation_keys)

    def _preview(
        self, store: SQLiteRunStore, run_id: str, operation_keys: Sequence[str] | None,
    ) -> dict[str, Any]:
        # Imported here, so that Jobs commands without a preview do not load the
        # durable workflow modules the reuse decisions use.
        from smythe.reuse import ACTIONS, REASONS

        snapshot = store.snapshot(run_id)
        if operation_keys is not None:
            selected: list[dict[str, Any]] = []
            for key in operation_keys:
                operation = next((item for item in snapshot["operations"]
                                  if key in (item["operation_key"], item["operation_id"])), None)
                if operation is None:
                    raise JobNotFoundError(f"{run_id}:{key}")
                if operation not in selected:
                    selected.append(operation)
            snapshot["operations"] = selected
        succeeded = {item["accepted_attempt_id"] for item in snapshot["operations"]
                     if item["status"] == OperationStatus.SUCCEEDED.value}
        check_artifact_integrity(snapshot, [row for row in snapshot["artifacts"] if row["accepted"]
                                            and row["attempt_id"] in succeeded], byte_limit=None)
        annotate_delivery(snapshot, store.delivery_records(run_id),
                          store.delivery_reevaluations(run_id), self._registered())
        root = Path(snapshot["manifest_root"])
        attachments = [{"relative_path": item["relative_path"], "sha256": item["sha256"],
                        "status": _attachment_status(item, root)}
                       for item in store.plan_attachments(run_id)]
        changed = {item["relative_path"] for item in attachments if item["status"] == "changed"}
        unavailable = {item["relative_path"] for item in attachments
                       if item["status"] not in ("verified", "changed")}
        cost = snapshot["cost"]
        spent = cost["confirmed_microusd"] + cost["exposure_microusd"]
        overrun = snapshot["status"] == RunStatus.BUDGET_OVERRUN.value or spent > cost["approved_microusd"]
        items = []
        for operation in snapshot["operations"]:
            status, sent = operation["status"], set(operation["spec"].get("attachments") or ())
            reasons = {"source_changed"} if sent & changed else set()
            if sent & unavailable:
                reasons.add("source_unavailable")
            if status == OperationStatus.SUCCEEDED.value:
                reasons.update(_PREVIEW_REASONS.get(code, code)
                               for code in operation["delivery_eligibility"]["reasons"])
            elif status in _STATE_REASONS:
                reasons.add(_STATE_REASONS[status])
            else:
                raise RunStoreError("Jobs run has invalid operation status")
            action = max((REASONS.get(code) or "review" for code in reasons),
                         key=ACTIONS.index, default="reuse")
            action = "regenerate" if action == "rerender" else action
            remaining = max(0, operation["max_attempts"] - operation["attempt_count"])
            items.append({
                "operation_key": operation["operation_key"],
                "operation_id": operation["operation_id"],
                "status": status, "action": action, "reasons": sorted(reasons),
                "attempts_remaining": remaining,
                "new_plan_required": action == "regenerate" and bool(
                    sent & changed or not remaining or overrun),
            })
        counts = dict.fromkeys(_PREVIEW_ACTIONS, 0)
        for item in items:
            counts[item["action"]] += 1
        budget = {key: cost[key] for key in ("approved_microusd", "confirmed_microusd",
                                             "exposure_microusd", "reserved_microusd")}
        return {
            "version": 1, "run_id": run_id, "status": snapshot["status"],
            "budget": budget | {"remaining_microusd": max(0, cost["approved_microusd"] - spent
                                                          - cost["reserved_microusd"]),
                                "overrun": overrun, "unit": cost["unit"]},
            "attachments": attachments, "counts": counts, "operations": items,
        }

    def _registered(self) -> dict[str, ValidatorIdentity]:
        """The identity this runner registered under each validator name."""
        return {name: identity for name, (identity, _validator) in self._validators.items()}

    def _delivery_view(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Recheck each accepted attempt's bytes and derive delivery eligibility.

        A resumed run's earlier outputs are delivered only if their bytes still
        match the recorded SHA-256. A changed, missing or unreadable file is
        reported in ``artifact_integrity`` and makes its operation ineligible;
        recorded statuses and acceptance flags are never changed. Files of an
        attempt that an integrity repair replaced are not rechecked. A
        decision counts only when it was made under this runner's validators.
        """
        attempts = {item["accepted_attempt_id"] for item in snapshot["operations"]}
        accepted = [artifact for artifact in snapshot["artifacts"]
                    if artifact["accepted"] and artifact["attempt_id"] in attempts]
        summary = check_artifact_integrity(snapshot, accepted, byte_limit=None)
        snapshot["artifact_integrity"] = summary | {
            "scope": "accepted artifact bytes after resume; acceptance is unchanged",
        }
        run_id = snapshot["run_id"]
        return annotate_delivery(snapshot, self.store.delivery_records(run_id),
                                 self.store.delivery_reevaluations(run_id), self._registered())

    def _integrity_findings(self, run_id: str, operation_keys: Sequence[str]) -> list[dict[str, Any]]:
        """Hash every accepted file of the selected succeeded operations again.

        Each file of an operation's accepted attempt is read whole, without the
        inspection byte limit. Operations in other states get no findings; the
        store refuses them by their state.
        """
        snapshot = self.store.snapshot(run_id)
        selected = set(operation_keys)
        attempts = {item["accepted_attempt_id"] for item in snapshot["operations"]
                    if item["status"] == OperationStatus.SUCCEEDED.value
                    and selected & {item["operation_key"], item["operation_id"]}}
        artifacts = [artifact for artifact in snapshot["artifacts"]
                     if artifact["accepted"] and artifact["attempt_id"] in attempts]
        check_artifact_integrity(snapshot, artifacts, byte_limit=None)
        return [{"artifact_id": artifact["artifact_id"], "relative_path": artifact["relative_path"],
                 "sha256": artifact["sha256"], "status": artifact["integrity"]["status"],
                 "detail": artifact["integrity"]["detail"][:1024]} for artifact in artifacts]

    def _preflight(self, manifest: JobManifestV1, root: Path) -> JobPlanV1:
        """Rebuild a stored run's plan with this runner's registered validators."""
        validators = [validator for _identity, validator in self._validators.values()]
        return preflight_job(manifest, manifest_root=root, validators=validators)

    def _validate_dispatch_inputs(self, plan: JobPlanV1, root: Path) -> None:
        fingerprints = {item.relative_path: item for item in plan.attachments}
        for fingerprint in fingerprints.values():
            _read_bound_attachment(fingerprint, root)
        for operation in plan.operations:
            for identity in operation.validators:
                registered = self._validators.get(identity.name)
                if registered is None or registered[0] != identity:
                    raise PreflightError(
                        f"operation {operation.operation_key!r} needs validator "
                        f"{identity.name!r} version {identity.version!r}, which this runner "
                        "has not registered with the same identity"
                    )
            self.providers.preflight(operation)
            self.providers.get(operation)

    async def _execute_with_lease(
        self,
        run_id: str,
        plan: JobPlanV1 | None,
        root: Path,
        *,
        recover: bool = False,
        reroll: tuple[list[str], str, bool] | None = None,
        pause_generation: int | None = None,
        adopt: tuple[str, str] | None = None,
        repair: bool = False,
        recheck: tuple[list[str], str] | None = None,
    ) -> dict[str, Any]:
        """Own the run for recovery, state transitions, and provider execution, adoption or a recheck.

        A recheck has no plan: its deadline is this runner's ``max_wall_seconds``
        or the default, and it dispatches nothing to finalize.
        """

        owner_id = f"runner-{uuid4().hex}"
        lease = self.store.acquire_run_lease(
            run_id, owner_id, ttl_s=self.lease_ttl_s, pause_generation=pause_generation,
        )
        execution_task: asyncio.Task[dict[str, Any]] | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        primary_error: BaseException | None = None
        try:
            if reroll is not None and repair:
                operation_keys, reason, _acknowledge_unknown = reroll
                # Execution claims the artifact directory after the repair is
                # queued. Claim it first, so a directory that cannot be claimed
                # refuses the repair with nothing queued.
                await asyncio.to_thread(
                    self._claim_artifact_directory, self.store.snapshot(run_id), root,
                )
                findings = await asyncio.to_thread(self._integrity_findings, run_id, operation_keys)
                self.store.queue_integrity_repair(
                    run_id, operation_keys, reason=reason, findings=findings, lease=lease,
                )
            elif reroll is not None:
                operation_keys, reason, acknowledge_unknown = reroll
                self.store.queue_reroll(
                    run_id,
                    operation_keys,
                    acknowledge_unknown=acknowledge_unknown,
                    reason=reason,
                    lease=lease,
                )
            if recover:
                self.store.recover_inflight(run_id, lease=lease)

            wall_timeout = _plan_timeout(
                plan,
                ("max_wall_seconds", "max_wall_s"),
                operator_ceiling=self.max_wall_seconds,
                fallback=DEFAULT_MAX_WALL_SECONDS,
            )
            if recheck is not None:
                work = self._recheck(run_id, *recheck, lease=lease)
            elif adopt is not None:
                work = self._adopt(run_id, plan, root, *adopt, lease=lease)
            else:
                work = self._execute(run_id, plan, root, lease=lease)
            execution_task = asyncio.create_task(asyncio.wait_for(work, timeout=wall_timeout))
            heartbeat_task = asyncio.create_task(self._heartbeat_lease(lease))
            done, _ = await asyncio.wait(
                {execution_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                # The heartbeat task only completes by failing. Stop execution
                # immediately rather than continuing without exclusive ownership.
                await heartbeat_task
                raise RunLeaseError(f"run {run_id!r} lease heartbeat stopped")
            return await execution_task
        except TimeoutError as exc:
            primary_error = exc
            # ``wait_for`` has already cancelled and awaited the execution
            # task. Every dispatched child therefore had a chance to journal
            # an unknown outcome before the aggregate status is derived.
            if recheck is None:
                self.store.finalize_run(run_id, lease=lease)
            raise TimeoutError(
                f"job run {run_id!r} exceeded its {wall_timeout:g}s wall deadline"
            ) from exc
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            for task in (execution_task, heartbeat_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (execution_task, heartbeat_task) if task is not None),
                return_exceptions=True,
            )
            try:
                self.store.release_run_lease(run_id, owner_id, lease=lease)
            except RunLeaseError:
                if primary_error is None:
                    raise

    async def _heartbeat_lease(self, lease: RunLease) -> None:
        while True:
            await asyncio.sleep(self.lease_heartbeat_s)
            self.store.heartbeat_run_lease(
                lease.run_id,
                lease.owner_id,
                lease=lease,
                ttl_s=self.lease_ttl_s,
            )

    async def _execute(
        self,
        run_id: str,
        plan: JobPlanV1,
        root: Path,
        *,
        lease: RunLease,
    ) -> dict[str, Any]:
        pending = self.store.pending_operations(run_id)
        # Unknown numeric accounting is terminal until explicitly acknowledged
        # through the existing reroll flow. Persisted classification prevents a
        # fresh runner from resuming other paid work with an invalid ledger.
        snapshot = self.store.snapshot(run_id)
        await asyncio.to_thread(self._claim_artifact_directory, snapshot, root)
        if any(
            item["status"] == OperationStatus.UNKNOWN_OUTCOME.value
            and (item["error"] or "").startswith(_INVALID_ACCOUNTING_PREFIX)
            for item in snapshot["operations"]
        ):
            pending = []
        if not pending:
            self.store.finalize_run(run_id, lease=lease)
            snapshot = self.store.snapshot(run_id)
            snapshot["execution_metrics"] = JobExecutionMetrics(0, 0, 0.0).to_dict()
            return snapshot

        operations = {item.operation_id: item for item in plan.operations}
        concurrency = min(plan.max_concurrency, len(pending))
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(1, concurrency * 2))
        active = 0
        peak_active = 0
        started = 0
        counter_lock = asyncio.Lock()
        stop_dispatch = asyncio.Event()
        pause_dispatch = asyncio.Event()
        drain = _Drain()
        start = time.perf_counter()

        async def producer() -> None:
            for record in pending:
                await queue.put(record)
            for _ in range(concurrency):
                await queue.put(None)

        async def worker() -> None:
            nonlocal active, peak_active, started
            while True:
                record = await queue.get()
                try:
                    if record is None:
                        return
                    if stop_dispatch.is_set():
                        continue
                    operation = operations[record["operation_id"]]
                    async with counter_lock:
                        active += 1
                        started += 1
                        peak_active = max(peak_active, active)
                    try:
                        await self._execute_operation(
                            run_id, operation, root, plan, stop_dispatch=stop_dispatch,
                            pause_dispatch=pause_dispatch, lease=lease,
                            artifact_directory=snapshot["artifact_directory"], drain=drain,
                        )
                    finally:
                        async with counter_lock:
                            active -= 1
                finally:
                    queue.task_done()

        async def watch_drain() -> None:
            # Admission cannot see a pause while every worker awaits a provider,
            # so poll for a drain deadline until the workers finish. A failed
            # read, such as of a locked database, keeps any deadline already
            # read and never stops the run; the next poll reads again.
            loop = asyncio.get_running_loop()
            poll_s = min(_DRAIN_POLL_S, self.lease_heartbeat_s)
            while (await asyncio.wait(workers, timeout=poll_s))[1]:
                try:
                    remaining = self.store.drain_time_remaining(run_id)
                except (sqlite3.Error, RunStoreError):
                    continue
                if remaining is not None:
                    drain.tighten(loop.time() + remaining)

        producer_task = asyncio.create_task(producer())
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        tasks = [producer_task, *workers, asyncio.create_task(watch_drain())]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            wall_s = time.perf_counter() - start
            self.last_metrics = JobExecutionMetrics(started, peak_active, wall_s)

        self.store.finalize_run(run_id, lease=lease)
        snapshot = self.store.snapshot(run_id)
        snapshot["execution_metrics"] = self.last_metrics.to_dict()
        return snapshot

    async def _adopt(
        self,
        run_id: str,
        plan: JobPlanV1,
        root: Path,
        operation_key: str,
        reason: str,
        *,
        lease: RunLease,
    ) -> dict[str, Any]:
        snapshot = self.store.snapshot(run_id)
        await asyncio.to_thread(self._claim_artifact_directory, snapshot, root)
        state = next((item for item in snapshot["operations"]
                      if operation_key in (item["operation_key"], item["operation_id"])), None)
        if state is None:
            raise JobNotFoundError(f"{run_id}:{operation_key}")
        if state["status"] != OperationStatus.UNKNOWN_OUTCOME.value:
            raise InvalidTransitionError(
                f"operation {state['operation_key']} in state {state['status']} "
                "has no unknown outcome to adopt"
            )
        attempt = max((item for item in snapshot["attempts"]
                       if item["operation_id"] == state["operation_id"]),
                      key=lambda item: item["attempt_number"])
        operation = next(item for item in plan.operations
                         if item.operation_id == state["operation_id"])
        check = await asyncio.to_thread(
            self._check_published,
            operation,
            attempt_id=attempt["attempt_id"],
            attempt_number=attempt["attempt_number"],
            validators=[self._validators[identity.name] for identity in operation.validators],
            run_root=_validated_run_root(root, output_directory=plan.output_directory, run_id=run_id,
                                         artifact_directory=snapshot["artifact_directory"]),
        )
        # The store checks again, in its transaction, that this is still the
        # operation's latest attempt and that its outcome is unknown.
        self.store.adopt_published_attempt(
            run_id, state["operation_id"], reason=reason, artifacts=check.records,
            delivery_acceptance=check.record, lease=lease,
        )
        self.store.finalize_run(run_id, lease=lease)
        snapshot = self.store.snapshot(run_id)
        snapshot["execution_metrics"] = JobExecutionMetrics(0, 0, 0.0).to_dict()
        return snapshot

    async def _recheck(
        self, run_id: str, operation_keys: list[str], reason: str, *, lease: RunLease,
    ) -> dict[str, Any]:
        snapshot = self.store.snapshot(run_id)
        decisions = await asyncio.to_thread(self._decide_again, snapshot, operation_keys)
        # The store checks again, in its transaction, that each operation still
        # names the attempt and accepted artifacts that were decided.
        self.store.record_reevaluation(run_id, decisions, reason=reason, lease=lease)
        snapshot = self.store.snapshot(run_id)
        snapshot["execution_metrics"] = JobExecutionMetrics(0, 0, 0.0).to_dict()
        return snapshot

    def _decide_again(self, snapshot: dict[str, Any], operation_keys: list[str]) -> list[dict[str, Any]]:
        """Each selected operation's recheck decision; any refusal comes before a validator runs."""
        selected: list[dict[str, Any]] = []
        for key in operation_keys:
            operation = next((item for item in snapshot["operations"]
                              if key in (item["operation_key"], item["operation_id"])), None)
            if operation is None:
                raise JobNotFoundError(f"{snapshot['run_id']}:{key}")
            if operation["status"] != OperationStatus.SUCCEEDED.value:
                raise InvalidTransitionError(
                    f"operation {operation['operation_key']} in state {operation['status']} "
                    "has no accepted output to recheck"
                )
            if operation not in selected:
                selected.append(operation)
        checks = []
        for operation in selected:
            name, spec = operation["operation_key"], operation["spec"]
            validators = []
            for identity in spec.get("validators", []):
                registered = self._validators.get(identity["name"])
                if registered is None:
                    raise PreflightError(
                        f"operation {name!r} needs validator {identity['name']!r}, which this "
                        "runner has not registered; pass it in validators="
                    )
                validators.append(registered)
            rows = [item for item in snapshot["artifacts"]
                    if item["accepted"] and item["attempt_id"] == operation["accepted_attempt_id"]]
            if not rows:
                raise InvalidTransitionError(f"operation {name} has no accepted file to recheck")
            checks.append((operation, ArtifactSpecV1.from_dict(spec["artifact"]), validators, rows))
        check_artifact_integrity(snapshot, [row for *_, rows in checks for row in rows],
                                 byte_limit=None)
        for operation, _spec, _validators, rows in checks:
            for row in rows:
                if row["integrity"]["status"] != "verified":
                    raise InvalidTransitionError(
                        f"operation {operation['operation_key']} cannot be rechecked while its "
                        f"accepted file {row['relative_path']} is {row['integrity']['status']}; "
                        "an integrity repair regenerates it"
                    )
        anchor, parts = _artifact_root(snapshot)
        return [self._decide(operation, spec, validators, rows,
                             [self._accepted_bytes(anchor, parts, row) for row in rows])
                for operation, spec, validators, rows in checks]

    @staticmethod
    def _accepted_bytes(anchor: Path, parts: tuple[str, ...], artifact: dict[str, Any]) -> bytes:
        """An accepted file's bytes, read once, unless they differ from the recorded ones."""
        path = artifact["relative_path"]
        try:
            flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(os.open(_check_path(anchor, (*parts, *_relative_parts(path))), flags),
                           "rb") as stream:
                data = stream.read(artifact["size_bytes"] + 1)
        except (OSError, ValueError) as exc:
            raise InvalidTransitionError(
                f"accepted file {path} could not be read again ({type(exc).__name__})") from exc
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise InvalidTransitionError(f"accepted file {path} changed while it was rechecked")
        return data

    @staticmethod
    def _decide(
        operation: dict[str, Any],
        spec: ArtifactSpecV1,
        validators: Sequence[tuple[ValidatorIdentity, Any]],
        rows: Sequence[dict[str, Any]],
        blobs: Sequence[bytes],
    ) -> dict[str, Any]:
        """The checks of one operation's in-memory accepted bytes, as a recheck records them."""
        builtin = artifact_spec_identity(spec)
        checked: dict[str, list[ValidationReport]] = {identity.name: [] for identity, _ in validators}
        reports: list[ValidationReport] = []
        errors: list[str] = []
        for row, data in zip(rows, blobs):
            path = row["relative_path"]
            filename = PurePosixPath(path).name
            inspection, findings = _artifact_findings(Artifact(data, spec.mime_type), spec, filename)
            reports.append(ValidationReport(path, tuple(findings), validator=builtin))
            errors.extend(finding.message for finding in findings)
            if inspection is None:
                continue
            subject = ValidationSubject(data)
            for identity, validator in validators:
                report, reasons = run_validator(identity, validator, subject, path, filename)
                checked[identity.name].append(report)
                errors.extend(reasons)
        results = [validator_result(builtin, reports)] + [
            validator_result(identity, checked[identity.name]) for identity, _ in validators]
        record = delivery_record(operation["accepted_attempt_id"], [], results, errors)
        return {"operation_id": operation["operation_id"], "attempt_id": record["attempt_id"],
                "outcome": record["decision"], "validators": record["validators"],
                "findings": record["reasons"],
                "artifacts": [{"relative_path": row["relative_path"], "sha256": row["sha256"]}
                              for row in rows]}

    async def _execute_operation(
        self,
        run_id: str,
        operation: PlannedOperationV1,
        root: Path,
        plan: JobPlanV1,
        *,
        stop_dispatch: asyncio.Event,
        pause_dispatch: asyncio.Event,
        lease: RunLease,
        artifact_directory: str,
        drain: _Drain,
    ) -> None:
        try:
            attempt = self.store.begin_attempt(run_id, operation.operation_id, lease=lease)
            ceiling = usd_to_micros(operation.max_cost_per_call_usd)
            permit = self.store.prepare_call(attempt["attempt_id"], ceiling, lease=lease)
        except RunPauseRequested:
            pause_dispatch.set()
            stop_dispatch.set()
            return
        except JobBudgetError:
            # A sibling call can discover and latch an overrun while queued
            # operations are waiting. No new attempt is admitted after that.
            return
        try:
            provider = self.providers.get(operation)
            attachments = (
                await asyncio.to_thread(self._load_attachments, operation, plan, root)
                if operation.attachments
                else []
            )
        except asyncio.CancelledError:
            try:
                self.store.fail_pre_dispatch(
                    permit.call_id,
                    "execution cancelled before provider dispatch",
                    retryable=True,
                    lease=lease,
                )
            except RunLeaseError:
                # Only the current owner may recover this attempt now.
                pass
            raise
        except Exception as exc:
            self.store.fail_pre_dispatch(permit.call_id, str(exc), lease=lease)
            return

        if stop_dispatch.is_set():
            # Attachment loading can yield while a sibling reports bad usage.
            # This call is still safe to release and has consumed no retry.
            self.store.fail_pre_dispatch(
                permit.call_id, ("dispatch stopped for durable pause" if pause_dispatch.is_set()
                                 else "dispatch stopped after invalid provider accounting"),
                retryable=True,
                lease=lease,
            )
            return

        try:
            self.store.mark_call_dispatched(permit.call_id, lease=lease)
        except RunPauseRequested:
            pause_dispatch.set()
            stop_dispatch.set()
            return
        except JobBudgetError as exc:
            self.store.fail_pre_dispatch(permit.call_id, str(exc), lease=lease)
            return

        call_timeout = _plan_timeout(
            plan,
            ("call_timeout_s", "call_timeout_seconds"),
            operator_ceiling=self.call_timeout_s,
            fallback=DEFAULT_CALL_TIMEOUT_S,
        )
        try:
            # A pause's drain deadline bounds only the wait for the provider;
            # a response that has arrived settles normally.
            async with _drain_bound(drain) as bound:
                drain.waits.add(bound)
                try:
                    result = await asyncio.wait_for(
                        provider.chat(
                            "Create exactly the requested artifact. Return no extra variants.",
                            [
                                ChatMessage(
                                    role="user",
                                    content=operation.prompt,
                                    attachments=attachments,
                                )
                            ],
                            operation.model,
                        ),
                        timeout=call_timeout,
                    )
                finally:
                    drain.waits.discard(bound)
        except asyncio.CancelledError:
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id, "execution cancelled after provider dispatch", lease=lease,
                )
            except RunLeaseError:
                pass
            raise
        except TimeoutError:
            self.store.mark_unknown_outcome(
                permit.call_id,
                "provider call was still running at the pause drain deadline" if bound.expired()
                else f"provider call exceeded its {call_timeout:g}s deadline after dispatch",
                lease=lease,
            )
            return
        except BudgetValidationError as exc:
            # Result construction itself can reject malformed provider usage.
            stop_dispatch.set()
            self.store.mark_unknown_outcome(
                permit.call_id, f"{_INVALID_ACCOUNTING_PREFIX}{exc}",
                lease=lease,
            )
            return
        except Exception as exc:
            # Prefix untrusted exception text so it cannot impersonate our
            # persisted invalid-accounting classification on a later resume.
            self.store.mark_unknown_outcome(
                permit.call_id, f"provider failure after dispatch: {exc}",
                lease=lease,
            )
            return

        try:
            # Everything after the dispatch boundary—including untrusted cost
            # metadata, filesystem durability, and the final SQLite
            # transaction—shares one conservative ambiguity boundary. Valid
            # accounting is settled first, so bytes that cannot be decoded or
            # exceed a limit become a deterministic rejection at that cost.
            cost_micros, cost_complete, cost_estimate = self._cost(result, operation)
            check = await asyncio.to_thread(
                self._persist_and_validate,
                result,
                operation,
                attempt_id=attempt["attempt_id"],
                attempt_number=attempt["attempt_number"],
                validators=[self._validators[identity.name] for identity in operation.validators],
                run_root=_validated_run_root(
                    root,
                    output_directory=plan.output_directory,
                    run_id=run_id,
                    artifact_directory=artifact_directory,
                ),
            )
            self.store.complete_call(
                permit.call_id,
                cost_microusd=cost_micros,
                cost_is_complete=cost_complete,
                cost_is_estimate=cost_estimate,
                artifacts=check.records,
                result_text=result.text,
                accepted=check.accepted,
                error="; ".join(check.errors) if check.errors else None,
                delivery_acceptance=check.record,
                lease=lease,
            )
        except asyncio.CancelledError:
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id,
                    "execution cancelled during post-response finalization",
                    lease=lease,
                )
            except RunLeaseError:
                pass
            raise
        except RunLeaseError:
            # A late response cannot settle or reclassify another owner's work.
            raise
        except BudgetValidationError as exc:
            stop_dispatch.set()
            self.store.mark_unknown_outcome(
                permit.call_id, f"{_INVALID_ACCOUNTING_PREFIX}{exc}",
                lease=lease,
            )
            return
        except Exception as exc:
            # The provider returned but trustworthy final accounting did not.
            # Preserve a conservative ambiguity state and never auto-rerun.
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id, f"post-response finalization failure: {exc}", lease=lease,
                )
            except Exception:
                raise exc
            return

    @staticmethod
    def _load_attachments(
        operation: PlannedOperationV1,
        plan: JobPlanV1,
        root: Path,
    ) -> list[Artifact]:
        fingerprints = {item.relative_path: item for item in plan.attachments}
        loaded: list[Artifact] = []
        for relative in operation.attachments:
            fingerprint = fingerprints[relative]
            loaded.append(
                Artifact(
                    data=_read_bound_attachment(fingerprint, root),
                    mime_type=fingerprint.mime_type,
                )
            )
        return loaded

    def _claim_artifact_directory(self, snapshot: dict[str, Any], root: Path) -> None:
        run_root = _validated_run_root(root, output_directory=snapshot["output_directory"],
                                       run_id=snapshot["run_id"],
                                       artifact_directory=snapshot["artifact_directory"])
        legacy = snapshot["artifact_namespace"] is None
        if legacy:
            run_root.mkdir(parents=True, exist_ok=True)
            for candidate in self.store.legacy_artifact_roots():
                if candidate["artifact_owner_id"] == snapshot["artifact_owner_id"]:
                    continue
                other = (Path(candidate["manifest_root"]) / candidate["output_directory"]
                         / candidate["artifact_directory"]).resolve()
                if (_confinement_path(other) == _confinement_path(run_root)
                        or other.exists() and other.samefile(run_root)):
                    raise ValueError("Legacy artifact directory aliases another run; ownership is ambiguous")
        owner_bytes = (json.dumps({"version": 1, "run_id": snapshot["run_id"],
                                   "artifact_owner_id": snapshot["artifact_owner_id"]},
                                  sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        claim_artifact_root(run_root, owner_bytes, legacy=legacy,
                            legacy_artifacts=snapshot["artifacts"] if legacy else [])

    @staticmethod
    def _cost(
        result: CompletionResult,
        operation: PlannedOperationV1,
    ) -> tuple[int, bool, bool]:
        # Custom adapters may mutate a valid result after its constructor ran.
        validate_completion_usage(result)
        if operation.provider is ProviderKind.OFFLINE:
            return 0, True, False
        ceiling = usd_to_micros(operation.max_cost_per_call_usd)
        observed: int | None = None
        if result.cost_usd is not None:
            try:
                value = Decimal(str(result.cost_usd))
            except Exception as exc:
                raise ValueError("provider cost_usd is not a valid decimal") from exc
            if not value.is_finite() or value < 0:
                raise ValueError("provider cost_usd must be finite and non-negative")
            try:
                observed = usd_to_micros(value)
            except ValueError as exc:
                raise BudgetValidationError(
                    f"provider cost_usd cannot be recorded: {exc}"
                ) from exc
            if observed > MAX_SQLITE_INTEGER:
                raise BudgetValidationError("provider cost_usd exceeds the durable integer limit")
        if result.cost_usd_unknown:
            return (
                max(ceiling, observed or 0),
                False,
                True,
            )
        if observed is not None:
            if result.cost_usd_is_estimate:
                return observed, False, True
            return observed, True, False
        return ceiling, False, True

    @staticmethod
    def _persist_and_validate(
        result: CompletionResult,
        operation: PlannedOperationV1,
        *,
        attempt_id: str,
        attempt_number: int,
        run_root: Path,
        validators: Sequence[tuple[ValidatorIdentity, Any]] = (),
        publish: bool = True,
    ) -> _ArtifactCheck:
        """Check a response's artifacts, then publish every decodable one.

        The built-in artifact check reports each finding with its code.
        Undecodable or over-limit bytes are hard findings and are never
        written; rejected bytes that decode are published for review with
        ``accepted=0``. Registered ``validators`` then check each decodable
        artifact's exact bytes; one that fails to decide never makes the call
        an unknown outcome. Artifact data that is not bytes breaks the
        provider contract and raises instead. With ``publish`` false nothing
        is written, as for bytes an attempt already published.
        """
        builtin = artifact_spec_identity(operation.artifact)
        checked: dict[str, list[ValidationReport]] = {
            identity.name: [] for identity, _validator in validators
        }

        def results(builtin_reports: list[ValidationReport]) -> list[dict[str, Any]]:
            return [validator_result(builtin, builtin_reports)] + [
                validator_result(identity, checked[identity.name])
                for identity, _validator in validators
            ]

        artifacts = list(result.artifacts or ())
        if any(not isinstance(artifact.data, bytes) for artifact in artifacts):
            raise TypeError("provider artifact data must be bytes")
        total_bytes = sum(len(artifact.data) for artifact in artifacts)
        refused: ValidationFinding | None = None
        if not artifacts:
            refused = hard_finding("no_artifacts", "provider returned no artifacts")
        elif len(artifacts) > MAX_ARTIFACTS_PER_CALL:
            refused = hard_finding(
                "artifact_count_limit",
                f"provider returned more than {MAX_ARTIFACTS_PER_CALL} artifacts",
                expected=MAX_ARTIFACTS_PER_CALL, observed=len(artifacts),
            )
        elif total_bytes > MAX_TOTAL_ARTIFACT_BYTES_PER_CALL:
            refused = hard_finding(
                "artifact_bytes_limit",
                "provider artifacts exceed the aggregate per-call byte limit",
                expected=MAX_TOTAL_ARTIFACT_BYTES_PER_CALL, observed=total_bytes,
            )
        if refused is not None:
            report = ValidationReport(RESPONSE_PATH, (refused,), validator=builtin)
            return _ArtifactCheck([], [refused.message], delivery_record(
                attempt_id, [], results([report]), [refused.message],
            ))
        safe_key = _SAFE_COMPONENT.sub("_", operation.operation_key).strip("._")
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        subjects: list[dict[str, Any]] = []
        reports: list[ValidationReport] = []
        pending_writes: list[tuple[Path, bytes]] = []
        for index, artifact in enumerate(artifacts):
            filename = _artifact_filename(operation, artifact, index)
            relative = Path("artifacts") / safe_key / f"attempt-{attempt_number:04d}" / filename
            inspection, findings = _artifact_findings(artifact, operation.artifact, filename)
            reports.append(ValidationReport(relative.as_posix(), tuple(findings), validator=builtin))
            errors.extend(finding.message for finding in findings)
            subjects.append({
                "path": relative.as_posix(),
                "sha256": (inspection.sha256 if inspection is not None
                           else hashlib.sha256(artifact.data).hexdigest()),
                "size_bytes": len(artifact.data),
                "published": inspection is not None,
            })
            if inspection is None:
                continue
            if validators:
                subject = ValidationSubject(artifact.data)
                for identity, validator in validators:
                    report, reasons = run_validator(
                        identity, validator, subject, relative.as_posix(), filename,
                    )
                    checked[identity.name].append(report)
                    errors.extend(reasons)
            destination = run_root / relative
            if publish:
                _prepare_confined_parent(destination, run_root)
            records.append(
                {
                    "artifact_id": uuid4().hex,
                    "relative_path": relative.as_posix(),
                    "mime_type": inspection.mime_type,
                    "sha256": inspection.sha256,
                    "size_bytes": inspection.size_bytes,
                    "width": inspection.width,
                    "height": inspection.height,
                }
            )
            pending_writes.append((destination, artifact.data))
        for destination, data in pending_writes if publish else ():
            atomic_publish_bytes(destination, data)
        return _ArtifactCheck(records, errors,
                              delivery_record(attempt_id, subjects, results(reports), errors))

    @staticmethod
    def _check_published(
        operation: PlannedOperationV1,
        *,
        attempt_id: str,
        attempt_number: int,
        run_root: Path,
        validators: Sequence[tuple[ValidatorIdentity, Any]] = (),
    ) -> _ArtifactCheck:
        """Check the bytes an attempt published as the response they came from.

        The attempt's directory must hold exactly the files a response with
        the artifact count the operation's profile requests is published as,
        and nothing else, and the response's checks must accept them.
        Symbolic links and junctions are refused, never followed.
        """
        def refuse(problems: list[str]) -> NoReturn:
            raise InvalidTransitionError(
                f"attempt {attempt_number} of operation {operation.operation_key} cannot be "
                "adopted: " + "; ".join(problems)
            )

        count = _requested_artifacts(operation)
        if type(count) is not int or not 1 <= count <= MAX_ARTIFACTS_PER_CALL:
            refuse([f"its profile requests {count!r} artifacts per call"])
        directory = ("artifacts", _SAFE_COMPONENT.sub("_", operation.operation_key).strip("._"),
                     f"attempt-{attempt_number:04d}")
        sample = Artifact(b"", operation.artifact.mime_type)
        expected = [_artifact_filename(operation, sample, index) for index in range(count)]
        try:
            found = os.listdir(_check_path(run_root, directory))
        except FileNotFoundError:
            found = []
        problems = [f"{name} is missing" for name in expected if name not in found]
        problems += [f"{name} is not part of the response" for name in sorted(found)
                     if name not in expected]
        if problems:
            refuse(problems)
        artifacts = []
        for name in expected:
            path = _check_path(run_root, (*directory, name))
            if not stat.S_ISREG(path.lstat().st_mode):
                refuse([f"{name} is not a regular file"])
            with path.open("rb") as stream:
                artifacts.append(Artifact(stream.read(MAX_ARTIFACT_BYTES + 1), sample.mime_type))
        check = JobRunner._persist_and_validate(
            CompletionResult(text="", artifacts=artifacts), operation, attempt_id=attempt_id,
            attempt_number=attempt_number, run_root=run_root, validators=validators, publish=False,
        )
        if not check.accepted:
            refuse(check.errors or [f"the decision is {check.record['decision']}"])
        return check


def _positive_finite_timeout(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _plan_timeout(
    plan: JobPlanV1,
    field_names: tuple[str, ...],
    *,
    operator_ceiling: float | None,
    fallback: float,
) -> float:
    """Honor the approved deadline, optionally tightened by the operator."""

    for field_name in field_names:
        value = getattr(plan, field_name, None)
        if value is not None:
            approved = _positive_finite_timeout(value, f"plan.{field_name}")
            break
    else:
        approved = fallback
    if operator_ceiling is None:
        return approved
    return min(approved, operator_ceiling)


def _artifact_filename(
    operation: PlannedOperationV1,
    artifact: Artifact,
    index: int,
) -> str:
    requested = operation.artifact.filename
    if requested and index == 0:
        return requested
    stem = Path(requested).stem if requested else "artifact"
    suffix = artifact.suffix
    return f"{stem}-{index + 1:02d}{suffix}"


def _requested_artifacts(operation: PlannedOperationV1) -> object:
    """The artifact count an operation's profile requests from its provider."""
    option = {ProviderKind.OFFLINE: "artifacts_per_call",
              ProviderKind.OPENAI_IMAGE: "n"}.get(operation.provider)
    return operation.options.get(option, 1) if option else 1


def _artifact_findings(
    artifact: Artifact,
    spec: ArtifactSpecV1,
    filename: str,
) -> tuple[ArtifactInspection | None, list[ValidationFinding]]:
    """The built-in artifact check: bounded decoding, then the spec's MIME and size.

    Inspection is None when the bytes cannot be decoded within the limits.
    Finding messages keep the attempt error text of earlier releases.
    """
    if len(artifact.data) > MAX_ARTIFACT_BYTES:
        return None, [hard_finding(
            "artifact_too_large",
            f"{filename}: provider artifact exceeds the {MAX_ARTIFACT_BYTES}-byte limit",
            expected=MAX_ARTIFACT_BYTES, observed=len(artifact.data),
        )]
    try:
        inspection = inspect_artifact(artifact.data, artifact.mime_type)
    except ArtifactInspectionError as exc:
        return None, [hard_finding("artifact_undecodable", f"{filename}: {exc}")]
    findings = []
    if inspection.mime_type != spec.mime_type:
        findings.append(hard_finding(
            "mime_type_mismatch",
            f"{filename}: expected {spec.mime_type}, observed {inspection.mime_type}",
            expected=spec.mime_type, observed=inspection.mime_type,
        ))
    if spec.width is not None and (
        inspection.width != spec.width or inspection.height != spec.height
    ):
        findings.append(hard_finding(
            "dimensions_mismatch",
            f"{filename}: expected {spec.width}x{spec.height}, observed "
            f"{inspection.width}x{inspection.height}",
            expected=[spec.width, spec.height], observed=[inspection.width, inspection.height],
        ))
    return inspection, findings


def _validate_run_id(run_id: str) -> str:
    """Validate the identifier syntax; persistent ownership handles aliases."""

    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-128 characters using letters, numbers, '.', '_', "
            "or '-', and must start with a letter or number"
        )
    return run_id


def _validated_run_root(
    manifest_root: Path,
    *,
    output_directory: str,
    run_id: str,
    artifact_directory: str | None = None,
) -> Path:
    """Resolve the run directory and re-check preflight's confinement at dispatch."""

    _validate_run_id(run_id)
    resolved_manifest_root = manifest_root.resolve()
    output_root = (resolved_manifest_root / output_directory).resolve()
    try:
        _confinement_path(output_root).relative_to(
            _confinement_path(resolved_manifest_root)
        )
    except ValueError as exc:
        raise ValueError("job output directory escapes the manifest root") from exc
    directory = run_id if artifact_directory is None else _validate_run_id(artifact_directory)
    run_root = (output_root / directory).resolve()
    try:
        _confinement_path(run_root).relative_to(_confinement_path(output_root))
    except ValueError as exc:  # pragma: no cover - run_id validation is defense in depth
        raise ValueError("job run directory escapes the approved output directory") from exc
    return run_root


def _prepare_confined_parent(destination: Path, run_root: Path) -> None:
    """Create an artifact parent and reject symlink redirection before writing."""

    resolved_run_root = run_root.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination.parent.resolve()
    try:
        _confinement_path(resolved_parent).relative_to(
            _confinement_path(resolved_run_root)
        )
    except ValueError as exc:
        raise ValueError("artifact destination escapes the approved run directory") from exc


def _read_bound_attachment(fingerprint: Any, root: Path) -> bytes:
    """Read an attachment only if it still matches the approved plan."""

    resolved_root = root.resolve()
    path = (resolved_root / fingerprint.relative_path).resolve()
    try:
        _confinement_path(path).relative_to(_confinement_path(resolved_root))
    except ValueError as exc:
        raise ValueError(
            f"attachment {fingerprint.relative_path!r} escapes the manifest root"
        ) from exc
    if not path.is_file():
        raise ValueError(f"attachment {fingerprint.relative_path!r} no longer exists")
    size = path.stat().st_size
    if size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachment {fingerprint.relative_path!r} exceeds " f"{MAX_ATTACHMENT_BYTES} bytes"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if size != fingerprint.size_bytes or not hmac.compare_digest(digest, fingerprint.sha256):
        raise ValueError(
            f"attachment {fingerprint.relative_path!r} changed after preflight; "
            "create a new plan and approval"
        )
    return data


def _attachment_status(fingerprint: dict[str, Any], root: Path) -> str:
    """Whether a planned attachment still holds the bytes its plan fingerprinted.

    ``verified`` or ``changed``, as dispatch would read it, else ``missing``,
    ``unsafe`` (outside the manifest root, or not a file) or ``unreadable``.
    """
    try:
        resolved_root = root.resolve()
        path = (resolved_root / fingerprint["relative_path"]).resolve()
        _confinement_path(path).relative_to(_confinement_path(resolved_root))
        if not path.is_file():
            return "unsafe" if path.exists() else "missing"
        with path.open("rb") as stream:
            data = stream.read(MAX_ATTACHMENT_BYTES + 1)
    except ValueError:
        return "unsafe"
    except OSError:
        return "unreadable"
    return "verified" if hashlib.sha256(data).hexdigest() == fingerprint["sha256"] else "changed"


def _confinement_path(path: Path) -> Path:
    """Normalize equivalent Windows namespace spellings for containment checks."""

    resolved = path.resolve()
    if os.name != "nt":
        return resolved
    value = str(resolved)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)
