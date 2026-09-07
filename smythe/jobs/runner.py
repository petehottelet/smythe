"""Bounded execution for preflighted artifact jobs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from smythe.jobs.artifact_io import (
    MAX_ARTIFACT_BYTES,
    atomic_write_bytes,
    inspect_artifact,
)
from smythe.jobs.models import (
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
    preflight_job,
    verify_approval,
)
from smythe.jobs.providers import ProviderPool
from smythe.jobs.store import (
    MAX_ARTIFACTS_PER_CALL,
    MAX_SQLITE_INTEGER,
    JobBudgetError,
    RunLeaseError,
    SQLiteRunStore,
)
from smythe.provider import Artifact, CompletionResult
from smythe.tools import ChatMessage


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MAX_TOTAL_ARTIFACT_BYTES_PER_CALL = 128 * 1024 * 1024


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


class JobRunner:
    """Execute only planned calls, with a durable boundary before dispatch."""

    def __init__(
        self,
        store: SQLiteRunStore,
        *,
        provider_pool: ProviderPool | None = None,
        lease_ttl_s: float = 30.0,
        lease_heartbeat_s: float | None = None,
        call_timeout_s: float | None = None,
        max_wall_seconds: float | None = None,
    ) -> None:
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
    ) -> dict[str, Any]:
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
        )
        _validate_run_id(created_id)
        return await self._execute_with_lease(created_id, plan, root)

    async def resume(self, run_id: str) -> dict[str, Any]:
        _validate_run_id(run_id)
        record = self.store.manifest_record(run_id)
        root = Path(record["manifest_root"])
        manifest = JobManifestV1.from_json(record["manifest_json"])
        plan = preflight_job(manifest, manifest_root=root)
        approval = JobApprovalV1.from_dict(record["approval"])
        verify_approval(plan, approval)
        self._validate_dispatch_inputs(plan, root)
        return await self._execute_with_lease(run_id, plan, root, recover=True)

    async def reroll(
        self,
        run_id: str,
        operation_keys: list[str],
        *,
        reason: str,
        acknowledge_unknown: bool = False,
    ) -> dict[str, Any]:
        _validate_run_id(run_id)
        record = self.store.manifest_record(run_id)
        root = Path(record["manifest_root"])
        manifest = JobManifestV1.from_json(record["manifest_json"])
        plan = preflight_job(manifest, manifest_root=root)
        approval = JobApprovalV1.from_dict(record["approval"])
        verify_approval(plan, approval)
        self._validate_dispatch_inputs(plan, root)
        return await self._execute_with_lease(
            run_id,
            plan,
            root,
            recover=True,
            reroll=(operation_keys, reason, acknowledge_unknown),
        )

    def _validate_dispatch_inputs(self, plan: JobPlanV1, root: Path) -> None:
        fingerprints = {item.relative_path: item for item in plan.attachments}
        for fingerprint in fingerprints.values():
            _read_bound_attachment(fingerprint, root)
        for operation in plan.operations:
            self.providers.preflight(operation)
            self.providers.get(operation)

    async def _execute_with_lease(
        self,
        run_id: str,
        plan: JobPlanV1,
        root: Path,
        *,
        recover: bool = False,
        reroll: tuple[list[str], str, bool] | None = None,
    ) -> dict[str, Any]:
        """Own the run for recovery, state transitions, and provider execution."""

        owner_id = f"runner-{uuid4().hex}"
        self.store.acquire_run_lease(run_id, owner_id, ttl_s=self.lease_ttl_s)
        execution_task: asyncio.Task[dict[str, Any]] | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        primary_error: BaseException | None = None
        try:
            if reroll is not None:
                operation_keys, reason, acknowledge_unknown = reroll
                self.store.queue_reroll(
                    run_id,
                    operation_keys,
                    acknowledge_unknown=acknowledge_unknown,
                    reason=reason,
                    lease_owner_id=owner_id,
                )
            if recover:
                self.store.recover_inflight(run_id, lease_owner_id=owner_id)

            wall_timeout = _plan_timeout(
                plan,
                ("max_wall_seconds", "max_wall_s"),
                operator_ceiling=self.max_wall_seconds,
                fallback=DEFAULT_MAX_WALL_SECONDS,
            )
            execution_task = asyncio.create_task(
                asyncio.wait_for(
                    self._execute(run_id, plan, root),
                    timeout=wall_timeout,
                )
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_lease(run_id, owner_id))
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
            self.store.finalize_run(run_id)
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
                self.store.release_run_lease(run_id, owner_id)
            except RunLeaseError:
                if primary_error is None:
                    raise

    async def _heartbeat_lease(self, run_id: str, owner_id: str) -> None:
        while True:
            await asyncio.sleep(self.lease_heartbeat_s)
            self.store.heartbeat_run_lease(
                run_id,
                owner_id,
                ttl_s=self.lease_ttl_s,
            )

    async def _execute(
        self,
        run_id: str,
        plan: JobPlanV1,
        root: Path,
    ) -> dict[str, Any]:
        pending = self.store.pending_operations(run_id)
        if not pending:
            self.store.finalize_run(run_id)
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
                    operation = operations[record["operation_id"]]
                    async with counter_lock:
                        active += 1
                        started += 1
                        peak_active = max(peak_active, active)
                    try:
                        await self._execute_operation(run_id, operation, root, plan)
                    finally:
                        async with counter_lock:
                            active -= 1
                finally:
                    queue.task_done()

        producer_task = asyncio.create_task(producer())
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        try:
            await asyncio.gather(producer_task, *workers)
        except BaseException:
            producer_task.cancel()
            for task in workers:
                task.cancel()
            await asyncio.gather(producer_task, *workers, return_exceptions=True)
            raise
        finally:
            wall_s = time.perf_counter() - start
            self.last_metrics = JobExecutionMetrics(started, peak_active, wall_s)

        self.store.finalize_run(run_id)
        snapshot = self.store.snapshot(run_id)
        snapshot["execution_metrics"] = self.last_metrics.to_dict()
        return snapshot

    async def _execute_operation(
        self,
        run_id: str,
        operation: PlannedOperationV1,
        root: Path,
        plan: JobPlanV1,
    ) -> None:
        try:
            attempt = self.store.begin_attempt(run_id, operation.operation_id)
            ceiling = usd_to_micros(operation.max_cost_per_call_usd)
            permit = self.store.prepare_call(attempt["attempt_id"], ceiling)
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
            self.store.fail_pre_dispatch(
                permit.call_id,
                "execution cancelled before provider dispatch",
                retryable=True,
            )
            raise
        except Exception as exc:
            self.store.fail_pre_dispatch(permit.call_id, str(exc))
            return

        try:
            self.store.mark_call_dispatched(permit.call_id)
        except JobBudgetError as exc:
            self.store.fail_pre_dispatch(permit.call_id, str(exc))
            return

        call_timeout = _plan_timeout(
            plan,
            ("call_timeout_s", "call_timeout_seconds"),
            operator_ceiling=self.call_timeout_s,
            fallback=DEFAULT_CALL_TIMEOUT_S,
        )
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
        except asyncio.CancelledError:
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id, "execution cancelled after provider dispatch"
                )
            finally:
                raise
        except TimeoutError:
            self.store.mark_unknown_outcome(
                permit.call_id,
                f"provider call exceeded its {call_timeout:g}s deadline after dispatch",
            )
            return
        except Exception as exc:
            self.store.mark_unknown_outcome(permit.call_id, str(exc))
            return

        try:
            # Everything after the dispatch boundary—including untrusted cost
            # metadata, image decoding, filesystem durability, and the final
            # SQLite transaction—shares one conservative ambiguity boundary.
            cost_micros, cost_complete, cost_estimate = self._cost(result, operation)
            artifact_records, errors = await asyncio.to_thread(
                self._persist_and_validate,
                result,
                operation,
                attempt_number=attempt["attempt_number"],
                run_root=_validated_run_root(
                    root,
                    output_directory=plan.output_directory,
                    run_id=run_id,
                ),
            )
            accepted = bool(artifact_records) and not errors
            self.store.complete_call(
                permit.call_id,
                cost_microusd=cost_micros,
                cost_is_complete=cost_complete,
                cost_is_estimate=cost_estimate,
                artifacts=artifact_records,
                result_text=result.text,
                accepted=accepted,
                error="; ".join(errors) if errors else None,
            )
        except asyncio.CancelledError:
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id,
                    "execution cancelled during post-response finalization",
                )
            finally:
                raise
        except Exception as exc:
            # The provider returned but trustworthy final accounting did not.
            # Preserve a conservative ambiguity state and never auto-rerun.
            try:
                self.store.mark_unknown_outcome(
                    permit.call_id, f"post-response finalization failure: {exc}"
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

    @staticmethod
    def _cost(
        result: CompletionResult,
        operation: PlannedOperationV1,
    ) -> tuple[int, bool, bool]:
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
            observed = usd_to_micros(value)
            if observed > MAX_SQLITE_INTEGER:
                raise ValueError("provider cost_usd exceeds the durable integer limit")
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
        attempt_number: int,
        run_root: Path,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not result.artifacts:
            return [], ["provider returned no artifacts"]
        if len(result.artifacts) > MAX_ARTIFACTS_PER_CALL:
            raise ValueError(f"provider returned more than {MAX_ARTIFACTS_PER_CALL} artifacts")
        safe_key = _SAFE_COMPONENT.sub("_", operation.operation_key).strip("._")
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        pending_writes: list[tuple[Path, bytes]] = []
        total_bytes = 0
        for index, artifact in enumerate(result.artifacts):
            if not isinstance(artifact.data, bytes):
                raise TypeError("provider artifact data must be bytes")
            if len(artifact.data) > MAX_ARTIFACT_BYTES:
                raise ValueError(f"provider artifact exceeds the {MAX_ARTIFACT_BYTES}-byte limit")
            total_bytes += len(artifact.data)
            if total_bytes > MAX_TOTAL_ARTIFACT_BYTES_PER_CALL:
                raise ValueError("provider artifacts exceed the aggregate per-call byte limit")
            inspection = inspect_artifact(artifact.data, artifact.mime_type)
            filename = _artifact_filename(operation, artifact, index)
            relative = Path("artifacts") / safe_key / f"attempt-{attempt_number:04d}" / filename
            destination = run_root / relative
            _prepare_confined_parent(destination, run_root)
            if inspection.mime_type != operation.artifact.mime_type:
                errors.append(
                    f"{filename}: expected {operation.artifact.mime_type}, "
                    f"observed {inspection.mime_type}"
                )
            if operation.artifact.width is not None and (
                inspection.width != operation.artifact.width
                or inspection.height != operation.artifact.height
            ):
                errors.append(
                    f"{filename}: expected {operation.artifact.width}x"
                    f"{operation.artifact.height}, observed "
                    f"{inspection.width}x{inspection.height}"
                )
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
        for destination, data in pending_writes:
            atomic_write_bytes(destination, data)
        return records, errors


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


def _validate_run_id(run_id: str) -> str:
    """Reject identifiers that could escape or alias an artifact directory."""

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
    run_root = (output_root / run_id).resolve()
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
