"""Deterministic expansion, cost preflight, and approval binding for jobs."""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from smythe.jobs.models import (
    ArtifactSpecV1,
    FrozenJSONValue,
    JobManifestV1,
    JSONValue,
    MANIFEST_VERSION,
    MAX_ATTACHMENT_BYTES,
    MAX_CANONICAL_PLAN_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_OPTIONS_JSON_BYTES,
    MAX_SIGNED_MICRO_USD,
    MAX_TOTAL_ATTACHMENT_BYTES,
    MAX_UNIQUE_ATTACHMENTS,
    ManifestValidationError,
    ProviderKind,
    detach_json,
    freeze_json,
    micros_to_usd,
    normalize_usd,
    usd_string,
    usd_to_micros,
)


class PreflightError(ValueError):
    """Raised before execution when a job cannot be bounded safely."""


class ApprovalError(ValueError):
    """Raised when an approval does not match the exact preflight plan."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value deterministically for hashes and signatures."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class AttachmentFingerprintV1:
    relative_path: str
    sha256: str
    size_bytes: int
    mime_type: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }


@dataclass(frozen=True, slots=True)
class PlannedOperationV1:
    operation_id: str
    template_key: str
    ordinal: int
    prompt: str
    profile_name: str
    provider: ProviderKind
    model: str
    options: Mapping[str, FrozenJSONValue | JSONValue]
    attachments: tuple[str, ...]
    artifact: ArtifactSpecV1
    max_attempts: int
    max_cost_per_call_usd: Decimal
    worst_case_cost_usd: Decimal

    def __post_init__(self) -> None:
        frozen_options = freeze_json(
            self.options,
            context=f"planned operation {self.operation_id}.options",
        )
        if not isinstance(frozen_options, Mapping):
            raise ManifestValidationError("planned operation options must be an object")
        detached_options = detach_json(
            frozen_options,
            context=f"planned operation {self.operation_id}.options",
        )
        if len(canonical_json_bytes(detached_options)) > MAX_OPTIONS_JSON_BYTES:
            raise ManifestValidationError(
                "planned operation options exceed " f"{MAX_OPTIONS_JSON_BYTES} UTF-8 JSON bytes"
            )
        object.__setattr__(self, "options", frozen_options)
        object.__setattr__(self, "attachments", tuple(self.attachments))

    @property
    def operation_key(self) -> str:
        """Unique human-readable key for logs and CLI selection."""
        return f"{self.template_key}[{self.ordinal}]"

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "operation_id": self.operation_id,
            "template_key": self.template_key,
            "ordinal": self.ordinal,
            "prompt": self.prompt,
            "profile_name": self.profile_name,
            "provider": self.provider.value,
            "model": self.model,
            "options": detach_json(
                self.options,
                context=f"planned operation {self.operation_id}.options",
            ),
            "attachments": list(self.attachments),
            "artifact": self.artifact.to_dict(),
            "max_attempts": self.max_attempts,
            "max_cost_per_call_usd": usd_string(self.max_cost_per_call_usd),
            "worst_case_cost_usd": usd_string(self.worst_case_cost_usd),
        }


@dataclass(frozen=True, slots=True)
class JobPlanV1:
    version: int
    name: str
    manifest_hash: str
    plan_hash: str
    manifest_json: str
    operations: tuple[PlannedOperationV1, ...]
    attachments: tuple[AttachmentFingerprintV1, ...]
    max_concurrency: int
    max_attempts: int
    call_timeout_s: float
    max_wall_seconds: float
    max_budget_usd: Decimal
    worst_case_cost_usd: Decimal
    output_directory: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "attachments", tuple(self.attachments))

    @property
    def max_budget_microusd(self) -> int:
        return usd_to_micros(self.max_budget_usd)

    @property
    def worst_case_cost_microusd(self) -> int:
        return usd_to_micros(self.worst_case_cost_usd)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "name": self.name,
            "manifest_hash": self.manifest_hash,
            "plan_hash": self.plan_hash,
            "operations": [operation.to_dict() for operation in self.operations],
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "max_concurrency": self.max_concurrency,
            "max_attempts": self.max_attempts,
            "call_timeout_s": self.call_timeout_s,
            "max_wall_seconds": self.max_wall_seconds,
            "max_budget_usd": usd_string(self.max_budget_usd),
            "worst_case_cost_usd": usd_string(self.worst_case_cost_usd),
            "output_directory": self.output_directory,
        }


def _plan_identity_payload(plan: JobPlanV1) -> dict[str, JSONValue]:
    payload = plan.to_dict()
    payload.pop("plan_hash")
    return payload


def _verify_plan_identity(plan: JobPlanV1) -> None:
    """Recompute both cached identities from the plan's present contents."""
    if not isinstance(plan.manifest_json, str):
        raise ApprovalError("plan manifest identity is invalid")
    try:
        manifest_payload = json.loads(plan.manifest_json)
        canonical_manifest = canonical_json_bytes(manifest_payload)
    except (
        json.JSONDecodeError,
        ManifestValidationError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise ApprovalError("plan manifest identity is invalid") from exc
    if canonical_manifest.decode("utf-8") != plan.manifest_json:
        raise ApprovalError("plan manifest JSON is not canonical")
    current_manifest_hash = "sha256:" + hashlib.sha256(canonical_manifest).hexdigest()
    if not isinstance(plan.manifest_hash, str) or not hmac.compare_digest(
        plan.manifest_hash, current_manifest_hash
    ):
        raise ApprovalError("plan manifest identity has drifted")
    try:
        current_plan_hash = _sha256_json(_plan_identity_payload(plan))
    except (
        ArithmeticError,
        AttributeError,
        ManifestValidationError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ApprovalError("plan identity is invalid") from exc
    if not isinstance(plan.plan_hash, str) or not hmac.compare_digest(
        plan.plan_hash, current_plan_hash
    ):
        raise ApprovalError("plan identity has drifted")


@dataclass(frozen=True, slots=True)
class JobApprovalV1:
    version: int
    manifest_hash: str
    plan_hash: str
    approved_max_cost_usd: Decimal
    token: str

    @property
    def approved_max_cost_microusd(self) -> int:
        return usd_to_micros(self.approved_max_cost_usd)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "manifest_hash": self.manifest_hash,
            "plan_hash": self.plan_hash,
            "approved_max_cost_usd": usd_string(self.approved_max_cost_usd),
            "token": self.token,
        }

    @classmethod
    def from_dict(cls, data: object) -> "JobApprovalV1":
        if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
            raise ApprovalError("approval must be an object")
        allowed = {
            "version",
            "manifest_hash",
            "plan_hash",
            "approved_max_cost_usd",
            "token",
        }
        unknown = set(data) - allowed
        missing = allowed - set(data)
        if unknown:
            raise ApprovalError(f"approval has unknown fields: {sorted(unknown)}")
        if missing:
            raise ApprovalError(f"approval is missing fields: {sorted(missing)}")
        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ApprovalError("approval.version must be an integer")
        for field_name in ("manifest_hash", "plan_hash", "token"):
            if not isinstance(data[field_name], str) or not data[field_name]:
                raise ApprovalError(f"approval.{field_name} must be a non-empty string")
        try:
            approved = normalize_usd(
                data["approved_max_cost_usd"],
                field_name="approval.approved_max_cost_usd",
            )
        except ManifestValidationError as exc:
            raise ApprovalError(str(exc)) from exc
        return cls(
            version=version,
            manifest_hash=data["manifest_hash"],
            plan_hash=data["plan_hash"],
            approved_max_cost_usd=approved,
            token=data["token"],
        )


def _safe_relative_path(path_text: str, root: Path, *, context: str) -> tuple[Path, str]:
    path = Path(path_text)
    if path.is_absolute():
        raise PreflightError(f"{context} must be relative to the manifest root")
    root = root.resolve()
    resolved = (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"{context} escapes the manifest root") from exc
    return resolved, relative.as_posix()


@dataclass(frozen=True, slots=True)
class _AttachmentCandidate:
    resolved: Path
    relative_path: str
    size_bytes: int
    device: int
    inode: int
    modified_ns: int
    changed_ns: int


def _attachment_candidate(path_text: str, root: Path) -> _AttachmentCandidate:
    resolved, relative = _safe_relative_path(path_text, root, context=f"attachment {path_text!r}")
    try:
        snapshot = resolved.stat()
    except OSError as exc:
        raise PreflightError(f"cannot stat attachment {relative}: {exc}") from exc
    if not stat.S_ISREG(snapshot.st_mode):
        raise PreflightError(f"attachment does not exist or is not a file: {relative}")
    if snapshot.st_size > MAX_ATTACHMENT_BYTES:
        limit_mib = MAX_ATTACHMENT_BYTES // (1024 * 1024)
        raise PreflightError(f"attachment exceeds {limit_mib} MiB: {relative}")
    return _AttachmentCandidate(
        resolved=resolved,
        relative_path=relative,
        size_bytes=snapshot.st_size,
        device=snapshot.st_dev,
        inode=snapshot.st_ino,
        modified_ns=snapshot.st_mtime_ns,
        changed_ns=snapshot.st_ctime_ns,
    )


def _matches_candidate(
    snapshot: os.stat_result,
    candidate: _AttachmentCandidate,
) -> bool:
    # st_ctime_ns is compared only on POSIX.  The candidate is captured
    # with a path stat and re-checked with a handle stat; on Windows
    # those two sources disagree on ctime for a recently modified file
    # (NTFS updates the directory entry lazily, so the path stat can lag
    # the file record by ~1ms) and an unmodified attachment would be
    # rejected as changed.  Size, device, inode, and mtime are the
    # portable signals that a file was substituted or rewritten.
    if os.name != "nt" and snapshot.st_ctime_ns != candidate.changed_ns:
        return False
    return (
        snapshot.st_size == candidate.size_bytes
        and snapshot.st_dev == candidate.device
        and snapshot.st_ino == candidate.inode
        and snapshot.st_mtime_ns == candidate.modified_ns
    )


def _fingerprint(candidate: _AttachmentCandidate) -> AttachmentFingerprintV1:
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with candidate.resolved.open("rb") as stream:
            if not _matches_candidate(os.fstat(stream.fileno()), candidate):
                raise PreflightError(
                    f"attachment changed after size validation: " f"{candidate.relative_path}"
                )
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > candidate.size_bytes or bytes_read > MAX_ATTACHMENT_BYTES:
                    raise PreflightError(
                        f"attachment grew while hashing: {candidate.relative_path}"
                    )
                digest.update(chunk)
            if bytes_read != candidate.size_bytes or not _matches_candidate(
                os.fstat(stream.fileno()), candidate
            ):
                raise PreflightError(f"attachment changed while hashing: {candidate.relative_path}")
    except PreflightError:
        raise
    except OSError as exc:
        raise PreflightError(f"cannot read attachment {candidate.relative_path}: {exc}") from exc

    mime_type = mimetypes.guess_type(candidate.relative_path)[0] or "application/octet-stream"
    return AttachmentFingerprintV1(
        relative_path=candidate.relative_path,
        sha256=digest.hexdigest(),
        size_bytes=bytes_read,
        mime_type=mime_type,
    )


def _operation_identity_payload(
    *,
    template: Any,
    ordinal: int,
    profile: Any,
    attachments: tuple[str, ...] | None = None,
) -> dict[str, JSONValue]:
    # The profile is part of the operation semantics. Changing a model or its
    # options yields a new ID; adding an unrelated template does not.
    template_payload = template.to_dict()
    if attachments is not None:
        template_payload["attachments"] = list(attachments)
    return {
        "version": MANIFEST_VERSION,
        "template": template_payload,
        "ordinal": ordinal,
        "profile": profile.to_dict(),
    }


def operation_id(
    *,
    template: Any,
    ordinal: int,
    profile: Any,
    attachments: tuple[str, ...] | None = None,
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            _operation_identity_payload(
                template=template,
                ordinal=ordinal,
                profile=profile,
                attachments=attachments,
            )
        )
    ).hexdigest()
    return "op_" + digest[:32]


def _canonical_plan_size_upper_bound(
    *,
    manifest: JobManifestV1,
    templates: list[Any],
    profiles: dict[str, Any],
    operation_costs: dict[str, int],
    attachment_aliases: dict[str, str],
    candidates_by_path: dict[str, _AttachmentCandidate],
    manifest_hash: str,
    output_directory: str,
    total_micros: int,
) -> int:
    """Bound expanded canonical JSON without materializing repeated operations."""
    attachment_payload = [
        {
            "relative_path": path,
            "sha256": "0" * 64,
            "size_bytes": candidates_by_path[path].size_bytes,
            "mime_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
        }
        for path in sorted(candidates_by_path)
    ]
    fixed_payload = {
        "version": MANIFEST_VERSION,
        "name": manifest.name,
        "manifest_hash": manifest_hash,
        "operations": [],
        "attachments": attachment_payload,
        "max_concurrency": manifest.execution.max_concurrency,
        "max_attempts": manifest.execution.max_attempts,
        "call_timeout_s": manifest.execution.call_timeout_s,
        "max_wall_seconds": manifest.execution.max_wall_seconds,
        "max_budget_usd": usd_string(manifest.execution.max_budget_usd),
        "worst_case_cost_usd": usd_string(micros_to_usd(total_micros)),
        "output_directory": output_directory,
    }
    total_bytes = len(canonical_json_bytes(fixed_payload))
    operation_count = 0
    for template in templates:
        profile = profiles[template.profile]
        canonical_attachments = [attachment_aliases[path] for path in template.attachments]
        representative = {
            "operation_id": "op_" + "0" * 32,
            "template_key": template.key,
            "ordinal": template.count - 1,
            "prompt": template.prompt,
            "profile_name": profile.name,
            "provider": profile.provider.value,
            "model": profile.model,
            "options": detach_json(
                profile.options,
                context=f"profiles.{profile.name}.options",
            ),
            "attachments": canonical_attachments,
            "artifact": template.artifact.to_dict(),
            "max_attempts": manifest.execution.max_attempts,
            "max_cost_per_call_usd": usd_string(profile.max_cost_per_call_usd),
            "worst_case_cost_usd": usd_string(micros_to_usd(operation_costs[template.key])),
        }
        total_bytes += len(canonical_json_bytes(representative)) * template.count
        operation_count += template.count
        if total_bytes > MAX_CANONICAL_PLAN_BYTES:
            return total_bytes
    if operation_count:
        total_bytes += operation_count - 1  # commas between operation objects
    return total_bytes


def preflight_job(manifest: JobManifestV1, *, manifest_root: str | Path) -> JobPlanV1:
    """Expand a manifest and prove its complete worst-case spend before work."""
    if not isinstance(manifest, JobManifestV1):
        raise TypeError("manifest must be a JobManifestV1")
    root = Path(manifest_root)
    if not root.is_dir():
        raise PreflightError(f"manifest_root is not a directory: {root}")

    # Output confinement is checked even though preflight does not create it.
    _, output_directory = _safe_relative_path(
        manifest.execution.output_directory,
        root,
        context="execution.output_directory",
    )

    profiles = {profile.name: profile for profile in manifest.profiles}
    templates = sorted(manifest.operations, key=lambda item: item.key)

    manifest_payload = manifest.to_dict()
    manifest_payload["profiles"] = sorted(
        manifest_payload["profiles"], key=lambda item: item["name"]
    )
    manifest_payload["operations"] = sorted(
        manifest_payload["operations"], key=lambda item: item["key"]
    )
    canonical_manifest = canonical_json_bytes(manifest_payload)
    if len(canonical_manifest) > MAX_MANIFEST_BYTES:
        raise PreflightError(f"canonical manifest exceeds {MAX_MANIFEST_BYTES} UTF-8 bytes")
    manifest_hash = "sha256:" + hashlib.sha256(canonical_manifest).hexdigest()

    # Prove all arithmetic and the inclusive budget before reading attachments
    # or allocating the expanded operation list.
    operation_costs: dict[str, int] = {}
    total_micros = 0
    for template in templates:
        try:
            profile = profiles[template.profile]
        except KeyError as exc:
            raise PreflightError(
                f"operation {template.key!r} references unknown profile " f"{template.profile!r}"
            ) from exc

        per_call_micros = usd_to_micros(profile.max_cost_per_call_usd)
        if profile.provider is ProviderKind.OFFLINE:
            if per_call_micros != 0:
                raise PreflightError(
                    f"offline profile {profile.name!r} must have a zero call ceiling"
                )
        elif per_call_micros <= 0:
            raise PreflightError(
                f"billable profile {profile.name!r} requires a positive inclusive "
                "max_cost_per_call_usd"
            )

        if per_call_micros > MAX_SIGNED_MICRO_USD // manifest.execution.max_attempts:
            raise PreflightError(
                f"operation {template.key!r} retry ceiling exceeds signed 63-bit "
                "microdollar capacity"
            )
        operation_cost_micros = per_call_micros * manifest.execution.max_attempts
        if operation_cost_micros and (
            template.count > (MAX_SIGNED_MICRO_USD - total_micros) // operation_cost_micros
        ):
            raise PreflightError("job worst-case cost exceeds signed 63-bit microdollar capacity")
        total_micros += operation_cost_micros * template.count
        operation_costs[template.key] = operation_cost_micros

    maximum_micros = usd_to_micros(manifest.execution.max_budget_usd)
    if total_micros > maximum_micros:
        raise PreflightError(
            "job worst-case cost "
            f"${usd_string(micros_to_usd(total_micros))} exceeds execution budget "
            f"${usd_string(manifest.execution.max_budget_usd)}"
        )

    referenced_attachments = sorted(
        {path for template in manifest.operations for path in template.attachments}
    )
    if len(referenced_attachments) > MAX_UNIQUE_ATTACHMENTS:
        raise PreflightError(
            "job references "
            f"{len(referenced_attachments)} attachment paths; maximum unique "
            f"attachments is {MAX_UNIQUE_ATTACHMENTS}"
        )
    attachment_aliases: dict[str, str] = {}
    candidates_by_path: dict[str, _AttachmentCandidate] = {}
    for path in referenced_attachments:
        candidate = _attachment_candidate(path, root)
        attachment_aliases[path] = candidate.relative_path
        candidates_by_path.setdefault(candidate.relative_path, candidate)
    if len(candidates_by_path) > MAX_UNIQUE_ATTACHMENTS:
        raise PreflightError(
            "job references "
            f"{len(candidates_by_path)} unique attachments; maximum is "
            f"{MAX_UNIQUE_ATTACHMENTS}"
        )
    aggregate_attachment_bytes = sum(
        candidate.size_bytes for candidate in candidates_by_path.values()
    )
    if aggregate_attachment_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
        limit_mib = MAX_TOTAL_ATTACHMENT_BYTES // (1024 * 1024)
        raise PreflightError(f"aggregate attachment size exceeds {limit_mib} MiB")
    plan_size_upper_bound = _canonical_plan_size_upper_bound(
        manifest=manifest,
        templates=templates,
        profiles=profiles,
        operation_costs=operation_costs,
        attachment_aliases=attachment_aliases,
        candidates_by_path=candidates_by_path,
        manifest_hash=manifest_hash,
        output_directory=output_directory,
        total_micros=total_micros,
    )
    if plan_size_upper_bound > MAX_CANONICAL_PLAN_BYTES:
        limit_mib = MAX_CANONICAL_PLAN_BYTES // (1024 * 1024)
        raise PreflightError(
            "expanded canonical plan upper bound "
            f"({plan_size_upper_bound} bytes) exceeds {limit_mib} MiB"
        )
    fingerprints = tuple(
        _fingerprint(candidates_by_path[path]) for path in sorted(candidates_by_path)
    )

    planned: list[PlannedOperationV1] = []
    seen_ids: set[str] = set()
    for template in templates:
        canonical_attachments = tuple(attachment_aliases[path] for path in template.attachments)
        if len(set(canonical_attachments)) != len(canonical_attachments):
            raise PreflightError(
                f"operation {template.key!r} contains duplicate attachment paths "
                "after normalization"
            )
        profile = profiles[template.profile]
        operation_cost_micros = operation_costs[template.key]
        for ordinal in range(template.count):
            op_id = operation_id(
                template=template,
                ordinal=ordinal,
                profile=profile,
                attachments=canonical_attachments,
            )
            if op_id in seen_ids:  # pragma: no cover - cryptographic collision guard
                raise PreflightError(f"operation ID collision: {op_id}")
            seen_ids.add(op_id)
            planned.append(
                PlannedOperationV1(
                    operation_id=op_id,
                    template_key=template.key,
                    ordinal=ordinal,
                    prompt=template.prompt,
                    profile_name=profile.name,
                    provider=profile.provider,
                    model=profile.model,
                    options=profile.options,
                    attachments=canonical_attachments,
                    artifact=template.artifact,
                    max_attempts=manifest.execution.max_attempts,
                    max_cost_per_call_usd=profile.max_cost_per_call_usd,
                    worst_case_cost_usd=micros_to_usd(operation_cost_micros),
                )
            )

    provisional_plan = JobPlanV1(
        version=MANIFEST_VERSION,
        name=manifest.name,
        manifest_hash=manifest_hash,
        plan_hash="",
        manifest_json=canonical_manifest.decode("utf-8"),
        operations=tuple(planned),
        attachments=fingerprints,
        max_concurrency=manifest.execution.max_concurrency,
        max_attempts=manifest.execution.max_attempts,
        call_timeout_s=manifest.execution.call_timeout_s,
        max_wall_seconds=manifest.execution.max_wall_seconds,
        max_budget_usd=manifest.execution.max_budget_usd,
        worst_case_cost_usd=micros_to_usd(total_micros),
        output_directory=output_directory,
    )
    actual_plan_bytes = len(canonical_json_bytes(_plan_identity_payload(provisional_plan)))
    if actual_plan_bytes > MAX_CANONICAL_PLAN_BYTES:  # pragma: no cover - estimator guard
        raise PreflightError("expanded canonical plan exceeds its admitted size cap")
    return replace(
        provisional_plan,
        plan_hash=_sha256_json(_plan_identity_payload(provisional_plan)),
    )


def _approval_token(
    *,
    manifest_hash: str,
    plan_hash: str,
    approved_micros: int,
) -> str:
    payload = {
        "version": MANIFEST_VERSION,
        "manifest_hash": manifest_hash,
        "plan_hash": plan_hash,
        "approved_max_cost_microusd": approved_micros,
    }
    return "approve_v1_" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def make_approval(
    plan: JobPlanV1,
    *,
    approved_max_cost_usd: object | None = None,
) -> JobApprovalV1:
    """Bind an explicit approved ceiling to the exact preflight plan."""
    _verify_plan_identity(plan)
    approved = normalize_usd(
        plan.max_budget_usd if approved_max_cost_usd is None else approved_max_cost_usd,
        field_name="approved_max_cost_usd",
    )
    approved_micros = usd_to_micros(approved)
    if approved_micros < plan.worst_case_cost_microusd:
        raise ApprovalError("approved ceiling is below the job's worst-case cost")
    if approved_micros > plan.max_budget_microusd:
        raise ApprovalError("approved ceiling exceeds the manifest budget")
    return JobApprovalV1(
        version=MANIFEST_VERSION,
        manifest_hash=plan.manifest_hash,
        plan_hash=plan.plan_hash,
        approved_max_cost_usd=approved,
        token=_approval_token(
            manifest_hash=plan.manifest_hash,
            plan_hash=plan.plan_hash,
            approved_micros=approved_micros,
        ),
    )


def verify_approval(plan: JobPlanV1, approval: JobApprovalV1) -> None:
    """Fail unless an approval matches the plan and its exact USD ceiling."""
    _verify_plan_identity(plan)
    if approval.version != MANIFEST_VERSION:
        raise ApprovalError("unsupported approval version")
    if not hmac.compare_digest(approval.manifest_hash, plan.manifest_hash):
        raise ApprovalError("approval manifest hash does not match")
    if not hmac.compare_digest(approval.plan_hash, plan.plan_hash):
        raise ApprovalError("approval plan hash does not match")
    if approval.approved_max_cost_microusd < plan.worst_case_cost_microusd:
        raise ApprovalError("approved ceiling is below the job's worst-case cost")
    if approval.approved_max_cost_microusd > plan.max_budget_microusd:
        raise ApprovalError("approved ceiling exceeds the manifest budget")
    expected = _approval_token(
        manifest_hash=approval.manifest_hash,
        plan_hash=approval.plan_hash,
        approved_micros=approval.approved_max_cost_microusd,
    )
    if not hmac.compare_digest(approval.token, expected):
        raise ApprovalError("approval token is invalid")
