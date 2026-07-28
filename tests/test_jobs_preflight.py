from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import smythe.jobs.preflight as preflight_module

from smythe.jobs import (
    ApprovalError,
    JobApprovalV1,
    JobManifestV1,
    PreflightError,
    make_approval,
    preflight_job,
    verify_approval,
)
from smythe.jobs.models import (
    MAX_ATTACHMENT_BYTES,
    MAX_TOTAL_ATTACHMENT_BYTES,
    MAX_TOTAL_OPERATIONS,
    MAX_UNIQUE_ATTACHMENTS,
)


def manifest_data(*, provider: str = "openai_image") -> dict:
    return {
        "version": 1,
        "name": "campaign-demo",
        "profiles": [
            {
                "name": "image",
                "provider": provider,
                "model": "offline" if provider == "offline" else "gpt-image-1.5",
                "max_cost_per_call_usd": 0 if provider == "offline" else 0.08,
                "options": {"quality": "medium"},
            }
        ],
        "operations": [
            {
                "key": "hero",
                "count": 2,
                "prompt": "Create the campaign hero",
                "profile": "image",
                "attachments": ["brand/logo.png"],
                "artifact": {"mime_type": "image/png", "width": 1200, "height": 630},
            }
        ],
        "execution": {
            "max_concurrency": 4,
            "max_attempts": 3,
            "max_budget_usd": 0.48 if provider != "offline" else 0,
            "output_directory": "output/campaign-demo",
        },
    }


def write_logo(root: Path, data: bytes = b"logo-v1") -> None:
    path = root / "brand" / "logo.png"
    path.parent.mkdir()
    path.write_bytes(data)


def test_preflight_expands_counts_fingerprints_inputs_and_prices_retries(tmp_path):
    write_logo(tmp_path)
    plan = preflight_job(JobManifestV1.from_dict(manifest_data()), manifest_root=tmp_path)

    assert len(plan.operations) == 2
    assert [operation.ordinal for operation in plan.operations] == [0, 1]
    assert len({operation.operation_id for operation in plan.operations}) == 2
    assert all(operation.operation_id.startswith("op_") for operation in plan.operations)
    assert all(operation.max_attempts == 3 for operation in plan.operations)
    assert all(
        operation.worst_case_cost_usd == Decimal("0.240000") for operation in plan.operations
    )
    assert plan.worst_case_cost_usd == Decimal("0.480000")
    assert plan.max_budget_microusd == 480_000
    assert plan.attachments[0].relative_path == "brand/logo.png"
    assert plan.attachments[0].size_bytes == len(b"logo-v1")
    assert plan.attachments[0].sha256


def test_preflight_supports_the_exact_5000_operation_target(tmp_path):
    data = manifest_data(provider="offline")
    data["operations"][0]["count"] = MAX_TOTAL_OPERATIONS
    data["operations"][0]["attachments"] = []

    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    assert len(plan.operations) == 5_000
    assert plan.worst_case_cost_microusd == 0
    verify_approval(plan, make_approval(plan))


def test_preflight_binds_approved_call_and_wall_deadlines_into_plan_identity(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    data["execution"]["call_timeout_s"] = 45.5
    data["execution"]["max_wall_seconds"] = 900
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    approval = make_approval(plan)

    assert plan.call_timeout_s == 45.5
    assert plan.max_wall_seconds == 900.0
    assert plan.to_dict()["call_timeout_s"] == 45.5
    assert plan.to_dict()["max_wall_seconds"] == 900.0
    verify_approval(plan, approval)

    with pytest.raises(ApprovalError, match="plan identity has drifted"):
        verify_approval(replace(plan, call_timeout_s=46.0), approval)


def test_operation_ids_are_stable_when_unrelated_template_is_added(tmp_path):
    write_logo(tmp_path)
    original = manifest_data()
    first = preflight_job(JobManifestV1.from_dict(original), manifest_root=tmp_path)

    changed = manifest_data()
    changed["operations"].append(
        {
            "key": "square",
            "count": 1,
            "prompt": "Create a square image",
            "profile": "image",
            "artifact": {"mime_type": "image/png"},
        }
    )
    changed["execution"]["max_budget_usd"] = 0.72
    second = preflight_job(JobManifestV1.from_dict(changed), manifest_root=tmp_path)

    assert [op.operation_id for op in first.operations] == [
        op.operation_id for op in second.operations if op.template_key == "hero"
    ]
    assert first.manifest_hash != second.manifest_hash
    assert first.plan_hash != second.plan_hash


def test_manifest_order_does_not_change_semantic_hashes(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    data["profiles"].append(
        {
            "name": "unused",
            "provider": "offline",
            "model": "offline",
            "max_cost_per_call_usd": 0,
        }
    )
    first = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)
    data["profiles"].reverse()
    second = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    assert first.manifest_hash == second.manifest_hash
    assert first.plan_hash == second.plan_hash


def test_attachment_content_is_bound_to_plan_not_manifest(tmp_path):
    write_logo(tmp_path, b"first")
    manifest = JobManifestV1.from_dict(manifest_data())
    first = preflight_job(manifest, manifest_root=tmp_path)
    (tmp_path / "brand" / "logo.png").write_bytes(b"second")
    second = preflight_job(manifest, manifest_root=tmp_path)

    assert first.manifest_hash == second.manifest_hash
    assert first.plan_hash != second.plan_hash
    assert first.attachments[0].sha256 != second.attachments[0].sha256


def test_planned_options_are_deeply_immutable_and_exports_are_detached(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    source_options = {"render": {"palette": ["green", {"glow": 0.8}]}}
    data["profiles"][0]["options"] = source_options
    manifest = JobManifestV1.from_dict(data)
    plan = preflight_job(manifest, manifest_root=tmp_path)
    options = plan.operations[0].options

    source_options["render"]["palette"][1]["glow"] = 0.1
    assert options["render"]["palette"][1]["glow"] == 0.8
    with pytest.raises((TypeError, AttributeError)):
        options["render"]["palette"].append("blue")
    with pytest.raises(TypeError):
        dict.__setitem__(options, "bypass", True)

    exported = plan.to_dict()
    exported["operations"][0]["options"]["render"]["palette"][1]["glow"] = 0.2
    assert options["render"]["palette"][1]["glow"] == 0.8


@pytest.mark.parametrize("attachment", ["../outside.png", "missing.png"])
def test_attachment_must_exist_inside_manifest_root(tmp_path, attachment):
    data = manifest_data()
    data["operations"][0]["attachments"] = [attachment]
    with pytest.raises(PreflightError):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_symlink_attachment_cannot_escape_manifest_root(tmp_path):
    outside = tmp_path.parent / "outside-job-secret.png"
    outside.write_bytes(b"secret")
    link = tmp_path / "logo.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    data = manifest_data()
    data["operations"][0]["attachments"] = ["logo.png"]
    with pytest.raises(PreflightError, match="escapes"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_attachment_size_is_rejected_from_stat_before_hashing(tmp_path, monkeypatch):
    oversized = tmp_path / "oversized.bin"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_ATTACHMENT_BYTES + 1)
    data = manifest_data(provider="offline")
    data["operations"][0]["attachments"] = [oversized.name]

    monkeypatch.setattr(
        preflight_module,
        "_fingerprint",
        lambda _candidate: pytest.fail("oversized attachment was opened"),
    )
    with pytest.raises(PreflightError, match="attachment exceeds"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_aggregate_attachment_cap_is_checked_before_hashing(tmp_path, monkeypatch):
    attachment_paths = []
    attachment_size = MAX_TOTAL_ATTACHMENT_BYTES // 8
    for index in range(9):
        path = tmp_path / f"asset-{index}.bin"
        with path.open("wb") as stream:
            stream.truncate(attachment_size)
        attachment_paths.append(path.name)
    data = manifest_data(provider="offline")
    data["operations"][0]["attachments"] = attachment_paths

    monkeypatch.setattr(
        preflight_module,
        "_fingerprint",
        lambda _candidate: pytest.fail("aggregate-overflow attachment was opened"),
    )
    with pytest.raises(PreflightError, match="aggregate attachment size"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_unique_attachment_cap_is_checked_before_hashing(tmp_path, monkeypatch):
    paths = []
    for index in range(MAX_UNIQUE_ATTACHMENTS + 1):
        path = tmp_path / f"asset-{index}.bin"
        path.write_bytes(b"")
        paths.append(path.name)
    data = manifest_data(provider="offline")
    data["operations"] = [
        {
            "key": f"batch-{start}",
            "prompt": "offline",
            "profile": "image",
            "attachments": paths[start : start + 16],
        }
        for start in range(0, len(paths), 16)
    ]

    monkeypatch.setattr(
        preflight_module,
        "_fingerprint",
        lambda _candidate: pytest.fail("excess unique attachments were opened"),
    )
    with pytest.raises(PreflightError, match="unique attachments"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_output_directory_cannot_escape_manifest_root(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    data["execution"]["output_directory"] = "../outside"
    with pytest.raises(PreflightError, match="escapes"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_preflight_fails_closed_for_unbounded_or_over_budget_paid_profile(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    data["profiles"][0]["max_cost_per_call_usd"] = 0
    with pytest.raises(PreflightError, match="positive inclusive"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    data = manifest_data()
    data["execution"]["max_budget_usd"] = 0.479999
    with pytest.raises(PreflightError, match="exceeds"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_offline_plan_requires_zero_ceiling_and_budget(tmp_path):
    write_logo(tmp_path)
    plan = preflight_job(
        JobManifestV1.from_dict(manifest_data(provider="offline")),
        manifest_root=tmp_path,
    )
    assert plan.worst_case_cost_microusd == 0

    data = manifest_data(provider="offline")
    data["profiles"][0]["max_cost_per_call_usd"] = 0.01
    data["execution"]["max_budget_usd"] = 1
    with pytest.raises(PreflightError, match="zero call ceiling"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_unknown_profile_fails_before_execution(tmp_path):
    write_logo(tmp_path)
    data = manifest_data()
    data["operations"][0]["profile"] = "missing"
    with pytest.raises(PreflightError, match="unknown profile"):
        preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)


def test_approval_is_bound_to_hashes_and_exact_ceiling(tmp_path):
    write_logo(tmp_path)
    plan = preflight_job(JobManifestV1.from_dict(manifest_data()), manifest_root=tmp_path)
    approval = make_approval(plan)

    verify_approval(plan, approval)
    verify_approval(plan, JobApprovalV1.from_dict(approval.to_dict()))
    assert approval.approved_max_cost_microusd == 480_000
    assert approval.token.startswith("approve_v1_")

    with pytest.raises(ApprovalError, match="below"):
        make_approval(plan, approved_max_cost_usd=0.479999)
    with pytest.raises(ApprovalError, match="exceeds"):
        make_approval(plan, approved_max_cost_usd=0.480001)

    with pytest.raises(ApprovalError, match="token"):
        verify_approval(plan, replace(approval, token=approval.token[:-1] + "0"))
    with pytest.raises(ApprovalError, match="plan hash"):
        verify_approval(plan, replace(approval, plan_hash="sha256:" + "0" * 64))


@pytest.mark.parametrize(
    "tampered_plan",
    [
        lambda plan: replace(plan, max_concurrency=plan.max_concurrency + 1),
        lambda plan: replace(plan, manifest_json=plan.manifest_json + " "),
        lambda plan: replace(
            plan,
            operations=(
                replace(plan.operations[0], prompt="mutated after approval"),
                *plan.operations[1:],
            ),
        ),
    ],
)
def test_approval_recomputes_current_manifest_and_plan_identity(
    tmp_path,
    tampered_plan,
):
    write_logo(tmp_path)
    plan = preflight_job(JobManifestV1.from_dict(manifest_data()), manifest_root=tmp_path)
    approval = make_approval(plan)

    with pytest.raises(ApprovalError, match="identity|canonical"):
        verify_approval(tampered_plan(plan), approval)
