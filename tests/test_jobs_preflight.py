from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import re
from types import MappingProxyType

import pytest
import smythe.jobs.preflight as preflight_module
import smythe.jobs.providers as providers_module

from smythe.jobs import (
    ApprovalError,
    JobApprovalV1,
    JobManifestV1,
    PreflightError,
    ProviderKind,
    make_approval,
    operation_input_digest,
    preflight_job,
    verify_approval,
)
from smythe.jobs.models import (
    MAX_ATTACHMENT_BYTES,
    MAX_TOTAL_ATTACHMENT_BYTES,
    MAX_TOTAL_OPERATIONS,
    MAX_UNIQUE_ATTACHMENTS,
)
from smythe.jobs.providers import ADAPTER_VERSIONS
from smythe.validation import ValidationReport, ValidatorIdentity


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


def test_existing_manifests_keep_their_hashes_operation_ids_and_approval_tokens(tmp_path):
    """Manifests without validators keep the identities earlier releases computed."""
    data = {
        "version": 1, "name": "hash-stability",
        "profiles": [{"name": "image", "provider": "openai_image", "model": "gpt-image-1.5",
                      "max_cost_per_call_usd": "0.08", "options": {"quality": "medium"}}],
        "operations": [
            {"key": "hero", "count": 2, "prompt": "Create the campaign hero", "profile": "image",
             "artifact": {"mime_type": "image/png", "width": 1200, "height": 630,
                          "filename": "hero.png"}},
            {"key": "square", "prompt": "Create a square", "profile": "image"},
        ],
        "execution": {"max_concurrency": 4, "max_attempts": 3, "max_budget_usd": "0.72",
                      "output_directory": "output/campaign"},
    }
    plan = preflight_job(JobManifestV1.from_dict(data), manifest_root=tmp_path)

    assert plan.manifest_hash == (
        "sha256:bd10391af6212b9ecee21b6d3f5187f2296cd9025a53614fb8b9f3aa2cde37c6")
    assert plan.plan_hash == (
        "sha256:5ad8c44330e5800fc504f11c9e59c66f4cebf5d1938ec74bd8e4ae1108ea94d0")
    assert [operation.operation_id for operation in plan.operations] == [
        "op_42263f62dcbc6b69423027a5baa2a9ed",
        "op_1d089d7e588ea177462b8cb794f63660",
        "op_cdd0a41200177a9b8d7e74ede2d0b08d",
    ]
    assert make_approval(plan).token == (
        "approve_v1_eef58c019ef36d23d8c96918db30888ab56eec8c0ad366286a7dc47178104490")
    assert all("validators" not in operation.to_dict()
               and "validators" not in operation.to_dict()["artifact"]
               for operation in plan.operations)


class PaletteValidator:
    def __init__(self, *, name="brand-palette", version="1", colors=("#000000",)):
        self.identity = ValidatorIdentity.for_config(name, version, {"colors": list(colors)})

    def validate(self, subject):
        return ValidationReport("artifact", validator=self.identity)


def validator_manifest(version="1"):
    data = manifest_data()
    data["operations"][0]["artifact"]["validators"] = [{"name": "brand-palette", "version": version}]
    return JobManifestV1.from_dict(data)


def test_validators_a_manifest_names_must_be_registered_before_any_run(tmp_path):
    write_logo(tmp_path)
    with pytest.raises(PreflightError, match="'brand-palette'.*not registered"):
        preflight_job(validator_manifest(), manifest_root=tmp_path)
    with pytest.raises(PreflightError, match="version '1'.*version '2' is registered"):
        preflight_job(validator_manifest(), manifest_root=tmp_path,
                      validators=[PaletteValidator(version="2")])

    validator = PaletteValidator()
    plan = preflight_job(validator_manifest(), manifest_root=tmp_path, validators=[validator])
    unchecked = preflight_job(JobManifestV1.from_dict(manifest_data()), manifest_root=tmp_path)

    assert all(operation.validators == (validator.identity,) for operation in plan.operations)
    assert plan.to_dict()["operations"][0]["validators"] == [validator.identity.to_dict()]
    assert plan.to_dict()["operations"][0]["artifact"]["validators"] == [
        {"name": "brand-palette", "version": "1"}]
    assert plan.plan_hash != unchecked.plan_hash
    assert {op.operation_id for op in plan.operations}.isdisjoint(
        op.operation_id for op in unchecked.operations)
    verify_approval(plan, make_approval(plan))


def test_a_validator_configuration_change_invalidates_the_approval(tmp_path):
    write_logo(tmp_path)
    first = preflight_job(validator_manifest(), manifest_root=tmp_path,
                          validators=[PaletteValidator()])
    changed = preflight_job(validator_manifest(), manifest_root=tmp_path,
                            validators=[PaletteValidator(colors=("#ffffff",))])

    assert [op.operation_id for op in first.operations] == [op.operation_id for op in changed.operations]
    assert first.manifest_hash == changed.manifest_hash
    assert first.plan_hash != changed.plan_hash
    with pytest.raises(ApprovalError, match="plan hash"):
        verify_approval(changed, make_approval(first))


def test_validator_registration_refuses_providers_duplicates_and_reserved_names(tmp_path):
    from smythe.provider import OfflineProvider

    write_logo(tmp_path)
    manifest = JobManifestV1.from_dict(manifest_data())
    with pytest.raises(TypeError, match="Provider cannot be a validator"):
        preflight_job(manifest, manifest_root=tmp_path, validators=[OfflineProvider()])
    with pytest.raises(ValueError, match="more than one validator named 'brand-palette'"):
        preflight_job(manifest, manifest_root=tmp_path,
                      validators=[PaletteValidator(), PaletteValidator(version="2")])
    with pytest.raises(ValueError, match="reserved"):
        preflight_job(manifest, manifest_root=tmp_path,
                      validators=[PaletteValidator(name="smythe.jobs.artifact-spec")])
    with pytest.raises(TypeError, match="list or tuple"):
        preflight_job(manifest, manifest_root=tmp_path,
                      validators={"brand-palette": PaletteValidator()})


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


def reference_manifest_data() -> dict:
    """A Gemini job whose operations each send two reference images."""
    return {
        "version": 1, "name": "input-digest",
        "profiles": [{"name": "render", "provider": "gemini_image",
                      "model": "gemini-2.5-flash-image", "max_cost_per_call_usd": "0.05",
                      "options": {"response_modalities": ["IMAGE"]}}],
        "operations": [{"key": "hero", "count": 2, "prompt": "Render the hero with the brand mark",
                        "profile": "render",
                        "attachments": ["brand/logo.png", "brand/palette.png"],
                        "artifact": {"mime_type": "image/png", "width": 64, "height": 64}}],
        "execution": {"max_concurrency": 2, "max_attempts": 2, "max_budget_usd": "0.20",
                      "output_directory": "outputs"},
    }


def write_references(root: Path) -> None:
    (root / "brand").mkdir()
    (root / "brand" / "logo.png").write_bytes(b"logo-bytes")
    (root / "brand" / "palette.png").write_bytes(b"palette-bytes")


def reference_plan(root: Path, data: dict | None = None):
    return preflight_job(JobManifestV1.from_dict(data or reference_manifest_data()),
                         manifest_root=root)


def input_digests(plan) -> list[str]:
    return [operation_input_digest(operation, plan.attachments) for operation in plan.operations]


def test_input_digest_leaves_operation_ids_plan_hash_and_approval_token_unchanged(tmp_path):
    write_references(tmp_path)
    plan = reference_plan(tmp_path)

    assert plan.plan_hash == (
        "sha256:476f09e53f989b40c73b3a6a151099f057e3f29deccaf0d1410a4cdaa1d13623")
    assert [operation.operation_id for operation in plan.operations] == [
        "op_b83e02dbd3391fc783dbc639afca697f",
        "op_89fa2e8fc204649162ac71d345eab47a",
    ]
    assert make_approval(plan).token == (
        "approve_v1_bd8b584b6bbda538316c21c48035ef97107089d882f529db156f2002f0b3cd49")
    # Version 1 of the digest; a change to its inputs or encoding fails here.
    assert input_digests(plan) == [
        "sha256:3ecb942e725b6651526af6d922206093612c1e79b49d15384eb3916398d83d59",
    ] * 2
    assert "input_digest" not in json.dumps(plan.to_dict())


def _more_samples(data: dict, root: Path) -> None:
    data["operations"][0]["count"] = 5
    data["execution"]["max_budget_usd"] = "0.50"


def _renamed_profile(data: dict, root: Path) -> None:
    data["profiles"][0]["name"] = "renamed"
    data["operations"][0]["profile"] = "renamed"


def _lower_ceiling(data: dict, root: Path) -> None:
    data["profiles"][0]["max_cost_per_call_usd"] = "0.04"


def _fewer_attempts(data: dict, root: Path) -> None:
    data["execution"]["max_attempts"] = 1


def _renamed_template(data: dict, root: Path) -> None:
    data["operations"][0]["key"] = "banner"


def _moved_attachment(data: dict, root: Path) -> None:
    (root / "assets").mkdir()
    (root / "assets" / "mark.png").write_bytes((root / "brand" / "logo.png").read_bytes())
    data["operations"][0]["attachments"][0] = "assets/mark.png"


@pytest.mark.parametrize("change", [
    _more_samples, _renamed_profile, _lower_ceiling, _fewer_attempts, _renamed_template,
    _moved_attachment,
])
def test_input_digest_ignores_sampling_limits_names_and_attachment_paths(tmp_path, change):
    write_references(tmp_path)
    original = reference_plan(tmp_path)
    data = reference_manifest_data()
    change(data, tmp_path)
    changed = reference_plan(tmp_path, data)

    assert changed.plan_hash != original.plan_hash
    # Every sample of the template shares the original digest.
    assert set(input_digests(changed)) == set(input_digests(original))
    assert len(set(input_digests(changed))) == 1


def _new_prompt(data: dict, root: Path) -> None:
    data["operations"][0]["prompt"] = "Render the banner with the brand mark"


def _new_model(data: dict, root: Path) -> None:
    data["profiles"][0]["model"] = "gemini-3-pro-image-preview"


def _new_options(data: dict, root: Path) -> None:
    data["profiles"][0]["options"] = {"response_modalities": ["TEXT", "IMAGE"]}


def _new_artifact_spec(data: dict, root: Path) -> None:
    data["operations"][0]["artifact"].update(width=128, height=128)


def _new_attachment_bytes(data: dict, root: Path) -> None:
    (root / "brand" / "logo.png").write_bytes(b"logo-bytes-v2")


def _new_attachment_type(data: dict, root: Path) -> None:
    (root / "brand" / "logo.jpg").write_bytes((root / "brand" / "logo.png").read_bytes())
    data["operations"][0]["attachments"][0] = "brand/logo.jpg"


def _new_attachment_order(data: dict, root: Path) -> None:
    data["operations"][0]["attachments"].reverse()


@pytest.mark.parametrize("change", [
    _new_prompt, _new_model, _new_options, _new_artifact_spec, _new_attachment_bytes,
    _new_attachment_type, _new_attachment_order,
])
def test_input_digest_changes_with_what_the_provider_is_asked_for(tmp_path, change):
    write_references(tmp_path)
    original = reference_plan(tmp_path)
    data = reference_manifest_data()
    change(data, tmp_path)
    changed = reference_plan(tmp_path, data)

    assert set(input_digests(changed)).isdisjoint(input_digests(original))


@pytest.mark.parametrize("guessed", [("application/octet-stream", None), (None, None),
                                     ("image/x-png", None)])
def test_input_digest_does_not_depend_on_the_platform_mime_database(tmp_path, monkeypatch, guessed):
    """The digest used the MIME type mimetypes guessed, which varies by platform
    and Python version (Windows Python 3.12 guesses no type for .webp)."""
    write_references(tmp_path)
    (tmp_path / "brand" / "mark.webp").write_bytes(b"mark-bytes")
    (tmp_path / "brand" / "photo.JPG").write_bytes(b"photo-bytes")
    data = reference_manifest_data()
    data["operations"][0]["attachments"] += ["brand/mark.webp", "brand/photo.JPG"]
    original = reference_plan(tmp_path, data)
    monkeypatch.setattr(preflight_module.mimetypes, "guess_type", lambda path, strict=True: guessed)
    elsewhere = reference_plan(tmp_path, data)

    assert {item.mime_type for item in elsewhere.attachments} != {
        item.mime_type for item in original.attachments}
    assert input_digests(elsewhere) == input_digests(original)


def test_input_digest_reads_attachment_types_from_a_fixed_extension_table():
    fingerprint = {"relative_path": "a.png", "sha256": "0" * 64, "mime_type": "text/plain"}
    operation = {"provider": "gemini_image", "model": "m", "options": {}, "prompt": "p",
                 "artifact": {}, "attachments": ["a.png"]}

    def digest(path: str, recorded: str = "text/plain") -> str:
        return operation_input_digest(operation | {"attachments": [path]},
                                      [fingerprint | {"relative_path": path, "mime_type": recorded}])

    assert digest("a.png") == digest("b.PNG", "image/png")
    assert digest("a.jpg") == digest("a.jpeg") == digest("a.JPEG", "application/octet-stream")
    assert len({digest("a.png"), digest("a.jpg"), digest("a.gif"), digest("a.webp")}) == 4
    # Any other type counts as application/octet-stream whatever was guessed.
    assert digest("notes.txt") == digest("notes.md", "text/markdown") == digest("blob")


def test_input_digest_changes_with_the_provider_adapter_version(tmp_path, monkeypatch):
    write_references(tmp_path)
    plan = reference_plan(tmp_path)
    original = input_digests(plan)
    bumped = dict(ADAPTER_VERSIONS)
    bumped[ProviderKind.GEMINI_IMAGE] += 1
    monkeypatch.setattr(providers_module, "ADAPTER_VERSIONS", MappingProxyType(bumped))

    assert set(input_digests(plan)).isdisjoint(original)
    verify_approval(plan, make_approval(plan))


def test_every_provider_kind_has_an_adapter_version():
    assert set(ADAPTER_VERSIONS) == set(ProviderKind)
    assert all(type(version) is int and version >= 1 for version in ADAPTER_VERSIONS.values())
    with pytest.raises(TypeError):
        ADAPTER_VERSIONS[ProviderKind.OFFLINE] = 2  # type: ignore[index]


def test_input_digest_of_a_stored_plan_matches_its_planned_operations(tmp_path):
    write_references(tmp_path)
    plan = reference_plan(tmp_path)
    stored = json.loads(json.dumps(plan.to_dict()))

    assert [operation_input_digest(operation, stored["attachments"])
            for operation in stored["operations"]] == input_digests(plan)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", input_digests(plan)[0])
    with pytest.raises(ValueError, match="brand/logo.png"):
        operation_input_digest(plan.operations[0], plan.attachments[1:])


@pytest.mark.parametrize("corrupt", ["object", "text", "null", "number"])
def test_input_digest_refuses_attachments_that_are_not_an_array(tmp_path, corrupt):
    """An object of planned paths used to digest like the array of its keys."""
    write_references(tmp_path)
    plan = reference_plan(tmp_path)
    stored = json.loads(json.dumps(plan.to_dict()))
    operation = stored["operations"][0]
    paths = operation["attachments"]
    assert paths
    # An in-memory tuple is an array too.
    assert operation_input_digest(operation | {"attachments": tuple(paths)},
                                  stored["attachments"]) == input_digests(plan)[0]
    attachments = {"object": dict.fromkeys(paths, 1), "text": paths[0],
                   "null": None, "number": 5}[corrupt]

    with pytest.raises(ValueError, match="attachments must be an array"):
        operation_input_digest(operation | {"attachments": attachments}, stored["attachments"])
