from __future__ import annotations

from decimal import Decimal

import pytest

from smythe.jobs import (
    JOB_MANIFEST_V1_JSON_SCHEMA,
    ArtifactSpecV1,
    ExecutionPolicyV1,
    JobManifestV1,
    ManifestValidationError,
    ProviderKind,
    normalize_usd,
    usd_to_micros,
)
from smythe.jobs.models import (
    DEFAULT_CALL_TIMEOUT_S,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_ARTIFACT_DIMENSION,
    MAX_ARTIFACT_FILENAME_BYTES,
    MAX_ARTIFACT_PIXELS,
    MAX_ATTACHMENTS_PER_OPERATION,
    MAX_ATTEMPTS,
    MAX_CONCURRENCY,
    MAX_CALL_TIMEOUT_S,
    MAX_OPERATION_COUNT,
    MAX_OPERATION_TEMPLATES,
    MAX_OPTIONS_JSON_BYTES,
    MAX_PROFILES,
    MAX_PROMPT_BYTES,
    MAX_SIGNED_MICRO_USD,
    MAX_TOTAL_OPERATIONS,
    MAX_WALL_SECONDS,
    MICRODOLLAR,
    MAX_USD,
)


def manifest_data() -> dict:
    return {
        "version": 1,
        "name": "campaign-demo",
        "profiles": [
            {
                "name": "image",
                "provider": "openai_image",
                "model": "gpt-image-1.5",
                "max_cost_per_call_usd": 0.08,
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
                "artifact": {
                    "mime_type": "image/png",
                    "width": 1200,
                    "height": 630,
                    "filename": "hero.png",
                },
            }
        ],
        "execution": {
            "max_concurrency": 4,
            "max_attempts": 2,
            "max_budget_usd": 1,
            "output_directory": "output/campaign-demo",
        },
    }


def test_manifest_strict_round_trip_is_json_compatible():
    manifest = JobManifestV1.from_dict(manifest_data())

    assert manifest.version == 1
    assert manifest.profiles[0].provider is ProviderKind.OPENAI_IMAGE
    assert manifest.profiles[0].max_cost_per_call_usd == Decimal("0.080000")
    assert manifest.operations[0].attachments == ("brand/logo.png",)
    assert manifest.operations[0].artifact == ArtifactSpecV1(
        mime_type="image/png", width=1200, height=630, filename="hero.png"
    )
    assert manifest.to_dict()["execution"]["max_budget_usd"] == "1.000000"
    assert JobManifestV1.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


@pytest.mark.parametrize(
    ("value", "micros"),
    [("0", 0), (0.03, 30_000), ("12.345678", 12_345_678)],
)
def test_money_normalizes_to_exact_microdollars(value, micros):
    amount = normalize_usd(value, field_name="test")
    assert usd_to_micros(amount) == micros


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "0.0000001"])
def test_money_rejects_unsafe_values(value):
    with pytest.raises(ManifestValidationError):
        normalize_usd(value, field_name="test")


def test_manifest_rejects_unknown_fields_and_versions():
    data = manifest_data()
    data["surprise"] = True
    with pytest.raises(ManifestValidationError, match="unknown fields"):
        JobManifestV1.from_dict(data)

    data = manifest_data()
    data["version"] = 2
    with pytest.raises(ManifestValidationError, match="unsupported"):
        JobManifestV1.from_dict(data)


def test_manifest_rejects_duplicate_names_and_invalid_attachment_shape():
    data = manifest_data()
    data["profiles"].append(dict(data["profiles"][0]))
    with pytest.raises(ManifestValidationError, match="duplicate profile"):
        JobManifestV1.from_dict(data)

    data = manifest_data()
    data["operations"][0]["attachments"] = "logo.png"
    with pytest.raises(ManifestValidationError, match="attachments must be an array"):
        JobManifestV1.from_dict(data)


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": "secret"},
        {"headers": {"Authorization": "Bearer secret"}},
        {"nested": {"private-key": "secret"}},
        {"password": "secret"},
    ],
)
def test_provider_options_reject_embedded_secrets(options):
    data = manifest_data()
    data["profiles"][0]["options"] = options
    with pytest.raises(ManifestValidationError, match="may contain a secret"):
        JobManifestV1.from_dict(data)


def test_provider_options_allow_environment_variable_references():
    data = manifest_data()
    data["profiles"][0]["options"] = {"api_key_env": "OPENAI_API_KEY"}
    manifest = JobManifestV1.from_dict(data)
    assert manifest.profiles[0].options == {"api_key_env": "OPENAI_API_KEY"}


def test_profile_options_are_deeply_detached_frozen_and_exported_detached():
    data = manifest_data()
    source_options = {"render": {"palette": ["green", {"intensity": 0.75}]}}
    data["profiles"][0]["options"] = source_options
    manifest = JobManifestV1.from_dict(data)
    options = manifest.profiles[0].options

    source_options["render"]["palette"][1]["intensity"] = 0.1
    assert options["render"]["palette"][1]["intensity"] == 0.75
    with pytest.raises((TypeError, AttributeError)):
        options["render"]["palette"].append("blue")
    with pytest.raises(TypeError):
        options["render"]["palette"][1]["intensity"] = 0.5
    with pytest.raises(TypeError):
        dict.__setitem__(options, "bypass", True)
    with pytest.raises(TypeError):
        list.append(options["render"]["palette"], "bypass")

    exported = manifest.to_dict()
    exported["profiles"][0]["options"]["render"]["palette"][1]["intensity"] = 0.2
    assert options["render"]["palette"][1]["intensity"] == 0.75


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("count", MAX_OPERATION_COUNT + 1, "operation.count exceeds"),
        ("prompt", "é" * (MAX_PROMPT_BYTES // 2 + 1), "UTF-8 bytes"),
        (
            "attachments",
            [f"asset-{index}.png" for index in range(MAX_ATTACHMENTS_PER_OPERATION + 1)],
            "attachments exceeds",
        ),
        (
            "artifact",
            {"width": MAX_ARTIFACT_DIMENSION + 1, "height": 1},
            "width exceeds",
        ),
        (
            "artifact",
            {
                "width": MAX_ARTIFACT_DIMENSION,
                "height": MAX_ARTIFACT_PIXELS // MAX_ARTIFACT_DIMENSION + 1,
            },
            "dimensions exceed",
        ),
    ],
    ids=["count", "prompt-bytes", "attachments", "dimension", "pixels"],
)
def test_operation_resource_caps_are_enforced(field, value, match):
    data = manifest_data()
    data["operations"][0][field] = value
    with pytest.raises(ManifestValidationError, match=match):
        JobManifestV1.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("max_concurrency", MAX_CONCURRENCY + 1, "max_concurrency exceeds"),
        ("max_attempts", MAX_ATTEMPTS + 1, "max_attempts exceeds"),
    ],
)
def test_execution_resource_caps_are_enforced(field, value, match):
    data = manifest_data()
    data["execution"][field] = value
    with pytest.raises(ManifestValidationError, match=match):
        JobManifestV1.from_dict(data)


def test_options_profiles_templates_and_total_operation_caps_are_enforced():
    data = manifest_data()
    data["profiles"][0]["options"] = {"blob": "x" * MAX_OPTIONS_JSON_BYTES}
    with pytest.raises(ManifestValidationError, match="options exceeds"):
        JobManifestV1.from_dict(data)

    data = manifest_data()
    base_profile = data["profiles"][0]
    data["profiles"] = [
        {**base_profile, "name": f"profile-{index}"} for index in range(MAX_PROFILES + 1)
    ]
    with pytest.raises(ManifestValidationError, match="profiles exceeds"):
        JobManifestV1.from_dict(data)

    data = manifest_data()
    base_operation = data["operations"][0]
    data["operations"] = [
        {**base_operation, "key": f"operation-{index}", "count": 1}
        for index in range(MAX_OPERATION_TEMPLATES + 1)
    ]
    with pytest.raises(ManifestValidationError, match="operations exceeds"):
        JobManifestV1.from_dict(data)

    data = manifest_data()
    data["operations"][0]["count"] = MAX_TOTAL_OPERATIONS
    data["operations"].append(
        {
            "key": "one-too-many",
            "prompt": "extra",
            "profile": "image",
        }
    )
    with pytest.raises(ManifestValidationError, match="expands to"):
        JobManifestV1.from_dict(data)


def test_exact_5000_operation_target_is_permitted_by_contract():
    data = manifest_data()
    data["operations"][0]["count"] = MAX_TOTAL_OPERATIONS
    manifest = JobManifestV1.from_dict(data)
    assert manifest.operations[0].count == 5_000


@pytest.mark.parametrize(
    "value",
    ["1e999999", MAX_USD + MICRODOLLAR, Decimal(MAX_SIGNED_MICRO_USD + 1)],
)
def test_money_rejects_quantize_failures_and_signed_63_bit_overflow(value):
    with pytest.raises(ManifestValidationError):
        normalize_usd(value, field_name="test")


def test_artifact_dimensions_must_be_paired_and_filename_confined():
    with pytest.raises(ManifestValidationError, match="supplied together"):
        ArtifactSpecV1(width=100)
    with pytest.raises(ManifestValidationError, match="filename"):
        ArtifactSpecV1(filename="../hero.png")


@pytest.mark.parametrize(
    "filename",
    [
        "CON.png",
        "CON .png",
        "prn.txt",
        "AUX.webp",
        "nul.jpeg",
        "COM1.png",
        "com9.jpg",
        "LPT1.png",
        "lpt9.gif",
    ],
)
def test_artifact_filename_rejects_reserved_dos_device_basenames(filename):
    with pytest.raises(ManifestValidationError, match="reserved DOS"):
        ArtifactSpecV1(filename=filename)


@pytest.mark.parametrize(
    "filename",
    [
        " hero.png",
        "hero.png ",
        "hero.png.",
        "hero:final.png",
        "hero?.png",
        "hero*.png",
        "hero|final.png",
        "hero<final>.png",
        'hero"final.png',
        "hero\x01.png",
        "../hero.png",
        "hero/variant.png",
        "hero\\variant.png",
    ],
)
def test_artifact_filename_rejects_portable_unsafe_forms(filename):
    with pytest.raises(ManifestValidationError, match="filename"):
        ArtifactSpecV1(filename=filename)


def test_artifact_filename_binds_supported_image_extension_to_declared_mime():
    assert ArtifactSpecV1(mime_type="image/jpeg", filename="hero.JPG").filename == "hero.JPG"
    assert ArtifactSpecV1(mime_type="image/jpeg", filename="hero.jpeg").filename == "hero.jpeg"
    assert ArtifactSpecV1(mime_type="image/webp", filename="hero.webp").filename == "hero.webp"

    with pytest.raises(ManifestValidationError, match="extension must match"):
        ArtifactSpecV1(mime_type="image/png", filename="hero.jpg")
    with pytest.raises(ManifestValidationError, match="does not match"):
        ArtifactSpecV1(mime_type="application/pdf", filename="report.png")
    with pytest.raises(ManifestValidationError, match="supported raster"):
        ArtifactSpecV1(mime_type="image/tiff", filename="hero.tiff")
    with pytest.raises(ManifestValidationError, match="exceeds"):
        ArtifactSpecV1(filename="a" * MAX_ARTIFACT_FILENAME_BYTES + ".png")


def test_execution_deadline_defaults_are_backward_compatible_and_serialized():
    manifest = JobManifestV1.from_dict(manifest_data())

    assert manifest.execution.call_timeout_s == DEFAULT_CALL_TIMEOUT_S
    assert manifest.execution.max_wall_seconds == DEFAULT_MAX_WALL_SECONDS
    serialized = manifest.to_dict()["execution"]
    assert serialized["call_timeout_s"] == DEFAULT_CALL_TIMEOUT_S
    assert serialized["max_wall_seconds"] == DEFAULT_MAX_WALL_SECONDS

    positional = ExecutionPolicyV1(1, 1, Decimal("0"), "legacy-output")
    assert positional.output_directory == "legacy-output"
    assert positional.call_timeout_s == DEFAULT_CALL_TIMEOUT_S


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("call_timeout_s", 0, "finite and positive"),
        ("call_timeout_s", float("inf"), "finite and positive"),
        ("call_timeout_s", MAX_CALL_TIMEOUT_S + 1, "exceeds"),
        ("max_wall_seconds", -1, "finite and positive"),
        ("max_wall_seconds", MAX_WALL_SECONDS + 1, "exceeds"),
    ],
)
def test_execution_deadline_contract_is_finite_positive_and_capped(
    field,
    value,
    match,
):
    data = manifest_data()
    data["execution"][field] = value
    with pytest.raises(ManifestValidationError, match=match):
        JobManifestV1.from_dict(data)


def test_manifest_schema_is_versioned_and_closed():
    assert JOB_MANIFEST_V1_JSON_SCHEMA["properties"]["version"] == {"const": 1}
    assert JOB_MANIFEST_V1_JSON_SCHEMA["additionalProperties"] is False
    profile_schema = JOB_MANIFEST_V1_JSON_SCHEMA["properties"]["profiles"]["items"]
    assert profile_schema["additionalProperties"] is False
    execution = JOB_MANIFEST_V1_JSON_SCHEMA["properties"]["execution"]["properties"]
    assert execution["call_timeout_s"]["default"] == DEFAULT_CALL_TIMEOUT_S
    assert execution["call_timeout_s"]["maximum"] == MAX_CALL_TIMEOUT_S
    assert execution["max_wall_seconds"]["default"] == DEFAULT_MAX_WALL_SECONDS
    assert execution["max_wall_seconds"]["maximum"] == MAX_WALL_SECONDS
