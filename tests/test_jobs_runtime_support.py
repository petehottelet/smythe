"""Tests for Jobs v1 manifest loading, provider profiles, and artifact I/O."""

from __future__ import annotations

import hashlib
import io
import sys
from types import SimpleNamespace

import pytest

from smythe.jobs.artifact_io import atomic_write_bytes, inspect_artifact
from smythe.jobs.loading import load_manifest
from smythe.jobs.models import ManifestValidationError
from smythe.jobs.preflight import preflight_job
from smythe.jobs.providers import (
    ProviderConfigurationError,
    ProviderPool,
    ProviderPreflightError,
    ProviderPreflightKind,
    preflight_provider_operation,
)
from smythe.provider import GeminiProvider, OfflineProvider, OpenAIImageProvider

Image = pytest.importorskip("PIL.Image")


def _png_bytes(size: tuple[int, int] = (320, 180)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _manifest(*, provider: str = "offline", attachments: list[str] | None = None):
    return {
        "version": 1,
        "name": "runtime-support",
        "profiles": [
            {
                "name": "default",
                "provider": provider,
                "model": "offline-image" if provider == "offline" else "gpt-image-1",
                "max_cost_per_call_usd": "0" if provider == "offline" else "0.25",
                "options": {"artifacts_per_call": 1} if provider == "offline" else {},
            }
        ],
        "operations": [
            {
                "key": "hero",
                "prompt": "Generate a hero image",
                "profile": "default",
                "attachments": attachments or [],
            }
        ],
        "execution": {
            "max_concurrency": 2,
            "max_attempts": 1,
            "max_budget_usd": "0" if provider == "offline" else "0.25",
        },
    }


def test_load_yaml_manifest_returns_source_root(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(
        """\
version: 1
name: yaml-job
profiles:
  - name: default
    provider: offline
    model: offline-image
    max_cost_per_call_usd: "0"
operations:
  - key: image
    prompt: Generate an image
    profile: default
execution:
  max_concurrency: 1
  max_attempts: 1
  max_budget_usd: "0"
""",
        encoding="utf-8",
    )

    manifest, root = load_manifest(path)

    assert manifest.name == "yaml-job"
    assert root == tmp_path.resolve()


def test_load_manifest_rejects_empty_document(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="empty"):
        load_manifest(path)


def test_provider_pool_reuses_offline_profile(tmp_path):
    path = tmp_path / "job.json"
    import json

    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    manifest, root = load_manifest(path)
    operation = preflight_job(manifest, manifest_root=root).operations[0]
    pool = ProviderPool()

    first = pool.get(operation)
    second = pool.get(operation)

    assert isinstance(first, OfflineProvider)
    assert second is first


def test_openai_reference_inputs_fail_before_provider_construction(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"logo")
    path = tmp_path / "job.json"
    import json

    path.write_text(
        json.dumps(_manifest(provider="openai_image", attachments=["logo.png"])),
        encoding="utf-8",
    )
    manifest, root = load_manifest(path)
    operation = preflight_job(manifest, manifest_root=root).operations[0]

    with pytest.raises(ProviderConfigurationError, match="attachments"):
        ProviderPool().get(operation)


def _provider_operation(tmp_path, *, provider, model, options=None, artifact=None,
                        attachments=None):
    data = _manifest(provider=provider, attachments=attachments)
    data["profiles"][0]["model"] = model
    data["profiles"][0]["options"] = options or {}
    if artifact is not None:
        data["operations"][0]["artifact"] = artifact
    manifest = load_manifest(_write_json(tmp_path / "provider.json", data))[0]
    return preflight_job(manifest, manifest_root=tmp_path).operations[0]


def _write_json(path, value):
    import json

    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_job_manifest_cannot_redirect_openai_credentials(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-image-1",
        options={"base_url": "https://attacker.invalid/v1"},
    )

    with pytest.raises(ProviderPreflightError, match="base_url") as caught:
        preflight_provider_operation(
            operation,
            environ={"OPENAI_API_KEY": "secret"},
            module_available=lambda _name: True,
        )

    assert caught.value.kind is ProviderPreflightKind.LOCAL


def test_provider_preflight_distinguishes_model_error_before_key(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-4o",
    )

    with pytest.raises(ProviderPreflightError) as caught:
        preflight_provider_operation(
            operation,
            environ={},
            module_available=lambda _name: True,
        )

    assert caught.value.kind is ProviderPreflightKind.MODEL


def test_provider_preflight_distinguishes_missing_key(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-image-1",
    )

    with pytest.raises(ProviderPreflightError) as caught:
        preflight_provider_operation(
            operation,
            environ={},
            module_available=lambda _name: True,
        )

    assert caught.value.kind is ProviderPreflightKind.KEY


def test_provider_preflight_distinguishes_missing_sdk(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-image-1",
    )

    with pytest.raises(ProviderPreflightError) as caught:
        preflight_provider_operation(
            operation,
            environ={"OPENAI_API_KEY": "secret"},
            module_available=lambda _name: False,
        )

    assert caught.value.kind is ProviderPreflightKind.SDK


def test_provider_preflight_accepts_complete_local_openai_setup(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-image-1",
        options={"size": "1024x1024", "quality": "low"},
    )

    preflight_provider_operation(
        operation,
        environ={"OPENAI_API_KEY": "secret"},
        module_available=lambda name: name == "openai",
    )


def test_durable_openai_pool_disables_sdk_retries_and_sets_timeout(
    tmp_path, monkeypatch
):
    operation = _provider_operation(
        tmp_path,
        provider="openai_image",
        model="gpt-image-2",
    )
    captured = {}
    client = object()

    def make_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=make_client),
    )
    provider = ProviderPool(request_timeout_s=12.5, max_retries=0).get(operation)

    assert isinstance(provider, OpenAIImageProvider)
    assert provider._get_client() is client
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, True])
def test_image_provider_cost_hints_must_be_finite_nonnegative_numbers(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        OpenAIImageProvider(max_cost_per_call_usd=value)
    with pytest.raises(ValueError, match="finite and non-negative"):
        GeminiProvider(max_cost_per_call_usd=value)


def test_gemini_image_profile_requires_image_modality(tmp_path):
    operation = _provider_operation(
        tmp_path,
        provider="gemini_image",
        model="gemini-3-pro-image-preview",
        options={"response_modalities": ["TEXT"]},
    )

    with pytest.raises(ProviderPreflightError, match="include IMAGE") as caught:
        preflight_provider_operation(
            operation,
            environ={"GOOGLE_API_KEY": "secret"},
            module_available=lambda _name: True,
        )

    assert caught.value.kind is ProviderPreflightKind.LOCAL


@pytest.mark.parametrize("mime_type", ["application/pdf", "application/octet-stream", "text/plain"])
def test_gemini_image_profile_rejects_an_artifact_it_cannot_return(tmp_path, mime_type):
    operation = _provider_operation(
        tmp_path,
        provider="gemini_image",
        model="gemini-3.1-flash-image",
        artifact={"mime_type": mime_type},
    )

    with pytest.raises(ProviderPreflightError, match="mime_type") as caught:
        preflight_provider_operation(
            operation,
            environ={"GOOGLE_API_KEY": "secret"},
            module_available=lambda _name: True,
        )

    assert caught.value.kind is ProviderPreflightKind.LOCAL


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/webp", "image/gif"])
def test_gemini_image_profile_accepts_raster_artifacts_and_reference_images(tmp_path, mime_type):
    (tmp_path / "logo.png").write_bytes(_png_bytes((4, 4)))
    operation = _provider_operation(
        tmp_path,
        provider="gemini_image",
        model="gemini-3.1-flash-image",
        artifact={"mime_type": mime_type, "width": 1024, "height": 1024},
        attachments=["logo.png"],
    )

    preflight_provider_operation(
        operation,
        environ={"GOOGLE_API_KEY": "secret"},
        module_available=lambda name: name == "google.genai",
    )


@pytest.mark.parametrize("artifact", [
    {"mime_type": "image/jpeg"},
    {"mime_type": "image/webp", "width": 1, "height": 1},
    {"mime_type": "application/octet-stream"},
    {"mime_type": "image/png", "width": 1024, "height": 1024},
    {"mime_type": "image/png", "width": 1, "height": 2},
], ids=["jpeg", "webp", "binary", "1024x1024", "1x2"])
def test_offline_profile_rejects_an_artifact_other_than_a_1x1_png(tmp_path, artifact):
    operation = _provider_operation(
        tmp_path, provider="offline", model="offline-image", artifact=artifact,
    )

    with pytest.raises(ProviderPreflightError, match="offline profile") as caught:
        preflight_provider_operation(operation, environ={}, module_available=lambda _name: False)

    assert caught.value.kind is ProviderPreflightKind.LOCAL


@pytest.mark.parametrize("artifact", [
    None, {"mime_type": "image/png"}, {"mime_type": "image/png", "width": 1, "height": 1},
])
def test_offline_profile_accepts_a_png_with_no_or_1x1_dimensions(tmp_path, artifact):
    operation = _provider_operation(
        tmp_path, provider="offline", model="offline-image", artifact=artifact,
    )

    preflight_provider_operation(operation, environ={}, module_available=lambda _name: False)
    assert isinstance(ProviderPool().get(operation), OfflineProvider)


def test_inspect_png_uses_content_not_declared_mime():
    data = _png_bytes()

    observed = inspect_artifact(data, "application/octet-stream")

    assert observed.mime_type == "image/png"
    assert (observed.width, observed.height) == (320, 180)
    assert observed.size_bytes == len(data)
    assert observed.sha256 == hashlib.sha256(data).hexdigest()


def test_atomic_write_replaces_complete_file(tmp_path):
    destination = tmp_path / "run" / "artifact.bin"
    atomic_write_bytes(destination, b"first")
    atomic_write_bytes(destination, b"replacement")

    assert destination.read_bytes() == b"replacement"
    assert list(destination.parent.glob("*.tmp")) == []
