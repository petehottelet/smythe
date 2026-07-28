"""Provider construction for manifest-declared artifact jobs."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping
from enum import Enum
from typing import Callable

from smythe.jobs.models import ProviderKind
from smythe.jobs.preflight import PlannedOperationV1
from smythe.provider import GeminiProvider, OfflineProvider, OpenAIImageProvider, Provider


class ProviderConfigurationError(ValueError):
    """Raised before dispatch when a profile cannot be constructed safely."""


class ProviderPreflightKind(str, Enum):
    """Stable categories for deterministic failures found before dispatch."""

    LOCAL = "local"
    MODEL = "model"
    KEY = "key"
    SDK = "sdk"


class ProviderPreflightError(ProviderConfigurationError):
    """A categorized provider failure that is safe to record pre-dispatch."""

    def __init__(self, kind: ProviderPreflightKind, message: str) -> None:
        self.kind = kind
        super().__init__(f"{kind.value} provider preflight failed: {message}")


ModuleAvailable = Callable[[str], bool]


def preflight_provider_operation(
    operation: PlannedOperationV1,
    *,
    environ: Mapping[str, str] | None = None,
    module_available: ModuleAvailable | None = None,
) -> None:
    """Validate a provider operation without creating a client or doing I/O.

    The optional dependencies make the check deterministic in tests.  In
    production they default to the trusted process environment and local
    module discovery.  No network request or SDK client construction occurs.
    """

    try:
        _validate_local_operation(operation)
    except ProviderConfigurationError as exc:
        raise ProviderPreflightError(ProviderPreflightKind.LOCAL, str(exc)) from exc

    if operation.provider is ProviderKind.OFFLINE:
        return

    model = operation.model.lower()
    if operation.provider is ProviderKind.OPENAI_IMAGE:
        if not model.startswith("gpt-image-"):
            raise ProviderPreflightError(
                ProviderPreflightKind.MODEL,
                "openai_image requires a gpt-image-* model, "
                f"got {operation.model!r}",
            )
        key_name = "OPENAI_API_KEY"
        sdk_name = "openai"
        install_hint = "pip install smythe[openai]"
    elif operation.provider is ProviderKind.GEMINI_IMAGE:
        if not any(hint in model for hint in ("image", "banana")):
            raise ProviderPreflightError(
                ProviderPreflightKind.MODEL,
                "gemini_image requires an image-output model name containing "
                f"'image' or 'banana', got {operation.model!r}",
            )
        key_name = "GOOGLE_API_KEY"
        sdk_name = "google.genai"
        install_hint = "pip install smythe[gemini]"
    else:  # pragma: no cover - enum and local validation are exhaustive
        raise ProviderPreflightError(
            ProviderPreflightKind.LOCAL,
            f"unsupported provider kind: {operation.provider!r}",
        )

    environment = os.environ if environ is None else environ
    if not str(environment.get(key_name, "")).strip():
        raise ProviderPreflightError(
            ProviderPreflightKind.KEY,
            f"profile {operation.profile_name!r} requires a non-empty "
            f"{key_name} in the trusted process environment",
        )

    checker = _module_available if module_available is None else module_available
    if not checker(sdk_name):
        raise ProviderPreflightError(
            ProviderPreflightKind.SDK,
            f"profile {operation.profile_name!r} requires {sdk_name!r}; "
            f"install it with `{install_hint}`",
        )


class ProviderPool:
    """Cache provider clients by immutable profile configuration.

    Durable Jobs disables SDK-level OpenAI retries: each journaled dispatch
    corresponds to one SDK request attempt. ``CallPermit.idempotency_key`` is a
    local correlation key only and is intentionally not sent as an unsupported
    Image API parameter or header.
    """

    def __init__(
        self,
        *,
        request_timeout_s: float = 300.0,
        max_retries: int = 0,
    ) -> None:
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(float(request_timeout_s))
            or request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be finite and positive")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        self._request_timeout_s = float(request_timeout_s)
        self._max_retries = max_retries
        self._providers: dict[str, Provider] = {}

    def get(self, operation: PlannedOperationV1) -> Provider:
        self.validate_operation(operation)
        key = json.dumps(
            {
                "provider": operation.provider.value,
                "model": operation.model,
                "options": _detached_options(operation),
                "ceiling": str(operation.max_cost_per_call_usd),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        provider = self._providers.get(key)
        if provider is None:
            provider = self._build(operation)
            self._providers[key] = provider
        return provider

    @staticmethod
    def validate_operation(operation: PlannedOperationV1) -> None:
        """Backward-compatible entry point for full provider preflight."""

        preflight_provider_operation(operation)

    @staticmethod
    def preflight(
        operation: PlannedOperationV1,
        *,
        environ: Mapping[str, str] | None = None,
        module_available: ModuleAvailable | None = None,
    ) -> None:
        preflight_provider_operation(
            operation,
            environ=environ,
            module_available=module_available,
        )

    def _build(self, operation: PlannedOperationV1) -> Provider:
        options = _detached_options(operation)
        if operation.provider is ProviderKind.OFFLINE:
            allowed = {"artifacts_per_call", "echo_prefix"}
            _reject_unknown(options, allowed, operation.profile_name)
            return OfflineProvider(
                artifacts_per_call=_positive_int(
                    options.get("artifacts_per_call", 1),
                    "artifacts_per_call",
                ),
                echo_prefix=_string(options.get("echo_prefix", "offline: "), "echo_prefix"),
            )

        ceiling = float(operation.max_cost_per_call_usd)
        if operation.provider is ProviderKind.OPENAI_IMAGE:
            allowed = {
                "size",
                "quality",
                "output_format",
                "output_compression",
                "moderation",
                "n",
            }
            if "base_url" in options:
                raise ProviderConfigurationError(
                    "base_url is forbidden in job manifests; an untrusted "
                    "manifest must not redirect an API-key-authenticated request. "
                    "A trusted operator may configure OPENAI_BASE_URL in the "
                    "process environment instead."
                )
            _reject_unknown(options, allowed, operation.profile_name)
            return OpenAIImageProvider(
                size=_string(options.get("size", "auto"), "size"),
                quality=_string(options.get("quality", "medium"), "quality"),
                output_format=_string(
                    options.get("output_format", "png"), "output_format"
                ),
                output_compression=_optional_int(
                    options.get("output_compression"), "output_compression"
                ),
                moderation=_string(options.get("moderation", "auto"), "moderation"),
                n=_positive_int(options.get("n", 1), "n"),
                max_cost_per_call_usd=ceiling,
                request_timeout_s=self._request_timeout_s,
                max_retries=self._max_retries,
            )

        if operation.provider is ProviderKind.GEMINI_IMAGE:
            allowed = {"response_modalities", "image_config"}
            _reject_unknown(options, allowed, operation.profile_name)
            response_modalities = options.get("response_modalities", ["TEXT", "IMAGE"])
            if not isinstance(response_modalities, list) or not all(
                isinstance(item, str) for item in response_modalities
            ):
                raise ProviderConfigurationError(
                    "response_modalities must be an array of strings"
                )
            image_config = options.get("image_config")
            if image_config is not None and not isinstance(image_config, Mapping):
                raise ProviderConfigurationError("image_config must be an object")
            return GeminiProvider(
                response_modalities=list(response_modalities),
                image_config=dict(image_config) if image_config is not None else None,
                max_cost_per_call_usd=ceiling,
            )

        raise ProviderConfigurationError(
            f"unsupported provider kind: {operation.provider!r}"
        )


def _reject_unknown(options: dict, allowed: set[str], profile: str) -> None:
    unknown = set(options) - allowed
    if unknown:
        raise ProviderConfigurationError(
            f"profile {profile!r} has unsupported options: {sorted(unknown)}"
        )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ProviderConfigurationError(f"{field} must be a string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderConfigurationError(f"{field} must be a positive integer")
    return value


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderConfigurationError(f"{field} must be an integer")
    return value


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _validate_local_operation(operation: PlannedOperationV1) -> None:
    """Validate deterministic profile/adapter contracts without SDK access."""

    options = _detached_options(operation)
    if operation.provider is ProviderKind.OFFLINE:
        _reject_unknown(
            options,
            {"artifacts_per_call", "echo_prefix"},
            operation.profile_name,
        )
        _positive_int(options.get("artifacts_per_call", 1), "artifacts_per_call")
        _string(options.get("echo_prefix", "offline: "), "echo_prefix")
        return

    if operation.provider is ProviderKind.OPENAI_IMAGE:
        if operation.attachments:
            raise ProviderConfigurationError(
                "OpenAI image attachments are not yet supported by Smythe's "
                "generation adapter; use Gemini image or remove attachments. "
                "Reference-image jobs must not silently discard their inputs."
            )
        if "base_url" in options:
            raise ProviderConfigurationError(
                "base_url is forbidden in job manifests; an untrusted manifest "
                "must not redirect an API-key-authenticated request"
            )
        allowed = {
            "size",
            "quality",
            "output_format",
            "output_compression",
            "moderation",
            "n",
        }
        _reject_unknown(options, allowed, operation.profile_name)
        size = _string(options.get("size", "auto"), "size")
        valid_sizes = {"auto", "1024x1024", "1536x1024", "1024x1536"}
        if size not in valid_sizes:
            raise ProviderConfigurationError(
                f"size must be one of {sorted(valid_sizes)}, got {size!r}"
            )
        quality = _string(options.get("quality", "medium"), "quality")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ProviderConfigurationError(f"unsupported quality: {quality!r}")
        output_format = _string(
            options.get("output_format", "png"), "output_format"
        )
        mime_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        if output_format not in mime_types:
            raise ProviderConfigurationError(
                f"unsupported output_format: {output_format!r}"
            )
        compression = _optional_int(
            options.get("output_compression"), "output_compression"
        )
        if compression is not None and not 0 <= compression <= 100:
            raise ProviderConfigurationError(
                "output_compression must be between 0 and 100"
            )
        if compression is not None and output_format == "png":
            raise ProviderConfigurationError(
                "output_compression is supported only for JPEG or WebP"
            )
        moderation = _string(options.get("moderation", "auto"), "moderation")
        if moderation not in {"auto", "low"}:
            raise ProviderConfigurationError(
                f"unsupported moderation: {moderation!r}"
            )
        if _positive_int(options.get("n", 1), "n") > 10:
            raise ProviderConfigurationError("n must be between 1 and 10")
        if operation.artifact.mime_type != mime_types[output_format]:
            raise ProviderConfigurationError(
                f"artifact.mime_type {operation.artifact.mime_type!r} does not "
                f"match output_format {output_format!r}"
            )
        if size != "auto" and operation.artifact.width is not None:
            width, height = (int(item) for item in size.split("x"))
            if (operation.artifact.width, operation.artifact.height) != (width, height):
                raise ProviderConfigurationError(
                    "artifact dimensions do not match the requested OpenAI size"
                )
        return

    if operation.provider is ProviderKind.GEMINI_IMAGE:
        _reject_unknown(
            options,
            {"response_modalities", "image_config"},
            operation.profile_name,
        )
        modalities = options.get("response_modalities", ["TEXT", "IMAGE"])
        if not isinstance(modalities, list) or not all(
            isinstance(item, str) for item in modalities
        ):
            raise ProviderConfigurationError(
                "response_modalities must be an array of strings"
            )
        if "IMAGE" not in {item.upper() for item in modalities}:
            raise ProviderConfigurationError(
                "gemini_image response_modalities must include IMAGE"
            )
        image_config = options.get("image_config")
        if image_config is not None and not isinstance(image_config, Mapping):
            raise ProviderConfigurationError("image_config must be an object")
        return

    raise ProviderConfigurationError(
        f"unsupported provider kind: {operation.provider!r}"
    )


def _detached_options(operation: PlannedOperationV1) -> dict:
    """Return ordinary JSON containers from an immutable approved operation."""

    options = operation.to_dict()["options"]
    if not isinstance(options, dict):  # pragma: no cover - plan contract enforces it
        raise ProviderConfigurationError("approved provider options must be an object")
    return options
