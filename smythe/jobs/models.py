"""Versioned, JSON-compatible contracts for durable artifact jobs."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

MANIFEST_VERSION = 1
MICRODOLLAR = Decimal("0.000001")

# Version-1 implementation limits are deliberately finite and leave enough room
# for the product's 5,000-operation benchmark target.  They are contract limits,
# not provider recommendations; callers must split larger campaigns into jobs.
MAX_SIGNED_MICRO_USD = (1 << 63) - 1
MAX_USD = Decimal(MAX_SIGNED_MICRO_USD) * MICRODOLLAR
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_CANONICAL_PLAN_BYTES = 64 * 1024 * 1024
MAX_PROFILES = 64
MAX_OPERATION_TEMPLATES = 512
MAX_TOTAL_OPERATIONS = 5_000
MAX_OPERATION_COUNT = MAX_TOTAL_OPERATIONS
MAX_CONCURRENCY = 256
MAX_ATTEMPTS = 10
MAX_PROMPT_BYTES = 64 * 1024
MAX_OPTIONS_JSON_BYTES = 64 * 1024
MAX_ATTACHMENTS_PER_OPERATION = 16
MAX_UNIQUE_ATTACHMENTS = 128
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_DIMENSION = 16_384
MAX_ARTIFACT_PIXELS = 64 * 1024 * 1024
MAX_ARTIFACT_FILENAME_BYTES = 255
DEFAULT_CALL_TIMEOUT_S = 300.0
MAX_CALL_TIMEOUT_S = 60 * 60.0
DEFAULT_MAX_WALL_SECONDS = 6 * 60 * 60.0
MAX_WALL_SECONDS = 7 * 24 * 60 * 60.0

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FrozenJSONValue: TypeAlias = (
    JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]
)

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_MIME_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_PORTABLE_FORBIDDEN_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_DOS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_IMAGE_EXTENSIONS_BY_MIME = {
    "image/gif": frozenset({".gif"}),
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/png": frozenset({".png"}),
    "image/webp": frozenset({".webp"}),
}
_KNOWN_IMAGE_EXTENSIONS = frozenset(
    extension for extensions in _IMAGE_EXTENSIONS_BY_MIME.values() for extension in extensions
)
_SECRET_KEYS = {
    "accesstoken",
    "apikey",
    "authorization",
    "bearertoken",
    "clientsecret",
    "credential",
    "credentials",
    "password",
    "privatekey",
    "secret",
    "secretkey",
    "token",
}


class ManifestValidationError(ValueError):
    """Raised when a version-1 job manifest violates its contract."""


class ProviderKind(str, Enum):
    OFFLINE = "offline"
    OPENAI_IMAGE = "openai_image"
    GEMINI_IMAGE = "gemini_image"


def normalize_usd(value: object, *, field_name: str) -> Decimal:
    """Return a non-negative amount with exact microdollar precision."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ManifestValidationError(f"{field_name} must be a USD number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ManifestValidationError(f"{field_name} must be finite")
    try:
        amount = Decimal(str(value))
    except (DecimalException, ValueError) as exc:
        raise ManifestValidationError(f"{field_name} must be a USD number") from exc
    if not amount.is_finite():
        raise ManifestValidationError(f"{field_name} must be finite")
    if amount < 0:
        raise ManifestValidationError(f"{field_name} must be non-negative")
    try:
        quantized = amount.quantize(MICRODOLLAR)
    except (DecimalException, ValueError) as exc:
        raise ManifestValidationError(f"{field_name} must fit exact microdollar precision") from exc
    if quantized != amount:
        raise ManifestValidationError(f"{field_name} supports at most six decimal places")
    if quantized > MAX_USD:
        raise ManifestValidationError(f"{field_name} exceeds signed 63-bit microdollar capacity")
    return quantized


def usd_to_micros(value: Decimal) -> int:
    """Convert a normalized USD Decimal to integer microdollars."""
    normalized = normalize_usd(value, field_name="USD amount")
    micros = int(normalized / MICRODOLLAR)
    if micros > MAX_SIGNED_MICRO_USD:  # pragma: no cover - normalize guard
        raise ManifestValidationError("USD amount exceeds signed 63-bit capacity")
    return micros


def micros_to_usd(value: int) -> Decimal:
    """Convert integer microdollars to a normalized USD Decimal."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SIGNED_MICRO_USD
    ):
        raise ManifestValidationError(
            "microdollar amount must be a non-negative signed 63-bit integer"
        )
    return (Decimal(value) * MICRODOLLAR).quantize(MICRODOLLAR)


def usd_string(value: Decimal) -> str:
    return format(value.quantize(MICRODOLLAR), "f")


def _strict_fields(
    data: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise ManifestValidationError(f"{context} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ManifestValidationError(f"{context} is missing required fields: {sorted(missing)}")


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ManifestValidationError(f"{context} must be an object")
    return value


def _require_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ManifestValidationError(f"{context} must match {_NAME_RE.pattern!r}")
    return value


def _json_copy(value: object, *, context: str, depth: int = 0) -> JSONValue:
    if depth > 32:
        raise ManifestValidationError(f"{context} is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _json_copy(item, context=f"{context}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            # Environment-variable names are references, not credential values.
            if not normalized.endswith("env") and (
                normalized in _SECRET_KEYS
                or normalized.endswith("apikey")
                or normalized.endswith("privatekey")
                or normalized.endswith("secretkey")
                or normalized.endswith("password")
            ):
                raise ManifestValidationError(
                    f"{context}.{key} may contain a secret; credentials must "
                    "come from the environment, never a manifest"
                )
            copied[key] = _json_copy(item, context=f"{context}.{key}", depth=depth + 1)
        return copied
    raise ManifestValidationError(f"{context} must contain only JSON values")


def _freeze_copied_json(value: JSONValue) -> FrozenJSONValue:
    if isinstance(value, list):
        return tuple(_freeze_copied_json(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_copied_json(item) for key, item in value.items()})
    return value


def freeze_json(value: object, *, context: str) -> FrozenJSONValue:
    """Deep-copy and freeze a JSON value for use in an approved identity."""
    return _freeze_copied_json(_json_copy(value, context=context))


def detach_json(value: object, *, context: str) -> JSONValue:
    """Return a mutable deep copy suitable for an external ``to_dict`` result."""
    return _json_copy(value, context=context)


def _json_size_bytes(value: JSONValue) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ManifestValidationError("JSON value is not valid UTF-8 JSON") from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"manifest JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestValidationError(f"manifest JSON contains non-finite numeric constant {value!r}")


def strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys at every object depth."""
    return json.loads(
        text,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


def _normalize_timeout(
    value: object,
    *,
    field_name: str,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, Decimal))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ManifestValidationError(f"{field_name} must be finite and positive")
    normalized = float(value)
    if normalized > maximum:
        raise ManifestValidationError(f"{field_name} exceeds {maximum:g} seconds")
    return normalized


def _validate_artifact_filename(filename: object, mime_type: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ManifestValidationError("artifact.filename must be a single portable filename")
    if filename != filename.strip() or filename.endswith((".", " ")):
        raise ManifestValidationError(
            "artifact.filename may not have leading/trailing whitespace or a trailing dot"
        )
    if filename in {".", ".."}:
        raise ManifestValidationError("artifact.filename must be a single portable filename")
    if any(
        character in _PORTABLE_FORBIDDEN_FILENAME_CHARS
        or unicodedata.category(character).startswith("C")
        for character in filename
    ):
        raise ManifestValidationError("artifact.filename contains a portable-unsafe character")
    if unicodedata.normalize("NFC", filename) != filename:
        raise ManifestValidationError("artifact.filename must use NFC Unicode normalization")
    if len(filename.encode("utf-8")) > MAX_ARTIFACT_FILENAME_BYTES:
        raise ManifestValidationError(
            f"artifact.filename exceeds {MAX_ARTIFACT_FILENAME_BYTES} UTF-8 bytes"
        )
    device_basename = filename.split(".", 1)[0].rstrip(" ").upper()
    if device_basename in _DOS_RESERVED_BASENAMES:
        raise ManifestValidationError(
            f"artifact.filename uses reserved DOS device name {device_basename!r}"
        )

    suffix = ("." + filename.rsplit(".", 1)[1].lower()) if "." in filename else ""
    expected = _IMAGE_EXTENSIONS_BY_MIME.get(mime_type)
    if expected is not None and suffix not in expected:
        allowed = ", ".join(sorted(expected))
        raise ManifestValidationError(
            f"artifact.filename extension must match {mime_type} ({allowed})"
        )
    if expected is None and suffix in _KNOWN_IMAGE_EXTENSIONS:
        raise ManifestValidationError(
            f"artifact.filename extension {suffix!r} does not match {mime_type}"
        )
    return filename


@dataclass(frozen=True, slots=True)
class ProviderProfileV1:
    name: str
    provider: ProviderKind
    model: str
    max_cost_per_call_usd: Decimal
    options: Mapping[str, FrozenJSONValue | JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "profile.name"))
        if isinstance(self.provider, str):
            try:
                object.__setattr__(self, "provider", ProviderKind(self.provider))
            except ValueError as exc:
                raise ManifestValidationError(
                    f"profile.provider is unsupported: {self.provider!r}"
                ) from exc
        if not isinstance(self.provider, ProviderKind):
            raise ManifestValidationError("profile.provider is unsupported")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ManifestValidationError("profile.model must be a non-empty string")
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(
            self,
            "max_cost_per_call_usd",
            normalize_usd(
                self.max_cost_per_call_usd,
                field_name=f"profiles.{self.name}.max_cost_per_call_usd",
            ),
        )
        copied = _json_copy(self.options, context=f"profiles.{self.name}.options")
        if not isinstance(copied, dict):
            raise ManifestValidationError("profile.options must be an object")
        if _json_size_bytes(copied) > MAX_OPTIONS_JSON_BYTES:
            raise ManifestValidationError(
                f"profiles.{self.name}.options exceeds "
                f"{MAX_OPTIONS_JSON_BYTES} UTF-8 JSON bytes"
            )
        object.__setattr__(self, "options", _freeze_copied_json(copied))

    @classmethod
    def from_dict(cls, data: object) -> "ProviderProfileV1":
        item = _require_mapping(data, "profile")
        _strict_fields(
            item,
            allowed={"name", "provider", "model", "max_cost_per_call_usd", "options"},
            required={"name", "provider", "model", "max_cost_per_call_usd"},
            context="profile",
        )
        return cls(
            name=item["name"],
            provider=item["provider"],
            model=item["model"],
            max_cost_per_call_usd=item["max_cost_per_call_usd"],
            options=item.get("options", {}),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "provider": self.provider.value,
            "model": self.model,
            "max_cost_per_call_usd": usd_string(self.max_cost_per_call_usd),
            "options": detach_json(self.options, context="profile.options"),
        }


@dataclass(frozen=True, slots=True)
class ArtifactSpecV1:
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    filename: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mime_type, str):
            raise ManifestValidationError("artifact.mime_type must be a MIME type")
        mime_type = self.mime_type.strip().lower()
        if not _MIME_RE.fullmatch(mime_type):
            raise ManifestValidationError("artifact.mime_type must be a MIME type")
        if mime_type.startswith("image/") and mime_type not in _IMAGE_EXTENSIONS_BY_MIME:
            raise ManifestValidationError(
                f"artifact.mime_type {mime_type!r} is not a supported raster image MIME"
            )
        object.__setattr__(self, "mime_type", mime_type)
        for field_name, value in (("width", self.width), ("height", self.height)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ManifestValidationError(f"artifact.{field_name} must be a positive integer")
            if value is not None and value > MAX_ARTIFACT_DIMENSION:
                raise ManifestValidationError(
                    f"artifact.{field_name} exceeds {MAX_ARTIFACT_DIMENSION}"
                )
        if (self.width is None) != (self.height is None):
            raise ManifestValidationError(
                "artifact.width and artifact.height must be supplied together"
            )
        if (
            self.width is not None
            and self.height is not None
            and self.width * self.height > MAX_ARTIFACT_PIXELS
        ):
            raise ManifestValidationError(
                f"artifact dimensions exceed {MAX_ARTIFACT_PIXELS} pixels"
            )
        if self.filename is not None:
            object.__setattr__(
                self,
                "filename",
                _validate_artifact_filename(self.filename, mime_type),
            )

    @classmethod
    def from_dict(cls, data: object) -> "ArtifactSpecV1":
        item = _require_mapping(data, "artifact")
        _strict_fields(
            item,
            allowed={"mime_type", "width", "height", "filename"},
            required=set(),
            context="artifact",
        )
        return cls(
            mime_type=item.get("mime_type", "image/png"),
            width=item.get("width"),
            height=item.get("height"),
            filename=item.get("filename"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {"mime_type": self.mime_type}
        if self.width is not None:
            result["width"] = self.width
            result["height"] = self.height
        if self.filename is not None:
            result["filename"] = self.filename
        return result


@dataclass(frozen=True, slots=True)
class OperationTemplateV1:
    key: str
    count: int
    prompt: str
    profile: str
    attachments: tuple[str, ...] = ()
    artifact: ArtifactSpecV1 = field(default_factory=ArtifactSpecV1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _require_name(self.key, "operation.key"))
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ManifestValidationError("operation.count must be a positive integer")
        if self.count > MAX_OPERATION_COUNT:
            raise ManifestValidationError(f"operation.count exceeds {MAX_OPERATION_COUNT}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ManifestValidationError("operation.prompt must be a non-empty string")
        object.__setattr__(self, "prompt", self.prompt.strip())
        try:
            prompt_bytes = len(self.prompt.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ManifestValidationError("operation.prompt must contain valid UTF-8 text") from exc
        if prompt_bytes > MAX_PROMPT_BYTES:
            raise ManifestValidationError(
                f"operation.prompt exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
            )
        object.__setattr__(self, "profile", _require_name(self.profile, "operation.profile"))
        if isinstance(self.attachments, (str, bytes)):
            raise ManifestValidationError("operation.attachments must be an array")
        paths = tuple(self.attachments)
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ManifestValidationError(
                "operation.attachments must contain non-empty path strings"
            )
        if len(set(paths)) != len(paths):
            raise ManifestValidationError("operation.attachments contains duplicates")
        if len(paths) > MAX_ATTACHMENTS_PER_OPERATION:
            raise ManifestValidationError(
                "operation.attachments exceeds " f"{MAX_ATTACHMENTS_PER_OPERATION} entries"
            )
        object.__setattr__(self, "attachments", paths)
        if not isinstance(self.artifact, ArtifactSpecV1):
            raise ManifestValidationError("operation.artifact must be an ArtifactSpecV1")

    @classmethod
    def from_dict(cls, data: object) -> "OperationTemplateV1":
        item = _require_mapping(data, "operation")
        _strict_fields(
            item,
            allowed={"key", "count", "prompt", "profile", "attachments", "artifact"},
            required={"key", "prompt", "profile"},
            context="operation",
        )
        attachments = item.get("attachments", [])
        if not isinstance(attachments, list):
            raise ManifestValidationError("operation.attachments must be an array")
        return cls(
            key=item["key"],
            count=item.get("count", 1),
            prompt=item["prompt"],
            profile=item["profile"],
            attachments=tuple(attachments),
            artifact=ArtifactSpecV1.from_dict(item.get("artifact", {})),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "key": self.key,
            "count": self.count,
            "prompt": self.prompt,
            "profile": self.profile,
            "attachments": list(self.attachments),
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicyV1:
    max_concurrency: int
    max_attempts: int
    max_budget_usd: Decimal
    output_directory: str = "smythe_artifacts/jobs"
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_concurrency", self.max_concurrency),
            ("max_attempts", self.max_attempts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ManifestValidationError(f"execution.{field_name} must be a positive integer")
        if self.max_concurrency > MAX_CONCURRENCY:
            raise ManifestValidationError(f"execution.max_concurrency exceeds {MAX_CONCURRENCY}")
        if self.max_attempts > MAX_ATTEMPTS:
            raise ManifestValidationError(f"execution.max_attempts exceeds {MAX_ATTEMPTS}")
        object.__setattr__(
            self,
            "max_budget_usd",
            normalize_usd(self.max_budget_usd, field_name="execution.max_budget_usd"),
        )
        object.__setattr__(
            self,
            "call_timeout_s",
            _normalize_timeout(
                self.call_timeout_s,
                field_name="execution.call_timeout_s",
                maximum=MAX_CALL_TIMEOUT_S,
            ),
        )
        object.__setattr__(
            self,
            "max_wall_seconds",
            _normalize_timeout(
                self.max_wall_seconds,
                field_name="execution.max_wall_seconds",
                maximum=MAX_WALL_SECONDS,
            ),
        )
        if not isinstance(self.output_directory, str) or not self.output_directory.strip():
            raise ManifestValidationError(
                "execution.output_directory must be a non-empty relative path"
            )
        object.__setattr__(self, "output_directory", self.output_directory.strip())

    @classmethod
    def from_dict(cls, data: object) -> "ExecutionPolicyV1":
        item = _require_mapping(data, "execution")
        _strict_fields(
            item,
            allowed={
                "max_concurrency",
                "max_attempts",
                "max_budget_usd",
                "call_timeout_s",
                "max_wall_seconds",
                "output_directory",
            },
            required={"max_concurrency", "max_attempts", "max_budget_usd"},
            context="execution",
        )
        return cls(
            max_concurrency=item["max_concurrency"],
            max_attempts=item["max_attempts"],
            max_budget_usd=item["max_budget_usd"],
            call_timeout_s=item.get("call_timeout_s", DEFAULT_CALL_TIMEOUT_S),
            max_wall_seconds=item.get("max_wall_seconds", DEFAULT_MAX_WALL_SECONDS),
            output_directory=item.get("output_directory", "smythe_artifacts/jobs"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "max_concurrency": self.max_concurrency,
            "max_attempts": self.max_attempts,
            "max_budget_usd": usd_string(self.max_budget_usd),
            "call_timeout_s": self.call_timeout_s,
            "max_wall_seconds": self.max_wall_seconds,
            "output_directory": self.output_directory,
        }


@dataclass(frozen=True, slots=True)
class JobManifestV1:
    name: str
    profiles: tuple[ProviderProfileV1, ...]
    operations: tuple[OperationTemplateV1, ...]
    execution: ExecutionPolicyV1
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version != MANIFEST_VERSION
        ):
            raise ManifestValidationError(
                f"manifest.version {self.version!r} is unsupported; expected 1"
            )
        object.__setattr__(self, "name", _require_name(self.name, "manifest.name"))
        object.__setattr__(self, "profiles", tuple(self.profiles))
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.profiles:
            raise ManifestValidationError("manifest.profiles must not be empty")
        if not self.operations:
            raise ManifestValidationError("manifest.operations must not be empty")
        if len(self.profiles) > MAX_PROFILES:
            raise ManifestValidationError(f"manifest.profiles exceeds {MAX_PROFILES} entries")
        if len(self.operations) > MAX_OPERATION_TEMPLATES:
            raise ManifestValidationError(
                f"manifest.operations exceeds {MAX_OPERATION_TEMPLATES} templates"
            )
        if any(not isinstance(profile, ProviderProfileV1) for profile in self.profiles):
            raise ManifestValidationError("manifest.profiles contains an invalid profile")
        if any(not isinstance(operation, OperationTemplateV1) for operation in self.operations):
            raise ManifestValidationError("manifest.operations contains an invalid operation")
        profile_names = [profile.name for profile in self.profiles]
        operation_keys = [operation.key for operation in self.operations]
        if len(set(profile_names)) != len(profile_names):
            raise ManifestValidationError("manifest has duplicate profile names")
        if len(set(operation_keys)) != len(operation_keys):
            raise ManifestValidationError("manifest has duplicate operation keys")
        total_operations = sum(operation.count for operation in self.operations)
        if total_operations > MAX_TOTAL_OPERATIONS:
            raise ManifestValidationError(
                "manifest expands to "
                f"{total_operations} operations; maximum is {MAX_TOTAL_OPERATIONS}"
            )
        if not isinstance(self.execution, ExecutionPolicyV1):
            raise ManifestValidationError("manifest.execution is invalid")

    @classmethod
    def from_dict(cls, data: object) -> "JobManifestV1":
        item = _require_mapping(data, "manifest")
        _strict_fields(
            item,
            allowed={"version", "name", "profiles", "operations", "execution"},
            required={"version", "name", "profiles", "operations", "execution"},
            context="manifest",
        )
        if not isinstance(item["profiles"], list):
            raise ManifestValidationError("manifest.profiles must be an array")
        if not isinstance(item["operations"], list):
            raise ManifestValidationError("manifest.operations must be an array")
        if len(item["profiles"]) > MAX_PROFILES:
            raise ManifestValidationError(f"manifest.profiles exceeds {MAX_PROFILES} entries")
        if len(item["operations"]) > MAX_OPERATION_TEMPLATES:
            raise ManifestValidationError(
                f"manifest.operations exceeds {MAX_OPERATION_TEMPLATES} templates"
            )
        return cls(
            version=item["version"],
            name=item["name"],
            profiles=tuple(ProviderProfileV1.from_dict(v) for v in item["profiles"]),
            operations=tuple(OperationTemplateV1.from_dict(v) for v in item["operations"]),
            execution=ExecutionPolicyV1.from_dict(item["execution"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "JobManifestV1":
        if not isinstance(text, str):
            raise ManifestValidationError("manifest JSON must be a string")
        try:
            document_bytes = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ManifestValidationError("manifest JSON must contain valid UTF-8 text") from exc
        if document_bytes > MAX_MANIFEST_BYTES:
            raise ManifestValidationError(f"manifest exceeds {MAX_MANIFEST_BYTES} UTF-8 bytes")
        try:
            data = strict_json_loads(text)
        except ManifestValidationError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ManifestValidationError(f"invalid manifest JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "name": self.name,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "operations": [operation.to_dict() for operation in self.operations],
            "execution": self.execution.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


JOB_MANIFEST_V1_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/petehottelet/smythe/schemas/job-manifest-v1.json",
    "title": "Smythe Job Manifest v1",
    "x-smythe-max-canonical-plan-bytes": MAX_CANONICAL_PLAN_BYTES,
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "name", "profiles", "operations", "execution"],
    "properties": {
        "version": {"const": 1},
        "name": {"type": "string", "pattern": _NAME_RE.pattern},
        "profiles": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PROFILES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "provider", "model", "max_cost_per_call_usd"],
                "properties": {
                    "name": {"type": "string", "pattern": _NAME_RE.pattern},
                    "provider": {"enum": [provider.value for provider in ProviderKind]},
                    "model": {"type": "string", "minLength": 1},
                    "max_cost_per_call_usd": {
                        "oneOf": [
                            {
                                "type": "number",
                                "minimum": 0,
                                "maximum": float(MAX_USD),
                            },
                            {"type": "string", "pattern": r"^\d+(?:\.\d{1,6})?$"},
                        ]
                    },
                    "options": {"type": "object"},
                },
            },
        },
        "operations": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_OPERATION_TEMPLATES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "prompt", "profile"],
                "properties": {
                    "key": {"type": "string", "pattern": _NAME_RE.pattern},
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_OPERATION_COUNT,
                    },
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_PROMPT_BYTES,
                    },
                    "profile": {"type": "string", "pattern": _NAME_RE.pattern},
                    "attachments": {
                        "type": "array",
                        "maxItems": MAX_ATTACHMENTS_PER_OPERATION,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "artifact": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "mime_type": {"type": "string", "pattern": r"^[^/]+/[^/]+$"},
                            "width": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_ARTIFACT_DIMENSION,
                            },
                            "height": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_ARTIFACT_DIMENSION,
                            },
                            "filename": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_ARTIFACT_FILENAME_BYTES,
                            },
                        },
                    },
                },
            },
        },
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_concurrency", "max_attempts", "max_budget_usd"],
            "properties": {
                "max_concurrency": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CONCURRENCY,
                },
                "max_attempts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ATTEMPTS,
                },
                "max_budget_usd": {
                    "oneOf": [
                        {
                            "type": "number",
                            "minimum": 0,
                            "maximum": float(MAX_USD),
                        },
                        {"type": "string", "pattern": r"^\d+(?:\.\d{1,6})?$"},
                    ]
                },
                "call_timeout_s": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": MAX_CALL_TIMEOUT_S,
                    "default": DEFAULT_CALL_TIMEOUT_S,
                },
                "max_wall_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": MAX_WALL_SECONDS,
                    "default": DEFAULT_MAX_WALL_SECONDS,
                },
                "output_directory": {"type": "string", "minLength": 1},
            },
        },
    },
}
