"""Typed contracts and preflight rules for image-asset jobs.

The types in this module deliberately have no imaging or provider dependency.
They describe the final deliverable contract; provider-native sizes and aspect
buckets belong to provider adapters, not to :class:`AssetSpec`.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable


_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RASTER_LOGO_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


class BrandMode(StrEnum):
    """Whether an asset package is exploratory or production-bound."""

    CONCEPT = "concept"
    PRODUCTION = "production"


class ImageFormat(StrEnum):
    """Supported finished raster formats."""

    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"


_DPI_CAPABLE_FORMATS = {ImageFormat.PNG, ImageFormat.JPEG}


class BrandMarkPolicy(StrEnum):
    """How the official brand mark is allowed to reach an asset."""

    NONE = "none"
    REFERENCE_ONLY = "reference_only"
    COMPOSITE_EXACT = "composite_exact"


class TextPolicy(StrEnum):
    """How visible copy is allowed to reach an asset."""

    NONE = "none"
    MODEL_RENDERED = "model_rendered"
    COMPOSITE_EXACT = "composite_exact"


class OverlayAnchor(StrEnum):
    """Deterministic placement anchors used by the finishing pass."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _enum(value: object, enum_type: type[StrEnum], *, name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {choices}, got {value!r}") from exc


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _ratio(value: object, *, name: str, maximum: float = 0.5) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not 0 <= normalized <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return normalized


def _color(value: object, *, name: str) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an RGB tuple")
    try:
        channels = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an RGB tuple") from exc
    if len(channels) != 3 or any(
        isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in channels
    ):
        raise ValueError(f"{name} must contain three integers between 0 and 255")
    return channels


@dataclass(frozen=True, slots=True)
class TextOverlaySpec:
    """Exact text to composite during deterministic finishing.

    A font may be omitted for a concept run. Production preflight and the
    finishing pass both require a caller-supplied font file so output does not
    depend on whatever fonts happen to be installed on the host.
    """

    text: str
    font_path: str | os.PathLike[str] | None = None
    font_size: int | None = None
    anchor: OverlayAnchor = OverlayAnchor.BOTTOM_CENTER
    margin_ratio: float = 0.0625
    fill: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _nonempty(self.text, name="overlay text"))
        if self.font_path is not None:
            object.__setattr__(self, "font_path", os.fspath(self.font_path))
        if self.font_size is not None:
            _positive_int(self.font_size, name="font_size")
        object.__setattr__(self, "anchor", _enum(self.anchor, OverlayAnchor, name="anchor"))
        object.__setattr__(
            self,
            "margin_ratio",
            _ratio(self.margin_ratio, name="margin_ratio"),
        )
        object.__setattr__(self, "fill", _color(self.fill, name="fill"))


@dataclass(frozen=True, slots=True)
class LogoOverlaySpec:
    """Placement for an exact raster brand-master composite."""

    anchor: OverlayAnchor = OverlayAnchor.BOTTOM_RIGHT
    width_ratio: float = 0.2
    margin_ratio: float = 0.04
    opacity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _enum(self.anchor, OverlayAnchor, name="anchor"))
        object.__setattr__(
            self,
            "width_ratio",
            _ratio(self.width_ratio, name="width_ratio", maximum=1.0),
        )
        if self.width_ratio == 0:
            raise ValueError("width_ratio must be greater than zero")
        object.__setattr__(
            self,
            "margin_ratio",
            _ratio(self.margin_ratio, name="margin_ratio"),
        )
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float)):
            raise TypeError("opacity must be a number")
        opacity = float(self.opacity)
        if not 0 < opacity <= 1:
            raise ValueError("opacity must be greater than zero and at most one")
        object.__setattr__(self, "opacity", opacity)


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """The deterministic final-output contract for one raster asset."""

    id: str
    prompt: str
    width: int
    height: int
    format: ImageFormat = ImageFormat.PNG
    dpi: tuple[int, int] | None = None
    alpha_required: bool | None = None
    mark_policy: BrandMarkPolicy = BrandMarkPolicy.NONE
    logo_overlay: LogoOverlaySpec | None = None
    text_policy: TextPolicy = TextPolicy.NONE
    text_overlays: tuple[TextOverlaySpec, ...] = field(default_factory=tuple)
    candidates: int = 1

    def __post_init__(self) -> None:
        asset_id = _nonempty(self.id, name="asset id")
        if not _ASSET_ID_RE.fullmatch(asset_id) or asset_id in {".", ".."}:
            raise ValueError(
                "asset id must start with an alphanumeric character and contain "
                "only letters, numbers, '.', '_', or '-'"
            )
        object.__setattr__(self, "id", asset_id)
        object.__setattr__(self, "prompt", _nonempty(self.prompt, name="asset prompt"))
        _positive_int(self.width, name="width")
        _positive_int(self.height, name="height")
        object.__setattr__(self, "format", _enum(self.format, ImageFormat, name="format"))
        if self.dpi is not None:
            if isinstance(self.dpi, (str, bytes)):
                raise TypeError("dpi must be a two-item integer tuple")
            try:
                dpi = tuple(self.dpi)
            except TypeError as exc:
                raise TypeError("dpi must be a two-item integer tuple") from exc
            if len(dpi) != 2:
                raise ValueError("dpi must contain exactly two values")
            for index, value in enumerate(dpi):
                _positive_int(value, name=f"dpi[{index}]")
            object.__setattr__(self, "dpi", dpi)
        if self.alpha_required is not None and not isinstance(self.alpha_required, bool):
            raise TypeError("alpha_required must be a boolean or None")
        if self.format == ImageFormat.JPEG and self.alpha_required is True:
            raise ValueError("JPEG assets cannot require an alpha channel")
        if self.dpi is not None and self.format not in _DPI_CAPABLE_FORMATS:
            raise ValueError(
                f"DPI metadata is unsupported for {self.format.value} assets; "
                "use PNG or JPEG"
            )
        object.__setattr__(
            self,
            "mark_policy",
            _enum(self.mark_policy, BrandMarkPolicy, name="mark_policy"),
        )
        object.__setattr__(
            self, "text_policy", _enum(self.text_policy, TextPolicy, name="text_policy")
        )
        overlays = tuple(self.text_overlays)
        if any(not isinstance(item, TextOverlaySpec) for item in overlays):
            raise TypeError("text_overlays must contain only TextOverlaySpec values")
        object.__setattr__(self, "text_overlays", overlays)
        _positive_int(self.candidates, name="candidates")

        if self.mark_policy == BrandMarkPolicy.COMPOSITE_EXACT:
            if self.logo_overlay is None:
                raise ValueError("COMPOSITE_EXACT mark policy requires logo_overlay")
        elif self.logo_overlay is not None:
            raise ValueError("logo_overlay requires COMPOSITE_EXACT mark policy")

        if self.text_policy == TextPolicy.COMPOSITE_EXACT:
            if not overlays:
                raise ValueError("COMPOSITE_EXACT text policy requires text_overlays")
        elif overlays:
            raise ValueError("text_overlays require COMPOSITE_EXACT text policy")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class BrandSpec:
    """Brand inputs and the trust mode under which they may be used."""

    name: str
    brief: str
    mode: BrandMode = BrandMode.CONCEPT
    logo_path: str | os.PathLike[str] | None = None
    logo_prompt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, name="brand name"))
        object.__setattr__(self, "brief", _nonempty(self.brief, name="brand brief"))
        object.__setattr__(self, "mode", _enum(self.mode, BrandMode, name="mode"))
        if self.logo_path is not None:
            object.__setattr__(self, "logo_path", os.fspath(self.logo_path))
        if self.logo_prompt is not None:
            object.__setattr__(
                self,
                "logo_prompt",
                _nonempty(self.logo_prompt, name="logo_prompt"),
            )
        if self.mode == BrandMode.PRODUCTION:
            if not self.logo_path:
                raise ValueError("production brand mode requires a supplied logo_path")
            if self.logo_prompt:
                raise ValueError("production brand mode does not allow a generated logo_prompt")


class AssetPreflightError(ValueError):
    """Raised when an asset package is unsafe or internally inconsistent."""

    def __init__(self, violations: Iterable[str]) -> None:
        self.violations = tuple(violations)
        super().__init__("Asset preflight failed: " + "; ".join(self.violations))


def preflight_assets(
    brand: BrandSpec,
    assets: Iterable[AssetSpec],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> tuple[AssetSpec, ...]:
    """Validate brand masters and production-only policies before generation.

    Returning a detached tuple makes the exact preflighted asset inventory easy
    for callers to hash or place in a job manifest. All violations are reported
    together and no provider interaction occurs in this layer.
    """

    if not isinstance(brand, BrandSpec):
        raise TypeError("brand must be a BrandSpec")
    inventory = tuple(assets)
    if not inventory:
        raise AssetPreflightError(["at least one asset is required"])
    if any(not isinstance(asset, AssetSpec) for asset in inventory):
        raise TypeError("assets must contain only AssetSpec values")

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    violations: list[str] = []
    ids = [asset.id for asset in inventory]
    duplicates = sorted(asset_id for asset_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        violations.append(f"duplicate asset ids: {duplicates}")

    logo_path: Path | None = None
    if brand.logo_path:
        candidate = Path(brand.logo_path)
        logo_path = candidate if candidate.is_absolute() else root / candidate
        if not logo_path.is_file():
            violations.append(f"brand logo file not found: {brand.logo_path}")

    for asset in inventory:
        prefix = f"asset {asset.id!r}"
        if brand.mode == BrandMode.PRODUCTION:
            if asset.mark_policy == BrandMarkPolicy.REFERENCE_ONLY:
                violations.append(
                    f"{prefix} uses reference-only branding; production requires "
                    "COMPOSITE_EXACT or NONE"
                )
            if asset.text_policy == TextPolicy.MODEL_RENDERED:
                violations.append(
                    f"{prefix} uses model-rendered text; production exact copy must "
                    "use COMPOSITE_EXACT"
                )
            for overlay in asset.text_overlays:
                if not overlay.font_path:
                    violations.append(
                        f"{prefix} production text overlay requires a supplied " "font_path"
                    )
                    continue
                font = Path(overlay.font_path)
                font = font if font.is_absolute() else root / font
                if not font.is_file():
                    violations.append(f"{prefix} font file not found: {overlay.font_path}")

        if asset.mark_policy == BrandMarkPolicy.COMPOSITE_EXACT:
            if logo_path is None:
                violations.append(f"{prefix} exact logo composite requires a supplied logo_path")
            elif logo_path.suffix.lower() not in _RASTER_LOGO_SUFFIXES:
                violations.append(
                    f"{prefix} exact logo composite requires a raster logo master; "
                    f"got {logo_path.suffix or 'no extension'}"
                )

    if violations:
        raise AssetPreflightError(violations)
    return inventory


__all__ = [
    "AssetPreflightError",
    "AssetSpec",
    "BrandMarkPolicy",
    "BrandMode",
    "BrandSpec",
    "ImageFormat",
    "LogoOverlaySpec",
    "OverlayAnchor",
    "TextOverlaySpec",
    "TextPolicy",
    "preflight_assets",
]
