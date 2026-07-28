"""Public contracts for deterministic, brand-aware image assets."""

from smythe.assets.finishing import (
    FinishReceipt,
    OverlayReceipt,
    finish_image,
    sha256_file,
)
from smythe.assets.models import (
    AssetPreflightError,
    AssetSpec,
    BrandMarkPolicy,
    BrandMode,
    BrandSpec,
    ImageFormat,
    LogoOverlaySpec,
    OverlayAnchor,
    TextOverlaySpec,
    TextPolicy,
    preflight_assets,
)
from smythe.assets.validation import (
    FindingSeverity,
    ValidationFinding,
    ValidationReport,
    advisory_finding,
    hard_finding,
    validate_image,
)

__all__ = [
    "AssetPreflightError",
    "AssetSpec",
    "BrandMarkPolicy",
    "BrandMode",
    "BrandSpec",
    "FindingSeverity",
    "FinishReceipt",
    "ImageFormat",
    "LogoOverlaySpec",
    "OverlayAnchor",
    "OverlayReceipt",
    "TextOverlaySpec",
    "TextPolicy",
    "ValidationFinding",
    "ValidationReport",
    "advisory_finding",
    "finish_image",
    "hard_finding",
    "preflight_assets",
    "sha256_file",
    "validate_image",
]
