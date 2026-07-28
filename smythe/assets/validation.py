"""Deterministic validation gates for finished image assets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from smythe.assets.finishing import FinishReceipt, OverlayReceipt, sha256_file
from smythe.assets.models import (
    AssetSpec,
    BrandMarkPolicy,
    BrandSpec,
    TextPolicy,
)


class FindingSeverity(StrEnum):
    """Whether a finding can fail deterministic acceptance."""

    HARD = "hard"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One stable, machine-readable validation observation."""

    code: str
    message: str
    severity: FindingSeverity = FindingSeverity.HARD
    expected: object | None = None
    observed: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("finding code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("finding message must be a non-empty string")
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        try:
            severity = FindingSeverity(self.severity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown finding severity: {self.severity!r}") from exc
        object.__setattr__(self, "severity", severity)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A report whose pass state is controlled only by hard findings."""

    path: str
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def hard_findings(self) -> tuple[ValidationFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.severity == FindingSeverity.HARD
        )

    @property
    def advisories(self) -> tuple[ValidationFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.severity == FindingSeverity.ADVISORY
        )

    @property
    def passed(self) -> bool:
        return not self.hard_findings


def hard_finding(
    code: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        message=message,
        severity=FindingSeverity.HARD,
        expected=expected,
        observed=observed,
    )


def advisory_finding(
    code: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        code=code,
        message=message,
        severity=FindingSeverity.ADVISORY,
        expected=expected,
        observed=observed,
    )


def _resolve(path: str | os.PathLike[str], base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _receipt_findings(
    path: Path,
    spec: AssetSpec,
    receipt: FinishReceipt,
    *,
    brand: BrandSpec | None,
    base_dir: Path,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    receipt_overlays: list[OverlayReceipt] = []
    for index, item in enumerate(receipt.overlays):
        if not isinstance(item, OverlayReceipt):
            findings.append(
                hard_finding(
                    "overlay_receipt_invalid",
                    "overlay receipt must be an OverlayReceipt value",
                    expected="OverlayReceipt",
                    observed={"index": index, "type": type(item).__name__},
                )
            )
            continue
        receipt_overlays.append(item)
        box = item.box
        if (
            not isinstance(box, (tuple, list))
            or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
        ):
            findings.append(
                hard_finding(
                    "overlay_receipt_box_invalid",
                    "overlay receipt box must contain four integer coordinates",
                    expected="(left, top, right, bottom)",
                    observed={"index": index, "kind": item.kind, "box": box},
                )
            )
            continue
        left, top, right, bottom = box
        if left >= right or top >= bottom:
            findings.append(
                hard_finding(
                    "overlay_receipt_box_invalid",
                    "overlay receipt box must have positive width and height",
                    expected="left < right and top < bottom",
                    observed={"index": index, "kind": item.kind, "box": box},
                )
            )
        if left < 0 or top < 0 or right > spec.width or bottom > spec.height:
            findings.append(
                hard_finding(
                    "overlay_receipt_box_out_of_bounds",
                    "overlay receipt box escapes the finished asset canvas",
                    expected=(0, 0, spec.width, spec.height),
                    observed={"index": index, "kind": item.kind, "box": box},
                )
            )
    actual_hash = sha256_file(path)
    if actual_hash != receipt.output_sha256:
        findings.append(
            hard_finding(
                "receipt_hash_mismatch",
                "finished bytes no longer match the finishing receipt",
                expected=receipt.output_sha256,
                observed=actual_hash,
            )
        )
    if receipt.output_size != spec.size:
        findings.append(
            hard_finding(
                "receipt_size_mismatch",
                "receipt output size does not match the asset specification",
                expected=spec.size,
                observed=receipt.output_size,
            )
        )
    if receipt.output_format != spec.format:
        findings.append(
            hard_finding(
                "receipt_format_mismatch",
                "receipt output format does not match the asset specification",
                expected=spec.format.value,
                observed=receipt.output_format.value,
            )
        )
    if receipt.requested_dpi != spec.dpi:
        findings.append(
            hard_finding(
                "receipt_dpi_mismatch",
                "receipt DPI request does not match the asset specification",
                expected=spec.dpi,
                observed=receipt.requested_dpi,
            )
        )

    if spec.mark_policy == BrandMarkPolicy.COMPOSITE_EXACT:
        logo_receipts = [item for item in receipt_overlays if item.kind == "logo"]
        if len(logo_receipts) != 1:
            findings.append(
                hard_finding(
                    "logo_receipt_missing",
                    "exact-mark asset requires exactly one logo overlay receipt",
                    expected=1,
                    observed=len(logo_receipts),
                )
            )
        elif brand is None or not brand.logo_path:
            findings.append(
                hard_finding(
                    "logo_master_unavailable",
                    "exact-mark receipt requires the approved BrandSpec logo master",
                )
            )
        else:
            logo_path = _resolve(brand.logo_path, base_dir)
            if not logo_path.is_file():
                findings.append(
                    hard_finding(
                        "logo_master_unavailable",
                        "approved logo master is unavailable during validation",
                        expected=str(logo_path),
                    )
                )
            else:
                master_hash = sha256_file(logo_path)
                if logo_receipts[0].source_sha256 != master_hash:
                    findings.append(
                        hard_finding(
                            "logo_master_hash_mismatch",
                            "logo overlay receipt does not reference the approved master",
                            expected=master_hash,
                            observed=logo_receipts[0].source_sha256,
                        )
                    )

    if spec.text_policy == TextPolicy.COMPOSITE_EXACT:
        text_receipts = [item for item in receipt_overlays if item.kind == "text"]
        if len(text_receipts) != len(spec.text_overlays):
            findings.append(
                hard_finding(
                    "text_receipt_count_mismatch",
                    "exact-copy overlays are not fully represented in the receipt",
                    expected=len(spec.text_overlays),
                    observed=len(text_receipts),
                )
            )
        for index, overlay in enumerate(spec.text_overlays):
            if index >= len(text_receipts):
                break
            recorded = text_receipts[index]
            if recorded.text != overlay.text:
                findings.append(
                    hard_finding(
                        "text_receipt_copy_mismatch",
                        "text receipt does not contain the required exact copy",
                        expected=overlay.text,
                        observed=recorded.text,
                    )
                )
            if overlay.font_path:
                font_path = _resolve(overlay.font_path, base_dir)
                if not font_path.is_file():
                    findings.append(
                        hard_finding(
                            "font_master_unavailable",
                            "supplied font master is unavailable during validation",
                            expected=str(font_path),
                        )
                    )
                else:
                    font_hash = sha256_file(font_path)
                    if recorded.font_sha256 != font_hash:
                        findings.append(
                            hard_finding(
                                "font_hash_mismatch",
                                "text receipt does not reference the supplied font",
                                expected=font_hash,
                                observed=recorded.font_sha256,
                            )
                        )
    return findings


def validate_image(
    path: str | os.PathLike[str],
    spec: AssetSpec,
    *,
    receipt: FinishReceipt | None = None,
    require_receipt: bool = False,
    brand: BrandSpec | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    dpi_tolerance: float = 1.0,
    advisory_findings: Iterable[ValidationFinding] = (),
) -> ValidationReport:
    """Validate content bytes against an asset specification.

    ``advisory_findings`` is intentionally a separate input and rejects hard
    findings. This keeps vision/OCR/judge observations visible without letting
    them silently redefine deterministic acceptance.
    """

    if not isinstance(spec, AssetSpec):
        raise TypeError("spec must be an AssetSpec")
    if receipt is not None and not isinstance(receipt, FinishReceipt):
        raise TypeError("receipt must be a FinishReceipt or None")
    if brand is not None and not isinstance(brand, BrandSpec):
        raise TypeError("brand must be a BrandSpec or None")
    if isinstance(dpi_tolerance, bool) or not isinstance(dpi_tolerance, (int, float)):
        raise TypeError("dpi_tolerance must be a number")
    if dpi_tolerance < 0:
        raise ValueError("dpi_tolerance must be non-negative")

    advisories = tuple(advisory_findings)
    if any(not isinstance(finding, ValidationFinding) for finding in advisories):
        raise TypeError("advisory_findings must contain ValidationFinding values")
    if any(finding.severity != FindingSeverity.ADVISORY for finding in advisories):
        raise ValueError("advisory_findings cannot contain hard findings")

    image_path = Path(path)
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    findings: list[ValidationFinding] = []
    if not image_path.is_file():
        findings.append(
            hard_finding(
                "file_missing",
                "finished artifact file does not exist",
                observed=str(image_path),
            )
        )
        return ValidationReport(str(image_path), tuple(findings) + advisories)

    exact_policy = (
        spec.mark_policy == BrandMarkPolicy.COMPOSITE_EXACT
        or spec.text_policy == TextPolicy.COMPOSITE_EXACT
    )
    if receipt is None and (require_receipt or exact_policy):
        findings.append(
            hard_finding(
                "receipt_missing",
                "a finishing receipt is required for exact or requested validation",
            )
        )
    elif receipt is not None:
        findings.extend(
            _receipt_findings(
                image_path,
                spec,
                receipt,
                brand=brand,
                base_dir=root,
            )
        )

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "Image validation requires Pillow; install the asset dependencies"
        ) from exc

    try:
        with Image.open(image_path) as image:
            image.load()
            observed_format = (image.format or "").upper()
            observed_size = image.size
            observed_dpi = image.info.get("dpi")
            has_alpha = "A" in image.getbands() or "transparency" in image.info
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        findings.append(
            hard_finding(
                "decode_failed",
                "artifact bytes are not a decodable image",
                observed=type(exc).__name__,
            )
        )
        return ValidationReport(str(image_path), tuple(findings) + advisories)

    if observed_format == "JPG":
        observed_format = "JPEG"
    if observed_format != spec.format.value:
        findings.append(
            hard_finding(
                "content_format_mismatch",
                "decoded image format does not match the asset specification",
                expected=spec.format.value,
                observed=observed_format or None,
            )
        )
    if observed_size != spec.size:
        findings.append(
            hard_finding(
                "dimensions_mismatch",
                "decoded pixel dimensions do not match the asset specification",
                expected=spec.size,
                observed=observed_size,
            )
        )

    if spec.dpi is not None:
        if not isinstance(observed_dpi, (tuple, list)) or len(observed_dpi) < 2:
            findings.append(
                hard_finding(
                    "dpi_missing",
                    "asset specification requires DPI metadata",
                    expected=spec.dpi,
                    observed=observed_dpi,
                )
            )
        else:
            actual_dpi = float(observed_dpi[0]), float(observed_dpi[1])
            if any(
                abs(actual - expected) > dpi_tolerance
                for actual, expected in zip(actual_dpi, spec.dpi, strict=True)
            ):
                findings.append(
                    hard_finding(
                        "dpi_mismatch",
                        "DPI metadata does not match the asset specification",
                        expected=spec.dpi,
                        observed=actual_dpi,
                    )
                )

    if spec.alpha_required is True and not has_alpha:
        findings.append(
            hard_finding(
                "alpha_channel_missing",
                "asset requires an alpha channel",
                expected=True,
                observed=False,
            )
        )
    elif spec.alpha_required is False and has_alpha:
        findings.append(
            hard_finding(
                "alpha_channel_unexpected",
                "asset must be opaque and contain no alpha channel",
                expected=False,
                observed=True,
            )
        )

    return ValidationReport(str(image_path), tuple(findings) + advisories)


__all__ = [
    "FindingSeverity",
    "ValidationFinding",
    "ValidationReport",
    "advisory_finding",
    "hard_finding",
    "validate_image",
]
