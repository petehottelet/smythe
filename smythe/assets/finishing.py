"""Deterministic, crash-safe finishing for typed image assets."""

from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from smythe.assets.models import (
    AssetSpec,
    BrandMarkPolicy,
    BrandMode,
    BrandSpec,
    ImageFormat,
    LogoOverlaySpec,
    OverlayAnchor,
    TextOverlaySpec,
    preflight_assets,
)


@dataclass(frozen=True, slots=True)
class OverlayReceipt:
    """Evidence describing one deterministic overlay operation."""

    kind: str
    box: tuple[int, int, int, int]
    source_sha256: str | None = None
    text: str | None = None
    font_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FinishReceipt:
    """Hash-bound evidence for resize, crop, overlays, and encoding."""

    source_sha256: str
    output_sha256: str
    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    output_size: tuple[int, int]
    output_format: ImageFormat
    requested_dpi: tuple[int, int] | None
    observed_dpi: tuple[float, float] | None
    scale_factor: float
    cropped_fraction: float
    overlays: tuple[OverlayReceipt, ...] = ()


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageStat, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "Image finishing requires Pillow; install the asset dependencies"
        ) from exc
    return Image, ImageDraw, ImageFont, ImageStat, UnidentifiedImageError


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | os.PathLike[str], base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _position(
    canvas: tuple[int, int],
    item: tuple[int, int],
    *,
    anchor: OverlayAnchor,
    margin_ratio: float,
) -> tuple[int, int]:
    width, height = canvas
    item_width, item_height = item
    margin_x = round(width * margin_ratio)
    margin_y = round(height * margin_ratio)

    if anchor in {
        OverlayAnchor.TOP_LEFT,
        OverlayAnchor.BOTTOM_LEFT,
    }:
        x = margin_x
    elif anchor in {
        OverlayAnchor.TOP_RIGHT,
        OverlayAnchor.BOTTOM_RIGHT,
    }:
        x = width - margin_x - item_width
    else:
        x = (width - item_width) // 2

    if anchor in {
        OverlayAnchor.TOP_LEFT,
        OverlayAnchor.TOP_CENTER,
        OverlayAnchor.TOP_RIGHT,
    }:
        y = margin_y
    elif anchor in {
        OverlayAnchor.BOTTOM_LEFT,
        OverlayAnchor.BOTTOM_CENTER,
        OverlayAnchor.BOTTOM_RIGHT,
    }:
        y = height - margin_y - item_height
    else:
        y = (height - item_height) // 2
    return max(0, x), max(0, y)


def _composite_logo(image, logo_path: Path, overlay: LogoOverlaySpec):
    Image, _, _, _, _ = _pillow()
    logo_bytes = logo_path.read_bytes()
    with Image.open(io.BytesIO(logo_bytes)) as logo_source:
        logo = logo_source.convert("RGBA")
    target_width = max(1, round(image.width * overlay.width_ratio))
    target_height = max(1, round(logo.height * target_width / logo.width))
    max_height = max(1, round(image.height * (1 - 2 * overlay.margin_ratio)))
    if target_height > max_height:
        target_height = max_height
        target_width = max(1, round(logo.width * target_height / logo.height))
    logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if overlay.opacity < 1:
        alpha = logo.getchannel("A").point(lambda channel: round(channel * overlay.opacity))
        logo.putalpha(alpha)
    x, y = _position(
        image.size,
        logo.size,
        anchor=overlay.anchor,
        margin_ratio=overlay.margin_ratio,
    )
    composited = image.convert("RGBA")
    composited.alpha_composite(logo, (x, y))
    return composited, OverlayReceipt(
        kind="logo",
        box=(x, y, x + target_width, y + target_height),
        source_sha256=hashlib.sha256(logo_bytes).hexdigest(),
    )


def _load_font(overlay: TextOverlaySpec, width: int, *, production: bool, base_dir: Path):
    _, _, ImageFont, _, _ = _pillow()
    font_size = overlay.font_size or max(14, width // 30)
    if not overlay.font_path:
        if production:
            raise ValueError("production text overlays require a supplied font_path")
        return ImageFont.load_default(), None, False
    font_path = _resolve(overlay.font_path, base_dir)
    if not font_path.is_file():
        raise FileNotFoundError(f"font file not found: {overlay.font_path}")
    font_bytes = font_path.read_bytes()
    return ImageFont.truetype(io.BytesIO(font_bytes), font_size), font_bytes, True


def _composite_text(
    image,
    overlay: TextOverlaySpec,
    *,
    production: bool,
    base_dir: Path,
):
    _, ImageDraw, ImageFont, ImageStat, _ = _pillow()
    font, font_bytes, scalable = _load_font(
        overlay, image.width, production=production, base_dir=base_dir
    )
    draw = ImageDraw.Draw(image)
    margin_x = round(image.width * overlay.margin_ratio)
    margin_y = round(image.height * overlay.margin_ratio)
    maximum_width = image.width - 2 * margin_x
    maximum_height = image.height - 2 * margin_y
    if maximum_width <= 0 or maximum_height <= 0:
        raise ValueError(
            "overlay margins leave no drawable canvas for exact text: "
            f"{overlay.text!r}"
        )
    bbox = draw.textbbox((0, 0), overlay.text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # A caller-supplied TrueType/OpenType font can be reduced deterministically
    # until exact copy fits. Pillow's built-in bitmap font cannot be resized.
    if scalable and (text_width > maximum_width or text_height > maximum_height):
        assert font_bytes is not None
        font_size = overlay.font_size or max(14, image.width // 30)
        while (
            (text_width > maximum_width or text_height > maximum_height)
            and font_size > 8
        ):
            font_size -= 1
            font = ImageFont.truetype(io.BytesIO(font_bytes), font_size)
            bbox = draw.textbbox((0, 0), overlay.text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if text_width > maximum_width or text_height > maximum_height:
        raise ValueError(
            "overlay text does not fit asset canvas: "
            f"{overlay.text!r} measures {text_width}x{text_height}, "
            f"available {maximum_width}x{maximum_height}"
        )

    x, y = _position(
        image.size,
        (text_width, text_height),
        anchor=overlay.anchor,
        margin_ratio=overlay.margin_ratio,
    )
    actual_box = (x, y, x + text_width, y + text_height)
    if not (
        0 <= actual_box[0] < actual_box[2] <= image.width
        and 0 <= actual_box[1] < actual_box[3] <= image.height
    ):
        raise ValueError(
            f"overlay text box escapes asset canvas: {actual_box!r} "
            f"outside {image.size!r}"
        )
    fill = overlay.fill
    if fill is None:
        sample = image.crop(actual_box).convert("L")
        luminance = ImageStat.Stat(sample).mean[0] if sample.width and sample.height else 255
        fill = (24, 24, 22) if luminance > 140 else (245, 244, 240)
    # Offset by the font's own bearing so the measured box lands at x/y.
    draw.text((x - bbox[0], y - bbox[1]), overlay.text, font=font, fill=fill)
    return image, OverlayReceipt(
        kind="text",
        box=actual_box,
        text=overlay.text,
        font_sha256=(
            hashlib.sha256(font_bytes).hexdigest() if font_bytes is not None else None
        ),
    )


def _atomic_save(image, destination: Path, image_format: ImageFormat, **save_kwargs) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
        image.save(temp_path, format=image_format.value, **save_kwargs)
        # Windows requires a writable descriptor for FlushFileBuffers, which
        # Python's fsync delegates to. The encoder has already closed the file.
        with temp_path.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def finish_image(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    spec: AssetSpec,
    *,
    brand: BrandSpec | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> FinishReceipt:
    """Resize-to-cover, center-crop, overlay, and atomically encode an asset.

    The prior destination remains intact if decoding, compositing, encoding, or
    replacement fails. A production call re-checks master/font requirements so
    callers cannot bypass preflight accidentally.
    """

    if not isinstance(spec, AssetSpec):
        raise TypeError("spec must be an AssetSpec")
    if brand is not None and not isinstance(brand, BrandSpec):
        raise TypeError("brand must be a BrandSpec or None")
    Image, _, _, _, _ = _pillow()
    source_path = Path(source)
    destination_path = Path(destination)
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    production = bool(brand and brand.mode == BrandMode.PRODUCTION)
    if production:
        assert brand is not None
        preflight_assets(brand, [spec], base_dir=root)
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    with Image.open(io.BytesIO(source_bytes)) as loaded:
        source_size = loaded.size
        has_alpha = "A" in loaded.getbands() or "transparency" in loaded.info
        working = loaded.convert("RGBA" if has_alpha or spec.alpha_required else "RGB")

    scale = max(spec.width / source_size[0], spec.height / source_size[1])
    resized_size = (
        max(spec.width, math.ceil(source_size[0] * scale)),
        max(spec.height, math.ceil(source_size[1] * scale)),
    )
    working = working.resize(resized_size, Image.Resampling.LANCZOS)
    left = (resized_size[0] - spec.width) // 2
    top = (resized_size[1] - spec.height) // 2
    working = working.crop((left, top, left + spec.width, top + spec.height))

    overlay_receipts: list[OverlayReceipt] = []
    if spec.mark_policy == BrandMarkPolicy.COMPOSITE_EXACT:
        if brand is None or not brand.logo_path:
            raise ValueError("exact logo composite requires BrandSpec.logo_path")
        assert spec.logo_overlay is not None
        logo_path = _resolve(brand.logo_path, root)
        if not logo_path.is_file():
            raise FileNotFoundError(f"brand logo file not found: {brand.logo_path}")
        working, receipt = _composite_logo(working, logo_path, spec.logo_overlay)
        overlay_receipts.append(receipt)

    for overlay in spec.text_overlays:
        working, receipt = _composite_text(
            working,
            overlay,
            production=production,
            base_dir=root,
        )
        overlay_receipts.append(receipt)

    if spec.format == ImageFormat.JPEG or spec.alpha_required is False:
        working = working.convert("RGB")
    elif spec.alpha_required is True:
        working = working.convert("RGBA")

    save_kwargs: dict[str, object] = {}
    if spec.dpi is not None:
        save_kwargs["dpi"] = spec.dpi
    if spec.format == ImageFormat.JPEG:
        save_kwargs.update(quality=92, optimize=False)
    elif spec.format == ImageFormat.PNG:
        save_kwargs["compress_level"] = 6
    elif spec.format == ImageFormat.WEBP:
        save_kwargs.update(quality=92, method=6)

    _atomic_save(working, destination_path, spec.format, **save_kwargs)

    observed_dpi: tuple[float, float] | None = None
    with Image.open(destination_path) as finished:
        recorded = finished.info.get("dpi")
        if isinstance(recorded, (tuple, list)) and len(recorded) >= 2:
            observed_dpi = float(recorded[0]), float(recorded[1])

    crop_fraction = 1 - ((spec.width * spec.height) / (resized_size[0] * resized_size[1]))
    return FinishReceipt(
        source_sha256=source_hash,
        output_sha256=sha256_file(destination_path),
        source_size=source_size,
        resized_size=resized_size,
        output_size=spec.size,
        output_format=spec.format,
        requested_dpi=spec.dpi,
        observed_dpi=observed_dpi,
        scale_factor=round(scale, 6),
        cropped_fraction=round(max(0.0, crop_fraction), 6),
        overlays=tuple(overlay_receipts),
    )


__all__ = ["FinishReceipt", "OverlayReceipt", "finish_image", "sha256_file"]
