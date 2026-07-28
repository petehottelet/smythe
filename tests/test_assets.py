"""Tests for the public, deterministic image-asset layer."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageFont  # noqa: E402

from smythe.assets import (  # noqa: E402
    AssetPreflightError,
    AssetSpec,
    BrandMarkPolicy,
    BrandMode,
    BrandSpec,
    ImageFormat,
    LogoOverlaySpec,
    TextOverlaySpec,
    TextPolicy,
    advisory_finding,
    finish_image,
    hard_finding,
    preflight_assets,
    validate_image,
)


def _image(
    path: Path,
    *,
    size: tuple[int, int] = (120, 80),
    image_format: str = "PNG",
    mode: str = "RGB",
    dpi: tuple[int, int] | None = None,
    color=(180, 150, 110),
) -> Path:
    kwargs = {"dpi": dpi} if dpi is not None else {}
    Image.new(mode, size, color).save(path, format=image_format, **kwargs)
    return path


def _font_path() -> Path:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        pytest.skip("DejaVu Sans is unavailable for deterministic font tests")
    path = Path(font.path)
    if not path.is_file():
        pytest.skip("resolved DejaVu Sans path is unavailable")
    return path


def test_specs_are_typed_frozen_and_reject_ambiguous_overlay_policies():
    spec = AssetSpec(
        id="hero",
        prompt="A clean product hero",
        width=1200,
        height=630,
        format="PNG",
    )
    assert spec.format is ImageFormat.PNG
    assert spec.size == (1200, 630)
    with pytest.raises(AttributeError):
        spec.width = 10
    with pytest.raises(ValueError, match="text_overlays require"):
        AssetSpec(
            id="bad",
            prompt="x",
            width=10,
            height=10,
            text_overlays=(TextOverlaySpec("Exact copy"),),
        )
    with pytest.raises(ValueError, match="logo_overlay requires"):
        AssetSpec(
            id="bad-logo",
            prompt="x",
            width=10,
            height=10,
            logo_overlay=LogoOverlaySpec(),
        )


def test_specs_reject_impossible_alpha_and_dpi_format_contracts():
    with pytest.raises(ValueError, match="JPEG assets cannot require an alpha"):
        AssetSpec(
            id="jpeg-alpha",
            prompt="x",
            width=100,
            height=100,
            format=ImageFormat.JPEG,
            alpha_required=True,
        )
    with pytest.raises(ValueError, match="DPI metadata is unsupported for WEBP"):
        AssetSpec(
            id="webp-print",
            prompt="x",
            width=100,
            height=100,
            format=ImageFormat.WEBP,
            dpi=(300, 300),
        )


def test_production_brand_requires_supplied_master_and_forbids_generation_prompt():
    with pytest.raises(ValueError, match="requires a supplied logo_path"):
        BrandSpec("Acme", "Industrial", mode=BrandMode.PRODUCTION)
    with pytest.raises(ValueError, match="does not allow"):
        BrandSpec(
            "Acme",
            "Industrial",
            mode=BrandMode.PRODUCTION,
            logo_path="logo.png",
            logo_prompt="invent one",
        )


def test_production_preflight_rejects_reference_only_mark_and_model_text(tmp_path):
    logo = _image(tmp_path / "logo.png", size=(30, 20))
    brand = BrandSpec("Acme", "Industrial", BrandMode.PRODUCTION, logo_path=logo)
    asset = AssetSpec(
        id="hero",
        prompt="x",
        width=100,
        height=100,
        mark_policy=BrandMarkPolicy.REFERENCE_ONLY,
        text_policy=TextPolicy.MODEL_RENDERED,
    )
    with pytest.raises(AssetPreflightError) as raised:
        preflight_assets(brand, [asset])
    message = str(raised.value)
    assert "reference-only branding" in message
    assert "model-rendered text" in message


def test_concept_preflight_allows_generated_reference_logo():
    brand = BrandSpec(
        "Concept Co",
        "Exploratory campaign",
        mode="concept",
        logo_prompt="Invent a simple mark",
    )
    asset = AssetSpec(
        id="hero",
        prompt="x",
        width=100,
        height=100,
        mark_policy="reference_only",
    )
    assert preflight_assets(brand, [asset]) == (asset,)


def test_preflight_reports_missing_files_fonts_and_duplicate_ids(tmp_path):
    brand = BrandSpec(
        "Acme",
        "Industrial",
        BrandMode.PRODUCTION,
        logo_path="missing.png",
    )
    overlay = TextOverlaySpec("Exact", font_path="missing.ttf")
    asset = AssetSpec(
        id="hero",
        prompt="x",
        width=100,
        height=100,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(overlay,),
    )
    with pytest.raises(AssetPreflightError) as raised:
        preflight_assets(brand, [asset, asset], base_dir=tmp_path)
    assert "duplicate asset ids" in str(raised.value)
    assert "logo file not found" in str(raised.value)
    assert "font file not found" in str(raised.value)


def test_finish_resize_cover_exact_format_dpi_and_receipt(tmp_path):
    source = _image(tmp_path / "source.png", size=(120, 80))
    destination = tmp_path / "print.png"
    spec = AssetSpec(
        id="print",
        prompt="x",
        width=90,
        height=120,
        format=ImageFormat.PNG,
        dpi=(300, 300),
        alpha_required=False,
    )
    receipt = finish_image(source, destination, spec)

    assert receipt.source_size == (120, 80)
    assert receipt.output_size == (90, 120)
    assert receipt.output_format is ImageFormat.PNG
    assert receipt.scale_factor == pytest.approx(1.5)
    assert receipt.cropped_fraction > 0
    report = validate_image(destination, spec, receipt=receipt, require_receipt=True)
    assert report.passed
    assert not report.findings


def test_production_exact_logo_and_copy_emit_hash_bound_receipts(tmp_path):
    source = _image(tmp_path / "source.png", size=(160, 100))
    logo = _image(
        tmp_path / "logo.png",
        size=(40, 20),
        mode="RGBA",
        color=(10, 20, 30, 255),
    )
    font = tmp_path / "approved-font.ttf"
    font.write_bytes(Path(_font_path()).read_bytes())
    brand = BrandSpec(
        "Acme",
        "Industrial",
        BrandMode.PRODUCTION,
        logo_path=logo,
    )
    spec = AssetSpec(
        id="hero",
        prompt="no text",
        width=160,
        height=100,
        mark_policy=BrandMarkPolicy.COMPOSITE_EXACT,
        logo_overlay=LogoOverlaySpec(width_ratio=0.2),
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(
            TextOverlaySpec(
                "EXACT COPY",
                font_path=font,
                font_size=14,
                fill=(255, 255, 255),
            ),
        ),
    )
    preflight_assets(brand, [spec])
    destination = tmp_path / "hero.png"
    receipt = finish_image(source, destination, spec, brand=brand)

    assert [overlay.kind for overlay in receipt.overlays] == ["logo", "text"]
    assert all(overlay.source_sha256 for overlay in receipt.overlays[:1])
    assert receipt.overlays[1].font_sha256
    report = validate_image(
        destination,
        spec,
        receipt=receipt,
        require_receipt=True,
        brand=brand,
    )
    assert report.passed


def test_exact_receipt_requires_available_trust_anchors(tmp_path):
    source = _image(tmp_path / "source.png", size=(160, 100))
    logo = _image(tmp_path / "logo.png", size=(40, 20), mode="RGBA")
    font = tmp_path / "approved-font.ttf"
    font.write_bytes(Path(_font_path()).read_bytes())
    brand = BrandSpec("Acme", "Industrial", BrandMode.PRODUCTION, logo_path=logo)
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=160,
        height=100,
        mark_policy=BrandMarkPolicy.COMPOSITE_EXACT,
        logo_overlay=LogoOverlaySpec(width_ratio=0.2),
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(TextOverlaySpec("EXACT", font_path=font, font_size=14),),
    )
    destination = tmp_path / "hero.png"
    receipt = finish_image(source, destination, spec, brand=brand)

    no_brand = validate_image(destination, spec, receipt=receipt)
    assert "logo_master_unavailable" in {
        finding.code for finding in no_brand.hard_findings
    }

    logo.unlink()
    font.unlink()
    missing_masters = validate_image(destination, spec, receipt=receipt, brand=brand)
    codes = {finding.code for finding in missing_masters.hard_findings}
    assert "logo_master_unavailable" in codes
    assert "font_master_unavailable" in codes


def test_exact_policy_requires_receipt_even_without_explicit_flag(tmp_path):
    image = _image(tmp_path / "hero.png", size=(100, 100))
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=100,
        height=100,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(TextOverlaySpec("EXACT"),),
    )

    report = validate_image(image, spec)

    assert "receipt_missing" in {finding.code for finding in report.hard_findings}


def test_finishing_receipt_hashes_the_same_source_bytes_that_are_decoded(
    tmp_path,
    monkeypatch,
):
    source = _image(tmp_path / "source.png", size=(20, 20), color=(255, 0, 0))
    approved_bytes = source.read_bytes()
    expected_hash = hashlib.sha256(approved_bytes).hexdigest()
    real_read_bytes = Path.read_bytes
    mutated = False

    def read_then_mutate(path):
        nonlocal mutated
        data = real_read_bytes(path)
        if path == source and not mutated:
            mutated = True
            _image(source, size=(20, 20), color=(0, 0, 0))
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_mutate)
    destination = tmp_path / "finished.png"
    spec = AssetSpec(id="hero", prompt="x", width=20, height=20)

    receipt = finish_image(source, destination, spec)

    assert receipt.source_sha256 == expected_hash
    with Image.open(destination) as output:
        assert output.convert("RGB").getpixel((10, 10)) == (255, 0, 0)


def test_production_finishing_rechecks_font_requirement(tmp_path):
    source = _image(tmp_path / "source.png")
    logo = _image(tmp_path / "logo.png", size=(30, 20))
    brand = BrandSpec("Acme", "Brief", BrandMode.PRODUCTION, logo_path=logo)
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=120,
        height=80,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(TextOverlaySpec("EXACT"),),
    )
    with pytest.raises(ValueError, match="requires a supplied font_path"):
        finish_image(source, tmp_path / "final.png", spec, brand=brand)


def test_finish_is_repeatable_with_pinned_inputs(tmp_path):
    source = _image(tmp_path / "source.png")
    font = _font_path()
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=120,
        height=80,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(TextOverlaySpec("SAME", font_path=font, font_size=12, fill=(0, 0, 0)),),
    )
    first = finish_image(source, tmp_path / "first.png", spec)
    second = finish_image(source, tmp_path / "second.png", spec)
    assert first.output_sha256 == second.output_sha256
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()


def test_exact_text_must_fit_asset_height_as_well_as_width(tmp_path):
    source = _image(tmp_path / "source.png", size=(100, 4))
    font = _font_path()
    spec = AssetSpec(
        id="short-banner",
        prompt="x",
        width=100,
        height=4,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(
            TextOverlaySpec(
                "EXACT COPY",
                font_path=font,
                font_size=40,
                margin_ratio=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not fit asset canvas"):
        finish_image(source, tmp_path / "too-short.png", spec)


def test_atomic_finishing_failure_preserves_prior_destination(tmp_path, monkeypatch):
    source = _image(tmp_path / "source.png")
    destination = tmp_path / "final.png"
    destination.write_bytes(b"prior-complete-artifact")
    spec = AssetSpec(id="hero", prompt="x", width=120, height=80)

    def fail_replace(_source, _destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr("smythe.assets.finishing.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        finish_image(source, destination, spec)
    assert destination.read_bytes() == b"prior-complete-artifact"
    assert not list(tmp_path.glob(".final.png.*.tmp"))


def test_validation_uses_decoded_content_not_filename_extension(tmp_path):
    misleading = _image(
        tmp_path / "looks-like-png.png",
        size=(100, 100),
        image_format="JPEG",
    )
    spec = AssetSpec(id="hero", prompt="x", width=100, height=100, format="PNG")
    report = validate_image(misleading, spec)
    assert not report.passed
    assert {finding.code for finding in report.hard_findings} == {"content_format_mismatch"}


def test_validation_reports_dimensions_dpi_alpha_and_decode_failures(tmp_path):
    opaque = _image(tmp_path / "opaque.png", size=(50, 40))
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=100,
        height=100,
        dpi=(300, 300),
        alpha_required=True,
    )
    report = validate_image(opaque, spec)
    assert {finding.code for finding in report.hard_findings} == {
        "dimensions_mismatch",
        "dpi_missing",
        "alpha_channel_missing",
    }

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    corrupt_report = validate_image(corrupt, spec)
    assert "decode_failed" in {finding.code for finding in corrupt_report.hard_findings}


def test_receipt_detects_valid_image_replacement(tmp_path):
    source = _image(tmp_path / "source.png")
    destination = tmp_path / "final.png"
    spec = AssetSpec(id="hero", prompt="x", width=120, height=80)
    receipt = finish_image(source, destination, spec)
    _image(destination, size=spec.size, color=(0, 0, 0))

    report = validate_image(destination, spec, receipt=receipt)
    assert not report.passed
    assert "receipt_hash_mismatch" in {finding.code for finding in report.hard_findings}


def test_validation_rejects_degenerate_and_out_of_bounds_overlay_receipt_boxes(tmp_path):
    source = _image(tmp_path / "source.png")
    font = _font_path()
    destination = tmp_path / "final.png"
    spec = AssetSpec(
        id="hero",
        prompt="x",
        width=120,
        height=80,
        text_policy=TextPolicy.COMPOSITE_EXACT,
        text_overlays=(TextOverlaySpec("EXACT", font_path=font, font_size=12),),
    )
    receipt = finish_image(source, destination, spec)
    overlay = receipt.overlays[0]

    outside = replace(receipt, overlays=(replace(overlay, box=(-1, 0, 30, 10)),))
    outside_report = validate_image(destination, spec, receipt=outside)
    assert "overlay_receipt_box_out_of_bounds" in {
        finding.code for finding in outside_report.hard_findings
    }

    degenerate = replace(receipt, overlays=(replace(overlay, box=(1, 1, 1, 10)),))
    degenerate_report = validate_image(destination, spec, receipt=degenerate)
    assert "overlay_receipt_box_invalid" in {
        finding.code for finding in degenerate_report.hard_findings
    }


def test_advisories_are_visible_but_cannot_fail_deterministic_acceptance(tmp_path):
    image = _image(tmp_path / "hero.png", size=(100, 100))
    spec = AssetSpec(id="hero", prompt="x", width=100, height=100)
    judge_note = advisory_finding(
        "vision_brand_score",
        "vision judge scored brand consistency below preference",
        expected=8,
        observed=7,
    )
    report = validate_image(image, spec, advisory_findings=[judge_note])
    assert report.passed
    assert report.advisories == (judge_note,)
    assert not report.hard_findings

    with pytest.raises(ValueError, match="cannot contain hard"):
        validate_image(
            image,
            spec,
            advisory_findings=[hard_finding("not_advisory", "must fail")],
        )


def test_receipt_can_be_required_as_a_hard_gate(tmp_path):
    image = _image(tmp_path / "hero.png", size=(100, 100))
    spec = AssetSpec(id="hero", prompt="x", width=100, height=100)
    report = validate_image(image, spec, require_receipt=True)
    assert not report.passed
    assert [finding.code for finding in report.hard_findings] == ["receipt_missing"]
