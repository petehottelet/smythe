"""Tests for design systems and deterministic aesthetic detectors."""

from __future__ import annotations

from pathlib import Path

import pytest

from smythe.design import (
    DEFAULT_ANTI_PATTERNS,
    DesignSystem,
    Finding,
    check_dimensions,
    check_flat_regions,
    check_near_duplicates,
    check_palette,
    design_verifier,
    dhash,
    inspect_asset,
)
from smythe.graph import Node

PIL = pytest.importorskip("PIL")


def _image(path: Path, size=(64, 64), color=(200, 120, 40), noise: int = 0):
    from PIL import Image

    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for x in range(size[0]):
            for y in range(size[1]):
                r, g, b = px[x, y]
                shift = ((x * 7 + y * 13 + noise) % 41) - 20
                px[x, y] = (
                    max(0, min(255, r + shift)),
                    max(0, min(255, g + shift)),
                    max(0, min(255, b + shift)),
                )
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# DesignSystem
# ---------------------------------------------------------------------------


def test_design_system_renders_prompt_text():
    system = DesignSystem(
        palette=["#f5b301", "#1a1a1a"],
        typography="Serif headlines, generous leading",
        variance=8,
        density=2,
        anti_patterns=["purple gradients"],
        notes="Print-safe margins",
    )
    text = system.to_prompt()
    assert "#f5b301" in text
    assert "Serif headlines" in text
    assert "8/10" in text and "experimental" in text
    assert "sparse" in text
    assert "purple gradients" in text
    assert "Print-safe margins" in text


def test_anti_patterns_default_to_the_common_model_failures():
    assert DesignSystem().anti_patterns == DEFAULT_ANTI_PATTERNS
    assert "purple-to-blue gradient backgrounds" in DesignSystem().to_prompt()


def test_dials_are_validated():
    for bad in (0, 11, -1):
        with pytest.raises(ValueError, match="between 1 and 10"):
            DesignSystem(variance=bad)
    with pytest.raises(TypeError):
        DesignSystem(density="high")


def test_design_system_is_immutable_and_normalized():
    system = DesignSystem(palette=["#000000"], anti_patterns=["x"])
    assert isinstance(system.palette, tuple)
    assert isinstance(system.anti_patterns, tuple)
    with pytest.raises(Exception):
        system.variance = 9  # frozen


def test_motion_only_appears_when_set():
    assert "Motion intensity" not in DesignSystem().to_prompt()
    assert "Motion intensity 9/10" in DesignSystem(motion=9).to_prompt()


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def test_dimension_detector(tmp_path):
    path = _image(tmp_path / "a.png", size=(100, 50))
    assert check_dimensions(path, 100, 50) == []
    findings = check_dimensions(path, 200, 50)
    assert findings and "expected 200x50" in findings[0].message


def test_flat_region_detector_catches_placeholder_boxes(tmp_path):
    """The unfilled white box a model leaves when reserving space."""
    flat = _image(tmp_path / "flat.png")
    findings = check_flat_regions(flat)
    assert findings and findings[0].detector == "flat-region"
    assert findings[0].hard is True

    textured = _image(tmp_path / "textured.png", noise=3)
    assert check_flat_regions(textured) == []


def test_palette_detector_is_advisory(tmp_path):
    on_brand = _image(tmp_path / "on.png", color=(245, 179, 1))
    assert check_palette(on_brand, ["#f5b301"]) == []

    off_brand = _image(tmp_path / "off.png", color=(10, 10, 200))
    findings = check_palette(off_brand, ["#f5b301"])
    assert findings and findings[0].hard is False, "a palette miss should not block"


def test_palette_detector_noop_without_palette(tmp_path):
    assert check_palette(_image(tmp_path / "x.png"), []) == []


def test_near_duplicate_detector(tmp_path):
    a = _image(tmp_path / "a.png", color=(200, 120, 40), noise=1)
    b = _image(tmp_path / "b.png", color=(200, 120, 40), noise=1)
    findings = check_near_duplicates([a, b])
    assert findings and findings[0].detector == "near-duplicate"

    c = _image(tmp_path / "c.png", color=(10, 200, 10), noise=17)
    assert check_near_duplicates([a, c], min_distance=1) == []


def test_dhash_is_stable_and_discriminating(tmp_path):
    a = _image(tmp_path / "a.png", noise=5)
    assert dhash(a) == dhash(a)
    b = _image(tmp_path / "b.png", color=(5, 5, 5), noise=31)
    assert dhash(a) != dhash(b)


def test_inspect_asset_runs_applicable_detectors(tmp_path):
    path = _image(tmp_path / "a.png", size=(64, 64))
    findings = inspect_asset(path, width=99, height=99)
    detectors = {f.detector for f in findings}
    assert "dimensions" in detectors
    assert "flat-region" in detectors


# ---------------------------------------------------------------------------
# design_verifier — detectors as a gate
# ---------------------------------------------------------------------------


def _target_with(paths):
    node = Node(id="gen", label="generate")
    node.metadata["artifacts"] = [
        {"path": str(p), "mime_type": "image/png"} for p in paths
    ]
    return node


def test_verifier_passes_clean_assets(tmp_path):
    good = _image(tmp_path / "good.png", noise=3)
    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([good]))
    assert verdict.passed is True


def test_verifier_fails_on_a_hard_finding(tmp_path):
    flat = _image(tmp_path / "flat.png")
    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([flat]))
    assert verdict.passed is False
    assert "flat-region" in verdict.reason


def test_verifier_ignores_advisory_findings_by_default(tmp_path):
    off_brand = _image(tmp_path / "off.png", color=(10, 10, 200), noise=3)
    system = DesignSystem(palette=["#f5b301"])
    target = _target_with([off_brand])
    assert design_verifier(system).verdict(Node(id="v", label="v"), target).passed
    strict = design_verifier(system, include_advisory=True)
    assert strict.verdict(Node(id="v", label="v"), target).passed is False


def test_verifier_catches_near_duplicates_across_a_set(tmp_path):
    a = _image(tmp_path / "a.png", color=(200, 120, 40), noise=1)
    b = _image(tmp_path / "b.png", color=(200, 120, 40), noise=1)
    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([a, b]))
    assert verdict.passed is False
    assert "near-duplicate" in verdict.reason


def test_verifier_passes_when_there_is_nothing_to_inspect():
    verdict = design_verifier().verdict(Node(id="v", label="v"), Node(id="t", label="t"))
    assert verdict.passed is True


def test_verifier_tolerates_missing_files(tmp_path):
    target = _target_with([tmp_path / "gone.png"])
    assert design_verifier().verdict(Node(id="v", label="v"), target).passed is True


def test_finding_str_marks_severity():
    assert str(Finding("d", "m")).startswith("[hard]")
    assert str(Finding("d", "m", hard=False)).startswith("[advisory]")
