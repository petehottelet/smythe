"""Tests for design systems and deterministic aesthetic detectors."""

from __future__ import annotations

import asyncio
import struct
import zlib
from pathlib import Path

import pytest

from smythe import design
from smythe.design import (
    DEFAULT_ANTI_PATTERNS,
    DesignSystem,
    Finding,
    check_blank,
    check_dimensions,
    check_flat_regions,
    check_near_duplicates,
    check_palette,
    design_verifier,
    dhash,
    inspect_asset,
)
from smythe.graph import ExecutionGraph, Node, Topology

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
    # A plain background is legitimate for logos, so this cannot block alone.
    assert findings[0].hard is False

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
# Blank frames
# ---------------------------------------------------------------------------


def _speckled(path: Path, specks: int = 12) -> Path:
    """A black frame with a few white noise pixels in distinct places."""
    from PIL import Image

    img = Image.new("RGB", (256, 256), (0, 0, 0))
    for i in range(specks):
        img.putpixel(((37 * i + 11) % 256, (91 * i + 5) % 256), (255, 255, 255))
    img.save(path, "PNG")
    return path


def _transparent(path: Path) -> Path:
    from PIL import Image

    Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(path, "PNG")
    return path


def _blank_jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (256, 256), (230, 220, 200)).save(path, "JPEG", quality=20)
    return path


def _faint_mark(path: Path) -> Path:
    """A deliberately faint tone-on-tone square, 13 levels off white."""
    from PIL import Image

    canvas = Image.new("RGB", (256, 256), (255, 255, 255))
    canvas.paste((242, 242, 242), (64, 64, 192, 192))
    canvas.save(path, "PNG")
    return path


def _product_on_plain_background(path: Path) -> Path:
    """A shaded product with a soft shadow on a light studio backdrop."""
    from PIL import Image, ImageDraw, ImageFilter

    canvas = Image.new("RGB", (512, 512), (238, 238, 236))
    shadow = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(shadow).ellipse((165, 410, 350, 440), fill=90)
    canvas.paste((150, 150, 150), (0, 0), shadow.filter(ImageFilter.GaussianBlur(9)))
    shade = Image.linear_gradient("L").rotate(90).resize((130, 260))
    blue = Image.new("L", shade.size, 160)
    canvas.paste(Image.merge("RGB", (shade, shade, blue)), (191, 150))
    canvas.save(path, "PNG")
    return path


def _black_logo_on_transparency(path: Path) -> Path:
    """Transparent pixels store black too, so only alpha shows the mark."""
    from PIL import Image, ImageDraw

    logo = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(logo).ellipse((64, 64, 192, 192), fill=(0, 0, 0, 255))
    logo.save(path, "PNG")
    return path


def test_blank_detector_names_the_colour_and_rounds_down(tmp_path):
    [black] = check_blank(_image(tmp_path / "black.png", size=(256, 256), color=(0, 0, 0)))
    assert (black.detector, black.hard) == ("blank-image", True)
    assert black.message.startswith("100% of the image is one flat colour (#000000)")

    # One speck leaves 4095 of 4096 cells uniform: 99.9%, not rounded up to 100%.
    [speckled] = check_blank(_speckled(tmp_path / "speckled.png", specks=1))
    assert speckled.message.startswith("99.9% of the image is one flat colour (#000000)")

    [clear] = check_blank(_transparent(tmp_path / "clear.png"))
    assert "(fully transparent)" in clear.message


def test_blank_detector_ignores_compression_noise_but_not_faint_content(tmp_path):
    assert [f.detector for f in check_blank(_blank_jpeg(tmp_path / "blank.jpg"))] == [
        "blank-image"
    ]
    faint = _faint_mark(tmp_path / "faint.png")
    assert check_blank(faint) == []
    assert check_blank(faint, tolerance=16) != []


def test_blank_detector_thresholds_are_tunable(tmp_path):
    speckled = _speckled(tmp_path / "speckled.png")
    assert check_blank(speckled, min_uniform=0.999) == []
    logo = _black_logo_on_transparency(tmp_path / "logo.png")
    assert check_blank(logo) == []
    assert check_blank(logo, min_uniform=0.5) != []


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
    wrong_size = _image(tmp_path / "wrong.png", noise=3)
    verifier = design_verifier(width=128, height=128)
    verdict = verifier.verdict(Node(id="v", label="v"), _target_with([wrong_size]))
    assert verdict.passed is False
    assert "dimensions" in verdict.reason


def _logo_on_white(path: Path) -> Path:
    from PIL import Image

    canvas = Image.new("RGB", (256, 256), (255, 255, 255))
    with Image.open(_image(path.with_name("mark.png"), size=(96, 96), noise=5)) as mark:
        canvas.paste(mark, (80, 80))
    canvas.save(path, "PNG")
    return path


def test_verifier_passes_a_logo_on_a_white_background(tmp_path):
    """Regression: a plain background used to force a paid regeneration."""
    logo = _logo_on_white(tmp_path / "logo.png")
    assert [f.detector for f in inspect_asset(logo)] == ["flat-region"]

    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([logo]))
    assert verdict.passed is True

    strict = design_verifier(include_advisory=True)
    strict_verdict = strict.verdict(Node(id="v", label="v"), _target_with([logo]))
    assert strict_verdict.passed is False
    assert "[advisory] flat-region" in strict_verdict.reason


BLANK_FRAMES = {
    "black.png": lambda path: _image(path, size=(256, 256), color=(0, 0, 0)),
    "white.png": lambda path: _image(path, size=(256, 256), color=(255, 255, 255)),
    "near-blank.png": _speckled,
    "blank.jpg": _blank_jpeg,
    "transparent.png": _transparent,
}


@pytest.mark.parametrize("name", sorted(BLANK_FRAMES))
def test_verifier_fails_a_blank_frame(tmp_path, name):
    """Regression: a correctly sized blank frame, what a provider can return
    when a safety filter trips, passed once flat regions became advisory."""
    path = BLANK_FRAMES[name](tmp_path / name)
    verifier = design_verifier(width=256, height=256)
    verdict = verifier.verdict(Node(id="v", label="v"), _target_with([path]))
    assert verdict.passed is False
    assert verdict.reason.startswith("[hard] blank-image: ")


CONTENT_ON_PLAIN_BACKGROUNDS = {
    "logo-on-white.png": _logo_on_white,
    "product.png": _product_on_plain_background,
    "logo-on-transparency.png": _black_logo_on_transparency,
}


@pytest.mark.parametrize("name", sorted(CONTENT_ON_PLAIN_BACKGROUNDS))
def test_verifier_passes_content_on_a_plain_background(tmp_path, name):
    path = CONTENT_ON_PLAIN_BACKGROUNDS[name](tmp_path / name)
    assert check_blank(path) == []
    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([path]))
    assert verdict.passed is True


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


def test_verifier_fails_when_the_only_artifact_is_missing(tmp_path):
    """Regression: a missing file was skipped and the verdict passed."""
    target = _target_with([tmp_path / "gone.png"])
    verdict = design_verifier().verdict(Node(id="v", label="v"), target)
    assert verdict.passed is False
    assert "[hard] missing-artifact: gone.png" in verdict.reason


def test_verifier_reports_one_missing_artifact_among_several(tmp_path):
    """Regression: near-duplicate hashing raised FileNotFoundError."""
    a = _image(tmp_path / "a.png", noise=3)
    b = _image(tmp_path / "b.png", color=(10, 200, 10), noise=17)
    target = _target_with([a, tmp_path / "gone.png", b])
    verdict = design_verifier().verdict(Node(id="v", label="v"), target)
    assert verdict.passed is False
    assert verdict.reason == "[hard] missing-artifact: gone.png does not exist"


def _spy_on_plugin_open(monkeypatch, plugin_class):
    calls = []
    original = plugin_class._open

    def spy(self):
        calls.append(plugin_class.__name__)
        return original(self)

    monkeypatch.setattr(plugin_class, "_open", spy)
    return calls


def test_detectors_refuse_formats_outside_png_jpeg_gif_webp(tmp_path, monkeypatch):
    from PIL import EpsImagePlugin, Image, TiffImagePlugin, UnidentifiedImageError

    calls = _spy_on_plugin_open(monkeypatch, EpsImagePlugin.EpsImageFile)
    calls += _spy_on_plugin_open(monkeypatch, TiffImagePlugin.TiffImageFile)
    tiff = tmp_path / "tiff.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(tiff, format="TIFF")
    eps = tmp_path / "eps.png"
    eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 64 64\nshowpage\n")

    for path in (tiff, eps):
        with pytest.raises(UnidentifiedImageError, match="PNG, JPEG, GIF, or WebP"):
            check_dimensions(path, 64, 64)
        with pytest.raises(UnidentifiedImageError):
            dhash(path)
        verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([path]))
        assert verdict.passed is False
        assert f"[hard] unreadable-artifact: {path.name}" in verdict.reason
    assert calls == []


def test_verifier_reports_a_decompression_bomb_as_a_failure(tmp_path):
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(
            ">I", zlib.crc32(kind + body)
        )

    header = struct.pack(">IIBBBBB", 10_000, 10_000, 8, 2, 0, 0, 0)
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b""))

    with pytest.warns(Warning):  # Pillow warns first; Smythe then refuses.
        verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([bomb]))
    assert verdict.passed is False
    assert "unreadable-artifact: bomb.png" in verdict.reason
    assert "pixel limit" in verdict.reason


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(
        ">I", zlib.crc32(kind + body)
    )


def _png(*chunks: bytes) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _ihdr(colour_type: int = 2) -> bytes:
    return _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, colour_type, 0, 0, 0))


def _idat(channels: int = 3) -> bytes:
    rows = b"".join(b"\x00" + bytes(range(8 * channels)) for _ in range(8))
    return _png_chunk(b"IDAT", zlib.compress(rows))


def _text_bomb() -> bytes:
    """A 1 KB zTXt chunk that inflates just past Pillow's text limit."""
    from PIL import PngImagePlugin

    inflated = b"A" * (PngImagePlugin.MAX_TEXT_CHUNK + 1)
    return _png_chunk(b"zTXt", b"Comment\x00\x00" + zlib.compress(inflated))


_IEND = _png_chunk(b"IEND", b"")
# CRC-valid PNGs that Pillow rejects with something other than OSError.
MALFORMED_PNGS = {
    # ValueError while opening.
    "truncated-ihdr": lambda: _png(
        _png_chunk(b"IHDR", struct.pack(">IIBBBB", 8, 8, 8, 2, 0, 0)), _IEND
    ),
    "text-bomb-before-data": lambda: _png(_ihdr(), _text_bomb(), _idat(), _IEND),
    # ValueError while loading: the chunk follows the image data.
    "text-bomb-after-data": lambda: _png(_ihdr(), _idat(), _text_bomb(), _IEND),
    # struct.error while loading.
    "short-chrm-after-data": lambda: _png(
        _ihdr(), _idat(), _png_chunk(b"cHRM", b"\x00" * 7), _IEND
    ),
    # AssertionError converting (AttributeError under python -O).
    "palette-without-plte": lambda: _png(
        _ihdr(colour_type=3), _png_chunk(b"tRNS", b"\x00"), _idat(channels=1), _IEND
    ),
}


@pytest.mark.parametrize("name", sorted(MALFORMED_PNGS))
def test_verifier_reports_malformed_images_as_unreadable(tmp_path, name):
    """Regression: only OSError, SyntaxError and DecompressionBombError were
    caught, so these escaped and aborted the run after both paid calls."""
    path = tmp_path / f"{name}.png"
    path.write_bytes(MALFORMED_PNGS[name]())
    verdict = design_verifier().verdict(Node(id="v", label="v"), _target_with([path]))
    assert verdict.passed is False
    assert verdict.reason.startswith(f"[hard] unreadable-artifact: {name}.png: ")


def test_a_malformed_image_is_regenerated_instead_of_aborting_the_run(tmp_path):
    """Regression: the verdict raised ValueError and the run aborted."""
    from smythe.async_executor import AsyncExecutor
    from smythe.provider import Artifact, CompletionResult, Provider
    from smythe.registry import Registry
    from smythe.tracer import Tracer

    malformed = MALFORMED_PNGS["truncated-ihdr"]()

    class ImageProvider(Provider):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, system, prompt, model):
            label = prompt.splitlines()[0].strip()
            self.calls.append(label)
            artifacts = [Artifact(data=malformed, mime_type="image/png")] if label == "draw" else []
            return CompletionResult(
                text="done", artifacts=artifacts, prompt_tokens=1, completion_tokens=1
            )

    draw = Node(id="draw", label="draw")
    judge = Node(
        id="judge", label="judge", depends_on=["draw"], verifies="draw", max_regenerations=1
    )
    for node in (draw, judge):
        node.metadata["model"] = "test-model"
    provider, tracer = ImageProvider(), Tracer()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=tracer,
        artifact_dir=tmp_path / "artifacts", verifier=design_verifier(),
    )

    asyncio.run(executor.run(ExecutionGraph(topology=[Topology.SERIAL], nodes=[draw, judge])))

    assert provider.calls == ["draw", "judge", "draw", "judge"]
    [regeneration] = [s for s in tracer.summary() if s["status"] == "regeneration"]
    assert regeneration["label"].startswith("[hard] unreadable-artifact: draw_00.png: ")


def test_verifier_does_not_disguise_its_own_errors_as_unreadable_files(tmp_path, monkeypatch):
    good = _image(tmp_path / "good.png", noise=3)
    target = _target_with([good])
    # A misconfigured palette is the caller's error, not the artifact's.
    with pytest.raises(ValueError, match="6-digit hex"):
        design_verifier(DesignSystem(palette=["#fff"])).verdict(Node(id="v", label="v"), target)

    def broken_detector(*args, **kwargs):
        raise TypeError("detector bug")

    monkeypatch.setattr(design, "_flat_region_findings", broken_detector)
    with pytest.raises(TypeError, match="detector bug"):
        design_verifier().verdict(Node(id="v", label="v"), target)


@pytest.mark.parametrize("error", [ImportError, MemoryError])
def test_verifier_propagates_environment_failures(tmp_path, monkeypatch, error):
    good = _image(tmp_path / "good.png", noise=3)

    def unavailable(*args, **kwargs):
        raise error("environment failure")

    monkeypatch.setattr(design, "open_image", unavailable)
    with pytest.raises(error):
        design_verifier().verdict(Node(id="v", label="v"), _target_with([good]))


def test_finding_str_marks_severity():
    assert str(Finding("d", "m")).startswith("[hard]")
    assert str(Finding("d", "m", hard=False)).startswith("[advisory]")
