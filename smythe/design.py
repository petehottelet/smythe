"""Design systems and deterministic aesthetic detectors.

Models regress to the mean aesthetically: trained on the same
templates, they reach for the same fonts, the same gradients, the same
centered card. Telling a model to "make it premium" does not fix that,
because the model already believes it is.

Two things do, and neither needs a bigger model:

**A durable design system.** Palette, type, density, and — most
usefully — the *anti-patterns* to avoid. A brief re-pasted per task
drifts; a design system every node inherits does not. The measured
version of this is brand-locking: handing the real logo to every node
raised consistency far more than any prompt wording did.

**Deterministic detectors.** Most defects need no judgment at all.
Wrong dimensions, a blank frame, off-palette pixels, near-duplicate
outputs, an unfilled placeholder rectangle — these are arithmetic.
Running them before an LLM judge is cheaper, faster, has zero variance,
and cannot be talked out of a verdict. The judge is then reserved for
what only judgment can settle.

Detectors plug into the verification tier (``smythe/verifier.py``), so
a hard finding sends work back for regeneration like any other failed
check::

    from smythe.design import DesignSystem, design_verifier

    system = DesignSystem(
        palette=["#f5b301", "#1a1a1a", "#faf7f0"],
        anti_patterns=["stock-photo handshake", "centered everything"],
    )
    swarm = Swarm(verifier=design_verifier(system), ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from smythe._images import open_image
from smythe.verifier import CallableVerifier, Verdict

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Anti-patterns worth avoiding by default. These are the shapes models
# fall into unprompted, not universal design sins — override freely.
DEFAULT_ANTI_PATTERNS = (
    "purple-to-blue gradient backgrounds",
    "centered hero text over a full-bleed stock photograph",
    "nested cards inside cards",
    "text set too small to read at the asset's real display size",
    "generic stock imagery with no relationship to the subject",
)

# Each detector reads a small sample of the decoded image, so its cost does
# not grow with the image once decoding is done.
_PALETTE_SAMPLE = (64, 64)
_FLAT_GRID = 12
_HASH_SAMPLE = (9, 8)
# Blank-frame test; the rationale is in check_blank's docstring.
_BLANK_GRID = 64
_BLANK_TOLERANCE = 8
_BLANK_MIN_UNIFORM = 0.995


def _clamp_dial(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int from 1 to 10, got {value!r}")
    if not 1 <= value <= 10:
        raise ValueError(f"{name} must be between 1 and 10, got {value}")
    return value


@dataclass(frozen=True)
class DesignSystem:
    """A durable design brief every node in a run inherits.

    The dials exist because "make it good" is unfalsifiable while
    ``variance=8, density=3`` is a specification: it can be held
    constant across forty assets so they feel like one artifact, or
    varied deliberately between them.

    Attributes:
        palette: Brand hex colours, most important first.
        typography: How type should be set.
        motion: 1-10, how much movement (for web/animated output).
        variance: 1-10, how far layouts may depart from the conventional.
        density: 1-10, how much information per viewport.
        anti_patterns: What to avoid. Usually more actionable than
            what to aim for, because the model already thinks it is
            aiming there.
        notes: Anything else that belongs in every prompt.
    """

    palette: tuple[str, ...] = ()
    typography: str = ""
    motion: int = 5
    variance: int = 5
    density: int = 5
    anti_patterns: tuple[str, ...] = DEFAULT_ANTI_PATTERNS
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "palette", tuple(self.palette))
        object.__setattr__(self, "anti_patterns", tuple(self.anti_patterns))
        for dial in ("motion", "variance", "density"):
            object.__setattr__(self, dial, _clamp_dial(dial, getattr(self, dial)))

    def to_prompt(self) -> str:
        """Render the system as prompt text to append to a node's label."""
        lines: list[str] = ["Design system (applies to everything you produce):"]
        if self.palette:
            lines.append(f"- Palette, in priority order: {', '.join(self.palette)}")
        if self.typography:
            lines.append(f"- Typography: {self.typography}")
        lines.append(
            f"- Layout variance {self.variance}/10 "
            f"({'conventional and safe' if self.variance <= 3 else 'experimental' if self.variance >= 8 else 'considered but not showy'})"
        )
        lines.append(
            f"- Information density {self.density}/10 "
            f"({'sparse, generous whitespace' if self.density <= 3 else 'dense' if self.density >= 8 else 'balanced'})"
        )
        if self.motion != 5:
            lines.append(f"- Motion intensity {self.motion}/10")
        if self.anti_patterns:
            lines.append("- Avoid, specifically:")
            lines.extend(f"  - {item}" for item in self.anti_patterns)
        if self.notes:
            lines.append(f"- {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Finding:
    """One deterministic observation about a produced asset."""

    detector: str
    message: str
    hard: bool = True

    def __str__(self) -> str:
        return f"[{'hard' if self.hard else 'advisory'}] {self.detector}: {self.message}"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"palette colour must be a 6-digit hex, got {value!r}")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def check_dimensions(path: Path, width: int, height: int) -> list[Finding]:
    """Exact-size compliance. No model can measure this better."""
    with open_image(path) as img:
        size = img.size
    return _dimension_findings(size, width, height)


def _dimension_findings(size: tuple[int, int], width: int, height: int) -> list[Finding]:
    if size != (width, height):
        return [Finding(
            "dimensions",
            f"expected {width}x{height}, got {size[0]}x{size[1]}",
        )]
    return []


def check_palette(
    path: Path,
    palette: tuple[str, ...] | list[str],
    *,
    tolerance: int = 60,
    min_on_brand: float = 0.45,
) -> list[Finding]:
    """Fraction of pixels near a brand colour.

    Sampled on a thumbnail: exact pixel accounting is neither necessary
    nor meaningful for photographic output.
    """
    if not palette:
        return []
    targets = [_hex_to_rgb(c) for c in palette]
    with open_image(path) as img:
        sample = img.convert("RGB").resize(_PALETTE_SAMPLE)
    return _palette_findings(sample, targets, tolerance=tolerance, min_on_brand=min_on_brand)


def _palette_findings(
    sample: PILImage,
    targets: list[tuple[int, int, int]],
    *,
    tolerance: int = 60,
    min_on_brand: float = 0.45,
) -> list[Finding]:
    raw = sample.tobytes()
    pixels = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    on_brand = 0
    for r, g, b in pixels:
        for tr, tg, tb in targets:
            if abs(r - tr) + abs(g - tg) + abs(b - tb) <= tolerance * 3:
                on_brand += 1
                break
    ratio = on_brand / len(pixels)
    if ratio < min_on_brand:
        return [Finding(
            "palette",
            f"only {ratio:.0%} of pixels are near the brand palette "
            f"(wanted {min_on_brand:.0%})",
            hard=False,
        )]
    return []


def check_blank(
    path: Path,
    *,
    tolerance: int = _BLANK_TOLERANCE,
    min_uniform: float = _BLANK_MIN_UNIFORM,
) -> list[Finding]:
    """Flag an image that is one flat colour: a blank frame.

    A generation that trips a provider's safety filter or fails partway
    can still return a correctly sized file that is entirely black or
    white, so this finding is hard. The image is area-averaged onto a
    64x64 grid, so each cell is the mean of its pixels and scattered
    noise, dithering and compression speckle average away. The image is
    blank when at least ``min_uniform`` of the cells (99.5%) are within
    ``tolerance`` levels (8 of 255) of its median colour in every
    channel. Cells are compared premultiplied by alpha: a fully
    transparent image is blank, and a black logo on a transparent
    background is not.

    With the defaults, a frame whose only content covers less than about
    0.5% of its area, such as a few stray pixels or a small corner
    watermark, is blank. A logo or product on a plain background covers
    far more and at most earns the advisory flat-region finding. Eight
    levels, about 3% of the range, is more than JPEG artifacts, GIF
    dithering or sensor-like noise leave on a flat field once averaged,
    and less than the contrast of even a faint tone-on-tone mark.
    """
    with open_image(path) as img:
        sample = _blank_sample(img)
    return _blank_findings(sample, tolerance=tolerance, min_uniform=min_uniform)


def _blank_sample(image: PILImage) -> PILImage:
    """Area-average *image* onto the blank-test grid, keeping transparency."""
    from PIL import Image

    has_alpha = "A" in image.getbands() or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB").resize(
        (_BLANK_GRID, _BLANK_GRID), Image.Resampling.BOX
    )


def _blank_findings(
    sample: PILImage,
    *,
    tolerance: int = _BLANK_TOLERANCE,
    min_uniform: float = _BLANK_MIN_UNIFORM,
) -> list[Finding]:
    raw = sample.tobytes()
    if sample.mode == "RGBA":
        # Premultiplied, a transparent cell hides whatever colour it stores.
        cells = [
            ((r * a + 127) // 255, (g * a + 127) // 255, (b * a + 127) // 255, a)
            for r, g, b, a in zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4])
        ]
    else:
        cells = [(r, g, b, 255) for r, g, b in zip(raw[0::3], raw[1::3], raw[2::3])]
    reference = tuple(sorted(channel)[len(channel) // 2] for channel in zip(*cells))
    uniform = sum(
        1
        for cell in cells
        if max(abs(value - ref) for value, ref in zip(cell, reference)) <= tolerance
    )
    if uniform / len(cells) < min_uniform:
        return []
    # Rounded down, so a near-blank frame never reads as 100%.
    percent = uniform * 1000 // len(cells) / 10
    return [Finding(
        "blank-image",
        f"{percent:g}% of the image is one flat colour ({_colour_name(reference)}) "
        "— a blank frame with no content",
    )]


def _colour_name(premultiplied: tuple[int, ...]) -> str:
    r, g, b, a = premultiplied
    if a == 0:
        return "fully transparent"
    if a < 255:
        r, g, b = (min(255, (c * 255 + a // 2) // a) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x} at {a * 100 // 255}% opacity"
    return f"#{r:02x}{g:02x}{b:02x}"


def check_flat_regions(
    path: Path, *, min_fraction: float = 0.06, grid: int = _FLAT_GRID,
) -> list[Finding]:
    """Find large, perfectly uniform rectangles.

    These are often an unfilled placeholder — the empty box a model
    leaves when told to reserve space for typography it was asked not
    to render. Photographic content is never this uniform, but a logo,
    icon, or product shot on a plain background legitimately is, so the
    finding is advisory: ``design_verifier`` blocks on it only with
    ``include_advisory=True``. An image with no content at all is the
    separate, hard finding from :func:`check_blank`.
    """
    with open_image(path) as img:
        sample = img.convert("RGB").resize((grid * 8, grid * 8))
    return _flat_region_findings(sample, min_fraction=min_fraction, grid=grid)


def _flat_region_findings(
    sample: PILImage, *, min_fraction: float = 0.06, grid: int = _FLAT_GRID,
) -> list[Finding]:
    flat_cells = 0
    for row in range(grid):
        for col in range(grid):
            box = (col * 8, row * 8, col * 8 + 8, row * 8 + 8)
            raw = sample.crop(box).tobytes()
            cell = {raw[i:i + 3] for i in range(0, len(raw), 3)}
            if len(cell) == 1:
                flat_cells += 1
    fraction = flat_cells / (grid * grid)
    if fraction >= min_fraction:
        return [Finding(
            "flat-region",
            f"{fraction:.0%} of the image is perfectly uniform — likely an "
            "unfilled placeholder rather than composition",
            hard=False,
        )]
    return []


def dhash(path: Path) -> int:
    """64-bit difference hash, for near-duplicate detection."""
    with open_image(path) as img:
        sample = img.convert("L").resize(_HASH_SAMPLE)
    return _dhash_bits(sample)


def _dhash_bits(sample: PILImage) -> int:
    # Grayscale tobytes() is one byte per pixel, so it indexes directly.
    px = sample.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def check_near_duplicates(
    paths: list[Path], *, min_distance: int = 8,
) -> list[Finding]:
    """Flag pairs that are too similar to be distinct deliverables.

    Wide fan-out occasionally returns the same picture twice from
    different prompts; the measured rate in this project's own image
    benchmark was one near-duplicate pair in a run of eight.
    """
    return _near_duplicate_findings([(p, dhash(p)) for p in paths], min_distance=min_distance)


def _near_duplicate_findings(
    hashes: list[tuple[Path, int]], *, min_distance: int = 8,
) -> list[Finding]:
    findings: list[Finding] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = bin(hashes[i][1] ^ hashes[j][1]).count("1")
            if distance < min_distance:
                findings.append(Finding(
                    "near-duplicate",
                    f"{hashes[i][0].name} and {hashes[j][0].name} differ by only "
                    f"{distance} bits (wanted >= {min_distance})",
                ))
    return findings


@dataclass(frozen=True)
class _Samples:
    """The small samples of one decoded image that the detectors read."""

    size: tuple[int, int]
    palette: PILImage
    flat: PILImage
    blank: PILImage
    gray: PILImage


def _decode_samples(path: Path) -> _Samples:
    """Decode *path* once and take every detector's sample from it.

    This is the only step that parses the file. Only one full-size
    conversion is alive at a time, so peak memory matches what a single
    detector needs on its own.
    """
    with open_image(path) as img:
        img.load()
        rgb = img.convert("RGB")
        palette = rgb.resize(_PALETTE_SAMPLE)
        flat = rgb.resize((_FLAT_GRID * 8, _FLAT_GRID * 8))
        del rgb
        return _Samples(
            size=img.size,
            palette=palette,
            flat=flat,
            blank=_blank_sample(img),
            gray=img.convert("L").resize(_HASH_SAMPLE),
        )


def _inspect_samples(
    samples: _Samples,
    *,
    system: DesignSystem | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    if width is not None and height is not None:
        findings.extend(_dimension_findings(samples.size, width, height))
    findings.extend(_blank_findings(samples.blank))
    findings.extend(_flat_region_findings(samples.flat))
    if system is not None and system.palette:
        targets = [_hex_to_rgb(c) for c in system.palette]
        findings.extend(_palette_findings(samples.palette, targets))
    return findings


def inspect_asset(
    path: str | Path,
    *,
    system: DesignSystem | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Finding]:
    """Run every applicable deterministic detector over one asset.

    The image is decoded once. An error decoding it propagates;
    ``design_verifier`` reports it as an ``unreadable-artifact`` finding.
    """
    return _inspect_samples(
        _decode_samples(Path(path)), system=system, width=width, height=height
    )


def design_verifier(
    system: DesignSystem | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    include_advisory: bool = False,
) -> CallableVerifier:
    """A verifier that gates on deterministic findings alone.

    Cheap, instant, and free of judge variance — run this before
    spending a model call on aesthetic judgment. Hard findings fail the
    verdict: wrong dimensions, a blank image (see :func:`check_blank`),
    near-duplicate artifacts, and a listed artifact that is missing or
    cannot be decoded as a PNG, JPEG, GIF, or WebP image. Advisory
    findings (palette, flat-region) are reported but do not fail by
    default, because a soft palette miss or a plain background is not
    worth paying to regenerate; ``include_advisory=True`` makes them
    block too.
    """

    def verdict(verifier_node, target) -> Verdict:
        records = target.metadata.get("artifacts") or []
        paths = [Path(r["path"]) for r in records if r.get("path")]
        if not paths:
            return Verdict(passed=True, reason="no artifacts to inspect")

        findings: list[Finding] = []
        hashes: list[tuple[Path, int]] = []
        for path in paths:
            if not path.exists():
                findings.append(Finding("missing-artifact", f"{path.name} does not exist"))
                continue
            try:
                samples = _decode_samples(path)
            except (ImportError, MemoryError):
                raise  # A missing Pillow or an exhausted host is not a bad file.
            except Exception as exc:
                # Pillow has no closed set of errors for malformed input.
                # Besides OSError and SyntaxError, fuzzing PNG, JPEG, GIF and
                # WebP files through this step raised ValueError (a truncated
                # IHDR chunk, a text or ICC chunk that inflates past Pillow's
                # limit), struct.error (a short cHRM chunk after the image
                # data) and a bare AssertionError from one of Pillow's own
                # asserts, which is an AttributeError under ``python -O``.
                # Only the restricted open and Pillow's decoding, conversion
                # and downsampling of this one file run inside the try, so
                # any failure there means the artifact is unreadable. The
                # detectors run below, outside it, so a bug in Smythe's own
                # checks still raises.
                detail = str(exc) or type(exc).__name__
                findings.append(Finding("unreadable-artifact", f"{path.name}: {detail}"))
                continue
            findings.extend(
                _inspect_samples(samples, system=system, width=width, height=height)
            )
            hashes.append((path, _dhash_bits(samples.gray)))
        if len(hashes) > 1:
            findings.extend(_near_duplicate_findings(hashes))

        blocking = [f for f in findings if f.hard or include_advisory]
        if not blocking:
            return Verdict(passed=True, reason="")
        return Verdict(
            passed=False,
            reason="; ".join(str(f) for f in blocking[:4]),
        )

    return CallableVerifier(verdict)
