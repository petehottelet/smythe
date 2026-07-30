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
Wrong dimensions, off-palette pixels, near-duplicate outputs, an
unfilled placeholder rectangle — these are arithmetic. Running them
before an LLM judge is cheaper, faster, has zero variance, and cannot
be talked out of a verdict. The judge is then reserved for what only
judgment can settle.

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

from smythe.verifier import CallableVerifier, Verdict

# Anti-patterns worth avoiding by default. These are the shapes models
# fall into unprompted, not universal design sins — override freely.
DEFAULT_ANTI_PATTERNS = (
    "purple-to-blue gradient backgrounds",
    "centered hero text over a full-bleed stock photograph",
    "nested cards inside cards",
    "text set too small to read at the asset's real display size",
    "generic stock imagery with no relationship to the subject",
)


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
    from PIL import Image

    with Image.open(path) as img:
        size = img.size
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
    from PIL import Image

    if not palette:
        return []
    targets = [_hex_to_rgb(c) for c in palette]
    with Image.open(path) as img:
        thumb = img.convert("RGB").resize((64, 64))
        raw = thumb.tobytes()
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


def check_flat_regions(
    path: Path, *, min_fraction: float = 0.06, grid: int = 12,
) -> list[Finding]:
    """Find large, perfectly uniform rectangles.

    These are usually an unfilled placeholder — the empty box a model
    leaves when told to reserve space for typography it was asked not
    to render. Photographic content is never this uniform.
    """
    from PIL import Image

    with Image.open(path) as img:
        thumb = img.convert("RGB").resize((grid * 8, grid * 8))
    flat_cells = 0
    for row in range(grid):
        for col in range(grid):
            box = (col * 8, row * 8, col * 8 + 8, row * 8 + 8)
            raw = thumb.crop(box).tobytes()
            cell = {raw[i:i + 3] for i in range(0, len(raw), 3)}
            if len(cell) == 1:
                flat_cells += 1
    fraction = flat_cells / (grid * grid)
    if fraction >= min_fraction:
        return [Finding(
            "flat-region",
            f"{fraction:.0%} of the image is perfectly uniform — likely an "
            "unfilled placeholder rather than composition",
        )]
    return []


def dhash(path: Path) -> int:
    """64-bit difference hash, for near-duplicate detection."""
    from PIL import Image

    with Image.open(path) as img:
        # Grayscale tobytes() is one byte per pixel, so it indexes directly.
        px = img.convert("L").resize((9, 8)).tobytes()
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
    findings: list[Finding] = []
    hashes = [(p, dhash(p)) for p in paths]
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


def inspect_asset(
    path: str | Path,
    *,
    system: DesignSystem | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[Finding]:
    """Run every applicable deterministic detector over one asset."""
    target = Path(path)
    findings: list[Finding] = []
    if width is not None and height is not None:
        findings.extend(check_dimensions(target, width, height))
    findings.extend(check_flat_regions(target))
    if system is not None and system.palette:
        findings.extend(check_palette(target, system.palette))
    return findings


def design_verifier(
    system: DesignSystem | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    include_advisory: bool = False,
) -> CallableVerifier:
    """A verifier that gates on deterministic findings alone.

    Cheap, instant, and free of judge variance — run this before
    spending a model call on aesthetic judgment. Advisory findings are
    reported but do not fail by default, because a soft palette miss is
    not worth paying to regenerate.
    """

    def verdict(verifier_node, target) -> Verdict:
        records = target.metadata.get("artifacts") or []
        paths = [Path(r["path"]) for r in records if r.get("path")]
        if not paths:
            return Verdict(passed=True, reason="no artifacts to inspect")

        findings: list[Finding] = []
        for path in paths:
            if not path.exists():
                continue
            findings.extend(
                inspect_asset(path, system=system, width=width, height=height)
            )
        if len(paths) > 1:
            findings.extend(check_near_duplicates(paths))

        blocking = [f for f in findings if f.hard or include_advisory]
        if not blocking:
            return Verdict(passed=True, reason="")
        return Verdict(
            passed=False,
            reason="; ".join(str(f) for f in blocking[:4]),
        )

    return CallableVerifier(verdict)
