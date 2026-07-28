"""Web banner ad suite: art-directed creative at the 7 standard IAB sizes.

    python benchmarks/run_banner_suite.py                                  # offline mechanics
    python benchmarks/run_banner_suite.py --live --k 3 --judge \
        --logo 00_project_files/osiris_light.png --max-cost-per-call-usd 0.06 \
        --publish-dir assets/osiris_ads

The quality recipe, distilled from the sources in docs comments below:

1. CREATIVE DIRECTION FIRST (canvas-design skill pattern): an LLM
   creative director turns the brand's aesthetic worldview into k
   *distinct* photographic concepts per placement — different subjects,
   angles, and moods sharing one palette — instead of sampling one
   scene prompt k times (same-prompt sampling produced near-duplicate
   pairs in image_benchmarks.md).
2. PHOTOGRAPHIC PROMPT LANGUAGE (Google's official Gemini image
   prompting guide): narrative scene sentences with shot type, subject,
   environment, light quality, and lens — not keyword salads — plus an
   explicit reserved negative-space zone per placement class.
3. THE MODEL NEVER RENDERS THE BRAND (this repo's own two-brand
   finding: marks and type garble systematically at small scale).
   Generation is pure photography with a reserved clean zone; the exact
   logo pixels, tagline, and CTA pill are composited deterministically
   after finishing — so the logo is *integrated* with the creative, and
   can never replace it.
4. AD-CRAFT JUDGING (display-ad practice: one focal point, hierarchy,
   CTA contrast, thumbnail legibility): a vision judge scores the
   FINISHED, overlaid banners per placement and picks the winner.

Shared with the asset suite: exact-spec finishing (resize-to-cover +
center-crop) with honesty metrics, budget ceilings on every live call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmarks.artifact_records import (  # noqa: E402
    environment_snapshot,
    image_mime_type,
    portable_path,
)
from benchmarks.run_asset_suite import (  # noqa: E402
    COST_PER_IMAGE_USD,
    MODEL,
    ensure_logo,
    finish,
    logo_intake_node,
)
from smythe import OfflineProvider, Swarm  # noqa: E402
from smythe.graph import (  # noqa: E402
    ExecutionGraph,
    FailurePolicy,
    Node,
    Topology,
)
from smythe.provider import GeminiProvider, OpenAIImageProvider  # noqa: E402

# (placement id, width, height, gemini aspect bucket)
SPECS = [
    ("leaderboard_728x90", 728, 90, "21:9"),
    ("mobile_banner_320x100", 320, 100, "21:9"),
    ("mobile_leaderboard_320x50", 320, 50, "21:9"),
    ("large_rectangle_336x280", 336, 280, "4:3"),
    ("inline_rectangle_300x250", 300, 250, "4:3"),
    ("half_page_300x600", 300, 600, "9:16"),
    ("skyscraper_160x600", 160, 600, "9:16"),
]

DEFAULT_BRAND = "benchmarks/brands/osiris_banners.json"

DIRECTOR_MODEL = "gemini-flash-latest"
DIRECTOR_COST_PER_TOKEN_USD = 0.000003
JUDGE_MODEL = "gemini-flash-lite-latest"

# Image-generation backend. ChatGPT (GPT Image) is prioritized: 'auto'
# picks OpenAI when an OpenAI key is present, else Gemini. GPT Image tops
# out at 3:2 / 2:3, so the ultra-wide strips crop harder than Gemini's
# native 21:9; the deterministic finishing pass still hits exact spec.
OPENAI_IMAGE_MODEL = "gpt-image-2"
OPENAI_SIZE_FOR_BUCKET = {
    "21:9": "1536x1024",   # landscape; hardest crop to the 8:1 strips
    "4:3": "1536x1024",    # landscape
    "9:16": "1024x1536",   # portrait
}


def resolve_image_provider(pref: str) -> str:
    """Resolve the generation backend, prioritizing ChatGPT when 'auto'."""
    if pref in ("openai", "gemini"):
        return pref
    return "openai" if os.environ.get("OPENAI_API_KEY") else "gemini"


def _generation_provider(
    kind: str, bucket: str, *, live: bool,
    max_cost_per_call_usd: float, openai_quality: str,
):
    """Return (provider, model) for one aspect bucket's generation wave."""
    if not live:
        return OfflineProvider(artifacts_per_call=1), "demo-image-model"
    if kind == "openai":
        return (
            OpenAIImageProvider(
                size=OPENAI_SIZE_FOR_BUCKET[bucket],
                quality=openai_quality,
                output_format="png",
                max_cost_per_call_usd=max_cost_per_call_usd,
            ),
            OPENAI_IMAGE_MODEL,
        )
    return (
        GeminiProvider(
            cost_per_image_usd=COST_PER_IMAGE_USD,
            max_cost_per_call_usd=max_cost_per_call_usd,
            image_config={"aspect_ratio": bucket},
        ),
        MODEL,
    )

# Distilled ad-craft canon fed to the creative director. Sources:
# Google's Gemini image prompting guide (narrative scene templates,
# negative-space template), the anthropics/skills canvas-design method
# (aesthetic worldview before execution, restraint, limited palette),
# and standard display-ad practice (single focal point, visual
# hierarchy, CTA contrast, five-second thumbnail test).
CRAFT_CANON = (
    "Craft rules for every concept: exactly ONE hero subject and one "
    "story per banner - if everything competes, nothing wins. The Osiris "
    "charger is the hero and must be LARGE, sharp, and unmistakably "
    "dominant - it should fill roughly a third to a half of the visible "
    "frame, shot close enough that its amber solar face and matte black "
    "body read instantly even at thumbnail size. Never a tiny product "
    "lost in a grand landscape; never the product cut off by the frame "
    "edge - keep the whole device well inside the safe area with a clear "
    "margin. Describe the scene as one flowing photographic sentence: "
    "shot type, subject, environment, quality of light, lens or camera "
    "angle. Use warm, saturated light on the focal subject and quiet, "
    "low-detail surroundings so the subject pops. Respect the reserved "
    "negative-space zone exactly - it must be clean, low-detail, and "
    "evenly lit, because the brand block is composited there later. "
    "Never mention or depict logos, wordmarks, taglines, text, UI, "
    "watermarks, or people's readable clothing print. Keep one "
    "consistent palette and mood across all placements so the campaign "
    "reads as one system."
)

# Reserved clean zone per placement class. Chosen to SURVIVE the
# center-crop in finish(): wide strips crop top/bottom (left/right
# composition survives), verticals crop left/right (top/bottom
# composition survives).
PLACEMENT_CLASSES = {
    "horizontal": "CRITICAL: only the horizontal center strip of the "
                  "frame survives - the top and bottom get cropped away "
                  "entirely, so place the hero product in the exact "
                  "vertical middle, large and complete, never near the "
                  "top or bottom edge. Put the product in the left third; "
                  "the right 40% of the frame must be clean, low-detail "
                  "negative space of any tonality - deep dusk shadow, "
                  "glowing sky, or warm sand all work (the brand system "
                  "adapts its logo variant to the zone)",
    "rectangle": "place the large hero product in the upper two-thirds "
                 "with comfortable margins so it is never clipped; make "
                 "the bottom 25% of the frame clean, low-detail negative "
                 "space - dark shadow or quiet light surface both work",
    "vertical": "place the large hero product in the narrow center "
                "column of the upper two-thirds, complete and never "
                "clipped at the sides; the bottom 30% of the frame must "
                "be clean, low-detail negative space - dark foreground "
                "shadow or calm gradient sky both work",
}

CROP_NOTES = {
    "21:9": "this becomes a very wide short strip: the top and bottom "
            "are cropped away hard, so the product MUST sit in the "
            "vertical center and stay within the middle third of the "
            "height - sky and foreground are expendable",
    "4:3": "the generation is cropped mildly toward square",
    "9:16": "the generation will be center-cropped to a much narrower "
            "column, so left and right edges are expendable",
}


def placement_class(w: int, h: int) -> str:
    if w / h >= 3:
        return "horizontal"
    if h / w >= 1.8:
        return "vertical"
    return "rectangle"


def load_brand(path: str) -> dict:
    """Load and validate a banner brand config against banner SPECS."""
    brand = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "brief", "logo_prompt", "scenes"}
    missing = required - brand.keys()
    if missing:
        raise ValueError(f"Brand config {path} missing keys: {sorted(missing)}")
    missing_scenes = {s[0] for s in SPECS} - brand["scenes"].keys()
    if missing_scenes:
        raise ValueError(
            f"Brand config {path} missing scenes: {sorted(missing_scenes)}"
        )
    return brand


DIRECTOR_PROMPT = """You are the campaign creative director for {name}.

Brand worldview: {worldview}

Campaign brief: {brief}

{craft}

Placements (id | final size | class | crop note | reserved zone):
{placement_table}

For EACH placement id, write {k} genuinely DISTINCT photographic
concepts - different hero subject or setting or camera treatment, never
three variations of one idea - that all share the campaign palette and
mood. Each concept is one flowing paragraph of plain photographic
language for an image model, and must end by restating that placement's
reserved-zone instruction in your own words.

Respond with STRICT JSON only, no prose, no code fences:
{{"placements": {{"<placement id>": ["<concept 1>", "<concept 2>", ...]}}}}"""


async def direct_concepts(
    brand: dict, live: bool, k: int,
) -> tuple[dict[str, list[str]], float, str]:
    """Creative-director stage: k distinct concepts per placement.

    Returns (concepts, cost_usd, source). Offline - and on any live
    parse failure - falls back to the brand config's hand-written
    scenes with variant treatments, so the suite always runs.
    """
    fallback = {
        pid: [
            brand["scenes"][pid] + variant
            for variant in (
                "",
                " Alternative treatment: closer macro framing of the "
                "same subject, shallower depth of field.",
                " Alternative treatment: wider environmental framing, "
                "subject smaller in a grander landscape.",
            )[:k]
        ]
        for pid, _, _, _ in SPECS
    }
    if not live:
        return fallback, 0.0, "offline-fallback"

    table = "\n".join(
        f"- {pid} | {w}x{h} | {placement_class(w, h)} | "
        f"{CROP_NOTES[bucket]} | {PLACEMENT_CLASSES[placement_class(w, h)]}"
        for pid, w, h, bucket in SPECS
    )
    prompt = DIRECTOR_PROMPT.format(
        name=brand["name"],
        worldview=brand.get("creative_worldview", brand["brief"]),
        brief=brand["brief"],
        # craft_canon is brand-overridable so the autoresearch optimizer
        # (run_banner_optimizer.py) can tune the ad-craft rules directly.
        craft=brand.get("craft_canon", CRAFT_CANON),
        placement_table=table,
        k=k,
    )
    # 7 placements x k concept paragraphs overflow the provider's 4096
    # default; give the director real headroom.
    result = await GeminiProvider(max_tokens=16384).complete(
        "You are an award-winning advertising creative director. "
        "STRICT JSON only.",
        prompt,
        DIRECTOR_MODEL,
    )
    cost = (
        result.cost_usd
        if result.cost_usd is not None
        else result.total_tokens * DIRECTOR_COST_PER_TOKEN_USD
    )
    try:
        text = result.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise json.JSONDecodeError("no JSON object", text, 0)
        parsed = json.loads(text[start:end + 1])["placements"]
        concepts = {}
        for pid, _, _, _ in SPECS:
            got = [c for c in parsed.get(pid, []) if isinstance(c, str) and c.strip()]
            if not got:
                raise KeyError(pid)
            while len(got) < k:
                got.append(fallback[pid][len(got) % len(fallback[pid])])
            concepts[pid] = got[:k]
        return concepts, cost, "director"
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        excerpt = result.text[:200].replace("\n", " ")
        return fallback, cost, f"fallback-after-parse-failure ({excerpt!r})"


async def run_bucket(
    bucket: str, specs: list, *, brand: dict, live: bool, out: Path,
    concepts: dict[str, list[str]], k: int, max_cost_per_call_usd: float,
    image_provider: str = "gemini", openai_quality: str = "high",
) -> dict:
    """One parallel generation wave for every placement in one aspect bucket.

    Generation is PURE PHOTOGRAPHY: no logo reference is attached and no
    brand rendering is requested - the exact brand block is composited
    deterministically after finishing (see brand_overlay).
    """
    provider, model = _generation_provider(
        image_provider, bucket, live=live,
        max_cost_per_call_usd=max_cost_per_call_usd,
        openai_quality=openai_quality,
    )
    no_text = brand.get(
        "no_text_instruction",
        " Absolutely no text, letters, numbers, logos, wordmarks, "
        "watermarks, or UI elements anywhere in the image.",
    )
    nodes = [
        Node(
            id=f"{pid}__c{i + 1}",
            label=(
                f"A photorealistic advertising photograph: "
                f"{concepts[pid][i]} {brand['brief']}{no_text}"
            ),
            failure_policy=FailurePolicy.RETRY,
            max_retries=2,
        )
        for pid, _, _, _ in specs
        for i in range(k)
    ]
    graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)
    swarm = Swarm(
        provider=provider,
        model=model,
        parallel=True,
        max_budget_usd=max(1.0, len(nodes) * max_cost_per_call_usd * 1.1),
        artifact_dir=out / "raw" / bucket.replace(":", "x"),
        retry_backoff_s=2.0,
    )
    result = await swarm.execute_async(graph)
    paths = {
        n.id: n.metadata["artifacts"][0]["path"]
        for n in result.graph.nodes
        if n.metadata.get("artifacts")
    }
    return {"paths": paths, "cost": result.total_cost_usd}


# ---------------------------------------------------------------------------
# Deterministic brand overlay - the exact logo, tagline, and CTA are
# composited into the reserved zone. Fonts fall back across platforms.
# ---------------------------------------------------------------------------

# CTA/tagline type: DIN-style Bahnschrift first (ships with Windows,
# reads "designed" in ad contexts), then semibold system fallbacks.
CTA_FONTS = ("bahnschrift.ttf", "segoeuisb.ttf", "seguisb.ttf",
             "arialbd.ttf", "DejaVuSans-Bold.ttf")
TEXT_FONTS = ("bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")


def _font(candidates: tuple[str, ...], size: int, variation: str | None = None):
    from PIL import ImageFont

    for name in candidates:
        try:
            f = ImageFont.truetype(name, size)
        except OSError:
            continue
        if variation:
            try:
                f.set_variation_by_name(variation)
            except (OSError, AttributeError):
                pass
        return f
    return ImageFont.load_default(size)


def _tracked_width(draw, text: str, font, tracking: int) -> int:
    return round(sum(draw.textlength(ch, font=font) for ch in text)
                 + tracking * max(0, len(text) - 1))


def _tracked_text(draw, xy, text: str, font, fill, tracking: int) -> None:
    """Letterspaced caps — the cheapest 'a designer touched this' signal."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def _load_logo_pieces(brand: dict, logo_path: str):
    """Whole logo + icon and wordmark slices (fractions from config)."""
    from PIL import Image

    logo = Image.open(logo_path).convert("RGBA")

    def bbox_crop(im):
        box = im.getchannel("A").getbbox()
        return im.crop(box) if box else im

    w, h = logo.size
    slices = brand.get("logo_slices", {})

    def slice_of(key):
        frac = slices.get(key)
        if not frac:
            return None
        x0, y0, x1, y1 = frac
        return bbox_crop(
            logo.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        )

    return {
        "full": bbox_crop(logo),
        "icon": slice_of("icon"),
        "wordmark": slice_of("wordmark"),
    }


def _derive_knockout(pieces: dict) -> dict:
    """Fallback dark-background variant: recolor dark ink to warm cream.

    Used only when the brand supplies no real dark-mode logo asset.
    Warm (amber) pixels are preserved; dark strokes become cream.
    """
    def recolor(im):
        if im is None:
            return None
        out = im.copy()
        px = out.load()
        for yy in range(out.height):
            for xx in range(out.width):
                r, g, b, a = px[xx, yy]
                if a == 0:
                    continue
                is_warm = r > 140 and r > b + 50 and g > b
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                if not is_warm and lum < 150:
                    px[xx, yy] = (247, 242, 232, a)
        return out

    return {k: recolor(v) for k, v in pieces.items()}


def load_logo_variants(brand: dict, logo_path: str,
                       dark_logo_path: str | None) -> dict:
    """Both tonal variants: 'light_bg' (primary) and 'dark_bg' (knockout).

    A real brand system ships both; using the light-background logo on a
    dark photograph (or hiding the photo under a white card to avoid it)
    is exactly the amateur move this pipeline previously made.
    """
    variants = {"light_bg": _load_logo_pieces(brand, logo_path)}
    dark_path = dark_logo_path or brand.get("logo_dark")
    if dark_path and Path(dark_path).is_file():
        variants["dark_bg"] = _load_logo_pieces(brand, dark_path)
        variants["dark_bg_source"] = "provided"
    else:
        variants["dark_bg"] = _derive_knockout(variants["light_bg"])
        variants["dark_bg_source"] = "derived-knockout"
    return variants


def _scaled(im, *, h: int | None = None, w: int | None = None):
    from PIL import Image

    if h is not None:
        w = max(1, round(im.width * h / im.height))
    else:
        h = max(1, round(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS)


def _region_stats(img, box) -> tuple[float, float]:
    """(mean luminance, luminance stddev) of a region."""
    import math

    strip = img.crop(box).convert("L")
    data = list(strip.getdata())
    if not data:
        return 255.0, 0.0
    mean = sum(data) / len(data)
    var = sum((d - mean) ** 2 for d in data) / len(data)
    return mean, math.sqrt(var)


def _zone_treatment(img, box) -> str:
    """How a designer treats the brand zone, from its actual tonality.

    - 'knockout': the zone is dark — set the dark-background logo and
      light type directly on the photograph. No card, no scrim.
    - 'primary': the zone is light and quiet — set the primary logo and
      dark type directly on the photograph.
    - 'scrim': the zone is busy or mid-toned — deepen it with a feathered
      warm-black gradient (the standard photo-ad lower-third move), then
      use the knockout treatment on top. Never a white card.
    """
    mean, std = _region_stats(img, box)
    if mean < 110:
        return "knockout"
    if mean > 160 and std < 42:
        return "primary"
    return "scrim"


def _scrim(img, box, direction: str = "down", max_alpha: int = 200) -> None:
    """Feathered warm-black gradient over *box*, ramping toward the
    bottom (or right, for horizontal strips)."""
    from PIL import Image

    x0, y0, x1, y1 = box
    mask = Image.new("L", img.size, 0)
    px = mask.load()
    if direction == "down":
        span = max(1, y1 - y0)
        for yy in range(y0, y1):
            a = round(max_alpha * min(1.0, (yy - y0) / (span * 0.55)))
            for xx in range(x0, x1):
                px[xx, yy] = a
    else:  # "right"
        span = max(1, x1 - x0)
        for xx in range(x0, x1):
            a = round(max_alpha * min(1.0, (xx - x0) / (span * 0.55)))
            for yy in range(y0, y1):
                px[xx, yy] = a
    shade = Image.new("RGBA", img.size, (22, 17, 12, 255))
    img.paste(shade, (0, 0), mask)


def _pill_size(draw, text: str, fsize: int) -> tuple[int, int]:
    """Measured CTA button dimensions at this font size.

    Proportions follow ad-craft practice (and the banner-design skill's
    44px minimum where the placement is tall enough): generous optical
    padding, letterspaced caps, small corner radius — not a bootstrap
    pill.
    """
    f = _font(CTA_FONTS, fsize, variation="SemiBold")
    tracking = max(1, round(fsize * 0.10))
    tw = _tracked_width(draw, text, f, tracking)
    # Compact proportions below 13px so small placements can keep a CTA.
    pad_x = round(fsize * (1.35 if fsize > 12 else 1.05))
    pad_y = round(fsize * (0.72 if fsize > 12 else 0.58))
    return tw + 2 * pad_x, fsize + 2 * pad_y


def _cta_pill(img, draw, cx: int, cy: int, text: str, fsize: int, palette):
    """The brand-accent CTA: amber fill, near-black tracked caps.

    Ink-on-amber measures ~6.9:1 contrast (over the 4.5:1 floor), the
    amber echoes the sun in the mark on any scene tonality, and the
    subtle radius reads 'designed', not 'default button'.
    """
    f = _font(CTA_FONTS, fsize, variation="SemiBold")
    tracking = max(1, round(fsize * 0.10))
    tw = _tracked_width(draw, text, f, tracking)
    w2, h2 = _pill_size(draw, text, fsize)
    box = [cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2]
    draw.rounded_rectangle(box, radius=max(2, round(h2 * 0.16)),
                           fill=tuple(palette["amber"]))
    bbox = draw.textbbox((0, 0), text, font=f)
    _tracked_text(draw, (cx - tw / 2, cy - (bbox[3] - bbox[1]) / 2 - bbox[1]),
                  text, f, tuple(palette["black"]), tracking)
    return w2


def brand_overlay(path: Path, *, brand: dict, variants: dict,
                  w: int, h: int) -> dict:
    """Composite the exact brand block into the reserved zone.

    Tonal system (what a designer actually does): the zone's measured
    luminance picks the treatment — the dark-background logo variant set
    directly on dark photography, the primary variant on clean light
    photography, or a feathered warm scrim first when the zone is busy.
    White cards are gone. Layout stays placement-aware: horizontal
    strips carry a lockup + CTA row, rectangles a bottom band, verticals
    a stacked bottom block.
    """
    from PIL import Image, ImageDraw

    palette = {
        "offwhite": brand.get("palette", {}).get("offwhite", [250, 246, 236]),
        "black": brand.get("palette", {}).get("black", [26, 26, 24]),
        "amber": brand.get("palette", {}).get("amber", [224, 154, 36]),
        "amber_dark": brand.get("palette", {}).get("amber_dark", [154, 123, 45]),
    }
    tagline = brand.get("tagline", "")
    cta = brand.get("cta", "")
    cls = placement_class(w, h)

    img = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    metrics: dict = {"placement_class": cls}

    def dress(zone_box, scrim_box, scrim_direction):
        """Pick treatment from tonality; return (pieces, tagline ink)."""
        treatment = _zone_treatment(img, zone_box)
        if treatment == "scrim":
            _scrim(img, scrim_box, direction=scrim_direction)
        metrics["treatment"] = treatment
        p = variants["light_bg" if treatment == "primary" else "dark_bg"]
        tag_ink = tuple(
            palette["amber_dark"] if treatment == "primary"
            else palette["amber"]
        )
        return p, tag_ink

    if cls == "horizontal":
        # Brand block in the right ~40%. Priority packing: wide strips
        # carry the full lockup + CTA; small strips degrade gracefully
        # (lockup only, then wordmark only) instead of shrinking the
        # wordmark below legibility.
        # Small strips get a slightly wider zone: the scrim/knockout
        # system can sit on photography, so pure negative space is no
        # longer a hard requirement. A banner without a CTA is an
        # incomplete ad (the judge is right), so every layout here
        # carries one — pill, stacked pill, or tracked text CTA.
        zone_x0 = round(w * (0.50 if w <= 400 else 0.60)) + round(h * 0.10)
        zone_w = w - zone_x0 - round(h * 0.12)
        zone_cx = zone_x0 + zone_w // 2
        gap = max(6, round(h * 0.12))
        p, tag_ink = dress(
            (zone_x0, 0, w, h),
            (max(0, zone_x0 - round(w * 0.08)), 0, w, h),
            "right",
        )
        fsize = max(10, round(h * 0.185))
        stack_fsize = max(9, round(h * 0.12))
        pill_w, _ph = _pill_size(draw, cta, fsize) if cta else (0, 0)
        sp_w, sp_h = _pill_size(draw, cta, stack_fsize) if cta else (0, 0)
        lockup = _scaled(p["full"], h=round(h * 0.72))
        wm_src = p["wordmark"] if p["wordmark"] is not None else p["full"]
        wm = _scaled(
            wm_src, h=round(h * (0.30 if p["wordmark"] is not None else 0.60)))
        if wm.width > zone_w:
            wm = _scaled(wm, w=zone_w)
        icon_p = p["icon"] if p["icon"] is not None else p["full"]
        ic = _scaled(icon_p, h=round(h * 0.62))
        f_txt = _font(CTA_FONTS, max(10, round(h * 0.17)),
                      variation="SemiBold")
        txt_track = max(1, round(h * 0.017))
        txt_w = _tracked_width(draw, cta, f_txt, txt_track) if cta else 0
        cta_ink = tuple(palette["amber"]
                        if metrics["treatment"] != "primary"
                        else palette["amber_dark"])

        layout = None
        if cta and h >= 70 and lockup.width + gap + pill_w <= zone_w:
            layout = "lockup+cta"
        elif cta and h >= 88 and max(wm.width, sp_w) <= zone_w \
                and wm.height + gap + sp_h <= h - 2 * gap:
            layout = "wordmark-over-cta"
        elif cta and h < 70 and wm.width + gap + txt_w <= zone_w:
            layout = "wordmark+cta-text"
        elif cta and h < 70 and ic.width + gap + txt_w <= zone_w:
            layout = "icon+cta-text"
        elif h >= 70 and lockup.width <= zone_w:
            layout = "lockup"
        else:
            layout = "wordmark"
        metrics["horizontal_layout"] = layout

        if layout == "lockup+cta":
            total = lockup.width + gap + pill_w
            x = zone_x0 + (zone_w - total) // 2
            img.alpha_composite(lockup, (x, (h - lockup.height) // 2))
            _cta_pill(img, draw, x + lockup.width + gap + pill_w // 2,
                      h // 2, cta, fsize, palette)
        elif layout == "wordmark-over-cta":
            block_h = wm.height + gap + sp_h
            y0 = (h - block_h) // 2
            img.alpha_composite(wm, (zone_cx - wm.width // 2, y0))
            _cta_pill(img, draw, zone_cx, y0 + wm.height + gap + sp_h // 2,
                      cta, stack_fsize, palette)
        elif layout in ("wordmark+cta-text", "icon+cta-text"):
            lead = wm if layout == "wordmark+cta-text" else ic
            total = lead.width + gap + txt_w
            x = zone_x0 + (zone_w - total) // 2
            img.alpha_composite(lead, (x, (h - lead.height) // 2))
            bbox = draw.textbbox((0, 0), cta, font=f_txt)
            _tracked_text(
                draw,
                (x + lead.width + gap,
                 h / 2 - (bbox[3] - bbox[1]) / 2 - bbox[1]),
                cta, f_txt, cta_ink, txt_track)
        elif layout == "lockup":
            img.alpha_composite(
                lockup, (zone_cx - lockup.width // 2,
                         (h - lockup.height) // 2))
        else:
            img.alpha_composite(
                wm, (zone_cx - wm.width // 2, (h - wm.height) // 2))
    elif cls == "rectangle":
        # Bottom band: lockup row left, CTA right; photography above.
        band_top = round(h * 0.75)
        p, _tag_ink = dress(
            (round(w * 0.03), band_top, w - round(w * 0.03), h),
            (0, round(h * 0.64), w, h),
            "down",
        )
        icon_p = p["icon"] if p["icon"] is not None else p["full"]
        cy = band_top + (h - band_top) // 2
        fsize = max(12, round(h * 0.05))
        pill_w, _ = _pill_size(draw, cta, fsize) if cta else (0, 0)
        pill_cx = w - round(w * 0.05) - pill_w // 2
        block_right = pill_cx - pill_w // 2 - round(w * 0.03)
        x = round(w * 0.06)
        ic = _scaled(icon_p, h=round((h - band_top) * 0.62))
        img.alpha_composite(ic, (x, cy - ic.height // 2))
        x += ic.width + round(w * 0.025)
        if p["wordmark"] is not None:
            wm = _scaled(p["wordmark"], h=round((h - band_top) * 0.30))
            if x + wm.width > block_right and block_right - x > 30:
                wm = _scaled(p["wordmark"], w=block_right - x)
            if x + wm.width <= block_right + 2:
                img.alpha_composite(wm, (x, cy - wm.height // 2))
        if cta:
            _cta_pill(img, draw, pill_cx, cy, cta, fsize, palette)
    else:  # vertical
        # Stack computed to FIT the zone: lockup, tagline, CTA, gaps.
        zone_top = round(h * 0.72)
        bottom = h - round(h * 0.028)
        p, tag_ink = dress(
            (round(w * 0.06), zone_top, w - round(w * 0.06), bottom),
            (0, round(h * 0.60), w, h),
            "down",
        )
        gap = max(4, round(h * 0.014))
        y = zone_top + gap
        tag_fsize = max(10, round(w * 0.048))
        cta_fsize = max(13, round(w * 0.060))
        pill_w, pill_h = _pill_size(draw, cta, cta_fsize) if cta else (0, 0)
        f_tag = _font(TEXT_FONTS, tag_fsize)
        tag_tracking = max(1, round(tag_fsize * 0.18))
        tag_w = (_tracked_width(draw, tagline, f_tag, tag_tracking)
                 if tagline else 0)
        tag_fits = bool(tagline) and tag_w < w * 0.88
        tag_h = round(tag_fsize * 1.4) if tag_fits else 0
        avail_lock = (bottom - y) - pill_h - tag_h - gap * (
            1 + (1 if tag_fits else 0) + (1 if cta else 0))
        lock = _scaled(p["full"], w=round(w * 0.46))
        if lock.height > avail_lock:
            lock = _scaled(p["full"], h=max(20, avail_lock))
        img.alpha_composite(lock, ((w - lock.width) // 2, y))
        y += lock.height + gap
        if tag_fits:
            _tracked_text(draw, ((w - tag_w) / 2, y), tagline, f_tag,
                          tag_ink, tag_tracking)
            y += tag_h + gap
        if cta:
            _cta_pill(img, draw, w // 2,
                      min(y + pill_h // 2, bottom - gap - pill_h // 2),
                      cta, cta_fsize, palette)

    img.convert("RGB").save(path, "PNG")
    return metrics


JUDGE_PROMPT = (
    "You are the brand art director for {name}. The FIRST attached image "
    "is the official brand logo (ground truth). The following {k} images "
    "are candidate versions of the SAME finished banner ad placement "
    "({placement}, {w}x{h}), in this order: {cids}. The brand block "
    "(logo, tagline, CTA) was composited deterministically - judge the "
    "whole finished banner the way a senior art director reviews comps: "
    "(a) focal_clarity - one hero subject that reads instantly at "
    "thumbnail size; (b) photo_quality - professional light, "
    "composition, no AI artifacts; (c) tonal_harmony - the brand block "
    "belongs to the photograph: correct logo variant for the zone's "
    "tonality (light-background mark on light zones, dark-background "
    "mark on dark zones), any scrim reads as intentional grading, "
    "nothing looks pasted on; (d) cta_craft - button color, typography, "
    "tracking, and proportions look professionally designed with >=4.5:1 "
    "label contrast, never a default UI widget; (e) defects - "
    "artifacts, clutter, accidental text in the photography. Be harsh: "
    "reserve 9-10 for work you would actually ship to a client. Score "
    "1-10 each and pick the single best ad. Respond with STRICT JSON "
    "only, no prose, no code fences: "
    '{{"candidates": [{{"id": "<cid>", "focal_clarity": <1-10>, '
    '"photo_quality": <1-10>, "tonal_harmony": <1-10>, '
    '"cta_craft": <1-10>, "defects": ["..."]}}], "winner": "<cid>"}}'
)


async def judge_placement(
    *, brand: dict, live: bool, out: Path, logo_path: str,
    pid: str, w: int, h: int, candidates: dict[str, str],
) -> dict:
    """Vision judge scores the finished candidates and picks a winner."""
    provider = GeminiProvider() if live else OfflineProvider()
    intake = logo_intake_node(logo_path)
    intake.metadata["artifacts"] = intake.metadata["artifacts"] + [
        {"path": p, "mime_type": image_mime_type(p)}
        for p in candidates.values()
    ]
    judge = Node(
        id=f"judge_{pid}",
        label=JUDGE_PROMPT.format(
            name=brand["name"], k=len(candidates), placement=pid,
            w=w, h=h, cids=", ".join(candidates.keys()),
        ),
        depends_on=[intake.id],
        attach_dep_artifacts=True,
        failure_policy=FailurePolicy.RETRY,
        max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[intake, judge])
    swarm = Swarm(
        provider=provider,
        model=JUDGE_MODEL if live else "demo-model",
        parallel=True,
        max_budget_usd=0.25,
        artifact_dir=out / "raw" / "judge",
    )
    result = await swarm.execute_async(graph)
    raw = str(result.graph.nodes[-1].result)
    verdict: dict = {"cost_usd": result.total_cost_usd, "raw": raw}
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        parsed = json.loads(text)
        verdict["scores"] = {c["id"]: c for c in parsed.get("candidates", [])}
        verdict["winner"] = parsed.get("winner")
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict["scores"] = {}
        verdict["winner"] = None
    return verdict


def make_contact_sheet(finals: dict[str, Path], out_path: Path) -> None:
    """Assemble the 7 finals into one review sheet (same grid as before)."""
    from PIL import Image

    pad = 24
    cols = {
        "left": ["skyscraper_160x600", "half_page_300x600"],
        "mid": ["leaderboard_728x90", "large_rectangle_336x280",
                "inline_rectangle_300x250"],
        "right": ["mobile_banner_320x100", "mobile_leaderboard_320x50"],
    }
    sheet = Image.new("RGB", (728 + 160 + 300 + pad * 4, 600 + pad * 2 + 40),
                      (233, 226, 210))
    x = pad
    for pid in cols["left"]:
        if pid not in finals:
            continue
        im = Image.open(finals[pid])
        sheet.paste(im, (x, pad))
        x += im.width + pad
    col_x, y = x, pad
    for pid in cols["mid"]:
        if pid not in finals:
            continue
        im = Image.open(finals[pid])
        sheet.paste(im, (col_x, y))
        y += im.height + pad
    x2, y2 = col_x + 336 + pad, pad + 90 + pad
    for pid in cols["right"]:
        if pid not in finals:
            continue
        im = Image.open(finals[pid])
        sheet.paste(im, (x2, y2))
        y2 += im.height + pad
    sheet.save(out_path)


async def main_async(
    brand: dict, live: bool, out: Path, logo_arg: str | None, judge: bool,
    k: int, max_cost_per_call_usd: float | None,
    publish_dir: str | None = None, dark_logo_arg: str | None = None,
    image_provider: str = "gemini", openai_quality: str = "high",
) -> dict:
    if live and max_cost_per_call_usd is None:
        configured = os.environ.get("GEMINI_IMAGE_MAX_COST_PER_CALL_USD")
        if not configured:
            raise ValueError(
                "Live generation requires --max-cost-per-call-usd or "
                "GEMINI_IMAGE_MAX_COST_PER_CALL_USD."
            )
        max_cost_per_call_usd = float(configured)
    max_cost_per_call_usd = max_cost_per_call_usd or 0.0
    slug = brand["name"].lower().replace(" ", "_")

    logo = await ensure_logo(
        brand=brand, live=live, out=out, provided=logo_arg,
        max_cost_per_call_usd=max_cost_per_call_usd,
    )
    variants = load_logo_variants(brand, logo["path"], dark_logo_arg)

    concepts, director_cost, director_source = await direct_concepts(
        brand, live, k,
    )

    buckets: dict[str, list] = {}
    for spec in SPECS:
        buckets.setdefault(spec[3], []).append(spec)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(
        run_bucket(
            bucket, specs, brand=brand, live=live, out=out,
            concepts=concepts, k=k,
            max_cost_per_call_usd=max_cost_per_call_usd,
            image_provider=image_provider, openai_quality=openai_quality,
        )
        for bucket, specs in buckets.items()
    ))
    generation_wall_s = time.perf_counter() - t0

    generated: dict[str, str] = {}
    for r in results:
        generated.update(r["paths"])
    total_cost = (
        sum(r["cost"] for r in results)
        + logo.get("cost_usd", 0.0)
        + director_cost
    )

    # Finish EVERY candidate to exact spec, overlay the brand block,
    # then judge the finished banners.
    cand_dir = out / "candidates"
    final_dir = out / "final"
    cand_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    report = []
    judge_cost = 0.0
    finals: dict[str, Path] = {}
    for pid, w, h, bucket in SPECS:
        entry: dict = {
            "placement": pid, "spec": f"{w}x{h} PNG", "aspect_bucket": bucket,
            "candidates": k, "concepts": concepts.get(pid, []),
        }
        finished: dict[str, str] = {}
        metrics: dict[str, dict] = {}
        for i in range(k):
            cid = f"c{i + 1}"
            src = generated.get(f"{pid}__{cid}")
            if not src:
                continue
            dst = cand_dir / f"{pid}__{cid}.png"
            metrics[cid] = finish(Path(src), dst, w, h, "PNG")
            metrics[cid].update(brand_overlay(
                dst, brand=brand, variants=variants, w=w, h=h,
            ))
            finished[cid] = str(dst)
        if not finished:
            entry["status"] = "MISSING"
            report.append(entry)
            continue

        winner = next(iter(finished))
        if judge and len(finished) > 1:
            verdict = await judge_placement(
                brand=brand, live=live, out=out, logo_path=logo["path"],
                pid=pid, w=w, h=h, candidates=finished,
            )
            judge_cost += verdict.get("cost_usd", 0.0)
            entry["judge"] = {
                cid: {key: s.get(key) for key in (
                    "focal_clarity", "photo_quality", "tonal_harmony",
                    "cta_craft", "defects",
                )}
                for cid, s in verdict.get("scores", {}).items()
            }
            # Pick by argmax of per-candidate scores rather than the
            # judge's single "winner" token — LLM judges show position
            # bias on the final attached image; scored criteria are
            # sturdier. The judge's token only breaks exact ties.
            def _mean_score(cid: str) -> float | None:
                s = verdict.get("scores", {}).get(cid, {})
                vals = [s.get(key) for key in (
                    "focal_clarity", "photo_quality", "tonal_harmony",
                    "cta_craft")]
                nums = [v for v in vals if isinstance(v, (int, float))]
                return sum(nums) / len(nums) if nums else None

            scored = {
                cid: m for cid in finished
                if (m := _mean_score(cid)) is not None
            }
            if scored:
                best = max(scored.values())
                top = [cid for cid, m in scored.items() if m == best]
                winner = (verdict["winner"]
                          if verdict.get("winner") in top else top[0])
                entry["score_means"] = {
                    cid: round(m, 2) for cid, m in scored.items()
                }
            elif verdict.get("winner") in finished:
                winner = verdict["winner"]
        entry["winner"] = winner
        entry.update(metrics.get(winner, {}))

        dst = final_dir / f"{slug}_{pid}.png"
        from shutil import copyfile
        copyfile(finished[winner], dst)
        from PIL import Image
        with Image.open(dst) as img:
            entry["status"] = "PASS" if img.size == (w, h) else "FAIL"
        entry["final_path"] = str(dst)
        finals[pid] = dst
        report.append(entry)

    sheet_path = final_dir / "contact_sheet.png"
    if finals:
        make_contact_sheet(finals, sheet_path)

    if publish_dir and finals:
        from shutil import copyfile
        pub = Path(publish_dir)
        pub.mkdir(parents=True, exist_ok=True)
        for pid, w, h, _ in SPECS:
            if pid in finals:
                copyfile(finals[pid], pub / f"{slug}_{pid}.png")
        copyfile(sheet_path, pub / "contact_sheet.png")

    for entry in report:
        if entry.get("final_path"):
            entry["final_path"] = portable_path(entry["final_path"])

    passed = sum(1 for e in report if e["status"] == "PASS")
    return {
        "benchmark": "banner-ad-suite",
        "brand": brand["name"],
        "mode": "LIVE" if live else "offline",
        "image_provider": image_provider if live else "offline",
        "model": (
            (OPENAI_IMAGE_MODEL if image_provider == "openai" else MODEL)
            if live else "offline"
        ),
        "director": {
            "model": DIRECTOR_MODEL if live else "offline",
            "source": director_source,
            "cost_usd": round(director_cost, 6),
        },
        "candidates_per_placement": k,
        "cost_per_image_usd": COST_PER_IMAGE_USD if live else 0,
        "environment": environment_snapshot("smythe", "google-genai", "pillow"),
        "placements_pass": f"{passed}/{len(SPECS)}",
        "brand_logo": {**logo, "path": portable_path(logo["path"]),
                       "dark_variant": variants["dark_bg_source"]},
        "generation_wall_s": round(generation_wall_s, 2),
        "cost_usd": round(total_cost + judge_cost, 4),
        "placements": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--brand", default=DEFAULT_BRAND)
    parser.add_argument("--logo", default=None,
                        help="path to the official brand logo image")
    parser.add_argument("--logo-dark", default=None,
                        help="dark-background logo variant (falls back to "
                             "brand config logo_dark, then a derived "
                             "knockout)")
    parser.add_argument("--k", type=int, default=3,
                        help="distinct art-directed candidates per placement")
    parser.add_argument("--judge", action="store_true",
                        help="vision-judge winner selection per placement")
    parser.add_argument("--max-cost-per-call-usd", type=float, default=None)
    parser.add_argument(
        "--image-provider", choices=["auto", "openai", "gemini"],
        default="auto",
        help="image generation backend; 'auto' prioritizes OpenAI GPT "
             "Image when an OpenAI key is present, else Gemini")
    parser.add_argument(
        "--openai-quality", choices=["low", "medium", "high", "auto"],
        default="high", help="GPT Image quality tier (openai only)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--publish-dir", default=None,
                        help="also copy winning finals + contact sheet here")
    args = parser.parse_args()

    # Director + vision judge always run on Gemini (a cross-vendor judge
    # for OpenAI-generated images actually removes self-preference bias),
    # so live mode needs a Google key regardless of the image backend.
    image_provider = resolve_image_provider(args.image_provider)
    live = args.live and bool(os.environ.get("GOOGLE_API_KEY"))
    if args.live and not live:
        print("--live requires GOOGLE_API_KEY (creative director + judge "
              "run on Gemini); falling back to offline.")
    if live and image_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("image-provider openai needs OPENAI_API_KEY; "
              "using Gemini for generation.")
        image_provider = "gemini"
    if live:
        print(f"[provider] images: {image_provider}"
              f"{' (' + OPENAI_IMAGE_MODEL + ', q=' + args.openai_quality + ')' if image_provider == 'openai' else ' (' + MODEL + ')'}"
              f" | director/judge: gemini")

    brand = load_brand(args.brand)
    out = Path(args.out or f"smythe_artifacts/banner_suite/{brand['name'].lower()}")
    record = asyncio.run(main_async(
        brand, live, out, args.logo, args.judge, max(1, args.k),
        args.max_cost_per_call_usd, publish_dir=args.publish_dir,
        dark_logo_arg=args.logo_dark,
        image_provider=image_provider, openai_quality=args.openai_quality,
    ))
    print(json.dumps(
        {k: v for k, v in record.items() if k not in ("placements",)},
        indent=2,
    ))
    for e in record["placements"]:
        print(f"  {e['placement']:28s} {e.get('status'):7s} "
              f"winner={e.get('winner')} crop={e.get('cropped_fraction')}")
    results = Path(args.results or "benchmarks/results/banner_suite.json")
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8",
    )
    print(f"record -> {results}")


if __name__ == "__main__":
    main()
