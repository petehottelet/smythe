"""The brand asset suite: 8 exact-spec launch assets from one brand config.

    python benchmarks/run_asset_suite.py                    # offline mechanics
    python benchmarks/run_asset_suite.py --live             # ~$0.35 on Gemini
    python benchmarks/run_asset_suite.py --live --logo path/to/logo.png
    python benchmarks/run_asset_suite.py --live --judge --brand benchmarks/brands/metacortex.json

Brand lock: assets for a brand must share its actual mark, not eight
independent hallucinations of one. The suite therefore confirms
possession of a brand logo first — pass an existing file with --logo,
or a dedicated logo node generates one — and every asset node receives
the logo pixels as a reference image (`attach_dep_artifacts=True`
feeding Gemini's image-input channel), locking the mark across the
whole parallel fan-out.

This is the README's broadcast-reduce example run for real, against the
hard part: exact pixel specifications (2400x1200, 1290x2796, 300 dpi
print) that no image model emits natively. The pipeline is honest about
that: each asset generates at the model's nearest supported aspect
bucket, then a deterministic finishing pass (resize-to-cover +
center-crop, Pillow) produces the exact spec. Compliance is checked
programmatically; upscale factors are reported because they are the
quality cost of exceeding native model resolution.

Aspect buckets run as parallel swarms concurrently (one Gemini
image_config per provider instance), so the whole suite is still one
parallel wave end to end.
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

from smythe import OfflineProvider, Swarm  # noqa: E402
from smythe.graph import (  # noqa: E402
    ExecutionGraph,
    FailurePolicy,
    Node,
    NodeStatus,
    Topology,
)
from smythe.provider import GeminiProvider  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

COST_PER_IMAGE_USD = 0.039
MODEL = "gemini-2.5-flash-image"

# (asset id, width, height, format, gemini aspect bucket) — scenes come
# from the brand config so the same spec sheet serves any brand.
SPECS = [
    ("hero", 2400, 1200, "PNG", "16:9"),
    ("instagram", 1080, 1080, "JPEG", "1:1"),
    ("x_banner", 1500, 500, "JPEG", "21:9"),
    ("story", 1080, 1920, "PNG", "9:16"),
    ("email_header", 600, 200, "PNG", "21:9"),
    ("appstore", 1290, 2796, "PNG", "9:16"),
    ("og_card", 1200, 630, "PNG", "16:9"),
    ("print_ad", 2550, 3300, "PNG", "3:4"),
]

DEFAULT_BRAND = "benchmarks/brands/osiris.json"


def load_brand(path: str) -> dict:
    """Load and validate a brand config (see benchmarks/brands/)."""
    brand = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "brief", "logo_prompt", "scenes"}
    missing = required - brand.keys()
    if missing:
        raise ValueError(f"Brand config {path} missing keys: {sorted(missing)}")
    scene_ids = {spec[0] for spec in SPECS}
    missing_scenes = scene_ids - brand["scenes"].keys()
    if missing_scenes:
        raise ValueError(
            f"Brand config {path} missing scenes: {sorted(missing_scenes)}"
        )
    return brand


async def ensure_logo(
    *, brand: dict, live: bool, out: Path, provided: str | None,
) -> dict:
    """Confirm possession of a brand logo, or create one first.

    Returns {"path": ..., "source": "provided"|"generated"}. A provided
    path must exist — a typo'd --logo must not silently fall through to
    generation and brand the whole suite with an invented mark.
    """
    if provided:
        path = Path(provided)
        if not path.is_file():
            raise FileNotFoundError(f"--logo file not found: {provided}")
        return {"path": str(path.resolve()), "source": "provided"}

    provider = (
        GeminiProvider(
            cost_per_image_usd=COST_PER_IMAGE_USD,
            image_config={"aspect_ratio": "1:1"},
        )
        if live
        else OfflineProvider(artifacts_per_call=1)
    )
    node = Node(
        id="brand_logo", label=brand["logo_prompt"],
        failure_policy=FailurePolicy.RETRY, max_retries=2,
    )
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    swarm = Swarm(
        provider=provider,
        model=MODEL if live else "demo-image-model",
        parallel=True,
        max_budget_usd=0.25,
        artifact_dir=out / "raw" / "logo",
        retry_backoff_s=2.0,
    )
    result = await swarm.execute_async(graph)
    records = result.graph.nodes[0].metadata.get("artifacts", [])
    if not records:
        raise RuntimeError("Logo generation returned no image artifact")
    return {
        "path": records[0]["path"],
        "source": "generated",
        "cost_usd": result.total_cost_usd,
    }


def logo_intake_node(logo_path: str) -> Node:
    """A pre-completed node carrying the brand logo as its artifact.

    Asset nodes depend on it with ``attach_dep_artifacts=True``, so the
    executor feeds the actual logo pixels to every generation call.
    """
    node = Node(
        id="brand_logo",
        label="Brand logo intake (asset provided to the swarm)",
        result="Official brand logo (attached as image).",
    )
    node.status = NodeStatus.COMPLETED
    node.metadata["artifacts"] = [{"path": logo_path, "mime_type": "image/png"}]
    return node


JUDGE_MODEL = "gemini-flash-lite-latest"

JUDGE_PROMPT = (
    "You are the brand art director. The FIRST attached image is the "
    "official {name} brand logo. The following images are the finished "
    "launch assets, in this order: {asset_ids}. For each asset, judge "
    "brand consistency against the official logo: is the exact mark "
    "reproduced (not a variant), are palette and typography on-brand, "
    "and are there defects (misspelled text, distorted mark, wrong "
    "colors)? Respond with STRICT JSON only, no prose, no code fences: "
    '{{"assets": [{{"id": "<asset id>", "brand_consistency": <1-10>, '
    '"defects": ["..."]}}], "overall": <1-10>}}'
)


async def judge_assets(
    *, brand: dict, live: bool, out: Path, logo_path: str,
    finals: dict[str, str],
) -> dict:
    """Reduce stage: a vision judge scores brand consistency per asset.

    Closes the loop from locked-by-construction to verified-by-
    measurement.  Returns parsed scores, or the raw text when the
    output isn't valid JSON (recorded either way — a judge that can't
    follow the schema is itself a finding).
    """
    provider = GeminiProvider() if live else OfflineProvider()
    intake = logo_intake_node(logo_path)
    intake.metadata["artifacts"] = (
        intake.metadata["artifacts"]
        + [{"path": p, "mime_type": "image/png"} for p in finals.values()]
    )
    judge = Node(
        id="brand_judge",
        label=JUDGE_PROMPT.format(
            name=brand["name"], asset_ids=", ".join(finals.keys()),
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
        verdict["scores"] = {a["id"]: a for a in parsed.get("assets", [])}
        verdict["overall"] = parsed.get("overall")
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict["scores"] = {}
        verdict["overall"] = None
    return verdict


def composite_tagline(path: Path, tagline: str) -> None:
    """Deterministically set the exact tagline onto a finished asset.

    The two-brand finding (see image_benchmarks.md): long/uncommon
    tagline words misrender systematically when generated as pixels.
    The production answer is to keep typography out of the model and
    composite the exact copy — bottom-centered, ink chosen by sampling
    the luminance behind the text.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(path).convert("RGB")
    w, h = img.size
    size = max(14, w // 30)
    font = None
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), tagline, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) // 2, h - th - max(10, h // 16)
    strip = img.crop((
        max(0, x), max(0, y), min(w, x + tw), min(h, y + th + 4),
    )).convert("L")
    data = list(strip.getdata())
    lum = sum(data) / max(1, len(data))
    ink = (24, 24, 22) if lum > 140 else (245, 244, 240)
    draw.text((x, y), tagline, font=font, fill=ink)
    fmt = "JPEG" if path.suffix == ".jpg" else "PNG"
    kwargs: dict = {"quality": 92} if fmt == "JPEG" else {}
    if path.stem == "print_ad":
        kwargs["dpi"] = (300, 300)
    img.save(path, fmt, **kwargs)


def finish(src: Path, dst: Path, width: int, height: int, fmt: str) -> dict:
    """Deterministically finish a generated image to its exact spec.

    Resize-to-cover then center-crop — never letterbox, never distort.
    Returns the honesty metrics (upscale factor, crop fraction).
    """
    from PIL import Image

    img = Image.open(src).convert("RGB")
    sw, sh = img.size
    scale = max(width / sw, height / sh)
    rw, rh = round(sw * scale), round(sh * scale)
    img = img.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    img = img.crop((left, top, left + width, top + height))
    save_kwargs: dict = {"dpi": (300, 300)} if dst.stem == "print_ad" else {}
    if fmt == "JPEG":
        save_kwargs["quality"] = 92
    img.save(dst, fmt, **save_kwargs)
    return {
        "generated_size": [sw, sh],
        "upscale_factor": round(scale, 2),
        "cropped_fraction": round(1 - (width * height) / (rw * rh), 3),
    }


async def run_bucket(
    bucket: str, specs: list, *, brand: dict, live: bool, out: Path,
    logo_path: str,
) -> dict:
    provider = (
        GeminiProvider(
            cost_per_image_usd=COST_PER_IMAGE_USD,
            image_config={"aspect_ratio": bucket},
        )
        if live
        else OfflineProvider(artifacts_per_call=1)
    )
    intake = logo_intake_node(logo_path)
    # Prompt policies are brand-overridable — this is the optimizer's
    # search space (see run_optimizer.py); defaults are the hand-tuned
    # versions that produced the published results.
    logo_lock = brand.get(
        "logo_lock_instruction",
        "The attached image is the official {name} brand logo. Reproduce "
        "this exact mark faithfully wherever the brand appears — never "
        "invent a different logo. ",
    ).format(name=brand["name"])
    no_text = (
        brand.get(
            "no_text_instruction",
            " Do not render any tagline, slogan, or small text anywhere in "
            "the image — leave clean negative space near the bottom for "
            "typography to be added later. The logo mark may appear on "
            "products and signage.",
        )
        if brand.get("composite_tagline")
        else ""
    )
    nodes = [
        Node(
            id=asset_id,
            label=(
                f"{logo_lock}Generate a "
                f"{brand['scenes'][asset_id]} {brand['brief']}{no_text}"
            ),
            depends_on=[intake.id],
            attach_dep_artifacts=True,
            failure_policy=FailurePolicy.RETRY,
            max_retries=2,
        )
        for asset_id, _, _, _, _ in specs
    ]
    graph = ExecutionGraph(
        topology=[Topology.BROADCAST_REDUCE], nodes=[intake, *nodes],
    )
    swarm = Swarm(
        provider=provider,
        model=MODEL if live else "demo-image-model",
        parallel=True,
        max_budget_usd=1.0,
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


async def main_async(
    brand: dict, live: bool, out: Path, provided_logo: str | None, judge: bool,
) -> dict:
    buckets: dict[str, list] = {}
    for spec in SPECS:
        buckets.setdefault(spec[4], []).append(spec)

    t_logo = time.perf_counter()
    logo = await ensure_logo(
        brand=brand, live=live, out=out, provided=provided_logo,
    )
    logo_wall_s = time.perf_counter() - t_logo

    t0 = time.perf_counter()
    results = await asyncio.gather(*(
        run_bucket(
            bucket, specs, brand=brand, live=live, out=out,
            logo_path=logo["path"],
        )
        for bucket, specs in buckets.items()
    ))
    generation_wall_s = time.perf_counter() - t0

    generated: dict[str, str] = {}
    for r in results:
        generated.update(r["paths"])
    generated.pop("brand_logo", None)
    total_cost = sum(r["cost"] for r in results) + logo.get("cost_usd", 0.0)

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for asset_id, width, height, fmt, bucket in SPECS:
        entry: dict = {
            "asset": asset_id,
            "spec": f"{width}x{height} {fmt}",
            "aspect_bucket": bucket,
        }
        src = generated.get(asset_id)
        if not src:
            entry["status"] = "MISSING"
            report.append(entry)
            continue
        dst = final_dir / f"{asset_id}.{'jpg' if fmt == 'JPEG' else 'png'}"
        entry.update(finish(Path(src), dst, width, height, fmt))
        if brand.get("composite_tagline") and brand.get("tagline"):
            composite_tagline(dst, brand["tagline"])
            entry["tagline_composited"] = True
        from PIL import Image
        with Image.open(dst) as img:
            ok = img.size == (width, height) and img.format == fmt
        entry["status"] = "PASS" if ok else "FAIL"
        entry["final_path"] = str(dst)
        report.append(entry)

    verdict = None
    if judge:
        finals = {
            e["asset"]: e["final_path"] for e in report if e.get("final_path")
        }
        verdict = await judge_assets(
            brand=brand, live=live, out=out, logo_path=logo["path"],
            finals=finals,
        )
        for e in report:
            score = verdict["scores"].get(e["asset"])
            if score:
                e["brand_consistency"] = score.get("brand_consistency")
                e["defects"] = score.get("defects", [])

    passed = sum(1 for e in report if e["status"] == "PASS")
    return {
        "benchmark": "brand-asset-suite",
        "brand": brand["name"],
        "mode": "LIVE" if live else "offline",
        "model": MODEL if live else "offline",
        "assets_pass": f"{passed}/{len(SPECS)}",
        "brand_logo": {**logo, "wall_s": round(logo_wall_s, 2)},
        "generation_wall_s": round(generation_wall_s, 2),
        "buckets_run_concurrently": len(buckets),
        "cost_usd": round(
            total_cost + (verdict.get("cost_usd", 0.0) if verdict else 0.0), 4,
        ),
        "brand_judge": (
            {"overall": verdict["overall"], "cost_usd": verdict["cost_usd"]}
            if verdict else None
        ),
        "assets": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--brand", default=DEFAULT_BRAND,
        help="brand config JSON (see benchmarks/brands/)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--logo", default=None,
        help="path to an existing brand logo; omitted = generate one first",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="score brand consistency per asset with a vision judge",
    )
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("GOOGLE_API_KEY"))
    if args.live and not live:
        print("--live requires GOOGLE_API_KEY; falling back to offline.")

    brand = load_brand(args.brand)
    slug = brand["name"].lower().replace(" ", "_")
    out = Path(args.out) if args.out else Path("smythe_artifacts/asset_suite") / slug
    payload = asyncio.run(main_async(brand, live, out, args.logo, args.judge))

    logo = payload["brand_logo"]
    print(f"[{payload['mode']}] logo {logo['source']} ({logo['wall_s']}s), "
          f"{payload['assets_pass']} specs met, "
          f"generation wall {payload['generation_wall_s']}s, "
          f"cost ${payload['cost_usd']}")
    for e in payload["assets"]:
        extra = (
            f" upscale x{e.get('upscale_factor')} crop {e.get('cropped_fraction')}"
            if "upscale_factor" in e else ""
        )
        if "brand_consistency" in e:
            extra += f" brand {e['brand_consistency']}/10"
            if e.get("defects"):
                extra += f" defects: {'; '.join(e['defects'])}"
        print(f"  {e['asset']:<13} {e['spec']:<16} {e['status']}{extra}")
    if payload.get("brand_judge") and payload["brand_judge"]["overall"] is not None:
        print(f"  overall brand consistency: {payload['brand_judge']['overall']}/10")

    results_path = args.results or (
        f"benchmarks/results/asset_suite_{slug}.json" if live
        else "benchmarks/results/asset_suite_offline.json"
    )
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
