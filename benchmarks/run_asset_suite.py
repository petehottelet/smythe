"""The Osiris asset suite: 8 exact-spec launch assets from one brief.

    python benchmarks/run_asset_suite.py            # offline mechanics
    python benchmarks/run_asset_suite.py --live     # ~$0.31 on Gemini

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
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology  # noqa: E402
from smythe.provider import GeminiProvider  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

COST_PER_IMAGE_USD = 0.039
MODEL = "gemini-2.5-flash-image"

BRIEF = (
    "for the launch of 'Osiris', a portable solar-powered phone charger. "
    "Brand palette: warm amber, matte black, off-white. Style: clean product "
    "photography, natural light, lifestyle context, premium minimalist design. "
    "Brand text where appropriate: 'OSIRIS - Power from the sun'. "
)

# (asset id, width, height, format, gemini aspect bucket, scene)
SPECS = [
    ("hero", 2400, 1200, "PNG", "16:9",
     "hero image: the product resting on a sunlit hiking trail"),
    ("instagram", 1080, 1080, "JPEG", "1:1",
     "Instagram post: lifestyle flat-lay with the product and travel gear"),
    ("x_banner", 1500, 500, "JPEG", "21:9",
     "X/Twitter banner: macro product detail across a wide composition"),
    ("story", 1080, 1920, "PNG", "9:16",
     "Story/Reel card: vertical lifestyle shot, product in hand outdoors"),
    ("email_header", 600, 200, "PNG", "21:9",
     "email header: slim newsletter announcement banner with negative space"),
    ("appstore", 1290, 2796, "PNG", "9:16",
     "App Store screenshot: the companion app UI on a phone, feature callout"),
    ("og_card", 1200, 630, "PNG", "16:9",
     "OG preview card: link-share thumbnail, product centered, room for title"),
    ("print_ad", 2550, 3300, "PNG", "3:4",
     "print ad: magazine full-page composition with generous margins"),
]


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


async def run_bucket(bucket: str, specs: list, *, live: bool, out: Path) -> dict:
    provider = (
        GeminiProvider(
            cost_per_image_usd=COST_PER_IMAGE_USD,
            image_config={"aspect_ratio": bucket},
        )
        if live
        else OfflineProvider(artifacts_per_call=1)
    )
    nodes = [
        Node(
            id=asset_id,
            label=f"Generate a {scene} {BRIEF}",
            failure_policy=FailurePolicy.RETRY,
            max_retries=2,
        )
        for asset_id, _, _, _, _, scene in specs
    ]
    graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)
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


async def main_async(live: bool, out: Path) -> dict:
    buckets: dict[str, list] = {}
    for spec in SPECS:
        buckets.setdefault(spec[4], []).append(spec)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(
        run_bucket(bucket, specs, live=live, out=out)
        for bucket, specs in buckets.items()
    ))
    generation_wall_s = time.perf_counter() - t0

    generated: dict[str, str] = {}
    for r in results:
        generated.update(r["paths"])
    total_cost = sum(r["cost"] for r in results)

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for asset_id, width, height, fmt, bucket, _ in SPECS:
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
        from PIL import Image
        with Image.open(dst) as img:
            ok = img.size == (width, height) and img.format == fmt
        entry["status"] = "PASS" if ok else "FAIL"
        entry["final_path"] = str(dst)
        report.append(entry)

    passed = sum(1 for e in report if e["status"] == "PASS")
    return {
        "benchmark": "osiris-asset-suite",
        "mode": "LIVE" if live else "offline",
        "model": MODEL if live else "offline",
        "assets_pass": f"{passed}/{len(SPECS)}",
        "generation_wall_s": round(generation_wall_s, 2),
        "buckets_run_concurrently": len(buckets),
        "cost_usd": round(total_cost, 4),
        "assets": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="smythe_artifacts/asset_suite")
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("GOOGLE_API_KEY"))
    if args.live and not live:
        print("--live requires GOOGLE_API_KEY; falling back to offline.")

    payload = asyncio.run(main_async(live, Path(args.out)))

    print(f"[{payload['mode']}] {payload['assets_pass']} specs met, "
          f"generation wall {payload['generation_wall_s']}s, "
          f"cost ${payload['cost_usd']}")
    for e in payload["assets"]:
        extra = (
            f" upscale x{e.get('upscale_factor')} crop {e.get('cropped_fraction')}"
            if "upscale_factor" in e else ""
        )
        print(f"  {e['asset']:<13} {e['spec']:<16} {e['status']}{extra}")

    results_path = args.results or (
        "benchmarks/results/asset_suite.json" if live
        else "benchmarks/results/asset_suite_offline.json"
    )
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
