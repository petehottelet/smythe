"""Parallel image-generation benchmark: the concurrency sweep.

The claim under test: Smythe's parallel executor converts provider
latency into wall-clock speedup at near-ideal efficiency, under an
honest budget cap. Every metric here is objective — wall time, per-node
latency, cost, decode/format compliance, pairwise dHash diversity — so
none of it inherits LLM-judge variance.

    python benchmarks/run_image_benchmarks.py                     # offline mechanics
    python benchmarks/run_image_benchmarks.py --live --k 1,3,8 --repeats 3
    python benchmarks/run_image_benchmarks.py --live --k 8 --images 25 --repeats 1

Offline mode (default, no keys consumed) exercises the full harness with
OfflineProvider's deterministic PNGs; timing in offline mode measures
the OS scheduler, not the work, and is reported but not meaningful.
Live mode requires GOOGLE_API_KEY and spends real money: images x
repeats x len(k) x cost_per_image (default $0.039).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parents[1]))

from smythe import OfflineProvider, Swarm  # noqa: E402
from smythe.graph import ExecutionGraph, FailurePolicy, Node, Topology  # noqa: E402
from smythe.provider import GeminiProvider  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

COST_PER_IMAGE_USD = 0.039
DEFAULT_MODEL = "gemini-2.5-flash-image"

CONSTRAINTS = (
    "Photorealistic natural photography, physically plausible light and "
    "atmosphere, complete bridge clearly recognizable, no text, no logos, "
    "no watermark, no fantastical architecture, no duplicate bridge. "
)

SCENES = [
    "dawn fog flowing through the towers, viewed from Battery Spencer",
    "crisp clear afternoon from the Marin Headlands, city skyline beyond",
    "blue hour from the city approach, vehicle light trails, bay reflections",
    "dramatic storm clouds breaking, shafts of sunlight on the deck",
    "aerial view from above the south tower looking north",
    "golden sunset from Baker Beach with waves in the foreground",
    "clear night under a full moon, deck lights reflected in the bay",
    "winter rain, wet roadway sheen, moody low clouds on the towers",
]


def build_graph(n_images: int) -> ExecutionGraph:
    nodes = [
        Node(
            id=f"img-{i:02d}",
            label=(
                f"Generate a realistic editorial photograph of the Golden Gate "
                f"Bridge. {CONSTRAINTS}Scene: {SCENES[i % len(SCENES)]}."
            ),
            failure_policy=FailurePolicy.RETRY,
            max_retries=2,
        )
        for i in range(n_images)
    ]
    return ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)


def dhash64(path: str) -> int | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path).convert("L").resize((9, 8))
    except Exception:
        return None
    px = list(img.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def analyze_artifacts(records: list[dict]) -> dict:
    """Objective checks: decode, dimensions, pairwise dHash distances."""
    try:
        from PIL import Image
    except ImportError:
        return {"decoded": None, "note": "pillow not installed"}
    decoded, sizes, hashes = 0, [], []
    for rec in records:
        try:
            with Image.open(rec["path"]) as img:
                img.load()
                sizes.append(list(img.size))
            decoded += 1
        except Exception:
            sizes.append(None)
            continue
        h = dhash64(rec["path"])
        if h is not None:
            hashes.append(h)
    distances = [
        bin(hashes[i] ^ hashes[j]).count("1")
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
    ]
    return {
        "decoded": decoded,
        "total": len(records),
        "sizes": sizes,
        "dhash_pairwise_min": min(distances) if distances else None,
        "dhash_pairwise_mean": round(mean(distances), 1) if distances else None,
    }


def run_once(*, k: int, n_images: int, live: bool, model: str, out_dir: Path) -> dict:
    provider = (
        GeminiProvider(cost_per_image_usd=COST_PER_IMAGE_USD)
        if live
        else OfflineProvider(artifacts_per_call=1)
    )
    graph = build_graph(n_images)
    swarm = Swarm(
        provider=provider,
        model=model if live else "demo-image-model",
        parallel=True,
        max_concurrency=k,
        max_budget_usd=max(2.0, n_images * COST_PER_IMAGE_USD * 1.5),
        artifact_dir=out_dir,
        retry_backoff_s=2.0,
    )
    t0 = time.perf_counter()
    result = swarm.execute(graph)
    wall_s = time.perf_counter() - t0

    durations = [
        span["duration_ms"] / 1000
        for span in result.trace
        if "duration_ms" in span
    ]
    completed = [n for n in result.graph.nodes if n.status.value == "completed"]
    records = [r for n in completed for r in n.metadata.get("artifacts", [])]
    concurrency_factor = (sum(durations) / wall_s) if wall_s and durations else None
    return {
        "k": k,
        "images_requested": n_images,
        "images_completed": len(records),
        "nodes_ok": f"{len(completed)}/{n_images}",
        "wall_s": round(wall_s, 2),
        "sum_node_latency_s": round(sum(durations), 2),
        "node_latency_s": [round(d, 2) for d in durations],
        "concurrency_factor": round(concurrency_factor, 2) if concurrency_factor else None,
        "efficiency_vs_ideal": (
            round(concurrency_factor / min(k, n_images), 3)
            if concurrency_factor
            else None
        ),
        "throughput_images_per_min": (
            round(len(records) * 60 / wall_s, 1) if wall_s else None
        ),
        "cost_usd": round(result.total_cost_usd, 4),
        "artifacts": analyze_artifacts(records) if live else {"offline": True},
        "execution_id": result.execution_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", default="1,3", help="comma-separated concurrency levels")
    parser.add_argument("--images", type=int, default=8, help="images per run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--live", action="store_true", help="spend real money")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument(
        "--artifact-dir", default="smythe_artifacts/image_benchmark",
    )
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("GOOGLE_API_KEY"))
    if args.live and not live:
        print("--live requires GOOGLE_API_KEY; falling back to offline.")
    ks = [int(x) for x in args.k.split(",")]

    est = len(ks) * args.repeats * args.images * COST_PER_IMAGE_USD
    mode = "LIVE" if live else "offline"
    print(f"[{mode}] k={ks} images={args.images} repeats={args.repeats}"
          + (f"  estimated cost ${est:.2f}" if live else ""))

    runs = []
    for k in ks:
        for rep in range(args.repeats):
            record = run_once(
                k=k, n_images=args.images, live=live, model=args.model,
                out_dir=Path(args.artifact_dir),
            )
            record["repeat"] = rep
            runs.append(record)
            print(
                f"  k={k} rep={rep}: wall={record['wall_s']}s "
                f"factor={record['concurrency_factor']} "
                f"eff={record['efficiency_vs_ideal']} "
                f"cost=${record['cost_usd']}"
            )

    summary = []
    for k in ks:
        cell = [r for r in runs if r["k"] == k]
        walls = [r["wall_s"] for r in cell]
        effs = [r["efficiency_vs_ideal"] for r in cell if r["efficiency_vs_ideal"]]
        summary.append({
            "k": k,
            "runs": len(cell),
            "wall_s_mean": round(mean(walls), 2),
            "wall_s_range": [min(walls), max(walls)],
            "efficiency_mean": round(mean(effs), 3) if effs else None,
            "throughput_mean_ipm": round(
                mean(r["throughput_images_per_min"] for r in cell), 1,
            ),
            "cost_total_usd": round(sum(r["cost_usd"] for r in cell), 4),
        })

    payload = {
        "benchmark": "image-concurrency-sweep",
        "mode": mode,
        "model": args.model if live else "offline",
        "cost_per_image_usd": COST_PER_IMAGE_USD if live else 0,
        "images_per_run": args.images,
        "summary": summary,
        "runs": runs,
    }
    out = args.out or (
        "benchmarks/results/image_k_sweep.json" if live
        else "benchmarks/results/image_k_sweep_offline.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
