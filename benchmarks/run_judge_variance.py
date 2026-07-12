"""Judge variance: re-score identical assets N times.

    python benchmarks/run_judge_variance.py --results benchmarks/results/asset_suite_metacortex_datacentres.json --runs 5

The optimizer loop (and any claim that a score delta means anything)
is gated on knowing how much the vision judge's numbers move when the
pixels don't. This re-judges the exact same finished assets N times and
reports the spread per asset and overall.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from run_asset_suite import judge_assets, load_brand  # noqa: E402


async def main_async(args) -> dict:
    prior = json.loads(Path(args.results).read_text(encoding="utf-8"))
    brand = load_brand(args.brand)
    logo_path = prior["brand_logo"]["path"]
    finals = {
        e["asset"]: e["final_path"]
        for e in prior["assets"]
        if e.get("final_path")
    }
    runs = []
    for i in range(args.runs):
        verdict = await judge_assets(
            brand=brand, live=True, out=Path(args.out),
            logo_path=logo_path, finals=finals,
        )
        overall = verdict["overall"]
        per_asset = {
            k: v.get("brand_consistency") for k, v in verdict["scores"].items()
        }
        runs.append({"overall": overall, "per_asset": per_asset,
                     "cost_usd": verdict["cost_usd"]})
        print(f"  run {i}: overall={overall} per-asset={per_asset}")

    overalls = [r["overall"] for r in runs if r["overall"] is not None]
    spread: dict = {
        "overall": {
            "values": overalls,
            "mean": round(mean(overalls), 2),
            "stdev": round(pstdev(overalls), 2),
            "range": [min(overalls), max(overalls)],
        },
        "per_asset": {},
    }
    for asset in finals:
        vals = [
            r["per_asset"].get(asset) for r in runs
            if r["per_asset"].get(asset) is not None
        ]
        if vals:
            spread["per_asset"][asset] = {
                "mean": round(mean(vals), 2),
                "stdev": round(pstdev(vals), 2),
                "range": [min(vals), max(vals)],
            }
    return {
        "benchmark": "judge-variance",
        "judged_results": args.results,
        "runs": args.runs,
        "cost_usd": round(sum(r["cost_usd"] for r in runs), 4),
        "spread": spread,
        "raw_runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="asset-suite results JSON to re-judge")
    parser.add_argument("--brand", default="benchmarks/brands/metacortex.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="smythe_artifacts/judge_variance")
    args = parser.parse_args()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY required (live judging only)")
    payload = asyncio.run(main_async(args))
    out = Path("benchmarks/results/judge_variance.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = payload["spread"]["overall"]
    print(f"overall: mean={s['mean']} stdev={s['stdev']} range={s['range']} "
          f"(cost ${payload['cost_usd']})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
