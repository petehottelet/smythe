"""Benchmark a 64-node glyph fan-out and assemble a digital-rain showcase.

Offline (default, zero cost):
    python benchmarks/run_glyph_screensaver.py

Explicitly budgeted GPT Image run:
    python benchmarks/run_glyph_screensaver.py --live --concurrency 8 \
      --max-cost-per-call-usd 0.01 --max-budget-usd 0.64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmarks.artifact_records import environment_snapshot, portable_path  # noqa: E402
from benchmarks.glyph_screensaver_assets import (  # noqa: E402
    GLYPH_COUNT,
    GLYPH_SPECS,
    TILE_SIZE,
    ProceduralGlyphProvider,
    assemble_animation,
    assemble_atlas,
    assemble_html,
    assemble_preview,
    glyph_prompt,
    normalize_tile,
    render_glyph_tile,
)
from smythe import OpenAIImageProvider, Swarm  # noqa: E402
from smythe.graph import (  # noqa: E402
    ExecutionGraph,
    FailurePolicy,
    Node,
    NodeStatus,
    Topology,
)

DEFAULT_MODEL = "gpt-image-2"
DEFAULT_CONCURRENCIES = (1, 4, 8, 16)
# A quarter-second is short for an image API but large enough that this
# benchmark measures bounded async fan-out instead of mostly local PNG
# journaling. Use --latency-s 0 for a separate fixed-overhead smoke profile.
DEFAULT_LATENCY_S = 0.25
IMAGE_GENERATION_GUIDE = "https://developers.openai.com/api/docs/guides/image-generation"


def build_graph(*, glyph_count: int, estimated_cost_per_call_usd: float) -> ExecutionGraph:
    """Build one independent Smythe node per fictional glyph."""
    specs = GLYPH_SPECS[:glyph_count]
    nodes = [
        Node(
            id=spec.id,
            label=glyph_prompt(spec),
            failure_policy=FailurePolicy.HALT,
            max_retries=0,
            metadata={"estimated_cost_usd": estimated_cost_per_call_usd},
        )
        for spec in specs
    ]
    return ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)


def _receipt_dict(receipt) -> dict[str, Any]:
    record = asdict(receipt)
    record["path"] = portable_path(record["path"])
    return record


def _node_errors(graph: ExecutionGraph) -> list[dict[str, str]]:
    return [
        {
            "node_id": node.id,
            "status": node.status.value,
            "error": str(node.result or "node did not complete"),
        }
        for node in graph.nodes
        if node.status is not NodeStatus.COMPLETED
    ]


def _validate_and_normalize(
    graph: ExecutionGraph,
    *,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    tile_paths: list[str] = []
    validation_errors: list[str] = []
    tile_dir = run_dir / "tiles"

    for node in graph.nodes:
        artifacts = node.metadata.get("artifacts", [])
        if node.status is not NodeStatus.COMPLETED:
            validation_errors.append(f"{node.id}: node status is {node.status.value}")
            continue
        if len(artifacts) != 1:
            validation_errors.append(
                f"{node.id}: expected one artifact, found {len(artifacts)}"
            )
            continue
        source = Path(artifacts[0]["path"])
        try:
            receipt = normalize_tile(source, tile_dir / f"{node.id}.png")
        except Exception as exc:
            validation_errors.append(f"{node.id}: normalization failed: {exc}")
            continue
        record = _receipt_dict(receipt)
        record["node_id"] = node.id
        receipts.append(record)
        tile_paths.append(receipt.path)
        if receipt.format != "PNG":
            validation_errors.append(f"{node.id}: expected PNG, got {receipt.format}")
        if (receipt.width, receipt.height) != (TILE_SIZE, TILE_SIZE):
            validation_errors.append(
                f"{node.id}: expected {TILE_SIZE}x{TILE_SIZE}, "
                f"got {receipt.width}x{receipt.height}"
            )

    unique_hashes = len({receipt["sha256"] for receipt in receipts})
    expected = len(graph.nodes)
    if len(receipts) != expected:
        validation_errors.append(
            f"expected {expected} valid tiles, found {len(receipts)}"
        )
    if unique_hashes != expected:
        validation_errors.append(
            f"expected {expected} unique tile hashes, found {unique_hashes}"
        )
    validation = {
        "passed": not validation_errors,
        "expected_tiles": expected,
        "valid_png_tiles": len(receipts),
        "tile_size": [TILE_SIZE, TILE_SIZE],
        "unique_tile_hashes": unique_hashes,
        "errors": validation_errors,
    }
    return receipts, tile_paths, validation


async def _run_once(
    *,
    mode: str,
    out: Path,
    glyph_count: int,
    concurrency: int,
    latency_s: float,
    model: str,
    max_cost_per_call_usd: float,
    max_budget_usd: float,
) -> tuple[dict[str, Any], list[str]]:
    provider = (
        ProceduralGlyphProvider(latency_s=latency_s, tile_size=TILE_SIZE)
        if mode == "offline"
        else OpenAIImageProvider(
            size="1024x1024",
            quality="low",
            output_format="png",
            n=1,
            max_cost_per_call_usd=max_cost_per_call_usd,
        )
    )
    graph = build_graph(
        glyph_count=glyph_count,
        estimated_cost_per_call_usd=max_cost_per_call_usd,
    )
    run_dir = out / "runs" / f"concurrency-{concurrency}"
    swarm = Swarm(
        provider=provider,
        model="procedural-glyph-v1" if mode == "offline" else model,
        parallel=True,
        max_concurrency=concurrency,
        max_budget_usd=max_budget_usd,
        artifact_dir=run_dir / "raw",
    )

    started = time.perf_counter()
    result = None
    execution_error: str | None = None
    try:
        result = await swarm.execute_async(graph)
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    generation_wall_s = time.perf_counter() - started

    validation_started = time.perf_counter()
    tile_receipts, tile_paths, validation = _validate_and_normalize(
        graph, run_dir=run_dir
    )
    validation_wall_s = time.perf_counter() - validation_started
    errors = _node_errors(graph)
    if execution_error is not None:
        errors.insert(
            0,
            {
                "node_id": "__execution__",
                "status": "failed",
                "error": execution_error,
            },
        )
    errors.extend(
        {
            "node_id": "__validation__",
            "status": "failed",
            "error": error,
        }
        for error in validation["errors"]
    )

    completed = sum(node.status is NodeStatus.COMPLETED for node in graph.nodes)
    run = {
        "concurrency": concurrency,
        "status": "passed" if not errors and validation["passed"] else "failed",
        "execution_id": result.execution_id if result is not None else None,
        "generation_wall_s": round(generation_wall_s, 6),
        "validation_wall_s": round(validation_wall_s, 6),
        "end_to_end_wall_s": round(generation_wall_s + validation_wall_s, 6),
        "completed_nodes": completed,
        "throughput_glyphs_per_s": round(completed / generation_wall_s, 4),
        "cost_usd": round(result.total_cost_usd, 6) if result is not None else None,
        "cost_is_complete": result.cost_is_complete if result is not None else False,
        "cost_contains_estimates": (
            result.cost_contains_estimates if result is not None else mode == "live"
        ),
        "validation": validation,
        "tile_receipts": tile_receipts,
        "errors": errors,
    }
    return run, tile_paths


def _assemble(tile_paths: Sequence[str], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas = assemble_atlas(tile_paths, output_dir / "glyph-atlas.png")
    preview = assemble_preview(tile_paths, output_dir / "glyph-rain-preview.png")
    animation = assemble_animation(tile_paths, output_dir / "glyph-rain-loop.gif")
    html = assemble_html(output_dir / "glyph-rain.html")
    return {
        "wall_s": round(time.perf_counter() - started, 6),
        "atlas": _receipt_dict(atlas),
        "preview": _receipt_dict(preview),
        "animation": _receipt_dict(animation),
        "html": _receipt_dict(html),
    }


async def run_benchmark(
    *,
    live: bool,
    out: Path,
    glyph_count: int,
    concurrencies: Sequence[int],
    latency_s: float,
    model: str,
    max_cost_per_call_usd: float | None,
    max_budget_usd: float | None,
) -> dict[str, Any]:
    """Execute the configured benchmark and return its portable evidence record."""
    mode = "live" if live else "offline"
    if live:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("--live requires OPENAI_API_KEY; no offline fallback is allowed")
        if (
            max_cost_per_call_usd is None
            or not math.isfinite(max_cost_per_call_usd)
            or max_cost_per_call_usd <= 0
        ):
            raise ValueError("--live requires a positive --max-cost-per-call-usd")
        if (
            max_budget_usd is None
            or not math.isfinite(max_budget_usd)
            or max_budget_usd <= 0
        ):
            raise ValueError("--live requires a positive --max-budget-usd")
        required_ceiling = glyph_count * max_cost_per_call_usd
        if max_budget_usd + 1e-12 < required_ceiling:
            raise ValueError(
                "--max-budget-usd is below the full-run reservation: "
                f"need at least ${required_ceiling:.6f} for {glyph_count} calls"
            )
    else:
        max_cost_per_call_usd = 0.0
        max_budget_usd = 0.0
        # Import Pillow, initialize its codecs, and start asyncio's worker pool
        # before measuring concurrency 1. Otherwise one-time setup would be
        # misreported as parallel speedup for every later run.
        render_glyph_tile(GLYPH_SPECS[0])
        await asyncio.to_thread(lambda: None)

    runs: list[dict[str, Any]] = []
    paths_by_concurrency: dict[int, list[str]] = {}
    for concurrency in concurrencies:
        run, tile_paths = await _run_once(
            mode=mode,
            out=out,
            glyph_count=glyph_count,
            concurrency=concurrency,
            latency_s=latency_s,
            model=model,
            max_cost_per_call_usd=max_cost_per_call_usd,
            max_budget_usd=max_budget_usd,
        )
        runs.append(run)
        paths_by_concurrency[concurrency] = tile_paths

    baseline = next((run for run in runs if run["concurrency"] == 1), None)
    for run in runs:
        if baseline is None or not baseline["generation_wall_s"]:
            run["speedup_vs_concurrency_1"] = None
            run["parallel_efficiency"] = None
            continue
        speedup = baseline["generation_wall_s"] / run["generation_wall_s"]
        run["speedup_vs_concurrency_1"] = round(speedup, 4)
        run["parallel_efficiency"] = round(speedup / run["concurrency"], 4)

    successful = [run for run in runs if run["status"] == "passed"]
    fastest = min(successful, key=lambda run: run["generation_wall_s"], default=None)
    assembly = None
    if fastest is not None and glyph_count == GLYPH_COUNT:
        assembly = _assemble(
            paths_by_concurrency[fastest["concurrency"]], out / "assembled"
        )

    total_cost = sum(
        run["cost_usd"] for run in runs if run["cost_usd"] is not None
    )
    return {
        "benchmark": "glyph-screensaver-fanout",
        "record_version": 1,
        "mode": mode,
        "status": "passed" if len(successful) == len(runs) else "failed",
        "protocol": {
            "graph_nodes": glyph_count,
            "topology": Topology.BROADCAST_REDUCE.value,
            "one_provider_call_per_node": True,
            "concurrencies": list(concurrencies),
            "offline_latency_s": latency_s if not live else None,
            "normalization": f"PNG {TILE_SIZE}x{TILE_SIZE}",
            "assembly_requires_glyphs": GLYPH_COUNT,
        },
        "provider": {
            "name": "OpenAIImageProvider" if live else "ProceduralGlyphProvider",
            "model": model if live else "procedural-glyph-v1",
            "size": "1024x1024" if live else f"{TILE_SIZE}x{TILE_SIZE}",
            "quality": "low" if live else "deterministic-procedural",
            "max_cost_per_call_usd": max_cost_per_call_usd,
            "max_budget_usd": max_budget_usd,
            "official_guide": IMAGE_GENERATION_GUIDE if live else None,
        },
        "environment": environment_snapshot("smythe", "pillow", "openai"),
        "runs": runs,
        "fastest_concurrency": fastest["concurrency"] if fastest else None,
        "assembly": assembly,
        "total_recorded_cost_usd": round(total_cost, 6),
        "errors": [error for run in runs for error in run["errors"]],
    }


def _parse_concurrencies(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrencies must be comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("concurrencies must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("concurrencies must not contain duplicates")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the paid OpenAI Image API")
    parser.add_argument("--out", default=None, help="artifact output directory")
    parser.add_argument("--results", default=None, help="JSON evidence-record path")
    parser.add_argument(
        "--glyphs",
        type=int,
        default=GLYPH_COUNT,
        help="number of glyph nodes (1-64); assembly requires the default 64",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=DEFAULT_CONCURRENCIES,
        help="offline sweep, comma-separated (default: 1,4,8,16)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="single concurrency used with --live"
    )
    parser.add_argument(
        "--latency-s",
        type=float,
        default=DEFAULT_LATENCY_S,
        help="simulated per-call latency for the offline provider",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GPT Image model for --live")
    parser.add_argument("--max-cost-per-call-usd", type=float, default=None)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    args = parser.parse_args()

    if not 1 <= args.glyphs <= GLYPH_COUNT:
        parser.error(f"--glyphs must be between 1 and {GLYPH_COUNT}")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.latency_s < 0:
        parser.error("--latency-s must be non-negative")
    if not args.live and 1 not in args.concurrencies:
        parser.error("offline --concurrencies must include 1 for the speedup baseline")
    if not args.live and (
        args.max_cost_per_call_usd is not None or args.max_budget_usd is not None
    ):
        parser.error("paid budget flags are accepted only with --live")

    mode = "live" if args.live else "offline"
    out = Path(args.out or f"smythe_artifacts/glyph_screensaver/{mode}")
    results = Path(
        args.results or f"benchmarks/results/glyph_screensaver_{mode}.json"
    )
    concurrencies = (args.concurrency,) if args.live else args.concurrencies
    try:
        payload = asyncio.run(
            run_benchmark(
                live=args.live,
                out=out,
                glyph_count=args.glyphs,
                concurrencies=concurrencies,
                latency_s=args.latency_s,
                model=args.model,
                max_cost_per_call_usd=args.max_cost_per_call_usd,
                max_budget_usd=args.max_budget_usd,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    _write_json(results, payload)

    print(
        f"[{mode}] {args.glyphs} nodes; fastest concurrency="
        f"{payload['fastest_concurrency']}; cost=${payload['total_recorded_cost_usd']:.6f}"
    )
    for run in payload["runs"]:
        print(
            f"  c={run['concurrency']:<2} {run['status']:<6} "
            f"wall={run['generation_wall_s']:.3f}s "
            f"throughput={run['throughput_glyphs_per_s']:.1f}/s "
            f"speedup={run['speedup_vs_concurrency_1']}"
        )
    print(f"Wrote {results}")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
