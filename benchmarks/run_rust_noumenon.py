"""Compare native and Python contour compilation, then verify native catalogs.

Compilation is the only matched timing boundary. Full pipeline times are
separate diagnostics: the native gates omit Python's descriptive style metrics,
and the two rasterizers and persistence policies differ. No paid calls occur.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_svg_glyph_benchmark import _make_pool, build_graph  # noqa: E402
from benchmarks.run_svg_v2_benchmark import (  # noqa: E402
    EXTENDED, V2Provider, run_once as python_full, validate_work,
)
from benchmarks.noumenon.glyph_design_v2 import generate_glyph  # noqa: E402
from smythe import Swarm  # noqa: E402

PROTOCOL = "noumenon-rust-comparison-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    tracked = subprocess.check_output(["git", "ls-files", "smythe", "benchmarks"], cwd=ROOT,
                                      text=True).splitlines()
    paths = {ROOT / name for name in tracked if name.endswith(".py")}
    paths.update((ROOT / "crates").rglob("*.rs"))
    paths.update((ROOT / "crates").glob("*/Cargo.toml"))
    paths.update(ROOT / name for name in (
        "Cargo.toml", "Cargo.lock", "benchmarks/run_rust_noumenon.py",
        "benchmarks/rust_noumenon_protocol.md", "benchmarks/noumenon/glyph_contours_v2.json",
        "benchmarks/partitions/glyph_svg_v2_256/contours-extra.json"))
    return {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}


def python_compile(count, workers, backend):
    started = time.perf_counter()
    pool = _make_pool(backend, workers)
    try:
        graph = build_graph(count)
        provider = V2Provider(pool, backend)
        swarm = Swarm(provider=provider, model="authored-svg-local-v2", parallel=True,
                      max_concurrency=workers, max_budget_usd=0.0, artifact_dir=None)
        asyncio.run(swarm.execute_async(graph))
        elapsed = time.perf_counter() - started
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    glyphs = [json.loads(node.result) for node in graph.nodes]
    return {"compilation_wall_s": elapsed, "provider_calls": len(provider.calls),
            "svg_sha256": [g["svg_sha256"] for g in glyphs]}


def native(binary, count, workers, out, *, mode="compile", resume=False):
    command = [str(binary), "noumenon", "--glyphs", str(count), "--workers", str(workers),
               "--out", str(out), "--mode", mode]
    if resume:
        command.append("--resume")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - started
    if result.returncode:
        # Do not retain arbitrary subprocess diagnostics in benchmark evidence.
        raise RuntimeError("Native pipeline failed; no comparison is claimable")
    report = json.loads(result.stdout)
    if not resume:
        if report.get("optimized") is not True:
            raise ValueError("A release binary is required")
        sources = report.get("implementation_sources", {})
        if not sources or any(digest((ROOT / name).read_bytes()) != value
                              for name, value in sources.items()):
            raise ValueError("Native binary does not match the current sources")
    report["process_wall_s"] = elapsed
    return report


def verify_catalog(out, count):
    catalog = json.loads((out / "catalog.json").read_bytes())
    if catalog["glyph_count"] != count or len(catalog["glyphs"]) != count:
        raise ValueError("Incomplete native catalog")
    hashes = []
    for i, entry in enumerate(catalog["glyphs"]):
        expected = generate_glyph(i, outlines=EXTENDED)
        actual = (out / f"GLYPH-{i:03}.svg").read_bytes()
        if (entry["glyph_id"] != expected["glyph_id"] or actual != expected["svg"].encode()
                or entry["svg_sha256"] != digest(actual)):
            raise ValueError("Native SVG differs from Python reference")
        hashes.append(digest(actual))
    return hashes


def _oracle_glyph(item):
    i, svg = item
    glyph = generate_glyph(i, outlines=EXTENDED)
    glyph.update(svg=svg, svg_sha256=digest(svg.encode()), sha256=digest(svg.encode()),
                 family=glyph["recipe"], profile="v2-cut-contours", seed=0)
    result = validate_work(i, json.dumps(glyph), "process")
    if result["status"] != "passed":
        raise ValueError("Native SVG failed the existing Python four-size validator")
    return {"glyph_id": result["glyph_id"], "svg_sha256": result["svg_sha256"],
            "pixel_sha256": result["pixel_sha256"],
            "measurement_sha256": digest(json.dumps(
                {"128": result["measurements"], **result["small_sizes"]},
                sort_keys=True, allow_nan=False).encode())}


def verify_full(out, count, workers):
    """Independent original v2 gates, outside all compared timing intervals."""
    verify_catalog(out, count)
    items = [(i, (out / f"GLYPH-{i:03}.svg").read_text(encoding="utf-8")) for i in range(count)]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        receipts = list(pool.map(_oracle_glyph, items))
    # The SVGs are byte-identical, so their Pillow renders and original all-pairs
    # gate must match. Run the gate independently instead of assuming it.
    from benchmarks.svg_glyph_measurements import find_near_matches
    from benchmarks.svg_glyphs import render_svg
    distinctness = find_near_matches([render_svg(svg) for _, svg in items],
                                    [r["glyph_id"] for r in receipts])
    if distinctness["near_matches"]:
        raise ValueError("Native catalog failed the existing Python distinctness gate")
    return {"status": "passed", "glyphs": receipts,
            "compared_pairs": distinctness["compared_pairs"], "near_matches": []}


def review(record):
    """Refuse missing/duplicate trials, changing outputs and invalid clocks."""
    plan = record["plan"]
    expected = {(n, w, b, r) for n in plan["sizes"] for w in plan["workers"]
                for b in ("rust", "python-thread", "python-process")
                for r in range(1, plan["repeats"] + 1)}
    seen, groups, outputs = set(), {}, {}
    for trial in record["trials"]:
        key = (trial["count"], trial["workers"], trial["backend"], trial["repeat"])
        elapsed = trial["compilation_wall_s"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate trial")
        if not math.isfinite(elapsed) or elapsed <= 0 or trial["provider_calls"] != key[0]:
            raise ValueError("Invalid clock or dispatch count")
        hashes = trial["svg_sha256"]
        if len(hashes) != key[0] or len(set(hashes)) != key[0]:
            raise ValueError("Incomplete or duplicate glyph output")
        if key[0] in outputs and outputs[key[0]] != hashes:
            raise ValueError("Cross-language or cross-trial outputs differ")
        if not trial["source_stable"]:
            raise ValueError("Source changed during trial")
        outputs[key[0]] = hashes
        seen.add(key)
        groups.setdefault(key[:3], []).append(elapsed)
    if seen != expected:
        raise ValueError("Missing trial")
    if 192 in outputs and 256 in outputs and outputs[192] != outputs[256][:192]:
        raise ValueError("Extension changed the approved glyphs")
    return [{"count": n, "workers": w, "backend": b, "median_s": statistics.median(times),
             "min_s": min(times), "max_s": max(times), "repeats": len(times)}
            for (n, w, b), times in sorted(groups.items())]


def run_benchmark(binary, out, *, sizes=(192, 256), workers=(1, 4, 8), repeats=3):
    if (not sizes or len(set(sizes)) != len(sizes) or any(type(n) is not int or not 1 <= n <= 256 for n in sizes)
            or not workers or len(set(workers)) != len(workers) or 1 not in workers
            or any(type(w) is not int or not 1 <= w <= 64 for w in workers)
            or type(repeats) is not int or not 1 <= repeats <= 20):
        raise ValueError("Invalid benchmark matrix")
    out.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    binary_sha = digest(binary.read_bytes())
    schedule = [(n, w, b, r) for r in range(1, repeats + 1) for n in sizes
                for w in workers for b in ("rust", "python-thread", "python-process")]
    random.Random(20261107).shuffle(schedule)
    plan = {"protocol": PROTOCOL, "sizes": list(sizes), "workers": list(workers), "repeats": repeats,
            "schedule": schedule, "source_sha256": sources, "binary_sha256": binary_sha,
            "timing_scope": "graph construction, pool startup including child imports, one fresh authored compilation per node and dispatch through each native Swarm representation; no parent imports, pool shutdown, raster validation, persistence or export",
            "seed": 20261107, "api_calls": 0, "cached_outputs": False}
    write_json(out / "plan.json", plan)
    record = {"plan": plan, "created_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"system": platform.system(), "release": platform.release(),
                              "python": platform.python_version(), "logical_cpus": os.cpu_count(),
                              "architecture": platform.machine(),
                              "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                              "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
                              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              "source_identity": "Exact source hashes include unpublished native preview files",
                              "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "shapely", "pillow")},
                              "desktop_scope": "Existing desktop applications remain open; no idle-host claim"},
              "trials": [], "full_pipeline": [], "claimable": False, "evidence_status": "diagnostic-native-preview"}
    for n, w, backend, repeat in schedule:
        if source_hashes() != sources or digest(binary.read_bytes()) != binary_sha:
            raise ValueError("Frozen sources or binary changed")
        print(f"Compilation: {backend} n={n} workers={w} repeat={repeat}", flush=True)
        if backend == "rust":
            folder = out / "catalogs" / f"n{n}-w{w}-r{repeat}"
            measured = native(binary, n, w, folder)
            trial = {key: measured[key] for key in ("compilation_wall_s", "provider_calls")}
            trial["svg_sha256"] = verify_catalog(folder, n)
        else:
            trial = python_compile(n, w, backend.removeprefix("python-"))
        trial.update(count=n, workers=w, backend=backend, repeat=repeat,
                     source_stable=source_hashes() == sources)
        record["trials"].append(trial)
        write_json(out / "trials" / f"{backend}-n{n}-w{w}-r{repeat}.json", trial)
    record["compilation_summaries"] = review(record)
    for n in sizes:
        for repeat in range(1, repeats + 1):
            w = max(workers)
            folder = out / "full" / f"rust-n{n}-r{repeat}"
            print(f"Full workflows: n={n} workers={w} repeat={repeat}", flush=True)
            rust = native(binary, n, w, folder, mode="full")
            python = python_full(glyph_count=n, concurrency=w, executor="process", repeat=repeat,
                                 worker_cap=w, out=out / "full" / f"python-n{n}-r{repeat}")
            if python["status"] != "passed":
                raise ValueError("Python full workflow failed")
            oracle = verify_full(folder, n, w)
            write_json(out / "oracle" / f"n{n}-r{repeat}.json", oracle)
            replay = native(binary, n, w, folder, mode="full", resume=True)
            if replay["provider_calls"] != 0:
                raise ValueError("Completed resume dispatched work")
            record["full_pipeline"].append({"count": n, "repeat": repeat, "workers": w,
                "rust": rust, "python_v2": {key: python[key] for key in (
                    "status", "end_to_end_wall_s", "generation_wall_s", "validation_wall_s", "assembly_wall_s",
                    "memory", "glyphs", "distinctness", "provider_calls", "api_calls")},
                "python_oracle": oracle, "python_oracle_passed": True, "resume_provider_calls": 0,
                "timings_comparable": False})
    if source_hashes() != sources or digest(binary.read_bytes()) != binary_sha:
        raise ValueError("Frozen sources or binary changed")
    record["status"] = "passed"
    record["full_pipeline_scope"] = "Separate diagnostic clocks: tiny-skia vs Pillow; native topology gates vs full Python style measurements; durable native receipts vs Python catalog export. No end-to-end speedup claim."
    write_json(out / "results.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release" / ("smythe.exe" if os.name == "nt" else "smythe"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    options = {"sizes": (4,), "workers": (1, 2), "repeats": 1} if args.smoke else {}
    run_benchmark(args.binary.resolve(), args.out, **options)


if __name__ == "__main__":
    main()
