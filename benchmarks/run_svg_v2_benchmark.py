"""Measure authored v2 contour compilation, validation and export through Smythe.

No model calls, simulated latency, cached SVG outputs, or design-time claims.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.artifact_records import environment_snapshot, portable_path  # noqa: E402
from benchmarks.run_svg_glyph_benchmark import (  # noqa: E402
    MemorySampler, _assemble, _make_pool, _validate_work, _write_json, build_graph,
)
from benchmarks.svg_glyph_measurements import find_near_matches, measure_glyph  # noqa: E402
from benchmarks.svg_glyphs import render_svg  # noqa: E402
from benchmarks.svg_v2_evidence import PROTOCOL, review_record  # noqa: E402
from screensaver.glyph_design_v2 import OUTLINES, generate_glyph  # noqa: E402
from smythe import Swarm  # noqa: E402
from smythe.graph import NodeStatus  # noqa: E402
from smythe.provider import CompletionResult, Provider  # noqa: E402

EXTRA_PATH = ROOT / "benchmarks/partitions/glyph_svg_v2_256/contours-extra.json"
EXTENDED = (*OUTLINES, *((r["recipe"], r["outline"]) for r in json.loads(EXTRA_PATH.read_bytes())))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes() -> dict:
    files = [p for p in (ROOT / "smythe").rglob("*.py")
             if "tmp" not in p.relative_to(ROOT).parts]
    files += [Path(__file__), EXTRA_PATH, *[ROOT / name for name in (
        "screensaver/glyph_design_v2.py", "screensaver/glyph_contours_v2.json",
        "benchmarks/svg_glyphs.py", "benchmarks/svg_glyph_measurements.py",
        "benchmarks/run_svg_glyph_benchmark.py", "benchmarks/artifact_records.py",
        "benchmarks/svg_v2_evidence.py",
        "benchmarks/svg_v2_protocol.md")]]
    return {portable_path(p): sha(p.read_bytes()) for p in sorted(set(files))}


def generate_work(index: int, executor: str) -> dict:
    clock = time.thread_time if executor == "thread" else time.process_time
    started, cpu = time.perf_counter(), clock()
    glyph = generate_glyph(index, outlines=EXTENDED)
    # The retained assembly format carries these descriptive fields. No random
    # seed is consumed: every complete contour recipe is fixed before sampling.
    glyph.update(family=glyph["recipe"], profile="v2-cut-contours", seed=0)
    return {"index": index, "payload": json.dumps(glyph, allow_nan=False),
            "worker_pid": os.getpid(), "worker_thread": threading.get_ident(),
            "worker_wall_s": time.perf_counter() - started, "worker_cpu_s": clock() - cpu}


def validate_work(index: int, payload: str, executor: str) -> dict:
    clock = time.thread_time if executor == "thread" else time.process_time
    started, cpu = time.perf_counter(), clock()
    result = _validate_work(index, payload, 128, executor)
    if result["status"] == "passed":
        try:
            if result["glyph_id"] != f"GLYPH-{index:03d}":
                raise ValueError("Glyph identity does not match its task")
            baseline = result["measurements"]
            if not baseline["threshold_topology_stable"]:
                raise ValueError("128 px threshold topology changed")
            small = {}
            for size in (16, 32, 64):
                measured = measure_glyph(render_svg(result["svg"], size))
                if measured["blank"] or any(measured[key] != baseline[key]
                                            for key in ("components_raw", "holes_raw")):
                    raise ValueError(f"{size} px components or counters changed")
                small[str(size)] = measured
            result["small_sizes"] = small
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["worker_wall_s"] = time.perf_counter() - started
    result["worker_cpu_s"] = clock() - cpu
    return result


class V2Provider(Provider):
    """One fresh geometry compilation per actual Smythe provider call."""

    def __init__(self, pool, executor):
        self.pool, self.executor = pool, executor
        self.calls = []
        self.in_flight = self.max_in_flight = 0

    def budget_estimate_usd(self, model):
        return 0.0

    async def complete(self, system, prompt, model):
        match = re.search(r"SVG_BENCH_REQUEST:(\{[^\r\n]+\})", prompt)
        if match is None:
            raise ValueError("Only explicit glyph compilation tasks are accepted")
        request = json.loads(match.group(1))
        if request["attempt"] != 0:
            raise ValueError("This protocol permits exactly one attempt")
        index = request["index"]
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        started = time.perf_counter()
        try:
            record = await asyncio.get_running_loop().run_in_executor(
                self.pool, generate_work, index, self.executor)
            self.calls.append({k: v for k, v in record.items() if k != "payload"}
                              | {"status": "passed", "provider_wall_s": time.perf_counter() - started})
            return CompletionResult(text=record["payload"], cost_usd=0.0,
                                    prompt_tokens=0, completion_tokens=0)
        except Exception as exc:
            self.calls.append({"index": index, "status": "failed", "error": str(exc)})
            raise
        finally:
            self.in_flight -= 1


def run_once(*, glyph_count, concurrency, executor, repeat, worker_cap, out):
    from PIL import Image

    workers = min(concurrency, worker_cap, os.cpu_count() or 1)
    started, cpu = time.perf_counter(), time.process_time()
    memory = MemorySampler()
    memory.start()
    pool = _make_pool(executor, workers)
    graph = build_graph(glyph_count)
    provider = V2Provider(pool, executor)
    swarm = Swarm(provider=provider, model="authored-svg-local-v2", parallel=True,
                  max_concurrency=concurrency, max_budget_usd=0.0, artifact_dir=None)
    phases = {"setup_wall_s": time.perf_counter() - started}
    errors, validated, distinctness, assembly, result = [], [], None, None, None
    try:
        phase = time.perf_counter()
        try:
            result = asyncio.run(swarm.execute_async(graph))
        except Exception as exc:
            errors.append(f"Execution: {type(exc).__name__}: {exc}")
        phases["generation_wall_s"] = time.perf_counter() - phase
        phase = time.perf_counter()
        futures = []
        for i, node in enumerate(graph.nodes):
            if node.status is NodeStatus.COMPLETED:
                futures.append((i, pool.submit(validate_work, i, str(node.result), executor)))
            else:
                errors.append(f"Incomplete node: {node.id}")
        for i, future in futures:
            try:
                validated.append(future.result())
            except Exception as exc:
                validated.append({"index": i, "status": "failed", "error": str(exc)})
        good = sorted((r for r in validated if r["status"] == "passed"), key=lambda r: r["index"])
        errors.extend(r["error"] for r in validated if r["status"] != "passed")
        if len(good) != glyph_count or len(provider.calls) != glyph_count:
            errors.append("Missing validated glyphs or provider calls")
        if len(good) == glyph_count:
            images = [Image.frombytes("RGBA", (128, 128), r["pixels"]) for r in good]
            distinctness = find_near_matches(images, [r["glyph_id"] for r in good])
            if distinctness["near_matches"]:
                errors.append("Aligned/reflected silhouettes exceed the frozen overlap threshold")
            for key in ("svg_sha256", "pixel_sha256", "glyph_id"):
                if len({r[key] for r in good}) != glyph_count:
                    errors.append(f"Duplicate {key}")
        phases["validation_wall_s"] = time.perf_counter() - phase
        phase = time.perf_counter()
        if not errors:
            assembly = _assemble(good, Path(out), overwrite=False,
                                 catalog_style={"scope": "v2 topology and distinctness gates",
                                                "distinctness": distinctness})
        phases["assembly_wall_s"] = time.perf_counter() - phase
    except Exception as exc:
        errors.append(f"Workflow: {type(exc).__name__}: {exc}")
    finally:
        phase = time.perf_counter()
        pool.shutdown(wait=True, cancel_futures=True)
        phases["worker_shutdown_wall_s"] = time.perf_counter() - phase
        resources = memory.finish()
    elapsed = time.perf_counter() - started
    parent_cpu = time.process_time() - cpu
    receipts = []
    for r in sorted(validated, key=lambda r: r["index"]):
        receipt = {k: r[k] for k in ("index", "glyph_id", "status", "svg_sha256", "pixel_sha256") if k in r}
        if r["status"] == "passed":
            receipt["measurement_sha256"] = sha(json.dumps(
                {"128": r["measurements"], **r["small_sizes"]}, sort_keys=True, allow_nan=False).encode())
        receipts.append(receipt)
    calls = provider.calls
    worker_cpu = None if executor == "thread" else sum(
        r.get("worker_cpu_s", 0) for r in [*calls, *validated])
    if executor == "thread":
        calls = [{k: v for k, v in r.items() if k != "worker_cpu_s"} for r in calls]
    return {"glyph_count": glyph_count, "executor": executor, "concurrency": concurrency,
            "workers": workers, "repeat": repeat, "status": "passed" if not errors else "failed",
            "execution_id": result.execution_id if result else None,
            "errors": errors, **phases, "end_to_end_wall_s": elapsed,
            "parent_process_cpu_s": parent_cpu, "worker_task_cpu_s": worker_cpu,
            "worker_cpu_scope": "process task CPU excludes worker startup/shutdown; threads use parent CPU only",
            "memory": resources, "provider_calls": len(calls),
            "max_in_flight_provider_calls": provider.max_in_flight,
            "completed_nodes": sum(n.status is NodeStatus.COMPLETED for n in graph.nodes),
            "valid_glyphs": sum(r["status"] == "passed" for r in receipts),
            "smythe_recorded_cost_usd": result.total_cost_usd if result else None,
            "api_calls": 0, "api_cost_usd": 0, "calls": calls, "glyphs": receipts,
            "distinctness": distinctness, "assembly": assembly}


def run_benchmark(out: Path, *, sizes=(192, 256), concurrencies=(1, 4, 8),
                  executors=("thread", "process"), repeats=3, worker_cap=8, seed=20260913):
    for values, label, maximum in ((sizes, "sizes", 256), (concurrencies, "concurrencies", 256)):
        if not values or len(set(values)) != len(values) or any(type(n) is not int or not 1 <= n <= maximum for n in values):
            raise ValueError(f"Invalid {label}")
    if 1 not in concurrencies or not executors or len(set(executors)) != len(executors) or any(e not in ("thread", "process") for e in executors):
        raise ValueError("Unique thread/process backends and a c1 baseline are required")
    if type(repeats) is not int or not 1 <= repeats <= 20 or type(worker_cap) is not int or not 1 <= worker_cap <= 64:
        raise ValueError("Invalid repetition or worker bounds")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    schedule = []
    rng = random.Random(seed)
    for r in range(1, repeats + 1):
        block = [{"glyph_count": n, "executor": e, "concurrency": c, "repeat": r}
                 for n in sizes for e in executors for c in concurrencies]
        rng.shuffle(block)
        schedule.extend(block)
    sources = source_hashes()
    plan = {"protocol_version": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sources, "sizes": list(sizes), "concurrencies": list(concurrencies),
            "executors": list(executors), "repeats": repeats, "worker_cap": worker_cap,
            "seed": seed, "schedule": schedule, "simulated_latency_s": 0,
            "cached_svg_outputs": False, "design_work_timed": False,
            "scope": "Compile fixed authored contours, validate at 16/32/64/128 px, compare every aligned/reflected pair, export every catalog; no planning model"}
    _write_json(out / "plan.json", plan)
    environment = environment_snapshot("smythe", "pillow", "numpy", "scipy", "shapely", "psutil")
    try:
        import psutil
        environment["physical_memory_bytes"] = psutil.virtual_memory().total
    except ImportError:
        environment["physical_memory_bytes"] = None
    environment["desktop_scope"] = "Existing desktop apps remain open; no idle-host claim. Owned tests, builds and other benchmarks must finish before sampling."
    record = {"protocol_version": PROTOCOL, "protocol": plan,
              "environment": environment,
              "logical_cpu_count": os.cpu_count() or 1, "runs": [], "claimable": False}
    for position, trial in enumerate(schedule, 1):
        if source_hashes() != sources:
            raise ValueError("Frozen sources changed before a trial")
        print(f"{position}/{len(schedule)}: {trial}", flush=True)
        name = f"n{trial['glyph_count']}-{trial['executor']}-c{trial['concurrency']}-r{trial['repeat']}"
        attempt_started_at = datetime.now(timezone.utc).isoformat()
        run = run_once(**trial, worker_cap=worker_cap, out=out / "runs" / name)
        run["attempt_started_at"] = attempt_started_at
        run["attempt_finished_at"] = datetime.now(timezone.utc).isoformat()
        run["source_stable"] = source_hashes() == sources
        _write_json(out / "receipts" / f"{name}.json", run)
        record["runs"].append(run)
        print(f"  {run['status']}: {run['end_to_end_wall_s']:.3f}s; {run['valid_glyphs']} valid", flush=True)
    try:
        record["summaries"] = review_record(record)
        record["status"] = "passed"
        record["claimable"] = repeats >= 3 and set(sizes) == {192, 256}
        record["known_measurement_defects"] = []
    except ValueError as exc:
        record.update(status="failed", known_measurement_defects=[str(exc)], summaries=[])
    record["evidence_status"] = "claimable-local-authored-workflow" if record["claimable"] else "diagnostic"
    record["claim_scope"] = "This authored contour compilation/validation/export workload and host; no model quality, API throughput, GPU timing or cross-platform performance claim"
    _write_json(out / "results.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Four-glyph thread/process checks; never claimable")
    args = parser.parse_args()
    kwargs = {"sizes": (4,), "concurrencies": (1, 2), "repeats": 1} if args.smoke else {}
    record = run_benchmark(args.out, **kwargs)
    print(json.dumps({"status": record["status"], "claimable": record["claimable"], "summaries": record["summaries"]}, indent=2))
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
