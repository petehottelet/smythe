"""Measure fresh original SVG generation through Smythe, with no paid calls.

python benchmarks/run_svg_glyph_benchmark.py --glyphs 192 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.artifact_records import environment_snapshot, portable_path  # noqa: E402
from smythe import Swarm  # noqa: E402
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology  # noqa: E402
from smythe.provider import CompletionResult, Provider  # noqa: E402

PROTOCOL_VERSION = "original-svg-workflow-v1"
DEFAULT_CONCURRENCIES = (1, 4, 8, 16, 32)
DEFAULT_EXECUTORS = ("thread", "process")
DEFAULT_OUT = Path("smythe_artifacts/svg_glyph_v1")
DEFAULT_RESULTS = Path("benchmarks/partitions/glyph_svg_v1/results.json")
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs/data/glyph-style-summary.json"
AUTHORING_FIELDS = ("authoring_parameters", "recipe", "attempt", "version", "authoring_weight")


def _api():
    return importlib.import_module("benchmarks.svg_glyphs")


def _measurement_api():
    return importlib.import_module("benchmarks.svg_glyph_measurements")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seconds(ns: int) -> float:
    return ns / 1_000_000_000


def _generate_work(index: int, attempt: int, cpu_basis: str = "process") -> dict[str, Any]:
    """Every invocation constructs a fresh glyph inside the provider boundary."""
    cpu_clock = time.thread_time_ns if cpu_basis == "thread" else time.process_time_ns
    wall, cpu = time.perf_counter_ns(), cpu_clock()
    result: dict[str, Any] = {"index": index, "attempt": attempt, "worker_pid": os.getpid(),
                                   "worker_thread": threading.get_ident()}
    try:
        glyph = _api().generate_glyph(index, attempt=attempt)
        if not isinstance(glyph, dict) or not isinstance(glyph.get("svg"), str):
            raise ValueError("generate_glyph must return a dict containing an SVG string")
        for key in ("glyph_id", "family", "profile", "seed"):
            if key not in glyph:
                raise ValueError(f"Generated glyph is missing {key}")
        if any(not isinstance(glyph[k], str) or not glyph[k] for k in ("glyph_id", "family", "profile")):
            raise ValueError("glyph_id, family, and profile must be nonempty strings")
        if isinstance(glyph["seed"], bool) or not isinstance(glyph["seed"], int):
            raise ValueError("Generated glyph seed must be an integer")
        # Exercise the actual provider wire representation, including serialization.
        result["payload"] = json.dumps(glyph, ensure_ascii=False, allow_nan=False)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["worker_wall_s"] = _seconds(time.perf_counter_ns() - wall)
    result["worker_cpu_s"] = _seconds(cpu_clock() - cpu)
    result["worker_cpu_basis"] = cpu_basis
    return result


def _validate_work(index: int, payload: str, size: int = 128, cpu_basis: str = "process") -> dict[str, Any]:
    """Parse, structurally/style validate, and rasterize each generated SVG."""
    from PIL import Image

    cpu_clock = time.thread_time_ns if cpu_basis == "thread" else time.process_time_ns
    wall, cpu = time.perf_counter_ns(), cpu_clock()
    result: dict[str, Any] = {"index": index, "worker_pid": os.getpid(),
                                   "worker_thread": threading.get_ident()}
    try:
        glyph = json.loads(payload)
        report = _api().validate_svg(glyph["svg"])
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise ValueError(f"SVG validation rejected output: {report}")
        if not isinstance(report.get("errors"), list):
            raise ValueError("SVG validation must report an explicit errors list")
        if report.get("errors"):
            raise ValueError(f"SVG validation reported errors despite passed=True: {report}")
        rendered = _api().render_svg(glyph["svg"], size=size)
        if not isinstance(rendered, Image.Image) or rendered.size != (size, size):
            raise ValueError(f"SVG renderer must return a {size}x{size} PIL image")
        image = rendered.convert("RGBA")
        if image.getchannel("A").getextrema() != (255, 255):
            raise ValueError("SVG renderer violated the opaque white-background contract")
        # The generator contract uses black ink on white; verify a real nonblank tile.
        rgb = image.convert("RGB")
        ink = sum(max(pixel) < 128 for pixel in rgb.getdata())
        if not 0 < ink < size * size:
            raise ValueError("Rendered glyph is blank or fills the entire tile")
        pixels = image.tobytes()
        measurements = _measurement_api().measure_glyph(image)
        if not isinstance(measurements, dict) or measurements.get("blank") is not False:
            raise ValueError("Glyph measurement reported a blank or unavailable silhouette")
        result.update(
            status="passed", glyph_id=glyph["glyph_id"], family=glyph["family"],
            profile=glyph["profile"], seed=glyph["seed"], svg=glyph["svg"],
            svg_sha256=_sha256(glyph["svg"].encode("utf-8")),
            pixel_sha256=_sha256(pixels), ink_pixels=ink, size=size,
            validation=report, measurements=measurements, pixels=pixels,
        )
        result.update({key: glyph[key] for key in AUTHORING_FIELDS if key in glyph})
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["worker_wall_s"] = _seconds(time.perf_counter_ns() - wall)
    result["worker_cpu_s"] = _seconds(cpu_clock() - cpu)
    result["worker_cpu_basis"] = cpu_basis
    return result


class SVGGenerationProvider(Provider):
    """Local CPU provider; never calls an API or serves precomputed glyphs."""

    def __init__(self, pool, *, executor: str = "process") -> None:
        self.pool = pool
        self.executor = executor
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def budget_estimate_usd(self, model: str) -> float:
        return 0.0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        match = re.search(r"SVG_BENCH_REQUEST:(\{[^\r\n]+\})", prompt)
        if match is None:
            raise ValueError("Unexpected provider request; this benchmark has no planning/API lane")
        request = json.loads(match.group(1))
        index, attempt = request["index"], request["attempt"]
        started = time.perf_counter_ns()
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            record = await asyncio.get_running_loop().run_in_executor(
                self.pool, _generate_work, index, attempt, self.executor,
            )
        except Exception as exc:
            record = {"index": index, "attempt": attempt, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.in_flight -= 1
        record["provider_wall_s"] = _seconds(time.perf_counter_ns() - started)
        self.calls.append(record)
        if record["status"] != "passed":
            raise RuntimeError(record["error"])
        return CompletionResult(text=record["payload"], cost_usd=0.0,
                                prompt_tokens=0, completion_tokens=0)


def build_graph(glyph_count: int) -> ExecutionGraph:
    return ExecutionGraph(
        topology=[Topology.BROADCAST_REDUCE],
        nodes=[Node(
            id=f"svg-{index:03d}",
            label="SVG_BENCH_REQUEST:" + json.dumps({"index": index, "attempt": 0}),
            failure_policy=FailurePolicy.SKIP, max_retries=0,
            metadata={"estimated_cost_usd": 0.0},
        ) for index in range(glyph_count)],
    )


class MemorySampler:
    """Sample aggregate RSS of this process and its worker descendants."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak: int | None = None
        self.samples = 0
        self.error: str | None = None
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.error = "psutil unavailable; memory not measured"

    def _sample(self) -> None:
        if self.psutil is None:
            return
        try:
            processes = [self.process, *self.process.children(recursive=True)]
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    continue
            self.peak = max(self.peak or 0, total)
            self.samples += 1
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied) as exc:
            self.error = str(exc)

    def _run(self) -> None:
        while not self.stop.wait(.02):
            self._sample()

    def start(self) -> None:
        self._sample()
        if self.psutil is not None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def finish(self) -> dict[str, Any]:
        self._sample()
        self.stop.set()
        if self.thread:
            self.thread.join()
        return {"sampled_peak_process_tree_rss_bytes": self.peak,
                "samples": self.samples, "interval_s": .02, "error": self.error,
                "scope": "parent plus descendants; RSS counts shared pages in each process"}


def _write_json(path: Path, data: dict, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assemble(validated: list[dict], destination: Path, *, overwrite: bool, catalog_style: dict | None = None) -> dict:
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    good = sorted((r for r in validated if r["status"] == "passed"), key=lambda r: r["index"])
    entries = []
    columns, size = 16, 128
    atlas = Image.new("RGBA", (columns * size, max(1, math.ceil(len(good) / columns)) * size), "white")
    for position, record in enumerate(good):
        name = f"glyph-{record['index']:03d}.svg"
        with (destination / name).open("w" if overwrite else "x", encoding="utf-8", newline="\n") as f:
            f.write(record["svg"])
        atlas.paste(Image.frombytes("RGBA", (size, size), record["pixels"]),
                    ((position % columns) * size, (position // columns) * size))
        entries.append({key: record[key] for key in (
            "index", "glyph_id", "family", "profile", "seed", "svg_sha256", "pixel_sha256",
        ) + AUTHORING_FIELDS if key in record} | {"file": name})
    manifest = {"glyph_count": len(good), "glyphs": entries}
    _write_json(destination / "catalog.json", manifest, overwrite=overwrite)
    _write_json(destination / "validation.json", {
        "catalog_style": catalog_style,
        "glyphs": [{k: v for k, v in record.items() if k not in ("pixels", "svg")}
                   for record in good],
    }, overwrite=overwrite)
    atlas_path = destination / "atlas.png"
    if atlas_path.exists() and not overwrite:
        raise FileExistsError(atlas_path)
    atlas.save(atlas_path)
    return {"catalog": portable_path(destination / "catalog.json"),
            "catalog_sha256": _sha256((destination / "catalog.json").read_bytes()),
            "atlas": portable_path(atlas_path), "atlas_sha256": _sha256(atlas_path.read_bytes()),
            "validation": portable_path(destination / "validation.json"),
            "validation_sha256": _sha256((destination / "validation.json").read_bytes()),
            "output_bytes": sum(path.stat().st_size for path in destination.iterdir() if path.is_file())}


def _make_pool(executor: str, workers: int):
    if executor == "thread":
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def run_once(*, glyph_count: int, concurrency: int, executor: str, repeat: int,
             worker_cap: int, out: Path, overwrite: bool = False,
             reference_summary: dict | None = None, optical_review: dict | None = None) -> tuple[dict, list[str]]:
    """One complete graph-to-files workflow, including worker startup/shutdown."""
    workers = min(concurrency, worker_cap, os.cpu_count() or 1)
    memory = MemorySampler()
    started, parent_cpu = time.perf_counter_ns(), time.process_time_ns()
    memory.start()
    graph = build_graph(glyph_count)
    pool = _make_pool(executor, workers)
    provider = SVGGenerationProvider(pool, executor=executor)
    swarm = Swarm(provider=provider, model="original-svg-local-v1", parallel=True,
                  max_concurrency=concurrency, max_budget_usd=0.0, artifact_dir=None)
    setup_s = _seconds(time.perf_counter_ns() - started)
    errors = []
    result = None
    validated: list[dict] = []
    generation_started = time.perf_counter_ns()
    try:
        result = asyncio.run(swarm.execute_async(graph))
    except Exception as exc:
        errors.append(f"execution: {type(exc).__name__}: {exc}")
    generation_s = _seconds(time.perf_counter_ns() - generation_started)
    validation_started = time.perf_counter_ns()
    futures = []
    for index, node in enumerate(graph.nodes):
        if node.status is NodeStatus.COMPLETED:
            try:
                futures.append((index, pool.submit(_validate_work, index, str(node.result), 128, executor)))
            except Exception as exc:
                validated.append({"index": index, "status": "failed",
                                  "error": f"{type(exc).__name__}: {exc}"})
        else:
            errors.append(f"node {node.id}: {node.status.value}: {node.result}")
    for index, future in futures:
        try:
            validated.append(future.result())
        except Exception as exc:
            validated.append({"index": index, "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}"})
    good = [r for r in validated if r["status"] == "passed"]
    catalog_style = None
    if len(good) == glyph_count:
        try:
            from PIL import Image
            if reference_summary is None:
                reference_summary = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
            images = [Image.frombytes("RGBA", (128, 128), r["pixels"]) for r in good]
            distinctness = _measurement_api().find_near_matches(images, [r["glyph_id"] for r in good])
            for duplicate_kind in ("exact_duplicates", "normalized_duplicates", "reflected_duplicates", "aligned_exact_pairs"):
                if distinctness.get(duplicate_kind):
                    errors.append(f"duplicate silhouettes: {duplicate_kind}")
            bound_review = optical_review
            if optical_review is not None:
                hashes = {r["glyph_id"]: r["svg_sha256"] for r in good}
                if optical_review.get("glyph_svg_sha256") != hashes:
                    bound_review = {**optical_review, "passed": False,
                                    "binding_error": "Optical review hashes do not match this generated catalog"}
            catalog_style = _measurement_api().evaluate_catalog(
                good, reference_summary, distinctness, optical_review=bound_review,
            )
        except Exception as exc:
            errors.append(f"catalog validation: {type(exc).__name__}: {exc}")
    validation_s = _seconds(time.perf_counter_ns() - validation_started)
    errors.extend(f"validation {r['index']}: {r['error']}" for r in validated if r["status"] != "passed")
    if len(good) != glyph_count:
        errors.append(f"expected {glyph_count} valid glyphs; found {len(good)}")
    for field in ("glyph_id", "svg_sha256", "pixel_sha256"):
        if len({r[field] for r in good}) != len(good):
            errors.append(f"duplicate {field}")
    if len(provider.calls) != glyph_count:
        errors.append(f"expected {glyph_count} provider calls; found {len(provider.calls)}")
    assembly_started = time.perf_counter_ns()
    assembly = None
    if not errors:
        try:
            assembly = _assemble(good, out, overwrite=overwrite, catalog_style=catalog_style)
        except Exception as exc:
            errors.append(f"assembly: {type(exc).__name__}: {exc}")
    assembly_s = _seconds(time.perf_counter_ns() - assembly_started)
    shutdown_started = time.perf_counter_ns()
    pool.shutdown(wait=True, cancel_futures=True)
    shutdown_s = _seconds(time.perf_counter_ns() - shutdown_started)
    resources = memory.finish()
    end_to_end_s = _seconds(time.perf_counter_ns() - started)
    parent_cpu_s = _seconds(time.process_time_ns() - parent_cpu)
    calls = [{k: v for k, v in call.items() if k != "payload"} for call in provider.calls]
    receipts = [{k: v for k, v in record.items() if k not in ("pixels", "svg", "measurements")}
                | ({"measurement_sha256": _sha256(json.dumps(record["measurements"], sort_keys=True,
                                                             allow_nan=False).encode())}
                   if "measurements" in record else {})
                for record in validated]
    # process_time is process-wide. Summing thread call CPU would double-count
    # overlapping work; use only the single parent-process CPU measurement there.
    worker_cpu = None if executor == "thread" else {
        "generation_s": sum(call.get("worker_cpu_s", 0) for call in calls),
        "validation_s": sum(record.get("worker_cpu_s", 0) for record in receipts),
        "scope": "worker task CPU only; startup/shutdown CPU is excluded",
    }
    if executor == "thread":
        for record in [*calls, *receipts]:
            record.pop("worker_cpu_s", None)
    run = {
        "executor": executor, "concurrency": concurrency, "workers": workers, "repeat": repeat,
        "status": "passed" if not errors else "failed", "errors": errors,
        "execution_id": result.execution_id if result else None,
        "setup_wall_s": setup_s, "generation_wall_s": generation_s,
        "validation_wall_s": validation_s, "assembly_wall_s": assembly_s,
        "worker_shutdown_wall_s": shutdown_s, "end_to_end_wall_s": end_to_end_s,
        "throughput_valid_glyphs_per_s": len(good) / end_to_end_s,
        "parent_process_cpu_s": parent_cpu_s, "worker_cpu": worker_cpu, "memory": resources,
        "provider_calls": len(calls), "successful_provider_calls": sum(c['status'] == 'passed' for c in calls),
        "failed_provider_calls": sum(c['status'] != 'passed' for c in calls),
        "max_in_flight_provider_calls": provider.max_in_flight,
        "workers_observed": len({(r.get("worker_pid"), r.get("worker_thread"))
                                  for r in [*calls, *receipts] if r.get("worker_pid")}),
        "completed_nodes": sum(n.status is NodeStatus.COMPLETED for n in graph.nodes),
        "valid_glyphs": len(good), "api_calls": 0, "api_cost_usd": 0.0,
        "smythe_recorded_cost_usd": result.total_cost_usd if result else None,
        "cost_scope": "provider API charges only; local hardware and energy are unpriced",
        "calls": calls, "glyphs": receipts, "assembly": assembly,
        "style_acceptance": {
            "accepted": bool(catalog_style and catalog_style.get("accepted") is True),
            "passes_numeric_shape_gates": bool(catalog_style and catalog_style.get("passes_numeric_shape_gates") is True),
            "failed_gates": catalog_style.get("failed_gates", []) if catalog_style else [],
            "optical_review_required": catalog_style.get("optical_review_required", True) if catalog_style else True,
            "status": catalog_style.get("status") if catalog_style else "unavailable",
            "report_sha256": _sha256(json.dumps(catalog_style, sort_keys=True, allow_nan=False).encode()),
        },
    }
    return run, [r["svg"] for r in sorted(good, key=lambda r: r['index'])]


def measure_rasterization(svgs: Sequence[str], *, sizes: Sequence[int], repeats: int) -> dict:
    """Separate static SVG-to-pixels timing, never a screensaver FPS claim."""
    samples = []
    for size in sizes:
        for repeat in range(1, repeats + 1):
            wall, cpu = time.perf_counter_ns(), time.process_time_ns()
            digest = hashlib.sha256()
            for svg in svgs:
                image = _api().render_svg(svg, size=size)
                if image.size != (size, size):
                    raise ValueError(f"Rasterization returned {image.size}, expected {(size, size)}")
                digest.update(image.convert("RGBA").tobytes())
            samples.append({"size": size, "repeat": repeat, "glyphs": len(svgs),
                            "wall_s": _seconds(time.perf_counter_ns() - wall),
                            "cpu_s": _seconds(time.process_time_ns() - cpu),
                            "pixel_batch_sha256": digest.hexdigest()})
    return {"scope": "sequential static SVG rasterization plus pixel hashing; not animation FPS",
            "generation_reused": "accepted SVG strings are inputs to this separate rendering test",
            "samples": samples}


def summarize(runs: list[dict], *, repeats: int) -> tuple[list[dict], dict | None, list[str]]:
    groups = {}
    defects = []
    expected_hashes = None
    for run in runs:
        key = (run["executor"], run["concurrency"])
        groups.setdefault(key, []).append(run)
        hashes = [(r.get("svg_sha256"), r.get("pixel_sha256"), r.get("measurement_sha256"))
                  for r in sorted(run["glyphs"], key=lambda r: r["index"])]
        if expected_hashes is None:
            expected_hashes = hashes
        elif hashes != expected_hashes:
            defects.append("Output hashes differ across repeated/concurrency runs")
    summaries = []
    for (executor, concurrency), samples in groups.items():
        passed = len(samples) == repeats and all(r["status"] == "passed" for r in samples)
        summary = {"executor": executor, "concurrency": concurrency, "workers": samples[0]["workers"],
                   "status": "passed" if passed else "failed", "repeats": len(samples)}
        for field in ("generation_wall_s", "validation_wall_s", "assembly_wall_s", "end_to_end_wall_s"):
            values = [r[field] for r in samples]
            summary[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        summaries.append(summary)
    for summary in summaries:
        baseline = next((s for s in summaries if s["executor"] == summary["executor"]
                         and s["concurrency"] == 1 and s["status"] == "passed"), None)
        summary["end_to_end_speedup_vs_same_executor_c1"] = (
            baseline["end_to_end_wall_s"]["median"] / summary["end_to_end_wall_s"]["median"]
            if baseline and summary["status"] == "passed" else None
        )
    fastest = min((s for s in summaries if s["status"] == "passed"),
                  key=lambda s: s["end_to_end_wall_s"]["median"], default=None)
    return summaries, fastest, sorted(set(defects))


def run_benchmark(*, out: Path = DEFAULT_OUT, results: Path | None = None,
                  glyph_count: int = 192, concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
                  executors: Sequence[str] = DEFAULT_EXECUTORS, repeats: int = 3,
                  worker_cap: int = 8, render_sizes: Sequence[int] = (64, 128, 512),
                  overwrite: bool = False, optical_review_path: Path | None = None) -> dict:
    if isinstance(glyph_count, bool) or not isinstance(glyph_count, int) or not 1 <= glyph_count <= 192:
        raise ValueError("glyph_count must be an integer between 1 and 192")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(worker_cap, bool) or not isinstance(worker_cap, int) or worker_cap < 1:
        raise ValueError("worker_cap must be a positive integer")
    if not concurrencies or 1 not in concurrencies or len(set(concurrencies)) != len(concurrencies):
        raise ValueError("concurrencies must be unique and include the c1 baseline")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in concurrencies):
        raise ValueError("concurrencies must be positive integers")
    if not executors or len(set(executors)) != len(executors) or any(e not in DEFAULT_EXECUTORS for e in executors):
        raise ValueError("executors must be unique choices from thread,process")
    if any(isinstance(s, bool) or not isinstance(s, int) or not 8 <= s <= 2048 for s in render_sizes):
        raise ValueError("render_sizes must contain integers between 8 and 2048")
    out = Path(out)
    results = Path(results) if results else (DEFAULT_RESULTS if out == DEFAULT_OUT else out / "results.json")
    if not overwrite and ((out.exists() and any(out.iterdir())) or results.exists()):
        raise FileExistsError("Output already exists; choose a new partition or pass --overwrite")
    _api()  # Import driver libraries once; never generate or warm-cache a glyph here.
    _measurement_api()
    reference_summary = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    optical_review = (json.loads(Path(optical_review_path).read_text(encoding="utf-8"))
                      if optical_review_path is not None else None)
    env = environment_snapshot("smythe", "pillow", "numpy", "scipy", "shapely", "psutil")
    source_files = [Path(__file__), Path(_api().__file__), Path(_measurement_api().__file__), REFERENCE_PATH]
    if optical_review_path is not None:
        source_files.append(Path(optical_review_path))
    source_hashes = {portable_path(path): _sha256(path.read_bytes()) for path in source_files}
    runs, rendered_svgs = [], []
    configurations = [(e, c) for e in executors for c in concurrencies]
    order = []
    for repeat in range(1, repeats + 1):
        # Rotate and alternate direction so c1 is not always the cold first run.
        ordered = configurations if repeat % 2 else list(reversed(configurations))
        offset = (repeat - 1) % len(configurations)
        shifted = ordered[offset:] + ordered[:offset]
        for executor, concurrency in shifted:
            order.append({"repeat": repeat, "executor": executor, "concurrency": concurrency})
            print(f"SVG benchmark {executor} c={concurrency} repeat={repeat}/{repeats}", flush=True)
            run, svgs = run_once(glyph_count=glyph_count, concurrency=concurrency, executor=executor,
                                 repeat=repeat, worker_cap=worker_cap,
                                 out=out / "runs" / f"r{repeat:02d}-{executor}-c{concurrency}",
                                 overwrite=overwrite, reference_summary=reference_summary,
                                 optical_review=optical_review)
            runs.append(run)
            if run["status"] == "passed" and not rendered_svgs:
                rendered_svgs = svgs
    summaries, fastest, defects = summarize(runs, repeats=repeats)
    passed = all(r["status"] == "passed" for r in runs) and not defects
    rendering = None
    if passed:
        try:
            rendering = measure_rasterization(rendered_svgs, sizes=render_sizes, repeats=repeats)
            for size in render_sizes:
                hashes = {r["pixel_batch_sha256"] for r in rendering["samples"] if r["size"] == size}
                if len(hashes) != 1:
                    defects.append(f"Static rasterization at {size}px differs across repeats")
        except Exception as exc:
            defects.append(f"Static rasterization failed: {type(exc).__name__}: {exc}")
    for path in source_files:
        try:
            unchanged = _sha256(path.read_bytes()) == source_hashes[portable_path(path)]
        except OSError:
            unchanged = False
        if not unchanged:
            defects.append(f"Benchmark input/source changed during campaign: {portable_path(path)}")
    passed = passed and not defects
    style_accepted = all(r["style_acceptance"]["accepted"] for r in runs)
    claimable = passed and repeats >= 3 and bool(rendered_svgs)
    readme_eligible = claimable and glyph_count == 192 and style_accepted
    payload = {
        "benchmark": "original-svg-glyph-workflow", "record_version": 1,
        "status": "passed" if passed else "failed", "protocol_version": PROTOCOL_VERSION,
        "evidence_status": "claimable" if claimable else "diagnostic", "claimable": claimable,
        "known_measurement_defects": defects, "catalog_style_accepted": style_accepted,
        "timing_claimable": claimable, "readme_promotion_eligible": readme_eligible,
        "claim_scope": "measured local procedural SVG workflow on this host; no model-quality, API-speed, or animation-FPS comparison",
        "protocol": {"glyph_count": glyph_count, "concurrencies": list(concurrencies),
                     "executors": list(executors), "repeats": repeats, "worker_cap": worker_cap,
                     "effective_worker_rule": "min(requested concurrency, worker cap, logical CPU count)",
                     "attempt_per_index": 0, "simulated_latency_s": 0, "cached_generation_outputs": False,
                     "graph": "one independent Smythe node/provider call per glyph; no planning model",
                     "failure_policy": "skip failed node, attempt remaining independent nodes; any failure disqualifies campaign",
                     "end_to_end_scope": "graph/pool setup, generation, validation, assembly, worker shutdown, resource sampling",
                     "excluded_from_workflow_timer": "CLI/library import, campaign metadata, final aggregate record, separate rasterization study",
                     "validation": "validate_svg, actual128px raster, full measured profile gates, all-pairs aligned/reflected distinctness, unique SVG/pixel/identity hashes",
                     "optical_review": portable_path(optical_review_path) if optical_review_path else None,
                     "order": order},
        "environment": env, "source_sha256": source_hashes,
        "logical_cpu_count": os.cpu_count(), "runs": runs, "summaries": summaries,
        "fastest_median_workflow": fastest, "rendering": rendering,
        "api_calls": 0, "api_cost_usd": 0.0,
        "errors": [error for run in runs for error in run["errors"]],
    }
    _write_json(results, payload, overwrite=overwrite)
    return payload


def _integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--glyphs", type=int, default=192)
    parser.add_argument("--concurrencies", type=_integers, default=DEFAULT_CONCURRENCIES)
    parser.add_argument("--executors", default=",".join(DEFAULT_EXECUTORS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-cap", type=int, default=8)
    parser.add_argument("--render-sizes", type=_integers, default=(64, 128, 512))
    parser.add_argument("--optical-review", type=Path, help="passed optical review bound to glyph_svg_sha256 map")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace this partition's outputs")
    args = parser.parse_args()
    try:
        payload = run_benchmark(out=args.out, results=args.results, glyph_count=args.glyphs,
                                concurrencies=args.concurrencies, executors=tuple(args.executors.split(",")),
                                repeats=args.repeats, worker_cap=args.worker_cap,
                                render_sizes=args.render_sizes, overwrite=args.overwrite,
                                optical_review_path=args.optical_review)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": payload["status"], "evidence_status": payload["evidence_status"],
                      "fastest_median_workflow": payload["fastest_median_workflow"]}, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
