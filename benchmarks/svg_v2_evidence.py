"""Standard-library validation of complete v2 workflow records."""

import math
import statistics

PROTOCOL = "authored-svg-workflow-v2"
PHASES = ("setup_wall_s", "generation_wall_s", "validation_wall_s", "assembly_wall_s",
          "worker_shutdown_wall_s")


def review_record(record):
    """Recompute every comparison from complete samples, never trusted summaries."""
    protocol = record["protocol"]
    if record["protocol_version"] != PROTOCOL or protocol["protocol_version"] != PROTOCOL:
        raise ValueError("Unexpected protocol version")
    if protocol["simulated_latency_s"] != 0 or protocol["cached_svg_outputs"] is not False or protocol["design_work_timed"] is not False:
        raise ValueError("Protocol scope changed")
    observed_order = [{k: run[k] for k in ("glyph_count", "executor", "concurrency", "repeat")}
                      for run in record["runs"]]
    if observed_order != protocol["schedule"]:
        raise ValueError("Frozen run order changed")
    expected = {(n, e, c, r) for n in protocol["sizes"] for e in protocol["executors"]
                for c in protocol["concurrencies"] for r in range(1, protocol["repeats"] + 1)}
    seen, hashes, groups = set(), {}, {}
    for run in record["runs"]:
        n, e, c, r = (run[k] for k in ("glyph_count", "executor", "concurrency", "repeat"))
        key = (n, e, c, r)
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or unexpected trial")
        seen.add(key)
        if run["status"] != "passed" or run["errors"] or run.get("source_stable") is not True:
            raise ValueError("Failed or source-drifted trial")
        for field in (*PHASES, "end_to_end_wall_s", "parent_process_cpu_s"):
            value = run[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid clock: {field}")
        if run["end_to_end_wall_s"] <= 0 or math.fsum(run[p] for p in PHASES) > run["end_to_end_wall_s"] + 1e-7:
            raise ValueError("Phase clocks exceed the complete workflow")
        for field in ("provider_calls", "completed_nodes", "valid_glyphs"):
            if type(run[field]) is not int or run[field] != n:
                raise ValueError(f"Incomplete count: {field}")
        if len(run["calls"]) != n or any(x["status"] != "passed" for x in run["calls"]):
            raise ValueError("Incomplete provider receipts")
        if sorted(x["index"] for x in run["calls"]) != list(range(n)):
            raise ValueError("Provider identity mismatch")
        if any(run[k] != 0 for k in ("api_calls", "api_cost_usd", "smythe_recorded_cost_usd")):
            raise ValueError("Unexpected paid usage")
        if run["workers"] != min(c, protocol["worker_cap"], record["logical_cpu_count"]):
            raise ValueError("Worker declaration mismatch")
        if not 1 <= run["max_in_flight_provider_calls"] <= c:
            raise ValueError("Concurrency bound exceeded")
        glyphs = run["glyphs"]
        if [g["index"] for g in glyphs] != list(range(n)) or any(g["status"] != "passed" for g in glyphs):
            raise ValueError("Incomplete glyph receipts")
        current = [(g["svg_sha256"], g["pixel_sha256"], g["measurement_sha256"]) for g in glyphs]
        if any(len({row[i] for row in current}) != n for i in (0, 1)):
            raise ValueError("Duplicate glyph hashes")
        if n in hashes and hashes[n] != current:
            raise ValueError("Cross-run glyph hashes differ")
        hashes[n] = current
        distinct = run["distinctness"]
        if distinct["compared_pairs"] != n * (n - 1) // 2 or distinct["threshold"] != .85:
            raise ValueError("Incomplete distinctness gate")
        if any(distinct[k] for k in ("near_matches", "exact_duplicates", "normalized_duplicates", "reflected_duplicates", "aligned_exact_pairs")):
            raise ValueError("Duplicate or near-matching silhouettes")
        if not run["assembly"] or run["assembly"]["output_bytes"] <= 0:
            raise ValueError("Missing assembled artifact")
        groups.setdefault((n, e, c), []).append(run)
    if seen != expected:
        raise ValueError("Incomplete campaign schedule")
    if 192 in hashes and 256 in hashes and hashes[192] != hashes[256][:192]:
        raise ValueError("The first 192 glyphs changed between sizes")
    summaries = []
    for (n, e, c), runs in sorted(groups.items()):
        values = [run["end_to_end_wall_s"] for run in runs]
        median = statistics.median(values)
        baseline = statistics.median(run["end_to_end_wall_s"] for run in groups[n, e, 1])
        selected = min(runs, key=lambda run: (abs(run["end_to_end_wall_s"] - median), run["repeat"]))
        summaries.append({"glyph_count": n, "executor": e, "concurrency": c,
                          "workers": runs[0]["workers"], "repeats": len(runs),
                          "median_s": median, "min_s": min(values), "max_s": max(values),
                          "speedup_vs_same_backend_c1": baseline / median,
                          "glyphs_per_s": n / median, "median_run_repeat": selected["repeat"]})
    return summaries

