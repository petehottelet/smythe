"""Monochrome workflow and memory charts from the complete v2 campaign."""

import hashlib
import json
import math
import statistics

from benchmarks.svg_v2_evidence import review_record

RECORD = "glyph_svg_v2_20260913.json"


def checked_record(path):
    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)
    p = record["protocol"]
    if (record["claimable"] is not True or record["status"] != "passed"
            or record["known_measurement_defects"] or set(p["sizes"]) != {192, 256}
            or p["repeats"] != 3 or set(p["executors"]) != {"thread", "process"}
            or set(p["concurrencies"]) != {1, 4, 8}):
        raise ValueError("A complete claimable 192/256 campaign is required")
    return record, review_record(record), hashlib.sha256(raw.encode()).hexdigest()


def render_workflow(path):
    from benchmarks.render_readme_charts import BLACK, SERIF, TRAJAN, WHITE, _patterns, _svg, _text

    record, rows, digest = checked_record(path)
    maximum = math.ceil(max(row["max_s"] for row in rows) / 5) * 5
    body = _patterns()
    body += _text(40, 36, "AUTHORED SVG WORKFLOW / V2", size=12, weight="700", tracking=1.6)
    body += _text(40, 76, "More output, same workflow.", size=30, family=SERIF, weight="700")
    body += _text(40, 106, "Three repetitions per cell · compilation, four-size validation, pair comparisons, file export", size=13)
    for panel, count in enumerate((192, 256)):
        x = 40 + panel * 540
        selected = [r for r in rows if r["glyph_count"] == count]
        best = min(selected, key=lambda r: r["median_s"])
        body += _text(x, 163, f"{count}", size=42, family=TRAJAN, weight="700")
        body += _text(x + 112, 148, f"{best['median_s']:.2f}s best median", size=18, weight="700")
        body += _text(x + 112, 173, f"{best['executor'].title()} c{best['concurrency']} · {best['speedup_vs_same_backend_c1']:.2f}× its c1 baseline", size=12)
        bar_x, bar_w = x + 118, 270
        for position, row in enumerate(selected):
            y = 210 + position * 45
            body += _text(x, y + 14, f"{row['executor'].title()} / c{row['concurrency']}", size=12)
            width = row["median_s"] / maximum * bar_w
            body += f'<rect x="{bar_x}" y="{y}" width="{width:.4f}" height="20" fill="{BLACK if count == 256 else WHITE}" stroke="{BLACK}"/>\n'
            lo, hi = (bar_x + row[key] / maximum * bar_w for key in ("min_s", "max_s"))
            body += f'<path d="M{lo:.4f} {y + 25}H{hi:.4f}M{lo:.4f} {y + 21}V{y + 29}M{hi:.4f} {y + 21}V{y + 29}" fill="none" stroke="{BLACK}"/>\n'
            body += _text(x + 490, y + 15, f"{row['median_s']:.2f}s", size=13, anchor="end", weight="700")
            for run in record["runs"]:
                if (run["glyph_count"], run["executor"], run["concurrency"]) == (count, row["executor"], row["concurrency"]):
                    px = bar_x + run["end_to_end_wall_s"] / maximum * bar_w
                    body += f'<circle cx="{px:.4f}" cy="{y + 25}" r="2.5" fill="{BLACK}" data-seconds="{run["end_to_end_wall_s"]}"/>\n'
        axis_y = 492
        body += f'<path d="M{bar_x} {axis_y}h{bar_w}" stroke="{BLACK}"/>\n'
        for tick in range(0, maximum + 1, 5):
            tx = bar_x + tick / maximum * bar_w
            body += _text(tx, axis_y + 19, str(tick), size=10, anchor="middle")
        body += _text(bar_x, axis_y + 41, "Complete wall time (seconds) · lower is better", size=11)
    best = min((r for r in rows if r["glyph_count"] == 256), key=lambda r: r["median_s"])
    run = next(r for r in record["runs"] if (r["glyph_count"], r["executor"], r["concurrency"], r["repeat"])
               == (256, best["executor"], best["concurrency"], best["median_run_repeat"]))
    phases = {"Compile": run["generation_wall_s"], "Validate": run["validation_wall_s"], "Export": run["assembly_wall_s"]}
    phases["Setup / shutdown / overhead"] = run["end_to_end_wall_s"] - math.fsum(phases.values())
    body += _text(40, 589, f"Inside one actual 256-glyph run: {run['executor']} c{run['concurrency']}, repetition {run['repeat']}", size=17, family=SERIF, weight="700")
    x = 40
    for i, (name, seconds) in enumerate(phases.items()):
        width = 1020 * seconds / run["end_to_end_wall_s"]
        fill = (BLACK, "url(#hatch)", WHITE, WHITE)[i]
        dash = ' stroke-dasharray="3 2"' if i == 3 else ""
        body += f'<rect x="{x:.5f}" y="610" width="{width:.5f}" height="25" fill="{fill}" stroke="{BLACK}"{dash} data-phase="{name}" data-seconds="{seconds}"/>\n'
        body += _text(40 + i * 255, 661, f"{name}: {seconds:.2f}s", size=12)
        x += width
    body += _text(40, 702, "36 workflows · zero API calls · hardware, energy and design labor unpriced", size=12)
    body += _text(40, 726, "Bars: medians. Points and whiskers: every repetition and min–max. Both panels share the same scale.", size=12)
    body += _text(40, 750, "Authored contours are fixed before sampling. This measures neither creative design nor screensaver FPS.", size=12)
    body += _text(40, 780, f"Source: {RECORD} · SHA-256 {digest[:16]}", size=10)
    return _svg(1100, 800, body, label="192 and 256 authored SVG glyph workflows: all 36 measurements and an actual median-run phase breakdown")


def render_memory(path):
    from benchmarks.render_readme_charts import BLACK, SERIF, WHITE, _svg, _text

    record, rows, digest = checked_record(path)
    memory = {}
    for row in rows:
        key = row["glyph_count"], row["executor"], row["concurrency"]
        runs = [r for r in record["runs"] if (r["glyph_count"], r["executor"], r["concurrency"]) == key]
        values = []
        for run in runs:
            sample = run["memory"]
            peak = sample["sampled_peak_process_tree_rss_bytes"]
            if type(peak) is not int or peak <= 0 or sample["samples"] <= 0 or sample["error"] is not None:
                raise ValueError("Complete sampled memory evidence is required")
            values.append(peak / 1024**2)
        memory[key] = values
    maximum = math.ceil(max(max(v) for v in memory.values()) / 100) * 100
    body = _text(40, 39, "PARALLELISM / MEMORY", size=12, weight="700", tracking=1.6)
    body += _text(40, 79, "The memory cost of more workers", size=30, family=SERIF, weight="700")
    body += _text(40, 109, "Median sampled peak RSS · parent plus worker processes · three repetitions per cell", size=13)
    for panel, count in enumerate((192, 256)):
        x = 40 + panel * 540
        body += _text(x, 157, f"{count} glyphs", size=24, family=SERIF, weight="700")
        for i, row in enumerate(r for r in rows if r["glyph_count"] == count):
            y = 185 + i * 45
            values = memory[count, row["executor"], row["concurrency"]]
            value = statistics.median(values)
            body += _text(x, y + 14, f"{row['executor'].title()} / c{row['concurrency']}", size=12)
            bx, bw = x + 118, 270
            body += f'<rect x="{bx}" y="{y}" width="{value / maximum * bw:.4f}" height="20" fill="{BLACK if count == 256 else WHITE}" stroke="{BLACK}"/>\n'
            for v in values:
                body += f'<circle cx="{bx + v / maximum * bw:.4f}" cy="{y + 26}" r="2.5" fill="{BLACK}"/>\n'
            body += _text(x + 490, y + 15, f"{value:.0f} MiB", size=13, weight="700", anchor="end")
        for tick in range(0, maximum + 1, 200 if maximum >= 800 else 100):
            body += _text(x + 118 + tick / maximum * 270, 484, str(tick), size=10, anchor="middle")
        body += _text(x + 118, 507, "Sampled peak RSS (MiB) · lower is better", size=11)
    body += _text(40, 554, "20 ms sampling can miss short peaks; RSS counts shared pages in each process. This is not GPU memory.", size=12)
    body += _text(40, 580, "More workers trade memory for shorter completion time. Both panels share the same scale.", size=12)
    body += _text(40, 614, f"Source: {RECORD} · SHA-256 {digest[:16]}", size=10)
    return _svg(1100, 635, body, label="Measured memory tradeoff for 192 and 256 SVG glyph workflows across thread and process worker counts")
