"""Render the README's benchmark charts from committed result records.

Deterministic: the SVGs are pure functions of the JSON evidence records, so
re-running this script after a benchmark rerun regenerates the charts and the
diff shows exactly what changed.

    python benchmarks/render_readme_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "assets" / "benchmarks"

PANEL = "#23221e"
BORDER = "#3a382f"
TITLE = "#ece4cf"
MUTED = "#a49d8a"
GRID = "#37352c"
GREEN = "#57e07f"
GREEN_DIM = "#2f9e55"
GOLD = "#c9a53f"
GRAY = "#8a8478"
GRAY_DIM = "#6e6960"
VALUE = "#e8e0cc"

FONT = "-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="{width}" height="{height}" rx="10" fill="{PANEL}" '
        f'stroke="{BORDER}"/>\n{body}</svg>\n'
    )


def _text(x, y, content, *, size=13, fill=VALUE, anchor="start", weight="normal",
          mono=False, opacity=1.0) -> str:
    family = MONO if mono else FONT
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
        f'opacity="{opacity}">{content}</text>\n'
    )


def _load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as handle:
        return json.load(handle)


def render_glyph_fanout() -> str | None:
    record = _load("glyph_screensaver_offline_realistic.json")
    runs = [run for run in record["runs"] if run["status"] == "passed"]
    if not runs:
        print("glyph fan-out: no passing runs; skipping")
        return None
    nodes = record["protocol"]["graph_nodes"]
    latency = record["protocol"]["offline_latency_s"]
    width, height = 640, 96 + 34 * len(runs) + 40
    top = 88
    label_x, bar_x = 24, 150
    bar_max = width - bar_x - 130
    max_speedup = max(run["speedup_vs_concurrency_1"] for run in runs)
    body = _text(24, 34, f"{nodes} parallel image tasks, one Smythe graph",
                 size=19, fill=TITLE, weight="600")
    body += _text(24, 56,
                  f"measured wall-clock speedup vs serial · simulated {latency}s "
                  "image-API latency · zero API cost", size=12.5, fill=MUTED)
    for index, run in enumerate(runs):
        y = top + index * 34
        speedup = run["speedup_vs_concurrency_1"]
        wall = run["generation_wall_s"]
        length = max(4, bar_max * speedup / max_speedup)
        color = GRAY if run["concurrency"] == 1 else GREEN
        opacity = 0.55 + 0.45 * (index / max(1, len(runs) - 1))
        body += _text(label_x, y + 15, f'c = {run["concurrency"]}', size=13,
                      fill=MUTED, mono=True)
        body += (
            f'<rect x="{bar_x}" y="{y}" width="{length:.1f}" height="21" rx="4" '
            f'fill="{color}" opacity="{opacity:.2f}"/>\n'
        )
        wall_label = f"{wall:,.0f} s" if wall >= 100 else f"{wall:.1f} s"
        body += _text(bar_x + length + 10, y + 15,
                      f"{speedup:.1f}× · {wall_label}", size=13,
                      fill=VALUE, mono=True, weight="600")
    body += _text(24, height - 18,
                  "benchmarks/run_glyph_screensaver.py · all "
                  f"{nodes} tiles valid + unique (SHA-256) at every concurrency",
                  size=11.5, fill=MUTED)
    return _svg(width, height, body)


def render_framework_h2h() -> str | None:
    record = _load("framework_h2h.json")
    rows = {row["system"]: row for row in record["summary"]}
    systems = [
        ("smythe_fixed", "Smythe", GREEN),
        ("langgraph_fixed", "LangGraph", GRAY),
        ("crewai_fixed", "CrewAI", GRAY_DIM),
    ]
    if any(system not in rows for system, _, _ in systems):
        print("framework h2h: expected systems missing; skipping")
        return None
    width, height = 640, 300
    body = _text(24, 34, "Same pipeline, same model — three frameworks",
                 size=19, fill=TITLE, weight="600")
    body += _text(24, 56,
                  f'research → analyze → write · executor {record["executor_model"]} '
                  "· blind cross-vendor judge · 15 runs each",
                  size=12.5, fill=MUTED)

    panels = (
        ("Tokens per deliverable", "tokens_mean", "{:,.0f}", 24),
        ("Wall seconds", "wall_s_mean", "{:.1f} s", 336),
    )
    for title, key, fmt, panel_x in panels:
        body += _text(panel_x, 92, title, size=13, fill=GOLD, weight="600")
        max_value = max(rows[system][key] for system, _, _ in systems)
        bar_max = 280
        for index, (system, label, color) in enumerate(systems):
            y = 108 + index * 42
            value = rows[system][key]
            length = max(4, bar_max * value / max_value)
            body += _text(panel_x, y + 14, label, size=12.5, fill=MUTED)
            body += (
                f'<rect x="{panel_x}" y="{y + 20}" width="{length:.1f}" '
                f'height="12" rx="3" fill="{color}"/>\n'
            )
            body += _text(panel_x + length + 8 if length < bar_max - 70
                          else panel_x + length - 8, y + 31,
                          fmt.format(value), size=12,
                          fill=VALUE if length < bar_max - 70 else PANEL,
                          mono=True, weight="600",
                          anchor="start" if length < bar_max - 70 else "end")
    quality = " · ".join(
        f'{label} {rows[system]["quality_mean"]:.1f}'
        for system, label, _ in systems
    )
    body += _text(24, height - 40,
                  f"blind quality (1–10): {quality} — ceiling-compressed; "
                  "this table discriminates efficiency, not quality",
                  size=11.5, fill=MUTED)
    body += _text(24, height - 20,
                  "benchmarks/run_framework_h2h.py · records in "
                  "benchmarks/results/framework_h2h.json",
                  size=11.5, fill=MUTED)
    return _svg(width, height, body)


def render_durability() -> str | None:
    record = _load("durability_kill_resume_v2.json")
    rows = record.get("cell_b_durability", [])
    frameworks: dict[str, list[dict]] = {}
    for row in rows:
        frameworks.setdefault(row["framework"], []).append(row)
    if "langgraph" not in frameworks or "smythe" not in frameworks:
        print("durability: langgraph rows not present yet; skipping chart")
        return None
    cell = record["protocol"]["cell_b"]
    n = cell["n"]
    price = 0.04

    def mean(values):
        return sum(values) / len(values)

    stats = {}
    for name, reps in frameworks.items():
        dup = [rep["duplicate_dispatches"] for rep in reps]
        stats[name] = {
            "dup_mean": mean(dup),
            "dup_min": min(dup),
            "dup_max": max(dup),
            "resume_mean": mean([rep["resume_wall_s"] for rep in reps]),
            "reps": len(reps),
        }
    width, height = 640, 264
    body = _text(24, 34, "Hard-kill mid-run: what do you re-pay on resume?",
                 size=19, fill=TITLE, weight="600")
    body += _text(24, 56,
                  f'{n}-node fan-out, process killed at {cell["kill_at_dispatches"]} '
                  f'dispatches · strongest persistence both sides · '
                  f'{stats["smythe"]["reps"]} reps',
                  size=12.5, fill=MUTED)
    bar_max = 300
    max_dup = max(stat["dup_mean"] for stat in stats.values())
    entries = (
        ("smythe", "Smythe — per-node checkpoints", GREEN),
        ("langgraph", "LangGraph — superstep checkpoints", GRAY),
    )
    for index, (name, label, color) in enumerate(entries):
        stat = stats[name]
        y = 92 + index * 62
        length = max(5, bar_max * stat["dup_mean"] / max_dup)
        body += _text(24, y + 12, label, size=13, fill=MUTED)
        body += (
            f'<rect x="24" y="{y + 20}" width="{length:.1f}" height="16" rx="4" '
            f'fill="{color}"/>\n'
        )
        body += _text(24 + length + 10, y + 33,
                      f'{stat["dup_mean"]:.1f} duplicated calls '
                      f'[{stat["dup_min"]}–{stat["dup_max"]}] · resume '
                      f'{stat["resume_mean"]:.1f}s',
                      size=12.5, fill=VALUE, mono=True, weight="600")
    smythe_cost = stats["smythe"]["dup_mean"] / n * 1000 * price
    langgraph_cost = stats["langgraph"]["dup_mean"] / n * 1000 * price
    body += _text(24, height - 42,
                  f"at $0.04/image on a 1,000-image job, that crash re-bills "
                  f"~${langgraph_cost:,.0f} (LangGraph) vs ~${smythe_cost:.2f} "
                  "(Smythe)", size=12.5, fill=GOLD, weight="600")
    body += _text(24, height - 20,
                  "benchmarks/run_durability_benchmark.py · duplicate dispatches "
                  "are exposure, not invoice proof · protocol: durability_benchmark.md",
                  size=11.5, fill=MUTED)
    return _svg(width, height, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts = {
        "glyph_fanout_speedup.svg": render_glyph_fanout,
        "framework_h2h.svg": render_framework_h2h,
        "durability_crash_cost.svg": render_durability,
    }
    for name, renderer in charts.items():
        try:
            svg = renderer()
        except FileNotFoundError as exc:
            print(f"{name}: missing input ({exc}); skipping")
            continue
        if svg is None:
            continue
        destination = OUT / name
        destination.write_text(svg, encoding="utf-8", newline="\n")
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
