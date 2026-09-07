"""Render public benchmark charts from committed result records.

The SVGs are deterministic functions of the JSON evidence.  Public graph
assets use a strict two-colour system: black, white, outlines, and patterns.

    python benchmarks/render_readme_charts.py
"""

from __future__ import annotations

import html
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

if __package__:
    from .glyph_screensaver_assets import GlyphSpec, get_glyph_specs
else:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from glyph_screensaver_assets import GlyphSpec, get_glyph_specs

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "assets" / "benchmarks"
GLYPH_OUT = ROOT / "assets" / "glyph_rain"

BLACK = "#000000"
WHITE = "#ffffff"
SERIF = "Georgia,'Times New Roman',serif"
SANS = "'Avenir Next',Avenir,'Helvetica Neue',Helvetica,sans-serif"
MONO = "'SFMono-Regular',Consolas,'Liberation Mono',monospace"
TRAJAN = "'Trajan Pro 3','Trajan Pro',Trajan,Cinzel,Georgia,serif"


def _svg(width: int, height: int, body: str, *, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">\n'
        f'<rect width="{width}" height="{height}" fill="{WHITE}"/>\n'
        f"{body}</svg>\n"
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 14,
    fill: str = BLACK,
    anchor: str = "start",
    weight: str = "normal",
    family: str = SANS,
    tracking: float | None = None,
) -> str:
    spacing = "" if tracking is None else f' letter-spacing="{tracking}"'
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
        f"{spacing}>{html.escape(content)}</text>\n"
    )


def _load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as handle:
        return json.load(handle)


def _patterns() -> str:
    return f"""<defs>
  <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse">
    <rect width="8" height="8" fill="{WHITE}"/>
    <path d="M-2 2L2-2M0 8L8 0M6 10L10 6" fill="none" stroke="{BLACK}" stroke-width="2"/>
  </pattern>
</defs>
"""


def _legend(x: float, y: float, label: str, style: str) -> str:
    if style == "solid":
        swatch = f'<rect x="{x}" y="{y - 12}" width="22" height="12" fill="{BLACK}"/>\n'
    elif style == "hatch":
        swatch = (
            f'<rect x="{x}" y="{y - 12}" width="22" height="12" '
            f'fill="url(#hatch)" stroke="{BLACK}"/>\n'
        )
    else:
        swatch = (
            f'<rect x="{x}" y="{y - 12}" width="22" height="12" '
            f'fill="{WHITE}" stroke="{BLACK}" stroke-width="2"/>\n'
        )
    return swatch + _text(x + 31, y, label, size=12, weight="600")


def _bar(x: float, y: float, width: float, style: str) -> str:
    if style == "solid":
        return f'<rect x="{x}" y="{y}" width="{width:.1f}" height="16" fill="{BLACK}"/>\n'
    fill = "url(#hatch)" if style == "hatch" else WHITE
    return (
        f'<rect x="{x}" y="{y}" width="{width:.1f}" height="16" '
        f'fill="{fill}" stroke="{BLACK}" stroke-width="2"/>\n'
    )


def _metric_panel(
    *,
    x: int,
    title: str,
    direction: str,
    values: tuple[tuple[str, float, str, str], ...],
    maximum: float,
) -> str:
    body = _text(x, 194, title, size=17, weight="700", family=SERIF)
    body += _text(x, 217, direction.upper(), size=10.5, weight="700", tracking=1.4)
    bar_width = 210
    for index, (label, value, rendered, style) in enumerate(values):
        label_y = 258 + index * 72
        body += _text(x, label_y, label, size=13, weight="600")
        body += _text(x + 260, label_y, rendered, size=13, anchor="end", family=MONO)
        body += _bar(x, label_y + 12, max(4, bar_width * value / maximum), style)
    return body


def _framework_rows() -> dict[str, dict]:
    record = _load("framework_h2h_rightsized.json")
    rows = {row["system"]: row for row in record["summary"]}
    required = ("smythe_fixed", "langgraph_fixed", "crewai_fixed")
    missing = [name for name in required if name not in rows]
    if missing:
        raise ValueError(f"framework comparison is missing: {', '.join(missing)}")
    if any(rows[name]["errors"] for name in required):
        raise ValueError("framework comparison contains failed runs")
    cells = []
    for name in required:
        cell = [run for run in record["records"] if run["system"] == name]
        if not cell or any(run.get("error") for run in cell):
            raise ValueError("framework comparison contains missing or failed runs")
        identities = {(run["task"], run["rep"]) for run in cell}
        if len(identities) != len(cell) or len(cell) != rows[name]["runs_ok"]:
            raise ValueError("framework comparison contains duplicate or missing runs")
        for metric, source, precision in (
            ("quality_mean", "quality", 2),
            ("tokens_mean", "tokens", 0),
            ("wall_s_mean", "wall_s", 2),
        ):
            values = [run.get(source) for run in cell]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (source == "quality" and (type(value) is not int or value > 10))
                for value in values
            ):
                raise ValueError(f"framework comparison has invalid {source} values")
            if round(sum(values) / len(values), precision) != rows[name][metric]:
                raise ValueError(f"framework comparison summary disagrees with raw {source}")
        cells.append(identities)
    if any(cell != cells[0] for cell in cells[1:]):
        raise ValueError("framework comparison has unmatched task/repetition cells")
    return rows


def render_framework_comparison() -> str:
    """Render a real three-framework comparison from the corrected record."""
    rows = _framework_rows()
    smythe = rows["smythe_fixed"]
    langgraph = rows["langgraph_fixed"]
    crewai = rows["crewai_fixed"]

    body = _patterns()
    body += _text(40, 55, "Framework comparison", size=31, weight="700", family=SERIF)
    body += _text(
        40,
        84,
        "Same five tasks and fixed three-stage pipeline; same executor model; blind cross-vendor judge",
        size=14,
    )
    body += f'<line x1="40" y1="112" x2="920" y2="112" stroke="{BLACK}" stroke-width="2"/>\n'
    body += _legend(40, 143, "Smythe", "solid")
    body += _legend(188, 143, "LangGraph", "outline")
    body += _legend(366, 143, "CrewAI", "hatch")

    series = (
        ("Smythe", "solid", smythe),
        ("LangGraph", "outline", langgraph),
        ("CrewAI", "hatch", crewai),
    )
    quality = tuple(
        (label, row["quality_mean"], f'{row["quality_mean"]:.2f}', style)
        for label, style, row in series
    )
    tokens = tuple(
        (label, row["tokens_mean"], f'{row["tokens_mean"]:,}', style)
        for label, style, row in series
    )
    wall = tuple(
        (label, row["wall_s_mean"], f'{row["wall_s_mean"]:.2f}s', style)
        for label, style, row in series
    )
    body += _metric_panel(
        x=40,
        title="Blind quality / 10",
        direction="Higher is better",
        values=quality,
        maximum=10,
    )
    body += f'<line x1="330" y1="178" x2="330" y2="452" stroke="{BLACK}"/>\n'
    body += _metric_panel(
        x=350,
        title="Mean tokens",
        direction="Lower is better",
        values=tokens,
        maximum=max(value for _, value, _, _ in tokens),
    )
    body += f'<line x1="640" y1="178" x2="640" y2="452" stroke="{BLACK}"/>\n'
    body += _metric_panel(
        x=660,
        title="Mean wall time",
        direction="Lower is better",
        values=wall,
        maximum=max(value for _, value, _, _ in wall),
    )
    body += f'<line x1="40" y1="476" x2="920" y2="476" stroke="{BLACK}"/>\n'
    body += _text(
        40,
        505,
        "15 runs per framework; zero errors; token usage measures the fixed pipeline",
        size=11.5,
    )
    body += _text(
        920,
        505,
        "framework_h2h_rightsized.json",
        size=11.5,
        anchor="end",
        family=MONO,
    )
    return _svg(
        960,
        530,
        body,
        label=(
            "Framework benchmark comparing Smythe, LangGraph, and CrewAI. "
            "Smythe has the highest observed blind score, fewest tokens, and lowest mean wall time."
        ),
    )


def render_framework_callouts() -> str:
    """Render restrained headline callouts from the corrected framework run."""
    rows = _framework_rows()
    smythe = rows["smythe_fixed"]
    crewai = rows["crewai_fixed"]
    token_reduction = 1 - smythe["tokens_mean"] / crewai["tokens_mean"]
    wall_reduction = 1 - smythe["wall_s_mean"] / crewai["wall_s_mean"]

    body = _text(40, 36, "MEASURED ADVANTAGES", size=11, weight="700", tracking=2.2)
    body += f'<line x1="40" y1="54" x2="920" y2="54" stroke="{BLACK}" stroke-width="2"/>\n'
    body += f'<line x1="480" y1="78" x2="480" y2="205" stroke="{BLACK}"/>\n'
    body += _text(40, 142, f"{token_reduction:.0%}", size=68, family=TRAJAN, weight="700")
    body += _text(
        42,
        174,
        "LOWER MEAN TOKEN LOAD THAN CREWAI",
        size=12,
        weight="700",
        tracking=1.2,
    )
    body += _text(
        42, 197, f'{smythe["tokens_mean"]:,} vs {crewai["tokens_mean"]:,} mean tokens', size=12
    )
    body += _text(520, 142, f"{wall_reduction:.0%}", size=68, family=TRAJAN, weight="700")
    body += _text(
        522, 174, "LOWER MEAN WALL TIME THAN CREWAI", size=12, weight="700", tracking=1.2
    )
    body += _text(
        522, 197, f'{smythe["wall_s_mean"]:.2f}s vs {crewai["wall_s_mean"]:.2f}s', size=12
    )
    body += f'<line x1="40" y1="220" x2="920" y2="220" stroke="{BLACK}"/>\n'
    body += _text(40, 242, "Same fixed pipeline; 15 runs per framework", size=10.5)
    body += _text(920, 242, "framework_h2h_rightsized.json", size=10.5, anchor="end", family=MONO)
    return _svg(
        960,
        258,
        body,
        label=(
            f"Smythe callouts: {token_reduction:.0%} lower mean token load than CrewAI "
            f"and {wall_reduction:.0%} lower mean wall time than CrewAI."
        ),
    )


def render_shape_efficiency() -> str:
    """Render the claimable task-shape comparison in strict black and white."""
    record = _load("shape_suite_v3.json")
    rows = {row["baseline"]: row for row in record["summary"]}
    expected = ("fixed_pipeline", "smythe_dynamic")
    if any(baseline not in rows for baseline in expected):
        raise ValueError("shape efficiency record is missing a required baseline")

    wall_means = {}
    matched_cells = []
    for baseline in expected:
        cell = [row for row in record["records"] if row["baseline"] == baseline]
        if any(row.get("error") for row in cell):
            raise ValueError("shape efficiency contains failed runs")
        identities = {(row["task"], row["rep"]) for row in cell}
        if len(identities) != len(cell) or len(cell) != rows[baseline]["runs"]:
            raise ValueError("shape efficiency contains duplicate or missing runs")
        matched_cells.append(identities)
        if not cell:
            raise ValueError(f"shape efficiency has no {baseline} runs")
        for run in cell:
            quality = run.get("quality")
            wall = run.get("wall_s")
            nodes = run.get("nodes")
            if (
                isinstance(quality, bool)
                or not isinstance(quality, (int, float))
                or not math.isfinite(quality)
                or not 1 <= quality <= 10
            ):
                raise ValueError("shape efficiency has invalid or missing quality scores")
            if (
                isinstance(wall, bool)
                or not isinstance(wall, (int, float))
                or not math.isfinite(wall)
                or wall <= 0
            ):
                raise ValueError("shape efficiency has invalid wall timing")
            if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
                raise ValueError("shape efficiency has invalid graph node counts")
        for source, metric in (("quality", "quality_mean"), ("nodes", "nodes_mean")):
            calculated = round(sum(run[source] for run in cell) / len(cell), 2)
            if calculated != rows[baseline][metric]:
                raise ValueError(f"shape efficiency summary disagrees with raw {source}")
        values = [row["wall_s"] for row in cell]
        wall_means[baseline] = sum(values) / len(values)
    if matched_cells[0] != matched_cells[1]:
        raise ValueError("shape efficiency has unmatched task/repetition cells")

    dynamic = rows["smythe_dynamic"]
    fixed = rows["fixed_pipeline"]
    series = (
        ("Smythe dynamic", "solid", dynamic),
        ("Fixed pipeline", "hatch", fixed),
    )
    quality = tuple(
        (label, row["quality_mean"], f'{row["quality_mean"]:.2f}', style)
        for label, style, row in series
    )
    nodes = tuple(
        (label, row["nodes_mean"], f'{row["nodes_mean"]:.2f}', style)
        for label, style, row in series
    )
    wall = tuple(
        (label, wall_means[row_name], f"{wall_means[row_name]:.1f}s", style)
        for (label, style, _), row_name in zip(
            series, ("smythe_dynamic", "fixed_pipeline"), strict=True
        )
    )

    body = _patterns()
    body += _text(40, 55, "Task-shaped execution", size=31, weight="700", family=SERIF)
    body += _text(
        40,
        84,
        "Five task shapes; three reps each; blind cross-vendor judge",
        size=14,
    )
    body += f'<line x1="40" y1="112" x2="920" y2="112" stroke="{BLACK}" stroke-width="2"/>\n'
    body += _legend(40, 143, "Smythe dynamic", "solid")
    body += _legend(232, 143, "Fixed pipeline", "hatch")
    body += _metric_panel(
        x=40, title="Blind quality / 10", direction="Higher is better", values=quality, maximum=10
    )
    body += f'<line x1="330" y1="178" x2="330" y2="382" stroke="{BLACK}"/>\n'
    body += _metric_panel(
        x=350,
        title="Mean graph nodes",
        direction="Task-shaped allocation",
        values=nodes,
        maximum=max(v for _, v, _, _ in nodes),
    )
    body += f'<line x1="640" y1="178" x2="640" y2="382" stroke="{BLACK}"/>\n'
    body += _metric_panel(
        x=660,
        title="Mean wall time",
        direction="Lower is better",
        values=wall,
        maximum=max(v for _, v, _, _ in wall),
    )

    wall_reduction = 1 - wall_means["smythe_dynamic"] / wall_means["fixed_pipeline"]
    body += f'<line x1="40" y1="406" x2="920" y2="406" stroke="{BLACK}"/>\n'
    body += _text(
        40,
        437,
        f"{wall_reduction:.0%} lower end-to-end wall time   /   15 runs per arm   /   same executor model",
        size=14,
        weight="700",
        family=SERIF,
    )
    body += _text(40, 468, "Planning included in wall time; quality is in the same measured band", size=11.5)
    body += _text(920, 468, "shape_suite_v3.json", size=11.5, anchor="end", family=MONO)
    return _svg(
        960,
        492,
        body,
        label=(
            "Task-shape benchmark. Smythe dynamic matches the fixed pipeline quality band "
            "with lower end-to-end wall time, including planning."
        ),
    )


def _glyph_runs(name: str) -> tuple[dict, list[dict]]:
    record = _load(name)
    runs = record.get("runs", [])
    expected = record.get("protocol", {}).get("graph_nodes")
    if (
        record.get("status") != "passed"
        or record.get("mode") != "offline"
        or not runs
        or expected not in (64, 128, 192, 256)
    ):
        raise ValueError(f"{name} is not a passing 64-, 128-, 192-, or 256-node glyph record")
    concurrencies = [run["concurrency"] for run in runs]
    if (
        concurrencies != record["protocol"]["concurrencies"]
        or len(set(concurrencies)) != len(concurrencies)
        or 1 not in concurrencies
    ):
        raise ValueError(f"{name} has an invalid concurrency sweep")
    baseline = next(run for run in runs if run["concurrency"] == 1)
    for run in runs:
        validation = run.get("validation", {})
        if (
            run.get("status") != "passed"
            or not validation.get("passed")
            or run.get("completed_nodes") != expected
            or validation.get("valid_png_tiles") != expected
            or validation.get("unique_tile_hashes") != expected
        ):
            raise ValueError(f"{name} contains a non-claimable glyph run")
        wall = run.get("generation_wall_s", 0)
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError(f"{name} has invalid generation timing")
        if not math.isclose(
            baseline["generation_wall_s"] / wall,
            run["speedup_vs_concurrency_1"],
            abs_tol=0.0001,
        ):
            raise ValueError(f"{name} speedup disagrees with measured timing")
    return record, runs


def _glyph_marker(kind: str, x: float, y: float) -> str:
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{BLACK}"/>\n'
    if kind == "square":
        return (
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
            f'fill="{WHITE}" stroke="{BLACK}" stroke-width="2"/>\n'
        )
    if kind == "diamond":
        return (
            f'<path d="M{x:.1f} {y - 6:.1f}L{x + 6:.1f} {y:.1f}'
            f'L{x:.1f} {y + 6:.1f}L{x - 6:.1f} {y:.1f}Z" fill="{BLACK}"/>\n'
        )
    if kind == "triangle":
        return (
            f'<path d="M{x:.1f} {y - 6:.1f}L{x + 6:.1f} {y + 5:.1f}'
            f'L{x - 6:.1f} {y + 5:.1f}Z" fill="{WHITE}" stroke="{BLACK}" '
            'stroke-width="2"/>\n'
        )
    raise ValueError(f"unknown glyph chart marker: {kind}")


def render_glyph_scaling() -> str:
    """Render the isolated 64-, 128-, 192-, and 256-node sweeps."""
    record_64, runs_64 = _glyph_runs("glyph_screensaver_64_offline_realistic.json")
    record_128, runs_128 = _glyph_runs("glyph_screensaver_128_offline_realistic.json")
    record_192, runs_192 = _glyph_runs("glyph_screensaver_offline_realistic.json")
    record_256, runs_256 = _glyph_runs("glyph_screensaver_256_offline_realistic.json")
    series = (
        (64, record_64, runs_64, "2 6", "triangle", 2.0),
        (128, record_128, runs_128, "12 5 2 5", "diamond", 2.0),
        (192, record_192, runs_192, None, "circle", 4.0),
        (256, record_256, runs_256, "9 7", "square", 2.0),
    )
    concurrencies = [run["concurrency"] for run in runs_64]
    if any(concurrencies != [run["concurrency"] for run in runs] for _, _, runs, *_ in series):
        raise ValueError("glyph scaling records use different concurrency sweeps")
    if any(record["protocol"]["offline_latency_s"] != 5.8 for _, record, *_ in series):
        raise ValueError("glyph scaling records do not use the matched 5.8-second latency")

    body = _text(40, 49, "ARTIFACT FAN-OUT", size=11, weight="700", tracking=2.2)
    body += _text(
        40,
        86,
        "Scaling across 64, 128, 192 and 256 nodes",
        size=27,
        weight="700",
        family=SERIF,
    )
    body += _text(
        40,
        111,
        "Matched 5.8-second simulated provider latency; every tile valid and unique",
        size=13,
    )
    body += f'<line x1="40" y1="132" x2="920" y2="132" stroke="{BLACK}" stroke-width="2"/>\n'

    legend_positions = (70, 282, 494, 706)
    for x, (nodes, _, _, dash, marker, width) in zip(legend_positions, series, strict=True):
        dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
        body += (
            f'<line x1="{x}" y1="158" x2="{x + 42}" y2="158" stroke="{BLACK}" '
            f'stroke-width="{width:g}"{dash_attr}/>\n'
        )
        body += _glyph_marker(marker, x + 21, 158)
        body += _text(x + 54, 162, f"{nodes} nodes", size=11.5, weight="700")

    plot_left, plot_top, plot_right, plot_bottom = 92, 196, 900, 378
    max_throughput = 10.0
    for tick in (0, 2.5, 5.0, 7.5, 10.0):
        y = plot_bottom - (tick / max_throughput) * (plot_bottom - plot_top)
        body += _text(73, y + 4, f"{tick:g}", size=10, anchor="end", family=MONO)
        body += (
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="{BLACK}" stroke-dasharray="2 7"/>\n'
        )
    body += _text(40, 288, "GLYPHS / S", size=9.5, weight="700", tracking=1.2)

    x_step = (plot_right - plot_left) / (len(concurrencies) - 1)
    x_positions = [plot_left + index * x_step for index in range(len(concurrencies))]
    for x, concurrency in zip(x_positions, concurrencies, strict=True):
        body += f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 6}" stroke="{BLACK}"/>\n'
        body += _text(x, plot_bottom + 24, str(concurrency), size=10.5, anchor="middle", family=MONO)
    body += _text(
        (plot_left + plot_right) / 2,
        plot_bottom + 48,
        "MAX CONCURRENCY",
        size=9.5,
        anchor="middle",
        weight="700",
        tracking=1.2,
    )

    def points(runs: list[dict]) -> list[tuple[float, float]]:
        return [
            (
                x,
                plot_bottom
                - (run["throughput_glyphs_per_s"] / max_throughput)
                * (plot_bottom - plot_top),
            )
            for x, run in zip(x_positions, runs, strict=True)
        ]

    for _, _, runs, dash, marker, width in series:
        series_points = points(runs)
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in series_points)
        dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
        body += (
            f'<polyline points="{path}" fill="none" stroke="{BLACK}" '
            f'stroke-width="{width:g}"{dash_attr}/>\n'
        )
        for x, y in series_points:
            body += _glyph_marker(marker, x, y)

    body += f'<line x1="40" y1="448" x2="920" y2="448" stroke="{BLACK}"/>\n'
    callout_x = (40, 260, 480, 700)
    for index, (x, (nodes, _, runs, *_)) in enumerate(zip(callout_x, series, strict=True)):
        speedup = runs[-1]["speedup_vs_concurrency_1"]
        body += _text(x, 499, f"{speedup:.2f}×", size=34, family=TRAJAN, weight="700")
        body += _text(
            x + 2,
            522,
            f"{nodes} NODES / CONCURRENCY 64",
            size=8.5,
            weight="700",
            tracking=0.8,
        )
        if index < len(callout_x) - 1:
            body += f'<line x1="{x + 202}" y1="464" x2="{x + 202}" y2="528" stroke="{BLACK}"/>\n'
    body += _text(
        40,
        557,
        "64 / 128 / 192 / 256 valid unique tiles at every concurrency",
        size=10.5,
    )
    body += _text(
        920,
        557,
        f'{record_192["protocol"]["offline_latency_s"]:.1f}s matched latency / $0 API cost',
        size=10.5,
        anchor="end",
        family=MONO,
    )
    return _svg(
        960,
        574,
        body,
        label=(
            "Throughput chart for isolated 64-, 128-, 192-, and 256-node Glyph Rain "
            "sweeps. Every run produces valid unique tiles at each measured concurrency."
        ),
    )


def render_glyph_pipeline() -> str:
    """Render the Glyph Rain example as a generated graph and execution envelope."""
    body = _text(40, 49, "FROM BRIEF TO SCREENSAVER", size=11, weight="700", tracking=2.2)
    body += _text(40, 86, "One example of Smythe at high fan-out", size=29, weight="700", family=SERIF)
    body += f'<line x1="40" y1="108" x2="920" y2="108" stroke="{BLACK}" stroke-width="2"/>\n'

    stages = (
        (40, 146, 176, 108, "01", "BRIEF", "Define the artifact"),
        (274, 146, 176, 108, "192", "GENERATE", "One node per glyph"),
        (508, 146, 176, 108, "192", "VERIFY", "Size + PNG + SHA-256"),
        (742, 146, 178, 108, "04", "ASSEMBLE", "Atlas + web + ports"),
    )
    for index, (x, y, width, height, number, title, detail) in enumerate(stages):
        fill = BLACK if index == len(stages) - 1 else WHITE
        ink = WHITE if fill == BLACK else BLACK
        body += (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="{fill}" stroke="{BLACK}" stroke-width="2"/>\n'
        )
        body += _text(x + 16, y + 41, number, size=31, fill=ink, weight="700", family=TRAJAN)
        body += _text(x + 16, y + 70, title, size=11, fill=ink, weight="700", tracking=1.5)
        body += _text(x + 16, y + 91, detail, size=11, fill=ink)
        if index < len(stages) - 1:
            arrow_x = x + width
            body += f'<line x1="{arrow_x}" y1="200" x2="{arrow_x + 58}" y2="200" stroke="{BLACK}" stroke-width="2"/>\n'
            body += f'<path d="M{arrow_x + 50} 194L{arrow_x + 58} 200L{arrow_x + 50} 206" fill="none" stroke="{BLACK}" stroke-width="2"/>\n'

    body += f'<line x1="40" y1="282" x2="920" y2="282" stroke="{BLACK}"/>\n'
    body += _text(
        40,
        306,
        "Generated execution topology",
        size=11,
        weight="700",
        tracking=1.1,
    )
    body += _text(
        920,
        306,
        "Bounded concurrency / objective gates / traces / per-node recovery",
        size=11,
        anchor="end",
    )
    return _svg(
        960,
        326,
        body,
        label=(
            "Glyph Rain example pipeline. A brief becomes a 192-node generated graph, "
            "each glyph is verified, and the outputs are assembled into four artifacts."
        ),
    )


def _glyph_strokes(spec: GlyphSpec, *, x: float, y: float, scale: float) -> str:
    body = f'<g transform="translate({x:.1f} {y:.1f}) scale({scale:.3f})">\n'
    for stroke in spec.strokes:
        kind, *values = stroke
        if kind == "l":
            x1, y1, x2, y2, width = values
            body += (
                f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
                f'stroke="{BLACK}" stroke-width="{width:g}" stroke-linecap="round"/>\n'
            )
        elif kind == "q":
            x1, y1, cx, cy, x2, y2, width = values
            body += (
                f'<path d="M{x1:g} {y1:g}Q{cx:g} {cy:g} {x2:g} {y2:g}" '
                f'fill="none" stroke="{BLACK}" stroke-width="{width:g}" '
                'stroke-linecap="round" stroke-linejoin="round"/>\n'
            )
        else:
            cx, cy, radius = values
            body += f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" fill="{BLACK}"/>\n'
    return body + "</g>\n"


def render_glyph_specimens() -> str:
    """Render selected committed glyph stroke programs as a line-art specimen table."""
    selected = (0, 3, 5, 12, 14, 21, 32, 44, 63, 80, 107, 151)
    catalog = get_glyph_specs()
    specs = tuple(catalog[index] for index in selected)
    body = _text(40, 49, "SELECTED GLYPHS", size=11, weight="700", tracking=2.2)
    body += _text(40, 86, "Twelve marks from the generated catalog", size=29, weight="700", family=SERIF)
    body += _text(920, 84, "12 / 192", size=18, anchor="end", family=TRAJAN, weight="700")
    body += f'<line x1="40" y1="108" x2="920" y2="108" stroke="{BLACK}" stroke-width="2"/>\n'

    columns = 6
    cell_width = 880 / columns
    cell_height = 151
    grid_y = 126
    for index, spec in enumerate(specs):
        row, column = divmod(index, columns)
        x = 40 + column * cell_width
        y = grid_y + row * cell_height
        body += (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width:.1f}" '
            f'height="{cell_height:.1f}" fill="{WHITE}" stroke="{BLACK}"/>\n'
        )
        scale = 0.68
        mark_x = x + (cell_width - 100 * scale) / 2
        mark_y = y + 7
        body += _glyph_strokes(spec, x=mark_x, y=mark_y, scale=scale)
        body += _text(
            x + cell_width / 2,
            y + 139,
            spec.id.upper(),
            size=9.5,
            anchor="middle",
            family=MONO,
            tracking=0.8,
        )

    body += _text(
        40,
        452,
        "Original deterministic stroke programs; no font or source-image extraction",
        size=11,
    )
    body += _text(920, 452, "GLYPH_SPECS", size=11, anchor="end", family=MONO)
    return _svg(
        960,
        470,
        body,
        label=(
            "Line-art table of twelve representative glyphs selected from the "
            "192-character deterministic Glyph Rain catalog."
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GLYPH_OUT.mkdir(parents=True, exist_ok=True)
    charts: dict[Path, Callable[[], str]] = {
        OUT / "framework_comparison.svg": render_framework_comparison,
        OUT / "framework_callouts.svg": render_framework_callouts,
        OUT / "shape_efficiency.svg": render_shape_efficiency,
        OUT / "glyph_scaling.svg": render_glyph_scaling,
        GLYPH_OUT / "glyph_pipeline.svg": render_glyph_pipeline,
        GLYPH_OUT / "glyph_specimens.svg": render_glyph_specimens,
    }
    for destination, renderer in charts.items():
        destination.write_text(renderer(), encoding="utf-8", newline="\n")
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
