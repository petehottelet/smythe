"""Render public benchmark charts from committed result records.

The SVGs are deterministic functions of the JSON evidence.  Public graph
assets use a strict two-colour system: black, white, outlines, and patterns.

    python benchmarks/render_readme_charts.py
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "assets" / "benchmarks"

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
        "15 runs per framework; zero errors; framework-native execution and accounting",
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
            "Smythe records the highest blind quality, fewest tokens, and lowest mean wall time."
        ),
    )


def render_framework_callouts() -> str:
    """Render restrained headline callouts from the corrected framework run."""
    rows = _framework_rows()
    smythe = rows["smythe_fixed"]
    langgraph = rows["langgraph_fixed"]
    crewai = rows["crewai_fixed"]
    token_reduction = 1 - smythe["tokens_mean"] / crewai["tokens_mean"]
    wall_reduction = 1 - smythe["wall_s_mean"] / langgraph["wall_s_mean"]

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
        522, 174, "LOWER MEAN WALL TIME THAN LANGGRAPH", size=12, weight="700", tracking=1.2
    )
    body += _text(
        522, 197, f'{smythe["wall_s_mean"]:.2f}s vs {langgraph["wall_s_mean"]:.2f}s', size=12
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
            f"and {wall_reduction:.0%} lower mean wall time than LangGraph."
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
    for baseline in expected:
        values = [
            row["wall_s"]
            for row in record["records"]
            if row["baseline"] == baseline and row.get("error") is None
        ]
        if not values:
            raise ValueError(f"shape efficiency has no passing {baseline} runs")
        wall_means[baseline] = sum(values) / len(values)

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
    cost = tuple(
        (label, row["cost_total_usd"], f'${row["cost_total_usd"]:.3f}', style)
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
        title="Total cost / 15 runs",
        direction="Lower is better",
        values=cost,
        maximum=max(v for _, v, _, _ in cost),
    )
    body += f'<line x1="640" y1="178" x2="640" y2="382" stroke="{BLACK}"/>\n'
    body += _metric_panel(
        x=660,
        title="Mean wall time",
        direction="Lower is better",
        values=wall,
        maximum=max(v for _, v, _, _ in wall),
    )

    cost_reduction = 1 - dynamic["cost_total_usd"] / fixed["cost_total_usd"]
    wall_reduction = 1 - wall_means["smythe_dynamic"] / wall_means["fixed_pipeline"]
    efficiency_gain = 1 - dynamic["usd_per_quality_point"] / fixed["usd_per_quality_point"]
    body += f'<line x1="40" y1="406" x2="920" y2="406" stroke="{BLACK}"/>\n'
    body += _text(
        40,
        437,
        f"{cost_reduction:.0%} lower cost   /   {wall_reduction:.0%} lower wall time   /   {efficiency_gain:.0%} lower cost per quality point",
        size=14,
        weight="700",
        family=SERIF,
    )
    body += _text(40, 468, "Quality remains in the same measured band", size=11.5)
    body += _text(920, 468, "shape_suite_v3.json", size=11.5, anchor="end", family=MONO)
    return _svg(
        960,
        492,
        body,
        label=(
            "Task-shape benchmark. Smythe dynamic matches the fixed pipeline quality band "
            "with lower measured cost and wall time."
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Callable[[], str]] = {
        "framework_comparison.svg": render_framework_comparison,
        "framework_callouts.svg": render_framework_callouts,
        "shape_efficiency.svg": render_shape_efficiency,
    }
    for name, renderer in charts.items():
        destination = OUT / name
        destination.write_text(renderer(), encoding="utf-8", newline="\n")
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
