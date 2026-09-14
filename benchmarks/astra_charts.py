"""Monochrome Astra/Sol distributions from a reviewed, complete study record."""

from collections import Counter
import hashlib
import json
import math
from statistics import median

from benchmarks.astra_analysis import percentile

ARMS = ("astra-fixed", "astra-dynamic", "sol-fixed", "sol-dynamic")
LABELS = ("Astra / fixed", "Astra / generated", "Sol / fixed", "Sol / generated")


def checked_record(path):
    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)
    review = json.loads(path.with_name("review.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    if (review["analysis_sha256"] != digest or review["status"] != "audited-main"
            or review["known_measurement_defects"]):
        raise ValueError("A bound evidence review with no known measurement defect is required")
    rows = record["all_trials"]
    cells = Counter((r["trial"]["arm_id"], r["trial"]["task_id"], r["trial"]["repetition"]) for r in rows)
    tasks = {task for _, task, _ in cells}
    expected = {(arm, task, rep) for arm in ARMS for task in tasks for rep in range(1, 6)}
    if len(tasks) != 10 or len(rows) != 200 or set(cells) != expected or set(cells.values()) != {1}:
        raise ValueError("All 200 balanced observations must appear in the chart")
    for row in rows:
        for key in ("cost_lower_usd", "cost_upper_usd", "wall_seconds"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Complete finite cost and timing observations are required")
        lo, hi = row["cost_lower_usd"], row["cost_upper_usd"]
        cost = row["cost_usd"]
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost)):
            raise ValueError("A known cost must be a finite number")
        if lo > hi or (row["cost_usd"] is None and (row["billing_status"] != "unknown-reserved" or lo == hi)):
            raise ValueError("Unknown costs require a positive retained reservation interval")
        if row["cost_usd"] is not None and (row["cost_usd"] != lo or lo != hi):
            raise ValueError("A known cost must equal both evidence bounds")
    unknown = sum(r["cost_usd"] is None for r in rows)
    if unknown != record["billing"]["unknown_cost_workflows"] or unknown != review.get("unknown_workflow_usage", 0):
        raise ValueError("Unknown cost counts differ from the evidence review")
    return record, review, digest


def render_distributions(path):
    from benchmarks.render_readme_charts import BLACK, SERIF, WHITE, _svg, _text

    record, review, digest = checked_record(path)
    body = _text(40, 38, "ASTRA / SOL · MATCHED WORKFLOWS", size=12, weight="700", tracking=1.5)
    body += _text(40, 80, "All workflows, with cost and timing evidence.", size=30, family=SERIF, weight="700")
    body += _text(40, 111, "200 runs · 10 synthetic tasks × 5 repetitions × 4 arms · planning included", size=13)
    for panel, (metric, label, precision) in enumerate((("wall_seconds", "Wall time (seconds)", 1),
                                                        ("cost_usd", "Generation cost (USD)", 4))):
        x = 40 + panel * 550
        bx, width = x + 145, 270
        rows = record["all_trials"]
        lower_key, upper_key = ("cost_lower_usd", "cost_upper_usd") if metric == "cost_usd" else (metric, metric)
        maximum = max(r[upper_key] for r in rows) * 1.06 or 1
        body += _text(x, 163, label, size=20, family=SERIF, weight="700")
        body += _text(x, 188, "Lower is better · shared scale across both models", size=11)
        for i, arm in enumerate(ARMS):
            arm_rows = [r for r in rows if r["trial"]["arm_id"] == arm]
            lower = sorted(r[lower_key] for r in arm_rows)
            upper = sorted(r[upper_key] for r in arm_rows)
            y = 247 + i * 83
            body += _text(x, y + 4, LABELS[i], size=12, weight="600")
            lo, hi = percentile(lower, .25), percentile(upper, .75)
            lx, hx = bx + lo / maximum * width, bx + hi / maximum * width
            body += f'<rect x="{lx:.4f}" y="{y - 13}" width="{hx - lx:.4f}" height="26" fill="{WHITE}" stroke="{BLACK}"/>\n'
            medians = median(lower), median(upper)
            for middle in sorted(set(medians)):
                mx = bx + middle / maximum * width
                body += f'<path d="M{mx:.4f} {y - 17}v34" stroke="{BLACK}" stroke-width="2"/>\n'
            for n, row in enumerate(sorted(arm_rows, key=lambda r: (r[lower_key], r["trial"]["trial_id"]))):
                py = y + ((n * 7) % 13 - 6) * 3
                if row[metric] is None:
                    left, right = (bx + row[k] / maximum * width for k in (lower_key, upper_key))
                    body += f'<path d="M{left:.4f} {py}H{right:.4f}M{left:.4f} {py - 4}v8M{right:.4f} {py - 4}v8" stroke="{BLACK}" stroke-dasharray="3 2" data-arm="{arm}" data-unknown="true" data-lower="{row[lower_key]}" data-upper="{row[upper_key]}"/>\n'
                    continue
                value = row[metric]
                px = bx + value / maximum * width
                fill = BLACK if arm.endswith("dynamic") else WHITE
                body += f'<circle cx="{px:.4f}" cy="{py}" r="2.3" fill="{fill}" stroke="{BLACK}" data-arm="{arm}" data-value="{value}"/>\n'
            middle_text = f"{medians[0]:.{precision}f}"
            if medians[0] != medians[1]:
                middle_text = f"{medians[0]:.3f}–{medians[1]:.3f}"
            body += _text(x + 505, y + 4, middle_text, size=12, weight="700", anchor="end")
        ay = 551
        body += f'<path d="M{bx} {ay}h{width}" stroke="{BLACK}"/>\n'
        for tick in range(5):
            value = maximum * tick / 4
            body += _text(bx + tick * width / 4, ay + 20, f"{value:.0f}" if panel == 0 else f"${value:.2f}", size=10, anchor="middle")
    for i, arm in enumerate(ARMS):
        stats = record["arms"][arm]
        x = 40 + i * 275
        body += _text(x, 620, LABELS[i], size=15, family=SERIF, weight="700")
        body += _text(x, 648, f"Accepted: {stats['accepted']}/50", size=14)
        body += _text(x, 673, f"Rubric mean: {stats['quality_mean']:.2f}/4", size=12)
        cpa = stats["cost_per_accepted_usd"]
        bounds = stats["cost_per_accepted_bounds_usd"]
        cost_text = "undefined" if bounds is None else (f"${cpa:.4f}" if cpa is not None else f"${bounds[0]:.4f}–${bounds[1]:.4f}")
        body += _text(x, 697, "Cost / accepted: " + cost_text, size=12)
    body += _text(40, 741, "Points show known observations; boxes span observed or bounded quartiles; ticks and values mark medians.", size=12)
    if record["billing"]["unknown_cost_workflows"]:
        body += _text(40, 765, "Dashed cost range: one failed call has no usage receipt. Its full reserve bounds the charge; no point is invented.", size=12)
    body += _text(40, 789, "All failures remain included. Charges use recorded usage × dated list prices; judge fees are separate.", size=12)
    body += _text(40, 813, "Amended study on reused tasks · descriptive comparison, not an external benchmark or untouched holdout.", size=12)
    body += _text(40, 846, f"Source: astra_20260913_main/analysis.json · SHA-256 {digest[:16]}", size=10)
    return _svg(1140, 870, body, label="All 200 Astra and Sol workflow costs or reserved-cost bounds, timings and acceptance counts")


def render_differences(path):
    from benchmarks.render_readme_charts import BLACK, SERIF, _svg, _text

    record, _, digest = checked_record(path)
    body = _text(40, 38, "GENERATED TOPOLOGY / FIXED PIPELINE", size=12, weight="700", tracking=1.5)
    body += _text(40, 80, "The orchestration effect, within each model", size=30, family=SERIF, weight="700")
    body += _text(40, 111, "Generated minus fixed · 95% task-clustered bootstrap intervals · 10 tasks, not 50 independent samples", size=12)
    for panel, (metric, title) in enumerate((("wall_seconds", "Difference in mean wall time (s)"),
                                              ("cost_usd", "Difference in mean generation cost ($)"))):
        x = 40 + panel * 550
        bx, width = x + 95, 365
        comparisons = record["within_model_comparisons"]
        intervals = [v for model in ("astra", "sol") if comparisons[model]["metrics"][metric] is not None
                     for v in comparisons[model]["metrics"][metric]["ci95"]]
        limit = max((abs(v) for v in intervals), default=1) * 1.2 or 1
        center = bx + width / 2
        body += _text(x, 166, title, size=19, family=SERIF, weight="700")
        body += f'<path d="M{center} 195v173" stroke="{BLACK}" stroke-dasharray="4 4"/>\n'
        for i, model in enumerate(("astra", "sol")):
            result = comparisons[model]["metrics"][metric]
            y = 239 + i * 98
            body += _text(x, y + 5, model.title(), size=16, family=SERIF, weight="700")
            if result is None:
                body += _text(center, y + 5, "Cost interval withheld: one unknown charge", size=11, anchor="middle")
                continue
            left, right, dot = (center + v / limit * width / 2 for v in (*result["ci95"], result["mean_difference"]))
            body += f'<path d="M{left:.4f} {y}H{right:.4f}M{left:.4f} {y - 6}v12M{right:.4f} {y - 6}v12" stroke="{BLACK}"/>\n'
            body += f'<circle cx="{dot:.4f}" cy="{y}" r="5" fill="{BLACK}"/>\n'
            body += _text(center, y + 30, f"{result['mean_difference']:+.4f} [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]", size=11, anchor="middle")
        for tick in (-1, 0, 1):
            body += _text(center + tick * width / 2, 407, f"{tick * limit:+.2f}" if tick else "0", size=11, anchor="middle")
        body += _text(bx, 436, "← Favors generated", size=12)
        body += _text(bx + width, 436, "Favors fixed →", size=12, anchor="end")
    body += _text(40, 492, "Intervals resample ten whole-task means (10,000 draws; seed 14173). The main study's failed run stays included.", size=12)
    body += _text(40, 518, "The report includes quality/success gates, both model contrasts, and the model-by-strategy interaction.", size=12)
    body += _text(40, 544, "Reused project-authored tasks limit generalization. An interval spanning zero does not establish a directional effect.", size=12)
    body += _text(40, 580, f"Source: astra_20260913_main/analysis.json · SHA-256 {digest[:16]}", size=10)
    return _svg(1140, 605, body, label="Paired generated-minus-fixed cost and time differences for Astra and Sol with task-clustered uncertainty")
