"""The shape suite: does generated topology pay off when shapes differ?

    python benchmarks/run_shape_suite.py                    # offline mechanics
    python benchmarks/run_shape_suite.py --live --reps 3    # ~$3

Every previously published smythe benchmark used a task set whose
members all fit one shape — research, analyse, write. On such a set a
*generated* topology cannot beat a *fixed* one: there is nothing to
adapt to, and the planning call is pure overhead. The published near-parity
result (dynamic 9.27 vs fixed 9.67) is therefore a fact about the task
set as much as about the framework.

This suite fixes that. Five tasks with deliberately different natural
shapes:

    trivial-transform   one step; a 3-step pipeline wastes two calls
    parallel-profiles   four independent subjects; serial conflates them
    deep-serial         each step consumes the previous step's number
    adversarial-claim   needs a case and then an attack on it
    mixed-audit         three separate defect classes, then a fix

Pre-registered hypothesis: mean quality will be close (LLM judges
compress), but **cost per accepted quality point** will favour dynamic
topology, because the fixed pipeline must overspend on the trivial task
and under-structure the parallel one. If that does not show up, the
honest conclusion is that generated topology does not pay for itself on
this workload either — and that result gets published too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from harness import BenchmarkTask, load_tasks, make_swarm, offline_provider  # noqa: E402

from smythe.provider import GeminiProvider, OpenAIProvider  # noqa: E402

EXECUTOR_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gemini-pro-latest"
BASELINES = ("single_agent", "fixed_pipeline", "smythe_dynamic")

JUDGE_PROMPT = """Score this deliverable against the rubric. Judge only \
whether the rubric criteria are met — ignore length and polish. Be strict: \
a deliverable that misses a criterion cannot score above 6. Respond with \
STRICT JSON only, no prose, no code fences: {{"overall": <1-10>}}

Rubric:
{rubric}

Deliverable:
{output}"""


def judge(output: str, rubric: list[str]) -> int | None:
    if not output.strip():
        return 1
    prompt = JUDGE_PROMPT.format(
        rubric="\n".join(f"- {r}" for r in rubric), output=output[:24000],
    )
    result = asyncio.run(GeminiProvider().complete(
        "You are a strict, fair judge. Strict JSON only.", prompt, JUDGE_MODEL,
    ))
    text = result.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return int(json.loads(text)["overall"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def run_one(task: BenchmarkTask, baseline: str, *, live: bool) -> dict:
    provider = OpenAIProvider() if live else offline_provider(baseline)
    model = EXECUTOR_MODEL if live else "demo-model"
    swarm = make_swarm(baseline, provider, model)
    started = time.perf_counter()
    result = swarm.execute(task.to_task())
    wall_s = round(time.perf_counter() - started, 2)

    graph = result.graph
    # Two conventions, both recorded. "terminal" is what the older
    # harness measured; "delivered" is what Swarm.execute actually hands
    # back. They diverge sharply for decomposed plans whose deliverable
    # is cumulative, and that divergence is itself a finding.
    terminals = [n for n in graph.nodes if not graph.dependents(n.id)]
    terminal_output = "\n\n".join(
        str(n.result) for n in terminals if n.result is not None
    )
    return {
        "task": task.name,
        "baseline": baseline,
        "nodes": len(graph.nodes),
        "depth": graph.depth,
        "topology": " -> ".join(t.value for t in graph.topology),
        "cost_usd": round(result.total_cost_usd, 6),
        "wall_s": None if not live else wall_s,
        "output_terminal": terminal_output,
        "output_delivered": result.output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("OPENAI_API_KEY"))
    if args.live and not live:
        print("--live needs OPENAI_API_KEY; running offline.")
    if live and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Live runs need GOOGLE_API_KEY for the judge.")

    tasks = load_tasks(Path(__file__).parent / "tasks_shapes")
    records: list[dict] = []
    for task in tasks:
        for baseline in BASELINES:
            for rep in range(args.reps):
                try:
                    record = run_one(task, baseline, live=live)
                    if live:
                        record["quality_terminal"] = judge(
                            record["output_terminal"], task.rubric)
                        record["quality"] = judge(
                            record["output_delivered"], task.rubric)
                    else:
                        record["quality"] = record["quality_terminal"] = None
                    record["error"] = None
                except Exception as exc:
                    record = {
                        "task": task.name, "baseline": baseline, "quality": None,
                        "quality_terminal": None,
                        "cost_usd": None, "wall_s": None, "nodes": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                record["rep"] = rep
                record.pop("output_terminal", None)
                record.pop("output_delivered", None)
                records.append(record)
                print(f"  {task.name:<20} {baseline:<16} rep={rep} "
                      f"q={record.get('quality')} nodes={record.get('nodes')} "
                      f"${record.get('cost_usd')}"
                      + (f" ERROR {record['error']}" if record["error"] else ""))

    summary = []
    for baseline in BASELINES:
        cell = [r for r in records if r["baseline"] == baseline and not r["error"]]
        quals = [r["quality"] for r in cell if r["quality"] is not None]
        quals_terminal = [
            r["quality_terminal"] for r in cell
            if r.get("quality_terminal") is not None
        ]
        costs = [r["cost_usd"] for r in cell if r["cost_usd"] is not None]
        entry = {
            "baseline": baseline,
            "runs": len(cell),
            "quality_mean": round(mean(quals), 2) if quals else None,
            "quality_min": min(quals) if quals else None,
            "quality_terminal_mean": (
                round(mean(quals_terminal), 2) if quals_terminal else None
            ),
            "nodes_mean": round(mean(r["nodes"] for r in cell), 2) if cell else None,
            "cost_total_usd": round(sum(costs), 5) if costs else None,
        }
        # The metric the homogeneous suite could not expose: what a point
        # of judged quality actually costs on a shape-varied workload.
        if quals and costs and sum(quals):
            entry["usd_per_quality_point"] = round(sum(costs) / sum(quals), 6)
        summary.append(entry)

    by_task = {}
    for task in tasks:
        by_task[task.name] = {
            b: {
                "quality": [
                    r["quality"] for r in records
                    if r["task"] == task.name and r["baseline"] == b
                ],
                "nodes": [
                    r["nodes"] for r in records
                    if r["task"] == task.name and r["baseline"] == b
                ],
            }
            for b in BASELINES
        }

    payload = {
        "benchmark": "shape-suite",
        "mode": "LIVE" if live else "offline",
        "executor_model": EXECUTOR_MODEL if live else "offline",
        "judge_model": JUDGE_MODEL if live else None,
        "reps": args.reps,
        "summary": summary,
        "by_task": by_task,
        "records": records,
    }
    out = args.out or (
        "benchmarks/results/shape_suite.json" if live
        else "benchmarks/results/shape_suite_offline.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
