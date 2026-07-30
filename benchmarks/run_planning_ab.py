"""Does planning need the executor's model?

    python benchmarks/run_planning_ab.py --live --reps 3   # ~$1.50

Generated topology pays for a planning call the fixed pipeline does not
— a measured 41% token premium in the framework head-to-head. But
planning is structured, low-creativity work: read a task, emit a small
JSON DAG. It is not obvious it needs the same model that does the
actual thinking.

This A/B holds everything constant except the planner:

    same-model   planning runs on the executor model (today's default)
    cheap        planning runs on a small, cheap model

Pre-registered hypothesis: quality is flat within judge noise and cost
falls. If quality drops, planning is *not* commodity work and the
default should stay — that result is as useful as the alternative.
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

from harness import BenchmarkTask, load_tasks  # noqa: E402

from smythe import Swarm  # noqa: E402
from smythe.provider import GeminiProvider, OpenAIProvider  # noqa: E402

EXECUTOR_MODEL = "gpt-5.4-mini"
CHEAP_PLANNER_MODEL = "gemini-flash-lite-latest"
JUDGE_MODEL = "gemini-pro-latest"

JUDGE_PROMPT = """Score this deliverable against the rubric. Be strict: \
10 is rare. Respond with STRICT JSON only, no prose, no code fences: \
{{"overall": <1-10>}}

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
    try:
        result = asyncio.run(GeminiProvider().complete(
            "You are a strict, fair judge. Strict JSON only.", prompt, JUDGE_MODEL,
        ))
    except Exception:
        return None
    text = result.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return int(json.loads(text)["overall"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def run_arm(task: BenchmarkTask, arm: str) -> dict:
    """Run one dynamic-topology execution under the named planner arm."""
    kwargs: dict = {
        "provider": OpenAIProvider(),
        "model": EXECUTOR_MODEL,
        "parallel": True,
        "max_budget_usd": 5.00,
        "artifact_dir": None,
    }
    if arm == "cheap":
        kwargs["planning_provider"] = GeminiProvider()
        kwargs["planning_model"] = CHEAP_PLANNER_MODEL
    swarm = Swarm(**kwargs)

    started = time.perf_counter()
    result = swarm.execute(task.to_task())
    wall_s = round(time.perf_counter() - started, 2)
    return {
        "task": task.name,
        "arm": arm,
        "nodes": len(result.graph.nodes),
        "depth": result.graph.depth,
        "cost_usd": round(result.total_cost_usd, 6),
        "wall_s": wall_s,
        "output": result.output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--out", default="benchmarks/results/planning_ab.json",
    )
    args = parser.parse_args()

    if not args.live:
        raise SystemExit("This A/B only means anything live; pass --live.")
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Needs OPENAI_API_KEY (executor) and GOOGLE_API_KEY.")

    tasks = load_tasks(Path(__file__).parent / "tasks")
    records: list[dict] = []
    for task in tasks:
        for arm in ("same-model", "cheap"):
            for rep in range(args.reps):
                try:
                    record = run_arm(task, arm)
                    record["quality"] = judge(record["output"], task.rubric)
                    record["error"] = None
                except Exception as exc:
                    record = {
                        "task": task.name, "arm": arm, "quality": None,
                        "cost_usd": None, "wall_s": None, "nodes": None,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                record["rep"] = rep
                record.pop("output", None)
                records.append(record)
                print(f"  {task.name:<24} {arm:<11} rep={rep} "
                      f"q={record.get('quality')} nodes={record.get('nodes')} "
                      f"${record.get('cost_usd')}"
                      + (f" ERROR {record['error']}" if record["error"] else ""))

    summary = []
    for arm in ("same-model", "cheap"):
        cell = [r for r in records if r["arm"] == arm and not r["error"]]
        quals = [r["quality"] for r in cell if r["quality"] is not None]
        costs = [r["cost_usd"] for r in cell if r["cost_usd"] is not None]
        summary.append({
            "arm": arm,
            "runs": len(cell),
            "quality_mean": round(mean(quals), 2) if quals else None,
            "quality_min": min(quals) if quals else None,
            "nodes_mean": round(mean(r["nodes"] for r in cell), 2) if cell else None,
            "cost_total_usd": round(sum(costs), 5) if costs else None,
            "wall_s_mean": round(mean(r["wall_s"] for r in cell), 2) if cell else None,
        })

    payload = {
        "benchmark": "planning-model-ab",
        "executor_model": EXECUTOR_MODEL,
        "cheap_planner_model": CHEAP_PLANNER_MODEL,
        "judge_model": JUDGE_MODEL,
        "reps": args.reps,
        "summary": summary,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
