"""Framework head-to-head: one semantic pipeline in Smythe, LangGraph, CrewAI.

    python benchmarks/run_framework_h2h.py --tasks 1 --reps 1
    python benchmarks/run_framework_h2h.py                       # full

This is an ecological comparison of idiomatic framework usage. Every
framework receives the same task, semantic research -> analyze -> write
specification, persona text, and executor model. Each framework packages
those ingredients, dependency context, and its own scaffolding differently,
so the provider-facing prompts are not byte-identical. The benchmark measures
the resulting end-to-end system, not isolated scheduler overhead.
``smythe_dynamic`` (the LLMArchitect) rides along as the fourth system.

Protocol notes:
- Executor: one model for every system (default gpt-5.4-mini).
- Judge: Gemini — a different vendor than the executor, addressing the
  self-preference caveat published with the v5 results. Blind: the
  judge sees output + rubric, never the system name.
- Metrics: blind quality (1-10), total tokens, wall seconds.
- smythe token counts are derived from its blended-rate cost tracking
  (cost / $3e-6); LangGraph counts come from usage_metadata; CrewAI
  from crew usage metrics. Derivations are noted in the records.
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

from harness import PIPELINE_SPECS, BenchmarkTask, load_tasks, make_swarm  # noqa: E402

from benchmarks.artifact_records import environment_snapshot  # noqa: E402
from smythe.provider import GeminiProvider, OpenAIProvider  # noqa: E402

EXECUTOR_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gemini-pro-latest"
BLENDED_RATE = 0.000003  # smythe Sentinel default $/token, for token derivation

JUDGE_PROMPT = """Score this deliverable against the rubric. Be strict: \
10 is rare. Respond with STRICT JSON only, no prose, no code fences: \
{{"overall": <1-10>}}

Rubric:
{rubric}

Deliverable:
{output}"""


def step_prompt(prefix: str, task: BenchmarkTask, context: str | None) -> str:
    parts = [
        prefix + task.goal,
        "Constraints:\n" + "\n".join(f"- {c}" for c in task.constraints),
    ]
    if context:
        parts.append("Context from prior steps:\n" + context)
    return "\n\n".join(parts)


def run_smythe(baseline: str, task: BenchmarkTask) -> dict:
    swarm = make_swarm(baseline, OpenAIProvider(), EXECUTOR_MODEL)
    t0 = time.perf_counter()
    result = swarm.execute(task.to_task())
    wall_s = time.perf_counter() - t0
    graph = result.graph
    terminals = [n for n in graph.nodes if not graph.dependents(n.id)]
    output = "\n\n".join(str(n.result) for n in terminals if n.result is not None)
    return {
        "output": output,
        "wall_s": round(wall_s, 2),
        "tokens": round(result.total_cost_usd / BLENDED_RATE),
        "tokens_source": "derived_from_blended_cost",
        "nodes": len(graph.nodes),
    }


def run_langgraph(task: BenchmarkTask) -> dict:
    from typing import TypedDict

    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(model=EXECUTOR_MODEL)
    usage = {"tokens": 0}

    class State(TypedDict, total=False):
        research: str
        analyze: str
        write: str

    def make_node(step_id: str, prefix: str, persona: str, prev: str | None):
        def node(state: State) -> State:
            context = state.get(prev) if prev else None
            msg = llm.invoke([
                ("system", persona),
                ("user", step_prompt(prefix, task, context)),
            ])
            meta = getattr(msg, "usage_metadata", None) or {}
            usage["tokens"] += meta.get("total_tokens", 0)
            return {step_id: msg.content}
        return node

    builder = StateGraph(State)
    prev = None
    for step_id, prefix, persona in PIPELINE_SPECS:
        builder.add_node(step_id, make_node(step_id, prefix, persona, prev))
        builder.add_edge(prev if prev else START, step_id)
        prev = step_id
    builder.add_edge(prev, END)
    app = builder.compile()

    t0 = time.perf_counter()
    final = app.invoke({})
    wall_s = time.perf_counter() - t0
    return {
        "output": final.get("write", ""),
        "wall_s": round(wall_s, 2),
        "tokens": usage["tokens"],
        "tokens_source": "usage_metadata",
        "nodes": len(PIPELINE_SPECS),
    }


def run_crewai(task: BenchmarkTask) -> dict:
    from crewai import LLM, Agent, Crew, Process
    from crewai import Task as CrewTask

    llm = LLM(model=f"openai/{EXECUTOR_MODEL}")
    agents, crew_tasks, prev_task = [], [], None
    for step_id, prefix, persona in PIPELINE_SPECS:
        agent = Agent(
            role=step_id.capitalize(),
            goal=prefix + task.goal,
            backstory=persona,
            llm=llm,
            verbose=False,
        )
        agents.append(agent)
        crew_task = CrewTask(
            description=step_prompt(prefix, task, None),
            expected_output=(
                "The final deliverable, meeting every constraint."
                if step_id == "write"
                else f"The {step_id} step's findings."
            ),
            agent=agent,
            context=[prev_task] if prev_task else [],
        )
        crew_tasks.append(crew_task)
        prev_task = crew_task
    crew = Crew(
        agents=agents, tasks=crew_tasks, process=Process.sequential,
        verbose=False,
    )
    t0 = time.perf_counter()
    result = crew.kickoff()
    wall_s = time.perf_counter() - t0
    metrics = getattr(crew, "usage_metrics", None)
    tokens = getattr(metrics, "total_tokens", 0) if metrics else 0
    return {
        "output": str(result),
        "wall_s": round(wall_s, 2),
        "tokens": tokens,
        "tokens_source": "crew_usage_metrics",
        "nodes": len(PIPELINE_SPECS),
    }


def judge(output: str, rubric: list[str]) -> int | None:
    prompt = JUDGE_PROMPT.format(
        rubric="\n".join(f"- {r}" for r in rubric),
        output=output[:24000],
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


SYSTEMS = {
    "smythe_fixed": lambda t: run_smythe("fixed_pipeline", t),
    "smythe_dynamic": lambda t: run_smythe("smythe_dynamic", t),
    "langgraph_fixed": run_langgraph,
    "crewai_fixed": run_crewai,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=None,
                        help="limit to first N tasks")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--skip", default="",
                        help="comma-separated systems to skip")
    parser.add_argument("--out", default="benchmarks/results/framework_h2h.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("OPENAI_API_KEY (executor) and GOOGLE_API_KEY (judge) required")

    tasks = load_tasks(Path(__file__).parent / "tasks")
    if args.tasks:
        tasks = tasks[: args.tasks]
    skip = set(filter(None, args.skip.split(",")))

    records = []
    for task in tasks:
        for system, runner in SYSTEMS.items():
            if system in skip:
                continue
            for rep in range(args.reps):
                try:
                    rec = runner(task)
                    rec["quality"] = judge(rec["output"], task.rubric)
                    rec["error"] = None
                except Exception as exc:  # record failures, don't hide them
                    rec = {"output": "", "wall_s": None, "tokens": None,
                           "quality": None, "error": f"{type(exc).__name__}: {exc}"}
                rec.update({"task": task.name, "system": system, "rep": rep})
                rec.pop("output", None)
                records.append(rec)
                print(f"  {task.name:<24} {system:<16} rep={rep} "
                      f"q={rec['quality']} tok={rec.get('tokens')} "
                      f"wall={rec.get('wall_s')}s"
                      + (f" ERROR {rec['error']}" if rec["error"] else ""))

    summary = []
    for system in SYSTEMS:
        cell = [r for r in records if r["system"] == system and not r["error"]]
        errors = sum(1 for r in records if r["system"] == system and r["error"])
        quals = [r["quality"] for r in cell if r["quality"] is not None]
        if cell:
            summary.append({
                "system": system,
                "runs_ok": len(cell),
                "errors": errors,
                "quality_mean": round(mean(quals), 2) if quals else None,
                "quality_range": [min(quals), max(quals)] if quals else None,
                "tokens_mean": round(mean(r["tokens"] for r in cell)),
                "wall_s_mean": round(mean(r["wall_s"] for r in cell), 2),
            })

    payload = {
        "benchmark": "framework-head-to-head",
        "comparison_type": "ecological_framework_comparison",
        "prompt_parity": (
            "same tasks, semantic step specifications, personas, and model; "
            "framework-native prompt and context packaging"
        ),
        "executor_model": EXECUTOR_MODEL,
        "judge_model": JUDGE_MODEL + " (different vendor than executor)",
        "environment": environment_snapshot(
            "smythe", "openai", "google-genai", "langgraph",
            "langchain-openai", "crewai",
        ),
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
