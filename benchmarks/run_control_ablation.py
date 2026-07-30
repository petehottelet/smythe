"""Do the control features earn their cost? Supervision and gating, ablated.

    python benchmarks/run_control_ablation.py                  # offline mechanics
    python benchmarks/run_control_ablation.py --live --reps 3  # ~$1

Smythe added three control features on the claim that they improve
outcomes: acceptance criteria (``done_when``), verification that gates
and regenerates (``verifies``), and a supervisor that revises the plan
mid-run. None of that was ever measured. This runs the four arms against
the shape suite, which is the only task set the project has that
produces a spread of scores rather than a wall of 9s and 10s.

    plain        default planning; constraints only
    criteria     constraints restated as done_when, plus a review node
                 that runs and is paid for but cannot send work back
    gated        identical plan, and that review node gates for real
    supervised   plain, plus LLMSupervisor with max_revisions=2

`criteria` and `gated` therefore execute the same graph for the same
token cost and differ only in whether a failed verdict is enforced.
Deleting the review node instead would have confounded gating with the
extra call it costs.

The review node is injected when the planner does not produce one. In
the first campaign the planner emitted a gate in only 2 of 15 gated
runs, which made that arm mostly a rerun of `plain` -- the arm reported
a 0.6-point lift from a mechanism that was absent from 13 of its runs.
Single-node plans are left ungated, since adding a node there would
compare 1-node against 2-node rather than isolating enforcement.

``done_when`` is the task's own ``constraints`` verbatim. That matters:
the system already sees the constraints in every arm, so no new
information enters with the criteria. The comparison isolates the
*mechanism* -- restating, checking, revising -- rather than rewarding an
arm for being told more. Feeding the judge's rubric in as done_when
would be teaching to the test and is deliberately not done.

Pre-registered hypothesis: the control features lift the *floor*, not
the mean. The shape suite's failures are concentrated (over-decomposed
trivial tasks, under-structured parallel ones), so the prediction is
fewer sub-7 runs at equal or slightly worse mean, bought with more
tokens. If the floor does not move, the honest conclusion is that these
features do not pay for themselves and further investment in them should
stop -- and that result gets published too.
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

from harness import BenchmarkTask, load_tasks, offline_provider  # noqa: E402

from smythe import LLMSupervisor, Supervisor, Swarm, Task  # noqa: E402
from smythe.graph import Node  # noqa: E402
from smythe.provider import GeminiProvider, OpenAIProvider  # noqa: E402

EXECUTOR_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gemini-pro-latest"
ARMS = ("plain", "criteria", "gated", "supervised")

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
    for attempt in range(1, 4):
        try:
            result = asyncio.run(GeminiProvider().complete(
                "You are a strict, fair judge. Strict JSON only.",
                prompt, JUDGE_MODEL,
            ))
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2.0 * attempt)
            continue
        text = result.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        try:
            return int(json.loads(text)["overall"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            if attempt == 3:
                return None
    return None


def make_task(bench_task: BenchmarkTask, arm: str) -> Task:
    """Build the Task for an arm.

    done_when is the constraints verbatim, so the criteria arms learn
    nothing the plain arm was not already told.
    """
    task = Task(goal=bench_task.goal, constraints=list(bench_task.constraints))
    if arm in ("criteria", "gated"):
        task.done_when = list(bench_task.constraints)
    return task


class CountingSupervisor(Supervisor):
    """Wraps a supervisor and records what it was asked and what it said.

    A supervisor that reviews and declines emits no trace span at all,
    so a run with zero revisions is indistinguishable from a run where
    supervision never engaged. That ambiguity made the first campaign's
    "0 revisions" result uninterpretable; counting here removes it.
    """

    def __init__(self, inner: Supervisor) -> None:
        self.inner = inner
        self.reviews = 0
        self.proposals = 0

    async def review(self, graph, node, *, task, revisions_remaining):
        self.reviews += 1
        revision = await self.inner.review(
            graph, node, task=task, revisions_remaining=revisions_remaining,
        )
        if revision is not None and not revision.is_empty:
            self.proposals += 1
        return revision


def make_swarm(arm: str, provider, model: str) -> tuple[Swarm, CountingSupervisor | None]:
    common = dict(provider=provider, model=model, parallel=True, max_budget_usd=5.00)
    if arm == "supervised":
        supervisor = CountingSupervisor(LLMSupervisor(provider))
        return Swarm(supervisor=supervisor, max_revisions=2, **common), supervisor
    return Swarm(**common), None


def ensure_gate(graph, *, criteria: list[str]) -> bool:
    """Add a verifier on the deliverable node if the plan has none.

    Returns True when one was injected. A single-node plan is left alone:
    gating it would compare a 1-node run against a 2-node run rather than
    isolating enforcement.
    """
    if any(node.verifies for node in graph.nodes) or len(graph.nodes) < 2:
        return False
    depended_on = {dep for node in graph.nodes for dep in node.depends_on}
    terminal = [n for n in graph.nodes if n.id not in depended_on]
    if len(terminal) != 1:
        return False
    target = terminal[0]
    graph.nodes.append(Node(
        id="injected-check",
        label=(
            "Check the deliverable against every criterion below. Reply "
            "PASS if all are met, or FAIL with the specific reasons.\n"
            + "\n".join(f"- {c}" for c in criteria)
        ),
        depends_on=[target.id],
        verifies=target.id,
        max_regenerations=1,
    ))
    return True


def run_one(bench_task: BenchmarkTask, arm: str, *, live: bool) -> dict:
    provider = OpenAIProvider() if live else offline_provider("smythe_dynamic")
    model = EXECUTOR_MODEL if live else "demo-model"
    swarm, supervisor = make_swarm(arm, provider, model)
    task = make_task(bench_task, arm)

    started = time.perf_counter()
    injected = False
    if arm in ("criteria", "gated"):
        graph = swarm.plan(task)
        # The planner emits a gate only sometimes (2 of 15 in the first
        # campaign), so leaving it to chance makes this arm mostly a
        # rerun of `plain`. Injecting one when it is missing is what lets
        # the arm answer the question it was built to ask.
        injected = ensure_gate(graph, criteria=task.done_when)
        if arm == "criteria":
            # Disarm via max_regenerations=0, the documented "advisory
            # verdict" mode, and keep `verifies` set. Clearing `verifies`
            # instead makes the node an ordinary terminal one, so
            # DELIVERABLE synthesis returns the PASS/FAIL verdict as the
            # deliverable -- which scored this arm 3.40 in the previous
            # campaign and measured nothing but the mistake.
            for node in graph.nodes:
                node.max_regenerations = 0
        result = swarm.execute(graph)
    else:
        result = swarm.execute(task)
    wall_s = round(time.perf_counter() - started, 2)

    graph = result.graph
    gates = [n.id for n in graph.nodes if n.verifies]
    regenerations = sum(
        n.metadata.get("regenerations_used", 0) for n in graph.nodes
    )
    revisions = sum(
        1 for span in result.trace if span.get("status") == "revision_applied"
    )
    # Counted separately: a rejected proposal is not the same result as
    # a supervisor that looked and had nothing to say.
    revisions_rejected = sum(
        1 for span in result.trace if span.get("status") == "revision_rejected"
    )
    return {
        "task": bench_task.name,
        "arm": arm,
        "nodes": len(graph.nodes),
        "depth": graph.depth,
        "topology": " -> ".join(t.value for t in graph.topology),
        # Whether the planner actually used the feature. A gated arm in
        # which no plan contains a gate is the same run as `criteria`,
        # and must be reported that way rather than as an effect.
        "gates": gates,
        "gate_injected": injected,
        "regenerations": regenerations,
        "revisions": revisions,
        "revisions_rejected": revisions_rejected,
        "reviews": supervisor.reviews if supervisor else 0,
        "proposals": supervisor.proposals if supervisor else 0,
        "cost_usd": round(result.total_cost_usd, 6),
        "wall_s": None if not live else wall_s,
        "output": result.output,
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
        for arm in ARMS:
            for rep in range(args.reps):
                try:
                    record = run_one(task, arm, live=live)
                    record["quality"] = (
                        judge(record["output"], task.rubric) if live else None
                    )
                    record["error"] = None
                except Exception as exc:
                    # One transient provider failure must not cost the
                    # whole campaign, as it has twice before.
                    record = {
                        "task": task.name, "arm": arm, "quality": None,
                        "nodes": None, "depth": None, "topology": None,
                        "gates": [], "gate_injected": False,
                        "regenerations": 0, "revisions": 0,
                        "revisions_rejected": 0, "reviews": 0, "proposals": 0,
                        "cost_usd": None, "wall_s": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                record["rep"] = rep
                record.pop("output", None)
                records.append(record)
                print(
                    f"  {task.name:<20} {arm:<11} rep={rep} "
                    f"q={record.get('quality')} nodes={record.get('nodes')} "
                    f"gates={len(record.get('gates') or [])} "
                    f"regen={record.get('regenerations')} "
                    f"rev={record.get('revisions')}/"
                    f"{record.get('proposals')}/{record.get('reviews')} "
                    f"${record.get('cost_usd')}"
                    + (f" ERROR {record['error']}" if record["error"] else ""),
                    flush=True,
                )

    summary = []
    for arm in ARMS:
        cell = [r for r in records if r["arm"] == arm and not r["error"]]
        quals = [r["quality"] for r in cell if r["quality"] is not None]
        costs = [r["cost_usd"] for r in cell if r["cost_usd"] is not None]
        summary.append({
            "arm": arm,
            "runs": len(cell),
            "scored": len(quals),
            "quality_mean": round(mean(quals), 2) if quals else None,
            "quality_min": min(quals) if quals else None,
            # The pre-registered metric. The mean is expected to move
            # little; whether the bad runs stop being bad is the claim.
            "floor_runs_under_7": sum(q < 7 for q in quals),
            "nodes_mean": round(mean(r["nodes"] for r in cell), 2) if cell else None,
            "plans_with_a_gate": sum(1 for r in cell if r["gates"]),
            "gates_injected": sum(1 for r in cell if r.get("gate_injected")),
            "regenerations_total": sum(r["regenerations"] for r in cell),
            "revisions_applied": sum(r["revisions"] for r in cell),
            "revisions_rejected": sum(r["revisions_rejected"] for r in cell),
            # The number that tells you whether the feature ran at all.
            "supervisor_reviews": sum(r["reviews"] for r in cell),
            "supervisor_proposals": sum(r["proposals"] for r in cell),
            "cost_total_usd": round(sum(costs), 5) if costs else None,
        })

    payload = {
        "benchmark": "control-ablation",
        "mode": "LIVE" if live else "offline",
        "executor_model": EXECUTOR_MODEL if live else "offline",
        "judge_model": JUDGE_MODEL if live else None,
        "reps": args.reps,
        "done_when_source": "task constraints verbatim (no rubric leakage)",
        "summary": summary,
        "records": records,
    }
    out = args.out or (
        "benchmarks/results/control_ablation.json" if live
        else "benchmarks/results/control_ablation_offline.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
