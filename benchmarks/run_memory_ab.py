"""Memory A/B: does PlannerMemory recall change later plans?

    python benchmarks/run_memory_ab.py                # offline mechanics
    python benchmarks/run_memory_ab.py --runs 3       # real A/B with judging

A family of five similar tasks runs twice, in sequence: once with
PlannerMemory enabled (each outcome recorded, recalled into the next
planning prompt) and once without. If the learning Architect claim
holds, the memory-on condition should show it on positions 2-5 -
different topology choices, cost, or quality. Position 1 is the
control within each sequence: memory is empty either way.

Each record notes whether the planning prompt actually contained
recalled history, so the wiring is observable, not assumed.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from judge import score_output

from smythe import Swarm, Task
from smythe.memory import PlannerMemory
from smythe.provider import Provider

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

BENCH_DIR = Path(__file__).parent
HISTORY_MARKER = "## Relevant past executions"

PRODUCTS = [
    "espresso makers", "cold-brew makers", "milk frothers",
    "pour-over kettles", "coffee grinders",
]

RUBRIC = [
    "Names concrete competitors, not placeholders",
    "Market landscape covers segments and price bands",
    "The one-page summary is decision-oriented, not a listicle",
    "Stays under the length constraint",
    "Claims are hedged where the evidence is thin",
]


def make_task(product: str) -> Task:
    return Task(
        goal=(f"Produce a competitive brief on portable {product}: "
              "market landscape, top competitors, and a one-page summary."),
        constraints=["Keep the final brief under 400 words"],
    )


class PlanningRecorder(Provider):
    """Delegates to a real provider while capturing planning prompts."""

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.planning_prompts: list[str] = []

    async def complete(self, system, prompt, model):
        from smythe.prompts import PLANNING_SYSTEM_PROMPT
        if system == PLANNING_SYSTEM_PROMPT:
            self.planning_prompts.append(prompt)
        return await self._inner.complete(system, prompt, model)

    async def chat(self, system, messages, model, tools=None):
        return await self._inner.chat(system, messages, model, tools=tools)


def make_provider(offline: bool):
    """Return (provider factory, executor model, judge model)."""
    if offline:
        from harness import OFFLINE_DYNAMIC_PLAN
        from smythe.provider import OfflineProvider
        return (lambda: OfflineProvider(plan=OFFLINE_DYNAMIC_PLAN),
                "offline-model", None)
    from smythe.provider import AnthropicProvider
    return lambda: AnthropicProvider(), "claude-opus-4-8", "claude-sonnet-5"


def run_sequence(memory_on: bool, run_index: int, *, offline: bool,
                 judge: bool) -> list[dict]:
    factory, model, judge_model = make_provider(offline)
    memory = None
    if memory_on:
        memory = PlannerMemory(
            path=Path(tempfile.mkdtemp(prefix="smythe-memab-")) / "history.jsonl"
        )

    records = []
    for position, product in enumerate(PRODUCTS):
        recorder = PlanningRecorder(factory())
        swarm = Swarm(provider=recorder, model=model, memory=memory,
                      parallel=True, max_budget_usd=2.00)
        task = make_task(product)
        started = time.perf_counter()
        result = swarm.execute(task)
        wall_ms = round((time.perf_counter() - started) * 1000, 1)

        graph = result.graph
        terminals = [n for n in graph.nodes if not graph.dependents(n.id)]
        output = "\n\n".join(str(n.result) for n in terminals
                             if n.result is not None)
        recalled = any(HISTORY_MARKER in p for p in recorder.planning_prompts)

        record = {
            "condition": "memory_on" if memory_on else "memory_off",
            "run": run_index,
            "position": position,
            "product": product,
            "history_recalled": recalled,
            "topology": " -> ".join(t.value for t in graph.topology),
            "nodes": len(graph.nodes),
            "cost_usd": round(result.total_cost_usd, 6),
            "wall_ms": None if offline else wall_ms,
            "output": output,
            "quality": None,
        }
        if judge and not offline:
            record["quality"] = score_output(
                factory(), judge_model, task.goal, RUBRIC, output,
            )
            record["judge_model"] = judge_model
        records.append(record)
        print(f"  done: {record['condition']} run={run_index} "
              f"pos={position} ({product})  nodes={record['nodes']}  "
              f"recalled={recalled}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="repeat each condition's full sequence N times")
    parser.add_argument("--offline", action="store_true",
                        help="force offline mode even if an API key is set")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="write results JSON to FILE")
    args = parser.parse_args()

    offline = args.offline or not os.environ.get("ANTHROPIC_API_KEY")
    if offline:
        print("Running offline: mechanics only, no quality scores, no cost.\n")

    records = []
    for run_index in range(args.runs):
        for memory_on in (False, True):
            records.extend(run_sequence(
                memory_on, run_index, offline=offline, judge=not offline,
            ))

    print(f"\n{'condition':<12} {'positions 2-5 quality':>22} "
          f"{'mean nodes':>11} {'mean cost':>10}")
    for condition in ("memory_off", "memory_on"):
        late = [r for r in records
                if r["condition"] == condition and r["position"] >= 1]
        scores = [r["quality"]["overall"] for r in late if r["quality"]]
        quality = (f"{sum(scores) / len(scores):.1f}" if scores else "-")
        nodes = sum(r["nodes"] for r in late) / len(late)
        cost = sum(r["cost_usd"] for r in late) / len(late)
        print(f"{condition:<12} {quality:>22} {nodes:>11.1f} {cost:>10.4f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = BENCH_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
