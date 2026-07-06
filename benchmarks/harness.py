"""Benchmark harness — run one task through three execution strategies.

Baselines:

- ``single_agent``   — one node, the whole goal. What you get from a
  bare LLM call with a good prompt.
- ``fixed_pipeline`` — the same hardcoded research -> analyze -> write
  pipeline for every task. What pipeline frameworks have you build.
- ``smythe_dynamic`` — the LLMArchitect designs a task-specific graph.
  The thing Smythe claims is worth the planning call.

Offline mode verifies the harness mechanics deterministically (no keys,
no cost); quality judging only makes sense against real providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from smythe import Swarm, Task
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.planner import DeterministicArchitect, SimpleArchitect
from smythe.provider import OfflineProvider, Provider
from smythe.registry import Registry

BASELINES = ("single_agent", "fixed_pipeline", "smythe_dynamic")


@dataclass
class BenchmarkTask:
    """A benchmark task: what to do, and the rubric a judge scores against."""

    name: str
    goal: str
    constraints: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)

    def to_task(self) -> Task:
        return Task(goal=self.goal, constraints=list(self.constraints))


def load_tasks(tasks_dir: str | Path) -> list[BenchmarkTask]:
    """Load every task YAML in the directory, sorted by filename."""
    tasks = []
    for path in sorted(Path(tasks_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        tasks.append(BenchmarkTask(
            name=data["name"],
            goal=data["goal"].strip(),
            constraints=data.get("constraints", []),
            rubric=data.get("rubric", []),
        ))
    return tasks


class FixedPipelineArchitect(DeterministicArchitect):
    """The hardcoded-pipeline strawman, held fixed across all tasks.

    Research -> analyze -> write, serial. Deliberately reasonable — the
    comparison is unfair if the fixed baseline is a caricature.
    """

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        registry = Registry()
        specs = [
            ("research", "Research the topic and gather the relevant facts for: ",
             "You are a thorough researcher. Gather facts, note uncertainty."),
            ("analyze", "Analyze the research findings and identify risks, "
             "trade-offs, and open questions for: ",
             "You are a critical analyst. Weigh evidence, surface risks."),
            ("write", "Write the final deliverable, meeting every constraint, for: ",
             "You write clear, structured deliverables."),
        ]
        nodes: list[Node] = []
        prev: str | None = None
        for node_id, label_prefix, persona in specs:
            agent = Agent(profile=AgentProfile(name=node_id.capitalize(), persona=persona))
            registry.register(agent)
            node = Node(
                id=node_id,
                label=label_prefix + task.goal,
                agent_id=agent.id,
                depends_on=[prev] if prev else [],
                metadata={"agent_name": agent.name},
            )
            nodes.append(node)
            prev = node_id
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)
        graph.validate()
        return graph, registry


# Generic plan the offline provider returns for the dynamic baseline, so
# harness mechanics (planning call, fork-join execution, synthesis) run
# without keys. Real dynamic-topology numbers require a real provider.
OFFLINE_DYNAMIC_PLAN = {
    "topology": ["fork_join"],
    "nodes": [
        {
            "id": "facts",
            "label": "Investigate the core facts and current state",
            "depends_on": [],
            "agent": {"name": "FactFinder",
                      "persona": "You establish what is known."},
        },
        {
            "id": "risks",
            "label": "Investigate risks, obstacles, and counterpoints",
            "depends_on": [],
            "agent": {"name": "RiskAnalyst",
                      "persona": "You find what could go wrong."},
        },
        {
            "id": "deliver",
            "label": "Synthesize findings into the final deliverable",
            "depends_on": ["facts", "risks"],
            "agent": {"name": "Synthesizer",
                      "persona": "You write the final deliverable."},
        },
    ],
}


def make_swarm(baseline: str, provider: Provider, model: str) -> Swarm:
    """Build the Swarm variant for a baseline. Same provider and model for all."""
    common = dict(provider=provider, model=model, parallel=True, max_budget_usd=5.00)
    if baseline == "single_agent":
        return Swarm(architect=SimpleArchitect(), **common)
    if baseline == "fixed_pipeline":
        return Swarm(architect=FixedPipelineArchitect(), **common)
    if baseline == "smythe_dynamic":
        return Swarm(**common)  # default LLMArchitect
    raise ValueError(f"Unknown baseline {baseline!r}; expected one of {BASELINES}")


def offline_provider(baseline: str) -> OfflineProvider:
    """Deterministic provider per baseline for mechanics-only runs."""
    if baseline == "smythe_dynamic":
        return OfflineProvider(plan=OFFLINE_DYNAMIC_PLAN)
    return OfflineProvider()


def run_one(
    bench_task: BenchmarkTask,
    baseline: str,
    provider: Provider,
    model: str,
    *,
    offline: bool,
) -> dict:
    """Execute one (task, baseline) pair and return its metrics record."""
    swarm = make_swarm(baseline, provider, model)
    started = time.perf_counter()
    result = swarm.execute(bench_task.to_task())
    wall_ms = round((time.perf_counter() - started) * 1000, 1)

    graph = result.graph
    terminals = [n for n in graph.nodes if not graph.dependents(n.id)]
    output = "\n\n".join(str(n.result) for n in terminals if n.result is not None)

    return {
        "task": bench_task.name,
        "baseline": baseline,
        "model": model,
        "offline": offline,
        "topology": " -> ".join(t.value for t in graph.topology),
        "nodes": len(graph.nodes),
        "depth": graph.depth,
        "cost_usd": round(result.total_cost_usd, 6),
        # Offline wall time measures the OS scheduler, not the work.
        "wall_ms": None if offline else wall_ms,
        "output": output,
        "quality": None,  # filled by the judge in real mode
    }
