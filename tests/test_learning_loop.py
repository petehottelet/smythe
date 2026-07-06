"""End-to-end: a recorded outcome reaches the next plan and changes it.

The unit layers are covered elsewhere (test_memory.py, test_llm_planner.py);
these tests prove the full loop through the Swarm: execute -> record ->
recall -> planning prompt -> different graph.
"""

import json

from smythe import Swarm, Task
from smythe.memory import PlannerMemory
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult, OfflineProvider

HISTORY_MARKER = "## Relevant past executions"

WIDE_PLAN = {
    "topology": ["fork_join"],
    "nodes": [
        {"id": "a", "label": "Research angle A", "depends_on": [],
         "agent": {"name": "A", "persona": "p"}},
        {"id": "b", "label": "Research angle B", "depends_on": [],
         "agent": {"name": "B", "persona": "p"}},
        {"id": "c", "label": "Research angle C", "depends_on": [],
         "agent": {"name": "C", "persona": "p"}},
        {"id": "join", "label": "Merge", "depends_on": ["a", "b", "c"],
         "agent": {"name": "J", "persona": "p"}},
    ],
}

NARROW_PLAN = {
    "topology": ["serial"],
    "nodes": [
        {"id": "solo", "label": "Do the whole thing", "depends_on": [],
         "agent": {"name": "Solo", "persona": "p"}},
    ],
}


class HistorySensitiveProvider(OfflineProvider):
    """Returns a different plan when the planning prompt carries history."""

    def __init__(self) -> None:
        super().__init__()
        self.planning_prompts: list[str] = []

    async def complete(self, system, prompt, model):
        if system == PLANNING_SYSTEM_PROMPT:
            self.planning_prompts.append(prompt)
            plan = NARROW_PLAN if HISTORY_MARKER in prompt else WIDE_PLAN
            return CompletionResult(
                text=json.dumps(plan), prompt_tokens=10, completion_tokens=10,
            )
        return await super().complete(system, prompt, model)


def make_swarm(tmp_path):
    provider = HistorySensitiveProvider()
    memory = PlannerMemory(path=tmp_path / "history.jsonl")
    swarm = Swarm(provider=provider, model="test-model", memory=memory,
                  max_budget_usd=1.00)
    return swarm, provider, memory


TASK_1 = Task(goal="Compare portable espresso makers across the market")
TASK_2 = Task(goal="Compare portable espresso grinders across the market")


def test_recorded_outcome_reaches_next_planning_prompt(tmp_path):
    swarm, provider, _memory = make_swarm(tmp_path)

    swarm.execute(TASK_1)
    assert len(provider.planning_prompts) == 1
    assert HISTORY_MARKER not in provider.planning_prompts[0]

    swarm.plan(TASK_2)
    assert len(provider.planning_prompts) == 2
    assert HISTORY_MARKER in provider.planning_prompts[1]
    # The recalled line carries the prior topology and outcome.
    assert "fork_join" in provider.planning_prompts[1]
    assert "success" in provider.planning_prompts[1]


def test_recalled_history_changes_the_plan(tmp_path):
    swarm, _provider, _memory = make_swarm(tmp_path)

    result1 = swarm.execute(TASK_1)
    assert len(result1.graph.nodes) == 4

    graph2 = swarm.plan(TASK_2)
    assert len(graph2.nodes) == 1


def test_unrelated_task_recalls_nothing(tmp_path):
    swarm, provider, _memory = make_swarm(tmp_path)

    swarm.execute(TASK_1)
    swarm.plan(Task(goal="Draft onboarding checklist welcoming new hires"))
    assert HISTORY_MARKER not in provider.planning_prompts[1]
