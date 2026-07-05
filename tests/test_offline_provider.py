"""Tests for OfflineProvider — the deterministic no-network provider."""

import asyncio

from smythe import Swarm, Task
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import OfflineProvider


def test_echo_mode_is_deterministic():
    p = OfflineProvider()
    r1 = asyncio.run(p.complete("sys", "Do the thing\nwith context", "m"))
    r2 = asyncio.run(p.complete("sys", "Do the thing\nwith context", "m"))
    assert r1.text == r2.text == "offline: Do the thing"


def test_scripted_responses_in_order_last_repeats():
    p = OfflineProvider(responses=["one", "two"])
    assert asyncio.run(p.complete("s", "a", "m")).text == "one"
    assert asyncio.run(p.complete("s", "b", "m")).text == "two"
    assert asyncio.run(p.complete("s", "c", "m")).text == "two"


def test_planning_prompt_returns_plan_json():
    plan = {"topology": ["serial"], "nodes": [{"id": "n1", "label": "L"}]}
    p = OfflineProvider(plan=plan)
    result = asyncio.run(p.complete(PLANNING_SYSTEM_PROMPT, "goal", "m"))
    assert '"topology"' in result.text
    assert result.prompt_tokens > 0


def test_calls_recorded_for_assertions():
    p = OfflineProvider()
    asyncio.run(p.complete("s", "First line\nsecond", "m"))
    assert p.calls == ["First line"]


def test_full_swarm_pipeline_runs_offline():
    """Architect -> executor -> synthesizer, no keys, no network."""
    plan = {
        "topology": ["serial"],
        "nodes": [
            {"id": "step-1", "label": "Do research", "depends_on": []},
            {"id": "step-2", "label": "Summarize it", "depends_on": ["step-1"]},
        ],
    }
    swarm = Swarm(provider=OfflineProvider(plan=plan), model="demo", max_budget_usd=1.0)
    result = swarm.execute(Task(goal="Research and summarize"))
    assert "offline: Do research" in result.output
    assert "offline: Summarize it" in result.output
    assert result.total_cost_usd > 0
