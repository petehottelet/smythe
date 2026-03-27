"""Tests for the Swarm top-level orchestrator."""

import pytest

from smythe import Swarm, Task


def test_swarm_construction():
    swarm = Swarm(max_budget_usd=1.00, model="gpt-4o")
    assert swarm.model == "gpt-4o"
    assert swarm.max_budget_usd == 1.00


def test_swarm_execute_raises_not_implemented():
    """Until an LLM provider is wired up, execute should raise NotImplementedError."""
    swarm = Swarm()
    task = Task(goal="Do something")

    with pytest.raises(NotImplementedError, match="LLM dispatch not yet implemented"):
        swarm.execute(task)
