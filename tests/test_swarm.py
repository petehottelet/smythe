"""Tests for the Swarm top-level orchestrator."""

import os
import tempfile

import pytest

from helpers import (
    ClassifierMockProvider,
    FixedArchitect,
    MockProvider,
    PlanningMockProvider,
)
from smythe import Swarm, Task
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.memory import PlannerMemory
from smythe.planner import LLMArchitect, SimpleArchitect
from smythe.provider import CompletionResult, Provider
from smythe.router import WhiteRabbit


def test_swarm_construction():
    swarm = Swarm(
        max_budget_usd=1.00, model="claude-opus-4-8",
        provider=MockProvider(), architect=SimpleArchitect(),
    )
    assert swarm.model == "claude-opus-4-8"
    assert swarm.max_budget_usd == 1.00


def test_plan_returns_assigned_graph():
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Do something")
    graph = swarm.plan(task)

    assert isinstance(graph, ExecutionGraph)
    assert graph.topology == [Topology.SERIAL]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].agent_id is not None


def test_execute_with_task():
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Do something")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None
    assert len(result.trace) == 1


def test_execute_with_graph():
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Do something")
    graph = swarm.plan(task)
    result = swarm.execute(graph)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_execute_parallel():
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect(), parallel=True)
    task = Task(goal="Do something in parallel")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_model_stamped_on_nodes():
    swarm = Swarm(model="claude-opus-4-8", provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Check model stamping")
    graph = swarm.plan(task)

    assert graph.nodes[0].metadata["model"] == "claude-opus-4-8"


def test_execute_reports_cost():
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect(), max_budget_usd=10.0)
    task = Task(goal="Check cost tracking")
    result = swarm.execute(task)

    assert result.total_cost_usd > 0


def test_swarm_auto_constructs_llm_planner():
    swarm = Swarm(provider=MockProvider())
    assert isinstance(swarm._architect, LLMArchitect)


def test_swarm_with_llm_planner_plans_and_executes():
    provider = PlanningMockProvider()
    swarm = Swarm(provider=provider)
    task = Task(goal="Do the thing")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_swarm_records_outcome_to_memory():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)

    try:
        memory = PlannerMemory(path=path)
        swarm = Swarm(
            provider=MockProvider(), architect=SimpleArchitect(),
            memory=memory,
        )
        task = Task(goal="Track this execution")
        swarm.execute(task)

        outcomes = memory.recall(Task(goal="Track this execution"))
        assert len(outcomes) == 1
        assert outcomes[0].success is True
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_planning_model_separate_from_execution_model():
    provider = PlanningMockProvider()
    swarm = Swarm(
        provider=provider,
        model="claude-opus-4-8",
        planning_model="claude-planning-model",
    )
    assert isinstance(swarm._architect, LLMArchitect)
    assert swarm._architect._planning_model == "claude-planning-model"


def test_execute_parallel_with_llm_planner():
    """parallel=True with an LLMArchitect must not crash from nested event loops."""
    provider = PlanningMockProvider()
    swarm = Swarm(provider=provider, parallel=True)
    task = Task(goal="Do the thing")
    result = swarm.execute(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_from_yaml_execute_no_args():
    """Swarm.from_yaml() should allow execute() with no arguments."""
    import tempfile

    yaml_content = """\
topology: serial
nodes:
  - id: step-1
    label: "Do something"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        path = f.name

    try:
        swarm = Swarm.from_yaml(path, provider=MockProvider())
        result = swarm.execute()

        assert result.output != ""
        assert result.graph.nodes[0].result is not None
    finally:
        os.unlink(path)


def test_execute_no_args_without_yaml_raises():
    """execute() with no args and no YAML graph should raise ValueError."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    with pytest.raises(ValueError, match="No task or graph provided"):
        swarm.execute()


# --- Router tests ---


def test_router_explicit_deterministic():
    """Router selects a deterministic planner by key."""
    site_planner = FixedArchitect("site")
    router = WhiteRabbit(
        deterministic={"site-builder": site_planner},
        autonomous=SimpleArchitect(),
        classifier_provider=ClassifierMockProvider("deterministic:site-builder"),
        classifier_model="test",
    )
    swarm = Swarm(provider=MockProvider(), router=router)
    task = Task(goal="Build a site")
    graph = swarm.plan(task)

    assert "site:" in graph.nodes[0].label


def test_router_classifier_constrained():
    """Router routes to constrained planner when classifier says so."""
    constrained = FixedArchitect("constrained")
    router = WhiteRabbit(
        constrained=constrained,
        autonomous=SimpleArchitect(),
        classifier_provider=ClassifierMockProvider("constrained"),
        classifier_model="test",
    )
    swarm = Swarm(provider=MockProvider(), router=router)
    task = Task(goal="Guided task")
    graph = swarm.plan(task)

    assert "constrained:" in graph.nodes[0].label


def test_router_fallback_to_autonomous():
    """Router falls back to autonomous when no classifier is set."""
    autonomous = FixedArchitect("autonomous")
    router = WhiteRabbit(autonomous=autonomous)
    swarm = Swarm(provider=MockProvider(), router=router)
    task = Task(goal="Ambiguous task")
    graph = swarm.plan(task)

    assert "autonomous:" in graph.nodes[0].label


@pytest.mark.asyncio
async def test_swarm_aplan_async():
    """aplan() should work from inside an async context."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Async plan")
    graph = await swarm.aplan(task)

    assert isinstance(graph, ExecutionGraph)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].agent_id is not None


@pytest.mark.asyncio
async def test_execute_async_directly():
    """execute_async() should plan and execute in a fully async path."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    task = Task(goal="Async execute")
    result = await swarm.execute_async(task)

    assert result.output != ""
    assert result.graph.nodes[0].result is not None


def test_router_unknown_classification_falls_back():
    """Unknown classifier output falls back to autonomous."""
    autonomous = FixedArchitect("fallback")
    router = WhiteRabbit(
        autonomous=autonomous,
        classifier_provider=ClassifierMockProvider("something-weird"),
        classifier_model="test",
    )
    swarm = Swarm(provider=MockProvider(), router=router)
    task = Task(goal="Unknown route")
    graph = swarm.plan(task)

    assert "fallback:" in graph.nodes[0].label


# --- Synthesizer model passthrough tests ---


class _ModelCapturingProvider(Provider):
    """Records the model string passed to each complete() call."""

    def __init__(self) -> None:
        self.models_seen: list[str] = []

    async def complete(self, system, prompt, model):
        self.models_seen.append(model)
        return CompletionResult(text="merged", prompt_tokens=5, completion_tokens=5)


def test_llm_merge_receives_swarm_model():
    """LLM_MERGE synthesis should receive the swarm's model, not an empty string."""
    from smythe.synthesizer import Synthesizer, SynthesisStrategy

    capture = _ModelCapturingProvider()
    swarm = Swarm(
        model="claude-opus-4-8",
        provider=capture,
        architect=SimpleArchitect(),
        synthesizer=Synthesizer(strategy=SynthesisStrategy.LLM_MERGE),
    )
    task = Task(goal="Test merge model")
    swarm.execute(task)

    merge_model = capture.models_seen[-1]
    assert merge_model == "claude-opus-4-8", (
        f"Expected synthesis to use 'claude-opus-4-8' but got {merge_model!r}"
    )


# --- Graph validation tests ---


def test_execute_validates_direct_graph():
    """Passing an invalid graph directly to execute() should raise ValueError."""
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect())
    bad_graph = ExecutionGraph(
        topology=[Topology.SERIAL],
        nodes=[Node(label="A", id="a", depends_on=["nonexistent"])],
    )
    with pytest.raises(ValueError, match="nonexistent"):
        swarm.execute(bad_graph)


def test_artifact_dirs_are_scoped_per_execution(tmp_path):
    """Two runs with identical node ids must not overwrite each other."""
    from smythe.graph import ExecutionGraph, Node, Topology
    from smythe.provider import OfflineProvider

    def make_graph():
        return ExecutionGraph(
            topology=[Topology.SERIAL], nodes=[Node(id="hero", label="make image")],
        )

    swarm = Swarm(
        provider=OfflineProvider(artifacts_per_call=1),
        model="demo", artifact_dir=tmp_path,
    )
    swarm.execute(make_graph())
    swarm.execute(make_graph())

    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 2
    for d in run_dirs:
        assert (d / "hero_00.png").exists()
