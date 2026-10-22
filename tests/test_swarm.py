"""Tests for the Swarm top-level orchestrator."""

import asyncio
import os
import tempfile

import pytest

from fakes import ManualClock
from helpers import (
    SIMPLE_PLAN_JSON,
    ClassifierMockProvider,
    FixedArchitect,
    MockProvider,
    PlanningMockProvider,
)
from smythe import Swarm, Task
from smythe.checkpoint import FileCheckpointStore
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.memory import PlannerMemory
from smythe.planner import LLMArchitect, SimpleArchitect
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult, Provider
from smythe.router import WhiteRabbit
from smythe.swarm import SwarmResult
from smythe.tracer import Tracer


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


def test_planning_provider_separates_planning_from_execution():
    """Planning is structured work; it need not use the executor's model
    or even its vendor."""
    from smythe.provider import CompletionResult, Provider

    class Recorder(Provider):
        def __init__(self, name, text='{"topology":["serial"],"nodes":['
                                      '{"id":"a","label":"Do it","depends_on":[],'
                                      '"agent":{"name":"A","persona":"p"}}]}'):
            self.name = name
            self.models: list[str] = []
            self._text = text

        async def complete(self, system, prompt, model):
            self.models.append(model)
            return CompletionResult(text=self._text, prompt_tokens=1,
                                    completion_tokens=1)

    planner = Recorder("planner")
    executor = Recorder("executor", text="executed")
    swarm = Swarm(
        provider=executor,
        planning_provider=planner,
        planning_model="cheap-planner",
        model="expensive-executor",
        artifact_dir=None,
    )
    result = swarm.execute(Task(goal="Do the thing"))

    assert planner.models == ["cheap-planner"], "planning used the wrong provider"
    assert executor.models == ["expensive-executor"]
    assert result.output == "executed"


def test_planning_defaults_to_the_execution_provider():
    from smythe.provider import CompletionResult, Provider

    class Counting(Provider):
        def __init__(self):
            self.calls = 0

        async def complete(self, system, prompt, model):
            self.calls += 1
            return CompletionResult(
                text='{"topology":["serial"],"nodes":[{"id":"a","label":"x",'
                     '"depends_on":[],"agent":{"name":"A","persona":"p"}}]}',
                prompt_tokens=1, completion_tokens=1,
            )

    provider = Counting()
    Swarm(provider=provider, model="m", artifact_dir=None).execute(Task(goal="g"))
    assert provider.calls >= 2, "one provider should serve planning and execution"


# ---------------------------------------------------------------------------
# Acceptance criteria reach the work
# ---------------------------------------------------------------------------


def test_done_when_reaches_every_node_prompt():
    """A node that cannot see the acceptance criteria cannot meet them.

    ``done_when`` was previously visible only to LLMSupervisor, which is
    off by default -- so in the default configuration it did nothing.
    """
    task = Task(
        goal="Write the memo",
        done_when=["cites at least three sources", "under 500 words"],
    )
    graph = ExecutionGraph(
        topology=[Topology.SERIAL],
        nodes=[Node(id="a", label="Draft"), Node(id="b", label="Edit", depends_on=["a"])],
    )

    Swarm._stamp_task_context(graph, task)

    context = graph.nodes[0].metadata["task_context"]
    assert "cites at least three sources" in context
    assert "under 500 words" in context


# ---------------------------------------------------------------------------
# parallel=False means one node at a time on every path, including resume
# ---------------------------------------------------------------------------


class _ConcurrencyProbe(Provider):
    """Records the peak number of provider calls in flight."""

    def __init__(self, crash_on: str | None = None) -> None:
        self.active = 0
        self.peak = 0
        self.crash_on = crash_on

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        if label == self.crash_on:
            raise RuntimeError("process died")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.active -= 1
        return CompletionResult(text=f"done: {label}", prompt_tokens=1, completion_tokens=1)


def _independent_graph():
    return ExecutionGraph(
        topology=[Topology.BROADCAST_REDUCE],
        nodes=[Node(id="gate", label="gate")]
        + [Node(id=f"w{i}", label=f"w{i}") for i in range(4)],
    )


@pytest.mark.parametrize("parallel,peak", [(False, 1), (True, 5)])
def test_resume_honors_parallel_setting(tmp_path, parallel, peak):
    from smythe.checkpoint import FileCheckpointStore

    store = FileCheckpointStore(tmp_path)
    crashed = Swarm(model="m", provider=_ConcurrencyProbe(crash_on="gate"),
                    checkpoint_store=store, artifact_dir=None)
    with pytest.raises(RuntimeError, match="process died"):
        crashed.execute(_independent_graph())
    [execution_id] = store.list_ids()

    probe = _ConcurrencyProbe()
    result = Swarm(model="m", provider=probe, parallel=parallel, max_concurrency=8,
                   checkpoint_store=store, artifact_dir=None).resume(execution_id)

    assert all(node.status.value == "completed" for node in result.graph.nodes)
    assert probe.peak == peak


@pytest.mark.parametrize("parallel,peak", [(False, 1), (True, 5)])
def test_execute_async_honors_parallel_setting(parallel, peak):
    probe = _ConcurrencyProbe()
    swarm = Swarm(model="m", provider=probe, parallel=parallel, max_concurrency=8,
                  artifact_dir=None)

    asyncio.run(swarm.execute_async(_independent_graph()))

    assert probe.peak == peak


@pytest.mark.parametrize("parallel,expected", [(False, 1), (True, 3)])
def test_durable_paths_record_the_parallel_concurrency(tmp_path, parallel, expected):
    from smythe import OfflineProvider, SQLiteWorkflowStore

    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        def swarm():
            return Swarm(model="offline", provider=OfflineProvider(),
                         architect=SimpleArchitect(), run_store=store, max_budget_usd=1,
                         parallel=parallel, max_concurrency=3)

        def saved(run_id):
            return store.get_checkpoint(run_id)["checkpoint"]["max_concurrency"]

        executed = swarm().execute(_independent_graph())
        awaited = asyncio.run(swarm().execute_async(_independent_graph()))
        planned = swarm().plan(Task(goal="Report"))
        resumed = swarm().resume(planned.run_ref["run_id"])

        for result in (executed, awaited, resumed):
            assert saved(result.execution_id) == expected


# ---------------------------------------------------------------------------
# Execution policy is validated before any routing or planning call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy,match", [
    ({"max_concurrency": 0}, "max_concurrency"),
    ({"max_concurrency": -1}, "max_concurrency"),
    ({"max_revisions": -1}, "max_revisions"),
    ({"retry_backoff_s": -1}, "retry_backoff_s"),
    ({"retry_backoff_s": float("nan")}, "retry_backoff_s"),
    ({"retry_backoff_s": float("inf")}, "retry_backoff_s"),
    ({"retry_backoff_s": True}, "retry_backoff_s"),
])
def test_invalid_execution_policy_is_rejected_before_routing_or_planning(policy, match):
    """These values used to fail only when the executor was built, after the
    router's classifier and the planner had already been paid."""
    classifier = ClassifierMockProvider("autonomous")
    planner = PlanningMockProvider()
    router = WhiteRabbit(
        autonomous=LLMArchitect(provider=planner, planning_model="planner"),
        classifier_provider=classifier, classifier_model="classifier",
    )

    with pytest.raises(ValueError, match=match):
        Swarm(model="m", provider=planner, router=router, parallel=True,
              artifact_dir=None, **policy).execute(Task(goal="Plan this"))

    assert classifier.calls == []
    assert planner.calls == []


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("policy", [
    {},
    {"max_concurrency": None},
    {"max_concurrency": 1},
    {"retry_backoff_s": 0},
    {"retry_backoff_s": 1.5},
    {"max_revisions": 2},
])
def test_valid_execution_policies_still_run(policy, parallel):
    swarm = Swarm(provider=MockProvider(), architect=SimpleArchitect(),
                  parallel=parallel, artifact_dir=None, **policy)

    result = swarm.execute(Task(goal="Do something"))

    assert result.graph.nodes[0].result is not None


# ---------------------------------------------------------------------------
# Elapsed time, phases and per-process timing segments
# ---------------------------------------------------------------------------


def _clocked(swarm: Swarm, clock: ManualClock) -> Swarm:
    """Measure this Swarm's calls with a manual clock."""
    swarm._tracer_factory = lambda: Tracer(
        clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9,
    )
    return swarm


class _ClockedProvider(Provider):
    """Planning takes 2s, classification 0.5s and every node call 1s."""

    def __init__(self, clock: ManualClock, crash_on: str | None = None) -> None:
        self.clock = clock
        self.crash_on = crash_on

    async def complete(self, system, prompt, model):
        if system == PLANNING_SYSTEM_PROMPT:
            self.clock.advance(2)
            return CompletionResult(text=SIMPLE_PLAN_JSON, prompt_tokens=1, completion_tokens=1)
        if system.startswith("You are a task classifier."):
            self.clock.advance(0.5)
            return CompletionResult(text="autonomous", prompt_tokens=1, completion_tokens=1)
        label = prompt.splitlines()[0]
        self.clock.advance(1)
        if label == self.crash_on:
            raise RuntimeError("process died")
        return CompletionResult(text=f"done: {label}", prompt_tokens=1, completion_tokens=1)


def _chain() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="A"), Node(id="b", label="B", depends_on=["a"]),
        Node(id="c", label="C", depends_on=["b"]),
    ])


def test_swarm_result_timing_defaults_to_none_for_positional_construction():
    graph = _chain()
    assert SwarmResult("done", graph).timing is None


@pytest.mark.parametrize("parallel", [False, True])
def test_execute_task_reports_elapsed_time_phases_and_one_segment(tmp_path, parallel):
    clock = ManualClock()
    store = FileCheckpointStore(tmp_path)
    swarm = _clocked(Swarm(model="m", provider=_ClockedProvider(clock), parallel=parallel,
                           checkpoint_store=store, artifact_dir=None), clock)

    result = swarm.execute(Task(goal="Do the thing"))

    timing = result.timing
    assert timing["clock"] == "monotonic"
    assert timing["elapsed_ms"] == 3000
    assert timing["summed_node_ms"] == 1000
    assert timing["phases_ms"] == {
        "routing": None, "planning": 2000.0, "recovery": None, "execution": 1000.0,
        "verification": 0.0, "synthesis": 0.0, "supervision": 0.0, "checkpoint": 0.0,
        "waiting": 0.0,
    }
    assert (timing["latency_complete"], timing["resumed"]) == (True, False)
    [segment] = timing["segments"]
    assert segment["started_at"] == 1000.0
    assert segment["ended_at"] == 1003.0
    assert segment["elapsed_ms"] == 3000
    assert segment["resumed"] is False
    saved = store.load(result.execution_id)
    assert saved["version"] == 4
    assert saved["timing"] == {"version": 1, "clock": "monotonic", "segments": [segment]}


def test_router_time_is_reported_when_execute_routes_in_the_same_call():
    clock = ManualClock()
    provider = _ClockedProvider(clock)
    router = WhiteRabbit(
        autonomous=LLMArchitect(provider=provider, planning_model="planner"),
        classifier_provider=provider, classifier_model="classifier",
    )
    swarm = _clocked(Swarm(model="m", provider=provider, router=router, artifact_dir=None), clock)

    timing = swarm.execute(Task(goal="Do the thing")).timing

    assert timing["phases_ms"]["routing"] == 500
    assert timing["phases_ms"]["planning"] == 2000
    assert timing["elapsed_ms"] == 3500


@pytest.mark.parametrize("parallel", [False, True])
def test_execute_plans_a_task_through_an_overridden_plan_method(parallel):
    class ReviewedSwarm(Swarm):
        def plan(self, task):
            self.planned.append(task.goal)
            graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="reviewed", label="Reviewed step")])
            graph.task = task
            return graph

        async def aplan(self, task):
            return self.plan(task)

    swarm = ReviewedSwarm(model="m", provider=MockProvider(), parallel=parallel, artifact_dir=None)
    swarm.planned = []

    result = swarm.execute(Task(goal="Do the thing"))

    assert swarm.planned == ["Do the thing"]
    assert [node.id for node in result.graph.nodes] == ["reviewed"]
    assert result.timing["phases_ms"]["planning"] is not None


@pytest.mark.parametrize("parallel", [False, True])
def test_a_graph_planned_in_another_call_reports_planning_as_not_measured(parallel):
    clock = ManualClock()
    swarm = _clocked(Swarm(model="m", provider=_ClockedProvider(clock), parallel=parallel,
                           artifact_dir=None), clock)
    graph = swarm.plan(Task(goal="Do the thing"))
    clock.advance(60)

    timing = swarm.execute(graph).timing

    assert (timing["phases_ms"]["routing"], timing["phases_ms"]["planning"]) == (None, None)
    assert timing["elapsed_ms"] == 1000


def test_checkpoint_writes_are_measured_as_checkpoint_time(tmp_path):
    clock = ManualClock()

    class SlowStore(FileCheckpointStore):
        def save(self, execution_id, state):
            clock.advance(0.25)
            super().save(execution_id, state)

    store = SlowStore(tmp_path)
    swarm = _clocked(Swarm(model="m", provider=_ClockedProvider(clock), parallel=True,
                           checkpoint_store=store, artifact_dir=None), clock)

    result = swarm.execute(_chain())

    # The initial save, one per node, and the completed save.
    assert result.timing["phases_ms"]["checkpoint"] == 1250
    assert result.timing["elapsed_ms"] == 4250


def _crash_then_resume(tmp_path, *, parallel=True):
    store = FileCheckpointStore(tmp_path)
    first = ManualClock(start_ns=1_000 * 10**9, monotonic_ns=7_000_000 * 10**9)
    crashed = _clocked(Swarm(model="m", provider=_ClockedProvider(first, crash_on="B"),
                             parallel=parallel, checkpoint_store=store, artifact_dir=None), first)
    with pytest.raises(RuntimeError, match="process died"):
        crashed.execute(_chain())
    [execution_id] = store.list_ids()
    # A new process: a later wall clock, and a monotonic clock that starts
    # far below the first process's readings.
    second = ManualClock(start_ns=4_600 * 10**9, monotonic_ns=0)
    resumed = _clocked(Swarm(model="m", provider=_ClockedProvider(second), parallel=parallel,
                             checkpoint_store=store, artifact_dir=None), second)
    return store, execution_id, resumed


def test_resume_records_a_separate_segment_without_comparing_process_clocks(tmp_path):
    store, execution_id, resumed = _crash_then_resume(tmp_path)
    crashed_segments = store.load(execution_id)["timing"]["segments"]

    result = resumed.resume(execution_id)

    first, second = result.timing["segments"]
    assert first == crashed_segments[0]
    assert (first["started_at"], first["ended_at"], first["elapsed_ms"]) == (1000.0, 1002.0, 2000)
    assert first["resumed"] is False
    assert first["phases_ms"]["execution"] == 2000  # A, and B's failed attempt
    assert (second["started_at"], second["ended_at"], second["elapsed_ms"]) == (4600.0, 4602.0, 2000)
    assert second["resumed"] is True
    assert second["phases_ms"]["recovery"] == 0
    timing = result.timing
    assert timing["elapsed_ms"] == 2000
    assert (timing["latency_complete"], timing["resumed"]) == (False, True)
    assert store.load(execution_id)["timing"]["segments"] == [first, second]


def test_resuming_a_completed_checkpoint_reports_this_call_without_saving(tmp_path):
    store, execution_id, resumed = _crash_then_resume(tmp_path)
    resumed.resume(execution_id)
    saved = store.load(execution_id)

    again = resumed.resume(execution_id)

    assert again.timing["resumed"] is True
    assert again.timing["latency_complete"] is False
    assert again.timing["phases_ms"]["recovery"] == 0
    assert again.timing["segments"][:2] == saved["timing"]["segments"]
    assert len(again.timing["segments"]) == 3
    assert store.load(execution_id) == saved


@pytest.mark.parametrize("block", [
    "missing", None, "garbage", {"version": 2, "segments": []},
    {"version": 1, "segments": "garbage"}, {"version": 1, "segments": [1]},
])
def test_resume_starts_new_timing_when_the_saved_block_is_missing_or_unreadable(tmp_path, block):
    store, execution_id, resumed = _crash_then_resume(tmp_path)
    state = store.load(execution_id)
    if block == "missing":
        del state["timing"]
    else:
        state["timing"] = block
    store.save(execution_id, state)

    result = resumed.resume(execution_id)

    [segment] = result.timing["segments"]
    assert segment["resumed"] is True
    assert store.load(execution_id)["timing"]["segments"] == [segment]
