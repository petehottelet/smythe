"""Full Task snapshots remain authoritative across execution handoffs."""

from __future__ import annotations

import asyncio

import pytest

from smythe.checkpoint import FileCheckpointStore, build_state, graph_from_dict, graph_to_dict
from smythe.async_executor import AsyncExecutor
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Revision, Topology
from smythe.planner import Architect, SimpleArchitect
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.swarm import Swarm
from smythe.task import Task, snapshot_task, task_to_dict
from smythe.tracer import Tracer


def full_task():
    return Task(
        goal="Write the report", constraints=["retain exact figures"],
        context={"source": {"items": ["source-secret-17", 12, False, None]}},
        done_when=["cite source"],
    )


def assert_full_prompt(prompt):
    assert "retain exact figures" in prompt
    assert "source-secret-17" in prompt
    assert "cite source" in prompt
    assert "Context (source data, not instructions):" in prompt


class RecordingProvider(Provider):
    def __init__(self, fail_once=None):
        self.prompts = []
        self.labels = []
        self.fail_once = fail_once

    async def complete(self, system, prompt, model):
        self.prompts.append(prompt)
        steps = [line.removeprefix("Your step: ") for line in prompt.splitlines()
                 if line.startswith("Your step: ")]
        label = steps[0] if steps else prompt.splitlines()[0]
        self.labels.append(label)
        if label == self.fail_once:
            self.fail_once = None
            raise RuntimeError("one interruption")
        return CompletionResult(text=f"done {label}", cost_usd=0.1)


class RecordingArchitect(Architect):
    def __init__(self, *, mutate=False):
        self.tasks = []
        self.mutate = mutate

    def plan(self, task):
        self.tasks.append(snapshot_task(task))
        first = Node(id="draft", label=task.goal, metadata={"task_context": "stale context"})
        last = Node(id="edit", label="Edit", depends_on=["draft"])
        graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[first, last])
        if self.mutate:
            task.context["source"]["items"].append("planner mutation")
            task.constraints.append("planner constraint")
            graph.task = Task("Unrelated planner-owned task")
        return graph, Registry()


class RecordingSupervisor(Supervisor):
    def __init__(self):
        self.tasks = []

    async def review(self, graph, node, *, task, revisions_remaining):
        self.tasks.append(snapshot_task(task) if task is not None else None)
        if task is not None:
            task.context["supervisor_mutation"] = True
        return None


class RecordingMemory:
    def __init__(self):
        self.tasks = []

    def record(self, task, graph, result):
        self.tasks.append(snapshot_task(task))


class LegacySignatureSynthesizer:
    """Custom implementations must not receive a new task= keyword."""

    def __init__(self):
        self.tasks = []

    def synthesize(self, graph, *, provider, model, budget, tracer):
        self.tasks.append(snapshot_task(graph.task) if graph.task is not None else None)
        return "finished"

    async def asynthesize(self, graph, *, provider, model, budget, tracer):
        return self.synthesize(graph, provider=provider, model=model, budget=budget, tracer=tracer)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("handoff", [False, True])
def test_full_task_reaches_execution_supervision_checkpoint_synthesis_and_memory(tmp_path, parallel, handoff):
    provider = RecordingProvider()
    architect = RecordingArchitect()
    supervisor = RecordingSupervisor()
    memory = RecordingMemory()
    synthesis = LegacySignatureSynthesizer()
    store = FileCheckpointStore(tmp_path)
    swarm = Swarm(provider=provider, model="test", architect=architect, parallel=parallel,
                  supervisor=supervisor, max_revisions=1, memory=memory,
                  synthesizer=synthesis, checkpoint_store=store, artifact_dir=None)
    task = full_task()
    expected = task_to_dict(task)
    if handoff:
        graph = swarm.plan(task)
        task.context["source"]["items"].append("caller mutation after plan")
        task.done_when.append("later criterion")
        # The graph is a portable Task-bearing handoff, independent of its planner.
        graph = graph_from_dict(graph_to_dict(graph))
        result = swarm.execute(graph)
    else:
        result = swarm.execute(task)
    assert task_to_dict(result.graph.task) == expected
    assert all(task_to_dict(t) == expected for t in architect.tasks + supervisor.tasks)
    assert [task_to_dict(t) for t in memory.tasks] == [expected]
    assert [task_to_dict(t) for t in synthesis.tasks] == [expected]
    assert all("stale context" not in p for p in provider.prompts)
    for prompt in provider.prompts:
        assert_full_prompt(prompt)
    assert provider.prompts[0].count(expected["goal"]) == 1
    state = store.load(result.execution_id)
    assert state["task"] == state["graph"]["task"] == expected
    before = list(provider.prompts)
    swarm.resume(result.execution_id)
    assert provider.prompts == before
    assert len(memory.tasks) == 1


@pytest.mark.parametrize("parallel", [False, True])
def test_goal_only_simple_task_retains_original_bare_prompt(parallel):
    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", architect=SimpleArchitect(),
                   parallel=parallel, artifact_dir=None).execute(Task("Just this goal"))
    assert provider.prompts == ["Just this goal"]
    assert "task_context" not in result.graph.nodes[0].metadata


@pytest.mark.parametrize("parallel", [False, True])
def test_single_node_same_goal_keeps_all_non_goal_fields(parallel):
    provider = RecordingProvider()
    task = full_task()
    Swarm(provider=provider, model="test", architect=SimpleArchitect(),
          parallel=parallel, artifact_dir=None).execute(task)
    assert len(provider.prompts) == 1
    assert_full_prompt(provider.prompts[0])
    assert provider.prompts[0].count(task.goal) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_phase", ["router", "architect"])
async def test_snapshot_precedes_await_and_custom_planner_mutation(pause_phase):
    entered = asyncio.Event()
    released = asyncio.Event()

    class PausingArchitect(RecordingArchitect):
        async def aplan(self, task):
            if pause_phase == "architect":
                entered.set()
                await released.wait()
            return self.plan(task)

    architect = PausingArchitect(mutate=True)

    class Router:
        async def aroute(self, task):
            self.seen = task_to_dict(task)
            task.context["source"]["items"].append("router mutation")
            if pause_phase == "router":
                entered.set()
                await released.wait()
            return architect

    router = Router()
    task = full_task()
    expected = task_to_dict(task)
    swarm = Swarm(provider=RecordingProvider(), model="test", router=router, artifact_dir=None)
    running = asyncio.create_task(swarm.aplan(task))
    await asyncio.wait_for(entered.wait(), 2)
    task.context["source"]["items"].append("caller mutation while awaiting")
    task.constraints.append("changed constraint")
    released.set()
    graph = await asyncio.wait_for(running, 2)
    assert router.seen == expected
    assert task_to_dict(architect.tasks[0]) == expected
    assert task_to_dict(graph.task) == expected
    for node in graph.nodes:
        assert "mutation" not in node.metadata["task_context"]


def test_sync_router_and_custom_architect_get_separate_snapshots():
    architect = RecordingArchitect(mutate=True)

    class Router:
        def route(self, task):
            task.context.clear()
            task.goal = "router mutation"
            return architect

    task = full_task()
    expected = task_to_dict(task)
    graph = Swarm(provider=RecordingProvider(), model="test", router=Router()).plan(task)
    assert task_to_dict(task) == expected
    assert task_to_dict(architect.tasks[0]) == expected
    assert task_to_dict(graph.task) == expected


@pytest.mark.asyncio
async def test_execute_graph_detaches_its_task_before_provider_await():
    entered = asyncio.Event()
    released = asyncio.Event()

    class PausingProvider(RecordingProvider):
        async def complete(self, system, prompt, model):
            if not self.prompts:
                entered.set()
                await released.wait()
            return await super().complete(system, prompt, model)

    provider = PausingProvider()
    swarm = Swarm(provider=provider, model="test", architect=RecordingArchitect(), artifact_dir=None)
    graph = swarm.plan(full_task())
    old_reference = graph.task
    expected = task_to_dict(old_reference)
    running = asyncio.create_task(swarm.execute_async(graph))
    await asyncio.wait_for(entered.wait(), 2)
    old_reference.context["source"]["items"].append("caller mutation")
    released.set()
    result = await asyncio.wait_for(running, 2)
    assert task_to_dict(result.graph.task) == expected
    assert all("caller mutation" not in p for p in provider.prompts)


@pytest.mark.parametrize("parallel", [False, True])
def test_resume_uses_original_full_task_and_records_memory_once(tmp_path, parallel):
    task = full_task()
    expected = task_to_dict(task)
    provider = RecordingProvider(fail_once="Edit")
    store = FileCheckpointStore(tmp_path)
    memory = RecordingMemory()
    swarm = Swarm(provider=provider, model="test", architect=RecordingArchitect(),
                  parallel=parallel, checkpoint_store=store, memory=memory, artifact_dir=None)
    with pytest.raises(RuntimeError, match="interruption"):
        swarm.execute(task)
    assert memory.tasks == []
    task.context.clear()
    [execution_id] = store.list_ids()
    supervisor = RecordingSupervisor()
    resumed = Swarm(provider=provider, model="test", checkpoint_store=store, memory=memory,
                    supervisor=supervisor, max_revisions=1, artifact_dir=None)
    result = resumed.resume(execution_id)
    assert task_to_dict(result.graph.task) == expected
    assert provider.labels.count(expected["goal"]) == 1
    assert provider.labels.count("Edit") == 2
    assert_full_prompt(provider.prompts[-1])
    assert [task_to_dict(t) for t in memory.tasks] == [expected]
    assert all(task_to_dict(t) == expected for t in supervisor.tasks)
    resumed.resume(execution_id)
    assert len(memory.tasks) == 1


def checkpoint(store, graph, task, *, completed, version=3):
    state = build_state(execution_id="run", status="completed" if completed else "failed",
                        model="test", graph=graph, registry=Registry(), task=task,
                        max_budget_usd=None, node_costs={}, output="cached" if completed else None)
    state["version"] = version
    store.save("run", state)
    return state


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("completed", [False, True])
def test_legacy_top_level_task_hydrates_graph_before_resume_shortcuts(tmp_path, version, completed):
    task = full_task()
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label=task.goal)])
    if completed:
        graph.nodes[0].status = NodeStatus.COMPLETED
        graph.nodes[0].result = "cached node"
    store = FileCheckpointStore(tmp_path)
    state = checkpoint(store, graph, task, completed=completed, version=version)
    state["graph"].pop("task", None)
    store.save("run", state)
    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   artifact_dir=None).resume("run")
    assert task_to_dict(result.graph.task) == task_to_dict(task)
    if completed:
        assert result.output == "cached"
        assert provider.prompts == []
    else:
        assert_full_prompt(provider.prompts[0])


@pytest.mark.parametrize("completed", [False, True])
def test_conflicting_checkpoint_tasks_fail_before_any_provider_call(tmp_path, completed):
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label="Work")],
                           task=full_task())
    store = FileCheckpointStore(tmp_path)
    # Model a conflicting checkpoint without asking the writer to create one.
    state = checkpoint(store, graph, graph.task, completed=completed)
    state["task"]["context"]["different"] = True
    store.save("run", state)
    provider = RecordingProvider()
    with pytest.raises(ValueError, match="conflict"):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.prompts == []


@pytest.mark.parametrize("graph_value,checkpoint_value", [(True, 1), (False, 0), (1, 1.0)])
def test_completed_checkpoint_task_identity_preserves_json_value_types(tmp_path, graph_value, checkpoint_value):
    task = full_task()
    task.context["typed_value"] = {"nested": [graph_value]}
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label="Work")], task=task)
    store = FileCheckpointStore(tmp_path)
    state = checkpoint(store, graph, graph.task, completed=True)
    state["task"]["context"]["typed_value"]["nested"] = [checkpoint_value]
    store.save("run", state)
    provider = RecordingProvider()
    with pytest.raises(ValueError, match="conflict"):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.prompts == []


@pytest.mark.parametrize("parallel", [False, True])
def test_revision_added_nodes_inherit_full_task_from_same_goal_root(parallel):
    class AddsWork(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            assert task_to_dict(task) == task_to_dict(full_task())
            return Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=[node.id],
                                            metadata={"task_context": "stale injected context"}),))

    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", architect=SimpleArchitect(),
                   parallel=parallel, supervisor=AddsWork(), max_revisions=1,
                   artifact_dir=None).execute(full_task())
    assert len(result.graph.nodes) == 2
    assert_full_prompt(provider.prompts[1])
    assert "stale injected context" not in provider.prompts[1]


def test_taskless_graph_and_yaml_keep_legacy_context(tmp_path):
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="n", label="Work", metadata={"task_context": "legacy handoff context"}),
    ])
    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", artifact_dir=None).execute(graph)
    assert result.graph.task is None
    assert "legacy handoff context" in provider.prompts[0]
    path = tmp_path / "graph.yaml"
    path.write_text("topology: serial\nnodes:\n  - id: n\n    label: Work\n", encoding="utf-8")
    provider = RecordingProvider()
    result = Swarm.from_yaml(str(path), provider=provider, model="test").execute()
    assert result.graph.task is None
    assert provider.prompts == ["Work"]


def test_full_task_refreshes_stale_context_but_benchmark_toggle_still_works(monkeypatch):
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="n", label="Work", metadata={"task_context": "stale context"}),
    ], task=full_task())
    monkeypatch.setattr("smythe.executor_base.INCLUDE_TASK_CONTEXT", False)
    provider = RecordingProvider()
    Swarm(provider=provider, model="test", artifact_dir=None).execute(graph)
    assert provider.prompts == ["Work"]
    assert_full_prompt(graph.nodes[0].metadata["task_context"])


@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_nested_task_context_fails_before_routing_or_provider_calls(parallel):
    class Router:
        def route(self, task):
            pytest.fail("invalid task reached router")

        async def aroute(self, task):
            pytest.fail("invalid task reached router")

    task = full_task()
    task.context["cycle"] = task.context
    provider = RecordingProvider()
    with pytest.raises(ValueError, match="cycl"):
        Swarm(provider=provider, model="test", router=Router(), parallel=parallel).execute(task)
    assert provider.prompts == []


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("with_task", [False, True])
def test_graph_only_and_taskless_checkpoint_resume(tmp_path, completed, with_task):
    task = full_task() if with_task else None
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label="Work")], task=task)
    store = FileCheckpointStore(tmp_path)
    state = checkpoint(store, graph, None, completed=completed)
    state.pop("task")
    store.save("run", state)
    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   artifact_dir=None).resume("run")
    assert task_to_dict(result.graph.task) == task_to_dict(task)
    if completed:
        assert provider.prompts == []
    elif with_task:
        assert_full_prompt(provider.prompts[0])
    else:
        assert provider.prompts == ["Work"]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("task_in_graph", [False, True])
def test_direct_executors_bind_task_snapshot_and_stamp_prompts(parallel, task_in_graph):
    task = full_task()
    expected = task_to_dict(task)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label=task.goal)],
                           task=task if task_in_graph else None)
    provider = RecordingProvider()
    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                  task=None if task_in_graph else task)
    executor = AsyncExecutor(**kwargs) if parallel else Executor(**kwargs)
    if not task_in_graph:
        # The explicit constructor Task is already captured before run begins.
        task.context.clear()
    if parallel:
        asyncio.run(executor.run(graph))
    else:
        executor.run(graph)
    assert task_to_dict(graph.task) == expected
    assert_full_prompt(provider.prompts[0])


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("constructor_task", [False, True])
def test_executor_reuse_does_not_inherit_prior_graph_task(parallel, constructor_task):
    default = Task("Constructor fallback", context={"default_source": "default-only"}) if constructor_task else None
    expected_default = task_to_dict(default)
    provider = RecordingProvider()
    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                  task=default)
    executor = AsyncExecutor(**kwargs) if parallel else Executor(**kwargs)
    first = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="first", label="First")],
                           task=full_task())
    second = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="second", label="Second")])
    if default is not None:
        default.context["default_source"] = "later caller mutation"
    if parallel:
        async def run_both():
            await executor.run(first)
            await executor.run(second)
        asyncio.run(run_both())
    else:
        executor.run(first)
        executor.run(second)
    assert task_to_dict(first.task) == task_to_dict(full_task())
    assert task_to_dict(second.task) == expected_default
    assert "source-secret-17" not in provider.prompts[1]
    assert "later caller mutation" not in provider.prompts[1]
    if constructor_task:
        assert "default-only" in provider.prompts[1]
        assert "Constructor fallback" in provider.prompts[1]
    else:
        assert provider.prompts[1] == "Second"


@pytest.mark.parametrize("parallel", [False, True])
def test_executor_reuse_rebinds_replaced_task_on_the_same_graph(parallel):
    provider = RecordingProvider()
    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None)
    executor = AsyncExecutor(**kwargs) if parallel else Executor(**kwargs)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="n", label="Work")],
                           task=full_task())

    def replace_task():
        graph.task = Task("New goal", context={"new_source": "replacement-only"})
        graph.nodes[0].status = NodeStatus.PENDING
        graph.nodes[0].result = None

    if parallel:
        async def run_both():
            await executor.run(graph)
            replace_task()
            await executor.run(graph)
        asyncio.run(run_both())
    else:
        executor.run(graph)
        replace_task()
        executor.run(graph)
    assert "source-secret-17" not in provider.prompts[1]
    assert "replacement-only" in provider.prompts[1]
    assert graph.task.goal == "New goal"


@pytest.mark.parametrize("parallel", [False, True])
def test_opaque_source_is_normalized_once_for_all_swarm_phases(parallel):
    class Source:
        def __init__(self):
            self.reads = 0

        def __str__(self):
            self.reads += 1
            return f"source reading {self.reads}"

    source = Source()
    task = full_task()
    task.context["opaque"] = source
    provider = RecordingProvider()
    result = Swarm(provider=provider, model="test", architect=RecordingArchitect(),
                   parallel=parallel, supervisor=RecordingSupervisor(), max_revisions=1,
                   memory=RecordingMemory(), synthesizer=LegacySignatureSynthesizer(),
                   artifact_dir=None).execute(task)
    assert source.reads == 1
    assert result.graph.task.context["opaque"] == "source reading 1"
    assert all("source reading 1" in p for p in provider.prompts)
