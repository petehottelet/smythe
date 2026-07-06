"""Tests for ExecutorBase shared helpers."""

from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class DummyProvider(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


def _make_base() -> ExecutorBase:
    return ExecutorBase(
        provider=DummyProvider(),
        registry=Registry(),
        tracer=Tracer(),
    )


def test_deps_satisfied_all_completed():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.COMPLETED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_skipped():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.SKIPPED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_failed_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.FAILED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_with_pending_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_no_deps():
    base = _make_base()
    a = Node(label="A", id="a")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    assert base.deps_satisfied(a, graph) is True


# --- terminal deliverable note -----------------------------------------

class PromptCapturingProvider(Provider):
    def __init__(self):
        self.prompts = []

    async def complete(self, system, prompt, model):
        self.prompts.append(prompt)
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


def test_terminal_node_prompt_carries_deliverable_note():
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    mid = Node(label="Mid", id="mid")
    prompt = ExecutorBase.build_user_prompt(mid, {"a": "r"}, is_terminal=False)
    assert TERMINAL_DELIVERABLE_NOTE not in prompt

    terminal = Node(label="Final", id="final")
    prompt = ExecutorBase.build_user_prompt(terminal, {"a": "r"}, is_terminal=True)
    assert TERMINAL_DELIVERABLE_NOTE in prompt

    # A terminal node with no upstream context needs no note.
    prompt = ExecutorBase.build_user_prompt(terminal, {}, is_terminal=True)
    assert TERMINAL_DELIVERABLE_NOTE not in prompt


def test_executor_marks_only_terminal_nodes():
    from smythe.executor import Executor
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    provider = PromptCapturingProvider()
    executor = Executor(provider=provider, registry=Registry(), tracer=Tracer())
    a = Node(label="Gather", id="a")
    b = Node(label="Deliver", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    executor.run(graph)

    first, last = provider.prompts
    assert TERMINAL_DELIVERABLE_NOTE not in first
    assert TERMINAL_DELIVERABLE_NOTE in last


# --- task context propagation ------------------------------------------

def test_root_node_prompt_carries_task_context():
    node = Node(label="Research the market", id="r")
    node.metadata["task_context"] = "Evaluate the widget market\n\nConstraints:\n- be brief"
    prompt = ExecutorBase.build_user_prompt(node, {})
    assert prompt.startswith("Overall task:\nEvaluate the widget market")
    assert "Your step: Research the market" in prompt


def test_task_context_respects_ablation_toggle(monkeypatch):
    import smythe.executor_base as eb

    node = Node(label="Research", id="r")
    node.metadata["task_context"] = "The big goal"
    monkeypatch.setattr(eb, "INCLUDE_TASK_CONTEXT", False)
    prompt = ExecutorBase.build_user_prompt(node, {})
    assert prompt == "Research"


def test_swarm_plan_stamps_task_context_on_roots_only():

    from smythe import Swarm, Task
    from smythe.provider import OfflineProvider

    plan = {
        "topology": ["fork_join"],
        "nodes": [
            {"id": "a", "label": "Angle A", "depends_on": [],
             "agent": {"name": "A", "persona": "p"}},
            {"id": "b", "label": "Angle B", "depends_on": [],
             "agent": {"name": "B", "persona": "p"}},
            {"id": "join", "label": "Merge", "depends_on": ["a", "b"],
             "agent": {"name": "J", "persona": "p"}},
        ],
    }
    swarm = Swarm(provider=OfflineProvider(plan=plan), model="test-model")
    task = Task(goal="Study the gizmo market", constraints=["stay brief"])
    graph = swarm.plan(task)

    by_id = {n.id: n for n in graph.nodes}
    assert "Study the gizmo market" in by_id["a"].metadata["task_context"]
    assert "- stay brief" in by_id["a"].metadata["task_context"]
    assert "task_context" in by_id["b"].metadata
    assert "task_context" not in by_id["join"].metadata


def test_single_node_graph_is_not_stamped():
    from smythe import Swarm, Task
    from smythe.planner import SimpleArchitect
    from smythe.provider import OfflineProvider

    swarm = Swarm(provider=OfflineProvider(), model="test-model",
                  architect=SimpleArchitect())
    graph = swarm.plan(Task(goal="Do the thing"))
    assert "task_context" not in graph.nodes[0].metadata
