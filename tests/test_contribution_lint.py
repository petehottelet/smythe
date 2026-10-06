"""The contribution lint: parallel siblings that could only repeat each other's work."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from fakes import Crash, crash_when
from helpers import MockProvider
from smythe import (
    FileCheckpointStore, LocalOnly, OfflineProvider, SQLiteWorkflowStore, Swarm, Task,
    lint_contributions,
)
from smythe import planner
from smythe.agent import Agent, AgentProfile
from smythe.checkpoint import graph_to_dict
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.mcp import MCPServerSpec
from smythe.planner import COORDINATION_RECORD_VERSION, DeterministicArchitect
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.router import WhiteRabbit
from smythe.workflow_tools import READ_STEP_KEY

EXAMPLE = Path(__file__).parents[1] / "examples" / "17_contribution_templates.py"
REPEATED = ["research-0", "research-1", "research-2"]


def fork(*nodes: Node) -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], list(nodes))


def sibling(node_id: str, **fields) -> Node:
    """A research step that reads the brief, as its siblings do."""
    values = {"label": "Research the vendor", "agent_id": "researcher", "depends_on": ["brief"]}
    return Node(id=node_id, **{**values, **fields})


def agent(name: str, persona: str = "", capabilities=(), servers=()) -> Agent:
    return Agent(AgentProfile(name, persona=persona, capabilities=list(capabilities),
                              mcp_servers=list(servers)), id=name)


def registry_of(*agents: Agent) -> Registry:
    registry = Registry()
    for item in agents:
        registry.register(item)
    return registry


# ---------------------------------------------------------------------------
# The lint
# ---------------------------------------------------------------------------


def test_identical_parallel_siblings_are_flagged_in_graph_order():
    graph = fork(
        Node(id="brief", label="Read the brief"),
        sibling("r1"),
        Node(id="price", label="Price the contract", depends_on=["brief"]),
        sibling("r2"),
        Node(id="d1", label="Draft the summary", depends_on=["brief"]),
        Node(id="merge", label="Merge", depends_on=["r1", "r2", "r3", "price", "d1", "d2"]),
        sibling("r3"),
        Node(id="d2", label="Draft the summary", depends_on=["brief"]),
    )

    assert lint_contributions(graph) == [["r1", "r2", "r3"], ["d1", "d2"]]


@pytest.mark.parametrize("change", [
    {"label": "Research the customer"},
    {"agent_id": "analyst"},
    {"required_capabilities": ["legal"]},
    {"depends_on": ["brief", "notes"]},
    {"metadata": {"model": "other-model"}},
    {"verifies": "notes"},
    {"metadata": {READ_STEP_KEY: {"profile": "web", "url": "https://example.test/b"}}},
    {"attach_dep_artifacts": True},
], ids=["label", "agent", "capabilities", "inputs", "model", "verification-target",
        "read-step", "image-inputs"])
def test_siblings_that_differ_in_one_contribution_pass(change):
    notes = Node(id="notes", label="Collect the notes")
    brief = Node(id="brief", label="Read the brief")

    assert lint_contributions(fork(brief, notes, sibling("a"), sibling("b"))) == [["a", "b"]]
    assert lint_contributions(fork(brief, notes, sibling("a"), sibling("b", **change))) == []


def test_dependency_and_capability_order_do_not_make_siblings_distinct():
    graph = fork(
        Node(id="x", label="X"), Node(id="y", label="Y"),
        sibling("a", depends_on=["x", "y"], required_capabilities=["research", "web"]),
        sibling("b", depends_on=["y", "x"], required_capabilities=["web", "research"]),
    )

    assert lint_contributions(graph) == [["a", "b"]]


def test_a_chain_of_identical_steps_is_not_parallel():
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="first", label="Revise the draft"),
        Node(id="second", label="Revise the draft", depends_on=["first"]),
    ])

    assert lint_contributions(graph) == []


def test_registry_agents_compare_by_what_they_give_a_worker():
    graph = fork(Node(id="a", label="Research the vendor"), Node(id="b", label="Research the vendor"))
    registry = Registry()
    registry.assign(graph)
    a, b = graph.nodes

    # The generalists a registry creates differ only in name and id.
    assert a.agent_id != b.agent_id
    assert lint_contributions(graph) == []
    assert lint_contributions(graph, registry) == [["a", "b"]]

    same = registry_of(agent("one", "You research vendors.", ["research"]),
                       agent("two", "You research vendors.", ["research"]))
    a.agent_id, b.agent_id = "one", "two"
    assert lint_contributions(graph, same) == [["a", "b"]]

    # Each agent differs from "one" in one field a worker receives.
    for other in (agent("two", "You audit vendor security.", ["research"]),
                  agent("two", "You research vendors.", ["research", "finance"]),
                  agent("two", "You research vendors.", ["research"],
                        [MCPServerSpec("search", "stdio", command="search-server")])):
        distinct = registry_of(agent("one", "You research vendors.", ["research"]), other)
        assert lint_contributions(graph, distinct) == []

    # An agent the registry does not hold compares by its id.
    assert lint_contributions(graph, registry_of(agent("one"))) == []
    b.agent_id = "one"
    assert lint_contributions(graph, Registry()) == [["a", "b"]]


def test_the_lint_never_changes_the_graph():
    graph = fork(Node(id="brief", label="Read the brief"), sibling("a"), sibling("b"))
    registry = Registry()
    before = deepcopy(graph_to_dict(graph))

    assert lint_contributions(graph, registry) == [["a", "b"]]
    assert graph_to_dict(graph) == before
    assert graph.planning is None
    assert registry.list_agents() == []


# ---------------------------------------------------------------------------
# Routed plans record what the lint finds
# ---------------------------------------------------------------------------


class FanOutArchitect(DeterministicArchitect):
    """Plans one research step per label, a merge, and a record of its own."""

    def __init__(self, labels) -> None:
        self.labels = list(labels)

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        nodes = [Node(id=f"research-{i}", label=label) for i, label in enumerate(self.labels)]
        nodes.append(Node(id="merge", label="Merge the research", depends_on=[n.id for n in nodes]))
        graph = ExecutionGraph([Topology.FORK_JOIN], nodes)
        graph.planning = {"templates": [{"name": "fan-out"}]}
        return graph, Registry()


DUPLICATED = ["Research the vendor"] * 3
DISTINCT = ["Research the vendor's pricing", "Research the vendor's security record",
            "Research the vendor's support terms"]


def routed(labels, provider=None, **kwargs) -> Swarm:
    return Swarm(provider=provider or MockProvider(), model="test-model", artifact_dir=None,
                 router=WhiteRabbit(autonomous=FanOutArchitect(labels)), **kwargs)


def coordination(groups) -> dict:
    return {"version": COORDINATION_RECORD_VERSION, "duplicate_siblings": groups}


def test_a_routed_plan_records_repeated_siblings_without_changing_the_plan():
    planned = FanOutArchitect(DUPLICATED).plan(Task("Compare vendors"))[0]
    graph = routed(DUPLICATED).plan(Task("Compare vendors"))

    assert COORDINATION_RECORD_VERSION == 1
    assert graph.planning["coordination"] == coordination([REPEATED])
    # Each component keeps its own key.
    assert set(graph.planning) == {"templates", "routing", "coordination"}
    assert graph.planning["templates"] == [{"name": "fan-out"}]
    assert graph.planning["routing"]["method"] == "fallback_no_classifier"
    assert [(n.id, n.label, n.depends_on) for n in graph.nodes] == [
        (n.id, n.label, n.depends_on) for n in planned.nodes]
    assert all("coordination" not in node.metadata for node in graph.nodes)


@pytest.mark.asyncio
async def test_an_async_routed_plan_records_repeated_siblings():
    graph = await routed(DUPLICATED).aplan(Task("Compare vendors"))

    assert graph.planning["coordination"] == coordination([REPEATED])


def test_a_routed_plan_with_distinct_contributions_records_nothing():
    graph = routed(DISTINCT).plan(Task("Compare vendors"))

    assert set(graph.planning) == {"templates", "routing"}


class CustomRouter(WhiteRabbit):
    def route(self, task):
        return self._fallback()


@pytest.mark.parametrize("swarm", [
    lambda: Swarm(provider=MockProvider(), model="test-model", artifact_dir=None,
                  architect=FanOutArchitect(DUPLICATED)),
    lambda: Swarm(provider=MockProvider(), model="test-model", artifact_dir=None,
                  router=CustomRouter(autonomous=FanOutArchitect(DUPLICATED))),
], ids=["no-router", "custom-router"])
def test_a_plan_without_a_routing_record_is_not_linted(swarm):
    graph = swarm().plan(Task("Compare vendors"))

    assert graph.planning == {"templates": [{"name": "fan-out"}]}


class FailOnceProvider(Provider):
    """Fails the first call for the merge step, as a lost connection would."""

    def __init__(self) -> None:
        self.failed = False

    async def complete(self, system, prompt, model):
        if "Merge the research" in prompt and not self.failed:
            self.failed = True
            raise RuntimeError("connection lost")
        return CompletionResult(text=f"done: {prompt[-30:]}", prompt_tokens=5, completion_tokens=5)


def test_the_record_round_trips_through_a_checkpoint_and_resume(tmp_path):
    store = FileCheckpointStore(tmp_path)
    provider = FailOnceProvider()
    with pytest.raises(RuntimeError, match="connection lost"):
        routed(DUPLICATED, provider, checkpoint_store=store).execute(Task("Compare vendors"))
    [execution_id] = store.list_ids()
    saved = store.load(execution_id)["graph"]["planning"]
    assert saved["coordination"] == coordination([REPEATED])

    resumed = Swarm(provider=provider, model="test-model", artifact_dir=None,
                    checkpoint_store=store).resume(execution_id)

    assert resumed.graph.planning == saved
    assert store.load(execution_id)["status"] == "completed"
    assert store.load(execution_id)["graph"]["planning"] == saved


# ---------------------------------------------------------------------------
# Durable runs
# ---------------------------------------------------------------------------


def durable(store, labels) -> Swarm:
    router = WhiteRabbit(classifier_provider=OfflineProvider(), classifier_model="offline",
                         autonomous=LocalOnly(lambda: FanOutArchitect(labels), "fan-out", "1"))
    return Swarm(model="offline", provider=OfflineProvider(), router=router, run_store=store,
                 max_budget_usd=1)


def test_a_durable_plan_saves_the_record_and_replays_it_without_linting(tmp_path, monkeypatch):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        point = crash_when(monkeypatch, store, "save_checkpoint",
                           lambda lease, revision, state, **_: state["status"] == "planned")
        with pytest.raises(Crash):
            durable(store, DUPLICATED).plan(Task("Compare vendors"))
        point.restore()
        (run,) = store.list_runs()
        run_id = run["run_id"]
        result = store.load_operation(run_id, "planning")["result"]
        recorded = result["graph"]["planning"]
        assert recorded["coordination"] == coordination([REPEATED])
        assert set(recorded) == {"templates", "routing", "coordination"}
        # Only the routing decision is repeated beside the saved graph.
        assert set(result) == {"graph", "registry", "routing"}

        def refuse(*args, **kwargs):
            raise AssertionError("a replayed plan was linted again")

        monkeypatch.setattr(planner, "lint_contributions", refuse)
        resumed = durable(store, DUPLICATED).resume(run_id)

        assert resumed.graph.planning == recorded
        assert store.get_checkpoint(run_id)["checkpoint"]["graph"]["planning"] == recorded


def test_a_durable_plan_with_distinct_contributions_records_nothing(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store, DISTINCT).execute(Task("Compare vendors"))
        operation = store.load_operation(result.execution_id, "planning")

    assert set(operation["result"]["graph"]["planning"]) == {"templates", "routing"}
    assert set(result.graph.planning) == {"templates", "routing"}


# ---------------------------------------------------------------------------
# The contribution templates example
# ---------------------------------------------------------------------------


def test_the_example_templates_pass_and_its_repeated_fan_out_is_flagged():
    spec = importlib.util.spec_from_file_location("contribution_templates_example", EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    task = example.TASK

    for template in example.TEMPLATES:
        nodes, registry = template.builder(task)
        assert len(nodes) > 1
        assert lint_contributions(fork(*nodes), registry) == [], template.name
    nodes, registry = example.REPEATED.builder(task)
    assert lint_contributions(fork(*nodes), registry) == [[node.id for node in nodes]]

    graph = example.routed_swarm([template.name for template in example.TEMPLATES]).plan(task)
    assert len(graph.nodes) == sum(len(t.builder(task)[0]) for t in example.TEMPLATES)
    assert graph.planning["routing"]["tier"] == "constrained"
    assert "coordination" not in graph.planning

    graph = example.routed_swarm(["source_extracts", example.REPEATED.name]).plan(task)
    flagged = graph.planning["coordination"]["duplicate_siblings"]
    assert flagged == [[f"{example.REPEATED.name}-1-research-{i}" for i in range(3)]]
    assert [node.id for node in graph.nodes][3:] == flagged[0]
