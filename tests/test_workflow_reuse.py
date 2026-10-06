"""A successor of a completed lineage run reuses the results that still hold.

The research runs here read versioned sources from the fake external service
in ``fixture_service`` over loopback HTTP and extract from each read with a
model node. Each model reply names its step and the digest of the prompt it
was given, so equal replies mean equal prompts. The reuse evaluator decides
which nodes a successor keeps; every other node runs again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

import smythe.workflow
from fakes import Crash, crash_when
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptancePolicy, ContextSource, CurrentState, LineagePolicy, LocalOnly, MarkdownRenderer,
    OfflineProvider, ReadProfile, SimpleArchitect, SourceStatus, SQLiteWorkflowStore, Swarm, Task,
    TextValidator, WorkflowBindingError,
)
from smythe._canonical import sha256_hex
from smythe._run_inspection import inspect_journal
from smythe.agent import Agent, AgentProfile
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.lineage import LINEAGE_REUSE_KEY, load_lineage
from smythe.provider import CompletionResult
from smythe.registry import Registry
from smythe.reuse import evaluate
from smythe.workflow_store import WorkflowStateError, _canonical, _sha

QUESTION = "Should new subscriptions default to monthly billing?"
TEXTS = {"survey": "In 2019 most customers preferred annual billing.",
         "renewals": "In 2026, 71% of customers chose monthly billing over annual billing.",
         "churn": "Monthly subscribers churned twice as often as annual ones in 2025."}
CHANGED = "Monthly subscribers churned no more often than annual ones in 2026."
SCOPE = "research-team"
PLAN = sha256_hex("repair plan")
REASON = "The churn source has a new version"
KEPT = ["read-survey", "extract-survey", "read-renewals", "extract-renewals"]


def content(body: bytes) -> str:
    """Decode the fake service's source record to its text."""
    return json.loads(body)["content"]


class Brief:
    """Writes the brief from every node's result: deterministic local code."""

    def synthesize(self, graph: ExecutionGraph) -> str:
        return "\n\n".join(f"## {node.id}\n{node.result}" for node in graph.nodes if node.result)


def install_replies(patch, log: list[str]) -> None:
    """Answer each model step with its label and its prompt's digest, logging the label."""

    async def complete(self, system, prompt, model):
        step = next((line for line in prompt.splitlines() if line.startswith("Your step: ")), "")
        label = step.removeprefix("Your step: ")
        log.append(label)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        return CompletionResult("PASS" if label == "Judge" else f"{label}: {digest}", cost_usd=0)

    patch.setattr(OfflineProvider, "complete", complete)


@pytest.fixture
def replies(monkeypatch) -> list[str]:
    log: list[str] = []
    install_replies(monkeypatch, log)
    return log


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


@pytest.fixture
def library():
    service = FixtureService()
    records = {source_id: service.add_source(source_id, text) for source_id, text in TEXTS.items()}
    with serve_http(service) as endpoint:
        yield service, records, endpoint


def strict() -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[TextValidator()])


def profile(endpoint) -> ReadProfile:
    return ReadProfile(name="library", version="1", url=endpoint.url + "/sources/{source_id}",
                       input_schema={"source_id": "string"}, metered=False, repeat_safe=True,
                       decode=content, decoder_version="fixture-source-v1", timeout_s=10.0,
                       allow_private_networks=True)


def research(store, endpoint, tmp_path, **options) -> Swarm:
    """A durable research run whose model steps are each given a context snapshot."""
    settings = {
        "model": "offline", "provider": OfflineProvider(), "architect": SimpleArchitect(),
        "run_store": store, "max_budget_usd": 1, "read_tools": [profile(endpoint)],
        "context_sources": [],
        "synthesizer": LocalOnly(Brief, "brief-writer", "1", role="synthesizer"),
        "renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts",
        "acceptance": strict(), "lineage": LineagePolicy(access_scope=SCOPE),
    }
    settings.update(options)
    return Swarm(**settings)


def research_graph(endpoint) -> ExecutionGraph:
    """One read step per source, and an extraction from each read."""
    reads = profile(endpoint)
    nodes = []
    for source_id in TEXTS:
        nodes.append(reads.step(f"read-{source_id}", {"source_id": source_id}))
        nodes.append(Node(id=f"extract-{source_id}", label=f"Extract the {source_id} finding",
                          depends_on=[f"read-{source_id}"]))
    graph = ExecutionGraph([Topology.FORK_JOIN], nodes)
    graph.task = Task(QUESTION)
    return graph


def successor(swarm: Swarm, run_id: str, reuse, **changes):
    """Run a successor of *run_id* through the runtime, as a repair would."""
    options = {"lineage_sha256": load_lineage(swarm._run_store, run_id).sha256, "reuse": reuse,
               "plan_sha256": PLAN, "reason": REASON,
               "max_concurrency": swarm._execution_concurrency()}
    options.update(changes)
    return asyncio.run(swarm._workflow_runtime()._run_successor(run_id, **options))


class Catalog:
    """Describes each source by the service's latest version, without reading it."""

    def __init__(self, service: FixtureService) -> None:
        self.service = service

    def describe(self, source_id):
        source = self.service.snapshot()["sources"].get(source_id)
        return None if source is None else SourceStatus(sha256=source["sha256"])


def decide(store, run_id, service):
    """What the run's lineage calls for now, under the current sources and configuration."""
    lineage = load_lineage(store, run_id)
    policy = strict()
    state = CurrentState(
        catalog=Catalog(service), validators=policy.validator_identities,
        policy_sha256=policy.sha256,
        tools={"library": {"version": "1", "decoder_version": "fixture-source-v1"}},
        components={"brief-writer": "1"},
        agents=[node["inputs"]["agent_sha256"] for node in lineage.nodes
                if "agent_sha256" in node["inputs"]],
        integrity={run_id: "verified"}, lineage=LineagePolicy(access_scope=SCOPE))
    return evaluate(lineage, state)


def source_reads(service) -> dict[str, int]:
    return {source_id: item["reads"] for source_id, item in service.snapshot()["sources"].items()}


def calls(store, run_id) -> list[tuple[str, str]]:
    return [(call["key"]["phase"], call["key"]["scope_id"])
            for call in store.inspect_run(run_id)["calls"]]


def results(graph) -> dict[str, str]:
    return {node.id: node.result for node in graph.nodes}


def by_id(lineage) -> dict[str, dict]:
    return {node["id"]: node for node in lineage.to_dict()["nodes"]}


def applying_checkpoints(store, run_id, operation_id) -> list[int]:
    """The revisions of the run's checkpoints that consumed the operation."""
    rows = store._db.execute("SELECT revision, operations_json FROM workflow_checkpoints "
                             "WHERE run_id=? ORDER BY revision", (run_id,)).fetchall()
    return [row["revision"] for row in rows if operation_id in json.loads(row["operations_json"])]


def tamper(store, run_id, node_id, text):
    """Rewrite one node's result in the run's latest checkpoint, keeping the row consistent."""
    row = store._db.execute("SELECT revision, checkpoint_json FROM workflow_checkpoints "
                            "WHERE run_id=? ORDER BY revision DESC LIMIT 1", (run_id,)).fetchone()
    checkpoint = json.loads(row["checkpoint_json"])
    for node in checkpoint["graph"]["nodes"]:
        if node["id"] == node_id:
            node["result"] = text
    stored = _canonical(checkpoint)
    with store._db:
        store._db.execute("UPDATE workflow_checkpoints SET checkpoint_json=?, checkpoint_sha=? "
                          "WHERE run_id=? AND revision=?", (stored, _sha(stored), run_id,
                                                            row["revision"]))


# ---------------------------------------------------------------------------
# A successor after one source changed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def changed(tmp_path_factory):
    """A completed run, then a successor after the churn source changed, then a full run."""
    tmp_path = tmp_path_factory.mktemp("reuse")
    service = FixtureService()
    for source_id, text in TEXTS.items():
        service.add_source(source_id, text)
    log: list[str] = []
    with (pytest.MonkeyPatch.context() as patch, serve_http(service) as endpoint,
          SQLiteWorkflowStore(tmp_path / "runs.db") as store):
        install_replies(patch, log)
        swarm = research(store, endpoint, tmp_path)
        first = swarm.execute(research_graph(endpoint))
        service.change_source("churn", CHANGED)
        decision = decide(store, first.execution_id, service)
        reuse = {node.id: list(node.reasons) for node in decision.nodes if node.action == "reuse"}
        reads, labels = source_reads(service), len(log)
        second = successor(swarm, first.execution_id, reuse)
        read_again = {source_id: count - reads[source_id]
                      for source_id, count in source_reads(service).items()}
        called_again = log[labels:]
        # The same graph computed in full from the changed sources.
        full = swarm.execute(research_graph(endpoint))
        yield SimpleNamespace(store=store, endpoint=endpoint, swarm=swarm, first=first,
                              second=second, full=full, decision=decision, reuse=reuse,
                              read_again=read_again, called_again=called_again,
                              lineage=load_lineage(store, first.execution_id))


def test_a_successor_runs_only_what_a_changed_source_affects(changed):
    actions = {node.id: node.action for node in changed.decision.nodes}
    assert changed.decision.action == "regenerate"
    assert {node_id for node_id, action in actions.items() if action == "reuse"} == set(KEPT)
    assert changed.reuse == {node_id: [] for node_id in KEPT}
    second = changed.second
    assert second.accepted and second.execution_id != changed.first.execution_id
    # Only the changed source is read again, and only its extraction calls a model.
    assert changed.read_again == {"survey": 0, "renewals": 0, "churn": 1}
    assert changed.called_again == ["Extract the churn finding"]
    assert calls(changed.store, second.execution_id) == [
        ("tool", "node/read-churn"), ("execution", "node/extract-churn")]
    assert second.workflow_accounting["call_count"] == 2
    # Reused results are the predecessor's, byte for byte.
    before, after = results(changed.first.graph), results(second.graph)
    assert {node_id: after[node_id] for node_id in KEPT} == {node_id: before[node_id]
                                                              for node_id in KEPT}
    assert after["read-churn"] == CHANGED != before["read-churn"]
    # Every result and the delivered document equal those of a full recomputation.
    assert after == results(changed.full.graph)
    assert second.output == changed.full.output != changed.first.output


def test_a_successor_lineage_names_its_predecessor_and_each_reused_result(changed):
    first, second = changed.first.execution_id, changed.second.execution_id
    lineage = load_lineage(changed.store, second)
    assert lineage.coverage == "complete"
    assert lineage.to_dict()["predecessor"] == {"store_id": changed.store.store_id,
                                                "run_id": first,
                                                "lineage_sha256": changed.lineage.sha256}
    assert lineage.to_dict()["repair"] == {"plan_sha256": PLAN, "reason": REASON}
    earlier, nodes = by_id(changed.lineage), by_id(lineage)
    for node_id in KEPT:
        # A reused node keeps the inputs its predecessor recorded.
        assert nodes[node_id]["reused_from"] == {"run_id": first, "generation": 0}
        assert {key: nodes[node_id][key] for key in ("kind", "inputs", "result_sha256")} == {
            key: earlier[node_id][key] for key in ("kind", "inputs", "result_sha256")}
    for node_id in ("read-churn", "extract-churn", "synthesis"):
        assert "reused_from" not in nodes[node_id]
    assert nodes["read-churn"]["inputs"]["sources"][0]["sha256"] == hashlib.sha256(
        CHANGED.encode()).hexdigest()
    # The successor's own graph records the repair, and nothing it planned before.
    assert changed.second.graph.planning == {"repair": {
        "predecessor": lineage.to_dict()["predecessor"], "plan_sha256": PLAN, "reason": REASON}}


def test_reuse_operations_carry_no_calls_and_are_consumed_once(changed):
    store, second = changed.store, changed.second.execution_id
    operations = store.list_operations(second, "reuse")
    assert sorted(item["operation_key"] for item in operations) == sorted(
        f"reuse/{node_id}" for node_id in KEPT)
    earlier = by_id(changed.lineage)
    for operation in operations:
        node_id = operation["inputs"]["node_id"]
        assert operation["state"] == "applied" and operation["consumed_call_ids"] == []
        assert operation["inputs"] == {
            "version": 1,
            "predecessor": {"store_id": store.store_id, "run_id": changed.first.execution_id,
                            "lineage_sha256": changed.lineage.sha256},
            "node_id": node_id, "generation": 0,
            "result_sha256": earlier[node_id]["result_sha256"],
            "records": {"lineage_node": sha256_hex(earlier[node_id]), "source_evidence": None},
            "reasons": [], "plan_sha256": PLAN}
        assert operation["result"]["text"] == results(changed.first.graph)[node_id]
        assert operation["result"]["lineage"] == earlier[node_id]
        # One checkpoint consumed it: the one that saved the node completed.
        assert applying_checkpoints(store, second, operation["operation_id"]) == [
            operation["applied_revision"]]
        node = next(item for item in changed.second.graph.nodes if item.id == node_id)
        assert node.metadata[LINEAGE_REUSE_KEY] == operation["inputs"]


def test_inspection_never_returns_reused_text(changed):
    store, second = changed.store, changed.second.execution_id
    kept = [results(changed.first.graph)[node_id] for node_id in KEPT]
    for view in (store.inspect_run(second), inspect_journal(store, second), store.list_runs()):
        text = json.dumps(view)
        assert not any(item in text for item in kept)


# ---------------------------------------------------------------------------
# What a successor refuses before it creates a run
# ---------------------------------------------------------------------------


def refused(changed, error, match, *, swarm=None, **changes):
    """Check a successor request fails with *error* and creates no run."""
    runs = len(changed.store.list_runs())
    with pytest.raises(error, match=match):
        successor(swarm or changed.swarm, changed.first.execution_id,
                  changes.pop("reuse", changed.reuse), **changes)
    assert len(changed.store.list_runs()) == runs


def test_a_successor_in_another_access_scope_is_refused(changed, tmp_path):
    other = research(changed.store, changed.endpoint, tmp_path,
                     lineage=LineagePolicy(access_scope="other-team"))
    refused(changed, WorkflowStateError, "no lineage record in access scope 'other-team'",
            swarm=other)


def test_a_lineage_digest_other_than_the_saved_record_is_refused(changed):
    refused(changed, WorkflowStateError, "is not the one given", lineage_sha256=sha256_hex("x"))


@pytest.mark.parametrize(("reuse", "match"), [
    ({"extract-survey": []}, "depending on a node that runs again"),
    ({"ghost": []}, "has no node 'ghost'"),
    ({"synthesis": []}, "has no node 'synthesis'"),
])
def test_reuse_needs_a_saved_node_whose_dependencies_are_reused(changed, reuse, match):
    refused(changed, WorkflowBindingError, match, reuse=reuse)


@pytest.mark.parametrize(("changes", "match"), [
    ({"reuse": ["read-survey"]}, "reuse must map node ids"),
    ({"reuse": {"read-survey": ["source_moved"]}}, "reuse must map node ids"),
    ({"reuse": {"read-survey": "none"}}, "reuse must map node ids"),
    ({"plan_sha256": "plan"}, "plan_sha256"),
    ({"reason": ""}, "reason"),
    ({"reason": "line\nbreak"}, "reason"),
])
def test_malformed_requests_are_refused(changed, changes, match):
    refused(changed, WorkflowBindingError, match, **changes)


class Reviewer:
    """A supervisor that never revises."""

    def review(self, graph, node, *, task, revisions_remaining):
        return None


@pytest.mark.parametrize(("options", "match"), [
    ({"lineage": LineagePolicy(access_scope=SCOPE, reuse=False)}, "reuse on"),
    ({"supervisor": LocalOnly(Reviewer, "reviewer", "1", role="supervisor")}, "supervisor"),
])
def test_reuse_needs_a_lineage_policy_with_reuse_on_and_no_supervisor(
        changed, tmp_path, options, match):
    swarm = research(changed.store, changed.endpoint, tmp_path, **options)
    refused(changed, WorkflowBindingError, match, swarm=swarm)


def test_a_successor_needs_a_lineage_policy(changed, tmp_path):
    swarm = research(changed.store, changed.endpoint, tmp_path, lineage=None)
    refused(changed, WorkflowBindingError, "needs a lineage policy", swarm=swarm, reuse={})


# ---------------------------------------------------------------------------
# Predecessor records that refuse reuse
# ---------------------------------------------------------------------------


def test_a_tampered_predecessor_result_is_refused_before_any_call(
        store, library, tmp_path, replies):
    service, _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    tamper(store, first, "extract-survey", "Annual billing is preferred.")
    reads, labels = source_reads(service), list(replies)
    with pytest.raises(WorkflowStateError, match="'extract-survey'.*not the one its lineage records"):
        successor(swarm, first, {"read-survey": [], "extract-survey": []})
    assert [run["run_id"] for run in store.list_runs()] == [first]
    assert source_reads(service) == reads and replies == labels


def test_a_node_that_used_a_removed_source_is_refused(store, library, tmp_path, replies):
    service, records, endpoint = library
    swarm = research(store, endpoint, tmp_path, context_sources=[
        ContextSource(record.source_id, record.content, str(record.version))
        for record in records.values()])
    first = swarm.execute(research_graph(endpoint)).execution_id
    swarm.remove_source(first, "survey", "Withdrawn by its owner")
    reads, labels = source_reads(service), list(replies)
    with pytest.raises(WorkflowStateError, match="a source it used was removed"):
        successor(swarm, first, {"read-survey": []})
    # Every extraction was given every source, the removed one included.
    with pytest.raises(WorkflowStateError, match="a source it used was removed"):
        successor(swarm, first, {"read-renewals": [], "extract-renewals": []})
    assert [run["run_id"] for run in store.list_runs()] == [first]
    assert source_reads(service) == reads and replies == labels


def test_gates_and_the_nodes_they_judge_are_never_reused(store, tmp_path, replies):
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1)])
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1, context_sources=[], acceptance=strict(),
                  lineage=LineagePolicy())
    first = swarm.execute(graph)
    assert first.accepted
    for reuse in ({"draft": []}, {"draft": [], "judge": []}):
        with pytest.raises(WorkflowBindingError, match="gates, the nodes they judge"):
            successor(swarm, first.execution_id, reuse)
    assert len(store.list_runs()) == 1
    # Without reuse, the successor runs the whole graph again.
    labels = len(replies)
    again = successor(swarm, first.execution_id, {})
    assert again.accepted and replies[labels:] == ["Draft", "Judge"]
    assert results(again.graph) == results(first.graph)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_a_resumed_successor_replays_its_reuse_without_reading_the_predecessor(
        store, library, tmp_path, replies, monkeypatch):
    service, _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint))
    service.change_source("churn", CHANGED)
    reuse = {node_id: [] for node_id in KEPT}
    # Lost after the reuse operations committed, before the checkpoint that consumes them.
    stop = crash_when(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs:
                      checkpoint["status"] == "running")
    with pytest.raises(Crash):
        successor(swarm, first.execution_id, reuse)
    stop.restore()
    (run_id,) = [run["run_id"] for run in store.list_runs() if run["run_id"] != first.execution_id]
    written = {item["operation_key"]: item for item in store.list_operations(run_id, "reuse")}
    assert sorted(written) == sorted(f"reuse/{node_id}" for node_id in KEPT)
    assert {item["state"] for item in written.values()} == {"completed"}
    assert calls(store, run_id) == []

    def refuse(*args, **kwargs):
        pytest.fail("A resumed successor loaded its predecessor again")

    monkeypatch.setattr(smythe.workflow, "_predecessor", refuse)
    for name in ("get_checkpoint", "load_operation", "list_operations", "load_run"):
        original = getattr(store, name)

        def guarded(target, *args, _original=original, **kwargs):
            if target == first.execution_id:
                pytest.fail("A resumed successor read its predecessor's records")
            return _original(target, *args, **kwargs)

        monkeypatch.setattr(store, name, guarded)
    reads, labels = source_reads(service), len(replies)
    result = swarm.resume(run_id)
    assert result.accepted
    assert replies[labels:] == ["Extract the churn finding"]
    assert source_reads(service) == {**reads, "churn": reads["churn"] + 1}
    assert calls(store, run_id) == [("tool", "node/read-churn"), ("execution", "node/extract-churn")]
    # The same operations, never written twice, each consumed by one checkpoint.
    operations = store.list_operations(run_id, "reuse")
    assert {item["operation_key"]: item["operation_id"] for item in operations} == {
        key: item["operation_id"] for key, item in written.items()}
    for operation in operations:
        assert operation["state"] == "applied"
        assert operation["result"] == written[operation["operation_key"]]["result"]
        assert applying_checkpoints(store, run_id, operation["operation_id"]) == [
            operation["applied_revision"]]
    before = results(first.graph)
    assert {node_id: results(result.graph)[node_id] for node_id in KEPT} == {
        node_id: before[node_id] for node_id in KEPT}
    # Resuming the completed successor checks its lineage against its own journal.
    again = swarm.resume(run_id)
    assert again.accepted and again.acceptance == result.acceptance


def test_a_resumed_successor_checks_its_predecessor_before_reusing(
        store, library, tmp_path, replies, monkeypatch):
    service, _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    # Lost after the planned checkpoint, before its first reuse operation.
    stop = crash_when(monkeypatch, store, "begin_operation",
                      lambda lease, key, kind, inputs: kind == "reuse")
    with pytest.raises(Crash):
        successor(swarm, first, {"read-survey": [], "extract-survey": []})
    stop.restore()
    (run_id,) = [run["run_id"] for run in store.list_runs() if run["run_id"] != first]
    assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "planned"
    tamper(store, first, "read-survey", "Customers preferred nothing.")
    reads, labels = source_reads(service), list(replies)
    with pytest.raises(WorkflowStateError, match="'read-survey'.*not the one its lineage records"):
        swarm.resume(run_id)
    assert calls(store, run_id) == []
    assert source_reads(service) == reads and replies == labels
    assert [item["state"] for item in store.list_operations(run_id, "reuse")] == ["started"]


def test_a_resumed_successor_refuses_a_reuse_that_does_not_match_its_inputs(
        store, library, tmp_path, replies, monkeypatch):
    _, _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint)).execution_id
    stop = crash_when(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs:
                      checkpoint["status"] == "running")
    with pytest.raises(Crash):
        successor(swarm, first, {"read-survey": []})
    stop.restore()
    (run_id,) = [run["run_id"] for run in store.list_runs() if run["run_id"] != first]
    (operation,) = store.list_operations(run_id, "reuse")
    data = dict(operation["result"], text="Other text")
    stored = _canonical(data)
    with store._db:
        store._db.execute("UPDATE workflow_operations SET result_json=?, result_sha=? "
                          "WHERE operation_id=?", (stored, _sha(stored), operation["operation_id"]))
    with pytest.raises(WorkflowStateError, match="saved reuse of node 'read-survey' does not match"):
        swarm.resume(run_id)
    assert calls(store, run_id) == []


# ---------------------------------------------------------------------------
# What a new graph cannot carry, and what a successor keeps
# ---------------------------------------------------------------------------


def test_a_new_graph_cannot_carry_reused_results_or_a_repair_record(store, replies):
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1, acceptance=strict(),
                  lineage=LineagePolicy())
    marked = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", metadata={LINEAGE_REUSE_KEY: {"node_id": "draft"}})])
    with pytest.raises(WorkflowBindingError, match="cannot inherit execution"):
        swarm.execute(marked)
    repaired = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft")])
    repaired.planning = {"repair": {"plan_sha256": PLAN, "reason": REASON}}
    with pytest.raises(WorkflowBindingError, match="cannot carry a repair record"):
        swarm.execute(repaired)
    assert replies == []
    assert all(store.inspect_run(run["run_id"])["call_count"] == 0 for run in store.list_runs())
    # Other planning records are kept as before.
    repaired.planning = {"routing": {"tier": "single"}}
    assert swarm.execute(repaired).accepted


class Planner:
    """A local planner that registers its own analyst for both steps."""

    def plan(self, task):
        registry = Registry()
        registry.register(Agent(AgentProfile("Analyst", persona="Reads billing data closely."),
                                id="analyst"))
        graph = ExecutionGraph([Topology.SERIAL], [
            Node(id="collect", label="Collect the facts", agent_id="analyst"),
            Node(id="write", label="Write the answer", agent_id="analyst",
                 depends_on=["collect"])])
        return graph, registry


def test_a_successor_keeps_the_agents_its_predecessor_planned(store, tmp_path, replies):
    swarm = Swarm(model="offline", provider=OfflineProvider(),
                  architect=LocalOnly(Planner, "planner", "1"), run_store=store,
                  max_budget_usd=1, context_sources=[], acceptance=strict(),
                  lineage=LineagePolicy())
    first = swarm.execute(Task(QUESTION))
    assert first.accepted
    labels = len(replies)
    second = successor(swarm, first.execution_id, {"collect": []})
    assert second.accepted and replies[labels:] == ["Write the answer"]
    assert results(second.graph) == results(first.graph)
    earlier, nodes = by_id(load_lineage(store, first.execution_id)), by_id(
        load_lineage(store, second.execution_id))
    # The regenerated step ran as the same planned agent.
    assert nodes["write"]["inputs"]["agent_sha256"] == earlier["write"]["inputs"]["agent_sha256"]
    assert nodes["collect"]["reused_from"] == {"run_id": first.execution_id, "generation": 0}


def test_refreshing_an_unfinished_successor_runs_its_reused_nodes_again(
        store, library, tmp_path, replies, monkeypatch):
    service, _, endpoint = library
    swarm = research(store, endpoint, tmp_path)
    first = swarm.execute(research_graph(endpoint))
    service.change_source("churn", CHANGED)
    # Lost after the reused nodes were saved complete, before the changed source was read.
    prepare = store.prepare_call

    def stop_before_read(lease, key, **kwargs):
        if key.scope_id == "node/read-churn":
            raise Crash("Simulated process loss")
        return prepare(lease, key, **kwargs)

    monkeypatch.setattr(store, "prepare_call", stop_before_read)
    with pytest.raises(Crash):
        successor(swarm, first.execution_id, {node_id: [] for node_id in KEPT})
    monkeypatch.setattr(store, "prepare_call", prepare)
    (run_id,) = [run["run_id"] for run in store.list_runs() if run["run_id"] != first.execution_id]
    applied = {item["operation_id"]: item["applied_revision"]
               for item in store.list_operations(run_id, "reuse")}
    assert len(applied) == len(KEPT) and None not in applied.values()
    # New sources start every node's next generation: a reused result was
    # bound to the predecessor's inputs, so its node runs again too.
    swarm.refresh_context(run_id, [ContextSource("notes", "Annual plans were discounted.")])
    reads, labels = source_reads(service), len(replies)
    result = swarm.resume(run_id)
    assert result.accepted
    assert source_reads(service) == {source_id: count + 1 for source_id, count in reads.items()}
    assert sorted(replies[labels:]) == sorted(f"Extract the {source_id} finding"
                                              for source_id in TEXTS)
    assert all(LINEAGE_REUSE_KEY not in node.metadata for node in result.graph.nodes)
    # The reuse operations stay applied once, by the checkpoint that first consumed them.
    for operation in store.list_operations(run_id, "reuse"):
        assert applying_checkpoints(store, run_id, operation["operation_id"]) == [
            applied[operation["operation_id"]]]
    lineage = load_lineage(store, run_id)
    assert not any("reused_from" in node for node in lineage.to_dict()["nodes"])
    assert lineage.to_dict()["predecessor"]["run_id"] == first.execution_id
    assert {node["generation"] for node in lineage.to_dict()["nodes"][:-1]} == {1}
