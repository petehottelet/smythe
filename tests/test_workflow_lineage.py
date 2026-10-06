"""Durable runs record their artifact's lineage once, when they complete.

The research runs here read versioned sources from the fake external service
in ``fixture_service`` over loopback HTTP, extract from each with a model
node given a context snapshot, and publish a brief written by local code, so
every input a node had is known and the record can be checked against what
the service served.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import smythe.workflow
from fakes import Crash, crash_when
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptanceDecision, AcceptancePolicy, ContextSource, DeliveryAcceptance, LineagePolicy,
    LocalOnly, MarkdownRenderer, OfflineProvider, ReadProfile, SimpleArchitect,
    SQLiteWorkflowStore, Swarm, Task, TextValidator, WorkflowBindingError,
)
from smythe._canonical import canonical_bytes, sha256_hex
from smythe._run_inspection import inspect_journal
from smythe.context import text_sha256
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.lineage import build_lineage, load_lineage, validator_set_sha256
from smythe.workflow import WorkflowRuntime, current_decision
from smythe.workflow_artifact import PublishedArtifact, current_artifact
from smythe.workflow_store import _canonical, _sha

QUESTION = "Should new subscriptions default to monthly billing?"
TEXTS = {"survey": "In 2019 most customers preferred annual billing.",
         "renewals": "In 2026, 71% of customers chose monthly billing over annual billing."}
NODES = ["read-survey", "extract-survey", "read-renewals", "extract-renewals", "synthesis"]
READER = {"name": "library", "version": "1", "decoder_version": "fixture-source-v1"}


def content(body: bytes) -> str:
    """Decode the fake service's source record to its text."""
    return json.loads(body)["content"]


class Brief:
    """Writes the brief from every node's result: deterministic local code."""

    def synthesize(self, graph: ExecutionGraph) -> str:
        return "\n\n".join(f"## {node.id}\n{node.result}" for node in graph.nodes if node.result)


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


def research(store, library, tmp_path, **options) -> Swarm:
    """A durable research run: the run's sources are the service's first versions."""
    _, records, endpoint = library
    settings = {
        "model": "offline", "provider": OfflineProvider(), "architect": SimpleArchitect(),
        "run_store": store, "max_budget_usd": 1, "read_tools": [profile(endpoint)],
        "context_sources": [ContextSource(record.source_id, record.content, str(record.version))
                            for record in records.values()],
        "synthesizer": LocalOnly(Brief, "brief-writer", "1", role="synthesizer"),
        "renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts",
        "acceptance": strict(), "lineage": LineagePolicy(access_scope="research-team"),
    }
    settings.update(options)
    return Swarm(**settings)


def research_graph(library) -> ExecutionGraph:
    """One read step per source, and an extraction from each read."""
    reads = profile(library[2])
    nodes = []
    for source_id in TEXTS:
        nodes.append(reads.step(f"read-{source_id}", {"source_id": source_id}))
        nodes.append(Node(id=f"extract-{source_id}", label=f"Extract the {source_id} finding",
                          depends_on=[f"read-{source_id}"]))
    graph = ExecutionGraph([Topology.FORK_JOIN], nodes)
    graph.task = Task(QUESTION)
    return graph


def durable(store, **options) -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, acceptance=strict(), **options)


def serial(count: int) -> ExecutionGraph:
    ids = [f"n{index}" for index in range(count)]
    return ExecutionGraph([Topology.SERIAL], [
        Node(id=node_id, label=f"Step {node_id}", depends_on=ids[index - 1:index] if index else [])
        for index, node_id in enumerate(ids)])


def by_id(lineage) -> dict[str, dict]:
    return {node["id"]: node for node in lineage.to_dict()["nodes"]}


def source_reads(service) -> dict[str, int]:
    return {source_id: item["reads"] for source_id, item in service.snapshot()["sources"].items()}


def call_ids(store, run_id) -> list[str]:
    return [call["call_id"] for call in store.inspect_run(run_id)["calls"]]


# ---------------------------------------------------------------------------
# What a completed run records
# ---------------------------------------------------------------------------


def test_lineage_lists_what_each_node_read_and_binds_the_document_and_decision(
        store, library, tmp_path):
    _, records, _ = library
    result = research(store, library, tmp_path).execute(research_graph(library))
    assert result.accepted
    run_id = result.execution_id
    lineage = load_lineage(store, run_id)
    assert lineage.coverage == "complete" and lineage.access_scope == "research-team"
    assert lineage.run == {"store_id": store.store_id, "run_id": run_id,
                           "recipe_sha256": store.load_run(run_id)["config_sha256"],
                           "task_sha256": result.acceptance.task_sha256}
    nodes = by_id(lineage)
    assert list(nodes) == NODES
    served = {source_id: {"source_id": source_id, "version": "1", "sha256": record.sha256}
              for source_id, record in records.items()}
    for source_id in TEXTS:
        read, extract = nodes[f"read-{source_id}"], nodes[f"extract-{source_id}"]
        # A read step records the profile it read with and the source it returned.
        assert (read["kind"], read["coverage"], read["depends_on"]) == ("read", "declared", [])
        assert read["inputs"] == {"sources": [served[source_id]], "tool": READER}
        assert read["result_sha256"] == records[source_id].sha256
        # A model node records the snapshot it was given: the run's sources at their versions.
        assert (extract["kind"], extract["coverage"]) == ("model", "declared")
        assert extract["depends_on"] == [f"read-{source_id}"]
        assert extract["inputs"]["sources"] == list(served.values())
        assert extract["inputs"]["context"] == {"snapshot_key": f"context/extract-{source_id}/0",
                                                "assembly_version": 1}
        assert set(extract["inputs"]) == {"sources", "context", "agent_sha256"}
        extracted = next(node for node in result.graph.nodes if node.id == extract["id"])
        assert extract["result_sha256"] == text_sha256(extracted.result)
    # The brief: local code whose inputs are exactly the graph's results.
    synthesized = store.load_operation(run_id, "synthesis")["result"]["output"]
    assert nodes["synthesis"] == {
        "id": "synthesis", "generation": 0, "kind": "synthesis", "depends_on": NODES[:-1],
        "result_sha256": text_sha256(synthesized),
        "inputs": {"sources": [], "component": {"name": "brief-writer", "version": "1"}},
        "coverage": "declared"}
    # The artifact's digest is that of the published bytes.
    publication = current_artifact(store, run_id).publication
    published = publication.path.read_bytes()
    assert lineage.artifact == {
        "relative_path": f"{run_id}/document.md", "sha256": hashlib.sha256(published).hexdigest(),
        "size_bytes": len(published), "media_type": "text/markdown; charset=utf-8",
        "renderer": {"name": "markdown_renderer", "version": "1",
                     "spec_sha256": sha256_hex({"type": "markdown_renderer", "version": 1})}}
    # The acceptance block binds the decision the run applied.
    policy = strict()
    assert lineage.to_dict()["acceptance"] == {
        "operation_key": current_decision(store, run_id)["operation_key"],
        "record_sha256": sha256_hex(result.acceptance.to_dict()), "decision": "accepted",
        "policy_sha256": policy.sha256,
        "validator_set_sha256": validator_set_sha256(policy.validator_identities),
        "validators": [identity.to_dict() for identity in policy.validator_identities]}
    # Ids, versions and digests only.
    encoded = canonical_bytes(lineage.to_dict())
    assert all(text.encode() not in encoded for text in TEXTS.values())
    (operation,) = store.list_operations(run_id, "lineage")
    assert (operation["operation_key"], operation["state"]) == ("lineage", "applied")


def test_a_read_of_another_version_is_recorded_by_its_own_digest(store, library, tmp_path):
    service, records, _ = library
    changed = service.change_source("renewals", "In 2026, monthly billing led renewals.")
    result = research(store, library, tmp_path).execute(research_graph(library))
    nodes = by_id(load_lineage(store, result.execution_id))
    # The read returned version 2, which is none of the run's sources: its version is unknown.
    assert nodes["read-renewals"]["inputs"]["sources"] == [
        {"source_id": "renewals", "version": None, "sha256": changed.sha256}]
    assert nodes["read-survey"]["inputs"]["sources"][0]["version"] == "1"
    # The extraction was still given the run's sources, at version 1.
    assert {item["sha256"] for item in nodes["extract-renewals"]["inputs"]["sources"]} == {
        record.sha256 for record in records.values()}


def test_nodes_without_a_snapshot_are_undeclared_and_the_run_partial(store):
    result = durable(store, lineage=LineagePolicy()).execute(serial(2))
    assert result.accepted
    lineage = load_lineage(store, result.execution_id)
    nodes = by_id(lineage)
    assert lineage.coverage == "partial" and list(nodes) == ["n0", "n1", "synthesis"]
    for node_id in ("n0", "n1"):
        # A node called without a context snapshot was given inputs nothing recorded.
        assert nodes[node_id]["coverage"] == "undeclared"
        assert set(nodes[node_id]["inputs"]) == {"sources", "agent_sha256"}
        assert nodes[node_id]["inputs"]["sources"] == []
    # The default DELIVERABLE synthesis calls no model, so it is declared.
    assert nodes["synthesis"]["inputs"]["component"] == {"name": "synthesizer/deliverable",
                                                         "version": "1"}
    assert nodes["synthesis"]["coverage"] == "declared"


def test_a_snapshot_that_does_not_match_the_recorded_results_is_undeclared(
        store, library, tmp_path, monkeypatch):
    def assembled(graph, recipe, snapshots, *args, **kwargs):
        snapshots = dict(snapshots)
        # One node lost its snapshot; another's names other dependency output.
        del snapshots["extract-survey"]
        given = snapshots["extract-renewals"]
        snapshots["extract-renewals"] = replace(given, dependencies=(
            replace(given.dependencies[0], output_sha256=sha256_hex("other")),))
        return build_lineage(graph, recipe, snapshots, *args, **kwargs)

    monkeypatch.setattr(smythe.workflow, "build_lineage", assembled)
    result = research(store, library, tmp_path).execute(research_graph(library))
    lineage = load_lineage(store, result.execution_id)
    coverage = {node["id"]: node["coverage"] for node in lineage.to_dict()["nodes"]}
    assert lineage.coverage == "partial"
    assert coverage == {"read-survey": "declared", "extract-survey": "undeclared",
                        "read-renewals": "declared", "extract-renewals": "undeclared",
                        "synthesis": "declared"}
    assert "context" not in by_id(lineage)["extract-survey"]["inputs"]


def test_runs_without_a_lineage_policy_write_no_lineage(store, library, tmp_path):
    plain = research(store, library, tmp_path, lineage=None).execute(research_graph(library))
    strict_only = durable(store).execute(serial(1))
    for result in (plain, strict_only):
        assert result.accepted
        assert store.list_operations(result.execution_id, "lineage") == []
        assert load_lineage(store, result.execution_id) is None


def test_inspection_never_returns_the_lineage(store, library, tmp_path):
    result = research(store, library, tmp_path).execute(research_graph(library))
    run_id = result.execution_id
    lineage = load_lineage(store, run_id)
    agents = [node["inputs"]["agent_sha256"] for node in lineage.to_dict()["nodes"]
              if "agent_sha256" in node["inputs"]]
    for view in (store.inspect_run(run_id), inspect_journal(store, run_id),
                 store.list_runs()):
        text = json.dumps(view)
        assert '"lineage' not in text and lineage.sha256 not in text
        assert not any(agent in text for agent in agents)


def read_call(node_id: str, text: str, *, generation: int = 0) -> dict:
    """The journal's record of an accepted read, as ``load_replay()`` returns it."""
    return {"run_id": "run-1", "key": {"phase": "tool", "scope_id": f"node/{node_id}",
                                       "generation": generation, "invocation": 0, "attempt": 0,
                                       "turn": 0},
            "provider": {"kind": "read_tool", "name": "pages", "version": "3"},
            "decoder_version": "utf8-text-v1", "result_state": "applied",
            "decoded_result": {"text": text}}


def test_build_lineage_orders_names_and_labels_from_the_records_it_is_given(tmp_path):
    def step(node_id, path):
        return Node(id=node_id, label=node_id,
                    metadata={"read_step": {"profile": "pages", "input": {"path": path}}})

    graph = ExecutionGraph([Topology.SERIAL], [
        # Listed before its dependency, twice over, and named like the synthesis.
        Node(id="synthesis", label="Draft", depends_on=["fetch", "fetch"]),
        step("fetch", "a"), step("page", "b"), step("stale", "c"),
        Node(id="check", label="Check", depends_on=["synthesis"], verifies="synthesis"),
    ])
    texts = {"fetch": "Known source text", "page": "An unlisted page", "stale": "Old text"}
    for node in graph.nodes:
        node.status, node.result = NodeStatus.COMPLETED, texts.get(node.id, f"text of {node.id}")
    graph.run_ref = {"version": 1, "store_id": "store-1", "run_id": "run-1",
                     "recipe_sha256": sha256_hex("recipe")}
    policy = strict()
    recipe = {"components": {"synthesizer": {"type": "synthesizer", "version": 1,
                                             "strategy": "llm_merge", "completion": {}}},
              "acceptance": policy.to_dict(), "lineage": LineagePolicy().to_dict()}
    renderer = {"type": "local_only", "role": "renderer", "identity": "r" * 300, "version": "2"}
    publication = PublishedArtifact(
        sequence=0, directory=str(tmp_path / "run-1"), name="document.txt",
        sha256=sha256_hex("document"), size_bytes=8, media_type="text/plain; charset=utf-8",
        renderer=renderer, synthesis="synthesis", output_sha256=sha256_hex("output"))
    decision = DeliveryAcceptance(AcceptanceDecision.ACCEPTED, policy.sha256,
                                  output_sha256=sha256_hex("document"), strategy="llm_merge")
    known = ContextSource("known", texts["fetch"], "v2")
    reads = {"fetch": read_call("fetch", texts["fetch"]), "page": read_call("page", texts["page"]),
             "stale": read_call("stale", texts["stale"], generation=1)}
    lineage = build_lineage(graph, recipe, {}, reads, publication,
                            ("acceptance/" + sha256_hex("inputs"), decision),
                            {"type": "registry", "version": 1, "agents": []},
                            synthesis=("synthesis", "merged"), sources=[known])
    nodes = by_id(lineage)
    assert list(nodes) == ["fetch", "page", "stale", "synthesis", "check", "synthesis#2"]
    assert nodes["synthesis"]["depends_on"] == ["fetch"] and nodes["check"]["kind"] == "gate"
    tool = {"name": "pages", "version": "3", "decoder_version": "utf8-text-v1"}
    # Text that is one of the run's sources is that source, though the step names no id.
    assert nodes["fetch"]["inputs"] == {"sources": [{"source_id": "known", "version": "v2",
                                                     "sha256": known.sha256}], "tool": tool}
    # Other text under no named id cannot be named; a read of another generation is not this one.
    assert (nodes["page"]["inputs"], nodes["page"]["coverage"]) == (
        {"sources": [], "tool": tool}, "undeclared")
    assert (nodes["stale"]["inputs"], nodes["stale"]["coverage"]) == ({"sources": []}, "undeclared")
    # A synthesis that calls a model is undeclared; its id stays apart from the graph's.
    assert nodes["synthesis#2"] == {
        "id": "synthesis#2", "generation": 0, "kind": "synthesis",
        "depends_on": ["fetch", "page", "stale", "synthesis", "check"],
        "result_sha256": text_sha256("merged"), "inputs": {"sources": []},
        "coverage": "undeclared"}
    assert lineage.coverage == "partial" and lineage.run["task_sha256"] is None
    # A label too long for a record is its digest.
    assert lineage.to_dict()["artifact"] == {
        "relative_path": "run-1/document.txt", "sha256": sha256_hex("document"), "size_bytes": 8,
        "media_type": "text/plain; charset=utf-8",
        "renderer": {"name": "sha256:" + text_sha256("r" * 300), "version": "2",
                     "spec_sha256": sha256_hex(renderer)}}


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_a_run_lost_before_its_lineage_writes_it_once_on_resume(
        store, library, tmp_path, monkeypatch):
    service, _, _ = library
    stop = crash_when(monkeypatch, store, "begin_operation",
                      lambda lease, key, kind, inputs: kind == "lineage")
    failed = crash_when(monkeypatch, store, "save_checkpoint", lambda *args, **kwargs: stop.fired)
    with pytest.raises(Crash):
        research(store, library, tmp_path).execute(research_graph(library))
    failed.restore()
    stop.restore()
    (run,) = store.list_runs()
    run_id = run["run_id"]
    assert store.list_operations(run_id, "lineage") == [] and load_lineage(store, run_id) is None
    (decision,) = store.list_operations(run_id, "acceptance")
    assert decision["state"] == "completed"
    calls, reads = call_ids(store, run_id), source_reads(service)

    result = research(store, library, tmp_path).resume(run_id)
    assert result.accepted
    assert call_ids(store, run_id) == calls and source_reads(service) == reads
    (operation,) = store.list_operations(run_id, "lineage")
    assert operation["state"] == "applied"
    lineage = load_lineage(store, run_id)
    assert lineage.coverage == "complete"
    assert lineage.acceptance["operation_key"] == decision["operation_key"]


def test_a_lineage_written_before_a_lost_checkpoint_is_replayed(
        store, library, tmp_path, monkeypatch):
    service, _, _ = library
    stop = crash_when(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs:
                      checkpoint["status"] == "completed")
    with pytest.raises(Crash):
        research(store, library, tmp_path).execute(research_graph(library))
    stop.restore()
    (run,) = store.list_runs()
    run_id = run["run_id"]
    (written,) = store.list_operations(run_id, "lineage")
    assert written["state"] == "completed" and load_lineage(store, run_id) is None
    calls, reads = call_ids(store, run_id), source_reads(service)

    def refuse(*args, **kwargs):
        pytest.fail("A committed lineage was assembled again")

    monkeypatch.setattr(smythe.workflow, "build_lineage", refuse)
    result = research(store, library, tmp_path).resume(run_id)
    assert result.accepted
    (applied,) = store.list_operations(run_id, "lineage")
    assert applied["operation_id"] == written["operation_id"] and applied["state"] == "applied"
    assert applied["result"] == written["result"]
    assert call_ids(store, run_id) == calls and source_reads(service) == reads
    # Resuming the completed run checks the record against the journal again.
    monkeypatch.setattr(smythe.workflow, "build_lineage", build_lineage)
    again = research(store, library, tmp_path).resume(run_id)
    assert again.accepted and again.acceptance == result.acceptance


def test_a_run_whose_nodes_ran_again_never_replays_an_earlier_lineage(
        store, library, tmp_path, monkeypatch):
    service, _, _ = library
    stop = crash_when(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs:
                      checkpoint["status"] == "completed")
    with pytest.raises(Crash):
        research(store, library, tmp_path).execute(research_graph(library))
    stop.restore()
    (run,) = store.list_runs()
    run_id = run["run_id"]
    (stale,) = store.list_operations(run_id, "lineage")
    assert (stale["operation_key"], stale["state"]) == ("lineage", "completed")
    # The sources changed while the run was stopped, so every node runs again.
    changed = [service.change_source(source_id, text + " Revised.")
               for source_id, text in TEXTS.items()]
    research(store, library, tmp_path).refresh_context(
        run_id, [ContextSource(record.source_id, record.content, str(record.version))
                 for record in changed])

    result = research(store, library, tmp_path).resume(run_id)
    assert result.accepted
    (synthesis,) = [item["operation_key"] for item in store.list_operations(run_id, "synthesis")
                    if item["applied_revision"] is not None]
    assert synthesis.startswith("synthesis/")
    # The record of this graph state is written under its own key and applied alone.
    applied = [item for item in store.list_operations(run_id, "lineage")
               if item["applied_revision"] is not None]
    assert [item["operation_key"] for item in applied] == [
        "lineage/" + synthesis.removeprefix("synthesis/")]
    lineage = load_lineage(store, run_id)
    # Every graph node is at its second generation; the synthesis is named by its key.
    assert {node["id"]: node["generation"] for node in lineage.to_dict()["nodes"]} == {
        node_id: 1 for node_id in NODES[:-1]} | {synthesis: 0}
    assert lineage.acceptance["operation_key"] == current_decision(store, run_id)["operation_key"]
    assert lineage.acceptance["operation_key"] != stale["result"]["lineage"]["acceptance"][
        "operation_key"]
    nodes = by_id(lineage)
    for record in changed:
        assert nodes[f"read-{record.source_id}"]["inputs"]["sources"] == [
            {"source_id": record.source_id, "version": "2", "sha256": record.sha256}]
    # The earlier graph state's record stays in the journal, never applied.
    (kept,) = [item for item in store.list_operations(run_id, "lineage")
               if item["operation_key"] == "lineage"]
    assert kept["applied_revision"] is None and kept["result"] == stale["result"]
    # Resuming the completed run checks the record of its final graph state.
    again = research(store, library, tmp_path).resume(run_id)
    assert again.accepted and again.acceptance == result.acceptance


def test_a_completed_run_whose_lineage_no_longer_matches_is_not_delivered(store):
    result = durable(store, lineage=LineagePolicy()).execute(serial(2))
    run_id = result.execution_id
    (operation,) = store.list_operations(run_id, "lineage")
    data = operation["result"]
    data["lineage"]["nodes"][0]["result_sha256"] = sha256_hex("other output")
    stored = _canonical(data)
    with store._db:
        store._db.execute("UPDATE workflow_operations SET result_json=?, result_sha=? "
                          "WHERE operation_id=?", (stored, _sha(stored), operation["operation_id"]))
    again = durable(store, lineage=LineagePolicy()).resume(run_id)
    assert again.acceptance.decision is AcceptanceDecision.INDETERMINATE and not again.accepted
    assert again.rejected_output == result.output
    assert "the run's lineage record does not match its journal" in again.acceptance.reasons


# ---------------------------------------------------------------------------
# Ids, cost and size
# ---------------------------------------------------------------------------


def test_lineage_runs_refuse_blank_ids_before_any_call(store):
    blank = ExecutionGraph([Topology.SERIAL], [Node(id=" ", label="Blank")])
    with pytest.raises(WorkflowBindingError, match="blank"):
        durable(store, lineage=LineagePolicy()).execute(blank)
    with pytest.raises(WorkflowBindingError, match="blank"):
        durable(store, lineage=LineagePolicy(),
                context_sources=[ContextSource(" ", "Text")]).execute(serial(1))
    assert [store.inspect_run(run["run_id"])["call_count"] for run in store.list_runs()] == [0]
    # Without lineage, such ids are used as before.
    assert durable(store).execute(blank).accepted


def test_checkpoint_save_cost_does_not_grow_with_history_under_lineage(tmp_path, monkeypatch):
    saves = []
    original = WorkflowRuntime._save

    def counted(self, *args, **kwargs):
        statements = []
        self.store._db.set_trace_callback(statements.append)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.store._db.set_trace_callback(None)
            saves.append(len(statements))

    monkeypatch.setattr(WorkflowRuntime, "_save", counted)

    def resumed_saves(prior, lineage):
        """SQL statements per save while resuming a run with ``prior`` finished nodes."""
        ids = [f"prior-{index}" for index in range(prior)]
        nodes = [Node(id=node, label=node, depends_on=ids[index - 1:index])
                 for index, node in enumerate(ids)]
        nodes.append(Node(id="last", label="Last", depends_on=ids[-1:]))
        with SQLiteWorkflowStore(tmp_path / f"history-{prior}-{lineage}.db") as store:
            swarm = durable(store, lineage=LineagePolicy() if lineage else None)
            prepare = store.prepare_call

            def stop_before_last(lease, key, **kwargs):
                if key.scope_id == "node/last":
                    raise Crash("Simulated process loss")
                return prepare(lease, key, **kwargs)

            monkeypatch.setattr(store, "prepare_call", stop_before_last)
            with pytest.raises(Crash):
                swarm.execute(ExecutionGraph([Topology.SERIAL], nodes))
            monkeypatch.setattr(store, "prepare_call", prepare)
            (run,) = store.list_runs()
            saves.clear()
            assert swarm.resume(run["run_id"]).accepted
            assert (load_lineage(store, run["run_id"]) is None) is not lineage
            return list(saves)

    short, long = resumed_saves(1, True), resumed_saves(5, True)
    plain_short, plain_long = resumed_saves(1, False), resumed_saves(5, False)
    # Opening checkpoint, the last node's checkpoint and the completed one.
    assert len(short) == len(long) == 3 and short == long
    # Only the completed save differs: it consumes the lineage operation too.
    assert short[:-1] == plain_short[:-1] and long[:-1] == plain_long[:-1]
    assert short[-1] - plain_short[-1] == long[-1] - plain_long[-1] > 0


def test_lineage_bytes_grow_with_nodes_not_with_events(store):
    def recorded(count, every):
        result = durable(store, lineage=LineagePolicy(),
                         checkpoint_every_n_nodes=every).execute(serial(count))
        lineage = load_lineage(store, result.execution_id)
        events = store.inspect_run(result.execution_id)["events"]
        return len(canonical_bytes(lineage.to_dict())), len(events)

    (few, few_events), (many, many_events) = recorded(2, 100), recorded(2, 1)
    assert many_events > few_events and many == few
    three, four = recorded(3, 100)[0], recorded(4, 100)[0]
    # Each node adds the same number of bytes.
    assert four - three == three - few > 0
