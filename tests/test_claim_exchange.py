"""Claim exchange: a step given its dependencies' claim records in place of their full output."""

from __future__ import annotations

import copy
import json
import re

import pytest

import fakes
import smythe.workflow
from fakes import ManualClock
from smythe import (
    AcceptancePolicy, ClaimPassage, ClaimRecord, ContextSource, LineagePolicy, LLMArchitect,
    OfflineProvider, SimpleArchitect, SourceEvidence, SourceEvidenceValidator, SourceRecord,
    SQLiteWorkflowStore, Swarm, Task,
)
from smythe.context import load_context_snapshot, text_sha256
from smythe.evidence import CLAIM_EXCHANGE_NOTE, claim_exchange, render_source_evidence
from smythe.executor_base import SKIPPED_DEPENDENCY_RESULT, ExecutorBase
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.lineage import build_lineage, load_lineage
from smythe.loader import build_graph_from_model_output
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import CallKey

POLL = ContextSource(
    "survey-2019", "An informal 2019 poll of 40 visitors: most customers prefer annual billing.",
    version="2019")
AUDIT = ContextSource(
    "renewals-2026", "Audited renewals for 2026 show that 71% of 18,400 customers chose monthly "
    "billing over annual billing.", version="2026-q2")
SOURCES = (POLL, AUDIT)

PREFERENCE = ClaimRecord(
    "billing-preference", "Most customers prefer annual billing.",
    supporting=(ClaimPassage("survey-2019", "most customers prefer annual billing"),),
    contradicting=(ClaimPassage("renewals-2026", "71% of 18,400 customers chose monthly billing"),),
    scope="Customers on self-serve plans", unresolved=("Whom did the 2019 poll ask?",),
    conclusion="Treat the preference as contested.",
)
MAJORITY = ClaimRecord(
    "monthly-majority", "Monthly billing is the majority choice in 2026.",
    supporting=(ClaimPassage("renewals-2026", "71% of 18,400 customers chose monthly billing"),),
    unresolved=("Does the audit include enterprise contracts?",),
)
# The poll step's evidence retains its source's text; the record exchanged leaves it out.
POLL_EVIDENCE = SourceEvidence((SourceRecord.from_context_source(POLL),
                                SourceRecord.from_context_source(AUDIT).without_text()),
                               (PREFERENCE,))
AUDIT_EVIDENCE = SourceEvidence((SourceRecord.from_context_source(AUDIT).without_text(),),
                                (MAJORITY,))
# Each research step's working transcript, which a step exchanging claims never sees.
POLL_NOTES = "Working notes: I read the 2019 poll twice and compared its wording.\n\n"
AUDIT_NOTES = "Working notes: the audit tables list renewals by plan and region.\n\n"
POLL_OUTPUT = POLL_NOTES + render_source_evidence(POLL_EVIDENCE) + "\n"
AUDIT_OUTPUT = AUDIT_NOTES + render_source_evidence(AUDIT_EVIDENCE) + "\n"


def exchanged(evidence: SourceEvidence) -> str:
    """The compact record of *evidence*: its claims, and its sources without text."""
    sources = tuple(source.without_text() for source in evidence.sources)
    block = render_source_evidence(SourceEvidence(sources, evidence.claims))
    return f"{CLAIM_EXCHANGE_NOTE}\n{block}"


_BLOCK = re.compile(r"```source-evidence\n(.*?)\n```", re.S)


def write_from_claims(prompt: str) -> str:
    """An artifact built only from the claim records in *prompt*, as a writer would build it.

    Its body quotes every passage, contradictions included, and it carries
    one evidence block that merges the records it was given.
    """
    sources: dict[str, SourceRecord] = {}
    claims: dict[str, ClaimRecord] = {}
    for block in _BLOCK.findall(prompt):
        evidence = SourceEvidence.from_dict(json.loads(block))
        sources.update((source.source_id, source) for source in evidence.sources)
        claims.update((claim.claim_id, claim) for claim in evidence.claims)
    merged = SourceEvidence(tuple(sources.values()), tuple(claims.values()))
    body = "# Billing\n\n" + "\n".join(
        f'- {role}: "{passage.quote}" ({passage.source_id})'
        for _, role, passage in merged.passages()) + "\n\n"
    return body + render_source_evidence(merged) + "\n"


def step(prompt: str) -> str:
    """The step a prompt asks for: its ``Your step:`` line, or else its first line."""
    for line in prompt.splitlines():
        if line.startswith("Your step: "):
            return line.removeprefix("Your step: ")
    return prompt.splitlines()[0]


REPLIES = {"Read the poll": POLL_OUTPUT, "Read the audit": AUDIT_OUTPUT,
           "Write the brief": write_from_claims, "Review the brief": "Reviewed."}


class Scripted(Provider):
    """Answer each step from REPLIES and keep the prompt each step was sent."""

    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    async def complete(self, system, prompt, model):
        label = step(prompt)
        self.prompts[label] = prompt
        reply = REPLIES[label]
        return CompletionResult(reply(prompt) if callable(reply) else reply, cost_usd=0.0)


def research_graph(**writer_metadata) -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [
        Node(id="poll", label="Read the poll"),
        Node(id="audit", label="Read the audit"),
        Node(id="write", label="Write the brief", depends_on=["poll", "audit"],
             metadata=dict(writer_metadata)),
    ])


def strict() -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[SourceEvidenceValidator(SOURCES)])


def ordinary(provider: Provider, **options) -> Swarm:
    return Swarm(model="test-model", provider=provider, architect=SimpleArchitect(), **options)


# ---------------------------------------------------------------------------
# The compact record
# ---------------------------------------------------------------------------


def test_the_compact_record_keeps_claims_citations_and_questions_and_drops_the_body():
    record = claim_exchange(POLL_OUTPUT)
    assert record == exchanged(POLL_EVIDENCE)
    assert record.startswith(CLAIM_EXCHANGE_NOTE + "\n```source-evidence\n")
    (block,) = _BLOCK.findall(record)
    data = json.loads(block)
    assert data["claims"] == [PREFERENCE.to_dict()]
    # Each source keeps its id, version and digest, and no text.
    assert [(item["source_id"], item["source_version"], item["sha256"], item["text"])
            for item in data["sources"]] == [("survey-2019", "2019", POLL.sha256, None),
                                             ("renewals-2026", "2026-q2", AUDIT.sha256, None)]
    assert "Working notes" not in record and POLL.text not in record
    assert len(record) < len(POLL_OUTPUT)
    # A tombstone is passed as it stands: identity and digest, nothing of the content.
    gone = SourceEvidence((SourceRecord.from_context_source(AUDIT).tombstoned("deleted"),),
                          (MAJORITY,))
    assert claim_exchange(AUDIT_NOTES + render_source_evidence(gone)) == exchanged(gone)


@pytest.mark.parametrize("output", [
    pytest.param(POLL_NOTES, id="no-evidence"),
    pytest.param(POLL_NOTES + render_source_evidence(SourceEvidence(POLL_EVIDENCE.sources)),
                 id="no-claims"),
    pytest.param(POLL_NOTES + "```source-evidence\n{\"version\": 1}\n```\n", id="invalid-block"),
    pytest.param(POLL_OUTPUT + render_source_evidence(AUDIT_EVIDENCE), id="two-blocks"),
    pytest.param({"claims": []}, id="not-text"),
])
def test_output_without_a_readable_claim_record_has_no_compact_record(output):
    assert claim_exchange(output) is None


# ---------------------------------------------------------------------------
# Dependency results
# ---------------------------------------------------------------------------


def completed(node_id: str, result) -> Node:
    return Node(id=node_id, label=node_id, result=result, status=NodeStatus.COMPLETED)


def base() -> ExecutorBase:
    return ExecutorBase(provider=Scripted(), registry=Registry(), tracer=Tracer(),
                        artifact_dir=None)


def test_a_node_exchanging_claims_is_given_compact_records_and_other_results_in_full():
    plain = completed("plain", "A dependency without evidence.")
    empty = completed("empty", POLL_NOTES + render_source_evidence(
        SourceEvidence(POLL_EVIDENCE.sources)))
    skipped = Node(id="skipped", label="skipped", result="error: provider failed",
                   status=NodeStatus.SKIPPED)
    drawn = completed("drawn", AUDIT_OUTPUT)
    drawn.metadata["artifacts"] = [{"path": "/runs/drawn_00.png", "mime_type": "image/png"}]
    deps = [completed("poll", POLL_OUTPUT), plain, empty, skipped, drawn]
    writer = Node(id="write", label="Write", depends_on=[dep.id for dep in deps],
                  metadata={"exchange": "claims"})
    graph = ExecutionGraph([Topology.FORK_JOIN], [*deps, writer])
    results = base().gather_dep_results(writer, graph)
    assert results == {
        "poll": exchanged(POLL_EVIDENCE),
        "plain": "A dependency without evidence.",
        "empty": empty.result,
        "skipped": SKIPPED_DEPENDENCY_RESULT,
        # Artifacts are exchanged beside the claims.
        "drawn": exchanged(AUDIT_EVIDENCE)
        + "\n\nArtifact files from this step:\n- /runs/drawn_00.png (image/png)",
    }


@pytest.mark.parametrize("metadata", [{}, {"role": "adversarial"}])
def test_a_node_that_does_not_opt_in_is_given_full_results(metadata):
    writer = Node(id="write", label="Write", depends_on=["poll"], metadata=metadata)
    graph = ExecutionGraph([Topology.SERIAL], [completed("poll", POLL_OUTPUT), writer])
    assert base().gather_dep_results(writer, graph) == {"poll": POLL_OUTPUT}


@pytest.mark.parametrize("mode", ["claim", "Claims", "transcript", None, ["claims"]])
def test_an_unknown_exchange_mode_is_refused(mode):
    writer = Node(id="write", label="Write", depends_on=["poll"], metadata={"exchange": mode})
    graph = ExecutionGraph([Topology.SERIAL], [completed("poll", POLL_OUTPUT), writer])
    with pytest.raises(ValueError, match="metadata 'exchange' must be 'claims'"):
        base().gather_dep_results(writer, graph)


# ---------------------------------------------------------------------------
# Ordinary runs
# ---------------------------------------------------------------------------


def test_a_writer_exchanging_claims_sees_citations_and_questions_and_no_transcript():
    provider = Scripted()
    result = ordinary(provider, acceptance=strict()).execute(research_graph(exchange="claims"))
    prompt = provider.prompts["Write the brief"]
    compact = {"poll": exchanged(POLL_EVIDENCE), "audit": exchanged(AUDIT_EVIDENCE)}
    assert prompt == ExecutorBase.build_user_prompt(result.graph.nodes[2], compact,
                                                    is_terminal=True)
    for claim in (PREFERENCE, MAJORITY):
        assert claim.text in prompt and all(question in prompt for question in claim.unresolved)
        for passage in claim.supporting + claim.contradicting:
            assert json.dumps(passage.quote) in prompt and json.dumps(passage.source_id) in prompt
    assert "Working notes" not in prompt and POLL.text not in prompt
    # The artifact built from the exchanged records alone passes every mechanical check.
    assert result.accepted
    assert [item.validator for item in result.acceptance.validators] == [
        SourceEvidenceValidator(SOURCES).identity]
    assert result.output == write_from_claims(prompt)


def test_prompts_change_only_for_nodes_that_opt_in():
    full = {"poll": POLL_OUTPUT, "audit": AUDIT_OUTPUT}
    baseline = Scripted()
    ordinary(baseline).execute(research_graph())
    graph = research_graph()
    assert baseline.prompts["Write the brief"] == ExecutorBase.build_user_prompt(
        graph.nodes[2], full, is_terminal=True)
    # A sibling that does not opt in is sent exactly what it was sent before.
    graph.nodes.append(Node(id="review", label="Review the brief", depends_on=["poll", "audit"]))
    graph.nodes[2].metadata["exchange"] = "claims"
    mixed = Scripted()
    ordinary(mixed).execute(graph)
    assert mixed.prompts["Review the brief"] == ExecutorBase.build_user_prompt(
        graph.nodes[3], full, is_terminal=True)
    assert mixed.prompts["Write the brief"] != baseline.prompts["Write the brief"]
    for label in ("Read the poll", "Read the audit"):
        assert mixed.prompts[label] == baseline.prompts[label]


def test_an_ordinary_run_refuses_an_unknown_exchange_mode_before_any_call():
    provider = Scripted()
    with pytest.raises(ValueError, match="Node 'write' metadata 'exchange' must be 'claims'"):
        ordinary(provider).execute(research_graph(exchange="claim"))
    assert provider.prompts == {}


# ---------------------------------------------------------------------------
# Durable runs
# ---------------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch):
    """Answer each step of a durable run from REPLIES; keep every prompt in order."""
    prompts: list[str] = []

    async def complete(self, system, prompt, model):
        prompts.append(prompt)
        reply = REPLIES[step(prompt)]
        return CompletionResult(reply(prompt) if callable(reply) else reply, cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return prompts


def durable(store, **options) -> Swarm:
    options.setdefault("acceptance", strict())
    options.setdefault("context_sources", SOURCES)
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def by_id(lineage) -> dict[str, dict]:
    return {node["id"]: node for node in lineage.to_dict()["nodes"]}


def journaled_prompt(store, run_id: str, node_id: str) -> str:
    """The prompt of a node's call, exactly as its journaled request binds it."""
    record = store.lookup_call(run_id, CallKey("execution", f"node/{node_id}"))
    return json.loads(record["request_json"])["prompt"]


def test_a_durable_run_binds_the_compact_records_its_writer_was_given(tmp_path, sent):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store).execute(research_graph(exchange="claims"))
        run_id = result.execution_id
        prompt = journaled_prompt(store, run_id, "write")
        # The snapshot names the exact text the writer saw: each compact record.
        snapshot = load_context_snapshot(store, run_id, "write", 0)
        assert {item.node_id: item.output_sha256 for item in snapshot.dependencies} == {
            "poll": text_sha256(exchanged(POLL_EVIDENCE)),
            "audit": text_sha256(exchanged(AUDIT_EVIDENCE))}
        assert store.audit(run_id)["ok"]
    assert prompt == sent[-1]
    assert exchanged(POLL_EVIDENCE) in prompt and exchanged(AUDIT_EVIDENCE) in prompt
    assert "Working notes" not in prompt
    # The final artifact passes the source evidence checks against the run's sources.
    assert result.accepted and result.output == write_from_claims(prompt)


def test_a_resumed_writer_is_given_the_compact_records_its_call_was_bound_to(
    tmp_path, monkeypatch, sent,
):
    image = tmp_path / "paused.db"
    with SQLiteWorkflowStore(tmp_path / "live.db") as store:
        writer = CallKey("execution", "node/write")

        def before_writer(lease, call_id, _):
            record = store.lookup_call(lease.run_id, writer)
            return record is not None and record["call_id"] == call_id

        point = fakes.crash_when(monkeypatch, store, "claim_dispatch", before_writer, image=image,
                                 connection=store._db)
        with pytest.raises(fakes.Crash):
            durable(store).execute(research_graph(exchange="claims"))
        (run,) = store.list_runs()
    assert point.fired and [step(prompt) for prompt in sent] == ["Read the poll", "Read the audit"]
    with SQLiteWorkflowStore(image, clock_ns=ManualClock.after(120)) as paused:
        bound = journaled_prompt(paused, run["run_id"], "write")
        result = durable(paused).resume(run["run_id"])
        assert paused.audit(run["run_id"])["ok"]
    # Only the writer was called, with the prompt its journaled call was bound to.
    assert sent[2:] == [bound] and exchanged(POLL_EVIDENCE) in bound
    assert result.accepted


def test_lineage_declares_a_writer_given_compact_records_of_the_recorded_results(
    tmp_path, monkeypatch, sent,
):
    built = []

    def recorded(graph, *args, **kwargs):
        # The same records, for a writer that does not exchange claims.
        plain = copy.deepcopy(graph)
        del plain.nodes[2].metadata["exchange"]
        built.append(by_id(build_lineage(plain, *args, **kwargs)))
        return build_lineage(graph, *args, **kwargs)

    monkeypatch.setattr(smythe.workflow, "build_lineage", recorded)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        result = durable(store, lineage=LineagePolicy()).execute(research_graph(exchange="claims"))
        lineage = load_lineage(store, result.execution_id)
    assert result.accepted and lineage.coverage == "complete"
    nodes = by_id(lineage)
    assert nodes["write"]["coverage"] == "declared"
    assert nodes["write"]["inputs"]["context"]["snapshot_key"] == "context/write/0"
    # Its snapshot names compact records, which only a node exchanging claims is given.
    (plain,) = built
    assert plain["write"]["coverage"] == "undeclared"


def test_a_durable_run_refuses_an_unknown_exchange_mode_before_any_call(tmp_path, sent):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        with pytest.raises(WorkflowBindingError, match="metadata 'exchange' must be 'claims'"):
            durable(store).execute(research_graph(exchange="transcript"))
        (run,) = store.list_runs()
        # Nothing was planned, saved or called.
        assert store.get_checkpoint(run["run_id"]) is None
        assert store.inspect_run(run["run_id"])["calls"] == []
    assert sent == []


# ---------------------------------------------------------------------------
# Generated plans
# ---------------------------------------------------------------------------


def test_a_generated_plan_cannot_set_the_exchange_key():
    plan = {"nodes": [{"id": "write", "label": "Write", "metadata": {"exchange": "claims"}}]}
    with pytest.raises(ValueError, match=r"unsupported field\(s\) \['exchange'\]"):
        build_graph_from_model_output(plan)


def test_a_planner_reply_that_sets_the_exchange_key_is_repaired_without_it():
    replies = [
        json.dumps({"nodes": [{"id": "read", "label": "Read"},
                              {"id": "write", "label": "Write", "depends_on": ["read"],
                               "metadata": {"exchange": "claims"}}]}),
        json.dumps({"nodes": [{"id": "read", "label": "Read"},
                              {"id": "write", "label": "Write", "depends_on": ["read"]}]}),
    ]
    prompts: list[str] = []

    class Planning(Provider):
        async def complete(self, system, prompt, model):
            prompts.append(prompt)
            return CompletionResult(replies.pop(0), cost_usd=0.0)

    graph, _ = LLMArchitect(Planning(), planning_model="test-model", max_retries=1).plan(
        Task("Write a billing brief"))
    assert "['exchange']" in prompts[1]
    assert all("exchange" not in node.metadata for node in graph.nodes)
