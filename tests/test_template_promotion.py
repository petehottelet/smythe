"""Promoting structure that runs proved to a candidate template record.

Planner history groups accepted runs by their graph's structure digest. A
structure that enough accepted runs share becomes a candidate record whose
source runs name those runs and whose failure examples keep only a category
and an evidence digest for each failed or not-accepted run of that structure.
A candidate is only a candidate: nothing is activated. A record also builds
its graph as a deterministic architect, with no planning call, which a
durable run pins by identity and version.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json

import pytest

import smythe
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.acceptance import AcceptancePolicy
from smythe.constrained_planner import ConstrainedArchitect
from smythe.constrained_prompts import CONSTRAINED_SYSTEM_PROMPT
from smythe.distill import DistillationError
from smythe.graph import ExecutionGraph, Node, Topology, structure_sha256
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.optimize import PolicyRegistry
from smythe.planner import DeterministicArchitect
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.swarm import Swarm
from smythe.task import Task
from smythe.templates import (
    MAX_FAILURE_EXAMPLES, TemplateArchitect, TemplateCatalog, TemplateRecord,
    candidate_from_graph, distill_record, repeated_structures,
)
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding
from smythe.workflow_binding import LocalOnly
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowConflictError
from test_distill import _completed_gated_graph, _completed_graph

MARKER = "DEFECT-4c1e"
GOALS = ("Summarize the quarter", "Review the annual budget", "Plan the offsite")
# The row fields a failure example's evidence digest covers, each when present.
EVIDENCE = ("execution_id", "structure_sha256", "acceptance", "verification",
            "rejected_output_sha256", "failure_category", "failure_type")


class RateLimitError(Exception):
    """An SDK-style status error whose message quotes the provider's response."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 429


class Steps(Provider):
    """Answers a Draft step with *draft* and a Judge step with *verdict*; an exception is raised."""

    def __init__(self, draft, verdict="PASS") -> None:
        self.draft, self.verdict = draft, verdict

    async def complete(self, system, prompt, model):
        step = next(line for line in prompt.splitlines() if line.startswith("Your step: "))
        answer = self.draft if step.startswith("Your step: Draft") else self.verdict
        if isinstance(answer, BaseException):
            raise answer
        return CompletionResult(text=answer, prompt_tokens=1, completion_tokens=1)


class Forbids:
    """A final validator that rejects delivered text holding the marker."""

    identity = ValidatorIdentity.for_config("forbids", "1", {"marker": MARKER})

    def validate(self, subject):
        findings = ((hard_finding("forbidden_text", "the output holds the marker"),)
                    if MARKER in subject.text() else ())
        return ValidationReport("output", findings, validator=self.identity)


def memo(task: Task | None) -> ExecutionGraph:
    """A draft that a judge gates."""
    return ExecutionGraph(topology=[Topology.SERIAL], task=task, nodes=[
        Node(id="draft", label="Draft the memo"),
        Node(id="judge", label="Judge the memo", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])


def run(memory, goal, draft="A memo.", *, strict=True):
    """Run the memo graph for *goal*; a strict run accepts output without the marker."""
    acceptance = AcceptancePolicy(mode="strict", final_validators=[Forbids()]) if strict else None
    swarm = Swarm(model="m", provider=Steps(draft), memory=memory, artifact_dir=None,
                  acceptance=acceptance)
    return swarm.execute(memo(Task(goal)))


def evidence(row: ExecutionOutcome) -> str:
    return sha256_hex({key: getattr(row, key) for key in EVIDENCE if getattr(row, key) is not None})


def outcome(**fields) -> ExecutionOutcome:
    """A history row of the memo structure; *fields* replace its defaults."""
    values = dict(task_goal="Summarize the quarter", task_constraints=[], topology=["serial"],
                  node_count=2, total_cost_usd=0.0, total_duration_ms=0.0, success=True,
                  acceptance="accepted", execution_id="run-0",
                  structure_sha256=structure_sha256(memo(None)))
    return ExecutionOutcome(**{**values, **fields})


def write(path, rows) -> None:
    path.write_text("".join(json.dumps(row if isinstance(row, (dict, list)) else asdict(row)) + "\n"
                            for row in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading history
# ---------------------------------------------------------------------------


def test_iter_outcomes_reads_each_valid_row_oldest_first_and_never_writes(tmp_path):
    path = tmp_path / "history.jsonl"
    memory = PlannerMemory(path)
    assert list(memory.iter_outcomes()) == [] and not path.exists()

    first, second = outcome(execution_id="run-1"), outcome(execution_id="run-2", acceptance="rejected")
    malformed = [dict(asdict(outcome()), acceptance="great"), dict(asdict(outcome()), success="yes"),
                 dict(asdict(outcome()), structure_sha256="abc"),
                 dict(asdict(outcome()), failure_category="Not A Class"), ["a", "list"]]
    write(path, [first, *malformed, second])
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n   \n{not json\n")
    before = path.read_bytes()

    rows = memory.iter_outcomes()
    path.unlink()  # The file is read when iter_outcomes is called.

    assert list(rows) == [first, second]
    path.write_bytes(before)
    # Recall skips exactly the rows iter_outcomes skips.
    assert memory.recall(Task("Summarize the quarter"), k=10) == [first, second]
    assert list(memory.iter_outcomes()) == [first, second]
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Repeated accepted structure
# ---------------------------------------------------------------------------


def test_accepted_runs_of_one_structure_yield_a_candidate(tmp_path):
    path = tmp_path / "history.jsonl"
    memory = PlannerMemory(path)
    results = [run(memory, goal) for goal in GOALS]
    digest = structure_sha256(results[0].graph)
    before = path.read_bytes()

    repeated = repeated_structures(memory, min_accepted=3)
    record = candidate_from_graph(results[-1].graph, rows=memory.iter_outcomes(), name="memo",
                                  min_accepted=3)

    assert [result.acceptance.decision for result in results] == ["accepted"] * 3
    assert list(repeated) == [digest]
    assert [row.execution_id for row in repeated[digest]] == [r.execution_id for r in results]
    assert all(type(row) is ExecutionOutcome for row in repeated[digest])
    assert repeated_structures(memory, min_accepted=4) == {}
    assert [dict(item) for item in record.source_runs] == [
        {"execution_id": result.execution_id, "structure_sha256": digest, "acceptance": "accepted"}
        for result in results]
    assert record.failure_examples == ()
    # The candidate is the distilled graph with its evidence; evidence never changes its version.
    distilled = distill_record(results[-1].graph, name="memo")
    assert record.template_version == distilled.template_version
    assert record.to_dict() == {**distilled.to_dict(), "source_runs": [
        dict(item) for item in record.source_runs]}
    nodes, _ = record.to_template().builder(Task("Draft the next memo"))
    assert [(node.id, node.verifies) for node in nodes] == [("draft", None), ("judge", "draft")]
    assert path.read_bytes() == before


def test_a_candidate_is_stored_only_as_a_candidate(tmp_path):
    memory = PlannerMemory(tmp_path / "history.jsonl")
    result = [run(memory, goal) for goal in GOALS][-1]
    catalog = TemplateCatalog(tmp_path / "catalog")

    record = candidate_from_graph(result.graph, rows=memory.iter_outcomes(), name="memo",
                                  min_accepted=3)
    assert not (tmp_path / "catalog").exists()
    version = catalog.put(record)

    assert catalog.get(version) == record and catalog.versions("memo") == [version]
    with PolicyRegistry(tmp_path / "registry.sqlite3") as registry:
        with pytest.raises(KeyError, match="no version activated"):
            catalog.active_templates(registry, ["memo"])


def test_completed_but_rejected_runs_do_not_count(tmp_path):
    memory = PlannerMemory(tmp_path / "history.jsonl")
    accepted = [run(memory, goal) for goal in GOALS[:2]]
    rejected = [run(memory, goal, f"A memo with {MARKER}") for goal in GOALS]
    unevaluated = run(memory, GOALS[0], strict=False)
    rows = list(memory.iter_outcomes())

    assert [(row.success, row.acceptance) for row in rows] == [
        (True, "accepted")] * 2 + [(True, "rejected")] * 3 + [(True, "not_evaluated")]
    assert {row.structure_sha256 for row in rows} == {structure_sha256(unevaluated.graph)}
    assert repeated_structures(memory, min_accepted=3) == {}
    with pytest.raises(DistillationError, match="2 accepted runs of this structure, fewer than 3"):
        candidate_from_graph(rejected[0].graph, rows=rows, name="memo", min_accepted=3)

    record = candidate_from_graph(rejected[0].graph, rows=rows, name="memo", min_accepted=2)
    assert [item["execution_id"] for item in record.source_runs] == [
        result.execution_id for result in accepted]
    # A rejected run is negative evidence; a run that was not evaluated is neither.
    assert [dict(item) for item in record.failure_examples] == [
        {"category": "rejected", "evidence_sha256": evidence(row)} for row in rows[2:5]]


def test_failure_rows_become_failure_examples_without_text(tmp_path):
    path = tmp_path / "history.jsonl"
    memory = PlannerMemory(path, record_failures=True)
    accepted = run(memory, GOALS[0])
    rejected = run(memory, GOALS[1], f"A memo with {MARKER}")
    with pytest.raises(RateLimitError):
        run(memory, GOALS[2], RateLimitError(f"rate limited: {MARKER}"))
    rows = list(memory.iter_outcomes())
    failed = rows[2]

    record = candidate_from_graph(accepted.graph, rows=rows, name="memo")

    # History keeps the evidence, including the provider's message.
    assert MARKER in rejected.rejected_output and MARKER in failed.failure_excerpt
    assert failed.failure_category and failed.failure_type == "RateLimitError"
    assert rows[1].rejected_output_sha256 and rows[1].acceptance == "rejected"
    assert [dict(item) for item in record.failure_examples] == [
        {"category": "rejected", "evidence_sha256": evidence(rows[1])},
        {"category": failed.failure_category, "evidence_sha256": evidence(failed)},
    ]
    text = canonical_bytes(record.to_dict()).decode("utf-8")
    for secret in (MARKER, "rate limited", "RateLimitError", *GOALS):
        assert secret not in text
    assert [item["execution_id"] for item in record.source_runs] == [accepted.execution_id]


def test_failure_examples_keep_the_latest_distinct_evidence(tmp_path):
    path = tmp_path / "history.jsonl"
    failures = [outcome(execution_id=f"run-{index}", acceptance="not_evaluated",
                        failure_category="transient", failure_type="RuntimeError")
                for index in range(1, MAX_FAILURE_EXAMPLES + 5)]
    # A row with the same evidence as an earlier one counts once, at its latest place.
    repeat = outcome(execution_id="run-2", acceptance="not_evaluated",
                     failure_category="transient", failure_type="RuntimeError",
                     task_goal="Another goal", total_cost_usd=1.0)
    write(path, [outcome(), *failures, repeat])

    record = candidate_from_graph(memo(None), rows=PlannerMemory(path).iter_outcomes(),
                                  name="memo", require_success=False)

    kept = [*failures[5:], failures[1]][-MAX_FAILURE_EXAMPLES:]
    assert len(failures) > MAX_FAILURE_EXAMPLES
    assert [item["evidence_sha256"] for item in record.failure_examples] == [
        evidence(row) for row in kept]
    assert evidence(repeat) == evidence(failures[1])


def test_only_identified_accepted_runs_of_the_same_structure_count(tmp_path):
    other = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="draft", label="Draft")])
    rows = [
        outcome(execution_id="run-1"),
        outcome(execution_id="run-1", task_goal="The same run recorded again"),
        outcome(execution_id=None),
        outcome(execution_id="run-2", failure_category="transient", failure_type="RuntimeError"),
        outcome(execution_id="run-3", structure_sha256=structure_sha256(other)),
        outcome(execution_id="run-4", structure_sha256=None),
        outcome(execution_id="run-5", acceptance=None),
        outcome(execution_id="run-6", acceptance="needs_review"),
        outcome(execution_id="run-7", acceptance="indeterminate", success=False),
        outcome(execution_id="run-8"),
    ]
    path = tmp_path / "history.jsonl"
    write(path, rows)

    repeated = repeated_structures(PlannerMemory(path), min_accepted=1)
    record = candidate_from_graph(memo(None), rows=rows, name="memo", require_success=False)

    assert {digest: [row.execution_id for row in found] for digest, found in repeated.items()} == {
        structure_sha256(memo(None)): ["run-1", "run-8"], structure_sha256(other): ["run-3"]}
    assert [item["execution_id"] for item in record.source_runs] == ["run-1", "run-8"]
    # A failure row is negative evidence whatever acceptance it records.
    assert [item["category"] for item in record.failure_examples] == [
        "transient", "needs_review", "indeterminate"]


def test_label_review_findings_carry_over_to_the_candidate(tmp_path):
    memory = PlannerMemory(tmp_path / "history.jsonl")
    result = run(memory, "Draft the quarterly memo")

    record = candidate_from_graph(result.graph, rows=memory.iter_outcomes(), name="memo")

    assert record.to_dict()["review"]["label_findings"] == [
        {"node": "draft", "contains_goal": False, "words": ["draft", "memo"]}]


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2", None])
def test_min_accepted_must_be_a_positive_integer(tmp_path, value):
    memory = PlannerMemory(tmp_path / "history.jsonl")
    with pytest.raises(ValueError, match="min_accepted must be a positive integer"):
        repeated_structures(memory, min_accepted=value)
    with pytest.raises(ValueError, match="min_accepted must be a positive integer"):
        candidate_from_graph(memo(None), rows=[outcome()], name="memo", min_accepted=value,
                             require_success=False)


def test_candidate_rows_must_be_history_outcomes():
    with pytest.raises(TypeError, match="ExecutionOutcome"):
        candidate_from_graph(memo(None), rows=[asdict(outcome())], name="memo",
                             require_success=False)


def test_a_graph_without_a_structure_digest_is_refused():
    graph = memo(None)
    graph.nodes[0].metadata["role"] = object()
    with pytest.raises(DistillationError, match="structure"):
        candidate_from_graph(graph, rows=[outcome()], name="memo", require_success=False)


def test_promotion_names_are_exported():
    for name in ("candidate_from_graph", "repeated_structures"):
        assert name in smythe.__all__ and getattr(smythe, name) is getattr(smythe.templates, name)


# ---------------------------------------------------------------------------
# Deterministic templates
# ---------------------------------------------------------------------------


class Selects(Provider):
    """A template-selection model that picks the named templates in order."""

    def __init__(self, *names: str) -> None:
        self.names = names

    async def complete(self, system, prompt, model):
        return CompletionResult(json.dumps([{"template": name} for name in self.names]))


class NoCalls(Provider):
    async def complete(self, system, prompt, model):
        raise AssertionError("a deterministic template makes no provider call")


def gated_record() -> TemplateRecord:
    graph, registry = _completed_gated_graph()
    return distill_record(graph, name="memo", registry=registry)


def shape(graph, registry):
    """What a plan builds, with agents by their profile rather than their random ids."""
    agents = {agent.id: agent.profile for agent in registry.list_agents()}
    return {
        "structure": structure_sha256(graph), "topology": graph.topology,
        "planning": graph.planning, "agents": len(agents),
        "nodes": [(node.id, node.label, node.depends_on, node.verifies, node.status,
                   node.required_capabilities, node.metadata, node.max_regenerations,
                   (agents[node.agent_id].name, agents[node.agent_id].persona,
                    agents[node.agent_id].capabilities))
                  for node in graph.nodes],
    }


def test_as_architect_builds_what_selecting_the_template_builds_without_a_call():
    record = gated_record()
    task = Task("Write the quarterly memo")
    selected = ConstrainedArchitect(Selects("memo"), [record.to_template()]).plan(task)

    architect = record.as_architect()
    built = architect.plan(task)
    again = asyncio.run(architect.aplan(task))

    assert type(architect) is TemplateArchitect and isinstance(architect, DeterministicArchitect)
    assert architect.record is record
    assert shape(*built) == shape(*selected) == shape(*again)
    assert [node.id for node in built[0].nodes] == ["memo-0-draft", "memo-0-check"]
    assert built[0].planning == {"templates": [
        {"name": "memo", "index": 0, "version": record.template_version}]}
    # Each plan builds its own nodes and agents.
    assert built[0].nodes[0] is not again[0].nodes[0]
    assert {agent.id for agent in built[1].list_agents()}.isdisjoint(
        agent.id for agent in again[1].list_agents())

    graph = Swarm(model="m", provider=NoCalls(), architect=architect, artifact_dir=None).plan(task)
    assert [node.id for node in graph.nodes] == ["memo-0-draft", "memo-0-check"]
    assert graph.planning["templates"] == built[0].planning["templates"]


def test_as_architect_builds_a_record_of_any_size():
    blueprint = [{"id": f"step-{index}", "label": f"Step {index}",
                  "depends_on": [f"step-{index - 1}"] if index else [], "capabilities": [],
                  "profile": None, "verifies": None, "max_regenerations": 0,
                  "failure_policy": "halt", "max_retries": 0, "timeout_s": None,
                  "attach_dep_artifacts": False, "metadata": {}} for index in range(80)]
    record = TemplateRecord(name="long", description="Eighty steps", blueprint=blueprint)

    graph, _ = record.as_architect().plan(Task("Do the long thing"))

    assert len(graph.nodes) == 80 and graph.nodes[-1].depends_on == ["long-0-step-78"]


def test_a_template_architect_builds_only_a_record():
    with pytest.raises(TypeError, match="TemplateRecord"):
        TemplateArchitect(gated_record().to_template())


def test_as_architect_pins_the_template_identity_and_version_in_durable_runs(
    tmp_path, monkeypatch,
):
    systems = []

    async def complete(self, system, prompt, model):
        systems.append(system)
        return CompletionResult("Step done", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    graph, registry = _completed_graph()
    first = distill_record(graph, name="learned", registry=registry)
    changed = first.to_dict()
    changed["blueprint"][0]["max_retries"] = 2
    second = TemplateRecord.from_dict(changed)
    declaration = first.as_architect().local_only()

    def durable(record, store):
        return Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                     architect=record.as_architect().local_only())

    assert declaration == LocalOnly(object, "template:learned", first.template_version)
    assert declaration.role == "architect" and type(declaration.factory()) is TemplateArchitect
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        recipes = [durable(record, store)._workflow_runtime().recipe
                   for record in (first, first, second)]
        assert recipes[0] == recipes[1] != recipes[2]
        assert recipes[0]["components"]["architect"] == {
            "type": "local_only", "role": "architect", "identity": "template:learned",
            "version": first.template_version}

        run_id = durable(first, store).plan(Task("Assess gadgets")).run_ref["run_id"]
        with pytest.raises(WorkflowConflictError, match="saved recipe"):
            durable(second, store).resume(run_id)
        result = durable(first, store).resume(run_id)
        saved = store.load_run(run_id)

    assert [node.id for node in result.graph.nodes] == ["learned-0-research", "learned-0-review"]
    assert result.output == "Step done" and saved["status"] == "completed"
    assert result.graph.planning == {"templates": [
        {"name": "learned", "index": 0, "version": first.template_version}]}
    # Only the steps called a provider: planning never did.
    assert systems and not {PLANNING_SYSTEM_PROMPT, CONSTRAINED_SYSTEM_PROMPT} & set(systems)
