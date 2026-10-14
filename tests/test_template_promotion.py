"""Promoting structure that runs proved to a candidate template record.

Planner history groups accepted runs by their graph's structure digest. A
structure that enough accepted runs share becomes a candidate record whose
source runs name those runs and whose failure examples keep only a category
and an evidence digest for each failed or not-accepted run of that structure.
A candidate is only a candidate: nothing is activated. A record also builds
its graph as a deterministic architect, with no planning call, which a
durable run pins by identity and version. A candidate built in an
environment snapshot no longer applies, and is no longer offered, once a
tool, capability, validator, skill, model or MCP allowlist it relied on
changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import json
import logging
import typing

import pytest

import smythe
from smythe._canonical import canonical_bytes, sha256_hex
from smythe.acceptance import AcceptancePolicy
from smythe.agent import Agent, AgentProfile
from smythe.agent_skills import AgentSkillsProvider
from smythe.constrained_planner import ConstrainedArchitect
from smythe.constrained_prompts import CONSTRAINED_SYSTEM_PROMPT
from smythe.distill import DistillationError
from smythe.graph import ExecutionGraph, Node, Topology, structure_sha256
from smythe.local_tools import CompositeToolRuntime, LocalTool, LocalToolRuntime
from smythe.mcp import MCPServerSpec, MCPToolRuntime
from smythe.memory import ExecutionOutcome, PlannerMemory
from smythe.optimize import PolicyRegistry
from smythe.planner import DeterministicArchitect
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.registry import Registry
from smythe.skills import SkillRef
from smythe.swarm import Swarm
from smythe.task import Task
from smythe.templates import (
    APPLICABILITY_REASONS, MAX_FAILURE_EXAMPLES, ApplicabilityReport, TemplateApplicabilityError,
    TemplateArchitect, TemplateCatalog, TemplateEnvironment, TemplateRecord, TemplateRecordError,
    TemplateValidationError, candidate_from_graph, check_applicability, distill_record,
    repeated_structures, validate_template_record,
)
from smythe.tools import ToolSpec
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding
from smythe.workflow_binding import LocalOnly
from smythe.workflow_store import SQLiteWorkflowStore, WorkflowConflictError
from test_distill import _completed_gated_graph, _completed_graph
from test_template_lifecycle import _in_effect

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


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}}
MODEL = "gpt-6-astra"
TARGET = "planner.template.memo"


def skill_file(root, body: str) -> None:
    folder = root / "memo-style"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: memo-style\ndescription: How memos are written\n---\n{body}\n",
        encoding="utf-8")


def environment(tmp_path, *, schema=SCHEMA, tools=True, capabilities=("writing",),
                validator="1", body="Write short sentences.", skills=True, models=(MODEL,),
                allowed=("read_file",)) -> TemplateEnvironment:
    """Capture an environment from live objects; each argument changes one part of it."""
    root = tmp_path / "skills" / sha256_hex([body, skills])[:12]
    if skills:
        skill_file(root, body)
    root.mkdir(parents=True, exist_ok=True)
    registry = Registry(skill_provider=AgentSkillsProvider({"writer": [root]}))
    registry.register(Agent(id="writer", profile=AgentProfile(
        name="Writer", capabilities=list(capabilities))))
    registry.register(Agent(id="reader", profile=AgentProfile(
        name="Reader", capabilities=["research"],
        mcp_servers=[MCPServerSpec("fs", "stdio", command="fs-server", allowed_tools=allowed)])))
    search = LocalTool(ToolSpec("search", "Search the notes", schema), lambda arguments: "found", "1")
    runtime = LocalToolRuntime([search] if tools else [], agents={"writer": ["search"] if tools else []})
    identity = ValidatorIdentity.for_config("forbids", validator, {"marker": MARKER})
    return TemplateEnvironment.capture(registry=registry, tool_runtime=runtime,
                                       validators=[identity], models=models)


def proven(tmp_path) -> TemplateRecord:
    """A candidate whose draft step needs a writer, built in the captured environment."""
    graph = memo(None)
    graph.nodes[0].required_capabilities = ["writing"]
    graph.nodes[1].required_capabilities = ["judging"]
    row = outcome(structure_sha256=structure_sha256(graph))
    return candidate_from_graph(graph, rows=[row], name="memo", require_success=False,
                                environment=environment(tmp_path))


def test_capture_reads_what_a_run_would_use(tmp_path):
    captured = environment(tmp_path)
    search = LocalTool(ToolSpec("search", "Search the notes", SCHEMA), lambda arguments: "x", "1")
    other = LocalTool(ToolSpec("lookup", "Look up a word", {"type": "object"}), str, "3")
    composite = CompositeToolRuntime(LocalToolRuntime([search], agents={}), MCPToolRuntime(),
                                     LocalToolRuntime([other], agents={}))

    assert captured.capabilities == {"writing", "memo-style", "research"}
    assert dict(captured.skills) == {
        "memo-style": (AgentSkillsProvider({"writer": [tmp_path / "skills" / sha256_hex(
            ["Write short sentences.", True])[:12]]}).skill("memo-style").version,)}
    assert captured.skills["memo-style"][0].startswith("sha256:") and captured.skills_complete
    assert {name: dict(tool) for name, tool in captured.tools.items()} == {"search": search.describe()}
    assert dict(captured.mcp_allowlists) == {"fs": ("read_file",)}
    assert captured.validators == {ValidatorIdentity.for_config("forbids", "1", {"marker": MARKER})}
    assert captured.models == {MODEL}
    assert TemplateEnvironment.capture().capabilities == frozenset()
    assert set(TemplateEnvironment.capture(tool_runtime=composite).tools) == {"search", "lookup"}
    # Validators may be given as validators; their identities are kept.
    assert TemplateEnvironment.capture(validators=[Forbids()]).validators == {Forbids.identity}


def test_capture_merges_the_allowlists_of_one_server():
    def reader(name, allowed):
        server = MCPServerSpec("fs", "stdio", command="fs-server", allowed_tools=allowed)
        return Agent(profile=AgentProfile(name=name, mcp_servers=[server]))

    registry = Registry()
    for name, allowed in (("one", ("read_file",)), ("two", ("list", "read_file"))):
        registry.register(reader(name, allowed))
    assert dict(TemplateEnvironment.capture(registry=registry).mcp_allowlists) == {
        "fs": ("list", "read_file")}
    registry.register(reader("three", None))  # Every tool of the server.
    assert dict(TemplateEnvironment.capture(registry=registry).mcp_allowlists) == {"fs": None}


class Lists:
    """A skill provider listing fixed ``(name, version)`` skills per agent; it raises for *fails*."""

    def __init__(self, skills, *, fails=()) -> None:
        self.skills, self.fails = skills, set(fails)

    def list_agent_skills(self, agent_id):
        if agent_id in self.fails:
            raise RuntimeError("the skill service is unavailable")
        return [SkillRef(name, version=version) for name, version in self.skills.get(agent_id, ())]


def agents(*ids, provider=None) -> Registry:
    registry = Registry(skill_provider=provider)
    for agent_id in ids:
        registry.register(Agent(id=agent_id, profile=AgentProfile(name=agent_id.title())))
    return registry


def needing(*skills) -> TemplateRecord:
    """The memo record whose prerequisites name each ``(name, version)`` skill."""
    return replace(distill_record(memo(None), name="memo", require_success=False), prerequisites={
        "capabilities": [], "skills": [{"name": name, "version": version} for name, version in skills],
        "tools": [], "validators": []})


def test_capture_keeps_every_version_a_skill_is_listed_at():
    provider = Lists({"writer": [("memo-style", "2")], "editor": [("memo-style", "1")]})
    for order in (("writer", "editor"), ("editor", "writer")):
        captured = TemplateEnvironment.capture(registry=agents(*order, provider=provider))
        assert dict(captured.skills) == {"memo-style": ("1", "2")} and captured.skills_complete

    # A skill must be listed at exactly the versions a record names.
    assert check_applicability(needing(("memo-style", "1"), ("memo-style", "2")), captured).valid
    for record, environment in ((needing(("memo-style", "2")), captured),
                                (needing(("memo-style", "1"), ("memo-style", "2")),
                                 TemplateEnvironment(skills={"memo-style": "2"}))):
        assert check_applicability(record, environment) == ApplicabilityReport(
            False, ("skill_changed",),
            ("skill 'memo-style' is not listed at exactly the versions the template relies on",))
    # A candidate records each version it relied on.
    candidate = candidate_from_graph(memo(None), rows=[outcome()], name="memo",
                                     require_success=False, environment=captured)
    assert candidate.to_dict()["prerequisites"]["skills"] == [
        {"name": "memo-style", "version": "1"}, {"name": "memo-style", "version": "2"}]
    assert check_applicability(candidate, captured).valid


def test_skills_a_provider_cannot_list_leave_templates_that_name_skills_unoffered(
    tmp_path, caplog,
):
    provider = Lists({"writer": [("memo-style", "2")]}, fails={"editor"})
    with caplog.at_level(logging.WARNING, logger="smythe.templates"):
        captured = TemplateEnvironment.capture(registry=agents("writer", "editor",
                                                              provider=provider))
    assert captured.skills_complete is False and dict(captured.skills) == {"memo-style": ("2",)}
    warnings = [entry.getMessage() for entry in caplog.records if entry.name == "smythe.templates"]
    assert len(warnings) == 1 and "'editor'" in warnings[0]

    record = needing(("memo-style", "2"))
    report = check_applicability(record, captured)
    assert report == ApplicabilityReport(False, ("skills_unreadable",), (
        "the skills agents offer could not all be read, and the template relies on skills",))
    assert "skills_unreadable" in APPLICABILITY_REASONS
    # A record that names no skill still applies.
    assert check_applicability(replace(record, prerequisites=None), captured).valid
    catalog = TemplateCatalog(tmp_path / "catalog")
    catalog.put(record)
    with _in_effect(tmp_path / "registry.sqlite3",
                    {TARGET: {TARGET: record.template_version}}) as registry:
        assert catalog.active_templates(registry, ["memo"], environment=captured) == []
        with pytest.raises(TemplateApplicabilityError, match="could not all be read"):
            catalog.active_local_templates(registry, ["memo"], environment=captured, strict=True)
    # Nor can a candidate record what it relied on from such a snapshot.
    with pytest.raises(ValueError, match="skills could not all be read"):
        candidate_from_graph(memo(None), rows=[outcome()], name="memo", require_success=False,
                             environment=captured)
    assert TemplateEnvironment().skills_complete is True


def test_capture_reads_providers_and_composite_runtimes_through_public_accessors():
    provider = Lists({"writer": [("memo-style", "2")]})
    registry = agents("writer", provider=provider)
    assert registry.skill_provider is provider and Registry().skill_provider is None
    with pytest.raises(AttributeError):
        registry.skill_provider = None  # type: ignore[misc]

    search = LocalTool(ToolSpec("search", "Search the notes", SCHEMA), lambda arguments: "x", "1")
    other = LocalTool(ToolSpec("lookup", "Look up a word", {"type": "object"}), str, "3")
    local, mcp = LocalToolRuntime([search], agents={}), MCPToolRuntime()
    inner = CompositeToolRuntime(local, mcp)
    outer = CompositeToolRuntime(inner, LocalToolRuntime([other], agents={}))
    assert inner.runtimes == (local, mcp) and outer.runtimes[0] is inner
    assert set(TemplateEnvironment.capture(tool_runtime=outer).tools) == {"search", "lookup"}


def test_a_candidate_built_in_an_environment_records_what_it_relied_on(tmp_path):
    captured = environment(tmp_path)
    record = proven(tmp_path)
    data = record.to_dict()

    # Only the capability a node without an agent needs and an agent offers.
    assert data["prerequisites"] == {
        "capabilities": ["writing"],
        "skills": [{"name": "memo-style", "version": captured.skills["memo-style"][0]}],
        "tools": [dict(captured.tools["search"])],
        "validators": [ValidatorIdentity.for_config("forbids", "1", {"marker": MARKER}).to_dict()]}
    assert data["proven_with"] == {"models_sha256": sha256_hex([MODEL]),
                                   "mcp_allowlists_sha256": sha256_hex({"fs": ["read_file"]})}
    assert validate_template_record(data) == record == TemplateRecord.from_dict(data)
    assert check_applicability(record, captured) == ApplicabilityReport(True)
    assert check_applicability(record, environment(tmp_path, capabilities=("writing", "edit"),
                                                   validator="1")).valid
    # A record without prerequisites applies anywhere.
    assert check_applicability(distill_record(memo(None), name="memo", require_success=False),
                               TemplateEnvironment()).valid


CHANGES = {
    "tool-schema": (dict(schema={"type": "object", "properties": {"q": {"type": "string"}}}),
                    "tool_changed"),
    "tool-removed": (dict(tools=False), "tool_removed"),
    "capability-removed": (dict(capabilities=()), "capability_removed"),
    "validator-version": (dict(validator="2"), "validator_set_changed"),
    "skill-digest": (dict(body="Write long sentences."), "skill_changed"),
    "skill-removed": (dict(skills=False), "skill_removed"),
    "model-id": (dict(models=("gpt-6-astra-mini",)), "model_changed"),
    "model-added": (dict(models=(MODEL, "gpt-6-astra-mini")), "model_changed"),
    "mcp-allowlist": (dict(allowed=("read_file", "write_file")), "mcp_allowlist_changed"),
}


@pytest.mark.parametrize("case", CHANGES)
def test_a_changed_environment_invalidates_the_template_with_a_named_reason(
    tmp_path, case, caplog,
):
    changes, reason = CHANGES[case]
    record = proven(tmp_path)
    catalog = TemplateCatalog(tmp_path / "catalog")
    catalog.put(record)
    before, after = environment(tmp_path), environment(tmp_path, **changes)

    report = check_applicability(record, after)

    assert (report.valid, report.reasons) == (False, (reason,)) and len(report.details) == 1
    assert reason in APPLICABILITY_REASONS
    with _in_effect(tmp_path / "registry.sqlite3",
                    {TARGET: {TARGET: record.template_version}}) as registry:
        assert [item.name for item in catalog.active_templates(
            registry, ["memo"], environment=before)] == ["memo"]
        with caplog.at_level(logging.WARNING, logger="smythe.templates"):
            assert catalog.active_templates(registry, ["memo"], environment=after) == []
            assert catalog.active_local_templates(registry, ["memo"], environment=after) == []
        for active in (catalog.active_templates, catalog.active_local_templates):
            with pytest.raises(TemplateApplicabilityError) as caught:
                active(registry, ["memo"], environment=after, strict=True)
            assert (caught.value.name, caught.value.version, caught.value.report) == (
                "memo", record.template_version, report)
        # Without an environment, the version in effect is offered as before.
        assert len(catalog.active_templates(registry, ["memo"])) == 1
    warnings = [entry.getMessage() for entry in caplog.records if entry.name == "smythe.templates"]
    assert len(warnings) == 2 and all(report.details[0] in text and "'memo'" in text
                                      for text in warnings)


def test_every_reason_is_reported_once_with_each_detail(tmp_path):
    record = proven(tmp_path)
    expected = [{"node": "draft", "description": "The memo",
                 "validators": [ValidatorIdentity.for_config("length", "1").to_dict()]}]
    record = replace(record, expected_outputs=expected)
    after = environment(tmp_path, tools=False, validator="2", models=("other",), skills=False)

    report = check_applicability(record, after)

    assert report.reasons == ("model_changed", "skill_removed", "tool_removed",
                              "validator_set_changed")
    assert len(report.details) == 5 and not report.valid
    assert tuple(sorted(APPLICABILITY_REASONS)) == APPLICABILITY_REASONS


def test_proven_with_is_part_of_the_template_version(tmp_path):
    record = proven(tmp_path)
    data = record.to_dict()
    other = dict(data, proven_with={**data["proven_with"], "models_sha256": sha256_hex(["other"])})
    bare = {key: value for key, value in data.items() if key != "proven_with"}

    assert len({TemplateRecord.from_dict(item).template_version
                for item in (data, other, bare)}) == 3
    for bad, error in (({**data["proven_with"], "models": [MODEL]}, "unknown keys"),
                       ({"models_sha256": sha256_hex([MODEL])}, "missing keys"),
                       ({**data["proven_with"], "models_sha256": MODEL}, "hexadecimal")):
        with pytest.raises(TemplateRecordError, match=error):
            TemplateRecord.from_dict(dict(data, proven_with=bad))
    # A model key inside the section would grant authority and is refused.
    with pytest.raises(TemplateValidationError, match="authority"):
        validate_template_record(dict(data, proven_with={**data["proven_with"], "model": MODEL}))


@pytest.mark.parametrize("changes, error", [
    (dict(capabilities="writing"), TypeError),
    (dict(capabilities=[""]), ValueError),
    (dict(skills={"memo-style": None}), ValueError),
    (dict(skills={"memo-style": []}), ValueError),
    (dict(skills={"memo-style": 2}), TypeError),
    (dict(skills_complete="false"), TypeError),
    (dict(tools=[{"name": "search", "version": "1"}]), ValueError),
    (dict(tools=[{"name": "search", "version": "1", "schema_sha256": "0" * 64}] * 2), ValueError),
    (dict(mcp_allowlists={"fs": "read_file"}), TypeError),
    (dict(validators=[{"name": "forbids"}]), TypeError),
    (dict(models=[7]), ValueError),
], ids=["capabilities-text", "blank-capability", "unversioned-skill", "skill-without-versions",
        "skill-version-number", "skills-complete-text", "tool-without-schema", "duplicate-tool",
        "allowlist-text", "validator-data", "model-not-text"])
def test_an_environment_refuses_malformed_parts(changes, error):
    with pytest.raises(error):
        TemplateEnvironment(**changes)


def test_an_environment_is_read_only():
    captured = TemplateEnvironment(skills={"memo-style": "sha256:" + "0" * 64,
                                           "tone": ["2", "1", "2"]},
                                   mcp_allowlists={"fs": ["read_file"]})
    # A skill takes one version or several, each kept once and sorted.
    assert dict(captured.skills) == {"memo-style": ("sha256:" + "0" * 64,), "tone": ("1", "2")}
    with pytest.raises(TypeError):
        captured.skills["other"] = "1"
    with pytest.raises(AttributeError):
        captured.models = frozenset({"other"})


def test_applicability_takes_a_record_and_an_environment(tmp_path):
    record = proven(tmp_path)
    with pytest.raises(TypeError, match="TemplateEnvironment"):
        check_applicability(record, {"models": [MODEL]})
    with pytest.raises(TypeError, match="TemplateRecord"):
        check_applicability(record.to_dict(), TemplateEnvironment())
    with pytest.raises(TypeError, match="environment must be a TemplateEnvironment"):
        candidate_from_graph(memo(None), rows=[outcome()], name="memo", require_success=False,
                             environment={"models": [MODEL]})


@pytest.mark.parametrize("member", [
    repeated_structures, candidate_from_graph, check_applicability, TemplateRecord.as_architect,
    TemplateArchitect.__init__, TemplateArchitect.plan, TemplateArchitect.local_only,
    TemplateEnvironment, TemplateEnvironment.capture, ApplicabilityReport,
    TemplateCatalog.active_templates, TemplateCatalog.active_local_templates,
    PlannerMemory.iter_outcomes,
], ids=lambda member: member.__qualname__)
def test_public_type_hints_resolve(member):
    typing.get_type_hints(getattr(member, "__func__", member))


def test_applicability_names_are_exported():
    for name in ("ApplicabilityReport", "TemplateApplicabilityError", "TemplateEnvironment",
                 "check_applicability"):
        assert name in smythe.__all__ and getattr(smythe, name) is getattr(smythe.templates, name)
