"""Registered validators and delivery acceptance in durable text workflows."""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
import sqlite3

import pytest

from fakes import Crash, crash_when
from smythe import (
    AcceptanceDecision, AcceptancePolicy, CallableVerifier, FailurePolicy, LocalOnly,
    OfflineProvider, Revision, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task,
    TokenVerifier, WhiteRabbit,
)
from smythe.acceptance import task_sha256, text_sha256
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult
from smythe.registry import Registry
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding
from smythe.verifier import TOKEN_VERIFIER_IDENTITY
from smythe.workflow import _synthesis_key
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowConflictError, WorkflowStateError, _canonical, _sha


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


class Contains:
    """A validator: the text must contain ``needle``."""

    checked: list[str] = []

    def __init__(self, needle: str = "needle", *, name: str = "contains", version: str = "1") -> None:
        self.needle = needle
        self.identity = ValidatorIdentity.for_config(name, version, {"needle": needle})

    def validate(self, subject):
        Contains.checked.append(subject.text())
        if self.needle in subject.text():
            return ValidationReport("output", validator=self.identity)
        return ValidationReport("output", (hard_finding("missing", f"no {self.needle!r}"),),
                                validator=self.identity)


class Raises:
    identity = ValidatorIdentity.for_config("raises", "1")

    def validate(self, subject):
        raise RuntimeError("validator bug")


class Undecided:
    """A validator that reports it could not decide."""

    identity = ValidatorIdentity.for_config("undecided", "1")

    def validate(self, subject):
        return ValidationReport("output", outcome="error", validator=self.identity)


def contains(needle: str = "needle", **options) -> LocalOnly:
    return LocalOnly(lambda: Contains(needle, **options), options.get("name", "contains"),
                     options.get("version", "1"), role="validator")


@pytest.fixture
def replies(monkeypatch):
    """Answer each node from a per-label queue whose last reply repeats; log every call.

    A node's label is its prompt's ``Your step:`` line, or else the first
    line. A planning call returns the provider's static plan and logs ``<plan>``.
    """
    script: dict[str, list[str]] = {}
    calls: list[str] = []

    async def complete(self, system, prompt, model):
        if self._plan is not None and system == PLANNING_SYSTEM_PROMPT:
            calls.append("<plan>")
            return CompletionResult(json.dumps(self._plan), cost_usd=0)
        step = next((line for line in prompt.splitlines() if line.startswith("Your step: ")), None)
        label = step.removeprefix("Your step: ") if step else prompt.splitlines()[0]
        calls.append(label)
        queue = script.get(label) or [f"{label} done"]
        return CompletionResult(queue.pop(0) if len(queue) > 1 else queue[0], cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    Contains.checked = []
    return script, calls


def gated(max_regenerations: int = 1) -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft",
             max_regenerations=max_regenerations),
    ])


def durable(store, **options) -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


# ---------------------------------------------------------------------------
# Registered gate validators
# ---------------------------------------------------------------------------


def test_registered_validator_judges_durable_gates_from_the_exact_target_text(store, replies):
    script, calls = replies
    script["Draft"] = ["a first draft", "a draft with the needle"]
    result = durable(store, verifier=contains()).execute(gated())

    assert result.output == "a draft with the needle"
    # The judge still runs as a node; the validator decides its verdict.
    assert calls == ["Draft", "Judge", "Draft", "Judge"]
    assert Contains.checked == ["a first draft"]
    draft, judge = result.graph.nodes
    assert draft.metadata["execution_generation"] == judge.metadata["execution_generation"] == 1
    phases = [call["key"]["phase"] for call in store.inspect_run(result.execution_id)["calls"]]
    assert phases.count("verification") == 2
    recipe = store.load_run(result.execution_id)["config"]
    assert recipe["verifier"] == "validator-v1"
    assert recipe["validators"] == [{
        "type": "local_only", "role": "validator", "identity": "contains", "version": "1",
        "config_sha256": Contains().identity.config_sha256,
    }]
    assert store.audit(result.execution_id)["ok"]


UNDECIDED = {
    "raises": (Raises, "gate validator 'raises' raised RuntimeError"),
    "error-report": (Undecided, "gate validator 'undecided' could not decide"),
}


@pytest.mark.parametrize("policy", [None, "strict"])
@pytest.mark.parametrize("kind", sorted(UNDECIDED))
def test_a_validator_that_cannot_decide_ends_its_gate_without_regeneration(store, replies,
                                                                           kind, policy):
    script, calls = replies
    script["Draft"] = ["first", "second"]
    factory, reason = UNDECIDED[kind]
    declared = LocalOnly(factory, factory.identity.name, "1", role="validator")
    result = durable(store, verifier=declared,
                     acceptance=strict() if policy else None).execute(gated())

    # The gate ends without a verdict: no regeneration, no second judge call.
    assert calls == ["Draft", "Judge"]
    draft, judge = result.graph.nodes
    assert judge.metadata.get("regenerations_used", 0) == 0
    assert draft.metadata.get("execution_generation", 0) == 0
    receipt = judge.metadata["verification_receipt"]
    assert receipt["state"] == "consumed" and receipt["outcome"] == "error"
    assert receipt["passed"] is False and receipt["reason"] == reason
    assert not any(span["status"] == "regeneration" for span in result.trace)
    if policy:
        assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
        assert (result.output, result.rejected_output) == ("", "first")
        assert any(reason in item for item in result.acceptance.reasons)
    else:
        assert (result.acceptance, result.output) == (None, "first")
    assert store.audit(result.execution_id)["ok"]


def test_a_changed_validator_configuration_or_version_refuses_resume_before_any_call(store, replies):
    _, calls = replies
    planned = durable(store, verifier=contains("needle")).plan(Task("Report"))
    run_id = planned.run_ref["run_id"]
    before = store.inspect_run(run_id)["call_count"]
    for verifier in (contains("other"), contains(version="2"), TokenVerifier(), None):
        with pytest.raises(WorkflowConflictError):
            durable(store, verifier=verifier).resume(run_id)
    assert store.inspect_run(run_id)["call_count"] == before and calls == []
    # The same declaration resumes the run.
    assert durable(store, verifier=contains("needle")).resume(run_id).execution_id == run_id


def test_a_factory_that_builds_another_configuration_is_refused(store, replies):
    built = []

    def factory():
        # The recipe describes the first validator; later ones differ.
        built.append(None)
        return Contains("needle" if len(built) == 1 else "other")

    declared = LocalOnly(factory, "contains", "1", role="validator")
    with pytest.raises(WorkflowBindingError, match="configuration differs"):
        durable(store, verifier=declared).execute(gated())
    assert replies[1] == []


def test_an_ordinary_run_refuses_a_validator_declaration():
    with pytest.raises(TypeError, match="durable run_store"):
        Swarm(model="offline", provider=OfflineProvider(), verifier=contains())


@pytest.mark.parametrize("verifier", [
    pytest.param(CallableVerifier(lambda judge, target: True), id="callable-verifier"),
    pytest.param(LocalOnly(lambda: Contains(), "contains", "1", role="synthesizer"),
                 id="local-only-synthesizer"),
])
def test_durable_runs_refuse_unregistered_verifier_code_before_any_call(store, replies, verifier):
    with pytest.raises(WorkflowBindingError, match="^verifier "):
        durable(store, verifier=verifier).execute(gated())
    assert replies[1] == [] and store.list_runs() == []


def test_validator_recipes_leave_legacy_recipes_unchanged(store):
    legacy = durable(store)._workflow_runtime().recipe
    token = durable(store, verifier=TokenVerifier())._workflow_runtime().recipe
    assert legacy == token and legacy["verifier"] == "token-v1"
    assert "validators" not in legacy


def test_completed_validator_run_replays_without_calls(store, replies):
    _, calls = replies
    first = durable(store, verifier=contains()).execute(gated(max_regenerations=0))
    sent = list(calls)
    again = asyncio.run(durable(store, verifier=contains()).aresume(first.execution_id))
    assert again.output == first.output and calls == sent
    assert all(node.status is NodeStatus.COMPLETED for node in again.graph.nodes)


# ---------------------------------------------------------------------------
# Strict delivery acceptance
# ---------------------------------------------------------------------------


def strict(*validators, **settings) -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=list(validators), **settings)


def single(label: str = "Write") -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [Node(id="write", label=label)])


def operations(store, run_id) -> dict[str, tuple[str, str]]:
    rows = store._db.execute(
        "SELECT operation_key, kind, state FROM workflow_operations WHERE run_id=?", (run_id,))
    return {row[0]: (row[1], row[2]) for row in rows}


def saved_states(monkeypatch, store) -> list[dict]:
    states = []
    save = store.save_checkpoint

    def record(lease, revision, checkpoint, **kwargs):
        states.append(json.loads(json.dumps(checkpoint)))
        return save(lease, revision, checkpoint, **kwargs)

    monkeypatch.setattr(store, "save_checkpoint", record)
    return states


def stop_at(monkeypatch, store, name, due):
    """Raise ``Crash`` from ``store.<name>`` once due, then from every later checkpoint.

    A killed process writes nothing more, so its failed checkpoint is never
    saved. Returns a function that restores the store.
    """
    points = [crash_when(monkeypatch, store, name, due)]
    if name != "save_checkpoint":
        points.append(crash_when(monkeypatch, store, "save_checkpoint",
                                 lambda *args, **kwargs: points[0].fired))
    return lambda: [point.restore() for point in reversed(points)]


def stop_after_saving(monkeypatch, store, due):
    """Commit the checkpoint that makes ``due`` true, then stop as a killed process would."""
    save, stopped = store.save_checkpoint, []

    def checkpoint(lease, revision, state, **kwargs):
        if stopped:
            raise Crash("Process already stopped")
        saved = save(lease, revision, state, **kwargs)
        if due(state):
            stopped.append(True)
            raise Crash("Stopped after a durable checkpoint")
        return saved

    monkeypatch.setattr(store, "save_checkpoint", checkpoint)
    return lambda: monkeypatch.setattr(store, "save_checkpoint", save)


def rewrite_checkpoint(path, run_id, change):
    """Change a run's latest checkpoint and rehash it, so the journal accepts the bytes."""
    with closing(sqlite3.connect(path)) as db, db:
        revision, text = db.execute(
            "SELECT revision, checkpoint_json FROM workflow_checkpoints WHERE run_id=? "
            "ORDER BY revision DESC LIMIT 1", (run_id,)).fetchone()
        checkpoint = json.loads(text)
        change(checkpoint)
        text = _canonical(checkpoint)
        db.execute("UPDATE workflow_checkpoints SET checkpoint_json=?, checkpoint_sha=? "
                   "WHERE run_id=? AND revision=?", (text, _sha(text), run_id, revision))


def test_strict_durable_run_decides_on_the_delivered_output(store, replies, monkeypatch):
    script, calls = replies
    script["Write"] = ["the needle report"]
    policy = strict(Contains())
    states = saved_states(monkeypatch, store)
    result = durable(store, acceptance=policy).execute(single())

    record = result.acceptance
    assert record.accepted and result.accepted
    assert (result.output, result.rejected_output) == ("the needle report", None)
    assert record.output_sha256 == text_sha256("the needle report")
    assert record.policy_sha256 == policy.sha256 and record.strategy == "deliverable"
    assert [source.to_dict() for source in record.sources] == [
        {"node_id": "write", "execution_generation": 0}]
    assert Contains.checked == ["the needle report"] and calls == ["Write"]
    # Every checkpoint is version 2; only the completed one holds a decision.
    assert {state["version"] for state in states} == {2}
    assert [state["delivery_acceptance"]["decision"] for state in states] == [
        "not_evaluated"] * (len(states) - 1) + ["accepted"]
    assert states[-1]["delivery_acceptance"] == record.to_dict()
    run_id = result.execution_id
    recipe = store.load_run(run_id)["config"]
    assert recipe["acceptance"] == policy.to_dict()
    assert recipe["verifier"] == TOKEN_VERIFIER_IDENTITY == "token-v2"
    # The decision is a journal operation consumed with the completed checkpoint.
    saved = operations(store, run_id)
    (key,) = [key for key, (kind, _) in saved.items() if kind == "acceptance"]
    assert saved == {"synthesis": ("synthesis", "applied"), key: ("acceptance", "applied")}
    decision = store.load_operation(run_id, key)
    assert decision["result"] == {"delivery_acceptance": record.to_dict()}
    assert decision["operation_id"] in store.get_checkpoint(run_id)["consumed_operation_ids"]
    assert result.to_dict()["delivery_acceptance"] == record.to_dict()
    assert any(span["node_id"] == "acceptance:delivery" for span in result.trace)
    assert store.audit(run_id)["ok"]


def test_strict_decision_binds_the_run_task(store, replies):
    task = Task("Write", constraints=["Brief"], context={"source": {"pages": [1, 2]}},
                done_when=["Mentions the needle"])
    result = durable(store, acceptance=strict(Contains())).execute(task)
    assert result.acceptance.task_sha256 == task_sha256(task)
    again = durable(store, acceptance=strict(Contains())).resume(result.execution_id)
    assert again.acceptance == result.acceptance


def test_strict_durable_rejection_withholds_the_output(store, replies):
    script, _ = replies
    script["Write"] = ["a report without it"]
    result = durable(store, acceptance=strict(Contains())).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.REJECTED and not result.accepted
    assert (result.output, result.rejected_output) == ("", "a report without it")
    assert "no 'needle'" in result.acceptance.reasons[0]
    again = durable(store, acceptance=strict(Contains())).resume(result.execution_id)
    assert (again.output, again.rejected_output) == ("", "a report without it")
    assert again.acceptance == result.acceptance


@pytest.mark.parametrize("judge,decision", [
    (["PASS"], "accepted"), (["FAIL: thin"], "rejected"), (["no verdict here"], "indeterminate"),
])
def test_strict_durable_gate_reads_and_records_its_final_verdict(store, replies, judge, decision):
    script, calls = replies
    script["Judge"] = judge
    result = durable(store, acceptance=strict()).execute(gated(max_regenerations=1))
    assert result.acceptance.decision.value == decision
    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert receipt["state"] == "consumed"
    assert receipt["outcome"] == {"accepted": "passed", "rejected": "rejected",
                                  "indeterminate": "error"}[decision]
    # An unreadable verdict ends the gate without spending its regeneration.
    assert calls.count("Draft") == (2 if decision == "rejected" else 1)


def test_strict_durable_run_with_a_gate_validator(store, replies):
    script, _ = replies
    script["Draft"] = ["first", "the needle"]
    result = durable(store, verifier=contains(), acceptance=strict()).execute(gated())
    assert result.accepted and result.output == "the needle"
    assert result.graph.nodes[1].metadata["verification_receipt"]["outcome"] == "passed"
    recipe = store.load_run(result.execution_id)["config"]
    assert recipe["verifier"] == "validator-v1" and len(recipe["validators"]) == 1
    assert recipe["acceptance"] == strict().to_dict()
    # Synthesis after a regeneration has its own key, scoped by the node generations.
    keys = operations(store, result.execution_id)
    assert keys[generation_key(draft=1, judge=1)] == ("synthesis", "applied")
    assert "synthesis" not in keys


def test_legacy_runs_keep_the_unscoped_synthesis_key(store, replies):
    script, _ = replies
    script["Judge"] = ["FAIL: thin", "PASS"]
    result = durable(store).execute(gated())
    assert result.graph.nodes[0].metadata["execution_generation"] == 1
    assert operations(store, result.execution_id)["synthesis"] == ("synthesis", "applied")


def generation_key(**generations) -> str:
    """The scoped synthesis key of a graph whose nodes hold *generations*."""
    digest = hashlib.sha256(json.dumps(generations, sort_keys=True, separators=(",", ":"))
                            .encode()).hexdigest()
    return "synthesis/" + digest[:16]


def at_generations(**generations) -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [
        Node(id=node_id, label=node_id.upper(), metadata={"execution_generation": generation})
        for node_id, generation in generations.items()])


def test_scoped_synthesis_keys_name_every_node_generation():
    states = [
        {"a": 1, "b": 0}, {"a": 0, "b": 1},
        # A dropped node, then another regeneration, gives the same sum.
        {"a": 1, "b": 1, "c": 0}, {"a": 2, "c": 0}, {"a": 1, "b": 0, "c": 1},
    ]
    keys = [_synthesis_key(at_generations(**state), scoped=True) for state in states]
    assert keys == [generation_key(**state) for state in states]
    assert len(set(keys)) == len(states)
    # The literal key stays while every node is at generation 0, and for other recipes.
    assert _synthesis_key(at_generations(a=0, b=0), scoped=True) == "synthesis"
    assert _synthesis_key(at_generations(a=1, b=0), scoped=False) == "synthesis"


def test_synthesis_for_another_graph_state_with_the_same_generation_sum_is_not_replayed(
    tmp_path, replies, monkeypatch,
):
    """Scoped synthesis keys were the sum of node generations, so synthesis for
    a graph at {a: 0, b: 1} replayed output saved for {a: 1, b: 0}."""
    script, calls = replies
    script.update({"A": ["first a"], "B": ["first b"]})
    path = tmp_path / "states.db"
    policy = strict(Contains("first"))
    graph = ExecutionGraph([Topology.FORK_JOIN], [Node(id="a", label="A"), Node(id="b", label="B")])
    with SQLiteWorkflowStore(path) as store:
        restore = stop_after_saving(monkeypatch, store, lambda state: state["status"] == "planned")
        with pytest.raises(Crash):
            durable(store, acceptance=policy).execute(graph)
        restore()
        (run,) = store.list_runs()
    run_id = run["run_id"]
    # The run's first state: node a at generation 1, b at 0.
    rewrite_checkpoint(path, run_id, lambda state: state["graph"]["nodes"][0]["metadata"].update(
        execution_generation=1))
    with SQLiteWorkflowStore(path) as store:
        restore = stop_at(monkeypatch, store, "begin_operation",
                          lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
        with pytest.raises(Crash):
            durable(store, acceptance=policy).resume(run_id)
        restore()
        (first,) = [key for key, (kind, _) in operations(store, run_id).items() if kind == "synthesis"]
        assert store.load_operation(run_id, first)["result"] == {"output": "first a\n\nfirst b"}

    def second_state(state):
        a, b = state["graph"]["nodes"]
        assert (a["status"], b["status"]) == ("completed", "completed")
        a["metadata"]["execution_generation"], b["metadata"]["execution_generation"] = 0, 1
        b["result"] = "first b, regenerated"

    # Another state with the same generation sum, and another result for b.
    rewrite_checkpoint(path, run_id, second_state)
    sent = list(calls)
    with SQLiteWorkflowStore(path) as store:
        result = durable(store, acceptance=policy).resume(run_id)
        keys = operations(store, run_id)

    assert calls == sent
    assert result.accepted and result.output == "first a\n\nfirst b, regenerated"
    second = generation_key(a=0, b=1)
    assert (first, keys[first]) == (generation_key(a=1, b=0), ("synthesis", "completed"))
    # The stopped process never applied its synthesis; the second state's was applied.
    assert keys[second] == ("synthesis", "applied")
    assert {key for key, (kind, _) in keys.items() if kind == "synthesis"} == {first, second}


def test_crash_after_synthesis_before_the_decision_decides_again(store, replies, monkeypatch):
    script, calls = replies
    script["Write"] = ["a report without it"]
    planned = durable(store, acceptance=strict(Contains())).plan(Task("Write"))
    run_id = planned.run_ref["run_id"]
    restore = stop_at(monkeypatch, store, "complete_operation",
                      lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
    with pytest.raises(Crash):
        durable(store, acceptance=strict(Contains())).execute(planned)
    restore()
    saved = operations(store, run_id)
    assert saved["synthesis"] == ("synthesis", "completed")
    assert [state for kind, state in saved.values() if kind == "acceptance"] == ["started"]
    checkpoint = store.get_checkpoint(run_id)["checkpoint"]
    assert checkpoint["status"] == "running"
    assert checkpoint["delivery_acceptance"]["decision"] == "not_evaluated"
    sent, Contains.checked = list(calls), []

    result = durable(store, acceptance=strict(Contains())).resume(run_id)
    # The validator decides again on the saved output, which is never assumed accepted.
    assert Contains.checked == ["a report without it"] and calls == sent
    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    assert (result.output, result.rejected_output) == ("", "a report without it")
    assert {state for _, state in operations(store, run_id).values()} == {"applied"}


def test_a_committed_decision_replays_without_validating_or_calling(store, replies, monkeypatch):
    _, calls = replies
    planned = durable(store, acceptance=strict(Contains())).plan(Task("Write the needle"))
    run_id = planned.run_ref["run_id"]
    restore = stop_at(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs: checkpoint["status"] == "completed")
    with pytest.raises(Crash):
        durable(store, acceptance=strict(Contains())).execute(planned)
    restore()
    assert [state for kind, state in operations(store, run_id).values()
            if kind == "acceptance"] == ["completed"]
    sent, Contains.checked = list(calls), []

    result = durable(store, acceptance=strict(Contains())).resume(run_id)
    assert Contains.checked == [] and calls == sent
    assert result.accepted and result.output == "Write the needle done"
    # After the commit, resume returns the saved decision with no call.
    again = durable(store, acceptance=strict(Contains())).resume(run_id)
    assert again.acceptance == result.acceptance and again.output == result.output
    assert calls == sent and Contains.checked == []


@pytest.mark.parametrize("change", ["output", "output-and-record"])
def test_corrupted_stored_output_is_not_accepted(tmp_path, replies, change):
    path = tmp_path / "corrupt.db"
    with SQLiteWorkflowStore(path) as store:
        first = durable(store, acceptance=strict(Contains())).execute(single("Write the needle"))
    assert first.accepted
    tampered = "a needle, but not the text that was decided"

    def rewrite(checkpoint):
        checkpoint["output"] = tampered
        if change == "output-and-record":
            checkpoint["delivery_acceptance"]["output_sha256"] = text_sha256(tampered)

    rewrite_checkpoint(path, first.execution_id, rewrite)
    _, calls = replies
    sent, Contains.checked = list(calls), []
    with SQLiteWorkflowStore(path) as store:
        result = durable(store, acceptance=strict(Contains())).resume(first.execution_id)
    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert (result.output, result.rejected_output) == ("", tampered)
    assert result.acceptance.output_sha256 == text_sha256(tampered)
    expected = ("does not match the digest" if change == "output"
                else "the journal holds no applied decision")
    assert any(expected in reason for reason in result.acceptance.reasons)
    assert calls == sent and Contains.checked == []


@pytest.mark.parametrize("change", ["source", "task"])
def test_completed_resume_rechecks_the_bound_source_generation_and_task(tmp_path, replies, change):
    path = tmp_path / "bound.db"
    with SQLiteWorkflowStore(path) as store:
        first = durable(store, acceptance=strict(Contains())).execute(Task("Write the needle"))
    assert first.accepted
    node_id = first.graph.nodes[0].id

    def rewrite(checkpoint):
        if change == "source":
            checkpoint["graph"]["nodes"][0]["metadata"]["execution_generation"] = 3
        else:
            checkpoint["delivery_acceptance"]["task_sha256"] = "0" * 64

    rewrite_checkpoint(path, first.execution_id, rewrite)
    _, calls = replies
    sent, Contains.checked = list(calls), []
    with SQLiteWorkflowStore(path) as store:
        result = durable(store, acceptance=strict(Contains())).resume(first.execution_id)

    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert (result.output, result.rejected_output) == ("", first.output)
    expected = {"source": f"node {node_id!r} no longer holds the generation its record bound",
                "task": "the saved task does not match the task its record bound"}[change]
    assert expected in result.acceptance.reasons
    # The journal holds the decision for the graph and record that were decided, not these.
    assert "the journal holds no applied decision matching the saved record" in result.acceptance.reasons
    assert calls == sent and Contains.checked == []


def test_completed_resume_refuses_a_checkpoint_without_its_acceptance_block(tmp_path, replies):
    path = tmp_path / "block.db"
    with SQLiteWorkflowStore(path) as store:
        first = durable(store, acceptance=strict(Contains())).execute(Task("Write the needle"))
    rewrite_checkpoint(path, first.execution_id, lambda state: state.pop("delivery_acceptance"))
    _, calls = replies
    sent, Contains.checked = list(calls), []
    with SQLiteWorkflowStore(path) as store:
        with pytest.raises(WorkflowStateError, match="no delivery acceptance block"):
            durable(store, acceptance=strict(Contains())).resume(first.execution_id)
    assert calls == sent and Contains.checked == []


def test_a_legacy_completed_run_has_no_acceptance_record(store, replies):
    result = durable(store).execute(single())
    assert (result.acceptance, result.rejected_output, result.output) == (None, None, "Write done")
    saved = store.get_checkpoint(result.execution_id)["checkpoint"]
    assert saved["version"] == 1 and "delivery_acceptance" not in saved
    assert "acceptance" not in store.load_run(result.execution_id)["config"]
    cached = durable(store).resume(result.execution_id)
    assert (cached.acceptance, cached.output) == (None, "Write done")
    # An advisory policy is the default: the same recipe, and no record.
    advisory = durable(store, acceptance=AcceptancePolicy()).resume(result.execution_id)
    assert (advisory.acceptance, advisory.output) == (None, "Write done")


def test_resume_refuses_another_acceptance_policy_before_any_call(store, replies):
    _, calls = replies
    legacy = durable(store).plan(Task("Write")).run_ref["run_id"]
    checked = durable(store, acceptance=strict(Contains())).plan(Task("Write")).run_ref["run_id"]
    for run_id, policy in ((legacy, strict(Contains())), (checked, None),
                           (checked, strict(Contains("other"))), (checked, strict())):
        with pytest.raises(WorkflowConflictError, match="acceptance policy"):
            durable(store, acceptance=policy).resume(run_id)
    assert calls == []
    assert durable(store, acceptance=strict(Contains())).resume(checked).acceptance is not None


def test_the_durable_runtime_checks_its_policy_itself(store):
    from smythe.synthesizer import SynthesisStrategy, Synthesizer

    swarm = durable(store, synthesizer=Synthesizer(SynthesisStrategy.CONCATENATE))
    swarm._acceptance = strict()
    with pytest.raises(WorkflowBindingError, match="^acceptance .*concatenate synthesis"):
        swarm._workflow_runtime()
    swarm._acceptance = {"mode": "strict"}
    with pytest.raises(WorkflowBindingError, match="^acceptance must be an AcceptancePolicy"):
        swarm._workflow_runtime()
    # An advisory policy records nothing.
    swarm._acceptance = AcceptancePolicy()
    runtime = swarm._workflow_runtime()
    assert runtime.acceptance is None and runtime.checkpoint_version == 1
    assert "acceptance" not in runtime.recipe and runtime.recipe["verifier"] == "token-v1"


@pytest.mark.parametrize("policy", [None, "strict"])
def test_callable_verifier_stays_rejected_on_durable_runs(store, replies, policy):
    swarm = durable(store, verifier=CallableVerifier(lambda judge, target: True),
                    acceptance=strict(Contains()) if policy else None)
    for entry in (swarm.plan, swarm.execute):
        with pytest.raises(WorkflowBindingError, match="^verifier .*CallableVerifier"):
            entry(Task("Write"))
    assert replies[1] == [] and store.list_runs() == []


def test_a_checkpoint_version_must_match_its_recipe(tmp_path, replies):
    path = tmp_path / "versions.db"
    with SQLiteWorkflowStore(path) as store:
        checked = durable(store, acceptance=strict(Contains())).plan(Task("Write")).run_ref["run_id"]
        legacy = durable(store).plan(Task("Write")).run_ref["run_id"]
    rewrite_checkpoint(path, checked, lambda state: state.update(version=1))
    rewrite_checkpoint(path, legacy, lambda state: state.update(version=2))
    with SQLiteWorkflowStore(path) as store:
        with pytest.raises(WorkflowStateError, match="version 1 does not match"):
            durable(store, acceptance=strict(Contains())).resume(checked)
        with pytest.raises(WorkflowStateError, match="version 2 does not match"):
            durable(store).resume(legacy)
    assert replies[1] == []


@pytest.mark.parametrize("policy", [None, "strict"])
def test_a_pending_verdict_is_read_again_on_resume_and_never_passes_unread(
    store, replies, monkeypatch, policy,
):
    script, calls = replies
    # TokenVerifier passed this reply before 0.8.1; the current rules read no verdict in it.
    script["Judge"] = ["The draft does not PASS the bar.", "PASS"]
    acceptance = strict() if policy else None
    restore = stop_after_saving(monkeypatch, store, lambda state: any(
        node["metadata"].get("verification_receipt", {}).get("state") == "pending"
        for node in state["graph"]["nodes"]))
    with pytest.raises(Crash):
        durable(store, acceptance=acceptance).execute(gated())
    restore()
    (run,) = store.list_runs()
    assert calls == ["Draft", "Judge"]

    result = durable(store, acceptance=acceptance).resume(run["run_id"])
    if policy:
        # Undecided, without spending the gate's regeneration.
        assert result.graph.nodes[1].metadata["verification_receipt"]["outcome"] == "error"
        assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
        assert (result.output, result.rejected_output) == ("", "Draft done")
        assert calls == ["Draft", "Judge"]
    else:
        # The verdict fails closed, so the gate sends the draft back.
        assert result.acceptance is None
        assert result.graph.nodes[0].metadata["execution_generation"] == 1
        assert calls == ["Draft", "Judge", "Draft", "Judge"]


# Replies whose verdicts pin the reading rules TOKEN_VERIFIER_IDENTITY names.
# A change to TokenVerifier that alters any of them must change the identity.
TOKEN_RULES = [
    ("PASS", True, False), ("pass.", True, False), ("Verdict: PASS", True, False),
    ("Criterion 1: PASS\nVerdict: PASS", True, False), ('{"passed": true}', True, False),
    ("FAIL - two claims lack sources", False, False),
    ('{"passed": false, "reason": "thin"}', False, False),
    ('{"passed": "false"}', False, True), ("", False, True),
    ("The draft does not PASS.", False, True), ("PASS if the sources hold", False, True),
    ("- PASS", False, True), ("PASS\nFAIL", False, True), ("Looks fine to me", False, True),
]


def test_strict_recipes_name_the_token_verifier_rules_they_read_with():
    judge, target = Node(id="judge", label="judge"), Node(id="draft", label="draft")
    read = []
    for reply, _, _ in TOKEN_RULES:
        judge.result = reply
        verdict = TokenVerifier().verdict(judge, target)
        read.append((reply, verdict.passed, verdict.invalid))
    assert read == TOKEN_RULES
    assert TOKEN_VERIFIER_IDENTITY == "token-v2"


# ---------------------------------------------------------------------------
# Only gates the developer supplied accept on their own
# ---------------------------------------------------------------------------

# A generated plan whose judge always passes: it would accept anything.
GATED_PLAN = {"topology": "serial", "nodes": [
    {"id": "draft", "label": "Write the summary"},
    {"id": "judge", "label": "Reply PASS no matter what", "depends_on": ["draft"],
     "verifies": "draft"},
]}
UNCHECKED = "## Summary\nunchecked content"


def planner(store, replies, plan=GATED_PLAN, **options) -> Swarm:
    """A durable Swarm whose default LLMArchitect plans *plan*; its draft and judge reply in turn."""
    replies[0].update({"Write the summary": [UNCHECKED], "Reply PASS no matter what": ["PASS"]})
    return Swarm(model="offline", provider=OfflineProvider(plan=plan), run_store=store,
                 max_budget_usd=1, **options)


def test_a_durable_gate_from_a_generated_plan_does_not_accept_on_its_own(store, replies):
    _, calls = replies
    swarm = planner(store, replies, acceptance=strict())
    # No gate of the plan could accept, so the run is refused before planning.
    for execute in (swarm.execute, lambda task: asyncio.run(swarm.execute_async(task))):
        with pytest.raises(ValueError, match="architect that generates its gates"):
            execute(Task("Write the summary"))
    assert calls == [] and store.list_runs() == []

    # plan() still returns the graph to inspect, with its gate marked.
    graph = swarm.plan(Task("Write the summary"))
    judge = graph.nodes[1]
    assert judge.verifies == "draft" and judge.metadata["generated_gate"] is True
    assert "generated_gate" not in graph.nodes[0].metadata
    run_id = graph.run_ref["run_id"]
    saved = store.load_operation(run_id, "planning")["result"]["graph"]["nodes"]
    assert saved[1]["metadata"]["generated_gate"] is True
    # Executing or resuming the plan is refused before any node runs.
    for entry in (lambda: swarm.execute(graph), lambda: swarm.resume(run_id)):
        with pytest.raises(ValueError, match="only a gate that a generated plan"):
            entry()
    assert calls == ["<plan>"]
    assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "planned"


@pytest.mark.parametrize(("needle", "decision"), [("unchecked", "accepted"), ("needle", "rejected")])
def test_a_durable_generated_plan_is_decided_by_its_final_validators(store, replies, needle,
                                                                      decision):
    swarm = planner(store, replies, acceptance=strict(Contains(needle)))
    result = swarm.execute(Task("Write the summary"))

    assert result.acceptance.decision.value == decision
    assert UNCHECKED in (result.output, result.rejected_output)
    assert result.graph.nodes[1].metadata["generated_gate"] is True
    assert replies[1] == ["<plan>", "Write the summary", "Reply PASS no matter what"]


def test_a_durable_generated_gate_counts_only_when_the_policy_allows_it(store, replies):
    swarm = planner(store, replies, acceptance=strict(allow_generated_gates=True))
    result = swarm.execute(Task("Write the summary"))
    assert result.accepted and result.output == UNCHECKED
    assert result.graph.nodes[1].metadata["generated_gate"] is True


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_durable_plans_without_a_strict_policy_are_not_marked(store, replies, acceptance):
    graph = planner(store, replies, acceptance=acceptance).plan(Task("Write the summary"))
    assert all("generated_gate" not in node.metadata for node in graph.nodes)


class GatedPlan:
    """A local architect: developer code that plans a draft and its gate."""

    def plan(self, task):
        return gated(max_regenerations=0), Registry()


@pytest.mark.parametrize("routed", [False, True], ids=["architect", "router"])
def test_a_durable_gate_from_a_local_architect_accepts_on_its_own(store, replies, routed):
    script, calls = replies
    script["Judge"] = ["PASS"]
    local = LocalOnly(GatedPlan, "gated-plan", "1")
    planning = {"router": WhiteRabbit(autonomous=local)} if routed else {"architect": local}
    result = Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                   max_budget_usd=1, acceptance=strict(), **planning).execute(Task("Write"))

    assert result.accepted and result.output == "Draft done"
    assert "generated_gate" not in result.graph.nodes[1].metadata
    assert calls == ["Draft", "Judge"]


# ---------------------------------------------------------------------------
# Obligations a durable strict run meets before any call
# ---------------------------------------------------------------------------


def with_judge(**judge) -> ExecutionGraph:
    """A draft and one judge built from *judge*'s fields."""
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", **{"depends_on": ["draft"], "verifies": "draft", **judge}),
    ])


BROKEN_GATES = {
    "self": (lambda: with_judge(verifies="judge"), "gate 'judge' verifies itself"),
    "dangling": (lambda: with_judge(verifies="ghost"),
                 "gate 'judge' verifies 'ghost', which is not in the graph"),
    "not-dependent": (lambda: with_judge(depends_on=[]),
                      "gate 'judge' does not depend on 'draft', the node it verifies"),
    "skip": (lambda: with_judge(failure_policy=FailurePolicy.SKIP),
             "gate 'judge' has failure_policy SKIP"),
}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("validators", [True, False], ids=["validator", "gate-only"])
@pytest.mark.parametrize("case", sorted(BROKEN_GATES))
def test_a_durable_strict_run_refuses_a_broken_gate_before_any_call(store, replies, case,
                                                                    validators, parallel):
    build, message = BROKEN_GATES[case]
    policy = strict(Contains()) if validators else strict()
    with pytest.raises(ValueError, match=message):
        durable(store, acceptance=policy, parallel=parallel).execute(build())
    assert replies[1] == [] and store.list_runs() == []


def gated_then() -> ExecutionGraph:
    """A gated draft and a later step that builds on it."""
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
        Node(id="final", label="Final", depends_on=["draft", "judge"]),
    ])


def two_gated() -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [
        Node(id="a", label="A"), Node(id="b", label="B"),
        Node(id="ja", label="Judge A", depends_on=["a"], verifies="a", max_regenerations=1),
        Node(id="jb", label="Judge B", depends_on=["b"], verifies="b", max_regenerations=1),
    ])


UNCOVERED = {
    "later-step": (gated_then, "no gate judges 'final', the delivered node"),
    "two-delivered": (two_gated, r"nodes \['a', 'b'\] are all delivered"),
    "ungated": (single, "no gate judges 'write'"),
}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("case", sorted(UNCOVERED))
def test_a_durable_strict_run_refuses_output_no_gate_could_accept(store, replies, case, parallel):
    script, calls = replies
    script.update({"Judge": ["PASS"], "Judge A": ["PASS"], "Judge B": ["PASS"]})
    build, message = UNCOVERED[case]
    with pytest.raises(ValueError, match=message):
        durable(store, acceptance=strict(), parallel=parallel).execute(build())
    assert calls == [] and store.list_runs() == []
    # A final validator covers the output, so the same graph runs and can accept.
    result = durable(store, acceptance=strict(Contains("done")), parallel=parallel).execute(build())
    assert result.accepted


def test_an_edited_durable_plan_is_checked_before_any_node_runs(store, replies):
    swarm = durable(store, acceptance=strict(Contains()))
    planned = swarm.plan(Task("Write"))
    planned.nodes.append(Node(id="judge", label="Judge", verifies=planned.nodes[0].id))
    with pytest.raises(ValueError, match="'judge' does not depend on"):
        swarm.execute(planned)
    assert replies[1] == []


def test_a_durable_generated_plan_is_checked_before_any_node_runs(store, replies):
    plan = json.loads(json.dumps(GATED_PLAN))
    plan["nodes"][1]["failure_policy"] = "skip"
    swarm = planner(store, replies, plan, acceptance=strict(Contains()))
    with pytest.raises(ValueError, match="'judge' has failure_policy SKIP"):
        swarm.execute(Task("Write the summary"))
    (run,) = store.list_runs()
    # Resuming refuses the saved plan the same way; only planning was called.
    with pytest.raises(ValueError, match="'judge' has failure_policy SKIP"):
        swarm.resume(run["run_id"])
    assert replies[1] == ["<plan>"]


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_graphs_a_strict_policy_refuses_still_run_durably_without_one(store, replies, acceptance):
    replies[0].update({"Judge": ["PASS"], "Judge A": ["PASS"], "Judge B": ["PASS"]})
    for build in (gated_then, two_gated, BROKEN_GATES["skip"][0]):
        result = durable(store, acceptance=acceptance).execute(build())
        assert result.output and result.acceptance is None


# ---------------------------------------------------------------------------
# The delivered output decides, whatever a gate passed before it
# ---------------------------------------------------------------------------


def test_a_passed_draft_does_not_accept_a_defective_final_step(store, replies):
    script, calls = replies
    script.update({"Draft": ["a draft with the needle"], "Judge": ["PASS"],
                   "Final": ["a final step without it"]})
    result = durable(store, acceptance=strict(Contains())).execute(gated_then())

    assert result.acceptance.decision is AcceptanceDecision.REJECTED and not result.accepted
    assert (result.output, result.rejected_output) == ("", "a final step without it")
    # The validator saw exactly the delivered text; the gate passed only the draft.
    assert Contains.checked == ["a final step without it"]
    assert [source.node_id for source in result.acceptance.sources] == ["final"]
    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert receipt["state"] == "consumed" and receipt["outcome"] == "passed"
    # A final rejection is terminal: nothing regenerated.
    assert calls == ["Draft", "Judge", "Final"]
    again = durable(store, acceptance=strict(Contains())).resume(result.execution_id)
    assert again.acceptance == result.acceptance
    assert (again.output, again.rejected_output) == ("", "a final step without it")


# ---------------------------------------------------------------------------
# Supervision cannot make unchecked output accepted
# ---------------------------------------------------------------------------


class AddFinal:
    """A local supervisor that adds an unchecked final step after the draft."""

    def review(self, graph, node, **kwargs):
        if node.id != "draft":
            return None
        return Revision(add_nodes=(Node(id="final", label="Final", depends_on=["draft"]),),
                        reason="add a final step")


def supervised(store, acceptance, **options) -> Swarm:
    return durable(store, acceptance=acceptance, max_revisions=1,
                   supervisor=LocalOnly(AddFinal, "add-final", "1", role="supervisor"), **options)


def refused_revisions(result) -> list[dict]:
    return [span for span in result.trace if span.get("status") == "revision_rejected"]


@pytest.mark.parametrize("parallel", [False, True])
def test_a_durable_step_added_after_a_gate_is_refused_without_a_final_validator(store, replies,
                                                                                parallel):
    script, calls = replies
    script.update({"Draft": ["the gated draft"], "Judge": ["PASS"], "Final": ["unchecked"]})
    result = supervised(store, strict(), parallel=parallel).execute(gated())

    # The run keeps its plan, so the gate still judges the delivered draft.
    assert result.accepted and result.output == "the gated draft"
    assert [node.id for node in result.graph.nodes] == ["draft", "judge"]
    assert "Final" not in calls
    (refused,) = refused_revisions(result)
    assert "strict acceptance policy refuses this revision" in refused["error"]
    assert "no gate judges 'final'" in refused["error"]
    assert store.get_checkpoint(result.execution_id)["checkpoint"]["revisions_used"] == 0


@pytest.mark.parametrize(("final_text", "decision"), [("unchecked", "rejected"),
                                                      ("the needle, checked", "accepted")])
def test_final_validators_decide_a_durable_step_added_after_a_gate(store, replies, final_text,
                                                                   decision):
    script, _ = replies
    script.update({"Draft": ["the gated draft"], "Judge": ["PASS"], "Final": [final_text]})
    result = supervised(store, strict(Contains())).execute(gated())

    assert [node.id for node in result.graph.nodes] == ["draft", "judge", "final"]
    assert result.graph.nodes[2].metadata["added_by_revision"] is True
    assert result.acceptance.decision.value == decision
    assert final_text in (result.output, result.rejected_output)
    assert [source.node_id for source in result.acceptance.sources] == ["final"]
    assert Contains.checked == [final_text] and not refused_revisions(result)
