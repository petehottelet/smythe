"""Registered validators and delivery acceptance in durable text workflows."""

from __future__ import annotations

import asyncio

import pytest

from smythe import (
    CallableVerifier, LocalOnly, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm, Task,
    TokenVerifier,
)
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowConflictError


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


def contains(needle: str = "needle", **options) -> LocalOnly:
    return LocalOnly(lambda: Contains(needle, **options), options.get("name", "contains"),
                     options.get("version", "1"), role="validator")


@pytest.fixture
def replies(monkeypatch):
    """Answer each node from a per-label queue whose last reply repeats; log every call."""
    script: dict[str, list[str]] = {}
    calls: list[str] = []

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
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


def test_a_validator_that_cannot_decide_fails_its_gate(store, replies):
    script, _ = replies
    script["Draft"] = ["first", "second"]
    declared = LocalOnly(Raises, "raises", "1", role="validator")
    result = durable(store, verifier=declared).execute(gated())

    # The verdict fails closed, so the gate regenerates within its allowance.
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert result.output == "second"
    span = next(item for item in result.trace if item["status"] == "regeneration")
    assert span["label"] == "gate validator 'raises' raised RuntimeError"


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
