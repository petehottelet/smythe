"""Strict delivery acceptance on ordinary runs: the exact delivered output decides."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from smythe import (
    AcceptancePolicy,
    CallableVerifier,
    DeliveryAcceptance,
    DeterministicArchitect,
    OfflineProvider,
    PlannerMemory,
    Swarm,
    Task,
    ValidationReport,
    ValidationSubject,
    ValidatorIdentity,
    Verdict,
)
from smythe.acceptance import (
    AcceptanceArtifact,
    evaluate_delivery,
    task_sha256,
    text_sha256,
    validate_acceptance_obligations,
)
from smythe.async_executor import AsyncExecutor
from smythe.checkpoint import CheckpointStore
from smythe.executor import Executor
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.provider import Artifact, CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import LLMSupervisor
from smythe.synthesizer import (
    MERGE_SYSTEM_PROMPT,
    SynthesisStrategy,
    Synthesizer,
    deliverable_sources,
)
from smythe.tracer import Tracer
from smythe.validation import hard_finding
from smythe.verifier import TokenVerifier, Verifier
from test_supervisor import GatedRunProvider, ScriptedSupervisor

STRATEGIES = list(SynthesisStrategy)
PATHS = ["sync", "async", "resume"]


class Script(Provider):
    """Answers each node from a per-label queue; an exception in the queue is raised.

    The last queued answer repeats. ``LLM_MERGE`` synthesis gets ``merge``.
    """

    def __init__(self, script: dict[str, list], *, merge: str = "## Summary\nMerged.",
                 artifacts: dict[str, list[Artifact]] | None = None) -> None:
        self._script = {label: list(answers) for label, answers in script.items()}
        self._merge = merge
        self._artifacts = artifacts or {}
        self.runs: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        if system == MERGE_SYSTEM_PROMPT:
            label = "__merge__"
            answer = self._merge
        else:
            step = [line for line in prompt.splitlines() if line.startswith("Your step: ")]
            label = step[0][len("Your step: "):] if step else prompt.splitlines()[0].strip()
            queue = self._script.get(label) or ["## Summary\nok"]
            answer = queue.pop(0) if len(queue) > 1 else queue[0]
        self.runs[label] = self.runs.get(label, 0) + 1
        if isinstance(answer, BaseException):
            raise answer
        return CompletionResult(text=answer, prompt_tokens=1, completion_tokens=1,
                                artifacts=list(self._artifacts.get(label, ())))


class MemoryStore(CheckpointStore):
    """Keeps every saved state as JSON data, like a file store would."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}
        self.saves: list[dict] = []

    def save(self, execution_id, state):
        snapshot = json.loads(json.dumps(state))
        self.states[execution_id] = snapshot
        self.saves.append(snapshot)

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)

    def only_id(self) -> str:
        (execution_id,) = self.states
        return execution_id


class Forbids:
    """A final validator: the delivered text must not contain ``marker``."""

    def __init__(self, marker: str = "DEFECT", *, name: str = "forbids") -> None:
        self.marker = marker
        self.identity = ValidatorIdentity.for_config(name, "1", {"marker": marker})
        self.subjects: list[ValidationSubject] = []

    def validate(self, subject):
        self.subjects.append(subject)
        if self.marker in subject.text():
            return ValidationReport("output", (
                hard_finding("forbidden_text", f"output contains {self.marker!r}"),
            ), validator=self.identity)
        return ValidationReport("output", validator=self.identity)


class Returns:
    """A final validator that returns or raises whatever it was given."""

    def __init__(self, result, *, name: str = "returns") -> None:
        self.result = result
        self.identity = ValidatorIdentity.for_config(name, "1")

    def validate(self, subject):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _strict(*validators, **settings) -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=list(validators), **settings)


def _swarm(provider, *, acceptance=None, store=None, strategy=SynthesisStrategy.DELIVERABLE,
           **options) -> Swarm:
    return Swarm(
        provider=provider, model="test-model", artifact_dir=None,
        synthesizer=Synthesizer(strategy), checkpoint_store=store if store is not None else MemoryStore(),
        acceptance=acceptance, **options,
    )


def _gated(*, max_regenerations: int = 1, final: bool = False,
           judge_policy: FailurePolicy = FailurePolicy.HALT) -> ExecutionGraph:
    """A draft judged by a gate; with ``final``, a step that builds on the judged draft."""
    nodes = [
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=max_regenerations, failure_policy=judge_policy),
    ]
    if final:
        nodes.append(Node(id="final", label="final", depends_on=["draft", "judge"]))
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)


def _run(swarm: Swarm, graph: ExecutionGraph, path: str, store: MemoryStore):
    """Run *graph* on the sync, async or resume path; resume follows a crash."""
    if path == "sync":
        return swarm.execute(graph)
    if path == "async":
        return asyncio.run(swarm.execute_async(graph))
    with pytest.raises(RuntimeError, match="crash before final"):
        swarm.execute(graph)
    assert store.states[store.only_id()]["status"] == "failed"
    return swarm.resume(store.only_id())


def _final_script(final_text: str, path: str) -> dict:
    final = [RuntimeError("crash before final"), final_text] if path == "resume" else [final_text]
    return {"draft": ["## Summary\nA careful draft."], "judge": ["PASS"], "final": final}


def _not_accepted(result, decision: str) -> None:
    assert result.acceptance is not None
    assert result.acceptance.decision.value == decision
    assert result.accepted is False
    assert result.output == ""
    assert result.rejected_output is not None
    assert result.acceptance.output_sha256 == text_sha256(result.rejected_output)


def _refuses(policy: AcceptancePolicy, graph: ExecutionGraph, *, match: str,
             parallel: bool = False) -> None:
    """A strict Swarm refuses *graph* before any provider call or checkpoint."""
    provider, store = Script({}), MemoryStore()
    with pytest.raises(ValueError, match=match):
        _swarm(provider, acceptance=policy, store=store, parallel=parallel).execute(graph)
    assert provider.runs == {} and store.saves == []


def _decided(graph: ExecutionGraph, policy: AcceptancePolicy,
             provider) -> tuple[DeliveryAcceptance, str]:
    """Run *graph* on an executor, outside any Swarm, and decide its DELIVERABLE output."""
    for node in graph.nodes:
        node.metadata.setdefault("model", "test-model")
    Executor(provider, Registry(), Tracer(), artifact_dir=None, acceptance=policy).run(graph)
    sources, used_fallback = deliverable_sources(graph)
    output = Synthesizer().synthesize(graph)
    return evaluate_delivery(policy, graph, output, strategy="deliverable", sources=sources,
                             used_fallback=used_fallback), output


# ---------------------------------------------------------------------------
# An intermediate draft passes its gate; the final output decides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_passed_draft_does_not_accept_a_defective_final_output(strategy, path):
    store = MemoryStore()
    validator = Forbids()
    provider = Script(_final_script("## Summary\nDEFECT in the final step.", path),
                      merge="## Summary\nMerged, with a DEFECT.")
    swarm = _swarm(provider, acceptance=_strict(validator), store=store, strategy=strategy)

    result = _run(swarm, _gated(final=True), path, store)

    _not_accepted(result, "rejected")
    assert "DEFECT" in result.rejected_output
    # The validator saw exactly the delivered bytes, once, after synthesis.
    assert validator.subjects[-1].data == result.rejected_output.encode("utf-8")
    assert result.acceptance.strategy == strategy.value
    assert result.acceptance.validators[0].outcome.value == "rejected"
    assert any("forbids" in reason for reason in result.acceptance.reasons)
    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert receipt["state"] == "consumed" and receipt["outcome"] == "passed"
    # A final rejection is terminal: nothing regenerated.
    assert provider.runs["draft"] == 1 and provider.runs["judge"] == 1
    saved = store.states[result.execution_id]
    assert saved["version"] == 5 and saved["delivery_acceptance"] == result.acceptance.to_dict()
    assert saved["output"] == result.rejected_output


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_clean_final_output_is_accepted(strategy, path):
    store = MemoryStore()
    provider = Script(_final_script("## Summary\nThe final step.", path))
    swarm = _swarm(provider, acceptance=_strict(Forbids()), store=store, strategy=strategy)

    result = _run(swarm, _gated(final=True), path, store)

    assert result.accepted is True
    assert result.rejected_output is None
    assert result.acceptance.output_sha256 == text_sha256(result.output)
    assert result.acceptance.reasons
    assert store.states[result.execution_id]["delivery_acceptance"]["decision"] == "accepted"


def test_record_binds_the_deliverable_sources_and_generations():
    provider = Script({"draft": ["## Summary\nv1", "## Summary\nv2"], "judge": ["FAIL: thin", "PASS"]})
    result = _swarm(provider, acceptance=_strict(Forbids())).execute(_gated(final=True))

    assert result.accepted
    record = result.acceptance
    assert [(s.node_id, s.execution_generation) for s in record.sources] == [("final", 1)]
    assert record.used_fallback is False
    assert record.task_sha256 is None
    assert record.policy_sha256 == _strict(Forbids()).sha256


def test_record_binds_every_merged_source():
    provider = Script(_final_script("## Summary\nfinal", "sync"))
    result = _swarm(provider, acceptance=_strict(Forbids()),
                    strategy=SynthesisStrategy.CONCATENATE).execute(_gated(final=True))
    assert [s.node_id for s in result.acceptance.sources] == ["draft", "judge", "final"]


def test_record_binds_the_task_snapshot():
    task = Task("Write the summary", done_when=["no defects"])
    graph = _gated()
    graph.task = task
    result = _swarm(Script({"judge": ["PASS"]}), acceptance=_strict(Forbids())).execute(graph)
    assert result.acceptance.task_sha256 == task_sha256(task)


def test_final_validators_make_no_provider_call_and_add_no_node():
    provider = Script({"judge": ["PASS"]})
    result = _swarm(provider, acceptance=_strict(Forbids(), Forbids("OTHER", name="other"))
                    ).execute(_gated())
    assert result.accepted
    assert provider.runs == {"draft": 1, "judge": 1}
    assert [node.id for node in result.graph.nodes] == ["draft", "judge"]
    assert [r.validator.name for r in result.acceptance.validators] == ["forbids", "other"]


# ---------------------------------------------------------------------------
# Gate-derived acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
def test_a_passed_gate_accepts_its_exact_target(path):
    store = MemoryStore()
    draft = ["## Summary\nDraft."]
    if path == "resume":
        draft.insert(0, RuntimeError("crash before final"))
    provider = Script({"draft": draft, "judge": ["PASS"]})
    swarm = _swarm(provider, acceptance=_strict(), store=store)

    result = _run(swarm, _gated(), path, store)

    assert result.accepted is True
    assert result.output == "## Summary\nDraft."
    assert result.acceptance.validators == ()
    assert any("judge" in reason for reason in result.acceptance.reasons)


def test_a_passed_gate_does_not_accept_a_later_step():
    # A Swarm refuses the run before any call; decided anyway, it is not accepted.
    _refuses(_strict(), _gated(final=True), match="no gate judges 'final'")
    provider = Script(_final_script("## Summary\nA later step.", "sync"))
    record, output = _decided(_gated(final=True), _strict(), provider)
    assert record.decision.value == "indeterminate"
    assert output == "## Summary\nA later step."


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s is not SynthesisStrategy.DELIVERABLE])
def test_merged_output_needs_a_final_validator(strategy):
    with pytest.raises(ValueError, match="final validator"):
        _swarm(Script({}), acceptance=_strict(), strategy=strategy)


def test_a_custom_synthesizer_needs_a_final_validator():
    class Upper:
        def synthesize(self, graph, **_):
            return "## SUMMARY"

    with pytest.raises(ValueError, match="final validator"):
        Swarm(provider=Script({}), model="test-model", synthesizer=Upper(), acceptance=_strict())
    swarm = Swarm(provider=Script({"judge": ["PASS"]}), model="test-model", artifact_dir=None,
                  synthesizer=Upper(), acceptance=_strict(Forbids()))
    result = swarm.execute(_gated())
    assert result.accepted and result.acceptance.strategy == "custom"


def test_deliverable_fallback_needs_a_final_validator():
    def graph() -> ExecutionGraph:
        return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
            Node(id="draft", label="draft"),
            Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
                 max_regenerations=1),
            Node(id="final", label="final", depends_on=["draft"],
                 failure_policy=FailurePolicy.SKIP),
        ])

    _refuses(_strict(), graph(), match="no gate judges 'final'")
    provider = Script({"draft": ["## Summary\nDraft."], "judge": ["PASS"],
                       "final": [RuntimeError("unavailable")]})
    executed = graph()
    record, output = _decided(executed, _strict(), provider)

    assert record.decision.value == "indeterminate" and record.used_fallback is True
    draft, judge = executed.nodes[:2]
    assert deliverable_sources(executed) == ([draft, judge], True)
    assert output == "## Summary\nDraft.\n\nPASS"


def _two_gates() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[
        Node(id="a", label="a"), Node(id="b", label="b"),
        Node(id="ja", label="ja", depends_on=["a"], verifies="a", max_regenerations=1),
        Node(id="jb", label="jb", depends_on=["b"], verifies="b", max_regenerations=1),
    ])


def test_two_gated_deliverables_need_a_final_validator():
    _refuses(_strict(), _two_gates(), match=r"nodes \['a', 'b'\] are all delivered")
    record, _ = _decided(_two_gates(), _strict(), Script({"ja": ["PASS"], "jb": ["PASS"]}))
    assert record.decision.value == "indeterminate"
    result = _swarm(Script({"ja": ["PASS"], "jb": ["PASS"]}),
                    acceptance=_strict(Forbids())).execute(_two_gates())
    assert result.accepted


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


def _planner(**options) -> tuple[OfflineProvider, Swarm]:
    """A Swarm whose LLMArchitect plans GATED_PLAN; the draft and judge reply in turn."""
    provider = OfflineProvider(plan=GATED_PLAN, responses=[UNCHECKED, "PASS"])
    return provider, Swarm(provider=provider, model="test-model", artifact_dir=None,
                           checkpoint_store=MemoryStore(), **options)


def test_a_gate_from_a_generated_plan_does_not_accept_on_its_own():
    provider, swarm = _planner(acceptance=_strict())
    graph = swarm.plan(Task("Write the summary"))
    judge = graph.nodes[1]
    assert judge.verifies == "draft" and judge.metadata["generated_gate"] is True
    assert "generated_gate" not in graph.nodes[0].metadata

    record, _ = _decided(graph, _strict(), provider)

    # The plan's judge passed its draft, but a model wrote that gate.
    assert judge.metadata["advisory_verdict"]["outcome"] == "passed"
    assert record.decision.value == "indeterminate"
    assert any("generated plan or a supervisor revision" in reason for reason in record.reasons)


def test_a_generated_gate_counts_only_when_the_policy_allows_it():
    policy = _strict(allow_generated_gates=True)
    provider, swarm = _planner(acceptance=policy)
    result = swarm.execute(Task("Write the summary"))

    assert result.accepted and result.output == UNCHECKED
    assert result.graph.nodes[1].metadata["generated_gate"] is True


def test_a_generated_plan_is_accepted_through_its_final_validators():
    provider, swarm = _planner(acceptance=_strict(Forbids("unchecked")))
    result = swarm.execute(Task("Write the summary"))
    _not_accepted(result, "rejected")
    assert result.rejected_output == UNCHECKED

    provider, swarm = _planner(acceptance=_strict(Forbids()))
    assert swarm.execute(Task("Write the summary")).accepted


def test_a_gate_from_a_developer_architect_accepts_on_its_own():
    class Gated(DeterministicArchitect):
        def plan(self, task):
            return _gated(max_regenerations=1), Registry()

    swarm = _swarm(Script({"judge": ["PASS"]}), acceptance=_strict(), architect=Gated())
    result = swarm.execute(Task("Write the summary"))

    assert result.accepted and result.output == "## Summary\nok"
    assert "generated_gate" not in result.graph.nodes[1].metadata


def test_a_gate_a_revision_added_does_not_accept_on_its_own():
    graph = _gated(max_regenerations=0)
    graph.nodes[1].metadata["added_by_revision"] = True
    record, _ = _decided(graph, _strict(), Script({"judge": ["PASS"]}))
    assert record.decision.value == "indeterminate"

    graph = _gated(max_regenerations=0)
    graph.nodes[1].metadata["added_by_revision"] = True
    policy = _strict(allow_generated_gates=True)
    assert _decided(graph, policy, Script({"judge": ["PASS"]}))[0].accepted


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_plans_without_a_strict_policy_are_not_marked(acceptance):
    provider, swarm = _planner(acceptance=acceptance)
    graph = swarm.plan(Task("Write the summary"))
    assert all("generated_gate" not in node.metadata for node in graph.nodes)


# ---------------------------------------------------------------------------
# Obligations a strict run must meet before any call
# ---------------------------------------------------------------------------


def _with_judge(**judge) -> ExecutionGraph:
    """A draft and one judge built from *judge*'s fields."""
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", **{"depends_on": ["draft"], "verifies": "draft", **judge}),
    ])


BROKEN_GATES = {
    "self": (_with_judge(verifies="judge"), "gate 'judge' verifies itself"),
    "dangling": (_with_judge(verifies="ghost"),
                 "gate 'judge' verifies 'ghost', which is not in the graph"),
    "not-dependent": (_with_judge(depends_on=[]),
                      "gate 'judge' does not depend on 'draft', the node it verifies"),
    "skip": (_with_judge(failure_policy=FailurePolicy.SKIP),
             "gate 'judge' has failure_policy SKIP"),
}


@pytest.mark.parametrize("case", sorted(BROKEN_GATES))
def test_obligations_refuse_a_gate_that_cannot_judge_its_target(case):
    graph, message = BROKEN_GATES[case]
    # Final validators cover the output, so only the gate is at fault.
    with pytest.raises(ValueError, match=message):
        validate_acceptance_obligations(graph, _strict(Forbids()))


def test_obligations_accept_a_gate_that_reaches_its_target_through_another_node():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="review", label="review", depends_on=["draft"], verifies="draft"),
        Node(id="judge", label="judge", depends_on=["review"], verifies="draft",
             max_regenerations=1),
    ])
    validate_acceptance_obligations(graph, _strict())
    validate_acceptance_obligations(graph, _strict(Forbids()))


@pytest.mark.parametrize(("graph", "message"), [
    pytest.param(_gated(final=True), "no gate judges 'final', the delivered node", id="later-step"),
    pytest.param(_two_gates(), r"nodes \['a', 'b'\] are all delivered", id="two-delivered"),
    pytest.param(ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="draft", label="d")]),
                 "no gate judges 'draft'", id="ungated"),
])
def test_obligations_need_a_check_that_covers_the_delivered_output(graph, message):
    with pytest.raises(ValueError, match=message):
        validate_acceptance_obligations(graph, _strict())
    validate_acceptance_obligations(graph, _strict(Forbids()))


def test_obligations_count_only_gates_the_developer_supplied():
    graph = _gated()
    graph.nodes[1].metadata["generated_gate"] = True
    with pytest.raises(ValueError, match="only a gate that a generated plan or a supervisor"):
        validate_acceptance_obligations(graph, _strict())
    validate_acceptance_obligations(graph, _strict(allow_generated_gates=True))
    validate_acceptance_obligations(graph, _strict(Forbids()))


def test_obligations_need_a_final_validator_for_other_synthesis():
    with pytest.raises(ValueError, match="llm_merge synthesis needs a final validator"):
        validate_acceptance_obligations(_gated(), _strict(), strategy="llm_merge")
    validate_acceptance_obligations(_gated(), _strict(Forbids()), strategy="llm_merge")


def test_obligations_list_every_unmet_obligation():
    graph = _gated(final=True, judge_policy=FailurePolicy.SKIP)
    with pytest.raises(ValueError) as caught:
        validate_acceptance_obligations(graph, _strict())
    assert "failure_policy SKIP" in str(caught.value)
    assert "no gate judges 'final'" in str(caught.value)


@pytest.mark.parametrize("policy", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_obligations_need_a_strict_policy(policy):
    with pytest.raises(ValueError, match="needs a strict AcceptancePolicy"):
        validate_acceptance_obligations(_gated(), policy)


@pytest.mark.parametrize("case", sorted(BROKEN_GATES))
@pytest.mark.parametrize("parallel", [False, True])
def test_a_strict_swarm_refuses_a_broken_gate_before_any_call(case, parallel):
    graph, message = BROKEN_GATES[case]
    _refuses(_strict(Forbids()), graph, match=message, parallel=parallel)


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_graphs_a_strict_policy_refuses_still_run_without_one(acceptance):
    for graph in (_gated(final=True), _two_gates(), _gated(judge_policy=FailurePolicy.SKIP)):
        result = _swarm(Script({"judge": ["PASS"], "ja": ["PASS"], "jb": ["PASS"]}),
                        acceptance=acceptance).execute(graph)
        assert result.output and result.acceptance is None


def test_a_strict_swarm_refuses_a_yaml_gate_that_may_be_skipped(tmp_path):
    path = tmp_path / "skip.yaml"
    path.write_text(
        "topology: serial\nnodes:\n  - id: draft\n    label: draft\n"
        "  - id: judge\n    label: judge\n    depends_on: [draft]\n    verifies: draft\n"
        "    failure_policy: skip\n",
        encoding="utf-8",
    )
    provider = Script({})
    swarm = Swarm.from_yaml(str(path), model="test-model", provider=provider,
                            acceptance=_strict(Forbids()))
    with pytest.raises(ValueError, match="failure_policy SKIP"):
        swarm.execute()
    assert provider.runs == {}


def test_a_strict_swarm_refuses_to_plan_when_no_gate_of_the_plan_could_accept():
    provider, swarm = _planner(acceptance=_strict())
    with pytest.raises(ValueError, match="architect that generates its gates"):
        swarm.execute(Task("Write the summary"))
    with pytest.raises(ValueError, match="architect that generates its gates"):
        asyncio.run(swarm.execute_async(Task("Write the summary")))
    assert provider.calls == []

    # plan() still returns the graph to inspect; executing it is refused.
    graph = swarm.plan(Task("Write the summary"))
    assert len(provider.calls) == 1
    with pytest.raises(ValueError, match="only a gate that a generated plan"):
        swarm.execute(graph)
    assert len(provider.calls) == 1


def test_a_generated_plan_is_checked_before_any_node_runs():
    plan = json.loads(json.dumps(GATED_PLAN))
    plan["nodes"][1]["failure_policy"] = "skip"
    provider = OfflineProvider(plan=plan, responses=[UNCHECKED, "PASS"])
    swarm = Swarm(provider=provider, model="test-model", artifact_dir=None,
                  checkpoint_store=MemoryStore(), acceptance=_strict(Forbids()))

    with pytest.raises(ValueError, match="'judge' has failure_policy SKIP"):
        swarm.execute(Task("Write the summary"))
    assert len(provider.calls) == 1  # the planning call only


# ---------------------------------------------------------------------------
# Supervision cannot make unchecked output accepted
# ---------------------------------------------------------------------------


def _after_draft(*extra: Node, rewire=None) -> Revision:
    """A revision that adds a ``final`` step after the gated draft, and *extra* nodes."""
    final = Node(id="final", label="final", depends_on=["draft"])
    return Revision(add_nodes=(final, *extra), rewire=rewire or {}, reason="add a final step")


def _supervised(provider, revision, *, acceptance, parallel=False):
    supervisor = ScriptedSupervisor(revision)
    swarm = _swarm(provider, acceptance=acceptance, supervisor=supervisor, max_revisions=1,
                   parallel=parallel)
    return supervisor, swarm


def _refused_revisions(result) -> list[dict]:
    return [span for span in result.trace if span.get("status") == "revision_rejected"]


@pytest.mark.parametrize("parallel", [False, True])
def test_a_step_added_after_a_gate_is_refused_without_a_final_validator(parallel):
    provider = Script({"draft": ["## Summary\nGated draft."], "judge": ["PASS"],
                       "final": ["## Summary\nUnchecked."]})
    supervisor, swarm = _supervised(provider, _after_draft(), acceptance=_strict(),
                                    parallel=parallel)

    result = swarm.execute(_gated())

    # The run keeps its plan, so the gate still judges the delivered draft.
    assert result.accepted and result.output == "## Summary\nGated draft."
    assert [node.id for node in result.graph.nodes] == ["draft", "judge"]
    assert "final" not in provider.runs and supervisor.reviews[0] == "draft"
    (refused,) = _refused_revisions(result)
    assert "strict acceptance policy refuses this revision" in refused["error"]
    assert "no gate judges 'final'" in refused["error"]


def test_a_refused_revision_leaves_no_mark_on_its_nodes():
    final = Node(id="final", label="final", depends_on=["draft"])
    revision = Revision(add_nodes=(final,), rewire={})
    _supervised(Script({"judge": ["PASS"]}), revision, acceptance=_strict())[1].execute(_gated())
    assert "added_by_revision" not in final.metadata


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_a_step_added_after_a_gate_still_applies_without_a_strict_policy(acceptance):
    provider = Script({"judge": ["PASS"], "final": ["## Summary\nUnchecked."]})
    result = _supervised(provider, _after_draft(), acceptance=acceptance)[1].execute(_gated())
    assert result.output == "## Summary\nUnchecked." and not _refused_revisions(result)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(("final_text", "decision"), [("## Summary\nDEFECT added late.", "rejected"),
                                                      ("## Summary\nClean.", "accepted")])
def test_final_validators_decide_a_step_added_after_a_gate(final_text, decision, parallel):
    provider = Script({"judge": ["PASS"], "final": [final_text]})
    supervisor, swarm = _supervised(provider, _after_draft(), acceptance=_strict(Forbids()),
                                    parallel=parallel)

    result = swarm.execute(_gated())

    assert [node.id for node in result.graph.nodes] == ["draft", "judge", "final"]
    assert result.acceptance.decision.value == decision
    assert final_text in (result.output, result.rejected_output)
    assert result.acceptance.sources[0].node_id == "final"
    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert receipt["outcome"] == "passed"  # the draft passed; the added step decides


def test_a_gate_a_revision_adds_counts_only_when_the_policy_allows_it():
    gate = Node(id="check", label="check", depends_on=["final"], verifies="final")
    provider = Script({"judge": ["PASS"], "check": ["PASS"], "final": ["## Summary\nFinal."]})

    result = _supervised(provider, _after_draft(gate), acceptance=_strict())[1].execute(_gated())
    assert result.output == "## Summary\nok" and result.accepted
    assert "only a gate that a generated plan" in _refused_revisions(result)[0]["error"]

    policy = _strict(allow_generated_gates=True)
    gate = Node(id="check", label="check", depends_on=["final"], verifies="final")
    result = _supervised(provider, _after_draft(gate), acceptance=policy)[1].execute(_gated())
    assert result.accepted and result.output == "## Summary\nFinal."
    assert result.graph.nodes[-1].metadata["added_by_revision"] is True


def test_a_revision_that_adds_a_second_delivered_node_is_refused():
    revision = Revision(add_nodes=(Node(id="notes", label="notes"),), reason="extra notes")
    result = _supervised(Script({"judge": ["PASS"]}), revision,
                         acceptance=_strict())[1].execute(_gated())
    assert result.accepted and [n.id for n in result.graph.nodes] == ["draft", "judge"]
    assert "are all delivered" in _refused_revisions(result)[0]["error"]


def test_a_revision_ahead_of_the_gated_node_still_applies():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="research", label="research"),
        Node(id="draft", label="draft", depends_on=["research"]),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])
    revision = Revision(add_nodes=(Node(id="facts", label="facts", depends_on=["research"]),),
                        rewire={"draft": ("research", "facts")}, reason="check facts first")
    provider = Script({"judge": ["PASS"], "draft": ["## Summary\nDraft."]})
    result = _supervised(provider, revision, acceptance=_strict())[1].execute(graph)

    assert result.accepted and result.output == "## Summary\nDraft."
    assert provider.runs["facts"] == 1 and not _refused_revisions(result)


def test_a_revision_that_adds_a_gate_which_may_be_skipped_is_refused():
    gate = Node(id="check", label="check", depends_on=["final"], verifies="final",
                failure_policy=FailurePolicy.SKIP)
    result = _supervised(Script({"judge": ["PASS"]}), _after_draft(gate),
                         acceptance=_strict(Forbids()))[1].execute(_gated())
    assert [n.id for n in result.graph.nodes] == ["draft", "judge"]
    assert "'check' has failure_policy SKIP" in _refused_revisions(result)[0]["error"]


def test_an_executor_refuses_only_obligations_a_revision_breaks():
    # The graph already delivers an ungated step, so a strict run could not
    # accept it; a revision that changes nothing about that still applies.
    graph = _gated(final=True)
    for node in graph.nodes:
        node.metadata["model"] = "test-model"
    revision = Revision(add_nodes=(Node(id="notes", label="notes", depends_on=["draft"]),),
                        rewire={"final": ("draft", "judge", "notes")}, reason="notes first")
    executor = Executor(Script({"judge": ["PASS"]}), Registry(), Tracer(), artifact_dir=None,
                        supervisor=ScriptedSupervisor(revision), max_revisions=1,
                        acceptance=_strict())
    executor.run(graph)
    assert executor.revisions_used == 1
    assert [n.id for n in graph.nodes] == ["draft", "judge", "final", "notes"]


class FinalReviewProvider(GatedRunProvider):
    """The judge passes the draft; a final review adds a ``polish`` step with a defect."""

    async def complete(self, system, prompt, model):
        result = await super().complete(system, prompt, model)
        label = prompt.splitlines()[0]
        if label == "judge":
            result.text = "PASS"
        elif label == "polish":
            result.text = "## Summary\nDEFECT added in the final review."
        return result


POLISH = {"change": True, "reason": "polish the deliverable",
          "add": [{"id": "polish", "label": "polish", "depends_on": ["draft"]}]}


@pytest.mark.parametrize("parallel", [False, True])
def test_a_final_llm_review_cannot_make_unchecked_output_accepted(parallel):
    provider = FinalReviewProvider(POLISH)
    swarm = _swarm(provider, acceptance=_strict(), parallel=parallel,
                   supervisor=LLMSupervisor(provider), max_revisions=1)

    result = swarm.execute(_gated())

    assert provider.calls == ["draft", "judge"]
    assert result.accepted and result.output == "body"
    assert "no gate judges 'polish'" in _refused_revisions(result)[0]["error"]

    provider = FinalReviewProvider(POLISH)
    swarm = _swarm(provider, acceptance=_strict(Forbids()), parallel=parallel,
                   supervisor=LLMSupervisor(provider), max_revisions=1)
    result = swarm.execute(_gated())

    assert provider.calls == ["draft", "judge", "polish"]
    _not_accepted(result, "rejected")
    assert result.rejected_output == "## Summary\nDEFECT added in the final review."


# ---------------------------------------------------------------------------
# Exhaustion, invalid verdicts and failing judges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sync", [True, False])
def test_exhaustion_reads_the_final_verdict_and_rejects(sync):
    provider = Script({"draft": ["## Summary\nv1", "## Summary\nv2"], "judge": ["FAIL: no sources"]})
    swarm = _swarm(provider, acceptance=_strict(Forbids()), parallel=not sync)

    result = swarm.execute(_gated(max_regenerations=1))

    _not_accepted(result, "rejected")
    assert result.rejected_output == "## Summary\nv2"
    assert provider.runs == {"draft": 2, "judge": 2}
    judge = result.graph.nodes[1]
    assert judge.metadata["regenerations_used"] == 1
    receipt = judge.metadata["verification_receipt"]
    assert receipt["outcome"] == "rejected" and receipt["exhausted"] is True
    assert receipt["passed"] is False and receipt["reason"] == "FAIL: no sources"
    assert any("judge" in reason and "no sources" in reason for reason in result.acceptance.reasons)
    # Every final validator passed, but the exhausted gate decides.
    assert result.acceptance.validators[0].outcome.value == "passed"


def test_exhaustion_decision_follows_the_policy():
    provider = Script({"judge": ["FAIL"]})
    result = _swarm(provider, acceptance=_strict(on_exhausted="needs_review")).execute(_gated())
    _not_accepted(result, "needs_review")


def test_a_rejection_before_exhaustion_still_regenerates():
    provider = Script({"draft": ["## Summary\nv1", "## Summary\nv2"], "judge": ["FAIL: thin", "PASS"]})
    result = _swarm(provider, acceptance=_strict()).execute(_gated(max_regenerations=2))
    assert result.accepted and result.output == "## Summary\nv2"
    assert provider.runs == {"draft": 2, "judge": 2}


@pytest.mark.parametrize("reply", [
    pytest.param("", id="empty"),
    pytest.param("   \n ", id="blank"),
    pytest.param('{"passed": tru', id="truncated-json"),
    pytest.param('{"passed": "yes", "reason": "fine"}', id="invalid-json-value"),
    pytest.param('{"passed": true, "passed": false}', id="conflicting-json"),
    pytest.param("Looks fine to me.", id="no-verdict"),
    pytest.param("PASS\nOne claim would FAIL review.", id="ambiguous"),
    pytest.param("PASS " * 50_000, id="too-long"),
])
@pytest.mark.parametrize("sync", [True, False])
def test_an_unreadable_verdict_is_indeterminate_without_regeneration(reply, sync):
    provider = Script({"judge": [reply]})
    swarm = _swarm(provider, acceptance=_strict(Forbids()), parallel=not sync)

    result = swarm.execute(_gated(max_regenerations=2))

    _not_accepted(result, "indeterminate")
    assert provider.runs == {"draft": 1, "judge": 1}
    judge = result.graph.nodes[1]
    assert "regenerations_used" not in judge.metadata
    assert judge.metadata["verification_receipt"]["outcome"] == "error"


def test_invalid_verdict_decision_follows_the_policy():
    result = _swarm(Script({"judge": ["no verdict"]}),
                    acceptance=_strict(on_invalid="needs_review")).execute(_gated())
    _not_accepted(result, "needs_review")


def test_a_skipped_judge_leaves_acceptance_indeterminate():
    # A Swarm refuses a judge that may be skipped; one skipped anyway decides nothing.
    _refuses(_strict(Forbids()), _gated(judge_policy=FailurePolicy.SKIP),
             match="'judge' has failure_policy SKIP")
    provider = Script({"judge": [RuntimeError("judge unavailable")]})
    graph = _gated(judge_policy=FailurePolicy.SKIP)
    record, _ = _decided(graph, _strict(Forbids()), provider)

    assert graph.nodes[1].status is NodeStatus.SKIPPED
    assert "verification_receipt" not in graph.nodes[1].metadata
    assert record.decision.value == "indeterminate"
    assert any("judge" in reason and "skipped" in reason for reason in record.reasons)


def test_a_failed_judge_leaves_acceptance_indeterminate():
    graph = _gated()
    draft, judge = graph.nodes
    draft.status, draft.result = NodeStatus.COMPLETED, "## Summary\nDraft."
    judge.status, judge.result = NodeStatus.FAILED, "provider error"

    record = evaluate_delivery(_strict(Forbids()), graph, draft.result, strategy="deliverable",
                               sources=[draft])

    assert record.decision.value == "indeterminate"
    assert any("judge" in reason and "failed" in reason for reason in record.reasons)


def test_a_raising_verifier_is_indeterminate_without_regeneration_or_abort():
    def broken(verifier_node, target):
        raise ZeroDivisionError("division by zero")

    provider = Script({"judge": ["PASS"]})
    result = _swarm(provider, acceptance=_strict(Forbids()),
                    verifier=CallableVerifier(broken)).execute(_gated(max_regenerations=2))

    _not_accepted(result, "indeterminate")
    assert provider.runs == {"draft": 1, "judge": 1}
    judge = result.graph.nodes[1]
    assert "regenerations_used" not in judge.metadata
    receipt = judge.metadata["verification_receipt"]
    assert receipt["state"] == "consumed" and receipt["outcome"] == "error"
    assert receipt["reason"] == "verifier raised ZeroDivisionError"


def test_a_raising_verifier_still_stops_a_legacy_run():
    def broken(verifier_node, target):
        raise ZeroDivisionError("division by zero")

    swarm = _swarm(Script({}), verifier=CallableVerifier(broken))
    with pytest.raises(ZeroDivisionError):
        swarm.execute(_gated())


class _ReturnsBool(Verifier):
    def verdict(self, verifier_node, verified_node):
        return True


@pytest.mark.parametrize("verifier", [
    pytest.param(CallableVerifier(lambda v, t: "PASS"), id="string"),
    pytest.param(CallableVerifier(lambda v, t: 1), id="int"),
    pytest.param(CallableVerifier(lambda v, t: Verdict(passed="yes")), id="verdict-string"),
    pytest.param(CallableVerifier(lambda v, t: Verdict(passed=True, invalid=True)),
                 id="verdict-marked-invalid"),
    pytest.param(CallableVerifier(lambda v, t: ValidationReport("x", outcome="error")),
                 id="error-report"),
    pytest.param(_ReturnsBool(), id="bare-bool"),
])
def test_a_verdict_that_is_not_a_bool_is_never_accepted(verifier):
    provider = Script({"judge": ["PASS"]})
    result = _swarm(provider, acceptance=_strict(), verifier=verifier).execute(_gated())
    _not_accepted(result, "indeterminate")
    assert provider.runs == {"draft": 1, "judge": 1}


@pytest.mark.parametrize(("validator", "detail"), [
    pytest.param(Returns(ZeroDivisionError("boom")), "raised ZeroDivisionError", id="raises"),
    pytest.param(Returns(True), "returned bool", id="bool"),
    pytest.param(Returns("passed"), "returned str", id="string"),
    pytest.param(Returns(ValidationReport("x", outcome="error")), "could not decide", id="error"),
    pytest.param(Returns(ValidationReport(
        "x", validator=ValidatorIdentity.for_config("someone-else", "1"))),
        "another validator", id="other-identity"),
])
def test_a_final_validator_that_cannot_decide_is_indeterminate(validator, detail):
    provider = Script({"judge": ["PASS"]})
    result = _swarm(provider, acceptance=_strict(validator)).execute(_gated())

    _not_accepted(result, "indeterminate")
    assert result.acceptance.validators[0].outcome.value == "error"
    assert any(detail in reason for reason in result.acceptance.reasons)
    assert provider.runs == {"draft": 1, "judge": 1}


def test_an_async_validate_result_is_closed_and_indeterminate():
    class Lazy:
        identity = ValidatorIdentity.for_config("lazy", "1")

        def validate(self, subject):
            return self._check()

        async def _check(self):
            return ValidationReport("x")

    result = _swarm(Script({"judge": ["PASS"]}), acceptance=_strict(Lazy())).execute(_gated())
    _not_accepted(result, "indeterminate")


# ---------------------------------------------------------------------------
# Gates without regeneration, artifacts and supporting records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("reply", "decision"), [("PASS", "accepted"), ("FAIL: thin", "rejected"),
                                                 ("unsure", "indeterminate")])
def test_an_advisory_gate_verdict_is_read_and_recorded(reply, decision):
    provider = Script({"judge": [reply]})
    result = _swarm(provider, acceptance=_strict()).execute(_gated(max_regenerations=0))

    assert result.acceptance.decision.value == decision
    judge = result.graph.nodes[1]
    assert "verification_receipt" not in judge.metadata
    recorded = judge.metadata["advisory_verdict"]
    assert recorded["target_id"] == "draft" and recorded["target_generation"] == 0
    assert recorded["outcome"] == {"accepted": "passed", "rejected": "rejected",
                                   "indeterminate": "error"}[decision]
    assert provider.runs == {"draft": 1, "judge": 1}


def test_a_legacy_advisory_gate_is_never_read():
    result = _swarm(Script({"judge": ["FAIL"]})).execute(_gated(max_regenerations=0))
    assert "advisory_verdict" not in result.graph.nodes[1].metadata
    assert result.output == "## Summary\nok" and result.acceptance is None


def test_an_advisory_verdict_follows_regeneration_of_its_target():
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="review", label="review", depends_on=["draft"], verifies="draft"),
        Node(id="judge", label="judge", depends_on=["draft", "review"], verifies="draft",
             max_regenerations=1),
    ])
    provider = Script({"draft": ["## Summary\nv1", "## Summary\nv2"], "review": ["PASS"],
                       "judge": ["FAIL: thin", "PASS"]})
    result = _swarm(provider, acceptance=_strict()).execute(graph)

    assert result.accepted and result.output == "## Summary\nv2"
    assert provider.runs == {"draft": 2, "review": 2, "judge": 2}
    recorded = result.graph.nodes[1].metadata["advisory_verdict"]
    assert recorded["target_generation"] == 1 and recorded["judge_generation"] == 1


_PASSED_ADVISORY_VERDICT = {"version": 1, "judge_generation": 0, "target_id": "draft",
                            "target_generation": 0, "outcome": "passed", "passed": True}


@pytest.mark.parametrize("sync", [True, False])
def test_a_rerun_advisory_judge_records_its_new_verdict(sync):
    graph = _gated(max_regenerations=0)
    first = _swarm(Script({"judge": ["PASS"]}), acceptance=_strict(), parallel=not sync).execute(graph)
    assert first.accepted
    for node in graph.nodes:
        node.status, node.result = NodeStatus.PENDING, None

    provider = Script({"draft": ["## Summary\nDEFECTIVE"], "judge": ["FAIL: defective"]})
    result = _swarm(provider, acceptance=_strict(), parallel=not sync).execute(graph)

    _not_accepted(result, "rejected")
    assert result.rejected_output == "## Summary\nDEFECTIVE"
    assert provider.runs == {"draft": 1, "judge": 1}
    recorded = result.graph.nodes[1].metadata["advisory_verdict"]
    assert recorded["outcome"] == "rejected" and recorded["reason"] == "FAIL: defective"


@pytest.mark.parametrize("parallel", [False, True])
def test_an_advisory_verdict_in_a_yaml_file_does_not_decide_the_run(tmp_path, parallel):
    path = tmp_path / "seeded.yaml"
    path.write_text(
        "topology: serial\nnodes:\n  - id: draft\n    label: draft\n"
        "  - id: judge\n    label: judge\n    depends_on: [draft]\n    verifies: draft\n"
        f"    metadata:\n      advisory_verdict: {json.dumps(_PASSED_ADVISORY_VERDICT)}\n",
        encoding="utf-8",
    )
    provider = Script({"judge": ["FAIL: defective"]})
    swarm = Swarm.from_yaml(str(path), model="test-model", provider=provider, parallel=parallel,
                            acceptance=_strict())
    assert swarm._yaml_graph.nodes[1].metadata["advisory_verdict"] == _PASSED_ADVISORY_VERDICT

    result = swarm.execute()

    _not_accepted(result, "rejected")
    recorded = result.graph.nodes[1].metadata["advisory_verdict"]
    assert recorded["outcome"] == "rejected" and recorded["reason"] == "FAIL: defective"


@pytest.mark.parametrize("executor", [Executor, AsyncExecutor])
def test_an_executor_replaces_an_advisory_verdict_when_its_judge_runs(executor):
    graph = _gated(max_regenerations=0)
    for node in graph.nodes:
        node.metadata["model"] = "test-model"
    graph.nodes[1].metadata["advisory_verdict"] = dict(_PASSED_ADVISORY_VERDICT)
    provider = Script({"judge": ["FAIL: defective"]})

    run = executor(provider, Registry(), Tracer(), artifact_dir=None, acceptance=_strict()).run(graph)
    if inspect.isawaitable(run):
        asyncio.run(run)

    recorded = graph.nodes[1].metadata["advisory_verdict"]
    assert recorded["outcome"] == "rejected" and recorded["reason"] == "FAIL: defective"


def _completed_gate(max_regenerations: int) -> ExecutionGraph:
    """A gate whose judge completed before the run, with a passing verdict recorded."""
    graph = _gated(max_regenerations=max_regenerations)
    draft, judge = graph.nodes
    draft.status, draft.result = NodeStatus.COMPLETED, "## Summary\nDEFECTIVE"
    judge.status, judge.result = NodeStatus.COMPLETED, "FAIL: defective"
    judge.metadata["verification_target_generation"] = 0
    if max_regenerations:
        judge.metadata["verification_receipt"] = {
            "version": 1, "state": "consumed", "judge_generation": 0, "target_id": "draft",
            "target_generation": 0, "passed": True, "outcome": "passed", "reason": "",
        }
    else:
        judge.metadata["advisory_verdict"] = dict(_PASSED_ADVISORY_VERDICT)
    return graph


@pytest.mark.parametrize("max_regenerations", [0, 1])
@pytest.mark.parametrize("parallel", [False, True])
def test_a_fresh_strict_run_refuses_a_judge_that_completed_with_a_verdict(max_regenerations, parallel):
    provider = Script({})
    swarm = _swarm(provider, acceptance=_strict(), parallel=parallel)
    with pytest.raises(ValueError, match="'judge' completed before this run"):
        swarm.execute(_completed_gate(max_regenerations))
    assert provider.runs == {}

    # A finished strict run's graph is refused too; resuming its checkpoint is not.
    store = MemoryStore()
    first = _swarm(Script({"judge": ["PASS"]}), acceptance=_strict(), store=store,
                   parallel=parallel).execute(_gated(max_regenerations=max_regenerations))
    assert first.accepted
    with pytest.raises(ValueError, match="'judge' completed before this run"):
        _swarm(provider, acceptance=_strict(), parallel=parallel).execute(first.graph)
    assert _swarm(provider, acceptance=_strict(), store=store).resume(first.execution_id).accepted
    assert provider.runs == {}


def _completed_judge_without_a_verdict(kind: str) -> ExecutionGraph:
    """A gate whose judge completed before the run with a reply but no recorded verdict."""
    graph = _gated(max_regenerations=0 if kind == "bound-advisory" else 1)
    draft, judge = graph.nodes
    draft.status, draft.result = NodeStatus.COMPLETED, "## Summary\nDEFECTIVE"
    judge.status, judge.result = NodeStatus.COMPLETED, "PASS"
    if kind != "no-record":
        judge.metadata["verification_target_generation"] = 0
    if kind == "pending-receipt":
        judge.metadata["verification_receipt"] = {
            "version": 1, "state": "pending", "judge_generation": 0, "target_id": "draft",
            "target_generation": 0,
        }
    return graph


@pytest.mark.parametrize("kind", ["pending-receipt", "bound-advisory", "no-record"])
@pytest.mark.parametrize("parallel", [False, True])
def test_a_fresh_strict_run_refuses_a_judge_that_completed_without_a_verdict(kind, parallel):
    # The run would read the judge's earlier reply, or none, instead of a verdict it produced.
    provider = Script({})
    store = MemoryStore()
    swarm = _swarm(provider, acceptance=_strict(), store=store, parallel=parallel)
    graph = _completed_judge_without_a_verdict(kind)

    with pytest.raises(ValueError, match="'judge' completed before this run"):
        swarm.execute(graph)

    assert provider.runs == {} and store.saves == []
    assert graph.nodes[1].metadata.get("verification_receipt", {}).get("state") in (None, "pending")
    assert "advisory_verdict" not in graph.nodes[1].metadata


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
def test_a_completed_gate_still_runs_without_a_strict_policy(acceptance):
    provider = Script({})
    result = _swarm(provider, acceptance=acceptance).execute(_completed_gate(1))
    assert result.output == "## Summary\nDEFECTIVE" and result.acceptance is None
    assert provider.runs == {}


def _drawing_swarm(tmp_path, policy, *, judge="PASS", store=None) -> Swarm:
    provider = Script({"judge": [judge]},
                      artifacts={"draft": [Artifact(data=b"\x89PNG", mime_type="image/png")]})
    return Swarm(provider=provider, model="test-model", artifact_dir=tmp_path,
                 checkpoint_store=store if store is not None else MemoryStore(), acceptance=policy)


@pytest.mark.parametrize("policy", [_strict(Forbids()), _strict()], ids=["validator", "gate"])
def test_a_gated_artifact_deliverable_is_accepted_with_its_file_digests(tmp_path, policy):
    result = _drawing_swarm(tmp_path, policy).execute(_gated())

    assert result.accepted, result.acceptance.reasons
    assert result.acceptance.artifacts == (AcceptanceArtifact(
        "draft", "draft_00.png", hashlib.sha256(b"\x89PNG").hexdigest(), 4, "image/png",
    ),)
    data = result.acceptance.to_dict()
    assert data["artifacts"] == [{
        "node_id": "draft", "name": "draft_00.png",
        "sha256": hashlib.sha256(b"\x89PNG").hexdigest(), "size_bytes": 4, "mime_type": "image/png",
    }]
    assert DeliveryAcceptance.from_dict(data) == result.acceptance


def test_delivered_files_need_a_passed_gate_on_their_node(tmp_path):
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="draft", label="draft")])
    result = _drawing_swarm(tmp_path, _strict(Forbids())).execute(graph)

    _not_accepted(result, "indeterminate")
    assert any("no passed gate judged node 'draft'" in reason for reason in result.acceptance.reasons)
    assert [artifact.name for artifact in result.acceptance.artifacts] == ["draft_00.png"]


def test_delivered_files_need_gate_acceptance_in_the_policy(tmp_path):
    policy = _strict(Forbids(), allow_gate_acceptance=False)
    result = _drawing_swarm(tmp_path / "files", policy).execute(_gated())

    _not_accepted(result, "indeterminate")
    assert any("does not allow gate acceptance" in reason for reason in result.acceptance.reasons)


@pytest.mark.parametrize("change", ["unhashed", "changed", "missing"])
def test_delivered_files_must_still_hold_their_recorded_bytes(tmp_path, change):
    policy = _strict(Forbids())
    result = _drawing_swarm(tmp_path, policy).execute(_gated())
    assert result.accepted
    draft = result.graph.nodes[0]
    [record] = draft.metadata["artifacts"]
    if change == "unhashed":
        del record["sha256"]
    elif change == "changed":
        Path(record["path"]).write_bytes(b"\x89PNX")
    else:
        Path(record["path"]).unlink()

    decision = evaluate_delivery(policy, result.graph, result.output, strategy="deliverable",
                                 sources=[draft])

    assert decision.decision == "indeterminate" and decision.artifacts == ()
    expected = {"unhashed": "not hashed", "changed": "changed after it was written",
                "missing": "is missing or cannot be read"}[change]
    assert any(expected in reason for reason in decision.reasons), decision.reasons


def test_completed_resume_rechecks_delivered_file_bytes(tmp_path):
    store = MemoryStore()
    first = _drawing_swarm(tmp_path, _strict(Forbids()), store=store).execute(_gated())
    assert first.accepted
    again = _drawing_swarm(tmp_path, _strict(Forbids()), store=store).resume(first.execution_id)
    assert again.acceptance == first.acceptance and again.output == first.output

    Path(first.graph.nodes[0].metadata["artifacts"][0]["path"]).write_bytes(b"\x89PNX")
    result = _drawing_swarm(tmp_path, _strict(Forbids()), store=store).resume(first.execution_id)

    _not_accepted(result, "indeterminate")
    assert result.acceptance.artifacts == first.acceptance.artifacts
    assert any("draft_00.png" in reason for reason in result.acceptance.reasons)


@pytest.mark.parametrize(("script", "settings", "decision"), [
    pytest.param({"draft": ["## Summary\nDEFECT"], "judge": ["PASS"]}, {}, "rejected",
                 id="validator-rejects"),
    pytest.param({"judge": ["FAIL: thin"]}, {}, "rejected", id="gate-exhausted"),
    pytest.param({"judge": ["FAIL: thin"]}, {"on_exhausted": "needs_review"}, "needs_review",
                 id="gate-exhausted-review"),
    pytest.param({"judge": ["PASS"]}, {}, "indeterminate", id="all-checks-pass"),
    pytest.param({"judge": ["PASS"]}, {"on_invalid": "needs_review"}, "needs_review",
                 id="all-checks-pass-review"),
])
def test_output_from_a_node_that_wrote_files_is_never_accepted(tmp_path, script, settings,
                                                                decision):
    # A valid rejection still decides; only when nothing rejects do the
    # unhashed files make the decision on_invalid.
    provider = Script(script, artifacts={"draft": [Artifact(data=b"\x89PNG", mime_type="image/png")]})
    swarm = Swarm(provider=provider, model="test-model", artifact_dir=tmp_path,
                  checkpoint_store=MemoryStore(), acceptance=_strict(Forbids(), **settings))
    result = swarm.execute(_gated())

    _not_accepted(result, decision)
    assert any("not hashed" in reason for reason in result.acceptance.reasons)


def test_trace_records_the_decision_without_new_summary_keys():
    result = _swarm(Script({"judge": ["FAIL"]}), acceptance=_strict()).execute(_gated())
    entries = [entry for entry in result.trace if entry["node_id"] == "acceptance:delivery"]
    assert len(entries) == 1
    assert entries[0]["status"] == "acceptance_rejected"
    assert entries[0]["duration_ms"] == 0
    keys = {"node_id", "label", "agent_id", "status", "duration_ms", "error"}
    assert set(entries[0]) == keys
    assert all(set(entry) >= keys for entry in result.trace)


def test_memory_records_the_decision(tmp_path):
    memory = PlannerMemory(tmp_path / "history.jsonl")
    graph = _gated()
    graph.task = Task("Write the summary")
    _swarm(Script({"judge": ["FAIL"]}), acceptance=_strict(), memory=memory).execute(graph)
    (line,) = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["acceptance"] == "rejected"


def test_from_yaml_takes_the_policy_as_a_keyword(tmp_path):
    path = tmp_path / "gated.yaml"
    path.write_text(
        "topology: serial\nnodes:\n  - id: draft\n    label: draft\n"
        "  - id: judge\n    label: judge\n    depends_on: [draft]\n"
        "    verifies: draft\n    max_regenerations: 1\n",
        encoding="utf-8",
    )
    swarm = Swarm.from_yaml(str(path), model="test-model", provider=Script({"judge": ["FAIL"]}),
                            acceptance=_strict())
    assert swarm.acceptance == _strict()
    result = swarm.execute()
    _not_accepted(result, "rejected")


# ---------------------------------------------------------------------------
# Legacy and advisory runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy()], ids=["none", "advisory"])
@pytest.mark.parametrize("parallel", [False, True])
def test_legacy_and_advisory_runs_keep_their_output_and_version_4(acceptance, parallel):
    store = MemoryStore()
    provider = Script({"draft": ["## Summary\nv1", "## Summary\nDEFECT"], "judge": ["FAIL"]})
    swarm = _swarm(provider, acceptance=acceptance, store=store, parallel=parallel)

    result = swarm.execute(_gated(max_regenerations=1))

    assert result.output == "## Summary\nDEFECT"
    assert (result.acceptance, result.rejected_output, result.accepted) == (None, None, False)
    assert all(state["version"] == 4 for state in store.saves)
    assert all("requires" not in state and "delivery_acceptance" not in state
               for state in store.saves)
    receipt = result.graph.nodes[1].metadata["verification_receipt"]
    assert "outcome" not in receipt and receipt["reason"] == "unavailable or exhausted"
    assert not any(entry["node_id"] == "acceptance:delivery" for entry in result.trace)
    assert swarm.resume(result.execution_id).acceptance is None


def test_strict_runs_write_version_5_at_every_save():
    store = MemoryStore()
    policy = _strict(Forbids())
    graph = _gated()
    graph.task = Task("Write the summary")
    result = _swarm(Script({"judge": ["PASS"]}), acceptance=policy, store=store).execute(graph)

    assert len(store.saves) > 2
    assert all(state["version"] == 5 and state["requires"] == ["delivery_acceptance"]
               for state in store.saves)
    for state in store.saves[:-1]:
        assert state["status"] == "running"
        assert state["delivery_acceptance"]["decision"] == "not_evaluated"
        assert state["delivery_acceptance"]["policy_sha256"] == policy.sha256
        assert state["delivery_acceptance"]["task_sha256"] == task_sha256(graph.task)
    assert store.saves[-1]["status"] == "completed"
    assert store.saves[-1]["delivery_acceptance"] == result.acceptance.to_dict()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _crashed_strict_run(store, policy):
    provider = Script({"judge": [RuntimeError("crash before verdict"), "PASS"]})
    swarm = _swarm(provider, acceptance=policy, store=store)
    with pytest.raises(RuntimeError, match="crash before verdict"):
        swarm.execute(_gated())
    state = store.states[store.only_id()]
    assert state["status"] == "failed" and state["version"] == 5
    return provider


def test_an_unfinished_strict_run_resumes_under_its_policy():
    store = MemoryStore()
    policy = _strict(Forbids())
    provider = _crashed_strict_run(store, policy)

    result = _swarm(provider, acceptance=_strict(Forbids()), store=store).resume(store.only_id())

    assert result.accepted and result.output == "## Summary\nok"
    assert provider.runs == {"draft": 1, "judge": 2}
    assert store.states[result.execution_id]["delivery_acceptance"]["decision"] == "accepted"


@pytest.mark.parametrize("acceptance", [None, AcceptancePolicy(), _strict(Forbids("OTHER"))],
                         ids=["none", "advisory", "other-policy"])
def test_an_unfinished_strict_run_is_refused_under_another_policy(acceptance):
    store = MemoryStore()
    provider = _crashed_strict_run(store, _strict(Forbids()))
    before = dict(provider.runs)

    with pytest.raises(ValueError, match="strict delivery acceptance"):
        _swarm(provider, acceptance=acceptance, store=store).resume(store.only_id())
    assert provider.runs == before
    assert store.states[store.only_id()]["status"] == "failed"


def test_an_unfinished_legacy_run_is_refused_under_a_strict_policy():
    store = MemoryStore()
    provider = Script({"judge": [RuntimeError("crash before verdict"), "PASS"]})
    with pytest.raises(RuntimeError):
        _swarm(provider, store=store).execute(_gated())
    before = dict(provider.runs)

    with pytest.raises(ValueError, match="without strict delivery acceptance"):
        _swarm(provider, acceptance=_strict(), store=store).resume(store.only_id())
    assert provider.runs == before


def _completed_strict_run(store, policy=None, script=None):
    provider = Script(script or {"judge": ["PASS"]})
    result = _swarm(provider, acceptance=policy or _strict(Forbids()), store=store).execute(_gated())
    return provider, result


@pytest.mark.parametrize("acceptance", [None, _strict(Forbids())], ids=["none", "same-policy"])
def test_a_completed_strict_run_returns_its_record(acceptance):
    store = MemoryStore()
    provider, first = _completed_strict_run(store)
    calls = dict(provider.runs)

    result = _swarm(provider, acceptance=acceptance, store=store).resume(first.execution_id)

    assert result.acceptance == first.acceptance and result.accepted
    assert result.output == first.output
    assert provider.runs == calls


def test_changed_output_bytes_on_completed_resume_are_not_accepted():
    store = MemoryStore()
    provider, first = _completed_strict_run(store)
    store.states[first.execution_id]["output"] = first.output + " (edited)"

    result = _swarm(provider, acceptance=_strict(Forbids()), store=store).resume(first.execution_id)

    _not_accepted(result, "indeterminate")
    assert result.rejected_output == first.output + " (edited)"
    assert any("does not match" in reason for reason in result.acceptance.reasons)
    assert provider.runs == {"draft": 1, "judge": 1}


def test_a_rejected_strict_run_stays_rejected_on_resume():
    store = MemoryStore()
    provider, first = _completed_strict_run(store, script={"judge": ["FAIL"]})
    result = _swarm(provider, acceptance=_strict(Forbids()), store=store).resume(first.execution_id)
    assert result.acceptance == first.acceptance
    assert (result.output, result.rejected_output) == ("", first.rejected_output)


def test_a_different_policy_does_not_inherit_a_completed_decision():
    store = MemoryStore()
    provider, first = _completed_strict_run(store)
    other = _strict(Forbids("OTHER"))

    result = _swarm(provider, acceptance=other, store=store).resume(first.execution_id)

    _not_accepted(result, "indeterminate")
    assert result.acceptance.policy_sha256 == other.sha256
    assert result.rejected_output == first.output


@pytest.mark.parametrize("version", [1, 3, 4, 5])
@pytest.mark.parametrize("acceptance", [_strict(), _strict(Forbids())], ids=["gate", "validator"])
def test_a_strict_swarm_withholds_a_completed_run_without_a_record(version, acceptance, caplog):
    store = MemoryStore()
    provider = Script({"draft": ["## Summary\nDEFECT"], "judge": ["PASS"]})
    first = _swarm(provider, store=store).execute(_gated())
    state = store.states[first.execution_id]
    assert (state["version"], first.output) == (4, "## Summary\nDEFECT")
    # Version 5 without the requirement holds no record either.
    state["version"] = version
    if version == 5:
        state["requires"] = []

    with caplog.at_level("WARNING", logger="smythe.swarm"):
        result = _swarm(provider, acceptance=acceptance, store=store).resume(first.execution_id)

    assert (result.output, result.rejected_output) == ("", "## Summary\nDEFECT")
    assert (result.acceptance, result.accepted) == (None, False)
    assert provider.runs == {"draft": 1, "judge": 1}
    assert "no delivery acceptance record" in caplog.text
    assert "evaluate_delivery" in caplog.text and "re-run the task" in caplog.text
    # The checkpoint is left as it was, and a Swarm without a strict policy
    # still returns the stored text as the legacy run's output.
    assert store.states[first.execution_id] == state
    for legacy in (None, AcceptancePolicy()):
        again = _swarm(provider, acceptance=legacy, store=store).resume(first.execution_id)
        assert (again.output, again.rejected_output, again.acceptance) == (first.output, None, None)


@pytest.mark.parametrize("change", ["task", "source"])
def test_completed_resume_reverifies_the_bound_identities(change):
    store = MemoryStore()
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="draft", label="draft")])
    graph.task = Task("Write the summary")
    provider = Script({})
    first = _swarm(provider, acceptance=_strict(Forbids()), store=store).execute(graph)
    assert first.accepted
    state = store.states[first.execution_id]
    if change == "task":
        state["task"]["goal"] = state["graph"]["task"]["goal"] = "Write something else"
    else:
        state["graph"]["nodes"][0]["metadata"]["execution_generation"] = 3

    result = _swarm(provider, acceptance=_strict(Forbids()), store=store).resume(first.execution_id)

    _not_accepted(result, "indeterminate")
    assert provider.runs == {"draft": 1}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_swarm_rejects_a_value_that_is_not_a_policy():
    with pytest.raises(TypeError, match="AcceptancePolicy"):
        Swarm(provider=Script({}), model="test-model", acceptance={"mode": "strict"})


def test_a_durable_swarm_takes_its_strict_policy_into_the_run_recipe(tmp_path):
    from smythe import OfflineProvider, SQLiteWorkflowStore

    policy = _strict(Forbids())
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        runtime = Swarm(provider=OfflineProvider(), model="offline", run_store=store,
                        acceptance=policy)._workflow_runtime()
        assert runtime.acceptance is policy
        assert runtime.recipe["acceptance"] == policy.to_dict()
        # Gate acceptance alone still needs DELIVERABLE synthesis.
        with pytest.raises(ValueError, match="final validator"):
            Swarm(provider=OfflineProvider(), model="offline", run_store=store,
                  synthesizer=Synthesizer(SynthesisStrategy.CONCATENATE), acceptance=_strict())


def test_executors_take_the_policy_as_the_last_keyword_only_argument():
    for executor in (ExecutorBase, AsyncExecutor):
        parameters = list(inspect.signature(executor).parameters.values())
        assert parameters[-1].name == "acceptance"
        assert parameters[-1].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="AcceptancePolicy"):
        Executor(Script({}), Registry(), Tracer(), acceptance="strict")


def test_an_executor_reads_strict_verdicts_without_a_swarm():
    graph = _gated(max_regenerations=1)
    for node in graph.nodes:
        node.metadata["model"] = "test-model"
    provider = Script({"judge": ["FAIL: thin"]})
    Executor(provider, Registry(), Tracer(), artifact_dir=None, acceptance=_strict()).run(graph)
    assert graph.nodes[1].metadata["verification_receipt"]["outcome"] == "rejected"


# ---------------------------------------------------------------------------
# Reading verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("reply", "invalid"), [
    ("PASS", False), ("FAIL: thin", False), ('{"passed": false, "reason": "thin"}', False),
    ("Verdict: FAIL", False), ("", True), ("no verdict here", True), ('{"passed": "no"}', True),
    ("PASS\nbut it would FAIL", True), ("The draft does not PASS.", True),
])
def test_token_verifier_marks_unreadable_verdicts_invalid(reply, invalid):
    verdict = TokenVerifier().verdict(Node(id="j", label="j", result=reply), Node(id="t", label="t"))
    assert verdict.invalid is invalid
    if invalid:
        assert verdict.passed is False


def test_invalid_marking_leaves_verdict_equality_unchanged():
    assert Verdict(False, "unreadable", invalid=True) == Verdict(False, "unreadable")
    assert CallableVerifier(lambda v, t: "FAIL").verdict(Node(id="j", label="j"),
                                                         Node(id="t", label="t")).invalid is True
    assert CallableVerifier(lambda v, t: False).verdict(Node(id="j", label="j"),
                                                        Node(id="t", label="t")).invalid is False
