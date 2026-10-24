"""Evidence-triggered supervision: deterministic triggers decide when a review happens."""

from __future__ import annotations

import asyncio
import json

import pytest

from smythe import (
    AcceptancePolicy,
    EvidenceTriggeredSupervisor,
    LocalOnly,
    ModelProfile,
    NoAllowedProfileError,
    OfflineProvider,
    ProviderFeatures,
    RetryPolicy,
    ReviewTriggers,
    SimpleArchitect,
    SQLiteWorkflowStore,
    Swarm,
    ValidationReport,
    ValidatorIdentity,
    WorkflowBindingError,
)
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.evidence import ClaimPassage, ClaimRecord, SourceEvidence, render_source_evidence
from smythe.executor import Executor
from smythe.graph import REVISION_ADDED_KEY, ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import (
    EVIDENCE_REVIEW_PROMPT,
    REVIEW_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    FiredTrigger,
    LLMSupervisor,
    Supervisor,
)
from smythe.tools import TOOL_ERRORS_KEY, ToolCall, ToolResult
from smythe.tracer import Tracer
from smythe.validation import hard_finding
from test_tool_loop import SimpleRuntime

NO_CHANGE = json.dumps({"change": False, "reason": "the plan still fits"})


class Script(Provider):
    """Answers each node by its label and every review with *review*, recording the reviews."""

    def __init__(self, answers: dict | None = None, *, review: str = NO_CHANGE) -> None:
        self.answers = answers or {}
        self.review = review
        self.reviews: list[str] = []
        self.calls: list[str] = []

    async def complete(self, system, prompt, model):
        if system == SUPERVISOR_SYSTEM_PROMPT:
            self.reviews.append(prompt)
            return CompletionResult(self.review, prompt_tokens=1, completion_tokens=1)
        label = prompt.splitlines()[0]
        self.calls.append(label)
        answer = self.answers.get(label, f"{label} done")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, CompletionResult):
            return answer
        return CompletionResult(answer, prompt_tokens=1, completion_tokens=1)


class ToolScript(Script):
    """The step makes two tool calls, then answers in text."""

    async def chat(self, system, messages, model, tools=None):
        if len(messages) == 1:
            calls = [ToolCall("t1", "calc.add", {}), ToolCall("t2", "calc.add", {})]
            return CompletionResult("", prompt_tokens=1, completion_tokens=1,
                                    tool_calls=calls, stop_reason="tool_use")
        return CompletionResult("the sum is unavailable", prompt_tokens=1, completion_tokens=1)


def _graph(*nodes: Node) -> ExecutionGraph:
    for node in nodes:
        node.metadata.setdefault("model", "test-model")
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=list(nodes))


def _run(graph, provider, supervisor, *, parallel=False, max_revisions=2, **options):
    """Run *graph* on an executor; return the executor and its tracer."""
    tracer = Tracer()
    executor_class = AsyncExecutor if parallel else Executor
    executor = executor_class(provider=provider, registry=Registry(), tracer=tracer,
                              artifact_dir=None, supervisor=supervisor,
                              max_revisions=max_revisions, **options)
    if parallel:
        asyncio.run(executor.run(graph))
    else:
        executor.run(graph)
    return executor, tracer


def _triggered(tracer) -> list[dict]:
    return [span["review_trigger"] for span in tracer.summary() if "review_trigger" in span]


def _model_review(provider, triggers=None, **options):
    return EvidenceTriggeredSupervisor(LLMSupervisor(provider, **options), triggers)


# ---------------------------------------------------------------------------
# No trigger, no review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_run_without_a_trigger_makes_no_supervision_call(parallel):
    graph = _graph(
        Node(id="left", label="left"), Node(id="right", label="right"),
        Node(id="join", label="join", depends_on=["left", "right"]),
        Node(id="final", label="final", depends_on=["join"]),
    )
    provider = Script()

    executor, tracer = _run(graph, provider, _model_review(provider), parallel=parallel)

    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes)
    assert provider.reviews == [] and _triggered(tracer) == []
    assert executor.revision_log == [] and executor.revisions_used == 0
    # The same run under routine review asks the model after every node.
    routine = Script()
    _run(_graph(*(Node(id=node.id, label=node.label, depends_on=list(node.depends_on))
                  for node in graph.nodes)),
         routine, LLMSupervisor(routine, only_terminal=False), parallel=parallel)
    assert len(routine.reviews) == 4


def test_a_swarm_run_without_a_trigger_makes_no_supervision_call():
    provider = Script()
    swarm = Swarm(provider=provider, model="test-model", artifact_dir=None, max_revisions=2,
                  supervisor=_model_review(provider))
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="draft"), Node(id="final", label="final", depends_on=["draft"]),
    ])

    result = swarm.execute(graph)

    assert result.output == "final done"
    assert provider.reviews == [] and provider.calls == ["draft", "final"]
    assert not [span for span in result.trace if span["node_id"].startswith("supervisor:")]


# ---------------------------------------------------------------------------
# Each anomaly fires one review that carries its evidence
# ---------------------------------------------------------------------------


def _contradicted() -> str:
    claim = ClaimRecord("growth", "Revenue grew", supporting=(ClaimPassage("q3", "grew 4%"),),
                        contradicting=(ClaimPassage("audit", "fell 2%"),))
    return "Revenue grew.\n\n" + render_source_evidence(SourceEvidence(claims=(claim,)))


def _skipping(answer) -> tuple[ExecutionGraph, Script, dict]:
    graph = _graph(Node(id="a", label="a"),
                   Node(id="b", label="b", depends_on=["a"], failure_policy=FailurePolicy.SKIP),
                   Node(id="c", label="c", depends_on=["b"]))
    return graph, Script({"b": answer}), {}


def _gated(answer: str, **gate) -> tuple[ExecutionGraph, Script, dict]:
    graph = _graph(Node(id="draft", label="draft"),
                   Node(id="check", label="check", depends_on=["draft"], verifies="draft", **gate))
    return graph, Script({"check": answer}), {}


def _costly() -> tuple[ExecutionGraph, Script, dict]:
    graph = _graph(Node(id="a", label="a", metadata={"estimated_cost_usd": 0.001}))
    answer = CompletionResult("a done", prompt_tokens=1, completion_tokens=1, cost_usd=0.01)
    return graph, Script({"a": answer}), {"budget": Sentinel(max_budget_usd=1.0)}


def _tool_errors() -> tuple[ExecutionGraph, Script, dict]:
    runtime = SimpleRuntime(handler=lambda call: ToolResult(call.id, "calculator offline",
                                                            is_error=True))
    return _graph(Node(id="a", label="a")), ToolScript(), {"tool_runtime": runtime}


ANOMALIES = {
    "failed_or_skipped": (
        lambda: _skipping(RuntimeError("source offline")), ReviewTriggers(), "c",
        "'b' skipped: source offline"),
    "verifier_rejected_or_exhausted": (
        lambda: _gated("FAIL - no sources", max_regenerations=1), ReviewTriggers(), "check",
        "gate 'check' did not pass 'draft' after 1 regeneration(s): unavailable or exhausted"),
    "acceptance_not_accepted": (
        lambda: (*_gated("maybe")[:2], {"acceptance": AcceptancePolicy(mode="strict")}),
        ReviewTriggers(), "check", "gate 'check' has no valid verdict on 'draft'"),
    "contradiction_records": (
        lambda: (_graph(Node(id="a", label="a")), Script({"a": _contradicted()}), {}),
        ReviewTriggers(), "a", "claim 'growth' has 1 contradicting passage(s) from audit"),
    "incomplete_output": (
        lambda: _skipping(CompletionResult("partial", prompt_tokens=1, completion_tokens=1,
                                           stop_reason="max_tokens")),
        ReviewTriggers(), "c", "'b' ended without complete output (stop_reason='max_tokens')"),
    "empty_or_short_output": (
        lambda: (_graph(Node(id="a", label="a")), Script({"a": "  \n "}), {}),
        ReviewTriggers(), "a", "'a' returned 0 characters; at least 1 expected"),
    "cost_over_estimate": (
        _costly, ReviewTriggers(cost_over_estimate=2.0), "a",
        "'a' cost $0.010000, 10.0 times its estimate of $0.001000"),
    "tool_errors": (
        _tool_errors, ReviewTriggers(min_tool_errors=2), "a", "2 tool calls of 'a' returned errors"),
}


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("anomaly", sorted(ANOMALIES))
def test_each_anomaly_fires_exactly_one_review_carrying_its_evidence(anomaly, parallel):
    build, triggers, after, evidence = ANOMALIES[anomaly]
    graph, provider, options = build()

    executor, tracer = _run(graph, provider, _model_review(provider, triggers),
                            parallel=parallel, **options)

    [prompt] = provider.reviews
    assert f"- {anomaly}: {evidence}" in prompt
    [traced] = _triggered(tracer)
    assert traced["after_node"] == after and anomaly in traced["triggers"]
    # The review still carries the usual prompt, after the evidence.
    assert prompt.startswith(EVIDENCE_REVIEW_PROMPT.split("{evidence}")[0])
    assert f"The step that just finished was {after!r}." in prompt
    assert executor.revision_log == []  # No change proposed, so no decision to log.


def test_a_strict_gate_that_rejects_fires_both_gate_anomalies_in_one_review():
    graph, provider, _ = _gated("FAIL - the figures are unsourced")

    _run(graph, provider, _model_review(provider),
         acceptance=AcceptancePolicy(mode="strict"))

    [prompt] = provider.reviews
    assert ("- verifier_rejected_or_exhausted: gate 'check' did not pass 'draft': "
            "FAIL - the figures are unsourced") in prompt
    assert "- acceptance_not_accepted: gate 'check' has a failing final verdict on 'draft'" in prompt


def test_a_review_point_fires_once_and_a_pending_join_fires_when_it_becomes_ready():
    graph = _graph(Node(id="left", label="left"), Node(id="right", label="right"),
                   Node(id="join", label="join", depends_on=["left", "right"]))
    provider = Script()
    triggers = ReviewTriggers(review_after={"left"}, fan_in=True, final=True)

    _, tracer = _run(graph, provider, _model_review(provider, triggers))

    assert [(span["after_node"], span["triggers"]) for span in _triggered(tracer)] == [
        ("left", ["review_after"]), ("right", ["fan_in"]), ("join", ["final"])]
    assert len(provider.reviews) == 3
    assert "- fan_in: 'join', which joins 2 inputs, is ready to run" in provider.reviews[1]
    assert "- final: no step is pending or running" in provider.reviews[2]


def test_a_gate_that_passes_and_full_output_fire_nothing():
    graph, provider, _ = _gated("PASS", max_regenerations=1)

    _run(graph, provider, _model_review(provider), acceptance=AcceptancePolicy(mode="strict"))

    assert provider.reviews == []


def test_triggers_read_recorded_state_without_running_anything():
    triggers = ReviewTriggers(review_after={"a"}, cost_over_estimate=1.5, min_tool_errors=1)
    node = Node(id="a", label="a", status=NodeStatus.COMPLETED, result="text",
                metadata={"cost_usd": 0.03, "estimated_cost_usd": 0.01, TOOL_ERRORS_KEY: 1})
    graph = ExecutionGraph([Topology.SERIAL], [node])

    fired = triggers.fired(graph, node)

    assert [trigger.name for trigger in fired] == ["review_after", "cost_over_estimate",
                                                   "tool_errors"]
    assert fired == triggers.fired(graph, node)
    # An unknown cost, or a cost without an estimate, is not over its estimate.
    node.metadata["cost_usd_unknown"] = True
    assert "cost_over_estimate" not in [trigger.name for trigger in triggers.fired(graph, node)]
    del node.metadata["cost_usd_unknown"], node.metadata["estimated_cost_usd"]
    assert "cost_over_estimate" not in [trigger.name for trigger in triggers.fired(graph, node)]
    # A switch or threshold turned off fires nothing.
    contradicted = Node(id="c", label="c", status=NodeStatus.COMPLETED, result=_contradicted())
    empty = Node(id="e", label="e", status=NodeStatus.COMPLETED, result="")
    quiet = ReviewTriggers(contradiction_records=False, min_output_chars=None)
    for item in (contradicted, empty):
        assert quiet.fired(ExecutionGraph([Topology.SERIAL], [item]), item) == ()


def test_an_unreadable_evidence_block_fires_no_contradiction_trigger():
    node = Node(id="a", label="a", status=NodeStatus.COMPLETED,
                result="text\n```source-evidence\n{not json\n```")
    assert ReviewTriggers().fired(ExecutionGraph([Topology.SERIAL], [node]), node) == ()


# ---------------------------------------------------------------------------
# Deterministic actions and inner supervisors
# ---------------------------------------------------------------------------


def _add_redo(graph, node, fired):
    return Revision(
        add_nodes=(Node(id=f"redo-{node.id}", label=f"redo {node.id}", depends_on=[node.id]),),
        reason="; ".join(line for trigger in fired for line in trigger.evidence),
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_deterministic_action_adds_a_step_without_a_model_call_and_records_the_diff(parallel):
    provider = Script({"a": ""})
    graph = _graph(Node(id="a", label="a"))
    supervisor = EvidenceTriggeredSupervisor(None, ReviewTriggers(), on_trigger=_add_redo)

    executor, tracer = _run(graph, provider, supervisor, parallel=parallel)

    assert provider.reviews == [] and provider.calls == ["a", "redo a"]
    redo = graph.nodes[-1]
    assert (redo.id, redo.status, redo.metadata[REVISION_ADDED_KEY]) == (
        "redo-a", NodeStatus.COMPLETED, True)
    [entry] = executor.revision_log
    assert entry == {
        "after_node": "a", "applied": True, "change": "+1", "added": ["redo-a"], "dropped": [],
        "rewired": [], "retained": ["a"], "invalidated": ["a"], "preserved_gates": [],
        "revisions_remaining_after": 1, "added_total": 1, "trigger": "empty_or_short_output",
        "reason": "'a' returned 0 characters; at least 1 expected", "detail": None,
    }
    [span] = [span for span in tracer.summary() if "revision" in span]
    assert span["status"] == "revision_applied"
    assert span["revision"]["trigger"] == "empty_or_short_output"


def test_an_action_that_returns_none_lets_the_inner_supervisor_review():
    provider = Script({"a": ""})
    seen = []

    async def decline(graph, node, fired):
        seen.append([trigger.name for trigger in fired])
        return None

    supervisor = EvidenceTriggeredSupervisor(LLMSupervisor(provider), on_trigger=decline)
    _run(_graph(Node(id="a", label="a")), provider, supervisor)

    assert seen == [["empty_or_short_output"]] and len(provider.reviews) == 1
    # Without an inner supervisor, None keeps the plan.
    quiet = Script({"a": ""})
    executor, _ = _run(_graph(Node(id="a", label="a")), quiet,
                       EvidenceTriggeredSupervisor(None, on_trigger=lambda *_: None))
    assert quiet.reviews == [] and executor.revision_log == []


def test_a_custom_inner_supervisor_reviews_only_when_a_trigger_fires():
    reviewed = []

    class Notes(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            reviewed.append(node.id)

    graph, provider, _ = _skipping(RuntimeError("down"))
    _run(graph, provider, EvidenceTriggeredSupervisor(Notes()))

    assert reviewed == ["c"]


def test_an_inner_llm_supervisor_keeps_its_proposal_caps():
    proposal = {"change": True, "reason": "more", "add": [
        {"id": f"x{index}", "label": "more", "depends_on": ["a"]} for index in range(2)]}
    provider = Script({"a": ""}, review=json.dumps(proposal))
    graph = _graph(Node(id="a", label="a"))

    executor, _ = _run(graph, provider, _model_review(provider, max_added_nodes=1))

    assert len(provider.reviews) == 1 and executor.revision_log == []
    assert [node.id for node in graph.nodes] == ["a"]


def test_the_wrapped_prompt_is_the_evidence_then_the_unchanged_review_prompt():
    node = Node(id="a", label="a", status=NodeStatus.COMPLETED, result="")
    graph = ExecutionGraph([Topology.SERIAL], [node])
    provider = Script()
    inner = LLMSupervisor(provider)

    asyncio.run(EvidenceTriggeredSupervisor(inner).review(graph, node, task=None,
                                                          revisions_remaining=1))

    [prompt] = provider.reviews
    standard = inner._build_prompt(graph, node, None, 1)
    assert prompt == EVIDENCE_REVIEW_PROMPT.format(
        evidence="- empty_or_short_output: 'a' returned 0 characters; at least 1 expected",
        review=standard)
    assert standard.startswith(REVIEW_PROMPT.split("{task_description}")[0])
    # LLMSupervisor's own review still asks for its schedule and sends its own prompt.
    asyncio.run(inner.review(graph, node, task=None, revisions_remaining=1))
    assert provider.reviews[-1] == standard


def test_no_review_without_revisions_left():
    node = Node(id="a", label="a", status=NodeStatus.COMPLETED, result="")
    graph = ExecutionGraph([Topology.SERIAL], [node])
    calls = []
    supervisor = EvidenceTriggeredSupervisor(None, on_trigger=lambda *args: calls.append(args))

    assert asyncio.run(supervisor.review(graph, node, task=None, revisions_remaining=0)) is None
    assert calls == []


# ---------------------------------------------------------------------------
# Strict acceptance still decides an added final step
# ---------------------------------------------------------------------------


class Forbids:
    """A final validator: the delivered text must not contain ``marker``."""

    def __init__(self, marker: str = "DEFECT") -> None:
        self.marker = marker
        self.identity = ValidatorIdentity.for_config("forbids", "1", {"marker": marker})

    def validate(self, subject):
        if self.marker in subject.text():
            return ValidationReport("output", (
                hard_finding("forbidden_text", f"output contains {self.marker!r}"),
            ), validator=self.identity)
        return ValidationReport("output", validator=self.identity)


def test_a_triggered_final_step_is_still_strictly_validated():
    proposal = {"change": True, "reason": "reconcile the contradiction", "add": [
        {"id": "final", "label": "final", "depends_on": ["draft"]}]}
    provider = Script({"draft": _contradicted(), "final": "Reconciled, with a DEFECT."},
                      review=json.dumps(proposal))
    swarm = Swarm(provider=provider, model="test-model", artifact_dir=None, max_revisions=1,
                  supervisor=_model_review(provider),
                  acceptance=AcceptancePolicy(mode="strict", final_validators=[Forbids()]))

    result = swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="draft")]))

    assert len(provider.reviews) == 1 and provider.calls == ["draft", "final"]
    assert result.acceptance.decision.value == "rejected" and result.output == ""
    assert result.rejected_output == "Reconciled, with a DEFECT."
    assert [source.node_id for source in result.acceptance.sources] == ["final"]


def test_a_triggered_step_after_a_gated_output_is_refused_without_final_validators():
    graph, provider, _ = _gated("PASS")
    supervisor = EvidenceTriggeredSupervisor(
        None, ReviewTriggers(review_after={"check"}),
        on_trigger=lambda graph, node, fired: Revision(
            add_nodes=(Node(id="final", label="final", depends_on=["draft"]),), reason="polish"),
    )

    executor, _ = _run(graph, provider, supervisor, acceptance=AcceptancePolicy(mode="strict"))

    [entry] = executor.revision_log
    assert entry["applied"] is False and entry["trigger"] == "review_after"
    assert entry["detail"].startswith("a strict acceptance policy refuses this revision")
    assert [node.id for node in graph.nodes] == ["draft", "check"]
    assert provider.calls == ["draft", "check"]


# ---------------------------------------------------------------------------
# Durable runs
# ---------------------------------------------------------------------------


class ProcessLost(BaseException):
    pass


def test_durable_replay_of_triggered_supervision_makes_no_new_calls(tmp_path, monkeypatch):
    from smythe.workflow import revision_log

    reviews = []
    proposal = {"change": True, "reason": "the draft needs a check",
                "add": [{"id": "check", "label": "Check the draft", "depends_on": ["draft"]}]}

    async def complete(self, system, prompt, model):
        if system == SUPERVISOR_SYSTEM_PROMPT:
            reviews.append(prompt)
            return CompletionResult(json.dumps(proposal), cost_usd=0)
        return CompletionResult(f"Done: {prompt.splitlines()[0]}", cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    with SQLiteWorkflowStore(tmp_path / "triggered.db") as store:
        provider = OfflineProvider()
        swarm = Swarm(
            model="offline", provider=provider, architect=SimpleArchitect(), run_store=store,
            supervisor=_model_review(provider, ReviewTriggers(review_after={"draft"}),
                                     model="offline"),
            max_revisions=2,
        )
        original = store.save_checkpoint
        crashed = []

        def save(lease, revision, state, **kwargs):
            if crashed:
                raise ProcessLost("Process already stopped")
            if any(node["metadata"].get(REVISION_ADDED_KEY) and node["status"] == "completed"
                   for node in state["graph"]["nodes"]):
                crashed.append(lease.run_id)
                raise ProcessLost("Stopped after the added check completed")
            return original(lease, revision, state, **kwargs)

        monkeypatch.setattr(store, "save_checkpoint", save)
        with pytest.raises(ProcessLost):
            swarm.execute(ExecutionGraph([Topology.SERIAL], [
                Node(id="draft", label="Draft"),
                Node(id="final", label="Final", depends_on=["draft"]),
            ]))
        run_id = crashed[0]
        assert len(reviews) == 1
        assert "- review_after: 'draft' is a declared review point" in reviews[0]
        described = store.load_run(run_id)["config"]["components"]["supervisor"]
        assert described["type"] == "evidence_triggered_supervisor"
        assert described["triggers"] == ReviewTriggers(review_after={"draft"}).to_dict()
        assert described["inner"]["type"] == "llm_supervisor"

        monkeypatch.setattr(store, "save_checkpoint", original)
        resumed = swarm.resume(run_id)

        assert len(reviews) == 1  # The journaled review is replayed, never sent again.
        supervision = [call for call in store.inspect_run(run_id)["calls"]
                       if call["key"]["phase"] == "supervision"]
        assert len(supervision) == 1
        assert [node.id for node in resumed.graph.nodes] == ["draft", "final", "check"]
        assert all(node.status is NodeStatus.COMPLETED for node in resumed.graph.nodes)
        [entry] = revision_log(store, run_id)
        assert (entry["after_node"], entry["applied"], entry["added"], entry["trigger"]) == (
            "draft", True, ["check"], "review_after")
        assert store.audit(run_id)["ok"]


def test_durable_runs_refuse_an_action_no_recipe_can_name(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "refused.db") as store:
        provider = OfflineProvider()
        with pytest.raises(WorkflowBindingError, match="on_trigger"):
            Swarm(model="offline", provider=provider, architect=SimpleArchitect(),
                  run_store=store, max_revisions=1,
                  supervisor=EvidenceTriggeredSupervisor(None, on_trigger=_add_redo),
                  ).execute(ExecutionGraph([Topology.SERIAL], [Node(id="a", label="a")]))
        assert provider.calls == []


def test_the_supervisor_describes_and_binds_its_inner_supervisor():
    class Subclass(EvidenceTriggeredSupervisor):
        pass

    provider = OfflineProvider()
    supervisor = EvidenceTriggeredSupervisor(LLMSupervisor(provider, model="m"),
                                             ReviewTriggers(min_tool_errors=3))
    description = supervisor.workflow_description(default_model="m")
    assert description == {
        "type": "evidence_triggered_supervisor", "version": 1,
        "triggers": ReviewTriggers(min_tool_errors=3).to_dict(),
        "inner": LLMSupervisor(provider, model="m").workflow_description(),
    }
    assert supervisor.workflow_providers() == (provider,)
    with pytest.raises(WorkflowBindingError, match="LocalOnly"):
        Subclass(LLMSupervisor(provider, model="m")).workflow_description()
    local = LocalOnly(lambda: EvidenceTriggeredSupervisor(None, on_trigger=_add_redo),
                      "redo", "1", role="supervisor")
    wrapped = EvidenceTriggeredSupervisor(local)
    assert wrapped.workflow_description()["inner"] == local.workflow_description()
    assert wrapped.workflow_providers() == ()


# ---------------------------------------------------------------------------
# Checks an inner LLMSupervisor still gets
# ---------------------------------------------------------------------------


class Declares(Provider):
    def __init__(self, sdk_retries: int | None = 0) -> None:
        self.sdk_retries, self.calls = sdk_retries, []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=self.sdk_retries)

    async def complete(self, system, prompt, model):
        self.calls.append(model)
        return CompletionResult("done", prompt_tokens=1, completion_tokens=1)


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_wrapped_reviewer_whose_sdk_may_retry_is_refused_before_any_call(parallel):
    provider, reviewer = Declares(), Declares(None)
    swarm = Swarm(model="m", provider=provider, retry_policy=RetryPolicy(), artifact_dir=None,
                  parallel=parallel, supervisor=_model_review(reviewer), max_revisions=1)

    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="a", label="a")]))
    executor = Executor(provider, Registry(), Tracer(), artifact_dir=None,
                        retry_policy=RetryPolicy(), supervisor=_model_review(reviewer),
                        max_revisions=1)
    with pytest.raises(ValueError, match="retry_policy must be the only retry owner"):
        executor.run(_graph(Node(id="a", label="a")))
    assert provider.calls == [] and reviewer.calls == []


def test_a_wrapped_model_less_reviewer_with_another_providers_profile_is_refused():
    default, fast = Declares(), Declares()
    swarm = Swarm(model="m", provider=default, artifact_dir=None, max_revisions=1,
                  profiles=[ModelProfile("fast", fast, "fast-model", 0.25)],
                  supervisor=_model_review(default))

    with pytest.raises(NoAllowedProfileError, match=r"LLMSupervisor\(provider, model="):
        swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="a", label="a")]))
    assert default.calls == [] and fast.calls == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_triggers_round_trip_through_their_recipe_data():
    triggers = ReviewTriggers(review_after=["b", "a"], fan_in=True, cost_over_estimate=2,
                              min_tool_errors=1, contradiction_records=False)
    data = triggers.to_dict()
    assert data["review_after"] == ["a", "b"] and data["cost_over_estimate"] == 2.0
    assert json.loads(json.dumps(data)) == data
    assert ReviewTriggers.from_dict(data) == triggers
    with pytest.raises(ValueError, match="exactly the keys"):
        ReviewTriggers.from_dict({**data, "surprise": True})
    with pytest.raises(ValueError, match="exactly the keys"):
        ReviewTriggers.from_dict({key: value for key, value in data.items() if key != "final"})


@pytest.mark.parametrize("settings, message", [
    ({"review_after": "draft"}, "review_after"),
    ({"review_after": {1}}, "review_after"),
    ({"fan_in": 1}, "fan_in"),
    ({"failed_or_skipped": None}, "failed_or_skipped"),
    ({"min_output_chars": 0}, "min_output_chars"),
    ({"min_output_chars": True}, "min_output_chars"),
    ({"min_tool_errors": 1.5}, "min_tool_errors"),
    ({"cost_over_estimate": 0}, "cost_over_estimate"),
    ({"cost_over_estimate": float("inf")}, "cost_over_estimate"),
    ({"cost_over_estimate": True}, "cost_over_estimate"),
])
def test_invalid_triggers_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        ReviewTriggers(**settings)


def test_the_supervisor_needs_something_to_do_when_a_trigger_fires():
    with pytest.raises(ValueError, match="inner supervisor or on_trigger"):
        EvidenceTriggeredSupervisor(None)
    with pytest.raises(TypeError, match="triggers"):
        EvidenceTriggeredSupervisor(None, {"final": True}, on_trigger=_add_redo)
    with pytest.raises(TypeError, match="inner"):
        EvidenceTriggeredSupervisor(object())
    with pytest.raises(TypeError, match="on_trigger"):
        EvidenceTriggeredSupervisor(None, on_trigger="redo")
    assert EvidenceTriggeredSupervisor(None, on_trigger=_add_redo).triggers == ReviewTriggers()
    assert FiredTrigger("final", ("x",)).evidence == ("x",)
