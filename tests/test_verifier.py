"""Tests for verification that gates — a failed check sends work back."""

from __future__ import annotations

import asyncio

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer
from smythe.verifier import CallableVerifier, TokenVerifier, Verdict


class ScriptedProvider(Provider):
    """Returns queued outputs per node id, tracking how often each ran."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self.runs: dict[str, int] = {}

    async def complete(self, system, prompt, model):
        node_id = prompt.splitlines()[0].strip()
        self.runs[node_id] = self.runs.get(node_id, 0) + 1
        queue = self._script.get(node_id) or ["ok"]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return CompletionResult(text=text, prompt_tokens=1, completion_tokens=1)


def _gated_graph(max_regenerations: int = 2) -> ExecutionGraph:
    draft = Node(id="draft", label="draft")
    judge = Node(
        id="judge", label="judge", depends_on=["draft"],
        verifies="draft", max_regenerations=max_regenerations,
    )
    for node in (draft, judge):
        node.metadata["model"] = "test-model"
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[draft, judge])


def _run(graph, provider, *, tracer=None, verifier=None, sync=False):
    kwargs = dict(
        provider=provider, registry=Registry(), tracer=tracer or Tracer(),
        artifact_dir=None, verifier=verifier,
    )
    if sync:
        return Executor(**kwargs).run(graph)
    return asyncio.run(AsyncExecutor(**kwargs).run(graph))


# ---------------------------------------------------------------------------
# TokenVerifier
# ---------------------------------------------------------------------------


def test_token_verifier_reads_json():
    node = Node(id="j", label="j")
    node.result = '{"passed": false, "reason": "missing citations"}'
    verdict = TokenVerifier().verdict(node, Node(id="t", label="t"))
    assert verdict.passed is False
    assert verdict.reason == "missing citations"


def test_token_verifier_reads_prose():
    node = Node(id="j", label="j")
    node.result = "FAIL - the draft contradicts the source data."
    assert TokenVerifier().verdict(node, Node(id="t", label="t")).passed is False

    node.result = "PASS. Everything checks out."
    assert TokenVerifier().verdict(node, Node(id="t", label="t")).passed is True


def test_token_verifier_strips_code_fences():
    node = Node(id="j", label="j")
    node.result = '```json\n{"passed": false, "reason": "typo"}\n```'
    assert TokenVerifier().verdict(node, Node(id="t", label="t")).passed is False


def _verdict(text):
    node = Node(id="j", label="j")
    node.result = text
    return TokenVerifier().verdict(node, Node(id="t", label="t"))


@pytest.mark.parametrize("text", [
    "", None, "   ", "The weather is nice today.", "Looks good to me, approved.", "ok",
    '{"reason": "no verdict field"}',
])
def test_unreadable_verdict_fails_closed(text):
    """A gate must not wave work through because its judge was unreadable."""
    verdict = _verdict(text)
    assert verdict.passed is False
    assert verdict.reason


@pytest.mark.parametrize("value", ['"false"', '"true"', "1", "0", "null", '"PASS"', "[]"])
def test_json_passed_must_be_a_boolean(value):
    verdict = _verdict('{"passed": %s, "reason": "r"}' % value)
    assert verdict.passed is False
    assert "JSON boolean" in verdict.reason


def test_json_true_passes_with_its_reason():
    verdict = _verdict('{"passed": true, "reason": "every claim is cited"}')
    assert verdict.passed is True
    assert verdict.reason == "every claim is cited"


def test_fenced_json_after_prose_is_read():
    assert _verdict('My verdict:\n```json\n{"passed": true}\n```').passed is True
    assert _verdict('My verdict:\n```\n{"passed": false}\n```').passed is False


def test_first_keyword_wins():
    node = Node(id="j", label="j")
    node.result = "FAILED: this does not pass muster"
    assert TokenVerifier().verdict(node, Node(id="t", label="t")).passed is False


@pytest.mark.parametrize("text", [
    "The draft does not fail any criterion. PASS",
    "No criterion failed.\n\nVerdict: **PASS**",
    "PASSED - all three criteria hold",
    "pass",
    "Pass.",
    "**PASSED**",
])
def test_lowercase_prose_is_not_a_verdict(text):
    assert _verdict(text).passed is True


@pytest.mark.parametrize("text", ["fail", "Failed.", "Verdict: FAIL - two claims lack sources"])
def test_explicit_failures_fail(text):
    assert _verdict(text).passed is False


@pytest.mark.parametrize("text", [
    "Criterion 1: PASS\nCriterion 2: FAIL",
    "PASS/FAIL",
    "I would say PASS, but on reflection FAIL.",
])
def test_conflicting_verdict_words_fail_closed(text):
    verdict = _verdict(text)
    assert verdict.passed is False
    assert "ambiguous" in verdict.reason


def test_pathological_json_fails_closed_instead_of_raising():
    assert _verdict("[" * 100_000 + "]" * 100_000).passed is False
    assert _verdict('{"passed": ' + "1" * 5000 + "}").passed is False


def test_unreadable_verdicts_regenerate_only_up_to_the_limit():
    """Failing closed stays bounded: the run finishes with the last output."""
    graph = _gated_graph(max_regenerations=2)
    provider = ScriptedProvider({"draft": ["v1", "v2", "v3"], "judge": ["Looks fine to me."]})
    _run(graph, provider)

    assert provider.runs["draft"] == 3
    assert graph.nodes[1].metadata["regenerations_used"] == 2
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)
    assert graph.nodes[0].result == "v3"


def test_advisory_gate_never_reads_an_unreadable_verdict():
    graph = _gated_graph(max_regenerations=0)
    provider = ScriptedProvider({"draft": ["v1"], "judge": ["Looks fine to me."]})
    _run(graph, provider)
    assert provider.runs == {"draft": 1, "judge": 1}


def test_callable_verifier_accepts_bool_or_verdict():
    node, target = Node(id="j", label="j"), Node(id="t", label="t")
    assert CallableVerifier(lambda v, t: True).verdict(node, target).passed is True
    assert CallableVerifier(lambda v, t: False).verdict(node, target).passed is False
    verdict = CallableVerifier(
        lambda v, t: Verdict(passed=False, reason="too short"),
    ).verdict(node, target)
    assert verdict.reason == "too short"


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def test_failed_verification_regenerates_the_target():
    graph = _gated_graph()
    provider = ScriptedProvider({
        "draft": ["first attempt", "second attempt"],
        "judge": ["FAIL: not good enough", "PASS"],
    })
    _run(graph, provider)

    assert provider.runs["draft"] == 2, "the draft should have been redone once"
    assert provider.runs["judge"] == 2
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)
    assert graph.nodes[0].result == "second attempt"


def test_passing_verification_does_not_regenerate():
    graph = _gated_graph()
    provider = ScriptedProvider({"draft": ["fine"], "judge": ["PASS"]})
    _run(graph, provider)
    assert provider.runs == {"draft": 1, "judge": 1}


def test_regeneration_is_bounded():
    graph = _gated_graph(max_regenerations=2)
    provider = ScriptedProvider({"draft": ["bad"], "judge": ["FAIL: still wrong"]})
    _run(graph, provider)

    # One original run plus exactly two regenerations.
    assert provider.runs["draft"] == 3
    assert graph.nodes[1].metadata["regenerations_used"] == 2
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)


def test_zero_regenerations_makes_the_verdict_advisory():
    graph = _gated_graph(max_regenerations=0)
    provider = ScriptedProvider({"draft": ["bad"], "judge": ["FAIL"]})
    _run(graph, provider)
    assert provider.runs == {"draft": 1, "judge": 1}


def test_downstream_work_is_redone_too():
    """A failed draft invalidates whatever was written from it."""
    draft = Node(id="draft", label="draft")
    judge = Node(
        id="judge", label="judge", depends_on=["draft"],
        verifies="draft", max_regenerations=1,
    )
    summary = Node(id="summary", label="summary", depends_on=["draft"])
    for node in (draft, judge, summary):
        node.metadata["model"] = "test-model"
    graph = ExecutionGraph(
        topology=[Topology.SERIAL], nodes=[draft, judge, summary],
    )
    provider = ScriptedProvider({
        "draft": ["v1", "v2"], "judge": ["FAIL: redo", "PASS"], "summary": ["s"],
    })
    _run(graph, provider)

    assert provider.runs["summary"] == 2, "summary was written from a rejected draft"
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)


def test_regeneration_is_traced():
    graph = _gated_graph()
    tracer = Tracer()
    provider = ScriptedProvider({
        "draft": ["bad", "good"], "judge": ["FAIL: missing sources", "PASS"],
    })
    _run(graph, provider, tracer=tracer)

    events = [s for s in tracer.summary() if "regeneration" in s]
    assert len(events) == 1
    assert events[0]["regeneration"]["target"] == "draft"
    assert events[0]["regeneration"]["attempt"] == 1
    assert events[0]["regeneration"]["limit"] == 2
    assert "missing sources" in events[0]["label"]


def test_custom_verifier_gates_objectively():
    """Verification needs no model — any rule can gate."""
    graph = _gated_graph(max_regenerations=1)
    provider = ScriptedProvider({
        "draft": ["short", "a much longer and more complete draft"],
        "judge": ["anything"],
    })
    verifier = CallableVerifier(lambda v, target: len(str(target.result)) > 20)
    _run(graph, provider, verifier=verifier)

    assert provider.runs["draft"] == 2
    assert graph.nodes[0].result.startswith("a much longer")


def test_sync_executor_regenerates():
    graph = _gated_graph()
    provider = ScriptedProvider({
        "draft": ["bad", "good"], "judge": ["FAIL: redo", "PASS"],
    })
    _run(graph, provider, sync=True)
    assert provider.runs["draft"] == 2
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)


def test_verifier_fields_survive_checkpoint():
    from smythe.checkpoint import graph_from_dict, graph_to_dict

    graph = _gated_graph(max_regenerations=3)
    restored = graph_from_dict(graph_to_dict(graph))
    assert restored.nodes[1].verifies == "draft"
    assert restored.nodes[1].max_regenerations == 3


def test_unknown_verify_target_is_ignored():
    node = Node(id="judge", label="judge", verifies="ghost", max_regenerations=1)
    node.metadata["model"] = "test-model"
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    provider = ScriptedProvider({"judge": ["FAIL"]})
    _run(graph, provider)
    assert node.status is NodeStatus.COMPLETED


@pytest.mark.parametrize("sync", [False, True])
def test_budget_is_charged_for_each_regeneration(sync):
    from smythe.budget import Sentinel

    graph = _gated_graph(max_regenerations=1)
    provider = ScriptedProvider({
        "draft": ["bad", "good"], "judge": ["FAIL: redo", "PASS"],
    })
    budget = Sentinel(max_budget_usd=10.0)
    kwargs = dict(
        provider=provider, registry=Registry(), tracer=Tracer(),
        artifact_dir=None, budget=budget,
    )
    if sync:
        Executor(**kwargs).run(graph)
    else:
        asyncio.run(AsyncExecutor(**kwargs).run(graph))

    # 2 draft runs + 2 judge runs, all billed.
    assert budget.total_cost_usd > 0
    assert provider.runs["draft"] == 2
