"""Tests for verification that gates — a failed check sends work back."""

from __future__ import annotations

import asyncio
import time

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


_FORMAT_ECHO = (
    'Expected format:\n```json\n{"passed": true, "reason": "..."}\n```\n'
    'My verdict:\n```json\n{"passed": false, "reason": "uncited claims"}\n```'
)

# Every reply from the 0.8.1 verdict-parser review, plus clean replies whose
# meaning must not change.  A false PASS lets a defective deliverable
# through, so each ambiguous reply must FAIL.
VERDICTS = [
    # An uppercase PASS that is negated, conditional or quoted is not a verdict.
    pytest.param("The memo failed criterion 2 (no sources), so it cannot PASS.", False,
                 id="cannot-pass"),
    pytest.param("Rejected: the draft does not PASS.", False, id="rejected-does-not-pass"),
    pytest.param("REJECTED - the draft does not PASS.", False, id="REJECTED-does-not-pass"),
    pytest.param("It FAILS to PASS criterion 2.", False, id="fails-to-pass"),
    pytest.param("NOT PASS", False, id="not-pass"),
    pytest.param("Does it PASS? No.", False, id="question"),
    pytest.param("The memo would PASS if the citations were added.", False, id="would-pass-if"),
    pytest.param('The draft claims "all tests PASS", which is false.', False, id="quoted-pass"),
    pytest.param("Criterion 1: PASS\nCriterion 2: PASS\nCriterion 3: not met (no sources)\n"
                 "Overall: not approved", False, id="criteria-without-final-verdict"),
    pytest.param("The draft does not PASS.", False, id="does-not-pass"),
    pytest.param("fail - does not PASS", False, id="lowercase-fail-does-not-pass"),
    # JSON cannot be overridden by prose, and its strings are never prose.
    pytest.param('{"passed": false, "reason": "criterion 2 is not a PASS"}.', False,
                 id="json-false-then-period"),
    pytest.param('Verdict: {"passed": false, "reason": "no PASS on criterion 2"}', False,
                 id="labelled-json-false"),
    pytest.param('{"passed": false, "reason": "no sources"}\nOverall it would PASS with citations.',
                 False, id="json-false-then-prose"),
    pytest.param('FAIL\n```json\n{"passed": true}\n```', False, id="prose-fail-json-true"),
    pytest.param(_FORMAT_ECHO, False, id="echoed-format-example"),
    pytest.param('```json\n{"passed": true}\n```\nOn reflection, criterion 3 is unmet: FAIL', False,
                 id="json-true-then-prose-fail"),
    pytest.param('PASS\n```json\n{"passed": false}\n```', False, id="prose-pass-json-false"),
    pytest.param('{"passed": false, "passed": true}', False, id="repeated-passed-key"),
    pytest.param('{"passed": false, "reason": "the format is {"passed": true}"}', False,
                 id="malformed-json-around-a-verdict"),
    pytest.param('Verdict: PASS\n{"passed": tru', False, id="truncated-json"),
    pytest.param('{"passed": true, "criteria": [{"name": "sources", "passed": false}]}', False,
                 id="nested-false"),
    pytest.param('{"verdict": {"passed": true}}', False, id="nested-true-is-not-a-verdict"),
    pytest.param('[{"passed": true}]', False, id="array-is-not-a-verdict"),
    pytest.param('{"result": "PASS"}', False, id="json-string-is-not-prose"),
    # A PASS must stand alone; other positions and contradictions fail.
    pytest.param("PASS\n\nOn reflection, the memo does not PASS criterion 3.", False,
                 id="pass-then-negated-pass"),
    pytest.param("Verdict: PASS - provided the citations are added", False, id="conditional-pass"),
    pytest.param("PASS, if the citations are added.", False, id="pass-comma-if"),
    pytest.param("Criterion 1 - sources\nVerdict: PASS\nCriterion 2 - length\n"
                 "Verdict: not met (950 words)", False, id="unreadable-verdict-line"),
    pytest.param("The memo FAILED criterion 2.\n\nVerdict: PASS", False, id="fail-word-then-pass"),
    pytest.param("```\nFAILED test_sources\n```\nPASS", False, id="fail-word-in-code-block"),
    pytest.param("Result: PASS\nResult: not met", False, id="result-is-not-a-verdict-label"),
    pytest.param("Everything checks out: PASS", False, id="arbitrary-label"),
    pytest.param("- PASS\n- PASS\n- missing sources", False, id="bulleted-pass"),
    pytest.param("1. PASS\n2. PASS\n3. missing sources", False, id="numbered-pass"),
    pytest.param("Test log:\n```\nPASS\n```\nThe draft omits the failing case.", False,
                 id="pass-in-code-block"),
    pytest.param("> PASS\n\nThat status line in the draft is wrong.", False,
                 id="pass-in-blockquote"),
    pytest.param("PASS-THROUGH mode is undocumented, so criterion 2 is unmet.", False,
                 id="hyphenated-word"),
    pytest.param("See PASS.md; criterion 2 is unmet.", False, id="file-name"),
    pytest.param("PASS?", False, id="pass-question"),
    # Avoidable false FAILs from the review now pass.
    pytest.param("PASS/FAIL: PASS", True, id="choice-label"),
    pytest.param("Verdict (PASS or FAIL): PASS", True, id="choice-in-label"),
    pytest.param("Verdict: PASSES", True, id="passes"),
    pytest.param('```JSON\n{"passed": true}\n```', True, id="uppercase-fence-tag"),
    pytest.param('{"passed": true}\nThe draft is fine.', True, id="json-then-prose"),
    pytest.param('Here is my verdict: {"passed": true}', True, id="prose-then-json"),
    pytest.param('The rubric says "answer PASS or FAIL". PASS.', True, id="rubric-echo"),
    pytest.param("FAIL or PASS? PASS.", True, id="reversed-choice"),
    pytest.param("PASS-FAIL grading applies.\nVerdict: PASS", True, id="hyphenated-choice"),
    # Clean replies keep their meaning.
    pytest.param("PASS", True, id="PASS"),
    pytest.param("FAIL", False, id="FAIL"),
    pytest.param("Pass.", True, id="bare-any-case"),
    pytest.param("PASS. Everything checks out.", True, id="leading-pass"),
    pytest.param("FAIL - the draft contradicts the source data.", False, id="leading-fail"),
    pytest.param("PASS — all criteria met", True, id="em-dash"),
    pytest.param("PASS (3/3 criteria met)", True, id="parenthetical"),
    pytest.param("The draft does not fail any criterion. PASS", True, id="trailing-pass"),
    pytest.param("Is the memo ready? PASS", True, id="after-question"),
    pytest.param("Overall: PASS", True, id="overall"),
    pytest.param("Final verdict - PASS", True, id="final-verdict"),
    pytest.param("**Verdict:** PASS", True, id="bold-label"),
    pytest.param("## Verdict: FAIL", False, id="heading-fail"),
    pytest.param("Verdict:\nPASS", True, id="label-on-own-line"),
    pytest.param("My final verdict is PASS", True, id="verdict-is"),
    pytest.param("| Verdict | PASS |", True, id="table-row"),
    pytest.param("Criterion 1: PASS\nCriterion 2: PASS\n\nOverall: PASS", True,
                 id="criteria-with-final-verdict"),
    pytest.param("Criterion 1: PASS\nCriterion 2: FAIL\n\nOverall: PASS", False,
                 id="criteria-contradict-final-verdict"),
    pytest.param('{"passed": true, "reason": "every claim is cited"}', True, id="json-true"),
    pytest.param('{"passed": false}', False, id="json-false"),
    pytest.param('My verdict:\n```json\n{"passed": true}\n```', True, id="fenced-json"),
    pytest.param('Verdict: PASS\n{"passed": true, "reason": "fine"}', True,
                 id="json-agrees-with-prose"),
]


@pytest.mark.parametrize(("text", "passed"), VERDICTS)
def test_verdict_table(text, passed):
    assert _verdict(text).passed is passed


@pytest.mark.parametrize(("text", "reason"), [
    ('```JSON\n{"passed": false, "reason": "typo"}\n```', "typo"),
    ('Verdict: {"passed": false, "reason": "no PASS on criterion 2"}', "no PASS on criterion 2"),
    ('{"passed": false, "reason": "no sources"}\nOverall it would PASS with citations.',
     "no sources"),
    ('Here is my verdict: {"passed": true, "reason": "cited"}', "cited"),
])
def test_json_verdict_is_read_through_surrounding_text(text, reason):
    assert _verdict(text).reason == reason


@pytest.mark.parametrize(("text", "expected"), [
    ('FAIL\n```json\n{"passed": true}\n```', "ambiguous verdict"),
    (_FORMAT_ECHO, "ambiguous verdict"),
    ('{"passed": false, "passed": true}', "repeats 'passed'"),
    ('Verdict: PASS\n{"passed": tru', "not part of valid JSON"),
    ("It can't PASS.", "negated or conditional PASS"),
    ("Verdict: not a pass", "unreadable verdict"),
])
def test_rejected_verdicts_say_why(text, expected):
    verdict = _verdict(text)
    assert verdict.passed is False
    assert expected in verdict.reason


def test_unexamined_json_verdict_fails_closed():
    """At most 64 JSON candidates are decoded; a verdict beyond them fails the reply."""
    verdict = _verdict('{"note": 1} ' * 64 + '{"passed": true}')
    assert verdict.passed is False
    assert "not part of valid JSON" in verdict.reason
    assert _verdict('{"note": 1} ' * 63 + '{"passed": true}').passed is True


def test_negated_pass_sends_the_draft_back():
    graph = _gated_graph(max_regenerations=1)
    provider = ScriptedProvider({
        "draft": ["v1", "v2"], "judge": ["Rejected: the draft does not PASS.", "PASS"],
    })
    _run(graph, provider)
    assert provider.runs["draft"] == 2
    assert graph.nodes[0].result == "v2"


def _timed_verdict(text):
    start = time.perf_counter()
    verdict = _verdict(text)
    return verdict, time.perf_counter() - start


@pytest.mark.parametrize(("text", "passed"), [
    pytest.param("```" + " " * 2000 + "x", False, id="open-fence-then-spaces"),
    pytest.param("PASS\n```\n" + "\n" * 2000 + "end", True, id="open-fence-then-newlines"),
])
def test_unclosed_code_fence_is_read_quickly(text, passed):
    """A cubic fence regex took 4 to 16 seconds on these 2 KB replies."""
    verdict, seconds = _timed_verdict(text)
    assert verdict.passed is passed
    assert seconds < 0.5


@pytest.mark.parametrize("text", [
    pytest.param("```" + " " * 190_000 + "x", id="open-fence-then-spaces"),
    pytest.param("PASS\n```\n" + "\n" * 190_000 + "end", id="open-fence-then-newlines"),
    pytest.param("{" * 190_000, id="open-braces"),
    pytest.param('{"a":' * 38_000, id="nested-objects"),
    pytest.param('[{"a": 1}, ' * 17_000, id="nested-arrays"),
    pytest.param('"passed": ' * 19_000, id="stray-keys"),
    pytest.param("PASS " * 10_000, id="verdict-words"),
    pytest.param("Verdict: x\n" * 5_000, id="verdict-labels"),
])
def test_large_adversarial_reply_is_read_in_linear_time(text):
    """CI runs this under coverage, so the bound only rules out super-linear time."""
    verdict, seconds = _timed_verdict(text)
    assert isinstance(verdict.passed, bool)
    assert seconds < 5


def test_overlong_reply_fails_closed():
    verdict = _verdict("PASS\n" + "x" * 200_001)
    assert verdict.passed is False
    assert "too long" in verdict.reason


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
