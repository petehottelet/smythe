"""Tests for adaptive supervision — revising a plan while it runs."""

from __future__ import annotations

import asyncio
import json

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.executor import Executor
from smythe.graph import (
    REVISION_ADDED_KEY,
    SYNTHESIS_NODE_ID,
    ExecutionGraph,
    Node,
    NodeStatus,
    Revision,
    RevisionError,
    Topology,
)
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT, LLMSupervisor, Supervisor
from smythe.tracer import Tracer


class EchoProvider(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text="done", prompt_tokens=1, completion_tokens=1)


class ScriptedSupervisor(Supervisor):
    """Returns queued revisions, one per review, then stops."""

    def __init__(self, *revisions, raises: Exception | None = None) -> None:
        self._queue = list(revisions)
        self._raises = raises
        self.reviews: list[str] = []

    async def review(self, graph, node, *, task, revisions_remaining):
        self.reviews.append(node.id)
        if self._raises is not None:
            raise self._raises
        return self._queue.pop(0) if self._queue else None


def _graph(*labels: str) -> ExecutionGraph:
    nodes = []
    prev = None
    for i, label in enumerate(labels):
        node = Node(id=f"n{i}", label=label, depends_on=[prev] if prev else [])
        node.metadata["model"] = "test-model"
        nodes.append(node)
        prev = node.id
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)


def _executor(supervisor=None, *, max_revisions=0, tracer=None):
    return AsyncExecutor(
        provider=EchoProvider(),
        registry=Registry(),
        tracer=tracer or Tracer(),
        artifact_dir=None,
        supervisor=supervisor,
        max_revisions=max_revisions,
    )


# ---------------------------------------------------------------------------
# ExecutionGraph.apply_revision — validation
# ---------------------------------------------------------------------------


def test_add_node_extends_the_graph():
    graph = _graph("first")
    extra = Node(id="extra", label="added", depends_on=["n0"])
    graph.apply_revision(Revision(add_nodes=(extra,), reason="gap"))
    assert [n.id for n in graph.nodes] == ["n0", "extra"]


def test_drop_removes_pending_node():
    graph = _graph("first", "second")
    graph.apply_revision(Revision(drop_node_ids=("n1",)))
    assert [n.id for n in graph.nodes] == ["n0"]


def test_rewire_redirects_a_pending_node():
    graph = _graph("first", "second")
    inserted = Node(id="mid", label="inserted", depends_on=["n0"])
    graph.apply_revision(
        Revision(add_nodes=(inserted,), rewire={"n1": ("mid",)}),
    )
    assert graph.nodes[-1].id == "mid"
    assert next(n for n in graph.nodes if n.id == "n1").depends_on == ["mid"]


def test_cannot_drop_completed_work():
    graph = _graph("first", "second")
    graph.nodes[0].status = NodeStatus.COMPLETED
    with pytest.raises(RevisionError, match="only pending work"):
        graph.apply_revision(Revision(drop_node_ids=("n0",)))


def test_cannot_drop_running_work():
    graph = _graph("first", "second")
    graph.nodes[1].status = NodeStatus.RUNNING
    with pytest.raises(RevisionError, match="only pending work"):
        graph.apply_revision(Revision(drop_node_ids=("n1",)))


def test_cannot_drop_unknown_node():
    graph = _graph("first")
    with pytest.raises(RevisionError, match="unknown node"):
        graph.apply_revision(Revision(drop_node_ids=("ghost",)))


def test_cannot_add_duplicate_id():
    graph = _graph("first")
    with pytest.raises(RevisionError, match="already exists"):
        graph.apply_revision(Revision(add_nodes=(Node(id="n0", label="clash"),)))


def test_cannot_orphan_a_dependent():
    graph = _graph("first", "second")
    with pytest.raises(RevisionError, match="depend on missing node"):
        graph.apply_revision(Revision(drop_node_ids=("n0",)))


def test_cannot_introduce_a_cycle():
    graph = _graph("first", "second")
    with pytest.raises(RevisionError, match="cycle"):
        graph.apply_revision(Revision(rewire={"n0": ("n1",)}))


def test_cannot_rewire_finished_node():
    """A completed node's inputs are already consumed; re-pointing them
    would silently invalidate a result the run has already banked."""
    graph = _graph("first", "second")
    graph.nodes[1].status = NodeStatus.COMPLETED
    extra = Node(id="extra", label="added")
    with pytest.raises(RevisionError, match="only pending work"):
        graph.apply_revision(
            Revision(add_nodes=(extra,), rewire={"n1": ("n0", "extra")}),
        )


def test_rejected_revision_leaves_graph_untouched():
    graph = _graph("first", "second")
    before = [(n.id, list(n.depends_on)) for n in graph.nodes]
    bad = Node(id="bad", label="added", depends_on=["nonexistent"])
    with pytest.raises(RevisionError):
        graph.apply_revision(Revision(add_nodes=(bad,), drop_node_ids=("n1",)))
    assert [(n.id, list(n.depends_on)) for n in graph.nodes] == before


def test_empty_revision_is_a_noop():
    graph = _graph("first")
    graph.apply_revision(Revision())
    assert len(graph.nodes) == 1


def test_deep_revision_applies_add_drop_and_rewire_after_validation():
    graph = _graph(*["step"] * 5_000)
    nodes = list(graph.nodes)
    graph.nodes.reverse()
    nodes[0].status = NodeStatus.COMPLETED
    nodes[0].result = "saved result"
    inserted = Node(id="inserted", label="Inserted", depends_on=["n2499"])
    tail = Node(id="tail", label="New tail", depends_on=["n4998"])

    graph.apply_revision(Revision(
        add_nodes=(inserted, tail), drop_node_ids=("n4999",),
        rewire={"n2500": (inserted.id,)},
    ))

    assert [id(node) for node in graph.nodes] == [
        *(id(node) for node in reversed(nodes[:-1])), id(inserted), id(tail),
    ]
    assert nodes[2500].depends_on == [inserted.id]
    assert nodes[0].status is NodeStatus.COMPLETED
    assert nodes[0].result == "saved result"
    assert graph.depth == 5_000
    graph.validate()


@pytest.mark.parametrize("invalid_dependency", ["n4998", "missing"])
def test_rejected_deep_revision_preserves_graph_and_proposed_nodes(invalid_dependency):
    graph = _graph(*["step"] * 5_000)
    graph.nodes.reverse()
    original_list = graph.nodes
    graph.nodes[-1].result = {"retained": ["evidence"]}
    before = [
        (id(node), id(node.depends_on), tuple(node.depends_on), node.status, id(node.result))
        for node in graph.nodes
    ]
    extra = Node(id="extra", label="Proposed", depends_on=["n4998"])
    extra_dependencies = extra.depends_on

    with pytest.raises(RevisionError, match="cycle|missing node"):
        graph.apply_revision(Revision(
            add_nodes=(extra,), drop_node_ids=("n4999",),
            rewire={"n0": (invalid_dependency,)},
        ))

    assert graph.nodes is original_list
    assert [
        (id(node), id(node.depends_on), tuple(node.depends_on), node.status, id(node.result))
        for node in graph.nodes
    ] == before
    assert graph.nodes[-1].result == {"retained": ["evidence"]}
    assert extra.depends_on is extra_dependencies
    assert extra.depends_on == ["n4998"]
    assert extra.status is NodeStatus.PENDING


def test_added_nodes_are_marked_only_when_the_revision_applies():
    graph = _graph("first")
    added = Node(id="extra", label="added", depends_on=["n0"])
    graph.apply_revision(Revision(add_nodes=(added,)))
    assert added.metadata[REVISION_ADDED_KEY] is True
    assert REVISION_ADDED_KEY not in graph.nodes[0].metadata

    refused = Node(id="refused", label="added", depends_on=["missing"])
    with pytest.raises(RevisionError):
        graph.apply_revision(Revision(add_nodes=(refused,)))
    assert refused.metadata == {}


def test_cannot_add_the_reserved_synthesis_id():
    """The synthesizer books its charge under this id; a node sharing it
    would merge its cost with synthesis in the budget breakdown."""
    graph = _graph("first")
    with pytest.raises(RevisionError, match="reserved"):
        graph.apply_revision(Revision(add_nodes=(Node(id=SYNTHESIS_NODE_ID, label="x"),)))
    assert [n.id for n in graph.nodes] == ["n0"]


# ---------------------------------------------------------------------------
# Verification gates cannot be revised away
# ---------------------------------------------------------------------------


def _gated_graph() -> ExecutionGraph:
    nodes = [
        Node(id="research", label="research"),
        Node(id="draft", label="draft", depends_on=["research"]),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=2),
    ]
    for node in nodes:
        node.metadata["model"] = "test-model"
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)


def _structure(graph):
    return [(n.id, list(n.depends_on)) for n in graph.nodes]


@pytest.mark.parametrize("revision, message", [
    (Revision(drop_node_ids=("judge",)), "cannot drop verifier 'judge'"),
    (Revision(drop_node_ids=("draft",)), "verifier 'judge' judges it"),
    (Revision(drop_node_ids=("draft",), rewire={"judge": ("research",)}),
     "verifier 'judge' judges it"),
    (Revision(rewire={"judge": ("research",)}), "cannot rewire verifier 'judge' off 'draft'"),
    (Revision(add_nodes=(Node(id="summary", label="s", depends_on=["draft"]),),
              rewire={"judge": ("summary",)}),
     "cannot rewire verifier 'judge' off 'draft'"),
])
def test_revision_cannot_remove_or_bypass_a_gate(revision, message):
    """A verdict is only read when the judge runs after, and sees, its target."""
    graph = _gated_graph()
    before = _structure(graph)
    with pytest.raises(RevisionError, match=message):
        graph.apply_revision(revision)
    assert _structure(graph) == before


def test_verifier_target_is_protected_even_without_a_dependency_edge():
    graph = _gated_graph()
    graph.nodes[2].depends_on = ["research"]  # a hand-built gate may omit the edge
    with pytest.raises(RevisionError, match="verifier 'judge' judges it"):
        graph.apply_revision(Revision(drop_node_ids=("draft",)))


def test_work_around_a_gate_can_still_be_revised():
    graph = _gated_graph()
    graph.nodes[0].status = NodeStatus.COMPLETED
    graph.apply_revision(Revision(
        add_nodes=(Node(id="facts", label="Check facts", depends_on=["research"]),
                   Node(id="publish", label="Publish", depends_on=["judge"])),
        rewire={"judge": ("draft", "facts"), "draft": ("research", "facts")},
    ))
    judge = graph.nodes[2]
    assert judge.depends_on == ["draft", "facts"] and judge.verifies == "draft"
    graph.apply_revision(Revision(drop_node_ids=("publish",)))
    assert [n.id for n in graph.nodes] == ["research", "draft", "judge", "facts"]


class GatedRunProvider(Provider):
    """The supervisor proposes a fixed change; the judge fails every draft."""

    def __init__(self, proposal: dict, *, slow_draft: bool = False) -> None:
        self._proposal = json.dumps(proposal)
        self._slow_draft = slow_draft
        self.calls: list[str] = []

    async def complete(self, system, prompt, model):
        if system == SUPERVISOR_SYSTEM_PROMPT:
            return CompletionResult(text=self._proposal, prompt_tokens=1, completion_tokens=1)
        label = prompt.splitlines()[0]
        self.calls.append(label)
        if label == "draft" and self._slow_draft:
            # Lets a judge rewired onto "research" finish before the draft.
            await asyncio.sleep(0.2)
        text = "FAIL - missing citations" if label == "judge" else "body"
        return CompletionResult(text=text, prompt_tokens=1, completion_tokens=1)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("proposal, refusal", [
    ({"change": True, "reason": "verification is redundant", "drop": ["judge"]},
     "cannot drop verifier 'judge'"),
    ({"change": True, "reason": "check sooner", "rewire": {"judge": ["research"]}},
     "cannot rewire verifier 'judge' off 'draft'"),
])
def test_supervisor_proposal_cannot_remove_or_bypass_a_gate(parallel, proposal, refusal):
    """The review prompt carries worker output, so source data can steer it.

    Dropping the judge used to accept the draft unverified; rewiring it
    onto "research" let a parallel judge finish first and its FAIL was
    discarded. The gate must regenerate exactly as it does unsupervised.
    """
    provider = GatedRunProvider(proposal, slow_draft=parallel)
    graph = _gated_graph()
    tracer = Tracer()
    options = dict(
        provider=provider, registry=Registry(), tracer=tracer, artifact_dir=None,
        supervisor=LLMSupervisor(provider, review_after={"research"}), max_revisions=1,
    )
    if parallel:
        executor = AsyncExecutor(max_concurrency=4, **options)
        asyncio.run(executor.run(graph))
    else:
        executor = Executor(**options)
        executor.run(graph)

    judge = graph.nodes[2]
    assert _structure(graph) == _structure(_gated_graph())
    assert provider.calls == ["research", "draft", "judge"] + ["draft", "judge"] * 2
    assert judge.metadata["regenerations_used"] == 2
    assert executor.revisions_used == 0
    rejected = [s for s in tracer.summary() if s.get("status") == "revision_rejected"]
    assert len(rejected) == 1 and refusal in rejected[0]["error"]


# ---------------------------------------------------------------------------
# Executor integration
# ---------------------------------------------------------------------------


def test_added_node_actually_executes():
    graph = _graph("first")
    extra = Node(id="extra", label="the missing step", depends_on=["n0"])
    supervisor = ScriptedSupervisor(Revision(add_nodes=(extra,), reason="gap"))
    executor = _executor(supervisor, max_revisions=1)

    asyncio.run(executor.run(graph))

    assert [n.id for n in graph.nodes] == ["n0", "extra"]
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)
    assert extra.result == "done"
    # Added mid-run, so it never passed through Swarm.plan.
    assert extra.metadata["model"] == "test-model"


def test_dropped_node_never_runs():
    graph = _graph("first", "second")
    supervisor = ScriptedSupervisor(Revision(drop_node_ids=("n1",), reason="redundant"))
    executor = _executor(supervisor, max_revisions=1)

    asyncio.run(executor.run(graph))

    assert [n.id for n in graph.nodes] == ["n0"]


def test_revision_budget_is_enforced():
    graph = _graph("first")
    revisions = [
        Revision(add_nodes=(Node(id=f"x{i}", label="more", depends_on=["n0"]),))
        for i in range(4)
    ]
    supervisor = ScriptedSupervisor(*revisions)
    executor = _executor(supervisor, max_revisions=2)

    asyncio.run(executor.run(graph))

    assert executor.revisions_used == 2
    assert len([n for n in graph.nodes if n.id.startswith("x")]) == 2


def test_zero_budget_means_supervisor_is_never_consulted():
    graph = _graph("first")
    supervisor = ScriptedSupervisor(
        Revision(add_nodes=(Node(id="extra", label="x", depends_on=["n0"]),)),
    )
    asyncio.run(_executor(supervisor, max_revisions=0).run(graph))
    assert supervisor.reviews == []
    assert len(graph.nodes) == 1


def test_invalid_revision_is_rejected_without_failing_the_run():
    graph = _graph("first", "second")
    tracer = Tracer()
    # The review fires after n0 completes, so dropping it is doubly
    # invalid: it is finished work, and n1 still depends on it.
    supervisor = ScriptedSupervisor(Revision(drop_node_ids=("n0",), reason="bad idea"))
    executor = _executor(supervisor, max_revisions=1, tracer=tracer)

    asyncio.run(executor.run(graph))

    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)
    assert executor.revisions_used == 0
    rejected = [
        s for s in tracer.summary() if s.get("status") == "revision_rejected"
    ]
    assert rejected and "only pending work" in rejected[0]["error"]


def test_supervisor_exception_does_not_fail_the_run():
    graph = _graph("first")
    tracer = Tracer()
    supervisor = ScriptedSupervisor(raises=RuntimeError("supervisor exploded"))
    executor = _executor(supervisor, max_revisions=1, tracer=tracer)

    asyncio.run(executor.run(graph))

    assert graph.nodes[0].status is NodeStatus.COMPLETED
    rejected = [s for s in tracer.summary() if s.get("status") == "revision_rejected"]
    assert "supervisor exploded" in rejected[0]["error"]


def test_applied_revision_is_traced():
    graph = _graph("first")
    extra = Node(id="extra", label="added", depends_on=["n0"])
    supervisor = ScriptedSupervisor(
        Revision(add_nodes=(extra,), reason="closes the evidence gap"),
    )
    tracer = Tracer()
    asyncio.run(_executor(supervisor, max_revisions=1, tracer=tracer).run(graph))

    applied = [s for s in tracer.summary() if s.get("status") == "revision_applied"]
    assert len(applied) == 1
    assert applied[0]["label"] == "closes the evidence gap"
    assert applied[0]["revision"]["added"] == ["extra"]
    assert applied[0]["revision"]["after_node"] == "n0"


def test_sync_executor_supports_revision():
    graph = _graph("first")
    extra = Node(id="extra", label="added", depends_on=["n0"])
    supervisor = ScriptedSupervisor(Revision(add_nodes=(extra,)))
    executor = Executor(
        provider=EchoProvider(), registry=Registry(), tracer=Tracer(),
        artifact_dir=None, supervisor=supervisor, max_revisions=1,
    )

    executor.run(graph)

    assert [n.id for n in graph.nodes] == ["n0", "extra"]
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)


def test_no_supervisor_leaves_execution_untouched():
    graph = _graph("first", "second")
    asyncio.run(_executor().run(graph))
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)


def test_negative_max_revisions_rejected():
    with pytest.raises(ValueError, match="max_revisions"):
        _executor(max_revisions=-1)


# ---------------------------------------------------------------------------
# LLMSupervisor parsing
# ---------------------------------------------------------------------------


def _parse(text):
    return LLMSupervisor._parse(text)


def test_parse_no_change():
    assert _parse('{"change": false, "reason": "plan is fine"}') is None


def test_parse_add():
    revision = _parse(
        '{"change": true, "reason": "missing evidence", '
        '"add": [{"id": "verify", "label": "Verify the claims", '
        '"depends_on": ["write"]}]}'
    )
    assert revision is not None
    assert revision.add_nodes[0].id == "verify"
    assert revision.add_nodes[0].depends_on == ["write"]
    assert revision.reason == "missing evidence"


def test_parse_strips_code_fences():
    revision = _parse(
        '```json\n{"change": true, "drop": ["n1"], "reason": "redundant"}\n```'
    )
    assert revision is not None
    assert revision.drop_node_ids == ("n1",)


def test_parse_malformed_json_is_no_change():
    assert _parse("I think the plan looks good, honestly") is None
    assert _parse("") is None
    assert _parse("```") is None


def test_parse_ignores_entries_without_labels():
    assert _parse('{"change": true, "add": [{"id": "x"}]}') is None


def test_parse_rewire():
    revision = _parse(
        '{"change": true, "rewire": {"n1": ["n0", "mid"]}, "reason": "reorder"}'
    )
    assert revision is not None
    assert revision.rewire == {"n1": ("n0", "mid")}


@pytest.mark.parametrize("change", ['"false"', '"true"', "1", '"yes"', "null", "[true]"])
def test_parse_change_must_be_json_true(change):
    text = '{"change": %s, "drop": ["n1"], "reason": "x"}' % change
    assert _parse(text) is None


@pytest.mark.parametrize("payload", [
    '"rewire": ["n1", "n0"]',
    '"rewire": {"n1": "n0"}',
    '"rewire": {"n1": [1]}',
    '"add": {"id": "x", "label": "y"}',
    '"add": "verify the claims"',
    '"add": ["verify the claims"]',
    '"add": [{"id": "x", "label": "y", "depends_on": 5}]',
    '"add": [{"id": "x", "label": "y", "depends_on": "n0"}]',
    '"add": [{"id": 7, "label": "y"}]',
    '"add": [{"id": "ok", "label": "Keep"}, {"id": "bad"}]',
    '"drop": "n1"',
    '"drop": ["n1", 2]',
    '"drop": {"n1": true}',
])
def test_parse_malformed_fields_are_no_change(payload):
    """rewire as a list used to raise AttributeError despite the contract."""
    assert _parse('{"change": true, %s, "reason": "x"}' % payload) is None


def test_parse_pathological_json_is_no_change():
    assert _parse("[" * 100_000 + "]" * 100_000) is None
    assert _parse('{"change": true, "drop": [' + "1" * 5000 + "]}") is None


@pytest.mark.parametrize("node_id", ["fix\nup", "x" * 65, "has space", "café", "a/b"])
def test_parse_refuses_added_ids_outside_the_plan_id_pattern(node_id):
    """A durable journal rejects such ids on the node's first call, and every
    resume replays the same checkpoint, so the run could never finish."""
    add = [{"id": node_id, "label": "Fix it", "depends_on": ["n0"]}]
    assert _parse(json.dumps({"change": True, "reason": "gap", "add": add})) is None


def test_parse_accepts_added_ids_in_the_plan_id_pattern():
    add = [{"id": "fix-up_2", "label": "Fix it", "depends_on": ["n0"]}]
    revision = _parse(json.dumps({"change": True, "reason": "gap", "add": add}))
    assert [node.id for node in revision.add_nodes] == ["fix-up_2"]


def _additions(count):
    return json.dumps({"change": True, "reason": "gaps", "add": [
        {"id": f"extra{i}", "label": f"Extra {i}", "depends_on": ["n0"]} for i in range(count)
    ]})


def test_parse_refuses_proposals_over_the_addition_cap():
    parse = LLMSupervisor._parse
    assert parse(_additions(3), max_added_nodes=2) is None
    assert len(parse(_additions(2), max_added_nodes=2).add_nodes) == 2
    assert len(parse(_additions(12)).add_nodes) == 12  # no cap unless one is given


@pytest.mark.parametrize("length, accepted", [(500, True), (501, False), (1_000_000, False)])
def test_parse_refuses_added_labels_over_the_length_cap(length, accepted):
    """A 1 MB label used to become the added step's prompt; the reason was
    already capped at 500 characters."""
    add = [{"id": "ok", "label": "Keep"}, {"id": "long", "label": "L" * length}]
    revision = _parse(json.dumps({"change": True, "reason": "gap", "add": add}))
    assert (revision is not None) is accepted
    if accepted:
        assert revision.add_nodes[1].label == "L" * length


class ProposingProvider(Provider):
    """Answers the supervisor with a fixed proposal; every node with "done"."""

    def __init__(self, proposal: str) -> None:
        self._proposal = proposal

    async def complete(self, system, prompt, model):
        from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT

        text = self._proposal if system == SUPERVISOR_SYSTEM_PROMPT else "done"
        return CompletionResult(text=text, prompt_tokens=1, completion_tokens=1)


def test_llm_supervisor_ignores_a_revision_over_the_default_cap(caplog):
    graph = _graph("first")
    supervisor = LLMSupervisor(ProposingProvider(_additions(4)), only_terminal=False)
    with caplog.at_level("WARNING", logger="smythe.supervisor"):
        asyncio.run(_executor(supervisor, max_revisions=3).run(graph))

    assert [n.id for n in graph.nodes] == ["n0"]
    assert "adds 4 nodes; max_added_nodes is 3" in caplog.text


def test_llm_supervisor_cap_is_configurable():
    graph = _graph("first")
    supervisor = LLMSupervisor(
        ProposingProvider(_additions(4)), only_terminal=False, max_added_nodes=4,
    )
    asyncio.run(_executor(supervisor, max_revisions=1).run(graph))
    assert [n.id for n in graph.nodes] == ["n0", "extra0", "extra1", "extra2", "extra3"]


def test_llm_supervisor_cannot_add_the_reserved_synthesis_id():
    add = [{"id": SYNTHESIS_NODE_ID, "label": "Merge", "depends_on": ["n0"]}]
    proposal = json.dumps({"change": True, "reason": "gap", "add": add})
    graph, tracer = _graph("first"), Tracer()
    supervisor = LLMSupervisor(ProposingProvider(proposal), only_terminal=False)
    asyncio.run(_executor(supervisor, max_revisions=1, tracer=tracer).run(graph))

    assert [n.id for n in graph.nodes] == ["n0"]
    rejected = [s for s in tracer.summary() if s.get("status") == "revision_rejected"]
    assert "id is reserved" in rejected[0]["error"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_max_added_nodes_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="max_added_nodes"):
        LLMSupervisor(EchoProvider(), max_added_nodes=value)


def test_non_default_addition_cap_is_recorded_and_bound():
    from smythe.provider import OfflineProvider

    default = LLMSupervisor(OfflineProvider(), model="m").workflow_description()
    assert "max_added_nodes" not in default

    custom = LLMSupervisor(OfflineProvider(), model="m", max_added_nodes=1)
    assert custom.workflow_description()["max_added_nodes"] == 1

    class Binding:
        default_model = "m"

        def snapshot_provider(self, provider):
            return provider

    bound = custom.bind_run(Binding())
    assert bound.workflow_description() == custom.workflow_description()


# ---------------------------------------------------------------------------
# Run-level cap on supervised growth
# ---------------------------------------------------------------------------


class GrowingProvider(Provider):
    """Each review proposes ``per_review`` new steps with fresh ids; nodes answer "done".

    A step labelled ``fail_once`` raises on its first call, standing in
    for a crash between two parts of a run.
    """

    def __init__(self, per_review: int, *, root: str, fail_once: str | None = None) -> None:
        self._per_review = per_review
        self._root = root
        self._fail_once = fail_once
        self.reviews = 0

    async def complete(self, system, prompt, model):
        if system != SUPERVISOR_SYSTEM_PROMPT:
            if prompt.splitlines()[0] == self._fail_once:
                self._fail_once = None
                raise RuntimeError("process lost")
            return CompletionResult(text="done", prompt_tokens=1, completion_tokens=1)
        self.reviews += 1
        add = [{"id": f"r{self.reviews}-{i}", "label": f"Extra {i}", "depends_on": [self._root]}
               for i in range(self._per_review)]
        return CompletionResult(
            text=json.dumps({"change": True, "reason": "gap", "add": add}),
            prompt_tokens=1, completion_tokens=1,
        )


def _added(graph):
    return [n.id for n in graph.nodes if n.metadata.get(REVISION_ADDED_KEY)]


def test_supervised_growth_is_capped_across_revisions(caplog):
    """max_added_nodes bounds one revision; without a run-level cap, ten
    revisions of two nodes each grew a one-node plan to 21 nodes."""
    graph = _graph("first")
    supervisor = LLMSupervisor(GrowingProvider(2, root="n0"), only_terminal=False)
    executor = _executor(supervisor, max_revisions=10)
    with caplog.at_level("WARNING", logger="smythe.supervisor"):
        asyncio.run(executor.run(graph))

    assert len(_added(graph)) == 8  # the default equals the generated-plan node limit
    assert len(graph.nodes) == 9
    assert all(n.status is NodeStatus.COMPLETED for n in graph.nodes)
    assert executor.revisions_used == 4
    assert "adds 2 nodes to 8 earlier additions; max_total_added_nodes is 8" in caplog.text


def test_growth_cap_counts_the_additions_a_proposal_keeps():
    """An added step that never ran frees its place when a proposal drops it."""
    graph = _graph("first", "second")
    graph.nodes[0].status = NodeStatus.COMPLETED
    graph.apply_revision(Revision(add_nodes=(Node(id="extra", label="Extra", depends_on=["n0"]),)))
    proposal = {"change": True, "reason": "swap", "add": [
        {"id": "better", "label": "Better", "depends_on": ["n0"]},
    ]}

    def review(proposal):
        supervisor = LLMSupervisor(ProposingProvider(json.dumps(proposal)),
                                   review_after={"n0"}, max_total_added_nodes=1)
        return asyncio.run(supervisor.review(
            graph, graph.nodes[0], task=None, revisions_remaining=1,
        ))

    assert review(proposal) is None
    revision = review(dict(proposal, drop=["extra"]))
    assert [n.id for n in revision.add_nodes] == ["better"]
    assert revision.drop_node_ids == ("extra",)


def test_supervised_growth_cap_survives_checkpoint_resume(tmp_path, caplog):
    """The count is read from saved node metadata, so resume cannot refill it."""
    from smythe.checkpoint import FileCheckpointStore
    from smythe.swarm import Swarm

    store = FileCheckpointStore(tmp_path)
    provider = GrowingProvider(2, root="a", fail_once="b")

    def swarm():
        supervisor = LLMSupervisor(provider, only_terminal=False, max_total_added_nodes=2)
        return Swarm(provider=provider, model="test-model", checkpoint_store=store,
                     artifact_dir=None, supervisor=supervisor, max_revisions=3)

    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="a", label="a"), Node(id="b", label="b", depends_on=["a"]),
    ])
    with pytest.raises(RuntimeError, match="process lost"):
        swarm().execute(graph)
    [execution_id] = store.list_ids()
    state = store.load(execution_id)
    saved = {n["id"]: n["metadata"].get(REVISION_ADDED_KEY) for n in state["graph"]["nodes"]}
    assert saved == {"a": None, "b": None, "r1-0": True, "r1-1": True}
    assert state["control"]["revisions_used"] == 1

    with caplog.at_level("WARNING", logger="smythe.supervisor"):
        result = swarm().resume(execution_id)

    assert [n.id for n in result.graph.nodes] == ["a", "b", "r1-0", "r1-1"]
    assert all(n.status is NodeStatus.COMPLETED for n in result.graph.nodes)
    assert provider.reviews == 4  # every later proposal was refused, none applied
    assert store.load(execution_id)["control"]["revisions_used"] == 1
    assert "max_total_added_nodes is 2" in caplog.text


def test_supervised_growth_cap_survives_durable_resume(tmp_path, monkeypatch):
    from smythe import OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm

    class ProcessLost(BaseException):
        pass

    reviews = []

    async def complete(self, system, prompt, model):
        if system != SUPERVISOR_SYSTEM_PROMPT:
            return CompletionResult(f"Done: {prompt}", cost_usd=0)
        reviews.append(prompt)
        add = [{"id": f"r{len(reviews)}-{i}", "label": f"Extra {i}", "depends_on": ["n0"]}
               for i in range(2)]
        proposal = {"change": True, "reason": "gap", "add": add}
        return CompletionResult(json.dumps(proposal), cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    with SQLiteWorkflowStore(tmp_path / "growth.db") as store:
        provider = OfflineProvider()
        swarm = Swarm(
            model="offline", provider=provider, architect=SimpleArchitect(), run_store=store,
            supervisor=LLMSupervisor(provider, model="offline", only_terminal=False,
                                     max_total_added_nodes=2),
            max_revisions=3,
        )
        original = store.save_checkpoint
        crashed = []

        def save(lease, revision, state, **kwargs):
            if crashed:
                raise ProcessLost("Process already stopped")
            if any(node["metadata"].get(REVISION_ADDED_KEY) and node["status"] == "completed"
                   for node in state["graph"]["nodes"]):
                crashed.append(lease.run_id)
                raise ProcessLost("Stopped after an added node completed")
            return original(lease, revision, state, **kwargs)

        monkeypatch.setattr(store, "save_checkpoint", save)
        with pytest.raises(ProcessLost):
            swarm.execute(ExecutionGraph([Topology.SERIAL], [Node(id="n0", label="First")]))
        run_id = crashed[0]
        saved = store.get_checkpoint(run_id)["checkpoint"]
        assert [node["id"] for node in saved["graph"]["nodes"]
                if node["metadata"].get(REVISION_ADDED_KEY)] == ["r1-0", "r1-1"]
        assert saved["revisions_used"] == 1

        monkeypatch.setattr(store, "save_checkpoint", original)
        resumed = swarm.resume(run_id)
        assert [node.id for node in resumed.graph.nodes] == ["n0", "r1-0", "r1-1"]
        assert all(node.status is NodeStatus.COMPLETED for node in resumed.graph.nodes)
        assert len(reviews) == 3  # the applied review, then one refused review per added node
        assert store.get_checkpoint(run_id)["checkpoint"]["revisions_used"] == 1
        assert store.audit(run_id)["ok"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "8", None])
def test_max_total_added_nodes_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="max_total_added_nodes"):
        LLMSupervisor(EchoProvider(), max_total_added_nodes=value)


def test_default_supervisor_description_is_unchanged():
    """Durable recipes hash this description; new defaults must not alter it."""
    from smythe.provider import OfflineProvider

    default = LLMSupervisor(OfflineProvider(), model="m").workflow_description()
    assert set(default) == {"type", "version", "provider", "model", "review_after", "only_terminal"}

    custom = LLMSupervisor(OfflineProvider(), model="m", max_total_added_nodes=2)
    assert custom.workflow_description() == dict(default, max_total_added_nodes=2)

    class Binding:
        default_model = "m"

        def snapshot_provider(self, provider):
            return provider

    assert custom.bind_run(Binding()).workflow_description() == custom.workflow_description()


def test_only_terminal_review_gate():
    graph = _graph("first", "second")
    supervisor = LLMSupervisor(EchoProvider())
    # n0 has a pending dependent, so it is mid-stream; n1 is terminal.
    graph.nodes[0].status = NodeStatus.COMPLETED
    assert supervisor._should_review(graph, graph.nodes[0]) is False
    graph.nodes[1].status = NodeStatus.COMPLETED
    assert supervisor._should_review(graph, graph.nodes[1]) is True


def test_default_review_gate_fires_when_a_fan_in_becomes_ready():
    left = Node(id="left", label="left", status=NodeStatus.COMPLETED)
    right = Node(id="right", label="right", status=NodeStatus.PENDING)
    join = Node(id="join", label="join", depends_on=["left", "right"])
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[left, right, join])
    supervisor = LLMSupervisor(EchoProvider())

    assert supervisor._should_review(graph, left) is False
    right.status = NodeStatus.COMPLETED
    assert supervisor._should_review(graph, right) is True


def test_parallel_terminal_leaves_produce_one_final_review():
    first = Node(id="first", label="first", status=NodeStatus.COMPLETED)
    last = Node(id="last", label="last", status=NodeStatus.PENDING)
    graph = ExecutionGraph(topology=[Topology.FORK_JOIN], nodes=[first, last])
    supervisor = LLMSupervisor(EchoProvider())

    assert supervisor._should_review(graph, first) is False
    last.status = NodeStatus.COMPLETED
    assert supervisor._should_review(graph, last) is True


def test_review_after_targets_specific_nodes():
    graph = _graph("first", "second")
    supervisor = LLMSupervisor(EchoProvider(), review_after={"n0"})
    assert supervisor._should_review(graph, graph.nodes[0]) is True
    assert supervisor._should_review(graph, graph.nodes[1]) is False


# ---------------------------------------------------------------------------
# Cached structure and durability after a revision
# ---------------------------------------------------------------------------


class PromptCapturingProvider(Provider):
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    async def complete(self, system, prompt, model):
        return CompletionResult(text="done", prompt_tokens=1, completion_tokens=1)

    async def chat(self, system, messages, model, tools=None):
        self.prompts[str(len(self.prompts))] = messages[0].content
        return CompletionResult(text="done", prompt_tokens=1, completion_tokens=1)


def test_revision_refreshes_cached_terminal_detection():
    """A node that gains a dependent mid-run stops being the deliverable.

    ExecutorBase caches dependency edges for speed; a revision must
    rebuild that cache or the old terminal keeps being told its output
    is the final deliverable.
    """
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    graph = _graph("first", "second")
    added = Node(id="extra", label="new terminal", depends_on=["n1"])
    supervisor = ScriptedSupervisor(Revision(add_nodes=(added,)))
    provider = PromptCapturingProvider()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=Tracer(),
        artifact_dir=None, supervisor=supervisor, max_revisions=1,
    )

    asyncio.run(executor.run(graph))

    assert [n.id for n in graph.nodes] == ["n0", "n1", "extra"]
    # n1 ran after the revision, so it must not carry the terminal note.
    n1_prompt = provider.prompts["1"]
    assert TERMINAL_DELIVERABLE_NOTE not in n1_prompt
    # The node the revision added is the real deliverable.
    assert TERMINAL_DELIVERABLE_NOTE in provider.prompts["2"]


def test_revised_graph_survives_checkpoint_roundtrip():
    from smythe.checkpoint import graph_from_dict, graph_to_dict

    graph = _graph("first")
    added = Node(id="extra", label="added by supervisor", depends_on=["n0"])
    supervisor = ScriptedSupervisor(Revision(add_nodes=(added,), reason="gap"))
    asyncio.run(_executor(supervisor, max_revisions=1).run(graph))

    restored = graph_from_dict(graph_to_dict(graph))
    assert [n.id for n in restored.nodes] == ["n0", "extra"]
    assert restored.nodes[1].depends_on == ["n0"]
    assert restored.nodes[1].metadata["model"] == "test-model"
    assert all(n.status is NodeStatus.COMPLETED for n in restored.nodes)


def test_truncated_supervisor_review_applies_no_revision(caplog):
    class TruncatedProposal(ProposingProvider):
        async def complete(self, system, prompt, model):
            result = await super().complete(system, prompt, model)
            from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT

            if system == SUPERVISOR_SYSTEM_PROMPT:
                result.stop_reason = "max_tokens"
            return result

    graph = _graph("first")
    supervisor = LLMSupervisor(TruncatedProposal(_additions(1)), only_terminal=False)
    with caplog.at_level("WARNING", logger="smythe.supervisor"):
        asyncio.run(_executor(supervisor, max_revisions=1).run(graph))

    assert [n.id for n in graph.nodes] == ["n0"]
    assert "cut off at the output token limit" in caplog.text


@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter", "incomplete"])
def test_refused_supervisor_review_applies_no_revision(caplog, stop_reason):
    class RefusedProposal(ProposingProvider):
        async def complete(self, system, prompt, model):
            result = await super().complete(system, prompt, model)
            from smythe.supervisor import SUPERVISOR_SYSTEM_PROMPT

            if system == SUPERVISOR_SYSTEM_PROMPT:
                result.stop_reason = stop_reason
            return result

    graph = _graph("first")
    supervisor = LLMSupervisor(RefusedProposal(_additions(1)), only_terminal=False)
    with caplog.at_level("WARNING", logger="smythe.supervisor"):
        asyncio.run(_executor(supervisor, max_revisions=1).run(graph))

    assert [n.id for n in graph.nodes] == ["n0"]
    assert "refused, filtered or stopped early" in caplog.text
    assert f"stop_reason={stop_reason!r}" in caplog.text
