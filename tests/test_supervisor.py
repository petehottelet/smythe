"""Tests for adaptive supervision — revising a plan while it runs."""

from __future__ import annotations

import asyncio

import pytest

from smythe.async_executor import AsyncExecutor
from smythe.executor import Executor
from smythe.graph import (
    ExecutionGraph,
    Node,
    NodeStatus,
    Revision,
    RevisionError,
    Topology,
)
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.supervisor import LLMSupervisor, Supervisor
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


def test_only_terminal_review_gate():
    graph = _graph("first", "second")
    supervisor = LLMSupervisor(EchoProvider())
    # n0 has a pending dependent, so it is mid-stream; n1 is terminal.
    assert supervisor._should_review(graph, graph.nodes[0]) is False
    assert supervisor._should_review(graph, graph.nodes[1]) is True


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
