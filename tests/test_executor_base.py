"""Tests for ExecutorBase shared helpers."""

import os

from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider
from smythe.registry import Registry
from smythe.tracer import Tracer


class DummyProvider(Provider):
    async def complete(self, system, prompt, model):
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


def _make_base() -> ExecutorBase:
    return ExecutorBase(
        provider=DummyProvider(),
        registry=Registry(),
        tracer=Tracer(),
    )


def test_deps_satisfied_all_completed():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.COMPLETED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_skipped():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.SKIPPED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is True


def test_deps_satisfied_with_failed_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    a.status = NodeStatus.FAILED
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_with_pending_upstream():
    base = _make_base()
    a = Node(label="A", id="a")
    b = Node(label="B", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])
    assert base.deps_satisfied(b, graph) is False


def test_deps_satisfied_no_deps():
    base = _make_base()
    a = Node(label="A", id="a")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a])
    assert base.deps_satisfied(a, graph) is True


# --- terminal deliverable note -----------------------------------------

class PromptCapturingProvider(Provider):
    def __init__(self):
        self.prompts = []

    async def complete(self, system, prompt, model):
        self.prompts.append(prompt)
        return CompletionResult(text="ok", prompt_tokens=1, completion_tokens=1)


def test_terminal_node_prompt_carries_deliverable_note():
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    mid = Node(label="Mid", id="mid")
    prompt = ExecutorBase.build_user_prompt(mid, {"a": "r"}, is_terminal=False)
    assert TERMINAL_DELIVERABLE_NOTE not in prompt

    terminal = Node(label="Final", id="final")
    prompt = ExecutorBase.build_user_prompt(terminal, {"a": "r"}, is_terminal=True)
    assert TERMINAL_DELIVERABLE_NOTE in prompt

    # A terminal node with no upstream context needs no note.
    prompt = ExecutorBase.build_user_prompt(terminal, {}, is_terminal=True)
    assert TERMINAL_DELIVERABLE_NOTE not in prompt


def test_executor_marks_only_terminal_nodes():
    from smythe.executor import Executor
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    provider = PromptCapturingProvider()
    executor = Executor(provider=provider, registry=Registry(), tracer=Tracer())
    a = Node(label="Gather", id="a")
    b = Node(label="Deliver", id="b", depends_on=["a"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[a, b])

    executor.run(graph)

    first, last = provider.prompts
    assert TERMINAL_DELIVERABLE_NOTE not in first
    assert TERMINAL_DELIVERABLE_NOTE in last


# --- task context propagation ------------------------------------------

def test_root_node_prompt_carries_task_context():
    node = Node(label="Research the market", id="r")
    node.metadata["task_context"] = "Evaluate the widget market\n\nConstraints:\n- be brief"
    prompt = ExecutorBase.build_user_prompt(node, {})
    assert prompt.startswith("Overall task:\nEvaluate the widget market")
    assert "Your step: Research the market" in prompt


def test_task_context_respects_ablation_toggle(monkeypatch):
    import smythe.executor_base as eb

    node = Node(label="Research", id="r")
    node.metadata["task_context"] = "The big goal"
    monkeypatch.setattr(eb, "INCLUDE_TASK_CONTEXT", False)
    prompt = ExecutorBase.build_user_prompt(node, {})
    assert prompt == "Research"


def test_swarm_plan_stamps_task_context_on_every_node():
    # Downstream nodes need the artifact too: dependency results carry
    # analyses of it, and a verifier that can't see the artifact hedges.
    from smythe import Swarm, Task
    from smythe.provider import OfflineProvider

    plan = {
        "topology": ["fork_join"],
        "nodes": [
            {"id": "a", "label": "Angle A", "depends_on": [],
             "agent": {"name": "A", "persona": "p"}},
            {"id": "b", "label": "Angle B", "depends_on": [],
             "agent": {"name": "B", "persona": "p"}},
            {"id": "join", "label": "Merge", "depends_on": ["a", "b"],
             "agent": {"name": "J", "persona": "p"}},
        ],
    }
    swarm = Swarm(provider=OfflineProvider(plan=plan), model="test-model")
    task = Task(goal="Study the gizmo market", constraints=["stay brief"])
    graph = swarm.plan(task)

    for node in graph.nodes:
        assert "Study the gizmo market" in node.metadata["task_context"]
        assert "- stay brief" in node.metadata["task_context"]


def test_single_node_graph_is_not_stamped():
    from smythe import Swarm, Task
    from smythe.planner import SimpleArchitect
    from smythe.provider import OfflineProvider

    swarm = Swarm(provider=OfflineProvider(), model="test-model",
                  architect=SimpleArchitect())
    graph = swarm.plan(Task(goal="Do the thing"))
    assert "task_context" not in graph.nodes[0].metadata


# ---------------------------------------------------------------------------
# Artifact persistence (finalize_node_result)
# ---------------------------------------------------------------------------

from smythe.executor import Executor  # noqa: E402
from smythe.provider import Artifact  # noqa: E402


def _artifact_result(*, text: str = "done", images: int = 1) -> CompletionResult:
    return CompletionResult(
        text=text,
        artifacts=[Artifact(data=b"\x89PNG-bytes", mime_type="image/png")] * images,
    )


def _make_base_with_dir(artifact_dir) -> ExecutorBase:
    return ExecutorBase(
        provider=DummyProvider(),
        registry=Registry(),
        tracer=Tracer(),
        artifact_dir=artifact_dir,
    )


def test_finalize_persists_artifacts_and_records_paths(tmp_path):
    base = _make_base_with_dir(tmp_path / "assets")
    node = Node(label="make an image", id="hero")
    base.finalize_node_result(node, _artifact_result(images=2))

    records = node.metadata["artifacts"]
    assert len(records) == 2
    for record in records:
        from pathlib import Path
        path = Path(record["path"])
        assert path.exists()
        assert path.read_bytes() == b"\x89PNG-bytes"
        assert record["mime_type"] == "image/png"
    assert records[0]["path"].endswith("hero_00.png")
    # node.result stays the provider text verbatim (JSON consumers rely
    # on it); paths reach dependents via gather_dep_results instead.
    assert node.result == "done"


def test_finalize_replaces_artifact_atomically(tmp_path, monkeypatch):
    base = _make_base_with_dir(tmp_path)
    destination = tmp_path / "hero_00.png"
    destination.write_bytes(b"previous-complete-image")
    real_replace = os.replace
    observed: dict[str, object] = {}

    def inspect_replace(source, target):
        observed["source"] = Path(source)
        observed["old_bytes"] = Path(target).read_bytes()
        assert Path(source).parent == Path(target).parent
        real_replace(source, target)

    monkeypatch.setattr("smythe.executor_base.os.replace", inspect_replace)
    node = Node(label="make an image", id="hero")
    base.finalize_node_result(node, _artifact_result())

    assert observed["old_bytes"] == b"previous-complete-image"
    assert destination.read_bytes() == b"\x89PNG-bytes"
    assert not Path(observed["source"]).exists()


def test_finalize_artifact_only_result_gets_listing_text(tmp_path):
    base = _make_base_with_dir(tmp_path)
    node = Node(label="image only", id="img")
    base.finalize_node_result(node, _artifact_result(text="", images=1))
    assert "Generated artifacts:" in node.result
    assert "img_00.png" in node.result


def test_finalize_without_artifact_dir_notes_discard(tmp_path):
    base = _make_base_with_dir(None)
    node = Node(label="image", id="img")
    base.finalize_node_result(node, _artifact_result(text="", images=1))
    assert "discarded" in node.result
    assert node.metadata["artifacts_discarded"] == 1
    assert "artifacts" not in node.metadata


def test_finalize_plain_text_unchanged(tmp_path):
    base = _make_base_with_dir(tmp_path)
    node = Node(label="text", id="t")
    base.finalize_node_result(node, CompletionResult(text="just text"))
    assert node.result == "just text"
    assert "artifacts" not in node.metadata
    assert not any(tmp_path.iterdir())  # no dir contents created


class ArtifactProvider(Provider):
    """Provider whose every completion carries one PNG artifact."""

    async def complete(self, system, prompt, model):
        return CompletionResult(
            text="generated",
            prompt_tokens=5,
            completion_tokens=5,
            artifacts=[Artifact(data=b"\x89PNG-e2e", mime_type="image/png")],
        )


def test_executor_persists_artifacts_end_to_end(tmp_path):
    node = Node(label="Generate the hero image", id="hero")
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[node])
    executor = Executor(
        provider=ArtifactProvider(),
        registry=Registry(),
        tracer=Tracer(),
        artifact_dir=tmp_path / "out",
    )
    executor.run(graph)

    assert node.status == NodeStatus.COMPLETED
    assert (tmp_path / "out" / "hero_00.png").read_bytes() == b"\x89PNG-e2e"
    assert node.metadata["artifacts"][0]["mime_type"] == "image/png"
    assert node.result == "generated"


# ---------------------------------------------------------------------------
# Parallel hardening: sanitization, dep artifact injection, backoff
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from smythe.executor_base import _safe_filename_component  # noqa: E402


def test_safe_filename_component_sanitizes_hostile_ids(tmp_path):
    base = _make_base_with_dir(tmp_path)
    for hostile in ("../evil", "step:1", "a/b\\c", "..", "con?*|"):
        node = Node(label="x", id=hostile)
        base.finalize_node_result(node, _artifact_result(text="", images=1))
        path = Path(node.metadata["artifacts"][0]["path"])
        # File landed inside the artifact dir, not outside it.
        assert path.exists()
        assert tmp_path.resolve() in path.resolve().parents


def test_safe_filename_component_keeps_distinct_ids_distinct():
    assert _safe_filename_component("a/b") != _safe_filename_component("a_b")
    assert _safe_filename_component("safe-id.1") == "safe-id.1"


def test_gather_dep_results_injects_artifact_paths(tmp_path):
    base = _make_base_with_dir(tmp_path)
    dep = Node(label="make image", id="dep")
    dep.status = NodeStatus.COMPLETED
    base.finalize_node_result(dep, _artifact_result(text='{"ok": true}', images=1))
    # The stored result is pure JSON...
    import json
    assert json.loads(dep.result) == {"ok": True}

    consumer = Node(label="use image", id="use", depends_on=["dep"])
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[dep, consumer])
    dep_results = base.gather_dep_results(consumer, graph)
    # ...but dependents see the artifact paths appended.
    assert "dep_00.png" in dep_results["dep"]
    assert '{"ok": true}' in dep_results["dep"]


def test_retry_backoff_defaults_off_and_validates():
    base = _make_base_with_dir(None)
    assert base.retry_delay_s(1) == 0.0
    import pytest
    with pytest.raises(ValueError, match="retry_backoff_s"):
        ExecutorBase(
            provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
            retry_backoff_s=-1,
        )


def test_retry_backoff_grows_with_attempts():
    base = ExecutorBase(
        provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
        retry_backoff_s=0.5,
    )
    assert base.retry_delay_s(0) == 0.0
    for attempt, ceiling in ((1, 0.5), (2, 1.0), (3, 2.0)):
        for _ in range(5):
            d = base.retry_delay_s(attempt)
            assert 0.0 <= d <= ceiling


# ---------------------------------------------------------------------------
# Vision judge: attach_dep_artifacts
# ---------------------------------------------------------------------------


class AttachmentCapturingProvider(Provider):
    """Records the attachments of every chat() call."""

    def __init__(self):
        self.seen_attachments = []

    async def complete(self, system, prompt, model):
        return CompletionResult(text="ok")

    async def chat(self, system, messages, model, tools=None):
        self.seen_attachments.append(list(messages[0].attachments))
        return CompletionResult(text="judged")


def _graph_with_image_dep(tmp_path, *, attach: bool, image_bytes=b"\x89PNG-dep"):
    img = tmp_path / "cand_00.png"
    img.write_bytes(image_bytes)
    dep = Node(label="make image", id="cand")
    dep.status = NodeStatus.COMPLETED
    dep.result = "made it"
    dep.metadata["artifacts"] = [{"path": str(img), "mime_type": "image/png"}]
    judge = Node(
        label="judge it", id="judge", depends_on=["cand"],
        attach_dep_artifacts=attach,
    )
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[dep, judge]), judge


def test_judge_node_receives_dep_images_as_attachments(tmp_path):
    from smythe.executor import Executor

    graph, judge = _graph_with_image_dep(tmp_path, attach=True)
    provider = AttachmentCapturingProvider()
    executor = Executor(
        provider=provider, registry=Registry(), tracer=Tracer(),
        artifact_dir=None,
    )
    executor.run(graph)
    assert judge.status == NodeStatus.COMPLETED
    (attachments,) = provider.seen_attachments
    assert len(attachments) == 1
    assert attachments[0].data == b"\x89PNG-dep"
    assert attachments[0].mime_type == "image/png"


def test_attach_disabled_sends_no_attachments(tmp_path):
    from smythe.executor import Executor

    graph, _ = _graph_with_image_dep(tmp_path, attach=False)
    provider = AttachmentCapturingProvider()
    Executor(
        provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
    ).run(graph)
    (attachments,) = provider.seen_attachments
    assert attachments == []


def test_missing_and_non_image_artifacts_are_skipped(tmp_path):
    base = _make_base_with_dir(None)
    dep = Node(label="mixed", id="dep")
    dep.status = NodeStatus.COMPLETED
    ok = tmp_path / "ok.png"
    ok.write_bytes(b"\x89PNG-ok")
    dep.metadata["artifacts"] = [
        {"path": str(tmp_path / "gone.png"), "mime_type": "image/png"},
        {"path": str(ok), "mime_type": "application/pdf"},
        {"path": str(ok), "mime_type": "image/png"},
    ]
    judge = Node(label="judge", id="j", depends_on=["dep"], attach_dep_artifacts=True)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[dep, judge])
    attachments = base.load_dep_image_artifacts(judge, graph)
    assert len(attachments) == 1
    assert attachments[0].data == b"\x89PNG-ok"


def test_attachment_count_is_capped(tmp_path):
    base = _make_base_with_dir(None)
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG")
    dep = Node(label="many", id="dep")
    dep.status = NodeStatus.COMPLETED
    dep.metadata["artifacts"] = [
        {"path": str(img), "mime_type": "image/png"}
    ] * (base.MAX_ATTACHED_IMAGES + 5)
    judge = Node(label="judge", id="j", depends_on=["dep"], attach_dep_artifacts=True)
    graph = ExecutionGraph(topology=[Topology.SERIAL], nodes=[dep, judge])
    assert len(base.load_dep_image_artifacts(judge, graph)) == base.MAX_ATTACHED_IMAGES
