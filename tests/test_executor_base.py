"""Tests for ExecutorBase shared helpers."""

import errno
import inspect
import os

import pytest

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


class PromptsByStep(Provider):
    """Records each node's prompt under its first line, the node's label; ``Check`` passes."""

    def __init__(self):
        self.prompts = {}

    async def complete(self, system, prompt, model):
        label = prompt.splitlines()[0]
        self.prompts[label] = prompt
        return CompletionResult(text="PASS" if label == "Check" else f"{label} output",
                                prompt_tokens=1, completion_tokens=1)


def _draft_and_judge():
    """Research feeds a draft that a judge checks; no node depends on the judge."""
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(label="Research", id="research"),
        Node(label="Draft", id="draft", depends_on=["research"]),
        Node(label="Check", id="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])


def _run(graph, *, parallel, **options):
    import asyncio

    from smythe.async_executor import AsyncExecutor
    from smythe.executor import Executor

    provider = PromptsByStep()
    executor_type = AsyncExecutor if parallel else Executor
    executor = executor_type(provider=provider, registry=Registry(), tracer=Tracer(),
                             artifact_dir=None, **options)
    if parallel:
        asyncio.run(executor.run(graph))
    else:
        executor.run(graph)
    assert all(node.status is NodeStatus.COMPLETED for node in graph.nodes)
    return provider.prompts


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_gated_deliverable_receives_the_terminal_note_and_its_judge_does_not(parallel):
    """The judge's edge to the draft it verifies does not make the draft
    non-terminal: the note goes to the node DELIVERABLE synthesis returns."""
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE
    from smythe.synthesizer import deliverable_sources

    graph = _draft_and_judge()
    prompts = _run(graph, parallel=parallel)

    assert TERMINAL_DELIVERABLE_NOTE in prompts["Draft"]
    assert TERMINAL_DELIVERABLE_NOTE not in prompts["Check"]
    assert TERMINAL_DELIVERABLE_NOTE not in prompts["Research"]
    assert [node.id for node in deliverable_sources(graph)[0]] == ["draft"]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_prompt_assembly_version_1_keeps_the_note_on_terminal_judges(parallel):
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    prompts = _run(_draft_and_judge(), parallel=parallel, prompt_assembly=1)

    assert TERMINAL_DELIVERABLE_NOTE in prompts["Check"]
    assert TERMINAL_DELIVERABLE_NOTE not in prompts["Draft"]
    assert TERMINAL_DELIVERABLE_NOTE not in prompts["Research"]


def test_a_step_that_depends_on_the_judge_is_the_terminal_step():
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE

    graph = _draft_and_judge()
    graph.nodes.append(Node(label="Publish", id="publish", depends_on=["judge", "draft"]))
    prompts = _run(graph, parallel=False)

    assert TERMINAL_DELIVERABLE_NOTE in prompts["Publish"]
    assert not any(TERMINAL_DELIVERABLE_NOTE in prompts[label]
                   for label in ("Research", "Draft", "Check"))


@pytest.mark.parametrize("strict", [False, True], ids=["legacy", "strict"])
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_swarm_gives_the_terminal_note_to_the_gated_draft_and_not_its_judge(parallel, strict):
    """A Swarm places the note as its executors do: on the node its
    DELIVERABLE synthesis returns, never on the judge that verifies it."""
    from smythe.acceptance import AcceptancePolicy
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE
    from smythe.swarm import Swarm

    provider = PromptsByStep()
    swarm = Swarm(provider=provider, model="test", parallel=parallel, artifact_dir=None,
                  acceptance=AcceptancePolicy(mode="strict") if strict else None)
    result = swarm.execute(_draft_and_judge())

    assert TERMINAL_DELIVERABLE_NOTE in provider.prompts["Draft"]
    assert TERMINAL_DELIVERABLE_NOTE not in provider.prompts["Check"]
    assert TERMINAL_DELIVERABLE_NOTE not in provider.prompts["Research"]
    # The judged draft is what the run delivers, and under a strict policy
    # its passed gate accepts it.
    assert result.output == "Draft output"
    assert result.accepted is strict


def test_a_step_that_depends_only_on_the_judge_is_a_final_step_beside_the_draft():
    from smythe.executor_base import TERMINAL_DELIVERABLE_NOTE
    from smythe.synthesizer import deliverable_sources

    graph = _draft_and_judge()
    graph.nodes.append(Node(label="Publish", id="publish", depends_on=["judge"]))
    prompts = _run(graph, parallel=False)

    assert TERMINAL_DELIVERABLE_NOTE in prompts["Draft"]
    assert TERMINAL_DELIVERABLE_NOTE in prompts["Publish"]
    assert not any(TERMINAL_DELIVERABLE_NOTE in prompts[label] for label in ("Research", "Check"))
    assert [node.id for node in deliverable_sources(graph)[0]] == ["draft", "publish"]


@pytest.mark.parametrize("value", [0, 3, True, 2.0, "2", None])
def test_executors_refuse_an_unknown_prompt_assembly_version(value):
    from smythe.async_executor import AsyncExecutor
    from smythe.executor import Executor

    for executor_type in (ExecutorBase, Executor, AsyncExecutor):
        with pytest.raises(ValueError, match="prompt_assembly"):
            executor_type(provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
                          prompt_assembly=value)
        parameter = inspect.signature(executor_type).parameters["prompt_assembly"]
        assert (parameter.kind, parameter.default) == (inspect.Parameter.KEYWORD_ONLY, 2)


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


def test_finalize_records_each_artifacts_digest_and_size(tmp_path):
    base = _make_base_with_dir(tmp_path)
    node = Node(label="make an image", id="hero")
    base.finalize_node_result(node, _artifact_result(images=2))

    import hashlib

    for record in node.metadata["artifacts"]:
        assert set(record) == {"path", "mime_type", "sha256", "size_bytes"}
        assert record["sha256"] == hashlib.sha256(b"\x89PNG-bytes").hexdigest()
        assert record["size_bytes"] == len(b"\x89PNG-bytes")


def _strict_base(artifact_dir) -> ExecutorBase:
    from smythe.acceptance import AcceptancePolicy

    return ExecutorBase(provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
                        artifact_dir=artifact_dir, acceptance=AcceptancePolicy(mode="strict"))


def test_strict_artifacts_name_their_generation_after_the_first(tmp_path):
    base = _strict_base(tmp_path)
    first = Node(label="make an image", id="hero")
    base.finalize_node_result(first, _artifact_result(images=2))
    later = Node(label="make an image", id="hero", metadata={"execution_generation": 2})
    base.finalize_node_result(later, _artifact_result(images=1))

    assert [Path(r["path"]).name for r in first.metadata["artifacts"]] == ["hero_00.png", "hero_01.png"]
    assert [Path(r["path"]).name for r in later.metadata["artifacts"]] == ["hero_g2_00.png"]
    # Legacy runs keep one name per position whatever the generation.
    legacy = Node(label="make an image", id="hero", metadata={"execution_generation": 2})
    _make_base_with_dir(tmp_path / "legacy").finalize_node_result(legacy, _artifact_result())
    assert Path(legacy.metadata["artifacts"][0]["path"]).name == "hero_00.png"


def test_strict_artifacts_never_replace_an_existing_file(tmp_path, monkeypatch):
    def no_replace(*args):
        pytest.fail("strict publication replaced a file")

    monkeypatch.setattr("smythe.executor_base.os.replace", no_replace)
    (tmp_path / "hero_00.png").write_bytes(b"earlier run")
    (tmp_path / "hero_00-2.png").write_bytes(b"earlier crash")
    node = Node(label="make an image", id="hero")
    _strict_base(tmp_path).finalize_node_result(node, _artifact_result())

    [record] = node.metadata["artifacts"]
    assert Path(record["path"]).name == "hero_00-3.png"
    assert Path(record["path"]).read_bytes() == b"\x89PNG-bytes"
    assert (tmp_path / "hero_00.png").read_bytes() == b"earlier run"
    assert (tmp_path / "hero_00-2.png").read_bytes() == b"earlier crash"
    assert not list(tmp_path.glob(".*.tmp"))


def _refuse_links(monkeypatch, code: int) -> None:
    def link(source, destination):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr("smythe._publish.os.link", link)


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL],
                         ids=["EPERM", "ENOTSUP", "EOPNOTSUPP", "EINVAL"])
def test_strict_artifacts_on_a_filesystem_without_hard_links_name_the_cause(
        tmp_path, monkeypatch, code):
    """A failed link used to surface as a bare OSError such as "Operation not
    permitted", which named neither hard links nor what to change."""
    _refuse_links(monkeypatch, code)
    node = Node(label="make an image", id="hero")
    with pytest.raises(OSError) as raised:
        _strict_base(tmp_path).finalize_node_result(node, _artifact_result())

    message = str(raised.value)
    assert raised.value.errno == code and isinstance(raised.value.__cause__, OSError)
    assert "Filesystem does not support exclusive artifact publication" in message
    assert "hard links" in message and "choose an artifact_dir" in message.lower()
    assert str(tmp_path) in message
    assert list(tmp_path.iterdir()) == [] and "artifacts" not in node.metadata


def test_publication_names_the_missing_hard_links_for_jobs_too(tmp_path, monkeypatch):
    from smythe.jobs.artifact_io import atomic_publish_bytes

    _refuse_links(monkeypatch, errno.EPERM)
    with pytest.raises(PermissionError, match="exclusive artifact publication") as raised:
        atomic_publish_bytes(tmp_path / "artifact.bin", b"bytes")
    assert "hard links" in str(raised.value) and "artifact_dir" not in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_other_link_failures_keep_their_own_error(tmp_path, monkeypatch):
    _refuse_links(monkeypatch, errno.ENOSPC)
    with pytest.raises(OSError) as raised:
        _strict_base(tmp_path).finalize_node_result(Node(label="image", id="hero"), _artifact_result())
    assert raised.value.errno == errno.ENOSPC and "hard links" not in str(raised.value)


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


@pytest.mark.parametrize(
    "value",
    [True, False, float("nan"), float("inf"), float("-inf"), 10**400, -0.5, "0.5", None],
    ids=["true", "false", "nan", "inf", "-inf", "huge-int", "negative", "string", "none"],
)
def test_retry_backoff_rejects_values_that_are_not_finite_non_negative_numbers(value):
    """True was accepted as a one-second backoff, NaN made every retry delay
    NaN, and infinity made a retry sleep forever."""
    from smythe.async_executor import AsyncExecutor

    for executor_type in (ExecutorBase, Executor, AsyncExecutor):
        with pytest.raises(ValueError, match="retry_backoff_s must be a finite non-negative number"):
            executor_type(
                provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
                retry_backoff_s=value,
            )


@pytest.mark.parametrize("value", [0, 0.0, 2, 0.25])
def test_retry_backoff_accepts_finite_non_negative_numbers(value):
    base = ExecutorBase(
        provider=DummyProvider(), registry=Registry(), tracer=Tracer(),
        retry_backoff_s=value,
    )
    assert 0.0 <= base.retry_delay_s(2) <= 2 * value


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


def _mocked_openai_image_provider():
    import base64
    from unittest.mock import AsyncMock, MagicMock

    from smythe.provider import OpenAIImageProvider

    item = MagicMock()
    item.b64_json = base64.b64encode(b"generated").decode()
    response = MagicMock(data=[item], usage=MagicMock(input_tokens=1, output_tokens=1))
    provider = OpenAIImageProvider(api_key="fake")
    provider._client = MagicMock()
    provider._client.images.generate = AsyncMock(return_value=response)
    return provider


def test_image_node_fails_instead_of_dropping_attached_dependency_images(tmp_path):
    """A node that attaches its dependencies' images must not reach the
    Image API without them."""
    graph, node = _graph_with_image_dep(tmp_path, attach=True)
    node.metadata["model"] = "gpt-image-2"
    provider = _mocked_openai_image_provider()

    with pytest.raises(ValueError, match="must not silently discard their inputs"):
        Executor(
            provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
        ).run(graph)

    assert node.status is NodeStatus.FAILED
    provider._client.images.generate.assert_not_called()


def test_image_node_without_attached_dependency_images_still_generates(tmp_path):
    graph, node = _graph_with_image_dep(tmp_path, attach=False)
    node.metadata["model"] = "gpt-image-2"
    provider = _mocked_openai_image_provider()

    Executor(
        provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
    ).run(graph)

    assert node.status is NodeStatus.COMPLETED
    provider._client.images.generate.assert_called_once()


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


def _graph_with_recorded_image(tmp_path, data=b"\x89PNG-recorded"):
    """A dependency whose image record carries the digest written with the file."""
    import hashlib

    image = tmp_path / "cand_00.png"
    image.write_bytes(data)
    graph, judge = _graph_with_image_dep(tmp_path, attach=True, image_bytes=data)
    graph.nodes[0].metadata["artifacts"] = [{
        "path": str(image), "mime_type": "image/png",
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }]
    return graph, judge, image


def test_dependency_image_is_attached_only_with_its_recorded_bytes(tmp_path):
    from smythe.executor_base import ArtifactIntegrityError

    base = _make_base_with_dir(None)
    graph, judge, image = _graph_with_recorded_image(tmp_path)
    assert [item.data for item in base.load_dep_image_artifacts(judge, graph)] == [b"\x89PNG-recorded"]
    # The file is replaced after its record was written, for example while
    # the run was paused. The judge must not see bytes the record never bound.
    image.write_bytes(b"\x89PNG-replaced")
    with pytest.raises(ArtifactIntegrityError, match="'cand_00.png' of node 'cand' changed"):
        base.load_dep_image_artifacts(judge, graph)
    # A record without a digest, as written before digests were recorded,
    # still attaches what it reads.
    del graph.nodes[0].metadata["artifacts"][0]["sha256"]
    assert [item.data for item in base.load_dep_image_artifacts(judge, graph)] == [b"\x89PNG-replaced"]


def test_judge_fails_before_calling_on_a_changed_dependency_image(tmp_path):
    from smythe.executor_base import ArtifactIntegrityError

    graph, judge, image = _graph_with_recorded_image(tmp_path)
    image.write_bytes(b"\x89PNG-replaced")
    provider = AttachmentCapturingProvider()
    with pytest.raises(ArtifactIntegrityError):
        Executor(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None).run(graph)
    assert judge.status is NodeStatus.FAILED
    assert provider.seen_attachments == []
