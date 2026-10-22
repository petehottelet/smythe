"""Durable verification decisions survive each control-transition crash point."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from smythe import VerificationRecoveryError
from smythe.acceptance import AcceptancePolicy
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointStore,
    FileCheckpointStore,
    build_state,
    graph_from_dict,
    graph_to_dict,
)
from smythe.executor import Executor
from smythe.executor_base import (
    REJECTED_EXCERPT_BYTES,
    REJECTED_GENERATIONS_KEY,
    REJECTED_GENERATIONS_MAX_BYTES,
)
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import Artifact, CompletionResult, Provider
from smythe.registry import Registry
from smythe.swarm import Swarm
from smythe.validation import ValidationReport, ValidatorIdentity
from smythe.verifier import CallableVerifier
from smythe.tracer import Tracer


class ScriptedProvider(Provider):
    def __init__(self, judge="PASS"):
        self.calls = []
        self.judge = judge

    async def complete(self, system, prompt, model):
        node_id = prompt.splitlines()[0]
        self.calls.append(node_id)
        text = self.judge if node_id == "judge" else f"fresh {node_id}"
        return CompletionResult(text=text, cost_usd=0.1)


def graph_with_gate():
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"],
             verifies="draft", max_regenerations=1),
        Node(id="consumer", label="consumer", depends_on=["draft", "judge"]),
    ])


def completed_gate(*, state="pending", passed=False):
    graph = graph_with_gate()
    draft, judge, _ = graph.nodes
    draft.status = judge.status = NodeStatus.COMPLETED
    draft.result = "old draft"
    judge.result = "PASS" if passed else "FAIL old draft"
    judge.metadata["verification_receipt"] = {
        "version": 1, "state": state, "target_id": "draft",
        "judge_generation": 0, "target_generation": 0,
    }
    draft.metadata["cost_usd"] = judge.metadata["cost_usd"] = 0.1
    return graph


def save_graph(store, graph, *, version=CHECKPOINT_VERSION, completed=False):
    state = build_state(
        execution_id="run", status="completed" if completed else "failed",
        model="test", graph=graph, registry=Registry(), task=None,
        max_budget_usd=10.0, node_costs={"draft": 0.1, "judge": 0.1},
        output="stale cached output" if completed else None,
    )
    state["version"] = version
    store.save("run", state)
    return state


@pytest.mark.parametrize("completed", [False, True])
def test_resume_consumes_pending_verdict_before_dispatch_or_cached_output(tmp_path, completed):
    graph = completed_gate()
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=completed)
    provider = ScriptedProvider()
    result = Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == ["draft", "judge", "consumer"]
    assert result.output == "fresh consumer"
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert result.total_cost_usd == pytest.approx(0.5)
    assert all(n.metadata["execution_generation"] == 1 for n in result.graph.nodes)


@pytest.mark.parametrize("passed", [False, True])
def test_consumed_verdict_is_not_rejudged_on_resume(tmp_path, passed):
    graph = completed_gate(state="consumed", passed=passed)
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph)
    provider = ScriptedProvider()

    def must_not_rejudge(*args):
        pytest.fail("a consumed verdict was evaluated again")

    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   verifier=CallableVerifier(must_not_rejudge)).resume("run")
    assert provider.calls == ["consumer"]
    assert result.graph.nodes[0].result == "old draft"


@pytest.mark.parametrize("parallel", [False, True])
def test_control_saves_bypass_node_batching_and_snapshot_before_callbacks(tmp_path, parallel):
    class RecordingStore(FileCheckpointStore):
        def __init__(self):
            super().__init__(tmp_path)
            self.states = []

        def save(self, execution_id, state):
            self.states.append(state)  # Deliberately retain, without our own deepcopy.
            super().save(execution_id, state)

    store = RecordingStore()
    graph = graph_with_gate()
    provider = ScriptedProvider(judge="FAIL")
    result = Swarm(provider=provider, model="test", parallel=parallel,
                   checkpoint_store=store, checkpoint_every_n_nodes=100).execute(graph)
    judges = [state["graph"]["nodes"][1] for state in store.states]
    pending = next(j for j in judges if j["metadata"].get("verification_receipt", {}).get("state") == "pending")
    intent = next(j for j in judges if "regeneration_intent" in j["metadata"])
    reset = next(j for j in judges if j["metadata"].get("execution_generation") == 1
                 and j["status"] == "pending")
    assert pending["status"] == "completed"
    assert "regenerations_used" not in pending["metadata"]
    assert intent["metadata"]["regenerations_used"] == 1
    assert reset["result"] is None
    assert "regeneration_intent" not in reset["metadata"]
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert all(s["version"] == CHECKPOINT_VERSION for s in store.states)


@pytest.mark.parametrize("sync", [False, True])
def test_pending_control_write_failure_preserves_paid_verdict_without_retry(sync):
    graph = graph_with_gate()
    graph.nodes[1].failure_policy = FailurePolicy.RETRY
    graph.nodes[1].max_retries = 3
    provider = ScriptedProvider(judge="FAIL")
    budget = Sentinel(10.0)

    def broken_control_write():
        raise OSError("disk unavailable")

    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), budget=budget,
                  artifact_dir=None, on_control_update=broken_control_write)
    with pytest.raises(VerificationRecoveryError, match="persist") as caught:
        if sync:
            Executor(**kwargs).run(graph)
        else:
            asyncio.run(AsyncExecutor(**kwargs).run(graph))
    assert isinstance(caught.value.__cause__, OSError)
    assert provider.calls == ["draft", "judge"]
    assert graph.nodes[1].status is NodeStatus.COMPLETED
    assert graph.nodes[1].result == "FAIL"
    assert graph.nodes[1].metadata["verification_receipt"]["state"] == "pending"
    assert budget.total_cost_usd == pytest.approx(0.2)


@pytest.mark.parametrize("sync", [False, True])
def test_regular_update_failure_after_pending_save_cannot_retry_the_paid_judge(sync):
    graph = graph_with_gate()
    graph.nodes[1].failure_policy = FailurePolicy.RETRY
    graph.nodes[1].max_retries = 3
    provider = ScriptedProvider(judge="FAIL")
    saved = []

    def save_control():
        saved.append(graph_to_dict(graph))

    def broken_node_update(node):
        if node.id == "judge":
            raise OSError("observer failed after receipt save")

    kwargs = dict(provider=provider, registry=Registry(), tracer=Tracer(), artifact_dir=None,
                  on_control_update=save_control, on_node_update=broken_node_update)
    with pytest.raises(VerificationRecoveryError) as caught:
        if sync:
            Executor(**kwargs).run(graph)
        else:
            asyncio.run(AsyncExecutor(**kwargs).run(graph))
    assert isinstance(caught.value.__cause__, OSError)
    assert provider.calls == ["draft", "judge"]
    assert graph.nodes[1].status is NodeStatus.COMPLETED
    assert saved[0]["nodes"][1]["metadata"]["verification_receipt"]["state"] == "pending"


@pytest.mark.parametrize("partial_reset", [False, True])
def test_saved_intent_replays_idempotently_without_reevaluation_or_refilled_allowance(tmp_path, partial_reset):
    graph = completed_gate()
    graph.nodes[2].status = NodeStatus.SKIPPED
    graph.nodes[2].result = "obsolete skipped result"
    graph.nodes[0].metadata["artifacts"] = [{"path": "obsolete.png"}]
    graph.nodes[2].metadata["artifacts_discarded"] = 2
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    assert intent is not None
    if partial_reset:
        # Model a reset interrupted before its control commit. Absolute
        # generations/counters make replay independent of the interruption point.
        graph.nodes[0].status = NodeStatus.PENDING
        graph.nodes[0].result = None
        graph.nodes[0].metadata["execution_generation"] = 1
        # A durable intent is authoritative; its old receipt is no longer needed.
        graph.nodes[1].metadata.pop("verification_receipt")
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph)
    provider = ScriptedProvider()

    def must_not_rejudge(*args):
        pytest.fail("a saved rejection intent was evaluated again")

    result = Swarm(provider=provider, model="test", checkpoint_store=store,
                   verifier=CallableVerifier(must_not_rejudge)).resume("run")
    assert provider.calls == ["draft", "judge", "consumer"]
    assert result.graph.nodes[1].metadata["regenerations_used"] == 1
    assert all(n.metadata["execution_generation"] == 1 for n in result.graph.nodes)
    assert all("artifacts" not in n.metadata and "artifacts_discarded" not in n.metadata
               for n in result.graph.nodes)
    assert result.total_cost_usd == pytest.approx(0.5)
    before = list(provider.calls)
    Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == before


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("legacy_kind", ["ambiguous", "completed", "exhausted", "advisory"])
def test_legacy_verdict_recovery_is_conservative(tmp_path, version, legacy_kind):
    graph = completed_gate()
    judge = graph.nodes[1]
    judge.metadata.pop("verification_receipt")
    if legacy_kind == "exhausted":
        judge.metadata["regenerations_used"] = 1
    if legacy_kind == "advisory":
        judge.max_regenerations = 0
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, version=version, completed=legacy_kind == "completed")
    provider = ScriptedProvider()
    swarm = Swarm(provider=provider, model="test", checkpoint_store=store)
    if legacy_kind == "ambiguous":
        with pytest.raises(VerificationRecoveryError, match="Legacy checkpoint"):
            swarm.resume("run")
        assert provider.calls == []
    else:
        result = swarm.resume("run")
        assert provider.calls == ([] if legacy_kind == "completed" else ["consumer"])
        assert result.graph.nodes[0].result == "old draft"


@pytest.mark.parametrize("field,value", [
    ("judge_generation", True), ("judge_generation", -1),
    ("judge_generation", 2), ("target_generation", "0"),
    ("target_generation", 1), ("target_id", "consumer"),
    ("version", True), ("state", "unrecognized"),
])
def test_invalid_receipt_blocks_even_completed_cached_output(tmp_path, field, value):
    graph = completed_gate(state="consumed")
    graph.nodes[1].metadata["verification_receipt"][field] = value
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=True)
    provider = ScriptedProvider()
    with pytest.raises(VerificationRecoveryError):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []


@pytest.mark.parametrize("mutation", ["count", "inventory", "generation", "null"])
def test_malformed_intent_rejection_is_atomic(tmp_path, mutation):
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    if mutation == "count":
        intent["regenerations_used"] = 2
    elif mutation == "inventory":
        intent["affected_generations"].pop("consumer")
    elif mutation == "generation":
        intent["affected_generations"]["draft"] = True
    else:
        graph.nodes[1].metadata["regeneration_intent"] = None
    before = copy.deepcopy(graph_to_dict(graph))
    with pytest.raises(VerificationRecoveryError):
        executor.run(graph)
    assert graph_to_dict(graph) == before


@pytest.mark.parametrize("target_id", [[], {}, None])
@pytest.mark.parametrize("completed", [False, True])
def test_malformed_intent_target_blocks_resume_with_typed_error(tmp_path, target_id, completed):
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    intent = executor.prepare_regeneration(graph.nodes[1], graph)
    intent["target_id"] = target_id
    store = FileCheckpointStore(tmp_path)
    save_graph(store, graph, completed=completed)
    provider = ScriptedProvider()
    with pytest.raises(VerificationRecoveryError):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []


def test_intent_cannot_overwrite_work_from_a_newer_generation():
    graph = completed_gate()
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer())
    executor.prepare_regeneration(graph.nodes[1], graph)
    graph.nodes[0].metadata["execution_generation"] = 1
    graph.nodes[0].result = "already newer work"
    before = graph_to_dict(graph)
    with pytest.raises(VerificationRecoveryError, match="newer result"):
        executor.run(graph)
    assert graph_to_dict(graph) == before


def test_graph_snapshots_detach_nested_control_state():
    graph = completed_gate()
    snapshot = graph_to_dict(graph)
    restored = graph_from_dict(snapshot)
    graph.nodes[1].metadata["verification_receipt"]["state"] = "consumed"
    assert snapshot["nodes"][1]["metadata"]["verification_receipt"]["state"] == "pending"
    assert restored.nodes[1].metadata["verification_receipt"]["state"] == "pending"


class DrawingProvider(Provider):
    """Each draft call returns new image bytes; the judge replies from a queue."""

    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)
        self.draws = 0

    async def complete(self, system, prompt, model):
        node_id = prompt.splitlines()[0]
        if node_id == "judge":
            return CompletionResult(text=self.verdicts.pop(0), cost_usd=0.1)
        self.draws += 1
        image = Artifact(data=f"image {self.draws}".encode(), mime_type="image/png")
        return CompletionResult(text=f"{node_id} v{self.draws}", cost_usd=0.1, artifacts=[image])


def _gated_drawing():
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft", max_regenerations=1),
    ])


def test_strict_regeneration_keeps_the_rejected_files(tmp_path):
    swarm = Swarm(provider=DrawingProvider("FAIL: too dark", "PASS"), model="test",
                  artifact_dir=tmp_path, acceptance=AcceptancePolicy(mode="strict"))
    result = swarm.execute(_gated_drawing())

    run_dir = tmp_path / result.execution_id[:12]
    assert (run_dir / "draft_00.png").read_bytes() == b"image 1"
    assert (run_dir / "draft_g1_00.png").read_bytes() == b"image 2"
    [record] = result.graph.nodes[0].metadata["artifacts"]
    assert Path(record["path"]) == (run_dir / "draft_g1_00.png").resolve()
    assert record["sha256"] == hashlib.sha256(b"image 2").hexdigest()


def test_legacy_regeneration_still_writes_one_name_per_position(tmp_path):
    swarm = Swarm(provider=DrawingProvider("FAIL: too dark", "PASS"), model="test",
                  artifact_dir=tmp_path)
    result = swarm.execute(_gated_drawing())

    run_dir = tmp_path / result.execution_id[:12]
    assert sorted(path.name for path in run_dir.iterdir()) == ["draft_00.png"]
    assert (run_dir / "draft_00.png").read_bytes() == b"image 2"
    assert all(REJECTED_GENERATIONS_KEY not in node.metadata for node in result.graph.nodes)


def test_strict_regeneration_records_each_rejected_generation(tmp_path):
    swarm = Swarm(provider=DrawingProvider("FAIL: too dark", "PASS"), model="test",
                  artifact_dir=tmp_path, acceptance=AcceptancePolicy(mode="strict"))
    result = swarm.execute(_gated_drawing())
    draft, judge = result.graph.nodes

    [record] = draft.metadata[REJECTED_GENERATIONS_KEY]
    reason = record.pop("reason")
    assert "too dark" in reason
    assert record == {
        "version": 1, "generation": 0, "status": "completed", "judge_id": "judge",
        "result_sha256": hashlib.sha256(b"draft v1").hexdigest(), "result_bytes": 8,
        "excerpt": "draft v1",
        "artifacts": [{
            "path": str((tmp_path / result.execution_id[:12] / "draft_00.png").resolve()),
            "mime_type": "image/png", "sha256": hashlib.sha256(b"image 1").hexdigest(),
            "size_bytes": 7,
        }],
    }
    assert Path(record["artifacts"][0]["path"]).read_bytes() == b"image 1"
    [verdict] = judge.metadata[REJECTED_GENERATIONS_KEY]
    assert (verdict["generation"], verdict["excerpt"], verdict["reason"]) == (0, "FAIL: too dark", reason)
    assert draft.metadata["execution_generation"] == 1 and draft.result == "draft v2"


class JsonMemoryStore(CheckpointStore):
    """Keeps each saved state as detached JSON data, without file writes."""

    def __init__(self):
        self.states = {}

    def save(self, execution_id, state):
        self.states[execution_id] = json.loads(json.dumps(state))

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


class LongDraftProvider(Provider):
    TEXT = "caf\u00e9 " * 600  # 3,600 UTF-8 bytes

    def __init__(self):
        self.verdicts = ["FAIL: too long", "PASS"]

    async def complete(self, system, prompt, model):
        if prompt.splitlines()[0] == "judge":
            return CompletionResult(text=self.verdicts.pop(0), cost_usd=0.1)
        image = Artifact(data=b"image", mime_type="image/png")
        return CompletionResult(text=self.TEXT, cost_usd=0.1, artifacts=[image])


def test_a_long_rejected_result_keeps_an_excerpt_and_a_sidecar(tmp_path):
    swarm = Swarm(provider=LongDraftProvider(), model="test", artifact_dir=tmp_path,
                  acceptance=AcceptancePolicy(mode="strict"))
    result = swarm.execute(_gated_drawing())

    [record] = result.graph.nodes[0].metadata[REJECTED_GENERATIONS_KEY]
    full = LongDraftProvider.TEXT.encode("utf-8")
    assert record["result_bytes"] == len(full)
    assert record["result_sha256"] == hashlib.sha256(full).hexdigest()
    excerpt = record["excerpt"].encode("utf-8")
    assert len(excerpt) <= REJECTED_EXCERPT_BYTES < len(full) and full.startswith(excerpt)
    sidecar = Path(record["sidecar"])
    assert sidecar.name == "draft_g0_rejected.txt"
    assert sidecar.read_bytes() == full


def test_a_failed_sidecar_write_leaves_the_reset_for_resume(tmp_path, monkeypatch):
    import smythe.executor_base as executor_base

    store = JsonMemoryStore()
    publish = executor_base.publish_unique

    def no_sidecars(directory, name, data):
        if name.endswith("_rejected.txt"):
            raise OSError("disk full")
        return publish(directory, name, data)

    monkeypatch.setattr(executor_base, "publish_unique", no_sidecars)
    swarm = Swarm(provider=LongDraftProvider(), model="test", artifact_dir=tmp_path,
                  checkpoint_store=store, acceptance=AcceptancePolicy(mode="strict"))
    with pytest.raises(VerificationRecoveryError, match="retain"):
        swarm.execute(_gated_drawing())

    [execution_id] = store.states
    draft, judge = store.states[execution_id]["graph"]["nodes"]
    assert draft["status"] == "completed" and REJECTED_GENERATIONS_KEY not in draft["metadata"]
    assert "regeneration_intent" in judge["metadata"]
    assert not list(tmp_path.rglob("*_rejected.txt"))

    monkeypatch.setattr(executor_base, "publish_unique", publish)
    provider = LongDraftProvider()
    provider.verdicts = ["PASS"]
    result = Swarm(provider=provider, model="test", artifact_dir=tmp_path, checkpoint_store=store,
                   acceptance=AcceptancePolicy(mode="strict")).resume(execution_id)

    assert result.accepted
    [record] = result.graph.nodes[0].metadata[REJECTED_GENERATIONS_KEY]
    assert Path(record["sidecar"]).read_bytes() == LongDraftProvider.TEXT.encode("utf-8")


def test_without_an_artifact_dir_a_rejected_result_keeps_only_its_excerpt():
    swarm = Swarm(provider=LongDraftProvider(), model="test", artifact_dir=None,
                  acceptance=AcceptancePolicy(mode="strict"))
    result = swarm.execute(_gated_drawing())

    [record] = result.graph.nodes[0].metadata[REJECTED_GENERATIONS_KEY]
    assert "sidecar" not in record and "artifacts" not in record
    assert record["artifacts_discarded"] == 1
    assert len(record["excerpt"].encode("utf-8")) <= REJECTED_EXCERPT_BYTES


class PassesAnything:
    """A final validator that passes every output, so the gate alone decides."""

    identity = ValidatorIdentity.for_config("passes-anything", "1")

    def validate(self, subject):
        return ValidationReport("output", validator=self.identity)


@pytest.mark.parametrize("sidecars", [True, False])
def test_rejected_generation_records_stay_within_the_checkpoint_byte_cap(tmp_path, monkeypatch, sidecars):
    """Ordinary checkpoints rewrite the whole graph on every save, so the
    records a strict run keeps are capped in bytes per checkpoint.

    A record names its sidecar by absolute path, so with sidecars the size
    of each record, and whether the space left at the cap still fits one
    without its excerpt, depend on where the test runs. Without sidecars the
    records have the same sizes everywhere.
    """
    # Each of these characters is escaped to six bytes of checkpoint JSON.
    text = "\u00e9\u2603 " * 700

    class Verbose(Provider):
        async def complete(self, system, prompt, model):
            node_id = prompt.splitlines()[0]
            reply = f"FAIL: {text}" if node_id == "judge" else f"{node_id} {text}"
            return CompletionResult(text=reply, cost_usd=0.0)

    steps = [Node(id=f"step{i}", label=f"step{i}", depends_on=["draft"]) for i in range(4)]
    # The judge runs after every step, so each rejection discards 6 results.
    judge = Node(id="judge", label="judge", depends_on=["draft", *(n.id for n in steps)],
                 verifies="draft", max_regenerations=6)
    nodes = [Node(id="draft", label="draft"), *steps, judge]
    store = JsonMemoryStore()
    artifacts = tmp_path / "artifacts"
    # Sidecar durability is not under test here; flushing each one is slow on some hosts.
    monkeypatch.setattr("smythe._publish.os.fsync", lambda descriptor: None)
    # The four steps are all delivered, so a strict run needs a final
    # validator; one that passes everything leaves the exhausted gate to decide.
    policy = AcceptancePolicy(mode="strict", final_validators=[PassesAnything()])
    result = Swarm(provider=Verbose(), model="test", artifact_dir=artifacts if sidecars else None,
                   checkpoint_store=store, acceptance=policy).execute(
        ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes))
    assert result.acceptance.decision == "rejected"
    assert result.acceptance.validators[0].outcome.value == "passed"

    # The last checkpoint holds every record the run kept.
    state = store.states[result.execution_id]
    FileCheckpointStore(tmp_path / "with").save(result.execution_id, state)
    path = tmp_path / "with" / f"{result.execution_id}.json"
    records = [record for node in state["graph"]["nodes"]
               for record in node["metadata"].get(REJECTED_GENERATIONS_KEY, [])]
    # 6 nodes were reset 6 times; once the cap is reached, records drop their
    # excerpt and then are not kept at all.
    assert 0 < len(records) < 6 * 6
    assert all("excerpt" in record for record in records if record["generation"] == 0)
    if sidecars:
        # A record that is not kept leaves no sidecar behind.
        written = sorted(str(p.resolve()) for p in artifacts.rglob("*_rejected.txt"))
        assert written == sorted(record["sidecar"] for record in records)
    else:
        assert any("excerpt" not in record for record in records)
        assert not artifacts.exists()
        assert all("sidecar" not in record for record in records)

    for node in state["graph"]["nodes"]:
        node["metadata"].pop(REJECTED_GENERATIONS_KEY, None)
    FileCheckpointStore(tmp_path / "without").save(result.execution_id, state)
    growth = path.stat().st_size - (tmp_path / "without" / path.name).stat().st_size
    assert 0 < growth <= REJECTED_GENERATIONS_MAX_BYTES


def test_strict_replay_of_a_partly_applied_reset_records_each_generation_once():
    graph = completed_gate()
    graph.nodes[1].result = "FAIL: old draft"
    executor = Executor(provider=ScriptedProvider(), registry=Registry(), tracer=Tracer(),
                        artifact_dir=None, acceptance=AcceptancePolicy(mode="strict"))
    assert executor.prepare_regeneration(graph.nodes[1], graph) is not None
    # An interruption after the draft was reset, before the reset was saved.
    draft, judge, consumer = graph.nodes
    draft.status, draft.result = NodeStatus.PENDING, None
    draft.metadata["execution_generation"] = 1

    executor.recover_verification(graph)

    assert REJECTED_GENERATIONS_KEY not in draft.metadata
    assert REJECTED_GENERATIONS_KEY not in consumer.metadata  # it never ran
    [record] = judge.metadata[REJECTED_GENERATIONS_KEY]
    assert (record["generation"], record["excerpt"]) == (0, "FAIL: old draft")


@pytest.mark.parametrize("version", [True, False, 3.0, "3", None])
def test_checkpoint_version_requires_an_actual_supported_integer(tmp_path, version):
    store = FileCheckpointStore(tmp_path)
    save_graph(store, completed_gate(state="consumed"), version=version, completed=True)
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="version"):
        Swarm(provider=provider, model="test", checkpoint_store=store).resume("run")
    assert provider.calls == []
