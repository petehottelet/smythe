"""A retry policy with ``persist_attempts`` bounds a node's attempts across resume.

Each attempt is saved as spent before its call, keyed by the node's
``execution_generation``, so a crash and resume continue the allowance instead
of refilling it: a ``RETRY`` node makes at most ``1 + max_retries`` provider
calls in total. Checkpoints that carry the allowance are version 5 and list
``attempt_allowance`` in ``requires``, so a reader that does not know it refuses
them, and a run resumes only under the policy it recorded. Default runs keep
writing version 4.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from fakes import Crash, crash_when, kill_and_reap, launch_child, wait_for_barrier
from smythe import FileCheckpointStore, RetryPolicy, Swarm
from smythe import checkpoint as checkpoints
from smythe import swarm as swarm_module
from smythe.async_executor import AsyncExecutor
from smythe.checkpoint import CheckpointStore, attempt_allowance_from_state, build_state
from smythe.cli import EXIT_LOCAL_ERROR, EXIT_OK, main
from smythe.executor import Executor
from smythe.executor_base import ExecutorBase
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider, ProviderFeatures
from smythe.registry import Registry
from smythe.tracer import Tracer

MAX_RETRIES = 2
ALLOWANCE = 1 + MAX_RETRIES
POLICY = RetryPolicy(persist_attempts=True)
MODES = ["serial", "parallel"]


class Flaky(Provider):
    """Fails or answers each call in turn, repeating the last outcome; it sends each call once.

    An outcome is reply text or an exception to raise, such as a transient
    ``ConnectionError`` or a ``Crash`` that stops the run where a killed
    process would. ``calls`` holds every prompt it received.
    """

    def __init__(self, *outcomes) -> None:
        self.outcomes = outcomes or (ConnectionError("connection reset"),)
        self.calls: list[str] = []

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0)

    async def complete(self, system, prompt, model):
        self.calls.append(prompt)
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return CompletionResult(outcome, prompt_tokens=1, completion_tokens=1)


class Store(CheckpointStore):
    """Keeps each saved state as detached JSON data, as a file store would, without file writes."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    def save(self, execution_id, state):
        self.states[execution_id] = json.loads(json.dumps(state))

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


def step(**options) -> Node:
    options.setdefault("max_retries", MAX_RETRIES)
    return Node(id="step", label="Step", failure_policy=FailurePolicy.RETRY, **options)


def graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [step()])


def build(store, provider, policy=POLICY, *, parallel: bool = False) -> Swarm:
    return Swarm(model="test-model", provider=provider, checkpoint_store=store, artifact_dir=None,
                 retry_policy=policy, parallel=parallel)


def used(state, node_id: str = "step"):
    """The attempts a saved state records for *node_id*, or None."""
    record = state.get("control", {}).get("attempts_used", {}).get(node_id)
    return None if record is None else record["attempts"]


def saved(store):
    (execution_id,) = store.list_ids()
    return execution_id, store.load(execution_id)


# ---------------------------------------------------------------------------
# The policy and the saved allowance
# ---------------------------------------------------------------------------


def test_persist_attempts_is_an_opt_in_policy_field():
    assert RetryPolicy().persist_attempts is False
    assert POLICY.persist_attempts is True and POLICY != RetryPolicy()
    for value in (1, "yes", None):
        with pytest.raises(ValueError, match="persist_attempts"):
            RetryPolicy(persist_attempts=value)


def test_a_policy_round_trips_through_its_saved_fields():
    policy = RetryPolicy(retry_classes={"transient", "provider_rejected"}, honor_retry_after=False,
                         backoff_max_s=5, deadline_s=30, persist_attempts=True)
    fields = policy.to_dict()
    assert fields == {"retry_classes": ["provider_rejected", "transient"],
                      "honor_retry_after": False, "backoff_max_s": 5.0, "deadline_s": 30.0,
                      "persist_attempts": True}
    assert RetryPolicy.from_dict(fields) == policy
    with pytest.raises(ValueError, match="later_field"):
        RetryPolicy.from_dict({**fields, "later_field": 1})
    for data in (["transient"], {**fields, "retry_classes": [["transient"]]},
                 {**fields, "deadline_s": "30"}):
        with pytest.raises(ValueError):
            RetryPolicy.from_dict(data)


@pytest.mark.parametrize("mode", MODES)
def test_a_persisted_run_saves_its_policy_and_attempts_as_version_5(tmp_path, mode):
    store = FileCheckpointStore(tmp_path)
    provider = Flaky(ConnectionError("reset"), "done")

    result = build(store, provider, parallel=mode == "parallel").execute(graph())

    assert result.output == "done" and len(provider.calls) == 2
    _, state = saved(store)
    assert state["version"] == 5 and state["requires"] == ["attempt_allowance"]
    assert state["control"] == {
        "revisions_used": 0, "retry_policy": POLICY.to_dict(),
        "attempts_used": {"step": {"execution_generation": 0, "attempts": 2}},
    }
    assert "delivery_acceptance" not in state
    assert attempt_allowance_from_state(state) == (
        POLICY, {"step": {"execution_generation": 0, "attempts": 2}})


@pytest.mark.parametrize("policy", [None, RetryPolicy()], ids=["no-policy", "not-persisted"])
def test_default_checkpoints_stay_version_4(policy):
    store = Store()

    build(store, Flaky(ConnectionError("reset"), "done"), policy).execute(graph())

    _, state = saved(store)
    assert state["version"] == checkpoints.CHECKPOINT_VERSION == 4
    assert "requires" not in state and state["control"] == {"revisions_used": 0}
    assert attempt_allowance_from_state(state) is None


def test_build_state_lists_the_allowance_after_delivery_acceptance():
    from smythe.acceptance import DeliveryAcceptance

    common = dict(execution_id="run1", status="running", model="m",
                  graph=ExecutionGraph([Topology.SERIAL], [step()]), registry=Registry(),
                  task=None, max_budget_usd=None, node_costs={})
    control = {"revisions_used": 0, "retry_policy": POLICY.to_dict(), "attempts_used": {}}
    plain = build_state(**common, control=control)
    both = build_state(**common, control=control,
                       delivery_acceptance=DeliveryAcceptance("not_evaluated", "b" * 64))
    assert plain["version"] == 5 and plain["requires"] == ["attempt_allowance"]
    assert both["requires"] == ["delivery_acceptance", "attempt_allowance"]
    assert attempt_allowance_from_state(both) == (POLICY, {})


# ---------------------------------------------------------------------------
# Crash and resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("boundary", ["record", "call"])
def test_a_crash_between_attempts_does_not_refill_the_allowance(monkeypatch, mode, boundary):
    """The process is lost exactly where the second attempt is saved, or once it is sent.

    At ``record`` the save that would spend the second attempt never lands, so
    the checkpoint still records one; at ``call`` it landed and the call was
    sent, so it records two. Either way the run, crash and resume together
    make exactly ``1 + max_retries`` calls.
    """
    store = Store()
    if boundary == "record":
        first = Flaky()
        crash_when(monkeypatch, store, "save", lambda execution_id, state: used(state) == 2)
    else:
        first = Flaky(ConnectionError("reset"), Crash("process lost during the call"))
        # A killed process writes nothing more, not even its failed checkpoint.
        crash_when(monkeypatch, store, "save", lambda execution_id, state: len(first.calls) == 2)

    with pytest.raises(Crash):
        build(store, first, parallel=mode == "parallel").execute(graph())
    monkeypatch.undo()

    execution_id, state = saved(store)
    assert state["status"] == "running"
    assert used(state) == {"record": 1, "call": 2}[boundary] == len(first.calls)

    second = Flaky()
    with pytest.raises(ConnectionError):
        build(store, second).resume(execution_id)

    assert len(first.calls) + len(second.calls) == ALLOWANCE
    _, state = saved(store)
    assert state["status"] == "failed" and used(state) == ALLOWANCE

    # A spent allowance stays spent: the node fails again without a call.
    third = Flaky("done")
    with pytest.raises(RuntimeError, match="used all 3 attempts"):
        build(store, third).resume(execution_id)
    assert third.calls == []


def test_without_persist_attempts_a_resume_refills_the_allowance(monkeypatch):
    store = Store()
    first = Flaky(ConnectionError("reset"), Crash("process lost during the call"))
    crash_when(monkeypatch, store, "save", lambda execution_id, state: len(first.calls) == 2)
    with pytest.raises(Crash):
        build(store, first, RetryPolicy()).execute(graph())
    monkeypatch.undo()

    second = Flaky()
    with pytest.raises(ConnectionError):
        build(store, second, RetryPolicy()).resume(saved(store)[0])

    assert (len(first.calls), len(second.calls)) == (2, ALLOWANCE)


def test_a_resumed_attempt_can_succeed_within_the_allowance(monkeypatch):
    store = Store()
    first = Flaky(ConnectionError("reset"), Crash("process lost during the call"))
    crash_when(monkeypatch, store, "save", lambda execution_id, state: len(first.calls) == 2)
    with pytest.raises(Crash):
        build(store, first).execute(graph())
    monkeypatch.undo()

    second = Flaky("done")
    result = build(store, second).resume(saved(store)[0])

    assert result.output == "done" and len(second.calls) == 1
    assert used(saved(store)[1]) == ALLOWANCE


def test_a_failed_attempt_record_stops_the_node_before_its_call(monkeypatch):
    store = Store()
    save = store.save

    def failing(execution_id, state):
        if state["status"] == "running" and used(state) == 2:
            raise OSError("disk full")
        save(execution_id, state)

    monkeypatch.setattr(store, "save", failing)
    provider = Flaky()
    with pytest.raises(OSError, match="disk full"):
        build(store, provider).execute(graph())

    # The attempt whose record was not saved was never sent, and is not counted.
    assert len(provider.calls) == 1
    _, state = saved(store)
    assert state["status"] == "failed" and used(state) == 1


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


class Drafts(Provider):
    """A draft that fails once per generation, and a judge that rejects the first draft."""

    def __init__(self) -> None:
        self.drafts: list[str] = []
        self.verdicts = ["FAIL: too short", "PASS"]

    def feature_support(self, model):
        return ProviderFeatures(sdk_retries=0)

    async def complete(self, system, prompt, model):
        if prompt.startswith("Check"):
            return CompletionResult(self.verdicts.pop(0), prompt_tokens=1, completion_tokens=1)
        self.drafts.append(prompt)
        if len(self.drafts) % 2:
            raise ConnectionError("reset")
        return CompletionResult(f"draft {len(self.drafts) // 2}", prompt_tokens=1,
                                completion_tokens=1)


def test_a_regenerated_node_gets_a_fresh_allowance():
    store = Store()
    provider = Drafts()
    gated = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft", failure_policy=FailurePolicy.RETRY, max_retries=1),
        Node(id="judge", label="Check", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])

    result = build(store, provider).execute(gated)

    # Each generation spent its whole allowance of two attempts.
    assert result.output == "draft 2" and len(provider.drafts) == 4
    _, state = saved(store)
    assert state["control"]["attempts_used"] == {
        "draft": {"execution_generation": 1, "attempts": 2},
        "judge": {"execution_generation": 1, "attempts": 1},
    }


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


EXECUTORS = {"serial": Executor, "parallel": AsyncExecutor}


def execute(executor, run_graph: ExecutionGraph) -> None:
    if isinstance(executor, Executor):
        executor.run(run_graph)
    else:
        asyncio.run(executor.run(run_graph))


def run(mode, provider, *, policy=POLICY, **options):
    executor = EXECUTORS[mode](provider=provider, registry=Registry(), tracer=Tracer(),
                               artifact_dir=None, retry_policy=policy, **options)
    run_graph = graph()
    execute(executor, run_graph)
    return executor, run_graph.nodes[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("generation,calls", [(0, 1), (1, ALLOWANCE)])
def test_an_executor_starts_from_the_attempts_its_generation_used(mode, generation, calls):
    provider = Flaky()
    seed = {"step": {"execution_generation": generation, "attempts": 2}}

    with pytest.raises(ConnectionError):
        run(mode, provider, attempts_used=seed)

    # A record for another generation leaves the node a fresh allowance.
    assert len(provider.calls) == calls


@pytest.mark.parametrize("mode", MODES)
def test_a_crash_at_notify_update_between_attempts_keeps_the_saved_count(monkeypatch, mode):
    """Without a control hook each attempt is saved through ``notify_update``.

    The process is lost in the ``notify_update`` that would save the second
    attempt, so that attempt is never sent, and an executor seeded with the
    last saved count makes the rest of the allowance and no more.
    """
    saves = []
    first = Flaky()
    executor = EXECUTORS[mode](provider=first, registry=Registry(), tracer=Tracer(),
                               artifact_dir=None, retry_policy=POLICY,
                               on_node_update=lambda node: saves.append(executor.attempts_used))
    crash_when(monkeypatch, ExecutorBase, "notify_update",
               lambda self, node: self.attempts_used["step"]["attempts"] == 2)

    with pytest.raises(Crash):
        execute(executor, graph())
    monkeypatch.undo()

    assert len(first.calls) == 1
    assert saves == [{"step": {"execution_generation": 0, "attempts": 1}}]
    second = Flaky()
    with pytest.raises(ConnectionError):
        run(mode, second, attempts_used=saves[-1])
    assert len(first.calls) + len(second.calls) == ALLOWANCE


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_with_a_spent_allowance_skips_a_skip_node_without_a_call(mode):
    provider = Flaky("done")
    executor = EXECUTORS[mode](provider=provider, registry=Registry(), tracer=Tracer(),
                               artifact_dir=None, retry_policy=POLICY,
                               attempts_used={"step": {"execution_generation": 0, "attempts": 1}})
    skipped = ExecutionGraph([Topology.SERIAL], [
        Node(id="step", label="Step", failure_policy=FailurePolicy.SKIP)])

    execute(executor, skipped)

    assert provider.calls == [] and skipped.nodes[0].status is NodeStatus.SKIPPED
    assert "used all 1 attempts" in skipped.nodes[0].result
    assert executor.attempts_used == {"step": {"execution_generation": 0, "attempts": 1}}


@pytest.mark.parametrize("mode", MODES)
def test_an_executor_reports_the_attempts_it_spent(mode):
    executor, node = run(mode, Flaky(ConnectionError("reset"), "done"))

    assert node.status is NodeStatus.COMPLETED
    assert executor.attempts_used == {"step": {"execution_generation": 0, "attempts": 2}}
    # The report is a copy.
    executor.attempts_used["step"]["attempts"] = 0
    assert executor.attempts_used["step"]["attempts"] == 2


@pytest.mark.parametrize("mode", MODES)
def test_a_reused_executor_gives_a_later_graph_its_full_allowance(mode):
    """The seed and an earlier graph's attempts belong to that run, not to the next.

    The seeded first run spends the last of its node's allowance. A later
    graph with the same node id is a new run, so its node gets all its
    attempts: two failures and then an answer.
    """
    provider = Flaky("done", ConnectionError("reset"), ConnectionError("reset"), "again")
    executor = EXECUTORS[mode](provider=provider, registry=Registry(), tracer=Tracer(),
                               artifact_dir=None, retry_policy=POLICY,
                               attempts_used={"step": {"execution_generation": 0, "attempts": 2}})
    first, second = graph(), graph()

    execute(executor, first)
    assert len(provider.calls) == 1 and first.nodes[0].result == "done"
    assert executor.attempts_used == {"step": {"execution_generation": 0, "attempts": ALLOWANCE}}

    execute(executor, second)

    assert second.nodes[0].status is NodeStatus.COMPLETED and second.nodes[0].result == "again"
    assert len(provider.calls) == 1 + ALLOWANCE
    assert executor.attempts_used == {"step": {"execution_generation": 0, "attempts": ALLOWANCE}}


@pytest.mark.parametrize("attempts_used,policy,error", [
    ({"step": {"execution_generation": 0, "attempts": 1}}, RetryPolicy(), "persist_attempts"),
    ({"step": {"execution_generation": 0, "attempts": 1}}, None, "persist_attempts"),
    ({"step": {"execution_generation": 0, "attempts": -1}}, POLICY, "attempts"),
    ({"step": {"execution_generation": True, "attempts": 1}}, POLICY, "execution_generation"),
    ({"step": {"execution_generation": 0}}, POLICY, "attempts"),
    ({"step": {"execution_generation": 0, "attempts": 1, "extra": 1}}, POLICY, "attempts"),
    ({"step": [0, 1]}, POLICY, "attempts"),
    ([("step", 1)], POLICY, "attempts"),
])
def test_an_executor_refuses_malformed_attempts_used(attempts_used, policy, error):
    with pytest.raises(ValueError, match=error):
        AsyncExecutor(provider=Flaky(), registry=Registry(), tracer=Tracer(),
                      retry_policy=policy, attempts_used=attempts_used)


# ---------------------------------------------------------------------------
# Readers and resume policy
# ---------------------------------------------------------------------------


def finished(monkeypatch, status):
    """A store holding one persisted run that is ``completed`` or ``running``."""
    store = Store()
    if status == "completed":
        build(store, Flaky("done")).execute(graph())
        return store
    first = Flaky(ConnectionError("reset"), Crash("process lost during the call"))
    with monkeypatch.context() as patch:
        crash_when(patch, store, "save", lambda execution_id, state: len(first.calls) == 2)
        with pytest.raises(Crash):
            build(store, first).execute(graph())
    return store


@pytest.mark.parametrize("status", ["completed", "running"])
def test_a_reader_that_knows_only_delivery_acceptance_refuses_the_allowance(monkeypatch, status):
    store = finished(monkeypatch, status)
    execution_id, state = saved(store)
    assert state["requires"] == ["attempt_allowance"]
    monkeypatch.setattr(checkpoints, "KNOWN_CHECKPOINT_REQUIREMENTS",
                        frozenset({"delivery_acceptance"}))
    provider = Flaky("done")

    with pytest.raises(ValueError, match="attempt_allowance"):
        build(store, provider).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("status", ["completed", "running"])
def test_a_reader_of_versions_1_to_4_refuses_the_allowance(monkeypatch, status):
    store = finished(monkeypatch, status)
    execution_id, state = saved(store)
    monkeypatch.setattr(swarm_module, "SUPPORTED_CHECKPOINT_VERSIONS", (1, 2, 3, 4))
    provider = Flaky("done")

    with pytest.raises(ValueError, match="version 5"):
        build(store, provider).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


@pytest.mark.parametrize("policy", [
    None, RetryPolicy(), RetryPolicy(persist_attempts=True, deadline_s=30),
    RetryPolicy(persist_attempts=True, retry_classes={"transient", "provider_rejected"}),
], ids=["no-policy", "not-persisted", "deadline", "classes"])
def test_resume_under_a_changed_policy_is_refused_before_any_call(monkeypatch, policy):
    store = finished(monkeypatch, "running")
    execution_id, state = saved(store)
    provider = Flaky("done")

    with pytest.raises(ValueError, match="retry policy"):
        build(store, provider, policy).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


def test_a_run_saved_without_the_allowance_cannot_resume_under_it(monkeypatch):
    store = Store()
    first = Flaky(ConnectionError("reset"), Crash("process lost during the call"))
    crash_when(monkeypatch, store, "save", lambda execution_id, state: len(first.calls) == 2)
    with pytest.raises(Crash):
        build(store, first, RetryPolicy()).execute(graph())
    monkeypatch.undo()
    execution_id, state = saved(store)
    provider = Flaky("done")

    with pytest.raises(ValueError, match="persist_attempts"):
        build(store, provider).resume(execution_id)
    assert provider.calls == [] and store.load(execution_id) == state


def test_a_completed_run_resumes_under_any_policy_without_a_call(monkeypatch):
    store = finished(monkeypatch, "completed")
    execution_id, _ = saved(store)
    provider = Flaky("other")

    result = build(store, provider, None).resume(execution_id)

    assert result.output == "done" and provider.calls == []


@pytest.mark.parametrize("change", [
    lambda state: state["control"].pop("attempts_used"),
    lambda state: state["control"].pop("retry_policy"),
    lambda state: state["control"]["retry_policy"].update(persist_attempts=False),
    lambda state: state["control"]["retry_policy"].update(backoff_max_s=-1),
    lambda state: state["control"]["attempts_used"].update(step={"attempts": 1}),
    lambda state: state.update(requires=[]),
], ids=["no-attempts", "no-policy", "not-persisted", "bad-policy", "bad-record", "unlisted"])
@pytest.mark.parametrize("status", ["completed", "running"])
def test_a_malformed_allowance_is_refused_before_any_call(monkeypatch, status, change):
    store = finished(monkeypatch, status)
    execution_id, state = saved(store)
    change(state)
    store.save(execution_id, state)
    provider = Flaky("done")

    with pytest.raises(ValueError):
        build(store, provider).resume(execution_id)
    assert provider.calls == []


def test_inspection_names_the_retry_policy_and_refuses_a_malformed_allowance(tmp_path, monkeypatch,
                                                                            capsys):
    execution_id, state = saved(finished(monkeypatch, "running"))
    files = FileCheckpointStore(tmp_path)
    files.save(execution_id, state)
    command = ["runs", "--checkpoints", str(tmp_path), "--json", "inspect", execution_id]

    assert main(command) == EXIT_OK
    value = json.loads(capsys.readouterr().out)["inspect"]
    assert value["format"] == {"checkpoint_version": 5, "requires": ["attempt_allowance"]}
    assert value["next_action"]["action"] == "resume"
    assert str(POLICY.to_dict()) in value["next_action"]["detail"]

    state["control"].pop("attempts_used")
    files.save(execution_id, state)
    assert main(command) == EXIT_LOCAL_ERROR
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["type"] == "RunInspectionError" and "attempt_allowance" in error["message"]


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_checkpoints_ignore_allowance_keys(version):
    state = build_state(execution_id="run1", status="running", model="m",
                        graph=graph(), registry=Registry(), task=None, max_budget_usd=None,
                        node_costs={})
    state["version"] = version
    state["control"].update(retry_policy=POLICY.to_dict(), attempts_used={})
    assert attempt_allowance_from_state(state) is None


# ---------------------------------------------------------------------------
# A real process kill
# ---------------------------------------------------------------------------


class Held(Flaky):
    """The child's provider: its first call fails, its second holds until the child is killed."""

    def __init__(self, folder) -> None:
        super().__init__()
        self.folder = folder

    async def complete(self, system, prompt, model):
        from fakes import write_barrier

        self.calls.append(prompt)
        if len(self.calls) == 1:
            raise ConnectionError("reset")
        write_barrier(self.folder / "held.json", {"calls": len(self.calls)})
        await asyncio.sleep(600)
        raise AssertionError("The child was not killed")


CHILD = r"""
import sys
from pathlib import Path

from smythe import FileCheckpointStore
from test_attempt_allowance import Held, build, graph

folder = Path(sys.argv[1])
build(FileCheckpointStore(folder / "checkpoints"), Held(folder)).execute(graph())
"""


def test_a_killed_run_resumes_within_its_allowance(tmp_path):
    process = launch_child(CHILD, tmp_path)
    try:
        assert wait_for_barrier(process, tmp_path / "held.json", timeout=60) == {"calls": 2}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0

    store = FileCheckpointStore(tmp_path / "checkpoints")
    execution_id, state = saved(store)
    # The second attempt was saved as spent before it was sent.
    assert state["status"] == "running" and used(state) == 2

    provider = Flaky()
    with pytest.raises(ConnectionError):
        build(store, provider).resume(execution_id)

    assert 2 + len(provider.calls) == ALLOWANCE
    assert used(saved(store)[1]) == ALLOWANCE
