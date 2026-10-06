"""A model profile's ``fallback`` routes a node's retry to another approved profile.

After an attempt fails with a ``transient``, ``provider_rejected`` or
``quality_defect`` failure, a node's next attempt calls the fallback of the
profile that failed: a new attempt with its own trace span, which records the
profile it called and the one it fell back from. The retry still needs the
node's failure policy, and any retry policy, to allow it, so attempts across
profiles share one allowance, persisted ones included. After an
``unknown_outcome``, whose call may still be charged, the retry stays on its
profile. A node timeout that ends a sent, unanswered call has that class only
under ``retain_uncertain_dispatch``; otherwise it is ``transient`` and its
retry falls back. A fallback attempt reserves its own profile's ceiling and
takes that profile's capacity turn and slot.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest

from fakes import RATE_LIMITED, Crash, HttpError, ManualClock
from smythe import (
    CapacityLimit, CapacityPolicy, ModelProfile, RetryPolicy, SentinelAlert, Swarm,
    UncertainDispatchError,
)
from smythe.async_executor import AsyncExecutor
from smythe.budget import Sentinel
from smythe.checkpoint import CheckpointStore
from smythe.executor import Executor
from smythe.graph import ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology
from smythe.provider import CompletionResult, Provider, ProviderFeatures
from smythe.registry import Registry
from smythe.scheduling import fallback_chain, profile_map
from smythe.tools import ToolLoopLimitError
from smythe.tracer import Tracer

MODES = ["serial", "parallel"]
SENDS_ONCE = ProviderFeatures(sdk_retries=0)
HANG = object()  # A call that never answers.
TIMEOUT = 0.02
CALL = 0.01
RATE_LIMIT_ERROR = type(RATE_LIMITED.exception())


class Script(Provider):
    """Answers each call with its name, or with the next scheduled outcome.

    An outcome is reply text, a ``CompletionResult``, an ``HttpError`` raised
    as its SDK status error, another exception to raise, or ``HANG``, a call
    that waits until it is cancelled. Once the outcomes run out, calls answer
    ``"<name> wrote <step>"``. ``calls`` lists each call's model and step,
    ``starts`` each call's start on *clock*, and ``events`` when each call
    started and ended. A call yields to the event loop *yields* times before
    its outcome. ``peak`` is the most calls to each model in flight at once,
    counted with *shared*'s calls when given. It sends each call once.
    """

    def __init__(self, name: str, *outcomes, features: ProviderFeatures = SENDS_ONCE,
                 clock: ManualClock | None = None, yields: int = 0,
                 shared: Script | None = None) -> None:
        self.name, self.outcomes, self.features = name, list(outcomes), features
        self.clock, self.yields = clock, yields
        self.calls: list[tuple[str, str]] = []
        self.starts: list[float] = []
        self.events: list[str] = []
        self.in_flight: Counter[str] = Counter() if shared is None else shared.in_flight
        self.peak: Counter[str] = Counter() if shared is None else shared.peak

    def feature_support(self, model):
        return self.features

    async def complete(self, system, prompt, model):
        step = next((line.removeprefix("Your step: ") for line in prompt.splitlines()
                     if line.startswith("Your step: ")), prompt.split("\n")[0])
        self.calls.append((model, step))
        self.starts.append(self.clock.monotonic() if self.clock else 0.0)
        self.events.append(f"start {step}")
        self.in_flight[model] += 1
        self.peak[model] = max(self.peak[model], self.in_flight[model])
        try:
            for _ in range(self.yields):
                await asyncio.sleep(0)
            outcome = self.outcomes.pop(0) if self.outcomes else f"{self.name} wrote {step}"
            if outcome is HANG:
                await asyncio.Event().wait()
            if isinstance(outcome, HttpError):
                raise outcome.exception()
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, CompletionResult):
                return outcome
            return CompletionResult(outcome, cost_usd=CALL)
        finally:
            self.in_flight[model] -= 1
            self.events.append(f"end {step}")


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


def ladder(fast: Provider, strong: Provider, **fast_options) -> list[ModelProfile]:
    """``fast`` at a 0.25 ceiling, which falls back to ``strong`` at 1.0."""
    return [ModelProfile("fast", fast, "fast-model", 0.25, fallback="strong", **fast_options),
            ModelProfile("strong", strong, "strong-model", 1.0)]


def polish(node_id: str = "polish", profile: str = "fast", **options) -> Node:
    options.setdefault("failure_policy", FailurePolicy.RETRY)
    options.setdefault("max_retries", 2)
    return Node(id=node_id, label=node_id.capitalize(), metadata={"profile": profile}, **options)


def make(mode: str, profiles, *, default: Provider | None = None, **options):
    cls = Executor if mode == "serial" else AsyncExecutor
    options.setdefault("tracer", Tracer())
    return cls(provider=default or Script("default"), registry=Registry(), artifact_dir=None,
               profiles=profiles, **options)


def run(executor, *nodes: Node) -> ExecutionGraph:
    graph = ExecutionGraph([Topology.SERIAL], list(nodes))
    if isinstance(executor, Executor):
        executor.run(graph)
    else:
        asyncio.run(executor.run(graph))
    return graph


def attempts(tracer: Tracer) -> list[tuple]:
    """Each attempt span's number, outcome, failure class, profile and the profile it left."""
    return [(span.attempt, span.outcome, span.metadata.get("failure_class"),
             span.metadata.get("profile"), span.metadata.get("fallback_from"))
            for span in tracer.spans]


def record_reservations(monkeypatch) -> list[tuple[str, float]]:
    reserved: list[tuple[str, float]] = []
    reserve = Sentinel.reserve

    def recording(self, node_id, estimated_cost, **options):
        reserved.append((node_id, estimated_cost))
        return reserve(self, node_id, estimated_cost, **options)

    monkeypatch.setattr(Sentinel, "reserve", recording)
    return reserved


# ---------------------------------------------------------------------------
# Declaring a fallback
# ---------------------------------------------------------------------------


def test_a_profile_names_its_fallback_and_the_chain_follows_it():
    fast, strong, best = Script("fast"), Script("strong"), Script("best")
    named = profile_map([
        ModelProfile("fast", fast, "fast-model", 0.25, fallback="strong"),
        ModelProfile("strong", strong, "strong-model", 1.0, fallback="best"),
        ModelProfile("best", best, "best-model", 2.0),
    ])
    assert named["best"].fallback is None
    assert [item.name for item in fallback_chain(named["fast"], named)] == ["fast", "strong", "best"]
    assert [item.name for item in fallback_chain(named["best"], named)] == ["best"]


@pytest.mark.parametrize("fallback", ["fast", "", "1strong", "has space", 3, ["strong"]])
def test_a_malformed_fallback_is_refused_when_the_profile_is_built(fallback):
    with pytest.raises(ValueError, match="fallback"):
        ModelProfile("fast", Script("fast"), "fast-model", 0.25, fallback=fallback)


@pytest.mark.parametrize("profiles,match", [
    pytest.param(lambda fast, strong: [ModelProfile("fast", fast, "fast-model", 0.25,
                                                    fallback="strong")],
                 "not one of", id="not-allowed"),
    pytest.param(lambda fast, strong: [
        ModelProfile("fast", fast, "fast-model", 0.25, fallback="strong"),
        ModelProfile("strong", strong, "strong-model", 1.0, fallback="fast"),
    ], "loop", id="loop"),
    pytest.param(lambda fast, strong: [
        ModelProfile("fast", fast, "fast-model", 0.25, fallback="strong",
                     requires={"exact_accounting"}),
        ModelProfile("strong", strong, "strong-model", 1.0),
    ], "exact_accounting", id="fewer-requirements"),
])
@pytest.mark.parametrize("where", ["swarm", "serial", "parallel"])
def test_a_fallback_that_cannot_serve_its_profiles_nodes_is_refused(profiles, match, where):
    fast, strong = Script("fast"), Script("strong")
    with pytest.raises(ValueError, match=match):
        if where == "swarm":
            Swarm(model="default-model", provider=Script("default"), profiles=profiles(fast, strong))
        else:
            make(where, profiles(fast, strong))
    assert fast.calls == [] and strong.calls == []


def test_a_fallback_may_require_more_than_the_profile_it_serves():
    features = ProviderFeatures(exact_accounting=True, sdk_retries=0)
    profiles = [ModelProfile("fast", Script("fast"), "fast-model", 0.25, fallback="strong"),
                ModelProfile("strong", Script("strong", features=features), "strong-model", 1.0,
                             requires={"exact_accounting"})]
    assert set(profile_map(profiles)) == {"fast", "strong"}


# ---------------------------------------------------------------------------
# Which failures fall back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", [None, RetryPolicy()], ids=["no-policy", "policy"])
def test_after_a_rate_limit_the_next_attempt_calls_the_fallback_profile(mode, policy):
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    tracer = Tracer()
    graph = run(make(mode, ladder(fast, strong), retry_policy=policy, tracer=tracer), polish())

    (node,) = graph.nodes
    assert node.status is NodeStatus.COMPLETED and node.result == "strong wrote Polish"
    assert fast.calls == [("fast-model", "Polish")]
    assert strong.calls == [("strong-model", "Polish")]
    # Attempt 1 is a new attempt on the next profile, and its span says so.
    assert attempts(tracer) == [
        (0, "failed", "transient", "fast", None),
        (1, "completed", None, "strong", "fast"),
    ]
    assert all("profile" not in entry and "fallback_from" not in entry
               for entry in tracer.summary())
    # The node still names its own profile.
    assert node.metadata == {"profile": "fast"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("outcome,failure_class,falls_back", [
    pytest.param(HttpError(503, "api_error"), "transient", True, id="server-error"),
    pytest.param(CompletionResult("cut", cost_usd=CALL, stop_reason="max_tokens"),
                 "provider_rejected", True, id="truncated"),
    pytest.param(ToolLoopLimitError("no answer within the tool loop"), "quality_defect", True,
                 id="tool-loop-limit"),
    pytest.param(HttpError(404, "not_found_error"), "input_error", False, id="not-found"),
    pytest.param(HttpError(401, "authentication_error"), "authority_denied", False,
                 id="unauthorized"),
])
def test_only_fallback_classes_fall_back_and_others_retry_on_the_same_profile(
    mode, outcome, failure_class, falls_back,
):
    fast, strong = Script("fast", outcome), Script("strong")
    tracer = Tracer()
    policy = RetryPolicy(retry_classes={failure_class})
    graph = run(make(mode, ladder(fast, strong), retry_policy=policy, tracer=tracer), polish())

    assert graph.nodes[0].status is NodeStatus.COMPLETED
    second = ("strong", "fast") if falls_back else ("fast", None)
    assert attempts(tracer) == [(0, "failed", failure_class, "fast", None),
                                (1, "completed", None, *second)]
    assert len(fast.calls) == (1 if falls_back else 2)
    assert len(strong.calls) == (1 if falls_back else 0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("listed", [True, False], ids=["unknown-outcome-listed", "default"])
def test_no_fallback_after_an_unknown_outcome(mode, listed):
    # The unanswered call may still be charged: a retry, if the policy makes
    # one, stays on its profile and is never sent to the fallback.
    fast, strong = Script("fast", HANG), Script("strong")
    tracer = Tracer()
    classes = {"transient", "unknown_outcome"} if listed else {"transient"}
    policy = RetryPolicy(retry_classes=classes, retain_uncertain_dispatch=True)
    budget = Sentinel(5.0)
    executor = make(mode, ladder(fast, strong), retry_policy=policy, tracer=tracer, budget=budget)
    node = polish(timeout_s=TIMEOUT)

    if listed:
        run(executor, node)
        assert node.status is NodeStatus.COMPLETED and node.result == "fast wrote Polish"
        assert attempts(tracer) == [(0, "timed_out", "unknown_outcome", "fast", None),
                                    (1, "completed", None, "fast", None)]
    else:
        with pytest.raises(UncertainDispatchError):
            run(executor, node)
        assert attempts(tracer) == [(0, "timed_out", "unknown_outcome", "fast", None)]
    assert strong.calls == []
    # The unanswered call's reservation is kept as its unknown exposure.
    assert budget.cost_is_complete is False and budget.breakdown()["polish"] >= 0.25


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", [None, RetryPolicy()], ids=["no-policy", "policy"])
def test_without_kept_exposure_a_timed_out_call_is_transient_and_its_retry_falls_back(
    mode, policy,
):
    # Without retain_uncertain_dispatch, a node timeout that ends a sent call
    # is an ordinary transient timeout, so the retry calls the fallback while
    # the first call may still run and be charged.
    fast, strong = Script("fast", HANG), Script("strong")
    tracer = Tracer()
    budget = Sentinel(5.0)
    executor = make(mode, ladder(fast, strong), retry_policy=policy, tracer=tracer, budget=budget)
    graph = run(executor, polish(timeout_s=TIMEOUT))

    (node,) = graph.nodes
    assert node.status is NodeStatus.COMPLETED and node.result == "strong wrote Polish"
    assert fast.calls == [("fast-model", "Polish")]
    assert strong.calls == [("strong-model", "Polish")]
    assert attempts(tracer) == [(0, "timed_out", "transient", "fast", None),
                                (1, "completed", None, "strong", "fast")]
    # The timed-out call's reservation was released, so only the fallback's
    # charge is recorded.
    assert budget.breakdown() == {"polish": CALL} and budget.cost_is_complete is True


@pytest.mark.parametrize("mode", MODES)
def test_a_node_that_may_not_retry_never_falls_back(mode):
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    with pytest.raises(RATE_LIMIT_ERROR):
        run(make(mode, ladder(fast, strong)), polish(failure_policy=FailurePolicy.HALT))
    assert len(fast.calls) == 1 and strong.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_a_chain_escalates_one_profile_per_attempt_and_stays_on_its_last(mode):
    fast = Script("fast", RATE_LIMITED)
    strong = Script("strong", RATE_LIMITED)
    best = Script("best", RATE_LIMITED)
    profiles = [ModelProfile("fast", fast, "fast-model", 0.25, fallback="strong"),
                ModelProfile("strong", strong, "strong-model", 1.0, fallback="best"),
                ModelProfile("best", best, "best-model", 2.0)]
    tracer = Tracer()
    run(make(mode, profiles, retry_policy=RetryPolicy(), tracer=tracer), polish(max_retries=3))

    assert attempts(tracer) == [
        (0, "failed", "transient", "fast", None),
        (1, "failed", "transient", "strong", "fast"),
        (2, "failed", "transient", "best", "strong"),
        (3, "completed", None, "best", None),
    ]
    assert [len(item.calls) for item in (fast, strong, best)] == [1, 1, 2]


# ---------------------------------------------------------------------------
# One allowance across profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_attempts_across_profiles_share_the_nodes_allowance(mode):
    fast = Script("fast", RATE_LIMITED, RATE_LIMITED)
    strong = Script("strong", RATE_LIMITED, RATE_LIMITED)
    tracer = Tracer()
    with pytest.raises(RATE_LIMIT_ERROR):
        run(make(mode, ladder(fast, strong), retry_policy=RetryPolicy(), tracer=tracer),
            polish(max_retries=2))
    # 1 + max_retries attempts in all, whichever profile each one called.
    assert [span.metadata["profile"] for span in tracer.spans] == ["fast", "strong", "strong"]
    assert len(fast.calls) + len(strong.calls) == 3


@pytest.mark.parametrize("mode", MODES)
def test_attempts_across_profiles_stay_within_the_persisted_allowance(mode):
    store = Store()
    policy = RetryPolicy(persist_attempts=True)
    graph = ExecutionGraph([Topology.SERIAL], [polish(max_retries=2)])
    fast = Script("fast", RATE_LIMITED)
    strong = Script("strong", Crash("process lost during the fallback's call"))
    with pytest.raises(Crash):
        Swarm(model="default-model", provider=Script("default"), profiles=ladder(fast, strong),
              checkpoint_store=store, retry_policy=policy, artifact_dir=None,
              parallel=mode == "parallel").execute(graph)
    (execution_id,) = store.list_ids()
    assert store.load(execution_id)["control"]["attempts_used"]["polish"]["attempts"] == 2

    # The resumed attempt is the node's first in this process, so it calls the
    # profile the node names. It is also its last: the allowance holds three.
    resumed_fast, resumed_strong = Script("fast", RATE_LIMITED), Script("strong")
    with pytest.raises(RATE_LIMIT_ERROR):
        Swarm(model="default-model", provider=Script("default"),
              profiles=ladder(resumed_fast, resumed_strong), checkpoint_store=store,
              retry_policy=policy, artifact_dir=None,
              parallel=mode == "parallel").resume(execution_id)

    calls = [*fast.calls, *strong.calls, *resumed_fast.calls, *resumed_strong.calls]
    assert calls == [("fast-model", "Polish"), ("strong-model", "Polish"), ("fast-model", "Polish")]
    assert store.load(execution_id)["control"]["attempts_used"]["polish"]["attempts"] == 3


@pytest.mark.parametrize("mode", MODES)
def test_a_swarm_run_falls_back_and_saves_the_profile_its_node_names(mode):
    store = Store()
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    graph = ExecutionGraph([Topology.SERIAL], [polish()])
    result = Swarm(model="default-model", provider=Script("default"),
                   profiles=ladder(fast, strong), checkpoint_store=store,
                   retry_policy=RetryPolicy(), artifact_dir=None,
                   parallel=mode == "parallel").execute(graph)

    assert result.output == "strong wrote Polish"
    node = result.graph.nodes[0]
    assert node.metadata["profile"] == "fast" and node.metadata["model"] == "fast-model"
    (execution_id,) = store.list_ids()
    saved = store.load(execution_id)
    assert saved["graph"]["nodes"][0]["metadata"]["model"] == "fast-model"
    assert saved["requires"] == ["model_profiles"]


# ---------------------------------------------------------------------------
# Checks before any call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["swarm", "serial", "parallel"])
def test_a_retry_policy_refuses_a_fallback_whose_sdk_may_retry_before_any_call(where):
    fast, strong = Script("fast"), Script("strong", features=ProviderFeatures())
    graph = ExecutionGraph([Topology.SERIAL], [polish()])
    with pytest.raises(ValueError, match="sdk_retries"):
        if where == "swarm":
            Swarm(model="default-model", provider=Script("default"),
                  profiles=ladder(fast, strong), retry_policy=RetryPolicy(),
                  artifact_dir=None).execute(graph)
        else:
            run(make(where, ladder(fast, strong), retry_policy=RetryPolicy()), *graph.nodes)
    assert fast.calls == [] and strong.calls == []


def test_a_fallback_that_cannot_take_a_nodes_image_inputs_is_refused_before_any_call():
    fast = Script("fast", features=ProviderFeatures(attachments=True, sdk_retries=0))
    strong = Script("strong", features=ProviderFeatures(attachments=False, sdk_retries=0))
    graph = ExecutionGraph([Topology.SERIAL], [polish(attach_dep_artifacts=True)])
    with pytest.raises(ValueError, match="image"):
        Swarm(model="default-model", provider=Script("default"), profiles=ladder(fast, strong),
              artifact_dir=None).execute(graph)
    assert fast.calls == [] and strong.calls == []


@pytest.mark.parametrize("mode", MODES)
def test_a_fallbacks_capacity_hint_is_read_before_any_call(mode):
    class BadHint(Script):
        def capacity_hint(self, model):
            return {"max_in_flight": 0}

    fast, strong = Script("fast"), BadHint("strong")
    executor = make(mode, ladder(fast, strong), capacity_policy=CapacityPolicy())
    with pytest.raises(ValueError, match="capacity_hint"):
        run(executor, polish())
    assert fast.calls == [] and strong.calls == []


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_fallback_attempt_reserves_its_own_profiles_ceiling(monkeypatch, mode):
    reserved = record_reservations(monkeypatch)
    budget = Sentinel(5.0)
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    run(make(mode, ladder(fast, strong), budget=budget), polish())

    assert reserved == [("polish", 0.25), ("polish", 1.0)]
    assert budget.breakdown() == {"polish": CALL} and budget._reservations == {}


@pytest.mark.parametrize("mode", MODES)
def test_a_fallback_after_a_billed_failure_reserves_its_own_ceiling(monkeypatch, mode):
    reserved = record_reservations(monkeypatch)
    budget = Sentinel(5.0)
    truncated = CompletionResult("cut", cost_usd=CALL, stop_reason="max_tokens")
    fast, strong = Script("fast", truncated), Script("strong")
    run(make(mode, ladder(fast, strong), budget=budget), polish())

    assert reserved == [("polish", 0.25), ("polish", 1.0)]
    # The truncated reply's charge stays, beside the fallback's.
    assert budget.breakdown() == {"polish": 2 * CALL} and budget._reservations == {}


@pytest.mark.parametrize("mode", MODES)
def test_a_budget_that_cannot_hold_the_fallbacks_ceiling_stops_before_its_call(mode):
    budget = Sentinel(0.5)
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    node = polish()
    with pytest.raises(SentinelAlert):
        run(make(mode, ladder(fast, strong), budget=budget), node)

    assert len(fast.calls) == 1 and strong.calls == []
    assert node.status is NodeStatus.FAILED
    assert budget._reservations == {} and budget.total_cost_usd == 0


@pytest.mark.parametrize("mode", MODES)
def test_a_regenerated_node_starts_again_on_the_profile_it_names(monkeypatch, mode):
    reserved = record_reservations(monkeypatch)
    budget = Sentinel(5.0)
    fast, strong = Script("fast", RATE_LIMITED), Script("strong")
    judge = Script("judge", "FAIL: not polished enough", "PASS")
    profiles = [*ladder(fast, strong), ModelProfile("judge", judge, "judge-model", 0.5)]
    gate = Node(id="judge", label="Judge", depends_on=["polish"], verifies="polish",
                max_regenerations=1, metadata={"profile": "judge"})
    graph = run(make(mode, profiles, budget=budget), polish(), gate)

    assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 2
    assert fast.calls == [("fast-model", "Polish")] * 2
    assert strong.calls == [("strong-model", "Polish")]
    # The fallback ended with its attempts: the regenerated node reserves and
    # calls the profile it names.
    assert [amount for node_id, amount in reserved if node_id == "polish"] == [0.25, 1.0, 0.25]


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_fallback_attempt_takes_its_profiles_capacity_turn(mode):
    clock = ManualClock()
    tracer = Tracer(clock=clock.monotonic, wall_clock=lambda: clock.time_ns() / 1e9)
    fast = Script("fast", RATE_LIMITED, clock=clock)
    strong = Script("strong", clock=clock)
    policy = CapacityPolicy(limits={"strong-model": CapacityLimit(min_interval_s=5)})
    first = Node(id="first", label="First", metadata={"profile": "strong"})
    second = polish(depends_on=["first"])
    run(make(mode, ladder(fast, strong), capacity_policy=policy, sleep=clock.advance,
             tracer=tracer), first, second)

    assert strong.calls == [("strong-model", "First"), ("strong-model", "Polish")]
    # The fallback's call waited out the strong model's interval; the fast
    # model has none.
    assert fast.starts == [0.0] and strong.starts == [0.0, 5.0]
    assert tracer.timing()["waiting_ms"]["capacity"] == 5000


def test_a_fallback_attempt_moves_its_slot_to_the_fallback_model():
    fast = Script("fast", RATE_LIMITED)
    strong = Script("strong", yields=20)
    policy = CapacityPolicy(limits={"strong-model": CapacityLimit(max_in_flight=1)})
    executor = make("parallel", ladder(fast, strong), capacity_policy=policy, max_concurrency=4)
    graph = run(executor, Node(id="first", label="First", metadata={"profile": "strong"}),
                polish())

    assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 2
    # Polish was admitted on the fast model; its fallback waited for the strong
    # model's only slot instead of running beside First.
    assert strong.peak["strong-model"] == 1
    assert strong.events == ["start First", "end First", "start Polish", "end Polish"]
    assert [executor._capacity.bucket(item, model).in_flight
            for item, model in ((fast, "fast-model"), (strong, "strong-model"))] == [0, 0]
    assert executor._capacity.slots == {}


def test_a_node_waiting_to_move_its_slot_holds_none_so_two_fallbacks_cannot_block_each_other():
    # Each node's fallback calls the model the other node holds the only slot of.
    left = Script("left", RATE_LIMITED, yields=5)
    right = Script("right", RATE_LIMITED, yields=5, shared=left)
    profiles = [ModelProfile("left", left, "left-model", 0.25, fallback="right-on-left"),
                ModelProfile("right-on-left", left, "right-model", 0.25),
                ModelProfile("right", right, "right-model", 0.25, fallback="left-on-right"),
                ModelProfile("left-on-right", right, "left-model", 0.25)]
    policy = CapacityPolicy(limits={"left-model": CapacityLimit(max_in_flight=1),
                                    "right-model": CapacityLimit(max_in_flight=1)})
    executor = make("parallel", profiles, capacity_policy=policy, max_concurrency=4)
    graph = ExecutionGraph([Topology.SERIAL], [polish("a", "left"), polish("b", "right")])

    asyncio.run(asyncio.wait_for(executor.run(graph), 10))

    assert [node.status for node in graph.nodes] == [NodeStatus.COMPLETED] * 2
    assert left.calls == [("left-model", "A"), ("right-model", "A")]
    assert right.calls == [("right-model", "B"), ("left-model", "B")]
    assert left.peak == Counter({"left-model": 1, "right-model": 1})
