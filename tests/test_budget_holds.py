"""Allowance held for verification and synthesis cannot be spent by other nodes.

``Swarm(reserve_for_checks=True)`` holds, before any node runs, the estimate
that each pending verifier and an ``LLM_MERGE`` synthesis will reserve. A hold
counts toward every other node's admission but is never spend, and its owner
converts it into its own reservation. Every call here charges exactly what it
estimated, so each admission is exact, and every ledger a ``Swarm`` builds is
recorded so a test can read the run's holds and reservations.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

import smythe.swarm as swarm_module
from smythe.async_executor import AsyncExecutor
from smythe.budget import BudgetValidationError, Sentinel, SentinelAlert
from smythe.checkpoint import CheckpointStore
from smythe.graph import (
    SYNTHESIS_NODE_ID, ExecutionGraph, FailurePolicy, Node, NodeStatus, Revision, Topology,
)
from smythe.provider import CompletionResult, OfflineProvider, Provider
from smythe.registry import Registry
from smythe.supervisor import Supervisor
from smythe.swarm import Swarm
from smythe.synthesizer import MERGE_SYSTEM_PROMPT, SynthesisStrategy, Synthesizer
from smythe.task import Task
from smythe.tracer import Tracer
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import SQLiteWorkflowStore

MODES = ["serial", "parallel"]
PRICE = 0.10
MERGE = SynthesisStrategy.LLM_MERGE
DELIVERABLE = SynthesisStrategy.DELIVERABLE


class Priced(Provider):
    """Estimates and charges PRICE for every call; fails the steps named in *fail*.

    A call is named by its node's label, or ``synthesis`` for a merge, and
    ``judge`` replies with each of *verdicts* in turn.
    """

    def __init__(self, *verdicts: str, fail: tuple[str, ...] = (), estimate: bool = True) -> None:
        self.verdicts, self.fail, self.estimate = list(verdicts), set(fail), estimate
        self.calls: list[str] = []

    def budget_estimate_usd(self, model):
        return PRICE if self.estimate else None

    async def complete(self, system, prompt, model):
        name = "synthesis" if system == MERGE_SYSTEM_PROMPT else prompt.split("\n")[0]
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"boom: {name}")
        text = self.verdicts.pop(0) if name == "judge" else f"{name} done"
        if self.estimate:
            return CompletionResult(text=text, cost_usd=PRICE)
        return CompletionResult(text=text, prompt_tokens=100, completion_tokens=50)


class MemoryStore(CheckpointStore):
    """Keeps every saved state as JSON data, like a file store would."""

    def __init__(self) -> None:
        self.states: dict[str, dict] = {}
        self.saves: list[dict] = []

    def save(self, execution_id, state):
        self.states[execution_id] = json.loads(json.dumps(state))
        self.saves.append(self.states[execution_id])

    def load(self, execution_id):
        state = self.states.get(execution_id)
        return None if state is None else json.loads(json.dumps(state))

    def delete(self, execution_id):
        self.states.pop(execution_id, None)

    def list_ids(self):
        return sorted(self.states)


@pytest.fixture
def ledgers(monkeypatch):
    """Every Sentinel a Swarm builds from here on, recording its holds and reservations."""
    created: list[Sentinel] = []

    class Recording(Sentinel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.events: list[tuple] = []
            created.append(self)

        def hold(self, key, amount):
            super().hold(key, amount)
            self.events.append(("hold", key, amount))

        def reserve(self, node_id, estimated_cost, *, hard_ceiling=False, from_hold=False):
            super().reserve(node_id, estimated_cost, hard_ceiling=hard_ceiling, from_hold=from_hold)
            self.events.append(("reserve", node_id, from_hold))
            # No admission may take allowance another key still holds.
            if self.max_budget_usd is not None:
                assert self.total_cost_usd + sum(self.holds.values()) <= self.max_budget_usd + 1e-9

    monkeypatch.setattr(swarm_module, "Sentinel", Recording)
    return created


def run_ledger(created: list[Sentinel]) -> Sentinel:
    """The ledger of the last run: the last one the Swarm asked to hold or reserve."""
    return [ledger for ledger in created if ledger.events][-1]


def build(provider, *, budget, holds, mode="parallel", synthesis=MERGE, **kwargs) -> Swarm:
    return Swarm(model="test-model", provider=provider, max_budget_usd=budget,
                 synthesizer=Synthesizer(synthesis), parallel=mode == "parallel",
                 reserve_for_checks=holds, artifact_dir=None, **kwargs)


def fan_out(width: int) -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.BROADCAST_REDUCE],
                          nodes=[Node(id=f"w{i}", label=f"w{i}") for i in range(width)])


def gated(*, policy: FailurePolicy = FailurePolicy.HALT, regenerations: int = 1,
          after: bool = False) -> ExecutionGraph:
    nodes = [
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=regenerations, failure_policy=policy),
    ]
    if after:
        nodes.append(Node(id="after", label="after", depends_on=["judge"]))
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=nodes)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_holds_count_toward_admission_but_are_never_spend():
    ledger = Sentinel(max_budget_usd=1.0)
    ledger.hold("judge", 0.3)
    ledger.hold(SYNTHESIS_NODE_ID, 0.2)
    assert ledger.holds == {"judge": 0.3, SYNTHESIS_NODE_ID: 0.2}
    assert ledger.total_cost_usd == 0.0 and ledger.breakdown() == {}

    ledger.reserve("w0", 0.5)  # Exactly what the holds leave.
    with pytest.raises(SentinelAlert) as refused:
        ledger.reserve("w1", 0.01)
    assert refused.value.node_id == "w1"
    assert ledger.total_cost_usd == pytest.approx(0.5)
    ledger.record("w0", CompletionResult(text="w0", cost_usd=0.5))

    # Each owner's own hold makes room for its reservation.
    ledger.reserve("judge", 0.3, from_hold=True)
    assert ledger.holds == {SYNTHESIS_NODE_ID: 0.2}
    ledger.record("judge", CompletionResult(text="PASS", cost_usd=0.3))
    ledger.reserve(SYNTHESIS_NODE_ID, 0.2, from_hold=True)
    ledger.record(SYNTHESIS_NODE_ID, CompletionResult(text="merged", cost_usd=0.2))
    assert ledger.holds == {}
    assert ledger.total_cost_usd == pytest.approx(1.0)
    assert ledger.breakdown() == pytest.approx({"w0": 0.5, "judge": 0.3, SYNTHESIS_NODE_ID: 0.2})


def test_a_serial_check_leaves_held_allowance_alone():
    ledger = Sentinel(max_budget_usd=1.0)
    ledger.record("w0", CompletionResult(text="w0", cost_usd=0.7))
    ledger.check("w1")
    ledger.hold("judge", 0.3)
    with pytest.raises(SentinelAlert) as refused:
        ledger.check("w1")
    assert refused.value.node_id == "w1"
    ledger.release_hold("judge")
    ledger.check("w1")
    assert ledger.holds == {} and ledger.total_cost_usd == pytest.approx(0.7)


def test_holds_are_validated_and_refused_without_changing_the_ledger():
    ledger = Sentinel(max_budget_usd=1.0)
    ledger.hold("judge", 0.4)
    with pytest.raises(ValueError, match="already has a budget hold"):
        ledger.hold("judge", 0.1)
    with pytest.raises(SentinelAlert) as refused:
        ledger.hold(SYNTHESIS_NODE_ID, 0.7)
    assert refused.value.node_id == SYNTHESIS_NODE_ID
    for amount in (-0.1, math.nan, math.inf, True, "0.1"):
        with pytest.raises(BudgetValidationError):
            ledger.hold("other", amount)
    assert ledger.holds == {"judge": 0.4}

    ledger.reserve("w0", 0.2)
    with pytest.raises(ValueError, match="already has a budget reservation"):
        ledger.hold("w0", 0.1)
    with pytest.raises(ValueError, match="from_hold=True"):
        ledger.reserve("judge", 0.4)
    with pytest.raises(ValueError, match="has no budget hold"):
        ledger.reserve("w1", 0.1, from_hold=True)
    # A conversion that does not fit keeps its hold.
    with pytest.raises(SentinelAlert):
        ledger.reserve("judge", 0.9, from_hold=True)
    assert ledger.holds == {"judge": 0.4} and ledger.total_cost_usd == pytest.approx(0.2)

    ledger.release_hold("judge")
    ledger.release_hold("never-held")
    assert ledger.holds == {}
    # An unlimited ledger holds without refusing.
    Sentinel().hold("judge", 1e9)


def test_restore_keeps_holds_out_of_the_restored_total():
    ledger = Sentinel(max_budget_usd=1.0)
    ledger.hold("judge", 0.25)
    ledger.restore({"w0": 0.5})
    assert ledger.total_cost_usd == pytest.approx(0.5)
    assert ledger.holds == {"judge": 0.25}


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_without_holds_a_wide_fan_out_spends_the_synthesis_allowance(mode, ledgers):
    provider = Priced()
    with pytest.raises(SentinelAlert) as refused:
        build(provider, budget=1.0, holds=False, mode=mode).execute(fan_out(10))

    assert refused.value.node_id == SYNTHESIS_NODE_ID
    assert provider.calls == [f"w{i}" for i in range(10)]
    ledger = run_ledger(ledgers)
    assert ledger.total_cost_usd == pytest.approx(1.0)
    assert not any(event[0] == "hold" or event[-1] is True for event in ledger.events)


@pytest.mark.parametrize("mode", MODES)
def test_with_holds_a_wide_fan_out_cannot_starve_llm_merge_synthesis(mode, ledgers):
    provider = Priced()
    with pytest.raises(SentinelAlert) as refused:
        build(provider, budget=1.0, holds=True, mode=mode).execute(fan_out(10))

    # The first worker whose reservation would reach into the synthesis
    # hold is refused before its call, so synthesis is never the step that
    # finds its allowance spent.
    assert refused.value.node_id in {f"w{i}" for i in range(10)}
    assert "synthesis" not in provider.calls and len(provider.calls) <= 9
    ledger = run_ledger(ledgers)
    assert ledger.events[0] == ("hold", SYNTHESIS_NODE_ID, PRICE)
    assert ledger.total_cost_usd <= 1.0 - PRICE + 1e-9
    assert ledger.holds == {}

    # A budget that covers every worker and the hold completes, and the
    # merge reserves its own hold.
    provider = Priced()
    result = build(provider, budget=1.1, holds=True, mode=mode).execute(fan_out(10))
    assert provider.calls[-1] == "synthesis" and result.output == "synthesis done"
    assert ("reserve", SYNTHESIS_NODE_ID, True) in run_ledger(ledgers).events
    assert result.total_cost_usd == pytest.approx(1.1)


@pytest.mark.parametrize("mode", MODES)
def test_holds_are_never_reported_as_spend(mode, ledgers):
    totals = []
    for holds in (False, True):
        store = MemoryStore()
        swarm = build(Priced("PASS"), budget=5.0, holds=holds, mode=mode, checkpoint_store=store)
        graph = gated()
        graph.nodes.append(Node(id="extra", label="extra"))
        result = swarm.execute(graph)
        ledger = run_ledger(ledgers)
        held = [event for event in ledger.events if event[0] == "hold"]
        assert held == ([("hold", "judge", PRICE), ("hold", SYNTHESIS_NODE_ID, PRICE)]
                        if holds else [])
        assert ledger.holds == {}
        # No checkpoint records a hold or counts one in its costs.
        assert all(set(state["budget"]) == {"max_budget_usd", "node_costs", "token_estimate_nodes"}
                   and state["version"] == 4 for state in store.saves)
        [state] = store.states.values()
        totals.append((result.total_cost_usd, result.cost_is_complete,
                       result.cost_contains_estimates, state["budget"]))
    assert totals[0] == totals[1]
    # draft, judge, extra and the merge, each at PRICE.
    assert totals[1][0] == pytest.approx(4 * PRICE)


def test_synthesis_converts_its_hold_even_without_a_budget_estimate(ledgers):
    """A merge priced by tokens holds and reserves the default token estimate."""
    provider = Priced(estimate=False)
    swarm = build(provider, budget=1.0, holds=True, mode="serial")
    result = swarm.execute(fan_out(2))

    ledger = run_ledger(ledgers)
    amount = 2000 * ledger.cost_per_token
    assert ledger.events[0] == ("hold", SYNTHESIS_NODE_ID, amount)
    assert ("reserve", SYNTHESIS_NODE_ID, True) in ledger.events
    assert result.total_cost_usd == pytest.approx(3 * 150 * ledger.cost_per_token)


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_regenerated_verifier_converts_a_hold_placed_again(mode, ledgers):
    provider = Priced("FAIL: the draft misses a section", "PASS")
    result = build(provider, budget=0.45, holds=True, mode=mode,
                   synthesis=DELIVERABLE).execute(gated())

    assert provider.calls == ["draft", "judge", "draft", "judge"]
    assert result.output == "draft done"
    ledger = run_ledger(ledgers)
    assert ledger.events == [
        ("hold", "judge", PRICE),
        ("reserve", "draft", False), ("reserve", "judge", True),
        # The regeneration holds the rerun's allowance before the draft runs again.
        ("hold", "judge", PRICE),
        ("reserve", "draft", False), ("reserve", "judge", True),
    ]
    assert ledger.holds == {}
    assert result.total_cost_usd == pytest.approx(4 * PRICE)


@pytest.mark.parametrize("mode", MODES)
def test_a_regeneration_rerun_cannot_spend_its_verifiers_allowance(mode, ledgers):
    # After the first round $0.20 is spent: $0.15 is left, enough for the
    # draft's rerun or the judge's, not both.
    provider = Priced("FAIL: the draft misses a section", "PASS")
    with pytest.raises(SentinelAlert) as refused:
        build(provider, budget=0.35, holds=False, mode=mode,
              synthesis=DELIVERABLE).execute(gated())
    # Without holds the draft's rerun is paid for and its check is refused.
    assert refused.value.node_id == "judge"
    assert provider.calls == ["draft", "judge", "draft"]

    provider = Priced("FAIL: the draft misses a section", "PASS")
    with pytest.raises(SentinelAlert) as refused:
        build(provider, budget=0.35, holds=True, mode=mode,
              synthesis=DELIVERABLE).execute(gated())
    # With holds the draft's rerun is refused before its call.
    assert refused.value.node_id == "draft"
    assert provider.calls == ["draft", "judge"]
    assert run_ledger(ledgers).total_cost_usd == pytest.approx(2 * PRICE)


@pytest.mark.parametrize("mode", MODES)
def test_a_regeneration_that_cannot_hold_its_verifier_stops_before_the_rerun(mode, ledgers):
    # After the first round $0.05 is left, less than the judge's rerun needs.
    store = MemoryStore()
    provider = Priced("FAIL: the draft misses a section", "PASS")
    with pytest.raises(SentinelAlert) as refused:
        build(provider, budget=0.25, holds=True, mode=mode, synthesis=DELIVERABLE,
              checkpoint_store=store).execute(gated())

    assert refused.value.node_id == "judge"
    assert provider.calls == ["draft", "judge"]
    # The regeneration itself was applied and saved before the hold was refused.
    [state] = store.states.values()
    assert state["status"] == "failed"
    nodes = {node["id"]: node for node in state["graph"]["nodes"]}
    assert all(node["status"] == "pending" and node["metadata"]["execution_generation"] == 1
               and "regeneration_intent" not in node["metadata"] for node in nodes.values())
    assert run_ledger(ledgers).holds == {}


@pytest.mark.parametrize("mode", MODES)
def test_a_verifier_holds_its_own_estimate(mode, ledgers):
    graph = gated(regenerations=0)
    graph.nodes[1].metadata["estimated_cost_usd"] = 0.25
    provider = Priced("PASS")
    build(provider, budget=1.0, holds=True, mode=mode, synthesis=DELIVERABLE).execute(graph)
    ledger = run_ledger(ledgers)
    assert ledger.events[0] == ("hold", "judge", 0.25)
    assert ("reserve", "judge", True) in ledger.events


def test_an_executor_holds_its_own_token_estimate_for_a_verifier():
    ledger = Sentinel(max_budget_usd=1.0)
    executor = AsyncExecutor(provider=Priced("PASS", estimate=False), registry=Registry(),
                             tracer=Tracer(), budget=ledger, estimated_tokens_per_node=500,
                             artifact_dir=None)
    graph = gated(regenerations=0)
    executor.hold_verifier_budget(graph)
    assert ledger.holds == {"judge": 500 * ledger.cost_per_token}
    asyncio.run(executor.run(graph))
    assert ledger.holds == {} and graph.nodes[1].status is NodeStatus.COMPLETED


@pytest.mark.parametrize("mode", MODES)
def test_a_verifier_a_revision_adds_reserves_when_admitted(mode, ledgers):
    class AddJudge(Supervisor):
        async def review(self, graph, node, *, task, revisions_remaining):
            judge = Node(id="judge", label="judge", depends_on=["draft"], verifies="draft")
            return Revision(add_nodes=(judge,), reason="check the draft")

    provider = Priced("PASS")
    result = build(provider, budget=1.0, holds=True, mode=mode, synthesis=DELIVERABLE,
                   supervisor=AddJudge(), max_revisions=1).execute(
        ExecutionGraph(topology=[Topology.SERIAL], nodes=[Node(id="draft", label="draft")]))

    assert provider.calls == ["draft", "judge"]
    assert run_ledger(ledgers).events == [("reserve", "draft", False), ("reserve", "judge", False)]
    assert result.total_cost_usd == pytest.approx(2 * PRICE)


def test_a_serial_verifier_without_an_estimate_reserves_what_it_held(ledgers):
    """A serial node without an estimate is only checked; a held verifier reserves its hold."""
    provider = Priced("PASS", estimate=False)
    build(provider, budget=1.0, holds=True, mode="serial",
          synthesis=DELIVERABLE).execute(gated(regenerations=0))
    ledger = run_ledger(ledgers)
    amount = 2000 * ledger.cost_per_token
    assert ledger.events == [("hold", "judge", amount), ("reserve", "judge", True)]
    assert ledger.total_cost_usd == pytest.approx(2 * 150 * ledger.cost_per_token)


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_holds_are_released_when_a_run_fails(mode, ledgers):
    provider = Priced(fail=("w1",))
    graph = fan_out(3)
    graph.nodes.append(Node(id="judge", label="judge", depends_on=["w0"], verifies="w0"))
    with pytest.raises(RuntimeError, match="boom: w1"):
        build(provider, budget=1.0, holds=True, mode=mode).execute(graph)
    ledger = run_ledger(ledgers)
    assert {event[1] for event in ledger.events if event[0] == "hold"} == {
        "judge", SYNTHESIS_NODE_ID}
    assert ledger.holds == {}
    assert "synthesis" not in provider.calls


@pytest.mark.parametrize("mode", MODES)
def test_a_failed_verifier_releases_its_allowance_to_later_work(mode, ledgers):
    # $0.25 covers draft and after only if the skipped judge's allowance comes back.
    provider = Priced(fail=("judge",))
    result = build(provider, budget=0.25, holds=True, mode=mode, synthesis=DELIVERABLE).execute(
        gated(policy=FailurePolicy.SKIP, regenerations=0, after=True))

    judge, after = result.graph.nodes[1], result.graph.nodes[2]
    assert judge.status is NodeStatus.SKIPPED and after.status is NodeStatus.COMPLETED
    assert provider.calls == ["draft", "judge", "after"]
    assert run_ledger(ledgers).holds == {}
    assert result.total_cost_usd == pytest.approx(2 * PRICE)


@pytest.mark.parametrize("mode", MODES)
def test_a_failed_merge_releases_its_allowance(mode, ledgers):
    provider = Priced(fail=("synthesis",))
    with pytest.raises(RuntimeError, match="boom: synthesis"):
        build(provider, budget=1.0, holds=True, mode=mode).execute(fan_out(2))
    ledger = run_ledger(ledgers)
    assert ("reserve", SYNTHESIS_NODE_ID, True) in ledger.events
    assert ledger.holds == {} and ledger.total_cost_usd == pytest.approx(2 * PRICE)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_resume_holds_again_from_configuration(ledgers):
    store = MemoryStore()
    graph = gated(regenerations=0)
    graph.nodes.insert(0, Node(id="w0", label="w0"))
    with pytest.raises(RuntimeError, match="boom: draft"):
        build(Priced(fail=("draft",)), budget=1.0, holds=True, mode="serial",
              checkpoint_store=store).execute(graph)
    [execution_id] = store.list_ids()

    provider = Priced("PASS")
    result = build(provider, budget=1.0, holds=True, mode="serial",
                   checkpoint_store=store).resume(execution_id)
    assert provider.calls == ["draft", "judge", "synthesis"]
    ledger = run_ledger(ledgers)
    assert ledger.events[:2] == [("hold", "judge", PRICE), ("hold", SYNTHESIS_NODE_ID, PRICE)]
    assert ledger.holds == {}
    # w0 was paid before the failure; the resumed run adds three calls.
    assert result.total_cost_usd == pytest.approx(4 * PRICE)


def test_holds_need_a_budget_limit(ledgers):
    provider = Priced("PASS")
    build(provider, budget=None, holds=True, mode="parallel").execute(gated(regenerations=0))
    assert not any(event[0] == "hold" for ledger in ledgers for event in ledger.events)
    assert provider.calls == ["draft", "judge", "synthesis"]


def test_reserve_for_checks_must_be_a_bool():
    with pytest.raises(TypeError, match="reserve_for_checks"):
        Swarm(model="offline", provider=OfflineProvider(), reserve_for_checks="yes")


def test_durable_runs_refuse_reserve_for_checks_by_name_before_any_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(OfflineProvider, "complete",
                        lambda self, *args, **kwargs: calls.append(args))
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        swarm = Swarm(model="offline", provider=OfflineProvider(), run_store=store,
                      max_budget_usd=1, reserve_for_checks=True)
        for entry in (swarm.plan, swarm.execute):
            with pytest.raises(WorkflowBindingError) as caught:
                entry(Task("Report"))
            assert str(caught.value).startswith("reserve_for_checks "), caught.value
        assert store.list_runs() == []
    assert calls == []
