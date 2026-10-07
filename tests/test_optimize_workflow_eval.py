"""Autotune trials that each run one durable text workflow, with exact cost.

Every test is offline: durable runs use ``OfflineProvider`` or the Responses
wire fake, and the process-loss case kills a real child process.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
import sqlite3
import time

import pytest

from fakes import (
    LOST_RESPONSE, HttpError, NativeTransport, kill_and_reap, launch_child, wait_for_barrier,
)
from smythe import (
    AcceptancePolicy, OfflineProvider, RunControl, SecurityStop, SQLiteWorkflowStore, Swarm, Task,
)
from smythe._money import MICROUSD, NANOUSD, from_units
from smythe.graph import NodeStatus
from smythe.optimize.contracts import (
    Candidate, ExperimentContract, MetricObjective, MutableFieldRule, ObjectiveDirection,
)
from smythe.optimize.engine import (
    OptimizationNeedsAttention, OptimizationRunner, TrialContext, TrialOutcome,
)
from smythe.optimize.ledger import ExperimentLedger, TrialStatus
from smythe.optimize.workflow_eval import (
    WorkflowTrialError, trial_run_id, workflow_trial_evaluator,
)
from smythe.provider_responses import OpenAIResponsesProvider
import smythe.optimize.ledger as ledger_module

EVALUATOR = "sha256:" + "e" * 64
CEILING = 20_000  # micro-USD per trial: 20,000,000 nanoUSD of workflow budget
ONE_NODE = {"nodes": [{"id": "draft", "label": "Draft"}]}
# The fixed texts the runner stores for a trial the adapter cannot report.
REFUSED = "trial workflow was refused before it started"
EXISTS = "trial already has a workflow, which is never run again"
FAILED = "trial workflow failed with all of its spend confirmed"
UNKNOWN = "trial workflow left reserved or unknown spend"
MEASURE = "trial measurement failed"


def _contract(*, ceiling: int = CEILING) -> ExperimentContract:
    return ExperimentContract(
        name="workflow_trials",
        objectives=(MetricObjective("quality", ObjectiveDirection.MAXIMIZE, primary=True),),
        mutable_fields=("plan.nodes",),
        mutable_field_rules={"plan.nodes": MutableFieldRule("integer", minimum=1, maximum=3)},
        required_gates=("delivered",), development_repetitions=1, confirmation_repetitions=3,
        holdout_repetitions=3, max_candidates=2, max_parallel_candidates=1, max_trials=14,
        max_wall_seconds=300, max_budget_microusd=14 * ceiling,
        per_trial_reservation_microusd=ceiling, confidence=0.8, min_improvement=0.5, base_seed=7,
    )


def _candidates(contract):
    incumbent = Candidate(contract=contract, policy={"plan.nodes": 1}, hypothesis="baseline")
    challenger = Candidate(contract=contract, policy={"plan.nodes": 2}, hypothesis="two steps",
                           parent=incumbent)
    return incumbent, challenger


def task_for(context):
    return Task(f"Write report {context.split} {context.seed}")


def measure(context, result):
    done = sum(node.status is NodeStatus.COMPLETED for node in result.graph.nodes)
    return {"quality": done}, {"delivered": bool(result.output)}


def offline_factory(store, calls=None):
    """Build each trial's durable Swarm: one plan step per ``plan.nodes``."""

    def build(context, max_budget_usd):
        if calls is not None:
            calls.append(context.trial_key)
        steps = context.candidate.policy["plan.nodes"]
        plan = {"nodes": [{"id": f"step{index}", "label": f"Step {index}"}
                          for index in range(steps)]}
        return Swarm(model="offline", provider=OfflineProvider(plan=plan), run_store=store,
                     max_budget_usd=max_budget_usd)

    return build


def native_factory(store, calls=None):
    """Build each trial's durable Swarm on the Responses wire fake."""

    def build(context, max_budget_usd):
        if calls is not None:
            calls.append(context.trial_key)
        provider = OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)
        return Swarm(model="gpt-6-astra", provider=provider, run_store=store,
                     max_budget_usd=max_budget_usd)

    return build


def _context(trial_key="trial_v1_" + "0" * 64, *, ceiling=CEILING, nodes=1):
    contract = _contract()
    candidate = Candidate(contract=contract, policy={"plan.nodes": nodes}, hypothesis="direct")
    return TrialContext(
        campaign_id="campaign", optimization_plan_hash="sha256:" + "a" * 64,
        trial_key=trial_key, candidate=candidate, split="development", phase="direct", seed=1,
        evaluator_hash=EVALUATOR, ceiling_microusd=ceiling,
        deadline_monotonic=time.monotonic() + 60,
    )


def _events(path):
    with closing(sqlite3.connect(path)) as db:
        return [row[0] for row in db.execute("SELECT payload_json FROM trial_events")]


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


def test_offline_durable_trials_complete_at_zero_cost(tmp_path):
    # The smallest campaign that can promote: one development trial for each
    # policy, then the contract's minimum of three paired confirmation and
    # three paired holdout seeds. The ledger's durability is not under test.
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db", durability="normal") as ledger):
        evaluate = workflow_trial_evaluator(offline_factory(store), task_for, measure)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR)
        result = asyncio.run(runner.run(incumbent, [challenger]))
        trials = ledger.list_trials(result.campaign_id)
        snapshot = ledger.snapshot(result.campaign_id)
        runs = store.list_runs()

    # Two steps beat one on every paired seed, through confirmation and holdout.
    assert result.promoted and len(trials) == 14
    assert {trial.status for trial in trials} == {TrialStatus.COMPLETED}
    assert {trial.actual_cost_microusd for trial in trials} == {0}
    assert snapshot["cost"] == {"confirmed_microusd": 0, "reserved_microusd": 0,
                                "unknown_exposure_microusd": 0, "total_exposure_microusd": 0}
    # One completed workflow per trial, named for its trial, each with its budget.
    assert sorted(run["run_id"] for run in runs) == sorted(
        trial_run_id(trial.trial_key) for trial in trials)
    for run in runs:
        assert run["status"] == "completed" and run["budget_nanousd"] == CEILING * 1000
        assert run["confirmed_nanousd"] == run["reserved_nanousd"] == run["unknown_nanousd"] == 0


def test_priced_trials_round_confirmed_nanousd_up_to_whole_microusd(tmp_path, monkeypatch):
    transport = NativeTransport(plan=ONE_NODE).install(monkeypatch)
    # One cache-write token costs 12,500 nanoUSD, so the step costs 1,502,500
    # nanoUSD and, with the 1,500,000 nanoUSD plan, the trial 3,002,500.
    transport.overrides["execution"] = {"usage": {
        "input_tokens": 100, "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1}}}
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db") as ledger):
        evaluate = workflow_trial_evaluator(native_factory(store), task_for, measure)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR)
        result = asyncio.run(runner.run(incumbent, [challenger]))
        trials = ledger.list_trials(result.campaign_id)
        snapshot = ledger.snapshot(result.campaign_id)
        journal = {run["run_id"]: run for run in store.list_runs()}

    # Equal quality stops the campaign after development: two trials.
    assert not result.promoted and len(trials) == 2
    for trial in trials:
        run = journal[trial_run_id(trial.trial_key)]
        assert run["confirmed_nanousd"] == 3_002_500
        assert not run["reserved_nanousd"] and not run["unknown_nanousd"]
        # Up, never down: 3,002 micro-USD would under-report the charge.
        assert trial.status is TrialStatus.COMPLETED and trial.actual_cost_microusd == 3_003
        assert from_units(3_003, MICROUSD) > from_units(3_002_500, NANOUSD) > from_units(
            3_002, MICROUSD)
    assert snapshot["cost"]["confirmed_microusd"] == 6_006
    assert len(transport.requests) == 4


def test_a_dropped_response_makes_the_trial_unknown_without_provider_text(tmp_path, monkeypatch):
    transport = NativeTransport(plan=ONE_NODE).install(monkeypatch)
    transport.schedule("execution", 0, LOST_RESPONSE)
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    built = []
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db") as ledger):
        evaluate = workflow_trial_evaluator(native_factory(store, built), task_for, measure)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    campaign_id="dropped")
        with pytest.raises(OptimizationNeedsAttention) as caught:
            asyncio.run(runner.run(incumbent, [challenger]))
        (trial,) = ledger.list_trials("dropped")
        snapshot = ledger.snapshot("dropped")
        (run,) = store.list_runs()
        state = store.inspect_run(run["run_id"])

        # The campaign stopped at the first trial and stays stopped.
        again = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                   campaign_id="dropped")
        with pytest.raises(OptimizationNeedsAttention):
            asyncio.run(again.run(incumbent, [challenger]))

    assert isinstance(caught.value.__cause__, WorkflowTrialError)
    assert trial.status is TrialStatus.UNKNOWN
    assert trial.error == f"evaluator raised WorkflowTrialError after dispatch: {UNKNOWN}"
    assert snapshot["trial_counts"] == {"unknown": 1}
    assert snapshot["cost"]["unknown_exposure_microusd"] == CEILING
    assert snapshot["cost"]["confirmed_microusd"] == 0
    # The journal keeps the lost call's exposure; the ledger keeps the trial's ceiling.
    assert run["run_id"] == trial_run_id(trial.trial_key)
    assert state["unknown_calls"] == 1 and state["unknown_nanousd"] > 0
    # No provider or transport text reaches the ledger.
    for payload in _events(tmp_path / "autotune.db"):
        for text in ("Connection", "response arrived", "billing", "resp_", "req_"):
            assert text not in payload
    assert built == [trial.trial_key]
    assert [record.outcome for record in transport.log
            if record.operation == "response"] == ["delivered", "lost_response"]


def test_a_failed_workflow_with_known_spend_keeps_provider_text_out(tmp_path, monkeypatch):
    transport = NativeTransport(plan=ONE_NODE).install(monkeypatch)
    # A 403 is refused before any model work: the journal settles it at zero.
    canary, message = "private_detail_7f3a", "private provider message 7f3a"
    transport.schedule("execution", None, HttpError(403, canary, message))
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db") as ledger):
        evaluate = workflow_trial_evaluator(native_factory(store), task_for, measure)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    campaign_id="rejected")
        with pytest.raises(OptimizationNeedsAttention) as caught:
            asyncio.run(runner.run(incumbent, [challenger]))
        (trial,) = ledger.list_trials("rejected")
        (run,) = store.list_runs()

    # The workflow's own error stays chained in the process for diagnosis.
    assert "HTTP 403" in str(caught.value.__cause__.__cause__)
    assert trial.status is TrialStatus.UNKNOWN
    assert trial.error == f"evaluator raised WorkflowTrialError after dispatch: {FAILED}"
    assert run["confirmed_nanousd"] > 0  # the plan call
    assert not run["reserved_nanousd"] and not run["unknown_nanousd"]
    for payload in _events(tmp_path / "autotune.db"):
        assert canary not in payload and message not in payload and "403" not in payload


# ---------------------------------------------------------------------------
# One trial at a time
# ---------------------------------------------------------------------------


def test_a_trial_runs_its_workflow_at_most_once(tmp_path):
    built = []
    context = _context()
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        evaluate = workflow_trial_evaluator(offline_factory(store, built), task_for, measure)
        outcome = asyncio.run(evaluate(context))
        before = store.inspect_run(trial_run_id(context.trial_key))
        with pytest.raises(WorkflowTrialError) as caught:
            asyncio.run(evaluate(context))
        after = store.inspect_run(trial_run_id(context.trial_key))
        runs = store.list_runs()

    assert outcome == TrialOutcome(metrics={"quality": 1}, gates={"delivered": True},
                                   actual_cost_microusd=0)
    assert str(caught.value) == EXISTS
    assert before == after and len(runs) == 1
    # The second evaluation built a Swarm but neither created nor resumed a run.
    assert built == [context.trial_key] * 2


def test_the_workflow_budget_is_the_trial_ceiling_exactly(tmp_path):
    budgets = []

    def build(context, max_budget_usd):
        budgets.append(max_budget_usd)
        return Swarm(model="offline", provider=OfflineProvider(plan=ONE_NODE), run_store=store,
                     max_budget_usd=max_budget_usd)

    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        asyncio.run(workflow_trial_evaluator(build, task_for, measure)(_context(ceiling=1_234_567)))
        (run,) = store.list_runs()
    assert budgets == [1.234567] and run["budget_nanousd"] == 1_234_567_000


@pytest.mark.parametrize("problem", ["no store", "other budget", "unlimited", "not a swarm",
                                     "not a task", "factory raises", "durable refusal",
                                     "revoked run control"])
def test_a_trial_the_adapter_cannot_set_up_creates_no_run(tmp_path, problem):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:

        def build(context, max_budget_usd):
            if problem == "factory raises":
                raise RuntimeError("factory detail")
            if problem == "not a swarm":
                return object()
            budget = {"other budget": max_budget_usd / 2, "unlimited": None}.get(
                problem, max_budget_usd)
            # A scripted OfflineProvider builds a Swarm, but no durable recipe.
            provider = (OfflineProvider(responses=["scripted"]) if problem == "durable refusal"
                        else OfflineProvider(plan=ONE_NODE))
            control = RunControl()
            if problem == "revoked run control":
                control.revoke("trial-stop")
            return Swarm(model="offline", provider=provider,
                         run_store=None if problem == "no store" else store,
                         max_budget_usd=budget, run_control=control)

        def tasks(context):
            return "Write a report" if problem == "not a task" else task_for(context)

        evaluate = workflow_trial_evaluator(build, tasks, measure)
        with pytest.raises(WorkflowTrialError) as caught:
            asyncio.run(evaluate(_context()))
        assert str(caught.value) == REFUSED and caught.value.__cause__ is not None
        assert store.list_runs() == []
    if problem == "revoked run control":
        assert isinstance(caught.value.__cause__, SecurityStop)


def test_a_ceiling_a_float_budget_cannot_hold_is_refused_before_building(tmp_path):
    built = []
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        evaluate = workflow_trial_evaluator(offline_factory(store, built), task_for, measure)
        with pytest.raises(WorkflowTrialError, match=f"^{REFUSED}$"):
            asyncio.run(evaluate(_context(ceiling=(1 << 63) - 1)))
        assert store.list_runs() == [] and built == []


@pytest.mark.parametrize("value", [RuntimeError("measure detail"),
                                   ([("quality", 1)], {"delivered": True})])
def test_a_measure_failure_has_fixed_text(tmp_path, value):
    def broken(context, result):
        if isinstance(value, Exception):
            raise value
        return value

    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        evaluate = workflow_trial_evaluator(offline_factory(store), task_for, broken)
        with pytest.raises(WorkflowTrialError, match=f"^{MEASURE}$"):
            asyncio.run(evaluate(_context()))
        (run,) = store.list_runs()
    # The workflow itself completed; only its measurement failed.
    assert run["status"] == "completed"


def test_spend_left_on_a_returned_workflow_is_unknown(tmp_path, monkeypatch):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        inspect = SQLiteWorkflowStore.inspect_run

        def reserved(self, run_id):
            return {**inspect(self, run_id), "reserved_nanousd": 1}

        evaluate = workflow_trial_evaluator(offline_factory(store), task_for, measure)
        monkeypatch.setattr(SQLiteWorkflowStore, "inspect_run", reserved)
        with pytest.raises(WorkflowTrialError, match=f"^{UNKNOWN}$"):
            asyncio.run(evaluate(_context()))


def _chain(error):
    """*error* and every exception chained behind it, by cause or context."""
    chain = []
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    return chain


def strict_factory(store):
    """Durable Swarms whose generated plan no gate could accept under a strict policy."""

    def build(context, max_budget_usd):
        return Swarm(model="offline", provider=OfflineProvider(plan=ONE_NODE), run_store=store,
                     max_budget_usd=max_budget_usd, acceptance=AcceptancePolicy(mode="strict"))

    return build


def test_a_plan_only_the_run_refuses_leaves_its_run_with_nothing_spent(tmp_path):
    context = _context()
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        evaluate = workflow_trial_evaluator(strict_factory(store), task_for, measure)
        with pytest.raises(WorkflowTrialError, match=f"^{FAILED}$") as caught:
            asyncio.run(evaluate(context))
        (run,) = store.list_runs()
        state = store.inspect_run(run["run_id"])

    # The run's own planning check refused it once it existed, before any call.
    assert "architect that generates its gates" in str(caught.value.__cause__)
    assert run["run_id"] == trial_run_id(context.trial_key) and run["status"] == "running"
    assert run["confirmed_nanousd"] == run["reserved_nanousd"] == run["unknown_nanousd"] == 0
    assert state["calls"] == []


def test_an_unreadable_state_keeps_the_workflow_failure_in_the_chain(tmp_path, monkeypatch):
    inspect, aresume = SQLiteWorkflowStore.inspect_run, Swarm.aresume
    resumed = []

    async def resume(self, run_id):
        try:
            return await aresume(self, run_id)
        finally:
            resumed.append(run_id)

    def unreadable(self, run_id):
        # The run reads its own state; only the adapter's read afterwards fails.
        if resumed:
            raise sqlite3.OperationalError("database is locked")
        return inspect(self, run_id)

    monkeypatch.setattr(Swarm, "aresume", resume)
    monkeypatch.setattr(SQLiteWorkflowStore, "inspect_run", unreadable)
    with SQLiteWorkflowStore(tmp_path / "runs.db") as store:
        evaluate = workflow_trial_evaluator(strict_factory(store), task_for, measure)
        with pytest.raises(WorkflowTrialError, match=f"^{UNKNOWN}$") as caught:
            asyncio.run(evaluate(_context()))

    # The unreadable state is the direct cause; the workflow's own failure
    # stays chained behind it.
    chain = _chain(caught.value)
    assert isinstance(chain[1], sqlite3.OperationalError)
    assert any("architect that generates its gates" in str(error) for error in chain[2:])


def test_an_outcome_the_runner_rejects_is_recorded_as_an_invalid_outcome(tmp_path):
    def unscored(context, result):
        return {"score": 1}, {"delivered": bool(result.output)}

    contract = _contract()
    incumbent, challenger = _candidates(contract)
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db") as ledger):
        evaluate = workflow_trial_evaluator(offline_factory(store), task_for, unscored)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    campaign_id="unscored")
        with pytest.raises(OptimizationNeedsAttention, match="invalid outcome") as caught:
            asyncio.run(runner.run(incumbent, [challenger]))
        (trial,) = ledger.list_trials("unscored")
        (run,) = store.list_runs()

    # The adapter reported the outcome, and the runner's own check refused it.
    assert type(caught.value.__cause__) is ValueError
    assert trial.status is TrialStatus.UNKNOWN
    assert trial.error == (
        "evaluator returned an invalid outcome: ValueError: metrics must exactly match "
        "contract objectives; missing=['quality'], extra=['score']")
    assert run["run_id"] == trial_run_id(trial.trial_key) and run["status"] == "completed"


@pytest.mark.parametrize("name", ["runtime_factory", "task_for", "measure"])
def test_every_part_must_be_callable(name):
    parts = {"runtime_factory": offline_factory(None), "task_for": task_for, "measure": measure}
    parts[name] = "not callable"
    with pytest.raises(TypeError, match=name):
        workflow_trial_evaluator(**parts)


# ---------------------------------------------------------------------------
# Process loss
# ---------------------------------------------------------------------------

# Child code: a campaign whose first trial holds its workflow's step call
# until the test kills the process.
CHILD = r"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fakes import NativeTransport, write_barrier
from smythe import SQLiteWorkflowStore
from smythe.optimize.engine import OptimizationRunner
from smythe.optimize.ledger import ExperimentLedger
from smythe.optimize.workflow_eval import workflow_trial_evaluator
from test_optimize_workflow_eval import (
    EVALUATOR, ONE_NODE, _candidates, _contract, measure, native_factory, task_for,
)

folder = Path(sys.argv[1])
transport = NativeTransport(plan=ONE_NODE).install(SimpleNamespace(setattr=setattr))


def hold(prompt):
    write_barrier(folder / "held.json", {"held": True})
    time.sleep(600)


transport.replies["execution"] = hold
contract = _contract()
incumbent, challenger = _candidates(contract)
with SQLiteWorkflowStore(folder / "runs.db") as store, ExperimentLedger(folder / "autotune.db") as ledger:
    evaluate = workflow_trial_evaluator(native_factory(store), task_for, measure)
    runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                campaign_id="killed", lease_ttl_s=3600)
    asyncio.run(runner.run(incumbent, [challenger]))
"""


def test_a_killed_runner_leaves_its_trial_dispatched_and_resume_never_reruns_it(
        tmp_path, monkeypatch):
    process = launch_child(CHILD, tmp_path)
    try:
        assert wait_for_barrier(process, tmp_path / "held.json", timeout=120) == {"held": True}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0

    transport = NativeTransport(plan=ONE_NODE).install(monkeypatch)
    contract = _contract()
    incumbent, challenger = _candidates(contract)
    built = []
    with (SQLiteWorkflowStore(tmp_path / "runs.db") as store,
          ExperimentLedger(tmp_path / "autotune.db") as ledger):
        (trial,) = ledger.list_trials("killed")
        assert trial.status is TrialStatus.DISPATCHED
        (run,) = store.list_runs()
        assert run["run_id"] == trial_run_id(trial.trial_key)
        before = store.inspect_run(run["run_id"])
        # The step call was sent and never answered.
        assert [call["state"] for call in before["calls"]][-1] == "dispatched"

        # A successor takes the campaign once the dead runner's lease lapses.
        lease = ledger.get_campaign_lease("killed")
        monkeypatch.setattr(ledger_module, "_lease_now_ns", lambda: lease.expires_at_ns + 1)
        evaluate = workflow_trial_evaluator(native_factory(store, built), task_for, measure)
        runner = OptimizationRunner(contract, ledger, evaluate, evaluator_hash=EVALUATOR,
                                    campaign_id="killed")
        with pytest.raises(OptimizationNeedsAttention, match="dispatched"):
            asyncio.run(runner.run(incumbent, [challenger]))

        assert ledger.get_trial(trial.trial_key).status is TrialStatus.DISPATCHED
        assert ledger.snapshot("killed")["cost"]["reserved_microusd"] == CEILING
        assert store.list_runs() == [run] and store.inspect_run(run["run_id"]) == before
    # The workflow was neither rebuilt nor sent again.
    assert built == [] and transport.requests == [] and transport.counts == []
    assert json.loads((tmp_path / "held.json").read_text(encoding="utf-8")) == {"held": True}
