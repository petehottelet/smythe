"""Bounded exploration runs Autotune campaigns on an out-of-process proposer's challengers.

Each round reserves the proposer's ceiling in the policy registry, runs the
proposer program as a real child process, settles its declared spend, and
runs one campaign against a fixed incumbent on a fake evaluator. Every stop
condition ends the loop, the exposure cap stops it before a campaign that
would exceed it, a killed proposer leaves its charge unknown at the ceiling,
refused proposals write nothing to the ledger, and nothing is activated. The
proposer's view and request hold no holdout metric, seed or trial and no path,
and its environment holds no credentials.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import json
import os
import sqlite3
import time

import pytest

from fakes import child_executable
from smythe._redact import redact_paths
from smythe.optimize import (
    OptimizationError,
    OptimizationNeedsAttention,
    OptimizationRunner,
    PolicyRegistry,
    RegistryError,
    TrialContext,
    TrialOutcome,
)
from smythe.optimize.contracts import (
    Candidate,
    ExperimentContract,
    MetricObjective,
    MutableFieldRule,
)
from smythe.optimize.exploration import (
    ExplorationBudget,
    SubprocessProposer,
    explore,
    proposer_view,
)
from smythe.optimize.ledger import ExperimentLedger

TARGET = "planner.template.report"
FIELD = TARGET + ".max_sections"
TONE = TARGET + ".tone"
EVALUATOR = "sha256:" + "e" * 64
WRITER = "smythe_policy_registry_writer_version"
TRIAL_COST = 5
RESERVATION = 10
# One challenger plans 14 trials and two plan 15, each reserved at 10 micro-USD.
ONE_CHALLENGER, TWO_CHALLENGERS = 140, 150

# The proposer program: it saves the request and its environment's variable
# names for the test, then replies with the scripted reply for the round.
PROPOSER = r"""
import json
import os
import sys
import time

folder = sys.argv[1]
raw = sys.stdin.read()
round_ = json.loads(raw)["round"]
with open(os.path.join(folder, f"request-{round_}.json"), "w", encoding="utf-8") as handle:
    handle.write(raw)
with open(os.path.join(folder, f"environment-{round_}.json"), "w", encoding="utf-8") as handle:
    json.dump(sorted(os.environ), handle)
with open(os.path.join(folder, "replies.json"), encoding="utf-8") as handle:
    reply = json.load(handle)[round_ - 1]
if reply == "hang":
    while True:
        with open(os.path.join(folder, "beat"), "ab") as handle:
            handle.write(b".")
        time.sleep(0.05)
if reply == "fail":
    sys.exit(3)
sys.stdout.write(reply if isinstance(reply, str) else json.dumps(reply))
"""


def _contract(*, mutable=(FIELD, TONE)) -> ExperimentContract:
    rules = {FIELD: MutableFieldRule("integer", minimum=1, maximum=12),
             TONE: MutableFieldRule("string", choices=("plain", "formal")),
             "budget.max_usd": MutableFieldRule("integer", minimum=0, maximum=9)}
    return ExperimentContract(
        name="report_sections",
        objectives=(MetricObjective("quality", "maximize", primary=True),),
        mutable_fields=mutable,
        mutable_field_rules={path: rules[path] for path in mutable},
        required_gates=("valid",),
        development_repetitions=1,
        confirmation_repetitions=3,
        holdout_repetitions=3,
        max_candidates=3,
        max_parallel_candidates=2,
        max_trials=30,
        max_wall_seconds=600,
        max_budget_microusd=1_000,
        per_trial_reservation_microusd=RESERVATION,
        confidence=0.8,
        min_improvement=0.1,
        base_seed=17,
    )


def _incumbent(contract: ExperimentContract) -> Candidate:
    return Candidate(contract, {FIELD: 4, TONE: "plain"}, "4 sections")


async def _more_sections_score_higher(context: TrialContext) -> TrialOutcome:
    noise = context.seed % 9973 / 1e9
    return TrialOutcome(metrics={"quality": context.candidate.policy[FIELD] + noise},
                        gates={"valid": True}, actual_cost_microusd=TRIAL_COST)


async def _holdout_reverses_the_ranking(context: TrialContext) -> TrialOutcome:
    """More sections win development and confirmation but lose on the holdout.

    Every holdout value is unique and found in no other split.
    """
    sections = context.candidate.policy[FIELD]
    noise = context.seed % 9973 / 1e7
    quality = 20.123456789 - sections + noise if context.split == "holdout" else sections + noise
    return TrialOutcome(metrics={"quality": quality}, gates={"valid": True},
                        actual_cost_microusd=TRIAL_COST)


def _reply(*sections: int, spent: int = 0, tone: str = "plain") -> dict:
    return {"proposals": [{"policy": {FIELD: count, TONE: tone}, "hypothesis": f"{count} sections"}
                          for count in sections],
            "spent_microusd": spent}


def _budget(**changes) -> ExplorationBudget:
    limits = {"max_campaigns": 10, "max_total_exposure_microusd": 10_000,
              "max_proposer_microusd": 1_000, **changes}
    return ExplorationBudget(**limits)


def _proposer(folder, *, ceiling: int = 10, timeout_s: float = 60) -> SubprocessProposer:
    return SubprocessProposer([child_executable(), "-c", PROPOSER, str(folder)],
                              ceiling_microusd=ceiling, timeout_s=timeout_s)


def _runner(ledger, *, evaluate=_more_sections_score_higher, single_change=False, **options):
    return OptimizationRunner(_contract(), ledger, evaluate, evaluator_hash=EVALUATOR,
                              bootstrap_resamples=10, single_change=single_change, **options)


@pytest.fixture
def stores(tmp_path):
    """An Autotune ledger and a policy registry apart from the proposer's folder."""
    store = tmp_path / "store"
    folder = tmp_path / "proposer"
    store.mkdir()
    folder.mkdir()
    ledger = ExperimentLedger(store / "optimize.sqlite3", durability="normal")
    registry = PolicyRegistry(store / "registry.sqlite3")
    try:
        yield ledger, registry, folder
    finally:
        ledger.close()
        registry.close()


def _explore(stores, replies, budget=None, *, ceiling=10, timeout_s=60, **runner_options):
    ledger, registry, folder = stores
    (folder / "replies.json").write_text(json.dumps(replies), encoding="utf-8")
    runner = _runner(ledger, **runner_options)
    return asyncio.run(explore(runner, registry, _proposer(folder, ceiling=ceiling,
                                                          timeout_s=timeout_s),
                               _incumbent(runner.contract), budget or _budget()))


def _dump(path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return list(connection.iterdump())


def _count(path, table: str) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _kinds(record: dict) -> list[tuple[int, str]]:
    return [(event["round"], event["kind"]) for event in record["events"]]


def _requests(folder) -> int:
    return len(list(folder.glob("request-*.json")))


def _assert_nothing_activated(registry: PolicyRegistry) -> None:
    for table in ("activations", "reviews"):
        assert _count(registry.path, table) == 0
    assert registry.history(TARGET) == ()


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

STOPS = {
    # reason: (replies, budget changes, campaigns run, proposer calls)
    "promoted": ([_reply(6)], {}, 1, 1),
    "max_non_promotions": ([_reply(3), _reply(2), _reply(1)], {"max_non_promotions": 2}, 2, 2),
    "max_campaigns": ([_reply(3), _reply(2), _reply(1)], {"max_campaigns": 2}, 2, 2),
    # Two calls settle 20; a third at its 10 ceiling would pass 25.
    "max_proposer_spend": ([_reply(3, spent=10), _reply(2, spent=10), _reply(1)],
                           {"max_proposer_microusd": 25}, 2, 2),
    # The first campaign stops in development: two trials expose 10.
    "max_total_exposure": ([_reply(3), _reply(2)],
                           {"max_total_exposure_microusd": 10 + ONE_CHALLENGER - 1}, 1, 1),
    "proposer_finished": ([_reply(3), {"proposals": [], "spent_microusd": 2}], {}, 1, 2),
}


@pytest.mark.parametrize("reason", sorted(STOPS))
def test_each_stop_condition_ends_the_loop(stores, reason):
    replies, changes, campaigns, calls = STOPS[reason]
    ledger, registry, folder = stores
    result = _explore(stores, replies, _budget(**changes))

    assert result.stop_reason == reason
    assert len(result.campaigns) == campaigns == len(ledger.list_campaigns())
    assert [item.promoted for item in result.campaigns] == [reason == "promoted"] + [False] * (
        campaigns - 1)
    assert _requests(folder) == calls
    record = registry.exploration(result.exploration_id)
    assert record["stop_reason"] == reason
    assert record["campaign_ids"] == [item.campaign_id for item in result.campaigns]
    assert _kinds(record)[-1] == (calls + (reason in ("max_campaigns", "max_proposer_spend",
                                                      "max_total_exposure")), "stopped")
    settled = sum(reply["spent_microusd"] for reply in replies[:calls])
    assert record["proposer_cost"] == {"reserved_microusd": 0, "settled_microusd": settled,
                                       "unknown_microusd": 0, "total_exposure_microusd": settled}
    assert (result.proposer_settled_microusd, result.proposer_unknown_microusd) == (settled, 0)
    exposure = sum(ledger.snapshot(item.campaign_id)["cost"]["total_exposure_microusd"]
                   for item in result.campaigns)
    assert result.campaign_exposure_microusd == exposure
    assert [event["microusd"] for event in record["events"]
            if event["kind"] == "campaign_ended"] == [
        ledger.snapshot(item.campaign_id)["cost"]["total_exposure_microusd"]
        for item in result.campaigns]
    _assert_nothing_activated(registry)


def test_the_exposure_cap_stops_before_a_campaign_that_would_exceed_it(stores):
    ledger, registry, folder = stores
    # After the first campaign exposes 10, one challenger fits but two do not.
    cap = 10 + TWO_CHALLENGERS - 1
    result = _explore(stores, [_reply(3), _reply(2, 1, spent=3)],
                      _budget(max_total_exposure_microusd=cap))

    assert result.stop_reason == "max_total_exposure"
    assert f"would reserve {TWO_CHALLENGERS} micro-USD" in result.detail
    assert result.campaign_exposure_microusd == 10
    assert len(ledger.list_campaigns()) == 1
    record = registry.exploration(result.exploration_id)
    assert _kinds(record) == [
        (1, "proposer_reserved"), (1, "proposer_settled"), (1, "campaign_started"),
        (1, "campaign_ended"), (2, "proposer_reserved"), (2, "proposer_settled"), (2, "stopped")]
    started = next(event for event in record["events"] if event["kind"] == "campaign_started")
    assert started["microusd"] == ONE_CHALLENGER
    assert record["proposer_cost"]["settled_microusd"] == 3


def test_explore_never_activates_a_promoted_decision(stores):
    ledger, registry, _ = stores
    result = _explore(stores, [_reply(6)])
    (promoted,) = result.campaigns
    assert result.stop_reason == "promoted" and promoted.promoted
    _assert_nothing_activated(registry)
    # The decision waits for an explicit review and activation, as any other does.
    review = registry.record_review(ledger, promoted.campaign_id, reviewer="report-owner",
                                    verdict="approve")
    registry.activate(TARGET, review.review_id, expected_seq=0, actor="operator",
                      reason="reviewed exploration result")
    assert registry.resolve(TARGET).policy == {FIELD: 6, TONE: "plain"}


# ---------------------------------------------------------------------------
# Proposer failures and charges
# ---------------------------------------------------------------------------


def _beats(folder) -> int:
    beat = folder / "beat"
    return beat.stat().st_size if beat.exists() else 0


def _assert_stopped_beating(folder) -> None:
    before = _beats(folder)
    time.sleep(0.5)
    assert _beats(folder) == before


@pytest.mark.parametrize("reply, message", [
    ("hang", "did not reply within 5 seconds and was killed"),
    ("fail", "exited with status 3"),
    ("not json", "is not a JSON object"),
    ({"proposals": []}, "declares no whole spent_microusd"),
    ({"proposals": [], "spent_microusd": True}, "declares no whole spent_microusd"),
])
def test_a_failed_proposer_leaves_its_charge_unknown_at_the_ceiling(stores, reply, message):
    ledger, registry, folder = stores
    started = time.monotonic()
    result = _explore(stores, [reply], ceiling=12, timeout_s=5)

    assert result.stop_reason == "proposer_failed" and message in result.detail
    assert time.monotonic() - started < 60
    assert result.campaigns == () and ledger.list_campaigns() == []
    record = registry.exploration(result.exploration_id)
    assert _kinds(record) == [(1, "proposer_reserved"), (1, "proposer_unknown"), (1, "stopped")]
    assert record["events"][1]["microusd"] == 12 and message in record["events"][1]["detail"]
    assert record["proposer_cost"] == {"reserved_microusd": 0, "settled_microusd": 0,
                                       "unknown_microusd": 12, "total_exposure_microusd": 12}
    assert result.proposer_unknown_microusd == 12
    if reply == "hang":
        # The program was running at its deadline and was killed.
        assert _beats(folder) > 0
        _assert_stopped_beating(folder)


def test_a_declaration_above_the_ceiling_is_settled_and_stops_exploration(stores):
    ledger, registry, _ = stores
    result = _explore(stores, [_reply(6, spent=15)], ceiling=10)
    assert result.stop_reason == "proposer_failed"
    assert "declared 15 micro-USD above its 10 ceiling" in result.detail
    assert ledger.list_campaigns() == []
    record = registry.exploration(result.exploration_id)
    assert record["proposer_cost"]["settled_microusd"] == 15
    assert _kinds(record) == [(1, "proposer_reserved"), (1, "proposer_settled"), (1, "stopped")]


def test_a_proposer_that_cannot_start_leaves_its_charge_unknown(stores, tmp_path):
    ledger, registry, _ = stores
    proposer = SubprocessProposer([str(tmp_path / "missing-proposer")], ceiling_microusd=7,
                                  timeout_s=5)
    runner = _runner(ledger)
    result = asyncio.run(explore(runner, registry, proposer, _incumbent(runner.contract),
                                 _budget()))
    assert result.stop_reason == "proposer_failed"
    assert result.detail == "the proposer could not be started"
    assert registry.exploration(result.exploration_id)["proposer_cost"]["unknown_microusd"] == 7


def test_cancelling_exploration_kills_the_proposer_and_records_its_charge(stores):
    ledger, registry, folder = stores
    (folder / "replies.json").write_text(json.dumps(["hang"]), encoding="utf-8")
    runner = _runner(ledger)

    async def cancel_while_proposing():
        task = asyncio.create_task(explore(
            runner, registry, _proposer(folder, ceiling=9, timeout_s=120),
            _incumbent(runner.contract), _budget(), exploration_id="cancelled"))
        deadline = time.monotonic() + 60
        while not _beats(folder):
            assert time.monotonic() < deadline and not task.done()
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.monotonic()
    asyncio.run(cancel_while_proposing())
    assert time.monotonic() - started < 60
    _assert_stopped_beating(folder)
    record = registry.exploration("cancelled")
    assert _kinds(record) == [(1, "proposer_reserved"), (1, "proposer_unknown"), (1, "stopped")]
    assert record["events"][1]["detail"] == "interrupted"
    assert record["stop_reason"] == "interrupted"
    assert record["proposer_cost"]["unknown_microusd"] == 9


def test_a_campaign_that_needs_attention_stops_exploration_and_is_raised(stores):
    ledger, registry, folder = stores

    async def lose_the_outcome(context: TrialContext) -> TrialOutcome:
        raise ConnectionError("response lost after dispatch")

    (folder / "replies.json").write_text(json.dumps([_reply(6)]), encoding="utf-8")
    runner = _runner(ledger, evaluate=lose_the_outcome)
    with pytest.raises(OptimizationNeedsAttention):
        asyncio.run(explore(runner, registry, _proposer(folder), _incumbent(runner.contract),
                            _budget(), exploration_id="interrupted"))
    record = registry.exploration("interrupted")
    assert record["stop_reason"] == "interrupted"
    assert _kinds(record)[-2:] == [(1, "campaign_started"), (1, "stopped")]
    assert record["campaign_ids"] == [campaign["campaign_id"]
                                      for campaign in ledger.list_campaigns()]


# ---------------------------------------------------------------------------
# Proposal bounds
# ---------------------------------------------------------------------------


def _proposal(policy: dict, hypothesis: str = "a change") -> dict:
    return {"policy": policy, "hypothesis": hypothesis}


REFUSED = {
    "reserved path": ([_proposal({"budget.max_usd": 5})], "sets reserved paths: budget.max_usd"),
    "reserved path in other letter case": (
        [_proposal({FIELD: 6, "Acceptance.mode": "lenient"})],
        "sets reserved paths: Acceptance.mode"),
    "path outside the allowlist": ([_proposal({FIELD: 6, TARGET + ".unlisted": 1})],
                                   "is not mutable"),
    "value outside its rule": ([_proposal({FIELD: 99, TONE: "plain"})], "proposal 0 is refused"),
    "two changes under single_change": ([_proposal({FIELD: 6, TONE: "formal"})],
                                        "exactly one policy path"),
    "the incumbent's own policy": ([_proposal({FIELD: 4, TONE: "plain"})],
                                   "cannot repeat the incumbent policy"),
    "more challengers than the contract allows": (
        [_proposal({FIELD: count, TONE: "plain"}, f"{count} sections") for count in (6, 7, 8)],
        "exceed max_candidates"),
    "a proposal without its hypothesis": ([{"policy": {FIELD: 6, TONE: "plain"}}],
                                          "exactly a policy and a hypothesis"),
    "no proposals list": (None, "no proposals list"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_proposals_outside_the_bounds_are_refused_without_ledger_writes(stores, case):
    proposals, message = REFUSED[case]
    ledger, registry, _ = stores
    before = _dump(ledger.path)
    reply = {"spent_microusd": 1} if proposals is None else {"proposals": proposals,
                                                              "spent_microusd": 1}
    result = _explore(stores, [reply], single_change=True)

    assert result.stop_reason == "proposal_refused" and message in result.detail
    assert _dump(ledger.path) == before
    record = registry.exploration(result.exploration_id)
    assert _kinds(record) == [(1, "proposer_reserved"), (1, "proposer_settled"), (1, "stopped")]
    assert record["proposer_cost"]["settled_microusd"] == 1


def test_proposals_may_not_rerun_a_campaign_or_reuse_a_holdout(stores):
    ledger, registry, folder = stores
    repeated = _explore(stores, [_reply(3), _reply(3)])
    assert repeated.stop_reason == "proposal_refused"
    assert f"repeat campaign {repeated.campaigns[0].campaign_id}" in repeated.detail

    promoted = _explore(stores, [_reply(6)])
    assert promoted.stop_reason == "promoted"
    before = _dump(ledger.path)
    # The same policy under a new hypothesis is a new campaign, but its holdout is used.
    again = _explore(stores, [{"proposals": [_proposal({FIELD: 6, TONE: "plain"}, "six again")],
                               "spent_microusd": 0}])
    assert again.stop_reason == "proposal_refused" and "already used" in again.detail
    assert _dump(ledger.path) == before
    assert len({repeated.exploration_id, promoted.exploration_id, again.exploration_id}) == 3


# ---------------------------------------------------------------------------
# What the proposer sees
# ---------------------------------------------------------------------------


def _values(value):
    """Every key, string and number in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)
    else:
        yield value


def test_the_proposer_view_holds_no_holdout_values_paths_or_credentials(
    stores, tmp_path, monkeypatch,
):
    monkeypatch.setenv("SMYTHE_EXPLORATION_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SMYTHE_EXPLORATION_TOKEN", "not-a-real-token")
    monkeypatch.setenv("SMYTHE_MCP_URL", "http://127.0.0.1:9/mcp")
    monkeypatch.setenv("SMYTHE_EXPLORATION_SETTING", "kept")
    ledger, registry, folder = stores
    # Both campaigns pass confirmation and fail the holdout, so the loop goes on.
    result = _explore(stores, [_reply(6), _reply(5), {"proposals": [], "spent_microusd": 0}],
                      evaluate=_holdout_reverses_the_ranking)
    assert result.stop_reason == "proposer_finished"
    campaign_ids = [item.campaign_id for item in result.campaigns]
    assert [item.confirmation_assessment.promote for item in result.campaigns] == [True, True]
    assert [item.promoted for item in result.campaigns] == [False, False]

    holdout = [trial for campaign_id in campaign_ids
               for trial in ledger.list_trials(campaign_id, split="holdout")]
    assert len(holdout) == 12
    hidden_numbers = {trial.metrics["quality"] for trial in holdout} | {
        trial.seed for trial in holdout}
    with closing(sqlite3.connect(ledger.path)) as connection:
        nonces = [row[0].hex() for row in connection.execute("SELECT holdout_nonce FROM campaigns")]
    hidden_text = [trial.trial_key for trial in holdout] + nonces + [
        ledger.get_holdout_seed_commitment(campaign_id).removeprefix("sha256:")
        for campaign_id in campaign_ids] + [str(tmp_path), tmp_path.name, "optimize.sqlite3"]

    view = proposer_view(ledger, campaign_ids)
    sent = json.loads((folder / "request-3.json").read_text(encoding="utf-8"))
    assert sent["evidence"] == json.loads(json.dumps(view))
    assert sent["round"] == 3 and sent["ceiling_microusd"] == 10
    assert sent["incumbent"]["policy"] == {FIELD: 4, TONE: "plain"}
    for request in folder.glob("request-*.json"):
        for value in _values(json.loads(request.read_text(encoding="utf-8"))):
            if isinstance(value, str):
                assert redact_paths(value) == value
                assert not any(item in value for item in hidden_text), value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                assert value not in hidden_numbers, value

    # Development and confirmation evidence is complete; the holdout is pass or fail.
    for campaign, outcome in zip(view["campaigns"], result.campaigns, strict=True):
        assert {trial["split"] for trial in campaign["trials"]} == {"development",
                                                                   "confirmation"}
        assert len(campaign["trials"]) == 2 + 6
        decision = campaign["decision"]
        assert (decision["holdout"], decision["promoted"], decision["reason"]) == (
            "failed", False, None)
        assert decision["confirmation"]["promote"] is True
        assert decision["development_scores"][0]["candidate_id"] == (
            outcome.selected_candidate_id)
        assert {item["candidate_id"] for item in campaign["candidates"]} == {
            outcome.incumbent_candidate_id, outcome.selected_candidate_id}

    for environment in folder.glob("environment-*.json"):
        names = set(json.loads(environment.read_text(encoding="utf-8")))
        assert "SMYTHE_EXPLORATION_SETTING" in names
        assert not names & {"SMYTHE_EXPLORATION_API_KEY", "SMYTHE_EXPLORATION_TOKEN",
                            "SMYTHE_MCP_URL"}


def test_a_view_shows_the_reason_of_a_decision_made_before_the_holdout(stores):
    ledger, _, _ = stores
    result = _explore(stores, [_reply(3), {"proposals": [], "spent_microusd": 0}])
    (campaign,) = proposer_view(ledger, [result.campaigns[0].campaign_id])["campaigns"]
    assert campaign["decision"]["holdout"] == "not_run"
    assert campaign["decision"]["reason"] == result.campaigns[0].reason
    assert campaign["decision"]["confirmation"] is None
    assert {trial["split"] for trial in campaign["trials"]} == {"development"}


# ---------------------------------------------------------------------------
# Arguments and the registry's exploration tables
# ---------------------------------------------------------------------------


def test_explore_refuses_a_fixed_campaign_or_reserved_mutable_fields_before_recording(stores):
    ledger, registry, folder = stores
    proposer = _proposer(folder)
    contract = _contract()
    fixed = OptimizationRunner(contract, ledger, _more_sections_score_higher,
                               evaluator_hash=EVALUATOR, campaign_id="fixed")
    with pytest.raises(ValueError, match="names no campaign_id"):
        asyncio.run(explore(fixed, registry, proposer, _incumbent(contract), _budget()))
    unsafe = _contract(mutable=(FIELD, "budget.max_usd"))
    runner = OptimizationRunner(unsafe, ledger, _more_sections_score_higher,
                                evaluator_hash=EVALUATOR)
    with pytest.raises(OptimizationError, match="reserved=budget.max_usd"):
        asyncio.run(explore(runner, registry, proposer,
                            Candidate(unsafe, {FIELD: 4}, "4 sections"), _budget()))
    with pytest.raises(OptimizationError, match="incumbent must be a candidate"):
        asyncio.run(explore(_runner(ledger), registry, proposer,
                            Candidate(unsafe, {FIELD: 4}, "4 sections"), _budget()))
    assert _count(registry.path, "explorations") == 0
    assert _requests(folder) == 0 and ledger.list_campaigns() == []


@pytest.mark.parametrize("arguments", [
    {"max_campaigns": 0}, {"max_campaigns": True}, {"max_total_exposure_microusd": -1},
    {"max_proposer_microusd": 1.5}, {"max_non_promotions": 0},
])
def test_budgets_are_whole_limits(arguments):
    with pytest.raises(ValueError):
        _budget(**arguments)


@pytest.mark.parametrize("argv, options", [
    ((), {}), ("python", {}), ([""], {}), (["python"], {"ceiling_microusd": -1}),
    (["python"], {"timeout_s": 0}), (["python"], {"timeout_s": float("inf")}),
])
def test_subprocess_proposers_need_a_command_a_ceiling_and_a_deadline(argv, options):
    with pytest.raises(ValueError):
        SubprocessProposer(argv, **{"ceiling_microusd": 1, "timeout_s": 1.0, **options})


def _exploring(path) -> None:
    """A registry with one exploration whose first proposer call settled."""
    with PolicyRegistry(path) as registry:
        registry._start_exploration("exploration-1", {"budget": {"max_campaigns": 1}})
        registry._record_exploration("exploration-1", 1, "proposer_reserved", microusd=10)
        registry._record_exploration("exploration-1", 1, "proposer_settled", microusd=4)


@pytest.mark.parametrize("statement", [
    "UPDATE explorations SET created_at_ns = 2",
    "DELETE FROM explorations",
    "UPDATE exploration_events SET microusd = 0",
    "DELETE FROM exploration_events",
    "INSERT OR REPLACE INTO explorations VALUES ('exploration-1', '{}', 1)",
    "INSERT OR REPLACE INTO exploration_events VALUES "
    "('exploration-1', 2, 1, 'proposer_settled', 0, NULL, '', 1)",
    # An event that skips the exploration's next seq.
    "INSERT INTO exploration_events VALUES ('exploration-1', 4, 1, 'stopped', NULL, NULL, 'x', 1)",
])
def test_exploration_records_are_append_only(tmp_path, statement):
    path = tmp_path / "registry.sqlite3"
    _exploring(path)
    before = _dump(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as raw:
        raw.create_function(WRITER, 0, lambda: 1)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(statement)
    assert _dump(path) == before
    insert = ("INSERT INTO exploration_events VALUES "
              "('exploration-1', 3, 1, 'stopped', NULL, NULL, 'x', 1)")
    for version in (0, 2):
        with closing(sqlite3.connect(path, isolation_level=None)) as other:
            other.create_function(WRITER, 0, lambda version=version: version)
            with pytest.raises(sqlite3.IntegrityError, match="schema v1 writer required"):
                other.execute(insert)
    assert _dump(path) == before


def test_exploration_events_follow_the_charge_lifecycle(tmp_path):
    path = tmp_path / "registry.sqlite3"
    _exploring(path)
    with PolicyRegistry(path) as registry:
        with pytest.raises(RegistryError, match="already exists"):
            registry._start_exploration("exploration-1", {})
        with pytest.raises(ValueError, match="exploration_id"):
            registry._start_exploration("../exploration", {})
        with pytest.raises(RegistryError, match="cannot record proposer_settled now"):
            registry._record_exploration("exploration-1", 1, "proposer_settled", microusd=1)
        registry._record_exploration("exploration-1", 2, "proposer_reserved", microusd=10)
        for kind, fields in (("proposer_reserved", {"microusd": 10}),
                             ("campaign_started", {"microusd": 140, "campaign_id": "c"}),
                             ("stopped", {"detail": "promoted"})):
            with pytest.raises(RegistryError, match=f"cannot record {kind} now"):
                registry._record_exploration("exploration-1", 2, kind, **fields)
        assert registry.exploration("exploration-1")["proposer_cost"]["reserved_microusd"] == 10
        registry._record_exploration("exploration-1", 2, "proposer_unknown", microusd=10,
                                     detail="killed")
        registry._record_exploration("exploration-1", 2, "stopped", detail="proposer_failed")
        with pytest.raises(RegistryError, match="cannot record stopped now"):
            registry._record_exploration("exploration-1", 2, "stopped", detail="again")
        with pytest.raises(RegistryError, match="unknown exploration"):
            registry.exploration("exploration-2")
    with PolicyRegistry(path, read_only=True) as reader:
        record = reader.exploration("exploration-1")
        with pytest.raises(RegistryError, match="read-only"):
            reader._start_exploration("exploration-2", {})
    assert record["settings"] == {"budget": {"max_campaigns": 1}}
    assert _kinds(record) == [(1, "proposer_reserved"), (1, "proposer_settled"),
                              (2, "proposer_reserved"), (2, "proposer_unknown"), (2, "stopped")]
    assert record["proposer_cost"] == {"reserved_microusd": 0, "settled_microusd": 4,
                                       "unknown_microusd": 10, "total_exposure_microusd": 14}
    assert record["stop_reason"] == "proposer_failed" and record["campaign_ids"] == []
    assert not any(str(tmp_path) in line for line in _dump(path))
    assert os.path.basename(path) not in json.dumps(record)
