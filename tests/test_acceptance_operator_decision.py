"""Authorized operator decisions on output a strict policy left for review."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib

import pytest

from fakes import Crash
from fixture_connector import HmacApprovalHost
import smythe.acceptance
import smythe.checkpoint
from smythe import OfflineProvider, SimpleArchitect, Swarm
from smythe._run_inspection import inspect_checkpoint, inspect_journal
from smythe.acceptance import (
    OPERATOR_DECISION_VERSION,
    AcceptanceSource,
    DeliveryAcceptance,
    OperatorDecision,
    OperatorReference,
    operator_decision_sha256,
    text_sha256,
)
from smythe.actions import ApprovalRejected
from smythe.checkpoint import (
    DELIVERY_ACCEPTANCE_HISTORY_LIMIT,
    build_state,
    delivery_acceptance_from_state,
    delivery_acceptance_history,
    delivery_acceptance_sequence,
)
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.registry import Registry
from smythe.workflow import current_decision, decision_history
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowStateError
from test_strict_acceptance import (
    Forbids,
    MemoryStore,
    Returns,
    Script,
    _crashed_strict_run,
    _draft_only,
    _strict,
    _swarm,
)
from test_workflow_acceptance import (  # noqa: F401
    Contains,
    Raises,
    checkpoint_without_decision,
    durable,
    operations,
    replies,
    run_state,
    single,
    stop_at,
    store,
    strict,
)

EVIDENCE = hashlib.sha256(b"review notes: the summary was read and approved").hexdigest()
OTHER_EVIDENCE = hashlib.sha256(b"other notes").hexdigest()
FAR = 2**62


class RecordingHost(HmacApprovalHost):
    """The test approval host, recording every digest it is asked to verify."""

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.verified: list[str] = []

    def verify(self, assertion, intent_sha256, now_ns):
        self.verified.append(intent_sha256)
        return super().verify(assertion, intent_sha256, now_ns)


@pytest.fixture
def host() -> RecordingHost:
    return RecordingHost()


def sign(host, run_id, sequence, record, decision="accepted", evidence=EVIDENCE, *,
         expires_at_ns=FAR) -> str:
    """An assertion *host* issues for *decision* on *record*, the run's decision at *sequence*."""
    digest = operator_decision_sha256(run_id, sequence, record, decision, evidence)
    return host.issue(digest, subject="reviewer-3", tenant="acme", expires_at_ns=expires_at_ns)


def decide(host, assertion, decision="accepted", evidence=EVIDENCE) -> OperatorDecision:
    return OperatorDecision(decision, evidence, assertion=assertion, verifier=host)


def reference(host) -> OperatorReference:
    return OperatorReference(host.issuer, "reviewer-3", "acme", host.identity, EVIDENCE)


def review_run(checkpoints, *, left="needs_review", provider=None):
    """A completed ordinary run whose final validator raised, so its output was left *left*."""
    provider = provider or Script({})
    policy = _strict(Returns(RuntimeError("validator bug")), on_invalid=left)
    result = _swarm(provider, acceptance=policy, store=checkpoints).execute(_draft_only())
    assert result.acceptance.decision.value == left and result.output == ""
    return result, policy, provider


def unchanged(state: dict) -> dict:
    """A checkpoint state without what a decision records: the decision, its features and time."""
    return {key: value for key, value in state.items() if key not in (
        "delivery_acceptance", "delivery_acceptance_history", "requires", "updated_at", "timing")}


# ---------------------------------------------------------------------------
# Ordinary checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("left", ["needs_review", "indeterminate"])
def test_an_operator_accepts_output_a_strict_policy_left_undecided(host, left):
    checkpoints = MemoryStore()
    original, policy, provider = review_run(checkpoints, left=left)
    run_id = original.execution_id
    saved = deepcopy(checkpoints.states[run_id])
    assertion = sign(host, run_id, 0, original.acceptance)

    result = _swarm(provider, store=checkpoints).reevaluate(
        run_id, operator=decide(host, assertion))

    assert result.accepted and result.output == original.rejected_output
    assert result.rejected_output is None and result.execution_id == run_id
    record = result.acceptance
    assert record.version == OPERATOR_DECISION_VERSION == 2
    assert record.operator == reference(host) and record.validators == ()
    assert record.reasons == ("an operator accepted the output after review",)
    # It decides the bytes the earlier record bound, under that record's bindings.
    for name in ("output_sha256", "policy_sha256", "strategy", "sources", "used_fallback",
                 "task_sha256", "scope"):
        assert getattr(record, name) == getattr(original.acceptance, name)
    assert any(span["node_id"] == "acceptance:delivery"
               and span["status"] == "acceptance_accepted" for span in result.trace)
    # No node ran and no provider was called; costs, budget and counters are as saved.
    assert provider.runs == {"draft": 1}
    assert result.total_cost_usd == original.total_cost_usd
    state = checkpoints.states[run_id]
    assert unchanged(state) == unchanged(saved)
    assert state["version"] == 5
    assert state["requires"] == ["delivery_acceptance", "operator_decision"]
    assert state["delivery_acceptance"] == record.to_dict()
    # The earlier record is kept, not rewritten.
    assert delivery_acceptance_history(state) == ((0, original.acceptance),)
    assert delivery_acceptance_sequence(state) == 1

    # Delivery follows the current record: on resume and in inspection.
    resumed = _swarm(provider, acceptance=policy, store=checkpoints).resume(run_id)
    assert resumed.accepted and resumed.acceptance == record
    assert resumed.output == original.rejected_output
    view = inspect_checkpoint(checkpoints, run_id)
    assert view["delivery"]["decision"] == "accepted" and view["delivery"]["output_delivered"]
    assert view["delivery"]["record"]["operator"] == reference(host).to_dict()
    assert view["decisions"]["current"] == 1
    assert view["format"]["requires"] == ["delivery_acceptance", "operator_decision"]
    assert provider.runs == {"draft": 1}
    assert len(host.verified) == 1


def test_an_operator_rejection_withholds_the_output(host):
    checkpoints = MemoryStore()
    original, policy, provider = review_run(checkpoints)
    run_id = original.execution_id
    assertion = sign(host, run_id, 0, original.acceptance, "rejected")

    result = _swarm(provider, store=checkpoints).reevaluate(
        run_id, operator=decide(host, assertion, "rejected"))

    assert result.acceptance.decision.value == "rejected" and not result.accepted
    assert (result.output, result.rejected_output) == ("", original.rejected_output)
    assert result.acceptance.operator == reference(host)
    assert result.acceptance.reasons == ("an operator rejected the output after review",)
    resumed = _swarm(provider, acceptance=policy, store=checkpoints).resume(run_id)
    assert resumed.acceptance == result.acceptance and resumed.output == ""
    assert provider.runs == {"draft": 1}


def _accepted_by_policy(host, checkpoints):
    return _swarm(Script({}), acceptance=_strict(Forbids()),
                  store=checkpoints).execute(_draft_only())


def _rejected_by_policy(host, checkpoints):
    return _swarm(Script({}), acceptance=_strict(Forbids("Summary")),
                  store=checkpoints).execute(_draft_only())


def _decided_by_operator(decision):
    def run(host, checkpoints):
        original, _, provider = review_run(checkpoints)
        assertion = sign(host, original.execution_id, 0, original.acceptance, decision)
        return _swarm(provider, store=checkpoints).reevaluate(
            original.execution_id, operator=decide(host, assertion, decision))
    return run


DECIDED = {
    "policy-accepted": _accepted_by_policy,
    "policy-rejected": _rejected_by_policy,
    "operator-accepted": _decided_by_operator("accepted"),
    "operator-rejected": _decided_by_operator("rejected"),
}


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
@pytest.mark.parametrize("case", sorted(DECIDED))
def test_an_accepted_or_rejected_decision_cannot_be_overridden(host, case, decision):
    checkpoints = MemoryStore()
    current = DECIDED[case](host, checkpoints)
    run_id = current.execution_id
    sequence = delivery_acceptance_sequence(checkpoints.states[run_id])
    before, saves = deepcopy(checkpoints.states[run_id]), len(checkpoints.saves)
    verified = len(host.verified)
    assertion = sign(host, run_id, sequence, current.acceptance, decision)

    with pytest.raises(ValueError, match="needs_review or indeterminate") as raised:
        _swarm(Script({}), store=checkpoints).reevaluate(
            run_id, operator=decide(host, assertion, decision))

    assert f"is {current.acceptance.decision.value}, which stands" in str(raised.value)
    # The assertion was never verified and nothing was written.
    assert len(host.verified) == verified
    assert checkpoints.states[run_id] == before and len(checkpoints.saves) == saves


def _forged(host, run_id, record, other):
    return sign(HmacApprovalHost(issuer=host.issuer), run_id, 0, record)


def _tampered(host, run_id, record, other):
    assertion = sign(host, run_id, 0, record)
    return assertion[:-2] + ("AA" if not assertion.endswith("AA") else "BB")


def _for_another_run(host, run_id, record, other):
    return sign(host, other.execution_id, 0, other.acceptance)


def _for_another_decision(host, run_id, record, other):
    return sign(host, run_id, 0, record, "rejected")


def _for_other_evidence(host, run_id, record, other):
    return sign(host, run_id, 0, record, evidence=OTHER_EVIDENCE)


def _for_another_sequence(host, run_id, record, other):
    return sign(host, run_id, 1, record)


def _for_another_record(host, run_id, record, other):
    edited = DeliveryAcceptance.from_dict({**record.to_dict(), "reasons": ["edited"]})
    return sign(host, run_id, 0, edited)


def _expired(host, run_id, record, other):
    return sign(host, run_id, 0, record, expires_at_ns=1)


def _revoked(host, run_id, record, other):
    assertion = sign(host, run_id, 0, record)
    host.revoke(assertion)
    return assertion


MISBOUND = {
    "forged": (_forged, "signature"),
    "tampered": (_tampered, "signature"),
    "another-run": (_for_another_run, "approves another digest"),
    "another-decision": (_for_another_decision, "approves another digest"),
    "other-evidence": (_for_other_evidence, "approves another digest"),
    "another-sequence": (_for_another_sequence, "approves another digest"),
    "another-record": (_for_another_record, "approves another digest"),
    "expired": (_expired, "expired"),
    "revoked": (_revoked, "revoked"),
}


@pytest.mark.parametrize("case", sorted(MISBOUND))
def test_forged_or_misbound_assertions_are_refused(host, case):
    checkpoints = MemoryStore()
    original, _, provider = review_run(checkpoints)
    # Another run with the same output and the same record content.
    other, _, _ = review_run(checkpoints, provider=provider)
    run_id = original.execution_id
    before, saves = deepcopy(checkpoints.states), len(checkpoints.saves)
    make, reason = MISBOUND[case]
    assertion = make(host, run_id, original.acceptance, other)

    with pytest.raises(ApprovalRejected, match=reason):
        _swarm(provider, store=checkpoints).reevaluate(run_id, operator=decide(host, assertion))

    assert checkpoints.states == before and len(checkpoints.saves) == saves
    assert provider.runs == {"draft": 2}


def test_a_reader_that_does_not_know_operator_decisions_refuses_the_state(host, monkeypatch):
    checkpoints = MemoryStore()
    original, policy, provider = review_run(checkpoints)
    run_id = original.execution_id
    _swarm(provider, store=checkpoints).reevaluate(
        run_id, operator=decide(host, sign(host, run_id, 0, original.acceptance)))
    completed = checkpoints.load(run_id)
    checkpoints.states["unfinished"] = {**deepcopy(completed), "status": "running"}

    # A reader that knows only the delivery acceptance feature refuses the
    # state, finished or not, before any provider call.
    with monkeypatch.context() as patch:
        patch.setattr(smythe.checkpoint, "KNOWN_CHECKPOINT_REQUIREMENTS",
                      frozenset({"delivery_acceptance"}))
        for execution_id in (run_id, "unfinished"):
            for swarm in (_swarm(provider, acceptance=policy, store=checkpoints),
                          _swarm(provider, store=checkpoints)):
                with pytest.raises(ValueError, match=r"cannot read: \['operator_decision'\]"):
                    swarm.resume(execution_id)
    # A reader that knows only version 1 records refuses the record itself.
    with monkeypatch.context() as patch:
        patch.setattr(smythe.acceptance, "SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS", (1,))
        with pytest.raises(ValueError, match="record version 2"):
            DeliveryAcceptance.from_dict(completed["delivery_acceptance"])
        with pytest.raises(ValueError, match="record version 2"):
            _swarm(provider, acceptance=policy, store=checkpoints).resume(run_id)
    assert provider.runs == {"draft": 1}
    assert checkpoints.load(run_id) == completed

    # This build refuses an operator decision that the state does not list,
    # and one that an unfinished state carries.
    unlisted = {**deepcopy(completed), "requires": ["delivery_acceptance"]}
    with pytest.raises(ValueError, match="operator_decision"):
        delivery_acceptance_from_state(unlisted)
    with pytest.raises(ValueError, match="completed checkpoint"):
        delivery_acceptance_from_state(checkpoints.load("unfinished"))
    assert delivery_acceptance_from_state(completed).operator == reference(host)


def test_a_later_reevaluation_keeps_the_operator_decision_while_the_state_holds_it(host):
    checkpoints = MemoryStore()
    original, _, provider = review_run(checkpoints)
    run_id = original.execution_id
    swarm = _swarm(provider, store=checkpoints)
    decided = swarm.reevaluate(
        run_id, operator=decide(host, sign(host, run_id, 0, original.acceptance)))

    # A re-evaluation still records a new decision; the operator's is kept.
    again = swarm.reevaluate(run_id, acceptance=_strict(Forbids()))

    assert again.accepted and again.acceptance.operator is None
    assert again.acceptance.version == 1
    state = checkpoints.states[run_id]
    assert state["requires"] == ["delivery_acceptance", "operator_decision"]
    assert delivery_acceptance_history(state) == ((0, original.acceptance),
                                                  (1, decided.acceptance))
    unlisted = {**deepcopy(state), "requires": ["delivery_acceptance"]}
    with pytest.raises(ValueError, match="operator_decision"):
        delivery_acceptance_history(unlisted)

    # Once the bounded history drops it, the state no longer needs the feature.
    for _ in range(DELIVERY_ACCEPTANCE_HISTORY_LIMIT):
        swarm.reevaluate(run_id, acceptance=_strict(Forbids()))
    state = checkpoints.states[run_id]
    assert all(record.operator is None for _, record in delivery_acceptance_history(state))
    assert state["requires"] == ["delivery_acceptance"]
    assert provider.runs == {"draft": 1}


def test_an_operator_decision_needs_a_completed_decided_run(host):
    # Advisory and legacy runs keep version 4 checkpoints and have no record.
    legacy_store = MemoryStore()
    legacy = _swarm(Script({}), store=legacy_store).execute(_draft_only())
    saved = deepcopy(legacy_store.states[legacy.execution_id])
    assert saved["version"] == 4
    record = DeliveryAcceptance("needs_review", "a" * 64, output_sha256=text_sha256(legacy.output),
                                strategy="deliverable", sources=(AcceptanceSource("draft"),),
                                reasons=("review",))
    operator = decide(host, sign(host, legacy.execution_id, 0, record))
    with pytest.raises(ValueError, match="no delivery acceptance record"):
        _swarm(Script({}), store=legacy_store).reevaluate(legacy.execution_id, operator=operator)
    assert legacy_store.states[legacy.execution_id] == saved

    # An unfinished run is resumed, not decided.
    checkpoints = MemoryStore()
    provider = _crashed_strict_run(checkpoints, _strict(Forbids()))
    unfinished = deepcopy(checkpoints.states[checkpoints.only_id()])
    with pytest.raises(ValueError, match="not a completed run"):
        _swarm(provider, store=checkpoints).reevaluate(checkpoints.only_id(), operator=operator)
    assert checkpoints.states[checkpoints.only_id()] == unfinished

    # Changed output is never decided.
    checkpoints = MemoryStore()
    original, _, provider = review_run(checkpoints)
    checkpoints.states[original.execution_id]["output"] += " (edited)"
    before = deepcopy(checkpoints.states[original.execution_id])
    operator = decide(host, sign(host, original.execution_id, 0, original.acceptance))
    with pytest.raises(ValueError, match="cannot be re-evaluated"):
        _swarm(provider, store=checkpoints).reevaluate(original.execution_id, operator=operator)
    assert checkpoints.states[original.execution_id] == before

    with pytest.raises(ValueError, match="not both"):
        _swarm(provider, store=checkpoints).reevaluate(
            original.execution_id, operator=operator, acceptance=_strict(Forbids()))
    with pytest.raises(TypeError, match="OperatorDecision"):
        _swarm(provider, store=checkpoints).reevaluate(original.execution_id, operator="accepted")
    with pytest.raises(ValueError, match="checkpoint_store"):
        Swarm(provider=provider, model="test-model").reevaluate("run", operator=operator)
    assert host.verified == [] and provider.runs == {"draft": 1}


# ---------------------------------------------------------------------------
# Records and requests
# ---------------------------------------------------------------------------


def _review_record(**overrides) -> DeliveryAcceptance:
    fields = {"decision": "needs_review", "policy_sha256": "a" * 64,
              "output_sha256": text_sha256("the report"), "strategy": "deliverable",
              "sources": (AcceptanceSource("write"),), "reasons": ("review it",)}
    return DeliveryAcceptance(**{**fields, **overrides})


def test_an_operator_record_is_a_version_2_record_that_names_its_decider():
    operator = OperatorReference("idp", "reviewer-3", "acme", "fixture-hmac/1", EVIDENCE)
    record = _review_record(decision="accepted", reasons=(), version=2, operator=operator)

    data = record.to_dict()
    assert data["version"] == 2
    assert data["operator"] == {"issuer": "idp", "subject": "reviewer-3", "tenant": "acme",
                                "verifier": "fixture-hmac/1", "evidence_sha256": EVIDENCE}
    assert DeliveryAcceptance.from_dict(data) == record
    # A policy's record keeps version 1 and its keys.
    assert "operator" not in _review_record().to_dict()
    assert _review_record().to_dict()["version"] == 1

    with pytest.raises(ValueError, match="version 2"):
        _review_record(decision="accepted", reasons=(), operator=operator)
    with pytest.raises(ValueError, match="version 2"):
        _review_record(version=2)
    with pytest.raises(ValueError, match="accepted or rejected"):
        _review_record(version=2, operator=operator)
    with pytest.raises(ValueError, match="missing keys"):
        DeliveryAcceptance.from_dict({key: value for key, value in data.items()
                                      if key != "operator"})
    with pytest.raises(ValueError, match="unknown keys"):
        DeliveryAcceptance.from_dict({**_review_record().to_dict(), "operator": data["operator"]})
    with pytest.raises(ValueError, match="unknown keys"):
        DeliveryAcceptance.from_dict({**data, "operator": {**data["operator"], "role": "admin"}})
    for name in ("issuer", "subject", "tenant", "verifier"):
        with pytest.raises(ValueError, match=name):
            OperatorReference(**{**operator.to_dict(), name: ""})
    with pytest.raises(ValueError, match="evidence_sha256"):
        OperatorReference(**{**operator.to_dict(), "evidence_sha256": "not a digest"})


def test_a_built_state_lists_the_operator_decision_it_holds():
    operator = OperatorReference("idp", "reviewer-3", "acme", "fixture-hmac/1", EVIDENCE)
    record = _review_record(decision="accepted", reasons=(), version=2, operator=operator)
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="write", label="Write")])

    def state(status):
        return build_state(execution_id="run1", status=status, model="m", graph=graph,
                           registry=Registry(), task=None, max_budget_usd=None, node_costs={},
                           output="the report", delivery_acceptance=record)

    completed = state("completed")
    assert completed["version"] == 5
    assert completed["requires"] == ["delivery_acceptance", "operator_decision"]
    assert delivery_acceptance_from_state(completed) == record
    with pytest.raises(ValueError, match="completed checkpoint"):
        delivery_acceptance_from_state(state("running"))


def test_an_operator_decision_request_is_checked_when_it_is_made(host):
    assert decide(host, "assertion").decision.value == "accepted"
    for decision in ("needs_review", "indeterminate", "not_evaluated", "approved"):
        with pytest.raises(ValueError, match="decision"):
            decide(host, "assertion", decision)
    with pytest.raises(ValueError, match="evidence_sha256"):
        decide(host, "assertion", evidence="0" * 63)
    for assertion in ("", None, b"assertion"):
        with pytest.raises(ValueError, match="assertion"):
            decide(host, assertion)

    class AsyncVerifier:
        identity = "async/1"

        async def verify(self, assertion, intent_sha256, now_ns): ...

        def still_authorized(self, principal, intent, now_ns): ...

    for verifier in (AsyncVerifier(), OfflineProvider(), HmacApprovalHost):
        with pytest.raises(TypeError):
            OperatorDecision("accepted", EVIDENCE, assertion="assertion", verifier=verifier)
    assert "assertion=" not in repr(decide(host, "secret-assertion"))


def test_the_approved_digest_binds_the_run_the_decision_and_the_evidence():
    record = _review_record()
    digest = operator_decision_sha256("run-1", 0, record, "accepted", EVIDENCE)
    assert digest == operator_decision_sha256("run-1", 0, record, "accepted", EVIDENCE)
    assert len({digest,
                operator_decision_sha256("run-2", 0, record, "accepted", EVIDENCE),
                operator_decision_sha256("run-1", 1, record, "accepted", EVIDENCE),
                operator_decision_sha256("run-1", 0, _review_record(reasons=("other",)),
                                         "accepted", EVIDENCE),
                operator_decision_sha256("run-1", 0, record, "rejected", EVIDENCE),
                operator_decision_sha256("run-1", 0, record, "accepted", OTHER_EVIDENCE)}) == 6
    for arguments in (("", 0, record, "accepted", EVIDENCE),
                      ("run-1", -1, record, "accepted", EVIDENCE),
                      ("run-1", True, record, "accepted", EVIDENCE),
                      ("run-1", 0, record.to_dict(), "accepted", EVIDENCE),
                      ("run-1", 0, record, "needs_review", EVIDENCE),
                      ("run-1", 0, record, "accepted", "digest")):
        with pytest.raises((TypeError, ValueError)):
            operator_decision_sha256(*arguments)


# ---------------------------------------------------------------------------
# Durable workflows
# ---------------------------------------------------------------------------


def review_policy():
    return strict(Raises(), on_invalid="needs_review")


def durable_review(journal, answers):
    script, _ = answers
    script["Write"] = ["the report"]
    result = durable(journal, acceptance=review_policy()).execute(single())
    assert result.acceptance.decision.value == "needs_review" and result.output == ""
    return result


def durable_operator(host, journal, run_id, decision="accepted"):
    current = current_decision(journal, run_id)
    return decide(host, sign(host, run_id, current["sequence"], current["record"], decision),
                  decision)


def test_a_durable_operator_decision_is_a_new_journaled_decision(
        store, replies, host):  # noqa: F811
    _, calls = replies
    original = durable_review(store, replies)
    run_id = original.execution_id
    before, saved = run_state(store, run_id), checkpoint_without_decision(store, run_id)
    revision = store.get_checkpoint(run_id)["revision"]
    sent = list(calls)

    # A Swarm without any policy can apply the decision.
    result = durable(store).reevaluate(run_id, operator=durable_operator(host, store, run_id))

    assert result.accepted and result.output == "the report"
    assert result.acceptance.operator == reference(host)
    assert result.acceptance.policy_sha256 == original.acceptance.policy_sha256
    assert result.workflow_accounting == original.workflow_accounting
    # No node ran and no provider was called; calls, charges, budget, counters
    # and recipe are unchanged, and only the decision differs in a new checkpoint.
    assert calls == sent
    assert run_state(store, run_id) == before
    assert checkpoint_without_decision(store, run_id) == saved
    current = store.get_checkpoint(run_id)
    assert current["revision"] == revision + 1 and current["checkpoint"]["version"] == 2
    assert current["checkpoint"]["delivery_acceptance"] == result.acceptance.to_dict()
    # The decision is a new acceptance operation; the earlier one stays.
    history = decision_history(store, run_id)
    assert [(item["sequence"], item["record"]) for item in history] == [
        (0, original.acceptance), (1, result.acceptance)]
    assert history[1]["operation_key"].startswith("acceptance/1/")
    assert history[1]["inputs"]["operator"] == {"decision": "accepted",
                                                **reference(host).to_dict()}
    assert history[1]["revision"] == current["revision"]
    assert store.audit(run_id)["ok"]

    # Delivery follows the current decision, on resume and in inspection.
    resumed = durable(store, acceptance=review_policy()).resume(run_id)
    assert resumed.accepted and resumed.acceptance == result.acceptance
    view = inspect_journal(store, run_id)
    assert view["delivery"]["decision"] == "accepted" and view["delivery"]["output_delivered"]
    assert view["decisions"]["current"] == 1
    assert calls == sent and len(host.verified) == 1

    # A later re-evaluation reads the operator's decision as current.
    again = durable(store).reevaluate(run_id, acceptance=strict(Contains("report")))
    assert again.accepted and current_decision(store, run_id)["sequence"] == 2
    assert calls == sent


def test_a_durable_operator_rejection_withholds_the_output(store, replies, host):  # noqa: F811
    original = durable_review(store, replies)
    run_id = original.execution_id

    result = durable(store).reevaluate(
        run_id, operator=durable_operator(host, store, run_id, "rejected"))

    assert result.acceptance.decision.value == "rejected" and result.acceptance.operator
    assert (result.output, result.rejected_output) == ("", "the report")
    assert durable(store, acceptance=review_policy()).resume(run_id).output == ""


def test_durable_operator_decisions_refuse_what_they_cannot_decide(
        store, replies, host):  # noqa: F811
    _, calls = replies
    reviewed = durable_review(store, replies).execution_id
    other = durable_review(store, replies).execution_id
    accepted = durable(store, acceptance=strict(Contains("report"))).execute(
        single()).execution_id
    legacy = durable(store).execute(single()).execution_id
    sent = list(calls)
    runs = (reviewed, other, accepted, legacy)
    before = {run_id: operations(store, run_id) for run_id in runs}

    # Accepted output stands.
    with pytest.raises(WorkflowStateError, match="is accepted, which stands"):
        durable(store).reevaluate(accepted, operator=durable_operator(host, store, accepted))
    assert host.verified == []
    # An assertion for another run, a forged one and an expired one are refused.
    record = current_decision(store, reviewed)["record"]
    for assertion in (sign(host, other, 0, current_decision(store, other)["record"]),
                      sign(HmacApprovalHost(issuer=host.issuer), reviewed, 0, record),
                      sign(host, reviewed, 0, record, expires_at_ns=1)):
        with pytest.raises(ApprovalRejected):
            durable(store).reevaluate(reviewed, operator=decide(host, assertion))
    # A run without a strict policy has no decision to resolve.
    with pytest.raises(WorkflowStateError, match="without a strict acceptance policy"):
        durable(store).reevaluate(legacy, operator=decide(host, sign(host, legacy, 0, record)))
    assert store.get_checkpoint(legacy)["checkpoint"]["version"] == 1
    with pytest.raises(ValueError, match="not both"):
        durable(store).reevaluate(reviewed, operator=decide(host, "assertion"),
                                  acceptance=strict(Contains()))
    assert calls == sent
    assert {run_id: operations(store, run_id) for run_id in runs} == before

    # Once decided, the operator's decision stands too.
    durable(store).reevaluate(reviewed, operator=durable_operator(host, store, reviewed))
    with pytest.raises(WorkflowStateError, match="is accepted, which stands"):
        durable(store).reevaluate(reviewed, operator=durable_operator(host, store, reviewed))
    assert calls == sent


def test_the_durable_runtime_needs_a_policy_or_an_operator_decision(store):  # noqa: F811
    runtime = durable(store)._workflow_runtime()
    with pytest.raises(WorkflowBindingError, match="strict acceptance policy"):
        asyncio.run(runtime.reevaluate("run"))


def test_a_durable_operator_decision_stopped_before_its_checkpoint_is_finished_by_the_next(
        store, replies, host, monkeypatch):  # noqa: F811
    original = durable_review(store, replies)
    run_id = original.execution_id
    operator = durable_operator(host, store, run_id)
    restore = stop_at(monkeypatch, store, "save_checkpoint", lambda *args, **kwargs: True)
    with pytest.raises(Crash):
        durable(store).reevaluate(run_id, operator=operator)
    restore()
    (stopped,) = [key for key, (kind, state) in operations(store, run_id).items()
                  if kind == "acceptance" and state == "completed"]
    assert stopped.startswith("acceptance/1/")
    # The committed but unapplied decision is not current.
    assert current_decision(store, run_id)["record"] == original.acceptance

    result = durable(store).reevaluate(run_id, operator=operator)

    assert result.accepted and current_decision(store, run_id)["operation_key"] == stopped
    assert [item["sequence"] for item in decision_history(store, run_id)] == [0, 1]
    assert store.audit(run_id)["ok"]


def test_an_older_reader_refuses_a_durable_operator_decision(
        store, replies, host, monkeypatch):  # noqa: F811
    _, calls = replies
    original = durable_review(store, replies)
    run_id = original.execution_id
    durable(store).reevaluate(run_id, operator=durable_operator(host, store, run_id))
    sent = list(calls)

    monkeypatch.setattr(smythe.acceptance, "SUPPORTED_DELIVERY_ACCEPTANCE_VERSIONS", (1,))
    with pytest.raises(WorkflowStateError, match="delivery acceptance is invalid"):
        current_decision(store, run_id)
    with pytest.raises(WorkflowStateError, match="delivery acceptance is invalid"):
        durable(store, acceptance=review_policy()).resume(run_id)
    assert calls == sent


def test_an_ordinary_swarm_passes_its_operator_decision_to_a_durable_run(
        store, replies, host):  # noqa: F811
    original = durable_review(store, replies)
    run_id = original.execution_id
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1, acceptance=strict(Contains("other")))

    result = swarm.reevaluate(run_id, operator=durable_operator(host, store, run_id))

    # The Swarm's own policy plays no part in an operator's decision.
    assert result.accepted and result.acceptance.policy_sha256 == review_policy().sha256
