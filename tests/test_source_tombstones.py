"""Removing a source from a durable run leaves a tombstone, and the run never uses its text again.

A tombstone keeps the source's identity, digest, reason and time and none of
its text. Records that hold the text refuse to give it back, an unfinished run
needs other sources before it resumes, and a completed run's earlier decision
becomes ``indeterminate`` until it is re-evaluated: nothing is rebuilt from
the journal's copy.
"""

from __future__ import annotations

import json

import pytest

from fakes import Crash, crash_when
from smythe import (
    AcceptanceDecision, AcceptancePolicy, ClaimPassage, ClaimRecord, ContextSource,
    DeliveryAcceptance, MarkdownRenderer, OfflineProvider, SimpleArchitect, SourceEvidence,
    SourceEvidenceValidator, SourceRecord, SQLiteWorkflowStore, Swarm, Task,
)
from smythe.cli import EXIT_OK, main
from smythe.context import SourceRemovedError, load_context_snapshot, load_context_sources
from smythe.evidence import (
    SOURCE_TOMBSTONE_OPERATION, load_source_evidence, load_source_tombstones, source_tombstone_key,
)
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider import CompletionResult
from smythe.validation import ValidationOutcome
from smythe.workflow import current_decision
from smythe.workflow_artifact import current_artifact
from smythe.workflow_binding import WorkflowBindingError
from smythe.workflow_store import WorkflowStateError
from test_acceptance_operator_decision import RecordingHost, durable_operator
from test_source_evidence import CLAIM, KEPT, MISLEADING, SOURCES, STRONG, artifact

REASON = "deleted at the owner's request"
REMOVED_AT = "2026-10-01T09:00:00Z"
# Evidence that quotes only the 2019 poll, so it holds without the audit.
POLL_ONLY = SourceEvidence((SourceRecord.from_context_source(MISLEADING).without_text(),), (
    ClaimRecord("poll", "A 2019 poll favoured annual billing.",
                supporting=(ClaimPassage("survey-2019", "most customers prefer annual billing"),)),))


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


class Calls(list):
    """Each call's step label and prompt; ``script`` maps a label to its reply."""

    def __init__(self) -> None:
        super().__init__()
        self.script: dict[str, str] = {}


@pytest.fixture
def calls(monkeypatch):
    log = Calls()

    async def complete(self, system, prompt, model):
        lines = prompt.splitlines()
        label = next((line.removeprefix("Your step: ") for line in lines
                      if line.startswith("Your step: ")), lines[0])
        log.append((label, prompt))
        return CompletionResult(log.script.get(label, f"{label} done"), cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return log


@pytest.fixture
def replies(calls):
    return calls.script


def policy(*sources) -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict",
                            final_validators=[SourceEvidenceValidator(sources or SOURCES)])


def durable(store, **options) -> Swarm:
    options.setdefault("acceptance", policy())
    options.setdefault("context_sources", SOURCES)
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def single() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [Node(id="write", label="Write")])


def tombstone_of(source: ContextSource) -> SourceRecord:
    return SourceRecord.from_context_source(source).tombstoned(REASON, removed_at=REMOVED_AT)


def remove(store, run_id, source=STRONG, **options) -> SourceRecord:
    return durable(store, **options).remove_source(run_id, source.source_id, REASON,
                                                   removed_at=REMOVED_AT)


# ---------------------------------------------------------------------------
# The tombstone
# ---------------------------------------------------------------------------


def test_removing_a_source_journals_a_tombstone_without_its_text(store, replies):
    replies["Write"] = artifact(KEPT)
    result = durable(store).execute(single())
    run_id, revision = result.execution_id, store.get_checkpoint(result.execution_id)["revision"]
    calls = store.inspect_run(run_id)["call_count"]

    tombstone = remove(store, run_id)
    assert tombstone == tombstone_of(STRONG) and tombstone.state == "tombstoned"
    assert tombstone.to_dict() == {
        "version": 1, "source_id": "renewals-2026", "source_version": "2026-q2",
        "sha256": STRONG.sha256, "text": None, "reference": None, "tombstone": REASON,
        "removed_at": REMOVED_AT,
    }
    key = source_tombstone_key("renewals-2026", STRONG.sha256)
    operation = store.load_operation(run_id, key)
    assert (operation["kind"], operation["state"]) == (SOURCE_TOMBSTONE_OPERATION, "applied")
    assert STRONG.text not in json.dumps(operation)
    # A new completed checkpoint applied it; nothing was called.
    saved = store.get_checkpoint(run_id)
    assert saved["revision"] == revision + 1 and saved["checkpoint"]["status"] == "completed"
    assert operation["operation_id"] in saved["consumed_operation_ids"]
    assert store.inspect_run(run_id)["call_count"] == calls
    assert load_source_tombstones(store, run_id) == (tombstone,)
    assert store.audit(run_id)["ok"]

    # Removing it again returns the first tombstone and writes nothing.
    again = durable(store).remove_source(run_id, "renewals-2026", "another reason")
    assert again == tombstone and store.get_checkpoint(run_id)["revision"] == revision + 1


def test_a_removed_sources_text_is_never_read_back(store, replies):
    replies["Write"] = artifact(KEPT)
    result = durable(store).execute(single())
    run_id = result.execution_id
    assert load_context_sources(store, run_id, 0) == SOURCES
    assert load_context_snapshot(store, run_id, "write").sources == SOURCES
    remove(store, run_id)

    with pytest.raises(SourceRemovedError, match="'renewals-2026' was removed from this run"):
        load_context_sources(store, run_id, 0)
    with pytest.raises(SourceRemovedError, match=REASON):
        load_context_snapshot(store, run_id, "write")
    # The evidence names the removal and keeps the other source.
    shown = load_source_evidence(store, run_id, include_text=True)
    assert [(item["source_id"], item["available"], item["text"], item["removed"])
            for item in shown["sources"]] == [
        ("survey-2019", True, MISLEADING.text, None),
        ("renewals-2026", False, None, {"reason": REASON, "removed_at": REMOVED_AT})]
    assert STRONG.text not in json.dumps(shown)


def test_evidence_that_retained_a_removed_text_does_not_return_it(store, replies):
    retained = SourceEvidence((SourceRecord.from_context_source(MISLEADING).without_text(),
                               SourceRecord.from_context_source(STRONG)), (CLAIM,))
    replies["Write"] = artifact(KEPT, retained)
    result = durable(store).execute(single())
    assert result.accepted
    remove(store, result.execution_id)
    shown = load_source_evidence(store, result.execution_id, include_text=True)
    removed = shown["sources"][1]
    assert (removed["state"], removed["available"], removed["text"]) == ("retained", False, None)
    assert removed["text_bytes"] == len(STRONG.text.encode())


# ---------------------------------------------------------------------------
# Completed runs
# ---------------------------------------------------------------------------


def test_a_decision_made_before_the_removal_no_longer_holds(store, replies):
    replies["Write"] = artifact(KEPT)
    result = durable(store).execute(single())
    run_id = result.execution_id
    assert result.accepted
    remove(store, run_id)

    resumed = durable(store).resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert (resumed.output, resumed.rejected_output) == ("", artifact(KEPT))
    assert ("source 'renewals-2026' was removed from the run after this decision "
            f"({REASON}); re-evaluate the run") in resumed.acceptance.reasons
    # The journal still holds the decision the run made.
    assert current_decision(store, run_id)["record"] == result.acceptance


@pytest.mark.parametrize("sources,reason", [
    ((MISLEADING, tombstone_of(STRONG)), "source_unavailable"),
    (SOURCES, "holds the text of the removed source 'renewals-2026'"),
], ids=["tombstone", "retained-text"])
def test_reevaluation_never_accepts_evidence_quoted_from_a_removed_source(store, replies, sources,
                                                                          reason):
    replies["Write"] = artifact(KEPT)
    run_id = durable(store).execute(single()).execution_id
    remove(store, run_id)
    again = durable(store).reevaluate(run_id, acceptance=policy(*sources))
    assert again.acceptance.decision is AcceptanceDecision.INDETERMINATE and not again.accepted
    assert reason in again.acceptance.reasons[0]
    assert again.acceptance.validators[0].outcome is ValidationOutcome.ERROR


def test_a_new_decision_that_holds_the_tombstone_stands(store, replies):
    replies["Write"] = artifact("# Billing\n\nA 2019 poll favoured annual billing.\n\n", POLL_ONLY)
    run_id = durable(store).execute(single()).execution_id
    remove(store, run_id)
    assert durable(store).resume(run_id).acceptance.decision is AcceptanceDecision.INDETERMINATE

    held = policy(MISLEADING, tombstone_of(STRONG))
    again = durable(store).reevaluate(run_id, acceptance=held)
    assert again.accepted
    # The re-evaluation came after the removal, so resuming keeps it.
    assert durable(store).resume(run_id).acceptance == again.acceptance



def test_an_operator_decision_made_after_a_removal_holds_on_resume(store, replies):
    replies["Write"] = artifact(KEPT)
    run_id = durable(store).execute(single()).execution_id
    remove(store, run_id)
    left = durable(store).reevaluate(run_id, acceptance=policy(MISLEADING, tombstone_of(STRONG)))
    assert left.acceptance.decision is AcceptanceDecision.INDETERMINATE

    decided = durable(store).reevaluate(
        run_id, operator=durable_operator(RecordingHost(), store, run_id))
    assert decided.accepted
    # The operator decided under the removal, so resuming keeps the decision.
    resumed = durable(store).resume(run_id)
    assert resumed.accepted and resumed.acceptance == decided.acceptance

def test_a_published_document_is_delivered_under_a_decision_that_holds_the_tombstone(
    tmp_path, replies, capsys,
):
    replies["Write"] = artifact("# Billing\n\nA 2019 poll favoured annual billing.\n\n", POLL_ONLY)
    journal = tmp_path / "runs.db"
    publishing = {"renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts"}
    with SQLiteWorkflowStore(journal) as store:
        result = durable(store, **publishing).execute(single())
        run_id = result.execution_id
        assert result.accepted
        remove(store, run_id, **publishing)
        assert current_artifact(store, run_id).reasons == ("not_accepted",)

        held = policy(MISLEADING, tombstone_of(STRONG))
        again = durable(store, **publishing).reevaluate(run_id, acceptance=held)
        assert again.accepted and again.output == result.output
        # The accessor, resume and inspection all deliver the document under the new decision.
        current = current_artifact(store, run_id)
        assert current.decision == again.acceptance and current.eligible
        resumed = durable(store, **publishing).resume(run_id)
        assert resumed.acceptance == again.acceptance and resumed.output == result.output
    code = main(["runs", "--journal", str(journal), "--json", "inspect", run_id])
    value = json.loads(capsys.readouterr().out)["inspect"]
    assert code == EXIT_OK and value["delivery"]["decision"] == "accepted"
    assert value["decisions"]["current"] == 1
    assert value["artifact"]["delivery_eligibility"] == {"eligible": True, "reasons": []}

    # A later removal leaves the new decision indeterminate, under the policy that made it.
    with SQLiteWorkflowStore(journal) as store:
        remove(store, run_id, MISLEADING, **publishing)
        current = current_artifact(store, run_id)
        resumed = durable(store, **publishing).resume(run_id)
    assert current.decision.decision is AcceptanceDecision.INDETERMINATE
    assert current.decision == resumed.acceptance
    assert current.decision.policy_sha256 == again.acceptance.policy_sha256
    assert current.reasons == ("not_accepted",)


def stop_reevaluation(monkeypatch, store, run_id, acceptance) -> str:
    """Re-evaluate until the decision commits, stop before its checkpoint; return its key."""
    stop = crash_when(monkeypatch, store, "save_checkpoint")
    with pytest.raises(Crash):
        durable(store).reevaluate(run_id, acceptance=acceptance)
    stop.restore()
    (stopped,) = [item for item in store.list_operations(run_id, "acceptance")
                  if item["applied_revision"] is None]
    return stopped["operation_key"]


def test_a_reevaluation_stopped_before_a_removal_decides_again(store, replies, monkeypatch):
    replies["Write"] = artifact(KEPT)
    run_id = durable(store).execute(single()).execution_id
    stopped = stop_reevaluation(monkeypatch, store, run_id, policy())
    # The stopped decision accepted the run, but no checkpoint applied it.
    assert DeliveryAcceptance.from_dict(
        store.load_operation(run_id, stopped)["result"]["delivery_acceptance"]).accepted
    remove(store, run_id)

    # The same policy still holds the removed text, so it cannot check the run,
    # and the decision stopped before the removal is never finished.
    again = durable(store).reevaluate(run_id, acceptance=policy())
    assert again.acceptance.decision is AcceptanceDecision.INDETERMINATE and not again.accepted
    assert "holds the text of the removed source 'renewals-2026'" in again.acceptance.reasons[0]
    decision = current_decision(store, run_id)
    assert decision["sequence"] == 2 and decision["operation_key"] != stopped
    assert decision["inputs"]["tombstones"] == [source_tombstone_key("renewals-2026", STRONG.sha256)]
    assert store.load_operation(run_id, stopped)["state"] == "completed"
    resumed = durable(store).resume(run_id)
    assert resumed.acceptance == again.acceptance and not resumed.accepted


def test_a_decision_stopped_before_a_removal_never_holds_after_it(store, replies, monkeypatch):
    replies["Write"] = artifact(KEPT)
    run_id = durable(store).execute(single()).execution_id
    stopped = stop_reevaluation(monkeypatch, store, run_id, policy())
    remove(store, run_id)
    # A checkpoint that applies the stopped decision after the removal, as one
    # written by a process that never saw the tombstone would.
    saved = store.get_checkpoint(run_id)
    operation = store.load_operation(run_id, stopped)
    checkpoint = dict(saved["checkpoint"], delivery_acceptance=operation["result"]["delivery_acceptance"])
    lease = store.acquire_lease(run_id, "earlier-process", ttl_s=60)
    store.save_checkpoint(lease, saved["revision"], checkpoint,
                          consumed_operation_ids=[operation["operation_id"]])
    store.release_lease(lease)
    assert current_decision(store, run_id)["operation_key"] == stopped

    resumed = durable(store).resume(run_id)
    assert resumed.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert ("source 'renewals-2026' was removed from the run after this decision "
            f"({REASON}); re-evaluate the run") in resumed.acceptance.reasons


def test_a_reevaluation_stopped_after_a_removal_is_finished_by_the_next(store, replies,
                                                                        monkeypatch):
    replies["Write"] = artifact("# Billing\n\nA 2019 poll favoured annual billing.\n\n", POLL_ONLY)
    run_id = durable(store).execute(single()).execution_id
    remove(store, run_id)
    held = policy(MISLEADING, tombstone_of(STRONG))
    stopped = stop_reevaluation(monkeypatch, store, run_id, held)

    again = durable(store).reevaluate(run_id, acceptance=held)
    assert again.accepted
    assert current_decision(store, run_id)["operation_key"] == stopped
    assert durable(store).resume(run_id).acceptance == again.acceptance


def test_a_decision_made_under_a_removal_its_checkpoint_left_unapplied_holds(store, replies,
                                                                            monkeypatch):
    replies["Write"] = artifact("# Billing\n\nA 2019 poll favoured annual billing.\n\n", POLL_ONLY)
    run_id = durable(store).execute(single()).execution_id
    # The removal commits its tombstone, then stops before its checkpoint.
    stop = crash_when(monkeypatch, store, "save_checkpoint")
    with pytest.raises(Crash):
        remove(store, run_id)
    stop.restore()
    held = policy(MISLEADING, tombstone_of(STRONG))
    again = durable(store).reevaluate(run_id, acceptance=held)
    assert again.accepted

    # The decision was made under the tombstone, so applying it later does not undo it.
    assert remove(store, run_id) == tombstone_of(STRONG)
    assert durable(store).resume(run_id).acceptance == again.acceptance


# ---------------------------------------------------------------------------
# Unfinished runs
# ---------------------------------------------------------------------------


def test_an_unfinished_run_resumes_only_with_other_sources(store, calls):
    planned = durable(store, acceptance=None).plan(Task("Write"))
    run_id = planned.run_ref["run_id"]
    remove(store, run_id, acceptance=None)
    assert store.get_checkpoint(run_id)["checkpoint"]["status"] == "planned"

    with pytest.raises(WorkflowStateError, match="give the run other sources with refresh_context"):
        durable(store, acceptance=None).resume(run_id)
    assert calls == [] and store.inspect_run(run_id)["call_count"] == 0
    # The removed text cannot come back as a refreshed source.
    with pytest.raises(WorkflowStateError, match="'renewals-2026' was removed"):
        durable(store, acceptance=None).refresh_context(run_id, (STRONG,))

    replacement = ContextSource("renewals-2026", "Renewal figures are pending review.",
                                version="2026-q3")
    assert durable(store, acceptance=None).refresh_context(run_id, (MISLEADING, replacement)) == 1
    result = durable(store, acceptance=None).resume(run_id)
    assert result.output == "Write done"
    ((label, prompt),) = calls
    assert label == "Write" and STRONG.text not in prompt and replacement.text in prompt


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_remove_source_refuses_what_it_cannot_remove(store, replies):
    replies["Write"] = artifact(KEPT)
    run_id = durable(store).execute(single()).execution_id
    with pytest.raises(WorkflowStateError, match="not one of the run's current sources"):
        durable(store).remove_source(run_id, "unknown", REASON)
    for reason in ("", "x" * 501, "two\nlines", 7):
        with pytest.raises(WorkflowBindingError, match="reason must be"):
            durable(store).remove_source(run_id, "renewals-2026", reason)
    plain = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1)
    without = plain.execute(single()).execution_id
    with pytest.raises(WorkflowStateError, match="does not use context assembly"):
        plain.remove_source(without, "renewals-2026", REASON)
    with pytest.raises(ValueError, match="needs a durable run_store"):
        Swarm(model="offline", provider=OfflineProvider()).remove_source(run_id, "x", REASON)
    assert load_source_tombstones(store, run_id) == ()
