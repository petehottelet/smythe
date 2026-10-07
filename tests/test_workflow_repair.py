"""A completed lineage run is repaired in place: its document published again, or decided again.

The research runs here read versioned sources from the fake external service
in ``fixture_service`` over loopback HTTP, extract from each read with a model
node given a context snapshot, and publish a brief written by local code. A
damaged document is rendered again from the run's saved synthesis and decided
again; a changed validator gets a new decision. Neither reads a source, calls
a model or runs a node, and every earlier record stays in the journal.

The last section applies whole repair plans with ``Swarm.repair``, including
items that regenerate in a successor run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import smythe
import test_workflow_reuse as successors
from fakes import Crash, crash_when
from fixture_connector import HmacApprovalHost
from fixture_service import FixtureService, serve_http
from smythe import (
    AcceptancePolicy, LineagePolicy, LocalOnly, MarkdownRenderer, OfflineProvider, RepairResult,
    SimpleArchitect, SourceStatus, SQLiteWorkflowStore, Swarm, Task, ValidationReport,
    ValidatorIdentity, WorkflowBindingError, archive_heads,
)
from smythe._canonical import sha256_hex
from smythe.acceptance import OperatorDecision, operator_decision_sha256
from smythe.cli import EXIT_OK, main
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.lineage import load_lineage, validator_set_sha256
from smythe.workflow import current_decision, decision_history
from smythe.workflow_artifact import current_artifact
from smythe.workflow_store import (
    WorkflowBudgetError, WorkflowConflictError, WorkflowStateError, _canonical, _sha,
)
from test_workflow_lineage import TEXTS, Brief, research, research_graph

PLAN = sha256_hex("repair plan")
REASON = "The published brief was damaged"
EVIDENCE = sha256_hex("review notes: the brief was read and approved")


class Counted:
    """A final validator that passes any text and records each text it checks."""

    checked: list[str] = []

    def __init__(self, version: str = "1") -> None:
        self.identity = ValidatorIdentity.for_config("counted", version, {})

    def validate(self, subject):
        Counted.checked.append(subject.text())
        return ValidationReport("output", (), validator=self.identity)


class Raises:
    """A final validator that cannot decide."""

    identity = ValidatorIdentity.for_config("raises", "1")

    def validate(self, subject):
        raise RuntimeError("validator bug")


class PlainBrief:
    """A LocalOnly renderer other than the run's."""

    filename = "brief.txt"

    def render(self, output, task):
        return f"BRIEF: {task.goal}\n{output}\n"


class Catalog:
    """Describes each source by the service's latest version, without reading it."""

    def __init__(self, service: FixtureService) -> None:
        self.service = service

    def describe(self, source_id):
        source = self.service.snapshot()["sources"].get(source_id)
        return None if source is None else SourceStatus(version=str(source["version"]),
                                                        sha256=source["sha256"])


@pytest.fixture(autouse=True)
def fresh_checks():
    Counted.checked = []


@pytest.fixture
def journal(tmp_path) -> Path:
    return tmp_path / "runs.db"


@pytest.fixture
def store(journal):
    with SQLiteWorkflowStore(journal) as value:
        yield value


@pytest.fixture
def library():
    service = FixtureService()
    records = {source_id: service.add_source(source_id, text) for source_id, text in TEXTS.items()}
    with serve_http(service) as endpoint:
        yield service, records, endpoint


@pytest.fixture
def model_calls(monkeypatch) -> list[str]:
    """The prompt of every model call the offline provider answers."""
    prompts: list[str] = []
    original = OfflineProvider.complete

    async def complete(self, system, prompt, model):
        prompts.append(prompt)
        return await original(self, system, prompt, model)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return prompts


def policy(version: str = "1", *validators, **settings) -> AcceptancePolicy:
    return AcceptancePolicy(mode="strict", final_validators=[Counted(version), *validators],
                            **settings)


def completed(store, library, tmp_path, **options):
    """A completed research run, and the Swarm that ran it."""
    options.setdefault("acceptance", policy())
    swarm = research(store, library, tmp_path, **options)
    result = swarm.execute(research_graph(library))
    return swarm, result


def rerender(swarm: Swarm, run_id: str, plan_sha256=PLAN, reason=REASON):
    """Apply a repair plan's ``rerender`` item, as a repair would."""
    return asyncio.run(swarm._arerender(run_id, plan_sha256=plan_sha256, reason=reason))


def recheck(swarm: Swarm, run_id: str, plan_sha256=PLAN, reason=REASON):
    """Apply a repair plan's ``recheck`` item, as a repair would."""
    return asyncio.run(swarm._arecheck(run_id, plan_sha256=plan_sha256, reason=reason))


def damage(path: Path, how: str) -> None:
    data = path.read_bytes()
    if how == "flip":
        middle = len(data) // 2
        path.write_bytes(data[:middle] + bytes([data[middle] ^ 0x20]) + data[middle + 1:])
    elif how == "truncate":
        path.write_bytes(data[:-1])
    else:
        path.unlink()


def journal_rows(store, run_id) -> dict[str, tuple]:
    """Each journal operation of the run by key: its kind, state and the digests of what it holds."""
    rows = store._db.execute("SELECT operation_key, kind, state, inputs_sha, result_sha "
                             "FROM workflow_operations WHERE run_id=?", (run_id,))
    return {row[0]: tuple(row[1:]) for row in rows}


def keys(store, run_id, prefix: str) -> list[str]:
    return sorted(key for key in journal_rows(store, run_id) if key.split("/")[0] == prefix)


def call_ids(store, run_id) -> list[str]:
    return [call["call_id"] for call in store.inspect_run(run_id)["calls"]]


def source_reads(service) -> dict[str, int]:
    return {source_id: item["reads"] for source_id, item in service.snapshot()["sources"].items()}


def listing(directory: Path) -> list[str]:
    return sorted(item.name for item in directory.iterdir())


def planned(swarm: Swarm, service) -> tuple[dict, str]:
    """The run's item in a repair plan made now, and the plan's digest."""
    plan = swarm.plan_repair(sources=Catalog(service)).to_dict()
    (entry,) = plan["items"]
    return entry, plan["plan_sha256"]


def tamper(store, run_id, key, change) -> None:
    """Rewrite one operation's result, keeping the row's own digest consistent."""
    operation = store.load_operation(run_id, key)
    result = change(operation["result"])
    stored = _canonical(result)
    with store._db:
        store._db.execute("UPDATE workflow_operations SET result_json=?, result_sha=? "
                          "WHERE operation_id=?", (stored, _sha(stored), operation["operation_id"]))


# ---------------------------------------------------------------------------
# A damaged document is published again
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["flip", "truncate", "delete"])
def test_a_damaged_document_is_rendered_again_and_decided_again_without_new_research(
        store, library, tmp_path, model_calls, how):
    service = library[0]
    swarm, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    assert result.accepted and Counted.checked == [document]
    original, lineage = current_artifact(store, run_id).publication, load_lineage(store, run_id)
    damage(original.path, how)
    damaged = None if how == "delete" else original.path.read_bytes()
    before, calls = journal_rows(store, run_id), call_ids(store, run_id)
    reads, prompts = source_reads(service), list(model_calls)
    Counted.checked = []

    # A rendering fault calls for rendering again, never for new research.
    entry, plan_sha256 = planned(swarm, service)
    assert entry["action"] == "rerender" and entry["status"] == "invalid"
    assert entry["reasons"] == ["artifact_missing" if how == "delete" else "artifact_changed"]
    assert entry["work"]["new_runs"] == entry["work"]["read_calls"] == 0

    record = rerender(swarm, run_id, plan_sha256)

    # The saved synthesis was rendered again into a new file under the
    # publication's own name; the damaged file keeps what it holds.
    assert (record.sequence, record.name) == (1, "document-p1.md")
    assert record.sha256 == original.sha256 and record.output_sha256 == original.output_sha256
    assert record.path.read_bytes() == document.encode("utf-8")
    if damaged is None:
        assert not original.path.exists()
    else:
        assert original.path.read_bytes() == damaged
    repair = {"plan_sha256": plan_sha256, "reason": REASON}
    assert store.load_operation(run_id, "publication/1")["inputs"] == {
        "sequence": 1, "synthesis": "synthesis", "output_sha256": original.output_sha256,
        "renderer": original.renderer, "republishes": 0, "repair": repair}
    # Nothing was researched: no read, no model call and no journal call.
    assert source_reads(service) == reads and model_calls == prompts
    assert call_ids(store, run_id) == calls
    # Every earlier operation is kept as it was.
    after = journal_rows(store, run_id)
    assert {key: after[key] for key in before} == before

    # The same bytes were decided again, as a new decision; none was copied.
    assert Counted.checked == [document]
    history = decision_history(store, run_id)
    assert [entry["sequence"] for entry in history] == [0, 1]
    assert history[0]["record"] == result.acceptance
    assert history[1]["operation_key"].startswith("acceptance/1/")
    assert history[1]["operation_key"] != history[0]["operation_key"]
    assert history[1]["inputs"]["publication"] == 1 and history[1]["inputs"]["repair"] == repair
    assert history[1]["record"].accepted
    assert history[1]["record"].output_sha256 == original.sha256

    # A new lineage record supersedes the run's record, naming the new file and decision.
    current = load_lineage(store, run_id)
    assert current.artifact["relative_path"] == f"{run_id}/document-p1.md"
    assert current.artifact["sha256"] == lineage.artifact["sha256"]
    assert current.acceptance["operation_key"] == history[1]["operation_key"]
    assert current.to_dict()["nodes"] == lineage.to_dict()["nodes"]
    superseding = store.load_operation(run_id, "lineage/1")
    assert superseding["state"] == "applied"
    assert superseding["inputs"]["supersedes"] == lineage.sha256
    assert superseding["inputs"]["repair"] == repair
    assert store.load_operation(run_id, "lineage")["state"] == "applied"

    # The run delivers its document again, and nothing is left to repair.
    artifact = current_artifact(store, run_id)
    assert artifact.publication == record and artifact.eligible
    assert artifact.superseded == (original,)
    resumed = research(store, library, tmp_path, acceptance=policy()).resume(run_id)
    assert resumed.accepted and resumed.output == document
    assert resumed.acceptance == history[1]["record"]
    entry, _ = planned(swarm, service)
    assert (entry["action"], entry["status"]) == ("reuse", "current")
    assert source_reads(service) == reads and model_calls == prompts
    assert store.audit(run_id)["ok"]


def test_republish_publishes_a_lineage_run_again_without_a_plan(store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    original = current_artifact(store, run_id).publication
    damage(original.path, "delete")

    record = swarm.republish(run_id)

    assert (record.sequence, record.name, record.sha256) == (1, "document-p1.md", original.sha256)
    assert store.load_operation(run_id, "publication/1")["inputs"]["repair"] is None
    decision = current_decision(store, run_id)
    assert decision["sequence"] == 1 and "repair" not in decision["inputs"]
    assert store.load_operation(run_id, "lineage/1")["inputs"]["repair"] is None
    assert current_artifact(store, run_id).eligible
    assert research(store, library, tmp_path, acceptance=policy()).resume(run_id).accepted


def test_a_document_damaged_again_is_published_under_the_next_name(store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    original = current_artifact(store, run_id).publication
    damage(original.path, "flip")
    first = rerender(swarm, run_id)
    repaired = load_lineage(store, run_id)
    damage(first.path, "truncate")

    second = rerender(swarm, run_id)

    assert (second.sequence, second.name) == (2, "document-p2.md")
    assert second.path.read_bytes() == document.encode("utf-8")
    assert store.load_operation(run_id, "publication/2")["inputs"]["republishes"] == 1
    assert [entry["sequence"] for entry in decision_history(store, run_id)] == [0, 1, 2]
    assert store.load_operation(run_id, "lineage/2")["inputs"]["supersedes"] == repaired.sha256
    assert load_lineage(store, run_id).artifact["relative_path"] == f"{run_id}/document-p2.md"
    assert current_artifact(store, run_id).superseded == (original, first)
    assert research(store, library, tmp_path, acceptance=policy()).resume(run_id).accepted


def test_republishing_identical_bytes_records_a_second_distinct_decision(
        store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    damage(current_artifact(store, run_id).publication.path, "delete")
    Counted.checked = []

    rerender(swarm, run_id)

    decisions = store.list_operations(run_id, "acceptance")
    assert len(decisions) == 2 and len({item["operation_key"] for item in decisions}) == 2
    assert all(item["state"] == "applied" for item in decisions)
    first, second = sorted(decisions, key=lambda item: item["applied_revision"])
    # The same bytes under the same policy, yet the validator ran again.
    assert first["result"] == {"delivery_acceptance": result.acceptance.to_dict()}
    assert Counted.checked == [document]
    assert {key: value for key, value in second["inputs"].items()
            if key not in ("sequence", "publication", "repair")} == first["inputs"]


# ---------------------------------------------------------------------------
# A changed validator gets an explicit new decision
# ---------------------------------------------------------------------------


def test_a_changed_validator_gets_a_new_decision_and_no_node_runs(
        store, library, tmp_path, model_calls):
    service = library[0]
    _, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    lineage = load_lineage(store, run_id)
    before, calls = journal_rows(store, run_id), call_ids(store, run_id)
    reads, prompts = source_reads(service), list(model_calls)
    Counted.checked = []

    # The validator's version was bumped: its earlier decision is not inherited.
    bumped = research(store, library, tmp_path, acceptance=policy("2"))
    entry, plan_sha256 = planned(bumped, service)
    assert entry["action"] == "recheck"
    assert entry["reasons"] == ["policy_changed", "validator_set_changed"]

    decided = recheck(bumped, run_id, plan_sha256)

    assert decided.accepted and decided.output == document
    assert decided.acceptance.policy_sha256 == policy("2").sha256
    assert Counted.checked == [document]
    # No node ran: no read, no model call, no journal call, no new publication.
    assert source_reads(service) == reads and model_calls == prompts
    assert call_ids(store, run_id) == calls
    assert keys(store, run_id, "publication") == ["publication/0"]
    after = journal_rows(store, run_id)
    assert {key: after[key] for key in before} == before
    # The new decision is recorded with the plan and reason; the prior one is retained.
    history = decision_history(store, run_id)
    assert [(entry["sequence"], entry["record"]) for entry in history] == [
        (0, result.acceptance), (1, decided.acceptance)]
    repair = {"plan_sha256": plan_sha256, "reason": REASON}
    assert history[1]["inputs"]["repair"] == repair and "publication" not in history[1]["inputs"]
    # The lineage now binds the new decision and validator set.
    current = load_lineage(store, run_id)
    assert current.acceptance["operation_key"] == history[1]["operation_key"]
    assert current.acceptance["policy_sha256"] == policy("2").sha256
    assert current.acceptance["validator_set_sha256"] == validator_set_sha256(
        policy("2").validator_identities)
    assert current.artifact == lineage.artifact
    superseding = store.load_operation(run_id, "lineage/1")
    assert superseding["inputs"]["supersedes"] == lineage.sha256
    assert superseding["inputs"]["repair"] == repair

    # Nothing is left to recheck, and the run resumes, under its recipe's
    # policy, with its current decision.
    assert planned(bumped, service)[0]["action"] == "reuse"
    resumed = research(store, library, tmp_path, acceptance=policy()).resume(run_id)
    assert resumed.accepted and resumed.acceptance == decided.acceptance
    assert current_artifact(store, run_id).eligible
    assert source_reads(service) == reads and model_calls == prompts


def test_a_plain_reevaluation_of_a_lineage_run_supersedes_its_lineage(store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    lineage = load_lineage(store, run_id)

    again = swarm.reevaluate(run_id, acceptance=policy("2"))

    assert again.accepted
    superseding = store.load_operation(run_id, "lineage/1")
    assert superseding["state"] == "applied" and superseding["inputs"]["repair"] is None
    assert superseding["inputs"]["supersedes"] == lineage.sha256
    assert load_lineage(store, run_id).acceptance["policy_sha256"] == policy("2").sha256
    assert "repair" not in current_decision(store, run_id)["inputs"]
    assert research(store, library, tmp_path, acceptance=policy()).resume(run_id).accepted


def test_an_operator_decision_on_a_lineage_run_supersedes_its_lineage(store, library, tmp_path):
    review = policy("1", Raises(), on_invalid="needs_review")
    swarm, result = completed(store, library, tmp_path, acceptance=review)
    run_id = result.execution_id
    assert result.acceptance.decision.value == "needs_review"
    lineage = load_lineage(store, run_id)
    assert lineage.acceptance["decision"] == "needs_review"
    host = HmacApprovalHost()
    current = current_decision(store, run_id)
    digest = operator_decision_sha256(run_id, current["sequence"], current["record"], "accepted",
                                      EVIDENCE)
    assertion = host.issue(digest, subject="reviewer-3", tenant="acme", expires_at_ns=2**62)

    decided = swarm.reevaluate(run_id, operator=OperatorDecision(
        "accepted", EVIDENCE, assertion=assertion, verifier=host))

    assert decided.accepted
    superseding = load_lineage(store, run_id)
    assert superseding.acceptance["decision"] == "accepted"
    assert superseding.acceptance["operation_key"] == current_decision(store, run_id)[
        "operation_key"]
    # The operator resolved the policy's decision, so its validators stay those of that policy.
    assert superseding.acceptance["validators"] == lineage.acceptance["validators"]
    assert superseding.acceptance["policy_sha256"] == review.sha256
    assert store.load_operation(run_id, "lineage/1")["inputs"]["supersedes"] == lineage.sha256
    assert research(store, library, tmp_path, acceptance=review).resume(run_id).accepted
    with pytest.raises(WorkflowBindingError, match="operator"):
        asyncio.run(swarm._workflow_runtime().reevaluate(
            run_id, operator=OperatorDecision("accepted", EVIDENCE, assertion=assertion,
                                              verifier=host),
            plan_sha256=PLAN, reason=REASON))


@pytest.mark.parametrize("recorded", [True, False], ids=["reuse-on", "reuse-off"])
def test_an_operator_decision_under_the_other_reuse_setting_keeps_the_run_s_lineage(
        store, library, tmp_path, recorded):
    review = policy("1", Raises(), on_invalid="needs_review")
    _, result = completed(store, library, tmp_path, acceptance=review,
                          lineage=LineagePolicy(access_scope="research-team", reuse=recorded))
    run_id = result.execution_id
    config = store.load_run(run_id)["config"]
    lineage = load_lineage(store, run_id)
    host = HmacApprovalHost()
    current = current_decision(store, run_id)
    digest = operator_decision_sha256(run_id, current["sequence"], current["record"], "accepted",
                                      EVIDENCE)
    assertion = host.issue(digest, subject="reviewer-3", tenant="acme", expires_at_ns=2**62)
    switched = LineagePolicy(access_scope="research-team", reuse=not recorded)

    # A decision reuses nothing, so the reuse setting does not bind it.
    decided = research(store, library, tmp_path, acceptance=review, lineage=switched).reevaluate(
        run_id, operator=OperatorDecision("accepted", EVIDENCE, assertion=assertion,
                                          verifier=host))

    assert decided.accepted
    superseding = load_lineage(store, run_id)
    assert superseding.acceptance["decision"] == "accepted"
    assert superseding.acceptance["policy_sha256"] == review.sha256
    assert superseding.acceptance["validators"] == lineage.acceptance["validators"]
    # Everything else is built from the run's saved recipe, not the deciding Swarm's.
    assert superseding.run["recipe_sha256"] == store.load_run(run_id)["config_sha256"]
    assert ({**superseding.to_dict(), "acceptance": None}
            == {**lineage.to_dict(), "acceptance": None})
    assert store.load_run(run_id)["config"] == config
    assert config["lineage"]["reuse"] is recorded
    # The run resumes under the policy it recorded, and only under it.
    assert research(store, library, tmp_path, acceptance=review,
                    lineage=LineagePolicy(access_scope="research-team", reuse=recorded)
                    ).resume(run_id).accepted
    with pytest.raises(WorkflowConflictError, match="lineage policy"):
        research(store, library, tmp_path, acceptance=review, lineage=switched).resume(run_id)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_republication_is_refused_when_the_document_is_intact(store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    directory = current_artifact(store, run_id).publication.path.parent
    before, revision = journal_rows(store, run_id), store.load_run(run_id)["revision"]
    Counted.checked = []

    with pytest.raises(WorkflowStateError, match="Nothing to repair"):
        rerender(swarm, run_id)
    with pytest.raises(WorkflowStateError, match="Nothing to repair"):
        swarm.republish(run_id)

    assert journal_rows(store, run_id) == before and store.load_run(run_id)["revision"] == revision
    assert listing(directory) == [".smythe-run-owner.json", "document.md"]
    assert Counted.checked == []


@pytest.mark.parametrize("change", [
    {"renderer": LocalOnly(PlainBrief, "plain-brief", "1", role="renderer")},
    {"synthesizer": LocalOnly(Brief, "brief-writer", "2", role="synthesizer")},
], ids=["renderer", "synthesizer"])
def test_republication_is_refused_when_the_renderer_or_synthesizer_changed(
        store, library, tmp_path, change):
    _, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    damage(current_artifact(store, run_id).publication.path, "delete")
    before, revision = journal_rows(store, run_id), store.load_run(run_id)["revision"]

    # Another renderer or synthesizer regenerates in a successor; it never republishes.
    changed = research(store, library, tmp_path, acceptance=policy(), **change)
    with pytest.raises(WorkflowConflictError, match="differ"):
        rerender(changed, run_id)

    assert journal_rows(store, run_id) == before and store.load_run(run_id)["revision"] == revision
    assert Counted.checked == [result.output]


def test_republication_is_refused_when_the_synthesis_digest_does_not_match_the_lineage(
        store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    damage(current_artifact(store, run_id).publication.path, "flip")
    tamper(store, run_id, "synthesis", lambda saved: {"output": saved["output"] + " Edited."})
    before, revision = journal_rows(store, run_id), store.load_run(run_id)["revision"]

    with pytest.raises(WorkflowStateError, match="synthesis.*lineage"):
        rerender(swarm, run_id)

    assert journal_rows(store, run_id) == before and store.load_run(run_id)["revision"] == revision
    assert listing(current_artifact(store, run_id).publication.path.parent) == [
        ".smythe-run-owner.json", "document.md"]


def test_a_file_with_other_bytes_under_the_publication_name_is_never_replaced(
        store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    original = current_artifact(store, run_id).publication
    damage(original.path, "delete")
    taken = original.path.with_name("document-p1.md")
    taken.write_bytes(b"another document\n")
    revision = store.load_run(run_id)["revision"]

    with pytest.raises(WorkflowStateError, match="document-p1.md"):
        rerender(swarm, run_id)
    assert taken.read_bytes() == b"another document\n"
    assert store.load_run(run_id)["revision"] == revision
    assert not current_artifact(store, run_id).eligible

    # Once the name is free, the same repair completes.
    taken.unlink()
    record = rerender(swarm, run_id)
    assert (record.sequence, record.name) == (1, "document-p1.md")
    assert record.path.read_bytes() == document.encode("utf-8")


def test_a_repair_names_a_plan_digest_and_a_reason(store, library, tmp_path):
    swarm, result = completed(store, library, tmp_path)
    run_id = result.execution_id
    damage(current_artifact(store, run_id).publication.path, "delete")
    revision = store.load_run(run_id)["revision"]

    for plan_sha256, reason in (("not a digest", REASON), (PLAN, ""), (PLAN, "line\nbreak"),
                                (PLAN, None), (None, REASON)):
        with pytest.raises(WorkflowBindingError):
            rerender(swarm, run_id, plan_sha256, reason)
        with pytest.raises(WorkflowBindingError):
            recheck(swarm, run_id, plan_sha256, reason)
    assert store.load_run(run_id)["revision"] == revision


def test_a_run_without_lineage_records_no_repair(store, tmp_path):
    swarm = Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                  run_store=store, max_budget_usd=1, acceptance=policy(),
                  renderer=MarkdownRenderer(), artifact_dir=tmp_path / "artifacts")
    graph = ExecutionGraph([Topology.SERIAL], [Node(id="draft", label="Draft the brief")])
    result = swarm.execute(graph)
    run_id = result.execution_id
    damage(current_artifact(store, run_id).publication.path, "delete")
    revision = store.load_run(run_id)["revision"]

    with pytest.raises(WorkflowStateError, match="no lineage"):
        rerender(swarm, run_id)
    with pytest.raises(WorkflowStateError, match="no lineage"):
        recheck(swarm, run_id)
    assert store.load_run(run_id)["revision"] == revision

    # Without a plan, its document is published again as before: the same
    # bytes under a free name, with its decision unchanged.
    record = swarm.republish(run_id)
    assert (record.sequence, record.name) == (1, "document.md")
    assert store.load_operation(run_id, "publication/1")["inputs"] == {
        "sequence": 1, "republishes": 0, "sha256": record.sha256, "size_bytes": record.size_bytes}
    assert [entry["sequence"] for entry in decision_history(store, run_id)] == [0]


# ---------------------------------------------------------------------------
# Crash boundaries
# ---------------------------------------------------------------------------

CRASHES = {
    "file-written": ("complete_operation",
                     lambda lease, key, result, **kwargs: key.startswith("publication/")),
    "published-not-decided": ("begin_operation",
                              lambda lease, key, kind, inputs: kind == "acceptance"),
    "decided-not-saved": ("save_checkpoint",
                          lambda lease, revision, checkpoint, **kwargs:
                          checkpoint["status"] == "completed"),
}


@pytest.mark.parametrize("boundary", sorted(CRASHES))
def test_an_interrupted_republication_converges_on_one_publication_and_one_decision(
        store, library, tmp_path, model_calls, monkeypatch, boundary):
    service = library[0]
    swarm, result = completed(store, library, tmp_path)
    run_id, document = result.execution_id, result.output
    original = current_artifact(store, run_id).publication
    damage(original.path, "flip")
    calls, reads, prompts = call_ids(store, run_id), source_reads(service), list(model_calls)
    Counted.checked = []
    name, due = CRASHES[boundary]
    stop = crash_when(monkeypatch, store, name, due)
    with pytest.raises(Crash):
        rerender(swarm, run_id)
    stop.restore()
    # Until the repair finishes, the damaged document is still the current one.
    assert current_artifact(store, run_id).publication == original
    assert not research(store, library, tmp_path, acceptance=policy()).resume(run_id).accepted

    record = rerender(swarm, run_id)

    assert (record.sequence, record.name, record.sha256) == (1, "document-p1.md", original.sha256)
    assert listing(original.path.parent) == [".smythe-run-owner.json", "document-p1.md",
                                             "document.md"]
    assert keys(store, run_id, "publication") == ["publication/0", "publication/1"]
    assert [entry["sequence"] for entry in decision_history(store, run_id)] == [0, 1]
    assert len(store.list_operations(run_id, "acceptance")) == 2
    assert keys(store, run_id, "lineage") == ["lineage", "lineage/1"]
    assert all(row[1] == "applied" for row in journal_rows(store, run_id).values())
    # The new bytes were decided once, and nothing was researched.
    assert Counted.checked == [document]
    assert call_ids(store, run_id) == calls and source_reads(service) == reads
    assert model_calls == prompts
    resumed = research(store, library, tmp_path, acceptance=policy()).resume(run_id)
    assert resumed.accepted and resumed.output == document
    assert store.audit(run_id)["ok"]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_inspection_shows_the_publication_a_republication_superseded(
        journal, library, tmp_path, capsys):
    with SQLiteWorkflowStore(journal) as store:
        swarm, result = completed(store, library, tmp_path)
        run_id = result.execution_id
        original = current_artifact(store, run_id).publication
        damage(original.path, "truncate")
        rerender(swarm, run_id)

    code = main(["runs", "--journal", str(journal), "--json", "inspect", run_id])
    value = json.loads(capsys.readouterr().out)["inspect"]
    assert code == EXIT_OK
    artifact = value["artifact"]
    assert (artifact["publication"]["sequence"], artifact["publication"]["name"]) == (
        1, "document-p1.md")
    assert artifact["superseded"] == [
        {"sequence": 0, "name": "document.md", "sha256": original.sha256}]
    assert artifact["delivery_eligibility"] == {"eligible": True, "reasons": []}
    assert value["decisions"]["current"] == 1
    assert value["lineage"]["artifact"]["relative_path"] == f"{run_id}/document-p1.md"
    assert value["lineage"]["artifact"]["integrity"] == "verified"

    assert main(["runs", "--journal", str(journal), "inspect", run_id]) == EXIT_OK
    text = capsys.readouterr().out
    assert "(publication 1)" in text
    assert "  Supersedes: publication 0 (document.md)" in text


# ---------------------------------------------------------------------------
# Applying a repair plan
# ---------------------------------------------------------------------------
#
# The runs below are the three-source research of ``test_workflow_reuse``:
# each source is read and then extracted by a model node whose reply names its
# step and the digest of its prompt, and local code writes the brief. They
# are decided under the counted validator, so its version bump calls for a
# recheck.

SCOPE = "research-team"
IDLE = {"new_runs": 0, "read_calls": 0, "model_calls": 0, "reused_nodes": 0, "publications": 0,
        "acceptance_evaluations": 0}


@pytest.fixture
def archive_sources():
    """The three sources the research reads, served over loopback HTTP."""
    service = FixtureService()
    for source_id, text in successors.TEXTS.items():
        service.add_source(source_id, text)
    with serve_http(service) as endpoint:
        yield service, endpoint


@pytest.fixture
def replies(monkeypatch) -> list[str]:
    """The label of each model step, answered with its label and its prompt's digest."""
    log: list[str] = []
    successors.install_replies(monkeypatch, log)
    return log


def maintained(store, endpoint, tmp_path, **options) -> Swarm:
    """A Swarm that runs and repairs the three-source research under the counted policy."""
    options.setdefault("acceptance", policy())
    return successors.research(store, endpoint, tmp_path, **options)


def survey_only(endpoint) -> ExecutionGraph:
    graph = successors.research_graph(endpoint)
    graph.nodes = [node for node in graph.nodes if node.id.endswith("survey")]
    return graph


def run_rows(store, run_id) -> list[str]:
    """The run's rows in the journal's SQL dump: its run, calls and their evidence,
    checkpoints, operations, invocations and events."""
    calls = [row[0] for row in store._db.execute(
        "SELECT call_id FROM workflow_calls WHERE run_id=?", (run_id,))]
    values = [f"'{value}'" for value in (run_id, *calls)]
    return [line for line in store._db.iterdump()
            if line.startswith("INSERT") and any(value in line for value in values)]


def journal_calls(store, run_id) -> list[tuple[str, str]]:
    return [(call["key"]["phase"], call["key"]["scope_id"])
            for call in store.inspect_run(run_id)["calls"]]


def repair(swarm: Swarm, plan, catalog, **changes) -> RepairResult:
    """Apply *plan* as approved, with an allowance for one new run at the research budget."""
    arguments = {"plan_sha256": plan.plan_sha256, "max_total_budget_usd": 1, "reason": REASON,
                 "sources": catalog, **changes}
    return swarm.repair(plan, **arguments)


def test_a_plan_is_applied_item_by_item_by_each_action(
        store, archive_sources, tmp_path, replies):
    service, endpoint = archive_sources
    swarm = maintained(store, endpoint, tmp_path)
    full = swarm.execute(successors.research_graph(endpoint)).execution_id
    damaged = swarm.execute(survey_only(endpoint)).execution_id
    intact = swarm.execute(survey_only(endpoint)).execution_id
    waiting = swarm.plan(Task(successors.QUESTION)).run_ref["run_id"]
    damage(current_artifact(store, damaged).publication.path, "flip")
    service.change_source("churn", successors.CHANGED)
    # The counted validator's version was bumped, too.
    bumped = maintained(store, endpoint, tmp_path, acceptance=policy("2"))
    catalog = successors.Catalog(service)
    plan = bumped.plan_repair(sources=catalog)
    assert [(item["run_id"], item["action"]) for item in plan.to_dict()["items"]] == [
        (full, "regenerate"), (damaged, "rerender"), (intact, "recheck"), (waiting, "review")]
    rows, reads, labels = run_rows(store, full), source_reads(service), len(replies)
    Counted.checked = []

    result = repair(bumped, plan, catalog)

    (successor,) = result.successor_run_ids
    items = result.to_dict()["items"]
    assert [(item["run_id"], item["action"], item["outcome"]) for item in items] == [
        (full, "regenerate", "applied"), (damaged, "rerender", "applied"),
        (intact, "recheck", "applied"), (waiting, "review", "reported")]
    assert items[0]["successor_run_id"] == successor and "successor_run_id" not in items[1]
    assert [item.get("decision") for item in items] == ["accepted", "accepted", "accepted", None]
    # Each item's work, from the journal: the successor read and extracted one
    # source and kept four results; the in-place repairs ran no node.
    assert [item["work"] for item in items] == [
        IDLE | {"new_runs": 1, "read_calls": 1, "model_calls": 1, "reused_nodes": 4,
                "publications": 1, "acceptance_evaluations": 1},
        IDLE | {"publications": 1, "acceptance_evaluations": 1},
        IDLE | {"acceptance_evaluations": 1}, IDLE]
    assert result.work == IDLE | {"new_runs": 1, "read_calls": 1, "model_calls": 1,
                                  "reused_nodes": 4, "publications": 2,
                                  "acceptance_evaluations": 3}
    assert replies[labels:] == ["Extract the churn finding"]
    assert source_reads(service) == {**reads, "churn": reads["churn"] + 1}
    assert journal_calls(store, successor) == [("tool", "node/read-churn"),
                                               ("execution", "node/extract-churn")]
    # Every decision is a new one under the bumped policy, binding the plan.
    assert set(result.decisions) == {successor, damaged, intact} and len(Counted.checked) == 3
    assert all(record.accepted and record.policy_sha256 == policy("2").sha256
               for record in result.decisions.values())
    bound = {"plan_sha256": plan.plan_sha256, "reason": REASON}
    for run_id in (damaged, intact):
        decision = current_decision(store, run_id)
        assert decision["record"] == result.decisions[run_id]
        assert decision["inputs"]["repair"] == bound
    assert load_lineage(store, successor).to_dict()["repair"] == bound
    assert result.to_dict()["decisions"][successor] == result.decisions[successor].to_dict()
    # The regenerated run's rows are as they were: the successor's calls and costs are its own.
    assert run_rows(store, full) == rows
    # Only the run that has not completed is left; the successor replaced its predecessor.
    after = bumped.plan_repair(sources=catalog).to_dict()["items"]
    assert [(item["run_id"], item["action"]) for item in after] == [
        (damaged, "reuse"), (intact, "reuse"), (waiting, "review"), (successor, "reuse")]
    assert isinstance(result, RepairResult) and "RepairResult" in smythe.__all__
    # The result is read-only, and compares and hashes like any record.
    copy = RepairResult(result.plan_sha256, items, result.successor_run_ids, result.decisions)
    assert copy == result and hash(copy) == hash(result)
    with pytest.raises(TypeError):
        result.items[0]["outcome"] = "failed"


@pytest.fixture(scope="module")
def affected(tmp_path_factory):
    """A completed run, a plan made before its churn source changed and one made after."""
    root = tmp_path_factory.mktemp("affected")
    service = FixtureService()
    for source_id, text in successors.TEXTS.items():
        service.add_source(source_id, text)
    log: list[str] = []
    with (pytest.MonkeyPatch.context() as patch, serve_http(service) as endpoint,
          SQLiteWorkflowStore(root / "runs.db") as store):
        successors.install_replies(patch, log)
        swarm = maintained(store, endpoint, root)
        first = swarm.execute(successors.research_graph(endpoint)).execution_id
        catalog = successors.Catalog(service)
        earlier = swarm.plan_repair(sources=catalog)
        service.change_source("churn", successors.CHANGED)
        plan = swarm.plan_repair(sources=catalog)
        assert plan.to_dict()["items"][0]["work"]["may_incur_charges"]
        yield SimpleNamespace(store=store, service=service, endpoint=endpoint, root=root,
                              log=log, first=first, catalog=catalog, earlier=earlier, plan=plan)


def configured(affected, **options) -> Swarm:
    return maintained(affected.store, affected.endpoint, affected.root, **options)


REFUSALS = {
    # The ordinary path has no journal, no lineage and no successor runs.
    "ordinary": (ValueError, "durable run_store", lambda case: (Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
        max_budget_usd=1, acceptance=policy()), {})),
    "lineage": (ValueError, "lineage policy", lambda case: (configured(case, lineage=None), {})),
    "supervisor": (WorkflowBindingError, "supervisor", lambda case: (configured(
        case, supervisor=LocalOnly(successors.Reviewer, "reviewer", "1", role="supervisor")), {})),
    "plan": (TypeError, "RepairPlan", lambda case: (configured(case),
                                                    {"plan": case.plan.to_dict()})),
    "digest": (ValueError, "not the digest", lambda case: (
        configured(case), {"plan_sha256": sha256_hex("another plan")})),
    "reason": (WorkflowBindingError, "reason", lambda case: (configured(case),
                                                             {"reason": "line\nbreak"})),
    # Previewed before the churn source changed: planned again now, it differs.
    "stale": (WorkflowConflictError, "stale", lambda case: (configured(case), {
        "plan": case.earlier, "plan_sha256": case.earlier.plan_sha256})),
    "allowance": (WorkflowBudgetError, "more than max_total_budget_usd", lambda case: (
        configured(case), {"max_total_budget_usd": 0.999999999})),
    "unbounded": (WorkflowBudgetError, "needs max_budget_usd", lambda case: (
        configured(case, max_budget_usd=None), {"max_total_budget_usd": 100})),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_repair_is_refused_before_anything_is_written_or_called(affected, case):
    error, match, build = REFUSALS[case]
    swarm, changes = build(affected)
    store, service = affected.store, affected.service
    arguments = {"plan": affected.plan, "plan_sha256": affected.plan.plan_sha256,
                 "max_total_budget_usd": 1, "reason": REASON, "sources": affected.catalog,
                 **changes}
    runs, rows = store.list_runs(), run_rows(store, affected.first)
    reads, labels = source_reads(service), list(affected.log)
    with pytest.raises(error, match=match):
        swarm.repair(arguments.pop("plan"), **arguments)
    # No run was created or changed, and nothing was read or called.
    assert store.list_runs() == runs and run_rows(store, affected.first) == rows
    assert source_reads(service) == reads and affected.log == labels


def test_a_failed_item_stops_the_repair_and_later_items_stay_unapplied(
        store, archive_sources, tmp_path, replies):
    service, endpoint = archive_sources
    swarm, catalog = maintained(store, endpoint, tmp_path), successors.Catalog(service)
    first = swarm.execute(survey_only(endpoint)).execution_id
    second = swarm.execute(survey_only(endpoint)).execution_id
    for run_id in (first, second):
        damage(current_artifact(store, run_id).publication.path, "delete")
    # Another file already holds the name the first run's republication takes.
    taken = current_artifact(store, first).publication.path.with_name("document-p1.md")
    taken.write_bytes(b"another document\n")
    plan = swarm.plan_repair(sources=catalog)
    assert [item["action"] for item in plan.to_dict()["items"]] == ["rerender", "rerender"]
    untouched, labels = journal_rows(store, second), len(replies)

    # Rendering again needs no new run, so no allowance.
    result = repair(swarm, plan, catalog, max_total_budget_usd=0)

    failed, unapplied = result.to_dict()["items"]
    assert (failed["run_id"], failed["outcome"]) == (first, "failed")
    assert failed["error"].startswith("WorkflowStateError") and "document-p1.md" in failed["error"]
    assert "decision" not in failed and failed["work"]["new_runs"] == 0
    assert unapplied == {"run_id": second, "action": "rerender", "outcome": "unapplied",
                         "work": IDLE}
    assert result.successor_run_ids == () and dict(result.decisions) == {}
    # The failure was not retried, and the second run was never touched.
    assert taken.read_bytes() == b"another document\n"
    assert journal_rows(store, second) == untouched
    assert not current_artifact(store, second).eligible and replies[labels:] == []
    # Once the name is free, the plan made now repairs both runs.
    taken.unlink()
    again = swarm.plan_repair(sources=catalog)
    fixed = repair(swarm, again, catalog, max_total_budget_usd=0)
    assert [item["outcome"] for item in fixed.to_dict()["items"]] == ["applied", "applied"]
    assert all(current_artifact(store, run_id).eligible for run_id in (first, second))


def test_with_reuse_off_a_repair_runs_every_affected_node_with_identical_bytes(
        store, archive_sources, tmp_path, replies):
    service, endpoint = archive_sources
    first = maintained(store, endpoint, tmp_path).execute(
        successors.research_graph(endpoint)).execution_id
    service.change_source("churn", successors.CHANGED)
    # The rollback switch: the same scope with reuse turned off.
    off = maintained(store, endpoint, tmp_path,
                     lineage=LineagePolicy(access_scope=SCOPE, reuse=False))
    catalog = successors.Catalog(service)
    plan = off.plan_repair(sources=catalog)
    (item,) = plan.to_dict()["items"]
    assert item["action"] == "regenerate" and "reuse_disabled" in item["reasons"]
    assert {node["action"] for node in item["nodes"]} == {"regenerate"}
    assert item["work"]["nodes_reused"] == {}
    labels = len(replies)

    result = repair(off, plan, catalog)

    (successor,) = result.successor_run_ids
    assert result.to_dict()["items"][0]["outcome"] == "applied"
    # Nothing is reused: every read and every extraction runs again.
    assert store.list_operations(successor, "reuse") == []
    assert sorted(replies[labels:]) == sorted(f"Extract the {source_id} finding"
                                              for source_id in successors.TEXTS)
    assert result.work == IDLE | {"new_runs": 1, "read_calls": 3, "model_calls": 3,
                                  "publications": 1, "acceptance_evaluations": 1}
    # The document is byte for byte what a new run of the changed sources publishes.
    full = maintained(store, endpoint, tmp_path).execute(successors.research_graph(endpoint))
    published = current_artifact(store, successor).publication.path.read_bytes()
    assert published == current_artifact(store, full.execution_id).publication.path.read_bytes()
    assert published == full.output.encode("utf-8")
    # Lineage stays readable: the successor names its predecessor and reuses nothing.
    lineage = load_lineage(store, successor).to_dict()
    assert lineage["predecessor"]["run_id"] == first
    assert not any("reused_from" in node for node in lineage["nodes"])
    assert load_lineage(store, first) is not None


def test_a_lost_successor_is_a_repair_in_progress_that_resumes_rather_than_restarts(
        store, archive_sources, tmp_path, replies, monkeypatch):
    service, endpoint = archive_sources
    swarm, catalog = maintained(store, endpoint, tmp_path), successors.Catalog(service)
    first = swarm.execute(successors.research_graph(endpoint)).execution_id
    service.change_source("churn", successors.CHANGED)
    plan = swarm.plan_repair(sources=catalog)
    # Lost after the successor's reuse operations committed, before it ran a node.
    stop = crash_when(monkeypatch, store, "save_checkpoint",
                      lambda lease, revision, checkpoint, **kwargs:
                      checkpoint["status"] == "running")
    with pytest.raises(Crash):
        repair(swarm, plan, catalog)
    stop.restore()
    (successor,) = [run["run_id"] for run in store.list_runs() if run["run_id"] != first]

    # The lost successor is no head; the run it repairs waits for it.
    assert archive_heads(store, SCOPE) == [first]
    again = swarm.plan_repair(sources=catalog)
    (item,) = again.to_dict()["items"]
    assert (item["run_id"], item["action"], item["status"]) == (first, "review", "stale")
    assert "repair_in_progress" in item["reasons"] and item["repairs_in_progress"] == [successor]
    assert item["work"]["new_runs"] == 0
    # The plan applied before is stale now, and the new one starts nothing.
    with pytest.raises(WorkflowConflictError, match="stale"):
        repair(swarm, plan, catalog)
    reported = repair(swarm, again, catalog, max_total_budget_usd=0)
    assert [entry["outcome"] for entry in reported.to_dict()["items"]] == ["reported"]
    assert len(store.list_runs()) == 2

    # Ordinary resume finishes the repair, and the successor replaces its predecessor.
    labels = len(replies)
    resumed = swarm.resume(successor)
    assert resumed.accepted and replies[labels:] == ["Extract the churn finding"]
    assert archive_heads(store, SCOPE) == [successor]
    (item,) = swarm.plan_repair(sources=catalog).to_dict()["items"]
    assert (item["run_id"], item["action"], item["status"]) == (successor, "reuse", "current")
    assert "repairs_in_progress" not in item


# ---------------------------------------------------------------------------
# A recheck under the reuse switch
# ---------------------------------------------------------------------------


def test_with_reuse_off_a_recheck_applies_and_the_items_after_it_still_apply(
        store, archive_sources, tmp_path, replies):
    service, endpoint = archive_sources
    intact = maintained(store, endpoint, tmp_path).execute(survey_only(endpoint)).execution_id
    full = maintained(store, endpoint, tmp_path).execute(
        successors.research_graph(endpoint)).execution_id
    service.change_source("churn", successors.CHANGED)
    # The rollback switch, with the counted validator's version bumped too.
    off = LineagePolicy(access_scope=SCOPE, reuse=False)
    swarm = maintained(store, endpoint, tmp_path, acceptance=policy("2"), lineage=off)
    catalog = successors.Catalog(service)
    plan = swarm.plan_repair(sources=catalog)
    # A changed validator alone still calls for a recheck, which reuses nothing.
    assert [(item["run_id"], item["action"]) for item in plan.to_dict()["items"]] == [
        (intact, "recheck"), (full, "regenerate")]
    Counted.checked = []

    result = repair(swarm, plan, catalog)

    items = result.to_dict()["items"]
    assert [(item["run_id"], item["action"], item["outcome"]) for item in items] == [
        (intact, "recheck", "applied"), (full, "regenerate", "applied")]
    assert items[0]["work"] == IDLE | {"acceptance_evaluations": 1}
    decision = current_decision(store, intact)
    assert decision["record"] == result.decisions[intact]
    assert decision["record"].policy_sha256 == policy("2").sha256
    assert decision["inputs"]["repair"] == {"plan_sha256": plan.plan_sha256, "reason": REASON}
    assert load_lineage(store, intact).acceptance["policy_sha256"] == policy("2").sha256
    # The regenerated item kept nothing.
    (successor,) = result.successor_run_ids
    assert store.list_operations(successor, "reuse") == []
    # The rechecked run keeps the lineage policy it recorded: resuming it
    # still needs that policy, and rendering its document again, which reuses
    # its saved synthesis, is refused with reuse off.
    assert store.load_run(intact)["config"]["lineage"] == LineagePolicy(
        access_scope=SCOPE).to_dict()
    with pytest.raises(WorkflowConflictError, match="lineage policy"):
        maintained(store, endpoint, tmp_path, lineage=off).resume(intact)
    with pytest.raises(WorkflowConflictError, match="lineage policy"):
        swarm.republish(intact)
    resumed = maintained(store, endpoint, tmp_path).resume(intact)
    assert resumed.accepted and resumed.acceptance == result.decisions[intact]
