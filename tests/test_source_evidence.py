"""Source and claim records, and the mechanical checks that hold an artifact to them."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from fakes import Crash, crash_when
from smythe import (
    AcceptanceDecision, AcceptancePolicy, ClaimPassage, ClaimRecord, ContextSnapshot,
    ContextSource, MarkdownRenderer, OfflineProvider, SimpleArchitect, SourceEvidence,
    SourceEvidenceValidator, SourceRecord, SQLiteWorkflowStore, Swarm, Task,
)
import smythe.workflow
from smythe.acceptance import ValidationSubject, check_validator, evaluate_delivery
from smythe.cli import EXIT_OK, main
from smythe.context import sources_sha256, text_sha256
from smythe.evidence import (
    EVIDENCE_BLOCK, SOURCE_EVIDENCE_OPERATION, load_source_evidence, read_source_evidence,
    render_source_evidence, source_evidence_inputs, source_evidence_key, source_evidence_result,
)
from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
from smythe.provider import CompletionResult
from smythe.synthesizer import MERGE_SYSTEM_PROMPT
from smythe.validation import FindingSeverity, ValidationOutcome
from smythe.validators import SOURCE_EVIDENCE_CHECKS, SOURCE_EVIDENCE_VALIDATOR_VERSION
from smythe.workflow_artifact import current_artifact

# A misleading source and a stronger one that contradicts it.
MISLEADING = ContextSource(
    "survey-2019", "An informal 2019 poll of 40 visitors: most customers prefer annual billing.",
    version="2019")
STRONG = ContextSource(
    "renewals-2026", "Audited renewals for 2026 show that 71% of 18,400 customers chose monthly "
    "billing over annual billing.", version="2026-q2")
SOURCES = (MISLEADING, STRONG)

CLAIM = ClaimRecord(
    "billing-preference", "Most customers prefer annual billing.",
    supporting=(ClaimPassage("survey-2019", "most customers prefer annual billing"),),
    contradicting=(ClaimPassage("renewals-2026", "71% of 18,400 customers chose monthly billing"),),
    scope="Customers on self-serve plans", unresolved=("Whom did the 2019 poll ask?",),
    conclusion="Treat the preference as contested.",
)
EVIDENCE = SourceEvidence(tuple(SourceRecord.from_context_source(source).without_text()
                                for source in SOURCES), (CLAIM,))
# The contradiction stated in the artifact's own words, and a body that drops it.
KEPT = ("# Billing\n\nA 2019 poll suggested most customers prefer annual billing, but audited "
        "renewals show that 71% of 18,400 customers chose monthly billing.\n\n")
DROPPED = "# Billing\n\nMost customers prefer annual billing.\n\n"


def artifact(body: str = KEPT, evidence: SourceEvidence = EVIDENCE) -> str:
    return body + render_source_evidence(evidence) + "\n"


def check(text: str, sources=SOURCES, spec=None):
    return SourceEvidenceValidator(sources, spec).validate(ValidationSubject.from_text(text))


def codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def evidence_with(**changes) -> SourceEvidence:
    """EVIDENCE with its claim's fields replaced."""
    return SourceEvidence(EVIDENCE.sources, (dataclasses.replace(CLAIM, **changes),))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_a_source_record_retains_text_or_names_the_source_by_version_or_digest():
    retained = SourceRecord.from_context_source(STRONG)
    assert retained.state == "retained" and retained.text == STRONG.text
    assert retained.sha256 == STRONG.sha256 == text_sha256(STRONG.text)
    assert retained.source_version == "2026-q2"
    reference = retained.without_text()
    assert reference.state == "reference" and reference.text is None
    assert (reference.source_id, reference.source_version, reference.sha256) == (
        "renewals-2026", "2026-q2", STRONG.sha256)
    assert SourceRecord("doc", source_version="v1", reference="https://example.com/doc").state == (
        "reference")
    assert SourceRecord("doc", sha256=STRONG.sha256).state == "reference"
    with pytest.raises(ValueError, match="needs its text, its sha256 or its version"):
        SourceRecord("doc", reference="https://example.com/doc")
    with pytest.raises(ValueError, match="does not match its sha256"):
        SourceRecord("doc", "other text", sha256=STRONG.sha256)


def test_a_tombstone_keeps_the_identity_and_digest_and_nothing_of_the_content():
    record = SourceRecord("renewals-2026", STRONG.text, "2026-q2", reference="https://example.com/r")
    tombstone = record.tombstoned("deleted at the owner's request", removed_at="2026-10-01T00:00Z")
    assert tombstone.state == "tombstoned"
    assert tombstone.to_dict() == {
        "version": 1, "source_id": "renewals-2026", "source_version": "2026-q2",
        "sha256": STRONG.sha256, "text": None, "reference": None,
        "tombstone": "deleted at the owner's request", "removed_at": "2026-10-01T00:00Z",
    }
    assert SourceRecord.from_dict(tombstone.to_dict()) == tombstone
    with pytest.raises(ValueError, match="cannot keep its text"):
        SourceRecord("doc", "text", tombstone="deleted")
    with pytest.raises(ValueError, match="must keep its sha256"):
        SourceRecord("doc", source_version="v1", tombstone="deleted")
    with pytest.raises(ValueError, match="cannot keep a reference"):
        SourceRecord("doc", sha256=STRONG.sha256, reference="x", tombstone="deleted")
    with pytest.raises(ValueError, match="removed_at belongs to a tombstone"):
        SourceRecord("doc", "text", removed_at="2026-10-01")


def test_records_round_trip_through_canonical_json():
    retained = SourceEvidence(tuple(SourceRecord.from_context_source(item) for item in SOURCES),
                              (CLAIM, ClaimRecord("plain", "A claim with nothing cited.")))
    for record in (retained, EVIDENCE, CLAIM, retained.sources[0]):
        data = json.loads(json.dumps(record.to_dict()))
        assert type(record).from_dict(data) == record
        assert data["version"] == 1
    assert CLAIM.to_dict() == {
        "version": 1, "claim_id": "billing-preference",
        "text": "Most customers prefer annual billing.",
        "supporting": [{"source_id": "survey-2019", "quote": "most customers prefer annual billing"}],
        "contradicting": [{"source_id": "renewals-2026",
                           "quote": "71% of 18,400 customers chose monthly billing"}],
        "scope": "Customers on self-serve plans", "unresolved": ["Whom did the 2019 poll ask?"],
        "conclusion": "Treat the preference as contested.",
    }
    # The digest is of the canonical record, so an equal record has the same one.
    assert EVIDENCE.sha256 == SourceEvidence.from_dict(EVIDENCE.to_dict()).sha256
    assert EVIDENCE.sha256 != retained.sha256


def _changed(data, path, value):
    data = json.loads(json.dumps(data))
    target = data
    for part in path[:-1]:
        target = target[part]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


@pytest.mark.parametrize("path,value,message", [
    (["version"], 2, "Unsupported source evidence version"),
    (["version"], "1", "Unsupported source evidence version"),
    (["version"], True, "Unsupported source evidence version"),
    (["sources", 0, "version"], 2, "Unsupported source record version"),
    (["claims", 0, "version"], 0, "Unsupported claim record version"),
    (["notes"], "extra", "unknown keys"),
    (["claims", 0, "verdict"], "supported", "unknown keys"),
    (["claims", 0, "supporting", 0, "page"], 3, "unknown keys"),
    (["sources", 0, "reference"], KeyError, "missing keys"),
    (["claims", 0, "unresolved"], "one question", "must be a list"),
    (["claims", 0, "text"], 7, "must be a string"),
    (["claims", 0, "contradicting", 0, "quote"], "  ", "must not be empty"),
    (["sources", 0, "sha256"], "ABC", "64 lowercase"),
    (["sources", 0, "text"], "text that is not the source", "does not match its sha256"),
    (["sources", 1, "source_id"], "survey-2019", "source ids must be distinct"),
])
def test_records_reject_unknown_versions_keys_and_values(path, value, message):
    with pytest.raises(ValueError, match=message):
        SourceEvidence.from_dict(_changed(EVIDENCE.to_dict(), path, value))


def test_records_refuse_values_of_the_wrong_kind():
    with pytest.raises(ValueError, match="claim ids must be distinct"):
        SourceEvidence(claims=(CLAIM, CLAIM))
    with pytest.raises(TypeError, match="sequence of ClaimPassage"):
        ClaimRecord("c", "claim", supporting=("survey-2019",))
    with pytest.raises(TypeError, match="sequence of questions"):
        ClaimRecord("c", "claim", unresolved="Why?")
    with pytest.raises(ValueError, match="control characters"):
        ClaimPassage("bad\nid", "quote")
    with pytest.raises(TypeError, match="sequence of SourceRecord"):
        SourceEvidence(sources=SOURCES)
    with pytest.raises(ValueError, match="valid Unicode"):
        SourceRecord("doc", "\ud800")


def test_summaries_leave_out_source_text():
    retained = SourceEvidence(tuple(SourceRecord.from_context_source(item) for item in SOURCES),
                              (CLAIM,))
    summary = retained.summary()
    assert STRONG.text not in json.dumps(summary)
    assert summary["sources"][1] == {
        "version": 1, "source_id": "renewals-2026", "source_version": "2026-q2",
        "sha256": STRONG.sha256, "reference": None, "tombstone": None, "removed_at": None,
        "text_bytes": len(STRONG.text.encode()), "state": "retained",
    }
    assert summary["claims"] == [CLAIM.to_dict()]


# ---------------------------------------------------------------------------
# The evidence block an artifact carries
# ---------------------------------------------------------------------------


def test_an_artifact_carries_one_evidence_block_beside_its_body():
    text = artifact(KEPT) + "Appendix after the evidence.\n"
    evidence, body = read_source_evidence(text)
    assert evidence == EVIDENCE
    assert body == KEPT + "Appendix after the evidence.\n"
    assert render_source_evidence(EVIDENCE).startswith(f"```{EVIDENCE_BLOCK}\n")
    # Windows line breaks and an indented, longer fence read the same.
    crlf = text.replace("\n", "\r\n")
    assert read_source_evidence(crlf) == (EVIDENCE, body.replace("\n", "\r\n"))
    fenced = KEPT + "  ````source-evidence  \n" + json.dumps(EVIDENCE.to_dict()) + "\n````\n"
    assert read_source_evidence(fenced) == (EVIDENCE, KEPT)
    assert read_source_evidence(KEPT) == (None, KEPT)


def test_evidence_blocks_follow_markdown_fences():
    block = render_source_evidence(EVIDENCE)
    # A fence line inside another fenced block is content, not a second block.
    quoted = artifact() + "~~~markdown\n" + block + "\n~~~\n"
    assert read_source_evidence(quoted)[0] == EVIDENCE
    assert read_source_evidence("```text\n" + block + "\n```\n") == (
        None, "```text\n" + block + "\n```\n")
    # A quote holding a fence cannot close the block early.
    fence = SourceEvidence(EVIDENCE.sources, (dataclasses.replace(CLAIM, text="Claims\n```\nend"),))
    assert read_source_evidence(artifact(evidence=fence))[0] == fence
    with pytest.raises(ValueError, match="2 source-evidence blocks"):
        read_source_evidence(artifact() + block + "\n")
    with pytest.raises(ValueError, match="is not closed"):
        read_source_evidence(KEPT + "```source-evidence\n{}\n")
    # A backtick fence's info string cannot hold a backtick, so this opens nothing.
    assert read_source_evidence("```source-evidence`\n{}\n```\n")[0] is None


@pytest.mark.parametrize("payload,message", [
    ('{"version": 1, "version": 1, "sources": [], "claims": []}', "duplicate key"),
    ('{"version": 1, "sources": [], "claims": [], "n": NaN}', "not strict JSON"),
    ('{"version": 1, "sources": [], "claims": []} trailing', "not strict JSON"),
    ('[]', "must be a JSON object"),
    ('{"version": 2, "sources": [], "claims": []}', "Unsupported source evidence version"),
])
def test_an_evidence_block_holds_one_strict_json_record(payload, message):
    with pytest.raises(ValueError, match=message):
        read_source_evidence(f"{KEPT}```{EVIDENCE_BLOCK}\n{payload}\n```\n")


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def test_the_validator_identity_pins_its_checks_and_each_source():
    validator = SourceEvidenceValidator(SOURCES)
    identity = validator.identity
    assert (identity.name, identity.version) == ("smythe.source-evidence", "1")
    assert SOURCE_EVIDENCE_VALIDATOR_VERSION == "1"
    assert check_validator(validator) == identity
    assert validator.spec == {"checks": list(SOURCE_EVIDENCE_CHECKS), "min_claims": 1}
    assert validator.sources == tuple(SourceRecord.from_context_source(item) for item in SOURCES)
    # The order of sources and checks does not matter; their content does.
    assert SourceEvidenceValidator(SOURCES[::-1], {"checks": SOURCE_EVIDENCE_CHECKS[::-1]}).identity == (
        identity)
    snapshot = ContextSnapshot(node_id="write", generation=0, step="Write", goal=None,
                               sources=SOURCES)
    assert SourceEvidenceValidator(snapshot).identity == identity
    edited = ContextSource("renewals-2026", STRONG.text + " Revised.", version="2026-q2")
    tombstoned = SourceRecord.from_context_source(STRONG).tombstoned("deleted")
    others = [
        SourceEvidenceValidator((MISLEADING, edited)),
        SourceEvidenceValidator((MISLEADING, tombstoned)),
        SourceEvidenceValidator((MISLEADING,)),
        SourceEvidenceValidator(SOURCES, {"checks": ["passages"]}),
        SourceEvidenceValidator(SOURCES, {"min_claims": 2}),
        SourceEvidenceValidator(SOURCES, name="evidence"),
    ]
    assert len({identity, *(other.identity for other in others)}) == 7
    # The identity names each source's digest, never its text.
    policy = AcceptancePolicy(mode="strict", final_validators=[validator])
    assert STRONG.text not in json.dumps(policy.to_dict())


@pytest.mark.parametrize("sources,spec,error,message", [
    (SOURCES, {"checks": []}, ValueError, "at least one"),
    (SOURCES, {"checks": ["passages", "support"]}, ValueError, "unknown: \\['support'\\]"),
    (SOURCES, {"checks": ["passages", "passages"]}, ValueError, "must not repeat"),
    (SOURCES, {"strict": True}, ValueError, "unknown keys"),
    (SOURCES, {"min_claims": -1}, ValueError, "min_claims must be an integer"),
    (SOURCES, {"min_claims": True}, ValueError, "min_claims must be an integer"),
    ((MISLEADING, MISLEADING), None, ValueError, "distinct"),
    ((SourceRecord("doc", source_version="v1"),), None, ValueError, "needs its sha256"),
    (("survey-2019",), None, TypeError, "ContextSource or SourceRecord"),
    ("survey-2019", None, TypeError, "sequence"),
])
def test_invalid_specifications_and_sources_are_refused(sources, spec, error, message):
    with pytest.raises(error, match=message):
        SourceEvidenceValidator(sources, spec)


def test_a_contradiction_the_evidence_lists_must_reach_the_artifact():
    kept = check(artifact(KEPT))
    assert kept.outcome is ValidationOutcome.PASSED and kept.findings == ()
    dropped = check(artifact(DROPPED))
    assert dropped.outcome is ValidationOutcome.REJECTED
    assert codes(dropped) == ["contradiction_missing"]
    assert dropped.findings[0].observed == {"claim_id": "billing-preference",
                                            "source_id": "renewals-2026"}
    assert dropped.validator == SourceEvidenceValidator(SOURCES).identity
    # The evidence block quotes the contradiction, but it is not the artifact's body.
    assert CLAIM.contradicting[0].quote in artifact(DROPPED)
    # The supporting passage is quoted from the misleading source and passes:
    # mechanical checks do not weigh how good a source is.
    assert CLAIM.supporting[0].quote in MISLEADING.text


def _source(evidence, index, **changes):
    sources = list(evidence.sources)
    sources[index] = dataclasses.replace(sources[index], **changes)
    return SourceEvidence(tuple(sources), evidence.claims)


@pytest.mark.parametrize("text,expected", [
    pytest.param(artifact(evidence=SourceEvidence(
        EVIDENCE.sources + (SourceRecord("rumour", source_version="1"),), EVIDENCE.claims)),
        ["source_unknown"], id="unknown-source"),
    pytest.param(artifact(evidence=_source(EVIDENCE, 1, source_version="2025-q4")),
                 ["source_version_mismatch"], id="other-version"),
    pytest.param(artifact(evidence=_source(EVIDENCE, 1, sha256="0" * 64)),
                 ["source_digest_mismatch"], id="other-digest"),
    pytest.param(artifact(evidence=evidence_with(supporting=(
        ClaimPassage("survey-2019", "most customers favor annual billing"),))),
        ["passage_not_found"], id="paraphrased-quote"),
    pytest.param(artifact(evidence=evidence_with(supporting=(
        ClaimPassage("survey-2019", "most  customers prefer annual billing"),))),
        ["passage_not_found"], id="respaced-quote"),
    pytest.param(artifact(evidence=evidence_with(supporting=(
        ClaimPassage("renewals-2026", "most customers prefer annual billing"),))),
        ["passage_not_found"], id="quote-from-another-source"),
    pytest.param(artifact(evidence=SourceEvidence(EVIDENCE.sources[1:], EVIDENCE.claims)),
                 ["reference_dangling"], id="unlisted-source"),
    pytest.param(artifact(evidence=evidence_with(supporting=(
        ClaimPassage("rumour", "most customers prefer annual billing"),))),
        ["source_unknown", "reference_dangling", "passage_not_found"], id="invented-source"),
    pytest.param(artifact(evidence=SourceEvidence(EVIDENCE.sources)), ["claims_too_few"],
                 id="no-claims"),
    pytest.param(KEPT, ["evidence_missing"], id="no-block"),
    pytest.param(KEPT + "```source-evidence\n{\"version\": 1}\n```\n", ["evidence_invalid"],
                 id="invalid-block"),
])
def test_each_mechanical_check_rejects_its_defect(text, expected):
    report = check(text)
    assert report.outcome is ValidationOutcome.REJECTED
    assert codes(report) == expected
    assert all(finding.severity is FindingSeverity.HARD for finding in report.findings)


def test_evidence_that_restates_a_source_text_is_held_to_its_digest():
    # A retained text in the evidence carries its own digest, which must match.
    invented = SourceRecord("renewals-2026", "71% of 18,400 customers chose monthly billing",
                            "2026-q2")
    report = check(artifact(evidence=SourceEvidence((EVIDENCE.sources[0], invented),
                                                    EVIDENCE.claims)))
    assert codes(report) == ["source_digest_mismatch"]


def test_bytes_that_are_not_utf8_are_rejected():
    report = SourceEvidenceValidator(SOURCES).validate(ValidationSubject(b"\xff" + artifact().encode()))
    assert report.outcome is ValidationOutcome.REJECTED and codes(report) == ["not_utf8"]


def test_selected_checks_run_alone():
    passages = check(artifact(DROPPED), spec={"checks": ["passages"]})
    assert passages.passed
    contradictions = check(artifact(evidence=SourceEvidence(EVIDENCE.sources[1:], EVIDENCE.claims)),
                           spec={"checks": ["contradictions"]})
    assert contradictions.passed
    # Evidence without claims passes only when the specification allows it.
    assert check(artifact(evidence=SourceEvidence(EVIDENCE.sources)), spec={"min_claims": 0}).passed
    two = check(artifact(), spec={"min_claims": 2})
    assert codes(two) == ["claims_too_few"] and two.findings[0].observed == 1


def test_a_tombstoned_source_cannot_be_checked_and_is_never_accepted():
    gone = (MISLEADING, SourceRecord.from_context_source(STRONG).tombstoned(
        "deleted at the owner's request", removed_at="2026-10-01"))
    report = check(artifact(KEPT), sources=gone)
    assert report.outcome is ValidationOutcome.ERROR and not report.passed
    assert codes(report) == ["source_unavailable"]
    assert report.findings[0].observed == {"source_id": "renewals-2026", "state": "tombstoned"}
    assert "deleted at the owner's request" in report.findings[0].message
    # The tombstone keeps the digest, so version and digest checks still hold.
    assert check(artifact(evidence=_source(EVIDENCE, 1, sha256="0" * 64)), sources=gone).outcome is (
        ValidationOutcome.REJECTED)
    # A rejection found elsewhere still decides.
    rejected = check(artifact(DROPPED), sources=gone)
    assert rejected.outcome is ValidationOutcome.REJECTED
    assert codes(rejected) == ["contradiction_missing", "source_unavailable"]
    # Strict acceptance never accepts output it could not check.
    policy = AcceptancePolicy(mode="strict", final_validators=[SourceEvidenceValidator(gone)])
    node = Node(id="write", label="Write", status=NodeStatus.COMPLETED, result=artifact(KEPT))
    record = evaluate_delivery(policy, ExecutionGraph([Topology.SERIAL], [node]), artifact(KEPT),
                               strategy="deliverable", sources=[node])
    assert record.decision is AcceptanceDecision.INDETERMINATE
    assert "source_unavailable" in record.reasons[0]


def test_a_report_lists_a_bounded_number_of_findings():
    claims = tuple(ClaimRecord(f"claim-{index}", "A claim.", contradicting=(
        ClaimPassage("renewals-2026", "71% of 18,400 customers chose monthly billing"),))
        for index in range(40))
    report = check(artifact(DROPPED, SourceEvidence(EVIDENCE.sources, claims)))
    assert report.outcome is ValidationOutcome.REJECTED and len(report.findings) == 32
    assert codes(report)[:31] == ["contradiction_missing"] * 31
    assert report.findings[-1].code == "findings_truncated" and report.findings[-1].observed == 9


def test_evidence_handling_leaves_the_merge_prompt_unchanged():
    assert hashlib.sha256(MERGE_SYSTEM_PROMPT.encode()).hexdigest() == (
        "3a4f21833a9a0c65acc00160de23e8333dde48ef9ac633212944b023a1efb380")


# ---------------------------------------------------------------------------
# Durable text workflows
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    with SQLiteWorkflowStore(tmp_path / "runs.db") as value:
        yield value


@pytest.fixture
def replies(monkeypatch):
    """Answer each node from a per-label queue whose last reply repeats; log every call."""
    script: dict[str, list[str]] = {}
    calls: list[str] = []

    async def complete(self, system, prompt, model):
        step = next(line for line in prompt.splitlines() if line.startswith("Your step: "))
        label = step.removeprefix("Your step: ")
        calls.append(label)
        queue = script.get(label) or [f"{label} done"]
        return CompletionResult(queue.pop(0) if len(queue) > 1 else queue[0], cost_usd=0)

    monkeypatch.setattr(OfflineProvider, "complete", complete)
    return script, calls


def durable(store, validator=None, **options) -> Swarm:
    validators = [validator or SourceEvidenceValidator(SOURCES)]
    options.setdefault("acceptance", AcceptancePolicy(mode="strict", final_validators=validators))
    options.setdefault("context_sources", SOURCES)
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1, **options)


def single() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [Node(id="write", label="Write")])


def operations(store, run_id) -> dict[str, tuple[str, str]]:
    rows = store._db.execute(
        "SELECT operation_key, kind, state FROM workflow_operations WHERE run_id=?", (run_id,))
    return {row[0]: (row[1], row[2]) for row in rows}


def evidence_operations(store, run_id) -> list[str]:
    return [key for key, (kind, _) in operations(store, run_id).items()
            if kind == SOURCE_EVIDENCE_OPERATION]


def test_a_durable_run_withholds_an_artifact_that_drops_a_listed_contradiction(store, replies):
    script, _ = replies
    script["Write"] = [artifact(DROPPED)]
    result = durable(store).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.REJECTED and not result.accepted
    assert (result.output, result.rejected_output) == ("", artifact(DROPPED))
    assert "contradiction_missing" in result.acceptance.reasons[0]
    # The evidence the output carried is journaled with the run's source set,
    # and the completed checkpoint consumes it with the decision.
    run_id = result.execution_id
    (key,) = evidence_operations(store, run_id)
    operation = store.load_operation(run_id, key)
    binding = {"revision": 0, "sha256": sources_sha256(SOURCES)}
    assert operation["inputs"] == source_evidence_inputs(artifact(DROPPED), binding)
    assert key == source_evidence_key(operation["inputs"])
    assert operation["state"] == "applied"
    assert operation["result"] == {"source_evidence": EVIDENCE.to_dict(), "problem": None}
    assert operation["operation_id"] in store.get_checkpoint(run_id)["consumed_operation_ids"]
    assert store.audit(run_id)["ok"]


def test_a_durable_run_delivers_an_artifact_that_keeps_the_contradiction(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT)]
    result = durable(store).execute(single())
    assert result.accepted and result.output == artifact(KEPT)
    assert [item.validator for item in result.acceptance.validators] == [
        SourceEvidenceValidator(SOURCES).identity]

    # Inspection shows the claims and each source's identity, never its text.
    loaded = load_source_evidence(store, result.execution_id)
    assert loaded["state"] == "applied" and loaded["problem"] is None
    assert loaded["output_sha256"] == text_sha256(artifact(KEPT))
    assert loaded["context_sources"] == {"revision": 0, "sha256": sources_sha256(SOURCES)}
    assert loaded["claims"] == [CLAIM.to_dict()]
    assert [(item["source_id"], item["sha256"], item["state"], item["available"])
            for item in loaded["sources"]] == [
        ("survey-2019", MISLEADING.sha256, "reference", True),
        ("renewals-2026", STRONG.sha256, "reference", True)]
    assert all(source.text not in json.dumps(loaded) for source in SOURCES)
    # The text the run retained is shown only when asked for.
    shown = load_source_evidence(store, result.execution_id, include_text=True)
    assert [item["text"] for item in shown["sources"]] == [MISLEADING.text, STRONG.text]
    # A completed run replays its evidence and decision without another call.
    again = durable(store).resume(result.execution_id)
    assert again.acceptance == result.acceptance
    assert len(evidence_operations(store, result.execution_id)) == 1


def test_inspection_marks_sources_the_run_does_not_hold(store, replies):
    script, _ = replies
    stated = SourceEvidence((
        EVIDENCE.sources[0],
        SourceRecord.from_context_source(STRONG).tombstoned("deleted"),
        SourceRecord("renewals-2026-draft", source_version="draft"),
    ), EVIDENCE.claims)
    script["Write"] = [artifact(KEPT, stated)]
    result = durable(store).execute(single())
    # The listed draft is not a source the run was given.
    assert not result.accepted and "source_unknown" in result.acceptance.reasons[0]
    shown = load_source_evidence(store, result.execution_id, include_text=True)
    assert [(item["state"], item["available"], item["text"]) for item in shown["sources"]] == [
        ("reference", True, MISLEADING.text), ("tombstoned", False, None),
        ("reference", False, None)]


def test_output_without_evidence_records_why(store, replies):
    script, _ = replies
    script["Write"] = [KEPT]
    result = durable(store).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    assert "evidence_missing" in result.acceptance.reasons[0]
    loaded = load_source_evidence(store, result.execution_id)
    assert loaded["problem"] == "the output carries no source-evidence block"
    assert (loaded["sources"], loaded["claims"]) == ([], [])


def test_a_tombstoned_source_leaves_a_durable_decision_indeterminate(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT)]
    gone = SourceEvidenceValidator((MISLEADING, SourceRecord.from_context_source(STRONG).tombstoned(
        "deleted at the owner's request")))
    result = durable(store, gone).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert (result.output, result.rejected_output) == ("", artifact(KEPT))
    assert "source_unavailable" in result.acceptance.reasons[0]


def test_the_recorded_problem_keeps_no_account_identifier(store, replies):
    organization = "org-" + "A1b2" * 6  # built here, so no tracked file holds an identifier
    unreadable = KEPT + f'```{EVIDENCE_BLOCK}\n{{"{organization}": 1, "{organization}": 2}}\n```\n'
    problem = source_evidence_result(unreadable)["problem"]
    assert problem == ("the source-evidence block is not strict JSON: duplicate key "
                       "'org-[redacted]'")
    script, _ = replies
    script["Write"] = [unreadable]
    result = durable(store).execute(single())
    loaded = load_source_evidence(store, result.execution_id)
    assert loaded["problem"] == problem and loaded["claims"] == []


def test_runs_without_an_evidence_validator_record_no_evidence(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT)]
    plain = durable(store, acceptance=None).execute(single())
    assert plain.acceptance is None and evidence_operations(store, plain.execution_id) == []
    assert load_source_evidence(store, plain.execution_id) is None
    planned = durable(store).plan(Task("Write"))
    assert load_source_evidence(store, planned.run_ref["run_id"]) is None


def test_evidence_is_read_once_and_replayed_after_a_crash(store, replies, monkeypatch):
    script, calls = replies
    script["Write"] = [artifact(KEPT)]
    reads = []
    original = smythe.workflow.source_evidence_result

    def counted(output):
        reads.append(output)
        return original(output)

    monkeypatch.setattr(smythe.workflow, "source_evidence_result", counted)
    planned = durable(store).plan(Task("Write"))
    run_id = planned.run_ref["run_id"]
    stop = crash_when(monkeypatch, store, "complete_operation",
                      lambda lease, key, *args, **kwargs: key.startswith("acceptance/"))
    failed = crash_when(monkeypatch, store, "save_checkpoint", lambda *args, **kwargs: stop.fired)
    with pytest.raises(Crash):
        durable(store).execute(planned)
    failed.restore()
    stop.restore()
    (key,) = evidence_operations(store, run_id)
    assert operations(store, run_id)[key] == (SOURCE_EVIDENCE_OPERATION, "completed")
    assert reads == [artifact(KEPT)]
    sent = list(calls)

    result = durable(store).resume(run_id)
    assert result.accepted and calls == sent and reads == [artifact(KEPT)]
    assert operations(store, run_id)[key] == (SOURCE_EVIDENCE_OPERATION, "applied")


# A revised audit the run was never given, and an artifact whose evidence cites it.
REVISED = ContextSource(
    "renewals-2026", "Audited renewals for 2026 show that 71% of 18,400 customers chose monthly "
    "billing over annual billing. Revised in October.", version="2026-q3")
REVISED_EVIDENCE = SourceEvidence(tuple(SourceRecord.from_context_source(source).without_text()
                                        for source in (MISLEADING, REVISED)), (CLAIM,))


def test_evidence_checked_against_sources_the_run_was_not_given_is_never_accepted(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT, REVISED_EVIDENCE)]
    # The validator's sources pass this evidence, but the run's nodes saw the 2026-q2 audit.
    stale = SourceEvidenceValidator((MISLEADING, REVISED))
    assert check(artifact(KEPT, REVISED_EVIDENCE), sources=(MISLEADING, REVISED)).passed
    result = durable(store, stale).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE and not result.accepted
    assert (result.output, result.rejected_output) == ("", artifact(KEPT, REVISED_EVIDENCE))
    (outcome,) = result.acceptance.validators
    assert outcome.validator == stale.identity and outcome.outcome is ValidationOutcome.ERROR
    reason = result.acceptance.reasons[0]
    assert "checks other sources than the run's current source set (revision 0)" in reason
    assert "'renewals-2026'" in reason
    # The decision is journaled, so resuming the completed run gives it again.
    assert durable(store, stale).resume(result.execution_id).acceptance == result.acceptance


def test_a_run_given_new_sources_needs_a_validator_for_them(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT)]
    planned = durable(store).plan(Task("Write"))
    run_id = planned.run_ref["run_id"]
    assert durable(store).refresh_context(run_id, (MISLEADING, REVISED)) == 1
    # The recipe's validator still checks the sources of revision 0.
    result = durable(store).resume(run_id)
    assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
    assert "(revision 1)" in result.acceptance.reasons[0]
    # Re-evaluating under a validator for the run's current sources decides again.
    current = SourceEvidenceValidator((MISLEADING, REVISED))
    policy = AcceptancePolicy(mode="strict", final_validators=[current])
    again = durable(store).reevaluate(run_id, acceptance=policy)
    assert again.acceptance.decision is AcceptanceDecision.REJECTED
    assert "source_version_mismatch" in again.acceptance.reasons[0]


def test_a_published_document_reevaluated_for_new_sources_is_delivered(tmp_path, replies, capsys):
    script, _ = replies
    script["Write"] = [artifact(KEPT, REVISED_EVIDENCE)]
    journal = tmp_path / "runs.db"
    publishing = {"renderer": MarkdownRenderer(), "artifact_dir": tmp_path / "artifacts"}
    with SQLiteWorkflowStore(journal) as store:
        run_id = durable(store, **publishing).plan(Task("Write")).run_ref["run_id"]
        assert durable(store, **publishing).refresh_context(run_id, (MISLEADING, REVISED)) == 1
        result = durable(store, **publishing).resume(run_id)
        assert result.acceptance.decision is AcceptanceDecision.INDETERMINATE
        assert current_artifact(store, run_id).reasons == ("not_accepted",)

        current = SourceEvidenceValidator((MISLEADING, REVISED))
        again = durable(store, **publishing).reevaluate(
            run_id, acceptance=AcceptancePolicy(mode="strict", final_validators=[current]))
        assert again.accepted and again.output == result.rejected_output
        # The accessor, resume and inspection all deliver the document under the new decision.
        published = current_artifact(store, run_id)
        assert published.decision == again.acceptance and published.eligible
        resumed = durable(store, **publishing).resume(run_id)
        assert resumed.acceptance == again.acceptance and resumed.output == again.output
    code = main(["runs", "--journal", str(journal), "--json", "inspect", run_id])
    value = json.loads(capsys.readouterr().out)["inspect"]
    assert code == EXIT_OK and value["delivery"]["decision"] == "accepted"
    assert value["decisions"]["current"] == 1
    assert value["artifact"]["delivery_eligibility"] == {"eligible": True, "reasons": []}


def test_reevaluating_under_a_validator_for_other_sources_is_never_accepted(store, replies):
    script, _ = replies
    script["Write"] = [artifact(KEPT, REVISED_EVIDENCE)]
    result = durable(store).execute(single())
    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    stale = AcceptancePolicy(mode="strict",
                             final_validators=[SourceEvidenceValidator((MISLEADING, REVISED))])
    again = durable(store).reevaluate(result.execution_id, acceptance=stale)
    assert again.acceptance.decision is AcceptanceDecision.INDETERMINATE and not again.accepted
    assert "checks other sources than the run's current source set" in again.acceptance.reasons[0]


def test_a_judge_still_decides_whether_evidence_supports_a_claim(store, replies):
    script, _ = replies
    script["Draft"] = [artifact(KEPT)]
    script["Judge"] = ["FAIL: the support comes from an informal poll"]
    graph = ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Draft"),
        Node(id="judge", label="Judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])
    result = durable(store).execute(graph)
    # Every mechanical check passed, and the judge's verdict still rejects.
    assert [item.outcome for item in result.acceptance.validators] == [ValidationOutcome.PASSED]
    assert result.acceptance.decision is AcceptanceDecision.REJECTED
    assert any("informal poll" in reason for reason in result.acceptance.reasons)
