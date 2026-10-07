"""From sources to an accepted artifact: plan, read, interrupt, resume and inspect.

    python examples/16_source_to_artifact.py

A durable research run answers one question from three of the bundled
sources (examples/source_to_artifact/sources/). It is planned first, so the
graph can be inspected before anything runs. Each source is then fetched by a
journaled read step from a small HTTP server inside this process, on the
loopback address (examples/_source_library.py): there is no other network
traffic and no API key.

The run stops part way, as if the process died, after its second read is
saved in the journal but before the run checkpoints it. Resuming replays that
saved read instead of fetching the source again, reads the last source, and
writes a brief whose claims quote the sources and carry a source-evidence
block. A strict acceptance policy checks the published document's exact
bytes: every quote must be verbatim in the source the run was given, and the
contradiction the evidence lists must reach the brief. Inspection then shows
the accepted document, the reads, the evidence and the questions left open,
without printing source text or response bodies.

In a live run, model steps would draft the brief. Here a deterministic local
synthesizer writes it from the sources it read; it is declared with LocalOnly,
so the run's recipe records its identity and version.

Docs: docs/source-evidence.md and docs/runs.md
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from _source_library import library_profile, read_sources, source_library

from smythe import (
    AcceptancePolicy, ClaimPassage, ClaimRecord, ContextSource, LocalOnly, MarkdownRenderer,
    OfflineProvider, ReadProfile, SourceEvidence, SourceEvidenceValidator, SourceRecord,
    SQLiteWorkflowStore, Swarm, Task, TextValidator,
)
from smythe.cli import main as smythe_cli
from smythe.evidence import load_source_evidence, render_source_evidence
from smythe.graph import ExecutionGraph, Topology
from smythe.registry import Registry
from smythe.workflow_artifact import current_artifact

for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

VERSIONS = {"survey-2019": "2019-spring", "renewals-2026": "2026-q2", "support-2026": "2026-09"}
QUESTION = "Should new subscriptions default to monthly or annual billing?"

# The analyst's reading of the sources: each claim and the exact passages it rests on.
# The synthesizer below checks every passage against the text the run read.
CLAIMS = (
    ClaimRecord(
        "billing-preference", "Most customers choose monthly billing.",
        supporting=(ClaimPassage("renewals-2026",
                                 "71% chose monthly billing over annual billing"),
                    ClaimPassage("support-2026",
                                 "mostly asked to move from annual to monthly billing")),
        contradicting=(ClaimPassage("survey-2019", "most customers prefer annual billing"),),
        scope="Self-serve subscriptions",
        unresolved=("Whom did the 2019 poll ask, and would they answer the same way today?",),
        conclusion="Default new subscriptions to monthly billing.",
    ),
    ClaimRecord(
        "renewal-rate", "Monthly billing does not lower renewal rates.",
        supporting=(ClaimPassage("renewals-2026", "Customers on monthly plans renewed at the "
                                 "same rate as customers on annual plans"),),
        unresolved=("Why do customers ask to move from annual to monthly billing?",),
    ),
)


def load_sources() -> tuple[ContextSource, ...]:
    """The bundled sources, with their versions; the run is given exactly these."""
    return read_sources(VERSIONS)


# ---------------------------------------------------------------------------
# Local components: the plan and the brief
# ---------------------------------------------------------------------------


class ReadingPlan:
    """Plans one read step per source; a LocalOnly architect, so it may declare reads."""

    def __init__(self, profile: ReadProfile, source_ids: list[str]) -> None:
        self.profile, self.source_ids = profile, source_ids

    def plan(self, task: Task):
        nodes = [self.profile.step(f"read-{source_id}", {"source_id": source_id},
                                   label=f"Read {source_id}")
                 for source_id in self.source_ids]
        return ExecutionGraph([Topology.FORK_JOIN], nodes), Registry()


class EvidenceBrief:
    """Writes the brief from the read sources, with its claims as source evidence.

    Deterministic local code: a passage that is not in the text the run read
    is left out, so the brief never quotes what it did not read.
    """

    def __init__(self, versions: dict[str, str]) -> None:
        self.versions = versions

    def synthesize(self, graph: ExecutionGraph) -> str:
        read = {node.metadata["read_step"]["input"]["source_id"]: node.result
                for node in graph.nodes if "read_step" in node.metadata}
        claims = [claim for claim in CLAIMS
                  if all(passage.quote in read.get(passage.source_id, "")
                         for _, _, passage in SourceEvidence(claims=(claim,)).passages())]
        findings = [f"- {claim.text} " + "; ".join(
            f'"{passage.quote}" [{passage.source_id}]' for passage in claim.supporting)
            for claim in claims]
        contradictions = [f'- Against "{claim.text}": "{passage.quote}" [{passage.source_id}]'
                          for claim in claims for passage in claim.contradicting]
        questions = [f"- {question}" for claim in claims for question in claim.unresolved]
        conclusions = [claim.conclusion for claim in claims if claim.conclusion]
        evidence = SourceEvidence(tuple(
            SourceRecord.from_context_source(ContextSource(
                source_id, text, self.versions[source_id])).without_text()
            for source_id, text in sorted(read.items())), tuple(claims))
        return "\n\n".join([
            "## Findings", "\n".join(findings),
            "## Contradictions", "\n".join(contradictions),
            "## Open questions", "\n".join(questions),
            "## Recommendation", " ".join(conclusions),
            render_source_evidence(evidence),
        ])


# ---------------------------------------------------------------------------
# A simulated crash
# ---------------------------------------------------------------------------


class PowerCut(BaseException):
    """Stands in for the process dying: nothing catches it and nothing more is written."""


@contextmanager
def power_cut_when_checkpointing(store: SQLiteWorkflowStore, node_id: str):
    """Stop the run as the journal checkpoints *node_id*'s completion, and write nothing after."""
    save, cut = store.save_checkpoint, []

    def save_checkpoint(lease, revision, checkpoint, **kwargs):
        if cut or any(node["id"] == node_id and node["status"] == "completed"
                      for node in checkpoint["graph"]["nodes"]):
            cut.append(node_id)
            raise PowerCut(f"power lost while checkpointing {node_id}")
        return save(lease, revision, checkpoint, **kwargs)

    store.save_checkpoint = save_checkpoint
    try:
        yield cut
    finally:
        del store.save_checkpoint


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main() -> None:
    print("Running offline: bundled sources, a loopback source library, OfflineProvider, no keys.\n")
    sources = load_sources()
    ids = [source.source_id for source in sources]
    with source_library(sources) as (port, answered), TemporaryDirectory(prefix="smythe-") as work:
        journal = Path(work) / "runs.db"
        profile = library_profile(port)
        policy = AcceptancePolicy(mode="strict", final_validators=[
            SourceEvidenceValidator(sources),
            TextValidator({"sections": ["Findings", "Contradictions", "Open questions"]}),
        ])
        with SQLiteWorkflowStore(journal) as store:
            swarm = Swarm(
                model="offline", provider=OfflineProvider(), run_store=store, max_budget_usd=0.10,
                max_concurrency=1, read_tools=[profile], context_sources=sources,
                architect=LocalOnly(lambda: ReadingPlan(profile, ids), "reading-plan", "1"),
                synthesizer=LocalOnly(lambda: EvidenceBrief(VERSIONS), "evidence-brief", "1",
                                      role="synthesizer"),
                renderer=MarkdownRenderer(), artifact_dir=Path(work) / "artifacts", acceptance=policy,
            )

            print("=== 1. Plan ===")
            graph = swarm.plan(Task(QUESTION))
            run_id = graph.run_ref["run_id"]
            for node in graph.nodes:
                step = node.metadata["read_step"]
                print(f"  {node.id}: read {step['input']['source_id']} with {step['profile']} "
                      f"version {profile.version}")
            print(f"  run {run_id}: planned; no source has been read\n")

            print("=== 2. Run, interrupted ===")
            with power_cut_when_checkpointing(store, f"read-{ids[1]}") as cut:
                try:
                    swarm.execute(graph)
                except PowerCut as stop:
                    print(f"  stopped: {stop}")
            assert cut, "the run should have been interrupted"
            saved = [call for call in store.inspect_run(run_id)["calls"]
                     if call["result_state"] in ("accepted", "applied")]
            print(f"  the library answered {len(answered)} reads; the journal holds "
                  f"{len(saved)} saved responses\n")

            print("=== 3. Resume ===")
            before = len(answered)
            result = swarm.resume(run_id)
            print(f"  the library answered {len(answered) - before} more read "
                  f"({', '.join(answered[before:])}); the saved read of {ids[1]} was replayed")
            print(f"  each source was read once: {sorted(answered) == sorted(ids)}\n")

            print("=== 4. Accepted artifact ===")
            artifact = current_artifact(store, run_id)
            print(f"  path:        {artifact.publication.path}")
            print(f"  sha256:      {artifact.publication.sha256}")
            print(f"  decision:    {result.acceptance.decision.value} "
                  f"({'; '.join(result.acceptance.reasons)})")
            for check in result.acceptance.validators:
                print(f"  validator:   {check.validator.name} {check.validator.version}: "
                      f"{check.outcome.value}")
            print(f"  eligible:    {artifact.eligible}\n")

            print("=== 5. Evidence ===")
            evidence = load_source_evidence(store, run_id)
            for source in evidence["sources"]:
                print(f"  source {source['source_id']} version {source['source_version']}: "
                      f"sha256 {source['sha256'][:12]}, available {source['available']}")
            for claim in evidence["claims"]:
                print(f"  claim {claim['claim_id']}: {len(claim['supporting'])} supporting, "
                      f"{len(claim['contradicting'])} contradicting, "
                      f"{len(claim['unresolved'])} open question(s)")
            assert result.accepted and artifact.eligible

        print("\n=== 6. smythe runs --journal runs.db inspect RUN_ID ===")
        code = smythe_cli(["runs", "--journal", str(journal), "inspect", run_id])
        print(f"(exit code {code})")
        assert code == 0


if __name__ == "__main__":
    main()
