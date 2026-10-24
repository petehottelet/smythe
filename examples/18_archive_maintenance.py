"""Maintain an archive of accepted briefs: repair what changed, then compare with a full rerun.

    python examples/18_archive_maintenance.py

Four durable research runs each answer one question from the bundled sources
(examples/source_to_artifact/sources/), read by journaled read steps from a
library on the loopback address (examples/_source_library.py). Two sources
are shared: two runs read renewals-2026 and two read support-2026. Each run
extracts a finding from each read with a model step, publishes a brief
written by local code, accepts it under a strict policy and records its
artifact lineage.

Then three things change, as they do in a maintained archive:

- renewals-2026 is revised, and the library serves its new version;
- one published brief is damaged on disk;
- one final validator's version is bumped.

A repair plan says, before anything runs, what each brief calls for. The two
runs that read renewals-2026 regenerate only that read, its extraction and
the brief, and keep their other results. The damaged brief is rendered again
from its saved synthesis, and the run that nothing else touched is decided
again under the new validator. Applying the plan does that under an explicit
allowance. The kept results and the changed conclusions are then inspected
separately. A full recomputation of the four questions into a second journal
publishes the same bytes with the same decisions, and a table compares the
work each took.

Offline runs cost nothing, so the comparison counts work, not money. In a
live run, model steps would extract the findings; here OfflineProvider
answers them, and a deterministic local synthesizer writes each brief.

Docs: docs/lineage.md
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import NamedTuple

from _source_library import SourceLibrary, read_versions

from smythe import (
    AcceptancePolicy, LineagePolicy, LocalOnly, MarkdownRenderer, OfflineProvider, RepairPlan,
    SimpleArchitect, SQLiteWorkflowStore, Swarm, Task, TextValidator, ValidationReport,
    ValidatorIdentity,
)
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.lineage import load_lineage
from smythe.validation import hard_finding
from smythe.workflow import current_decision
from smythe.workflow_artifact import current_artifact

# Each bundled source's versions, oldest first.
VERSIONS = {
    "survey-2019": ["2019-spring"],
    "renewals-2026": ["2026-q2", "2026-q3"],
    "support-2026": ["2026-09"],
    "pricing-2026": ["2026-07"],
    "staffing-2026": ["2026-08"],
    "trials-2026": ["2026-06"],
    "checkout-2026": ["2026-06"],
}
REVISED = "renewals-2026"
SCOPE = "research-desk"
RUN_BUDGET = 0.10
REASON = "renewals-2026 was revised, a brief was damaged and brief-quotes is now version 2"


class Item(NamedTuple):
    """One question in the archive and the sources its run reads."""

    key: str
    question: str
    sources: tuple[str, ...]


ITEMS = (
    Item("billing-default", "Should new subscriptions default to monthly or annual billing?",
         ("survey-2019", "renewals-2026", "support-2026")),
    Item("renewal-forecast", "How should the 2027 renewal forecast weight monthly plans?",
         ("renewals-2026", "pricing-2026")),
    Item("support-staffing", "Does the billing support queue need another specialist?",
         ("support-2026", "staffing-2026")),
    Item("trial-length", "Should the free trial stay at 14 days?",
         ("trials-2026", "checkout-2026")),
)
DAMAGED = "support-staffing"

# The analyst's reading: the words that mark the sentence of each source a
# brief's conclusion quotes.
FINDINGS = {
    "survey-2019": "concluded that",
    "renewals-2026": "chose monthly billing",
    "support-2026": "asked to switch plans",
    "pricing-2026": "The annual plan costs",
    "staffing-2026": "waited",
    "trials-2026": "converted",
    "checkout-2026": "Abandoned checkouts",
}

# Units of work, as the journals record them.
UNITS = ("reads", "model steps", "local steps", "publications", "acceptance evaluations",
         "new runs")


# ---------------------------------------------------------------------------
# Local components: the brief and its final validator
# ---------------------------------------------------------------------------


def finding(source_id: str, text: str) -> str:
    """The sentence of *text* that the conclusion quotes, or an empty string."""
    return next((line.strip() for line in text.splitlines() if FINDINGS[source_id] in line), "")


class Brief:
    """Writes a brief from what a run read and extracted: deterministic local code."""

    def synthesize(self, graph: ExecutionGraph) -> str:
        read = {node.metadata["read_step"]["input"]["source_id"]: node.result
                for node in graph.nodes if "read_step" in node.metadata}
        notes = [node.result for node in graph.nodes if "read_step" not in node.metadata]
        return "\n\n".join([
            "## Sources", "\n".join(
                f"- {source_id}: sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}"
                for source_id, text in read.items()),
            "## Notes", "\n".join(f"- {note}" for note in notes),
            "## Conclusion", "\n".join(
                f'- {source_id}: "{finding(source_id, text)}"' for source_id, text in read.items()),
        ])


def section(text: str, heading: str) -> list[str]:
    """The list lines of a brief's section, without their markers."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line == f"## {heading}"
        elif inside and line.startswith("- "):
            lines.append(line[2:])
    return lines


class QuoteCheck:
    """A final validator: the conclusion quotes each source the brief lists.

    Version 2 also refuses a quotation that is empty.
    """

    def __init__(self, version: str = "1") -> None:
        self.version = version
        self.identity = ValidatorIdentity.for_config("brief-quotes", version,
                                                     {"section": "Conclusion"})

    def validate(self, subject) -> ValidationReport:
        text = subject.text()
        quoted = {line.split(":", 1)[0]: line.split(":", 1)[-1].strip()
                  for line in section(text, "Conclusion")}
        findings = [hard_finding("source_not_quoted", f"{source_id} is listed but not quoted")
                    for source_id in (line.split(":", 1)[0] for line in section(text, "Sources"))
                    if source_id not in quoted]
        if self.version != "1":
            findings += [hard_finding("empty_quotation", f"the quotation of {source_id} is empty")
                         for source_id, quotation in quoted.items() if quotation == '""']
        return ValidationReport("output", tuple(findings), validator=self.identity)


def policy(validator: str = "1") -> AcceptancePolicy:
    """The strict policy, with brief-quotes at version *validator*."""
    return AcceptancePolicy(mode="strict", final_validators=[
        TextValidator({"sections": ["Sources", "Notes", "Conclusion"]}), QuoteCheck(validator)])


# ---------------------------------------------------------------------------
# The research runs
# ---------------------------------------------------------------------------


def research_swarm(store: SQLiteWorkflowStore, library: SourceLibrary, artifact_dir: Path, *,
                   validator: str = "1", reuse: bool = True) -> Swarm:
    """A Swarm whose durable runs record lineage in the archive's access scope.

    ``context_sources=[]`` assembles each model step's context from the
    results it depends on and no other text, so its lineage names exactly
    the read it extracts from.
    """
    return Swarm(
        model="offline", provider=OfflineProvider(), architect=SimpleArchitect(), run_store=store,
        max_budget_usd=RUN_BUDGET, max_concurrency=1, read_tools=[library.profile()],
        context_sources=[],
        synthesizer=LocalOnly(Brief, "archive-brief", "1", role="synthesizer"),
        renderer=MarkdownRenderer(), artifact_dir=artifact_dir, acceptance=policy(validator),
        lineage=LineagePolicy(access_scope=SCOPE, reuse=reuse),
    )


def item_graph(item: Item, library: SourceLibrary) -> ExecutionGraph:
    """A read step for each of the item's sources, and an extraction from each read."""
    reads, nodes = library.profile(), []
    for source_id in item.sources:
        nodes.append(reads.step(f"read-{source_id}", {"source_id": source_id},
                                label=f"Read {source_id}"))
        nodes.append(Node(id=f"extract-{source_id}", label=f"Extract the {source_id} finding",
                          depends_on=[f"read-{source_id}"]))
    graph = ExecutionGraph([Topology.FORK_JOIN], nodes)
    graph.task = Task(item.question)
    return graph


def run_items(swarm: Swarm, library: SourceLibrary) -> dict[str, str]:
    """Run each item once, in order; return each item's run id by its key."""
    runs = {}
    for item in ITEMS:
        result = swarm.execute(item_graph(item, library))
        assert result.accepted, result.acceptance.reasons
        runs[item.key] = result.execution_id
    return runs


def damage(path: Path) -> None:
    """Flip one bit in the middle of a published file, as a faulty disk or editor might."""
    data = path.read_bytes()
    middle = len(data) // 2
    path.write_bytes(data[:middle] + bytes([data[middle] ^ 0x20]) + data[middle + 1:])


def journal_work(store: SQLiteWorkflowStore) -> dict[str, int]:
    """The work a journal records, summed over its runs.

    Reads and model steps are the run's journaled calls, local steps the
    syntheses local code wrote, and publications and acceptance evaluations
    the documents published and decided.
    """
    totals = dict.fromkeys(UNITS, 0)
    for run in store.list_runs():
        run_id = run["run_id"]
        phases = [call["key"]["phase"] for call in store.inspect_run(run_id)["calls"]]
        counts = {"reads": phases.count("tool"), "model steps": len(phases) - phases.count("tool"),
                  "local steps": len(store.list_operations(run_id, "synthesis")),
                  "publications": len(store.list_operations(run_id, "publication")),
                  "acceptance evaluations": len(store.list_operations(run_id, "acceptance")),
                  "new runs": 1}
        for unit in UNITS:
            totals[unit] += counts[unit]
    return totals


def delivered(store: SQLiteWorkflowStore, run_id: str) -> tuple[str, dict]:
    """The SHA-256 of a run's current published document, and its current decision."""
    artifact = current_artifact(store, run_id)
    assert artifact is not None and artifact.eligible
    return artifact.publication.sha256, current_decision(store, run_id)["record"].to_dict()


# ---------------------------------------------------------------------------
# What the example prints
# ---------------------------------------------------------------------------


def count(number: int, unit: str) -> str:
    return f"{number} {unit if number == 1 else unit + 's'}"


def show_plan(plan: RepairPlan, keys: dict[str, str]) -> None:
    for item in plan.to_dict()["items"]:
        print(f"  {keys[item['run_id']]:<17} {item['action']:<11} {item['status']:<8} "
              f"{', '.join(item['reasons'])}")
        work = item["work"]
        if item["action"] == "regenerate":
            for action, verb in (("reuse", "keeps"), ("regenerate", "runs ")):
                print(f"      {verb} " + ", ".join(node["id"] for node in item["nodes"]
                                                  if node["action"] == action))
        else:
            print("      no node runs")
        print("      predicted: " + ", ".join(count(work[key], unit) for key, unit in (
            ("new_runs", "new run"), ("read_calls", "read"), ("model_steps", "model step"),
            ("publications", "publication"),
            ("acceptance_evaluations", "acceptance evaluation"))))


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding and stream.encoding.lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8")
    print("Running offline: bundled sources, a loopback source library, OfflineProvider, no keys.\n")
    with SourceLibrary(read_versions(VERSIONS)) as library, \
            TemporaryDirectory(prefix="smythe-") as work:
        work = Path(work)
        with SQLiteWorkflowStore(work / "archive.db") as store:
            print("=== 1. The archive: four accepted briefs ===")
            runs = run_items(research_swarm(store, library, work / "artifacts"), library)
            keys = {run_id: key for key, run_id in runs.items()}
            for item in ITEMS:
                read = ", ".join(f"{source_id} {library.current(source_id).version}"
                                 for source_id in item.sources)
                sha256 = current_artifact(store, runs[item.key]).publication.sha256
                print(f"  {item.key:<17} run {runs[item.key][:12]}  brief {sha256[:12]}  "
                      f"read {read}")
            print(f"  the library answered {len(library.reads)} reads\n")

            print("=== 2. What changed ===")
            earlier = library.current(REVISED)
            revised = library.release(REVISED)
            print(f"  {REVISED} was revised: version {earlier.version} -> {revised.version}")
            damaged = current_artifact(store, runs[DAMAGED]).publication
            damage(damaged.path)
            print(f"  {DAMAGED}'s published {damaged.name} lost a bit on disk")
            print("  brief-quotes, a final validator, is now version 2\n")
            maintained = research_swarm(store, library, work / "artifacts", validator="2")

            print("=== 3. The repair plan, before anything runs ===")
            plan = maintained.plan_repair(sources=library)
            show_plan(plan, keys)
            new_runs = sum(item["work"]["new_runs"] for item in plan.to_dict()["items"])
            print(f"  plan sha256 {plan.plan_sha256[:12]}; nothing was read or called\n")

            print("=== 4. Apply it ===")
            allowance = new_runs * RUN_BUDGET
            print(f"  allowance: {allowance:.2f} USD, for {count(new_runs, 'new run')} of up to "
                  f"{RUN_BUDGET:.2f} USD each")
            before, answered = journal_work(store), len(library.reads)
            result = maintained.repair(plan, plan_sha256=plan.plan_sha256,
                                       max_total_budget_usd=allowance, reason=REASON,
                                       sources=library)
            after = journal_work(store)
            selective = {unit: after[unit] - before[unit] for unit in UNITS}
            heads = dict(runs)
            for entry in result.to_dict()["items"]:
                key, work_done = keys[entry["run_id"]], entry["work"]
                successor = entry.get("successor_run_id")
                if successor is not None:
                    heads[key] = successor
                where = f"successor run {successor[:12]}" if successor else "in place"
                print(f"  {key:<17} {entry['action']:<11} {entry['outcome']} ({where}), "
                      f"decision {entry['decision']}")
                print(f"      {count(work_done['read_calls'], 'read')}, "
                      f"{count(work_done['model_calls'], 'model step')}, "
                      f"{count(work_done['reused_nodes'], 'kept result')}, "
                      f"{count(work_done['publications'], 'publication')}, "
                      f"{count(work_done['acceptance_evaluations'], 'acceptance evaluation')}")
            assert len(library.reads) - answered == selective["reads"]
            print()

            print("=== 5. Kept results and changed conclusions, inspected separately ===")
            for item in ITEMS:
                if heads[item.key] == runs[item.key]:
                    continue
                lineage = load_lineage(store, heads[item.key]).to_dict()
                earlier_nodes = {node["id"]: node
                                 for node in load_lineage(store, runs[item.key]).nodes}
                kept = [node for node in lineage["nodes"] if "reused_from" in node]
                print(f"  {item.key}: run {heads[item.key][:12]} keeps {len(kept)} results "
                      f"of run {lineage['predecessor']['run_id'][:12]}")
                for node in kept:
                    same = node["result_sha256"] == earlier_nodes[node["id"]]["result_sha256"]
                    print(f"      {node['id']:<23} result {node['result_sha256'][:12]}, "
                          f"{'the same as' if same else 'NOT'} its predecessor's")
                    assert same
            old, new = finding(REVISED, earlier.text), finding(REVISED, revised.text)
            print(f"  Conclusions that rest on {REVISED}, which said \"{old}\":")
            for item in ITEMS:
                if REVISED in item.sources:
                    text = current_artifact(store, heads[item.key]).publication.path.read_text(
                        encoding="utf-8")
                    quoted = next(line for line in section(text, "Conclusion")
                                  if line.startswith(REVISED))
                    print(f"      {item.key:<17} {quoted}")
                    assert new in text and old not in text
            repaired = current_artifact(store, runs[DAMAGED]).publication
            print(f"  {DAMAGED}: published again as {repaired.name}, the same bytes as before: "
                  f"{repaired.sha256 == damaged.sha256}")
            untouched = next(item.key for item in ITEMS
                             if item.key != DAMAGED and REVISED not in item.sources)
            versions = [f"{check.validator.name} {check.validator.version}" for check in
                        current_decision(store, runs[untouched])["record"].validators]
            print(f"  {untouched}: decided again under {', '.join(versions)}; no node ran\n")
            selected = {item.key: delivered(store, heads[item.key]) for item in ITEMS}

        print("=== 6. Full recomputation into a second journal ===")
        with SQLiteWorkflowStore(work / "full.db") as full_store:
            full_runs = run_items(research_swarm(full_store, library, work / "full-artifacts",
                                                 validator="2"), library)
            full = journal_work(full_store)
            recomputed = {item.key: delivered(full_store, full_runs[item.key]) for item in ITEMS}
        print(f"  {len(full_runs)} new runs computed every node again\n")

        print("=== 7. Selective repair against full recomputation ===")
        print(f"  {'work':<24}{'selective':>10}{'full':>8}")
        for unit in UNITS:
            print(f"  {unit:<24}{selective[unit]:>10}{full[unit]:>8}")
        print(f"\n  {'item':<19}{'same bytes':<12}{'same decision':<15}brief sha256")
        for item in ITEMS:
            (sha256, decision), (full_sha256, full_decision) = selected[item.key], recomputed[
                item.key]
            print(f"  {item.key:<19}{'yes' if sha256 == full_sha256 else 'NO':<12}"
                  f"{'yes' if decision == full_decision else 'NO':<15}{sha256[:12]}")
        assert selected == recomputed
        assert all(selective[unit] <= full[unit] for unit in UNITS)
        assert sum(selective.values()) < sum(full.values())


if __name__ == "__main__":
    main()
