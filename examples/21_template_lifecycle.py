"""A template's lifecycle: recorded, evaluated, rejected, promoted by mistake and rolled back.

    python examples/21_template_lifecycle.py

Three accepted briefing runs share one generated plan: a step per section,
each skipped when it fails. Planner history records the runs, and
candidate_from_graph() distills the plan into a template record that names
them. validate_template_record() checks the record as an untrusted
candidate, and a TemplateCatalog stores it as the predecessor.

A second record, written as a candidate, retries every step once and leaves
out the Timeline step, which no briefing needs. Validation accepts it: it
grants nothing and keeps every gate, check and prerequisite its parent
declares. A record whose first step grants itself a tool is refused, and the
catalog is unchanged.

An Autotune campaign compares the candidate with the predecessor and with a
single worker that writes the whole brief, a reference arm under
must_not_be_worse. The evaluator below is inline and deterministic: one
trial runs every task of its split's pack once, and a strict acceptance
policy per task family judges each brief. Development and confirmation hold
briefings, some with a step whose first answer fails; the holdout holds
incident reviews, which need a Timeline. The candidate is better in
development and confirmation and worse on the holdout, so it is rejected,
and the policy registry refuses to approve that decision.

A second evaluator is fooled: its holdout holds briefings only. There the
candidate is promoted, a reviewer approves the decision, and it is activated
for new runs. The next incident review built from it is rejected for its
missing Timeline. A rollback restores the predecessor and writes only the
registry; an architect built before it keeps the version it was built with,
and a new run resolves the predecessor and is accepted.

Everything runs offline and deterministically, with no keys and no cost. The
fixtures show the mechanics only; no number here measures a template.

Docs: docs/learning.md, docs/optimize.md#policy-registry
"""

import asyncio
import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from smythe import (
    AcceptancePolicy, CompletionResult, ConstrainedArchitect, OfflineProvider, PlannerMemory,
    SimpleArchitect, Swarm, Task, TemplateCatalog, TemplateRecord, TemplateValidationError,
    TextValidator, candidate_from_graph, repeated_structures, template_target,
    validate_template_record,
)
from smythe.optimize import (
    Candidate, EvaluatorManifest, ExperimentContract, ExperimentLedger, MetricObjective,
    MutableFieldRule, OptimizationRunner, PolicyRegistry, RegistryError, TrialContext,
    TrialOutcome, sha256_prefixed,
)
from smythe.prompts import PLANNING_SYSTEM_PROMPT

NAME = "brief"
TARGET = template_target(NAME)  # "planner.template.brief"
SINGLE_WORKER = "single_worker"
STEPS = ("Findings", "Risks", "Timeline", "Sources")
# The sections each task family's deliverable needs, in order.
SECTIONS = {"briefing": ("Findings", "Risks", "Sources"),
            "incident": ("Findings", "Timeline", "Sources")}
GOALS = {"briefing": "Brief the operations team on {}.",
         "incident": "Review the incident in which {}."}
# The plan a model would generate: one step per section, skipped when it fails.
PLAN = {"topology": ["fork_join"], "nodes": [
    {"id": step.lower(), "label": f"Write the {step} section", "failure_policy": "skip"}
    for step in STEPS]}
# Each family's strict acceptance policy: the brief must hold its sections, in order.
POLICIES = {family: AcceptancePolicy(mode="strict", final_validators=[
    TextValidator({"sections": list(sections)})]) for family, sections in SECTIONS.items()}
_STEP = re.compile(r"Write the (\w+) section")
_LISTED = re.compile(r"Sections: ([^.]+)\.")


@dataclass(frozen=True)
class FixtureTask:
    """One fixture task. The first answer to each step named in ``flaky`` fails."""

    family: str
    subject: str
    flaky: tuple[str, ...] = ()

    @property
    def goal(self) -> str:
        sections = ", ".join(SECTIONS[self.family])
        return f"{GOALS[self.family].format(self.subject)} Sections: {sections}."


SOURCE_TASKS = (FixtureTask("briefing", "the move to a new storage vendor"),
                FixtureTask("briefing", "the quarterly access audit"),
                FixtureTask("briefing", "the new on-call rotation"))
PACKS = {
    "development": (FixtureTask("briefing", "last week's payment outage", ("Risks",)),
                    FixtureTask("briefing", "the rise in checkout latency"),
                    FixtureTask("briefing", "the holiday capacity plan", ("Sources",))),
    "confirmation": (FixtureTask("briefing", "the backup restore drill", ("Findings",)),
                     FixtureTask("briefing", "the queue migration"),
                     FixtureTask("briefing", "the new alerting thresholds", ("Risks",))),
    # Held out: a family that development and confirmation do not contain.
    "holdout": (FixtureTask("incident", "a DNS change took the status page offline"),
                FixtureTask("incident", "a stuck queue delayed order emails"),
                FixtureTask("incident", "an expired certificate broke the partner API")),
}
# The fooled evaluator holds out more briefings, so nothing it runs needs a Timeline.
FOOLED_PACKS = {**PACKS, "holdout": (
    FixtureTask("briefing", "the database upgrade window", ("Findings",)),
    FixtureTask("briefing", "the vendor contract renewal"),
    FixtureTask("briefing", "the office network refresh", ("Sources",)))}


class FixtureWriter(OfflineProvider):
    """The fixture model: it writes the section a step names, or every section a task lists.

    A planning call gets PLAN. The first answer to a step named in *flaky*
    fails, as a dropped connection would; asking again succeeds.
    """

    def __init__(self, flaky: tuple[str, ...] = ()) -> None:
        super().__init__(plan=PLAN)
        self.flaky = set(flaky)
        self.step_calls = 0

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if system == PLANNING_SYSTEM_PROMPT:
            return await super().complete(system, prompt, model)
        self.step_calls += 1
        lines = prompt.splitlines()
        step = next((line.removeprefix("Your step: ") for line in lines
                     if line.startswith("Your step: ")), lines[0])
        named = _STEP.match(step)
        if named and named.group(1) in self.flaky:
            self.flaky.discard(named.group(1))
            raise ConnectionError(f"the connection dropped before the {named.group(1)} answer")
        sections = ([named.group(1)] if named
                    else [item.strip() for item in _LISTED.search(prompt).group(1).split(",")])
        text = "\n\n".join(f"## {section}\n\nThe {section.lower()} of this brief."
                           for section in sections)
        return CompletionResult(text, prompt_tokens=40, completion_tokens=60)


@dataclass(frozen=True)
class Outcome:
    """One run: its acceptance decision and reasons, its step calls and the versions it used."""

    decision: str
    reasons: tuple[str, ...]
    step_calls: int
    versions: tuple[str, ...]


def run_task(task: FixtureTask, architect) -> Outcome:
    """Run *task* once on *architect*, judged by its family's strict policy."""
    writer = FixtureWriter(task.flaky)
    swarm = Swarm(model="offline", provider=writer, architect=architect,
                  acceptance=POLICIES[task.family], artifact_dir=None)
    try:
        result = swarm.execute(Task(task.goal))
    except Exception as error:  # A run that raised was never decided.
        return Outcome("error", (type(error).__name__,), writer.step_calls, ())
    used = tuple(item["version"] for item in (result.graph.planning or {}).get("templates", []))
    return Outcome(result.acceptance.decision.value, result.acceptance.reasons,
                   writer.step_calls, used)


class PackEvaluator:
    """The evaluator: one trial runs every task of its split's pack once on one arm.

    A policy names a template version the catalog holds, whose record builds
    each graph without a planning call, or the single worker. A trial reports
    the briefs accepted, the primary metric, and the step calls made, and its
    gate requires a decision for every task. It is deterministic, so every
    trial of one arm on one split reports the same numbers.
    """

    def __init__(self, catalog: TemplateCatalog, packs: dict) -> None:
        self.catalog = catalog
        self.packs = packs

    def manifest(self) -> EvaluatorManifest:
        """What defines this evaluation: this file, each split's tasks and the policies."""
        return EvaluatorManifest(
            code_digest="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            split_manifests={split: sha256_prefixed([[task.family, task.subject, list(task.flaky)]
                                                     for task in tasks])
                             for split, tasks in self.packs.items()},
            validators=tuple(identity for policy in POLICIES.values()
                             for identity in policy.validator_identities))

    def run_pass(self, split: str, value: str) -> tuple[dict, dict]:
        accepted = step_calls = decided = 0
        for task in self.packs[split]:
            architect = (SimpleArchitect() if value == SINGLE_WORKER
                         else self.catalog.get(value).as_architect())
            outcome = run_task(task, architect)
            accepted += outcome.decision == "accepted"
            decided += outcome.decision != "error"
            step_calls += outcome.step_calls
        return ({"accepted": accepted, "step_calls": step_calls},
                {"every_task_decided": decided == len(self.packs[split])})

    async def __call__(self, context: TrialContext) -> TrialOutcome:
        metrics, gates = await asyncio.to_thread(
            self.run_pass, context.split, context.candidate.policy[TARGET])
        return TrialOutcome(metrics=metrics, gates=gates, actual_cost_microusd=0)


def campaign(ledger_path: Path, evaluator: PackEvaluator, predecessor: str, candidate: str):
    """Compare *candidate* with *predecessor*, the incumbent, and with the single worker."""
    contract = ExperimentContract(
        name="brief-template",
        objectives=(MetricObjective("accepted", "maximize", primary=True),
                    MetricObjective("step_calls", "minimize")),
        mutable_fields=(TARGET,),
        mutable_field_rules={TARGET: MutableFieldRule(
            "string", choices=(predecessor, candidate, SINGLE_WORKER))},
        required_gates=("every_task_decided",),
        development_repetitions=1, confirmation_repetitions=3, holdout_repetitions=3,
        max_candidates=3, max_parallel_candidates=1, max_trials=20, max_wall_seconds=600,
        max_budget_microusd=0, per_trial_reservation_microusd=0,
        confidence=0.95, min_improvement=0.5, base_seed=21,
    )
    incumbent = Candidate(contract, {TARGET: predecessor}, "The predecessor stays in effect.")
    challenger = Candidate(contract, {TARGET: candidate},
                           "Retries recover failed answers, and briefings need no Timeline.",
                           parent=incumbent)
    reference = Candidate(contract, {TARGET: SINGLE_WORKER}, "One worker writes the whole brief.")
    with ExperimentLedger(ledger_path, durability="normal") as ledger:
        runner = OptimizationRunner(contract, ledger, evaluator,
                                    evaluator_hash=evaluator.manifest().evaluator_hash)
        return asyncio.run(runner.run(incumbent, (challenger,), references=(reference,),
                                      reference_rule="must_not_be_worse"))


# What a verdict against the single-worker reference says, from the candidate's side.
AGAINST_ONE_WORKER = {"improved": "the candidate does better than one worker",
                      "inconclusive": "no difference from one worker is shown",
                      "worse": "one worker does better, which must_not_be_worse refuses"}


def show_campaign(result) -> None:
    (development,) = result.development_scores
    means = (development["metric_means"]["accepted"],
             development["incumbent_metric_means"]["accepted"])
    print(f"  development   accepted {means[0]:g} vs {means[1]:g}")
    for stage, assessment in (("confirmation", result.confirmation_assessment),
                              ("holdout", result.holdout_assessment)):
        primary = assessment.primary
        low, high = primary.confidence_interval
        print(f"  {stage:<13} accepted {primary.candidate_mean:g} vs {primary.baseline_mean:g}, "
              f"interval [{low:g}, {high:g}] over {primary.sample_count} pairs: "
              f"{assessment.verdict}")
    (single,) = result.references
    print(f"  single worker {single['verdict']}: {AGAINST_ONE_WORKER[single['verdict']]}")
    print(f"  decision      promoted {result.promoted}, verdict {result.verdict}")


def files(root: Path) -> dict[str, str]:
    """Every file under *root* with the SHA-256 of its bytes."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def new_architect(catalog: TemplateCatalog, registry: PolicyRegistry) -> ConstrainedArchitect:
    """An architect for new runs, offering the version in effect; a scripted model selects it."""
    return ConstrainedArchitect(OfflineProvider(responses=[f'[{{"template": "{NAME}"}}]']),
                                catalog.active_templates(registry, [NAME]), model="offline")


def lifecycle(work: Path) -> None:
    catalog = TemplateCatalog(work / "templates")
    ledger_path, registry_path = work / "optimize.sqlite3", work / "registry.sqlite3"

    print("=== 1. Distill, record, validate and catalog ===")
    memory = PlannerMemory(work / "history.jsonl")
    for task in SOURCE_TASKS:
        swarm = Swarm(model="offline", provider=FixtureWriter(), memory=memory,
                      acceptance=POLICIES["briefing"], artifact_dir=None)
        result = swarm.execute(Task(task.goal))
        print(f"  run {result.execution_id[:8]}: {result.acceptance.decision.value}")
    ((structure, runs),) = repeated_structures(memory, min_accepted=3).items()
    record = candidate_from_graph(result.graph, rows=memory.iter_outcomes(), name=NAME,
                                  min_accepted=3, description="A brief written a section a step")
    predecessor = catalog.put(validate_template_record(record))
    print(f"  structure {structure[:12]}... repeated by {len(runs)} accepted runs")
    print(f"  predecessor {predecessor[:12]}...: steps "
          f"{[node['id'] for node in record.blueprint]}, {len(record.source_runs)} source runs")

    # A candidate written from the predecessor; its runs are not the candidate's evidence.
    data = {key: value for key, value in record.to_dict().items()
            if key not in ("source_runs", "review")}
    data.update(description="Retries every step once and leaves out the Timeline step",
                parent_version=predecessor,
                blueprint=[dict(node, failure_policy="retry", max_retries=1)
                           for node in data["blueprint"] if node["id"] != "timeline"])
    forged = {**data, "blueprint": [dict(data["blueprint"][0], tools=["shell"]),
                                    *data["blueprint"][1:]]}
    before = files(catalog.root)
    try:
        catalog.put(forged)
    except TemplateValidationError as error:
        assert files(catalog.root) == before, "a refused candidate changed the catalog"
        print(f"  refused a candidate at node {error.node!r}, field {error.field!r}; "
              f"catalog unchanged: {files(catalog.root) == before}")
    else:
        raise SystemExit("the catalog stored a candidate that grants itself a tool")
    candidate = catalog.put(TemplateRecord.from_dict(data))  # Validated against its parent.
    print(f"  candidate {candidate[:12]}...: steps {[node['id'] for node in data['blueprint']]}, "
          f"each retried once; validated against its parent")

    print("\n=== 2. A campaign against the predecessor and a single worker ===")
    honest = campaign(ledger_path, PackEvaluator(catalog, PACKS), predecessor, candidate)
    show_campaign(honest)
    assert not honest.promoted, "the honest campaign promoted the harmful candidate"
    print("  Every pair differs by the same amount, so each interval has zero width:")
    print("  that reflects a deterministic evaluator, not measured precision.")

    print("\n=== 3. The harmful candidate is rejected ===")
    with PolicyRegistry(registry_path) as registry:
        try:
            registry.record_review(ledger_path, honest.campaign_id, reviewer="brief-owner",
                                   verdict="approve")
        except RegistryError as error:
            print(f"  approval refused: {str(error).rsplit('; ', 1)[-1]}")
        else:
            raise SystemExit("a review approved a candidate its campaign did not promote")
        review = registry.record_review(ledger_path, honest.campaign_id, reviewer="brief-owner",
                                        verdict="reject", note="incident reviews lose a section")
        print(f"  review recorded: {review.verdict}, by {review.reviewer} (declared)")
        print(f"  in effect for {TARGET}: {registry.resolve(TARGET) or 'nothing'}")

    print("\n=== 4. A fooled evaluator promotes it; it is reviewed and activated ===")
    fooled = campaign(ledger_path, PackEvaluator(catalog, FOOLED_PACKS), predecessor, candidate)
    show_campaign(fooled)
    assert fooled.promoted, "the fooled campaign did not promote the candidate"
    with PolicyRegistry(registry_path) as registry:
        approval = registry.record_review(ledger_path, fooled.campaign_id, reviewer="brief-owner",
                                          verdict="approve", note="better on every split")
        registry.activate(TARGET, approval.review_id, expected_seq=0, actor="release-operator",
                          reason="approved for new briefs")
        earlier = new_architect(catalog, registry)
        outcome = run_task(PACKS["holdout"][0], earlier)
        print(f"  in effect: {registry.resolve(TARGET).policy[TARGET][:12]}..., the candidate")
        print(f"  a new incident review built from {outcome.versions[0][:12]}...: "
              f"{outcome.decision}")
        print(f"    {outcome.reasons[0]}")
        assert outcome.versions[0] == candidate and outcome.decision == "rejected"

        print("\n=== 5. Roll back ===")
        ledger_before, catalog_before = files(work)["optimize.sqlite3"], files(catalog.root)
        registry.rollback(TARGET, to_seq=1, expected_seq=2, actor="release-operator",
                          reason="incident reviews lose their Timeline")
        unchanged = (files(work)["optimize.sqlite3"] == ledger_before
                     and files(catalog.root) == catalog_before)
        print(f"  the ledger and the catalog are unchanged: {unchanged}")
        assert unchanged, "the rollback changed the ledger or the catalog"
        for event in registry.history(TARGET):
            print(f"  seq {event.seq} {event.action:<9} {event.policy[TARGET][:12]}...")
        kept = run_task(PACKS["holdout"][0], earlier).versions[0]
        print(f"  the architect built before the rollback still builds {kept[:12]}...")
        assert kept == candidate

        print("\n=== 6. A new run resolves the predecessor ===")
        outcome = run_task(PACKS["holdout"][0], new_architect(catalog, registry))
        print(f"  a new incident review built from {outcome.versions[0][:12]}...: "
              f"{outcome.decision}")
        assert outcome.versions[0] == predecessor and outcome.decision == "accepted"
    with ExperimentLedger(ledger_path, read_only=True) as ledger:
        for result in (honest, fooled):
            (saved,) = ledger.list_decisions(result.campaign_id)
            print(f"  {saved['decision_id'][:24]}... still reads: promoted {saved['promoted']}")


def main() -> None:
    # Each step failure the fixtures plant would log a warning; keep the output to the story.
    logging.getLogger("smythe.tracer").setLevel(logging.ERROR)
    print("Running offline with a fixture model (no keys, no cost).\n")
    with tempfile.TemporaryDirectory(prefix="smythe-template-lifecycle-",
                                     ignore_cleanup_errors=True) as work:
        lifecycle(Path(work))


if __name__ == "__main__":
    main()
