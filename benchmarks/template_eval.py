"""Offline template evaluation: a template version against its predecessor and one worker.

    python benchmarks/template_eval.py
    python benchmarks/template_eval.py --seed 3 --out PATH

A template version is evaluated as an Autotune policy for its policy registry
target, ``planner.template.report``. The policy maps the target to a version
held in a template catalog, or to ``single_agent``, and each arm comes from
that value alone:

    <version>      a ConstrainedArchitect over the catalog's record of that
                   version, whose selection an OfflineProvider scripts; a task
                   in a family the record does not apply to runs on a
                   SimpleArchitect instead
    single_agent   a SimpleArchitect: one worker receives the whole task

The predecessor version is the incumbent and the candidate version the
challenger. ``single_agent`` is a sealed reference arm under the
``must_not_be_worse`` rule, so all three run on the same confirmation and
holdout seeds, and a candidate that beats its predecessor but is worse than
one worker is not promoted.

One trial is one pass over a split's tasks. Task packs are generated in
Python and pinned by the SHA-256 of their manifests: development and
confirmation packs hold briefing tasks, and the holdout pack holds incident
reviews, a task family neither earlier split contains. The holdout pack is
built, and checked against its pin, only when a holdout trial runs. Every
pack also holds a code-change task, a family the template must not apply
to. Each family has one strict acceptance policy, a TextValidator that
requires the family's sections, and every arm is judged by it. Planted
faults make a section step's first answer fail, as a transient failure of
one step in a decomposed plan; a single worker's whole-task answer is not
such a step.

Metrics per trial are the pass's accepted outcomes (primary), attempted
tasks, attributable cost in micro-USD and planned nodes. Required gates:
every outcome was decided by its family's policy, every call's usage is
known, no task of an excluded family ran a template, and at least one
outcome was accepted. The evaluator manifest binds this file's digest, each
split's pinned pack digest with the acceptance policies, the validator
identities and the fixture price. Cost per accepted outcome is reported as
a ratio of sums, and the campaign's own spend apart from it.

The record runs four fixture candidates against one predecessor: an
improvement, a harmful candidate that drops a section only incident reviews
require, a partial improvement that loses to the single worker, and one
that applies to the excluded family. Fixture runs establish harness
mechanics, not template quality: every step answers with fixture content
and declared usage, the thresholds are placeholders, the decisions activate
nothing, and the record is labeled offline-mechanics, claimable false,
api_calls 0. It changes no default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.artifact_records import scrub_record  # noqa: E402
from benchmarks.astra_campaign._json import digest  # noqa: E402
from benchmarks.offline_arms import (  # noqa: E402
    FIXTURE_MODEL, FixtureFault, declared_tokens, execute_attempt, offline_record, policy_record,
    policy_unchanged,
)
from benchmarks.provider_usage import BLENDED_USD_PER_TOKEN  # noqa: E402

from smythe import AcceptancePolicy, Swarm, Task, TextValidator  # noqa: E402
from smythe._money import CEILING, MICROUSD, NANOUSD, convert, to_units  # noqa: E402
from smythe.constrained_planner import ConstrainedArchitect  # noqa: E402
from smythe.optimize import (  # noqa: E402
    Candidate, EvaluatorManifest, ExperimentContract, ExperimentLedger, MetricObjective,
    MutableFieldRule, OptimizationRunner, TrialContext, TrialOutcome, sha256_prefixed,
)
from smythe.optimize.ledger import holdout_identity  # noqa: E402
from smythe.planner import SimpleArchitect  # noqa: E402
from smythe.provider import (  # noqa: E402
    CompletionResult, OfflineProvider, Provider, ProviderFeatures,
)
from smythe.templates import TemplateCatalog, TemplateRecord, template_target  # noqa: E402

DEFAULT_OUT = Path(__file__).with_name("results") / "template_eval_offline.json"
RECORD_VERSION = 1
TEMPLATE = "report"
TARGET = template_target(TEMPLATE)
SINGLE_AGENT = "single_agent"
# A stage fails when the single worker is better on the primary objective.
REFERENCE_RULE = "must_not_be_worse"
SELECTION_MODEL = "offline-fixture-selector"
SPLITS = ("development", "confirmation", "holdout")
ROLES = ("incumbent", "challenger", "reference")
# Declared usage is priced at the harness's blended token rate, in nanoUSD
# per token; a pass's total is converted to microUSD rounded up.
NANOUSD_PER_TOKEN = to_units(BLENDED_USD_PER_TOKEN, NANOUSD)
PRICE_SNAPSHOT_VERSION = f"offline-fixture-{NANOUSD_PER_TOKEN}-nanousd-per-token-v1"
REQUIRED_GATES = ("acceptance.policy_unchanged", "accounting.complete",
                  "applicability.excluded_untouched", "accepted.nonzero")
# Trials per split; Autotune's floor for the paired splits is three.
DEVELOPMENT_TRIALS, CONFIRMATION_TRIALS, HOLDOUT_TRIALS = 1, 3, 3
# A notional per-trial reservation for declared fixture usage; nothing is spent.
TRIAL_RESERVATION_MICROUSD = 1_000_000
# Harness source the evaluator hash binds, so a changed evaluator is a new identity.
EVALUATOR_SOURCES = ("benchmarks/offline_arms.py", "benchmarks/template_eval.py")
PACK_VERSION = 1
MAX_OUTPUT_CHARS = 8_000
# The sections each family's deliverable needs, which its acceptance policy requires.
SECTIONS = {
    "briefing": ("Findings", "Risks", "Sources"),
    "incident-review": ("Findings", "Timeline", "Sources"),
    "code-change": ("Change", "Tests"),
}
GOALS = {
    "briefing": "Brief the operations team on {subject}.",
    "incident-review": "Review the incident in which {subject}.",
    "code-change": "Summarize the change that {subject}.",
}
# The family no template version may be applied to.
EXCLUDED_FAMILY = "code-change"
SCOPE = (
    "Fixture runs establish harness mechanics, not template quality. Every step answers with "
    "fixture content and declared usage, faults are planted where the fixture versions differ, "
    "and the holdout family is committed with the harness: held out from development and "
    "confirmation, not hidden from whoever writes candidates. The campaign thresholds are "
    "placeholders, no decision is reviewed or activated, no default changes, and no number "
    "here is a product, README or roadmap claim."
)
DECISION_STATUS = {
    "mechanics_only": True, "activates": False,
    "note": "each decision is the Autotune engine's own, kept as it ran, in a ledger the run "
            "discards; a promoted decision on fixtures is never reviewed or activated",
}
_SECTION_STEP = re.compile(r"Write the ([A-Z][A-Za-z]*) section")
_SECTION_LIST = re.compile(r"Sections: ([^.]+)\.")


class PackError(ValueError):
    """A generated task pack does not match the SHA-256 it is pinned to."""


@dataclass(frozen=True)
class EvalTask:
    """One generated task: its family, its goal, and the section steps whose first answer fails.

    The goal ends with the sections the family's deliverable needs, so a
    single worker writes them all. ``sha256`` identifies the task's data.
    """

    family: str
    task_id: str
    goal: str
    planted: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "task_id": self.task_id, "goal": self.goal,
                "planted": list(self.planted)}

    @property
    def sha256(self) -> str:
        return sha256_prefixed(self.to_dict())

    def task(self) -> Task:
        return Task(goal=self.goal)


def _task(family: str, task_id: str, subject: str, *planted: str) -> EvalTask:
    unknown = set(planted) - set(SECTIONS[family])
    if unknown:
        raise ValueError(f"Task {task_id!r} plants a fault in sections {sorted(unknown)} "
                         f"that a {family} deliverable does not have")
    goal = f"{GOALS[family].format(subject=subject)} Sections: {', '.join(SECTIONS[family])}."
    return EvalTask(family, task_id, goal, tuple(planted))


@dataclass(frozen=True)
class TaskPack:
    """Generated tasks and the families no template may apply to, identified by their manifest."""

    name: str
    tasks: tuple[EvalTask, ...]
    excluded_families: tuple[str, ...]
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = [task.task_id for task in self.tasks]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Task pack {self.name!r} needs tasks with distinct ids")
        object.__setattr__(self, "manifest_sha256", sha256_prefixed(self.manifest()))

    def manifest(self) -> dict[str, Any]:
        return {"pack": self.name, "version": PACK_VERSION,
                "excluded_families": list(self.excluded_families),
                "tasks": [task.to_dict() for task in self.tasks]}


@dataclass(frozen=True)
class PinnedPack:
    """A task pack's builder and the SHA-256 its manifest must have.

    The pin identifies the pack without building it, so an evaluator
    manifest can bind a pack whose tasks have not been loaded.
    """

    name: str
    build: Callable[[], TaskPack]
    sha256: str

    def load(self) -> TaskPack:
        pack = self.build()
        if pack.name != self.name or pack.manifest_sha256 != self.sha256:
            raise PackError(f"Task pack {self.name!r} does not match its pinned SHA-256")
        return pack


def development_pack() -> TaskPack:
    return TaskPack("template-eval-development", (
        _task("briefing", "briefing-outage", "last week's payment gateway outage"),
        _task("briefing", "briefing-vendor", "the move to a new storage vendor", "Risks"),
        _task("briefing", "briefing-audit", "the quarterly access audit", "Sources"),
        _task("code-change", "change-retry", "adds jitter to the retry helper"),
    ), (EXCLUDED_FAMILY,))


def confirmation_pack() -> TaskPack:
    return TaskPack("template-eval-confirmation", (
        _task("briefing", "briefing-latency", "the rise in checkout latency"),
        _task("briefing", "briefing-oncall", "the new on-call rotation", "Sources"),
        _task("briefing", "briefing-capacity", "the capacity plan for the holiday peak", "Risks"),
        _task("code-change", "change-logging", "removes duplicate request logging"),
    ), (EXCLUDED_FAMILY,))


def holdout_pack() -> TaskPack:
    return TaskPack("template-eval-holdout", (
        _task("incident-review", "incident-dns", "a DNS change took the status page offline"),
        _task("incident-review", "incident-queue", "a stuck queue delayed order emails",
              "Timeline"),
        _task("incident-review", "incident-cert", "an expired certificate broke the partner API",
              "Sources"),
        _task("code-change", "change-cache", "bounds the size of the session cache"),
    ), (EXCLUDED_FAMILY,))


# Each split's pack and its pinned manifest SHA-256; a changed pack needs a new pin.
PACKS = {
    "development": PinnedPack(
        "template-eval-development", development_pack,
        "sha256:ed74b0e52ba7462e151071ac9b8ffdc45a8fc6a1e5b3c3e39ba6d903c62d5779"),
    "confirmation": PinnedPack(
        "template-eval-confirmation", confirmation_pack,
        "sha256:fc353f4d3ab99d5f39fad6af07f82a1a0454fb0221b27e7215f1b5647a2e6db1"),
    "holdout": PinnedPack(
        "template-eval-holdout", holdout_pack,
        "sha256:f74ee7702e30ecd426b2b221dd6b00c91151aaf05dbca8d241a4a042405fc831"),
}


def acceptance_policy(family: str) -> AcceptancePolicy:
    """The strict policy that judges every arm's output on *family*: its sections, in order."""
    return AcceptancePolicy(mode="strict", final_validators=[
        TextValidator({"sections": list(SECTIONS[family]), "max_chars": MAX_OUTPUT_CHARS})])


POLICIES = {family: acceptance_policy(family) for family in SECTIONS}


def split_manifest(pack: PinnedPack) -> dict[str, Any]:
    """What one split is: its pinned task pack and the policy that judges each family."""
    return {"pack": pack.name, "pack_sha256": pack.sha256,
            "acceptance": {family: POLICIES[family].sha256 for family in sorted(POLICIES)}}


def code_digest() -> str:
    """SHA-256 of the evaluator's source files, line endings normalized."""
    return sha256_prefixed({name: digest((REPO_ROOT / name).read_bytes())
                            for name in EVALUATOR_SOURCES})


def evaluator_manifest(packs: Mapping[str, PinnedPack]) -> EvaluatorManifest:
    """The evaluator identity: code, each split's pack and policies, validators and price."""
    return EvaluatorManifest(
        code_digest=code_digest(),
        split_manifests={split: sha256_prefixed(split_manifest(packs[split])) for split in SPLITS},
        validators=tuple(identity for family in sorted(POLICIES)
                         for identity in POLICIES[family].validator_identities),
        price_snapshot_version=PRICE_SNAPSHOT_VERSION)


def template_applies(record: TemplateRecord, family: str) -> bool:
    """Whether *record*'s applicability admits a task of *family*."""
    applicability = record.applicability or {}
    if family in applicability.get("excluded_task_families", ()):
        return False
    families = applicability.get("task_families", ())
    return not families or family in families


class SectionWriter(Provider):
    """The fixture model for every step: it writes the sections a step names.

    A step labeled ``Write the <Section> section`` gets that section. Any
    other step, such as a single worker's whole task, gets every section its
    goal lists after ``Sections:``. Each section is a Markdown heading and
    one line of fixture text. The first answer to a section step named in
    *planted* raises :class:`FixtureFault` before any usage, as a transient
    failure; a later answer to that step succeeds. Usage is declared from
    the size of each prompt and answer.
    """

    def __init__(self, planted: Iterable[str] = ()) -> None:
        self.planted = set(planted)
        self.calls: list[dict[str, Any]] = []

    def feature_support(self, model: str) -> ProviderFeatures:
        return ProviderFeatures(model_supported=True, text=True, tools=False, attachments=False,
                                image_output=False, cost_ceiling_required=False, sdk_retries=0)

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        lines = prompt.splitlines() or [""]
        step = next((line.removeprefix("Your step: ") for line in lines
                     if line.startswith("Your step: ")), lines[0])
        section = _SECTION_STEP.fullmatch(step)
        if section is not None:
            sections = [section.group(1)]
            if sections[0] in self.planted:
                self.planted.discard(sections[0])
                self.calls.append({"step": sections[0], "failed": True, "tokens": 0})
                raise FixtureFault(f"planted transient failure of the {sections[0]} step")
        else:
            listed = _SECTION_LIST.search(step)
            sections = ([] if listed is None
                        else [item.strip() for item in listed.group(1).split(",")])
        text = "\n\n".join(f"## {name}\n\nFixture text for the {name} section."
                           for name in sections)
        result = CompletionResult(text, declared_tokens(system + prompt), declared_tokens(text))
        self.calls.append({"step": section.group(1) if section else "task", "failed": False,
                           "tokens": result.total_tokens})
        return result


class ScriptedSelection(OfflineProvider):
    """An OfflineProvider whose scripted answer selects template *name*; it keeps each usage."""

    def __init__(self, name: str) -> None:
        super().__init__(responses=[json.dumps([{"template": name}])])
        self.usage: list[int] = []

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        result = await super().complete(system, prompt, model)
        self.usage.append(result.total_tokens)
        return result


def run_task(task: EvalTask, record: TemplateRecord | None) -> dict:
    """Run *task* once on the arm *record* defines, None for the single worker, and record it.

    A record's template runs only on a task of a family it applies to;
    any other task runs on a SimpleArchitect. Every outcome is recorded,
    whatever it is.
    """
    applies = record is not None and template_applies(record, task.family)
    selector = ScriptedSelection(TEMPLATE) if applies else None
    writer = SectionWriter(task.planted)
    policy = POLICIES[task.family]

    def build(checkpoints):
        architect = (ConstrainedArchitect(selector, [record.to_template()], model=SELECTION_MODEL)
                     if applies else SimpleArchitect())
        return Swarm(provider=writer, model=FIXTURE_MODEL, architect=architect,
                     acceptance=policy, checkpoint_store=checkpoints)

    run = execute_attempt(build, task.task())
    acceptance, graph = run.acceptance, run.graph or {}
    decision = None if acceptance is None else acceptance["decision"]
    nodes = graph.get("nodes", [])
    selection = [] if selector is None else selector.usage
    tokens = sum(call["tokens"] for call in writer.calls) + sum(selection)
    return scrub_record({
        "family": task.family,
        "task_id": task.task_id,
        "task_sha256": task.sha256,
        "planted": list(task.planted),
        "template_applied": applies,
        # What the run built, from its graph: each template record a selection used.
        "template_versions": [item["version"] for item in
                              (graph.get("planning") or {}).get("templates", [])],
        "outcome": "error" if run.error is not None else decision,
        "decision": decision,
        "reasons": [] if acceptance is None else list(acceptance.get("reasons", [])),
        "policy_sha256": None if acceptance is None else acceptance["policy_sha256"],
        "validators": [] if acceptance is None else [
            {**item["validator"], "outcome": item["outcome"]} for item in acceptance["validators"]],
        "output_sha256": (None if run.output is None
                          else hashlib.sha256(run.output.encode()).hexdigest()),
        "error": (None if run.error is None
                  else {"type": type(run.error).__name__, "message": str(run.error)[:300]}),
        "selection_calls": len(selection),
        "step_calls": len(writer.calls),
        "failed_calls": sum(call["failed"] for call in writer.calls),
        "tokens": tokens,
        "cost_nanousd": tokens * NANOUSD_PER_TOKEN,
        # The fixture providers declare every call's usage; a raised call had none.
        "cost_known": run.result is None or run.result.cost_is_complete,
        "node_count": len(nodes),
        "node_statuses": dict(sorted(Counter(node.get("status") for node in nodes).items())),
    })


def pass_order(tasks: Sequence[EvalTask], seed: int) -> list[EvalTask]:
    """The order one trial runs *tasks* in: by SHA-256 of the seed and each task id."""
    return sorted(tasks, key=lambda task: hashlib.sha256(
        json.dumps([seed, task.task_id]).encode()).hexdigest())


def _task_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    """What a trial keeps of one task's record; the pack lists the task itself."""
    return {"task_id": item["task_id"], "family": item["family"], "outcome": item["outcome"],
            "reasons": item["reasons"] if item["outcome"] != "accepted" else [],
            "template_built": bool(item["template_versions"]),
            **{key: item[key] for key in ("selection_calls", "step_calls", "failed_calls",
                                          "tokens", "node_count", "node_statuses")}}


def run_pass(pack: TaskPack, record: TemplateRecord | None, seed: int) -> dict:
    """One trial: every task of *pack* once on the arm *record* defines, with its metrics and gates.

    The gates read each task's full record; the trial keeps a summary of each.
    """
    results = [run_task(task, record) for task in pass_order(pack.tasks, seed)]
    excluded = set(pack.excluded_families)
    accepted = sum(item["outcome"] == "accepted" for item in results)
    metrics = {
        "accepted_count": accepted,
        "attempted_count": len(results),
        "attributable_cost_microusd": convert(sum(item["cost_nanousd"] for item in results),
                                              NANOUSD, MICROUSD, CEILING),
        "node_count": sum(item["node_count"] for item in results),
    }
    gates = {
        "acceptance.policy_unchanged": all(
            policy_unchanged({"attempts": [item]}, POLICIES[item["family"]]) for item in results),
        "accounting.complete": all(item["cost_known"] for item in results),
        "applicability.excluded_untouched": not any(
            item["template_versions"] or item["selection_calls"]
            for item in results if item["family"] in excluded),
        "accepted.nonzero": accepted > 0,
    }
    return {"pack": pack.name, "pack_sha256": pack.manifest_sha256, "metrics": metrics,
            "gates": gates, "tasks": [_task_summary(item) for item in results]}


class TemplateEvaluator:
    """The trial evaluator: one trial is one pass over its split's task pack.

    A pack is built, and checked against its pin, when the first trial of
    its split runs, so the holdout pack is loaded only by a holdout trial.
    A policy's value is a template version this catalog holds, or
    ``single_agent``. ``trials`` keeps every trial in evaluation order.
    """

    def __init__(self, catalog: TemplateCatalog,
                 packs: Mapping[str, PinnedPack] | None = None) -> None:
        self.catalog = catalog
        self.packs = dict(PACKS if packs is None else packs)
        if set(self.packs) != set(SPLITS):
            raise ValueError(f"packs must name exactly the splits {list(SPLITS)}")
        self.loaded: dict[str, TaskPack] = {}
        self.trials: list[dict] = []
        self._records: dict[str, TemplateRecord] = {}

    def manifest(self) -> EvaluatorManifest:
        return evaluator_manifest(self.packs)

    def pack(self, split: str) -> TaskPack:
        if split not in self.loaded:
            self.loaded[split] = self.packs[split].load()
        return self.loaded[split]

    def arm(self, value: str) -> TemplateRecord | None:
        """The record a policy value names, or None for the single worker."""
        if value == SINGLE_AGENT:
            return None
        if value not in self._records:
            record = self.catalog.get(value)
            if record.name != TEMPLATE:
                raise ValueError(f"Template version {value} is named {record.name!r}, "
                                 f"not {TEMPLATE!r}")
            self._records[value] = record
        return self._records[value]

    async def __call__(self, context: TrialContext) -> TrialOutcome:
        pack = self.pack(context.split)
        value = context.candidate.policy[TARGET]
        trial = await asyncio.to_thread(run_pass, pack, self.arm(value), context.seed)
        cost = trial["metrics"]["attributable_cost_microusd"]
        if cost > context.ceiling_microusd:
            raise RuntimeError("declared fixture usage exceeded the trial reservation")
        self.trials.append({"trial_key": context.trial_key, "split": context.split,
                            "role": context.phase.split(".", 1)[0],
                            "candidate_id": context.candidate.candidate_id, "value": value,
                            "seed": context.seed, **trial})
        return TrialOutcome(metrics=trial["metrics"], gates=trial["gates"],
                            actual_cost_microusd=cost)


def cost_per_accepted_outcome(trials: Iterable[Mapping[str, Any]]) -> float | str:
    """Cost per accepted outcome: the trials' summed cost over their summed accepted outcomes.

    A ratio of sums, never a mean of per-trial ratios, so a trial counts
    by its outcomes. With no accepted outcome it is ``"undefined"``, never
    zero or infinity.
    """
    cost = accepted = 0
    for trial in trials:
        cost += trial["metrics"]["attributable_cost_microusd"]
        accepted += trial["metrics"]["accepted_count"]
    return "undefined" if accepted == 0 else round(cost / accepted, 3)


def arm_report(trials: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict]]:
    """Each role's totals on each split it ran: outcomes, cost and cost per accepted outcome."""
    report: dict[str, dict[str, dict]] = {}
    for role in ROLES:
        for split in SPLITS:
            own = [trial for trial in trials if trial["role"] == role and trial["split"] == split]
            if own:
                report.setdefault(role, {})[split] = {
                    "trials": len(own),
                    "accepted": sum(trial["metrics"]["accepted_count"] for trial in own),
                    "attempted": sum(trial["metrics"]["attempted_count"] for trial in own),
                    "cost_microusd": sum(trial["metrics"]["attributable_cost_microusd"]
                                         for trial in own),
                    "cost_per_accepted_outcome_microusd": cost_per_accepted_outcome(own),
                }
    return report


def campaign_contract(values: Sequence[str], *, seed: int) -> ExperimentContract:
    """The contract for an incumbent, one challenger and the single-worker reference."""
    trials = 2 * DEVELOPMENT_TRIALS + 3 * (CONFIRMATION_TRIALS + HOLDOUT_TRIALS)
    return ExperimentContract(
        name=f"template-eval-{TEMPLATE}",
        objectives=(MetricObjective("accepted_count", "maximize", primary=True),
                    MetricObjective("attributable_cost_microusd", "minimize"),
                    MetricObjective("node_count", "minimize"),
                    MetricObjective("attempted_count", "maximize")),
        mutable_fields=(TARGET,),
        mutable_field_rules={TARGET: MutableFieldRule("string", choices=(*values, SINGLE_AGENT))},
        development_repetitions=DEVELOPMENT_TRIALS, confirmation_repetitions=CONFIRMATION_TRIALS,
        holdout_repetitions=HOLDOUT_TRIALS,
        max_candidates=3, max_parallel_candidates=1, max_trials=trials, max_wall_seconds=900,
        max_budget_microusd=trials * TRIAL_RESERVATION_MICROUSD,
        per_trial_reservation_microusd=TRIAL_RESERVATION_MICROUSD,
        # Placeholders: promotion thresholds are not decided for this evaluation.
        confidence=0.8, min_improvement=0.0, base_seed=seed, required_gates=REQUIRED_GATES,
    )


async def run_campaign(
    catalog: TemplateCatalog, predecessor: str, candidate: str, *, seed: int = 0,
    evaluator: TemplateEvaluator | None = None,
) -> dict:
    """Evaluate template version *candidate* against *predecessor* and the single worker.

    The predecessor is the incumbent, the candidate the challenger, and
    ``single_agent`` a reference under ``must_not_be_worse``, in a ledger
    that is discarded afterwards.
    """
    evaluator = TemplateEvaluator(catalog) if evaluator is None else evaluator
    manifest = evaluator.manifest()
    contract = campaign_contract((predecessor, candidate), seed=seed)
    incumbent = Candidate(contract=contract, policy={TARGET: predecessor},
                          hypothesis="The predecessor template version stays in effect.")
    challenger = Candidate(contract=contract, policy={TARGET: candidate}, parent=incumbent,
                           hypothesis="The candidate template version replaces its predecessor.")
    reference = Candidate(contract=contract, policy={TARGET: SINGLE_AGENT},
                          hypothesis="One worker receives the whole task, with no template.")
    roles = {"incumbent": incumbent, "challenger": challenger, "reference": reference}
    with tempfile.TemporaryDirectory(prefix="smythe-template-eval-",
                                     ignore_cleanup_errors=True) as work:
        with ExperimentLedger(Path(work) / "campaign.sqlite3", durability="normal") as ledger:
            runner = OptimizationRunner(contract, ledger, evaluator,
                                        evaluator_hash=manifest.evaluator_hash)
            result = await runner.run(incumbent, (challenger,), references=(reference,),
                                      reference_rule=REFERENCE_RULE)
    decision = result.to_dict()
    spend = decision.pop("ledger_snapshot")["cost"]
    decision.pop("trial_keys")
    holdout_trials = [index for index, trial in enumerate(evaluator.trials)
                      if trial["split"] == "holdout"]
    return {
        "roles": {"incumbent": "predecessor", "challenger": "candidate",
                  "references": [SINGLE_AGENT]},
        "policies": {role: dict(item.policy) for role, item in roles.items()},
        "candidates": {role: item.candidate_id for role, item in roles.items()},
        "reference_rule": REFERENCE_RULE,
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "thresholds": "placeholders for offline mechanics; promotion thresholds are not decided",
        "decision_status": DECISION_STATUS,
        "evaluator_manifest": manifest.to_dict(),
        "evaluator_hash": manifest.evaluator_hash,
        "split_manifests": {split: split_manifest(evaluator.packs[split]) for split in SPLITS},
        "holdout_identity": holdout_identity(contract, manifest.evaluator_hash),
        "holdout": {"pack": evaluator.packs["holdout"].name,
                    "held_out": "a task family development and confirmation do not contain",
                    "loaded": "holdout" in evaluator.loaded,
                    "first_holdout_trial": holdout_trials[0] if holdout_trials else None},
        "decision": decision,
        "report": {
            "arms": arm_report(evaluator.trials),
            "campaign_cost_microusd": spend,
            "note": "cost per accepted outcome is the summed cost over the summed accepted "
                    "outcomes of an arm's trials; the campaign's own spend is reported apart "
                    "and never netted against a difference between arms",
        },
        "trials": evaluator.trials,
    }


def _step(section: str, *, retries: int) -> dict[str, Any]:
    """A blueprint node that writes *section*: skipped when it fails, or retried *retries* times."""
    return {"id": section.lower(), "label": f"Write the {section} section", "depends_on": [],
            "capabilities": [], "profile": None, "verifies": None, "max_regenerations": 0,
            "failure_policy": "retry" if retries else "skip", "max_retries": retries,
            "timeout_s": None, "attach_dep_artifacts": False, "metadata": {}}


_APPLICABILITY = {"task_families": [], "keywords": [], "required_context_keys": [],
                  "excluded_task_families": [EXCLUDED_FAMILY]}
# Each fixture version: what it changes, its sections, the steps it retries
# once, and whether it keeps the predecessor's exclusion.
FIXTURE_VERSIONS = {
    "predecessor": ("Writes Findings, Risks, Timeline and Sources, one step each; a failed "
                    "step is skipped.", ("Findings", "Risks", "Timeline", "Sources"), (), True),
    "improvement": ("Retries every section step once.",
                    ("Findings", "Risks", "Timeline", "Sources"),
                    ("Findings", "Risks", "Timeline", "Sources"), True),
    "harmful": ("Retries every section step once and drops the Timeline step, which no "
                "briefing needs.", ("Findings", "Risks", "Sources"),
                ("Findings", "Risks", "Sources"), True),
    "partial": ("Retries the Sources step once.", ("Findings", "Risks", "Timeline", "Sources"),
                ("Sources",), True),
    "unexcluded": ("Retries every section step once and no longer excludes code changes.",
                   ("Findings", "Risks", "Timeline", "Sources"),
                   ("Findings", "Risks", "Timeline", "Sources"), False),
}


def fixture_catalog(root: str | Path) -> tuple[TemplateCatalog, dict[str, str]]:
    """A catalog under *root* holding every fixture version, and each label's version."""
    catalog = TemplateCatalog(root)
    versions: dict[str, str] = {}
    for label, (description, sections, retried, excludes) in FIXTURE_VERSIONS.items():
        record = TemplateRecord(
            name=TEMPLATE, description=description,
            blueprint=[_step(section, retries=int(section in retried)) for section in sections],
            applicability=_APPLICABILITY if excludes else None,
            parent_version=versions.get("predecessor"))
        versions[label] = catalog.put(record)
    return catalog, versions


def _pack_record(pack: TaskPack) -> dict[str, Any]:
    return {"pack": pack.name, "sha256": pack.manifest_sha256,
            "excluded_families": list(pack.excluded_families),
            "tasks": [{"task_id": task.task_id, "family": task.family, "sha256": task.sha256,
                       "planted": list(task.planted)} for task in pack.tasks]}


def evaluate(*, seed: int) -> dict:
    """Run every fixture candidate's campaign against the predecessor and build the record."""
    with tempfile.TemporaryDirectory(prefix="smythe-template-catalog-",
                                     ignore_cleanup_errors=True) as root:
        catalog, versions = fixture_catalog(root)
        labels = {version: label for label, version in versions.items()}
        campaigns = {}
        for label in FIXTURE_VERSIONS:
            if label == "predecessor":
                continue
            campaign = asyncio.run(run_campaign(catalog, versions["predecessor"],
                                                versions[label], seed=seed))
            for trial in campaign["trials"]:
                trial["arm"] = labels.get(trial["value"], trial["value"])
            campaigns[label] = campaign
        templates = {label: {"template_version": version,
                             "record": catalog.get(version).to_dict()}
                     for label, version in versions.items()}
    # Described only now, after every campaign has run.
    packs = {split: _pack_record(PACKS[split].load()) for split in SPLITS}
    return offline_record(
        "template-evaluation",
        record_version=RECORD_VERSION,
        scope=SCOPE,
        default_change=False,
        promotion={"eligible": False, "reasons": [
            "fixture content establishes harness mechanics, not template quality",
            "promotion thresholds are not decided; the campaigns' are placeholders",
            "the holdout family is committed with the harness, so it is held out from "
            "development and confirmation but not hidden from whoever writes candidates",
            "no decision is reviewed or activated in a policy registry",
        ]},
        target=TARGET,
        arms={
            "template version": "a ConstrainedArchitect over the catalog record of the version "
                                "the policy names, with an OfflineProvider scripting its "
                                "selection; a task of a family the record does not apply to "
                                "runs on a SimpleArchitect",
            SINGLE_AGENT: "a SimpleArchitect: one worker receives the whole task; a sealed "
                          "reference arm",
        },
        templates=templates,
        acceptance={family: policy_record(policy) for family, policy in POLICIES.items()},
        packs=packs,
        cost_protocol={"unit": "microUSD", "source": "declared fixture usage",
                       "model": FIXTURE_MODEL, "selection_model": SELECTION_MODEL,
                       "nanousd_per_token": NANOUSD_PER_TOKEN,
                       "price_snapshot_version": PRICE_SNAPSHOT_VERSION,
                       "conversion": "each pass's nanoUSD total in microUSD, rounded up",
                       "scope": "the template selection and every step of every task in a pass",
                       "invoice_cost": False},
        campaigns=campaigns,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    record = evaluate(seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    for label, campaign in record["campaigns"].items():
        decision = campaign["decision"]
        references = [item.get("verdict") for item in decision.get("references", [])]
        print(f"  {label:<12} verdict {decision['verdict']:<12} "
              f"promoted {decision['promoted']!s:<5} single_agent {references}")
    print(f"Wrote {out} (offline mechanics; not claimable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
