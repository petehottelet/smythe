"""Offline selector comparison: four ways to choose a graph, on both task families.

    python benchmarks/run_selector_comparison.py
    python benchmarks/run_selector_comparison.py --heldout-pack DIR --out PATH

Arms, each decided by the same strict acceptance policy:

    single_worker       one worker receives the whole task; no planning call
    fixed_template      the research -> analyze -> write pipeline for every task
    generated_plan      the LLMArchitect plans a task-specific graph
    adaptive_selection  a WhiteRabbit router picks one of the other three per
                        task: developer routing rules first, then a scripted
                        classifier whose usage is declared and charged

Families, development-style material only:

    task-shapes         benchmarks/tasks_shapes/*.yaml
    supplied-source     the Astra pack's development (pilot) tasks and their
                        source packs, loaded with manifest verification

For each family the harness runs every arm on every task in a seeded,
counterbalanced (Williams) order, with a matched allowance of two attempts
per cell. One task per family plants a failure in every arm's first attempt
(a raised fault in task-shapes, a rejected output in supplied-source), so
the record shows failed attempts kept and charged. Pinned regression cases
(a reference output that must be accepted, a planted defect that must be
rejected) give the evaluator's no_new_failures gate. Then an Autotune
campaign compares the incumbent generated_plan with the challenger
adaptive_selection, with single_worker and fixed_template as references, on
task-indexed trials; its evaluator hash comes from an EvaluatorManifest that
binds the split manifests, the validator identities and the fixture price.

Every adaptive attempt is charged its selector cost (zero for a rule, the
classifier's declared usage otherwise) and carries its routing time from
the run's timing. Fixture runs establish harness mechanics, not arm
rankings: model steps answer with fixture content and declared usage, the
campaign's thresholds are placeholders, and the record is labeled
offline-mechanics, claimable false, api_calls 0. It changes no default.

No held-out pack is committed. Without --heldout-pack the holdout split
reuses the development tasks and the record says so. With it, a pack in the
Astra pack format that lives outside the repository is verified against its
manifest and its main-stage tasks become the supplied-source holdout split;
the record must then also be written outside the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.astra_campaign import (  # noqa: E402
    CampaignPlanError, TaskCase, TaskPack, check_output, load_task_pack, provider_task,
)
from benchmarks.astra_campaign._json import TEXT_HASH_POLICY, digest  # noqa: E402
from benchmarks.astra_campaign.tasks import MAX_OUTPUT_BYTES  # noqa: E402
from benchmarks.harness import PIPELINE_SPECS, SELECTION_RULES, load_tasks  # noqa: E402
from benchmarks.offline_arms import (  # noqa: E402
    ARMS, CLASSIFIER_MODEL, FIXTURE_MODEL, MICROUSD_PER_TOKEN, PRICE_SNAPSHOT_VERSION,
    SCHEDULE_ALGORITHM, ArmTask, ScriptedClassifier, offline_record, policy_record,
    policy_unchanged, run_cell, summarize, williams_schedule,
)

from smythe import AcceptancePolicy, JSONValidator, Task, TextValidator  # noqa: E402
from smythe.optimize import (  # noqa: E402
    Candidate, EvaluatorManifest, ExperimentContract, ExperimentLedger, MetricObjective,
    MutableFieldRule, OptimizationRunner, TrialContext, TrialOutcome, sha256_prefixed,
)

SHAPES_DIR = Path(__file__).with_name("tasks_shapes")
DEFAULT_OUT = Path(__file__).with_name("results") / "selector_comparison_offline.json"
RECORD_VERSION = 1
INCUMBENT, CHALLENGER = "generated_plan", "adaptive_selection"
REFERENCES = ("single_worker", "fixed_template")
REQUIRED_GATES = ("accepted.nonzero", "acceptance.policy_unchanged", "accounting.complete",
                  "regression.no_new_failures")
ATTEMPT_ALLOWANCE = 2
# Trials per split; Autotune's floor for the paired splits is three.
DEVELOPMENT_TRIALS, CONFIRMATION_TRIALS, HOLDOUT_TRIALS = 1, 3, 3
# A notional per-trial reservation for declared fixture usage; nothing is spent.
TRIAL_RESERVATION_MICROUSD = 1_000_000
HYPOTHESES = {
    "single_worker": "One worker receives the whole task.",
    "fixed_template": "The fixed research, analysis and writing pipeline runs every task.",
    "generated_plan": "The architect generates a task-specific plan for every task.",
    "adaptive_selection": "Routing rules, then a classifier, choose one worker, the fixed "
                          "pipeline or a generated plan for each task.",
}
# Harness source the evaluator hash binds, so a changed evaluator is a new identity.
EVALUATOR_SOURCES = ("benchmarks/harness.py", "benchmarks/offline_arms.py",
                     "benchmarks/run_selector_comparison.py")
SCOPE = (
    "Fixture runs establish harness mechanics, not arm rankings. Every model step answers "
    "with fixture content and declared usage, so differences in acceptance, cost and time "
    "reflect fixture design and arm structure. The campaign thresholds are placeholders, no "
    "default changes, and no number here is a product, README or roadmap claim."
)


@dataclass(frozen=True)
class Family:
    """One task family: its tasks, its strict policy and what its splits cover."""

    name: str
    source: str
    tasks: tuple[ArmTask, ...]
    policy: AcceptancePolicy
    split_manifest: dict
    regression_task: str
    holdout: tuple[ArmTask, ...] = ()
    holdout_manifest: dict | None = None


def shape_family() -> Family:
    """The task-shape suite; deep-serial's first attempts meet a planted fault."""
    files = sorted(SHAPES_DIR.glob("*.yaml"))
    benches = load_tasks(SHAPES_DIR)
    tasks = tuple(
        ArmTask("task-shapes", bench.name, bench.to_task(),
                deliverable=f"Fixture deliverable for {bench.name}.\n\n{bench.goal}",
                defect="TODO.", sha256=digest(path.read_bytes()),
                planted="fault" if bench.name == "deep-serial" else None)
        for bench, path in zip(benches, files, strict=True))
    policy = AcceptancePolicy(mode="strict", final_validators=[
        TextValidator({"min_chars": 40, "max_chars": MAX_OUTPUT_BYTES})])
    manifest = {"family": "task-shapes", "text_hash_policy": TEXT_HASH_POLICY,
                "files": {f"benchmarks/tasks_shapes/{path.name}": digest(path.read_bytes())
                          for path in files}}
    return Family("task-shapes", "benchmarks/tasks_shapes/*.yaml", tasks, policy, manifest,
                  regression_task="trivial-transform")


def _place(value, path, item):
    """Put *item* at *path* inside *value*, creating objects and arrays on the way."""
    if not path:
        return item
    head, rest = path[0], path[1:]
    if isinstance(head, int):
        container = value if isinstance(value, list) else []
        container.extend([None] * (head + 1 - len(container)))
        container[head] = _place(container[head], rest, item)
    else:
        container = value if isinstance(value, dict) else {}
        container[head] = _place(container.get(head), rest, item)
    return container


_FILLERS = {
    "equals": lambda check: check["expected"],
    "number": lambda check: check["expected"],
    "nonempty": lambda check: "fixture answer".ljust(check["min_chars"], "."),
    "object": lambda check: {f"item-{n}": "fixture"
                             for n in range(1, max(check["minimum"], 1) + 1)},
    "length": lambda check: ["fixture"] * check["expected"],
    "keys": lambda check: dict.fromkeys(check["expected"], "fixture"),
    "citations": lambda check: list(check["allowed"][:max(check["minimum"], 1)]),
}


def reference_answer(case: TaskCase) -> str:
    """A JSON answer built from the task's answer key that passes every deterministic check.

    Shallower checks are placed first, so a deeper check fills in the object
    a shallower one declared. An answer key no single value satisfies is a
    fixture error.
    """
    value = None
    for check in sorted(case.checks, key=lambda item: len(item["path"])):
        value = _place(value, check["path"], _FILLERS[check["kind"]](check))
    text = json.dumps(value, sort_keys=True)
    if not check_output(case, text)["deterministic_passed"]:
        raise CampaignPlanError(f"No reference answer satisfies the checks of {case.task_id}")
    return text


def supplied_source_tasks(
    pack: TaskPack, stage: str, *, planted: str | None = None,
) -> tuple[ArmTask, ...]:
    """A pack stage's tasks with their sources; outputs are adjudicated on the answer key.

    The fixture answer is the reference answer, and the planted defect is
    that answer inside a Markdown fence, which strict JSON acceptance rejects.
    """
    tasks = []
    for case in pack.for_stage(stage):
        data = provider_task(case)
        answer = reference_answer(case)
        tasks.append(ArmTask(
            "supplied-source", case.task_id,
            Task(goal=data["goal"], constraints=data["constraints"],
                 done_when=data["done_when"], context=data["context"]),
            deliverable=answer, defect=f"```json\n{answer}\n```", sha256=case.task_sha256,
            planted="defect" if case.task_id == planted else None,
            adjudicate=partial(check_output, case)))
    return tuple(tasks)


def _stage_manifest(pack: TaskPack, stage: str) -> dict:
    return {"family": "supplied-source", "text_hash_policy": TEXT_HASH_POLICY,
            "pack_manifest_sha256": pack.manifest_sha256, "stage": stage,
            "tasks": {case.task_id: [case.task_sha256, case.source_sha256]
                      for case in pack.for_stage(stage)}}


def supplied_source_family(heldout: TaskPack | None = None) -> Family:
    """The Astra pack's development tasks; pilot-relays' first attempts meet a planted defect."""
    pack = load_task_pack()
    policy = AcceptancePolicy(mode="strict", final_validators=[
        JSONValidator({"max_bytes": MAX_OUTPUT_BYTES})])
    family = Family(
        "supplied-source", "benchmarks/astra_campaign/data (pilot stage)",
        supplied_source_tasks(pack, "pilot", planted="pilot-relays"), policy,
        _stage_manifest(pack, "pilot"), regression_task="pilot-calendar")
    if heldout is None:
        return family
    return replace(family, holdout=supplied_source_tasks(heldout, "main"),
                   holdout_manifest=_stage_manifest(heldout, "main"))


def _inside_repository(path: str | Path) -> bool:
    return Path(path).resolve().is_relative_to(REPO_ROOT.resolve())


def load_heldout_pack(directory: str | Path) -> TaskPack:
    """Load a held-out pack from outside the repository, verifying its manifest.

    The pack uses the Astra pack format: every file must match the SHA-256
    its manifest lists, and nothing unlisted, missing or linked is accepted.
    The committed pack is not held out, so a copy of it is refused.
    """
    if _inside_repository(directory):
        raise CampaignPlanError("A held-out pack must live outside the repository")
    pack = load_task_pack(Path(directory))
    if pack.manifest_sha256 == load_task_pack().manifest_sha256:
        raise CampaignPlanError("The committed task pack is not a held-out pack")
    return pack


def run_regression(family: Family) -> dict:
    """Pinned cases for the no_new_failures gate, run once per arm with one attempt each.

    A case pinned ``accepted`` fails when its output is not accepted; one
    pinned ``rejected`` fails when its planted defect is accepted.
    """
    task = next(item for item in family.tasks if item.task_id == family.regression_task)
    cases = (("reference-output", replace(task, planted=None), "accepted"),
             ("planted-defect", replace(task, deliverable=task.defect, planted=None), "rejected"))
    arms = {}
    for arm in ARMS:
        runs = []
        for name, case_task, pinned in cases:
            cell = run_cell(case_task, arm, family.policy, allowance=1)
            observed = cell["attempts"][-1]["outcome"]
            runs.append({"case": name, "task_id": task.task_id, "pinned": pinned,
                         "observed": observed,
                         "passed": (observed == "accepted") == (pinned == "accepted"),
                         "attempts": cell["attempts"]})
        arms[arm] = {"passed": all(run["passed"] for run in runs), "cases": runs}
    return {"mode": "no_new_failures", "task_id": task.task_id, "arms": arms}


def run_sweep(family: Family, *, seed: int, repetitions: int) -> tuple[list, list]:
    """Every arm on every task and repetition, in the seeded Williams order."""
    schedule = williams_schedule(family.name, [task.task_id for task in family.tasks],
                                 seed=seed, repetitions=repetitions)
    tasks = {task.task_id: task for task in family.tasks}
    cells = []
    for slot in schedule:
        cell = run_cell(tasks[slot["task_id"]], slot["arm"], family.policy,
                        allowance=ATTEMPT_ALLOWANCE, repetition=slot["repetition"])
        cells.append({"ordinal": slot["ordinal"], "block": slot["block"],
                      "position": slot["position"], **cell})
    return schedule, cells


def code_digest() -> str:
    """SHA-256 of the evaluator's source files, line endings normalized."""
    return sha256_prefixed({name: digest((REPO_ROOT / name).read_bytes())
                            for name in EVALUATOR_SOURCES})


def _contract(family: Family, seed: int) -> ExperimentContract:
    # The incumbent and challenger in development, then every arm on both paired splits.
    trials = (2 * DEVELOPMENT_TRIALS
              + (2 + len(REFERENCES)) * (CONFIRMATION_TRIALS + HOLDOUT_TRIALS))
    return ExperimentContract(
        name=f"selector-comparison-{family.name}",
        objectives=(MetricObjective("cost_microusd", "minimize", primary=True),
                    MetricObjective("accepted_outcome_rate", "maximize"),
                    MetricObjective("elapsed_ms", "minimize")),
        mutable_fields=("arm",),
        mutable_field_rules={"arm": MutableFieldRule("string", choices=ARMS)},
        development_repetitions=DEVELOPMENT_TRIALS, confirmation_repetitions=CONFIRMATION_TRIALS,
        holdout_repetitions=HOLDOUT_TRIALS,
        max_candidates=len(ARMS), max_parallel_candidates=1, max_trials=trials,
        max_wall_seconds=900, max_budget_microusd=trials * TRIAL_RESERVATION_MICROUSD,
        per_trial_reservation_microusd=TRIAL_RESERVATION_MICROUSD,
        # Placeholders: promotion thresholds are not decided for this comparison.
        confidence=0.8, min_improvement=0.0, base_seed=seed, required_gates=REQUIRED_GATES,
    )


async def run_campaign(family: Family, regression: dict, *, seed: int) -> dict:
    """The Autotune campaign: incumbent, challenger and two references on task-indexed trials."""
    contract = _contract(family, seed)
    candidates = {arm: Candidate(contract=contract, policy={"arm": arm},
                                 hypothesis=HYPOTHESES[arm]) for arm in ARMS}
    arm_of = {candidate.candidate_id: arm for arm, candidate in candidates.items()}
    holdout = family.holdout or family.tasks
    holdout_manifest = family.holdout_manifest or family.split_manifest
    development = sha256_prefixed(family.split_manifest)
    manifest = EvaluatorManifest(
        code_digest=code_digest(),
        split_manifests={"development": development, "confirmation": development,
                         "holdout": sha256_prefixed(holdout_manifest)},
        validators=family.policy.validator_identities,
        price_snapshot_version=PRICE_SNAPSHOT_VERSION)
    pools = {"development": family.tasks, "confirmation": family.tasks, "holdout": holdout}
    trials: list[dict] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        arm = context.candidate.policy["arm"]
        task = pools[context.split][context.task_index]
        cell = await asyncio.to_thread(run_cell, task, arm, family.policy,
                                       allowance=ATTEMPT_ALLOWANCE)
        if cell["cost_microusd"] > context.ceiling_microusd:
            raise RuntimeError("declared fixture usage exceeded the trial reservation")
        gates = {
            "accepted.nonzero": cell["accepted"],
            "acceptance.policy_unchanged": policy_unchanged(cell, family.policy),
            "accounting.complete": all(item["cost_known"] and not item["cost_is_estimate"]
                                       for item in cell["attempts"]),
            "regression.no_new_failures": regression["arms"][arm]["passed"],
        }
        trials.append({"trial_key": context.trial_key, "split": context.split,
                       "role": context.phase.split(".", 1)[0], "task_index": context.task_index,
                       "gates": gates, **cell})
        return TrialOutcome(
            metrics={"cost_microusd": cell["cost_microusd"],
                     "accepted_outcome_rate": float(cell["accepted"]),
                     "elapsed_ms": cell["elapsed_ms"]},
            gates=gates, actual_cost_microusd=cell["cost_microusd"])

    with tempfile.TemporaryDirectory(prefix="smythe-selector-", ignore_cleanup_errors=True) as work:
        with ExperimentLedger(Path(work) / "campaign.sqlite3", durability="normal") as ledger:
            runner = OptimizationRunner(
                contract, ledger, evaluate, evaluator_hash=manifest.evaluator_hash,
                task_pools={split: len(tasks) for split, tasks in pools.items()})
            result = await runner.run(
                candidates[INCUMBENT], (candidates[CHALLENGER],),
                references=tuple(candidates[arm] for arm in REFERENCES))
    decision = result.to_dict()
    for key in ("ledger_snapshot", "trial_keys"):
        decision.pop(key)
    decision["incumbent_arm"] = arm_of[result.incumbent_candidate_id]
    decision["selected_arm"] = arm_of.get(result.selected_candidate_id)
    for item in (*decision["development_scores"], *decision.get("references", [])):
        item["arm"] = arm_of[item["candidate_id"]]
    return {
        "roles": {"incumbent": INCUMBENT, "challenger": CHALLENGER,
                  "references": list(REFERENCES)},
        "candidates": {arm: candidate.candidate_id for arm, candidate in candidates.items()},
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "thresholds": "placeholders for offline mechanics; promotion thresholds are not decided",
        "evaluator_manifest": manifest.to_dict(),
        "evaluator_hash": manifest.evaluator_hash,
        "split_manifests": {"development": family.split_manifest,
                            "confirmation": family.split_manifest,
                            "holdout": holdout_manifest},
        "task_pools": {split: [task.task_id for task in tasks] for split, tasks in pools.items()},
        "holdout": ({"source": "heldout-pack", "held_out": True}
                    if family.holdout else
                    {"source": "development-stand-in", "held_out": False,
                     "note": "no held-out pack was supplied, so the holdout split reuses the "
                             "development tasks and its verdict shows mechanics only"}),
        "decision": decision,
        "trials": trials,
    }


def compare(families: list[Family], *, seed: int, repetitions: int) -> dict:
    """Run every family's sweep, regression cases and campaign, and build the record."""
    results = {}
    for family in families:
        schedule, cells = run_sweep(family, seed=seed, repetitions=repetitions)
        regression = run_regression(family)
        campaign = asyncio.run(run_campaign(family, regression, seed=seed))
        results[family.name] = {
            "source": family.source,
            "tasks": [{"task_id": task.task_id, "sha256": task.sha256, "planted": task.planted}
                      for task in family.tasks],
            "acceptance_policy": policy_record(family.policy),
            "adjudication": ("deterministic answer-key checks, apart from acceptance"
                             if any(task.adjudicate for task in family.tasks) else None),
            "summary": summarize(cells),
            "runs": cells,
            "regression": regression,
            "campaign": campaign,
        }
    return offline_record(
        "selector-comparison",
        record_version=RECORD_VERSION,
        scope=SCOPE,
        default_change=False,
        promotion={"eligible": False, "reasons": [
            "offline fixture content establishes harness mechanics, not arm rankings",
            "promotion thresholds are not decided; the campaign's are placeholders",
            "a promotion needs a held-out pack kept outside the repository",
        ]},
        arms=list(ARMS),
        arm_definitions={
            "single_worker": "SimpleArchitect: one worker receives the whole task",
            "fixed_template": {"description": "the research -> analyze -> write pipeline",
                               "template_sha256": sha256_prefixed(
                                   [list(spec) for spec in PIPELINE_SPECS])},
            "generated_plan": "LLMArchitect: the fixture model returns a fork-join plan",
            "adaptive_selection": "WhiteRabbit: routing rules, then the scripted classifier, "
                                  "choose single, deterministic:pipeline or autonomous",
        },
        selection={
            "rules": [rule.to_dict() for rule in SELECTION_RULES],
            "classifier": {"model": CLASSIFIER_MODEL,
                           "routes": [list(route) for route in ScriptedClassifier().routes],
                           "default": ScriptedClassifier().default,
                           "usage": "declared: UTF-8 bytes / 4 tokens at the fixture price"},
            "charge": "each adaptive attempt's cost includes its selector cost, and its elapsed "
                      "time includes its routing time, recorded from the run's timing",
        },
        schedule={"algorithm": SCHEDULE_ALGORITHM, "seed": seed, "repetitions": repetitions,
                  "workflow_concurrency": 1, "attempt_allowance": ATTEMPT_ALLOWANCE},
        cost_protocol={"unit": "microUSD", "source": "declared fixture usage",
                       "model": FIXTURE_MODEL, "microusd_per_token": MICROUSD_PER_TOKEN,
                       "price_snapshot_version": PRICE_SNAPSHOT_VERSION,
                       "scope": "routing, planning, and every attempt's steps and synthesis",
                       "invoice_cost": False},
        timing_protocol={"clock": "process-local monotonic",
                         "elapsed": "each attempt's SwarmResult.timing elapsed_ms; a raised "
                                    "attempt's from its failed checkpoint",
                         "selector_latency": "phases_ms.routing",
                         "note": "local time on one machine; not a latency claim"},
        families=results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1, choices=range(1, 11),
                        metavar="1-10")
    parser.add_argument("--heldout-pack", default=None,
                        help="a held-out pack directory outside the repository")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    heldout = None
    if args.heldout_pack is not None:
        if _inside_repository(args.out):
            parser.error("with --heldout-pack, write the record outside the repository")
        try:
            heldout = load_heldout_pack(args.heldout_pack)
        except CampaignPlanError as exc:
            parser.error(str(exc))
    record = compare([shape_family(), supplied_source_family(heldout)],
                     seed=args.seed, repetitions=args.repetitions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, family in record["families"].items():
        for arm, entry in family["summary"].items():
            print(f"  {name:<16} {arm:<19} accepted {entry['accepted']}/{entry['cells']}  "
                  f"failed attempts {entry['failed_attempts']}  cost {entry['cost_microusd']} uUSD")
        decision = family["campaign"]["decision"]
        print(f"  {name:<16} campaign verdict {decision['verdict']}; "
              f"references {[item['verdict'] for item in decision.get('references', [])]}")
    print(f"Wrote {out} (offline mechanics; not claimable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
