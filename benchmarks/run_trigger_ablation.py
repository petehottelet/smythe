"""Offline trigger ablation: routine model review against evidence-triggered review.

    python benchmarks/run_trigger_ablation.py
    python benchmarks/run_trigger_ablation.py --seed 3 --out PATH

Arms, each running the same generated plans with the same fixture worker
and decided by the same strict acceptance policy:

    routine    LLMSupervisor on its default schedule: a review when a
               multi-input join becomes ready and when the graph finishes
    triggered  EvidenceTriggeredSupervisor around the same LLMSupervisor,
               with the default ReviewTriggers: a review only when a
               trigger fires, with what it found ahead of the usual prompt
    none       no supervisor

Both supervised arms may revise twice and review with one scripted offline
reviewer. It reads the run state the review prompt shows, never the
evidence ahead of it, so one state gets one decision in either arm: when a
completed step's output is empty, notes an input that was not covered, or
is a failing verdict, it adds a step that revises the deliverable;
otherwise it proposes no change.

Fixture tasks plan the fork-join research plan (two investigations, then
the deliverable), some with a gate on the deliverable. Clean tasks run as
planned. Each anomaly task plants one anomaly in its steps: an empty risk
investigation, a failed or truncated one under a skip policy, a source
evidence block with a contradicting passage, or a gate that rejects the
deliverable after its regeneration. Each anomaly is of a kind the default
triggers detect, and each is met by every arm; an anomaly that no trigger
detects would be reviewed only on a routine schedule.

Every arm runs every task once in a seeded, counterbalanced order, with one
attempt per cell: review is the only recovery under test, so no retry can
repair what review missed. A run executes one node at a time, so every arm
meets the same order of completions. The record counts each arm's
supervision calls, their declared cost, the triggers that fired, the
revisions applied and the outcomes strict acceptance accepted. Then an
Autotune campaign compares the incumbent routine with the challenger
triggered, with none as a reference arm, on task-indexed trials that cover
every task in each paired split. Its primary objective is supervision
calls, and accepted outcomes must not regress.

Fixture runs establish harness mechanics, not a measured benefit: the
reviewer is scripted, the anomalies are planted where the triggers look,
and the record is labeled offline-mechanics, claimable false, api_calls 0.
Live evaluation needs an approved budget and decided thresholds. It changes
no default.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.astra_campaign._json import TEXT_HASH_POLICY, canonical, digest  # noqa: E402
from benchmarks.harness import OFFLINE_DYNAMIC_PLAN, make_swarm  # noqa: E402
from benchmarks.offline_arms import (  # noqa: E402
    FIXTURE_MODEL, MICROUSD_PER_TOKEN, PRICE_SNAPSHOT_VERSION, ArmTask, AttemptRun, FixtureFault,
    attempt_record, declared_tokens, execute_attempt, offline_record, policy_record,
    policy_unchanged, run_cell, summarize,
)

from smythe import (  # noqa: E402
    AcceptancePolicy, EvidenceTriggeredSupervisor, LLMSupervisor, ReviewTriggers, Task,
    TextValidator,
)
from smythe.evidence import (  # noqa: E402
    ClaimPassage, ClaimRecord, SourceEvidence, render_source_evidence,
)
from smythe.optimize import (  # noqa: E402
    Candidate, EvaluatorManifest, ExperimentContract, ExperimentLedger, MetricObjective,
    MutableFieldRule, OptimizationRunner, TrialContext, TrialOutcome, sha256_prefixed,
)
from smythe.prompts import PLANNING_SYSTEM_PROMPT  # noqa: E402
from smythe.provider import CompletionResult, Provider, ProviderFeatures  # noqa: E402
from smythe.supervisor import EVIDENCE_REVIEW_PROMPT, SUPERVISOR_SYSTEM_PROMPT  # noqa: E402

DEFAULT_OUT = Path(__file__).with_name("results") / "trigger_ablation_offline.json"
RECORD_VERSION = 1
ARMS = ("routine", "triggered", "none")
INCUMBENT, CHALLENGER, REFERENCE = ARMS
FAMILY = "trigger-fixtures"
REVIEWER_MODEL = "offline-fixture-reviewer"
MAX_REVISIONS = 2
TRIGGERS = ReviewTriggers()
# Review is the only recovery under test, so a cell has one attempt.
ATTEMPT_ALLOWANCE = 1
DEVELOPMENT_TRIALS = 1
TRIAL_RESERVATION_MICROUSD = 1_000_000
REQUIRED_GATES = ("acceptance.policy_unchanged", "accounting.complete")
EVALUATOR_SOURCES = ("benchmarks/harness.py", "benchmarks/offline_arms.py",
                     "benchmarks/run_trigger_ablation.py")
SCHEDULE_ALGORITHM = "sha256-order-williams-three-arm-v1"
# Every order of three arms: across every six blocks each arm takes each
# position twice and directly follows each other arm twice.
SCHEDULE_ROWS = tuple(permutations(range(len(ARMS))))

# Each planted anomaly and the default triggers that detect it, in the order they fire.
ANOMALIES = {
    "empty_output": ("empty_or_short_output",),
    "failed_step": ("failed_or_skipped",),
    "truncated_step": ("failed_or_skipped", "incomplete_output"),
    "contradicting_evidence": ("contradiction_records",),
    "rejecting_gate": ("verifier_rejected_or_exhausted", "acceptance_not_accepted"),
}
ANOMALY_PLANTS = {
    "empty_output": "the risk investigation answers with empty text",
    "failed_step": "the risk investigation raises, under a skip policy",
    "truncated_step": "the risk investigation is cut off at its output limit, under a skip policy",
    "contradicting_evidence": "the facts investigation carries a source evidence block whose "
                              "claim has a contradicting passage",
    "rejecting_gate": "a gate rejects the deliverable after its one regeneration",
}
HYPOTHESES = {
    "routine": "LLMSupervisor reviews at every multi-input join and when the graph finishes.",
    "triggered": "Deterministic review triggers call the same reviewer only on recorded "
                 "evidence.",
    "none": "No supervisor reviews the run.",
}
SCOPE = (
    "Fixture runs establish harness mechanics, not a measured benefit. A scripted reviewer "
    "decides every review, the fixture worker writes every step, and the anomalies are "
    "planted of kinds the default triggers detect, so equal accepted outcomes and fewer "
    "supervision calls follow from the fixtures and arm structure. The campaign thresholds "
    "are placeholders, no default changes, and no number here is a product, README or "
    "roadmap claim."
)

# Step ids of the fixture plans, and the step a revision adds.
FACTS, RISKS, DELIVER, CHECK, REVISE = "facts", "risks", "deliver", "check", "revise"
CHECK_LABEL = "Check that the deliverable's risks name a mitigation"
REVISE_LABEL = "Revise the final deliverable so that it covers every section"
# How the fixture worker notes a section whose input was missing or empty.
NOT_COVERED = "Not covered:"
CONSTRAINT = "Write the deliverable in Markdown with a Findings section and a Risks section."


@dataclass(frozen=True)
class AblationTask(ArmTask):
    """A fixture task and the anomaly its steps meet, or None for a clean task.

    ``findings`` and ``risks`` are the bodies of the deliverable's two
    sections, and ``gated`` adds a gate on the deliverable. ``deliverable``
    is what a complete run delivers and ``defect`` what it delivers when the
    risk investigation's output is missing or empty.
    """

    anomaly: str | None = None
    findings: str = ""
    risks: str = ""
    gated: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.anomaly is not None and self.anomaly not in ANOMALIES:
            raise ValueError(
                f"Unknown anomaly {self.anomaly!r}; expected one of {list(ANOMALIES)}")

    @property
    def expected_triggers(self) -> tuple[str, ...]:
        return () if self.anomaly is None else ANOMALIES[self.anomaly]


def _section(name: str, body: str | None) -> str:
    if body is None:
        return f"{NOT_COVERED} {name.lower()}; its input was missing or empty."
    return f"## {name}\n{body}"


def fixture_task(
    task_id: str, goal: str, findings: str, risks: str, *, anomaly: str | None = None,
    gated: bool = False,
) -> AblationTask:
    """One fixture task; a rejecting-gate anomaly needs, and adds, the gate."""
    gated = gated or anomaly == "rejecting_gate"
    spec = {"family": FAMILY, "task_id": task_id, "goal": goal, "constraints": [CONSTRAINT],
            "findings": findings, "risks": risks, "anomaly": anomaly, "gated": gated}
    return AblationTask(
        FAMILY, task_id, Task(goal=goal, constraints=[CONSTRAINT]),
        deliverable=f"{_section('Findings', findings)}\n\n{_section('Risks', risks)}",
        defect=f"{_section('Findings', findings)}\n\n{_section('Risks', None)}",
        sha256=digest(canonical(spec).encode("utf-8")),
        anomaly=anomaly, findings=findings, risks=risks, gated=gated,
    )


def fixture_tasks() -> tuple[AblationTask, ...]:
    """Four clean tasks, one of them gated, and one task for each planted anomaly."""
    return (
        fixture_task("rail-pass", "Assess demand for a regional rail pass.",
                     "Ridership on the corridor grew 12% last year.",
                     "A fare increase could push commuters back to driving."),
        fixture_task("clinic-hours", "Evaluate extending a clinic's evening hours.",
                     "Most missed appointments fall after 5 pm.",
                     "Evening staffing may need overtime pay."),
        fixture_task("warehouse-robots", "Evaluate a pilot of warehouse picking robots.",
                     "Picking accounts for half of warehouse labor hours.",
                     "Robots may stall on irregular parcels."),
        fixture_task("library-app", "Assess relaunching a library's mobile app.",
                     "Two thirds of renewals already happen online.",
                     "Older patrons may resist a new sign-in flow. Mitigation: keep the "
                     "desk renewal path.", gated=True),
        fixture_task("solar-permits", "Assess a faster permit process for rooftop solar.",
                     "Permits take six weeks on average.",
                     "Faster review could miss structural problems.", anomaly="empty_output"),
        fixture_task("port-expansion", "Evaluate expanding the container port's night shift.",
                     "Night berths sit idle four nights a week.",
                     "Residents may object to night-time noise.", anomaly="failed_step"),
        fixture_task("bike-share", "Assess adding electric bikes to the bike share.",
                     "Trips over three kilometres doubled in a year.",
                     "Battery swaps add a daily maintenance round.", anomaly="truncated_step"),
        fixture_task("coffee-sourcing", "Evaluate sourcing coffee from a new cooperative.",
                     "The cooperative's harvest grew 20% this year.",
                     "A single source concentrates supply risk.",
                     anomaly="contradicting_evidence"),
        fixture_task("payroll-migration", "Assess migrating payroll to a hosted service.",
                     "Payroll runs take two days of manual checks.",
                     "Migration could delay one pay cycle.", anomaly="rejecting_gate"),
    )


def acceptance_policy() -> AcceptancePolicy:
    """The strict policy every arm shares: both sections, with content."""
    return AcceptancePolicy(mode="strict", final_validators=[
        TextValidator({"min_chars": 40, "sections": ["Findings", "Risks"]})])


def task_plan(task: AblationTask) -> dict:
    """The plan the fixture worker returns for *task*: the fork-join research plan."""
    plan = copy.deepcopy(OFFLINE_DYNAMIC_PLAN)
    nodes = {node["id"]: node for node in plan["nodes"]}
    if task.anomaly in ("failed_step", "truncated_step"):
        nodes[RISKS]["failure_policy"] = "skip"
    if task.gated:
        plan["nodes"].append({
            "id": CHECK, "label": CHECK_LABEL, "depends_on": [DELIVER], "verifies": DELIVER,
            "max_regenerations": 1,
            "agent": {"name": "Checker", "persona": "You check deliverables strictly."},
        })
    return plan


def _features() -> ProviderFeatures:
    return ProviderFeatures(model_supported=True, text=True, tools=False, attachments=False,
                            image_output=False, cost_ceiling_required=False, sdk_retries=0)


class FixtureWorker(Provider):
    """Deterministic offline worker: it plans the task, then answers each step by its label.

    The investigations write their sections, or meet the task's anomaly.
    The deliver step writes each section whose body its inputs carry and
    notes each one they do not, so a run whose risk investigation is missing
    or empty delivers the task's ``defect``. A revision step writes the whole
    deliverable. Usage is declared from the size of each prompt and answer,
    as ``offline_arms.FixtureModel`` declares it.
    """

    def __init__(self, task: AblationTask) -> None:
        self.task = task
        self.plan = task_plan(task)
        self.steps = {node["label"]: node["id"] for node in self.plan["nodes"]}
        self.calls: list[dict[str, Any]] = []

    def feature_support(self, model: str) -> ProviderFeatures:
        return _features()

    def _step(self, prompt: str) -> str | None:
        label = prompt.split("Your step: ", 1)[-1].split("\n", 1)[0]
        return REVISE if label == REVISE_LABEL else self.steps.get(label)

    def _answer(self, step: str | None, prompt: str) -> tuple[str, str | None]:
        task = self.task
        if step == FACTS:
            text = _section("Findings", task.findings)
            if task.anomaly == "contradicting_evidence":
                claim = ClaimRecord(
                    "harvest", task.findings,
                    supporting=(ClaimPassage("cooperative-report", "grew 20%"),),
                    contradicting=(ClaimPassage("import-records", "grew 5%"),))
                text += "\n\n" + render_source_evidence(SourceEvidence(claims=(claim,)))
            return text, None
        if step == RISKS:
            if task.anomaly == "empty_output":
                return "", None
            if task.anomaly == "failed_step":
                raise FixtureFault("planted failure: the risk register is unavailable")
            if task.anomaly == "truncated_step":
                return "## Risks\nThe main risk", "max_tokens"
            return _section("Risks", task.risks), None
        if step == DELIVER:
            return "\n\n".join(_section(name, body if body in prompt else None)
                               for name, body in (("Findings", task.findings),
                                                  ("Risks", task.risks))), None
        if step == CHECK:
            if task.anomaly == "rejecting_gate":
                return "FAIL - the Risks section names no mitigation", None
            return "PASS", None
        if step == REVISE:
            return task.deliverable, None
        raise FixtureFault("no fixture answer for this step")

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        planning = system == PLANNING_SYSTEM_PROMPT
        step = None if planning else self._step(prompt)
        try:
            text, stop = (json.dumps(self.plan), None) if planning else self._answer(step, prompt)
        except FixtureFault:
            self.calls.append({"planning": False, "failed": True, "tokens": 0, "step": step})
            raise
        result = CompletionResult(text, declared_tokens(system + prompt), declared_tokens(text),
                                  stop_reason=stop)
        self.calls.append({"planning": planning, "failed": False,
                           "tokens": result.total_tokens, "step": step})
        return result


# Where LLMSupervisor's REVIEW_PROMPT places the run state, and how a
# triggered review's prompt begins.
_COMPLETED = "\n\nCompleted work so far:\n"
_REMAINING = "\n\nRemaining planned steps (not yet executed):\n"
_FINISHED = "\n\nThe step that just finished was "
_ENTRY = re.compile(r"^\[([A-Za-z0-9_-]{1,64})\] [^\n]*$", re.MULTILINE)
_EVIDENCE_LEAD = EVIDENCE_REVIEW_PROMPT.split("{evidence}", 1)[0]


def review_state(prompt: str) -> tuple[dict[str, str], list[str], str]:
    """The run state a review prompt shows: completed outputs, pending ids, the step just done.

    Read from the end of the prompt, where ``REVIEW_PROMPT`` places it, so
    evidence ahead of it is never read. Completed outputs are as the prompt
    quotes them.
    """
    head, found, tail = prompt.rpartition(_FINISHED)
    head, remaining_found, remaining = head.rpartition(_REMAINING)
    _, completed_found, completed = head.rpartition(_COMPLETED)
    if not (found and remaining_found and completed_found):
        raise ValueError("The prompt is not a review prompt")
    entries = list(_ENTRY.finditer(completed))
    outputs = {
        entry.group(1): completed[entry.end():entries[index + 1].start()
                                  if index + 1 < len(entries) else len(completed)].strip()
        for index, entry in enumerate(entries)
    }
    pending = _ENTRY.findall(remaining)
    return outputs, pending, tail.split("'", 2)[1]


def _defective(output: str) -> bool:
    return not output or NOT_COVERED in output or output.startswith("FAIL")


class ScriptedReviewer(Provider):
    """A deterministic offline reviewer that answers ``LLMSupervisor`` review prompts.

    It decides from :func:`review_state`, never from trigger evidence. A
    completed step is defective when its output is empty, notes an input
    that was not covered, or is a failing verdict. With a defective step and
    no revision step yet, it adds one after the deliver step; otherwise it
    proposes no change. Its usage is declared at the fixture price and
    reported as ``cost_usd``, as ``offline_arms.ScriptedClassifier`` does.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def feature_support(self, model: str) -> ProviderFeatures:
        return _features()

    @staticmethod
    def decide(prompt: str) -> dict[str, Any]:
        outputs, pending, _ = review_state(prompt)
        if REVISE in outputs or REVISE in pending:
            return {"change": False, "reason": "a revision step already covers the gap"}
        defective = [step for step, output in outputs.items() if _defective(output)]
        if not defective:
            return {"change": False, "reason": "the remaining plan still fits"}
        return {"change": True, "reason": f"revise the deliverable: {', '.join(defective)} "
                                          "left it incomplete",
                "add": [{"id": REVISE, "label": REVISE_LABEL, "depends_on": [DELIVER]}]}

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if system != SUPERVISOR_SYSTEM_PROMPT:
            raise ValueError("The scripted reviewer answers review prompts only")
        decision = self.decide(prompt)
        reply = json.dumps(decision)
        prompt_tokens, completion_tokens = declared_tokens(system + prompt), declared_tokens(reply)
        cost = (prompt_tokens + completion_tokens) * MICROUSD_PER_TOKEN
        self.calls.append({
            "after_node": review_state(prompt)[2],
            "with_evidence": prompt.startswith(_EVIDENCE_LEAD),
            "change": decision["change"], "reply": reply,
            "tokens": prompt_tokens + completion_tokens, "cost_microusd": cost,
        })
        return CompletionResult(reply, prompt_tokens, completion_tokens,
                                cost_usd=float(Decimal(cost) / 1_000_000))


def _supervisor(arm: str, reviewer: ScriptedReviewer):
    if arm == "none":
        return None
    inner = LLMSupervisor(reviewer, model=REVIEWER_MODEL)
    return inner if arm == "routine" else EvidenceTriggeredSupervisor(inner, TRIGGERS)


def _supervision(run: AttemptRun, reviewer: ScriptedReviewer) -> dict[str, Any]:
    """What supervision did in one attempt, and its declared charge.

    Calls and usage are the reviewer's own; triggers and decisions come from
    the run's trace, which a raised attempt does not return.
    """
    trace = [] if run.result is None else run.result.trace
    decisions = [span for span in trace
                 if span.get("status") in ("revision_applied", "revision_rejected")]
    applied = [span for span in decisions if span["status"] == "revision_applied"]
    return {
        "calls": len(reviewer.calls),
        "proposals": sum(call["change"] for call in reviewer.calls),
        "reviews": [{key: call[key] for key in ("after_node", "with_evidence", "change")}
                    for call in reviewer.calls],
        "reviews_triggered": [span["review_trigger"] for span in trace if "review_trigger" in span],
        "revisions_applied": len(applied),
        "revisions_refused": len(decisions) - len(applied),
        "added_steps": [node for span in applied for node in span["revision"]["added"]],
        "tokens": sum(call["tokens"] for call in reviewer.calls),
        "cost_microusd": sum(call["cost_microusd"] for call in reviewer.calls),
        "latency_ms": ((run.timing or {}).get("phases_ms") or {}).get("supervision"),
    }


def run_attempt(
    arm_task: AblationTask, arm: str, policy: AcceptancePolicy, *, attempt: int,
) -> dict:
    """Run one attempt of *arm* on *arm_task* and record it, with what supervision did."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    worker, reviewer = FixtureWorker(arm_task), ScriptedReviewer()
    supervisor = _supervisor(arm, reviewer)
    # One node at a time: every arm meets the same order of completions, so
    # each review point sees the same state in every arm.
    run = execute_attempt(
        lambda checkpoints: make_swarm(
            "smythe_dynamic", worker, FIXTURE_MODEL, parallel=False, acceptance=policy,
            checkpoint_store=checkpoints, supervisor=supervisor,
            max_revisions=0 if supervisor is None else MAX_REVISIONS),
        arm_task.task)
    return attempt_record(arm_task, run, worker, attempt=attempt, planted=arm_task.anomaly,
                          charges={"supervision": _supervision(run, reviewer)})


def run_task(arm_task: AblationTask, arm: str, policy: AcceptancePolicy, *,
             repetition: int = 1) -> dict:
    """One cell: *arm* on *arm_task*, with its supervision calls summed."""
    cell = run_cell(arm_task, arm, policy, allowance=ATTEMPT_ALLOWANCE, repetition=repetition,
                    attempt_runner=run_attempt)
    cell["anomaly"] = arm_task.anomaly
    cell["supervision_calls"] = sum(item["supervision"]["calls"] for item in cell["attempts"])
    return cell


def balanced_schedule(task_ids: Sequence[str], *, seed: int, repetitions: int = 1) -> list[dict]:
    """Order task and repetition blocks, and counterbalance the three arms inside each block.

    Arms and blocks are ordered by SHA-256 of a seeded key, and block *i*
    runs its arms in ``SCHEDULE_ROWS[(i + offset) % 6]``, so across every
    six blocks each arm takes each position twice and directly follows each
    other arm twice.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")

    def key(value: object) -> str:
        payload = json.dumps([SCHEDULE_ALGORITHM, seed, value], sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    ordered = sorted(ARMS, key=key)
    blocks = sorted(((task_id, rep) for task_id in task_ids for rep in range(1, repetitions + 1)),
                    key=lambda pair: key(list(pair)))
    shift = int(key("row-offset")[:8], 16) % len(SCHEDULE_ROWS)
    schedule = []
    for block, (task_id, rep) in enumerate(blocks):
        for position, index in enumerate(SCHEDULE_ROWS[(block + shift) % len(SCHEDULE_ROWS)]):
            schedule.append({"ordinal": len(schedule), "block": block, "position": position,
                             "task_id": task_id, "repetition": rep, "arm": ordered[index]})
    return schedule


def run_sweep(tasks: Sequence[AblationTask], policy: AcceptancePolicy, *, seed: int,
              repetitions: int) -> list[dict]:
    """Every arm on every task and repetition, in the seeded counterbalanced order."""
    by_id = {task.task_id: task for task in tasks}
    cells = []
    for slot in balanced_schedule(list(by_id), seed=seed, repetitions=repetitions):
        cell = run_task(by_id[slot["task_id"]], slot["arm"], policy, repetition=slot["repetition"])
        cells.append({"ordinal": slot["ordinal"], "block": slot["block"],
                      "position": slot["position"], **cell})
    return cells


def supervision_summary(cells: Sequence[dict]) -> dict[str, dict]:
    """Per-arm supervision totals: calls, proposals, revisions, triggers and charge."""
    summary = {}
    for arm in ARMS:
        items = [item["supervision"] for cell in cells if cell["arm"] == arm
                 for item in cell["attempts"]]
        summary[arm] = {
            "calls": sum(item["calls"] for item in items),
            "reviews_with_evidence": sum(review["with_evidence"] for item in items
                                         for review in item["reviews"]),
            "proposals": sum(item["proposals"] for item in items),
            "revisions_applied": sum(item["revisions_applied"] for item in items),
            "revisions_refused": sum(item["revisions_refused"] for item in items),
            "triggers": dict(sorted(Counter(
                name for item in items for review in item["reviews_triggered"]
                for name in review["triggers"]).items())),
            "cost_microusd": sum(item["cost_microusd"] for item in items),
        }
    return summary


def compare_arms(cells: Sequence[dict]) -> dict[str, Any]:
    """Accepted outcomes and supervision calls per arm, task by task and in total."""
    rows: dict[tuple[str, int], dict[str, dict]] = {}
    for cell in cells:
        rows.setdefault((cell["task_id"], cell["repetition"]), {})[cell["arm"]] = cell
    tasks = [{"task_id": task_id, "repetition": rep,
              "anomaly": next(iter(arms.values()))["anomaly"],
              "accepted": {arm: arms[arm]["accepted"] for arm in ARMS},
              "supervision_calls": {arm: arms[arm]["supervision_calls"] for arm in ARMS}}
             for (task_id, rep), arms in sorted(rows.items())]
    accepted = {arm: sum(row["accepted"][arm] for row in tasks) for arm in ARMS}
    calls = {arm: sum(row["supervision_calls"][arm] for row in tasks) for arm in ARMS}
    return {
        "accepted": accepted,
        "supervision_calls": calls,
        "routine_and_triggered_accept_the_same_tasks": all(
            row["accepted"][INCUMBENT] == row["accepted"][CHALLENGER] for row in tasks),
        "triggered_makes_fewer_calls": calls[CHALLENGER] < calls[INCUMBENT],
        "accepted_only_with_supervision": sorted({
            row["task_id"] for row in tasks
            if row["accepted"][INCUMBENT] and not row["accepted"][REFERENCE]}),
        "tasks": tasks,
    }


def split_manifest(tasks: Sequence[AblationTask]) -> dict:
    return {"family": FAMILY, "text_hash_policy": TEXT_HASH_POLICY,
            "tasks": {task.task_id: task.sha256 for task in tasks}}


def code_digest() -> str:
    """SHA-256 of the evaluator's source files, line endings normalized."""
    return sha256_prefixed({name: digest((REPO_ROOT / name).read_bytes())
                            for name in EVALUATOR_SOURCES})


def _contract(tasks: int, seed: int) -> ExperimentContract:
    # Each paired split runs every arm once on every task.
    trials = 2 * DEVELOPMENT_TRIALS + len(ARMS) * 2 * tasks
    return ExperimentContract(
        name="trigger-ablation",
        objectives=(MetricObjective("supervision_calls", "minimize", primary=True),
                    # The criterion under test: no accepted outcome is lost.
                    MetricObjective("accepted_outcome_rate", "maximize", max_regression=0.0),
                    MetricObjective("cost_microusd", "minimize")),
        mutable_fields=("arm",),
        mutable_field_rules={"arm": MutableFieldRule("string", choices=ARMS)},
        development_repetitions=DEVELOPMENT_TRIALS, confirmation_repetitions=tasks,
        holdout_repetitions=tasks,
        max_candidates=len(ARMS), max_parallel_candidates=1, max_trials=trials,
        max_wall_seconds=900, max_budget_microusd=trials * TRIAL_RESERVATION_MICROUSD,
        per_trial_reservation_microusd=TRIAL_RESERVATION_MICROUSD,
        # Placeholders: promotion thresholds are not decided for this ablation.
        confidence=0.8, min_improvement=0.0, base_seed=seed, required_gates=REQUIRED_GATES,
    )


async def run_campaign(tasks: Sequence[AblationTask], policy: AcceptancePolicy, *,
                       seed: int) -> dict:
    """The Autotune campaign: incumbent routine, challenger triggered, reference none."""
    tasks = tuple(tasks)
    contract = _contract(len(tasks), seed)
    candidates = {arm: Candidate(contract=contract, policy={"arm": arm},
                                 hypothesis=HYPOTHESES[arm]) for arm in ARMS}
    arm_of = {candidate.candidate_id: arm for arm, candidate in candidates.items()}
    manifest_hash = sha256_prefixed(split_manifest(tasks))
    manifest = EvaluatorManifest(
        code_digest=code_digest(),
        split_manifests=dict.fromkeys(("development", "confirmation", "holdout"), manifest_hash),
        validators=policy.validator_identities,
        price_snapshot_version=PRICE_SNAPSHOT_VERSION)
    trials: list[dict] = []

    async def evaluate(context: TrialContext) -> TrialOutcome:
        arm = context.candidate.policy["arm"]
        cell = await asyncio.to_thread(run_task, tasks[context.task_index], arm, policy)
        if cell["cost_microusd"] > context.ceiling_microusd:
            raise RuntimeError("declared fixture usage exceeded the trial reservation")
        gates = {
            "acceptance.policy_unchanged": policy_unchanged(cell, policy),
            "accounting.complete": all(item["cost_known"] and not item["cost_is_estimate"]
                                       for item in cell["attempts"]),
        }
        trials.append({"trial_key": context.trial_key, "split": context.split,
                       "role": context.phase.split(".", 1)[0], "task_index": context.task_index,
                       "gates": gates, **cell})
        return TrialOutcome(
            metrics={"supervision_calls": cell["supervision_calls"],
                     "accepted_outcome_rate": float(cell["accepted"]),
                     "cost_microusd": cell["cost_microusd"]},
            gates=gates, actual_cost_microusd=cell["cost_microusd"])

    with tempfile.TemporaryDirectory(prefix="smythe-triggers-", ignore_cleanup_errors=True) as work:
        with ExperimentLedger(Path(work) / "campaign.sqlite3", durability="normal") as ledger:
            runner = OptimizationRunner(
                contract, ledger, evaluate, evaluator_hash=manifest.evaluator_hash,
                task_pools=dict.fromkeys(("development", "confirmation", "holdout"), len(tasks)))
            result = await runner.run(candidates[INCUMBENT], (candidates[CHALLENGER],),
                                      references=(candidates[REFERENCE],))
    decision = result.to_dict()
    for key in ("ledger_snapshot", "trial_keys"):
        decision.pop(key)
    decision["incumbent_arm"] = arm_of[result.incumbent_candidate_id]
    decision["selected_arm"] = arm_of.get(result.selected_candidate_id)
    for item in (*decision["development_scores"], *decision.get("references", [])):
        item["arm"] = arm_of[item["candidate_id"]]
    return {
        "roles": {"incumbent": INCUMBENT, "challenger": CHALLENGER, "references": [REFERENCE]},
        "candidates": {arm: candidate.candidate_id for arm, candidate in candidates.items()},
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "thresholds": "placeholders for offline mechanics; promotion thresholds are not decided",
        "decision_scope": "the decision of a campaign on fixtures, in a ledger the run discards: "
                          "it activates nothing and changes no default",
        "evaluator_manifest": manifest.to_dict(),
        "evaluator_hash": manifest.evaluator_hash,
        "split_manifest": split_manifest(tasks),
        "task_pool": [task.task_id for task in tasks],
        "holdout": {"source": "development-stand-in", "held_out": False,
                    "note": "the holdout split reuses the fixture tasks, so its verdict shows "
                            "mechanics only"},
        "decision": decision,
        "trials": trials,
    }


def ablate(tasks: Sequence[AblationTask], *, seed: int, repetitions: int) -> dict:
    """Run the sweep and the campaign on *tasks*, and build the record."""
    policy = acceptance_policy()
    cells = run_sweep(tasks, policy, seed=seed, repetitions=repetitions)
    campaign = asyncio.run(run_campaign(tasks, policy, seed=seed))
    summary = summarize(cells, ARMS)
    for arm, entry in supervision_summary(cells).items():
        summary[arm]["supervision"] = entry
    return offline_record(
        "trigger-ablation",
        record_version=RECORD_VERSION,
        scope=SCOPE,
        default_change=False,
        promotion={"eligible": False, "reasons": [
            "a scripted reviewer and planted anomalies establish harness mechanics, not a "
            "measured benefit",
            "promotion thresholds are not decided; the campaign's are placeholders",
            "live evaluation needs an approved budget and decided thresholds",
        ]},
        arms=list(ARMS),
        arm_definitions={
            "routine": "LLMSupervisor(only_terminal=True): a review when a multi-input join "
                       "becomes ready and when the graph finishes",
            "triggered": "EvidenceTriggeredSupervisor(LLMSupervisor, ReviewTriggers()): a review "
                         "only when a trigger fires, its evidence ahead of the review prompt",
            "none": "no supervisor",
        },
        supervision={
            "max_revisions": MAX_REVISIONS,
            "reviewer": {"model": REVIEWER_MODEL,
                         "decides_from": "the run state the review prompt shows, never the "
                                         "trigger evidence ahead of it",
                         "rule": "a completed step whose output is empty, notes an input that "
                                 "was not covered, or is a failing verdict gets one revision "
                                 "step after the deliver step; otherwise no change",
                         "usage": "declared: UTF-8 bytes / 4 tokens at the fixture price"},
            "triggers": TRIGGERS.to_dict(),
            "charge": "each attempt's cost includes its review calls' declared usage, which "
                      "an ordinary run keeps outside its own ledger",
        },
        fixtures={
            "family": FAMILY,
            "plan": "the fork-join research plan the fixture worker returns: facts and risks, "
                    "then deliver; a gated task adds a gate on deliver with one regeneration",
            "anomalies": {kind: {"planted": ANOMALY_PLANTS[kind],
                                 "expected_triggers": list(triggers)}
                          for kind, triggers in ANOMALIES.items()},
        },
        tasks=[{"task_id": task.task_id, "sha256": task.sha256, "anomaly": task.anomaly,
                "gated": task.gated, "expected_triggers": list(task.expected_triggers)}
               for task in tasks],
        acceptance_policy=policy_record(policy),
        schedule={"algorithm": SCHEDULE_ALGORITHM, "seed": seed, "repetitions": repetitions,
                  "workflow_concurrency": 1, "attempt_allowance": ATTEMPT_ALLOWANCE},
        cost_protocol={"unit": "microUSD", "source": "declared fixture usage",
                       "model": FIXTURE_MODEL, "reviewer_model": REVIEWER_MODEL,
                       "microusd_per_token": MICROUSD_PER_TOKEN,
                       "price_snapshot_version": PRICE_SNAPSHOT_VERSION,
                       "scope": "planning, every step, and every review call",
                       "invoice_cost": False},
        timing_protocol={"clock": "process-local monotonic",
                         "elapsed": "each attempt's SwarmResult.timing elapsed_ms",
                         "supervision_latency": "phases_ms.supervision",
                         "note": "local time on one machine; not a latency claim"},
        summary=summary,
        comparison=compare_arms(cells),
        runs=cells,
        campaign=campaign,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1, choices=range(1, 11),
                        metavar="1-10")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    record = ablate(fixture_tasks(), seed=args.seed, repetitions=args.repetitions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    for arm, entry in record["summary"].items():
        print(f"  {arm:<10} accepted {entry['accepted']}/{entry['cells']}  supervision calls "
              f"{entry['supervision']['calls']}  cost {entry['cost_microusd']} uUSD")
    decision = record["campaign"]["decision"]
    print(f"  campaign verdict {decision['verdict']}; "
          f"reference {[item['verdict'] for item in decision.get('references', [])]}")
    print(f"Wrote {out} (offline mechanics; not claimable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
