"""Shared offline arm runner for comparison and ablation harnesses.

Each arm plans and executes one task with deterministic local providers
under a strict acceptance policy that every arm shares, so the validator
identities that decide each run are identical across arms. A cell is one
task, arm and repetition with a matched allowance of attempts: an attempt
that raises, or whose output the policy does not accept, is recorded and
charged, and another follows until one is accepted or the allowance is
spent. Every attempt, failed ones included, stays in the cell. A harness
with arms of its own runs its attempts with :func:`execute_attempt`, records
them with :func:`attempt_record` and passes its runner to :func:`run_cell`.

The only providers are the fixture model and the scripted classifier
defined here, or a harness's own deterministic fixtures, and each attempt
builds them itself, so a record's ``api_calls`` is 0 by construction.
Their answers are fixture content written for the tasks and their usage
is declared, not billed: a run establishes harness mechanics (routing,
planning, acceptance, accounting, timing and attempt records), never model
quality, real cost or latency, or an arm ranking.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from benchmarks.artifact_records import environment_snapshot, scrub_record
from benchmarks.astra_campaign.schedule import WILLIAMS_ROWS
from benchmarks.harness import ADAPTIVE_SELECTION, OFFLINE_DYNAMIC_PLAN, make_swarm
from benchmarks.provider_usage import BLENDED_USD_PER_TOKEN

from smythe import AcceptancePolicy, Task
from smythe.checkpoint import CheckpointStore, graph_to_dict
from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import CompletionResult, Provider, ProviderFeatures
from smythe.task import snapshot_task

ARMS = ("single_worker", "fixed_template", "generated_plan", "adaptive_selection")
# The benchmarks/harness.py baseline that builds each arm.
ARM_BASELINES = {
    "single_worker": "single_agent",
    "fixed_template": "fixed_pipeline",
    "generated_plan": "smythe_dynamic",
    "adaptive_selection": ADAPTIVE_SELECTION,
}
FIXTURE_MODEL = "offline-fixture"
CLASSIFIER_MODEL = "offline-fixture-classifier"
# Declared usage is priced at the harness's blended token rate, kept in
# integer microUSD so that no amount is rounded.
MICROUSD_PER_TOKEN = round(BLENDED_USD_PER_TOKEN * 1_000_000)
PRICE_SNAPSHOT_VERSION = f"offline-fixture-{MICROUSD_PER_TOKEN}-microusd-per-token-v1"
SCHEDULE_ALGORITHM = "sha256-order-williams-four-arm-v1"
# Below this many runs a p95 lies between the two largest values.
SMALL_SAMPLE_RUNS = 20


def declared_tokens(text: str) -> int:
    """The fixture's declared token count for *text*: its UTF-8 bytes / 4, rounded up."""
    return (len(text.encode("utf-8")) + 3) // 4


def microusd(cost_usd: float) -> int:
    """A reported USD amount in microUSD, rounded up so that no charge shrinks."""
    return math.ceil(Decimal(repr(cost_usd)) * 1_000_000)


class FixtureFault(RuntimeError):
    """A planted transient failure: an attempt's first answer from the fixture model fails."""


class FixtureModel(Provider):
    """Deterministic offline model for every arm's planning and steps.

    A planning call returns ``plan``, as ``harness.offline_provider`` does
    for a generated plan. Every other call answers ``text``, so whichever
    step delivers a run's output delivers the fixture answer. With
    ``fail_first`` the first non-planning call raises :class:`FixtureFault`.
    Usage is declared from the size of each prompt and answer.
    """

    def __init__(self, text: str, *, plan: dict | None = None, fail_first: bool = False) -> None:
        self.text, self.plan, self.fail_first = text, plan, fail_first
        self.calls: list[dict[str, Any]] = []

    def feature_support(self, model: str) -> ProviderFeatures:
        return ProviderFeatures(model_supported=True, text=True, tools=False, attachments=False,
                                image_output=False, cost_ceiling_required=False, sdk_retries=0)

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        planning = self.plan is not None and system == PLANNING_SYSTEM_PROMPT
        answered = any(not call["planning"] for call in self.calls)
        if not planning and self.fail_first and not answered:
            self.calls.append({"planning": False, "failed": True, "tokens": 0})
            raise FixtureFault("planted transient failure of this attempt's first answer")
        text = json.dumps(self.plan) if planning else self.text
        result = CompletionResult(text, declared_tokens(system + prompt), declared_tokens(text))
        self.calls.append({"planning": planning, "failed": False, "tokens": result.total_tokens})
        return result


class ScriptedClassifier(Provider):
    """A deterministic routing classifier whose usage is declared.

    It reads only the goal in the router's prompt and names the tier of the
    first route whose phrase the goal contains, else ``default``. Its reply
    declares the prompt and reply sizes as tokens and their fixture price as
    ``cost_usd``, so the routing decision records the usage a harness charges.
    """

    def __init__(
        self,
        routes: Sequence[tuple[str, str]] = (("audit", "deterministic:pipeline"),),
        default: str = "autonomous",
    ) -> None:
        self.routes, self.default = tuple(routes), default
        self.calls: list[dict[str, Any]] = []

    def feature_support(self, model: str) -> ProviderFeatures:
        return ProviderFeatures(model_supported=True, text=True, tools=False, attachments=False,
                                image_output=False, cost_ceiling_required=False, sdk_retries=0)

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        goal = prompt.removeprefix("Goal: ").split("\n\nTask data (JSON", 1)[0].casefold()
        reply = next((tier for phrase, tier in self.routes if phrase in goal), self.default)
        prompt_tokens, completion_tokens = declared_tokens(system + prompt), declared_tokens(reply)
        cost = (prompt_tokens + completion_tokens) * MICROUSD_PER_TOKEN
        self.calls.append({"reply": reply, "tokens": prompt_tokens + completion_tokens,
                           "cost_microusd": cost})
        return CompletionResult(reply, prompt_tokens, completion_tokens,
                                cost_usd=float(Decimal(cost) / 1_000_000))


class _AttemptCheckpoints(CheckpointStore):
    """In-memory checkpoints for one attempt, so a raised attempt keeps its graph and timing."""

    def __init__(self) -> None:
        self._states: dict[str, str] = {}

    def save(self, execution_id: str, state: dict[str, Any]) -> None:
        self._states[execution_id] = json.dumps(state)

    def load(self, execution_id: str) -> dict[str, Any] | None:
        saved = self._states.get(execution_id)
        return None if saved is None else json.loads(saved)

    def delete(self, execution_id: str) -> None:
        self._states.pop(execution_id, None)

    def list_ids(self) -> list[str]:
        return sorted(self._states)

    def last(self) -> dict[str, Any] | None:
        ids = self.list_ids()
        return self.load(ids[-1]) if len(ids) == 1 else None


@dataclass(frozen=True)
class ArmTask:
    """One fixture task: what every arm is given and what the fixture model answers.

    ``deliverable`` is the fixture answer for the task and ``defect`` a
    planted defective answer that the family's acceptance policy must
    reject. ``planted`` names what the first attempt of every arm meets:
    None, ``"fault"`` (its first answer raises :class:`FixtureFault`) or
    ``"defect"`` (every step answers ``defect``). ``adjudicate`` checks a
    delivered output against an evaluator answer key, apart from
    acceptance, or is None when the family has no offline key.
    ``sha256`` identifies the task's source material.
    """

    family: str
    task_id: str
    task: Task = field(compare=False)
    deliverable: str
    defect: str
    sha256: str
    planted: str | None = None
    adjudicate: Callable[[str], dict[str, Any]] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.planted not in (None, "fault", "defect"):
            raise ValueError(f"Unknown planted failure {self.planted!r}")


def policy_record(policy: AcceptancePolicy) -> dict[str, Any]:
    """A strict policy as a record holds it: its canonical form and SHA-256."""
    return {**policy.to_dict(), "sha256": policy.sha256}


def _selection(planning: dict[str, Any] | None, timing: dict[str, Any] | None,
               classifier: ScriptedClassifier) -> dict[str, Any]:
    """What adaptive selection cost and how long it took, charged to the attempt.

    The decision and routing time come from the run's graph and timing. A
    rule decides without a classifier call and costs nothing; a classifier
    decision costs the usage the decision records. The charge is never less
    than the classifier calls the attempt made.
    """
    decision = (planning or {}).get("routing")
    called = sum(call["cost_microusd"] for call in classifier.calls)
    usage = None if decision is None else decision["usage"]
    if usage is None:
        recorded, source = 0, "no classifier call" if not classifier.calls else "classifier calls"
    elif usage["cost_usd"] is None or usage["cost_usd_unknown"]:
        recorded, source = 0, "classifier calls"
    else:
        recorded, source = microusd(usage["cost_usd"]), "declared classifier usage"
    phases = (timing or {}).get("phases_ms") or {}
    return {
        "method": None if decision is None else decision["method"],
        "tier": None if decision is None else decision["tier"],
        "reasons": [] if decision is None else decision["reasons"],
        "policy_version": None if decision is None else decision["policy_version"],
        "classifier_calls": len(classifier.calls),
        "cost_microusd": max(recorded, called),
        "cost_source": source,
        "latency_ms": phases.get("routing"),
    }


def _graph_shape(graph: dict[str, Any] | None) -> dict[str, Any] | None:
    if graph is None:
        return None
    nodes = graph.get("nodes", [])
    statuses = Counter(node.get("status") for node in nodes)
    return {"topology": " -> ".join(graph.get("topology", [])), "nodes": len(nodes),
            "node_statuses": dict(sorted(statuses.items()))}


@dataclass(frozen=True)
class AttemptRun:
    """What one executed attempt left behind, whatever its outcome.

    ``result`` is the run's ``SwarmResult``, or None when the run raised
    ``error``. A raised attempt's ``acceptance`` (its undecided record),
    ``graph`` and ``timing`` come from the checkpoint it saved as it failed.
    ``output`` is the delivered text, a rejected one included.
    """

    result: Any
    error: Exception | None
    acceptance: dict[str, Any] | None
    output: str | None
    graph: dict[str, Any] | None
    timing: dict[str, Any] | None
    elapsed_ms: float
    latency_source: str


def execute_attempt(build: Callable[[CheckpointStore], Any], task: Task) -> AttemptRun:
    """Run *task* once on the Swarm that *build* makes around in-memory checkpoints."""
    checkpoints = _AttemptCheckpoints()
    swarm = build(checkpoints)
    started = time.perf_counter()
    try:
        result, error = swarm.execute(snapshot_task(task)), None
    except Exception as exc:  # every failed attempt is recorded, not raised
        result, error = None, exc
    measured_ms = round((time.perf_counter() - started) * 1000, 1)

    if result is not None:
        acceptance = None if result.acceptance is None else result.acceptance.to_dict()
        output = result.output if result.rejected_output is None else result.rejected_output
        graph = graph_to_dict(result.graph)
        timing = result.timing
        elapsed, latency_source = timing["elapsed_ms"], "swarm_timing"
    else:
        # A raised attempt saved a failed checkpoint: its graph, timing and
        # the undecided acceptance record that binds the policy.
        state = checkpoints.last() or {}
        acceptance = state.get("delivery_acceptance")
        output = None
        graph = state.get("graph")
        segments = (state.get("timing") or {}).get("segments") or []
        timing = segments[-1] if segments else None
        elapsed = measured_ms if timing is None else timing["elapsed_ms"]
        latency_source = "harness_clock" if timing is None else "failed_checkpoint_timing"
    return AttemptRun(result, error, acceptance, output, graph, timing, elapsed, latency_source)


def attempt_record(
    arm_task: ArmTask,
    run: AttemptRun,
    model: Provider,
    *,
    attempt: int,
    planted: str | None,
    charges: dict[str, dict[str, Any] | None] | None = None,
) -> dict:
    """The record of one attempt: its outcome, acceptance, calls, cost and timing.

    *model* is the fixture model whose ``calls`` the attempt made. Each of
    *charges* names a further charge beside the model's, such as a selector's
    or a supervisor's: a dict whose ``cost_microusd`` joins the attempt's
    cost, or None when the arm has none.
    """
    acceptance, output, error, timing = run.acceptance, run.output, run.error, run.timing
    charges = dict(charges or {})
    decision = None if acceptance is None else acceptance["decision"]
    outcome = "error" if error is not None else decision
    tokens = sum(call["tokens"] for call in model.calls)
    cost = tokens * MICROUSD_PER_TOKEN + sum(
        charge["cost_microusd"] for charge in charges.values() if charge)
    record = {
        "attempt": attempt,
        "planted": planted,
        "outcome": outcome,
        "decision": decision,
        "reasons": [] if acceptance is None else list(acceptance.get("reasons", [])),
        "policy_sha256": None if acceptance is None else acceptance["policy_sha256"],
        "validators": [] if acceptance is None else [
            {**item["validator"], "outcome": item["outcome"]} for item in acceptance["validators"]],
        "output_sha256": None if output is None else hashlib.sha256(output.encode()).hexdigest(),
        "output_bytes": None if output is None else len(output.encode()),
        "adjudication": (arm_task.adjudicate(output)
                         if arm_task.adjudicate is not None and output is not None else None),
        "error": (None if error is None
                  else {"type": type(error).__name__, "message": str(error)[:300]}),
        "provider_calls": len(model.calls),
        "provider_calls_failed": sum(call["failed"] for call in model.calls),
        "planning_calls": sum(call["planning"] for call in model.calls),
        "tokens": tokens,
        "model_cost_microusd": tokens * MICROUSD_PER_TOKEN,
        **charges,
        "cost_microusd": cost,
        # The fixture providers declare every call's usage, so each charge is known.
        "cost_known": True,
        "cost_is_estimate": False,
        "elapsed_ms": run.elapsed_ms,
        "latency_source": run.latency_source,
        "phases_ms": None if timing is None else timing.get("phases_ms"),
        "graph": _graph_shape(run.graph),
    }
    return scrub_record(record)


def run_attempt(arm_task: ArmTask, arm: str, policy: AcceptancePolicy, *, attempt: int) -> dict:
    """Run one attempt of *arm* on *arm_task* and record it, whatever its outcome."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    planted = arm_task.planted if attempt == 1 else None
    model = FixtureModel(arm_task.defect if planted == "defect" else arm_task.deliverable,
                         plan=OFFLINE_DYNAMIC_PLAN, fail_first=planted == "fault")
    classifier = ScriptedClassifier()
    run = execute_attempt(
        lambda checkpoints: make_swarm(ARM_BASELINES[arm], model, FIXTURE_MODEL,
                                       classifier=classifier, classifier_model=CLASSIFIER_MODEL,
                                       acceptance=policy, checkpoint_store=checkpoints),
        arm_task.task)
    selector = (_selection((run.graph or {}).get("planning"), run.timing, classifier)
                if arm == ADAPTIVE_SELECTION else None)
    return attempt_record(arm_task, run, model, attempt=attempt, planted=planted,
                          charges={"selector": selector})


def run_cell(
    arm_task: ArmTask,
    arm: str,
    policy: AcceptancePolicy,
    *,
    allowance: int = 2,
    repetition: int = 1,
    attempt_runner: Callable[..., dict] = run_attempt,
) -> dict:
    """Run *arm* on *arm_task* until an attempt is accepted or *allowance* attempts are spent.

    The cell is accepted only when its last attempt is. Its cost and
    elapsed time cover every attempt, and so does its selector charge.
    *attempt_runner* runs and records one attempt, called as
    :func:`run_attempt` is; a harness with arms of its own passes its own.
    """
    if type(allowance) is not int or allowance < 1:
        raise ValueError("allowance must be a positive integer")
    attempts = []
    for number in range(1, allowance + 1):
        attempts.append(attempt_runner(arm_task, arm, policy, attempt=number))
        if attempts[-1]["outcome"] == "accepted":
            break
    cell = {
        "family": arm_task.family,
        "task_id": arm_task.task_id,
        "task_sha256": arm_task.sha256,
        "arm": arm,
        "repetition": repetition,
        "accepted": attempts[-1]["outcome"] == "accepted",
        "attempt_allowance": allowance,
        "failed_attempts": sum(item["outcome"] != "accepted" for item in attempts),
        "cost_microusd": sum(item["cost_microusd"] for item in attempts),
        "elapsed_ms": round(sum(item["elapsed_ms"] for item in attempts), 1),
        "attempts": attempts,
    }
    if arm == ADAPTIVE_SELECTION:
        latencies = [item["selector"]["latency_ms"] for item in attempts]
        cell["selector_cost_microusd"] = sum(item["selector"]["cost_microusd"] for item in attempts)
        cell["selector_latency_ms"] = (None if None in latencies
                                       else round(sum(latencies), 1))
    return cell


def policy_unchanged(cell: dict, policy: AcceptancePolicy) -> bool:
    """Whether every attempt in *cell* was decided by exactly *policy* and its validators."""
    identities = sorted((item.name, item.version, item.config_sha256)
                        for item in policy.validator_identities)
    for item in cell["attempts"]:
        if item["policy_sha256"] != policy.sha256:
            return False
        if item["validators"] and sorted(
                (v["name"], v["version"], v["config_sha256"]) for v in item["validators"]
        ) != identities:
            return False
    return True


def williams_schedule(
    label: str, task_ids: Sequence[str], arms: Sequence[str] = ARMS, *, seed: int,
    repetitions: int = 1,
) -> list[dict]:
    """Order task and repetition blocks, and counterbalance four arms inside each block.

    Arms and blocks are ordered by SHA-256 of a seeded key, and block *i*
    runs its arms in Williams row ``(i + offset) % 4``, so across every four
    blocks each arm takes each position once and follows each other arm once.
    """
    if len(arms) != 4 or len(set(arms)) != 4:
        raise ValueError("A Williams schedule needs four distinct arms")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")

    def key(value: object) -> str:
        payload = json.dumps([SCHEDULE_ALGORITHM, seed, label, value], sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    ordered = sorted(arms, key=key)
    blocks = sorted(((task_id, rep) for task_id in task_ids for rep in range(1, repetitions + 1)),
                    key=lambda pair: key(list(pair)))
    shift = int(key("row-offset")[:8], 16) % 4
    schedule = []
    for block, (task_id, rep) in enumerate(blocks):
        for position, index in enumerate(WILLIAMS_ROWS[(block + shift) % 4]):
            schedule.append({"ordinal": len(schedule), "block": block, "position": position,
                             "task_id": task_id, "repetition": rep, "arm": ordered[index]})
    return schedule


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear interpolation between closest ranks."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(cells: Sequence[dict], arms: Sequence[str] = ARMS) -> dict[str, dict]:
    """Per-arm totals: accepted rate, cost per accepted outcome, attempts, elapsed and selection.

    Every cell stays in its arm's denominator. With no accepted outcome the
    cost per accepted outcome is None (undefined), never a number.
    """
    summary = {}
    for arm in arms:
        own = [cell for cell in cells if cell["arm"] == arm]
        accepted = sum(cell["accepted"] for cell in own)
        cost = sum(cell["cost_microusd"] for cell in own)
        attempts = [item for cell in own for item in cell["attempts"]]
        elapsed = [cell["elapsed_ms"] for cell in own]
        entry = {
            "cells": len(own),
            "accepted": accepted,
            "accepted_rate": None if not own else round(accepted / len(own), 6),
            "attempts": len(attempts),
            "failed_attempts": sum(cell["failed_attempts"] for cell in own),
            "false_acceptances": sum(
                item["outcome"] == "accepted" and item["adjudication"] is not None
                and not item["adjudication"]["deterministic_passed"] for item in attempts),
            "cost_microusd": cost,
            "cost_per_accepted_microusd": None if not accepted else round(cost / accepted, 3),
            "elapsed_ms": None if not elapsed else {
                "runs": len(elapsed), "p50": round(percentile(elapsed, 0.5), 1),
                "p95": round(percentile(elapsed, 0.95), 1),
                "label": ("small sample; not a tail estimate"
                          if len(elapsed) < SMALL_SAMPLE_RUNS else None)},
        }
        if arm == ADAPTIVE_SELECTION:
            selections = [item["selector"] for item in attempts]
            latencies = [item["latency_ms"] for item in selections
                         if item["latency_ms"] is not None]
            entry["selection"] = {
                "cost_microusd": sum(item["cost_microusd"] for item in selections),
                "latency_ms_total": round(sum(latencies), 1),
                "latency_measured": len(latencies),
                "methods": dict(sorted(Counter(str(item["method"]) for item in selections).items())),
                "tiers": dict(sorted(Counter(str(item["tier"]) for item in selections).items())),
            }
        summary[arm] = entry
    return summary


def offline_record(benchmark: str, **body: Any) -> dict:
    """A record labeled as offline mechanics: never claimable, with no API calls or spend."""
    return scrub_record({
        "benchmark": benchmark,
        "evidence_status": "offline-mechanics",
        "claimable": False,
        "api_calls": 0,
        "paid_execution_allowed": False,
        "mode": "offline",
        **body,
        "environment": environment_snapshot("smythe"),
    })
