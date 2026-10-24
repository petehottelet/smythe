"""Offline policy ablations: drain, capacity admission, budget holds and contradiction synthesis.

    python benchmarks/run_policy_ablations.py
    python benchmarks/run_policy_ablations.py --overhead-repetitions 1 --out PATH

Each ablation runs a baseline arm and a policy arm, built from one
configuration that differs only in the policy under test, on the same
fixtures:

    drain_ordinary           AsyncExecutor without and with
                             TerminationPolicy(drain_timeout_s=30), on a
                             manual clock
    drain_durable            a durable Swarm without and with the same
                             policy, on the shared fake Responses transport
    capacity_admission       AsyncExecutor without and with a CapacityPolicy
                             limit at the fixture provider's quota, on a
                             manual clock; then the provider-concurrency
                             simulator at the concurrency each arm admits
    budget_holds             Swarm(reserve_for_checks=False) and True, one
                             node in flight at a time
    contradiction_synthesis  Synthesizer() and
                             Synthesizer(retain_contradictions=True) under
                             one strict source-evidence acceptance policy

Every arm's record keeps each attempt, the calls that failed, were
cancelled, refused or rejected included, with its error and failure class,
and groups the failures that share one cause as correlated errors: a
sibling cancelled because another node failed, or the same failure on
several attempts or nodes.

A matched-graph scheduler overhead test runs the fixed research -> analyze
-> write pipeline under an ordinary Swarm, a durable Swarm and a minimal
reference asyncio executor, on the same fake transport with the same
scripted replies and 0 ms or a declared constant of latency per wire call.
Their outputs must be byte-identical. Overhead is each executor's elapsed
time minus the reference's, per run and per node, with the time spent in
checkpoint and journal stores reported apart. The durable runtime also
counts each node's input tokens on the wire, and those calls wait the
declared latency too: its overhead includes that wait, which the record
lists as extra_wire_latency_ms. It is real local time on one machine:
machine-specific, never a product claim.

Fixture runs establish mechanics, never a measured benefit: the providers
are deterministic fixtures, manual-clock times are declared, and the record
is labeled offline-mechanics, claimable false, api_calls 0. Every policy
stays opt-in and experimental; no default changes.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import hashlib
import heapq
import itertools
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# The shared deterministic fakes (wire transports, manual clock, gates) live with the tests.
sys.path.append(str(REPO_ROOT / "tests"))

from benchmarks.astra_campaign._json import canonical, digest  # noqa: E402
from benchmarks.harness import FixedPipelineArchitect  # noqa: E402
from benchmarks.offline_arms import (  # noqa: E402
    FIXTURE_MODEL, ArmTask, FixtureModel, attempt_record, execute_attempt, offline_record,
    policy_record, run_cell, summarize,
)
from fakes import RATE_LIMITED, Gate, HttpError, ManualClock, NativeTransport  # noqa: E402

from smythe import (  # noqa: E402
    AcceptancePolicy, CapacityLimit, CapacityPolicy, ClaimPassage, ClaimRecord, ContextSource,
    RetryPolicy, SecurityStop, Sentinel, SentinelAlert, SimpleArchitect, SourceEvidence,
    SourceEvidenceValidator, SourceRecord, SQLiteWorkflowStore, Swarm, SynthesisStrategy,
    Synthesizer, Task, TerminationPolicy,
)
from smythe._failures import classify_failure  # noqa: E402
from smythe.async_executor import AsyncExecutor  # noqa: E402
from smythe.checkpoint import FileCheckpointStore  # noqa: E402
from smythe.evidence import read_source_evidence, render_source_evidence  # noqa: E402
from smythe.graph import (  # noqa: E402
    SYNTHESIS_NODE_ID, ExecutionGraph, FailurePolicy, Node, NodeStatus, Topology,
)
from smythe.optimize import ConcurrencyScenario, simulate_concurrency  # noqa: E402
from smythe.planner import DeterministicArchitect  # noqa: E402
from smythe.provider import CompletionResult, Provider, ProviderFeatures  # noqa: E402
from smythe.provider_responses import OpenAIResponsesProvider  # noqa: E402
from smythe.registry import Registry  # noqa: E402
from smythe.synthesizer import MERGE_SYSTEM_PROMPT  # noqa: E402
from smythe.tracer import Tracer  # noqa: E402
from smythe.workflow_store import WorkflowStateError  # noqa: E402

DEFAULT_OUT = Path(__file__).with_name("results") / "policy_ablations_offline.json"
RECORD_VERSION = 1
ARMS = ("baseline", "policy")
NS_PER_S = 1_000_000_000
SCOPE = (
    "Fixture runs establish mechanics, not a measured benefit. Deterministic fixture providers "
    "decide every call, manual-clock times are declared, and the failures are planted where each "
    "policy acts, so the differences between arms follow from the fixtures and the policy under "
    "test. Scheduler overhead is real local time on one machine. No policy becomes a default, "
    "and no number here is a product, README or roadmap claim."
)


# ---------------------------------------------------------------------------
# One change per ablation
# ---------------------------------------------------------------------------


def one_change(baseline: dict, policy: dict) -> list[str]:
    """The configuration keys on which two arms differ, sorted."""
    missing = object()
    return sorted(key for key in baseline.keys() | policy.keys()
                  if baseline.get(key, missing) != policy.get(key, missing))


def ablation_arms(base: dict, varied: str, off: Any, on: Any) -> dict[str, dict]:
    """The baseline and policy arms: *base* with *varied* set to *off*, then to *on*."""
    if varied in base:
        raise ValueError(f"{varied!r} is the varied policy; the base configuration cannot set it")
    arms = {"baseline": {**base, varied: off}, "policy": {**base, varied: on}}
    if one_change(*arms.values()) != [varied]:
        raise ValueError(f"The arms must differ in {varied!r} alone")
    return arms


def _error(exc: BaseException | None) -> dict | None:
    return None if exc is None else {"type": type(exc).__name__, "message": str(exc)[:300]}


def correlated_errors(failures: list[dict]) -> list[dict]:
    """Group the failures that share one cause.

    A failure that another caused names that node in ``caused_by``, and
    the failures one node caused form one group. Failures that name no
    cause share one when they share a signature: the same outcome, error
    type, failure class and message. A signature seen once is not a group.
    """
    groups: dict[tuple, dict] = {}
    for failure in failures:
        if failure.get("caused_by") is not None:
            key: tuple = ("caused_by", failure["caused_by"])
            group = groups.setdefault(key, {"kind": "consequence",
                                            "cause": f"the failure of {failure['caused_by']!r}",
                                            "failures": []})
        else:
            error = failure.get("error") or {}
            signature = {"outcome": failure["outcome"], "error_type": error.get("type"),
                         "failure_class": failure.get("failure_class"),
                         "message": error.get("message")}
            key = ("signature", json.dumps(signature, sort_keys=True))
            group = groups.setdefault(key, {"kind": "shared_signature", "signature": signature,
                                            "failures": []})
        group["failures"].append({name: failure.get(name) for name in ("node", "attempt")})
    result = []
    for group in groups.values():
        if group["kind"] == "shared_signature" and len(group["failures"]) < 2:
            continue
        group["count"] = len(group["failures"])
        result.append(group)
    return result


def ablation_case(case: str, fixture: str, arms: dict[str, dict], results: dict[str, dict],
                  measures: dict[str, Callable[[dict], Any]], **extra: Any) -> dict:
    """One case's record: its fixture, both arms, their results and each measure side by side.

    ``differs_in`` lists the configuration keys on which the arms differ.
    """
    return {"case": case, "fixture": fixture, **extra,
            "differs_in": one_change(arms["baseline"], arms["policy"]), "arms": arms,
            "results": results,
            "comparison": {name: {arm: read(result) for arm, result in results.items()}
                           for name, read in measures.items()}}


# ---------------------------------------------------------------------------
# Virtual time
# ---------------------------------------------------------------------------


class _InlineThreads(concurrent.futures.ThreadPoolExecutor):
    """Runs what the event loop hands to a worker thread at once, on the loop's own thread.

    Under virtual time every step must finish before the clock moves; work
    still running on a thread when it did would make the order of events
    depend on real timing.
    """

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # The caller's future carries any failure.
            future.set_exception(exc)
        return future


class VirtualTime:
    """Discrete-event time on a ``ManualClock`` for one asyncio run.

    ``sleep`` waits until the clock reaches its wake time. ``run`` drives a
    coroutine: it lets every task go as far as it can, then moves the clock
    to the earliest wake time and wakes those sleepers, until the coroutine
    ends. Fixture calls and the executor's own waits (retry backoff, the
    drain deadline) take exactly their declared time, and no real time
    passes, so every reading is exact and every run repeats.
    """

    SETTLE_PASSES = 64

    def __init__(self) -> None:
        self.clock = ManualClock()
        self._sleepers: list[tuple[int, int, asyncio.Future]] = []
        self._order = itertools.count()

    def tracer(self) -> Tracer:
        return Tracer(clock=self.clock.monotonic,
                      wall_clock=lambda: self.clock.time_ns() / NS_PER_S)

    def now_ms(self) -> int:
        return self.clock.monotonic_ns() // 1_000_000

    async def sleep(self, seconds: float) -> None:
        wake = self.clock.monotonic_ns() + round(seconds * NS_PER_S)
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (wake, next(self._order), future))
        await future

    def run(self, coroutine) -> Any:
        return asyncio.run(self._drive(coroutine))

    async def _drive(self, coroutine) -> Any:
        asyncio.get_running_loop().set_default_executor(_InlineThreads(max_workers=1))
        task = asyncio.ensure_future(coroutine)
        while True:
            for _ in range(self.SETTLE_PASSES):
                await asyncio.sleep(0)
            if task.done():
                return task.result()
            while self._sleepers and self._sleepers[0][2].done():
                heapq.heappop(self._sleepers)
            if not self._sleepers:
                task.cancel()
                raise RuntimeError("The run waits on something no virtual timer ends")
            wake = self._sleepers[0][0]
            self.clock.advance((wake - self.clock.monotonic_ns()) / NS_PER_S)
            while self._sleepers and self._sleepers[0][0] <= wake:
                future = heapq.heappop(self._sleepers)[2]
                if not future.done():
                    future.set_result(None)


def _no_sdk_retries(model: str) -> ProviderFeatures:
    return ProviderFeatures(sdk_retries=0)


# ---------------------------------------------------------------------------
# Drain on ordinary runs
# ---------------------------------------------------------------------------

DRAIN_TIMEOUT_S = 30.0
NODE_ESTIMATE_USD = 0.25
CALL_COST_USD = 0.125


class TimedSteps(Provider):
    """A fixture provider whose calls take declared virtual time, then answer or raise.

    *steps* maps a node's label to its seconds and an error factory, or
    None to answer; any other node answers after one second. Each call is
    recorded with its start, its end and its outcome: answered, failed or
    cancelled. An answered call costs ``CALL_COST_USD``. The provider
    declares no SDK retries.
    """

    def __init__(self, time: VirtualTime,
                 steps: dict[str, tuple[float, Callable[[], Exception] | None]]) -> None:
        self.time, self.steps = time, steps
        self.calls: list[dict[str, Any]] = []

    feature_support = staticmethod(_no_sdk_retries)

    def budget_estimate_usd(self, model: str) -> None:
        return None

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        label = prompt.splitlines()[0]
        seconds, fault = self.steps.get(label, (1.0, None))
        call: dict[str, Any] = {"node": label, "start_ms": self.time.now_ms()}
        self.calls.append(call)
        try:
            await self.time.sleep(seconds)
        except asyncio.CancelledError:
            call.update(end_ms=self.time.now_ms(), outcome="cancelled")
            raise
        call["end_ms"] = self.time.now_ms()
        if fault is not None:
            error = fault()
            call.update(outcome="failed", error=_error(error))
            raise error
        call["outcome"] = "answered"
        return CompletionResult(f"{label} done", cost_usd=CALL_COST_USD)


def _planted_failure() -> Exception:
    return RuntimeError("planted failure")


def _planted_security_stop() -> Exception:
    return SecurityStop("fixture-revoked")


DRAIN_CASES = {
    "answers_within_drain": (
        "fail raises a planted failure at 1 s while sibling, in flight, answers at 3 s; queued "
        "waits for a slot",
        {"fail": (1.0, _planted_failure), "sibling": (3.0, None)}),
    "straggles_past_deadline": (
        "fail raises a planted failure at 1 s; sibling would answer only at 60 s, after the "
        "30 s drain deadline",
        {"fail": (1.0, _planted_failure), "sibling": (60.0, None)}),
    "security_stop": (
        "fail raises SecurityStop at 1 s while sibling, in flight, would answer at 3 s",
        {"fail": (1.0, _planted_security_stop), "sibling": (3.0, None)}),
}
DRAIN_NODES = ("fail", "sibling", "queued")


def _stopped_by(calls: list[dict], statuses: dict[str, str]) -> str | None:
    """The failed node whose failure stopped the run: its last call ended first.

    Of nodes whose last calls ended together, the first in graph order, as
    the executor takes them.
    """
    order = {node_id: index for index, node_id in enumerate(statuses)}
    last = {call["node"]: call for call in calls}
    ended = [(call["end_ms"], order[node_id]) for node_id, call in last.items()
             if statuses[node_id] == NodeStatus.FAILED.value and call["outcome"] == "failed"]
    return None if not ended else list(statuses)[min(ended)[1]]


def _attempt_failures(calls: list[dict], tracer: Tracer, statuses: dict[str, str]) -> list[dict]:
    """Each failed or cancelled call, with its attempt, error, failure class and cause.

    The class is the one the tracer recorded on the attempt's span. A call
    cancelled in flight was cancelled because the run stopped, so the node
    whose failure stopped it caused the cancellation.
    """
    classes = {(span.node_id, span.attempt): span.metadata["failure_class"]
               for span in tracer.spans if "failure_class" in span.metadata}
    cause = _stopped_by(calls, statuses)
    attempts: Counter[str] = Counter()
    failures = []
    for call in calls:
        node = call["node"]
        attempts[node] += 1
        if call["outcome"] == "failed":
            failures.append({"node": node, "attempt": attempts[node], "outcome": "failed",
                             "error": call.get("error"),
                             "failure_class": classes.get((node, attempts[node] - 1)),
                             "caused_by": None})
        elif call["outcome"] == "cancelled":
            failures.append({"node": node, "attempt": attempts[node], "outcome": "cancelled",
                             "error": None, "failure_class": None, "caused_by": cause})
    return failures


def run_drain_ordinary_arm(case: str, config: dict) -> dict:
    """One arm of a drain case: the fork graph on the async executor, under virtual time."""
    _, steps = DRAIN_CASES[case]
    time_ = VirtualTime()
    provider = TimedSteps(time_, steps)
    nodes = [Node(id=node_id, label=node_id,
                  metadata={"estimated_cost_usd": config["node_estimate_usd"]})
             for node_id in DRAIN_NODES]
    budget, tracer = Sentinel(config["max_budget_usd"]), time_.tracer()
    termination = (None if config["termination"] is None
                   else TerminationPolicy(**config["termination"]))
    executor = AsyncExecutor(provider=provider, registry=Registry(), tracer=tracer,
                             budget=budget, artifact_dir=None,
                             max_concurrency=config["max_concurrency"], termination=termination,
                             sleep=time_.sleep)
    try:
        time_.run(executor.run(ExecutionGraph([Topology.FORK_JOIN], nodes)))
        error = None
    except Exception as exc:  # The planted failure stops the run; it is recorded.
        error = exc
    kept = budget.breakdown()
    statuses = {node.id: node.status.value for node in nodes}
    failures = _attempt_failures(provider.calls, tracer, statuses)
    return {
        "error": _error(error),
        "elapsed_ms": time_.now_ms(),
        "calls": provider.calls,
        "nodes": statuses,
        "drain": executor.drain_record,
        "kept_cost_usd": kept,
        "reserved_usd": budget.total_cost_usd - sum(kept.values()),
        "cost_is_complete": budget.cost_is_complete,
        "waits_ms": tracer.timing()["waiting_ms"],
        "failures": failures,
        "correlated_errors": correlated_errors(failures),
        "resume": {
            "nodes_to_run": sorted(node_id for node_id, status in statuses.items()
                                   if status != NodeStatus.COMPLETED.value),
            # A cancelled call was sent, and its node runs it again on resume.
            "sent_calls_repeated": sum(call["outcome"] == "cancelled" for call in provider.calls),
        },
    }


DRAIN_MEASURES = {
    "completed": lambda result: sorted(node for node, status in result["nodes"].items()
                                       if status == NodeStatus.COMPLETED.value),
    "kept_cost_usd": lambda result: sum(result["kept_cost_usd"].values()),
    "cost_is_complete": lambda result: result["cost_is_complete"],
    "sent_calls_repeated": lambda result: result["resume"]["sent_calls_repeated"],
    "drain": lambda result: result["drain"],
    "drain_wait_ms": lambda result: result["waits_ms"]["drain"],
    "elapsed_ms": lambda result: result["elapsed_ms"],
}


def drain_ordinary() -> dict:
    arms = ablation_arms(
        {"executor": "AsyncExecutor", "graph": "fork_join: " + ", ".join(DRAIN_NODES),
         "max_concurrency": 2, "max_budget_usd": 10.0, "node_estimate_usd": NODE_ESTIMATE_USD,
         "call_cost_usd": CALL_COST_USD, "failure_policy": "halt", "clock": "manual"},
        "termination", None, {"drain_timeout_s": DRAIN_TIMEOUT_S})
    cases = []
    for case, (fixture, _) in DRAIN_CASES.items():
        results = {arm: run_drain_ordinary_arm(case, config) for arm, config in arms.items()}
        cases.append(ablation_case(case, fixture, arms, results, DRAIN_MEASURES))
    return {
        "policy": "TerminationPolicy",
        "varied": "termination",
        "question": "When a failure stops an ordinary run, does draining keep the work already in "
                    "flight that a cancellation would lose, without delaying a security stop?",
        "clock": "manual: each fixture call and the drain deadline take declared virtual time",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Drain on durable runs
# ---------------------------------------------------------------------------

DURABLE_MODEL = "gpt-6-astra"
UNAUTHORIZED = HttpError(401, "invalid_api_key", "Incorrect API key provided")
DURABLE_CASES = {
    "answered_within_drain": (
        "Halt's request is refused with HTTP 401 once Sibling's request is sent and held; "
        "Sibling answers as soon as the run waits for it"),
    "deadline_passes": (
        "Halt's request is refused with HTTP 401 once Sibling's request is sent and held; "
        "the drain deadline passes before Sibling answers"),
}


class _Patches:
    """Attribute patches that are undone in reverse order.

    ``setattr`` matches the object that a fake transport's ``install()``
    expects, as pytest's ``monkeypatch`` does.
    """

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        while self._undo:
            target, name, value = self._undo.pop()
            setattr(target, name, value)


@contextmanager
def patched() -> Iterator[_Patches]:
    patches = _Patches()
    try:
        yield patches
    finally:
        patches.undo()


class HeldSibling(NativeTransport):
    """The shared Responses wire fake that holds Sibling and refuses Halt once.

    With *held*, Sibling's generation request reaches the provider and waits
    at the gate, and Halt's first generation request waits until Sibling has
    reached it, then is refused with HTTP 401, a zero-cost rejection.
    Without a gate every call is answered.
    """

    def __init__(self, held: Gate | None = None) -> None:
        super().__init__()
        self.held, self.refused = held, held is None

    async def _call(self, operation, payload):
        prompt = self._prompt(payload)
        if self.held is not None and operation == "response" and prompt.startswith("Sibling"):
            self.schedule("execution", self._ordinals[operation, "execution"], self.held)
        elif operation == "response" and prompt.startswith("Halt") and not self.refused:
            self.refused = True
            if not await self.held.reached():
                raise RuntimeError("Sibling's request never reached the provider")
            self.schedule("execution", self._ordinals[operation, "execution"], UNAUTHORIZED)
        return await super()._call(operation, payload)

    def sent(self, label: str) -> int:
        return sum(self._prompt(payload).startswith(label) for payload in self.requests)


def durable_swarm(store: SQLiteWorkflowStore, config: dict) -> Swarm:
    termination = (None if config["termination"] is None
                   else TerminationPolicy(**config["termination"]))
    provider = OpenAIResponsesProvider(api_key="offline-fixture",
                                       max_output_tokens=config["max_output_tokens"])
    return Swarm(model=DURABLE_MODEL, provider=provider, run_store=store, parallel=True,
                 max_concurrency=config["max_concurrency"],
                 max_budget_usd=config["max_budget_usd"], termination=termination)


def _durable_graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.FORK_JOIN], [Node(id="halt", label="Halt"),
                                                 Node(id="sibling", label="Sibling")])


def journal_summary(inspected: dict) -> dict:
    """What a durable run's journal holds: its state, its totals and each node's calls."""
    calls: dict[str, list[dict]] = {}
    for call in inspected["calls"]:
        key = call["key"]
        if key["phase"] != "execution":
            continue
        calls.setdefault(key["scope_id"].removeprefix("node/"), []).append(
            {name: call.get(name) for name in ("state", "billing_state", "result_state",
                                               "cost_nanousd", "reserved_nanousd",
                                               "unknown_nanousd")})
    return {
        "status": inspected["status"],
        "blocked_reason": inspected["blocked_reason"],
        "ok": inspected["ok"],
        "confirmed_nanousd": inspected["confirmed_nanousd"],
        "reserved_nanousd": inspected["reserved_nanousd"],
        "unknown_nanousd": inspected["unknown_nanousd"],
        "unknown_calls": inspected["unknown_calls"],
        "calls": dict(sorted(calls.items())),
    }


def _durable_failures(journal: dict, error: Exception | None) -> list[dict]:
    """The rejected call, with the run's error, and each call the stop left unknown.

    A sent call left unknown was cut off because the rejected call's node
    stopped the run, so that node caused it.
    """
    rejected = [(node, attempt) for node, calls in journal["calls"].items()
                for attempt, call in enumerate(calls, start=1)
                if call["result_state"] == "rejected"]
    failures = [{"node": node, "attempt": attempt, "outcome": "rejected", "error": _error(error),
                 "failure_class": None if error is None else classify_failure(error),
                 "caused_by": None}
                for node, attempt in rejected]
    cause = rejected[0][0] if rejected else None
    failures += [{"node": node, "attempt": attempt, "outcome": "unknown_exposure", "error": None,
                  "failure_class": None, "caused_by": cause}
                 for node, calls in journal["calls"].items()
                 for attempt, call in enumerate(calls, start=1) if call["state"] == "unknown"]
    return failures


def run_drain_durable_arm(case: str, config: dict) -> dict:
    """One arm of a durable drain case: run, inspect the journal, then try to resume."""
    held, waits = Gate(), []
    original = AsyncExecutor._drain_wait

    async def drain_wait(executor, seconds):
        # Stands for the drain's deadline: the held sibling answers as soon as
        # the run waits for it, or the deadline passes at once.
        waits.append(seconds)
        if case == "answered_within_drain":
            held.release()
            await original(executor, seconds)

    with tempfile.TemporaryDirectory(prefix="smythe-drain-", ignore_cleanup_errors=True) as work, \
            patched() as patches:
        path = Path(work) / "runs.db"
        wire = HeldSibling(held).install(patches)
        patches.setattr(AsyncExecutor, "_drain_wait", drain_wait)
        try:
            with SQLiteWorkflowStore(path) as store:
                try:
                    durable_swarm(store, config).execute(_durable_graph())
                    error = None
                except Exception as exc:  # Halt's rejection stops the run; it is recorded.
                    error = exc
                (run,) = store.list_runs()
                journal = journal_summary(store.inspect_run(run["run_id"]))
        finally:
            held.release()
        resumed = HeldSibling().install(patches)
        with SQLiteWorkflowStore(path) as store:
            try:
                result = durable_swarm(store, config).resume(run["run_id"])
                resume = {"outcome": "completed", "error": None,
                          "cost_is_complete": result.cost_is_complete}
            except WorkflowStateError as exc:
                resume = {"outcome": "refused", "error": _error(exc), "cost_is_complete": None}
            resume["sent"] = {label: resumed.sent(label) for label in ("Halt", "Sibling")}
            resume["journal"] = journal_summary(store.inspect_run(run["run_id"]))
    failures = _durable_failures(journal, error)
    return {
        "error": _error(error),
        "sent": {label: wire.sent(label) for label in ("Halt", "Sibling")},
        "drain_waits_s": waits,
        "journal": journal,
        "failures": failures,
        "correlated_errors": correlated_errors(failures),
        "resume": resume,
    }


DURABLE_MEASURES = {
    "sibling_calls": lambda result: result["journal"]["calls"].get("sibling"),
    "unknown_nanousd": lambda result: result["journal"]["unknown_nanousd"],
    "blocked_reason": lambda result: result["journal"]["blocked_reason"],
    "resume": lambda result: result["resume"]["outcome"],
    "sent_on_resume": lambda result: result["resume"]["sent"],
    "drain_waits_s": lambda result: result["drain_waits_s"],
}


def drain_durable() -> dict:
    arms = ablation_arms(
        {"swarm": "durable", "run_store": "SQLiteWorkflowStore",
         "graph": "fork_join: halt, sibling",
         "provider": "OpenAIResponsesProvider on the shared fake Responses transport",
         "model": DURABLE_MODEL, "max_output_tokens": 100, "max_concurrency": 2,
         "max_budget_usd": 1.0, "failure_policy": "halt", "clock": "real"},
        "termination", None, {"drain_timeout_s": DRAIN_TIMEOUT_S})
    cases = []
    for case, fixture in DURABLE_CASES.items():
        results = {arm: run_drain_durable_arm(case, config) for arm, config in arms.items()}
        cases.append(ablation_case(case, fixture, arms, results, DURABLE_MEASURES))
    return {
        "policy": "TerminationPolicy",
        "varied": "termination",
        "question": "When a failure stops a durable run, does draining settle a sibling's sent "
                    "call instead of leaving permanent unknown exposure that blocks resume?",
        "clock": "real; a fixture hook stands for the drain deadline, so the held sibling "
                 "answers as soon as the run waits for it, or the deadline passes at once",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Capacity admission
# ---------------------------------------------------------------------------

QUOTA_IN_FLIGHT = 2
CAPACITY_MODEL = "m"
CAPACITY_NODES = 8
CALL_S = 1.0
REJECTION_S = 0.25


class QuotaProvider(Provider):
    """A fixture provider that serves at most ``quota`` calls at once.

    A call that arrives while ``quota`` calls are in flight is refused with
    HTTP 429 (no ``Retry-After``) after ``REJECTION_S``; any other answers
    after ``CALL_S``. Calls, outcomes and the most calls in flight are
    recorded. It declares no SDK retries.
    """

    def __init__(self, time: VirtualTime, quota: int) -> None:
        self.time, self.quota = time, quota
        self.calls: list[dict[str, Any]] = []
        self.in_flight = self.peak = 0

    feature_support = staticmethod(_no_sdk_retries)

    def budget_estimate_usd(self, model: str) -> None:
        return None

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        label = prompt.splitlines()[0]
        over = self.in_flight >= self.quota
        call: dict[str, Any] = {"node": label, "start_ms": self.time.now_ms()}
        self.calls.append(call)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self.time.sleep(REJECTION_S if over else CALL_S)
        except asyncio.CancelledError:
            call["outcome"] = "cancelled"
            raise
        finally:
            self.in_flight -= 1
            call["end_ms"] = self.time.now_ms()
        if over:
            error = RATE_LIMITED.exception()
            call.update(outcome="failed", error=_error(error))
            raise error
        call["outcome"] = "answered"
        return CompletionResult(f"{label} done", cost_usd=CALL_COST_USD)


class _UpperJitter:
    """Draws the top of every jitter range, so each retry backoff equals its ceiling."""

    def uniform(self, low: float, high: float) -> float:
        return high


def _capacity_policy(config: dict | None) -> CapacityPolicy | None:
    if config is None:
        return None
    return CapacityPolicy(limits={model: CapacityLimit(**limit)
                                  for model, limit in config["limits"].items()},
                          round_robin=config["round_robin"],
                          cooldown_max_s=config["cooldown_max_s"])


def run_capacity_arm(config: dict) -> dict:
    """One arm of a capacity case: a wide fan-out against a provider quota, under virtual time."""
    time_ = VirtualTime()
    provider = QuotaProvider(time_, config["provider_quota_in_flight"])
    nodes = [Node(id=f"w{index}", label=f"w{index}",
                  metadata={"model": CAPACITY_MODEL, "estimated_cost_usd": NODE_ESTIMATE_USD},
                  failure_policy=FailurePolicy.RETRY, max_retries=config["max_retries"])
             for index in range(config["nodes"])]
    tracer = time_.tracer()
    executor = AsyncExecutor(
        provider=provider, registry=Registry(), tracer=tracer,
        budget=Sentinel(config["max_budget_usd"]), artifact_dir=None,
        max_concurrency=config["max_concurrency"],
        capacity_policy=_capacity_policy(config["capacity_policy"]), sleep=time_.sleep,
        retry_backoff_s=config["retry_backoff_s"], rng=_UpperJitter(),
        retry_policy=RetryPolicy(**config["retry_policy"]))
    try:
        time_.run(executor.run(ExecutionGraph([Topology.FORK_JOIN], nodes)))
        error = None
    except Exception as exc:  # A node out of retries stops the run; it is recorded.
        error = exc
    statuses = {node.id: node.status.value for node in nodes}
    failures = _attempt_failures(provider.calls, tracer, statuses)
    return {
        "error": _error(error),
        "elapsed_ms": time_.now_ms(),
        "calls": len(provider.calls),
        "answered": sum(call["outcome"] == "answered" for call in provider.calls),
        "rejected_429": sum(call["outcome"] == "failed" for call in provider.calls),
        "cancelled": sum(call["outcome"] == "cancelled" for call in provider.calls),
        "peak_in_flight": provider.peak,
        "nodes": dict(sorted(Counter(statuses.values()).items())),
        "waits_ms": tracer.timing()["waiting_ms"],
        "call_starts": [[call["node"], call["start_ms"], call["outcome"]]
                        for call in provider.calls],
        "failures": failures,
        "correlated_errors": correlated_errors(failures),
    }


CAPACITY_CASES = {
    "retries_cover_overload": (
        "eight ready nodes, eight free slots and a provider that serves two calls at once; "
        "each node may retry four times", 4),
    "retries_run_out": (
        "the same fan-out; each node may retry once", 1),
}
SIMULATED_SEEDS = 5
SIMULATED_MAX_CONCURRENCY = 16


def _simulated_concurrency(config: dict) -> int:
    limit = config["capacity_policy"]
    if limit is None:
        return config["max_concurrency"]
    return min(config["max_concurrency"], limit["limits"][CAPACITY_MODEL]["max_in_flight"])


def run_simulated_arm(config: dict, *, seed: int) -> dict:
    """The provider-concurrency simulator at the concurrency an arm admits, over paired seeds."""
    scenario = ConcurrencyScenario(**config["scenario"])
    concurrency = _simulated_concurrency(config)
    trials = [{"seed": trial_seed,
               **{name: round(value, 3) for name, value in simulate_concurrency(
                   {"max_concurrency": concurrency}, scenario=scenario, seed=trial_seed,
                   split="development").items()}}
              for trial_seed in range(seed, seed + SIMULATED_SEEDS)]
    names = [name for name in trials[0] if name != "seed"]
    return {"admitted_concurrency": concurrency, "evaluator_hash": scenario.evaluator_hash,
            "trials": trials,
            "mean": {name: round(statistics.fmean(trial[name] for trial in trials), 3)
                     for name in names}}


CAPACITY_MEASURES = {
    "nodes": lambda result: result["nodes"],
    "rejected_429": lambda result: result["rejected_429"],
    "calls": lambda result: result["calls"],
    "peak_in_flight": lambda result: result["peak_in_flight"],
    "retry_backoff_ms": lambda result: result["waits_ms"]["retry_backoff"],
    "capacity_wait_ms": lambda result: result["waits_ms"]["capacity"],
    "elapsed_ms": lambda result: result["elapsed_ms"],
}


def capacity_admission(*, seed: int) -> dict:
    limit = {"limits": {CAPACITY_MODEL: {"max_in_flight": QUOTA_IN_FLIGHT,
                                         "min_interval_s": None}},
             "round_robin": False, "cooldown_max_s": 60.0}
    cases = []
    for case, (fixture, retries) in CAPACITY_CASES.items():
        arms = ablation_arms(
            {"executor": "AsyncExecutor", "graph": f"fork_join: {CAPACITY_NODES} nodes on model "
             f"{CAPACITY_MODEL!r}", "nodes": CAPACITY_NODES, "max_concurrency": CAPACITY_NODES,
             "provider_quota_in_flight": QUOTA_IN_FLIGHT, "call_s": CALL_S,
             "rejection_s": REJECTION_S, "failure_policy": "retry", "max_retries": retries,
             "retry_policy": {"backoff_max_s": 8.0}, "retry_backoff_s": 1.0,
             "jitter": "top of each range", "max_budget_usd": 10.0, "clock": "manual"},
            "capacity_policy", None, limit)
        results = {arm: run_capacity_arm(config) for arm, config in arms.items()}
        cases.append(ablation_case(case, fixture, arms, results, CAPACITY_MEASURES))
    simulated_limit = {"limits": {CAPACITY_MODEL: {"max_in_flight": 8, "min_interval_s": None}},
                       "round_robin": False, "cooldown_max_s": 60.0}
    arms = ablation_arms(
        {"model": "smythe.optimize.concurrency simulator",
         "scenario": {"work_items": 250, "provider_capacity": 8},
         "max_concurrency": SIMULATED_MAX_CONCURRENCY, "split": "development",
         "seeds": [seed, seed + SIMULATED_SEEDS - 1]},
        "capacity_policy", None, simulated_limit)
    cases.append(ablation_case(
        "simulated_provider",
        "250 work items against a simulated provider with capacity 8, congestion latency and "
        "overload failures; each arm runs at the concurrency it admits, on the same seeds",
        arms, {arm: run_simulated_arm(config, seed=seed) for arm, config in arms.items()},
        {"admitted_concurrency": lambda result: result["admitted_concurrency"],
         "mean": lambda result: result["mean"]}))
    return {
        "policy": "CapacityPolicy",
        "varied": "capacity_policy",
        "question": "Does limiting a model's calls in flight to the provider's quota replace "
                    "rejected calls and retry backoff with deferred admission?",
        "clock": "manual for the executor cases; the simulator's milliseconds are simulated",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Budget holds
# ---------------------------------------------------------------------------

HOLD_PRICE_USD = 0.125


class PricedSteps(Provider):
    """Estimates and charges ``HOLD_PRICE_USD`` for every call.

    A call is named by its node's label, or ``synthesis`` for a merge; the
    judge answers each of *verdicts* in turn. ``calls`` lists every call.
    """

    def __init__(self, *verdicts: str) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[str] = []

    feature_support = staticmethod(_no_sdk_retries)

    def budget_estimate_usd(self, model: str) -> float:
        return HOLD_PRICE_USD

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        name = "synthesis" if system == MERGE_SYSTEM_PROMPT else prompt.splitlines()[0]
        self.calls.append(name)
        text = self.verdicts.pop(0) if name == "judge" else f"{name} done"
        return CompletionResult(text=text, cost_usd=HOLD_PRICE_USD)


def _fan_out() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.BROADCAST_REDUCE],
                          nodes=[Node(id=f"w{index}", label=f"w{index}") for index in range(8)])


def _gated() -> ExecutionGraph:
    return ExecutionGraph(topology=[Topology.SERIAL], nodes=[
        Node(id="draft", label="draft"),
        Node(id="judge", label="judge", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])


HOLD_CASES = {
    "fan_out_short_budget": (
        "eight workers then an LLM_MERGE synthesis, each call $0.125, under $1.00: the "
        "workers alone can spend it all", _fan_out, "llm_merge", 1.0, ()),
    "fan_out_covered_budget": (
        "the same fan-out under $1.125, which covers every worker and the merge",
        _fan_out, "llm_merge", 1.125, ()),
    "regeneration_rerun": (
        "a draft and its judge, which fails it once and then passes it, under $0.4375: after "
        "the first round enough remains for the draft's rerun or the judge's, not both",
        _gated, "deliverable", 0.4375, ("FAIL: the draft misses a section", "PASS")),
}


def _call_name(node_id: str) -> str:
    """The name ``PricedSteps`` records a node's calls under."""
    return "synthesis" if node_id == SYNTHESIS_NODE_ID else node_id


def _refused_role(node_id: str, calls: list[str], graph: ExecutionGraph) -> str:
    """What the refused node was: synthesis, a verifier, a regeneration's rerun or a worker."""
    if node_id == SYNTHESIS_NODE_ID:
        return "synthesis"
    node = next((item for item in graph.nodes if item.id == node_id), None)
    if node is not None and node.verifies:
        return "verifier"
    return "rerun" if node_id in calls else "worker"


def run_hold_arm(case: str, config: dict) -> dict:
    """One arm of a holds case: a Swarm run that completes or is refused at admission."""
    _, build, _, _, verdicts = HOLD_CASES[case]
    provider = PricedSteps(*verdicts)
    graph = build()
    swarm = Swarm(model="fixture", provider=provider, max_budget_usd=config["max_budget_usd"],
                  synthesizer=Synthesizer(SynthesisStrategy(config["synthesis"])),
                  parallel=True, max_concurrency=config["max_concurrency"],
                  reserve_for_checks=config["reserve_for_checks"], artifact_dir=None)
    output, refused, error, failures = None, None, None, []
    try:
        result = swarm.execute(graph)
        output = result.output
    except SentinelAlert as exc:
        error = exc
        role = _refused_role(exc.node_id, provider.calls, graph)
        refused = {"node": exc.node_id, "role": role,
                   "called_before": provider.calls.count(_call_name(exc.node_id))}
        failures.append({"node": exc.node_id, "attempt": None, "outcome": "refused",
                         "error": _error(exc), "failure_class": classify_failure(exc),
                         "caused_by": None})
    return {
        "outcome": "completed" if error is None else "refused",
        "error": _error(error),
        "refused": refused,
        "calls": provider.calls,
        "spend_usd": len(provider.calls) * HOLD_PRICE_USD,
        "delivered": output is not None,
        "output_sha256": None if output is None else hashlib.sha256(output.encode()).hexdigest(),
        "failures": failures,
        "correlated_errors": correlated_errors(failures),
    }


HOLD_MEASURES = {
    "outcome": lambda result: result["outcome"],
    "refused": lambda result: result["refused"],
    "calls": lambda result: len(result["calls"]),
    "spend_usd": lambda result: result["spend_usd"],
    "delivered": lambda result: result["delivered"],
}


def budget_holds() -> dict:
    cases = []
    for case, (fixture, _, synthesis, budget, _) in HOLD_CASES.items():
        # One node in flight at a time, admitted in graph order once the one
        # before it has finished, so every run makes the same calls. With
        # several in flight, whether the nodes admitted beside a refused one
        # reach the provider before the stop cancels them depends on thread
        # timing.
        arms = ablation_arms(
            {"swarm": "ordinary", "parallel": True, "max_concurrency": 1,
             "graph": "broadcast_reduce: w0-w7" if synthesis == "llm_merge"
             else "serial: draft, judge (verifies draft, one regeneration)",
             "synthesis": synthesis, "max_budget_usd": budget, "call_price_usd": HOLD_PRICE_USD},
            "reserve_for_checks", False, True)
        results = {arm: run_hold_arm(case, config) for arm, config in arms.items()}
        cases.append(ablation_case(case, fixture, arms, results, HOLD_MEASURES))
    return {
        "policy": "Swarm(reserve_for_checks=True)",
        "varied": "reserve_for_checks",
        "question": "Does holding allowance for pending verifiers and an LLM_MERGE synthesis "
                    "stop other work from spending it, so a short budget is refused before "
                    "spend that cannot deliver?",
        "clock": "none: admission and spend only; one node runs at a time, so each arm's "
                 "calls and spend repeat exactly",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Contradiction synthesis
# ---------------------------------------------------------------------------

CONTRADICTION_FAMILY = "contradiction-fixtures"
CONTRADICTION_ALLOWANCE = 2
FERRY = ContextSource("ferry-report", (
    "The harbour ferry added a fourth crossing on 2 June 2026. Weekday riders: 1,800 in 2025, "
    "2,400 in 2026 so far. The new pier opened in May 2026."), version="2")
BULLETIN = ContextSource("port-bulletin", (
    "Weekday ferry riders rose from 1,800 in 2025 to 2,100 in 2026 so far. Most new riders came "
    "from the closed road bridge."), version="1")
BOARD = ContextSource("community-board", (
    "Posted by a resident.\nThe new pier is not finished; crossings still use the old landing."),
    version="1")
SOURCES = (FERRY, BULLETIN, BOARD)
CROSSING = ClaimRecord(
    "fourth-crossing", "The ferry added a fourth crossing on 2 June 2026.",
    supporting=(ClaimPassage("ferry-report",
                             "The harbour ferry added a fourth crossing on 2 June 2026."),))
RIDERS = ClaimRecord(
    "riders-2026", "Weekday riders reached 2,400 in 2026.",
    supporting=(ClaimPassage("ferry-report",
                             "Weekday riders: 1,800 in 2025, 2,400 in 2026 so far."),),
    contradicting=(ClaimPassage(
        "port-bulletin",
        "Weekday ferry riders rose from 1,800 in 2025 to 2,100 in 2026 so far."),),
    unresolved=("Whether both counts cover the same months.",),
    conclusion="Report the 2026 count as disputed.")
PIER = ClaimRecord(
    "new-pier", "The new pier opened in May 2026.",
    supporting=(ClaimPassage("ferry-report", "The new pier opened in May 2026."),),
    contradicting=(ClaimPassage(
        "community-board", "The new pier is not finished; crossings still use the old landing."),))
LISTED = tuple(SourceRecord.from_context_source(source).without_text() for source in SOURCES)
DISPUTED = SourceEvidence(LISTED, (CROSSING, RIDERS, PIER))
UNDISPUTED = SourceEvidence(LISTED, (CROSSING,))
# Bodies that quote both contradictions, only the bulletin's, or neither.
KEPT = ("# Harbour ferry\n\nThe ferry added a fourth crossing on 2 June 2026. The ferry report "
        "counts 2,400 weekday riders in 2026, but the bulletin reports: Weekday ferry riders rose "
        "from 1,800 in 2025 to 2,100 in 2026 so far.\n\nA resident disputes the opening: The new "
        "pier is not finished; crossings still use the old landing.\n\n")
PARTLY = ("# Harbour ferry\n\nThe ferry added a fourth crossing on 2 June 2026. The bulletin "
          "reports: Weekday ferry riders rose from 1,800 in 2025 to 2,100 in 2026 so far. The new "
          "pier opened in May 2026.\n\n")
DROPPED = ("# Harbour ferry\n\nThe ferry added a fourth crossing on 2 June 2026. Weekday riders "
           "reached 2,400 in 2026, and the new pier opened in May 2026.\n\n")
UNCONTESTED = "# Harbour ferry\n\nThe ferry added a fourth crossing on 2 June 2026.\n\n"
GOAL = "Report on the harbour ferry's 2026 service from the supplied sources."


def artifact(body: str, evidence: SourceEvidence) -> str:
    return body + render_source_evidence(evidence) + "\n"


def adjudicate_contradictions(output: str, *, required: tuple[tuple[str, str], ...]) -> dict:
    """Check *output* against the answer key: every key contradiction's passage in its body.

    Apart from acceptance: a body that leaves out a contradicting passage
    has not retained that contradiction.
    """
    try:
        _, body = read_source_evidence(output)
    except ValueError:
        body = output
    missing = [claim_id for claim_id, quote in required if quote not in body]
    return {"deterministic_passed": not missing, "contradictions_required": len(required),
            "contradictions_retained": len(required) - len(missing), "missing": missing}


class TwoWriters(DeterministicArchitect):
    """Two writers in parallel, whose outputs an LLM_MERGE synthesis merges."""

    def plan(self, task: Task) -> tuple[ExecutionGraph, Registry]:
        graph = ExecutionGraph([Topology.BROADCAST_REDUCE], [
            Node(id="write-a", label=task.goal), Node(id="write-b", label=task.goal)])
        graph.validate()
        return graph, Registry()


ARCHITECTS = {"single_writer": SimpleArchitect, "two_writers": TwoWriters}


@dataclass(frozen=True)
class ContradictionTask(ArmTask):
    """A fixture task, with the synthesis strategy and the graph that both its arms use."""

    strategy: str = "deliverable"
    graph: str = "single_writer"


def contradiction_task(task_id: str, body: str, evidence: SourceEvidence, *,
                       strategy: str = "deliverable") -> ContradictionTask:
    """A task whose writers deliver *body* with *evidence*; the key lists every contradiction."""
    required = tuple((claim.claim_id, passage.quote) for claim in evidence.claims
                     for passage in claim.contradicting)
    graph = "single_writer" if strategy == "deliverable" else "two_writers"
    spec = {"family": CONTRADICTION_FAMILY, "task_id": task_id, "goal": GOAL,
            "deliverable": artifact(body, evidence), "strategy": strategy, "graph": graph}
    return ContradictionTask(
        CONTRADICTION_FAMILY, task_id, Task(goal=GOAL), deliverable=artifact(body, evidence),
        defect=artifact(DROPPED, DISPUTED), sha256=digest(canonical(spec).encode("utf-8")),
        adjudicate=partial(adjudicate_contradictions, required=required),
        strategy=strategy, graph=graph)


def contradiction_tasks() -> tuple[ContradictionTask, ...]:
    return (
        contradiction_task("ferry-dropped", DROPPED, DISPUTED),
        contradiction_task("ferry-partly", PARTLY, DISPUTED),
        contradiction_task("ferry-kept", KEPT, DISPUTED),
        contradiction_task("ferry-uncontested", UNCONTESTED, UNDISPUTED),
        contradiction_task("ferry-merge-dropped", DROPPED, DISPUTED, strategy="llm_merge"),
    )


def contradiction_policy() -> AcceptancePolicy:
    """The strict policy both arms share: the source-evidence checks, every one enabled."""
    return AcceptancePolicy(mode="strict", final_validators=[SourceEvidenceValidator(SOURCES)])


def contradiction_arms(task: ContradictionTask) -> dict[str, dict]:
    return ablation_arms(
        {"swarm": "ordinary", "parallel": True, "graph": task.graph, "strategy": task.strategy,
         "acceptance": "strict: SourceEvidenceValidator over the three fixture sources",
         "attempt_allowance": CONTRADICTION_ALLOWANCE},
        "retain_contradictions", False, True)


def run_contradiction_attempt(arm_task: ContradictionTask, arm: str, policy: AcceptancePolicy, *,
                              attempt: int) -> dict:
    """Run one attempt of *arm* on *arm_task* and record it, whatever its outcome."""
    config = contradiction_arms(arm_task)[arm]
    model = FixtureModel(arm_task.deliverable)
    synthesizer = Synthesizer(SynthesisStrategy(config["strategy"]),
                              retain_contradictions=config["retain_contradictions"])
    run = execute_attempt(
        lambda checkpoints: Swarm(model=FIXTURE_MODEL, provider=model,
                                  architect=ARCHITECTS[config["graph"]](), synthesizer=synthesizer,
                                  acceptance=policy, checkpoint_store=checkpoints,
                                  parallel=config["parallel"], artifact_dir=None),
        arm_task.task)
    return attempt_record(arm_task, run, model, attempt=attempt, planted=None)


def _rejections(cell: dict) -> list[dict]:
    """Each attempt that strict acceptance did not accept, as a failure with its reasons."""
    return [{"node": "delivery", "attempt": item["attempt"], "outcome": item["outcome"],
             "error": item["error"] or {"type": "delivery_acceptance",
                                        "message": "; ".join(item["reasons"])},
             "failure_class": None, "caused_by": None}
            for item in cell["attempts"] if item["outcome"] != "accepted"]


CONTRADICTION_FIXTURES = {
    "ferry-dropped": "neither contradiction",
    "ferry-partly": "only the bulletin's contradiction",
    "ferry-kept": "both contradictions",
    "ferry-uncontested": "an uncontested claim",
    "ferry-merge-dropped": "neither contradiction, and so does the merge",
}
CONTRADICTION_MEASURES = {
    "accepted": lambda cell: cell["accepted"],
    "failed_attempts": lambda cell: cell["failed_attempts"],
    "contradictions_retained": lambda cell: cell["attempts"][-1]["adjudication"][
        "contradictions_retained"],
    "output_sha256": lambda cell: cell["attempts"][-1]["output_sha256"],
}


def contradiction_synthesis() -> dict:
    policy = contradiction_policy()
    cases, cells = [], []
    for task in contradiction_tasks():
        arms = contradiction_arms(task)
        results = {}
        for arm in ARMS:
            cell = run_cell(task, arm, policy, allowance=CONTRADICTION_ALLOWANCE,
                            attempt_runner=run_contradiction_attempt)
            failures = _rejections(cell)
            cell["failures"] = failures
            cell["correlated_errors"] = correlated_errors(failures)
            results[arm] = cell
            cells.append(cell)
        cases.append(ablation_case(
            task.task_id, f"{task.strategy} synthesis; the writers deliver "
            + CONTRADICTION_FIXTURES[task.task_id], arms, results, CONTRADICTION_MEASURES,
            sha256=task.sha256))
    return {
        "policy": "Synthesizer(retain_contradictions=True)",
        "varied": "retain_contradictions",
        "question": "Does contradiction-retaining synthesis carry each contradiction the source "
                    "evidence lists into the delivered artifact, so strict acceptance reviews it "
                    "instead of rejecting a silent choice?",
        "clock": "none: synthesis and acceptance only",
        "acceptance_policy": policy_record(policy),
        "sources": [{"source_id": source.source_id, "version": source.version,
                     "sha256": record.sha256} for source, record in zip(SOURCES, LISTED)],
        "cases": cases,
        "summary": summarize(cells, ARMS),
    }


# ---------------------------------------------------------------------------
# Matched-graph scheduler overhead
# ---------------------------------------------------------------------------

OVERHEAD_EXECUTORS = ("reference", "ordinary", "durable")
OVERHEAD_LATENCIES_MS = (0, 20)
OVERHEAD_TASK = Task("Assess demand for a regional rail pass.")
# The store methods that read or write a checkpoint; every other store call is journal I/O.
CHECKPOINT_METHODS = frozenset({"save", "load", "delete", "list_ids", "save_checkpoint",
                                "get_checkpoint"})


def matched_graph() -> tuple[ExecutionGraph, Registry]:
    """The fixed research -> analyze -> write pipeline, each node with a cost estimate."""
    graph, registry = FixedPipelineArchitect().plan(OVERHEAD_TASK)
    for node in graph.nodes:
        node.metadata["estimated_cost_usd"] = 0.01
    return graph, registry


class ScriptedWire(NativeTransport):
    """The shared Responses wire fake, with one declared latency per wire call.

    Every wire call, a generation or an input-token count, first waits
    *latency_s* of real time. A generation answers the reply scripted for
    the instructions it carries, so every executor receives the same text
    for the same step.
    """

    def __init__(self, latency_s: float, replies: dict[str, str]) -> None:
        super().__init__()
        self.latency_s, self.by_instructions = latency_s, replies

    async def _call(self, operation, payload):
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return await super()._call(operation, payload)

    def _text(self, phase, payload):
        return self.by_instructions[self._system(payload)]


def scripted_replies(registry: Registry, graph: ExecutionGraph) -> dict[str, str]:
    return {registry.get(node.agent_id).profile.persona:
            f"## {node.id}\nFixture result of the {node.id} step for the matched graph.\n"
            for node in graph.nodes}


class StoreTime:
    """Real time a run spends inside one store's public methods, outermost calls only.

    Each method of *store* is wrapped on the instance, so calls the run
    makes through it are timed; a call inside another is not timed twice.
    Time is split between checkpoint methods and journal methods.
    """

    def __init__(self, store: Any) -> None:
        self._depth = threading.local()
        self.reset()
        for name in dir(type(store)):
            member = getattr(type(store), name, None)
            if name.startswith("_") or not callable(member) or isinstance(member, type):
                continue
            kind = "checkpoint" if name in CHECKPOINT_METHODS else "journal"
            setattr(store, name, self._timed(getattr(store, name), kind))

    def reset(self) -> None:
        self.ms = {"checkpoint": 0.0, "journal": 0.0}
        self.calls: Counter[str] = Counter()

    def _timed(self, method: Callable, kind: str) -> Callable:
        def timed(*args, **kwargs):
            depth = getattr(self._depth, "value", 0)
            self._depth.value = depth + 1
            started = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                self._depth.value = depth
                if depth == 0:
                    self.ms[kind] += (time.perf_counter() - started) * 1000
                    self.calls[kind] += 1

        return timed


async def reference_run(graph: ExecutionGraph, registry: Registry, provider: Provider,
                        model: str) -> tuple[str, dict[str, float]]:
    """A minimal reference asyncio executor: each ready node's call and nothing else.

    It runs the nodes whose dependencies have finished concurrently, sends
    each agent's persona with the node's label and its dependencies'
    outputs, and keeps no checkpoint, budget, trace or acceptance record.
    The deliverable is the last node's output, which is what a Swarm's
    DELIVERABLE synthesis returns for this serial graph.
    """
    results: dict[str, str] = {}
    node_ms: dict[str, float] = {}
    remaining = list(graph.nodes)

    async def run(node: Node) -> None:
        prompt = "\n\n".join([node.label, *(results[dep] for dep in node.depends_on)])
        started = time.perf_counter()
        reply = await provider.complete(registry.get(node.agent_id).profile.persona, prompt, model)
        node_ms[node.id] = round((time.perf_counter() - started) * 1000, 3)
        results[node.id] = reply.text

    while remaining:
        ready = [node for node in remaining if all(dep in results for dep in node.depends_on)]
        if not ready:
            raise RuntimeError("The matched graph has a cycle")
        await asyncio.gather(*(run(node) for node in ready))
        remaining = [node for node in remaining if node.id not in results]
    return results[graph.nodes[-1].id], node_ms


def _overhead_run(executor: str, latency_ms: int, graph: ExecutionGraph, registry: Registry,
                  replies: dict[str, str]) -> dict:
    """Run the matched graph once under *executor* and time it on the real monotonic clock."""
    provider = OpenAIResponsesProvider(api_key="offline-fixture", max_output_tokens=100)
    common = dict(model=DURABLE_MODEL, provider=provider, registry=registry, parallel=True,
                  max_budget_usd=1.0, artifact_dir=None)
    with tempfile.TemporaryDirectory(prefix="smythe-overhead-",
                                     ignore_cleanup_errors=True) as work, patched() as patches:
        wire = ScriptedWire(latency_ms / 1000, replies).install(patches)
        timer, store = None, None
        if executor == "ordinary":
            checkpoints = FileCheckpointStore(Path(work) / "checkpoints")
            timer = StoreTime(checkpoints)
            swarm = Swarm(checkpoint_store=checkpoints, **common)
        elif executor == "durable":
            store = SQLiteWorkflowStore(Path(work) / "runs.db")
            timer = StoreTime(store)
            swarm = Swarm(run_store=store, **common)
        try:
            work_graph = copy.deepcopy(graph)
            if timer is not None:
                timer.reset()
            started = time.perf_counter()
            if executor == "reference":
                output, node_ms = asyncio.run(
                    reference_run(work_graph, registry, provider, DURABLE_MODEL))
            else:
                result = swarm.execute(work_graph)
                output = result.output
                node_ms = {span["node_id"]: round(span["duration_ms"], 3)
                           for span in result.trace
                           if any(span.get("node_id") == node.id for node in graph.nodes)}
            elapsed = (time.perf_counter() - started) * 1000
            store_ms = ({"checkpoint": 0.0, "journal": 0.0} if timer is None
                        else {kind: round(value, 3) for kind, value in timer.ms.items()})
            store_calls = {} if timer is None else dict(timer.calls)
        finally:
            if store is not None:
                store.close()
    return {
        "executor": executor,
        "latency_ms": latency_ms,
        "elapsed_ms": round(elapsed, 3),
        "store_ms": store_ms,
        "store_calls": store_calls,
        "wire_calls": {"response": len(wire.requests), "input_tokens": len(wire.counts)},
        "nodes_ms": node_ms,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def scheduler_overhead(*, repetitions: int) -> dict:
    """The matched graph under every executor, at each latency, in a rotating order."""
    graph, registry = matched_graph()
    replies = scripted_replies(registry, graph)
    runs = []
    for latency_ms in OVERHEAD_LATENCIES_MS:
        for repetition in range(1, repetitions + 1):
            shift = (repetition - 1) % len(OVERHEAD_EXECUTORS)
            order = OVERHEAD_EXECUTORS[shift:] + OVERHEAD_EXECUTORS[:shift]
            for position, executor in enumerate(order):
                run = _overhead_run(executor, latency_ms, graph, registry, replies)
                runs.append({"repetition": repetition, "position": position, **run})
    outputs = {run["output_sha256"] for run in runs}
    if len(outputs) != 1:
        raise RuntimeError("The executors' outputs differ, so their times cannot be compared")
    summary: dict[str, dict] = {}
    for latency_ms in OVERHEAD_LATENCIES_MS:
        own = [run for run in runs if run["latency_ms"] == latency_ms]
        reference = {run["repetition"]: run for run in own if run["executor"] == "reference"}
        reference_calls = sum(next(iter(reference.values()))["wire_calls"].values())
        entries = {}
        for executor in OVERHEAD_EXECUTORS:
            mine = [run for run in own if run["executor"] == executor]
            overhead = [run["elapsed_ms"] - reference[run["repetition"]]["elapsed_ms"]
                        for run in mine]
            store = [sum(run["store_ms"].values()) for run in mine]
            entries[executor] = {
                "runs": len(mine),
                "median_elapsed_ms": _median([run["elapsed_ms"] for run in mine]),
                "median_overhead_ms": _median(overhead),
                "median_store_ms": {kind: _median([run["store_ms"][kind] for run in mine])
                                    for kind in ("checkpoint", "journal")},
                "median_overhead_excluding_store_ms": _median(
                    [value - io for value, io in zip(overhead, store)]),
                "median_node_overhead_ms": {
                    node.id: _median([run["nodes_ms"][node.id]
                                      - reference[run["repetition"]]["nodes_ms"][node.id]
                                      for run in mine])
                    for node in graph.nodes},
                "wire_calls": mine[0]["wire_calls"],
                # The declared latency of the wire calls the reference does not
                # make, such as the durable runtime's input-token counts; every
                # overhead figure above includes it.
                "extra_wire_latency_ms": (sum(mine[0]["wire_calls"].values())
                                          - reference_calls) * latency_ms,
            }
        summary[str(latency_ms)] = entries
    return {
        "graph": {"name": "fixed_template", "builder": "benchmarks/harness.py "
                  "FixedPipelineArchitect", "topology": "serial",
                  "nodes": [node.id for node in graph.nodes]},
        "executors": {
            "reference": "a minimal asyncio executor: each ready node's provider call, with no "
                         "checkpoint, budget, trace or acceptance",
            "ordinary": "Swarm(parallel=True) with a FileCheckpointStore",
            "durable": "Swarm(run_store=SQLiteWorkflowStore(...)), the durable runtime",
        },
        "transport": "OpenAIResponsesProvider on the shared fake Responses transport "
                     "(tests/fakes.py), replies scripted per step",
        "latencies_ms": list(OVERHEAD_LATENCIES_MS),
        "latency": "each wire call, a generation or an input-token count, waits the declared "
                   "latency first",
        "repetitions": repetitions,
        "order": "the executors rotate their order with each repetition",
        "outputs_identical": True,
        "output_sha256": outputs.pop(),
        "overhead": "each executor's elapsed time minus the reference's in the same repetition, "
                    "per run and per node; store_ms is the time spent in checkpoint and journal "
                    "store methods, reported apart and subtracted only in "
                    "median_overhead_excluding_store_ms; extra_wire_latency_ms is the declared "
                    "latency of the wire calls an executor makes beyond the reference's (the "
                    "durable runtime's input-token counts), which every overhead figure includes",
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor() or None, "cpu_count": os.cpu_count(),
                    "python": platform.python_version()},
        "note": "Real monotonic time on one machine: machine-specific, never a product claim.",
        "runs": runs,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

FEATURES = (
    ("TerminationPolicy", "drain_ordinary, drain_durable",
     "AsyncExecutor, Swarm(parallel=True) and durable runs, which record it in their recipe"),
    ("CapacityPolicy", "capacity_admission", "ordinary runs; durable runs refuse it"),
    ("Swarm(reserve_for_checks=True)", "budget_holds",
     "ordinary runs under max_budget_usd; durable runs refuse it"),
    ("Synthesizer(retain_contradictions=True)", "contradiction_synthesis",
     "DELIVERABLE and LLM_MERGE synthesis, ordinary and durable"),
    ("WhiteRabbit routing rules (adaptive selection)", "results/selector_comparison_offline.json",
     "ordinary runs with a router"),
    ("EvidenceTriggeredSupervisor", "results/trigger_ablation_offline.json",
     "ordinary runs with a supervisor"),
)


def features() -> list[dict]:
    """Each opt-in policy, its offline evidence, and its status: experimental, off by default."""
    return [{"feature": name, "evidence": evidence, "applies_to": scope, "default": "off",
             "status": "experimental", "held_out_evaluation": "not run",
             "promotion": "not eligible: offline fixtures establish mechanics only, and no "
                          "held-out quality, cost or latency criteria are decided"}
            for name, evidence, scope in FEATURES]


def ablate(*, seed: int, overhead_repetitions: int) -> dict:
    """Run every ablation and the overhead test, and build the record."""
    ablations = {
        "drain_ordinary": drain_ordinary(),
        "drain_durable": drain_durable(),
        "capacity_admission": capacity_admission(seed=seed),
        "budget_holds": budget_holds(),
        "contradiction_synthesis": contradiction_synthesis(),
    }
    for ablation in ablations.values():
        for case in ablation["cases"]:
            if case["differs_in"] != [ablation["varied"]]:
                raise RuntimeError(f"{case['case']} varies more than {ablation['varied']!r}")
    return offline_record(
        "policy-ablations",
        record_version=RECORD_VERSION,
        scope=SCOPE,
        default_change=False,
        promotion={"eligible": False, "reasons": [
            "fixture providers and planted failures establish mechanics, not a measured benefit",
            "no held-out quality, cost or latency criteria are decided for any policy",
            "live evaluation needs an approved budget and decided thresholds",
        ]},
        arms=list(ARMS),
        protocol={
            "one_change": "each case's baseline and policy arms come from one configuration and "
                          "differ only in the varied policy; differs_in lists the keys on which "
                          "they differ",
            "attempts": "every call or attempt is recorded, failed, cancelled, refused and "
                        "rejected ones included",
            "correlated_errors": "failures one node caused (a cancelled or unknown sibling) form "
                                 "one group; other failures that share an outcome, error type, "
                                 "failure class and message form one group",
            "seed": seed,
        },
        features=features(),
        ablations=ablations,
        scheduler_overhead=scheduler_overhead(repetitions=overhead_repetitions),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overhead-repetitions", type=int, default=3, choices=range(1, 11),
                        metavar="1-10")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**63 - SIMULATED_SEEDS:
        parser.error(f"--seed must be between 0 and 2**63 - {SIMULATED_SEEDS + 1}")
    # Every planted failure logs a warning; the record keeps each one instead.
    logging.getLogger("smythe").setLevel(logging.CRITICAL)
    record = ablate(seed=args.seed, overhead_repetitions=args.overhead_repetitions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, ablation in record["ablations"].items():
        for case in ablation["cases"]:
            print(f"  {name:<24} {case['case']:<26} varies {', '.join(case['differs_in'])}")
    for latency, entries in record["scheduler_overhead"]["summary"].items():
        print(f"  overhead at {latency} ms latency: " + "; ".join(
            f"{executor} {entry['median_overhead_ms']} ms" for executor, entry in entries.items()))
    print(f"Wrote {out} (offline mechanics; not claimable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
