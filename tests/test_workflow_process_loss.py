"""A durable strict run killed around its acceptance decision recovers exactly.

Each case kills a real child process with TerminateProcess or SIGKILL, so no
cleanup handler runs and the child's lease is never released. The run then
resumes in this process, past the lost lease, against the same journal.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from fakes import ManualClock, NativeTransport, kill_and_reap, launch_child, wait_for_barrier
from smythe import AcceptancePolicy, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm
from smythe.acceptance import text_sha256
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.provider_responses import OpenAIResponsesProvider
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding
from test_workflow_acceptance import generation_key

# A durable OfflineProvider echoes each step: "offline: <label>".
DELIVERED = "offline: Write the summary"


class Mentions:
    """A final validator: the delivered text must mention ``word``."""

    validated: list[str] = []

    def __init__(self, word: str = "summary") -> None:
        self.word = word
        self.identity = ValidatorIdentity.for_config("mentions", "1", {"word": word})

    def validate(self, subject):
        Mentions.validated.append(subject.text())
        if self.word in subject.text():
            return ValidationReport("output", validator=self.identity)
        return ValidationReport("output", (hard_finding("missing", f"no {self.word!r}"),),
                                validator=self.identity)


def build(store, word: str = "summary") -> Swarm:
    return Swarm(model="offline", provider=OfflineProvider(), architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1,
                 acceptance=AcceptancePolicy(mode="strict", final_validators=[Mentions(word)]))


def graph() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="read", label="Read the source"),
        Node(id="write", label="Write the summary", depends_on=["read"]),
    ])


# Child code that holds the run at ``stop`` until the test kills the child.
HOLD = r"""
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe import SQLiteWorkflowStore
from smythe.workflow import WorkflowRuntime
from test_workflow_process_loss import Mentions

path, folder, stop = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def hold(where):
    write_barrier(folder / "stopped.json", {"stop": where})
    time.sleep(600)


if stop == "deciding":
    # Synthesis is saved; the validator is checking the output.
    check = Mentions.validate
    Mentions.validate = lambda self, subject: hold(stop) or check(self, subject)
elif stop == "decided":
    # The decision is committed; the completed checkpoint is not.
    save = SQLiteWorkflowStore.save_checkpoint

    def save_checkpoint(self, lease, revision, checkpoint, **kwargs):
        if checkpoint["status"] == "completed":
            hold(stop)
        return save(self, lease, revision, checkpoint, **kwargs)

    SQLiteWorkflowStore.save_checkpoint = save_checkpoint
else:
    # The completed checkpoint, with its decision, is committed.
    WorkflowRuntime._result = lambda self, *args, **kwargs: hold(stop)
"""

CHILD = HOLD + r"""
from test_workflow_process_loss import build, graph

with SQLiteWorkflowStore(path) as store:
    build(store, sys.argv[4]).execute(graph())
"""


def operations(path, run_id):
    with closing(sqlite3.connect(path)) as db:
        return {kind: state for kind, state in db.execute(
            "SELECT kind, state FROM workflow_operations WHERE run_id=?", (run_id,))}


@pytest.mark.parametrize("word,decision", [("summary", "accepted"), ("sources", "rejected")])
@pytest.mark.parametrize("stop", ["deciding", "decided", "completed"])
def test_killed_durable_run_resumes_its_decision_without_new_calls(tmp_path, monkeypatch,
                                                                   stop, word, decision):
    path = tmp_path / "runs.db"
    process = launch_child(CHILD, path, tmp_path, stop, word)
    try:
        assert wait_for_barrier(process, tmp_path / "stopped.json", timeout=60) == {"stop": stop}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0

    # The killed child still holds the lease; a successor resumes once it lapses.
    with SQLiteWorkflowStore(path, clock_ns=ManualClock.after(120)) as store:
        (run,) = store.list_runs()
        run_id = run["run_id"]
        before = store.inspect_run(run_id)
        kinds = operations(path, run_id)
        assert kinds["synthesis"] in ("completed", "applied")
        assert kinds.get("acceptance") == {"deciding": "started", "decided": "completed",
                                           "completed": "applied"}[stop]

        calls = []

        async def complete(self, system, prompt, model):
            calls.append(prompt)
            raise AssertionError("Resume made a provider call")

        monkeypatch.setattr(OfflineProvider, "complete", complete)
        Mentions.validated = []
        result = build(store, word).resume(run_id)
        after = store.inspect_run(run_id)

    # Only a decision the child never committed is made again, on the same bytes.
    assert Mentions.validated == ([DELIVERED] if stop == "deciding" else [])
    assert calls == [] and after["calls"] == before["calls"]
    assert result.acceptance.decision.value == decision
    assert result.acceptance.output_sha256 == text_sha256(DELIVERED)
    if decision == "accepted":
        assert (result.output, result.rejected_output) == (DELIVERED, None)
    else:
        assert (result.output, result.rejected_output) == ("", DELIVERED)
    assert operations(path, run_id) == {"synthesis": "applied", "acceptance": "applied"}
    assert after["status"] == "completed" and after["ok"]
    assert any(segment["interrupted"] for segment in result.timing["segments"])


# ---------------------------------------------------------------------------
# Priced calls and a regenerated draft
# ---------------------------------------------------------------------------

# The native fake bills each generation 100 input and 10 output tokens.
CALL_NANOUSD = 1_500_000
DRAFTS = ["An outline.", "A summary of the source."]


def priced_transport(monkeypatch) -> NativeTransport:
    """Priced native responses: the judge fails the first draft and passes the second."""
    transport = NativeTransport().install(monkeypatch)
    drafts, verdicts = list(DRAFTS), ["FAIL: not a summary", "PASS"]
    transport.replies["execution"] = lambda prompt: (
        verdicts if "Check the summary" in prompt else drafts).pop(0)
    return transport


def priced(store) -> Swarm:
    provider = OpenAIResponsesProvider(api_key="test-key", max_output_tokens=100)
    return Swarm(model="gpt-6-astra", provider=provider, architect=SimpleArchitect(),
                 run_store=store, max_budget_usd=1,
                 acceptance=AcceptancePolicy(mode="strict", final_validators=[Mentions()]))


def gated_draft() -> ExecutionGraph:
    return ExecutionGraph([Topology.SERIAL], [
        Node(id="draft", label="Write the summary"),
        Node(id="judge", label="Check the summary", depends_on=["draft"], verifies="draft",
             max_regenerations=1),
    ])


PRICED_CHILD = HOLD + r"""
from types import SimpleNamespace

from test_workflow_process_loss import gated_draft, priced, priced_transport

# Serve the fake transport to the provider, as pytest's monkeypatch does.
priced_transport(SimpleNamespace(setattr=setattr))
with SQLiteWorkflowStore(path) as store:
    priced(store).execute(gated_draft())
"""


def operation_keys(path, run_id):
    with closing(sqlite3.connect(path)) as db:
        return {key: (kind, state) for key, kind, state in db.execute(
            "SELECT operation_key, kind, state FROM workflow_operations WHERE run_id=?", (run_id,))}


@pytest.mark.parametrize("stop", ["deciding", "decided", "completed"])
def test_killed_priced_run_keeps_its_charges_generations_and_regeneration(tmp_path, monkeypatch,
                                                                          stop):
    path = tmp_path / "priced.db"
    process = launch_child(PRICED_CHILD, path, tmp_path, stop)
    try:
        assert wait_for_barrier(process, tmp_path / "stopped.json", timeout=60) == {"stop": stop}
        reaped = kill_and_reap(process)
    finally:
        kill_and_reap(process)
    assert reaped.killed and reaped.returncode != 0

    # Any provider call on resume would reach this fake and be logged.
    transport = priced_transport(monkeypatch)
    with SQLiteWorkflowStore(path, clock_ns=ManualClock.after(120)) as store:
        (run,) = store.list_runs()
        run_id = run["run_id"]
        before = store.inspect_run(run_id)
        saved = {node["id"]: node["metadata"]
                 for node in store.get_checkpoint(run_id)["checkpoint"]["graph"]["nodes"]}
        Mentions.validated = []
        result = priced(store).resume(run_id)
        after = store.inspect_run(run_id)

    # The child paid for two drafts and two verdicts, and settled every charge.
    assert before["call_count"] == 4 and before["confirmed_nanousd"] == 4 * CALL_NANOUSD
    assert before["reserved_nanousd"] == before["unknown_nanousd"] == 0
    # Resume sent nothing and left every call and charge as the child settled it.
    assert transport.log == [] and after["calls"] == before["calls"]
    assert (after["confirmed_nanousd"], after["call_count"]) == (4 * CALL_NANOUSD, 4)
    assert result.workflow_accounting["confirmed_nanousd"] == 4 * CALL_NANOUSD
    assert result.total_cost_usd == 4 * CALL_NANOUSD / 1e9
    # The draft's second generation, after one regeneration, is the one decided.
    draft, judge = result.graph.nodes
    assert draft.metadata["execution_generation"] == judge.metadata["execution_generation"] == 1
    assert judge.metadata["regenerations_used"] == 1
    assert len(draft.metadata["rejected_generations"]) == 1
    for node in result.graph.nodes:
        for key in ("execution_generation", "regenerations_used", "rejected_generations"):
            assert node.metadata.get(key) == saved[node.id].get(key)
    assert [source.to_dict() for source in result.acceptance.sources] == [
        {"node_id": "draft", "execution_generation": 1}]
    assert result.accepted and result.output == DRAFTS[1]
    # Only a decision the child never committed is made again.
    assert Mentions.validated == ([DRAFTS[1]] if stop == "deciding" else [])
    keys = operation_keys(path, run_id)
    assert keys[generation_key(draft=1, judge=1)] == ("synthesis", "applied")
    assert "synthesis" not in keys
    assert [state for kind, state in keys.values() if kind == "acceptance"] == ["applied"]
    assert after["status"] == "completed" and after["ok"]
