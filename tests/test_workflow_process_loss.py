"""A durable strict run killed around its acceptance decision recovers exactly.

Each case kills a real child process with TerminateProcess or SIGKILL, so no
cleanup handler runs and the child's lease is never released. The run then
resumes in this process, past the lost lease, against the same journal.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from fakes import ManualClock, kill_and_reap, launch_child, wait_for_barrier
from smythe import AcceptancePolicy, OfflineProvider, SimpleArchitect, SQLiteWorkflowStore, Swarm
from smythe.acceptance import text_sha256
from smythe.graph import ExecutionGraph, Node, Topology
from smythe.validation import ValidationReport, ValidatorIdentity, hard_finding

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


CHILD = r"""
import sys
import time
from pathlib import Path

from fakes import write_barrier
from smythe import SQLiteWorkflowStore
from smythe.workflow import WorkflowRuntime
from test_workflow_process_loss import Mentions, build, graph

path, folder, stop, word = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]


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

with SQLiteWorkflowStore(path) as store:
    build(store, word).execute(graph())
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
