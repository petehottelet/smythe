"""A repair killed while its successor reuses results recovers without repeating work.

A child process applies a repair plan whose one item regenerates a research
run in a successor, and is killed with TerminateProcess or SIGKILL while the
successor's ``reuse`` operations commit, so no cleanup handler runs and its
lease is never released. This process then finds the repaired run as it was
and the repair in progress, and finishes the successor with an ordinary
resume, past the lost lease. The runs are the three-source research of
``test_workflow_reuse``, served by this process over loopback HTTP.
"""

from __future__ import annotations

import json

import pytest

import smythe.workflow
from fakes import ManualClock, kill_and_reap, launch_child, wait_for_barrier
from fixture_service import FixtureService, serve_http
from smythe import SQLiteWorkflowStore, archive_heads
from test_workflow_repair import journal_calls, run_rows
from test_workflow_reuse import (
    CHANGED, KEPT, TEXTS, Catalog, install_replies, research, research_graph,
)

SCOPE = "research-team"

# Child code: apply the plan and hold where ``stop`` says until the test kills it.
CHILD = r"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fakes import write_barrier
from smythe import SourceStatus, SQLiteWorkflowStore
from test_workflow_reuse import install_replies, research

path, folder, url, digest, stop = Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:6]
known = json.loads(sys.argv[6])


class Known:
    # The catalog the test described the sources with.
    def describe(self, source_id):
        return SourceStatus(sha256=known[source_id]) if source_id in known else None


def hold(where):
    write_barrier(folder / "stopped.json", {"stop": where})
    time.sleep(600)


if stop == "reused":
    # The successor's first reuse operation is committed; the others are not begun.
    complete = SQLiteWorkflowStore.complete_operation

    def complete_operation(self, lease, key, result, **kwargs):
        operation = complete(self, lease, key, result, **kwargs)
        if key.startswith("reuse/"):
            hold(stop)
        return operation

    SQLiteWorkflowStore.complete_operation = complete_operation
else:
    # Every reuse operation is committed; the checkpoint that consumes them is not.
    save = SQLiteWorkflowStore.save_checkpoint

    def save_checkpoint(self, lease, revision, checkpoint, **kwargs):
        if checkpoint["status"] == "running":
            hold(stop)
        return save(self, lease, revision, checkpoint, **kwargs)

    SQLiteWorkflowStore.save_checkpoint = save_checkpoint

install_replies(SimpleNamespace(setattr=setattr), [])
with SQLiteWorkflowStore(path) as store:
    swarm = research(store, SimpleNamespace(url=url), folder)
    plan = swarm.plan_repair(sources=Known())
    swarm.repair(plan, plan_sha256=digest, max_total_budget_usd=1,
                 reason="The churn source has a new version", sources=Known())
"""


def source_reads(service) -> dict[str, int]:
    return {source_id: item["reads"] for source_id, item in service.snapshot()["sources"].items()}


@pytest.mark.parametrize("stop", ["reused", "seeded"])
def test_a_killed_repair_leaves_its_predecessor_intact_and_resumes_without_repeating_work(
        tmp_path, monkeypatch, stop):
    path, service = tmp_path / "runs.db", FixtureService()
    for source_id, text in TEXTS.items():
        service.add_source(source_id, text)
    log: list[str] = []
    install_replies(monkeypatch, log)
    with serve_http(service) as endpoint:
        with SQLiteWorkflowStore(path) as store:
            first = research(store, endpoint, tmp_path).execute(research_graph(endpoint))
            service.change_source("churn", CHANGED)
            plan = research(store, endpoint, tmp_path).plan_repair(sources=Catalog(service))
            ((item,), digest) = plan.to_dict()["items"], plan.plan_sha256
            assert (item["run_id"], item["action"]) == (first.execution_id, "regenerate")
            rows = run_rows(store, first.execution_id)
        known = {source_id: source["sha256"]
                 for source_id, source in service.snapshot()["sources"].items()}
        reads = source_reads(service)

        process = launch_child(CHILD, path, tmp_path, endpoint.url, digest, stop, json.dumps(known))
        try:
            assert wait_for_barrier(process, tmp_path / "stopped.json", timeout=300) == {
                "stop": stop}
            reaped = kill_and_reap(process)
        finally:
            kill_and_reap(process)
        assert reaped.killed and reaped.returncode != 0

        # The child's 60 second lease is still held; this process resumes once it lapses.
        with SQLiteWorkflowStore(path, clock_ns=ManualClock.after(120)) as store:
            (successor,) = [run["run_id"] for run in store.list_runs()
                            if run["run_id"] != first.execution_id]
            # The repaired run is exactly as it was, and the successor called nothing.
            assert run_rows(store, first.execution_id) == rows
            assert journal_calls(store, successor) == [] and source_reads(service) == reads
            committed = {operation["operation_key"]: operation
                         for operation in store.list_operations(successor, "reuse")}
            assert len(committed) == (1 if stop == "reused" else len(KEPT))
            assert {operation["state"] for operation in committed.values()} == {"completed"}

            # The plan reports the repair in progress, which is no head of its own.
            swarm = research(store, endpoint, tmp_path)
            (pending,) = swarm.plan_repair(sources=Catalog(service)).to_dict()["items"]
            assert (pending["run_id"], pending["action"]) == (first.execution_id, "review")
            assert "repair_in_progress" in pending["reasons"]
            assert pending["repairs_in_progress"] == [successor]
            assert archive_heads(store, SCOPE) == [first.execution_id]

            # Resume reuses what the child committed without running it again.
            loaded, original = [], smythe.workflow._reused

            def reused(predecessor, node_id):
                loaded.append(node_id)
                return original(predecessor, node_id)

            monkeypatch.setattr(smythe.workflow, "_reused", reused)
            labels = len(log)
            result = swarm.resume(successor)

            assert result.accepted
            kept = {operation["inputs"]["node_id"] for operation in committed.values()}
            assert sorted(loaded) == sorted(set(KEPT) - kept)
            reuse = {operation["operation_key"]: operation
                     for operation in store.list_operations(successor, "reuse")}
            assert sorted(reuse) == sorted(f"reuse/{node_id}" for node_id in KEPT)
            for key, operation in committed.items():
                assert reuse[key]["operation_id"] == operation["operation_id"]
                assert reuse[key]["result"] == operation["result"]
            assert {operation["state"] for operation in reuse.values()} == {"applied"}
            # One read and one model call, each made once: no duplicate. The
            # frozen clock gives both one creation time, so their order is not kept.
            assert sorted(journal_calls(store, successor)) == [
                ("execution", "node/extract-churn"), ("tool", "node/read-churn")]
            assert source_reads(service) == {**reads, "churn": reads["churn"] + 1}
            assert log[labels:] == ["Extract the churn finding"]
            assert run_rows(store, first.execution_id) == rows
            assert archive_heads(store, SCOPE) == [successor]
            assert store.audit(successor)["ok"]
            assert any(segment["interrupted"] for segment in result.timing["segments"])
