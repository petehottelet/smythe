from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from benchmarks.run_durability_benchmark import (
    _append_call_event,
    _call_event_summary,
    _count_lines,
    _parse_result,
    _run_to_completion,
    _worker_cmd,
    kill_when_dispatches_reach,
    smythe_worker,
)


def test_parse_result_extracts_last_result_line():
    stdout = "noise\nRESULT {\"a\": 1}\nmore noise\nRESULT {\"a\": 2}\n"
    assert _parse_result(stdout) == {"a": 2}
    with pytest.raises(RuntimeError):
        _parse_result("no result here\n")


def test_smythe_worker_logs_one_call_per_node(tmp_path, capsys):
    log = tmp_path / "calls.log"
    smythe_worker(argparse.Namespace(
        n=6, latency_ms=1, concurrency=3, calls_log=str(log),
        ckpt_dir="none", mode="run",
    ))
    result = _parse_result(capsys.readouterr().out)
    assert result["completed"] == 6
    summary = _call_event_summary(log)
    assert _count_lines(log) == 12
    assert summary["dispatches"] == 6
    assert summary["completions"] == 6
    assert summary["duplicate_dispatches"] == 0
    assert summary["inflight_attempts"] == 0


def test_event_summary_counts_repeated_dispatch_and_inflight_exposure(tmp_path):
    log = tmp_path / "calls.jsonl"
    _append_call_event(
        log,
        event="dispatched",
        operation_id="n0001",
        attempt_id="attempt-a",
        durable=True,
    )
    _append_call_event(
        log,
        event="dispatched",
        operation_id="n0001",
        attempt_id="attempt-b",
        durable=True,
    )
    _append_call_event(
        log,
        event="completed",
        operation_id="n0001",
        attempt_id="attempt-b",
        durable=True,
    )

    summary = _call_event_summary(log)
    assert summary["dispatches"] == 2
    assert summary["completions"] == 1
    assert summary["unique_operation_ids"] == 1
    assert summary["duplicate_dispatches"] == 1
    assert summary["inflight_attempts"] == 1


def test_event_summary_rejects_completion_without_matching_dispatch(tmp_path):
    log = tmp_path / "calls.jsonl"
    _append_call_event(
        log,
        event="completed",
        operation_id="n0001",
        attempt_id="attempt-a",
        durable=False,
    )

    with pytest.raises(RuntimeError, match="no matching dispatch"):
        _call_event_summary(log)


def test_hard_kill_and_resume_duplicates_at_most_one_wave(tmp_path):
    """The durability guarantee, exercised with a real process kill.

    With checkpoint_every_n_nodes=1, a crash may re-dispatch work that was
    exposed but not yet checkpointed: the in-flight concurrency wave plus
    any completion/checkpoint race inside the kill-poll window, never the
    completed remainder of the graph in this controlled protocol.
    """
    n, kill_at, concurrency = 24, 12, 4
    log = tmp_path / "calls.log"
    ckpt = tmp_path / "ckpt"

    kw = dict(n=n, latency_ms=100, concurrency=concurrency,
              calls_log=log, persist=ckpt)
    at_kill = kill_when_dispatches_reach(
        _worker_cmd("smythe", mode="run", **kw), log, kill_at)
    assert at_kill["dispatches"] >= kill_at
    assert at_kill["inflight_attempts"] >= 1

    result = _run_to_completion(_worker_cmd("smythe", mode="resume", **kw))
    assert result["completed"] == n

    summary = _call_event_summary(log)
    assert summary["unique_operation_ids"] == n
    assert 0 <= summary["duplicate_dispatches"] <= 2 * concurrency
    replayed = {
        operation_id
        for operation_id, count in summary["dispatch_counts"].items()
        if count > 1
    }
    assert len(replayed) == summary["duplicate_dispatches"]


def test_worker_cmd_uses_this_interpreter_and_harness():
    cmd = _worker_cmd("smythe", n=4, latency_ms=1, concurrency=2,
                      calls_log=Path("x.log"), persist=None, mode="run")
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("run_durability_benchmark.py")
    assert "smythe-worker" in cmd
    assert "--ckpt-dir" in cmd and "none" in cmd
    assert "--durable-events" not in cmd

    durable = _worker_cmd(
        "smythe", n=4, latency_ms=1, concurrency=2,
        calls_log=Path("x.log"), persist=Path("ckpt"), mode="run",
    )
    assert "--durable-events" in durable
