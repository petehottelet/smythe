"""Durability and fan-out overhead benchmark: smythe vs LangGraph.

Two cells, both offline (no API keys, no cost), both using the same
simulated provider call. Each attempt records a ``dispatched`` event before
the latency and a ``completed`` event afterwards. The dispatch event is a
conservative record of possible spend exposure, not proof of provider billing.

Cell A — fan-out scheduler overhead. N independent nodes (a pure
broadcast), 25 ms simulated calls, concurrency 16, no persistence.
Metric: wall time vs ideal, per-node scheduler overhead.

Cell B — kill-and-resume durability granularity. N=64 nodes, 100 ms
calls, concurrency 8, strongest persistence on both sides (smythe:
``FileCheckpointStore`` with ``checkpoint_every_n_nodes=1``; LangGraph:
``AsyncSqliteSaver`` with ``durability="sync"``). The orchestrator
HARD-KILLS the worker process (TerminateProcess / SIGKILL — a real
crash, no cleanup) once half the operations are durably recorded as
dispatched, then resumes. Metric: repeated dispatches by operation ID plus
in-flight exposure at the kill point. A repeated dispatch may become duplicate
spend on a real image or LLM workload.

Why this discriminates: smythe checkpoints after every node, so a crash
re-runs at most the in-flight concurrency wave. A superstep-granular
engine persists a wide parallel wave only at its boundary, so a
mid-wave crash re-runs every completed sibling in that wave — and the
wider the fan-out, the worse that gets.

Run (needs langgraph + langgraph-checkpoint-sqlite for the comparison
cells; smythe-only cells run without them):

    python benchmarks/run_durability_benchmark.py
    python benchmarks/run_durability_benchmark.py --quick
    python benchmarks/run_durability_benchmark.py --cell b

The worker subcommands (``smythe-worker`` / ``langgraph-worker``) are
internal; the orchestrator launches them via ``sys.executable``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import operator
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parents[1]))

PAYLOAD = ("durability benchmark payload text segment for realistic size. " * 33)[:2048]
RESULTS_DIR = Path(__file__).parent / "results"


def _append_call_event(
    path: Path,
    *,
    event: str,
    operation_id: str,
    attempt_id: str,
    durable: bool,
) -> None:
    """Append one complete JSONL event, optionally forcing it to stable storage."""

    payload = json.dumps(
        {
            "schema": "smythe.durability-call-event.v1",
            "event": event,
            "operation_id": operation_id,
            "attempt_id": attempt_id,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab", buffering=0) as handle:
        handle.write(payload)
        if durable:
            os.fsync(handle.fileno())


def _read_call_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    with open(path, "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"invalid call event at {path}:{line_number}"
                ) from exc
            required = {"schema", "event", "operation_id", "attempt_id"}
            if not isinstance(event, dict) or not required <= event.keys():
                raise RuntimeError(f"invalid call event at {path}:{line_number}")
            if event["schema"] != "smythe.durability-call-event.v1":
                raise RuntimeError(
                    f"unsupported call event schema at {path}:{line_number}"
                )
            if event["event"] not in {"dispatched", "completed"}:
                raise RuntimeError(f"invalid call event type at {path}:{line_number}")
            events.append(event)
    return events


def _call_event_summary(path: Path) -> dict:
    events = _read_call_events(path)
    dispatch_counts: dict[str, int] = {}
    completed_attempts: set[str] = set()
    attempt_operations: dict[str, str] = {}
    for event in events:
        attempt_id = event["attempt_id"]
        operation_id = event["operation_id"]
        if event["event"] == "dispatched":
            if attempt_id in attempt_operations:
                raise RuntimeError(f"attempt {attempt_id!r} was dispatched more than once")
            dispatch_counts[operation_id] = dispatch_counts.get(operation_id, 0) + 1
            attempt_operations[attempt_id] = operation_id
        else:
            if attempt_operations.get(attempt_id) != operation_id:
                raise RuntimeError(
                    f"completion for attempt {attempt_id!r} has no matching dispatch"
                )
            if attempt_id in completed_attempts:
                raise RuntimeError(f"attempt {attempt_id!r} completed more than once")
            completed_attempts.add(attempt_id)
    return {
        "events": len(events),
        "dispatches": sum(dispatch_counts.values()),
        "completions": len(completed_attempts),
        "unique_operation_ids": len(dispatch_counts),
        "duplicate_dispatches": sum(
            max(0, count - 1) for count in dispatch_counts.values()
        ),
        "inflight_attempts": len(set(attempt_operations) - completed_attempts),
        "dispatch_counts": dispatch_counts,
    }


def _source_identity() -> dict:
    repo = Path(__file__).resolve().parents[1]
    identity = {
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": "unavailable",
        "git_dirty": None,
    }
    try:
        base = ["git", "-c", f"safe.directory={repo.as_posix()}"]
        revision = subprocess.run(
            [*base, "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        status = subprocess.run(
            [*base, "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        identity["git_revision"] = revision.stdout.strip()
        identity["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


# --------------------------------------------------------------------------
# smythe worker
# --------------------------------------------------------------------------

def smythe_worker(args: argparse.Namespace) -> None:
    from smythe.checkpoint import FileCheckpointStore
    from smythe.graph import ExecutionGraph, Node, NodeStatus, Topology
    from smythe.provider import CompletionResult, Provider
    from smythe.swarm import Swarm

    class SimCallProvider(Provider):
        """Record dispatch exposure, sleep, record completion, return a result."""

        def __init__(self, latency_s: float, calls_log: Path) -> None:
            self._latency_s = latency_s
            self._calls_log = calls_log

        async def complete(self, system, prompt, model) -> CompletionResult:
            node_id = prompt.split("\n", 1)[0]  # label == node id here
            attempt_id = uuid4().hex
            _append_call_event(
                self._calls_log,
                event="dispatched",
                operation_id=node_id,
                attempt_id=attempt_id,
                durable=getattr(args, "durable_events", False),
            )
            await asyncio.sleep(self._latency_s)
            _append_call_event(
                self._calls_log,
                event="completed",
                operation_id=node_id,
                attempt_id=attempt_id,
                durable=getattr(args, "durable_events", False),
            )
            return CompletionResult(
                text=f"{node_id}:{PAYLOAD}", prompt_tokens=40, completion_tokens=60,
            )

    provider = SimCallProvider(args.latency_ms / 1000.0, Path(args.calls_log))
    store = None if args.ckpt_dir == "none" else FileCheckpointStore(args.ckpt_dir)

    t_build = time.perf_counter()
    nodes = [Node(label=f"n{i:04d}", id=f"n{i:04d}") for i in range(args.n)]
    graph = ExecutionGraph(topology=[Topology.BROADCAST_REDUCE], nodes=nodes)
    build_s = time.perf_counter() - t_build

    swarm = Swarm(
        provider=provider, parallel=True, max_concurrency=args.concurrency,
        artifact_dir=None, model="sim-model",
        checkpoint_store=store, checkpoint_every_n_nodes=1,
    )
    t0 = time.perf_counter()
    if args.mode == "run":
        result = swarm.execute(graph)
    else:
        ids = store.list_ids()
        if len(ids) != 1:
            raise SystemExit(f"expected exactly one checkpoint, found {ids}")
        result = swarm.resume(ids[0])
    wall = time.perf_counter() - t0

    completed = sum(
        1 for nd in result.graph.nodes if nd.status == NodeStatus.COMPLETED
    )
    print("RESULT " + json.dumps({
        "framework": "smythe", "mode": args.mode, "n": args.n,
        "completed": completed, "wall_s": round(wall, 4),
        "build_s": round(build_s, 4), "durability": "file/every=1",
    }))


# --------------------------------------------------------------------------
# LangGraph worker
# --------------------------------------------------------------------------

def langgraph_worker(args: argparse.Namespace) -> None:
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        results: Annotated[list, operator.add]

    calls_log = Path(args.calls_log)
    latency_s = args.latency_ms / 1000.0

    def make_node(node_id: str):
        async def node_fn(state: State):
            attempt_id = uuid4().hex
            _append_call_event(
                calls_log,
                event="dispatched",
                operation_id=node_id,
                attempt_id=attempt_id,
                durable=getattr(args, "durable_events", False),
            )
            await asyncio.sleep(latency_s)
            _append_call_event(
                calls_log,
                event="completed",
                operation_id=node_id,
                attempt_id=attempt_id,
                durable=getattr(args, "durable_events", False),
            )
            return {"results": [f"{node_id}:{PAYLOAD}"]}
        return node_fn

    async def main_async() -> None:
        t_build = time.perf_counter()
        g = StateGraph(State)
        for i in range(args.n):
            nid = f"n{i:04d}"
            g.add_node(nid, make_node(nid))
            g.add_edge(START, nid)
            g.add_edge(nid, END)
        config: dict = {"max_concurrency": args.concurrency}
        durability_used = "n/a"

        async def invoke(app, payload):
            nonlocal durability_used
            try:
                out = await app.ainvoke(payload, config=config, durability="sync")
                durability_used = "sync"
                return out
            except TypeError:  # older langgraph without the durability kwarg
                durability_used = "default"
                return await app.ainvoke(payload, config=config)

        if args.db == "none":
            app = g.compile()
            build_s = time.perf_counter() - t_build
            t0 = time.perf_counter()
            final = await app.ainvoke({"results": []}, config=config)
            wall = time.perf_counter() - t0
        else:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            config["configurable"] = {"thread_id": "bench"}
            async with AsyncSqliteSaver.from_conn_string(args.db) as saver:
                app = g.compile(checkpointer=saver)
                build_s = time.perf_counter() - t_build
                payload = {"results": []} if args.mode == "run" else None
                t0 = time.perf_counter()
                final = await invoke(app, payload)
                wall = time.perf_counter() - t0

        unique_done = len({r.split(":", 1)[0] for r in final["results"]})
        print("RESULT " + json.dumps({
            "framework": "langgraph", "mode": args.mode, "n": args.n,
            "completed": unique_done, "wall_s": round(wall, 4),
            "build_s": round(build_s, 4), "durability": durability_used,
        }))

    asyncio.run(main_async())


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def _parse_result(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no RESULT line in worker output:\n{stdout[-2000:]}")


def _run_to_completion(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed: {cmd}\n{proc.stderr[-3000:]}")
    return _parse_result(proc.stdout)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _count_call_events(path: Path, event: str) -> int:
    return sum(1 for item in _read_call_events(path) if item["event"] == event)


def _clean(*paths: Path) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def _worker_cmd(fw: str, *, n: int, latency_ms: float, concurrency: int,
                calls_log: Path, persist: Path | None, mode: str) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), f"{fw}-worker",
           "--n", str(n), "--latency-ms", str(latency_ms),
           "--concurrency", str(concurrency), "--calls-log", str(calls_log),
           "--mode", mode]
    if persist is not None:
        cmd.append("--durable-events")
    if fw == "smythe":
        cmd += ["--ckpt-dir", str(persist) if persist else "none"]
    else:
        cmd += ["--db", str(persist) if persist else "none"]
    return cmd


def kill_when_dispatches_reach(cmd: list[str], log: Path, threshold: int,
                               timeout_s: float = 120.0) -> dict:
    """Start ``cmd`` and hard-kill it at a durable dispatch threshold."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"worker exited (rc={proc.returncode}) before the kill "
                    f"threshold; dispatches={_count_call_events(log, 'dispatched')}")
            if _count_call_events(log, "dispatched") >= threshold:
                proc.kill()  # TerminateProcess / SIGKILL — a real crash
                proc.wait(timeout=30)
                return _call_event_summary(log)
            if time.time() - t0 > timeout_s:
                raise RuntimeError("timeout waiting for kill threshold")
            time.sleep(0.02)
    finally:
        if proc.poll() is None:
            proc.kill()


def kill_when_calls_reach(cmd: list[str], log: Path, threshold: int,
                          timeout_s: float = 120.0) -> int:
    """Compatibility wrapper returning the dispatch count at the kill point."""

    return kill_when_dispatches_reach(cmd, log, threshold, timeout_s)["dispatches"]


def cell_a_overhead(frameworks: list[str], sizes: tuple[int, ...],
                    reps: int, work_dir: Path) -> list[dict]:
    print("\n=== Cell A: fan-out scheduler overhead "
          "(25 ms calls, c=16, no persistence) ===")
    rows = []
    for n in sizes:
        ideal = (n / 16) * 0.025
        for fw in frameworks:
            walls, builds = [], []
            for rep in range(reps):
                log = work_dir / f"a_{fw}_{n}_{rep}.log"
                _clean(log)
                res = _run_to_completion(_worker_cmd(
                    fw, n=n, latency_ms=25, concurrency=16,
                    calls_log=log, persist=None, mode="run"))
                events = _call_event_summary(log)
                if (
                    res["completed"] != n
                    or events["dispatches"] != n
                    or events["completions"] != n
                    or events["unique_operation_ids"] != n
                    or events["duplicate_dispatches"] != 0
                ):
                    raise RuntimeError(f"incomplete run: {res}")
                walls.append(res["wall_s"])
                builds.append(res["build_s"])
                _clean(log)
            row = dict(
                framework=fw, n=n, ideal_s=round(ideal, 3),
                wall_mean_s=round(statistics.mean(walls), 3),
                wall_min_s=round(min(walls), 3),
                wall_max_s=round(max(walls), 3),
                overhead_per_node_ms=round(
                    (statistics.mean(walls) - ideal) / n * 1000, 3),
                build_mean_s=round(statistics.mean(builds), 3),
            )
            rows.append(row)
            print(f"  {fw:9s} N={n:5d} wall={row['wall_mean_s']:7.3f}s "
                  f"[{row['wall_min_s']:.3f}-{row['wall_max_s']:.3f}] "
                  f"(ideal {ideal:6.3f}s, {row['overhead_per_node_ms']:6.3f} "
                  f"ms/node) build={row['build_mean_s']:6.3f}s")
    return rows


def cell_b_durability(frameworks: list[str], n: int, kill_at: int,
                      reps: int, work_dir: Path) -> list[dict]:
    print(f"\n=== Cell B: kill-and-resume durability "
          f"(N={n}, 100 ms calls, c=8, hard kill at {kill_at} dispatches) ===")
    rows = []
    for fw in frameworks:
        for rep in range(reps):
            log = work_dir / f"b_{fw}_{rep}.log"
            persist = work_dir / (f"b_ckpt_{fw}_{rep}" if fw == "smythe"
                                  else f"b_{fw}_{rep}.sqlite")
            _clean(log, persist)
            kw = dict(n=n, latency_ms=100, concurrency=8, calls_log=log,
                      persist=persist)
            at_kill = kill_when_dispatches_reach(
                _worker_cmd(fw, mode="run", **kw), log, kill_at)
            res = _run_to_completion(_worker_cmd(fw, mode="resume", **kw))
            final = _call_event_summary(log)
            expected = {f"n{i:04d}" for i in range(n)}
            observed = set(final["dispatch_counts"])
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise RuntimeError(
                    f"dispatch inventory mismatch: missing={missing}, extra={extra}"
                )
            replayed = sorted(
                operation_id
                for operation_id, count in final["dispatch_counts"].items()
                if count > 1
            )
            row = dict(
                framework=fw,
                rep=rep,
                n=n,
                dispatches_at_kill=at_kill["dispatches"],
                completions_at_kill=at_kill["completions"],
                inflight_attempts_at_kill=at_kill["inflight_attempts"],
                total_dispatches=final["dispatches"],
                total_completions=final["completions"],
                duplicate_dispatches=final["duplicate_dispatches"],
                replayed_operation_ids=replayed,
                resume_completed=res["completed"],
                resume_wall_s=res["wall_s"], durability=res["durability"],
            )
            rows.append(row)
            print(f"  {fw:9s} rep{rep}: killed at {at_kill['dispatches']:3d} "
                  f"dispatches ({at_kill['inflight_attempts']} in flight), "
                  f"total {final['dispatches']:3d} -> repeated "
                  f"{row['duplicate_dispatches']:3d} "
                  f"| resume {res['completed']}/{n} in {res['wall_s']:6.2f}s "
                  f"({row['durability']})")
            _clean(log, persist)
    for fw in frameworks:
        dups = [r["duplicate_dispatches"] for r in rows if r["framework"] == fw]
        print(f"  {fw:9s} repeated dispatches: mean {statistics.mean(dups):.1f} "
              f"[{min(dups)}-{max(dups)}]")
    return rows


def orchestrate(args: argparse.Namespace) -> None:
    frameworks = ["smythe"]
    if importlib.util.find_spec("langgraph") is not None:
        frameworks.append("langgraph")
    else:
        print("langgraph not installed - running smythe cells only "
              "(pip install langgraph langgraph-checkpoint-sqlite to compare)")
    if "langgraph" in frameworks and importlib.util.find_spec(
            "langgraph.checkpoint.sqlite") is None and args.cell in ("all", "b"):
        print("langgraph-checkpoint-sqlite not installed - skipping "
              "LangGraph in cell B")
        cell_b_frameworks = [f for f in frameworks if f != "langgraph"]
    else:
        cell_b_frameworks = frameworks

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reps = 1 if args.quick else 3
    sizes = (64,) if args.quick else (64, 256, 1024)

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("smythe", "langgraph", "langgraph-checkpoint-sqlite"):
        try:
            from importlib.metadata import version
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = "n/a"
    if versions["smythe"] == "n/a":
        try:
            from smythe import __version__

            versions["smythe"] = f"{__version__} (source tree)"
        except Exception:
            pass

    out: dict = {
        "benchmark_schema": "smythe.durability-benchmark.v2",
        "evidence_semantics": (
            "duplicate_dispatches conservatively counts repeated operation IDs "
            "recorded before simulated remote work; it is exposure, not invoice proof"
        ),
        "protocol": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "versions": versions,
            "source": _source_identity(),
            "cell_a": {"latency_ms": 25, "concurrency": 16, "reps": reps,
                       "sizes": list(sizes), "persistence": "none"},
            "cell_b": {"n": 64, "latency_ms": 100, "concurrency": 8,
                       "kill_at_dispatches": 32, "reps": reps,
                       "call_event_schema": "smythe.durability-call-event.v1",
                       "call_event_fsync": True,
                       "smythe": "FileCheckpointStore, checkpoint_every_n_nodes=1",
                       "langgraph": "AsyncSqliteSaver, durability='sync'"},
        },
    }
    if args.cell in ("all", "a"):
        out["cell_a_overhead"] = cell_a_overhead(frameworks, sizes, reps, work_dir)
    if args.cell in ("all", "b"):
        out["cell_b_durability"] = cell_b_durability(
            cell_b_frameworks, 64, 32, reps, work_dir)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nrecords -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command")

    for name in ("smythe-worker", "langgraph-worker"):
        w = sub.add_parser(name)
        w.add_argument("--n", type=int, required=True)
        w.add_argument("--latency-ms", type=float, required=True)
        w.add_argument("--concurrency", type=int, required=True)
        w.add_argument("--calls-log", required=True)
        w.add_argument("--durable-events", action="store_true")
        w.add_argument("--mode", choices=["run", "resume"], default="run")
        w.add_argument("--ckpt-dir", default="none")
        w.add_argument("--db", default="none")

    run = sub.add_parser("run")
    run.add_argument("--cell", choices=["all", "a", "b"], default="all")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--work-dir", default="smythe_artifacts/durability-bench")
    run.add_argument("--out", default=None)

    argv = sys.argv[1:] or ["run"]
    if argv[0] not in ("smythe-worker", "langgraph-worker", "run"):
        argv = ["run", *argv]
    args = ap.parse_args(argv)

    if args.command == "smythe-worker":
        smythe_worker(args)
    elif args.command == "langgraph-worker":
        langgraph_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
