"""The benchmark harness must run offline and match its committed sample.

If a change legitimately moves the offline results, regenerate with:

    python benchmarks/run_benchmarks.py --out results/offline_sample.json
"""

import json
import subprocess
import sys
from pathlib import Path

from test_examples_smoke import _offline_env

BENCH_DIR = Path(__file__).parents[1] / "benchmarks"
SAMPLE = BENCH_DIR / "results" / "offline_sample.json"


def run_benchmarks(*extra_args):
    return subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_benchmarks.py"), *extra_args],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_offline_run_matches_committed_sample(tmp_path):
    out = tmp_path / "results.json"
    proc = run_benchmarks("--out", str(out))
    assert proc.returncode == 0, (
        f"harness failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    fresh = out.read_text(encoding="utf-8")
    committed = SAMPLE.read_text(encoding="utf-8")
    assert fresh == committed, (
        "offline benchmark results drifted from results/offline_sample.json. "
        "If intentional, regenerate with: python benchmarks/run_benchmarks.py "
        "--out results/offline_sample.json"
    )


def test_offline_run_covers_all_baselines_and_tasks(tmp_path):
    out = tmp_path / "results.json"
    proc = run_benchmarks("--out", str(out))
    assert proc.returncode == 0
    records = json.loads(out.read_text(encoding="utf-8"))

    combos = {(r["task"], r["baseline"]) for r in records}
    tasks = {r["task"] for r in records}
    baselines = {r["baseline"] for r in records}
    assert baselines == {"single_agent", "fixed_pipeline", "smythe_dynamic"}
    assert len(tasks) >= 2
    assert len(combos) == len(tasks) * len(baselines)

    for r in records:
        assert r["offline"] is True
        assert r["quality"] is None
        assert r["wall_ms"] is None
        assert r["cost_usd"] > 0
        assert r["nodes"] >= 1
        assert r["output"]

    # The baselines must actually differ in shape, or the comparison is fake.
    by_baseline = {r["baseline"]: r for r in records if r["task"] == sorted(tasks)[0]}
    assert by_baseline["single_agent"]["nodes"] == 1
    assert by_baseline["fixed_pipeline"]["topology"] == "serial"
    assert by_baseline["smythe_dynamic"]["topology"] != "serial"


def test_task_filter_and_unknown_task():
    proc = run_benchmarks("--task", "research-memo", "--baseline", "single_agent")
    assert proc.returncode == 0
    assert "research-memo / single_agent" in proc.stdout
    assert "acquisition-diligence" not in proc.stdout

    proc = run_benchmarks("--task", "does-not-exist")
    assert proc.returncode == 2
