"""The memory A/B harness must run offline and prove its own wiring."""

import json
import subprocess
import sys
from pathlib import Path

from test_examples_smoke import _offline_env

BENCH_DIR = Path(__file__).parents[1] / "benchmarks"


def test_memory_ab_offline_mechanics(tmp_path):
    out = tmp_path / "memab.json"
    proc = subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_memory_ab.py"),
         "--offline", "--out", str(out)],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"memory A/B failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    records = json.loads(out.read_text(encoding="utf-8"))
    assert len(records) == 10  # 2 conditions x 5 positions

    off = [r for r in records if r["condition"] == "memory_off"]
    on = [r for r in records if r["condition"] == "memory_on"]

    # Without memory nothing is ever recalled.
    assert not any(r["history_recalled"] for r in off)
    # With memory, position 0 has nothing to recall; later positions do —
    # the goals share enough keywords for PlannerMemory's overlap recall.
    on_by_pos = {r["position"]: r for r in on}
    assert on_by_pos[0]["history_recalled"] is False
    assert all(on_by_pos[p]["history_recalled"] for p in range(1, 5))
