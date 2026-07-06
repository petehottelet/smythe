"""The flagship demo's committed artifacts must match a fresh fixture run.

Regenerates graph.mmd, trace.json, and memo.md into a temp directory
and compares them byte-for-byte against examples/acquisition_diligence/
expected/. If a change moves the demo's output, regenerate the
artifacts intentionally:

    python examples/acquisition_diligence/run.py --write-artifacts expected
"""

import subprocess
import sys
from pathlib import Path

from test_examples_smoke import _offline_env

DEMO_DIR = Path(__file__).parents[1] / "examples" / "acquisition_diligence"
EXPECTED_DIR = DEMO_DIR / "expected"
ARTIFACTS = ("graph.mmd", "trace.json", "memo.md")


def test_fixture_run_reproduces_committed_artifacts(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py"), "--write-artifacts", str(tmp_path)],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"demo failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    for name in ARTIFACTS:
        fresh = (tmp_path / name).read_text(encoding="utf-8")
        committed = (EXPECTED_DIR / name).read_text(encoding="utf-8")
        assert fresh == committed, (
            f"{name} drifted from the committed artifact. If intentional, "
            f"regenerate with: python examples/acquisition_diligence/run.py "
            f"--write-artifacts expected"
        )


def test_demo_prints_plan_and_memo():
    proc = subprocess.run(
        [sys.executable, str(DEMO_DIR / "run.py")],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert 'TaskGraph(topology="fork-join' in proc.stdout
    assert "adversarial" in proc.stdout
    assert "RECOMMENDATION" in proc.stdout
