"""The README's offline quickstart runs as written after `pip install smythe`."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _offline_quickstart() -> str:
    readme = ROOT.joinpath("README.md").read_text(encoding="utf-8")
    section = readme.split("## Quickstart", 1)[1].split("### With a real model", 1)[0]
    # The first install line is the bare package: no provider extra, no pin.
    assert "```bash\npip install smythe\n```" in section
    [code] = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    return code


def test_offline_quickstart_runs_without_extras_or_api_keys(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    # The graph printout uses box-drawing characters; decode them the same
    # way on every platform.
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", _offline_quickstart()],
        cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'TaskGraph(topology="fork-join")'
    assert "├─ fork (parallel):" in lines
    assert lines[-2] == "offline: Recommend one database"
    assert lines[-1] == "True"
    assert tmp_path.joinpath("smythe-runs.db").is_file()
