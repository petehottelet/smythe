"""Check usable public typing against a wheel installed outside the checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

POSITIVE = '''from typing import assert_type
from smythe import OfflineProvider, Swarm, SwarmResult, Task
from smythe.graph import ExecutionGraph

task = Task(goal="Summarize a local fixture", constraints=["Be concise"])
assert_type(task.goal, str)
assert_type(task.constraints, list[str])
swarm = Swarm(provider=OfflineProvider(), model="offline", max_concurrency=2)
assert_type(swarm.plan(task), ExecutionGraph)
assert_type(swarm.execute(task), SwarmResult)
result = swarm.execute(task)
assert_type(result.output, str)
assert_type(result.total_cost_usd, float)
assert_type(result.graph, ExecutionGraph)

async def consume() -> None:
    assert_type(await swarm.aplan(task), ExecutionGraph)
    assert_type(await swarm.execute_async(task), SwarmResult)
'''

NEGATIVE = '''from smythe import Swarm, Task
Task(goal=123)
Swarm(max_concurrency="eight")
Swarm().execute(123)
'''


def verify(python: Path) -> None:
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "MYPYPATH"):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="smythe-consumer-types-") as scratch:
        root = Path(scratch)
        subprocess.run(
            [str(python.resolve()), "-I", "-c",
             "import importlib.resources as r; assert r.files('smythe').joinpath('py.typed').is_file()"],
            cwd=root, env=environment, check=True,
        )
        (root / "mypy.ini").write_text("[mypy]\nstrict = True\nfollow_imports = silent\n", encoding="utf-8")
        for name, source in (("positive.py", POSITIVE), ("negative.py", NEGATIVE)):
            (root / name).write_text(source, encoding="utf-8")
            result = subprocess.run(
                [str(python.resolve()), "-I", "-m", "mypy", "--config-file", "mypy.ini", name],
                cwd=root, env=environment, capture_output=True, text=True, timeout=120,
            )
            if name == "positive.py":
                if result.returncode != 0:
                    raise ValueError("Public consumer types failed:\n" + result.stdout + result.stderr)
            else:
                errors = [line for line in result.stdout.splitlines() if ": error:" in line]
                if result.returncode != 1 or len(errors) != 3 or not all("[arg-type]" in line for line in errors):
                    raise ValueError("Invalid public calls were not rejected as expected:\n" + result.stdout + result.stderr)
    print(json.dumps({"public_type_assertions": "passed", "invalid_calls_rejected": 3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    verify(parser.parse_args().python)
