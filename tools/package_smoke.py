"""Verify source rebuilds and an installed wheel in a clean external environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from distribution import ROOT, audit, compare_wheels


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def verify(dist: Path, work: Path) -> None:
    dist, work = dist.resolve(), work.resolve()
    if work.exists():
        raise ValueError("Use a fresh smoke-test directory")
    work.mkdir(parents=True)
    wheels, sources = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and source archive")
    report = audit(sources[0], wheels[0])
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "MYPYPATH", "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        environment.pop(key, None)
    environment["PYTHONIOENCODING"] = "utf-8"
    with tarfile.open(sources[0]) as archive:
        archive.extractall(work / "source", filter="data")
    source = next((work / "source").iterdir())
    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "direct"), str(ROOT)],
        cwd=work, environment=environment)
    compare_wheels(wheels[0], next((work / "direct").glob("*.whl")))
    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "rebuilt"), str(source)],
        cwd=work, environment=environment)
    compare_wheels(wheels[0], next((work / "rebuilt").glob("*.whl")))
    run([sys.executable, "-m", "venv", str(work / "venv")], cwd=work, environment=environment)
    executable = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = str(executable)
    run([python, "-m", "pip", "install", str(wheels[0]) + "[dev]", "hatchling"],
        cwd=work, environment=environment)
    run([python, "-I", "-c",
         "import smythe, smythe.assets, smythe.jobs, smythe.optimize; "
         "from pathlib import Path; import sys; "
         "assert Path(smythe.__file__).is_relative_to(Path(sys.prefix)); "
         "from smythe import Swarm, Task, OfflineProvider, SimpleArchitect; "
         "result = Swarm(architect=SimpleArchitect(), provider=OfflineProvider(), model='offline').execute(Task(goal='Offline package check')); "
         "assert result.output and len(result.graph.nodes) == 1; print('Installed offline graph passed')"],
        cwd=work, environment=environment)
    run([python, "-I", "-m", "smythe.cli", "--help"], cwd=work, environment=environment)
    run([python, "-I", "-m", "smythe.cli", "jobs", "schema", "--json"], cwd=work, environment=environment)
    run([python, "-I", str(ROOT / "tools/check_consumer_types.py"), "--python", python],
        cwd=work, environment=environment)
    run([python, "-m", "pytest", "tests", "--distribution", "-q"], cwd=source, environment=environment)
    report["checks"] = {"checkout_and_source_wheels_identical": True,
                        "installed_offline_graph": True, "installed_public_typing": True,
                        "source_distribution_tests": True}
    (work / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    verify(args.dist, args.work)
