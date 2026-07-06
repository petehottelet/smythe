"""Smoke tests: every example must run offline, end to end, exit 0.

API-key env vars are stripped so the examples exercise the
OfflineProvider path — CI needs no credentials and spends no tokens.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("0*.py"))

# Env-gated examples print instructions and exit 0 when credentials are
# absent; they don't produce the offline-run marker.
GATED = {"06_mcp_github.py", "07_mcp_saas.py"}


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "GITHUB_PERSONAL_ACCESS_TOKEN", "SMYTHE_MCP_URL",
    ):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs_offline(script):
    proc = subprocess.run(
        [sys.executable, str(script)],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    if script.name not in GATED:
        assert "OfflineProvider" in proc.stdout or "offline" in proc.stdout


def test_examples_were_discovered():
    assert len(EXAMPLE_SCRIPTS) >= 4
