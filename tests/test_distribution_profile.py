"""A missing repository must not silently pass its normal test suite."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).parent


@pytest.fixture
def profile_project(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_bytes((SOURCE / "conftest.py").read_bytes())
    profile = {
        "required_repository_files": ["benchmarks/fixture.json"],
        "excluded_modules": {"test_repo.py": "Needs absent benchmark code"},
        "excluded_cases": {"test_core.py::test_example": "Needs omitted example"},
    }
    (tests / "distribution_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (tests / "test_repo.py").write_text("import intentionally_missing_benchmark\n", encoding="utf-8")
    (tests / "test_core.py").write_text(
        "def test_core(): assert True\ndef test_example(): assert False\n", encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return tmp_path


def invoke(root, *args):
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-I", "-m", "pytest", "-q", *args], cwd=root,
        env=environment, capture_output=True, text=True, timeout=30,
    )


def test_distribution_excludes_before_import_and_keeps_core_cases(profile_project):
    result = invoke(profile_project, "--distribution")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout


def test_default_profile_fails_when_required_fixture_disappears(profile_project):
    result = invoke(profile_project)
    assert result.returncode == 4
    assert "Full-checkout test materials are missing" in result.stderr


def test_distribution_does_not_hide_an_unlisted_missing_import(profile_project):
    (profile_project / "tests/test_new.py").write_text("import unknown_missing_module\n")
    result = invoke(profile_project, "--distribution")
    assert result.returncode == 2
    assert "unknown_missing_module" in result.stdout


def test_manifest_names_existing_tests_and_explains_every_exclusion():
    profile = json.loads((SOURCE / "distribution_profile.json").read_text(encoding="utf-8"))
    for name, reason in profile["excluded_modules"].items():
        assert (SOURCE / name).is_file(), name
        assert reason.strip()
    for node, reason in profile["excluded_cases"].items():
        filename, name = node.split("::")
        assert f"def {name}(" in (SOURCE / filename).read_text(encoding="utf-8")
        assert reason.strip()
