"""Explicit source-package profile; full-checkout coverage remains the default."""

import json
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
PROFILE = json.loads((TESTS / "distribution_profile.json").read_text(encoding="utf-8"))


def pytest_addoption(parser):
    parser.addoption(
        "--distribution", action="store_true", default=False,
        help="Run the explicit library-only profile from a source archive or sparse checkout",
    )


def pytest_configure(config):
    if config.getoption("distribution"):
        return
    missing = [name for name in PROFILE["required_repository_files"]
               if not (ROOT / name).is_file()]
    if missing:
        raise pytest.UsageError(
            "Full-checkout test materials are missing: " + ", ".join(missing)
            + ". Restore the files, or explicitly use --distribution for a source archive "
            "or sparse checkout."
        )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("distribution") and collection_path.parent == TESTS:
        if collection_path.name in PROFILE["excluded_modules"]:
            return True
    return None


def pytest_collection_modifyitems(config, items):
    if not config.getoption("distribution"):
        return
    kept, excluded = [], []
    for item in items:
        name = f"{item.path.name}::{item.name}"
        (excluded if name in PROFILE["excluded_cases"] else kept).append(item)
    items[:] = kept
    if excluded:
        config.hook.pytest_deselected(items=excluded)


def pytest_report_header(config):
    if config.getoption("distribution"):
        return (f"Distribution profile: {len(PROFILE['excluded_modules'])} repository-only "
                "modules excluded before import; see tests/distribution_profile.json")
    return "Full-checkout profile: all repository materials required"
