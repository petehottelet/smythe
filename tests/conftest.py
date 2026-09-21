"""Explicit source-package profile; full-checkout coverage remains the default."""

import ipaddress
import json
from pathlib import Path
import socket
import sys

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
PROFILE = json.loads((TESTS / "distribution_profile.json").read_text(encoding="utf-8"))


def _offline_network(event, args):
    """Reject external Python socket traffic, including during test collection.

    Local SDK wire fixtures and MCP IPC remain available. This is a test
    guardrail, not an OS sandbox: separately launched processes need their own
    offline fixtures. No credential-dependent live tests belong in this suite.
    """
    if event == "socket.getaddrinfo":
        host = args[0]
    elif event in {"socket.connect", "socket.sendto"}:
        sock, address = args
        if sock.family not in {socket.AF_INET, socket.AF_INET6}:
            return
        host = address[0]
    else:
        return
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host in {None, "localhost"}:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise RuntimeError("Offline test suite blocked external network dispatch")


sys.addaudithook(_offline_network)


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
