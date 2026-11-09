"""Evidence gates for the optional native preview; no Rust installation needed."""
from copy import deepcopy
import json
import subprocess

import pytest

from benchmarks.run_rust_noumenon import review
from benchmarks import run_rust_noumenon as benchmark


def record():
    return {"plan": {"sizes": [4], "workers": [1], "repeats": 1}, "trials": [
        {"count": 4, "workers": 1, "backend": backend, "repeat": 1,
         "compilation_wall_s": .1, "provider_calls": 4, "source_stable": True,
         "svg_sha256": [str(i) for i in range(4)]}
        for backend in ("rust", "python-thread", "python-process")]}


def test_all_backends_are_required_before_summarizing():
    assert len(review(record())) == 3
    damaged = record()
    damaged["trials"].pop()
    with pytest.raises(ValueError, match="Missing"):
        review(damaged)


@pytest.mark.parametrize("damage", ["duplicate", "hash", "count", "clock", "source", "width"])
def test_comparison_refuses_misleading_evidence(damage):
    damaged = deepcopy(record())
    trial = damaged["trials"][0]
    if damage == "duplicate":
        damaged["trials"].append(deepcopy(trial))
    elif damage == "hash":
        trial["svg_sha256"][0] = "changed"
    elif damage == "count":
        trial["provider_calls"] = 3
    elif damage == "clock":
        trial["compilation_wall_s"] = float("nan")
    elif damage == "source":
        trial["source_stable"] = False
    else:
        trial["svg_sha256"].pop()
    with pytest.raises(ValueError):
        review(damaged)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("protocol", ["noumenon-rust-v1", "noumenon-rust-journal-v1", None])
def test_historical_comparison_refuses_a_different_timing_contract(tmp_path, monkeypatch, resume, protocol):
    source = tmp_path / "source.rs"
    source.write_bytes(b"fixture")
    receipt = {
        "protocol": protocol,
        "optimized": True,
        "implementation_sources": {"source.rs": benchmark.digest(b"fixture")},
    }
    report = {"saved_report": receipt, "provider_calls": 0} if resume else receipt
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(
        benchmark.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, json.dumps(report)),
    )
    if protocol == "noumenon-rust-v1":
        assert benchmark.native(tmp_path / "smythe", 4, 1, tmp_path / "out", resume=resume)
    else:
        with pytest.raises(ValueError, match="historical comparison"):
            benchmark.native(tmp_path / "smythe", 4, 1, tmp_path / "out", resume=resume)
