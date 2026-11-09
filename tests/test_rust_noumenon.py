"""Evidence gates for the optional native preview; no Rust installation needed."""
from copy import deepcopy

import pytest

from benchmarks.run_rust_noumenon import review


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
