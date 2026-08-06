from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from benchmarks.run_glyph_screensaver import (
    DEFAULT_LATENCY_S,
    _parse_concurrencies,
    _parse_partition,
    _resolve_output_paths,
    _write_json,
    build_graph,
    run_benchmark,
)
from smythe.graph import FailurePolicy, Topology


def test_default_latency_models_remote_work_not_local_journaling():
    assert DEFAULT_LATENCY_S == 0.25


def test_build_graph_creates_one_independent_bounded_node_per_glyph():
    graph = build_graph(glyph_count=4, estimated_cost_per_call_usd=0.125)

    assert graph.topology == [Topology.BROADCAST_REDUCE]
    assert [node.id for node in graph.nodes] == [
        "glyph-000",
        "glyph-001",
        "glyph-002",
        "glyph-003",
    ]
    assert all(not node.depends_on for node in graph.nodes)
    assert all("CYBER_GLYPH_ID=" in node.label for node in graph.nodes)
    assert all(node.failure_policy is FailurePolicy.HALT for node in graph.nodes)
    assert all(node.max_retries == 0 for node in graph.nodes)
    assert all(node.metadata["estimated_cost_usd"] == 0.125 for node in graph.nodes)


def test_build_graph_supports_256_unique_glyph_nodes():
    graph = build_graph(glyph_count=256, estimated_cost_per_call_usd=0)

    assert len(graph.nodes) == 256
    assert graph.nodes[0].id == "glyph-000"
    assert graph.nodes[-1].id == "glyph-255"
    assert len({node.id for node in graph.nodes}) == 256


def test_parse_concurrencies_is_strict():
    assert _parse_concurrencies("1, 4,8") == (1, 4, 8)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_concurrencies("1,0")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_concurrencies("1,1")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_concurrencies("fast")


def test_non_flagship_output_defaults_are_partitioned():
    out, results, partition = _resolve_output_paths(
        mode="offline",
        glyph_count=256,
        partition="256_offline_realistic",
        out=None,
        results=None,
    )

    assert partition == "256_offline_realistic"
    assert out == Path(
        "smythe_artifacts/glyph_screensaver/partitions/256_offline_realistic"
    )
    assert results == Path(
        "benchmarks/results/glyph_screensaver_256_offline_realistic.json"
    )

    flagship_out, flagship_results, flagship_partition = _resolve_output_paths(
        mode="offline",
        glyph_count=192,
        partition=None,
        out=None,
        results=None,
    )
    assert flagship_partition is None
    assert flagship_out == Path("smythe_artifacts/glyph_screensaver/offline")
    assert flagship_results == Path("benchmarks/results/glyph_screensaver_offline.json")


def test_partition_name_rejects_path_traversal():
    assert _parse_partition("256_offline_realistic") == "256_offline_realistic"
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_partition("../offline")


def test_offline_run_benchmark_smoke_is_zero_cost_and_objective(tmp_path):
    payload = asyncio.run(
        run_benchmark(
            live=False,
            out=tmp_path / "artifacts",
            glyph_count=4,
            concurrencies=(1, 2),
            latency_s=0.001,
            model="unused",
            max_cost_per_call_usd=None,
            max_budget_usd=None,
        )
    )

    assert payload["status"] == "passed"
    assert payload["mode"] == "offline"
    assert payload["total_recorded_cost_usd"] == 0
    assert payload["assembly"] is None
    assert len(payload["runs"]) == 2
    assert all(run["completed_nodes"] == 4 for run in payload["runs"])
    assert all(run["validation"]["passed"] for run in payload["runs"])
    assert all(run["validation"]["unique_tile_hashes"] == 4 for run in payload["runs"])
    assert payload["runs"][0]["speedup_vs_concurrency_1"] == 1.0


def test_live_preflight_never_falls_back_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no offline fallback"):
        asyncio.run(
            run_benchmark(
                live=True,
                out=tmp_path,
                glyph_count=1,
                concurrencies=(1,),
                latency_s=0,
                model="gpt-image-2",
                max_cost_per_call_usd=0.01,
                max_budget_usd=0.01,
            )
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_live_preflight_rejects_non_finite_budget_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    with pytest.raises(ValueError, match="positive --max-cost-per-call-usd"):
        asyncio.run(
            run_benchmark(
                live=True,
                out=tmp_path,
                glyph_count=1,
                concurrencies=(1,),
                latency_s=0,
                model="gpt-image-1.5",
                max_cost_per_call_usd=value,
                max_budget_usd=1,
            )
        )


def test_json_record_write_is_atomic_and_utf8(tmp_path):
    path = tmp_path / "record.json"
    _write_json(path, {"label": "128x128"})

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert "128x128" in path.read_text(encoding="utf-8")
    assert not list(Path(tmp_path).glob("*.tmp"))
