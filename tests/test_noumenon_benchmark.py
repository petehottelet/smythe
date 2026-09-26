from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_noumenon
from benchmarks.noumenon_assets import MAX_GLYPH_COUNT, get_glyph_specs, render_glyph_tile
from benchmarks.run_noumenon import (
    DEFAULT_LATENCY_S,
    _parse_concurrencies,
    _parse_partition,
    _resolve_output_paths,
    _write_json,
    build_graph,
    main,
    run_benchmark,
)
from smythe.graph import FailurePolicy, Topology

_SPECS = {spec.id: spec for spec in get_glyph_specs(MAX_GLYPH_COUNT)}
_GLYPH_ID = re.compile(r"CYBER_GLYPH_ID=(glyph-[0-9]{3})")


def _opaque_on_black(data: bytes) -> bytes:
    """Mimic an opaque RGBA container: the glyph flattened onto black."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as tile:
        rgba = tile.convert("RGBA")
    ground = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    ground.alpha_composite(rgba)
    buffer = io.BytesIO()
    ground.save(buffer, format="PNG")
    return buffer.getvalue()


def _install_fake_openai(monkeypatch, *, opaque: bool = False) -> list[dict]:
    """Serve procedural glyph PNGs through a fake Image API client; no network."""
    calls: list[dict] = []

    class Images:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            spec = _SPECS[_GLYPH_ID.search(kwargs["prompt"]).group(1)]
            data = render_glyph_tile(spec, size=256)
            if opaque:
                data = _opaque_on_black(data)
            return SimpleNamespace(
                data=[SimpleNamespace(b64_json=base64.b64encode(data).decode())],
                usage=SimpleNamespace(input_tokens=12, output_tokens=34),
            )

    class Client:
        images = Images()

        async def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **_: Client())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    return calls


def _live(tmp_path, **overrides):
    kwargs = {
        "live": True,
        "out": tmp_path / "artifacts",
        "glyph_count": 3,
        "concurrencies": (2,),
        "latency_s": 0,
        "model": "gpt-image-2.5-flare",
        "max_cost_per_call_usd": 0.01,
        "max_budget_usd": 0.05,
    }
    kwargs.update(overrides)
    return asyncio.run(run_benchmark(**kwargs))


def _offline(tmp_path, **overrides):
    kwargs = {
        "live": False,
        "out": tmp_path / "artifacts",
        "glyph_count": 4,
        "concurrencies": (1, 2),
        "latency_s": 0,
        "model": "unused",
        "max_cost_per_call_usd": None,
        "max_budget_usd": None,
    }
    kwargs.update(overrides)
    return asyncio.run(run_benchmark(**kwargs))


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


def test_transparent_graph_prompts_ask_for_transparency_not_black():
    default = build_graph(glyph_count=2, estimated_cost_per_call_usd=0)
    transparent = build_graph(glyph_count=2, estimated_cost_per_call_usd=0, background="transparent")

    assert all("solid black background" in node.label for node in default.nodes)
    for node in transparent.nodes:
        assert "fully transparent background" in node.label
        assert "black background" not in node.label
        assert f"CYBER_GLYPH_ID={node.id}" in node.label


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
        "smythe_artifacts/noumenon/partitions/256_offline_realistic"
    )
    assert results == Path(
        "benchmarks/results/noumenon_256_offline_realistic.json"
    )

    flagship_out, flagship_results, flagship_partition = _resolve_output_paths(
        mode="offline",
        glyph_count=192,
        partition=None,
        out=None,
        results=None,
    )
    assert flagship_partition is None
    assert flagship_out == Path("smythe_artifacts/noumenon/offline")
    assert flagship_results == Path("benchmarks/results/noumenon_offline.json")


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
    # The default lane makes no transparency claim and writes no SVGs.
    assert payload["protocol"]["background"] == "auto"
    assert payload["protocol"]["transparency_check"] is None
    assert payload["protocol"]["vectorize"] is False
    assert all(run["vectorization"] is None for run in payload["runs"])
    assert all("transparent_tiles" not in run["validation"] for run in payload["runs"])


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


def test_non_default_background_and_vectorize_use_isolated_outputs():
    out, results, partition = _resolve_output_paths(
        mode="live",
        glyph_count=192,
        partition=None,
        out=None,
        results=None,
        background="transparent",
        vectorize=True,
    )
    assert partition == "live_transparent_svg"
    assert out == Path("smythe_artifacts/noumenon/partitions/live_transparent_svg")
    assert results == Path("benchmarks/results/noumenon_live_transparent_svg.json")

    def partition_for(**kwargs):
        defaults = {"partition": None, "out": None, "results": None}
        return _resolve_output_paths(**defaults, **kwargs)[2]

    assert partition_for(mode="offline", glyph_count=8, background="transparent") == (
        "glyphs_8_offline_transparent"
    )
    assert partition_for(mode="offline", glyph_count=192, vectorize=True) == "offline_svg"
    assert partition_for(mode="live", glyph_count=192) is None


def test_offline_transparent_vectorized_run_records_alpha_and_svg_gates(tmp_path):
    payload = _offline(tmp_path, background="transparent", vectorize=True)

    assert payload["status"] == "passed", payload["errors"]
    assert payload["provider"]["background"] is None
    protocol = payload["protocol"]
    assert protocol["background"] == "transparent"
    assert protocol["normalization"] == "PNG 128x128 RGBA; provider alpha preserved"
    assert protocol["transparency_check"]["corner_size_px"] == 8
    assert protocol["vectorize"] is True
    assert protocol["vectorization"]["iou_threshold"] == 0.98
    for run in payload["runs"]:
        assert run["validation"]["transparent_tiles"] == 4
        vectorization = run["vectorization"]
        assert vectorization["passed"]
        assert vectorization["valid_svgs"] == vectorization["expected_svgs"] == 4
        assert vectorization["iou_min"] == vectorization["iou_max"] == 1.0
        assert vectorization["svg_bytes"] == sum(
            receipt["svg"]["byte_size"] for receipt in run["tile_receipts"]
        )
        assert run["vectorization_wall_s"] > 0
        assert run["end_to_end_wall_s"] == pytest.approx(
            run["generation_wall_s"] + run["validation_wall_s"] + run["vectorization_wall_s"],
            abs=2e-6,
        )
        for receipt in run["tile_receipts"]:
            assert receipt["transparency"]["passed"]
            assert receipt["transparency"]["reasons"] == []
            svg = Path(receipt["svg"]["path"])
            assert svg == Path(receipt["path"]).with_suffix(".svg")
            assert hashlib.sha256(svg.read_bytes()).hexdigest() == receipt["svg"]["sha256"]
            assert receipt["svg"]["mask_source"] == "alpha"


def test_invalid_svg_fails_the_run_and_is_recorded(tmp_path, monkeypatch):
    strict = functools.partial(run_noumenon.vectorize_tile, iou_threshold=1.01)
    monkeypatch.setattr(run_noumenon, "vectorize_tile", strict)

    payload = _offline(tmp_path, glyph_count=2, concurrencies=(1,), vectorize=True)

    assert payload["status"] == "failed"
    [run] = payload["runs"]
    assert run["validation"]["passed"]
    assert run["vectorization"]["valid_svgs"] == 0
    assert not run["vectorization"]["passed"]
    assert len(run["errors"]) == 2
    assert all(
        error["node_id"] == "__vectorization__" and "below the 1.01 threshold" in error["error"]
        for error in run["errors"]
    )


def test_live_transparent_lane_sends_background_and_gates_alpha_and_svgs(
    tmp_path, monkeypatch
):
    calls = _install_fake_openai(monkeypatch)

    payload = _live(tmp_path, background="transparent", vectorize=True)

    assert payload["status"] == "passed", payload["errors"]
    assert len(calls) == 3
    assert all(
        (call["model"], call["background"], call["output_format"])
        == ("gpt-image-2.5-flare", "transparent", "png")
        for call in calls
    )
    assert payload["provider"]["model"] == "gpt-image-2.5-flare"
    assert payload["provider"]["background"] == "transparent"
    assert payload["protocol"]["background"] == "transparent"
    [run] = payload["runs"]
    assert run["cost_usd"] == pytest.approx(0.03)
    assert run["validation"]["transparent_tiles"] == 3
    assert run["vectorization"]["valid_svgs"] == 3
    for receipt in run["tile_receipts"]:
        assert receipt["transparency"]["passed"]
        assert receipt["transparency"]["source_has_transparency"] is True
        assert receipt["svg"]["valid"] and receipt["svg"]["iou"] == 1.0


def test_live_transparent_lane_fails_closed_on_opaque_model_output(tmp_path, monkeypatch):
    _install_fake_openai(monkeypatch, opaque=True)

    payload = _live(tmp_path, background="transparent", vectorize=True)

    assert payload["status"] == "failed"
    [run] = payload["runs"]
    assert run["validation"]["valid_png_tiles"] == 3
    assert run["validation"]["transparent_tiles"] == 0
    assert all(not receipt["transparency"]["passed"] for receipt in run["tile_receipts"])
    assert all(receipt["transparency"]["corner_max_alpha"] == 255 for receipt in run["tile_receipts"])
    assert all("svg" not in receipt for receipt in run["tile_receipts"])
    assert run["vectorization"]["valid_svgs"] == 0
    assert any(
        "provider output has no transparent pixels" in error["error"]
        for error in payload["errors"]
    )


def test_default_live_lane_keeps_its_payload_and_makes_no_transparency_claim(
    tmp_path, monkeypatch
):
    calls = _install_fake_openai(monkeypatch, opaque=True)

    payload = _live(tmp_path, model="gpt-image-2")

    assert payload["status"] == "passed", payload["errors"]
    assert calls and all(call["model"] == "gpt-image-2" for call in calls)
    assert all("background" not in call for call in calls)
    assert payload["provider"]["background"] == "auto"
    assert payload["protocol"]["normalization"] == "PNG 128x128"
    assert payload["protocol"]["transparency_check"] is None
    [run] = payload["runs"]
    assert "transparent_tiles" not in run["validation"]
    assert run["vectorization"] is None and run["vectorization_wall_s"] is None
    assert all("transparency" not in receipt for receipt in run["tile_receipts"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "gpt-image-2"}, "gpt-image-2 does not support transparent backgrounds"),
        ({"model": "GPT-Image-2-2026-04-21"}, "does not support transparent backgrounds"),
        (
            {"live_provider": "gemini", "model": "gemini-2.5-flash-image"},
            "only with --live-provider openai",
        ),
    ],
)
def test_transparent_preflight_refuses_unsupported_pairings_before_dispatch(
    tmp_path, monkeypatch, overrides, message
):
    calls = _install_fake_openai(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _live(tmp_path, background="transparent", **overrides)
    assert calls == []


def test_unknown_background_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="--background must be one of"):
        _offline(tmp_path, background="opaque")


def test_cli_records_background_and_vectorize_settings(tmp_path, monkeypatch, capsys):
    results = tmp_path / "record.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_noumenon.py", "--background", "transparent", "--vectorize",
            "--glyphs", "2", "--concurrencies", "1", "--latency-s", "0",
            "--out", str(tmp_path / "artifacts"), "--results", str(results),
        ],
    )

    main()

    record = json.loads(results.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["partition"] == "glyphs_2_offline_transparent_svg"
    assert (record["protocol"]["background"], record["protocol"]["vectorize"]) == (
        "transparent",
        True,
    )
    assert "transparent=2/2 svg=2/2 iou=1.0000-1.0000" in capsys.readouterr().out


def test_cli_refuses_transparent_background_with_the_default_model(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_noumenon.py", "--live", "--background", "transparent",
            "--max-cost-per-call-usd", "0.01", "--max-budget-usd", "2",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    assert "gpt-image-2 does not support transparent backgrounds" in capsys.readouterr().err
