"""Regression tests for the isolated, real-work SVG workflow benchmark."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
import pytest

from benchmarks import run_svg_glyph_benchmark as bench


@pytest.fixture
def glyph_api(monkeypatch, tmp_path):
    calls = []
    lock = threading.Lock()
    source = tmp_path / "fixture_generator.py"
    source.write_text("# Test-only independent SVG generator fixture\n", encoding="utf-8")

    def generate(index, attempt=0):
        with lock:
            calls.append((index, attempt))
        return {"glyph_id": f"original-{index}", "family": "test", "profile": "test",
                "seed": 1700 + index + attempt,
                "authoring_parameters": {"fixedweight": 18, "structure": index, "seed_offset": 0},
                "recipe": "test rectangle", "attempt": attempt, "version": "fixture-v1", "authoring_weight": 18,
                "svg": f'<svg viewBox="0 0 128 128"><rect x="20" y="20" width="{20 + index * 4}" height="80"/></svg>'}

    def validate(svg):
        root = ET.fromstring(svg)
        return {"passed": root.tag == "svg", "errors": []}

    def render(svg, size=128):
        rectangle = ET.fromstring(svg).find("rect")
        image = Image.new("RGBA", (size, size), "white")
        scale = size / 128
        x, y, w, h = (float(rectangle.attrib[key]) for key in ("x", "y", "width", "height"))
        ImageDraw.Draw(image).rectangle((x * scale, y * scale, (x + w) * scale, (y + h) * scale), fill="black")
        return image

    api = SimpleNamespace(generate_glyph=generate, validate_svg=validate,
                          render_svg=render, __file__=str(source), calls=calls)
    monkeypatch.setattr(bench, "_api", lambda: api)
    measurement = SimpleNamespace(
        __file__=str(source),
        measure_glyph=lambda image: {"blank": False, "ink_pixels": sum(max(p[:3]) < 128 for p in image.getdata())},
        find_near_matches=lambda images, ids: {"near_matches": []},
        evaluate_catalog=lambda records, reference, distinctness, optical_review=None: {
            "accepted": len(records) == 192 and bool(optical_review and optical_review.get("passed")),
            "passes_numeric_shape_gates": True, "failed_gates": [],
            "status": "complete" if len(records) == 192 else "calibration",
            "optical_review_required": not bool(optical_review and optical_review.get("passed")),
            "optical_review": optical_review,
        },
    )
    monkeypatch.setattr(bench, "_measurement_api", lambda: measurement)
    api.measurement = measurement
    return api


def test_fresh_generation_runs_through_smythe_and_counts_whole_workflow(glyph_api, tmp_path):
    result = bench.run_benchmark(out=tmp_path / "campaign", glyph_count=4, concurrencies=(1, 2),
                                 executors=("thread",), repeats=3, render_sizes=(64,), worker_cap=2)
    assert len(glyph_api.calls) == 4 * 2 * 3
    assert all(attempt == 0 for _, attempt in glyph_api.calls)
    assert result["status"] == "passed"
    assert result["claimable"] is True  # Timing eligibility does not claim style achievement.
    assert result["readme_promotion_eligible"] is False
    assert result["catalog_style_accepted"] is False
    assert result["api_calls"] == result["api_cost_usd"] == 0
    assert result["known_measurement_defects"] == []
    for run in result["runs"]:
        assert run["provider_calls"] == run["successful_provider_calls"] == run["completed_nodes"] == 4
        assert run["failed_provider_calls"] == 0
        assert run["smythe_recorded_cost_usd"] == 0
        assert run["max_in_flight_provider_calls"] <= run["concurrency"]
        assert run["workers"] <= min(2, run["concurrency"])
        stages = sum(run[key] for key in ("setup_wall_s", "generation_wall_s", "validation_wall_s",
                                          "assembly_wall_s", "worker_shutdown_wall_s"))
        assert run["end_to_end_wall_s"] >= stages
        assert run["assembly"]["output_bytes"] > 0
        assert run["worker_cpu"] is None  # Thread process-time cannot be summed per call.
        assert all("worker_cpu_s" not in r for r in run["calls"])
        assert len({r["pixel_sha256"] for r in run["glyphs"]}) == 4
        catalog = json.loads(bench.Path(run["assembly"]["catalog"]).read_text(encoding="utf-8"))
        validation = json.loads(bench.Path(run["assembly"]["validation"]).read_text(encoding="utf-8"))
        for records in (run["glyphs"], catalog["glyphs"], validation["glyphs"]):
            for record in records:
                assert record["authoring_parameters"] == {"fixedweight": 18, "structure": record["index"], "seed_offset": 0}
                assert record["recipe"] == "test rectangle"
                assert record["attempt"] == 0
                assert record["version"] == "fixture-v1"
                assert record["authoring_weight"] == 18
    assert result["fastest_median_workflow"]["end_to_end_wall_s"]["median"] > 0
    assert len(result["rendering"]["samples"]) == 3
    assert "not animation FPS" in result["rendering"]["scope"]
    on_disk = json.loads((tmp_path / "campaign/results.json").read_text(encoding="utf-8"))
    assert on_disk["source_sha256"]
    assert on_disk["protocol"]["cached_generation_outputs"] is False


def test_failed_generation_is_counted_and_disqualifies_campaign(glyph_api, tmp_path):
    generate = glyph_api.generate_glyph

    def fail_one(index, attempt=0):
        if index == 1:
            raise RuntimeError("deliberate generation failure")
        return generate(index, attempt)

    glyph_api.generate_glyph = fail_one
    result = bench.run_benchmark(out=tmp_path / "failed", glyph_count=4, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=())
    run = result["runs"][0]
    assert result["status"] == "failed"
    assert result["claimable"] is False
    assert result["evidence_status"] == "diagnostic"
    assert run["provider_calls"] == 4
    assert run["successful_provider_calls"] == 3
    assert run["failed_provider_calls"] == 1
    assert run["assembly"] is None
    assert any("deliberate generation failure" in e for e in run["errors"])


def test_rejected_validation_and_blank_raster_fail_closed(glyph_api, tmp_path):
    glyph_api.validate_svg = lambda svg: {"passed": True, "errors": ["rejected"]}
    result = bench.run_benchmark(out=tmp_path / "invalid", glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=())
    assert result["runs"][0]["valid_glyphs"] == 0
    assert result["claimable"] is False
    glyph_api.validate_svg = lambda svg: {"passed": True, "errors": []}
    glyph_api.render_svg = lambda svg, size=128: Image.new("RGBA", (size, size), "white")
    result = bench.run_benchmark(out=tmp_path / "blank", glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=())
    assert result["status"] == "failed"
    assert any("blank" in e for e in result["errors"])


def test_distinct_metadata_cannot_hide_duplicate_shapes(glyph_api, tmp_path):
    original = glyph_api.generate_glyph

    def duplicate(index, attempt=0):
        return original(index, attempt) | {"svg": original(0, attempt)["svg"]}

    glyph_api.generate_glyph = duplicate
    result = bench.run_benchmark(out=tmp_path / "duplicate", glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=())
    assert result["claimable"] is False
    assert "duplicate pixel_sha256" in result["runs"][0]["errors"]


def test_output_guard_precedes_generation_and_requires_explicit_overwrite(glyph_api, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "results.json").write_text('{"historical": true}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        bench.run_benchmark(out=out, glyph_count=2, concurrencies=(1,), executors=("thread",), repeats=1)
    assert glyph_api.calls == []
    assert json.loads((out / "results.json").read_text()) == {"historical": True}
    result = bench.run_benchmark(out=out, glyph_count=2, concurrencies=(1,), executors=("thread",),
                                 repeats=1, render_sizes=(), overwrite=True)
    assert result["status"] == "passed"
    assert result["claimable"] is False  # One repeat is a smoke check, not a headline.


@pytest.mark.parametrize("options", [
    {"glyph_count": True}, {"glyph_count": 0}, {"repeats": 0}, {"worker_cap": 0},
    {"concurrencies": (4,)}, {"concurrencies": (1, 1)}, {"concurrencies": (1, False)},
    {"executors": ("remote",)}, {"render_sizes": (float("nan"),)},
])
def test_invalid_protocol_arguments_fail_before_work(glyph_api, tmp_path, options):
    with pytest.raises(ValueError):
        bench.run_benchmark(out=tmp_path / "invalid-args", **options)
    assert glyph_api.calls == []


def test_mismatched_repeat_hashes_are_measurement_defects():
    def run(digest):
        return {"executor": "thread", "concurrency": 1, "workers": 1, "status": "passed",
                "glyphs": [{"index": 0, "svg_sha256": digest, "pixel_sha256": digest}],
                "generation_wall_s": .1, "validation_wall_s": .2,
                "assembly_wall_s": .3, "end_to_end_wall_s": .7}

    summaries, fastest, defects = bench.summarize([run("a"), run("b")], repeats=2)
    assert defects == ["Output hashes differ across repeated/concurrency runs"]
    assert summaries[0]["end_to_end_wall_s"]["median"] == .7
    assert fastest is not None  # Parent claim eligibility additionally requires no defects.


def test_optical_review_is_bound_to_exact_generated_svg_hashes(glyph_api, tmp_path):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"passed": True, "glyph_svg_sha256": {"wrong": "digest"}}), encoding="utf-8")
    result = bench.run_benchmark(out=tmp_path / "bound-review", glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=(), optical_review_path=review)
    run = result["runs"][0]
    assert run["style_acceptance"]["optical_review_required"] is True
    report_path = bench.Path(run["assembly"]["validation"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["catalog_style"]["optical_review"]["passed"] is False
    assert "do not match" in report["catalog_style"]["optical_review"]["binding_error"]


def test_style_gaps_keep_valid_workflow_diagnostic(glyph_api, tmp_path):
    glyph_api.measurement.evaluate_catalog = lambda *a, **kw: {
        "accepted": False, "passes_numeric_shape_gates": False,
        "failed_gates": ["classic.stroke_width"], "status": "calibration",
        "optical_review_required": True,
    }
    result = bench.run_benchmark(out=tmp_path / "style-gaps", glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=1, render_sizes=())
    assert result["status"] == "passed"
    assert result["claimable"] is False
    assert result["runs"][0]["style_acceptance"]["failed_gates"] == ["classic.stroke_width"]
    assert result["runs"][0]["assembly"] is not None

def test_spawned_workers_execute_real_original_generator(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "orient_polygons"):
        pytest.skip("Original SVG generation requires Shapely2.1 or newer")
    result = bench.run_benchmark(out=tmp_path / "real-process", glyph_count=2,
                                 concurrencies=(1, 2), executors=("process",), repeats=1,
                                 worker_cap=2, render_sizes=(64,))
    assert result["status"] == "passed"
    assert result["evidence_status"] == "diagnostic"
    for run in result["runs"]:
        assert run["completed_nodes"] == run["valid_glyphs"] == 2
        assert run["workers_observed"] >= 1
        assert run["worker_cpu"]["generation_s"] > 0
        assert all(call["worker_cpu_basis"] == "process" for call in run["calls"])
        assert run["style_acceptance"]["status"] == "calibration"


@pytest.mark.parametrize("alpha", [0, 254])
def test_transparent_rgb_ink_is_not_accepted_as_a_visible_glyph(glyph_api, alpha):
    def invisible(svg, size=128):
        image = Image.new("RGBA", (size, size), (0, 0, 0, alpha))
        image.putpixel((0, 0), (255, 255, 255, alpha))
        return image

    glyph_api.render_svg = invisible
    receipt = bench._validate_work(0, json.dumps(glyph_api.generate_glyph(0)))
    assert receipt["status"] == "failed"
    assert "opaque" in receipt["error"]


def test_measurement_blank_flag_rejects_an_otherwise_nonblank_raster(glyph_api):
    glyph_api.measurement.measure_glyph = lambda image: {"blank": True}
    receipt = bench._validate_work(0, json.dumps(glyph_api.generate_glyph(0)))
    assert receipt["status"] == "failed"
    assert "blank" in receipt["error"]


@pytest.mark.parametrize("kind", ["normalized_duplicates", "reflected_duplicates", "aligned_exact_pairs"])
def test_geometrically_duplicate_shapes_disqualify_timing(glyph_api, tmp_path, kind):
    glyph_api.measurement.find_near_matches = lambda images, ids: {kind: [[ids[0], ids[1]]]}
    result = bench.run_benchmark(out=tmp_path / kind, glyph_count=2, concurrencies=(1,),
                                 executors=("thread",), repeats=3, render_sizes=())
    assert result["timing_claimable"] is False
    assert result["status"] == "failed"
    assert f"duplicate silhouettes: {kind}" in result["runs"][0]["errors"]
