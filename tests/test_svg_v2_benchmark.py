"""The 256-glyph extension and full-workflow evidence keep their boundaries."""

from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks import run_svg_v2_benchmark as runner
from benchmarks.svg_glyphs import render_svg
from screensaver.glyph_design_v2 import generate_glyph, write_study


def test_extension_preserves_every_reviewed_svg_and_adds_64_valid_shapes():
    assert len(runner.EXTENDED) == 256
    for index in range(192):
        assert generate_glyph(index) == generate_glyph(index, outlines=runner.EXTENDED)
    for index in range(192, 256):
        result = runner.generate_work(index, "thread")
        verified = runner.validate_work(index, result["payload"], "thread")
        assert verified["status"] == "passed", verified
        assert set(verified["small_sizes"]) == {"16", "32", "64"}


def test_extended_sheet_has_every_cell_including_the_last_row(tmp_path):
    from PIL import Image

    directory = tmp_path / "catalog"
    manifest = write_study(directory, outlines=runner.EXTENDED)
    assert manifest["count"] == 256
    assert len(list(directory.glob("GLYPH-*.svg"))) == 256
    sheet = Image.open(directory / "contact-sheet-128.png")
    assert sheet.size == (2560, 2880)
    assert Image.open(directory / "sheet-6.png").size == (2560, 180)
    last = sheet.crop((15 * 160 + 16, 15 * 180 + 8, 15 * 160 + 144, 15 * 180 + 136))
    assert last.tobytes() == render_svg(generate_glyph(255, outlines=runner.EXTENDED)["svg"]).convert("RGB").tobytes()
    with pytest.raises(FileExistsError):
        write_study(directory, outlines=runner.EXTENDED)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return runner.run_benchmark(tmp_path_factory.mktemp("v2-smoke") / "run", sizes=(4,),
                                concurrencies=(1, 2), repeats=1)


def test_real_smythe_thread_and_process_workflows_include_validation_and_export(smoke):
    assert smoke["status"] == "passed"
    assert smoke["claimable"] is False
    assert len(smoke["runs"]) == 4
    assert runner.review_record(smoke) == smoke["summaries"]
    for run in smoke["runs"]:
        assert run["execution_id"]
        assert run["assembly"]["output_bytes"] > 0
        path = runner.ROOT / run["assembly"]["catalog"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["assembly"]["catalog_sha256"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "order", "source", "count", "api", "clock",
                                    "phase", "hash", "near_match", "pair_count", "assembly", "cache",
                                    "latency", "design", "provider_identity", "workers", "concurrency"])
def test_claim_review_rejects_incomplete_or_misleading_evidence(smoke, damage):
    record = deepcopy(smoke)
    run = record["runs"][0]
    if damage == "missing":
        record["runs"].pop()
    elif damage == "duplicate":
        record["runs"][1] = deepcopy(run)
    elif damage == "order":
        record["runs"].reverse()
    elif damage == "source":
        run["source_stable"] = "true"
    elif damage == "count":
        run["valid_glyphs"] -= 1
    elif damage == "api":
        run["api_cost_usd"] = .01
    elif damage == "clock":
        run["end_to_end_wall_s"] = float("nan")
    elif damage == "phase":
        run["validation_wall_s"] = run["end_to_end_wall_s"] + 1
    elif damage == "hash":
        run["glyphs"][0]["svg_sha256"] = "changed"
    elif damage == "near_match":
        run["distinctness"]["near_matches"] = [{"iou": .9}]
    elif damage == "pair_count":
        run["distinctness"]["compared_pairs"] -= 1
    elif damage == "assembly":
        run["assembly"] = None
    elif damage == "provider_identity":
        run["calls"][0]["index"] = 999
    elif damage == "workers":
        run["workers"] = 999
    elif damage == "concurrency":
        run["max_in_flight_provider_calls"] = 999
    else:
        key, value = {"cache": ("cached_svg_outputs", True), "latency": ("simulated_latency_s", 1),
                      "design": ("design_work_timed", True)}[damage]
        record["protocol"][key] = value
    with pytest.raises(ValueError):
        runner.review_record(record)


def test_validation_rejects_small_size_topology_loss(monkeypatch):
    original = runner.measure_glyph

    def lost_component(image):
        result = original(image)
        if image.width == 16:
            result["components_raw"] += 1
        return result

    monkeypatch.setattr(runner, "measure_glyph", lost_component)
    payload = runner.generate_work(17, "thread")["payload"]
    result = runner.validate_work(17, payload, "thread")
    assert result["status"] == "failed"
    assert "16 px" in result["error"]


def test_invalid_or_existing_campaign_is_rejected_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_once", lambda **kwargs: pytest.fail("Must fail before work"))
    with pytest.raises(ValueError):
        runner.run_benchmark(tmp_path / "bad", sizes=(257,))
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "results.json"
    sentinel.write_text(json.dumps({"preserve": True}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.run_benchmark(existing)
    assert json.loads(sentinel.read_bytes()) == {"preserve": True}
