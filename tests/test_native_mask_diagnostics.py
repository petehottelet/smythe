"""Fixed-position native raster diagnostics preserve failures and tiny counters."""

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from screensaver.verification.analyze_native_masks import (
    ROOT,
    analyze,
    directed_distance,
    read_image,
    symmetric_max,
    topology,
)


@pytest.mark.parametrize(
    "offset,expected,outside",
    [((0, 0), 0.0, 0), ((0, 1), 1.0, 0), ((1, 1), math.sqrt(2), 1), ((0, 2), 2.0, 1)],
)
def test_euclidean_pixel_distance_does_not_align_or_allow_diagonal_neighbors(
    offset, expected, outside
):
    actual = np.zeros((8, 8), dtype=bool)
    source = actual.copy()
    actual[2, 2] = True
    source[2 + offset[0], 2 + offset[1]] = True
    forward, reverse = directed_distance(actual, source), directed_distance(source, actual)
    assert forward == reverse
    assert forward["max_px"] == pytest.approx(expected)
    assert forward["pixels_farther_than_1px"] == outside
    assert symmetric_max(forward, reverse) == pytest.approx(expected)


def test_empty_and_missing_masks_have_distinct_serializable_results():
    blank = np.zeros((8, 8), dtype=bool)
    shape = blank.copy()
    shape[2:4, 2:4] = True
    assert directed_distance(blank, blank) == {"max_px": 0.0, "pixels_farther_than_1px": 0}
    missing = directed_distance(shape, blank)
    assert missing == {"max_px": None, "pixels_farther_than_1px": 4}
    assert symmetric_max(missing, directed_distance(blank, shape)) is None
    assert "null" in json.dumps(missing, allow_nan=False)


def test_raw_counters_keep_one_pixel_holes_separate_from_four_pixel_cutoff():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:11, 1:11] = True
    mask[3, 3] = False
    mask[6:8, 6:8] = False
    result = topology(mask)
    assert result["components"] == result["components_area_ge4"] == 1
    assert result["holes"] == 2
    assert result["hole_areas"] == [4, 1]
    assert result["holes_area_ge4"] == 1
    # Filling the tiny counter changes raw topology even when the >=4 result agrees.
    mask[3, 3] = True
    filled = topology(mask)
    assert filled["holes"] == 1
    assert filled["holes_area_ge4"] == result["holes_area_ge4"]


def test_diagonal_foreground_connection_and_background_hole_use_dual_connectivity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1, 1] = mask[2, 2] = False
    result = topology(mask)
    assert result["components"] == 1
    assert result["holes"] == 1  # The center connects to the exterior only diagonally.
    assert result["hole_areas"] == [1]


def _atlas(path, *, empty=False, missing=None, fill_blank=False):
    pixels = np.zeros((1024, 1024), dtype=np.uint8)
    if not empty:
        for slot in range(249):
            if slot == missing or slot == 4:
                continue
            y, x = slot // 16 * 64, slot % 16 * 64
            pixels[y + 10 : y + 30, x + 10 : x + 30] = 255
    if fill_blank:
        pixels[10, 4 * 64 + 10] = 255
    Image.fromarray(pixels).save(path)


def test_all_blank_atlases_report_missing_shapes_without_nan(tmp_path):
    path = tmp_path / "blank.png"
    _atlas(path, empty=True)
    result = analyze(path, path, ROOT / "screensaver/native-catalog.json", "blank-test")
    assert result["status"] == "diagnostic; not an acceptance receipt"
    assert result["summary"]["blank_slot_empty_in_both"]
    assert result["summary"]["missing_shape_slots"] == [slot for slot in range(249) if slot != 4]
    assert result["summary"]["centroid_delta_median_xy_px"] is None
    assert result["summary"]["gray_coverage_area_relative_change_median"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("defect", ["missing", "filled_blank"])
def test_complete_atlas_diagnostic_reports_missing_geometry_and_filled_blank(tmp_path, defect):
    source, actual = tmp_path / "source.png", tmp_path / "actual.png"
    _atlas(source)
    _atlas(actual, missing=57 if defect == "missing" else None, fill_blank=defect == "filled_blank")
    result = analyze(actual, source, ROOT / "screensaver/native-catalog.json", defect)
    if defect == "missing":
        assert result["summary"]["missing_shape_slots"] == [57]
        assert result["glyphs"][57]["raw_iou"] == 0
        assert result["glyphs"][57]["symmetric_mask_distance_max_px"] is None
        assert result["glyphs"][57]["source_to_actual"]["pixels_farther_than_1px"] == 400
    else:
        assert not result["summary"]["blank_slot_empty_in_both"]
        assert result["glyphs"][4]["raw_iou"] == 0
        assert result["glyphs"][4]["actual_to_source"]["pixels_farther_than_1px"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("defect", ["transparent", "colored", "resized"])
def test_atlas_reader_rejects_invisible_or_incomparable_inputs(tmp_path, defect):
    path = tmp_path / "invalid.png"
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
    if defect == "transparent":
        image.putalpha(0)
    elif defect == "colored":
        image.putpixel((3, 3), (0, 255, 0, 255))
    else:
        image = image.resize((512, 512))
    image.save(path)
    with pytest.raises(ValueError):
        read_image(path)


def test_cli_refuses_to_overwrite_a_receipt_before_reading_inputs(tmp_path):
    destination = tmp_path / "receipt.json"
    destination.write_bytes(b"existing evidence\n")
    command = [
        sys.executable,
        str(ROOT / "screensaver/verification/analyze_native_masks.py"),
        "--actual",
        "missing.png",
        "--source",
        "missing.png",
        "--out",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "Refusing to overwrite" in result.stderr
    assert destination.read_bytes() == b"existing evidence\n"


def test_script_root_resolves_to_repository_not_screensaver_directory():
    assert ROOT == Path(__file__).resolve().parents[1]
