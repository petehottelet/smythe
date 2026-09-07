"""Deterministic geometry, topology, direction, and distinctness regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from benchmarks.svg_glyph_measurements import (  # noqa: E402
    FAMILY_QUOTAS, METRICS, SHAPE_METRICS, SENSITIVITY_METRIC,
    evaluate_catalog, find_near_matches, measure_glyph,
)


ROOT = Path(__file__).resolve().parents[1]


def rectangle(size=(128, 128), bounds=(20, 30, 99, 89)):
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).rectangle(bounds, fill="black")
    return image


def asymmetric(size=64):
    mask = np.zeros((size, size), dtype=bool)
    mask[8:55, 10:20] = True
    mask[43:55, 10:42] = True
    mask[13:23, 33:45] = True
    return mask


def test_exact_bounds_area_and_normalization_use_actual_image_dimensions():
    result = measure_glyph(rectangle(size=(129, 128)))
    assert result["bbox"] == [20, 30, 100, 90]
    assert result["ink_pixels"] == 80 * 60
    assert result["bbox_ratio"] == 80 / 60
    assert result["bbox_width_fraction"] == 80 / 129
    assert result["bbox_height_fraction"] == 60 / 128
    assert result["canvas_ink_fraction"] == 4800 / (129 * 128)
    assert result["padding_left_fraction"] == 20 / 129
    assert result["padding_right_fraction"] == 29 / 129
    assert result["components_ge4px"] == 1
    assert result["holes_ge4px"] == 0
    assert result["horizontal_mirror_iou"] == 1
    assert result["threshold_topology_stable"] is True


def test_transparency_is_white_composited_and_threshold_is_strict():
    assert measure_glyph(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))["blank"] is True
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 127))
    assert measure_glyph(image)["blank"] is True  # Composite gray128 is not ink.
    image.putpixel((4, 4), (0, 0, 0, 128))
    result = measure_glyph(image)
    assert result["ink_pixels"] == 1
    assert result["components_raw"] == 1
    assert result["components_ge4px"] == 0
    assert result["threshold_topology_stable"] is False


def test_components_are_eight_connected_but_counter_connectivity_is_four():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True  # Corner contact is one ink component.
    assert measure_glyph(mask)["components_ge4px"] == 1
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:12, 3:12] = True
    mask[5:8, 5:8] = False
    mask[8:10, 8:10] = False  # Diagonal white contact is two holes.
    result = measure_glyph(mask)
    assert result["holes_raw"] == result["holes_ge4px"] == 2
    assert result["hole_areas"] == [9, 4]


def test_collapsed_compact_centerline_retains_medial_point_and_holes():
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:8, 6:8] = True
    result = measure_glyph(mask)
    assert result["skeleton_pixels"] == 1
    assert result["stroke_width_px"]["median"] == 2
    assert result["qualified_orientation_samples"] == 0
    assert result["horizontal_skeleton_fraction"] is None
    mask[3:12, 3:12] = True
    mask[6:8, 6:8] = False
    result = measure_glyph(mask)
    assert result["holes_ge4px"] == 1
    assert result["counter_width_median_px"] == 2
    assert result["counter_width_median_canvas_fraction"] == 2 / 16


def test_orientation_uses_local_pca_and_spacing_measures_pixel_centers():
    result = measure_glyph(rectangle(bounds=(10, 55, 117, 64)))
    assert result["horizontal_skeleton_fraction"] == 1
    assert result["vertical_skeleton_fraction"] == 0
    assert result["diagonal_skeleton_fraction"] == 0
    assert result["qualified_orientation_samples"] > 50
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 4:8] = True
    mask[8:16, 13:17] = True
    result = measure_glyph(mask)
    assert result["component_gap_center_distance_min_px"] == 6
    assert result["component_gap_center_distance_min_canvas_fraction"] == 6 / 32


def test_measurements_repeat_and_do_not_return_artwork():
    image = rectangle()
    first, second = measure_glyph(image), measure_glyph(image)
    assert first == second
    assert len(first["pixel_sha256"]) == len(first["bbox_shape_sha256"]) == 64
    json.dumps(first, allow_nan=False)
    assert not {"image", "mask", "svg", "paths", "commands"} & set(first)


def test_uniform_scale_translation_and_mirror_do_not_create_new_identity():
    mask = asymmetric()
    translated = np.pad(mask, ((17, 9), (11, 19)))
    scaled = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    mirrored = np.fliplr(mask)
    report = find_near_matches([mask, translated, scaled, mirrored], ["a", "b", "c", "d"])
    assert report["compared_pairs"] == 6
    assert len(report["near_matches"]) == 6
    assert all(match["iou"] == 1 for match in report["near_matches"])
    assert len(report["aligned_exact_pairs"]) == 6
    assert any(match["reflection"] == "horizontal" for match in report["near_matches"])
    assert report["originality_claim"] is False


def test_comparison_never_anisotropically_stretches_shapes():
    wide = np.zeros((64, 64), dtype=bool)
    wide[25:35, 5:55] = True
    tall = wide.T
    report = find_near_matches([wide, tall])
    assert report["near_matches"] == []
    with pytest.raises(ValueError, match="Blank"):
        find_near_matches([np.zeros((10, 10), dtype=bool)])
    with pytest.raises(ValueError, match="unique glyph ID"):
        find_near_matches([wide, tall], ["same", "same"])
    with pytest.raises(ValueError, match="Require"):
        find_near_matches([wide], size=64.5)
    with pytest.raises(ValueError, match="Require"):
        find_near_matches([wide], threshold=float("nan"))


def _synthetic_reference():
    bands = {key: {"min": 0, "p10": .2, "p25": .4, "median": .5,
                   "p75": .6, "p90": .8, "max": 1} for key in METRICS}
    group = {"n": 10, "dimensions": {"128x128": 10}, "metrics": bands,
             "component_counts": {"1": 10}, "hole_counts": {"0": 10}}
    return {"groups": {"classic": copy.deepcopy(group), "new-only": copy.deepcopy(group)},
            "observational_families": {}}


def _records(count=192):
    families = [family for family, quota in FAMILY_QUOTAS.items() for _ in range(quota)]
    return [{"glyph_id": f"g{i:03}", "profile": "classic-like" if i % 2 == 0 else "expanded-like",
             "family": families[i], "measurements": {
                 **dict.fromkeys(METRICS, .5), "components_ge4px": 1, "holes_ge4px": 0,
                 "components_raw": 2, "holes_raw": 1,
                 "threshold_topology_stable": True, "bbox_shape_sha256": f"unique-{i}",
                 "qualified_orientation_samples": 10, "width": 128, "height": 128,
             }} for i in range(count)]


def test_profile_gates_report_median_band_and_frequency_failures_without_relaxing():
    records = _records()
    reference = _synthetic_reference()
    distinctness = {"near_matches": [], "count": 192, "compared_pairs": 18336}
    report = evaluate_catalog(records, reference, distinctness=distinctness)
    assert report["passes_numeric_shape_gates"] is True
    assert report["accepted"] is False  # An optical review is still needed.
    reviewed = evaluate_catalog(records, reference, distinctness=distinctness,
                                optical_review={"passed": True, "reviewed_near_pairs": []})
    assert reviewed["accepted"] is True
    incomplete = evaluate_catalog(records, reference, distinctness={"near_matches": []},
                                  optical_review={"passed": True})
    assert "catalog.incomplete_distinctness" in incomplete["failed_gates"]
    assert incomplete["accepted"] is False
    # Median remains in the IQR, but25% outside P10/P90 fails the80% gate.
    for record in records[:48]:
        record["measurements"]["bbox_ratio"] = .95
        record["measurements"]["components_ge4px"] = 2
    report = evaluate_catalog(records, reference)
    assert "classic.bbox_ratio" in report["failed_gates"]
    assert "new-only.bbox_ratio" in report["failed_gates"]
    assert report["profiles"]["classic"]["gates"]["bbox_ratio"]["median_in_reference_iqr"] is True
    assert report["profiles"]["classic"]["gates"]["bbox_ratio"]["fraction_in_reference_p10_p90"] == .75
    assert "classic.component_counts" in report["failed_gates"]
    assert report["accepted"] is False


def test_calibration_has_no_completed_catalog_claim_and_families_have_no_percentile_gates():
    report = evaluate_catalog(_records(24), _synthetic_reference())
    assert report["status"] == "calibration"
    assert report["accepted"] is False
    assert report["count"] == 24
    assert all(not family["percentile_gates_applied"] for family in report["families"].values())
    assert "two reference examples" in report["families"]["stacked_marks"]["rationale"]


def test_raster_sensitivity_failure_is_reported_separately_and_missing_data_does_not_pass():
    records = _records()
    for record in records:
        record["measurements"][SENSITIVITY_METRIC] = 0
    report = evaluate_catalog(records, _synthetic_reference())
    assert report["passes_numeric_shape_gates"] is True
    assert f"classic.{SENSITIVITY_METRIC}" in report["raster_sensitivity_failed_gates"]
    assert report["strict_all_continuous_failed_gates"]
    for record in records:
        record["measurements"][SHAPE_METRICS[0]] = None
    report = evaluate_catalog(records, _synthetic_reference())
    assert f"classic.{SHAPE_METRICS[0]}" in report["failed_gates"]
    assert report["profiles"]["classic"]["gates"][SHAPE_METRICS[0]]["eligible_count"] == 0


def test_invalid_values_duplicates_and_unstable_topology_cannot_pass():
    records = _records()
    records[0]["measurements"]["bbox_ratio"] = float("nan")
    records[0]["measurements"]["bbox_shape_sha256"] = "unique-1"
    records[0]["measurements"]["threshold_topology_stable"] = False
    report = evaluate_catalog(records, _synthetic_reference())
    assert "classic.bbox_ratio" in report["failed_gates"]
    assert "catalog.duplicate_silhouettes" in report["failed_gates"]
    assert "catalog.threshold_topology" in report["failed_gates"]
    assert report["profiles"]["classic"]["gates"]["bbox_ratio"]["invalid_count"] == 1
    json.dumps(report, allow_nan=False)


def test_committed_reference_bands_normalize_pixel_gaps_and_counters():
    reference = json.loads((ROOT / "docs/data/glyph-style-summary.json").read_text(encoding="utf-8"))
    report = evaluate_catalog(_records(24), reference)
    gate = report["profiles"]["classic"]["gates"]["component_gap_center_distance_min_canvas_fraction"]
    assert gate["reference"]["median"] == 11.5 / 128
    gate = report["profiles"]["classic"]["gates"]["counter_width_median_canvas_fraction"]
    assert gate["reference"]["median"] == 14 / 128


def test_normalized_values_do_not_hide_incompatible_measurement_resolution():
    records = _records()
    records[0]["measurements"]["height"] = 16
    report = evaluate_catalog(records, _synthetic_reference())
    assert "catalog.measurement_resolution" in report["failed_gates"]
    assert report["measurement_resolution_failures"] == ["g000"]


def test_missing_required_or_topologically_eligible_measurements_cannot_pass():
    records = _records()
    for record in records[2:]:
        record["measurements"]["bbox_ratio"] = None
    report = evaluate_catalog(records, _synthetic_reference())
    gate = report["profiles"]["classic"]["gates"]["bbox_ratio"]
    assert not gate["passed"]
    assert gate["missing_count"] == 95
    assert gate["expected_eligible_count"] == 96
    for metric in ("component_gap_center_distance_min_canvas_fraction",
                   "counter_width_median_canvas_fraction", "horizontal_skeleton_fraction"):
        records = _records()
        records[0]["measurements"][metric] = None
        report = evaluate_catalog(records, _synthetic_reference())
        assert not report["profiles"]["classic"]["gates"][metric]["passed"]
    records[0]["measurements"]["qualified_orientation_samples"] = 0
    report = evaluate_catalog(records, _synthetic_reference())
    assert report["profiles"]["classic"]["gates"]["horizontal_skeleton_fraction"]["passed"]
