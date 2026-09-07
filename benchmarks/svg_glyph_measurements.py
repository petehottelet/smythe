"""Measure original glyphs against Smythe's published numeric style brief.

This module contains no reference artwork, coordinates, or network access.
Measurements reproduce the independently authored research method recorded in
``docs/data/glyph-style-method.json``. NumPy, SciPy and Pillow are optional:
install ``smythe[glyphs]`` before invoking image measurement.

Profile gates apply only to a completed 192-glyph catalog. A smaller set gets a
calibration report, including every failed gate, without an acceptance claim.
Numeric similarity is art-direction evidence, never proof of originality.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Iterable, Sequence


METHOD_VERSION = "svg-style-v1"
FAMILY_QUOTAS = {
    "numeral_operator": 45,
    "bar_hook": 50,
    "stacked_marks": 3,
    "diagonal_lozenge": 17,
    "roofed_curves": 14,
    "rounded_loop_interlock": 39,
    "mixed": 24,
}
PROFILE_NAMES = {
    "classic": "classic", "classic-like": "classic",
    "expanded": "new-only", "expanded-like": "new-only", "new-only": "new-only",
}
# The same shape properties as the published study, in dimensionless units.
SHAPE_METRICS = (
    "bbox_ratio", "canvas_ink_fraction", "bbox_ink_fraction",
    "bbox_width_fraction", "bbox_height_fraction",
    "stroke_width_median_canvas_fraction",
    "horizontal_mirror_iou", "vertical_mirror_iou", "rotation_180_iou",
    "horizontal_skeleton_fraction", "vertical_skeleton_fraction", "diagonal_skeleton_fraction",
    "padding_left_fraction", "padding_right_fraction",
    "padding_top_fraction", "padding_bottom_fraction",
    "component_gap_center_distance_min_canvas_fraction", "counter_width_median_canvas_fraction",
    "stroke_width_p90_p10_ratio",
)
SENSITIVITY_METRIC = "threshold_96_160_ink_change_fraction"
METRICS = (*SHAPE_METRICS, SENSITIVITY_METRIC)
PIXEL_ALIASES = {
    "component_gap_center_distance_min_canvas_fraction": "component_gap_center_distance_min_px",
    "counter_width_median_canvas_fraction": "counter_width_median_px",
}


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
        from scipy import ndimage
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Glyph measurements require Pillow, NumPy and SciPy; install smythe[glyphs]."
        ) from exc
    return np, Image, ndimage, cKDTree


def _gray_and_mask(image):
    np, Image, _, _ = _dependencies()
    if isinstance(image, np.ndarray):
        if image.ndim != 2 or image.dtype != np.bool_:
            raise TypeError("An array input must be a two-dimensional boolean ink mask")
        if not all(image.shape):
            raise ValueError("The glyph image must have positive dimensions")
        gray = np.where(image, 0, 255).astype(np.uint8)
        base = Image.fromarray(gray).convert("RGBA")
    elif isinstance(image, Image.Image):
        if image.width < 1 or image.height < 1:
            raise ValueError("The glyph image must have positive dimensions")
        base = Image.new("RGBA", image.size, "white")
        base.alpha_composite(image.convert("RGBA"))
        gray = np.asarray(base.convert("L"))
    else:
        raise TypeError("measure_glyph expects a PIL image or boolean ink mask")
    return base, gray, gray < 128


def _thin(mask):
    """Zhang–Suen thinning, with simultaneous deletion in each subiteration."""
    np, _, _, _ = _dependencies()
    padded = np.pad(mask, 1).copy()
    while True:
        changed = False
        for step in (0, 1):
            center = padded[1:-1, 1:-1]
            neighbors = [
                padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
                padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
            ]
            count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = sum(
                (~neighbors[i] & neighbors[(i + 1) % 8]).astype(np.uint8) for i in range(8)
            )
            north, east, south, west = (neighbors[i] for i in (0, 2, 4, 6))
            guard = (
                ~(north & east & south) & ~(east & south & west)
                if step == 0 else ~(north & east & west) & ~(north & south & west)
            )
            remove = center & (count >= 2) & (count <= 6) & (transitions == 1) & guard
            if remove.any():
                center[remove] = False
                changed = True
        if not changed:
            return padded[1:-1, 1:-1]


def _labels(mask, connectivity=8):
    np, _, ndimage, _ = _dependencies()
    structure = np.ones((3, 3)) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    return ndimage.label(mask, structure=structure)


def _component_areas(mask, connectivity=8):
    np, _, _, _ = _dependencies()
    labels, _ = _labels(mask, connectivity)
    return sorted(np.bincount(labels.ravel())[1:].tolist(), reverse=True)


def _medial(mask, connectivity=8):
    np, _, ndimage, _ = _dependencies()
    skeleton = _thin(mask)
    # Padding defines outside the image as background, including an all-ink
    # input. It is identical to the reference calculation for padded artwork.
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    labels, count = _labels(mask, connectivity)
    for label in range(1, count + 1):
        region = labels == label
        if not (skeleton & region).any():
            y, x = np.unravel_index(np.argmax(distance * region), mask.shape)
            skeleton[y, x] = True
    return skeleton, distance


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _summary(values: Iterable[float]) -> dict | None:
    finite = [float(value) for value in values
              if isinstance(value, (int, float)) and math.isfinite(value)]
    if not finite:
        return None
    return {
        "n": len(finite), "min": min(finite), "p10": _percentile(finite, 10),
        "p25": _percentile(finite, 25), "median": _percentile(finite, 50),
        "p75": _percentile(finite, 75), "p90": _percentile(finite, 90), "max": max(finite),
    }


def _iou(a, b) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def measure_glyph(image) -> dict[str, Any]:
    """Return measurements of an unlit image; True denotes ink in a bool mask.

    Pixel measures retain their original units. Normalized widths, gaps and
    counters divide by image height; x padding/bounds divide by image width.
    No image, mask, vector coordinates, or external source is returned.
    """
    np, _, ndimage, cKDTree = _dependencies()
    base, gray, mask = _gray_and_mask(image)
    height, width = mask.shape
    result = {
        "measurement_version": METHOD_VERSION,
        "width": width, "height": height,
        "pixel_sha256": hashlib.sha256(np.asarray(base).tobytes()).hexdigest(),
        "ink_pixels": int(mask.sum()), "canvas_ink_fraction": float(mask.mean()),
        "blank": not bool(mask.any()),
    }
    if not mask.any():
        return result
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max() + 1), int(ys.min()), int(ys.max() + 1)
    crop = mask[y0:y1, x0:x1]
    bbox_width, bbox_height = x1 - x0, y1 - y0
    skeleton, distance = _medial(mask)
    widths = 2 * distance[skeleton]
    width_stats = _summary(widths)
    points = np.argwhere(skeleton)
    angles = []
    if len(points):
        tree = cKDTree(points)
        for indices in tree.query_ball_point(points, r=5):
            if len(indices) < 4:
                continue
            covariance = np.cov(points[indices].T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if eigenvalues[1] < 1 or eigenvalues[1] < 4 * max(eigenvalues[0], 1e-8):
                continue
            vector = eigenvectors[:, 1]
            angles.append(float(np.degrees(np.arctan2(vector[0], vector[1])) % 180))
    angles = np.asarray(angles)
    horizontal = int(((angles < 10) | (angles > 170)).sum())
    vertical = int(((angles > 80) & (angles < 100)).sum())
    diagonal = int((((angles > 33) & (angles < 57)) | ((angles > 123) & (angles < 147))).sum())
    holes_mask = ndimage.binary_fill_holes(mask) & ~mask
    components = _component_areas(mask)
    holes = _component_areas(holes_mask, 4)
    result.update({
        "bbox": [x0, y0, x1, y1], "bbox_width": bbox_width, "bbox_height": bbox_height,
        "bbox_ratio": bbox_width / bbox_height, "bbox_ink_fraction": float(crop.mean()),
        "bbox_width_fraction": bbox_width / width, "bbox_height_fraction": bbox_height / height,
        "bbox_shape_sha256": hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest(),
        "components_raw": len(components), "components_ge4px": sum(area >= 4 for area in components),
        "component_areas": components, "holes_raw": len(holes),
        "holes_ge4px": sum(area >= 4 for area in holes), "hole_areas": holes,
        "padding_left_fraction": x0 / width, "padding_right_fraction": (width - x1) / width,
        "padding_top_fraction": y0 / height, "padding_bottom_fraction": (height - y1) / height,
        "stroke_width_px": {key: round(value, 5) for key, value in width_stats.items() if key != "n"},
        "stroke_width_median_canvas_fraction": float(np.median(widths) / height),
        "skeleton_pixels": int(skeleton.sum()), "qualified_orientation_samples": len(angles),
        "horizontal_skeleton_fraction": horizontal / len(angles) if len(angles) else None,
        "vertical_skeleton_fraction": vertical / len(angles) if len(angles) else None,
        "diagonal_skeleton_fraction": diagonal / len(angles) if len(angles) else None,
        "orientation_histogram_15deg": np.histogram(angles, bins=np.arange(0, 181, 15))[0].tolist(),
        "horizontal_mirror_iou": _iou(crop, np.fliplr(crop)),
        "vertical_mirror_iou": _iou(crop, np.flipud(crop)), "rotation_180_iou": _iou(crop, np.flip(crop)),
        SENSITIVITY_METRIC: float(((gray < 160).sum() - (gray < 96).sum()) / mask.sum()),
    })
    for threshold in (96, 160):
        alternate = gray < threshold
        result[f"threshold{threshold}_components"] = len(_component_areas(alternate))
        result[f"threshold{threshold}_holes"] = len(
            _component_areas(ndimage.binary_fill_holes(alternate) & ~alternate, 4)
        )
    result["threshold_topology_stable"] = all(
        result[f"threshold{threshold}_{kind}"] == result[f"{kind}_raw"]
        for threshold in (96, 160) for kind in ("components", "holes")
    )
    labels, count = _labels(mask)
    gaps = []
    for label in range(1, count):
        other_distance = ndimage.distance_transform_edt(labels != label)
        for other in range(label + 1, count + 1):
            gaps.append(float(other_distance[labels == other].min()))
    gap = min(gaps) if gaps else None
    counter_skeleton, counter_distance = _medial(holes_mask, 4)
    counter = float(np.median(2 * counter_distance[counter_skeleton])) if counter_skeleton.any() else None
    result.update({
        "component_gap_center_distance_min_px": gap,
        "component_gap_center_distance_min_canvas_fraction": gap / height if gap is not None else None,
        "counter_width_median_px": counter,
        "counter_width_median_canvas_fraction": counter / height if counter is not None else None,
        "stroke_width_p90_p10_ratio": (
            result["stroke_width_px"]["p90"] / result["stroke_width_px"]["p10"]
        ),
    })
    return result


def _canonical_mask(image, size=64, margin=4):
    """Tight crop, aspect-preserving uniform scale, and centered comparison field."""
    np, Image, _, _ = _dependencies()
    _, _, mask = _gray_and_mask(image)
    if not mask.any():
        raise ValueError("Blank masks cannot participate in distinctness checks")
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    factor = size / max(crop.shape)
    dimensions = (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255).resize(dimensions, Image.Resampling.NEAREST)
    ) > 0
    canvas = np.zeros((size + 2 * margin, size + 2 * margin), dtype=bool)
    y = (canvas.shape[0] - resized.shape[0]) // 2
    x = (canvas.shape[1] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _mask_bits(mask) -> int:
    np, _, _, _ = _dependencies()
    return int.from_bytes(np.packbits(mask.ravel(), bitorder="little").tobytes(), "little")


def _aligned_iou(left: int, variants: list[tuple[str, int]], stride: int, radius: int, minimum=0):
    best = {"iou": 0.0, "reflection": "none", "translation": [0, 0]}
    left_area = left.bit_count()
    for reflection, right in variants:
        right_area = right.bit_count()
        if min(left_area, right_area) / max(left_area, right_area) < max(best["iou"], minimum):
            continue
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shift = dy * stride + dx
                shifted = right << shift if shift >= 0 else right >> -shift
                intersection = (left & shifted).bit_count()
                union = left_area + right_area - intersection
                overlap = intersection / union
                if overlap > best["iou"]:
                    best = {"iou": overlap, "reflection": reflection, "translation": [dx, dy]}
    return best


def find_near_matches(
    images: Sequence, glyph_ids: Sequence[str] | None = None, *, threshold=0.85,
    size=64, translation_radius=2,
) -> dict:
    """Flag aligned silhouettes, including reflections; never infer legal originality.

    Images are uniformly scaled so their longest tight-crop dimension is 64,
    preserving aspect ratio. A +/-2 comparison-pixel translation search follows.
    Four reflections (none, x, y, xy) include a half turn but no arbitrary
    rotation, shear, or anisotropic stretching. Packed integer masks keep the
    all-pairs comparison bounded for a 192-glyph catalog.
    """
    np, _, _, _ = _dependencies()
    if (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or not 0 < threshold <= 1
            or isinstance(size, bool) or not isinstance(size, int) or not 8 <= size <= 256
            or isinstance(translation_radius, bool) or not isinstance(translation_radius, int)
            or not 0 <= translation_radius <= 8):
        raise ValueError("Require 0<threshold<=1, 8<=size<=256, and 0<=translation_radius<=8")
    ids = list(glyph_ids) if glyph_ids is not None else [str(i) for i in range(len(images))]
    if len(ids) != len(images) or len(set(ids)) != len(ids):
        raise ValueError("Provide one unique glyph ID per image")
    exact, normalized, reflected = defaultdict(list), defaultdict(list), defaultdict(list)
    prepared = []
    margin = translation_radius + 2
    for glyph_id, image in zip(ids, images):
        _, _, raw = _gray_and_mask(image)
        exact[hashlib.sha256(str(raw.shape).encode() + raw.tobytes()).hexdigest()].append(glyph_id)
        mask = _canonical_mask(raw, size, margin)
        variants = [
            ("none", _mask_bits(mask)), ("horizontal", _mask_bits(np.fliplr(mask))),
            ("vertical", _mask_bits(np.flipud(mask))), ("both", _mask_bits(np.flip(mask))),
        ]
        # Identical reflections need not be compared repeatedly.
        variants = list({bits: (name, bits) for name, bits in reversed(variants)}.values())
        primary = _mask_bits(mask)
        normalized[primary].append(glyph_id)
        reflected[min(bits for _, bits in variants)].append(glyph_id)
        prepared.append((primary, variants, mask.shape[1]))
    matches = []
    comparisons = 0
    for right in range(len(prepared)):
        for left in range(right):
            comparisons += 1
            left_bits = prepared[left][0]
            _, variants, stride = prepared[right]
            best = _aligned_iou(left_bits, variants, stride, translation_radius, threshold)
            if best["iou"] >= threshold:
                matches.append({"glyph_ids": [ids[left], ids[right]], **best})
    def groups(values):
        return sorted([members for members in values.values() if len(members) > 1])
    return {
        "method": "tight crop; longest dimension uniform-scaled; centered; bounded translation; x/y reflections",
        "comparison_size": size, "translation_radius": translation_radius,
        "threshold": threshold, "count": len(images), "compared_pairs": comparisons,
        "exact_duplicates": groups(exact), "normalized_duplicates": groups(normalized),
        "reflected_duplicates": groups(reflected),
        "aligned_exact_pairs": [match["glyph_ids"] for match in matches if match["iou"] == 1],
        "near_matches": sorted(matches, key=lambda item: (-item["iou"], item["glyph_ids"])),
        "requires_optical_review": bool(matches),
        "originality_claim": False,
    }


def _record(record: dict) -> dict:
    merged = {**record, **record.get("measurements", {})}
    merged["profile"] = PROFILE_NAMES.get(str(record.get("profile", "")), "unknown")
    for normalized, pixels in PIXEL_ALIASES.items():
        if normalized not in merged and merged.get(pixels) is not None and merged.get("height"):
            merged[normalized] = merged[pixels] / merged["height"]
    return merged


def _reference_metrics(reference: dict) -> dict:
    metrics = dict(reference["metrics"])
    dimensions = reference.get("dimensions", {"128x128": reference.get("n", 0)})
    heights = {int(size.split("x")[1]) for size in dimensions}
    # Every published reference image has height128, including its one width129
    # image. A mixed-height reference requires per-row normalization instead.
    if len(heights) != 1:
        raise ValueError("Reference pixel distributions need a single source height for normalization")
    height = heights.pop()
    for normalized, pixels in PIXEL_ALIASES.items():
        if normalized not in metrics and pixels in metrics:
            metrics[normalized] = (
                {key: value / height for key, value in metrics[pixels].items()}
                if metrics[pixels] else None
            )
    return metrics


def _group_summary(records: list[dict]) -> dict:
    orientation_count = sum(record.get("qualified_orientation_samples", 0) for record in records)
    return {
        "n": len(records),
        "metrics": {key: _summary(record.get(key) for record in records) for key in METRICS},
        "component_counts": dict(sorted(Counter(str(r.get("components_ge4px")) for r in records).items())),
        "hole_counts": dict(sorted(Counter(str(r.get("holes_ge4px")) for r in records).items())),
        "qualified_orientation_samples": orientation_count,
        "pooled_direction_fractions": {
            key: sum((r.get(key) or 0) * r.get("qualified_orientation_samples", 0) for r in records)
            / orientation_count if orientation_count else None
            for key in ("horizontal_skeleton_fraction", "vertical_skeleton_fraction", "diagonal_skeleton_fraction")
        },
    }


def _continuous_gate(records: list[dict], metric: str, reference: dict | None) -> dict:
    def eligible(record):
        if metric == "component_gap_center_distance_min_canvas_fraction":
            return record.get("components_raw", record.get("components_ge4px", 2)) > 1
        if metric == "counter_width_median_canvas_fraction":
            return record.get("holes_raw", record.get("holes_ge4px", 1)) > 0
        if metric in {"horizontal_skeleton_fraction", "vertical_skeleton_fraction",
                      "diagonal_skeleton_fraction"}:
            return record.get("qualified_orientation_samples", 1) > 0
        return True
    eligible_records = [record for record in records if eligible(record)]
    missing = sum(record.get(metric) is None for record in eligible_records)
    values = [record.get(metric) for record in eligible_records if record.get(metric) is not None]
    invalid = sum(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    summary = _summary(finite)
    if not reference or summary is None:
        return {"passed": False, "reason": "missing reference band or no eligible measurements",
                "eligible_count": len(finite), "invalid_count": invalid,
                "missing_count": missing, "expected_eligible_count": len(eligible_records),
                "summary": summary}
    # The committed numeric bands have five decimal places. Half a final
    # decimal accommodates serialization rounding, not a style-gate relaxation.
    tolerance = 0.0000051
    median_pass = reference["p25"] - tolerance <= summary["median"] <= reference["p75"] + tolerance
    inside = sum(reference["p10"] - tolerance <= value <= reference["p90"] + tolerance for value in finite)
    fraction = inside / len(finite)
    band_pass = fraction + 1e-12 >= 0.80
    return {
        "passed": median_pass and band_pass and invalid == 0 and missing == 0,
        "median_in_reference_iqr": median_pass, "fraction_in_reference_p10_p90": fraction,
        "required_band_fraction": 0.80, "eligible_count": len(finite),
        "ineligible_count": len(records) - len(eligible_records), "invalid_count": invalid,
        "missing_count": missing, "expected_eligible_count": len(eligible_records),
        "summary": summary, "reference": reference,
        "reason": "median must lie inside reference IQR; at least 80% must lie inside P10–P90",
    }


def evaluate_catalog(
    records: Sequence[dict], reference_summary: dict, distinctness: dict | None = None,
    *, optical_review: dict | None = None,
) -> dict:
    """Audit assigned profiles, all seven families, frequencies, and distinctness.

    Every normalized continuous shape metric is gated. Raster anti-alias
    sensitivity is reported in a separate strict audit because it describes
    rasterization rather than glyph geometry. Topology must remain stable at
    thresholds 96/128/160. No failing metric is removed or silently relaxed.
    """
    normalized = [_record(record) for record in records]
    valid = [record for record in normalized if not record.get("blank", False)]
    failures, findings, profiles, families = [], [], {}, {}
    complete = len(records) == 192
    if not complete:
        findings.append(f"Calibration only: {len(records)}/192 glyphs; completed-profile acceptance is not claimed.")
    ids = [record.get("glyph_id") for record in normalized]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        failures.append("catalog.glyph_ids")
    if len(valid) != len(normalized):
        failures.append("catalog.blank_glyphs")
    unknown_profiles = [record.get("glyph_id") for record in normalized if record["profile"] == "unknown"]
    if unknown_profiles:
        failures.append("catalog.unknown_profiles")
    unknown_families = [record.get("glyph_id") for record in normalized if record.get("family") not in FAMILY_QUOTAS]
    if unknown_families:
        failures.append("catalog.unknown_families")
    unstable = [r.get("glyph_id") for r in valid if r.get("threshold_topology_stable") is not True]
    if unstable:
        failures.append("catalog.threshold_topology")
    wrong_resolution = [r.get("glyph_id") for r in valid if r.get("height") != 128]
    if wrong_resolution:
        failures.append("catalog.measurement_resolution")
        findings.append("Profile gates require 128px-high renders: PCA radius 5 and area >=4 topology are pixel-based.")
    sensitivity_failures = []
    for profile in ("classic", "new-only"):
        population = [record for record in valid if record["profile"] == profile]
        reference = reference_summary["groups"][profile]
        ref_metrics = _reference_metrics(reference)
        groups = _group_summary(population)
        groups["gates"] = {
            metric: _continuous_gate(population, metric, ref_metrics.get(metric)) for metric in METRICS
        }
        for metric, gate in groups["gates"].items():
            if not gate["passed"]:
                (sensitivity_failures if metric == SENSITIVITY_METRIC else failures).append(f"{profile}.{metric}")
            if gate.get("eligible_count", 0) < 10:
                findings.append(f"{profile}.{metric}: only {gate.get('eligible_count', 0)} eligible glyphs; inspect optically.")
        groups["frequency_gates"] = {}
        for kind in ("component_counts", "hole_counts"):
            actual, expected = groups[kind], reference[kind]
            bins = sorted(set(actual) | set(expected))
            comparisons = {
                key: {
                    "actual_fraction": actual.get(key, 0) / len(population) if population else 0,
                    "reference_fraction": expected.get(key, 0) / reference["n"],
                } for key in bins
            }
            for value in comparisons.values():
                value["difference_pp"] = 100 * abs(value["actual_fraction"] - value["reference_fraction"])
                value["passed"] = value["difference_pp"] <= 10 + 1e-10
            passed = bool(population) and all(value["passed"] for value in comparisons.values())
            groups["frequency_gates"][kind] = {"passed": passed, "allowed_difference_pp": 10, "bins": comparisons}
            if not passed:
                failures.append(f"{profile}.{kind}")
        profiles[profile] = groups
    for family, quota in FAMILY_QUOTAS.items():
        population = [record for record in valid if record.get("family") == family]
        groups = _group_summary(population)
        reference = reference_summary.get("observational_families", {}).get(family)
        groups["reference_metrics"] = _reference_metrics(reference) if reference else None
        groups["proposed_quota"] = quota
        groups["quota_matches"] = len(population) == quota
        groups["percentile_gates_applied"] = False
        groups["rationale"] = "Art-direction group for optical review; profile gates govern numeric acceptance."
        if family == "stacked_marks":
            groups["rationale"] += " Its two reference examples cannot support percentile gates."
        if complete and not groups["quota_matches"]:
            failures.append(f"family.{family}.quota")
        families[family] = groups
    exact_hashes = defaultdict(list)
    for record in valid:
        if record.get("bbox_shape_sha256"):
            exact_hashes[record["bbox_shape_sha256"]].append(record.get("glyph_id"))
    exact_duplicates = [members for members in exact_hashes.values() if len(members) > 1]
    if exact_duplicates:
        failures.append("catalog.duplicate_silhouettes")
    if distinctness is None:
        findings.append("Aligned/reflected near-match review has not been supplied; final acceptance remains pending.")
    else:
        expected_pairs = len(records) * (len(records) - 1) // 2
        if (distinctness.get("compared_pairs") != expected_pairs
                or distinctness.get("count", len(records)) != len(records)):
            failures.append("catalog.incomplete_distinctness")
        if (distinctness.get("normalized_duplicates") or distinctness.get("reflected_duplicates")
                or distinctness.get("aligned_exact_pairs")):
            failures.append("catalog.aligned_duplicate_silhouettes")
    reviewed_pairs = {tuple(sorted(pair)) for pair in (optical_review or {}).get("reviewed_near_pairs", [])}
    pending_near_matches = bool(distinctness and any(
        tuple(sorted(match["glyph_ids"])) not in reviewed_pairs
        for match in distinctness.get("near_matches", [])
    ))
    if pending_near_matches:
        findings.append("Near-match flags require optical review; overlap alone neither proves nor disproves originality.")
    optical_pass = bool(optical_review and optical_review.get("passed") is True)
    passed = not failures
    return {
        "method_version": METHOD_VERSION, "status": "complete" if complete else "calibration",
        "count": len(records), "expected_count": 192, "profiles": profiles, "families": families,
        "all": _group_summary(valid),
        "passes_numeric_shape_gates": passed,
        "accepted": complete and passed and distinctness is not None and not pending_near_matches and optical_pass,
        "optical_review_required": not optical_pass or pending_near_matches,
        "optical_review": optical_review,
        "failed_gates": failures,
        "strict_all_continuous_failed_gates": failures + sensitivity_failures,
        "raster_sensitivity_failed_gates": sensitivity_failures,
        "raster_sensitivity_policy": "Reported explicitly; anti-alias distribution is not a geometry-style gate.",
        "topology_threshold_failures": unstable, "exact_duplicate_silhouettes": exact_duplicates,
        "measurement_resolution_failures": wrong_resolution,
        "unknown_profiles": unknown_profiles, "unknown_families": unknown_families,
        "distinctness": distinctness, "findings": findings,
        "acceptance_rationale": (
            "Complete catalog, declared family quotas, every profile shape median within its reference IQR, "
            "at least 80% in each reference P10–P90 band, topology frequencies within 10 percentage points, "
            "stable threshold topology, distinct silhouettes, and separate optical review. Numeric acceptance "
            "does not assert an exact style match or establish originality."
        ),
    }
