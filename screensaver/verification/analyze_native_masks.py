"""Read-only, unaligned 64px native/source atlas diagnostics; never an acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
FG8 = ndimage.generate_binary_structure(2, 2)
BG4 = ndimage.generate_binary_structure(2, 1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    path = path.resolve()
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": sha(path),
    }


def read_image(path):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    if rgba.shape != (1024, 1024, 4):
        raise ValueError("Expected an unscaled 1024x1024 atlas (16 columns, 64px cells)")
    if not (rgba[:, :, 3] == 255).all():
        raise ValueError("Atlas must be opaque")
    if not np.array_equal(rgba[:, :, 0], rgba[:, :, 1]) or not np.array_equal(
        rgba[:, :, 1], rgba[:, :, 2]
    ):
        raise ValueError("Expected grayscale white ink on black")
    return rgba[:, :, 0]


def topology(mask):
    labels, count = ndimage.label(mask, structure=FG8)
    areas = np.bincount(labels.ravel())[1:]
    negatives, negative_count = ndimage.label(~mask, structure=BG4)
    border = set(
        np.concatenate((negatives[0], negatives[-1], negatives[:, 0], negatives[:, -1])).tolist()
    )
    negative_areas = np.bincount(negatives.ravel())
    holes = [
        int(negative_areas[index]) for index in range(1, negative_count + 1) if index not in border
    ]
    return {
        "components": count,
        "component_areas": sorted(map(int, areas), reverse=True),
        "holes": len(holes),
        "hole_areas": sorted(holes, reverse=True),
        "components_area_ge4": int((areas >= 4).sum()),
        "holes_area_ge4": sum(area >= 4 for area in holes),
    }


def bounds(mask):
    y, x = np.where(mask)
    return None if len(x) == 0 else [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def centroid(mask):
    y, x = np.where(mask)
    return None if len(x) == 0 else [float(x.mean() + 0.5), float(y.mean() + 0.5)]


def directed_distance(a, b):
    if not a.any():
        return {"max_px": 0.0, "pixels_farther_than_1px": 0}
    if not b.any():
        return {"max_px": None, "pixels_farther_than_1px": int(a.sum())}
    distances = ndimage.distance_transform_edt(~b)[a]
    return {"max_px": float(distances.max()), "pixels_farther_than_1px": int((distances > 1).sum())}


def symmetric_max(a, b):
    return None if a["max_px"] is None or b["max_px"] is None else max(a["max_px"], b["max_px"])


def analyze(actual_path, source_path, catalog_path, label):
    a, b = read_image(actual_path), read_image(source_path)
    native = json.loads(catalog_path.read_text(encoding="utf-8"))
    if (
        native.get("count") != 249
        or native.get("blank_index") != 4
        or len(native.get("source_glyphs", [])) != 249
    ):
        raise ValueError("Expected the 249-slot native catalog with blank slot 4")
    rows = []
    for slot in range(249):
        y, x = slot // 16 * 64, slot % 16 * 64
        ag, bg = a[y : y + 64, x : x + 64], b[y : y + 64, x : x + 64]
        am, bm = ag >= 128, bg >= 128
        union, intersection = int((am | bm).sum()), int((am & bm).sum())
        ba, bb, ca, cb = bounds(am), bounds(bm), centroid(am), centroid(bm)
        ab, be = directed_distance(am, bm), directed_distance(bm, am)
        boundary_a = am & ~ndimage.binary_erosion(am, structure=FG8, border_value=0)
        boundary_b = bm & ~ndimage.binary_erosion(bm, structure=FG8, border_value=0)
        edge_ab, edge_ba = (
            directed_distance(boundary_a, boundary_b),
            directed_distance(boundary_b, boundary_a),
        )
        rows.append(
            {
                "slot": slot,
                "glyph_id": native["source_glyphs"][slot]["glyph_id"],
                "family": "reference" if slot < 57 else "original",
                "raw_iou": intersection / union if union else 1.0,
                "actual_ink_pixels": int(am.sum()),
                "source_ink_pixels": int(bm.sum()),
                "symmetric_difference_pixels": int((am ^ bm).sum()),
                "actual_to_source": ab,
                "source_to_actual": be,
                "symmetric_mask_distance_max_px": symmetric_max(ab, be),
                "symmetric_boundary_distance_max_px": symmetric_max(edge_ab, edge_ba),
                "actual_bounds": ba,
                "source_bounds": bb,
                "bounds_delta": None
                if ba is None or bb is None
                else [u - v for u, v in zip(ba, bb)],
                "centroid_delta_px": None
                if ca is None or cb is None
                else [u - v for u, v in zip(ca, cb)],
                "actual_topology": topology(am),
                "source_topology": topology(bm),
                "mean_absolute_gray_difference": float(
                    np.abs(ag.astype(float) - bg.astype(float)).mean()
                ),
                "gray_coverage_area_actual": float(ag.sum() / 255),
                "gray_coverage_area_source": float(bg.sum() / 255),
            }
        )
    visible = [row for row in rows if row["slot"] != 4]
    distances = [row["symmetric_mask_distance_max_px"] for row in rows]
    boundary_distances = [row["symmetric_boundary_distance_max_px"] for row in rows]
    centroids = [
        row["centroid_delta_px"] for row in visible if row["centroid_delta_px"] is not None
    ]
    coverage_changes = [
        (row["gray_coverage_area_actual"] - row["gray_coverage_area_source"])
        / row["gray_coverage_area_source"]
        for row in visible
        if row["gray_coverage_area_source"]
    ]
    summary = {
        "glyphs": 249,
        "visible_glyphs": 248,
        "blank_slot": 4,
        "blank_slot_empty_in_both": rows[4]["actual_ink_pixels"]
        == rows[4]["source_ink_pixels"]
        == 0,
        "missing_shape_slots": [
            row["slot"]
            for row in visible
            if not row["actual_ink_pixels"] or not row["source_ink_pixels"]
        ],
        "raw_iou_min": min(row["raw_iou"] for row in visible),
        "raw_iou_mean_visible": float(np.mean([row["raw_iou"] for row in visible])),
        "raw_iou_mean_all249": float(np.mean([row["raw_iou"] for row in rows])),
        "raw_iou_p10_p50_p90_visible": np.quantile(
            [row["raw_iou"] for row in visible], [0.1, 0.5, 0.9]
        ).tolist(),
        "raw_iou_below99_count": sum(row["raw_iou"] < 0.99 for row in visible),
        "symmetric_mask_distance_max_px": None if None in distances else max(distances),
        "symmetric_boundary_distance_max_px": None
        if None in boundary_distances
        else max(boundary_distances),
        "actual_pixels_farther_than_1px": sum(
            row["actual_to_source"]["pixels_farther_than_1px"] for row in rows
        ),
        "source_pixels_farther_than_1px": sum(
            row["source_to_actual"]["pixels_farther_than_1px"] for row in rows
        ),
        "glyphs_farther_than_1px": [
            row["slot"]
            for row in rows
            if row["symmetric_mask_distance_max_px"] is None
            or row["symmetric_mask_distance_max_px"] > 1
        ],
        "max_abs_bounds_delta_px": max(
            (abs(d) for row in visible for d in row["bounds_delta"] or []), default=None
        ),
        "bounds_changed_glyph_count": sum(any(row["bounds_delta"] or []) for row in visible),
        "centroid_delta_median_xy_px": np.median(centroids, axis=0).tolist() if centroids else None,
        "gray_coverage_area_relative_change_median": float(np.median(coverage_changes))
        if coverage_changes
        else None,
        "symmetric_difference_pixels": sum(row["symmetric_difference_pixels"] for row in rows),
    }
    for key in ("components", "holes", "components_area_ge4", "holes_area_ge4"):
        summary[key + "_mismatch_slots"] = [
            row["slot"]
            for row in visible
            if row["actual_topology"][key] != row["source_topology"][key]
        ]
        summary[key + "_actual_source_totals"] = [
            sum(row[which][key] for row in rows) for which in ("actual_topology", "source_topology")
        ]
    return {
        "version": "native-mask-diagnostic-v2",
        "status": "diagnostic; not an acceptance receipt",
        "label": label,
        "actual": identity(actual_path),
        "source": identity(source_path),
        "native_catalog": identity(catalog_path),
        "catalog_sha256": native["catalog_sha256"],
        "analysis_script": identity(Path(__file__)),
        "analysis_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "platform": platform.platform(),
        },
        "method": {
            "cell_size": 64,
            "columns": 16,
            "margin": 0,
            "ink": "white on opaque black",
            "threshold": "RGB>=128 (all RGB channels equal)",
            "alignment": "none: no shift, scale, rotation, or flip correction",
            "distance": "Euclidean distance between foreground pixel centers; both directed sets measured independently; symmetric value is their maximum",
            "one_pixel_scope": "Each thresholded foreground pixel lies at most 1 Euclidean pixel from foreground in the other mask. Adjacent horizontal/vertical coverage differences fit; diagonal sqrt(2) separation does not. This check alone does not establish exact subpixel registration or identical stroke coverage.",
            "boundary_distance": "Additional distance between 8-neighbor foreground boundary pixel centers; both directions",
            "topology": "Foreground 8-connectivity; holes are enclosed 4-connected background components. Raw area >= 1 counts and area >= 4 counts reported separately; no erosion/dilation correction.",
            "claim_scope": "A spatial coverage diagnostic, not a replacement for raw IoU or the existing Linux >= 0.99 source-silhouette gate.",
        },
        "summary": summary,
        "glyphs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "screensaver/native-catalog.json")
    parser.add_argument("--label", default="native64")
    parser.add_argument(
        "--published-actual",
        help="Planned repository-relative copied atlas path; same bytes required",
    )
    parser.add_argument(
        "--published-source",
        help="Planned repository-relative copied oracle path; same bytes required",
    )
    parser.add_argument("--artifact-receipt", type=Path)
    parser.add_argument(
        "--ci-artifact-receipt",
        type=Path,
        help="Original downloaded execution receipt, when publication uses a normalized copy",
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--workflow-run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite existing diagnostic: {args.out}")
    result = analyze(args.actual, args.source, args.catalog, args.label)
    if args.published_actual or args.published_source:
        if not args.published_actual or not args.published_source:
            parser.error("Supply both published artifact paths")
        for path in (args.published_actual, args.published_source):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                parser.error("Published paths must stay within the repository")
        result["published_artifacts"] = {
            "actual": {"path": Path(args.published_actual).as_posix(), "sha256": sha(args.actual)},
            "source": {"path": Path(args.published_source).as_posix(), "sha256": sha(args.source)},
            "copy_contract": "These paths identify byte-identical publication destinations; observations retain their original input paths.",
        }
    if args.artifact_receipt:
        result["native_execution_receipt"] = identity(args.artifact_receipt)
    if args.ci_artifact_receipt:
        result["original_ci_execution_receipt"] = identity(args.ci_artifact_receipt)
    if args.source_commit:
        result["compiled_source_commit"] = args.source_commit
    if args.workflow_run:
        result["workflow_run"] = args.workflow_run
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"out": str(args.out), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
