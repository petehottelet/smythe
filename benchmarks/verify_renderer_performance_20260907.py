"""Independently reproduce the frozen renderer/control arithmetic review.

Uses only the Python standard library. Reads retained evidence; never launches
a browser or alters an existing receipt. Run with --out NEW_REVIEW_JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = (("classic", 1), ("3d", 1), ("3d", 2), ("classic", 2), ("classic", 3), ("3d", 3))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(left, right):
    _require(
        math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7),
        f"Arithmetic mismatch: {left}, {right}",
    )


def _quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    index = math.floor(position)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (
        position - index
    )


def _load(path):
    return json.loads(path.read_bytes())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gl_identity(browser):
    gl = browser["gl"]
    gpu = browser["systemInfo"]["gpu"]
    _require(
        browser["headless"] is True and "RTX 4090" in gl["unmaskedRenderer"], "GL context mismatch"
    )
    _require(
        any(
            device.get("vendorId") == 4318 and device.get("deviceId") == 9860
            for device in gpu["devices"]
        ),
        "Matching physical device missing",
    )
    device = next(
        device
        for device in gpu["devices"]
        if device.get("vendorId") == 4318 and device.get("deviceId") == 9860
    )
    auxiliary = gpu["auxAttributes"]["glRenderer"]
    # Chrome's auxiliary string appends the observed device-driver version;
    # the actual WebGL context exposes the same ANGLE identity without it.
    with_driver = gl["unmaskedRenderer"][:-1] + "-" + device["driverVersion"] + ")"
    _require(auxiliary in (gl["unmaskedRenderer"], with_driver), "Auxiliary GL mismatch")
    return gl["unmaskedRenderer"], gl["unmaskedVendor"], auxiliary


def verify_campaign(root=ROOT, *, source_root=None):
    """Return the deterministic review of the six primary and three control runs."""
    root = Path(root)
    source_root = root if source_root is None else Path(source_root)
    directory = root / "benchmarks/results"
    plan = _load(directory / "glyph_rain_regl_20260907_followup1_plan.json")
    control_plan = _load(directory / "glyph_rain_regl_20260907_cadence_plan.json")
    aggregate_path = directory / "glyph_rain_regl_20260907_f1.json"
    aggregate = _load(aggregate_path)
    source_files = {}
    for group in (plan["sourceSha256"], plan["harnessSha256"]):
        for name, digest in group.items():
            path = f"screensaver/svg-preview/{name}"
            _require(
                path not in source_files or source_files[path] == digest,
                "Inconsistent source declarations",
            )
            source_files[path] = digest
    source_files[plan["protocol"]["path"]] = plan["protocol"]["sha256"]
    for path, digest in control_plan["sourceSha256"].items():
        _require(
            path not in source_files or source_files[path] == digest,
            "Primary/control source disagreement",
        )
        source_files[path] = digest
    for path, digest in source_files.items():
        resolved = (source_root / path).resolve()
        _require(resolved.is_relative_to(source_root.resolve()), "Source path escaped root")
        _require(_sha(resolved) == digest, f"Frozen source bytes mismatch: {path}")
    rows, previous, identity, gl_identity = [], None, None, None
    for position, (preset, repetition) in enumerate(ORDER, 1):
        path = directory / f"glyph_rain_regl_20260907_f1_r{repetition}_{preset}.json"
        record = _load(path)
        _require(
            record["status"] == "completed" and record["validation"]["valid"] is True,
            "Primary invalid",
        )
        _require(
            record["scenario"]
            == {"preset": preset, "repetition": repetition, "position": position},
            "Order mismatch",
        )
        _require(
            record["sourceSha256"] == plan["sourceSha256"]
            and record["harnessSha256"] == plan["harnessSha256"],
            "Frozen hash mismatch",
        )
        _require(record["servedSourceSha256"] == record["sourceSha256"], "Served hash mismatch")
        _require(
            record["sourceStable"] and record["harnessStable"] and record["protocolStable"],
            "Source drift",
        )
        _require(record["protocolSha256"] == plan["protocol"]["sha256"], "Protocol drift")
        _require(
            not record["browserErrors"] and not record["invalidReason"], "Browser invalidation"
        )
        _require(
            record["warmupSeconds"] == 5
            and record["requestedDurationSeconds"] == 60
            and record["measuredDurationSeconds"] >= 60,
            "Duration mismatch",
        )
        startup = record["startup"]
        _require(
            startup["simulationTick"] == 0 and startup["rainTime"] == 0 and startup["paused"],
            "Startup advanced",
        )
        initial = record["initialStats"]
        _require(
            initial["viewport"]
            == {
                "width": 1920,
                "height": 1080,
                "physicalWidth": 1440,
                "physicalHeight": 810,
                "dpr": 1,
                "renderScale": 0.75,
            },
            "Viewport mismatch",
        )
        config = initial["config"]
        _require(
            config["originalMix"] == 10
            and config["numColumns"] == 80
            and config["density"] == 1
            and config["fps"] == 60,
            "Scene mismatch",
        )
        _require(
            initial["referenceGlyphCount"] == 56
            and initial["catalogCount"] == 192
            and initial["totalVisibleShapes"] == 248,
            "Catalog mismatch",
        )
        _require(config == record["finalStats"]["config"], "Settings changed")
        browser = record["browser"]
        current_gl = _gl_identity(browser)
        if gl_identity is not None:
            _require(current_gl == gl_identity, "Primary GL identity changed")
        gl_identity = current_gl
        _require(
            browser["headless"] and "RTX 4090" in browser["gl"]["unmaskedRenderer"],
            "Backend mismatch",
        )
        _require(
            any(
                device.get("vendorId") == 4318 for device in browser["systemInfo"]["gpu"]["devices"]
            ),
            "NVIDIA device missing",
        )
        current_identity = (
            browser["version"],
            browser["commandLine"]["comparisonArguments"],
            record["host"],
        )
        if identity is not None:
            _require(current_identity == identity, "Browser or host declaration changed")
        identity = current_identity
        if previous is not None:
            _require(record["attemptStartedAt"] >= previous, "Overlapping attempts")
        previous = record["attemptFinishedAt"]
        _require(record["attemptStartedAt"] < previous, "Invalid attempt chronology")
        intervals = record["samples"]["frameIntervalMs"]
        cpu = record["samples"]["cpuSubmissionMs"]
        _require(
            len(cpu) == len(intervals) + 1
            and all(math.isfinite(value) and value > 0 for value in intervals),
            "Invalid intervals",
        )
        _require(all(math.isfinite(value) and value >= 0 for value in cpu), "Invalid CPU samples")
        duration = math.fsum(intervals) / 1000
        _close(duration, record["measuredDurationSeconds"])
        average, p95, cpu_p95 = (
            len(intervals) / duration,
            _quantile(intervals, 0.95),
            _quantile(cpu, 0.95),
        )
        _close(average, record["averageFramesPerSecond"])
        _close(p95, record["frameIntervalMs"]["p95"])
        _close(cpu_p95, record["cpuSubmissionMs"]["p95"])
        target_passed = p95 <= 16.7 and average >= 59.5
        _require(target_passed is record["validation"]["targetPassed"], "Gate mismatch")
        _require(not target_passed, "Observed campaign differs from the target-miss review")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
                "preset": preset,
                "repetition": repetition,
                "durationSeconds": duration,
                "averageDrawsPerSecond": average,
                "frameIntervalP95Ms": p95,
                "cpuSubmissionP95Ms": cpu_p95,
                "targetPassed": False,
            }
        )
    controls = []
    for repetition in range(1, 4):
        path = directory / f"glyph_rain_regl_20260907_cadence_r{repetition}.json"
        record = _load(path)
        _require(
            record["status"] == "completed"
            and record["sourcesUnchanged"]
            and not record["samples"]["errors"],
            "Control invalid",
        )
        _require(
            type(record["repetition"]) is int and record["repetition"] == repetition,
            "Control order mismatch",
        )
        _require(
            previous <= record["attemptStartedAt"] < record["attemptFinishedAt"],
            "Control chronology mismatch",
        )
        previous = record["attemptFinishedAt"]
        _require(_gl_identity(record["browser"]) == gl_identity, "Control GL identity mismatch")
        startup = record["startup"]
        _require(
            startup["width"] == 1920
            and startup["height"] == 1080
            and startup["dpr"] == 1
            and startup["visibility"] == "visible"
            and startup["reducedMotion"] is True,
            "Control viewport/readiness mismatch",
        )
        expected = {
            key.split("screensaver/svg-preview/")[1]: value
            for key, value in control_plan["sourceSha256"].items()
            if key.startswith("screensaver/svg-preview/")
            and not key.endswith("verify-raf-control.mjs")
        }
        _require(record["sourceSha256"] == expected, "Control source mismatch")
        _require(
            record["protocolSha256"]
            == control_plan["sourceSha256"]["benchmarks/renderer_cadence_control_20260907.md"],
            "Control protocol mismatch",
        )
        _require(
            record["browser"]["version"] == identity[0]
            and record["browser"]["commandLine"]["comparisonArguments"] == identity[1],
            "Control browser mismatch",
        )
        _require(
            record["host"]["powerProfile"] == identity[2]["powerProfile"]
            and record["host"]["workloads"] == identity[2]["competingWorkloads"],
            "Control host mismatch",
        )
        samples = record["samples"]
        _require(samples["raw"][0] - samples["warmupStartedAt"] >= 5000, "Control warmup too short")
        _require(
            set(samples["selected"]).issubset(samples["raw"]),
            "Control selected samples not in source stream",
        )
        for name in ("raw", "selected"):
            values = samples[name]
            intervals = [right - left for left, right in zip(values, values[1:])]
            _require(
                all(math.isfinite(value) and value > 0 for value in intervals),
                "Invalid control intervals",
            )
            elapsed = values[-1] - values[0]
            _require(elapsed >= 60000, "Control duration too short")
            _close(record[name]["durationMs"], elapsed)
            _close(record[name]["averageCallbacksPerSecond"], 1000 * len(intervals) / elapsed)
            _close(record[name]["intervalMs"]["p95"], _quantile(intervals, 0.95))
        controls.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
                "repetition": repetition,
                "raw": record["raw"],
                "selected": record["selected"],
                "omittedCallbacks": len(samples["raw"]) - len(samples["selected"]),
            }
        )
    _require(
        aggregate["status"] == "completed"
        and not aggregate["targetPassed"]
        and not aggregate["hardwareTargetPassed"]
        and not aggregate["claimable"],
        "Aggregate status mismatch",
    )
    return {
        "schemaVersion": 2,
        "status": "passed",
        "evidenceStatus": "reviewed-diagnostic-target-not-met",
        "claimable": False,
        "primaryTargetPassed": False,
        "method": "Independent Python standard-library raw interval summation and inclusive-linear quantile recomputation; actual frozen source/protocol byte hashes; chronological primary/control schedule; startup/config/canvas/catalog checks; matching active GL, physical device and launch identity in primary/control; control timestamps increasing with selected subset. No physical presentation, GPU timing or idle-host claim.",
        "sourceVerification": {
            "policy": "SHA-256 of actual raw bytes in the explicit source checkout",
            "files": source_files,
        },
        "previousReview": {
            "path": "benchmarks/results/glyph_rain_regl_20260907_f1_review.json",
            "sha256": _sha(directory / "glyph_rain_regl_20260907_f1_review.json"),
            "scope": "Initial arithmetic review compared declared source hashes; this v2 also checks actual source bytes and explicit control environment/chronology.",
        },
        "rendererFreezeCommit": "41e4f9ddf2c659e3c93f0bb0ebd5b9582bbe8b7a",
        "observedLaterHead": "5c1939bef718849213f5ef27296b6cbf898a2f25",
        "headChangeScope": "Completion-tracker documentation only; all frozen source bytes match",
        "host": identity[2],
        "primary": rows,
        "cadenceControl": controls,
        "primaryAggregate": {
            "path": "benchmarks/results/glyph_rain_regl_20260907_f1.json",
            "sha256": _sha(aggregate_path),
        },
        "interpretation": "Blank-page source cadence closely matches rain. This supports an environment-cadence explanation, without proving its cause or changing failed primary gates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Checkout retaining the frozen primary and control source bytes",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; retained reviews cannot be overwritten")
    review = verify_campaign(source_root=args.source_root)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(review, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"status": review["status"], "sha256": _sha(args.out), "output": str(args.out)})
    )


if __name__ == "__main__":
    main()
