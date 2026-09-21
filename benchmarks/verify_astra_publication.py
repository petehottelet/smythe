"""Reproduce the sealed Astra main evidence and human supplement offline.

Run with python -m benchmarks.verify_astra_publication --out NEW_DIRECTORY.
Only extracted copies of the ledgers are opened. Published evidence is read-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "benchmarks/results/astra_20260913_main"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_verified(archive, manifest, destination):
    require(digest(archive.read_bytes()) == manifest["archive_sha256"], "Archive hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(len(names) == len(set(names)), "Duplicate archive members")
        require(set(names) == set(manifest["members"]), "Archive inventory mismatch")
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                    and ":" not in name, "Unsafe archive member")
            target = (destination / name).resolve()
            require(target.is_relative_to(destination.resolve()), "Escaping archive member")
            data = source.read(name)
            expected = manifest["members"][name]
            require(len(data) == expected["bytes"] and digest(data) == expected["sha256"],
                    "Member hash mismatch: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(data)
    return len(names)


def verify(out):
    from benchmarks.astra_main_evidence import review_main
    from benchmarks.render_readme_charts import render_astra_differences, render_astra_distributions

    out.mkdir(parents=True, exist_ok=False)
    manifest = read(PUBLIC / "archive-manifest.json")
    extracted = out / "extracted"
    count = extract_verified(PUBLIC / "evidence.zip", manifest, extracted)
    computed, audited = review_main(
        main_directory=extracted / "main", continuation_directory=extracted / "main-continuation",
        judge_directory=extracted / "judging", bindings_path=extracted / "calibration/main-judgment-bindings.json",
        source_archive=extracted / "sources/typed-main-source.zip",
        continuation_source_archive=extracted / "sources/continuation-source.zip",
    )
    require(computed == read(PUBLIC / "analysis.json"), "Analysis differs from committed results")
    reviewed = read(PUBLIC / "review.json")
    require(reviewed["analysis_sha256"] == digest((PUBLIC / "analysis.json").read_bytes()),
            "Review is not bound to analysis")
    require(reviewed["claimable"] and not reviewed["known_measurement_defects"], "Evidence not claimable")
    for key in ("validated_outcomes", "validated_request_policies", "validated_task_inputs",
                "main_workflow_lower_nanousd", "main_workflow_upper_nanousd", "held_unknown_nanousd"):
        require(audited[key] == reviewed[key], "Audit differs: " + key)
    samples = read(PUBLIC / "human-main-manifest.json")
    response = read(PUBLIC / "human-main-response.json")
    human = reviewed["human_main_review"]
    require(human["manifest_sha256"] == digest((PUBLIC / "human-main-manifest.json").read_bytes()),
            "Human sample binding differs")
    require(human["response_sha256"] == digest((PUBLIC / "human-main-response.json").read_bytes()),
            "Human response binding differs")
    require(response["sample_manifest_sha256"] == human["manifest_sha256"], "Unbound human response")
    outcomes = {}
    for stage in ("main", "main-continuation"):
        for path in (extracted / stage).glob("*.outcome.json"):
            row = read(path)
            require(row["run_id"] not in outcomes, "Duplicate outcome")
            outcomes[row["run_id"]] = row
    ratings = {row["sample_id"]: row for row in response["ratings"]}
    require(len(ratings) == len(response["ratings"]) == len(samples), "Human sample inventory differs")
    reviewed_ids = set()
    for sample in samples:
        rating = ratings[sample["sample_id"]]
        require(digest(sample["output"].encode()) == sample["output_sha256"]
                == rating["output_sha256"], "Human output hash mismatch")
        require(rating["accepted"] is True and rating["score"] == 4, "Human decision differs")
        for run_id in sample["run_ids"]:
            require(outcomes[run_id]["output"] == sample["output"], "Human output differs from outcome")
            reviewed_ids.add(run_id)
    require(reviewed_ids == set(audited["disputed_output_run_ids"]), "Unreviewed disputes")
    charts = {}
    for name, render in (("astra_workflows.svg", render_astra_distributions),
                         ("astra_differences.svg", render_astra_differences)):
        data = render().encode()
        # Git may check out SVG text with CRLF on Windows; compare canonical LF bytes.
        committed = (ROOT / "assets/benchmarks" / name).read_bytes().replace(b"\r\n", b"\n")
        require(data == committed, "Chart differs: " + name)
        (out / name).write_bytes(data)
        charts[name] = digest(data)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "smythe", "benchmarks"],
        cwd=ROOT, text=True,
    ).split("\0")
    sources = [ROOT / name for name in set(names) if name.endswith(".py")]
    inputs = [PUBLIC / name for name in ("archive-manifest.json", "analysis.json", "review.json",
                                        "human-main-manifest.json", "human-main-response.json")]
    result = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "kind": "offline-reproduction-not-a-new-campaign", "provider_calls": 0,
              "archive_sha256": manifest["archive_sha256"], "verified_archive_members": count,
              "analysis_reproduced_exactly": True, "workflows": computed["workflows"],
              "accepted_primary": sum(arm["accepted"] for arm in computed["arms"].values()),
              "independent_tasks": computed["independent_tasks"],
              "request_policies_verified": audited["validated_request_policies"],
              "human_samples_bound": len(samples), "unreviewed_disputes": 0,
              "held_unknown_nanousd": audited["held_unknown_nanousd"],
              "charts_sha256_lf": charts, "claim_scope": reviewed["claim_scope"],
              "withheld_claims": reviewed["withheld_claims"],
              "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in inputs},
              "source_sha256_lf": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes().replace(b"\r\n", b"\n"))
                                   for p in sorted(sources)}}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.out)
    print(f"Verified {report['workflows']} outcomes, {report['verified_archive_members']} members, "
          f"{report['human_samples_bound']} human samples; no provider calls")
