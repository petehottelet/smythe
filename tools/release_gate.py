"""Require a version-matched release tag and successful main CI at its exact SHA."""

import json
import os
from pathlib import Path
import re
import subprocess
import tomllib


def validate_release(ref, version, sha, runs, repository):
    if ref != f"refs/tags/v{version}":
        raise ValueError("Publish only the tag matching pyproject.toml's version")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected the full checked-out commit SHA")
    for run in runs:
        if (run.get("head_sha") == sha and run.get("event") == "push"
                and run.get("head_branch") == "main"
                and run.get("head_repository", {}).get("full_name") == repository
                and run.get("status") == "completed" and run.get("conclusion") == "success"):
            return run["html_url"]
    raise ValueError("No successful main-branch CI run at the exact release commit")


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    ref, repository = os.environ["GITHUB_REF"], os.environ["GITHUB_REPOSITORY"]
    if ref != f"refs/tags/v{version}":
        raise ValueError("Publish only the tag matching pyproject.toml's version")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository identity")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    pages = subprocess.check_output([
        "gh", "api", "--paginate", "--slurp",
        f"repos/{repository}/actions/workflows/ci.yml/runs?head_sha={sha}&event=push&per_page=100",
    ], text=True)
    runs = [run for page in json.loads(pages) for run in page["workflow_runs"]]
    print("Qualified release CI: " + validate_release(ref, version, sha, runs, repository))


if __name__ == "__main__":
    main()
