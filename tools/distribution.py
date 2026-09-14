"""Audit distribution contents and compare checkout/source-archive wheels.

Run from the repository root. The checked-in file manifest is an exact
allowlist: update it deliberately after adding or removing package/test tools.
"""

from __future__ import annotations

import argparse
from collections import Counter
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools/distribution-files.json"
SOURCE_LIMIT = 1_000_000
WHEEL_LIMIT = 400_000
ROOT_FILES = {"pyproject.toml", "hatch_build.py", "README.md", "CHANGELOG.md", "LICENSE", ".gitignore"}
DATA_FILES = {"smythe/py.typed", "tests/distribution_profile.json", "tools/distribution-files.json"}
FORBIDDEN = {"00_project_files", "__pycache__", "tmp", "temp", "dist", "build", "output",
             ".venv", "venv", "env", ".git", ".verify-venv"}


def source_path_allowed(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if any(part in FORBIDDEN or part.startswith(".env") for part in path.parts):
        return False
    if name in ROOT_FILES | DATA_FILES:
        return True
    return path.parts[0] in {"smythe", "tests", "tools"} and path.suffix == ".py"


def write_manifest(root: Path = ROOT) -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    names = sorted({name for name in tracked if name and source_path_allowed(name)}
                   | {"tools/distribution-files.json"})
    missing = [name for name in ROOT_FILES if name not in names]
    if missing:
        raise ValueError(f"Missing tracked build inputs: {missing}")
    (root / "tools/distribution-files.json").write_text(
        json.dumps({"version": 1, "source_files": names}, indent=2) + "\n", encoding="utf-8",
    )


def _record(path: Path, members: dict[str, bytes]) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unpacked_bytes": sum(map(len, members.values())), "members": len(members),
            "top_level_counts": dict(sorted(Counter(name.split('/')[0] for name in members).items()))}


def wheel_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel members")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError("Unsafe wheel member")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Wheel symlink is not allowed")
        return {name: archive.read(name) for name in names}


def audit(sdist: Path, wheel: Path, *, root: Path = ROOT) -> dict:
    expected = set(json.loads((root / "tools/distribution-files.json").read_text(encoding="utf-8"))["source_files"])
    if not all(source_path_allowed(name) for name in expected):
        raise ValueError("Distribution manifest contains a forbidden source path")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if sdist.stat().st_size > SOURCE_LIMIT or wheel.stat().st_size > WHEEL_LIMIT:
        raise ValueError("Distribution exceeds its compressed size ceiling")
    source = {}
    prefix = f"smythe-{version}/"
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError(f"Unexpected source archive member: {member.name}")
            name = member.name[len(prefix):]
            if name in source:
                raise ValueError("Duplicate source archive member")
            if name != "PKG-INFO" and (name not in expected or not source_path_allowed(name)):
                raise ValueError(f"Unapproved source member: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable source member")
            source[name] = stream.read()
    if set(source) != expected | {"PKG-INFO"}:
        raise ValueError(f"Source inventory mismatch: {sorted((expected | {'PKG-INFO'}) ^ set(source))}")
    package = wheel_members(wheel)
    metadata_dir = f"smythe-{version}.dist-info"
    allowed_metadata = {f"{metadata_dir}/{name}" for name in
                        ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE")}
    expected_package = {name for name in expected if name.startswith("smythe/")}
    if set(package) != expected_package | allowed_metadata:
        raise ValueError(f"Wheel inventory mismatch: {sorted(set(package) ^ (expected_package | allowed_metadata))}")
    for name in expected_package:
        if source[name] != package[name]:
            raise ValueError(f"Source/wheel package bytes differ: {name}")
    metadata = package[f"{metadata_dir}/METADATA"]
    if metadata != source["PKG-INFO"]:
        raise ValueError("Source/wheel metadata differs")
    parsed = BytesParser().parsebytes(metadata)
    if parsed["Name"] != "smythe" or parsed["Version"] != version:
        raise ValueError("Incorrect distribution identity")
    if parsed["Description-Content-Type"] != "text/markdown":
        raise ValueError("Missing Markdown description")
    if b"smythe = smythe.cli:main" not in package[f"{metadata_dir}/entry_points.txt"]:
        raise ValueError("Missing Smythe CLI entry point")
    if f"https://raw.githubusercontent.com/petehottelet/smythe/v{version}/assets/wordmark.svg".encode() not in metadata:
        raise ValueError("README image was not resolved to its release tag")
    return {"source": _record(sdist, source), "wheel": _record(wheel, package)}


def compare_wheels(first: Path, second: Path) -> None:
    a, b = wheel_members(first), wheel_members(second)
    if a != b:
        changed = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
        raise ValueError(f"Rebuilt wheel differs: {changed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    if args.compare:
        compare_wheels(*args.compare)
        print("Rebuilt wheel: every member and metadata byte matches")
    if args.dist:
        sdists, wheels = list(args.dist.glob("*.tar.gz")), list(args.dist.glob("*.whl"))
        if len(sdists) != 1 or len(wheels) != 1:
            parser.error("Audit a fresh directory containing exactly one source archive and wheel")
        report = json.dumps(audit(sdists[0], wheels[0]), indent=2) + "\n"
        if args.out:
            args.out.write_text(report, encoding="utf-8")
        print(report, end="")


if __name__ == "__main__":
    main()
