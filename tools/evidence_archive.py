"""Build and verify the SHA-256-pinned archive of retired benchmark evidence.

Retired records leave the tree but stay verifiable. The committed pointer pins
the archive's SHA-256 and describes each group. The archive's manifest pins
every member and the commit it was read from, so anyone can rebuild the
archive from git history and compare it byte for byte.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "benchmarks/archive/evidence-2026-09-25.json"
STATUSES = frozenset({"superseded", "diagnostic", "historical"})
GROUP_FIELDS = ("id", "title", "status", "description")
DATE = (2020, 1, 1, 0, 0, 0)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _readme(manifest: dict) -> bytes:
    lines = [f"# {manifest['name']}", "",
             "Benchmark evidence retired from the Smythe repository tree, byte for",
             f"byte as it existed at {manifest['source']['ref']} "
             f"(commit {manifest['source']['commit']}).",
             "MANIFEST.json lists every file with its size and SHA-256; the files",
             "are under files/ at their original repository paths.", ""]
    for group in manifest["groups"]:
        lines += [f"## {group['title']} ({group['status']})", "", group["description"], "",
                  f"{group['files']} files, {group['bytes']:,} bytes.", ""]
    lines += ["Verify with `python tools/evidence_archive.py verify ARCHIVE --against-git`",
              "from a Smythe checkout.", ""]
    return "\n".join(lines).encode()


def _members(manifest: dict, files: dict[str, bytes]) -> dict[str, bytes]:
    base = manifest["name"] + "/"
    members = {base + "files/" + path: data for path, data in files.items()}
    members[base + "MANIFEST.json"] = _json(manifest)
    members[base + "README.md"] = _readme(manifest)
    return members


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def build(spec: dict, output: Path, *, root: Path = ROOT) -> dict:
    """Archive the files each group's patterns match at ``spec["ref"]``; return the pointer."""
    commit = git(root, "rev-parse", "--verify", spec["ref"] + "^{commit}").decode().strip()
    tracked = [p for p in git(root, "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0") if p]
    files: dict[str, bytes] = {}
    groups, entries = [], []
    for group in spec["groups"]:
        if group["status"] not in STATUSES:
            raise ValueError(f"Unknown evidence status: {group['status']}")
        paths: set[str] = set()
        for pattern in group["patterns"]:
            matched = {p for p in tracked if fnmatch.fnmatchcase(p, pattern)}
            if not matched:
                raise ValueError(f"{pattern} matches no file at {spec['ref']}")
            paths |= matched
        if overlap := sorted(paths & files.keys()):
            raise ValueError(f"Files belong to more than one group: {overlap}")
        for path in sorted(paths):
            files[path] = git(root, "cat-file", "blob", f"{commit}:{path}")
            entries.append({"path": path, "group": group["id"], "bytes": len(files[path]),
                            "sha256": _sha(files[path])})
        groups.append({**{key: group[key] for key in GROUP_FIELDS}, "files": len(paths),
                       "bytes": sum(len(files[p]) for p in paths)})
    manifest = {"version": 1, "name": spec["name"],
                "source": {"ref": spec["ref"], "commit": commit},
                "groups": groups, "files": sorted(entries, key=lambda e: e["path"])}
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{spec['name']}.zip"
    _write_zip(path, _members(manifest, files))
    data = path.read_bytes()
    return {"version": 1, "archive": path.name, "bytes": len(data), "sha256": _sha(data),
            "release_asset": spec["release_asset"], "source": manifest["source"],
            "manifest_sha256": _sha(_json(manifest)), "groups": groups,
            "files": len(entries), "total_bytes": sum(e["bytes"] for e in entries)}


def verify(archive: Path, pointer: dict, *, against_git: bool = False, root: Path = ROOT) -> dict:
    """Check the archive against its pointer and, optionally, against git history."""
    data = archive.read_bytes()
    if (len(data), _sha(data)) != (pointer["bytes"], pointer["sha256"]):
        raise ValueError("Archive size or SHA-256 does not match the pointer")
    base = pointer["archive"].removesuffix(".zip") + "/"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate members")
        members = {name: bundle.read(name) for name in names}
    raw = members.get(base + "MANIFEST.json")
    if raw is None or _sha(raw) != pointer["manifest_sha256"]:
        raise ValueError("Archive manifest does not match the pointer")
    manifest = json.loads(raw)
    if manifest["source"] != pointer["source"] or manifest["groups"] != pointer["groups"]:
        raise ValueError("Archive source or groups do not match the pointer")
    files = {e["path"]: e for e in manifest["files"]}
    if len(files) != pointer["files"] or set(names) != set(_members(manifest, {p: b"" for p in files})):
        raise ValueError("Archive inventory does not match its manifest")
    if members[base + "README.md"] != _readme(manifest):
        raise ValueError("Archive README does not match its manifest")
    for path, entry in files.items():
        content = members[base + "files/" + path]
        if (len(content), _sha(content)) != (entry["bytes"], entry["sha256"]):
            raise ValueError(f"Archived file checksum mismatch: {path}")
        if against_git and git(root, "cat-file", "blob", f"{manifest['source']['commit']}:{path}") != content:
            raise ValueError(f"Archived file differs from git history: {path}")
    return {"archive": pointer["archive"], "sha256": pointer["sha256"], "files": len(files),
            "checked_against_git": against_git}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="build an archive from a JSON spec")
    make.add_argument("spec", type=Path)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--pointer", type=Path, help="also write the pointer JSON here")
    check = commands.add_parser("verify", help="verify an archive against its pointer")
    check.add_argument("archive", type=Path)
    check.add_argument("--pointer", type=Path, default=POINTER)
    check.add_argument("--against-git", action="store_true",
                       help="also compare every file with the recorded commit")
    args = parser.parse_args(argv)
    if args.command == "build":
        pointer = build(json.loads(args.spec.read_text(encoding="utf-8")), args.out)
        if args.pointer:
            args.pointer.write_bytes(_json(pointer))
        print(json.dumps({key: pointer[key] for key in ("archive", "bytes", "sha256", "files")}))
    else:
        pointer = json.loads(args.pointer.read_text(encoding="utf-8"))
        print(json.dumps(verify(args.archive, pointer, against_git=args.against_git)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
