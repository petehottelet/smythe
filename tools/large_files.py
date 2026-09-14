"""Warn when a change introduces a binary blob over 1 MiB; never rewrite evidence."""

import argparse
import json
from pathlib import Path
import subprocess

LIMIT = 1024 * 1024


def new_large_binaries(base: str, head: str = "HEAD", *, root: Path = Path(".")) -> list[dict]:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    # Resolve refs before using them as revision operands or Git expressions.
    base = git("rev-parse", "--verify", base + "^{commit}").decode().strip()
    head = git("rev-parse", "--verify", head + "^{commit}").decode().strip()
    changed = git("diff", "--name-only", "--diff-filter=AM", "-z", base, head).split(b"\0")
    findings = []
    for raw in changed:
        if not raw:
            continue
        name = raw.decode("utf-8")
        blob = git("rev-parse", head + ":" + name).decode().strip()
        size = int(git("cat-file", "-s", blob))
        if size <= LIMIT:
            continue
        content = git("cat-file", "blob", blob)
        try:
            content.decode("utf-8")
            binary = b"\0" in content
        except UnicodeDecodeError:
            binary = True
        if binary:
            findings.append({"path": name, "bytes": size, "blob": blob})
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    findings = new_large_binaries(args.base)
    print(json.dumps(findings, indent=2))
    if findings:
        # Keep untrusted filenames out of GitHub workflow command syntax.
        print("::warning::New binary files exceed 1 MiB. Review the inventory and evidence retention policy.")
