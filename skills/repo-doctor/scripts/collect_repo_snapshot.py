"""Collect a deterministic release-readiness snapshot of a local repository.

The snapshot is the only repository evidence the audit agents see, so it
must be compact, stable, and safe: files are read but never executed,
key config files are truncated, and likely secret values are redacted
before they can enter the output.

    python collect_repo_snapshot.py <path> [--output snapshot.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", ".nox", ".idea",
    ".vscode", "dist", "build", "htmlcov", ".eggs",
}

MAX_TREE_ENTRIES = 400
MAX_IMPORTANT_FILES = 20
MAX_BYTES_PER_FILE = 4000

_LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rs": "Rust", ".go": "Go", ".rb": "Ruby", ".java": "Java", ".c": "C",
    ".cpp": "C++", ".cs": "C#", ".sh": "Shell", ".ps1": "PowerShell",
}

_ECOSYSTEM_MARKERS = {
    "python": ("pyproject.toml", "setup.py", "setup.cfg"),
    "javascript": ("package.json",),
    "rust": ("Cargo.toml",),
    "go": ("go.mod",),
}

_KEY_NAMES = (
    "pyproject.toml", "setup.py", "setup.cfg", "package.json", "Cargo.toml",
    "go.mod", "pytest.ini", "tox.ini", "noxfile.py", "mkdocs.yml",
)
_KEY_PREFIXES = (
    "README", "CHANGELOG", "LICENSE", "CONTRIBUTING", "SECURITY", "CODE_OF_CONDUCT",
)
_DOCS_INDEXES = ("docs/index.md", "docs/index.rst")

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),                     # Anthropic/OpenAI keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),                # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                          # AWS access key ids
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),              # Slack tokens
    re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['"]([^'"\s]{12,})['"]"""
    ),
)


def _redact(text: str) -> tuple[str, list[str]]:
    """Replace likely secret values, returning clean text and redacted stubs."""
    found: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        value = match.group(1) if match.groups() else match.group(0)
        found.append(value[:4] + "***")
        return match.group(0).replace(value, "[REDACTED]")

    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_sub, text)
    return text, found


def _walk(root: Path) -> list[str]:
    """Relative posix paths of all files under root, sorted, caches pruned."""
    out: list[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        for entry in sorted(current.iterdir(), key=lambda p: p.name):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                out.append(entry.relative_to(root).as_posix())
    return sorted(out)


def _git_head(root: Path) -> tuple[str | None, str | None]:
    """(branch, commit) parsed from .git textually — no git subprocess."""
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None, None
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("ref:"):
        return None, text or None
    ref = text.split(None, 1)[1]
    ref_file = root / ".git" / ref
    commit = None
    if ref_file.is_file():
        commit = ref_file.read_text(encoding="utf-8", errors="replace").strip() or None
    return ref.rsplit("/", 1)[-1], commit


def collect_snapshot(
    root: Path | str,
    *,
    max_tree_entries: int = MAX_TREE_ENTRIES,
    max_important_files: int = MAX_IMPORTANT_FILES,
    max_bytes_per_file: int = MAX_BYTES_PER_FILE,
) -> dict:
    """Build the snapshot dict for a local repository directory."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    rel = _walk(root)
    top_level = {p for p in rel if "/" not in p}
    top_dirs = {p.split("/", 1)[0] for p in rel if "/" in p}
    workflows = sorted(
        p for p in rel
        if p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml"))
    )

    ext_counts: dict[str, int] = {}
    for p in rel:
        lang = _LANGUAGES.get(Path(p).suffix.lower())
        if lang:
            ext_counts[lang] = ext_counts.get(lang, 0) + 1
    languages = sorted(ext_counts, key=lambda k: (-ext_counts[k], k))[:5]

    ecosystems = [
        eco for eco, markers in _ECOSYSTEM_MARKERS.items()
        if any(m in top_level for m in markers)
    ] or ["generic"]

    # Key config files, extracted (truncated) with secrets redacted.
    candidates = [n for n in _KEY_NAMES if n in top_level]
    candidates += sorted(p for p in top_level if p.upper().startswith(_KEY_PREFIXES))
    candidates += workflows
    candidates += [p for p in _DOCS_INDEXES if p in rel]

    important: dict[str, str] = {}
    secret_suspects: list[dict[str, str]] = []
    files_read = bytes_read = 0
    content_truncated = False
    for path in candidates[:max_important_files]:
        raw = (root / path).read_text(encoding="utf-8", errors="replace")
        files_read += 1
        bytes_read += len(raw)
        text = raw[:max_bytes_per_file]
        if len(raw) > max_bytes_per_file:
            text += "\n...[truncated]"
            content_truncated = True
        clean, found = _redact(text)
        important[path] = clean
        secret_suspects += [{"file": path, "redacted": stub} for stub in found]

    # Env-style files are scanned for secrets but never included.
    for path in rel:
        name = path.rsplit("/", 1)[-1].lower()
        if name.startswith(".env") or "credentials" in name:
            raw = (root / path).read_text(encoding="utf-8", errors="replace")
            files_read += 1
            bytes_read += len(raw)
            _, found = _redact(raw[:max_bytes_per_file])
            secret_suspects += [{"file": path, "redacted": stub} for stub in found]

    has_release_workflow = any(
        marker in important.get(w, "").lower()
        for w in workflows
        for marker in ("pypi", "publish", "release")
    ) or any(("release" in w or "publish" in w) for w in workflows)

    signals = {
        "has_tests": "tests" in top_dirs or bool(
            top_level & {"pytest.ini", "tox.ini", "noxfile.py"}
        ) or any(n.startswith(("jest.config", "vitest.config")) for n in top_level),
        "has_ci": bool(workflows),
        "has_license": any(p.upper().startswith("LICENSE") for p in top_level),
        "has_security_policy": any(p.upper().startswith("SECURITY") for p in top_level),
        "has_examples": bool(top_dirs & {"examples", "demo", "demos"})
        or any(p.endswith(".ipynb") for p in rel),
        "has_changelog": any(p.upper().startswith("CHANGELOG") for p in top_level),
        "has_release_workflow": has_release_workflow,
        "has_benchmarks": "benchmarks" in top_dirs
        or any("benchmark" in p.lower() for p in rel),
    }

    release_targets = []
    if "python" in ecosystems:
        release_targets.append("pypi")
    if "javascript" in ecosystems:
        release_targets.append("npm")
    if has_release_workflow:
        release_targets.append("github_release")

    branch, commit = _git_head(root)
    return {
        "repo": {"name": root.name, "url": None, "default_branch": branch, "commit": commit},
        "detected": {
            "languages": languages,
            "ecosystems": ecosystems,
            "release_targets": release_targets or ["unknown"],
        },
        "files": {
            "tree_summary": rel[:max_tree_entries],
            "important_files": important,
        },
        "signals": signals,
        "risks": {"secret_suspects": secret_suspects},
        "limits": {
            "files_read": files_read,
            "bytes_read": bytes_read,
            "truncated": content_truncated
            or len(rel) > max_tree_entries
            or len(candidates) > max_important_files,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="local repository directory")
    parser.add_argument("--output", help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    snapshot = collect_snapshot(Path(args.path))
    text = json.dumps(snapshot, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
