"""Tiny read-only filesystem MCP server used by example 05.

Serves files under the directory given as its first argument.
Requires the mcp extra: pip install smythe[mcp]
"""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()

server = FastMCP("files")


def _safe(path: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"path {path!r} escapes the served directory")
    return resolved


@server.tool()
def list_files() -> str:
    """List the files available in the served directory."""
    return "\n".join(sorted(p.name for p in ROOT.iterdir() if p.is_file()))


@server.tool()
def read_file(name: str) -> str:
    """Read one file from the served directory."""
    return _safe(name).read_text(encoding="utf-8")


if __name__ == "__main__":
    server.run(transport="stdio")
