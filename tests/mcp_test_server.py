"""Minimal MCP stdio server used by the end-to-end runtime tests.

Run directly (python tests/mcp_test_server.py); the tests spawn it as a
stdio subprocess via MCPServerSpec(command=sys.executable, args=[this file]).
"""

import os
import time

from mcp.server.fastmcp import FastMCP

server = FastMCP("smythe-test-server")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def get_env(name: str) -> str:
    """Return the value of an environment variable, or <unset>."""
    return os.environ.get(name, "<unset>")


@server.tool()
def slow(seconds: float) -> str:
    """Sleep, then return. Used to exercise timeouts."""
    time.sleep(seconds)
    return "finally done"


if __name__ == "__main__":
    server.run(transport="stdio")
