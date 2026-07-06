"""MCP client runtime — agents consume MCP servers as tool sources.

Implements the ToolRuntime interface (plans/04 section 5.4) on the
official ``mcp`` SDK.  Install with the extra: ``pip install smythe[mcp]``.

This module itself imports only the standard library and smythe — the
SDK is imported lazily when a session opens, so declaring MCPServerSpec
in configs never requires the extra.

Secrets: never put them in ``env`` (literal, serialized as-is in YAML
and checkpoints).  List the variable *names* in ``env_passthrough``;
values are read from the running process's environment at session
start, so nothing secret is ever written to disk.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from smythe.skills import SkillRef
from smythe.tools import ToolCall, ToolResult, ToolRuntime, ToolSession, ToolSpec, content_to_text

if TYPE_CHECKING:
    from smythe.agent import Agent
    from smythe.registry import Registry

logger = logging.getLogger("smythe.mcp")

# Above this many tools without an allowlist, every loop iteration sends
# a schema payload big enough to hurt cost and tool selection.
LARGE_TOOLSET_WARNING_THRESHOLD = 20

_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class MCPConfigError(Exception):
    """Raised for invalid MCP server configuration."""


@dataclass(frozen=True)
class MCPServerSpec:
    """Declaration of one MCP server an agent may use.

    Attributes:
        name: Namespace prefix for the server's tools ("fs" -> "fs.read_file").
        transport: "stdio" (subprocess) or "http" (streamable HTTP).
        command / args: Executable and arguments (stdio only).
        env: Literal, NON-SECRET environment values (paths, flags).
            Serialized as-is in YAML and checkpoints.
        env_passthrough: Names of environment variables whose values are
            resolved from os.environ when the session opens.  Use this
            for tokens and keys.
        url: Endpoint (http only).
        allowed_tools: Allowlist of remote tool names; None means all
            (discouraged for large servers).
        tool_timeout_s: Per-tool-call timeout.
    """

    name: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    env_passthrough: tuple[str, ...] = ()
    url: str | None = None
    allowed_tools: tuple[str, ...] | None = None
    tool_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not _SERVER_NAME_RE.match(self.name):
            raise MCPConfigError(
                f"MCP server name {self.name!r} must match {_SERVER_NAME_RE.pattern} "
                "(it becomes the tool namespace prefix)"
            )
        if self.transport not in ("stdio", "http"):
            raise MCPConfigError(
                f"MCP server {self.name!r}: transport must be 'stdio' or 'http', "
                f"got {self.transport!r}"
            )
        if self.transport == "stdio" and not self.command:
            raise MCPConfigError(f"MCP server {self.name!r}: stdio transport requires 'command'")
        if self.transport == "http" and not self.url:
            raise MCPConfigError(f"MCP server {self.name!r}: http transport requires 'url'")
        # Normalize sequences that arrive as lists (YAML, checkpoints).
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "env_passthrough", tuple(self.env_passthrough))
        if self.allowed_tools is not None:
            object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))

    def resolve_env(self) -> dict[str, str] | None:
        """Merge literal env with passthrough values from os.environ."""
        missing = [n for n in self.env_passthrough if n not in os.environ]
        if missing:
            raise MCPConfigError(
                f"MCP server {self.name!r}: environment variable(s) "
                f"{', '.join(missing)} listed in env_passthrough are not set"
            )
        env = dict(self.env or {})
        env.update({n: os.environ[n] for n in self.env_passthrough})
        return env or None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (env_passthrough stores names only — no secrets)."""
        data = asdict(self)
        data["args"] = list(self.args)
        data["env_passthrough"] = list(self.env_passthrough)
        if self.allowed_tools is not None:
            data["allowed_tools"] = list(self.allowed_tools)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerSpec:
        allowed = data.get("allowed_tools")
        return cls(
            name=data["name"],
            transport=data["transport"],
            command=data.get("command"),
            args=tuple(data.get("args", ())),
            env=data.get("env"),
            env_passthrough=tuple(data.get("env_passthrough", ())),
            url=data.get("url"),
            allowed_tools=tuple(allowed) if allowed is not None else None,
            tool_timeout_s=data.get("tool_timeout_s", 30.0),
        )


@dataclass
class _DispatchEntry:
    session: Any  # mcp ClientSession
    remote_name: str
    timeout_s: float


class MCPToolSession(ToolSession):
    """Live tool session over one or more connected MCP servers."""

    def __init__(self, tools: list[ToolSpec], dispatch: dict[str, _DispatchEntry]) -> None:
        self._tools = tools
        self._dispatch = dispatch

    @property
    def tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, tool_call: ToolCall) -> ToolResult:
        entry = self._dispatch.get(tool_call.name)
        if entry is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Unknown tool {tool_call.name!r}",
                is_error=True,
            )
        try:
            result = await entry.session.call_tool(
                entry.remote_name,
                tool_call.arguments,
                read_timeout_seconds=timedelta(seconds=entry.timeout_s),
            )
        except Exception as exc:
            return ToolResult(tool_call_id=tool_call.id, content=str(exc), is_error=True)
        return ToolResult(
            tool_call_id=tool_call.id,
            content=content_to_text(result.content),
            is_error=bool(result.isError),
        )


def discover_tools(
    spec: MCPServerSpec, client_session: Any, listing: Any,
) -> tuple[list[ToolSpec], dict[str, _DispatchEntry]]:
    """Convert one server's tool listing into namespaced ToolSpecs + dispatch.

    Applies the allowed_tools filter and logs the large-toolset warning.
    Split out from session opening so it is testable over the SDK's
    in-memory transport.
    """
    remote_tools = list(listing.tools)
    if spec.allowed_tools is not None:
        allowed = set(spec.allowed_tools)
        remote_tools = [t for t in remote_tools if t.name in allowed]
    elif len(remote_tools) > LARGE_TOOLSET_WARNING_THRESHOLD:
        logger.warning(
            "MCP server %r exposes %d tools with no allowed_tools filter; "
            "every loop iteration sends every schema to the model. "
            "Set allowed_tools to the subset this agent needs.",
            spec.name, len(remote_tools),
        )

    tool_specs: list[ToolSpec] = []
    dispatch: dict[str, _DispatchEntry] = {}
    for tool in remote_tools:
        display = f"{spec.name}.{tool.name}"
        tool_specs.append(ToolSpec(
            name=display,
            description=tool.description or "",
            input_schema=tool.inputSchema or {"type": "object"},
        ))
        dispatch[display] = _DispatchEntry(
            session=client_session,
            remote_name=tool.name,
            timeout_s=spec.tool_timeout_s,
        )
    return tool_specs, dispatch


class MCPToolRuntime(ToolRuntime):
    """Opens per-node MCP sessions from each agent's declared servers.

    With no constructor argument, servers come from
    ``agent.profile.mcp_servers``; pass ``servers=[...]`` to give every
    node the same fixed set instead.
    """

    def __init__(self, servers: list[MCPServerSpec] | None = None) -> None:
        self._servers = servers

    def _specs_for(self, agent: Agent | None) -> list[MCPServerSpec]:
        if self._servers is not None:
            return self._servers
        if agent is not None:
            return list(getattr(agent.profile, "mcp_servers", []) or [])
        return []

    def open(self, agent: Agent | None):
        return self._open(agent)

    @asynccontextmanager
    async def _open(self, agent: Agent | None):
        specs = self._specs_for(agent)
        if not specs:
            yield MCPToolSession([], {})
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise ImportError(
                "MCP servers are configured but the mcp SDK is not installed. "
                "Install the extra: pip install smythe[mcp]"
            ) from exc

        stack = AsyncExitStack()
        try:
            all_tools: list[ToolSpec] = []
            dispatch: dict[str, _DispatchEntry] = {}
            for spec in specs:
                if spec.transport == "stdio":
                    params = StdioServerParameters(
                        command=spec.command,
                        args=list(spec.args),
                        env=spec.resolve_env(),
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                else:
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(spec.url)
                    )
                client_session = await stack.enter_async_context(ClientSession(read, write))
                await client_session.initialize()
                listing = await client_session.list_tools()
                tool_specs, entries = discover_tools(spec, client_session, listing)
                all_tools.extend(tool_specs)
                dispatch.update(entries)
            yield MCPToolSession(all_tools, dispatch)
        finally:
            # Teardown must survive cancellation (timeout_s cancels the
            # loop mid-call) or stdio subprocesses leak.  anyio cancel
            # scopes must exit in the task that entered them, so the
            # close runs in-task — no shielding to another task.  Under
            # timeout_s, asyncio.wait_for waits for this cleanup to
            # complete before raising TimeoutError, so subprocesses are
            # reaped before the node fails.
            try:
                await stack.aclose()
            except Exception:
                logger.warning("MCP session teardown failed", exc_info=True)


class MCPSkillProvider:
    """SkillProvider that grounds agent capabilities in their MCP tools.

    Plugs into the registry's existing hydration machinery, so the same
    servers that power execution also drive capability-based assignment:

        provider = MCPSkillProvider()
        registry = Registry(skill_provider=provider)
        provider.attach(registry)

    Allowlisted tools contribute skills statically — no connection is
    made.  For servers without an allowlist, ``await prefetch(agents)``
    once to discover tools live; discovered names are cached per agent.
    """

    def __init__(self) -> None:
        self._registry: Registry | None = None
        self._discovered: dict[str, list[str]] = {}

    def attach(self, registry: Registry) -> None:
        self._registry = registry

    def list_agent_skills(self, agent_id: str) -> list[SkillRef]:
        agent = self._registry.get(agent_id) if self._registry else None
        if agent is None:
            return []
        names: list[str] = []
        for spec in getattr(agent.profile, "mcp_servers", []) or []:
            if spec.allowed_tools is not None:
                names.extend(f"{spec.name}.{t}" for t in spec.allowed_tools)
        names.extend(self._discovered.get(agent_id, []))
        return [SkillRef(name=n, source="mcp") for n in dict.fromkeys(names)]

    async def prefetch(self, agents) -> None:
        """Discover tools for servers that have no allowlist.

        Opens a short-lived session per agent that needs discovery and
        caches the namespaced tool names.  Call once before planning /
        assignment when you rely on non-allowlisted servers.
        """
        for agent in agents:
            specs = [
                s for s in getattr(agent.profile, "mcp_servers", []) or []
                if s.allowed_tools is None
            ]
            if not specs:
                continue
            runtime = MCPToolRuntime(servers=specs)
            async with runtime.open(agent) as session:
                self._discovered[agent.id] = [t.name for t in session.tools]
            if self._registry is not None:
                self._registry.refresh_agent_capabilities(agent.id)
