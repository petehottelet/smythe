"""Agent model — persistent identity with capabilities and memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from smythe.mcp import MCPServerSpec


@dataclass
class AgentProfile:
    """Static description of what an agent can do and how it behaves.

    Attributes:
        name: Human-readable agent name.
        persona: System-prompt-level personality / role description.
        capabilities: Tags describing areas of expertise (e.g. "research", "critique").
        mcp_servers: MCP servers this agent may use as tool sources
            (executed via MCPToolRuntime; see smythe.mcp).
    """

    name: str
    persona: str = ""
    capabilities: list[str] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)


@dataclass
class Agent:
    """A persistent agent instance tracked across executions.

    Attributes:
        id: Unique identifier (auto-generated).
        profile: Static capability/persona description.
        history: Append-only log of past execution summaries for learning.
    """

    profile: AgentProfile
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.profile.name
