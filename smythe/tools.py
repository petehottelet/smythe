"""Neutral tool-calling types — the provider-agnostic contract.

Providers map these to their native wire formats (Anthropic tool_use
blocks, OpenAI function calling, Gemini function declarations).  Nothing
above the provider layer touches vendor-specific shapes.

Tool names are namespaced "<server>.<tool>" for humans and traces, but
some vendors reject "." in tool names, so the wire form replaces it
with "__" (see wire_name / display_name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Strictest naming rule across the three vendors, applied to the wire form.
_WIRE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def wire_name(name: str) -> str:
    """Convert a display name ("fs.read_file") to the wire form ("fs__read_file")."""
    return name.replace(".", "__")


def display_name(wire: str) -> str:
    """Convert a wire form back to the display name."""
    return wire.replace("__", ".")


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call.

    Attributes:
        name: Display name, optionally namespaced ("fs.read_file").
        description: What the tool does and when to call it.
        input_schema: JSON Schema (object type) for the arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def __post_init__(self) -> None:
        wire = wire_name(self.name)
        if not _WIRE_NAME_RE.match(wire):
            raise ValueError(
                f"Tool name {self.name!r} is invalid: wire form {wire!r} must "
                f"match {_WIRE_NAME_RE.pattern}"
            )


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool.

    Attributes:
        id: Call identifier, echoed back in the matching ToolResult.
            Provider-assigned where the vendor supplies one (Anthropic,
            OpenAI); synthesized for vendors that don't (Gemini).
        name: Display name of the tool.
        arguments: Parsed arguments dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing a ToolCall, fed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ChatMessage:
    """One turn in a tool-aware conversation.

    Assistant turns may carry tool_calls; user turns may carry
    tool_results (the responses to the preceding assistant turn's calls).
    """

    role: str  # "user" | "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


def content_to_text(parts: Any) -> str:
    """Flatten a tool-result content list to text. Never raises.

    Text parts are concatenated; non-text parts (images, embedded
    resources) become placeholder markers so a screenshot-returning
    tool cannot crash a node.  Duck-typed so it works on MCP SDK
    content objects without importing mcp.
    """
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts

    out: list[str] = []
    try:
        iterator = iter(parts)
    except TypeError:
        return str(parts)

    for part in iterator:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            out.append(text)
            continue
        ptype = getattr(part, "type", None) or type(part).__name__
        mime = getattr(part, "mimeType", None) or getattr(part, "mime_type", None)
        data = getattr(part, "data", None)
        detail = f": {mime}" if isinstance(mime, str) else ""
        if isinstance(data, (str, bytes)):
            size = len(data)
            detail += f", {size // 1024}KB" if size >= 1024 else f", {size}B"
        out.append(f"[{ptype}{detail} - not passed to the model]")
    return "\n".join(out)
