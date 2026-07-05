"""Shared provider selection for the examples.

Each example runs offline out of the box using smythe's built-in
OfflineProvider (via DemoProvider below, which adds a canned research
plan). Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY to run
the same example against a real model.
"""

from __future__ import annotations

import os
import sys

from smythe.provider import (
    AnthropicProvider,
    GeminiProvider,
    OfflineProvider,
    OpenAIProvider,
    Provider,
)

# ExecutionGraph.__str__ and the tracer render with characters that
# legacy cp1252 Windows consoles can't encode.
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8")

DEMO_PLAN = {
    "topology": ["fork_join"],
    "nodes": [
        {
            "id": "research-market",
            "label": "Research the market landscape for solar chargers",
            "depends_on": [],
            "agent": {
                "name": "MarketResearcher",
                "persona": "You are a rigorous market researcher.",
                "capabilities": ["research"],
            },
        },
        {
            "id": "research-competitors",
            "label": "Profile the top three competing products",
            "depends_on": [],
            "agent": {
                "name": "CompetitorAnalyst",
                "persona": "You analyze competing products objectively.",
                "capabilities": ["research"],
            },
        },
        {
            "id": "summarize",
            "label": "Merge both research tracks into a one-page brief",
            "depends_on": ["research-market", "research-competitors"],
            "agent": {
                "name": "Editor",
                "persona": "You write crisp executive summaries.",
                "capabilities": ["summarize"],
            },
        },
    ],
}


class DemoProvider(OfflineProvider):
    """OfflineProvider preloaded with the demo research plan."""

    def __init__(self) -> None:
        super().__init__(plan=DEMO_PLAN, echo_prefix="[demo output for: ")

    async def complete(self, system, prompt, model):
        result = await super().complete(system, prompt, model)
        if result.text.startswith(self._echo_prefix):
            result.text += "]"
        return result


def pick_provider() -> tuple[Provider, str]:
    """Return (provider, model) — a real provider if an API key is set, else the demo."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(), "claude-mythos"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        return GeminiProvider(), "gemini-3-flash"
    print("No API key found - running offline with smythe's built-in OfflineProvider.")
    print("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY for real output.\n")
    return DemoProvider(), "demo-model"
