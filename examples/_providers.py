"""Shared provider selection for the examples.

Each example runs offline out of the box using DemoProvider, which
returns canned responses. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or
GOOGLE_API_KEY to run the same example against a real model.
"""

from __future__ import annotations

import json
import os
import sys

# ExecutionGraph.__str__ renders with box-drawing characters; legacy
# Windows consoles default to cp1252 and crash on them.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from smythe.prompts import PLANNING_SYSTEM_PROMPT
from smythe.provider import (
    AnthropicProvider,
    CompletionResult,
    GeminiProvider,
    OpenAIProvider,
    Provider,
)

DEMO_PLAN = json.dumps({
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
})


class DemoProvider(Provider):
    """Offline stand-in: canned plan JSON for planning calls, canned text otherwise."""

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        if system == PLANNING_SYSTEM_PROMPT:
            return CompletionResult(
                text=DEMO_PLAN, prompt_tokens=250, completion_tokens=400,
            )
        first_line = prompt.split("\n")[0]
        return CompletionResult(
            text=f"[demo output for: {first_line[:60]}]",
            prompt_tokens=40,
            completion_tokens=60,
        )


def pick_provider() -> tuple[Provider, str]:
    """Return (provider, model) — a real provider if an API key is set, else the demo."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(), "claude-mythos"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(), "gpt-5.2"
    if os.environ.get("GOOGLE_API_KEY"):
        return GeminiProvider(), "gemini-3-flash"
    print("No API key found - running offline with the built-in DemoProvider.")
    print("Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY for real output.\n")
    return DemoProvider(), "demo-model"
