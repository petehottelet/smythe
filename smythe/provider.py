"""Provider abstraction — async LLM backends for node execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    """Response from a provider call, including token usage for budget tracking."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(ABC):
    """Abstract base for LLM providers.

    All providers are async-native so the parallel executor can
    fan out calls concurrently.  The serial executor wraps calls
    with asyncio.run().
    """

    @abstractmethod
    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        """Send a prompt to the LLM and return text + token usage."""


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API."""

    def __init__(self, *, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "Install the anthropic extra: pip install smythe[anthropic]"
                ) from exc
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        """Send a prompt to the Anthropic API and return the result.

        Note: only the first text block is used; multi-block responses
        are not concatenated.
        """
        client = self._get_client()
        response = await client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b for b in response.content if hasattr(b, "text")]
        text = text_blocks[0].text if text_blocks else ""
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
        )


class OpenAIProvider(Provider):
    """Provider backed by the OpenAI Chat Completions API."""

    def __init__(self, *, api_key: str | None = None, max_tokens: int = 4096) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "Install the openai extra: pip install smythe[openai]"
                ) from exc
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def complete(self, system: str, prompt: str, model: str) -> CompletionResult:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        usage = response.usage
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )
