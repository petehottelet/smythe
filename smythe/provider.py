"""Provider abstraction — async LLM backends for node execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract base for LLM providers.

    All providers are async-native so the parallel executor can
    fan out calls concurrently.  The serial executor wraps calls
    with asyncio.run().
    """

    @abstractmethod
    async def complete(self, system: str, prompt: str, model: str) -> str:
        """Send a prompt to the LLM and return the text response."""


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

    async def complete(self, system: str, prompt: str, model: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


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

    async def complete(self, system: str, prompt: str, model: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
